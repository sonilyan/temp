#!/usr/bin/env python3
# Convert ONLY BF16 tensors in a GGUF file to F16.
# All other tensor types (including Q8_0) and GGUF metadata are preserved.
#
# Run with the Python environment containing ComfyUI's torch and gguf:
#   python convert_gguf_bf16_to_f16.py input.gguf output-f16.gguf
#
# Add --overwrite to replace an existing output file.

from __future__ import annotations

import argparse
import collections
import os
from pathlib import Path
import gc
import numpy as np
import torch
import gguf


def qtype_name(tensor) -> str:
    qtype = getattr(tensor, "tensor_type", None)
    return getattr(qtype, "name", str(qtype))


def copy_metadata(reader, writer) -> None:
    # GGUFWriter writes general.architecture from its constructor.
    # GGUF.* fields are virtual/internal fields and must not be copied.
    for field in reader.fields.values():
        if (
            field.name == gguf.Keys.General.ARCHITECTURE
            or field.name.startswith("GGUF.")
        ):
            continue

        value_type = field.types[0]
        sub_type = (
            field.types[-1]
            if value_type == gguf.GGUFValueType.ARRAY
            else None
        )
        value = field.contents()

        if value is not None:
            writer.add_key_value(
                field.name,
                value,
                value_type,
                sub_type=sub_type,
            )


def bf16_to_f16_array(source_array: np.ndarray, shape, endianess):
    """
    GGUFReader may expose BF16 payload as uint8 bytes rather than uint16.
    Reinterpret each pair of bytes as one BF16 value, then numerically
    convert to F16. `shape` is the logical GGUF tensor shape.
    """
    raw = np.asarray(source_array)
    if raw.dtype != np.uint8:
        if raw.dtype.itemsize != 2:
            raise RuntimeError(
                f"Unexpected BF16 storage dtype: {raw.dtype}"
            )
        bits = np.array(raw, copy=True, order="C").view(np.uint16).reshape(-1)
    else:
        if raw.nbytes % 2:
            raise RuntimeError(
                f"BF16 payload has an odd byte count: {raw.nbytes}"
            )

        # The GGUF payload uses the file's byte order.
        endian_name = str(endianess).upper()
        byte_order = ">" if "BIG" in endian_name else "<"
        bits = np.frombuffer(
            np.ascontiguousarray(raw).tobytes(),
            dtype=np.dtype(byte_order + "u2"),
        ).copy()

    expected_elements = int(np.prod(shape, dtype=np.int64))
    if bits.size != expected_elements:
        raise RuntimeError(
            "BF16 payload size does not match tensor shape: "
            f"{bits.size} values vs shape {tuple(shape)} "
            f"({expected_elements} values)"
        )

    # Torch expects native-endian uint16 storage for the BF16 bit pattern.
    if not bits.dtype.isnative:
        bits = bits.astype(np.uint16, copy=True)

    bits = np.ascontiguousarray(bits.reshape(tuple(int(x) for x in shape)))
    tensor = torch.from_numpy(bits).view(torch.bfloat16)
    converted = tensor.to(dtype=torch.float16).contiguous()

    # Count finite BF16 values that overflowed to +/-inf in F16.
    overflow_count = int(
        (torch.isfinite(tensor) & torch.isinf(converted)).sum().item()
    )
    return converted.cpu().numpy(), overflow_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert BF16 tensors in GGUF to F16 while preserving all "
            "other tensor types and metadata."
        )
    )
    parser.add_argument("input", type=Path, help="Source GGUF file")
    parser.add_argument("output", type=Path, help="New GGUF file")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file",
    )
    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    partial_path = output_path.with_name(output_path.name + ".partial")

    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    if input_path == output_path:
        parser.error("Input and output must be different files.")
    if output_path.exists() and not args.overwrite:
        parser.error(
            f"Output already exists: {output_path} "
            "(use --overwrite to replace it)"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if partial_path.exists():
        partial_path.unlink()

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print("Opening GGUF and scanning tensor types...")

    reader = gguf.GGUFReader(str(input_path), "r")
    bf16_type = gguf.GGMLQuantizationType.BF16

    before_counts = collections.Counter(qtype_name(t) for t in reader.tensors)
    bf16_tensors = [
        tensor for tensor in reader.tensors
        if tensor.tensor_type == bf16_type
    ]

    print("Source tensor types:")
    for name, count in sorted(before_counts.items()):
        print(f"  {name:10s} {count:6d}")
    print(f"BF16 tensors to convert: {len(bf16_tensors)}")

    if not bf16_tensors:
        print("No BF16 tensors found; nothing to convert.")
        return 0

    arch_field = reader.get_field(gguf.Keys.General.ARCHITECTURE)
    if arch_field is None:
        raise RuntimeError("GGUF is missing general.architecture.")
    arch = arch_field.contents()

    writer = None
    try:
        writer = gguf.GGUFWriter(
            str(partial_path),
            arch=arch,
            endianess=reader.endianess,
        )

        alignment_field = reader.get_field(gguf.Keys.General.ALIGNMENT)
        if alignment_field is not None:
            writer.data_alignment = int(alignment_field.contents())

        copy_metadata(reader, writer)

        # Register logical tensor shapes, not the shape of the raw byte view.
        # No tensor payloads are accumulated in memory here.
        for tensor in reader.tensors:
            logical_shape = tuple(int(x) for x in tensor.shape)
            if tensor.tensor_type == bf16_type:
                writer.add_tensor_info(
                    tensor.name,
                    logical_shape[::-1],
                    np.dtype(np.float16),
                    int(np.prod(logical_shape, dtype=np.int64)) * 2,
                )
            else:
                writer.add_tensor_info(
                    tensor.name,
                    tensor.data.shape,
                    tensor.data.dtype,
                    tensor.data.nbytes,
                    raw_dtype=tensor.tensor_type,
                )

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()

        converted_count = 0
        overflow_count = 0

        for index, tensor in enumerate(reader.tensors, start=1):
            if tensor.tensor_type == bf16_type:
                data, overflows = bf16_to_f16_array(
                    tensor.data,
                    tensor.shape,
                    reader.endianess,
                )
                writer.write_tensor_data(
                    data,
                    tensor_endianess=reader.endianess,
                )
                converted_count += 1
                overflow_count += overflows
                del data
            else:
                # Copy original packed bytes and preserve the original qtype.
                writer.write_tensor_data(
                    tensor.data,
                    tensor_endianess=reader.endianess,
                )

            if index % 25 == 0 or index == len(reader.tensors):
                print(
                    f"\rWriting tensors: {index}/{len(reader.tensors)}",
                    end="",
                    flush=True,
                )

        writer.close()
        writer = None
        print()

        # Validate before publishing the completed output file.
        print("Verifying output GGUF...")
        check = gguf.GGUFReader(str(partial_path), "r")
        after_counts = collections.Counter(qtype_name(t) for t in check.tensors)

        if len(check.tensors) != len(reader.tensors):
            raise RuntimeError("Validation failed: tensor count changed.")
        if after_counts.get("BF16", 0) != 0:
            raise RuntimeError("Validation failed: BF16 tensors remain.")

        expected_counts = before_counts.copy()
        expected_counts.pop("BF16", None)
        expected_counts["F16"] += len(bf16_tensors)

        if after_counts != expected_counts:
            raise RuntimeError(
                "Validation failed: unexpected output qtype counts.\n"
                f"Expected: {dict(expected_counts)}\n"
                f"Actual:   {dict(after_counts)}"
            )

        original_info = {
            t.name: tuple(int(x) for x in t.shape)
            for t in reader.tensors
        }
        output_info = {
            t.name: tuple(int(x) for x in t.shape)
            for t in check.tensors
        }
        if original_info != output_info:
            missing = set(original_info) - set(output_info)
            extra = set(output_info) - set(original_info)

            print(f"Missing tensor names: {len(missing)}")
            print(f"Extra tensor names: {len(extra)}")

            for name in list(missing)[:10]:
                print(f"MISSING: {name}")

            for name in list(extra)[:10]:
                print(f"EXTRA: {name}")

            shape_mismatches = []
            for name in set(original_info) & set(output_info):
                if original_info[name] != output_info[name]:
                    shape_mismatches.append(
                        (name, original_info[name], output_info[name])
                    )

            print(f"Shape mismatches: {len(shape_mismatches)}")

            for name, old_shape, new_shape in shape_mismatches[:20]:
                print(f"SHAPE: {name}")
                print(f"  Original: {old_shape}")
                print(f"  Output:   {new_shape}")

            raise RuntimeError("Validation failed: see differences above.")
        del check
        gc.collect()
        os.replace(partial_path, output_path)

        input_size = input_path.stat().st_size
        output_size = output_path.stat().st_size

        print("\nConversion successful.")
        print(f"Converted BF16 -> F16: {converted_count}")
        print(f"Finite values overflowing to +/-inf: {overflow_count}")
        print("Output tensor types:")
        for name, count in sorted(after_counts.items()):
            print(f"  {name:10s} {count:6d}")
        print(f"Input size : {input_size / (1024**3):.3f} GiB")
        print(f"Output size: {output_size / (1024**3):.3f} GiB")
        print(f"Saved file : {output_path}")

        if overflow_count:
            print(
                "WARNING: Some finite BF16 values overflowed F16. "
                "Do not use this output until those tensors are investigated."
            )
        else:
            print("No finite-value overflow was detected during conversion.")

        return 0

    except Exception:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if partial_path.exists():
            try:
                partial_path.unlink()
            except Exception:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
