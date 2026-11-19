# my_unet_gguf_loader.py
#
# Based on leejet/ComfyUI-GGUF (Apache-2.0).
# Standalone Qwen-Image 2.1 compatible GGUF UNet loader.
#
# Features:
# - Qwen-Image 2.1 architecture support
# - Qwen fused Linear activation: input_act="swiglu"
# - Linear residual / residual_scale support
# - Optional GPU Q8_0 row-chunk dequantization
# - Standard ComfyUI MODEL output and ModelPatcher integration
# - Process-level model cache with GGUF Reader retention
#
# Install:
#   ComfyUI/custom_nodes/my_unet_gguf_loader.py
#
# Restart ComfyUI after installation.

import os
import logging
import warnings
import collections
import inspect
import threading

import torch
import torch.nn.functional as F
import gguf

import comfy.ops
import comfy.sd
import comfy.lora
import comfy.float
import comfy.utils
import comfy.model_patcher
import comfy.model_management
import folder_paths


_LOG_PREFIX = "[MyGGUFStream]"


# ------------------------------------------------------------
# Optional persistent VRAM residency
# ------------------------------------------------------------

_explicit_unload_state = threading.local()


def _is_my_gguf_patcher(patcher):
    return bool(
        patcher is not None
        and getattr(patcher, "_my_gguf_keep_loaded", False)
    )


def _install_persistent_vram_hooks():
    mm = comfy.model_management
    loaded_model_cls = getattr(mm, "LoadedModel", None)

    if loaded_model_cls is None:
        logging.warning(
            "%s LoadedModel class unavailable; persistent residency is disabled",
            _LOG_PREFIX,
        )
        return

    # 1) Establish the strong reference only after a successful model load.
    original_model_load = loaded_model_cls.model_load

    if not getattr(original_model_load, "_my_gguf_keepalive_hook", False):

        def model_load_with_keepalive(self, *args, **kwargs):
            result = original_model_load(self, *args, **kwargs)
            patcher = self.model

            if _is_my_gguf_patcher(patcher):
                if getattr(self, "_my_gguf_keepalive", None) is not patcher:
                    self._my_gguf_keepalive = patcher
            else:
                if hasattr(self, "_my_gguf_keepalive"):
                    del self._my_gguf_keepalive

            return result

        model_load_with_keepalive._my_gguf_keepalive_hook = True
        model_load_with_keepalive._my_gguf_original = original_model_load
        loaded_model_cls.model_load = model_load_with_keepalive

    # 2) Release the strong reference after a complete ComfyUI unload.
    original_model_unload = loaded_model_cls.model_unload

    if not getattr(original_model_unload, "_my_gguf_keepalive_hook", False):

        def model_unload_with_keepalive(self, *args, **kwargs):
            result = original_model_unload(self, *args, **kwargs)

            if result is True and hasattr(self, "_my_gguf_keepalive"):
                del self._my_gguf_keepalive

            return result

        model_unload_with_keepalive._my_gguf_keepalive_hook = True
        model_unload_with_keepalive._my_gguf_original = original_model_unload
        loaded_model_cls.model_unload = model_unload_with_keepalive

    # 3) Normal memory pressure must not evict a keepalive model.
    #    Explicit user-requested unload operations temporarily bypass this.
    original_free_memory = mm.free_memory

    if not getattr(original_free_memory, "_my_gguf_keepalive_hook", False):
        signature = inspect.signature(original_free_memory)

        def free_memory_with_keepalive(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            keep_loaded = list(
                bound.arguments.get("keep_loaded") or ()
            )

            if not getattr(_explicit_unload_state, "active", False):
                for loaded_model in tuple(mm.current_loaded_models):
                    try:
                        patcher = loaded_model.model

                        if not _is_my_gguf_patcher(patcher):
                            continue

                        if loaded_model.is_dead():
                            continue

                        if (
                            getattr(
                                loaded_model,
                                "_my_gguf_keepalive",
                                None,
                            )
                            is not patcher
                        ):
                            loaded_model._my_gguf_keepalive = patcher

                        if not any(
                            item is loaded_model
                            for item in keep_loaded
                        ):
                            keep_loaded.append(loaded_model)

                    except Exception:
                        continue

            bound.arguments["keep_loaded"] = keep_loaded

            return original_free_memory(
                *bound.args,
                **bound.kwargs,
            )

        free_memory_with_keepalive._my_gguf_keepalive_hook = True
        free_memory_with_keepalive._my_gguf_original = original_free_memory
        mm.free_memory = free_memory_with_keepalive

    # 4) Explicit unload APIs are allowed to unload keepalive models.
    for function_name in (
        "unload_all_models",
        "unload_model_and_clones",
    ):
        original = getattr(mm, function_name, None)

        if (
            original is None
            or getattr(
                original,
                "_my_gguf_explicit_unload_hook",
                False,
            )
        ):
            continue

        def make_explicit_unload_wrapper(func, name):

            def explicit_unload_wrapper(*args, **kwargs):
                previous = getattr(
                    _explicit_unload_state,
                    "active",
                    False,
                )

                _explicit_unload_state.active = True

                try:
                    return func(*args, **kwargs)
                finally:
                    _explicit_unload_state.active = previous

            explicit_unload_wrapper._my_gguf_explicit_unload_hook = True
            explicit_unload_wrapper._my_gguf_original = func
            explicit_unload_wrapper.__name__ = getattr(
                func,
                "__name__",
                name,
            )

            return explicit_unload_wrapper

        setattr(
            mm,
            function_name,
            make_explicit_unload_wrapper(
                original,
                function_name,
            ),
        )


_install_persistent_vram_hooks()


# ------------------------------------------------------------
# GGUF tensor helpers
# ------------------------------------------------------------

TORCH_COMPATIBLE_QTYPES = (
    None,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
)


def is_torch_compatible(tensor):
    return (
        tensor is None
        or getattr(
            tensor,
            "tensor_type",
            None,
        ) in TORCH_COMPATIBLE_QTYPES
    )


def is_quantized(tensor):
    return not is_torch_compatible(tensor)


class GGMLTensor(torch.Tensor):

    def __new__(
        cls,
        *args,
        tensor_type,
        tensor_shape,
        patches=None,
        **kwargs
    ):
        return super().__new__(
            cls,
            *args,
            **kwargs
        )

    def __init__(
        self,
        *args,
        tensor_type,
        tensor_shape,
        patches=None,
        **kwargs
    ):
        super().__init__()

        self.tensor_type = tensor_type
        self.tensor_shape = torch.Size(tensor_shape)
        self.patches = [] if patches is None else patches

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)

        new.tensor_type = getattr(
            self,
            "tensor_type",
            None,
        )

        new.tensor_shape = getattr(
            self,
            "tensor_shape",
            new.data.shape,
        )

        new.patches = getattr(
            self,
            "patches",
            [],
        ).copy()

        if hasattr(self, "is_largest_weight"):
            new.is_largest_weight = self.is_largest_weight

        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def copy_(self, *args, **kwargs):
        try:
            return super().copy_(*args, **kwargs)
        except Exception as e:
            logging.warning(
                "%s ignoring copy_: %s",
                _LOG_PREFIX,
                e,
            )

    def new_empty(self, size, *args, **kwargs):
        new_tensor = super().new_empty(
            size,
            *args,
            **kwargs,
        )

        return GGMLTensor(
            new_tensor,
            tensor_type=getattr(
                self,
                "tensor_type",
                None,
            ),
            tensor_shape=size,
            patches=getattr(
                self,
                "patches",
                [],
            ).copy(),
        )

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()

        return self.tensor_shape


# ------------------------------------------------------------
# Dequantization
# ------------------------------------------------------------

def _q8_0_dequantize_raw(raw, output_shape, dtype):
    """
    Q8_0:
      block size = 32 values
      block size in bytes = 34
      first 2 bytes = FP16 scale
      next 32 bytes = INT8 values
    """
    if not raw.is_contiguous():
        raw = raw.contiguous()

    raw = raw.view(torch.uint8).reshape(-1, 34)

    scale_bytes = raw[:, :2].contiguous()
    value_bytes = raw[:, 2:].contiguous()

    scales = scale_bytes.view(torch.float16)
    values = value_bytes.view(torch.int8)

    if dtype is not None:
        scales = scales.to(dtype)

    result = scales * values

    return result.reshape(output_shape)


def dequantize_tensor(
    tensor,
    dtype=None,
    dequant_dtype=None,
):
    qtype = getattr(
        tensor,
        "tensor_type",
        None,
    )

    output_shape = getattr(
        tensor,
        "tensor_shape",
        tensor.shape,
    )

    if qtype in TORCH_COMPATIBLE_QTYPES:
        return tensor.to(dtype)

    target_dtype = (
        dtype
        if dequant_dtype == "target"
        else dequant_dtype
    )

    if qtype == gguf.GGMLQuantizationType.Q8_0:
        result = _q8_0_dequantize_raw(
            tensor.data,
            output_shape,
            target_dtype,
        )

        return (
            result.to(dtype)
            if dtype is not None
            else result
        )

    # Compatibility fallback for other GGUF quantization types.
    logging.warning(
        "%s qtype %s uses GGUF CPU dequant fallback",
        _LOG_PREFIX,
        getattr(
            qtype,
            "name",
            repr(qtype),
        ),
    )

    cpu_array = tensor.detach().cpu().numpy()

    result = gguf.quants.dequantize(
        cpu_array,
        qtype,
    )

    result = torch.from_numpy(result).to(tensor.device)

    if target_dtype is not None:
        result = result.to(target_dtype)

    if dtype is not None:
        result = result.to(dtype)

    return result


# ------------------------------------------------------------
# GGUF metadata / loader
# ------------------------------------------------------------

IMG_ARCH_LIST = {
    "flux",
    "sd1",
    "sdxl",
    "sd3",
    "aura",
    "hidream",
    "cosmos",
    "ltxv",
    "hyvid",
    "wan",
    "lumina2",
    "qwen_image",
    "qwen_image21",
    "ideogram4",
    "krea2",
    "minimax",
}

TXT_ARCH_LIST = {
    "t5",
    "t5encoder",
    "llama",
    "qwen2vl",
    "qwen3",
    "qwen3vl",
    "qwen35",
    "gemma3",
}

VIS_TYPE_LIST = {
    "clip-vision",
    "mmproj",
}


def get_orig_shape(reader, tensor_name):
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"

    field = reader.get_field(field_key)

    if field is None:
        return None

    if (
        len(field.types) != 2
        or field.types[0] != gguf.GGUFValueType.ARRAY
        or field.types[1] != gguf.GGUFValueType.INT32
    ):
        raise TypeError(
            f"Bad original shape metadata for {field_key}: {field.types}"
        )

    return torch.Size(
        tuple(
            int(field.parts[i][0])
            for i in field.data
        )
    )


def get_field(reader, field_name, field_type):
    field = reader.get_field(field_name)

    if field is None:
        return None

    if field_type == str:
        if (
            len(field.types) != 1
            or field.types[0] != gguf.GGUFValueType.STRING
        ):
            raise TypeError(
                f"Bad type for GGUF {field_name}: {field.types!r}"
            )

        return str(
            field.parts[field.data[-1]],
            encoding="utf-8",
        )

    if field_type in (int, float, bool):
        return field_type(
            field.parts[field.data[-1]].item()
        )

    raise TypeError(
        f"Unknown field type {field_type}"
    )


def get_gguf_metadata(reader):
    metadata = {}

    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)

            if len(field.types) != 1:
                continue

            field_type = field.types[0]

            if field_type == gguf.GGUFValueType.STRING:
                metadata[field_name] = str(
                    field.parts[field.data[-1]],
                    "utf-8",
                )

            elif field_type == gguf.GGUFValueType.INT32:
                metadata[field_name] = int(
                    field.parts[field.data[-1]]
                )

            elif field_type == gguf.GGUFValueType.F32:
                metadata[field_name] = float(
                    field.parts[field.data[-1]]
                )

            elif field_type == gguf.GGUFValueType.BOOL:
                metadata[field_name] = bool(
                    field.parts[field.data[-1]]
                )

        except Exception:
            continue

    return metadata


def gguf_sd_loader(
    path,
    handle_prefix="model.diffusion_model.",
    return_reader=False,
):
    reader = gguf.GGUFReader(path)

    has_prefix = False

    if handle_prefix is not None:
        prefix_len = len(handle_prefix)

        tensor_names = {
            tensor.name
            for tensor in reader.tensors
        }

        has_prefix = any(
            name.startswith(handle_prefix)
            for name in tensor_names
        )

    tensors = []

    for tensor in reader.tensors:
        tensor_name = tensor.name
        sd_key = tensor_name

        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue

            sd_key = tensor_name[prefix_len:]

        tensors.append(
            (sd_key, tensor)
        )

    arch_str = get_field(
        reader,
        "general.architecture",
        str,
    )

    type_str = get_field(
        reader,
        "general.type",
        str,
    )

    if arch_str in (None, "pig", "cow"):
        raise ValueError(
            "GGUF architecture metadata is missing or incompatible. "
            "This standalone loader requires a GGUF with a recognized "
            "architecture field."
        )

    if arch_str not in IMG_ARCH_LIST:
        raise ValueError(
            f"Unsupported GGUF architecture: {arch_str!r}"
        )

    state_dict = {}

    for sd_key, tensor in tensors:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given NumPy array is not writable",
            )

            torch_tensor = torch.from_numpy(tensor.data)

        shape = get_orig_shape(
            reader,
            tensor.name,
        )

        if shape is None:
            shape = torch.Size(
                tuple(
                    int(v)
                    for v in reversed(tensor.shape)
                )
            )

        if tensor.tensor_type in (
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
        ):
            torch_tensor = torch_tensor.view(*shape)

        wrapped = GGMLTensor(
            torch_tensor,
            tensor_type=tensor.tensor_type,
            tensor_shape=shape,
        )

        state_dict[sd_key] = wrapped

        # Quantized 1D tensors are dequantized before model construction.
        if len(shape) <= 1 and is_quantized(wrapped):
            state_dict[sd_key] = dequantize_tensor(
                wrapped,
                dtype=torch.float32,
            )

    quantized = {
        key: value
        for key, value in state_dict.items()
        if is_quantized(value)
    }

    if quantized:
        largest_key = max(
            quantized,
            key=lambda key: quantized[key].numel(),
        )

        state_dict[largest_key].is_largest_weight = True

    extra = {
        "arch_str": arch_str,
        "metadata": get_gguf_metadata(reader),
    }

    if return_reader:
        return state_dict, extra, reader

    return state_dict, extra


# ------------------------------------------------------------
# Patch helpers
# ------------------------------------------------------------

def move_patch_to_device(item, device):
    if isinstance(item, torch.Tensor):
        return item.to(
            device,
            non_blocking=True,
        )

    if isinstance(item, tuple):
        return tuple(
            move_patch_to_device(x, device)
            for x in item
        )

    if isinstance(item, list):
        return [
            move_patch_to_device(x, device)
            for x in item
        ]

    return item


def chained_hasattr(obj, chained_attr):
    probe = obj

    for attr in chained_attr.split("."):
        if not hasattr(probe, attr):
            return False

        probe = getattr(
            probe,
            attr,
        )

    return True


def get_torch_compiler_disable_decorator():

    def dummy_decorator(*args, **kwargs):

        def noop(fn):
            return fn

        return noop

    try:
        from packaging import version

        if not chained_hasattr(
            torch,
            "compiler.disable",
        ):
            return dummy_decorator

        if version.parse(
            torch.__version__
        ) >= version.parse("2.8"):
            return dummy_decorator

        if chained_hasattr(
            torch,
            "_dynamo.config.nontraceable_tensor_subclasses",
        ):
            return dummy_decorator

        return torch.compiler.disable

    except Exception:
        return dummy_decorator


torch_compiler_disable = get_torch_compiler_disable_decorator()


# ------------------------------------------------------------
# Qwen fused activation / residual compatibility
# ------------------------------------------------------------

def apply_input_activation(
    input_tensor,
    input_act=None,
    act_weight=None,
    act_eps=0.0,
):
    if input_act is None:
        return input_tensor

    helper = getattr(
        comfy.ops,
        "_eager_input_act",
        None,
    )

    if callable(helper):
        return helper(
            input_tensor,
            input_act,
            act_weight,
            act_eps,
        )

    if str(input_act).lower() == "swiglu":
        if input_tensor.shape[-1] % 2 != 0:
            raise RuntimeError(
                "SwiGLU input dimension must be divisible by 2"
            )

        gate, value = input_tensor.chunk(
            2,
            dim=-1,
        )

        return F.silu(gate) * value

    raise NotImplementedError(
        f"Unsupported fused Linear activation: {input_act!r}"
    )


def apply_linear_residual(
    output,
    residual=None,
    residual_scale=None,
):
    if residual is None:
        return output

    helper = getattr(
        comfy.ops,
        "_linear_residual",
        None,
    )

    if callable(helper):
        return helper(
            output,
            residual,
            residual_scale,
        )

    if residual_scale is None:
        return output + residual

    return torch.addcmul(
        residual,
        output,
        residual_scale,
    )


# ------------------------------------------------------------
# GPU Q8_0 row chunking
# ------------------------------------------------------------

def get_q8_0_raw_rows(weight):
    """
    Validate that raw Q8_0 storage can be interpreted as:
        [out_features, in_features / 32 * 34]

    Returns:
        raw_rows, in_features, out_features
    or:
        None
    """
    try:
        if getattr(
            weight,
            "tensor_type",
            None,
        ) != gguf.GGMLQuantizationType.Q8_0:
            return None

        logical_shape = tuple(
            int(v)
            for v in getattr(
                weight,
                "tensor_shape",
                weight.shape,
            )
        )

        if len(logical_shape) != 2:
            return None

        out_features, in_features = logical_shape

        if in_features <= 0 or out_features <= 0:
            return None

        if in_features % 32 != 0:
            return None

        raw = weight.data

        if not raw.is_contiguous():
            return None

        row_bytes = (
            in_features // 32
        ) * 34

        raw_u8 = raw.view(torch.uint8)

        expected_bytes = (
            out_features * row_bytes
        )

        if raw_u8.numel() != expected_bytes:
            logging.warning(
                "%s Q8_0 layout mismatch: logical=%s raw_numel=%s expected=%s",
                _LOG_PREFIX,
                logical_shape,
                raw_u8.numel(),
                expected_bytes,
            )
            return None

        raw_rows = raw_u8.reshape(
            out_features,
            row_bytes,
        )

        return (
            raw_rows,
            in_features,
            out_features,
        )

    except Exception as e:
        logging.warning(
            "%s cannot validate Q8_0 raw layout: %s",
            _LOG_PREFIX,
            e,
        )
        return None


def dequantize_q8_0_row_chunk(
    raw_rows,
    in_features,
    dtype,
):
    """
    Dequantize a group of complete Q8_0 rows on the current device.
    """
    row_count = raw_rows.shape[0]

    blocks_per_row = in_features // 32

    blocks = raw_rows.reshape(
        row_count * blocks_per_row,
        34,
    )

    scales = blocks[:, :2].contiguous().view(torch.float16)
    values = blocks[:, 2:].contiguous().view(torch.int8)

    scales = scales.to(dtype)

    result = scales * values

    return result.reshape(
        row_count,
        in_features,
    )


# ------------------------------------------------------------
# GGML layer implementation
# ------------------------------------------------------------

class GGMLLayer(torch.nn.Module):
    comfy_cast_weights = True

    dequant_dtype = None
    patch_dtype = None
    gpu_dequant_chunk_rows = 0

    def is_ggml_quantized(
        self,
        *,
        weight=None,
        bias=None,
    ):
        if weight is None:
            weight = self.weight

        if bias is None:
            bias = self.bias

        return (
            is_quantized(weight)
            or is_quantized(bias)
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        *args,
        **kwargs,
    ):
        weight = state_dict.get(
            f"{prefix}weight"
        )

        bias = state_dict.get(
            f"{prefix}bias"
        )

        if (
            self.is_ggml_quantized(
                weight=weight,
                bias=bias,
            )
            or isinstance(
                self,
                torch.nn.Linear,
            )
        ):
            return self.ggml_load_from_state_dict(
                state_dict,
                prefix,
                *args,
                **kwargs,
            )

        if (
            isinstance(
                self,
                torch.nn.Embedding,
            )
            and self.weight.shape[0] >= 64 * 1024
        ):
            return self.ggml_load_from_state_dict(
                state_dict,
                prefix,
                *args,
                **kwargs,
            )

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            *args,
            **kwargs,
        )

    def ggml_load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        prefix_len = len(prefix)

        for key, value in state_dict.items():
            if key[prefix_len:] == "weight":
                self.weight = torch.nn.Parameter(
                    value,
                    requires_grad=False,
                )

            elif (
                key[prefix_len:] == "bias"
                and value is not None
            ):
                self.bias = torch.nn.Parameter(
                    value,
                    requires_grad=False,
                )

            else:
                if key.startswith(prefix):
                    unexpected_keys.append(key)

        if (
            getattr(
                self,
                "weight",
                None,
            ) is None
            and isinstance(
                self,
                torch.nn.Linear,
            )
        ):
            value = torch.zeros(
                self.in_features,
                self.out_features,
            )

            self.weight = torch.nn.Parameter(
                value,
                requires_grad=False,
            )

            missing_keys.append(
                prefix + "weight"
            )

        if getattr(
            getattr(
                self,
                "weight",
                None,
            ),
            "is_largest_weight",
            False,
        ):
            self.largest_layer = True

    def _save_to_state_dict(
        self,
        *args,
        **kwargs,
    ):
        if self.is_ggml_quantized():
            return self.ggml_save_to_state_dict(
                *args,
                **kwargs,
            )

        return super()._save_to_state_dict(
            *args,
            **kwargs,
        )

    def ggml_save_to_state_dict(
        self,
        destination,
        prefix,
        keep_vars,
    ):
        weight = torch.zeros_like(
            self.weight,
            device=torch.device("meta"),
        )

        destination[prefix + "weight"] = weight

        if self.bias is not None:
            bias = torch.zeros_like(
                self.bias,
                device=torch.device("meta"),
            )

            destination[prefix + "bias"] = bias

        if getattr(
            self,
            "largest_layer",
            False,
        ):
            shape = getattr(
                self.weight,
                "tensor_shape",
                self.weight.shape,
            )

            dtype = (
                self.dequant_dtype
                if self.dequant_dtype
                and self.dequant_dtype != "target"
                else torch.float16
            )

            temp = torch.empty(
                *shape,
                device=torch.device("meta"),
                dtype=dtype,
            )

            destination[prefix + "temp.weight"] = temp

        return

    def get_weight(
        self,
        tensor,
        dtype,
    ):
        if tensor is None:
            return None

        patch_list = []
        device = tensor.device
        patch_key = None

        for patches, key in getattr(
            tensor,
            "patches",
            [],
        ):
            patch_list += move_patch_to_device(
                patches,
                device,
            )

            patch_key = key

        weight = dequantize_tensor(
            tensor,
            dtype,
            self.dequant_dtype,
        )

        if isinstance(
            weight,
            GGMLTensor,
        ):
            weight = torch.Tensor(weight)

        if patch_list:
            if self.patch_dtype is None:
                weight = comfy.lora.calculate_weight(
                    patch_list,
                    weight,
                    patch_key,
                )

            else:
                patch_dtype = (
                    dtype
                    if self.patch_dtype == "target"
                    else self.patch_dtype
                )

                weight = comfy.lora.calculate_weight(
                    patch_list,
                    weight,
                    patch_key,
                    patch_dtype,
                )

        return weight

    @torch_compiler_disable()
    def cast_bias_weight(
        s,
        input=None,
        dtype=None,
        device=None,
        bias_dtype=None,
    ):
        if input is not None:
            if dtype is None:
                dtype = getattr(
                    input,
                    "dtype",
                    torch.float32,
                )

            if bias_dtype is None:
                bias_dtype = dtype

            if device is None:
                device = input.device

        bias = None

        non_blocking = (
            comfy.model_management.device_supports_non_blocking(
                device
            )
        )

        if s.bias is not None:
            bias = s.get_weight(
                s.bias.to(device),
                dtype,
            )

            bias = comfy.ops.cast_to(
                bias,
                bias_dtype,
                device,
                non_blocking=non_blocking,
                copy=False,
            )

        weight = s.get_weight(
            s.weight.to(device),
            dtype,
        )

        weight = comfy.ops.cast_to(
            weight,
            dtype,
            device,
            non_blocking=non_blocking,
            copy=False,
        )

        return weight, bias

    def forward_comfy_cast_weights(
        self,
        input,
        *args,
        **kwargs,
    ):
        if self.is_ggml_quantized():
            output = self.forward_ggml_cast_weights(
                input,
                *args,
                **kwargs,
            )

        else:
            output = super().forward_comfy_cast_weights(
                input,
                *args,
                **kwargs,
            )

        if isinstance(
            output,
            GGMLTensor,
        ):
            output = torch.Tensor(output)

        return output

    def forward_ggml_cast_weights(
        self,
        input,
        *args,
        **kwargs,
    ):
        raise NotImplementedError
# ------------------------------------------------------------
# ComfyUI operations
# ------------------------------------------------------------

class GGMLOps(comfy.ops.manual_cast):

    class Linear(
        GGMLLayer,
        comfy.ops.manual_cast.Linear,
    ):
        gpu_dequant_chunk_rows = 0

        def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
        ):
            torch.nn.Module.__init__(self)

            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

            self._chunk_fallback_logged = False

        def _full_linear(
            self,
            input,
            input_act=None,
            act_weight=None,
            act_eps=0.0,
            residual=None,
            residual_scale=None,
        ):
            input = apply_input_activation(
                input,
                input_act,
                act_weight,
                act_eps,
            )

            weight, bias = self.cast_bias_weight(input)

            output = F.linear(
                input,
                weight,
                bias,
            )

            return apply_linear_residual(
                output,
                residual,
                residual_scale,
            )

        def _chunked_q8_linear(
            self,
            input,
            input_act=None,
            act_weight=None,
            act_eps=0.0,
            residual=None,
            residual_scale=None,
        ):
            chunk_rows = int(
                getattr(
                    self,
                    "gpu_dequant_chunk_rows",
                    0,
                )
            )

            if chunk_rows <= 0:
                return None

            weight_parameter = self.weight

            if not isinstance(
                weight_parameter,
                GGMLTensor,
            ):
                return None

            if getattr(
                weight_parameter,
                "tensor_type",
                None,
            ) != gguf.GGMLQuantizationType.Q8_0:
                return None

            # LoRA patches require the full patched weight path.
            if getattr(
                weight_parameter,
                "patches",
                [],
            ):
                if not self._chunk_fallback_logged:
                    logging.warning(
                        "%s Linear has LoRA patches; "
                        "using full dequantization for this layer",
                        _LOG_PREFIX,
                    )

                    self._chunk_fallback_logged = True

                return None

            input = apply_input_activation(
                input,
                input_act,
                act_weight,
                act_eps,
            )

            if input.shape[-1] != self.in_features:
                raise RuntimeError(
                    f"Linear input mismatch: "
                    f"{input.shape[-1]} != {self.in_features}"
                )

            device = input.device
            dtype = input.dtype

            moved_weight = weight_parameter.to(device)

            validated = get_q8_0_raw_rows(
                moved_weight
            )

            if validated is None:
                if not self._chunk_fallback_logged:
                    logging.warning(
                        "%s Q8_0 raw layout not suitable for "
                        "row chunking; using full dequantization",
                        _LOG_PREFIX,
                    )

                    self._chunk_fallback_logged = True

                return None

            raw_rows, in_features, out_features = validated

            if (
                in_features != self.in_features
                or out_features != self.out_features
            ):
                if not self._chunk_fallback_logged:
                    logging.warning(
                        "%s Q8_0 logical shape mismatch; "
                        "using full dequantization",
                        _LOG_PREFIX,
                    )

                    self._chunk_fallback_logged = True

                return None

            bias = None

            if self.bias is not None:
                bias = self.get_weight(
                    self.bias.to(device),
                    dtype,
                )

                bias = comfy.ops.cast_to(
                    bias,
                    dtype,
                    device,
                    non_blocking=(
                        comfy.model_management.device_supports_non_blocking(
                            device
                        )
                    ),
                    copy=False,
                )

            output_shape = (
                *input.shape[:-1],
                out_features,
            )

            output = input.new_empty(output_shape)

            for start in range(
                0,
                out_features,
                chunk_rows,
            ):
                end = min(
                    start + chunk_rows,
                    out_features,
                )

                raw_chunk = raw_rows[start:end]

                weight_chunk = dequantize_q8_0_row_chunk(
                    raw_chunk,
                    in_features,
                    dtype,
                )

                bias_chunk = (
                    bias[start:end]
                    if bias is not None
                    else None
                )

                result_chunk = F.linear(
                    input,
                    weight_chunk,
                    bias_chunk,
                )

                output[..., start:end].copy_(
                    result_chunk
                )

                del weight_chunk
                del result_chunk

            return apply_linear_residual(
                output,
                residual,
                residual_scale,
            )

        def forward_ggml_cast_weights(
            self,
            input,
            input_act=None,
            act_weight=None,
            act_eps=0.0,
            residual=None,
            residual_scale=None,
            *args,
            **kwargs,
        ):
            chunked = self._chunked_q8_linear(
                input,
                input_act=input_act,
                act_weight=act_weight,
                act_eps=act_eps,
                residual=residual,
                residual_scale=residual_scale,
            )

            if chunked is not None:
                return chunked

            return self._full_linear(
                input,
                input_act=input_act,
                act_weight=act_weight,
                act_eps=act_eps,
                residual=residual,
                residual_scale=residual_scale,
            )

    class Conv2d(
        GGMLLayer,
        comfy.ops.manual_cast.Conv2d,
    ):
        def forward_ggml_cast_weights(
            self,
            input,
            *args,
            **kwargs,
        ):
            weight, bias = self.cast_bias_weight(input)

            return self._conv_forward(
                input,
                weight,
                bias,
            )

    class Embedding(
        GGMLLayer,
        comfy.ops.manual_cast.Embedding,
    ):
        def forward_ggml_cast_weights(
            self,
            input,
            out_dtype=None,
        ):
            output_dtype = out_dtype

            if self.weight.dtype in (
                torch.float16,
                torch.bfloat16,
            ):
                out_dtype = None

            weight, _bias = self.cast_bias_weight(
                device=input.device,
                dtype=out_dtype,
            )

            return F.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            ).to(dtype=output_dtype)

    class LayerNorm(
        GGMLLayer,
        comfy.ops.manual_cast.LayerNorm,
    ):
        def forward_ggml_cast_weights(
            self,
            input,
        ):
            if self.weight is None:
                return super().forward_comfy_cast_weights(input)

            weight, bias = self.cast_bias_weight(input)

            return F.layer_norm(
                input,
                self.normalized_shape,
                weight,
                bias,
                self.eps,
            )

    class GroupNorm(
        GGMLLayer,
        comfy.ops.manual_cast.GroupNorm,
    ):
        def forward_ggml_cast_weights(
            self,
            input,
        ):
            weight, bias = self.cast_bias_weight(input)

            return F.group_norm(
                input,
                self.num_groups,
                weight,
                bias,
                self.eps,
            )


# ------------------------------------------------------------
# GGUF ModelPatcher
# ------------------------------------------------------------

def update_folder_names_and_paths(
    key,
    targets,
):
    base = folder_paths.folder_names_and_paths.get(
        key,
        ([], {}),
    )

    base_paths = (
        base[0]
        if isinstance(
            base[0],
            (list, set, tuple),
        )
        else []
    )

    target = next(
        (
            item
            for item in targets
            if item in folder_paths.folder_names_and_paths
        ),
        targets[0],
    )

    original, _ = folder_paths.folder_names_and_paths.get(
        target,
        ([], {}),
    )

    folder_paths.folder_names_and_paths[key] = (
        original or base_paths,
        {".gguf"},
    )


update_folder_names_and_paths(
    "unet_gguf",
    ["diffusion_models", "unet"],
)


class GGUFModelPatcher(
    comfy.model_patcher.ModelPatcher
):
    patch_on_device = False
    mmap_released = False
    named_modules_to_munmap = {}

    def patch_weight_to_device(
        self,
        key,
        device_to=None,
        inplace_update=False,
    ):
        if key not in self.patches:
            return

        weight = comfy.utils.get_attr(
            self.model,
            key,
        )

        patches = self.patches[key]

        if is_quantized(weight):
            out_weight = weight.to(device_to)

            patches = move_patch_to_device(
                patches,
                self.load_device
                if self.patch_on_device
                else self.offload_device,
            )

            out_weight.patches = [
                (patches, key)
            ]

        else:
            inplace_update = (
                self.weight_inplace_update
                or inplace_update
            )

            if key not in self.backup:
                self.backup[key] = collections.namedtuple(
                    "Dimension",
                    ["weight", "inplace_update"],
                )(
                    weight.to(
                        device=self.offload_device,
                        copy=inplace_update,
                    ),
                    inplace_update,
                )

            if device_to is not None:
                temp_weight = (
                    comfy.model_management.cast_to_device(
                        weight,
                        device_to,
                        torch.float32,
                        copy=True,
                    )
                )

            else:
                temp_weight = weight.to(
                    torch.float32,
                    copy=True,
                )

            out_weight = comfy.lora.calculate_weight(
                patches,
                temp_weight,
                key,
            )

            out_weight = comfy.float.stochastic_rounding(
                out_weight,
                weight.dtype,
            )

        if inplace_update:
            comfy.utils.copy_to_param(
                self.model,
                key,
                out_weight,
            )

        else:
            comfy.utils.set_attr_param(
                self.model,
                key,
                out_weight,
            )

    def unpatch_model(
        self,
        device_to=None,
        unpatch_weights=True,
    ):
        if unpatch_weights:
            for parameter in self.model.parameters():
                if is_torch_compatible(parameter):
                    continue

                patches = getattr(
                    parameter,
                    "patches",
                    [],
                )

                if patches:
                    parameter.patches = []

        return super().unpatch_model(
            device_to=device_to,
            unpatch_weights=unpatch_weights,
        )

    def pin_weight_to_device(self, key):
        op_key = key.rsplit(".", 1)[0]

        if (
            not self.mmap_released
            and op_key in self.named_modules_to_munmap
        ):
            module = self.named_modules_to_munmap[op_key]

            module.to(self.load_device).to(
                self.offload_device
            )

            del self.named_modules_to_munmap[op_key]

        super().pin_weight_to_device(key)

    def load(
        self,
        *args,
        force_patch_weights=False,
        **kwargs,
    ):
        if not self.mmap_released:
            self.named_modules_to_munmap = dict(
                self.model.named_modules()
            )

        super().load(
            *args,
            force_patch_weights=True,
            **kwargs,
        )

        if not self.mmap_released:
            linked = []

            if kwargs.get(
                "lowvram_model_memory",
                0,
            ) > 0:
                for name, module in self.named_modules_to_munmap.items():
                    if hasattr(module, "weight"):
                        device = getattr(
                            module.weight,
                            "device",
                            None,
                        )

                        if device == self.offload_device:
                            linked.append(
                                (name, module)
                            )
                            continue

                    if hasattr(module, "bias"):
                        device = getattr(
                            module.bias,
                            "device",
                            None,
                        )

                        if device == self.offload_device:
                            linked.append(
                                (name, module)
                            )
                            continue

            if (
                linked
                and self.load_device != self.offload_device
            ):
                for name, module in linked:
                    module.to(self.load_device).to(
                        self.offload_device
                    )

            self.mmap_released = True
            self.named_modules_to_munmap = {}

    def clone(self, *args, **kwargs):
        source_class = self.__class__

        self.__class__ = GGUFModelPatcher

        try:
            new = super().clone(
                *args,
                **kwargs,
            )

        finally:
            self.__class__ = source_class

        new.__class__ = GGUFModelPatcher

        new.patch_on_device = getattr(
            self,
            "patch_on_device",
            False,
        )

        new.mmap_released = getattr(
            self,
            "mmap_released",
            False,
        )

        new._my_gguf_keep_loaded = getattr(
            self,
            "_my_gguf_keep_loaded",
            False,
        )

        if source_class != GGUFModelPatcher:
            new.size = 0

        return new


# ------------------------------------------------------------
# Process-level model cache
# ------------------------------------------------------------

_MODEL_CACHE_LOCK = threading.RLock()

# Single-entry cache:
# Keep the most recently loaded model and its GGUF Reader alive.
_MODEL_CACHE = None


def _resolve_dtype_option(value):
    if value in ("default", None):
        return None

    if value == "target":
        return "target"

    dtype = getattr(
        torch,
        value,
        None,
    )

    if dtype is None:
        raise ValueError(
            f"Unsupported dtype option: {value!r}"
        )

    return dtype


def _dtype_cache_key(value):
    if value is None:
        return "default"

    if value == "target":
        return "target"

    return str(value)


def _make_model_cache_key(
    unet_path,
    chunk_rows,
    dequant_dtype,
    patch_dtype,
    patch_on_device,
):
    real_path = os.path.normcase(
        os.path.realpath(
            os.path.abspath(unet_path)
        )
    )

    stat = os.stat(real_path)

    file_size = stat.st_size

    mtime_ns = getattr(
        stat,
        "st_mtime_ns",
        int(stat.st_mtime * 1_000_000_000),
    )

    return (
        real_path,
        file_size,
        mtime_ns,
        int(chunk_rows),
        _dtype_cache_key(dequant_dtype),
        _dtype_cache_key(patch_dtype),
        bool(patch_on_device),
    )


def _configure_linear_instances(
    model_patcher,
    dequant_dtype,
    patch_dtype,
    chunk_rows,
):
    diffusion_model = model_patcher.model

    for module in diffusion_model.modules():
        if isinstance(
            module,
            GGMLOps.Linear,
        ):
            # Instance-level values prevent one cached model's
            # settings from being overwritten by another model.
            module.dequant_dtype = dequant_dtype
            module.patch_dtype = patch_dtype
            module.gpu_dequant_chunk_rows = int(chunk_rows)


def _build_cached_model(
    unet_path,
    chunk_rows,
    dequant_dtype,
    patch_dtype,
    patch_on_device,
):
    global _MODEL_CACHE

    ops = GGMLOps()

    # These values are used during model construction.
    # They are also copied onto each Linear instance afterwards.
    ops.Linear.dequant_dtype = dequant_dtype
    ops.Linear.patch_dtype = patch_dtype
    ops.Linear.gpu_dequant_chunk_rows = int(chunk_rows)

    state_dict, extra, reader = gguf_sd_loader(
        unet_path,
        return_reader=True,
    )

    kwargs = {}

    valid_params = inspect.signature(
        comfy.sd.load_diffusion_model_state_dict
    ).parameters

    if "metadata" in valid_params:
        kwargs["metadata"] = extra.get(
            "metadata",
            {},
        )

    model = comfy.sd.load_diffusion_model_state_dict(
        state_dict,
        model_options={
            "custom_operations": ops,
        },
        **kwargs,
    )

    if model is None:
        raise RuntimeError(
            f"ComfyUI could not identify this GGUF model: {unet_path}"
        )

    model = GGUFModelPatcher.clone(model)

    _configure_linear_instances(
        model,
        dequant_dtype,
        patch_dtype,
        chunk_rows,
    )

    model.patch_on_device = bool(patch_on_device)
    model._my_gguf_keep_loaded = False

    # Keep both the patcher and reader alive.
    # The reader retains the GGUF mmap-backed tensor storage.
    _MODEL_CACHE = {
        "model": model,
        "reader": reader,
    }

    logging.info(
        "%s model cached: arch=%s",
        _LOG_PREFIX,
        extra.get("arch_str"),
    )

    return model


def _get_cached_model(
    unet_path,
    chunk_rows,
    dequant_dtype,
    patch_dtype,
    patch_on_device,
):
    global _MODEL_CACHE

    cache_key = _make_model_cache_key(
        unet_path,
        chunk_rows,
        dequant_dtype,
        patch_dtype,
        patch_on_device,
    )

    # Keep lookup and model construction under one lock.
    # This prevents two concurrent prompts from loading the same GGUF.
    with _MODEL_CACHE_LOCK:
        if (
            _MODEL_CACHE is not None
            and _MODEL_CACHE.get("key") == cache_key
        ):
            logging.info(
                "%s cache HIT: %s",
                _LOG_PREFIX,
                os.path.basename(unet_path),
            )

            base_model = _MODEL_CACHE["model"]

        else:
            logging.info(
                "%s cache MISS: %s",
                _LOG_PREFIX,
                os.path.basename(unet_path),
            )

            base_model = _build_cached_model(
                unet_path,
                chunk_rows,
                dequant_dtype,
                patch_dtype,
                patch_on_device,
            )

            _MODEL_CACHE["key"] = cache_key

        # Return an independent ModelPatcher clone.
        # ComfyUI can manage patches and load state on this clone
        # without replacing the cached base patcher.
        model = base_model.clone()

        model.patch_on_device = bool(patch_on_device)

        return model


# ------------------------------------------------------------
# Node
# ------------------------------------------------------------

class MyUNetLoaderGGUFStream:

    @classmethod
    def INPUT_TYPES(cls):
        unet_names = folder_paths.get_filename_list(
            "unet_gguf"
        )

        return {
            "required": {
                "unet_name": (unet_names,),

                "gpu_dequant_chunk_rows": (
                    [
                        "关闭",
                        "512",
                        "1024",
                        "2048",
                        "4096",
                        "8192",
                    ],
                    {
                        "default": "2048",
                    },
                ),

                "dequant_dtype": (
                    [
                        "default",
                        "target",
                        "float32",
                        "float16",
                        "bfloat16",
                    ],
                    {
                        "default": "default",
                    },
                ),

                "patch_dtype": (
                    [
                        "default",
                        "target",
                        "float32",
                        "float16",
                        "bfloat16",
                    ],
                    {
                        "default": "default",
                    },
                ),

                "patch_on_device": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),

                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/GGUF"
    TITLE = "My UNet Loader (GGUF GPU Stream)"

    def load_unet(
        self,
        unet_name,
        gpu_dequant_chunk_rows="2048",
        dequant_dtype="default",
        patch_dtype="default",
        patch_on_device=False,
        keep_model_loaded=False,
    ):
        if gpu_dequant_chunk_rows == "关闭":
            chunk_rows = 0
        else:
            chunk_rows = int(
                gpu_dequant_chunk_rows
            )

        dequant_dtype_value = _resolve_dtype_option(
            dequant_dtype
        )

        patch_dtype_value = _resolve_dtype_option(
            patch_dtype
        )

        unet_path = folder_paths.get_full_path(
            "unet_gguf",
            unet_name,
        )

        if (
            unet_path is None
            or not os.path.isfile(unet_path)
        ):
            raise FileNotFoundError(
                f"GGUF model not found: {unet_name}"
            )

        model = _get_cached_model(
            unet_path=unet_path,
            chunk_rows=chunk_rows,
            dequant_dtype=dequant_dtype_value,
            patch_dtype=patch_dtype_value,
            patch_on_device=patch_on_device,
        )

        # keep_model_loaded is a runtime residency preference.
        # It does not change the underlying model construction.
        model._my_gguf_keep_loaded = bool(
            keep_model_loaded
        )

        return (model,)


NODE_CLASS_MAPPINGS = {
    "MyUNetLoaderGGUFStream": MyUNetLoaderGGUFStream,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyUNetLoaderGGUFStream": (
        "My UNet Loader (GGUF GPU Stream)"
    ),
}