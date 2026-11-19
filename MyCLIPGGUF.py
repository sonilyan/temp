# MyCLIPGGUF.py
# A single-file, Qwen Image 2.1 / Qwen3-VL-8B GGUF loader with streamed
# Linear weights and token-selected Embedding rows.
#
# Install as:
#   ComfyUI/custom_nodes/MyCLIPGGUF.py
#
# Requires the existing ComfyUI-GGUF custom node to be installed alongside
# ComfyUI. This file does not modify that plugin or ComfyUI core.

import os
import sys
import types
import importlib
import logging
import gc
import time
import threading

import torch
import torch.nn.functional as F

import folder_paths
import comfy.ops
import comfy.sd
import comfy.model_management


_LOG = logging.getLogger("MyCLIPGGUF")
_VENDOR_PACKAGE_NAME = "_myclipgguf_vendor"
_VENDOR = None

# Keep one fully constructed CLIP alive across repeated workflow loops.
# The model's streamable matrix parameters retain their CPU/mmap-backed
# storage through this reference, so later loops do not reopen either GGUF.
_CLIP_CACHE = None
_CLIP_CACHE_LOCK = threading.RLock()


def _memory_info(label):
    """Log PyTorch CUDA/ROCm allocator statistics without clearing memory."""
    try:
        if not torch.cuda.is_available():
            return
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        _LOG.info(
            "MyCLIPGGUF Memory [%s]: allocated=%.2f GiB, reserved=%.2f GiB, "
            "device_free=%.2f GiB, device_total=%.2f GiB",
            label, allocated / (1024 ** 3), reserved / (1024 ** 3),
            free_bytes / (1024 ** 3), total_bytes / (1024 ** 3),
        )
    except Exception as exc:
        _LOG.debug("MyCLIPGGUF: memory query failed at %s: %s", label, exc)


def _stage_done(stage, started):
    _LOG.info("MyCLIPGGUF: %s 完成，耗时 %.2f 秒", stage, time.perf_counter() - started)


def _clip_cache_key(clip_path, chunk_rows):
    """Fingerprint the text GGUF and the mmproj selected by ComfyUI-GGUF."""
    text_path = os.path.normcase(os.path.realpath(clip_path))

    def _file_signature(path):
        stat = os.stat(path)
        return (os.path.normcase(os.path.realpath(path)), stat.st_size, stat.st_mtime_ns)

    text_signature = _file_signature(text_path)
    vendor = _load_vendor_modules()
    strip_suffix = vendor["loader"].strip_quant_suffix
    text_name = os.path.splitext(os.path.basename(text_path))[0].lower()
    text_name = strip_suffix(text_name)
    root = os.path.dirname(text_path)
    mmproj_candidates = []
    for filename in os.listdir(root):
        name, ext = os.path.splitext(filename)
        if ext.lower() == ".gguf" and "mmproj" in name.lower() and text_name in name.lower():
            mmproj_candidates.append(os.path.join(root, filename))
    # The upstream loader uses the first os.listdir() match. Include all
    # matching candidates in the key so additions/removals also invalidate it.
    mmproj_signature = tuple(_file_signature(path) for path in mmproj_candidates)
    return (text_signature, mmproj_signature, int(chunk_rows))


def _cleanup_after_text_encode():
    """Collect unreachable Python objects and release unused allocator blocks."""
    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                _LOG.debug("MyCLIPGGUF: CUDA synchronize skipped: %s", exc)
        _memory_info("TE cleanup before")
        collected = gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as exc:
                _LOG.debug("MyCLIPGGUF: empty_cache skipped: %s", exc)
        _memory_info("TE cleanup after")
        _LOG.info("MyCLIPGGUF: TE 清理完成，GC 回收对象数=%d；不清理 Windows 文件缓存。", collected)
    except Exception as exc:
        _LOG.warning("MyCLIPGGUF: TE 清理失败：%s", exc)


def _install_encode_diagnostics(clip):
    """Wrap the public TE entry point; internal vision stages are not instrumented."""
    model = getattr(clip, "cond_stage_model", None)
    original = getattr(model, "encode_token_weights", None) if model is not None else None
    if not callable(original):
        _LOG.warning("MyCLIPGGUF: 找不到 encode_token_weights，跳过 TE 诊断包装。")
        return
    if getattr(original, "_myclipgguf_diagnostics", False):
        return

    def wrapped(*args, **kwargs):
        started = time.perf_counter()
        _LOG.info("MyCLIPGGUF: TE 编码开始。")
        _memory_info("TE start")
        try:
            return original(*args, **kwargs)
        finally:
            _stage_done("TE 编码", started)
            _cleanup_after_text_encode()

    wrapped._myclipgguf_diagnostics = True
    wrapped._myclipgguf_original = original
    model.encode_token_weights = wrapped
    _LOG.info("MyCLIPGGUF: 已安装 TE 编码进度与清理诊断。")


# -----------------------------------------------------------------------------
# Locate and import the installed ComfyUI-GGUF modules without importing its
# package __init__.py. The plugin's own loader/ops/dequant code is reused, so
# GGUF parsing and supported quantization formats remain consistent with the
# installed plugin version.
# -----------------------------------------------------------------------------

def _find_gguf_plugin_dir():
    roots = [os.path.dirname(os.path.abspath(__file__))]
    try:
        roots.extend(folder_paths.get_folder_paths("custom_nodes"))
    except Exception:
        pass

    checked = set()
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        root = os.path.abspath(root)
        if root in checked:
            continue
        checked.add(root)

        direct = [
            os.path.join(root, "ComfyUI-GGUF"),
            os.path.join(root, "ComfyUI-GGUF-main"),
        ]
        for candidate in direct:
            if (os.path.isfile(os.path.join(candidate, "loader.py"))
                    and os.path.isfile(os.path.join(candidate, "ops.py"))
                    and os.path.isfile(os.path.join(candidate, "dequant.py"))):
                return candidate

        try:
            children = os.listdir(root)
        except OSError:
            children = []
        for child in children:
            if "gguf" not in child.lower():
                continue
            candidate = os.path.join(root, child)
            if (os.path.isdir(candidate)
                    and os.path.isfile(os.path.join(candidate, "loader.py"))
                    and os.path.isfile(os.path.join(candidate, "ops.py"))
                    and os.path.isfile(os.path.join(candidate, "dequant.py"))):
                return candidate

    raise RuntimeError(
        "MyCLIPGGUF: 找不到 ComfyUI-GGUF 插件目录。"
        "请先安装 ComfyUI-GGUF，并确认它位于 ComfyUI/custom_nodes 下。"
    )


def _load_vendor_modules():
    global _VENDOR
    if _VENDOR is not None:
        return _VENDOR

    plugin_dir = _find_gguf_plugin_dir()

    # Create a lightweight package namespace. This enables the plugin's
    # relative imports (.ops / .dequant) but avoids executing its __init__.py,
    # which would register the original node classes a second time.
    package = types.ModuleType(_VENDOR_PACKAGE_NAME)
    package.__path__ = [plugin_dir]
    package.__package__ = _VENDOR_PACKAGE_NAME
    package.__file__ = os.path.join(plugin_dir, "__init__.py")
    package.__spec__ = importlib.machinery.ModuleSpec(
        _VENDOR_PACKAGE_NAME, loader=None, is_package=True
    )
    package.__spec__.submodule_search_locations = [plugin_dir]
    sys.modules[_VENDOR_PACKAGE_NAME] = package

    vendor_ops = importlib.import_module(_VENDOR_PACKAGE_NAME + ".ops")
    vendor_loader = importlib.import_module(_VENDOR_PACKAGE_NAME + ".loader")
    vendor_dequant = importlib.import_module(_VENDOR_PACKAGE_NAME + ".dequant")
    vendor_nodes = importlib.import_module(_VENDOR_PACKAGE_NAME + ".nodes")

    _VENDOR = {
        "dir": plugin_dir,
        "ops": vendor_ops,
        "loader": vendor_loader,
        "dequant": vendor_dequant,
        "nodes": vendor_nodes,
    }
    _LOG.info("MyCLIPGGUF: 使用 ComfyUI-GGUF 插件目录：%s", plugin_dir)
    return _VENDOR


def _install_gguf_rms_norm_compat():
    """Make GGUF Tensor subclasses safe at the PyTorch RMSNorm boundary.

    GGUF tensors can leak into RMSNorm as either the activation input or the
    norm weight. Activations are already computed values, so only remove the
    Tensor subclass. GGUF weights are dequantized first when their qtype is
    not directly torch-compatible; the resulting small vector is then
    converted to a plain torch.Tensor.
    """
    try:
        import comfy.rmsnorm as comfy_rmsnorm
        import comfy.model_management as model_management
    except Exception as exc:
        _LOG.warning("MyCLIPGGUF: 无法导入 RMSNorm 依赖，跳过兼容补丁：%s", exc)
        return

    if getattr(comfy_rmsnorm.rms_norm, "_myclipgguf_compat", False):
        return

    original_rms_norm = comfy_rmsnorm.rms_norm

    def _plain_tensor(value):
        if isinstance(value, torch.Tensor) and type(value) is not torch.Tensor:
            return value.as_subclass(torch.Tensor)
        return value

    def _gguf_safe_rms_norm(x, weight, eps):
        # x.float() can preserve GGMLTensor's subclass. It is an activation,
        # not quantized model storage, so stripping the subclass is sufficient.
        x = _plain_tensor(x)

        # Norm weights are tiny vectors. Dequantize only if this is a GGUF
        # tensor whose qtype cannot be consumed as a normal torch tensor.
        if isinstance(weight, torch.Tensor) and hasattr(weight, "tensor_type"):
            vendor = _load_vendor_modules()
            if not vendor["dequant"].is_torch_compatible(weight):
                weight = vendor["dequant"].dequantize_tensor(weight, dtype=x.dtype)

        # Match Comfy's normal dtype/device casting, then guarantee that
        # F.rms_norm receives an ordinary Tensor rather than a GGML subclass.
        weight = model_management.cast_to(weight, dtype=x.dtype, device=x.device)
        weight = _plain_tensor(weight)

        return torch.nn.functional.rms_norm(
            x, weight.shape, weight=weight, eps=eps
        )

    _gguf_safe_rms_norm._myclipgguf_compat = True
    _gguf_safe_rms_norm._myclipgguf_original = original_rms_norm
    comfy_rmsnorm.rms_norm = _gguf_safe_rms_norm
    _LOG.warning(
        "MyCLIPGGUF: 已安装 GGUF RMSNorm 兼容补丁；输入激活只剥离 Tensor 子类，"
        "GGUF 权重按需反量化后再传给 PyTorch。"
    )


# Install at custom-node import time, before any sampling begins.
_install_gguf_rms_norm_compat()


# -----------------------------------------------------------------------------
# A GGMLTensor subclass which keeps only 2D matrix weights on CPU. Those
# matrices are streamed by MyLinear / MyEmbedding. Other tensors retain the
# installed GGUF plugin's normal .to() behavior (e.g. vision position tables,
# convolution weights and normalization weights).
# -----------------------------------------------------------------------------

MyGGMLTensor = None


def _ensure_my_tensor_class():
    """Create MyGGMLTensor as a real subclass of the installed plugin tensor."""
    global MyGGMLTensor
    if MyGGMLTensor is not None:
        return MyGGMLTensor

    vendor_base = _load_vendor_modules()["ops"].GGMLTensor

    class _MyGGMLTensor(vendor_base):
        def __new__(cls, data, *, tensor_type, tensor_shape, patches=None, stream_cpu=False):
            if isinstance(data, torch.Tensor) and type(data) is not torch.Tensor:
                data = data.as_subclass(torch.Tensor)
            return super().__new__(
                cls,
                data,
                tensor_type=tensor_type,
                tensor_shape=tensor_shape,
                patches=list(patches or []),
            )

        def __init__(self, data, *, tensor_type, tensor_shape, patches=None, stream_cpu=False):
            super().__init__(
                data,
                tensor_type=tensor_type,
                tensor_shape=tensor_shape,
                patches=list(patches or []),
            )
            self.my_stream_cpu = bool(stream_cpu)

        @property
        def shape(self):
            if not hasattr(self, "tensor_shape"):
                self.tensor_shape = self.size()
            return self.tensor_shape

        def to(self, *args, **kwargs):
            # Streamable 2D matrices stay in their original CPU/mmap storage.
            # MyLinear / MyEmbedding explicitly transfer only the active rows.
            if getattr(self, "my_stream_cpu", False):
                return self
            result = super().to(*args, **kwargs)
            result.my_stream_cpu = getattr(self, "my_stream_cpu", False)
            return result

    _MyGGMLTensor.__name__ = "MyGGMLTensor"
    _MyGGMLTensor.__qualname__ = "MyGGMLTensor"
    MyGGMLTensor = _MyGGMLTensor
    return MyGGMLTensor


def _wrap_gguf_tensors(state_dict):
    """Wrap vendor GGMLTensor values and mark 2D weight matrices for streaming."""
    vendor = _load_vendor_modules()
    base_type = vendor["ops"].GGMLTensor
    tensor_type = _ensure_my_tensor_class()
    converted = 0
    streamed = 0

    for key, value in list(state_dict.items()):
        if not isinstance(value, base_type):
            continue
        logical_shape = getattr(value, "tensor_shape", value.shape)
        # GGUF linear/embedding matrices are [out_features, in_features].
        # Position embeddings are not Linear weights and must remain movable.
        is_matrix_weight = (
            len(logical_shape) == 2
            and key.endswith(".weight")
            and "pos_embed" not in key.lower()
        )
        # The text encoder tensors produced directly by gguf_sd_loader carry
        # tensor_type. Some mmproj tensors, however, have passed through view /
        # concat operations and may be GGMLTensor subclasses without metadata.
        # Those visual tensors are deliberately not wrapped by this function.
        qtype = getattr(value, "tensor_type", None)
        if qtype is None and is_matrix_weight:
            raise RuntimeError(
                "MyCLIPGGUF: 文本权重 %s 缺少 tensor_type 元数据；"
                "无法安全判断 GGUF 量化格式。原始类型=%s，shape=%s。"
                % (key, type(value).__name__, tuple(logical_shape))
            )
        if qtype is None:
            continue

        raw = value.as_subclass(torch.Tensor)

        # F16/F32 tensors are already ordinary PyTorch-compatible data; they
        # do not need GGMLTensor dispatch. Keeping them as GGMLTensor breaks
        # direct ops such as F.rms_norm(), which call __torch_function__ and
        # have no GGMLTensor implementation. Only streamed matrix weights
        # need our metadata-carrying subclass. All other torch-compatible
        # tensors (norm vectors, biases, positional tables, etc.) become plain
        # torch.Tensor values and follow ComfyUI's normal loading/casting.
        is_torch_compatible = vendor["dequant"].is_torch_compatible(value)
        if not is_matrix_weight and is_torch_compatible:
            state_dict[key] = raw
            converted += 1
            continue

        wrapped = tensor_type(
            raw,
            tensor_type=qtype,
            tensor_shape=logical_shape,
            patches=getattr(value, "patches", []),
            stream_cpu=is_matrix_weight,
        )
        if getattr(value, "is_largest_weight", False):
            wrapped.is_largest_weight = True
        state_dict[key] = wrapped
        converted += 1
        streamed += int(is_matrix_weight)

    _LOG.info(
        "MyCLIPGGUF: GGMLTensor 包装完成：%d 个；其中 %d 个二维权重保持 CPU/mmap 并采用流式读取。",
        converted, streamed,
    )
    return state_dict


# -----------------------------------------------------------------------------
# Raw row extraction and per-row dequantization.
# The code checks the GGUF matrix's physical byte layout before using it. If a
# tensor is not row-major in the expected GGUF layout, it fails explicitly
# rather than silently calculating incorrect values.
# -----------------------------------------------------------------------------

def _plain_tensor(tensor):
    if isinstance(tensor, torch.Tensor) and type(tensor) is not torch.Tensor:
        return tensor.as_subclass(torch.Tensor)
    return tensor


def _get_matrix_rows(weight, *, device, dtype, start=None, end=None, row_ids=None):
    vendor = _load_vendor_modules()
    gguf = importlib.import_module("gguf")
    dequant = vendor["dequant"]

    logical_shape = tuple(int(x) for x in getattr(weight, "tensor_shape", weight.shape))
    if len(logical_shape) != 2:
        raise RuntimeError(
            "MyCLIPGGUF: 流式矩阵必须是二维权重，实际 shape=%s" % (logical_shape,)
        )

    out_features, in_features = logical_shape
    qtype = getattr(weight, "tensor_type", None)
    raw = _plain_tensor(weight)

    if raw.device.type != "cpu":
        raise RuntimeError(
            "MyCLIPGGUF: 发现流式权重已经离开 CPU：device=%s，shape=%s。"
            "这意味着模型加载阶段发生了整矩阵搬运。" % (raw.device, logical_shape)
        )
    if not raw.is_contiguous():
        raise RuntimeError(
            "MyCLIPGGUF: GGUF 原始矩阵不是连续存储，拒绝在未确认布局时复制整张权重。"
            "physical_shape=%s logical_shape=%s" % (tuple(raw.shape), logical_shape)
        )

    if dequant.is_torch_compatible(weight):
        if raw.numel() != out_features * in_features:
            raise RuntimeError(
                "MyCLIPGGUF: F16/F32 权重元素数不匹配：physical=%d logical=%s"
                % (raw.numel(), logical_shape)
            )
        matrix = raw.reshape(out_features, in_features)
        if row_ids is not None:
            ids = row_ids.to(device="cpu", dtype=torch.long)
            selected = matrix.index_select(0, ids)
        else:
            selected = matrix[int(start):int(end)]
        return selected.to(device=device, dtype=dtype, non_blocking=True)

    if qtype not in dequant.dequantize_functions:
        raise RuntimeError(
            "MyCLIPGGUF: 当前插件的分块反量化表不支持 qtype=%s。"
            "为避免整矩阵反量化造成 OOM，本节点不会自动退回整表反量化。"
            % getattr(qtype, "name", repr(qtype))
        )

    try:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    except Exception as e:
        raise RuntimeError("MyCLIPGGUF: 无法读取 GGUF qtype 的 block/type size：%s" % e) from e

    block_size = int(block_size)
    type_size = int(type_size)
    if in_features % block_size != 0:
        raise RuntimeError(
            "MyCLIPGGUF: 输入维度不能按量化 block 对齐：in_features=%d block_size=%d qtype=%s"
            % (in_features, block_size, getattr(qtype, "name", qtype))
        )

    row_bytes = (in_features // block_size) * type_size
    raw_bytes = raw.view(torch.uint8)
    expected_bytes = out_features * row_bytes
    if raw_bytes.numel() != expected_bytes:
        raise RuntimeError(
            "MyCLIPGGUF: GGUF 物理布局不符合逐行读取假设："
            "physical_shape=%s physical_bytes=%d expected_bytes=%d "
            "logical_shape=%s row_bytes=%d qtype=%s。"
            % (
                tuple(raw.shape), raw_bytes.numel(), expected_bytes,
                logical_shape, row_bytes, getattr(qtype, "name", qtype),
            )
        )

    byte_matrix = raw_bytes.reshape(out_features, row_bytes)
    if row_ids is not None:
        ids = row_ids.to(device="cpu", dtype=torch.long)
        selected_bytes = byte_matrix.index_select(0, ids)
    else:
        selected_bytes = byte_matrix[int(start):int(end)]

    # Transfer only this row block's packed bytes, then dequantize on the GPU.
    packed = selected_bytes.to(device=device, non_blocking=True)
    blocks = packed.reshape(-1, type_size)
    dequantized = dequant.dequantize_functions[qtype](
        blocks, block_size, type_size, dtype
    )
    rows_count = int(row_ids.numel()) if row_ids is not None else int(end) - int(start)
    return dequantized.reshape(rows_count, in_features).to(dtype=dtype)


def _full_weight_fallback(layer, input_tensor, output_dtype=None):
    """Correctness fallback for patched weights (e.g. LoRA)."""
    dtype = output_dtype or input_tensor.dtype
    weight = layer.get_weight(layer.weight, dtype)
    weight = weight.to(device=input_tensor.device, dtype=dtype)
    bias = None
    if layer.bias is not None:
        bias = layer.get_weight(layer.bias, dtype)
        bias = bias.to(device=input_tensor.device, dtype=dtype)
    return weight, bias


def _save_stream_state_dict(layer, destination, prefix, temp_rows):
    """Meta-only state-dict representation with streamed, not full, temp size."""
    meta = torch.device("meta")
    weight = layer.weight
    destination[prefix + "weight"] = torch.empty(
        tuple(weight.shape), device=meta, dtype=weight.dtype
    )
    if layer.bias is not None:
        destination[prefix + "bias"] = torch.empty(
            tuple(layer.bias.shape), device=meta, dtype=layer.bias.dtype
        )
    if getattr(layer, "largest_layer", False):
        dtype = getattr(layer, "dequant_dtype", None)
        if dtype is None or dtype == "target":
            dtype = torch.float16
        if isinstance(layer, torch.nn.Embedding):
            temp_shape = (int(temp_rows), int(layer.embedding_dim))
        else:
            temp_shape = (int(temp_rows), int(layer.in_features))
        destination[prefix + "temp.weight"] = torch.empty(
            temp_shape, device=meta, dtype=dtype
        )


# -----------------------------------------------------------------------------
# Custom operations. All ordinary ComfyUI-GGUF operations are inherited;
# only Linear and Embedding are replaced with streamed implementations.
# -----------------------------------------------------------------------------

def _make_operations_class(chunk_rows):
    vendor = _load_vendor_modules()
    vendor_ops = vendor["ops"]
    chunk_rows = int(chunk_rows)

    class MyLinear(vendor_ops.GGMLLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            # Avoid allocating a full-precision placeholder matrix before GGUF
            # state-dict assignment (important for large Qwen layers on Windows).
            torch.nn.Module.__init__(self)
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.weight = None
            self.bias = None
            self.weight_comfy_model_dtype = dtype
            self.bias_comfy_model_dtype = dtype
            self.comfy_need_lazy_init_bias = bool(bias)

        def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
            return _save_stream_state_dict(self, destination, prefix, chunk_rows)

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            if isinstance(self.weight, MyGGMLTensor) and getattr(self.weight, "my_stream_cpu", False):
                return self.ggml_save_to_state_dict(destination, prefix, keep_vars)
            return super()._save_to_state_dict(destination, prefix, keep_vars)

        def forward_comfy_cast_weights(self, input_tensor):
            weight = self.weight
            if not isinstance(weight, MyGGMLTensor) or not getattr(weight, "my_stream_cpu", False):
                if self.is_ggml_quantized():
                    return vendor_ops.GGMLOps.Linear.forward_ggml_cast_weights(self, input_tensor)
                return comfy.ops.manual_cast.Linear.forward_comfy_cast_weights(self, input_tensor)

            if getattr(weight, "patches", None):
                # LoRA/weight patches retain the original plugin behavior.
                full_weight, full_bias = _full_weight_fallback(self, input_tensor)
                return F.linear(input_tensor, full_weight, full_bias)

            if input_tensor.shape[-1] != self.in_features:
                raise RuntimeError(
                    "MyCLIPGGUF: Linear 输入维度错误：input=%s in_features=%d"
                    % (tuple(input_tensor.shape), self.in_features)
                )

            input_shape = tuple(input_tensor.shape)
            x = input_tensor.reshape(-1, self.in_features)
            output = torch.empty(
                (x.shape[0], self.out_features),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )

            bias = None
            if self.bias is not None:
                bias = _plain_tensor(self.bias).to(
                    device=input_tensor.device,
                    dtype=input_tensor.dtype,
                    non_blocking=True,
                )

            with torch.no_grad():
                for row_start in range(0, self.out_features, chunk_rows):
                    row_end = min(row_start + chunk_rows, self.out_features)
                    w = _get_matrix_rows(
                        weight,
                        device=input_tensor.device,
                        dtype=input_tensor.dtype,
                        start=row_start,
                        end=row_end,
                    )
                    b = bias[row_start:row_end] if bias is not None else None
                    output[:, row_start:row_end] = F.linear(x, w, b)
                    del w

            return output.reshape(*input_shape[:-1], self.out_features)

    class MyEmbedding(vendor_ops.GGMLLayer, comfy.ops.manual_cast.Embedding):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                     norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                     _freeze=False, device=None, dtype=None):
            # Do not allocate the 151936 x 4096 embedding table during init.
            torch.nn.Module.__init__(self)
            self.num_embeddings = int(num_embeddings)
            self.embedding_dim = int(embedding_dim)
            self.padding_idx = padding_idx
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.sparse = sparse
            self.weight = None
            self.bias = None
            self.weight_comfy_model_dtype = dtype
            self._my_embedding_dtype = dtype

        def ggml_save_to_state_dict(self, destination, prefix, keep_vars):
            # The embedding path materializes only the token rows present in the
            # current prompt. Reserve for up to 8192 unique rows, not the vocab.
            return _save_stream_state_dict(self, destination, prefix, 8192)

        def _save_to_state_dict(self, destination, prefix, keep_vars):
            if isinstance(self.weight, MyGGMLTensor) and getattr(self.weight, "my_stream_cpu", False):
                return self.ggml_save_to_state_dict(destination, prefix, keep_vars)
            return super()._save_to_state_dict(destination, prefix, keep_vars)

        def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs):
            # Always use the GGUF-aware parameter assignment, including F16/F32
            # embedding tables that are larger than the plugin's 64K threshold.
            return self.ggml_load_from_state_dict(
                state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs,
            )

        def forward_comfy_cast_weights(self, input_ids, out_dtype=None):
            weight = self.weight
            if not isinstance(weight, MyGGMLTensor) or not getattr(weight, "my_stream_cpu", False):
                if self.is_ggml_quantized():
                    return vendor_ops.GGMLOps.Embedding.forward_ggml_cast_weights(
                        self, input_ids, out_dtype=out_dtype
                    )
                return comfy.ops.manual_cast.Embedding.forward_comfy_cast_weights(
                    self, input_ids, out_dtype=out_dtype
                )

            if getattr(weight, "patches", None):
                target_dtype = out_dtype or self._my_embedding_dtype or torch.float16
                full_weight = self.get_weight(weight, target_dtype).to(
                    device=input_ids.device, dtype=target_dtype
                )
                return F.embedding(
                    input_ids, full_weight, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse,
                ).to(dtype=out_dtype or target_dtype)

            target_dtype = out_dtype or self._my_embedding_dtype
            if target_dtype is None:
                qtype = getattr(weight, "tensor_type", None)
                qname = getattr(qtype, "name", "")
                target_dtype = torch.float32 if qname == "F32" else torch.float16

            flat_ids = input_ids.reshape(-1).to(dtype=torch.long)
            unique_ids, inverse = torch.unique(
                flat_ids, sorted=True, return_inverse=True
            )
            rows = _get_matrix_rows(
                weight,
                device=input_ids.device,
                dtype=target_dtype,
                row_ids=unique_ids,
            )

            remapped = inverse.reshape(input_ids.shape).to(device=input_ids.device)
            if self.max_norm is not None:
                mapped_padding_idx = None
                if self.padding_idx is not None:
                    matches = torch.nonzero(unique_ids == self.padding_idx, as_tuple=False)
                    if matches.numel() > 0:
                        mapped_padding_idx = int(matches[0, 0].item())
                result = F.embedding(
                    remapped, rows, mapped_padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse,
                )
            else:
                result = rows.index_select(0, remapped.reshape(-1)).reshape(
                    *input_ids.shape, self.embedding_dim
                )
            return result.to(dtype=out_dtype or target_dtype)

        def host_rows(self, input_ids, out_dtype=None):
            # Used by some ComfyUI host-row paths. Reuse the same selected-row
            # implementation rather than materializing the vocabulary table.
            return self.forward_comfy_cast_weights(input_ids, out_dtype=out_dtype)

    class MyGGMLOps(vendor_ops.GGMLOps):
        Linear = MyLinear
        Embedding = MyEmbedding

    MyGGMLOps.__name__ = "MyGGMLOps_Chunk%d" % chunk_rows
    return MyGGMLOps


# -----------------------------------------------------------------------------
# Qwen Image 2.1-specific GGUF state-dict preparation.
# Do NOT call vendor gguf_clip_loader(): that function eagerly dequantizes the
# entire token_embd.weight table. We intentionally call gguf_sd_loader() and
# reproduce only the Qwen3-VL key mapping + matching mmproj loading.
# -----------------------------------------------------------------------------

def _load_qwen_image21_state_dict(clip_path):
    vendor = _load_vendor_modules()
    loader = vendor["loader"]

    stage_start = time.perf_counter()
    _LOG.info("MyCLIPGGUF: [1/4] 开始读取文本 GGUF 权重：%s", os.path.basename(clip_path))
    _memory_info("before text GGUF")
    state_dict, extra = loader.gguf_sd_loader(clip_path, is_text_model=True)
    _stage_done("文本 GGUF 权重读取", stage_start)
    _memory_info("after text GGUF")
    arch = extra.get("arch_str")
    if arch != "qwen3vl":
        raise RuntimeError(
            "MyCLIPGGUF 目前只支持 Qwen Image 2.1 的 qwen3vl GGUF，"
            "文件架构检测结果为：%r" % arch
        )

    token_key = "token_embd.weight"
    if token_key not in state_dict:
        raise RuntimeError("MyCLIPGGUF: GGUF 中找不到 token_embd.weight。")

    token_shape = tuple(int(x) for x in state_dict[token_key].shape)
    if len(token_shape) != 2 or token_shape[1] != 4096:
        raise RuntimeError(
            "MyCLIPGGUF 目前针对 Qwen3-VL-8B / Qwen Image 2.1，"
            "预期 token_embd.weight 的 hidden size 为 4096，实际为 %s。" % (token_shape,)
        )

    # Match the original plugin's llama.cpp -> ComfyUI key conversion, but
    # deliberately skip its eager token_embd.weight dequantization.
    stage_start = time.perf_counter()
    state_dict = loader.sd_map_replace(state_dict, loader.LLAMA_SD_MAP)
    _stage_done("文本权重 key 映射", stage_start)

    # Wrap the text GGUF tensors BEFORE loading/merging mmproj. The mmproj
    # loader performs view/reshape/concatenation on some F16/F32 tensors; those
    # derived tensors can remain GGMLTensor subclasses while losing custom
    # tensor_type attributes. They are ordinary visual weights and must not be
    # treated as streamable text matrices anyway.
    stage_start = time.perf_counter()
    state_dict = _wrap_gguf_tensors(state_dict)
    _stage_done("文本 GGUF 流式权重包装", stage_start)

    # Keep the original plugin's mmproj discovery, Qwen3-VL merger mapping,
    # split-QKV reconstruction and visual state-dict behavior.
    stage_start = time.perf_counter()
    _LOG.info("MyCLIPGGUF: [2/4] 开始读取视觉塔 mmproj 权重。")
    _memory_info("before mmproj")
    vision_state = loader.gguf_mmproj_loader(clip_path)
    _stage_done("视觉塔 mmproj 读取", stage_start)
    _memory_info("after mmproj")
    if not vision_state:
        raise RuntimeError(
            "Qwen3-VL / Qwen Image 2.1 必须有匹配的 mmproj GGUF。"
            "请把对应 mmproj 文件放在文本编码器 GGUF 同一目录，并确保文件名包含匹配的模型名。"
        )
    vision_state = {"model." + key: value for key, value in vision_state.items()}
    state_dict.update(vision_state)

    # Text matrix weights remain backed by their original CPU/mmap storage.
    # Visual mmproj weights retain the plugin's original loading behavior.

    _LOG.info(
        "MyCLIPGGUF: [3/4] 权重准备完成；arch=%s，token_embd=%s；"
        "词表保持按 token 行读取。",
        arch, token_shape,
    )
    return state_dict, extra
# -----------------------------------------------------------------------------
# ComfyUI node
# -----------------------------------------------------------------------------

class MyCLIPGGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        vendor = _load_vendor_modules()
        del vendor
        names = sorted(set(
            folder_paths.get_filename_list("clip_gguf")
            + [x for x in folder_paths.get_filename_list("clip") if x.lower().endswith(".gguf")]
        ))
        if not names:
            names = folder_paths.get_filename_list("clip_gguf")
        return {
            "required": {
                "model": ("MODEL",),
                "clip_name": (names,),
                "linear_chunk_rows": ([128, 256, 512, 1024, 2048], {"default": 512}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/CLIP"
    TITLE = "My CLIP GGUF Loader (Qwen Image 2.1 Stream)"

    @staticmethod
    def _resolve_clip_path(clip_name):
        path = folder_paths.get_full_path("clip_gguf", clip_name)
        if path is None:
            path = folder_paths.get_full_path("clip", clip_name)
        if path is None or not os.path.isfile(path):
            raise FileNotFoundError("MyCLIPGGUF: 找不到 GGUF 文件：%s" % clip_name)
        if not path.lower().endswith(".gguf"):
            raise ValueError("MyCLIPGGUF 只接受 .gguf 文件：%s" % path)
        return path

    def load_clip(self, model, clip_name, linear_chunk_rows=512):
        _LOG.info("MyCLIPGGUF: 开始预加载 DiT 到 GPU。")
        _memory_info("before DiT preload")

        comfy.model_management.load_models_gpu([model])

        _memory_info("after DiT preload")
        _LOG.info("MyCLIPGGUF: DiT GPU 预加载完成。")
        global _CLIP_CACHE
        total_start = time.perf_counter()
        vendor = _load_vendor_modules()
        clip_path = self._resolve_clip_path(clip_name)
        cache_key = _clip_cache_key(clip_path, linear_chunk_rows)

        # Serialize cache lookup/build. Returning the same CLIP object is
        # intentional: ComfyUI may load/offload its patcher between loops, but
        # its CPU/mmap-backed GGUF matrix tensors remain owned by the cached
        # model and are not reparsed from disk.
        with _CLIP_CACHE_LOCK:
            if _CLIP_CACHE is not None and _CLIP_CACHE["key"] == cache_key:
                clip = _CLIP_CACHE["clip"]
                _LOG.info(
                    "MyCLIPGGUF: CLIP_CACHE_HIT：%s；Linear chunk_rows=%d；"
                    "复用现有 CLIP/mmap，不重新读取文本 GGUF 或 mmproj。",
                    os.path.basename(clip_path), int(linear_chunk_rows),
                )
                _memory_info("CLIP cache hit")
                return (clip,)

            _LOG.info(
                "MyCLIPGGUF: CLIP_CACHE_MISS：%s；Linear chunk_rows=%d；开始构建。",
                os.path.basename(clip_path), int(linear_chunk_rows),
            )
            state_dict, extra = _load_qwen_image21_state_dict(clip_path)

            stage_start = time.perf_counter()
            _LOG.info("MyCLIPGGUF: [4/4] 开始构建 CLIP 文本编码器。")
            _memory_info("before CLIP build")
            operations = _make_operations_class(linear_chunk_rows)
            model_options = {
                "custom_operations": operations,
                "initial_device": comfy.model_management.text_encoder_offload_device(),
            }

            clip_type = comfy.sd.CLIPType.QWEN_IMAGE
            clip = comfy.sd.load_text_encoder_state_dicts(
                state_dicts=[state_dict],
                clip_type=clip_type,
                model_options=model_options,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
            if clip is None or getattr(clip, "patcher", None) is None:
                raise RuntimeError("MyCLIPGGUF: ComfyUI 没有成功构建 Qwen Image 2.1 CLIP。")

            # Reuse the installed plugin's GGUF-aware patcher. Streamable 2D
            # MyGGMLTensor parameters refuse full-device migration, while other
            # model parameters follow the ordinary ComfyUI load/offload behavior.
            clip.patcher = vendor["nodes"].GGUFModelPatcher.clone(clip.patcher)
            _stage_done("CLIP 构建及 patcher 克隆", stage_start)
            _memory_info("after CLIP build")

            # Keep the mmap-backed model alive in _CLIP_CACHE. Only temporary
            # state-dict containers are dropped; the model owns the tensor views.
            del state_dict
            del extra
            collected = gc.collect()
            _LOG.info("MyCLIPGGUF: 临时 state_dict/extra 引用已释放，GC 回收对象数=%d。", collected)
            _memory_info("after temporary state_dict release")
            _install_encode_diagnostics(clip)

            # Single-entry cache avoids retaining multiple large text encoders.
            _CLIP_CACHE = {"key": cache_key, "clip": clip}

            _LOG.info(
                "MyCLIPGGUF: CLIP 加载完成并已缓存：%s；Linear chunk_rows=%d；"
                "总耗时 %.2f 秒。",
                os.path.basename(clip_path), int(linear_chunk_rows),
                time.perf_counter() - total_start,
            )
            return (clip,)


NODE_CLASS_MAPPINGS = {
    "MyCLIPGGUFLoader": MyCLIPGGUFLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyCLIPGGUFLoader": MyCLIPGGUFLoader.TITLE,
}