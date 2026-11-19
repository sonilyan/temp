
import torch

import comfy.model_management

from comfy.ldm.qwen_image21.model import (
    QwenImage21Transformer2DModel,
)

from comfy.ldm.wan.model_animate2 import (
    PoseBranchCache,
    quantize_cache,
)


# ============================================================
# Qwen21 Mixed KV Cache
#
# Repository:
# patientx-cfz/comfyui-rocm
#
# Allocation:
# Strict Transformer Block count
#
# 32 Blocks / 50%:
# GPU: Block 0-15
# CPU: Block 16-31
#
# Original PoseBranchCache:
# - quantize_cache
# - prefetch
# - take
# - stream synchronization
# - pinned memory
# ============================================================


VERSION = "1.1.0"

PREFIX = "[QWEN21 MIXED CACHE]"


def log(message):
    print(f"{PREFIX} {message}", flush=True)


def format_bytes(value):
    return f"{value / (1024 ** 3):.3f} GB"


def tensor_bytes(obj, visited=None):

    if visited is None:
        visited = set()

    if id(obj) in visited:
        return 0

    visited.add(id(obj))

    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()

    if isinstance(obj, dict):
        return sum(
            tensor_bytes(v, visited)
            for v in obj.values()
        )

    if isinstance(obj, (tuple, list)):
        return sum(
            tensor_bytes(v, visited)
            for v in obj
        )

    if hasattr(obj, "__dict__"):
        return tensor_bytes(vars(obj), visited)

    return 0


# ============================================================
# Mixed PoseBranchCache
# ============================================================

class MixedPoseBranchCache(PoseBranchCache):

    def __init__(
        self,
        store_device,
        dtype,
        gpu_ratio,
        total_blocks,
    ):

        # Use CPU for cache keys and native slot management.
        super().__init__(
            store_device=torch.device("cpu"),
            dtype=dtype,
        )

        self.compute_device = torch.device(
            store_device
        )

        self.gpu_ratio = max(
            0.0,
            min(100.0, float(gpu_ratio))
        )

        self.total_blocks = int(total_blocks)

        # Strict block-count allocation.
        self.gpu_blocks = int(
            self.total_blocks * self.gpu_ratio / 100.0 + 0.5
        )

        self.gpu_blocks = max(
            0,
            min(self.total_blocks, self.gpu_blocks)
        )

        self.cpu_blocks = (
            self.total_blocks - self.gpu_blocks
        )

        self.gpu_bytes = 0
        self.cpu_bytes = 0

        self.block_report = {}

        log(
            f"Created: dtype={dtype}, "
            f"ratio={self.gpu_ratio:.1f}%, "
            f"blocks={self.total_blocks}"
        )

        log(
            f"Allocation: GPU={self.gpu_blocks} blocks, "
            f"CPU={self.cpu_blocks} blocks"
        )

    # --------------------------------------------------------
    # Device selection
    # --------------------------------------------------------

    def _choose_device(self, index):

        if index < 0 or index >= self.total_blocks:

            raise IndexError(
                f"Invalid Qwen21 block index: {index}"
            )

        if index < self.gpu_blocks:

            return self.compute_device

        return torch.device("cpu")

    # --------------------------------------------------------
    # Slot selection
    # --------------------------------------------------------

    def select(self, key, create=True):

        # Keep original PoseBranchCache slot behavior.
        result = super().select(
            key,
            create=create,
        )

        if self.slot is not None:

            if "block_devices" not in self.slot:

                self.slot["block_devices"] = {}

            if "block_sizes" not in self.slot:

                self.slot["block_sizes"] = {}

            if "pinned" not in self.slot:

                self.slot["pinned"] = []

        if result:

            log("Existing cache slot selected")

        elif create and self.slot is not None:

            self.gpu_bytes = 0
            self.cpu_bytes = 0

            self.block_report = {}

            log("New cache slot created")

        return result

    # --------------------------------------------------------
    # Store one block
    # --------------------------------------------------------

    def put(self, index, tensor):

        if self.slot is None:

            raise RuntimeError(
                "Qwen21 Mixed Cache: no active cache slot"
            )

        target_device = self._choose_device(index)

        original_shape = tensor.shape

        # Original ComfyUI quantization.
        tensor, params = quantize_cache(
            tensor,
            self.dtype,
        )

        # Actual quantized tensor size.
        data_size = tensor_bytes(tensor)

        params_size = tensor_bytes(params)

        block_size = data_size + params_size

        # Copy to the assigned device.
        tensor = tensor.to(
            target_device,
            copy=True,
        )

        # Preserve original pinned CPU cache behavior.
        if target_device.type == "cpu":

            if comfy.model_management.pin_memory(
                tensor,
                evict_active=False,
            ):

                self.slot["pinned"].append(tensor)

        # Store quantization parameters on the same device.
        if params is not None:

            params = params.to_device(
                target_device
            )

        self.slot["blocks"][index] = tensor

        self.slot["params"][index] = params

        self.slot["block_devices"][index] = str(
            target_device
        )

        self.slot["block_sizes"][index] = block_size

        if self.dtype in ("int8", "int4"):

            self.slot["shape"] = original_shape

        if target_device.type == "cpu":

            self.cpu_bytes += block_size

        else:

            self.gpu_bytes += block_size

        self.block_report[index] = {

            "device": str(target_device),

            "bytes": block_size,

        }

        log(
            f"PUT block={index:02d} "
            f"device={target_device} "
            f"bytes={block_size / 1024**2:.2f} MB "
            f"GPU={format_bytes(self.gpu_bytes)} "
            f"CPU={format_bytes(self.cpu_bytes)}"
        )

        # Print statistics after all blocks are stored.
        if len(self.slot["blocks"]) == self.total_blocks:

            self.print_statistics()

    # --------------------------------------------------------
    # Prefetch
    # --------------------------------------------------------

    def prefetch(self, index, device, dtype):

        if self.slot is None:
            return

        if index not in self.slot["blocks"]:
            return

        source = self.slot["blocks"][index]

        log(
            f"PREFETCH block={index:02d} "
            f"source={source.device} "
            f"target={device}"
        )

        # Use original ComfyUI prefetch implementation.
        super().prefetch(
            index,
            device,
            dtype,
        )

    # --------------------------------------------------------
    # Read
    # --------------------------------------------------------

    def take(
        self,
        index,
        device,
        dtype,
        batch_size,
    ):

        tensor = super().take(
            index,
            device,
            dtype,
            batch_size,
        )

        log(
            f"TAKE block={index:02d} "
            f"device={tensor.device} "
            f"shape={tuple(tensor.shape)}"
        )

        return tensor

    # --------------------------------------------------------
    # Cache size
    # --------------------------------------------------------

    def _slot_bytes(self, slot):

        total = 0

        for index, tensor in slot["blocks"].items():

            total += tensor_bytes(tensor)

            params = slot["params"].get(index)

            total += tensor_bytes(params)

        return total

    def memory_bytes(self):

        gpu = 0
        cpu = 0

        for slot in self.slots:

            for index, tensor in slot["blocks"].items():

                size = tensor_bytes(tensor)

                params = slot["params"].get(index)

                size += tensor_bytes(params)

                if tensor.device.type == "cpu":

                    cpu += size

                else:

                    gpu += size

        return gpu, cpu

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def print_statistics(self):

        gpu, cpu = self.memory_bytes()

        total = gpu + cpu

        actual_ratio = (
            gpu / total * 100.0
            if total > 0 else 0.0
        )

        gpu_count = sum(
            1
            for item in self.block_report.values()
            if item["device"] != "cpu"
        )

        cpu_count = sum(
            1
            for item in self.block_report.values()
            if item["device"] == "cpu"
        )

        log("=" * 55)

        log("CACHE ALLOCATION COMPLETE")

        log(
            f"GPU blocks: {gpu_count}/{self.total_blocks}"
        )

        log(
            f"CPU blocks: {cpu_count}/{self.total_blocks}"
        )

        log(
            f"GPU actual: {format_bytes(gpu)}"
        )

        log(
            f"CPU actual: {format_bytes(cpu)}"
        )

        log(
            f"TOTAL: {format_bytes(total)}"
        )

        log(
            f"GPU actual byte ratio: {actual_ratio:.2f}%"
        )

        log(
            f"GPU requested block ratio: {self.gpu_ratio:.2f}%"
        )

        log("=" * 55)

    # --------------------------------------------------------
    # Release
    # --------------------------------------------------------

    def free(self):

        log("Releasing mixed cache")

        super().free()

        self.gpu_bytes = 0
        self.cpu_bytes = 0

        self.block_report = {}

        log("Mixed cache released")


# ============================================================
# Qwen21 select_prefix_cache patch
# ============================================================

if not hasattr(
    QwenImage21Transformer2DModel,
    "_mixed_cache_original_select",
):

    QwenImage21Transformer2DModel._mixed_cache_original_select = (
        QwenImage21Transformer2DModel.select_prefix_cache
    )


_original_select = (
    QwenImage21Transformer2DModel._mixed_cache_original_select
)


def mixed_select_prefix_cache(
    self,
    key,
    cache_bytes,
    device,
    options,
):

    # Native behavior when mixed mode is not selected.
    if not options.get("mixed", False):

        if isinstance(
            self.prefix_cache,
            MixedPoseBranchCache,
        ):

            self.prefix_cache.free()

            self.prefix_cache = None

        return _original_select(
            self,
            key,
            cache_bytes,
            device,
            options,
        )

    if options.get("device") == "off":

        return None, False

    dtype = options.get(
        "dtype",
        "default",
    )

    gpu_ratio = options.get(
        "gpu_ratio",
        50,
    )

    total_blocks = len(
        self.transformer_blocks
    )

    cache = self.prefix_cache

    # Rebuild cache when configuration changes.
    if isinstance(cache, MixedPoseBranchCache):

        if (
            cache.dtype != dtype
            or cache.gpu_ratio != float(gpu_ratio)
            or cache.compute_device != torch.device(device)
            or cache.total_blocks != total_blocks
        ):

            log("Cache configuration changed; rebuilding")

            cache.free()

            cache = None

            self.prefix_cache = None

    # Replace native cache if necessary.
    elif cache is not None:

        log("Replacing native cache with mixed cache")

        if hasattr(cache, "free"):

            cache.free()

        cache = None

        self.prefix_cache = None

    # Reuse an existing cache slot.
    if cache is not None:

        if cache.select(
            key,
            create=False,
        ):

            return (
                cache,
                cache.filled(total_blocks),
            )

    # Create new cache.
    if cache is None:

        cache = MixedPoseBranchCache(

            store_device=device,

            dtype=dtype,

            gpu_ratio=gpu_ratio,

            total_blocks=total_blocks,

        )

        self.prefix_cache = cache

    # Create or select slot.
    cache.select(
        key,
        create=True,
    )

    if cache.slot is None:

        log("ERROR: Failed to create cache slot")

        return None, False

    return cache, False


QwenImage21Transformer2DModel.select_prefix_cache = (
    mixed_select_prefix_cache
)


log("QwenImage21Transformer2DModel patched")


# ============================================================
# Custom Node
# ============================================================

class Qwen21MixedKVCache:

    @classmethod
    def INPUT_TYPES(cls):

        return {

            "required": {

                "model": (
                    "MODEL",
                ),

                "gpu_ratio": (

                    "INT",

                    {

                        "default": 50,

                        "min": 0,

                        "max": 100,

                        "step": 5,

                    },

                ),

                "dtype": (

                    [

                        "default",

                        "int8",

                        "int4",

                    ],

                    {

                        "default": "default",

                    },

                ),

            },

        }

    RETURN_TYPES = (
        "MODEL",
    )

    RETURN_NAMES = (
        "MODEL",
    )

    FUNCTION = "apply"

    CATEGORY = "model/conditioning/qwen image"

    DESCRIPTION = (
        "Qwen Image 2.1 GPU/CPU mixed KV cache. "
        "Strict Transformer Block count allocation."
    )

    def apply(
        self,
        model,
        gpu_ratio,
        dtype,
    ):

        m = model.clone()

        transformer_options = (
            m.model_options.setdefault(
                "transformer_options",
                {},
            )
        )

        options = transformer_options.setdefault(
            "qwen_image21_cache",
            {},
        )

        options.update({

            # Force the native Qwen21 cache path,
            # including dynamic VRAM mode.

            "device": "cpu",

            "dtype": dtype,

            "mixed": True,

            "gpu_ratio": gpu_ratio,

        })

        log(
            f"Node applied: "
            f"GPU blocks={gpu_ratio}% "
            f"dtype={dtype}"
        )

        return (
            m,
        )


# ============================================================
# Registration
# ============================================================

NODE_CLASS_MAPPINGS = {

    "Qwen21MixedKVCache": Qwen21MixedKVCache,

}

NODE_DISPLAY_NAME_MAPPINGS = {

    "Qwen21MixedKVCache": "Qwen21 Mixed KV Cache",

}


log(f"Loaded version {VERSION}")

log("Custom node registered successfully")
