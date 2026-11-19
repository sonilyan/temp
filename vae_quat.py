
import logging
import torch


NODE_NAME = "VAE Quad Attention"
LOGGER = logging.getLogger("VAE_QUAD_ATTENTION")

_CALL_LOGGED = False


def _get_quad_attention():
    """使用 ComfyUI 原生 Sub-Quadratic Attention。"""
    from comfy.ldm.modules.attention import attention_sub_quad
    return attention_sub_quad


def _vae_quad_attention(q, k, v):
    """
    Wan/Qwen VAE Attention 输入:
        [B, C, H, W]

    转换:
        [B, H*W, C]

    使用单头 Attention，保持原始 VAE 计算方式。
    """

    global _CALL_LOGGED

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise RuntimeError(
            "[VAE QUAD] Expected BCHW Q/K/V, "
            f"got q={tuple(q.shape)}, "
            f"k={tuple(k.shape)}, "
            f"v={tuple(v.shape)}"
        )

    if q.shape != k.shape or q.shape != v.shape:
        raise RuntimeError(
            "[VAE QUAD] Q/K/V shape mismatch"
        )

    original_shape = q.shape

    batch, channels, height, width = original_shape
    tokens = height * width

    # BCHW -> B(HW)C
    # 不主动 contiguous，避免额外复制整张 Q/K/V。
    q_seq = q.flatten(2).transpose(1, 2)
    k_seq = k.flatten(2).transpose(1, 2)
    v_seq = v.flatten(2).transpose(1, 2)

    if not _CALL_LOGGED:
        LOGGER.warning(
            "[VAE QUAD] Activated: "
            "shape=%s tokens=%d channels=%d "
            "dtype=%s device=%s",
            tuple(original_shape),
            tokens,
            channels,
            q.dtype,
            q.device,
        )
        _CALL_LOGGED = True

    attention_sub_quad = _get_quad_attention()

    # 单头，使用 ComfyUI 原生分块 Sub-Quadratic Attention。
    result = attention_sub_quad(
        q_seq,
        k_seq,
        v_seq,
        heads=1,
    )

    # B(HW)C -> BCHW
    result = result.transpose(1, 2).reshape(original_shape)

    return result


def _get_vae_model(vae):
    if vae is None:
        raise RuntimeError("[VAE QUAD] VAE input is None")

    model = getattr(vae, "first_stage_model", None)

    if model is None:
        raise RuntimeError(
            "[VAE QUAD] Cannot find vae.first_stage_model"
        )

    if not callable(getattr(model, "modules", None)):
        raise RuntimeError(
            "[VAE QUAD] first_stage_model is not a torch module"
        )

    return model


def _is_compatible_attention(module):
    """
    通过实际成员结构识别 Attention。

    支持:
      1. Wan/Qwen AttentionBlock:
         norm + to_qkv + proj + optimized_attention

      2. 标准 ComfyUI AttnBlock:
         norm + q + k + v + proj_out + optimized_attention

    不再依赖类名必须等于 AttnBlock。
    """

    if not callable(
        getattr(module, "optimized_attention", None)
    ):
        return False

    # Wan / Qwen VAE
    is_wan_attention = (
        hasattr(module, "norm")
        and hasattr(module, "to_qkv")
        and hasattr(module, "proj")
    )

    # 标准 ComfyUI VAE
    is_standard_attention = (
        hasattr(module, "norm")
        and hasattr(module, "q")
        and hasattr(module, "k")
        and hasattr(module, "v")
        and hasattr(module, "proj_out")
    )

    return is_wan_attention or is_standard_attention


def _find_vae_attention_modules(model):
    modules = []

    for module in model.modules():
        if _is_compatible_attention(module):
            modules.append(module)

    return modules


class VAEQuadAttention:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "mode": (
                    ["quad", "restore"],
                    {
                        "default": "quad"
                    }
                ),
            }
        }

    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("vae",)

    FUNCTION = "apply"
    CATEGORY = "advanced/vae"

    DESCRIPTION = (
        "Replace Qwen/Wan VAE Attention with "
        "ComfyUI native Sub-Quadratic Attention."
    )

    def apply(self, vae, mode):

        model = _get_vae_model(vae)

        attention_modules = _find_vae_attention_modules(model)

        if not attention_modules:
            # 输出实际模块信息，方便后续适配其他 VAE。
            candidates = []

            for module in model.modules():
                if hasattr(module, "optimized_attention"):
                    candidates.append(
                        module.__class__.__name__
                    )

            raise RuntimeError(
                "[VAE QUAD] No compatible Attention module found. "
                f"Modules with optimized_attention: {candidates}"
            )

        changed = 0
        restored = 0

        if mode == "quad":

            # 先检查原生实现是否可导入。
            _get_quad_attention()

            for module in attention_modules:

                # 保存原始 Attention，仅保存一次。
                if not hasattr(
                    module,
                    "_vae_quad_original_attention"
                ):
                    module._vae_quad_original_attention = (
                        module.optimized_attention
                    )

                module.optimized_attention = _vae_quad_attention

                changed += 1

            LOGGER.warning(
                "[VAE QUAD] Successfully patched %d Attention modules.",
                changed,
            )

            for module in attention_modules:
                LOGGER.warning(
                    "[VAE QUAD] Patched module: %s",
                    module.__class__.__name__,
                )

        elif mode == "restore":

            for module in attention_modules:

                original = getattr(
                    module,
                    "_vae_quad_original_attention",
                    None,
                )

                if original is None:
                    continue

                module.optimized_attention = original

                delattr(
                    module,
                    "_vae_quad_original_attention",
                )

                restored += 1

            LOGGER.warning(
                "[VAE QUAD] Restored %d Attention modules.",
                restored,
            )

        else:
            raise ValueError(
                f"[VAE QUAD] Unknown mode: {mode}"
            )

        return (vae,)


NODE_CLASS_MAPPINGS = {
    "VAEQuadAttention": VAEQuadAttention,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VAEQuadAttention": "VAE Quad Attention",
}
