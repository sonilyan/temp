
import types
import torch


# ============================================================
# Qwen Image 2.1 Fixed 9-Block VAE Decode
#
# 固定 3x3 九块
# 全局 Middle / Attention
# 九块串行解码
# 最后一层 CausalConv3d 分条计算
#
# 不使用 Tiled VAE blending
# 不改变模型权重
# 不修改 ComfyUI 核心
# ============================================================

NODE_NAME = "Qwen21Fixed4VAEDecode"

SPATIAL_SCALE = 16
HALO = 16

GRID_ROWS = 3
GRID_COLS = 3

# 最后一层 CausalConv3d 的内部行条带目标元素数
HEAD_STRIP_ELEMS = 2 ** 22


def _get_decoder(model):
    decoder = getattr(model, "decoder", None)

    if decoder is None:
        raise RuntimeError(
            "[Qwen21Fixed9] Cannot find model.decoder"
        )

    required = [
        "conv1",
        "middle",
        "upsamples",
        "head",
    ]

    for name in required:
        if not hasattr(decoder, name):
            raise RuntimeError(
                f"[Qwen21Fixed9] Decoder missing: {name}"
            )

    return decoder


def _strip_final_conv(layer, x):
    """
    对最后一层 CausalConv3d 按行分条计算。

    Qwen Image 解码使用单帧输入。
    空间卷积核为 3x3，stride=1，padding=1。
    每条输出只依赖本行及上下各一行输入。

    分条之间保留必要的输入 Halo，
    只复制有效中心行，不进行融合。
    """

    if x.ndim != 5 or x.shape[2] != 1:
        raise RuntimeError(
            "[Qwen21Fixed9] Expected [B,C,1,H,W], "
            f"got {tuple(x.shape)}"
        )

    kernel = tuple(getattr(layer, "kernel_size", (1, 3, 3)))
    dilation = tuple(getattr(layer, "dilation", (1, 1, 1)))
    stride = tuple(getattr(layer, "stride", (1, 1, 1)))
    padding = tuple(getattr(layer, "padding", (0, 1, 1)))

    if stride[-2:] != (1, 1):
        raise RuntimeError(
            f"[Qwen21Fixed9] Unsupported spatial stride: {stride}"
        )

    if kernel[-2] % 2 == 0:
        raise RuntimeError(
            "[Qwen21Fixed9] Even-height kernel is unsupported"
        )

    radius = (
        dilation[-2] * (kernel[-2] - 1) + 1
    ) // 2

    if padding[-2] != radius:
        raise RuntimeError(
            "[Qwen21Fixed9] Unexpected height padding: "
            f"expected {radius}, got {padding}"
        )

    b, _, _, height, width = x.shape

    elems_per_row = max(
        1,
        x.shape[0] * x.shape[1] * width
    )

    rows_per_strip = max(
        1,
        HEAD_STRIP_ELEMS // elems_per_row
    )

    rows_per_strip = min(height, rows_per_strip)

    out = None

    for y0 in range(0, height, rows_per_strip):

        y1 = min(height, y0 + rows_per_strip)

        sy0 = max(0, y0 - radius)
        sy1 = min(height, y1 + radius)

        strip_in = x[:, :, :, sy0:sy1, :]

        strip_out = layer(strip_in)

        if (
            strip_out.shape[-2] != strip_in.shape[-2]
            or strip_out.shape[-1] != strip_in.shape[-1]
        ):
            raise RuntimeError(
                "[Qwen21Fixed9] Strip convolution changed "
                "spatial dimensions: "
                f"{tuple(strip_in.shape[-2:])} -> "
                f"{tuple(strip_out.shape[-2:])}"
            )

        cy0 = y0 - sy0
        cy1 = cy0 + (y1 - y0)

        valid = strip_out[:, :, :, cy0:cy1, :]

        if out is None:
            out = torch.empty(
                (
                    b,
                    strip_out.shape[1],
                    1,
                    height,
                    strip_out.shape[-1],
                ),
                device=strip_out.device,
                dtype=strip_out.dtype,
            )

        out[:, :, :, y0:y1, :].copy_(valid)

        del strip_in
        del strip_out
        del valid

    if out is None:
        raise RuntimeError(
            "[Qwen21Fixed9] No convolution strips were processed"
        )

    return out


def _run_middle(decoder, x):
    """
    Middle 必须在完整特征图上执行。
    Attention 保持全局计算，不进行切块。
    """

    for layer in decoder.middle:
        x = layer(x)

    return x


def _run_decoder_tile(decoder, x):
    """
    对单个 tile 执行完整 Upsample 和 Head。
    """

    feat_idx = [0]

    for layer in decoder.upsamples:

        x = layer(
            x,
            feat_cache=None,
            feat_idx=feat_idx,
            first_chunk=True,
        )

    for index, layer in enumerate(decoder.head):

        if (
            index == len(decoder.head) - 1
            and hasattr(layer, "_padding")
            and hasattr(layer, "kernel_size")
        ):
            y = _strip_final_conv(layer, x)
            del x
            x = y

        else:
            x = layer(x)

    return x


def _fixed9_decode(model, z):
    """
    替代 WanVAE.decode 的内部解码函数。

    输入:
        z: [B, C, H, W]

    输出:
        [B, C, H*16, W*16]
    """

    if z.ndim != 4:
        raise RuntimeError(
            f"[Qwen21Fixed9] Expected 4D latent, got {z.shape}"
        )

    decoder = _get_decoder(model)

    if len(decoder.upsamples) != 5:
        raise RuntimeError(
            "[Qwen21Fixed9] Expected 5 decoder upsample stages, "
            f"got {len(decoder.upsamples)}"
        )

    b, c, h, w = z.shape

    if h < 2 or w < 2:
        raise RuntimeError(
            "[Qwen21Fixed9] Latent dimensions are too small"
        )

    # --------------------------------------------------------
    # 1. Latent -> Decoder input
    # --------------------------------------------------------

    x = model.conv2(z.unsqueeze(2))

    # --------------------------------------------------------
    # 2. Decoder initial convolution
    # --------------------------------------------------------

    x = decoder.conv1(x)

    if x.ndim != 5 or x.shape[2] != 1:
        raise RuntimeError(
            "[Qwen21Fixed9] Expected single-frame decoder feature, "
            f"got {tuple(x.shape)}"
        )

    # --------------------------------------------------------
    # 3. Global Middle / Attention
    # --------------------------------------------------------

    x = _run_middle(decoder, x)

    # --------------------------------------------------------
    # 4. 固定 3x3 九块
    # --------------------------------------------------------

    _, _, _, feature_h, feature_w = x.shape

    y_edges = [
        (feature_h * i) // GRID_ROWS
        for i in range(GRID_ROWS + 1)
    ]

    x_edges = [
        (feature_w * i) // GRID_COLS
        for i in range(GRID_COLS + 1)
    ]

    cores = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):

            cores.append(
                (
                    y_edges[row],
                    y_edges[row + 1],
                    x_edges[col],
                    x_edges[col + 1],
                )
            )

    output = None

    # --------------------------------------------------------
    # 5. 串行解码九块
    # --------------------------------------------------------

    for y0, y1, x0, x1 in cores:

        # 扩展 Halo，保留卷积边缘上下文。

        hy0 = max(0, y0 - HALO)
        hy1 = min(feature_h, y1 + HALO)

        hx0 = max(0, x0 - HALO)
        hx1 = min(feature_w, x1 + HALO)

        tile = x[
            :,
            :,
            :,
            hy0:hy1,
            hx0:hx1,
        ].contiguous()

        # 当前 tile 完整解码。

        tile_out = _run_decoder_tile(
            decoder,
            tile,
        )

        if tile_out.ndim != 5 or tile_out.shape[2] != 1:
            raise RuntimeError(
                "[Qwen21Fixed9] Unexpected tile output shape: "
                f"{tuple(tile_out.shape)}"
            )

        # ----------------------------------------------------
        # 6. 计算当前核心区域在输出中的位置
        # ----------------------------------------------------

        crop_y0 = (y0 - hy0) * SPATIAL_SCALE
        crop_y1 = crop_y0 + (y1 - y0) * SPATIAL_SCALE

        crop_x0 = (x0 - hx0) * SPATIAL_SCALE
        crop_x1 = crop_x0 + (x1 - x0) * SPATIAL_SCALE

        tile_core = tile_out[
            :,
            :,
            :,
            crop_y0:crop_y1,
            crop_x0:crop_x1,
        ]

        # ----------------------------------------------------
        # 7. 创建最终输出缓冲区
        # ----------------------------------------------------

        if output is None:

            output = torch.empty(
                (
                    b,
                    tile_out.shape[1],
                    1,
                    feature_h * SPATIAL_SCALE,
                    feature_w * SPATIAL_SCALE,
                ),
                device=tile_out.device,
                dtype=tile_out.dtype,
            )

        # ----------------------------------------------------
        # 8. 写入当前核心区域
        #
        # 不做融合。
        # 不做重叠混合。
        # ----------------------------------------------------

        oy0 = y0 * SPATIAL_SCALE
        oy1 = y1 * SPATIAL_SCALE

        ox0 = x0 * SPATIAL_SCALE
        ox1 = x1 * SPATIAL_SCALE

        output[
            :,
            :,
            :,
            oy0:oy1,
            ox0:ox1,
        ].copy_(tile_core)

        # 当前块完成后释放引用。

        del tile
        del tile_out
        del tile_core

    # --------------------------------------------------------
    # 9. 输出
    # --------------------------------------------------------

    if output is None:
        raise RuntimeError(
            "[Qwen21Fixed9] No output was generated"
        )

    return output.squeeze(2)


# ============================================================
# ComfyUI Node
# ============================================================

class Qwen21Fixed4VAEDecode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "samples": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent/advanced"

    DESCRIPTION = (
        "Qwen Image 2.1 fixed 3x3 VAE decode. "
        "Global attention followed by serial nine-block decoding."
    )

    def decode(self, vae, samples):

        if not hasattr(vae, "first_stage_model"):
            raise RuntimeError(
                "[Qwen21Fixed9] Invalid VAE object"
            )

        model = vae.first_stage_model

        if not hasattr(model, "conv2"):
            raise RuntimeError(
                "[Qwen21Fixed9] This does not look like "
                "the Qwen/Wan VAE implementation"
            )

        if not hasattr(model, "decode"):
            raise RuntimeError(
                "[Qwen21Fixed9] VAE model has no decode method"
            )

        if "samples" not in samples:
            raise RuntimeError(
                "[Qwen21Fixed9] LATENT has no samples"
            )

        z = samples["samples"]

        # 保存原始 decode 方法。
        # 只在本次节点执行期间替换。

        had_instance_decode = "decode" in model.__dict__

        old_instance_decode = model.__dict__.get("decode")

        def replacement_decode(_self, latent, *args, **kwargs):
            return _fixed9_decode(
                _self,
                latent,
            )

        try:

            model.decode = types.MethodType(
                replacement_decode,
                model,
            )

            # 通过 ComfyUI VAE wrapper 解码，
            # 保留原本的输入输出处理流程。

            image = vae.decode(z)

        finally:

            # 恢复原始方法。

            if had_instance_decode:
                model.decode = old_instance_decode
            else:
                delattr(model, "decode")

        return (image,)


# ============================================================
# Registration
# ============================================================

NODE_CLASS_MAPPINGS = {
    "Qwen21Fixed4VAEDecode": Qwen21Fixed4VAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen21Fixed4VAEDecode": "Qwen 2.1 VAE Decode (Fixed 3x3 / 9 Blocks)",
}
