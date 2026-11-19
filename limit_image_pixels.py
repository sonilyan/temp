
import math
import comfy.utils


class LimitImagePixels:

    UPSCALE_METHODS = [
        "nearest-exact",
        "bilinear",
        "area",
        "bicubic",
        "lanczos",
    ]

    MODES = [
        "限制最大像素",
        "应用目标像素",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

                "max_pixels": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 100.0,
                    "step": 0.05,
                }),

                "mode": (cls.MODES, {
                    "default": "限制最大像素",
                }),

                "upscale_method": (
                    cls.UPSCALE_METHODS,
                    {"default": "lanczos"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "resized")

    FUNCTION = "process"
    CATEGORY = "image/transform"

    def process(self, image, max_pixels, mode, upscale_method):

        samples = image.movedim(-1, 1)

        height = samples.shape[2]
        width = samples.shape[3]

        current_pixels = width * height
        target_pixels = max_pixels * 1024 * 1024

        # 限制最大像素：只缩小，不放大
        if mode == "限制最大像素":
            if current_pixels <= target_pixels:
                return (image, False)

        # 应用目标像素：大于目标就缩小，小于目标就放大
        else:
            if current_pixels == target_pixels:
                return (image, False)

        # 根据目标像素计算缩放比例，保持宽高比
        scale_by = math.sqrt(target_pixels / current_pixels)

        new_width = max(1, round(width * scale_by))
        new_height = max(1, round(height * scale_by))

        # 避免舍入后尺寸没有变化
        if new_width == width and new_height == height:
            return (image, False)

        # 使用 ComfyUI 原生缩放函数，不裁剪
        result = comfy.utils.common_upscale(
            samples,
            new_width,
            new_height,
            upscale_method,
            "disabled",
        )

        result = result.movedim(1, -1)

        return (result, True)


NODE_CLASS_MAPPINGS = {
    "LimitImagePixels": LimitImagePixels
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LimitImagePixels": "限制/应用目标像素"
}
