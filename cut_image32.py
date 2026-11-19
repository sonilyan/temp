
class LimitImage:
    ALIGNMENT = 32

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "image/upscaling"
    DESCRIPTION = "Crop right and top edges to align dimensions to multiples of 32. No resizing."

    def execute(self, image):
        height, width = image.shape[1], image.shape[2]

        new_width = (width // self.ALIGNMENT) * self.ALIGNMENT
        new_height = (height // self.ALIGNMENT) * self.ALIGNMENT

        result = image[
            :,
            height - new_height:,
            :new_width,
            :
        ]

        return (result,)


NODE_CLASS_MAPPINGS = {
    "LimitImage": LimitImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LimitImage": "限制图像（32倍数裁剪）"
}
