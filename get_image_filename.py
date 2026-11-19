import os

class GetImageFilename:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "suffix": ("STRING", {"default": "_"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "get_filename"
    CATEGORY = "utils"

    def get_filename(self, image, suffix="_", prompt=None, unique_id=None):
        filename = ""
        try:
            if prompt and unique_id is not None:
                node = prompt.get(str(unique_id))
                if node:
                    link = node.get("inputs", {}).get("image")
                    if isinstance(link, list) and len(link) == 2:
                        src_node = prompt.get(str(link[0]))
                        if src_node:
                            raw = src_node.get("inputs", {}).get("image")
                            if isinstance(raw, str) and raw:
                                base = os.path.splitext(os.path.basename(raw))[0]
                                filename = base + suffix
        except Exception:
            pass
        return (filename,)

NODE_CLASS_MAPPINGS = {"GetImageFilename": GetImageFilename}
NODE_DISPLAY_NAME_MAPPINGS = {"GetImageFilename": "Get Image Filename"}