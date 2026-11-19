"""
Qwen Image 2.1 Super-Resolution V3 for ComfyUI.

V3:
  - Vision and VAE use the same aligned reference image dimensions.
  - No reference resizing.
  - If dimensions are not multiples of 32, crop TOP and RIGHT only.
  - Optional mild texture suppression on Vision input.
  - Target output aspect ratio follows the aligned reference image.
  - Viggle Turbo sigma schedules (6/9/10/11/12 steps).

No face detection, FaceID, landmark processing, or local face constraints.

Install:
  ComfyUI/custom_nodes/qwen21_super_resolution_v3.py
"""

import math

import torch
import torch.nn.functional as F

import comfy.model_management
import node_helpers


def _round_to_multiple(value, multiple=32):
    """Round to nearest multiple. Used by target output sizing."""
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _crop_reference_to_32(image, multiple=32):
    """
    Align reference image dimensions by cropping only.

    Keep the bottom-left corner:
      - crop excess pixels from TOP;
      - crop excess pixels from RIGHT.

    No resizing, no interpolation, no aspect ratio distortion.

    IMAGE layout: BHWC

    Returns:
        reference_image,
        source_width,
        source_height,
        reference_width,
        reference_height,
        crop_top,
        crop_right
    """
    if image.ndim != 4:
        raise ValueError(
            f"[Qwen21SR V3] Expected BHWC reference image, got {tuple(image.shape)}"
        )

    src_h = int(image.shape[1])
    src_w = int(image.shape[2])

    if src_w < multiple or src_h < multiple:
        raise ValueError(
            f"[Qwen21SR V3] Reference image {src_w}x{src_h} is smaller "
            f"than the required minimum {multiple}x{multiple}."
        )

    ref_w = (src_w // multiple) * multiple
    ref_h = (src_h // multiple) * multiple

    crop_right = src_w - ref_w
    crop_top = src_h - ref_h

    if crop_top == 0 and crop_right == 0:
        reference_image = image
    else:
        reference_image = image[
            :,
            crop_top:crop_top + ref_h,
            :ref_w,
            :
        ]

    return (
        reference_image,
        src_w,
        src_h,
        ref_w,
        ref_h,
        crop_top,
        crop_right,
    )


def _suppress_vision_texture(image, strength):
    """
    Mild full-image low-pass blending for the Qwen3-VL input only.

    image:
        BHWC RGB tensor, normally in [0, 1].

    strength:
        0.0 = unchanged input
        0.5 = maximum supported suppression

    Uses a small separable 5x5 Gaussian kernel.
    This does not modify the VAE reference image or its latent.
    """
    strength = float(strength)

    if strength <= 0.0:
        return image

    if strength > 0.5:
        raise ValueError(
            "[Qwen21SR V3] Texture suppression strength must be between 0.0 and 0.5."
        )

    if image.ndim != 4:
        raise ValueError(
            f"[Qwen21SR V3] Expected BHWC image, got {tuple(image.shape)}"
        )

    channels = int(image.shape[-1])
    if channels != 3:
        raise ValueError(
            f"[Qwen21SR V3] Texture suppression expects RGB, got {channels} channels."
        )

    x = image.movedim(-1, 1)

    k = torch.tensor(
        [1.0, 4.0, 6.0, 4.0, 1.0],
        device=x.device,
        dtype=x.dtype,
    ) / 16.0

    kernel_2d = k[:, None] * k[None, :]
    kernel = kernel_2d.view(1, 1, 5, 5)
    kernel = kernel.expand(channels, 1, 5, 5).contiguous()

    padded = F.pad(x, (2, 2, 2, 2), mode="reflect")

    blurred = F.conv2d(
        padded,
        kernel,
        stride=1,
        padding=0,
        groups=channels,
    )

    result = x * (1.0 - strength) + blurred * strength

    return result.movedim(1, -1)


class Qwen21SRV3Encode:
    """
    Qwen Image 2.1 V3 reference-conditioned reconstruction encoder.

    Reference image handling:
      - No resizing.
      - Crop TOP and RIGHT only when dimensions are not multiples of 32.
      - Vision and VAE use the same aligned reference dimensions.

    Vision preprocessing:
      1. Use the aligned reference image directly.
      2. Composite alpha over white.
      3. Apply optional mild full-image texture suppression.
      4. Pass RGB image to CLIP/Qwen3-VL.

    VAE preprocessing:
      - Use the aligned reference image directly.
      - No texture suppression.
    """

    DEFAULT_PROMPT = (
        "Create a faithful high-resolution reconstruction of the reference image. "
        "Preserve the exact subject identity, face, age appearance, hairstyle, "
        "expression, pose, body proportions, clothing, accessories, objects, "
        "camera angle, framing, perspective, composition, colors, lighting, "
        "and background. Do not redesign, replace, add, or remove anything. "
        "Reconstruct plausible fine details that are missing or blurred, including "
        "natural skin texture, individual hair strands, fabric weave, seams, "
        "small surface textures, clean edges, and realistic fine structures. "
        "Keep all shapes, positions, and proportions consistent with the reference. "
        "The result should look like the same photograph captured at higher resolution."
    )

    DEFAULT_NEGATIVE = (
        "different person, changed identity, changed face, altered expression, "
        "different hairstyle, changed pose, changed body proportions, changed clothing, "
        "new objects, missing objects, changed composition, different camera angle, "
        "changed background, distorted anatomy, extra fingers, missing fingers, "
        "warped hands, plastic skin, waxy skin, excessive sharpening, halos, "
        "invented patterns, painterly, illustration, cartoon, CGI, text, watermark"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),

                "target_megapixels": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1,
                        "max": 8,
                        "step": 0.5,
                        "round": 0.01,
                        "tooltip": (
                            "Target output area in megapixels. "
                            "Aspect ratio follows the aligned reference image."
                        ),
                    },
                ),

                "vision_texture_suppression": (
                    "FLOAT",
                    {
                        "default": 0.20,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": (
                            "Mild full-image low-pass blending on the Vision input only. "
                            "0 = disabled. VAE reference image is not modified."
                        ),
                    },
                ),

                "prompt": (
                    "STRING",
                    {
                        "default": cls.DEFAULT_PROMPT,
                        "multiline": True,
                    },
                ),

                "negative_prompt": (
                    "STRING",
                    {
                        "default": cls.DEFAULT_NEGATIVE,
                        "multiline": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"
    CATEGORY = "conditioning/qwen_image_2.1/V3"

    def encode(
        self,
        clip,
        vae,
        image,
        target_megapixels=2.0,
        vision_texture_suppression=0.20,
        prompt=DEFAULT_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE,
    ):
        if image is None or image.ndim != 4 or image.shape[0] < 1:
            raise ValueError(
                "[Qwen21SR V3] Input image must be a non-empty ComfyUI IMAGE tensor."
            )

        if not 0.0 <= float(vision_texture_suppression) <= 0.5:
            raise ValueError(
                "[Qwen21SR V3] vision_texture_suppression must be between 0.0 and 0.5."
            )

        if float(target_megapixels) <= 0:
            raise ValueError(
                "[Qwen21SR V3] target_megapixels must be greater than zero."
            )

        # Use one source image.
        source = image[:1]

        # ---------------------------------------------------------
        # 1. Align reference image by cropping only.
        #    No resize, no interpolation.
        #    Keep bottom-left; remove top and right excess pixels.
        # ---------------------------------------------------------
        (
            reference_image,
            src_w,
            src_h,
            ref_w,
            ref_h,
            crop_top,
            crop_right,
        ) = _crop_reference_to_32(
            source,
            multiple=32,
        )

        # ---------------------------------------------------------
        # 2. Target resolution.
        #    Aspect ratio follows the actual aligned reference image.
        # ---------------------------------------------------------
        aspect = ref_w / float(ref_h)
        target_area = float(target_megapixels) * 1_000_000.0

        target_w = _round_to_multiple(
            math.sqrt(target_area * aspect), 32
        )
        target_h = _round_to_multiple(
            math.sqrt(target_area / aspect), 32
        )

        # ---------------------------------------------------------
        # 3. Vision image.
        #    No independent resize.
        #    Vision and VAE use the same aligned reference dimensions.
        # ---------------------------------------------------------
        vision_image = reference_image

        # Match original behavior: Qwen3-VL receives RGB
        # composited over white.
        vision_rgb = vision_image[:, :, :, :3]

        if vision_image.shape[-1] > 3:
            alpha = vision_image[:, :, :, 3:4]
            vision_rgb = vision_rgb * alpha + (1.0 - alpha)

        # ---------------------------------------------------------
        # 4. V3 texture suppression.
        #    Only modifies Vision pixels, not dimensions.
        #    VAE reference remains untouched.
        # ---------------------------------------------------------
        vision_rgb = _suppress_vision_texture(
            vision_rgb,
            float(vision_texture_suppression),
        )

        # ---------------------------------------------------------
        # 5. VAE reference latent.
        #    Uses the original aligned image, without resizing.
        # ---------------------------------------------------------
        reference_latents = vae.encode(reference_image)

        if reference_latents.ndim != 4 or reference_latents.shape[1] != 64:
            raise RuntimeError(
                "[Qwen21SR V3] The connected VAE did not produce a "
                "Qwen 2.1 64-channel latent; "
                f"got {tuple(reference_latents.shape)}. "
                "Check that this is the Qwen Image 2.1 VAE."
            )

        # ---------------------------------------------------------
        # 6. Qwen Image 2.1 CLIP encoding.
        # ---------------------------------------------------------
        tokens_positive = clip.tokenize(
            prompt,
            images=[vision_rgb],
            keep_vision=False,
            prevent_empty_text=True,
        )

        tokens_negative = clip.tokenize(
            negative_prompt,
            images=[vision_rgb],
            keep_vision=False,
            prevent_empty_text=True,
        )

        positive = clip.encode_from_tokens_scheduled(
            tokens_positive
        )

        negative = clip.encode_from_tokens_scheduled(
            tokens_negative
        )

        # Attach the VAE reference latent exactly once.
        positive = node_helpers.conditioning_set_values(
            positive,
            {"reference_latents": [reference_latents]},
            append=True,
        )

        negative = node_helpers.conditioning_set_values(
            negative,
            {"reference_latents": [reference_latents]},
            append=True,
        )

        # ---------------------------------------------------------
        # 7. Target latent.
        #    Preserve original V1 target latent calculation.
        # ---------------------------------------------------------
        latent_h = target_h // 16
        latent_w = target_w // 16

        target_latent = torch.zeros(
            [1, 64, latent_h, latent_w],
            device=comfy.model_management.intermediate_device(),
        )

        # ---------------------------------------------------------
        # 8. Diagnostics.
        # ---------------------------------------------------------
        print(
            "[Qwen21SR V3 ENCODE] "
            f"source={src_w}x{src_h} "
            f"({src_w * src_h / 1_000_000:.3f}MP), "
            f"reference_vae={ref_w}x{ref_h}, "
            f"crop_top={crop_top}, "
            f"crop_right={crop_right}, "
            f"reference_resize=OFF, "
            f"vision_resize=OFF, "
            f"vision_input={ref_w}x{ref_h}, "
            f"reference_latent={tuple(reference_latents.shape)}, "
            f"texture_suppression={float(vision_texture_suppression):.3f}, "
            f"target={target_w}x{target_h} "
            f"({target_w * target_h / 1_000_000:.3f}MP), "
            f"target_latent={tuple(target_latent.shape)}, "
            f"target_tokens={latent_h * latent_w}",
            flush=True,
        )

        if image.shape[0] > 1:
            print(
                f"[Qwen21SR V3 ENCODE] WARNING: input batch has "
                f"{image.shape[0]} images; only the first image is used.",
                flush=True,
            )

        return (
            positive,
            negative,
            {"samples": target_latent},
        )


class Qwen21SRV3Sigmas:
    """
    Viggle Turbo v0.3 sigma-node schedules.

    6 steps:  official Viggle v0.3 six-node schedule.
    9 steps:  official Viggle v0.3 schedule (7 Turbo + 2 Base nodes).
    10 steps: existing experimental Viggle schedule (7 Turbo + 3 Base nodes).
    11 steps: experimental Viggle schedule (7 Turbo + 4 Base nodes).
    12 steps: experimental Viggle schedule (7 Turbo + 5 Base nodes).

    Resolution-dependent dynamic shift is retained.
    No Qwen Base terminal stretching is applied.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "steps": (
                    ["6", "10","10_2","12","14"],
                    {"default": "10"},
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers/V3"

    BASE_IMAGE_SEQ_LEN = 256
    BASE_SHIFT = 0.5
    MAX_IMAGE_SEQ_LEN = 8192
    MAX_SHIFT = 0.9

    SIX_STEP_NODES = (
        1.0,
        0.9375,
        0.875,
        0.75,
        0.5,
        0.25,
    )

    TEN_STEP_NODES = (
        1.0,
        23.0 / 24.0,
        22.0 / 24.0,
        0.875,
        0.75,
        0.5,
        0.25,
        1.0 / 6.0,
        1.0 / 11.0,
        1.0 / 20.0,
    )

    TEN2_STEP_NODES = (
        1.0,
        0.9375,
        0.875,
        19.5 / 24.0,
        0.75,
        0.625,
        0.5,
        0.25,
        1.0 / 11.0,
        1.0 / 20.0,
    )
    
    TA_STEP_NODES = (
        1.0,
        23.0 / 24.0,
        22.0 / 24.0,
        0.875,
        20.0 / 24.0,
        19.0 / 24.0,
        0.75,
        0.5,
        0.25,
        1.0 / 6.0,
        1.0 / 11.0,
        1.0 / 20.0,
    )
    
    TB_STEP_NODES = (
        1.0,
        23.0 / 24.0,
        22.0 / 24.0,
        0.875,
        20.0 / 24.0,
        19.0 / 24.0,
        0.75,
        0.625,
        0.5,
        0.375,
        0.25,
        1.0 / 6.0,
        1.0 / 11.0,
        1.0 / 20.0,
    )

    def get_sigmas(self, latent, steps="10"):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError(
                "[Qwen21SR V3 SIGMAS] A valid Qwen 2.1 target LATENT is required."
            )

        samples = latent["samples"]

        if samples.ndim != 4 or samples.shape[1] != 64:
            raise ValueError(
                "[Qwen21SR V3 SIGMAS] Expected Qwen Image 2.1 latent "
                f"shape [B, 64, H, W], got {tuple(samples.shape)}."
            )

        if steps == "6":
            nodes = self.SIX_STEP_NODES
        elif steps == "10":
            nodes = self.TEN_STEP_NODES
        elif steps == "10_2":
            nodes = self.TEN2_STEP_NODES
        elif steps == "12":
            nodes = self.TA_STEP_NODES
        elif steps == "14":
            nodes = self.TB_STEP_NODES
        else:
            raise ValueError(
                f"[Qwen21SR V3 SIGMAS] Unsupported step count: {steps}"
            )

        latent_h = int(samples.shape[-2])
        latent_w = int(samples.shape[-1])

        # V3 target latents do not carry downscale_ratio_spacial.
        # Viggle scheduler default ratio is 1.0.
        spatial_ratio = float(
            latent.get("downscale_ratio_spacial", 16)
        ) / 16.0

        token_h = round(latent_h * spatial_ratio)
        token_w = round(latent_w * spatial_ratio)
        image_seq_len = token_h * token_w

        mu = self.BASE_SHIFT + (
            (self.MAX_SHIFT - self.BASE_SHIFT)
            * (image_seq_len - self.BASE_IMAGE_SEQ_LEN)
            / (self.MAX_IMAGE_SEQ_LEN - self.BASE_IMAGE_SEQ_LEN)
        )

        # Apply resolution-dependent FlowMatch shift.
        # No Qwen Base terminal stretching.
        selected_nodes = torch.tensor(
            nodes,
            dtype=torch.float64,
            device="cpu",
        )

        exp_mu = math.exp(mu)

        sigmas = exp_mu / (
            exp_mu + (1.0 / selected_nodes - 1.0)
        )

        sigmas = torch.cat(
            [sigmas, sigmas.new_zeros(1)]
        ).to(torch.float32)

        if not torch.isfinite(sigmas).all():
            raise RuntimeError(
                "[Qwen21SR V3 SIGMAS] Generated non-finite sigma values."
            )

        if torch.any(sigmas[:-1] <= sigmas[1:]):
            raise RuntimeError(
                "[Qwen21SR V3 SIGMAS] Sigma schedule is not strictly descending: "
                f"{sigmas.tolist()}"
            )

        print(
            "[Qwen21SR V3 SIGMAS] "
            f"mode=Viggle-{steps}, "
            f"latent={latent_w}x{latent_h}, "
            f"image_seq_len={image_seq_len}, "
            f"mu={mu:.7f}, "
            f"terminal_stretch=OFF, "
            f"selected_t={[round(float(v), 8) for v in selected_nodes.tolist()]}, "
            f"sigmas={[round(float(v), 8) for v in sigmas.tolist()]}",
            flush=True,
        )

        return (sigmas,)


NODE_CLASS_MAPPINGS = {
    "Qwen21SRV3Encode": Qwen21SRV3Encode,
    "Qwen21SRV3Sigmas": Qwen21SRV3Sigmas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen21SRV3Encode": (
        "Qwen 2.1 Super-Resolution V3 Encode "
        "(No Reference Resize)"
    ),
    "Qwen21SRV3Sigmas": (
        "Qwen 2.1 V3 Viggle Sigma Schedule (6/9/10/11/12)"
    ),
}