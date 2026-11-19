"""
Qwen-Image 2.1 TE FP16 runtime patch for ComfyUI.

Install:
    Copy this file to:
    ComfyUI/custom_nodes/Qwen21_FP16_Patch.py

This patch is intentionally limited to the built-in Qwen-Image 2.1 TE
(Qwen3-VL 8B). It does not change other text encoders, the DiT, or the VAE.

What it changes:
  1. Forces the Qwen-Image 2.1 TE module construction dtype to FP16.
  2. Overrides the Qwen-Image 2.1 ModelPatcher compute dtype to FP16.
  3. Uses FP16 for token, image, and padding embeddings in process_tokens().
  4. Feeds FP16 pixels into the Qwen3-VL vision tower.
  5. Casts interpolated vision positional embeddings and vision RoPE to FP16.
  6. Casts text MRoPE frequencies to FP16 before the Q/K rotary operation.
  7. Adds targeted dtype trace hooks inside the Qwen-Image 2.1 vision/text towers.

The existing final conditioning conversion to FP32 is deliberately preserved.
Quantization metadata is passed through unchanged.
"""

import logging
import numbers
import torch

_LOG = logging.getLogger("Qwen21_FP16_Patch")
_PATCH_NAME = "Qwen-Image 2.1 TE FP16"
_PATCHED = False


def _is_qwen21_te(model, qwen21_module):
    """Identify only the Qwen-Image 2.1 TE wrapper/model."""
    try:
        return isinstance(model, qwen21_module.QwenImage21TEModel)
    except Exception:
        # Fallback for unusual import/reload arrangements.
        for cls in type(model).__mro__:
            if cls.__name__.startswith("QwenImage21TEModel"):
                return True
    return False


def _cast_nested_fp16(value):
    """Cast rotary tensors while preserving the tuple/list structure."""
    if torch.is_tensor(value):
        return value.to(dtype=torch.float16)
    if isinstance(value, tuple):
        return tuple(_cast_nested_fp16(item) for item in value)
    if isinstance(value, list):
        return [_cast_nested_fp16(item) for item in value]
    return value


def _install_vision_dtype_wrappers(qwen35_module):
    """Keep Qwen3-VL vision positional additions and RoPE in FP16."""
    for method_name in ("fast_pos_embed_interpolate", "rot_pos_emb"):
        original = getattr(qwen35_module.Qwen35VisionModel, method_name, None)
        if original is None or getattr(original, "_qwen21_fp16_wrapped", False):
            continue

        def make_wrapper(original_method, name):
            def wrapped(self, *args, **kwargs):
                result = original_method(self, *args, **kwargs)
                if getattr(self, "_qwen21_fp16_patch", False) and torch.is_tensor(result):
                    result = result.to(dtype=getattr(self, "_qwen21_fp16_dtype", torch.float16))
                    logged_attr = f"_qwen21_{name}_logged"
                    if not getattr(self, logged_attr, False):
                        _LOG.info(
                            "[Qwen21 FP16] vision %s output=%s",
                            name,
                            result.dtype,
                        )
                        setattr(self, logged_attr, True)
                return result

            wrapped._qwen21_fp16_wrapped = True
            wrapped._qwen21_fp16_original = original_method
            return wrapped

        setattr(qwen35_module.Qwen35VisionModel, method_name, make_wrapper(original, method_name))


def _install_vision_dtype_hooks(visual):
    """Log the actual dtype at key boundaries inside this Qwen21 vision tower."""
    if getattr(visual, "_qwen21_fp16_hooks_installed", False):
        return

    def make_hook(name):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) and output else output
            if torch.is_tensor(value):
                _LOG.info("[Qwen21 FP16 TRACE] vision.%s output=%s", name, value.dtype)
        return hook

    targets = []
    if getattr(visual, "patch_embed", None) is not None:
        targets.append(("patch_embed", visual.patch_embed))
    blocks = getattr(visual, "blocks", None)
    if blocks:
        targets.append(("blocks.0", blocks[0]))
        if len(blocks) > 1:
            targets.append((f"blocks.{len(blocks) // 2}", blocks[len(blocks) // 2]))
        if len(blocks) > 2:
            targets.append((f"blocks.{len(blocks) - 1}", blocks[-1]))
    if getattr(visual, "merger", None) is not None:
        targets.append(("merger", visual.merger))
    for name, module in targets:
        module.register_forward_hook(make_hook(name))

    for index, module in enumerate(getattr(visual, "deepstack_merger_list", []) or []):
        module.register_forward_hook(make_hook(f"deepstack_merger.{index}"))

    visual._qwen21_fp16_hooks_installed = True


def _make_fp16_process_tokens(qwen21_module):
    """
    Same token/image embedding assembly as SDClipModel.process_tokens(),
    but all assembled embeddings remain FP16 for Qwen-Image 2.1 only.
    """
    def process_tokens_fp16(self, tokens, device):
        end_token = self.special_tokens.get("end", None)
        pad_token = self.special_tokens.get("pad", -1)
        cmp_token = pad_token if end_token is None else end_token

        embeds_out = []
        attention_masks = []
        num_tokens = []
        target_dtype = torch.float16

        for x in tokens:
            attention_mask = []
            tokens_temp = []
            other_embeds = []
            eos = False
            index = 0
            left_pad = False

            for y in x:
                if isinstance(y, numbers.Integral):
                    token = int(y)
                    if index == 0 and token == pad_token:
                        left_pad = True

                    if eos or (left_pad and token == pad_token):
                        attention_mask.append(0)
                    else:
                        attention_mask.append(1)
                        left_pad = False

                    tokens_temp.append(token)
                    if not eos and token == cmp_token and not left_pad:
                        if end_token is None:
                            attention_mask[-1] = 0
                        eos = True
                else:
                    other_embeds.append((index, y))
                index += 1

            tokens_embed = torch.tensor(
                [tokens_temp], device=device, dtype=torch.long
            )
            tokens_embed = self.transformer.get_input_embeddings()(
                tokens_embed, out_dtype=target_dtype
            )

            index = 0
            pad_extra = 0
            embeds_info = []

            for o in other_embeds:
                emb = o[1]
                if torch.is_tensor(emb):
                    emb = {"type": "embedding", "data": emb}

                extra = None
                emb_type = emb.get("type", None)

                if emb_type == "embedding":
                    emb = emb.get("data", None)
                else:
                    if hasattr(self.transformer, "preprocess_embed"):
                        emb, extra = self.transformer.preprocess_embed(
                            emb, device=device
                        )
                    else:
                        emb = None

                if emb is None:
                    index -= 1
                    continue

                ind = index + o[0]
                emb = emb.view(1, -1, emb.shape[-1]).to(
                    device=device, dtype=target_dtype
                )
                emb_shape = emb.shape[1]

                if emb.shape[-1] == tokens_embed.shape[-1]:
                    tokens_embed = torch.cat(
                        [tokens_embed[:, :ind], emb, tokens_embed[:, ind:]],
                        dim=1,
                    )
                    attention_mask = (
                        attention_mask[:ind]
                        + [1] * emb_shape
                        + attention_mask[ind:]
                    )
                    index += emb_shape - 1
                    embeds_info.append(
                        {
                            "type": emb_type,
                            "index": ind,
                            "size": emb_shape,
                            "extra": extra,
                        }
                    )
                else:
                    index -= 1
                    pad_extra += emb_shape
                    _LOG.warning(
                        "Qwen21 FP16 patch: embedding shape mismatch; "
                        "embedding ignored (%s != %s)",
                        emb.shape[-1],
                        tokens_embed.shape[-1],
                    )

            if pad_extra > 0:
                padd_embed = self.transformer.get_input_embeddings()(
                    torch.tensor(
                        [[self.special_tokens["pad"]] * pad_extra],
                        device=device,
                        dtype=torch.long,
                    ),
                    out_dtype=target_dtype,
                )
                tokens_embed = torch.cat([tokens_embed, padd_embed], dim=1)
                attention_mask = attention_mask + [0] * pad_extra

            embeds_out.append(tokens_embed)
            attention_masks.append(attention_mask)
            num_tokens.append(sum(attention_mask))

        if not getattr(self, "_qwen21_fp16_process_logged", False):
            _LOG.info(
                "[Qwen21 FP16] process_tokens: token/image/pad embeddings=%s",
                target_dtype,
            )
            self._qwen21_fp16_process_logged = True

        result_embeds = torch.cat(embeds_out)
        if not getattr(self, "_qwen21_fp16_final_embed_logged", False):
            _LOG.info("[Qwen21 FP16] assembled transformer input=%s", result_embeds.dtype)
            self._qwen21_fp16_final_embed_logged = True

        return (
            result_embeds,
            torch.tensor(attention_masks, device=device, dtype=torch.long),
            num_tokens,
            embeds_info,
        )

    return process_tokens_fp16


def _install():
    global _PATCHED
    if _PATCHED:
        return

    # Import only ComfyUI's built-in modules. No third-party dependency.
    import comfy.sd1_clip as sd1_clip
    import comfy.model_patcher as model_patcher
    import comfy.text_encoders.qwen3vl as qwen3vl
    import comfy.text_encoders.qwen35 as qwen35
    import comfy.text_encoders.qwen_vl as qwen_vl
    import comfy.text_encoders.qwen_image21 as qwen21

    _install_vision_dtype_wrappers(qwen35)

    # ------------------------------------------------------------------
    # 1) Force the Qwen-Image 2.1 TE factory to construct the model as FP16.
    #    Preserve the original quantization metadata and model options.
    # ------------------------------------------------------------------
    if not getattr(qwen21.te, "_qwen21_fp16_wrapped", False):
        original_te_factory = qwen21.te

        def te_factory_fp16(
            dtype_llama=None,
            llama_quantization_metadata=None,
        ):
            # Subclass the concrete Qwen-Image 2.1 TE class directly.
            # Do not subclass original_te_factory(...): its generated __init__
            # would overwrite our FP16 dtype again with dtype_llama.
            class QwenImage21TEModelFP16(qwen21.QwenImage21TEModel):
                def __init__(self, device="cpu", dtype=None, model_options=None):
                    if model_options is None:
                        model_options = {}
                    else:
                        model_options = model_options.copy()
                    if llama_quantization_metadata is not None:
                        model_options["quantization_metadata"] = llama_quantization_metadata
                    super().__init__(
                        device=device,
                        dtype=torch.float16,
                        model_options=model_options,
                    )

            QwenImage21TEModelFP16.__name__ = "QwenImage21TEModelFP16"
            QwenImage21TEModelFP16.__qualname__ = "QwenImage21TEModelFP16"
            return QwenImage21TEModelFP16

        te_factory_fp16._qwen21_fp16_wrapped = True
        te_factory_fp16._qwen21_fp16_original = original_te_factory
        qwen21.te = te_factory_fp16

    # ------------------------------------------------------------------
    # 2) Ensure ModelPatcher's compute dtype is FP16 for this TE only.
    #    This intercepts the existing sd.py float32 request before weights
    #    are loaded/cast. Other models retain the original behavior.
    # ------------------------------------------------------------------
    patcher_classes = []
    for name in ("ModelPatcher", "CoreModelPatcher"):
        cls = getattr(model_patcher, name, None)
        if cls is not None and cls not in patcher_classes:
            patcher_classes.append(cls)

    for cls in patcher_classes:
        current = getattr(cls, "set_model_compute_dtype", None)
        if current is None or getattr(current, "_qwen21_fp16_wrapped", False):
            continue

        original_set_dtype = current

        def make_set_dtype_wrapper(original_method):
            def set_model_compute_dtype_fp16(self, dtype):
                model = getattr(self, "model", None)
                if _is_qwen21_te(model, qwen21):
                    dtype = torch.float16
                    if not getattr(self, "_qwen21_fp16_patcher_logged", False):
                        _LOG.info(
                            "[Qwen21 FP16] ModelPatcher compute dtype forced to FP16"
                        )
                        self._qwen21_fp16_patcher_logged = True
                return original_method(self, dtype)

            set_model_compute_dtype_fp16._qwen21_fp16_wrapped = True
            set_model_compute_dtype_fp16._qwen21_fp16_original = original_method
            return set_model_compute_dtype_fp16

        setattr(cls, "set_model_compute_dtype", make_set_dtype_wrapper(original_set_dtype))

    # ------------------------------------------------------------------
    # 3) Mark the Qwen-Image 2.1 transformer instance so the vision input
    #    patch below cannot affect other Qwen3-VL users.
    # ------------------------------------------------------------------
    qwen21_clip_cls = qwen21.QwenImage21Qwen3VLClipModel
    if not getattr(qwen21_clip_cls.__init__, "_qwen21_fp16_wrapped", False):
        original_clip_init = qwen21_clip_cls.__init__

        def qwen21_clip_init_fp16(self, *args, **kwargs):
            original_clip_init(self, *args, **kwargs)
            transformer = getattr(self, "transformer", None)
            if transformer is not None:
                transformer._qwen21_fp16_patch = True
                transformer.dtype = torch.float16

                # The vision tower is Qwen35VisionModel. GGUF mmproj files may
                # retain FP32 positional-embedding tensors even when the model
                # compute dtype is FP16. fast_pos_embed_interpolate() then
                # returns FP32, and x + pos_embeds silently promotes the whole
                # vision stream to FP32. Cast the positional result at its
                # boundary, and do the same for the FP32 RoPE frequencies.
                visual = getattr(transformer, "visual", None)
                if visual is not None:
                    visual._qwen21_fp16_patch = True
                    visual._qwen21_fp16_dtype = torch.float16
                    _install_vision_dtype_hooks(visual)

                # Llama2_ computes text MRoPE frequencies in FP32. On this
                # target model, keep the frequencies in the activation dtype
                # before apply_rope_split_half so Q/K do not get promoted.
                llama_model = getattr(transformer, "model", None)
                if llama_model is not None and not getattr(llama_model, "_qwen21_fp16_rope_wrapped", False):
                    original_compute_freqs = llama_model.compute_freqs_cis

                    def compute_freqs_cis_fp16(position_ids, device):
                        freqs = original_compute_freqs(position_ids, device)
                        freqs = _cast_nested_fp16(freqs)
                        if not getattr(llama_model, "_qwen21_fp16_rope_logged", False):
                            def first_dtype(value):
                                if torch.is_tensor(value):
                                    return value.dtype
                                if isinstance(value, (tuple, list)):
                                    for item in value:
                                        found = first_dtype(item)
                                        if found is not None:
                                            return found
                                return None
                            _LOG.info("[Qwen21 FP16] text RoPE dtype=%s", first_dtype(freqs))
                            llama_model._qwen21_fp16_rope_logged = True
                        return freqs

                    llama_model.compute_freqs_cis = compute_freqs_cis_fp16
                    llama_model._qwen21_fp16_rope_wrapped = True

                if llama_model is not None:
                    layers = getattr(llama_model, "layers", None)
                    if layers and not getattr(llama_model, "_qwen21_fp16_layer_hook", False):
                        def first_text_block_hook(_module, _inputs, output):
                            value = output[0] if isinstance(output, (tuple, list)) and output else output
                            if torch.is_tensor(value):
                                _LOG.info("[Qwen21 FP16 TRACE] text.layers.0 output=%s", value.dtype)
                        layers[0].register_forward_hook(first_text_block_hook)
                        llama_model._qwen21_fp16_layer_hook = True
            self._qwen21_fp16_patch = True

        qwen21_clip_init_fp16._qwen21_fp16_wrapped = True
        qwen21_clip_init_fp16._qwen21_fp16_original = original_clip_init
        qwen21_clip_cls.__init__ = qwen21_clip_init_fp16

    # ------------------------------------------------------------------
    # 4) Replace only the Qwen-Image 2.1 process_tokens path.
    #    QwenImage21Qwen3VLClipModel.process_tokens() calls super(), so
    #    this SDClipModel wrapper catches that call and leaves all other
    #    SDClipModel subclasses untouched.
    # ------------------------------------------------------------------
    if not getattr(sd1_clip.SDClipModel.process_tokens, "_qwen21_fp16_wrapped", False):
        original_process_tokens = sd1_clip.SDClipModel.process_tokens
        fp16_process_tokens = _make_fp16_process_tokens(qwen21)

        def process_tokens_dispatch(self, tokens, device):
            if isinstance(self, qwen21_clip_cls):
                return fp16_process_tokens(self, tokens, device)
            return original_process_tokens(self, tokens, device)

        process_tokens_dispatch._qwen21_fp16_wrapped = True
        process_tokens_dispatch._qwen21_fp16_original = original_process_tokens
        sd1_clip.SDClipModel.process_tokens = process_tokens_dispatch

    # ------------------------------------------------------------------
    # 5) Feed FP16 image pixels into the Qwen3-VL vision tower, only when
    #    the transformer was marked by Qwen-Image 2.1's clip model.
    # ------------------------------------------------------------------
    if not getattr(qwen3vl.Qwen3VL.preprocess_embed, "_qwen21_fp16_wrapped", False):
        original_preprocess_embed = qwen3vl.Qwen3VL.preprocess_embed

        def preprocess_embed_fp16(self, embed, device):
            if (
                getattr(self, "_qwen21_fp16_patch", False)
                and embed.get("type") == "image"
            ):
                image, grid = qwen_vl.process_qwen2vl_images(
                    embed["data"],
                    patch_size=16,
                    image_mean=[0.5, 0.5, 0.5],
                    image_std=[0.5, 0.5, 0.5],
                )
                image = image.to(device=device, dtype=torch.float16)
                merged, deepstack = self.visual(image, grid)

                if not getattr(self, "_qwen21_fp16_vision_logged", False):
                    deepstack_dtypes = [x.dtype for x in deepstack] if deepstack is not None else []
                    _LOG.info(
                        "[Qwen21 FP16] vision input=%s, vision output=%s, DeepStack=%s",
                        image.dtype,
                        merged.dtype,
                        deepstack_dtypes,
                    )
                    self._qwen21_fp16_vision_logged = True

                return merged, {"grid": grid, "deepstack": deepstack}

            return original_preprocess_embed(self, embed, device)

        preprocess_embed_fp16._qwen21_fp16_wrapped = True
        preprocess_embed_fp16._qwen21_fp16_original = original_preprocess_embed
        qwen3vl.Qwen3VL.preprocess_embed = preprocess_embed_fp16

    _PATCHED = True
    _LOG.warning(
        "============================================================\n"
        "[Qwen21 FP16 Patch] Installed.\n"
        "Target: built-in Qwen-Image 2.1 / Qwen3-VL 8B TE only.\n"
        "TE compute, token embeddings, vision positional embeddings/RoPE and text RoPE: FP16.\n"
        "Vision/text intermediate dtype tracing enabled; final conditioning remains FP32.\n"
        "DiT, VAE and other models remain unchanged.\n"
        "============================================================"
    )


try:
    _install()
except Exception:
    _LOG.exception(
        "[Qwen21 FP16 Patch] Installation failed. "
        "The original ComfyUI code has not been edited."
    )


# This is a startup-only patch; no visible workflow node is required.
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
