"""Qwen-Image-2.1 Viggle Turbo nodes for ComfyUI.

Modified from the official Viggle v0.3 ComfyUI node.

Adds a ComfyUI-native port of the official v0.3 10-step experimental behavior:
  steps 1-7: Viggle Turbo LoRA ON
  steps 8-10: base Qwen-Image-2.1, LoRA OFF
  at the first base step, ComfyUI Qwen-Image-2.1 prefix K/V cache is
  invalidated so the base model rebuilds it with LoRA disabled.

"""

import json
import math

import torch
import torch.nn.functional as F

import comfy.patcher_extension
import comfy.utils
import folder_paths


print("[ViggleTurbo] 10-step debug build loaded.", flush=True)


class ViggleTurboSigmas:
    """Viggle Turbo v0.3 scheduler: selectable 6, 9, or 10 steps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": (
                    "LATENT",
                    {"tooltip": "Latent used to calculate the resolution-dependent sigma shift."},
                ),
                "steps": (
                    ["6", "10"],
                    {"default": "6", "tooltip": "6=all Turbo; 10=7 Turbo+3 Base."},
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    # Official v0.3 6-step schedule.
    SIX_STEP_NODES = (
        1.0,
        0.9375,
        0.875,
        0.75,
        0.5,
        0.25,
    )

    # Existing experimental 10-step mode:
    # 7 Turbo steps + 3 Base steps.
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

    def get_sigmas(self, latent, steps):
        s = latent["samples"]
        r = latent.get("downscale_ratio_spacial", 16) / 16
        tokens = round(s.shape[-2] * r) * round(s.shape[-1] * r)
        mu = 0.5 + (0.9 - 0.5) * (tokens - 256) / (8192 - 256)

        if steps == "6":
            nodes = self.SIX_STEP_NODES
        else:
            nodes = self.TEN_STEP_NODES

        t = torch.tensor(nodes, dtype=torch.float64)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / t - 1))
        return (torch.cat([sigmas, sigmas.new_zeros(1)]).float(),)


def lora_fwd(x, ab):
    return F.linear(F.linear(x, ab[0].to(x.dtype)), ab[1].to(x.dtype))


def add_hook(mod, ab):
    return mod.register_forward_hook(
        lambda m, inp, out: out + lora_fwd(inp[0], ab)
    )


def add_mlp_hooks(mlp, gate, up, down):
    # fused SwiGLU: gate_up = [gate_layer; proj], and `out` runs inside
    # an int8/fp16 kernel that bypasses its hooks.
    h = {}

    def gate_up_hook(m, inp, out):
        h["gu"] = out + torch.cat(
            [lora_fwd(inp[0], gate), lora_fwd(inp[0], up)], -1
        )
        return h["gu"]

    def mlp_hook(m, inp, out):
        g, u = h.pop("gu").chunk(2, -1)
        return out + lora_fwd(F.silu(g) * u, down)

    return [
        mlp.gate_up.register_forward_hook(gate_up_hook),
        mlp.register_forward_hook(mlp_hook),
    ]


def _get_transformer_options(args, kwargs):
    """Get transformer_options from the actual ComfyUI diffusion wrapper call."""
    options = kwargs.get("transformer_options")
    if isinstance(options, dict):
        return options

    # QwenImage21Transformer2DModel.forward() passes transformer_options as
    # the 6th positional argument to WrapperExecutor.execute():
    # x, timestep, context, ref_latents, image_slots, transformer_options.
    for arg in reversed(args):
        if isinstance(arg, dict) and (
            "sample_sigmas" in arg
            or "sigmas" in arg
            or "qwen_image21_cache" in arg
            or "wrappers" in arg
        ):
            return arg

    # Compatibility fallback for wrapper/signature variants.
    for arg in reversed(args):
        if isinstance(arg, dict):
            nested = arg.get("transformer_options")
            if isinstance(nested, dict):
                return nested
    return None


def _get_step_index(transformer_options):
    """Map the current sigma to its position in ComfyUI's sample_sigmas.

    ComfyUI exposes both:
      transformer_options["sample_sigmas"] = full sampler schedule
      transformer_options["sigmas"]        = current model sigma

    This is intentionally sigma-based because ComfyUI's sampler step index is
    not universally exposed to transformer_options across sampler variants.
    """
    if not isinstance(transformer_options, dict):
        return None

    sample_sigmas = transformer_options.get("sample_sigmas")
    current_sigmas = transformer_options.get("sigmas")

    if sample_sigmas is None or current_sigmas is None:
        return None

    if not torch.is_tensor(sample_sigmas):
        sample_sigmas = torch.as_tensor(sample_sigmas)
    if not torch.is_tensor(current_sigmas):
        current_sigmas = torch.as_tensor(current_sigmas)

    sample_sigmas = sample_sigmas.flatten()
    current_sigmas = current_sigmas.flatten()

    if sample_sigmas.numel() < 2 or current_sigmas.numel() < 1:
        return None

    # In normal ComfyUI samplers this is one scalar sigma. For batched CFG,
    # transformer_options still carries the same scalar schedule value.
    current = current_sigmas[0].to(
        device=sample_sigmas.device,
        dtype=sample_sigmas.dtype,
    )

    matches = torch.nonzero(
        torch.isclose(sample_sigmas, current, rtol=1e-4, atol=1e-5)
    )
    if matches.numel() == 0:
        return None

    return int(matches[0, 0].item())


def _get_viggle_schedule(transformer_options):
    """Return 6, 9, or 10 for a matching Viggle schedule, otherwise None."""
    if not isinstance(transformer_options, dict):
        return None

    sample_sigmas = transformer_options.get("sample_sigmas")
    if sample_sigmas is None:
        return None

    try:
        sig = torch.as_tensor(sample_sigmas).flatten().float()
    except Exception:
        return None

    if sig.numel() not in (7, 10, 11) or not torch.isclose(
        sig[-1], sig.new_zeros(()), atol=1e-6, rtol=0
    ):
        return None

    if not torch.isclose(sig[0], sig.new_ones(()), atol=1e-5, rtol=1e-5):
        return None

    # 6-step official schedule.
    if sig.numel() == 7:
        q = 1.0 / sig[:6].clamp_min(1e-8) - 1.0
        ref = q[5]
        expected = sig.new_tensor([1.0 / 15.0, 1.0 / 7.0, 1.0 / 3.0, 1.0, 3.0])
        actual = q[1:6] / ref
        if torch.isfinite(ref) and ref > 0 and torch.allclose(
            actual, expected, rtol=2e-3, atol=2e-4
        ):
            return 6

    # 9-step official schedule.
    if sig.numel() == 10:
        q = 1.0 / sig[:9].clamp_min(1e-8) - 1.0
        ref = q[5]
        expected = sig.new_tensor(
            [1.0 / 23.0, 1.0 / 11.0, 1.0 / 7.0, 1.0 / 3.0,
             1.0, 3.0, 5.0, 11.0]
        )
        actual = q[1:9] / ref
        if torch.isfinite(ref) and ref > 0 and torch.allclose(
            actual, expected, rtol=2e-3, atol=2e-4
        ):
            return 9

    # Existing 10-step schedule.
    if sig.numel() == 11:
        q = 1.0 / sig[:10].clamp_min(1e-8) - 1.0
        ref = q[5]
        expected = sig.new_tensor(
            [1.0 / 23.0, 1.0 / 11.0, 1.0 / 7.0, 1.0 / 3.0,
             1.0, 3.0, 5.0, 8.0, 17.0]
        )
        actual = q[1:10] / ref
        if torch.isfinite(ref) and ref > 0 and torch.allclose(
            actual, expected, rtol=2e-3, atol=2e-4
        ):
            return 10

    return None


def _reset_qwen_prefix_cache(dm):
    """Invalidate ComfyUI Qwen-Image-2.1's prefix K/V cache.

    ComfyUI's Qwen-Image-2.1 implementation owns this cache. Unlike the
    Diffusers implementation, it does not use a kv_cache_mode='extract'
    argument. Resetting the cache before the first Base-model forward makes
    that forward miss the old Turbo cache and rebuild the prefix K/V using the
    LoRA-disabled weights.
    """
    reset = getattr(dm, "reset_prefix_cache", None)
    if not callable(reset):
        return False

    # The first Base step must rebuild its prefix cache even if the current
    # flag was disabled by another model-patcher state. QwenImage21Cache's
    # per-workflow device="off" setting is still respected by select_prefix_cache.
    reset(True)
    return True


def run_with_lora(lora, executor, state, *args, **kwargs):
    dm = executor.class_obj

    transformer_options = _get_transformer_options(args, kwargs)

    # One diagnostic line on the first diffusion-model invocation. This is
    # intentionally explicit so a real ComfyUI run can prove whether the
    # wrapper receives sample_sigmas/sigmas at all.
    if not state["debug_printed"]:
        state["debug_printed"] = True
        if isinstance(transformer_options, dict):
            sample = transformer_options.get("sample_sigmas")
            current = transformer_options.get("sigmas")
            sample_shape = tuple(sample.shape) if torch.is_tensor(sample) else type(sample).__name__
            current_shape = tuple(current.shape) if torch.is_tensor(current) else type(current).__name__
            sample_values = (
                [round(float(v), 7) for v in sample.detach().flatten().cpu().tolist()]
                if torch.is_tensor(sample) else repr(sample)
            )
            current_values = (
                [round(float(v), 7) for v in current.detach().flatten().cpu().tolist()]
                if torch.is_tensor(current) else repr(current)
            )
            print(
                "[ViggleTurbo DEBUG] transformer_options received; "
                f"keys={sorted(transformer_options.keys())}",
                flush=True,
            )
            print(
                "[ViggleTurbo DEBUG] sample_sigmas="
                f"{sample_values} shape={sample_shape}; "
                f"sigmas={current_values} shape={current_shape}",
                flush=True,
            )
        else:
            print(
                "[ViggleTurbo DEBUG] ERROR: transformer_options not found "
                f"in kwargs/args; kwargs={list(kwargs.keys())}, "
                f"arg_types={[type(a).__name__ for a in args]}",
                flush=True,
            )

    viggle_steps = _get_viggle_schedule(transformer_options)
    step_index = _get_step_index(transformer_options)

    if viggle_steps is not None:
        if not state["schedule_printed"]:
            state["schedule_printed"] = True
            print(
                "[ViggleTurbo] "
                f"{viggle_steps}-step detected: current step index={step_index}.",
                flush=True,
            )
        if step_index is None:
            raise RuntimeError(
                "[ViggleTurbo] Fixed 10-step schedule detected, but the current "
                "sampler sigma could not be matched to sample_sigmas. "
                "Generation stopped to prevent accidentally applying Turbo LoRA "
                "during the final Base steps."
            )

        previous_step = state["last_step"]
        state["last_step"] = step_index
        if step_index != previous_step:
            print(
                f"[ViggleTurbo] 10-step step {step_index + 1}/10: "
                + ("LoRA ON" if step_index < 7 else "LoRA OFF / Base"),
                flush=True,
            )
        if step_index == 0 and previous_step != 0:
            state["reset_done"] = False

        # 6-step: all six steps use Turbo LoRA.
        # 9-step: first 7 Turbo, last 2 Base.
        # 10-step: first 7 Turbo, last 3 Base.
        if viggle_steps in (9, 10) and step_index >= 7:
            if not state["reset_done"]:
                ok = _reset_qwen_prefix_cache(dm)
                state["reset_done"] = True

                if ok:
                    print(
                        "[ViggleTurbo] 10-step: after step 7 -> "
                        "LoRA OFF + Qwen-Image-2.1 prefix K/V cache reset.",
                        flush=True,
                    )
                else:
                    print(
                        "[ViggleTurbo] ERROR: 10-step requires "
                        "Qwen-Image-2.1 reset_prefix_cache(); "
                        "LoRA will still be OFF, but the old prefix cache "
                        "could not be invalidated.",
                        flush=True,
                    )

            # Do NOT install LoRA hooks for the Base refinement steps.
            return executor(*args, **kwargs)

        # Step 0 is a useful sanity check: the first 7 steps should be the
        # Turbo branch.
        if step_index == 0 and not state["started"]:
            state["started"] = True
            print(
                "[ViggleTurbo] 10-step: steps 1-7 LoRA ON.",
                flush=True,
            )

    hooks = []

    # Preserve the exact official ComfyUI runtime-LoRA implementation for
    # the Turbo portion. The hooks exist only for this single forward.
    for ab in lora.values():
        if ab[0].device != args[0].device:
            ab[0], ab[1] = ab[0].to(args[0].device), ab[1].to(args[0].device)

    for name, ab in lora.items():
        parent, _, leaf = name.rpartition(".")
        parent_mod = dm.get_submodule(parent)

        if not getattr(parent_mod, "fused", False):
            hooks.append(add_hook(dm.get_submodule(name), ab))
        elif leaf == "out":
            hooks += add_mlp_hooks(
                parent_mod,
                lora[parent + ".gate_layer"],
                lora[parent + ".proj"],
                ab,
            )

    try:
        return executor(*args, **kwargs)
    finally:
        for hk in hooks:
            hk.remove()



class ViggleTurboLora:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {"tooltip": "diffusers-format Qwen-Image-2.1 LoRA."},
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -4.0,
                        "max": 4.0,
                        "step": 0.05,
                        "tooltip": "Keep 1.0 for viggle-turbo.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Applies the LoRA at runtime. "
        "Selectable 6/9/10-step schedule. 6=all Turbo; 9=7 Turbo+2 Base; "
        "10=7 Turbo+3 Base. Base tail disables LoRA and re-extracts prefix K/V cache."
    )

    def load(self, model, lora_name, strength):
        sd, meta = comfy.utils.load_torch_file(
            folder_paths.get_full_path_or_raise("loras", lora_name),
            return_metadata=True,
        )

        cfg = json.loads((meta or {}).get("lora_adapter_metadata", "{}"))
        scale = (
            strength
            * cfg.get("transformer.lora_alpha", 1)
            / cfg.get("transformer.r", 1)
        )

        lora = {
            k.removeprefix("transformer.").removesuffix(".lora_A.weight"): [
                sd[k],
                sd[k.replace("lora_A", "lora_B")] * scale,
            ]
            for k in sd
            if k.endswith(".lora_A.weight")
        }

        m = model.clone()
        state = {
            "started": False,
            "last_step": None,
            "reset_done": False,
            "debug_printed": False,
            "schedule_printed": False,
        }
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "viggle_turbo_lora",
            lambda executor, *a, **kw: run_with_lora(
                lora, executor, state, *a, **kw
            ),
        )
        return (m,)


NODE_CLASS_MAPPINGS = {
    "ViggleTurboSigmas": ViggleTurboSigmas,
    "ViggleTurboLora": ViggleTurboLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ViggleTurboSigmas": "Qwen-Image-2.1 Viggle Turbo Sigmas (6/9/10)",
    "ViggleTurboLora": "Qwen-Image-2.1 Viggle Turbo LoRA (unmerged)",
}
