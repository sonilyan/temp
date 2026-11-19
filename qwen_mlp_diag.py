
"""
Qwen-Image-2.1 Chunked SwiGLU MLP
=================================

Patch SwiGLUFeedForward.forward() to process the sequence dimension
in chunks, reducing peak intermediate activation memory.

Supports:
    fused=False:
        out(silu(gate_layer(x)) * proj(x))

    fused=True:
        linear_input_act(out, gate_up(x), "swiglu")

Weights remain on their original devices.
No CPU offload, no weight conversion, no empty_cache calls.

Default chunk size: 2048 tokens.
Override with environment variable:
    QWEN21_MLP_CHUNK_SIZE=1024
    QWEN21_MLP_CHUNK_SIZE=2048
    QWEN21_MLP_CHUNK_SIZE=4096

Set to 0 to disable chunking.
"""

import os
import sys
import functools

import torch
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.qwen_image21.model import SwiGLUFeedForward


PATCH_TAG = "_qwen21_chunked_mlp_patch"

# Shared phase state. Attention reads this module through the stable alias below.
# 0 = normal/default behavior
# 1 = first sampler step: Streaming Attention + chunked MLP
# 2 = remaining sampler steps: Legacy Attention + full MLP
PHASE = 0
_PHASE_ALIAS = "qwen21_phase_state"
sys.modules[_PHASE_ALIAS] = sys.modules[__name__]

_BYPASS_LOGGED = set()
_SAMPLER_HOOK_TAG = "_qwen21_phase_callback_hook"


def set_phase(value):
    """Set the shared Qwen21 sampling phase."""
    global PHASE
    PHASE = int(value)


def get_phase():
    return PHASE


def _is_qwen21_model(model_patcher):
    """Return True only for the Qwen Image 2.1 transformer."""
    base_model = getattr(model_patcher, "model", None)
    diffusion_model = getattr(base_model, "diffusion_model", None)
    if diffusion_model is None:
        diffusion_model = getattr(model_patcher, "diffusion_model", None)
    class_name = type(diffusion_model).__name__ if diffusion_model is not None else ""
    return class_name == "QwenImage21Transformer2DModel"


def install_sampler_phase_hook():
    """Wrap ComfyUI's existing preview callback; no custom sampler node/core edit."""
    try:
        import latent_preview
    except Exception as exc:
        print(
            "[QWEN21 PHASE] Could not import latent_preview; "
            f"sampler phase switching is unavailable: {exc}",
            flush=True,
        )
        return False

    current = latent_preview.prepare_callback
    if getattr(current, _SAMPLER_HOOK_TAG, False):
        return True

    @functools.wraps(current)
    def patched_prepare_callback(model, steps, x0_output_dict=None):
        callback = current(model, steps, x0_output_dict)
        if not _is_qwen21_model(model):
            # A non-Qwen sampler must never inherit a stale phase.
            set_phase(0)
            return callback

        step_count = max(0, int(steps))
        set_phase(1)
        print(
            f"[QWEN21 PHASE] START phase=1 steps={step_count} "
            "(Streaming Attention + chunked MLP)",
            flush=True,
        )

        @functools.wraps(callback)
        def phase_callback(step, x0, x, total_steps):
            try:
                return callback(step, x0, x, total_steps)
            finally:
                # ComfyUI's sampler callback is called after a denoising step.
                # Therefore step 0 has completed before we switch to phase 2.
                try:
                    step_index = int(step)
                except (TypeError, ValueError):
                    step_index = -1
                try:
                    callback_steps = int(total_steps)
                except (TypeError, ValueError):
                    callback_steps = step_count

                if step_index == 0:
                    set_phase(2)
                    print(
                        "[QWEN21 PHASE] SWITCH phase=2 "
                        "(Legacy Attention + full MLP)",
                        flush=True,
                    )

                if callback_steps > 0 and step_index + 1 >= callback_steps:
                    set_phase(0)
                    print("[QWEN21 PHASE] END phase=0", flush=True)

        return phase_callback

    setattr(patched_prepare_callback, _SAMPLER_HOOK_TAG, True)
    latent_preview.prepare_callback = patched_prepare_callback
    print(
        "[QWEN21 PHASE] Hook installed on latent_preview.prepare_callback",
        flush=True,
    )
    return True

try:
    CHUNK_SIZE = int(os.environ.get("QWEN21_MLP_CHUNK_SIZE", "4096"))
except (TypeError, ValueError):
    CHUNK_SIZE = 4096

if CHUNK_SIZE < 0:
    CHUNK_SIZE = 4096


def _original_forward(self, x):
    """Original ComfyUI implementation, unchanged."""
    if self.fused:
        return comfy.ops.linear_input_act(
            self.out,
            self.gate_up(x),
            "swiglu",
        )

    return self.out(
        F.silu(self.gate_layer(x)) * self.proj(x)
    )


def _chunked_forward(self, x):
    """
    Split only the sequence dimension (-2).

    Every token is processed independently by the MLP, so this does
    not change the mathematical operation or token ordering.
    """

    # Phase 2 deliberately bypasses chunking and uses the original full MLP.
    if PHASE == 2:
        bypass_key = (tuple(x.shape), str(x.dtype), str(x.device), self.fused)
        if bypass_key not in _BYPASS_LOGGED:
            _BYPASS_LOGGED.add(bypass_key)
            print(
                "[QWEN21 MLP] BYPASS phase=2 full_forward "
                f"shape={tuple(x.shape)} dtype={x.dtype} "
                f"device={x.device} fused={self.fused}",
                flush=True,
            )
        return _original_forward(self, x)

    # Phase 0 keeps the pre-existing default behavior; phase 1 chunks.
    # Disabled mode: execute the original implementation.
    if CHUNK_SIZE == 0:
        return _original_forward(self, x)

    # Expected input: [B, S, C].
    if x.ndim < 2:
        return _original_forward(self, x)

    seq_len = x.shape[-2]

    # Small inputs do not need chunking.
    if seq_len <= CHUNK_SIZE:
        return _original_forward(self, x)

    chunk_count = (seq_len + CHUNK_SIZE - 1) // CHUNK_SIZE

    # Print once per input shape, avoiding excessive console output.
    shape_key = (tuple(x.shape), str(x.dtype), str(x.device), self.fused)

    seen_shapes = getattr(self, "_qwen21_chunk_seen_shapes", None)
    if seen_shapes is None:
        seen_shapes = set()
        self._qwen21_chunk_seen_shapes = seen_shapes

    if shape_key not in seen_shapes:
        seen_shapes.add(shape_key)


    outputs = []

    for start in range(0, seq_len, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, seq_len)

        # Slice tokens; do not slice channels.
        x_chunk = x.narrow(-2, start, end - start)

        if self.fused:
            # Preserve the original fused operation.
            gate_up = self.gate_up(x_chunk)

            y_chunk = comfy.ops.linear_input_act(
                self.out,
                gate_up,
                "swiglu",
            )

            del gate_up

        else:
            # Preserve the original non-fused operation and ordering.
            gate = self.gate_layer(x_chunk)
            activated_gate = F.silu(gate)

            del gate

            up = self.proj(x_chunk)

            hidden = activated_gate * up

            del activated_gate
            del up

            y_chunk = self.out(hidden)

            del hidden

        outputs.append(y_chunk)

        del x_chunk

    # Concatenate in original token order.
    # This retains ordinary PyTorch autograd semantics.
    result = torch.cat(outputs, dim=-2)

    del outputs

    if result.shape[-2] != seq_len:
        raise RuntimeError(
            "[QWEN21 MLP CHUNK] Output sequence length mismatch: "
            f"input={seq_len}, output={result.shape[-2]}"
        )

    if shape_key in seen_shapes:
        # Only print completion on the first invocation for this shape.
        seen_shapes.remove(shape_key)

    return result


def install_patch():
    current = SwiGLUFeedForward.forward

    if getattr(current, PATCH_TAG, False):
        print(
            "[QWEN21 MLP CHUNK] Patch already installed.",
            flush=True,
        )
        return

    # Keep the original method available for clean fallback.
    SwiGLUFeedForward._qwen21_original_forward = current

    @functools.wraps(current)
    def patched_forward(self, x):
        return _chunked_forward(self, x)

    setattr(patched_forward, PATCH_TAG, True)

    SwiGLUFeedForward.forward = patched_forward

    print(
        "[QWEN21 MLP CHUNK] Installed successfully. "
        f"chunk_size={CHUNK_SIZE}",
        flush=True,
    )


install_patch()
install_sampler_phase_hook()
