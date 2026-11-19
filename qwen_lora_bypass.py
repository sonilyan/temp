import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import comfy.utils
import folder_paths


class QwenHookLora:
    """
    Qwen Image 2.1 专用 LoRA Bypass 加载器。

    设计目标：
    - 只提供 Bypass，不提供 Merge，避免量化底模反量化/大矩阵 patch。
    - 使用 ComfyUI 官方 BypassInjectionManager 管理注入和卸载。
    - 不修改 comfy.ops 全局函数。
    - fused img_mlp.out 通过官方 BypassForwardHook 设置 comfy_force_forward，
      使 linear_input_act 走模块 forward，从而正常进入 LoRA Bypass。
    - LoRA 权重保留文件原始 dtype 存储；增量计算时转 FP32。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "loaders"

    @staticmethod
    def _read_safetensors_metadata(path):
        """Read optional safetensors metadata without making it a hard dependency."""
        try:
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as f:
                return f.metadata() or {}
        except Exception:
            return {}

    @staticmethod
    def _metadata_alpha(metadata):
        for key in ("ss_network_alpha", "network_alpha", "lora_alpha"):
            value = metadata.get(key)
            if value is not None:
                try:
                    return float(value), key
                except (TypeError, ValueError):
                    continue
        return None, None

    @staticmethod
    def _strip_prefix(name):
        core = name
        prefixes = (
            "model.diffusion_model.",
            "diffusion_model.",
            "transformer.",
            "base_model.model.",
        )
        while True:
            for prefix in prefixes:
                if core.startswith(prefix):
                    core = core[len(prefix):]
                    break
            else:
                return core

    @staticmethod
    def _suffix_matches(core, module_names):
        parts = core.split(".")
        return [
            name
            for name in module_names
            if len(name.split(".")) >= len(parts)
            and name.split(".")[-len(parts):] == parts
        ]

    @staticmethod
    def _shape_ok(a, b, target_shape):
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            return False, "A/B 不是 Tensor"
        if a.ndim != 2 or b.ndim != 2:
            return False, (
                f"A/B 必须是二维矩阵，实际 A{tuple(a.shape)} B{tuple(b.shape)}"
            )
        if a.shape[0] != b.shape[1]:
            return False, f"rank 不一致，A{tuple(a.shape)} B{tuple(b.shape)}"
        if len(target_shape) != 2:
            return False, f"目标权重不是二维 Linear，shape={tuple(target_shape)}"
        if a.shape[1] != target_shape[1] or b.shape[0] != target_shape[0]:
            return False, (
                f"目标维度不匹配，A{tuple(a.shape)} B{tuple(b.shape)} "
                f"target{tuple(target_shape)}"
            )
        return True, "OK"

    @staticmethod
    def _make_adapter_class():
        """
        Create a train-base adapter so ComfyUI's BypassInjectionManager can
        manage its device placement and injection lifecycle.

        Parameters retain the LoRA file dtype in VRAM. The low-rank products
        are explicitly accumulated in FP32 inside bypass_forward().
        """
        try:
            from comfy.weight_adapter.base import WeightAdapterTrainBase
        except Exception as exc:
            raise RuntimeError(
                "当前 ComfyUI 缺少 WeightAdapterTrainBase，"
                "无法使用官方 Bypass 注入接口。请更新 ComfyUI。"
            ) from exc

        class QwenLoRAAdapter(WeightAdapterTrainBase):
            def __init__(self, down, up, alpha, layer_name, execution_tracker, is_split=False):
                super().__init__()
                self.is_split = is_split
                if is_split:
                    # 接收的是 (A_gate, A_proj) 和 (B_gate, B_proj)
                    self.down_gate = nn.Parameter(down[0].detach(), requires_grad=False)
                    self.down_proj = nn.Parameter(down[1].detach(), requires_grad=False)
                    self.up_gate = nn.Parameter(up[0].detach(), requires_grad=False)
                    self.up_proj = nn.Parameter(up[1].detach(), requires_grad=False)
                    self.rank = int(down[0].shape[0])
                else:
                    self.down = nn.Parameter(down.detach(), requires_grad=False)
                    self.up = nn.Parameter(up.detach(), requires_grad=False)
                    self.rank = int(down.shape[0])

                self.alpha = nn.Parameter(
                    torch.tensor(float(alpha), dtype=torch.float32, device="cpu"),
                    requires_grad=False,
                )
                self.layer_name = layer_name
                self.execution_tracker = execution_tracker

            def _record_execution(self):
                tracker = self.execution_tracker
                with tracker["lock"]:
                    tracker["calls"] += 1
                    tracker["seen"].add(self.layer_name)
                    if self.layer_name in tracker["fused_names"]:
                        tracker["fused_seen"].add(self.layer_name)

                    unique_count = len(tracker["seen"])
                    if (
                        tracker["next_milestone"] < len(tracker["milestones"])
                        and unique_count >= tracker["milestones"][tracker["next_milestone"]]
                    ):
                        while (
                            tracker["next_milestone"] < len(tracker["milestones"])
                            and unique_count >= tracker["milestones"][tracker["next_milestone"]]
                        ):
                            tracker["next_milestone"] += 1

                        print(
                            "[QwenHookLora] 实际执行覆盖: "
                            f"{unique_count}/{tracker['expected_layers']} 层 | "
                            "fused img_mlp.out: "
                            f"{len(tracker['fused_seen'])}/{tracker['expected_fused']} 层 | "
                            f"Hook累计调用={tracker['calls']}"
                        )

            def h(self, x, base_out):
                x32 = x.to(dtype=torch.float32)
                scale = (
                    self.alpha.to(device=x.device, dtype=torch.float32)
                    / self.rank
                ) * float(getattr(self, "multiplier", 1.0))
                
                if self.is_split:
                    # 双矩阵独立计算
                    down_gate32 = self.down_gate.to(device=x.device, dtype=torch.float32)
                    up_gate32 = self.up_gate.to(device=x.device, dtype=torch.float32)
                    down_proj32 = self.down_proj.to(device=x.device, dtype=torch.float32)
                    up_proj32 = self.up_proj.to(device=x.device, dtype=torch.float32)
                    
                    delta_gate = F.linear(F.linear(x32, down_gate32), up_gate32)
                    delta_proj = F.linear(F.linear(x32, down_proj32), up_proj32)
                    
                    # 拼接起来，模拟 gate_up 的输出
                    delta = torch.cat([delta_gate, delta_proj], dim=-1)
                else:
                    down32 = self.down.to(device=x.device, dtype=torch.float32)
                    up32 = self.up.to(device=x.device, dtype=torch.float32)
                    hidden = F.linear(x32, down32)
                    delta = F.linear(hidden, up32)
                    
                return delta * scale

            def bypass_forward(self, org_forward, x, *args, **kwargs):
                input_act = kwargs.get("input_act")
                lora_input = x
                if input_act is not None:
                    if input_act == "swiglu":
                        gate, up = x.chunk(2, dim=-1)
                        lora_input = F.silu(gate) * up
                    elif input_act == "gelu_tanh":
                        lora_input = F.gelu(x, approximate="tanh")
                    else:
                        raise RuntimeError(
                            f"QwenLoRAAdapter 不支持 Linear input_act={input_act!r}"
                        )

                base_out = org_forward(x, *args, **kwargs)
                delta = self.h(lora_input, base_out)
                result = (base_out.to(torch.float32) + delta).to(base_out.dtype)

                self._record_execution()
                return result

            def passive_memory_usage(self):
                return sum(
                    p.numel() * p.element_size() for p in self.parameters()
                )

        return QwenLoRAAdapter

    def load(self, model, lora_name, strength):
        print("\n[QwenHookLora] ================ Bypass 最终版 ================")
        print(
            f"[QwenHookLora] LoRA: {lora_name} | strength={float(strength)} "
            "| mode=官方 Bypass（仅旁路，不修改底模权重）"
        )

        if float(strength) == 0.0:
            print("[QwenHookLora] strength=0，直接返回原模型。")
            print("[QwenHookLora] ========================================\n")
            return (model,)

        try:
            from comfy.weight_adapter import BypassInjectionManager
        except Exception as exc:
            raise RuntimeError(
                "当前 ComfyUI 没有 comfy.weight_adapter.BypassInjectionManager。"
                "此版本无法安全处理 Qwen fused MLP Bypass，请更新 ComfyUI。"
            ) from exc

        if not hasattr(model, "set_injections"):
            raise RuntimeError(
                "当前 ModelPatcher 不支持 set_injections()，"
                "无法安全安装 Bypass。请更新 ComfyUI。"
            )

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
        metadata = self._read_safetensors_metadata(lora_path)
        global_alpha, global_alpha_key = self._metadata_alpha(metadata)

        dm = model.model.diffusion_model
        module_shapes = {}
        for name, mod in dm.named_modules():
            weight = getattr(mod, "weight", None)
            shape = getattr(weight, "shape", None)
            if shape is not None:
                try:
                    module_shapes[name] = tuple(shape)
                except TypeError:
                    pass

        resolved = {}
        matched_direct = 0
        matched_split = 0
        matched_merged = 0
        failed_count = 0
        failed_examples = []
        alpha_sources = {"layer": 0, "metadata": 0, "missing": 0}
        pending_gate = {}
        pending_proj = {}

        for key, a_tensor in sd.items():
            if not ("lora_A" in key or "lora_down" in key):
                continue

            base = (
                key.split("lora_A")[0]
                .rstrip(".")
                .split("lora_down")[0]
                .rstrip(".")
            )

            b_key = None
            if key.endswith(".weight"):
                for candidate in (
                    key.replace("lora_A", "lora_B"),
                    key.replace("lora_down", "lora_up"),
                ):
                    if candidate in sd:
                        b_key = candidate
                        break

            if b_key is None:
                failed_count += 1
                if len(failed_examples) < 8:
                    failed_examples.append(
                        f"{base}: 找不到配对的 B/up tensor"
                    )
                continue

            b_tensor = sd[b_key]
            if not isinstance(a_tensor, torch.Tensor) or not isinstance(
                b_tensor, torch.Tensor
            ):
                failed_count += 1
                if len(failed_examples) < 8:
                    failed_examples.append(f"{base}: A/B 不是 Tensor")
                continue

            alpha = None
            alpha_source = "missing"
            for suffix in (".alpha", ".lora_alpha"):
                alpha_key = base + suffix
                if alpha_key in sd:
                    try:
                        raw_alpha = sd[alpha_key]
                        alpha = float(
                            raw_alpha.item()
                            if isinstance(raw_alpha, torch.Tensor)
                            else raw_alpha
                        )
                        alpha_source = "layer"
                    except (TypeError, ValueError, RuntimeError):
                        pass
                    break

            if alpha is None and global_alpha is not None:
                alpha = global_alpha
                alpha_source = f"metadata:{global_alpha_key}"

            if alpha_source == "layer":
                alpha_sources["layer"] += 1
            elif alpha_source.startswith("metadata:"):
                alpha_sources["metadata"] += 1
            else:
                alpha_sources["missing"] += 1

            core = self._strip_prefix(base)

            matched_real = core if core in module_shapes else None
            if matched_real is None:
                candidates = self._suffix_matches(core, module_shapes.keys())
                if len(candidates) == 1:
                    matched_real = candidates[0]
                elif len(candidates) > 1:
                    failed_count += 1
                    if len(failed_examples) < 8:
                        failed_examples.append(
                            f"{core}: 后缀匹配有歧义 {candidates[:4]}"
                        )
                    continue

            if matched_real is not None:
                ok, reason = self._shape_ok(
                    a_tensor, b_tensor, module_shapes[matched_real]
                )
                if not ok:
                    failed_count += 1
                    if len(failed_examples) < 8:
                        failed_examples.append(f"{core}: {reason}")
                    continue
                if matched_real in resolved:
                    failed_count += 1
                    if len(failed_examples) < 8:
                        failed_examples.append(f"{core}: 重复映射到同一目标")
                    continue
                resolved[matched_real] = (a_tensor, b_tensor, alpha)
                matched_direct += 1
                continue

            if "img_mlp.gate_up" in core or "img.mlp.gate.up" in core:
                gate_suffix = (
                    core.replace("img_mlp.gate_up", "img_mlp.gate_layer")
                    .replace("img.mlp.gate.up", "img.mlp.gate_layer")
                )
                proj_suffix = (
                    core.replace("img_mlp.gate_up", "img_mlp.proj")
                    .replace("img.mlp.gate.up", "img.mlp.proj")
                )
                gates = self._suffix_matches(gate_suffix, module_shapes.keys())
                projs = self._suffix_matches(proj_suffix, module_shapes.keys())

                if (
                    len(gates) == 1
                    and len(projs) == 1
                    and b_tensor.ndim == 2
                    and b_tensor.shape[0] % 2 == 0
                ):
                    half = b_tensor.shape[0] // 2
                    gate_b = b_tensor[:half, :].contiguous()
                    proj_b = b_tensor[half:, :].contiguous()
                    ok_g, _ = self._shape_ok(
                        a_tensor, gate_b, module_shapes[gates[0]]
                    )
                    ok_p, _ = self._shape_ok(
                        a_tensor, proj_b, module_shapes[projs[0]]
                    )
                    if ok_g and ok_p:
                        if gates[0] in resolved or projs[0] in resolved:
                            failed_count += 1
                            if len(failed_examples) < 8:
                                failed_examples.append(
                                    f"{core}: gate/proj 目标重复"
                                )
                            continue
                        resolved[gates[0]] = (a_tensor, gate_b, alpha)
                        resolved[projs[0]] = (a_tensor, proj_b, alpha)
                        matched_split += 1
                        continue

                failed_count += 1
                if len(failed_examples) < 8:
                    failed_examples.append(
                        f"{core}: fused gate_up 拆分失败或目标不唯一"
                    )
                continue

            if (
                "img_mlp.gate_layer" in core
                or "img_mlp.proj" in core
                or "img_mlp.up_proj" in core
            ):
                if "img_mlp.gate_layer" in core:
                    parent = core.split(".img_mlp.gate_layer")[0] + ".img_mlp"
                    part = "gate"
                elif "img_mlp.proj" in core:
                    parent = core.split(".img_mlp.proj")[0] + ".img_mlp"
                    part = "proj"
                else:
                    parent = core.split(".img_mlp.up_proj")[0] + ".img_mlp"
                    part = "proj"

                target_core = parent + ".gate_up"
                targets = self._suffix_matches(
                    target_core, module_shapes.keys()
                )
                if len(targets) == 1:
                    target = targets[0]
                    bucket = pending_gate if part == "gate" else pending_proj
                    if target in bucket:
                        failed_count += 1
                        if len(failed_examples) < 8:
                            failed_examples.append(
                                f"{core}: 分离 gate/proj 存在重复目标"
                            )
                        continue
                    bucket[target] = (a_tensor, b_tensor, alpha)
                    continue

            failed_count += 1
            if len(failed_examples) < 8:
                failed_examples.append(core)

        # 处理合并 / 分离降级逻辑
        for target in set(pending_gate) & set(pending_proj):
            a_gate, b_gate, alpha_gate = pending_gate[target]
            a_proj, b_proj, alpha_proj = pending_proj[target]

            shape_ok = (a_gate.shape == a_proj.shape)
            value_ok = torch.allclose(a_gate.float(), a_proj.float(), atol=1e-4, rtol=1e-4) if shape_ok else False
            alpha_ok = (alpha_gate == alpha_proj)
            ndim_ok = (b_gate.ndim == 2 and b_proj.ndim == 2)
            dim_ok = (b_gate.shape[1] == b_proj.shape[1]) if ndim_ok else False

            if shape_ok and alpha_ok and ndim_ok and dim_ok:
                final_b = torch.cat((b_gate, b_proj), dim=0)
                ok, reason = self._shape_ok(
                    a_gate, final_b, module_shapes[target]
                )
                if ok and target not in resolved:
                    if value_ok:
                        # A 矩阵一致，完美合并
                        resolved[target] = (a_gate, final_b, alpha_gate)
                    else:
                        # A 矩阵不一致，存入分离模式元组：5元组
                        resolved[target] = (a_gate, b_gate, a_proj, b_proj, alpha_gate)
                        print(f"[QwenHookLora] ⚠️ {target}: A 矩阵数值不一致，采用分离旁路注入")
                    matched_merged += 1
                    continue
                else:
                    reason = f"合并后形状校验失败: {reason}"
            else:
                reasons = []
                if not shape_ok:
                    reasons.append(f"A形状不同:{a_gate.shape}vs{a_proj.shape}")
                if not alpha_ok:
                    reasons.append(f"Alpha不同:{alpha_gate}vs{alpha_proj}")
                if not ndim_ok:
                    reasons.append("B不是二维矩阵")
                elif not dim_ok:
                    reasons.append(f"B矩阵列数不同:{b_gate.shape[1]}vs{b_proj.shape[1]}")
                reason = " | ".join(reasons)

            failed_count += 1
            if len(failed_examples) < 8:
                failed_examples.append(
                    f"{target}: 合并失败（{reason}）"
                )

        unmatched_split = (set(pending_gate) ^ set(pending_proj))
        if unmatched_split:
            failed_count += len(unmatched_split)
            for target in list(unmatched_split)[:8 - len(failed_examples)]:
                failed_examples.append(
                    f"{target}: 只找到 gate 或 proj，缺少配对分支"
                )

        print(
            f"[QwenHookLora] 匹配: direct={matched_direct} "
            f"split={matched_split} merged={matched_merged} failed={failed_count}"
        )
        print(
            "[QwenHookLora] Alpha 来源: "
            f"layer={alpha_sources['layer']} "
            f"metadata={alpha_sources['metadata']} "
            f"missing={alpha_sources['missing']}"
        )
        if global_alpha is not None:
            print(
                f"[QwenHookLora] 全局 metadata alpha: "
                f"{global_alpha} ({global_alpha_key})"
            )
        if failed_examples:
            print(f"[QwenHookLora] 匹配异常样例: {failed_examples}")

        if not resolved:
            print("[QwenHookLora] 没有可应用的 LoRA 层，返回原模型。")
            print("[QwenHookLora] ========================================\n")
            return (model,)

        m = model.clone()
        dm_clone = m.model.diffusion_model
        manager = BypassInjectionManager()
        AdapterClass = self._make_adapter_class()

        fused_names = {
            name for name in resolved
            if ".img_mlp.out" in name or name.endswith("img_mlp.out")
        }
        milestones = sorted({1, 32, 64, 128, len(resolved)})
        execution_tracker = {
            "lock": threading.Lock(),
            "calls": 0,
            "seen": set(),
            "fused_names": fused_names,
            "fused_seen": set(),
            "expected_layers": len(resolved),
            "expected_fused": len(fused_names),
            "milestones": milestones,
            "next_milestone": 0,
        }

        adapter_count = 0
        for name, val in resolved.items():
            if len(val) == 3:
                a_tensor, b_tensor, alpha = val
                effective_alpha = float(alpha) if alpha is not None else float(a_tensor.shape[0])
                adapter = AdapterClass(
                    down=a_tensor,
                    up=b_tensor,
                    alpha=effective_alpha,
                    layer_name=name,
                    execution_tracker=execution_tracker,
                    is_split=False,
                )
            elif len(val) == 5:
                a_gate, b_gate, a_proj, b_proj, alpha = val
                effective_alpha = float(alpha) if alpha is not None else float(a_gate.shape[0])
                adapter = AdapterClass(
                    down=(a_gate, a_proj),
                    up=(b_gate, b_proj),
                    alpha=effective_alpha,
                    layer_name=name,
                    execution_tracker=execution_tracker,
                    is_split=True,
                )
            manager.add_adapter(name, adapter, strength=float(strength))
            adapter_count += 1

        injections = manager.create_injections(dm_clone)
        hook_count = manager.get_hook_count()

        if hook_count != adapter_count:
            raise RuntimeError(
                f"Bypass 注入准备失败：LoRA 目标 {adapter_count} 层，"
                f"ComfyUI Bypass manager 仅创建 {hook_count} 个 Hook。"
                "为避免部分 LoRA 静默丢失，已中止加载。"
            )

        if not injections:
            raise RuntimeError(
                "ComfyUI Bypass manager 没有生成注入项，已中止加载。"
            )

        m.set_injections("qwen_hook_lora", injections)

        fused_out_count = sum(
            1
            for name in resolved
            if ".img_mlp.out" in name or name.endswith("img_mlp.out")
        )
        print(
            f"[QwenHookLora] Bypass 注入准备完成: "
            f"{hook_count}/{adapter_count} 层"
        )
        print(
            f"[QwenHookLora] fused img_mlp.out: {fused_out_count} 层；"
            "由官方 BypassInjectionManager 的 comfy_force_forward 路径处理"
        )
        print(
            "[QwenHookLora] LoRA 因子保留原始存储精度；"
            "旁路矩阵计算 FP32；不修改量化底模权重"
        )
        print("[QwenHookLora] 全局 Monkey Patch: OFF")
        print("[QwenHookLora] ========================================\n")
        return (m,)


NODE_CLASS_MAPPINGS = {
    "QwenHookLora": QwenHookLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenHookLora": "Qwen-Image LoRA (Qwen Bypass)",
}