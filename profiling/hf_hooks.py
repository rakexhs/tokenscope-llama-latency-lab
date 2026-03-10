"""PyTorch forward-hook based module timing for HuggingFace models.

Registers hooks on key module types to capture per-component latency
during autoregressive decode steps.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.nn as nn


class HookTimer:
    """Accumulates forward-pass timing for named modules via hooks."""

    def __init__(self) -> None:
        self.records: dict[str, list[float]] = defaultdict(list)
        self._handles: list[torch.utils.hooks.RemovableHook] = []
        self._start_times: dict[str, int] = {}

    def _pre_hook(self, name: str):
        def hook(module: nn.Module, inputs: Any) -> None:
            # Synchronize to attribute time to this module (esp. on CUDA/MPS).
            p = next(module.parameters(), None)
            if p is not None and getattr(p, "is_cuda", False) and torch.cuda.is_available():
                torch.cuda.synchronize()
            if p is not None and getattr(p, "is_mps", False):
                try:
                    if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                except Exception:
                    pass
            self._start_times[name] = time.perf_counter_ns()
        return hook

    def _post_hook(self, name: str):
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            p = next(module.parameters(), None)
            if p is not None and getattr(p, "is_cuda", False) and torch.cuda.is_available():
                torch.cuda.synchronize()
            if p is not None and getattr(p, "is_mps", False):
                try:
                    if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                except Exception:
                    pass
            end = time.perf_counter_ns()
            start = self._start_times.pop(name, end)
            self.records[name].append((end - start) / 1_000_000)
        return hook

    def register(self, model: nn.Module) -> None:
        """Walk the model and register hooks on known component types."""
        component_map = self._classify_modules(model)
        for component_name, modules in component_map.items():
            for mod_name, mod in modules:
                hook_name = f"{component_name}/{mod_name}"
                h1 = mod.register_forward_pre_hook(self._pre_hook(hook_name))
                h2 = mod.register_forward_hook(self._post_hook(hook_name))
                self._handles.extend([h1, h2])

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def summary(self) -> dict[str, dict[str, float]]:
        """Return {component: {mean_ms, total_ms, count}} aggregated by component prefix."""
        import numpy as np

        grouped: dict[str, list[float]] = defaultdict(list)
        for name, times in self.records.items():
            component = name.split("/")[0]
            grouped[component].extend(times)

        result = {}
        for comp, times in grouped.items():
            arr = np.array(times)
            result[comp] = {
                "mean_ms": float(arr.mean()),
                "total_ms": float(arr.sum()),
                "count": len(arr),
            }
        return result

    @staticmethod
    def _classify_modules(model: nn.Module) -> dict[str, list[tuple[str, nn.Module]]]:
        """Classify named modules into architectural component buckets."""
        components: dict[str, list[tuple[str, nn.Module]]] = defaultdict(list)

        for name, mod in model.named_modules():
            lower = name.lower()
            cls_name = type(mod).__name__.lower()
            in_attn_path = ".attn" in lower or ".attention" in lower

            # Fine-grained classification.  Identify Q/K/V projection layers, softmax,
            # as well as traditional high-level components.
            # Projection layers often appear as Linear modules with names ending in q_proj/k_proj/v_proj/o_proj.
            if any(k in lower for k in ("embed", "wte", "wpe")):
                components["embedding"].append((name, mod))
            elif "q_proj" in lower:
                components["q_proj"].append((name, mod))
            elif "k_proj" in lower:
                components["k_proj"].append((name, mod))
            elif "v_proj" in lower:
                components["v_proj"].append((name, mod))
            elif "o_proj" in lower or "out_proj" in lower:
                components["o_proj"].append((name, mod))
            # GPT-2 style fused projections (QKV and output) live under attn.{c_attn,c_proj}
            elif in_attn_path and "c_attn" in lower:
                components["qkv_proj"].append((name, mod))
            elif in_attn_path and lower.endswith("c_proj"):
                components["o_proj"].append((name, mod))
            elif any(k in cls_name for k in ("softmax",)) or "softmax" in lower:
                components["softmax"].append((name, mod))
            elif any(k in cls_name for k in ("attention", "attn")):
                # High-level attention block not otherwise classified
                if "self" in lower or "attn" in lower:
                    components["attention"].append((name, mod))
            elif any(k in cls_name for k in ("mlp", "feedforward", "dense")):
                components["mlp"].append((name, mod))
            elif any(k in cls_name for k in ("layernorm", "rmsnorm", "ln_")):
                components["layernorm"].append((name, mod))
            elif "lm_head" in lower or "output" in lower:
                components["lm_head"].append((name, mod))

        return dict(components)


@contextmanager
def hooked_model(model: nn.Module) -> Generator[HookTimer, None, None]:
    """Context manager that installs timing hooks and yields the timer."""
    timer = HookTimer()
    timer.register(model)
    try:
        yield timer
    finally:
        timer.remove()
