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
            if torch.cuda.is_available() and next(module.parameters(), torch.tensor(0)).is_cuda:
                torch.cuda.synchronize()
            self._start_times[name] = time.perf_counter_ns()
        return hook

    def _post_hook(self, name: str):
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            if torch.cuda.is_available() and next(module.parameters(), torch.tensor(0)).is_cuda:
                torch.cuda.synchronize()
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

            if any(k in lower for k in ("embed", "wte", "wpe")):
                components["embedding"].append((name, mod))
            elif any(k in cls_name for k in ("attention", "attn")):
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
