"""Capture machine and software environment for reproducibility."""

from __future__ import annotations

import datetime
import os
import platform
import subprocess
import sys

import psutil

from bench.results_schema import EnvironmentSnapshot


def _safe_import_version(module: str) -> str:
    try:
        mod = __import__(module)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "not installed"


def _gpu_info() -> tuple[str, float]:
    """Return (gpu_name, vram_gb). Best-effort; returns empty on failure."""
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return name, round(vram, 2)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS", 0.0
    except Exception:
        pass
    return "", 0.0


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except Exception:
        return ""


def capture_environment() -> EnvironmentSnapshot:
    """Snapshot current environment into a reproducibility record."""
    gpu_name, vram = _gpu_info()
    mem = psutil.virtual_memory()
    cpu_model = platform.processor() or platform.machine()

    try:
        import cpuinfo  # type: ignore

        cpu_model = cpuinfo.get_cpu_info().get("brand_raw", cpu_model)
    except Exception:
        pass

    return EnvironmentSnapshot(
        os=f"{platform.system()} {platform.release()}",
        python_version=sys.version.split()[0],
        cpu_model=cpu_model,
        cpu_cores=os.cpu_count() or 0,
        ram_gb=round(mem.total / (1024**3), 2),
        gpu_name=gpu_name,
        vram_gb=vram,
        torch_version=_safe_import_version("torch"),
        transformers_version=_safe_import_version("transformers"),
        llama_cpp_version=_safe_import_version("llama_cpp"),
        git_sha=_git_sha(),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
