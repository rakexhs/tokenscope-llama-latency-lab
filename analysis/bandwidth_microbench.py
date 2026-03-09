"""Memory bandwidth micro-benchmark for empirical bandwidth estimation.

Measures achievable memory bandwidth on the current device to feed into
the roofline model and predictor fits.

Usage:
    python -m analysis.bandwidth_microbench [--device cpu|cuda]
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from bench.utils.io import write_csv


def _measure_numpy_bandwidth(size_mb: int = 256, iterations: int = 10) -> float:
    """Estimate memory bandwidth via numpy array copy (GB/s)."""
    n_bytes = size_mb * 1024 * 1024
    n_elements = n_bytes // 8  # float64
    src = np.random.randn(n_elements)

    # Warmup
    _ = src.copy()

    total_bytes = 0
    start = time.perf_counter_ns()
    for _ in range(iterations):
        dst = src.copy()
        total_bytes += n_bytes * 2  # read + write
    end = time.perf_counter_ns()

    elapsed_s = (end - start) / 1e9
    return (total_bytes / elapsed_s) / (1024**3)  # GB/s


def _measure_torch_bandwidth(device: str, size_mb: int = 256, iterations: int = 10) -> float:
    """Estimate memory bandwidth via torch tensor copy (GB/s)."""
    import torch

    n_bytes = size_mb * 1024 * 1024
    n_elements = n_bytes // 4  # float32

    src = torch.randn(n_elements, device=device)

    # Warmup
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    _ = src.clone()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    total_bytes = 0
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        _ = src.clone()
        total_bytes += n_bytes * 2
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    end = time.perf_counter_ns()

    elapsed_s = (end - start) / 1e9
    return (total_bytes / elapsed_s) / (1024**3)


def measure_bandwidth(
    device: str = "cpu",
    size_mb: int = 256,
    iterations: int = 10,
) -> dict[str, Any]:
    """Measure bandwidth for the given device. Returns results dict."""
    if device.startswith("cuda"):
        try:
            import torch

            bw = _measure_torch_bandwidth(device, size_mb, iterations)
            return {"device": device, "bandwidth_gb_s": round(bw, 2), "method": "torch_clone"}
        except Exception as e:
            print(f"[WARN] CUDA bandwidth test failed: {e}. Falling back to CPU.")

    if device == "mps":
        print("[NOTE] MPS bandwidth: using CPU numpy proxy. See docs for limitations.")

    bw = _measure_numpy_bandwidth(size_mb, iterations)
    method = "numpy_copy"
    return {"device": device, "bandwidth_gb_s": round(bw, 2), "method": method}


def run_bandwidth_suite(
    devices: list[str] | None = None,
    results_dir: str = "results",
) -> list[dict[str, Any]]:
    """Run bandwidth micro-benchmarks for each device and save results."""
    if devices is None:
        devices = ["cpu"]
        try:
            import torch

            if torch.cuda.is_available():
                devices.append("cuda")
        except ImportError:
            pass

    results = []
    for dev in devices:
        print(f"[Bandwidth] Measuring {dev}...")
        r = measure_bandwidth(dev)
        results.append(r)
        print(f"  {dev}: {r['bandwidth_gb_s']:.2f} GB/s ({r['method']})")

    from pathlib import Path

    csv_path = Path(results_dir) / "summary" / "device_bandwidth.csv"
    write_csv(csv_path, results)
    print(f"[Bandwidth] Saved: {csv_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory bandwidth micro-benchmark")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    run_bandwidth_suite([args.device], args.results_dir)


if __name__ == "__main__":
    main()
