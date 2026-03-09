"""Energy-per-token estimation (bonus).

Uses nvidia-smi power sampling on NVIDIA GPUs. Gracefully skips
if nvidia-smi is unavailable or the platform lacks GPU.

Usage:
    python -m analysis.energy_estimation --results_dir results
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from bench.utils.io import write_csv


def nvidia_smi_available() -> bool:
    """Check if nvidia-smi is available."""
    return shutil.which("nvidia-smi") is not None


def sample_power_watts() -> float | None:
    """Read current GPU power draw via nvidia-smi (watts)."""
    if not nvidia_smi_available():
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            timeout=5,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception:
        return None


def estimate_energy_per_token(
    decode_fn: Any = None,
    n_tokens: int = 32,
    sampling_interval_ms: float = 50,
) -> dict[str, Any]:
    """Estimate energy per token by sampling GPU power during decode.

    If decode_fn is provided, it's called as decode_fn() and power is
    sampled during execution. Otherwise, returns a placeholder.
    """
    if not nvidia_smi_available():
        return {
            "available": False,
            "message": "nvidia-smi not found. Energy estimation requires NVIDIA GPU.",
        }

    power_samples: list[float] = []
    start = time.perf_counter()

    if decode_fn is not None:
        import threading

        stop_sampling = threading.Event()

        def sampler():
            while not stop_sampling.is_set():
                w = sample_power_watts()
                if w is not None:
                    power_samples.append(w)
                time.sleep(sampling_interval_ms / 1000)

        t = threading.Thread(target=sampler, daemon=True)
        t.start()
        decode_fn()
        stop_sampling.set()
        t.join(timeout=2)
    else:
        for _ in range(10):
            w = sample_power_watts()
            if w is not None:
                power_samples.append(w)
            time.sleep(sampling_interval_ms / 1000)

    elapsed_s = time.perf_counter() - start

    if not power_samples:
        return {"available": False, "message": "No power samples collected."}

    avg_power_w = np.mean(power_samples)
    total_joules = avg_power_w * elapsed_s
    joules_per_token = total_joules / max(n_tokens, 1)

    return {
        "available": True,
        "avg_power_w": round(float(avg_power_w), 2),
        "elapsed_s": round(elapsed_s, 3),
        "total_joules": round(float(total_joules), 3),
        "joules_per_token": round(float(joules_per_token), 4),
        "n_tokens": n_tokens,
        "n_power_samples": len(power_samples),
    }


def save_energy_results(
    results: list[dict[str, Any]],
    results_dir: str = "results",
    run_id: str = "latest",
) -> None:
    """Save energy estimation results to CSV."""
    valid = [r for r in results if r.get("available")]
    if not valid:
        print("[Energy] No energy data to save (nvidia-smi unavailable).")
        return

    path = Path(results_dir) / "summary" / f"energy_{run_id}.csv"
    write_csv(path, valid)
    print(f"[Energy] Saved: {path}")


def plot_energy(
    results: list[dict[str, Any]],
    output_stem: str = "results/figures/energy_per_token",
) -> None:
    """Plot joules/token vs prompt length if data is available."""
    valid = [r for r in results if r.get("available") and "prompt_length" in r]
    if not valid:
        print("[Energy] No energy data available for plotting.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from analysis.figure_style import apply_style, save_fig

    apply_style()

    x = [r["prompt_length"] for r in valid]
    y = [r["joules_per_token"] for r in valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "o-", color="#E74C3C", linewidth=2, markersize=8)
    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Energy per Token (Joules)")
    ax.set_title("Energy per Token vs. Context Length")

    save_fig(fig, output_stem)
    print(f"[Energy] Plot saved: {output_stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Energy-per-token estimation")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    result = estimate_energy_per_token()
    if result.get("available"):
        print(f"[Energy] Avg power: {result['avg_power_w']:.1f} W")
        print(f"[Energy] Joules/token: {result['joules_per_token']:.4f}")
        save_energy_results([result], args.results_dir)
    else:
        print(f"[Energy] {result.get('message', 'Not available.')}")


if __name__ == "__main__":
    main()
