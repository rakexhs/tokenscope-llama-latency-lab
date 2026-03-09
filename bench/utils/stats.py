"""Robust statistical utilities: IQR filtering, bootstrap CIs, summary stats."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def iqr_filter(data: NDArray[np.float64], k: float = 1.5) -> NDArray[np.float64]:
    """Remove outliers outside [Q1 - k*IQR, Q3 + k*IQR]."""
    if len(data) < 4:
        return data
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    mask = (data >= lo) & (data <= hi)
    filtered = data[mask]
    return filtered if len(filtered) > 0 else data


def bootstrap_ci(
    data: NDArray[np.float64],
    n_boot: int = 5000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Returns (ci_low, ci_high).
    """
    if len(data) < 2:
        val = float(data[0]) if len(data) == 1 else 0.0
        return val, val
    rng = np.random.default_rng(rng_seed)
    boot_means = np.empty(n_boot)
    n = len(data)
    for i in range(n_boot):
        sample = data[rng.integers(0, n, size=n)]
        boot_means[i] = sample.mean()
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boot_means, [alpha * 100, (1 - alpha) * 100])
    return float(lo), float(hi)


def robust_summary(
    values: list[float],
    apply_iqr: bool = True,
) -> dict[str, float]:
    """Compute mean, median, p5, p95, std, and 95% bootstrap CI."""
    arr = np.array(values, dtype=np.float64)
    if apply_iqr:
        arr = iqr_filter(arr)
    if len(arr) == 0:
        return {
            "mean": 0.0, "median": 0.0, "std": 0.0,
            "p5": 0.0, "p95": 0.0, "ci_low": 0.0, "ci_high": 0.0,
            "n": 0,
        }
    ci_low, ci_high = bootstrap_ci(arr)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": len(arr),
    }


def steady_state_latencies(
    per_token_traces: list[list[float]],
    skip: int = 2,
) -> list[float]:
    """Extract steady-state per-token latencies by skipping the first N decode
    tokens from each trial (to exclude warmup/JIT effects within a generation).
    """
    flat: list[float] = []
    for trace in per_token_traces:
        flat.extend(trace[skip:])
    return flat
