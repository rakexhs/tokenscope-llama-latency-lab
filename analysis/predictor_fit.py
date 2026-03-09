"""Fit a simple latency predictor from sweep results.

Model:
    per_token_ms ≈ a * (weight_bytes / BW) + b * (kv_bytes / BW) + c_overhead

Fits coefficients per (device, backend, model) group using least squares.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from analysis.figure_style import PALETTE, apply_style, save_fig
from bench.utils.io import write_csv


def build_features(
    rows: list[dict[str, Any]],
    bandwidth_gb_s: float,
    model_weight_bytes: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build feature matrix X and target vector y for predictor fit.

    Features:
        x0 = weight_bytes_streamed / bandwidth  (ms)
        x1 = kv_bytes_accessed / bandwidth       (ms)
        x2 = 1.0                                 (constant overhead)

    Target:
        y = per_token_mean_ms
    """
    bw_bytes_per_ms = bandwidth_gb_s * 1e9 / 1e3  # bytes/ms

    X_rows = []
    y_vals = []
    for row in rows:
        prompt_len = row.get("prompt_length", 0)
        per_token_ms = row.get("per_token_mean_ms") or row.get("steady_per_token_mean_ms", 0)
        if per_token_ms <= 0 or prompt_len <= 0:
            continue

        # Approximate KV bytes accessed per decode token (all layers, all positions)
        kv_per_token_bytes = row.get("_kv_per_token_bytes", 0)
        if kv_per_token_bytes <= 0:
            # Fallback heuristic: 2 * seq_len * 128 * 2 (K+V, head_dim=128, f16)
            kv_per_token_bytes = 2 * prompt_len * 128 * 2

        weight_time = model_weight_bytes / bw_bytes_per_ms if bw_bytes_per_ms > 0 else 0
        kv_time = kv_per_token_bytes / bw_bytes_per_ms if bw_bytes_per_ms > 0 else 0

        X_rows.append([weight_time, kv_time, 1.0])
        y_vals.append(per_token_ms)

    return np.array(X_rows, dtype=np.float64), np.array(y_vals, dtype=np.float64)


def fit_predictor(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> dict[str, Any]:
    """Fit predictor via least squares. Returns coefficients and error metrics."""
    if len(X) < 3:
        return {"coefficients": [0, 0, 0], "mae": 0, "mape": 0, "r2": 0, "n_samples": len(X)}

    # Non-negative least squares for physical interpretability
    from scipy.optimize import nnls

    coeffs, residual = nnls(X, y)

    y_pred = X @ coeffs
    errors = np.abs(y - y_pred)
    mae = float(errors.mean())
    mape = float((errors / np.maximum(y, 1e-9)).mean() * 100)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        "coefficients": coeffs.tolist(),
        "coeff_labels": ["a_weight", "b_kv", "c_overhead"],
        "mae": round(mae, 4),
        "mape": round(mape, 2),
        "r2": round(float(r2), 4),
        "n_samples": len(X),
        "y_pred": y_pred.tolist(),
    }


def fit_and_save(
    rows: list[dict[str, Any]],
    bandwidth_gb_s: float,
    model_weight_bytes: float,
    results_dir: str = "results",
    label: str = "default",
) -> dict[str, Any]:
    """End-to-end: build features, fit, save coefficients and plot."""
    X, y = build_features(rows, bandwidth_gb_s, model_weight_bytes)
    result = fit_predictor(X, y)

    # Save coefficients
    coeff_row = {
        "label": label,
        "a_weight": result["coefficients"][0],
        "b_kv": result["coefficients"][1],
        "c_overhead": result["coefficients"][2],
        "mae": result["mae"],
        "mape": result["mape"],
        "r2": result["r2"],
        "n_samples": result["n_samples"],
        "bandwidth_gb_s": bandwidth_gb_s,
        "model_weight_bytes": model_weight_bytes,
    }
    coeff_path = Path(results_dir) / "summary" / "predictor_coeffs.csv"
    from bench.utils.io import append_csv

    append_csv(coeff_path, coeff_row)

    # Plot predicted vs measured
    if result["n_samples"] >= 3:
        _plot_predicted_vs_measured(y, np.array(result["y_pred"]), result, label, results_dir)

    return result


def _plot_predicted_vs_measured(
    y_true: NDArray[np.float64],
    y_pred: NDArray[np.float64],
    result: dict[str, Any],
    label: str,
    results_dir: str,
) -> None:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(y_true, y_pred, c=PALETTE[0], s=40, alpha=0.7, edgecolors="white", linewidth=0.5)
    lo = min(y_true.min(), y_pred.min()) * 0.9
    hi = max(y_true.max(), y_pred.max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="Perfect prediction")
    ax.set_xlabel("Measured per-token (ms)")
    ax.set_ylabel("Predicted per-token (ms)")
    ax.set_title(f"Predictor Fit — {label}\nR²={result['r2']:.3f}  MAE={result['mae']:.3f} ms")
    ax.legend()
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    stem = str(Path(results_dir) / "figures" / f"predictor_{label}")
    save_fig(fig, stem)
    print(f"[Predictor] Plot saved: {stem}.png")
