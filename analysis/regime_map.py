"""Regime classification: categorize runs as BANDWIDTH-BOUND, KV-BOUND,
OVERHEAD-BOUND, or MIXED based on predictor decomposition.

Generates a regime map figure: prompt length on x-axis, color = regime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from analysis.figure_style import COLORS, apply_style, save_fig
from bench.utils.io import write_csv


def classify_regime(
    a_weight: float,
    b_kv: float,
    c_overhead: float,
    weight_time: float,
    kv_time: float,
    threshold: float = 0.5,
) -> str:
    """Classify a single run into a bottleneck regime.

    Uses the predictor's decomposition:
        predicted = a * weight_time + b * kv_time + c_overhead

    If one term contributes > threshold of total → that regime.
    Otherwise → MIXED.
    """
    total = a_weight * weight_time + b_kv * kv_time + c_overhead
    if total <= 0:
        return "MIXED"

    w_frac = (a_weight * weight_time) / total
    kv_frac = (b_kv * kv_time) / total
    oh_frac = c_overhead / total

    if w_frac >= threshold:
        return "BANDWIDTH-BOUND"
    if kv_frac >= threshold:
        return "KV-BOUND"
    if oh_frac >= threshold:
        return "OVERHEAD-BOUND"
    return "MIXED"


def build_regime_map(
    rows: list[dict[str, Any]],
    coefficients: list[float],
    bandwidth_gb_s: float,
    model_weight_bytes: float,
) -> list[dict[str, Any]]:
    """Classify each row and build regime map data."""
    a, b, c = coefficients
    bw_bytes_per_ms = bandwidth_gb_s * 1e9 / 1e3

    results = []
    for row in rows:
        prompt_len = row.get("prompt_length", 0)
        if prompt_len <= 0:
            continue

        kv_per_token_bytes = row.get("_kv_per_token_bytes", 2 * prompt_len * 128 * 2)
        weight_time = model_weight_bytes / bw_bytes_per_ms if bw_bytes_per_ms > 0 else 0
        kv_time = kv_per_token_bytes / bw_bytes_per_ms if bw_bytes_per_ms > 0 else 0

        regime = classify_regime(a, b, c, weight_time, kv_time)

        results.append({
            "prompt_length": prompt_len,
            "regime": regime,
            "weight_frac": a * weight_time / max(a * weight_time + b * kv_time + c, 1e-9),
            "kv_frac": b * kv_time / max(a * weight_time + b * kv_time + c, 1e-9),
            "overhead_frac": c / max(a * weight_time + b * kv_time + c, 1e-9),
            "predicted_ms": a * weight_time + b * kv_time + c,
            "measured_ms": row.get("per_token_mean_ms", 0),
        })

    return results


def plot_regime_map(
    regime_data: list[dict[str, Any]],
    title: str = "Bottleneck Regime Map",
    output_stem: str = "results/figures/regime_map",
) -> None:
    """Generate a color-coded regime map by prompt length."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    apply_style()

    if not regime_data:
        print("[Regime] No data to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), height_ratios=[1, 2])

    # Top: color bar showing regime by prompt length
    prompt_lens = [d["prompt_length"] for d in regime_data]
    regimes = [d["regime"] for d in regime_data]
    regime_colors = [COLORS.get(r, COLORS["MIXED"]) for r in regimes]

    ax1.bar(range(len(prompt_lens)), [1] * len(prompt_lens), color=regime_colors, width=1.0)
    ax1.set_xticks(range(len(prompt_lens)))
    ax1.set_xticklabels(prompt_lens, rotation=45, ha="right")
    ax1.set_yticks([])
    ax1.set_xlabel("Prompt Length (tokens)")
    ax1.set_title(title)

    handles = [mpatches.Patch(color=COLORS[r], label=r)
               for r in ["BANDWIDTH-BOUND", "KV-BOUND", "OVERHEAD-BOUND", "MIXED"]]
    ax1.legend(handles=handles, loc="upper right", fontsize=8)

    # Bottom: stacked area of fractions
    x = list(range(len(regime_data)))
    w_fracs = [d["weight_frac"] for d in regime_data]
    kv_fracs = [d["kv_frac"] for d in regime_data]
    oh_fracs = [d["overhead_frac"] for d in regime_data]

    ax2.stackplot(
        x, w_fracs, kv_fracs, oh_fracs,
        labels=["Weight BW", "KV BW", "Overhead"],
        colors=[COLORS["BANDWIDTH-BOUND"], COLORS["KV-BOUND"], COLORS["OVERHEAD-BOUND"]],
        alpha=0.8,
    )
    ax2.set_xticks(range(len(prompt_lens)))
    ax2.set_xticklabels(prompt_lens, rotation=45, ha="right")
    ax2.set_xlabel("Prompt Length (tokens)")
    ax2.set_ylabel("Fraction of Predicted Latency")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    save_fig(fig, output_stem)
    print(f"[Regime] Saved: {output_stem}.png")


def save_regime_csv(
    regime_data: list[dict[str, Any]],
    results_dir: str = "results",
    label: str = "default",
) -> None:
    path = Path(results_dir) / "summary" / f"regime_map_{label}.csv"
    write_csv(path, regime_data)
    print(f"[Regime] CSV saved: {path}")
