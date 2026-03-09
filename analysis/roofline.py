"""Simplified roofline model for autoregressive decode analysis.

Estimates arithmetic intensity for attention and MLP components,
classifies them as compute-bound or memory-bound relative to the
device's roofline ridge point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analysis.figure_style import PALETTE, apply_style, save_fig


@dataclass
class RooflineParams:
    """Device + workload parameters for roofline analysis."""

    peak_compute_gflops: float  # Peak GFLOP/s
    peak_bandwidth_gb_s: float  # Peak memory bandwidth GB/s

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOP/byte) at the roofline ridge."""
        if self.peak_bandwidth_gb_s <= 0:
            return float("inf")
        return self.peak_compute_gflops / self.peak_bandwidth_gb_s


def estimate_attention_ai(
    seq_len: int,
    head_dim: int = 128,
    n_heads: int = 32,
    bytes_per_elem: float = 2.0,
) -> dict[str, float]:
    """Estimate arithmetic intensity for attention during decode.

    During decode (batch=1, generating one new token), the attention for
    each head computes:
      - Q*K^T: (1 x d) @ (d x S) → 2*d*S FLOPs, reads d + d*S elements
      - softmax: ~5*S FLOPs
      - attn*V: (1 x S) @ (S x d) → 2*S*d FLOPs, reads S + S*d elements

    Total FLOPs per head ≈ 4*d*S + 5*S
    Total bytes per head ≈ (2*d*S + d + S*d + S) * bytes_per_elem
                         ≈ 3*d*S * bytes_per_elem (dominant)
    """
    d = head_dim
    s = seq_len
    flops_per_head = 4 * d * s + 5 * s
    bytes_per_head = (3 * d * s + d + s) * bytes_per_elem

    total_flops = flops_per_head * n_heads
    total_bytes = bytes_per_head * n_heads

    ai = total_flops / total_bytes if total_bytes > 0 else 0
    return {
        "component": "attention",
        "flops": total_flops,
        "bytes": total_bytes,
        "arithmetic_intensity": ai,
        "seq_len": seq_len,
    }


def estimate_mlp_ai(
    hidden_dim: int = 4096,
    intermediate_dim: int = 11008,
    bytes_per_elem: float = 2.0,
) -> dict[str, float]:
    """Estimate arithmetic intensity for MLP during decode (batch=1).

    MLP typically has:
      - gate_proj: (1 x H) @ (H x I) → 2*H*I FLOPs, reads H*I weights
      - up_proj:   (1 x H) @ (H x I) → 2*H*I FLOPs, reads H*I weights
      - down_proj: (1 x I) @ (I x H) → 2*I*H FLOPs, reads I*H weights
      Total FLOPs ≈ 6*H*I (for SwiGLU-style MLP)
      Total weight bytes ≈ 3*H*I * bytes_per_elem
    """
    h = hidden_dim
    i = intermediate_dim
    total_flops = 6 * h * i
    total_bytes = 3 * h * i * bytes_per_elem

    ai = total_flops / total_bytes if total_bytes > 0 else 0
    return {
        "component": "mlp",
        "flops": total_flops,
        "bytes": total_bytes,
        "arithmetic_intensity": ai,
    }


def classify_component(
    ai: float,
    roofline: RooflineParams,
) -> str:
    """Classify as COMPUTE-BOUND or MEMORY-BOUND based on roofline ridge point."""
    return "COMPUTE-BOUND" if ai >= roofline.ridge_point else "MEMORY-BOUND"


def roofline_analysis(
    peak_compute_gflops: float = 100.0,
    peak_bandwidth_gb_s: float = 50.0,
    seq_lengths: list[int] | None = None,
    arch_params: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run roofline analysis for attention and MLP at various seq lengths."""
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048]
    if arch_params is None:
        arch_params = {"head_dim": 128, "n_heads": 32, "hidden_dim": 4096,
                       "intermediate_dim": 11008, "bytes_per_elem": 2.0}

    roofline = RooflineParams(peak_compute_gflops, peak_bandwidth_gb_s)
    results = []

    for sl in seq_lengths:
        attn = estimate_attention_ai(
            sl, arch_params["head_dim"], arch_params["n_heads"],
            arch_params["bytes_per_elem"],
        )
        attn["classification"] = classify_component(attn["arithmetic_intensity"], roofline)
        attn["ridge_point"] = roofline.ridge_point
        results.append(attn)

    mlp = estimate_mlp_ai(
        arch_params["hidden_dim"], arch_params["intermediate_dim"],
        arch_params["bytes_per_elem"],
    )
    mlp["classification"] = classify_component(mlp["arithmetic_intensity"], roofline)
    mlp["ridge_point"] = roofline.ridge_point
    results.append(mlp)

    return results


def plot_roofline(
    peak_compute_gflops: float,
    peak_bandwidth_gb_s: float,
    components: list[dict[str, Any]],
    output_stem: str = "results/figures/roofline",
) -> None:
    """Generate a roofline plot with component markers."""
    import matplotlib.pyplot as plt

    apply_style()

    roofline = RooflineParams(peak_compute_gflops, peak_bandwidth_gb_s)
    ridge = roofline.ridge_point

    fig, ax = plt.subplots(figsize=(9, 6))

    # Roofline ceiling
    ai_range = np.logspace(-2, 3, 500)
    perf = np.minimum(peak_compute_gflops, ai_range * peak_bandwidth_gb_s)
    ax.loglog(ai_range, perf, "k-", linewidth=2, label="Roofline ceiling")
    ax.axvline(ridge, color="gray", linestyle="--", alpha=0.5, label=f"Ridge ({ridge:.1f} FLOP/B)")

    # Plot components
    for i, comp in enumerate(components):
        ai = comp["arithmetic_intensity"]
        attainable = min(peak_compute_gflops, ai * peak_bandwidth_gb_s)
        label = comp.get("component", "?")
        if "seq_len" in comp:
            label += f" (S={comp['seq_len']})"
        color = PALETTE[i % len(PALETTE)]
        ax.plot(ai, attainable, "o", color=color, markersize=10, label=label, zorder=5)

    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Attainable Performance (GFLOP/s)")
    ax.set_title("Roofline Model — Decode Components")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0.01, 1000)

    save_fig(fig, output_stem)
    print(f"[Roofline] Saved: {output_stem}.png")
