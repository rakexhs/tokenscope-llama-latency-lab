"""Centralized matplotlib style for publication-quality figures.

All plots in the project should call apply_style() before rendering.
NO seaborn — pure matplotlib with carefully chosen defaults.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Consistent color palette (colorblind-friendly, high contrast)
COLORS = {
    "embedding": "#4C72B0",
    # Fine-grained attention sub-components
    "q_proj": "#1F77B4",  # blue
    "k_proj": "#FF7F0E",  # orange
    "v_proj": "#2CA02C",  # green
    "o_proj": "#D62728",  # red
    "softmax": "#9467BD",  # purple
    "attention": "#DD8452",
    "mlp": "#55A868",
    "layernorm": "#C44E52",
    "lm_head": "#8172B3",
    "overhead": "#937860",
    "sampling": "#DA8BC3",
    "default": "#64B5CD",
    # Sweep / regime colors
    "primary": "#2C3E50",
    "secondary": "#E74C3C",
    "tertiary": "#27AE60",
    "quaternary": "#8E44AD",
    "quinary": "#F39C12",
    # KV quant
    "f16": "#2C3E50",
    "q8_0": "#E74C3C",
    "q4_0": "#27AE60",
    # Regime labels
    "BANDWIDTH-BOUND": "#3498DB",
    "KV-BOUND": "#E74C3C",
    "OVERHEAD-BOUND": "#F39C12",
    "MIXED": "#95A5A6",
}

PALETTE = [
    "#2C3E50", "#E74C3C", "#27AE60", "#8E44AD",
    "#F39C12", "#3498DB", "#1ABC9C", "#E67E22",
]


def apply_style() -> None:
    """Apply consistent matplotlib rcParams across all figures."""
    mpl.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "font.size": 11,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.prop_cycle": mpl.cycler(color=PALETTE),
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "legend.fontsize": 9,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
    })


def save_fig(fig: mpl.figure.Figure, path_stem: str) -> None:
    """Save figure as both PNG and PDF."""
    from pathlib import Path

    p = Path(path_stem)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{path_stem}.png", bbox_inches="tight", dpi=150)
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    plt.close(fig)
