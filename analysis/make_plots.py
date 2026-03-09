"""Generate all analysis plots from benchmark results.

Usage:
    python -m analysis.make_plots --results_dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.figure_style import COLORS, PALETTE, apply_style, save_fig
from analysis.load_results import extract_series, group_by, load_aggregate_csv


def _detect_inflections(
    x: np.ndarray,
    y: np.ndarray,
    min_segment: int = 2,
) -> list[dict[str, Any]]:
    """Detect slope change inflection points via piecewise linear fit."""
    if len(x) < 4:
        return []

    best_idx = -1
    best_cost = float("inf")

    for split in range(min_segment, len(x) - min_segment):
        x1, y1 = x[:split], y[:split]
        x2, y2 = x[split:], y[split:]

        c1 = np.polyfit(x1, y1, 1)
        c2 = np.polyfit(x2, y2, 1)

        r1 = np.sum((np.polyval(c1, x1) - y1) ** 2)
        r2 = np.sum((np.polyval(c2, x2) - y2) ** 2)
        cost = r1 + r2

        if cost < best_cost:
            best_cost = cost
            best_idx = split

    if best_idx <= 0:
        return []

    slope_before = np.polyfit(x[:best_idx], y[:best_idx], 1)[0]
    slope_after = np.polyfit(x[best_idx:], y[best_idx:], 1)[0]
    ratio = slope_after / slope_before if abs(slope_before) > 1e-9 else float("inf")

    if abs(ratio - 1.0) < 0.15:
        return []

    return [{
        "metric": "per_token_mean_ms",
        "prompt_length": int(x[best_idx]),
        "slope_before": round(float(slope_before), 6),
        "slope_after": round(float(slope_after), 6),
        "ratio": round(float(ratio), 3),
    }]


def plot_scaling(
    rows: list[dict[str, Any]],
    output_dir: str,
    group_key: str = "model_id",
) -> list[dict[str, Any]]:
    """Plot per-token latency vs prompt length, grouped by model/config."""
    apply_style()
    groups = group_by(rows, group_key)
    all_inflections: list[dict[str, Any]] = []

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, (label, group_rows) in enumerate(groups.items()):
        x, y = extract_series(group_rows, "prompt_length", "per_token_mean_ms")
        if len(x) == 0:
            continue

        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, y, "o-", color=color, label=str(label), markersize=7)

        # CI bands
        _, ci_lo = extract_series(group_rows, "prompt_length", "per_token_ci_low_ms")
        _, ci_hi = extract_series(group_rows, "prompt_length", "per_token_ci_high_ms")
        if len(ci_lo) == len(x):
            ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.15)

        # Inflection detection
        inflections = _detect_inflections(x, y)
        for inf in inflections:
            inf["group"] = str(label)
            ax.axvline(inf["prompt_length"], color=color, linestyle=":", alpha=0.6)
            ax.annotate(
                f"↑ slope ×{inf['ratio']:.1f}",
                xy=(inf["prompt_length"], np.interp(inf["prompt_length"], x, y)),
                fontsize=8, color=color,
                xytext=(10, 10), textcoords="offset points",
            )
        all_inflections.extend(inflections)

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Per-token Latency (ms)")
    ax.set_title("Decode Latency Scaling with Context Length")
    ax.legend()
    save_fig(fig, f"{output_dir}/scaling_per_token")

    # Save inflection CSV
    if all_inflections:
        from bench.utils.io import write_csv
        write_csv(f"{output_dir}/../summary/inflections_sweep.csv", all_inflections)

    return all_inflections


def plot_ttft_scaling(rows: list[dict[str, Any]], output_dir: str) -> None:
    """Plot TTFT vs prompt length."""
    apply_style()
    groups = group_by(rows, "model_id")

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (label, group_rows) in enumerate(groups.items()):
        x, y = extract_series(group_rows, "prompt_length", "ttft_mean_ms")
        if len(x) == 0:
            continue
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, y, "s-", color=color, label=str(label), markersize=7)

        _, ci_lo = extract_series(group_rows, "prompt_length", "ttft_ci_low_ms")
        _, ci_hi = extract_series(group_rows, "prompt_length", "ttft_ci_high_ms")
        if len(ci_lo) == len(x):
            ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.15)

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time to First Token vs. Context Length")
    ax.legend()
    save_fig(fig, f"{output_dir}/ttft_scaling")


def plot_throughput(rows: list[dict[str, Any]], output_dir: str) -> None:
    """Plot throughput vs prompt length."""
    apply_style()
    groups = group_by(rows, "model_id")

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (label, group_rows) in enumerate(groups.items()):
        x, y = extract_series(group_rows, "prompt_length", "throughput_mean_tok_s")
        if len(x) == 0:
            continue
        ax.plot(x, y, "^-", color=PALETTE[i % len(PALETTE)], label=str(label), markersize=7)

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Decode Throughput vs. Context Length")
    ax.legend()
    save_fig(fig, f"{output_dir}/throughput_scaling")


def plot_kv_quant_comparison(rows: list[dict[str, Any]], output_dir: str) -> None:
    """Plot KV-cache quantization speedup across context lengths."""
    apply_style()

    kv_configs = {}
    for row in rows:
        key = f"{row.get('kv_type_k', 'f16')}/{row.get('kv_type_v', 'f16')}"
        kv_configs.setdefault(key, []).append(row)

    if len(kv_configs) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for i, (kv_label, group_rows) in enumerate(kv_configs.items()):
        x, y = extract_series(group_rows, "prompt_length", "per_token_mean_ms")
        if len(x) == 0:
            continue
        color = COLORS.get(kv_label.split("/")[0], PALETTE[i % len(PALETTE)])
        ax1.plot(x, y, "o-", color=color, label=f"KV={kv_label}", markersize=7)

    ax1.set_xlabel("Prompt Length (tokens)")
    ax1.set_ylabel("Per-token Latency (ms)")
    ax1.set_title("KV-Cache Quantization: Latency")
    ax1.legend()

    # Speedup relative to f16/f16
    baseline_rows = kv_configs.get("f16/f16", [])
    if baseline_rows:
        bx, by = extract_series(baseline_rows, "prompt_length", "per_token_mean_ms")
        for i, (kv_label, group_rows) in enumerate(kv_configs.items()):
            if kv_label == "f16/f16":
                continue
            gx, gy = extract_series(group_rows, "prompt_length", "per_token_mean_ms")
            common = np.intersect1d(bx, gx)
            if len(common) == 0:
                continue
            b_vals = np.array([by[bx == c][0] for c in common])
            g_vals = np.array([gy[gx == c][0] for c in common])
            speedup = b_vals / np.maximum(g_vals, 1e-9)
            color = COLORS.get(kv_label.split("/")[0], PALETTE[i % len(PALETTE)])
            ax2.plot(common, speedup, "o-", color=color, label=f"KV={kv_label}", markersize=7)

        ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Prompt Length (tokens)")
        ax2.set_ylabel("Speedup vs. f16/f16")
        ax2.set_title("KV-Cache Quantization: Speedup")
        ax2.legend()

    fig.tight_layout()
    save_fig(fig, f"{output_dir}/kv_quant_comparison")


def plot_token_trace(rows: list[dict[str, Any]], output_dir: str) -> None:
    """Plot per-token latency trace (token index vs latency)."""
    apply_style()

    # Load raw JSONL for per-token traces
    from analysis.load_results import load_raw_jsonl
    raw = load_raw_jsonl(str(Path(output_dir).parent))

    traces = [r for r in raw if "per_token_ms" in r and r.get("per_token_ms")]
    if not traces:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    all_traces = [np.array(t["per_token_ms"]) for t in traces[:10]]
    if not all_traces:
        return

    max_len = max(len(t) for t in all_traces)
    for i, trace in enumerate(all_traces[:5]):
        ax.plot(range(len(trace)), trace, alpha=0.3, color=PALETTE[0], linewidth=0.8)

    # Compute p50/p95 across trials
    padded = np.full((len(all_traces), max_len), np.nan)
    for i, t in enumerate(all_traces):
        padded[i, :len(t)] = t

    p50 = np.nanmedian(padded, axis=0)
    p95 = np.nanpercentile(padded, 95, axis=0)
    x_idx = np.arange(max_len)

    ax.plot(x_idx, p50, color=PALETTE[0], linewidth=2, label="p50")
    ax.plot(x_idx, p95, color=PALETTE[1], linewidth=1.5, linestyle="--", label="p95")
    ax.fill_between(x_idx, p50, p95, color=PALETTE[1], alpha=0.1)

    ax.set_xlabel("Token Index (decode step)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Per-Token Latency Trace")
    ax.legend()
    save_fig(fig, f"{output_dir}/token_trace")


def plot_kv_cache_size(output_dir: str) -> None:
    """Plot KV cache size vs sequence length for known architectures."""
    from analysis.kv_cache_model import CACHE_SIZES, KNOWN_ARCHITECTURES, kv_cache_curve

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    archs_to_plot = ["llama-7b", "llama2-70b", "llama3-8b"]
    for i, name in enumerate(archs_to_plot):
        arch = KNOWN_ARCHITECTURES.get(name)
        if arch is None:
            continue
        seq, kv_mb = kv_cache_curve(arch, max_seq=8192)
        ax.plot(seq, kv_mb, "-", color=PALETTE[i], label=arch.name, linewidth=2)

    # Cache size reference lines
    for label, size_bytes in list(CACHE_SIZES.items())[:4]:
        ax.axhline(size_bytes / (1024**2), color="gray", linestyle=":", alpha=0.4)
        ax.text(100, size_bytes / (1024**2) * 1.05, label, fontsize=7, color="gray")

    ax.set_xlabel("Sequence Length (tokens)")
    ax.set_ylabel("KV Cache Size (MB)")
    ax.set_title("KV Cache Memory Footprint vs. Sequence Length")
    ax.legend()
    ax.set_xlim(0, 8192)
    save_fig(fig, f"{output_dir}/kv_cache_size")


def make_all_plots(results_dir: str = "results") -> None:
    """Generate all plots from available results."""
    fig_dir = str(Path(results_dir) / "figures")
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    rows = load_aggregate_csv(results_dir)

    print(f"[Plots] Loaded {len(rows)} result rows from {results_dir}/summary/agg_latest.csv")

    if rows:
        inflections = plot_scaling(rows, fig_dir)
        plot_ttft_scaling(rows, fig_dir)
        plot_throughput(rows, fig_dir)
        plot_kv_quant_comparison(rows, fig_dir)
        plot_token_trace(rows, fig_dir)
        print(f"[Plots] Inflection points found: {len(inflections)}")
    else:
        print("[Plots] No aggregate data found. Run benchmarks first.")

    # Always generate analytical plots
    plot_kv_cache_size(fig_dir)

    print(f"[Plots] All figures saved to {fig_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all analysis plots")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    make_all_plots(args.results_dir)


if __name__ == "__main__":
    main()
