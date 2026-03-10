"""Auto-generate a structured findings report from benchmark results.

Reads the latest results and produces a Markdown report suitable for
grading and presentation.

Usage:
    python -m analysis.findings_report --results_dir results
"""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path
from typing import Any

import numpy as np

from analysis.load_results import (
    group_by,
    load_aggregate_csv,
    load_decomp_csvs,
    load_manifests,
)
from analysis.report_tables import (
    inflection_table,
    kv_quant_comparison_table,
    regime_summary_table,
    summary_table,
)
from bench.methodology import methodology_text
from bench.utils.io import atomic_write


def _load_csv_safe(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def generate_report(results_dir: str = "results", system_name: str = "") -> str:
    """Generate the full findings report as a markdown string."""
    rd = Path(results_dir)
    rows = load_aggregate_csv(results_dir)
    manifests = load_manifests(results_dir)
    decomps = load_decomp_csvs(results_dir)
    inflections = _load_csv_safe(rd / "summary" / "inflections_sweep.csv")
    regime_data = []
    for f in sorted((rd / "summary").glob("regime_map_*.csv")):
        regime_data.extend(_load_csv_safe(f))
    predictor_coeffs = _load_csv_safe(rd / "summary" / "predictor_coeffs.csv")
    bandwidth_data = _load_csv_safe(rd / "summary" / "device_bandwidth.csv")

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sections: list[str] = []

    # Title
    title = "# TokenScope Findings Report"
    if system_name:
        title += f" — {system_name}"
    sections.append(title + "\n")
    if system_name:
        sections.append(f"**System:** `{system_name}`\n")
    sections.append(f"_Generated: {ts}_\n")

    # Executive Summary
    sections.append("## Executive Summary\n")
    key_findings = _extract_key_findings(rows, inflections, decomps, regime_data)
    for finding in key_findings:
        sections.append(f"- {finding}")
    sections.append("")

    # Methodology
    sections.append(methodology_text())

    # Scaling Results
    sections.append("## Scaling Analysis\n")
    if rows:
        sections.append("### Per-Token Latency vs. Context Length\n")
        sections.append("![Scaling](../figures/scaling_per_token.png)\n")
        sections.append("### TTFT vs. Context Length\n")
        sections.append("![TTFT](../figures/ttft_scaling.png)\n")
        sections.append("### Throughput\n")
        sections.append("![Throughput](../figures/throughput_scaling.png)\n")

    # Inflection Points
    sections.append("### Inflection Points\n")
    sections.append(inflection_table(inflections))

    # Token Trace
    if (rd / "figures" / "token_trace.png").exists():
        sections.append("### Per-Token Latency Trace\n")
        sections.append("![Token Trace](../figures/token_trace.png)\n")

    # Decomposition
    sections.append("## Latency Decomposition\n")
    if decomps:
        for d in decomps:
            sections.append(f"### {d['file']}\n")
            sections.append(summary_table(d["components"]))
            fig_name = d["file"].replace(".csv", ".png").replace("decomp_", "decomp_stacked_")
            if (rd / "figures" / fig_name).exists():
                sections.append(f"\n![Decomposition](../figures/{fig_name})\n")
    else:
        sections.append("_Run `python -m profiling.decompose_decode` to generate._\n")

    # Bottleneck Narrative
    sections.append("## Architectural Bottleneck Analysis\n")

    sections.append("### What Dominates Where\n")
    if decomps:
        for d in decomps:
            comps = d["components"]
            top = max(comps, key=lambda c: float(c.get("pct_of_total", 0)))
            sections.append(
                f"- In `{d['file']}`, **{top['component']}** dominates at "
                f"{top.get('pct_of_total', '?')}% of total decode time.\n"
            )

    if bandwidth_data:
        sections.append("### Device Bandwidth\n")
        sections.append(summary_table(bandwidth_data))

    # KV Cache Size
    sections.append("### KV Cache Size Scaling\n")
    sections.append("![KV Cache](../figures/kv_cache_size.png)\n")

    # Roofline
    if (rd / "figures" / "roofline.png").exists():
        sections.append("### Roofline Analysis\n")
        sections.append("![Roofline](../figures/roofline.png)\n")

    # Predictor
    if predictor_coeffs:
        sections.append("### Latency Predictor Fit\n")
        sections.append(summary_table(predictor_coeffs))
        for c in predictor_coeffs:
            label = c.get("label", "default")
            fig = rd / "figures" / f"predictor_{label}.png"
            if fig.exists():
                sections.append(f"\n![Predictor](../figures/predictor_{label}.png)\n")

    # Regime Map
    if regime_data:
        sections.append("### Regime Classification\n")
        sections.append(regime_summary_table(regime_data))
        if (rd / "figures" / "regime_map.png").exists():
            sections.append("\n![Regime Map](../figures/regime_map.png)\n")

    # Speculative decoding (Bonus)
    spec_baseline = [r for r in rows if "loop_decode" in str(r.get("backend", ""))]
    spec_decode = [r for r in rows if "spec_decode" in str(r.get("backend", ""))]
    if spec_baseline and spec_decode:
        sections.append("## Speculative Decoding Comparison (Bonus)\n")
        b_pt = [r.get("per_token_mean_ms") for r in spec_baseline if r.get("per_token_mean_ms")]
        s_pt = [r.get("per_token_mean_ms") for r in spec_decode if r.get("per_token_mean_ms")]
        if b_pt and s_pt:
            speedup = np.mean(b_pt) / max(np.mean(s_pt), 1e-9)
            sections.append(f"Speculative decoding achieves **{speedup:.2f}×** speedup vs. baseline.\n")
        if (rd / "figures" / "spec_decode_comparison.png").exists():
            sections.append("\n![Spec Decode](../figures/spec_decode_comparison.png)\n")
        sections.append("")

    # Optimization (KV-cache quant)
    sections.append("## Optimization: KV-Cache Quantization\n")
    kv_rows = [r for r in rows if r.get("kv_type_k") != r.get("kv_type_v", "?")
               or r.get("kv_type_k") != "f16"]
    if kv_rows or (rd / "figures" / "kv_quant_comparison.png").exists():
        if kv_rows:
            sections.append(kv_quant_comparison_table(kv_rows))
        sections.append("\n![KV Quant](../figures/kv_quant_comparison.png)\n")
        sections.append(
            "KV-cache quantization reduces memory footprint linearly with precision. "
            "At long contexts where KV cache access dominates, lower precision (q8_0, q4_0) "
            "reduces bandwidth pressure and improves decode latency. At short contexts, "
            "the overhead of dequantization may offset gains, placing the workload in the "
            "overhead-bound regime.\n"
        )
    else:
        sections.append(
            "_Run `python -m bench.sweep --config configs/sweep_kv_cache.yaml` "
            "to generate KV-cache quantization results._\n"
        )

    # Hardware Recommendations
    sections.append("## What Would I Change in Hardware?\n")
    sections.append(
        "Based on the analysis, autoregressive decode at batch size 1 is overwhelmingly "
        "memory-bandwidth-bound. The following hardware changes would yield the largest "
        "improvements:\n\n"
        "1. **Higher memory bandwidth** — HBM3/HBM3E provides 2-3× the bandwidth of HBM2E. "
        "Since decode is bandwidth-bound, this translates nearly linearly to throughput.\n"
        "2. **Larger on-chip SRAM / L2 cache** — Keeping the KV cache on-chip eliminates "
        "repeated HBM reads. NVIDIA H100 already allocates 50MB of L2, but this is only "
        "sufficient for ~1K tokens with LLaMA-7B.\n"
        "3. **Kernel fusion** — Fusing attention + softmax + value projection into a single "
        "kernel reduces memory round-trips and launch overhead.\n"
        "4. **On-chip KV buffering** — Dedicated scratchpad for KV cache (e.g., in NPU "
        "architectures) would avoid DRAM accesses entirely for moderate context lengths.\n"
        "5. **Reduced-precision compute** — INT8/INT4 for KV cache reduces bytes per element "
        "without requiring full weight quantization.\n"
    )

    # Energy
    energy_files = list((rd / "summary").glob("energy_*.csv"))
    if energy_files:
        sections.append("## Energy per Token (Bonus)\n")
        energy_data = _load_csv_safe(energy_files[0])
        if energy_data:
            sections.append(summary_table(energy_data))
        if (rd / "figures" / "energy_per_token.png").exists():
            sections.append("\n![Energy](../figures/energy_per_token.png)\n")

    # Threats to Validity
    sections.append("## Threats to Validity\n")
    sections.append(
        "1. **Timing synchronization**: MPS timing uses wall-clock with `mps.synchronize()` "
        "barriers, which may include queue delays not present in CUDA event timing.\n"
        "2. **Profiler overhead**: Hook-based decomposition adds non-trivial overhead "
        "(function call + synchronization per module per step). Results should be interpreted "
        "as relative proportions, not absolute values.\n"
        "3. **Thermal throttling**: Extended benchmark runs may trigger thermal throttling, "
        "particularly on laptops. IQR filtering partially mitigates this.\n"
        "4. **OS scheduling jitter**: Background processes can cause latency spikes. "
        "We mitigate via warmup, multiple trials, and IQR filtering.\n"
        "5. **Small model approximation**: Tiny-GPT2 results may not reflect the "
        "architectural bottlenecks of production-scale LLaMA models (7B+).\n"
        "6. **Bandwidth proxy**: numpy/torch copy benchmarks measure achievable bandwidth, "
        "not peak. Actual model inference may achieve higher or lower utilization.\n"
    )

    return "\n".join(sections)


def _extract_key_findings(
    rows: list[dict[str, Any]],
    inflections: list[dict[str, Any]],
    decomps: list[dict[str, Any]],
    regime_data: list[dict[str, Any]],
) -> list[str]:
    """Extract 3-6 key bullet-point findings."""
    findings = []

    if rows:
        import numpy as np

        ttfts = [r.get("ttft_mean_ms", 0) for r in rows if r.get("ttft_mean_ms")]
        pts = [r.get("per_token_mean_ms", 0) for r in rows if r.get("per_token_mean_ms")]
        if ttfts:
            findings.append(
                f"TTFT ranges from {min(ttfts):.1f} ms to {max(ttfts):.1f} ms "
                f"across tested configurations."
            )
        if pts:
            findings.append(
                f"Steady-state per-token latency: {np.median(pts):.2f} ms median "
                f"({1000 / np.median(pts):.0f} tok/s)."
            )

    if inflections:
        inf = inflections[0]
        findings.append(
            f"Inflection point detected at prompt length {inf.get('prompt_length', '?')}: "
            f"slope increases by {inf.get('ratio', '?')}×."
        )

    if decomps:
        top_comp = max(decomps[0]["components"],
                       key=lambda c: float(c.get("pct_of_total", 0)))
        findings.append(
            f"Dominant decode component: {top_comp['component']} "
            f"({top_comp.get('pct_of_total', '?')}% of total)."
        )

    if regime_data:
        regimes = [r.get("regime", "?") for r in regime_data]
        from collections import Counter

        most_common = Counter(regimes).most_common(1)[0]
        findings.append(f"Most common bottleneck regime: {most_common[0]} ({most_common[1]} runs).")

    if not findings:
        findings.append("Run benchmarks and analysis to populate findings.")

    return findings[:6]


def write_report(results_dir: str = "results", system_name: str = "") -> str:
    """Generate and write the findings report."""
    report = generate_report(results_dir, system_name=system_name)

    rd = Path(results_dir) / "report"
    rd.mkdir(parents=True, exist_ok=True)

    latest = rd / "report_latest.md"
    atomic_write(latest, report)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    timestamped = rd / f"report_{ts}.md"
    atomic_write(timestamped, report)

    print(f"[Report] Written: {latest}")
    print(f"[Report] Written: {timestamped}")
    return str(latest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate findings report")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--system", type=str, default=None,
        help="System name to generate report for (prompted if not provided)",
    )
    args = parser.parse_args()

    from bench.utils.system_name import resolve_results_dir

    results_dir, system_name = resolve_results_dir(args.results_dir, cli_system=args.system)
    write_report(results_dir, system_name=system_name)


if __name__ == "__main__":
    main()
