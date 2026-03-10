"""GPU model forensics: compare GGUF (llama.cpp) vs HF (PyTorch) results.

This script assumes you have already:

- Run a **GGUF GPU pipeline** (e.g. `make full-gpu SYSTEM=RTX4090_GGUF MODEL=...gguf`)
- Run **HF GPU forensics** on the same base model (e.g. `make decompose-gpu` / `make profiler`
  with `SYSTEM=RTX4090_HF` and `MODEL=<hf_id>`)

It then:

- Loads aggregate benchmark results (`agg_latest.csv`) for both systems
- Generates side‑by‑side plots (per‑token, TTFT, throughput vs prompt length)
- Copies key artifacts (reports, figures, KV‑quant plots, HF decomp/profiler)
- Writes a small Markdown report describing the comparison

Usage (from project root):

    python -m analysis.gpu_model_forensics \\
        --results_dir results \\
        --gguf_system RTX4090_GGUF \\
        --hf_system RTX4090_HF
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.figure_style import PALETTE, apply_style, save_fig
from analysis.load_results import group_by


def _normalize_model_id(model_id: str) -> str:
    """Extract a short, comparable model name from full path or HF ID."""
    if not model_id:
        return "unknown"
    # Extract filename for GGUF
    m = re.search(r"([^/\\]+\.gguf)$", str(model_id))
    if m:
        return m.group(1)
    # HF model ID (e.g., sshleifer/tiny-gpt2 -> tiny-gpt2)
    if "/" in str(model_id):
        return str(model_id).split("/")[-1]
    return str(model_id)


def _load_agg(results_dir: Path, system: str) -> list[dict[str, Any]]:
    """Load agg_latest.csv for a single system."""
    path = results_dir / system / "summary" / "agg_latest.csv"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            typed: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    typed[k] = int(v)
                except (ValueError, TypeError):
                    try:
                        typed[k] = float(v)
                    except (ValueError, TypeError):
                        typed[k] = v
            typed["system_name"] = system
            typed["model_short"] = _normalize_model_id(typed.get("model_id", ""))
            rows.append(typed)
    return rows


def _infer_common_model_short(gguf_rows: Iterable[dict[str, Any]], hf_rows: Iterable[dict[str, Any]]) -> str:
    """Try to infer a common short model name between GGUF and HF runs."""
    gguf_models = {r.get("model_short", "") for r in gguf_rows}
    hf_models = {r.get("model_short", "") for r in hf_rows}
    common = (gguf_models & hf_models) - {""}
    if common:
        return sorted(common)[0]
    # Fall back to any non-empty, preferably from HF
    for s in sorted(hf_models):
        if s:
            return s
    for s in sorted(gguf_models):
        if s:
            return s
    return "unknown_model"


def _filter_by_model(rows: list[dict[str, Any]], model_short: str) -> list[dict[str, Any]]:
    if not model_short or model_short == "unknown_model":
        return rows
    return [r for r in rows if r.get("model_short") == model_short]


def _plot_metric_vs_prompt(
    gguf_rows: list[dict[str, Any]],
    hf_rows: list[dict[str, Any]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    """Generic helper to plot a scalar metric vs prompt length for GGUF vs HF."""
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    def _aggregate(rows: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
        by_pl: dict[int, list[float]] = {}
        for r in rows:
            pl = r.get("prompt_length")
            val = r.get(metric_key)
            if pl is None or val is None or val == "":
                continue
            by_pl.setdefault(int(pl), []).append(float(val))
        if not by_pl:
            return [], []
        xs = sorted(by_pl.keys())
        ys = [float(np.mean(by_pl[k])) for k in xs]
        return xs, ys

    x_g, y_g = _aggregate(gguf_rows)
    x_h, y_h = _aggregate(hf_rows)

    if x_g and y_g:
        ax.plot(x_g, y_g, "o-", color=PALETTE[0], label="GGUF (llama.cpp)", linewidth=2, markersize=7)
    if x_h and y_h:
        ax.plot(x_h, y_h, "s-", color=PALETTE[1], label="HF (PyTorch)", linewidth=2, markersize=7)

    ax.set_xlabel("Prompt Length (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(left=0)
    save_fig(fig, str(output_path))


def plot_gpu_model_comparison(
    gguf_rows: list[dict[str, Any]],
    hf_rows: list[dict[str, Any]],
    output_dir: Path,
    model_short: str,
) -> None:
    """Create GGUF vs HF comparison plots for a single model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gguf_rows = _filter_by_model(gguf_rows, model_short)
    hf_rows = _filter_by_model(hf_rows, model_short)

    if not gguf_rows or not hf_rows:
        print("[GPU Forensics] Warning: missing data for one of GGUF/HF; plots may be empty.")

    _plot_metric_vs_prompt(
        gguf_rows,
        hf_rows,
        metric_key="per_token_mean_ms",
        ylabel="Per-token Latency (ms)",
        title=f"GPU Per-Token Latency vs Context — {model_short}",
        output_path=output_dir / "gpu_model_per_token",
    )

    _plot_metric_vs_prompt(
        gguf_rows,
        hf_rows,
        metric_key="ttft_mean_ms",
        ylabel="TTFT (ms)",
        title=f"GPU Time to First Token vs Context — {model_short}",
        output_path=output_dir / "gpu_model_ttft",
    )

    _plot_metric_vs_prompt(
        gguf_rows,
        hf_rows,
        metric_key="throughput_mean_tok_s",
        ylabel="Throughput (tok/s)",
        title=f"GPU Throughput vs Context — {model_short}",
        output_path=output_dir / "gpu_model_throughput",
    )


def copy_gpu_artifacts(results_dir: Path, gguf_system: str, hf_system: str, output_dir: Path) -> None:
    """Copy key per-system artifacts into a unified GPU forensics bundle."""
    gguf_dir = results_dir / gguf_system
    hf_dir = results_dir / hf_system

    gguf_out = output_dir / "gguf_system"
    hf_out = output_dir / "hf_system"
    gguf_out.mkdir(parents=True, exist_ok=True)
    hf_out.mkdir(parents=True, exist_ok=True)

    # Copy GGUF figures + report (benchmarks + KV-cache quant)
    gguf_fig = gguf_dir / "figures"
    if gguf_fig.exists():
        for name in [
            "scaling_per_token",
            "ttft_scaling",
            "throughput_scaling",
            "token_trace",
            "kv_cache_size",
            "kv_quant_comparison",
        ]:
            for ext in (".png", ".pdf"):
                src = gguf_fig / f"{name}{ext}"
                if src.exists():
                    shutil.copy(src, gguf_out / f"{name}{ext}")
        # Copy any decomp plots if present
        for f in gguf_fig.glob("decomp_stacked_*.png"):
            shutil.copy(f, gguf_out / f.name)
        for f in gguf_fig.glob("decomp_stacked_*.pdf"):
            shutil.copy(f, gguf_out / f.name)

    gguf_report = gguf_dir / "report" / "report_latest.md"
    if gguf_report.exists():
        shutil.copy(gguf_report, gguf_out / "report_latest.md")

    # Copy HF decomp, profiler, and report
    hf_fig = hf_dir / "figures"
    if hf_fig.exists():
        for f in hf_fig.glob("decomp_stacked_*.png"):
            shutil.copy(f, hf_out / f.name)
        for f in hf_fig.glob("decomp_stacked_*.pdf"):
            shutil.copy(f, hf_out / f.name)

    hf_summary = hf_dir / "summary"
    if hf_summary.exists():
        for name in ["torch_profile_ops.csv", "torch_profile_ops.md"]:
            src = hf_summary / name
            if src.exists():
                shutil.copy(src, hf_out / name)

    hf_report = hf_dir / "report" / "report_latest.md"
    if hf_report.exists():
        shutil.copy(hf_report, hf_out / "report_latest.md")


def write_gpu_forensics_report(
    gguf_system: str,
    hf_system: str,
    model_short: str,
    output_dir: Path,
) -> None:
    """Small Markdown summary describing the GPU forensics bundle."""
    lines = [
        "# GPU Model Forensics Report",
        "",
        f"**Model:** `{model_short}`",
        "",
        "This folder combines:",
        "",
        f"- **GGUF / llama.cpp GPU results** from system `{gguf_system}` (benchmark + KV-cache quantization).",
        f"- **HF / PyTorch GPU forensics** from system `{hf_system}` (latency decomposition + torch.profiler).",
        "",
        "## Contents",
        "",
        "### Root",
        "",
        "- `gpu_model_per_token.*` — GGUF vs HF per-token latency vs context length.",
        "- `gpu_model_ttft.*` — GGUF vs HF time to first token vs context length.",
        "- `gpu_model_throughput.*` — GGUF vs HF throughput vs context length.",
        "",
        "### `gguf_system/`",
        "",
        "- Selected figures from the GGUF GPU pipeline (scaling, TTFT, throughput, KV cache, KV quant).",
        "- `report_latest.md` — TokenScope report for the GGUF GPU runs.",
        "",
        "### `hf_system/`",
        "",
        "- Latency decomposition stacked plots (`decomp_stacked_*.{png,pdf}`).",
        "- Torch profiler outputs (`torch_profile_ops.{csv,md}`).",
        "- `report_latest.md` — TokenScope report for the HF GPU runs.",
        "",
        "Use these side-by-side artifacts to discuss how runtime, KV-cache behavior, and framework/overhead differ between GGUF (llama.cpp) and HF (PyTorch) for the same base model on GPU.",
        "",
    ]
    path = output_dir / "gpu_model_forensics_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    results_dir: str | Path = "results",
    gguf_system: str = "",
    hf_system: str = "",
) -> None:
    """Entry point for programmatic use."""
    if not gguf_system or not hf_system:
        raise ValueError("gguf_system and hf_system are required.")

    results_dir = Path(results_dir)
    gguf_rows = _load_agg(results_dir, gguf_system)
    hf_rows = _load_agg(results_dir, hf_system)

    if not gguf_rows:
        print(f"[GPU Forensics] Warning: no agg_latest.csv for GGUF system '{gguf_system}'.")
    if not hf_rows:
        print(f"[GPU Forensics] Warning: no agg_latest.csv for HF system '{hf_system}'.")

    model_short = _infer_common_model_short(gguf_rows, hf_rows)

    output_dir = results_dir / "GPU_Model_Forensics" / model_short
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[GPU Forensics] Comparing GGUF='{gguf_system}' vs HF='{hf_system}' for model '{model_short}'.")

    print("[GPU Forensics] Generating comparison plots...")
    plot_gpu_model_comparison(gguf_rows, hf_rows, output_dir, model_short)

    print("[GPU Forensics] Copying per-system artifacts...")
    copy_gpu_artifacts(results_dir, gguf_system, hf_system, output_dir)

    print("[GPU Forensics] Writing summary report...")
    write_gpu_forensics_report(gguf_system, hf_system, model_short, output_dir)

    print(f"[GPU Forensics] Done. Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GGUF vs HF GPU model forensics bundle")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--gguf_system", type=str, required=True, help="SYSTEM name used for GGUF GPU runs")
    parser.add_argument("--hf_system", type=str, required=True, help="SYSTEM name used for HF GPU runs")
    args = parser.parse_args()
    run(results_dir=args.results_dir, gguf_system=args.gguf_system, hf_system=args.hf_system)


if __name__ == "__main__":
    main()

