"""Latency decomposition for HF transformer decode steps.

Runs a short decode window (e.g. 16-32 tokens) with hook-based timing
to attribute latency to embedding, attention, MLP, layernorm, lm_head,
and framework overhead (residual).

Usage:
    python -m profiling.decompose_decode --model sshleifer/tiny-gpt2 --device cpu --n_tokens 16
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bench.registry import make_run_id
from bench.utils.env_info import capture_environment
from bench.utils.io import write_csv, write_json
from bench.utils.prompts import make_prompt
from profiling.hf_hooks import HookTimer, hooked_model


def decompose_decode(
    model_id: str,
    device: str = "cpu",
    prompt_length: int = 64,
    n_tokens: int = 16,
    dtype: str = "auto",
    results_dir: str = "results",
) -> dict[str, Any]:
    """Run hook-instrumented decode and return component-level latency breakdown."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from bench.backends.hf_backend import _resolve_dtype

    resolved_dtype = _resolve_dtype(dtype, device)

    print(f"[Decompose] Loading {model_id} ({resolved_dtype}) on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=resolved_dtype)
    model.to(device)
    model.eval()

    prompt = make_prompt(prompt_length, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    print(f"[Decompose] Decoding {n_tokens} tokens with hooks...")

    with torch.inference_mode(), hooked_model(model) as timer:
        # Prefill
        total_start = time.perf_counter_ns()
        outputs = model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past = outputs.past_key_values
        next_token = logits.argmax(dim=-1)

        # Decode loop
        step_times: list[float] = []
        for step in range(n_tokens - 1):
            step_start = time.perf_counter_ns()
            next_input = next_token.unsqueeze(0)
            outputs = model(input_ids=next_input, past_key_values=past, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past = outputs.past_key_values
            next_token = logits.argmax(dim=-1)
            step_end = time.perf_counter_ns()
            step_times.append((step_end - step_start) / 1_000_000)

        total_end = time.perf_counter_ns()

    total_ms = (total_end - total_start) / 1_000_000
    component_summary = timer.summary()

    # Compute overhead (time not accounted for by hooks)
    hooked_total = sum(v["total_ms"] for v in component_summary.values())
    overhead_ms = max(0, total_ms - hooked_total)

    # Build CSV rows
    rows = []
    for comp, stats in component_summary.items():
        rows.append({
            "component": comp,
            "mean_ms": round(stats["mean_ms"], 4),
            "total_ms": round(stats["total_ms"], 4),
            "count": stats["count"],
            "pct_of_total": round(100 * stats["total_ms"] / total_ms, 2) if total_ms > 0 else 0,
        })
    rows.append({
        "component": "overhead",
        "mean_ms": round(overhead_ms / max(n_tokens, 1), 4),
        "total_ms": round(overhead_ms, 4),
        "count": n_tokens,
        "pct_of_total": round(100 * overhead_ms / total_ms, 2) if total_ms > 0 else 0,
    })
    rows.sort(key=lambda r: r["total_ms"], reverse=True)

    # Save
    env = capture_environment()
    run_id = make_run_id({"decompose": model_id, "n_tokens": n_tokens}, env.git_sha)
    csv_path = Path(results_dir) / "summary" / f"decomp_{run_id}.csv"
    write_csv(csv_path, rows)

    # Generate stacked bar plot
    _plot_decomposition(rows, run_id, results_dir)

    print(f"\n[Decompose] Total: {total_ms:.2f} ms for {n_tokens} tokens")
    print(f"  Component breakdown:")
    for r in rows:
        print(f"    {r['component']:15s}  {r['total_ms']:8.2f} ms  ({r['pct_of_total']:5.1f}%)")
    print(f"\n  CSV: {csv_path}")

    return {
        "run_id": run_id,
        "total_ms": total_ms,
        "n_tokens": n_tokens,
        "components": rows,
        "step_times_ms": step_times,
    }


def _plot_decomposition(rows: list[dict], run_id: str, results_dir: str) -> None:
    """Generate a stacked bar chart of component latencies."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from analysis.figure_style import apply_style, COLORS

    apply_style()

    components = [r["component"] for r in rows if r["total_ms"] > 0]
    totals = [r["total_ms"] for r in rows if r["total_ms"] > 0]
    pcts = [r["pct_of_total"] for r in rows if r["total_ms"] > 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS.get(c, COLORS["default"]) for c in components]
    bars = ax.barh(components, totals, color=colors, edgecolor="white", linewidth=0.5)

    for bar, pct in zip(bars, pcts):
        w = bar.get_width()
        ax.text(w + max(totals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Total Time (ms)")
    ax.set_title(f"Latency Decomposition — {run_id}")
    ax.invert_yaxis()

    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"decomp_stacked_{run_id}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Plot: {fig_dir}/decomp_stacked_{run_id}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency decomposition for HF decode")
    parser.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt_length", type=int, default=64)
    parser.add_argument("--n_tokens", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--system", type=str, default=None,
        help="System name for organizing results (prompted if not provided)",
    )
    args = parser.parse_args()

    from bench.utils.system_name import resolve_results_dir

    results_dir, _ = resolve_results_dir(args.results_dir, cli_system=args.system)

    decompose_decode(
        model_id=args.model,
        device=args.device,
        prompt_length=args.prompt_length,
        n_tokens=args.n_tokens,
        dtype=args.dtype,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
