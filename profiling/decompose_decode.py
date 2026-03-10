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
from bench.utils.timers import sync_device
from profiling.hf_hooks import HookTimer, hooked_model


def _load_bandwidth_gb_s(results_dir: str, device: str) -> float | None:
    """Load measured device bandwidth (GB/s) from summary/device_bandwidth.csv if present."""
    import csv

    path = Path(results_dir) / "summary" / "device_bandwidth.csv"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        want = "cuda" if device.startswith("cuda") else device
        for r in rows:
            if str(r.get("device", "")).strip() == want:
                return float(r.get("bandwidth_gb_s", "0") or 0) or None
    except Exception:
        return None
    return None


def _kv_cache_bytes_per_token(model: Any, dtype: torch.dtype) -> dict[str, float]:
    """Estimate KV-cache bytes read/write per decode token from model config."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return {"kv_read_bytes": 0.0, "kv_write_bytes": 0.0}

    n_layers = int(getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "n_layer", 0) or 0)
    n_heads = int(getattr(cfg, "num_attention_heads", 0) or getattr(cfg, "n_head", 0) or 0)
    hidden = int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "n_embd", 0) or 0)
    n_kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or 0) or n_heads
    if n_layers <= 0 or n_heads <= 0 or hidden <= 0:
        return {"kv_read_bytes": 0.0, "kv_write_bytes": 0.0}

    head_dim = hidden // max(n_heads, 1)
    bytes_per_elem = 2.0 if dtype in (torch.float16, torch.bfloat16) else 4.0

    # Write: append one K and V vector per layer per decode token.
    kv_write = 2.0 * n_layers * n_kv_heads * head_dim * bytes_per_elem
    # Read depends on current context length; computed outside.
    return {"kv_read_bytes": 0.0, "kv_write_bytes": kv_write}


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

    # Optional KV-cache estimate based on measured bandwidth (if available).
    bw_gb_s = _load_bandwidth_gb_s(results_dir, device)
    kv_static = _kv_cache_bytes_per_token(model, resolved_dtype)

    # Decode-only measurement: run prefill once, then time decode steps.
    with torch.inference_mode(), hooked_model(model) as timer:
        # Prefill (not part of decode decomposition)
        sync_device(device)
        outputs = model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past = outputs.past_key_values

        # Initialize next token once (sampling time excluded from decomposition)
        next_token = logits.argmax(dim=-1)

        # Discard any hook records from prefill so breakdown reflects decode-only.
        timer.records.clear()
        timer._start_times.clear()  # type: ignore[attr-defined]

        step_times: list[float] = []
        sampling_times: list[float] = []

        # Decode loop: each step is (forward one token) + (sampling).
        for step in range(n_tokens):
            sync_device(device)
            step_start = time.perf_counter_ns()

            next_input = next_token.view(1, 1)
            outputs = model(input_ids=next_input, past_key_values=past, use_cache=True)
            logits = outputs.logits[:, -1, :]
            past = outputs.past_key_values

            # Sampling / decoding logic (separate from module hooks)
            samp_start = time.perf_counter_ns()
            next_token = logits.argmax(dim=-1)
            sync_device(device)
            samp_end = time.perf_counter_ns()

            step_end = time.perf_counter_ns()
            step_times.append((step_end - step_start) / 1_000_000)
            sampling_times.append((samp_end - samp_start) / 1_000_000)

    total_ms = float(np.sum(step_times)) if step_times else 0.0
    sampling_ms = float(np.sum(sampling_times)) if sampling_times else 0.0
    component_summary = timer.summary()

    # Compute overhead (time not accounted for by hooks)
    hooked_total = sum(v["total_ms"] for v in component_summary.values())
    overhead_ms = max(0.0, total_ms - hooked_total - sampling_ms)

    # Build CSV rows
    rows = []
    for comp, stats in component_summary.items():
        rows.append({
            "component": comp,
            "per_token_ms": round(stats["total_ms"] / max(n_tokens, 1), 4),
            "total_ms": round(stats["total_ms"], 4),
            "count": stats["count"],
            "pct_of_total": round(100 * stats["total_ms"] / total_ms, 2) if total_ms > 0 else 0,
            "source": "measured",
        })
    # Sampling / decoding logic
    rows.append({
        "component": "sampling",
        "per_token_ms": round(sampling_ms / max(n_tokens, 1), 4),
        "total_ms": round(sampling_ms, 4),
        "count": n_tokens,
        "pct_of_total": round(100 * sampling_ms / total_ms, 2) if total_ms > 0 else 0,
        "source": "measured",
    })
    rows.append({
        "component": "overhead",
        "per_token_ms": round(overhead_ms / max(n_tokens, 1), 4),
        "total_ms": round(overhead_ms, 4),
        "count": n_tokens,
        "pct_of_total": round(100 * overhead_ms / total_ms, 2) if total_ms > 0 else 0,
        "source": "measured",
    })

    # KV-cache read/write attribution (modeled from bytes and measured bandwidth)
    # We split the measured "attention" bucket (if present) into compute + KV traffic.
    if bw_gb_s is not None and bw_gb_s > 0:
        # Estimate per-step KV bytes read as context grows during decode.
        cfg = getattr(model, "config", None)
        seq0 = int(input_ids.shape[1])
        bytes_per_elem = 2.0 if resolved_dtype in (torch.float16, torch.bfloat16) else 4.0
        n_layers = int(getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "n_layer", 0) or 0) if cfg else 0
        n_heads = int(getattr(cfg, "num_attention_heads", 0) or getattr(cfg, "n_head", 0) or 0) if cfg else 0
        hidden = int(getattr(cfg, "hidden_size", 0) or getattr(cfg, "n_embd", 0) or 0) if cfg else 0
        n_kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or 0) if cfg else 0
        if n_kv_heads <= 0:
            n_kv_heads = n_heads
        head_dim = hidden // max(n_heads, 1) if (hidden and n_heads) else 0

        kv_read_bytes_total = 0.0
        kv_write_bytes_total = float(kv_static.get("kv_write_bytes", 0.0)) * max(n_tokens, 1)
        if n_layers > 0 and n_kv_heads > 0 and head_dim > 0:
            for i in range(n_tokens):
                seq_len = seq0 + i
                kv_read_bytes_total += 2.0 * n_layers * n_kv_heads * head_dim * seq_len * bytes_per_elem

        gb = 1024**3
        kv_read_ms = (kv_read_bytes_total / (bw_gb_s * gb)) * 1000.0 if kv_read_bytes_total > 0 else 0.0
        kv_write_ms = (kv_write_bytes_total / (bw_gb_s * gb)) * 1000.0 if kv_write_bytes_total > 0 else 0.0

        # Replace attention bucket with attention_compute + kv_cache_{read,write}.
        attn_idx = next((i for i, r in enumerate(rows) if r["component"] == "attention" and r.get("source") == "measured"), None)
        if attn_idx is not None:
            attn_total_ms = float(rows[attn_idx]["total_ms"])
            attn_compute_ms = max(0.0, attn_total_ms - kv_read_ms - kv_write_ms)
            rows.pop(attn_idx)
            rows.append({
                "component": "attention_compute",
                "per_token_ms": round(attn_compute_ms / max(n_tokens, 1), 4),
                "total_ms": round(attn_compute_ms, 4),
                "count": n_tokens,
                "pct_of_total": round(100 * attn_compute_ms / total_ms, 2) if total_ms > 0 else 0,
                "source": "derived",
            })

        rows.append({
            "component": "kv_cache_read",
            "per_token_ms": round(kv_read_ms / max(n_tokens, 1), 4),
            "total_ms": round(kv_read_ms, 4),
            "count": n_tokens,
            "pct_of_total": round(100 * kv_read_ms / total_ms, 2) if total_ms > 0 else 0,
            "source": "modeled",
        })
        rows.append({
            "component": "kv_cache_write",
            "per_token_ms": round(kv_write_ms / max(n_tokens, 1), 4),
            "total_ms": round(kv_write_ms, 4),
            "count": n_tokens,
            "pct_of_total": round(100 * kv_write_ms / total_ms, 2) if total_ms > 0 else 0,
            "source": "modeled",
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

    filtered = [r for r in rows if float(r.get("per_token_ms", 0)) > 0]
    components = [r["component"] for r in filtered]
    totals = [float(r.get("per_token_ms", 0)) for r in filtered]
    pcts = [r.get("pct_of_total", 0) for r in filtered]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [COLORS.get(c, COLORS["default"]) for c in components]
    bars = ax.barh(components, totals, color=colors, edgecolor="white", linewidth=0.5)

    for bar, pct in zip(bars, pcts):
        w = bar.get_width()
        ax.text(w + max(totals) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Per-Token Time (ms)")
    ax.set_title(f"Decode Latency Decomposition — {run_id}")
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
