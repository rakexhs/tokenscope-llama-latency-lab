"""Run registry: unique run IDs, config hashing, and manifest writing."""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from bench.results_schema import BenchmarkResult, EnvironmentSnapshot, RunConfig
from bench.utils.io import write_json


def config_hash(config: dict[str, Any]) -> str:
    """Deterministic short hash of a config dict."""
    canonical = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:8]


def make_run_id(config: dict[str, Any], git_sha: str = "") -> str:
    """Generate a unique run ID: timestamp + config hash + optional git SHA."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
    ch = config_hash(config)
    parts = [ts, ch]
    if git_sha:
        parts.append(git_sha[:7])
    return "_".join(parts)


def raw_path(results_dir: str | Path, run_id: str) -> Path:
    return Path(results_dir) / "raw" / f"{run_id}.jsonl"


def summary_path(results_dir: str | Path, run_id: str) -> Path:
    return Path(results_dir) / "summary" / f"bench_{run_id}.csv"


def agg_path(results_dir: str | Path) -> Path:
    return Path(results_dir) / "summary" / "agg_latest.csv"


def manifest_path(results_dir: str | Path, run_id: str) -> Path:
    return Path(results_dir) / "report" / f"manifest_{run_id}.json"


def write_manifest(
    results_dir: str | Path,
    run_id: str,
    config: dict[str, Any],
    env: EnvironmentSnapshot,
    file_paths: dict[str, str],
) -> Path:
    """Write a run manifest linking config, env, and output file paths."""
    manifest = {
        "run_id": run_id,
        "config_hash": config_hash(config),
        "config": config,
        "environment": asdict(env),
        "output_files": file_paths,
    }
    path = manifest_path(results_dir, run_id)
    write_json(path, manifest)
    return path


def save_result(
    result: BenchmarkResult,
    results_dir: str | Path = "results",
) -> dict[str, str]:
    """Persist a BenchmarkResult: raw JSONL, summary CSV, manifest.

    Returns dict of output file paths.
    """
    from bench.utils.io import append_csv, write_jsonl
    from bench.utils.stats import robust_summary, steady_state_latencies

    rd = Path(results_dir)
    run_id = result.config.run_id

    # Raw JSONL
    rp = raw_path(rd, run_id)
    records = [{"config": asdict(result.config), "env": asdict(result.env)}]
    for t in result.trials:
        records.append(asdict(t))
    write_jsonl(rp, records)

    # Summary CSV
    ttfts = [t.ttft_ms for t in result.trials]
    all_per_token = []
    for t in result.trials:
        all_per_token.extend(t.per_token_ms)
    steady = steady_state_latencies(
        [t.per_token_ms for t in result.trials],
        skip=result.config.steady_state_skip,
    )

    ttft_stats = robust_summary(ttfts, apply_iqr=result.config.iqr_filter)
    pt_stats = robust_summary(all_per_token, apply_iqr=result.config.iqr_filter)
    st_stats = robust_summary(steady, apply_iqr=result.config.iqr_filter)
    e2e_vals = [t.end_to_end_ms for t in result.trials]
    tp_vals = [t.throughput_tok_s for t in result.trials]

    import numpy as np

    row = {
        "run_id": run_id,
        "backend": result.config.backend,
        "device": result.config.device,
        "model_id": result.config.model_id,
        "dtype": result.config.dtype,
        "kv_type_k": result.config.kv_type_k,
        "kv_type_v": result.config.kv_type_v,
        "prompt_length": result.config.prompt_length,
        "output_length": result.config.output_length,
        "n_trials": len(result.trials),
        "ttft_mean_ms": ttft_stats["mean"],
        "ttft_median_ms": ttft_stats["median"],
        "ttft_p95_ms": ttft_stats["p95"],
        "ttft_ci_low_ms": ttft_stats["ci_low"],
        "ttft_ci_high_ms": ttft_stats["ci_high"],
        "per_token_mean_ms": pt_stats["mean"],
        "per_token_median_ms": pt_stats["median"],
        "per_token_p95_ms": pt_stats["p95"],
        "per_token_ci_low_ms": pt_stats["ci_low"],
        "per_token_ci_high_ms": pt_stats["ci_high"],
        "e2e_mean_ms": float(np.mean(e2e_vals)) if e2e_vals else 0.0,
        "throughput_mean_tok_s": float(np.mean(tp_vals)) if tp_vals else 0.0,
        "steady_per_token_mean_ms": st_stats["mean"],
        "steady_per_token_ci_low_ms": st_stats["ci_low"],
        "steady_per_token_ci_high_ms": st_stats["ci_high"],
    }

    sp = summary_path(rd, run_id)
    from bench.utils.io import write_csv

    write_csv(sp, [row])
    append_csv(agg_path(rd), row)

    # Manifest
    paths = {"raw": str(rp), "summary": str(sp), "aggregate": str(agg_path(rd))}
    mp = write_manifest(rd, run_id, asdict(result.config), result.env, paths)
    paths["manifest"] = str(mp)

    return paths
