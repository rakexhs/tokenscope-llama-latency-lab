"""Main benchmark CLI entry point.

Usage:
    python -m bench.run_bench --config configs/bench_default.yaml [--override key=value ...]
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from bench.backends import get_backend
from bench.backends.base import ProgressCallback
from bench.methodology import DEFAULT_CONFIG
from bench.registry import make_run_id, save_result
from bench.results_schema import BenchmarkResult, RunConfig
from bench.utils.env_info import capture_environment
from bench.utils.prompts import make_prompt
from bench.utils.system_name import resolve_results_dir


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _apply_dot_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply key=value overrides with dot-separated paths."""
    config = copy.deepcopy(config)
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        parts = key.split(".")
        target = config
        for p in parts[:-1]:
            if p not in target:
                target[p] = {}
            target = target[p]
        # Auto-cast types
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"  # type: ignore[assignment]
        else:
            try:
                val = int(val)  # type: ignore[assignment]
            except ValueError:
                try:
                    val = float(val)  # type: ignore[assignment]
                except ValueError:
                    pass
        target[parts[-1]] = val
    return config


def load_config(config_path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML config, merge with defaults, apply CLI overrides."""
    with open(config_path) as f:
        file_cfg = yaml.safe_load(f) or {}
    config = _deep_merge(DEFAULT_CONFIG, file_cfg)
    if overrides:
        config = _apply_dot_overrides(config, overrides)
    return config


def _make_token_progress_bar(desc: str, total_tokens: int, position: int) -> tuple[tqdm, ProgressCallback]:
    """Create a nested token-level progress bar and callback."""
    token_bar = tqdm(
        total=total_tokens,
        desc=desc,
        position=position,
        leave=False,
        dynamic_ncols=True,
        disable=total_tokens <= 0,
    )

    def _update(tokens_done: int, total: int) -> None:
        if token_bar.disable:
            return
        token_bar.total = max(total, 1)
        token_bar.n = min(tokens_done, token_bar.total)
        token_bar.refresh()

    return token_bar, _update


def run_benchmark(
    config: dict[str, Any],
    results_dir: str = "results",
    system_name: str = "",
) -> dict[str, str]:
    """Execute a full benchmark run and persist results."""
    env = capture_environment()
    env.system_name = system_name
    run_id = make_run_id(config, env.git_sha)

    backend_name = config["backend"]
    device = config["device"]
    gen_cfg = config["generation"]
    bench_cfg = config["benchmark"]

    prompt_length = gen_cfg["prompt_length"]
    output_length = gen_cfg["output_length"]
    temperature = gen_cfg.get("temperature", 0.0)
    top_p = gen_cfg.get("top_p", 1.0)
    seed = gen_cfg.get("seed", 42)
    batch_size = gen_cfg.get("batch_size", 1)
    warmup_runs = bench_cfg["warmup_runs"]
    trials = bench_cfg["trials"]

    run_config = RunConfig(
        run_id=run_id,
        backend=backend_name,
        device=device,
        model_id=config["model"]["id_or_path"],
        dtype=config.get("hf", {}).get("dtype", "auto") if backend_name == "hf" else "gguf",
        kv_type_k=config.get("llamacpp", {}).get("kv_type_k", "f16"),
        kv_type_v=config.get("llamacpp", {}).get("kv_type_v", "f16"),
        prompt_length=prompt_length,
        output_length=output_length,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        warmup_runs=warmup_runs,
        trials=trials,
        iqr_filter=bench_cfg.get("iqr_filter", True),
        steady_state_skip=bench_cfg.get("steady_state_skip", 2),
        n_threads=config.get("llamacpp", {}).get("n_threads", 0),
        n_gpu_layers=config.get("llamacpp", {}).get("n_gpu_layers", 0),
        batch_size=batch_size,
    )

    print(f"[TokenScope] Run ID: {run_id}")
    print(f"  Backend: {backend_name} | Device: {device}")
    print(f"  Model: {config['model']['id_or_path']}")
    print(f"  Prompt length: {prompt_length} | Output length: {output_length}")
    print(f"  Warmup: {warmup_runs} | Trials: {trials}")

    backend = get_backend(backend_name, config)
    print("[TokenScope] Loading model...")
    backend.load_model()

    # Synthesize prompt
    tokenizer = getattr(backend, "tokenizer", None)
    prompt = make_prompt(prompt_length, tokenizer)
    print(f"  Prompt chars: {len(prompt)}")

    # Warmup
    print(f"[TokenScope] Running {warmup_runs} warmup(s)...")
    for warmup_idx in tqdm(
        range(warmup_runs),
        desc="Warmups",
        disable=warmup_runs < 1,
        dynamic_ncols=True,
        position=0,
    ):
        token_bar, progress_callback = _make_token_progress_bar(
            desc=f"Warmup {warmup_idx + 1}/{warmup_runs}",
            total_tokens=output_length,
            position=1,
        )
        try:
            backend.run_trial(
                prompt,
                output_length,
                trial_idx=-1,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                progress_callback=progress_callback,
                batch_size=batch_size,
            )
        finally:
            token_bar.close()

    # Trials
    print(f"[TokenScope] Running {trials} trial(s)...")
    trial_records = []
    for i in tqdm(range(trials), desc="Trials", disable=trials < 1, dynamic_ncols=True, position=0):
        token_bar, progress_callback = _make_token_progress_bar(
            desc=f"Trial {i + 1}/{trials}",
            total_tokens=output_length,
            position=1,
        )
        try:
            record = backend.run_trial(
                prompt,
                output_length,
                trial_idx=i,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                progress_callback=progress_callback,
                batch_size=batch_size,
            )
            trial_records.append(record)
        finally:
            token_bar.close()

    result = BenchmarkResult(
        config=run_config,
        env=env,
        trials=trial_records,
        metadata={"model_info": backend.model_info()},
    )

    paths = save_result(result, results_dir)
    backend.unload()

    print(f"\n[TokenScope] Results saved:")
    for label, p in paths.items():
        print(f"  {label}: {p}")

    # Quick summary
    import numpy as np

    ttfts = [t.ttft_ms for t in trial_records]
    all_pt = []
    for t in trial_records:
        all_pt.extend(t.per_token_ms)
    if ttfts:
        print(f"\n  TTFT: {np.mean(ttfts):.2f} ms (mean), {np.median(ttfts):.2f} ms (median)")
    if all_pt:
        print(f"  Per-token: {np.mean(all_pt):.2f} ms (mean), {np.median(all_pt):.2f} ms (median)")
        print(f"  Throughput: {1000 / np.mean(all_pt):.1f} tok/s (from mean per-token)")

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TokenScope benchmark harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--results_dir", type=str, default="results", help="Base results directory")
    parser.add_argument(
        "--system", type=str, default=None,
        help="System name for organizing results (e.g. MacBook_Pro_M3). "
             "If not provided, you will be prompted interactively.",
    )
    parser.add_argument(
        "--override", nargs="*", default=[], metavar="KEY=VALUE",
        help="key=value overrides (e.g. --override device=cuda model.id_or_path=foo)",
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    results_dir, system_name = resolve_results_dir(args.results_dir, cli_system=args.system)
    config = load_config(args.config, args.override)
    run_benchmark(config, results_dir, system_name=system_name)


if __name__ == "__main__":
    main()
