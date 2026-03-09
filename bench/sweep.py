"""Sweep runner: execute parameterized benchmark sweeps from YAML configs.

Usage:
    python -m bench.sweep --config configs/sweep_sequence.yaml
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import yaml

from bench.run_bench import load_config, run_benchmark


def expand_sweep(sweep_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a sweep config into a list of individual run configs.

    The sweep config has a 'base' section (merged with defaults) and a
    'sweep' section mapping dot-paths to lists of values.  The Cartesian
    product of all sweep dimensions is computed.
    """
    base = sweep_config.get("base", {})
    sweep_axes = sweep_config.get("sweep", {})

    if not sweep_axes:
        return [base]

    keys = list(sweep_axes.keys())
    value_lists = [sweep_axes[k] for k in keys]

    configs: list[dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        overrides = [f"{k}={v}" for k, v in zip(keys, combo)]
        # Load from a dummy base to get defaults, then apply overrides
        from bench.run_bench import _apply_dot_overrides, _deep_merge
        from bench.methodology import DEFAULT_CONFIG

        merged = _deep_merge(DEFAULT_CONFIG, base)
        merged = _apply_dot_overrides(merged, overrides)
        configs.append(merged)

    return configs


def run_sweep(
    config_path: str,
    results_dir: str = "results",
) -> list[dict[str, str]]:
    """Run all configs in a sweep and return list of output paths."""
    with open(config_path) as f:
        sweep_config = yaml.safe_load(f) or {}

    configs = expand_sweep(sweep_config)
    total = len(configs)
    print(f"[TokenScope Sweep] {total} configuration(s) to run from {config_path}")

    all_paths: list[dict[str, str]] = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"[Sweep {i}/{total}]")
        print(f"{'='*60}")
        try:
            paths = run_benchmark(cfg, results_dir)
            all_paths.append(paths)
        except Exception as e:
            print(f"[WARN] Sweep config {i} failed: {e}")
            all_paths.append({"error": str(e)})

    print(f"\n[TokenScope Sweep] Completed {len(all_paths)}/{total} runs.")
    print(f"  Aggregate CSV: {results_dir}/summary/agg_latest.csv")
    return all_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="TokenScope sweep runner")
    parser.add_argument("--config", type=str, required=True, help="Path to sweep YAML config")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    run_sweep(args.config, args.results_dir)


if __name__ == "__main__":
    main()
