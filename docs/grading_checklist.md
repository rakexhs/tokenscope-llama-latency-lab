# Grading Checklist

Maps each rubric requirement to the exact command to run and output files to inspect.

## Goal 1: Benchmark Harness (TTFT, per-token, end-to-end)

| Requirement | Command | Output |
|-------------|---------|--------|
| Benchmark with warmup + trials | `python -m bench.run_bench --config configs/bench_default.yaml --override backend=hf device=cpu model.id_or_path=sshleifer/tiny-gpt2 generation.output_length=32 generation.prompt_length=64` | `results/raw/<run_id>.jsonl` |
| TTFT measurement | (included above) | `ttft_ms` field in JSONL and summary CSV |
| Per-token trace | (included above) | `per_token_ms` array in JSONL |
| End-to-end timing | (included above) | `end_to_end_ms` field |
| Outlier handling (IQR) | Enabled by default | Summary CSV has IQR-filtered stats |
| Timing diagram | N/A (documentation) | `docs/timing_diagram.md` |
| Reproducibility snapshot | (automatic) | `results/report/manifest_<run_id>.json` |

## Goal 2: Latency Decomposition

| Requirement | Command | Output |
|-------------|---------|--------|
| Decompose decode into components | `python -m profiling.decompose_decode --model sshleifer/tiny-gpt2 --device cpu --n_tokens 16` | `results/summary/decomp_<run_id>.csv` |
| Stacked bar chart | (generated automatically) | `results/figures/decomp_stacked_<run_id>.png` |
| torch.profiler analysis | `python -m profiling.torch_profiler_decode --model sshleifer/tiny-gpt2` | `results/summary/torch_profile_ops.csv` + `.md` |

## Goal 3: Scaling Analysis + Inflection Points

| Requirement | Command | Output |
|-------------|---------|--------|
| Sequence length sweep | `python -m bench.sweep --config configs/sweep_sequence.yaml` | `results/summary/agg_latest.csv` |
| Model size sweep | `python -m bench.sweep --config configs/sweep_models.yaml` | `results/summary/agg_latest.csv` |
| Precision sweep | `python -m bench.sweep --config configs/sweep_precision.yaml` | `results/summary/agg_latest.csv` |
| Scaling plots | `python -m analysis.make_plots --results_dir results` | `results/figures/scaling_per_token.png` |
| TTFT scaling | (included above) | `results/figures/ttft_scaling.png` |
| Inflection points | (auto-detected in make_plots) | `results/summary/inflections_sweep.csv` |

## Goal 4: Architectural Bottleneck Reasoning

| Requirement | Command | Output |
|-------------|---------|--------|
| KV cache size model | `python -m analysis.make_plots` | `results/figures/kv_cache_size.png` |
| Bandwidth micro-benchmark | `python -m analysis.bandwidth_microbench --device cpu` | `results/summary/device_bandwidth.csv` |
| Roofline analysis | See `analysis/roofline.py` | `results/figures/roofline.png` |
| Predictor fit | See `analysis/predictor_fit.py` | `results/summary/predictor_coeffs.csv` |
| Regime map | See `analysis/regime_map.py` | `results/figures/regime_map.png` |
| Architecture notes | N/A (documentation) | `docs/architecture_notes.md` |

## Goal 5: KV-Cache Quantization Optimization

| Requirement | Command | Output |
|-------------|---------|--------|
| KV quant sweep | `python -m bench.sweep --config configs/sweep_kv_cache.yaml --override model.id_or_path=/path/to/model.gguf` | `results/summary/agg_latest.csv` |
| Comparison plots | `python -m analysis.make_plots` | `results/figures/kv_quant_comparison.png` |
| Documentation | N/A | `docs/kv_cache_quantization.md` |

## Bonus: Cross-Platform

| Requirement | Command | Output |
|-------------|---------|--------|
| CPU vs GPU comparison | `python -m bench.sweep --config configs/devices_example.yaml` | `results/summary/agg_latest.csv` |

## Bonus: Energy Estimation

| Requirement | Command | Output |
|-------------|---------|--------|
| Energy per token | `python -m analysis.energy_estimation` | `results/summary/energy_*.csv` |

## Auto-Generated Report

| Requirement | Command | Output |
|-------------|---------|--------|
| Findings report | `python -m analysis.findings_report --results_dir results` | `results/report/report_latest.md` |

## Full Pipeline (One-Shot)

```bash
# 1. Install
pip install -e ".[hf,dev]"

# 2. Run benchmarks (pass SYSTEM= to identify this machine)
make bench-cpu SYSTEM=My_Laptop
make sweep-seq SYSTEM=My_Laptop

# 3. Profiling
make decompose SYSTEM=My_Laptop

# 4. Analysis (generates per-system plots and report)
make plots SYSTEM=My_Laptop
make report SYSTEM=My_Laptop

# 5. Tests
make test

# 6. List all systems with results
make systems
```

All results are stored under `results/{system_name}/` for easy cross-platform comparison.
