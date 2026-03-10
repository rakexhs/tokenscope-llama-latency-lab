# Grading Checklist

Maps each rubric requirement to the exact command to run and output files to inspect.

## Goal 1: Benchmark Harness (TTFT, per-token, end-to-end)

| Requirement | Command | Output |
|-------------|---------|--------|
| Benchmark with warmup + trials | `python3 -m bench.run_bench --config configs/bench_default.yaml --override backend=hf device=cpu model.id_or_path=sshleifer/tiny-gpt2 generation.output_length=32 generation.prompt_length=64` | `results/raw/<run_id>.jsonl` |
| Batched decode support | Pass `generation.batch_size=2` (≤4) via config or override to generate multiple sequences concurrently. | Results reflect per-token latency averaged across the batch |
| TTFT measurement | (included above) | `ttft_ms` field in JSONL and summary CSV |
| Per-token trace | (included above) | `per_token_ms` array in JSONL |
| End-to-end timing | (included above) | `end_to_end_ms` field |
| Outlier handling (IQR) | Enabled by default | Summary CSV has IQR-filtered stats |
| Timing diagram | N/A (documentation) | `docs/timing_diagram.md` |
| Reproducibility snapshot | (automatic) | `results/report/manifest_<run_id>.json` |

## Goal 2: Latency Decomposition

| Requirement | Command | Output |
|-------------|---------|--------|
| Decompose decode into components | `python3 -m profiling.decompose_decode --model sshleifer/tiny-gpt2 --device cpu --n_tokens 16` | `results/summary/decomp_<run_id>.csv` |
| Fine-grained attribution (Q/K/V/o_proj, sampling, KV-cache r/w, overhead) | Module hooks identify projection layers (e.g., `q_proj/k_proj/v_proj/o_proj` or GPT-2 `c_attn/c_proj`). Sampling is timed explicitly. KV-cache read/write are estimated from model config + measured device bandwidth (`summary/device_bandwidth.csv`) and, when possible, used to split the attention bucket into `attention_compute` + `kv_cache_read/write`. | Inspect per-component times in `decomp_<run_id>.csv` |
| Stacked bar chart | (generated automatically) | `results/figures/decomp_stacked_<run_id>.png` |
| torch.profiler analysis | `python3 -m profiling.torch_profiler_decode --model sshleifer/tiny-gpt2` | `results/summary/torch_profile_ops.csv` + `.md` |

## Goal 3: Scaling Analysis + Inflection Points

| Requirement | Command | Output |
|-------------|---------|--------|
| Sequence length sweep | `python3 -m bench.sweep --config configs/sweep_sequence.yaml` | `results/summary/agg_latest.csv` |
| Model size sweep | `python3 -m bench.sweep --config configs/sweep_models.yaml` (HF) or `configs/sweep_models_gguf.yaml` (GGUF) | `results/summary/agg_latest.csv` |
| Precision sweep | `python3 -m bench.sweep --config configs/sweep_precision.yaml` | `results/summary/agg_latest.csv` |
| Batch size sweep (optional) | Pass different `generation.batch_size` values in sweep to observe how per-token latency scales with small batches. | `results/summary/agg_latest.csv` |
| Scaling plots | `python3 -m analysis.make_plots --results_dir results` | `results/figures/scaling_per_token.png` |
| TTFT scaling | (included above) | `results/figures/ttft_scaling.png` |
| Inflection points | (auto-detected in make_plots) | `results/summary/inflections_sweep.csv` |

## Goal 4: Architectural Bottleneck Reasoning

| Requirement | Command | Output |
|-------------|---------|--------|
| KV cache size model | `python3 -m analysis.make_plots` | `results/figures/kv_cache_size.png` |
| Bandwidth micro-benchmark | `python3 -m analysis.bandwidth_microbench --device cpu` | `results/summary/device_bandwidth.csv` |
| Roofline analysis | `make plots` (calls `make_plots` which runs roofline) | `results/figures/roofline.png` |
| Predictor fit | `make plots` (runs predictor from agg + bandwidth) | `results/summary/predictor_coeffs.csv`, `figures/predictor_default.png` |
| Regime map | `make plots` (runs regime from predictor coeffs) | `results/figures/regime_map.png`, `summary/regime_map_default.csv` |
| Architecture notes | N/A (documentation) | `docs/architecture_notes.md` |

**Note:** Roofline, predictor, and regime are generated automatically by `make plots` when `device_bandwidth.csv` and `agg_latest.csv` exist. Run `make bandwidth-cpu` (or `bandwidth-gpu`) before `make plots`.

## Goal 5: KV-Cache Quantization Optimization

| Requirement | Command | Output |
|-------------|---------|--------|
| KV quant sweep | `python3 -m bench.sweep --config configs/sweep_kv_cache.yaml --override model.id_or_path=/path/to/model.gguf` | `results/summary/agg_latest.csv` |
| Comparison plots | `python3 -m analysis.make_plots` | `results/figures/kv_quant_comparison.png` |
| Documentation | N/A | `docs/kv_cache_quantization.md` |

## Bonus: Cross-Platform

| Requirement | Command | Output |
|-------------|---------|--------|
| CPU vs GPU comparison | Run benchmarks on multiple machines with different `SYSTEM=`, then `make cross-platform` | `results/Cross-Platform Comp Result/` |
| Speculative decoding | `make sweep-spec` or `python -m bench.sweep --config configs/sweep_spec_decode.yaml` | `agg_latest.csv` with `hf_loop_decode` vs `hf_spec_decode`; `figures/spec_decode_comparison.png` |

## Bonus: Energy Estimation

| Requirement | Command | Output |
|-------------|---------|--------|
| Energy per token | `python3 -m analysis.energy_estimation` | `results/summary/energy_*.csv` |
| Cross-platform energy (optional) | On systems with NVIDIA GPUs, sampling uses `nvidia-smi`.  Energy measurement for CPU/MPS is best obtained via external tools (e.g. `powermetrics` on macOS).  See `analysis/energy_estimation.py`. | Joules per token in CSV |

## Auto-Generated Report

| Requirement | Command | Output |
|-------------|---------|--------|
| Findings report | `python3 -m analysis.findings_report --results_dir results` | `results/report/report_latest.md` |

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
