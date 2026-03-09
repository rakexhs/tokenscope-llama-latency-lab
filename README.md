# TokenScope: Token-Generation Latency Forensics + KV-Cache Architecture Study

**Advanced Computer Architecture Course Project**

TokenScope is an end-to-end performance forensics lab for autoregressive LLM inference. It combines rigorous benchmarking with architectural modeling to measure, attribute, and explain token-generation latency in LLaMA-family models.

---

## Key Capabilities

- **Benchmark Harness** — TTFT, per-token trace, end-to-end timing with warmup, trials, IQR outlier filtering, and bootstrap confidence intervals
- **Latency Decomposition** — Hook-based attribution of decode time to embedding, attention, MLP, LayerNorm, lm_head, and framework overhead
- **Scaling Analysis** — Automated sweeps across sequence length, model size, and precision with inflection-point detection
- **Architectural Modeling** — KV-cache sizing, bandwidth micro-benchmarks, roofline analysis, latency predictor fitting, and bottleneck regime classification
- **KV-Cache Quantization** — Implemented optimization comparing f16/q8_0/q4_0 KV precision with speedup analysis
- **Auto-Generated Report** — Publication-quality figures and a structured findings report from the latest results

## Quick Start (CPU-only, ~2 minutes)

```bash
# Install
pip install -e ".[hf,dev]"

# Run benchmark with tiny model
# You will be prompted: "Enter System Name:" — type a name for this machine
python -m bench.run_bench \
  --config configs/bench_default.yaml \
  --system My_Laptop \
  --override backend=hf device=cpu \
  model.id_or_path=sshleifer/tiny-gpt2 \
  generation.output_length=32 generation.prompt_length=64

# Generate plots and report for that system
python -m analysis.make_plots --system My_Laptop
python -m analysis.findings_report --system My_Laptop
```

## Multi-System Workflow

Every command prompts for a **System Name** to keep each machine's results
cleanly separated. Results are organized into per-system subdirectories:

```
results/
├── MacBook_Pro_M3/          # Your laptop
│   ├── raw/
│   ├── summary/
│   ├── figures/
│   └── report/
├── Lab_RTX4090/             # Lab workstation
│   ├── raw/
│   ├── summary/
│   ├── figures/
│   └── report/
└── Server_A100/             # Cloud GPU
    └── ...
```

### Three ways to set the system name

1. **`--system` flag** (recommended for scripts):
   ```bash
   python -m bench.run_bench --config configs/bench_default.yaml --system Lab_RTX4090
   ```

2. **`TOKENSCOPE_SYSTEM` environment variable**:
   ```bash
   export TOKENSCOPE_SYSTEM=Lab_RTX4090
   python -m bench.run_bench --config configs/bench_default.yaml
   ```

3. **Interactive prompt** (default if neither is set):
   ```
   Enter System Name: Lab_RTX4090
   ```

### Makefile shortcut

```bash
make bench-cpu SYSTEM=MacBook_Pro_M3
make sweep-seq SYSTEM=MacBook_Pro_M3
make plots SYSTEM=MacBook_Pro_M3
make report SYSTEM=MacBook_Pro_M3

# List all systems with saved results
make systems
```

Each system gets its own aggregate CSV, figures, and findings report, making
it straightforward to compare across machines in your survey/presentation.

## Repository Structure

```
tokenscope-llama-latency-lab/
├── bench/                     # Benchmark harness
│   ├── run_bench.py           # Main CLI entry point
│   ├── sweep.py               # Sweep runner for YAML-defined parameter sweeps
│   ├── methodology.py         # Methodology constants and documentation
│   ├── registry.py            # Run ID generation, config hashing, manifest writing
│   ├── results_schema.py      # Canonical dataclasses for all result records
│   ├── backends/
│   │   ├── base.py            # Abstract backend interface
│   │   ├── hf_backend.py      # HuggingFace Transformers (loop_decode + generate)
│   │   └── llamacpp_backend.py # llama.cpp via llama-cpp-python (GGUF, KV quant)
│   └── utils/
│       ├── env_info.py        # Environment snapshot for reproducibility
│       ├── stats.py           # IQR filter, bootstrap CI, robust summary
│       ├── timers.py          # perf_counter_ns, CUDA events, device sync
│       ├── token_tracing.py   # Per-token timestamp recording
│       ├── io.py              # Atomic JSONL/CSV/JSON writes
│       └── prompts.py         # Fixed-token-length prompt synthesis
├── profiling/                 # Latency decomposition
│   ├── decompose_decode.py    # Hook-instrumented decode profiling
│   ├── hf_hooks.py            # Module-level timing hooks
│   ├── torch_profiler_decode.py # torch.profiler operator analysis
│   ├── nsys_instructions.md   # NVIDIA Nsight Systems guide
│   └── perf_cpu_instructions.md # Linux perf guide
├── analysis/                  # Analysis + visualization
│   ├── figure_style.py        # Centralized matplotlib style (no seaborn)
│   ├── load_results.py        # Result loading and aggregation
│   ├── kv_cache_model.py      # Analytical KV cache sizing model
│   ├── bandwidth_microbench.py # Empirical memory bandwidth measurement
│   ├── roofline.py            # Roofline model + component classification
│   ├── predictor_fit.py       # Latency predictor (BW model) fitting
│   ├── regime_map.py          # Bottleneck regime classification
│   ├── energy_estimation.py   # Energy-per-token estimation (nvidia-smi)
│   ├── make_plots.py          # Generate all analysis plots
│   ├── report_tables.py       # Markdown table generators
│   └── findings_report.py     # Auto-generate findings report
├── configs/                   # YAML experiment configurations
│   ├── bench_default.yaml     # Default single-run config
│   ├── sweep_sequence.yaml    # Sequence-length sweep
│   ├── sweep_models.yaml      # Model-size sweep
│   ├── sweep_precision.yaml   # Precision sweep
│   ├── sweep_kv_cache.yaml    # KV-cache quantization sweep
│   └── devices_example.yaml   # Cross-platform comparison
├── docs/                      # Documentation
│   ├── timing_diagram.md      # Mermaid timing diagram
│   ├── methodology.md         # Measurement methodology
│   ├── reproducibility.md     # Reproduction guide
│   ├── architecture_notes.md  # Architectural analysis + TTFT discussion
│   ├── kv_cache_quantization.md # KV-quant experiment design
│   └── grading_checklist.md   # Rubric → command → output mapping
├── results/                   # Output (gitignored except .gitkeep)
│   ├── raw/                   # Per-run JSONL traces
│   ├── summary/               # Aggregated CSVs
│   ├── figures/               # PNG + PDF plots
│   └── report/                # Auto-generated reports + manifests
├── tests/                     # Pytest suite (no model weights needed)
├── notebooks/
│   └── analyze_results.ipynb  # Interactive analysis (optional)
├── pyproject.toml
├── Makefile
└── .github/workflows/ci.yml
```

## Rubric Coverage

| Goal | What | Key Commands |
|------|------|-------------|
| **1** | Benchmark harness | `python -m bench.run_bench --config configs/bench_default.yaml` |
| **2** | Latency decomposition | `python -m profiling.decompose_decode` |
| **3** | Scaling + inflections | `python -m bench.sweep --config configs/sweep_sequence.yaml` |
| **4** | Bottleneck reasoning | `python -m analysis.make_plots` + roofline/regime analysis |
| **5** | KV-cache quantization | `python -m bench.sweep --config configs/sweep_kv_cache.yaml` |
| **Bonus** | Cross-platform | `configs/devices_example.yaml` |
| **Bonus** | Energy estimation | `python -m analysis.energy_estimation` |
| **Bonus** | TTFT optimization | `docs/architecture_notes.md` |

See [`docs/grading_checklist.md`](docs/grading_checklist.md) for a complete mapping.

## Usage Guide

### Single Benchmark Run

```bash
python -m bench.run_bench --config configs/bench_default.yaml \
  --system My_Laptop \
  --override backend=hf device=cpu model.id_or_path=sshleifer/tiny-gpt2
```

### llama.cpp with GGUF Model

```bash
pip install llama-cpp-python

python -m bench.run_bench --config configs/bench_default.yaml \
  --system Lab_RTX4090 \
  --override backend=llamacpp device=cpu \
  model.id_or_path=/path/to/llama-2-7b-q4_k_m.gguf
```

### Parameter Sweeps

```bash
# Sequence length sweep
python -m bench.sweep --config configs/sweep_sequence.yaml --system My_Laptop

# KV-cache quantization sweep (requires GGUF)
python -m bench.sweep --config configs/sweep_kv_cache.yaml --system Lab_RTX4090 \
  --override model.id_or_path=/path/to/model.gguf
```

### Full Analysis Pipeline

```bash
python -m analysis.make_plots --system My_Laptop
python -m analysis.findings_report --system My_Laptop
# Report: results/My_Laptop/report/report_latest.md
```

### Latency Decomposition

```bash
python -m profiling.decompose_decode \
  --model sshleifer/tiny-gpt2 --device cpu --n_tokens 16 --system My_Laptop
```

## Optimization Summary: KV-Cache Quantization

KV-cache quantization reduces the memory bandwidth required per decode
token by storing cached keys and values at lower precision:

| KV Precision | Bytes/elem | Memory Reduction | Expected Speedup |
|-------------|-----------|-----------------|-----------------|
| f16 (baseline) | 2.0 | — | — |
| q8_0 | 1.0 | 2× | Significant at long contexts |
| q4_0 | 0.5 | 4× | Maximum at long contexts |

The optimization is most effective when KV cache access dominates decode
latency (long contexts, small models). At short contexts where weight
streaming dominates, the benefit is minimal.

See [`docs/kv_cache_quantization.md`](docs/kv_cache_quantization.md) for
full experiment design and architectural interpretation.

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests/ -v
```

Tests validate statistical utilities, schema serialization, KV cache math,
and predictor fit correctness — all without requiring model weights.

## License

MIT — see [LICENSE](LICENSE).
