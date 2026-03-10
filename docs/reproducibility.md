# Reproducibility Guide

## Quick Start (CPU-only, no model downloads required)

```bash
# Clone and install
git clone <repo-url> && cd tokenscope-llama-latency-lab
pip install -e ".[hf,dev]"

# Run minimal benchmark
python3 -m bench.run_bench \
  --config configs/bench_default.yaml \
  --override backend=hf device=cpu \
  model.id_or_path=sshleifer/tiny-gpt2 \
  generation.output_length=32 generation.prompt_length=64

# Try batched inference (batch_size=2)
python3 -m bench.run_bench \
  --config configs/bench_default.yaml \
  --override backend=hf device=cpu \
  model.id_or_path=sshleifer/tiny-gpt2 \
  generation.output_length=32 generation.prompt_length=64 \
  generation.batch_size=2

# Evaluate speculative decoding (requires a draft model)
python3 -m bench.run_bench \
  --config configs/bench_default.yaml \
  --override backend=hf device=cpu \
  hf.mode=spec_decode \
  hf.spec.draft_model_id=sshleifer/tiny-gpt2 \
  hf.spec.draft_steps=4

# Generate plots and report
python3 -m analysis.make_plots --results_dir results
python3 -m analysis.findings_report --results_dir results
```

## Full Reproduction Checklist

### 1. Environment Setup

```bash
# Python 3.10+ required
python3 --version

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -e ".[all]"
```

### 2. Sequence Length Sweep

```bash
python3 -m bench.sweep --config configs/sweep_sequence.yaml
```

Expected output: `results/summary/agg_latest.csv` with rows for each
prompt length (32, 64, 128, 256, 512).

### 3. Model Size Sweep

```bash
python3 -m bench.sweep --config configs/sweep_models.yaml
```

### 4. Latency Decomposition

```bash
python3 -m profiling.decompose_decode \
  --model sshleifer/tiny-gpt2 --device cpu --n_tokens 16
```

Expected output: `results/summary/decomp_*.csv` and
`results/figures/decomp_stacked_*.png`.

### 5. Analysis Pipeline

```bash
# All plots
python3 -m analysis.make_plots --results_dir results

# Bandwidth micro-benchmark
python3 -m analysis.bandwidth_microbench --device cpu

# Findings report
python3 -m analysis.findings_report --results_dir results
```

### 6. Tests

```bash
python3 -m pytest tests/ -v
```

## With Real LLaMA Models

### llama.cpp + GGUF

```bash
# Install llama-cpp-python
pip install llama-cpp-python

# Download a GGUF model (example: LLaMA-2-7B Q4_K_M)
# Place at /path/to/llama-2-7b-q4_k_m.gguf

python3 -m bench.run_bench \
  --config configs/bench_default.yaml \
  --override backend=llamacpp device=cpu \
  model.id_or_path=/path/to/llama-2-7b-q4_k_m.gguf

# KV-cache quantization sweep
python3 -m bench.sweep --config configs/sweep_kv_cache.yaml \
  --override model.id_or_path=/path/to/llama-2-7b-q4_k_m.gguf
```

### HuggingFace + GPU

```bash
pip install torch transformers accelerate

python3 -m bench.run_bench \
  --config configs/bench_default.yaml \
  --override backend=hf device=cuda \
  model.id_or_path=meta-llama/Llama-2-7b-hf
```

## Verifying Results

### Expected Patterns

1. **TTFT scales super-linearly with prompt length** (quadratic attention).
2. **Per-token latency increases linearly** with context (KV cache growth).
3. **Decomposition shows MLP dominance** for small models; attention grows
   with sequence length.
4. **KV quantization helps at long contexts** (bandwidth-bound regime).

### Comparing Across Runs

The aggregate CSV (`results/summary/agg_latest.csv`) accumulates rows
from all runs. Use the `run_id` column to distinguish experiments.

### Config Hashing

Two runs with the same `config_hash` (first 8 chars of SHA-256 of the
config JSON) used identical settings. This allows reliable A/B comparison.
