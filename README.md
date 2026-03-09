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
- **Multi-System Support** — Every command tags results by system name so you can benchmark across machines and compare side-by-side

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Required |
| git | Required |
| pip | Required |
| GPU PyTorch | Only for CUDA/MPS runs |
| llama-cpp-python | Only for GGUF model runs |
| make | Pre-installed on macOS/Linux; on Windows use WSL or `pip install make` |

---

## Setup on a New Machine

### 1. Clone the repository

```bash
git clone https://github.com/rakexhs/tokenscope-llama-latency-lab.git
cd tokenscope-llama-latency-lab
```

### 2. Create a virtual environment

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

**Windows (use WSL recommended):**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
```

### 3. Install dependencies

**CPU-only (works everywhere, no GPU needed):**

```bash
make install-cpu
```

**All backends (includes llama-cpp-python for GGUF models):**

```bash
make install
```

### 4. Verify installation

```bash
make test
```

All tests should pass. No model weights are needed.

---

## System Name

Every command uses `SYSTEM=` to tag which machine produced the results.
This keeps each machine's data in its own folder under `results/`.

```
results/
├── MacBook_Pro_M3/        # Your laptop
│   ├── raw/               # Per-run JSONL traces
│   ├── summary/           # Aggregated CSVs
│   ├── figures/           # PNG + PDF plots
│   └── report/            # Findings reports + manifests
├── Lab_RTX4090/           # Lab workstation
│   └── ...
└── Server_A100/           # Cloud GPU
    └── ...
```

If you forget `SYSTEM=`, you will be prompted interactively:

```
Enter System Name: MacBook_Pro_M3
```

To see all systems with saved results:

```bash
make systems
```

---

## Run Everything in One Command

Pick the one that matches your hardware:

```bash
# CPU-only machine (no GPU or GGUF needed)
make full-cpu SYSTEM=My_Machine

# NVIDIA GPU machine (requires a GGUF model file)
make full-gpu SYSTEM=My_Machine MODEL=/path/to/model.gguf

# Apple Silicon Mac
make full-mps SYSTEM=My_Machine
```

Each full pipeline runs: **test → benchmark → sweep → decompose → bandwidth → plots → report**.

- `full-cpu` and `full-mps` use `sshleifer/tiny-gpt2` (auto-downloaded, ~500 KB).
- `full-gpu` uses your GGUF model for benchmarks and sweeps, plus `tiny-gpt2` for HF decomposition.

Your findings report will be at: `results/My_Machine/report/report_latest.md`

> **Note:** `full-gpu` requires `MODEL=`. If you forget it, you'll get a clear error message.

---

## Step-by-Step Commands

If you prefer to run each step individually, here is every command.
Always pass `SYSTEM=Your_Machine_Name` to keep results organized.

### Benchmarks

| Command | What it does |
|---|---|
| `make bench-cpu SYSTEM=X` | CPU benchmark with tiny-gpt2 (works everywhere) |
| `make bench-gpu SYSTEM=X MODEL=/path/to.gguf` | GPU benchmark with a GGUF model via llama.cpp |
| `make bench-mps SYSTEM=X` | Apple Silicon MPS benchmark with tiny-gpt2 |

### Sweeps

| Command | What it does |
|---|---|
| `make sweep-seq SYSTEM=X` | Sequence-length sweep (CPU, tiny-gpt2) |
| `make sweep-seq-gpu SYSTEM=X MODEL=/path/to.gguf` | Sequence-length sweep on GPU (llama.cpp) |
| `make sweep-models SYSTEM=X` | Model-size sweep |
| `make sweep-precision SYSTEM=X` | Precision sweep (fp32 vs. fp16/bf16) |
| `make sweep-kv SYSTEM=X MODEL=/path/to.gguf` | KV-cache quantization sweep (f16/q8_0/q4_0) |

### Profiling

| Command | What it does |
|---|---|
| `make decompose SYSTEM=X` | Latency decomposition on CPU (tiny-gpt2) |
| `make decompose-gpu SYSTEM=X` | Latency decomposition on CUDA GPU (tiny-gpt2, HF) |
| `make profiler SYSTEM=X` | torch.profiler operator-level analysis |
| `make bandwidth-cpu SYSTEM=X` | Memory bandwidth micro-benchmark (CPU) |
| `make bandwidth-gpu SYSTEM=X` | Memory bandwidth micro-benchmark (CUDA) |
| `make bandwidth-mps SYSTEM=X` | Memory bandwidth micro-benchmark (MPS fallback) |
| `make energy SYSTEM=X` | Energy-per-token estimation (NVIDIA only, safe skip) |

### Analysis & Reporting

| Command | What it does |
|---|---|
| `make plots SYSTEM=X` | Generate all figures (scaling, TTFT, throughput, KV cache, etc.) |
| `make report SYSTEM=X` | Generate findings report (Markdown with embedded plots) |
| `make analysis SYSTEM=X` | Both plots + report in one command |

### Utilities

| Command | What it does |
|---|---|
| `make test` | Run the test suite |
| `make lint` | Run ruff linter on all source code |
| `make systems` | List all systems with saved results |
| `make clean` | Delete all generated results |
| `make help` | Show all available commands |

---

## Platform-Specific Guides

### CPU-Only Machine (any OS)

```bash
git clone https://github.com/rakexhs/tokenscope-llama-latency-lab.git
cd tokenscope-llama-latency-lab
python3 -m venv .venv
source .venv/bin/activate
make install-cpu
make full-cpu SYSTEM=My_CPU_Machine
```

Or step by step:

```bash
make test
make bench-cpu SYSTEM=My_CPU_Machine
make sweep-seq SYSTEM=My_CPU_Machine
make decompose SYSTEM=My_CPU_Machine
make bandwidth-cpu SYSTEM=My_CPU_Machine
make plots SYSTEM=My_CPU_Machine
make report SYSTEM=My_CPU_Machine
```

### NVIDIA GPU Machine (CUDA)

```bash
git clone https://github.com/rakexhs/tokenscope-llama-latency-lab.git
cd tokenscope-llama-latency-lab
python3 -m venv .venv
source .venv/bin/activate
make install
make full-gpu SYSTEM=RTX4090_Lab MODEL=/path/to/llama-2-7b-q4_k_m.gguf
```

Or step by step:

```bash
make test
make bench-gpu SYSTEM=RTX4090_Lab MODEL=/path/to/llama-2-7b-q4_k_m.gguf
make sweep-seq-gpu SYSTEM=RTX4090_Lab MODEL=/path/to/llama-2-7b-q4_k_m.gguf
make sweep-kv SYSTEM=RTX4090_Lab MODEL=/path/to/llama-2-7b-q4_k_m.gguf
make decompose-gpu SYSTEM=RTX4090_Lab
make bandwidth-gpu SYSTEM=RTX4090_Lab
make energy SYSTEM=RTX4090_Lab
make plots SYSTEM=RTX4090_Lab
make report SYSTEM=RTX4090_Lab
```

### Apple Silicon Mac (MPS)

```bash
git clone https://github.com/rakexhs/tokenscope-llama-latency-lab.git
cd tokenscope-llama-latency-lab
python3 -m venv .venv
source .venv/bin/activate
make install-cpu
make full-mps SYSTEM=MacBook_Pro_M3
```

Or step by step:

```bash
make test
make bench-mps SYSTEM=MacBook_Pro_M3
make bench-cpu SYSTEM=MacBook_Pro_M3
make sweep-seq SYSTEM=MacBook_Pro_M3
make decompose SYSTEM=MacBook_Pro_M3
make bandwidth-cpu SYSTEM=MacBook_Pro_M3
make plots SYSTEM=MacBook_Pro_M3
make report SYSTEM=MacBook_Pro_M3
```

---

## Using Real LLaMA Models

### GGUF Models via llama.cpp (recommended — lower RAM, runs on CPU too)

Download a GGUF from [HuggingFace](https://huggingface.co/TheBloke), then:

```bash
make install
make bench-gpu SYSTEM=My_Machine MODEL=/path/to/llama-2-7b-q4_k_m.gguf
```

### KV-Cache Quantization Experiments (requires GGUF)

Pass the model path directly via `MODEL=` — no need to edit YAML files:

```bash
make sweep-kv SYSTEM=My_Machine MODEL=/path/to/llama-2-7b-q4_k_m.gguf
make plots SYSTEM=My_Machine
make report SYSTEM=My_Machine
```

### HuggingFace Models (requires access approval + GPU RAM)

```bash
huggingface-cli login
python -m bench.run_bench \
  --config configs/bench_default.yaml \
  --system My_GPU \
  --override backend=hf device=cuda \
  model.id_or_path=meta-llama/Llama-2-7b-hf
```

---

## Where to Find Results

After running benchmarks and analysis for a system named `My_Laptop`:

| What | Path |
|---|---|
| Raw trial data (JSONL) | `results/My_Laptop/raw/*.jsonl` |
| Per-run summary CSV | `results/My_Laptop/summary/bench_*.csv` |
| Aggregate CSV (all runs) | `results/My_Laptop/summary/agg_latest.csv` |
| Decomposition CSV | `results/My_Laptop/summary/decomp_*.csv` |
| Inflection points | `results/My_Laptop/summary/inflections_sweep.csv` |
| Run manifests | `results/My_Laptop/report/manifest_*.json` |
| Findings report | `results/My_Laptop/report/report_latest.md` |
| All plots (PNG + PDF) | `results/My_Laptop/figures/` |

---

## Repository Structure

```
tokenscope-llama-latency-lab/
├── bench/                      # Benchmark harness
│   ├── run_bench.py            #   Main benchmark CLI
│   ├── sweep.py                #   Sweep runner (supports --override)
│   ├── methodology.py          #   Methodology constants
│   ├── registry.py             #   Run ID, config hash, manifests
│   ├── results_schema.py       #   Result dataclasses
│   ├── backends/
│   │   ├── base.py             #   Abstract backend
│   │   ├── hf_backend.py       #   HuggingFace Transformers
│   │   └── llamacpp_backend.py #   llama.cpp (GGUF, KV quant)
│   └── utils/
│       ├── system_name.py      #   System name prompt + directory resolution
│       ├── env_info.py         #   Environment snapshot
│       ├── stats.py            #   IQR filter, bootstrap CI
│       ├── timers.py           #   Timing (perf_counter, CUDA events)
│       ├── token_tracing.py    #   Per-token timestamps
│       ├── io.py               #   Atomic file writes
│       └── prompts.py          #   Prompt synthesis
├── profiling/                  # Latency decomposition
│   ├── decompose_decode.py     #   Hook-based decode profiling
│   ├── hf_hooks.py             #   Module-level timing hooks
│   ├── torch_profiler_decode.py #  torch.profiler analysis
│   ├── nsys_instructions.md    #   NVIDIA Nsight Systems guide
│   └── perf_cpu_instructions.md #  Linux perf guide
├── analysis/                   # Analysis + visualization
│   ├── figure_style.py         #   Matplotlib style (no seaborn)
│   ├── load_results.py         #   Result loading
│   ├── kv_cache_model.py       #   KV cache sizing model
│   ├── bandwidth_microbench.py #   Memory bandwidth measurement
│   ├── roofline.py             #   Roofline model
│   ├── predictor_fit.py        #   Latency predictor fitting
│   ├── regime_map.py           #   Bottleneck classification
│   ├── energy_estimation.py    #   Energy-per-token (nvidia-smi)
│   ├── make_plots.py           #   Plot generation
│   ├── report_tables.py        #   Markdown tables
│   └── findings_report.py      #   Auto-generated report
├── configs/                    # YAML experiment configs
│   ├── bench_default.yaml
│   ├── sweep_sequence.yaml     #   CPU sequence sweep
│   ├── sweep_sequence_gpu.yaml #   GPU sequence sweep (llama.cpp)
│   ├── sweep_models.yaml
│   ├── sweep_precision.yaml
│   ├── sweep_kv_cache.yaml
│   └── devices_example.yaml
├── docs/                       # Documentation
├── results/                    # Output (per-system subdirectories)
├── tests/                      # Pytest suite
├── notebooks/                  # Interactive analysis
├── Makefile                    # All commands live here
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Rubric Coverage

| Goal | What | Command |
|------|------|---------|
| **1** | Benchmark harness | `make bench-cpu SYSTEM=X` |
| **2** | Latency decomposition | `make decompose SYSTEM=X` |
| **3** | Scaling + inflections | `make sweep-seq SYSTEM=X` |
| **4** | Bottleneck reasoning | `make plots SYSTEM=X` (roofline + regime analysis) |
| **5** | KV-cache quantization | `make sweep-kv SYSTEM=X MODEL=...` |
| **Bonus** | Cross-platform | Run on multiple machines with different `SYSTEM=` |
| **Bonus** | Energy estimation | `make energy SYSTEM=X` |
| **Bonus** | TTFT optimization | See `docs/architecture_notes.md` |

See [`docs/grading_checklist.md`](docs/grading_checklist.md) for the complete rubric-to-command mapping.

## Optimization Summary: KV-Cache Quantization

| KV Precision | Bytes/elem | Memory Reduction | Expected Speedup |
|---|---|---|---|
| f16 (baseline) | 2.0 | — | — |
| q8_0 | 1.0 | 2x | Significant at long contexts |
| q4_0 | 0.5 | 4x | Maximum at long contexts |

The optimization is most effective when KV cache access dominates decode
latency (long contexts, small models). At short contexts where weight
streaming dominates, the benefit is minimal.

See [`docs/kv_cache_quantization.md`](docs/kv_cache_quantization.md) for
full experiment design and architectural interpretation.

## Troubleshooting

| Problem | Solution |
|---|---|
| `MODEL is required` error | GPU/GGUF targets need `MODEL=/path/to/file.gguf` |
| `ModuleNotFoundError: No module named 'torch'` | Run `make install-cpu` or install PyTorch from [pytorch.org](https://pytorch.org) |
| `ModuleNotFoundError: No module named 'llama_cpp'` | Run `make install` (includes llama-cpp-python) |
| `error: externally-managed-environment` | Create a virtual environment first (see Setup section) |
| MPS errors on Apple Silicon | Use `make bench-cpu` as fallback; MPS support varies by PyTorch version |
| `FileNotFoundError` for GGUF model | Provide the full absolute path via `MODEL=/full/path/to/file.gguf` |
| No plots generated | Run benchmarks first (`make bench-cpu`), then `make plots` |
| Tests fail | Run `make install-cpu` to ensure all dependencies are installed |

## License

MIT — see [LICENSE](LICENSE).
