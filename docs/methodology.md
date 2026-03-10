# Measurement Methodology

## Design Principles

1. **Measure what matters**: Token emission timestamps, not internal model timers.
2. **Preserve raw data**: Never discard raw traces; apply filters at analysis time.
3. **Document assumptions**: Every statistical choice is explicitly stated.
4. **Reproduce everything**: Full environment snapshot in every run record.

## Benchmark Protocol

### Warmup Phase

**Default: 3 warmup runs, discarded.**

Rationale: The first few inferences trigger JIT compilation (in PyTorch),
memory allocation, CUDA context initialization, and CPU cache warming.
Warmup runs ensure the steady-state performance is measured, not the
cold-start behavior.

### Trial Phase

**Default: 10 trials.**

Each trial runs the full generation pipeline: prompt encoding → prefill →
autoregressive decode → result collection. Trials are independent — the
model state (KV cache) is reset between trials.

For batched inference experiments (`generation.batch_size > 1`), multiple
prompts are decoded concurrently.  The per-token trace records timestamps
for each token emission across the batch; summary statistics reflect the
average latency per token across all sequences.

### Timing Instrumentation

See [timing_diagram.md](timing_diagram.md) for the complete measurement pipeline.

- **CPU**: `time.perf_counter_ns()` — monotonic, nanosecond resolution.
- **CUDA**: `perf_counter_ns` with `torch.cuda.synchronize()` barriers before
  each timestamp to ensure GPU work is complete.
- **MPS**: `perf_counter_ns` with `torch.mps.synchronize()`. Limitation:
  MPS does not support event-based timing; wall-clock includes queue delays.

### Outlier Handling

**IQR filter (k=1.5, optional, enabled by default).**

Values outside [Q1 - 1.5·IQR, Q3 + 1.5·IQR] are excluded from summary
statistics. Raw data is always preserved in JSONL output.

Common outlier causes:
- OS context switches
- Garbage collection pauses
- Thermal throttling
- Background process interference

### Steady-State Per-Token Statistics

**Default: skip first 2 decode tokens per trial.**

The first few decode tokens often show elevated latency due to KV cache
allocation, memory layout initialization, and branch predictor warmup.
Steady-state statistics exclude these tokens for a more representative
view of sustained decode performance.

When using speculative decoding (`hf.mode = spec_decode`) the draft model
proposes multiple tokens in a single step.  Per-token timestamps in this
mode correspond to accepted tokens after verification by the target model.

### Confidence Intervals

**Bootstrap 95% CI, 5000 iterations, seeded RNG.**

Trial-level means are resampled to produce confidence intervals. This is
more robust than parametric CIs when the sample size is small (n=10) and
the distribution may be skewed.

## Reproducibility Snapshot

Every run record includes:

| Field | Source |
|-------|--------|
| OS, kernel | `platform.system()`, `platform.release()` |
| Python version | `sys.version` |
| CPU model, cores | `platform.processor()`, `os.cpu_count()` |
| RAM | `psutil.virtual_memory().total` |
| GPU name, VRAM | `torch.cuda.get_device_name()` |
| PyTorch version | `torch.__version__` |
| Transformers version | `transformers.__version__` |
| llama-cpp-python version | `llama_cpp.__version__` |
| Git SHA | `git rev-parse --short HEAD` |
| Config hash | SHA-256 of canonical JSON config |
| Timestamp | UTC ISO-8601 |
| All config parameters | Full config dict |

## Run Registry

Each run is assigned a unique ID:
```
{timestamp}_{config_hash}_{git_sha}
20260308T143022_a1b2c3d4_e5f6g7h
```

Output files follow a consistent naming convention:
- `results/raw/{run_id}.jsonl` — raw trial data
- `results/summary/bench_{run_id}.csv` — aggregated summary
- `results/summary/agg_latest.csv` — append-mode aggregate across runs
- `results/report/manifest_{run_id}.json` — full run manifest

All file writes are atomic (write to temp file, then `os.replace`).
