# NVIDIA Nsight Systems Profiling Instructions

## Overview

Nsight Systems (`nsys`) provides GPU-level trace profiling for CUDA workloads. It captures kernel launches, memory transfers, CUDA API calls, and CPU-side activity with minimal overhead.

## Prerequisites

- NVIDIA GPU with CUDA support
- Nsight Systems installed (bundled with CUDA Toolkit ≥11.0)
- `nsys` on PATH

## Profiling a Benchmark Run

```bash
# Profile a single benchmark run
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=results/nsys_bench \
  --force-overwrite true \
  python -m bench.run_bench \
    --config configs/bench_default.yaml \
    --override backend=hf device=cuda benchmark.trials=1 benchmark.warmup_runs=1
```

## Profiling the Decomposition

```bash
nsys profile \
  --trace=cuda,nvtx \
  --output=results/nsys_decomp \
  python -m profiling.decompose_decode \
    --model meta-llama/Llama-2-7b-hf --device cuda --n_tokens 16
```

## Viewing Results

```bash
# Generate summary report
nsys stats results/nsys_bench.nsys-rep

# Open in Nsight Systems GUI
nsys-ui results/nsys_bench.nsys-rep
```

## Key Metrics to Look For

1. **Kernel occupancy**: Are GPU SMs fully utilized?
2. **Memory transfer overlap**: Do H2D/D2H transfers overlap with compute?
3. **Kernel launch gaps**: Idle time between consecutive kernels indicates CPU-bound launch overhead.
4. **Attention vs MLP kernel time**: Which kernel class dominates total GPU time?
5. **KV cache memory operations**: Identify reads/writes to the KV cache region.

## Interpreting for Autoregressive Decode

During batch-size-1 decode, expect:
- Very short kernels (low arithmetic intensity)
- Large fraction of time in memory reads (KV cache + weights)
- CPU launch overhead may be significant relative to kernel duration
- Pipeline bubbles between consecutive decode steps

## Export to Chrome Trace (Optional)

```bash
nsys export --type=json results/nsys_bench.nsys-rep -o results/nsys_bench.json
```

Open in `chrome://tracing` for a visual timeline.
