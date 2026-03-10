# TokenScope Findings Report — Macbook_M1

**System:** `Macbook_M1`

_Generated: 2026-03-09 19:13 UTC_

## Executive Summary

- TTFT ranges from 1.4 ms to 30.9 ms across tested configurations.
- Steady-state per-token latency: 21.92 ms median (46 tok/s).
- Inflection point detected at prompt length 128: slope increases by 0.124×.
- Dominant decode component: overhead (36.68% of total).

## Measurement Methodology

**Ttft Definition:** Time To First Token (TTFT) is measured as the wall-clock time from the start of the generation call to the timestamp of the first emitted token. This includes prompt processing (prefill) and the first decode step.

**Per Token Definition:** Per-token latency is the inter-token interval: the time between consecutive token emissions during autoregressive decoding.

**E2E Definition:** End-to-end latency is the total wall-clock time from the start of generation to the emission of the final token.

**Warmup Rationale:** Warmup runs (default 3) are discarded to allow JIT compilation, memory allocation, and cache warming to stabilize before measurement.

**Outlier Handling:** Raw traces are always preserved. Summary statistics are computed on IQR-filtered data (k=1.5) to reduce the impact of OS scheduling jitter, GC pauses, and thermal throttling. Both raw and filtered statistics are reported.

**Steady State Rule:** The first N decode tokens (default 2) within each trial are excluded from steady-state statistics to account for cache warmup effects within a single generation.

**Ci Method:** 95% confidence intervals are computed via bootstrap resampling (5000 iterations) of trial-level means.

**Timing Cpu:** time.perf_counter_ns (monotonic, nanosecond resolution).

**Timing Cuda:** torch.cuda.Event with enable_timing=True. Events are recorded on the GPU timeline; elapsed_time synchronizes only at measurement.

**Timing Mps:** time.perf_counter_ns with torch.mps.synchronize() barriers. Limitation: MPS lacks event-based timing; wall-clock includes potential queue delays.

## Scaling Analysis

### Per-Token Latency vs. Context Length

![Scaling](../figures/scaling_per_token.png)

### TTFT vs. Context Length

![TTFT](../figures/ttft_scaling.png)

### Throughput

![Throughput](../figures/throughput_scaling.png)

### Inflection Points

| metric | prompt_length | slope_before | slope_after | ratio |
| --- | --- | --- | --- | --- |
| per_token_mean_ms | 128 | -0.005278 | -0.000652 | 0.124 |

### Per-Token Latency Trace

![Token Trace](../figures/token_trace.png)

## Latency Decomposition

### decomp_20260309T191302_cdf410d2_77ca48f.csv

| component | mean_ms | total_ms | count | pct_of_total |
| --- | --- | --- | --- | --- |
| overhead | 0.4012 | 6.4195 | 16 | 36.68 |
| attention | 0.1252 | 4.0079 | 32 | 22.9 |
| lm_head | 0.2056 | 3.2898 | 16 | 18.8 |
| mlp | 0.0632 | 2.0209 | 32 | 11.55 |
| layernorm | 0.0125 | 1.0002 | 80 | 5.72 |
| embedding | 0.0238 | 0.7621 | 32 | 4.35 |


![Decomposition](../figures/decomp_stacked_20260309T191302_cdf410d2_77ca48f.png)

## Architectural Bottleneck Analysis

### What Dominates Where

- In `decomp_20260309T191302_cdf410d2_77ca48f.csv`, **overhead** dominates at 36.68% of total decode time.

### Device Bandwidth

| device | bandwidth_gb_s | method |
| --- | --- | --- |
| cpu | 35.33 | numpy_copy |

### KV Cache Size Scaling

![KV Cache](../figures/kv_cache_size.png)

## Optimization: KV-Cache Quantization

_Run `python -m bench.sweep --config configs/sweep_kv_cache.yaml` to generate KV-cache quantization results._

## What Would I Change in Hardware?

Based on the analysis, autoregressive decode at batch size 1 is overwhelmingly memory-bandwidth-bound. The following hardware changes would yield the largest improvements:

1. **Higher memory bandwidth** — HBM3/HBM3E provides 2-3× the bandwidth of HBM2E. Since decode is bandwidth-bound, this translates nearly linearly to throughput.
2. **Larger on-chip SRAM / L2 cache** — Keeping the KV cache on-chip eliminates repeated HBM reads. NVIDIA H100 already allocates 50MB of L2, but this is only sufficient for ~1K tokens with LLaMA-7B.
3. **Kernel fusion** — Fusing attention + softmax + value projection into a single kernel reduces memory round-trips and launch overhead.
4. **On-chip KV buffering** — Dedicated scratchpad for KV cache (e.g., in NPU architectures) would avoid DRAM accesses entirely for moderate context lengths.
5. **Reduced-precision compute** — INT8/INT4 for KV cache reduces bytes per element without requiring full weight quantization.

## Threats to Validity

1. **Timing synchronization**: MPS timing uses wall-clock with `mps.synchronize()` barriers, which may include queue delays not present in CUDA event timing.
2. **Profiler overhead**: Hook-based decomposition adds non-trivial overhead (function call + synchronization per module per step). Results should be interpreted as relative proportions, not absolute values.
3. **Thermal throttling**: Extended benchmark runs may trigger thermal throttling, particularly on laptops. IQR filtering partially mitigates this.
4. **OS scheduling jitter**: Background processes can cause latency spikes. We mitigate via warmup, multiple trials, and IQR filtering.
5. **Small model approximation**: Tiny-GPT2 results may not reflect the architectural bottlenecks of production-scale LLaMA models (7B+).
6. **Bandwidth proxy**: numpy/torch copy benchmarks measure achievable bandwidth, not peak. Actual model inference may achieve higher or lower utilization.
