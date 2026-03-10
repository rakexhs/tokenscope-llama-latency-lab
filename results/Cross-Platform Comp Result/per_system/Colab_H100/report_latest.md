# TokenScope Findings Report — Colab_H100

**System:** `Colab_H100`

_Generated: 2026-03-09 18:18 UTC_

## Executive Summary

- TTFT ranges from 1.2 ms to 49.5 ms across tested configurations.
- Steady-state per-token latency: 10.09 ms median (99 tok/s).
- Inflection point detected at prompt length 128: slope increases by 0.007×.
- Dominant decode component: attention (43.7% of total).

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
| per_token_mean_ms | 128 | 0.057532 | 0.000423 | 0.007 |

### Per-Token Latency Trace

![Token Trace](../figures/token_trace.png)

## Latency Decomposition

### decomp_20260309T181814_cdf410d2_77ca48f.csv

| component | mean_ms | total_ms | count | pct_of_total |
| --- | --- | --- | --- | --- |
| attention | 7.661 | 245.1528 | 32 | 43.7 |
| embedding | 3.0453 | 97.4483 | 32 | 17.37 |
| mlp | 2.9335 | 93.8726 | 32 | 16.73 |
| overhead | 4.2291 | 67.6655 | 16 | 12.06 |
| lm_head | 1.8777 | 30.0436 | 16 | 5.36 |
| layernorm | 0.3354 | 26.8306 | 80 | 4.78 |


![Decomposition](../figures/decomp_stacked_20260309T181814_cdf410d2_77ca48f.png)

## Architectural Bottleneck Analysis

### What Dominates Where

- In `decomp_20260309T181814_cdf410d2_77ca48f.csv`, **attention** dominates at 43.7% of total decode time.

### Device Bandwidth

| device | bandwidth_gb_s | method |
| --- | --- | --- |
| cuda | 2756.37 | torch_clone |

### KV Cache Size Scaling

![KV Cache](../figures/kv_cache_size.png)

## Optimization: KV-Cache Quantization

| kv_type_k | kv_type_v | prompt_length | per_token_mean_ms | ttft_mean_ms | throughput_mean_tok_s |
| --- | --- | --- | --- | --- | --- |
| q8_0 | f16 | 128 | 10.200 | 10.725 | 85.465 |
| q4_0 | f16 | 128 | 9.957 | 10.385 | 89.653 |
| q8_0 | f16 | 256 | 10.244 | 10.704 | 68.616 |
| q4_0 | f16 | 256 | 10.465 | 11.144 | 72.904 |
| q8_0 | f16 | 512 | 10.429 | 11.204 | 72.481 |
| q4_0 | f16 | 512 | 11.752 | 20.254 | 61.888 |
| q8_0 | f16 | 1024 | 12.446 | 22.056 | 61.462 |
| q4_0 | f16 | 1024 | 11.873 | 12.795 | 69.418 |


![KV Quant](../figures/kv_quant_comparison.png)

KV-cache quantization reduces memory footprint linearly with precision. At long contexts where KV cache access dominates, lower precision (q8_0, q4_0) reduces bandwidth pressure and improves decode latency. At short contexts, the overhead of dequantization may offset gains, placing the workload in the overhead-bound regime.

## What Would I Change in Hardware?

Based on the analysis, autoregressive decode at batch size 1 is overwhelmingly memory-bandwidth-bound. The following hardware changes would yield the largest improvements:

1. **Higher memory bandwidth** — HBM3/HBM3E provides 2-3× the bandwidth of HBM2E. Since decode is bandwidth-bound, this translates nearly linearly to throughput.
2. **Larger on-chip SRAM / L2 cache** — Keeping the KV cache on-chip eliminates repeated HBM reads. NVIDIA H100 already allocates 50MB of L2, but this is only sufficient for ~1K tokens with LLaMA-7B.
3. **Kernel fusion** — Fusing attention + softmax + value projection into a single kernel reduces memory round-trips and launch overhead.
4. **On-chip KV buffering** — Dedicated scratchpad for KV cache (e.g., in NPU architectures) would avoid DRAM accesses entirely for moderate context lengths.
5. **Reduced-precision compute** — INT8/INT4 for KV cache reduces bytes per element without requiring full weight quantization.

## Energy per Token (Bonus)

| available | avg_power_w | elapsed_s | total_joules | joules_per_token | n_tokens | n_power_samples |
| --- | --- | --- | --- | --- | --- | --- |
| True | 117.41 | 0.724 | 85.009 | 2.6565 | 32 | 10 |

## Threats to Validity

1. **Timing synchronization**: MPS timing uses wall-clock with `mps.synchronize()` barriers, which may include queue delays not present in CUDA event timing.
2. **Profiler overhead**: Hook-based decomposition adds non-trivial overhead (function call + synchronization per module per step). Results should be interpreted as relative proportions, not absolute values.
3. **Thermal throttling**: Extended benchmark runs may trigger thermal throttling, particularly on laptops. IQR filtering partially mitigates this.
4. **OS scheduling jitter**: Background processes can cause latency spikes. We mitigate via warmup, multiple trials, and IQR filtering.
5. **Small model approximation**: Tiny-GPT2 results may not reflect the architectural bottlenecks of production-scale LLaMA models (7B+).
6. **Bandwidth proxy**: numpy/torch copy benchmarks measure achievable bandwidth, not peak. Actual model inference may achieve higher or lower utilization.
