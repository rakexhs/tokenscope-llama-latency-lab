# KV-Cache Quantization: Experiment Design and Analysis

## Motivation

During autoregressive decoding, the KV cache is read for every generated
token. As sequence length grows, KV cache bandwidth becomes a significant
fraction of total decode time. Quantizing the cached keys and values
reduces the bytes transferred per token, potentially improving latency.

## Implementation

### llama.cpp KV Type Support

llama.cpp supports configurable KV cache precision via `type_k` and `type_v`
parameters. We expose these through the benchmark config:

```yaml
llamacpp:
  kv_type_k: q8_0   # f16 | q8_0 | q4_0
  kv_type_v: q8_0
```

Internally, these map to GGML tensor types:
- `f16` → `GGML_TYPE_F16` (2 bytes/element)
- `q8_0` → `GGML_TYPE_Q8_0` (1 byte/element, block-quantized)
- `q4_0` → `GGML_TYPE_Q4_0` (0.5 bytes/element, block-quantized)

### Experiment Configuration

```yaml
# configs/sweep_kv_cache.yaml
sweep:
  generation.prompt_length: [128, 256, 512, 1024]
  llamacpp.kv_type_k: [f16, q8_0, q4_0]
  llamacpp.kv_type_v: [f16, q8_0, q4_0]
```

Full Cartesian product: 4 × 3 × 3 = 36 configurations.
Each with 5 trials, 2 warmup runs.

## Expected Results

### Memory Footprint Reduction

For LLaMA-7B (32 layers, 32 KV heads, head_dim=128):

| KV Type | Bytes/elem | KV per token (all layers) | KV at 2048 tokens |
|---------|-----------|---------------------------|-------------------|
| f16/f16 | 2.0 | 524,288 B (512 KB) | 1.07 GB |
| q8_0    | 1.0 | 262,144 B (256 KB) | 537 MB |
| q4_0    | 0.5 | 131,072 B (128 KB) | 268 MB |

### Latency Impact

**At long contexts (>512 tokens):**
KV cache reads dominate. Reducing precision cuts bandwidth requirement
proportionally → latency reduction close to 2× (q8_0) or 4× (q4_0)
for the KV-bound component.

**At short contexts (<256 tokens):**
Weight streaming dominates decode latency. KV quantization saves little
absolute time, and the dequantization overhead may slightly increase
per-token cost → neutral or slight regression.

### Quality Impact

KV-cache quantization introduces numerical error in attention computations.
Empirical observations from the llama.cpp community:

- **q8_0**: Negligible quality loss for most tasks. Perplexity increase < 0.1.
- **q4_0**: Measurable quality degradation at long contexts. Attention
  patterns may shift, particularly for tasks requiring precise long-range
  retrieval.

We focus on latency measurement; quality evaluation is outside scope.

## Interpreting Results

The sweep produces per-token latency at each (prompt_length, kv_type) pair.
Key plots:

1. **Per-token latency vs. prompt length** — one line per KV config.
   Expect convergence at short contexts, divergence at long contexts.

2. **Speedup vs. f16 baseline** — (f16_latency / quantized_latency).
   Expect speedup > 1 at long contexts, ≈ 1 at short contexts.

3. **Regime map** — shows where each KV config transitions from
   overhead-bound to KV-bound.

## Architectural Interpretation

KV-cache quantization is fundamentally a **bandwidth optimization**:

```
decode_time ≈ weight_stream_time + kv_access_time + overhead
```

Quantizing KV reduces `kv_access_time` without affecting `weight_stream_time`.
The optimization is effective when:

```
kv_access_time / total_decode_time > threshold (e.g., 20%)
```

This condition is met at long contexts and with small models (where weight
streaming is fast relative to KV access).

## Running the Sweep

```bash
# With a GGUF model file
python -m bench.sweep --config configs/sweep_kv_cache.yaml \
  --override model.id_or_path=/path/to/model.gguf

# Generate comparison plots
python -m analysis.make_plots --results_dir results
```
