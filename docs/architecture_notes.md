# Architecture Notes: LLM Inference Performance

## The Two Phases of Autoregressive Generation

### Prefill (Prompt Processing)

The prefill phase processes all prompt tokens in parallel through the
transformer. For a prompt of length S:

- **Attention complexity**: O(S² · d) — each token attends to all preceding tokens
- **MLP complexity**: O(S · H · I) — each token independently through the MLP
- **Memory access**: Reads model weights once; creates the initial KV cache

Prefill is typically **compute-bound** because it processes many tokens
simultaneously, yielding high arithmetic intensity (many FLOPs per byte loaded).

### Decode (Token Generation)

Each decode step generates one token by:
1. Reading the full KV cache (all prior keys/values)
2. Computing attention with the new query against all cached keys
3. Passing through the MLP (weight reads dominate)
4. Sampling the next token

For batch size 1, decode is overwhelmingly **memory-bandwidth-bound**:
- Arithmetic intensity ≈ 1-2 FLOP/byte
- Full model weights are streamed from memory for each token
- KV cache reads grow linearly with generated sequence length

## Why TTFT Scales Super-Linearly

TTFT = prefill time + first decode step.

The prefill phase has quadratic attention complexity in prompt length S.
For self-attention:

```
FLOPs ∝ n_layers × n_heads × S² × head_dim
```

At short prompts, the quadratic term is small and TTFT appears linear.
As S grows, the S² term dominates, and TTFT grows super-linearly.

### What Could Reduce TTFT

1. **Flash Attention**: Reduces memory access from O(S²) to O(S) by tiling
   the attention computation, keeping intermediate results in SRAM.
   Does not reduce FLOPs but dramatically improves throughput.

2. **Fused attention kernels**: Combining QKV projection + attention +
   output projection into a single kernel eliminates intermediate memory
   round-trips.

3. **Better prefill batching**: Processing the prompt in chunks can
   reduce peak memory usage and enable overlapping compute with memory access.

4. **Kernel fusion for LayerNorm + residual**: Eliminates separate memory
   reads/writes for the residual add and layer normalization.

5. **KV cache layout optimization**: Contiguous memory layout for K and V
   across layers enables more efficient memory access patterns.

6. **Speculative decoding** (future work): Uses a small "draft" model to
   propose multiple tokens verified by the large model in one forward pass.
   Reduces the number of large-model forward passes needed.  TokenScope
   now includes an experimental speculative decoding mode (`hf.mode=spec_decode`).
   Provide `hf.spec.draft_model_id` and `hf.spec.draft_steps` in the config to
   enable this optimization.  If the draft model is omitted, the code
   gracefully falls back to baseline decode.

## Memory Bandwidth: The Decode Bottleneck

### Weight Streaming

For a model with W total weight bytes, each decode step must stream all
weights through the compute units:

```
weight_time ≈ W / bandwidth
```

For LLaMA-7B in fp16: W ≈ 14 GB
At 50 GB/s bandwidth: weight_time ≈ 280 ms → ~3.5 tok/s

### KV Cache Access

In addition to weights, each decode step reads the full KV cache:

```
kv_bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_elem
kv_time  ≈ kv_bytes / bandwidth
```

For LLaMA-7B at seq_len=2048, fp16:
```
kv_bytes = 2 × 32 × 32 × 128 × 2048 × 2 = 1.07 GB
kv_time  ≈ 21.4 ms at 50 GB/s
```

At long contexts, KV reads become a significant fraction of total decode time.

### The Crossover Point

There exists a sequence length where KV cache access time exceeds weight
streaming time. Beyond this point, KV optimization (quantization, GQA,
sliding window) becomes the primary lever for latency reduction.

## Grouped-Query Attention (GQA)

LLaMA-2-70B and LLaMA-3 use GQA, which shares K/V heads across multiple
query heads:

```
KV heads = total_heads / group_size
```

This reduces KV cache by `group_size`×, directly reducing bandwidth
requirements. LLaMA-3-8B uses 8 KV heads vs. 32 query heads (4× reduction).

## KV-Cache Quantization

Reducing KV cache precision from fp16 to int8/int4 cuts cache size by
2-4× without requiring model retraining. The attention computation:

```
attention = softmax(Q @ K^T / sqrt(d)) @ V
```

K and V values can tolerate moderate quantization noise because:
- Softmax concentrates weights on a few high-attention positions
- Averaging over many key-value pairs reduces individual errors
- The model was trained with residual connections that propagate corrections

See [kv_cache_quantization.md](kv_cache_quantization.md) for experimental details.

## Batch Size Effects (Beyond Scope)

This study focuses on batch size 1 (interactive, single-user) inference.
At larger batch sizes, arithmetic intensity increases and the workload
shifts toward compute-bound. This is the domain of throughput-oriented
serving systems like vLLM, which are outside our scope.
