"""Prompt synthesis utilities for fixed-token-length prompts."""

from __future__ import annotations

from typing import Optional

# Fixed vocabulary passage for deterministic prompt lengths.
# Chosen to be semantically coherent and avoid tokenizer edge cases.
_BASE_PASSAGE = (
    "The architecture of modern processors balances compute throughput with memory "
    "bandwidth. As transformer models scale, the key-value cache grows linearly with "
    "sequence length, creating memory pressure during autoregressive decoding. Each "
    "decode step reads the full KV cache to compute attention scores, making memory "
    "bandwidth the dominant bottleneck at long contexts. Hardware designers address "
    "this through larger caches, higher bandwidth memory, and specialized attention "
    "accelerators. Software optimizations include KV cache quantization, grouped-query "
    "attention, and sliding window mechanisms that bound cache size. Understanding "
    "these interactions requires careful measurement of latency at each architectural "
    "layer. Performance forensics combines microbenchmarks with analytical models to "
    "attribute latency to specific components. This enables targeted optimization of "
    "the true bottleneck rather than guesswork. The roofline model provides a framework "
    "for classifying workloads as compute-bound or memory-bound based on arithmetic "
    "intensity. For autoregressive decoding with batch size one, the arithmetic intensity "
    "is typically very low, placing the workload firmly in the memory-bound regime. "
)


def make_prompt(target_tokens: int, tokenizer: Optional[object] = None) -> str:
    """Generate a prompt that is approximately `target_tokens` tokens long.

    If a tokenizer is provided, iteratively adjusts length.
    Otherwise uses a character-based heuristic (~4 chars/token).
    """
    if tokenizer is None:
        chars_per_token = 4.5
        target_chars = int(target_tokens * chars_per_token)
        passage = _BASE_PASSAGE
        while len(passage) < target_chars:
            passage += _BASE_PASSAGE
        return passage[:target_chars]

    passage = _BASE_PASSAGE
    while True:
        ids = tokenizer.encode(passage)  # type: ignore[union-attr]
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids  # type: ignore[union-attr]
        if isinstance(ids, list):
            n = len(ids)
        else:
            n = ids.shape[-1] if hasattr(ids, "shape") else len(ids)
        if n >= target_tokens:
            break
        passage += _BASE_PASSAGE

    # Binary search to trim to exact length
    lo, hi = 0, len(passage)
    while lo < hi:
        mid = (lo + hi) // 2
        ids = tokenizer.encode(passage[:mid])  # type: ignore[union-attr]
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids  # type: ignore[union-attr]
        n = len(ids) if isinstance(ids, list) else (ids.shape[-1] if hasattr(ids, "shape") else len(ids))
        if n < target_tokens:
            lo = mid + 1
        else:
            hi = mid
    return passage[:lo]


def fixed_prompts(lengths: list[int], tokenizer: Optional[object] = None) -> dict[int, str]:
    """Generate a dict of {target_length: prompt_string} for each requested length."""
    return {length: make_prompt(length, tokenizer) for length in lengths}
