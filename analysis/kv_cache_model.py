"""Analytical model for KV-cache memory footprint and scaling.

Computes KV cache size as a function of model architecture and sequence
length. Provides cache-threshold estimation and visualization utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class ModelArchitecture:
    """Architectural parameters for KV cache sizing."""

    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    bytes_per_elem_k: float = 2.0  # f16 default
    bytes_per_elem_v: float = 2.0

    @property
    def kv_bytes_per_token(self) -> float:
        """Bytes of KV cache added per token per layer."""
        k_per_token = self.n_kv_heads * self.head_dim * self.bytes_per_elem_k
        v_per_token = self.n_kv_heads * self.head_dim * self.bytes_per_elem_v
        return k_per_token + v_per_token

    @property
    def total_kv_bytes_per_token(self) -> float:
        """Bytes of KV cache added per token across all layers."""
        return self.kv_bytes_per_token * self.n_layers


# Pre-defined architectures for common LLaMA variants
KNOWN_ARCHITECTURES: dict[str, ModelArchitecture] = {
    "llama-7b": ModelArchitecture("LLaMA-7B", n_layers=32, n_kv_heads=32, head_dim=128),
    "llama-13b": ModelArchitecture("LLaMA-13B", n_layers=40, n_kv_heads=40, head_dim=128),
    "llama2-7b": ModelArchitecture("LLaMA-2-7B", n_layers=32, n_kv_heads=32, head_dim=128),
    "llama2-70b": ModelArchitecture("LLaMA-2-70B", n_layers=80, n_kv_heads=8, head_dim=128),
    "llama3-8b": ModelArchitecture("LLaMA-3-8B", n_layers=32, n_kv_heads=8, head_dim=128),
    "tiny-gpt2": ModelArchitecture("tiny-gpt2", n_layers=2, n_kv_heads=2, head_dim=32),
}

BYTES_PER_ELEM = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}

# Common cache sizes for reference lines (bytes)
CACHE_SIZES = {
    "L2 (256 KB)": 256 * 1024,
    "L2 (1 MB)": 1 * 1024 * 1024,
    "LLC (8 MB)": 8 * 1024 * 1024,
    "LLC (32 MB)": 32 * 1024 * 1024,
    "LLC (64 MB)": 64 * 1024 * 1024,
}


def kv_cache_bytes(
    arch: ModelArchitecture,
    seq_len: int,
) -> float:
    """Total KV cache size in bytes for a given sequence length."""
    return arch.total_kv_bytes_per_token * seq_len


def kv_cache_table(
    arch: ModelArchitecture,
    seq_lengths: list[int],
) -> list[dict[str, Any]]:
    """Compute KV cache stats for a list of sequence lengths."""
    rows = []
    for sl in seq_lengths:
        total = kv_cache_bytes(arch, sl)
        rows.append({
            "seq_len": sl,
            "kv_total_bytes": total,
            "kv_total_mb": total / (1024**2),
            "kv_per_token_bytes": arch.total_kv_bytes_per_token,
            "kv_per_token_kb": arch.total_kv_bytes_per_token / 1024,
        })
    return rows


def cache_threshold_seq_len(
    arch: ModelArchitecture,
    cache_bytes: float,
) -> int:
    """Sequence length at which KV cache exceeds a given cache size."""
    bpt = arch.total_kv_bytes_per_token
    if bpt <= 0:
        return 0
    return int(cache_bytes / bpt)


def kv_cache_curve(
    arch: ModelArchitecture,
    max_seq: int = 4096,
    step: int = 64,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (seq_lengths, kv_mb) arrays for plotting."""
    seq = np.arange(step, max_seq + 1, step, dtype=np.float64)
    kv_mb = seq * arch.total_kv_bytes_per_token / (1024**2)
    return seq, kv_mb


def compare_kv_precision(
    arch_name: str,
    seq_lengths: list[int],
    precisions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compare KV cache sizes across K/V precisions."""
    if precisions is None:
        precisions = ["f16", "q8_0", "q4_0"]

    base_arch = KNOWN_ARCHITECTURES.get(arch_name)
    if base_arch is None:
        raise ValueError(f"Unknown architecture: {arch_name}")

    rows = []
    for prec in precisions:
        bpe = BYTES_PER_ELEM.get(prec, 2.0)
        arch = ModelArchitecture(
            name=base_arch.name,
            n_layers=base_arch.n_layers,
            n_kv_heads=base_arch.n_kv_heads,
            head_dim=base_arch.head_dim,
            bytes_per_elem_k=bpe,
            bytes_per_elem_v=bpe,
        )
        for sl in seq_lengths:
            total = kv_cache_bytes(arch, sl)
            rows.append({
                "precision": prec,
                "seq_len": sl,
                "kv_total_mb": total / (1024**2),
                "kv_per_token_bytes": arch.total_kv_bytes_per_token,
            })
    return rows
