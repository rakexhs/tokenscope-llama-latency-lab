"""Tests for KV-cache analytical model."""

import pytest

from analysis.kv_cache_model import (
    BYTES_PER_ELEM,
    KNOWN_ARCHITECTURES,
    ModelArchitecture,
    cache_threshold_seq_len,
    compare_kv_precision,
    kv_cache_bytes,
    kv_cache_table,
)


class TestModelArchitecture:
    def test_kv_bytes_per_token(self):
        arch = ModelArchitecture(
            name="test",
            n_layers=32,
            n_kv_heads=32,
            head_dim=128,
            bytes_per_elem_k=2.0,
            bytes_per_elem_v=2.0,
        )
        # per layer: 32 * 128 * 2 (K) + 32 * 128 * 2 (V) = 8192 + 8192 = 16384
        assert arch.kv_bytes_per_token == 16384
        # total: 16384 * 32 layers = 524288 = 512 KB/token
        assert arch.total_kv_bytes_per_token == 524288

    def test_gqa_reduces_kv(self):
        full = ModelArchitecture("full", 32, 32, 128)
        gqa = ModelArchitecture("gqa", 32, 8, 128)  # 4× fewer KV heads
        assert gqa.total_kv_bytes_per_token == full.total_kv_bytes_per_token / 4


class TestKVCacheBytes:
    def test_linear_scaling(self):
        arch = KNOWN_ARCHITECTURES["llama-7b"]
        size_1 = kv_cache_bytes(arch, 100)
        size_2 = kv_cache_bytes(arch, 200)
        assert size_2 == pytest.approx(size_1 * 2)

    def test_zero_seq_len(self):
        arch = KNOWN_ARCHITECTURES["llama-7b"]
        assert kv_cache_bytes(arch, 0) == 0


class TestCacheThreshold:
    def test_threshold_correctness(self):
        arch = KNOWN_ARCHITECTURES["llama-7b"]
        cache_size = 8 * 1024 * 1024  # 8 MB
        threshold = cache_threshold_seq_len(arch, cache_size)
        assert threshold > 0
        # KV at threshold should be <= cache_size
        assert kv_cache_bytes(arch, threshold) <= cache_size
        # KV at threshold + 1 should exceed
        assert kv_cache_bytes(arch, threshold + 1) > cache_size


class TestKVCacheTable:
    def test_table_shape(self):
        arch = KNOWN_ARCHITECTURES["tiny-gpt2"]
        table = kv_cache_table(arch, [128, 256, 512])
        assert len(table) == 3
        assert "seq_len" in table[0]
        assert "kv_total_mb" in table[0]
        assert table[0]["seq_len"] == 128


class TestCompareKVPrecision:
    def test_precision_reduces_size(self):
        rows = compare_kv_precision("llama-7b", [1024])
        f16_row = next(r for r in rows if r["precision"] == "f16")
        q8_row = next(r for r in rows if r["precision"] == "q8_0")
        q4_row = next(r for r in rows if r["precision"] == "q4_0")

        assert q8_row["kv_total_mb"] == pytest.approx(f16_row["kv_total_mb"] / 2)
        assert q4_row["kv_total_mb"] == pytest.approx(f16_row["kv_total_mb"] / 4)

    def test_unknown_arch_raises(self):
        with pytest.raises(ValueError):
            compare_kv_precision("nonexistent-model", [128])
