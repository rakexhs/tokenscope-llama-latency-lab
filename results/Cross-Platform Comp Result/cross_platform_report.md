# Cross-Platform Comparison Report

**TokenScope Latency Lab** — Full Rubric Coverage

Mac M1 | WSL/Windows | Colab H100

---

## Rubric Coverage

| Goal | What | Status |
|------|------|--------|
| 1 | Benchmark harness | ✓ agg_latest, per-token, TTFT, throughput |
| 2 | Latency decomposition | ✓ decomp comparison |
| 3 | Scaling + inflections | ✓ scaling plots, inflections_sweep |
| 4 | Bottleneck reasoning | ✓ roofline, regime map, predictor |
| 5 | KV-cache quantization | ✓ per_system/kv_quant_comparison (Colab) |
| Bonus | Cross-platform | ✓ This report |
| Bonus | Energy estimation | ✓ csv/cross_platform_energy.csv (Colab) |
| Bonus | TTFT optimization | ✓ docs/architecture_notes.md |

---

## Executive Summary

### Mac M1 (Apple Silicon)

- **Per-token latency:** 21.9 ms median (range 0.5–24.7 ms)
- **TTFT:** 20.8 ms median (range 1.4–30.9 ms)
- **Throughput:** 43 tok/s median (range 34–1795 tok/s)

### WSL / Windows

- **Per-token latency:** 27.4 ms median (range 1.4–31.4 ms)
- **TTFT:** 28.7 ms median (range 3.3–86.3 ms)
- **Throughput:** 35 tok/s median (range 30–640 tok/s)

### Colab H100 (NVIDIA)

- **Per-token latency:** 10.1 ms median (range 0.7–12.5 ms)
- **TTFT:** 10.7 ms median (range 1.2–49.5 ms)
- **Throughput:** 88 tok/s median (range 61–1259 tok/s)

## Device Bandwidth

| System | Device | Bandwidth (GB/s) |
|--------|--------|------------------|
| Mac M1 (Apple Silicon) | cpu | 35.3 |
| WSL / Windows | cpu | 3.8 |
| Colab H100 (NVIDIA) | cuda | 2756.4 |

## Energy Estimation (NVIDIA GPU)

- **Colab H100 (NVIDIA):** 2.656 J/token

## Key Findings

1. **Colab H100 (CUDA)** achieves the highest throughput and lowest per-token latency due to GPU acceleration and high memory bandwidth (~2.7 TB/s).
2. **Mac M1** benefits from unified memory and MPS when available; CPU fallback shows competitive performance vs. WSL.
3. **WSL/Windows** runs on CPU with lower memory bandwidth (~3.8 GB/s) than Mac M1 (~35 GB/s), leading to higher per-token latency.
4. **Latency decomposition** differs by platform: Colab H100 is attention-dominated; Mac/WSL show higher overhead proportion.

## Generated Artifacts

### Figures (root + figures/)

| Figure | Description |
|--------|-------------|
| `cross_platform_per_token.png` | Per-token latency vs. context length |
| `cross_platform_ttft.png` | Time to first token vs. context length |
| `cross_platform_throughput.png` | Throughput (tok/s) vs. context length |
| `cross_platform_bandwidth.png` | Device memory bandwidth |
| `cross_platform_decomp.png` | Latency decomposition |
| `roofline_*.png` | Roofline model per system |
| `regime_map_*.png` | Bottleneck regime map per system |

### Per-System (`per_system/<system>/`)

Scaling, TTFT, throughput, token trace, KV cache, KV quant (when available), decomp stacked.

### CSVs (`csv/`)

summary, bandwidth, pivot (per-token, throughput), decomp, inflections, energy, predictor_coeffs, regime_map_*

### Docs (`docs/`)

architecture_notes.md (TTFT optimization), kv_cache_quantization.md
