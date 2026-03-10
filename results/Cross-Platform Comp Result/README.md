# Cross-Platform Comparison Results

**Full rubric coverage** — all goals and bonus items.

## Platforms

- **Mac M1** (Apple Silicon)
- **WSL_Windows** (Windows Subsystem for Linux)
- **Colab_H100** (Google Colab with NVIDIA H100)

## Rubric Coverage

| Goal | What | Location |
|------|------|----------|
| **1** | Benchmark harness | `csv/cross_platform_summary.csv`, scaling plots |
| **2** | Latency decomposition | `cross_platform_decomp.png`, `per_system/*/decomp_stacked_*` |
| **3** | Scaling + inflections | `cross_platform_per_token/ttft/throughput`, `csv/cross_platform_inflections.csv` |
| **4** | Bottleneck reasoning | `roofline_*.png`, `regime_map_*.png`, `csv/cross_platform_predictor_coeffs.csv` |
| **5** | KV-cache quantization | `per_system/Colab_H100/kv_quant_comparison.png` |
| **Bonus** | Cross-platform | This folder |
| **Bonus** | Energy estimation | `csv/cross_platform_energy.csv` |
| **Bonus** | TTFT optimization | `docs/architecture_notes.md` |

## Contents

### Figures (root + `figures/`)

| File | Description |
|------|-------------|
| `cross_platform_per_token.png` | Per-token latency vs. context length |
| `cross_platform_ttft.png` | Time to first token vs. context length |
| `cross_platform_throughput.png` | Throughput (tok/s) vs. context length |
| `cross_platform_bandwidth.png` | Device memory bandwidth |
| `cross_platform_decomp.png` | Latency decomposition |
| `roofline_*.png` | Roofline model per system |
| `regime_map_*.png` | Bottleneck regime map per system |

### Per-System (`per_system/<system>/`)

Scaling, TTFT, throughput, token trace, KV cache, KV quant (when available), decomp stacked, report.

### CSVs (`csv/`)

| File | Description |
|------|-------------|
| `cross_platform_summary.csv` | Full benchmark summary |
| `cross_platform_bandwidth.csv` | Device bandwidth |
| `cross_platform_pivot_*.csv` | Pivot tables |
| `cross_platform_decomp.csv` | Latency decomposition |
| `cross_platform_inflections.csv` | Inflection points |
| `cross_platform_energy.csv` | Energy per token (NVIDIA) |
| `cross_platform_predictor_coeffs.csv` | Predictor coefficients |
| `regime_map_*.csv` | Regime classification per system |

### Docs (`docs/`)

- `architecture_notes.md` — TTFT optimization, architecture notes
- `kv_cache_quantization.md` — KV-cache quantization

### Report

- `cross_platform_report.md` — Executive summary and rubric coverage

## Regenerating

```bash
make cross-platform
```

Or:

```bash
python -m analysis.cross_platform_compare --results_dir results
```
