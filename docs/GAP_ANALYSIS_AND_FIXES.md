# TokenScope Lab2 — Gap Analysis & Implemented Fixes

This document summarizes the gap analysis against the Project 6 rubric and the fixes implemented to achieve full coverage, including the +10% bonus.

---

## Rubric Coverage Summary

| Category | Weight | Status | Notes |
|----------|--------|--------|-------|
| Benchmark Methodology | 20% | ✓ | TTFT, per-token, end-to-end; warmup, trials, IQR |
| Latency Decomposition | 20% | ✓ | Embedding, Q/K/V/o_proj, softmax, MLP, LayerNorm, lm_head |
| Scaling Analysis | 15% | ✓ | Sequence length, model size, precision; inflections |
| Architectural Reasoning | 20% | ✓ | Roofline, predictor, regime; KV cache model |
| Optimization Proposal | 15% | ✓ | KV-cache quantization; before/after estimate |
| Report Quality | 10% | ✓ | Auto-generated report, plots |
| **Bonus** | **+10%** | ✓ | Cross-platform, TTFT, spec_decode, energy |

---

## Gaps Identified and Fixes Applied

### 1. Single-System Bottleneck Analysis (Goal 4)

**Gap:** Roofline, predictor fit, and regime map were only produced by `cross_platform_compare.py`. Single-system runs (`make plots`) did not generate these artifacts.

**Fix:** Added `_run_single_system_bottleneck_analysis()` to `analysis/make_plots.py`. When `device_bandwidth.csv` and `agg_latest.csv` exist, `make plots` now:
- Runs predictor fit and saves `predictor_coeffs.csv`, `predictor_default.png`
- Builds regime map and saves `regime_map_default.csv`, `regime_map.png`
- Generates roofline plot `roofline.png`

**Requirement:** Run `make bandwidth-cpu` (or `bandwidth-gpu`) before `make plots` for bottleneck analysis.

---

### 2. KV-Cache Quantization on All Systems (Bonus / Goal 5)

**Gap:** `sweep-kv` was only in `full-gpu`. `full-cpu` and `full-mps` did not run KV-cache quantization.

**Fix:** Added conditional `_KV_SWEEP` in the Makefile. When `MODEL=/path/to/model.gguf` is provided:
- `full-cpu` and `full-mps` now include `sweep-kv` in their pipeline
- KV-cache quantization runs on CPU and MPS when a GGUF model is used

**Usage:** `make full-cpu SYSTEM=My_Machine MODEL=/path/to/llama-7b.gguf`

---

### 3. Speculative Decoding Comparison (Bonus)

**Gap:** No sweep config or comparison plot for baseline vs. speculative decoding.

**Fix:**
- Added `configs/sweep_spec_decode.yaml` — compares `loop_decode` vs `spec_decode`
- Added `make sweep-spec` target
- Added `plot_spec_decode_comparison()` in `make_plots.py` — generates `spec_decode_comparison.png`
- Added speculative decoding section to `findings_report.py` when both modes are present
- Updated cross-platform report to mention speculative decoding

**Usage:** `make sweep-spec SYSTEM=X` then `make plots`

---

### 4. Model-Size Sweep Without HF Access

**Gap:** `sweep_models.yaml` uses HF model IDs (llama-7b, llama-13b) which require HuggingFace access. No GGUF-only option.

**Fix:** Added `configs/sweep_models_gguf.yaml` and `make sweep-models-gguf`. Users can edit the YAML with their GGUF paths to run model-size scaling without HF.

---

### 5. Documentation Accuracy

**Gap:** `docs/grading_checklist.md` stated roofline/regime were "generated automatically in make_plots" but they were not. Cross-platform bonus section was incomplete.

**Fix:** Updated grading checklist with correct commands and outputs. Added note that `bandwidth-cpu` must run before `make plots` for bottleneck analysis.

---

## Bonus Items Checklist

| Bonus Item | Implementation |
|------------|----------------|
| **Cross-platform (CPU vs GPU)** | `make cross-platform`; `analysis/cross_platform_compare.py` |
| **First-token latency optimization** | `docs/architecture_notes.md` |
| **Speculative decoding comparison** | `make sweep-spec`; `configs/sweep_spec_decode.yaml`; `spec_decode_comparison.png` |
| **Energy-per-token estimation** | `make energy`; `analysis/energy_estimation.py` (NVIDIA); graceful skip on non-NVIDIA |

---

## Remaining Limitations (Documented)

1. **Energy on non-NVIDIA** — CPU/MPS require external tools (e.g. `powermetrics` on macOS).
2. **Precision sweep on CPU** — fp16/bf16 fall back to fp32; documented in `sweep_precision.yaml`.

---

## New Commands Summary

| Command | Purpose |
|---------|---------|
| `make sweep-spec` | Speculative decoding vs baseline |
| `make sweep-models-gguf` | Model-size sweep (GGUF only) |
| `make full-cpu MODEL=path.gguf` | Full CPU pipeline with KV sweep |
