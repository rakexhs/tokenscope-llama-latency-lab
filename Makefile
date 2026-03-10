.PHONY: help install install-cpu test lint systems \
       bench-cpu bench-gpu bench-mps \
       sweep-seq sweep-seq-gpu sweep-models sweep-models-gguf sweep-precision sweep-kv sweep-spec \
       plots report analysis cross-platform gpu-forensics cpu-forensics mps-forensics \
       decompose decompose-gpu profiler \
       bandwidth bandwidth-cpu bandwidth-gpu bandwidth-mps energy \
       full-cpu full-gpu full-mps \
       clean _require-model

PYTHON ?= python3

# ── User-facing variables ──────────────────────────────────────────────
# MODEL  — model path or HF ID (defaults to tiny-gpt2 if not provided)
# SYSTEM — machine name tag for organizing results
MODEL ?= sshleifer/tiny-gpt2

# Auto-detect backend: llamacpp for .gguf files, hf otherwise.
# Override with BACKEND= if needed.
ifeq ($(suffix $(MODEL)),.gguf)
  _AUTO_BACKEND = llamacpp
else
  _AUTO_BACKEND = hf
endif
BACKEND ?= $(_AUTO_BACKEND)

# For HF-only tools (decompose, profiler): if MODEL is a GGUF file these
# tools cannot use it, so fall back to tiny-gpt2 automatically.
ifeq ($(suffix $(MODEL)),.gguf)
  _HF_MODEL = sshleifer/tiny-gpt2
else
  _HF_MODEL = $(MODEL)
endif

SYSTEM_FLAG = $(if $(SYSTEM),--system $(SYSTEM),)

# Guard: fails when a GGUF MODEL is explicitly required but not provided.
_require-model:
	@if [ "$(MODEL)" = "sshleifer/tiny-gpt2" ]; then \
		echo "ERROR: MODEL is required. Usage: make <target> SYSTEM=X MODEL=/path/to/model.gguf"; \
		exit 1; \
	fi

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  MODEL=<path|hf_id>  Model to use (default: sshleifer/tiny-gpt2)."
	@echo "                      .gguf files auto-select llamacpp backend."
	@echo "  SYSTEM=<name>       Tag results by machine name."
	@echo ""
	@echo "  Example: make full-cpu SYSTEM=MacBook_M1 MODEL=/path/to/llama.gguf"

# ── Setup ──────────────────────────────────────────────────────────────
install: ## Install with all extras (HF + llama.cpp + dev)
	pip install -e ".[all]"

install-cpu: ## Install with HF backend only (no GPU/llama.cpp deps)
	pip install -e ".[hf,dev]"

test: ## Run test suite (no model weights needed)
	@$(PYTHON) -c "import pytest" >/dev/null 2>&1 || ( \
		echo "ERROR: pytest is not installed in this Python environment."; \
		echo "       Run: make install-cpu  (or: make install)"; \
		exit 1; \
	)
	$(PYTHON) -m pytest tests/ -v

lint: ## Run ruff linter
	ruff check bench/ profiling/ analysis/ tests/

systems: ## List all systems with saved results
	@$(PYTHON) -c "from bench.utils.system_name import list_systems; [print(s) for s in list_systems()]"

# ── Benchmarks ─────────────────────────────────────────────────────────
bench-cpu: ## CPU benchmark (MODEL= optional, defaults to tiny-gpt2)
	$(PYTHON) -m bench.run_bench \
		--config configs/bench_default.yaml $(SYSTEM_FLAG) \
		--override backend=$(BACKEND) device=cpu \
		model.id_or_path="$(MODEL)" \
		generation.output_length=32 generation.prompt_length=64

bench-gpu: _require-model ## GPU benchmark (requires MODEL=/path/to/model.gguf)
	$(PYTHON) -m bench.run_bench \
		--config configs/bench_default.yaml $(SYSTEM_FLAG) \
		--override backend=llamacpp device=cuda \
		model.id_or_path="$(MODEL)"

bench-mps: ## MPS benchmark (MODEL= optional, defaults to tiny-gpt2)
	$(PYTHON) -m bench.run_bench \
		--config configs/bench_default.yaml $(SYSTEM_FLAG) \
		--override backend=$(BACKEND) device=mps \
		model.id_or_path="$(MODEL)" \
		generation.output_length=32 generation.prompt_length=64

# ── Sweeps ─────────────────────────────────────────────────────────────
sweep-seq: ## Sequence-length sweep (MODEL= optional, defaults to tiny-gpt2)
	$(PYTHON) -m bench.sweep --config configs/sweep_sequence.yaml $(SYSTEM_FLAG) \
		--override backend=$(BACKEND) model.id_or_path="$(MODEL)"

sweep-seq-gpu: _require-model ## Sequence-length sweep on GPU (requires MODEL=)
	$(PYTHON) -m bench.sweep --config configs/sweep_sequence_gpu.yaml $(SYSTEM_FLAG) \
		--override model.id_or_path="$(MODEL)"

sweep-models: ## Model-size sweep (HF models)
	$(PYTHON) -m bench.sweep --config configs/sweep_models.yaml $(SYSTEM_FLAG)

sweep-models-gguf: ## Model-size sweep (GGUF only). Edit config paths first.
	$(PYTHON) -m bench.sweep --config configs/sweep_models_gguf.yaml $(SYSTEM_FLAG)

sweep-precision: ## Precision sweep
	$(PYTHON) -m bench.sweep --config configs/sweep_precision.yaml $(SYSTEM_FLAG)

sweep-kv: _require-model ## KV-cache quantization sweep (requires MODEL= GGUF)
	$(PYTHON) -m bench.sweep --config configs/sweep_kv_cache.yaml $(SYSTEM_FLAG) \
		--override model.id_or_path="$(MODEL)"

sweep-spec: ## Speculative decoding vs baseline (Bonus)
	$(PYTHON) -m bench.sweep --config configs/sweep_spec_decode.yaml $(SYSTEM_FLAG)

# ── Profiling ──────────────────────────────────────────────────────────
# decompose/profiler use HF backend only. If MODEL is a .gguf file they
# automatically fall back to tiny-gpt2; pass an HF model ID to override.
decompose: ## Latency decomposition on CPU (HF model)
	$(PYTHON) -m profiling.decompose_decode \
		--model "$(_HF_MODEL)" --device cpu --n_tokens 16 $(SYSTEM_FLAG)

decompose-gpu: ## Latency decomposition on CUDA GPU (HF model)
	$(PYTHON) -m profiling.decompose_decode \
		--model "$(_HF_MODEL)" --device cuda --n_tokens 16 $(SYSTEM_FLAG)

profiler: ## torch.profiler operator analysis (HF model)
	$(PYTHON) -m profiling.torch_profiler_decode \
		--model "$(_HF_MODEL)" --device cpu $(SYSTEM_FLAG)

bandwidth-cpu: ## Memory bandwidth micro-benchmark (CPU)
	$(PYTHON) -m analysis.bandwidth_microbench --device cpu $(SYSTEM_FLAG)

bandwidth-gpu: ## Memory bandwidth micro-benchmark (CUDA GPU)
	$(PYTHON) -m analysis.bandwidth_microbench --device cuda $(SYSTEM_FLAG)

bandwidth-mps: ## Memory bandwidth micro-benchmark (MPS / CPU fallback)
	$(PYTHON) -m analysis.bandwidth_microbench --device cpu $(SYSTEM_FLAG)

bandwidth: bandwidth-cpu ## Memory bandwidth micro-benchmark (alias for bandwidth-cpu)

energy: ## Energy-per-token estimation (NVIDIA GPU only, safe skip otherwise)
	$(PYTHON) -m analysis.energy_estimation $(SYSTEM_FLAG)

# ── Analysis ───────────────────────────────────────────────────────────
plots: ## Generate all plots from results
	$(PYTHON) -m analysis.make_plots $(SYSTEM_FLAG)

report: ## Generate findings report (Markdown)
	$(PYTHON) -m analysis.findings_report $(SYSTEM_FLAG)

analysis: plots report ## Full analysis pipeline (plots + report)

cross-platform: ## Cross-platform comparison (Mac M1, WSL_Windows, Colab_H100)
	$(PYTHON) -m analysis.cross_platform_compare --results_dir results
	@echo "\n[TokenScope] Cross-platform comparison complete. See results/Cross-Platform Comp Result/"

gpu-forensics: ## Combine GGUF + HF GPU results into a single forensics bundle (requires GGUF_SYSTEM and HF_SYSTEM)
	@if [ -z "$(GGUF_SYSTEM)" ] || [ -z "$(HF_SYSTEM)" ]; then \
		echo "ERROR: GGUF_SYSTEM and HF_SYSTEM are required."; \
		echo "Usage: make gpu-forensics GGUF_SYSTEM=RTX4090_GGUF HF_SYSTEM=RTX4090_HF"; \
		exit 1; \
	fi
	$(PYTHON) -m analysis.gpu_model_forensics --results_dir results --gguf_system "$(GGUF_SYSTEM)" --hf_system "$(HF_SYSTEM)"
	@echo "\n[TokenScope] GPU model forensics bundle complete. See results/Model_Forensics/"

cpu-forensics: ## Combine GGUF + HF CPU results into a single forensics bundle (requires GGUF_SYSTEM and HF_SYSTEM)
	@if [ -z "$(GGUF_SYSTEM)" ] || [ -z "$(HF_SYSTEM)" ]; then \
		echo "ERROR: GGUF_SYSTEM and HF_SYSTEM are required."; \
		echo "Usage: make cpu-forensics GGUF_SYSTEM=Mac_CPU_GGUF HF_SYSTEM=Mac_CPU_HF"; \
		exit 1; \
	fi
	$(PYTHON) -m analysis.gpu_model_forensics --results_dir results --gguf_system "$(GGUF_SYSTEM)" --hf_system "$(HF_SYSTEM)"
	@echo "\n[TokenScope] CPU model forensics bundle complete. See results/Model_Forensics/"

mps-forensics: ## Combine GGUF + HF MPS results into a single forensics bundle (requires GGUF_SYSTEM and HF_SYSTEM)
	@if [ -z "$(GGUF_SYSTEM)" ] || [ -z "$(HF_SYSTEM)" ]; then \
		echo "ERROR: GGUF_SYSTEM and HF_SYSTEM are required."; \
		echo "Usage: make mps-forensics GGUF_SYSTEM=Mac_MPS_GGUF HF_SYSTEM=Mac_MPS_HF"; \
		exit 1; \
	fi
	$(PYTHON) -m analysis.gpu_model_forensics --results_dir results --gguf_system "$(GGUF_SYSTEM)" --hf_system "$(HF_SYSTEM)"
	@echo "\n[TokenScope] MPS model forensics bundle complete. See results/Model_Forensics/"

# When MODEL is a .gguf file, include KV-cache quantization sweep in full-cpu/full-mps
_KV_SWEEP = $(if $(filter gguf,$(suffix $(MODEL))),sweep-kv,)

# ── Full Pipelines (all experiments) ───────────────────────────────────
full-cpu: test bench-cpu sweep-seq sweep-models sweep-precision decompose profiler bandwidth-cpu $(_KV_SWEEP) plots report ## Full CPU pipeline — all experiments (add sweep-kv when MODEL=path.gguf)
	@echo "\n[TokenScope] Full CPU pipeline complete. See results/$${SYSTEM:-<system>}/report/report_latest.md"

full-gpu: _require-model test bench-gpu sweep-seq-gpu sweep-models sweep-precision sweep-kv decompose-gpu profiler bandwidth-gpu energy plots report ## Full GPU pipeline — all experiments (requires MODEL=)
	@echo "\n[TokenScope] Full GPU pipeline complete. See results/$${SYSTEM:-<system>}/report/report_latest.md"

full-mps: test bench-mps sweep-seq sweep-models sweep-precision decompose profiler bandwidth-cpu $(_KV_SWEEP) plots report ## Full MPS pipeline — all experiments (add sweep-kv when MODEL=path.gguf)
	@echo "\n[TokenScope] Full MPS pipeline complete. See results/$${SYSTEM:-<system>}/report/report_latest.md"

# ── Cleanup ────────────────────────────────────────────────────────────
clean: ## Remove ALL generated results (keeps .gitkeep)
	find results/ -type f ! -name '.gitkeep' -delete
	@echo "Cleaned results/"
