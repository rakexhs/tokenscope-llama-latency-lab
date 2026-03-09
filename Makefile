.PHONY: help install install-cpu test lint systems \
       bench-cpu bench-gpu bench-mps \
       sweep-seq sweep-seq-gpu sweep-models sweep-precision sweep-kv \
       plots report analysis \
       decompose decompose-gpu profiler \
       bandwidth bandwidth-cpu bandwidth-gpu bandwidth-mps energy \
       full-cpu full-gpu full-mps \
       clean _require-model

PYTHON ?= python
# Set SYSTEM= on the command line to skip the interactive prompt.
# Example: make bench-cpu SYSTEM=MacBook_Pro_M3
SYSTEM_FLAG = $(if $(SYSTEM),--system $(SYSTEM),)
MODEL_OVERRIDE = $(if $(MODEL),--override model.id_or_path=$(MODEL),)

# Guard target: fails with a clear message when MODEL is not set.
_require-model:
ifndef MODEL
	$(error MODEL is required. Usage: make <target> SYSTEM=X MODEL=/path/to/model.gguf)
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  All commands accept SYSTEM=<name> to tag results by machine."
	@echo "  GPU/GGUF commands require MODEL=/path/to/model.gguf."
	@echo "  Example: make bench-gpu SYSTEM=Lab_RTX4090 MODEL=/data/llama-7b-q4.gguf"

# ── Setup ──────────────────────────────────────────────────────────────
install: ## Install with all extras (HF + llama.cpp + dev)
	pip install -e ".[all]"

install-cpu: ## Install with HF backend only (no GPU/llama.cpp deps)
	pip install -e ".[hf,dev]"

test: ## Run test suite (no model weights needed)
	$(PYTHON) -m pytest tests/ -v

lint: ## Run ruff linter
	ruff check bench/ profiling/ analysis/ tests/

systems: ## List all systems with saved results
	@$(PYTHON) -c "from bench.utils.system_name import list_systems; [print(s) for s in list_systems()]"

# ── Benchmarks ─────────────────────────────────────────────────────────
bench-cpu: ## CPU benchmark (tiny-gpt2, no GPU needed)
	$(PYTHON) -m bench.run_bench \
		--config configs/bench_default.yaml $(SYSTEM_FLAG) \
		--override backend=hf device=cpu \
		model.id_or_path=sshleifer/tiny-gpt2 \
		generation.output_length=32 generation.prompt_length=64

bench-gpu: _require-model ## GPU benchmark (requires MODEL=/path/to/model.gguf)
	$(PYTHON) -m bench.run_bench \
		--config configs/bench_default.yaml $(SYSTEM_FLAG) \
		--override backend=llamacpp device=cuda \
		model.id_or_path="$(MODEL)"

bench-mps: ## Apple Silicon MPS benchmark (tiny-gpt2)
	$(PYTHON) -m bench.run_bench \
		--config configs/bench_default.yaml $(SYSTEM_FLAG) \
		--override backend=hf device=mps \
		model.id_or_path=sshleifer/tiny-gpt2 \
		generation.output_length=32 generation.prompt_length=64

# ── Sweeps ─────────────────────────────────────────────────────────────
sweep-seq: ## Sequence-length sweep (CPU, tiny-gpt2)
	$(PYTHON) -m bench.sweep --config configs/sweep_sequence.yaml $(SYSTEM_FLAG)

sweep-seq-gpu: _require-model ## Sequence-length sweep on GPU (requires MODEL=)
	$(PYTHON) -m bench.sweep --config configs/sweep_sequence_gpu.yaml $(SYSTEM_FLAG) \
		--override model.id_or_path="$(MODEL)"

sweep-models: ## Model-size sweep
	$(PYTHON) -m bench.sweep --config configs/sweep_models.yaml $(SYSTEM_FLAG)

sweep-precision: ## Precision sweep
	$(PYTHON) -m bench.sweep --config configs/sweep_precision.yaml $(SYSTEM_FLAG)

sweep-kv: _require-model ## KV-cache quantization sweep (requires MODEL=)
	$(PYTHON) -m bench.sweep --config configs/sweep_kv_cache.yaml $(SYSTEM_FLAG) \
		--override model.id_or_path="$(MODEL)"

# ── Profiling ──────────────────────────────────────────────────────────
decompose: ## Latency decomposition on CPU (tiny-gpt2)
	$(PYTHON) -m profiling.decompose_decode \
		--model sshleifer/tiny-gpt2 --device cpu --n_tokens 16 $(SYSTEM_FLAG)

decompose-gpu: ## Latency decomposition on CUDA GPU (tiny-gpt2, HF backend)
	$(PYTHON) -m profiling.decompose_decode \
		--model sshleifer/tiny-gpt2 --device cuda --n_tokens 16 $(SYSTEM_FLAG)

profiler: ## torch.profiler operator analysis (CPU)
	$(PYTHON) -m profiling.torch_profiler_decode \
		--model sshleifer/tiny-gpt2 --device cpu $(SYSTEM_FLAG)

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

# ── Full Pipelines ─────────────────────────────────────────────────────
full-cpu: test bench-cpu sweep-seq decompose bandwidth-cpu plots report ## Full pipeline — CPU-only machine
	@echo "\n[TokenScope] Full CPU pipeline complete. See results/$${SYSTEM:-<system>}/report/report_latest.md"

full-gpu: _require-model test bench-gpu sweep-seq-gpu decompose-gpu bandwidth-gpu energy plots report ## Full pipeline — NVIDIA GPU (requires MODEL=)
	@echo "\n[TokenScope] Full GPU pipeline complete. See results/$${SYSTEM:-<system>}/report/report_latest.md"

full-mps: test bench-mps sweep-seq decompose bandwidth-cpu plots report ## Full pipeline — Apple Silicon Mac
	@echo "\n[TokenScope] Full MPS pipeline complete. See results/$${SYSTEM:-<system>}/report/report_latest.md"

# ── Cleanup ────────────────────────────────────────────────────────────
clean: ## Remove ALL generated results (keeps .gitkeep)
	find results/ -type f ! -name '.gitkeep' -delete
	@echo "Cleaned results/"
