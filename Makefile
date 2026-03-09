.PHONY: help install install-cpu test lint systems \
       bench-cpu bench-gpu bench-mps \
       sweep-seq sweep-models sweep-precision sweep-kv \
       plots report analysis \
       decompose decompose-gpu profiler bandwidth energy \
       full-cpu full-gpu full-mps \
       clean

PYTHON ?= python
# Set SYSTEM= on the command line to skip the interactive prompt.
# Example: make bench-cpu SYSTEM=MacBook_Pro_M3
SYSTEM_FLAG = $(if $(SYSTEM),--system $(SYSTEM),)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  All commands accept SYSTEM=<name> to tag results by machine."
	@echo "  Example: make bench-cpu SYSTEM=MacBook_Pro_M3"

# ── Setup ──────────────────────────────────────────────────────────────
install: ## Install package with all extras (HF + llama.cpp + dev)
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
bench-cpu: ## Run CPU benchmark (tiny-gpt2, no GPU needed)
	$(PYTHON) -m bench.run_bench \
		--config configs/bench_default.yaml $(SYSTEM_FLAG) \
		--override backend=hf device=cpu \
		model.id_or_path=sshleifer/tiny-gpt2 \
		generation.output_length=32 generation.prompt_length=64

bench-gpu: ## Run GPU benchmark (set MODEL=/path/to/model.gguf)
	$(PYTHON) -m bench.run_bench \
		--config configs/bench_default.yaml $(SYSTEM_FLAG) \
		--override backend=llamacpp device=cuda \
		model.id_or_path=$(MODEL)

bench-mps: ## Run Apple Silicon MPS benchmark (tiny-gpt2)
	$(PYTHON) -m bench.run_bench \
		--config configs/bench_default.yaml $(SYSTEM_FLAG) \
		--override backend=hf device=mps \
		model.id_or_path=sshleifer/tiny-gpt2 \
		generation.output_length=32 generation.prompt_length=64

# ── Sweeps ─────────────────────────────────────────────────────────────
sweep-seq: ## Sequence-length sweep
	$(PYTHON) -m bench.sweep --config configs/sweep_sequence.yaml $(SYSTEM_FLAG)

sweep-models: ## Model-size sweep
	$(PYTHON) -m bench.sweep --config configs/sweep_models.yaml $(SYSTEM_FLAG)

sweep-precision: ## Precision sweep
	$(PYTHON) -m bench.sweep --config configs/sweep_precision.yaml $(SYSTEM_FLAG)

sweep-kv: ## KV-cache quantization sweep (set MODEL= for GGUF)
	$(PYTHON) -m bench.sweep --config configs/sweep_kv_cache.yaml $(SYSTEM_FLAG)

# ── Profiling ──────────────────────────────────────────────────────────
decompose: ## Latency decomposition on CPU (tiny-gpt2)
	$(PYTHON) -m profiling.decompose_decode \
		--model sshleifer/tiny-gpt2 --device cpu --n_tokens 16 $(SYSTEM_FLAG)

decompose-gpu: ## Latency decomposition on CUDA GPU (tiny-gpt2)
	$(PYTHON) -m profiling.decompose_decode \
		--model sshleifer/tiny-gpt2 --device cuda --n_tokens 16 $(SYSTEM_FLAG)

profiler: ## torch.profiler operator analysis (CPU)
	$(PYTHON) -m profiling.torch_profiler_decode \
		--model sshleifer/tiny-gpt2 --device cpu $(SYSTEM_FLAG)

bandwidth: ## Memory bandwidth micro-benchmark
	$(PYTHON) -m analysis.bandwidth_microbench --device cpu $(SYSTEM_FLAG)

energy: ## Energy-per-token estimation (NVIDIA GPU only, safe skip otherwise)
	$(PYTHON) -m analysis.energy_estimation $(SYSTEM_FLAG)

# ── Analysis ───────────────────────────────────────────────────────────
plots: ## Generate all plots from results
	$(PYTHON) -m analysis.make_plots $(SYSTEM_FLAG)

report: ## Generate findings report (Markdown)
	$(PYTHON) -m analysis.findings_report $(SYSTEM_FLAG)

analysis: plots report ## Run full analysis pipeline (plots + report)

# ── Full Pipelines (one command to run everything) ─────────────────────
full-cpu: test bench-cpu sweep-seq decompose bandwidth plots report ## Full pipeline for CPU-only machine
	@echo "\n[TokenScope] Full CPU pipeline complete. See results/$${SYSTEM:-<system>}/report/report_latest.md"

full-gpu: test bench-gpu sweep-seq decompose-gpu bandwidth energy plots report ## Full pipeline for NVIDIA GPU machine
	@echo "\n[TokenScope] Full GPU pipeline complete. See results/$${SYSTEM:-<system>}/report/report_latest.md"

full-mps: test bench-mps sweep-seq decompose bandwidth plots report ## Full pipeline for Apple Silicon Mac
	@echo "\n[TokenScope] Full MPS pipeline complete. See results/$${SYSTEM:-<system>}/report/report_latest.md"

# ── Cleanup ────────────────────────────────────────────────────────────
clean: ## Remove ALL generated results (keeps .gitkeep)
	find results/ -type f ! -name '.gitkeep' -delete
	@echo "Cleaned results/"
