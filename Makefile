.PHONY: help install test lint bench-cpu bench-gpu sweep-seq sweep-kv plots report clean systems

PYTHON ?= python
# Set SYSTEM= on the command line to skip the interactive prompt.
# Example: make bench-cpu SYSTEM=MacBook_Pro_M3
SYSTEM_FLAG = $(if $(SYSTEM),--system $(SYSTEM),)

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Tip: pass SYSTEM=<name> to skip the interactive system prompt."
	@echo "  Example: make bench-cpu SYSTEM=MacBook_Pro_M3"

install: ## Install package with all extras
	pip install -e ".[all]"

install-cpu: ## Install with HF backend only (no GPU deps)
	pip install -e ".[hf,dev]"

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v

lint: ## Run ruff linter
	ruff check bench/ profiling/ analysis/ tests/

systems: ## List systems with saved results
	@$(PYTHON) -c "from bench.utils.system_name import list_systems; [print(s) for s in list_systems()]"

# ── Benchmarks ─────────────────────────────────────────────────────────
bench-cpu: ## Run minimal CPU benchmark (tiny-gpt2)
	$(PYTHON) -m bench.run_bench \
		--config configs/bench_default.yaml $(SYSTEM_FLAG) \
		--override backend=hf device=cpu \
		model.id_or_path=sshleifer/tiny-gpt2 \
		generation.output_length=32 generation.prompt_length=64

bench-gpu: ## Run GPU benchmark (requires GGUF model path in MODEL env var)
	$(PYTHON) -m bench.run_bench \
		--config configs/bench_default.yaml $(SYSTEM_FLAG) \
		--override backend=llamacpp device=cuda \
		model.id_or_path=$(MODEL)

# ── Sweeps ─────────────────────────────────────────────────────────────
sweep-seq: ## Sequence-length sweep (CPU, tiny model)
	$(PYTHON) -m bench.sweep --config configs/sweep_sequence.yaml $(SYSTEM_FLAG)

sweep-models: ## Model sweep
	$(PYTHON) -m bench.sweep --config configs/sweep_models.yaml $(SYSTEM_FLAG)

sweep-precision: ## Precision sweep
	$(PYTHON) -m bench.sweep --config configs/sweep_precision.yaml $(SYSTEM_FLAG)

sweep-kv: ## KV-cache quantization sweep
	$(PYTHON) -m bench.sweep --config configs/sweep_kv_cache.yaml $(SYSTEM_FLAG)

# ── Analysis ───────────────────────────────────────────────────────────
plots: ## Generate all plots from results
	$(PYTHON) -m analysis.make_plots $(SYSTEM_FLAG)

report: ## Generate findings report
	$(PYTHON) -m analysis.findings_report $(SYSTEM_FLAG)

analysis: plots report ## Run full analysis pipeline

# ── Profiling ──────────────────────────────────────────────────────────
decompose: ## Run latency decomposition (HF backend)
	$(PYTHON) -m profiling.decompose_decode \
		--model sshleifer/tiny-gpt2 --device cpu --n_tokens 16 $(SYSTEM_FLAG)

# ── Cleanup ────────────────────────────────────────────────────────────
clean: ## Remove generated results (keeps .gitkeep)
	find results/ -type f ! -name '.gitkeep' -delete
	@echo "Cleaned results/"
