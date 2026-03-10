"""Methodology constants and documentation for the benchmark harness.

Centralizes the measurement methodology so it can be referenced by both
the runtime code and the auto-generated reports.
"""

from __future__ import annotations

METHODOLOGY = {
    "ttft_definition": (
        "Time To First Token (TTFT) is measured as the wall-clock time from "
        "the start of the generation call to the timestamp of the first emitted "
        "token. This includes prompt processing (prefill) and the first decode step."
    ),
    "per_token_definition": (
        "Per-token latency is the inter-token interval: the time between "
        "consecutive token emissions during autoregressive decoding."
    ),
    "e2e_definition": (
        "End-to-end latency is the total wall-clock time from the start of "
        "generation to the emission of the final token."
    ),
    "warmup_rationale": (
        "Warmup runs (default 3) are discarded to allow JIT compilation, "
        "memory allocation, and cache warming to stabilize before measurement."
    ),
    "outlier_handling": (
        "Raw traces are always preserved. Summary statistics are computed on "
        "IQR-filtered data (k=1.5) to reduce the impact of OS scheduling "
        "jitter, GC pauses, and thermal throttling. Both raw and filtered "
        "statistics are reported."
    ),
    "steady_state_rule": (
        "The first N decode tokens (default 2) within each trial are excluded "
        "from steady-state statistics to account for cache warmup effects "
        "within a single generation."
    ),
    "ci_method": (
        "95% confidence intervals are computed via bootstrap resampling "
        "(5000 iterations) of trial-level means."
    ),
    "timing_cpu": "time.perf_counter_ns (monotonic, nanosecond resolution).",
    "timing_cuda": (
        "torch.cuda.Event with enable_timing=True. Events are recorded on "
        "the GPU timeline; elapsed_time synchronizes only at measurement."
    ),
    "timing_mps": (
        "time.perf_counter_ns with torch.mps.synchronize() barriers. "
        "Limitation: MPS lacks event-based timing; wall-clock includes "
        "potential queue delays."
    ),
}


DEFAULT_CONFIG = {
    "backend": "hf",
    "device": "cpu",
    "model": {"id_or_path": "sshleifer/tiny-gpt2"},
    "generation": {
        "prompt_length": 128,
        "output_length": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        # Number of prompts to generate concurrently.  Use 1 for interactive
        # settings; small values up to 4 allow exploration of batch effects.
        "batch_size": 1,
    },
    "benchmark": {
        "warmup_runs": 3,
        "trials": 10,
        "iqr_filter": True,
        "steady_state_skip": 2,
    },
    "llamacpp": {
        "kv_type_k": "f16",
        "kv_type_v": "f16",
        "n_threads": 0,
        "n_gpu_layers": 0,
    },
    "hf": {
        "mode": "loop_decode",
        "dtype": "auto",
        # Speculative decoding configuration.  When mode="spec_decode", the draft
        # model identifier and the number of draft tokens proposed per step must
        # be provided here.  If draft_model_id is not set, speculative decoding
        # falls back to baseline loop_decode.  See bench/backends/hf_backend.py.
        "spec": {
            "draft_model_id": "",
            "draft_steps": 4,
        },
    },
}


def methodology_text() -> str:
    """Return a formatted methodology description for reports."""
    lines = ["## Measurement Methodology\n"]
    for key, val in METHODOLOGY.items():
        label = key.replace("_", " ").title()
        lines.append(f"**{label}:** {val}\n")
    return "\n".join(lines)
