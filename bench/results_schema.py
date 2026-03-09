"""Canonical schema for benchmark results.

Every raw result record and summary record conforms to these dataclasses
so downstream analysis can rely on stable column names.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EnvironmentSnapshot:
    """Machine / software environment captured at run time."""

    system_name: str = ""
    os: str = ""
    python_version: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    ram_gb: float = 0.0
    gpu_name: str = ""
    vram_gb: float = 0.0
    torch_version: str = ""
    transformers_version: str = ""
    llama_cpp_version: str = ""
    git_sha: str = ""
    timestamp: str = ""


@dataclass
class RunConfig:
    """Flattened view of a single benchmark run configuration."""

    run_id: str = ""
    config_hash: str = ""
    backend: str = ""
    device: str = ""
    model_id: str = ""
    dtype: str = ""
    kv_type_k: str = "f16"
    kv_type_v: str = "f16"
    prompt_length: int = 0
    output_length: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    warmup_runs: int = 3
    trials: int = 10
    iqr_filter: bool = True
    steady_state_skip: int = 2
    n_threads: int = 0
    n_gpu_layers: int = 0


@dataclass
class TrialRecord:
    """Raw timing data from a single trial."""

    trial_idx: int = 0
    ttft_ms: float = 0.0
    end_to_end_ms: float = 0.0
    throughput_tok_s: float = 0.0
    per_token_ms: list[float] = field(default_factory=list)
    generated_tokens: int = 0


@dataclass
class BenchmarkResult:
    """Complete result for one benchmark invocation (all trials)."""

    config: RunConfig = field(default_factory=RunConfig)
    env: EnvironmentSnapshot = field(default_factory=EnvironmentSnapshot)
    trials: list[TrialRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchmarkResult":
        env = EnvironmentSnapshot(**d.get("env", {}))
        config = RunConfig(**d.get("config", {}))
        trials = [TrialRecord(**t) for t in d.get("trials", [])]
        return cls(config=config, env=env, trials=trials, metadata=d.get("metadata", {}))


@dataclass
class SummaryRow:
    """Aggregated summary row for CSV output."""

    run_id: str = ""
    system_name: str = ""
    backend: str = ""
    device: str = ""
    model_id: str = ""
    dtype: str = ""
    kv_type_k: str = "f16"
    kv_type_v: str = "f16"
    prompt_length: int = 0
    output_length: int = 0
    n_trials: int = 0
    ttft_mean_ms: float = 0.0
    ttft_median_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_ci_low_ms: float = 0.0
    ttft_ci_high_ms: float = 0.0
    per_token_mean_ms: float = 0.0
    per_token_median_ms: float = 0.0
    per_token_p95_ms: float = 0.0
    per_token_ci_low_ms: float = 0.0
    per_token_ci_high_ms: float = 0.0
    e2e_mean_ms: float = 0.0
    throughput_mean_tok_s: float = 0.0
    steady_per_token_mean_ms: float = 0.0
    steady_per_token_ci_low_ms: float = 0.0
    steady_per_token_ci_high_ms: float = 0.0

    def header(self) -> list[str]:
        return list(asdict(self).keys())

    def values(self) -> list[Any]:
        return list(asdict(self).values())
