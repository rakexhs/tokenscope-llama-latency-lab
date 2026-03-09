"""Abstract base class for benchmark backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from bench.results_schema import TrialRecord
from bench.utils.token_tracing import TokenTrace


class Backend(ABC):
    """Interface that every backend must implement."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory. Called once before benchmarking."""

    @abstractmethod
    def generate_traced(
        self,
        prompt: str,
        output_length: int,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> TokenTrace:
        """Run a single generation, returning a TokenTrace with per-token timestamps."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""

    @abstractmethod
    def model_info(self) -> dict[str, Any]:
        """Return dict of model metadata (param count, dtype, etc.)."""

    def unload(self) -> None:
        """Optional: release model resources."""

    def run_trial(
        self,
        prompt: str,
        output_length: int,
        trial_idx: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 42,
    ) -> TrialRecord:
        """Run a single trial and return a TrialRecord."""
        trace = self.generate_traced(prompt, output_length, temperature, top_p, seed)
        return TrialRecord(
            trial_idx=trial_idx,
            ttft_ms=trace.ttft_ms,
            end_to_end_ms=trace.end_to_end_ms,
            throughput_tok_s=trace.throughput_tok_s,
            per_token_ms=trace.per_token_ms,
            generated_tokens=trace.n_tokens,
        )
