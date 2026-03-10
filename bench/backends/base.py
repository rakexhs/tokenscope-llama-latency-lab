"""Abstract base class for benchmark backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from bench.results_schema import TrialRecord
from bench.utils.token_tracing import TokenTrace

ProgressCallback = Callable[[int, int], None]


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
        progress_callback: ProgressCallback | None = None,
        *,
        batch_size: int = 1,
    ) -> TokenTrace:
        """Run a single generation, returning a TokenTrace with per-token timestamps.

        Args:
            prompt: The prompt string to condition on.  For batch_size > 1, the same
                prompt will be replicated across the batch.
            output_length: Number of tokens to generate (per sequence).
            temperature: Sampling temperature.
            top_p: Nucleus sampling top_p.
            seed: Random seed for deterministic sampling.
            progress_callback: Optional callback to report progress.
            batch_size: Number of sequences to generate concurrently.  Backends may
                support only batch_size=1; unsupported batch sizes should raise
                NotImplementedError.
        """

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
        progress_callback: ProgressCallback | None = None,
        *,
        batch_size: int = 1,
    ) -> TrialRecord:
        """Run a single trial and return a TrialRecord."""
        trace = self.generate_traced(
            prompt,
            output_length,
            temperature,
            top_p,
            seed,
            progress_callback=progress_callback,
            batch_size=batch_size,
        )
        return TrialRecord(
            trial_idx=trial_idx,
            ttft_ms=trace.ttft_ms,
            end_to_end_ms=trace.end_to_end_ms,
            throughput_tok_s=trace.throughput_tok_s,
            per_token_ms=trace.per_token_ms,
            generated_tokens=trace.n_tokens,
        )
