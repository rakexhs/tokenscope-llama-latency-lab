"""Token-level timestamp tracing for per-token latency measurement."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TokenTrace:
    """Records the precise timestamp of each emitted token."""

    start_ns: int = 0
    token_timestamps_ns: list[int] = field(default_factory=list)

    def mark_start(self) -> None:
        self.start_ns = time.perf_counter_ns()

    def mark_token(self) -> None:
        self.token_timestamps_ns.append(time.perf_counter_ns())

    @property
    def ttft_ms(self) -> float:
        """Time to first token in milliseconds."""
        if not self.token_timestamps_ns:
            return 0.0
        return (self.token_timestamps_ns[0] - self.start_ns) / 1_000_000

    @property
    def per_token_ms(self) -> list[float]:
        """Inter-token latency in milliseconds for each decode step."""
        if len(self.token_timestamps_ns) < 2:
            return []
        deltas: list[float] = []
        prev = self.token_timestamps_ns[0]
        for ts in self.token_timestamps_ns[1:]:
            deltas.append((ts - prev) / 1_000_000)
            prev = ts
        return deltas

    @property
    def end_to_end_ms(self) -> float:
        if not self.token_timestamps_ns:
            return 0.0
        return (self.token_timestamps_ns[-1] - self.start_ns) / 1_000_000

    @property
    def n_tokens(self) -> int:
        return len(self.token_timestamps_ns)

    @property
    def throughput_tok_s(self) -> float:
        e2e = self.end_to_end_ms
        if e2e <= 0:
            return 0.0
        return self.n_tokens / (e2e / 1000)
