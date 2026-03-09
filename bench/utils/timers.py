"""High-resolution timing utilities for CPU, CUDA, and MPS."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


def perf_counter_ms() -> float:
    """Return current time in milliseconds (monotonic, high resolution)."""
    return time.perf_counter_ns() / 1_000_000


class CPUTimer:
    """Simple wall-clock timer using perf_counter_ns."""

    def __init__(self) -> None:
        self._start: int = 0
        self._end: int = 0

    def start(self) -> None:
        self._start = time.perf_counter_ns()

    def stop(self) -> None:
        self._end = time.perf_counter_ns()

    def elapsed_ms(self) -> float:
        return (self._end - self._start) / 1_000_000


class CUDATimer:
    """CUDA event-based timer for accurate GPU timing without excessive syncs."""

    def __init__(self) -> None:
        import torch

        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._start.record()  # type: ignore[union-attr]

    def stop(self) -> None:
        self._end.record()  # type: ignore[union-attr]
        import torch

        torch.cuda.synchronize()

    def elapsed_ms(self) -> float:
        return self._start.elapsed_time(self._end)  # type: ignore[union-attr]


def get_timer(device: str) -> CPUTimer | CUDATimer:
    """Factory: return the best timer for the given device string."""
    if device.startswith("cuda"):
        try:
            return CUDATimer()
        except Exception:
            pass
    # MPS and CPU both use wall-clock; MPS limitation is documented
    return CPUTimer()


@contextmanager
def timed_section(device: str = "cpu") -> Generator[dict[str, float], None, None]:
    """Context manager that yields a dict; on exit, populates 'elapsed_ms'."""
    result: dict[str, float] = {}
    timer = get_timer(device)
    timer.start()
    try:
        yield result
    finally:
        timer.stop()
        result["elapsed_ms"] = timer.elapsed_ms()


def sync_device(device: str) -> None:
    """Issue a synchronization barrier for the device if needed."""
    if device.startswith("cuda"):
        try:
            import torch

            torch.cuda.synchronize()
        except Exception:
            pass
    elif device == "mps":
        try:
            import torch

            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass
