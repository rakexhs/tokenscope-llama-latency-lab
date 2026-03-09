"""Backend registry for benchmark harness."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bench.backends.base import Backend


def get_backend(name: str, config: dict) -> "Backend":
    """Instantiate a backend by name."""
    if name == "hf":
        from bench.backends.hf_backend import HFBackend

        return HFBackend(config)
    elif name == "llamacpp":
        from bench.backends.llamacpp_backend import LlamaCppBackend

        return LlamaCppBackend(config)
    else:
        raise ValueError(f"Unknown backend: {name!r}. Choose 'hf' or 'llamacpp'.")
