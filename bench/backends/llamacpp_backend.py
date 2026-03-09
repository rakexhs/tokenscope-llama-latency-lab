"""llama.cpp backend via llama-cpp-python with KV-cache quantization support."""

from __future__ import annotations

import os
from typing import Any

from bench.backends.base import Backend, ProgressCallback
from bench.utils.token_tracing import TokenTrace

# Mapping from config strings to llama-cpp-python KV type constants.
# These match the GGMLType enum values used by llama.cpp.
_KV_TYPE_MAP = {
    "f16": 1,    # GGML_TYPE_F16
    "q8_0": 8,   # GGML_TYPE_Q8_0
    "q4_0": 2,   # GGML_TYPE_Q4_0
}


class LlamaCppBackend(Backend):
    """llama.cpp backend using llama-cpp-python for GGUF model inference."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.llm = None
        self.device_str = config.get("device", "cpu")
        self.model_path = config.get("model", {}).get("id_or_path", "")
        llama_cfg = config.get("llamacpp", {})
        self.kv_type_k = llama_cfg.get("kv_type_k", "f16")
        self.kv_type_v = llama_cfg.get("kv_type_v", "f16")
        self.n_threads = llama_cfg.get("n_threads", 0) or os.cpu_count() or 4
        self.n_gpu_layers = llama_cfg.get("n_gpu_layers", 0)

    def load_model(self) -> None:
        from llama_cpp import Llama

        kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_ctx": 4096,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": False,
        }

        # KV-cache quantization: set type_k and type_v if supported
        kv_k = _KV_TYPE_MAP.get(self.kv_type_k)
        kv_v = _KV_TYPE_MAP.get(self.kv_type_v)
        if kv_k is not None:
            kwargs["type_k"] = kv_k
        if kv_v is not None:
            kwargs["type_v"] = kv_v

        try:
            self.llm = Llama(**kwargs)
        except TypeError:
            # Older llama-cpp-python may not support type_k/type_v
            kwargs.pop("type_k", None)
            kwargs.pop("type_v", None)
            self.llm = Llama(**kwargs)
            if self.kv_type_k != "f16" or self.kv_type_v != "f16":
                print(
                    f"[WARN] KV type args not supported by this llama-cpp-python version. "
                    f"Falling back to default (f16)."
                )
                self.kv_type_k = "f16"
                self.kv_type_v = "f16"

    def name(self) -> str:
        return "llamacpp"

    def model_info(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "kv_type_k": self.kv_type_k,
            "kv_type_v": self.kv_type_v,
            "n_threads": self.n_threads,
            "n_gpu_layers": self.n_gpu_layers,
        }

    def generate_traced(
        self,
        prompt: str,
        output_length: int,
        temperature: float,
        top_p: float,
        seed: int,
        progress_callback: ProgressCallback | None = None,
    ) -> TokenTrace:
        """Stream tokens from llama.cpp, recording emission timestamps."""
        assert self.llm is not None

        trace = TokenTrace()
        trace.mark_start()
        emitted_tokens = 0

        gen_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": output_length,
            "stream": True,
            "echo": False,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        else:
            gen_kwargs["temperature"] = 0.01  # llama.cpp requires > 0
            gen_kwargs["top_p"] = 1.0

        if seed >= 0:
            gen_kwargs["seed"] = seed

        for chunk in self.llm.create_completion(**gen_kwargs):
            text = chunk["choices"][0].get("text", "")
            if text:
                trace.mark_token()
                emitted_tokens += 1
                if progress_callback is not None:
                    progress_callback(emitted_tokens, output_length)
            if chunk["choices"][0].get("finish_reason") is not None:
                break

        return trace

    def unload(self) -> None:
        del self.llm
        self.llm = None
