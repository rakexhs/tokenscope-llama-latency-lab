"""HuggingFace Transformers backend with loop_decode and generate modes."""

from __future__ import annotations

import time
from typing import Any

import torch

from bench.backends.base import Backend, ProgressCallback
from bench.utils.timers import sync_device
from bench.utils.token_tracing import TokenTrace


def _resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    if dtype_str == "auto":
        if device.startswith("cuda"):
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    return mapping.get(dtype_str, torch.float32)


class HFBackend(Backend):
    """HuggingFace Transformers backend supporting loop_decode and generate modes."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device_str = config.get("device", "cpu")
        hf_cfg = config.get("hf", {})
        self.mode = hf_cfg.get("mode", "loop_decode")
        self.dtype = _resolve_dtype(hf_cfg.get("dtype", "auto"), self.device_str)
        self.model_id = config.get("model", {}).get("id_or_path", "sshleifer/tiny-gpt2")

    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device_str)
        self.model.eval()

    def name(self) -> str:
        return f"hf_{self.mode}"

    def model_info(self) -> dict[str, Any]:
        n_params = sum(p.numel() for p in self.model.parameters()) if self.model else 0
        return {
            "model_id": self.model_id,
            "n_params": n_params,
            "dtype": str(self.dtype),
            "mode": self.mode,
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
        if self.mode == "loop_decode":
            return self._loop_decode(
                prompt,
                output_length,
                temperature,
                top_p,
                seed,
                progress_callback=progress_callback,
            )
        return self._generate_mode(
            prompt,
            output_length,
            temperature,
            top_p,
            seed,
            progress_callback=progress_callback,
        )

    @torch.inference_mode()
    def _loop_decode(
        self,
        prompt: str,
        output_length: int,
        temperature: float,
        top_p: float,
        seed: int,
        progress_callback: ProgressCallback | None = None,
    ) -> TokenTrace:
        """Token-by-token decoding with past_key_values for precise timing."""
        assert self.model is not None and self.tokenizer is not None

        torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device_str)
        input_ids = inputs["input_ids"]

        trace = TokenTrace()
        sync_device(self.device_str)
        trace.mark_start()

        # Prefill: process the full prompt
        outputs = self.model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        next_token = self._sample(logits, temperature, top_p)
        sync_device(self.device_str)
        trace.mark_token()  # TTFT

        generated = [next_token.item()]
        tokens_done = 1
        if progress_callback is not None:
            progress_callback(tokens_done, output_length)

        # Decode loop
        for _ in range(output_length - 1):
            next_input = next_token.view(1, 1)
            outputs = self.model(
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            next_token = self._sample(logits, temperature, top_p)
            sync_device(self.device_str)
            trace.mark_token()
            generated.append(next_token.item())
            tokens_done += 1
            if progress_callback is not None:
                progress_callback(tokens_done, output_length)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return trace

    @torch.inference_mode()
    def _generate_mode(
        self,
        prompt: str,
        output_length: int,
        temperature: float,
        top_p: float,
        seed: int,
        progress_callback: ProgressCallback | None = None,
    ) -> TokenTrace:
        """Use model.generate() — less granular but included for comparison."""
        assert self.model is not None and self.tokenizer is not None

        torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device_str)

        trace = TokenTrace()
        sync_device(self.device_str)
        trace.mark_start()

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": output_length,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        sync_device(self.device_str)

        n_new = output_ids.shape[1] - inputs["input_ids"].shape[1]
        end_ns = time.perf_counter_ns()
        start_ns = trace.start_ns
        # Approximate uniform token spacing (generate mode limitation)
        interval = (end_ns - start_ns) / max(n_new, 1)
        for i in range(n_new):
            trace.token_timestamps_ns.append(int(start_ns + interval * (i + 1)))
        if progress_callback is not None and n_new > 0:
            progress_callback(n_new, max(n_new, output_length))

        return trace

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1).squeeze()
        scaled = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
            cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[mask] = float("-inf")
            scaled = sorted_logits.scatter(1, sorted_idx, sorted_logits)
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs.squeeze(0), 1).squeeze()

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
