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
        *,
        batch_size: int = 1,
    ) -> TokenTrace:
        """Generate tokens with optional batching.

        Dispatch generation based on the configured mode. Supported modes:

          * ``loop_decode`` — baseline autoregressive decoding with per-token tracing.
          * ``generate``    — uses HuggingFace ``model.generate`` for one-shot output (less granular).
          * ``spec_decode`` — speculative decoding.  A draft model proposes multiple tokens
            ahead which the target model verifies in a single forward pass.  This mode
            requires additional settings in the ``hf.spec`` subsection of the config:

            - ``draft_model_id``: identifier of the smaller draft model (e.g. ``sshleifer/tiny-gpt2``).
            - ``draft_steps``: number of tokens the draft model will propose at a time.

        The default is ``loop_decode``.
        """
        if self.mode == "loop_decode":
            return self._loop_decode(
                prompt,
                output_length,
                temperature,
                top_p,
                seed,
                progress_callback=progress_callback,
                batch_size=batch_size,
            )
        elif self.mode == "generate":
            return self._generate_mode(
                prompt,
                output_length,
                temperature,
                top_p,
                seed,
                progress_callback=progress_callback,
                batch_size=batch_size,
            )
        elif self.mode == "spec_decode":
            return self._speculative_decode(
                prompt,
                output_length,
                temperature,
                top_p,
                seed,
                progress_callback=progress_callback,
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Unsupported generation mode: {self.mode}")

    @torch.inference_mode()
    def _loop_decode(
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
        """Token-by-token decoding with past_key_values for precise timing."""
        assert self.model is not None and self.tokenizer is not None

        torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device_str)
        input_ids = inputs["input_ids"]
        # Replicate prompt for batching if needed
        if batch_size > 1:
            # Expand the input_ids to [batch_size, seq_len]
            input_ids = input_ids.expand(batch_size, -1).contiguous()

        trace = TokenTrace()
        sync_device(self.device_str)
        trace.mark_start()

        # Prefill: process the full prompt
        outputs = self.model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values

        next_token = self._sample(logits, temperature, top_p)
        # next_token shape: [batch_size]
        sync_device(self.device_str)
        trace.mark_token()  # TTFT (first decode step)

        tokens_done = 1
        if progress_callback is not None:
            progress_callback(tokens_done, output_length)

        # Decode loop
        # We'll stop early only when all sequences hit EOS
        for _ in range(output_length - 1):
            # next_input: [batch_size, 1]
            next_input = next_token.view(-1, 1)
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
            tokens_done += 1
            if progress_callback is not None:
                progress_callback(tokens_done, output_length)

            # Break if all sequences have generated EOS
            if batch_size > 1:
                # next_token is a tensor of shape [batch_size]
                # If all tokens == eos, stop
                if torch.all(next_token == self.tokenizer.eos_token_id):
                    break
            else:
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
        *,
        batch_size: int = 1,
    ) -> TokenTrace:
        """Use model.generate() — less granular but included for comparison."""
        assert self.model is not None and self.tokenizer is not None

        torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device_str)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        if batch_size > 1:
            input_ids = input_ids.expand(batch_size, -1).contiguous()
            new_inputs: dict[str, Any] = {"input_ids": input_ids}
            if attention_mask is not None:
                new_inputs["attention_mask"] = attention_mask.expand(batch_size, -1).contiguous()
            inputs = new_inputs

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

        n_new = output_ids.shape[1] - input_ids.shape[1]
        end_ns = time.perf_counter_ns()
        start_ns = trace.start_ns
        # Approximate uniform token spacing (generate mode limitation)
        interval = (end_ns - start_ns) / max(n_new, 1)
        for i in range(n_new):
            trace.token_timestamps_ns.append(int(start_ns + interval * (i + 1)))
        if progress_callback is not None and n_new > 0:
            progress_callback(n_new, max(n_new, output_length))

        return trace

    @torch.inference_mode()
    def _speculative_decode(
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
        """Speculative decoding implementation.

        This algorithm pairs a small "draft" model with the main (target) model.  The draft
        model proposes a sequence of ``draft_steps`` tokens ahead, which the target model
        then verifies in one forward pass.  If the target model agrees with the draft, all
        proposed tokens are accepted; otherwise the mismatch position is corrected by the
        target model and generation continues from there.

        See: https://huggingface.co/blog/speculative-decoding for details.

        For simplicity, if a draft model is not provided, this function falls back to
        baseline ``loop_decode``.  Speculative decoding currently supports only
        batch_size=1.  Per-token timing reflects the accepted tokens.
        """
        # Fallback if batch_size > 1: not yet implemented
        if batch_size > 1:
            raise NotImplementedError("speculative decoding currently supports only batch_size=1")

        # Ensure target model is loaded
        assert self.model is not None and self.tokenizer is not None

        # Retrieve speculative configuration from self.config
        hf_cfg: dict[str, Any] = self.config.get("hf", {}) if hasattr(self, "config") else {}
        spec_cfg: dict[str, Any] = hf_cfg.get("spec", {}) if hf_cfg else {}
        draft_model_id = spec_cfg.get("draft_model_id")
        draft_steps: int = int(spec_cfg.get("draft_steps", 4))

        # If no draft model specified, fall back to loop_decode
        if not draft_model_id:
            return self._loop_decode(
                prompt,
                output_length,
                temperature,
                top_p,
                seed,
                progress_callback=progress_callback,
                batch_size=batch_size,
            )

        # Load draft model lazily
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not hasattr(self, "_draft_model"):
            self._draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_id)
            if self._draft_tokenizer.pad_token is None:
                self._draft_tokenizer.pad_token = self._draft_tokenizer.eos_token
            self._draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_id,
                torch_dtype=self.dtype,
            )
            self._draft_model.to(self.device_str)
            self._draft_model.eval()

        draft_model = self._draft_model
        draft_tokenizer = self._draft_tokenizer

        torch.manual_seed(seed)

        # Initial tokenization for target and draft models.  We reuse the same prompt
        inputs_tgt = self.tokenizer(prompt, return_tensors="pt").to(self.device_str)
        inputs_draft = draft_tokenizer(prompt, return_tensors="pt").to(self.device_str)

        input_ids_tgt = inputs_tgt["input_ids"]
        input_ids_draft = inputs_draft["input_ids"]

        trace = TokenTrace()
        # Start timing BEFORE prefill so TTFT matches the harness definition.
        sync_device(self.device_str)
        trace.mark_start()

        # Initialize caches (prefill)
        outputs_tgt = self.model(input_ids=input_ids_tgt, use_cache=True)
        past_tgt = outputs_tgt.past_key_values
        # Last logits
        logits_tgt = outputs_tgt.logits[:, -1, :]

        outputs_draft = draft_model(input_ids=input_ids_draft, use_cache=True)
        past_draft = outputs_draft.past_key_values
        logits_draft = outputs_draft.logits[:, -1, :]

        # Sampling helper for both models
        def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
            return self._sample(logits, temperature, top_p)

        generated_tokens: list[int] = []

        # Generate tokens iteratively
        for _ in range(output_length):
            # Draft proposes draft_steps tokens sequentially
            draft_proposed: list[int] = []
            for _ in range(draft_steps):
                draft_token = sample_from_logits(logits_draft)
                token_id = draft_token.item() if isinstance(draft_token, torch.Tensor) else int(draft_token)
                draft_proposed.append(token_id)

                # Update draft past and logits for next step
                next_input_draft = torch.tensor([[token_id]], device=self.device_str, dtype=input_ids_draft.dtype)
                outputs_draft = draft_model(
                    input_ids=next_input_draft,
                    past_key_values=past_draft,
                    use_cache=True,
                )
                past_draft = outputs_draft.past_key_values
                logits_draft = outputs_draft.logits[:, -1, :]

                if len(generated_tokens) + len(draft_proposed) >= output_length:
                    break

            # Verify proposed tokens using target model in a single pass
            # Build a tensor of shape [1, len(draft_proposed)]
            proposed_tensor = torch.tensor([draft_proposed], device=self.device_str, dtype=input_ids_tgt.dtype)
            outputs_tgt = self.model(
                input_ids=proposed_tensor,
                past_key_values=past_tgt,
                use_cache=True,
            )
            # New logits for each proposed token
            logits_seq = outputs_tgt.logits
            past_tgt = outputs_tgt.past_key_values

            # Determine how many proposals to accept
            accept_count = 0
            for idx, prop_token in enumerate(draft_proposed):
                target_logits = logits_seq[:, idx, :]
                target_next = sample_from_logits(target_logits)
                tgt_id = target_next.item() if isinstance(target_next, torch.Tensor) else int(target_next)
                if tgt_id == prop_token:
                    accept_count += 1
                else:
                    # Mismatch: accept tokens up to this point and break
                    break

            # If zero accepted, use the target's first prediction
            if accept_count == 0:
                next_id = tgt_id
                accept_tokens = [next_id]
            else:
                accept_tokens = draft_proposed[:accept_count]
                # If mismatch occurred after some accepted tokens, append the target-corrected token
                if accept_count < len(draft_proposed):
                    accept_tokens.append(tgt_id)

            # Append accepted tokens to output and update trace
            for token_id in accept_tokens:
                generated_tokens.append(token_id)
                sync_device(self.device_str)
                trace.mark_token()
                if progress_callback is not None:
                    progress_callback(len(generated_tokens), output_length)
                # Stop if EOS or output length reached
                if token_id == self.tokenizer.eos_token_id or len(generated_tokens) >= output_length:
                    break

            # Update past_tgt for any tokens beyond accept_tokens (already done in outputs_tgt)
            # Update past_draft state: if we accepted tokens, we need to update the draft past
            # to reflect consumed tokens.  We regenerate draft past by passing accepted tokens
            # through the draft model sequentially.
            for token_id in accept_tokens:
                next_input_draft = torch.tensor([[token_id]], device=self.device_str, dtype=input_ids_draft.dtype)
                outputs_draft = draft_model(
                    input_ids=next_input_draft,
                    past_key_values=past_draft,
                    use_cache=True,
                )
                past_draft = outputs_draft.past_key_values
                logits_draft = outputs_draft.logits[:, -1, :]

            # If we've generated enough tokens, break out
            if len(generated_tokens) >= output_length:
                break

        return trace

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """Sample next token(s) from logits with optional top_p and temperature.

        Supports both single-example ([vocab]) and batched ([batch_size, vocab]) logits.
        Returns a tensor of shape [batch_size] (or scalar for batch_size=1).
        """
        if temperature <= 0:
            # Greedy argmax over last dimension
            return logits.argmax(dim=-1).squeeze()
        scaled = logits / temperature
        if top_p < 1.0:
            # Apply nucleus sampling per row (batched or unbatched).
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            # Mask tokens where cumulative probability exceeds top_p.
            # Keep the first token above the threshold by shifting the mask right.
            mask = cumprobs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            # Scatter back to original vocabulary order.
            unsorted = torch.full_like(sorted_logits, float("-inf"))
            scaled = unsorted.scatter(dim=-1, index=sorted_idx, src=sorted_logits)
        probs = torch.softmax(scaled, dim=-1)
        if probs.dim() == 1:
            return torch.multinomial(probs, 1).squeeze()
        # probs: [batch_size, vocab] -> sample one token per row
        return torch.multinomial(probs, 1).squeeze(1)

    def unload(self) -> None:
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
