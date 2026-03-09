"""torch.profiler-based decode profiling for HF models.

Captures a short decode window and exports top operators by CPU/CUDA time.

Usage:
    python -m profiling.torch_profiler_decode --model sshleifer/tiny-gpt2 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from bench.utils.io import atomic_write
from bench.utils.prompts import make_prompt


def profile_decode(
    model_id: str = "sshleifer/tiny-gpt2",
    device: str = "cpu",
    prompt_length: int = 64,
    n_tokens: int = 8,
    results_dir: str = "results",
) -> str:
    """Profile decode steps and save operator-level timing CSV + markdown."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model.to(device).eval()

    prompt = make_prompt(prompt_length, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)

    with torch.inference_mode():
        # Warmup
        out = model(input_ids=inputs["input_ids"], use_cache=True)
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1)

        with profile(activities=activities, record_shapes=True) as prof:
            for step in range(n_tokens):
                with record_function(f"decode_step_{step}"):
                    out = model(
                        input_ids=next_tok.unsqueeze(0),
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = out.past_key_values
                    next_tok = out.logits[:, -1, :].argmax(dim=-1)

    # Extract table
    key_averages = prof.key_averages()
    table_str = key_averages.table(sort_by="cpu_time_total", row_limit=30)

    # Build CSV
    csv_lines = ["name,cpu_time_total_us,cuda_time_total_us,calls"]
    for evt in sorted(key_averages, key=lambda e: -e.cpu_time_total):
        cuda_time = evt.cuda_time_total if hasattr(evt, "cuda_time_total") else 0
        csv_lines.append(f'"{evt.key}",{evt.cpu_time_total},{cuda_time},{evt.count}')

    out_dir = Path(results_dir) / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "torch_profile_ops.csv"
    atomic_write(csv_path, "\n".join(csv_lines) + "\n")

    md_path = out_dir / "torch_profile_ops.md"
    md_content = f"""# torch.profiler Operator Summary

Model: `{model_id}` | Device: `{device}` | Decode tokens: {n_tokens}

## How to Read This Table

- **cpu_time_total**: Wall-clock time spent in this operator on CPU (microseconds).
- **cuda_time_total**: Time spent on GPU kernels (only with CUDA profiling).
- **calls**: Number of invocations across all decode steps.
- Top operators by CPU time reveal where compute is spent.
- For memory-bound decode, `aten::mm` / `aten::addmm` (matrix multiply) dominate.

## Top Operators

```
{table_str}
```

Raw CSV: `{csv_path}`
"""
    atomic_write(md_path, md_content)

    print(f"[Profiler] Saved: {csv_path}")
    print(f"[Profiler] Saved: {md_path}")
    print(table_str)

    return str(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="torch.profiler decode analysis")
    parser.add_argument("--model", type=str, default="sshleifer/tiny-gpt2")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prompt_length", type=int, default=64)
    parser.add_argument("--n_tokens", type=int, default=8)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument(
        "--system", type=str, default=None,
        help="System name for organizing results (prompted if not provided)",
    )
    args = parser.parse_args()

    from bench.utils.system_name import resolve_results_dir

    results_dir, _ = resolve_results_dir(args.results_dir, cli_system=args.system)

    profile_decode(
        model_id=args.model,
        device=args.device,
        prompt_length=args.prompt_length,
        n_tokens=args.n_tokens,
        results_dir=results_dir,
    )


if __name__ == "__main__":
    main()
