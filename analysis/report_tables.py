"""Generate markdown tables from benchmark results for the findings report."""

from __future__ import annotations

from typing import Any


def summary_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    """Render a list of dicts as a markdown table."""
    if not rows:
        return "_No data._\n"

    if columns is None:
        columns = list(rows[0].keys())

    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body_lines = []
    for row in rows:
        vals = []
        for c in columns:
            v = row.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        body_lines.append("| " + " | ".join(vals) + " |")

    return "\n".join([header, sep] + body_lines) + "\n"


def inflection_table(inflections: list[dict[str, Any]]) -> str:
    """Format inflection points as a markdown table."""
    if not inflections:
        return "_No inflection points detected._\n"

    columns = ["metric", "prompt_length", "slope_before", "slope_after", "ratio"]
    return summary_table(inflections, columns)


def kv_quant_comparison_table(rows: list[dict[str, Any]]) -> str:
    """Format KV-cache quantization comparison."""
    columns = [
        "kv_type_k", "kv_type_v", "prompt_length",
        "per_token_mean_ms", "ttft_mean_ms", "throughput_mean_tok_s",
    ]
    return summary_table(rows, columns)


def regime_summary_table(regime_data: list[dict[str, Any]]) -> str:
    """Summarize regime classifications."""
    columns = ["prompt_length", "regime", "weight_frac", "kv_frac", "overhead_frac"]
    return summary_table(regime_data, columns)
