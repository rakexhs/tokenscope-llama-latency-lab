"""Load and aggregate benchmark results from the results directory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_aggregate_csv(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load the aggregate CSV (agg_latest.csv) as a list of dicts."""
    path = Path(results_dir) / "summary" / "agg_latest.csv"
    if not path.exists():
        return []
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            typed_row: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    typed_row[k] = int(v)
                except (ValueError, TypeError):
                    try:
                        typed_row[k] = float(v)
                    except (ValueError, TypeError):
                        typed_row[k] = v
            rows.append(typed_row)
    return rows


def load_raw_jsonl(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load all raw JSONL files from results/raw/."""
    raw_dir = Path(results_dir) / "raw"
    if not raw_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for f in sorted(raw_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_manifests(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load all manifest JSON files."""
    report_dir = Path(results_dir) / "report"
    if not report_dir.exists():
        return []
    manifests = []
    for f in sorted(report_dir.glob("manifest_*.json")):
        with open(f) as fh:
            manifests.append(json.load(fh))
    return manifests


def load_decomp_csvs(results_dir: str | Path) -> list[dict[str, Any]]:
    """Load decomposition CSVs."""
    summary_dir = Path(results_dir) / "summary"
    results: list[dict[str, Any]] = []
    for f in sorted(summary_dir.glob("decomp_*.csv")):
        with open(f) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            results.append({"file": f.name, "components": rows})
    return results


def group_by(rows: list[dict[str, Any]], key: str) -> dict[Any, list[dict[str, Any]]]:
    """Group rows by a given key."""
    groups: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        val = row.get(key)
        groups.setdefault(val, []).append(row)
    return groups


def extract_series(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract x, y arrays from a list of dicts, sorted by x."""
    pairs = [(row[x_key], row[y_key]) for row in rows if x_key in row and y_key in row]
    if not pairs:
        return np.array([]), np.array([])
    pairs.sort(key=lambda p: p[0])
    x = np.array([p[0] for p in pairs], dtype=np.float64)
    y = np.array([p[1] for p in pairs], dtype=np.float64)
    return x, y
