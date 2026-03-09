"""Atomic file I/O utilities for JSONL and CSV output."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any


def atomic_write(path: str | Path, content: str) -> None:
    """Write content to a file atomically (write-to-temp then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str) + "\n"
    with open(path, "a") as f:
        f.write(line)


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write a list of records as a JSONL file atomically."""
    lines = [json.dumps(r, default=str) for r in records]
    atomic_write(path, "\n".join(lines) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    records: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dicts as CSV atomically."""
    if not rows:
        return
    path = Path(path)
    fieldnames = list(rows[0].keys())
    lines: list[str] = []
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    atomic_write(path, buf.getvalue())


def append_csv(path: str | Path, row: dict[str, Any]) -> None:
    """Append a single row to a CSV. Creates header if file doesn't exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_dataclass_csv(path: str | Path, items: list[Any]) -> None:
    """Save a list of dataclass instances to CSV."""
    if not items:
        return
    rows = [asdict(item) for item in items]
    write_csv(path, rows)


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON data atomically."""
    content = json.dumps(data, indent=2, default=str)
    atomic_write(path, content + "\n")
