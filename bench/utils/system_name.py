"""System name prompt and results directory resolution.

Every benchmark, profiling, and analysis command routes through here to
determine which system subdirectory to use under results/.

The system name can be provided via:
  1. --system CLI flag (highest priority)
  2. TOKENSCOPE_SYSTEM environment variable
  3. Interactive prompt (fallback)

The resolved results directory becomes: {results_dir}/{system_name}/
This keeps each machine's data fully separated for cross-platform surveys.
"""

from __future__ import annotations

import os
import re
from pathlib import Path


def _sanitize(name: str) -> str:
    """Convert a human-friendly name to a filesystem-safe directory name.

    Replaces spaces/special chars with underscores, strips leading/trailing junk.
    """
    name = name.strip()
    name = re.sub(r"[^\w\-.]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_") or "default_system"


def prompt_system_name(cli_value: str | None = None) -> str:
    """Resolve the system name from CLI flag, env var, or interactive prompt.

    Returns a sanitized, filesystem-safe system name.
    """
    # 1. Explicit CLI flag
    if cli_value:
        return _sanitize(cli_value)

    # 2. Environment variable
    env_val = os.environ.get("TOKENSCOPE_SYSTEM")
    if env_val:
        return _sanitize(env_val)

    # 3. Interactive prompt
    print("\n" + "=" * 60)
    print("  TokenScope — System Identification")
    print("=" * 60)
    print("  Enter a name for this system to organize results.")
    print("  Examples: MacBook_Pro_M3, Lab_RTX4090, Server_A100")
    print("  (Tip: set --system or TOKENSCOPE_SYSTEM to skip this prompt)")
    print()
    raw = input("  Enter System Name: ").strip()
    if not raw:
        raw = "default_system"
    name = _sanitize(raw)
    print(f"  -> Using system name: {name}")
    print("=" * 60 + "\n")
    return name


def resolve_results_dir(
    base_results_dir: str = "results",
    system_name: str | None = None,
    cli_system: str | None = None,
) -> tuple[str, str]:
    """Resolve the per-system results directory.

    Returns (results_dir, system_name) where results_dir is
    '{base_results_dir}/{system_name}'.

    Creates the directory tree if it doesn't exist.
    """
    if system_name is None:
        system_name = prompt_system_name(cli_system)

    results_dir = str(Path(base_results_dir) / system_name)

    # Create standard subdirectories
    for sub in ("raw", "summary", "figures", "report"):
        Path(results_dir, sub).mkdir(parents=True, exist_ok=True)

    return results_dir, system_name


def list_systems(base_results_dir: str = "results") -> list[str]:
    """List all system names that have results."""
    rd = Path(base_results_dir)
    if not rd.exists():
        return []
    return sorted(
        d.name for d in rd.iterdir()
        if d.is_dir() and (d / "summary").exists()
    )
