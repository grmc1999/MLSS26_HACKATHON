"""Shared helpers for the autoresearch pipeline scripts.

Usage:
    from pipeline_utils import utc_now_iso, ensure_parent_dir, write_json, ...
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def reject_empty_required(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"Required field '{field_name}' is None")
    s = str(value).strip()
    if not s or s.lower() in ("n/a", "unknown", "na", "none", ""):
        raise ValueError(f"Required field '{field_name}' has invalid value: {repr(value)}")
    return s


def run_command(
    cmd: list[str],
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, timeout=timeout, capture_output=True, text=True)


def parse_shape(s: str) -> tuple[int, ...]:
    s = s.strip().strip("()[]").replace(" ", "")
    if not s:
        raise ValueError(f"Cannot parse empty shape: {repr(s)}")
    return tuple(int(x) for x in s.split(","))


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def get_git_commit_hash() -> Optional[str]:
    try:
        r = run_command(["git", "rev-parse", "HEAD"])
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def get_git_status_dirty() -> bool:
    try:
        r = run_command(["git", "status", "--porcelain"])
        return bool(r.stdout.strip()) if r.returncode == 0 else False
    except Exception:
        return False
