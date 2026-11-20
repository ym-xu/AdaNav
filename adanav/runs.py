"""Run records shared by the scripts: timestamps, git state, command line, atomic JSON writes."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from adanav.config import REPO_ROOT


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_state() -> dict:
    """HEAD commit and the tracked files with uncommitted changes (best effort)."""
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "modified": git("status", "--porcelain", "--untracked-files=no").splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "modified": []}


def redacted_argv() -> list:
    argv = list(sys.argv)
    for i, a in enumerate(argv):
        if a == "--api-key" and i + 1 < len(argv):
            argv[i + 1] = "***"
        elif a.startswith("--api-key="):
            argv[i] = "--api-key=***"
    return argv


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
