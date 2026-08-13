from __future__ import annotations

import os
from pathlib import Path


def load_project_env(path: Path | None = None) -> Path | None:
    """Load simple KEY=VALUE settings without overwriting process variables."""
    source = path or (Path(__file__).resolve().parents[2] / ".env")
    if not source.is_file():
        return None
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
            value = value[1:-1]
        if name:
            os.environ.setdefault(name, value)
    return source
