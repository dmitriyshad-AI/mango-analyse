from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def cli_run_out_root(
    *,
    project_root: Path,
    out_root: Path,
    generated_at: datetime | None = None,
) -> Path:
    root = project_root.expanduser().resolve(strict=False)
    base = out_root.expanduser()
    if not base.is_absolute():
        base = root / base
    base = base.resolve(strict=False)
    stamp = _timestamp(generated_at or datetime.now(timezone.utc))
    first = base / f"run_{stamp}"
    if not first.exists():
        return first
    for index in range(2, 10_000):
        candidate = base / f"run_{stamp}_{index:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique run output directory under {base}")


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
