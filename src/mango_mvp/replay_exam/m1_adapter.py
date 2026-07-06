from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_replay_m1_manifest(*, set_path: Path, out_path: Path, command: list[str], repo_root: Path | None = None) -> Path:
    root = (repo_root or Path.cwd()).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    payload = {
        "schema_version": "wappi_replay_m1_manifest_v1",
        "metric": "chat_only_replay",
        "head": head,
        "set_path": str(set_path),
        "set_sha256": _sha256(set_path),
        "command": command,
        "live_writes_allowed": False,
        "parallelism": "dialogs_only",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
