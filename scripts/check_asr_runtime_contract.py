#!/usr/bin/env python3
"""Read-only ASR runtime preflight.

The script checks which local Python environment can run the Mango ASR UI
without mutating project data. It does not start ASR, does not touch SQLite
state, and does not write files.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_MODULES = (
    "sqlalchemy",
    "dotenv",
    "mango_mvp.cli",
    "mango_mvp.gui",
    "mlx_whisper",
    "gigaam",
)

CANDIDATE_PYTHONS = (
    ".venv-asrbench/bin/python",
    ".venv/bin/python",
    "stable_runtime/venv_stable/bin/python",
    "/Users/dmitrijfabarisov/.codex/skill-venv/bin/python",
    "/usr/bin/python3",
)

BROKEN_VENV_PATTERN = "venv_stable.broken_20260407"
SCAN_ROOTS = ("docs", "scripts", "stable_runtime")
SCAN_GLOBS = ("*.md", "*.py", "*.sh")

ACTIVE_ASR_BATCH_DB = (
    "product_data/mango_update_after_20260512_20260521_v1/"
    "asr_ui_batch/mango_after_20260512_asr_only.sqlite"
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _check_python(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    result: dict[str, Any] = {
        "path": _display_path(path),
        "exists": path.exists(),
        "ok": False,
        "missing_modules": list(REQUIRED_MODULES),
        "error": "",
    }
    if not path.exists():
        result["error"] = "python_not_found"
        return result

    code = r"""
import importlib.util
import json

mods = %r
missing = []
errors = {}
for mod in mods:
    try:
        spec = importlib.util.find_spec(mod)
    except Exception as exc:
        spec = None
        errors[mod] = repr(exc)
    if spec is None:
        missing.append(mod)
print(json.dumps({"missing_modules": missing, "errors": errors}, ensure_ascii=False))
""" % (REQUIRED_MODULES,)
    env = os.environ.copy()
    src = str(PROJECT_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    try:
        proc = subprocess.run(
            [str(path), "-c", code],
            cwd=str(PROJECT_ROOT),
            env=env,
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - defensive local preflight
        result["error"] = f"probe_failed: {exc!r}"
        return result

    if proc.returncode != 0:
        result["error"] = (proc.stderr or proc.stdout or f"exit_{proc.returncode}").strip()
        return result

    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:  # pragma: no cover - defensive local preflight
        result["error"] = f"bad_probe_json: {exc!r}; stdout={proc.stdout!r}; stderr={proc.stderr!r}"
        return result

    missing = list(payload.get("missing_modules") or [])
    result["missing_modules"] = missing
    result["module_errors"] = payload.get("errors") or {}
    result["ok"] = not missing
    return result


def _find_broken_venv_references() -> list[str]:
    cmd = [
        "rg",
        "--files-with-matches",
        BROKEN_VENV_PATTERN,
        *SCAN_ROOTS,
    ]
    for glob in SCAN_GLOBS:
        cmd.extend(["-g", glob])
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=False,
            text=True,
            capture_output=True,
            timeout=60,
        )
    except FileNotFoundError:
        return ["rg_not_available"]
    if proc.returncode not in (0, 1):
        return [f"rg_failed: {(proc.stderr or proc.stdout).strip()}"]
    refs = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line:
            refs.append(line)
    return sorted(refs)


def build_report() -> dict[str, Any]:
    python_checks = [_check_python(path) for path in CANDIDATE_PYTHONS]
    passing = [item for item in python_checks if item.get("ok")]
    preferred = passing[0]["path"] if passing else ""
    active_db = PROJECT_ROOT / ACTIVE_ASR_BATCH_DB
    broken_refs = _find_broken_venv_references()
    runtime_launchers = [
        path
        for path in broken_refs
        if path.startswith("stable_runtime/") and path.endswith(".sh")
    ]
    docs_refs = [path for path in broken_refs if path.startswith("docs/")]
    script_refs = [path for path in broken_refs if path.startswith("scripts/")]

    return {
        "schema_version": "asr_runtime_contract_preflight_v1",
        "project_root": str(PROJECT_ROOT),
        "required_modules": list(REQUIRED_MODULES),
        "preferred_python": preferred,
        "active_runtime_ok": bool(passing),
        "python_checks": python_checks,
        "active_asr_batch_db": {
            "path": ACTIVE_ASR_BATCH_DB,
            "exists": active_db.exists(),
            "wal_exists": active_db.with_suffix(active_db.suffix + "-wal").exists(),
            "shm_exists": active_db.with_suffix(active_db.suffix + "-shm").exists(),
        },
        "legacy_broken_venv_references": {
            "pattern": BROKEN_VENV_PATTERN,
            "count": len(broken_refs),
            "files": broken_refs,
            "runtime_launcher_count": len(runtime_launchers),
            "runtime_launcher_files": runtime_launchers,
            "docs_reference_count": len(docs_refs),
            "script_reference_count": len(script_refs),
        },
        "safe_to_start_asr_ui": bool(passing),
        "notes": [
            "Read-only check only: no ASR, no DB writes, no file mutations.",
            "Legacy broken-venv references are warnings; do not use those old launchers without updating them.",
            ".venv-asrbench is the current preferred ASR runtime if it passes this check.",
        ],
    }


def main() -> int:
    report = build_report()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["active_runtime_ok"]:
        print(f"\nOK: ASR runtime is available: {report['preferred_python']}")
        if report["legacy_broken_venv_references"]["count"]:
            launcher_count = report["legacy_broken_venv_references"]["runtime_launcher_count"]
            print(
                "WARN: found legacy references to deleted "
                f"{BROKEN_VENV_PATTERN}: "
                f"{report['legacy_broken_venv_references']['count']} files "
                f"({launcher_count} stable_runtime shell launchers)"
            )
        return 0

    print("\nBLOCKED: no Python environment satisfies ASR runtime contract.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
