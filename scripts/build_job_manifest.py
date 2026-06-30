#!/usr/bin/env python3
"""Build a small git-based M1 eval job manifest.

The manifest intentionally references tracked repository files by relative path
and SHA256 instead of copying a full repository bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "mango_git_job_manifest_v1_2026_06_30"
DEFAULT_SNAPSHOT = "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json"
DEFAULT_OUT_DIR = Path.home() / "Yandex.Disk.localized" / "OpenClaw" / "Actual Mango Tests" / "_jobs"
SAFE_ENV_NAME_RE = re.compile(r"^(TELEGRAM_[A-Z0-9_]+|DIALOGUE_CONTRACT_DEBUG_TRACE)$")
SAFE_ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_.:/\\-]{0,128}$")


class ManifestError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def discover_repo_root(start: Path) -> Path:
    try:
        return Path(git_output(start, "rev-parse", "--show-toplevel")).resolve()
    except subprocess.CalledProcessError as exc:
        raise ManifestError(f"not a git repository: {start}") from exc


def normalize_rel_path(repo: Path, raw: str, *, label: str) -> str:
    value = str(raw or "").strip()
    if not value:
        raise ManifestError(f"{label} is required")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ManifestError(f"{label} must be a repository-relative path")
    full = (repo / candidate).resolve(strict=False)
    repo_resolved = repo.resolve()
    if repo_resolved not in (full, *full.parents):
        raise ManifestError(f"{label} escapes repository root")
    if not full.is_file():
        raise ManifestError(f"{label} not found: {value}")
    if full.is_symlink():
        raise ManifestError(f"{label} must not be a symlink: {value}")
    return candidate.as_posix()


def ensure_tracked(repo: Path, rel_path: str, *, label: str) -> None:
    result = subprocess.run(["git", "ls-files", "--error-unmatch", rel_path], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise ManifestError(f"{label} is not tracked by git: {rel_path}")


def parse_env_flags(values: Sequence[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in values:
        key, sep, value = str(raw).partition("=")
        if not sep:
            raise ManifestError(f"env flag must be KEY=VALUE: {raw}")
        key = key.strip()
        value = value.strip()
        if not SAFE_ENV_NAME_RE.fullmatch(key):
            raise ManifestError(f"env flag key is not allowed: {key}")
        if not SAFE_ENV_VALUE_RE.fullmatch(value):
            raise ManifestError(f"env flag value is not allowed for {key}")
        env[key] = value
    return env


def merge_env_json(env: dict[str, str], raw_json: str | None) -> dict[str, str]:
    if not raw_json:
        return env
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ManifestError("--env-flags-json must be a JSON object") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError("--env-flags-json must be a JSON object")
    merged = dict(env)
    for key, value in payload.items():
        item = parse_env_flags([f"{key}={value}"])
        merged.update(item)
    return merged


def build_run_cmd(
    *,
    set_rel_path: str,
    snapshot_rel_path: str,
    env_flags: Mapping[str, str],
    parallel: int,
    judge_prompt_version: str,
    out_dir: str,
    extra_args: Sequence[str],
) -> list[str]:
    command = [
        "env",
        *[f"{key}={value}" for key, value in sorted(env_flags.items())],
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONPATH=src",
        "python3",
        "scripts/run_telegram_dynamic_client_sim.py",
        "--scenarios",
        set_rel_path,
        "--snapshot",
        snapshot_rel_path,
        "--parallel",
        str(parallel),
        "--judge-prompt-version",
        judge_prompt_version,
        "--out-dir",
        out_dir,
        *extra_args,
    ]
    return command


def build_manifest(
    *,
    repo: Path,
    set_rel_path: str,
    snapshot_rel_path: str,
    env_flags: Mapping[str, str],
    parallel: int,
    max_hours: float,
    judge_prompt_version: str = "v9.1",
    run_out_dir: str | None = None,
    extra_args: Sequence[str] = (),
    commit_sha: str | None = None,
) -> dict[str, object]:
    repo = repo.resolve()
    commit = str(commit_sha or git_output(repo, "rev-parse", "HEAD")).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ManifestError("commit_sha must be a 40-char git SHA")
    set_rel = normalize_rel_path(repo, set_rel_path, label="set_rel_path")
    snapshot_rel = normalize_rel_path(repo, snapshot_rel_path, label="snapshot_rel_path")
    ensure_tracked(repo, set_rel, label="set_rel_path")
    ensure_tracked(repo, snapshot_rel, label="snapshot_rel_path")
    parallel_i = int(parallel)
    max_hours_f = float(max_hours)
    if parallel_i < 1 or parallel_i > 8:
        raise ManifestError("parallel must be 1..8")
    if max_hours_f <= 0 or max_hours_f > 24:
        raise ManifestError("max_hours must be >0 and <=24")
    run_dir = run_out_dir or f"runs/job_{commit[:12]}"
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "commit_sha": commit,
        "set_rel_path": set_rel,
        "set_sha256": sha256_file(repo / set_rel),
        "set_size_bytes": (repo / set_rel).stat().st_size,
        "snapshot_rel_path": snapshot_rel,
        "snapshot_sha256": sha256_file(repo / snapshot_rel),
        "snapshot_size_bytes": (repo / snapshot_rel).stat().st_size,
        "env_flags": dict(sorted(env_flags.items())),
        "parallel": parallel_i,
        "max_hours": max_hours_f,
        "judge_prompt_version": judge_prompt_version,
        "run_out_dir": run_dir,
        "run_cmd": build_run_cmd(
            set_rel_path=set_rel,
            snapshot_rel_path=snapshot_rel,
            env_flags=env_flags,
            parallel=parallel_i,
            judge_prompt_version=judge_prompt_version,
            out_dir=run_dir,
            extra_args=tuple(extra_args),
        ),
        "m1_manual_procedure": [
            "git fetch yandex",
            f"git checkout {commit}",
            f"test \"$(git rev-parse HEAD)\" = \"{commit}\"",
            f"shasum -a 256 {set_rel}",
            f"shasum -a 256 {snapshot_rel}",
            "Run run_cmd from this manifest.",
        ],
    }
    return payload


def write_manifest(out_dir: Path, manifest: Mapping[str, object], *, force: bool = False) -> Path:
    commit = str(manifest["commit_sha"])
    out_dir = out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"job_{commit[:12]}.json"
    if path.exists() and not force:
        raise ManifestError(f"manifest already exists: {path}")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small git checkout based M1 eval job manifest.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--set", dest="set_rel_path", required=True, help="Tracked repo-relative scenario/eval set path.")
    parser.add_argument("--snapshot", dest="snapshot_rel_path", default=DEFAULT_SNAPSHOT, help="Tracked repo-relative KB snapshot path.")
    parser.add_argument("--env-flag", action="append", default=[], help="Safe env flag as KEY=VALUE. Only TELEGRAM_* and DIALOGUE_CONTRACT_DEBUG_TRACE are allowed.")
    parser.add_argument("--env-flags-json", help="JSON object with safe env flags.")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=3)
    parser.add_argument("--judge-prompt-version", default="v9.1")
    parser.add_argument("--run-out-dir", help="Out dir passed to run_telegram_dynamic_client_sim.py.")
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra argument appended to run_telegram_dynamic_client_sim.py command.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--commit-sha", help="Override commit SHA; defaults to git HEAD.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        repo = discover_repo_root(args.repo.resolve())
        env_flags = merge_env_json(parse_env_flags(args.env_flag), args.env_flags_json)
        manifest = build_manifest(
            repo=repo,
            set_rel_path=args.set_rel_path,
            snapshot_rel_path=args.snapshot_rel_path,
            env_flags=env_flags,
            parallel=args.parallel,
            max_hours=args.max_hours,
            judge_prompt_version=args.judge_prompt_version,
            run_out_dir=args.run_out_dir,
            extra_args=tuple(args.extra_arg),
            commit_sha=args.commit_sha,
        )
        path = write_manifest(args.out_dir, manifest, force=args.force)
    except (ManifestError, subprocess.CalledProcessError, OSError) as exc:
        print(f"build_job_manifest: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
