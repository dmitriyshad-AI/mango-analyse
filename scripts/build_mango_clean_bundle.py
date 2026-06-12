#!/usr/bin/env python3
"""Build a tracked-file ``mango_clean_<sha>`` bundle with a final manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.m1_watcher import sha256_file, write_text_atomic


DEFAULT_TESTS_ROOT = Path.home() / "Yandex.Disk.localized" / "OpenClaw" / "Actual Mango Tests"
DEFAULT_KB_SNAPSHOT = "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json"
DEFAULT_PILOT_CONFIG = "pilot_gold_v1"
DEFAULT_GOLD_PACK_VERSION = "real_manager_gold_2026-06-08"


def _git(args: list[str], repo: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _tracked_files(repo: Path) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=repo)
    return [repo / item.decode("utf-8") for item in output.split(b"\0") if item]


def write_bundle_manifest(bundle_dir: Path, bundle_id: str, head: str) -> Path:
    files = []
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        rel = path.relative_to(bundle_dir).as_posix()
        files.append({"path": rel, "size": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": "mango_clean_manifest_v1",
        "bundle_id": bundle_id,
        "head": head,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "file_count": len(files),
        "files": files,
    }
    manifest_path = bundle_dir / "manifest.json"
    write_text_atomic(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest_path


def build_bundle(
    repo: Path,
    out_root: Path,
    *,
    kb_snapshot: str = DEFAULT_KB_SNAPSHOT,
    extra_info_lines: tuple[str, ...] = (),
) -> Path:
    head = _git(["rev-parse", "HEAD"], repo)
    short = head[:8]
    bundle_id = f"mango_clean_{short}"
    out_dir = out_root / bundle_id
    if out_dir.exists():
        raise FileExistsError(f"bundle already exists: {out_dir}")
    out_dir.mkdir(parents=True)
    for src in _tracked_files(repo):
        rel = src.relative_to(repo)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    branch = _git(["branch", "--show-current"], repo)
    info_lines = [
        bundle_id,
        f"source: {repo}",
        f"branch: {branch}",
        f"head: {head}",
        f"created: {datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        f"kb_snapshot: {kb_snapshot}",
        f"pilot_config: {DEFAULT_PILOT_CONFIG}",
        f"gold_pack_version: {DEFAULT_GOLD_PACK_VERSION}",
        "judge_prompt_version: v9",
        *extra_info_lines,
        "",
    ]
    info = "\n".join(info_lines)
    write_text_atomic(out_dir / "BUNDLE_INFO.txt", info)
    write_bundle_manifest(out_dir, bundle_id, head)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build mango_clean bundle with manifest.json written last.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--out-root", type=Path, default=DEFAULT_TESTS_ROOT)
    parser.add_argument("--kb-snapshot", default=DEFAULT_KB_SNAPSHOT)
    parser.add_argument("--info-line", action="append", default=[], help="Extra BUNDLE_INFO.txt line, written before manifest.")
    args = parser.parse_args(argv)
    out_dir = build_bundle(
        args.repo.resolve(),
        args.out_root,
        kb_snapshot=args.kb_snapshot,
        extra_info_lines=tuple(args.info_line),
    )
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
