#!/usr/bin/env python3
"""Generate a local project audit report without touching runtime data."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LARGE_FILE_MB = 50


EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
}


RUNTIME_PATTERNS = (
    "stable_runtime",
    "2026-03-09--26",
    "2026-03-05-21-06-49-ч1",
    "2026-03-05-21-06-49-ч2",
    "telegram_exports (2)",
    "external_m1_jan_mar_2025_asr_only_20260504_result_20260506_103021",
    "external_m1_jan2025_test300_20260503",
)


@dataclass(frozen=True)
class PythonDocStats:
    files: int = 0
    modules_with_docstring: int = 0
    public_functions: int = 0
    public_functions_with_docstring: int = 0
    classes: int = 0
    classes_with_docstring: int = 0
    parse_errors: int = 0


def run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return subprocess.run(
        cmd,
        cwd=ROOT,
        check=check,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def iter_files(base: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for filename in filenames:
            files.append(Path(dirpath) / filename)
    return files


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def dir_size(path: Path) -> int:
    total = 0
    if path.is_file():
        return file_size(path)
    for item in iter_files(path):
        total += file_size(item)
    return total


def fmt_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024 or unit == "T":
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}T"


def count_paths() -> dict[str, int]:
    def count_dirs(name: str) -> int:
        total = 0
        for dirpath, dirnames, _ in os.walk(ROOT):
            if ".git" in Path(dirpath).parts:
                dirnames[:] = []
                continue
            total += sum(1 for d in dirnames if d == name)
        return total

    def count_files(name: str) -> int:
        total = 0
        for _, dirnames, filenames in os.walk(ROOT):
            if ".git" in dirnames:
                dirnames.remove(".git")
            total += sum(1 for f in filenames if f == name)
        return total

    return {
        "pycache_dirs": count_dirs("__pycache__"),
        "ds_store_files": count_files(".DS_Store"),
        "pytest_cache_dirs": count_dirs(".pytest_cache"),
        "egg_info_dirs": sum(1 for _ in (ROOT / "src").glob("*.egg-info")),
    }


def python_doc_stats(root: Path) -> dict[str, int]:
    stats = {
        "files": 0,
        "modules_with_docstring": 0,
        "public_functions": 0,
        "public_functions_with_docstring": 0,
        "classes": 0,
        "classes_with_docstring": 0,
        "parse_errors": 0,
    }
    if not root.exists():
        return stats
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        stats["files"] += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            stats["parse_errors"] += 1
            continue
        if ast.get_docstring(tree):
            stats["modules_with_docstring"] += 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("_") and node.name not in {"__init__", "__call__"}:
                    continue
                stats["public_functions"] += 1
                if ast.get_docstring(node):
                    stats["public_functions_with_docstring"] += 1
            elif isinstance(node, ast.ClassDef):
                stats["classes"] += 1
                if ast.get_docstring(node):
                    stats["classes_with_docstring"] += 1
    return stats


def script_catalog() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    scripts_dir = ROOT / "scripts"
    for path in sorted(scripts_dir.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        first_lines = "\n".join(text.splitlines()[:80])
        if "ArgumentParser" in first_lines:
            interface = "argparse"
        elif path.suffix == ".sh":
            interface = "shell"
        else:
            interface = "manual"

        name = path.name
        if name.startswith(("prepare_", "prefill_", "requeue_", "finalize_", "monitor_")):
            status = "maintenance"
        elif name.startswith(("build_", "write_", "promote_", "match_", "normalize_", "export_")):
            status = "production_candidate"
        elif name.startswith(("benchmark_", "evaluate_", "run_analyze_ab_test", "summarize_")):
            status = "research"
        elif "telegram" in name:
            status = "one_off_or_research"
        elif name in {"git_bootstrap.sh", "autocommit_push_loop.sh", "start_autocommit_push.sh", "stop_autocommit_push.sh"}:
            status = "devops_legacy"
        else:
            status = "needs_review"

        description = ""
        for line in first_lines.splitlines():
            stripped = line.strip()
            if stripped.startswith('"""') and len(stripped) > 3:
                description = stripped.strip('"')
                break
            if "description=" in stripped:
                description = stripped
                break
            if stripped.startswith("#") and not stripped.startswith("#!"):
                description = stripped.lstrip("#").strip()
                break

        rows.append(
            {
                "script": rel(path),
                "status": status,
                "interface": interface,
                "description_hint": description,
            }
        )
    return rows


def db_inventory() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for pattern in ("*.db", "*.sqlite", "*.db-wal", "*.db-shm"):
        for path in ROOT.rglob(pattern):
            if ".git" in path.parts:
                continue
            size = file_size(path)
            name = path.name
            if ".before_" in name or "_before_" in name:
                category = "backup_candidate"
            elif "ra_missing_all_20260506" in rel(path):
                category = "final_ra_source"
            elif "final_processing_coverage_report" in rel(path):
                category = "coverage_artifact"
            elif path.parent.name in {"ab_tests", "benchmarks"} or "ab_tests" in path.parts or "benchmarks" in path.parts:
                category = "research_archive_candidate"
            elif path.parent == ROOT:
                category = "root_runtime_db"
            else:
                category = "runtime_db"
            rows.append(
                {
                    "path": rel(path),
                    "size_bytes": str(size),
                    "size": fmt_size(size),
                    "category": category,
                }
            )
    rows.sort(key=lambda row: int(row["size_bytes"]), reverse=True)
    return rows


def large_files(min_mb: int) -> list[dict[str, str]]:
    threshold = min_mb * 1024 * 1024
    rows: list[dict[str, str]] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        size = file_size(path)
        if size >= threshold:
            rows.append({"path": rel(path), "size_bytes": str(size), "size": fmt_size(size)})
    rows.sort(key=lambda row: int(row["size_bytes"]), reverse=True)
    return rows


def top_level_sizes() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(ROOT.iterdir(), key=lambda p: p.name.lower()):
        if path.name == ".git":
            continue
        size = dir_size(path)
        rows.append({"path": path.name, "size_bytes": str(size), "size": fmt_size(size)})
    rows.sort(key=lambda row: int(row["size_bytes"]), reverse=True)
    return rows


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(out_dir: Path, data: dict[str, object]) -> None:
    md = [
        "# Project Audit Report",
        "",
        f"Generated: {data['generated_at']}",
        "",
        "## Summary",
        "",
        f"- Git changed/untracked lines: {data['git_status_lines']}",
        f"- Tests: {data['tests_status']}",
        f"- Cache counts: `{json.dumps(data['cache_counts'], ensure_ascii=False)}`",
        "",
        "## Python Documentation Stats",
        "",
        "| Area | Files | Module docs | Public funcs | Func docs | Classes | Class docs | Parse errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for area, stats in data["python_doc_stats"].items():  # type: ignore[index]
        md.append(
            "| {area} | {files} | {modules_with_docstring} | {public_functions} | "
            "{public_functions_with_docstring} | {classes} | {classes_with_docstring} | {parse_errors} |".format(
                area=area,
                **stats,
            )
        )
    md.extend(
        [
            "",
            "## Output Files",
            "",
            "- `summary.json`",
            "- `db_inventory.tsv`",
            "- `large_files.tsv`",
            "- `top_level_sizes.tsv`",
            "- `script_catalog.tsv`",
            "- `git_status.txt`",
            "",
            "## Cleanup Rule",
            "",
            "This report is read-only. It intentionally does not delete DBs, audio, transcripts, exports, or runtime folders.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Mango Analyse project audit artifacts.")
    parser.add_argument("--out-dir", default="", help="Output directory. Default: stable_runtime/project_audit_<timestamp>.")
    parser.add_argument("--skip-tests", action="store_true", help="Do not run pytest.")
    parser.add_argument("--large-file-mb", type=int, default=DEFAULT_LARGE_FILE_MB)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "stable_runtime" / f"project_audit_{timestamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    git_status = run(["git", "status", "--short"]).stdout
    (out_dir / "git_status.txt").write_text(git_status, encoding="utf-8")

    tests_status = "skipped"
    tests_output = ""
    if not args.skip_tests:
        test_result = run(["python3", "-m", "pytest", "-q"])
        tests_output = test_result.stdout
        tests_status = "passed" if test_result.returncode == 0 else f"failed_rc_{test_result.returncode}"
    (out_dir / "pytest.txt").write_text(tests_output, encoding="utf-8")

    py_stats = {
        "src/mango_mvp": python_doc_stats(ROOT / "src" / "mango_mvp"),
        "scripts": python_doc_stats(ROOT / "scripts"),
        "tests": python_doc_stats(ROOT / "tests"),
    }

    write_tsv(out_dir / "db_inventory.tsv", db_inventory())
    write_tsv(out_dir / "large_files.tsv", large_files(args.large_file_mb))
    write_tsv(out_dir / "top_level_sizes.tsv", top_level_sizes())
    write_tsv(out_dir / "script_catalog.tsv", script_catalog())

    summary = {
        "generated_at": timestamp,
        "root": str(ROOT),
        "git_status_lines": len([line for line in git_status.splitlines() if line.strip()]),
        "tests_status": tests_status,
        "cache_counts": count_paths(),
        "python_doc_stats": py_stats,
        "runtime_patterns": RUNTIME_PATTERNS,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(out_dir, summary)
    print(out_dir)
    return 0 if tests_status in {"passed", "skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
