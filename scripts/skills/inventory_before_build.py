#!/usr/bin/env python3
"""Check whether a requested feature already exists before building it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mango_mvp.maintenance.project_inventory import ProjectInventoryConfig, build_project_inventory


DEFAULT_GRAPH = Path("/Users/dmitrijfabarisov/Projects/_mango_graphify_current/output/graphify-out/graph.json")


@dataclass(frozen=True)
class InventoryCandidate:
    source: str
    query: str
    status: str
    evidence: str


@dataclass(frozen=True)
class InventoryResult:
    status: str
    candidates: list[InventoryCandidate]
    inventory_summary: dict[str, Any]


def _run(root: Path, command: Sequence[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


def _git_log_pickaxe(root: Path, needle: str) -> list[InventoryCandidate]:
    completed = _run(root, ["git", "log", "--all", "--oneline", "--max-count=8", f"-S{needle}"])
    candidates: list[InventoryCandidate] = []
    for line in completed.stdout.splitlines():
        if line.strip():
            candidates.append(InventoryCandidate("git_log_S", needle, "history_hit", line.strip()))
    return candidates


def _graphify_query(root: Path, question: str, graph: Path) -> list[InventoryCandidate]:
    if not graph.exists():
        return [InventoryCandidate("graphify", question, "skipped", f"graph not found: {graph}")]
    completed = _run(
        root,
        [
            "python3",
            "scripts/graphify_structural_query.py",
            "--graph",
            str(graph),
            "--repo",
            str(root),
            "--budget",
            "2200",
            question,
        ],
        timeout=90,
    )
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        return [InventoryCandidate("graphify", question, "error", output[-600:])]
    lines = [line.strip() for line in output.splitlines() if "src=" in line or line.startswith("- ")]
    evidence = "\n".join(lines[:12]) or output[:1000]
    return [InventoryCandidate("graphify", question, "candidate" if evidence else "empty", evidence)]


def run_inventory(root: Path, *, keywords: Sequence[str], symbols: Sequence[str], graph: Path = DEFAULT_GRAPH) -> InventoryResult:
    candidates: list[InventoryCandidate] = []
    all_terms = [item for item in [*keywords, *symbols] if str(item).strip()]
    for term in all_terms:
        candidates.extend(_git_log_pickaxe(root, term))
    if all_terms:
        candidates.extend(_graphify_query(root, " ".join(all_terms), graph))

    with tempfile.TemporaryDirectory(prefix="mango_inventory_before_build_") as tmp:
        summary = build_project_inventory(ProjectInventoryConfig(project_root=root, out_root=Path(tmp)))
    if summary:
        candidates.append(
            InventoryCandidate(
                "build_project_inventory",
                "project",
                "summary",
                f"db_files={summary.get('db_files')} archive_candidate_rows={summary.get('archive_candidate_rows')}",
            )
        )
    has_existing = any(item.status in {"history_hit", "candidate"} for item in candidates)
    return InventoryResult(status="FOUND" if has_existing else "CLEAR", candidates=candidates, inventory_summary=summary)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory existing work before building a feature.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--keywords", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    result = run_inventory(args.root.resolve(), keywords=_split_csv(args.keywords), symbols=_split_csv(args.symbols), graph=args.graph)
    if args.json:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(f"{result.status}: existing-work inventory")
        for item in result.candidates:
            print(f"- {item.source} [{item.status}] {item.query}: {item.evidence.splitlines()[0] if item.evidence else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
