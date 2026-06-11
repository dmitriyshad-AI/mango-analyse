#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SNAPSHOT = Path("D1_audit_backlog/subscription_llm_refactor_body_snapshot_2026-06-11.json")
DEFAULT_BEFORE_FOCUSED = Path("audits/_inbox/subscription_llm_refactor_wave1_20260611_151847/coverage_focused.json")
DEFAULT_BEFORE_REPLAY = Path("audits/_inbox/subscription_llm_refactor_wave1_20260611_151847/coverage_replay.json")
MONOLITH_PATH = Path("src/mango_mvp/channels/subscription_llm.py")
PARTS_DIR = Path("src/mango_mvp/channels/subscription_llm_parts")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def coverage_lines(payload: Mapping[str, Any]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for path, data in payload.get("files", {}).items():
        executed = data.get("executed_lines") or ()
        result[path] = {int(line) for line in executed}
    return result


def def_start(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    decorators = getattr(node, "decorator_list", ())
    return min([node.lineno, *[item.lineno for item in decorators]])


def current_def_spans(root: Path) -> dict[str, tuple[str, int, int]]:
    paths = sorted((root / PARTS_DIR).rglob("*.py")) if (root / PARTS_DIR).exists() else [root / MONOLITH_PATH]
    result: dict[str, tuple[str, int, int]] = {}
    duplicates: dict[str, list[str]] = {}
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = str(path.relative_to(root))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if node.name in result:
                duplicates.setdefault(node.name, [result[node.name][0]]).append(rel)
            result[node.name] = (rel, def_start(node), int(getattr(node, "end_lineno", node.lineno)))
    if duplicates:
        detail = "; ".join(f"{name}: {paths}" for name, paths in sorted(duplicates.items()))
        raise SystemExit(f"duplicate definitions in current tree: {detail}")
    return result


def old_body_line_map(snapshot: Mapping[str, Any]) -> dict[int, tuple[str, int]]:
    mapping: dict[int, tuple[str, int]] = {}
    for name, record in snapshot.get("definitions", {}).items():
        start = int(record["lineno"])
        end = int(record["end_lineno"])
        for line in range(start, end + 1):
            mapping[line] = (name, line - start)
    return mapping


def check_one(
    *,
    label: str,
    before_path: Path,
    after_path: Path,
    snapshot: Mapping[str, Any],
    current_spans: Mapping[str, tuple[str, int, int]],
) -> dict[str, Any]:
    before = coverage_lines(load_json(before_path))
    after_payload = load_json(after_path)
    after = coverage_lines(after_payload)
    after_executable = {
        path: set(lines).union(int(line) for line in (after_payload.get("files", {}).get(path, {}).get("missing_lines") or ()))
        for path, lines in after.items()
    }
    old_executed = before.get(str(MONOLITH_PATH), set())
    old_map = old_body_line_map(snapshot)
    checked = 0
    skipped_unmoved = 0
    skipped_non_executable = 0
    missing: list[dict[str, Any]] = []
    unmapped: list[int] = []
    for old_line in sorted(old_executed):
        item = old_map.get(old_line)
        if item is None:
            continue
        name, offset = item
        current = current_spans.get(name)
        if current is None:
            unmapped.append(old_line)
            continue
        current_path, current_start, current_end = current
        if current_path.endswith("/monolith.py"):
            skipped_unmoved += 1
            continue
        current_line = current_start + offset
        if current_line > current_end:
            missing.append(
                {
                    "definition": name,
                    "old_line": old_line,
                    "current_path": current_path,
                    "current_line": current_line,
                    "reason": "line_outside_current_span",
                }
            )
            continue
        if current_line not in after_executable.get(current_path, set()):
            skipped_non_executable += 1
            continue
        checked += 1
        if current_line not in after.get(current_path, set()):
            missing.append(
                {
                    "definition": name,
                    "old_line": old_line,
                    "current_path": current_path,
                    "current_line": current_line,
                    "reason": "not_executed_after",
                }
            )
    return {
        "label": label,
        "before": str(before_path),
        "after": str(after_path),
        "checked_body_lines": checked,
        "skipped_unmoved_body_lines": skipped_unmoved,
        "skipped_non_executable_lines": skipped_non_executable,
        "unmapped_old_lines": unmapped,
        "missing": missing,
        "status": "ok" if not missing and not unmapped else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--before-focused", type=Path, default=DEFAULT_BEFORE_FOCUSED)
    parser.add_argument("--before-replay", type=Path, default=DEFAULT_BEFORE_REPLAY)
    parser.add_argument("--after-focused", type=Path, required=True)
    parser.add_argument("--after-replay", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    snapshot = load_json(root / args.snapshot if not args.snapshot.is_absolute() else args.snapshot)
    current_spans = current_def_spans(root)

    def root_path(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    results = [
        check_one(
            label="focused",
            before_path=root_path(args.before_focused),
            after_path=root_path(args.after_focused),
            snapshot=snapshot,
            current_spans=current_spans,
        ),
        check_one(
            label="replay",
            before_path=root_path(args.before_replay),
            after_path=root_path(args.after_replay),
            snapshot=snapshot,
            current_spans=current_spans,
        ),
    ]
    payload = {
        "status": "ok" if all(item["status"] == "ok" for item in results) else "fail",
        "results": results,
    }
    if args.out:
        out = root_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        for item in results:
            print(
                f"{item['label']}: {item['status']} "
                f"checked={item['checked_body_lines']} missing={len(item['missing'])} unmapped={len(item['unmapped_old_lines'])}"
            )
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
