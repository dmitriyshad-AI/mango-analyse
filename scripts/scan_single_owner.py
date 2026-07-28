#!/usr/bin/env python3
"""Read-only inventory of function owners, references and structural duplicates."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
def _python_files(root: Path, relative: str) -> tuple[Path, ...]:
    base = root / relative
    return tuple(sorted(base.rglob("*.py"))) if base.exists() else ()
class _LocalNameNormalizer(ast.NodeTransformer):
    def __init__(self, names: Iterable[str]) -> None:
        self._mapping = {name: f"local_{index}" for index, name in enumerate(dict.fromkeys(names))}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self._mapping:
            node.id = self._mapping[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self._mapping:
            node.arg = self._mapping[node.arg]
        return node
def normalized_ast_hash(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    clone = copy.deepcopy(node)
    clone.name = "function"
    local_names = [arg.arg for arg in (*clone.args.posonlyargs, *clone.args.args, *clone.args.kwonlyargs)]
    if clone.args.vararg:
        local_names.append(clone.args.vararg.arg)
    if clone.args.kwarg:
        local_names.append(clone.args.kwarg.arg)
    local_names.extend(
        item.id for item in ast.walk(clone) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
    )
    clone = _LocalNameNormalizer(local_names).visit(clone)
    ast.fix_missing_locations(clone)
    return hashlib.sha256(ast.dump(clone, include_attributes=False).encode("utf-8")).hexdigest()[:16]
def _trees(paths: Iterable[Path]) -> tuple[tuple[Path, ast.Module], ...]:
    return tuple(
        (path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for path in paths
    )
def _dynamic_references(trees: Iterable[tuple[Path, ast.Module]], root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for path, tree in trees:
        for node in ast.walk(tree):
            values: list[ast.AST] = []
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                if any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                    values = list(getattr(node.value, "elts", ()))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "__all__"
                and node.func.attr in {"append", "extend"}
            ):
                values = list(getattr(node.args[0], "elts", node.args[:1])) if node.args else []
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"getattr", "hasattr"}
                and len(node.args) >= 2
            ):
                values = [node.args[1]]
            for value in values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    result[value.value].append(f"{path.relative_to(root)}:{node.lineno}")
    return result
def _call_counts(trees: Iterable[tuple[Path, ast.Module]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, tree in trees:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                counts[node.func.id] += 1
            elif isinstance(node.func, ast.Attribute):
                counts[node.func.attr] += 1
    return counts
def scan(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    source_trees = _trees(_python_files(root, "src/mango_mvp"))
    test_trees = _trees(_python_files(root, "tests"))
    dynamic = _dynamic_references((*source_trees, *test_trees), root)
    source_calls, test_calls = _call_counts(source_trees), _call_counts(test_trees)
    rows: list[dict[str, Any]] = []
    for path, tree in source_trees:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if dynamic.get(node.name):
                status = "dynamic_referenced"
            elif source_calls[node.name]:
                status = "referenced"
            elif test_calls[node.name]:
                status = "referenced_only_by_tests"
            elif node.name.startswith("_"):
                status = "unreferenced"
            else:
                status = "dynamic_or_external_unknown"
            rows.append({
                "name": node.name,
                "file": str(path.relative_to(root)),
                "line": node.lineno,
                "ast_hash": normalized_ast_hash(node),
                "status": status,
                "source_calls": source_calls[node.name],
                "test_calls": test_calls[node.name],
                "dynamic_sites": sorted(dynamic.get(node.name, ())),
            })
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[row["ast_hash"]].append(f'{row["file"]}:{row["line"]}')
    for row in rows:
        row["duplicate_sites"] = sorted(groups[row["ast_hash"]]) if len(groups[row["ast_hash"]]) > 1 else []
    return sorted(rows, key=lambda row: (row["file"], row["line"], row["name"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = scan(args.root)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
    else:
        for row in rows:
            duplicate = f" duplicates={len(row['duplicate_sites'])}" if row["duplicate_sites"] else ""
            print(f"{row['status']:28} {row['file']}:{row['line']} {row['name']}{duplicate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
