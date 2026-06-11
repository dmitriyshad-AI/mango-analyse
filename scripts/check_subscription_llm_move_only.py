#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import subprocess
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "subscription_llm_move_only_v1_2026_06_11"
DEFAULT_SNAPSHOT = Path("D1_audit_backlog/subscription_llm_refactor_body_snapshot_2026-06-11.json")
DEFAULT_REPORT = Path("D1_audit_backlog/subscription_llm_refactor_import_diff_2026-06-11.md")
MODULE_PATH = Path("src/mango_mvp/channels/subscription_llm.py")
PARTS_DIR = Path("src/mango_mvp/channels/subscription_llm_parts")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def current_git_head(root: Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=root, text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else ""


def read_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def top_level_defs(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    return [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


def normalize_ast_node(node: ast.AST) -> str:
    clone = copy.deepcopy(node)
    for item in ast.walk(clone):
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(item, attr):
                setattr(item, attr, None)
    return ast.dump(clone, include_attributes=False, annotate_fields=True)


def code_consts(value: Any) -> Any:
    if isinstance(value, types.CodeType):
        return code_signature(value)
    if isinstance(value, tuple):
        return [code_consts(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(code_consts(item) for item in value)
    return value


def code_signature(code: types.CodeType) -> Mapping[str, Any]:
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code": code.co_code.hex(),
        "consts": [code_consts(item) for item in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def compiled_body_signature(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> Mapping[str, Any]:
    clone = copy.deepcopy(node)
    module = ast.Module(body=[clone], type_ignores=[])
    ast.fix_missing_locations(module)
    compiled = compile(module, filename="<subscription_llm_move_only>", mode="exec")
    candidates = [item for item in compiled.co_consts if isinstance(item, types.CodeType) and item.co_name == node.name]
    if not candidates:
        raise RuntimeError(f"compiled code object not found for {node.name}")
    return code_signature(candidates[0])


def import_section(tree: ast.Module) -> list[str]:
    return [ast.unparse(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]


def build_def_record(path: Path, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, root: Path) -> dict[str, Any]:
    normalized_ast = normalize_ast_node(node)
    compiled = compiled_body_signature(node)
    return {
        "name": node.name,
        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
        "source_path": str(path.relative_to(root)),
        "lineno": node.lineno,
        "end_lineno": getattr(node, "end_lineno", node.lineno),
        "ast_sha256": hashlib.sha256(normalized_ast.encode("utf-8")).hexdigest(),
        "bytecode_sha256": sha256_json(compiled),
        "normalized_ast": normalized_ast,
        "compiled_body": compiled,
    }


def build_snapshot(root: Path) -> dict[str, Any]:
    path = root / MODULE_PATH
    tree = read_tree(path)
    defs = [build_def_record(path, node, root) for node in top_level_defs(tree)]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commit": current_git_head(root),
        "source_path": str(MODULE_PATH),
        "definitions": {item["name"]: item for item in defs},
        "definition_order": [item["name"] for item in defs],
        "imports": import_section(tree),
    }


def iter_parts_py(root: Path) -> Iterable[Path]:
    parts = root / PARTS_DIR
    if not parts.exists():
        return ()
    return sorted(path for path in parts.rglob("*.py") if "__pycache__" not in path.parts)


def collect_current_definitions(root: Path, *, prefer_parts: bool) -> tuple[str, dict[str, Any], list[str]]:
    paths = list(iter_parts_py(root)) if prefer_parts else []
    mode = "parts"
    if not paths:
        paths = [root / MODULE_PATH]
        mode = "monolith"
    definitions: dict[str, Any] = {}
    duplicates: dict[str, list[str]] = {}
    imports: list[str] = []
    for path in paths:
        tree = read_tree(path)
        imports.extend(f"{path.relative_to(root)}: {line}" for line in import_section(tree))
        for node in top_level_defs(tree):
            record = build_def_record(path, node, root)
            if node.name in definitions:
                duplicates.setdefault(node.name, [definitions[node.name]["source_path"]]).append(str(path.relative_to(root)))
            definitions[node.name] = record
    if duplicates:
        details = "; ".join(f"{name}: {paths}" for name, paths in sorted(duplicates.items()))
        raise SystemExit(f"duplicate moved definitions: {details}")
    return mode, definitions, imports


def load_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"body snapshot not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"unsupported body snapshot schema: {payload.get('schema_version')}")
    return payload


def compare(snapshot: Mapping[str, Any], current: Mapping[str, Any], *, mode: str) -> list[str]:
    errors: list[str] = []
    expected = snapshot["definitions"]
    missing = sorted(set(expected) - set(current))
    if missing:
        errors.append(f"missing definitions in {mode}: {missing[:40]}")
    if mode == "parts":
        extra = sorted(set(current) - set(expected))
        if extra:
            errors.append(f"extra definitions in parts: {extra[:40]}")
    for name in sorted(set(expected).intersection(current)):
        before = expected[name]
        after = current[name]
        if before["ast_sha256"] != after["ast_sha256"]:
            errors.append(f"{name}: normalized AST changed")
        if before["bytecode_sha256"] != after["bytecode_sha256"]:
            errors.append(f"{name}: compiled body changed")
    return errors


def write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} already exists; pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_import_report(path: Path, snapshot: Mapping[str, Any], current_imports: list[str], *, mode: str) -> None:
    before = [f"{MODULE_PATH}: {line}" for line in snapshot.get("imports", [])]
    before_set = set(before)
    after_set = set(current_imports)
    added = sorted(after_set - before_set)
    removed = sorted(before_set - after_set)
    lines = [
        "# subscription_llm move-only import diff",
        "",
        f"- mode: `{mode}`",
        f"- before imports: `{len(before)}`",
        f"- after imports: `{len(current_imports)}`",
        "",
        "Allowed differences are import-section only. Function/class bodies are checked by AST and compiled body hash.",
        "",
        "## Added imports",
        "",
        *(f"- `{item}`" for item in added),
        "",
        "## Removed imports",
        "",
        *(f"- `{item}`" for item in removed),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move-only checker for subscription_llm refactor waves.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--write-snapshot", action="store_true")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    snapshot_path = args.snapshot if args.snapshot.is_absolute() else root / args.snapshot
    if args.write_snapshot:
        snapshot = build_snapshot(root)
        write_json(snapshot_path, snapshot, overwrite=args.force)
    else:
        snapshot = load_snapshot(snapshot_path)
    mode, current_defs, current_imports = collect_current_definitions(root, prefer_parts=not args.write_snapshot)
    errors = compare(snapshot, current_defs, mode=mode)
    report_path = args.report if args.report.is_absolute() else root / args.report
    write_import_report(report_path, snapshot, current_imports, mode=mode)
    result = {
        "status": "failed" if errors else "ok",
        "mode": mode,
        "definitions_checked": len(set(snapshot["definitions"]).intersection(current_defs)),
        "snapshot_definitions": len(snapshot["definitions"]),
        "errors": errors,
        "report": str(report_path.relative_to(root) if report_path.is_relative_to(root) else report_path),
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(f"subscription_llm move-only {result['status']}: mode={mode} checked={result['definitions_checked']}")
        for error in errors[:20]:
            print(f"- {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
