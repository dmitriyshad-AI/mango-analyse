#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "subscription_llm_facade_exports_v1_2026_06_11"
MODULE_NAME = "mango_mvp.channels.subscription_llm"
PARTS_MODULE_NAME = "mango_mvp.channels.subscription_llm_parts"
DEFAULT_SNAPSHOT = Path("D1_audit_backlog/subscription_llm_refactor_exports_snapshot_2026-06-11.json")
DEFAULT_AST_MAP = Path("D1_audit_backlog/subscription_llm_refactor_ast_current_2026-06-11.json")
DEFAULT_AST_REPORT = Path("D1_audit_backlog/subscription_llm_refactor_ast_diff_2026-06-11.md")

KEY_EXPORTS = (
    "_CodexRetryableError",
    "_PromptProviderError",
    "SubscriptionDraftResult",
    "SubscriptionLlmDraftProvider",
    "DraftGenerationResult",
    "CodexExecDraftProvider",
    "FakeDraftProvider",
)

PHASE0_PLAN_METRICS = {
    "lines": 13463,
    "ast_top_level_items": 793,
    "defs": 474,
    "classes": 8,
    "assigns": 278,
    "uppercase_constants": 273,
    "env_constants": 49,
    "regex_RE": 82,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def monolith_path(root: Path) -> Path:
    return root / "src" / "mango_mvp" / "channels" / "subscription_llm.py"


def load_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def target_names(target: ast.AST) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            yield from target_names(item)


def literal_text_like(node: ast.AST | None) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and bool(node.value.strip())
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(literal_text_like(item) for item in node.elts)
    return False


def assignment_kind(name: str, node: ast.Assign | ast.AnnAssign) -> str:
    value = getattr(node, "value", None)
    if name.isupper():
        return "constant"
    if literal_text_like(value):
        return "text_assignment"
    return "assignment"


def imported_names(tree: ast.Module) -> set[str]:
    result: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                result.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                result.add(alias.asname or alias.name)
    return result


def local_symbols(tree: ast.Module) -> list[dict[str, Any]]:
    symbols: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            symbols.append(
                {
                    "name": node.name,
                    "kind": kind,
                    "lineno": node.lineno,
                    "end_lineno": getattr(node, "end_lineno", node.lineno),
                }
            )
            seen.add(node.name)
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in target_names(target):
                    if name in seen:
                        continue
                    symbols.append(
                        {
                            "name": name,
                            "kind": assignment_kind(name, node),
                            "lineno": node.lineno,
                            "end_lineno": getattr(node, "end_lineno", node.lineno),
                        }
                    )
                    seen.add(name)
    return symbols


def externally_imported_names(root: Path) -> set[str]:
    result: set[str] = set()
    search_roots = (root / "src", root / "scripts", root / "tests")
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*.py"):
            if path == monolith_path(root):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == MODULE_NAME:
                    for alias in node.names:
                        if alias.name != "*":
                            result.add(alias.name)
    return result


def current_git_head(root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def metrics(path: Path, tree: ast.Module, symbols: list[Mapping[str, Any]]) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    assign_nodes = [node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))]
    names = {str(item["name"]) for item in symbols}
    return {
        "lines": text.count("\n") + (0 if text.endswith("\n") else 1),
        "ast_top_level_items": len(tree.body),
        "defs": sum(1 for item in symbols if item["kind"] == "function"),
        "classes": sum(1 for item in symbols if item["kind"] == "class"),
        "assigns": len(assign_nodes),
        "local_names": len(names),
        "uppercase_constants": sum(1 for name in names if name.isupper()),
        "env_constants": sum(1 for name in names if name.endswith("_ENV")),
        "regex_RE": sum(1 for name in names if name.endswith("_RE")),
    }


def build_snapshot(root: Path) -> dict[str, Any]:
    path = monolith_path(root)
    tree = load_tree(path)
    symbols = local_symbols(tree)
    local_names = sorted(str(item["name"]) for item in symbols)
    external_names = externally_imported_names(root)
    imported_compat_names = sorted(name for name in external_names if name not in set(local_names))
    export_names = sorted(set(local_names).union(imported_compat_names))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_commit": current_git_head(root),
        "module": MODULE_NAME,
        "source_path": str(path.relative_to(root)),
        "local_names": local_names,
        "compat_imported_names": imported_compat_names,
        "external_from_import_names": sorted(external_names),
        "export_names": export_names,
        "metrics": metrics(path, tree, symbols),
        "symbols": symbols,
    }


def write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} already exists; pass --force to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"snapshot not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"unsupported snapshot schema: {payload.get('schema_version')}")
    return payload


def parts_package_exists(root: Path) -> bool:
    return (root / "src" / "mango_mvp" / "channels" / "subscription_llm_parts").exists()


def assert_facade_is_thin(root: Path) -> None:
    tree = load_tree(monolith_path(root))
    bad_defs = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    if bad_defs:
        raise SystemExit(f"facade contains definitions after parts split: {bad_defs[:20]}")
    bad_assigns: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in target_names(target):
                    if name != "__all__":
                        bad_assigns.append(name)
    if bad_assigns:
        raise SystemExit(f"facade contains runtime assignments after parts split: {bad_assigns[:20]}")


def check_exports(root: Path, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    facade = importlib.import_module(MODULE_NAME)
    export_names = list(snapshot["export_names"])
    missing = [name for name in export_names if not hasattr(facade, name)]
    if missing:
        raise SystemExit(f"facade misses exports: {missing[:50]}")

    parts_mode = parts_package_exists(root)
    if not parts_mode:
        key_missing = [name for name in KEY_EXPORTS if not hasattr(facade, name)]
        if key_missing:
            raise SystemExit(f"monolith misses key exports: {key_missing}")
        return {
            "mode": "monolith",
            "export_count": len(export_names),
            "identity_checked": 0,
            "identity_skipped": len(export_names),
            "status": "ok",
        }

    assert_facade_is_thin(root)
    parts = importlib.import_module(PARTS_MODULE_NAME)
    if not hasattr(facade, "__all__"):
        raise SystemExit("facade __all__ is required after parts split")
    if set(facade.__all__) != set(export_names):
        missing_all = sorted(set(export_names) - set(facade.__all__))
        extra_all = sorted(set(facade.__all__) - set(export_names))
        raise SystemExit(f"facade __all__ mismatch missing={missing_all[:30]} extra={extra_all[:30]}")
    if not hasattr(parts, "__all__") or set(parts.__all__) != set(export_names):
        raise SystemExit("parts __all__ must equal frozen export snapshot")

    identity_mismatches = []
    for name in export_names:
        if not hasattr(parts, name):
            identity_mismatches.append(f"{name}: missing in parts")
            continue
        if getattr(facade, name) is not getattr(parts, name):
            identity_mismatches.append(name)
    if identity_mismatches:
        raise SystemExit(f"facade/parts identity mismatch: {identity_mismatches[:50]}")
    return {
        "mode": "parts",
        "export_count": len(export_names),
        "identity_checked": len(export_names),
        "identity_skipped": 0,
        "status": "ok",
    }


def ast_report(snapshot: Mapping[str, Any]) -> str:
    current = dict(snapshot["metrics"])
    rows = []
    for key, phase0 in PHASE0_PLAN_METRICS.items():
        now = int(current.get(key, 0))
        rows.append(f"| {key} | {phase0} | {now} | {now - phase0:+d} |")
    return "\n".join(
        [
            "# subscription_llm AST refresh before baseline freeze",
            "",
            f"- source commit: `{snapshot.get('source_commit')}`",
            f"- local names: `{len(snapshot.get('local_names', []))}`",
            f"- compat imported names: `{len(snapshot.get('compat_imported_names', []))}`",
            f"- export names frozen for facade: `{len(snapshot.get('export_names', []))}`",
            "",
            "| metric | phase0 plan section 3 | current AST | drift |",
            "|---|---:|---:|---:|",
            *rows,
            "",
            "Notes:",
            "- phase0 section 3 is kept as historical map;",
            "- future facade checks must use the frozen export snapshot, not current AST;",
            "- compat imported names are only aliases imported from subscription_llm by src/scripts/tests.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check subscription_llm facade exports against a frozen AST snapshot.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--write-snapshot", action="store_true")
    parser.add_argument("--write-ast-map", type=Path, default=None)
    parser.add_argument("--write-report", type=Path, default=None)
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

    ast_map_path = args.write_ast_map or (DEFAULT_AST_MAP if args.write_snapshot else None)
    if ast_map_path is not None:
        ast_map_path = ast_map_path if ast_map_path.is_absolute() else root / ast_map_path
        write_json(ast_map_path, snapshot, overwrite=args.force)
    report_path = args.write_report or (DEFAULT_AST_REPORT if args.write_snapshot else None)
    if report_path is not None:
        report_path = report_path if report_path.is_absolute() else root / report_path
        if report_path.exists() and not args.force:
            raise SystemExit(f"{report_path} already exists; pass --force to overwrite")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(ast_report(snapshot), encoding="utf-8")

    result = check_exports(root, snapshot)
    result["snapshot"] = str(snapshot_path.relative_to(root) if snapshot_path.is_relative_to(root) else snapshot_path)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(f"subscription_llm facade exports OK: mode={result['mode']} exports={result['export_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
