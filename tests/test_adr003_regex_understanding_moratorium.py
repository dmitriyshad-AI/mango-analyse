from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


SNAPSHOT_PATH = Path(__file__).resolve().parent / "fixtures/adr003_runtime_channel_regex_snapshot.json"
DIRECT_PATH_PATTERN_SNAPSHOT_PATH = (
    Path(__file__).resolve().parent / "fixtures/adr003_direct_path_text_patterns_snapshot.json"
)
DIRECT_PATH_PATTERN_FILES = (
    "src/mango_mvp/channels/answer_safety_classifier.py",
    "src/mango_mvp/channels/conversation_intent_plan.py",
    "src/mango_mvp/channels/fact_scope_spec.py",
    "src/mango_mvp/channels/p0_recall_spec.py",
    "src/mango_mvp/channels/rules_engine.py",
    "src/mango_mvp/channels/semantic_roles.py",
    "src/mango_mvp/channels/subscription_llm_parts/contracts.py",
    "src/mango_mvp/channels/subscription_llm_parts/direct_path.py",
    "src/mango_mvp/channels/subscription_llm_parts/post_layers.py",
    "src/mango_mvp/channels/subscription_llm_parts/provider.py",
    "src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py",
    "src/mango_mvp/channels/subscription_llm_parts/support.py",
    "src/mango_mvp/channels/subscription_llm_parts/text_hygiene.py",
    "src/mango_mvp/channels/telegram_pilot_context_builder.py",
)
INLINE_RE_FUNCTIONS = frozenset({"search", "match", "fullmatch", "findall", "finditer", "split", "sub", "subn"})
TEXT_TABLE_NAME_PARTS = (
    "ACTION",
    "ALIAS",
    "CUE",
    "FACET",
    "INTENT",
    "KEYWORD",
    "MARKER",
    "PATTERN",
    "PHRASE",
    "SCOPE",
    "TERM",
    "TOKEN",
    "TOPIC",
)
TEXT_LIKE_EXPR_PARTS = (
    "client",
    "draft",
    "lower",
    "message",
    "normalized",
    "query",
    "question",
    "text",
    "utterance",
    "value",
)

CHANNEL_REGEX_BUDGET = {
    "src/mango_mvp/channels/answer_quality_rewriter.py": 4,
    "src/mango_mvp/channels/contracts.py": 1,
    "src/mango_mvp/channels/dialogue_contract_pipeline.py": 27,
    "src/mango_mvp/channels/dialogue_memory.py": 27,
    "src/mango_mvp/channels/fact_claim_audit.py": 1,
    "src/mango_mvp/channels/few_shot_reference.py": 1,
    "src/mango_mvp/channels/humanity_guards.py": 1,
    "src/mango_mvp/channels/humanity_linter.py": 2,
    "src/mango_mvp/channels/manager_handoff_summary.py": 1,
    "src/mango_mvp/channels/p0_recall_spec.py": 11,
    "src/mango_mvp/channels/rules_engine.py": 7,
    "src/mango_mvp/channels/subscription_llm_parts/contracts.py": 11,
    "src/mango_mvp/channels/subscription_llm_parts/direct_path.py": 10,
    "src/mango_mvp/channels/subscription_llm_parts/policy_routing.py": 16,
    "src/mango_mvp/channels/subscription_llm_parts/post_layers.py": 73,
    "src/mango_mvp/channels/subscription_llm_parts/provider.py": 3,
    "src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py": 11,
    "src/mango_mvp/channels/subscription_llm_parts/support.py": 3,
    "src/mango_mvp/channels/subscription_llm_parts/text_hygiene.py": 7,
    "src/mango_mvp/channels/telegram_pilot_reporting.py": 7,
}


def _is_re_compile_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "compile"
        and isinstance(func.value, ast.Name)
        and func.value.id == "re"
    )


def _target_name(parent: ast.AST | None) -> str:
    if isinstance(parent, ast.Assign):
        targets = parent.targets
    elif isinstance(parent, ast.AnnAssign):
        targets = [parent.target]
    else:
        return ""
    names: list[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)
    return ",".join(names)


def _pattern_literal(node: ast.Call) -> str:
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    if node.args:
        return ast.unparse(node.args[0])
    return ""


def _flags_expr(node: ast.Call) -> str:
    flags: list[str] = []
    if len(node.args) >= 2:
        flags.append(ast.unparse(node.args[1]))
    for keyword in node.keywords:
        if keyword.arg == "flags":
            flags.append(ast.unparse(keyword.value))
    return " | ".join(flags)


def _regex_snapshot(repo: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((repo / "src/mango_mvp/channels").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        stack: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.visit_FunctionDef(node)

            def visit_Call(self, node: ast.Call) -> None:
                if _is_re_compile_call(node):
                    pattern = _pattern_literal(node)
                    rows.append(
                        {
                            "path": str(path.relative_to(repo)),
                            "qualname": ".".join(stack) or "<module>",
                            "target": _target_name(parents.get(node)),
                            "pattern_sha256": hashlib.sha256(pattern.encode("utf-8")).hexdigest(),
                            "flags": _flags_expr(node),
                            "pattern_preview": pattern[:160],
                        }
                    )
                self.generic_visit(node)

        Visitor().visit(tree)
    return sorted(rows, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))


def _is_inline_re_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in INLINE_RE_FUNCTIONS
        and isinstance(func.value, ast.Name)
        and func.value.id == "re"
    )


def _is_text_table_name(name: str) -> bool:
    return name.isupper() and any(part in name for part in TEXT_TABLE_NAME_PARTS)


def _value_signature(node: ast.AST) -> tuple[str, str]:
    rendered = ast.unparse(node)
    normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest(), rendered[:240]


def _direct_path_text_pattern_snapshot(repo: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rel_path in DIRECT_PATH_PATTERN_FILES:
        path = repo / rel_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        stack: list[str] = []

        class Visitor(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self.visit_FunctionDef(node)

            def visit_Assign(self, node: ast.Assign) -> None:
                for target in node.targets:
                    if isinstance(target, ast.Name) and _is_text_table_name(target.id):
                        signature, preview = _value_signature(node.value)
                        rows.append(
                            {
                                "path": rel_path,
                                "qualname": ".".join(stack) or "<module>",
                                "node_kind": "text_table",
                                "symbol": target.id,
                                "value_sha256": signature,
                                "value_preview": preview,
                            }
                        )
                self.generic_visit(node)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                if isinstance(node.target, ast.Name) and node.value is not None and _is_text_table_name(node.target.id):
                    signature, preview = _value_signature(node.value)
                    rows.append(
                        {
                            "path": rel_path,
                            "qualname": ".".join(stack) or "<module>",
                            "node_kind": "text_table",
                            "symbol": node.target.id,
                            "value_sha256": signature,
                            "value_preview": preview,
                        }
                    )
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                if _is_re_compile_call(node) or _is_inline_re_call(node):
                    func = node.func
                    assert isinstance(func, ast.Attribute)
                    pattern = _pattern_literal(node)
                    rows.append(
                        {
                            "path": rel_path,
                            "qualname": ".".join(stack) or "<module>",
                            "node_kind": "regex_call",
                            "symbol": f"re.{func.attr}",
                            "target": _target_name(parents.get(node)),
                            "pattern_sha256": hashlib.sha256(pattern.encode("utf-8")).hexdigest(),
                            "flags": _flags_expr(node),
                            "pattern_preview": pattern[:160],
                        }
                    )
                self.generic_visit(node)

            def visit_Compare(self, node: ast.Compare) -> None:
                if (
                    isinstance(node.left, ast.Constant)
                    and isinstance(node.left.value, str)
                    and any(isinstance(operator, (ast.In, ast.NotIn)) for operator in node.ops)
                ):
                    comparators = " ".join(ast.unparse(comparator) for comparator in node.comparators)
                    if any(part in comparators.casefold() for part in TEXT_LIKE_EXPR_PARTS):
                        expression = ast.unparse(node)
                        operator = " ".join(type(item).__name__ for item in node.ops)
                        rows.append(
                            {
                                "path": rel_path,
                                "qualname": ".".join(stack) or "<module>",
                                "node_kind": "string_contains",
                                "symbol": operator,
                                "expression_sha256": hashlib.sha256(expression.encode("utf-8")).hexdigest(),
                                "expression_preview": expression[:240],
                            }
                        )
                self.generic_visit(node)

        Visitor().visit(tree)
    return sorted(rows, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))


def test_adr003_no_new_runtime_channel_regex_without_review() -> None:
    repo = Path(__file__).resolve().parents[1]
    channel_files = sorted((repo / "src/mango_mvp/channels").rglob("*.py"))
    files_with_regex = {
        str(path.relative_to(repo)): path.read_text(encoding="utf-8").count("re.compile(")
        for path in channel_files
        if "re.compile(" in path.read_text(encoding="utf-8")
    }

    unexpected = sorted(set(files_with_regex) - set(CHANNEL_REGEX_BUDGET))
    assert unexpected == [], (
        "ADR-003 forbids adding new runtime channel regex/keyword understanding. "
        "If this is an output scrub or infrastructure parser, document it in "
        "docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md and update this allowlist."
    )
    for rel_path, count in files_with_regex.items():
        assert count <= CHANNEL_REGEX_BUDGET[rel_path], (
            f"{rel_path} added {count - CHANNEL_REGEX_BUDGET[rel_path]} runtime regex. "
            "New client-meaning detection must go through SemanticFrame/eval, not regex."
        )


def test_adr003_runtime_channel_regex_snapshot_is_frozen() -> None:
    repo = Path(__file__).resolve().parents[1]
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    actual = _regex_snapshot(repo)

    assert actual == expected, (
        "ADR-003 freezes runtime channel regex while understanding migrates to SemanticFrame. "
        "If a regex change is truly an output scrub/fail-closed/infrastructure parser, document the reason "
        "in docs/ADR003_REGEX_UNDERSTANDING_MORATORIUM.md, add or update an eval case, and refresh "
        "tests/fixtures/adr003_runtime_channel_regex_snapshot.json intentionally."
    )


def test_adr003_direct_path_text_patterns_snapshot_is_frozen() -> None:
    repo = Path(__file__).resolve().parents[1]
    expected = json.loads(DIRECT_PATH_PATTERN_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    actual = _direct_path_text_pattern_snapshot(repo)

    assert actual == expected, (
        "ADR-003 freezes direct-path text understanding patterns while SemanticFrame becomes the source of meaning. "
        "This guard covers re.compile, inline re.search/sub/... and uppercase keyword/marker tables in direct-path "
        "runtime files. New client-meaning rules must be eval cases + SemanticFrame calibration, not regex/keywords. "
        "If the change is an output scrub/fail-closed/infrastructure parser, document the reason and refresh "
        "tests/fixtures/adr003_direct_path_text_patterns_snapshot.json intentionally."
    )
