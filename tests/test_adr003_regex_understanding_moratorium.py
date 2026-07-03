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
    "src/mango_mvp/channels/actions.py",
    "src/mango_mvp/channels/answer_quality_rewriter.py",
    "src/mango_mvp/channels/conversation_intent_plan.py",
    "src/mango_mvp/channels/dialogue_memory.py",
    "src/mango_mvp/channels/fact_scope_spec.py",
    "src/mango_mvp/channels/held_state.py",
    "src/mango_mvp/channels/new_lead_funnel.py",
    "src/mango_mvp/channels/p0_recall_spec.py",
    "src/mango_mvp/channels/rules_engine.py",
    "src/mango_mvp/channels/semantic_roles.py",
    "src/mango_mvp/channels/text_signals.py",
    "src/mango_mvp/channels/subscription_llm_parts/contracts.py",
    "src/mango_mvp/channels/subscription_llm_parts/direct_path.py",
    "src/mango_mvp/channels/subscription_llm_parts/policy_routing.py",
    "src/mango_mvp/channels/subscription_llm_parts/post_layers.py",
    "src/mango_mvp/channels/subscription_llm_parts/provider.py",
    "src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py",
    "src/mango_mvp/channels/subscription_llm_parts/support.py",
    "src/mango_mvp/channels/subscription_llm_parts/text_hygiene.py",
    "src/mango_mvp/channels/telegram_pilot_context_builder.py",
)
INLINE_RE_FUNCTIONS = frozenset({"search", "match", "fullmatch", "findall", "finditer", "split", "sub", "subn"})
MARKER_HELPER_NAMES = frozenset({"has_marker", "has_any_marker", "has_word_marker", "has_exact_word"})
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
    "src/mango_mvp/channels/subscription_llm_parts/post_layers.py": 64,
    "src/mango_mvp/channels/subscription_llm_parts/provider.py": 3,
    "src/mango_mvp/channels/subscription_llm_parts/reliable_answerer.py": 11,
    "src/mango_mvp/channels/subscription_llm_parts/support.py": 3,
    "src/mango_mvp/channels/subscription_llm_parts/text_hygiene.py": 7,
    "src/mango_mvp/channels/telegram_pilot_reporting.py": 7,
}

CHANNEL_MARKER_HELPER_BUDGET: dict[str, int] = {
    "src/mango_mvp/channels/actions.py": 4,
    "src/mango_mvp/channels/answer_quality_rewriter.py": 79,
    "src/mango_mvp/channels/conversation_intent_plan.py": 37,
    "src/mango_mvp/channels/dialogue_memory.py": 23,
    "src/mango_mvp/channels/fact_scope_spec.py": 8,
    "src/mango_mvp/channels/held_state.py": 2,
    "src/mango_mvp/channels/new_lead_funnel.py": 31,
    "src/mango_mvp/channels/semantic_roles.py": 42,
    "src/mango_mvp/channels/subscription_llm_parts/policy_routing.py": 62,
    "src/mango_mvp/channels/subscription_llm_parts/post_layers.py": 38,
    "src/mango_mvp/channels/text_signals.py": 1,
}

ACTIVE_BEHAVIOR_ALLOWED_FALSE_BUDGET = {
    "src/mango_mvp/channels/subscription_llm_parts/provider.py": 1,
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


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _marker_helper_name(node: ast.Call) -> str:
    name = _call_name(node)
    normalized = name.lstrip("_")
    if normalized in MARKER_HELPER_NAMES:
        return name
    return ""


def _is_literal_marker_value(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.elts)
    return False


def _has_literal_marker_args(node: ast.Call) -> bool:
    return bool(node.args[1:]) and all(_is_literal_marker_value(item) for item in node.args[1:])


def _marker_args_signature(node: ast.Call) -> tuple[str, str]:
    marker_args = ast.Tuple(elts=list(node.args[1:]), ctx=ast.Load())
    normalized = ast.dump(marker_args, annotate_fields=True, include_attributes=False)
    rendered = ", ".join(ast.unparse(item) for item in node.args[1:])
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest(), rendered[:240]


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
                marker_helper = _marker_helper_name(node)
                if marker_helper and _has_literal_marker_args(node):
                    signature, preview = _marker_args_signature(node)
                    rows.append(
                        {
                            "path": rel_path,
                            "qualname": ".".join(stack) or "<module>",
                            "node_kind": "marker_helper_call",
                            "symbol": marker_helper,
                            "args_sha256": signature,
                            "args_preview": preview,
                        }
                    )
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


def _marker_helper_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("node_kind") == "marker_helper_call":
            counts[row["path"]] = counts.get(row["path"], 0) + 1
    return counts


def _channel_marker_helper_call_counts(repo: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted((repo / "src/mango_mvp/channels").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        class Visitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                if _marker_helper_name(node):
                    rel_path = str(path.relative_to(repo))
                    counts[rel_path] = counts.get(rel_path, 0) + 1
                self.generic_visit(node)

        Visitor().visit(tree)
    return counts


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


def test_adr003_direct_path_marker_helper_budget_is_frozen() -> None:
    repo = Path(__file__).resolve().parents[1]
    actual = _channel_marker_helper_call_counts(repo)

    unexpected = sorted(set(actual) - set(CHANNEL_MARKER_HELPER_BUDGET))
    assert unexpected == [], (
        "ADR-003 marker-helper guard found has_marker/has_any_marker calls in new files. "
        "New client-meaning marker tables must move to SemanticFrame/eval, not helper keywords."
    )
    for rel_path, count in actual.items():
        assert count <= CHANNEL_MARKER_HELPER_BUDGET[rel_path], (
            f"{rel_path} added {count - CHANNEL_MARKER_HELPER_BUDGET[rel_path]} marker-helper calls. "
            "New client-meaning detection must go through SemanticFrame/eval, not marker helpers."
        )


def test_adr003_no_active_behavior_allowed_false_sentinel_in_runtime_code() -> None:
    repo = Path(__file__).resolve().parents[1]
    counts: dict[str, int] = {}
    for path in sorted((repo / "src/mango_mvp/channels").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        count = text.count('"active_behavior_allowed": False') + text.count("'active_behavior_allowed': False")
        if count:
            counts[str(path.relative_to(repo))] = count

    unexpected = sorted(set(counts) - set(ACTIVE_BEHAVIOR_ALLOWED_FALSE_BUDGET))
    assert unexpected == [], (
        "ADR-003 forbids new report-only shadow loops that hard-code active_behavior_allowed=False in runtime code. "
        "Decide activate/remove explicitly instead of adding another inert shadow trace."
    )
    for rel_path, count in counts.items():
        assert count <= ACTIVE_BEHAVIOR_ALLOWED_FALSE_BUDGET[rel_path], (
            f"{rel_path} added {count - ACTIVE_BEHAVIOR_ALLOWED_FALSE_BUDGET[rel_path]} "
            "active_behavior_allowed=False sentinels. New shadow traces require explicit ADR decision."
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
