from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


GRAPHIFY_COMMIT = "fd470faeee16e9f42e3f47204824a2002a1f899c"
GRAPHIFY_PACKAGE = "graphifyy"
GRAPHIFY_REPO_URL = "https://github.com/safishamsi/graphify.git"
DEFAULT_STAMP_DATE = "2026-06-14"

LLM_ENV_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "MOONSHOT_API_KEY",
    "DEEPSEEK_API_KEY",
    "GRAPHIFY_OPENAI_MODEL",
)

CODE_SUFFIXES = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".java",
    ".c",
    ".h",
    ".cc",
    ".cpp",
    ".hpp",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
}

KB_V65_STRUCTURAL_PATHS = (
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/facts_registry.csv",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/facts_registry.jsonl",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/source_registry.csv",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/source_registry.json",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/knowledge_chunks.csv",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/post_filter_registry.json",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/quality_report.json",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/semantic_review.json",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup/v6_1_build_result.json",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_bot_pack/manifest.json",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_bot_pack/bot_fact_index.json",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_bot_pack/bot_gold_answers.json",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_bot_pack/bot_template_registry.json",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_bot_pack/client_safe_facts_foton.jsonl",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_bot_pack/client_safe_facts_unpk.jsonl",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_bot_pack/manager_only_or_internal_facts.jsonl",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_bot_pack/source_registry.json",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_bot_pack/post_filter_registry.json",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_employee_pack/manifest.json",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_employee_pack/CLIENT_SAFE_FACTS_FOTON.csv",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_employee_pack/CLIENT_SAFE_FACTS_UNPK.csv",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_employee_pack/MANAGER_ONLY_FACTS.csv",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources/release_manifest.yaml",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources/facts/bot_policy.yaml",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources/facts/facts_internal_only.yaml",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources/facts/facts_for_bot_FOTON.yaml",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources/facts/facts_for_bot_UNPK.yaml",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources/facts/brand_rules.yaml",
    "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources/facts/gold_answers_v3.yaml",
)

STRUCTURAL_SOURCE_DATA_PATHS = (
    "src/mango_mvp/channels/rules_registry.yaml",
)

PERFORMANCE_EXCLUDES = (
    ".git/**",
    ".venv/**",
    ".venv-asrbench/**",
    "__pycache__/**",
    ".cache/**",
    ".codex_local/**",
    ".pytest_cache/**",
    "tmp/**",
    "graphify-out/**",
    "stable_runtime/**",
    "audits/**",
    "runs/**",
    "test_runs/**",
    "**/*.db",
    "**/*.sqlite",
    "**/*.sqlite3",
    "**/*.xls",
    "**/*.xlsx",
    "**/*.docx",
    "**/*.zip",
    "**/*.mp3",
    "**/*.wav",
    "**/*.m4a",
    "**/*.mp4",
)

RAW_LABEL_PATTERNS = (
    "Contacts.xls",
    "all_whatsapp_chats.txt",
    "mango_mvp.db",
    "ai_office.db",
    "sales_workbook.xlsx",
    "*DataExport*",
    "*write_off_visits*",
    "transcripts*",
    "telegram_exports*",
    "messages*",
    "product_data/channel_archive",
    "product_data/audio_working_store_20260523_v1",
    "product_data/customer_timeline",
    "product_data/customer_profiles",
    "Финансовая модель/01_исходники",
    "Финансовая модель/03_выгрузки_tallanto",
)

INTERNAL_MARKERS = (
    "internal_only",
    "manager_only_route",
    "forbidden_for_client",
    "MANAGER_ONLY",
)

CURATED_QUERY_HINTS: tuple[Mapping[str, Any], ...] = (
    {
        "terms": ("p0", "возврат", "жалоб", "суд", "спор", "оплат", "payment", "refund", "complaint"),
        "paths": (
            "src/mango_mvp/channels/rules_engine.py",
            "src/mango_mvp/channels/p0_recall_spec.py",
            "src/mango_mvp/channels/subscription_llm_parts/post_layers.py",
            "src/mango_mvp/channels/answer_safety.py",
        ),
        "guidance": "P0/спорные темы не выводить из карты: подтвердить правило в rules_engine.py и p0_recall_spec.py.",
    },
    {
        "terms": ("бренд", "фотон", "унпк", "foton", "unpk", "сравн", "цен", "price", "brand"),
        "paths": (
            "src/mango_mvp/channels/rules_engine.py",
            "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources/facts/brand_rules.yaml",
            "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_employee_pack/CLIENT_SAFE_FACTS_FOTON.csv",
            "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_employee_pack/CLIENT_SAFE_FACTS_UNPK.csv",
            "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_employee_pack/MANAGER_ONLY_FACTS.csv",
        ),
        "guidance": "Бренды/цены не сравнивать по карте; проверять client-safe и manager-only границы в сырье.",
    },
    {
        "terms": ("распис", "групп", "час", "связ", "schedule", "hours"),
        "paths": (
            "src/mango_mvp/channels/rules_engine.py",
            "src/mango_mvp/channels/rules_registry.yaml",
            "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources/facts/facts_for_bot_FOTON.yaml",
            "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources/facts/facts_for_bot_UNPK.yaml",
        ),
        "guidance": "Расписание и часы связи считать фактами из сырья, не из графового вывода.",
    },
    {
        "terms": ("скрипт", "scripts", "script", "опас", "safe", "asr", "resolve", "write", "live"),
        "paths": (
            "docs/SCRIPT_SAFETY_MATRIX.md",
            "AGENTS.md",
            "scripts/graphify_structural_build.py",
            "scripts/graphify_structural_mcp_stdio.py",
        ),
        "guidance": "Live/write/HTTP/ASR/R+A действия не запускать без отдельного подтверждения; Graphify здесь read-only.",
    },
    {
        "terms": ("graphify", "карта", "источник", "истин", "сыр", "head", "http", "write", "update", "mcp"),
        "paths": (
            "AGENTS.md",
            "ARCHITECTURE.md",
            "src/mango_mvp/graphify_structural.py",
            "scripts/graphify_structural_query.py",
            "scripts/graphify_structural_mcp_stdio.py",
        ),
        "guidance": "Карта только навигация; отрицательный вывод при stale-карте проверять через rg/source.",
    },
    {
        "terms": ("cloud", "semantic", "смыслов", "облач", "threat", "gold", "answers", "перескаж"),
        "paths": (
            "src/mango_mvp/graphify_structural.py",
            "ARCHITECTURE.md",
            "D1_audit_backlog/GRAPHIFY_TZ25_pilot_questions_2026-06-14.md",
            "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources/facts/gold_answers_v3.yaml",
        ),
        "guidance": "Cloud semantic layer не запускался; пересказ прозы/gold answers по карте запрещён, читать только сырой источник.",
    },
    {
        "terms": ("client_safe", "manager_only", "internal_only", "forbidden_for_client", "манаг", "внутрен"),
        "paths": (
            "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_employee_pack/CLIENT_SAFE_FACTS_FOTON.csv",
            "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_employee_pack/CLIENT_SAFE_FACTS_UNPK.csv",
            "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_employee_pack/MANAGER_ONLY_FACTS.csv",
            "product_data/knowledge_base/kb_release_20260603_v6_5_summer_format_cleanup_sources/facts/facts_internal_only.yaml",
            "src/mango_mvp/graphify_structural.py",
        ),
        "guidance": "MANAGER_ONLY/internal_only не считать client-safe даже при совпадении похожих фактов.",
    },
)


@dataclass(frozen=True)
class PreparedSource:
    revision: str
    source_dir: Path
    copied_code_files: tuple[str, ...]
    indexed_structured_files: tuple[Mapping[str, Any], ...]
    raw_label_nodes: tuple[Mapping[str, str], ...]
    manifest_path: Path


def run_git(repo_root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def current_revision(repo_root: Path) -> str:
    return run_git(repo_root, ["rev-parse", "HEAD"])


def assert_outside_repo(repo_root: Path, path: Path) -> None:
    repo_resolved = repo_root.resolve()
    path_resolved = path.resolve()
    if path_resolved == repo_resolved or repo_resolved in path_resolved.parents:
        raise ValueError(f"{path_resolved} must be outside git worktree {repo_resolved}")


def extract_git_archive(repo_root: Path, revision: str, archive_dir: Path) -> Path:
    assert_outside_repo(repo_root, archive_dir)
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar") as tmp:
        with tmp.file as fh:
            subprocess.run(["git", "archive", "--format=tar", revision], cwd=repo_root, check=True, stdout=fh)
        with tarfile.open(tmp.name, "r") as tar:
            tar.extractall(archive_dir)
    return archive_dir


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_code_file(path: Path) -> bool:
    return path.suffix in CODE_SUFFIXES


def list_src_code_files(clean_root: Path) -> list[Path]:
    src_root = clean_root / "src"
    if not src_root.exists():
        return []
    return sorted(path for path in src_root.rglob("*") if path.is_file() and is_code_file(path))


def classify_structured_scope(rel_path: str, text: str) -> str:
    lowered_path = rel_path.casefold()
    lowered_text = text.casefold()
    if (
        "manager_only" in lowered_path
        or "facts_internal_only" in lowered_path
        or any(marker.casefold() in lowered_text for marker in INTERNAL_MARKERS)
    ):
        return "manager_only_or_internal"
    if "client_safe" in lowered_path:
        return "client_safe_candidate"
    return "structural_metadata"


def marker_counts(text: str) -> dict[str, int]:
    return {marker: text.count(marker) for marker in INTERNAL_MARKERS}


def count_records(path: Path) -> int:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as fh:
                return max(sum(1 for _ in csv.DictReader(fh)), 0)
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                return len(data)
    except Exception:
        return 0
    return 0


def summarize_structured_file(path: Path, clean_root: Path) -> dict[str, Any]:
    rel_path = rel_posix(path, clean_root)
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "source_path": rel_path,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "records": count_records(path),
        "suffix": path.suffix.lower(),
        "scope": classify_structured_scope(rel_path, text),
        "marker_counts": marker_counts(text),
    }


def list_structured_files(clean_root: Path) -> list[Path]:
    files: list[Path] = []
    for rel in (*KB_V65_STRUCTURAL_PATHS, *STRUCTURAL_SOURCE_DATA_PATHS):
        path = clean_root / rel
        if path.exists() and path.is_file():
            files.append(path)
    return sorted(dict.fromkeys(files))


def sanitize_identifier(value: str) -> str:
    base = re.sub(r"[^0-9A-Za-z_]+", "_", value)
    base = re.sub(r"_+", "_", base).strip("_").lower()
    if not base:
        base = "node"
    if base[0].isdigit():
        base = f"n_{base}"
    return base[:96]


def raw_label_type(path: str) -> str:
    lower = path.casefold()
    if lower.endswith((".xls", ".xlsx")):
        return "spreadsheet_dump_label"
    if lower.endswith((".db", ".sqlite", ".sqlite3")):
        return "database_dump_label"
    if "whatsapp" in lower or "telegram" in lower or "messages" in lower:
        return "message_dump_label"
    if "audio" in lower or lower.endswith((".mp3", ".wav", ".m4a", ".mp4")):
        return "media_dump_label"
    return "raw_data_label"


def find_raw_label_nodes(clean_root: Path) -> list[dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    for pattern in RAW_LABEL_PATTERNS:
        for path in clean_root.glob(pattern):
            rel = rel_posix(path, clean_root)
            labels[rel] = {"source_path": rel, "label_type": raw_label_type(rel), "content_policy": "label_only_no_content"}
    return [labels[key] for key in sorted(labels)]


def generated_structural_index(structured: Sequence[Mapping[str, Any]], raw_labels: Sequence[Mapping[str, str]], revision: str) -> str:
    lines = [
        '"""Generated Graphify structural index for Mango.',
        "",
        "This module is generated from structured local sources only.",
        "It intentionally contains source paths, hashes, counts, and safety scopes,",
        "not cloud-semantic summaries and not raw client dumps.",
        '"""',
        "",
        f"GRAPHIFY_STRUCTURAL_REVISION = {revision!r}",
        f"KB_V65_STRUCTURAL_SOURCE_COUNT = {len(structured)!r}",
        f"RAW_LABEL_NODE_COUNT = {len(raw_labels)!r}",
        "",
    ]
    for item in structured:
        rel = str(item["source_path"])
        ident = sanitize_identifier(rel)
        lines.extend(
            [
                f"def kb_v65_source_{ident}():",
                f"    return {dict(item)!r}",
                "",
            ]
        )
    for item in raw_labels:
        rel = str(item["source_path"])
        ident = sanitize_identifier(rel)
        lines.extend(
            [
                f"def raw_data_label_{ident}():",
                f"    return {dict(item)!r}",
                "",
            ]
        )
    lines.extend(
        [
            "def graphify_structural_scope_rules():",
            "    return {",
            "        'map_role': 'navigation_only',",
            "        'truth_source': 'raw_source_files',",
            "        'semantic_cloud_layer': 'not_run_by_design',",
            "        'manager_only_policy': 'preserve_marker_never_client_safe',",
            "    }",
            "",
        ]
    )
    return "\n".join(lines)


def copy_code_tree(clean_root: Path, source_dir: Path) -> tuple[str, ...]:
    copied: list[str] = []
    for path in list_src_code_files(clean_root):
        rel = rel_posix(path, clean_root)
        target = source_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(rel)
    return tuple(sorted(copied))


def ensure_code_only_source(source_dir: Path) -> None:
    non_code = [
        rel_posix(path, source_dir)
        for path in source_dir.rglob("*")
        if path.is_file() and not is_code_file(path)
    ]
    if non_code:
        raise ValueError(f"Graphify structural source must be code-only; non-code files: {non_code[:20]}")


def prepare_structural_source(
    *,
    repo_root: Path,
    revision: str,
    clean_archive_dir: Path,
    source_dir: Path,
) -> PreparedSource:
    assert_outside_repo(repo_root, clean_archive_dir)
    assert_outside_repo(repo_root, source_dir)
    clean_root = extract_git_archive(repo_root, revision, clean_archive_dir)
    if source_dir.exists():
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    copied = copy_code_tree(clean_root, source_dir)
    structured = tuple(summarize_structured_file(path, clean_root) for path in list_structured_files(clean_root))
    raw_labels = tuple(find_raw_label_nodes(clean_root))
    index_path = source_dir / "src" / "mango_mvp" / "graphify_structural_index.py"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(generated_structural_index(structured, raw_labels, revision), encoding="utf-8")
    ensure_code_only_source(source_dir)
    manifest = {
        "revision": revision,
        "source_dir": str(source_dir),
        "copied_code_files": copied,
        "indexed_structured_files": list(structured),
        "raw_label_nodes": list(raw_labels),
        "performance_excludes": list(PERFORMANCE_EXCLUDES),
        "semantic_cloud_layer": "not_run_by_design",
    }
    manifest_path = source_dir / "graphify_structural_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    # The manifest is kept next to the source for operators, but Graphify must see code only.
    manifest_path_outside_code = source_dir.parent / f"{source_dir.name}_manifest.json"
    manifest_path_outside_code.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest_path.unlink()
    return PreparedSource(
        revision=revision,
        source_dir=source_dir,
        copied_code_files=copied,
        indexed_structured_files=structured,
        raw_label_nodes=raw_labels,
        manifest_path=manifest_path_outside_code,
    )


def graphify_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in LLM_ENV_KEYS:
        env.pop(key, None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_checked(command: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def tree_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        hashes[rel_posix(path, root)] = sha256_file(path)
    return hashes


def graphify_direct_url() -> Mapping[str, Any]:
    tool_python = Path.home() / ".local/share/uv/tools/graphifyy/bin/python"
    if not tool_python.exists():
        return {}
    code = (
        "import importlib.metadata as m, json; "
        "dist=m.distribution('graphifyy'); "
        "txt=dist.read_text('direct_url.json') or '{}'; "
        "print(txt)"
    )
    try:
        out = run_checked([str(tool_python), "-c", code], cwd=Path.cwd()).stdout.strip()
        return json.loads(out) if out else {}
    except Exception:
        return {}


def verify_graphify_pin() -> dict[str, Any]:
    version = run_checked(["graphify", "--version"], cwd=Path.cwd()).stdout.strip()
    direct_url = graphify_direct_url()
    requested = json.dumps(direct_url, sort_keys=True)
    commit_ok = GRAPHIFY_COMMIT in requested
    return {
        "package": GRAPHIFY_PACKAGE,
        "version_output": version,
        "expected_commit": GRAPHIFY_COMMIT,
        "direct_url": direct_url,
        "commit_ok": commit_ok,
    }


def graph_stats(graph_path: Path) -> dict[str, int]:
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    return {
        "nodes": len(data.get("nodes", [])),
        "edges": len(data.get("edges", [])),
        "hyperedges": len(data.get("hyperedges", [])),
        "input_tokens": int(data.get("input_tokens") or 0),
        "output_tokens": int(data.get("output_tokens") or 0),
    }


def write_deterministic_graph_report(
    report_path: Path,
    *,
    prepared: PreparedSource,
    graph_stats_value: Mapping[str, int],
    graphify_pin: Mapping[str, Any],
    stamp_date: str,
) -> None:
    scopes: dict[str, int] = {}
    for item in prepared.indexed_structured_files:
        scope = str(item.get("scope") or "unknown")
        scopes[scope] = scopes.get(scope, 0) + 1
    lines = [
        f"Graphify structural map: commit {prepared.revision} date {stamp_date} source {prepared.source_dir}",
        "# Mango Graphify Structural Report",
        "",
        "## Scope",
        "- Layer: structural only.",
        "- Cloud semantic layer: NOT_RUN_BY_DESIGN.",
        "- Input source: clean git archive, not working tree.",
        "- Graphify input is code-only: `src/**/*.py` plus generated Python index for structured v6.5 files and raw-data labels.",
        "",
        "## Graph",
        f"- Nodes: {graph_stats_value['nodes']}",
        f"- Edges: {graph_stats_value['edges']}",
        f"- Hyperedges: {graph_stats_value['hyperedges']}",
        f"- Token cost: input={graph_stats_value['input_tokens']} output={graph_stats_value['output_tokens']}",
        "",
        "## Sources",
        f"- Copied code files: {len(prepared.copied_code_files)}",
        f"- Indexed structured files: {len(prepared.indexed_structured_files)}",
        f"- Raw data label nodes: {len(prepared.raw_label_nodes)}",
        f"- Structured scopes: {json.dumps(scopes, ensure_ascii=False, sort_keys=True)}",
        "",
        "## Safety",
        "- `graphify-out/` is local and must not be committed or sent outside.",
        "- `MANAGER_ONLY`, `internal_only`, `manager_only_route`, `forbidden_for_client` are preserved as internal/manager-only markers.",
        "- Map answers are navigation hints only; facts, numbers, prompts, P0, brand and guard claims require source-file confirmation.",
        "- Server mode allowed for dialogs: stdio only, read-only graph tools only; HTTP mode is forbidden for this pilot.",
        "",
        "## Package",
        f"- Package: {GRAPHIFY_PACKAGE}",
        f"- Expected commit: {GRAPHIFY_COMMIT}",
        f"- Version output: {graphify_pin.get('version_output', '')}",
        f"- Commit check: {graphify_pin.get('commit_ok')}",
        "",
        "## Key Source Hints",
        "- `src/mango_mvp/channels/rules_engine.py` — P0, brand separation and deterministic rule routing.",
        "- `src/mango_mvp/channels/subscription_llm_parts/post_layers.py` — post-draft guards and output verifiers.",
        "- `src/mango_mvp/graphify_structural_index.py` in the generated source tree — structured v6.5 source labels and scopes.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def normalize_graphify_output(graph_dir: Path) -> None:
    cache_dir = graph_dir / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    labels = graph_dir / ".graphify_labels.json"
    if labels.exists():
        labels.unlink()
    graphify_manifest = graph_dir / "manifest.json"
    if graphify_manifest.exists():
        graphify_manifest.unlink()


def write_output_manifest(
    output_graph_dir: Path,
    *,
    prepared: PreparedSource,
    repo_root: Path,
    graphify_pin: Mapping[str, Any],
    stamp_date: str,
) -> Path:
    manifest = {
        "revision": prepared.revision,
        "stamp_date": stamp_date,
        "repo_root": str(repo_root),
        "source_dir": str(prepared.source_dir),
        "graphify_output_dir": str(output_graph_dir),
        "semantic_cloud_layer": "not_run_by_design",
        "llm_env_removed": list(LLM_ENV_KEYS),
        "build_seconds": "recorded_in_run_summary_not_in_reproducible_graph_artifact",
        "copied_code_file_count": len(prepared.copied_code_files),
        "indexed_structured_file_count": len(prepared.indexed_structured_files),
        "raw_label_node_count": len(prepared.raw_label_nodes),
        "indexed_structured_files": list(prepared.indexed_structured_files),
        "raw_label_nodes": list(prepared.raw_label_nodes),
        "performance_excludes": list(PERFORMANCE_EXCLUDES),
        "graphify_pin": graphify_pin,
    }
    path = output_graph_dir / "mango_structural_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def build_once(
    *,
    repo_root: Path,
    prepared: PreparedSource,
    output_parent: Path,
    stamp_date: str = DEFAULT_STAMP_DATE,
) -> dict[str, Any]:
    import time

    assert_outside_repo(repo_root, output_parent)
    if output_parent.exists():
        shutil.rmtree(output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)
    graphify_pin = verify_graphify_pin()
    started = time.monotonic()
    extract = run_checked(
        ["graphify", "extract", str(prepared.source_dir), "--no-cluster", "--out", str(output_parent)],
        cwd=repo_root,
        env=graphify_env(),
    )
    graph_dir = output_parent / "graphify-out"
    build_seconds = time.monotonic() - started
    graph_stats_value = graph_stats(graph_dir / "graph.json")
    normalize_graphify_output(graph_dir)
    write_deterministic_graph_report(
        graph_dir / "GRAPH_REPORT.md",
        prepared=prepared,
        graph_stats_value=graph_stats_value,
        graphify_pin=graphify_pin,
        stamp_date=stamp_date,
    )
    write_output_manifest(
        graph_dir,
        prepared=prepared,
        repo_root=repo_root,
        graphify_pin=graphify_pin,
        stamp_date=stamp_date,
    )
    return {
        "extract_stdout": extract.stdout,
        "extract_stderr": extract.stderr,
        "cluster_stdout": "not_run_no_cluster_structural_layer",
        "cluster_stderr": "",
        "graph_dir": str(graph_dir),
        "hashes": tree_hashes(graph_dir),
        "build_seconds": build_seconds,
        "graphify_pin": graphify_pin,
    }


def build_reproducible(
    *,
    repo_root: Path,
    revision: str,
    clean_archive_dir: Path,
    source_dir: Path,
    output_parent: Path,
    stamp_date: str = DEFAULT_STAMP_DATE,
) -> dict[str, Any]:
    prepared = prepare_structural_source(
        repo_root=repo_root,
        revision=revision,
        clean_archive_dir=clean_archive_dir,
        source_dir=source_dir,
    )
    first = build_once(repo_root=repo_root, prepared=prepared, output_parent=output_parent, stamp_date=stamp_date)
    first_hashes = dict(first["hashes"])
    second = build_once(repo_root=repo_root, prepared=prepared, output_parent=output_parent, stamp_date=stamp_date)
    second_hashes = dict(second["hashes"])
    return {
        "revision": revision,
        "source_dir": str(source_dir),
        "output_parent": str(output_parent),
        "graph_dir": second["graph_dir"],
        "prepared_manifest": str(prepared.manifest_path),
        "copied_code_file_count": len(prepared.copied_code_files),
        "indexed_structured_file_count": len(prepared.indexed_structured_files),
        "raw_label_node_count": len(prepared.raw_label_nodes),
        "first_hashes": first_hashes,
        "second_hashes": second_hashes,
        "reproducible": first_hashes == second_hashes,
        "first_build_seconds": round(first["build_seconds"], 3),
        "second_build_seconds": round(second["build_seconds"], 3),
        "graphify_pin": second["graphify_pin"],
        "extract_stdout": second["extract_stdout"],
        "extract_stderr": second["extract_stderr"],
        "cluster_stdout": second["cluster_stdout"],
        "cluster_stderr": second["cluster_stderr"],
    }


def load_output_manifest(graph_path: Path) -> Mapping[str, Any]:
    graph_dir = graph_path.parent if graph_path.name == "graph.json" else graph_path
    manifest_path = graph_dir / "mango_structural_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def stale_banner(repo_root: Path, graph_path: Path) -> str:
    manifest = load_output_manifest(graph_path)
    graph_rev = str(manifest.get("revision") or "")
    head = current_revision(repo_root)
    if graph_rev and graph_rev != head:
        return f"ВНИМАНИЕ: карта построена на {graph_rev[:12]}, сейчас {head[:12]}; проверяй выводы в сырье."
    return f"Карта построена на текущей ревизии {head[:12]}; факты всё равно подтверждай в исходнике."


def _query_matches_terms(query: str, terms: Sequence[str]) -> bool:
    folded = query.casefold()
    return any(term.casefold() in folded for term in terms)


def curated_source_hints(query: str) -> list[str]:
    hints: list[str] = []
    seen: set[str] = set()
    for entry in CURATED_QUERY_HINTS:
        if not _query_matches_terms(query, tuple(str(term) for term in entry["terms"])):
            continue
        for path in tuple(str(item) for item in entry["paths"]):
            if path not in seen:
                seen.add(path)
                hints.append(path)
    return hints


def curated_guidance(query: str) -> list[str]:
    guidance: list[str] = []
    seen: set[str] = set()
    for entry in CURATED_QUERY_HINTS:
        if not _query_matches_terms(query, tuple(str(term) for term in entry["terms"])):
            continue
        text = str(entry["guidance"])
        if text not in seen:
            seen.add(text)
            guidance.append(text)
    return guidance


def normalize_source_hint(graph_path: Path, source_file: str) -> str:
    manifest = load_output_manifest(graph_path)
    source_dir_text = str(manifest.get("source_dir") or "")
    if source_dir_text:
        try:
            rel = Path(source_file).resolve().relative_to(Path(source_dir_text).resolve())
            rel_text = rel.as_posix()
            if rel_text == "src/mango_mvp/graphify_structural_index.py":
                return "src/mango_mvp/graphify_structural_index.py (generated; raw structured paths are in mango_structural_manifest.json)"
            return rel_text
        except (OSError, ValueError):
            pass
    return source_file


def graph_source_hints(graph_path: Path, query: str, *, limit: int = 12) -> list[str]:
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    terms = [term.casefold() for term in re.findall(r"[\w_]+", query) if len(term) > 2]
    scored: list[tuple[int, str]] = []
    for node in data.get("nodes", []):
        text = " ".join(str(node.get(key) or "") for key in ("id", "label", "source_file"))
        folded = text.casefold()
        score = sum(1 for term in terms if term in folded)
        source_file = str(node.get("source_file") or "")
        if score and source_file:
            scored.append((score, source_file))
    scored.sort(key=lambda item: (-item[0], item[1]))
    result: list[str] = []
    seen: set[str] = set()
    for _, source in scored:
        hint = normalize_source_hint(graph_path, source)
        if hint in seen:
            continue
        seen.add(hint)
        result.append(hint)
        if len(result) >= limit:
            break
    for hint in curated_source_hints(query):
        if hint not in seen:
            seen.add(hint)
            result.append(hint)
        if len(result) >= limit:
            break
    return result


def cli_build(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Mango Graphify structural layer from a clean git archive.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default=None)
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--stamp-date", default=DEFAULT_STAMP_DATE)
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args(argv)
    repo_root = args.repo.resolve()
    revision = args.revision or current_revision(repo_root)
    work_root = args.work_root or (repo_root.parent / f"{repo_root.name}_graphify_structural")
    assert_outside_repo(repo_root, work_root)
    clean_archive_dir = work_root / "clean_archive"
    source_dir = work_root / "source_code_only"
    output_parent = work_root / "output"
    result = build_reproducible(
        repo_root=repo_root,
        revision=revision,
        clean_archive_dir=clean_archive_dir,
        source_dir=source_dir,
        output_parent=output_parent,
        stamp_date=args.stamp_date,
    )
    if not result["graphify_pin"].get("commit_ok"):
        result["reproducible"] = False
        result["pin_error"] = "graphifyy installed package direct_url does not contain the pinned commit"
    if not result["reproducible"]:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    summary_path = args.summary or (work_root / "structural_build_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cli_query(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only query wrapper for Mango Graphify structural map.")
    parser.add_argument("question")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--budget", default="2000")
    args = parser.parse_args(argv)
    graph_path = args.graph.resolve()
    print(stale_banner(args.repo.resolve(), graph_path))
    completed = run_checked(
        ["graphify", "query", args.question, "--graph", str(graph_path), "--budget", str(args.budget)],
        cwd=args.repo.resolve(),
        env=graphify_env(),
    )
    print(completed.stdout.rstrip())
    hints = graph_source_hints(graph_path, args.question)
    if hints:
        print("\nSource path hints:")
        for hint in hints:
            print(f"- {hint}")
    guidance = curated_guidance(args.question)
    if guidance:
        print("\nGuard notes:")
        for item in guidance:
            print(f"- {item}")
    print("\nПравило: путь выше только подсказывает, где читать; факт подтвердить в исходном файле.")
    return 0
