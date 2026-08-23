#!/usr/bin/env python3
"""Find existing Mango implementations before a code task starts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from mango_mvp.graphify_structural import graph_source_hints, load_output_manifest, stale_banner
from scripts.preflight import parse_worktrees_porcelain


DEFAULT_GRAPH = Path("/Users/dmitrijfabarisov/Projects/_mango_graphify_current/output/graphify-out/graph.json")
CODE_ROOTS = ("src", "scripts", "tests", ".claude", ".agents")
OWNER_PREFIXES = ("src/", "scripts/", ".claude/", ".agents/")
AUDIT_FILES = frozenset({"manifest.json", "changed_files.txt", "implementation_notes.md", "risk_review.md", "backward_compatibility.md"})


@dataclass(frozen=True)
class InventoryCandidate:
    classification: str
    source: str
    path: str
    line: int | None
    sha: str | None
    symbol: str
    why_relevant: str
    verified_in_raw_source: bool
    worktree: str | None = None


@dataclass(frozen=True)
class InventoryResult:
    schema_version: str
    feature_id: str
    problem_id: str
    repo_head: str
    branch: str
    worktree: str
    status_fingerprint: str
    graph_revision: str
    graph_matches_head: bool
    generator_version: str
    generator_command_sha256: str
    queries: list[str]
    coverage: dict[str, str]
    candidates: list[InventoryCandidate]
    decision: str
    selected_owner: dict[str, str]
    unresolved: list[str]
    generated_at: str


def _run(root: Path, command: Sequence[str], *, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)


def _git(root: Path, *args: str) -> str:
    result = _run(root, ["git", "-c", "core.quotepath=off", *args])
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr[-300:]}")
    return result.stdout


def _rg_hits(root: Path, term: str, paths: Sequence[str] = CODE_ROOTS, *, ignore_case: bool = False) -> list[tuple[str, int]]:
    existing = [path for path in paths if (root / path).exists()]
    if not existing:
        return []
    command = ["rg", "--json", "--sort", "path", "-n", "-F"]
    result = _run(root, [*command, *(["-i"] if ignore_case else []), term, *existing], timeout=30)
    if result.returncode not in {0, 1}:
        raise RuntimeError(f"rg failed for {term!r}: {result.stderr[-300:]}")
    hits: list[tuple[str, int]] = []
    for raw in result.stdout.splitlines():
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if item.get("type") != "match":
            continue
        data = item["data"]
        pair = (str(data["path"]["text"]), int(data["line_number"]))
        if not any(path == pair[0] for path, _line in hits):
            hits.append(pair)
        if len(hits) > 200:
            raise RuntimeError(f"query too broad: {term!r}")
    return hits


def _dirty_paths(status: str) -> set[str]:
    paths: set[str] = set()
    for line in status.splitlines():
        path = line[3:].strip().strip('"')
        if " -> " in path:
            old, new = path.split(" -> ", 1)
            paths.update({old.strip().strip('"'), new.strip().strip('"')})
        elif path:
            paths.add(path)
    return paths


def _owner_text(text: str, symbol: str) -> bool:
    return any(marker in text.strip() for marker in (f"def {symbol}", f"class {symbol}", f"{symbol} =", f"{symbol}=", f"{symbol}:"))


def _owns_symbol(root: Path, path: str, line: int, symbol: str) -> bool:
    candidate = root / path
    if not candidate.is_file() or candidate.is_symlink():
        return False
    text = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()[line - 1].strip()
    return _owner_text(text, symbol)


def _surface(root: Path) -> tuple[list[Any], str, dict[str, set[str]], dict[str, str]]:
    worktrees_text = _git(root, "worktree", "list", "--porcelain")
    entries = parse_worktrees_porcelain(worktrees_text)
    statuses = {
        entry.path: _git(Path(entry.path), "status", "--porcelain=v1", "--untracked-files=all")
        for entry in entries if Path(entry.path).is_dir()
    }
    return entries, worktrees_text, {path: _dirty_paths(status) for path, status in statuses.items()}, statuses


def _fingerprint(root: Path, head: str, worktrees_text: str, entries: Sequence[Any], dirty: dict[str, set[str]], statuses: dict[str, str], terms: Sequence[str], graph: Path, metadata: Sequence[InventoryCandidate]) -> str:
    refs = _git(root, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads", "refs/remotes", "refs/tags")
    digest = hashlib.sha256((head + "\n" + worktrees_text + refs + json.dumps(list(terms), ensure_ascii=False)).encode())
    manifest = graph.parent / "mango_structural_manifest.json"
    if manifest.is_file() and not manifest.is_symlink():
        digest.update(manifest.read_bytes())
    for entry in sorted(entries, key=lambda item: item.path):
        worktree = Path(entry.path)
        if not worktree.is_dir():
            continue
        digest.update((entry.path + "\n" + statuses.get(entry.path, "")).encode())
        digest.update(_git(worktree, "diff", "--binary", "HEAD", "--", *CODE_ROOTS).encode())
        for path in sorted(dirty.get(entry.path, set())):
            candidate = worktree / path
            if candidate.is_symlink():
                digest.update(path.encode() + b"\0SYMLINK\0" + str(candidate.readlink()).encode())
                continue
            if not candidate.is_file() or not any(path == item or path.startswith(item + "/") for item in CODE_ROOTS):
                continue
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            if any(term in text for term in terms):
                digest.update(path.encode() + b"\0" + text.encode())
    for item in metadata:
        candidate = root / item.path
        if item.path and candidate.is_file() and not candidate.is_symlink():
            digest.update(item.path.encode() + b"\0" + candidate.read_bytes())
    return "sha256:" + digest.hexdigest()


def _history(root: Path, term: str, head: str, donor_shas: Sequence[str]) -> list[InventoryCandidate]:
    out: list[InventoryCandidate] = []
    if donor_shas:
        grep = _run(root, ["git", "grep", "-n", "-F", term, *donor_shas, "--", *CODE_ROOTS])
        if grep.returncode == 0:
            row = next((line for line in grep.stdout.splitlines() if _owner_text(line.split(":", 3)[-1], term)), "")
            if row:
                sha, path, line, _text = row.split(":", 3)
                out.append(InventoryCandidate("DONOR_REF", "git_ref", path, int(line), sha, term, "unmerged ref tree contains exact owner", True))
        elif grep.returncode not in {0, 1}:
            raise RuntimeError(f"git grep donor failed: {grep.stderr[-300:]}")
    latest = _git(root, "log", head, "--max-count=1", "--format=@@%H%x09%s", "--patch", "--unified=0", f"-S{term}", "--", *CODE_ROOTS)
    header = next((line for line in latest.splitlines() if line.startswith("@@")), "")
    patch = [line for line in latest.splitlines() if not line.startswith(("---", "+++"))]
    if header and any(line.startswith("-") and term in line for line in patch) and not any(line.startswith("+") and term in line for line in patch):
        sha, _, subject = header[2:].partition("\t")
        path = next((line.split(" b/", 1)[-1] for line in latest.splitlines() if line.startswith("diff --git ")), "")
        out.append(InventoryCandidate("REMOVED_INTENTIONALLY", "git_history", path, None, sha, term, subject, True))
    return out


def _metadata_hits(root: Path, term: str) -> list[InventoryCandidate]:
    candidates: list[InventoryCandidate] = []
    for path, line in _rg_hits(root, term, ("tasks", "docs/DECISIONS_LOG.md"), ignore_case=True):
        candidates.append(InventoryCandidate("FALSE_MATCH", "task_or_decision", path, line, None, term, "metadata lead; raw code confirmation required", False))
    audits = root / "audits" / "_inbox"
    if audits.exists():
        for path in audits.glob("*/*"):
            if path.name not in AUDIT_FILES or not path.is_file():
                continue
            if path.name == "manifest.json":
                try:
                    if json.loads(path.read_text(encoding="utf-8")).get("schema_version") == "mango_claude_context_pack_v1":
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            for line_no, text in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if term in text:
                    candidates.append(InventoryCandidate("FALSE_MATCH", "audit_metadata", str(path.relative_to(root)), line_no, None, term, "audit lead; raw code confirmation required", False))
                    break
    return candidates


def _decision(candidates: Sequence[InventoryCandidate], change: str, graph_fresh: bool) -> tuple[str, dict[str, str], list[str]]:
    partial = [item for item in candidates if item.classification == "PARTIAL_WORKTREE"]
    active = [item for item in candidates if item.classification in {"ACTIVE_REUSE", "ACTIVE_EXTEND"}]
    donors = [item for item in candidates if item.classification == "DONOR_REF"]
    removed = [item for item in candidates if item.classification == "REMOVED_INTENTIONALLY"]
    if partial:
        return "stop", {}, ["dirty_code_unclassified"]
    conflicts = {item.symbol for item in active if len({candidate.path for candidate in active if candidate.symbol == item.symbol}) > 1}
    if conflicts:
        return "stop", {}, ["active_owner_conflict:" + ",".join(sorted(conflicts))]
    if active:
        owner = active[0]
        decision = "reuse" if change in {"new", "remove"} else "extend"
        return decision, {"path": owner.path, "symbol": owner.symbol, "sha": owner.sha or "WORKTREE"}, []
    if any(item.source == "raw_keyword" for item in candidates):
        return "stop", {}, ["keyword_lead_requires_classification"]
    if removed:
        return "stop", {}, ["removed_intentionally_requires_owner_decision"]
    if donors:
        owner = donors[0]
        return "port", {"path": owner.path, "symbol": owner.symbol, "sha": owner.sha or ""}, []
    if not graph_fresh:
        return "stop", {}, ["graphify_stale_absence_not_proven"]
    return "new", {}, []


def run_inventory(
    root: Path,
    *,
    feature_id: str,
    problem_id: str,
    change: str,
    keywords: Sequence[str],
    symbols: Sequence[str],
    graph: Path = DEFAULT_GRAPH,
) -> InventoryResult:
    root = root.resolve()
    terms = list(dict.fromkeys(term.strip() for term in [*symbols, *keywords] if len(term.strip()) >= 4))
    if not terms:
        raise ValueError("at least one precise keyword or symbol is required")
    head = _git(root, "rev-parse", "HEAD").strip()
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip()
    manifest = load_output_manifest(graph) if graph.exists() else {}
    graph_rev = str(manifest.get("revision") or "")
    graph_fresh = bool(graph_rev and graph_rev == head)
    entries, worktrees_text, dirty, statuses = _surface(root)
    graph_banner = stale_banner(root, graph) if graph.exists() else "graph missing"
    donor_shas = _git(root, "for-each-ref", f"--no-merged={head}", "--format=%(objectname)", "refs/heads", "refs/remotes", "refs/tags").splitlines()
    candidates: list[InventoryCandidate] = []
    for entry in entries:
        worktree = Path(entry.path)
        for path in sorted(dirty.get(entry.path, set())):
            candidate = worktree / path
            if candidate.is_symlink() and any(path == item or path.startswith(item + "/") for item in CODE_ROOTS):
                candidates.append(InventoryCandidate(
                    "PARTIAL_WORKTREE", "dirty_symlink", path, None, getattr(entry, "head", None),
                    "dirty_code_symlink", "unclassified symlink in code surface", False, entry.path,
                ))
    raw_keys: set[tuple[str, str]] = set()
    symbol_terms = set(symbols)
    for term in terms:
        hints = graph_source_hints(graph, term) if graph.exists() else []
        term_hits: list[tuple[Any, Path, str, int, str]] = []
        for entry in entries:
            worktree = Path(entry.path)
            if not worktree.is_dir():
                continue
            worktree_head = _git(worktree, "rev-parse", "HEAD").strip()
            for path, line in _rg_hits(worktree, term, ignore_case=term not in symbol_terms):
                term_hits.append((entry, worktree, path, line, worktree_head))
            for path in sorted(dirty.get(entry.path, set())):
                if not any(path == item or path.startswith(item + "/") for item in CODE_ROOTS):
                    continue
                previous = _run(worktree, ["git", "show", f"HEAD:{path}"])
                if previous.returncode == 0 and term in previous.stdout and not any(hit[2] == path for hit in term_hits):
                    line = next(index for index, text in enumerate(previous.stdout.splitlines(), 1) if term in text)
                    term_hits.append((entry, worktree, path, line, worktree_head))
        current_paths = {path for _entry, worktree, path, _line, _sha in term_hits if worktree == root}
        seen_clean: set[tuple[str, str]] = set()
        for entry, worktree, path, line, worktree_head in term_hits:
            if path in dirty.get(entry.path, set()):
                classification = "PARTIAL_WORKTREE"
                source = "raw_symbol" if term in symbol_terms else "raw_keyword"
            elif term not in symbol_terms or not path.startswith(OWNER_PREFIXES) or not _owns_symbol(worktree, path, line, term):
                classification, source = "FALSE_MATCH", "raw_keyword" if term not in symbol_terms else "raw_reference"
            elif worktree == root:
                classification, source = ("ACTIVE_EXTEND" if change != "new" else "ACTIVE_REUSE"), "raw_symbol"
            elif path in current_paths:
                current, other = root / path, worktree / path
                if current.is_file() and other.is_file() and not current.is_symlink() and not other.is_symlink() and current.read_bytes() == other.read_bytes():
                    continue
                classification, source = "DONOR_REF", "raw_symbol"
            else:
                classification, source = "DONOR_REF", "raw_symbol"
            clean_key = (classification, path)
            if classification not in {"PARTIAL_WORKTREE", "ACTIVE_EXTEND", "ACTIVE_REUSE"} and clean_key in seen_clean:
                continue
            seen_clean.add(clean_key)
            candidates.append(InventoryCandidate(classification, source, path, line, worktree_head, term, "exact raw-source match", True, entry.path))
            raw_keys.add((term, Path(path).name))
        for hint in hints:
            hint_path = hint.split(" ", 1)[0]
            if (term, Path(hint_path).name) not in raw_keys:
                candidates.append(InventoryCandidate("FALSE_MATCH", "graphify", hint, None, graph_rev or None, term, graph_banner, False))
        if term in symbol_terms:
            candidates.extend(_history(root, term, head, donor_shas))
        candidates.extend(_metadata_hits(root, term))
    for term in (feature_id, problem_id):
        candidates.extend(_metadata_hits(root, term))
    unique = {(item.classification, item.source, item.path, item.line, item.sha, item.symbol): item for item in candidates}
    candidates = sorted(unique.values(), key=lambda item: (item.classification, item.source, item.path, item.line or 0, item.sha or "", item.symbol))
    actionable = [item for item in candidates if item.classification not in {"FALSE_MATCH", "IMPLEMENTED_OFF"}]
    if not actionable and graph_fresh:
        candidates.append(InventoryCandidate("ABSENT_PROVEN", "inventory", "", None, head, terms[0], "all permitted surfaces checked", True))
    decision, owner, unresolved = _decision(candidates, change, graph_fresh)
    coverage = {key: "completed" for key in ("graphify", "worktrees", "raw_rg", "git_refs", "tasks", "audits", "decisions")}
    coverage["graphify"] = "completed" if graph.exists() and graph_rev else "missing"
    command_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    generated_at = datetime.now(timezone.utc).isoformat()
    return InventoryResult(
        "mango_prebuild_inventory_v1", feature_id, problem_id, head, branch, str(root),
        _fingerprint(root, head, worktrees_text, entries, dirty, statuses, [feature_id, problem_id, *terms], graph, [item for item in candidates if item.source in {"task_or_decision", "audit_metadata"}]), graph_rev, graph_fresh,
        "mango_inventory_before_build_v2", "sha256:" + command_hash, [feature_id, problem_id, *terms], coverage, candidates, decision, owner, unresolved, generated_at,
    )


def _write_outputs(result: InventoryResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    (out_dir / "prebuild_inventory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"# Prebuild inventory: {result.feature_id}", "", f"- Решение: `{result.decision}`", f"- HEAD: `{result.repo_head}`", f"- Graphify: `{result.graph_revision or 'missing'}`", "", "## Кандидаты"]
    lines.extend(f"- `{item.classification}` {item.source}: `{item.path or item.sha}` ({item.symbol})" for item in result.candidates)
    (out_dir / "prebuild_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inventory existing work before building a feature.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--feature-id", required=True)
    parser.add_argument("--problem-id", required=True)
    parser.add_argument("--change", choices=("new", "extend", "fix", "remove"), required=True)
    parser.add_argument("--keywords", default="")
    parser.add_argument("--symbols", default="")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    split = lambda value: [item.strip() for item in value.split(",") if item.strip()]
    result = run_inventory(args.root, feature_id=args.feature_id, problem_id=args.problem_id, change=args.change, keywords=split(args.keywords), symbols=split(args.symbols), graph=args.graph)
    if args.out_dir:
        _write_outputs(result, args.out_dir)
    if args.json or not args.out_dir:
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 1 if result.decision == "stop" else 0


if __name__ == "__main__":
    raise SystemExit(main())
