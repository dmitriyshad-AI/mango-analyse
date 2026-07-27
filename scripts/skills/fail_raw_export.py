#!/usr/bin/env python3
"""Export raw evidence for failed bot runs without adding new judgments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts.make_audit_pack import mask_pii


TRANSCRIPTS = "dynamic_dialog_transcripts.jsonl"
JUDGE_RESULTS = "dynamic_judge_results.jsonl"
SUMMARY = "dynamic_summary.json"


@dataclass(frozen=True)
class FailEvidence:
    source_run: str
    dialog_id: str
    verdict: str
    route: str
    client_text: str
    bot_text: str
    rationale: str
    fact_audit: Any
    number_audit: Any
    context_items: Any
    first_failing_turn: Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
        if isinstance(value, Mapping):
            rows.append(dict(value))
    return rows


def _dialog_id(row: Mapping[str, Any]) -> str:
    for key in ("dialog_id", "scenario_id", "id", "case_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _judge_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = row.get("judge_result")
    return payload if isinstance(payload, Mapping) else row


def _verdict(row: Mapping[str, Any]) -> str:
    payload = _judge_payload(row)
    return str(payload.get("verdict") or payload.get("status") or "").strip().upper()


def _last_turn(row: Mapping[str, Any]) -> Mapping[str, Any]:
    turns = row.get("turns") or row.get("messages") or row.get("dialogue")
    if isinstance(turns, list) and turns:
        last = turns[-1]
        return last if isinstance(last, Mapping) else {}
    return row


def _failing_turn(row: Mapping[str, Any], judge: Mapping[str, Any]) -> Mapping[str, Any]:
    expected = judge.get("first_failing_turn")
    turns = row.get("turns")
    if expected is not None and isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, Mapping) and str(turn.get("turn")) == str(expected):
                return turn
    return _last_turn(row)


def _text_from(row: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value:
            return mask_pii(str(value))
    return ""


def _candidate_run_dirs(run_dir: Path) -> tuple[Path, ...]:
    if (run_dir / TRANSCRIPTS).is_file() and (run_dir / JUDGE_RESULTS).is_file():
        return (run_dir,)
    parents = [run_dir, *(path for path in run_dir.rglob("*_full") if path.is_dir())]
    pairs = [(parent / "B", parent / "ON") for parent in parents if all(
        (parent / leg / TRANSCRIPTS).is_file() and (parent / leg / JUDGE_RESULTS).is_file() for leg in ("B", "ON")
    )]
    if len(pairs) != 1:
        raise FileNotFoundError(f"expected exactly one complete B/ON full run under {run_dir}; found {len(pairs)}")
    return pairs[0]


def collect_fail_evidence(run_dir: Path, *, source_run: str = ".") -> list[FailEvidence]:
    transcripts = _read_jsonl(run_dir / TRANSCRIPTS)
    judge_rows = _read_jsonl(run_dir / JUDGE_RESULTS)
    judges = {_dialog_id(row): row for row in judge_rows if _dialog_id(row)}
    result: list[FailEvidence] = []
    for row in transcripts:
        dialog_id = _dialog_id(row)
        judge = judges.get(dialog_id, {})
        combined = {**row, **({"judge_result": judge} if judge else {})}
        verdict = _verdict(judge or row)
        if verdict != "FAIL":
            continue
        turn = _failing_turn(row, judge)
        judge_payload = _judge_payload(judge or row)
        result.append(
            FailEvidence(
                source_run=source_run,
                dialog_id=mask_pii(dialog_id),
                verdict=verdict,
                route=str(turn.get("bot_route") or turn.get("route") or row.get("route") or ""),
                client_text=_text_from(turn, ("client_text", "client_message", "user_text", "message")),
                bot_text=_text_from(turn, ("bot_text", "draft_text", "bot_draft_text", "answer")),
                rationale=mask_pii(str(judge_payload.get("rationale") or judge_payload.get("reason") or "")),
                fact_audit=judge_payload.get("fact_audit") or turn.get("judge_fact_audit") or turn.get("fact_audit") or row.get("fact_audit"),
                number_audit=judge_payload.get("number_audit") or turn.get("number_audit") or row.get("number_audit"),
                context_items=turn.get("bot_safe_context_items") or turn.get("ctx_items") or turn.get("context_items") or turn.get("crm_context") or row.get("ctx_items"),
                first_failing_turn=judge.get("first_failing_turn"),
            )
        )
    return result


def _summary_flags(path: Path) -> dict[str, Any]:
    summary = _read_json(path / SUMMARY)
    run_config = summary.get("run_config") if isinstance(summary.get("run_config"), Mapping) else {}
    flags = run_config.get("key_flags") if isinstance(run_config.get("key_flags"), Mapping) else {}
    return dict(flags)


def write_export(run_dir: Path, out_dir: Path, *, compare: Path | None = None) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = _candidate_run_dirs(run_dir)
    evidence = [
        item
        for source in run_dirs
        for item in collect_fail_evidence(
            source,
            source_run="." if source == run_dir else str(source.relative_to(run_dir)),
        )
    ]
    evidence.sort(key=lambda item: (item.source_run, item.dialog_id))
    jsonl = out_dir / "fail_raw_evidence.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for item in evidence:
            handle.write(mask_pii(json.dumps(asdict(item), ensure_ascii=False, default=str)) + "\n")
    md_lines = [f"# FAIL raw export\n\nRun: `{run_dir}`\n\nFAIL rows: {len(evidence)}\n"]
    for item in evidence:
        md_lines.append(f"\n## {item.dialog_id}\n\n- source_run: `{item.source_run}`\n- route: `{item.route}`\n- rationale: {item.rationale}\n\n**client**: {item.client_text}\n\n**bot**: {item.bot_text}\n")
    if compare is not None:
        left = _summary_flags(run_dir)
        right = _summary_flags(compare)
        md_lines.append("\n## Flag diff\n")
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                md_lines.append(f"- `{key}`: `{left.get(key)}` -> `{right.get(key)}`")
    md = out_dir / "fail_raw_evidence.md"
    md.write_text(mask_pii("\n".join(md_lines) + "\n"), encoding="utf-8")
    return {"status": "FAILS_EXPORTED" if evidence else "NO_FAILS_FOUND", "fail_count": len(evidence),
            "source_runs": ["." if path == run_dir else str(path.relative_to(run_dir)) for path in run_dirs],
            "jsonl": str(jsonl), "md": str(md)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export raw FAIL evidence from a dynamic run.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)
    out_dir = args.out_dir or (args.run_dir / "fail_raw_export")
    print(json.dumps(write_export(args.run_dir, out_dir, compare=args.compare), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
