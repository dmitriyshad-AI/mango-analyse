#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.report_adr003_overhandoff_levers import _load_gold_rows
from scripts.report_adr003_semantic_frame_eval import _load_transcripts, _strict_bool, _turns
from mango_mvp.knowledge_base.product_existence_axes_catalog import (
    build_product_existence_axes_catalog,
    verify_product_format_exists,
)


SCHEMA_VERSION = "adr003_existence_fact_verification_v1_2026_07_02"
SELF_ROUTES = {"bot_answer_self", "bot_answer_self_for_pilot", "self_answer"}
HANDOFF_ROUTES = {"manager_only", "draft_for_manager"}
RISK_MARKERS = {
    "p0",
    "refund",
    "legal",
    "complaint",
    "payment",
    "paid",
    "money",
    "оплат",
    "возврат",
    "жалоб",
    "договор",
}
NON_EXISTENCE_REFERENCE_MARKERS = {
    "price",
    "cost",
    "payment",
    "pay",
    "refund",
    "guarantee",
    "certificate",
    "discount",
    "trial",
    "цена",
    "стоим",
    "оплат",
    "возврат",
    "гарант",
    "сертифик",
    "скид",
    "пробн",
}
SUBJECT_ALIASES = {
    "physics": ("physics", "физик"),
    "физика": ("physics", "физик"),
    "math": ("math", "математ"),
    "математика": ("math", "математ"),
    "informatics": ("informatics", "programming", "информат", "программирован"),
    "информатика": ("informatics", "programming", "информат", "программирован"),
    "russian": ("russian", "русск"),
    "русский": ("russian", "русск"),
}
FORMAT_ALIASES = {
    "online": ("online", "онлайн", "soholms", "soho"),
    "онлайн": ("online", "онлайн", "soholms", "soho"),
    "offline": ("offline", "очно", "очная", "очный"),
    "очно": ("offline", "очно", "очная", "очный"),
}
PROGRAM_ALIASES = {
    "olympiad": ("olympiad", "олимпиад", "физтех", "рсош"),
    "олимпиад": ("olympiad", "олимпиад", "физтех", "рсош"),
    "camp": ("camp", "лагер", "лвш", "выезд", "школ", "летн", "смен"),
    "лагерь": ("camp", "лагер", "лвш", "выезд", "школ", "летн", "смен"),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report whether ADR-003 existence/format questions have fresh client-safe fact evidence."
    )
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--kb-snapshot", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(transcripts=args.transcripts, gold=args.gold, kb_snapshot=args.kb_snapshot)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "adr003_existence_fact_verification_report.json"
    md_path = args.out_dir / "adr003_existence_fact_verification_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))
    return 0


def build_report(*, transcripts: Path, gold: Path, kb_snapshot: Path) -> dict[str, Any]:
    dialogs = _load_transcripts(transcripts)
    turn_map = _build_turn_map(dialogs)
    gold_rows = _load_gold_rows(gold)
    kb_facts = _load_kb_facts(kb_snapshot)
    product_catalog = build_product_existence_axes_catalog(kb_facts)
    rows: list[dict[str, Any]] = []
    missing_turns: list[dict[str, Any]] = []
    for gold_row in gold_rows:
        key = (str(gold_row.get("dialog_id") or ""), int(gold_row.get("turn") or 0))
        turn = turn_map.get(key)
        if turn is None:
            missing_turns.append({"dialog_id": key[0], "turn": key[1]})
            continue
        row = _classify(gold_row=gold_row, turn=turn, product_catalog=product_catalog)
        rows.append(row)
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_rev": _source_rev(),
        "inputs": {
            "transcripts": str(transcripts),
            "transcripts_sha256": _sha256(transcripts),
            "gold": str(gold),
            "gold_sha256": _sha256(gold),
            "kb_snapshot": str(kb_snapshot),
            "kb_snapshot_sha256": _sha256(kb_snapshot),
        },
        "totals": _totals(rows, missing_turns),
        "breakdowns": _breakdowns(rows),
        "groups": _groups(rows),
        "acceptance": _acceptance(rows, missing_turns),
        "notes": [
            "Report-only scorer: runtime route/text/prompt behavior is unchanged.",
            "KB evidence comes from product_existence_axes_catalog, not ad-hoc report string matching.",
            "A future active step still needs runtime metadata wiring and paired eval before any route demotion.",
        ],
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), Mapping) else {}
    acceptance = report.get("acceptance") if isinstance(report.get("acceptance"), Mapping) else {}
    groups = report.get("groups") if isinstance(report.get("groups"), Mapping) else {}
    lines = [
        "# ADR-003 F2c Existence/Format Fact Verification",
        "",
        f"- Status: `{acceptance.get('status', 'unknown')}`",
        f"- Source rev: `{report.get('source_rev', 'unknown')}`",
        f"- Gold rows: `{totals.get('gold_rows', 0)}`",
        f"- Gold safe self rows: `{totals.get('gold_safe_self_rows', 0)}`",
        f"- Existence/format rows: `{totals.get('existence_format_rows', 0)}`",
        f"- Current handoff rows: `{totals.get('current_handoff_rows', 0)}`",
        f"- Handoff with exact KB evidence: `{totals.get('handoff_with_exact_kb_evidence', 0)}`",
        f"- Handoff without exact KB evidence: `{totals.get('handoff_without_exact_kb_evidence', 0)}`",
        f"- Already self with exact KB evidence: `{totals.get('already_self_with_exact_kb_evidence', 0)}`",
        f"- Already self without exact KB evidence: `{totals.get('already_self_without_exact_kb_evidence', 0)}`",
        f"- Excluded danger/money/P0 rows: `{totals.get('excluded_danger_money_p0', 0)}`",
        "",
        "## Группы",
        "",
    ]
    for name in (
        "handoff_with_exact_kb_evidence",
        "handoff_without_exact_kb_evidence",
        "already_self_with_exact_kb_evidence",
        "already_self_without_exact_kb_evidence",
        "excluded_danger_money_p0",
        "not_existence_format",
    ):
        value = groups.get(name) if isinstance(groups.get(name), Mapping) else {}
        lines.append(f"- `{name}`: `{value.get('count', 0)}`")
    lines.extend(["", "## Handoff с KB-доказательством", ""])
    evidence_group = groups.get("handoff_with_exact_kb_evidence") if isinstance(groups.get("handoff_with_exact_kb_evidence"), Mapping) else {}
    for item in evidence_group.get("examples", [])[:20]:
        lines.append(_example_markdown(item))
    lines.extend(["", "## Handoff без KB-доказательства", ""])
    missing_group = groups.get("handoff_without_exact_kb_evidence") if isinstance(groups.get("handoff_without_exact_kb_evidence"), Mapping) else {}
    for item in missing_group.get("examples", [])[:20]:
        lines.append(_example_markdown(item))
    lines.extend(["", "## Acceptance Notes", ""])
    for note in acceptance.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _example_markdown(item: Mapping[str, Any]) -> str:
    requested = item.get("requested_product") if isinstance(item.get("requested_product"), Mapping) else {}
    best = item.get("best_kb_match") if isinstance(item.get("best_kb_match"), Mapping) else {}
    return (
        f"- `{item.get('dialog_id')}#{item.get('turn')}` route=`{item.get('route')}` "
        f"action=`{item.get('requested_action')}` axes=`{','.join(item.get('requested_axes') or [])}` "
        f"evidence=`{item.get('evidence_level')}`\n"
        f"  - client: {item.get('client_excerpt')}\n"
        f"  - requested: brand={requested.get('brand','')} subject={requested.get('subject','')} "
        f"grade={requested.get('grade','')} format={requested.get('format','')} program={requested.get('program_kind','')}\n"
        f"  - best_fact: {best.get('fact_key','')} ({best.get('match_level','')}, hits={','.join(best.get('axis_hits') or [])})"
    )


def _build_turn_map(dialogs: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    result: dict[tuple[str, int], Mapping[str, Any]] = {}
    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or "")
        for index, turn in enumerate(_turns(dialog), 1):
            merged = dict(turn)
            merged.setdefault("__dialog_brand", dialog.get("brand"))
            merged.setdefault("__dialog_id", dialog_id)
            result[(dialog_id, int(turn.get("turn") or index))] = merged
    return result


def _classify(
    *,
    gold_row: Mapping[str, Any],
    turn: Mapping[str, Any],
    product_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    expected = gold_row.get("expected") if isinstance(gold_row.get("expected"), Mapping) else gold_row
    frame = turn.get("bot_semantic_frame") if isinstance(turn.get("bot_semantic_frame"), Mapping) else {}
    requested_product = frame.get("requested_product") if isinstance(frame.get("requested_product"), Mapping) else {}
    route = str(turn.get("bot_route") or "")
    is_gold_safe_self = (
        _strict_bool(expected.get("must_handoff")) is False
        and _clean(expected.get("risk_class")) == "safe"
        and _clean(expected.get("answerability")) == "answer_self"
    )
    row: dict[str, Any] = {
        "dialog_id": str(gold_row.get("dialog_id") or turn.get("__dialog_id") or ""),
        "turn": int(gold_row.get("turn") or turn.get("turn") or 0),
        "brand": _active_brand(turn, requested_product),
        "route": route,
        "client_excerpt": _redacted_excerpt(turn.get("client_message")),
        "bot_excerpt": _redacted_excerpt(turn.get("bot_text")),
        "requested_action": _clean(frame.get("requested_action")),
        "frame_must_handoff": _strict_bool(frame.get("must_handoff")),
        "frame_risk_class": _clean(frame.get("risk_class")),
        "frame_answerability": _clean(frame.get("answerability")),
        "frame_confidence": _float_or_none(frame.get("confidence")),
        "requested_product": {
            "brand": str(requested_product.get("brand") or "").strip().casefold(),
            "subject": str(requested_product.get("subject") or "").strip(),
            "grade": str(requested_product.get("grade") or "").strip(),
            "format": str(requested_product.get("format") or "").strip(),
            "program_kind": str(requested_product.get("program_kind") or "").strip(),
            "raw_text": str(requested_product.get("raw_text") or "").strip()[:160],
        },
        "gold": {
            "expected": expected,
            "review_label": str(gold_row.get("review_label") or ""),
            "notes": str(gold_row.get("notes") or "")[:240],
        },
        "is_gold_safe_self": is_gold_safe_self,
        "current_handoff": route in HANDOFF_ROUTES,
        "current_self": route in SELF_ROUTES,
    }
    row["requested_axes"] = _requested_axes(row["requested_product"])
    row["existence_format_candidate"] = _is_existence_format_candidate(row)
    row["excluded_danger_money_p0"] = _danger_money_p0(turn, frame, gold_row)
    runtime_evidence = _runtime_fact_evidence(turn, row["requested_product"])
    product_check = _product_existence_check(product_catalog, row["requested_product"], active_brand=row["brand"])
    kb_matches = _product_check_matches(product_check)
    row["runtime_fact_evidence"] = runtime_evidence
    row["product_existence_check"] = product_check
    row["best_kb_match"] = kb_matches[0] if kb_matches else {}
    row["kb_match_count"] = len(kb_matches)
    row["evidence_level"] = _evidence_level(runtime_evidence, product_check)
    if not is_gold_safe_self:
        row["group"] = "not_gold_safe_self"
    elif not row["existence_format_candidate"]:
        row["group"] = "not_existence_format"
    elif row["excluded_danger_money_p0"]:
        row["group"] = "excluded_danger_money_p0"
    elif route in HANDOFF_ROUTES and row["evidence_level"] == "kb_exact":
        row["group"] = "handoff_with_exact_kb_evidence"
    elif route in HANDOFF_ROUTES:
        row["group"] = "handoff_without_exact_kb_evidence"
    elif route in SELF_ROUTES and row["evidence_level"] == "kb_exact":
        row["group"] = "already_self_with_exact_kb_evidence"
    elif route in SELF_ROUTES:
        row["group"] = "already_self_without_exact_kb_evidence"
    else:
        row["group"] = "not_existence_format"
    return row


def _load_kb_facts(path: Path) -> list[Mapping[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    facts = data.get("facts") if isinstance(data, Mapping) else data
    if not isinstance(facts, list):
        return []
    return [item for item in facts if isinstance(item, Mapping)]


def _product_existence_check(
    product_catalog: Mapping[str, Any],
    requested: Mapping[str, Any],
    *,
    active_brand: str,
) -> dict[str, Any]:
    raw_text = str(requested.get("raw_text") or "")
    check = verify_product_format_exists(
        product_catalog,
        brand=active_brand or str(requested.get("brand") or ""),
        grade=requested.get("grade"),
        subject=str(requested.get("subject") or raw_text),
        format=str(requested.get("format") or raw_text),
        program_kind=str(requested.get("program_kind") or raw_text),
        product_family=str(requested.get("product_family") or ""),
    )
    return dict(check)


def _product_check_matches(product_check: Mapping[str, Any]) -> list[dict[str, Any]]:
    matches = product_check.get("matches")
    if not isinstance(matches, Sequence):
        return []
    result: list[dict[str, Any]] = []
    requested_axes = product_check.get("query_axes") if isinstance(product_check.get("query_axes"), Mapping) else {}
    status = str(product_check.get("status") or "")
    for entry in matches:
        if not isinstance(entry, Mapping):
            continue
        match_level = "exact" if status in {"exists", "not_offered"} else status or "unknown"
        result.append(
            {
                "fact_key": str(entry.get("source_fact_key") or entry.get("source_fact_id") or "").strip(),
                "fact_type": str(entry.get("source_fact_type") or "").strip(),
                "product": str(entry.get("product_family") or entry.get("program_kind") or "").strip(),
                "valid_until": str(entry.get("valid_until") or "").strip(),
                "axis_hits": [
                    key
                    for key in ("subject", "grade", "format", "program_kind", "product_family")
                    if requested_axes.get(key)
                ],
                "requested_axes": [key for key, value in requested_axes.items() if value],
                "match_level": match_level,
                "existence_status": str(entry.get("existence_status") or ""),
                "client_safe_text_excerpt": _redacted_excerpt(entry.get("client_safe_text"), limit=240),
                "score": 100 if match_level == "exact" else 0,
            }
        )
    return result[:8]


def _runtime_fact_evidence(turn: Mapping[str, Any], requested: Mapping[str, Any]) -> dict[str, Any]:
    facts = [str(item or "") for item in (turn.get("bot_confirmed_facts") or []) if str(item or "").strip()]
    requested_axes = _requested_axes(requested)
    best: dict[str, Any] = {}
    for fact in facts:
        hits = _axis_hits(requested, _normalize(fact))
        exact = bool(requested_axes) and set(hits) >= set(requested_axes)
        score = len(hits) + (10 if exact else 0)
        if score > int(best.get("score") or -1):
            best = {
                "axis_hits": hits,
                "requested_axes": requested_axes,
                "match_level": "exact" if exact else "partial" if hits else "none",
                "text_excerpt": _redacted_excerpt(fact, limit=240),
                "score": score,
            }
    if not best:
        best = {"axis_hits": [], "requested_axes": requested_axes, "match_level": "none", "score": 0}
    return best


def _evidence_level(runtime: Mapping[str, Any], product_check: Mapping[str, Any]) -> str:
    if product_check.get("status") in {"exists", "not_offered"}:
        return "kb_exact"
    if runtime.get("match_level") == "exact":
        return "runtime_exact"
    if product_check.get("status") in {"unknown", "needs_slot"}:
        return str(product_check.get("status"))
    if runtime.get("match_level") == "partial":
        return "runtime_partial"
    return "none"


def _requested_axes(requested: Mapping[str, Any]) -> list[str]:
    axes: list[str] = []
    if str(requested.get("subject") or "").strip():
        axes.append("subject")
    if _grade_digits(requested.get("grade")):
        axes.append("grade")
    if _canonical_format(requested.get("format")):
        axes.append("format")
    if _canonical_program(requested.get("program_kind") or requested.get("raw_text")):
        axes.append("program_kind")
    return axes


def _axis_hits(requested: Mapping[str, Any], haystack: str) -> list[str]:
    hits: list[str] = []
    subject = _canonical_subject(requested.get("subject") or requested.get("raw_text"))
    if subject and any(alias in haystack for alias in SUBJECT_ALIASES[subject]):
        hits.append("subject")
    grade = _grade_digits(requested.get("grade") or requested.get("raw_text"))
    if grade and _grade_in_text(grade, haystack):
        hits.append("grade")
    fmt = _canonical_format(requested.get("format") or requested.get("raw_text"))
    if fmt and any(alias in haystack for alias in FORMAT_ALIASES[fmt]):
        hits.append("format")
    program = _canonical_program(requested.get("program_kind") or requested.get("raw_text"))
    if program and any(alias in haystack for alias in PROGRAM_ALIASES[program]):
        hits.append("program_kind")
    return hits


def _is_existence_format_candidate(row: Mapping[str, Any]) -> bool:
    action = str(row.get("requested_action") or "")
    notes = str(row.get("gold", {}).get("notes") or "").casefold()
    text = " ".join(
        [
            str(row.get("client_excerpt") or ""),
            str(row.get("bot_excerpt") or ""),
            str(row.get("requested_product", {}).get("raw_text") if isinstance(row.get("requested_product"), Mapping) else ""),
            notes,
        ]
    ).casefold()
    if any(marker in text for marker in NON_EXISTENCE_REFERENCE_MARKERS):
        return False
    requested = row.get("requested_product") if isinstance(row.get("requested_product"), Mapping) else {}
    if action == "check_availability" and row.get("is_gold_safe_self"):
        return True
    if any(marker in notes for marker in ("existence", "format", "course", "camp", "существ", "формат", "курс", "лагер")):
        return True
    return False


def _danger_money_p0(turn: Mapping[str, Any], frame: Mapping[str, Any], gold_row: Mapping[str, Any]) -> bool:
    text = " ".join(
        [
            str(turn.get("__dialog_id") or ""),
            str(turn.get("client_message") or ""),
            str(gold_row.get("notes") or ""),
            str(frame.get("payment_readiness") or ""),
            str(frame.get("risk_class") or ""),
            " ".join(str(flag) for flag in (turn.get("bot_safety_flags") or [])),
        ]
    ).casefold()
    return any(marker in text for marker in RISK_MARKERS)


def _active_brand(turn: Mapping[str, Any], requested: Mapping[str, Any]) -> str:
    for value in (turn.get("brand"), turn.get("__dialog_brand"), turn.get("active_brand"), requested.get("brand")):
        clean = _clean(value)
        if clean in {"foton", "unpk"}:
            return clean
    return ""


def _canonical_subject(value: Any) -> str:
    text = _normalize(value)
    for canonical, aliases in SUBJECT_ALIASES.items():
        if any(alias in text for alias in aliases):
            return canonical
    return ""


def _canonical_format(value: Any) -> str:
    text = _normalize(value)
    for canonical, aliases in FORMAT_ALIASES.items():
        if any(alias in text for alias in aliases):
            return canonical
    return ""


def _canonical_program(value: Any) -> str:
    text = _normalize(value)
    for canonical, aliases in PROGRAM_ALIASES.items():
        if any(alias in text for alias in aliases):
            return canonical
    return ""


def _grade_digits(value: Any) -> str:
    match = re.search(r"\b([1-9]|1[01])\b", str(value or ""))
    return match.group(1) if match else ""


def _grade_in_text(grade: str, text: str) -> bool:
    if not grade:
        return False
    return bool(re.search(rf"(^|\D){re.escape(grade)}(\D|$)", text))


def _totals(rows: Sequence[Mapping[str, Any]], missing_turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    safe_rows = [row for row in rows if row.get("is_gold_safe_self")]
    existence_rows = [row for row in safe_rows if row.get("existence_format_candidate")]
    return {
        "gold_rows": len(rows),
        "missing_gold_turns": len(missing_turns),
        "missing_gold_turn_examples": list(missing_turns)[:10],
        "gold_safe_self_rows": len(safe_rows),
        "existence_format_rows": len(existence_rows),
        "current_handoff_rows": sum(1 for row in existence_rows if row.get("current_handoff")),
        "handoff_with_exact_kb_evidence": sum(1 for row in existence_rows if row.get("group") == "handoff_with_exact_kb_evidence"),
        "handoff_without_exact_kb_evidence": sum(1 for row in existence_rows if row.get("group") == "handoff_without_exact_kb_evidence"),
        "already_self_with_exact_kb_evidence": sum(1 for row in existence_rows if row.get("group") == "already_self_with_exact_kb_evidence"),
        "already_self_without_exact_kb_evidence": sum(1 for row in existence_rows if row.get("group") == "already_self_without_exact_kb_evidence"),
        "excluded_danger_money_p0": sum(1 for row in existence_rows if row.get("group") == "excluded_danger_money_p0"),
    }


def _breakdowns(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    existence = [row for row in rows if row.get("existence_format_candidate")]
    return {
        "by_group": _counter(existence, "group"),
        "by_route": _counter(existence, "route"),
        "by_requested_action": _counter(existence, "requested_action"),
        "by_evidence_level": _counter(existence, "evidence_level"),
    }


def _groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group in (
        "handoff_with_exact_kb_evidence",
        "handoff_without_exact_kb_evidence",
        "already_self_with_exact_kb_evidence",
        "already_self_without_exact_kb_evidence",
        "excluded_danger_money_p0",
        "not_existence_format",
    ):
        grouped = [row for row in rows if row.get("group") == group]
        result[group] = {"count": len(grouped), "examples": grouped[:50]}
    return result


def _acceptance(rows: Sequence[Mapping[str, Any]], missing_turns: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    notes: list[str] = []
    status = "pass"
    if missing_turns:
        status = "needs_review"
        notes.append("Some gold rows are absent from transcripts.")
    if any(row.get("group") == "handoff_with_exact_kb_evidence" for row in rows):
        notes.append("There are current handoff rows with exact KB evidence; these are candidates for a future fact-gated shadow, not active use.")
    else:
        notes.append("No current handoff rows have exact KB evidence in this scorer; route-only active remains a no-go.")
    if any(row.get("group") == "already_self_without_exact_kb_evidence" for row in rows):
        notes.append("Some already-self existence/format answers lack exact KB evidence in metadata/scorer; improve fact trace before active policy.")
    notes.append("Any active self-answer still requires verified exact facts in runtime metadata, not this offline diagnostic matcher.")
    return {"status": status, "notes": notes}


def _counter(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "") for row in rows).items()))


def _normalize(value: Any) -> str:
    text = str(value or "").casefold()
    text = text.replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clean(value: Any) -> str:
    return str(value or "").strip().casefold()


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _redacted_excerpt(value: Any, *, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", "[email]", text)
    text = re.sub(r"(?<!\d)(?:\+?7|8)?[\s(.-]*\d{3}[\s).-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)", "[phone]", text)
    text = re.sub(r"(?<!\d)\d{7,}(?!\d)", "[id]", text)
    return text[:limit]


def _source_rev() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
