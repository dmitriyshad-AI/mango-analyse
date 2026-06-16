#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.dialogue_memory import MEMORY_PROVENANCE_ENV, build_dialogue_memory
from mango_mvp.channels.new_lead_funnel import (
    ANCHORED_BARE_GRADE_ENV,
    build_lead_funnel_state,
    extract_format,
    extract_grade,
)


CASES: tuple[Mapping[str, Any], ...] = (
    {"id": "P1", "text": "физика 8 онлайн", "brand": "unpk", "expected_on_grade": "8", "positive": True},
    {"id": "P2", "text": "8 класс математика онлайн", "brand": "foton", "expected_on_grade": "8", "positive": True},
    {"id": "P3", "text": "информатика 10 очно", "brand": "unpk", "expected_on_grade": "10", "positive": True},
    {"id": "P4", "text": "9 класс, ОГЭ", "brand": "unpk", "expected_on_grade": "9", "positive": True},
    {
        "id": "P5",
        "text": "а по физике?",
        "initial_text": "физика 8 онлайн",
        "brand": "unpk",
        "expected_on_grade": "8",
        "positive": True,
    },
    {"id": "N1", "text": "звоните на 8 800 555 35 35, физика онлайн", "brand": "unpk", "trap": True},
    {"id": "N2", "text": "нам с 8 лет можно? физика онлайн", "brand": "unpk", "trap": True},
    {"id": "N3", "text": "в 8:00 удобно? физика онлайн", "brand": "unpk", "trap": True},
    {"id": "N4", "text": "у меня 2 детей, физика онлайн", "brand": "unpk", "trap": True},
    {"id": "N5", "text": "информатика и математика 8 онлайн", "brand": "unpk", "multi_subject": True},
    {"id": "N6", "text": "онлайн и очно сравниваю, физика 8", "brand": "unpk", "format_unselected": True},
    {"id": "N7", "text": "дважды списали, верните", "brand": "unpk", "p0": True},
    {"id": "N8", "text": "дважды списали, верните; а в каком классе берёте?", "brand": "unpk", "p0": True},
    {"id": "N9", "text": "посмотрите баланс по +7 916 123 45 67", "brand": "unpk", "trap": True},
    {"id": "N10", "text": "физика 8 онлайн", "brand": "unpk", "off_parity": True},
)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.parallel < 1:
        raise SystemExit("--parallel must be >= 1")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    previous_grade = os.environ.get(ANCHORED_BARE_GRADE_ENV)
    previous_provenance = os.environ.get(MEMORY_PROVENANCE_ENV)
    for mode in ("off", "on"):
        os.environ[MEMORY_PROVENANCE_ENV] = "1"
        os.environ[ANCHORED_BARE_GRADE_ENV] = "1" if mode == "on" else "0"
        try:
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                future_to_case = {executor.submit(run_case, case, mode=mode): case for case in CASES}
                for future in as_completed(future_to_case):
                    rows.append(future.result())
        finally:
            restore_env(ANCHORED_BARE_GRADE_ENV, previous_grade)
            restore_env(MEMORY_PROVENANCE_ENV, previous_provenance)
    rows.sort(key=lambda row: (str(row["case_id"]), str(row["mode"])))
    summary = build_summary(rows, out_dir=out_dir, parallel=args.parallel)

    write_jsonl(out_dir / "tz124_slot_anchor_results.jsonl", rows)
    (out_dir / "transcripts.md").write_text(render_transcripts(rows), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["gate_passed"] else 2


def run_case(case: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    memory = build_case_memory(case)
    view = dict(memory.to_prompt_view())
    lead = build_lead_funnel_state(
        str(case["text"]),
        active_brand=str(case.get("brand") or "unpk"),
        topic_id="theme:001_pricing",
    )
    known_slots = dict(view.get("known_slots") or {})
    return {
        "case_id": case["id"],
        "mode": mode,
        "client_text": case["text"],
        "memory_known_slots": known_slots,
        "memory_slot_sources": dict(view.get("slot_sources") or {}),
        "memory_client_confirmed_slots": dict(view.get("client_confirmed_slots") or {}),
        "memory_risk_flags": list(view.get("risk_flags") or ()),
        "memory_handoff_state": view.get("handoff_state"),
        "lead_known_slots": lead.known_slots.to_json_dict(),
        "lead_missing_slots": list(lead.missing_slots),
        "lead_next_step_type": lead.next_step_type,
        "flat_extract_grade": extract_grade(str(case["text"])),
        "flat_extract_format": extract_format(str(case["text"])),
        "checks": case_checks(case, mode=mode, known_slots=known_slots, view=view, lead=lead),
        "price_mentions": [],
    }


def build_case_memory(case: Mapping[str, Any]) -> Any:
    if case.get("initial_text"):
        initial = build_dialogue_memory(
            current_message=str(case["initial_text"]),
            active_brand=str(case.get("brand") or "unpk"),
            session_id=f"tz124:{case['id']}",
        )
        return build_dialogue_memory(
            current_message=str(case["text"]),
            active_brand=str(case.get("brand") or "unpk"),
            previous_memory=initial,
            session_id=f"tz124:{case['id']}",
        )
    return build_dialogue_memory(
        current_message=str(case["text"]),
        active_brand=str(case.get("brand") or "unpk"),
        session_id=f"tz124:{case['id']}",
    )


def case_checks(
    case: Mapping[str, Any],
    *,
    mode: str,
    known_slots: Mapping[str, Any],
    view: Mapping[str, Any],
    lead: Any,
) -> dict[str, bool]:
    grade = str(known_slots.get("grade") or "")
    subject = str(known_slots.get("subject") or "")
    fmt = str(known_slots.get("format") or "")
    checks: dict[str, bool] = {}
    if case.get("positive") and mode == "on":
        checks["positive_grade"] = grade == str(case.get("expected_on_grade") or "")
        checks["route_not_manager"] = view.get("handoff_state") != "required"
    if case.get("trap") and mode == "on":
        checks["trap_no_grade"] = grade == ""
    if case.get("multi_subject") and mode == "on":
        checks["no_single_subject_scope"] = subject == "" or "," in subject
    if case.get("format_unselected") and mode == "on":
        checks["format_unselected"] = fmt == "" and extract_format(str(case["text"])) == ""
    if case.get("p0") and mode == "on":
        checks["p0_handoff"] = view.get("handoff_state") == "required"
    if case.get("off_parity") and mode == "off":
        checks["off_no_bare_grade"] = grade == "" and lead.known_slots.grade == "" and extract_grade(str(case["text"])) == ""
    return checks


def build_summary(rows: list[Mapping[str, Any]], *, out_dir: Path, parallel: int) -> dict[str, Any]:
    failed_checks: list[Mapping[str, Any]] = []
    for row in rows:
        for name, passed in dict(row.get("checks") or {}).items():
            if not passed:
                failed_checks.append({"case_id": row["case_id"], "mode": row["mode"], "check": name})
    on_rows = [row for row in rows if row.get("mode") == "on"]
    off_rows = [row for row in rows if row.get("mode") == "off"]
    stop_conditions = {
        "false_grade_from_number_trap": any(
            row["case_id"] in {"N1", "N2", "N3", "N4", "N9"}
            and (row.get("memory_known_slots") or {}).get("grade")
            for row in on_rows
        ),
        "format_choice_selected": any(
            row["case_id"] == "N6" and (row.get("memory_known_slots") or {}).get("format") for row in on_rows
        ),
        "price_under_extracted_class": any(row.get("price_mentions") for row in on_rows),
        "off_changed_bare_grade": any(
            row["case_id"] == "N10" and (row.get("memory_known_slots") or {}).get("grade") for row in off_rows
        ),
    }
    return {
        "schema_version": "tz124_slot_anchor_pack_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "parallel": parallel,
        "rows_total": len(rows),
        "mode_counts": dict(Counter(str(row.get("mode")) for row in rows).most_common()),
        "failed_checks": failed_checks,
        "stop_conditions": stop_conditions,
        "gate_passed": not failed_checks and not any(stop_conditions.values()),
        "llm_calls_total": 0,
        "safety": {
            "writes_crm": False,
            "writes_tallanto": False,
            "writes_amo": False,
            "runs_asr": False,
            "touches_stable_runtime": False,
            "touches_dialogue_contract_pipeline": False,
        },
    }


def render_transcripts(rows: list[Mapping[str, Any]]) -> str:
    lines = ["# TZ-124 Slot Anchor Pack Transcripts", ""]
    for row in rows:
        known = row.get("memory_known_slots") or {}
        lines.extend(
            [
                f"## {row['case_id']} / {row['mode']}",
                "",
                f"CLIENT: {row['client_text']}",
                f"MEMORY_GRADE: {known.get('grade', '')}",
                f"MEMORY_SUBJECT: {known.get('subject', '')}",
                f"MEMORY_FORMAT: {known.get('format', '')}",
                f"ROUTE_HINT: handoff_state={row.get('memory_handoff_state')} next_step={row.get('lead_next_step_type')}",
                f"PRICE_MENTIONS: {', '.join(row.get('price_mentions') or [])}",
                f"CHECKS: {json.dumps(row.get('checks') or {}, ensure_ascii=False, sort_keys=True)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TZ-124 compact slot extraction pack OFF->ON.")
    parser.add_argument("--out-dir", default="audits/_inbox/tz124_slot_anchor_pack_20260616")
    parser.add_argument("--parallel", type=int, default=4)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
