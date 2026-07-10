from __future__ import annotations

import json
from pathlib import Path

from scripts import report_adr003_exact_proof_injection_shadow as report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _gold(dialog_id: str) -> dict:
    return {
        "dialog_id": dialog_id,
        "turn": 1,
        "expected": {
            "must_handoff": False,
            "risk_class": "safe",
            "answerability": "answer_self",
            "requested_action": "answer_question",
        },
        "notes": "safe reference: course existence/format without live seats",
    }


def _fact(*, valid_until: str = "2026-08-31") -> dict:
    return {
        "brand": "unpk",
        "fact_key": "lvsh_mendeleevo_2026.directions.fizmat.classes",
        "fact_type": "course_parameter",
        "product": "lvsh_mendeleevo_2026",
        "program_kind": "camp_lvsh",
        "client_safe_text": "УНПК: ЛВШ Менделеево для физико-математического направления рассчитана на 5-10 классы.",
        "allowed_for_client_answer": True,
        "forbidden_for_client": False,
        "internal_only": False,
        "valid_until": valid_until,
        "structured_value": {"classes_raw": "5-10", "raw_value": "5-10", "valid_until": valid_until},
    }


def _turn(
    *,
    route: str = "manager_only",
    message_type: str = "context_update",
    frame: dict | None = None,
    missing_facts: list[str] | None = None,
) -> dict:
    semantic_frame = {
        "risk_class": "safe",
        "answerability": "answer_self",
        "requested_action": "answer_question",
        "must_handoff": False,
        "confidence": 0.95,
        "requested_product": {
            "brand": "unpk",
            "subject": "",
            "grade": "5",
            "format": "",
            "program_kind": "летняя школа",
            "raw_text": "ребёнок закончил 5 класс",
        },
    }
    semantic_frame.update(frame or {})
    return {
        "turn": 1,
        "client_message": "Ребёнок закончил 5 класс",
        "bot_text": "Передам менеджеру.",
        "bot_route": route,
        "bot_message_type": message_type,
        "bot_reason_class": "policy_permission",
        "bot_safety_flags": ["manager_approval_required", "no_auto_send", f"message_type_{message_type}"],
        "bot_missing_facts": missing_facts if missing_facts is not None else ["актуальное наличие мест"],
        "bot_semantic_frame": semantic_frame,
    }


def _build(tmp_path: Path, *, turn: dict, facts: list[dict] | None = None) -> dict:
    transcripts = tmp_path / "transcripts.jsonl"
    gold = tmp_path / "gold.jsonl"
    kb = tmp_path / "kb.json"
    _write_jsonl(transcripts, [{"dialog_id": "d1", "brand": "unpk", "turns": [turn]}])
    _write_jsonl(gold, [_gold("d1")])
    _write_json(kb, {"facts": facts or [_fact()]})
    return report.build_report(transcripts=transcripts, gold=gold, kb_snapshot=kb, as_of_date=report._parse_date("2026-07-02"))


def test_fresh_exact_proof_alone_does_not_demote_manager_only(tmp_path: Path) -> None:
    result = _build(tmp_path, turn=_turn(message_type="question", missing_facts=[]))

    assert result["totals"]["manager_only_exact_proof_rows"] == 1
    assert result["totals"]["fresh_client_safe_exact_proof"] == 1
    assert result["totals"]["evidence_only_sufficient_rows"] == 1
    case = result["cases"][0]
    assert case["fresh_client_safe_exact_proof"] is True
    assert case["evidence_only_sufficient"] is True
    assert case["residual_blockers"] == ["route_is_manager_only"]
    assert result["acceptance"]["active_readiness"] == "no_go"


def test_context_update_and_live_missing_facts_remain_residual_blockers(tmp_path: Path) -> None:
    result = _build(tmp_path, turn=_turn())

    case = result["cases"][0]
    assert "message_type_context_update" in case["residual_blockers"]
    assert "runtime_missing_live_or_operational_facts" in case["residual_blockers"]
    assert case["evidence_only_sufficient"] is False


def test_low_confidence_remains_residual_blocker(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        turn=_turn(message_type="question", missing_facts=[], frame={"confidence": 0.86}),
    )

    case = result["cases"][0]
    assert "frame_confidence_below_threshold" in case["residual_blockers"]
    assert case["evidence_only_sufficient"] is False


def test_manager_action_frame_remains_residual_blocker(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        turn=_turn(
            message_type="question",
            missing_facts=[],
            frame={
                "risk_class": "manager_action",
                "answerability": "manager_only",
                "requested_action": "check_availability",
                "must_handoff": True,
                "confidence": 0.94,
            },
        ),
    )

    case = result["cases"][0]
    assert "frame_risk_not_safe" in case["residual_blockers"]
    assert "frame_answerability_not_self" in case["residual_blockers"]
    assert "frame_action_not_safe_reference" in case["residual_blockers"]
    assert "frame_must_handoff" in case["residual_blockers"]
    assert case["evidence_only_sufficient"] is False


def test_expired_exact_proof_is_not_fresh_client_safe(tmp_path: Path) -> None:
    _ = tmp_path
    entry = {
        "existence_status": "exists",
        "client_safe_text": "УНПК: ЛВШ Менделеево рассчитана на 5-10 классы.",
        "valid_until": "2026-06-30",
    }

    assert report._fresh_client_safe_exact_proof(
        entry,
        best={},
        as_of_date=report._parse_date("2026-07-02"),
    ) is False


def test_markdown_is_redacted_from_client_text(tmp_path: Path) -> None:
    result = _build(tmp_path, turn=_turn())
    rendered = report.render_markdown(result)

    assert "Ребёнок закончил 5 класс" not in rendered
    assert "Передам менеджеру" not in rendered
    assert "lvsh_mendeleevo_2026.directions.fizmat.classes" in rendered
