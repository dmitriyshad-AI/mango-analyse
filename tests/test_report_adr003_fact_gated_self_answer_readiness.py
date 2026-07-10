from __future__ import annotations

import json
from pathlib import Path

from scripts import report_adr003_fact_gated_self_answer_readiness as report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _gold(dialog_id: str, *, notes: str = "safe reference: course existence/format without live seats") -> dict:
    return {
        "dialog_id": dialog_id,
        "turn": 1,
        "expected": {
            "must_handoff": False,
            "risk_class": "safe",
            "answerability": "answer_self",
            "requested_action": "answer_question",
        },
        "notes": notes,
    }


def _dialog(
    dialog_id: str,
    *,
    route: str,
    frame: dict | None = None,
    safety_flags: list[str] | None = None,
    client_message: str = "Олимпиадная физика онлайн для 9 класса есть?",
) -> dict:
    semantic_frame = {
        "risk_class": "safe",
        "answerability": "answer_self",
        "requested_action": "answer_question",
        "deal_stage": "qualification",
        "payment_readiness": "none",
        "must_handoff": False,
        "confidence": 0.94,
        "requested_product": {
            "brand": "unpk",
            "subject": "физика",
            "grade": "9 класс",
            "format": "онлайн",
            "program_kind": "олимпиадная подготовка",
            "raw_text": "олимпиадная физика онлайн для 9 класса",
        },
    }
    semantic_frame.update(frame or {})
    return {
        "dialog_id": dialog_id,
        "brand": "unpk",
        "turns": [
            {
                "turn": 1,
                "brand": "unpk",
                "client_message": client_message,
                "bot_text": "Менеджер поможет проверить подходящий вариант.",
                "bot_route": route,
                "bot_safety_flags": safety_flags or ["manager_approval_required", "no_auto_send"],
                "bot_semantic_frame": semantic_frame,
                "bot_semantic_frame_self_answer_shadow": {
                    "guards": {"actual_p0": False, "blocking_flags": []}
                },
                "bot_missing_facts": [],
            }
        ],
    }


def _fact(*, text: str = "Физика, 9 класс, олимпиадная группа, онлайн, старт 19.09.2026.") -> dict:
    return {
        "brand": "unpk",
        "fact_key": "schedule_2026_27.physics_9_olympiad_online",
        "fact_type": "deadline",
        "product": "schedule_2026_27",
        "client_safe_text": text,
        "allowed_for_client_answer": True,
        "forbidden_for_client": False,
        "internal_only": False,
        "valid_until": "2027-05-30",
    }


def _negative_fact() -> dict:
    return {
        "brand": "unpk",
        "fact_key": "unpk.no_chemistry",
        "fact_type": "program",
        "product": "unavailable",
        "client_safe_text": "Химии сейчас нет.",
        "allowed_for_client_answer": True,
        "forbidden_for_client": False,
        "internal_only": False,
        "valid_until": "2027-05-30",
        "structured_value": {"negative_fact": True},
    }


def _build(tmp_path: Path, *, dialogs: list[dict], facts: list[dict], gold_rows: list[dict] | None = None) -> dict:
    transcripts = tmp_path / "transcripts.jsonl"
    gold = tmp_path / "gold.jsonl"
    kb = tmp_path / "kb.json"
    rows = gold_rows or [_gold(dialog["dialog_id"]) for dialog in dialogs]
    _write_jsonl(transcripts, dialogs)
    _write_jsonl(gold, rows)
    _write_json(kb, {"facts": facts})
    return report.build_report(transcripts=transcripts, gold=gold, kb_snapshot=kb)


def test_draft_for_manager_with_exact_proof_is_strict_f3_candidate(tmp_path: Path) -> None:
    result = _build(tmp_path, dialogs=[_dialog("d1", route="draft_for_manager")], facts=[_fact()])

    assert result["totals"]["strict_f3_draft_candidates"] == 1
    assert result["acceptance"]["active_readiness"] == "needs_claude_reggrade_before_active"
    candidate = result["groups"]["strict_f3_draft_candidate"]["examples"][0]
    assert candidate["strict_f3_candidate"] is True
    assert candidate["evidence_level"] == "kb_exact"


def test_manager_only_with_exact_proof_is_not_active_candidate(tmp_path: Path) -> None:
    result = _build(tmp_path, dialogs=[_dialog("d1", route="manager_only")], facts=[_fact()])

    assert result["totals"]["strict_f3_draft_candidates"] == 0
    assert result["totals"]["manager_only_exact_proof_needs_policy"] == 1
    assert result["acceptance"]["active_readiness"] == "no_go"
    row = result["groups"]["manager_only_exact_proof_needs_policy"]["examples"][0]
    assert row["blocked_reasons"] == ["route_is_manager_only"]


def test_manager_only_with_non_self_frame_is_still_policy_bucket(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[
            _dialog(
                "d1",
                route="manager_only",
                frame={
                    "risk_class": "manager_action",
                    "answerability": "manager_only",
                    "requested_action": "check_availability",
                    "must_handoff": True,
                    "confidence": 0.94,
                },
            )
        ],
        facts=[_fact()],
    )

    assert result["totals"]["strict_f3_draft_candidates"] == 0
    assert result["totals"]["manager_only_exact_proof_needs_policy"] == 1
    row = result["groups"]["manager_only_exact_proof_needs_policy"]["examples"][0]
    assert "route_is_manager_only" in row["blocked_reasons"]
    assert "frame_risk_not_safe" in row["blocked_reasons"]
    assert "frame_must_handoff" in row["blocked_reasons"]


def test_no_exact_proof_blocks_candidate(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[_dialog("d1", route="draft_for_manager")],
        facts=[_fact(text="Физика, онлайн-формат есть.")],
    )

    assert result["totals"]["strict_f3_draft_candidates"] == 0
    assert result["totals"]["blocked_no_exact_proof"] == 1
    row = result["groups"]["blocked_no_exact_proof"]["examples"][0]
    assert row["evidence_level"] == "unknown"


def test_frame_must_be_safe_self_before_candidate(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[
            _dialog(
                "d1",
                route="draft_for_manager",
                frame={
                    "risk_class": "manager_action",
                    "answerability": "manager_only",
                    "requested_action": "check_availability",
                    "must_handoff": True,
                    "confidence": 0.94,
                },
            )
        ],
        facts=[_fact()],
    )

    assert result["totals"]["strict_f3_draft_candidates"] == 0
    assert result["totals"]["blocked_frame_not_self"] == 1
    row = result["groups"]["blocked_frame_not_self"]["examples"][0]
    assert "frame_risk_not_safe" in row["blocked_reasons"]
    assert "frame_must_handoff" in row["blocked_reasons"]


def test_missing_frame_risk_or_answerability_blocks_strict_candidate(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[
            _dialog(
                "d1",
                route="draft_for_manager",
                frame={
                    "risk_class": "",
                    "answerability": "",
                    "requested_action": "answer_question",
                    "must_handoff": False,
                    "confidence": 0.95,
                },
            )
        ],
        facts=[_fact()],
    )

    assert result["totals"]["strict_f3_draft_candidates"] == 0
    assert result["totals"]["blocked_frame_not_self"] == 1
    row = result["groups"]["blocked_frame_not_self"]["examples"][0]
    assert "frame_risk_not_safe" in row["blocked_reasons"]
    assert "frame_answerability_not_self" in row["blocked_reasons"]


def test_danger_money_p0_is_excluded_even_with_exact_proof(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[
            _dialog(
                "p0_paid_transfer_case",
                route="draft_for_manager",
                safety_flags=["manager_approval_required", "p0_model_led"],
            )
        ],
        facts=[_fact()],
    )

    assert result["totals"]["strict_f3_draft_candidates"] == 0
    assert result["totals"]["excluded_danger_money_p0"] == 1


def test_money_question_is_not_a_strict_candidate_without_p0_flag(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[
            _dialog(
                "d1",
                route="draft_for_manager",
                client_message="Как оплатить летнюю школу?",
                frame={
                    "requested_product": {
                        "brand": "unpk",
                        "subject": "",
                        "grade": "5 класс",
                        "format": "",
                        "program_kind": "летняя школа",
                        "raw_text": "летняя школа 5 класс",
                    }
                },
            )
        ],
        facts=[_fact(text="ЛВШ УНПК для 5 класса есть.")],
    )

    assert result["totals"]["strict_f3_draft_candidates"] == 0
    assert result["totals"]["existence_format_rows"] == 0


def test_not_offered_exact_proof_can_be_safe_candidate(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[
            _dialog(
                "d1",
                route="draft_for_manager",
                client_message="Химия в УНПК есть?",
                frame={
                    "requested_product": {
                        "brand": "unpk",
                        "subject": "химия",
                        "grade": "",
                        "format": "",
                        "program_kind": "",
                        "raw_text": "химия",
                    }
                },
            )
        ],
        facts=[_negative_fact()],
    )

    assert result["totals"]["strict_f3_draft_candidates"] == 1
    row = result["groups"]["strict_f3_draft_candidate"]["examples"][0]
    assert row["product_existence_check"]["status"] == "not_offered"


def test_report_does_not_truncate_after_fifty_existence_rows(tmp_path: Path) -> None:
    dialogs = [_dialog(f"d{i}", route="draft_for_manager") for i in range(55)]

    result = _build(tmp_path, dialogs=dialogs, facts=[_fact()])

    assert result["totals"]["strict_f3_draft_candidates"] == 55
