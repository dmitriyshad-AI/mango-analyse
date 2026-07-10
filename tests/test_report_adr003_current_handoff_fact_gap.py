from __future__ import annotations

import json
from pathlib import Path

from scripts import report_adr003_current_handoff_fact_gap as report


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _queue_row(
    dialog_id: str,
    *,
    route: str = "draft_for_manager",
    action: str = "answer_question",
    next_workstream: str = "fact_verification_or_retrieval_needed",
    source_alignment_status: str = "",
    source_fact_key: str = "",
    proof_reason: str = "no_exact_fact_keys",
) -> dict:
    return {
        "dialog_id": dialog_id,
        "turn": 1,
        "route": route,
        "requested_action": action,
        "next_autonomy_workstream": next_workstream,
        "frame_risk_class": "missing_facts",
        "frame_answerability": "manager_only",
        "frame_must_handoff": True,
        "frame_confidence": 0.88,
        "proof_reconciliation_status": "blocked",
        "proof_reconciliation_reason": proof_reason,
        "source_alignment_status": source_alignment_status,
        "source_fact_key": source_fact_key,
    }


def _queue(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "queue.json"
    _write_json(
        path,
        {
            "real_lever_analysis": {
                "current_handoff_queue": {
                    "examples": rows,
                }
            }
        },
    )
    return path


def _turn(
    dialog_id: str,
    *,
    route: str = "draft_for_manager",
    brand: str = "unpk",
    action: str = "answer_question",
    proof_status: str = "blocked",
    proof_reason: str = "no_exact_fact_keys",
    proof_source_fact_key: str = "",
    existence_reason: str = "no_exact_fact_keys",
    missing_facts: list[str] | None = None,
    requested_product: dict | None = None,
) -> dict:
    return {
        "dialog_id": dialog_id,
        "brand": brand,
        "turns": [
            {
                "turn": 1,
                "brand": brand,
                "client_message": "Вопрос клиента",
                "bot_text": "Передам менеджеру.",
                "bot_route": route,
                "bot_missing_facts": missing_facts or ["актуальный факт"],
                "bot_fact_retrieval_trace": {
                    "candidate_count": 0,
                    "required_fact_keys": [],
                    "selected_exact_ids": [],
                    "selected_adjacent_ids": [],
                    "mode": "off",
                },
                "bot_semantic_frame": {
                    "risk_class": "missing_facts",
                    "answerability": "manager_only",
                    "requested_action": action,
                    "must_handoff": True,
                    "confidence": 0.88,
                    "requested_product": requested_product
                    or {
                        "brand": brand,
                        "format": "",
                        "grade": "",
                        "program_kind": "летняя школа",
                        "raw_text": "летняя школа",
                        "subject": "",
                    },
                },
                "bot_semantic_frame_existence_proof_shadow": {
                    "status": "blocked",
                    "reason": existence_reason,
                },
                "bot_semantic_frame_proof_reconciliation_shadow": {
                    "status": proof_status,
                    "reason": proof_reason,
                    "proof_reason": existence_reason,
                    "source_fact_key": proof_source_fact_key,
                    "result_missing_facts": missing_facts or ["актуальный факт"],
                },
            }
        ],
    }


def _kb(tmp_path: Path, facts: list[dict]) -> Path:
    path = tmp_path / "kb.json"
    _write_json(path, {"facts": facts})
    return path


def _transcripts(tmp_path: Path, dialogs: list[dict]) -> Path:
    path = tmp_path / "transcripts.jsonl"
    _write_jsonl(path, dialogs)
    return path


def _fact(
    *,
    brand: str = "unpk",
    fact_key: str = "camp.classes",
    fact_type: str = "course_parameter",
    product: str = "camp",
    program_kind: str = "camp",
    text: str = "УНПК: ЛВШ рассчитана на 5-10 классы.",
    structured_value: dict | None = None,
) -> dict:
    return {
        "brand": brand,
        "fact_key": fact_key,
        "fact_id": f"fact:{brand}:{fact_key}",
        "fact_type": fact_type,
        "product": product,
        "program_kind": program_kind,
        "client_safe_text": text,
        "allowed_for_client_answer": True,
        "forbidden_for_client": False,
        "internal_only": False,
        "valid_until": "2026-12-31",
        "structured_value": structured_value or {"classes_raw": "5-10", "valid_until": "2026-12-31"},
    }


def _build(tmp_path: Path, *, rows: list[dict], dialogs: list[dict], facts: list[dict]) -> dict:
    return report.build_report(
        queue_report=_queue(tmp_path, rows),
        transcripts=_transcripts(tmp_path, dialogs),
        kb_snapshot=_kb(tmp_path, facts),
    )


def test_proof_axis_mismatch_blocks_fact_renderer(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        rows=[
            _queue_row(
                "d1",
                next_workstream="fix_proof_axis_alignment",
                source_alignment_status="blocked_source_axis_mismatch",
                source_fact_key="camp.classes",
                proof_reason="fresh_proof_contradicts_missing_facts_frame",
            )
        ],
        dialogs=[
            _turn(
                "d1",
                proof_status="would_reconcile_to_safe_reference",
                proof_reason="fresh_proof_contradicts_missing_facts_frame",
                proof_source_fact_key="camp.classes",
                missing_facts=["подтвержденное наличие питания"],
            )
        ],
        facts=[_fact(fact_key="camp.classes")],
    )

    case = result["cases"][0]
    assert case["root_cause"] == "proof_axis_mismatch"
    assert "boarding_food" in case["missing_fact_categories"]
    assert "class_grade" in case["source_fact_categories"]
    assert "boarding_food" in case["kb_support"]["uncovered_categories"]
    assert case["active_behavior_allowed"] is False
    assert result["totals"]["proof_axis_mismatch"] == 1
    assert result["acceptance"]["active_readiness"] == "no_go"


def test_frame_action_blocks_existence_proof_when_safe_age_question_is_check_availability(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        rows=[
            _queue_row(
                "age",
                route="manager_only",
                action="check_availability",
                next_workstream="fact_verification_or_retrieval_needed",
                proof_reason="requested_action_not_answer_question",
            )
        ],
        dialogs=[
            _turn(
                "age",
                route="manager_only",
                action="check_availability",
                proof_reason="requested_action_not_answer_question",
                existence_reason="requested_action_not_answer_question",
                missing_facts=["наличие программ для ребёнка после 5-го класса"],
                requested_product={
                    "brand": "unpk",
                    "format": "",
                    "grade": "закончил 5-й класс",
                    "program_kind": "летняя смена",
                    "raw_text": "мой 5-й закончил",
                    "subject": "",
                },
            )
        ],
        facts=[_fact(fact_key="camp.classes")],
    )

    case = result["cases"][0]
    assert case["root_cause"] == "frame_action_blocks_existence_proof"
    assert case["kb_support"]["product_check_status"] == "exists"
    assert case["recommended_next_step"].startswith("Calibrate SemanticFrame")
    assert result["totals"]["frame_action_blocks_proof"] == 1


def test_partial_price_platform_facts_are_reported_but_stay_no_go(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        rows=[
            _queue_row(
                "price",
                route="manager_only",
                action="answer_question",
                next_workstream="fact_verification_or_retrieval_needed",
                proof_reason="no_exact_fact_keys",
            )
        ],
        dialogs=[
            _turn(
                "price",
                route="manager_only",
                brand="foton",
                action="answer_question",
                proof_reason="no_exact_fact_keys",
                existence_reason="required_axis_missing",
                missing_facts=["platform.current", "класс ребёнка для точной цены"],
                requested_product={
                    "brand": "foton",
                    "format": "онлайн",
                    "grade": "",
                    "program_kind": "regular_course",
                    "raw_text": "онлайн, семестр",
                    "subject": "",
                },
            )
        ],
        facts=[
            _fact(
                brand="foton",
                fact_key="price.online.semester",
                fact_type="price",
                product="regular_course",
                program_kind="regular",
                text="Фотон: 5-11 класс, онлайн, семестр — 29 750 ₽.",
                structured_value={"valid_until": "2026-12-31"},
            ),
            _fact(
                brand="foton",
                fact_key="online_platform_transition",
                fact_type="process",
                product="regular_course",
                program_kind="regular",
                text="С лета 2026 онлайн-занятия проходят на платформе SohoLMS.",
                structured_value={"valid_until": "2026-12-31"},
            ),
        ],
    )

    case = result["cases"][0]
    assert case["root_cause"] == "partial_facts_available_but_slot_needed"
    assert case["kb_support"]["price_fact_count"] == 1
    assert case["kb_support"]["platform_fact_count"] == 1
    assert case["kb_support"]["proven_parts"] == ["price_cost", "platform_current"]
    assert case["kb_support"]["missing_slots"] == ["grade"]
    assert "partial-answer shadow" in case["recommended_next_step"]
    assert result["totals"]["partial_facts_slot_needed"] == 1
    dumped = json.dumps(case, ensure_ascii=False)
    assert "29 750" not in dumped
    assert "SohoLMS" not in dumped


def test_camp_price_is_not_support_for_online_regular_course_price(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        rows=[
            _queue_row(
                "price",
                route="manager_only",
                action="answer_question",
                next_workstream="fact_verification_or_retrieval_needed",
            )
        ],
        dialogs=[
            _turn(
                "price",
                route="manager_only",
                brand="foton",
                action="answer_question",
                proof_reason="no_exact_fact_keys",
                existence_reason="required_axis_missing",
                missing_facts=["класс ребёнка для точной цены"],
                requested_product={
                    "brand": "foton",
                    "format": "онлайн",
                    "grade": "",
                    "program_kind": "regular_course",
                    "raw_text": "онлайн, семестр",
                    "subject": "",
                },
            )
        ],
        facts=[
            _fact(
                brand="foton",
                fact_key="lvsh.price",
                fact_type="price",
                text="Фотон: ЛВШ Менделеево, текущая цена — 114 000 ₽.",
            )
        ],
    )

    case = result["cases"][0]
    assert case["kb_support"]["price_fact_count"] == 0
    assert case["root_cause"] == "required_axis_missing_no_exact_fact"


def test_known_grade_removes_grade_missing_slot_for_price(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        rows=[
            _queue_row(
                "price",
                route="manager_only",
                action="answer_question",
                next_workstream="fact_verification_or_retrieval_needed",
            )
        ],
        dialogs=[
            _turn(
                "price",
                route="manager_only",
                brand="foton",
                action="answer_question",
                proof_reason="no_exact_fact_keys",
                existence_reason="required_axis_missing",
                missing_facts=["класс ребёнка для точной цены"],
                requested_product={
                    "brand": "foton",
                    "format": "онлайн",
                    "grade": "5 класс",
                    "program_kind": "regular_course",
                    "raw_text": "5 класс онлайн, семестр",
                    "subject": "",
                },
            )
        ],
        facts=[
            _fact(
                brand="foton",
                fact_key="price.online.semester",
                fact_type="price",
                product="regular_course",
                program_kind="regular",
                text="Фотон: 5-11 класс, онлайн, семестр — 29 750 ₽.",
            )
        ],
    )

    case = result["cases"][0]
    assert case["kb_support"]["price_fact_count"] == 1
    assert case["kb_support"]["missing_slots"] == []
    assert case["active_behavior_allowed"] is False


def test_danger_adjacent_rows_are_kept_out_of_autonomy(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        rows=[
            _queue_row(
                "danger",
                next_workstream="danger_adjacent_do_not_lower",
            )
        ],
        dialogs=[_turn("danger")],
        facts=[_fact()],
    )

    case = result["cases"][0]
    assert case["root_cause"] == "danger_adjacent_do_not_lower"
    assert case["why_not_active"] == ["report_only", "danger_adjacent_do_not_lower", "missing_facts_present"]
    assert result["totals"]["danger_excluded"] == 1
