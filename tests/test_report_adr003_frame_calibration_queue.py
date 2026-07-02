from __future__ import annotations

import json
from pathlib import Path

from scripts import report_adr003_frame_calibration_queue as report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _gold(
    dialog_id: str,
    *,
    must_handoff: bool = False,
    requested_action: str = "answer_question",
    notes: str = "safe reference: course existence/format without live seats",
) -> dict:
    return {
        "dialog_id": dialog_id,
        "turn": 1,
        "expected": {
            "must_handoff": must_handoff,
            "risk_class": "safe" if not must_handoff else "payment_dispute",
            "answerability": "answer_self" if not must_handoff else "manager_only",
            "requested_action": requested_action,
        },
        "review_label": "frame_too_cautious" if not must_handoff else "frame_too_confident",
        "notes": notes,
    }


def _turn(
    dialog_id: str,
    *,
    route: str = "manager_only",
    message_type: str = "context_update",
    missing_facts: list[str] | None = None,
    frame: dict | None = None,
    proof_reconciliation: dict | None = None,
    is_manager_deferral: bool = False,
    answer_quality_findings: list[str] | None = None,
    semantic_output_verifier: dict | None = None,
    authoritative_output_gate: dict | None = None,
    safety_flags: list[str] | None = None,
) -> dict:
    semantic_frame = {
        "risk_class": "manager_action",
        "answerability": "manager_only",
        "requested_action": "check_availability",
        "must_handoff": True,
        "confidence": 0.92,
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
        "dialog_id": dialog_id,
        "brand": "unpk",
        "turns": [
            {
                "turn": 1,
                "brand": "unpk",
                "client_message": "Ребёнок закончил 5 класс",
                "bot_text": "Передам менеджеру.",
                "bot_route": route,
                "bot_message_type": message_type,
                "bot_safety_flags": safety_flags
                if safety_flags is not None
                else ["manager_approval_required", "no_auto_send", f"message_type_{message_type}"],
                "bot_missing_facts": missing_facts if missing_facts is not None else ["актуальное наличие мест"],
                "bot_semantic_frame": semantic_frame,
                "bot_semantic_frame_proof_reconciliation_shadow": proof_reconciliation or {},
                "bot_semantic_frame_self_answer_shadow": {
                    "status": "blocked",
                    "reason": "route_not_draft_for_manager",
                    "self_class": "safe_reference",
                    "guards": {
                        "actual_p0": False,
                        "has_missing_facts": bool(missing_facts),
                        "blocking_flags": [],
                        "freshness": {"ok": False, "exact_fact_count": 0, "fresh_client_safe_count": 0},
                    },
                },
                "bot_is_manager_deferral": is_manager_deferral,
                "bot_answer_quality_findings": answer_quality_findings or [],
                "bot_authoritative_output_gate": authoritative_output_gate
                if authoritative_output_gate is not None
                else {"action": "pass", "checked": True, "findings": []},
                "bot_semantic_output_verifier": semantic_output_verifier or {},
            }
        ],
    }


def _fact() -> dict:
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
        "valid_until": "2026-08-31",
        "structured_value": {"classes_raw": "5-10", "raw_value": "5-10", "valid_until": "2026-08-31"},
    }


def _online_fact(**overrides: object) -> dict:
    fact = {
        "brand": "unpk",
        "fact_key": "online_platform.levels.1",
        "fact_type": "course_parameter",
        "product": "online_platform",
        "program_kind": "online",
        "client_safe_text": "УНПК: онлайн-занятия проходят на платформе SohoLMS.",
        "allowed_for_client_answer": True,
        "forbidden_for_client": False,
        "internal_only": False,
        "valid_until": "2026-08-31",
        "structured_value": {"raw_value": "SohoLMS", "valid_until": "2026-08-31"},
    }
    fact.update(overrides)
    return fact


def _template(fact_key: str, **overrides: object) -> dict:
    template = {
        "brand": "unpk",
        "client_send": False,
        "fact_id": f"fact:v3:unpk:{fact_key.replace('.', '_')}:test",
        "fact_key": fact_key,
        "fact_type": "course_parameter",
        "fallback_route": "manager_only",
        "template_id": "template:course_parameter:contextual_answer_v1",
        "template_text": "Собрать человеческую фразу из structured_value и контекста вопроса.",
    }
    template.update(overrides)
    return template


def _build(
    tmp_path: Path,
    dialogs: list[dict],
    gold_rows: list[dict],
    *,
    facts: list[dict] | None = None,
    templates: list[dict] | None = None,
) -> dict:
    transcripts = tmp_path / "transcripts.jsonl"
    gold = tmp_path / "gold.jsonl"
    kb = tmp_path / "kb.json"
    template_registry = tmp_path / "bot_template_registry.json"
    _write_jsonl(transcripts, dialogs)
    _write_jsonl(gold, gold_rows)
    _write_json(kb, {"facts": facts or [_fact(), _online_fact()]})
    if templates is not None:
        _write_json(template_registry, {"schema_version": "bot_template_registry_v1", "templates": templates})
    return report.build_report(
        transcripts=transcripts,
        gold=gold,
        kb_snapshot=kb,
        bot_template_registry=template_registry if templates is not None else None,
        as_of_date=report._parse_date("2026-07-02"),
    )


def test_existence_vs_availability_goes_to_frame_calibration_not_active(tmp_path: Path) -> None:
    result = _build(tmp_path, [_turn("d1")], [_gold("d1")])

    assert result["totals"]["true_frame_too_cautious"] == 1
    assert result["real_lever_analysis"]["totals"]["too_cautious_total"] == 1
    assert result["real_lever_analysis"]["totals"]["fact_assertion_required"] == 1
    assert result["real_lever_analysis"]["totals"]["clean_route_only_discussion"] == 0
    assert result["real_lever_analysis"]["totals"]["stable_existence_as_check_availability"] == 1
    assert result["real_lever_analysis"]["totals"]["stable_existence_as_enroll"] == 0
    assert result["real_lever_analysis"]["scope_confusion"]["count"] == 1
    assert result["real_lever_analysis"]["by_frame_requested_action"] == {"check_availability": 1}
    assert result["real_lever_analysis"]["by_lever_class"] == {"fact_assertion_required": 1}
    example = result["real_lever_analysis"]["examples"][0]
    assert example["user_scope"] == "stable_existence_format"
    assert example["frame_scope"] == "live_availability_or_enroll"
    assert example["scope_confusion"] is True
    assert result["workstreams"]["semanticframe_existence_vs_availability"]["count"] == 1
    assert result["workstreams"]["retrieval_delivery_runtime_missing_exact_proof"]["count"] == 1
    assert result["workstreams"]["conversation_plan_scope_missing"]["count"] == 1
    assert result["workstreams"]["policy_manager_only_exact_proof"]["count"] == 1
    assert result["workstreams"]["policy_context_update_exact_proof"]["count"] == 1
    assert result["acceptance"]["active_readiness"] == "no_go"
    item = result["workstreams"]["semanticframe_existence_vs_availability"]["examples"][0]
    assert item["active_allowed"] is False
    assert item["active_block_reason"] == "frame_confuses_safe_reference_with_live_availability_or_enrollment"
    assert "requested_action" in item["calibration_target"]


def test_factless_ack_status_is_reported_separately(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "ack",
                route="draft_for_manager",
                message_type="context_update",
                missing_facts=[],
                frame={
                    "risk_class": "safe",
                    "answerability": "answer_self",
                    "requested_action": "acknowledge_status",
                    "must_handoff": True,
                    "confidence": 0.94,
                },
            )
        ],
        [_gold("ack", notes="harmless ack/status, no fact assertion")],
    )

    real = result["real_lever_analysis"]
    assert real["totals"]["too_cautious_total"] == 1
    assert real["totals"]["factless_ack_status"] == 1
    assert real["totals"]["fact_assertion_required"] == 0
    assert real["totals"]["clean_route_only_discussion"] == 1
    assert real["by_lever_class"] == {"clean_factless_ack_status_discussion": 1}


def test_true_enroll_booking_request_is_negative_control(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "book",
                route="manager_only",
                message_type="question",
                frame={
                    "risk_class": "manager_action",
                    "answerability": "manager_only",
                    "requested_action": "enroll",
                    "must_handoff": True,
                    "confidence": 0.96,
                },
            )
        ],
        [_gold("book", must_handoff=True, requested_action="enroll", notes="true enrollment/booking request")],
    )

    real = result["real_lever_analysis"]
    assert real["totals"]["too_cautious_total"] == 0
    assert real["totals"]["true_live_availability_negative_controls"] == 1
    assert real["totals"]["true_enroll_booking_negative_controls"] == 1
    negative = real["negative_controls"][0]
    assert negative["expected_action"] == "enroll"
    assert negative["expected_scope"] == "live_availability_or_enroll"
    assert negative["active_block_reason"] == "negative_control_true_live_or_operational_request"


def test_manual_too_cautious_label_is_separate_from_true_frame_error(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                route="bot_answer_self_for_pilot",
                message_type="question",
                missing_facts=[],
                frame={
                    "risk_class": "safe",
                    "answerability": "answer_self",
                    "requested_action": "answer_question",
                    "must_handoff": False,
                    "confidence": 0.95,
                },
            )
        ],
        [_gold("d1", notes="safe reference: price/format")],
    )

    assert result["totals"]["manual_too_cautious_labels"] == 1
    assert result["totals"]["true_frame_too_cautious"] == 0
    assert result["real_lever_analysis"]["totals"]["too_cautious_total"] == 0
    assert result["workstreams"]["already_self_no_active_leverage"]["count"] == 1
    item = result["workstreams"]["already_self_no_active_leverage"]["examples"][0]
    assert item["active_allowed"] is False
    assert item["active_block_reason"] == "current_route_already_self_no_route_leverage"


def test_proof_reconciliation_shadow_is_reported_but_not_active_go(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                route="draft_for_manager",
                message_type="question",
                missing_facts=["platform.current"],
                frame={
                    "risk_class": "missing_facts",
                    "answerability": "manager_only",
                    "requested_action": "answer_question",
                    "must_handoff": True,
                    "confidence": 0.92,
                },
                proof_reconciliation={
                    "status": "would_reconcile_to_safe_reference",
                    "reason": "fresh_proof_contradicts_missing_facts_frame",
                    "route_before": "draft_for_manager",
                    "exact_fact_keys": ["online_platform.levels.1"],
                    "frame_before": {
                        "risk_class": "missing_facts",
                        "answerability": "manager_only",
                        "requested_action": "answer_question",
                        "must_handoff": True,
                    },
                },
            )
        ],
        [_gold("d1", notes="safe reference: platform/format without live seats")],
    )

    assert result["totals"]["proof_reconciliation_would_reconcile"] == 1
    assert result["real_lever_analysis"]["totals"]["proof_reconciliation_would_reconcile"] == 1
    assert result["real_lever_analysis"]["totals"]["proof_reconciliation_current_handoff"] == 1
    assert result["workstreams"]["semanticframe_proof_reconciliation_candidate"]["count"] == 1
    assert result["proof_reconciliation"]["would_reconcile"] == 1
    assert result["proof_reconciliation"]["examples"][0]["exact_fact_keys"] == ["online_platform.levels.1"]
    text_readiness = result["proof_reconciliation_text_readiness"]
    assert text_readiness["total"] == 1
    assert text_readiness["send_as_is_review_candidates"] == 0
    assert text_readiness["by_blocker"]["missing_facts_present"] == 1
    assert text_readiness["by_blocker"]["semantic_verifier_unavailable"] == 1
    assert text_readiness["source_fact_lookup_by_status"] == {"found": 1}
    assert text_readiness["source_fact_client_safe_text_present"] == 1
    assert text_readiness["text_candidate_readiness_by_status"] == {"source_text_ready": 1}
    assert result["real_lever_analysis"]["totals"]["proof_reconciliation_text_blocked"] == 1
    assert result["real_lever_analysis"]["totals"]["proof_reconciliation_send_as_is_review_candidates"] == 0
    assert result["real_lever_analysis"]["totals"]["proof_text_source_fact_ready"] == 1
    assert result["acceptance"]["active_readiness"] == "no_go"


def test_proof_reconciliation_text_readiness_can_mark_manual_review_candidate(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "ready",
                route="draft_for_manager",
                message_type="question",
                missing_facts=[],
                frame={
                    "risk_class": "missing_facts",
                    "answerability": "manager_only",
                    "requested_action": "answer_question",
                    "must_handoff": True,
                    "confidence": 0.93,
                },
                proof_reconciliation={
                    "status": "would_reconcile_to_safe_reference",
                    "reason": "fresh_proof_contradicts_missing_facts_frame",
                    "route_before": "draft_for_manager",
                    "exact_fact_keys": ["online_platform.levels.1"],
                    "result_missing_facts": [],
                },
                semantic_output_verifier={"action": "pass", "findings": []},
            )
        ],
        [_gold("ready", notes="safe reference: platform/format without live seats")],
    )

    readiness = result["proof_reconciliation_text_readiness"]
    assert readiness["send_as_is_review_candidates"] == 1
    assert readiness["by_status"] == {"send_as_is_review_candidate": 1}
    assert readiness["source_fact_lookup_by_status"] == {"found": 1}
    assert readiness["text_candidate_readiness_by_status"] == {"source_text_ready": 1}
    row = readiness["examples"][0]
    assert row["source_fact_client_safe_text_present"] is True
    assert row["source_fact_client_safe_text_hash"]
    assert row["source_fact_client_safe_text_length"] > 0
    assert row["structured_value_available"] is True
    assert row["structured_value_keys"] == ["raw_value", "valid_until"]
    assert row["raw_text_exported"] is False
    assert row["direct_quote_forbidden"] is True
    assert row["text_policy_readiness_status"] == "source_text_ready_requires_nonquote_policy"
    assert "direct_quote_forbidden" in row["text_policy_blockers"]
    assert "SohoLMS" not in json.dumps(row, ensure_ascii=False)
    assert row["text_candidate_blockers"] == [
        "shadow_only_text_candidate",
        "requires_template_or_text_policy",
    ]
    assert readiness["active_behavior_allowed"] is False
    assert result["real_lever_analysis"]["totals"]["proof_reconciliation_send_as_is_review_candidates"] == 1
    assert result["acceptance"]["active_readiness"] == "no_go"


def test_proof_text_policy_reports_template_registry_without_exporting_template_text(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "template-ready",
                route="draft_for_manager",
                message_type="question",
                missing_facts=[],
                proof_reconciliation={
                    "status": "would_reconcile_to_safe_reference",
                    "reason": "fresh_proof_contradicts_missing_facts_frame",
                    "route_before": "draft_for_manager",
                    "exact_fact_keys": ["template.fact"],
                    "result_missing_facts": [],
                },
                semantic_output_verifier={"action": "pass", "findings": []},
            )
        ],
        [_gold("template-ready", notes="safe reference: platform/format without live seats")],
        facts=[_online_fact(fact_key="template.fact", bot_template_required=True)],
        templates=[_template("template.fact")],
    )

    readiness = result["proof_reconciliation_text_readiness"]
    assert readiness["template_registry_by_status"] == {"found": 1}
    assert readiness["text_policy_readiness_by_status"] == {"template_registry_found_requires_renderer": 1}
    assert readiness["template_registry_found"] == 1
    assert readiness["direct_quote_forbidden"] == 1
    row = readiness["examples"][0]
    assert row["template_registry_status"] == "found"
    assert row["template_id"] == "template:course_parameter:contextual_answer_v1"
    assert row["template_text_length"] > 0
    assert row["template_text_hash"]
    assert "template_text" not in row
    assert "client_safe_text" not in row
    assert "Собрать человеческую фразу" not in json.dumps(row, ensure_ascii=False)
    assert row["raw_text_exported"] is False
    assert "requires_template_renderer" in row["text_policy_blockers"]
    assert result["acceptance"]["active_readiness"] == "no_go"


def test_proof_text_source_readiness_ignores_fallback_fact_text(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "fallback",
                route="draft_for_manager",
                message_type="question",
                missing_facts=[],
                frame={
                    "risk_class": "missing_facts",
                    "answerability": "manager_only",
                    "requested_action": "answer_question",
                    "must_handoff": True,
                    "confidence": 0.93,
                },
                proof_reconciliation={
                    "status": "would_reconcile_to_safe_reference",
                    "reason": "fresh_proof_contradicts_missing_facts_frame",
                    "route_before": "draft_for_manager",
                    "exact_fact_keys": ["online_platform.levels.1"],
                    "result_missing_facts": [],
                },
                semantic_output_verifier={"action": "pass", "findings": []},
            )
        ],
        [_gold("fallback", notes="safe reference: platform/format without live seats")],
        facts=[_online_fact(client_safe_text="", fact_text="Fallback text must not count as client-safe source.")],
    )

    readiness = result["proof_reconciliation_text_readiness"]
    assert readiness["source_fact_lookup_by_status"] == {"empty_client_safe_text": 1}
    assert readiness["text_candidate_readiness_by_status"] == {"blocked_empty_client_safe_text": 1}
    row = readiness["examples"][0]
    assert row["source_fact_client_safe_text_present"] is False
    assert row["source_fact_client_safe_text_hash"] == ""
    assert "empty_client_safe_text" in row["text_candidate_blockers"]
    assert row["raw_text_exported"] is False
    assert result["acceptance"]["active_readiness"] == "no_go"


def test_proof_text_source_readiness_blocks_wrong_brand_template_and_pii(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "brand",
                route="draft_for_manager",
                message_type="question",
                missing_facts=[],
                proof_reconciliation={
                    "status": "would_reconcile_to_safe_reference",
                    "reason": "fresh_proof_contradicts_missing_facts_frame",
                    "route_before": "draft_for_manager",
                    "exact_fact_keys": ["online_platform.levels.1"],
                    "result_missing_facts": [],
                },
                semantic_output_verifier={"action": "pass", "findings": []},
            ),
            _turn(
                "template",
                route="draft_for_manager",
                message_type="question",
                missing_facts=[],
                proof_reconciliation={
                    "status": "would_reconcile_to_safe_reference",
                    "reason": "fresh_proof_contradicts_missing_facts_frame",
                    "route_before": "draft_for_manager",
                    "exact_fact_keys": ["template.fact"],
                    "result_missing_facts": [],
                },
                semantic_output_verifier={"action": "pass", "findings": []},
            ),
            _turn(
                "pii",
                route="draft_for_manager",
                message_type="question",
                missing_facts=[],
                proof_reconciliation={
                    "status": "would_reconcile_to_safe_reference",
                    "reason": "fresh_proof_contradicts_missing_facts_frame",
                    "route_before": "draft_for_manager",
                    "exact_fact_keys": ["pii.fact"],
                    "result_missing_facts": [],
                },
                semantic_output_verifier={"action": "pass", "findings": []},
            ),
        ],
        [
            _gold("brand", notes="safe reference: platform/format without live seats"),
            _gold("template", notes="safe reference: platform/format without live seats"),
            _gold("pii", notes="safe reference: platform/format without live seats"),
        ],
        facts=[
            _online_fact(brand="foton"),
            _online_fact(fact_key="template.fact", bot_template_required=True),
            _online_fact(fact_key="pii.fact", client_safe_text="Пишите на test@example.com"),
        ],
    )

    readiness = result["proof_reconciliation_text_readiness"]
    assert readiness["source_fact_lookup_by_status"] == {"wrong_brand": 1, "found": 2}
    assert readiness["source_fact_client_safe_text_present"] == 3
    assert readiness["source_fact_client_safe_text_pii_signal"] == 1
    assert readiness["bot_template_required"] == 1
    assert readiness["template_registry_by_status"] == {"not_required": 2, "missing": 1}
    assert readiness["text_policy_readiness_by_status"] == {
        "blocked_wrong_brand": 1,
        "blocked_missing_bot_template": 1,
        "blocked_client_safe_text_pii_signal": 1,
    }
    by_turn = readiness["by_turn"]
    assert by_turn["brand#1"]["text_candidate_readiness_status"] == "blocked_wrong_brand"
    assert "bot_template_required" in by_turn["template#1"]["text_candidate_blockers"]
    assert by_turn["template#1"]["template_registry_status"] == "missing"
    assert by_turn["template#1"]["text_policy_readiness_status"] == "blocked_missing_bot_template"
    assert "client_safe_text_pii_signal" in by_turn["pii#1"]["text_candidate_blockers"]
    assert by_turn["pii#1"]["raw_text_exported"] is False
    assert result["acceptance"]["active_readiness"] == "no_go"


def test_proof_reconciliation_text_readiness_blocks_deferral_and_quality_findings(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "blocked",
                route="draft_for_manager",
                message_type="question",
                missing_facts=[],
                frame={
                    "risk_class": "missing_facts",
                    "answerability": "manager_only",
                    "requested_action": "answer_question",
                    "must_handoff": True,
                    "confidence": 0.93,
                },
                proof_reconciliation={
                    "status": "would_reconcile_to_safe_reference",
                    "reason": "fresh_proof_contradicts_missing_facts_frame",
                    "route_before": "draft_for_manager",
                    "exact_fact_keys": ["online_platform.levels.1"],
                    "result_missing_facts": [],
                },
                is_manager_deferral=True,
                answer_quality_findings=["rewrite_locked_high_risk_or_manager_only"],
                semantic_output_verifier={"action": "pass", "findings": []},
            )
        ],
        [_gold("blocked", notes="safe reference: platform/format without live seats")],
    )

    readiness = result["proof_reconciliation_text_readiness"]
    assert readiness["send_as_is_review_candidates"] == 0
    assert readiness["by_blocker"]["deferral_or_manager_text_signal"] == 1
    assert readiness["by_blocker"]["answer_quality_findings_present"] == 1
    row = readiness["examples"][0]
    assert row["active_behavior_allowed"] is False
    assert result["real_lever_analysis"]["totals"]["proof_reconciliation_text_blocked"] == 1


def test_low_confidence_safe_frame_is_calibration_work(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "d1",
                message_type="question",
                missing_facts=[],
                frame={
                    "risk_class": "safe",
                    "answerability": "answer_self",
                    "requested_action": "answer_question",
                    "must_handoff": False,
                    "confidence": 0.86,
                },
            )
        ],
        [_gold("d1", notes="safe reference: course existence/format without live seats")],
    )

    assert result["totals"]["true_frame_too_cautious"] == 0
    assert result["workstreams"]["semanticframe_low_confidence"]["count"] == 1
    assert result["workstreams"]["policy_manager_only_exact_proof"]["count"] == 1


def test_too_confident_handoff_gold_is_measurement_review(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        [
            _turn(
                "p0",
                route="manager_only",
                frame={
                    "risk_class": "safe",
                    "answerability": "answer_self",
                    "requested_action": "answer_question",
                    "must_handoff": False,
                    "confidence": 0.95,
                },
            )
        ],
        [_gold("p0", must_handoff=True, notes="payment dispute")],
    )

    assert result["totals"]["true_frame_too_confident"] == 1
    assert result["workstreams"]["measurement_review_unclear"]["count"] == 1
