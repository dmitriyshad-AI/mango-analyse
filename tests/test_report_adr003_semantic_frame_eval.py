import json
from pathlib import Path

from scripts import report_adr003_semantic_frame_eval as report
from scripts.run_telegram_dynamic_client_sim import load_dynamic_sim_input


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _frame(*, must_handoff: bool = True) -> dict:
    return {
        "intent": "live_availability",
        "risk_class": "manager_action",
        "deal_stage": "closing",
        "payment_readiness": "considering",
        "requested_product": {"brand": "foton", "raw_text": "курс"},
        "requested_action": "check_availability",
        "answerability": "manager_only",
        "must_handoff": must_handoff,
        "evidence": ["клиент просит проверить наличие места"],
        "confidence": 0.91,
    }


def _dialog(*, text: str = "Менеджер проверит наличие места.", include_frame: bool = True) -> dict:
    turn = {
        "turn": 1,
        "client_message": "Есть места?",
        "bot_route": "draft_for_manager",
        "bot_text": text,
        "bot_safety_flags": ["manager_approval_required"],
        "bot_manager_checklist": ["Проверить наличие места."],
    }
    if include_frame:
        turn["bot_semantic_frame"] = _frame()
        turn["bot_frame_decision_shadow"] = {
            "status": "ok",
            "comparisons": {
                "must_handoff_vs_route": "match",
                "p0_vs_actual": "mismatch",
                "action": {"status": "aligned"},
            },
        }
    return {"dialog_id": "d1", "brand": "foton", "turns": [turn]}


def _preblocked_p0_turn() -> dict:
    return {
        "turn": 2,
        "client_message": "Если места нет, возвращайте деньги.",
        "bot_route": "manager_only",
        "bot_text": "Позову менеджера.",
        "bot_safety_flags": [
            "manager_approval_required",
            "no_auto_send",
            "direct_path_preblocked_p0",
        ],
        "bot_direct_path": {
            "model_called": False,
            "preblocked": True,
            "preblock_reason": "p0_pre_gate",
            "text_composition_source": "deterministic_preblock",
        },
    }


def _timeout_turn() -> dict:
    return {
        "turn": 3,
        "client_message": "Есть выездная школа для 9 класса?",
        "bot_route": "manager_only",
        "bot_text": "Позову менеджера.",
        "bot_safety_flags": ["manager_approval_required", "no_auto_send"],
        "bot_direct_path": {
            "model_called": True,
            "preblocked": False,
            "preblock_reason": "",
            "reason_evidence": {"provider_error": "timeout"},
            "text_composition_source": "provider_runtime_fallback",
        },
    }


def _summary(total_calls: int = 3, *, frame_calls: int = 0, **extra_calls: int) -> dict:
    calls = {"total": total_calls, "bot_semantic_frame_shadow": frame_calls}
    calls.update(extra_calls)
    return {"llm_calls": calls, "hard_gate_failure_dialogs": []}


def test_report_accepts_clean_off_on_pair(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off_summary = tmp_path / "off_summary.json"
    on_summary = tmp_path / "on_summary.json"
    _write_jsonl(off_transcripts, [_dialog(include_frame=False)])
    _write_jsonl(on_transcripts, [_dialog(include_frame=True)])
    off_summary.write_text(json.dumps(_summary(3)), encoding="utf-8")
    on_summary.write_text(json.dumps(_summary(3)), encoding="utf-8")

    result = report.build_report(
        on_transcripts=on_transcripts,
        on_summary=on_summary,
        off_transcripts=off_transcripts,
        off_summary=off_summary,
    )

    assert result["acceptance"]["status"] == "pass"
    assert result["off_on_diff"]["route_text_diff_count"] == 0
    assert result["llm_calls"]["extra_total"] == 0
    assert result["acceptance"]["flags"]["extra_model_calls_expected"] is True
    assert result["semantic_frame"]["present_count"] == 1
    assert result["semantic_frame"]["complete_required_count"] == 1
    assert result["frame_decision_shadow"]["turn_count"] == 1


def test_report_frame_emission_excludes_p0_preblock_and_timeout_from_denominator(tmp_path: Path) -> None:
    on_transcripts = tmp_path / "on.jsonl"
    dialog = _dialog(include_frame=True)
    dialog["turns"].append(_preblocked_p0_turn())
    dialog["turns"].append(_timeout_turn())
    _write_jsonl(on_transcripts, [dialog])

    result = report.build_report(on_transcripts=on_transcripts)

    frame = result["semantic_frame"]
    assert frame["turns_total"] == 3
    assert frame["present_count"] == 1
    assert frame["missing_count"] == 2
    assert frame["preblocked_p0_count"] == 1
    assert frame["provider_timeout_count"] == 1
    assert frame["infra_timeout_present"] is True
    assert frame["eligible_model_called_turns"] == 1
    assert frame["eligible_frame_count"] == 1
    assert frame["eligible_frame_rate"] == 1.0
    assert result["acceptance"]["flags"]["semantic_frame_eligible_rate_ok"] is True
    assert any("Provider timeout is present" in note for note in result["acceptance"]["notes"])


def test_report_does_not_treat_manager_approval_flag_as_route_handoff(tmp_path: Path) -> None:
    on_transcripts = tmp_path / "on.jsonl"
    dialog = _dialog(include_frame=True)
    turn = dialog["turns"][0]
    turn["bot_route"] = "bot_answer_self_for_pilot"
    turn["bot_safety_flags"] = ["manager_approval_required", "no_auto_send"]
    turn["bot_semantic_frame"] = _frame(must_handoff=False)
    _write_jsonl(on_transcripts, [dialog])

    result = report.build_report(on_transcripts=on_transcripts)

    assert result["semantic_frame"]["must_handoff_vs_route"] == {"match": 1}
    assert result["semantic_frame"]["must_handoff_vs_p0_signal"] == {"match": 1}


def test_report_rejects_non_bool_must_handoff(tmp_path: Path) -> None:
    on_transcripts = tmp_path / "on.jsonl"
    dialog = _dialog(include_frame=True)
    dialog["turns"][0]["bot_semantic_frame"]["must_handoff"] = "false"
    _write_jsonl(on_transcripts, [dialog])

    result = report.build_report(on_transcripts=on_transcripts)

    assert result["acceptance"]["status"] == "needs_review"
    assert result["semantic_frame"]["complete_required_count"] == 0
    assert result["semantic_frame"]["missing_required_fields"] == {"must_handoff:invalid_bool": 1}
    assert result["semantic_frame"]["must_handoff"] == {"invalid": 1}


def test_report_treats_string_false_model_p0_as_not_p0(tmp_path: Path) -> None:
    on_transcripts = tmp_path / "on.jsonl"
    dialog = _dialog(include_frame=True)
    turn = dialog["turns"][0]
    turn["bot_route"] = "bot_answer_self_for_pilot"
    turn["bot_safety_flags"] = ["no_auto_send"]
    turn["bot_direct_path_model_p0"] = {"is_p0": "false"}
    turn["bot_semantic_frame"] = _frame(must_handoff=False)
    _write_jsonl(on_transcripts, [dialog])

    result = report.build_report(on_transcripts=on_transcripts)

    assert result["semantic_frame"]["must_handoff_vs_p0_signal"] == {"match": 1}


def test_report_accepts_expected_posthoc_frame_call_delta(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off_summary = tmp_path / "off_summary.json"
    on_summary = tmp_path / "on_summary.json"
    _write_jsonl(off_transcripts, [_dialog(include_frame=False)])
    _write_jsonl(on_transcripts, [_dialog(include_frame=True)])
    off_summary.write_text(json.dumps(_summary(3, frame_calls=0)), encoding="utf-8")
    on_summary.write_text(json.dumps(_summary(4, frame_calls=1)), encoding="utf-8")

    result = report.build_report(
        on_transcripts=on_transcripts,
        on_summary=on_summary,
        off_transcripts=off_transcripts,
        off_summary=off_summary,
    )

    assert result["acceptance"]["status"] == "pass"
    assert result["llm_calls"]["extra_total"] == 1
    assert result["llm_calls"]["extra_semantic_frame_shadow"] == 1
    assert result["acceptance"]["flags"]["extra_model_calls_expected"] is True


def test_report_accepts_paired_semantic_frame_enrichment_calls(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off_summary = tmp_path / "off_summary.json"
    on_summary = tmp_path / "on_summary.json"
    _write_jsonl(off_transcripts, [_dialog(include_frame=False)])
    _write_jsonl(on_transcripts, [_dialog(include_frame=True)])
    off_summary.write_text(json.dumps(_summary(3, frame_calls=0)), encoding="utf-8")
    on_summary.write_text(
        json.dumps(
            {
                "semantic_frame_enriched": True,
                "llm_calls": {"total": 1, "bot_semantic_frame_shadow": 1},
                "hard_gate_failure_dialogs": [],
            }
        ),
        encoding="utf-8",
    )

    result = report.build_report(
        on_transcripts=on_transcripts,
        on_summary=on_summary,
        off_transcripts=off_transcripts,
        off_summary=off_summary,
    )

    assert result["acceptance"]["status"] == "pass"
    assert result["llm_calls"]["mode"] == "semantic_frame_enrichment"
    assert result["llm_calls"]["raw_total_delta"] == -2
    assert result["llm_calls"]["extra_total"] == 1
    assert result["llm_calls"]["extra_semantic_frame_shadow"] == 1
    assert result["acceptance"]["flags"]["extra_model_calls_expected"] is True


def test_report_rejects_paired_enrichment_with_non_frame_calls(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off_summary = tmp_path / "off_summary.json"
    on_summary = tmp_path / "on_summary.json"
    _write_jsonl(off_transcripts, [_dialog(include_frame=False)])
    _write_jsonl(on_transcripts, [_dialog(include_frame=True)])
    off_summary.write_text(json.dumps(_summary(3, frame_calls=0)), encoding="utf-8")
    on_summary.write_text(
        json.dumps(
            {
                "semantic_frame_enriched": True,
                "semantic_frame_enrichment": {"status": "all", "turns_total": 1, "enriched_turns": 1},
                "llm_calls": {"total": 2, "bot_semantic_frame_shadow": 1, "memory": 1},
                "hard_gate_failure_dialogs": [],
            }
        ),
        encoding="utf-8",
    )

    result = report.build_report(
        on_transcripts=on_transcripts,
        on_summary=on_summary,
        off_transcripts=off_transcripts,
        off_summary=off_summary,
    )

    assert result["acceptance"]["status"] == "needs_review"
    assert result["llm_calls"]["mode"] == "semantic_frame_enrichment"
    assert result["llm_calls"]["on_non_frame_total"] == 1
    assert result["acceptance"]["flags"]["extra_model_calls_expected"] is False


def test_report_rejects_partial_paired_enrichment(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off_summary = tmp_path / "off_summary.json"
    on_summary = tmp_path / "on_summary.json"
    _write_jsonl(off_transcripts, [_dialog(include_frame=False)])
    _write_jsonl(on_transcripts, [_dialog(include_frame=True)])
    off_summary.write_text(json.dumps(_summary(3, frame_calls=0)), encoding="utf-8")
    on_summary.write_text(
        json.dumps(
            {
                "semantic_frame_enrichment": {"status": "partial", "turns_total": 2, "enriched_turns": 1},
                "llm_calls": {"total": 1, "bot_semantic_frame_shadow": 1},
                "hard_gate_failure_dialogs": [],
            }
        ),
        encoding="utf-8",
    )

    result = report.build_report(
        on_transcripts=on_transcripts,
        on_summary=on_summary,
        off_transcripts=off_transcripts,
        off_summary=off_summary,
    )

    assert result["acceptance"]["status"] == "needs_review"
    assert result["llm_calls"]["mode"] == "semantic_frame_enrichment_partial"
    assert result["acceptance"]["flags"]["extra_model_calls_expected"] is False


def test_report_flags_input_diff_even_when_bot_output_matches(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off = _dialog(include_frame=False)
    on = _dialog(include_frame=True)
    on["turns"][0]["client_message"] = "Другой вопрос"
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    assert result["acceptance"]["status"] == "needs_review"
    assert result["off_on_diff"]["route_text_diff_count"] == 0
    assert result["off_on_diff"]["input_diff_count"] == 1
    assert "input_turns_match" not in result["acceptance"]["flags"]
    assert result["inline_text_health_gate"]["status"] == "pass"


def test_report_flags_route_text_diff(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    _write_jsonl(off_transcripts, [_dialog(text="Менеджер проверит наличие места.", include_frame=False)])
    _write_jsonl(on_transcripts, [_dialog(text="Да, место есть.", include_frame=True)])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    assert result["acceptance"]["status"] == "needs_review"
    assert result["off_on_diff"]["route_text_diff_count"] == 1
    assert result["inline_text_health_gate"]["status"] == "pass"
    assert result["off_on_diff"]["diff_examples"][0]["changed"]["bot_text"]["off"] == "Менеджер проверит наличие места."
    assert result["off_on_diff"]["diff_examples"][0]["changed"]["bot_text"]["on"] == "Да, место есть."


def test_inline_text_health_gate_blocks_manager_to_self_route_flip(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off_summary = tmp_path / "off_summary.json"
    on_summary = tmp_path / "on_summary.json"
    off = _dialog(include_frame=False)
    on = _dialog(text="Да, место есть.", include_frame=True)
    on["turns"][0]["bot_route"] = "bot_answer_self_for_pilot"
    on["turns"][0]["bot_safety_flags"] = ["no_auto_send"]
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])
    off_summary.write_text(json.dumps(_summary(3)), encoding="utf-8")
    on_summary.write_text(json.dumps(_summary(3)), encoding="utf-8")

    result = report.build_report(
        on_transcripts=on_transcripts,
        on_summary=on_summary,
        off_transcripts=off_transcripts,
        off_summary=off_summary,
    )

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "fail"
    assert gate["route_flip_dangerous_count"] == 1
    assert gate["p0_route_lost_count"] == 0
    assert result["acceptance"]["flags"]["inline_text_health_gate_ok"] is False


def test_inline_text_health_gate_blocks_p0_route_loss(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off = _dialog(include_frame=False)
    off["turns"][0]["bot_route"] = "manager_only"
    off["turns"][0]["bot_safety_flags"] = ["payment_dispute", "manager_approval_required"]
    on = _dialog(text="Ответим сами.", include_frame=True)
    on["turns"][0]["bot_route"] = "bot_answer_self_for_pilot"
    on["turns"][0]["bot_safety_flags"] = ["no_auto_send"]
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "fail"
    assert gate["p0_route_lost_count"] == 1
    assert gate["route_flip_dangerous_count"] == 1


def test_inline_text_health_gate_treats_p0_hygiene_flag_diff_as_warning(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off_summary = tmp_path / "off_summary.json"
    on_summary = tmp_path / "on_summary.json"
    off = _dialog(include_frame=False)
    off["turns"][0]["bot_route"] = "draft_for_manager"
    off["turns"][0]["bot_safety_flags"] = ["payment_dispute", "manager_approval_required"]
    on = _dialog(include_frame=True)
    on["turns"][0]["bot_route"] = "draft_for_manager"
    on["turns"][0]["bot_safety_flags"] = ["manager_approval_required"]
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])
    off_summary.write_text(json.dumps(_summary(3)), encoding="utf-8")
    on_summary.write_text(json.dumps(_summary(3)), encoding="utf-8")

    result = report.build_report(
        on_transcripts=on_transcripts,
        on_summary=on_summary,
        off_transcripts=off_transcripts,
        off_summary=off_summary,
    )

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "needs_review"
    assert gate["p0_route_lost_count"] == 0
    assert gate["p0_hygiene_flag_diff_count"] == 1
    assert gate["p0_hygiene_lost_count"] == 1
    assert gate["p0_hygiene_added_count"] == 0
    assert result["acceptance"]["flags"]["inline_text_health_gate_ok"] is False


def test_inline_text_health_gate_splits_p0_hygiene_added(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off = _dialog(include_frame=False)
    off["turns"][0]["bot_route"] = "draft_for_manager"
    off["turns"][0]["bot_safety_flags"] = ["manager_approval_required"]
    on = _dialog(include_frame=True)
    on["turns"][0]["bot_route"] = "manager_only"
    on["turns"][0]["bot_safety_flags"] = ["payment_dispute", "manager_approval_required"]
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "needs_review"
    assert gate["p0_hygiene_lost_count"] == 0
    assert gate["p0_hygiene_added_count"] == 1


def test_inline_text_health_gate_verifies_new_numbers_against_fact_text(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off_summary = tmp_path / "off_summary.json"
    on_summary = tmp_path / "on_summary.json"
    off = _dialog(text="Стоимость есть в карточке курса.", include_frame=False)
    on = _dialog(text="Стоимость курса — 9 000 ₽.", include_frame=True)
    on["turns"][0]["number_audit"] = {
        "items": [
            {
                "claim_text": "9 000 ₽",
                "normalized": "9000",
                "level": "retrieved_match",
            }
        ]
    }
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])
    off_summary.write_text(json.dumps(_summary(3)), encoding="utf-8")
    on_summary.write_text(json.dumps(_summary(3)), encoding="utf-8")

    result = report.build_report(
        on_transcripts=on_transcripts,
        on_summary=on_summary,
        off_transcripts=off_transcripts,
        off_summary=off_summary,
    )

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "pass"
    assert gate["new_number_verified_turn_count"] == 1
    assert gate["new_number_unverified_count"] == 0
    assert gate["number_verified_by_audit_count"] == 1
    assert result["acceptance"]["flags"]["inline_text_health_gate_ok"] is True


def test_inline_text_health_gate_verifies_new_numbers_against_exact_turn_fact(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off = _dialog(text="Старт указан в карточке курса.", include_frame=False)
    on = _dialog(text="Старт курса — в 2026 году.", include_frame=True)
    on["turns"][0]["bot_fact_retrieval_trace"] = {"selected_exact_ids": ["course.start.client_safe_text"]}
    on["turns"][0]["bot_direct_path"] = {
        "retrieved_facts": {
            "course.start.client_safe_text": "Курс стартует в 2026 году.",
            "other.fact": "Нерелевантная стоимость — 12 345 ₽.",
        }
    }
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "pass"
    assert gate["new_number_unverified_count"] == 0
    assert gate["number_verified_by_exact_fact_count"] == 1


def test_inline_text_health_gate_verifies_academic_year_from_exact_fact_id_only(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off = _dialog(text="Расписание опубликовано.", include_frame=False)
    on = _dialog(text="Расписание на 2026/27 год опубликовано.", include_frame=True)
    on["turns"][0]["bot_fact_retrieval_trace"] = {
        "selected_exact_ids": ["schedule_2026_27.groups.group_start_date.client_safe_text"],
        "selected_adjacent_ids": ["schedule_2027_28.groups.group_start_date.client_safe_text"],
    }
    on["turns"][0]["bot_direct_path"] = {
        "retrieved_facts": {
            "schedule_2026_27.groups.group_start_date.client_safe_text": "Расписание опубликовано.",
            "schedule_2027_28.groups.group_start_date.client_safe_text": "Соседний учебный год.",
        }
    }
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "pass"
    assert gate["new_number_unverified_count"] == 0
    assert gate["number_verified_by_exact_fact_count"] == 2


def test_inline_text_health_gate_blocks_unverified_new_numbers(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off = _dialog(text="Стоимость есть в карточке курса.", include_frame=False)
    on = _dialog(text="Стоимость курса — 12345 ₽.", include_frame=True)
    on["turns"][0]["bot_confirmed_facts"] = ["client_safe_text: Стоимость курса — 9 000 ₽."]
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "fail"
    assert gate["new_number_unverified_count"] == 1
    assert gate["new_number_unverified_examples"][0]["new_numbers"] == ["12345 ₽"]


def test_inline_text_health_gate_does_not_verify_number_from_raw_fact_blob_without_number_audit(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off = _dialog(text="Стоимость есть в карточке курса.", include_frame=False)
    on = _dialog(text="Стоимость курса — 9 000 ₽.", include_frame=True)
    on["turns"][0]["bot_confirmed_facts"] = ["internal_only_text: Стоимость курса — 9 000 ₽."]
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "fail"
    assert gate["new_number_unverified_count"] == 1


def test_inline_text_health_gate_reports_adjacent_fact_number_as_review_warning(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off = _dialog(text="Стоимость есть в карточке курса.", include_frame=False)
    on = _dialog(text="Стоимость соседнего курса — 12 345 ₽.", include_frame=True)
    on["turns"][0]["bot_fact_retrieval_trace"] = {"selected_adjacent_ids": ["adjacent.price.client_safe_text"]}
    on["turns"][0]["bot_direct_path"] = {
        "retrieved_facts": {
            "adjacent.price.client_safe_text": "Соседний курс стоит 12 345 ₽.",
        }
    }
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "needs_review"
    assert gate["new_number_unverified_count"] == 0
    assert gate["new_number_adjacent_warning_count"] == 1
    assert gate["number_adjacent_warning_count"] == 1


def test_inline_text_health_gate_verifies_number_from_client_history_separately(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off = _dialog(text="Хорошо, подберём курс.", include_frame=False)
    on = _dialog(text="Для 6 класса можно смотреть следующий уровень.", include_frame=True)
    off["turns"][0]["client_message"] = "Сын 6-й закончил."
    off["turns"].append(
        {
            "turn": 2,
            "client_message": "Что выбрать дальше?",
            "bot_route": "draft_for_manager",
            "bot_text": "Хорошо, подберём курс.",
            "bot_safety_flags": ["manager_approval_required"],
        }
    )
    on["turns"][0]["client_message"] = "Сын 6-й закончил."
    on["turns"].append(
        {
            "turn": 2,
            "client_message": "Что выбрать дальше?",
            "bot_route": "draft_for_manager",
            "bot_text": "Для 6 класса можно смотреть следующий уровень.",
            "bot_safety_flags": ["manager_approval_required"],
            "bot_semantic_frame": _frame(),
        }
    )
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "pass"
    assert gate["new_number_unverified_count"] == 0
    assert gate["number_verified_by_client_history_count"] == 1


def test_inline_text_health_gate_explains_missing_turns_by_timeout_dialog(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off = _dialog(include_frame=False)
    on = _dialog(include_frame=True)
    on["turns"].append(_timeout_turn())
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "pass"
    assert gate["missing_baseline_turns"] == 1
    assert gate["missing_baseline_explained_count"] == 1
    assert gate["missing_baseline_unexplained_count"] == 0


def test_inline_text_health_gate_marks_unexplained_missing_turns_needs_review(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    off = _dialog(include_frame=False)
    on = _dialog(include_frame=True)
    on["turns"].append(
        {
            "turn": 2,
            "client_message": "Ещё вопрос.",
            "bot_route": "draft_for_manager",
            "bot_text": "Менеджер проверит.",
            "bot_safety_flags": ["manager_approval_required"],
        }
    )
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(on_transcripts, [on])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    gate = result["inline_text_health_gate"]
    assert gate["status"] == "needs_review"
    assert gate["missing_baseline_unexplained_count"] == 1


def test_report_compares_inline_with_posthoc_and_text_health(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    inline_transcripts = tmp_path / "inline.jsonl"
    posthoc_transcripts = tmp_path / "posthoc.jsonl"
    off = _dialog(text="Менеджер проверит наличие места.", include_frame=False)
    inline = _dialog(text="Менеджер проверит наличие места.", include_frame=True)
    inline["turns"][0]["bot_semantic_frame"]["requested_action"] = "answer_question"
    posthoc = _dialog(text="Менеджер проверит наличие места.", include_frame=True)
    posthoc["turns"][0]["bot_semantic_frame"]["requested_action"] = "check_availability"
    _write_jsonl(off_transcripts, [off])
    _write_jsonl(inline_transcripts, [inline])
    _write_jsonl(posthoc_transcripts, [posthoc])

    result = report.build_report(
        on_transcripts=inline_transcripts,
        off_transcripts=off_transcripts,
        posthoc_transcripts=posthoc_transcripts,
    )

    agreement = result["inline_vs_posthoc_agreement"]
    assert agreement["compared_turns"] == 1
    assert agreement["mismatch_count"] == 1
    assert agreement["per_field"]["frame.requested_action"]["mismatch"] == 1
    assert result["baseline_vs_inline_text_health"]["dangerous_flip_count"] == 0


def test_report_includes_reader_agreement_for_pure_semantic_readers(tmp_path: Path) -> None:
    on_transcripts = tmp_path / "on.jsonl"
    dialog = _dialog(include_frame=True)
    turn = dialog["turns"][0]
    turn["client_message"] = "Есть ли места для 8 класса по физике онлайн?"
    turn["bot_direct_path"] = {
        "model_intent": {
            "primary_intent": "live_availability",
            "sense": "seats",
            "confidence": 0.91,
        },
    }
    turn["bot_semantic_frame"]["requested_product"] = {
        "grade": "8 класс",
        "subject": "физика",
        "format": "онлайн",
        "raw_text": "8 класс физика онлайн",
    }
    turn["bot_semantic_frame"]["confidence"] = 0.91
    _write_jsonl(on_transcripts, [dialog])

    result = report.build_report(on_transcripts=on_transcripts)

    agreement = result["reader_agreement"]
    assert agreement["status"] == "compared"
    assert agreement["compared_turns"] == 1
    assert agreement["per_reader"]["sense_seats"]["match"] == 1
    assert agreement["per_reader"]["slot_grade"]["match"] == 1
    assert agreement["per_reader"]["slot_subject"]["match"] == 1
    assert agreement["per_reader"]["slot_format"]["match"] == 1
    assert agreement["per_reader"]["off_topic"]["match"] == 1
    assert agreement["mismatch_count"] == 0


def test_report_summarizes_self_answer_shadow_candidates_and_unsafe(tmp_path: Path) -> None:
    on_transcripts = tmp_path / "on.jsonl"
    safe = _dialog(include_frame=True)
    safe_turn = safe["turns"][0]
    safe_turn["bot_semantic_frame_self_answer_shadow"] = {
        "status": "would_demote_to_self",
        "reason": "safe_answer_self_fresh_fact",
        "self_class": "price",
        "route_after_if_active": "bot_answer_self_for_pilot",
        "guards": {
            "freshness": {
                "ok": True,
                "exact_fact_count": 2,
                "fresh_client_safe_count": 1,
            }
        },
    }
    unsafe = _dialog(include_frame=True)
    unsafe["dialog_id"] = "d2"
    unsafe_turn = unsafe["turns"][0]
    unsafe_turn["bot_safety_flags"] = ["refund", "manager_approval_required"]
    unsafe_turn["bot_semantic_frame_self_answer_shadow"] = {
        "status": "would_demote_to_self",
        "reason": "safe_answer_self_fresh_fact",
        "self_class": "refund",
        "route_after_if_active": "bot_answer_self_for_pilot",
        "frame": {
            "deal_stage": "post_payment",
            "payment_readiness": "paid",
            "requested_action": "refund_or_cancel",
        },
        "guards": {"freshness": {"ok": False}},
    }
    _write_jsonl(on_transcripts, [safe, unsafe])

    result = report.build_report(on_transcripts=on_transcripts)

    shadow = result["semantic_frame_self_answer_shadow"]
    assert shadow["turn_count"] == 2
    assert shadow["would_demote_count"] == 2
    assert shadow["would_demote_by_class"] == {"price": 1, "refund": 1}
    assert shadow["p0_lowered_count"] == 1
    assert shadow["money_lowered_count"] == 1
    assert shadow["operational_lowered_count"] == 1
    assert shadow["freshness_unknown_self_candidates"] == 1
    assert shadow["partial_freshness_self_candidates"] == 1
    assert len(shadow["unsafe_candidate_examples"]) == 5
    assert result["acceptance"]["status"] == "needs_review"
    assert result["acceptance"]["flags"]["self_answer_partial_freshness_zero"] is False
    assert any("partial freshness" in note for note in result["acceptance"]["notes"])


def test_report_summarizes_proof_reconciliation_shadow_without_active_allow(tmp_path: Path) -> None:
    on_transcripts = tmp_path / "on.jsonl"
    dialog = _dialog(include_frame=True)
    turn = dialog["turns"][0]
    turn["bot_semantic_frame_proof_reconciliation_shadow"] = {
        "status": "would_reconcile_to_safe_reference",
        "reason": "fresh_proof_contradicts_missing_facts_frame",
        "proof_status": "exact_fresh_client_safe_fact_found",
        "active_behavior_allowed": False,
        "exact_fact_keys": ["online_platform.levels.1"],
        "active_blockers": ["shadow_only_reconciliation", "requires_text_readiness_policy"],
    }
    _write_jsonl(on_transcripts, [dialog])

    result = report.build_report(on_transcripts=on_transcripts)

    shadow = result["semantic_frame_proof_reconciliation_shadow"]
    assert shadow["turn_count"] == 1
    assert shadow["would_reconcile_count"] == 1
    assert shadow["active_allowed_count"] == 0
    assert shadow["status"] == {"would_reconcile_to_safe_reference": 1}
    assert shadow["reasons"] == {"fresh_proof_contradicts_missing_facts_frame": 1}
    assert shadow["proof_status"] == {"exact_fresh_client_safe_fact_found": 1}
    assert shadow["candidate_examples"][0]["exact_fact_keys"] == ["online_platform.levels.1"]
    assert shadow["candidate_examples"][0]["active_blockers"] == [
        "shadow_only_reconciliation",
        "requires_text_readiness_policy",
    ]


def test_report_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    on_transcripts = tmp_path / "on.jsonl"
    on_summary = tmp_path / "on_summary.json"
    out_dir = tmp_path / "out"
    _write_jsonl(on_transcripts, [_dialog(include_frame=True)])
    on_summary.write_text(json.dumps(_summary(3)), encoding="utf-8")

    assert report.main(["--on-transcripts", str(on_transcripts), "--on-summary", str(on_summary), "--out-dir", str(out_dir)]) == 0

    json_report = json.loads((out_dir / "adr003_semantic_frame_eval_report.json").read_text(encoding="utf-8"))
    markdown = (out_dir / "adr003_semantic_frame_eval_report.md").read_text(encoding="utf-8")
    assert json_report["acceptance"]["status"] == "needs_review"
    assert json_report["acceptance"]["flags"]["inline_text_health_gate_ok"] is False
    assert json_report["acceptance"]["flags"]["extra_model_calls_expected"] is False
    assert json_report["semantic_frame"]["present_count"] == 1
    assert "OFF transcripts were not provided" in markdown


def test_e2_semantic_reading_scenario_set_is_loadable_and_uses_soft_shadow_note() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scenarios = repo_root / "product_data/telegram_dynamic_test_sets/adr003_semantic_reading_paket1_e2_20260703.jsonl"

    sim_input = load_dynamic_sim_input(scenarios)
    text = scenarios.read_text(encoding="utf-8")

    assert sim_input.simulator_spec
    assert sim_input.judge_spec
    assert len(sim_input.personas) == 156
    assert sum(1 for persona in sim_input.personas if persona.get("source_set") == "fix1b_negative_personas_20260703") == 10
    assert "shadow_changed_behavior" not in text
    assert "shadow_behavior_note" in text
