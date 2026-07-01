import json
from pathlib import Path

from scripts import report_adr003_semantic_frame_eval as report


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
    assert result["acceptance"]["flags"]["input_turns_match"] is False


def test_report_flags_route_text_diff(tmp_path: Path) -> None:
    off_transcripts = tmp_path / "off.jsonl"
    on_transcripts = tmp_path / "on.jsonl"
    _write_jsonl(off_transcripts, [_dialog(text="Менеджер проверит наличие места.", include_frame=False)])
    _write_jsonl(on_transcripts, [_dialog(text="Да, место есть.", include_frame=True)])

    result = report.build_report(on_transcripts=on_transcripts, off_transcripts=off_transcripts)

    assert result["acceptance"]["status"] == "needs_review"
    assert result["off_on_diff"]["route_text_diff_count"] == 1
    assert result["off_on_diff"]["diff_examples"][0]["changed"]["bot_text"]["off"] == "Менеджер проверит наличие места."
    assert result["off_on_diff"]["diff_examples"][0]["changed"]["bot_text"]["on"] == "Да, место есть."


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
    assert json_report["acceptance"]["flags"]["route_text_diff_zero"] is False
    assert json_report["acceptance"]["flags"]["extra_model_calls_expected"] is False
    assert json_report["semantic_frame"]["present_count"] == 1
    assert "OFF transcripts were not provided" in markdown
