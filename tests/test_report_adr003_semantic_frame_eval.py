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


def _summary(total_calls: int = 3, *, frame_calls: int = 0) -> dict:
    return {"llm_calls": {"total": total_calls, "bot_semantic_frame_shadow": frame_calls}, "hard_gate_failure_dialogs": []}


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
