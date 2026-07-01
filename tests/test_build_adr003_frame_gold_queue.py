import json
from pathlib import Path

from scripts import build_adr003_frame_gold_queue as queue


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _frame(*, must_handoff: bool) -> dict:
    return {
        "intent": "payment_forward",
        "risk_class": "safe",
        "deal_stage": "closing",
        "payment_readiness": "ready_to_pay",
        "requested_product": {"brand": "foton", "raw_text": "курс"},
        "requested_action": "send_payment_link",
        "answerability": "answerable",
        "must_handoff": must_handoff,
        "evidence": ["клиент просит ссылку на оплату"],
        "confidence": 0.86,
    }


def _dialog(turn: dict) -> dict:
    return {"dialog_id": "d1", "brand": "foton", "turns": [turn]}


def test_queue_includes_frame_handoff_current_self_mismatch(tmp_path: Path) -> None:
    transcripts = tmp_path / "transcripts.jsonl"
    _write_jsonl(
        transcripts,
        [
            _dialog(
                {
                    "turn": 1,
                    "client_message": "Можете прислать ссылку на оплату?",
                    "bot_route": "bot_answer_self_for_pilot",
                    "bot_text": "Можно оплатить по ссылке.",
                    "bot_safety_flags": ["no_auto_send"],
                    "bot_manager_checklist": [],
                    "bot_semantic_frame": _frame(must_handoff=True),
                }
            )
        ],
    )

    result = queue.build_queue(transcripts=transcripts)

    assert result["summary"]["queue_rows"] == 1
    assert result["summary"]["mismatch_types"] == {
        "frame_handoff_current_self+frame_handoff_no_p0_signal": 1
    }
    row = result["rows"][0]
    assert row["needs_gold_label"] is True
    assert row["mismatch_type"] == "frame_handoff_current_self+frame_handoff_no_p0_signal"
    assert row["current_route_handoff"] is False
    assert row["current_p0_signal"] is False
    assert row["frame_must_handoff"] is True


def test_queue_includes_frame_self_current_handoff_mismatch(tmp_path: Path) -> None:
    transcripts = tmp_path / "transcripts.jsonl"
    _write_jsonl(
        transcripts,
        [
            _dialog(
                {
                    "turn": 1,
                    "client_message": "Хочу вернуть деньги за прошлый курс.",
                    "bot_route": "manager_only",
                    "bot_text": "Передам менеджеру.",
                    "bot_safety_flags": ["refund"],
                    "bot_manager_checklist": ["Проверить оплату."],
                    "bot_semantic_frame": _frame(must_handoff=False),
                }
            )
        ],
    )

    result = queue.build_queue(transcripts=transcripts)

    assert result["summary"]["queue_rows"] == 1
    assert result["summary"]["mismatch_types"] == {
        "frame_self_current_handoff+frame_self_current_p0_signal": 1
    }
    row = result["rows"][0]
    assert row["current_route_handoff"] is True
    assert row["current_p0_signal"] is True
    assert row["frame_must_handoff"] is False


def test_queue_excludes_matches_by_default_and_can_include_them(tmp_path: Path) -> None:
    transcripts = tmp_path / "transcripts.jsonl"
    turn = {
        "turn": 1,
        "client_message": "Какая цена курса?",
        "bot_route": "bot_answer_self_for_pilot",
        "bot_text": "Стоимость зависит от формата.",
        "bot_safety_flags": ["manager_approval_required", "no_auto_send"],
        "bot_manager_checklist": [],
        "bot_semantic_frame": _frame(must_handoff=False),
    }
    _write_jsonl(transcripts, [_dialog(turn)])

    default_result = queue.build_queue(transcripts=transcripts)
    include_result = queue.build_queue(transcripts=transcripts, include_matches=True)

    assert default_result["summary"]["queue_rows"] == 0
    assert default_result["summary"]["framed_turns"] == 1
    assert include_result["summary"]["queue_rows"] == 1
    assert include_result["rows"][0]["needs_gold_label"] is False
    assert include_result["rows"][0]["mismatch_type"] == "match"


def test_queue_marks_no_frame_input_status(tmp_path: Path) -> None:
    transcripts = tmp_path / "transcripts.jsonl"
    _write_jsonl(
        transcripts,
        [
            _dialog(
                {
                    "turn": 1,
                    "client_message": "Какая цена курса?",
                    "bot_route": "bot_answer_self_for_pilot",
                    "bot_text": "Стоимость зависит от формата.",
                    "bot_safety_flags": ["no_auto_send"],
                    "bot_manager_checklist": [],
                }
            )
        ],
    )

    result = queue.build_queue(transcripts=transcripts)

    assert result["summary"]["input_status"] == "no_frame"
    assert result["summary"]["queue_rows"] == 0
    assert result["summary"]["missing_frame_turns"] == 1


def test_queue_marks_invalid_must_handoff(tmp_path: Path) -> None:
    transcripts = tmp_path / "transcripts.jsonl"
    bad_frame = _frame(must_handoff=False)
    bad_frame["must_handoff"] = "false"
    _write_jsonl(
        transcripts,
        [
            _dialog(
                {
                    "turn": 1,
                    "client_message": "Какая цена курса?",
                    "bot_route": "bot_answer_self_for_pilot",
                    "bot_text": "Стоимость зависит от формата.",
                    "bot_safety_flags": ["no_auto_send"],
                    "bot_manager_checklist": [],
                    "bot_semantic_frame": bad_frame,
                }
            )
        ],
    )

    result = queue.build_queue(transcripts=transcripts)

    assert result["summary"]["input_status"] == "invalid_frame"
    assert result["summary"]["invalid_frame_turns"] == 1
    assert result["rows"][0]["mismatch_type"] == "invalid_frame_must_handoff"
    assert result["rows"][0]["frame_must_handoff"] is None


def test_queue_marks_pii_risk_but_keeps_artifact_local(tmp_path: Path) -> None:
    transcripts = tmp_path / "transcripts.jsonl"
    _write_jsonl(
        transcripts,
        [
            _dialog(
                {
                    "turn": 1,
                    "client_message": "Мой телефон +7 999 111-22-33, почта test@example.com",
                    "bot_route": "bot_answer_self_for_pilot",
                    "bot_text": "Передам менеджеру.",
                    "bot_safety_flags": ["no_auto_send"],
                    "bot_manager_checklist": [],
                    "bot_semantic_frame": _frame(must_handoff=True),
                }
            )
        ],
    )

    result = queue.build_queue(transcripts=transcripts)

    assert result["summary"]["pii_risk"] is True
    assert result["summary"]["pii_risk_rows"] == 1
    assert result["rows"][0]["pii_risk"] is True


def test_queue_cli_writes_outputs(tmp_path: Path) -> None:
    transcripts = tmp_path / "transcripts.jsonl"
    out_dir = tmp_path / "out"
    _write_jsonl(
        transcripts,
        [
            _dialog(
                {
                    "turn": 1,
                    "client_message": "Хочу вернуть деньги.",
                    "bot_route": "manager_only",
                    "bot_text": "Передам менеджеру.",
                    "bot_safety_flags": ["refund"],
                    "bot_manager_checklist": [],
                    "bot_semantic_frame": _frame(must_handoff=False),
                }
            )
        ],
    )

    assert queue.main(["--transcripts", str(transcripts), "--out-dir", str(out_dir)]) == 0

    summary = json.loads((out_dir / "adr003_frame_gold_queue_summary.json").read_text(encoding="utf-8"))
    markdown = (out_dir / "adr003_frame_gold_queue_summary.md").read_text(encoding="utf-8")
    csv_text = (out_dir / "adr003_frame_gold_queue.csv").read_text(encoding="utf-8")
    assert summary["queue_rows"] == 1
    assert "frame_self_current_handoff+frame_self_current_p0_signal" in markdown
    assert "dialog_id,brand,turn,mismatch_type" in csv_text
