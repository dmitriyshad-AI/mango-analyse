import json
from pathlib import Path

from scripts import report_adr003_frame_gold_calibration as report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _frame(*, must_handoff: bool, risk_class: str = "safe", action: str = "answer_question", confidence: float = 0.91) -> dict:
    return {
        "intent": "pricing_question",
        "risk_class": risk_class,
        "deal_stage": "interest",
        "payment_readiness": "asking_price",
        "requested_product": {"brand": "foton", "raw_text": "курс"},
        "requested_action": action,
        "answerability": "answer_self" if not must_handoff else "manager_only",
        "must_handoff": must_handoff,
        "evidence": ["test"],
        "confidence": confidence,
    }


def _dialog(*, dialog_id: str, turn: dict) -> dict:
    return {"dialog_id": dialog_id, "brand": "foton", "turns": [turn]}


def test_gold_calibration_counts_too_cautious_and_confidence_buckets(tmp_path: Path) -> None:
    transcripts = tmp_path / "transcripts.jsonl"
    gold = tmp_path / "gold.jsonl"
    _write_jsonl(
        transcripts,
        [
            _dialog(
                dialog_id="safe_price",
                turn={
                    "turn": 1,
                    "client_message": "Сколько стоит курс?",
                    "bot_route": "draft_for_manager",
                    "bot_text": "Стоимость 100 ₽.",
                    "bot_safety_flags": ["manager_approval_required"],
                    "bot_manager_checklist": [],
                    "bot_semantic_frame": _frame(must_handoff=True, risk_class="manager_action", confidence=0.91),
                },
            ),
            _dialog(
                dialog_id="receipt",
                turn={
                    "turn": 1,
                    "client_message": "Отправил чек.",
                    "bot_route": "bot_answer_self_for_pilot",
                    "bot_text": "Спасибо.",
                    "bot_safety_flags": ["no_auto_send"],
                    "bot_manager_checklist": [],
                    "bot_semantic_frame": _frame(must_handoff=False, confidence=0.82),
                },
            ),
        ],
    )
    _write_jsonl(
        gold,
        [
            {
                "schema_version": report.GOLD_SCHEMA_VERSION,
                "dialog_id": "safe_price",
                "turn": 1,
                "expected": {
                    "must_handoff": False,
                    "risk_class": "safe",
                    "answerability": "answer_self",
                    "requested_action": "answer_question",
                },
            },
            {
                "schema_version": report.GOLD_SCHEMA_VERSION,
                "dialog_id": "receipt",
                "turn": 1,
                "expected": {
                    "must_handoff": True,
                    "risk_class": "manager_action",
                    "answerability": "manager_only",
                    "requested_action": "handoff_manager",
                },
            },
        ],
    )

    result = report.build_report(transcripts=transcripts, gold=gold)

    summary = result["summary"]
    assert summary["compared_rows"] == 2
    assert summary["too_cautious"] == 1
    assert summary["too_confident"] == 1
    assert summary["current_over_handoff_candidates"] == 1
    assert summary["must_handoff_accuracy"] == 0.0
    assert summary["confidence_buckets"]["0.90-1.00"]["too_cautious"] == 1
    assert result["acceptance"]["status"] == "blocked_for_active"


def test_gold_calibration_cli_writes_report(tmp_path: Path) -> None:
    transcripts = tmp_path / "transcripts.jsonl"
    gold = tmp_path / "gold.jsonl"
    out = tmp_path / "out"
    _write_jsonl(
        transcripts,
        [
            _dialog(
                dialog_id="address",
                turn={
                    "turn": 1,
                    "client_message": "Где занятия?",
                    "bot_route": "bot_answer_self_for_pilot",
                    "bot_text": "На Красносельской.",
                    "bot_safety_flags": ["no_auto_send"],
                    "bot_manager_checklist": [],
                    "bot_semantic_frame": _frame(must_handoff=False, confidence=0.86),
                },
            )
        ],
    )
    _write_jsonl(
        gold,
        [
            {
                "schema_version": report.GOLD_SCHEMA_VERSION,
                "dialog_id": "address",
                "turn": 1,
                "expected": {
                    "must_handoff": False,
                    "risk_class": "safe",
                    "answerability": "answer_self",
                    "requested_action": "answer_question",
                },
            }
        ],
    )

    assert report.main(["--transcripts", str(transcripts), "--gold", str(gold), "--out-dir", str(out)]) == 0
    saved = json.loads((out / "adr003_frame_gold_calibration_report.json").read_text(encoding="utf-8"))
    assert saved["summary"]["compared_rows"] == 1
    assert "ADR-003 Frame Gold Calibration" in (out / "adr003_frame_gold_calibration_report.md").read_text(encoding="utf-8")
