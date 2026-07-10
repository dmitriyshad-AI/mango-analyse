from __future__ import annotations

import json
from pathlib import Path

from scripts import report_adr003_existence_fact_verification as report


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
    route: str = "draft_for_manager",
    brand: str = "unpk",
    client_message: str = "Олимпиадная физика онлайн для 9 класса есть?",
    requested_product: dict | None = None,
    requested_action: str = "check_availability",
    safety_flags: list[str] | None = None,
) -> dict:
    return {
        "dialog_id": dialog_id,
        "brand": brand,
        "turns": [
            {
                "turn": 1,
                "brand": brand,
                "client_message": client_message,
                "bot_text": "Менеджер поможет проверить подходящий вариант.",
                "bot_route": route,
                "bot_safety_flags": safety_flags or ["manager_approval_required", "no_auto_send"],
                "bot_semantic_frame": {
                    "risk_class": "manager_action",
                    "answerability": "manager_only",
                    "requested_action": requested_action,
                    "deal_stage": "qualification",
                    "payment_readiness": "none",
                    "must_handoff": True,
                    "confidence": 0.94,
                    "requested_product": requested_product
                    or {
                        "brand": brand,
                        "subject": "физика",
                        "grade": "9",
                        "format": "онлайн",
                        "program_kind": "олимпиадная подготовка",
                        "raw_text": "Олимпиадная физика онлайн для 9 класса",
                    },
                },
                "bot_semantic_frame_self_answer_shadow": {
                    "guards": {
                        "actual_p0": False,
                        "has_missing_facts": True,
                        "blocking_flags": [],
                        "freshness": {"ok": False, "exact_fact_count": 0, "fresh_client_safe_count": 0},
                    }
                },
                "bot_missing_facts": ["programs.current"],
            }
        ],
    }


def _fact(
    *,
    brand: str = "unpk",
    fact_key: str = "schedule_2026_27.physics_9_olympiad_online",
    text: str = "Физика, 9 класс, олимпиадная группа, онлайн, старт 19.09.2026.",
    allowed: bool = True,
    valid_until: str = "2027-05-30",
) -> dict:
    return {
        "brand": brand,
        "fact_key": fact_key,
        "fact_type": "deadline",
        "product": "schedule_2026_27",
        "client_safe_text": text,
        "allowed_for_client_answer": allowed,
        "forbidden_for_client": False,
        "internal_only": False,
        "valid_until": valid_until,
    }


def _build(tmp_path: Path, *, dialogs: list[dict], gold_rows: list[dict], facts: list[dict]) -> dict:
    transcripts = tmp_path / "transcripts.jsonl"
    gold = tmp_path / "gold.jsonl"
    kb = tmp_path / "kb.json"
    _write_jsonl(transcripts, dialogs)
    _write_jsonl(gold, gold_rows)
    _write_json(kb, {"facts": facts})
    return report.build_report(transcripts=transcripts, gold=gold, kb_snapshot=kb)


def test_handoff_with_exact_kb_evidence_is_reported(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[_dialog("d1")],
        gold_rows=[_gold("d1")],
        facts=[_fact()],
    )

    assert result["totals"]["handoff_with_exact_kb_evidence"] == 1
    example = result["groups"]["handoff_with_exact_kb_evidence"]["examples"][0]
    assert example["evidence_level"] == "kb_exact"
    assert example["best_kb_match"]["axis_hits"] == ["subject", "grade", "format", "program_kind"]


def test_handoff_without_exact_evidence_stays_blocked(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[_dialog("d1")],
        gold_rows=[_gold("d1")],
        facts=[_fact(text="Физика, онлайн-формат есть.", fact_key="format.physics.online")],
    )

    assert result["totals"]["handoff_without_exact_kb_evidence"] == 1
    example = result["groups"]["handoff_without_exact_kb_evidence"]["examples"][0]
    assert example["evidence_level"] == "unknown"
    assert example["product_existence_check"]["status"] == "unknown"


def test_wrong_brand_fact_is_not_evidence(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[_dialog("d1", brand="unpk")],
        gold_rows=[_gold("d1")],
        facts=[_fact(brand="foton")],
    )

    assert result["groups"]["handoff_without_exact_kb_evidence"]["count"] == 1
    example = result["groups"]["handoff_without_exact_kb_evidence"]["examples"][0]
    assert example["evidence_level"] == "unknown"
    assert example["product_existence_check"]["status"] == "unknown"


def test_stale_or_not_client_safe_fact_is_not_evidence(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[_dialog("d1")],
        gold_rows=[_gold("d1")],
        facts=[
            _fact(fact_key="stale", valid_until="2024-01-01"),
            _fact(fact_key="internal", allowed=False),
        ],
    )

    assert result["groups"]["handoff_without_exact_kb_evidence"]["count"] == 1
    example = result["groups"]["handoff_without_exact_kb_evidence"]["examples"][0]
    assert example["evidence_level"] == "unknown"
    assert example["product_existence_check"]["status"] == "unknown"


def test_payment_deadline_fact_is_not_product_existence_evidence(tmp_path: Path) -> None:
    payment_fact = _fact(
        fact_key="payment.deadline",
        text="После записи срок оплаты: для выездной школы — 5 дней.",
    )

    result = _build(
        tmp_path,
        dialogs=[
            _dialog(
                "d1",
                requested_product={
                    "brand": "unpk",
                    "subject": "",
                    "grade": "",
                    "format": "",
                    "program_kind": "лагерь",
                    "product_family": "летняя школа",
                    "raw_text": "летняя школа",
                },
            )
        ],
        gold_rows=[_gold("d1")],
        facts=[payment_fact],
    )

    assert result["groups"]["handoff_without_exact_kb_evidence"]["count"] == 1
    example = result["groups"]["handoff_without_exact_kb_evidence"]["examples"][0]
    assert example["evidence_level"] == "unknown"
    assert example["product_existence_check"]["status"] == "unknown"


def test_registration_fact_is_not_product_existence_evidence(tmp_path: Path) -> None:
    registration_fact = _fact(
        fact_key="registration.open",
        text="Регистрация на летнюю школу для 5 класса открыта.",
    )

    result = _build(
        tmp_path,
        dialogs=[
            _dialog(
                "d1",
                requested_product={
                    "brand": "unpk",
                    "subject": "",
                    "grade": "5 класс",
                    "format": "",
                    "program_kind": "лагерь",
                    "product_family": "летняя школа",
                    "raw_text": "летняя школа для 5 класса",
                },
            )
        ],
        gold_rows=[_gold("d1")],
        facts=[registration_fact],
    )

    assert result["groups"]["handoff_without_exact_kb_evidence"]["count"] == 1
    example = result["groups"]["handoff_without_exact_kb_evidence"]["examples"][0]
    assert example["evidence_level"] == "unknown"
    assert example["product_existence_check"]["status"] == "unknown"


def test_danger_money_p0_rows_are_excluded(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[
            _dialog(
                "p0_paid_transfer_case",
                safety_flags=["manager_approval_required", "p0_model_led"],
            )
        ],
        gold_rows=[_gold("p0_paid_transfer_case", notes="safe reference: camp existence")],
        facts=[_fact()],
    )

    assert result["totals"]["excluded_danger_money_p0"] == 1
    assert result["groups"]["excluded_danger_money_p0"]["examples"][0]["evidence_level"] == "kb_exact"


def test_already_self_with_evidence_is_separate_from_handoff(tmp_path: Path) -> None:
    result = _build(
        tmp_path,
        dialogs=[_dialog("d1", route="bot_answer_self_for_pilot")],
        gold_rows=[_gold("d1")],
        facts=[_fact()],
    )

    assert result["totals"]["already_self_with_exact_kb_evidence"] == 1
    assert result["totals"]["handoff_with_exact_kb_evidence"] == 0
