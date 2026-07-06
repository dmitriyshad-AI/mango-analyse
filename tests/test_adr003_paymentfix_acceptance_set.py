from __future__ import annotations

import json
from pathlib import Path


PAYMENTFIX_SET = (
    Path(__file__).resolve().parents[1]
    / "product_data/telegram_dynamic_test_sets/adr003_acceptance_paymentfix_20260704.jsonl"
)


def _rows() -> list[dict[str, object]]:
    return [json.loads(line) for line in PAYMENTFIX_SET.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_adr003_paymentfix_acceptance_set_keeps_tail_16_regressions() -> None:
    personas = {str(row.get("dialog_id") or ""): row for row in _rows() if row.get("type") == "persona"}

    required = {
        "payfix_neg_foton_paid_no_access_01",
        "payfix_neg_unpk_paid_lesson_missing_01",
        "payfix_neg_foton_receipt_not_credited_01",
        "payfix_neg_unpk_refund_after_bad_access_01",
        "payfix_neg_foton_legal_refund_01",
        "payfix_neg_unpk_wrong_amount_01",
    }
    assert required <= set(personas)
    for dialog_id in required:
        row = personas[dialog_id]
        assert row["expected_route"] == "manager_only"
        assert row["injected_p0"] is True
        fail_criteria = str(row.get("fail_criteria") or "").casefold()
        success_criteria = str(row.get("success_criteria") or "").casefold()
        assert "self-answer" in fail_criteria
        assert "manager_only" in success_criteria or "менеджер" in success_criteria


def test_adr003_paymentfix_acceptance_judge_keeps_false_refund_hard_gate() -> None:
    judge = next(row for row in _rows() if row.get("type") == "judge_spec")
    gates = judge["hard_gates_fail_whole_dialog_if_violated"]
    assert "false_refund_text" in gates
    assert "p0_mishandled" in gates
