from __future__ import annotations

import json
from pathlib import Path


FIXTURE = Path("tests/fixtures/deal_aware_adversarial_cases.jsonl")


def test_deal_aware_adversarial_fixture_has_core_classes() -> None:
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 50
    assert len({row["case_id"] for row in rows}) == len(rows)

    classes = {row["class_id"] for row in rows}
    required = {
        "paid_context_only_suppresses_payment",
        "completed_payment_evidence_active_deal",
        "customer_side_payment_action",
        "true_service_feedback",
        "terminal_lost_reason_active_next_step",
        "existing_client_loss_reason",
        "no_tallanto_match",
        "multiple_tallanto_matches",
        "tenant_asr_terms",
        "protected_field_in_payload",
    }
    assert required <= classes

    for row in rows:
        assert row["case_id"].startswith("da-")
        assert row["layer"] in {"stage2_selector", "stage4_payload", "stage5_gate", "stage6_preflight", "readback", "batch_selection"}
        assert "input" in row
        assert "expected" in row
