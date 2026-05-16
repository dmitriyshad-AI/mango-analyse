from __future__ import annotations

import csv
from pathlib import Path

from mango_mvp.deal_aware.deal_state_classifier import (
    DealStatePaths,
    build_deal_state_classifier,
    classify_call_policy,
    classify_deal_state,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_active_deal_allows_full_stage4_generation() -> None:
    state = classify_deal_state(
        {
            "lead_id": "100",
            "status_name": "В работе",
            "pipeline_name": "Сделки B2C",
            "loss_reason": "",
        }
    )
    row = classify_call_policy(
        {
            "call_id": "c1",
            "selected_deal_id": "100",
            "attribution_decision": "linked_single_deal_candidate",
            "confidence_score": "0.82",
        },
        deal_by_id={"100": {"lead_id": "100", "status_name": "В работе"}},
        deal_state_by_id={"100": state},
        task_meta_by_deal={},
    )

    assert row["stage3_decision"] == "allow_stage4_full_active_deal_generation"
    assert row["deal_writeback_mode"] == "full_active"
    assert row["safe_for_stage4_generation"] == "Да"
    assert row["safe_for_live_deal_writeback_now"] == "Нет"


def test_paid_deal_allows_only_context_mode_and_flags_payment_next_step() -> None:
    state = classify_deal_state({"lead_id": "100", "status_name": "Оплата получена"})
    row = classify_call_policy(
        {
            "call_id": "c1",
            "selected_deal_id": "100",
            "attribution_decision": "linked_single_deal_candidate",
            "confidence_score": "0.90",
            "call_next_step": "Отправить ссылку на оплату",
        },
        deal_by_id={"100": {"lead_id": "100", "status_name": "Оплата получена"}},
        deal_state_by_id={"100": state},
        task_meta_by_deal={},
    )

    assert row["stage3_decision"] == "allow_stage4_context_only_paid_deal_generation"
    assert row["deal_writeback_mode"] == "context_only_paid_or_success"
    assert "paid_deal_has_payment_next_step_in_call" in row["stage3_risk_flags"]


def test_closed_existing_client_requires_manual_redirect_not_auto_write() -> None:
    state = classify_deal_state(
        {
            "lead_id": "100",
            "status_name": "Закрыто и не реализовано",
            "loss_reason": "Действующий клиент",
            "closed_at": "2026-05-01T10:00:00+00:00",
        }
    )

    assert state["deal_state_class"] == "closed_existing_client"
    assert state["deal_state_bucket"] == "manual_review"
    assert state["deal_state_policy"] == "manual_review_existing_client_redirect"


def test_closed_noise_deal_is_blocked() -> None:
    state = classify_deal_state(
        {
            "lead_id": "100",
            "status_name": "Закрыто и не реализовано",
            "loss_reason": "Не оставлял заявку",
            "closed_at": "2026-05-01T10:00:00+00:00",
        }
    )

    assert state["deal_state_class"] == "closed_noise_or_wrong_request"
    assert state["deal_state_bucket"] == "blocked"


def test_low_confidence_link_requires_manual_review() -> None:
    state = classify_deal_state({"lead_id": "100", "status_name": "В работе"})
    row = classify_call_policy(
        {
            "call_id": "c1",
            "selected_deal_id": "100",
            "attribution_decision": "linked_single_deal_candidate",
            "confidence_score": "0.60",
        },
        deal_by_id={"100": {"lead_id": "100", "status_name": "В работе"}},
        deal_state_by_id={"100": state},
        task_meta_by_deal={},
    )

    assert row["stage3_decision"] == "manual_review_low_stage2_confidence"
    assert row["safe_for_stage4_generation"] == "Нет"


def test_stage2_terminal_deal_gets_specific_stage3_closed_policy() -> None:
    state = classify_deal_state(
        {
            "lead_id": "100",
            "status_name": "Закрыто и не реализовано",
            "loss_reason": "Действующий клиент",
            "closed_at": "2026-05-01T10:00:00+00:00",
        }
    )
    row = classify_call_policy(
        {
            "call_id": "c1",
            "selected_deal_id": "100",
            "attribution_decision": "manual_review_single_terminal_deal_candidate",
            "confidence_score": "0.85",
        },
        deal_by_id={
            "100": {
                "lead_id": "100",
                "status_name": "Закрыто и не реализовано",
                "loss_reason": "Действующий клиент",
            }
        },
        deal_state_by_id={"100": state},
        task_meta_by_deal={},
    )

    assert row["stage3_bucket"] == "manual_review"
    assert row["stage3_decision"] == "manual_review_existing_client_redirect"
    assert row["deal_state_class"] == "closed_existing_client"


def test_stage2_terminal_won_deal_allows_context_only_generation() -> None:
    state = classify_deal_state({"lead_id": "100", "status_name": "Успешно"})
    row = classify_call_policy(
        {
            "call_id": "c1",
            "selected_deal_id": "100",
            "attribution_decision": "manual_review_single_terminal_deal_candidate",
            "confidence_score": "0.85",
        },
        deal_by_id={"100": {"lead_id": "100", "status_name": "Успешно"}},
        deal_state_by_id={"100": state},
        task_meta_by_deal={},
    )

    assert row["stage3_bucket"] == "allow"
    assert row["stage3_decision"] == "allow_stage4_context_only_paid_deal_generation"
    assert row["deal_writeback_mode"] == "context_only_paid_or_success"
    assert row["safe_for_stage4_generation"] == "Да"


def test_build_stage3_outputs_candidates_and_fail_closed_summary(tmp_path: Path) -> None:
    stage2 = tmp_path / "stage2"
    live = tmp_path / "live"
    out = tmp_path / "out"
    _write_csv(
        stage2 / "deal_call_links.csv",
        [
            {
                "call_id": "1",
                "selected_deal_id": "100",
                "attribution_decision": "linked_single_deal_candidate",
                "confidence_score": "0.85",
            },
            {
                "call_id": "2",
                "selected_deal_id": "101",
                "attribution_decision": "linked_single_deal_candidate",
                "confidence_score": "0.85",
            },
            {
                "call_id": "3",
                "selected_deal_id": "",
                "attribution_decision": "manual_review_no_deal_candidate",
                "confidence_score": "0",
            },
        ],
    )
    (stage2 / "summary.json").write_text('{"coverage": {"linked_rows": 2}}', encoding="utf-8")
    _write_csv(
        live / "amo_deals_snapshot.csv",
        [
            {"lead_id": "100", "status_name": "В работе"},
            {"lead_id": "101", "status_name": "Закрыто и не реализовано", "loss_reason": "Спам"},
        ],
    )
    _write_csv(
        live / "amo_tasks_snapshot.csv",
        [
            {
                "task_id": "t1",
                "entity_id": "100",
                "entity_type": "leads",
                "is_completed": "",
                "complete_till": "2026-01-01T00:00:00+00:00",
                "text": "Перезвонить",
            }
        ],
    )
    (live / "summary.json").write_text(
        '{"connection": {"before": {"connected": true}, "after": {"connected": true}}, "fetch": {"contacts_seen": 1, "leads_seen": 2, "tasks_seen": 1}}',
        encoding="utf-8",
    )

    summary = build_deal_state_classifier(
        DealStatePaths(stage2_attribution_root=stage2, amo_live_snapshot_root=live, out_root=out)
    )

    assert summary["safety"]["write_amo"] is False
    assert summary["coverage"]["stage4_generation_candidates"] == 1
    assert summary["coverage"]["stage4_deal_candidates"] == 1
    assert summary["coverage"]["blocked_rows"] == 1
    assert summary["coverage"]["manual_review_rows"] == 1
    assert summary["readiness"]["safe_to_write_deal_fields"] is False
    assert (out / "deal_stage4_generation_candidates.csv").exists()
    assert (out / "deal_stage4_deal_candidates.csv").exists()
    assert (out / "deal_aware_stage3_deal_state.sqlite").exists()
