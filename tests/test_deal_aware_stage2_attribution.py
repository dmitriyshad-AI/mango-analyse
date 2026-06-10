from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import mango_mvp.deal_aware.deal_attribution as deal_attribution
from mango_mvp.deal_aware.deal_attribution import (
    AttributionPaths,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    attribute_call_to_deal,
    build_deal_attribution_dry_run,
    build_phone_deal_candidates,
    confidence_bucket,
    single_candidate_confidence,
)

FIXTURE = Path("tests/fixtures/deal_aware_adversarial_cases.jsonl")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_single_sales_call_links_to_single_candidate() -> None:
    candidates = [
        candidate(
            "100",
            status_name="Ожидание оплаты",
            candidate_sources="amo_live_linked_contact | phone_rollup | amo_ready",
            deal_updated_at="2026-05-10T10:00:00+00:00",
        )
    ]
    row = attribute_call_to_deal(
        {
            "call_id": "c1",
            "phone": "+7 916 111-22-33",
            "contentful": "Да",
            "call_type": "sales_call",
        },
        candidates,
    )

    assert row["attribution_decision"] == "linked_single_deal_candidate"
    assert row["selected_deal_id"] == "100"
    assert row["safe_for_deal_writeback"] == "Да"
    assert row["confidence_bucket"] == "high"


def test_multiple_candidates_require_manual_review() -> None:
    row = attribute_call_to_deal(
        {"call_id": "c1", "phone": "+79161112233", "contentful": "Да", "call_type": "sales_call"},
        [
            candidate("100", status_name="В работе", is_active_deal=True),
            candidate("101", status_name="Закрыто и не реализовано", loss_reason="Дубль", is_active_deal=False),
            candidate("102", status_name="Закрыто и не реализовано", loss_reason="Действующий клиент", is_active_deal=False),
        ],
    )

    assert row["attribution_decision"] == "linked_single_deal_candidate"
    assert row["selected_deal_id"] == "100"


def test_legacy_unknown_multiple_candidates_require_manual_review() -> None:
    row = attribute_call_to_deal(
        {"call_id": "c1", "phone": "+79161112233", "contentful": "Да", "call_type": "sales_call"},
        [
            {"deal_id": "100", "candidate_sources": "phone_rollup", "contact_ids": ""},
            {"deal_id": "101", "candidate_sources": "phone_rollup", "contact_ids": ""},
        ],
    )

    assert row["attribution_decision"] == "manual_review_multiple_deal_candidates"
    assert row["safe_for_deal_writeback"] == "Нет"


def test_active_lead_wins_over_duplicate_same_phone() -> None:
    row = attribute_call_to_deal(
        {"call_id": "c1", "phone": "+79161112233", "contentful": "Да", "call_type": "sales_call"},
        [
            candidate("200", status_name="Закрыто и не реализовано", loss_reason="Дубль", is_active_deal=False),
            candidate("100", status_name="Ожидание оплаты", is_active_deal=True),
        ],
    )

    assert row["attribution_decision"] == "linked_single_deal_candidate"
    assert row["selected_deal_id"] == "100"


def test_multiple_active_leads_go_to_manual_review() -> None:
    row = attribute_call_to_deal(
        {"call_id": "c1", "phone": "+79161112233", "contentful": "Да", "call_type": "sales_call"},
        [
            candidate("100", status_name="В работе", is_active_deal=True),
            candidate("101", status_name="Ожидание оплаты", is_active_deal=True),
            candidate("200", status_name="Закрыто и не реализовано", loss_reason="Дубль", is_active_deal=False),
        ],
    )

    assert row["attribution_decision"] == "manual_review_multiple_active_deals"
    assert row["selected_deal_id"] == ""


def test_all_terminal_candidates_go_to_specific_manual_review() -> None:
    row = attribute_call_to_deal(
        {"call_id": "c1", "phone": "+79161112233", "contentful": "Да", "call_type": "sales_call"},
        [
            candidate("200", status_name="Закрыто и не реализовано", loss_reason="Дубль", is_active_deal=False),
            candidate("201", status_name="Закрыто и не реализовано", loss_reason="Действующий клиент", is_active_deal=False),
        ],
    )

    assert row["attribution_decision"] == "manual_review_all_candidates_terminal"
    assert row["selected_deal_id"] == ""


def test_duplicate_of_link_resolved_to_main_lead() -> None:
    candidates = build_phone_deal_candidates(
        phone_rollup=[{"phone": "+79161112233", "amo_lead_ids": "100 | 200", "amo_contact_ids": "300"}],
        amo_ready=[],
        amo_writebacks=[],
        live_contacts=[{"contact_id": "300", "phones": "+79161112233"}],
        live_deals=[
            {
                "lead_id": "100",
                "lead_name": "Активная ЛОШ",
                "linked_contact_ids": "300",
                "status_name": "В работе",
                "pipeline_name": "Лиды",
                "updated_at": "2026-05-10T10:00:00+00:00",
            },
            {
                "lead_id": "200",
                "lead_name": "Дубль",
                "linked_contact_ids": "300",
                "status_name": "Закрыто и не реализовано",
                "loss_reason": "Дубль",
                "pipeline_name": "Обзвон",
                "closed_at": "2026-05-10T11:00:00+00:00",
                "_links": {"duplicate_of": {"id": "100"}},
            },
        ],
    )

    row = attribute_call_to_deal(
        {"call_id": "c1", "phone": "+79161112233", "contentful": "Да", "call_type": "sales_call"},
        candidates["+79161112233"],
    )

    duplicate = next(item for item in candidates["+79161112233"] if item["deal_id"] == "200")
    assert duplicate["duplicate_of_lead_id"] == "100"
    assert row["attribution_decision"] == "linked_single_deal_candidate"
    assert row["selected_deal_id"] == "100"


def test_confidence_thresholds_post_recalibration(monkeypatch) -> None:
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return datetime(2026, 5, 25, tzinfo=timezone.utc if tz is not None else None)

    monkeypatch.setattr(deal_attribution, "datetime", FrozenDatetime)

    active_fresh_payment = candidate(
        "100",
        status_name="Ожидание оплаты",
        is_active_deal=True,
        candidate_sources="amo_live_linked_contact",
        deal_updated_at="2026-05-10T10:00:00+00:00",
    )
    active_old_perspective = candidate(
        "101",
        status_name="Перспектива",
        is_active_deal=True,
        candidate_sources="amo_live_linked_contact | phone_rollup",
        deal_updated_at="2025-11-01T10:00:00+00:00",
    )
    terminal_duplicate = candidate(
        "102",
        status_name="Закрыто и не реализовано",
        loss_reason="Дубль",
        is_active_deal=False,
        candidate_sources="amo_live_linked_contact",
        deal_updated_at="2026-05-10T10:00:00+00:00",
    )

    assert CONFIDENCE_HIGH_THRESHOLD == 0.76
    assert CONFIDENCE_MEDIUM_THRESHOLD == 0.69
    assert confidence_bucket(single_candidate_confidence(active_fresh_payment)) == "high"
    assert confidence_bucket(single_candidate_confidence(active_old_perspective)) == "medium"
    assert confidence_bucket(single_candidate_confidence(terminal_duplicate)) == "low"


def test_active_lead_vs_duplicate_frozen_corpus_passes_selector() -> None:
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [row for row in rows if row["class_id"] == "active_lead_vs_duplicate_same_phone"]

    assert len(cases) >= 15
    for case in cases:
        row = attribute_call_to_deal(case["input"]["call"], case["input"]["candidates"])
        expected = case["expected"]["stage2"]
        assert row["attribution_decision"] == expected["attribution_decision"], case["case_id"]
        if expected.get("selected_deal_id"):
            assert row["selected_deal_id"] == expected["selected_deal_id"], case["case_id"]
        if expected.get("duplicate_candidate_id"):
            duplicate = next(
                item
                for item in case["input"]["candidates"]
                if item["deal_id"] == expected["duplicate_candidate_id"]
            )
            assert duplicate["duplicate_of_lead_id"] == expected["duplicate_of_lead_id"], case["case_id"]


def test_non_sales_call_is_skipped_even_with_single_candidate() -> None:
    row = attribute_call_to_deal(
        {"call_id": "c1", "phone": "+79161112233", "contentful": "Да", "call_type": "service_call"},
        [{"deal_id": "100", "candidate_sources": "phone_rollup", "contact_ids": ""}],
    )

    assert row["attribution_decision"] == "skipped_non_sales_call"
    assert row["selected_deal_id"] == ""


def test_live_snapshot_adds_deal_candidate_from_linked_contact() -> None:
    candidates = build_phone_deal_candidates(
        phone_rollup=[],
        amo_ready=[],
        amo_writebacks=[],
        live_contacts=[
            {"contact_id": "200", "phones": "+79161112233"},
        ],
        live_deals=[
            {
                "lead_id": "100",
                "lead_name": "Летний лагерь",
                "linked_contact_ids": "200",
                "status_name": "Новая заявка",
                "pipeline_name": "Воронка",
            }
        ],
    )

    assert candidates["+79161112233"][0]["deal_id"] == "100"
    assert candidates["+79161112233"][0]["candidate_sources"] == "amo_live_linked_contact"
    assert candidates["+79161112233"][0]["deal_name"] == "Летний лагерь"


def test_build_stage2_attribution_outputs_csv_sqlite_and_fail_closed_summary(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1"
    live = tmp_path / "live"
    out = tmp_path / "out"
    _write_csv(
        stage1 / "call_snapshot.csv",
        [
            {"call_id": "1", "phone": "+79161112233", "contentful": "Да", "call_type": "sales_call"},
            {"call_id": "2", "phone": "+79161112233", "contentful": "Нет", "call_type": "sales_call"},
            {"call_id": "3", "phone": "+79169998877", "contentful": "Да", "call_type": "sales_call"},
        ],
    )
    _write_csv(stage1 / "phone_rollup.csv", [{"phone": "+79161112233", "amo_lead_ids": "100", "amo_contact_ids": "200"}])
    _write_csv(stage1 / "amo_ready_snapshot.csv", [])
    _write_csv(stage1 / "amo_writeback_snapshot.csv", [])
    _write_csv(live / "amo_deals_snapshot.csv", [{"lead_id": "100", "linked_contact_ids": "200"}])
    _write_csv(live / "amo_contacts_snapshot.csv", [{"contact_id": "200", "phones": "+79161112233"}])
    (live / "summary.json").write_text(
        '{"api_read_succeeded": true, "fetch": {"contacts_seen": 1, "leads_seen": 1}}',
        encoding="utf-8",
    )

    summary = build_deal_attribution_dry_run(
        AttributionPaths(stage1_snapshot_root=stage1, amo_live_snapshot_root=live, out_root=out)
    )

    assert summary["safety"]["write_amo"] is False
    assert summary["coverage"]["calls_seen"] == 3
    assert summary["decision_counts"]["linked_single_deal_candidate"] == 1
    assert summary["decision_counts"]["skipped_non_contentful_call"] == 1
    assert summary["decision_counts"]["manual_review_no_deal_candidate"] == 1
    assert summary["readiness"]["safe_to_write_deal_fields"] is False
    assert (out / "deal_call_links.csv").exists()
    assert (out / "deal_aware_stage2_attribution.sqlite").exists()


def test_build_stage2_attribution_blocks_linking_when_live_snapshot_failed(tmp_path: Path) -> None:
    stage1 = tmp_path / "stage1"
    live = tmp_path / "live"
    out = tmp_path / "out"
    _write_csv(
        stage1 / "call_snapshot.csv",
        [{"call_id": "1", "phone": "+79161112233", "contentful": "Да", "call_type": "sales_call"}],
    )
    _write_csv(stage1 / "phone_rollup.csv", [{"phone": "+79161112233", "amo_lead_ids": "100", "amo_contact_ids": "200"}])
    _write_csv(stage1 / "amo_ready_snapshot.csv", [])
    _write_csv(stage1 / "amo_writeback_snapshot.csv", [])
    _write_csv(live / "amo_deals_snapshot.csv", [])
    _write_csv(live / "amo_contacts_snapshot.csv", [])
    (live / "summary.json").write_text(
        '{"api_read_succeeded": false, "preflight_error": "token revoked", "fetch": {"contacts_seen": 0, "leads_seen": 0}}',
        encoding="utf-8",
    )

    summary = build_deal_attribution_dry_run(
        AttributionPaths(stage1_snapshot_root=stage1, amo_live_snapshot_root=live, out_root=out)
    )

    assert summary["coverage"]["linked_rows"] == 0
    assert summary["decision_counts"]["manual_review_live_amo_snapshot_unavailable"] == 1
    assert summary["coverage"]["safe_for_future_deal_writeback_rows"] == 0


def candidate(
    deal_id: str,
    *,
    status_name: str = "В работе",
    loss_reason: str = "",
    is_active_deal: bool = True,
    candidate_sources: str = "amo_live_linked_contact",
    deal_updated_at: str = "2026-05-10T10:00:00+00:00",
) -> dict[str, object]:
    return {
        "deal_id": deal_id,
        "contact_ids": "300",
        "candidate_sources": candidate_sources,
        "deal_name": f"Сделка {deal_id}",
        "pipeline_name": "Лиды",
        "status_name": status_name,
        "loss_reason": loss_reason,
        "is_terminal_deal": not is_active_deal,
        "is_active_deal": is_active_deal,
        "is_duplicate_or_existing_client": loss_reason in {"Дубль", "Действующий клиент"},
        "deal_updated_at": deal_updated_at,
    }
