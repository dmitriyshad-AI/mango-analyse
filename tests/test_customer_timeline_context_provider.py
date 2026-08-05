from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mango_mvp.customer_timeline.context_provider import (
    CustomerTimelineCoveragePaths,
    assert_timeline_stage_allowed,
    audit_customer_timeline_coverage,
    context_provider_safety_contract,
    evaluate_timeline_promotion,
    get_customer_context_for_phone,
)
from tests.test_customer_timeline_read_api import seed_timeline_db


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_context_provider_reads_from_timeline_db_when_enabled(tmp_path: Path) -> None:
    db_path, _customer_id = seed_timeline_db(tmp_path)

    context = get_customer_context_for_phone("+79161234567", timeline_db=db_path)

    assert context["source"] == "customer_timeline"
    assert context["found"] is True
    assert context["fallback_used"] is False
    assert context["timeline"]["items"][0]["event_type"] == "mango_call"
    assert context["readiness"]["safe_for_automatic_bot"] is False
    assert context["safety"]["write_crm"] is False


def test_context_provider_falls_back_when_timeline_db_missing(tmp_path: Path) -> None:
    missing_db = tmp_path / "missing" / "customer_timeline.sqlite"
    fallback_rows = [
        {
            "phone": "+79000000000",
            "started_at": "2026-05-15T10:00:00+03:00",
            "call_summary": "Клиент интересовался оплатой.",
            "next_step": "Перезвонить после проверки.",
        }
    ]

    context = get_customer_context_for_phone("+79000000000", timeline_db=missing_db, fallback_rows=fallback_rows)

    assert context["source"] == "fallback_rows"
    assert context["found"] is True
    assert context["fallback_used"] is True
    assert context["timeline"]["items"][0]["summary"] == "Клиент интересовался оплатой."
    assert any("timeline_unavailable" in warning for warning in context["warnings"])
    assert not missing_db.exists()
    assert not missing_db.parent.exists()


def test_context_provider_never_writes_live_systems() -> None:
    safety = context_provider_safety_contract()

    assert safety["write_crm"] is False
    assert safety["write_tallanto"] is False
    assert safety["send_email"] is False
    assert safety["send_messenger"] is False
    assert safety["run_asr"] is False
    assert safety["run_ra"] is False
    assert safety["write_customer_timeline_db"] is False
    assert safety["live_writeback_required"] is False


def test_context_provider_rejects_stable_runtime_output_path(tmp_path: Path) -> None:
    db_path, _customer_id = seed_timeline_db(tmp_path)
    candidates = tmp_path / "candidates.csv"
    _write_csv(candidates, [{"selected_deal_id": "1", "phones": "+79161234567"}])

    with pytest.raises(ValueError, match="stable_runtime"):
        audit_customer_timeline_coverage(
            CustomerTimelineCoveragePaths(
                deal_aware_candidates_csv=candidates,
                timeline_db=db_path,
                out_root=tmp_path / "stable_runtime" / "timeline_coverage",
            )
        )


def test_coverage_report_counts_missing_deal_aware_phones(tmp_path: Path) -> None:
    db_path, _customer_id = seed_timeline_db(tmp_path)
    candidates = tmp_path / "candidates.csv"
    out = tmp_path / "coverage"
    _write_csv(
        candidates,
        [
            {"selected_deal_id": "1", "phones": "+79161234567"},
            {"selected_deal_id": "2", "phones": "+79000000000"},
        ],
    )

    report = audit_customer_timeline_coverage(
        CustomerTimelineCoveragePaths(deal_aware_candidates_csv=candidates, timeline_db=db_path, out_root=out)
    )

    assert report["summary"]["deal_aware_unique_phones"] == 2
    assert report["summary"]["timeline_matched_phones"] == 1
    assert report["summary"]["timeline_missing_phones"] == 1
    assert (out / "timeline_coverage_report.csv").exists()
    assert json.loads((out / "summary.json").read_text(encoding="utf-8"))["coverage_ratio"] == 0.5


def test_timeline_primary_read_enabled_after_coverage_gate() -> None:
    promotion = evaluate_timeline_promotion(
        {
            "timeline_available": True,
            "deal_aware_unique_phones": 2,
            "timeline_missing_phones": 0,
            "coverage_ratio": 1.0,
        },
        preview_enabled=True,
        primary_read_enabled=True,
    )

    assert promotion["stages"]["timeline_primary_read_enabled"] is True
    assert_timeline_stage_allowed("timeline_primary_read_enabled", promotion)


def test_timeline_live_write_context_requires_verified_coverage() -> None:
    promotion = evaluate_timeline_promotion(
        {
            "timeline_available": True,
            "deal_aware_unique_phones": 2,
            "timeline_missing_phones": 1,
            "coverage_ratio": 0.5,
        },
        preview_enabled=True,
        primary_read_enabled=True,
        live_write_context_requested=True,
    )

    assert promotion["stages"]["timeline_live_write_context_allowed"] is False
    with pytest.raises(ValueError, match="not allowed"):
        assert_timeline_stage_allowed("timeline_live_write_context_allowed", promotion)
