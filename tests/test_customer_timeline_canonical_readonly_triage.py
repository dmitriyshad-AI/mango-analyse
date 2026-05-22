from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.customer_timeline.canonical_readonly_import import build_canonical_readonly_customer_timeline
from mango_mvp.customer_timeline.canonical_readonly_triage import (
    CanonicalReadonlyTriageConfig,
    build_canonical_readonly_timeline_triage,
    build_preview_plan,
    classify_customer_for_triage,
    classify_reason,
)
from tests.test_customer_timeline_canonical_readonly_import import _config


NOW = datetime(2026, 5, 21, 10, 0, tzinfo=timezone.utc)


def test_triage_report_splits_identity_risk_and_blocks_preview(tmp_path: Path) -> None:
    import_config = _config(tmp_path)
    build_canonical_readonly_customer_timeline(import_config)

    report = build_canonical_readonly_timeline_triage(
        CanonicalReadonlyTriageConfig(
            project_root=tmp_path,
            timeline_root=import_config.out_root,
            current_runtime_json=import_config.current_runtime_json,
            amo_contacts_csv=import_config.amo_contacts_csv,
            amo_deals_csv=import_config.amo_deals_csv,
            mail_handoff_db=import_config.mail_handoff_db,
            mail_bridge_db=import_config.mail_bridge_db,
            generated_at=NOW,
        )
    )

    assert report["summary"]["total_customers"] == 2
    assert report["summary"]["identity_risk_customers"] == 2
    assert report["summary"]["no_manual_review_reason_customers"] == 0
    assert report["summary"]["shared_amo_contact_customers"] == 2
    assert report["summary"]["shared_amo_lead_customers"] == 2
    assert report["summary"]["timeline_preview_enabled_allowed"] is False
    assert report["summary"]["timeline_primary_read_enabled_allowed"] is False
    assert report["preview_plan"]["decision"] == "manager_preview_only_not_bot"
    assert report["safety"]["write_crm"] is False
    assert report["safety"]["raw_personal_values_in_reports"] is False


def test_triage_public_reports_do_not_leak_raw_identity_values(tmp_path: Path) -> None:
    import_config = _config(tmp_path)
    build_canonical_readonly_customer_timeline(import_config)
    build_canonical_readonly_timeline_triage(
        CanonicalReadonlyTriageConfig(
            project_root=tmp_path,
            timeline_root=import_config.out_root,
            current_runtime_json=import_config.current_runtime_json,
            amo_contacts_csv=import_config.amo_contacts_csv,
            amo_deals_csv=import_config.amo_deals_csv,
            mail_handoff_db=import_config.mail_handoff_db,
            mail_bridge_db=import_config.mail_bridge_db,
            generated_at=NOW,
        )
    )

    out_dir = import_config.out_root / "triage"
    public_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            out_dir / "manual_review_triage_report.json",
            out_dir / "manual_review_triage_report.md",
            out_dir / "manager_preview_plan.md",
        ]
    )

    assert "Иван Петров" not in public_text
    assert "Мария Сидорова" not in public_text
    assert "parent@example.com" not in public_text
    assert "+79161234567" not in public_text
    assert not re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", public_text)


def test_triage_classifiers_are_conservative() -> None:
    assert classify_reason("shared_amo_contact_across_customers") == "identity_risk"
    assert classify_reason("source_manual_review_required") == "source_label"
    assert classify_customer_for_triage(brand="foton", reasons=()) == "manager_preview_candidate"
    assert classify_customer_for_triage(brand="unknown", reasons=()) == "unknown_brand_audit_only"
    assert classify_customer_for_triage(brand="foton", reasons=("source_manual_review_required",)) == "source_label_only"
    assert (
        classify_customer_for_triage(
            brand="foton",
            reasons=("source_manual_review_required", "shared_amo_lead_across_customers"),
        )
        == "identity_risk"
    )


def test_preview_plan_never_enables_bot_flags() -> None:
    plan = build_preview_plan(
        total=100,
        preview_candidates_by_brand={"foton": 10, "unpk": 5},
        identity_risk_by_brand={"foton": 20},
        source_label_only_by_brand={"unknown": 30},
        unknown_brand_count=50,
    )

    assert plan["eligible_manager_preview_candidates"] == 15
    assert plan["clean_smoke_batch_status"] == "blocked_insufficient_clean_candidates"
    assert plan["recommended_batches"][0]["target_size"] == 0
    assert plan["timeline_preview_enabled_allowed"] is False
    assert plan["timeline_primary_read_enabled_allowed"] is False
    assert "Не включать auto-answer." in plan["hard_rules"]
