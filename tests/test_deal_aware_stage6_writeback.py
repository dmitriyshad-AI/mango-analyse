from __future__ import annotations

import csv
import json
from pathlib import Path

from mango_mvp.deal_aware.deal_text_builder import DEAL_AI_FIELDS
from mango_mvp.deal_aware.deal_writeback import DealAwareStage6Paths, run_deal_aware_stage6_preflight
from scripts.readback_deal_aware_amo_fields import readback_findings, values_equivalent


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_row() -> dict[str, str]:
    row = {field: f"{field}: безопасный тестовый текст" for field in DEAL_AI_FIELDS}
    row.update(
        {
            "review_id": "deal-stage5-00001",
            "selected_deal_id": "123",
            "stage5_decision": "allow_stage6_dry_run",
            "stage5_warning_gate_count": "1",
            "AI-приоритет сделки": "warm",
            "candidate_phone_count": "1",
            "tallanto_context_status": "exact_phone_single",
        }
    )
    row["AI-дата следующего касания"] = "2026-05-15"
    row["AI-дата обновления сделки"] = "2026-05-13T11:57:33+00:00"
    return row


def _field_catalog() -> dict[str, object]:
    return {
        "synced_at": "2026-05-13T11:22:02+00:00",
        "fields": [
            {
                "id": 1000 + index,
                "name": field,
                "type": "date_time" if field == "AI-дата обновления сделки" else "textarea",
                "is_api_only": False,
            }
            for index, field in enumerate(DEAL_AI_FIELDS)
        ],
    }


def test_stage6_preflight_builds_dry_run_and_stays_fail_closed(tmp_path: Path) -> None:
    input_csv = tmp_path / "candidates.csv"
    stage5_summary = tmp_path / "stage5_summary.json"
    field_catalog = tmp_path / "lead_field_catalog_cache.json"
    out = tmp_path / "out"
    _write_csv(input_csv, [_candidate_row()])
    stage5_summary.write_text(
        json.dumps(
            {
                "schema_version": "deal_aware_stage5_quality_gate_v1",
                "readiness": {"passed_for_stage6_dry_run": True, "passed_for_live_writeback": False},
                "outputs": {"dry_run_candidates_csv": str(input_csv.resolve())},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    field_catalog.write_text(json.dumps(_field_catalog(), ensure_ascii=False), encoding="utf-8")

    summary = run_deal_aware_stage6_preflight(
        DealAwareStage6Paths(
            input_csv=input_csv,
            stage5_summary_json=stage5_summary,
            field_catalog_cache_json=field_catalog,
            out_root=out,
            stage20_size=1,
        )
    )

    assert summary["coverage"]["input_rows"] == 1
    assert summary["coverage"]["dry_run_rows"] == 1
    assert summary["coverage"]["stage20_candidate_rows"] == 1
    assert summary["readiness"]["passed_for_stage20_preflight"] is True
    assert summary["readiness"]["passed_for_live_writeback"] is False
    assert (out / "deal_stage6_dry_run_report.csv").exists()
    assert (out / "next_live_stage20_then_readback.sh").exists()


def test_stage6_preflight_blocks_missing_field_catalog_field(tmp_path: Path) -> None:
    input_csv = tmp_path / "candidates.csv"
    stage5_summary = tmp_path / "stage5_summary.json"
    field_catalog = tmp_path / "lead_field_catalog_cache.json"
    out = tmp_path / "out"
    _write_csv(input_csv, [_candidate_row()])
    stage5_summary.write_text(
        json.dumps(
            {
                "readiness": {"passed_for_stage6_dry_run": True, "passed_for_live_writeback": False},
                "outputs": {"dry_run_candidates_csv": str(input_csv.resolve())},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    payload = _field_catalog()
    payload["fields"] = payload["fields"][:-1]  # type: ignore[index]
    field_catalog.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    summary = run_deal_aware_stage6_preflight(
        DealAwareStage6Paths(
            input_csv=input_csv,
            stage5_summary_json=stage5_summary,
            field_catalog_cache_json=field_catalog,
            out_root=out,
            stage20_size=1,
        )
    )

    assert summary["coverage"]["blocked_rows"] == 1
    assert summary["readiness"]["passed_for_stage20_preflight"] is False
    assert summary["field_catalog_guard"]["missing_fields"]


def test_deal_readback_treats_iso_datetime_and_amo_timestamp_as_equivalent() -> None:
    assert values_equivalent("1778673453", "2026-05-13T11:57:33+00:00")
    findings = readback_findings(
        {"AI-дата обновления сделки": "1778673453", **{field: "ok" for field in DEAL_AI_FIELDS if field != "AI-дата обновления сделки"}},
        {"AI-дата обновления сделки": "2026-05-13T11:57:33+00:00", **{field: "ok" for field in DEAL_AI_FIELDS if field != "AI-дата обновления сделки"}},
        min_severity="P2",
    )
    assert [finding.risk_type for finding in findings] == []
