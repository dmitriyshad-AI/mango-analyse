from __future__ import annotations

import csv
from pathlib import Path

from mango_mvp.quality.transcript_quality_auto_fix_review import (
    AutoFixReviewConfig,
    build_transcript_quality_auto_fix_review,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_build_transcript_quality_auto_fix_review_splits_contentful_and_safe(tmp_path: Path) -> None:
    dry_run_root = tmp_path / "dry_run"
    _write_csv(
        dry_run_root / "auto_fix_candidates.csv",
        [
            {
                "source_filename": "safe_nonconv.mp3",
                "month": "2025-09",
                "current_call_type": "non_conversation",
                "current_contentful": "False",
                "guardrail_reason_codes": "no_live_marker|asr_artifact_marker",
                "history_summary_excerpt": "Автоответчик.",
                "transcript_excerpt": "Продолжение следует. Номер недоступен.",
            },
            {
                "source_filename": "contentful_service.mp3",
                "month": "2025-09",
                "current_call_type": "service_call",
                "current_contentful": "True",
                "guardrail_reason_codes": "no_live_marker|asr_artifact_marker",
                "history_summary_excerpt": "Нецелевой звонок.",
                "transcript_excerpt": "Субтитры сделал DimaTorzok.",
            },
            {
                "source_filename": "sales_review.mp3",
                "month": "2025-10",
                "current_call_type": "sales_call",
                "current_contentful": "True",
                "guardrail_reason_codes": "no_live_marker|asr_artifact_marker",
                "history_summary_excerpt": "Контакт не состоялся.",
                "transcript_excerpt": "Субтитры сделал DimaTorzok.",
            },
        ],
    )

    summary = build_transcript_quality_auto_fix_review(
        AutoFixReviewConfig(
            dry_run_root=dry_run_root,
            out_root=tmp_path / "review",
            sample_per_current_call_type=2,
            sample_per_month=2,
        )
    )

    assert summary["auto_fix_candidates"] == 3
    assert summary["contentful_auto_fix_candidates"] == 2
    assert summary["safe_non_contentful_auto_fix_candidates"] == 1
    assert summary["current_call_type_counts"] == {
        "non_conversation": 1,
        "service_call": 1,
        "sales_call": 1,
    }
    contentful_rows = _read_csv(tmp_path / "review" / "contentful_auto_fix_candidates.csv")
    safe_rows = _read_csv(tmp_path / "review" / "safe_non_contentful_auto_fix_candidates.csv")
    sample_rows = _read_csv(tmp_path / "review" / "review_sample.csv")
    assert [row["source_filename"] for row in contentful_rows] == ["contentful_service.mp3", "sales_review.mp3"]
    assert [row["source_filename"] for row in safe_rows] == ["safe_nonconv.mp3"]
    assert any(row["review_decision"] == "human_review_required_sales_call" for row in sample_rows)
    assert (tmp_path / "review" / "AUTO_FIX_REVIEW_REPORT.md").exists()
    assert (tmp_path / "review" / "summary.json").exists()
