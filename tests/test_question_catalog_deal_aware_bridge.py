from __future__ import annotations

import csv
from pathlib import Path

from mango_mvp.question_catalog.extractors import extract_call_questions


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_call_question_extractor_keeps_raw_call_id_in_metadata(tmp_path: Path) -> None:
    source = tmp_path / "enriched_reviews.csv"
    _write_csv(
        source,
        [
            {
                "started_at": "2026-05-10T10:00:00+00:00",
                "call_id": "call-raw-1",
                "recording_id": "rec-1",
                "moment_id": "moment-1",
                "customer_question": "Сколько стоит курс?",
            }
        ],
    )

    items, _ = extract_call_questions(source, tenant_id="foton")

    assert items[0].metadata["call_id"] == "call-raw-1"
    assert items[0].metadata["recording_id"] == "rec-1"
    assert items[0].metadata["moment_id"] == "moment-1"
    assert items[0].metadata["source_kind"] == "call"
    assert items[0].metadata["source_table"] == "enriched_reviews.csv"
