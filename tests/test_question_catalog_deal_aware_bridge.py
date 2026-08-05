from __future__ import annotations

import csv
from pathlib import Path

from mango_mvp.question_catalog.extractors import extract_call_questions
from mango_mvp.question_catalog.source_index import build_source_index, load_source_index, write_source_index


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


def test_source_index_maps_call_id_to_theme_id(tmp_path: Path) -> None:
    source = tmp_path / "enriched_reviews.csv"
    _write_csv(
        source,
        [
            {
                "started_at": "2026-05-10T10:00:00+00:00",
                "call_id": "call-raw-1",
                "customer_question": "Сколько стоит курс?",
            }
        ],
    )
    items, _ = extract_call_questions(source, tenant_id="foton")

    index = build_source_index(items)

    assert "call-raw-1" in index
    assert index["call-raw-1"]["theme_ids"] or index["call-raw-1"]["service_ids"]


def test_source_index_json_roundtrip_preserves_manager_only_mode(tmp_path: Path) -> None:
    rows = [
        {
            "call_id": "call-1",
            "theme_ids": "theme:009_refund",
            "service_ids": "",
            "policy_statuses": "manager_only",
            "bot_allowed_modes": "manager_only",
            "risk_flags": "manager_only",
        }
    ]

    output = write_source_index(tmp_path, rows)
    index = load_source_index(Path(output["json"]))

    assert index["call-1"]["theme_ids"] == ["theme:009_refund"]
    assert index["call-1"]["bot_allowed_modes"] == ["manager_only"]
    assert index["call-1"]["risk_flags"] == ["manager_only"]
