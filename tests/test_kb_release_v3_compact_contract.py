from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_kb_release_v3_from_claude_handoff import build_snapshot_v3, csv_value


RELEASE = Path(
    "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1"
)


def test_builder_keeps_one_authoritative_copy_of_snapshot_records() -> None:
    snapshot = build_snapshot_v3(
        run_id="test",
        sources=[{"source_id": "source:1"}],
        facts=[
            {
                "fact_id": "fact:1",
                "fact_key": "course.address",
                "fact_text": "Адрес центра.",
                "client_safe_text": "Адрес центра.",
                "allowed_for_client_answer": True,
                "usable_for_precise_answer": True,
                "brand": "foton",
                "fact_type": "address",
                "source_id": "source:1",
            }
        ],
        approval_queue=[{"approval_item_id": "approval:1"}],
        post_filter={},
        brand_rules={},
        bot_policy={},
    )

    assert set(snapshot).isdisjoint(
        {"facts_registry", "knowledge_chunks", "source_registry", "approval_queue"}
    )
    assert len(snapshot["facts"]) == 1
    assert len(snapshot["chunks"]) == 1
    assert len(snapshot["sources"]) == 1
    assert snapshot["summary"]["approval_queue_items"] == 1


def test_current_snapshot_matches_the_separate_canonical_registries() -> None:
    snapshot = json.loads((RELEASE / "kb_release_v3_snapshot.json").read_text(encoding="utf-8"))

    assert set(snapshot).isdisjoint(
        {"facts_registry", "knowledge_chunks", "source_registry", "approval_queue"}
    )
    facts = [json.loads(line) for line in (RELEASE / "facts_registry.jsonl").read_text(encoding="utf-8").splitlines()]
    sources = json.loads((RELEASE / "source_registry.json").read_text(encoding="utf-8"))["items"]
    with (RELEASE / "knowledge_chunks.csv").open(encoding="utf-8", newline="") as file:
        chunks = list(csv.DictReader(file))
    with (RELEASE / "approval_queue_for_rop_v3.csv").open(encoding="utf-8", newline="") as file:
        approval_queue = list(csv.DictReader(file))

    def csv_rows(items: list[dict[str, object]], fields: list[str]) -> list[dict[str, str]]:
        return [
            {
                field: "" if csv_value(item.get(field)) is None else str(csv_value(item.get(field)))
                for field in fields
            }
            for item in items
        ]

    assert snapshot["facts"] == facts
    assert snapshot["sources"] == sources
    assert csv_rows(snapshot["chunks"], list(chunks[0])) == chunks
    assert snapshot["summary"]["approval_queue_items"] == len(approval_queue)
