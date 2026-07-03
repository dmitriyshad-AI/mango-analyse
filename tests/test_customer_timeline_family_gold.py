from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from mango_mvp.customer_timeline.family_gold import FamilyGoldCheckConfig, check_family_gold


def test_family_gold_check_allows_only_flagged_count_mismatches(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _insert_family(db, customer_id="customer:ok", children=("Аня",), confidences=("high",))
    _insert_family(db, customer_id="customer:flagged", children=("Александр",), confidences=("low",))
    _insert_family(db, customer_id="customer:bad", children=("Даниил",), confidences=("low",))
    gold = _gold(
        tmp_path,
        [
            {"customer_id": "customer:ok", "expected_children_count": 1, "expected_link_confidence": "high", "flags": []},
            {
                "customer_id": "customer:flagged",
                "expected_children_count": 2,
                "expected_link_confidence": "medium",
                "flags": ["suspicious_initials:Камаренцев Э.Н."],
            },
            {"customer_id": "customer:bad", "expected_children_count": 2, "expected_link_confidence": "medium", "flags": []},
        ],
    )

    summary = check_family_gold(
        FamilyGoldCheckConfig(
            timeline_db=db,
            gold_jsonl=gold,
            allowed_root=tmp_path,
            out_json=tmp_path / ".codex_local" / "review" / "family_gold.json",
        )
    )

    assert summary["gold_rows"] == 3
    assert summary["exact_count_ok"] == 1
    assert summary["count_mismatches"] == 2
    assert summary["flagged_count_mismatches"] == 1
    assert summary["unflagged_count_mismatches"] == 1
    assert summary["false_high"] == 0
    assert summary["strict_pass"] is False
    assert summary["architect_review_required"] is True
    assert summary["flag_keys_seen"] == ["suspicious_initials"]
    assert (tmp_path / ".codex_local" / "review" / "family_gold.json").exists()


def test_family_gold_check_detects_false_high(tmp_path: Path) -> None:
    db = _db(tmp_path)
    _insert_family(db, customer_id="customer:false-high", children=("Миша",), confidences=("high",))
    gold = _gold(
        tmp_path,
        [
            {
                "customer_id": "customer:false-high",
                "expected_children_count": 1,
                "expected_link_confidence": "medium",
                "flags": [],
            }
        ],
    )

    summary = check_family_gold(FamilyGoldCheckConfig(timeline_db=db, gold_jsonl=gold, allowed_root=tmp_path))

    assert summary["false_high"] == 1
    assert summary["strict_pass"] is False


def _db(tmp_path: Path) -> Path:
    db = tmp_path / ".codex_local" / "staging" / "timeline.sqlite"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as con:
        con.executescript(
            """
            CREATE TABLE family_links_v1 (
              tenant_id TEXT NOT NULL,
              customer_id TEXT NOT NULL,
              child_key TEXT NOT NULL,
              canonical_name TEXT NOT NULL,
              name_variants_json TEXT NOT NULL,
              confidence TEXT NOT NULL,
              status TEXT NOT NULL,
              brand TEXT NOT NULL DEFAULT 'unknown',
              grades_json TEXT NOT NULL DEFAULT '[]',
              subjects_json TEXT NOT NULL DEFAULT '[]',
              source_refs_json TEXT NOT NULL DEFAULT '[]',
              record_json TEXT NOT NULL DEFAULT '{}',
              updated_at TEXT NOT NULL,
              PRIMARY KEY (tenant_id, customer_id, child_key)
            );
            """
        )
    return db


def _insert_family(db: Path, *, customer_id: str, children: tuple[str, ...], confidences: tuple[str, ...]) -> None:
    with sqlite3.connect(db) as con:
        for index, (name, confidence) in enumerate(zip(children, confidences), start=1):
            con.execute(
                """
                INSERT INTO family_links_v1 (
                  tenant_id, customer_id, child_key, canonical_name, name_variants_json,
                  confidence, status, record_json, updated_at
                )
                VALUES ('foton', ?, ?, ?, ?, ?, ?, ?, '2026-07-03T00:00:00+00:00')
                """,
                (
                    customer_id,
                    f"child-{index}",
                    name,
                    json.dumps([name], ensure_ascii=False),
                    confidence,
                    "confident" if confidence == "high" else "ambiguous",
                    json.dumps({"suspicious_reasons": []}, ensure_ascii=False),
                ),
            )


def _gold(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "family_gold.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return path
