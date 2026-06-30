from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1_sources"
SNAPSHOT_PATH = ROOT / "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json"

TARGET_PRICE_FACTS = {
    ("foton", "prices_regular_2026_27.offline_3_4_class.before_2026_07_01.semester"): {"amount": 28250},
    ("foton", "prices_regular_2026_27.offline_3_4_class.before_2026_07_01.year"): {"amount": 47000},
    ("foton", "prices_regular_2026_27.offline_5_11_class.before_2026_07_01.semester"): {"amount": 44600},
    ("foton", "prices_regular_2026_27.offline_5_11_class.before_2026_07_01.year"): {"amount": 74500},
    ("foton", "prices_regular_2026_27.online_3_4_class.before_2026_08_01.semester"): {"amount": 19000},
    ("foton", "prices_regular_2026_27.online_3_4_class.before_2026_08_01.year_range"): {
        "amount_min": 29900,
        "amount_max": 33300,
    },
    ("foton", "prices_regular_2026_27.online_5_11_class.before_2026_08_01.semester"): {"amount": 29750},
    ("foton", "prices_regular_2026_27.online_5_11_class.before_2026_08_01.year"): {"amount": 47250},
    ("unpk", "prices_regular_2026_27.offline_1_4_class.before_2026_07_01.semester"): {"amount": 31000},
    ("unpk", "prices_regular_2026_27.offline_1_4_class.before_2026_07_01.year"): {"amount": 51700},
    ("unpk", "prices_regular_2026_27.offline_5_11_class.before_2026_07_01.semester"): {"amount": 49000},
    ("unpk", "prices_regular_2026_27.offline_5_11_class.before_2026_07_01.year"): {"amount": 82000},
}

DEADLINE_MARKERS = (
    "до 01.07",
    "до 01.08",
    "до 1 июля",
    "до 1 августа",
    "действительна до",
    "действует до",
)


def _snapshot_facts(path: Path) -> list[dict[str, object]]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    return list(snapshot.get("facts") or [])


def _assert_tz156_price_facts(facts: list[dict[str, object]]) -> None:
    by_key = {(str(fact.get("brand") or ""), str(fact.get("fact_key") or "")): fact for fact in facts}

    assert set(TARGET_PRICE_FACTS).issubset(by_key)
    for key, expected in TARGET_PRICE_FACTS.items():
        fact = by_key[key]
        structured = dict(fact.get("structured_value") or {})

        assert fact.get("valid_until") == "2026-12-31"
        assert structured.get("valid_until") == "2026-12-31"
        assert fact.get("allowed_for_client_answer") is True
        assert fact.get("brand") == key[0]

        for amount_key, amount in expected.items():
            assert structured.get(amount_key) == amount

        client_safe_text = str(fact.get("client_safe_text") or "").casefold().replace("ё", "е")
        assert not any(marker in client_safe_text for marker in DEADLINE_MARKERS)

    for brand, fact_key in TARGET_PRICE_FACTS:
        assert (brand, f"{fact_key}.raw_value") not in by_key
        assert (brand, f"{fact_key}.valid_until") not in by_key


def test_tz156_pinned_snapshot_extends_price_valid_until_without_price_or_brand_drift() -> None:
    _assert_tz156_price_facts(_snapshot_facts(SNAPSHOT_PATH))


def test_tz156_rebuild_from_yaml_source_preserves_explicit_valid_until(tmp_path: Path) -> None:
    from scripts import build_kb_release_v3_from_claude_handoff as builder

    release_out = tmp_path / "release"
    handoff_out = tmp_path / "handoff"
    builder.build_kb_release_v3(
        run_id="kb_release_20260612_v6_7_staging_r4_1_tz156_test",
        handoff_dir=SOURCE_ROOT,
        out_dir=release_out,
        handoff_out_dir=handoff_out,
    )

    _assert_tz156_price_facts(_snapshot_facts(release_out / "kb_release_v3_snapshot.json"))
