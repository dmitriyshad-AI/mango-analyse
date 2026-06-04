from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from scripts.derive_kb_schedule_2026_27_sources import DerivationError, main


def test_schedule_derivation_writes_brand_split_sources_and_is_idempotent(tmp_path: Path) -> None:
    schedule_pack = _write_schedule_pack(tmp_path / "schedule_pack")
    source_dir = _write_source_dir(tmp_path / "sources")
    audit_pack = tmp_path / "audit"

    rc = main(
        [
            "--schedule-pack",
            str(schedule_pack),
            "--source-dir",
            str(source_dir),
            "--audit-pack",
            str(audit_pack),
            "--release-id",
            "kb_release_test_schedule",
            "--freshness-date",
            "2026-06-02",
        ]
    )

    assert rc == 0
    foton = _load_yaml(source_dir / "facts" / "facts_for_bot_FOTON.yaml")
    unpk = _load_yaml(source_dir / "facts" / "facts_for_bot_UNPK.yaml")
    manifest = _load_yaml(source_dir / "release_manifest.yaml")

    foton_groups = foton["schedule_2026_27"]["groups"]
    unpk_groups = unpk["schedule_2026_27"]["groups"]
    assert len(foton_groups) == 1
    assert len(unpk_groups) == 1
    assert manifest["release_id"] == "kb_release_test_schedule"
    assert manifest["freshness_check_date"] == "2026-06-02"
    assert {
        (item["file"], item["path"])
        for item in manifest["required_yaml_paths"]
    } >= {
        ("facts/facts_for_bot_FOTON.yaml", "schedule_2026_27.groups"),
        ("facts/facts_for_bot_UNPK.yaml", "schedule_2026_27.groups"),
    }

    foton_text = next(iter(foton_groups.values()))["client_safe_text"]
    unpk_text = next(iter(unpk_groups.values()))["client_safe_text"]
    assert "Математика, 5 класс, продвинутая группа, очно" in foton_text
    assert "Физика, 11 класс, ЕГЭ, онлайн" in unpk_text
    assert "Точное расписание конкретной группы уточняется" in foton_text
    for text in (foton_text, unpk_text):
        assert "Tallanto" not in text
        assert "source_id" not in text
        assert "fact_id" not in text
        assert "match_key" not in text
        assert "места есть" not in text.casefold()
    assert "УНПК" not in foton_text
    assert "Фотон" not in unpk_text

    ingest_rows = list(csv.DictReader((audit_pack / "ingest_decision.csv").open(encoding="utf-8")))
    assert [row["decision"] for row in ingest_rows] == ["include", "include"]
    assert ingest_rows[1]["note"] == "empty_filial_fallback_by_name"

    before = _source_hashes(source_dir)
    rc = main(
        [
            "--schedule-pack",
            str(schedule_pack),
            "--source-dir",
            str(source_dir),
            "--audit-pack",
            str(audit_pack),
            "--release-id",
            "kb_release_test_schedule",
            "--freshness-date",
            "2026-06-02",
        ]
    )
    assert rc == 0
    assert _source_hashes(source_dir) == before


def test_schedule_derivation_rejects_unclean_reconciliation(tmp_path: Path) -> None:
    schedule_pack = _write_schedule_pack(tmp_path / "schedule_pack", unclean=True)
    source_dir = _write_source_dir(tmp_path / "sources")

    with pytest.raises(DerivationError, match="Input gate is not clean"):
        main(
            [
                "--schedule-pack",
                str(schedule_pack),
                "--source-dir",
                str(source_dir),
                "--audit-pack",
                str(tmp_path / "audit"),
            ]
        )


def _write_schedule_pack(root: Path, *, unclean: bool = False) -> Path:
    root.mkdir(parents=True)
    comparisons = [
        {
            "status": "совпало",
            "mismatch_reason": "",
            "sheet_cell": "C13",
            "sheet_title": "Мат 5 продв",
            "brand": "Фотон",
            "format": "очно",
            "place": "Верхняя Красносельская, 30",
            "day": "суббота",
            "time": "10:00-12:00",
            "subject": "математика",
            "grade": "5",
            "track": "продвинутая",
            "match_key": "foton|krasnoselskaya|Сб|10:00-12:00|math|5|advanced",
            "tallanto_matches": [
                {
                    "id": "foton-1",
                    "name": "Мат 2026 Красносел 5 кл Сб 10.00-12.00 Продвинутый ФОТОН",
                    "date_start": "2026-09-12",
                    "date_finish": "2027-05-22",
                }
            ],
        },
        {
            "status": "нет в Tallanto" if unclean else "совпало",
            "mismatch_reason": "",
            "sheet_cell": "Z3",
            "sheet_title": "Физ 11 ЕГЭ",
            "brand": "УНПК МФТИ",
            "format": "онлайн",
            "place": "Онлайн",
            "day": "понедельник и среда",
            "time": "18:00-19:30",
            "subject": "физика",
            "grade": "11",
            "track": "ЕГЭ",
            "match_key": "unpk|onlajn_ano|Пн+Ср|18:00-19:30|physics|11|ege",
            "tallanto_matches": [
                {
                    "id": "unpk-1",
                    "name": "Физ 2026 Онлайн 11кл ЕГЭ ПН и СР18:00-19:30 (УНПК)",
                    "date_start": "2026-09-21",
                    "date_finish": "2027-05-26",
                }
            ],
        },
    ]
    summary = {
        "google_schedule_groups": 2,
        "tallanto_regular_groups": 2,
        "matched": 1 if unclean else 2,
        "mismatched_field": 0,
        "missing_in_tallanto": 1 if unclean else 0,
        "duplicate_tallanto_matches": 0,
        "extra_in_tallanto": 0,
        "status_counts": {"совпало": 1, "нет в Tallanto": 1} if unclean else {"совпало": 2},
    }
    (root / "schedule_vs_tallanto_comparison.json").write_text(
        json.dumps({"summary": summary, "comparisons": comparisons}, ensure_ascii=False),
        encoding="utf-8",
    )
    tallanto_rows = [
        {
            "id": "foton-1",
            "filial_key": "krasnoselskaya",
            "day": "Сб",
            "subject": "math",
            "track": "advanced",
            "date_start": "2026-09-12",
            "date_finish": "2027-05-22",
            "raw_record": {"filial": {"krasnoselskaya": "Красносельская"}},
            "match_key": "foton|krasnoselskaya|Сб|10:00-12:00|math|5|advanced",
        },
        {
            "id": "unpk-1",
            "filial_key": "onlajn_ano",
            "day": "Пн+Ср",
            "subject": "physics",
            "track": "ege",
            "date_start": "2026-09-21",
            "date_finish": "2027-05-26",
            "raw_record": {"filial": ""},
            "match_key": "unpk|onlajn_ano|Пн+Ср|18:00-19:30|physics|11|ege",
        },
    ]
    (root / "tallanto_schedule_normalized.json").write_text(
        json.dumps(tallanto_rows, ensure_ascii=False),
        encoding="utf-8",
    )
    return root


def _write_source_dir(root: Path) -> Path:
    facts = root / "facts"
    facts.mkdir(parents=True)
    (facts / "facts_for_bot_FOTON.yaml").write_text(
        "schema_version: test_foton\ncontacts_foton:\n  vk: https://vk.example\n",
        encoding="utf-8",
    )
    (facts / "facts_for_bot_UNPK.yaml").write_text(
        "schema_version: test_unpk\ncontacts_unpk:\n  email: edu@example.test\n",
        encoding="utf-8",
    )
    (root / "release_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "kb_release_manifest_v1",
                "release_id": "base",
                "freshness_check_date": "2026-05-20",
                "control_numbers": {"remove": [], "add": []},
                "required_yaml_paths": [],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return root


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _source_hashes(source_dir: Path) -> dict[str, str]:
    paths = [
        source_dir / "facts" / "facts_for_bot_FOTON.yaml",
        source_dir / "facts" / "facts_for_bot_UNPK.yaml",
        source_dir / "release_manifest.yaml",
    ]
    return {
        str(path.relative_to(source_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
