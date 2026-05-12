from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.demo_tenant import build_demo_tenant_product_root
from mango_mvp.productization.sanitized_real_demo import build_sanitized_real_demo_root
from scripts import mango_office_sanitized_real_demo


def test_sanitized_real_demo_copies_real_structure_without_sensitive_values(tmp_path: Path) -> None:
    source_root = tmp_path / "source_product_appliance"
    build_demo_tenant_product_root(product_root=source_root, replace_existing=True)
    source_db = source_root / "mango_product_appliance.sqlite"
    demo_root = tmp_path / "sanitized_real_demo"

    report = build_sanitized_real_demo_root(
        source_product_root=source_root,
        source_product_db_path=source_db,
        demo_product_root=demo_root,
        replace_existing=True,
    )

    demo_db = demo_root / "mango_product_appliance.sqlite"
    with sqlite3.connect(str(demo_db)) as con:
        calls = con.execute("select provider_call_id, source_filename, manager_display_name from product_calls order by provider_call_id").fetchall()
        inbox = con.execute("select provider_call_id, client_phone, manager_ref, recording_url from capture_inbox_items order by provider_call_id").fetchall()
    serialized = json.dumps(report, ensure_ascii=False) + demo_db.read_bytes().decode("latin1", errors="ignore")

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["product_calls"] == 4
    assert report["summary"]["capture_inbox_items"] == 4
    assert calls[0][0].startswith("CALL-")
    assert calls[0][1].endswith(".mp3")
    assert inbox[0][1].startswith("+7999000")
    assert inbox[0][3] is None
    assert "CALL-DEMO-1" not in serialized
    assert "Анна Demo" not in serialized
    assert report["safety"]["contains_real_personal_data"] is False
    assert report["safety"]["reads_runtime_db"] is False


def test_sanitized_real_demo_cli_writes_report(tmp_path: Path) -> None:
    source_root = tmp_path / "source_product_appliance"
    build_demo_tenant_product_root(product_root=source_root, replace_existing=True)
    source_db = source_root / "mango_product_appliance.sqlite"
    demo_root = tmp_path / "sanitized_real_demo"
    out = demo_root / "sanitized_real_demo_report.json"

    rc = mango_office_sanitized_real_demo.main(
        [
            "--source-product-root",
            str(source_root),
            "--source-product-db",
            str(source_db),
            "--demo-product-root",
            str(demo_root),
            "--out",
            str(out),
            "--replace",
        ]
    )

    saved = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert saved["summary"]["product_calls"] == 4
    assert saved["safety"]["stable_runtime_writes"] is False


def test_sanitized_real_demo_refuses_runtime_or_stable_runtime_source(tmp_path: Path) -> None:
    source_root = tmp_path / "stable_runtime"
    source_root.mkdir()
    source_db = source_root / "mango_product_appliance.sqlite"
    source_db.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="stable_runtime"):
        build_sanitized_real_demo_root(
            source_product_root=source_root,
            source_product_db_path=source_db,
            demo_product_root=tmp_path / "demo",
        )

    product_root = tmp_path / "product_appliance"
    product_root.mkdir()
    runtime_db = product_root / "mango_mvp.db"
    runtime_db.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime DB"):
        build_sanitized_real_demo_root(
            source_product_root=product_root,
            source_product_db_path=runtime_db,
            demo_product_root=tmp_path / "demo2",
        )


def test_sanitized_real_demo_refuses_overlapping_source_and_demo_roots(tmp_path: Path) -> None:
    source_root = tmp_path / "source_product_appliance"
    build_demo_tenant_product_root(product_root=source_root, replace_existing=True)
    source_db = source_root / "mango_product_appliance.sqlite"

    with pytest.raises(ValueError, match="separate"):
        build_sanitized_real_demo_root(
            source_product_root=source_root,
            source_product_db_path=source_db,
            demo_product_root=source_root,
            replace_existing=True,
        )
    assert source_db.exists()

    with pytest.raises(ValueError, match="inside source product root"):
        build_sanitized_real_demo_root(
            source_product_root=source_root,
            source_product_db_path=source_db,
            demo_product_root=source_root / "nested_demo",
            replace_existing=True,
        )


def test_sanitized_real_demo_skips_extra_source_columns(tmp_path: Path) -> None:
    source_root = tmp_path / "source_product_appliance"
    build_demo_tenant_product_root(product_root=source_root, replace_existing=True)
    source_db = source_root / "mango_product_appliance.sqlite"
    with sqlite3.connect(str(source_db)) as con:
        con.execute("ALTER TABLE product_calls ADD COLUMN future_private_note TEXT")
        con.execute("UPDATE product_calls SET future_private_note = 'real client private note'")
        con.commit()

    report = build_sanitized_real_demo_root(
        source_product_root=source_root,
        source_product_db_path=source_db,
        demo_product_root=tmp_path / "sanitized_real_demo",
        replace_existing=True,
    )
    serialized = json.dumps(report, ensure_ascii=False)

    assert report["copy"]["skipped_extra_source_columns"]["product_calls"] == ["future_private_note"]
    assert "real client private note" not in serialized


def test_sanitized_real_demo_sanitizes_amo_and_tallanto_snapshots(tmp_path: Path) -> None:
    source_root = tmp_path / "source_product_appliance"
    build_demo_tenant_product_root(product_root=source_root, replace_existing=True)
    snapshot_dir = source_root / "crm_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "amocrm_entities.json").write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "crm_provider": "amocrm",
                        "entity_type": "lead",
                        "entity_id": "501",
                        "entity_name": "ООО Реальный клиент",
                        "phones": ["+79001234567"],
                        "owner_id": "77",
                        "owner_name": "Иван Реальный",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (snapshot_dir / "tallanto_entities.csv").write_text(
        "entity_id,entity_type,entity_name,phone,owner_id,owner_name\n"
        "901,student,Мария Реальная,+79007654321,88,Ольга Реальная\n",
        encoding="utf-8",
    )

    report = build_sanitized_real_demo_root(
        source_product_root=source_root,
        source_product_db_path=source_root / "mango_product_appliance.sqlite",
        demo_product_root=tmp_path / "sanitized_real_demo",
        replace_existing=True,
    )

    amo = json.loads((tmp_path / "sanitized_real_demo" / "crm_snapshots" / "amocrm_entities.json").read_text(encoding="utf-8"))
    tallanto = json.loads((tmp_path / "sanitized_real_demo" / "crm_snapshots" / "tallanto_entities.json").read_text(encoding="utf-8"))
    serialized = json.dumps({"report": report, "amo": amo, "tallanto": tallanto}, ensure_ascii=False)

    assert report["summary"]["snapshots_written"] == 2
    assert amo["entities"][0]["entity_id"].startswith("CRM-")
    assert tallanto["entities"][0]["phones"][0].startswith("+7999000")
    assert "ООО Реальный клиент" not in serialized
    assert "Мария Реальная" not in serialized
    assert "+79001234567" not in serialized
