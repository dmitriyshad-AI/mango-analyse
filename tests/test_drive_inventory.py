from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mango_mvp.knowledge_base.drive_inventory import (
    FRESHNESS_FRESH_VERIFIED,
    STATUS_EXPORT_FAILED,
    STATUS_MANUAL_REVIEW_REQUIRED,
    STATUS_METADATA_ONLY,
    STATUS_NO_ACCESS,
    STATUS_PROCESSED,
    DriveSourceMetadata,
    DriveTextExport,
    SourceInventoryRecord,
    build_drive_inventory_records,
    build_required_source_inventory,
    default_google_drive_metadata_sources,
    inventory_to_json_payload,
    mark_manual_review_required,
    register_drive_inventory_source,
    register_local_inventory_source,
    sha256_text,
    write_inventory_csv,
    write_inventory_json,
)


def test_local_inventory_record_hashes_file_and_exports_json_csv(tmp_path: Path) -> None:
    source_path = tmp_path / "policy.csv"
    source_path.write_text("theme,answer\nprice,ask manager\n", encoding="utf-8")

    record = register_local_inventory_source(
        "policy.csv",
        base_dir=tmp_path,
        title="Policy CSV",
        fact_types=("rop_policy", "restriction"),
    )

    assert record.inventory_status == STATUS_PROCESSED
    assert record.read_succeeded is True
    assert record.sha256 == sha256_text("theme,answer\nprice,ask manager\n")
    assert record.sha256_source == "file"
    assert record.source_updated_at
    assert record.source_metadata["file_size_bytes"] == source_path.stat().st_size

    json_path = tmp_path / "source_inventory.json"
    csv_path = tmp_path / "source_inventory.csv"
    write_inventory_json([record], json_path)
    write_inventory_csv([record], csv_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "kc_source_inventory_v1"
    assert payload["summary"]["status_counts"][STATUS_PROCESSED] == 1
    assert payload["records"][0]["fact_types"] == ["rop_policy", "restriction"]

    with csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["inventory_status"] == STATUS_PROCESSED
    assert rows[0]["read_succeeded"] == "true"
    assert json.loads(rows[0]["source_metadata_json"])["required_path"] == "policy.csv"


def test_local_inventory_can_hash_extracted_text_instead_of_binary_file(tmp_path: Path) -> None:
    source_path = tmp_path / "exported.docx"
    source_path.write_bytes(b"not a real docx")

    record = register_local_inventory_source(
        source_path,
        title="Exported DOCX",
        fact_types=("documents",),
        extracted_text="Извлеченный текст для контроля изменений",
    )

    assert record.inventory_status == STATUS_PROCESSED
    assert record.sha256 == sha256_text("Извлеченный текст для контроля изменений")
    assert record.sha256_source == "text"


def test_missing_local_source_is_no_access_and_never_precise(tmp_path: Path) -> None:
    record = register_local_inventory_source(
        "missing.xlsx",
        base_dir=tmp_path,
        title="Missing approved questionnaire",
        fact_types=("rop_policy",),
    )

    assert record.inventory_status == STATUS_NO_ACCESS
    assert record.read_succeeded is False
    assert record.usable_for_precise_answer is False
    assert "missing" in record.limitation_reason.lower() or "unavailable" in record.limitation_reason.lower()


def test_drive_metadata_defaults_to_metadata_only_without_text_export() -> None:
    record = register_drive_inventory_source(
        {
            "id": "drive-1",
            "name": "УНПК Стоимость обучения",
            "mimeType": "application/vnd.google-apps.document",
            "modifiedTime": "2026-03-16T10:00:00Z",
            "webViewLink": "https://drive.google.com/file/d/drive-1/view",
            "fact_types": ("price", "payment_methods"),
        }
    )

    assert record.inventory_status == STATUS_METADATA_ONLY
    assert record.read_succeeded is False
    assert record.google_drive_file_id == "drive-1"
    assert record.google_drive_mime_type == "application/vnd.google-apps.document"
    assert record.sha256 is None
    assert record.usable_for_precise_answer is False
    assert "text was not exported" in record.limitation_reason


def test_drive_text_export_hashes_text_but_does_not_unlock_precise_answer_by_default() -> None:
    record = register_drive_inventory_source(
        DriveSourceMetadata(title="ФОТОН Стоимость обучения", file_id="drive-2", fact_types=("price",)),
        exported_text="Цена указана в документе, но требует проверки РОПом.",
    )

    assert record.inventory_status == STATUS_PROCESSED
    assert record.read_succeeded is True
    assert record.sha256 == sha256_text("Цена указана в документе, но требует проверки РОПом.")
    assert record.sha256_source == "text"
    assert record.freshness_status == "needs_manager_confirmation"
    assert record.usable_for_precise_answer is False


def test_precise_answer_requires_processed_fresh_verified_sha256_source() -> None:
    with pytest.raises(ValueError, match="fresh_verified"):
        register_drive_inventory_source(
            DriveSourceMetadata(title="Price doc", file_id="drive-3", fact_types=("price",)),
            exported_text="42 000 рублей",
            usable_for_precise_answer=True,
        )

    record = register_drive_inventory_source(
        DriveSourceMetadata(title="Verified price doc", file_id="drive-4", fact_types=("price",)),
        exported_text="42 000 рублей",
        freshness_status=FRESHNESS_FRESH_VERIFIED,
        usable_for_precise_answer=True,
        limitation_reason="",
    )

    assert record.usable_for_precise_answer is True
    assert record.inventory_status == STATUS_PROCESSED


def test_drive_inventory_hook_maps_no_access_export_failed_and_manual_review() -> None:
    items = [
        DriveSourceMetadata(title="No access doc", file_id="no-access"),
        DriveSourceMetadata(title="Broken export doc", file_id="broken"),
        DriveSourceMetadata(title="Needs review doc", file_id="review"),
    ]

    def hook(metadata: DriveSourceMetadata) -> DriveTextExport:
        if metadata.file_id == "no-access":
            raise PermissionError("403")
        if metadata.file_id == "broken":
            raise RuntimeError("cannot export")
        return DriveTextExport(
            text="Есть текст, но в нем неоднозначные цены.",
            status=STATUS_MANUAL_REVIEW_REQUIRED,
            status_reason="ambiguous_prices",
            limitation_reason="requires ROP confirmation",
        )

    records = build_drive_inventory_records(items, export_text_hook=hook)

    assert [record.inventory_status for record in records] == [
        STATUS_NO_ACCESS,
        STATUS_EXPORT_FAILED,
        STATUS_MANUAL_REVIEW_REQUIRED,
    ]
    assert records[0].status_reason == "google_drive_permission_denied"
    assert records[1].status_reason == "google_drive_export_failed"
    assert records[2].read_succeeded is True
    assert records[2].sha256 == sha256_text("Есть текст, но в нем неоднозначные цены.")
    assert records[2].usable_for_precise_answer is False


def test_manual_review_marker_preserves_source_metadata_and_disables_precise_answer(tmp_path: Path) -> None:
    source_path = tmp_path / "answers.csv"
    source_path.write_text("answer\nhistorical\n", encoding="utf-8")
    record = register_local_inventory_source(
        source_path,
        title="Historical manager answers",
        fact_types=("manager_answer_pattern",),
    )

    marked = mark_manual_review_required(record, "contains historical answers; not a fact source")

    assert marked.inventory_status == STATUS_MANUAL_REVIEW_REQUIRED
    assert marked.read_succeeded is True
    assert marked.usable_for_precise_answer is False
    assert marked.source_metadata["file_size_bytes"] == source_path.stat().st_size
    assert marked.limitation_reason == "contains historical answers; not a fact source"


def test_required_inventory_registers_local_sources_and_drive_price_docs_as_metadata_only(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "product_data/question_catalog").mkdir(parents=True)
    (tmp_path / "База знаний КЦ.docx").write_bytes(b"docx placeholder")
    (tmp_path / "docs/KC_BOT_KNOWLEDGE_BASE_DRAFT_2026-05-13.md").write_text("draft", encoding="utf-8")

    records = build_required_source_inventory(base_dir=tmp_path)
    titles = {record.title: record for record in records}

    assert titles["База знаний КЦ"].inventory_status == STATUS_PROCESSED
    assert titles["KC bot knowledge base draft"].inventory_status == STATUS_PROCESSED
    assert titles["ROP bot policy questionnaire v2 CSV"].inventory_status == STATUS_NO_ACCESS
    assert titles["УНПК Стоимость обучения и порядок оплаты на 26/2027 уч г от 16.03.26"].inventory_status == STATUS_METADATA_ONLY
    assert titles["ФОТОН Стоимость обучения и порядок оплаты на 26/27 уч от 16.03.26"].inventory_status == STATUS_METADATA_ONLY
    assert all(record.usable_for_precise_answer is False for record in records)


def test_inventory_summary_counts_all_required_statuses() -> None:
    records = [
        SourceInventoryRecord(
            source_id=f"source:test:{status}",
            title=f"Source {status}",
            source_type="local_file",
            path=f"{status}.txt",
            inventory_status=status,
            read_succeeded=status == STATUS_PROCESSED,
            fact_types=("unknown",),
            sha256=sha256_text(status) if status == STATUS_PROCESSED else None,
            sha256_source="text" if status == STATUS_PROCESSED else None,
        )
        for status in (
            STATUS_PROCESSED,
            STATUS_NO_ACCESS,
            STATUS_EXPORT_FAILED,
            STATUS_METADATA_ONLY,
            STATUS_MANUAL_REVIEW_REQUIRED,
        )
    ]

    payload = inventory_to_json_payload(records)

    assert payload["summary"]["status_counts"] == {
        STATUS_PROCESSED: 1,
        STATUS_NO_ACCESS: 1,
        STATUS_EXPORT_FAILED: 1,
        STATUS_METADATA_ONLY: 1,
        STATUS_MANUAL_REVIEW_REQUIRED: 1,
    }


def test_default_google_drive_metadata_sources_do_not_claim_live_access() -> None:
    records = build_drive_inventory_records(default_google_drive_metadata_sources())

    assert len(records) == 3
    assert all(record.inventory_status == STATUS_METADATA_ONLY for record in records)
    assert all(record.source_metadata["live_access_used"] is False for record in records)
