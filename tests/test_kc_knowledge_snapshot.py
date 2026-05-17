from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.build_kc_knowledge_snapshot import main as build_kc_knowledge_snapshot_cli
from mango_mvp.knowledge_base.fact_registry import (
    FACT_TYPE_PRICE,
    FACT_TYPE_SCHEDULE,
    FRESHNESS_METADATA_ONLY,
    FRESHNESS_NEEDS_MANAGER_CONFIRMATION,
    FRESHNESS_STALE,
    FRESHNESS_UNKNOWN,
    SOURCE_READ_STATUS_METADATA_ONLY,
    SOURCE_READ_STATUS_READ,
    FactSource,
    build_freshness_blocks,
    build_kc_knowledge_snapshot,
    default_google_drive_price_sources,
    extract_docx_sections,
    guard_kc_snapshot_output_root,
    register_google_drive_source,
    register_local_source,
    write_kc_knowledge_snapshot_outputs,
)


def test_docx_snapshot_extracts_sections(tmp_path: Path) -> None:
    docx_path = tmp_path / "kc.docx"
    _write_minimal_docx(
        docx_path,
        [
            ("Title", "Проверка активной сделки лид/клиента"),
            ("", "Перед звонком проверяем активные сделки в АМО и не создаем дубль."),
            ("Heading3", "Расписание"),
            ("", "Не обещаем конкретный слот без подтверждения менеджера."),
        ],
    )

    chunks = extract_docx_sections(docx_path, source_id="source:kc_docx_test")
    snapshot = build_kc_knowledge_snapshot(
        kc_docx_path=docx_path,
        max_docx_sections=10,
        sources=[
            FactSource(
                source_id="source:kc_docx_test",
                title="База знаний КЦ",
                source_kind="local_docx",
                fact_types=("manager_instruction", "schedule"),
                path=str(docx_path),
                freshness_status=FRESHNESS_UNKNOWN,
            )
        ],
    )

    assert [chunk.title for chunk in chunks] == ["Проверка активной сделки лид/клиента", "Расписание"]
    assert "manager_instruction" in chunks[0].fact_types
    assert FACT_TYPE_SCHEDULE in chunks[1].fact_types
    assert snapshot["schema_version"] == "kc_knowledge_snapshot_v1"
    assert len(snapshot["chunks"]) == 2
    assert snapshot["safety"]["send_full_docx_to_prompt"] is False


def test_google_drive_price_docs_registered() -> None:
    sources = default_google_drive_price_sources()

    titles = {source.title for source in sources}
    assert "УНПК Стоимость обучения и порядок оплаты на 26/2027 уч г от 16.03.26" in titles
    assert "ФОТОН Стоимость обучения и порядок оплаты на 26/27 уч от 16.03.26" in titles
    assert all(source.source_kind == "google_drive_doc" for source in sources)
    assert all(FACT_TYPE_PRICE in source.fact_types for source in sources)
    assert all(source.usable_for_precise_answer is False for source in sources)
    assert all(source.metadata["live_access_used"] is False for source in sources)


def test_snapshot_records_source_status_sha256_and_freshness(tmp_path: Path) -> None:
    docx_path = tmp_path / "kc.docx"
    md_path = tmp_path / "kb.md"
    _write_minimal_docx(
        docx_path,
        [
            ("Title", "Оплата"),
            ("", "Не называем точную цену без свежего подтверждения."),
        ],
    )
    md_path.write_text("# Программы\nЕсть очные и онлайн форматы. Точные условия проверяет менеджер.\n", encoding="utf-8")

    sources = [
        register_local_source(
            docx_path,
            title="KC docx",
            fact_types=("price", "manager_instruction"),
            freshness_status=FRESHNESS_UNKNOWN,
        ),
        register_local_source(
            md_path,
            title="KC draft",
            fact_types=("program",),
            freshness_status=FRESHNESS_NEEDS_MANAGER_CONFIRMATION,
        ),
        register_google_drive_source(title="Drive price doc", fact_types=("price",)),
    ]

    snapshot = build_kc_knowledge_snapshot(
        kc_docx_path=docx_path,
        sources=sources,
        run_id="test_snapshot",
        generated_at="2026-05-17T00:00:00+00:00",
        max_docx_sections=5,
        max_chars_per_section=180,
    )

    inventory = {row["title"]: row for row in snapshot["source_inventory"]}
    assert inventory["KC docx"]["read_status"] == SOURCE_READ_STATUS_READ
    assert len(inventory["KC docx"]["sha256"]) == 64
    assert inventory["KC draft"]["freshness_status"] == FRESHNESS_NEEDS_MANAGER_CONFIRMATION
    assert inventory["Drive price doc"]["read_status"] == SOURCE_READ_STATUS_METADATA_ONLY
    assert inventory["Drive price doc"]["freshness_status"] == FRESHNESS_METADATA_ONLY
    assert inventory["Drive price doc"]["usable_for_precise_answer"] is False
    assert len(inventory["Drive price doc"]["sha256"]) == 64
    assert snapshot["run_id"] == "test_snapshot"
    assert snapshot["generated_at"] == "2026-05-17T00:00:00+00:00"
    assert snapshot["mode"] == "read_only"
    assert snapshot["summary"]["sources_with_sha256"] == 3
    assert snapshot["summary"]["metadata_only_sources"] == 1
    assert snapshot["safety"]["stable_runtime_write"] is False
    assert all(len(chunk["text"]) <= 180 for chunk in snapshot["chunks"])


def test_snapshot_writer_outputs_core_files_and_rejects_stable_runtime(tmp_path: Path) -> None:
    docx_path = tmp_path / "kc.docx"
    _write_minimal_docx(docx_path, [("Title", "Расписание"), ("", "Слоты уточняет менеджер.")])
    source = register_local_source(
        docx_path,
        title="KC docx",
        fact_types=("schedule",),
        freshness_status=FRESHNESS_UNKNOWN,
    )
    snapshot = build_kc_knowledge_snapshot(
        kc_docx_path=docx_path,
        sources=[source],
        run_id="writer_test",
        generated_at="2026-05-17T00:00:00+00:00",
    )

    result = write_kc_knowledge_snapshot_outputs(tmp_path / "kb_out", snapshot)

    assert Path(result["snapshot_path"]).exists()
    assert (tmp_path / "kb_out" / "source_inventory.json").exists()
    assert (tmp_path / "kb_out" / "source_inventory.csv").exists()
    assert (tmp_path / "kb_out" / "knowledge_chunks.jsonl").exists()
    saved = json.loads(Path(result["snapshot_path"]).read_text(encoding="utf-8"))
    assert saved["schema_version"] == "kc_knowledge_snapshot_v1"
    with pytest.raises(ValueError, match="stable_runtime"):
        guard_kc_snapshot_output_root(tmp_path / "stable_runtime" / "kb")


def test_build_kc_knowledge_snapshot_cli_writes_snapshot(tmp_path: Path) -> None:
    docx_path = tmp_path / "kc.docx"
    out_root = tmp_path / "product_data" / "knowledge_base" / "cli"
    _write_minimal_docx(docx_path, [("Title", "Документы"), ("", "Точные условия проверяет менеджер.")])

    exit_code = build_kc_knowledge_snapshot_cli(
        [
            "--project-root",
            str(tmp_path),
            "--kc-docx",
            str(docx_path),
            "--out-root",
            str(out_root),
            "--run-id",
            "cli_test",
            "--generated-at",
            "2026-05-17T00:00:00+00:00",
            "--max-docx-sections",
            "5",
        ]
    )

    assert exit_code == 0
    saved = json.loads((out_root / "kc_snapshot_cli_test.json").read_text(encoding="utf-8"))
    assert saved["safety"]["stable_runtime_write"] is False
    assert saved["summary"]["sources_total"] >= 1


def test_fact_without_freshness_blocks_precise_answer() -> None:
    stale_price_source = FactSource(
        source_id="source:price_stale",
        title="Old price document",
        source_kind="local_csv",
        fact_types=("price",),
        path="product_data/question_catalog/old_prices.csv",
        freshness_status=FRESHNESS_STALE,
        usable_for_precise_answer=False,
    )

    blocks = build_freshness_blocks(["prices.current"], [stale_price_source])

    assert blocks == [
        {
            "fact_key": "prices.current",
            "fact_type": "price",
            "reason": "fact_source_not_fresh",
            "blocks_precise_answer": True,
            "safe_instruction": "Do not name exact price, discount, or payment deadline without a fresh verified price source.",
            "candidate_source_ids": ["source:price_stale"],
        }
    ]


def _write_minimal_docx(path: Path, paragraphs: list[tuple[str, str]]) -> None:
    body = "".join(_paragraph_xml(style, text) for style, text in paragraphs)
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}<w:sectPr/></w:body>"
        "</w:document>"
    )
    with ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)


def _paragraph_xml(style: str, text: str) -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{escaped}</w:t></w:r></w:p>"
