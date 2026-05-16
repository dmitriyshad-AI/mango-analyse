from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from mango_mvp.knowledge_base.fact_registry import (
    FACT_TYPE_PRICE,
    FACT_TYPE_SCHEDULE,
    FRESHNESS_STALE,
    FRESHNESS_UNKNOWN,
    FactSource,
    build_freshness_blocks,
    build_kc_knowledge_snapshot,
    default_google_drive_price_sources,
    extract_docx_sections,
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
