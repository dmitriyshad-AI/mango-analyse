from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zipfile import ZipFile
from xml.etree import ElementTree


KC_KNOWLEDGE_SNAPSHOT_SCHEMA_VERSION = "kc_knowledge_snapshot_v1"
FACT_REGISTRY_SCHEMA_VERSION = "kc_fact_registry_v1"

DEFAULT_KC_DOCX_PATH = Path("База знаний КЦ.docx")
DEFAULT_ROP_POLICY_CSV_PATH = Path("product_data/question_catalog/rop_bot_policy_questionnaire_v2_2026-05-15.csv")
DEFAULT_CUSTOMER_QUESTION_CLASSES_CSV_PATH = Path("product_data/question_catalog/customer_question_classes.csv")
DEFAULT_CUSTOMER_QUESTION_ITEMS_JSONL_PATH = Path("product_data/question_catalog/customer_question_items.jsonl")

DEFAULT_GOOGLE_DRIVE_PRICE_FOLDER_TITLE = "Google Drive: актуальные документы по ценам 2026/2027"
DEFAULT_GOOGLE_DRIVE_PRICE_DOC_TITLES = (
    "УНПК Стоимость обучения и порядок оплаты на 26/2027 уч г от 16.03.26",
    "ФОТОН Стоимость обучения и порядок оплаты на 26/27 уч от 16.03.26",
)

FRESHNESS_FRESH = "fresh"
FRESHNESS_UNKNOWN = "unknown"
FRESHNESS_STALE = "stale"
FRESHNESS_METADATA_ONLY = "metadata_only"
FRESHNESS_MISSING = "missing"

SOURCE_KIND_LOCAL_DOCX = "local_docx"
SOURCE_KIND_LOCAL_CSV = "local_csv"
SOURCE_KIND_LOCAL_JSONL = "local_jsonl"
SOURCE_KIND_GOOGLE_DRIVE_DOC = "google_drive_doc"

FACT_TYPE_ROP_POLICY = "rop_policy"
FACT_TYPE_PRICE = "price"
FACT_TYPE_SCHEDULE = "schedule"
FACT_TYPE_DOCUMENTS = "documents"
FACT_TYPE_PAYMENT_METHODS = "payment_methods"
FACT_TYPE_DISCOUNT = "discount"
FACT_TYPE_RESTRICTION = "restriction"
FACT_TYPE_MANAGER_INSTRUCTION = "manager_instruction"
FACT_TYPE_PROGRAM = "program"
FACT_TYPE_QUESTION_CATALOG = "question_catalog"

_FACT_KEY_ALIASES = {
    "prices": FACT_TYPE_PRICE,
    "price": FACT_TYPE_PRICE,
    "pricing": FACT_TYPE_PRICE,
    "schedule": FACT_TYPE_SCHEDULE,
    "schedules": FACT_TYPE_SCHEDULE,
    "documents": FACT_TYPE_DOCUMENTS,
    "document": FACT_TYPE_DOCUMENTS,
    "contract": FACT_TYPE_DOCUMENTS,
    "contracts": FACT_TYPE_DOCUMENTS,
    "payment_methods": FACT_TYPE_PAYMENT_METHODS,
    "payment_method": FACT_TYPE_PAYMENT_METHODS,
    "payment": FACT_TYPE_PAYMENT_METHODS,
    "discounts": FACT_TYPE_DISCOUNT,
    "discount": FACT_TYPE_DISCOUNT,
    "installment_terms": "installment",
    "installment": "installment",
    "matkap_procedure": FACT_TYPE_DOCUMENTS,
    "matkap_documents": FACT_TYPE_DOCUMENTS,
    "tax_deduction_procedure": FACT_TYPE_DOCUMENTS,
    "license_documents": FACT_TYPE_DOCUMENTS,
    "refund_policy": FACT_TYPE_DOCUMENTS,
    "programs": FACT_TYPE_PROGRAM,
    "program": FACT_TYPE_PROGRAM,
    "formats": FACT_TYPE_PROGRAM,
    "materials": FACT_TYPE_PROGRAM,
    "teachers": FACT_TYPE_PROGRAM,
    "age_levels": FACT_TYPE_PROGRAM,
    "trial_class": FACT_TYPE_PROGRAM,
    "addresses": "location",
    "location": "location",
    "camp": FACT_TYPE_PROGRAM,
    "camp_logistics": FACT_TYPE_PROGRAM,
    "transport": "location",
}

_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


@dataclass(frozen=True)
class FactSource:
    source_id: str
    title: str
    source_kind: str
    fact_types: tuple[str, ...]
    path: str | None = None
    google_drive_url: str | None = None
    last_updated_at: str | None = None
    sha256: str | None = None
    freshness_status: str = FRESHNESS_UNKNOWN
    usable_for_precise_answer: bool = False
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _stable_key(self.source_id))
        object.__setattr__(self, "title", _clean_text(self.title))
        object.__setattr__(self, "source_kind", _stable_key(self.source_kind))
        object.__setattr__(self, "fact_types", normalize_fact_types(self.fact_types))
        object.__setattr__(self, "path", _optional_text(self.path))
        object.__setattr__(self, "google_drive_url", _optional_text(self.google_drive_url))
        object.__setattr__(self, "last_updated_at", _optional_text(self.last_updated_at))
        object.__setattr__(self, "sha256", _optional_text(self.sha256))
        object.__setattr__(self, "freshness_status", _stable_key(self.freshness_status))
        object.__setattr__(self, "notes", _clean_text(self.notes))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.title:
            raise ValueError("FactSource.title must not be empty")
        if not self.fact_types:
            raise ValueError("FactSource.fact_types must not be empty")
        if not self.path and not self.google_drive_url:
            raise ValueError("FactSource requires path or google_drive_url")
        if self.usable_for_precise_answer and self.freshness_status != FRESHNESS_FRESH:
            raise ValueError("usable_for_precise_answer requires freshness_status='fresh'")

    @property
    def is_fresh(self) -> bool:
        return self.freshness_status == FRESHNESS_FRESH and self.usable_for_precise_answer

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fact_types"] = list(self.fact_types)
        return payload


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source_id: str
    title: str
    text: str
    fact_types: tuple[str, ...] = ()
    freshness_status: str = FRESHNESS_UNKNOWN
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "chunk_id", _stable_key(self.chunk_id))
        object.__setattr__(self, "source_id", _stable_key(self.source_id))
        object.__setattr__(self, "title", _clean_text(self.title))
        object.__setattr__(self, "text", _clean_text(self.text))
        object.__setattr__(self, "fact_types", normalize_fact_types(self.fact_types))
        object.__setattr__(self, "freshness_status", _stable_key(self.freshness_status))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.text:
            raise ValueError("KnowledgeChunk.text must not be empty")

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fact_types"] = list(self.fact_types)
        return payload


def register_local_source(
    path: str | Path,
    *,
    title: str | None = None,
    source_id: str | None = None,
    fact_types: Sequence[str],
    source_kind: str | None = None,
    last_updated_at: str | None = None,
    freshness_status: str = FRESHNESS_UNKNOWN,
    usable_for_precise_answer: bool = False,
    notes: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> FactSource:
    source_path = Path(path)
    kind = source_kind or _source_kind_for_path(source_path)
    source_title = title or source_path.name
    return FactSource(
        source_id=source_id or _source_id_from_parts(kind, source_title, str(source_path)),
        title=source_title,
        source_kind=kind,
        fact_types=tuple(fact_types),
        path=str(source_path),
        last_updated_at=last_updated_at,
        sha256=sha256_file(source_path) if source_path.exists() else None,
        freshness_status=freshness_status if source_path.exists() else FRESHNESS_MISSING,
        usable_for_precise_answer=usable_for_precise_answer,
        notes=notes,
        metadata=metadata or {},
    )


def register_google_drive_source(
    *,
    title: str,
    fact_types: Sequence[str],
    google_drive_url: str | None = None,
    source_id: str | None = None,
    last_updated_at: str | None = None,
    sha256: str | None = None,
    freshness_status: str = FRESHNESS_METADATA_ONLY,
    usable_for_precise_answer: bool = False,
    notes: str = "Registered as metadata only; use precise facts only after verified local extraction.",
    metadata: Mapping[str, Any] | None = None,
) -> FactSource:
    ref = google_drive_url or f"google-drive://manual-registration/{_stable_key(title)}"
    return FactSource(
        source_id=source_id or _source_id_from_parts(SOURCE_KIND_GOOGLE_DRIVE_DOC, title, ref),
        title=title,
        source_kind=SOURCE_KIND_GOOGLE_DRIVE_DOC,
        fact_types=tuple(fact_types),
        google_drive_url=ref,
        last_updated_at=last_updated_at,
        sha256=sha256,
        freshness_status=freshness_status,
        usable_for_precise_answer=usable_for_precise_answer,
        notes=notes,
        metadata=metadata or {},
    )


def default_google_drive_price_sources() -> list[FactSource]:
    return [
        register_google_drive_source(
            title=title,
            fact_types=(FACT_TYPE_PRICE, FACT_TYPE_PAYMENT_METHODS),
            metadata={"folder_title": DEFAULT_GOOGLE_DRIVE_PRICE_FOLDER_TITLE, "live_access_used": False},
        )
        for title in DEFAULT_GOOGLE_DRIVE_PRICE_DOC_TITLES
    ]


def build_default_kc_fact_registry(
    *,
    kc_docx_path: str | Path = DEFAULT_KC_DOCX_PATH,
    rop_policy_csv_path: str | Path = DEFAULT_ROP_POLICY_CSV_PATH,
    question_classes_path: str | Path = DEFAULT_CUSTOMER_QUESTION_CLASSES_CSV_PATH,
    question_items_path: str | Path = DEFAULT_CUSTOMER_QUESTION_ITEMS_JSONL_PATH,
) -> list[FactSource]:
    sources: list[FactSource] = [
        register_local_source(
            kc_docx_path,
            title="База знаний КЦ",
            fact_types=(
                FACT_TYPE_MANAGER_INSTRUCTION,
                FACT_TYPE_RESTRICTION,
                FACT_TYPE_PRICE,
                FACT_TYPE_SCHEDULE,
                FACT_TYPE_DOCUMENTS,
                FACT_TYPE_PAYMENT_METHODS,
                FACT_TYPE_PROGRAM,
            ),
            notes="Local KC knowledge base. Sections are chunked before prompt use; the whole docx must not be sent to LLM.",
        ),
        register_local_source(
            rop_policy_csv_path,
            title="ROP bot policy questionnaire v2 approved CSV",
            fact_types=(FACT_TYPE_ROP_POLICY, FACT_TYPE_RESTRICTION),
            notes="Canonical ROP routing and allowed-answer policy for Telegram pilot context.",
        ),
        register_local_source(
            question_classes_path,
            title="Customer question classes",
            fact_types=(FACT_TYPE_QUESTION_CATALOG,),
            notes="Question class taxonomy for matching a new customer message to policy and fact requirements.",
        ),
        register_local_source(
            question_items_path,
            title="Customer question items",
            fact_types=(FACT_TYPE_QUESTION_CATALOG,),
            notes="Historical redacted question examples. Use only short relevant fragments in prompt context.",
        ),
    ]
    sources.extend(default_google_drive_price_sources())
    return sources


def extract_docx_sections(
    path: str | Path,
    *,
    source_id: str | None = None,
    max_sections: int | None = None,
    max_chars_per_section: int = 1600,
) -> list[KnowledgeChunk]:
    blocks = _extract_docx_blocks(Path(path))
    doc_source_id = source_id or _source_id_from_parts(SOURCE_KIND_LOCAL_DOCX, Path(path).name, str(path))
    sections: list[KnowledgeChunk] = []
    current_title = Path(path).stem
    current_parts: list[str] = []
    current_start_index = 0

    def flush() -> None:
        nonlocal current_parts, current_title, current_start_index
        text = _clean_text(" ".join(current_parts))
        if not text:
            current_parts = []
            return
        section_text = _truncate_text(text, max_chars_per_section)
        chunk_payload = {"source_id": doc_source_id, "title": current_title, "text": section_text}
        sections.append(
            KnowledgeChunk(
                chunk_id=f"kc_chunk:{_digest_json(chunk_payload)[:20]}",
                source_id=doc_source_id,
                title=current_title,
                text=section_text,
                fact_types=classify_fact_types(f"{current_title} {section_text}"),
                metadata={"start_block_index": current_start_index},
            )
        )
        current_parts = []

    for block in blocks:
        style = block.get("style", "").lower()
        text = block["text"]
        is_heading = style.startswith("title") or style.startswith("heading")
        if is_heading:
            flush()
            current_title = text
            current_start_index = int(block["block_index"])
            continue
        current_parts.append(text)
    flush()
    if max_sections is not None:
        return sections[:max_sections]
    return sections


def build_kc_knowledge_snapshot(
    *,
    kc_docx_path: str | Path = DEFAULT_KC_DOCX_PATH,
    max_docx_sections: int = 80,
    max_chars_per_section: int = 1600,
    sources: Sequence[FactSource] | None = None,
) -> dict[str, Any]:
    registry_sources = list(sources) if sources is not None else build_default_kc_fact_registry(kc_docx_path=kc_docx_path)
    docx_source = next((source for source in registry_sources if source.source_kind == SOURCE_KIND_LOCAL_DOCX), None)
    chunks: list[KnowledgeChunk] = []
    if docx_source and docx_source.path and Path(docx_source.path).exists():
        chunks = extract_docx_sections(
            docx_source.path,
            source_id=docx_source.source_id,
            max_sections=max_docx_sections,
            max_chars_per_section=max_chars_per_section,
        )
    return {
        "schema_version": KC_KNOWLEDGE_SNAPSHOT_SCHEMA_VERSION,
        "fact_registry_schema_version": FACT_REGISTRY_SCHEMA_VERSION,
        "metadata": {
            "google_drive_price_folder_title": DEFAULT_GOOGLE_DRIVE_PRICE_FOLDER_TITLE,
            "google_drive_live_access_used": False,
            "docx_parsed_for_sections": bool(chunks),
            "max_docx_sections": max_docx_sections,
            "max_chars_per_section": max_chars_per_section,
        },
        "sources": [source.to_json_dict() for source in registry_sources],
        "chunks": [chunk.to_json_dict() for chunk in chunks],
        "safety": {
            "send_full_docx_to_prompt": False,
            "prices_require_fresh_verified_extract": True,
            "schedule_is_not_ready_for_precise_answer_by_default": True,
        },
    }


def build_freshness_blocks(required_fact_keys: Sequence[str], sources: Sequence[FactSource | Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized_sources = [_coerce_source(source) for source in sources]
    blocks: list[dict[str, Any]] = []
    for fact_key in _dedupe(_clean_text(key) for key in required_fact_keys if _clean_text(key)):
        fact_type = fact_type_from_key(fact_key)
        candidates = [source for source in normalized_sources if fact_type in source.fact_types]
        fresh_candidates = [source for source in candidates if source.is_fresh]
        if fresh_candidates:
            continue
        reason = "fact_source_missing" if not candidates else "fact_source_not_fresh"
        if fact_type == FACT_TYPE_SCHEDULE:
            reason = "schedule_source_not_ready"
        blocks.append(
            {
                "fact_key": fact_key,
                "fact_type": fact_type,
                "reason": reason,
                "blocks_precise_answer": True,
                "safe_instruction": _safe_instruction_for_fact_type(fact_type),
                "candidate_source_ids": [source.source_id for source in candidates],
            }
        )
    return blocks


def is_precise_answer_allowed(required_fact_keys: Sequence[str], sources: Sequence[FactSource | Mapping[str, Any]]) -> bool:
    return not build_freshness_blocks(required_fact_keys, sources)


def classify_fact_types(text: str) -> tuple[str, ...]:
    lower = text.lower()
    detected: list[str] = []
    keyword_groups = (
        (FACT_TYPE_ROP_POLICY, ("роп", "решение роп", "правило эскалации")),
        (FACT_TYPE_PRICE, ("стоим", "цена", "цены", "прайс", "руб", "оплата до")),
        (FACT_TYPE_SCHEDULE, ("распис", "слот", "дата", "время занятий", "день недели", "занятие")),
        (FACT_TYPE_DOCUMENTS, ("договор", "справ", "документ", "квитанц", "налог", "маткап", "чек")),
        (FACT_TYPE_PAYMENT_METHODS, ("оплат", "сбп", "реквизит", "карт", "платеж")),
        (FACT_TYPE_DISCOUNT, ("скид", "промокод", "акци")),
        (FACT_TYPE_MANAGER_INSTRUCTION, ("амо", "талланто", "менеджер", "история общения", "лид", "сделк")),
        (FACT_TYPE_RESTRICTION, ("нельзя", "не обещ", "обязательно", "важно", "только менеджер")),
        (FACT_TYPE_PROGRAM, ("курс", "интенсив", "летняя школа", "лш", "лвш", "программа", "предмет")),
    )
    for fact_type, keywords in keyword_groups:
        if any(keyword in lower for keyword in keywords):
            detected.append(fact_type)
    return tuple(dict.fromkeys(detected)) or (FACT_TYPE_MANAGER_INSTRUCTION,)


def fact_type_from_key(fact_key: str) -> str:
    base = _clean_text(fact_key).lower().split(".", 1)[0]
    return _FACT_KEY_ALIASES.get(base, _stable_key(base))


def normalize_fact_types(fact_types: Sequence[str] | None) -> tuple[str, ...]:
    if not fact_types:
        return ()
    return tuple(_dedupe(fact_type_from_key(item) for item in fact_types if _clean_text(item)))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_docx_blocks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(document_xml)
    body = root.find("w:body", _W_NS)
    if body is None:
        return []
    blocks: list[dict[str, Any]] = []
    for block_index, child in enumerate(list(body)):
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = _clean_text("".join(node.text or "" for node in child.findall(".//w:t", _W_NS)))
            if not text:
                continue
            style_node = child.find("./w:pPr/w:pStyle", _W_NS)
            style = style_node.attrib.get(f"{{{_W_NS['w']}}}val", "") if style_node is not None else ""
            blocks.append({"block_index": block_index, "block_type": "paragraph", "style": style, "text": text})
        elif tag == "tbl":
            cell_texts = [
                _clean_text("".join(node.text or "" for node in cell.findall(".//w:t", _W_NS)))
                for cell in child.findall(".//w:tc", _W_NS)
            ]
            text = _clean_text(" | ".join(item for item in cell_texts if item))
            if text:
                blocks.append({"block_index": block_index, "block_type": "table", "style": "table", "text": text})
    return blocks


def _coerce_source(source: FactSource | Mapping[str, Any]) -> FactSource:
    if isinstance(source, FactSource):
        return source
    return FactSource(
        source_id=str(source.get("source_id") or ""),
        title=str(source.get("title") or source.get("path") or source.get("google_drive_url") or ""),
        source_kind=str(source.get("source_kind") or "unknown"),
        fact_types=tuple(source.get("fact_types") or ()),
        path=source.get("path"),
        google_drive_url=source.get("google_drive_url"),
        last_updated_at=source.get("last_updated_at"),
        sha256=source.get("sha256"),
        freshness_status=str(source.get("freshness_status") or FRESHNESS_UNKNOWN),
        usable_for_precise_answer=bool(source.get("usable_for_precise_answer")),
        notes=str(source.get("notes") or ""),
        metadata=source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {},
    )


def _safe_instruction_for_fact_type(fact_type: str) -> str:
    if fact_type == FACT_TYPE_SCHEDULE:
        return "Do not name exact group time or lesson date. Use the schedule-safe template and ask manager to follow up."
    if fact_type == FACT_TYPE_PRICE:
        return "Do not name exact price, discount, or payment deadline without a fresh verified price source."
    if fact_type == FACT_TYPE_DOCUMENTS:
        return "Do not cite exact document, refund, tax, or legal terms without a fresh verified document source."
    return "Do not provide precise dynamic facts until the source is fresh and verified."


def _source_kind_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return SOURCE_KIND_LOCAL_DOCX
    if suffix == ".csv":
        return SOURCE_KIND_LOCAL_CSV
    if suffix == ".jsonl":
        return SOURCE_KIND_LOCAL_JSONL
    return "local_file"


def _source_id_from_parts(kind: str, title: str, ref: str) -> str:
    return f"source:{_stable_key(kind)}:{_digest_json({'title': title, 'ref': ref})[:16]}"


def _digest_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stable_key(value: Any) -> str:
    text = _clean_text(value).lower()
    text = text.replace("ё", "е")
    text = re.sub(r"[^a-z0-9а-я_.:-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        raise ValueError("stable key must not be empty")
    return text


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split())


def _optional_text(value: Any) -> str | None:
    text = _clean_text(value)
    return text or None


def _truncate_text(text: str, limit: int) -> str:
    cleaned = _clean_text(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)].rstrip() + "…"


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result
