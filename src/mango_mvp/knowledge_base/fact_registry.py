from __future__ import annotations

import hashlib
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zipfile import ZipFile
from xml.etree import ElementTree


KC_KNOWLEDGE_SNAPSHOT_SCHEMA_VERSION = "kc_knowledge_snapshot_v1"
FACT_REGISTRY_SCHEMA_VERSION = "kc_fact_registry_v1"
KC_KNOWLEDGE_SNAPSHOT_BUILDER_VERSION = "kc_knowledge_snapshot_builder_2026_05_17_a1"
DEFAULT_KC_SNAPSHOT_RUN_ID = "20260517_night_v1"
DEFAULT_KC_SNAPSHOT_OUT_ROOT = Path("product_data/knowledge_base/kb_night_20260517_v1")

DEFAULT_KC_DOCX_PATH = Path("База знаний КЦ.docx")
DEFAULT_KC_BOT_KNOWLEDGE_BASE_DRAFT_PATH = Path("docs/KC_BOT_KNOWLEDGE_BASE_DRAFT_2026-05-13.md")
DEFAULT_DRIVE_KC_KNOWLEDGE_BASE_BUILD_PLAN_PATH = Path("docs/DRIVE_KC_KNOWLEDGE_BASE_BUILD_PLAN_2026-05-13.md")
DEFAULT_ROP_POLICY_CSV_PATH = Path("product_data/question_catalog/rop_bot_policy_questionnaire_v2_2026-05-15.csv")
DEFAULT_APPROVED_ROP_POLICY_XLSX_PATH = Path("product_data/question_catalog/rop_bot_policy_questionnaire_APPROVED_2026-05-15.xlsx")
DEFAULT_CUSTOMER_QUESTION_CLASSES_CSV_PATH = Path("product_data/question_catalog/customer_question_classes.csv")
DEFAULT_CUSTOMER_QUESTION_ITEMS_JSONL_PATH = Path("product_data/question_catalog/customer_question_items.jsonl")
DEFAULT_CURRENT_FACT_SOURCE_REGISTRY_PATH = Path("product_data/question_catalog/current_fact_source_registry.json")
DEFAULT_FACT_REQUIREMENTS_CSV_PATH = Path("product_data/question_catalog/fact_requirements.csv")

DEFAULT_GOOGLE_DRIVE_PRICE_FOLDER_TITLE = "Google Drive: актуальные документы по ценам 2026/2027"
DEFAULT_GOOGLE_DRIVE_PRICE_DOC_TITLES = (
    "УНПК Стоимость обучения и порядок оплаты на 26/2027 уч г от 16.03.26",
    "ФОТОН Стоимость обучения и порядок оплаты на 26/27 уч от 16.03.26",
)
DEFAULT_SNAPSHOT_REQUIRED_FACT_KEYS = (
    "prices.current",
    "schedule.current",
    "documents.current",
    "discount.current",
    "payment_methods.current",
    "program.current",
)

FRESHNESS_FRESH = "fresh"
FRESHNESS_FRESH_VERIFIED = "fresh_verified"
FRESHNESS_DOCUMENT_VERIFIED = "document_verified"
FRESHNESS_NEEDS_MANAGER_CONFIRMATION = "needs_manager_confirmation"
FRESHNESS_UNKNOWN = "unknown"
FRESHNESS_STALE = "stale"
FRESHNESS_STALE_OR_CONFLICTING = "stale_or_conflicting"
FRESHNESS_METADATA_ONLY = "metadata_only"
FRESHNESS_MISSING = "missing"
FRESHNESS_INTERNAL_ONLY = "internal_only"
FRESHNESS_DO_NOT_USE = "do_not_use"

PRECISE_FRESHNESS_STATUSES = frozenset({FRESHNESS_FRESH, FRESHNESS_FRESH_VERIFIED, FRESHNESS_DOCUMENT_VERIFIED})

SOURCE_READ_STATUS_READ = "read"
SOURCE_READ_STATUS_MISSING = "missing"
SOURCE_READ_STATUS_METADATA_ONLY = "metadata_only"
SOURCE_READ_STATUS_UNREADABLE = "unreadable"
SOURCE_READ_STATUS_UNKNOWN = "unknown"

SOURCE_KIND_LOCAL_DOCX = "local_docx"
SOURCE_KIND_LOCAL_CSV = "local_csv"
SOURCE_KIND_LOCAL_JSON = "local_json"
SOURCE_KIND_LOCAL_JSONL = "local_jsonl"
SOURCE_KIND_LOCAL_MD = "local_markdown"
SOURCE_KIND_LOCAL_TEXT = "local_text"
SOURCE_KIND_LOCAL_XLSX = "local_xlsx"
SOURCE_KIND_GOOGLE_DRIVE_DOC = "google_drive_doc"

TEXT_CHUNK_SOURCE_KINDS = frozenset({SOURCE_KIND_LOCAL_DOCX, SOURCE_KIND_LOCAL_MD, SOURCE_KIND_LOCAL_TEXT})

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
    read_status: str = SOURCE_READ_STATUS_UNKNOWN
    freshness_status: str = FRESHNESS_UNKNOWN
    usable_for_precise_answer: bool = False
    status_reason: str = ""
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
        object.__setattr__(self, "read_status", _stable_key(self.read_status))
        object.__setattr__(self, "freshness_status", _stable_key(self.freshness_status))
        object.__setattr__(self, "status_reason", _clean_text(self.status_reason))
        object.__setattr__(self, "notes", _clean_text(self.notes))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.title:
            raise ValueError("FactSource.title must not be empty")
        if not self.fact_types:
            raise ValueError("FactSource.fact_types must not be empty")
        if not self.path and not self.google_drive_url:
            raise ValueError("FactSource requires path or google_drive_url")
        if self.usable_for_precise_answer and self.freshness_status not in PRECISE_FRESHNESS_STATUSES:
            raise ValueError("usable_for_precise_answer requires fresh verified freshness_status")

    @property
    def is_fresh(self) -> bool:
        return self.freshness_status in PRECISE_FRESHNESS_STATUSES and self.usable_for_precise_answer

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
    read_status = SOURCE_READ_STATUS_MISSING
    status_reason = "local_source_missing"
    sha256_value: str | None = None
    updated_at = last_updated_at
    if source_path.exists():
        try:
            sha256_value = sha256_file(source_path)
            read_status = SOURCE_READ_STATUS_READ
            status_reason = "local_source_readable"
            updated_at = updated_at or _file_mtime_iso(source_path)
        except OSError as exc:
            read_status = SOURCE_READ_STATUS_UNREADABLE
            status_reason = f"local_source_unreadable:{type(exc).__name__}"
    return FactSource(
        source_id=source_id or _source_id_from_parts(kind, source_title, str(source_path)),
        title=source_title,
        source_kind=kind,
        fact_types=tuple(fact_types),
        path=str(source_path),
        last_updated_at=updated_at,
        sha256=sha256_value,
        read_status=read_status,
        freshness_status=freshness_status if source_path.exists() else FRESHNESS_MISSING,
        usable_for_precise_answer=usable_for_precise_answer,
        status_reason=status_reason,
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
    metadata_payload = {
        "title": title,
        "google_drive_url": ref,
        "last_updated_at": last_updated_at,
        "fact_types": list(normalize_fact_types(fact_types)),
        "freshness_status": freshness_status,
    }
    merged_metadata = {
        "metadata_only": True,
        "live_access_used": False,
        "sha256_scope": "metadata_record",
        **dict(metadata or {}),
    }
    return FactSource(
        source_id=source_id or _source_id_from_parts(SOURCE_KIND_GOOGLE_DRIVE_DOC, title, ref),
        title=title,
        source_kind=SOURCE_KIND_GOOGLE_DRIVE_DOC,
        fact_types=tuple(fact_types),
        google_drive_url=ref,
        last_updated_at=last_updated_at,
        sha256=sha256 or sha256_text(json.dumps(metadata_payload, ensure_ascii=False, sort_keys=True)),
        read_status=SOURCE_READ_STATUS_METADATA_ONLY,
        freshness_status=freshness_status,
        usable_for_precise_answer=usable_for_precise_answer,
        status_reason="google_drive_registered_metadata_only",
        notes=notes,
        metadata=merged_metadata,
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
    kc_bot_knowledge_base_draft_path: str | Path = DEFAULT_KC_BOT_KNOWLEDGE_BASE_DRAFT_PATH,
    drive_kc_knowledge_base_build_plan_path: str | Path = DEFAULT_DRIVE_KC_KNOWLEDGE_BASE_BUILD_PLAN_PATH,
    rop_policy_csv_path: str | Path = DEFAULT_ROP_POLICY_CSV_PATH,
    approved_rop_policy_xlsx_path: str | Path = DEFAULT_APPROVED_ROP_POLICY_XLSX_PATH,
    question_classes_path: str | Path = DEFAULT_CUSTOMER_QUESTION_CLASSES_CSV_PATH,
    question_items_path: str | Path = DEFAULT_CUSTOMER_QUESTION_ITEMS_JSONL_PATH,
    current_fact_source_registry_path: str | Path = DEFAULT_CURRENT_FACT_SOURCE_REGISTRY_PATH,
    fact_requirements_path: str | Path = DEFAULT_FACT_REQUIREMENTS_CSV_PATH,
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
            kc_bot_knowledge_base_draft_path,
            title="KC bot knowledge base draft 2026-05-13",
            fact_types=(
                FACT_TYPE_MANAGER_INSTRUCTION,
                FACT_TYPE_RESTRICTION,
                FACT_TYPE_DOCUMENTS,
                FACT_TYPE_PAYMENT_METHODS,
                FACT_TYPE_PROGRAM,
            ),
            notes="Local draft with KC bot knowledge boundaries; read-only source for short chunks.",
        ),
        register_local_source(
            drive_kc_knowledge_base_build_plan_path,
            title="Drive KC knowledge base build plan 2026-05-13",
            fact_types=(FACT_TYPE_MANAGER_INSTRUCTION, FACT_TYPE_RESTRICTION, FACT_TYPE_PRICE, FACT_TYPE_DOCUMENTS),
            notes="Local plan for Google Drive knowledge processing; not a fresh fact source by itself.",
        ),
        register_local_source(
            rop_policy_csv_path,
            title="ROP bot policy questionnaire v2 approved CSV",
            fact_types=(FACT_TYPE_ROP_POLICY, FACT_TYPE_RESTRICTION),
            notes="Canonical ROP routing and allowed-answer policy for Telegram pilot context.",
        ),
        register_local_source(
            approved_rop_policy_xlsx_path,
            title="ROP bot policy questionnaire approved XLSX",
            fact_types=(FACT_TYPE_ROP_POLICY, FACT_TYPE_RESTRICTION),
            notes="Approved ROP policy workbook. Used for inventory and sha256; CSV remains the text-readable source.",
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
        register_local_source(
            current_fact_source_registry_path,
            title="Current fact source registry",
            fact_types=(
                FACT_TYPE_PRICE,
                FACT_TYPE_SCHEDULE,
                FACT_TYPE_DOCUMENTS,
                FACT_TYPE_PAYMENT_METHODS,
                FACT_TYPE_PROGRAM,
                FACT_TYPE_QUESTION_CATALOG,
            ),
            notes="Question catalog fact source registry. Entries still require manual freshness confirmation.",
        ),
        register_local_source(
            fact_requirements_path,
            title="Question catalog fact requirements",
            fact_types=(FACT_TYPE_QUESTION_CATALOG, FACT_TYPE_RESTRICTION),
            notes="Fact requirements and fallback rules for question classes.",
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


def extract_text_sections(
    path: str | Path,
    *,
    source_id: str | None = None,
    max_sections: int | None = None,
    max_chars_per_section: int = 700,
) -> list[KnowledgeChunk]:
    source_path = Path(path)
    text = source_path.read_text(encoding="utf-8-sig", errors="replace")
    doc_source_id = source_id or _source_id_from_parts(_source_kind_for_path(source_path), source_path.name, str(path))
    sections: list[KnowledgeChunk] = []
    current_title = source_path.stem
    current_parts: list[str] = []
    current_start_line = 1

    def flush() -> None:
        nonlocal current_parts, current_title, current_start_line
        cleaned = _clean_text(" ".join(current_parts))
        if not cleaned:
            current_parts = []
            return
        section_text = _truncate_text(cleaned, max_chars_per_section)
        chunk_payload = {"source_id": doc_source_id, "title": current_title, "text": section_text}
        sections.append(
            KnowledgeChunk(
                chunk_id=f"kc_chunk:{_digest_json(chunk_payload)[:20]}",
                source_id=doc_source_id,
                title=current_title,
                text=section_text,
                fact_types=classify_fact_types(f"{current_title} {section_text}"),
                metadata={"start_line": current_start_line},
            )
        )
        current_parts = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = _clean_text(raw_line)
        if not line:
            continue
        heading_match = re.match(r"^#{1,6}\s+(.+)$", line)
        if heading_match:
            flush()
            current_title = _clean_text(heading_match.group(1))
            current_start_line = line_no
            continue
        current_parts.append(line)
        if len(_clean_text(" ".join(current_parts))) >= max_chars_per_section:
            flush()
            current_title = source_path.stem
            current_start_line = line_no + 1
        if max_sections is not None and len(sections) >= max_sections:
            return sections[:max_sections]
    flush()
    if max_sections is not None:
        return sections[:max_sections]
    return sections


def build_kc_knowledge_snapshot(
    *,
    kc_docx_path: str | Path = DEFAULT_KC_DOCX_PATH,
    max_docx_sections: int = 80,
    max_chars_per_section: int = 700,
    sources: Sequence[FactSource] | None = None,
    run_id: str = DEFAULT_KC_SNAPSHOT_RUN_ID,
    generated_at: str | None = None,
    project_root: str | Path | None = None,
    required_fact_keys: Sequence[str] = DEFAULT_SNAPSHOT_REQUIRED_FACT_KEYS,
) -> dict[str, Any]:
    project_root_path = Path(project_root).expanduser().resolve() if project_root is not None else None
    if sources is not None:
        registry_sources = list(sources)
    else:
        registry_sources = build_default_kc_fact_registry(
            kc_docx_path=_resolve_path(kc_docx_path, project_root=project_root_path),
            kc_bot_knowledge_base_draft_path=_resolve_path(
                DEFAULT_KC_BOT_KNOWLEDGE_BASE_DRAFT_PATH, project_root=project_root_path
            ),
            drive_kc_knowledge_base_build_plan_path=_resolve_path(
                DEFAULT_DRIVE_KC_KNOWLEDGE_BASE_BUILD_PLAN_PATH, project_root=project_root_path
            ),
            rop_policy_csv_path=_resolve_path(DEFAULT_ROP_POLICY_CSV_PATH, project_root=project_root_path),
            approved_rop_policy_xlsx_path=_resolve_path(
                DEFAULT_APPROVED_ROP_POLICY_XLSX_PATH, project_root=project_root_path
            ),
            question_classes_path=_resolve_path(
                DEFAULT_CUSTOMER_QUESTION_CLASSES_CSV_PATH, project_root=project_root_path
            ),
            question_items_path=_resolve_path(DEFAULT_CUSTOMER_QUESTION_ITEMS_JSONL_PATH, project_root=project_root_path),
            current_fact_source_registry_path=_resolve_path(
                DEFAULT_CURRENT_FACT_SOURCE_REGISTRY_PATH, project_root=project_root_path
            ),
            fact_requirements_path=_resolve_path(DEFAULT_FACT_REQUIREMENTS_CSV_PATH, project_root=project_root_path),
        )
    registry_sources = [_with_inferred_local_read_status(source, project_root=project_root_path) for source in registry_sources]
    chunks, chunk_errors = build_knowledge_chunks_for_sources(
        registry_sources,
        project_root=project_root_path,
        max_docx_sections=max_docx_sections,
        max_text_sections_per_source=12,
        max_chars_per_section=max_chars_per_section,
    )
    freshness_blocks = build_freshness_blocks(required_fact_keys, registry_sources)
    source_inventory = build_source_inventory(registry_sources)
    summary = build_snapshot_summary(
        sources=registry_sources,
        chunks=chunks,
        freshness_blocks=freshness_blocks,
        chunk_errors=chunk_errors,
    )
    return {
        "schema_version": KC_KNOWLEDGE_SNAPSHOT_SCHEMA_VERSION,
        "fact_registry_schema_version": FACT_REGISTRY_SCHEMA_VERSION,
        "builder_version": KC_KNOWLEDGE_SNAPSHOT_BUILDER_VERSION,
        "run_id": _clean_text(run_id) or DEFAULT_KC_SNAPSHOT_RUN_ID,
        "generated_at": generated_at or _now_iso(),
        "mode": "read_only",
        "metadata": {
            "google_drive_price_folder_title": DEFAULT_GOOGLE_DRIVE_PRICE_FOLDER_TITLE,
            "google_drive_live_access_used": False,
            "docx_parsed_for_sections": any(chunk.metadata.get("source_kind") == SOURCE_KIND_LOCAL_DOCX for chunk in chunks),
            "max_docx_sections": max_docx_sections,
            "max_chars_per_section": max_chars_per_section,
            "project_root": str(project_root_path) if project_root_path else None,
        },
        "sources": [source.to_json_dict() for source in registry_sources],
        "source_inventory": source_inventory,
        "facts": [],
        "chunks": [chunk.to_json_dict() for chunk in chunks],
        "knowledge_chunks": [chunk.to_json_dict() for chunk in chunks],
        "manager_answer_patterns": [],
        "freshness_blocks": freshness_blocks,
        "conflicts": chunk_errors,
        "summary": summary,
        "safety": {
            "google_drive_write": False,
            "crm_write": False,
            "tallanto_write": False,
            "client_send": False,
            "stable_runtime_write": False,
            "send_full_docx_to_prompt": False,
            "prices_require_fresh_verified_extract": True,
            "schedule_is_not_ready_for_precise_answer_by_default": True,
        },
    }


def build_knowledge_chunks_for_sources(
    sources: Sequence[FactSource | Mapping[str, Any]],
    *,
    project_root: str | Path | None = None,
    max_docx_sections: int = 80,
    max_text_sections_per_source: int = 12,
    max_chars_per_section: int = 700,
) -> tuple[list[KnowledgeChunk], list[dict[str, Any]]]:
    root = Path(project_root).expanduser().resolve() if project_root is not None else None
    chunks: list[KnowledgeChunk] = []
    errors: list[dict[str, Any]] = []
    for raw_source in sources:
        source = _coerce_source(raw_source)
        if source.source_kind not in TEXT_CHUNK_SOURCE_KINDS:
            continue
        if not source.path:
            continue
        source_path = _resolve_path(source.path, project_root=root)
        if source.read_status != SOURCE_READ_STATUS_READ:
            if not (source.read_status == SOURCE_READ_STATUS_UNKNOWN and source_path.exists()):
                continue
        if not source_path.exists():
            continue
        try:
            if source.source_kind == SOURCE_KIND_LOCAL_DOCX:
                extracted = extract_docx_sections(
                    source_path,
                    source_id=source.source_id,
                    max_sections=max_docx_sections,
                    max_chars_per_section=max_chars_per_section,
                )
            else:
                extracted = extract_text_sections(
                    source_path,
                    source_id=source.source_id,
                    max_sections=max_text_sections_per_source,
                    max_chars_per_section=max_chars_per_section,
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "source_id": source.source_id,
                    "source_title": source.title,
                    "reason": "chunk_extraction_failed",
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
            )
            continue
        for chunk in extracted:
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk.chunk_id,
                    source_id=chunk.source_id,
                    title=chunk.title,
                    text=chunk.text,
                    fact_types=chunk.fact_types,
                    freshness_status=source.freshness_status,
                    metadata={
                        **dict(chunk.metadata),
                        "source_title": source.title,
                        "source_kind": source.source_kind,
                        "source_read_status": source.read_status,
                        "source_sha256": source.sha256,
                        "prompt_eligible": True,
                    },
                )
            )
    return chunks, errors


def build_source_inventory(sources: Sequence[FactSource | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_source_inventory_row(_coerce_source(source)) for source in sources]


def build_snapshot_summary(
    *,
    sources: Sequence[FactSource | Mapping[str, Any]],
    chunks: Sequence[KnowledgeChunk | Mapping[str, Any]],
    freshness_blocks: Sequence[Mapping[str, Any]],
    chunk_errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    normalized_sources = [_coerce_source(source) for source in sources]
    normalized_chunks = [_coerce_chunk_like(chunk) for chunk in chunks]
    return {
        "sources_total": len(normalized_sources),
        "sources_by_read_status": _count_by(normalized_sources, lambda source: source.read_status),
        "sources_by_freshness_status": _count_by(normalized_sources, lambda source: source.freshness_status),
        "sources_with_sha256": sum(1 for source in normalized_sources if source.sha256),
        "metadata_only_sources": sum(1 for source in normalized_sources if source.read_status == SOURCE_READ_STATUS_METADATA_ONLY),
        "precise_answer_sources": sum(1 for source in normalized_sources if source.is_fresh),
        "chunks_total": len(normalized_chunks),
        "chunks_by_freshness_status": _count_by(normalized_chunks, lambda chunk: chunk.freshness_status),
        "freshness_blocks_total": len(freshness_blocks),
        "chunk_extraction_errors": len(chunk_errors),
    }


def guard_kc_snapshot_output_root(path: str | Path, *, project_root: str | Path | None = None) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute() and project_root is not None:
        resolved = Path(project_root).expanduser() / resolved
    resolved = resolved.resolve()
    if any(part.casefold() == "stable_runtime" for part in resolved.parts):
        raise ValueError(f"KC knowledge snapshot output must not be under stable_runtime: {resolved}")
    return resolved


def write_kc_knowledge_snapshot_outputs(
    out_root: str | Path,
    snapshot: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    output_root = guard_kc_snapshot_output_root(out_root, project_root=project_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = _stable_key(snapshot.get("run_id") or DEFAULT_KC_SNAPSHOT_RUN_ID)
    snapshot_path = output_root / f"kc_snapshot_{run_id}.json"
    source_inventory = list(snapshot.get("source_inventory") or [])
    chunks = list(snapshot.get("chunks") or [])
    _write_json(snapshot_path, snapshot)
    _write_json(output_root / "source_inventory.json", source_inventory)
    _write_csv(output_root / "source_inventory.csv", [_flatten_inventory_row(row) for row in source_inventory])
    _write_jsonl(output_root / "knowledge_chunks.jsonl", chunks)
    _write_csv(output_root / "knowledge_chunks.csv", [_flatten_chunk_row(row) for row in chunks])
    _write_json(output_root / "quality_summary.json", snapshot.get("summary") or {})
    return {
        "out_root": str(output_root),
        "snapshot_path": str(snapshot_path),
        "source_inventory_json": str(output_root / "source_inventory.json"),
        "source_inventory_csv": str(output_root / "source_inventory.csv"),
        "knowledge_chunks_jsonl": str(output_root / "knowledge_chunks.jsonl"),
        "knowledge_chunks_csv": str(output_root / "knowledge_chunks.csv"),
        "quality_summary_json": str(output_root / "quality_summary.json"),
        "sources_total": len(source_inventory),
        "chunks_total": len(chunks),
        "safety": dict(snapshot.get("safety") or {}),
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
        read_status=str(source.get("read_status") or SOURCE_READ_STATUS_UNKNOWN),
        freshness_status=str(source.get("freshness_status") or FRESHNESS_UNKNOWN),
        usable_for_precise_answer=bool(source.get("usable_for_precise_answer")),
        status_reason=str(source.get("status_reason") or ""),
        notes=str(source.get("notes") or ""),
        metadata=source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {},
    )


def _with_inferred_local_read_status(source: FactSource, *, project_root: Path | None) -> FactSource:
    if source.read_status != SOURCE_READ_STATUS_UNKNOWN or not source.path:
        return source
    source_path = _resolve_path(source.path, project_root=project_root)
    if not source_path.exists():
        return FactSource(
            source_id=source.source_id,
            title=source.title,
            source_kind=source.source_kind,
            fact_types=source.fact_types,
            path=source.path,
            google_drive_url=source.google_drive_url,
            last_updated_at=source.last_updated_at,
            sha256=source.sha256,
            read_status=SOURCE_READ_STATUS_MISSING,
            freshness_status=FRESHNESS_MISSING,
            usable_for_precise_answer=False,
            status_reason=source.status_reason or "local_source_missing",
            notes=source.notes,
            metadata=source.metadata,
        )
    try:
        sha256_value = source.sha256 or sha256_file(source_path)
        last_updated_at = source.last_updated_at or _file_mtime_iso(source_path)
        read_status = SOURCE_READ_STATUS_READ
        status_reason = source.status_reason or "local_source_readable"
    except OSError as exc:
        sha256_value = source.sha256
        last_updated_at = source.last_updated_at
        read_status = SOURCE_READ_STATUS_UNREADABLE
        status_reason = source.status_reason or f"local_source_unreadable:{type(exc).__name__}"
    return FactSource(
        source_id=source.source_id,
        title=source.title,
        source_kind=source.source_kind,
        fact_types=source.fact_types,
        path=source.path,
        google_drive_url=source.google_drive_url,
        last_updated_at=last_updated_at,
        sha256=sha256_value,
        read_status=read_status,
        freshness_status=source.freshness_status,
        usable_for_precise_answer=source.usable_for_precise_answer,
        status_reason=status_reason,
        notes=source.notes,
        metadata=source.metadata,
    )


def _coerce_chunk_like(chunk: KnowledgeChunk | Mapping[str, Any]) -> KnowledgeChunk:
    if isinstance(chunk, KnowledgeChunk):
        return chunk
    return KnowledgeChunk(
        chunk_id=str(chunk.get("chunk_id") or "kc_chunk:manual"),
        source_id=str(chunk.get("source_id") or "source:manual"),
        title=str(chunk.get("title") or "Без названия"),
        text=str(chunk.get("text") or ""),
        fact_types=tuple(chunk.get("fact_types") or ()),
        freshness_status=str(chunk.get("freshness_status") or FRESHNESS_UNKNOWN),
        metadata=chunk.get("metadata") if isinstance(chunk.get("metadata"), Mapping) else {},
    )


def _source_inventory_row(source: FactSource) -> dict[str, Any]:
    return {
        "source_id": source.source_id,
        "title": source.title,
        "source_kind": source.source_kind,
        "path": source.path,
        "google_drive_url": source.google_drive_url,
        "last_updated_at": source.last_updated_at,
        "sha256": source.sha256,
        "read_status": source.read_status,
        "freshness_status": source.freshness_status,
        "fact_types": list(source.fact_types),
        "usable_for_precise_answer": source.usable_for_precise_answer,
        "status_reason": source.status_reason,
        "notes": source.notes,
        "metadata": dict(source.metadata),
    }


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
    if suffix in {".md", ".markdown"}:
        return SOURCE_KIND_LOCAL_MD
    if suffix == ".txt":
        return SOURCE_KIND_LOCAL_TEXT
    if suffix == ".csv":
        return SOURCE_KIND_LOCAL_CSV
    if suffix == ".json":
        return SOURCE_KIND_LOCAL_JSON
    if suffix == ".jsonl":
        return SOURCE_KIND_LOCAL_JSONL
    if suffix == ".xlsx":
        return SOURCE_KIND_LOCAL_XLSX
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


def _resolve_path(path: str | Path, *, project_root: Path | None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and project_root is not None:
        candidate = project_root / candidate
    return candidate.resolve()


def _file_mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_by(items: Iterable[Any], key_fn: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(key_fn(item) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _flatten_inventory_row(row: Mapping[str, Any]) -> dict[str, Any]:
    flattened = dict(row)
    flattened["fact_types"] = ";".join(str(item) for item in row.get("fact_types") or [])
    flattened["metadata"] = json.dumps(row.get("metadata") or {}, ensure_ascii=False, sort_keys=True)
    return flattened


def _flatten_chunk_row(row: Mapping[str, Any]) -> dict[str, Any]:
    flattened = dict(row)
    flattened["fact_types"] = ";".join(str(item) for item in row.get("fact_types") or [])
    flattened["metadata"] = json.dumps(row.get("metadata") or {}, ensure_ascii=False, sort_keys=True)
    return flattened


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _csv_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = (
        "source_id",
        "chunk_id",
        "title",
        "source_kind",
        "source_id",
        "path",
        "google_drive_url",
        "last_updated_at",
        "sha256",
        "read_status",
        "freshness_status",
        "fact_types",
        "usable_for_precise_answer",
        "status_reason",
        "text",
        "notes",
        "metadata",
    )
    seen: list[str] = []
    for name in preferred:
        if any(name in row for row in rows) and name not in seen:
            seen.append(name)
    for row in rows:
        for name in row:
            if name not in seen:
                seen.append(name)
    return seen or ["empty"]
