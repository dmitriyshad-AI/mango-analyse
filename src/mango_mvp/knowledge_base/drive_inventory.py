from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, Union


DRIVE_INVENTORY_SCHEMA_VERSION = "kc_drive_inventory_v1"

STATUS_PROCESSED = "processed"
STATUS_NO_ACCESS = "no_access"
STATUS_EXPORT_FAILED = "export_failed"
STATUS_METADATA_ONLY = "metadata_only"
STATUS_MANUAL_REVIEW_REQUIRED = "manual_review_required"
STATUS_NEEDS_OCR = "needs_ocr"
STATUS_NOT_RELEVANT = "not_relevant"

ALLOWED_STATUSES = {
    STATUS_PROCESSED,
    STATUS_NO_ACCESS,
    STATUS_EXPORT_FAILED,
    STATUS_METADATA_ONLY,
    STATUS_MANUAL_REVIEW_REQUIRED,
    STATUS_NEEDS_OCR,
    STATUS_NOT_RELEVANT,
}


@dataclass(frozen=True)
class DriveInventoryRecord:
    source_id: str
    title: str
    mime_type: str
    url: str
    path: str = ""
    drive_file_id: str = ""
    created_time: str = ""
    modified_time: str = ""
    processing_status: str = STATUS_METADATA_ONLY
    fact_types: tuple[str, ...] = ()
    contains_personal_data: bool = False
    freshness_status: str = "metadata_only"
    approval_status: str = "not_approved"
    usable_for_precise_answer: bool = False
    sha256_text: str = ""
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = str(self.processing_status or STATUS_METADATA_ONLY).strip()
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Unknown Drive processing status: {status}")
        object.__setattr__(self, "processing_status", status)
        object.__setattr__(self, "source_id", stable_source_id(self.source_id or self.title or self.url))
        object.__setattr__(self, "title", clean_text(self.title))
        object.__setattr__(self, "mime_type", clean_text(self.mime_type))
        object.__setattr__(self, "url", clean_text(self.url))
        object.__setattr__(self, "path", clean_text(self.path))
        object.__setattr__(self, "drive_file_id", clean_text(self.drive_file_id))
        object.__setattr__(self, "created_time", clean_text(self.created_time))
        object.__setattr__(self, "modified_time", clean_text(self.modified_time))
        object.__setattr__(self, "freshness_status", clean_text(self.freshness_status) or "metadata_only")
        object.__setattr__(self, "approval_status", clean_text(self.approval_status) or "not_approved")
        object.__setattr__(self, "sha256_text", clean_text(self.sha256_text))
        object.__setattr__(self, "notes", clean_text(self.notes))
        object.__setattr__(self, "fact_types", tuple(dedupe(clean_text(item) for item in self.fact_types if clean_text(item))))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.title:
            raise ValueError("DriveInventoryRecord.title must not be empty")
        if not self.url and not self.drive_file_id:
            raise ValueError("DriveInventoryRecord requires url or drive_file_id")
        if self.usable_for_precise_answer and (
            self.processing_status != STATUS_PROCESSED
            or self.freshness_status != "fresh"
            or self.approval_status not in {"approved", "approved_for_client"}
        ):
            raise ValueError("usable_for_precise_answer requires processed + fresh + approved source")

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fact_types"] = list(self.fact_types)
        payload["metadata"] = dict(self.metadata)
        return payload


def records_from_drive_items(items: Sequence[Mapping[str, Any]], *, parent_path: str = "") -> list[DriveInventoryRecord]:
    records: list[DriveInventoryRecord] = []
    for item in items:
        title = str(item.get("title") or item.get("name") or "").strip()
        if not title:
            continue
        file_id = str(item.get("id") or item.get("fileId") or "").strip()
        url = str(item.get("url") or item.get("webViewLink") or "").strip()
        mime_type = str(item.get("mime_type") or item.get("mimeType") or "").strip()
        path = "/".join(part for part in (parent_path.strip("/"), title) if part)
        records.append(
            DriveInventoryRecord(
                source_id=f"drive:{file_id or title}",
                title=title,
                mime_type=mime_type,
                url=url,
                path=path,
                drive_file_id=file_id,
                created_time=str(item.get("created_time") or item.get("createdTime") or ""),
                modified_time=str(item.get("modified_time") or item.get("modifiedTime") or ""),
                processing_status=STATUS_METADATA_ONLY,
                fact_types=infer_fact_types_from_title(title),
                freshness_status=infer_freshness_from_title(title),
                approval_status="not_approved",
                usable_for_precise_answer=False,
                notes="Drive item discovered read-only. Content must be exported and approved before precise answers.",
            )
        )
    return records


def build_inventory_payload(records: Sequence[DriveInventoryRecord | Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [coerce_record(record) for record in records]
    by_status: dict[str, int] = {}
    for record in normalized:
        by_status[record.processing_status] = by_status.get(record.processing_status, 0) + 1
    return {
        "schema_version": DRIVE_INVENTORY_SCHEMA_VERSION,
        "records": [record.to_json_dict() for record in normalized],
        "summary": {
            "records_total": len(normalized),
            "by_processing_status": by_status,
            "usable_for_precise_answer": sum(1 for record in normalized if record.usable_for_precise_answer),
            "metadata_only": sum(1 for record in normalized if record.processing_status == STATUS_METADATA_ONLY),
        },
    }


def write_inventory(records: Sequence[DriveInventoryRecord | Mapping[str, Any]], *, out_dir: str | Path) -> dict[str, Any]:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = build_inventory_payload(records)
    json_path = target / "source_inventory.json"
    csv_path = target / "source_inventory.csv"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_records_csv(payload["records"], csv_path)
    return payload


def write_records_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    fieldnames = [
        "source_id",
        "title",
        "mime_type",
        "url",
        "path",
        "drive_file_id",
        "created_time",
        "modified_time",
        "processing_status",
        "fact_types",
        "contains_personal_data",
        "freshness_status",
        "approval_status",
        "usable_for_precise_answer",
        "sha256_text",
        "notes",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fieldnames}
            row["fact_types"] = "|".join(record.get("fact_types") or [])
            writer.writerow(row)


def coerce_record(record: DriveInventoryRecord | Mapping[str, Any]) -> DriveInventoryRecord:
    if isinstance(record, DriveInventoryRecord):
        return record
    return DriveInventoryRecord(
        source_id=str(record.get("source_id") or record.get("id") or record.get("title") or ""),
        title=str(record.get("title") or ""),
        mime_type=str(record.get("mime_type") or ""),
        url=str(record.get("url") or ""),
        path=str(record.get("path") or ""),
        drive_file_id=str(record.get("drive_file_id") or record.get("id") or ""),
        created_time=str(record.get("created_time") or ""),
        modified_time=str(record.get("modified_time") or ""),
        processing_status=str(record.get("processing_status") or STATUS_METADATA_ONLY),
        fact_types=tuple(record.get("fact_types") or ()),
        contains_personal_data=bool(record.get("contains_personal_data")),
        freshness_status=str(record.get("freshness_status") or "metadata_only"),
        approval_status=str(record.get("approval_status") or "not_approved"),
        usable_for_precise_answer=bool(record.get("usable_for_precise_answer")),
        sha256_text=str(record.get("sha256_text") or ""),
        notes=str(record.get("notes") or ""),
        metadata=record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {},
    )


def infer_fact_types_from_title(title: str) -> tuple[str, ...]:
    text = title.casefold().replace("ё", "е")
    fact_types: list[str] = []
    if any(marker in text for marker in ("стоим", "цена", "оплат", "прайс")):
        fact_types.extend(["price", "payment_methods"])
    if any(marker in text for marker in ("распис", "групп")):
        fact_types.append("schedule")
    if any(marker in text for marker in ("договор", "лиценз", "справ", "возврат", "доверен")):
        fact_types.append("documents")
    if any(marker in text for marker in ("лвш", "лш", "летн", "курс", "онлайн", "школ")):
        fact_types.append("program")
    if any(marker in text for marker in ("скрипт", "call", "кц", "обзвон")):
        fact_types.append("manager_instruction")
    return tuple(dedupe(fact_types or ["manager_instruction"]))


def infer_freshness_from_title(title: str) -> str:
    text = title.casefold()
    if "26/27" in text or "2026" in text or "26-27" in text:
        return "unknown"
    if "2025" in text or "25/26" in text or "24-25" in text or "2024" in text or "2023" in text:
        return "stale"
    return "metadata_only"


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def stable_source_id(value: str) -> str:
    text = clean_text(value).casefold().replace("ё", "е")
    cleaned = "".join(char if char.isalnum() else "_" for char in text).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"source:drive:{cleaned[:60] or 'item'}:{digest}"


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split())


def dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


SOURCE_INVENTORY_SCHEMA_VERSION = "kc_source_inventory_v1"

INVENTORY_STATUSES = (
    STATUS_PROCESSED,
    STATUS_NO_ACCESS,
    STATUS_EXPORT_FAILED,
    STATUS_METADATA_ONLY,
    STATUS_MANUAL_REVIEW_REQUIRED,
)

SHA256_SOURCE_FILE = "file"
SHA256_SOURCE_TEXT = "text"

FRESHNESS_FRESH_VERIFIED = "fresh_verified"
FRESHNESS_UNKNOWN = "unknown"
FRESHNESS_METADATA_ONLY = "metadata_only"
FRESHNESS_NEEDS_MANAGER_CONFIRMATION = "needs_manager_confirmation"

SOURCE_TYPE_LOCAL_DOCX = "local_docx"
SOURCE_TYPE_LOCAL_MARKDOWN = "local_markdown"
SOURCE_TYPE_LOCAL_CSV = "local_csv"
SOURCE_TYPE_LOCAL_XLSX = "local_xlsx"
SOURCE_TYPE_LOCAL_JSON = "local_json"
SOURCE_TYPE_LOCAL_JSONL = "local_jsonl"
SOURCE_TYPE_GOOGLE_DRIVE_DOC = "google_drive_doc"
SOURCE_TYPE_GOOGLE_DRIVE_FOLDER = "google_drive_folder"

GOOGLE_DRIVE_KC_FOLDER_URL = "https://drive.google.com/drive/folders/15fYbkrGX1XOuSDX7rXs9Xi-88LxlsfCo?hl=ru"
GOOGLE_DRIVE_PRICE_DOC_TITLES = (
    "УНПК Стоимость обучения и порядок оплаты на 26/2027 уч г от 16.03.26",
    "ФОТОН Стоимость обучения и порядок оплаты на 26/27 уч от 16.03.26",
)

SOURCE_INVENTORY_CSV_COLUMNS = (
    "source_id",
    "title",
    "source_type",
    "path",
    "url",
    "google_drive_file_id",
    "google_drive_mime_type",
    "source_updated_at",
    "inventory_status",
    "status_reason",
    "read_succeeded",
    "fact_types",
    "usable_for_precise_answer",
    "freshness_status",
    "limitation_reason",
    "sha256",
    "sha256_source",
    "source_metadata_json",
)


@dataclass(frozen=True)
class SourceInventoryRecord:
    source_id: str
    title: str
    source_type: str
    path: str | None = None
    url: str | None = None
    google_drive_file_id: str | None = None
    google_drive_mime_type: str | None = None
    source_updated_at: str | None = None
    inventory_status: str = STATUS_METADATA_ONLY
    status_reason: str = ""
    read_succeeded: bool = False
    fact_types: tuple[str, ...] = ("unknown",)
    usable_for_precise_answer: bool = False
    freshness_status: str = FRESHNESS_UNKNOWN
    limitation_reason: str = ""
    sha256: str | None = None
    sha256_source: str | None = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", stable_source_id(self.source_id))
        object.__setattr__(self, "title", clean_text(self.title))
        object.__setattr__(self, "source_type", stable_key(self.source_type))
        object.__setattr__(self, "path", optional_text(self.path))
        object.__setattr__(self, "url", optional_text(self.url))
        object.__setattr__(self, "google_drive_file_id", optional_text(self.google_drive_file_id))
        object.__setattr__(self, "google_drive_mime_type", optional_text(self.google_drive_mime_type))
        object.__setattr__(self, "source_updated_at", optional_text(self.source_updated_at))
        object.__setattr__(self, "inventory_status", stable_key(self.inventory_status))
        object.__setattr__(self, "status_reason", clean_text(self.status_reason))
        object.__setattr__(self, "fact_types", normalize_fact_types(self.fact_types))
        object.__setattr__(self, "freshness_status", stable_key(self.freshness_status))
        object.__setattr__(self, "limitation_reason", clean_text(self.limitation_reason))
        object.__setattr__(self, "sha256", optional_text(self.sha256))
        object.__setattr__(self, "sha256_source", optional_text(self.sha256_source))
        object.__setattr__(self, "source_metadata", json_safe_mapping(self.source_metadata))
        if not self.title:
            raise ValueError("SourceInventoryRecord.title must not be empty")
        if self.inventory_status not in INVENTORY_STATUSES:
            raise ValueError(f"unsupported inventory_status={self.inventory_status!r}")
        if self.inventory_status == STATUS_PROCESSED and not self.read_succeeded:
            raise ValueError("inventory_status='processed' requires read_succeeded=True")
        if self.sha256 and (len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256)):
            raise ValueError("sha256 must be a lowercase hex SHA-256 digest")
        if self.sha256 and self.sha256_source not in {SHA256_SOURCE_FILE, SHA256_SOURCE_TEXT}:
            raise ValueError("sha256_source must be 'file' or 'text' when sha256 is present")
        if self.sha256_source and not self.sha256:
            raise ValueError("sha256_source requires sha256")
        if self.usable_for_precise_answer:
            if self.inventory_status != STATUS_PROCESSED or not self.read_succeeded:
                raise ValueError("usable_for_precise_answer requires a processed readable source")
            if self.freshness_status != FRESHNESS_FRESH_VERIFIED:
                raise ValueError("usable_for_precise_answer requires freshness_status='fresh_verified'")
            if not self.sha256:
                raise ValueError("usable_for_precise_answer requires sha256 evidence")

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fact_types"] = list(self.fact_types)
        payload["source_metadata"] = dict(self.source_metadata)
        return payload

    def to_csv_row(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "source_type": self.source_type,
            "path": self.path or "",
            "url": self.url or "",
            "google_drive_file_id": self.google_drive_file_id or "",
            "google_drive_mime_type": self.google_drive_mime_type or "",
            "source_updated_at": self.source_updated_at or "",
            "inventory_status": self.inventory_status,
            "status_reason": self.status_reason,
            "read_succeeded": "true" if self.read_succeeded else "false",
            "fact_types": "|".join(self.fact_types),
            "usable_for_precise_answer": "true" if self.usable_for_precise_answer else "false",
            "freshness_status": self.freshness_status,
            "limitation_reason": self.limitation_reason,
            "sha256": self.sha256 or "",
            "sha256_source": self.sha256_source or "",
            "source_metadata_json": json.dumps(
                dict(self.source_metadata), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        }


@dataclass(frozen=True)
class DriveSourceMetadata:
    title: str
    file_id: str | None = None
    url: str | None = None
    mime_type: str | None = None
    modified_time: str | None = None
    drive_path: str | None = None
    fact_types: tuple[str, ...] = ("unknown",)
    source_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", clean_text(self.title))
        object.__setattr__(self, "file_id", optional_text(self.file_id))
        object.__setattr__(self, "url", optional_text(self.url))
        object.__setattr__(self, "mime_type", optional_text(self.mime_type))
        object.__setattr__(self, "modified_time", optional_text(self.modified_time))
        object.__setattr__(self, "drive_path", optional_text(self.drive_path))
        object.__setattr__(self, "fact_types", normalize_fact_types(self.fact_types))
        object.__setattr__(self, "source_id", optional_text(self.source_id))
        object.__setattr__(self, "metadata", json_safe_mapping(self.metadata))
        if not self.title:
            raise ValueError("DriveSourceMetadata.title must not be empty")


@dataclass(frozen=True)
class DriveTextExport:
    text: str | None = None
    status: str = STATUS_PROCESSED
    status_reason: str = ""
    freshness_status: str = FRESHNESS_NEEDS_MANAGER_CONFIRMATION
    limitation_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", optional_text(self.text))
        object.__setattr__(self, "status", stable_key(self.status))
        object.__setattr__(self, "status_reason", clean_text(self.status_reason))
        object.__setattr__(self, "freshness_status", stable_key(self.freshness_status))
        object.__setattr__(self, "limitation_reason", clean_text(self.limitation_reason))
        object.__setattr__(self, "metadata", json_safe_mapping(self.metadata))
        if self.status not in INVENTORY_STATUSES:
            raise ValueError(f"unsupported DriveTextExport.status={self.status!r}")


DriveExportHook = Callable[[DriveSourceMetadata], Union[DriveTextExport, Mapping[str, Any], str, None]]
FactTypeResolver = Callable[[DriveSourceMetadata], Sequence[str]]


@dataclass(frozen=True)
class LocalSourceSpec:
    path: str
    title: str
    source_type: str
    fact_types: tuple[str, ...]
    source_id: str | None = None
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)


DEFAULT_LOCAL_SOURCE_SPECS = (
    LocalSourceSpec(
        path="База знаний КЦ.docx",
        title="База знаний КЦ",
        source_type=SOURCE_TYPE_LOCAL_DOCX,
        fact_types=("manager_instruction", "restriction", "price", "schedule", "documents", "payment_methods", "program"),
    ),
    LocalSourceSpec(
        path="docs/KC_BOT_KNOWLEDGE_BASE_DRAFT_2026-05-13.md",
        title="KC bot knowledge base draft",
        source_type=SOURCE_TYPE_LOCAL_MARKDOWN,
        fact_types=("price", "payment_methods", "documents", "program", "restriction"),
    ),
    LocalSourceSpec(
        path="docs/DRIVE_KC_KNOWLEDGE_BASE_BUILD_PLAN_2026-05-13.md",
        title="Drive KC knowledge base build plan",
        source_type=SOURCE_TYPE_LOCAL_MARKDOWN,
        fact_types=("source_inventory", "restriction", "manager_instruction"),
    ),
    LocalSourceSpec(
        path="product_data/question_catalog/rop_bot_policy_questionnaire_v2_2026-05-15.csv",
        title="ROP bot policy questionnaire v2 CSV",
        source_type=SOURCE_TYPE_LOCAL_CSV,
        fact_types=("rop_policy", "restriction"),
    ),
    LocalSourceSpec(
        path="product_data/question_catalog/rop_bot_policy_questionnaire_APPROVED_2026-05-15.xlsx",
        title="ROP bot policy questionnaire approved XLSX",
        source_type=SOURCE_TYPE_LOCAL_XLSX,
        fact_types=("rop_policy", "restriction"),
    ),
    LocalSourceSpec(
        path="product_data/question_catalog/customer_question_classes.csv",
        title="Customer question classes",
        source_type=SOURCE_TYPE_LOCAL_CSV,
        fact_types=("question_catalog",),
    ),
    LocalSourceSpec(
        path="product_data/question_catalog/customer_question_items.jsonl",
        title="Customer question items",
        source_type=SOURCE_TYPE_LOCAL_JSONL,
        fact_types=("question_catalog", "manager_answer_pattern"),
    ),
    LocalSourceSpec(
        path="product_data/question_catalog/current_fact_source_registry.json",
        title="Current fact source registry",
        source_type=SOURCE_TYPE_LOCAL_JSON,
        fact_types=("source_inventory", "fact_registry"),
    ),
    LocalSourceSpec(
        path="product_data/question_catalog/fact_requirements.csv",
        title="Fact requirements",
        source_type=SOURCE_TYPE_LOCAL_CSV,
        fact_types=("fact_requirement", "restriction"),
    ),
    LocalSourceSpec(
        path="product_data/question_catalog/question_answer_quality_review_2026-05-14_final.csv",
        title="Question answer quality review final",
        source_type=SOURCE_TYPE_LOCAL_CSV,
        fact_types=("manager_answer_pattern", "question_catalog"),
        required=False,
    ),
    LocalSourceSpec(
        path="product_data/question_catalog/approved_question_answers_draft.csv",
        title="Approved question answers draft",
        source_type=SOURCE_TYPE_LOCAL_CSV,
        fact_types=("manager_answer_pattern",),
        required=False,
    ),
)


def register_local_inventory_source(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
    title: str | None = None,
    source_id: str | None = None,
    source_type: str | None = None,
    fact_types: Sequence[str] = ("unknown",),
    extracted_text: str | None = None,
    freshness_status: str = FRESHNESS_UNKNOWN,
    usable_for_precise_answer: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> SourceInventoryRecord:
    display_path = Path(path)
    probe_path = display_path if display_path.is_absolute() else Path(base_dir or ".") / display_path
    source_title = title or display_path.name
    source_kind = source_type or source_type_for_path(display_path)
    source_metadata = {
        **dict(metadata or {}),
        "required_path": str(display_path),
        "source_exists": probe_path.exists(),
    }
    if probe_path.exists():
        digest = sha256_text(extracted_text) if extracted_text is not None else sha256_file(probe_path)
        sha_source = SHA256_SOURCE_TEXT if extracted_text is not None else SHA256_SOURCE_FILE
        source_metadata.update({"file_size_bytes": probe_path.stat().st_size, "file_suffix": probe_path.suffix.lower()})
        return SourceInventoryRecord(
            source_id=source_id or make_source_id(source_kind, source_title, str(display_path)),
            title=source_title,
            source_type=source_kind,
            path=str(display_path),
            source_updated_at=datetime.fromtimestamp(probe_path.stat().st_mtime, tz=timezone.utc).isoformat(),
            inventory_status=STATUS_PROCESSED,
            status_reason="local_source_hashed",
            read_succeeded=True,
            fact_types=tuple(fact_types),
            usable_for_precise_answer=usable_for_precise_answer,
            freshness_status=freshness_status,
            limitation_reason="" if usable_for_precise_answer else "not_verified_for_precise_answer",
            sha256=digest,
            sha256_source=sha_source,
            source_metadata=source_metadata,
        )
    return SourceInventoryRecord(
        source_id=source_id or make_source_id(source_kind, source_title, str(display_path)),
        title=source_title,
        source_type=source_kind,
        path=str(display_path),
        inventory_status=STATUS_NO_ACCESS,
        status_reason="local_source_not_found",
        read_succeeded=False,
        fact_types=tuple(fact_types),
        freshness_status=FRESHNESS_UNKNOWN,
        limitation_reason="source file is missing or unavailable in the local workspace",
        source_metadata=source_metadata,
    )


def register_drive_inventory_source(
    metadata: DriveSourceMetadata | Mapping[str, Any],
    *,
    fact_types: Sequence[str] | None = None,
    exported_text: str | None = None,
    export_status: str | None = None,
    status_reason: str = "",
    freshness_status: str | None = None,
    limitation_reason: str = "",
    usable_for_precise_answer: bool = False,
    extra_metadata: Mapping[str, Any] | None = None,
) -> SourceInventoryRecord:
    drive_metadata = coerce_drive_metadata(metadata)
    source_type = (
        SOURCE_TYPE_GOOGLE_DRIVE_FOLDER
        if drive_metadata.mime_type == "application/vnd.google-apps.folder"
        else SOURCE_TYPE_GOOGLE_DRIVE_DOC
    )
    source_metadata = dict(drive_metadata.metadata)
    source_metadata.update({"drive_path": drive_metadata.drive_path or "", "live_access_used": bool(source_metadata.get("live_access_used", False))})
    if extra_metadata:
        source_metadata.update(dict(extra_metadata))
    if exported_text is not None:
        status = export_status or STATUS_PROCESSED
        digest = sha256_text(exported_text)
        read_succeeded = True
        sha_source = SHA256_SOURCE_TEXT
        reason = status_reason or "google_drive_text_export_hashed"
        default_freshness = FRESHNESS_NEEDS_MANAGER_CONFIRMATION
        default_limitation = "drive text was exported but facts still require freshness and conflict review"
    else:
        status = export_status or STATUS_METADATA_ONLY
        digest = None
        read_succeeded = False
        sha_source = None
        reason = status_reason or "google_drive_metadata_only"
        default_freshness = FRESHNESS_METADATA_ONLY
        default_limitation = "source metadata is known, but text was not exported or read"
    return SourceInventoryRecord(
        source_id=drive_metadata.source_id
        or make_source_id(source_type, drive_metadata.title, drive_metadata.file_id or drive_metadata.url or ""),
        title=drive_metadata.title,
        source_type=source_type,
        url=drive_metadata.url,
        google_drive_file_id=drive_metadata.file_id,
        google_drive_mime_type=drive_metadata.mime_type,
        source_updated_at=drive_metadata.modified_time,
        inventory_status=status,
        status_reason=reason,
        read_succeeded=read_succeeded,
        fact_types=tuple(fact_types or drive_metadata.fact_types),
        usable_for_precise_answer=usable_for_precise_answer,
        freshness_status=freshness_status or default_freshness,
        limitation_reason=limitation_reason or default_limitation,
        sha256=digest,
        sha256_source=sha_source,
        source_metadata=source_metadata,
    )


def build_drive_inventory_records(
    drive_items: Sequence[DriveSourceMetadata | Mapping[str, Any]],
    *,
    export_text_hook: DriveExportHook | None = None,
    fact_type_resolver: FactTypeResolver | None = None,
) -> list[SourceInventoryRecord]:
    records: list[SourceInventoryRecord] = []
    for item in drive_items:
        metadata = coerce_drive_metadata(item)
        fact_types = tuple(fact_type_resolver(metadata)) if fact_type_resolver is not None else metadata.fact_types
        if export_text_hook is None:
            records.append(register_drive_inventory_source(metadata, fact_types=fact_types))
            continue
        try:
            export = coerce_drive_export(export_text_hook(metadata))
        except PermissionError as exc:
            records.append(
                register_drive_inventory_source(
                    metadata,
                    fact_types=fact_types,
                    export_status=STATUS_NO_ACCESS,
                    status_reason="google_drive_permission_denied",
                    freshness_status=FRESHNESS_METADATA_ONLY,
                    limitation_reason=exception_reason(exc),
                )
            )
            continue
        except Exception as exc:  # pragma: no cover - exact types depend on the caller hook.
            records.append(
                register_drive_inventory_source(
                    metadata,
                    fact_types=fact_types,
                    export_status=STATUS_EXPORT_FAILED,
                    status_reason="google_drive_export_failed",
                    freshness_status=FRESHNESS_METADATA_ONLY,
                    limitation_reason=exception_reason(exc),
                )
            )
            continue
        records.append(drive_record_from_export(metadata=metadata, fact_types=fact_types, export=export))
    return records


def default_google_drive_metadata_sources() -> list[DriveSourceMetadata]:
    return [
        DriveSourceMetadata(
            title="Google Drive: Внутренние документы с актуальной информацией",
            url=GOOGLE_DRIVE_KC_FOLDER_URL,
            mime_type="application/vnd.google-apps.folder",
            fact_types=("source_inventory",),
            metadata={"live_access_used": False, "registration_reason": "required_google_drive_folder"},
        ),
        *[
            DriveSourceMetadata(
                title=title,
                url=f"google-drive://manual-registration/{stable_key(title)}",
                fact_types=("price", "payment_methods"),
                metadata={"live_access_used": False, "folder_url": GOOGLE_DRIVE_KC_FOLDER_URL},
            )
            for title in GOOGLE_DRIVE_PRICE_DOC_TITLES
        ],
    ]


def build_required_source_inventory(
    *,
    base_dir: str | Path = ".",
    local_source_specs: Sequence[LocalSourceSpec] = DEFAULT_LOCAL_SOURCE_SPECS,
    drive_items: Sequence[DriveSourceMetadata | Mapping[str, Any]] | None = None,
    export_text_hook: DriveExportHook | None = None,
) -> list[SourceInventoryRecord]:
    records = [
        register_local_inventory_source(
            spec.path,
            base_dir=base_dir,
            title=spec.title,
            source_id=spec.source_id,
            source_type=spec.source_type,
            fact_types=spec.fact_types,
            metadata={**dict(spec.metadata), "required": spec.required},
        )
        for spec in local_source_specs
    ]
    records.extend(
        build_drive_inventory_records(
            drive_items if drive_items is not None else default_google_drive_metadata_sources(),
            export_text_hook=export_text_hook,
        )
    )
    return records


def mark_manual_review_required(record: SourceInventoryRecord, reason: str) -> SourceInventoryRecord:
    return replace(
        record,
        inventory_status=STATUS_MANUAL_REVIEW_REQUIRED,
        usable_for_precise_answer=False,
        freshness_status=FRESHNESS_NEEDS_MANAGER_CONFIRMATION,
        limitation_reason=reason,
        status_reason="manual_review_required",
    )


def inventory_to_json_payload(records: Sequence[SourceInventoryRecord | Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [coerce_inventory_record(record) for record in records]
    return {
        "schema_version": SOURCE_INVENTORY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": [record.to_json_dict() for record in normalized],
        "summary": summarize_inventory(normalized),
    }


def write_inventory_json(records: Sequence[SourceInventoryRecord | Mapping[str, Any]], path: str | Path) -> None:
    payload = inventory_to_json_payload(records)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_inventory_csv(records: Sequence[SourceInventoryRecord | Mapping[str, Any]], path: str | Path) -> None:
    normalized = [coerce_inventory_record(record) for record in records]
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(SOURCE_INVENTORY_CSV_COLUMNS))
        writer.writeheader()
        for record in normalized:
            writer.writerow(record.to_csv_row())


def summarize_inventory(records: Sequence[SourceInventoryRecord | Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [coerce_inventory_record(record) for record in records]
    by_status = {status: 0 for status in INVENTORY_STATUSES}
    by_type: dict[str, int] = {}
    for record in normalized:
        by_status[record.inventory_status] += 1
        by_type[record.source_type] = by_type.get(record.source_type, 0) + 1
    return {
        "source_count": len(normalized),
        "read_succeeded_count": sum(1 for record in normalized if record.read_succeeded),
        "usable_for_precise_answer_count": sum(1 for record in normalized if record.usable_for_precise_answer),
        "status_counts": by_status,
        "source_type_counts": dict(sorted(by_type.items())),
        "manual_review_required_count": by_status[STATUS_MANUAL_REVIEW_REQUIRED],
        "metadata_only_count": by_status[STATUS_METADATA_ONLY],
        "no_access_count": by_status[STATUS_NO_ACCESS],
        "export_failed_count": by_status[STATUS_EXPORT_FAILED],
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_source_id(source_type: str, title: str, reference: str) -> str:
    digest = hashlib.sha256((str(title) + "\n" + str(reference)).encode("utf-8")).hexdigest()[:16]
    return f"source:{stable_key(source_type)}:{digest}"


def drive_record_from_export(
    *,
    metadata: DriveSourceMetadata,
    fact_types: Sequence[str],
    export: DriveTextExport,
) -> SourceInventoryRecord:
    status = export.status
    status_reason = export.status_reason
    freshness_status = export.freshness_status
    limitation_reason = export.limitation_reason
    if export.text is None and status == STATUS_PROCESSED:
        status = STATUS_METADATA_ONLY
        status_reason = status_reason or "google_drive_hook_returned_no_text"
        freshness_status = FRESHNESS_METADATA_ONLY
        limitation_reason = limitation_reason or "export hook reported processed but did not return readable text"
    return register_drive_inventory_source(
        metadata,
        fact_types=fact_types,
        exported_text=export.text,
        export_status=status,
        status_reason=status_reason,
        freshness_status=freshness_status,
        limitation_reason=limitation_reason,
        extra_metadata=export.metadata,
    )


def coerce_inventory_record(record: SourceInventoryRecord | Mapping[str, Any]) -> SourceInventoryRecord:
    if isinstance(record, SourceInventoryRecord):
        return record
    return SourceInventoryRecord(
        source_id=str(record.get("source_id") or ""),
        title=str(record.get("title") or ""),
        source_type=str(record.get("source_type") or record.get("type") or "unknown"),
        path=record.get("path"),
        url=record.get("url") or record.get("link"),
        google_drive_file_id=record.get("google_drive_file_id") or record.get("file_id"),
        google_drive_mime_type=record.get("google_drive_mime_type") or record.get("mime_type"),
        source_updated_at=record.get("source_updated_at") or record.get("modified_time"),
        inventory_status=str(record.get("inventory_status") or record.get("status") or STATUS_METADATA_ONLY),
        status_reason=str(record.get("status_reason") or ""),
        read_succeeded=bool(record.get("read_succeeded")),
        fact_types=tuple(record.get("fact_types") or ("unknown",)),
        usable_for_precise_answer=bool(record.get("usable_for_precise_answer")),
        freshness_status=str(record.get("freshness_status") or FRESHNESS_UNKNOWN),
        limitation_reason=str(record.get("limitation_reason") or record.get("restriction_reason") or ""),
        sha256=record.get("sha256"),
        sha256_source=record.get("sha256_source"),
        source_metadata=record.get("source_metadata") if isinstance(record.get("source_metadata"), Mapping) else {},
    )


def coerce_drive_metadata(metadata: DriveSourceMetadata | Mapping[str, Any]) -> DriveSourceMetadata:
    if isinstance(metadata, DriveSourceMetadata):
        return metadata
    return DriveSourceMetadata(
        title=str(metadata.get("title") or metadata.get("name") or ""),
        file_id=metadata.get("file_id") or metadata.get("id"),
        url=metadata.get("url") or metadata.get("webViewLink") or metadata.get("web_view_link"),
        mime_type=metadata.get("mime_type") or metadata.get("mimeType"),
        modified_time=metadata.get("modified_time") or metadata.get("modifiedTime") or metadata.get("source_updated_at"),
        drive_path=metadata.get("drive_path") or metadata.get("path"),
        fact_types=tuple(metadata.get("fact_types") or ("unknown",)),
        source_id=metadata.get("source_id"),
        metadata=metadata.get("metadata") if isinstance(metadata.get("metadata"), Mapping) else metadata,
    )


def coerce_drive_export(export: DriveTextExport | Mapping[str, Any] | str | None) -> DriveTextExport:
    if isinstance(export, DriveTextExport):
        return export
    if export is None:
        return DriveTextExport(
            text=None,
            status=STATUS_METADATA_ONLY,
            status_reason="google_drive_hook_returned_no_text",
            freshness_status=FRESHNESS_METADATA_ONLY,
            limitation_reason="export hook did not return readable text",
        )
    if isinstance(export, str):
        return DriveTextExport(text=export, status=STATUS_PROCESSED, status_reason="google_drive_hook_text_exported")
    return DriveTextExport(
        text=export.get("text") or export.get("exported_text"),
        status=str(export.get("status") or STATUS_PROCESSED),
        status_reason=str(export.get("status_reason") or ""),
        freshness_status=str(export.get("freshness_status") or FRESHNESS_NEEDS_MANAGER_CONFIRMATION),
        limitation_reason=str(export.get("limitation_reason") or ""),
        metadata=export.get("metadata") if isinstance(export.get("metadata"), Mapping) else {},
    )


def source_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return SOURCE_TYPE_LOCAL_DOCX
    if suffix == ".md":
        return SOURCE_TYPE_LOCAL_MARKDOWN
    if suffix == ".csv":
        return SOURCE_TYPE_LOCAL_CSV
    if suffix == ".xlsx":
        return SOURCE_TYPE_LOCAL_XLSX
    if suffix == ".json":
        return SOURCE_TYPE_LOCAL_JSON
    if suffix == ".jsonl":
        return SOURCE_TYPE_LOCAL_JSONL
    return "local_file"


def exception_reason(exc: Exception) -> str:
    message = clean_text(str(exc))
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def normalize_fact_types(fact_types: Sequence[str] | None) -> tuple[str, ...]:
    return tuple(dedupe(stable_key(item) for item in fact_types or () if clean_text(item))) or ("unknown",)


def json_safe_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return {str(key): json_safe(item) for key, item in dict(value or {}).items()}


def json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return json_safe_mapping(value)
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def stable_key(value: Any) -> str:
    text = clean_text(value).casefold().replace("ё", "е")
    cleaned = "".join(char if char.isalnum() or char in "_.:-" else "_" for char in text)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ValueError("stable key must not be empty")
    return cleaned


def optional_text(value: Any) -> str | None:
    text = clean_text(value)
    return text or None
