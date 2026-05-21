from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ANSWER_REGISTRY_SCHEMA_VERSION = "kb_answer_registry_v1"

ALLOWED_BRANDS = frozenset({"foton", "unpk"})
ALLOWED_ROUTES = frozenset({"draft_for_manager", "manager_only", "bot_answer_self_for_pilot"})
SAFE_HANDOFF_MARKERS = (
    "менеджер",
    "свяжется",
    "передам",
    "уточнит",
    "проверит",
)

DEBUG_LEAK_RE = re.compile(
    r"\b(?:source_id|fact_id|freshness_status|structured_value|answer_id|snake_case|json)\b|"
    r"\b(?:Claude|Codex|GPT|Tallanto|AMO)\b|"
    r"\{[^{}]{8,}\}|\[[^\[\]]*source=[^\[\]]*\]",
    re.I,
)
BOT_IDENTITY_RE = re.compile(r"\b(?:я\s+бот|как\s+ии|нейросет|искусственн\w*\s+интеллект|автоматическ\w+\s+ответ|gpt|claude|codex)\b", re.I)
PRICE_RE = re.compile(r"(?<!\d)(\d[\d\s\u00a0]{0,8})\s*(?:₽|руб\.?|рублей|р\.)(?!\w)", re.I)
PERCENT_RE = re.compile(r"(?<!\d)(\d{1,3})\s*(?:%|процент\w*)", re.I)
DATE_RE = re.compile(
    r"(?<!\d)(?:\d{1,2}[./]\d{1,2}(?:[./]\d{2,4})?|\d{1,2}\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря))(?!\d)",
    re.I,
)
PROMISE_WORD_RE = re.compile(r"\b(?:гарант\w*|точно|обеща\w*|место\s+будет|верн[её]м|скидка\s+будет)\b", re.I)

FOTON_FORBIDDEN_RE = re.compile(r"\b(?:УНПК|АНО\s+ДПО|kmipt|Сретенк|Институтск|Пацаева|@unpk)", re.I)
UNPK_FORBIDDEN_RE = re.compile(r"\b(?:Фотон|ЦДПО|cdpofoton|Т-?Банк|Долями|@cdpo)", re.I)

PII_COLLECTION_RE = re.compile(
    r"\b(?:фио|имя|фамили\w*|договор|номер\s+договора|телефон|email|почт\w*|сумм\w*|причин\w*)\b",
    re.I,
)
APOLOGY_RE = re.compile(r"\b(?:извините|приносим\s+извинения|понимаю|сожалеем|неприятно)\b", re.I)


@dataclass(frozen=True)
class AnswerRegistryEntry:
    answer_id: str
    brand: str
    topic: str
    route: str
    template: str
    source_ids: tuple[str, ...] = ()
    subtopic: str = ""
    client_allowed: bool = True
    manager_review_required: bool = True
    must_include: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    missing_fact_fallback: str = ""
    status: str = "draft"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AnswerRegistryEntry":
        return cls(
            answer_id=_clean(payload.get("answer_id")),
            brand=_normalize_brand(payload.get("brand")),
            topic=_clean(payload.get("topic")),
            route=_clean(payload.get("route")),
            template=_clean(payload.get("template") or payload.get("template_text")),
            source_ids=tuple(_clean(item) for item in payload.get("source_ids") or () if _clean(item)),
            subtopic=_clean(payload.get("subtopic")),
            client_allowed=bool(payload.get("client_allowed", True)),
            manager_review_required=bool(payload.get("manager_review_required", True)),
            must_include=tuple(_clean(item) for item in payload.get("must_include") or () if _clean(item)),
            forbidden=tuple(_clean(item) for item in payload.get("forbidden") or () if _clean(item)),
            missing_fact_fallback=_clean(payload.get("missing_fact_fallback")),
            status=_clean(payload.get("status") or "draft"),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {},
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "answer_id": self.answer_id,
            "brand": self.brand,
            "topic": self.topic,
            "subtopic": self.subtopic,
            "route": self.route,
            "template": self.template,
            "source_ids": list(self.source_ids),
            "client_allowed": self.client_allowed,
            "manager_review_required": self.manager_review_required,
            "must_include": list(self.must_include),
            "forbidden": list(self.forbidden),
            "missing_fact_fallback": self.missing_fact_fallback,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AnswerRegistryIssue:
    code: str
    severity: str
    message: str
    answer_id: str = ""

    def to_json_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "answer_id": self.answer_id,
        }


@dataclass(frozen=True)
class DraftSemanticIssue:
    code: str
    severity: str
    message: str
    test_id: str = ""

    def to_json_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "test_id": self.test_id,
        }


def load_answer_registry(path: str | Path) -> list[AnswerRegistryEntry]:
    target = Path(path)
    if not target.exists():
        return []
    if target.suffix.lower() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
    elif target.suffix.lower() in {".jsonl", ".ndjson"}:
        return [
            AnswerRegistryEntry.from_mapping(json.loads(line))
            for line in target.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif target.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional local package
            raise RuntimeError("PyYAML is required to read YAML answer registry") from exc
        payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    else:
        raise ValueError(f"Unsupported answer registry format: {target.suffix}")
    entries = payload.get("answers") or payload.get("entries") or payload.get("templates") or []
    if isinstance(entries, Mapping):
        entries = list(entries.values())
    return [AnswerRegistryEntry.from_mapping(item) for item in entries if isinstance(item, Mapping)]


def validate_answer_registry(
    entries: Sequence[AnswerRegistryEntry | Mapping[str, Any]],
    *,
    known_source_ids: Iterable[str] = (),
) -> list[AnswerRegistryIssue]:
    normalized = [entry if isinstance(entry, AnswerRegistryEntry) else AnswerRegistryEntry.from_mapping(entry) for entry in entries]
    known_sources = {str(item).strip() for item in known_source_ids if str(item).strip()}
    issues: list[AnswerRegistryIssue] = []
    seen: set[str] = set()
    for entry in normalized:
        if not entry.answer_id:
            issues.append(AnswerRegistryIssue("missing_answer_id", "error", "У ответа нет answer_id."))
        elif entry.answer_id in seen:
            issues.append(AnswerRegistryIssue("duplicate_answer_id", "error", "answer_id повторяется.", entry.answer_id))
        seen.add(entry.answer_id)
        if entry.brand not in ALLOWED_BRANDS:
            issues.append(AnswerRegistryIssue("invalid_brand", "error", "Бренд должен быть foton или unpk.", entry.answer_id))
        if not entry.topic:
            issues.append(AnswerRegistryIssue("missing_topic", "error", "У ответа нет topic.", entry.answer_id))
        if entry.route not in ALLOWED_ROUTES:
            issues.append(AnswerRegistryIssue("invalid_route", "error", "Недопустимый route.", entry.answer_id))
        if not entry.template.strip():
            issues.append(AnswerRegistryIssue("missing_template", "error", "У ответа нет template.", entry.answer_id))
        if entry.client_allowed and _has_cross_brand_text(entry.template, entry.brand):
            issues.append(AnswerRegistryIssue("cross_brand_template", "error", "Шаблон смешивает бренды.", entry.answer_id))
        if entry.client_allowed and contains_debug_leak(entry.template):
            issues.append(AnswerRegistryIssue("debug_leak_in_template", "error", "Шаблон содержит служебные маркеры.", entry.answer_id))
        if entry.route != "manager_only" and not entry.source_ids:
            issues.append(
                AnswerRegistryIssue(
                    "missing_source_for_precise_answer",
                    "error",
                    "Для справочного ответа нужен источник или явный handoff.",
                    entry.answer_id,
                )
            )
        if known_sources:
            missing = [source for source in entry.source_ids if source not in known_sources]
            if missing:
                issues.append(
                    AnswerRegistryIssue(
                        "unknown_source_id",
                        "error",
                        f"source_id отсутствует в source_registry: {', '.join(missing)}",
                        entry.answer_id,
                    )
                )
        if entry.route == "manager_only" and not _has_safe_handoff(entry.template + " " + entry.missing_fact_fallback):
            issues.append(
                AnswerRegistryIssue(
                    "manager_only_without_handoff",
                    "warning",
                    "manager_only ответ должен явно передавать вопрос менеджеру.",
                    entry.answer_id,
                )
            )
    return issues


def select_answer_blocks(
    entries: Sequence[AnswerRegistryEntry | Mapping[str, Any]],
    *,
    brand: str,
    topics: Sequence[str],
    max_blocks: int = 3,
) -> list[AnswerRegistryEntry]:
    active_brand = _normalize_brand(brand)
    requested = {_clean(topic).casefold() for topic in topics if _clean(topic)}
    normalized = [entry if isinstance(entry, AnswerRegistryEntry) else AnswerRegistryEntry.from_mapping(entry) for entry in entries]
    selected: list[AnswerRegistryEntry] = []
    for entry in normalized:
        if entry.brand != active_brand:
            continue
        topic_tokens = {entry.topic.casefold(), entry.subtopic.casefold()}
        if requested and not (requested & topic_tokens):
            continue
        selected.append(entry)
        if len(selected) >= max_blocks:
            break
    return selected


def validate_draft_semantics(
    *,
    draft_text: str,
    brand: str,
    route: str = "",
    priority: str = "",
    category: str = "",
    subcategory: str = "",
    test_id: str = "",
    allowed_numeric_markers: Sequence[str] = (),
) -> list[DraftSemanticIssue]:
    text = _clean(draft_text)
    active_brand = _normalize_brand(brand)
    issues: list[DraftSemanticIssue] = []
    if contains_debug_leak(text):
        issues.append(DraftSemanticIssue("debug_leak", "error", "В черновике есть служебные маркеры.", test_id))
    if BOT_IDENTITY_RE.search(text):
        issues.append(DraftSemanticIssue("bot_identity_leak", "error", "Черновик раскрывает ИИ/бот-идентичность.", test_id))
    if _has_cross_brand_text(text, active_brand):
        issues.append(DraftSemanticIssue("cross_brand_leak", "error", "Черновик смешивает бренды.", test_id))
    issues.extend(_unsupported_numeric_issues(text, allowed_numeric_markers=allowed_numeric_markers, test_id=test_id))
    issues.extend(_business_plausibility_issues(text, test_id=test_id))
    risk = f"{category} {subcategory}".casefold()
    if route == "manager_only" or priority == "P0" or any(token in risk for token in ("refund", "legal", "complaint", "high_risk")):
        if any(token in risk for token in ("refund", "legal")) and PII_COLLECTION_RE.search(text):
            issues.append(DraftSemanticIssue("pii_collection_in_high_risk", "error", "Опасная тема собирает лишние данные.", test_id))
        if "complaint" in risk and APOLOGY_RE.search(text):
            issues.append(DraftSemanticIssue("company_apology_in_complaint", "error", "Жалоба содержит извинение/признание от лица компании.", test_id))
    return issues


def contains_debug_leak(text: str) -> bool:
    return bool(DEBUG_LEAK_RE.search(text or ""))


def semantic_passed(issues: Sequence[AnswerRegistryIssue | DraftSemanticIssue]) -> bool:
    return not any(issue.severity == "error" for issue in issues)


def _unsupported_numeric_issues(
    text: str,
    *,
    allowed_numeric_markers: Sequence[str],
    test_id: str,
) -> list[DraftSemanticIssue]:
    allowed = {_normalize_number(marker) for marker in allowed_numeric_markers if _normalize_number(marker)}
    issues: list[DraftSemanticIssue] = []
    for regex, code in ((PRICE_RE, "unsupported_money_or_price"), (PERCENT_RE, "unsupported_percent"), (DATE_RE, "unsupported_date")):
        for match in regex.finditer(text):
            raw = match.group(0)
            normalized = _normalize_number(raw)
            if allowed and normalized in allowed:
                continue
            if not allowed and PROMISE_WORD_RE.search(text[max(0, match.start() - 80) : match.end() + 80]):
                issues.append(DraftSemanticIssue(code, "error", f"Число выглядит как неподтверждённое обещание: {raw}", test_id))
    return issues


def _business_plausibility_issues(text: str, *, test_id: str) -> list[DraftSemanticIssue]:
    issues: list[DraftSemanticIssue] = []
    for match in PRICE_RE.finditer(text):
        amount = int(re.sub(r"\D", "", match.group(1)) or "0")
        if 0 < amount < 3000:
            issues.append(DraftSemanticIssue("implausible_course_price", "error", f"Цена курса выглядит неправдоподобно: {match.group(0)}", test_id))
    if re.search(r"\b(?:заняти[йя]|урок\w*|недел\w*)\b[^.\n]{0,40}\b(?:руб|₽)", text, re.I):
        issues.append(DraftSemanticIssue("course_parameter_as_price", "error", "Учебный параметр выглядит как цена.", test_id))
    return issues


def _has_cross_brand_text(text: str, brand: str) -> bool:
    if brand == "foton":
        return bool(FOTON_FORBIDDEN_RE.search(text or ""))
    if brand == "unpk":
        return bool(UNPK_FORBIDDEN_RE.search(text or ""))
    return False


def _has_safe_handoff(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(marker in lowered for marker in SAFE_HANDOFF_MARKERS)


def _normalize_brand(value: Any) -> str:
    text = _clean(value).casefold().replace("фотон", "foton").replace("унпк", "unpk")
    if text in {"foton", "unpk"}:
        return text
    return text


def _normalize_number(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def _clean(value: Any) -> str:
    return str(value or "").strip()
