from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MANAGER_ANSWER_PLAYBOOK_SCHEMA_VERSION = "manager_answer_playbook_v1"

DEFAULT_CATALOG_ROOT = Path("product_data/question_catalog")
DEFAULT_OUTPUT_ROOT = Path("product_data/knowledge_base/kb_night_20260517_v1")
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_MIN_SAMPLE_SIZE = 300
DEFAULT_PATTERN_LIMIT = 100

CLASSIFICATION_GOOD = "good"
CLASSIFICATION_UNSAFE = "unsafe"
CLASSIFICATION_OUTDATED = "outdated"
CLASSIFICATION_NO_ANSWER = "no_answer"

KEY_TOPICS = (
    "цена",
    "расписание",
    "оплата",
    "документы",
    "маткапитал",
    "налоговый вычет",
    "возврат",
    "пробное занятие",
    "программа",
    "доступ/ссылки",
    "жалобы",
)

CHANNELS = ("call", "telegram", "email")
RISK_GROUPS = ("safe", "commercial", "legal_docs", "negative")
CLASSIFICATIONS = (
    CLASSIFICATION_GOOD,
    CLASSIFICATION_OUTDATED,
    CLASSIFICATION_UNSAFE,
    CLASSIFICATION_NO_ANSWER,
)

SAMPLE_FIELDNAMES = (
    "sample_rank",
    "question_item_id",
    "question_class_id",
    "channel",
    "topic",
    "risk_group",
    "answer_classification",
    "answer_quality",
    "customer_question_safe",
    "context_summary",
    "historical_manager_answer_safe",
    "historical_answer_presence",
    "client_followup_observed",
    "technique",
    "unsafe_reasons",
    "outdated_or_unverified_fact_risks",
    "usable_as_example",
    "usable_as_fact",
    "safe_pattern_summary",
    "safe_pattern_template",
    "required_fact_keys",
    "bot_permission",
    "answer_evidence_status",
    "occurred_at",
    "source_ref",
)

PATTERN_FIELDNAMES = (
    "pattern_id",
    "channel_scope",
    "topic",
    "risk_group",
    "technique",
    "answer_classification",
    "safe_pattern_summary",
    "safe_pattern_template",
    "example_count",
    "source_question_item_ids",
    "usable_as_fact",
    "fact_safety_note",
    "forbidden_fact_types",
)

UNSAFE_FIELDNAMES = (
    "sample_rank",
    "question_item_id",
    "question_class_id",
    "channel",
    "topic",
    "risk_group",
    "answer_classification",
    "unsafe_reasons",
    "outdated_or_unverified_fact_risks",
    "historical_manager_answer_safe",
    "safe_pattern_summary",
    "usable_as_fact",
)


@dataclass(frozen=True)
class ManagerAnswerRow:
    question_item_id: str
    question_class_id: str
    channel: str
    topic: str
    risk_group: str
    customer_question_safe: str
    context_summary: str
    historical_manager_answer_safe: str
    historical_answer_presence: str
    client_followup_observed: str
    technique: str
    answer_classification: str
    answer_quality: str
    unsafe_reasons: tuple[str, ...] = ()
    outdated_or_unverified_fact_risks: tuple[str, ...] = ()
    usable_as_example: bool = False
    usable_as_fact: bool = False
    safe_pattern_summary: str = ""
    safe_pattern_template: str = ""
    required_fact_keys: tuple[str, ...] = ()
    bot_permission: str = ""
    answer_evidence_status: str = ""
    occurred_at: str = ""
    source_ref: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_sample_dict(self, *, sample_rank: int) -> dict[str, str]:
        return {
            "sample_rank": str(sample_rank),
            "question_item_id": self.question_item_id,
            "question_class_id": self.question_class_id,
            "channel": self.channel,
            "topic": self.topic,
            "risk_group": self.risk_group,
            "answer_classification": self.answer_classification,
            "answer_quality": self.answer_quality,
            "customer_question_safe": self.customer_question_safe,
            "context_summary": self.context_summary,
            "historical_manager_answer_safe": self.historical_manager_answer_safe,
            "historical_answer_presence": self.historical_answer_presence,
            "client_followup_observed": self.client_followup_observed,
            "technique": self.technique,
            "unsafe_reasons": "|".join(self.unsafe_reasons),
            "outdated_or_unverified_fact_risks": "|".join(self.outdated_or_unverified_fact_risks),
            "usable_as_example": _bool_text(self.usable_as_example),
            "usable_as_fact": _bool_text(self.usable_as_fact),
            "safe_pattern_summary": self.safe_pattern_summary,
            "safe_pattern_template": self.safe_pattern_template,
            "required_fact_keys": "|".join(self.required_fact_keys),
            "bot_permission": self.bot_permission,
            "answer_evidence_status": self.answer_evidence_status,
            "occurred_at": self.occurred_at,
            "source_ref": self.source_ref,
        }

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["unsafe_reasons"] = list(self.unsafe_reasons)
        payload["outdated_or_unverified_fact_risks"] = list(self.outdated_or_unverified_fact_risks)
        payload["required_fact_keys"] = list(self.required_fact_keys)
        payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True)
class ManagerAnswerPattern:
    pattern_id: str
    channel_scope: str
    topic: str
    risk_group: str
    technique: str
    answer_classification: str
    safe_pattern_summary: str
    safe_pattern_template: str
    example_count: int
    source_question_item_ids: tuple[str, ...]
    usable_as_fact: bool = False
    fact_safety_note: str = (
        "Historical manager answers are style examples only; they are not verified facts "
        "and must not be copied into a client answer without current fact checks."
    )
    forbidden_fact_types: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_question_item_ids"] = list(self.source_question_item_ids)
        payload["forbidden_fact_types"] = list(self.forbidden_fact_types)
        return payload


def build_manager_answer_playbook(
    catalog_root: str | Path = DEFAULT_CATALOG_ROOT,
    output_root: str | Path | None = None,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
    pattern_limit: int = DEFAULT_PATTERN_LIMIT,
    write_xlsx: bool = True,
) -> dict[str, Any]:
    catalog_path = Path(catalog_root)
    loaded = load_catalog_sources(catalog_path)
    rows = build_manager_answer_rows(
        loaded["items"],
        classes_by_id=loaded["classes_by_id"],
        review_by_class_id=loaded["review_by_class_id"],
        approved_by_class_id=loaded["approved_by_class_id"],
    )
    sample = select_stratified_sample(rows, sample_size=sample_size, min_sample_size=min_sample_size)
    patterns = build_manager_answer_patterns(sample, pattern_limit=pattern_limit)
    result = build_playbook_payload(sample, patterns, catalog_root=catalog_path)
    if output_root is not None:
        paths = write_manager_answer_outputs(
            Path(output_root),
            sample,
            patterns,
            result,
            write_xlsx=write_xlsx,
        )
        result["outputs"] = {key: str(path) if path is not None else None for key, path in paths.items()}
    return result


def load_question_items(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(Path(path))


def analyze_manager_answers(
    items: Sequence[Mapping[str, Any]],
    *,
    target_size: int = DEFAULT_SAMPLE_SIZE,
    minimum_size: int = DEFAULT_MIN_SAMPLE_SIZE,
    pattern_limit: int = DEFAULT_PATTERN_LIMIT,
) -> tuple[list[ManagerAnswerRow], list[ManagerAnswerPattern]]:
    rows = build_manager_answer_rows(items)
    sample = select_stratified_sample(rows, sample_size=target_size, min_sample_size=minimum_size)
    patterns = build_manager_answer_patterns(sample, pattern_limit=pattern_limit)
    return sample, patterns


def write_playbook_outputs(
    records: Sequence[ManagerAnswerRow],
    patterns: Sequence[ManagerAnswerPattern],
    *,
    out_dir: str | Path,
    write_xlsx: bool = True,
) -> dict[str, Any]:
    payload = build_playbook_payload(records, patterns, catalog_root=Path(""))
    paths = write_manager_answer_outputs(Path(out_dir), records, patterns, payload, write_xlsx=write_xlsx)
    return {
        "schema_version": payload["schema_version"],
        "mode": payload["mode"],
        "summary": payload["summary"],
        "safety": payload["safety"],
        "outputs": {key: str(path) if path is not None else None for key, path in paths.items()},
    }


def load_catalog_sources(catalog_root: Path) -> dict[str, Any]:
    items_path = catalog_root / "customer_question_items.jsonl"
    if not items_path.exists():
        raise FileNotFoundError(items_path)
    classes_path = catalog_root / "customer_question_classes.csv"
    review_path = catalog_root / "question_answer_quality_review_2026-05-14_final.csv"
    approved_path = catalog_root / "approved_question_answers_draft.csv"
    classes = read_csv(classes_path) if classes_path.exists() else []
    review_rows = read_csv(review_path) if review_path.exists() else []
    approved_rows = read_csv(approved_path) if approved_path.exists() else []
    return {
        "items": read_jsonl(items_path),
        "classes_by_id": {safe_text(row.get("question_class_id")): row for row in classes},
        "review_by_class_id": {safe_text(row.get("ID класса")): row for row in review_rows},
        "approved_by_class_id": {safe_text(row.get("question_class_id")): row for row in approved_rows},
    }


def build_manager_answer_rows(
    items: Sequence[Mapping[str, Any]],
    *,
    classes_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    review_by_class_id: Mapping[str, Mapping[str, Any]] | None = None,
    approved_by_class_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[ManagerAnswerRow]:
    class_map = classes_by_id or {}
    review_map = review_by_class_id or {}
    approved_map = approved_by_class_id or {}
    rows: list[ManagerAnswerRow] = []
    for item in items:
        class_id = safe_text(item.get("question_class_id"))
        class_row = class_map.get(class_id, {})
        review_row = review_map.get(class_id, {})
        approved_row = approved_map.get(class_id, {})
        rows.append(classify_manager_answer(item, class_row=class_row, review_row=review_row, approved_row=approved_row))
    return rows


def classify_manager_answer(
    item: Mapping[str, Any],
    *,
    class_row: Mapping[str, Any] | None = None,
    review_row: Mapping[str, Any] | None = None,
    approved_row: Mapping[str, Any] | None = None,
) -> ManagerAnswerRow:
    class_data = class_row or {}
    review_data = review_row or {}
    approved_data = approved_row or {}
    class_id = safe_text(item.get("question_class_id"))
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    answer_status = _first_text(
        item.get("answer_evidence_status"),
        metadata.get("answer_status"),
        class_data.get("answer_status"),
        approved_data.get("runtime_bot_permission"),
    )
    bot_permission = _first_text(
        metadata.get("bot_permission"),
        class_data.get("bot_permission"),
        approved_data.get("runtime_bot_permission"),
        approved_data.get("bot_permission"),
    )
    required_fact_keys = parse_list(
        _first_text(
            metadata.get("required_fact_keys"),
            class_data.get("required_fact_keys"),
            approved_data.get("required_fact_keys"),
        )
    )
    raw_question = _first_text(item.get("customer_text_redacted"), metadata.get("customer_text_for_rop"))
    raw_answer = _first_text(
        item.get("manager_text_redacted"),
        review_data.get("Исторический ответ менеджера (не утверждено)"),
    )
    topic = detect_topic(item, class_data, review_data)
    risk_group = detect_risk_group(item, class_data, review_data, topic=topic)
    unsafe_reasons = detect_unsafe_reasons(raw_answer, item, class_data, review_data, topic=topic, risk_group=risk_group)
    stale_risks = detect_outdated_or_unverified_fact_risks(raw_answer, item, class_data, review_data)
    no_answer = _is_missing_answer(raw_answer) or answer_status in {"needs_rop_answer", "not_enough_context"}
    if no_answer:
        classification = CLASSIFICATION_NO_ANSWER
    elif unsafe_reasons:
        classification = CLASSIFICATION_UNSAFE
    elif stale_risks:
        classification = CLASSIFICATION_OUTDATED
    else:
        classification = CLASSIFICATION_GOOD
    technique = infer_answer_technique(raw_answer, topic=topic, risk_group=risk_group, classification=classification)
    safe_summary, safe_template = safe_pattern_for(topic=topic, risk_group=risk_group, technique=technique)
    historical_answer = sanitize_manager_text(raw_answer, max_chars=700)
    if no_answer and not historical_answer:
        historical_answer = "Исторический ответ не найден."
    question_safe = sanitize_manager_text(raw_question, max_chars=420)
    answer_quality = _answer_quality(classification, answer_status=answer_status, unsafe_reasons=unsafe_reasons, stale_risks=stale_risks)
    usable_as_example = classification == CLASSIFICATION_GOOD
    return ManagerAnswerRow(
        question_item_id=safe_text(item.get("question_item_id")),
        question_class_id=class_id,
        channel=normalize_channel(item.get("source_channel")),
        topic=topic,
        risk_group=risk_group,
        customer_question_safe=question_safe,
        context_summary=build_context_summary(item, class_data, required_fact_keys=required_fact_keys),
        historical_manager_answer_safe=historical_answer,
        historical_answer_presence="missing" if no_answer else "present_redacted",
        client_followup_observed=safe_text(metadata.get("client_followup_observed")) or "not_reconstructed",
        technique=technique,
        answer_classification=classification,
        answer_quality=answer_quality,
        unsafe_reasons=tuple(unsafe_reasons),
        outdated_or_unverified_fact_risks=tuple(stale_risks),
        usable_as_example=usable_as_example,
        usable_as_fact=False,
        safe_pattern_summary=safe_summary,
        safe_pattern_template=safe_template,
        required_fact_keys=tuple(required_fact_keys),
        bot_permission=bot_permission,
        answer_evidence_status=answer_status,
        occurred_at=safe_text(item.get("occurred_at")),
        source_ref=sanitize_manager_text(item.get("source_ref"), max_chars=140),
        metadata={
            "answer_source": safe_text(item.get("answer_source")),
            "product": safe_text(item.get("product")),
            "format": safe_text(item.get("format")),
            "intent": safe_text(item.get("intent")),
        },
    )


def select_stratified_sample(
    rows: Sequence[ManagerAnswerRow],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
) -> list[ManagerAnswerRow]:
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    if min_sample_size < 1:
        raise ValueError("min_sample_size must be >= 1")
    if min_sample_size > sample_size:
        raise ValueError("min_sample_size must not be greater than sample_size")
    candidates = sorted(rows, key=_sample_sort_key)
    selected: list[ManagerAnswerRow] = []
    selected_ids: set[str] = set()

    def add_quota(values: Iterable[str], key: str, quota: int) -> None:
        for value in values:
            if len(selected) >= sample_size:
                return
            added = 0
            for row in candidates:
                if len(selected) >= sample_size or added >= quota:
                    break
                if _row_key_value(row, key) != value or row.question_item_id in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(row.question_item_id)
                added += 1

    channel_quota = max(1, sample_size // 12)
    topic_quota = max(1, sample_size // 25)
    risk_quota = max(1, sample_size // 16)
    quality_quota = max(1, sample_size // 20)
    add_quota(CHANNELS, "channel", channel_quota)
    add_quota(KEY_TOPICS, "topic", topic_quota)
    add_quota(_rare_topics(candidates), "topic", max(1, sample_size // 80))
    add_quota(RISK_GROUPS, "risk_group", risk_quota)
    add_quota(CLASSIFICATIONS, "answer_classification", quality_quota)

    for row in candidates:
        if len(selected) >= sample_size:
            break
        if row.question_item_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(row.question_item_id)

    if len(selected) < min_sample_size:
        raise RuntimeError(f"expected at least {min_sample_size} rows, got {len(selected)}")
    return selected[:sample_size]


def build_manager_answer_patterns(
    rows: Sequence[ManagerAnswerRow],
    *,
    pattern_limit: int = DEFAULT_PATTERN_LIMIT,
) -> list[ManagerAnswerPattern]:
    grouped: dict[tuple[str, str, str, str], list[ManagerAnswerRow]] = defaultdict(list)
    for row in rows:
        if row.answer_classification != CLASSIFICATION_GOOD:
            continue
        grouped[(row.channel, row.topic, row.risk_group, row.technique)].append(row)
    patterns: list[ManagerAnswerPattern] = []
    for index, (key, group) in enumerate(sorted(grouped.items(), key=_pattern_group_sort_key), start=1):
        if len(patterns) >= pattern_limit:
            break
        channel, topic, risk_group, technique = key
        source_ids = tuple(row.question_item_id for row in group[:8])
        forbidden_fact_types = tuple(sorted({risk for row in group for risk in row.outdated_or_unverified_fact_risks}))
        summary, template = safe_pattern_for(topic=topic, risk_group=risk_group, technique=technique)
        patterns.append(
            ManagerAnswerPattern(
                pattern_id=f"manager_answer_pattern:{index:03d}",
                channel_scope=channel,
                topic=topic,
                risk_group=risk_group,
                technique=technique,
                answer_classification=CLASSIFICATION_GOOD,
                safe_pattern_summary=summary,
                safe_pattern_template=template,
                example_count=len(group),
                source_question_item_ids=source_ids,
                usable_as_fact=False,
                forbidden_fact_types=forbidden_fact_types,
            )
        )
    return patterns


def build_playbook_payload(
    sample: Sequence[ManagerAnswerRow],
    patterns: Sequence[ManagerAnswerPattern],
    *,
    catalog_root: Path,
) -> dict[str, Any]:
    classification_counts = Counter(row.answer_classification for row in sample)
    topic_counts = Counter(row.topic for row in sample)
    channel_counts = Counter(row.channel for row in sample)
    risk_counts = Counter(row.risk_group for row in sample)
    return {
        "schema_version": MANAGER_ANSWER_PLAYBOOK_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "read_only",
        "catalog_root": str(catalog_root),
        "summary": {
            "sample_rows": len(sample),
            "pattern_rows": len(patterns),
            "classification_counts": dict(classification_counts),
            "topic_counts": dict(topic_counts),
            "channel_counts": dict(channel_counts),
            "risk_counts": dict(risk_counts),
            "answers_usable_as_fact": sum(1 for row in sample if row.usable_as_fact),
            "patterns_usable_as_fact": sum(1 for pattern in patterns if pattern.usable_as_fact),
        },
        "safety": {
            "manager_answers_are_facts": False,
            "raw_manager_answers_copied_to_client_layer": False,
            "personal_data_removed_from_patterns": True,
            "exact_dynamic_facts_removed_from_patterns": True,
            "client_send": False,
            "crm_write": False,
            "tallanto_write": False,
            "stable_runtime_write": False,
        },
        "sample": [row.to_json_dict() for row in sample],
        "manager_answer_patterns": [pattern.to_json_dict() for pattern in patterns],
    }


def write_manager_answer_outputs(
    output_root: Path,
    sample: Sequence[ManagerAnswerRow],
    patterns: Sequence[ManagerAnswerPattern],
    payload: Mapping[str, Any],
    *,
    write_xlsx: bool = True,
) -> dict[str, Path | None]:
    output_root.mkdir(parents=True, exist_ok=True)
    sample_rows = [row.to_sample_dict(sample_rank=index) for index, row in enumerate(sample, start=1)]
    unsafe_rows = [
        _unsafe_row(row, sample_rank=index)
        for index, row in enumerate(sample, start=1)
        if row.answer_classification != CLASSIFICATION_GOOD
    ]
    paths: dict[str, Path | None] = {
        "sample_csv": output_root / "manager_answer_sample_300_500.csv",
        "sample_jsonl": output_root / "manager_answer_sample_300_500.jsonl",
        "patterns_jsonl": output_root / "manager_answer_patterns.jsonl",
        "playbook_md": output_root / "manager_answer_playbook.md",
        "unsafe_csv": output_root / "unsafe_or_outdated_manager_answers.csv",
        "summary_json": output_root / "manager_answer_playbook_summary.json",
        "sample_xlsx": None,
    }
    write_csv(paths["sample_csv"], sample_rows, SAMPLE_FIELDNAMES)
    write_jsonl(paths["sample_jsonl"], [row.to_json_dict() for row in sample])
    write_jsonl(paths["patterns_jsonl"], [pattern.to_json_dict() for pattern in patterns])
    write_csv(paths["unsafe_csv"], unsafe_rows, UNSAFE_FIELDNAMES)
    paths["playbook_md"].write_text(render_manager_answer_playbook_md(payload), encoding="utf-8")
    paths["summary_json"].write_text(
        json.dumps(_summary_payload(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if write_xlsx:
        xlsx_path = output_root / "manager_answer_sample_300_500.xlsx"
        if write_sample_xlsx(xlsx_path, sample_rows):
            paths["sample_xlsx"] = xlsx_path
    return paths


def render_manager_answer_playbook_md(payload: Mapping[str, Any]) -> str:
    summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    patterns = payload.get("manager_answer_patterns")
    pattern_rows = patterns if isinstance(patterns, list) else []
    lines = [
        "# База приемов хороших ответов менеджеров",
        "",
        "Важно: это не база фактов. Исторические ответы менеджеров используются только как источник приемов и тона.",
        "",
        "## Метрики",
        "",
        f"- Строк в выборке: {summary.get('sample_rows', 0)}",
        f"- Хороших паттернов: {summary.get('pattern_rows', 0)}",
        f"- Ответов, пригодных как факт: {summary.get('answers_usable_as_fact', 0)}",
        f"- Паттернов, пригодных как факт: {summary.get('patterns_usable_as_fact', 0)}",
        "",
        "## Классификация ответов",
        "",
    ]
    for key, value in sorted((summary.get("classification_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Покрытие каналов", ""])
    for key, value in sorted((summary.get("channel_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Покрытие тем", ""])
    for key, value in sorted((summary.get("topic_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Безопасные приемы", ""])
    for pattern in pattern_rows[:25]:
        if not isinstance(pattern, Mapping):
            continue
        lines.append(
            f"- {pattern.get('channel_scope')} / {pattern.get('topic')} / "
            f"{pattern.get('technique')}: {pattern.get('safe_pattern_summary')}"
        )
    lines.extend(
        [
            "",
            "## Правила использования",
            "",
            "- Не копировать исторические ответы клиенту дословно.",
            "- Не считать цены, даты, проценты, расписание, возвраты и документы актуальными по ответу менеджера.",
            "- Передавать в prompt только безопасный пересказ приема и короткий шаблон, без персональных данных.",
            "- Для коммерческих, юридических и конфликтных тем оставлять ручную проверку менеджера.",
            "",
        ]
    )
    return "\n".join(lines)


def sanitize_manager_text(value: Any, *, max_chars: int = 700) -> str:
    text = safe_text(value)
    if not text:
        return ""
    replacements = (
        (EMAIL_RE, "[email]"),
        (URL_RE, "[ссылка]"),
        (PHONE_RE, "[телефон]"),
        (CURRENCY_RE, "[сумма]"),
        (PERCENT_RE, "[процент]"),
        (DATE_RE, "[дата]"),
        (DAY_MONTH_RE, "[дата]"),
        (TIME_RE, "[время]"),
        (LONG_NUMBER_RE, "[номер]"),
        (INTERNAL_ID_RE, "[внутренний_id]"),
    )
    cleaned = text
    for pattern, replacement in replacements:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned.replace("\xa0", " ")).strip()
    return truncate_text(cleaned, max_chars)


def detect_topic(
    item: Mapping[str, Any],
    class_row: Mapping[str, Any],
    review_row: Mapping[str, Any],
) -> str:
    text = " ".join(
        [
            safe_text(item.get("intent")),
            safe_text(item.get("dynamic_fact_types")),
            safe_text(class_row.get("parent_question_class")),
            safe_text(class_row.get("question_subclass")),
            safe_text(class_row.get("canonical_question")),
            safe_text(review_row.get("Крупный класс")),
            safe_text(review_row.get("Узкий класс")),
        ]
    ).casefold()
    ordered_patterns: tuple[tuple[str, re.Pattern[str]], ...] = (
        ("маткапитал", re.compile(r"маткап|материнск")),
        ("налоговый вычет", re.compile(r"налог|ндфл|вычет")),
        ("возврат", re.compile(r"возврат|перерасч[её]т|refund")),
        ("жалобы", re.compile(r"жалоб|претенз|недоволь|качество|негатив|quality_feedback")),
        ("доступ/ссылки", re.compile(r"доступ|ссылк|кабинет|логин|парол|technical_access")),
        ("пробное занятие", re.compile(r"пробн|диагност|тестов|trial")),
        ("цена", re.compile(r"стоим|цен|price")),
        ("расписание", re.compile(r"распис|schedule|день|время|график")),
        ("оплата", re.compile(r"оплат|плат[её]ж|квитанц|чек|сч[её]т|payment|installment")),
        ("документы", re.compile(r"документ|договор|справ|письм|documents|legal")),
        ("программа", re.compile(r"программ|предмет|курс|егэ|огэ|олимпиад|format|program|camp")),
    )
    for topic, pattern in ordered_patterns:
        if pattern.search(text):
            return topic
    parent = safe_text(class_row.get("parent_question_class")) or safe_text(review_row.get("Крупный класс"))
    return parent or "другое"


def detect_risk_group(
    item: Mapping[str, Any],
    class_row: Mapping[str, Any],
    review_row: Mapping[str, Any],
    *,
    topic: str,
) -> str:
    text = " ".join(
        [
            topic,
            safe_text(item.get("intent")),
            safe_text(item.get("dynamic_fact_types")),
            safe_text(class_row.get("parent_question_class")),
            safe_text(class_row.get("question_subclass")),
            safe_text(review_row.get("Группы риска")),
            safe_text(review_row.get("Риск ошибки")),
        ]
    ).casefold()
    if topic in {"маткапитал", "налоговый вычет", "возврат"} or re.search(r"юрид|договор|лиценз|претенз", text):
        return "legal_docs"
    if topic in {"жалобы"} or re.search(r"жалоб|негатив|недоволь|отмен", text):
        return "negative"
    if topic in {"цена", "оплата"} or re.search(r"скид|рассроч|стоим|цен|payment|price", text):
        return "commercial"
    return "safe"


def detect_unsafe_reasons(
    raw_answer: str,
    item: Mapping[str, Any],
    class_row: Mapping[str, Any],
    review_row: Mapping[str, Any],
    *,
    topic: str,
    risk_group: str,
) -> list[str]:
    text = safe_text(raw_answer)
    if not text:
        return []
    lower = text.casefold()
    reasons: list[str] = []
    if EMAIL_RE.search(text) or PHONE_RE.search(text) or URL_RE.search(text):
        reasons.append("contains_direct_contact_or_link")
    if INTERNAL_WORD_RE.search(text) or INTERNAL_ID_RE.search(text):
        reasons.append("contains_internal_system_reference")
    safety_flags = _list_from_any(item.get("safety_flags"))
    if any(flag in {"email_redacted", "phone_redacted", "person_name_redacted"} for flag in safety_flags):
        reasons.append("source_required_personal_data_redaction")
    bot_permission = _first_text(
        item.get("bot_permission"),
        (item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}).get("bot_permission"),
        class_row.get("bot_permission"),
    )
    if bot_permission == "manager_only" and risk_group in {"legal_docs", "negative"}:
        reasons.append("manager_only_high_risk_topic")
    if risk_group == "legal_docs" and re.search(r"гарант|точно|обязательн|верн[её]м|оформим|можно оплатить", lower):
        reasons.append("legal_or_refund_promise")
    if topic == "жалобы" and re.search(r"вы сами|это не наша|ничего не можем|не проблема", lower):
        reasons.append("dismissive_negative_reply")
    review_blocker = safe_text(review_row.get("Блокер качества"))
    if review_blocker and review_blocker not in {"нет", "false", "0"}:
        reasons.append("review_quality_blocker")
    return tuple_dedupe(reasons)


def detect_outdated_or_unverified_fact_risks(
    raw_answer: str,
    item: Mapping[str, Any],
    class_row: Mapping[str, Any],
    review_row: Mapping[str, Any],
) -> list[str]:
    text = safe_text(raw_answer)
    if not text:
        return []
    risks: list[str] = []
    dynamic_fact_types = set(_list_from_any(item.get("dynamic_fact_types")))
    dynamic_fact_types.update(parse_list(safe_text(class_row.get("required_fact_keys"))))
    if _is_truthy(item.get("requires_dynamic_facts")) or dynamic_fact_types:
        for fact_type in sorted(_normalize_fact_risk(value) for value in dynamic_fact_types if value):
            risks.append(f"requires_current_{fact_type}")
    status = _first_text(
        item.get("answer_evidence_status"),
        (item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}).get("answer_status"),
        class_row.get("answer_status"),
    )
    if status == "template_ready_needs_current_fact":
        risks.append("answer_status_requires_current_fact")
    if CURRENCY_RE.search(text):
        risks.append("contains_unverified_amount")
    if PERCENT_RE.search(text):
        risks.append("contains_unverified_percent")
    if DATE_RE.search(text) or DAY_MONTH_RE.search(text) or TIME_RE.search(text):
        risks.append("contains_unverified_date_or_time")
    if re.search(r"стоим|цена|скидк|рассроч|распис|оплат|возврат|налог|маткап", text, re.I):
        if not risks:
            risks.append("contains_dynamic_business_terms")
    review_fact = safe_text(review_row.get("Нужные актуальные факты"))
    if review_fact:
        risks.append("review_requires_actual_fact_check")
    return tuple_dedupe(risks)


def infer_answer_technique(
    raw_answer: str,
    *,
    topic: str,
    risk_group: str,
    classification: str,
) -> str:
    text = safe_text(raw_answer).casefold()
    if classification == CLASSIFICATION_NO_ANSWER:
        return "нет исторического ответа"
    if risk_group in {"legal_docs", "negative"}:
        return "ручная эскалация"
    if re.search(r"уточн|напиш|пришл|нужн|подскаж|какой|какая|какие", text):
        return "запрос нужных данных"
    if re.search(r"передам|свяж|верн[её]мся|провер|отправим|подбер", text):
        return "понятный следующий шаг"
    if re.search(r"можно|сначала|после|далее|оформ|проходит", text):
        return "объяснение порядка"
    if topic in {"цена", "оплата", "расписание"}:
        return "мягкая проверка актуальности"
    return "короткий содержательный ответ"


def safe_pattern_for(*, topic: str, risk_group: str, technique: str) -> tuple[str, str]:
    if topic == "цена":
        return (
            "Подтвердить вопрос о стоимости, не называть сумму из истории и передать менеджеру задачу проверить актуальный расчет.",
            "Понял(а) вопрос по стоимости. Актуальные условия нужно проверить по текущему файлу, менеджер подготовит точный расчет и вернется с ответом.",
        )
    if topic == "расписание":
        return (
            "Подтвердить запрос по времени, не называть точный слот из истории и передать менеджеру подбор расписания.",
            "Понял(а), нужен удобный вариант по расписанию. Менеджер проверит актуальные группы и предложит подходящий слот.",
        )
    if topic == "оплата":
        return (
            "Разобрать тип платежного вопроса, не давать реквизиты или суммы из истории и обозначить следующий шаг менеджера.",
            "Понял(а) вопрос по оплате. Менеджер проверит актуальные данные по договору и подскажет безопасный следующий шаг.",
        )
    if topic in {"маткапитал", "налоговый вычет", "возврат", "документы"}:
        return (
            "Аккуратно признать вопрос по документам, не давать юридических обещаний и отправить на ручную проверку.",
            "Понял(а), вопрос связан с документами и условиями. Передам менеджеру на проверку, чтобы ответ был по актуальным правилам.",
        )
    if topic == "пробное занятие":
        return (
            "Подтвердить интерес к пробному занятию, уточнить класс, предмет и формат без обещания конкретного слота.",
            "Понял(а), хотите подобрать пробное занятие. Уточним класс, предмет и формат, затем менеджер предложит подходящий вариант.",
        )
    if topic == "программа":
        return (
            "Ответить по сути программы общими словами и попросить менеджера проверить детали под класс, предмет и формат.",
            "Понял(а) вопрос по программе. Менеджер сверит детали под нужный класс, предмет и формат и даст точный ответ.",
        )
    if topic == "доступ/ссылки":
        return (
            "Признать проблему с доступом, попросить данные для поиска заявки и передать проверку менеджеру.",
            "Понял(а), есть вопрос по доступу или ссылке. Передам менеджеру, он проверит заявку и подскажет следующий шаг.",
        )
    if topic == "жалобы" or risk_group == "negative":
        return (
            "Признать неудобство без спорных обещаний, зафиксировать суть проблемы и передать менеджеру для проверки.",
            "Понимаю, ситуация неприятная. Передам менеджеру суть вопроса, он проверит детали и вернется с решением.",
        )
    if technique == "запрос нужных данных":
        return (
            "Попросить только данные, нужные для проверки, без лишних персональных деталей и без обещаний.",
            "Понял(а). Чтобы менеджер проверил вопрос, уточним только необходимые данные и вернемся с ответом.",
        )
    return (
        "Коротко ответить по сути, отделить стиль от фактов и оставить точные условия на проверку менеджера.",
        "Понял(а) вопрос. Менеджер проверит актуальные детали и вернется с точным ответом.",
    )


def build_context_summary(
    item: Mapping[str, Any],
    class_row: Mapping[str, Any],
    *,
    required_fact_keys: Sequence[str],
) -> str:
    parts = [
        f"класс: {safe_text(class_row.get('canonical_question')) or safe_text(item.get('question_class_id'))}",
        f"продукт: {safe_text(item.get('product')) or 'не указан'}",
        f"формат: {safe_text(item.get('format')) or 'не указан'}",
    ]
    if required_fact_keys:
        parts.append(f"нужны факты: {', '.join(required_fact_keys)}")
    return sanitize_manager_text("; ".join(parts), max_chars=320)


def normalize_channel(value: Any) -> str:
    text = safe_text(value).lower()
    if text in {"mail", "email", "e-mail"}:
        return "email"
    if text in {"tg", "telegram"}:
        return "telegram"
    if text in {"call", "phone", "звонок"}:
        return "call"
    return text or "unknown"


def parse_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [safe_text(item) for item in value if safe_text(item)]
    text = safe_text(value)
    if not text:
        return []
    parts = re.split(r"[|,;]\s*", text)
    return [part.strip() for part in parts if part.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: safe_text(row.get(key)) for key in fieldnames})


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_sample_xlsx(path: Path, rows: Sequence[Mapping[str, Any]]) -> bool:
    try:
        from openpyxl import Workbook
    except ImportError:
        return False
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "manager_answer_sample"
    worksheet.append(list(SAMPLE_FIELDNAMES))
    for row in rows:
        worksheet.append([safe_text(row.get(key)) for key in SAMPLE_FIELDNAMES])
    for column_cells in worksheet.columns:
        max_length = max(len(safe_text(cell.value)) for cell in column_cells)
        worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 12), 60)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return True


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def truncate_text(value: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", safe_text(value)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def tuple_dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _first_text(*values: Any) -> str:
    for value in values:
        text = safe_text(value)
        if text and text != "None":
            return text
    return ""


def _list_from_any(value: Any) -> list[str]:
    if isinstance(value, list):
        return [safe_text(item) for item in value if safe_text(item)]
    if isinstance(value, tuple):
        return [safe_text(item) for item in value if safe_text(item)]
    if isinstance(value, str):
        return parse_list(value)
    return []


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = safe_text(value).casefold()
    return text in {"1", "true", "yes", "y", "да"}


def _is_missing_answer(value: Any) -> bool:
    text = safe_text(value)
    normalized = re.sub(r"[\s.!?,]+", "", text.casefold())
    if not normalized:
        return True
    if normalized in {"да", "нет", "ок", "окей", "угу", "ага", "спасибо"}:
        return True
    return len(normalized) < 4


def _normalize_fact_risk(value: str) -> str:
    base = safe_text(value).split(".", 1)[0]
    aliases = {
        "prices": "price",
        "price": "price",
        "schedule": "schedule",
        "documents": "documents",
        "document": "documents",
        "payment": "payment",
        "payment_methods": "payment",
        "discount": "discount",
        "installment": "installment",
        "program": "program",
        "location": "location",
        "trial": "trial",
    }
    return aliases.get(base, re.sub(r"[^a-z0-9а-я_]+", "_", base.lower()).strip("_") or "fact")


def _answer_quality(
    classification: str,
    *,
    answer_status: str,
    unsafe_reasons: Sequence[str],
    stale_risks: Sequence[str],
) -> str:
    if classification == CLASSIFICATION_GOOD:
        return "good_candidate"
    if classification == CLASSIFICATION_NO_ANSWER:
        return "missing_or_not_enough_context"
    if unsafe_reasons:
        return "unsafe_for_client_layer"
    if stale_risks or answer_status == "template_ready_needs_current_fact":
        return "needs_current_fact_check"
    return "needs_review"


def _sample_sort_key(row: ManagerAnswerRow) -> tuple[int, int, str, str]:
    class_rank = {
        CLASSIFICATION_GOOD: 0,
        CLASSIFICATION_OUTDATED: 1,
        CLASSIFICATION_UNSAFE: 2,
        CLASSIFICATION_NO_ANSWER: 3,
    }.get(row.answer_classification, 9)
    answer_len_rank = 0 if 40 <= len(row.historical_manager_answer_safe) <= 500 else 1
    return (class_rank, answer_len_rank, row.topic, row.question_item_id)


def _row_key_value(row: ManagerAnswerRow, key: str) -> str:
    if key == "channel":
        return row.channel
    if key == "topic":
        return row.topic
    if key == "risk_group":
        return row.risk_group
    if key == "answer_classification":
        return row.answer_classification
    raise ValueError(f"unsupported stratification key: {key}")


def _rare_topics(rows: Sequence[ManagerAnswerRow]) -> list[str]:
    counts = Counter(row.topic for row in rows)
    return [topic for topic, _ in sorted(counts.items(), key=lambda item: (item[1], item[0])) if topic not in KEY_TOPICS]


def _pattern_group_sort_key(item: tuple[tuple[str, str, str, str], list[ManagerAnswerRow]]) -> tuple[int, str, str, str, str]:
    (channel, topic, risk_group, technique), rows = item
    return (-len(rows), topic, risk_group, technique, channel)


def _unsafe_row(row: ManagerAnswerRow, *, sample_rank: int) -> dict[str, str]:
    return {
        "sample_rank": str(sample_rank),
        "question_item_id": row.question_item_id,
        "question_class_id": row.question_class_id,
        "channel": row.channel,
        "topic": row.topic,
        "risk_group": row.risk_group,
        "answer_classification": row.answer_classification,
        "unsafe_reasons": "|".join(row.unsafe_reasons),
        "outdated_or_unverified_fact_risks": "|".join(row.outdated_or_unverified_fact_risks),
        "historical_manager_answer_safe": row.historical_manager_answer_safe,
        "safe_pattern_summary": row.safe_pattern_summary,
        "usable_as_fact": _bool_text(row.usable_as_fact),
    }


def _summary_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": payload.get("schema_version"),
        "generated_at": payload.get("generated_at"),
        "mode": payload.get("mode"),
        "catalog_root": payload.get("catalog_root"),
        "summary": payload.get("summary"),
        "safety": payload.get("safety"),
    }


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
URL_RE = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?7|8)?[\s(.-]*\d{3}[\s).-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{6,}(?!\d)")
CURRENCY_RE = re.compile(r"(?<!\w)\d[\d\s.,]*(?:₽|руб(?:\.|лей|ля|ль)?|тыс(?:\.|яч)?|к)(?!\w)", re.I)
PERCENT_RE = re.compile(r"(?<!\w)\d{1,3}\s*(?:%|процент(?:а|ов)?)(?!\w)", re.I)
DATE_RE = re.compile(r"(?<!\d)\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?(?!\d)")
DAY_MONTH_RE = re.compile(
    r"(?<!\d)\d{1,2}\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?!\w)",
    re.I,
)
TIME_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?!\d)")
INTERNAL_ID_RE = re.compile(r"\b(?:amo|амо|crm|црм|tallanto|талланто|lead|deal|contact)[:#\s-]*\d+\b", re.I)
INTERNAL_WORD_RE = re.compile(r"\b(?:amo|амо|crm|црм|tallanto|талланто|лид|сделк[аи]|внутренн\w+)\b", re.I)
