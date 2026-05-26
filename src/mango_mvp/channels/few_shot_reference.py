from __future__ import annotations

import os
import re
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


FEW_SHOT_REFERENCE_SCHEMA_VERSION = "telegram_few_shot_reference_v1_2026_05_23"
GOLD_ANSWER_CONTEXT_SCHEMA_VERSION = "telegram_gold_answer_context_v1_2026_05_25"
DEFAULT_WARM_PATH = Path("product_data/bot_improvement_candidates_20260523/01_gold_and_few_shot/few_shot_warm_answers_2026-05-23.yaml")
DEFAULT_ADVANCED_PATH = Path("product_data/bot_improvement_candidates_20260523/01_gold_and_few_shot/few_shot_advanced_pack_2026-05-23.yaml")
DEFAULT_GOLD_PATH = Path("product_data/bot_improvement_candidates_20260523/01_gold_and_few_shot/bot_gold_answers.json")
WARM_PATH_ENV = "MANGO_TELEGRAM_FEW_SHOT_WARM_PATH"
ADVANCED_PATH_ENV = "MANGO_TELEGRAM_FEW_SHOT_ADVANCED_PATH"
GOLD_PATH_ENV = "MANGO_TELEGRAM_GOLD_ANSWERS_V3_PATH"
GOLD_CONTEXT_ENV = "TELEGRAM_DRAFT_GOLD_V3_CONTEXT"
STYLE_EXAMPLE_LIMIT = 6
CORRECTION_EXAMPLE_LIMIT = 4

_P0_RE = re.compile(
    r"возврат|верн(ите|уть|ули|у)|суд|прокуратур|роспотреб|жалоб|претензи|договор.*не то|оплатил.*не вид",
    re.I,
)
_TOPIC_PATTERNS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("10_camps", "theme:026_camp_general", ("лагер", "лвш", "смен", "менделеево", "летн")),
    ("07_tax_deduction", "theme:008_tax_deduction", ("налог", "вычет", "справк")),
    ("06_matkap", "theme:007_matkap_payment", ("маткап", "материн")),
    ("01_pricing_with_validity", "theme:001_pricing", ("цен", "стоим", "сколько", "руб", "прайс")),
    ("02_installment_payment", "theme:006_installment", ("рассроч", "долями", "частями", "помесяч")),
    ("09_discounts_conditional", "theme:005_discounts", ("скид", "льгот", "промокод", "акци")),
    ("03_trial_class", "theme:023_trial_class", ("пробн",)),
    ("04_platform_records", "theme:014_format", ("платформ", "запис", "мтс линк", "zoom", "зум", "онлайн")),
    ("08_address", "theme:015_address", ("адрес", "где вы", "находитесь", "сретен", "пацаев", "красносельск")),
    ("05_schedule_availability", "theme:013_schedule", ("распис", "когда", "во сколько", "суббот", "воскрес")),
)


def build_few_shot_reference(
    *,
    message_text: str,
    active_brand: str,
    topic_id: str = "",
    confirmed_facts: Mapping[str, Any] | None = None,
    missing_facts: Sequence[str] = (),
    known_slots: Mapping[str, Any] | None = None,
    warm_path: str | Path | None = None,
    advanced_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Build compact style examples for the Telegram draft prompt.

    The returned examples are tone and structure references only. They must not be
    treated as facts; precise claims still require confirmed facts from the KB.
    """

    brand = _normalize_brand(active_brand)
    message = str(message_text or "")
    normalized = _normalize(message)
    topic_key = _infer_topic_key(normalized, topic_id=topic_id)
    confirmed_fact_available = bool(confirmed_facts)
    warm = load_warm_examples(warm_path)
    advanced = load_advanced_examples(advanced_path)

    style_examples: list[str] = []
    correction_examples: list[str] = []

    if _P0_RE.search(normalized):
        correction_examples.extend(_format_p0_examples(advanced))
    else:
        style_examples.extend(
            _format_warm_examples(
                warm,
                brand=brand,
                topic_key=topic_key,
                precise_fact_available=confirmed_fact_available,
            )
        )
        if not confirmed_fact_available:
            style_examples.extend(_format_no_fact_examples(advanced, brand=brand, topic_key=topic_key))
        correction_examples.extend(
            _format_correction_examples(
                advanced,
                brand=brand,
                topic_key=topic_key,
                missing_fact=not confirmed_fact_available,
                known_slots=known_slots or {},
            )
        )

    style_phrases = _style_phrases(advanced)
    result = {
        "schema_version": FEW_SHOT_REFERENCE_SCHEMA_VERSION,
        "purpose": "tone_and_structure_only_not_fact_source",
        "injection_rules": [
            "Примеры задают тон и структуру, но не являются источником фактов.",
            "Числа, даты, скидки, адреса, места и условия можно повторять только если они есть в confirmed_facts/facts_context.",
            "Класс, предмет, формат, расписание, цель клиента и соседнюю тему нельзя додумывать из примера: только из confirmed_facts/facts_context или слов клиента.",
            "Если пример конфликтует с активным брендом или подтверждёнными фактами, игнорируй пример.",
            "Сначала ответь на прямой вопрос; если факта нет, честно скажи, что подтвердит менеджер, и дай полезный ориентир без конкретики.",
        ],
        "detected_topic_key": topic_key,
        "active_brand": brand,
        "precise_fact_available": confirmed_fact_available,
        "style_examples": _dedupe(style_examples)[:STYLE_EXAMPLE_LIMIT],
        "correction_examples": _dedupe(correction_examples)[:CORRECTION_EXAMPLE_LIMIT],
        "warm_openers": style_phrases.get("warm_openers", [])[:3],
        "forbidden_phrases": style_phrases.get("forbidden_anywhere", [])[:6],
    }
    return {key: value for key, value in result.items() if value not in ({}, [], "", None)}


def build_gold_answer_context(
    *,
    message_text: str,
    active_brand: str,
    topic_id: str = "",
    confirmed_facts: Mapping[str, Any] | None = None,
    gold_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Build a compact gold-v3 context for tone and answer shape.

    Gold answers are intentionally not facts. Brand/topic selection and a minimal
    precise-claim check keep old or cross-brand examples out of the prompt.
    """

    if not _gold_context_enabled():
        return {}
    payload = load_gold_answers(gold_path)
    topics = payload.get("topics")
    if not isinstance(topics, Mapping):
        return {}
    brand = _normalize_brand(active_brand)
    if brand not in {"foton", "unpk"}:
        return {}
    normalized = _normalize(message_text)
    topic_key = _infer_gold_topic_key(normalized, topic_id=topic_id)
    selected_topics = [topic_key] if topic_key != "unknown" else []
    if topic_key == "unknown" and _identity_question(normalized):
        selected_topics = ["identity"]
    confirmed_blob = _confirmed_fact_blob(confirmed_facts or {})

    examples: list[Mapping[str, Any]] = []
    for key in selected_topics:
        record = topics.get(key)
        if not isinstance(record, Mapping):
            continue
        brand_record = record.get(brand) if isinstance(record.get(brand), Mapping) else record.get("common")
        if not isinstance(brand_record, Mapping):
            continue
        answer = str(brand_record.get("gold_answer_example") or "").strip()
        if not answer:
            continue
        if _precise_claims_need_confirmed_facts(answer) and not _precise_claims_are_supported(answer, confirmed_blob):
            continue
        examples.append(
            {
                "topic": key,
                "brand": brand if isinstance(record.get(brand), Mapping) else "common",
                "gold_answer_example": answer,
                "must_include": _clean_sequence(brand_record.get("must_include"), limit=5),
                "must_not_include": _clean_sequence(brand_record.get("must_not_include"), limit=5),
            }
        )

    global_rules = _clean_sequence(payload.get("global_rules"), limit=8)
    result = {
        "schema_version": GOLD_ANSWER_CONTEXT_SCHEMA_VERSION,
        "source_schema_version": str(payload.get("schema_version") or ""),
        "purpose": "tone_and_structure_only_not_fact_source",
        "active_brand": brand,
        "detected_topic": topic_key,
        "injection_rules": [
            "Gold-ответы задают тон и структуру, но не являются источником фактов.",
            "Числа, даты, скидки, адреса, места и условия повторять только при поддержке confirmed_facts/facts_context.",
            "Класс, предмет, формат, расписание, цель клиента и соседнюю тему нельзя додумывать из gold-примера.",
            "При конфликте с active_brand, confirmed_facts или правилами безопасности пример игнорируется.",
        ],
        "global_rules": global_rules,
        "examples": examples[:4],
    }
    return {key: value for key, value in result.items() if value not in ({}, [], "", None)}


def build_gold_answers_v3_summary(gold_context: Mapping[str, Any]) -> Mapping[str, Any]:
    if not gold_context:
        return {}
    return {
        "schema_version": gold_context.get("source_schema_version") or "gold_answers_v3",
        "purpose": "tone_and_structure_only_not_fact_source",
        "active_brand": gold_context.get("active_brand"),
        "detected_topic": gold_context.get("detected_topic"),
        "rules": gold_context.get("injection_rules", ()),
    }


def load_warm_examples(path: str | Path | None = None) -> Mapping[str, Any]:
    return _load_yaml_cached(str(_resolve_path(path, WARM_PATH_ENV, DEFAULT_WARM_PATH)))


def load_advanced_examples(path: str | Path | None = None) -> Mapping[str, Any]:
    return _load_yaml_cached(str(_resolve_path(path, ADVANCED_PATH_ENV, DEFAULT_ADVANCED_PATH)))


def load_gold_answers(path: str | Path | None = None) -> Mapping[str, Any]:
    return _load_json_cached(str(_resolve_path(path, GOLD_PATH_ENV, DEFAULT_GOLD_PATH)))


@lru_cache(maxsize=8)
def _load_yaml_cached(path_text: str) -> Mapping[str, Any]:
    path = Path(path_text)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


@lru_cache(maxsize=8)
def _load_json_cached(path_text: str) -> Mapping[str, Any]:
    path = Path(path_text)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _resolve_path(path: str | Path | None, env_name: str, default: Path) -> Path:
    if path is not None:
        return Path(path)
    env_value = os.environ.get(env_name)
    return Path(env_value) if env_value else default


def _format_warm_examples(
    payload: Mapping[str, Any],
    *,
    brand: str,
    topic_key: str,
    precise_fact_available: bool,
) -> list[str]:
    if not precise_fact_available:
        return []
    brand_examples = payload.get(brand)
    if not isinstance(brand_examples, Mapping):
        return []
    examples = brand_examples.get(topic_key)
    if not isinstance(examples, Sequence) or isinstance(examples, (str, bytes)):
        return []
    result: list[str] = []
    for example in examples[:2]:
        if not isinstance(example, Mapping):
            continue
        result.append(_format_example(example.get("client"), example.get("answer"), teaches="warm_direct_answer"))
    return result


def _format_no_fact_examples(payload: Mapping[str, Any], *, brand: str, topic_key: str) -> list[str]:
    result: list[str] = []
    for example in _records(payload.get("no_fact_examples")):
        example_brand = _normalize_brand(example.get("brand"))
        if example_brand not in {"any", brand}:
            continue
        example_topic = str(example.get("topic_id") or "")
        if topic_key == "unknown" or not _topic_matches(topic_key, example_topic):
            continue
        answer = _mask_precise_claims(str(example.get("answer") or ""))
        result.append(_format_example(example.get("client"), answer, teaches=str(example.get("teaches") or "no_fact_helpful")))
    return result


def _format_correction_examples(
    payload: Mapping[str, Any],
    *,
    brand: str,
    topic_key: str,
    missing_fact: bool,
    known_slots: Mapping[str, Any],
) -> list[str]:
    preferred_flags: list[str] = ["ignored_question", "templated_opening"]
    if known_slots:
        preferred_flags.append("reasked_known")
    if missing_fact:
        preferred_flags.append("fabricated_specific")
    else:
        preferred_flags.append("over_handoff")
    if topic_key in {"05_schedule_availability", "08_address"}:
        preferred_flags.append("invited_to_visit")

    rows = []
    for example in _records(payload.get("bad_good_pairs")):
        example_brand = _normalize_brand(example.get("brand"))
        if example_brand not in {"any", brand}:
            continue
        flag = str(example.get("flag") or "")
        if flag not in preferred_flags:
            continue
        rows.append((preferred_flags.index(flag), example))

    result: list[str] = []
    for _, example in sorted(rows, key=lambda item: item[0]):
        result.append(
            "\n".join(
                item
                for item in (
                    f"Флаг: {example.get('flag')}",
                    f"Плохо: {_mask_precise_claims(str(example.get('bad') or '')) if missing_fact else example.get('bad')}",
                f"Лучше: {_mask_precise_claims(str(example.get('good') or '')) if missing_fact else example.get('good')}",
                    f"Почему: {example.get('why')}",
                )
                if item and not item.endswith("None")
            )
        )
    return result


def _format_p0_examples(payload: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for example in _records(payload.get("p0_warm_handoff")):
        result.append(_format_example(example.get("client"), example.get("answer"), teaches=str(example.get("teaches") or "p0_handoff")))
    return result


def _style_phrases(payload: Mapping[str, Any]) -> Mapping[str, list[str]]:
    value = payload.get("style_phrases_no_facts")
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): [str(item) for item in rows[:6] if str(item).strip()]
        for key, rows in value.items()
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes))
    }


def _format_example(client: Any, answer: Any, *, teaches: str) -> str:
    return "\n".join(
        item
        for item in (
            f"Клиент: {str(client or '').strip()}",
            f"Образец: {str(answer or '').strip()}",
            f"Учит: {teaches}",
        )
        if item and not item.endswith(": ")
    )


def _mask_precise_claims(text: str) -> str:
    masked = str(text or "")
    masked = re.sub(r"\b\d{2}/\d{2}\b", "на новый учебный год", masked)
    masked = re.sub(r"\b\d[\d\s]*\s*₽", "[цену не называть без факта]", masked)
    masked = re.sub(r"\b\d+\s*%", "[процент не называть без факта]", masked)
    masked = re.sub(r"\b\d+\s*раз[а-я]*\s+в\s+недел[а-я]*", "[частоту занятий не называть без факта]", masked, flags=re.I)
    masked = re.sub(r"\b\d+\s*минут[а-я]*", "[длительность занятия не называть без факта]", masked, flags=re.I)
    masked = re.sub(r"\b\d{1,2}:\d{2}\b", "[время не называть без факта]", masked)
    return " ".join(masked.split())


def _infer_topic_key(normalized_message: str, *, topic_id: str) -> str:
    topic = str(topic_id or "").casefold()
    for key, canonical_topic_id, markers in _TOPIC_PATTERNS:
        if canonical_topic_id in topic:
            return key
        if any(marker in normalized_message for marker in markers):
            return key
    if "преподав" in normalized_message or "педагог" in normalized_message:
        return "teacher_question"
    if "дорого" in normalized_message:
        return "objection_expensive"
    return "unknown"


_GOLD_TOPIC_BY_FEW_SHOT_TOPIC: Mapping[str, str] = {
    "01_pricing_with_validity": "pricing",
    "02_installment_payment": "installment",
    "03_trial_class": "trial_class",
    "04_platform_records": "platform_records",
    "05_schedule_availability": "schedule_groups",
    "06_matkap": "matkap",
    "07_tax_deduction": "tax",
    "08_address": "addresses",
    "09_discounts_conditional": "discounts",
    "10_camps": "camps",
}


def _infer_gold_topic_key(normalized_message: str, *, topic_id: str) -> str:
    if _identity_question(normalized_message):
        return "identity"
    few_shot_topic = _infer_topic_key(normalized_message, topic_id=topic_id)
    return _GOLD_TOPIC_BY_FEW_SHOT_TOPIC.get(few_shot_topic, "unknown")


def _identity_question(normalized_message: str) -> bool:
    return any(marker in normalized_message for marker in ("ты бот", "вы бот", "ты ии", "вы ии", "кто отвечает", "с кем я общаюсь"))


def _gold_context_enabled() -> bool:
    value = os.environ.get(GOLD_CONTEXT_ENV)
    if value is None:
        return True
    return value.strip().casefold() not in {"0", "false", "no", "off", "нет"}


def _clean_sequence(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


def _confirmed_fact_blob(confirmed_facts: Mapping[str, Any]) -> str:
    try:
        raw = json.dumps(confirmed_facts, ensure_ascii=False)
    except TypeError:
        raw = str(confirmed_facts)
    return re.sub(r"\s+", "", raw.casefold())


def _precise_claims_need_confirmed_facts(text: str) -> bool:
    return bool(re.search(r"\d|₽|%|сретенк|красносельск|пацаев|мфти|мтс\s*линк", text, re.I))


def _precise_claims_are_supported(text: str, confirmed_blob: str) -> bool:
    if not confirmed_blob:
        return False
    numbers = [re.sub(r"\D", "", item) for item in re.findall(r"\d[\d\s\u00a0]*", text)]
    numbers = [item for item in numbers if item]
    for number in numbers:
        if number not in confirmed_blob:
            return False
    address_markers = ("сретенк", "красносельск", "пацаев", "мфти", "мтслинк")
    normalized_text = re.sub(r"\s+", "", text.casefold())
    return all(marker in confirmed_blob for marker in address_markers if marker in normalized_text)


def _topic_matches(topic_key: str, example_topic: str) -> bool:
    normalized = example_topic.casefold()
    if topic_key.split("_", 1)[0] in normalized:
        return True
    if topic_key == "01_pricing_with_validity" and "pricing" in normalized:
        return True
    if topic_key == "05_schedule_availability" and "schedule" in normalized:
        return True
    if topic_key == "10_camps" and "camp" in normalized:
        return True
    return False


def _records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _normalize(value: Any) -> str:
    return str(value or "").casefold().replace("ё", "е")


def _normalize_brand(value: Any) -> str:
    text = _normalize(value).strip()
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "унпк мфти"}:
        return "unpk"
    if text == "any":
        return "any"
    return "unknown"


def _dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value or "").split())
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result
