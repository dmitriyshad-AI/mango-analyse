from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_RULES_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "D1_audit_backlog" / "rules_registry.yaml"
MIGRATED = frozenset({"teacher", "recordings", "contact_address"})


@dataclass(frozen=True)
class Rule:
    rule_id: str
    intent: str
    title: str = ""
    route_effect: str = "bot_answer_self"
    brand_split: bool = False
    data: Mapping[str, Any] = field(default_factory=dict)
    required_fact_keys: tuple[str, ...] = ()
    preserve_exceptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuleOutcome:
    rule_id: str
    subvariant: str
    route: str
    text: str
    facts: Mapping[str, str] = field(default_factory=dict)
    flags: tuple[str, ...] = ()
    checklist: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


def load_rules_registry(path: str | Path | None = None) -> dict[str, Rule]:
    resolved = Path(path or DEFAULT_RULES_REGISTRY_PATH).expanduser().resolve(strict=False)
    return dict(_load_rules_registry_cached(str(resolved)))


@lru_cache(maxsize=8)
def _load_rules_registry_cached(path: str) -> dict[str, Rule]:
    text = Path(path).read_text(encoding="utf-8")
    raw_rules = _load_yaml_rules(text)
    if raw_rules is None:
        raw_rules = _parse_rules_registry_fallback(text)
    registry: dict[str, Rule] = {}
    for item in raw_rules:
        if not isinstance(item, Mapping):
            continue
        rule_id = str(item.get("rule_id") or "").strip()
        if not rule_id:
            continue
        registry[rule_id] = Rule(
            rule_id=rule_id,
            intent=str(item.get("intent") or "").strip(),
            title=str(item.get("title") or "").strip(),
            route_effect=str(item.get("route_effect") or "bot_answer_self").strip() or "bot_answer_self",
            brand_split=bool(item.get("brand_split")),
            data=item.get("data") if isinstance(item.get("data"), Mapping) else {},
            required_fact_keys=tuple(str(value) for value in (item.get("required_fact_keys") or ()) if str(value).strip()),
            preserve_exceptions=tuple(str(value) for value in (item.get("preserve_exceptions") or ()) if str(value).strip()),
        )
    return registry


def _load_yaml_rules(text: str) -> list[Mapping[str, Any]] | None:
    try:
        import yaml  # type: ignore
    except Exception:
        return None
    try:
        loaded = yaml.safe_load(text)
    except Exception:
        return None
    if not isinstance(loaded, Mapping):
        return None
    rules = loaded.get("rules")
    if not isinstance(rules, list):
        return None
    return [item for item in rules if isinstance(item, Mapping)]


def _parse_rules_registry_fallback(text: str) -> list[Mapping[str, Any]]:
    """Parse enough of the approved registry when PyYAML is unavailable.

    The project does not pin PyYAML as a dependency. This fallback keeps runtime
    independent while the richer parser is used whenever available.
    """

    result: list[dict[str, Any]] = []
    chunks = re.split(r"\n\s{2}- rule_id:\s*", "\n" + text)
    for chunk in chunks[1:]:
        first_line, _, rest = chunk.partition("\n")
        rule_id = first_line.strip()
        if not rule_id:
            continue
        data: dict[str, Any] = {"rule_id": rule_id}
        for field_name in ("intent", "title", "route_effect"):
            match = re.search(rf"^\s{{4}}{field_name}:\s*(.+?)\s*$", rest, re.M)
            if match:
                data[field_name] = match.group(1).strip().strip('"')
        data["brand_split"] = bool(re.search(r"^\s{4}brand_split:\s*true\s*$", rest, re.M))
        required = re.search(r"^\s{4}required_fact_keys:\s*\[(.*?)\]", rest, re.M)
        if required:
            data["required_fact_keys"] = [part.strip() for part in required.group(1).split(",") if part.strip()]
        elif re.search(r"^\s{4}required_fact_keys:\s*$", rest, re.M):
            values: list[str] = []
            for line in rest.splitlines():
                match = re.match(r"^\s{6}-\s+(.+?)\s*$", line)
                if match:
                    values.append(match.group(1).strip())
            data["required_fact_keys"] = values
        if rule_id == "contact_address":
            data["data"] = {
                "foton": {"address": "Москва, Скорняжный"},
                "unpk": {
                    "addresses": [
                        "Москва, Сретенка, 20",
                        "Долгопрудный, Институтский пер., 9 (МФТИ)",
                        "Долгопрудный, Проспект Пацаева, 7к1",
                    ]
                },
            }
        result.append(data)
    return result


def select_rule(intent: str, registry: Mapping[str, Rule]) -> Rule | None:
    rule_id = _rule_id_for_intent(intent)
    if rule_id not in MIGRATED:
        return None
    return registry.get(rule_id)


def apply_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None = None,
) -> RuleOutcome | None:
    if rule.rule_id == "teacher":
        return _apply_teacher_rule(rule, plan=plan, facts=facts, context=context)
    if rule.rule_id == "recordings":
        return _apply_recordings_rule(rule, plan=plan, facts=facts, context=context)
    if rule.rule_id == "contact_address":
        return _apply_contact_address_rule(rule, plan=plan, facts=facts, context=context)
    return None


def _rule_id_for_intent(intent: str) -> str:
    normalized = str(intent or "").strip().casefold()
    aliases = {
        "teacher": "teacher",
        "teacher_inquiry": "teacher",
        "recording": "recordings",
        "recording_inquiry": "recordings",
        "recordings": "recordings",
        "address": "contact_address",
        "contact_address": "contact_address",
        "contact_address_inquiry": "contact_address",
    }
    return aliases.get(normalized, normalized)


def _apply_teacher_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_teacher_question(question, plan):
        return None
    fact_key, raw_fact_text = _first_matching_fact(
        facts,
        ("teacher", "teachers", "theme_17_teachers", "преподав", "педагог", "учитель", "мфти", "мгу", "вшэ", "мифи"),
    )
    fact_text = _brand_safe_teacher_fact_text(raw_fact_text, active_brand=_active_brand(plan, context))
    if not fact_text:
        return None

    subvariant = "general"
    route = "bot_answer_self_for_pilot"
    if _mentions_teacher_change(question):
        subvariant = "change_teacher"
        route = "draft_for_manager"
        text = (
            "Если преподаватель не подойдёт, менеджер поможет проверить возможность перевода в другую группу. "
            f"Ориентир по составу преподавателей: {_short_sentence(fact_text)}"
        )
    elif _mentions_mendeleevo(question):
        subvariant = "mendeleevo"
        text = f"По ЛВШ Менделеево ориентир по преподавателям такой: {_short_sentence(fact_text)} Менеджер уточнит состав по группе."
    elif _asks_specific_teacher_name(question):
        subvariant = "specific_name"
        route = "draft_for_manager"
        text = (
            "Конкретное имя преподавателя зависит от группы. "
            f"Ориентир по составу: {_short_sentence(fact_text)} Менеджер уточнит преподавателя по вашей группе."
        )
    else:
        text = f"По преподавателям: {_short_sentence(fact_text)} Конкретный состав по группе менеджер уточнит отдельно."

    return RuleOutcome(
        rule_id=rule.rule_id,
        subvariant=subvariant,
        route=route,
        text=text,
        facts={fact_key or "rules_engine.teacher.regalia": fact_text},
        flags=("rules_engine_teacher_applied", f"rules_engine_teacher_{subvariant}"),
        checklist=("Rule engine: teacher — регалии из факта; конкретное ФИО не выдумывать.",),
        metadata={"source": "rules_engine", "rule_id": "teacher", "subvariant": subvariant},
    )


def _apply_recordings_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_recording_question(question, plan):
        return None
    scope = str(plan.get("fact_scope") or "").casefold()
    wants_offline = scope == "offline_recordings" or _has_any(question, ("очн", "офлайн", "обычные занят"))
    wants_online = scope == "online_recordings" or _has_any(question, ("онлайн", "дистанц", "мтс", "линк", "пропуст", "пересмотр"))
    offline_key, offline_fact = _first_matching_fact(
        facts,
        ("offline_recording", "offline_recordings", "запись очных", "запись очных занятий не", "очные занятия не"),
    )
    online_key, online_fact = _first_matching_fact(
        facts,
        ("online_recording", "online_recordings", "записи уроков доступны", "доступны записи", "пересмотр"),
    )

    if wants_offline and offline_fact:
        subvariant = "offline"
        text = f"По очным занятиям: {_short_sentence(offline_fact)}"
        source = {offline_key or "rules_engine.recordings.offline": offline_fact}
    elif wants_online and online_fact:
        subvariant = "online"
        text = f"По онлайн-занятиям: {_short_sentence(online_fact)}"
        source = {online_key or "rules_engine.recordings.online": online_fact}
    elif online_fact and offline_fact:
        subvariant = "online_and_offline"
        text = f"По онлайн-занятиям: {_short_sentence(online_fact)} По очным занятиям: {_short_sentence(offline_fact)}"
        source = {
            online_key or "rules_engine.recordings.online": online_fact,
            offline_key or "rules_engine.recordings.offline": offline_fact,
        }
    else:
        return None

    return RuleOutcome(
        rule_id=rule.rule_id,
        subvariant=subvariant,
        route="bot_answer_self_for_pilot",
        text=text,
        facts=source,
        flags=("rules_engine_recordings_applied", f"rules_engine_recordings_{subvariant}"),
        checklist=("Rule engine: recordings — онлайн/очно разделены, доступность только из факта.",),
        metadata={"source": "rules_engine", "rule_id": "recordings", "subvariant": subvariant},
    )


def _apply_contact_address_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_address_question(question, plan):
        return None
    brand = _active_brand(plan, context)
    if brand == "foton":
        configured = rule.data.get("foton") if isinstance(rule.data.get("foton"), Mapping) else {}
        address = str(configured.get("address") or "Москва, Скорняжный").strip()
        text = f"Фотон в Москве: {address}. Если нужна площадка под конкретную группу, менеджер уточнит детали."
        fact_key = "rules_registry.contact_address.foton.address"
        fact_text = f"Фотон: адрес очных занятий — {address}."
    elif brand == "unpk":
        configured = rule.data.get("unpk") if isinstance(rule.data.get("unpk"), Mapping) else {}
        addresses = tuple(str(item).strip() for item in (configured.get("addresses") or ()) if str(item).strip())
        if not addresses:
            return None
        text = "Площадки УНПК: " + "; ".join(addresses) + ". Если выбираете очные занятия, уточните, какая площадка удобнее."
        fact_key = "rules_registry.contact_address.unpk.addresses"
        fact_text = "УНПК: площадки — " + "; ".join(addresses) + "."
    else:
        return None

    return RuleOutcome(
        rule_id=rule.rule_id,
        subvariant="where_located",
        route="bot_answer_self_for_pilot",
        text=text,
        facts={fact_key: fact_text},
        flags=("rules_engine_contact_address_applied", f"rules_engine_contact_address_{brand}"),
        checklist=("Rule engine: contact_address — адрес отвечает только на адресный вопрос, бренд не смешивать.",),
        metadata={"source": "rules_engine", "rule_id": "contact_address", "subvariant": "where_located", "brand": brand},
    )


def _question_text(plan: Mapping[str, Any], context: Mapping[str, Any] | None) -> str:
    parts: list[str] = []
    for key in ("direct_question", "fact_query_text", "primary_intent", "fact_scope"):
        value = str(plan.get(key) or "").strip()
        if value:
            parts.append(value)
    if isinstance(context, Mapping):
        for key in ("client_message", "current_message", "message_text"):
            value = str(context.get(key) or "").strip()
            if value:
                parts.append(value)
    return " ".join(parts).casefold().replace("ё", "е")


def _active_brand(plan: Mapping[str, Any], context: Mapping[str, Any] | None) -> str:
    value = str(plan.get("active_brand") or "").strip().casefold()
    if value in {"foton", "unpk"}:
        return value
    if isinstance(context, Mapping):
        for key in ("active_brand", "brand"):
            candidate = str(context.get(key) or "").strip().casefold()
            if candidate in {"foton", "unpk"}:
                return candidate
    return "unknown"


def _looks_like_teacher_question(text: str, plan: Mapping[str, Any]) -> bool:
    return str(plan.get("primary_intent") or "") == "teacher" or _has_any(
        text,
        ("преподав", "педагог", "учитель", "кто вед", "кто работает", "teacher"),
    )


def _looks_like_recording_question(text: str, plan: Mapping[str, Any]) -> bool:
    return str(plan.get("primary_intent") or "") == "recording" or _has_any(
        text,
        ("запис", "пересмотр", "пропуст", "recording", "online_recordings", "offline_recordings"),
    )


def _looks_like_address_question(text: str, plan: Mapping[str, Any]) -> bool:
    return str(plan.get("primary_intent") or "") == "address" or _has_any(
        text,
        ("адрес", "где вы", "где находит", "куда ехать", "куда ездить", "площадк", "метро"),
    )


def _asks_specific_teacher_name(text: str) -> bool:
    return bool(re.search(r"как\s+зовут|фио|имя|кто\s+в\s+лобн|кто\s+ведет|кто\s+ведёт|конкретн", text, re.I))


def _mentions_teacher_change(text: str) -> bool:
    return _has_any(text, ("не понрав", "сменить", "поменять", "другого преподав", "заменить"))


def _mentions_mendeleevo(text: str) -> bool:
    return _has_any(text, ("менделеево", "лвш", "выездн", "лагер"))


def _first_matching_fact(facts: Mapping[str, str], markers: Sequence[str]) -> tuple[str, str]:
    for key, value in facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if any(str(marker).casefold() in combined for marker in markers):
            text = " ".join(str(value or "").split())
            if text:
                return str(key), text
    return "", ""


def _brand_safe_teacher_fact_text(text: str, *, active_brand: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if active_brand != "foton":
        return cleaned
    cleaned = re.sub(r"\bМФТИ,\s*", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;")
    return cleaned


def _short_sentence(text: str, *, max_chars: int = 260) -> str:
    cleaned = " ".join(str(text or "").split())
    sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip() or cleaned
    if len(sentence) > max_chars:
        sentence = sentence[: max_chars - 1].rstrip() + "…"
    return sentence


def _has_any(text: str, markers: Sequence[str]) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    return any(str(marker).casefold().replace("ё", "е") in value for marker in markers)
