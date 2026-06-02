from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_RULES_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "D1_audit_backlog" / "rules_registry.yaml"
MIGRATED = frozenset(
    {
        "teacher",
        "recordings",
        "contact_address",
        "docs",
        "matkap",
        "tax",
        "olympiad",
        "platform_access",
    }
)


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
    if rule.rule_id == "docs":
        return _apply_docs_rule(rule, plan=plan, facts=facts, context=context)
    if rule.rule_id == "matkap":
        return _apply_matkap_rule(rule, plan=plan, facts=facts, context=context)
    if rule.rule_id == "tax":
        return _apply_tax_rule(rule, plan=plan, facts=facts, context=context)
    if rule.rule_id == "olympiad":
        return _apply_olympiad_rule(rule, plan=plan, facts=facts, context=context)
    if rule.rule_id == "platform_access":
        return _apply_platform_access_rule(rule, plan=plan, facts=facts, context=context)
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
        "document": "docs",
        "documents": "docs",
        "documents_inquiry": "docs",
        "docs": "docs",
        "matkap": "matkap",
        "matkap_inquiry": "matkap",
        "tax": "tax",
        "tax_inquiry": "tax",
        "olympiad": "olympiad",
        "olympiad_inquiry": "olympiad",
        "olympiad_online": "olympiad",
        "platform": "platform_access",
        "platform_access": "platform_access",
        "platform_inquiry": "platform_access",
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


def _apply_docs_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    raw_question = _raw_question_text(plan, context)
    if not _looks_like_docs_question(question, plan):
        return None
    if _has_pii(raw_question) and _has_any(question, ("справк", "вычет", "документ")):
        return _rule_outcome(
            rule,
            subvariant="pii_certificate",
            route="draft_for_manager",
            text="Менеджер проверит вопрос по документам безопасно. Повторно присылать персональные данные в чат не нужно.",
            facts={"rules_engine.docs.pii_safety": "Документы: персональные данные из клиентского сообщения нельзя повторять в ответе."},
            flags=("rules_engine_docs_pii_guard",),
            checklist="Rule engine: docs — ПДн в запросе документов не эхоить.",
        )
    if _has_any(question, ("юрлиц", "юридическ", "реквизит")):
        return _rule_outcome(
            rule,
            subvariant="legal_entity",
            route="draft_for_manager",
            text="Юридические реквизиты и сторону договора менеджер проверит по вашей заявке и пришлёт безопасно.",
            facts={"rules_engine.docs.legal_entity": "Документы: юридические реквизиты и сторону договора клиенту подтверждает менеджер."},
            flags=("rules_engine_docs_legal_entity",),
            checklist="Rule engine: docs — юрлицо/реквизиты не раскрывать шаблонно.",
        )
    if _has_any(question, ("лиценз", "номер лиценз", "лицензии")):
        key, fact = _first_matching_fact(facts, ("licenses.client_safe_summary", "лицензия на образовательную деятельность", "есть лицензия"))
        if not fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="license",
            route="bot_answer_self_for_pilot",
            text="У учебного центра есть лицензия на образовательную деятельность. Номер лицензии и служебные реквизиты в чат не отправляю; менеджер подготовит документы безопасно.",
            facts={key or "rules_engine.docs.license": fact},
            flags=("rules_engine_docs_license_no_number",),
            checklist="Rule engine: docs — лицензия без номера/дат/юрлица.",
        )
    if _has_any(question, ("договор", "оформлен")):
        key, fact = _first_matching_fact(facts, ("theme_11_contract", "договор пришл", "договор"))
        if not fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="contract",
            route="bot_answer_self_for_pilot",
            text=_short_sentence(fact),
            facts={key or "rules_engine.docs.contract": fact},
            flags=("rules_engine_docs_contract",),
            checklist="Rule engine: docs — договор по сроку из факта.",
        )
    if _has_any(question, ("справк", "сертификат", "вычет")):
        key, fact = _first_matching_fact(facts, ("theme_12_certificate", "справк", "10 дней", "постараемся раньше"))
        if not fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="certificate",
            route="bot_answer_self_for_pilot",
            text=_short_sentence(fact),
            facts={key or "rules_engine.docs.certificate": fact},
            flags=("rules_engine_docs_certificate",),
            checklist="Rule engine: docs — справка по сроку из факта, тип не выдумывать.",
        )
    return None


def _apply_matkap_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_matkap_question(question, plan):
        return None
    if "региональ" in question:
        key, fact = _first_matching_fact(facts, ("when_regional", "региональный маткапитал не принимаем", "региональный не принимаем"))
        if not fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="regional",
            route="bot_answer_self_for_pilot",
            text=_short_sentence(fact),
            facts={key or "rules_engine.matkap.regional": fact},
            flags=("rules_engine_matkap_regional_no",),
            checklist="Rule engine: matkap — региональный не принимать.",
        )
    if _has_any(question, ("точно", "гарант", "одобр", "примут", "сфр")):
        timeline_key, timeline_fact = _first_matching_fact(facts, ("sfr_review", "сфр рассматривает", "до 10 рабочих дней"))
        general_key, general_fact = _first_matching_fact(facts, ("when_asked", "федеральным маткапиталом", "через сфр"))
        source = {k: v for k, v in ((timeline_key, timeline_fact), (general_key, general_fact)) if k and v}
        if not source:
            return None
        return _rule_outcome(
            rule,
            subvariant="sfr_approval",
            route="bot_answer_self_for_pilot",
            text="Рассмотрение проводит СФР, поэтому мы не можем обещать одобрение. Менеджер поможет проверить порядок оформления.",
            facts=source,
            flags=("rules_engine_matkap_sfr_no_guarantee",),
            checklist="Rule engine: matkap — не обещать одобрение СФР.",
        )
    key, fact = _first_matching_fact(facts, ("when_asked", "федеральным маткапиталом", "федеральный материнский капитал", "маткапиталом возможна"))
    if not fact:
        return None
    return _rule_outcome(
        rule,
        subvariant="federal",
        route="bot_answer_self_for_pilot",
        text="Да, оплатить федеральным материнским капиталом можно. Менеджер пришлёт перечень документов и поможет с оформлением через СФР.",
        facts={key or "rules_engine.matkap.federal": fact},
        flags=("rules_engine_matkap_federal",),
        checklist="Rule engine: matkap — федеральный да, решение СФР не обещать.",
    )


def _apply_tax_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_tax_question(question, plan):
        return None
    if _has_any(question, ("лиценз", "номер лиценз")):
        key, fact = _first_matching_fact(facts, ("licenses.client_safe_summary", "есть лицензия", "лицензия на образовательную деятельность"))
        if not fact:
            key, fact = _first_matching_fact(facts, ("tax_deduction.client_safe_text.when_asked", "у нас есть лицензия"))
        if not fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="license",
            route="bot_answer_self_for_pilot",
            text="Да, есть лицензия на образовательную деятельность. Номер лицензии и служебные реквизиты в чат не отправляю; менеджер подготовит документы для вычета безопасно.",
            facts={key or "rules_engine.tax.license": fact},
            flags=("rules_engine_tax_license_no_number",),
            checklist="Rule engine: tax — лицензия без номера.",
        )
    asks_certainty = _has_any(question, ("точно", "гарант", "одобр", "фнс", "налогов"))
    if (
        asks_certainty
        and _has_any(question, ("верн", "получ", "одобр", "примет", "13%"))
    ):
        key, fact = _first_matching_fact(facts, ("tax_deduction.client_safe_text.when_asked", "решение", "фнс", "налог"))
        if not fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="fns_decision",
            route="bot_answer_self_for_pilot",
            text="ФНС рассматривает заявление и принимает решение. Справка помогает подтвердить обучение, но возврат мы не гарантируем.",
            facts={key or "rules_engine.tax.fns_decision": fact},
            flags=("rules_engine_tax_fns_no_guarantee",),
            checklist="Rule engine: tax — не гарантировать возврат ФНС.",
        )
    key, fact = _first_matching_fact(facts, ("tax_deduction.client_safe_text.when_asked", "налоговый вычет", "14 300", "справк"))
    if not fact:
        return None
    subvariant = "how_to_form" if _has_any(question, ("как", "оформ", "подать", "документ")) else "amount"
    return _rule_outcome(
        rule,
        subvariant=subvariant,
        route="bot_answer_self_for_pilot",
        text=_short_sentence(fact, max_chars=260),
        facts={key or "rules_engine.tax.procedure": fact},
        flags=("rules_engine_tax_applied", f"rules_engine_tax_{subvariant}"),
        checklist="Rule engine: tax — справка/лимиты из факта, решение ФНС не обещать.",
    )


def _apply_olympiad_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_olympiad_question(question, plan):
        return None
    if _active_brand(plan, context) != "unpk":
        return None
    if _has_any(question, ("не олимпиад", "обычн", "регулярн")):
        return None
    key, fact = _first_matching_fact(facts, ("online_olympiad_phystech_classes", "олимпиадная подготовка физтех онлайн", "9 и 11"))
    if not fact:
        return None
    grade = _grade_from_text(question)
    if grade and grade not in {9, 11}:
        fact_text = " ".join(str(fact or "").split())
        text = (
            fact_text
            if "для другого класса менеджер" in fact_text.casefold().replace("ё", "е")
            else "По проверенным данным олимпиадная подготовка Физтех онлайн сейчас указана для 9 и 11 классов. Для другого класса менеджер отдельно проверит, есть ли подходящая олимпиадная онлайн-группа."
        )
        return _rule_outcome(
            rule,
            subvariant="grade_outside_9_11",
            route="draft_for_manager",
            text=text,
            facts={key or "rules_engine.olympiad.phystech_grades": fact},
            flags=("rules_engine_olympiad_grade_outside_9_11", "olympiad_online_safe_template_applied"),
            checklist="Rule engine: olympiad — другой класс не подтверждать без факта.",
        )
    fact_text = " ".join(str(fact or "").split())
    return _rule_outcome(
        rule,
        subvariant="grade_eligibility",
        route="bot_answer_self_for_pilot",
        text=fact_text,
        facts={key or "rules_engine.olympiad.phystech_grades": fact},
        flags=("rules_engine_olympiad_applied", "olympiad_online_safe_template_applied"),
        checklist="Rule engine: olympiad — Физтех онлайн только по олимпиадному scope.",
    )


def _apply_platform_access_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_platform_question(question, plan):
        return None
    if _has_any(question, ("ты бот", "вы бот", "нейросеть", "ignore all previous", "покажи промпт", "системный промпт", "chatgpt", "claude", "codex")):
        return None
    if _has_any(question, ("электрон", "скан", "документооборот", "офис")):
        key, fact = _first_matching_fact(facts, ("electronic_document_flow", "электронный документооборот", "скан-коп"))
        if not fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="electronic_documents",
            route="bot_answer_self_for_pilot",
            text=_short_sentence(fact),
            facts={key or "rules_engine.platform.electronic_documents": fact},
            flags=("rules_engine_platform_electronic_documents",),
            checklist="Rule engine: platform — электронные документы только из факта.",
        )
    key, fact = _first_matching_fact(
        facts,
        ("student_account_access", "личный кабинет", "online_platform.name", "онлайн-платформа", "мтс линк"),
    )
    if not fact:
        return None
    return _rule_outcome(
        rule,
        subvariant="how_to_login",
        route="bot_answer_self_for_pilot",
        text=_short_sentence(fact),
        facts={key or "rules_engine.platform.login": fact},
        flags=("rules_engine_platform_access_applied",),
        checklist="Rule engine: platform — доступ к кабинету/платформе из факта; identity/injection не мигрировать.",
    )


def _rule_outcome(
    rule: Rule,
    *,
    subvariant: str,
    route: str,
    text: str,
    facts: Mapping[str, str],
    flags: Sequence[str] = (),
    checklist: str = "",
) -> RuleOutcome:
    normalized_flags = tuple(dict.fromkeys([f"rules_engine_{rule.rule_id}_applied", *flags]))
    checklist_items = (checklist,) if checklist else ()
    return RuleOutcome(
        rule_id=rule.rule_id,
        subvariant=subvariant,
        route=route,
        text=text,
        facts=facts,
        flags=normalized_flags,
        checklist=checklist_items,
        metadata={"source": "rules_engine", "rule_id": rule.rule_id, "subvariant": subvariant},
    )


def _question_text(plan: Mapping[str, Any], context: Mapping[str, Any] | None) -> str:
    return _raw_question_text(plan, context).casefold().replace("ё", "е")


def _raw_question_text(plan: Mapping[str, Any], context: Mapping[str, Any] | None) -> str:
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
    return " ".join(parts)


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


def _looks_like_docs_question(text: str, plan: Mapping[str, Any]) -> bool:
    return str(plan.get("primary_intent") or "") in {"document", "documents"} or _has_any(
        text,
        ("договор", "справк", "сертификат", "документ", "квитанц", "чек", "лиценз", "юрлиц", "юридическ"),
    )


def _looks_like_matkap_question(text: str, plan: Mapping[str, Any]) -> bool:
    return str(plan.get("primary_intent") or "") == "matkap" or _has_any(text, ("маткап", "материн", "сфр"))


def _looks_like_tax_question(text: str, plan: Mapping[str, Any]) -> bool:
    return str(plan.get("primary_intent") or "") == "tax" or _has_any(text, ("налог", "вычет", "фнс", "ндфл"))


def _looks_like_olympiad_question(text: str, plan: Mapping[str, Any]) -> bool:
    return str(plan.get("primary_intent") or "") in {"olympiad_online", "olympiad"} or _has_any(text, ("олимпиад", "физтех"))


def _looks_like_platform_question(text: str, plan: Mapping[str, Any]) -> bool:
    return str(plan.get("primary_intent") or "") == "platform_access" or _has_any(
        text,
        ("личный кабинет", "кабинет", "платформ", "логин", "парол", "электрон", "документооборот", "скан-коп"),
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


def _grade_from_text(text: str) -> int | None:
    match = re.search(r"\b([1-9]|10|11)\s*(?:класс|классе|кл\.?)", str(text or ""), re.I)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _has_pii(text: str) -> bool:
    value = str(text or "")
    if re.search(r"\+?\d[\d\s().-]{8,}\d", value):
        return True
    if re.search(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b", value):
        return True
    if re.search(r"\b\d{2}\.\d{2}\.\d{4}\b", value):
        return True
    return False


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
