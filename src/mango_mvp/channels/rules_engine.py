from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.channels.p0_recall_spec import is_benign_hypothetical_refund


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
        "installment",
        "discount",
        "price",
        "format_choice",
        "trial",
        "camp_lvsh",
        "enrollment_process",
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
                "foton": {"address": "Москва, Верхняя Красносельская ул., 30"},
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
    if rule.rule_id == "installment":
        return _apply_installment_rule(rule, plan=plan, facts=facts, context=context)
    if rule.rule_id == "discount":
        return _apply_discount_rule(rule, plan=plan, facts=facts, context=context)
    if rule.rule_id == "price":
        return _apply_price_rule(rule, plan=plan, facts=facts, context=context)
    if rule.rule_id == "format_choice":
        return _apply_format_choice_rule(rule, plan=plan, facts=facts, context=context)
    if rule.rule_id == "trial":
        return _apply_trial_rule(rule, plan=plan, facts=facts, context=context)
    if rule.rule_id == "camp_lvsh":
        return _apply_camp_lvsh_rule(rule, plan=plan, facts=facts, context=context)
    if rule.rule_id == "enrollment_process":
        return _apply_enrollment_process_rule(rule, plan=plan, facts=facts, context=context)
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
        "installment": "installment",
        "installment_inquiry": "installment",
        "payment_method": "installment",
        "payment_by_invoice_monthly": "installment",
        "discount": "discount",
        "discounts": "discount",
        "discount_inquiry": "discount",
        "second_subject_discount": "discount",
        "multichild_discount": "discount",
        "discount_stacking": "discount",
        "price": "price",
        "pricing": "price",
        "price_fix": "price",
        "price_inquiry": "price",
        "course_price": "price",
        "format_price": "price",
        "grade_price": "price",
        "format": "format_choice",
        "format_choice": "format_choice",
        "format_choice_inquiry": "format_choice",
        "trial": "trial",
        "trial_inquiry": "trial",
        "trial_class": "trial",
        "trial_online_fragment": "trial",
        "trial_offline": "trial",
        "fragment_access": "trial",
        "camp": "camp_lvsh",
        "camp_inquiry": "camp_lvsh",
        "camp_lvsh": "camp_lvsh",
        "lvsh": "camp_lvsh",
        "live_availability": "camp_lvsh",
        "process": "enrollment_process",
        "process_inquiry": "enrollment_process",
        "enrollment": "enrollment_process",
        "enrollment_process": "enrollment_process",
        "how_to_enroll": "enrollment_process",
        "signup": "enrollment_process",
        "refund_policy": "enrollment_process",
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
        kb_fact_key, kb_address = _foton_address_from_facts(facts)
        address = kb_address or str(configured.get("address") or "Москва, Верхняя Красносельская ул., 30").strip()
        text = f"Фотон в Москве: {address}. Если нужна площадка под конкретную группу, менеджер уточнит детали."
        fact_key = kb_fact_key or "rules_registry.contact_address.foton.address"
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


def _foton_address_from_facts(facts: Mapping[str, str]) -> tuple[str, str]:
    for key, value in facts.items():
        if str(key).strip().casefold() != "contact.foton.address":
            continue
        cleaned = " ".join(str(value or "").split()).strip()
        if not cleaned:
            continue
        candidate = re.split(r"\s+[—-]\s+", cleaned, maxsplit=1)[-1].strip(" .")
        candidate = re.sub(r"^Фотон:\s*", "", candidate, flags=re.I).strip(" .")
        candidate = re.sub(r"^адрес(?:\s+и\s+место\s+занятий|\s+очных\s+занятий)?\s*[:—-]\s*", "", candidate, flags=re.I).strip(" .")
        if candidate and "москва" not in candidate.casefold():
            candidate = f"Москва, {candidate}"
        return str(key), candidate
    return "", ""


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


def _apply_installment_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_installment_question(question, plan):
        return None
    brand = _active_brand(plan, context)
    if brand not in {"foton", "unpk"} or _mentions_other_brand(question, brand):
        return None
    if _asks_invoice_transfer_without_installment(question):
        return None

    if brand == "foton":
        key, fact = _first_matching_fact(
            facts,
            ("installment", "рассроч", "оплатить обучение частями", "6, 10", "12 месяцев", "долями", "dolyami"),
        )
        if not fact:
            return None
        months = _installment_months(rule)
        text = (
            f"В Фотоне можно оплатить обучение частями: доступны варианты на {months} месяцев, "
            "а также сервис Долями. Конкретные условия и оформление зависят от выбранного способа оплаты; "
            "менеджер поможет подобрать удобный вариант."
        )
        return _rule_outcome(
            rule,
            subvariant="foton_available",
            route="bot_answer_self_for_pilot",
            text=text,
            facts={key or "rules_engine.installment.foton": fact},
            flags=("rules_engine_installment_foton",),
            checklist="Rule engine: installment — Фотон только из своего факта, не сравнивать с УНПК.",
        )

    absence_key, absence_fact = _first_matching_fact(
        facts,
        (
            "payment_options.bank_installment.absent",
            "банковской рассрочки нет",
            "рассрочки нет",
            "не банковская рассрочка",
            "отдельной банковской рассрочки нет",
        ),
    )
    period_key, period_fact = _first_matching_fact(
        facts,
        (
            "payment_options.client_safe_text.when_asked_about_installment",
            "помесячно",
            "за семестр",
            "за год",
            "растянуть оплату",
        ),
    )
    semester_key, semester_fact = _first_matching_fact_with_required(
        facts,
        ("semester", "семестр", "10%", "discounts.monthly_payment.pct"),
        required=("10%",),
    )
    year_key, year_fact = _first_matching_fact_with_required(
        facts,
        ("year", "год", "14%", "discounts.year", "year_discount"),
        required=("14%",),
    )
    if not (absence_fact and period_fact and semester_fact and year_fact):
        return None
    text = (
        "В УНПК рассрочки нет, это не банковская рассрочка, поэтому одобрение банка не требуется. "
        "Можно платить помесячно, за семестр или за год. "
        "При оплате за семестр действует скидка 10%, за год - 14%. "
        "Если нужно растянуть оплату, менеджер подскажет варианты под вашу ситуацию."
    )
    source = {
        absence_key or "rules_engine.installment.unpk.absence": absence_fact,
        period_key or "rules_engine.installment.unpk.periods": period_fact,
        semester_key or "rules_engine.installment.unpk.semester_discount": semester_fact,
        year_key or "rules_engine.installment.unpk.year_discount": year_fact,
    }
    return _rule_outcome(
        rule,
        subvariant="unpk_no_bank_installment",
        route="bot_answer_self_for_pilot",
        text=text,
        facts=source,
        flags=("rules_engine_installment_unpk_no_bank", "unpk_installment_approved_fallback_applied"),
        checklist="Rule engine: installment — УНПК: банковской рассрочки нет; периодические скидки только из фактов.",
    )


def _apply_discount_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_discount_question(question, plan):
        return None
    brand = _active_brand(plan, context)
    if brand not in {"foton", "unpk"} or _mentions_other_brand(question, brand):
        return None

    if _asks_promocode(question):
        return _rule_outcome(
            rule,
            subvariant="promocode_no_code",
            route="bot_answer_self_for_pilot",
            text="Промокод в чате не называю: менеджер проверит актуальные акции и условия под вашу заявку.",
            facts={"rules_engine.discount.promocode_policy": "Промокоды убраны из клиентского слоя; код клиенту не выдавать шаблонно."},
            flags=("rules_engine_discount_promocode_no_code",),
            checklist="Rule engine: discount — промокод не выдавать в клиентском тексте.",
        )

    second_key, second_fact = _first_matching_fact(
        facts,
        (
            "second_subject",
            "второй предмет",
            "последующий предмет",
            "второй и последующий",
            "30%",
            "20%",
        ),
    )
    online_second_key, online_second_fact = _first_matching_fact_with_required(
        facts,
        ("second_subject", "online", "онлайн", "30%"),
        required=("30%",),
    )
    offline_second_key, offline_second_fact = _first_matching_fact_with_required(
        facts,
        ("second_subject", "offline", "очн", "20%"),
        required=("20%",),
    )
    multichild_key, multichild_fact = _first_matching_fact(
        facts,
        ("multichild", "многодет", "удостоверение многодетной", "10%"),
    )
    stacking_key, stacking_fact = _first_matching_fact(
        facts,
        ("stacking", "не сумм", "наибольш", "применяется наибольшая"),
    )

    asks_second_subject = _asks_second_subject_discount(question, plan)
    asks_multichild = _asks_multichild_discount(question, plan)
    asks_stacking = _asks_discount_stacking(question, plan)

    if asks_second_subject and asks_multichild and (asks_stacking or stacking_fact):
        if not (second_fact and multichild_fact and stacking_fact):
            return None
        if brand == "foton":
            if online_second_fact and offline_second_fact:
                second_text = "На второй онлайн-предмет в Фотоне действует скидка 30%, на второй очный предмет — 20%."
                second_source = {
                    online_second_key or "rules_engine.discount.second_subject.online": online_second_fact,
                    offline_second_key or "rules_engine.discount.second_subject.offline": offline_second_fact,
                }
            elif online_second_fact:
                second_text = "На второй онлайн-предмет в Фотоне действует скидка 30%."
                second_source = {online_second_key or "rules_engine.discount.second_subject.online": online_second_fact}
            elif offline_second_fact:
                second_text = "На второй очный предмет в Фотоне действует скидка 20%."
                second_source = {offline_second_key or "rules_engine.discount.second_subject.offline": offline_second_fact}
            else:
                second_text = _short_sentence(second_fact)
                second_source = {second_key or "rules_engine.discount.second_subject": second_fact}
            text = (
                f"{second_text} Многодетная скидка — 10% по удостоверению. "
                "Скидки не суммируются: применяется наибольшая доступная."
            )
        else:
            text = (
                "На второй и последующий предмет одного ребёнка в УНПК действует скидка 20%. "
                "Многодетная скидка — 10% по удостоверению. Скидки не суммируются: применяется наибольшая доступная."
            )
            second_source = {second_key or "rules_engine.discount.second_subject": second_fact}
        return _rule_outcome(
            rule,
            subvariant="stacking_take_max",
            route="bot_answer_self_for_pilot",
            text=text,
            facts={
                **second_source,
                multichild_key or "rules_engine.discount.multichild": multichild_fact,
                stacking_key or "rules_engine.discount.stacking": stacking_fact,
            },
            flags=("rules_engine_discount_stacking_take_max",),
            checklist="Rule engine: discount — скидки не суммировать, применять наибольшую.",
        )

    if asks_stacking:
        if not stacking_fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="stacking_take_max",
            route="bot_answer_self_for_pilot",
            text="Скидки не суммируются: применяется наибольшая доступная скидка. Менеджер проверит условия под вашу ситуацию.",
            facts={stacking_key or "rules_engine.discount.stacking": stacking_fact},
            flags=("rules_engine_discount_stacking_take_max",),
            checklist="Rule engine: discount — скидки не суммировать, применять наибольшую.",
        )

    if asks_multichild:
        if not multichild_fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="multichild_status",
            route="bot_answer_self_for_pilot",
            text=(
                "Для детей из многодетной семьи есть скидка 10%; нужен статус многодетной семьи и подтверждающий документ. "
                "Скидка не суммируется с другими скидками: применяется наибольшая."
            ),
            facts={multichild_key or "rules_engine.discount.multichild": multichild_fact},
            flags=("rules_engine_discount_multichild_status",),
            checklist="Rule engine: discount — многодетная скидка по статусу семьи, не по числу детей в CRM.",
        )

    if _asks_mfti_employee_discount(question):
        key, fact = _first_matching_fact(facts, ("mfti", "мфти", "сотрудник", "10%"))
        if brand != "unpk" or not fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="mfti_staff",
            route="bot_answer_self_for_pilot",
            text="Сотрудникам МФТИ действует скидка 10%; нужен подтверждающий документ с места работы. Менеджер проверит документы и условия.",
            facts={key or "rules_engine.discount.mfti_staff": fact},
            flags=("rules_engine_discount_mfti_staff",),
            checklist="Rule engine: discount — скидка сотрудникам МФТИ только по подтверждающему документу.",
        )

    if _asks_period_discount(question):
        semester_key, semester_fact = _first_matching_fact_with_required(
            facts,
            ("monthly_payment", "семестр", "10%"),
            required=("10%",),
        )
        year_key, year_fact = _first_matching_fact_with_required(
            facts,
            ("year", "за год", "14%"),
            required=("14%",),
        )
        if brand != "unpk" or not (semester_fact and year_fact):
            return None
        return _rule_outcome(
            rule,
            subvariant="period_payment",
            route="bot_answer_self_for_pilot",
            text="В УНПК можно платить помесячно, за семестр или за год. При оплате за семестр действует скидка 10%, за год - 14%.",
            facts={
                semester_key or "rules_engine.discount.period_payment.semester": semester_fact,
                year_key or "rules_engine.discount.period_payment.year": year_fact,
            },
            flags=("rules_engine_discount_period_payment",),
            checklist="Rule engine: discount — периодические скидки УНПК только из факта.",
        )

    if asks_second_subject:
        if brand == "foton":
            if _has_any(question, ("онлайн", "дистанц")):
                if not online_second_fact:
                    return None
                text = (
                    "На второй онлайн-предмет в Фотоне действует скидка 30% для того же ребёнка. "
                    "Скидки не суммируются: если есть несколько оснований, применяется наибольшая доступная."
                )
                subvariant = "second_subject_foton_online"
                source_key, source_fact = online_second_key, online_second_fact
            elif _has_any(question, ("очно", "очный", "офлайн")):
                if not offline_second_fact:
                    return None
                text = (
                    "На второй очный предмет в Фотоне действует скидка 20% для того же ребёнка. "
                    "Скидки не суммируются: если есть несколько оснований, применяется наибольшая доступная."
                )
                subvariant = "second_subject_foton_offline"
                source_key, source_fact = offline_second_key, offline_second_fact
            else:
                if online_second_fact and offline_second_fact:
                    text = (
                        "На второй и последующий предмет одного и того же ребёнка в Фотоне действует скидка: "
                        "20% при очном формате и 30% при онлайн-формате. Скидки не суммируются."
                    )
                    source_key, source_fact = second_key or online_second_key, second_fact or online_second_fact
                elif online_second_fact:
                    text = (
                        "На второй онлайн-предмет в Фотоне действует скидка 30% для того же ребёнка. "
                        "Скидки не суммируются: если есть несколько оснований, применяется наибольшая доступная."
                    )
                    source_key, source_fact = online_second_key, online_second_fact
                elif offline_second_fact:
                    text = (
                        "На второй очный предмет в Фотоне действует скидка 20% для того же ребёнка. "
                        "Скидки не суммируются: если есть несколько оснований, применяется наибольшая доступная."
                    )
                    source_key, source_fact = offline_second_key, offline_second_fact
                elif second_fact:
                    text = _short_sentence(second_fact)
                    source_key, source_fact = second_key, second_fact
                else:
                    return None
                subvariant = "second_subject_foton"
        else:
            if not second_fact:
                return None
            text = (
                "На второй и последующий предмет одного и того же ребёнка в УНПК действует скидка 20%. "
                "Скидки не суммируются: если есть несколько оснований, применяется наибольшая доступная."
            )
            subvariant = "second_subject_unpk"
            source_key, source_fact = second_key, second_fact
        return _rule_outcome(
            rule,
            subvariant=subvariant,
            route="bot_answer_self_for_pilot",
            text=text,
            facts={source_key or f"rules_engine.discount.{subvariant}": source_fact},
            flags=("rules_engine_discount_second_subject", f"rules_engine_discount_{subvariant}"),
            checklist="Rule engine: discount — второй предмет по бренду и формату, без сравнения брендов.",
        )

    return None


def _apply_price_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_price_question(question, plan):
        return None
    brand = _active_brand(plan, context)
    if brand not in {"foton", "unpk"} or _mentions_other_brand(question, brand):
        return None
    if _mentions_camp_or_lvsh(question):
        return None
    requested_format = _requested_training_format(question, plan, context)
    if requested_format not in {"online", "offline"}:
        return None
    requested_grade = _requested_grade(plan, context)
    if requested_grade is None:
        return None
    scoped_facts = _price_facts_for_request(facts, brand=brand, requested_format=requested_format, requested_grade=requested_grade)
    if not scoped_facts:
        return None
    price_contract = _build_price_contract(
        question,
        active_brand=brand,
        requested_format=requested_format,
        requested_grade=requested_grade,
        fact_keys=tuple(scoped_facts),
    )
    selected_amount = _select_price_amount(price_contract, scoped_facts)
    if selected_amount is None:
        return None
    price_text = _price_text_from_scoped_facts(
        scoped_facts,
        selected_amount=selected_amount,
        requested_format=requested_format,
        requested_grade=requested_grade,
        question=question,
    )
    if not price_text:
        return None
    format_label = "онлайн" if requested_format == "online" else "очно"
    return _rule_outcome(
        rule,
        subvariant=f"{requested_format}_grade_scoped",
        route="bot_answer_self_for_pilot",
        text=price_text,
        facts=scoped_facts,
        flags=("rules_engine_price_format_matched", f"rules_engine_price_{requested_format}"),
        checklist=f"Rule engine: price — цена только из факта, active brand={brand}, format={format_label}, grade={requested_grade}.",
    )


def _apply_format_choice_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_format_choice_question(question, plan):
        return None
    brand = _active_brand(plan, context)
    if brand not in {"foton", "unpk"} or _mentions_other_brand(question, brand):
        return None
    if _mentions_camp_or_lvsh(question):
        return None
    online_key, online_fact = _first_matching_fact(
        facts,
        ("online_courses_format", "formats.online", "онлайн-курс", "онлайн формат", "онлайн-занят"),
    )
    offline_key, offline_fact = _first_matching_fact(
        facts,
        ("offline_courses_format", "formats.offline", "очные курсы", "очный формат", "очно"),
    )
    weekend_key, weekend_fact = _first_matching_fact(facts, ("weekend", "выходн", "суббот", "воскрес"))
    has_online = bool(online_fact)
    has_offline = bool(offline_fact)
    has_weekend = bool(weekend_fact)
    if not (has_online or has_offline or has_weekend):
        return None
    asks_disjunctive = _format_choice_is_disjunctive_question(question)
    asks_days = _has_any(question, ("по каким дням", "выходн", "суббот", "воскрес", "по дням"))
    parts: list[str] = []
    source: dict[str, str] = {}
    if asks_disjunctive:
        if has_online:
            parts.append("есть онлайн-формат")
            source[online_key or "rules_engine.format.online"] = online_fact
        if has_offline:
            parts.append("есть очный формат")
            source[offline_key or "rules_engine.format.offline"] = offline_fact
        if not parts:
            return None
        text = f"Формат за вас не выбираю: по подтверждённым фактам {', '.join(parts)}."
    elif _requested_training_format(question, plan, context) == "online":
        if not online_fact:
            return None
        source[online_key or "rules_engine.format.online"] = online_fact
        text = f"Онлайн-формат подтверждён: {_short_sentence(online_fact)}"
    elif _requested_training_format(question, plan, context) == "offline":
        if not offline_fact:
            return None
        source[offline_key or "rules_engine.format.offline"] = offline_fact
        text = f"Очный формат подтверждён: {_short_sentence(offline_fact)}"
    else:
        if has_online and has_offline:
            source[online_key or "rules_engine.format.online"] = online_fact
            source[offline_key or "rules_engine.format.offline"] = offline_fact
            text = "По подтверждённым фактам есть онлайн-формат и очный формат; формат за вас не выбираю."
        elif has_online:
            source[online_key or "rules_engine.format.online"] = online_fact
            text = f"Из подтверждённого есть онлайн-формат: {_short_sentence(online_fact)}"
        elif has_offline:
            source[offline_key or "rules_engine.format.offline"] = offline_fact
            text = f"Из подтверждённого есть очный формат: {_short_sentence(offline_fact)}"
        else:
            return None
    if asks_days and has_weekend:
        source[weekend_key or "rules_engine.format.weekend"] = weekend_fact
        text += " По дням есть разные слоты, в том числе по выходным; точный день менеджер сверит по группе."
    elif asks_days:
        text += " Точные дни конкретной группы менеджер сверит отдельно."
    return _rule_outcome(
        rule,
        subvariant="present_available_formats" if asks_disjunctive else "single_format_or_weekend",
        route="bot_answer_self_for_pilot",
        text=text,
        facts=source,
        flags=("rules_engine_format_choice_present_both" if asks_disjunctive else "rules_engine_format_choice_applied",),
        checklist="Rule engine: format_choice — не выбирать формат за клиента и не выдумывать отсутствующий формат.",
    )


def _apply_trial_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_trial_question(question, plan):
        return None
    brand = _active_brand(plan, context)
    if brand not in {"foton", "unpk"} or _mentions_other_brand(question, brand):
        return None
    if _direct_manager_request_without_process(question):
        return _rule_outcome(
            rule,
            subvariant="direct_manager_request",
            route="draft_for_manager",
            text="Передам запрос менеджеру: он свяжется и ответит по пробному формату.",
            facts={"rules_engine.trial.manager_request": "Прямой запрос менеджера по пробному не заменяется шаблоном про фрагмент."},
            flags=("rules_engine_trial_direct_manager_request",),
            checklist="Rule engine: trial — прямой запрос менеджера не переписывать в ответ про пробное.",
        )
    if brand == "foton" and _asks_offline_free_trial(question, plan, context):
        return _rule_outcome(
            rule,
            subvariant="offline_free_trial_guard",
            route="draft_for_manager",
            text=(
                "По очному формату бесплатное пробное по умолчанию не обещаю. "
                "Очный пробный шаг согласует менеджер при записи: он проверит подходящую группу, филиал и условия. "
                "Запрос передам именно как очный, без подмены на онлайн-фрагмент."
            ),
            facts={"rules_engine.trial.foton_offline_free_trial_guard": "Фотон: бесплатное очное пробное занятие не обещается шаблонно."},
            flags=("rules_engine_trial_offline_free_guard", "offline_free_trial_promise_guarded"),
            checklist="Rule engine: trial — Фотон: бесплатное очное пробное не обещать.",
        )
    if _client_negates_online(question):
        return None

    key, fact = _first_matching_fact(
        facts,
        ("trial", "пробн", "фрагмент занятия", "фрагмент урок", "online_fragment", "trial_class"),
    )
    if not fact:
        return None
    online_requested = _requested_training_format(question, plan, context) == "online" or _has_any(question, ("онлайн", "дистанц", "фрагмент"))
    data_question = _has_any(question, ("какие данные", "что нужно", "как получить", "ссылка", "способ", "регистрац", "запис"))
    ack = _has_any(question, ("жду", "давайте", "оставлю", "ок", "хорошо")) and _has_any(question, ("пробн", "фрагмент"))
    if brand == "foton" and online_requested:
        if data_question or ack:
            text = (
                "Для подбора онлайн-фрагмента в Фотоне достаточно класса, предмета и формата. "
                "Если эти данные уже есть в диалоге, повторять их не нужно; менеджер подтвердит условия просмотра."
            )
            subvariant = "online_fragment_process"
        else:
            text = _short_sentence(fact)
            if "дистанц" not in text.casefold() and "приезж" not in text.casefold():
                text += " Оформление проходит дистанционно — приезжать не нужно."
            subvariant = "online_fragment"
    elif brand == "unpk":
        if online_requested and (data_question or ack):
            text = (
                "По онлайн-фрагменту УНПК нужны только класс, предмет и формат. "
                "Если они уже есть в диалоге, повторять не нужно; менеджер подберёт фрагмент и подтвердит способ просмотра."
            )
            subvariant = "online_fragment_process"
        else:
            text = _short_sentence(fact)
            subvariant = "general"
    else:
        text = _short_sentence(fact)
        subvariant = "general"
    return _rule_outcome(
        rule,
        subvariant=subvariant,
        route="bot_answer_self_for_pilot",
        text=text,
        facts={key or "rules_engine.trial.fact": fact},
        flags=("rules_engine_trial_safe_template_applied", "trial_safe_template_applied"),
        checklist="Rule engine: trial — пробный формат только по active brand и по client-safe факту.",
    )


def _apply_camp_lvsh_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_camp_question(question, plan):
        return None
    brand = _active_brand(plan, context)
    if brand not in {"foton", "unpk"} or _mentions_other_brand(question, brand):
        return None
    if _negates_camp(question, plan):
        return None
    if _real_refund_claim(question):
        return _rule_outcome(
            rule,
            subvariant="camp_refund_p0",
            route="manager_only",
            text="Приняли обращение по возврату. Передам ответственному сотруднику, он проверит ситуацию и вернется с ответом.",
            facts={"rules_engine.camp.refund_p0": "Лагерь + реальная претензия на возврат — P0, автономный ответ запрещён."},
            flags=("rules_engine_camp_refund_p0", "high_risk_manager_only"),
            checklist="Rule engine: camp_lvsh — реальная претензия по возврату в лагере только менеджеру.",
        )
    if _asks_live_status_or_booking(question):
        return _rule_outcome(
            rule,
            subvariant="seats_or_booking_live",
            route="draft_for_manager",
            text=_camp_live_status_text(question, context),
            facts={"rules_engine.camp.seats_live": "Места и запись в лагере — live-данные; наличие проверяет менеджер."},
            flags=("rules_engine_camp_live_availability_handoff",),
            checklist="Rule engine: camp_lvsh — места/запись не обещать без live-проверки.",
        )
    if _has_any(question, ("звш", "зимн")):
        key, fact = _brand_scoped_first_matching_fact(facts, brand, ("zvsh", "зимн", "лист ожидания", "даты зимней"))
        if not fact:
            return _rule_outcome(
                rule,
                subvariant="zvsh_waitlist",
                route="draft_for_manager",
                text="Даты зимней выездной школы пока уточняются; менеджер запишет в лист ожидания и сообщит условия, когда расписание появится.",
                facts={"rules_engine.camp.zvsh_waitlist": "ЗВШ 2026/27: даты не опубликованы, доступен лист ожидания."},
                flags=("rules_engine_camp_zvsh_waitlist",),
                checklist="Rule engine: camp_lvsh — даты ЗВШ не выдумывать.",
            )
        return _rule_outcome(
            rule,
            subvariant="zvsh_waitlist",
            route="draft_for_manager",
            text=_short_sentence(fact),
            facts={key or "rules_engine.camp.zvsh": fact},
            flags=("rules_engine_camp_zvsh_waitlist",),
            checklist="Rule engine: camp_lvsh — ЗВШ только из факта/лист ожидания.",
        )

    residential = _camp_residential_requested(question, plan)
    city_day = _camp_city_day_requested(question, plan)
    living_transfer = _has_any(question, ("прожив", "питан", "трансфер", "добир", "включено", "что входит"))
    price = _has_any(question, ("цен", "стоим", "сколько", "₽", "руб"))
    grade = _requested_grade(plan, context) or _grade_from_text(question)

    markers: tuple[str, ...]
    if residential or living_transfer:
        markers = ("lvsh", "лвш", "менделеево", "прожив", "трансфер", "питание", "выездн")
    elif city_day:
        markers = ("city_camp", "city_day", "городск", "дневн", "летняя школа", "август")
    else:
        markers = ("camp", "лагер", "летняя школа", "лвш", "менделеево")
    key, fact = _brand_scoped_first_matching_fact(facts, brand, markers)
    if not fact and brand == "unpk" and residential and grade and grade >= 11:
        key, fact = _brand_scoped_first_matching_fact(facts, brand, ("grade_11", "11 класс", "11", "лвш"))
    if not fact:
        return None
    text = _short_sentence(fact, max_chars=360)
    if price and "₽" not in text and "руб" not in text.casefold():
        return None
    if "почти распродан" in text.casefold():
        suffix = " Наличие и запись по конкретной смене всё равно проверяет живой менеджер."
        if suffix.casefold() not in text.casefold():
            text += suffix
    return _rule_outcome(
        rule,
        subvariant="residential_lvsh" if residential else "city_day" if city_day else "overview",
        route="bot_answer_self_for_pilot",
        text=text,
        facts={key or "rules_engine.camp.fact": fact},
        flags=("rules_engine_camp_lvsh_applied",),
        checklist="Rule engine: camp_lvsh — бренд и тип лагеря разделены; места live не обещать.",
    )


def _apply_enrollment_process_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if _looks_like_installment_question(question, plan):
        return None
    if _has_any(question, ("закреп", "зафикс", "по текущ", "оплатить до")):
        return None
    if _real_refund_claim(question):
        return _rule_outcome(
            rule,
            subvariant="real_refund_p0",
            route="manager_only",
            text="Приняли обращение по возврату. Передам ответственному сотруднику, он проверит ситуацию и вернется с ответом.",
            facts={"rules_engine.enrollment.real_refund_p0": "Реальная претензия на возврат — P0, автономный процессный ответ запрещён."},
            flags=("rules_engine_enrollment_real_refund_p0", "high_risk_manager_only"),
            checklist="Rule engine: enrollment_process — реальный возврат только менеджеру.",
        )
    if is_benign_hypothetical_refund(question):
        key, fact = _first_matching_fact(
            facts,
            ("refund_post_payment", "неистраченных средств", "условия возврата", "возврат"),
        )
        text = (
            "Если заранее до оплаты или до старта уточняете правила возврата: возвращается остаток неистраченных средств. "
            "Точный порядок менеджер подтвердит по выбранному курсу и договору; это не оформляю как жалобу или заявление на возврат."
        )
        source = {key or "rules_engine.enrollment.presale_refund": fact or "Предпродажный вопрос о возврате: справка без обещания полной суммы."}
        return _rule_outcome(
            rule,
            subvariant="presale_refund_policy",
            route="bot_answer_self_for_pilot",
            text=text,
            facts=source,
            flags=("rules_engine_enrollment_presale_refund",),
            checklist="Rule engine: enrollment_process — предпродажный возврат не P0, но без обещания полной суммы.",
        )
    if not _looks_like_enrollment_process_question(question, plan):
        return None
    key, fact = _first_matching_fact(
        facts,
        ("process.enrollment", "как записаться", "оформ", "запис", "менеджер поможет", "заявк"),
    )
    if fact:
        text = _short_sentence(fact, max_chars=300)
    else:
        key = "rules_engine.enrollment.process"
        text = (
            "Чтобы записаться, менеджер уточнит класс, предмет, формат и подходящую группу, затем подскажет оформление заявки и оплату. "
            "Если класс, предмет и формат уже есть в диалоге, повторять их не нужно."
        )
    return _rule_outcome(
        rule,
        subvariant="how_to_enroll",
        route="bot_answer_self_for_pilot",
        text=text,
        facts={key: fact or "Запись оформляет менеджер после сверки класса, предмета, формата и группы."},
        flags=("rules_engine_enrollment_process_applied",),
        checklist="Rule engine: enrollment_process — процесс записи без подмены способов оплаты и без P0.",
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


def _looks_like_installment_question(text: str, plan: Mapping[str, Any]) -> bool:
    intent = str(plan.get("primary_intent") or "")
    if intent in {"installment", "installment_inquiry"}:
        return True
    if intent in {"payment_method", "payment_by_invoice_monthly"}:
        return _has_any(
            text,
            (
                "рассроч",
                "долями",
                "частями",
                "по частям",
                "не все сразу",
                "не всю сумму",
                "растянуть оплат",
                "платить постепенно",
                "одобр",
            ),
        ) or (
            _has_any(text, ("помесяч", "каждый месяц", "по месяцам"))
            and not _asks_invoice_transfer_without_installment(text)
        )
    return _has_any(
        text,
        (
            "рассроч",
            "долями",
            "частями",
            "по частям",
            "не все сразу",
            "не всю сумму",
            "растянуть оплат",
            "платить постепенно",
            "одобр",
        ),
    )


def _looks_like_discount_question(text: str, plan: Mapping[str, Any]) -> bool:
    return str(plan.get("primary_intent") or "") == "discount" or _has_any(
        text,
        (
            "скид",
            "льгот",
            "акци",
            "промокод",
            "второй предмет",
            "последующий предмет",
            "многодет",
            "суммир",
            "слож",
            "сотрудник мфти",
        ),
    )


def _asks_invoice_transfer_without_installment(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    invoice_or_transfer = _has_any(value, ("по счету", "по счёту", "счет", "счёт", "реквизит", "банковск", "перевод"))
    explicit_installment = _has_any(
        value,
        ("рассроч", "долями", "частями", "по частям", "не все сразу", "не всю сумму", "растянуть оплат", "одобр"),
    )
    negates_installment = _has_any(value, ("не рассроч", "не долями", "не частями", "не через банк", "не про рассроч"))
    return bool(invoice_or_transfer and (negates_installment or not explicit_installment))


def _mentions_other_brand(text: str, active_brand: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    if active_brand == "foton":
        return _has_any(value, ("унпк", "kmipt", "70369"))
    if active_brand == "unpk":
        return _has_any(value, ("фотон", "цдпо", "црдо", "cdpofoton"))
    return False


def _installment_months(rule: Rule) -> str:
    data = rule.data.get("foton") if isinstance(rule.data.get("foton"), Mapping) else {}
    months = data.get("months") if isinstance(data, Mapping) else ()
    normalized = [str(item).strip() for item in (months or ()) if str(item).strip()]
    if len(normalized) >= 2:
        return ", ".join(normalized[:-1]) + " или " + normalized[-1]
    return normalized[0] if normalized else "6, 10 или 12"


def _asks_promocode(text: str) -> bool:
    return _has_any(text, ("промокод", "код на скид", "купон"))


def _asks_second_subject_discount(text: str, plan: Mapping[str, Any]) -> bool:
    scope = str(plan.get("fact_scope") or "")
    return scope == "discount_second_subject" or _has_any(
        text,
        (
            "второй предмет",
            "второй онлайн",
            "второй очн",
            "последующий предмет",
            "два предмет",
            "еще предмет",
            "ещё предмет",
        ),
    ) or bool(re.search(r"(физик\w*[^.!?\n]{0,80}математ\w*|математ\w*[^.!?\n]{0,80}физик\w*)", text))


def _asks_multichild_discount(text: str, plan: Mapping[str, Any]) -> bool:
    scope = str(plan.get("fact_scope") or "")
    return scope == "discount_multichild" or _has_any(text, ("многодет", "трое дет", "три дет", "четверо дет", "пятеро дет"))


def _asks_discount_stacking(text: str, plan: Mapping[str, Any]) -> bool:
    scope = str(plan.get("fact_scope") or "")
    return scope == "discount_stacking" or _has_any(text, ("суммир", "слож", "вместе", "одновременно", "плюсу"))


def _asks_mfti_employee_discount(text: str) -> bool:
    return _has_any(text, ("сотрудник мфти", "работаю в мфти", "работник мфти"))


def _asks_period_discount(text: str) -> bool:
    return _has_any(text, ("семестр", "за год", "годом", "помесяч", "на год"))


def _looks_like_price_question(text: str, plan: Mapping[str, Any]) -> bool:
    intent = str(plan.get("primary_intent") or "")
    return intent in {"pricing", "price", "price_fix"} or _has_any(text, ("цен", "стоим", "сколько стоит", "прайс", "руб", "₽"))


def _looks_like_format_choice_question(text: str, plan: Mapping[str, Any]) -> bool:
    intent = str(plan.get("primary_intent") or "")
    return intent in {"format", "format_choice"} or bool(
        re.search(r"онлайн\s+или\s+очно|очно\s+или\s+онлайн|онлайн.+очно|очно.+онлайн|формат|как\s+проход", text, re.I)
    )


def _format_choice_is_disjunctive_question(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    return bool(
        ("онлайн" in value and _has_any(value, ("очно", "офлайн")) and "или" in value)
        or ("очно" in value and "онлайн" in value and "?" in value)
    )


def _mentions_camp_or_lvsh(text: str) -> bool:
    return _has_any(text, ("лагер", "лвш", "менделеев", "выездн", "смен", "camp"))


def _looks_like_trial_question(text: str, plan: Mapping[str, Any]) -> bool:
    intent = str(plan.get("primary_intent") or "")
    scope = str(plan.get("fact_scope") or "")
    return intent == "trial" or scope in {"trial_offline", "trial_online_fragment"} or _has_any(
        text,
        ("пробн", "фрагмент занятия", "фрагмент урок", "trial", "посмотреть подачу", "посмотреть урок"),
    )


def _direct_manager_request_without_process(text: str) -> bool:
    return _has_any(
        text,
        (
            "передайте менеджеру",
            "передай менеджеру",
            "дайте менеджера",
            "хочу менеджера",
            "пусть менеджер",
            "менеджер напиш",
            "менеджер скаж",
            "менеджер подтверд",
        ),
    ) and not _has_any(text, ("как", "способ", "получ", "ссыл", "регистрац", "запис", "оформ"))


def _client_negates_online(text: str) -> bool:
    return bool(re.search(r"\b(?:не|только\s+не)\s+онлайн\w*\b", str(text or "").casefold().replace("ё", "е")))


def _asks_offline_free_trial(text: str, plan: Mapping[str, Any], context: Mapping[str, Any] | None) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    scope = str(plan.get("fact_scope") or "").casefold()
    offline = scope == "trial_offline" or _requested_training_format(value, plan, context) == "offline" or _has_any(
        value,
        ("очно", "очный", "офлайн", "прийти", "приехать", "приезж"),
    )
    return bool(offline and _has_any(value, ("бесплат", "без оплаты", "free", "пробн")))


def _looks_like_camp_question(text: str, plan: Mapping[str, Any]) -> bool:
    intent = str(plan.get("primary_intent") or "")
    product = str(plan.get("product_family") or "")
    scope = str(plan.get("product_scope") or "") + " " + str(plan.get("fact_scope") or "")
    return intent in {"camp", "live_availability"} or product == "camp" or _mentions_camp_or_lvsh(" ".join([text, scope]))


def _negates_camp(text: str, plan: Mapping[str, Any]) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    intent = str(plan.get("primary_intent") or "")
    return bool(re.search(r"\bне\s+(?:лагер|лвш|лш|менделеев)\w*", value)) and intent not in {"camp", "live_availability"}


def _camp_residential_requested(text: str, plan: Mapping[str, Any]) -> bool:
    value = " ".join([str(text or ""), str(plan.get("product_scope") or ""), str(plan.get("fact_scope") or "")]).casefold().replace("ё", "е")
    no_lodging = _has_any(value, ("без прожив", "без ночев", "не выезд"))
    return _has_any(value, ("лвш", "менделеево", "выездн", "прожив", "трансфер", "питан")) and not no_lodging


def _camp_city_day_requested(text: str, plan: Mapping[str, Any]) -> bool:
    value = " ".join([str(text or ""), str(plan.get("product_scope") or ""), str(plan.get("fact_scope") or "")]).casefold().replace("ё", "е")
    return _has_any(value, ("городск", "дневн", "без проживания", "без прожив", "без ночев", "лш", "август"))


def _asks_live_status_or_booking(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    if _has_any(value, ("не про мест", "не о мест")):
        return False
    return _has_any(value, ("есть места", "налич", "брон", "заброни", "запишите", "записать на смен", "оформить место", "проверить места"))


def _camp_live_status_text(question: str, context: Mapping[str, Any] | None) -> str:
    details: list[str] = []
    grade = _known_slot_value(context, "grade") or str(_grade_from_text(question) or "")
    subject = _known_slot_value(context, "subject")
    if grade:
        details.append(f"{grade} класс")
    if subject:
        details.append(subject)
    suffix = f" по вашему запросу ({', '.join(details)})" if details else ""
    if _mentions_camp_or_lvsh(question):
        return f"По местам не буду обещать без проверки{suffix}. Передам менеджеру, чтобы он проверил наличие по конкретной смене."
    return f"По местам не буду обещать без проверки{suffix}. Передам менеджеру, чтобы он проверил наличие."


def _real_refund_claim(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    refund = _has_any(value, ("верните", "вернуть деньги", "возврат денег", "хочу вернуть", "оформить возврат", "деньги назад"))
    dispute = _has_any(value, ("оплатил", "оплатила", "оплата прошла", "занятий нет", "не было занятий", "недовол", "жалоб", "не устраивает"))
    return bool(refund and dispute and not is_benign_hypothetical_refund(value))


def _looks_like_enrollment_process_question(text: str, plan: Mapping[str, Any]) -> bool:
    intent = str(plan.get("primary_intent") or "")
    if intent in {"enrollment_process", "process", "process_inquiry"}:
        return True
    value = str(text or "").casefold().replace("ё", "е")
    return bool(
        re.search(r"\b(?:как|надо|нужно|можно\s+ли|оформ\w*|куда)\b[^.!?\n]{0,80}\b(?:запис|оформ|курс|обучен|занят)", value)
        or re.search(r"\b(?:записаться|записат(?:ь|ся)|оформиться|оформить(?:ся)?)\b", value)
    )


def _brand_scoped_first_matching_fact(facts: Mapping[str, str], active_brand: str, markers: Sequence[str]) -> tuple[str, str]:
    other_brand_markers = ("унпк", "kmipt") if active_brand == "foton" else ("фотон", "cdpofoton", "цдпо") if active_brand == "unpk" else ()
    for key, value in facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if other_brand_markers and any(marker in combined for marker in other_brand_markers):
            continue
        if any(str(marker).casefold().replace("ё", "е") in combined for marker in markers):
            text = " ".join(str(value or "").split())
            if text:
                return str(key), text
    return "", ""


def _requested_training_format(text: str, plan: Mapping[str, Any], context: Mapping[str, Any] | None) -> str:
    value = " ".join(
        str(part or "")
        for part in (
            text,
            plan.get("training_format"),
            plan.get("format"),
            plan.get("fact_scope"),
            _known_slot_value(context, "format"),
        )
    ).casefold().replace("ё", "е")
    has_online = _has_any(value, ("онлайн", "online", "дистанц"))
    has_offline = _has_any(value, ("очно", "очный", "офлайн", "offline", "сретен"))
    if has_online and not has_offline:
        return "online"
    if has_offline and not has_online:
        return "offline"
    return ""


def _requested_grade(plan: Mapping[str, Any], context: Mapping[str, Any] | None) -> int | None:
    for text in (
        _raw_question_text(plan, context),
        str(plan.get("grade") or ""),
        _known_slot_value(context, "grade"),
        _known_slot_value(context, "class"),
    ):
        grade = _grade_from_text(str(text or ""))
        if grade is not None:
            return grade
        if str(text or "").strip().isdigit():
            numeric = int(str(text).strip())
            if 1 <= numeric <= 11:
                return numeric
    return None


def _known_slot_value(context: Mapping[str, Any] | None, key: str) -> str:
    if not isinstance(context, Mapping):
        return ""
    candidates: list[Any] = []
    known = context.get("known_slots")
    if isinstance(known, Mapping):
        candidates.append(known.get(key))
    memory = context.get("dialogue_memory_view")
    if isinstance(memory, Mapping):
        memory_known = memory.get("known_slots")
        if isinstance(memory_known, Mapping):
            candidates.append(memory_known.get(key))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            value = str(candidate.get("value") or "").strip()
        else:
            value = str(candidate or "").strip()
        if value:
            return value
    return ""


def _price_facts_for_request(
    facts: Mapping[str, str],
    *,
    brand: str,
    requested_format: str,
    requested_grade: int,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if not _looks_like_regular_price_fact(combined):
            continue
        if brand == "foton" and _has_any(combined, ("унпк", "kmipt")):
            continue
        if brand == "unpk" and _has_any(combined, ("фотон", "cdpofoton", "цдпо")):
            continue
        if requested_format == "online" and not _fact_mentions_online(combined):
            continue
        if requested_format == "offline" and not _fact_mentions_offline(combined):
            continue
        if requested_format == "online" and _fact_mentions_offline(combined) and not _fact_mentions_online(combined):
            continue
        if requested_format == "offline" and _fact_mentions_online(combined) and not _fact_mentions_offline(combined):
            continue
        if not _fact_matches_grade(combined, requested_grade):
            continue
        text = " ".join(str(value or "").split())
        if text:
            result[str(key)] = text
    return result


def _looks_like_regular_price_fact(combined: str) -> bool:
    if "₽" not in combined and "руб" not in combined:
        return False
    if _has_any(combined, ("discount", "скидк", "вычет", "кэшбек", "cashback", "лагер", "лвш", "camp", "individual", "индивидуальн")):
        return False
    return bool(_has_any(combined, ("цен", "стоим", "price", "семестр", "год", "онлайн", "очно")))


def _fact_mentions_online(text: str) -> bool:
    return _has_any(text, ("онлайн", "online", "дистанц", "мтс линк"))


def _fact_mentions_offline(text: str) -> bool:
    return _has_any(text, ("очно", "очный", "очная", "офлайн", "offline"))


def _fact_matches_grade(text: str, requested_grade: int) -> bool:
    ranges = re.findall(r"\b([1-9]|10|11)\s*[-–]\s*([1-9]|10|11)\s*(?:класс|классы|кл)", text, re.I)
    if ranges:
        return any(int(start) <= requested_grade <= int(end) for start, end in ranges)
    exact = re.findall(r"\b([1-9]|10|11)\s*(?:класс|классе|кл\.?)", text, re.I)
    if exact:
        return requested_grade in {int(item) for item in exact}
    return False


def _build_price_contract(
    question: str,
    *,
    active_brand: str,
    requested_format: str,
    requested_grade: int,
    fact_keys: Sequence[str],
) -> Any:
    from mango_mvp.channels.dialogue_contract_pipeline import parse_contract

    format_value = "онлайн" if requested_format == "online" else "очно"
    raw = {
        "active_brand": active_brand,
        "current_question": question,
        "answerability": "answer_self",
        "known_slots": {
            "grade": {"value": str(requested_grade), "source": "rules_engine"},
            "format": {"value": format_value, "source": "rules_engine"},
        },
        "needed_fact_keys": list(fact_keys),
        "subquestions": [
            {
                "text": question,
                "answerable": "self",
                "needed_fact_keys": list(fact_keys),
            }
        ],
    }
    return parse_contract(raw, active_brand=active_brand, fact_key_catalog=tuple(fact_keys))


def _select_price_amount(contract: Any, facts: Mapping[str, str]) -> int | None:
    from mango_mvp.channels.dialogue_contract_pipeline import _price_for_composition

    return _price_for_composition(contract, facts)


def _price_text_from_scoped_facts(
    facts: Mapping[str, str],
    *,
    selected_amount: int,
    requested_format: str,
    requested_grade: int,
    question: str,
) -> str:
    from mango_mvp.channels.dialogue_contract_pipeline import _first_money_amount, _format_rub

    period = _requested_price_period(question)
    items: list[tuple[int, str, int]] = []
    for key, value in facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        amount = _first_money_amount(value)
        if amount is None:
            continue
        label = "год" if re.search(r"(?:^|[._\s])year(?:$|[._\s])|\bгод\s*[—-]", combined, re.I) else ""
        if not label and re.search(r"семестр|semester", combined, re.I):
            label = "семестр"
        if not label:
            label = "цена"
        if period and label != period:
            continue
        score = 0 if label == "семестр" else 1 if label == "год" else 2
        items.append((score, label, amount))
    if not items and selected_amount is not None:
        items.append((0, "цена", selected_amount))
    unique: dict[str, int] = {}
    for _, label, amount in sorted(items, key=lambda item: item[0]):
        unique.setdefault(label, amount)
    if not unique:
        return ""
    format_label = "онлайн" if requested_format == "online" else "очно"
    price_part = ", ".join(f"{label} — {_format_rub(amount)}" for label, amount in unique.items())
    return f"По подтверждённым ценам для {requested_grade} класса ({format_label}): {price_part}."


def _requested_price_period(text: str) -> str:
    value = str(text or "").casefold().replace("ё", "е")
    if re.search(r"\bгод\b|за\s+год|годов", value, re.I):
        return "год"
    if re.search(r"семестр|полугод", value, re.I):
        return "семестр"
    return ""


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


def _first_matching_fact_with_required(
    facts: Mapping[str, str],
    markers: Sequence[str],
    *,
    required: Sequence[str],
) -> tuple[str, str]:
    normalized_required = tuple(str(marker).casefold().replace("ё", "е") for marker in required if str(marker).strip())
    for key, value in facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if not all(marker in combined for marker in normalized_required):
            continue
        if any(str(marker).casefold().replace("ё", "е") in combined for marker in markers):
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
