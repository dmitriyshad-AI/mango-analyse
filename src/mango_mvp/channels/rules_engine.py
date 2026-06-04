from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mango_mvp.channels.p0_recall_spec import is_benign_hypothetical_refund


DEFAULT_RULES_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "D1_audit_backlog" / "rules_registry.yaml"
SELLING_MODE_ENV = "TELEGRAM_A_SELLING_MODE"
SELLING_SIGNALS_FULL_ENV = "TELEGRAM_A_SELLING_SIGNALS_FULL"
COVERAGE_ENV = "TELEGRAM_A_COVERAGE"
A_PROACTIVE_ENV = "TELEGRAM_A_PROACTIVE"
A_RICH_FORMAT_ENV = "TELEGRAM_A_RICH_FORMAT"
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
        "schedule",
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
    cross_brand = _cross_brand_current_center_outcome(rule, plan=plan, context=context)
    if cross_brand is not None:
        return cross_brand
    outcome: RuleOutcome | None
    if rule.rule_id == "teacher":
        outcome = _apply_teacher_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "recordings":
        outcome = _apply_recordings_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "contact_address":
        outcome = _apply_contact_address_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "docs":
        outcome = _apply_docs_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "matkap":
        outcome = _apply_matkap_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "tax":
        outcome = _apply_tax_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "olympiad":
        outcome = _apply_olympiad_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "platform_access":
        outcome = _apply_platform_access_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "installment":
        outcome = _apply_installment_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "discount":
        outcome = _apply_discount_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "price":
        outcome = _apply_price_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "format_choice":
        outcome = _apply_format_choice_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "trial":
        outcome = _apply_trial_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "camp_lvsh":
        outcome = _apply_camp_lvsh_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "enrollment_process":
        outcome = _apply_enrollment_process_rule(rule, plan=plan, facts=facts, context=context)
    elif rule.rule_id == "schedule":
        outcome = _apply_schedule_rule(rule, plan=plan, facts=facts, context=context)
    else:
        return None
    if outcome is None:
        outcome = _selling_signal_fallback_outcome(rule, plan=plan, facts=facts, context=context)
    return _apply_selling_variants(outcome, rule=rule, plan=plan, facts=facts, context=context)


def _cross_brand_current_center_outcome(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    brand = _active_brand(plan, context)
    if brand not in {"foton", "unpk"} or not _mentions_other_brand(question, brand):
        return None
    if not (_selling_signals_full_enabled(context) or bool(_selling_view(plan, context))):
        return None
    if _real_refund_claim(question):
        return None
    return RuleOutcome(
        rule_id=rule.rule_id,
        subvariant="cross_brand_current_center",
        route="bot_answer_self_for_pilot",
        text="Это отдельные организации, по вашему вопросу сориентирую в рамках текущего учебного центра.",
        facts={"rules_engine.cross_brand.current_center": "Кросс-бренд: не сравнивать организации, отвечать только в рамках active brand."},
        flags=("rules_engine_cross_brand_current_center", "cross_brand_safe_template_applied"),
        checklist=("Rule engine: cross-brand — не консультировать по другому бренду и не сравнивать условия.",),
        metadata={"source": "rules_engine", "rule_id": rule.rule_id, "subvariant": "cross_brand_current_center", "brand": brand},
    )


def _selling_signal_fallback_outcome(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    selling = _selling_view(plan, context)
    if not selling:
        return None
    question = _question_text(plan, context)
    brand = _active_brand(plan, context)
    if brand not in {"foton", "unpk"} or _mentions_other_brand(question, brand) or _real_refund_claim(question):
        return None
    if str(selling.get("readiness") or "exploring") == "ready":
        key, fact = _clean_selling_support_fact(
            facts,
            brand,
            ("process.enrollment", "enrollment", "запис", "заявк", "менеджер уточнит класс", "оформ"),
        )
        if fact:
            return RuleOutcome(
                rule_id=rule.rule_id,
                subvariant="selling_readiness_enrollment_fallback",
                route="bot_answer_self_for_pilot",
                text=f"Если готовы к записи: {_short_sentence(fact, max_chars=300)}",
                facts={key or "rules_engine.selling.readiness_enrollment": fact},
                flags=("rules_engine_selling_readiness", "rules_engine_selling_readiness_fallback"),
                checklist=("Rule engine: selling readiness — шаг записи только из client-safe enrollment-факта.",),
                metadata={"source": "rules_engine", "rule_id": rule.rule_id, "subvariant": "selling_readiness_enrollment_fallback"},
            )
        return RuleOutcome(
            rule_id=rule.rule_id,
            subvariant="selling_readiness_no_fact",
            route="draft_for_manager",
            text="Менеджер подтвердит порядок записи по выбранному курсу.",
            facts={"rules_engine.selling.readiness_no_fact": "Нет client-safe факта с шагами записи; порядок подтверждает менеджер."},
            flags=("rules_engine_selling_readiness", "rules_engine_selling_readiness_no_fact_handoff"),
            checklist=("Rule engine: selling readiness — без enrollment-факта не расписывать шаги.",),
            metadata={"source": "rules_engine", "rule_id": rule.rule_id, "subvariant": "selling_readiness_no_fact"},
        )
    if bool(selling.get("exit_signal")):
        suffix, suffix_facts = _selling_exit_step(facts, active_brand=brand)
        text = suffix or "Спокойно подумайте; если нужно, подскажу, что важно для решения по этому варианту."
        return RuleOutcome(
            rule_id=rule.rule_id,
            subvariant="selling_exit_fallback",
            route="bot_answer_self_for_pilot",
            text=text,
            facts=suffix_facts or {"rules_engine.selling.exit_neutral": "Exit signal: мягкий нейтральный следующий шаг без давления."},
            flags=("rules_engine_selling_exit_signal", "rules_engine_selling_exit_fallback"),
            checklist=("Rule engine: selling exit — мягкий удерживающий шаг без давления и без неподтверждённых фактов.",),
            metadata={"source": "rules_engine", "rule_id": rule.rule_id, "subvariant": "selling_exit_fallback"},
        )
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
        "schedule": "schedule",
        "schedule_inquiry": "schedule",
        "class_schedule": "schedule",
        "schedule_weekend": "schedule",
        "schedule_start": "schedule",
        "schedule_frequency": "schedule",
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
        text = f"Для ЛВШ Менделеево ориентир по преподавателям такой: {_short_sentence(fact_text)} Менеджер уточнит состав по группе."
    elif _asks_specific_teacher_name(question):
        subvariant = "specific_name"
        route = "draft_for_manager"
        text = (
            "Конкретное имя преподавателя зависит от группы. "
            f"Ориентир по составу: {_short_sentence(fact_text)} Менеджер уточнит преподавателя по вашей группе."
        )
    else:
        text = f"Про преподавателей могу дать такой ориентир: {_short_sentence(fact_text)} Конкретный состав по группе менеджер уточнит отдельно."

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
        text = f"По очным занятиям ориентир такой: {_short_sentence(offline_fact)}"
        source = {offline_key or "rules_engine.recordings.offline": offline_fact}
    elif wants_online and online_fact:
        subvariant = "online"
        text = f"По онлайн-занятиям порядок такой: {_short_sentence(online_fact)}"
        source = {online_key or "rules_engine.recordings.online": online_fact}
    elif online_fact and offline_fact:
        subvariant = "online_and_offline"
        text = f"По онлайн-занятиям: {_short_sentence(online_fact)} По очным — {_short_sentence(offline_fact)}"
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
        text = f"В Москве Фотон находится по адресу: {address}. Если нужна площадка под конкретную группу, менеджер уточнит детали."
        fact_key = kb_fact_key or "rules_registry.contact_address.foton.address"
        fact_text = f"Адрес очных занятий Фотона: {address}."
    elif brand == "unpk":
        configured = rule.data.get("unpk") if isinstance(rule.data.get("unpk"), Mapping) else {}
        addresses = tuple(str(item).strip() for item in (configured.get("addresses") or ()) if str(item).strip())
        if not addresses:
            return None
        text = "УНПК проводит очные занятия на площадках: " + "; ".join(addresses) + ". Если выбираете очные занятия, уточните, какая площадка удобнее."
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
            else "Проверенные данные сейчас такие: олимпиадная подготовка Физтех онлайн указана для 9 и 11 классов. Для другого класса менеджер отдельно проверит, есть ли подходящая олимпиадная онлайн-группа."
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
    if _coverage_enabled(context):
        covered = _apply_price_coverage_rule(rule, plan=plan, facts=facts, context=context, question=question, brand=brand)
        if covered is not None:
            return covered
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


def _apply_price_coverage_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
    question: str,
    brand: str,
) -> RuleOutcome | None:
    requested_grade = _requested_grade(plan, context)
    if _asks_two_format_price(question):
        if requested_grade is None:
            return None
        source: dict[str, str] = {}
        parts: list[str] = []
        missing: list[str] = []
        for requested_format in ("online", "offline"):
            scoped_facts = _price_facts_for_request(facts, brand=brand, requested_format=requested_format, requested_grade=requested_grade)
            if not scoped_facts:
                missing.append("онлайн" if requested_format == "online" else "очно")
                continue
            price_contract = _build_price_contract(
                question,
                active_brand=brand,
                requested_format=requested_format,
                requested_grade=requested_grade,
                fact_keys=tuple(scoped_facts),
            )
            selected_amount = _select_price_amount(price_contract, scoped_facts)
            if selected_amount is None:
                missing.append("онлайн" if requested_format == "online" else "очно")
                continue
            price_text = _price_text_from_scoped_facts(
                scoped_facts,
                selected_amount=selected_amount,
                requested_format=requested_format,
                requested_grade=requested_grade,
                question=question,
            )
            if not price_text:
                missing.append("онлайн" if requested_format == "online" else "очно")
                continue
            parts.append(price_text)
            source.update(scoped_facts)
        if not parts:
            return None
        if missing:
            parts.append(f"По формату {'/'.join(missing)} точную стоимость менеджер сверит отдельно.")
        return _rule_outcome(
            rule,
            subvariant="coverage_two_formats",
            route="bot_answer_self_for_pilot",
            text=" ".join(parts),
            facts=source,
            flags=(
                "rules_engine_coverage_price_two_formats",
                *("rules_engine_coverage_partial_missing" for _ in (1,) if missing),
            ),
            checklist="Rule engine coverage: составной вопрос по двум форматам — отвечать только найденными price-фактами.",
        )

    if _asks_multi_subject_price(question):
        discount_key, discount_fact = _first_matching_fact(
            facts,
            ("second_subject", "второй предмет", "последующий предмет", "на второй", "скидк"),
        )
        if not discount_fact:
            return None
        source = {discount_key or "rules_engine.coverage.second_subject_discount": discount_fact}
        parts = [f"По второму/последующим предметам подтверждено: {_short_sentence(discount_fact)}"]
        requested_format = _requested_training_format(question, plan, context)
        if requested_grade is not None and requested_format in {"online", "offline"}:
            scoped_facts = _price_facts_for_request(facts, brand=brand, requested_format=requested_format, requested_grade=requested_grade)
            if scoped_facts:
                price_contract = _build_price_contract(
                    question,
                    active_brand=brand,
                    requested_format=requested_format,
                    requested_grade=requested_grade,
                    fact_keys=tuple(scoped_facts),
                )
                selected_amount = _select_price_amount(price_contract, scoped_facts)
                if selected_amount is not None:
                    price_text = _price_text_from_scoped_facts(
                        scoped_facts,
                        selected_amount=selected_amount,
                        requested_format=requested_format,
                        requested_grade=requested_grade,
                        question=question,
                    )
                    if price_text:
                        parts.insert(0, price_text)
                        source.update(scoped_facts)
        parts.append("Итоговую сумму по выбранным предметам менеджер сверит отдельно, чтобы не выдумывать расчёт.")
        return _rule_outcome(
            rule,
            subvariant="coverage_multi_subjects",
            route="bot_answer_self_for_pilot",
            text=" ".join(parts),
            facts=source,
            flags=("rules_engine_coverage_price_multi_subjects",),
            checklist="Rule engine coverage: несколько предметов — не считать итог без факта, дать price/discount факты и честный хвост.",
        )

    return None


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
        text = f"Подскажу, что подтверждено, а формат удобнее выбрать вам: {', '.join(parts)}."
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
            text = "Подтверждены онлайн-формат и очный формат; формат за вас не выбираю."
        elif has_online:
            source[online_key or "rules_engine.format.online"] = online_fact
            text = f"Онлайн-формат подтверждён: {_short_sentence(online_fact)}"
        elif has_offline:
            source[offline_key or "rules_engine.format.offline"] = offline_fact
            text = f"Очный формат подтверждён: {_short_sentence(offline_fact)}"
        else:
            return None
    if asks_days and has_weekend:
        source[weekend_key or "rules_engine.format.weekend"] = weekend_fact
        text += " Дни бывают разными, в том числе по выходным; точный день менеджер сверит по группе."
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
    key, fact = _first_matching_fact(
        facts,
        ("trial", "пробн", "фрагмент занятия", "фрагмент урок", "online_fragment", "trial_class"),
    )
    if _direct_manager_request_without_process(question):
        if fact:
            text = "Да, пробный формат есть — менеджер подберёт доступный вариант и запишет."
            source = {key or "rules_engine.trial.manager_request_fact": fact}
        else:
            text = "Передам запрос менеджеру: он свяжется и ответит по пробному формату."
            source = {"rules_engine.trial.manager_request": "Прямой запрос менеджера по пробному без client-safe факта."}
        return _rule_outcome(
            rule,
            subvariant="direct_manager_request",
            route="draft_for_manager",
            text=text,
            facts=source,
            flags=("rules_engine_trial_direct_manager_request",),
            checklist="Rule engine: trial — прямой запрос менеджера: факт о пробном можно подтвердить, детали подбирает менеджер.",
        )
    if brand == "foton" and _asks_offline_free_trial(question, plan, context):
        return _rule_outcome(
            rule,
            subvariant="offline_free_trial_guard",
            route="draft_for_manager",
            text=(
                "Бесплатное пробное по умолчанию не обещаю для очного формата. "
                "Очный пробный шаг согласует менеджер при записи: он проверит подходящую группу, филиал и условия. "
                "Запрос передам именно как очный, без подмены на онлайн-фрагмент."
            ),
            facts={"rules_engine.trial.foton_offline_free_trial_guard": "Фотон: бесплатное очное пробное занятие не обещается шаблонно."},
            flags=("rules_engine_trial_offline_free_guard", "offline_free_trial_promise_guarded"),
            checklist="Rule engine: trial — Фотон: бесплатное очное пробное не обещать.",
        )
    if _client_negates_online(question):
        return None

    if not fact:
        return None
    online_requested = _requested_training_format(question, plan, context) == "online" or _has_any(question, ("онлайн", "дистанц", "фрагмент"))
    data_question = _has_any(question, ("какие данные", "что нужно", "как получить", "ссылка", "способ", "регистрац", "запис"))
    ack = _has_any(question, ("жду", "давайте", "оставлю", "ок", "хорошо")) and _has_any(question, ("пробн", "фрагмент"))
    if brand == "foton" and online_requested:
        if data_question or ack:
            text = (
                "Для подбора онлайн-фрагмента в Фотоне достаточно класса, предмета и формата. "
                "Менеджер подтвердит условия просмотра и согласует доступный вариант."
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
                "Для онлайн-фрагмента УНПК нужны только класс, предмет и формат. "
                "Менеджер подберёт фрагмент и подтвердит способ просмотра."
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

    if _asks_camp_transfer(question):
        key, fact = _camp_transfer_fact(facts, brand)
        if not fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="transfer",
            route="bot_answer_self_for_pilot",
            text=_camp_prefixed_answer("По трансферу", fact),
            facts={key or "rules_engine.camp.transfer": fact},
            flags=("rules_engine_camp_transfer_fact",),
            checklist="Rule engine: camp_lvsh — трансфер отвечать только transfer-фактом, не ценой ЛВШ.",
        )

    if _asks_camp_included(question):
        composition = _camp_included_composition(facts, brand)
        if composition is not None:
            return _rule_outcome(
                rule,
                subvariant="included_composition",
                route="bot_answer_self_for_pilot",
                text=composition.text,
                facts=composition.facts,
                flags=("rules_engine_camp_included_composition",),
                checklist="Rule engine: camp_lvsh — «что входит» собирается из проживания/питания/трансфера, не из общего program-hours факта.",
            )

    if price:
        key, fact = _camp_price_fact(facts, brand)
        if not fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="price",
            route="bot_answer_self_for_pilot",
            text=_camp_prefixed_answer("По стоимости ЛВШ", fact),
            facts={key or "rules_engine.camp.price": fact},
            flags=("rules_engine_camp_price_fact",),
            checklist="Rule engine: camp_lvsh — цена ЛВШ только из price-факта active brand.",
        )

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
    text = _camp_clean_fact_text(fact, max_chars=360)
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
        text = _warm_enrollment_process_text(fact)
    else:
        key = "rules_engine.enrollment.process"
        text = "Менеджер подтвердит порядок записи по выбранному курсу."
        return _rule_outcome(
            rule,
            subvariant="how_to_enroll_no_fact",
            route="draft_for_manager",
            text=text,
            facts={key: "Нет client-safe факта с шагами записи; порядок подтверждает менеджер."},
            flags=("rules_engine_enrollment_process_no_fact_handoff",),
            checklist="Rule engine: enrollment_process — без client-safe факта о записи не расписывать шаги автономно.",
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


def _warm_enrollment_process_text(fact: str) -> str:
    text = _short_sentence(fact, max_chars=300)
    lower = text.casefold()
    if "менеджер" in lower and ("оформ" in lower or "запис" in lower):
        return text
    return f"{text} Менеджер уточнит конкретную группу и поможет оформить запись."


def _apply_schedule_rule(
    rule: Rule,
    *,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    question = _question_text(plan, context)
    if not _looks_like_schedule_question(question, plan):
        return None
    brand = _active_brand(plan, context)
    if brand not in {"foton", "unpk"} or _mentions_other_brand(question, brand):
        return None
    scoped_facts = _brand_scoped_schedule_facts(facts, brand)
    if not scoped_facts:
        return None

    if _asks_schedule_frequency(question, plan):
        key, fact = _first_schedule_fact(scoped_facts, ("weekly_lessons", "1 раз в неделю", "раз в неделю"))
        if not fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="weekly_lessons",
            route="bot_answer_self_for_pilot",
            text=f"Подтверждённый график сейчас такой: {_short_sentence(fact)} Точные дни конкретной группы менеджер сверит по классу, предмету и площадке.",
            facts={key or "rules_engine.schedule.weekly_lessons": fact},
            flags=("rules_engine_schedule_weekly_lessons", "schedule_frequency_safe_template_applied"),
            checklist="Rule engine: schedule — кадэнс только из факта; дни группы не выдумывать.",
        )

    if _asks_schedule_start(question, plan):
        key, fact = _schedule_start_fact(scoped_facts, question)
        if not fact:
            return None
        return _rule_outcome(
            rule,
            subvariant="start_date",
            route="bot_answer_self_for_pilot",
            text=f"По старту занятий: {_short_sentence(fact)}",
            facts={key or "rules_engine.schedule.start": fact},
            flags=("rules_engine_schedule_start_date",),
            checklist="Rule engine: schedule — старт занятий из v6.4-факта по бренду/площадке.",
        )

    if _asks_weekend_schedule(question, plan):
        key, fact = _schedule_group_fact(scoped_facts, question, require_weekend=True)
        if fact:
            text = (
                f"По выходным есть такие варианты: {_short_sentence(fact, max_chars=260)} "
                "Точный вариант под вашу группу менеджер сверит."
            )
            source = {key or "rules_engine.schedule.weekend_group": fact}
        else:
            key, fact = _first_schedule_fact(scoped_facts, ("выходн", "суббот", "воскрес", "слот"))
            if not fact:
                return _schedule_manager_check_outcome(rule, subvariant="weekend_slots", question=question, facts=scoped_facts)
            text = f"Есть варианты на выходных: {_short_sentence(fact)} Точный день и группу менеджер сверит."
            source = {key or "rules_engine.schedule.weekend_guidance": fact}
        return _rule_outcome(
            rule,
            subvariant="weekend_slots",
            route="bot_answer_self_for_pilot",
            text=text,
            facts=source,
            flags=("rules_engine_schedule_weekend_soft",),
            checklist="Rule engine: schedule — выходные как мягкий ориентир, конкретный слот только из факта.",
        )

    if _asks_exact_class_schedule(question, plan):
        key, fact = _schedule_group_fact(scoped_facts, question)
        if fact:
            return _rule_outcome(
                rule,
                subvariant="class_days_time",
                route="bot_answer_self_for_pilot",
                text=f"Нашёл такую группу: {_short_sentence(fact, max_chars=300)} Если нужна финальная сверка по конкретной группе, менеджер подтвердит актуальность.",
                facts={key or "rules_engine.schedule.group": fact},
                flags=("rules_engine_schedule_group_fact",),
                checklist="Rule engine: schedule — дни/время только из v6.4-факта, контакт-часы не использовать.",
            )
        return _schedule_manager_check_outcome(rule, subvariant="class_days_time_unpublished", question=question, facts=scoped_facts)

    return None


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


def _apply_selling_variants(
    outcome: RuleOutcome | None,
    *,
    rule: Rule,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
) -> RuleOutcome | None:
    if outcome is None or outcome.route != "bot_answer_self_for_pilot":
        return outcome
    selling = _selling_view(plan, context)
    if not selling:
        return outcome
    det_outcome = _apply_selling_det_variants(outcome, rule=rule, plan=plan, facts=facts, context=context, selling=selling)
    if det_outcome is outcome:
        return outcome
    if _selling_mode(context) == "det":
        return det_outcome
    compose_fn = _selling_compose_fn(context)
    if compose_fn is None:
        return _mark_selling_gen_fallback(det_outcome, reason="composer_unavailable")
    candidate, error = _compose_selling_text(
        compose_fn,
        rule=rule,
        plan=plan,
        facts={**dict(facts), **dict(det_outcome.facts)},
        base_outcome=outcome,
        det_outcome=det_outcome,
        selling=selling,
        context=context,
    )
    if not candidate:
        return _mark_selling_gen_fallback(det_outcome, reason=error or "empty_candidate")
    valid, reason = _verify_selling_generated_text(
        candidate,
        facts={**dict(facts), **dict(det_outcome.facts)},
        active_brand=_active_brand(plan, context),
    )
    if not valid:
        return _mark_selling_gen_fallback(det_outcome, reason=reason)
    metadata = {
        **dict(det_outcome.metadata),
        "selling": {
            **(dict(det_outcome.metadata.get("selling") or {}) if isinstance(det_outcome.metadata.get("selling"), Mapping) else {}),
            "mode": "gen",
            "gen_applied": True,
            "gen_fallback": False,
        },
    }
    flags = tuple(dict.fromkeys([*det_outcome.flags, "rules_engine_selling_gen_applied"]))
    return replace(det_outcome, text=candidate, flags=flags, metadata=metadata)


def _apply_selling_det_variants(
    outcome: RuleOutcome,
    *,
    rule: Rule,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    context: Mapping[str, Any] | None,
    selling: Mapping[str, Any],
) -> RuleOutcome:
    brand = _active_brand(plan, context)
    all_facts = {**dict(facts), **dict(outcome.facts)}
    text = outcome.text
    source = dict(outcome.facts)
    flags = list(outcome.flags)
    applied: list[str] = []

    if str(selling.get("objection") or "none") == "price" and rule.rule_id in {"installment", "discount", "price"}:
        objection_text = _selling_price_objection_text(
            rule_id=rule.rule_id,
            brand=brand,
            base_text=text,
            facts=all_facts,
        )
        if objection_text:
            text = objection_text
            flags.append("rules_engine_selling_price_objection")
            applied.append("price_objection")

    if bool(selling.get("exit_signal")):
        suffix, suffix_facts = _selling_exit_step(all_facts, active_brand=brand)
        appended = _append_selling_suffix(text, source, suffix, suffix_facts)
        if appended is not None:
            text, source = appended
            flags.append("rules_engine_selling_exit_signal")
            applied.append("exit_signal")

    if _selling_signals_full_enabled(context):
        if bool(selling.get("anxiety")):
            suffix, suffix_facts = _selling_anxiety_step(all_facts, active_brand=brand)
            appended = _append_selling_suffix(text, source, suffix, suffix_facts)
            if appended is not None:
                text, source = appended
                flags.append("rules_engine_selling_anxiety")
                applied.append("anxiety")
        unmet_need = str(selling.get("unmet_need") or "").strip()
        if unmet_need:
            suffix, suffix_facts = _selling_unmet_need_step(all_facts, active_brand=brand)
            appended = _append_selling_suffix(text, source, suffix, suffix_facts)
            if appended is not None:
                text, source = appended
                flags.append("rules_engine_selling_unmet_need")
                applied.append("unmet_need")
        if str(selling.get("readiness") or "exploring") == "ready":
            suffix, suffix_facts = _selling_readiness_step(all_facts, active_brand=brand)
            appended = _append_selling_suffix(text, source, suffix, suffix_facts)
            if appended is not None:
                text, source = appended
                flags.append("rules_engine_selling_readiness")
                applied.append("readiness")

    proactive_step = _resolve_a2_proactive_step(
        selling=selling,
        question=_raw_question_text(plan, context),
        facts=all_facts,
        active_brand=brand,
        context=context,
    )
    if proactive_step != "none" and not applied:
        suffix, suffix_facts = _a2_proactive_step_text(
            proactive_step,
            selling=selling,
            facts=all_facts,
            active_brand=brand,
            context=context,
        )
        suffix_facts = {
            key: value
            for key, value in suffix_facts.items()
            if not _selling_fact_already_used(value, text=text, source=source)
        }
        appended = _append_selling_suffix(text, source, suffix, suffix_facts, rich_format=_rich_format_enabled(context))
        if appended is not None:
            text, source = appended
            flags.append(f"rules_engine_a2_{proactive_step}")
            applied.append(proactive_step)

    if not applied:
        return outcome
    text = _avoid_exact_rule_repeat(text, context)
    metadata = {
        **dict(outcome.metadata),
        "selling": {
            "applied": applied,
            "objection": str(selling.get("objection") or "none"),
            "exit_signal": bool(selling.get("exit_signal")),
            "anxiety": bool(selling.get("anxiety")),
            "unmet_need": str(selling.get("unmet_need") or "")[:120],
            "readiness": str(selling.get("readiness") or "exploring"),
            "proactive": {
                "enabled": _proactive_enabled(context),
                "step": proactive_step,
                "policy_source": "deterministic",
                "recent_ignored": _proactive_recent_ignored(context),
                "phone_known": _context_phone_known(context),
            },
        },
    }
    return replace(outcome, text=text, facts=source, flags=tuple(dict.fromkeys(flags)), metadata=metadata)


def _selling_mode(context: Mapping[str, Any] | None) -> str:
    if isinstance(context, Mapping):
        value = str(context.get("selling_mode") or context.get(SELLING_MODE_ENV) or "").strip().casefold()
        if value in {"gen", "det"}:
            return value
    value = os.getenv(SELLING_MODE_ENV, "gen").strip().casefold()
    return value if value in {"gen", "det"} else "gen"


def _selling_signals_full_enabled(context: Mapping[str, Any] | None) -> bool:
    if isinstance(context, Mapping):
        for key in ("selling_signals_full", SELLING_SIGNALS_FULL_ENV):
            if key in context:
                return _truthy(context.get(key))
    return _truthy(os.getenv(SELLING_SIGNALS_FULL_ENV))


def _proactive_enabled(context: Mapping[str, Any] | None) -> bool:
    if isinstance(context, Mapping):
        for key in ("a_proactive_enabled", "proactive_enabled", A_PROACTIVE_ENV):
            if key in context:
                return _truthy(context.get(key))
    return _truthy(os.getenv(A_PROACTIVE_ENV))


def _rich_format_enabled(context: Mapping[str, Any] | None) -> bool:
    if isinstance(context, Mapping):
        for key in ("a_rich_format_enabled", "rich_format_enabled", A_RICH_FORMAT_ENV):
            if key in context:
                return _truthy(context.get(key))
    return _truthy(os.getenv(A_RICH_FORMAT_ENV))


def _coverage_enabled(context: Mapping[str, Any] | None) -> bool:
    if isinstance(context, Mapping):
        for key in ("coverage_enabled", COVERAGE_ENV):
            if key in context:
                return _truthy(context.get(key))
    return _truthy(os.getenv(COVERAGE_ENV))


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on", "y"}


_A2_PROCEED_RE = re.compile(
    r"как\s+(?:записать|записаться|оплат|внести)|хочу\s+запис|готов[аы]?\s+(?:запис|оформ)|"
    r"запишите|свяжите|позвоните|оставлю\s+(?:телефон|номер)|можно\s+оплатить|где\s+оплат",
    re.I,
)


def _resolve_a2_proactive_step(
    *,
    selling: Mapping[str, Any],
    question: str,
    facts: Mapping[str, str],
    active_brand: str,
    context: Mapping[str, Any] | None,
) -> str:
    if not _proactive_enabled(context):
        return "none"
    if _a2_p0_signal(selling=selling, question=question):
        return "none"
    explicit_proceed = bool(_A2_PROCEED_RE.search(str(question or "")))
    if _proactive_recent_ignored(context) >= 2 and not explicit_proceed:
        return "none"
    if _current_message_has_phone(question):
        return "none"
    exit_signal = selling.get("exit_signal")
    if _truthy(exit_signal) or str(exit_signal or "").strip().casefold() in {"strong", "browsing"}:
        return "soft_close"
    objection = _a2_objection_type(selling)
    if objection == "refund":
        return "none"
    if objection in {"price", "time", "fit", "trust"}:
        return "handle_objection" if _a2_objection_fact_available(objection, facts=facts, active_brand=active_brand) else "none"
    if explicit_proceed:
        return "offer_enroll" if _a2_contact_shared(selling=selling, question=question, context=context) and _a2_enrollment_fact_available(facts, active_brand=active_brand) else "offer_callback"
    readiness = str(selling.get("readiness") or "exploring").strip().casefold()
    if readiness == "ready":
        return "offer_enroll" if _a2_contact_shared(selling=selling, question=question, context=context) and _a2_enrollment_fact_available(facts, active_brand=active_brand) else "offer_callback"
    missing = _a2_missing_qualifier(selling)
    if missing != "none" and missing not in _a2_already_asked(context) and _a2_qualifier_unlocks_fact(missing, facts=facts, active_brand=active_brand):
        return "qualify"
    if readiness in {"warm", "comparing"}:
        if not _truthy(selling.get("fit_confirmed")) and _a2_fit_fact_available(facts, active_brand=active_brand):
            return "fit_check"
        if not _truthy(selling.get("stalled_asked")):
            return "elicit_objection"
    return "none"


def _a2_offer_callback_text(context: Mapping[str, Any] | None) -> str:
    if _context_phone_known(context):
        return "Если удобно, передам менеджеру — подскажите, когда лучше связаться?"
    return "Если удобно, передам менеджеру — подскажите телефон и когда лучше связаться?"


def _a2_proactive_step_text(
    step: str,
    *,
    selling: Mapping[str, Any],
    facts: Mapping[str, str],
    active_brand: str,
    context: Mapping[str, Any] | None,
) -> tuple[str, Mapping[str, str]]:
    if step == "offer_callback":
        return _a2_offer_callback_text(context), {}
    if step == "soft_close":
        return "Хорошо, спокойно подумайте. Если понадобится, я рядом и подскажу по подтверждённым условиям.", {}
    if step == "qualify":
        return _a2_qualify_text(_a2_missing_qualifier(selling)), {}
    if step == "fit_check":
        return _a2_fit_check_text(facts, active_brand=active_brand)
    if step == "elicit_objection":
        return "Что для вас сейчас важнее всего при выборе — цена, расписание или формат?", {}
    if step == "handle_objection":
        return _a2_handle_objection_text(_a2_objection_type(selling), facts=facts, active_brand=active_brand)
    if step == "offer_enroll":
        return _a2_offer_enroll_text(facts, active_brand=active_brand)
    return "", {}


def _a2_p0_signal(*, selling: Mapping[str, Any], question: str) -> bool:
    objection = _a2_objection_type(selling)
    if objection == "refund":
        return True
    text = str(question or "").casefold().replace("ё", "е")
    return bool(re.search(r"верните\s+деньг|возврат|жалоб|суд|юрист|двойн(?:ое|ая)\s+списан|оплатил[аи]?,?\s+занят", text))


def _a2_objection_type(selling: Mapping[str, Any]) -> str:
    value = str(selling.get("objection_type") or selling.get("objection") or "none").strip().casefold()
    return value if value in {"price", "time", "fit", "trust", "refund"} else "none"


def _a2_contact_shared(*, selling: Mapping[str, Any], question: str, context: Mapping[str, Any] | None) -> bool:
    return _truthy(selling.get("contact_shared")) or _current_message_has_phone(question) or _context_phone_known(context)


def _a2_missing_qualifier(selling: Mapping[str, Any]) -> str:
    value = str(selling.get("missing_qualifier") or "none").strip().casefold()
    return value if value in {"class", "subject", "format", "goal"} else "none"


def _a2_already_asked(context: Mapping[str, Any] | None) -> set[str]:
    result: set[str] = set()
    if not isinstance(context, Mapping):
        return result
    for container in _proactive_state_containers(context):
        raw = container.get("already_asked") or container.get("qualifiers_asked")
        if isinstance(raw, Mapping):
            result.update(str(key).strip().casefold() for key, value in raw.items() if _truthy(value))
        elif isinstance(raw, (list, tuple, set)):
            result.update(str(item).strip().casefold() for item in raw if str(item).strip())
    return result


def _a2_qualifier_unlocks_fact(missing: str, *, facts: Mapping[str, str], active_brand: str) -> bool:
    markers = {
        "class": ("класс", "grade", "5-11", "9", "10", "11"),
        "subject": ("предмет", "математ", "физик", "информ", "subject"),
        "format": ("онлайн", "очно", "формат", "мтс линк"),
        "goal": ("егэ", "огэ", "олимпиад", "цель", "уров"),
    }.get(missing, ())
    _, fact = _clean_selling_support_fact(facts, active_brand, markers)
    return bool(fact)


def _a2_objection_fact_available(objection: str, *, facts: Mapping[str, str], active_brand: str) -> bool:
    return bool(_a2_objection_support_fact(objection, facts=facts, active_brand=active_brand)[1])


def _a2_objection_support_fact(objection: str, *, facts: Mapping[str, str], active_brand: str) -> tuple[str, str]:
    markers = {
        "price": ("рассроч", "помесяч", "скид", "долями", "оплат", "семестр", "год"),
        "time": ("распис", "выходн", "будн", "слот", "время", "день"),
        "fit": ("класс", "формат", "предмет", "уров", "программ", "олимпиад"),
        "trust": ("лиценз", "преподав", "мфти", "мгу", "результат", "запис"),
    }.get(objection, ())
    return _clean_selling_support_fact(facts, active_brand, markers)


def _a2_handle_objection_text(
    objection: str,
    *,
    facts: Mapping[str, str],
    active_brand: str,
) -> tuple[str, Mapping[str, str]]:
    key, fact = _a2_objection_support_fact(objection, facts=facts, active_brand=active_brand)
    if not fact:
        return "", {}
    if objection == "price":
        return f"По бюджету можно смотреть подтверждённые варианты оплаты: {_short_sentence(fact)}", {key: fact}
    if objection == "time":
        return f"По времени ориентир такой: {_short_sentence(fact)}", {key: fact}
    if objection == "fit":
        return f"По фактам рамка такая: {_short_sentence(fact)} Уровень и группу менеджер сверит отдельно.", {key: fact}
    if objection == "trust":
        return f"Для спокойствия можно опереться на подтверждённый факт: {_short_sentence(fact)}", {key: fact}
    return "", {}


def _a2_fit_fact_available(facts: Mapping[str, str], *, active_brand: str) -> bool:
    return bool(_clean_selling_support_fact(facts, active_brand, ("класс", "формат", "предмет", "программ", "уров"))[1])


def _a2_fit_check_text(facts: Mapping[str, str], *, active_brand: str) -> tuple[str, Mapping[str, str]]:
    key, fact = _clean_selling_support_fact(facts, active_brand, ("класс", "формат", "предмет", "программ", "уров"))
    if not fact:
        return "", {}
    return f"По фактам это рамка курса: {_short_sentence(fact)} Уровень и группу менеджер сверит отдельно.", {key: fact}


def _a2_qualify_text(missing: str) -> str:
    if missing == "class":
        return "Подскажите класс ребёнка — тогда сориентирую точнее по подходящему варианту."
    if missing == "subject":
        return "Подскажите предмет — тогда сориентирую точнее по программе."
    if missing == "format":
        return "Подскажите, удобнее онлайн или очно — тогда сориентирую точнее."
    if missing == "goal":
        return "Подскажите цель подготовки — школьная программа, ЕГЭ/ОГЭ или олимпиада?"
    return ""


def _a2_offer_enroll_text(facts: Mapping[str, str], *, active_brand: str) -> tuple[str, Mapping[str, str]]:
    key, fact = _brand_scoped_first_matching_fact(
        facts,
        active_brand,
        ("process.enrollment", "enrollment", "запис", "заявк", "оформ", "менеджер"),
    )
    if not fact:
        return "", {}
    return f"Если хотите продолжить, передам менеджеру — он поможет с записью: {_short_sentence(fact)}", {key: fact}


def _a2_enrollment_fact_available(facts: Mapping[str, str], *, active_brand: str) -> bool:
    return bool(
        _brand_scoped_first_matching_fact(
            facts,
            active_brand,
            ("process.enrollment", "enrollment", "запис", "заявк", "оформ", "менеджер"),
        )[1]
    )


def _current_message_has_phone(text: str) -> bool:
    return bool(re.search(r"(?:\+7|8|7)?[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}", str(text or "")))


def _context_phone_known(context: Mapping[str, Any] | None) -> bool:
    if not isinstance(context, Mapping):
        return False
    containers: list[Mapping[str, Any]] = []
    for key in ("known_slots", "known_dialog_fields", "known_client_fields", "client_identity"):
        value = context.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    memory = context.get("dialogue_memory_view")
    if isinstance(memory, Mapping):
        for key in ("known_slots", "client_confirmed_slots", "crm_known_slots"):
            value = memory.get(key)
            if isinstance(value, Mapping):
                containers.append(value)
    for container in containers:
        for key in ("phone_known", "phone", "normalized_phone", "client_phone"):
            raw = container.get(key)
            if isinstance(raw, Mapping):
                raw = raw.get("value")
            if str(raw or "").strip().casefold() not in {"", "false", "none", "0"}:
                return True
    return False


def _proactive_recent_ignored(context: Mapping[str, Any] | None) -> int:
    if not isinstance(context, Mapping):
        return 0
    for container in _proactive_state_containers(context):
        raw = container.get("recent_ignored")
        if raw is not None:
            try:
                return max(0, int(raw))
            except (TypeError, ValueError):
                pass
    history: Sequence[Any] = ()
    for container in _proactive_state_containers(context):
        raw_history = container.get("history") or container.get("proactive_history")
        if isinstance(raw_history, Sequence) and not isinstance(raw_history, (str, bytes)):
            history = raw_history
            break
    ignored = 0
    for item in reversed(tuple(history)):
        if not isinstance(item, Mapping):
            break
        if str(item.get("outcome") or item.get("status") or "").strip().casefold() == "ignored" or bool(item.get("ignored")):
            ignored += 1
            continue
        break
    return ignored


def _proactive_state_containers(context: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    for key in ("proactive_state", "a2_proactive_state"):
        value = context.get(key)
        if isinstance(value, Mapping):
            result.append(value)
    memory = context.get("dialogue_memory_view")
    if isinstance(memory, Mapping):
        for key in ("proactive_state", "a2_proactive_state"):
            value = memory.get(key)
            if isinstance(value, Mapping):
                result.append(value)
    return tuple(result)


def _selling_compose_fn(context: Mapping[str, Any] | None) -> Callable[[str], object] | None:
    if not isinstance(context, Mapping):
        return None
    value = context.get("selling_compose_fn") or context.get("a_selling_compose_fn")
    return value if callable(value) else None


def _compose_selling_text(
    compose_fn: Callable[[str], object],
    *,
    rule: Rule,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    base_outcome: RuleOutcome,
    det_outcome: RuleOutcome,
    selling: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> tuple[str, str]:
    prompt = _build_selling_compose_prompt(
        rule=rule,
        plan=plan,
        facts=facts,
        base_outcome=base_outcome,
        det_outcome=det_outcome,
        selling=selling,
        context=context,
    )
    try:
        raw = compose_fn(prompt)
    except Exception as exc:
        return "", f"composer_error:{exc.__class__.__name__}"
    if isinstance(raw, Mapping):
        text = str(raw.get("text") or raw.get("draft_text") or raw.get("answer") or "").strip()
    else:
        text = str(raw or "").strip()
    return (" ".join(text.split())[:900], "") if text else ("", "empty_candidate")


def _build_selling_compose_prompt(
    *,
    rule: Rule,
    plan: Mapping[str, Any],
    facts: Mapping[str, str],
    base_outcome: RuleOutcome,
    det_outcome: RuleOutcome,
    selling: Mapping[str, Any],
    context: Mapping[str, Any] | None,
) -> str:
    brand = _active_brand(plan, context)
    facts_text = "\n".join(f"- {key}: {value}" for key, value in facts.items())
    history = ""
    if isinstance(context, Mapping):
        recent = context.get("recent_messages")
        if isinstance(recent, (list, tuple)):
            history = "\n".join(str(item) for item in recent[-6:])
    return (
        "Ты формулируешь короткий клиентский ответ для учебного центра.\n"
        f"Активный бренд: {brand}. Нельзя упоминать другой бренд.\n"
        f"Вопрос клиента: {_raw_question_text(plan, context)}\n"
        f"Правило: {rule.rule_id}/{det_outcome.subvariant}. Selling: {dict(selling)}.\n"
        "Задача: признать сомнение клиента, дать ценность/способ оплаты из фактов и мягкий следующий шаг. "
        "Эмпатичное открытие варьируй: не начинай каждый ответ с «Понимаю»; можно сказать "
        "«Да, это важный момент», «Давайте разложу по вариантам» или сразу перейти к факту, если так естественнее. "
        "Без давления, без обещаний результата/поступления, 1-3 предложения.\n"
        "Числа, проценты, суммы, даты и сроки можно брать ТОЛЬКО дословно из фактов ниже. Не выдумывай.\n"
        "Если факта не хватает, не добавляй этот фрагмент.\n"
        f"Базовый факт-ответ: {base_outcome.text}\n"
        f"Детерминированный fallback: {det_outcome.text}\n"
        f"История:\n{history}\n"
        f"Факты:\n{facts_text}\n"
        "Верни строго JSON: {\"text\":\"...\"}"
    )


_SELLING_PRESSURE_RE = re.compile(
    r"только\s+сегодня|успейт|последн(?:ий|яя)\s+шанс|решайт[е]?\s+сейчас|"
    r"срочно\s+(?:оформ|запис|реш)|иначе\s+(?:мест|скид|цен)",
    re.I,
)
_SELLING_GUARANTEE_RE = re.compile(
    r"гарантир|100\s*%|обязательно\s+(?:поступ|сдад|получ)|точно\s+(?:поступ|сдад|получ)",
    re.I,
)
_SELLING_DIAGNOSIS_RE = re.compile(
    r"диагноз|продиагностир|исправим\s+на\s+(?:5|пят)|подтянем\s+на\s+(?:5|пят)|"
    r"сделаем\s+(?:отличник|хорошист)|реб[её]нок\s+точно\s+станет",
    re.I,
)
_SELLING_NUMBER_RE = re.compile(r"\d[\d\s\u00a0]*(?:[.,]\d+)?\s*(?:₽|%|руб(?:\.|лей|ля|ль)?|месяц(?:ев|а)?|дн(?:ей|я)?|раз(?:а)?|балл(?:ов|а)?)?")


def _verify_selling_generated_text(
    text: str,
    *,
    facts: Mapping[str, str],
    active_brand: str,
) -> tuple[bool, str]:
    candidate = str(text or "").strip()
    if not candidate:
        return False, "empty_candidate"
    if _mentions_other_brand(candidate, active_brand):
        return False, "other_brand"
    if _SELLING_PRESSURE_RE.search(candidate):
        return False, "pressure"
    if _SELLING_GUARANTEE_RE.search(candidate):
        return False, "guarantee"
    if _SELLING_DIAGNOSIS_RE.search(candidate):
        return False, "diagnosis_or_grade_promise"
    fact_text = " ".join(str(value or "") for value in facts.values())
    fact_norm = _normalize_number_surface(fact_text)
    for token in _SELLING_NUMBER_RE.findall(candidate):
        normalized = _normalize_number_surface(token)
        if normalized and normalized not in fact_norm:
            return False, f"unsupported_number:{token.strip()}"
    return True, ""


def _normalize_number_surface(text: str) -> str:
    return str(text or "").replace("\u00a0", " ").casefold().replace("ё", "е")


def _mark_selling_gen_fallback(outcome: RuleOutcome, *, reason: str) -> RuleOutcome:
    metadata = {
        **dict(outcome.metadata),
        "selling": {
            **(dict(outcome.metadata.get("selling") or {}) if isinstance(outcome.metadata.get("selling"), Mapping) else {}),
            "mode": "gen",
            "gen_applied": False,
            "gen_fallback": True,
            "gen_fallback_reason": reason,
        },
    }
    flags = tuple(dict.fromkeys([*outcome.flags, "rules_engine_selling_gen_fallback"]))
    return replace(outcome, flags=flags, metadata=metadata)


def _selling_view(plan: Mapping[str, Any], context: Mapping[str, Any] | None) -> Mapping[str, Any]:
    raw = plan.get("selling")
    if isinstance(raw, Mapping):
        return raw
    if isinstance(context, Mapping):
        intent_plan = context.get("conversation_intent_plan")
        if isinstance(intent_plan, Mapping) and isinstance(intent_plan.get("selling"), Mapping):
            return intent_plan["selling"]  # type: ignore[return-value]
    return {}


def _selling_price_objection_text(
    *,
    rule_id: str,
    brand: str,
    base_text: str,
    facts: Mapping[str, str],
) -> str:
    if rule_id == "installment" and brand == "foton":
        pointer = _foton_payment_pointer(facts)
        if not pointer:
            return ""
        pointer_sentence = pointer[:1].upper() + pointer[1:]
        return f"Понимаю, важно, чтобы вложение было посильным. {pointer_sentence}, чтобы не вносить всю сумму сразу. Подобрать удобный вариант?"
    if rule_id in {"installment", "discount"} and brand == "unpk":
        return _unpk_period_payment_objection_text(facts)
    if rule_id == "price":
        pointer = _foton_payment_pointer(facts) if brand == "foton" else _unpk_payment_pointer(facts) if brand == "unpk" else ""
        if pointer:
            return f"{base_text.rstrip()} Если по бюджету важно — {pointer}. Подсказать удобный вариант?"
        return f"{base_text.rstrip()} Если по бюджету важно, менеджер подскажет подходящий способ оплаты под вашу ситуацию."
    return ""


def _foton_payment_pointer(facts: Mapping[str, str]) -> str:
    _, fact = _brand_scoped_first_matching_fact(
        facts,
        "foton",
        ("installment", "рассроч", "оплатить обучение частями", "частями", "месяц", "долями", "dolyami"),
    )
    if not fact:
        return ""
    months = _month_options_from_fact(fact)
    has_dolyami = _has_any(fact, ("долями", "dolyami"))
    parts: list[str] = []
    if months:
        parts.append(f"оплату можно разбить на {months}")
    if has_dolyami:
        parts.append("доступен сервис Долями")
    return " или ".join(parts)


def _unpk_payment_pointer(facts: Mapping[str, str]) -> str:
    text = _unpk_period_payment_objection_text(facts)
    if not text:
        return ""
    return text.removeprefix("Понимаю. ").split(" Подсказать,", 1)[0].rstrip(".")


def _unpk_period_payment_objection_text(facts: Mapping[str, str]) -> str:
    _, period_fact = _first_matching_fact(
        facts,
        ("payment_options.client_safe_text.when_asked_about_installment", "помесячно", "за семестр", "за год", "растянуть оплату"),
    )
    _, semester_fact = _first_matching_fact_with_required(
        facts,
        ("semester", "семестр", "discounts.monthly_payment.pct", "discounts.semester_payment"),
        required=("10%",),
    )
    _, year_fact = _first_matching_fact_with_required(
        facts,
        ("year", "год", "discounts.year", "year_discount", "year_payment"),
        required=("14%",),
    )
    if not (period_fact and semester_fact and year_fact):
        return ""
    semester_pct = _first_percent(semester_fact)
    year_pct = _first_percent(year_fact)
    if not (semester_pct and year_pct):
        return ""
    return (
        "Понимаю. В УНПК можно платить помесячно, за семестр или за год; "
        f"при оплате за семестр действует скидка {semester_pct}, за год — {year_pct}. "
        "Подсказать, что выгоднее в вашем случае?"
    )


def _month_options_from_fact(fact: str) -> str:
    text = str(fact or "")
    match = re.search(r"((?:\d{1,2}\D{0,16}){1,6})месяц", text, re.I)
    if not match:
        return ""
    numbers = re.findall(r"\d{1,2}", match.group(1))
    if not numbers:
        return ""
    unique = list(dict.fromkeys(numbers))
    if len(unique) >= 2:
        return ", ".join(unique[:-1]) + " или " + unique[-1] + " месяцев"
    return unique[0] + " месяцев"


def _first_percent(text: str) -> str:
    match = re.search(r"\b(\d{1,2})\s*%", str(text or ""))
    return f"{match.group(1)}%" if match else ""


_BAD_SELLING_SUPPORT_FACT_RE = re.compile(
    r"адрес|место\s+занятий|площадк|сретен|красносельск|скорняж|пацаев|институтск|"
    r"контакт|телефон|на\s+связи|ежедневно\s+с\s+\d",
    re.I,
)


def _append_selling_suffix(
    text: str,
    source: Mapping[str, str],
    suffix: str,
    suffix_facts: Mapping[str, str],
    *,
    rich_format: bool = False,
) -> tuple[str, dict[str, str]] | None:
    suffix_text = " ".join(str(suffix or "").split()).strip()
    if not suffix_text:
        return None
    current_norm = _norm_for_repeat(text)
    if _norm_for_repeat(suffix_text) in current_norm:
        return None
    for fact in suffix_facts.values():
        if _selling_fact_already_used(fact, text=text, source=source):
            return None
    updated = dict(source)
    updated.update(suffix_facts)
    separator = "\n\n" if rich_format else " "
    return f"{text.rstrip()}{separator}{suffix_text}", updated


def _selling_fact_already_used(fact: str, *, text: str, source: Mapping[str, str]) -> bool:
    fact_norm = _norm_for_repeat(_short_sentence(fact))
    if not fact_norm:
        return False
    if fact_norm in _norm_for_repeat(text):
        return True
    return any(fact_norm == _norm_for_repeat(_short_sentence(value)) for value in source.values())


def _avoid_exact_rule_repeat(text: str, context: Mapping[str, Any] | None) -> str:
    current = _norm_for_repeat(text)
    if not current:
        return text
    if any(current == _norm_for_repeat(previous) for previous in _previous_rule_bot_texts(context)):
        return f"Если коротко: {text.strip()}"
    return text


def _previous_rule_bot_texts(context: Mapping[str, Any] | None) -> tuple[str, ...]:
    if not isinstance(context, Mapping):
        return ()
    result: list[str] = []
    for key in ("previous_bot_texts", "recent_bot_texts"):
        value = context.get(key)
        if isinstance(value, (list, tuple)):
            result.extend(str(item or "") for item in value if str(item or "").strip())
    recent = context.get("recent_messages")
    if isinstance(recent, (list, tuple)):
        for item in recent:
            if isinstance(item, Mapping):
                role = str(item.get("role") or item.get("sender") or item.get("direction") or "").casefold()
                if role in {"bot", "assistant", "outbound"}:
                    text = str(item.get("text") or item.get("message") or "").strip()
                    if text:
                        result.append(text)
    return tuple(result)


def _norm_for_repeat(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold().replace("ё", "е")).strip()


def _clean_selling_support_fact(
    facts: Mapping[str, str],
    active_brand: str,
    markers: Sequence[str],
) -> tuple[str, str]:
    other_brand_markers = ("унпк", "kmipt") if active_brand == "foton" else ("фотон", "cdpofoton", "цдпо") if active_brand == "unpk" else ()
    for key, value in facts.items():
        text = " ".join(str(value or "").split()).strip()
        combined = f"{key} {text}".casefold().replace("ё", "е")
        if not text:
            continue
        if other_brand_markers and any(marker in combined for marker in other_brand_markers):
            continue
        if not any(str(marker).casefold().replace("ё", "е") in combined for marker in markers):
            continue
        if _BAD_SELLING_SUPPORT_FACT_RE.search(text) or _BAD_SELLING_SUPPORT_FACT_RE.search(str(key)):
            continue
        return str(key), text
    return "", ""


def _selling_exit_step(facts: Mapping[str, str], *, active_brand: str) -> tuple[str, Mapping[str, str]]:
    key, fact = _clean_selling_support_fact(
        facts,
        active_brand,
        ("trial", "пробн", "фрагмент занятия", "фрагмент урок", "online_fragment", "trial_class"),
    )
    if fact:
        return (
            f"Если поможет решить, можно начать с подтверждённого пробного шага: {_short_sentence(fact)} Подсказать, как записаться?",
            {key or "rules_engine.selling.exit_trial": fact},
        )
    return (
        "Спокойно подумайте; если нужно, подскажу, что важно для решения по этому варианту.",
        {},
    )


def _selling_anxiety_step(facts: Mapping[str, str], *, active_brand: str) -> tuple[str, Mapping[str, str]]:
    key, fact = _clean_selling_support_fact(
        facts,
        active_brand,
        ("license", "лиценз", "документ", "юрлиц", "официаль"),
    )
    if fact:
        return (
            "Чтобы было спокойнее: по подтверждённым данным есть лицензия на образовательную деятельность; реквизиты можно запросить у менеджера.",
            {key or "rules_engine.selling.anxiety_license": fact},
        )
    key, fact = _clean_selling_support_fact(
        facts,
        active_brand,
        ("trial", "пробн", "фрагмент занятия", "фрагмент урок", "online_fragment", "trial_class"),
    )
    if fact:
        return (
            f"Чтобы было спокойнее, можно начать с подтверждённого пробного шага: {_short_sentence(fact)}",
            {key or "rules_engine.selling.anxiety_trial": fact},
        )
    return "", {}


def _selling_unmet_need_step(facts: Mapping[str, str], *, active_brand: str) -> tuple[str, Mapping[str, str]]:
    key, fact = _clean_selling_support_fact(
        facts,
        active_brand,
        ("teacher", "teachers", "преподав", "педагог", "мфти", "мгу", "вшэ", "мифи"),
    )
    if not fact:
        key, fact = _clean_selling_support_fact(
            facts,
            active_brand,
            ("process.enrollment", "запис", "менеджер уточнит класс", "подбер", "заявк"),
        )
    if not fact:
        key, fact = _clean_selling_support_fact(
            facts,
            active_brand,
            ("trial", "пробн", "фрагмент занятия", "online_fragment", "trial_class"),
        )
    if not fact:
        return "", {}
    return (
        f"Если по сути — вот что подтверждено: {_short_sentence(fact)}",
        {key or "rules_engine.selling.unmet_need_fact": fact},
    )


def _selling_readiness_step(facts: Mapping[str, str], *, active_brand: str) -> tuple[str, Mapping[str, str]]:
    key, fact = _brand_scoped_first_matching_fact(
        facts,
        active_brand,
        ("process.enrollment", "enrollment", "запис", "заявк", "менеджер уточнит класс", "оформ"),
    )
    if fact:
        return (
            f"Если готовы к следующему шагу: {_short_sentence(fact)}",
            {key or "rules_engine.selling.readiness_enrollment": fact},
        )
    return (
        "Если готовы к следующему шагу, менеджер подтвердит порядок записи по выбранному курсу.",
        {},
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


@dataclass(frozen=True)
class _CampIncludedComposition:
    text: str
    facts: Mapping[str, str]


def _asks_camp_transfer(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    return _has_any(value, ("трансфер", "из москвы", "ховрино", "добраться", "добир", "заезд", "отъезд"))


def _asks_camp_included(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    return _has_any(
        value,
        (
            "что входит",
            "что включено",
            "включено",
            "входит",
            "проживание",
            "прожив",
            "питание",
            "питан",
            "отдельно",
        ),
    )


def _camp_price_fact(facts: Mapping[str, str], brand: str) -> tuple[str, str]:
    return _camp_best_fact(
        facts,
        brand,
        required_markers=("price", "pricing", "стоим", "стоимость", "цен", "₽", "руб"),
        preferred_markers=("client_safe_text", "when_price_asked", "current_price", "текущая", "полная стоимость"),
        blocked_markers=("source conflicts", "internal", "конфликт", "спорн"),
        require_money=True,
    )


def _camp_transfer_fact(facts: Mapping[str, str], brand: str) -> tuple[str, str]:
    return _camp_best_fact(
        facts,
        brand,
        required_markers=("transfer", "трансфер", "ховрино", "заезд", "отъезд", "добир"),
        preferred_markers=("client_safe_text", "included", "включен", "сбор", "ховрино"),
        blocked_markers=("цен", "₽", "руб", "pricing"),
    )


def _camp_included_composition(facts: Mapping[str, str], brand: str) -> _CampIncludedComposition | None:
    accommodation_key, accommodation = _camp_best_fact(
        facts,
        brand,
        required_markers=("room_capacity", "прожив", "живут", "номер", "номерах", "room", "2-3 человека"),
        preferred_markers=("client_safe_text", "room_capacity", "accommodation", "living"),
        blocked_markers=("program", "72+", "стоим", "цен", "₽", "руб"),
    )
    meals_key, meals = _camp_best_fact(
        facts,
        brand,
        required_markers=("meals", "meal", "питание", "питан", "шведский", "приемов пищи", "разовое"),
        preferred_markers=("client_safe_text", "meals_description", "meals_per_day"),
        blocked_markers=("program", "72+", "стоим", "цен", "₽", "руб"),
    )
    transfer_key, transfer = _camp_transfer_fact(facts, brand)
    parts: list[str] = []
    source: dict[str, str] = {}
    if accommodation:
        parts.append("проживание: " + _camp_clean_fact_text(accommodation, max_chars=180))
        source[accommodation_key or "rules_engine.camp.accommodation"] = accommodation
    if meals and meals != accommodation:
        parts.append("питание: " + _camp_clean_fact_text(meals, max_chars=180))
        source[meals_key or "rules_engine.camp.meals"] = meals
    if transfer:
        parts.append("трансфер: " + _camp_clean_fact_text(transfer, max_chars=220))
        source[transfer_key or "rules_engine.camp.transfer"] = transfer
    if len(parts) < 2:
        return None
    return _CampIncludedComposition(
        text="В ЛВШ Менделеево могу подтвердить по фактам: " + "; ".join(parts) + ".",
        facts=source,
    )


def _camp_best_fact(
    facts: Mapping[str, str],
    brand: str,
    *,
    required_markers: Sequence[str],
    preferred_markers: Sequence[str] = (),
    blocked_markers: Sequence[str] = (),
    require_money: bool = False,
) -> tuple[str, str]:
    other_brand_markers = ("унпк", "kmipt") if brand == "foton" else ("фотон", "cdpofoton", "цдпо") if brand == "unpk" else ()
    candidates: list[tuple[int, str, str]] = []
    for key, value in facts.items():
        text = " ".join(str(value or "").split())
        if not str(key).strip() or not text:
            continue
        combined = f"{key} {text}".casefold().replace("ё", "е")
        if other_brand_markers and any(marker in combined for marker in other_brand_markers):
            continue
        if blocked_markers and any(str(marker).casefold().replace("ё", "е") in combined for marker in blocked_markers):
            continue
        if not any(str(marker).casefold().replace("ё", "е") in combined for marker in required_markers):
            continue
        if require_money and not _has_any(combined, ("₽", "руб")):
            continue
        score = 0
        if "client_safe_text" in combined:
            score += 10
        if "client safe" in combined:
            score += 8
        for index, marker in enumerate(preferred_markers):
            if str(marker).casefold().replace("ё", "е") in combined:
                score += max(1, 6 - index)
        if _has_any(combined, ("да.", "— да", "— бесплатно", "2026-05-20")):
            score -= 4
        candidates.append((score, str(key), text))
    if not candidates:
        return "", ""
    _, key, text = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    return key, text


def _camp_prefixed_answer(prefix: str, fact: str) -> str:
    return f"{prefix}: {_camp_clean_fact_text(fact, max_chars=360)}"


def _camp_clean_fact_text(fact: str, *, max_chars: int = 360) -> str:
    text = " ".join(str(fact or "").split())
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    text = re.sub(r"^(?:Фотон|УНПК):\s*", "", text).strip()
    text = re.sub(r"^(ЛВШ\s+Менделеево)\s+[—-]\s+\1\s+", r"\1 ", text, flags=re.I)
    return text


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
        return f"Места не буду обещать без проверки{suffix}. Передам менеджеру, чтобы он проверил наличие по конкретной смене."
    return f"Места не буду обещать без проверки{suffix}. Передам менеджеру, чтобы он проверил наличие."


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


def _looks_like_schedule_question(text: str, plan: Mapping[str, Any]) -> bool:
    intent = str(plan.get("primary_intent") or "")
    if intent in {"schedule", "schedule_inquiry", "class_schedule"}:
        return True
    scope = " ".join(str(plan.get(key) or "") for key in ("fact_scope", "schedule_scope", "topic_id"))
    value = f"{text} {scope}".casefold().replace("ё", "е")
    if _has_any(value, ("контакт", "на связи", "позвон", "телефон", "офис работает")) and not _has_any(value, ("занят", "групп", "распис")):
        return False
    return _has_any(
        value,
        (
            "распис",
            "по каким дням",
            "в какие дни",
            "когда занятия",
            "во сколько",
            "дни занятий",
            "время занятий",
            "раз в неделю",
            "старт",
            "начина",
            "выходн",
            "суббот",
            "воскрес",
            "schedule.current",
        ),
    )


def _asks_schedule_frequency(text: str, plan: Mapping[str, Any]) -> bool:
    value = f"{text} {plan.get('fact_scope') or ''}".casefold().replace("ё", "е")
    return _has_any(value, ("сколько раз в неделю", "раз в неделю", "часто занятия", "еженед"))


def _asks_schedule_start(text: str, plan: Mapping[str, Any]) -> bool:
    value = f"{text} {plan.get('fact_scope') or ''}".casefold().replace("ё", "е")
    return _has_any(value, ("когда старт", "старт", "начина", "с какого числа", "когда нач"))


def _asks_weekend_schedule(text: str, plan: Mapping[str, Any]) -> bool:
    value = f"{text} {plan.get('fact_scope') or ''}".casefold().replace("ё", "е")
    return _has_any(value, ("выходн", "суббот", "воскрес"))


def _asks_exact_class_schedule(text: str, plan: Mapping[str, Any]) -> bool:
    value = f"{text} {plan.get('fact_scope') or ''}".casefold().replace("ё", "е")
    return _has_any(
        value,
        (
            "по каким дням",
            "в какие дни",
            "когда занятия",
            "когда математ",
            "когда физик",
            "когда информат",
            "во сколько",
            "дни занятий",
            "время занятий",
            "распис",
        ),
    )


def _brand_scoped_schedule_facts(facts: Mapping[str, str], active_brand: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if not _is_schedule_fact(combined):
            continue
        if active_brand == "foton" and _has_any(combined, ("унпк", "kmipt", "сретен", "пацаева", "мфти")):
            continue
        if active_brand == "unpk" and _has_any(combined, ("фотон", "cdpofoton", "цдпо", "красносельская")):
            continue
        text = " ".join(str(value or "").split())
        if text:
            result[str(key)] = text
    return result


def _is_schedule_fact(combined: str) -> bool:
    if _is_contact_hours_fact(combined):
        return True
    return _has_any(
        combined,
        (
            "schedule_2026_27",
            "schedule.groups",
            "weekly_lessons",
            "учебный год 2026/27",
            "старт занятий",
            "расписание",
            "опубликованные группы",
            "конкретный вариант зависит",
            "разные слоты",
            "выходн",
            "суббот",
            "воскрес",
        ),
    )


def _is_contact_hours_fact(combined: str) -> bool:
    return _has_any(combined, ("contact_hours", "часы связи", "связаться")) and bool(
        re.search(r"10[:.]?00|18[:.]?00|пн\s*[–-]\s*вс|ежедневн", combined, re.I)
    )


def _first_schedule_fact(facts: Mapping[str, str], markers: Sequence[str]) -> tuple[str, str]:
    for key, value in facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if _is_contact_hours_fact(combined):
            continue
        if any(str(marker).casefold().replace("ё", "е") in combined for marker in markers):
            text = " ".join(str(value or "").split())
            if text:
                return str(key), text
    return "", ""


def _schedule_start_fact(facts: Mapping[str, str], question: str) -> tuple[str, str]:
    location_markers: tuple[str, ...] = ()
    if _has_any(question, ("москва", "красносельск", "сретен")):
        location_markers = ("moscow", "москва", "красносельск", "сретен")
    elif _has_any(question, ("онлайн", "дистанц")):
        location_markers = ("online", "онлайн")
    elif _has_any(question, ("пацаев", "долгопруд")):
        location_markers = ("patsayeva", "пацаев", "долгопруд")
    elif _has_any(question, ("институтск", "мфти")):
        location_markers = ("mfti_institutsky", "институтск", "мфти")
    if location_markers:
        key, fact = _first_schedule_fact(facts, ("start_by_location", *location_markers))
        if fact:
            return key, fact
    return _first_schedule_fact(facts, ("academic_year_2026_27.start", "старт занятий"))


def _schedule_group_fact(facts: Mapping[str, str], question: str, *, require_weekend: bool = False) -> tuple[str, str]:
    candidates: list[tuple[int, str, str]] = []
    grade = _grade_from_text(question)
    subject = _subject_from_text(question)
    requested_format = "online" if _has_any(question, ("онлайн", "дистанц")) else "offline" if _has_any(question, ("очно", "офлайн", "красносельск", "сретен", "пацаев")) else ""
    for key, value in facts.items():
        combined = f"{key} {value}".casefold().replace("ё", "е")
        if _is_contact_hours_fact(combined):
            continue
        if "schedule_2026_27" not in combined and "schedule.groups" not in combined:
            continue
        if require_weekend and not _fact_mentions_weekend(combined):
            continue
        if grade is not None and not _fact_matches_grade(combined, grade):
            continue
        if subject and not _fact_matches_subject(combined, subject):
            continue
        if requested_format == "online" and not _fact_mentions_online(combined):
            continue
        if requested_format == "offline" and not _fact_mentions_offline(combined):
            continue
        score = 0
        if grade is not None:
            score += 4
        if subject:
            score += 4
        if requested_format:
            score += 2
        if _fact_mentions_weekend(combined):
            score += 1
        candidates.append((score, str(key), " ".join(str(value or "").split())))
    if not candidates:
        return "", ""
    _, key, fact = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
    return key, fact


def _fact_mentions_weekend(text: str) -> bool:
    return _has_any(text, ("суббот", "воскрес", "выходн", "sat", "sun"))


def _subject_from_text(text: str) -> str:
    value = str(text or "").casefold().replace("ё", "е")
    subject_markers = {
        "math": ("математ",),
        "physics": ("физик",),
        "informatics": ("информат", "информатик"),
        "russian": ("русск", "русский"),
    }
    for subject, markers in subject_markers.items():
        if _has_any(value, markers):
            return subject
    return ""


def _fact_matches_subject(text: str, subject: str) -> bool:
    markers = {
        "math": ("math", "математ"),
        "physics": ("physics", "физик"),
        "informatics": ("informatics", "информат"),
        "russian": ("russian", "русск"),
    }.get(subject, ())
    return bool(markers and _has_any(text, markers))


def _schedule_manager_check_outcome(
    rule: Rule,
    *,
    subvariant: str,
    question: str,
    facts: Mapping[str, str],
) -> RuleOutcome:
    source: dict[str, str] = {}
    key, fact = _first_schedule_fact(facts, ("regular_courses_schedule_publication", "опубликованные группы", "расписание"))
    if fact:
        source[key or "rules_engine.schedule.publication"] = fact
    elif facts:
        key, fact = next(iter(facts.items()))
        source[str(key)] = str(fact)
    text = (
        "Точные дни конкретной группы без сверки не подтверждаю: менеджер проверит класс, предмет, формат и площадку "
        "по актуальному расписанию. Контактные часы 10:00-18:00 не считаю расписанием занятий."
    )
    return _rule_outcome(
        rule,
        subvariant=subvariant,
        route="draft_for_manager",
        text=text,
        facts=source or {"rules_engine.schedule.manager_check": "Точные дни группы требует сверки менеджером."},
        flags=("rules_engine_schedule_manager_check", "schedule_publication_safe_template_applied"),
        checklist="Rule engine: schedule — при отсутствии точной группы менеджер сверит; контакт-часы не использовать.",
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
    explicit_value = " ".join(
        str(part or "")
        for part in (
            text,
            plan.get("training_format"),
            plan.get("format"),
            plan.get("fact_scope"),
            _planner_slot_value(plan, "format"),
        )
    ).casefold().replace("ё", "е")
    explicit_format = _single_training_format(explicit_value)
    if explicit_format:
        return explicit_format
    value = " ".join([explicit_value, _known_slot_value(context, "format")]).casefold().replace("ё", "е")
    return _single_training_format(value)


def _single_training_format(value: str) -> str:
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
        _planner_slot_value(plan, "grade"),
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
    if _thread_slots_enabled(context):
        thread_slots = context.get("selling_thread_slots")
        if isinstance(thread_slots, Mapping):
            candidates.append(thread_slots.get(key))
            if key == "grade":
                candidates.append(thread_slots.get("class"))
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


def _thread_slots_enabled(context: Mapping[str, Any] | None) -> bool:
    if isinstance(context, Mapping):
        for key in ("thread_slots_enabled", "TELEGRAM_A_THREAD"):
            if key in context:
                return _truthy(context.get(key))
    return _truthy(os.getenv("TELEGRAM_A_THREAD"))


def _planner_slot_value(plan: Mapping[str, Any], key: str) -> str:
    if str(plan.get("rules_engine_intent_source") or "") != "planner":
        return ""
    slots = plan.get("planner_slots")
    if not isinstance(slots, Mapping):
        return ""
    value = slots.get(key)
    if isinstance(value, Mapping):
        value = value.get("value")
    return str(value or "").strip()


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
    return f"Для {requested_grade} класса ({format_label}) подтверждена такая стоимость: {price_part}."


def _requested_price_period(text: str) -> str:
    value = str(text or "").casefold().replace("ё", "е")
    if re.search(r"\bгод\b|за\s+год|годов", value, re.I):
        return "год"
    if re.search(r"семестр|полугод", value, re.I):
        return "семестр"
    return ""


def _asks_two_format_price(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    has_price = _has_any(value, ("стоим", "цен", "сколько", "плат"))
    has_online = _has_any(value, ("онлайн", "online", "дистанц"))
    has_offline = _has_any(value, ("очно", "очный", "офлайн", "offline", "сретен"))
    has_joiner = _has_any(value, (" и ", " или ", "вместе", "оба", "обоих", "сравн"))
    return has_price and has_online and has_offline and has_joiner


def _asks_multi_subject_price(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    has_price = _has_any(value, ("стоим", "цен", "сколько", "плат", "итогов"))
    has_multi_subject_marker = bool(re.search(r"\b(?:два|три|несколько)\s+предмет", value, re.I)) or _has_any(
        value,
        ("второй предмет", "последующий предмет", "предмета вместе", "предметы вместе"),
    )
    subject_hits = sum(1 for marker in ("математ", "физик", "информат", "русск", "обществ", "хим", "биолог") if marker in value)
    return has_price and (has_multi_subject_marker or subject_hits >= 2)


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
