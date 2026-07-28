from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.channels.answer_safety_classifier import classify_answer_safety
from mango_mvp.channels.dialogue_debug_trace import trace_event
from mango_mvp.channels.draft_prompt_builder import IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES, safe_schedule_template, should_force_manager_only
from mango_mvp.channels.fact_scope_spec import answer_scopes_allowed, detect_fact_scopes
from mango_mvp.channels.output_verification_floor import (
    AUTONOMY_SCOPE_PRECISION_ENV,
    is_near_repeat,
    parse_contract as parse_dialogue_contract,
    verify_output as verify_dialogue_contract_output,
)
from mango_mvp.channels.p0_recall_spec import HARD_P0_CODES, codes_from_text, is_benign_hypothetical_refund
from mango_mvp.channels.text_signals import has_any_marker, has_marker
from mango_mvp.channels.tone_block import apply_warm_frame
from mango_mvp.question_catalog.classifier import load_valid_theme_and_service_ids

from mango_mvp.channels.subscription_llm_parts.contracts import (
    BASE_SAFETY_FLAGS,
    SAFE_FALLBACK_DRAFT_TEXT,
    SubscriptionDraftResult,
)
from mango_mvp.channels.subscription_llm_parts.reliable_answerer import (
    preserve_partial_answer_for_live_status,
    reliable_answerer_step1_active_for_turn,
)
from mango_mvp.channels.subscription_llm_parts.semantic_reading import (
    SemanticReading,
    append_reading_trace_record,
    off_topic_reading_decision,
    reading_apply_class_enabled,
    reading_class_enabled,
    semantic_reading_trace_record,
    semantic_frame_from_metadata,
    semantic_reading_transition_metadata,
)
from mango_mvp.channels.subscription_llm_parts.support import (
    INTENT_MODEL_LED_ENV,
    MEMORY_PROVENANCE_ENV,
    PRESALE_PII_MEMORY_ENV,
    _active_brand,
    _append_fact_texts,
    _claim_supported_by_facts,
    _client_clean_fact_text,
    _direct_path_fact_value,
    _direct_path_template_fact_text,
    _direct_path_template_from_fact,
    _explicit_truthy_setting,
    _fact_match_anchors,
    _fresh_fact_texts,
    _has_dialogue_contract_retrieved_facts,
    _intent_model_led_enabled,
    _normalize_fact_match_text,
    _pilot_profile_default_on_flag_enabled,
    _p0_model_led_complaint_backstop,
    _p0_model_led_enabled,
    _p0_model_led_filter_high_risk_codes,
    _prose_model_led_enabled,
    _presale_prompt_child_name_value,
    _template_from_kb_enabled,
    _template_from_kb_trace_event,
    SEATS_DEFAULT_OPEN_REGULAR_SAFE_TEXT,
    _truthy_value,
)

ANSWER_CONTRACT_GREEN_TEMPLATE_REDUCTION_ENV = "TELEGRAM_ANSWER_CONTRACT_GREEN_TEMPLATE_REDUCTION"

SCOPE_FACT_GUARD_ENV = "TELEGRAM_SCOPE_FACT_GUARD"

A_THREAD_ENV = "TELEGRAM_A_THREAD"

PH2_OBJECTION_ENV = "TELEGRAM_PH2_OBJECTION"

PH2_ANXIETY_ENV = "TELEGRAM_PH2_ANXIETY"

SEATS_DEFAULT_OPEN_ENV = "TELEGRAM_SEATS_DEFAULT_OPEN"

PLANNER_INTENT_CONFIDENCE_THRESHOLD = 0.72
INTENT_MODEL_LED_CONFIDENCE_THRESHOLD = 0.72
INTENT_ACTIONS_FRAME_CONFIDENCE_THRESHOLD = 0.90

INTENT_MODEL_LED_TARGETS = frozenset({"live_availability", "schedule", "address", "camp", "price_fix"})
INTENT_MODEL_LED_ALLOWED = INTENT_MODEL_LED_TARGETS | frozenset({"off_topic", "other"})
INTENT_MODEL_LED_TOPIC_MAP = {
    "live_availability": "theme:026_camp_general",
    "schedule": "theme:013_schedule",
    "address": "theme:015_address",
    "camp": "theme:026_camp_general",
    "price_fix": "theme:001_pricing",
}
INTENT_MODEL_LED_ANSWER_POLICY = {
    "live_availability": ("answer_safe_parts_then_manager_live_check", "draft_for_manager"),
    "schedule": ("answer_directly_if_fact_verified", "bot_answer_self_for_pilot"),
    "address": ("answer_directly_if_fact_verified", "bot_answer_self_for_pilot"),
    "camp": ("answer_directly_if_fact_verified", "bot_answer_self_for_pilot"),
    "price_fix": ("answer_directly_if_fact_verified", "bot_answer_self_for_pilot"),
    "other": ("answer_directly_if_fact_verified", "bot_answer_self_for_pilot"),
}
INTENT_ACTIONS_FRAME_REQUESTED_ACTIONS = frozenset(
    {
        "answer_question",
        "check_availability",
        "enroll",
        "send_materials",
        "send_payment_link",
        "send_document",
        "refund_or_cancel",
        "handoff_manager",
        "unknown",
    }
)

PRICE_AMOUNT_RE = re.compile(r"\b\d[\d\s\u00a0]{1,9}\s*(?:₽|руб(?:\.|лей|ля|ль)?)", re.I)

CONCRETE_FACT_RE = re.compile(
    r"("
    r"\b\d{1,3}(?:[ \u00a0]\d{3})*\s*(?:₽|руб(?:\.|лей|ля|ль)?|%)"
    r"|\b\d{1,2}\s*(?:январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)"
    r"|\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?"
    r"|\b(?:понедельник|вторник|сред[ауеы]?|четверг|пятниц[ауеы]?|суббот[ауеы]?|воскресень[ея])\b"
    r"|\+?\d[\d\s().-]{5,}\d"
    r")",
    re.I,
)

UNKNOWN_TOPIC_FALLBACK_ID = "service:S2_unclear"

REFUND_ZERO_COLLECT_SAFE_TEXT = (
    "Приняли обращение. Передам его ответственному сотруднику, он вернётся с ответом. "
    "Пока ничего дополнительно присылать не нужно."
)

LEGAL_THREAT_SAFE_TEXT = (
    "Приняли обращение. Передам его ответственному сотруднику, он вернётся с ответом."
)

LEGAL_THREAT_PII_SAFE_TEXT = (
    "Приняли обращение. Передам его ответственному сотруднику, он вернётся с ответом."
)

COMPLAINT_SAFE_TEXT = "Передам обращение менеджеру, он вернётся с ответом."

PAYMENT_DISPUTE_SAFE_TEXT = (
    "Приняли вопрос по оплате. Передам его менеджеру: он проверит данные в системе и вернётся с ответом. "
    "Пока ничего дополнительно присылать не нужно."
)

_REFUND_ZERO_COLLECT_VARIANTS: tuple[str, ...] = (
    REFUND_ZERO_COLLECT_SAFE_TEXT,
    "Вопрос по возврату зафиксирован. Ответственный сотрудник вернётся с ответом; сейчас ничего дополнительно присылать не нужно.",
    "По возврату передам обращение ответственному сотруднику. Он вернётся с ответом, дополнительных данных пока не нужно.",
)

_COMPLAINT_SAFE_VARIANTS: tuple[str, ...] = (
    COMPLAINT_SAFE_TEXT,
    "Вопрос по жалобе зафиксирован. Менеджер вернётся с ответом.",
    "Передам обращение менеджеру, он разберет ситуацию и вернётся с ответом.",
)

_PAYMENT_DISPUTE_VARIANTS: tuple[str, ...] = (
    PAYMENT_DISPUTE_SAFE_TEXT,
    "Понимаю тревогу: по оплате нужно сверить данные в системе. Передам вопрос менеджеру, он проверит и вернётся с точным ответом.",
    "Вижу, что вопрос срочный. По платежу безопасно ответит менеджер после проверки в системе; передам ему это отдельно.",
    "По оплате не буду подтверждать статус без сверки. Передам вопрос менеджеру, он проверит данные и вернётся с ответом.",
)

_LEGAL_SAFE_VARIANTS: tuple[str, ...] = (
    LEGAL_THREAT_SAFE_TEXT,
    "Юридический вопрос зафиксирован. Ответственный сотрудник вернётся с ответом.",
    "Передам обращение ответственному сотруднику, он вернётся с ответом.",
)

SOFT_NEGATIVE_HANDOFF_SAFE_TEXT = (
    "Поняла, давайте не буду повторять общий ответ. Передам менеджеру контекст переписки, "
    "чтобы он ответил по вашему вопросу точнее."
)

RESULT_GUARANTEE_SAFE_TEXT = (
    "Мы не даём и не гарантируем конкретный балл: результат зависит от ученика, регулярности занятий "
    "и самостоятельной работы. Менеджер свяжется, уточнит цель и может показать, какая у нас статистика результатов."
)

ADMISSION_GUARANTEE_SAFE_TEXT = (
    "Мы не даём и не гарантируем поступление: результат зависит от ученика и выбранной траектории подготовки. "
    "Есть статистика: 97% наших учеников поступают в желаемые вузы. Менеджер свяжется и подробно поможет подобрать программу."
)

FOTON_SECOND_SUBJECT_DISCOUNT_TEXT = (
    "Да, скидка есть: на второй и последующий предмет одного и того же ребёнка при очном формате — 20%, "
    "при онлайн-формате — 30%. Скидки не суммируются. Менеджер проверит условия под вашу ситуацию."
)

UNPK_SECOND_SUBJECT_DISCOUNT_TEXT = (
    "Да, скидка есть: на второй и последующий предмет одного и того же ребёнка при очном формате — 20%, "
    "при онлайн-формате — 20%. Скидки не суммируются. Менеджер проверит условия под вашу ситуацию."
)

UNPK_MONTHLY_SEMESTER_DISCOUNT_TEXT = (
    "В УНПК можно платить помесячно, за семестр или за год. "
    "При оплате за семестр действует скидка 10%, за год - 14%. "
    "Если нужно растянуть оплату, менеджер подскажет варианты под вашу ситуацию."
)

MULTICHILD_DISCOUNT_TEXT = (
    "Да, для детей из многодетной семьи есть скидка 10%; нужно удостоверение многодетной семьи, "
    "даже если учится один ребёнок или два ребёнка. "
    "Скидка не суммируется с другими скидками: применяется наибольшая. Менеджер поможет проверить условия."
)

DISCOUNT_STACKING_SAFE_TEXT = "Скидки не суммируются: применяется наибольшая доступная скидка. Менеджер проверит условия под вашу ситуацию."

FOTON_INSTALLMENT_SAFE_TEXT = (
    "Да, в Фотоне можно оплатить обучение частями: доступны варианты на 6, 10 или 12 месяцев, "
    "а также сервис Долями. Это относится к очным и онлайн-курсам, ЛВШ, ЛШ и другим программам Фотона. "
    "По обычным курсам также можно обсудить помесячную оплату или оплату за семестр. "
    "Конкретные условия и оформление зависят от выбранного способа оплаты; менеджер поможет подобрать удобный вариант."
)

FOTON_CAMP_INSTALLMENT_SAFE_TEXT = (
    "Да, для ЛВШ, ЛШ и лагерей Фотона тоже можно оплатить частями: доступны варианты на 6, 10 или 12 месяцев, "
    "а также сервис Долями. Менеджер поможет выбрать способ оплаты и оформить его дистанционно."
)

FOTON_DOLYAMI_SAFE_TEXT = (
    "Да, Долями можно использовать в Фотоне. По точному числу частей и процентам не буду обещать без оформления: "
    "условия зависят от выбранного способа оплаты и платёжного сервиса. Подтверждённо: в Фотоне также доступны варианты "
    "оплаты частями на 6, 10 или 12 месяцев для очных и онлайн-курсов, ЛВШ, ЛШ и других программ. "
    "Менеджер поможет выбрать и оформить подходящий вариант дистанционно."
)

PROMOCODE_SAFE_TEXT = "Промокодов сейчас нет. Из реальных выгод: при оплате за семестр или за год выходит выгоднее — это уже учтено в прайсе."

UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT = (
    "В УНПК рассрочки нет, это не банковская рассрочка, поэтому одобрение банка не требуется. "
    "Можно платить помесячно, за семестр или за год. "
    "При оплате за семестр действует скидка 10%, за год - 14%. "
    "Если нужно растянуть оплату, менеджер подскажет варианты под вашу ситуацию."
)

UNPK_ZVSH_WAITLIST_SAFE_TEXT = (
    "Здравствуйте! Даты зимней выездной школы в Менделеево на новый учебный год пока уточняются. "
    "Мы ждём расписание; записаться можно прямо сейчас в лист ожидания. "
    "Как только расписание появится, менеджер свяжется с вами и сориентирует по условиям."
)

MATKAP_REGIONAL_SAFE_TEXT = "К сожалению, региональный не принимаем: работаем только с федеральным маткапиталом. Менеджер подскажет порядок оформления."

MATKAP_SFR_REVIEW_SAFE_TEXT = "Рассмотрение проводит СФР, поэтому мы не можем обещать одобрение. Менеджер поможет проверить порядок оформления."

MATKAP_FEDERAL_TIMING_SAFE_TEXT = (
    "Да, мы работаем с федеральным материнским капиталом. СФР рассматривает заявление до 10 рабочих дней, "
    "перевод занимает ещё до 5 рабочих дней, ориентир — до 15 рабочих дней. "
    "Решение принимает СФР. Перечень документов подготовит менеджер."
)

TAX_ONLINE_FORM_SAFE_TEXT = (
    "По онлайн-курсу это зависит от трактовки налоговой инспекции. Специалист и менеджер проверят, "
    "какие документы можно корректно подготовить по вашему курсу."
)

TAX_FNS_REVIEW_SAFE_TEXT = "ФНС рассматривает заявление и принимает решение. Справка помогает подтвердить обучение, а менеджер подскажет порядок оформления."

TAX_DEDUCTION_PROCESS_SAFE_TEXT = (
    "Налоговый вычет оформляется через налоговую: решение и выплату принимает ФНС. "
    "Со своей стороны поможем подготовить документы для вычета; справку готовим до 10 рабочих дней."
)

TAX_AMOUNT_SAFE_TEXT = (
    "Да, налоговый вычет оформить можно: у нас есть лицензия. "
    "За обучение ребёнка можно вернуть до 14 300 ₽ в год — это 13% с расходов до 110 000 ₽. "
    "Подать можно за 3 предыдущих года; за 2023 год и ранее действовал лимит 50 000 ₽, возврат до 6 500 ₽. "
    "Если занимаются двое детей, лимит считается отдельно на каждого ребёнка, то есть ориентир до 28 600 ₽ за год. "
    "Справку для вычета готовим до 10 рабочих дней; решение и сроки выплаты остаются на стороне ФНС."
)

TAX_LICENSE_SAFE_TEXT = "Да, есть лицензия на ведение образовательной деятельности. Менеджер поможет подготовить документы для налогового вычета."

UNPK_LVSH_SEATS_SAFE_TEXT = "Обычно группа 12-15 человек. По ЛВШ УНПК места распроданы; могу записать в лист ожидания. Как альтернатива — городская очная школа."

FOTON_LVSH_PRICE_SAFE_TEXT = (
    "ЛВШ Менделеево у Фотона сейчас стоит 93 100 ₽. "
    "Полная стоимость — 98 000 ₽. "
    "Места распроданы; могу записать в лист ожидания. Как альтернатива — городская очная школа."
)

FOTON_CAMP_OVERVIEW_SAFE_TEXT = (
    "У Фотона есть два летних формата: выездная школа в Менделеево и городская летняя школа в Москве. "
    "Подбираем смену по классу, предмету и формату; наличие мест по конкретной смене проверит менеджер."
)

FOTON_ONLINE_TRIAL_SAFE_TEXT = (
    "В онлайн-формате Фотона можем прислать вам фрагмент занятия — посмотреть подачу и уровень; оформление проходит дистанционно — приезжать не нужно. "
    "Условия просмотра фрагмента подтвердит менеджер перед записью."
)

UNPK_TRIAL_SAFE_TEXT = (
    "По очному формату сейчас обычно не начинаем с отдельного пробного занятия. "
    "По онлайн-формату можем прислать вам фрагмент занятия — посмотреть подачу и уровень. "
    "Если рассматриваете очный курс, менеджер расскажет про формат, преподавателей и поможет понять, подойдёт ли программа."
)

FOTON_OFFLINE_FREE_TRIAL_GUARD_TEXT = (
    "По очному формату бесплатное пробное по умолчанию не обещаю. "
    "Очный пробный шаг согласует менеджер при записи: он проверит подходящую группу, филиал и условия. "
    "Запрос передам именно как очный, без подмены на онлайн-фрагмент."
)

UNPK_LVSH_PRICE_SAFE_TEXT = (
    "ЛВШ Менделеево в УНПК сейчас стоит 114 000 ₽. "
    "В стоимость входит проживание и 5-разовое питание; места распроданы, могу записать в лист ожидания. "
    "Как альтернатива — городская очная школа."
)

UNPK_LVSH_LIVING_TRANSFER_SAFE_TEXT = (
    "Да, в ЛВШ Менделеево УНПК есть проживание и 5-разовое питание. "
    "Текущая цена сейчас — 114 000 ₽, полная стоимость — 120 000 ₽. "
    "Места распроданы; могу записать в лист ожидания. Как альтернатива — городская очная школа."
)

UNPK_LVSH_PRICE_DETAILS_SAFE_TEXT = (
    "По ЛВШ Менделеево в УНПК: полная стоимость — 120 000 ₽, текущая цена сейчас — 114 000 ₽. "
    "Места распроданы; могу записать в лист ожидания. Как альтернатива — городская очная школа."
)

UNPK_LVSH_GRADE_11_PRICE_DETAILS_SAFE_TEXT = (
    "По цене ЛВШ Менделеево в УНПК: полная стоимость — 120 000 ₽, текущая цена сейчас — 114 000 ₽. "
    "При этом сама ЛВШ обычно рассчитана на учеников, окончивших 5-10 класс; "
    "места распроданы; могу записать в лист ожидания. Как альтернатива — городская очная школа."
)

UNPK_LVSH_GRADE_11_SAFE_TEXT = (
    "По ЛВШ Менделеево важный момент: программа обычно рассчитана на учеников, окончивших 5-10 класс; "
    "ИТ-направление — на 7-10 класс. Для 11 класса менеджер проверит подходящую альтернативу под ваш предмет. "
    "Если говорить справочно о самой ЛВШ Менделеево, текущая цена сейчас — 114 000 ₽; места распроданы, могу записать в лист ожидания."
)

UNPK_CAMP_OVERVIEW_SAFE_TEXT = (
    "У УНПК есть два летних формата: выездная ЛВШ в Менделеево с проживанием и городская летняя школа без проживания. "
    "Подбирать лучше по классу, предмету и формату: с проживанием или дневная программа. "
    "Напишите класс ребёнка — сориентирую по подходящему варианту, а наличие мест проверит менеджер."
)

UNPK_CAMP_ONLINE_FORMAT_SAFE_TEXT = (
    "Летние лагеря и ЛВШ УНПК — очные форматы. Если нужен именно онлайн по вашему предмету, "
    "менеджер проверит актуальные варианты УНПК, расписание и стоимость, чтобы не сориентировать неверно."
)

FOTON_CITY_CAMP_AUGUST_SAFE_TEXT = (
    "Да, у Фотона есть дневная городская летняя школа в Москве: ЛШ Москва Фотон проходит 3-14 августа, "
    "адрес — Верхняя Красносельская. Менеджер проверит подходящую программу, смену и наличие мест под класс ребёнка."
)

FOTON_LVSH_DATES_SAFE_TEXT = "ЛВШ Менделеево у Фотона: 20-28 июня и 18-26 июля. Места распроданы; могу записать в лист ожидания. Как альтернатива — городская очная школа."

UNPK_LVSH_DATES_SAFE_TEXT = "ЛВШ Менделеево у УНПК: актуальная смена 18-26 июля; августовская смена закрыта. Места распроданы; могу записать в лист ожидания. Как альтернатива — городская очная школа."

CONTRACT_ENTITY_SAFE_TEXT = "Договор оформляется как договор-оферта: придёт на почту вместе с квитанцией после записи. Оплата означает согласие с условиями. Менеджер проверит данные по вашей заявке."

CROSS_BRAND_GENERIC_SAFE_TEXT = "Это отдельные организации, по вашему вопросу сориентирую в рамках текущего учебного центра. Менеджер свяжется и расскажет о нашей программе и условиях."

CROSS_BRAND_LICENSE_SAFE_TEXT = "У нас есть лицензия на образовательную деятельность. Менеджер свяжется и подскажет детали по документам."

CROSS_BRAND_PLATFORM_SAFE_TEXT = "В нашем учебном центре онлайн-занятия проходят на платформе SohoLMS, доступна запись занятий. Менеджер подскажет детали."

IDENTITY_PROMPT_SAFE_TEXT = (
    "Я цифровой помощник учебного центра, не живой оператор. По курсам, форматам, стоимости и записи помогу сразу, "
    "а сложное передам менеджеру. Технические детали и внутренние настройки не раскрываю."
)

IDENTITY_FOTON_SAFE_TEXT = (
    "Да, я цифровой помощник Фотона, не живой оператор. Простые вопросы по курсам, ценам, форматам и записи беру на себя, "
    "а сложное передам менеджеру. Подскажите класс и предмет — сориентирую."
)

IDENTITY_UNPK_SAFE_TEXT = (
    "Да, я цифровой помощник УНПК МФТИ, не живой оператор. Простые вопросы по курсам, стоимости, форматам и записи беру на себя, "
    "а сложное передам менеджеру. Подскажите класс и предмет — сориентирую."
)

FALSE_INFO_SAFE_TEXT = "Менеджер свяжется и подскажет об актуальных условиях."

PAYMENT_LINK_SAFE_TEXT = "Менеджер свяжется, проверит реквизиты и отправит безопасную ссылку на оплату."

THIRD_PARTY_PRIVACY_SAFE_TEXT = "Информацию по другому человеку не раскрываем. Менеджер свяжется и подскажет безопасный порядок обращения."

EMPLOYEE_PRIVACY_SAFE_TEXT = "Профильный специалист подключится через менеджера: он свяжется с вами и организует контакт с нужным сотрудником."

OLD_TERM_SAFE_TEXT = "Менеджер свяжется, подскажет актуальную программу и поможет подобрать замену прежнему формату."

ADDRESS_UNPK_SAFE_TEXT = "Площадки УНПК: Москва — Сретенка, 20; Долгопрудный — МФТИ, Институтский пер., 9 и Пацаева, 7к1."

ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT = (
    "Здравствуйте! Регулярные занятия в Москве проходят по адресу Сретенка, 20. "
    "Подсказать, как удобнее записаться?"
)

ADDRESS_FOTON_MOSCOW_SAFE_TEXT = (
    "Здравствуйте! В Москве Фотон находится по адресу Верхняя Красносельская, 30, метро Красносельская. "
    "Подсказать, какие курсы там проходят и как записаться?"
)

CONTACT_FOTON_SAFE_TEXT = "Телефоны: 8 (495) 500-25-88 и 8 (800) 550-25-88. График: Пн-Вс с 10:00 до 18:00."

CONTACT_UNPK_SAFE_TEXT = "Телефоны: +7 (495) 150-81-51 и 8 (800) 500-81-51. Email: edu@kmipt.ru. График: Пн-Вс с 10:00 до 18:00."

OFF_TOPIC_FOTON_SAFE_TEXT = "Я помогаю с вопросами об обучении в Фотоне. По другим темам не сориентирую, но могу помочь подобрать курс, формат, расписание или следующий шаг."

OFF_TOPIC_UNPK_SAFE_TEXT = "Я помогаю с вопросами об обучении в УНПК МФТИ. По другим темам не сориентирую, но могу помочь подобрать курс, формат, расписание или следующий шаг."

OFF_TOPIC_GENERIC_SAFE_TEXT = "Я помогаю с вопросами об обучении. По другим темам не сориентирую, но могу помочь подобрать курс, формат, расписание или следующий шаг."

PROGRAM_HANDOFF_SAFE_TEXT = "Менеджер свяжется и подскажет актуальную программу."

INDIVIDUAL_HANDOFF_SAFE_TEXT = "Менеджер свяжется и подскажет варианты индивидуальных занятий."

QUITTANCE_SAFE_TEXT = "Менеджер свяжется и подскажет, на каком оформлении будет квитанция."

BRAND_LOYALTY_FOTON_TEXT = "Рады, что выбрали Фотон! Менеджер свяжется и сориентирует по программе."

BRAND_LOYALTY_UNPK_TEXT = "Рады, что выбрали УНПК МФТИ! Менеджер свяжется и сориентирует по программе."

MISSING_PRICE_HELPFUL_TEXT = (
    "Могу сориентировать по вариантам: стоимость зависит от класса, формата и периода оплаты. "
    "Напишите, пожалуйста, класс ребёнка и какой формат удобнее — очно или онлайн. "
    "Менеджер проверит актуальную стоимость и предложит подходящий вариант."
)

MISSING_INTENSIVE_PRICE_HELPFUL_TEXT = (
    "По интенсивам стоимость зависит от класса, предмета, длительности и актуального набора. "
    "Точную цену сейчас не называю без проверки. Напишите, пожалуйста, класс ребёнка и предмет — "
    "менеджер проверит актуальную программу и стоимость."
)

MISSING_SCHEDULE_HELPFUL_TEXT = (
    "Расписание зависит от класса, предмета, формата и площадки. "
    "Напишите, пожалуйста, класс ребёнка, предмет и какие дни удобнее — суббота или воскресенье. "
    "Менеджер подберёт ближайший подходящий вариант."
)

MISSING_PROGRAM_HELPFUL_TEXT = (
    "Поможем подобрать программу под цель ребёнка: школьная база, подготовка к экзаменам или олимпиадам требуют разного темпа. "
    "Напишите класс, предмет и цель обучения — менеджер подскажет подходящий курс."
)

MISSING_DISCOUNT_HELPFUL_TEXT = (
    "Скидки зависят от программы и условий участия. "
    "Напишите, пожалуйста, какой курс рассматриваете и учился ли ребёнок у нас раньше — менеджер проверит доступные варианты."
)

MISSING_INSTALLMENT_HELPFUL_TEXT = (
    "Варианты оплаты зависят от программы и периода обучения. "
    "Напишите, пожалуйста, какой курс рассматриваете — менеджер подскажет, как удобнее распределить оплату."
)

MISSING_DOCS_HELPFUL_TEXT = (
    "По документам поможем сориентироваться безопасно: порядок зависит от типа документа и ситуации. "
    "Напишите, пожалуйста, какой документ нужен — справка, договор, чек или документы для вычета/маткапитала. "
    "Менеджер проверит перечень и подскажет следующий шаг."
)

MISSING_CAMP_HELPFUL_TEXT = (
    "По лагерям и выездным школам важно подобрать смену под класс, предмет и формат. "
    "Напишите, пожалуйста, класс ребёнка и интересующее направление — менеджер проверит актуальные смены и наличие мест."
)

MISSING_GENERAL_HELPFUL_TEXT = (
    "Помогу сориентироваться по обучению. Напишите, пожалуйста, класс ребёнка, предмет и цель: подтянуть школьную программу, "
    "подготовиться к экзамену или олимпиаде. По этим данным менеджер предложит подходящий следующий шаг."
)

KNOWN_CONTEXT_REPAIR_TEXT = (
    "Да, вижу данные из переписки — повторно присылать их не нужно. "
    "Отвечу по сути, а детали, которые требуют проверки по группе или месту, передам менеджеру."
)

PROMOCODE_DRAFT_RE = re.compile(r"\b(?:LVSH-VEB20|LVSH-KF-10|ABRAMOV|VAGIN)\b", re.I)

AUTONOMOUS_ROUTES = {"bot_answer_self", "bot_answer_self_for_pilot"}

AUTONOMY_MATRIX_SAFE_TOPIC_IDS = {
    "theme:001_pricing",
    "theme:005_discounts",
    "theme:006_installment",
    "theme:007_matkap_payment",
    "theme:008_tax_deduction",
    "theme:011_contract",
    "theme:012_certificates",
    "theme:013_schedule",
    "theme:014_format",
    "theme:015_address",
    "theme:016_program",
    "theme:018_materials_homework",
    "theme:019a_positive_feedback",
    "theme:020_enrollment",
    "theme:021_continuation",
    "theme:022_age_level_testing",
    "theme:023_trial_class",
    "theme:024_account_access",
    "theme:025_missing_links_access",
    "theme:026_camp_general",
    "theme:027_camp_living_conditions",
    "theme:028_transport_logistics",
    "service:S5_general_consultation",
}

HIGH_RISK_THEME_IDS = {
    "theme:009_refund",
    "theme:019b_negative_feedback",
    "theme:029_legal_question",
}

HIGH_RISK_MARKERS = (
    "refund",
    "legal",
    "negative",
    "возврат",
    "суд",
    "иск",
    "претензи",
    "роспотребнадзор",
    "жалоб",
)

COMBINED_NON_RISK_INPUT_RE = re.compile(
    r"сколько\s+сто|стоимост|цен[ауеы]?|прайс|расписан|когда|дат[аы]|лагер|л[вгз]ш|"
    r"курс|заняти|онлайн|очно|смен[аы]|математик|физик|информатик",
    re.I,
)

RESULT_GUARANTEE_INPUT_RE = re.compile(
    r"гарантир\w*[^.!?\n]{0,80}(?:балл|егэ|огэ|результат|сдаст)"
    r"|(?:сдаст|балл\w*|результат)[^.!?\n]{0,80}гарантир\w*"
    r"|точно[^.!?\n]{0,60}сдаст[^.!?\n]{0,60}(?:егэ|огэ|на\s*\d{2,3}\+?\s*(?:балл\w*)?)"
    r"|(?:\b90\b|\b100\b)[^.!?\n]{0,80}балл\w*"
    r"|гарантир\w*[^.!?\n]{0,80}диплом\w*"
    r"|диплом\w*[^.!?\n]{0,80}гарантир\w*",
    re.I,
)

ADMISSION_GUARANTEE_INPUT_RE = re.compile(
    r"гарантир\w*[^.!?\n]{0,80}(?:поступ\w*|пройд[её]?\w*)"
    r"|поступ\w*[^.!?\n]{0,80}гарантир\w*"
    r"|пройд[её]?\w*[^.!?\n]{0,80}гарантир\w*"
    r"|точно[^.!?\n]{0,60}поступ\w*"
    r"|точно[^.!?\n]{0,60}пройд[её]?\w*"
    r"|поступ\w*[^.!?\n]{0,60}точно",
    re.I,
)

PAYMENT_CONFIRMATION_RE = re.compile(
    r"оплат[ауы]\s+(?:отмечен|прошл|поступил|зачислен|получен)"
    r"|плат[её]ж\s+(?:прош[её]л|получен|зачислен)"
    r"|вижу,\s*что\s+оплат|оплата\s+есть|мы\s+получили\s+оплат",
    re.I,
)

FUTURE_PRICE_INPUT_RE = re.compile(
    r"\b(?:после\s+1\s+(?:июля|августа)|после\s+0?1[./-]0?7(?:[./-]\d{2,4})?|"
    r"в\s+август\w*|августовск\w+|после\s+повышени\w*|с\s+сентябр\w*|будущ\w+\s+цен\w*|цена\s+выраст\w*)\b",
    re.I,
)

PRECISE_CONDITION_RE = re.compile(
    r"\b\d[\d\s\u00a0]{1,9}\s*(?:руб|₽|р\.|%)|\bрассрочк\w*\s+доступн|\bскидк\w*\s+\d",
    re.I,
)

BRAND_FORBIDDEN_TERMS = {
    "foton": ("унпк", "унпк мфти", "ано дпо", "ноу унпк", "kmipt.ru"),
    "unpk": ("фотон", "цдпо", "црдо", "cdpofoton", "т-банк", "долями", "рассрочка через банк", "через банк"),
}

BRAND_OUTPUT_MARKERS = {
    "foton": ("фотон", "foton", "цдпо", "црдо", "cdpofoton"),
    "unpk": ("унпк", "унпк мфти", "unpk", "мфти", "kmipt"),
}

_BARE_N_POINTS_RE = re.compile(r"\b\d{1,3}\+?\s*балл\w*", re.I)

_N_POINTS_PROMISE_CONTEXT_RE = re.compile(
    r"(?:гарантир\w*|обеща\w*|получит\w*|получите|набрать|набер[её]т\w*|набер[её]те|сдаст\w*|"
    r"сдадите|поступит\w*|ваш\w*\s+реб[её]н\w*|ученик\w*)"
    r"[^.!?\n]{0,80}\b\d{1,3}\+?\s*балл\w*"
    r"|\b\d{1,3}\+?\s*балл\w*[^.!?\n]{0,80}"
    r"(?:гарантир\w*|обеща\w*|получит\w*|набрать|набер[её]т\w*|набер[её]те|сдаст\w*|сдадите|поступит\w*)",
    re.I,
)

UNSUPPORTED_PROMISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{1,3}(?:[,.]\d{1,2})?\s*(?:%|процент\w*)", re.I),
    re.compile(r"\b\d[\d\s\u00a0]{1,9}\s*(?:руб(?:\.|лей|ля|ль)?|₽|р\.)", re.I),
    _N_POINTS_PROMISE_CONTEXT_RE,
    re.compile(r"\b\d+\s*(?:к|тыс\.?|тысяч)\b", re.I),
    re.compile(
        r"\b(?:до|по)\s+\d{1,2}(?:[./-]\d{1,2}(?:[./-]\d{2,4})?|\s+"
        r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря))",
        re.I,
    ),
)

_SAFE_TEMPLATE_DISPATCHER_RECONSIDER_BLOCKING_FLAGS = {
    "cross_brand_safe_template_applied",
    "cross_brand_client_text_blocked",
    "brand_separation_guarded",
    "result_guarantee_safe_template_applied",
    "admission_guarantee_safe_template_applied",
    "unsupported_promise_detected",
    "zero_collect_legal_guarded",
    "zero_collect_refund_guarded",
    "complaint_apology_guarded",
    "payment_dispute_manager_only",
    "high_risk_manager_only",
    "rules_engine_olympiad_grade_outside_9_11",
    "placeholder_in_draft",
    "identity_disclosure_guarded",
}














def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0























def _metadata_with_self_route_deferral_cleared(metadata: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(metadata)
    pipeline = (
        dict(merged.get("dialogue_contract_pipeline"))
        if isinstance(merged.get("dialogue_contract_pipeline"), Mapping)
        else {}
    )
    if pipeline:
        pipeline["is_manager_deferral"] = False
        pipeline["reason_class"] = ""
        pipeline["reason_evidence"] = {}
        merged["dialogue_contract_pipeline"] = pipeline
    merged["is_manager_deferral"] = False
    merged["reason_class"] = ""
    return merged

def _metadata_with_guarded_original_text(
    metadata: Mapping[str, Any],
    text: str,
    *,
    guard: str,
) -> dict[str, Any]:
    merged = dict(metadata)
    original = " ".join(str(text or "").split())[:500]
    if not original:
        return merged
    merged.setdefault("guarded_original_text", original)
    if guard:
        merged.setdefault("guarded_original_text_guard", str(guard)[:80])
        guards = [str(item) for item in (merged.get("guarded_original_text_guards") or []) if str(item).strip()]
        if guard not in guards:
            guards.append(str(guard)[:80])
        merged["guarded_original_text_guards"] = guards[:8]
    return merged




def _unpk_moscow_address_template_from_kb(context: Optional[Mapping[str, Any]]) -> str:
    return _direct_path_template_from_fact(
        active_brand="unpk",
        fact_key="locations_unpk.addresses.1.address",
        literal_text=ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
        neutral_fallback="Адрес московской площадки УНПК лучше уточнит менеджер по выбранному формату.",
        context=context,
        render=lambda text: (
            f"Здравствуйте! Регулярные занятия в Москве проходят по адресу {_direct_path_fact_value(text)}. "
            "Если хотите, подскажу ближайшие группы."
        ),
    )



def _autonomy_scope_precision_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    explicit = _explicit_truthy_setting(
        context,
        AUTONOMY_SCOPE_PRECISION_ENV,
        aliases=("autonomy_scope_precision", "autonomy_scope_precision_enabled"),
    )
    if explicit is not None:
        return bool(explicit)
    return _pilot_profile_default_on_flag_enabled(
        context,
        AUTONOMY_SCOPE_PRECISION_ENV,
        aliases=("autonomy_scope_precision", "autonomy_scope_precision_enabled"),
    )


def _asks_unpk_regular_moscow_route(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    asks_route = bool(
        re.search(
            r"как\s+(?:доехать|добраться|попасть|пройти|проехать)|доехать|добраться|проезд|маршрут",
            value,
            re.I,
        )
    )
    mentions_regular_moscow = bool(re.search(r"моск|заняти|площадк|очн|регулярн|обычн|адрес", value, re.I))
    mentions_camp = bool(re.search(r"лвш|менделеев|лагер|camp|летн|трансфер|выездн", value, re.I))
    return asks_route and mentions_regular_moscow and not mentions_camp


def _autonomy_scope_precision_repaired_address_text(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    if not _autonomy_scope_precision_enabled(context) or _active_brand(context) != "unpk":
        return ""
    if not _asks_unpk_regular_moscow_route(client_message):
        return ""
    if not re.search(r"лвш|менделеев|льяловск|красный\s+воин", str(result.draft_text or "").casefold().replace("ё", "е"), re.I):
        return ""
    return _unpk_moscow_address_template_from_kb(context)




def _context_with_dialogue_contract_retrieved_facts(
    context: Optional[Mapping[str, Any]],
    result: SubscriptionDraftResult,
) -> Optional[Mapping[str, Any]]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    retrieved_sources = []
    if isinstance(pipeline.get("retrieved_facts"), Mapping):
        retrieved_sources.append(pipeline.get("retrieved_facts"))
    if isinstance(direct.get("retrieved_facts"), Mapping):
        retrieved_sources.append(direct.get("retrieved_facts"))
    direct_fact_metadata = (
        dict(direct.get("wide_fact_metadata"))
        if isinstance(direct.get("wide_fact_metadata"), Mapping)
        else {}
    )
    facts = {
        str(key): str(value)
        for retrieved in retrieved_sources
        for key, value in retrieved.items()
        if str(key).strip() and str(value).strip()
    }
    if not facts:
        return context

    merged: dict[str, Any] = dict(context) if isinstance(context, Mapping) else {}
    confirmed = dict(merged.get("confirmed_facts")) if isinstance(merged.get("confirmed_facts"), Mapping) else {}
    confirmed.update(facts)

    merged_pipeline = (
        dict(merged.get("dialogue_contract_pipeline"))
        if isinstance(merged.get("dialogue_contract_pipeline"), Mapping)
        else {}
    )
    merged_retrieved = (
        dict(merged_pipeline.get("retrieved_facts"))
        if isinstance(merged_pipeline.get("retrieved_facts"), Mapping)
        else {}
    )
    merged_retrieved.update(facts)
    merged_pipeline["retrieved_facts"] = merged_retrieved

    facts_context = dict(merged.get("facts_context")) if isinstance(merged.get("facts_context"), Mapping) else {}
    facts_context_confirmed = (
        dict(facts_context.get("confirmed_facts"))
        if isinstance(facts_context.get("confirmed_facts"), Mapping)
        else {}
    )
    facts_context_confirmed.update(facts)
    facts_context_payload = {
        "stale": False,
        "facts_stale": False,
        "fresh": True,
        "facts_fresh": True,
        "fresh_facts": True,
        "client_safe_fact_verified": True,
        "confirmed_facts": facts_context_confirmed,
    }
    if direct_fact_metadata:
        facts_context_metadata = (
            dict(facts_context.get("fact_metadata"))
            if isinstance(facts_context.get("fact_metadata"), Mapping)
            else {}
        )
        facts_context_metadata.update(
            {str(key): dict(value) for key, value in direct_fact_metadata.items() if isinstance(value, Mapping)}
        )
        facts_context_payload["fact_metadata"] = facts_context_metadata
    facts_context.update(facts_context_payload)

    quality = dict(merged.get("context_quality")) if isinstance(merged.get("context_quality"), Mapping) else {}
    quality["facts_stale"] = False

    merged_payload: dict[str, Any] = {
        "confirmed_facts": confirmed,
        "dialogue_contract_pipeline": merged_pipeline,
        "facts_context": facts_context,
        "context_quality": quality,
        "facts_fresh": True,
        "facts_stale": False,
    }
    if direct_fact_metadata:
        merged_payload["direct_path_fact_metadata"] = direct_fact_metadata
    merged.update(merged_payload)
    return merged

_GUARDCHAIN_RECOVERY_BLOCKING_FLAGS = {
    "cross_brand_safe_template_applied",
    "cross_brand_client_text_blocked",
    "brand_separation_guarded",
    "result_guarantee_safe_template_applied",
    "admission_guarantee_safe_template_applied",
    "unsupported_promise_detected",
    "zero_collect_legal_guarded",
    "zero_collect_refund_guarded",
    "complaint_apology_guarded",
    "payment_dispute_manager_only",
    "high_risk_manager_only",
    "rules_engine_olympiad_grade_outside_9_11",
}

def _pipeline_fact_texts(result: SubscriptionDraftResult) -> dict[str, str]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    facts: dict[str, Any] = {}
    if isinstance(pipeline.get("retrieved_facts"), Mapping):
        facts.update(dict(pipeline.get("retrieved_facts") or {}))
    direct = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    if isinstance(direct.get("retrieved_facts"), Mapping):
        facts.update(dict(direct.get("retrieved_facts") or {}))
    return {
        str(key): str(value)
        for key, value in facts.items()
        if str(key).strip() and str(value).strip()
    }

def _pipeline_contract(
    result: SubscriptionDraftResult,
    *,
    active_brand: str,
    fact_keys: Sequence[str],
):
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    return parse_dialogue_contract(
        pipeline.get("contract"),
        active_brand=active_brand,
        fact_key_catalog=tuple(fact_keys),
    )

def _verified_informational_answer(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
    template_name: str = "",
) -> bool:
    if result.route == "manager_only" or is_high_risk_result(result):
        return False
    if set(detect_high_risk_input_markers(client_message, context=context)):
        return False
    flags = set(result.safety_flags)
    if flags.intersection(_GUARDCHAIN_RECOVERY_BLOCKING_FLAGS):
        return False
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    if any(bool(metadata.get(flag)) for flag in _GUARDCHAIN_RECOVERY_BLOCKING_FLAGS):
        return False
    fact_texts = _pipeline_fact_texts(result)
    if not fact_texts:
        return False
    if not _claim_supported_by_facts(result.draft_text, tuple(fact_texts.values())):
        return False
    contract = _pipeline_contract(result, active_brand=_active_brand(context), fact_keys=tuple(fact_texts.keys()))
    if contract.is_p0:
        return False
    findings = verify_dialogue_contract_output(
        result.draft_text,
        facts=fact_texts,
        active_brand=_active_brand(context),
        contract=contract,
        client_message=client_message,
        context=context,
        previous_bot_texts=_humanity_previous_bot_texts(context),
    )
    if findings:
        return False
    if template_name in {"matkap", "tax"} and not _strict_informational_yield_ok(
        result,
        template_name=template_name,
        client_message=client_message,
        context=context,
        fact_texts=fact_texts,
    ):
        return False
    return True

def _strict_informational_yield_ok(
    result: SubscriptionDraftResult,
    *,
    template_name: str,
    client_message: str,
    context: Optional[Mapping[str, Any]],
    fact_texts: Mapping[str, str],
) -> bool:
    draft_text = str(result.draft_text or "")
    facts_blob = " ".join(str(value or "") for value in fact_texts.values())
    if _informational_yield_has_unbacked_concrete_anchors(draft_text, facts_blob=facts_blob):
        return False
    if _mentions_unbacked_children_rule(draft_text, facts_blob=facts_blob):
        return False
    if template_name == "tax" and _asks_non_tax_document_or_contract(client_message, context=context) and _answers_tax_deduction_scope(draft_text):
        return False
    if template_name == "matkap" and _asks_non_matkap_document_or_contract(client_message, context=context) and _answers_matkap_scope(draft_text):
        return False
    return True

def _informational_yield_has_unbacked_concrete_anchors(draft_text: str, *, facts_blob: str) -> bool:
    draft_anchors = _fact_match_anchors(draft_text)
    if not draft_anchors:
        return False
    fact_anchors = _fact_match_anchors(facts_blob)
    allowed_prefixes = ("number:", "date:", "condition:", "unit:")
    unbacked = {
        anchor
        for anchor in draft_anchors - fact_anchors
        if anchor.startswith(allowed_prefixes)
    }
    return bool(unbacked)

def _mentions_unbacked_children_rule(draft_text: str, *, facts_blob: str) -> bool:
    draft = str(draft_text or "").casefold().replace("ё", "е")
    if not re.search(r"\b(?:двое|двух|два|2)\s+(?:дет|реб)", draft, re.I):
        return False
    if re.search(r"\b(?:двое|двух|два|2)\s+(?:дет|реб)", str(facts_blob or "").casefold().replace("ё", "е"), re.I):
        return False
    return bool(re.search(r"скид|вычет|возврат|сумм|правил|действ", draft, re.I))

def _asks_non_tax_document_or_contract(client_message: str, *, context: Optional[Mapping[str, Any]] = None) -> bool:
    plan = _conversation_intent_plan(context)
    if str(plan.get("primary_intent") or "") == "tax":
        return False
    text = str(client_message or "").casefold().replace("ё", "е")
    if re.search(r"налог|вычет|фнс|ндфл|кнд|лиценз|справк", text, re.I):
        return False
    return bool(re.search(r"договор|оферт|оригинал|документ|акт|заявлен|подпис", text, re.I))

def _asks_non_matkap_document_or_contract(client_message: str, *, context: Optional[Mapping[str, Any]] = None) -> bool:
    plan = _conversation_intent_plan(context)
    if str(plan.get("primary_intent") or "") == "matkap":
        return False
    text = str(client_message or "").casefold().replace("ё", "е")
    if re.search(r"маткап|материнск|сфр|сертификат", text, re.I):
        return False
    return bool(re.search(r"договор|оферт|оригинал|документ|акт|заявлен|подпис", text, re.I))

def _answers_tax_deduction_scope(draft_text: str) -> bool:
    text = str(draft_text or "").casefold().replace("ё", "е")
    return bool(re.search(r"налог|вычет|фнс|ндфл|кнд|13\s*%|14\s*300|110\s*000", text, re.I))

def _answers_matkap_scope(draft_text: str) -> bool:
    text = str(draft_text or "").casefold().replace("ё", "е")
    return bool(re.search(r"маткап|материнск|сфр|сертификат", text, re.I))

def find_unsupported_numeric_promises(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    if _is_verified_safe_numeric_template(draft_text):
        return ()
    claims = _extract_numeric_promise_claims(draft_text)
    if not claims:
        return ()
    fact_texts = _fresh_fact_texts(context)
    return tuple(claim for claim in claims if not _claim_supported_by_facts(claim, fact_texts))

def _is_verified_safe_numeric_template(draft_text: str) -> bool:
    normalized = " ".join(str(draft_text or "").split())
    if not normalized:
        return False
    verified_templates = {
        FOTON_INSTALLMENT_SAFE_TEXT,
        FOTON_CAMP_INSTALLMENT_SAFE_TEXT,
        FOTON_DOLYAMI_SAFE_TEXT,
        FOTON_SECOND_SUBJECT_DISCOUNT_TEXT,
        UNPK_SECOND_SUBJECT_DISCOUNT_TEXT,
        UNPK_MONTHLY_SEMESTER_DISCOUNT_TEXT,
        UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT,
        MULTICHILD_DISCOUNT_TEXT,
        DISCOUNT_STACKING_SAFE_TEXT,
        ADMISSION_GUARANTEE_SAFE_TEXT,
        RESULT_GUARANTEE_SAFE_TEXT,
        MATKAP_FEDERAL_TIMING_SAFE_TEXT,
        TAX_AMOUNT_SAFE_TEXT,
        UNPK_LVSH_SEATS_SAFE_TEXT,
        FOTON_LVSH_PRICE_SAFE_TEXT,
        UNPK_LVSH_LIVING_TRANSFER_SAFE_TEXT,
        FOTON_CITY_CAMP_AUGUST_SAFE_TEXT,
        FOTON_LVSH_DATES_SAFE_TEXT,
        UNPK_LVSH_PRICE_SAFE_TEXT,
        UNPK_LVSH_PRICE_DETAILS_SAFE_TEXT,
        UNPK_LVSH_GRADE_11_PRICE_DETAILS_SAFE_TEXT,
        UNPK_CAMP_OVERVIEW_SAFE_TEXT,
        UNPK_CAMP_ONLINE_FORMAT_SAFE_TEXT,
        UNPK_LVSH_DATES_SAFE_TEXT,
    }
    return normalized in {" ".join(template.split()) for template in verified_templates}

def apply_subscription_policy_guards(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    route = result.route
    flags = list(result.safety_flags)
    manager_checklist = list(result.manager_checklist)
    metadata = dict(result.metadata)

    if result.topic_confidence < 0.70:
        route = "manager_only"
        flags.append("low_confidence_manager_only")
        manager_checklist.append("Модель не уверена в теме: проверить вручную.")
        metadata["forced_route_low_confidence"] = True

    if is_high_risk_result(result):
        route = "manager_only"
        flags.append("high_risk_manager_only")
        manager_checklist.append("Высокорисковая тема: не отправлять клиенту без ручной проверки.")
        metadata["forced_route_high_risk"] = True

    if result.message_type in {"non_question", "context_update", "wait_for_more", "manager_only"}:
        route = "manager_only"
        flags.append(f"message_type_{result.message_type}")
        metadata["forced_route_message_type"] = result.message_type

    if route == result.route and tuple(flags) == result.safety_flags and tuple(manager_checklist) == result.manager_checklist:
        return result
    return replace(
        result,
        route=route,
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(manager_checklist)),
        metadata=metadata,
    )

def _fix1b_has_paid_operation_context(result: SubscriptionDraftResult) -> bool:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    direct_p0 = metadata.get("direct_path_model_p0")
    if isinstance(direct_p0, Mapping) and str(direct_p0.get("p0_kind") or "") == "paid_operation_context":
        return True
    return "direct_path_model_p0_paid_operation_context" in set(result.safety_flags)


def apply_autonomy_matrix_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if result.route not in (*AUTONOMOUS_ROUTES, "draft_for_manager"):
        return result

    markers = set(detect_high_risk_input_markers(client_message, context=context))
    flags = list(result.safety_flags)
    checklist = list(result.manager_checklist)
    metadata = dict(result.metadata)
    original_route = result.route
    funnel = context.get("funnel_state") if isinstance(context, Mapping) and isinstance(context.get("funnel_state"), Mapping) else {}

    def demote(route: str, reason: str, checklist_item: str, *, draft_text: str | None = None) -> SubscriptionDraftResult:
        flags.append(reason)
        checklist.append(checklist_item)
        metadata[reason] = True
        return replace(
            result,
            route=route,
            draft_text=draft_text if draft_text else result.draft_text,
            safety_flags=tuple(dict.fromkeys(flags)),
            manager_checklist=tuple(dict.fromkeys(checklist)),
            metadata=metadata,
        )

    funnel_blocks_p0 = str(funnel.get("lead_stage") or "") == "p0_manager_only" or str(funnel.get("next_step_type") or "") == "manager_only_p0"
    if funnel_blocks_p0 and not (_p0_model_led_enabled(context) and not markers):
        flags.extend(("autonomy_blocked_funnel_p0", "high_risk_manager_only"))
        checklist.append("Автономный ответ запрещен: детерминированная воронка распознала P0/high-risk часть.")
        metadata["autonomy_blocked_funnel_p0"] = True
        return replace(
            result,
            route="manager_only",
            safety_flags=tuple(dict.fromkeys(flags)),
            manager_checklist=tuple(dict.fromkeys(checklist)),
            metadata=metadata,
        )

    direct_path = metadata.get("direct_path") if isinstance(metadata.get("direct_path"), Mapping) else {}
    complaint_topic_only = bool(
        result.topic_id == "theme:019b_negative_feedback"
        or str(direct_path.get("autonomy_topic") or "") == "theme:019b_negative_feedback"
        or str(direct_path.get("autonomy_topic_from") or "") == "theme:019b_negative_feedback"
    )
    high_risk_blocks_autonomy = bool(markers or is_high_risk_result(result))
    if (
        high_risk_blocks_autonomy
        and _p0_model_led_enabled(context)
        and not markers
        and complaint_topic_only
        and not _p0_model_led_complaint_backstop(client_message)
    ):
        high_risk_blocks_autonomy = False

    if high_risk_blocks_autonomy and not metadata.get("presale_refund_policy_manager_check"):
        flags.extend(("autonomy_blocked_high_risk", "high_risk_manager_only"))
        if _is_combined_high_risk_case(result, markers=markers, client_message=client_message, context=context):
            flags.append("combined_high_risk_manager_only")
            metadata["combined_high_risk_manager_only"] = True
        checklist.append("Автономный ответ запрещен: в сообщении есть P0/high-risk тема.")
        metadata["autonomy_blocked_high_risk"] = True
        return replace(
            result,
            route="manager_only",
            safety_flags=tuple(dict.fromkeys(flags)),
            manager_checklist=tuple(dict.fromkeys(checklist)),
            metadata=metadata,
        )

    if "asked_known_data_again" in result.safety_flags:
        return demote(
            "draft_for_manager",
            "autonomy_blocked_asked_known_data_again",
            "Автономный ответ запрещен: черновик повторно запросил уже известные данные клиента.",
        )
    repaired_address_text = _autonomy_scope_precision_repaired_address_text(result, client_message=client_message, context=context)
    if repaired_address_text:
        result = replace(result, draft_text=repaired_address_text)
        flags.append("autonomy_scope_precision_repaired_address")
        metadata["autonomy_scope_precision_repaired_address"] = True
    if result.message_type != "question":
        return demote(
            "draft_for_manager",
            "autonomy_default_cautious_message_type",
            "Автономный ответ запрещен: сообщение не является самостоятельным вопросом.",
        )
    if _active_brand(context) == "unknown":
        return demote(
            "draft_for_manager",
            "autonomy_default_cautious_unknown_brand",
            "Автономный ответ запрещен: активный бренд не определен.",
        )
    if not _autonomy_enabled(context):
        return demote(
            "draft_for_manager",
            "autonomy_default_cautious_no_policy",
            "Автономный ответ запрещен: нет явного разрешения матрицы автономности.",
        )
    if not _autonomy_topic_allowed(result.topic_id, context):
        return demote(
            "draft_for_manager",
            "autonomy_default_cautious_topic_not_allowed",
            "Автономный ответ запрещен: тема не входит в матрицу автономности.",
        )
    if _result_has_live_status_missing_fact(result, client_message=client_message, context=context) and not _is_verified_client_safe_template(result.draft_text):
        if reliable_answerer_step1_active_for_turn(client_message, context=context, result=result):
            preserved = preserve_partial_answer_for_live_status(
                result,
                reason="autonomy_default_cautious_live_status_missing",
                checklist_item="Автономный ответ запрещен: наличие места/группы/смены требует live-проверки менеджером.",
            )
            if preserved is not None:
                return preserved
        return demote(
            "draft_for_manager",
            "autonomy_default_cautious_live_status_missing",
            "Автономный ответ запрещен: наличие места/группы/смены требует live-проверки менеджером.",
            draft_text=_live_status_manager_check_text(client_message=client_message, context=context),
        )
    if _context_has_missing_fact_signal(context) and not _is_verified_client_safe_template(result.draft_text):
        return demote(
            "draft_for_manager",
            "autonomy_default_cautious_missing_facts",
            "Автономный ответ запрещен: есть недостающие факты.",
        )
    if not _has_client_safe_current_fact(context) and not _is_verified_client_safe_template(result.draft_text):
        return demote(
            "draft_for_manager",
            "autonomy_default_cautious_unverified_fact",
            "Автономный ответ запрещен: нет факта с флагами client-safe и актуальности.",
        )
    if "conversation_intent_plan_live_availability" in flags:
        return demote(
            "draft_for_manager",
            "autonomy_default_cautious_live_status_missing",
            "Автономный ответ запрещен: наличие места/группы/смены требует live-проверки менеджером.",
        )

    flags.append("autonomy_matrix_passed")
    metadata["autonomy_matrix_passed"] = True
    if original_route == "draft_for_manager":
        selected_fact_texts = _pipeline_fact_texts(result)
        supported_selected_answer = bool(
            selected_fact_texts
            and _claim_supported_by_facts(result.draft_text, tuple(selected_fact_texts.values()))
            and not _informational_yield_has_unbacked_concrete_anchors(
                result.draft_text,
                facts_blob=" ".join(selected_fact_texts.values()),
            )
        )
        verified_answer = (
            _is_verified_client_safe_template(result.draft_text)
            or _verified_informational_answer(result, client_message=client_message, context=context)
            or supported_selected_answer
        )
        if not verified_answer:
            flags.append("autonomy_matrix_kept_unverified_draft")
            checklist.append("Черновик не подтверждён выбранными фактами: оставить менеджеру без подстановки другого факта.")
            metadata["autonomy_matrix_kept_unverified_draft"] = True
            return replace(
                result,
                route="draft_for_manager",
                safety_flags=tuple(dict.fromkeys(flags)),
                manager_checklist=tuple(dict.fromkeys(checklist)),
                metadata=metadata,
            )
        flags.append("autonomy_matrix_promoted_safe_draft")
        checklist.append("Зелёная тема с проверенным клиентским фактом: можно отвечать автономно в пилотном режиме.")
        metadata["autonomy_matrix_promoted_safe_draft"] = True
    return replace(
        result,
        route="bot_answer_self_for_pilot" if original_route == "draft_for_manager" else result.route,
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(checklist)),
        metadata=metadata,
    )

def _is_verified_client_safe_template(draft_text: str) -> bool:
    normalized = " ".join(str(draft_text or "").split())
    if not normalized:
        return False
    verified_templates = {
        ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
        ADDRESS_UNPK_SAFE_TEXT,
        ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
        CONTACT_FOTON_SAFE_TEXT,
        CONTACT_UNPK_SAFE_TEXT,
        FOTON_INSTALLMENT_SAFE_TEXT,
        FOTON_CAMP_INSTALLMENT_SAFE_TEXT,
        FOTON_DOLYAMI_SAFE_TEXT,
        FOTON_SECOND_SUBJECT_DISCOUNT_TEXT,
        UNPK_SECOND_SUBJECT_DISCOUNT_TEXT,
        UNPK_MONTHLY_SEMESTER_DISCOUNT_TEXT,
        UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT,
        MULTICHILD_DISCOUNT_TEXT,
        DISCOUNT_STACKING_SAFE_TEXT,
        MATKAP_FEDERAL_TIMING_SAFE_TEXT,
        MATKAP_REGIONAL_SAFE_TEXT,
        MATKAP_SFR_REVIEW_SAFE_TEXT,
        TAX_AMOUNT_SAFE_TEXT,
        TAX_LICENSE_SAFE_TEXT,
        TAX_FNS_REVIEW_SAFE_TEXT,
        TAX_ONLINE_FORM_SAFE_TEXT,
        FOTON_LVSH_PRICE_SAFE_TEXT,
        UNPK_LVSH_PRICE_SAFE_TEXT,
        UNPK_LVSH_LIVING_TRANSFER_SAFE_TEXT,
        UNPK_LVSH_PRICE_DETAILS_SAFE_TEXT,
        UNPK_LVSH_GRADE_11_PRICE_DETAILS_SAFE_TEXT,
        UNPK_LVSH_GRADE_11_SAFE_TEXT,
        UNPK_CAMP_OVERVIEW_SAFE_TEXT,
        UNPK_CAMP_ONLINE_FORMAT_SAFE_TEXT,
        FOTON_CAMP_OVERVIEW_SAFE_TEXT,
        FOTON_LVSH_DATES_SAFE_TEXT,
        UNPK_LVSH_DATES_SAFE_TEXT,
        UNPK_LVSH_SEATS_SAFE_TEXT,
        FOTON_ONLINE_TRIAL_SAFE_TEXT,
        UNPK_TRIAL_SAFE_TEXT,
    }
    return normalized in {" ".join(template.split()) for template in verified_templates}

def _result_has_live_status_missing_fact(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> bool:
    client_text = str(client_message or "").casefold()
    if not _asks_live_status_or_booking_question(client_text):
        return False
    model_live = _intent_model_led_decision(result, context=context, target="live_availability")
    if model_live is False and not _asks_explicit_live_availability_question(client_text):
        return False
    missing_text = " ".join(str(item or "") for item in result.missing_facts).casefold()
    return has_any_marker(
        missing_text,
        (
            "availability",
            "налич",
            "мест",
            "групп",
            "смен",
            "брон",
        ),
    )

def _asks_live_status_or_booking_question(text: str) -> bool:
    normalized = str(text or "").casefold().replace("ё", "е")
    if has_any_marker(normalized, ("про оплат", "условия оплат", "не про мест", "не о мест", "не места")) and has_any_marker(
        normalized, ("оплат", "рассроч", "долями", "частями", "помесяч", "семестр")
    ):
        return False
    return has_any_marker(
        normalized,
        (
            "мест",
            "налич",
            "брон",
            "заброни",
            "оформить место",
            "проверки мест",
            "проверить места",
        )
    )


def _asks_explicit_live_availability_question(text: str) -> bool:
    normalized = str(text or "").casefold().replace("ё", "е")
    if not normalized or has_any_marker(normalized, ("не про мест", "не о мест", "не места")):
        return False
    patterns = (
        r"\bесть\s+ли\s+(?:свободн\w+\s+)?места?\b",
        r"\b(?:свободн\w+\s+)?места?\s+(?:есть|остал[оаи]сь|остались|будут|имеются)\b",
        r"\bостал[оаи]сь\s+(?:свободн\w+\s+)?места?\b",
        r"\bналичи[ея]\s+(?:свободн\w+\s+)?мест\b",
        r"\bсвободн\w+\s+места?\b",
        r"\bпровер(?:ить|ка|ки)\s+(?:свободн\w+\s+)?мест\b",
        r"\bзаброни(?:ровать|руем|ровать\s+)?\s+мест[оа]\b",
        r"\bоформить\s+мест[оа]\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _direct_path_model_intent_signal(result: SubscriptionDraftResult) -> Mapping[str, Any]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    signal = metadata.get("direct_path_model_intent")
    return signal if isinstance(signal, Mapping) else {}


def _direct_path_model_intent_primary(signal: Mapping[str, Any]) -> str:
    primary = str(signal.get("primary_intent") or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if primary in {"availability", "live_status", "seats", "seat_availability", "booking"}:
        primary = "live_availability"
    if primary in {"location", "venue", "place"}:
        primary = "address"
    if primary in {"price_lock", "current_terms", "fix_price"}:
        primary = "price_fix"
    if primary in {"out_of_scope", "offtopic", "not_related", "irrelevant"}:
        primary = "off_topic"
    if primary in {"general", "none", "unknown", "not_target"}:
        primary = "other"
    return primary if primary in INTENT_MODEL_LED_ALLOWED else ""


def _intent_model_led_decision(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
    target: str,
) -> Optional[bool]:
    if not _intent_model_led_enabled(context):
        return None
    primary = _direct_path_model_intent_primary(_direct_path_model_intent_signal(result))
    if not primary:
        return None
    if primary == "off_topic":
        return None
    if primary == target:
        return True
    if target in INTENT_MODEL_LED_TARGETS and primary in INTENT_MODEL_LED_ALLOWED:
        if _float_value(_direct_path_model_intent_signal(result).get("confidence")) < INTENT_MODEL_LED_CONFIDENCE_THRESHOLD:
            return None
        return False
    return None


def _intent_model_led_keyword_prefilter_intents(plan: Mapping[str, Any]) -> tuple[str, ...]:
    result: list[str] = []
    primary = str(plan.get("primary_intent") or "").strip()
    if primary in INTENT_MODEL_LED_TARGETS:
        result.append(primary)
    for key in ("keyword_signals", "topic_roles", "answer_topics"):
        value = plan.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
            continue
        for item in value:
            text = str(item or "").strip()
            if text in INTENT_MODEL_LED_TARGETS and text not in result:
                result.append(text)
    return tuple(result)


def _intent_model_led_inline_handoff_required(result: SubscriptionDraftResult) -> bool:
    direct_metadata = result.metadata.get("direct_path")
    frame = result.metadata.get("semantic_frame")
    if not isinstance(frame, Mapping) and isinstance(direct_metadata, Mapping):
        frame = direct_metadata.get("semantic_frame")
    return bool(
        isinstance(frame, Mapping)
        and str(frame.get("source") or "").strip().casefold() == "inline"
        and (
            _intent_actions_frame_bool(frame.get("must_handoff")) is True
            or str(frame.get("answerability") or "").strip().casefold() == "manager_only"
        )
    )


def _conversation_intent_plan_with_model_led(
    plan: Mapping[str, Any],
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
    client_message: str = "",
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not _intent_model_led_enabled(context):
        return plan, {}
    signal = _direct_path_model_intent_signal(result)
    model_intent = _direct_path_model_intent_primary(signal)
    if not model_intent:
        return plan, {}
    prefilter = _intent_model_led_keyword_prefilter_intents(plan)
    if not prefilter and _explicit_truthy_setting(
        context,
        INTENT_MODEL_LED_ENV,
        aliases=("intent_model_led", "intent_model_led_enabled", "model_intent_enabled"),
    ) is not True:
        return plan, {}
    original_intent = str(plan.get("primary_intent") or "").strip()
    if model_intent not in INTENT_MODEL_LED_TARGETS and original_intent not in INTENT_MODEL_LED_TARGETS:
        return plan, {}
    confidence = _float_value(signal.get("confidence"))
    trace_base = {
        "enabled": True,
        "original_primary_intent": original_intent,
        "model_primary_intent": model_intent,
        "keyword_prefilter": list(prefilter),
        "scope": str(signal.get("scope") or ""),
        "sense": str(signal.get("sense") or ""),
        "confidence": confidence,
        "reason": str(signal.get("reason") or ""),
    }
    if _intent_model_led_inline_handoff_required(result):
        return plan, {**trace_base, "applied": False, "skip_reason": "frame_must_handoff"}
    if model_intent == "off_topic":
        return plan, {**trace_base, "applied": False, "skip_reason": "off_topic_metadata_only"}
    if original_intent == "live_availability" and model_intent != "live_availability":
        direct_question = str(plan.get("direct_question") or "")
        if _asks_explicit_live_availability_question(" ".join((str(client_message or ""), direct_question))):
            return plan, {**trace_base, "applied": False, "skip_reason": "explicit_live_availability_floor"}
    if model_intent != original_intent and confidence < INTENT_MODEL_LED_CONFIDENCE_THRESHOLD:
        return plan, {**trace_base, "applied": False, "skip_reason": "low_confidence"}
    applied_intent = model_intent if model_intent in INTENT_MODEL_LED_TARGETS else "general_consultation"
    updated = dict(plan)
    updated["model_led_enabled"] = True
    updated["model_led_applied"] = True
    updated["model_led_original_primary_intent"] = original_intent
    updated["model_led_keyword_prefilter"] = list(prefilter)
    updated["model_led_primary_intent"] = model_intent
    updated["primary_intent"] = applied_intent
    if applied_intent in INTENT_MODEL_LED_TOPIC_MAP:
        updated["topic_id"] = INTENT_MODEL_LED_TOPIC_MAP[applied_intent]
    policy, route_bias = INTENT_MODEL_LED_ANSWER_POLICY.get(applied_intent, INTENT_MODEL_LED_ANSWER_POLICY["other"])
    updated["answer_policy"] = policy
    updated["route_bias"] = route_bias
    updated["model_led_signal"] = {
        "schema_version": str(signal.get("schema_version") or "direct_path_model_intent_v1_2026_06_25"),
        "primary_intent": model_intent,
        "scope": str(signal.get("scope") or ""),
        "sense": str(signal.get("sense") or ""),
        "confidence": confidence,
        "reason": str(signal.get("reason") or ""),
    }
    trace = {
        **trace_base,
        "applied": True,
        "applied_primary_intent": applied_intent,
    }
    return updated, trace

def _live_status_manager_check_text(*, client_message: str = "", context: Optional[Mapping[str, Any]] = None) -> str:
    known = {}
    if isinstance(context, Mapping):
        for key in ("known_slots", "known_dialog_fields"):
            value = context.get(key)
            if isinstance(value, Mapping):
                known.update({str(k): str(v) for k, v in value.items() if str(v or "").strip()})
        memory = context.get("dialogue_memory_view")
        if isinstance(memory, Mapping) and isinstance(memory.get("known_slots"), Mapping):
            known.update({str(k): str(v) for k, v in memory["known_slots"].items() if str(v or "").strip()})
    details = []
    if known.get("grade"):
        details.append(f"{known['grade']} класс")
    if known.get("subject"):
        details.append(str(known["subject"]))
    suffix = f" по вашему запросу ({', '.join(details)})" if details else ""
    text = str(client_message or "").casefold()
    if has_any_marker(text, ("как можно закреп", "как закреп", "как заброн", "как оформить место")):
        return (
            f"Сначала менеджер проверит наличие{suffix}. Если место есть, он подскажет оформление заявки и оплату; "
            "до проверки я не буду обещать, что место точно доступно."
        )
    if has_any_marker(
        text,
        (
            "что от меня нужно",
            "какие данные нужны",
            "что нужно для проверки",
            "что надо для проверки",
            "что нужно чтобы провер",
        )
    ):
        if details:
            camp_context = has_any_marker(
                " ".join(
                    [
                        text,
                        _dialog_context_haystack(context),
                        str(known.get("product") or known.get("known_course") or ""),
                    ]
                ),
                ("лагер", "лвш", "лш", "смен", "менделеево"),
            )
            optional_detail = (
                "Если есть предпочтение по датам смены, можно дописать его."
                if camp_context
                else "Если есть пожелания по расписанию или оплате, можно дописать их."
            )
            return (
                f"Для первичной проверки уже вижу: {', '.join(details)}. Повторно присылать это не нужно; "
                f"передам менеджеру, чтобы он проверил наличие. {optional_detail}"
            )
        return (
            "Для проверки мест нужны класс ребёнка, предмет или направление и желаемая смена/даты. "
            "После этого менеджер проверит наличие и подскажет следующий шаг."
        )
    if has_any_marker(text, ("лвш", "лагер", "смен", "менделеево")):
        return f"По местам не буду обещать без проверки{suffix}. Передам менеджеру, чтобы он проверил наличие по конкретной смене или группе."
    return f"По местам не буду обещать без проверки{suffix}. Передам менеджеру, чтобы он проверил наличие по конкретной группе."

def apply_payment_confirmation_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not _draft_confirms_payment(result):
        return result
    payment = _payment_context(context)
    amo_status = _payment_status(payment.get("amo_payment_status") or payment.get("amo_status"))
    tallanto_status = _payment_status(payment.get("tallanto_payment_status") or payment.get("tallanto_status"))
    conflict = _truthy_value(payment.get("payment_conflict") or payment.get("amo_tallanto_payment_conflict"))
    if conflict or (amo_status and tallanto_status and amo_status != tallanto_status):
        return _payment_guarded_result(result, reason="payment_source_conflict", checklist="Сверить AMO и Tallanto перед ответом по оплате.")
    if amo_status == "paid" and tallanto_status == "paid":
        return result
    return _payment_guarded_result(
        result,
        reason="payment_confirmation_without_two_sources",
        checklist="Проверить оплату в AMO и Tallanto перед подтверждением клиенту.",
    )

def apply_conversation_intent_plan_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    legacy_result = _apply_conversation_intent_plan_legacy_guard(
        result,
        client_message=client_message,
        context=context,
    )
    if reading_class_enabled(context, "live_status_read"):
        legacy_result = _live_status_read_transition_trace(
            result,
            legacy_result,
            context=context,
            stage="conversation_intent_plan",
            client_message=client_message,
        )
    if reading_class_enabled(context, "route_templates"):
        legacy_result = _route_templates_transition_trace(
            result,
            legacy_result,
            context=context,
            stage="autonomy_matrix",
            reason="conversation_intent_plan_legacy_shadow",
        )
    if not reading_class_enabled(context, "intent_actions"):
        return legacy_result
    return _apply_intent_actions_transition_guard(
        result,
        legacy_result,
        client_message=client_message,
        context=context,
    )


def apply_live_status_read_plan_trace(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not reading_class_enabled(context, "live_status_read"):
        return result
    legacy_result = _apply_conversation_intent_plan_legacy_guard(
        result,
        client_message=client_message,
        context=context,
    )
    return _live_status_read_transition_trace(
        result,
        legacy_result,
        context=context,
        stage="conversation_intent_plan_observer",
        trace_only=True,
        client_message=client_message,
    )


def _live_status_read_transition_trace(
    original: SubscriptionDraftResult,
    legacy_result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
    stage: str,
    trace_only: bool = False,
    client_message: str = "",
) -> SubscriptionDraftResult:
    if not trace_only and reading_apply_class_enabled(context, "live_status_read/conversation_intent_plan"):
        return _live_status_read_transition_apply(
            original,
            legacy_result,
            context=context,
            stage=stage,
            client_message=client_message,
        )
    reading = SemanticReading.from_result(original, context=context)
    frame = semantic_frame_from_metadata(original.metadata)
    plan = _conversation_intent_plan(context)
    legacy_live_status = "conversation_intent_plan_live_availability" in legacy_result.safety_flags
    frame_action = reading.requested_action if reading is not None else str(frame.get("requested_action") or "")
    frame_product = {}
    if isinstance(frame.get("requested_product"), Mapping):
        product = frame["requested_product"]  # type: ignore[index]
        frame_product = {
            key: str(product.get(key) or "")[:80]
            for key in ("grade", "subject", "format")
            if str(product.get(key) or "").strip()
        }
    changed_fields = _route_templates_changed_fields(original, legacy_result)
    decision = "legacy_live_status" if legacy_live_status else "legacy_not_live_status"
    conflicts: tuple[str, ...] = ()
    if legacy_live_status and frame_action not in {"check_availability", "enroll", "handoff_manager"}:
        conflicts = ("frame_requested_action",)
    elif not legacy_live_status and frame_action == "check_availability":
        conflicts = ("legacy_missing_live_status",)
    record = semantic_reading_trace_record(
        reading_class="live_status_read",
        enabled=True,
        status="shadow_only",
        decision=decision,
        reason="conversation_intent_plan_live_status_shadow",
        source=reading.source if reading is not None else str(frame.get("source") or ""),
        confidence=reading.frame_confidence if reading is not None else frame.get("confidence", 0.0),
        changed_fields=changed_fields,
        conflicts=conflicts,
        metadata={
            "stage": stage,
            "legacy_route": legacy_result.route,
            "original_route": original.route,
            "plan_primary_intent": str(plan.get("primary_intent") or ""),
            "frame_requested_action": frame_action,
            "frame_requested_product": frame_product,
        },
    )
    target = original if trace_only else legacy_result
    return replace(target, metadata=append_reading_trace_record(target.metadata, record))


def _live_status_frame_apply_fail_reason(reading: Optional[SemanticReading], frame: Mapping[str, Any]) -> str:
    if reading is None or not frame:
        return "no_frame"
    if reading.source != "inline":
        return "source_not_inline"
    if reading.frame_confidence < 0.90:
        return "low_confidence"
    risk_class = str(frame.get("risk_class") or "").strip().casefold()
    if risk_class in {
        "p0",
        "high_risk",
        "payment_dispute",
        "refund_claim",
        "refund",
        "legal",
        "legal_threat",
        "complaint",
        "money_dispute",
    }:
        return "risk_class_floor"
    if reading.requested_action not in {"check_availability", "enroll", "send_document"}:
        return "requested_action_not_live_status_relevant"
    return ""


def _live_status_hard_floor_reason(result: SubscriptionDraftResult) -> str:
    if result.route in {"blocked", "manager_only"}:
        return "hard_route_floor"
    if is_high_risk_result(result):
        return "p0_or_high_risk_floor"
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    direct_p0 = metadata.get("direct_path_model_p0")
    if isinstance(direct_p0, Mapping):
        direct_risk = str(direct_p0.get("risk_level") or "").strip().casefold()
        direct_kind = str(direct_p0.get("p0_kind") or direct_p0.get("p0_kind_raw") or "").strip().casefold()
        if bool(direct_p0.get("is_p0")) or direct_risk in {"high", "p0", "critical", "high_risk"}:
            return "p0_or_high_risk_floor"
        if direct_kind in {"refund", "payment_dispute", "complaint", "legal_threat", "contract_dispute", "cancellation_service_request"}:
            return "p0_or_high_risk_floor"
    flag_text = " ".join(str(flag or "") for flag in result.safety_flags).casefold()
    flags = {str(flag or "").strip() for flag in result.safety_flags}
    if any(
        marker in flag_text
        for marker in ("p0", "payment_dispute", "refund", "complaint", "legal", "high_risk", "manager_only_p0", "funnel_p0")
    ):
        return "p0_or_high_risk_floor"
    if _fix1b_has_paid_operation_context(result):
        return "paid_operation_floor"
    if flags & _SAFE_TEMPLATE_DISPATCHER_RECONSIDER_BLOCKING_FLAGS:
        return "brand_floor"
    if any(flag.startswith("payment_confirmation_") or flag == "payment_source_conflict" for flag in flags):
        return "payment_confirmation_floor"
    return ""


def _seats_default_open_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    explicit = _explicit_truthy_setting(
        context,
        SEATS_DEFAULT_OPEN_ENV,
        aliases=("seats_default_open", "seats_default_open_enabled"),
    )
    return bool(explicit) if explicit is not None else _pilot_profile_default_on_flag_enabled(context, SEATS_DEFAULT_OPEN_ENV)


def _seats_default_open_brand(value: Any) -> str:
    text = _normalize_fact_match_text(value)
    if has_any_marker(text, ("foton", "фотон", "цдпо", "cdpo")):
        return "foton"
    if has_any_marker(text, ("unpk", "унпк", "мфти", "mipt")):
        return "unpk"
    return ""


def _seats_default_open_exclusion_reason(
    *,
    frame: Mapping[str, Any],
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    requested_product = frame.get("requested_product") if isinstance(frame.get("requested_product"), Mapping) else {}
    active_brand = _seats_default_open_brand(_active_brand(context))
    product_brand = _seats_default_open_brand(requested_product.get("brand") if isinstance(requested_product, Mapping) else "")
    if active_brand not in {"foton", "unpk"}:
        return "brand_floor"
    if product_brand and product_brand != active_brand:
        return "brand_floor"
    product_text = " ".join(
        str(requested_product.get(key) or "")
        for key in ("brand", "subject", "grade", "format", "venue", "program_kind", "raw_text")
    )
    haystack = _normalize_fact_match_text(
        " ".join(
            [
                client_message,
                product_text,
                _dialog_context_haystack(context),
            ]
        )
    )
    if (
        re.search(r"\bсколько\s+(?:мест|человек)\b", haystack, re.I)
        or has_any_marker(haystack, ("размер группы", "наполняемость", "сколько учеников", "сколько детей"))
    ):
        return "group_size_question_floor"
    if has_any_marker(haystack, ("лвш", "лагер", "летн", "лш", "менделеево", "смен")):
        return "camp_or_shift_floor"
    if has_any_marker(haystack, ("индивидуал", "персональн", "репетитор", "один на один", "1 на 1")):
        return "individual_floor"
    if has_any_marker(haystack, ("лобня", "жуковск")):
        return "unsupported_city_floor"
    if has_any_marker(
        haystack,
        (
            "заброниру",
            "бронь",
            "зафиксиру",
            "закрепи",
            "оставьте место",
            "удержите место",
            "оформить место",
            "оформите место",
            "подтвердите место",
            "лист ожидания",
        ),
    ) or re.search(r"\bзапиш(?:ите|и|ем|ать)\s+(?:нас|меня|ребенка|ребенка|сына|дочь|его|ее|в\s+группу)\b", haystack, re.I):
        return "booking_operation_floor"
    return ""


def _seats_default_open_result(result: SubscriptionDraftResult, *, context: Optional[Mapping[str, Any]] = None) -> SubscriptionDraftResult:
    del context
    metadata = dict(result.metadata)
    direct = dict(metadata.get("direct_path") or {})
    metadata["seats_default_open_regular_groups"] = True
    metadata["availability_promise_allowlist"] = "seats_default_open_regular_groups"
    direct["seats_default_open_regular_groups"] = True
    metadata["direct_path"] = direct
    flags = [
        flag
        for flag in result.safety_flags
        if flag not in {*BASE_SAFETY_FLAGS, "conversation_intent_plan_live_availability", "semantic_frame_live_status_read_live_availability"}
    ]
    flags.append("seats_default_open_regular_groups")
    return replace(
        result,
        route="bot_answer_self_for_pilot",
        draft_text=SEATS_DEFAULT_OPEN_REGULAR_SAFE_TEXT,
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=(),
        forbidden_promises_detected=tuple(
            item for item in result.forbidden_promises_detected if item != "availability_promise"
        ),
        metadata=metadata,
    )


def _live_status_frame_guarded_result(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
    topic_id: str = "",
) -> SubscriptionDraftResult:
    route = result.route
    flags = list(result.safety_flags)
    checklist = list(result.manager_checklist)
    metadata = dict(result.metadata)
    if route in AUTONOMOUS_ROUTES:
        route = "draft_for_manager"
        flags.append("conversation_intent_plan_live_check_handoff")
        metadata["live_status_read_route_applied"] = "draft_for_manager"
    flags.extend(("conversation_intent_plan_live_availability", "semantic_frame_live_status_read_live_availability"))
    checklist.append(
        "SemanticFrame: вопрос про наличие/бронь/место требует проверки менеджером; не обещать место до проверки."
    )
    metadata["live_status_read_frame_applied"] = True
    return replace(
        result,
        topic_id=topic_id or result.topic_id,
        route=route,
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(checklist)),
        metadata=metadata,
    )


def _live_status_read_transition_apply(
    original: SubscriptionDraftResult,
    legacy_result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
    stage: str,
    client_message: str = "",
) -> SubscriptionDraftResult:
    reading = SemanticReading.from_result(original, context=context)
    frame = semantic_frame_from_metadata(original.metadata)
    plan = _conversation_intent_plan(context)
    frame_action = reading.requested_action if reading is not None else str(frame.get("requested_action") or "")
    legacy_live_status = "conversation_intent_plan_live_availability" in legacy_result.safety_flags
    fail_reason = (
        _live_status_hard_floor_reason(original)
        or _live_status_hard_floor_reason(legacy_result)
        or _live_status_frame_apply_fail_reason(reading, frame)
    )
    if fail_reason:
        chosen = legacy_result
        status = "fail_closed"
        decision = "legacy_more_conservative"
        reason = fail_reason
    elif frame_action == "check_availability":
        default_open_exclusion = (
            _seats_default_open_exclusion_reason(frame=frame, client_message=client_message, context=context)
            if _seats_default_open_enabled(context)
            else "seats_default_open_off"
        )
        if not default_open_exclusion:
            chosen = _seats_default_open_result(original, context=context)
            status = "applied"
            decision = "frame_check_availability_default_open"
            reason = "seats_default_open_regular_groups"
        else:
            chosen = legacy_result if legacy_live_status else _live_status_frame_guarded_result(
                original,
                context=context,
                topic_id=str(legacy_result.topic_id or ""),
            )
            status = "applied"
            decision = "frame_check_availability"
            reason = "frame_live_availability"
            if default_open_exclusion != "seats_default_open_off":
                reason = f"frame_live_availability:{default_open_exclusion}"
    else:
        chosen = original
        status = "applied"
        decision = "frame_not_live_status"
        reason = "frame_clears_legacy_live_status" if legacy_live_status else "frame_no_live_status"

    product = {}
    if isinstance(frame.get("requested_product"), Mapping):
        raw_product = frame["requested_product"]  # type: ignore[index]
        product = {
            key: str(raw_product.get(key) or "")[:80]
            for key in ("grade", "subject", "format")
            if str(raw_product.get(key) or "").strip()
        }
    conflicts: tuple[str, ...] = ()
    if frame_action == "check_availability" and not legacy_live_status:
        conflicts = ("legacy_missing_live_status",)
    elif legacy_live_status and frame_action in {"enroll", "send_document"}:
        conflicts = ("legacy_false_live_status",)
    record = semantic_reading_trace_record(
        reading_class="live_status_read",
        enabled=True,
        status=status,
        decision=decision,
        reason=reason,
        source=reading.source if reading is not None else str(frame.get("source") or ""),
        confidence=reading.frame_confidence if reading is not None else frame.get("confidence", 0.0),
        changed_fields=_route_templates_changed_fields(original, chosen),
        conflicts=conflicts,
        metadata={
            "stage": stage,
            "apply_enabled": True,
            "legacy_route": legacy_result.route,
            "original_route": original.route,
            "chosen_route": chosen.route,
            "plan_primary_intent": str(plan.get("primary_intent") or ""),
            "frame_requested_action": frame_action,
            "frame_requested_product": product,
            "legacy_live_status": legacy_live_status,
        },
    )
    return replace(chosen, metadata=append_reading_trace_record(chosen.metadata, record))


def _route_templates_transition_trace(
    original: SubscriptionDraftResult,
    legacy_result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
    stage: str,
    reason: str,
) -> SubscriptionDraftResult:
    if reading_apply_class_enabled(context, "route_templates/autonomy_matrix"):
        return _route_templates_transition_apply(
            original,
            legacy_result,
            context=context,
            stage=stage,
            reason=reason,
        )
    reading = SemanticReading.from_result(original, context=context)
    frame = semantic_frame_from_metadata(original.metadata)
    changed_fields: list[str] = []
    if legacy_result.route != original.route:
        changed_fields.append("route")
    if legacy_result.draft_text != original.draft_text:
        changed_fields.append("draft_text")
    if legacy_result.safety_flags != original.safety_flags:
        changed_fields.append("safety_flags")
    if legacy_result.manager_checklist != original.manager_checklist:
        changed_fields.append("manager_checklist")
    record = semantic_reading_trace_record(
        reading_class="route_templates",
        enabled=True,
        status="applied" if changed_fields else "shadow_only",
        decision="legacy_more_conservative",
        reason=reason,
        source=reading.source if reading is not None else str(frame.get("source") or ""),
        confidence=reading.frame_confidence if reading is not None else frame.get("confidence", 0.0),
        changed_fields=changed_fields,
        conflicts=("legacy_route",) if legacy_result.route != original.route else (),
        metadata=semantic_reading_transition_metadata(
            stage=stage,
            draft_before=original.draft_text,
            draft_after=legacy_result.draft_text,
            text_replacement=legacy_result.draft_text != original.draft_text,
            legacy_decision=legacy_result.route,
            frame_decision=reading.requested_action if reading is not None else str(frame.get("requested_action") or ""),
            chosen="legacy_more_conservative",
            extra={
                "legacy_route": legacy_result.route,
                "original_route": original.route,
                "primary_intent": reading.primary_intent if reading is not None else "",
            },
        ),
    )
    return replace(legacy_result, metadata=append_reading_trace_record(legacy_result.metadata, record))


def _route_templates_transition_apply(
    original: SubscriptionDraftResult,
    legacy_result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
    stage: str,
    reason: str,
) -> SubscriptionDraftResult:
    reading = SemanticReading.from_result(original, context=context)
    frame = semantic_frame_from_metadata(original.metadata)
    changed_fields = _route_templates_changed_fields(original, legacy_result)
    fail_reason = _route_templates_frame_apply_fail_reason(frame)
    chosen = legacy_result
    chosen_name = "legacy_more_conservative"
    status = "fail_closed" if fail_reason else "shadow_only"
    trace_reason = fail_reason or reason
    trace_changed_fields: Sequence[str] = changed_fields

    if not fail_reason:
        floor_reason = _route_templates_legacy_floor_reason(original, legacy_result)
        if floor_reason:
            status = "fail_closed"
            trace_reason = floor_reason
        elif not changed_fields:
            status = "shadow_only"
            trace_reason = reason
        elif legacy_result.draft_text != original.draft_text:
            status = "fail_closed"
            trace_reason = "legacy_text_replacement"
        else:
            chosen = original
            chosen_name = "frame_safe_original"
            status = "applied"
            trace_reason = reason

    record = semantic_reading_trace_record(
        reading_class="route_templates",
        enabled=True,
        status=status,
        decision=chosen_name,
        reason=trace_reason,
        source=reading.source if reading is not None else str(frame.get("source") or ""),
        confidence=reading.frame_confidence if reading is not None else frame.get("confidence", 0.0),
        changed_fields=trace_changed_fields,
        conflicts=("legacy_route",) if legacy_result.route != original.route else (),
        metadata=semantic_reading_transition_metadata(
            stage=stage,
            draft_before=original.draft_text,
            draft_after=chosen.draft_text,
            text_replacement=chosen.draft_text != original.draft_text,
            legacy_decision=legacy_result.route,
            frame_decision=reading.requested_action if reading is not None else str(frame.get("requested_action") or ""),
            chosen=chosen_name,
            extra={
                "apply_enabled": True,
                "legacy_route": legacy_result.route,
                "original_route": original.route,
                "primary_intent": reading.primary_intent if reading is not None else "",
                "answerability": str(frame.get("answerability") or ""),
                "must_handoff": _intent_actions_frame_bool(frame.get("must_handoff")) if frame else None,
                "risk_class": str(frame.get("risk_class") or ""),
            },
        ),
    )
    return replace(chosen, metadata=append_reading_trace_record(chosen.metadata, record))


def _route_templates_changed_fields(
    original: SubscriptionDraftResult,
    legacy_result: SubscriptionDraftResult,
) -> list[str]:
    changed_fields: list[str] = []
    if legacy_result.route != original.route:
        changed_fields.append("route")
    if legacy_result.topic_id != original.topic_id:
        changed_fields.append("topic_id")
    if legacy_result.draft_text != original.draft_text:
        changed_fields.append("draft_text")
    if legacy_result.safety_flags != original.safety_flags:
        changed_fields.append("safety_flags")
    if legacy_result.manager_checklist != original.manager_checklist:
        changed_fields.append("manager_checklist")
    return changed_fields


def _route_templates_frame_apply_fail_reason(frame: Mapping[str, Any]) -> str:
    if not frame:
        return "no_frame"
    if str(frame.get("source") or "").strip().casefold() != "inline":
        return "source_not_inline"
    try:
        confidence = float(frame.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.90:
        return "low_confidence"
    if str(frame.get("requested_action") or "").strip() != "answer_question":
        return "requested_action_not_answer_question"
    if str(frame.get("answerability") or "").strip() != "answer_self":
        return "answerability_not_answer_self"
    if _intent_actions_frame_bool(frame.get("must_handoff")) is not False:
        return "must_handoff_not_false"
    if str(frame.get("risk_class") or "").strip() != "safe":
        return "risk_class_not_safe"
    return ""


def _route_templates_legacy_floor_reason(
    original: SubscriptionDraftResult,
    legacy_result: SubscriptionDraftResult,
) -> str:
    if legacy_result.route in {"blocked", "manager_only"} or original.route in {"blocked", "manager_only"}:
        return "hard_route_floor"
    flag_text = " ".join(str(flag or "") for flag in (*original.safety_flags, *legacy_result.safety_flags)).casefold()
    flags = {str(flag or "").strip() for flag in (*original.safety_flags, *legacy_result.safety_flags)}
    if any(
        marker in flag_text
        for marker in ("p0", "payment_dispute", "refund", "complaint", "legal", "high_risk", "manager_only_p0", "funnel_p0")
    ):
        return "p0_or_high_risk_floor"
    if flags & _SAFE_TEMPLATE_DISPATCHER_RECONSIDER_BLOCKING_FLAGS:
        return "brand_floor"
    if any(flag.startswith("payment_confirmation_") or flag == "payment_source_conflict" for flag in flags):
        return "payment_confirmation_floor"
    if "conversation_intent_plan_live_availability" in legacy_result.safety_flags:
        return "live_availability_floor"
    if legacy_result.topic_id != original.topic_id:
        return "topic_id_floor"
    autonomy_cautious_false_positive = any(
        marker in flag_text
        for marker in (
            "autonomy_default_cautious_missing_facts",
            "autonomy_default_cautious_unverified_fact",
            "autonomy_default_cautious_topic_not_allowed",
        )
    )
    if flags.issuperset({"manager_approval_required", "no_auto_send"}) and not autonomy_cautious_false_positive:
        return "manual_approval_floor"
    return ""


def _apply_conversation_intent_plan_legacy_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    """Align draft topic/route with the context-level conversation plan.

    The plan is deliberately higher level than keyword rules: words such as
    "закрепить" or "бронь" may mean different things depending on the product
    focus. This guard uses the plan as an internal contract and never weakens
    P0, brand or fact-safety guards.
    """

    plan = _conversation_intent_plan(context)
    if not plan:
        return result
    plan, model_led_trace = _conversation_intent_plan_with_model_led(
        plan,
        result,
        context=context,
        client_message=client_message,
    )

    primary_intent = str(plan.get("primary_intent") or "").strip()
    plan_topic = str(plan.get("topic_id") or "").strip()
    answer_policy = str(plan.get("answer_policy") or "").strip()
    route_bias = str(plan.get("route_bias") or "").strip()
    route = result.route
    topic = str(result.topic_id or "").strip()
    draft_text = result.draft_text
    flags = list(result.safety_flags)
    checklist = list(result.manager_checklist)
    metadata = dict(result.metadata)

    valid_ids = load_valid_theme_and_service_ids()
    topic_from_plan = bool(plan_topic and plan_topic in valid_ids)
    semantic_non_p0 = _conversation_plan_semantic_non_p0(context, client_message=client_message)
    high_risk_plan = (
        primary_intent in {"refund", "legal_threat", "complaint", "payment_dispute"} or route_bias == "manager_only"
    ) and not semantic_non_p0

    metadata["conversation_intent_plan"] = _compact_conversation_intent_plan_for_metadata(plan)
    if model_led_trace:
        metadata["intent_model_led"] = dict(model_led_trace)

    if high_risk_plan:
        if topic_from_plan:
            topic = plan_topic
        route = "manager_only"
        flags.extend(("conversation_intent_plan_p0", "high_risk_manager_only"))
        checklist.append("План диалога распознал P0/high-risk тему: автономный ответ запрещён.")
        metadata["conversation_intent_plan_route_applied"] = "manager_only"

    elif (
        primary_intent != "live_availability"
        and topic_from_plan
        and topic != plan_topic
        and (not is_high_risk_result(result) or semantic_non_p0)
    ):
        original_high_risk = is_high_risk_result(result)
        topic = plan_topic
        flags.append("conversation_intent_plan_topic_applied")
        checklist.append(
            "Тема нормализована по плану смысла диалога: отдельные слова клиента использованы только как сигналы."
        )
        metadata["conversation_intent_plan_topic_from"] = result.topic_id
        if original_high_risk and semantic_non_p0:
            route = "draft_for_manager" if route == "manager_only" else route
            flags = _strip_false_p0_flags(flags)
            checklist.append("План смысла снял ложную P0-ветку: текущая реплика не содержит возврат, жалобу или юридическую угрозу.")
            metadata["conversation_intent_plan_false_p0_repaired"] = True

    if semantic_non_p0 and route == "manager_only" and is_high_risk_result(replace(result, route=route, safety_flags=tuple(flags))):
        route = "draft_for_manager"
        flags = _strip_false_p0_flags(flags)
        checklist.append("План смысла снял ложную P0-ветку: это предпродажный или справочный вопрос, а не спор.")
        metadata["conversation_intent_plan_false_p0_repaired"] = True

    if answer_policy == "answer_directly_if_fact_verified" and route_bias in AUTONOMOUS_ROUTES:
        flags.append("conversation_intent_plan_answer_first")
    if primary_intent:
        metadata["conversation_intent_primary_intent"] = primary_intent

    model_signal = _direct_path_model_intent_signal(result)
    if (
        _intent_model_led_enabled(context)
        and _direct_path_model_intent_primary(model_signal) == "off_topic"
        and (_float_value(model_signal.get("confidence")) or 0.0) >= INTENT_MODEL_LED_CONFIDENCE_THRESHOLD
        and not _intent_model_led_inline_handoff_required(result)
        and not high_risk_plan
        and not is_high_risk_result(result)
    ):
        draft_text = {
            "foton": OFF_TOPIC_FOTON_SAFE_TEXT,
            "unpk": OFF_TOPIC_UNPK_SAFE_TEXT,
        }.get(_active_brand(context), OFF_TOPIC_GENERIC_SAFE_TEXT)
        flags.append("intent_model_led_off_topic_safe_reply")
        metadata["intent_model_led"] = {
            **dict(metadata.get("intent_model_led") or {}),
            "enabled": True,
            "applied": True,
            "applied_primary_intent": "off_topic",
            "confidence": _float_value(model_signal.get("confidence")) or 0.0,
        }

    if (
        route == result.route
        and topic == result.topic_id
        and draft_text == result.draft_text
        and tuple(flags) == result.safety_flags
        and tuple(checklist) == result.manager_checklist
        and metadata == result.metadata
    ):
        return result

    return replace(
        result,
        topic_id=topic,
        route=route,
        draft_text=draft_text,
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(checklist)),
        metadata=metadata,
    )


def _intent_actions_route_rank(route: str) -> int:
    normalized = str(route or "").strip()
    if normalized in {"blocked", "manager_only"}:
        return 4
    if normalized == "draft_for_manager":
        return 3
    if normalized in AUTONOMOUS_ROUTES:
        return 1
    return 2


def _intent_actions_frame(result: SubscriptionDraftResult) -> Mapping[str, Any]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    frame = semantic_frame_from_metadata(metadata)
    return frame if isinstance(frame, Mapping) else {}


def _intent_actions_legacy_active_for_existing_pipeline(
    result: SubscriptionDraftResult,
    context: Optional[Mapping[str, Any]],
) -> bool:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    return _intent_model_led_enabled(context) and isinstance(metadata.get("direct_path_model_intent"), Mapping)


def _intent_actions_frame_fail_reason(frame: Mapping[str, Any]) -> str:
    if not frame:
        return "no_frame"
    if str(frame.get("source") or "").strip().casefold() != "inline":
        return "source_not_inline"
    requested_action = str(frame.get("requested_action") or "").strip()
    if requested_action not in INTENT_ACTIONS_FRAME_REQUESTED_ACTIONS:
        return "invalid_requested_action"
    try:
        confidence = float(frame.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < INTENT_ACTIONS_FRAME_CONFIDENCE_THRESHOLD:
        return "low_confidence"
    return ""


def _intent_actions_frame_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"1", "true", "yes", "y", "да", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "нет", "off"}:
            return False
    return None


def _intent_actions_trace(
    result: SubscriptionDraftResult,
    *,
    status: str,
    reason: str,
    frame: Mapping[str, Any],
    legacy_result: SubscriptionDraftResult,
    frame_result: Optional[SubscriptionDraftResult] = None,
    chosen: str = "legacy",
    changed_fields: Sequence[str] = (),
) -> SubscriptionDraftResult:
    metadata = {
        "legacy_route": legacy_result.route,
        "frame_route": frame_result.route if frame_result is not None else result.route,
        "chosen": chosen,
        "requested_action": str(frame.get("requested_action") or ""),
        "answerability": str(frame.get("answerability") or ""),
        "must_handoff": _intent_actions_frame_bool(frame.get("must_handoff")) if frame else None,
        "risk_class": str(frame.get("risk_class") or ""),
    }
    conflict_fields: tuple[str, ...] = ()
    if (frame_result is not None and frame_result.route != legacy_result.route) or result.route != legacy_result.route:
        conflict_fields = ("legacy_route",)
    record = semantic_reading_trace_record(
        reading_class="intent_actions",
        enabled=True,
        status=status,
        decision=chosen,
        reason=reason,
        source=str(frame.get("source") or "") if frame else "",
        confidence=frame.get("confidence", 0.0) if frame else 0.0,
        changed_fields=changed_fields,
        conflicts=conflict_fields,
        metadata=metadata,
    )
    return replace(result, metadata=append_reading_trace_record(result.metadata, record))


def _intent_actions_live_availability_result(
    result: SubscriptionDraftResult,
    *,
    frame: Mapping[str, Any],
) -> SubscriptionDraftResult:
    flags = list(result.safety_flags)
    checklist = list(result.manager_checklist)
    metadata = dict(result.metadata)
    route = result.route
    if route in AUTONOMOUS_ROUTES:
        route = "draft_for_manager"
        flags.append("semantic_frame_live_check_handoff")
        metadata["semantic_frame_intent_actions_route_applied"] = "draft_for_manager"
    flags.append("conversation_intent_plan_live_availability")
    flags.append("semantic_frame_intent_actions_live_availability")
    checklist.append(
        "SemanticFrame: вопрос про место/наличие/бронь требует проверки менеджером; не обещать место до проверки."
    )
    metadata["semantic_frame_intent_actions"] = {
        "requested_action": str(frame.get("requested_action") or ""),
        "confidence": frame.get("confidence", 0.0),
        "source": str(frame.get("source") or ""),
    }
    return replace(
        result,
        route=route,
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(checklist)),
        metadata=metadata,
    )


def _conversation_plan_live_availability_floor_result(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
) -> Optional[SubscriptionDraftResult]:
    plan = _conversation_intent_plan(context)
    internal_plan = _conversation_intent_plan_internal(context)
    primary_intent = str(plan.get("primary_intent") or "").strip()
    legacy_floor = _truthy_value(
        internal_plan.get("legacy_live_availability_floor_signal")
        or plan.get("legacy_live_availability_floor_signal")
    )
    if primary_intent != "live_availability" and not legacy_floor:
        return None
    return _intent_actions_live_availability_result(result, frame={})


def _apply_intent_actions_transition_guard(
    original: SubscriptionDraftResult,
    legacy_result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    frame = _intent_actions_frame(original)
    legacy_active = _intent_actions_legacy_active_for_existing_pipeline(original, context)
    previous_apply_active = reading_apply_class_enabled(context, "live_status_read/conversation_intent_plan")
    base_result = legacy_result if legacy_active or previous_apply_active else original
    base_name = "legacy" if legacy_active else ("previous_apply" if previous_apply_active else "original")
    fail_reason = _intent_actions_frame_fail_reason(frame)
    if fail_reason:
        plan_floor = _conversation_plan_live_availability_floor_result(base_result, context=context)
        if plan_floor is not None:
            return _intent_actions_trace(
                plan_floor,
                status="fail_closed",
                reason=fail_reason,
                frame=frame,
                legacy_result=legacy_result,
                frame_result=plan_floor,
                chosen="conversation_plan_live_availability_floor",
                changed_fields=_route_templates_changed_fields(base_result, plan_floor),
            )
        return _intent_actions_trace(
            base_result,
            status="fail_closed",
            reason=fail_reason,
            frame=frame,
            legacy_result=legacy_result,
            chosen=base_name,
        )

    requested_action = str(frame.get("requested_action") or "").strip()
    frame_result: Optional[SubscriptionDraftResult] = None
    reason = "no_matching_frame_action"

    if requested_action == "check_availability":
        if _truthy_value((base_result.metadata if isinstance(base_result.metadata, Mapping) else {}).get("seats_default_open_regular_groups")):
            frame_result = base_result
            reason = "seats_default_open_regular_groups"
        else:
            frame_result = _intent_actions_live_availability_result(base_result, frame=frame)
            reason = "frame_check_availability"

    if frame_result is None:
        return _intent_actions_trace(
            base_result,
            status="no_op",
            reason=reason,
            frame=frame,
            legacy_result=legacy_result,
            chosen=base_name,
        )

    if requested_action == "check_availability":
        chosen = frame_result
        chosen_name = "frame_check_availability"
    elif _intent_actions_route_rank(frame_result.route) > _intent_actions_route_rank(base_result.route):
        chosen = frame_result
        chosen_name = "frame_more_conservative"
    else:
        chosen = base_result
        chosen_name = base_name

    changed_fields: list[str] = []
    if chosen.route != base_result.route:
        changed_fields.append("route")
    if chosen.safety_flags != base_result.safety_flags:
        changed_fields.append("safety_flags")
    if chosen.manager_checklist != base_result.manager_checklist:
        changed_fields.append("manager_checklist")

    return _intent_actions_trace(
        chosen,
        status="applied" if changed_fields else "shadow_only",
        reason=reason,
        frame=frame,
        legacy_result=legacy_result,
        frame_result=frame_result,
        chosen=chosen_name,
        changed_fields=changed_fields,
    )


def _conversation_intent_plan(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    plan = context.get("conversation_intent_plan")
    return plan if isinstance(plan, Mapping) else {}

def _conversation_intent_plan_internal(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    plan = context.get("conversation_intent_plan_internal")
    return plan if isinstance(plan, Mapping) else {}










def _compact_conversation_intent_plan_for_metadata(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    keys = (
        "schema_version",
        "active_brand",
        "primary_intent",
        "topic_id",
        "direct_question",
        "topic_switch_decision",
        "product_family",
        "product_scope",
        "answer_policy",
        "route_bias",
        "required_fact_keys",
        "fact_scope",
        "blocked_neighbor_scopes",
        "topic_roles",
        "payment_method",
        "payment_source",
        "refund_frame",
        "enrollment_vs_recording",
        "transfer_sense",
        "answer_topics",
        "forbidden_pairs",
        "template_allowed",
        "next_step_hint",
        "model_led_enabled",
        "model_led_applied",
        "model_led_original_primary_intent",
        "model_led_keyword_prefilter",
        "model_led_primary_intent",
        "model_led_signal",
    )
    return {key: plan[key] for key in keys if key in plan}

def apply_known_context_redundant_question_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    """Catch drafts that ask again for data already known from safe context."""

    repeated = find_redundant_questions_for_known_context(result.draft_text, context=context)
    if not repeated:
        return result
    flags = tuple(
        dict.fromkeys(
            [
                *result.safety_flags,
                "asked_known_data_again",
                "human_tone_review_required",
            ]
        )
    )
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "Черновик просит данные, которые уже есть в контексте клиента: проверить и не отправлять как есть.",
            ]
        )
    )
    metadata = {
        **dict(result.metadata),
        "asked_known_data_again_fields": list(repeated),
    }
    route = "draft_for_manager" if result.route in AUTONOMOUS_ROUTES else result.route
    repair_text = _known_context_repair_text(result, client_message=client_message, context=context, repeated=repeated)
    guarded = replace(
        result,
        route=route,
        draft_text=repair_text,
        safety_flags=flags,
        manager_checklist=checklist,
        context_warnings=tuple(dict.fromkeys([*result.context_warnings, "asked_known_data_again"])),
        metadata=metadata,
    )
    if not reading_class_enabled(context, "route_templates"):
        return guarded
    reading = SemanticReading.from_result(result, context=context)
    record = semantic_reading_trace_record(
        reading_class="route_templates",
        enabled=True,
        status="applied",
        decision="legacy_more_conservative",
        reason="known_context_redundant_question_guard",
        source=reading.source if reading is not None else "",
        confidence=reading.frame_confidence if reading is not None else 0.0,
        changed_fields=("route", "draft_text"),
        conflicts=("reask_known_slots",),
        metadata=semantic_reading_transition_metadata(
            stage="redundant_guard",
            draft_before=result.draft_text,
            draft_after=guarded.draft_text,
            text_replacement=True,
            legacy_decision="repair_reask_known_slots",
            frame_decision=reading.requested_action if reading is not None else "",
            chosen="legacy_more_conservative",
            extra={"repeated_fields": list(repeated)},
        ),
    )
    return replace(guarded, metadata=append_reading_trace_record(guarded.metadata, record))


def apply_reask_read_trace(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not reading_class_enabled(context, "reask_read"):
        return result
    repeated = find_redundant_questions_for_known_context(result.draft_text, context=context)
    apply_enabled = reading_apply_class_enabled(context, "reask_read/final_text")
    guarded = result
    changed_fields: list[str] = []
    if repeated and apply_enabled:
        guarded = apply_known_context_redundant_question_guard(result, client_message=client_message, context=context)
        if guarded.route != result.route:
            changed_fields.append("route")
        if guarded.draft_text != result.draft_text:
            changed_fields.append("draft_text")
        if guarded.safety_flags != result.safety_flags:
            changed_fields.append("safety_flags")
        if guarded.manager_checklist != result.manager_checklist:
            changed_fields.append("manager_checklist")
    known = known_context_fields(context)
    hidden_slots = _semantic_hidden_slot_names(context)
    do_not_reask = _do_not_reask_slot_names_from_context(context)
    record = semantic_reading_trace_record(
        reading_class="reask_read",
        enabled=True,
        status="applied" if changed_fields else ("would_flag" if repeated else "shadow_only"),
        decision="known_slot_reask_applied" if changed_fields else ("known_slot_reask" if repeated else "no_reask_detected"),
        reason="direct_path_final_text_reask_observer",
        source="deterministic_observer",
        confidence=1.0,
        changed_fields=tuple(changed_fields),
        conflicts=("known_slot_reask",) if repeated else (),
        metadata={
            "stage": "direct_path_final_text",
            "apply_enabled": apply_enabled,
            "repeated_slot_keys": list(repeated),
            "known_slot_keys": sorted(key for key in known if key in {"grade", "subject", "format", "active_brand"}),
            "do_not_reask_slots": sorted(do_not_reask),
            "semantic_hidden_slot_names": sorted(hidden_slots),
            "hidden_slots_are_client_confirmed": False,
        },
    )
    return replace(guarded, metadata=append_reading_trace_record(guarded.metadata, record))


def apply_roles_read_trace(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not reading_class_enabled(context, "roles_read"):
        return result
    plan = _conversation_intent_plan(context)
    frame = semantic_frame_from_metadata(result.metadata)
    apply_enabled = reading_apply_class_enabled(context, "roles_read/refund_tax")
    guarded = result
    changed_fields: list[str] = []
    if apply_enabled and _roles_read_tax_non_refund_plan(plan) and _roles_read_refund_false_positive_result(result):
        guarded = _roles_read_tax_non_refund_result(result)
        if guarded.route != result.route:
            changed_fields.append("route")
        if guarded.topic_id != result.topic_id:
            changed_fields.append("topic_id")
        if guarded.draft_text != result.draft_text:
            changed_fields.append("draft_text")
        if guarded.safety_flags != result.safety_flags:
            changed_fields.append("safety_flags")
        if guarded.manager_checklist != result.manager_checklist:
            changed_fields.append("manager_checklist")
    record = semantic_reading_trace_record(
        reading_class="roles_read",
        enabled=True,
        status="applied" if changed_fields else "shadow_only",
        decision="tax_non_refund_template" if changed_fields else "roles_observed",
        reason="direct_path_final_roles_observer",
        source=str(frame.get("source") or "context"),
        confidence=frame.get("confidence", 0.0) if frame else 0.0,
        changed_fields=tuple(changed_fields),
        conflicts=("tax_vs_refund",) if changed_fields else (),
        metadata={
            "stage": "direct_path_final_roles",
            "apply_enabled": apply_enabled,
            "final_route": result.route,
            "final_topic_id": result.topic_id,
            "plan_primary_intent": str(plan.get("primary_intent") or ""),
            "plan_topic_id": str(plan.get("topic_id") or ""),
            "payment_source": str(plan.get("payment_source") or ""),
            "refund_frame": str(plan.get("refund_frame") or ""),
            "enrollment_vs_recording": str(plan.get("enrollment_vs_recording") or ""),
            "transfer_sense": str(plan.get("transfer_sense") or ""),
            "frame_requested_action": str(frame.get("requested_action") or ""),
            "frame_payment_readiness": str(frame.get("payment_readiness") or ""),
            "frame_risk_class": str(frame.get("risk_class") or ""),
        },
    )
    return replace(guarded, metadata=append_reading_trace_record(guarded.metadata, record))


def _roles_read_tax_non_refund_plan(plan: Mapping[str, Any]) -> bool:
    return (
        str(plan.get("primary_intent") or "").strip() == "tax"
        and str(plan.get("payment_source") or "").strip() == "tax_deduction"
        and str(plan.get("refund_frame") or "") == "none"
    )


def _roles_read_refund_false_positive_result(result: SubscriptionDraftResult) -> bool:
    return (
        result.topic_id == "theme:009_refund"
        or bool(_roles_read_refund_related_safety_flags(result.safety_flags))
    )


def _roles_read_refund_related_safety_flags(flags: Sequence[str]) -> tuple[str, ...]:
    refund_related_flags = {
        "zero_collect_refund_guarded",
        "presale_refund_policy_manager_check",
        "presale_refund_policy_non_p0",
    }
    return tuple(flag for flag in flags if str(flag or "") in refund_related_flags)


def _roles_read_tax_non_refund_result(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    # The frame owns the customer-facing meaning, but legacy/P0 floors still own
    # route hardening. A false "refund" text becomes a tax text; manager_only
    # stays manager_only when an upstream safety floor already required review.
    flags = tuple(dict.fromkeys([*result.safety_flags, "tax_deduction_safe_template_applied"]))
    return replace(
        result,
        topic_id="theme:008_tax_deduction",
        draft_text=TAX_DEDUCTION_PROCESS_SAFE_TEXT,
        safety_flags=flags,
        metadata={**dict(result.metadata), "roles_read_tax_non_refund_repaired": True},
    )


def _semantic_hidden_slot_names(context: Optional[Mapping[str, Any]]) -> set[str]:
    if not isinstance(context, Mapping):
        return set()
    result: set[str] = set()
    for container in (context, context.get("dialogue_memory_view")):
        if not isinstance(container, Mapping):
            continue
        slots = container.get("semantic_reading_slots")
        if isinstance(slots, Mapping):
            result.update(str(key or "").strip() for key in slots if str(key or "").strip())
    return result


def _do_not_reask_slot_names_from_context(context: Optional[Mapping[str, Any]]) -> set[str]:
    if not isinstance(context, Mapping):
        return set()
    result: set[str] = set()
    for container in (
        context,
        context.get("conversation_intent_plan"),
        context.get("planner_intent"),
        context.get("answer_contract"),
        context.get("dialogue_memory_view"),
    ):
        if not isinstance(container, Mapping):
            continue
        raw = container.get("do_not_reask_slots") or container.get("do_not_ask_again")
        if isinstance(raw, str):
            value = raw.strip()
            if value:
                result.add(value)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            result.update(str(item or "").strip() for item in raw if str(item or "").strip())
    return result


def _known_context_repair_text(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
    repeated: Sequence[str] = (),
) -> str:
    """Replace a repeated-data question with a useful answer that keeps known context."""

    known = known_context_fields(context)
    grade = str(known.get("grade") or "").strip()
    subject = str(known.get("subject") or "").strip()
    context_bits = []
    if grade:
        context_bits.append(f"{grade} класс")
    if subject:
        context_bits.append(subject)
    prefix = f"Поняла: {', '.join(context_bits)}. " if context_bits else "Поняла, продолжу с учётом уже сказанного. "

    draft = " ".join(str(result.draft_text or "").split())
    cleaned_draft = _remove_repeated_known_data_questions(draft, repeated=repeated)
    if cleaned_draft and len(cleaned_draft) >= 90:
        return cleaned_draft

    topic = str(result.topic_id or "")
    if topic in {"theme:026_camp_general", "theme:027_camp_living_conditions", "theme:028_transport_logistics"}:
        active_brand = _active_brand(context)
        text = prefix
        current = str(client_message or "").casefold().replace("ё", "е")
        if active_brand == "foton":
            if "онлайн" in current:
                return (
                    text
                    + "По онлайн-формату летней смены нужно проверить актуальную возможность. "
                    "Из подтверждённого у Фотона есть выездная школа в Менделеево и городская летняя школа в Москве; "
                    "менеджер подберёт вариант под ваш класс и цель."
                )
            return (
                text
                + "У Фотона есть выездная школа в Менделеево и городская летняя школа в Москве. "
                "Менеджер проверит подходящую смену и наличие мест под ваш класс."
            )
        if active_brand == "unpk":
            return (
                text
                + "По УНПК есть летние смены и ЛВШ Менделеево; менеджер проверит подходящую смену "
                "и наличие мест под ваш класс."
            )

    if draft and len(draft) >= 90 and not any(field in repeated for field in ("grade", "subject", "student_name", "parent_name")):
        return draft
    return KNOWN_CONTEXT_REPAIR_TEXT

def _remove_repeated_known_data_questions(text: str, *, repeated: Sequence[str]) -> str:
    value = str(text or "").strip()
    if not value or not repeated:
        return value
    sentence_parts = re.split(r"(?<=[.!?])\s+", value)
    cleaned: list[str] = []
    for sentence in sentence_parts:
        lowered = sentence.casefold().replace("ё", "е")
        drop = False
        if "grade" in repeated and re.search(r"(напишите|подскажите|уточните)[^.!?\n]{0,70}класс", lowered):
            drop = True
        if "subject" in repeated and re.search(r"(напишите|подскажите|уточните)[^.!?\n]{0,70}предмет", lowered):
            drop = True
        if "student_name" in repeated and re.search(r"(напишите|подскажите|уточните)[^.!?\n]{0,70}(имя|фио)", lowered):
            drop = True
        if "parent_name" in repeated and re.search(r"(ваше\s+имя|как\s+вас\s+зовут|фио\s+родител)", lowered):
            drop = True
        if "phone" in repeated and re.search(r"(телефон|номер\s+телефона|контактн\w+\s+номер)", lowered):
            drop = True
        if not drop:
            cleaned.append(sentence)
    result = " ".join(part.strip() for part in cleaned if part.strip())
    return result or value

def find_redundant_questions_for_known_context(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    known = known_context_fields(context)
    if not known:
        return ()
    text = str(draft_text or "").casefold().replace("ё", "е")
    repeated: list[str] = []
    if (
        known.get("student_name")
        and re.search(r"(фио|имя|как\s+зовут)[^.!?\n]{0,80}(реб[её]нк|ученик)", text)
        and not _legitimate_enrollment_student_name_request(text, context=context)
    ):
        repeated.append("student_name")
    if known.get("parent_name") and re.search(r"(ваше\s+имя|как\s+вас\s+зовут|фио\s+родител)", text):
        repeated.append("parent_name")
    if known.get("phone") and re.search(r"(телефон|номер\s+телефона|контактн\w+\s+номер)", text):
        repeated.append("phone")
    if known.get("grade") and re.search(r"(какой\s+класс|класс\s+реб[её]нк|напишите[^.!?\n]{0,40}класс|подскажите[^.!?\n]{0,40}класс)", text):
        repeated.append("grade")
    if known.get("subject") and re.search(r"(какой\s+предмет|предмет[^.!?\n]{0,30}интерес|напишите[^.!?\n]{0,40}предмет|подскажите[^.!?\n]{0,40}предмет)", text):
        repeated.append("subject")
    if known.get("format") and re.search(r"(онлайн\s+или\s+очн|очно\s+или\s+онлайн|какой\s+формат|формат[^.!?\n]{0,30}удоб)", text):
        repeated.append("format")
    if known.get("active_brand") and re.search(r"(фотон\s+или\s+унпк|какой\s+центр|какой\s+учебн\w+\s+центр)", text):
        repeated.append("active_brand")
    return tuple(dict.fromkeys(repeated))


def _legitimate_enrollment_student_name_request(text: str, *, context: Optional[Mapping[str, Any]]) -> bool:
    plan = _conversation_intent_plan(context)
    primary_intent = str(plan.get("primary_intent") or "").strip()
    topic_id = str(plan.get("topic_id") or "").strip()
    enrollment_vs_recording = str(plan.get("enrollment_vs_recording") or "").strip()
    if enrollment_vs_recording == "recording":
        return False
    if enrollment_vs_recording == "enroll" or primary_intent in {"enroll", "enrollment"} or topic_id == "theme:020_enrollment":
        return True
    return False

def apply_unstated_subject_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    subject_context = _context_with_dialogue_contract_retrieved_facts(context, result)
    allowed = _allowed_subjects_from_context(subject_context, client_message=client_message)
    unexpected = sorted(_mentioned_subjects(result.draft_text) - allowed)
    if not unexpected:
        return result
    safe_text = _unstated_subject_safe_text(subject_context, unexpected=unexpected)
    return replace(
        result,
        draft_text=safe_text,
        route="draft_for_manager" if result.route != "manager_only" else result.route,
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "unstated_subject_guarded", "manager_approval_required", "no_auto_send"])),
        manager_checklist=tuple(
            dict.fromkeys(
                [
                    *result.manager_checklist,
                    "Черновик добавил предмет/направление, которого клиент не называл: проверить и убрать перед отправкой.",
                ]
            )
        ),
        metadata={**dict(result.metadata), "unstated_subjects": unexpected},
    )

def _unstated_subject_safe_text(context: Optional[Mapping[str, Any]], *, unexpected: Sequence[str]) -> str:
    known = known_context_fields(context)
    grade = str(known.get("grade") or "").strip()
    details = f"{grade} класс" if grade else "класс ребёнка"
    product = str(known.get("product") or "").casefold()
    if "лш" in product or "лагер" in product or "летн" in product:
        return (
            f"Вижу {details}. Не буду подставлять предмет или направление, которое вы не называли. "
            "По летней программе менеджер проверит подходящую смену, уровень и наличие мест под ваш класс."
        )
    return (
        "Не буду подставлять предмет или направление, которое вы не называли. "
        "Если напишете предмет и класс, сориентирую по подходящему курсу и следующему шагу."
    )

SUBJECT_GUARD_MARKERS: Mapping[str, tuple[str, ...]] = {
    "математика": ("математ",),
    "физика": ("физик",),
    "информатика": ("информат",),
    "программирование": ("программирован",),
    "русский язык": ("русск",),
    "английский язык": ("англий",),
    "химия": ("хим",),
    "биология": ("биолог",),
}

def _allowed_subjects_from_context(context: Optional[Mapping[str, Any]], *, client_message: str = "") -> set[str]:
    allowed = _mentioned_subjects(client_message)
    known = known_context_fields(context)
    for value in (known.get("subject"),):
        allowed.update(_mentioned_subjects(value))
    if isinstance(context, Mapping):
        memory = context.get("dialogue_memory_view")
        if isinstance(memory, Mapping):
            for key in ("client_confirmed_slots", "crm_known_slots"):
                slots = memory.get(key)
                if isinstance(slots, Mapping):
                    allowed.update(_mentioned_subjects(slots.get("subject")))
        allowed.update(_subjects_from_retrieved_facts(context))
    return allowed

def _subjects_from_retrieved_facts(context: Mapping[str, Any]) -> set[str]:
    active_brand = _active_brand(context)
    if active_brand == "unknown":
        return set()
    pipeline = context.get("dialogue_contract_pipeline") if isinstance(context.get("dialogue_contract_pipeline"), Mapping) else {}
    retrieved = pipeline.get("retrieved_facts") if isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
    subjects: set[str] = set()
    for key, fact_text in retrieved.items():
        combined = f"{key} {fact_text}"
        if not _retrieved_fact_matches_active_brand(combined, active_brand):
            continue
        subjects.update(_mentioned_subjects(fact_text))
    return subjects

def _retrieved_fact_matches_active_brand(text: object, active_brand: str) -> bool:
    low = str(text or "").casefold().replace("ё", "е")
    has_foton = bool(re.search(r"\b(?:foton|фотон)\b|cdpofoton|цдпо|црдо", low, re.I))
    has_unpk = bool(re.search(r"\b(?:unpk|унпк)\b|kmipt", low, re.I))
    if active_brand == "foton" and has_unpk:
        return False
    if active_brand == "unpk" and has_foton:
        return False
    return True

def _mentioned_subjects(text: object) -> set[str]:
    value = str(text or "").casefold().replace("ё", "е")
    return {
        subject
        for subject, markers in SUBJECT_GUARD_MARKERS.items()
        if has_any_marker(value, markers)
    }

def known_context_fields(context: Optional[Mapping[str, Any]]) -> dict[str, str]:
    if not isinstance(context, Mapping):
        return {}
    result: dict[str, str] = {}
    for container_key in ("known_client_fields", "known_dialog_fields", "client_identity"):
        value = context.get(container_key)
        if isinstance(value, Mapping):
            _merge_known_context_fields(result, value)
    known_slots = context.get("known_slots")
    if isinstance(known_slots, Mapping):
        _merge_known_context_fields(result, known_slots)
    memory = context.get("dialogue_memory_view")
    if isinstance(memory, Mapping):
        memory_slots = memory.get("known_slots")
        if isinstance(memory_slots, Mapping):
            _merge_known_context_fields(result, memory_slots)
    plan = context.get("conversation_intent_plan")
    if isinstance(plan, Mapping):
        plan_slots = plan.get("known_slots")
        if isinstance(plan_slots, Mapping):
            _merge_known_context_fields(result, plan_slots, overwrite=True)
    funnel = context.get("funnel_state")
    if isinstance(funnel, Mapping):
        filled = funnel.get("filled_slots")
        if isinstance(filled, Mapping):
            _merge_known_context_fields(result, filled)
        slots = funnel.get("known_slots")
        if isinstance(slots, Mapping):
            _merge_known_context_fields(result, slots)
    active_brand = _active_brand(context)
    if active_brand != "unknown":
        result.setdefault("active_brand", active_brand)
    for summary_key in ("customer_context_summary", "known_context_summary"):
        summary = str(context.get(summary_key) or "")
        if summary:
            _merge_known_context_fields(result, _known_fields_from_text(summary))
    return {key: value for key, value in result.items() if str(value or "").strip()}

def _merge_known_context_fields(target: dict[str, str], source: Mapping[str, Any], *, overwrite: bool = False) -> None:
    aliases = {
        "parent_name": ("parent_name", "parent", "parent_full_name", "fio_parent", "parent_fio"),
        "student_name": ("student_name", "student", "student_full_name", "fio_student", "student_fio", "child_name"),
        "phone": ("phone", "normalized_phone", "client_phone"),
        "grade": ("grade", "class", "student_grade", "klass"),
        "subject": ("subject", "course_subject", "interest_subject"),
        "format": ("format", "course_format", "preferred_format"),
        "product": ("product", "program", "interest_product"),
        "active_brand": ("active_brand", "brand"),
        "known_course": ("known_course", "current_course", "course"),
        "current_group": ("current_group", "group", "tallanto_group"),
    }
    for normalized, keys in aliases.items():
        for key in keys:
            value = str(source.get(key) or "").strip()
            if value:
                if overwrite:
                    target[normalized] = value[:160]
                else:
                    target.setdefault(normalized, value[:160])
                break

def _known_fields_from_text(text: str) -> Mapping[str, str]:
    value = str(text or "")
    result: dict[str, str] = {}
    grade = re.search(r"\b(?P<grade>[1-9]|1[01])\s*(?:класс|кл\.?)\b", value, re.I)
    if grade:
        result["grade"] = grade.group("grade")
    subjects = []
    lowered = value.casefold().replace("ё", "е")
    for marker, canonical in (
        ("математ", "математика"),
        ("физик", "физика"),
        ("информат", "информатика"),
        ("программирован", "программирование"),
        ("русск", "русский язык"),
        ("англий", "английский язык"),
        ("хими", "химия"),
        ("биолог", "биология"),
    ):
        if marker in lowered:
            subjects.append(canonical)
    if subjects:
        result["subject"] = ", ".join(dict.fromkeys(subjects))
    return result





def apply_taxonomy_topic_guard(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    valid_ids = load_valid_theme_and_service_ids()
    topic_id = str(result.topic_id or "").strip()
    valid_alternatives = tuple(item for item in result.alternative_themes if item in valid_ids)
    invalid_alternatives = tuple(item for item in result.alternative_themes if item and item not in valid_ids)
    if topic_id in valid_ids and valid_alternatives == result.alternative_themes:
        return result

    flags = list(result.safety_flags)
    checklist = list(result.manager_checklist)
    metadata = dict(result.metadata)

    if topic_id not in valid_ids:
        flags.append("invalid_topic_id_normalized")
        checklist.append("LLM вернула тему не из утвержденного списка: проверить вручную.")
        metadata["original_invalid_topic_id"] = topic_id
        topic_id = UNKNOWN_TOPIC_FALLBACK_ID
    if invalid_alternatives:
        flags.append("invalid_alternative_themes_removed")
        metadata["invalid_alternative_themes"] = list(invalid_alternatives)

    return replace(
        result,
        topic_id=topic_id,
        alternative_themes=valid_alternatives,
        route="manager_only",
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(checklist)),
        metadata=metadata,
    )

def is_high_risk_result(result: SubscriptionDraftResult) -> bool:
    topic = result.topic_id.strip()
    if topic in HIGH_RISK_THEME_IDS:
        return True
    haystack = " ".join(
        [
            topic,
            result.broad_group,
            result.risk_level,
            *result.alternative_themes,
            *result.safety_flags,
            *result.context_warnings,
        ]
    ).casefold()
    return any(marker.casefold() in haystack for marker in HIGH_RISK_MARKERS)

def detect_high_risk_input_markers(client_message: str, *, context: Optional[Mapping[str, Any]] = None) -> tuple[str, ...]:
    decision = classify_answer_safety(
        client_message=client_message,
        context=context,
        topic_id="",
        route="",
        safety_flags=(),
    )
    codes = tuple(code for code in decision.risk_codes if code in HARD_P0_CODES)
    return _p0_model_led_filter_high_risk_codes(codes, client_message=client_message, context=context)

def _conversation_plan_semantic_non_p0(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
) -> bool:
    return classify_answer_safety(client_message=client_message, context=context).semantic_non_p0

def _strip_false_p0_flags(flags: Sequence[str]) -> list[str]:
    p0_markers = (
        "conversation_intent_plan_p0",
        "high_risk_manager_only",
        "legal_threat_topic_overrode_refund",
        "zero_collect_legal_guarded",
        "zero_collect_refund_guarded",
        "complaint_apology_guarded",
        "high_risk_input_manager_only",
        "autonomy_blocked_high_risk",
    )
    return [flag for flag in flags if not any(marker in str(flag or "") for marker in p0_markers)]






def _is_combined_high_risk_case(
    result: SubscriptionDraftResult,
    *,
    markers: set[str],
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> bool:
    if not (
        markers & {"refund", "legal", "complaint", "reputation_threat"}
        or result.topic_id in {"theme:009_refund", "theme:019b_negative_feedback", "theme:029_legal_question"}
    ):
        return False
    haystack = _semantic_haystack(result, client_message=client_message, context=context)
    return bool(COMBINED_NON_RISK_INPUT_RE.search(haystack))














def _scope_guard_has_missing_intent_fact(
    result: SubscriptionDraftResult,
    context: Optional[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> bool:
    if not _has_missing_fact_signal(result, context):
        return False
    required = _scope_guard_required_fact_keys(context, plan=plan)
    if not required:
        return True
    missing = _scope_guard_missing_fact_keys(result, context)
    if not missing:
        return _context_has_missing_fact_signal(context)
    required_roots = {_fact_key_root(item) for item in required}
    missing_roots = {_fact_key_root(item) for item in missing}
    return bool(required_roots & missing_roots) or _context_has_missing_fact_signal(context)

def _scope_guard_required_fact_keys(
    context: Optional[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> tuple[str, ...]:
    result: list[str] = []
    for item in plan.get("required_fact_keys", ()) or ():
        cleaned = str(item or "").strip()
        if cleaned:
            result.append(cleaned)
    if isinstance(context, Mapping):
        facts_context = context.get("facts_context")
        if isinstance(facts_context, Mapping):
            for item in facts_context.get("required_fact_keys", ()) or ():
                cleaned = str(item or "").strip()
                if cleaned:
                    result.append(cleaned)
    return tuple(dict.fromkeys(result))

def _scope_guard_missing_fact_keys(
    result: SubscriptionDraftResult,
    context: Optional[Mapping[str, Any]],
) -> tuple[str, ...]:
    items: list[str] = [str(item).strip() for item in result.missing_facts if str(item).strip()]
    if isinstance(context, Mapping):
        value = context.get("missing_facts")
        if isinstance(value, str):
            if value.strip():
                items.append(value.strip())
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            items.extend(str(item).strip() for item in value if str(item).strip())
        facts_context = context.get("facts_context")
        if isinstance(facts_context, Mapping):
            value = facts_context.get("missing_facts")
            if isinstance(value, str):
                if value.strip():
                    items.append(value.strip())
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                items.extend(str(item).strip() for item in value if str(item).strip())
    return tuple(dict.fromkeys(items))

def _fact_key_root(value: object) -> str:
    text = str(value or "").strip().casefold()
    if not text:
        return ""
    for separator in (":", ".", "/", " "):
        if separator in text:
            text = text.split(separator, 1)[0]
            break
    aliases = {
        "schedule": "schedule",
        "расписание": "schedule",
        "дни": "schedule",
        "payment_methods": "payment",
        "payment": "payment",
        "оплата": "payment",
        "dolyami": "dolyami",
        "долями": "dolyami",
        "documents": "documents",
        "документы": "documents",
        "matkap": "matkap",
        "маткап": "matkap",
        "discounts": "discounts",
        "discount": "discounts",
        "скидка": "discounts",
        "prices": "prices",
        "price": "prices",
    }
    return aliases.get(text, text)

def _scope_guard_has_foreign_concrete_fact(
    draft_text: str,
    *,
    requested_scope: str,
    blocked_neighbor_scopes: Sequence[str],
) -> bool:
    answer_scopes = _answer_fact_scopes(str(draft_text or ""))
    if answer_scopes and not answer_scopes_allowed(
        answer_scopes,
        requested_scope=requested_scope,
        blocked_neighbor_scopes=tuple(blocked_neighbor_scopes),
    ):
        return True
    has_concrete = bool(CONCRETE_FACT_RE.search(str(draft_text or "")) or PRICE_AMOUNT_RE.search(str(draft_text or "")))
    if not has_concrete:
        return False
    if requested_scope and not answer_scopes:
        return True
    if requested_scope and answer_scopes and requested_scope not in answer_scopes:
        return True
    if blocked_neighbor_scopes and answer_scopes & {str(item) for item in blocked_neighbor_scopes}:
        return True
    return False

def _scope_fact_detail_label(
    context: Optional[Mapping[str, Any]],
    *,
    result: Optional[SubscriptionDraftResult] = None,
    plan: Optional[Mapping[str, Any]] = None,
) -> str:
    plan_mapping = plan if isinstance(plan, Mapping) else _conversation_intent_plan(context)
    scope = str(plan_mapping.get("fact_scope") or "").strip()
    required = " ".join(_scope_guard_required_fact_keys(context, plan=plan_mapping)).casefold()
    missing = " ".join(_scope_guard_missing_fact_keys(result, context) if result is not None else ()).casefold()
    haystack = " ".join([scope, required, missing, str(plan_mapping.get("primary_intent") or "")]).casefold()
    if any(marker in haystack for marker in ("schedule", "распис", "дни")):
        return "дни и время занятий нужной группы"
    if any(marker in haystack for marker in ("dolyami", "долями")):
        return "условия оплаты через Долями"
    if any(marker in haystack for marker in ("payment", "оплата", "счет", "счёт")):
        return "способ оплаты по выбранному курсу"
    if any(marker in haystack for marker in ("discount", "скид", "second_subject")):
        return "скидку по вашему формату и предметам"
    if any(marker in haystack for marker in ("refund_policy", "refund", "возврат")):
        return "порядок возврата по выбранному курсу"
    if any(marker in haystack for marker in ("matkap", "маткап", "documents", "документ")):
        return "документы и порядок оформления маткапитала"
    if any(marker in haystack for marker in ("city_day_camp", "camp", "смен", "лагер")):
        return "нужную смену и формат лагеря"
    if any(marker in haystack for marker in ("trial", "пробн", "fragment", "фрагмент")):
        return "пробный формат или фрагмент занятия"
    return "эту деталь"

def _scope_fact_narrow_handoff_text(
    context: Optional[Mapping[str, Any]],
    *,
    result: Optional[SubscriptionDraftResult] = None,
    plan: Optional[Mapping[str, Any]] = None,
) -> str:
    detail = _scope_fact_detail_label(context, result=result, plan=plan)
    return (
        f"По этому вопросу у меня нет подтверждённого факта именно про {detail}, "
        "поэтому не буду подставлять похожую информацию из другой темы. "
        f"Передам менеджеру запрос именно про {detail}; он проверит и ответит точно."
    )

def _select_nonrepeating_text(variants: Sequence[str], previous_bot_texts: Sequence[str], *, fallback: str) -> str:
    for candidate in variants:
        text = str(candidate or "").strip()
        if text and not is_near_repeat(text, previous_bot_texts, threshold=0.82):
            return text
    return fallback

def _p0_text_with_antirepeat(kind: str, base: str, context: Optional[Mapping[str, Any]]) -> str:
    previous = _humanity_previous_bot_texts(context)
    if not previous or not is_near_repeat(base, previous, threshold=0.82):
        return base
    if kind == "refund":
        variants = _REFUND_ZERO_COLLECT_VARIANTS
    elif kind == "complaint":
        variants = _COMPLAINT_SAFE_VARIANTS
    elif kind == "payment_dispute":
        variants = _PAYMENT_DISPUTE_VARIANTS
    else:
        variants = _LEGAL_SAFE_VARIANTS
    return _select_nonrepeating_text(variants, previous, fallback=base)



def _answer_fact_scopes(text: str) -> set[str]:
    return detect_fact_scopes(text)


def _has_missing_fact_signal(result: SubscriptionDraftResult, context: Optional[Mapping[str, Any]]) -> bool:
    if result.missing_facts:
        return True
    return _context_has_missing_fact_signal(context)

def _context_has_missing_fact_signal(context: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(context, Mapping):
        return False
    if _truthy_value(context.get("facts_missing")) or _truthy_value(context.get("missing")):
        return True
    missing = context.get("missing_facts")
    if isinstance(missing, str):
        return bool(missing.strip())
    if isinstance(missing, Sequence) and not isinstance(missing, (str, bytes, bytearray)):
        return any(str(item or "").strip() for item in missing)
    facts_context = context.get("facts_context")
    if isinstance(facts_context, Mapping):
        return _truthy_value(facts_context.get("facts_missing")) or _truthy_value(facts_context.get("missing"))
    return False

def _dedupe_sentence(text: str, sentence: str) -> str:
    value = str(text or "")
    target = str(sentence or "").strip()
    if not target:
        return value
    first = value.find(target)
    if first < 0:
        return value
    before = value[: first + len(target)]
    after = value[first + len(target) :]
    after = after.replace(target, "")
    return before + after

def _autonomy_policy(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    policy = context.get("autonomy_policy")
    if isinstance(policy, Mapping):
        return policy
    rop_policy = context.get("rop_policy")
    if isinstance(rop_policy, Mapping) and isinstance(rop_policy.get("autonomy_policy"), Mapping):
        return rop_policy["autonomy_policy"]  # type: ignore[return-value,index]
    return {}

def _autonomy_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    policy = _autonomy_policy(context)
    return (
        _truthy_value(policy.get("allow_autonomous"))
        or _truthy_value(policy.get("enabled"))
        or _truthy_value(policy.get("bot_answer_self_enabled"))
        or _truthy_value(context.get("autonomy_enabled") if isinstance(context, Mapping) else None)
    )

def _autonomy_topic_allowed(topic_id: str, context: Optional[Mapping[str, Any]]) -> bool:
    topic = str(topic_id or "").strip()
    if topic not in AUTONOMY_MATRIX_SAFE_TOPIC_IDS:
        return False
    policy = _autonomy_policy(context)
    configured = policy.get("allowed_topic_ids") or policy.get("autonomous_topic_ids") or policy.get("topic_ids")
    if configured is None:
        return True
    configured_ids = {str(item or "").strip() for item in configured} if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes, bytearray)) else {str(configured or "").strip()}
    return topic in configured_ids

@dataclass(frozen=True)
class RouteDecision:
    route: str
    veto_category: str = ""
    safety_flags: tuple[str, ...] = ()
    manager_checklist: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    autonomous_candidate: bool = False









def _has_client_safe_current_fact(context: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(context, Mapping):
        return False
    if _truthy_value(context.get("client_safe_fact_verified")) or _truthy_value(context.get("autonomy_fact_verified")):
        return True
    return _mapping_has_client_safe_current_fact(context.get("confirmed_facts")) or _mapping_has_client_safe_current_fact(
        context.get("facts_context")
    )

def _mapping_has_client_safe_current_fact(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        if _truthy_value(value.get("internal_only")) or _truthy_value(value.get("stale")) or _truthy_value(value.get("facts_stale")):
            return False
        safe = any(
            _truthy_value(value.get(key))
            for key in (
                "client_safe",
                "client_allowed",
                "allowed_for_client_answer",
                "client_safe_fact",
                "client_safe_fact_verified",
                "pilot_allowed",
            )
        )
        current = any(
            _truthy_value(value.get(key))
            for key in (
                "fresh",
                "facts_fresh",
                "fresh_facts",
                "current",
                "actual",
                "is_actual",
                "document_verified",
                "fresh_verified",
            )
        )
        status_text = " ".join(
            str(value.get(key) or "")
            for key in ("freshness", "source_status", "approval_status", "verification_status", "status")
        ).casefold()
        current = current or any(marker in status_text for marker in ("fresh", "current", "actual", "document_verified", "verified"))
        if safe and current:
            return True
        return any(_mapping_has_client_safe_current_fact(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_mapping_has_client_safe_current_fact(item) for item in value)
    return False

def _humanity_previous_bot_texts(context: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    if not isinstance(context, Mapping):
        return ()
    result: list[str] = []
    memory = context.get("dialogue_memory_view")
    if isinstance(memory, Mapping):
        turns = memory.get("recent_turns")
        if isinstance(turns, Sequence) and not isinstance(turns, (str, bytes, bytearray)):
            for item in turns:
                if isinstance(item, Mapping) and str(item.get("role") or "").casefold() in {"bot", "assistant"}:
                    text = str(item.get("text") or "").strip()
                    if text:
                        result.append(text)
    recent = context.get("recent_messages")
    if isinstance(recent, Sequence) and not isinstance(recent, (str, bytes, bytearray)):
        for item in recent:
            text = str(item or "").strip()
            if text.casefold().startswith(("ответ:", "bot:", "бот:", "assistant:")):
                result.append(text.split(":", 1)[-1].strip())
    return tuple(dict.fromkeys(item for item in result[-20:] if item))


def _semantic_haystack(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    haystack = " ".join(
        [
            str(client_message or ""),
            result.draft_text,
            result.topic_id,
            result.broad_group,
            *result.alternative_themes,
            *result.context_warnings,
        ]
    ).casefold()
    if isinstance(context, Mapping):
        for key in ("risk_flags", "context_warnings"):
            value = context.get(key)
            text = " ".join(str(item or "") for item in value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else str(value or "")
            haystack += " " + text.casefold()
    return haystack

def _dialog_context_haystack(context: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(context, Mapping):
        return ""
    texts: list[str] = []
    for key in ("recent_messages", "dialog_messages", "conversation_messages"):
        value = context.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            texts.extend(str(item or "") for item in value[-8:])
        elif isinstance(value, str):
            texts.append(value)
    for key in ("customer_context_summary", "known_context_summary"):
        value = str(context.get(key) or "").strip()
        if value:
            texts.append(value)
    return " ".join(texts).casefold().replace("ё", "е")




def _draft_confirms_payment(result: SubscriptionDraftResult) -> bool:
    if result.topic_id == "theme:003_payment_status" and PAYMENT_CONFIRMATION_RE.search(result.draft_text):
        return True
    return bool(PAYMENT_CONFIRMATION_RE.search(result.draft_text))

def _payment_context(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    payment = context.get("payment_context")
    merged: dict[str, Any] = {}
    if isinstance(payment, Mapping):
        merged.update(payment)
    for key in (
        "amo_payment_status",
        "tallanto_payment_status",
        "amo_status",
        "tallanto_status",
        "payment_conflict",
        "amo_tallanto_payment_conflict",
        "payment_last_seen_at",
        "payment_source_confidence",
    ):
        if key in context:
            merged[key] = context[key]
    amo = context.get("amo_context")
    if isinstance(amo, Mapping):
        for key in ("payment_status", "amo_payment_status", "paid"):
            if key in amo and "amo_payment_status" not in merged:
                merged["amo_payment_status"] = amo[key]
    tallanto = context.get("tallanto_context")
    if isinstance(tallanto, Mapping):
        for key in ("payment_status", "tallanto_payment_status", "paid"):
            if key in tallanto and "tallanto_payment_status" not in merged:
                merged["tallanto_payment_status"] = tallanto[key]
    return merged

def _payment_status(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if isinstance(value, bool):
        return "paid" if value else "not_paid"
    if text in {"paid", "оплачено", "оплачен", "оплачена", "yes", "true", "1", "received", "success"}:
        return "paid"
    if text in {"not_paid", "не оплачено", "нет", "false", "0", "missing", "unpaid"}:
        return "not_paid"
    return text

def _payment_guarded_result(result: SubscriptionDraftResult, *, reason: str, checklist: str) -> SubscriptionDraftResult:
    return replace(
        result,
        route="manager_only",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, reason, "payment_confirmation_guarded"])),
        manager_checklist=tuple(dict.fromkeys([*result.manager_checklist, checklist])),
        metadata={**dict(result.metadata), reason: True},
    )


def _extract_numeric_promise_claims(text: str) -> tuple[str, ...]:
    source = str(text or "")
    claims: list[str] = []
    for pattern in UNSUPPORTED_PROMISE_PATTERNS:
        for match in pattern.finditer(source):
            if pattern is _N_POINTS_PROMISE_CONTEXT_RE:
                for points_match in _BARE_N_POINTS_RE.finditer(match.group(0)):
                    claim = " ".join(points_match.group(0).split())
                    if claim:
                        claims.append(claim)
                continue
            claim = " ".join(match.group(0).split())
            if claim:
                claims.append(claim)
    return tuple(dict.fromkeys(claims))
