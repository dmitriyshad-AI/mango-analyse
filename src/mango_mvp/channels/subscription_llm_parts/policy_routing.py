from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.channels.answer_safety_classifier import classify_answer_safety
from mango_mvp.channels.dialogue_debug_trace import trace_event
from mango_mvp.channels.dialogue_contract_pipeline import (
    _GENERIC_HANDOFF_TEXTS as dialogue_contract_generic_handoff_texts,
    _HANDOFF_EXHAUSTED_TEXTS as dialogue_contract_handoff_exhausted_texts,
    _handoff_factual_claim_text as dialogue_contract_handoff_factual_claim_text,
    _is_pure_handoff_text as dialogue_contract_is_pure_handoff_text,
    check_claim_faithfulness as check_dialogue_contract_faithfulness,
    concrete_anchors as dialogue_contract_concrete_anchors,
    faithfulness_shadow_enabled as dialogue_contract_faithfulness_shadow_enabled,
    faithfulness_shadow_events as dialogue_contract_faithfulness_shadow_events,
    faithfulness_shadow_record as dialogue_contract_faithfulness_shadow_record,
    new_concrete_anchors as dialogue_contract_new_concrete_anchors,
    parse_contract as parse_dialogue_contract,
    verify_output as verify_dialogue_contract_output,
)
from mango_mvp.channels.draft_prompt_builder import IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES, safe_schedule_template, should_force_manager_only
from mango_mvp.channels.fact_scope_spec import answer_scopes_allowed, detect_fact_scopes
from mango_mvp.channels.humanity_guards import is_near_repeat
from mango_mvp.channels.p0_recall_spec import HARD_P0_CODES, codes_from_text, is_benign_hypothetical_refund
from mango_mvp.channels.rules_engine import (
    RuleOutcome,
    apply_rule as apply_migrated_domain_rule,
    load_rules_registry,
    select_rule as select_migrated_domain_rule,
)
from mango_mvp.channels.text_signals import has_any_marker, has_marker
from mango_mvp.channels.tone_block import apply_warm_frame
from mango_mvp.insights.phase2_detectors import detect_anxiety, detect_objection
from mango_mvp.question_catalog.classifier import load_valid_theme_and_service_ids

from mango_mvp.channels.subscription_llm_parts.contracts import (
    BASE_SAFETY_FLAGS,
    SAFE_FALLBACK_DRAFT_TEXT,
    SubscriptionDraftResult,
)
from mango_mvp.channels.subscription_llm_parts.support import (
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
    _keep_answer_hard_anchors,
    _keep_answer_supported,
    _normalize_fact_match_text,
    _p0_model_led_complaint_backstop,
    _p0_model_led_enabled,
    _p0_model_led_filter_high_risk_codes,
    _presale_prompt_child_name_value,
    _template_from_kb_enabled,
    _template_from_kb_trace_event,
    _truthy_value,
)

ANSWER_CONTRACT_GREEN_TEMPLATE_REDUCTION_ENV = "TELEGRAM_ANSWER_CONTRACT_GREEN_TEMPLATE_REDUCTION"

RULES_ENGINE_PLANNER_INTENT_ENV = "TELEGRAM_RULES_ENGINE_PLANNER_INTENT"

SCOPE_FACT_GUARD_ENV = "TELEGRAM_SCOPE_FACT_GUARD"

A_THREAD_ENV = "TELEGRAM_A_THREAD"

PH2_OBJECTION_ENV = "TELEGRAM_PH2_OBJECTION"

PH2_ANXIETY_ENV = "TELEGRAM_PH2_ANXIETY"

STEP4_KEEP_ANSWER_ENV = "TELEGRAM_STEP4_KEEP_ANSWER"

PLANNER_INTENT_CONFIDENCE_THRESHOLD = 0.72

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

COMPLAINT_SAFE_TEXT = "Приняли обращение. Передам вопрос менеджеру, он вернётся с ответом."

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

OFF_TOPIC_INPUT_RE = re.compile(
    r"\b(?:iphone|айфон|андроид|android|биткоин|крипт\w*|курс\s+доллар|погод\w*|рецепт\w*|политик\w*|новост\w*)\b",
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

@dataclass(frozen=True)
class SafeTemplateSpec:
    name: str
    priority: int
    produce: Callable[[SubscriptionDraftResult, str, Optional[Mapping[str, Any]]], str]
    route_on_apply: str
    flag: str
    checklist: str
    extra_flags: tuple[str, ...] = ()
    topic_on_apply: str = ""
    topic_flag: str = ""

def _produce_cross_brand_template(
    result: SubscriptionDraftResult,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> str:
    return _cross_brand_safe_template(result, client_message=client_message, context=context)

def _produce_terminal_template(
    result: SubscriptionDraftResult,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> str:
    return _terminal_safe_template(result, client_message=client_message, context=context)

def _produce_result_guarantee_template(
    result: SubscriptionDraftResult,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> str:
    return RESULT_GUARANTEE_SAFE_TEXT if _is_result_guarantee_case(result, client_message=client_message, context=context) else ""

def _produce_admission_guarantee_template(
    result: SubscriptionDraftResult,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> str:
    return ADMISSION_GUARANTEE_SAFE_TEXT if _is_admission_guarantee_case(result, client_message=client_message, context=context) else ""

DIALOGUE_CONTRACT_V2_TEMPLATE_REGISTRY: tuple[SafeTemplateSpec, ...] = (
    SafeTemplateSpec(
        name="cross_brand",
        priority=10,
        produce=_produce_cross_brand_template,
        route_on_apply="keep_or_draft",
        flag="cross_brand_safe_template_applied",
        checklist="Кросс-бренд: не консультировать по другому бренду и не сравнивать условия.",
    ),
    SafeTemplateSpec(
        name="terminal",
        priority=20,
        produce=_produce_terminal_template,
        route_on_apply="terminal",
        flag="terminal_safe_template_applied",
        checklist="Терминальный случай: identity/адрес/контакты/офф-топик — безопасный шаблон.",
    ),
    SafeTemplateSpec(
        name="result_guarantee",
        priority=30,
        produce=_produce_result_guarantee_template,
        route_on_apply="draft_for_manager",
        flag="result_guarantee_safe_template_applied",
        checklist="Не гарантировать балл/результат: только программа и статистика.",
        extra_flags=("placeholder_in_draft",),
    ),
    SafeTemplateSpec(
        name="admission_guarantee",
        priority=31,
        produce=_produce_admission_guarantee_template,
        route_on_apply="draft_for_manager",
        flag="admission_guarantee_safe_template_applied",
        checklist="Не гарантировать поступление: только программа и статистика.",
        extra_flags=("placeholder_in_draft",),
    ),
)

_INFORMATIONAL_SAFE_TEMPLATE_NAMES = {"terminal"}

def _is_informational_terminal_template(text: str) -> bool:
    clean = str(text or "").strip()
    if _is_template_from_kb_terminal_text(clean):
        return True
    return clean in {
        ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
        ADDRESS_UNPK_SAFE_TEXT,
        ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
        CONTACT_FOTON_SAFE_TEXT,
        CONTACT_UNPK_SAFE_TEXT,
    }

def _is_template_from_kb_terminal_text(text: str) -> bool:
    clean = " ".join(str(text or "").split())
    if not clean:
        return False
    return (
        clean.startswith("Здравствуйте! В Москве Фотон находится по адресу ")
        or clean.startswith("Здравствуйте! Регулярные занятия в Москве проходят по адресу ")
        or clean.startswith("Площадки УНПК: Москва — ")
        or clean.startswith("Телефоны: ")
    )

def _safe_template_already_applied(result: SubscriptionDraftResult) -> bool:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    if isinstance(metadata.get("dialogue_contract_v2_template_dispatcher"), Mapping):
        return True
    return any(spec.flag in result.safety_flags or metadata.get(spec.flag) for spec in DIALOGUE_CONTRACT_V2_TEMPLATE_REGISTRY)

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

def _safe_template_can_yield_to_dispatcher(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
    shadow: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> bool:
    flags = set(result.safety_flags)
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    if isinstance(metadata.get("dialogue_contract_v2_template_dispatcher"), Mapping):
        return False
    if result.route == "manager_only" or is_high_risk_result(result):
        return False
    terminal_applied = "terminal_safe_template_applied" in flags or bool(metadata.get("terminal_safe_template_applied"))
    if terminal_applied and not _is_informational_terminal_template(result.draft_text):
        return False
    if flags.intersection(_SAFE_TEMPLATE_DISPATCHER_RECONSIDER_BLOCKING_FLAGS):
        return False
    if any(bool(metadata.get(flag)) for flag in _SAFE_TEMPLATE_DISPATCHER_RECONSIDER_BLOCKING_FLAGS):
        return False
    if _is_policy_c_identity_question(result, context=context):
        return True
    intent = str(shadow.get("selected_intent") or "").strip()
    return select_migrated_domain_rule(intent, registry) is not None

def _safe_template_route(result: SubscriptionDraftResult, spec: SafeTemplateSpec, text: str) -> str:
    if spec.route_on_apply == "manager_only":
        return "manager_only"
    if spec.route_on_apply == "draft_for_manager":
        return "draft_for_manager"
    if spec.route_on_apply == "keep_or_draft":
        return "manager_only" if result.route == "manager_only" else "draft_for_manager"
    if spec.route_on_apply == "terminal":
        if str(text or "") in {IDENTITY_FOTON_SAFE_TEXT, IDENTITY_UNPK_SAFE_TEXT}:
            return "bot_answer_self_for_pilot"
        return "draft_for_manager" if _is_terminal_direct_info_template(text) else "manager_only"
    return result.route

def _is_approved_policy_c_identity_text(text: str, *, active_brand: str) -> bool:
    clean_text = str(text or "").strip()
    if active_brand == "foton":
        return clean_text == IDENTITY_FOTON_SAFE_TEXT
    if active_brand == "unpk":
        return clean_text == IDENTITY_UNPK_SAFE_TEXT
    return False

def _policy_c_identity_allowed(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> bool:
    if not _is_policy_c_identity_question(result, context=context):
        return False
    if result.route == "manager_only" or is_high_risk_result(result):
        return False
    if bool(_dialogue_contract_mapping(result).get("is_p0")):
        return False
    return not detect_high_risk_input_markers(client_message, context=context)

def _is_terminal_direct_info_template(text: str) -> bool:
    clean = str(text or "")
    if _is_template_from_kb_terminal_text(clean):
        return True
    return clean in {
        ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
        ADDRESS_UNPK_SAFE_TEXT,
        ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
        CONTACT_FOTON_SAFE_TEXT,
        CONTACT_UNPK_SAFE_TEXT,
        IDENTITY_PROMPT_SAFE_TEXT,
        IDENTITY_FOTON_SAFE_TEXT,
        IDENTITY_UNPK_SAFE_TEXT,
        OFF_TOPIC_FOTON_SAFE_TEXT,
        OFF_TOPIC_UNPK_SAFE_TEXT,
        OFF_TOPIC_GENERIC_SAFE_TEXT,
        SOFT_NEGATIVE_HANDOFF_SAFE_TEXT,
    }

def _apply_safe_template_spec(
    result: SubscriptionDraftResult,
    spec: SafeTemplateSpec,
    text: str,
) -> SubscriptionDraftResult:
    clean_text = str(text or "").strip()
    if not clean_text:
        return result
    metadata = dict(result.metadata)
    if clean_text != str(result.draft_text or ""):
        metadata = _metadata_with_guarded_original_text(metadata, result.draft_text, guard=f"safe_template:{spec.name}")
    metadata[spec.flag] = True
    metadata["dialogue_contract_v2_template_dispatcher"] = {
        "applied": spec.name,
        "priority": spec.priority,
    }
    flags = tuple(dict.fromkeys([*result.safety_flags, spec.flag, *spec.extra_flags]))
    if spec.name == "terminal" and not _is_terminal_direct_info_template(clean_text):
        flags = tuple(dict.fromkeys([*flags, "placeholder_in_draft"]))
    topic_id = result.topic_id
    if spec.topic_on_apply and topic_id != spec.topic_on_apply:
        topic_id = spec.topic_on_apply
        if spec.topic_flag:
            flags = tuple(dict.fromkeys([*flags, spec.topic_flag]))
    checklist = tuple(dict.fromkeys([*result.manager_checklist, spec.checklist]))
    if spec.topic_flag and spec.topic_flag in flags:
        metadata[spec.topic_flag] = True
    return replace(
        result,
        topic_id=topic_id,
        route=_safe_template_route(result, spec, clean_text),
        draft_text=clean_text,
        safety_flags=flags,
        manager_checklist=checklist,
        metadata=metadata,
    )

def _dialogue_contract_retrieved_facts(result: SubscriptionDraftResult) -> dict[str, str]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    retrieved = pipeline.get("retrieved_facts") if isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
    return {str(key): str(value) for key, value in retrieved.items() if str(key).strip() and str(value).strip()}

def _dialogue_contract_mapping(result: SubscriptionDraftResult) -> Mapping[str, Any]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    contract = pipeline.get("contract") if isinstance(pipeline.get("contract"), Mapping) else {}
    return contract

def _migrated_rule_intent_from_dialogue_contract(result: SubscriptionDraftResult) -> str:
    contract = _dialogue_contract_mapping(result)
    topic_id = str(contract.get("topic_id") or result.topic_id or "").casefold()
    haystack = " ".join(
        str(item or "")
        for item in (
            contract.get("current_question"),
            contract.get("client_state"),
            contract.get("question_type"),
            " ".join(str(key or "") for key in (contract.get("needed_fact_keys") or ())),
            " ".join(str(item or "") for item in (contract.get("composite_subquestions") or ())),
            topic_id,
        )
    ).casefold()
    if "theme:015_address" in topic_id or re.search(
        r"\b(address|location|metro|locations?)\b|адрес|площадк|где\s+.*находит|где\s+.*занят",
        haystack,
        re.I,
    ):
        return "contact_address"
    if "theme:017_teachers" in topic_id or re.search(r"преподав|педагог|учитель|кто\s+вед", haystack, re.I):
        return "teacher"
    if "theme:018_materials_homework" in topic_id or re.search(r"record|recording|запис[ьи]|пересмотр", haystack, re.I):
        return "recordings"
    if "theme:024_account_access" in topic_id or re.search(r"личный кабинет|кабинет|платформ|логин|парол|электрон|документооборот|скан-коп", haystack, re.I):
        return "platform_access"
    if "theme:007_matkap_payment" in topic_id or re.search(r"маткап|материн|сфр", haystack, re.I):
        return "matkap"
    if "theme:008_tax_deduction" in topic_id or re.search(r"налог|вычет|фнс|ндфл", haystack, re.I):
        return "tax"
    if "theme:016_program" in topic_id and re.search(r"олимпиад|физтех", haystack, re.I):
        return "olympiad"
    if "theme:012_certificates" in topic_id or "theme:011_contract" in topic_id or re.search(r"договор|справк|сертификат|документ|квитанц|чек|лиценз|юрлиц", haystack, re.I):
        return "docs"
    if "theme:001_pricing" in topic_id or re.search(r"\b(price|pricing|prices?)\b|цен|стоим|сколько\s+стоит|прайс|₽|руб", haystack, re.I):
        return "pricing"
    if "theme:014_format" in topic_id or re.search(r"\bformat\b|формат|онлайн\s+или\s+очно|очно\s+или\s+онлайн", haystack, re.I):
        return "format"
    if "theme:013_schedule" in topic_id or re.search(
        r"\b(schedule|schedule_weekend)\b|распис|по\s+каким\s+дням|когда\s+занят|раз\s+в\s+недел|выходн|суббот|воскрес",
        haystack,
        re.I,
    ):
        return "schedule"
    if "theme:023_trial_class" in topic_id or re.search(r"пробн|фрагмент\s+занят|фрагмент\s+урок", haystack, re.I):
        return "trial"
    if (
        "theme:026_camp_general" in topic_id
        or "theme:027_camp_living_conditions" in topic_id
        or "theme:028_transport_logistics" in topic_id
        or re.search(r"лагер|лвш|лш|менделеев|выездн|смен", haystack, re.I)
    ):
        return "camp_lvsh"
    if "theme:020_enrollment" in topic_id or re.search(r"записаться|оформиться|оформить(?:ся)?|как\s+запис", haystack, re.I):
        return "enrollment_process"
    return ""

def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def _rules_engine_planner_intent_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping) and context.get(RULES_ENGINE_PLANNER_INTENT_ENV) is not None:
        return _truthy_value(context.get(RULES_ENGINE_PLANNER_INTENT_ENV))
    env_value = os.getenv(RULES_ENGINE_PLANNER_INTENT_ENV)
    if env_value is None:
        return True
    return _truthy_value(env_value)

def _planner_intent_candidate(contract: Mapping[str, Any], registry: Mapping[str, Any]) -> tuple[str, str, float, bool]:
    intent = str(contract.get("planner_intent") or "").strip()
    confidence = _float_value(contract.get("planner_confidence"))
    if not intent or confidence < PLANNER_INTENT_CONFIDENCE_THRESHOLD:
        return intent, "", confidence, False
    rule = select_migrated_domain_rule(intent, registry)
    return intent, getattr(rule, "rule_id", "") if rule is not None else "", confidence, rule is not None

def _is_policy_c_identity_question(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
) -> bool:
    contract = _dialogue_contract_mapping(result)
    plan = _conversation_intent_plan(context)
    haystack = " ".join(
        str(item or "")
        for item in (
            contract.get("current_question"),
            contract.get("client_state"),
            contract.get("question_type"),
            " ".join(str(item or "") for item in (contract.get("composite_subquestions") or ())),
            plan.get("direct_question"),
        )
    ).casefold()
    if not haystack:
        return False
    return bool(
        re.search(r"\b(?:бот|робот|ии|нейросет\w*|человек)\b", haystack, flags=re.I)
        or "с кем я общаюсь" in haystack
        or "живой оператор" in haystack
        or "живой человек" in haystack
    )

def _rules_engine_intent_shadow(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> Mapping[str, Any]:
    contract = _dialogue_contract_mapping(result)
    plan = _conversation_intent_plan(context)
    planner_intent, planner_rule, planner_confidence, planner_available = _planner_intent_candidate(contract, registry)
    keyword_intent = str(plan.get("primary_intent") or "").strip()
    keyword_rule = select_migrated_domain_rule(keyword_intent, registry)
    regex_intent = _migrated_rule_intent_from_dialogue_contract(result)
    regex_rule = select_migrated_domain_rule(regex_intent, registry)
    selected_source = "keyword" if keyword_rule is not None else "regex" if regex_rule is not None else ""
    selected_intent = keyword_intent if selected_source == "keyword" else regex_intent if selected_source == "regex" else ""
    planner_enabled = _rules_engine_planner_intent_enabled(context)
    identity_policy_c = _is_policy_c_identity_question(result, context=context)
    if identity_policy_c:
        selected_source = "identity_policy"
        selected_intent = "identity"
    elif planner_enabled and planner_available:
        selected_source = "planner"
        selected_intent = planner_intent
    return {
        "schema_version": "rules_engine_intent_shadow_v1_2026_06_02",
        "planner_intent": planner_intent,
        "planner_subvariant": str(contract.get("planner_subvariant") or ""),
        "planner_slots": dict(contract.get("planner_slots") or {}) if isinstance(contract.get("planner_slots"), Mapping) else {},
        "planner_confidence": round(planner_confidence, 3),
        "planner_available": planner_available,
        "planner_rule": planner_rule,
        "keyword_intent": keyword_intent,
        "keyword_rule": getattr(keyword_rule, "rule_id", "") if keyword_rule is not None else "",
        "regex_intent": regex_intent,
        "regex_rule": getattr(regex_rule, "rule_id", "") if regex_rule is not None else "",
        "agreement_planner_keyword": bool(planner_intent and planner_intent == keyword_intent),
        "agreement_planner_regex": bool(planner_intent and planner_intent == regex_intent),
        "selected_source": selected_source,
        "selected_intent": selected_intent,
        "planner_intent_enabled": planner_enabled,
        "planner_blocked_by_identity_policy": identity_policy_c,
    }

def _with_rules_engine_intent_shadow(
    result: SubscriptionDraftResult,
    *,
    shadow: Mapping[str, Any],
) -> SubscriptionDraftResult:
    metadata = dict(result.metadata)
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    pipeline = dict(pipeline)
    pipeline["rules_engine_intent_shadow"] = dict(shadow)
    metadata["dialogue_contract_pipeline"] = pipeline
    metadata["rules_engine_intent_shadow"] = dict(shadow)
    return replace(result, metadata=metadata)

def _rules_engine_facts(result: SubscriptionDraftResult, context: Optional[Mapping[str, Any]]) -> dict[str, str]:
    facts = _dialogue_contract_retrieved_facts(result)
    if isinstance(context, Mapping):
        confirmed = context.get("confirmed_facts")
        if isinstance(confirmed, Mapping):
            for key, value in confirmed.items():
                text = _client_clean_fact_text(value)
                if str(key).strip() and text:
                    facts.setdefault(str(key), text)
    return facts

def _apply_rules_engine_outcome(result: SubscriptionDraftResult, outcome: RuleOutcome) -> SubscriptionDraftResult:
    metadata = dict(result.metadata)
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    pipeline = dict(pipeline)
    retrieved = pipeline.get("retrieved_facts") if isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
    retrieved_facts = {str(key): str(value) for key, value in retrieved.items() if str(key).strip() and str(value).strip()}
    retrieved_facts.update({str(key): str(value) for key, value in outcome.facts.items() if str(key).strip() and str(value).strip()})
    pipeline["retrieved_facts"] = retrieved_facts
    pipeline["retrieved_fact_keys"] = list(dict.fromkeys([*(pipeline.get("retrieved_fact_keys") or ()), *retrieved_facts.keys()]))
    pipeline["rules_engine"] = {
        "applied": outcome.rule_id,
        "subvariant": outcome.subvariant,
        "route": outcome.route,
    }
    metadata["dialogue_contract_pipeline"] = pipeline
    metadata["rules_engine"] = {
        "applied": outcome.rule_id,
        "subvariant": outcome.subvariant,
        **dict(outcome.metadata),
    }
    flags = tuple(dict.fromkeys([*result.safety_flags, *outcome.flags]))
    checklist = tuple(dict.fromkeys([*result.manager_checklist, *outcome.checklist]))
    context_used = tuple(dict.fromkeys([*result.context_used, "rules_engine"]))
    return replace(
        result,
        route=outcome.route or result.route,
        draft_text=outcome.text,
        safety_flags=flags,
        manager_checklist=checklist,
        context_used=context_used,
        metadata=metadata,
    )

def _apply_migrated_rules_engine(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult | None:
    contract = _dialogue_contract_mapping(result)
    if (
        bool(contract.get("is_p0"))
        or detect_high_risk_input_markers(client_message, context=context)
        or is_high_risk_result(result)
    ):
        return None
    plan = _conversation_intent_plan(context)
    registry = load_rules_registry()
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    shadow = pipeline.get("rules_engine_intent_shadow") if isinstance(pipeline.get("rules_engine_intent_shadow"), Mapping) else {}
    if not shadow:
        shadow = _rules_engine_intent_shadow(result, context=context, registry=registry)
    selected_source = str(shadow.get("selected_source") or "")
    if selected_source == "identity_policy":
        return None
    intent = str(shadow.get("selected_intent") or plan.get("primary_intent") or "").strip()
    rule = select_migrated_domain_rule(intent, registry)
    intent_from_contract = selected_source in {"regex", "planner"}
    if rule is None:
        intent = _migrated_rule_intent_from_dialogue_contract(result)
        intent_from_contract = bool(intent)
        rule = select_migrated_domain_rule(intent, registry)
    if rule is None:
        return None
    if result.route == "manager_only" and not _manager_route_migrated_rules_override_allowed(result, intent=intent):
        if selected_source == "planner":
            fallback_intent = str(shadow.get("keyword_intent") or shadow.get("regex_intent") or "")
        elif intent_from_contract:
            fallback_intent = ""
        else:
            fallback_intent = _migrated_rule_intent_from_dialogue_contract(result)
        if not fallback_intent or fallback_intent == intent or not _manager_route_migrated_rules_override_allowed(result, intent=fallback_intent):
            return None
        fallback_rule = select_migrated_domain_rule(fallback_intent, registry)
        if fallback_rule is None:
            return None
        intent = fallback_intent
        rule = fallback_rule
        intent_from_contract = True
    facts = _rules_engine_facts(result, context)
    if _migrated_rules_keep_existing_verified_answer(result, client_message=client_message, context=context, facts=facts):
        return None
    direct_question = str(
        (contract.get("current_question") if intent_from_contract else plan.get("direct_question"))
        or contract.get("current_question")
        or client_message
        or ""
    )
    phase2_objection = _phase2_objection_signal(direct_question, context)
    phase2_anxiety = _phase2_anxiety_signal(direct_question, context)
    enriched_plan = {
        **dict(plan),
        "primary_intent": intent or str(plan.get("primary_intent") or ""),
        "planner_intent": str(contract.get("planner_intent") or ""),
        "planner_subvariant": str(contract.get("planner_subvariant") or ""),
        "planner_slots": dict(contract.get("planner_slots") or {}) if isinstance(contract.get("planner_slots"), Mapping) else {},
        "planner_confidence": _float_value(contract.get("planner_confidence")),
        "selling": _merged_selling_signals(
            contract.get("selling"),
            plan.get("selling"),
            phase2_objection=phase2_objection,
            phase2_anxiety=phase2_anxiety,
        ),
        "rules_engine_intent_source": str(shadow.get("selected_source") or ""),
        "direct_question": direct_question,
    }
    rule_context = _context_with_selling_thread_slots(context, contract=contract, client_message=client_message)
    outcome = apply_migrated_domain_rule(rule, plan=enriched_plan, facts=facts, context=rule_context)
    if outcome is None and (not intent_from_contract or selected_source == "planner"):
        fallback_intent = (
            str(shadow.get("keyword_intent") or shadow.get("regex_intent") or "")
            if selected_source == "planner"
            else _migrated_rule_intent_from_dialogue_contract(result)
        )
        if fallback_intent and fallback_intent != intent:
            fallback_rule = select_migrated_domain_rule(fallback_intent, registry)
            if fallback_rule is not None and (
                result.route != "manager_only"
                or _manager_route_migrated_rules_override_allowed(result, intent=fallback_intent)
            ):
                fallback_plan = {
                    **dict(plan),
                    "primary_intent": fallback_intent,
                    "direct_question": str(contract.get("current_question") or client_message or ""),
                    "selling": _merged_selling_signals(
                        contract.get("selling"),
                        plan.get("selling"),
                        phase2_objection=phase2_objection,
                        phase2_anxiety=phase2_anxiety,
                    ),
                }
                outcome = apply_migrated_domain_rule(fallback_rule, plan=fallback_plan, facts=facts, context=rule_context)
    if outcome is None:
        return None
    return _apply_rules_engine_outcome(result, outcome)

def _context_with_selling_thread_slots(
    context: Optional[Mapping[str, Any]],
    *,
    contract: Mapping[str, Any],
    client_message: str,
) -> Optional[Mapping[str, Any]]:
    if not _a_thread_enabled(context):
        return context
    current_slots = _selling_slots_from_contract_and_text(contract, client_message)
    memory_slots = _selling_slots_from_memory(context)
    slots: dict[str, str] = {}
    for key in ("subject", "grade", "format", "product"):
        if current_slots.get(key):
            slots[key] = current_slots[key]
        elif memory_slots.get(key) and not _text_explicitly_mentions_selling_slot(client_message, key):
            slots[key] = memory_slots[key]
    brand = _active_brand(context)
    if brand in {"foton", "unpk"}:
        slots["active_brand"] = brand
    if not slots:
        return context
    merged = dict(context or {})
    merged["selling_thread_slots"] = slots
    return merged

def _a_thread_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping) and context.get(A_THREAD_ENV) is not None:
        return _truthy_value(context.get(A_THREAD_ENV))
    if isinstance(context, Mapping) and context.get("thread_slots_enabled") is not None:
        return _truthy_value(context.get("thread_slots_enabled"))
    return _truthy_value(os.getenv(A_THREAD_ENV))

def _selling_slots_from_contract_and_text(contract: Mapping[str, Any], client_message: str) -> Mapping[str, str]:
    result: dict[str, str] = {}
    for container_name in ("planner_slots", "known_slots"):
        container = contract.get(container_name)
        if not isinstance(container, Mapping):
            continue
        for key, value in container.items():
            name = str(key or "").strip().casefold()
            if name not in {"subject", "grade", "format", "product", "product_family"}:
                continue
            if isinstance(value, Mapping):
                value = value.get("value")
            text = str(value or "").strip()
            if text:
                result["product" if name == "product_family" else name] = text
    text_slots = _selling_slots_from_text(client_message)
    result.update(text_slots)
    return result

def _selling_slots_from_memory(context: Optional[Mapping[str, Any]]) -> Mapping[str, str]:
    if not isinstance(context, Mapping):
        return {}
    result: dict[str, str] = {}
    known = context.get("known_slots")
    if isinstance(known, Mapping):
        _merge_selling_slot_values(result, known)
    memory = context.get("dialogue_memory_view")
    if isinstance(memory, Mapping):
        for key in ("known_slots", "topic_focus"):
            value = memory.get(key)
            if isinstance(value, Mapping):
                _merge_selling_slot_values(result, value)
    return result

def _merge_selling_slot_values(result: dict[str, str], raw: Mapping[str, Any]) -> None:
    for key, value in raw.items():
        name = str(key or "").strip().casefold()
        if name not in {"subject", "grade", "class", "format", "product", "product_family"}:
            continue
        if isinstance(value, Mapping):
            value = value.get("value")
        text = str(value or "").strip()
        if text:
            result.setdefault("grade" if name == "class" else "product" if name == "product_family" else name, text)

def _selling_slots_from_text(text: str) -> Mapping[str, str]:
    value = str(text or "").casefold().replace("ё", "е")
    result: dict[str, str] = {}
    match = re.search(r"\b([1-9]|10|11)\s*(?:класс|классе|кл\.?)", value)
    if match:
        result["grade"] = match.group(1)
    if re.search(r"\bонлайн\b|дистанц", value):
        result["format"] = "онлайн"
    elif re.search(r"\bочн|офлайн|сретен|красносельск", value):
        result["format"] = "очно"
    subject_markers = (
        ("математика", ("математ",)),
        ("физика", ("физик",)),
        ("информатика", ("информат",)),
        ("русский", ("русск",)),
    )
    for label, markers in subject_markers:
        if any(marker in value for marker in markers):
            result["subject"] = label
            break
    if any(marker in value for marker in ("лвш", "лагер", "смен")):
        result["product"] = "camp"
    elif any(marker in value for marker in ("олимпиад", "физтех")):
        result["product"] = "olympiad"
    elif any(marker in value for marker in ("курс", "занят")):
        result["product"] = "regular_course"
    return result

def _text_explicitly_mentions_selling_slot(text: str, key: str) -> bool:
    return key in _selling_slots_from_text(text)

def _phase2_objection_signal(text: str, context: Optional[Mapping[str, Any]]) -> str | None:
    if not _phase2_objection_enabled(context):
        return None
    detected = detect_objection(text)
    return detected if detected in {"price", "think", "compare", "format", "distance"} else None

def _phase2_anxiety_signal(text: str, context: Optional[Mapping[str, Any]]) -> str | None:
    if not _phase2_anxiety_enabled(context):
        return None
    detected = detect_anxiety(text)
    return detected if detected in {"capability", "late_start", "level_fit"} else None

def _merged_selling_signals(
    model_value: object,
    keyword_value: object,
    *,
    phase2_objection: str | None = None,
    phase2_anxiety: str | None = None,
) -> Mapping[str, Any]:
    model = model_value if isinstance(model_value, Mapping) else {}
    keyword = keyword_value if isinstance(keyword_value, Mapping) else {}
    model_objection = str(model.get("objection") or "none").strip().casefold()
    keyword_objection = str(keyword.get("objection") or "none").strip().casefold()
    detector_objection = str(phase2_objection or "none").strip().casefold()
    detector_anxiety = str(phase2_anxiety or "none").strip().casefold()
    allowed_objections = {"price", "think", "compare", "format", "distance"}
    objection = "none"
    for value in (model_objection, keyword_objection, detector_objection):
        if value in allowed_objections:
            objection = value
            break
    model_readiness = str(model.get("readiness") or "exploring").strip().casefold()
    keyword_readiness = str(keyword.get("readiness") or "exploring").strip().casefold()
    if model_readiness not in {"exploring", "comparing", "ready"}:
        model_readiness = "exploring"
    if keyword_readiness not in {"exploring", "comparing", "ready"}:
        keyword_readiness = "exploring"
    unmet_need = str(model.get("unmet_need") or "").strip() or str(keyword.get("unmet_need") or "").strip()
    return {
        "objection": objection,
        "phase2_objection": detector_objection if detector_objection in allowed_objections else "none",
        "exit_signal": bool(model.get("exit_signal")) or bool(keyword.get("exit_signal")),
        "anxiety": bool(model.get("anxiety")) or bool(keyword.get("anxiety")) or detector_anxiety in {"capability", "late_start", "level_fit"},
        "phase2_anxiety": detector_anxiety if detector_anxiety in {"capability", "late_start", "level_fit"} else "none",
        "unmet_need": " ".join(unmet_need.split())[:120],
        "readiness": model_readiness if model_readiness != "exploring" else keyword_readiness,
    }

def _manager_route_migrated_rules_override_allowed(result: SubscriptionDraftResult, *, intent: str) -> bool:
    if intent != "docs":
        return False
    contract = _dialogue_contract_mapping(result)
    haystack = " ".join(
        str(item or "")
        for item in (
            contract.get("current_question"),
            contract.get("client_state"),
            " ".join(str(item or "") for item in (contract.get("composite_subquestions") or ())),
            json.dumps(contract.get("known_slots") or {}, ensure_ascii=False, sort_keys=True),
            json.dumps(contract.get("assertable_slots") or {}, ensure_ascii=False, sort_keys=True),
        )
    )
    return bool(
        re.search(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b", haystack)
        or re.search(r"\b(?:фио|паспорт|снилс|инн|email|e-mail)\b", haystack, re.I)
    )

def _migrated_rules_keep_existing_verified_answer(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
    facts: Mapping[str, str],
) -> bool:
    if result.route not in {"bot_answer_self", "bot_answer_self_for_pilot"}:
        return False
    if not str(result.draft_text or "").strip() or not facts:
        return False
    if _step4_keep_answer_enabled(context):
        if not _keep_answer_supported(result.draft_text, tuple(facts.values())):
            return False
    else:
        if not _claim_supported_by_facts(result.draft_text, tuple(facts.values())):
            return False
    contract = _pipeline_contract(result, active_brand=_active_brand(context), fact_keys=tuple(facts.keys()))
    if getattr(contract, "is_p0", False):
        return False
    findings = verify_dialogue_contract_output(
        result.draft_text,
        facts=facts,
        active_brand=_active_brand(context),
        contract=contract,
        client_message=client_message,
        context=context,
        previous_bot_texts=_humanity_previous_bot_texts(context),
    )
    return not findings

def _pipeline_travel_estimate_applied(result: SubscriptionDraftResult) -> bool:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    estimate = pipeline.get("estimate") if isinstance(pipeline.get("estimate"), Mapping) else {}
    domain = str(estimate.get("estimate_domain") or "").strip()
    return bool(estimate.get("estimate_applied") or estimate.get("is_estimate")) and domain in {"travel_time", "route_logistics"}

def _yield_dispatcher_to_travel_estimate(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    metadata = dict(result.metadata)
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    pipeline = dict(pipeline)
    pipeline["travel_estimate_yielded_dispatcher"] = True
    metadata["dialogue_contract_pipeline"] = pipeline
    return replace(result, metadata=metadata)

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

def _safe_template_yield_result(
    result: SubscriptionDraftResult,
    *,
    spec: SafeTemplateSpec,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult | None:
    if spec.name not in _INFORMATIONAL_SAFE_TEMPLATE_NAMES:
        return None
    if spec.name == "terminal" and not _is_informational_terminal_template(result.draft_text):
        return None
    if spec.name == "terminal" and _is_policy_c_identity_question(result, context=context):
        return None
    if not _verified_informational_answer(result, client_message=client_message, context=context, template_name=spec.name):
        return None
    metadata = {
        **dict(result.metadata),
        "safe_template_yielded_to_verified_answer": True,
        "safe_template_yielded_spec": spec.name,
        "dialogue_contract_v2_template_dispatcher": {
            "yielded": spec.name,
            "priority": spec.priority,
        },
    }
    if _step4_keep_answer_enabled(context):
        metadata = _metadata_with_self_route_deferral_cleared(metadata)
    flags = tuple(dict.fromkeys([*result.safety_flags, "safe_template_yielded_to_verified_answer"]))
    return replace(result, safety_flags=flags, metadata=metadata)

def apply_dialogue_contract_v2_template_dispatcher(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult:
    registry = load_rules_registry()
    result = _with_rules_engine_intent_shadow(
        result,
        shadow=_rules_engine_intent_shadow(result, context=context, registry=registry),
    )
    shadow = result.metadata.get("rules_engine_intent_shadow") if isinstance(result.metadata, Mapping) else {}
    if not isinstance(shadow, Mapping):
        shadow = {}
    if _pipeline_travel_estimate_applied(result):
        trace_event(
            context,
            "safe_template_dispatcher",
            {
                "skipped": "travel_estimate_already_applied",
                "route": result.route,
                "topic_id": result.topic_id,
            },
        )
        return _yield_dispatcher_to_travel_estimate(result)
    if _safe_template_already_applied(result):
        if _safe_template_can_yield_to_dispatcher(result, context=context, shadow=shadow, registry=registry):
            trace_event(
                context,
                "safe_template_dispatcher",
                {
                    "reconsidered": "already_applied",
                    "selected_source": shadow.get("selected_source"),
                    "selected_intent": shadow.get("selected_intent"),
                    "route": result.route,
                    "safety_flags": result.safety_flags,
                },
            )
        else:
            trace_event(
                context,
                "safe_template_dispatcher",
                {
                    "skipped": "already_applied",
                    "route": result.route,
                    "safety_flags": result.safety_flags,
                },
            )
            return result
    migrated = _apply_migrated_rules_engine(result, client_message=client_message, context=context)
    if migrated is not None:
        trace_event(
            context,
            "safe_template_dispatcher",
            {
                "applied": "rules_engine",
                "rule": migrated.metadata.get("rules_engine", {}).get("applied") if isinstance(migrated.metadata.get("rules_engine"), Mapping) else "",
                "route_before": result.route,
                "route_after": migrated.route,
                "draft_text": migrated.draft_text,
            },
        )
        return migrated
    for spec in sorted(DIALOGUE_CONTRACT_V2_TEMPLATE_REGISTRY, key=lambda item: item.priority):
        text = spec.produce(result, client_message, context)
        if text:
            identity_policy_text = spec.name == "terminal" and _is_approved_policy_c_identity_text(
                text,
                active_brand=_active_brand(context),
            )
            if identity_policy_text and not _policy_c_identity_allowed(
                result,
                client_message=client_message,
                context=context,
            ):
                trace_event(
                    context,
                    "safe_template_dispatcher",
                    {
                        "skipped": "identity_policy_blocked_by_p0_or_high_risk",
                        "priority": spec.priority,
                        "route": result.route,
                        "safety_flags": result.safety_flags,
                    },
                )
                continue
            yielded = _safe_template_yield_result(result, spec=spec, client_message=client_message, context=context)
            if yielded is not None:
                trace_event(
                    context,
                    "safe_template_dispatcher",
                    {
                        "applied": "",
                        "yielded": spec.name,
                        "priority": spec.priority,
                        "route": yielded.route,
                        "topic_id": yielded.topic_id,
                    },
                )
                return yielded
            guarded = _apply_safe_template_spec(result, spec, text)
            if identity_policy_text:
                trace_event(
                    context,
                    "safe_template_dispatcher",
                    {
                        "applied": spec.name,
                        "identity_policy_locked": True,
                        "priority": spec.priority,
                        "flag": spec.flag,
                        "route_before": result.route,
                        "route_after": guarded.route,
                        "topic_before": result.topic_id,
                        "topic_after": guarded.topic_id,
                        "draft_text": guarded.draft_text,
                    },
                )
                return guarded
            recovery_candidate = _validated_guardchain_recovery_candidate(
                guarded,
                client_message=client_message,
                context=context,
            )
            if recovery_candidate:
                recovered_flags = tuple(
                    dict.fromkeys([*guarded.safety_flags, "cite_only_recover_at_guardchain"])
                )
                recovered_metadata = {
                    **dict(guarded.metadata),
                    "cite_only_recover_at_guardchain": True,
                    "cite_only_recover_at_guardchain_source": "safe_template_dispatcher",
                }
                if _step4_keep_answer_enabled(context):
                    recovered_metadata = _metadata_with_self_route_deferral_cleared(recovered_metadata)
                recovered = replace(
                    guarded,
                    route="bot_answer_self_for_pilot",
                    draft_text=recovery_candidate,
                    safety_flags=recovered_flags,
                    metadata=recovered_metadata,
                )
                trace_event(
                    context,
                    "safe_template_dispatcher",
                    {
                        "applied": spec.name,
                        "yielded_recovery_candidate": True,
                        "priority": spec.priority,
                        "flag": spec.flag,
                        "route_before": result.route,
                        "route_after": recovered.route,
                        "topic_before": result.topic_id,
                        "topic_after": recovered.topic_id,
                    },
                )
                return recovered
            trace_event(
                context,
                "safe_template_dispatcher",
                {
                    "applied": spec.name,
                    "priority": spec.priority,
                    "flag": spec.flag,
                    "route_before": result.route,
                    "route_after": guarded.route,
                    "topic_before": result.topic_id,
                    "topic_after": guarded.topic_id,
                    "draft_text": guarded.draft_text,
                },
            )
            return guarded
    trace_event(
        context,
        "safe_template_dispatcher",
        {
            "applied": "",
            "route": result.route,
            "topic_id": result.topic_id,
        },
    )
    return result

def _foton_address_template_from_kb(context: Optional[Mapping[str, Any]]) -> str:
    return _direct_path_template_from_fact(
        active_brand="foton",
        fact_key="locations_foton.addresses.1.address",
        literal_text=ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
        neutral_fallback="Адрес московской площадки Фотона лучше уточнит менеджер по выбранному формату.",
        context=context,
        render=lambda text: (
            f"Здравствуйте! В Москве Фотон находится по адресу {_direct_path_fact_value(text)}, метро Красносельская. "
            "Если хотите, подскажу, какие группы есть на этой площадке."
        ),
    )

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

def _unpk_all_addresses_template_from_kb(context: Optional[Mapping[str, Any]]) -> str:
    if not _template_from_kb_enabled(context):
        return ADDRESS_UNPK_SAFE_TEXT
    sretenka = _direct_path_fact_value(
        _direct_path_template_fact_text(active_brand="unpk", fact_key="locations_unpk.addresses.1.address", context=context)
    )
    mfti = _direct_path_fact_value(
        _direct_path_template_fact_text(active_brand="unpk", fact_key="locations_unpk.addresses.2.address", context=context)
    )
    patsayeva = _direct_path_fact_value(
        _direct_path_template_fact_text(active_brand="unpk", fact_key="locations_unpk.addresses.3.address", context=context)
    )
    if not (sretenka and mfti and patsayeva):
        _template_from_kb_trace_event(context, {"fact_key": "locations_unpk.addresses.*.address", "outcome": "fallback"})
        return "Площадки УНПК лучше уточнить у менеджера по выбранному формату."
    _template_from_kb_trace_event(context, {"fact_key": "locations_unpk.addresses.*.address", "outcome": "hit"})
    return f"Площадки УНПК: Москва — {sretenka}; Долгопрудный — {mfti} и {patsayeva}."

def _foton_contact_template_from_kb(context: Optional[Mapping[str, Any]]) -> str:
    if not _template_from_kb_enabled(context):
        return CONTACT_FOTON_SAFE_TEXT
    phone = _direct_path_fact_value(
        _direct_path_template_fact_text(active_brand="foton", fact_key="contacts_foton.phone", context=context)
    )
    toll_free = _direct_path_fact_value(
        _direct_path_template_fact_text(active_brand="foton", fact_key="contacts_foton.toll_free", context=context)
    )
    email = _direct_path_fact_value(
        _direct_path_template_fact_text(active_brand="foton", fact_key="contacts_foton.email", context=context)
    )
    if not (phone and toll_free and email):
        _template_from_kb_trace_event(context, {"fact_key": "contacts_foton.phone+toll_free+email", "outcome": "fallback"})
        return "Актуальные контакты Фотона лучше уточнить у менеджера."
    _template_from_kb_trace_event(context, {"fact_key": "contacts_foton.phone+toll_free+email", "outcome": "hit"})
    return f"Телефоны: {phone} и {toll_free}. Email: {email}. График: Пн-Вс с 10:00 до 18:00."

def _unpk_contact_template_from_kb(context: Optional[Mapping[str, Any]]) -> str:
    if not _template_from_kb_enabled(context):
        return CONTACT_UNPK_SAFE_TEXT
    phone = _direct_path_fact_value(
        _direct_path_template_fact_text(active_brand="unpk", fact_key="contacts_unpk.phone", context=context)
    )
    toll_free = _direct_path_fact_value(
        _direct_path_template_fact_text(active_brand="unpk", fact_key="contacts_unpk.toll_free", context=context)
    )
    email = _direct_path_fact_value(
        _direct_path_template_fact_text(active_brand="unpk", fact_key="contacts_unpk.email", context=context)
    )
    if not (phone and toll_free and email):
        _template_from_kb_trace_event(context, {"fact_key": "contacts_unpk.phone+toll_free+email", "outcome": "fallback"})
        return "Актуальные контакты УНПК лучше уточнить у менеджера."
    _template_from_kb_trace_event(context, {"fact_key": "contacts_unpk.phone+toll_free+email", "outcome": "hit"})
    return f"Телефоны: {phone} и {toll_free}. Email: {email}. График: Пн-Вс с 10:00 до 18:00."

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
    facts_context.update(
        {
            "stale": False,
            "facts_stale": False,
            "fresh": True,
            "facts_fresh": True,
            "fresh_facts": True,
            "client_safe_fact_verified": True,
            "confirmed_facts": facts_context_confirmed,
        }
    )

    quality = dict(merged.get("context_quality")) if isinstance(merged.get("context_quality"), Mapping) else {}
    quality["facts_stale"] = False

    merged.update(
        {
            "confirmed_facts": confirmed,
            "dialogue_contract_pipeline": merged_pipeline,
            "facts_context": facts_context,
            "context_quality": quality,
            "facts_fresh": True,
            "facts_stale": False,
        }
    )
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
    if _step4_keep_answer_enabled(context):
        if not _keep_answer_supported(result.draft_text, tuple(fact_texts.values())):
            return False
    else:
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

def _safe_template_applied_name(result: SubscriptionDraftResult) -> str:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    dispatcher = metadata.get("dialogue_contract_v2_template_dispatcher")
    if isinstance(dispatcher, Mapping):
        return str(dispatcher.get("applied") or dispatcher.get("yielded") or "").strip()
    for spec in DIALOGUE_CONTRACT_V2_TEMPLATE_REGISTRY:
        if spec.flag in result.safety_flags or metadata.get(spec.flag):
            return spec.name
    return ""

def _has_informational_safe_template(result: SubscriptionDraftResult) -> bool:
    if _safe_template_applied_name(result) in _INFORMATIONAL_SAFE_TEMPLATE_NAMES:
        return True
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    return any(
        spec.name in _INFORMATIONAL_SAFE_TEMPLATE_NAMES
        and (spec.flag in result.safety_flags or metadata.get(spec.flag))
        for spec in DIALOGUE_CONTRACT_V2_TEMPLATE_REGISTRY
    )

def _manager_only_recovery_yield_allowed(result: SubscriptionDraftResult, *, client_message: str) -> bool:
    applied = _safe_template_applied_name(result)
    if applied == "terminal":
        text = str(client_message or "").casefold().replace("ё", "е")
        if has_any_marker(
            text,
            (
                "ты бот",
                "вы бот",
                "нейросеть",
                "живой человек",
                "живой оператор",
                "с кем я общаюсь",
                "ignore all previous",
                "system prompt",
                "системный промпт",
                "покажи промпт",
                "chatgpt",
                "gpt",
                "openai",
                "claude",
                "codex",
            ),
        ):
            return False
        return has_any_marker(text, ("личный кабинет", "кабинет", "платформ", "зайти", "войти", "доступ"))
    return applied in _INFORMATIONAL_SAFE_TEMPLATE_NAMES or _has_informational_safe_template(result)

def _validated_guardchain_recovery_candidate(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> str:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    candidate = str(pipeline.get("recovery_candidate") or "").strip()
    fact_texts = _pipeline_fact_texts(result)
    contract = _pipeline_contract(result, active_brand=_active_brand(context), fact_keys=tuple(fact_texts.keys()))
    if not candidate:
        candidate = _recovery_candidate_from_informational_facts(
            result,
            contract=contract,
            fact_texts=fact_texts,
            client_message=client_message,
            context=context,
        )
    elif pipeline.get("recovery_candidate_validated") is not True:
        return ""
    if not candidate:
        return ""
    if (
        _safe_template_applied_name(result)
        and result.route in {"manager_only", "draft_for_manager"}
        and not _manager_only_recovery_yield_allowed(result, client_message=client_message)
    ):
        return ""
    if is_high_risk_result(result) or set(detect_high_risk_input_markers(client_message, context=context)):
        return ""
    flags = set(result.safety_flags)
    if flags.intersection(_GUARDCHAIN_RECOVERY_BLOCKING_FLAGS):
        return ""
    if any(bool(metadata.get(flag)) for flag in _GUARDCHAIN_RECOVERY_BLOCKING_FLAGS):
        return ""

    if not fact_texts:
        return ""
    if contract.is_p0:
        return ""
    findings = verify_dialogue_contract_output(
        candidate,
        facts=fact_texts,
        active_brand=_active_brand(context),
        contract=contract,
        client_message=client_message,
        context=context,
        previous_bot_texts=_humanity_previous_bot_texts(context),
    )
    if findings:
        return ""
    return candidate

def _recovery_candidate_from_informational_facts(
    result: SubscriptionDraftResult,
    *,
    contract: Any,
    fact_texts: Mapping[str, str],
    client_message: str,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    if not fact_texts or getattr(contract, "is_p0", False) or getattr(contract, "answerability", "") != "answer_self":
        return ""
    selected: list[str] = []
    for key, text in fact_texts.items():
        if _informational_fact_matches_question(
            key,
            text,
            result=result,
            contract=contract,
            client_message=client_message,
        ):
            cleaned = _client_clean_fact_text(text)
            if cleaned:
                selected.append(cleaned)
    if not selected:
        return ""
    unique = list(dict.fromkeys(selected))
    if len(unique) == 1:
        return apply_warm_frame(f"По подтверждённым данным: {unique[0]}", context=context, kind="informational_recovery")
    return apply_warm_frame(
        "По подтверждённым данным: " + " ".join(unique[:3]),
        context=context,
        kind="informational_recovery",
    )

def _informational_fact_matches_question(
    key: str,
    text: str,
    *,
    result: SubscriptionDraftResult,
    contract: Any,
    client_message: str,
) -> bool:
    haystack = " ".join(
        [
            str(key or ""),
            str(text or ""),
            str(result.topic_id or ""),
            str(getattr(contract, "current_question", "") or ""),
            str(client_message or ""),
        ]
    ).casefold().replace("ё", "е")
    key_text = str(key or "").casefold().replace("ё", "е")
    question = " ".join(
        [
            str(getattr(contract, "current_question", "") or ""),
            str(client_message or ""),
            str(result.topic_id or ""),
        ]
    ).casefold().replace("ё", "е")
    tax_scope = "tax" in key_text or has_any_marker(haystack, ("налог", "вычет", "фнс", "кнд"))
    if tax_scope and has_any_marker(question, ("налог", "вычет", "фнс", "кнд")):
        if has_any_marker(question, ("сумм", "сколько", "13%", "110 000", "14 300", "верн", "возврат", "точно", "гарант", "одобр")):
            return False
        return True
    matkap_scope = "matkap" in key_text or has_any_marker(haystack, ("маткап", "материнск", "сфр", "сертификат"))
    if matkap_scope and has_any_marker(question, ("маткап", "материнск", "сфр", "сертификат")):
        if has_any_marker(question, ("одобр", "точно", "гарант", "региональ")):
            return False
        return True
    checks = (
        (("trial" in key_text or has_any_marker(haystack, ("пробн", "фрагмент занятия", "фрагмент урок"))), ("пробн", "фрагмент", "посмотреть занят", "посмотреть урок")),
        (("olympiad" in key_text or "phystech" in key_text or has_any_marker(haystack, ("олимпиад", "физтех"))), ("олимпиад", "физтех")),
        (("platform" in key_text or "cabinet" in key_text or has_any_marker(haystack, ("личный кабинет", "учебн", "платформ"))), ("личный кабинет", "кабинет", "платформ", "зайти", "войти")),
    )
    return any(scope_ok and has_any_marker(question, question_markers) for scope_ok, question_markers in checks)

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

def apply_input_policy_guards(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    markers = detect_high_risk_input_markers(client_message, context=context)
    if not markers:
        return result
    autonomy_flags = ("autonomy_blocked_high_risk",) if result.route in AUTONOMOUS_ROUTES else ()
    flags = tuple(dict.fromkeys([*result.safety_flags, "high_risk_input_manager_only", "high_risk_manager_only", *autonomy_flags]))
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "Исходное сообщение клиента содержит высокорисковую тему: проверить вручную.",
            ]
        )
    )
    return replace(
        result,
        route="manager_only",
        safety_flags=flags,
        manager_checklist=checklist,
        metadata={
            **dict(result.metadata),
            "forced_route_high_risk_input": list(markers),
            **({"autonomy_blocked_high_risk": True} if autonomy_flags else {}),
        },
    )

def apply_high_risk_content_guards(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    safety_decision = classify_answer_safety(
        client_message=client_message,
        context=context,
        topic_id=result.topic_id,
        route=result.route,
        safety_flags=result.safety_flags,
    )
    markers = (
        set(safety_decision.risk_codes)
        if safety_decision.p0_required
        else {code for code in safety_decision.risk_codes if code in HARD_P0_CODES}
    )
    markers = set(_p0_model_led_filter_high_risk_codes(tuple(markers), client_message=client_message, context=context))
    model_p0_meta = result.metadata.get("direct_path_model_p0") if isinstance(result.metadata, Mapping) else {}
    model_p0_complaint = bool(
        isinstance(model_p0_meta, Mapping)
        and model_p0_meta.get("is_p0")
        and str(model_p0_meta.get("p0_kind") or "").strip() == "complaint"
    )
    p0_model_led_suppressed_complaint = bool(
        _p0_model_led_enabled(context)
        and safety_decision.primary_risk in {"complaint", "reputation_threat"}
        and not _p0_model_led_complaint_backstop(client_message)
        and not model_p0_complaint
    )
    topic = str(result.topic_id or "").strip()
    flags = list(result.safety_flags)
    checklist = list(result.manager_checklist)
    metadata = dict(result.metadata)
    metadata["answer_safety"] = safety_decision.to_json_dict()
    if p0_model_led_suppressed_complaint:
        metadata["p0_model_led_complaint_suppressed"] = True
    forbidden_promises = list(result.forbidden_promises_detected)
    route = result.route
    draft_text = result.draft_text
    semantic_non_p0 = _conversation_plan_semantic_non_p0(context, client_message=client_message)
    skip_quality_template_overwrite = (
        _answer_quality_was_rewritten(result)
        and not markers
        and topic not in HIGH_RISK_THEME_IDS
    )
    contract = _answer_contract(context)
    contract_primary_intent = str(contract.get("primary_intent") or "").strip()
    answer_contract_controls_green_templates = bool(
        _answer_contract_green_template_reduction_enabled(context)
        and contract
        and not contract.get("p0_required")
        and contract.get("must_answer_first")
        and _draft_addresses_question(
            _normalize_for_template_decision(result.draft_text),
            str(contract.get("direct_question") or client_message or ""),
            intent=contract_primary_intent,
        )
        and (
            contract_primary_intent
            in {
                "schedule",
                "format",
                "address",
                "identity",
                "general_consultation",
            }
            or (contract_primary_intent == "installment" and _active_brand(context) == "unpk")
        )
    )
    conversation_plan_controls_green_templates = _conversation_plan_controls_green_templates(
        result,
        context=context,
        client_message=client_message,
        risk_markers=markers,
    )
    if answer_contract_controls_green_templates:
        metadata["answer_contract_controls_green_templates"] = True
    if conversation_plan_controls_green_templates:
        metadata["conversation_plan_controls_green_templates"] = True
    skip_green_template_overwrite = (
        skip_quality_template_overwrite
        or answer_contract_controls_green_templates
        or conversation_plan_controls_green_templates
        or _conversation_plan_template_blocked_by_substantive_answer(result, context=context)
    )

    if not skip_green_template_overwrite and _is_unpk_installment_case(result, client_message=client_message, context=context):
        route = result.route if result.route in AUTONOMOUS_ROUTES else "draft_for_manager"
        draft_text = UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT
        flags.append("unpk_installment_approved_fallback_applied")
        checklist.append("Проверить, что вопрос относится к УНПК и к оплате частями/по периодам.")
        metadata["unpk_installment_approved_fallback_applied"] = True

    if not skip_quality_template_overwrite and _is_unpk_zvsh_case(result, client_message=client_message, context=context):
        route = "manager_only" if route == "manager_only" else "draft_for_manager"
        draft_text = UNPK_ZVSH_WAITLIST_SAFE_TEXT
        flags.append("unpk_zvsh_waitlist_safe_template_applied")
        checklist.append("ЗВШ Менделеево: не обещать даты, использовать лист ожидания.")
        metadata["unpk_zvsh_waitlist_safe_template_applied"] = True

    cross_brand_template = _cross_brand_safe_template(result, client_message=client_message, context=context)
    if cross_brand_template:
        route = "draft_for_manager" if cross_brand_template == UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT else "manager_only"
        draft_text = cross_brand_template
        flags.append("cross_brand_safe_template_applied")
        checklist.append("Кросс-бренд: не консультировать по другому бренду и не сравнивать условия.")
        metadata["cross_brand_safe_template_applied"] = True

    def cross_brand_guarded() -> bool:
        return bool(
            metadata.get("cross_brand_safe_template_applied")
            or "cross_brand_safe_template_applied" in flags
            or "brand_separation_guarded" in flags
        )

    if not skip_quality_template_overwrite and not cross_brand_guarded() and _is_future_price_case(result, client_message=client_message, context=context):
        route = "manager_only"
        draft_text = "Передам вопрос менеджеру: он проверит актуальную стоимость на нужный период и свяжется с вами."
        flags.extend(("future_price_handoff_applied", "manager_approval_required", "no_auto_send"))
        checklist.append("Будущая цена: не называть суммы после повышения, передать менеджеру.")
        metadata["future_price_handoff_applied"] = True

    terminal_template = "" if cross_brand_guarded() else _terminal_safe_template(result, client_message=client_message, context=context)
    if terminal_template:
        direct_info_template = terminal_template in {
            ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
            ADDRESS_UNPK_SAFE_TEXT,
            ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
            CONTACT_FOTON_SAFE_TEXT,
            CONTACT_UNPK_SAFE_TEXT,
            IDENTITY_PROMPT_SAFE_TEXT,
            IDENTITY_FOTON_SAFE_TEXT,
            IDENTITY_UNPK_SAFE_TEXT,
            OFF_TOPIC_FOTON_SAFE_TEXT,
            OFF_TOPIC_UNPK_SAFE_TEXT,
            OFF_TOPIC_GENERIC_SAFE_TEXT,
            SOFT_NEGATIVE_HANDOFF_SAFE_TEXT,
        }
        green_terminal_template = terminal_template in {
            ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
            ADDRESS_UNPK_SAFE_TEXT,
            ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
            CONTACT_FOTON_SAFE_TEXT,
            CONTACT_UNPK_SAFE_TEXT,
        }
        if green_terminal_template and skip_green_template_overwrite:
            terminal_template = ""
            metadata["terminal_green_template_skipped_by_answer_contract"] = True
    if terminal_template:
        direct_info_template = terminal_template in {
            ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
            ADDRESS_UNPK_SAFE_TEXT,
            ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
            CONTACT_FOTON_SAFE_TEXT,
            CONTACT_UNPK_SAFE_TEXT,
            IDENTITY_PROMPT_SAFE_TEXT,
            IDENTITY_FOTON_SAFE_TEXT,
            IDENTITY_UNPK_SAFE_TEXT,
            OFF_TOPIC_FOTON_SAFE_TEXT,
            OFF_TOPIC_UNPK_SAFE_TEXT,
            OFF_TOPIC_GENERIC_SAFE_TEXT,
            SOFT_NEGATIVE_HANDOFF_SAFE_TEXT,
        }
        route = "draft_for_manager" if direct_info_template else "manager_only"
        draft_text = terminal_template
        flags.append("terminal_safe_template_applied")
        if not direct_info_template:
            flags.append("placeholder_in_draft")
        if "всош" in str(client_message or "").casefold():
            flags.append("unpk_zvsh_waitlist_safe_template_applied")
        if direct_info_template:
            checklist.append("Справочный вопрос: проверить, что ответ относится к активному бренду.")
        else:
            checklist.append("Вопрос требует безопасной служебной обработки без раскрытия лишних данных.")
        metadata["terminal_safe_template_applied"] = True
        metadata["terminal_direct_info_template_applied"] = direct_info_template

    presale_refund_template = (
        ""
        if cross_brand_guarded()
        else _presale_refund_policy_template(result, client_message=client_message, context=context)
    )
    if presale_refund_template:
        route = "bot_answer_self_for_pilot"
        topic = "service:S5_general_consultation"
        draft_text = presale_refund_template
        flags.extend(("presale_refund_policy_manager_check", "presale_refund_policy_non_p0"))
        checklist.append("Предпродажный вопрос о возврате: не оформлять как жалобу/P0, условия подтверждает менеджер.")
        metadata["presale_refund_policy_manager_check"] = True
        metadata["presale_refund_policy_non_p0"] = True

    offline_free_trial_guard = "" if cross_brand_guarded() else _foton_offline_free_trial_guard_template(
        result,
        client_message=client_message,
        context=context,
    )
    if offline_free_trial_guard:
        route = "bot_answer_self_for_pilot"
        topic = "theme:023_trial_class"
        draft_text = offline_free_trial_guard
        flags.append("offline_free_trial_promise_guarded")
        checklist.append("Фотон: не обещать бесплатное очное пробное по умолчанию; условия подтверждает менеджер.")
        metadata["offline_free_trial_promise_guarded"] = True

    if _is_admission_guarantee_case(result, client_message=client_message, context=context):
        route = "manager_only" if route == "manager_only" else "draft_for_manager"
        draft_text = ADMISSION_GUARANTEE_SAFE_TEXT
        flags.extend(("admission_guarantee_safe_template_applied", "placeholder_in_draft"))
        checklist.append("Не гарантировать поступление: можно говорить только про программу и статистику.")
        metadata["admission_guarantee_safe_template_applied"] = True
    elif _is_result_guarantee_case(result, client_message=client_message, context=context):
        route = "manager_only" if route == "manager_only" else "draft_for_manager"
        draft_text = RESULT_GUARANTEE_SAFE_TEXT
        flags.extend(("result_guarantee_safe_template_applied", "placeholder_in_draft"))
        checklist.append("Не гарантировать баллы или поступление: можно говорить только про программу и статистику.")
        metadata["result_guarantee_safe_template_applied"] = True

    scope_missing_guard_template = _scope_fact_missing_guard_template(
        replace(result, route=route, draft_text=draft_text, safety_flags=tuple(dict.fromkeys(flags)), metadata=metadata),
        context=context,
    )
    if scope_missing_guard_template and not cross_brand_guarded():
        route = "draft_for_manager"
        draft_text = scope_missing_guard_template
        flags.append("scope_fact_guard_applied")
        checklist.append("Нужного факта по текущему вопросу нет: не подставлять соседнюю область, передать менеджеру узко по недостающей детали.")
        metadata["scope_fact_guard_applied"] = True

    scope_guard_template = "" if metadata.get("scope_fact_guard_applied") else _fact_scope_guard_template(draft_text, context=context)
    if scope_guard_template and not cross_brand_guarded():
        route = "draft_for_manager"
        draft_text = scope_guard_template
        flags.append("fact_scope_guard_applied")
        checklist.append("Ответ ссылался на соседнюю область фактов: не подставлять её вместо спрошенной темы.")
        metadata["fact_scope_guard_applied"] = True

    forbidden_pair_template = _forbidden_pair_guard_template(draft_text, context=context)
    if forbidden_pair_template and not cross_brand_guarded():
        route = "draft_for_manager"
        draft_text = forbidden_pair_template
        flags.append("forbidden_pair_guard_applied")
        checklist.append("План ответа запретил смешивать разные оси в одном ответе.")
        metadata["forbidden_pair_guard_applied"] = True

    if (
        not cross_brand_guarded()
        and not metadata.get("terminal_safe_template_applied")
        and not metadata.get("result_guarantee_safe_template_applied")
        and not metadata.get("admission_guarantee_safe_template_applied")
        and not _is_reputation_only_case(markers=markers)
        and not (semantic_non_p0 and "legal" not in markers)
        and safety_decision.primary_risk == "legal"
        and _is_legal_threat_case(result, markers=markers)
    ):
        if topic != "theme:029_legal_question":
            topic = "theme:029_legal_question"
            flags.append("legal_threat_topic_overrode_refund")
            metadata["legal_threat_topic_overrode"] = result.topic_id
        else:
            flags.append("legal_threat_topic_overrode_refund")
        route = "manager_only"
        draft_text = LEGAL_THREAT_PII_SAFE_TEXT if _client_message_contains_pii(client_message) else LEGAL_THREAT_SAFE_TEXT
        flags.extend(("zero_collect_legal_guarded", "high_risk_manager_only"))
        checklist.append("Юридическая угроза: не собирать данные у клиента, передать ответственному сотруднику.")
        metadata["zero_collect_legal_guarded"] = True

    if (
        not metadata.get("zero_collect_legal_guarded")
        and not metadata.get("tax_safe_template_applied")
        and not metadata.get("matkap_safe_template_applied")
        and not metadata.get("presale_refund_policy_manager_check")
        and not (semantic_non_p0 and safety_decision.primary_risk == "refund")
        and not (semantic_non_p0 and "refund" not in markers)
        and _is_refund_case(result, markers=markers)
    ):
        route = "manager_only"
        draft_text = _p0_text_with_antirepeat("refund", REFUND_ZERO_COLLECT_SAFE_TEXT, context)
        flags.extend(("zero_collect_refund_guarded", "high_risk_manager_only"))
        checklist.append("Возврат: не собирать ФИО, договор, оплату, телефон, email, сумму или причину в черновике.")
        metadata["zero_collect_refund_guarded"] = True

    if (
        not metadata.get("zero_collect_legal_guarded")
        and not p0_model_led_suppressed_complaint
        and not (semantic_non_p0 and "complaint" not in markers and "reputation_threat" not in markers)
        and _is_complaint_case(result, markers=markers)
    ):
        if "reputation_threat" in markers and "legal" not in markers:
            topic = "theme:019b_negative_feedback"
        route = "manager_only"
        draft_text = _p0_text_with_antirepeat("complaint", COMPLAINT_SAFE_TEXT, context)
        flags.extend(("complaint_apology_guarded", "high_risk_manager_only"))
        checklist.append("Жалоба: не извиняться от лица компании и не признавать вину в авточерновике.")
        metadata["complaint_apology_guarded"] = True

    has_concrete_safe_template = bool(
        metadata.get("camp_safe_template_applied")
        and re.search(r"\b\d{1,3}(?:[ \u00a0]\d{3})+\s*₽", str(draft_text or ""))
    )
    missing_fact_template = (
        ""
        if metadata.get("terminal_safe_template_applied")
        or metadata.get("fact_scope_guard_applied")
        or metadata.get("forbidden_pair_guard_applied")
        or metadata.get("schedule_confirmation_safe_template_applied")
        or skip_green_template_overwrite
        or has_concrete_safe_template
        or _skip_missing_fact_template_by_answer_contract(
            context,
            answer_contract_controls_green_templates=answer_contract_controls_green_templates,
        )
        else _missing_fact_helpful_template(
            replace(result, topic_id=topic, route=route, draft_text=draft_text, safety_flags=tuple(dict.fromkeys(flags))),
            client_message=client_message,
            context=context,
            markers=markers,
        )
    )
    if missing_fact_template:
        route = "draft_for_manager"
        draft_text = missing_fact_template
        flags.append("missing_fact_helpful_template_applied")
        checklist.append("Точного факта нет: дать полезный общий ответ, задать безопасные уточнения и не обещать точные условия.")
        metadata["missing_fact_helpful_template_applied"] = True

    if route == "manager_only" and _is_combined_high_risk_case(
        result,
        markers=markers,
        client_message=client_message,
        context=context,
    ):
        flags.append("combined_high_risk_manager_only")
        checklist.append("Комбинированный вопрос с high-risk частью: не отвечать на безопасную часть отдельно.")
        metadata["combined_high_risk_manager_only"] = True

    softened_deadline_text = _soften_current_price_deadline_text(draft_text, client_message=client_message)
    if softened_deadline_text != draft_text:
        draft_text = softened_deadline_text
        flags.append("current_price_deadline_softened")
        metadata["current_price_deadline_softened"] = True
    softened_checklist = [
        _soften_current_price_deadline_text(item, client_message=client_message)
        for item in checklist
    ]
    if softened_checklist != checklist:
        checklist = softened_checklist
        flags.append("current_price_deadline_softened")
        metadata["current_price_deadline_softened"] = True

    if (
        "unsupported_promise_detected" in flags
        and "brand_unknown_precise_condition_blocked" not in flags
        and not find_unsupported_numeric_promises(draft_text, context=context)
    ):
        flags = [flag for flag in flags if flag != "unsupported_promise_detected"]
        unsupported_claims = {str(item) for item in metadata.get("unsupported_promises", []) or []}
        if unsupported_claims:
            forbidden_promises = [item for item in forbidden_promises if str(item) not in unsupported_claims]
        metadata.pop("unsupported_promises", None)
        metadata["unsupported_promise_resolved_by_softener"] = True
        flags.append("unsupported_promise_resolved_by_softener")
        flag_text = " ".join(flags).casefold()
        if route == "manager_only" and not any(
            marker in flag_text
            for marker in (
                "zero_collect",
                "legal",
                "complaint",
                "payment_confirmation",
                "brand_separation",
                "cross_brand",
                "future_price_handoff",
                "high_risk",
            )
        ):
            route = "draft_for_manager"

    if (
        safety_decision.p0_required
        and not p0_model_led_suppressed_complaint
        and not safety_decision.semantic_non_p0
        and not metadata.get("presale_refund_policy_manager_check")
    ):
        primary_risk = safety_decision.primary_risk
        route = "manager_only"
        if primary_risk == "legal":
            topic = "theme:029_legal_question"
            draft_text = LEGAL_THREAT_PII_SAFE_TEXT if _client_message_contains_pii(client_message) else LEGAL_THREAT_SAFE_TEXT
            draft_text = _p0_text_with_antirepeat("legal", draft_text, context)
            flags.extend(("final_p0_text_override", "zero_collect_legal_guarded", "high_risk_manager_only"))
            checklist.append("Финальный P0 override: юридическая угроза, не собирать данные и не продавать.")
        elif primary_risk == "refund":
            topic = "theme:009_refund"
            draft_text = _p0_text_with_antirepeat("refund", REFUND_ZERO_COLLECT_SAFE_TEXT, context)
            flags.extend(("final_p0_text_override", "zero_collect_refund_guarded", "high_risk_manager_only"))
            checklist.append("Финальный P0 override: возврат, не собирать ФИО/договор/телефон/email/сумму/причину.")
        elif primary_risk in {"complaint", "reputation_threat"}:
            topic = "theme:019b_negative_feedback"
            draft_text = _p0_text_with_antirepeat("complaint", COMPLAINT_SAFE_TEXT, context)
            flags.extend(("final_p0_text_override", "complaint_apology_guarded", "high_risk_manager_only"))
            checklist.append("Финальный P0 override: жалоба/негатив, без извинений от лица компании и без признания вины.")
        elif primary_risk == "payment_dispute":
            topic = "theme:003_payment_status"
            draft_text = _p0_text_with_antirepeat("payment_dispute", PAYMENT_DISPUTE_SAFE_TEXT, context)
            flags.extend(("final_p0_text_override", "payment_dispute_manager_only", "high_risk_manager_only"))
            checklist.append("Финальный P0 override: спор по оплате, проверка только менеджером по системе.")
        metadata["final_p0_text_override"] = primary_risk or True

    if draft_text != result.draft_text:
        metadata = _metadata_with_guarded_original_text(metadata, result.draft_text, guard="high_risk_content_guards")

    if (
        route == result.route
        and draft_text == result.draft_text
        and tuple(flags) == result.safety_flags
        and tuple(forbidden_promises) == result.forbidden_promises_detected
        and metadata == result.metadata
    ):
        return result
    return replace(
        result,
        topic_id=topic,
        route=route,
        draft_text=draft_text,
        forbidden_promises_detected=tuple(dict.fromkeys(forbidden_promises)),
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(checklist)),
        metadata=metadata,
    )

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
    if _result_has_live_status_missing_fact(result, client_message=client_message) and not _is_verified_client_safe_template(result.draft_text):
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

    flags.append("autonomy_matrix_passed")
    metadata["autonomy_matrix_passed"] = True
    draft_text = result.draft_text
    if _draft_is_low_value_without_exact_fact(draft_text) and not _is_verified_client_safe_template(draft_text):
        fact_answer = _promoted_verified_fact_text(result, context=context, client_message=client_message)
        if fact_answer:
            draft_text = fact_answer
            flags.append("autonomy_verified_fact_answer_template_applied")
            metadata["autonomy_verified_fact_answer_template_applied"] = True
    if original_route == "draft_for_manager":
        flags.append("autonomy_matrix_promoted_safe_draft")
        checklist.append("Зелёная тема с проверенным клиентским фактом: можно отвечать автономно в пилотном режиме.")
        metadata["autonomy_matrix_promoted_safe_draft"] = True
    return replace(
        result,
        route="bot_answer_self_for_pilot" if original_route == "draft_for_manager" else result.route,
        draft_text=draft_text,
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

def _result_has_live_status_missing_fact(result: SubscriptionDraftResult, *, client_message: str = "") -> bool:
    client_text = str(client_message or "").casefold()
    if not _asks_live_status_or_booking_question(client_text):
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
    """Align draft topic/route with the context-level conversation plan.

    The plan is deliberately higher level than keyword rules: words such as
    "закрепить" or "бронь" may mean different things depending on the product
    focus. This guard uses the plan as an internal contract and never weakens
    P0, brand or fact-safety guards.
    """

    plan = _conversation_intent_plan(context)
    if not plan:
        return result

    primary_intent = str(plan.get("primary_intent") or "").strip()
    plan_topic = str(plan.get("topic_id") or "").strip()
    answer_policy = str(plan.get("answer_policy") or "").strip()
    route_bias = str(plan.get("route_bias") or "").strip()
    route = result.route
    topic = str(result.topic_id or "").strip()
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

    if high_risk_plan:
        if topic_from_plan:
            topic = plan_topic
        route = "manager_only"
        flags.extend(("conversation_intent_plan_p0", "high_risk_manager_only"))
        checklist.append("План диалога распознал P0/high-risk тему: автономный ответ запрещён.")
        metadata["conversation_intent_plan_route_applied"] = "manager_only"

    elif primary_intent == "live_availability":
        if topic_from_plan and topic != plan_topic:
            topic = plan_topic
            flags.append("conversation_intent_plan_topic_applied")
            metadata["conversation_intent_plan_topic_from"] = result.topic_id
        if route in AUTONOMOUS_ROUTES:
            route = "draft_for_manager"
            flags.append("conversation_intent_plan_live_check_handoff")
            metadata["conversation_intent_plan_route_applied"] = "draft_for_manager"
        flags.append("conversation_intent_plan_live_availability")
        checklist.append(
            "План диалога: вопрос про место/наличие/бронь требует проверки менеджером; не обещать место до проверки."
        )

    elif topic_from_plan and topic != plan_topic and (not is_high_risk_result(result) or semantic_non_p0):
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

    if (
        route == result.route
        and topic == result.topic_id
        and tuple(flags) == result.safety_flags
        and tuple(checklist) == result.manager_checklist
        and metadata == result.metadata
    ):
        return result

    return replace(
        result,
        topic_id=topic,
        route=route,
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(checklist)),
        metadata=metadata,
    )

def _conversation_intent_plan(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    plan = context.get("conversation_intent_plan")
    return plan if isinstance(plan, Mapping) else {}

def _answer_contract(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    contract = context.get("answer_contract")
    return contract if isinstance(contract, Mapping) else {}

def _answer_contract_green_template_reduction_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping):
        value = context.get("answer_contract_green_template_reduction_enabled")
        if value is not None:
            return _truthy_value(value)
    env_value = os.getenv(ANSWER_CONTRACT_GREEN_TEMPLATE_REDUCTION_ENV)
    if env_value is not None:
        return _truthy_value(env_value)
    return True

def _conversation_plan_controls_green_templates(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
    client_message: str,
    risk_markers: set[str],
) -> bool:
    existing_flags = " ".join(str(item) for item in result.safety_flags)
    manager_only_due_to_numeric_review = result.route == "manager_only" and "unsupported_promise_detected" in existing_flags
    if risk_markers or result.topic_id in HIGH_RISK_THEME_IDS or (result.route == "manager_only" and not manager_only_due_to_numeric_review):
        return False
    plan = _conversation_intent_plan(context)
    intent = str(plan.get("primary_intent") or "").strip()
    if intent not in {
        "pricing",
        "price_fix",
        "installment",
        "discount",
        "trial",
        "camp",
        "schedule",
        "format",
        "address",
        "document",
        "matkap",
        "tax",
        "general_consultation",
    }:
        return False
    if str(plan.get("answer_policy") or "") not in {
        "answer_directly_if_fact_verified",
        "answer_safe_parts_then_manager_live_check",
        "help_then_one_question",
    }:
        return False
    draft = _normalize_for_template_decision(result.draft_text)
    if not draft:
        return False
    if _looks_like_low_value_handoff_only(draft):
        return False
    question = str(plan.get("direct_question") or client_message or "")
    scope = str(plan.get("fact_scope") or "")
    if scope:
        answer_scopes = detect_fact_scopes(draft)
        if answer_scopes and not answer_scopes_allowed(
            answer_scopes,
            requested_scope=scope,
            blocked_neighbor_scopes=tuple(str(item) for item in plan.get("blocked_neighbor_scopes", ()) or ()),
        ):
            return False
        if answer_scopes and scope in answer_scopes and len(draft) >= 60:
            return True
    if _draft_addresses_question(draft, question, intent=intent):
        return True
    return bool(scope and len(draft) >= 120 and not _looks_like_generic_template(draft))

def _conversation_plan_template_blocked_by_substantive_answer(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]],
) -> bool:
    plan = _conversation_intent_plan(context)
    if plan.get("template_allowed") is not False:
        return False
    if str(plan.get("route_bias") or "") == "manager_only" or plan.get("p0_required") is True:
        return False
    draft = _normalize_for_template_decision(result.draft_text)
    if not draft or _looks_like_low_value_handoff_only(draft):
        return False
    answer_topics = tuple(str(item) for item in plan.get("answer_topics", ()) or () if str(item).strip())
    return bool(answer_topics and len(draft) >= 80)

def _normalize_for_template_decision(value: object) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").split())

def _looks_like_low_value_handoff_only(text: str) -> bool:
    if len(text) >= 140 and not _looks_like_generic_template(text):
        return False
    return has_any_marker(text, ("передам вопрос менеджеру", "менеджер подскажет", "менеджер проверит")) and not any(
        marker in text
        for marker in ("₽", "руб", "%", "долями", "рассроч", "скид", "пробн", "фрагмент", "адрес", "сретен", "красносельск", "пацаева")
    )

def _looks_like_generic_template(text: str) -> bool:
    generic = (
        "чтобы не назвать невер",
        "нужно уточнить у менеджера",
        "менеджер подскажет варианты под вашу ситуацию",
        "передам менеджеру запрос на точную проверку",
    )
    return has_any_marker(text, generic)

def _draft_addresses_question(draft_text: str, question: str, *, intent: str) -> bool:
    question_norm = _normalize_for_template_decision(question)
    if not question_norm:
        return False
    if intent in {"pricing", "price_fix"}:
        return bool(PRICE_AMOUNT_RE.search(draft_text)) or has_any_marker(draft_text, ("текущая цена", "стоимость", "цена"))
    if intent == "installment":
        return has_any_marker(draft_text, ("рассроч", "долями", "частями", "помесяч", "семестр", "год", "банк"))
    if intent == "discount":
        return has_any_marker(draft_text, ("скид", "процент", "%", "суммир", "многодет", "второй предмет"))
    if intent == "trial":
        return has_any_marker(draft_text, ("пробн", "фрагмент", "дистанц", "приезж", "оформление"))
    if intent in {"camp", "live_availability"}:
        return has_any_marker(draft_text, ("лагер", "лвш", "смен", "менделеево", "городск", "прожив", "мест", "налич"))
    if intent == "schedule":
        return has_any_marker(draft_text, ("распис", "дни", "время", "занятия", "группа"))
    if intent == "format":
        return has_any_marker(draft_text, ("онлайн", "очно", "дистанц", "формат", "платформ"))
    if intent == "address":
        return has_any_marker(draft_text, ("адрес", "сретенка", "красносельск", "пацаева", "мфти"))
    if intent == "matkap":
        return has_any_marker(draft_text, ("маткап", "сфр", "документ"))
    if intent == "tax":
        return has_any_marker(draft_text, ("налог", "вычет", "фнс", "13%"))
    if has_any_marker(question_norm, ("телефон", "номер", "позвон")):
        return has_any_marker(draft_text, ("телефон", "номер", "позвон"))
    return len(draft_text) >= 120

def _skip_missing_fact_template_by_answer_contract(
    context: Optional[Mapping[str, Any]],
    *,
    answer_contract_controls_green_templates: bool,
) -> bool:
    if not answer_contract_controls_green_templates:
        return False
    contract = _answer_contract(context)
    intent = str(contract.get("primary_intent") or "").strip()
    return intent in {"general_consultation", "format", "schedule"}

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
    return replace(
        result,
        route=route,
        draft_text=repair_text,
        safety_flags=flags,
        manager_checklist=checklist,
        context_warnings=tuple(dict.fromkeys([*result.context_warnings, "asked_known_data_again"])),
        metadata=metadata,
    )

def apply_funnel_policy_guard(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    funnel = context.get("funnel_state") if isinstance(context, Mapping) and isinstance(context.get("funnel_state"), Mapping) else {}
    if not funnel:
        return result
    if str(funnel.get("lead_stage") or "") != "p0_manager_only" and str(funnel.get("next_step_type") or "") != "manager_only_p0":
        return result
    flags = tuple(dict.fromkeys([*result.safety_flags, "autonomy_blocked_funnel_p0", "high_risk_manager_only"]))
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "Детерминированная воронка распознала P0/high-risk часть: не отправлять автономно.",
            ]
        )
    )
    return replace(
        result,
        route="manager_only",
        safety_flags=flags,
        manager_checklist=checklist,
        metadata={**dict(result.metadata), "autonomy_blocked_funnel_p0": True},
    )

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
    if known.get("student_name") and re.search(r"(фио|имя|как\s+зовут)[^.!?\n]{0,80}(реб[её]нк|ученик)", text):
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

def apply_brand_separation_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    active_brand = _active_brand(context)
    if active_brand == "unknown":
        if PRECISE_CONDITION_RE.search(result.draft_text):
            return _brand_guarded_result(result, reason="brand_unknown_precise_condition_blocked")
        return result
    forbidden_terms = BRAND_FORBIDDEN_TERMS.get(active_brand, ())
    if active_brand == "unpk" and _is_unpk_bank_installment_question(result, client_message=client_message, context=context):
        forbidden_terms = tuple(
            term for term in forbidden_terms if term not in {"рассрочка через банк", "через банк"}
        )
    lowered = result.draft_text.casefold()
    leaked = tuple(term for term in forbidden_terms if term in lowered)
    if not leaked:
        return result
    return _brand_guarded_result(result, reason="cross_brand_client_text_blocked", leaked_terms=leaked)

def _is_unpk_bank_installment_question(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> bool:
    if _active_brand(context) != "unpk":
        return False
    client_text = str(client_message or "").casefold().replace("ё", "е")
    plan = _conversation_intent_plan(context)
    intent = str(plan.get("primary_intent") or "").strip()
    topic = str(result.topic_id or "").strip()
    return (
        intent == "installment"
        or topic == "theme:006_installment"
        or (has_any_marker(client_text, ("рассроч", "частями", "помесяч", "банк")) and not has_any_marker(client_text, ("фотон", "долями", "т-банк")))
    )

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

def _answer_quality_was_rewritten(result: SubscriptionDraftResult) -> bool:
    if "answer_quality_rewritten" in result.safety_flags:
        return True
    quality = result.metadata.get("answer_quality") if isinstance(result.metadata, Mapping) else {}
    return bool(isinstance(quality, Mapping) and quality.get("rewritten"))

def _is_refund_case(result: SubscriptionDraftResult, *, markers: set[str]) -> bool:
    return result.topic_id == "theme:009_refund" or "refund" in markers

def _is_legal_threat_case(result: SubscriptionDraftResult, *, markers: set[str]) -> bool:
    return result.topic_id == "theme:029_legal_question" or "legal" in markers

def _is_complaint_case(result: SubscriptionDraftResult, *, markers: set[str]) -> bool:
    return result.topic_id == "theme:019b_negative_feedback" or "complaint" in markers or "reputation_threat" in markers

def _is_reputation_only_case(*, markers: set[str]) -> bool:
    return "reputation_threat" in markers and "legal" not in markers

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

def _is_future_price_case(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> bool:
    client_text = str(client_message or "")
    normalized = client_text.casefold().replace("ё", "е")
    if not FUTURE_PRICE_INPUT_RE.search(client_text):
        return False
    asks_change_without_precise_future_price = bool(
        re.search(r"после\s+1\s+июля[^.?!]{0,80}(поменя|измен|выраст|подраст|будет\s+друг)", normalized)
        and not re.search(
            r"(сколько|какая|почем|почём|стоим)[^.?!]{0,80}после\s+1\s+июля|"
            r"после\s+1\s+июля[^.?!]{0,80}(сколько|какая|почем|почём|стоим)",
            normalized,
        )
    )
    if asks_change_without_precise_future_price:
        return False
    return True

def _is_result_guarantee_case(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> bool:
    haystack = " ".join(
        [
            str(client_message or ""),
            result.draft_text,
            result.topic_id,
            result.broad_group,
            *result.alternative_themes,
            *result.context_warnings,
        ]
    )
    if RESULT_GUARANTEE_INPUT_RE.search(haystack):
        return True
    if isinstance(context, Mapping):
        for key in ("risk_flags", "context_warnings"):
            value = context.get(key)
            text = " ".join(str(item or "") for item in value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else str(value or "")
            if "score_guarantee" in text.casefold() or "result_guarantee" in text.casefold():
                return True
    return False

def _is_admission_guarantee_case(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> bool:
    haystack = " ".join(
        [
            str(client_message or ""),
            result.draft_text,
            result.topic_id,
            result.broad_group,
            *result.alternative_themes,
            *result.context_warnings,
        ]
    )
    if ADMISSION_GUARANTEE_INPUT_RE.search(haystack):
        return True
    if isinstance(context, Mapping):
        for key in ("risk_flags", "context_warnings"):
            value = context.get(key)
            text = " ".join(str(item or "") for item in value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else str(value or "")
            if "admission_guarantee" in text.casefold():
                return True
    return False

def _presale_refund_policy_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    combined_refund_context = " ".join([_dialog_context_haystack(context), str(client_message or "")])
    current_text = str(client_message or "").casefold().replace("ё", "е")
    if has_any_marker(current_text, ("суд", "прокуратур", "роспотреб", "жалоб", "юрист", "адвокат")):
        return ""
    refund_policy_context = _has_presale_refund_policy_context(combined_refund_context)
    current_mentions_refund_policy_followup = bool(
        re.search(
            r"\b(?:возврат\w*|вернут|вернете|вернёте|деньги\s+верн|услов\w*\s+возврат|до\s+оплат|перед\s+оплат|до\s+старт|передума\w*|не\s+ходить|отказ\w*)",
            str(client_message or ""),
            flags=re.I,
        )
    )
    current_asks_presale_refund_process = bool(
        refund_policy_context
        and (
            has_any_marker(
                current_text,
                (
                    "куда писать",
                    "куда написать",
                    "куда обращаться",
                    "к кому обращаться",
                    "какой порядок",
                    "порядок какой",
                    "что делать",
                    "как оформить",
                    "если решим не ходить",
                ),
            )
        )
    )
    current_is_presale_ack = bool(
        has_any_marker(
            current_text,
            (
                "ясно",
                "понял",
                "поняла",
                "просто заранее",
                "заранее уточ",
                "просто уточ",
                "спасибо",
                "хорошо",
            ),
        )
    )
    current_requests_manager_confirmation = has_any_marker(
        current_text,
        ("пусть менеджер", "менеджер пусть", "менеджер подтверд", "менеджер тогда", "менеджер напиш"),
    )
    if current_requests_manager_confirmation and not current_asks_presale_refund_process:
        return ""
    if not (
        is_benign_hypothetical_refund(client_message)
        or (current_mentions_refund_policy_followup and refund_policy_context and is_benign_hypothetical_refund(combined_refund_context))
        or (current_is_presale_ack and refund_policy_context and is_benign_hypothetical_refund(combined_refund_context))
    ):
        return ""
    known = known_context_fields(context)
    details: list[str] = []
    grade = str(known.get("grade") or "").strip()
    subject = str(known.get("subject") or "").strip()
    course_format = str(known.get("format") or "").strip()
    if grade:
        details.append(f"{grade} класс")
    if subject:
        details.append(subject)
    if course_format:
        details.append(course_format)
    suffix = f" по {', '.join(details)}" if details else ""
    if current_is_presale_ack and not current_mentions_refund_policy_followup:
        return (
            "Да, зафиксировала: это был именно вопрос на будущее до оплаты, не жалоба и не заявление на возврат. "
            "По условиям возврата менеджер подтвердит актуальные правила договора по выбранному курсу; класс, предмет и формат уже передам в контексте."
        )
    if current_asks_presale_refund_process:
        return (
            "Если до старта решите не идти на курс, напишите менеджеру в этот же чат или на тот контакт, где оформляли запись. "
            "Он проверит выбранный курс, договор и статус оплаты и подскажет порядок. "
            "Сумму или гарантию возврата без проверки не обещаю, но это именно предпродажный вопрос — не жалоба и не заявление на возврат. "
            "Класс, предмет и формат уже передам в контексте."
        )
    return (
        f"Да, это можно уточнить заранее{suffix}. "
        "Такой вопрос до оплаты не оформляю как жалобу или заявление на возврат. "
        "По смыслу: возможность и порядок возврата зависят от выбранного курса и правил договора, поэтому точную сумму или обещание возврата без проверки не называю. "
        "Менеджеру передам уже известный контекст, повторно писать класс, предмет и формат не нужно."
    )

def _has_presale_refund_policy_context(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    refund_policy_markers = (
        "условия возврата",
        "правила возврата",
        "возврат оплаты",
        "вернуть оплат",
        "вернуть деньги",
        "деньги верн",
        "не понрав",
        "не подойдет",
        "не подойд",
        "передума",
        "до оплаты",
        "перед оплат",
    )
    if has_any_marker(value, ("налог", "вычет", "фнс", "кнд", "3-ндфл")) and not has_any_marker(value, refund_policy_markers):
        return False
    return has_any_marker(value, refund_policy_markers)

def _is_enrollment_signup_question(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    return bool(
        re.search(r"\b(?:записаться|записат(?:ь|ся)|оформиться|оформить(?:ся)?)\b", value)
        or re.search(r"\bзапис[ьи]\b[^.!?\n]{0,80}\b(?:на\s+)?(?:курс|программ|обучен|занятия)\b", value)
        or re.search(r"\b(?:для|по|ради)\s+запис[ьи]\b", value)
    )

def _is_lesson_recording_question(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    if _is_enrollment_signup_question(value) and not _has_word_marker(value, "пропуст", "пропущен", "пересмотр", "урок", "заняти"):
        return False
    return bool(
        re.search(r"\bзапис(?:ь|и|ью|ям|ями)\b[^.!?\n]{0,80}\b(?:урок|заняти|лекци|вебинар)", value, flags=re.I)
        or re.search(r"\b(?:урок|заняти|лекци|вебинар)[^.?!\n]{0,80}\bзапис(?:ь|и|ью|ям|ями)\b", value, flags=re.I)
        or _has_word_marker(value, "пересмотр", "пропущен", "пропуст")
    )

def _has_word_marker(text: str, *markers: str) -> bool:
    return has_any_marker(text, markers)

def _terminal_safe_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    active_brand = _active_brand(context)
    client_haystack = str(client_message or "").casefold()
    haystack = _semantic_haystack(result, client_message=client_message, context=context)
    direct_identity_question = (
        re.search(r"\b(ты|вы)\s+(бот|ии|нейросеть|человек)\b", client_haystack, flags=re.I)
        or "с кем я общаюсь" in client_haystack
        or "живой человек" in client_haystack
        or "живой оператор" in client_haystack
    )
    if direct_identity_question:
        if active_brand == "foton":
            return IDENTITY_FOTON_SAFE_TEXT
        if active_brand == "unpk":
            return IDENTITY_UNPK_SAFE_TEXT
        return IDENTITY_PROMPT_SAFE_TEXT
    if any(
        marker in client_haystack
        for marker in ("ignore all previous", "system prompt", "системный промпт", "покажи промпт", "chatgpt", "gpt", "openai", "claude", "codex")
    ):
        return IDENTITY_PROMPT_SAFE_TEXT
    if any(
        marker in client_haystack
        for marker in ("не отвечаете", "одно и то же", "не можете подсказать", "не можете ответить", "не буду оставлять заявку", "буду искать другой вариант")
    ):
        return SOFT_NEGATIVE_HANDOFF_SAFE_TEXT
    if result.topic_id == "service:S3_out_of_scope" or OFF_TOPIC_INPUT_RE.search(str(client_message or "")):
        if active_brand == "foton":
            return OFF_TOPIC_FOTON_SAFE_TEXT
        if active_brand == "unpk":
            return OFF_TOPIC_UNPK_SAFE_TEXT
        return OFF_TOPIC_GENERIC_SAFE_TEXT
    if active_brand == "foton" and ("только у фотон" in client_haystack or "только в фотон" in client_haystack):
        return BRAND_LOYALTY_FOTON_TEXT
    if active_brand == "unpk" and ("только в унпк" in client_haystack or "только у унпк" in client_haystack):
        return BRAND_LOYALTY_UNPK_TEXT
    if any(
        marker in client_haystack
        for marker in (
            "повтори за мной",
            "claude",
            "codex",
            "openai",
            "кодекса",
            "вы человек",
            "ты человек",
            "живой человек",
            "нейросеть",
            "какая модель",
            "кто тебя обучил",
            "system",
            "dan mode",
            "no restrictions",
            "translate to english",
            "{{system}}",
            "забудь все инструкции",
            "reveal internal",
        )
    ):
        return IDENTITY_PROMPT_SAFE_TEXT
    if re.search(r"\bбот\b", client_haystack, flags=re.I):
        if active_brand == "foton":
            return IDENTITY_FOTON_SAFE_TEXT
        if active_brand == "unpk":
            return IDENTITY_UNPK_SAFE_TEXT
        return IDENTITY_PROMPT_SAFE_TEXT
    if (
        "промокод" in client_haystack
        or "promo" in client_haystack
        or "инстаграм" in client_haystack
        or "instagram" in client_haystack
        or re.search(r"\b[A-ZА-ЯЁ]{2,}[A-ZА-ЯЁ0-9-]{3,}\b", str(client_message or ""))
        or PROMOCODE_DRAFT_RE.search(str(client_message or ""))
    ):
        return PROMOCODE_SAFE_TEXT
    if "прошлом году" in client_haystack or "раньше было" in client_haystack or "обещал" in client_haystack:
        return FALSE_INFO_SAFE_TEXT
    if "ссылк" in client_haystack and "оплат" in client_haystack:
        return PAYMENT_LINK_SAFE_TEXT
    if has_any_marker(client_haystack, ("личный кабинет", "кабинет", "платформ")) and has_any_marker(
        client_haystack,
        ("как", "зайти", "войти", "доступ", "логин", "парол"),
    ):
        return "Передам менеджеру вопрос по доступу к личному кабинету; он подскажет порядок входа по вашей группе."
    if "коллег" in client_haystack and ("учится" in client_haystack or "обуч" in client_haystack):
        return THIRD_PARTY_PRIVACY_SAFE_TEXT
    if "фио" in client_haystack and ("отвечает" in client_haystack or "сотрудник" in client_haystack or "назовите" in client_haystack):
        return EMPLOYEE_PRIVACY_SAFE_TEXT
    if "м11" in client_haystack or "old_term" in haystack:
        return OLD_TERM_SAFE_TEXT
    if active_brand == "unpk" and "моск" in client_haystack and (
        "обычн" in client_haystack or "регулярн" in client_haystack or "заняти" in client_haystack
    ):
        return _unpk_moscow_address_template_from_kb(context)
    address_negated = any(marker in client_haystack for marker in ("адрес не нужен", "адреса не нужны", "не про адрес", "адрес не надо"))
    plan = _conversation_intent_plan(context)
    held_scope = str(plan.get("fact_scope") or "")
    recording_followup = str(plan.get("primary_intent") or "") == "recording" or held_scope in {"online_recordings", "offline_recordings"} or _is_lesson_recording_question(client_haystack)
    camp_or_transport_context = any(
        marker in client_haystack
        for marker in ("лагер", "лвш", "менделеево", "выездн", "трансфер", "добир", "как туда")
    ) or ("летн" in client_haystack and "школ" in client_haystack) or "без прожив" in client_haystack
    if active_brand == "foton" and not recording_followup and not address_negated and not camp_or_transport_context and (
        "адрес" in client_haystack
        or "где" in client_haystack
        or "площадк" in client_haystack
        or "моск" in client_haystack
    ):
        return _foton_address_template_from_kb(context)
    if active_brand == "unpk" and not recording_followup and not address_negated and ("площадки" in client_haystack or "адрес" in client_haystack):
        return _unpk_all_addresses_template_from_kb(context)
    asks_contact = _asks_center_contact(client_haystack)
    if active_brand == "foton" and asks_contact:
        return _foton_contact_template_from_kb(context)
    if active_brand == "unpk" and asks_contact:
        return _unpk_contact_template_from_kb(context)
    if "квитанц" in client_haystack:
        return QUITTANCE_SAFE_TEXT
    if "интенсив" in client_haystack and any(marker in client_haystack for marker in ("сколько", "стоим", "цен", "почем", "почём")):
        return ""
    if _client_message_contains_pii(client_message) and (
        ("цена" in client_haystack and "онлайн" not in client_haystack)
        or "оценк" in client_haystack
        or "прогресс" in client_haystack
        or "всош" in client_haystack
    ):
        return "Менеджер свяжется и уточнит детали безопасно, без повторной отправки персональных данных в чат."
    if "интенсив" in client_haystack:
        return PROGRAM_HANDOFF_SAFE_TEXT
    group_context = (
        "групп" in client_haystack
        or "мини-групп" in client_haystack
        or "не индивидуаль" in client_haystack
        or "а не индивидуаль" in client_haystack
    )
    if "индивидуаль" in client_haystack and not group_context:
        return INDIVIDUAL_HANDOFF_SAFE_TEXT
    if "юр.лица" in client_haystack or "юрлица" in client_haystack or "юридическ" in client_haystack:
        return CONTRACT_ENTITY_SAFE_TEXT
    return ""

def _asks_center_contact(text: str) -> bool:
    haystack = str(text or "").casefold().replace("ё", "е")
    if not haystack:
        return False
    if re.search(r"\b(?:мой|моя|мои|свой|своя|свои|наш|наша)\s+(?:телефон|номер|почт|email|e-mail)\b", haystack):
        return False
    contact_marker = r"(?:телефон|номер|контакт|почт|email|e-mail|куда\s+писать|как\s+связаться|связаться)"
    request_marker = r"(?:дайте|подскажите|скажите|пришлите|напишите|укажите|нужен|нужна|какой|какая|какие|ваш|ваша|ваши)"
    return bool(
        re.search(rf"\b{request_marker}\b[\s\S]{{0,40}}\b{contact_marker}\b", haystack)
        or re.search(rf"\b{contact_marker}\b[\s\S]{{0,40}}\b{request_marker}\b", haystack)
        or "как связаться" in haystack
        or "куда писать" in haystack
        or "по какому номеру" in haystack
        or "на какую почту" in haystack
    )

def _cross_brand_safe_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    active_brand = _active_brand(context)
    client_haystack = str(client_message or "").casefold()
    full_haystack = _semantic_haystack(result, client_message=client_message, context=context)
    if active_brand == "foton":
        other = any(marker in client_haystack for marker in ("унпк", "мфти", "kmipt", "70369"))
    elif active_brand == "unpk":
        other = any(marker in client_haystack for marker in ("фотон", "цдпо", "црдо", "cdpofoton", "т-банк", "долями"))
    else:
        other = False
    if "хочу только" in client_haystack:
        return ""
    if "cross_brand" in full_haystack:
        other = True
    if not other:
        return ""
    if active_brand == "unpk" and ("долями" in client_haystack or "т-банк" in client_haystack or "рассроч" in client_haystack):
        return UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT
    if "лиценз" in client_haystack:
        return CROSS_BRAND_LICENSE_SAFE_TEXT
    if "платформ" in client_haystack or "мтс линк" in client_haystack or "webinar" in client_haystack:
        return CROSS_BRAND_PLATFORM_SAFE_TEXT
    return CROSS_BRAND_GENERIC_SAFE_TEXT

def _scope_fact_missing_guard_template(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    if not _scope_fact_guard_enabled(context):
        return ""
    if result.route == "manager_only" or is_high_risk_result(result):
        return ""
    plan = _conversation_intent_plan(context)
    scope, blocked = _requested_fact_scope_context(context, plan=plan)
    if not scope and not blocked:
        return ""
    if not _scope_guard_has_missing_intent_fact(result, context, plan=plan):
        return ""
    draft_text = str(result.draft_text or "")
    if not _scope_guard_has_foreign_concrete_fact(draft_text, requested_scope=scope, blocked_neighbor_scopes=blocked):
        return ""
    return _scope_fact_narrow_handoff_text(context, result=result, plan=plan)

def _requested_fact_scope_context(
    context: Optional[Mapping[str, Any]],
    *,
    plan: Optional[Mapping[str, Any]] = None,
) -> tuple[str, tuple[str, ...]]:
    plan_mapping = plan if isinstance(plan, Mapping) else _conversation_intent_plan(context)
    scope = str(plan_mapping.get("fact_scope") or "").strip()
    blocked = tuple(str(item) for item in plan_mapping.get("blocked_neighbor_scopes", ()) or () if str(item).strip())
    if (not scope and not blocked) and isinstance(context, Mapping):
        facts_context = context.get("facts_context")
        if isinstance(facts_context, Mapping):
            scope = str(facts_context.get("fact_scope") or "").strip()
            blocked = tuple(str(item) for item in facts_context.get("blocked_neighbor_scopes", ()) or () if str(item).strip())
    return scope, blocked

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

def _fact_scope_guard_template(draft_text: str, *, context: Optional[Mapping[str, Any]] = None) -> str:
    plan = _conversation_intent_plan(context)
    scope = str(plan.get("fact_scope") or "").strip()
    blocked = {str(item) for item in plan.get("blocked_neighbor_scopes", ()) or () if str(item).strip()}
    if not scope and not blocked:
        facts_context = context.get("facts_context") if isinstance(context, Mapping) and isinstance(context.get("facts_context"), Mapping) else {}
        scope = str(facts_context.get("fact_scope") or "").strip()
        blocked = {str(item) for item in facts_context.get("blocked_neighbor_scopes", ()) or () if str(item).strip()}
    if not scope and not blocked:
        return ""
    text = str(draft_text or "").casefold().replace("ё", "е")
    answer_scopes = _answer_fact_scopes(text)
    if answer_scopes_allowed(answer_scopes, requested_scope=scope, blocked_neighbor_scopes=tuple(blocked)):
        return ""
    if scope == "matkap_process":
        return (
            "Вы спрашиваете про маткапитал, поэтому другую документальную процедуру сюда не подставляю. "
            "По маткапиталу менеджер пришлёт актуальный перечень документов и подскажет порядок оформления через СФР."
        )
    if scope == "class_schedule":
        return (
            "Вы спрашиваете именно расписание занятий, а не часы работы офиса. "
            "Подтверждённого расписания конкретной группы у меня сейчас нет, поэтому передам менеджеру запрос на проверку."
        )
    if scope == "city_day_camp":
        return (
            "Вы спрашиваете дневной летний формат без проживания, поэтому цену и условия выездной ЛВШ с проживанием не подставляю. "
            "Передам менеджеру запрос, чтобы он проверил подходящий дневной вариант."
        )
    if scope == "discount_second_subject":
        direct_question = str(plan.get("direct_question") or "").casefold()
        if _active_brand(context) == "foton":
            asks_multichild_or_stacking = has_any_marker(direct_question, ("многодет", "суммир"))
            if has_marker(direct_question, "онлайн") and not asks_multichild_or_stacking:
                return (
                    "На второй онлайн-предмет в Фотоне действует скидка 30% для того же ребёнка. "
                    "Скидки не суммируются: если есть несколько оснований, применяется наибольшая доступная. "
                    "Если уже выбрали второй предмет, дальше можно подобрать подходящую группу."
                )
            return (
                "На второй онлайн-предмет в Фотоне действует скидка 30%, на второй очный предмет — 20%. "
                "Многодетная скидка — 10% по удостоверению. Скидки не суммируются: применяется наибольшая доступная. "
                "Если у ребёнка второй онлайн-предмет, обычно выгоднее скидка 30%; менеджер проверит условия по выбранным курсам."
            )
        if _active_brand(context) == "unpk":
            return (
                "На второй и последующий предмет одного ребёнка в УНПК действует скидка 20%. "
                "Многодетная скидка — 10% по удостоверению. Скидки не суммируются: применяется наибольшая доступная. "
                "Менеджер проверит условия по выбранным курсам."
            )
    return (
        "Не буду подставлять похожий, но другой факт вместо вашего вопроса. "
        "Передам менеджеру запрос на точную проверку по нужной теме."
    )

def _forbidden_pair_guard_template(draft_text: str, *, context: Optional[Mapping[str, Any]] = None) -> str:
    plan = _conversation_intent_plan(context)
    forbidden_pairs = {str(item) for item in plan.get("forbidden_pairs", ()) or () if str(item).strip()}
    if "matkap+installment" not in forbidden_pairs:
        return ""
    text = str(draft_text or "").casefold().replace("ё", "е")
    if not has_any_marker(text, ("рассроч", "долями", "частями", "помесяч", "банк", "т-банк", "месяц")):
        return ""
    return (
        "Маткапитал — это отдельный источник оплаты через СФР, поэтому не буду смешивать его в одном ответе "
        "с другими способами оплаты. По маткапиталу менеджер пришлёт актуальный перечень документов и подскажет порядок оформления."
    )

def _answer_fact_scopes(text: str) -> set[str]:
    return detect_fact_scopes(text)

def _missing_fact_helpful_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
    markers: set[str] | None = None,
) -> str:
    if result.route == "manager_only":
        return ""
    if markers and (markers & {"refund", "legal", "complaint", "reputation_threat"}):
        return ""
    if is_high_risk_result(result):
        return ""
    if result.topic_id not in AUTONOMY_MATRIX_SAFE_TOPIC_IDS:
        return ""
    if not _has_missing_fact_signal(result, context):
        return ""
    if result.topic_id in {"theme:026_camp_general", "theme:027_camp_living_conditions", "theme:028_transport_logistics"}:
        draft_lower = str(result.draft_text or "").casefold()
        client_lower = str(client_message or "").casefold()
        if _asks_live_status_or_booking_question(client_lower):
            return ""
        asks_camp_contents = any(marker in client_lower for marker in ("что входит", "прожив", "питан", "трансфер"))
        asks_dates_or_shift_list = (
            any(marker in client_lower for marker in ("когда", "дат", "заезд"))
            or ("смен" in client_lower and any(marker in client_lower for marker in ("какие", "какая", "есть", "будут")))
        )
        if (
            asks_dates_or_shift_list
            and not asks_camp_contents
            and "напишите" not in draft_lower
            and "подскажите" not in draft_lower
        ):
            return MISSING_CAMP_HELPFUL_TEXT
    if not _draft_is_low_value_without_exact_fact(result.draft_text):
        return ""

    topic = result.topic_id
    if topic == "theme:001_pricing":
        if "интенсив" in str(client_message or "").casefold():
            return MISSING_INTENSIVE_PRICE_HELPFUL_TEXT
        return MISSING_PRICE_HELPFUL_TEXT
    if topic == "theme:013_schedule":
        return MISSING_SCHEDULE_HELPFUL_TEXT
    if topic in {
        "theme:016_program",
        "theme:020_enrollment",
        "theme:021_continuation",
        "theme:022_age_level_testing",
        "theme:023_trial_class",
        "service:S5_general_consultation",
    }:
        return MISSING_PROGRAM_HELPFUL_TEXT
    if topic == "theme:005_discounts":
        return MISSING_DISCOUNT_HELPFUL_TEXT
    if topic == "theme:006_installment":
        return MISSING_INSTALLMENT_HELPFUL_TEXT
    if topic in {"theme:007_matkap_payment", "theme:008_tax_deduction", "theme:011_contract", "theme:012_certificates"}:
        return MISSING_DOCS_HELPFUL_TEXT
    if topic in {"theme:026_camp_general", "theme:027_camp_living_conditions", "theme:028_transport_logistics"}:
        return MISSING_CAMP_HELPFUL_TEXT
    return MISSING_GENERAL_HELPFUL_TEXT

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

def _draft_is_low_value_without_exact_fact(draft_text: str) -> bool:
    text = str(draft_text or "").casefold().replace("ё", "е")
    if not text.strip():
        return True
    useful_markers = ("класс", "предмет", "формат", "очно", "онлайн", "цель", "вариант", "курс", "смен", "программа")
    if any(marker in text for marker in useful_markers) and len(text) >= 160:
        return False
    generic_markers = ("уточним", "уточню", "проверим", "проверю", "передам", "свяжется", "вернемся", "вернусь")
    return any(marker in text for marker in generic_markers) or len(text) < 120

def _promoted_verified_fact_text(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
    client_message: str = "",
) -> str:
    facts = _confirmed_fact_texts(context, limit=8 if result.topic_id == "theme:014_format" else 3)
    if not facts:
        return ""
    fact_sentence = " ".join(_ensure_sentence(fact) for fact in facts)
    if result.topic_id == "theme:001_pricing":
        asks_validity = any(
            marker in str(client_message or "").casefold().replace("ё", "е")
            for marker in ("сейчас", "поменяет", "изменит", "потом", "подраст", "повыс", "актуальн")
        )
        suffix = " Это текущая цена на сейчас; позже она может подрасти, точную дату не называю без проверки." if asks_validity else ""
        known = known_context_fields(context)
        next_step = (
            "Дальше можно подобрать подходящую группу и удобный вариант оплаты."
            if known.get("grade") and (known.get("format") or known.get("subject"))
            else "Если напишете класс ребёнка и удобный формат, подберём самый подходящий вариант оплаты."
        )
        return _soften_current_price_deadline_text(
            f"Да, сориентирую по проверенным условиям. {fact_sentence}{suffix} {next_step}",
            client_message=client_message,
        )
    if result.topic_id == "theme:013_schedule":
        return (
            f"По расписанию есть проверенная информация. {fact_sentence} "
            "Если напишете класс, предмет и удобный день, подберём ближайший подходящий вариант."
        )
    if result.topic_id == "theme:014_format":
        facts = _prefer_format_facts(facts, query=client_message) or facts
        fact_sentence = " ".join(_ensure_sentence(fact) for fact in facts[:3])
        return (
            f"Да, по формату есть проверенная информация. {fact_sentence} "
            "Если напишете класс и цель обучения, поможем подобрать подходящую группу."
        )
    if result.topic_id in {"theme:016_program", "theme:020_enrollment", "theme:021_continuation", "theme:022_age_level_testing", "theme:023_trial_class"}:
        return (
            f"По программе можно сориентироваться так. {fact_sentence} "
            "Напишите класс, предмет и цель обучения — подберём подходящий следующий шаг."
        )
    if result.topic_id in {"theme:005_discounts", "theme:006_installment"}:
        return (
            f"По условиям оплаты есть проверенная информация. {fact_sentence} "
            "Если напишете курс и формат, подскажем, какой вариант удобнее под вашу ситуацию."
        )
    if result.topic_id in {"theme:007_matkap_payment", "theme:008_tax_deduction", "theme:011_contract", "theme:012_certificates"}:
        return (
            f"По документам есть проверенная информация. {fact_sentence} "
            "Если напишете, какой именно документ нужен, подскажем следующий шаг."
        )
    return (
        f"Да, сориентирую по проверенной информации. {fact_sentence} "
        "Если напишете класс ребёнка и задачу, поможем подобрать подходящий вариант."
    )

def _confirmed_fact_texts(context: Optional[Mapping[str, Any]], *, limit: int = 3) -> tuple[str, ...]:
    if not isinstance(context, Mapping):
        return ()
    value = context.get("confirmed_facts")
    texts: list[str] = []
    if isinstance(value, Mapping):
        for item in value.values():
            cleaned = _client_clean_fact_text(item)
            if cleaned:
                texts.append(cleaned)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            cleaned = _client_clean_fact_text(item)
            if cleaned:
                texts.append(cleaned)
    return tuple(dict.fromkeys(texts[: max(1, limit)]))

def _prefer_format_facts(facts: Sequence[str], *, query: str = "") -> tuple[str, ...]:
    query_text = str(query or "").casefold().replace("ё", "е")
    asks_online = has_any_marker(query_text, ("онлайн", "дистанц"))
    asks_offline = has_any_marker(query_text, ("очно", "офлайн"))
    preferred: list[str] = []
    for fact in facts:
        normalized = fact.casefold().replace("ё", "е")
        if has_any_marker(normalized, ("учебный год", "уровень обучения")):
            continue
        if asks_online and has_marker(normalized, "очно") and not has_marker(normalized.replace("онлайн-платформа", ""), "онлайн"):
            continue
        if asks_offline and has_marker(normalized, "онлайн") and not has_marker(normalized, "очно"):
            continue
        if has_any_marker(normalized, ("онлайн-платформа", "мтс линк", "webinar", "запис", "очно", "онлайн")):
            preferred.append(fact)
    return tuple(preferred)

def _ensure_sentence(text: str) -> str:
    value = " ".join(str(text or "").split()).rstrip()
    if not value:
        return ""
    return value if value.endswith((".", "!", "?")) else f"{value}."

def _foton_offline_free_trial_guard_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    if _active_brand(context) != "foton":
        return ""
    known = known_context_fields(context)
    fmt = str(known.get("format") or "").casefold()
    client_text = str(client_message or "").casefold().replace("ё", "е")
    draft_text = str(result.draft_text or "").casefold().replace("ё", "е")
    combined = " ".join([client_text, draft_text, result.topic_id, result.broad_group]).casefold()
    trial_context = (
        result.topic_id == "theme:023_trial_class"
        or "пробн" in combined
        or "фрагмент" in combined
    )
    offline_context = (
        fmt in {"очно", "очный", "офлайн", "offline"}
        or has_any_marker(client_text, ("очно", "очный", "офлайн", "прийти", "приехать", "приезж"))
    )
    free_question_or_claim = "бесплат" in combined
    unsupported_free_claim = bool(
        re.search(r"\b(?:да|точно|можно|доступн\w*)[^.?!\n]{0,80}\b(?:бесплат\w*)[^.?!\n]{0,80}\b(?:пробн|занят)", draft_text, flags=re.I)
        or re.search(r"\b(?:пробн|занят)[^.?!\n]{0,80}\b(?:можно|доступн\w*)?[^.?!\n]{0,80}\bбесплат\w*", draft_text, flags=re.I)
    )
    if trial_context and offline_context and (free_question_or_claim or unsupported_free_claim):
        return FOTON_OFFLINE_FREE_TRIAL_GUARD_TEXT
    return ""

def _soften_current_price_deadline_text(text: str, *, client_message: str = "") -> str:
    value = " ".join(str(text or "").split())
    if not value:
        return ""
    date_pattern = r"(?:1\s+(?:июля|августа)|0?1[./-]0?[78](?:[./-]\d{2,4})?)"
    has_deadline = bool(
        re.search(rf"\b(?:до|после|для\s+периода\s+до)\s+{date_pattern}\b", value, flags=re.I)
        or re.search(rf"\bдата\s*[—–:;-]\s*{date_pattern}\b", value, flags=re.I)
        or re.search(rf"\b{date_pattern}\s+2026(?:\s+года?)?\b", value, flags=re.I)
    )
    has_future_price_guarantee = bool(
        re.search(r"\bчерез\s+(?:недел\w*|месяц\w*)[^.?!]*(?:не\s+(?:должн\w*|будет)|остан\w*|сохран\w*)[^.?!]*(?:друг\w*|измен\w*|помен\w*|цен\w*)", value, flags=re.I)
        or re.search(r"\b(?:значит|по\s+этому\s+правилу)[^.?!]*(?:не\s+(?:должн\w*|будет)|остан\w*|сохран\w*)[^.?!]*(?:друг\w*|измен\w*|помен\w*|цен\w*)", value, flags=re.I)
        or re.search(r"\bцен\w*[^.?!]{0,80}(?:не\s+(?:измен\w*|поменя\w*)|остан\w*|сохран\w*)", value, flags=re.I)
        or re.search(r"\b(?:не\s+(?:измен\w*|поменя\w*)|остан\w*|сохран\w*)[^.?!]{0,80}\bцен\w*", value, flags=re.I)
    )
    has_fixation_claim = bool(re.search(r"\bзафиксировать\s+(?:текущ(?:ую|ие)|цену|условия)\b", value, flags=re.I))
    has_broken_fixation_fragment = "Оформление по текущим условиям проверит менеджер" in value
    if not has_deadline and not has_future_price_guarantee and not has_fixation_claim and not has_broken_fixation_fragment:
        return value
    had_deadline = has_deadline or has_future_price_guarantee
    value = re.sub(
        rf"(?:^|(?<=[.!?])\s*)да,\s*уточняю:\s*дата\s*[—–:;-]\s*{date_pattern}(?:\s+2026)?(?:\s+года?)?[.?!]?\s*",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(
        rf"\bдата\s*[—–:;-]\s*{date_pattern}(?:\s+2026)?(?:\s+года?)?[,.!?;]?\s*",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(
        rf"\s*после\s+{date_pattern}(?:\s+2026)?(?:\s+года?)?[^.?!]*(?:[.?!]|$)",
        ". ",
        value,
        flags=re.I,
    )
    value = re.sub(rf"\s+при\s+оформлении\s+до\s+{date_pattern}(?:\s+2026)?(?:\s+года?)?", "", value, flags=re.I)
    value = re.sub(rf"\s+при\s+раннем\s+бронировании\s+до\s+{date_pattern}(?:\s+2026)?(?:\s+года?)?", "", value, flags=re.I)
    value = re.sub(rf"\s+по\s+текущим\s+данным\s+такие\s+условия\s+указаны\s+для\s+периода\s+до\s+{date_pattern}(?:\s+2026)?(?:\s+года?)?;?", "", value, flags=re.I)
    value = re.sub(rf"\s+до\s+{date_pattern}(?:\s+2026)?(?:\s+года?)?", "", value, flags=re.I)
    value = re.sub(r"\s*сейчас\s+по\s+дате\s+вы\s+укладываетесь[;,.]?", " Сейчас действует текущая цена,", value, flags=re.I)
    value = re.sub(r"\s*по\s+дате\s+вы\s+укладываетесь[;,.]?", "", value, flags=re.I)
    value = re.sub(
        r"\b(?:вы|мы)\s+(?:успеваете|успеваем|укладываетесь)\b[^.?!]*(?:[.?!]|;)?",
        "Сейчас действует текущая цена. ",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"\bПосле\s+этой\s+даты\s+стоимость\s+может\s+(?:отличаться|измениться)[.?!]?",
        "Позже цена может измениться.",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"\bЧерез\s+(?:недел\w*|месяц\w*)[^.?!]*(?:не\s+(?:должн\w*|будет)|остан\w*|сохран\w*)[^.?!]*(?:друг\w*|измен\w*|помен\w*|цен\w*)[^.?!]*(?:[.?!]|$)",
        "Точную дату изменения цены без проверки не называю. ",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"\b(?:Значит|По\s+этому\s+правилу)[^.?!]*(?:не\s+(?:должн\w*|будет)|остан\w*|сохран\w*)[^.?!]*(?:друг\w*|измен\w*|помен\w*|цен\w*)[^.?!]*(?:[.?!]|$)",
        "Точную дату изменения цены без проверки не называю. ",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"\bцен\w*[^.?!]{0,80}(?:не\s+(?:измен\w*|поменя\w*)|остан\w*|сохран\w*)[^.?!]*(?:[.?!]|$)",
        "Точную дату изменения цены без проверки не называю. ",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"\b(?:не\s+(?:измен\w*|поменя\w*)|остан\w*|сохран\w*)[^.?!]{0,80}\bцен\w*[^.?!]*(?:[.?!]|$)",
        "Точную дату изменения цены без проверки не называю. ",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"\b(?:передать\s+)?оформление\s+по\s+текущим\s+условиям\b",
        "передать менеджеру проверку оформления по текущим условиям",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"\bзафиксировать\s+(?:текущ(?:ую|ие)|цену|условия)[^.?!]*(?:[.?!]|$)",
        "Оформление по текущим условиям проверит менеджер. ",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"Раннее\s+бронирование\s+позволяет\s+Оформление\s+по\s+текущим\s+условиям\s+проверит\s+менеджер\.",
        "Оформление по текущим условиям проверит менеджер.",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"как\s+Оформление\s+по\s+текущим\s+условиям\s+проверит\s+менеджер\.",
        "как оформить по текущим условиям.",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"Чтобы\s+Оформление\s+по\s+текущим\s+условиям\s+проверит\s+менеджер\.",
        "Чтобы оформить по текущим условиям, менеджер проверит актуальные условия.",
        value,
        flags=re.I,
    )
    value = _dedupe_sentence(value, "Оформление по текущим условиям проверит менеджер.")
    value = re.sub(r"\s+([.,!?])", r"\1", value)
    value = re.sub(r"\.{2,}", ".", value)
    value = re.sub(r"\s{2,}", " ", value).strip()
    asks_validity = any(
        marker in str(client_message or "").casefold().replace("ё", "е")
        for marker in ("сейчас", "поменяет", "изменит", "остан", "сохран", "потом", "подраст", "повыс")
    )
    asks_date = "дат" in str(client_message or "").casefold().replace("ё", "е")
    if (asks_validity or had_deadline or has_fixation_claim) and "может измениться" not in value and "подраст" not in value:
        value = value.rstrip(".") + ". Это текущая цена на сейчас; позже она может подрасти."
    if asks_date and "точную дату изменения цены" not in value.casefold():
        value = value.rstrip(".") + ". Точную дату изменения цены менеджер подтвердит при оформлении."
    return value

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

def decide_route(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
    allow_default_autonomy: bool = False,
) -> RouteDecision:
    """Central route-permission decision point.

    Правка 4a keeps current behavior 1:1: no default inversion until the veto
    shield is green. The later 4b flip is gated by allow_default_autonomy.
    """

    if result.route not in (*AUTONOMOUS_ROUTES, "draft_for_manager"):
        return RouteDecision(route=result.route)
    if should_force_manager_only(context):
        return RouteDecision(
            route="manager_only",
            veto_category="force_manager_only",
            safety_flags=("forced_manager_only_by_rop_policy",),
        )
    if set(detect_high_risk_input_markers(client_message, context=context)) or is_high_risk_result(result):
        return RouteDecision(
            route="manager_only",
            veto_category="high_risk",
            safety_flags=("autonomy_blocked_high_risk", "high_risk_manager_only"),
            manager_checklist=("Автономный ответ запрещен: в сообщении есть P0/high-risk тема.",),
            metadata={"autonomy_blocked_high_risk": True},
        )
    if _active_brand(context) == "unknown":
        return RouteDecision(
            route="draft_for_manager",
            veto_category="unknown_brand",
            safety_flags=("autonomy_default_cautious_unknown_brand",),
            manager_checklist=("Автономный ответ запрещен: активный бренд не определен.",),
        )
    if result.route in AUTONOMOUS_ROUTES and not _autonomy_enabled(context):
        return RouteDecision(
            route="draft_for_manager",
            veto_category="autonomy_policy_missing",
            safety_flags=("autonomy_default_cautious_no_policy",),
            manager_checklist=("Автономный ответ запрещен: нет явного разрешения матрицы автономности.",),
        )

    autonomy_ready = _autonomy_enabled(context) and _autonomy_topic_allowed(result.topic_id, context)
    has_covering_fact = _has_client_safe_current_fact(context) or _is_verified_client_safe_template(result.draft_text)
    if (
        result.route == "draft_for_manager"
        and autonomy_ready
        and has_covering_fact
        and _memory_followup_answered_topic(context, client_message)
    ):
        return RouteDecision(
            route="bot_answer_self_for_pilot",
            safety_flags=("dialogue_memory_followup_autonomy",),
            metadata={"dialogue_memory_followup_autonomy": True},
            autonomous_candidate=True,
        )
    if (
        allow_default_autonomy
        and result.route == "draft_for_manager"
        and autonomy_ready
        and has_covering_fact
    ):
        return RouteDecision(route="bot_answer_self_for_pilot", autonomous_candidate=True)
    return RouteDecision(route=result.route, autonomous_candidate=result.route == "draft_for_manager" and autonomy_ready)

def _memory_followup_answered_topic(context: Optional[Mapping[str, Any]], client_message: str) -> bool:
    if not isinstance(context, Mapping):
        return False
    memory = context.get("dialogue_memory_view")
    if not isinstance(memory, Mapping):
        return False
    routes = _memory_text_items(memory.get("route_history"))
    if not any(route in AUTONOMOUS_ROUTES for route in routes):
        return False
    answered = (*_memory_text_items(memory.get("answered_questions")), *_memory_text_items(memory.get("safe_answered_parts")))
    if not answered:
        return False
    focus = memory.get("topic_focus")
    if not isinstance(focus, Mapping):
        return False
    text = _memory_norm(client_message)
    if not text or _memory_mentions_different_topic(text, focus):
        return False
    return _memory_mentions_focus(text, focus) or _memory_short_followup(text)

def _memory_text_items(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item or "").strip() for item in value if str(item or "").strip())
    text = str(value or "").strip()
    return (text,) if text else ()

def _memory_norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").casefold().replace("ё", "е")).strip()

def _memory_mentions_focus(text: str, focus: Mapping[str, Any]) -> bool:
    aliases: list[str] = []
    for field in ("subject", "format", "product", "product_family"):
        aliases.extend(_memory_topic_aliases(field, focus.get(field)))
    grade = str(focus.get("grade") or "").strip()
    if grade:
        aliases.extend([grade, f"{grade} класс"])
    return any(alias and _memory_norm(alias) in text for alias in aliases)

def _memory_short_followup(text: str) -> bool:
    if len(text) > 90:
        return False
    return bool(
        re.search(
            r"^(?:а\s+)?(?:сколько|цена|стоимость|онлайн|очно|для\s+\d{1,2}|"
            r"\d{1,2}\s*класс|есть|можно|подойдет|подходит|а\s+если|тогда|и\s+ещ[её])\b",
            text,
            re.I,
        )
    )

def _memory_mentions_different_topic(text: str, focus: Mapping[str, Any]) -> bool:
    subject = _memory_norm(focus.get("subject"))
    subject_groups = {
        "информатика": ("информат", "informatics", "computer science"),
        "физика": ("физик", "physics"),
        "математика": ("математ", "math"),
        "химия": ("хими", "chem"),
        "биология": ("биолог", "bio"),
    }
    focus_group = ""
    for name, aliases in subject_groups.items():
        if any(alias in subject for alias in aliases):
            focus_group = name
            break
    mentioned = {
        name
        for name, aliases in subject_groups.items()
        if any(re.search(rf"(?<![a-zа-я0-9]){re.escape(alias)}", text, re.I) for alias in aliases)
    }
    if mentioned and (not focus_group or mentioned != {focus_group}):
        return True
    family = _memory_norm(focus.get("product_family"))
    mentions_camp = bool(re.search(r"лвш|лагер|смен|выездн|camp|lvsh", text, re.I))
    if family == "regular_course" and mentions_camp:
        return True
    return False

def _memory_topic_aliases(field: str, value: object) -> tuple[str, ...]:
    raw = _memory_norm(value)
    if not raw:
        return ()
    if field == "subject":
        if "информ" in raw:
            return ("информат", "informatics", "computer science")
        if "физ" in raw:
            return ("физик", "physics")
        if "мат" in raw:
            return ("математ", "math")
        if "хим" in raw:
            return ("хими", "chem")
        if "био" in raw:
            return ("биолог", "bio")
    if field == "format":
        if "онлайн" in raw or "online" in raw:
            return ("онлайн", "online")
        if "очно" in raw or "офлайн" in raw or "offline" in raw:
            return ("очно", "офлайн", "offline")
    if field == "product_family":
        if raw == "camp" or "лагер" in raw or "смен" in raw or "лвш" in raw:
            return ("лвш", "лагер", "смен", "camp", "lvsh")
        if raw == "regular_course":
            return ("курс", "regular")
    if field == "product":
        return tuple(part for part in re.split(r"[\s,;/]+", raw) if len(part) >= 3)
    return (raw,)

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

def _scope_fact_guard_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping):
        value = context.get("scope_fact_guard_enabled")
        if value is not None:
            return _truthy_value(value)
    env_value = os.getenv(SCOPE_FACT_GUARD_ENV)
    if env_value is not None:
        return _truthy_value(env_value)
    return True

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

def _client_message_contains_pii(client_message: str) -> bool:
    text = str(client_message or "")
    return bool(
        re.search(r"\+?\d[\d\s().-]{7,}\d", text)
        or re.search(r"\bдоговор\w*\s*(?:номер|№)?\s*\d+", text, flags=re.I)
        or re.search(r"\b(?:фио|паспорт|снилс|инн|email|e-mail)\b", text, flags=re.I)
    )

def _is_unpk_installment_case(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> bool:
    if _active_brand(context) != "unpk":
        return False
    client_lower = str(client_message or "").casefold()
    if result.message_type != "question" and not any(
        marker in client_lower
        for marker in (
            "?",
            "можно",
            "как",
            "банк",
            "одобр",
            "услов",
            "не все сразу",
            "не всё сразу",
        )
    ):
        return False
    if result.topic_id == "theme:006_installment":
        return True
    haystack = " ".join(
        [
            str(client_message or ""),
            result.draft_text,
            result.broad_group,
            *result.alternative_themes,
            *result.context_warnings,
        ]
    ).casefold()
    return bool(
        "рассроч" in haystack
        or "частями" in haystack
        or "по частям" in haystack
        or "помесяч" in haystack
        or "не все сразу" in haystack
        or "не всё сразу" in haystack
        or "не всю сумму" in haystack
        or "разбить оплат" in haystack
        or "растянуть оплат" in haystack
        or "платить постепенно" in haystack
    )

def _is_unpk_zvsh_case(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> bool:
    if _active_brand(context) != "unpk":
        return False
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
    return "звш" in haystack or ("зимн" in haystack and ("менделеево" in haystack or "лагер" in haystack or "школ" in haystack))

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

def _brand_guarded_result(
    result: SubscriptionDraftResult,
    *,
    reason: str,
    leaked_terms: Sequence[str] = (),
) -> SubscriptionDraftResult:
    precise_condition_flags: tuple[str, ...] = ()
    precise_condition_claims: tuple[str, ...] = ()
    if reason == "brand_unknown_precise_condition_blocked":
        precise_condition_flags = ("unsupported_promise_detected",)
        precise_condition_claims = _extract_numeric_promise_claims(result.draft_text)
    draft_text = (
        CROSS_BRAND_GENERIC_SAFE_TEXT
        if reason == "cross_brand_client_text_blocked"
        else SAFE_FALLBACK_DRAFT_TEXT
    )
    extra_flags = ("cross_brand_safe_template_applied",) if reason == "cross_brand_client_text_blocked" else ()
    return replace(
        result,
        route="manager_only",
        draft_text=draft_text,
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, *precise_condition_claims])),
        safety_flags=tuple(
            dict.fromkeys([*result.safety_flags, reason, "brand_separation_guarded", *extra_flags, *precise_condition_flags])
        ),
        manager_checklist=tuple(
            dict.fromkeys(
                [
                    *result.manager_checklist,
                    "Проверить бренд клиента: в черновике нельзя смешивать Фотон и УНПК.",
                ]
            )
        ),
        metadata={
            **_metadata_with_guarded_original_text(result.metadata, result.draft_text, guard=reason),
            reason: True,
            "forbidden_brand_terms": list(leaked_terms),
            **({"unsupported_promises": list(precise_condition_claims)} if precise_condition_claims else {}),
        },
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

def _step4_keep_answer_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping) and context.get(STEP4_KEEP_ANSWER_ENV) is not None:
        return _truthy_value(context.get(STEP4_KEEP_ANSWER_ENV))
    if isinstance(context, Mapping) and context.get("step4_keep_answer_enabled") is not None:
        return _truthy_value(context.get("step4_keep_answer_enabled"))
    return _truthy_value(os.getenv(STEP4_KEEP_ANSWER_ENV))

def _phase2_objection_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        for key in (PH2_OBJECTION_ENV, "phase2_objection_enabled"):
            if key in context:
                return _truthy_value(context.get(key))
    return _truthy_value(os.getenv(PH2_OBJECTION_ENV))

def _phase2_anxiety_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        for key in (PH2_ANXIETY_ENV, "phase2_anxiety_enabled"):
            if key in context:
                return _truthy_value(context.get(key))
    return _truthy_value(os.getenv(PH2_ANXIETY_ENV))
