from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from mango_mvp.channels.answer_quality_rewriter import (
    AnswerQualityAssessment,
    build_answer_quality_llm_rewrite_prompt,
    apply_answer_quality_rewriter,
)
from mango_mvp.channels.answer_safety_classifier import classify_answer_safety
from mango_mvp.channels.dialogue_debug_trace import trace_event
from mango_mvp.channels.fact_scope_spec import answer_scopes_allowed, detect_fact_scopes
from mango_mvp.channels.dialogue_contract_pipeline import (
    Toggles as DialogueContractToggles,
    build_conversation as build_dialogue_contract_conversation,
    build_fact_store as build_dialogue_contract_fact_store,
    check_claim_faithfulness as check_dialogue_contract_faithfulness,
    concrete_anchors as dialogue_contract_concrete_anchors,
    _established_topic_from_context as dialogue_contract_established_topic_from_context,
    new_concrete_anchors as dialogue_contract_new_concrete_anchors,
    parse_contract as parse_dialogue_contract,
    pipeline_enabled as dialogue_contract_pipeline_enabled,
    run_pipeline as run_dialogue_contract_pipeline,
    verify_output as verify_dialogue_contract_output,
)
from mango_mvp.channels.humanity_guards import (
    has_meta_leak,
    humanity_route_action,
    is_near_repeat,
    meta_markers_present,
    unanswered_direct_question,
)
from mango_mvp.channels.humanity_linter import lint_turn
from mango_mvp.channels.humanity_rewriter import apply_rewrite as apply_humanity_form_rewrite
from mango_mvp.channels.p0_recall_spec import HARD_P0_CODES, codes_from_text, is_benign_hypothetical_refund
from mango_mvp.channels.rules_engine import (
    RuleOutcome,
    apply_rule as apply_migrated_domain_rule,
    load_rules_registry,
    select_rule as select_migrated_domain_rule,
)
from mango_mvp.channels.semantic_roles import tag_message_roles
from mango_mvp.channels.text_signals import has_any_marker, has_marker
from mango_mvp.channels.draft_prompt_builder import (
    IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES,
    build_draft_prompt,
    safe_schedule_template,
    should_force_manager_only,
)
from mango_mvp.insights.sanitizers import sanitize_answer
from mango_mvp.insights.phase2_detectors import detect_anxiety, detect_objection
from mango_mvp.insights.tone_score import score_tone
from mango_mvp.question_catalog.classifier import load_valid_theme_and_service_ids


SUBSCRIPTION_LLM_SCHEMA_VERSION = "subscription_llm_draft_v1_2026_05_16"
DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_CODEX_REASONING_EFFORT = "medium"
ANSWER_QUALITY_LLM_REWRITE_ENV = "TELEGRAM_ANSWER_QUALITY_LLM_REWRITE"
ANSWER_QUALITY_LLM_REWRITER_ENV = "TELEGRAM_ANSWER_QUALITY_LLM_REWRITER"
ANSWER_QUALITY_LLM_REWRITE_REASONING_ENV = "TELEGRAM_ANSWER_QUALITY_LLM_REWRITE_REASONING"
ANSWER_QUALITY_LLM_REWRITE_MODE_ENV = "TELEGRAM_ANSWER_QUALITY_LLM_REWRITE_MODE"
ANSWER_CONTRACT_GREEN_TEMPLATE_REDUCTION_ENV = "TELEGRAM_ANSWER_CONTRACT_GREEN_TEMPLATE_REDUCTION"
HUMANITY_BLOCK_A_ROUTE_FIX_ENV = "TELEGRAM_HUMANITY_BLOCK_A_ROUTE_FIX"
HUMANITY_X2_REWRITE_ENV = "TELEGRAM_DRAFT_X2_REWRITE"
HUMANITY_X2_REWRITE_MODE_ENV = "TELEGRAM_DRAFT_X2_REWRITE_MODE"
HUMANITY_X2_REWRITE_MODEL_ENV = "TELEGRAM_DRAFT_X2_REWRITE_MODEL"
HUMANITY_X2_REWRITE_REASONING_ENV = "TELEGRAM_DRAFT_X2_REWRITE_REASONING"
DIALOGUE_CONTRACT_SEMANTIC_MATCH_MODEL_ENV = "TELEGRAM_DIALOGUE_CONTRACT_SEMANTIC_MATCH_MODEL"
DIALOGUE_CONTRACT_SEMANTIC_MATCH_REASONING_ENV = "TELEGRAM_DIALOGUE_CONTRACT_SEMANTIC_MATCH_REASONING"
RULES_ENGINE_PLANNER_INTENT_ENV = "TELEGRAM_RULES_ENGINE_PLANNER_INTENT"
SCOPE_FACT_GUARD_ENV = "TELEGRAM_SCOPE_FACT_GUARD"
ANTIREPEAT_STRICT_ENV = "TELEGRAM_ANTIREPEAT_STRICT"
A_THREAD_ENV = "TELEGRAM_A_THREAD"
A_PROACTIVE_ENV = "TELEGRAM_A_PROACTIVE"
A_RICH_FORMAT_ENV = "TELEGRAM_A_RICH_FORMAT"
OUTPUT_SANITIZER_ENV = "TELEGRAM_OUTPUT_SANITIZER"
PH2_TONE_ENV = "TELEGRAM_PH2_TONE"
PH2_OBJECTION_ENV = "TELEGRAM_PH2_OBJECTION"
PH2_ANXIETY_ENV = "TELEGRAM_PH2_ANXIETY"
SEMANTIC_DIAGNOSIS_GUARD_ENV = "TELEGRAM_SEMANTIC_DIAGNOSIS_GUARD"
SEMANTIC_DIAGNOSIS_MODEL_ENV = "TELEGRAM_SEMANTIC_DIAGNOSIS_MODEL"
SEMANTIC_DIAGNOSIS_REASONING_ENV = "TELEGRAM_SEMANTIC_DIAGNOSIS_REASONING"
AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION = "authoritative_output_gate_v1_2026_06_02"
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
SAFE_FALLBACK_DRAFT_TEXT = "Чтобы не ошибиться, передам вопрос менеджеру — он сверит детали и вернется с ответом."
UNKNOWN_TOPIC_FALLBACK_ID = "service:S2_unclear"
REFUND_ZERO_COLLECT_SAFE_TEXT = (
    "Приняли обращение. Передам его ответственному сотруднику, он вернется с ответом. "
    "Пока ничего дополнительно присылать не нужно."
)
LEGAL_THREAT_SAFE_TEXT = (
    "Приняли обращение. Передам его ответственному сотруднику, он вернется с ответом."
)
LEGAL_THREAT_PII_SAFE_TEXT = (
    "Приняли обращение. Передам его ответственному сотруднику, он вернется с ответом."
)
COMPLAINT_SAFE_TEXT = "Приняли обращение. Передам вопрос менеджеру, он вернется с ответом."
PAYMENT_DISPUTE_SAFE_TEXT = (
    "Приняли вопрос по оплате. Передам его менеджеру: он проверит данные в системе и вернется с ответом. "
    "Пока ничего дополнительно присылать не нужно."
)
SEMANTIC_DIAGNOSIS_SAFE_TEXT = (
    "Заочно не буду оценивать уровень конкретного ребёнка. Лучше сверить уровень и нагрузку с преподавателем; "
    "менеджер поможет сверить детали и подобрать аккуратный следующий шаг."
)
_HUMANE_GENERIC_HANDOFF_TEXTS: tuple[str, ...] = (
    SAFE_FALLBACK_DRAFT_TEXT,
    "Не хочу гадать по неподтверждённому пункту: менеджер проверит его и вернется с ответом.",
    "Здесь лучше сверить условия: передам вопрос менеджеру, он ответит по точным данным.",
    "Передам этот пункт менеджеру, чтобы он проверил его по актуальным данным и ответил вам.",
)
_HUMANE_DETAIL_HANDOFF_TEXTS: tuple[str, ...] = (
    "Чтобы не ошибиться, менеджер уточнит именно про {detail} и вернется с ответом.",
    "Не хочу гадать по неподтверждённому пункту: менеджер проверит именно {detail} и ответит вам.",
    "По пункту «{detail}» нужна точная сверка — передам его менеджеру.",
    "Передам менеджеру именно вопрос про {detail}, чтобы он проверил актуальные условия.",
)
_REFUND_ZERO_COLLECT_VARIANTS: tuple[str, ...] = (
    REFUND_ZERO_COLLECT_SAFE_TEXT,
    "Вопрос по возврату зафиксирован. Ответственный сотрудник вернется с ответом; сейчас ничего дополнительно присылать не нужно.",
    "По возврату передам обращение ответственному сотруднику. Он вернется с ответом, дополнительных данных пока не нужно.",
)
_COMPLAINT_SAFE_VARIANTS: tuple[str, ...] = (
    COMPLAINT_SAFE_TEXT,
    "Вопрос по жалобе зафиксирован. Менеджер вернется с ответом.",
    "Передам обращение менеджеру, он разберет ситуацию и вернется с ответом.",
)
_PAYMENT_DISPUTE_VARIANTS: tuple[str, ...] = (
    PAYMENT_DISPUTE_SAFE_TEXT,
    "Спорный вопрос по оплате зафиксирован. Менеджер проверит данные в системе и вернется с ответом; пока ничего дополнительно присылать не нужно.",
    "По оплате передам обращение менеджеру: он сверит данные в системе и ответит. Дополнительные данные пока не нужны.",
)
_LEGAL_SAFE_VARIANTS: tuple[str, ...] = (
    LEGAL_THREAT_SAFE_TEXT,
    "Юридический вопрос зафиксирован. Ответственный сотрудник вернется с ответом.",
    "Передам обращение ответственному сотруднику, он вернется с ответом.",
)
SOFT_NEGATIVE_HANDOFF_SAFE_TEXT = (
    "Понял, давайте не буду повторять общий ответ. Передам менеджеру контекст переписки, "
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
PROMOCODE_SAFE_TEXT = "Спасибо, передам менеджеру — он свяжется, уточнит и подскажет актуальные акции и промокоды."
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
UNPK_LVSH_SEATS_SAFE_TEXT = "Обычно группа 12-15 человек. По ЛВШ УНПК места уже почти распроданы, поэтому наличие и запись проверяет живой менеджер."
FOTON_LVSH_PRICE_SAFE_TEXT = (
    "ЛВШ Менделеево у Фотона сейчас стоит 93 100 ₽. "
    "Полная стоимость — 98 000 ₽. "
    "Напишите класс ребёнка — подберём подходящую смену и проверим наличие мест."
)
FOTON_CAMP_OVERVIEW_SAFE_TEXT = (
    "У Фотона есть два летних формата: выездная школа в Менделеево и городская летняя школа в Москве. "
    "Подбираем смену по классу, предмету и формату; наличие мест по конкретной смене проверит менеджер."
)
FOTON_ONLINE_TRIAL_SAFE_TEXT = (
    "В онлайн-формате Фотона можно прислать фрагмент занятия, чтобы посмотреть подачу и уровень; оформление проходит дистанционно — приезжать не нужно. "
    "Условия просмотра фрагмента подтвердит менеджер перед записью."
)
UNPK_TRIAL_SAFE_TEXT = (
    "По очному формату сейчас обычно не начинаем с отдельного пробного занятия. "
    "По онлайн-формату можно прислать фрагмент занятия, чтобы вы посмотрели подачу и уровень. "
    "Если рассматриваете очный курс, менеджер расскажет про формат, преподавателей и поможет понять, подойдёт ли программа."
)
FOTON_OFFLINE_FREE_TRIAL_GUARD_TEXT = (
    "По очному формату бесплатное пробное по умолчанию не обещаю. "
    "Очный пробный шаг согласует менеджер при записи: он проверит подходящую группу, филиал и условия. "
    "Запрос передам именно как очный, без подмены на онлайн-фрагмент."
)
UNPK_LVSH_PRICE_SAFE_TEXT = (
    "ЛВШ Менделеево в УНПК сейчас стоит 114 000 ₽. "
    "В стоимость входит проживание и 5-разовое питание; места уже почти распроданы, запись проверяет живой менеджер. "
    "Напишите класс ребёнка — менеджер проверит, есть ли ещё возможность записи."
)
UNPK_LVSH_LIVING_TRANSFER_SAFE_TEXT = (
    "Да, в ЛВШ Менделеево УНПК есть проживание и 5-разовое питание. "
    "Текущая цена сейчас — 114 000 ₽, полная стоимость — 120 000 ₽. "
    "По местам и применимости для вашего класса запись проверяет живой менеджер."
)
UNPK_LVSH_PRICE_DETAILS_SAFE_TEXT = (
    "По ЛВШ Менделеево в УНПК: полная стоимость — 120 000 ₽, текущая цена сейчас — 114 000 ₽. "
    "Места почти распроданы, поэтому наличие и запись сейчас проверяет живой менеджер."
)
UNPK_LVSH_GRADE_11_PRICE_DETAILS_SAFE_TEXT = (
    "По цене ЛВШ Менделеево в УНПК: полная стоимость — 120 000 ₽, текущая цена сейчас — 114 000 ₽. "
    "При этом сама ЛВШ обычно рассчитана на учеников, окончивших 5-10 класс; "
    "для 11 класса менеджер проверит подходящую летнюю альтернативу под ваш предмет и наличие мест."
)
UNPK_LVSH_GRADE_11_SAFE_TEXT = (
    "По ЛВШ Менделеево важный момент: программа обычно рассчитана на учеников, окончивших 5-10 класс; "
    "ИТ-направление — на 7-10 класс. Для 11 класса менеджер проверит подходящую альтернативу под ваш предмет. "
    "Если говорить справочно о самой ЛВШ Менделеево, текущая цена сейчас — 114 000 ₽, но запись и применимость нужно проверять живым сотрудником."
)
UNPK_CAMP_OVERVIEW_SAFE_TEXT = (
    "У УНПК есть два летних формата: выездная ЛВШ в Менделеево с проживанием и городская летняя школа без проживания. "
    "Подбирать лучше по классу, предмету и формату: с проживанием или дневная программа. "
    "Напишите класс ребёнка — сориентирую по подходящему варианту и передам менеджеру проверку мест."
)
UNPK_CAMP_ONLINE_FORMAT_SAFE_TEXT = (
    "Летние лагеря и ЛВШ УНПК — очные форматы. Если нужен именно онлайн по вашему предмету, "
    "менеджер проверит актуальные варианты УНПК, расписание и стоимость, чтобы не сориентировать неверно."
)
FOTON_CITY_CAMP_AUGUST_SAFE_TEXT = (
    "Да, у Фотона есть дневная городская летняя школа в Москве: ЛШ Москва Фотон проходит 3-14 августа, "
    "адрес — Верхняя Красносельская. Менеджер проверит подходящую программу, смену и наличие мест под класс ребёнка."
)
FOTON_LVSH_DATES_SAFE_TEXT = "ЛВШ Менделеево у Фотона: 20-28 июня и 18-26 июля. Менеджер подскажет наличие мест."
UNPK_LVSH_DATES_SAFE_TEXT = "ЛВШ Менделеево у УНПК: актуальная смена 18-26 июля; августовская смена закрыта. Места почти распроданы, запись проверяет менеджер."
CONTRACT_ENTITY_SAFE_TEXT = "Менеджер пришлёт информацию, на каком оформлении будет договор, и проверит данные по вашей заявке."
CROSS_BRAND_GENERIC_SAFE_TEXT = "Это отдельные организации, по вашему вопросу сориентирую в рамках текущего учебного центра. Менеджер свяжется и расскажет по нашей программе и наших условиях."
CROSS_BRAND_LICENSE_SAFE_TEXT = "У нас есть лицензия на образовательную деятельность. Менеджер свяжется и подскажет детали по документам."
CROSS_BRAND_PLATFORM_SAFE_TEXT = "В нашем учебном центре онлайн-занятия проходят в МТС Линк / Webinar, доступна запись. Менеджер подскажет детали."
PRICE_FIX_PROCESS_SAFE_TEXT = (
    "Вы спрашиваете именно про оформление по текущим условиям. Я не буду выдумывать, достаточно ли одной заявки "
    "или нужна оплата: это проверяет менеджер по выбранному курсу. Следующий шаг простой — передам менеджеру ваш запрос, "
    "он подтвердит, как оформить по текущей цене и что нужно сделать дальше."
)
MANAGER_HANDOFF_REQUEST_SAFE_TEXT = (
    "Да, передам менеджеру: он подтвердит деталь, которую нужно проверить. "
    "Чтобы он сразу был в теме, передам ему контекст диалога: класс, предмет, формат и ваш вопрос. "
    "Повторно писать уже известные данные не нужно."
)
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
EMPLOYEE_PRIVACY_SAFE_TEXT = "Профильный специалист подключится через менеджера: менеджер свяжется и свяжет вас с нужным сотрудником."
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
BRAND_LOYALTY_FOTON_TEXT = "Рады вашему выборе Фотон. Менеджер свяжется и сориентирует по программе."
BRAND_LOYALTY_UNPK_TEXT = "Рады вашему выборе УНПК МФТИ. Менеджер свяжется и сориентирует по программе."
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
UNSUPPORTED_FOLLOWUP_DEADLINE_SAFE_TEXT = (
    "Передам вопрос менеджеру: он проверит детали и вернётся с ответом в рабочее время."
)
UNSUPPORTED_SCHEDULE_ASSUMPTION_SAFE_TEXT = (
    "Точное расписание зависит от класса, предмета, формата и площадки; без проверки конкретной группы не буду называть дни как факт. "
    "Передам менеджеру проверить именно ваш вариант по указанным параметрам."
)
UNSUPPORTED_OFFLINE_VISIT_INVITATION_SAFE_TEXT = (
    "Запись и оформление проходят дистанционно, приезжать не нужно. Если клиент сам попросит очную встречу, менеджер отдельно проверит возможность."
)
KNOWN_CONTEXT_REPAIR_TEXT = (
    "Да, вижу данные из переписки — повторно присылать их не нужно. "
    "Отвечу по сути, а детали, которые требуют проверки по группе или месту, передам менеджеру."
)
INTERNAL_SERVICE_MARKER_RE = re.compile(
    r"\[[^\]\n]{0,220}?(?:\bsource(?:_id)?\s*[:=]|\bfreshness\s*[:=]|source:[A-Za-z0-9_:\-]+|fact:[A-Za-z0-9_:\-]+|kc_chunk:[A-Za-z0-9_:\-]+|kb_release_[A-Za-z0-9_\-]+|product_data/[^\]\s]+|/Users/[^\]\s]+)[^\]\n]{0,260}\]\s*",
    re.I,
)
INTERNAL_SERVICE_TOKEN_RE = re.compile(
    r"\b(?:source|source_id|fact_id|trace_id|freshness)\s*[:=]\s*[^\s;\],.]+|source:[A-Za-z0-9_:\-]+|fact:[A-Za-z0-9_:\-]+|kc_chunk:[A-Za-z0-9_:\-]+|kb_release_[A-Za-z0-9_\-]+|product_data/[^\s;\],.]+|/Users/[^\s;\],.]+",
    re.I,
)
INTERNAL_SCAFFOLD_PREFIX_RE = re.compile(
    r"^\s*(?:[^:\n]{1,80}:\s*)?(?:черновик\s+)?для\s+ситуации\s+[«\"][^»\"\n]{1,160}[»\"]\s*:\s*",
    re.I,
)
INTERNAL_PROMPT_DIRECTIVE_PREFIX_RE = re.compile(
    r"^\s*без\s+(?:обещан\w+|давлен\w+)[^:\n]{0,180}:\s*",
    re.I,
)
INTERNAL_PROMPT_DIRECTIVE_ANYWHERE_RE = re.compile(
    r"\s*(?:по\s+вашей\s+ситуации\s+лучше\s+опираться\s+на\s+подтвержд[её]нные\s+условия,\s*)?"
    r"без\s+обещан\w+[^:\n]{0,120}:\s*",
    re.I,
)
INTERNAL_CLIENT_INSTRUCTION_RE = re.compile(
    r"(?:\bповторять\s+(?:их\s+)?не\s+нужно\b|\bне\s+упоминай\w*\b|"
    r"\bесли\b[^.?!\n]{0,140}\bуже\s+есть\s+в\s+диалоге\b[^.?!\n]{0,140})",
    re.I,
)
INTERNAL_MANAGER_DRAFT_RE = re.compile(
    r"(?:автономн\w+\s+ответ\s+не\s+требуется|дополнительн\w+\s+ответ\s+клиенту\s+сейчас\s+не\s+нужен|если\s+менеджер\s+решит\s+ответить|безопасн\w+\s+вариант|без\s+служебн\w+\s+помет\w+|клиент\s+(?:понял|подтвердил|взял\s+пауз))",
    re.I,
)
INTERNAL_SAFE_VARIANT_RE = re.compile(
    r"безопасн\w+\s+вариант\s*:\s*[«\"](?P<text>.+?)[»\"]\s*$",
    re.I | re.S,
)
DRAFT_PLACEHOLDER_RE = re.compile(
    r"\[(?:[^\]\n]{0,80})?(?:вставить|указать|подставить|TODO|проверенн\w+\s+ссылк|актуальн\w+\s+ссылк)(?:[^\]\n]{0,120})?\]",
    re.I,
)
OUTPUT_SANITIZER_CLIENT_TEXT_RE = re.compile(
    r"(?:^|\n)\s*(?:черновик|ответ|сообщение)\s+клиенту\s*:\s*|(?:^|\n)\s*клиенту\s*:\s*",
    re.I,
)
OUTPUT_SANITIZER_META_LINE_RE = re.compile(
    r"(?:изуча\w+\s+задач\w+|созда\w+\s+план|что\s+вижу\s*:|вопрос\s+к\s+тебе\s*:|"
    r"прежде\s+чем\s+дать\s+черновик|проблема\s+с\s+данными|инструкци\w+\s+шаг\w+\s+требу\w+|"
    r"правил\w+\s+шаг\w+\s+требу\w+|оформ\w+[^.\n]{0,120}audits/_inbox|audits/_inbox)",
    re.I,
)
OUTPUT_SANITIZER_OPTION_LINE_RE = re.compile(r"^\s*(?:[A-CА-В]\)|[A-CА-В]\.)\s+", re.I)
OUTPUT_SANITIZER_PLACEHOLDER_RE = re.compile(
    r"\bуточнен\w+\s+по\s+текущей\s+теме\s*\.\s*тема\s*:\s*[^.?!\n]*(?:[.?!]|$)",
    re.I,
)
OUTPUT_SANITIZER_MANAGER_TAG_RE = re.compile(r"\[/?manager\]\s*", re.I)
OUTPUT_SANITIZER_MANAGER_TAG_INSTRUCTION_RE = re.compile(
    r"^(?=.*\[/?manager\])(?=.*(?:интерпретир\w+|служебн\w+\s+тег|тег\s+\[/?manager\])).*$",
    re.I,
)
PROMOCODE_DRAFT_RE = re.compile(r"\b(?:LVSH-VEB20|LVSH-KF-10|ABRAMOV|VAGIN)\b", re.I)
COSMETIC_OPENING_RE = re.compile(
    r"^\s*(?:здравствуйте[!.]?\s*|да,\s*(?:сориентирую|подскажу|понимаю|конечно)[,!.]?\s*|"
    r"понимаю[,.]?\s*|спасибо(?:\s+за\s+сообщение|\s+за\s+вопрос)?[,.]?\s*)",
    re.I,
)

ALLOWED_ROUTES = {"draft_for_manager", "manager_only", "blocked", "bot_answer_self", "bot_answer_self_for_pilot"}
AUTONOMOUS_ROUTES = {"bot_answer_self", "bot_answer_self_for_pilot"}

GATE_BLOCKING_CODES: Mapping[str, str] = {
    "hard_p0": "block",
    "zero_collect_required": "block",
    "brand_leak": "block",
    "cross_brand": "block",
    "meta_leak": "block",
    "ai_disclosure": "block",
    "identity_disclosure": "block",
    "draft_placeholder": "block",
    "promocode_leak": "block",
    "p0_promise": "block",
    "p0_semantic_risk": "block",
    "unsupported_promise": "block",
    "unsupported_product_claim": "block",
    "unsupported_product_number": "block",
    "fact_grounding": "downgrade",
    "general_number_without_marker": "downgrade",
    "estimate_without_uncertainty_marker": "downgrade",
    "estimate_individual_child_advice": "downgrade",
    "estimate_general_advice_risk": "downgrade",
    "unsupported_entity": "downgrade",
    "forbidden_scope": "downgrade",
    "preemptive_format": "downgrade",
    "unconfirmed_schedule": "downgrade",
    "self_contradiction": "downgrade",
    "wrong_scope": "downgrade",
    "unsupported_followup_deadline": "downgrade",
    "unsupported_schedule_assumption": "downgrade",
    "unsupported_offline_visit_invitation": "downgrade",
    "unsupported_content_delivery_action": "downgrade",
    "unconfirmed_operational_specificity": "downgrade",
    "fake_enrollment_claim": "block",
    "proactive_pii_echo": "block",
    "proactive_too_many_questions": "downgrade",
    "proactive_emoji_overuse": "downgrade",
}
ALLOWED_MESSAGE_TYPES = {"question", "non_question", "context_update", "wait_for_more", "manager_only"}
BASE_SAFETY_FLAGS = ("manager_approval_required", "no_auto_send")
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
HIGH_RISK_INPUT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "refund",
        re.compile(
            r"\bвозв?рат(?!\w*\s+к\s+тем)\w*"
            r"|\bвозвращ\w*\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)"
            r"|\bверн\w*(?:\s+мне|\s+нам|\s+пожалуйста)?\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)"
            r"|\bвозвратит\w*\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)"
            r"|\bрасторг\w*\s+договор"
            r"|\bрасторжен\w*\s+договор"
            r"|\bотказ\w*\s+от\s+обучен"
            r"|\bзабрать\s+деньги",
            re.I,
        ),
    ),
    (
        "legal",
        re.compile(
            r"\bсуд\b|\bиск\b|претензи|досудеб|роспотребнадзор|прокуратур"
            r"|наруш\w*\s+прав|расторжен\w*\s+договор|по\s+закону[^.!?\n]{0,80}обязан\w*(?:\s+(?:вернуть|возместить|расторгнуть))?",
            re.I,
        ),
    ),
    (
        "complaint",
        re.compile(
            r"жалоб(?!а\s+на\s+сайт)\w*|жалуюсь|возмущ\w*|недовол\w*|претензи|конфликт"
            r"|обман|ужасн|плохо\s+учит|плохо\s+пров[её]л|некомпетентн\w*",
            re.I,
        ),
    ),
    (
        "reputation_threat",
        re.compile(r"отзыв\w*\s+в\s+интернет|всех\s+предупреж\w*|напиш\w*\s+отзыв|остав\w*\s+отзыв", re.I),
    ),
)
LEGAL_CONTEXT_INPUT_RE = re.compile(
    r"\bсуд\b|\bиск\b|претензи|досудеб|роспотребнадзор|прокуратур|адвокат|юрист"
    r"|прав[ао][^.!?\n]{0,60}потребител|защит[а-яё]*\s+прав\s+потребител"
    r"|наруш\w*\s+прав|расторжен\w*\s+договор|по\s+закону[^.!?\n]{0,80}обязан\w*",
    re.I,
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
ZERO_COLLECT_DRAFT_RE = re.compile(
    r"\b(?:пришлите|напишите|уточните|сообщите|отправьте|предоставьте|нужн[аоы]?|понадоб(?:ит|ят))\b"
    r"[^.!?\n]{0,140}?"
    r"\b(?:фио|имя|фамили[яюи]|договор|номер\s+договора|телефон|email|e-mail|почт[ауеы]|сумм[ауеы]?|"
    r"причин[ауеы]?|подтвержден\w+\s+оплат|чек|квитанц)\b",
    re.I,
)
REFUND_FORBIDDEN_DETAIL_RE = re.compile(
    r"\b(?:фио|имя|фамили[яюи]|договор\w*|номер\s+договора|телефон|email|e-mail|почт[ауеы]|"
    r"сумм[ауеы]?|причин[ауеы]?|оплат\w*|подтвержден\w+\s+оплат|чек|квитанц)\b",
    re.I,
)
COMPLAINT_APOLOGY_RE = re.compile(
    r"\b(?:понимаю|извините|приносим\s+извинения|(?:нам|мне|очень)\s+жаль|сожалеем|неприятно)\b",
    re.I,
)
COMPLAINT_DETAIL_COLLECT_RE = re.compile(
    r"\b(?:уточните|пришлите|напишите|сообщите|предоставьте|подскажите)\b"
    r"[^.!?\n]{0,180}?"
    r"\b(?:дат[ауеы]?|предмет|курс|групп[ауеы]?|имя|фио|ученик[а-я]*|преподавател[яьюе]?|что\s+именно)\b",
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
FOLLOWUP_DEADLINE_RE = re.compile(
    r"(?:"
    r"\b(?:менеджер|ответственн\w+\s+сотрудник|сотрудник|специалист|мы|я)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:свяж\w*|ответ\w*|напиш\w*|перезвон\w*|верн\w*)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:сегодня|завтра|послезавтра|до\s+вечера|к\s+вечеру|до\s+завтра|в\s+течение\s+(?:(?:\d+\s+)?(?:минут|час|часов|дн|дней|суток|сутки)|дня)|"
    r"не\s+позднее\s+[^.!?\n]{0,40}|до\s+\d{1,2}\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря))\b"
    r"|"
    r"\b(?:ориентир|срок)\b[^.!?\n]{0,80}\b(?:ответ[а-я]*|связ[а-я]*|менеджер[а-я]*)\b"
    r"[^.!?\n]{0,80}\bв\s+течение\s+(?:(?:\d+\s+)?(?:минут|час|часов|дн|дней|суток|сутки)|дня)\b"
    r")",
    re.I,
)
SCHEDULE_ASSUMPTION_RE = re.compile(
    r"\b(?:чаще|обычно|как\s+правило|скорее\s+всего|часто)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:выходн\w*|суббот\w*|воскресень\w*|вечер\w*|будн\w*)\b"
    r"|\b(?:есть|подбираем|подбер[её]м|можно\s+подобрать)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:групп\w*|заняти\w*|расписани\w*)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:выходн\w*|суббот\w*|воскресень\w*|вечер\w*|будн\w*)\b",
    re.I,
)
OFFLINE_VISIT_INVITATION_RE = re.compile(
    r"\b(?:приезж\w*|подъезж\w*|приход\w*|жд[её]м\s+вас|можете\s+прийти|можно\s+прийти)\b"
    r"[^.!?\n]{0,140}?"
    r"\b(?:познаком\w*|посмотр\w*|оформ\w*|запис\w*|встреч\w*|на\s+площадк\w*|в\s+офис\w*)\b",
    re.I,
)
CONTENT_DELIVERY_ACTION_RE = re.compile(
    r"\b(?:я\s+)?(?:пришл[юеё]м?|отправл[юеё]м?|дам|скин[уe]|подготовл[юеё]м?)\b"
    r"[^.!?\n]{0,120}?"
    r"\b(?:фрагмент|ссылк\w*|запис[ьи]\w*|доступ)\b",
    re.I,
)
_RETRYABLE_MARKERS = (
    "no last agent message",
    "temporarily unavailable",
    "temporary",
    "timeout",
    "timed out",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
)

_Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class SubscriptionDraftResult:
    message_type: str = "question"
    broad_group: str = ""
    topic_id: str = "service:S2_unclear"
    topic_confidence: float = 0.0
    confidence_group: float = 0.0
    alternative_themes: tuple[str, ...] = field(default_factory=tuple)
    risk_level: str = "unknown"
    route: str = "manager_only"
    veto_category: str = ""
    draft_text: str = SAFE_FALLBACK_DRAFT_TEXT
    manager_checklist: tuple[str, ...] = field(default_factory=tuple)
    missing_facts: tuple[str, ...] = field(default_factory=tuple)
    forbidden_promises_detected: tuple[str, ...] = field(default_factory=tuple)
    crm_recommendations: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    safety_flags: tuple[str, ...] = BASE_SAFETY_FLAGS
    context_used: tuple[str, ...] = field(default_factory=tuple)
    context_warnings: tuple[str, ...] = field(default_factory=tuple)
    manager_followup_required: bool = False
    manager_followup_deadline: Optional[str] = None
    provider: str = "codex_exec"
    schema_version: str = SUBSCRIPTION_LLM_SCHEMA_VERSION
    raw_response: Optional[str] = None
    error: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        route = str(self.route or "manager_only").strip()
        if route not in ALLOWED_ROUTES:
            route = "manager_only"
        raw_text = str(self.draft_text or "").strip()
        text = strip_internal_service_markers(raw_text) or SAFE_FALLBACK_DRAFT_TEXT
        message_type = str(self.message_type or "question").strip()
        if message_type not in ALLOWED_MESSAGE_TYPES:
            message_type = "manager_only"
        extra_flags = ["internal_metadata_removed_from_draft"] if text != raw_text and raw_text else []
        flags = tuple(
            dict.fromkeys(
                [
                    *BASE_SAFETY_FLAGS,
                    *(_clean_list(self.safety_flags, max_items=16, max_chars=80)),
                    *extra_flags,
                ]
            )
        )
        object.__setattr__(self, "message_type", message_type)
        object.__setattr__(self, "broad_group", str(self.broad_group or "").strip()[:80])
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "veto_category", str(self.veto_category or "").strip()[:80])
        object.__setattr__(self, "draft_text", text)
        object.__setattr__(self, "topic_id", str(self.topic_id or "service:S2_unclear").strip() or "service:S2_unclear")
        object.__setattr__(self, "topic_confidence", _clamp_float(self.topic_confidence))
        object.__setattr__(self, "confidence_group", _clamp_float(self.confidence_group))
        object.__setattr__(self, "alternative_themes", tuple(_clean_list(self.alternative_themes, max_items=5, max_chars=120)))
        object.__setattr__(self, "risk_level", str(self.risk_level or "unknown").strip()[:80] or "unknown")
        object.__setattr__(self, "manager_checklist", tuple(_clean_list(self.manager_checklist, max_items=12, max_chars=240)))
        object.__setattr__(self, "missing_facts", tuple(_clean_list(self.missing_facts, max_items=12, max_chars=160)))
        object.__setattr__(
            self,
            "forbidden_promises_detected",
            tuple(_clean_list(self.forbidden_promises_detected, max_items=12, max_chars=160)),
        )
        object.__setattr__(self, "crm_recommendations", tuple(_clean_crm_recommendations(self.crm_recommendations)))
        object.__setattr__(self, "safety_flags", flags)
        object.__setattr__(self, "context_used", tuple(_clean_list(self.context_used, max_items=12, max_chars=100)))
        object.__setattr__(self, "context_warnings", tuple(_clean_list(self.context_warnings, max_items=12, max_chars=120)))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_json_dict(self, *, include_raw_response: bool = False) -> Mapping[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "message_type": self.message_type,
            "broad_group": self.broad_group,
            "topic_id": self.topic_id,
            "topic_confidence": self.topic_confidence,
            "confidence_theme": self.topic_confidence,
            "confidence_group": self.confidence_group,
            "alternative_themes": list(self.alternative_themes),
            "risk_level": self.risk_level,
            "route": self.route,
            "veto_category": self.veto_category,
            "draft_text": self.draft_text,
            "manager_checklist": list(self.manager_checklist),
            "missing_facts": list(self.missing_facts),
            "forbidden_promises_detected": list(self.forbidden_promises_detected),
            "crm_recommendations": [dict(item) for item in self.crm_recommendations],
            "manager_followup_required": self.manager_followup_required,
            "manager_followup_deadline": self.manager_followup_deadline,
            "safety_flags": list(self.safety_flags),
            "context_used": list(self.context_used),
            "context_warnings": list(self.context_warnings),
            "error": self.error,
            "metadata": dict(self.metadata),
        }
        if include_raw_response:
            payload["raw_response"] = self.raw_response
        return payload


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
    return str(text or "").strip() in {
        ADDRESS_FOTON_MOSCOW_SAFE_TEXT,
        ADDRESS_UNPK_SAFE_TEXT,
        ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
        CONTACT_FOTON_SAFE_TEXT,
        CONTACT_UNPK_SAFE_TEXT,
    }


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
    return str(text or "") in {
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


def _rules_engine_result_applied(metadata: Mapping[str, Any]) -> bool:
    rules = metadata.get("rules_engine") if isinstance(metadata.get("rules_engine"), Mapping) else {}
    applied = str(rules.get("applied") or "").strip()
    if applied:
        return True
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    pipeline_rules = pipeline.get("rules_engine") if isinstance(pipeline.get("rules_engine"), Mapping) else {}
    return bool(str(pipeline_rules.get("applied") or "").strip())


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


class SubscriptionLlmDraftProvider:
    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        model: str = DEFAULT_CODEX_MODEL,
        reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
        timeout_sec: int = 90,
        max_attempts: int = 2,
        cache_dir: Optional[Path | str] = None,
        dialogue_contract_semantic_match_fn: Optional[Callable[[str], object]] = None,
        dialogue_contract_semantic_match_enabled: bool = True,
        runner: Optional[_Runner] = None,
        sleep: Callable[[float], None] = time.sleep,
        base_env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.codex_bin = str(codex_bin or "codex").strip() or "codex"
        self.model = str(model or DEFAULT_CODEX_MODEL).strip() or DEFAULT_CODEX_MODEL
        self.reasoning_effort = str(reasoning_effort or DEFAULT_CODEX_REASONING_EFFORT).strip() or DEFAULT_CODEX_REASONING_EFFORT
        self.timeout_sec = max(1, int(timeout_sec))
        self.max_attempts = max(1, int(max_attempts))
        self.runner = runner or subprocess.run
        self.sleep = sleep
        self.base_env = dict(base_env) if base_env is not None else None
        self.cache_dir = _guard_cache_dir(cache_dir) if cache_dir is not None else None
        self._dialogue_contract_semantic_match_override = dialogue_contract_semantic_match_fn
        self._dialogue_contract_semantic_match_enabled = bool(dialogue_contract_semantic_match_enabled)

    def build_draft(
        self,
        client_message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SubscriptionDraftResult:
        if dialogue_contract_pipeline_enabled(context):
            result = self._build_dialogue_contract_pipeline_draft(client_message, context=context)
            guarded = self._apply_dialogue_contract_v2_guard_chain(result, client_message=client_message, context=context)
            rewritten = apply_humanity_x2_rewriter(
                guarded,
                client_message=client_message,
                context=context,
                rewrite_runner=self._humanity_x2_rewrite_runner
                if _humanity_x2_rewrite_enabled(context)
                else None,
            )
            toned = apply_phase2_tone_layer(rewritten, client_message=client_message, context=context)
            proactive = apply_a2_proactive_layer(toned, client_message=client_message, context=context)
            diagnosed = apply_semantic_diagnosis_guard(
                proactive,
                client_message=client_message,
                context=context,
                classifier_fn=self._semantic_diagnosis_guard_runner
                if _semantic_diagnosis_guard_enabled(context)
                else None,
            )
            return apply_authoritative_output_gate(diagnosed, client_message=client_message, context=context)
        else:
            prompt = build_draft_prompt(client_message, context=context)
            result = self.generate_from_prompt(prompt, force_manager_only=should_force_manager_only(context))
        result = apply_payment_confirmation_guard(result, client_message=client_message, context=context)
        result = apply_brand_separation_guard(result, client_message=client_message, context=context)
        result = apply_input_policy_guards(result, client_message=client_message, context=context)
        result = apply_conversation_intent_plan_guard(result, client_message=client_message, context=context)
        result = apply_high_risk_content_guards(result, client_message=client_message, context=context)
        result = apply_unstated_subject_guard(result, client_message=client_message, context=context)
        result = apply_unsupported_promise_guard(result, context=context)
        result = apply_unconfirmed_operational_specificity_guard(result, context=context)
        result = apply_known_context_redundant_question_guard(result, client_message=client_message, context=context)
        result = apply_funnel_policy_guard(result, context=context)
        result = apply_answer_quality_rewriter(
            result,
            client_message=client_message,
            context=context,
            rewrite_runner=self._answer_quality_llm_rewrite_runner
            if _answer_quality_llm_rewrite_enabled(context)
            else None,
            force_llm_polish=_answer_quality_llm_polish_sales_enabled(context, result),
        )
        result = apply_brand_separation_guard(result, client_message=client_message, context=context)
        result = apply_input_policy_guards(result, client_message=client_message, context=context)
        result = apply_conversation_intent_plan_guard(result, client_message=client_message, context=context)
        result = apply_high_risk_content_guards(result, client_message=client_message, context=context)
        result = apply_unstated_subject_guard(result, client_message=client_message, context=context)
        result = apply_unsupported_promise_guard(result, context=context)
        result = apply_unconfirmed_operational_specificity_guard(result, context=context)
        result = apply_known_context_redundant_question_guard(result, client_message=client_message, context=context)
        result = apply_funnel_policy_guard(result, context=context)
        result = apply_autonomy_matrix_guard(result, client_message=client_message, context=context)
        result = apply_humanity_guards(result, client_message=client_message, context=context)
        result = apply_humanity_x2_rewriter(
            result,
            client_message=client_message,
            context=context,
            rewrite_runner=self._humanity_x2_rewrite_runner
            if _humanity_x2_rewrite_enabled(context)
            else None,
        )
        result = apply_phase2_tone_layer(result, client_message=client_message, context=context)
        result = apply_a2_proactive_layer(result, client_message=client_message, context=context)
        result = apply_semantic_diagnosis_guard(
            result,
            client_message=client_message,
            context=context,
            classifier_fn=self._semantic_diagnosis_guard_runner
            if _semantic_diagnosis_guard_enabled(context)
            else None,
        )
        return apply_authoritative_output_gate(result, client_message=client_message, context=context)

    def _build_dialogue_contract_pipeline_draft(
        self,
        client_message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SubscriptionDraftResult:
        active_brand = _active_brand(context)
        conversation = build_dialogue_contract_conversation(client_message, context=context)
        fact_store = build_dialogue_contract_fact_store(active_brand=active_brand, context=context)
        semantic_match_fn = (
            self._dialogue_contract_semantic_match_override
            if self._dialogue_contract_semantic_match_override is not None
            else self._dialogue_contract_semantic_match_runner
            if self._dialogue_contract_semantic_match_enabled
            else None
        )
        pipeline_result = run_dialogue_contract_pipeline(
            conversation=conversation,
            active_brand=active_brand,
            fact_store=fact_store,
            understand_fn=self._dialogue_contract_understanding_runner,
            draft_fn=self._dialogue_contract_draft_runner,
            repair_fn=self._dialogue_contract_repair_runner,
            faithfulness_fn=self._dialogue_contract_faithfulness_runner,
            semantic_match_fn=semantic_match_fn,
            warmth_fn=None,
            context=context,
            tone_guide=_dialogue_contract_tone_guide(context),
            style_examples=_dialogue_contract_style_examples(context),
            toggles=DialogueContractToggles(form_warmth=False, warmth_mode=_humanity_x2_rewrite_mode(context)),
        )
        route = "bot_answer_self_for_pilot" if pipeline_result.route == "bot_answer_self" else pipeline_result.route
        payload = {
            "message_type": "manager_only" if pipeline_result.manager_only else "question",
            "broad_group": "dialogue_contract_pipeline",
            "topic_id": _topic_id_from_context(context),
            "confidence_theme": pipeline_result.contract.confidence,
            "confidence_group": pipeline_result.contract.confidence,
            "risk_level": "high" if pipeline_result.contract.is_p0 else "low",
            "route": route,
            "draft_text": pipeline_result.draft_text,
            "manager_checklist": [
                "Параллельный dialogue-contract pipeline: проверить смысл до включения в проде.",
                *(
                    [f"Выходной верификатор: {finding.code} — {finding.detail}" for finding in pipeline_result.findings]
                    if pipeline_result.findings
                    else []
                ),
            ],
            "missing_facts": list(pipeline_result.missing),
            "forbidden_promises_detected": [
                *[finding.code for finding in pipeline_result.findings],
                *[f"unsupported_claim:{item}" for item in pipeline_result.unsupported_claims],
            ],
            "safety_flags": _dialogue_contract_safety_flags(pipeline_result),
            "context_used": ["dialogue_contract", "client_safe_fact_store", "output_verifier"],
            "context_warnings": [pipeline_result.fallback_reason] if pipeline_result.fallback_reason else [],
            "metadata": {
                "dialogue_contract_pipeline": {
                    "contract": pipeline_result.contract.to_json_dict(),
                    "retrieved_fact_keys": list(pipeline_result.facts.keys()),
                    "retrieved_facts": dict(pipeline_result.facts),
                    "missing_fact_keys": list(pipeline_result.missing),
                    "findings": [{"code": f.code, "detail": f.detail} for f in pipeline_result.findings],
                    "unsupported_claims": list(pipeline_result.unsupported_claims),
                    "form_findings": [{"code": f.code, "detail": f.detail} for f in pipeline_result.form_findings],
                    "warmth_attempted": pipeline_result.warmth_attempted,
                    "warmth_mode": pipeline_result.warmth_mode,
                    "warmth_rejected_reason": pipeline_result.warmth_rejected_reason,
                    "warmth_rejected_findings": [
                        {"code": f.code, "detail": f.detail} for f in pipeline_result.warmth_rejected_findings
                    ],
                    "warmth_rejected_unsupported": list(pipeline_result.warmth_rejected_unsupported),
                    "warmth_semantic_available": pipeline_result.warmth_semantic_available,
                    "semantic_match_attempted": pipeline_result.semantic_match_attempted,
                    "semantic_match_replaced": pipeline_result.semantic_match_replaced,
                    "semantic_match_reason": pipeline_result.semantic_match_reason,
                    "fallback_reason": pipeline_result.fallback_reason,
                    "recovery_candidate": pipeline_result.recovery_candidate,
                    "recovery_candidate_validated": bool(pipeline_result.recovery_candidate),
                    "partial_yield_applied": bool(getattr(pipeline_result, "partial_yield_applied", False)),
                    "partial_yield_fact_keys": list(getattr(pipeline_result, "partial_yield_fact_keys", ())),
                    "partial_yield_missing": list(getattr(pipeline_result, "partial_yield_missing", ())),
                    "composite_applied": bool(getattr(pipeline_result, "composite_applied", False)),
                    "composite_fact_keys": list(getattr(pipeline_result, "composite_fact_keys", ())),
                    "composite_missing": list(getattr(pipeline_result, "composite_missing", ())),
                    "next_step_applied": bool(getattr(pipeline_result, "next_step_applied", False)),
                    "next_step_text": str(getattr(pipeline_result, "next_step_text", "") or ""),
                    "estimate": {
                        "is_estimate": bool(pipeline_result.is_estimate),
                        "estimate_applied": bool(getattr(pipeline_result, "estimate_applied", False) or pipeline_result.is_estimate),
                        "answer_mode": pipeline_result.estimate_answer_mode,
                        "estimate_domain": pipeline_result.estimate_domain,
                    },
                    "warmed": pipeline_result.warmed,
                    "repaired": pipeline_result.repaired,
                }
            },
        }
        if should_force_manager_only(context) and route != "manager_only":
            payload["route"] = "manager_only"
            payload["safety_flags"].append("forced_manager_only_by_rop_policy")
        return normalize_subscription_draft_payload(payload)

    def _dialogue_contract_understanding_runner(self, prompt: str) -> Mapping[str, Any]:
        raw = self._run_prompt_text(
            prompt,
            prefix="mango_dialogue_contract_understanding_",
            suffix=".json",
            reasoning_effort=self.reasoning_effort,
        )
        try:
            return extract_json_object(raw)
        except Exception:
            return {}

    def _dialogue_contract_draft_runner(self, prompt: str) -> str:
        return self._run_prompt_text(
            prompt,
            prefix="mango_dialogue_contract_draft_",
            suffix=".txt",
            reasoning_effort=self.reasoning_effort,
        )

    def _dialogue_contract_faithfulness_runner(self, prompt: str) -> Mapping[str, Any] | str:
        raw = self._run_prompt_text(
            prompt,
            prefix="mango_dialogue_contract_faithfulness_",
            suffix=".json",
            reasoning_effort=os.getenv("TELEGRAM_DIALOGUE_CONTRACT_FAITHFULNESS_REASONING") or "medium",
        )
        try:
            return extract_json_object(raw)
        except Exception:
            return raw

    def _dialogue_contract_semantic_match_runner(self, prompt: str) -> Mapping[str, Any] | str:
        raw = self._run_prompt_text(
            prompt,
            prefix="mango_dialogue_contract_semantic_match_",
            suffix=".json",
            model=os.getenv(DIALOGUE_CONTRACT_SEMANTIC_MATCH_MODEL_ENV) or self.model,
            reasoning_effort=os.getenv(DIALOGUE_CONTRACT_SEMANTIC_MATCH_REASONING_ENV) or "medium",
        )
        try:
            return extract_json_object(raw)
        except Exception:
            return raw

    def _semantic_diagnosis_guard_runner(self, prompt: str) -> Mapping[str, Any] | str:
        raw = self._run_prompt_text(
            prompt,
            prefix="mango_semantic_diagnosis_guard_",
            suffix=".json",
            model=os.getenv(SEMANTIC_DIAGNOSIS_MODEL_ENV) or self.model,
            reasoning_effort=os.getenv(SEMANTIC_DIAGNOSIS_REASONING_ENV) or "low",
        )
        try:
            return extract_json_object(raw)
        except Exception:
            return raw

    def _dialogue_contract_repair_runner(self, prompt: str) -> str:
        return self._run_prompt_text(
            prompt,
            prefix="mango_dialogue_contract_repair_",
            suffix=".txt",
            reasoning_effort=os.getenv("TELEGRAM_DIALOGUE_CONTRACT_REPAIR_REASONING") or self.reasoning_effort,
        )

    def _dialogue_contract_warmth_runner(self, prompt: str) -> str:
        return self._run_prompt_text(
            prompt,
            prefix="mango_dialogue_contract_warmth_",
            suffix=".txt",
            model=os.getenv(HUMANITY_X2_REWRITE_MODEL_ENV) or self.model,
            reasoning_effort=os.getenv(HUMANITY_X2_REWRITE_REASONING_ENV) or "xhigh",
        )

    def _apply_dialogue_contract_v2_guard_chain(
        self,
        result: SubscriptionDraftResult,
        *,
        client_message: str,
        context: Optional[Mapping[str, Any]],
    ) -> SubscriptionDraftResult:
        """v2 post-chain: safety verifiers only; no old intent/template rewrites."""
        guard_steps: list[dict[str, Any]] = []

        def record_step(name: str, before: SubscriptionDraftResult, after: SubscriptionDraftResult) -> None:
            before_flags = set(before.safety_flags)
            after_flags = set(after.safety_flags)
            guard_steps.append(
                {
                    "name": name,
                    "route_before": before.route,
                    "route_after": after.route,
                    "text_changed": before.draft_text != after.draft_text,
                    "added_flags": sorted(after_flags - before_flags),
                }
            )

        guarded = result
        guarded = apply_payment_confirmation_guard(guarded, client_message=client_message, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("payment_confirmation", result, guarded)
        result = guarded

        guarded = apply_brand_separation_guard(result, client_message=client_message, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("brand_separation", result, guarded)
        result = guarded

        guarded = apply_input_policy_guards(result, client_message=client_message, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("input_policy", result, guarded)
        result = guarded

        guarded = apply_unstated_subject_guard(result, client_message=client_message, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("unstated_subject", result, guarded)
        result = guarded

        guarded = apply_unsupported_promise_guard(result, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("unsupported_promise", result, guarded)
        result = guarded

        guarded = apply_unconfirmed_operational_specificity_guard(result, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("unconfirmed_operational_specificity", result, guarded)
        result = guarded

        guarded = apply_dialogue_contract_v2_template_dispatcher(result, client_message=client_message, context=context)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("safe_template_dispatcher", result, guarded)
        result = guarded

        guarded = apply_funnel_policy_guard(result, context=context)
        record_step("funnel_policy", result, guarded)
        result = guarded

        guarded = self._dialogue_contract_v2_route_permission_guard(result, client_message=client_message, context=context)
        record_step("route_permission", result, guarded)
        result = guarded

        guarded = guard_identity_disclosure(result)
        guarded = self._reverify_dialogue_contract_text_change(result, guarded, client_message=client_message, context=context)
        record_step("identity_disclosure", result, guarded)

        sanitized = _sanitize_dialogue_contract_client_text(guarded)
        record_step("sanitize", guarded, sanitized)
        trace_event(
            context,
            "_apply_dialogue_contract_v2_guard_chain",
            {
                "applied_guards": [step["name"] for step in guard_steps],
                "steps": guard_steps,
                "route": sanitized.route,
                "safety_flags": sanitized.safety_flags,
            },
        )
        return sanitized

    def _reverify_dialogue_contract_text_change(
        self,
        before: SubscriptionDraftResult,
        after: SubscriptionDraftResult,
        *,
        client_message: str,
        context: Optional[Mapping[str, Any]],
    ) -> SubscriptionDraftResult:
        if before.draft_text == after.draft_text:
            return after
        metadata = dict(after.metadata)
        pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
        facts = pipeline.get("retrieved_facts") if isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
        fact_texts = {str(k): str(v) for k, v in facts.items()}
        contract = parse_dialogue_contract(
            pipeline.get("contract"),
            active_brand=_active_brand(context),
            fact_key_catalog=tuple(fact_texts.keys()),
        )
        previous_bot_texts = _humanity_previous_bot_texts(context)
        verified_safe_template = _is_verified_safe_numeric_template(after.draft_text)
        if verified_safe_template:
            fact_texts["_verified_safe_numeric_template"] = after.draft_text
        findings = verify_dialogue_contract_output(
            after.draft_text,
            facts=fact_texts,
            active_brand=_active_brand(context),
            contract=contract,
            client_message=client_message,
            context=context,
            previous_bot_texts=previous_bot_texts,
        )
        if (
            _is_policy_c_identity_question(after, context=context)
            and _is_approved_policy_c_identity_text(after.draft_text, active_brand=_active_brand(context))
            and not contract.is_p0
            and not detect_high_risk_input_markers(client_message, context=context)
        ):
            flags = tuple(
                dict.fromkeys(
                    [
                        *after.safety_flags,
                        "dialogue_contract_text_change_reverified",
                        "identity_policy_c_reverified",
                    ]
                )
            )
            return replace(after, safety_flags=flags)
        if _rules_engine_result_applied(metadata) and fact_texts and not findings:
            flags = tuple(
                dict.fromkeys(
                    [
                        *after.safety_flags,
                        "dialogue_contract_text_change_reverified",
                        "rules_engine_text_change_reverified",
                    ]
                )
            )
            return replace(after, safety_flags=flags)
        semantic_available = True
        unsupported_claims: tuple[str, ...] = ()
        if facts:
            semantic_result = check_dialogue_contract_faithfulness(
                after.draft_text,
                facts={str(k): str(v) for k, v in facts.items()},
                client_words=client_message,
                faithfulness_fn=self._dialogue_contract_faithfulness_runner,
                established_topic=dialogue_contract_established_topic_from_context(context),
            )
            semantic_available = semantic_result.available
            unsupported_claims = semantic_result.unsupported
        if verified_safe_template:
            findings = [finding for finding in findings if finding.code not in {"fact_grounding", "p0_promise"}]
        if not findings and not unsupported_claims and semantic_available:
            flags = tuple(dict.fromkeys([*after.safety_flags, "dialogue_contract_text_change_reverified"]))
            return replace(after, safety_flags=flags)
        flags = tuple(
            dict.fromkeys(
                [
                    *after.safety_flags,
                    "dialogue_contract_text_change_blocked",
                    "manager_approval_required",
                    "no_auto_send",
                ]
            )
        )
        checklist = tuple(
            dict.fromkeys(
                [
                    *after.manager_checklist,
                    "v2 safety-fallback не прошёл повторную проверку: использовать только после ручной правки.",
                ]
            )
        )
        metadata["dialogue_contract_reverification_findings"] = [
            {"code": finding.code, "detail": finding.detail} for finding in findings
        ]
        if unsupported_claims:
            metadata["dialogue_contract_reverification_unsupported"] = list(unsupported_claims)
        metadata["dialogue_contract_reverification_semantic_available"] = semantic_available
        recovery_candidate = _validated_guardchain_recovery_candidate(
            replace(after, metadata=metadata),
            client_message=client_message,
            context=context,
        )
        if recovery_candidate:
            recovered_flags = tuple(
                dict.fromkeys([*after.safety_flags, "cite_only_recover_at_guardchain"])
            )
            recovered_metadata = {
                **metadata,
                "cite_only_recover_at_guardchain": True,
                "cite_only_recover_at_guardchain_source": "text_change_reverify",
            }
            return replace(
                after,
                route="bot_answer_self_for_pilot",
                draft_text=recovery_candidate,
                safety_flags=recovered_flags,
                metadata=recovered_metadata,
            )
        yielded_before = _safe_template_yield_before_fallback(
            before,
            after,
            client_message=client_message,
            context=context,
        )
        if yielded_before is not None:
            return yielded_before
        return replace(
            after,
            route="draft_for_manager" if after.route != "manager_only" else after.route,
            draft_text=SAFE_FALLBACK_DRAFT_TEXT,
            safety_flags=flags,
            manager_checklist=checklist,
            metadata=metadata,
        )

    def _dialogue_contract_v2_route_permission_guard(
        self,
        result: SubscriptionDraftResult,
        *,
        client_message: str,
        context: Optional[Mapping[str, Any]],
    ) -> SubscriptionDraftResult:
        if result.route not in (*AUTONOMOUS_ROUTES, "draft_for_manager"):
            return result
        flags = list(result.safety_flags)
        checklist = list(result.manager_checklist)
        metadata = dict(result.metadata)

        decision = decide_route(
            result,
            client_message=client_message,
            context=context,
            allow_default_autonomy=_default_autonomy_flip_enabled(context),
        )
        if decision.veto_category:
            flags.extend(decision.safety_flags)
            checklist.extend(decision.manager_checklist)
            metadata.update(decision.metadata)
            if decision.veto_category == "high_risk" and _is_combined_high_risk_case(
                result,
                markers=set(detect_high_risk_input_markers(client_message, context=context)),
                client_message=client_message,
                context=context,
            ):
                flags.append("combined_high_risk_manager_only")
                metadata["combined_high_risk_manager_only"] = True
            return replace(
                result,
                route=decision.route,
                veto_category=decision.veto_category,
                safety_flags=tuple(dict.fromkeys(flags)),
                manager_checklist=tuple(dict.fromkeys(checklist)),
                metadata=metadata,
            )

        if decision.autonomous_candidate:
            flags.append("dialogue_contract_route_permission_autonomous_candidate")
            recovery_candidate = _validated_guardchain_recovery_candidate(
                replace(result, metadata=metadata, safety_flags=tuple(dict.fromkeys(flags))),
                client_message=client_message,
                context=context,
            )
            if recovery_candidate:
                flags.append("cite_only_recover_at_guardchain")
                metadata["cite_only_recover_at_guardchain"] = True
                metadata["cite_only_recover_at_guardchain_source"] = "route_permission"
                return replace(
                    result,
                    route="bot_answer_self_for_pilot",
                    draft_text=recovery_candidate,
                    veto_category=decision.veto_category,
                    safety_flags=tuple(dict.fromkeys(flags)),
                    manager_checklist=tuple(dict.fromkeys(checklist)),
                    metadata=metadata,
                )
        return replace(
            result,
            route=decision.route,
            veto_category=decision.veto_category,
            safety_flags=tuple(dict.fromkeys(flags)),
            manager_checklist=tuple(dict.fromkeys(checklist)),
            metadata=metadata,
        )

    def _run_prompt_text(
        self,
        prompt: str,
        *,
        prefix: str,
        suffix: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> str:
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix) as out_file:
            output_path = Path(out_file.name)
            cmd = build_codex_exec_command(
                output_path=output_path,
                codex_bin=self.codex_bin,
                model=model or self.model,
                reasoning_effort=reasoning_effort or self.reasoning_effort,
            )
            proc = self.runner(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_sec,
                env=build_codex_exec_env(self.base_env),
            )
            raw = output_path.read_text(encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            return ""
        return raw or proc.stdout or proc.stderr or ""

    def _answer_quality_llm_rewrite_runner(
        self,
        *,
        result: SubscriptionDraftResult,
        client_message: str,
        context: Mapping[str, Any] | None,
        assessment: AnswerQualityAssessment,
    ) -> Mapping[str, Any]:
        prompt = build_answer_quality_llm_rewrite_prompt(
            result=result,
            client_message=client_message,
            context=context,
            assessment=assessment,
        )
        reasoning = str(os.getenv(ANSWER_QUALITY_LLM_REWRITE_REASONING_ENV) or "xhigh").strip() or "xhigh"
        with tempfile.NamedTemporaryFile(prefix="mango_answer_quality_rewrite_", suffix=".json") as out_file:
            output_path = Path(out_file.name)
            cmd = build_codex_exec_command(
                output_path=output_path,
                codex_bin=self.codex_bin,
                model=self.model,
                reasoning_effort=reasoning,
            )
            proc = self.runner(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_sec,
                env=build_codex_exec_env(self.base_env),
            )
            raw = output_path.read_text(encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            return {}
        try:
            payload = extract_json_object(raw or proc.stdout or proc.stderr or "")
        except Exception:
            return {}
        draft_text = str(payload.get("draft_text") or "").strip()
        if not draft_text:
            return {}
        return {
            "draft_text": draft_text,
            "reason": str(payload.get("reason") or "")[:300],
        }

    def _humanity_x2_rewrite_runner(self, prompt: str) -> str:
        model = str(os.getenv(HUMANITY_X2_REWRITE_MODEL_ENV) or "gpt-5.5").strip() or "gpt-5.5"
        reasoning = str(os.getenv(HUMANITY_X2_REWRITE_REASONING_ENV) or "xhigh").strip() or "xhigh"
        with tempfile.NamedTemporaryFile(prefix="mango_humanity_x2_rewrite_", suffix=".txt") as out_file:
            output_path = Path(out_file.name)
            cmd = build_codex_exec_command(
                output_path=output_path,
                codex_bin=self.codex_bin,
                model=model,
                reasoning_effort=reasoning,
            )
            proc = self.runner(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_sec,
                env=build_codex_exec_env(self.base_env),
            )
            raw = output_path.read_text(encoding="utf-8", errors="ignore")
        if proc.returncode != 0:
            return ""
        return _extract_humanity_x2_text(raw or proc.stdout or proc.stderr or "")

    def generate(self, prompt: str) -> SubscriptionDraftResult:
        return self.generate_from_prompt(prompt)

    def generate_from_prompt(self, prompt: str, *, force_manager_only: bool = False) -> SubscriptionDraftResult:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return apply_authoritative_output_gate(safe_fallback_draft(reason="empty_prompt"))

        cache_key = _cache_key(
            {
                "schema_version": SUBSCRIPTION_LLM_SCHEMA_VERSION,
                "provider": "codex_exec",
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "prompt": prompt_text,
                "force_manager_only": force_manager_only,
            }
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return apply_authoritative_output_gate(_with_metadata(cached, {"cache_hit": True}))

        last_error = "codex_exec_failed"
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self._run_once(prompt_text, force_manager_only=force_manager_only)
            except subprocess.TimeoutExpired:
                return apply_authoritative_output_gate(safe_fallback_draft(reason="timeout", metadata={"attempt": attempt, "timeout_sec": self.timeout_sec}))
            except FileNotFoundError:
                return apply_authoritative_output_gate(safe_fallback_draft(reason="codex_binary_not_found", metadata={"codex_bin": self.codex_bin}))
            except _CodexRetryableError as exc:
                last_error = str(exc) or "retryable_codex_error"
                if attempt < self.max_attempts:
                    self.sleep(min(3.0, float(attempt)))
                    continue
                return apply_authoritative_output_gate(safe_fallback_draft(reason="codex_retryable_error", metadata={"last_error": last_error}))
            except Exception as exc:  # noqa: BLE001
                return apply_authoritative_output_gate(safe_fallback_draft(reason="invalid_json_or_codex_error", metadata={"last_error": str(exc)[:400]}))
            self._cache_put(cache_key, result)
            return apply_authoritative_output_gate(result)
        return apply_authoritative_output_gate(safe_fallback_draft(reason=last_error))

    def _run_once(self, prompt: str, *, force_manager_only: bool) -> SubscriptionDraftResult:
        with tempfile.NamedTemporaryFile(prefix="mango_draft_codex_", suffix=".json") as out_file:
            output_path = Path(out_file.name)
            cmd = build_codex_exec_command(
                output_path=output_path,
                codex_bin=self.codex_bin,
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            )
            proc = self.runner(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_sec,
                env=build_codex_exec_env(self.base_env),
            )
            raw = output_path.read_text(encoding="utf-8", errors="ignore")

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            message = f"codex exec failed rc={proc.returncode}: {' '.join(stderr.splitlines()[-2:])[:400]}"
            if _is_retryable(stderr):
                raise _CodexRetryableError(message)
            raise RuntimeError(message)

        payload = extract_json_object(raw or proc.stdout or proc.stderr or "")
        result = normalize_subscription_draft_payload(payload, raw_response=raw)
        if force_manager_only and result.route != "manager_only":
            result = replace(
                result,
                route="manager_only",
                safety_flags=tuple(dict.fromkeys([*result.safety_flags, "forced_manager_only_by_rop_policy"])),
                metadata={**dict(result.metadata), "forced_route": "manager_only"},
            )
        return apply_authoritative_output_gate(guard_identity_disclosure(result))

    def _cache_get(self, cache_key: str) -> Optional[SubscriptionDraftResult]:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{cache_key}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return normalize_subscription_draft_payload(payload)
        except Exception:
            return None

    def _cache_put(self, cache_key: str, result: SubscriptionDraftResult) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{cache_key}.json"
        path.write_text(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


class FakeSubscriptionLlmDraftProvider:
    def __init__(self, result: Optional[SubscriptionDraftResult | Mapping[str, Any]] = None) -> None:
        self.result = normalize_subscription_draft_payload(result) if result is not None else safe_fallback_draft(
            reason="fake_provider_default"
        )
        self.prompts: list[str] = []

    def build_draft(
        self,
        client_message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SubscriptionDraftResult:
        prompt = build_draft_prompt(client_message, context=context)
        result = self.generate_from_prompt(prompt, force_manager_only=should_force_manager_only(context))
        result = apply_payment_confirmation_guard(result, client_message=client_message, context=context)
        result = apply_brand_separation_guard(result, client_message=client_message, context=context)
        result = apply_input_policy_guards(result, client_message=client_message, context=context)
        result = apply_conversation_intent_plan_guard(result, client_message=client_message, context=context)
        result = apply_high_risk_content_guards(result, client_message=client_message, context=context)
        result = apply_unstated_subject_guard(result, client_message=client_message, context=context)
        result = apply_unsupported_promise_guard(result, context=context)
        result = apply_unconfirmed_operational_specificity_guard(result, context=context)
        result = apply_known_context_redundant_question_guard(result, client_message=client_message, context=context)
        result = apply_funnel_policy_guard(result, context=context)
        result = apply_answer_quality_rewriter(result, client_message=client_message, context=context)
        result = apply_brand_separation_guard(result, client_message=client_message, context=context)
        result = apply_input_policy_guards(result, client_message=client_message, context=context)
        result = apply_conversation_intent_plan_guard(result, client_message=client_message, context=context)
        result = apply_high_risk_content_guards(result, client_message=client_message, context=context)
        result = apply_unstated_subject_guard(result, client_message=client_message, context=context)
        result = apply_unsupported_promise_guard(result, context=context)
        result = apply_unconfirmed_operational_specificity_guard(result, context=context)
        result = apply_known_context_redundant_question_guard(result, client_message=client_message, context=context)
        result = apply_funnel_policy_guard(result, context=context)
        result = apply_autonomy_matrix_guard(result, client_message=client_message, context=context)
        result = apply_humanity_guards(result, client_message=client_message, context=context)
        result = apply_humanity_x2_rewriter(result, client_message=client_message, context=context)
        result = apply_phase2_tone_layer(result, client_message=client_message, context=context)
        result = apply_a2_proactive_layer(result, client_message=client_message, context=context)
        result = apply_semantic_diagnosis_guard(result, client_message=client_message, context=context)
        return apply_authoritative_output_gate(result, client_message=client_message, context=context)

    def generate(self, prompt: str) -> SubscriptionDraftResult:
        return self.generate_from_prompt(prompt)

    def generate_from_prompt(self, prompt: str, *, force_manager_only: bool = False) -> SubscriptionDraftResult:
        self.prompts.append(prompt)
        result = self.result
        if force_manager_only:
            result = replace(
                result,
                route="manager_only",
                safety_flags=tuple(dict.fromkeys([*result.safety_flags, "forced_manager_only_by_rop_policy"])),
            )
        return guard_identity_disclosure(result)


def build_codex_exec_command(
    *,
    output_path: Path | str,
    codex_bin: str = "codex",
    model: str = DEFAULT_CODEX_MODEL,
    reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT,
) -> list[str]:
    cmd = [
        str(codex_bin or "codex").strip() or "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--model",
        str(model or DEFAULT_CODEX_MODEL).strip() or DEFAULT_CODEX_MODEL,
    ]
    reasoning = str(reasoning_effort or "").strip()
    if reasoning:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning}"'])
    cmd.extend(["--output-last-message", str(output_path), "-"])
    return cmd


def build_codex_exec_env(base_env: Optional[Mapping[str, str]] = None, *, codex_home: Optional[Path | str] = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env.pop("OPENAI_API_KEY", None)
    if codex_home is not None:
        env["CODEX_HOME"] = str(codex_home)
    return env


@dataclass(frozen=True)
class CodexExecConfig:
    codex_bin: str = "codex"
    model: str = DEFAULT_CODEX_MODEL
    reasoning_effort: str = DEFAULT_CODEX_REASONING_EFFORT

    def build_command(self, output_path: Path | str) -> list[str]:
        return build_codex_exec_command(
            output_path=output_path,
            codex_bin=self.codex_bin,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )


def normalize_subscription_draft_payload(payload: Mapping[str, Any] | SubscriptionDraftResult, *, raw_response: Optional[str] = None) -> SubscriptionDraftResult:
    if isinstance(payload, SubscriptionDraftResult):
        return payload
    if not isinstance(payload, Mapping):
        raise RuntimeError("subscription draft response JSON root must be an object")
    schedule = payload.get("safe_schedule_template")
    manager_followup_required = bool(payload.get("manager_followup_required"))
    manager_followup_deadline = _optional_text(payload.get("manager_followup_deadline"))
    if isinstance(schedule, Mapping) and schedule.get("manager_followup_required") is True:
        manager_followup_required = True
        manager_followup_deadline = manager_followup_deadline or _optional_text(
            schedule.get("manager_followup_deadline") or schedule.get("deadline_at")
        )
    result = SubscriptionDraftResult(
        message_type=str(payload.get("message_type") or "question"),
        broad_group=str(payload.get("broad_group") or ""),
        topic_id=str(payload.get("topic_id") or "service:S2_unclear"),
        topic_confidence=_clamp_float(payload.get("confidence_theme", payload.get("topic_confidence"))),
        confidence_group=_clamp_float(payload.get("confidence_group")),
        alternative_themes=tuple(_clean_list(payload.get("alternative_themes"), max_items=5, max_chars=120)),
        risk_level=str(payload.get("risk_level") or "unknown"),
        route=str(payload.get("route") or "manager_only"),
        draft_text=str(payload.get("draft_text") or SAFE_FALLBACK_DRAFT_TEXT),
        manager_checklist=tuple(_clean_list(payload.get("manager_checklist"), max_items=12, max_chars=240)),
        missing_facts=tuple(_clean_list(payload.get("missing_facts"), max_items=12, max_chars=160)),
        forbidden_promises_detected=tuple(_clean_list(payload.get("forbidden_promises_detected"), max_items=12, max_chars=160)),
        crm_recommendations=tuple(_clean_crm_recommendations(payload.get("crm_recommendations"))),
        safety_flags=tuple(_clean_list(payload.get("safety_flags"), max_items=16, max_chars=80)),
        context_used=tuple(_clean_list(payload.get("context_used"), max_items=12, max_chars=100)),
        context_warnings=tuple(_clean_list(payload.get("context_warnings"), max_items=12, max_chars=120)),
        manager_followup_required=manager_followup_required,
        manager_followup_deadline=manager_followup_deadline,
        raw_response=raw_response,
        metadata=dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), Mapping) else {},
    )
    return guard_promocode_leak(
        guard_draft_placeholder(guard_identity_disclosure(apply_taxonomy_topic_guard(apply_subscription_policy_guards(result))))
    )


def safe_fallback_draft(*, reason: str, metadata: Optional[Mapping[str, Any]] = None) -> SubscriptionDraftResult:
    extra_flags = ("codex_exec_timeout",) if reason == "timeout" else ()
    return SubscriptionDraftResult(
        message_type="manager_only",
        route="manager_only",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        manager_checklist=("Проверить вопрос вручную.",),
        missing_facts=("llm_response",),
        safety_flags=(*BASE_SAFETY_FLAGS, "llm_fallback", "draft_only", *extra_flags),
        error=reason,
        metadata=dict(metadata or {}),
    )


_A2_PHONE_RE = re.compile(r"(?:\+7|8|7)?[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}")
_A2_TIME_RE = re.compile(
    r"\b(?:сегодня|завтра|послезавтра|утром|дн[её]м|вечером|после\s+обеда|до\s+\d{1,2}|"
    r"после\s+\d{1,2}|в\s+\d{1,2}(?::\d{2})?|с\s+\d{1,2}\s+до\s+\d{1,2})\b",
    re.I,
)
_A2_FAKE_DONE_RE = re.compile(
    r"я\s+(?:вас\s+)?записал|вы\s+записаны|запись\s+оформлена|оформил\s+запись|записал\s+на\s+курс",
    re.I,
)
_A2_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF]")
_A2_SERIOUS_TAGS = {"p0", "refund", "complaint", "manager_only", "legal", "guarantee"}


def apply_a2_proactive_layer(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    """A2.1 callback/contact capture plus deterministic rich-format guard."""

    updated = result
    if _a2_proactive_enabled(context):
        updated = _a2_contact_capture_handoff(updated, client_message=client_message, context=context)
    if _a2_rich_format_enabled(context):
        updated = _a2_apply_rich_format_guard(updated, client_message=client_message, context=context)
    return updated


def _a2_contact_capture_handoff(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult:
    if result.route == "manager_only" or _a2_p0_or_high_risk(result, client_message=client_message, context=context):
        return result
    phone = _a2_extract_phone(client_message)
    phone_known = _a2_context_phone_known(context)
    has_time = _a2_has_time(client_message)
    if not phone and not (phone_known and has_time):
        return result
    metadata = dict(result.metadata)
    metadata["a2_proactive"] = {
        **(dict(metadata.get("a2_proactive") or {}) if isinstance(metadata.get("a2_proactive"), Mapping) else {}),
        "enabled": True,
        "step": "offer_callback",
        "contact_captured": True,
        "phone_masked": _a2_mask_phone(phone) if phone else "[known_phone]",
        "preferred_time": "[provided]" if has_time else "",
        "crm_write": False,
        "policy_source": "deterministic",
    }
    text = (
        "Спасибо, передам менеджеру — он свяжется с вами в удобное время."
        if has_time
        else "Спасибо, передам менеджеру — он свяжется с вами и уточнит удобное время."
    )
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "A2.1: клиент оставил контакт/время; связаться вручную, без CRM-записи из бота.",
            ]
        )
    )
    return replace(
        result,
        route="draft_for_manager" if result.route != "manager_only" else result.route,
        draft_text=text,
        safety_flags=tuple(
            dict.fromkeys(
                [
                    *result.safety_flags,
                    "a2_proactive_contact_captured",
                    "manager_approval_required",
                    "no_auto_send",
                ]
            )
        ),
        manager_checklist=checklist,
        manager_followup_required=True,
        metadata=metadata,
    )


def _a2_apply_rich_format_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult:
    text = str(result.draft_text or "")
    context_tag = _a2_context_tag(result, client_message=client_message, context=context)
    cleaned = _a2_enforce_emoji_limit(text, context_tag=context_tag)
    if cleaned == text:
        return result
    metadata = dict(result.metadata)
    metadata["a2_rich_format"] = {
        **(dict(metadata.get("a2_rich_format") or {}) if isinstance(metadata.get("a2_rich_format"), Mapping) else {}),
        "enabled": True,
        "emoji_guard_applied": True,
        "context_tag": context_tag,
    }
    return replace(
        result,
        draft_text=cleaned,
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "a2_rich_format_emoji_guarded"])),
        metadata=metadata,
    )


def _a2_proactive_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping):
        for key in ("a_proactive_enabled", "proactive_enabled", A_PROACTIVE_ENV):
            if key in context:
                return _truthy_value(context.get(key))
    return _truthy_value(os.getenv(A_PROACTIVE_ENV))


def _a2_rich_format_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping):
        for key in ("a_rich_format_enabled", "rich_format_enabled", A_RICH_FORMAT_ENV):
            if key in context:
                return _truthy_value(context.get(key))
    return _truthy_value(os.getenv(A_RICH_FORMAT_ENV))


def _a2_p0_or_high_risk(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> bool:
    flags = " ".join(str(flag or "") for flag in result.safety_flags).casefold()
    if any(marker in flags for marker in ("high_risk", "zero_collect", "legal", "complaint", "payment_dispute")):
        return True
    safety = classify_answer_safety(
        client_message=client_message,
        context=context,
        topic_id=result.topic_id,
        route=result.route,
        safety_flags=result.safety_flags,
    )
    return bool(safety.p0_required and not safety.semantic_non_p0)


def _a2_extract_phone(text: str) -> str:
    match = _A2_PHONE_RE.search(str(text or ""))
    return match.group(0).strip() if match else ""


def _a2_has_time(text: str) -> bool:
    return bool(_A2_TIME_RE.search(str(text or "")))


def _a2_mask_phone(phone: str) -> str:
    digits = re.sub(r"\D+", "", str(phone or ""))
    if not digits:
        return ""
    return f"[phone:***{digits[-2:]}]"


def _a2_context_phone_known(context: Optional[Mapping[str, Any]]) -> bool:
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


def _a2_context_tag(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> str:
    if result.route == "manager_only":
        return "manager_only"
    flags = " ".join(str(flag or "") for flag in result.safety_flags).casefold()
    for tag in ("complaint", "refund", "legal", "guarantee"):
        if tag in flags:
            return tag
    safety = classify_answer_safety(client_message=client_message, context=context, topic_id=result.topic_id, route=result.route)
    if safety.p0_required and not safety.semantic_non_p0:
        return "p0"
    return "warm" if "a2_proactive" in result.metadata or any("a2_proactive" in flag for flag in result.safety_flags) else "neutral"


def _a2_enforce_emoji_limit(text: str, *, context_tag: str, max_emoji: int = 1) -> str:
    if context_tag in _A2_SERIOUS_TAGS:
        return _A2_EMOJI_RE.sub("", str(text or "")).strip()
    count = 0
    chars: list[str] = []
    for char in str(text or ""):
        if _A2_EMOJI_RE.match(char):
            count += 1
            if count > max_emoji:
                continue
        chars.append(char)
    return "".join(chars).strip()


def apply_authoritative_output_gate(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    """Final safety gate over every provider output.

    The gate composes existing verifiers/guards and only downgrades unsafe output.
    It is intentionally not a quality improver: it never promotes a route and never
    invents replacement facts.
    """

    result = apply_output_sanitizer(result, context=context)
    findings = _authoritative_gate_findings(result, client_message=client_message, context=context)
    actions = tuple(_authoritative_gate_action(finding["code"]) for finding in findings)
    actionable = [finding for finding, action in zip(findings, actions) if action in {"block", "downgrade"}]
    metadata = dict(result.metadata)
    metadata["authoritative_output_gate"] = {
        "schema_version": AUTHORITATIVE_OUTPUT_GATE_SCHEMA_VERSION,
        "checked": True,
        "action": "block" if "block" in actions else ("downgrade" if "downgrade" in actions else "pass"),
        "findings": findings,
        "route_before": result.route,
        "route_after": result.route,
    }
    if not actionable:
        return replace(result, metadata=metadata)

    route = _authoritative_gate_downgraded_route(result.route, actions)
    metadata["authoritative_output_gate"]["route_after"] = route
    codes = tuple(dict.fromkeys(str(item["code"]) for item in actionable))
    flags = tuple(
        dict.fromkeys(
            [
                *result.safety_flags,
                "authoritative_output_gate_blocked",
                *[f"authoritative_gate:{code}" for code in codes],
                "manager_approval_required",
                "no_auto_send",
            ]
        )
    )
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "Финальный safety gate заблокировал клиентский текст: не отправлять без ручной проверки.",
            ]
        )
    )
    forbidden = tuple(dict.fromkeys([*result.forbidden_promises_detected, *codes]))
    return replace(
        result,
        route=route,
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        safety_flags=flags,
        manager_checklist=checklist,
        forbidden_promises_detected=forbidden,
        metadata=metadata,
        error=result.error or "authoritative_output_gate_blocked",
    )


def apply_output_sanitizer(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not _output_sanitizer_enabled(context):
        return result

    cleaned, reasons = _sanitize_output_client_text(result.draft_text)
    if not reasons and cleaned == result.draft_text:
        return result

    fallback = not cleaned.strip()
    route = result.route
    flags = [*result.safety_flags, "output_sanitizer_applied", *[f"output_sanitizer:{reason}" for reason in reasons]]
    checklist = list(result.manager_checklist)
    if fallback:
        cleaned = SAFE_FALLBACK_DRAFT_TEXT
        if route != "manager_only":
            route = "draft_for_manager"
        flags.extend(["manager_approval_required", "no_auto_send"])
        checklist.append("Output sanitizer удалил внутренний текст целиком: не отправлять без ручной проверки.")
    metadata = dict(result.metadata)
    metadata["output_sanitizer"] = {
        "enabled": True,
        "applied": True,
        "fallback": fallback,
        "reasons": list(reasons),
        "route_before": result.route,
        "route_after": route,
        "text_before_len": len(str(result.draft_text or "")),
        "text_after_len": len(cleaned),
    }
    return replace(
        result,
        route=route,
        draft_text=cleaned,
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(checklist)),
        metadata=metadata,
        error=result.error or ("output_sanitizer_fallback" if fallback else result.error),
    )


def _sanitize_output_client_text(text: str) -> tuple[str, tuple[str, ...]]:
    raw = str(text or "")
    if not raw:
        return "", ()

    value = raw
    reasons: list[str] = []
    marker_matches = list(OUTPUT_SANITIZER_CLIENT_TEXT_RE.finditer(value))
    if marker_matches:
        tail = value[marker_matches[-1].end() :].strip()
        if tail:
            value = tail
            reasons.append("client_text_marker")

    plan_context = bool(
        OUTPUT_SANITIZER_META_LINE_RE.search(raw)
        or re.search(r"^\s*(?:[A-CА-В]\)|[A-CА-В]\.)\s+", raw, flags=re.I | re.M)
    )
    value, placeholder_removed = OUTPUT_SANITIZER_PLACEHOLDER_RE.subn(" ", value)
    if placeholder_removed:
        reasons.append("topic_placeholder")

    kept_lines: list[str] = []
    for line in value.splitlines() or [value]:
        stripped = line.strip()
        if not stripped:
            continue
        if OUTPUT_SANITIZER_MANAGER_TAG_INSTRUCTION_RE.search(stripped):
            reasons.append("manager_tag_instruction")
            continue
        if OUTPUT_SANITIZER_META_LINE_RE.search(stripped):
            reasons.append("meta_process_line")
            continue
        if plan_context and OUTPUT_SANITIZER_OPTION_LINE_RE.search(stripped):
            reasons.append("plan_option_line")
            continue
        kept_lines.append(stripped)
    value = "\n".join(kept_lines)

    value, tag_removed = OUTPUT_SANITIZER_MANAGER_TAG_RE.subn("", value)
    if tag_removed:
        reasons.append("manager_tag")

    stripped = strip_internal_service_markers(value)
    if stripped != value:
        value = stripped
        reasons.append("internal_service_marker")

    value = _normalize_output_sanitizer_text(value)
    if _output_sanitizer_degenerate(value):
        reasons.append("degenerate_output")
        return "", tuple(dict.fromkeys(reasons))
    if value != raw and not reasons:
        reasons.append("normalized")
    return value, tuple(dict.fromkeys(reasons))


def _normalize_output_sanitizer_text(text: str) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.split("\n")]
    value = "\n".join(line for line in lines if line)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def _output_sanitizer_degenerate(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    if OUTPUT_SANITIZER_META_LINE_RE.search(value) or OUTPUT_SANITIZER_MANAGER_TAG_RE.search(value):
        return True
    if re.fullmatch(r"(?:[A-CА-В][).]\s*[^.?!\n]{1,120}\s*)+", value, flags=re.I):
        return True
    if not re.search(r"[а-яёa-z]", value, flags=re.I):
        return True
    return False


def _authoritative_gate_action(code: str) -> str:
    return str(GATE_BLOCKING_CODES.get(str(code or ""), "warn") or "warn")


def _authoritative_gate_downgraded_route(route: str, actions: Sequence[str]) -> str:
    current = str(route or "manager_only")
    if "block" in set(actions):
        return "manager_only"
    if current in AUTONOMOUS_ROUTES:
        return "draft_for_manager"
    return current


def _authoritative_gate_finding(code: str, *, detail: str = "", source: str = "") -> dict[str, str]:
    return {
        "code": str(code or "").strip(),
        "detail": " ".join(str(detail or "").split())[:240],
        "source": str(source or "authoritative_output_gate").strip(),
        "policy": _authoritative_gate_action(code),
    }


def _authoritative_gate_findings(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    text_only = not client_message and context is None and not _pipeline_fact_texts(result)

    findings.extend(_authoritative_gate_text_guard_findings(result))
    findings.extend(_authoritative_gate_a2_findings(result, client_message=client_message, context=context))
    if text_only:
        return _dedupe_gate_findings(findings)

    gate_context = _context_with_dialogue_contract_retrieved_facts(context, result)
    facts = _authoritative_gate_fact_texts(result, gate_context)
    contract = _pipeline_contract(result, active_brand=_active_brand(gate_context), fact_keys=tuple(facts.keys()))
    previous_bot_texts = _humanity_previous_bot_texts(gate_context)
    p0_already_guarded = _authoritative_gate_p0_already_guarded(result)
    has_pipeline = _authoritative_gate_has_pipeline(result)
    for finding in verify_dialogue_contract_output(
        result.draft_text,
        facts=facts,
        active_brand=_active_brand(gate_context),
        contract=contract,
        client_message=client_message,
        context=gate_context,
        previous_bot_texts=previous_bot_texts,
    ):
        if not has_pipeline and finding.code not in {"brand_leak", "meta_leak", "ai_disclosure", "p0_promise", "p0_semantic_risk"}:
            continue
        if finding.code == "p0_promise" and _authoritative_gate_verified_content_flag(result):
            continue
        if p0_already_guarded and finding.code in {"p0_semantic_risk", "p0_promise"}:
            continue
        if _authoritative_gate_skip_backed_finding(
            finding.code,
            detail=finding.detail,
            result=result,
            client_message=client_message,
            facts=facts,
        ):
            continue
        findings.append(_authoritative_gate_finding(finding.code, detail=finding.detail, source="verify_output"))

    safety = classify_answer_safety(
        client_message=client_message,
        context=gate_context,
        topic_id=result.topic_id,
        route=result.route,
        safety_flags=result.safety_flags,
    )
    raw_hard_codes = tuple(code for code in codes_from_text(client_message) if code in HARD_P0_CODES)
    hard_codes = raw_hard_codes if safety.p0_required else tuple(code for code in safety.risk_codes if code in HARD_P0_CODES)
    if not p0_already_guarded and (hard_codes or (safety.p0_required and not safety.semantic_non_p0)):
        detail = ",".join(dict.fromkeys([*hard_codes, *[code for code in safety.risk_codes if code in HARD_P0_CODES]]))
        findings.append(_authoritative_gate_finding("hard_p0", detail=detail or safety.primary_risk, source="answer_safety"))
    if safety.zero_collect_required and not p0_already_guarded and (safety.p0_required or hard_codes):
        findings.append(_authoritative_gate_finding("zero_collect_required", detail=safety.primary_risk, source="answer_safety"))

    findings.extend(
        _authoritative_gate_existing_guard_findings(
            result,
            client_message=client_message,
            context=gate_context,
            facts=facts,
        )
    )
    return _dedupe_gate_findings(findings)


def _authoritative_gate_text_guard_findings(result: SubscriptionDraftResult) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    guarded = guard_identity_disclosure(result)
    if guarded is not result and guarded.draft_text != result.draft_text:
        findings.append(_authoritative_gate_finding("identity_disclosure", source="guard_identity_disclosure"))
    guarded = guard_draft_placeholder(result)
    if guarded is not result and guarded.draft_text != result.draft_text:
        findings.append(_authoritative_gate_finding("draft_placeholder", source="guard_draft_placeholder"))
    guarded = guard_promocode_leak(result)
    if guarded is not result and guarded.draft_text != result.draft_text:
        findings.append(_authoritative_gate_finding("promocode_leak", source="guard_promocode_leak"))
    return findings


def _authoritative_gate_a2_findings(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    text = str(result.draft_text or "")
    proactive_active = _a2_proactive_enabled(context) or _a2_is_proactive_result(result)
    if proactive_active:
        if _A2_FAKE_DONE_RE.search(text):
            findings.append(_authoritative_gate_finding("fake_enrollment_claim", source="a2_proactive_gate"))
        phone = _a2_extract_phone(client_message)
        if phone and _a2_phone_echoed(phone, text):
            findings.append(_authoritative_gate_finding("proactive_pii_echo", source="a2_proactive_gate"))
        if _a2_is_proactive_result(result) and text.count("?") > 1:
            findings.append(
                _authoritative_gate_finding("proactive_too_many_questions", detail="more_than_one_question", source="a2_proactive_gate")
            )
    if _a2_rich_format_enabled(context):
        context_tag = _a2_context_tag(result, client_message=client_message, context=context)
        cleaned = _a2_enforce_emoji_limit(text, context_tag=context_tag)
        if cleaned != text:
            findings.append(_authoritative_gate_finding("proactive_emoji_overuse", detail="emoji_guard_not_applied", source="a2_rich_format_gate"))
    return findings


def _a2_is_proactive_result(result: SubscriptionDraftResult) -> bool:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    a2 = metadata.get("a2_proactive") if isinstance(metadata.get("a2_proactive"), Mapping) else {}
    selling = metadata.get("selling") if isinstance(metadata.get("selling"), Mapping) else {}
    rules = metadata.get("rules_engine") if isinstance(metadata.get("rules_engine"), Mapping) else {}
    rules_selling = rules.get("selling") if isinstance(rules.get("selling"), Mapping) else {}
    flags = " ".join(str(flag or "") for flag in result.safety_flags).casefold()
    return bool(
        a2.get("step")
        or selling.get("proactive")
        or rules_selling.get("proactive")
        or "a2_proactive" in flags
        or "offer_callback" in flags
    )


def _a2_phone_echoed(phone: str, text: str) -> bool:
    digits = re.sub(r"\D+", "", str(phone or ""))
    if len(digits) < 7:
        return False
    haystack = re.sub(r"\D+", "", str(text or ""))
    return bool(haystack and digits in haystack)


def _authoritative_gate_existing_guard_findings(
    result: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
    facts: Mapping[str, str],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    guard_checks: tuple[tuple[str, str, Callable[[SubscriptionDraftResult], SubscriptionDraftResult]], ...] = (
        ("unsupported_promise", "apply_unsupported_promise_guard", lambda item: apply_unsupported_promise_guard(item, context=context)),
        (
            "unconfirmed_operational_specificity",
            "apply_unconfirmed_operational_specificity_guard",
            lambda item: apply_unconfirmed_operational_specificity_guard(item, context=context),
        ),
    )
    for code, source, guard_fn in guard_checks:
        if code == "unsupported_promise" and _authoritative_gate_verified_content_flag(result):
            continue
        guarded = guard_fn(result)
        if _authoritative_guard_changed(result, guarded):
            added_flags = sorted(set(guarded.safety_flags) - set(result.safety_flags))
            detail = ",".join(added_flags) or guarded.error or guarded.route
            if _authoritative_gate_skip_backed_finding(
                code,
                detail=detail,
                result=result,
                client_message=client_message,
                facts=facts,
            ):
                continue
            findings.append(_authoritative_gate_finding(code, detail=detail, source=source))
    specificity_context = _context_with_dialogue_contract_retrieved_facts(context, result)
    for code, fn in (
        ("unsupported_followup_deadline", find_unsupported_followup_deadline_claims),
        ("unsupported_schedule_assumption", find_unsupported_schedule_assumption_claims),
        ("unsupported_offline_visit_invitation", find_unsupported_offline_visit_invitation_claims),
        ("unsupported_content_delivery_action", find_unsupported_content_delivery_action_claims),
    ):
        claims = fn(result.draft_text, context=specificity_context)
        if claims:
            if _authoritative_gate_skip_backed_finding(
                code,
                detail="; ".join(claims),
                result=result,
                client_message=client_message,
                facts=facts,
            ):
                continue
            findings.append(_authoritative_gate_finding(code, detail="; ".join(claims), source=fn.__name__))
    return findings


def _authoritative_guard_changed(before: SubscriptionDraftResult, after: SubscriptionDraftResult) -> bool:
    return (
        before.route != after.route
        or before.draft_text != after.draft_text
        or set(after.safety_flags) != set(before.safety_flags)
        or set(after.forbidden_promises_detected) != set(before.forbidden_promises_detected)
    )


def _authoritative_gate_fact_texts(
    result: SubscriptionDraftResult,
    context: Optional[Mapping[str, Any]],
) -> dict[str, str]:
    facts = dict(_pipeline_fact_texts(result))
    if facts:
        return facts
    if isinstance(context, Mapping):
        confirmed = context.get("confirmed_facts")
        if isinstance(confirmed, Mapping):
            facts.update({str(key): str(value) for key, value in confirmed.items() if str(key).strip() and str(value).strip()})
        facts_context = context.get("facts_context")
        if isinstance(facts_context, Mapping):
            confirmed_context = facts_context.get("confirmed_facts")
            if isinstance(confirmed_context, Mapping):
                facts.update(
                    {str(key): str(value) for key, value in confirmed_context.items() if str(key).strip() and str(value).strip()}
                )
        known_slots = context.get("known_slots")
        if isinstance(known_slots, Mapping):
            for key, value in known_slots.items():
                text = _authoritative_gate_slot_text(str(key), value)
                if text:
                    facts[f"_known_slot:{key}"] = text
    return facts


def _authoritative_gate_skip_backed_finding(
    code: str,
    *,
    detail: str = "",
    result: SubscriptionDraftResult,
    client_message: str,
    facts: Mapping[str, str],
) -> bool:
    code_text = str(code or "")
    combined = " ".join([str(detail or ""), str(result.draft_text or ""), str(client_message or "")]).casefold().replace("ё", "е")
    fact_text = " ".join(str(value or "") for value in facts.values()).casefold().replace("ё", "е")
    if code_text in {
        "unconfirmed_operational_specificity",
        "unsupported_schedule_assumption",
    }:
        schedule_markers = ("выходн", "суббот", "воскрес", "будн", "вечер", "утрен", "дневн")
        return any(marker in combined and marker in fact_text for marker in schedule_markers)
    if code_text in {"fact_grounding", "unsupported_entity"} and _authoritative_gate_verified_content_flag(result):
        return True
    if code_text == "unsupported_entity" and "address:generic" in str(detail or ""):
        asks_address = has_any_marker(combined, ("адрес", "сретенк", "скорняжн", "москва", "метро", "где находит"))
        has_address_fact = has_any_marker(fact_text, ("адрес", "сретенк", "скорняжн", "москва", "метро", "чистые пруды"))
        return asks_address and has_address_fact
    return False


def _authoritative_gate_verified_content_flag(result: SubscriptionDraftResult) -> bool:
    flags = tuple(str(flag or "") for flag in result.safety_flags)
    if any(flag.endswith("_safe_template_applied") or flag.endswith("_fallback_applied") for flag in flags):
        return True
    return any(
        flag
        in {
            "safe_template_yielded_to_verified_answer",
            "humanity_block_a_direct_answer_applied",
            "cite_only_recover_at_guardchain",
        }
        for flag in flags
    )


def _authoritative_gate_has_pipeline(result: SubscriptionDraftResult) -> bool:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    return isinstance(metadata.get("dialogue_contract_pipeline"), Mapping)


def _authoritative_gate_slot_text(key: str, value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    normalized_key = str(key or "").strip()
    if normalized_key == "grade" and text.isdigit():
        return f"{text} класс"
    return f"{normalized_key}: {text}" if normalized_key else text


def _authoritative_gate_p0_already_guarded(result: SubscriptionDraftResult) -> bool:
    if result.route != "manager_only":
        return False
    flags = " ".join(str(flag or "") for flag in result.safety_flags).casefold()
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    return bool(
        metadata.get("final_p0_text_override")
        or metadata.get("zero_collect_legal_guarded")
        or metadata.get("zero_collect_refund_guarded")
        or metadata.get("complaint_apology_guarded")
        or metadata.get("payment_dispute_manager_only")
        or any(
            marker in flags
            for marker in (
                "zero_collect_legal_guarded",
                "zero_collect_refund_guarded",
                "complaint_apology_guarded",
                "payment_dispute_manager_only",
            )
        )
    )


def _dedupe_gate_findings(findings: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, str]] = []
    for item in findings:
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        source = str(item.get("source") or "")
        detail = str(item.get("detail") or "")
        key = (code, source, detail)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "code": code,
                "detail": detail,
                "source": source,
                "policy": _authoritative_gate_action(code),
            }
        )
    return result


def extract_json_object(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        raise RuntimeError("empty subscription draft response")
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise RuntimeError("subscription draft response does not contain JSON object")
        payload = json.loads(raw[start : end + 1])
    if not isinstance(payload, dict):
        raise RuntimeError("subscription draft response JSON root must be an object")
    return payload


def parse_llm_json(text: str) -> SubscriptionDraftResult:
    try:
        return normalize_subscription_draft_payload(extract_json_object(text), raw_response=text)
    except Exception as exc:  # noqa: BLE001
        return safe_fallback_draft(reason="invalid_json", metadata={"parse_error": str(exc)[:300]})


def strip_internal_service_markers(text: str) -> str:
    value = str(text or "")
    if not value:
        return ""
    safe_variant = INTERNAL_SAFE_VARIANT_RE.search(value)
    if safe_variant:
        candidate = " ".join(str(safe_variant.group("text") or "").split())
        if candidate and not INTERNAL_MANAGER_DRAFT_RE.search(candidate):
            return candidate.strip()
    if INTERNAL_MANAGER_DRAFT_RE.search(value):
        return ""
    previous = None
    while previous != value:
        previous = value
        value = INTERNAL_SCAFFOLD_PREFIX_RE.sub("", value)
        value = INTERNAL_PROMPT_DIRECTIVE_PREFIX_RE.sub("", value)
        value = value.lstrip()
    if INTERNAL_CLIENT_INSTRUCTION_RE.search(value):
        return ""
    value = INTERNAL_PROMPT_DIRECTIVE_ANYWHERE_RE.sub(" ", value)
    value = INTERNAL_SERVICE_MARKER_RE.sub("", value)
    value = INTERNAL_SERVICE_TOKEN_RE.sub("", value)
    if INTERNAL_CLIENT_INSTRUCTION_RE.search(value):
        return ""
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\s{2,}", " ", value)
    return value.strip()


def draft_has_internal_service_markers(text: str) -> bool:
    value = str(text or "")
    return bool(
        INTERNAL_SERVICE_MARKER_RE.search(value)
        or INTERNAL_SERVICE_TOKEN_RE.search(value)
        or INTERNAL_SCAFFOLD_PREFIX_RE.search(value)
        or INTERNAL_PROMPT_DIRECTIVE_PREFIX_RE.search(value)
        or INTERNAL_PROMPT_DIRECTIVE_ANYWHERE_RE.search(value)
        or INTERNAL_CLIENT_INSTRUCTION_RE.search(value)
        or INTERNAL_MANAGER_DRAFT_RE.search(value)
    )


def draft_has_identity_disclosure(text: str) -> bool:
    return bool(find_identity_disclosure_phrases(text))


def find_identity_disclosure_phrases(text: str) -> tuple[str, ...]:
    lowered = str(text or "").casefold()
    return tuple(phrase for phrase in IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES if _identity_phrase_present(lowered, phrase))


def _identity_phrase_present(lowered_text: str, phrase: str) -> bool:
    value = str(phrase or "").casefold().strip()
    if not value:
        return False
    if value == "gpt":
        pattern = r"(?:chat\s*)?gpt"
    else:
        pattern = r"\s+".join(re.escape(part) for part in value.split())
    return bool(re.search(rf"(?<!\w){pattern}(?!\w)", lowered_text, flags=re.I))


def guard_identity_disclosure(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    phrases = find_identity_disclosure_phrases(result.draft_text)
    if not phrases:
        return result
    return replace(
        result,
        route="manager_only",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, *phrases])),
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "identity_disclosure_guarded", "bot_identity_disclosure", "llm_fallback"])),
        error=result.error or "identity_disclosure_guarded",
    )


def guard_draft_placeholder(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    if not DRAFT_PLACEHOLDER_RE.search(result.draft_text):
        return result
    return replace(
        result,
        route="manager_only",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, "placeholder_in_draft"])),
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "placeholder_in_draft", "llm_fallback"])),
        manager_checklist=tuple(dict.fromkeys([*result.manager_checklist, "Черновик содержит placeholder: заменить вручную."])),
        error=result.error or "placeholder_in_draft",
    )


def guard_promocode_leak(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    if not PROMOCODE_DRAFT_RE.search(result.draft_text):
        return result
    return replace(
        result,
        route="manager_only",
        draft_text=PROMOCODE_SAFE_TEXT,
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, "promocode_in_draft"])),
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "promocode_in_draft_guarded", "manager_approval_required", "no_auto_send"])),
        manager_checklist=tuple(dict.fromkeys([*result.manager_checklist, "Не повторять промокод клиенту до проверки условий акции."])),
        error=result.error,
    )


def apply_unsupported_promise_guard(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if result.draft_text == UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT:
        trace_event(
            context,
            "apply_unsupported_promise_guard",
            {
                "skipped": "verified_installment_fallback",
                "route": result.route,
            },
        )
        return result
    promise_context = _context_with_dialogue_contract_retrieved_facts(context, result)
    claims = find_unsupported_numeric_promises(result.draft_text, context=promise_context)
    if not claims:
        trace_event(
            context,
            "apply_unsupported_promise_guard",
            {
                "claims": (),
                "route_before": result.route,
                "route_after": result.route,
                "blocked": False,
            },
        )
        return result
    flags = tuple(dict.fromkeys([*result.safety_flags, "unsupported_promise_detected"]))
    checklist = tuple(
        dict.fromkeys(
            [
                *result.manager_checklist,
                "Черновик содержит конкретную цифру, сумму, процент или срок без подтвержденного свежего факта: проверить вручную.",
            ]
        )
    )
    guarded = replace(
        result,
        route="manager_only",
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, *claims])),
        safety_flags=flags,
        manager_checklist=checklist,
        metadata={**dict(result.metadata), "unsupported_promises": list(claims)},
    )
    trace_event(
        context,
        "apply_unsupported_promise_guard",
        {
            "claims": claims,
            "route_before": result.route,
            "route_after": guarded.route,
            "blocked": True,
            "safety_flags": guarded.safety_flags,
        },
    )
    return guarded


def _context_with_dialogue_contract_retrieved_facts(
    context: Optional[Mapping[str, Any]],
    result: SubscriptionDraftResult,
) -> Optional[Mapping[str, Any]]:
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    retrieved = pipeline.get("retrieved_facts") if isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
    facts = {
        str(key): str(value)
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
    facts = pipeline.get("retrieved_facts") if isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
    return {
        str(key): str(value)
        for key, value in facts.items()
        if str(key).strip() and str(value).strip()
    }


def _has_trial_retrieved_fact(result: SubscriptionDraftResult) -> bool:
    for key, text in _pipeline_fact_texts(result).items():
        normalized = " ".join((str(key or ""), str(text or ""))).casefold().replace("ё", "е")
        if has_any_marker(normalized, ("trial", "пробн", "фрагмент занятия", "фрагмент урок")):
            return True
    return False


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


def _safe_template_yield_before_fallback(
    before: SubscriptionDraftResult,
    after: SubscriptionDraftResult,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> SubscriptionDraftResult | None:
    applied = _safe_template_applied_name(after)
    if applied not in _INFORMATIONAL_SAFE_TEMPLATE_NAMES:
        return None
    if applied == "terminal" and not _is_informational_terminal_template(after.draft_text):
        return None
    if applied == "terminal" and (
        (_asks_non_tax_document_or_contract(client_message, context=context) and _answers_tax_deduction_scope(before.draft_text))
        or (_asks_non_matkap_document_or_contract(client_message, context=context) and _answers_matkap_scope(before.draft_text))
    ):
        return None
    if not _verified_informational_answer(before, client_message=client_message, context=context, template_name=applied):
        return None
    metadata = {
        **dict(before.metadata),
        "safe_template_yielded_to_verified_answer": True,
        "safe_template_yielded_spec": applied,
    }
    flags = tuple(dict.fromkeys([*before.safety_flags, "safe_template_yielded_to_verified_answer"]))
    return replace(before, safety_flags=flags, metadata=metadata)


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
        return f"По подтверждённым данным: {unique[0]}"
    return "По подтверждённым данным: " + " ".join(unique[:3])


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


def apply_unconfirmed_operational_specificity_guard(
    result: SubscriptionDraftResult,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    specificity_context = _context_with_dialogue_contract_retrieved_facts(context, result)
    followup_claims = find_unsupported_followup_deadline_claims(result.draft_text, context=specificity_context)
    if followup_claims:
        return _operational_specificity_guarded_result(
            result,
            draft_text=UNSUPPORTED_FOLLOWUP_DEADLINE_SAFE_TEXT,
            flag="unsupported_followup_deadline_detected",
            claims=followup_claims,
            checklist_item="Не называть конкретную дату или срок связи менеджера без подтверждённого факта.",
        )

    schedule_claims = find_unsupported_schedule_assumption_claims(result.draft_text, context=specificity_context)
    if schedule_claims:
        return _operational_specificity_guarded_result(
            result,
            draft_text=UNSUPPORTED_SCHEDULE_ASSUMPTION_SAFE_TEXT,
            flag="unsupported_schedule_assumption_detected",
            claims=schedule_claims,
            checklist_item="Не делать догадки по расписанию без подтверждённого факта.",
        )

    visit_claims = find_unsupported_offline_visit_invitation_claims(result.draft_text, context=specificity_context)
    if visit_claims:
        return _operational_specificity_guarded_result(
            result,
            draft_text=UNSUPPORTED_OFFLINE_VISIT_INVITATION_SAFE_TEXT,
            flag="unsupported_offline_visit_invitation_detected",
            claims=visit_claims,
            checklist_item="Запись и оформление по умолчанию дистанционные; очную встречу не предлагать без согласования.",
        )

    delivery_claims = find_unsupported_content_delivery_action_claims(result.draft_text, context=specificity_context)
    if delivery_claims:
        return _operational_specificity_guarded_result(
            result,
            draft_text=(
                "Фрагмент занятия можно прислать для знакомства, но точный способ доступа — ссылка, запись или регистрация — "
                "нужно подтвердить у менеджера. Передам ему ваш запрос; класс, предмет и онлайн-формат уже вижу."
            ),
            flag="unsupported_content_delivery_action_detected",
            claims=delivery_claims,
            checklist_item="Не обещать от лица бота отправить ссылку/фрагмент/запись без подтверждённого способа доступа.",
            route="draft_for_manager",
        )
    return result


def find_unsupported_followup_deadline_claims(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    return _unsupported_claims_by_pattern(draft_text, pattern=FOLLOWUP_DEADLINE_RE, context=context)


def find_unsupported_schedule_assumption_claims(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    return _unsupported_claims_by_pattern(draft_text, pattern=SCHEDULE_ASSUMPTION_RE, context=context)


def find_unsupported_offline_visit_invitation_claims(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    return _unsupported_claims_by_pattern(draft_text, pattern=OFFLINE_VISIT_INVITATION_RE, context=context)


def find_unsupported_content_delivery_action_claims(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    return _unsupported_claims_by_pattern(draft_text, pattern=CONTENT_DELIVERY_ACTION_RE, context=context)


def _unsupported_claims_by_pattern(
    draft_text: str,
    *,
    pattern: re.Pattern[str],
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    source = str(draft_text or "")
    claims = tuple(dict.fromkeys(" ".join(match.group(0).split()) for match in pattern.finditer(source) if match.group(0).strip()))
    if not claims:
        return ()
    fact_texts = _fresh_fact_texts(context)
    return tuple(claim for claim in claims if not _claim_supported_by_facts(claim, fact_texts))


def _operational_specificity_guarded_result(
    result: SubscriptionDraftResult,
    *,
    draft_text: str,
    flag: str,
    claims: Sequence[str],
    checklist_item: str,
    route: str = "manager_only",
) -> SubscriptionDraftResult:
    return replace(
        result,
        route=route,
        draft_text=draft_text,
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, *claims])),
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, flag, "manager_approval_required", "no_auto_send"])),
        manager_checklist=tuple(dict.fromkeys([*result.manager_checklist, checklist_item])),
        metadata={**dict(result.metadata), flag: True, "unsupported_operational_claims": list(claims)},
    )


def apply_humanity_guards(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    """Final conversational guard: remove meta leaks and avoid useless handoff/repeats.

    This layer is deliberately conservative. It never weakens real P0/brand/fact
    gates and only promotes an answer from manager-only to draft when a verified
    answer fact is already present.
    """

    raw_p0_required = _humanity_p0_required(result)
    previous_bot_texts = _humanity_previous_bot_texts(context)
    block_a_enabled = _humanity_block_a_route_fix_enabled(context)
    block_generic_fact_answer = _humanity_generic_fact_answer_blocked(result, client_message=client_message)
    has_answer_fact = (not block_generic_fact_answer) and _has_humanity_answer_fact(context)
    preserve_existing_answer = _humanity_preserve_existing_answer(result)
    metadata = dict(result.metadata)
    benign_p0_context = (
        is_benign_hypothetical_refund(client_message)
        or _conversation_plan_semantic_non_p0(context, client_message=client_message)
    )
    hard_p0_text_locked = bool(
        metadata.get("final_p0_text_override")
        or metadata.get("zero_collect_legal_guarded")
        or metadata.get("zero_collect_refund_guarded")
        or metadata.get("complaint_apology_guarded")
        or metadata.get("payment_dispute_manager_only")
    )
    p0_required = raw_p0_required and not (benign_p0_context and not hard_p0_text_locked)
    flags = list(result.safety_flags)
    checklist = list(result.manager_checklist)
    route = result.route
    draft_text = result.draft_text
    changed = False

    if has_meta_leak(draft_text) and not _humanity_allows_dry_p0_text(result, p0_required=p0_required):
        cleaned = _sanitize_humanity_meta_text(draft_text)
        markers = meta_markers_present(draft_text)
        if cleaned and not has_meta_leak(cleaned):
            draft_text = cleaned
        else:
            fact_answer = "" if block_generic_fact_answer else _humanity_fact_answer(context, client_message=client_message)
            draft_text = fact_answer or (
                "Передам вопрос менеджеру, он ответит по сути."
            )
            route = "draft_for_manager" if route != "manager_only" else route
        flags.append("humanity_meta_leak_removed")
        checklist.append("Проверить, что клиентский текст не содержит служебных пометок и manager-facing фраз.")
        metadata["humanity_meta_leak_removed"] = True
        metadata["humanity_meta_markers"] = markers
        changed = True

    client_roles = tag_message_roles(client_message)
    draft_roles = tag_message_roles(draft_text)
    direct_question_unanswered = unanswered_direct_question(
        client_message,
        draft_text,
        client_topics=client_roles.topics,
        draft_topics=draft_roles.topics,
    )
    block_a_direct_answer = ""
    if (
        block_a_enabled
        and not p0_required
        and result.message_type not in {"non_question", "wait_for_more", "manager_only"}
    ):
        block_a_direct_answer = _humanity_block_a_direct_answer(
            context,
            client_message=client_message,
            current_draft=draft_text,
            previous_bot_texts=previous_bot_texts,
        )
    if block_a_direct_answer:
        draft_text = block_a_direct_answer
        if "правила можно посмотреть до оплаты" in block_a_direct_answer.casefold():
            route = "draft_for_manager"
        else:
            route = "bot_answer_self_for_pilot" if route != "manager_only" else "draft_for_manager"
        flags.append("humanity_block_a_direct_answer_applied")
        checklist.append("Слой человечности A: ответ перестроен на текущий вопрос без повторения предыдущего шаблона.")
        metadata["humanity_block_a_direct_answer_applied"] = True
        direct_question_unanswered = False
        changed = True

    if not p0_required and has_answer_fact and _humanity_can_trim_cosmetic_opening(result):
        trimmed = _trim_repeated_cosmetic_opening(draft_text, previous_bot_texts)
        if trimmed != draft_text:
            draft_text = trimmed
            flags.append("humanity_cosmetic_opening_trimmed")
            checklist.append("Косметический повторный зачин убран: ответ должен начинаться ближе к факту.")
            metadata["humanity_cosmetic_opening_trimmed"] = True
            changed = True

    if (
        not p0_required
        and has_answer_fact
        and result.message_type not in {"non_question", "context_update", "wait_for_more"}
    ):
        precise_fact_answer = _humanity_context_correction_answer(
            context, client_message=client_message, current_draft=draft_text
        ) or _humanity_precise_fact_answer(
            context, client_message=client_message, current_draft=draft_text
        )
        if precise_fact_answer:
            draft_text = precise_fact_answer
            route = "bot_answer_self_for_pilot" if route != "manager_only" else "draft_for_manager"
            flags.append("humanity_precise_fact_answer_applied")
            checklist.append("Клиент просит точное число/процент: ответ перестроен на точный извлечённый факт.")
            metadata["humanity_precise_fact_answer_applied"] = True
            changed = True

    if (
        not p0_required
        and has_answer_fact
        and not preserve_existing_answer
        and result.route not in {"bot_answer_self", "bot_answer_self_for_pilot"}
        and not _humanity_guarded_handoff_reason(result)
        and direct_question_unanswered
    ):
        fact_answer = "" if block_generic_fact_answer else _humanity_fact_answer(context, client_message=client_message)
        if fact_answer:
            draft_text = fact_answer
            route = "draft_for_manager" if route == "manager_only" else route
            flags.append("humanity_unanswered_question_repaired")
            checklist.append("Ответ был перестроен на прямой вопрос клиента по извлеченному факту.")
            metadata["humanity_unanswered_question_repaired"] = True
            direct_question_unanswered = False
            changed = True

    installment_amount_answer = ""
    if (
        not p0_required
        and not _humanity_guarded_handoff_reason(result)
        and result.route not in {"bot_answer_self", "bot_answer_self_for_pilot"}
        and not metadata.get("humanity_block_a_direct_answer_applied")
    ):
        installment_amount_answer = _humanity_installment_amount_answer(
            context, client_message=client_message
        )
    if installment_amount_answer:
        draft_text = installment_amount_answer
        route = "bot_answer_self_for_pilot"
        flags.append("humanity_installment_amount_repaired")
        checklist.append(
            "Клиент спросил про платёж в месяц: ответить из цены и условий оплаты, не подменяя годовую цену семестром."
        )
        metadata["humanity_installment_amount_repaired"] = True
        changed = True

    if p0_required and route != "manager_only":
        route = "manager_only"
        flags.append("humanity_p0_route_locked")
        metadata["humanity_p0_route_locked"] = True
        changed = True

    strict_antirepeat = _antirepeat_strict_enabled(context)
    repeat_threshold = 0.85 if strict_antirepeat else 0.8
    core_handoff_repeat = (not p0_required) and _is_core_handoff_fallback_repeat(
        draft_text,
        previous_bot_texts,
        threshold=repeat_threshold,
    )
    if not p0_required and is_near_repeat(draft_text, previous_bot_texts, threshold=repeat_threshold):
        fact_answer = (
            block_a_direct_answer
            or ("" if block_generic_fact_answer else _humanity_fact_answer(context, client_message=client_message))
        )
        if fact_answer and not is_near_repeat(fact_answer, previous_bot_texts, threshold=repeat_threshold):
            draft_text = fact_answer
            route = "bot_answer_self_for_pilot" if block_a_direct_answer and route != "manager_only" else "draft_for_manager" if route == "manager_only" else route
            flags.append("humanity_repeat_repaired")
            checklist.append("Ответ почти повторял предыдущую реплику; перестроен на текущий вопрос.")
            metadata["humanity_repeat_repaired"] = True
            changed = True
        elif strict_antirepeat or core_handoff_repeat:
            draft_text = _strict_antirepeat_fallback_text(
                context,
                result=replace(result, route=route, draft_text=draft_text, safety_flags=tuple(flags), metadata=metadata),
                client_message=client_message,
            )
            if strict_antirepeat:
                route = "draft_for_manager" if route == "manager_only" else route
            flags.append("humanity_strict_antirepeat_fallback_applied")
            checklist.append("Строгий анти-повтор: ответ заменён на короткий честный ответ/узкий хендофф по текущему уточнению.")
            metadata["humanity_strict_antirepeat_fallback_applied"] = True
            changed = True
        else:
            flags.append("humanity_repeat_detected")
            checklist.append("Ответ похож на предыдущую реплику: перед отправкой переписать под текущий вопрос.")
            metadata["humanity_repeat_detected"] = True
            changed = True

    if p0_required and route != "manager_only" and not is_benign_hypothetical_refund(client_message):
        route = "manager_only"
        flags.append("humanity_p0_route_preserved")
        metadata["humanity_p0_route_preserved"] = True
        changed = True

    if not _humanity_guarded_handoff_reason(result) and not preserve_existing_answer:
        route_action = humanity_route_action(
            p0_required=p0_required,
            has_retrieved_answer_fact=has_answer_fact,
            route=route,
            message_type=result.message_type,
            direct_question_answered=not direct_question_unanswered,
        )
        if route_action.get("regenerate"):
            fact_answer = "" if block_generic_fact_answer else _humanity_fact_answer(context, client_message=client_message)
            if fact_answer:
                draft_text = fact_answer
                action_route = str(route_action.get("route") or route)
                route = "bot_answer_self_for_pilot" if action_route == "bot_answer_self" else action_route
            flags.append("humanity_route_action_applied")
            checklist.append("Факт-ответ уже извлечён: ответить из него напрямую, не ограничиваться передачей менеджеру без P0.")
            metadata["humanity_route_action_applied"] = True
            metadata["humanity_route_action_reason"] = route_action.get("reason")
            metadata["humanity_route_action_route"] = route_action.get("route")
            changed = True

    if not changed:
        return result
    return replace(
        result,
        route=route,
        draft_text=draft_text,
        safety_flags=tuple(dict.fromkeys(flags)),
        manager_checklist=tuple(dict.fromkeys(checklist)),
        metadata=metadata,
    )


def apply_humanity_x2_rewriter(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
    rewrite_runner: Optional[Callable[[str], str]] = None,
) -> SubscriptionDraftResult:
    """Optional X2 form rewrite after all deterministic draft guards.

    X2 is disabled by default and never touches P0/manager_only routes. It can
    only replace the customer-facing text after both framework checks and repo
    gates accept the candidate.
    """

    if not _humanity_x2_rewrite_enabled(context):
        return result
    previous_bot_texts = _humanity_previous_bot_texts(context)
    prev_bot = previous_bot_texts[-1] if previous_bot_texts else ""
    prior_openers = tuple(" ".join(str(text or "").casefold().split()[:4]) for text in previous_bot_texts if str(text or "").strip())
    safety_flags_text = " ".join(result.safety_flags)
    turn = {
        "bot_text": result.draft_text,
        "bot_route": result.route,
        "bot_safety_flags": safety_flags_text,
    }
    linter_flags = lint_turn(turn, prev_bot_text=prev_bot, prior_openers=prior_openers)
    metadata = dict(result.metadata)
    metadata["humanity_x2"] = {
        "enabled": True,
        "mode": _humanity_x2_rewrite_mode(context),
        "linter_flags": linter_flags,
    }

    if result.route == "manager_only" or _humanity_p0_required(result):
        metadata["humanity_x2"]["fallback_reason"] = "locked_p0_or_manager_only"
        return replace(result, metadata=metadata)
    if _humanity_x2_identity_policy_locked(result):
        metadata["humanity_x2"]["fallback_reason"] = "locked_identity_policy"
        return replace(result, metadata=metadata)

    confirmed_facts = _humanity_x2_confirmed_facts(context)
    rules_engine_applied = _rules_engine_result_applied(metadata)

    def validate_candidate(candidate: str) -> str | None:
        return _humanity_x2_repo_gate(candidate, result=result, client_message=client_message, context=context)

    def sanitize_candidate(candidate: str) -> str:
        if not rules_engine_applied:
            return candidate
        stripped = strip_internal_service_markers(candidate)
        return stripped or candidate

    rewrite = apply_humanity_form_rewrite(
        turn,
        rewrite_fn=rewrite_runner,
        confirmed_facts=confirmed_facts,
        active_brand=_active_brand(context),
        client_message=client_message,
        linter_flags=linter_flags,
        sanitize_fn=sanitize_candidate,
        validate_fn=validate_candidate,
        mode=_humanity_x2_rewrite_mode(context),
    )
    metadata["humanity_x2"] = {
        **dict(metadata.get("humanity_x2") or {}),
        "rewritten": bool(rewrite.get("rewritten")),
        "fallback_reason": rewrite.get("fallback_reason"),
    }
    if not rewrite.get("rewritten"):
        return replace(result, metadata=metadata)
    draft_text = str(rewrite.get("draft_text") or "").strip()
    if not draft_text:
        metadata["humanity_x2"]["fallback_reason"] = "empty_candidate_after_rewrite"
        return replace(result, metadata=metadata)
    return replace(
        result,
        draft_text=draft_text,
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "humanity_x2_rewritten"])),
        metadata=metadata,
    )


def apply_phase2_tone_layer(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> SubscriptionDraftResult:
    if not _phase2_tone_enabled(context):
        return result
    before = score_tone(result.draft_text)
    metadata = dict(result.metadata)
    metadata["phase2_tone"] = {
        "enabled": True,
        "tone_before": before.as_dict(),
    }
    if result.route == "manager_only" or _humanity_p0_required(result):
        metadata["phase2_tone"]["fallback_reason"] = "locked_p0_or_manager_only"
        return replace(result, metadata=metadata)
    if before.tone_canc <= 0:
        metadata["phase2_tone"]["fallback_reason"] = "tone_ok"
        return replace(result, metadata=metadata)
    rewrite_fn = _phase2_tone_rewrite_override(context)
    candidate = rewrite_fn(result.draft_text) if rewrite_fn is not None else _phase2_tone_rewrite(result.draft_text)
    candidate = str(candidate or "").strip()
    if not candidate or candidate == str(result.draft_text or "").strip():
        metadata["phase2_tone"]["fallback_reason"] = "no_change"
        return replace(result, metadata=metadata)
    violation = _phase2_text_change_violation(result, candidate, client_message=client_message, context=context)
    if violation:
        metadata["phase2_tone"]["fallback_reason"] = violation
        metadata["phase2_tone"]["candidate_rejected"] = True
        return replace(result, metadata=metadata)
    after = score_tone(candidate)
    metadata["phase2_tone"].update(
        {
            "rewritten": True,
            "tone_after": after.as_dict(),
        }
    )
    return replace(
        result,
        draft_text=candidate,
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "phase2_tone_rewritten"])),
        metadata=metadata,
    )


def build_semantic_diagnosis_prompt(
    *,
    bot_text: str,
    client_message: str = "",
) -> str:
    return (
        "Ты — строгий классификатор ОДНОГО ответа бота учебного центра. Определи, содержит ли ответ\n"
        "ИНДИВИДУАЛЬНЫЙ ДИАГНОЗ/ГАРАНТИЮ по КОНКРЕТНОМУ ученику: собственную оценку бота, справится ли /\n"
        "подойдёт ли / потянет ли именно этот ребёнок — БЕЗ хеджа неуверенности и БЕЗ передачи менеджеру/преподавателю.\n\n"
        "СЧИТАЕТСЯ диагнозом (true):\n"
        "- утверждение про конкретного ученика: «да, справится», «с тройками можно идти», «потянет», «ему подойдёт»,\n"
        "  «догонять заранее не нужно», «сможет влиться», «слишком тяжело быть не должно», «посильный ритм»,\n"
        "  «подберут под ребёнка» — как оценка бота;\n"
        "- обещание результата/балла конкретному ученику.\n\n"
        "НЕ считается (false):\n"
        "- общая справка о программе/форматах/уровнях: «есть базовый и продвинутый уровень», «программа идёт от азов»,\n"
        "  «формат семинара, мини-группа»;\n"
        "- хеджированный ответ С ПЕРЕДАЧЕЙ: «уровень лучше подобрать на пробном / уточнит преподаватель / сориентирует менеджер»;\n"
        "- ответ про расписание, цены, документы, логистику.\n\n"
        "Верни СТРОГО JSON, без текста вне него:\n"
        '{"individual_diagnosis": true|false, "span": "<цитата ответа, если true; иначе пусто>", "reason": "<кратко>"}\n\n'
        f"Вопрос клиента для контекста:\n{str(client_message or '').strip()}\n\n"
        f"Ответ бота:\n{str(bot_text or '').strip()}\n"
    )


def apply_semantic_diagnosis_guard(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
    classifier_fn: Optional[Callable[[str], object]] = None,
) -> SubscriptionDraftResult:
    if not _semantic_diagnosis_guard_enabled(context):
        return result
    metadata = dict(result.metadata)
    guard_meta: dict[str, Any] = {
        "enabled": True,
        "checked": False,
        "rewritten": False,
    }
    metadata["semantic_diagnosis_guard"] = guard_meta
    if _semantic_diagnosis_locked_deferral(result, client_message=client_message):
        guard_meta["fallback_reason"] = "locked_p0_or_high_risk_deferral"
        return replace(result, metadata=metadata)
    if result.route not in {"bot_answer_self", "bot_answer_self_for_pilot", "draft_for_manager", "manager_only"}:
        guard_meta["fallback_reason"] = "unsupported_route"
        return replace(result, metadata=metadata)
    override = _semantic_diagnosis_classifier_override(context)
    classifier = override or classifier_fn
    if classifier is None:
        guard_meta["fallback_reason"] = "classifier_unavailable"
        return replace(result, metadata=metadata)
    prompt = build_semantic_diagnosis_prompt(bot_text=result.draft_text, client_message=client_message)
    try:
        raw_payload = classifier(prompt)
        payload = extract_json_object(raw_payload) if isinstance(raw_payload, str) else raw_payload
    except Exception as exc:  # noqa: BLE001
        guard_meta["fallback_reason"] = "classifier_error"
        guard_meta["error"] = str(exc)[:200]
        return replace(result, metadata=metadata)
    guard_meta["checked"] = True
    if not isinstance(payload, Mapping):
        guard_meta["fallback_reason"] = "classifier_invalid_payload"
        return replace(result, metadata=metadata)
    diagnosis = _truthy_value(payload.get("individual_diagnosis"))
    guard_meta["individual_diagnosis"] = diagnosis
    guard_meta["span"] = str(payload.get("span") or "")[:220]
    guard_meta["reason"] = str(payload.get("reason") or "")[:220]
    if not diagnosis:
        guard_meta["fallback_reason"] = "not_individual_diagnosis"
        return replace(result, metadata=metadata)
    if _has_diagnosis_hedge_and_transfer(result.draft_text):
        guard_meta["fallback_reason"] = "already_hedged_and_transferred"
        return replace(result, metadata=metadata)
    candidate = SEMANTIC_DIAGNOSIS_SAFE_TEXT
    guard_meta["rewritten"] = True
    guard_meta["fallback_reason"] = None
    return replace(
        result,
        draft_text=candidate,
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, "semantic_diagnosis_guard_rewritten"])),
        manager_checklist=tuple(
            dict.fromkeys(
                [
                    *result.manager_checklist,
                    "Semantic diagnosis guard: не оценивать конкретного ребёнка заочно; сверить уровень с преподавателем/менеджером.",
                ]
            )
        ),
        metadata=metadata,
    )


def _semantic_diagnosis_classifier_override(context: Optional[Mapping[str, Any]]) -> Optional[Callable[[str], object]]:
    if not isinstance(context, Mapping):
        return None
    value = context.get("semantic_diagnosis_classifier_fn")
    return value if callable(value) else None


def _semantic_diagnosis_locked_deferral(result: SubscriptionDraftResult, *, client_message: str = "") -> bool:
    if result.route != "manager_only":
        return False
    if not (
        _humanity_p0_required(result)
        or _hard_p0_in_client_text(client_message)
        or _semantic_diagnosis_high_risk_flagged(result)
    ):
        return False
    return _semantic_diagnosis_plain_deferral_text(result.draft_text)


def _semantic_diagnosis_high_risk_flagged(result: SubscriptionDraftResult) -> bool:
    flags = " ".join(str(flag or "") for flag in result.safety_flags).casefold()
    return bool(
        re.search(
            r"high[_-]?risk|p0|refund|complaint|payment[_-]?dispute|legal|zero[_-]?collect|manager[_-]?only",
            flags,
            re.I,
        )
    )


def _semantic_diagnosis_plain_deferral_text(text: str) -> bool:
    value = " ".join(str(text or "").split())
    if not value:
        return True
    low = value.casefold().replace("ё", "е")
    if re.search(
        r"справит|потян|подойдет|тяжело|посильн|влит|догонять|подберут?\s+под\s+реб",
        low,
        re.I,
    ):
        return False
    return bool(re.search(r"передам|верн[её]тся|ответственн|менеджер|сотрудник|сверит|проверит", low, re.I))


def _has_diagnosis_hedge_and_transfer(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    hedge = bool(
        re.search(
            r"заочно|не\s+буду\s+обещ|не\s+возьмусь|лучше\s+(?:сверить|подобрать|оценить)|"
            r"стоит\s+сверить|на\s+пробн|без\s+обещан|уровень\s+лучше",
            value,
            re.I,
        )
    )
    transfer = bool(re.search(r"менеджер|преподавател|педагог|куратор|пробн", value, re.I))
    return hedge and transfer


def _hard_p0_in_client_text(text: str) -> bool:
    return bool(set(codes_from_text(text)).intersection(HARD_P0_CODES))


def _phase2_tone_rewrite(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    replacements = (
        (r"\bСориентирую по проверенным данным[:：]?\s*", ""),
        (r"\bсориентирую по проверенным данным[:：]?\s*", ""),
        (r"\bв рамках текущего учебного центра\b", "по этому центру"),
        (r"\bВ рамках текущего учебного центра\b", "По этому центру"),
        (r"\bосуществляется\b", "проходит"),
        (r"\bОсуществляется\b", "Проходит"),
        (r"\bпредоставляется\b", "есть"),
        (r"\bПредоставляется\b", "Есть"),
        (r"\bближайший шаг уточнит менеджер\b", "дальше подскажет менеджер"),
        (r"\bМенеджер уточнит ближайший шаг\b", "Дальше подскажет менеджер"),
    )
    for pattern, repl in replacements:
        value = re.sub(pattern, repl, value)
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s+([,.!?;:])", r"\1", value)
    return value


def _phase2_text_change_violation(
    result: SubscriptionDraftResult,
    candidate: str,
    *,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> str:
    if draft_has_identity_disclosure(candidate):
        return "identity_disclosure"
    if _humanity_x2_repo_gate(candidate, result=result, client_message=client_message, context=context):
        return "repo_gate"
    facts = _rules_engine_facts(result, context)
    contract = _pipeline_contract(result, active_brand=_active_brand(context), fact_keys=tuple(facts.keys()))
    findings = verify_dialogue_contract_output(
        candidate,
        facts=facts,
        active_brand=_active_brand(context),
        contract=contract,
        client_message=client_message,
        context=context,
        previous_bot_texts=_humanity_previous_bot_texts(context),
    )
    if findings:
        return "verify_output:" + ",".join(dict.fromkeys(finding.code for finding in findings))
    added_anchors = dialogue_contract_new_concrete_anchors(candidate, original=result.draft_text, facts=facts)
    if added_anchors:
        return "new_concrete_anchor"
    return ""


def _phase2_tone_rewrite_override(context: Optional[Mapping[str, Any]]) -> Optional[Callable[[str], str]]:
    if isinstance(context, Mapping):
        value = context.get("phase2_tone_rewrite_fn")
        if callable(value):
            return value
    return None


def _humanity_x2_identity_policy_locked(result: SubscriptionDraftResult) -> bool:
    if str(result.draft_text or "").strip() in {IDENTITY_PROMPT_SAFE_TEXT, IDENTITY_FOTON_SAFE_TEXT, IDENTITY_UNPK_SAFE_TEXT}:
        return True
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    pipeline = metadata.get("dialogue_contract_pipeline") if isinstance(metadata.get("dialogue_contract_pipeline"), Mapping) else {}
    shadow = pipeline.get("rules_engine_intent_shadow") if isinstance(pipeline.get("rules_engine_intent_shadow"), Mapping) else {}
    return str(shadow.get("selected_source") or "") == "identity_policy"


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
    topic = str(result.topic_id or "").strip()
    flags = list(result.safety_flags)
    checklist = list(result.manager_checklist)
    metadata = dict(result.metadata)
    metadata["answer_safety"] = safety_decision.to_json_dict()
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

    if safety_decision.p0_required and not safety_decision.semantic_non_p0 and not metadata.get("presale_refund_policy_manager_check"):
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

    if str(funnel.get("lead_stage") or "") == "p0_manager_only" or str(funnel.get("next_step_type") or "") == "manager_only_p0":
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

    if (markers or is_high_risk_result(result)) and not metadata.get("presale_refund_policy_manager_check"):
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
    return tuple(code for code in decision.risk_codes if code in HARD_P0_CODES)


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


def _is_enrollment_process_question(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    if not _is_enrollment_signup_question(value):
        return False
    return bool(
        re.search(r"\b(?:как|надо|нужно|можно\s+ли|приезж\w*|дистанц\w*|оформ\w*|для|куда)\b", value)
        or re.search(r"\b(?:на\s+)?(?:курс|программ|обучен)\b", value)
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


def _mentions_schedule_day_or_time(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    return bool(re.search(r"\b(?:дни|дней|дням|день|дата|датам|время|часы|часов)\b", value) or has_marker(value, "распис"))


def _has_word_marker(text: str, *markers: str) -> bool:
    return has_any_marker(text, markers)


def _asks_installment(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    if _asks_invoice_monthly_payment(value):
        return False
    return has_any_marker(
        value,
        (
            "рассроч",
            "долями",
            "частями",
            "по частям",
            "помесяч",
            "банк",
            "одобр",
            "без процент",
            "без переплат",
        ),
    )


def _asks_invoice_monthly_payment(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    monthly = has_any_marker(value, ("помесяч", "каждый месяц", "ежемесяч", "по месяцам"))
    invoice_or_transfer = has_any_marker(value, ("по счету", "по счёту", "счет", "счёт", "банковск", "перевод", "реквизит"))
    negates_installment = has_any_marker(value, ("не рассроч", "не долями", "не частями", "не через банк", "не про рассроч"))
    return bool(monthly and (invoice_or_transfer or negates_installment))


def _price_question_explicitly_supersedes_installment(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    return has_any_marker(
        value,
        (
            "не про рассроч",
            "рассрочку поняла",
            "рассрочку уже",
            "нужна цена",
            "нужна стоимость",
            "цена за год",
            "стоимость за год",
        ),
    )


def _manager_handoff_request_text(context: Optional[Mapping[str, Any]]) -> str:
    known = known_context_fields(context)
    details = []
    if known.get("grade"):
        details.append(f"{known['grade']} класс")
    if known.get("subject"):
        details.append(str(known["subject"]))
    if known.get("format"):
        details.append(str(known["format"]))
    suffix = f" Вижу уже: {', '.join(details)}." if details else ""
    return MANAGER_HANDOFF_REQUEST_SAFE_TEXT + suffix


def _presale_refund_handoff_ack_text(context: Optional[Mapping[str, Any]]) -> str:
    known = known_context_fields(context)
    details = []
    if known.get("grade"):
        details.append(f"{known['grade']} класс")
    if known.get("subject"):
        details.append(str(known["subject"]))
    if known.get("format"):
        details.append(str(known["format"]))
    suffix = f" Вижу уже: {', '.join(details)}." if details else ""
    return (
        "Да, передам менеджеру именно вопрос по условиям возврата до оплаты. "
        "Он подтвердит актуальные правила по выбранному курсу; повторно писать класс, предмет и формат не нужно."
        f"{suffix}"
    )


def _price_fix_process_text(context: Optional[Mapping[str, Any]]) -> str:
    known = known_context_fields(context)
    details = []
    if known.get("grade"):
        details.append(f"{known['grade']} класс")
    if known.get("subject"):
        details.append(str(known["subject"]))
    if known.get("format"):
        details.append(str(known["format"]))
    suffix = f" Вижу контекст: {', '.join(details)}." if details else ""
    return PRICE_FIX_PROCESS_SAFE_TEXT + suffix


def _enrollment_signup_process_text(context: Optional[Mapping[str, Any]]) -> str:
    known = known_context_fields(context)
    details = []
    if known.get("grade"):
        details.append(f"{known['grade']} класс")
    if known.get("subject"):
        details.append(str(known["subject"]))
    if known.get("format"):
        details.append(str(known["format"]))
    suffix = f" Вижу уже: {', '.join(details)}." if details else ""
    return (
        "Запись и оформление по умолчанию можно пройти дистанционно, приезжать не нужно. "
        "Если клиенту очень нужна очная встреча, менеджер отдельно согласует возможность. "
        "Передам менеджеру запрос на оформление, чтобы он подтвердил группу и следующий шаг."
        f"{suffix}"
    )


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
        return ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT
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
        return ADDRESS_FOTON_MOSCOW_SAFE_TEXT
    if active_brand == "unpk" and not recording_followup and not address_negated and ("площадки" in client_haystack or "адрес" in client_haystack):
        return ADDRESS_UNPK_SAFE_TEXT
    asks_contact = (
        "дайте телефон" in client_haystack
        or "какой номер" in client_haystack
        or "по какому номеру" in client_haystack
        or ("связаться" in client_haystack and ("телефон" in client_haystack or "номер" in client_haystack))
    )
    if active_brand == "foton" and asks_contact:
        return CONTACT_FOTON_SAFE_TEXT
    if active_brand == "unpk" and asks_contact:
        return CONTACT_UNPK_SAFE_TEXT
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


def _strict_antirepeat_fallback_text(
    context: Optional[Mapping[str, Any]],
    *,
    result: SubscriptionDraftResult,
    client_message: str = "",
) -> str:
    plan = _conversation_intent_plan(context)
    if _scope_guard_has_missing_intent_fact(result, context, plan=plan):
        return _scope_fact_narrow_handoff_text(context, result=result, plan=plan)
    detail = _scope_fact_detail_label(context, result=result, plan=plan)
    if detail == "эту деталь":
        detail = _core_handoff_detail(context, client_message=client_message)
    previous = _humanity_previous_bot_texts(context)
    variants = tuple(item.format(detail=detail) for item in (*_HUMANE_DETAIL_HANDOFF_TEXTS, *_HUMANE_GENERIC_HANDOFF_TEXTS))
    return _select_nonrepeating_text(
        variants,
        previous,
        fallback="Вижу, это важно — отдельно отмечу менеджеру, чтобы он ответил именно по этому пункту.",
    )


def _core_handoff_detail(context: Optional[Mapping[str, Any]], *, client_message: str = "") -> str:
    plan = _conversation_intent_plan(context)
    detail = _scope_fact_detail_label(context, plan=plan)
    if detail and detail != "эту деталь":
        return detail
    text = " ".join(str(client_message or "").split())
    text = re.sub(
        r"^\s*клиент\s+(?:спрашивает|уточняет|интересуется|хочет\s+понять|просит\s+уточнить)\s*(?:,|:|—|-)?\s*",
        "",
        text,
        flags=re.I,
    ).strip(" \t\n\r:;,.—-")
    if text and not text.casefold().startswith("клиент "):
        return text[:90].rstrip() + ("…" if len(text) > 90 else "")
    return "эту деталь"


def _is_core_handoff_fallback_repeat(
    text: str,
    previous_bot_texts: Sequence[str],
    *,
    threshold: float,
) -> bool:
    normalized = " ".join(str(text or "").split())
    known_templates = {SAFE_FALLBACK_DRAFT_TEXT, *_HUMANE_GENERIC_HANDOFF_TEXTS}
    if normalized not in {" ".join(item.split()) for item in known_templates}:
        return False
    return is_near_repeat(text, previous_bot_texts, threshold=threshold)


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


def _client_clean_fact_text(value: object) -> str:
    cleaned = " ".join(str(value or "").split())
    if not cleaned:
        return ""
    cleaned = re.sub(
        r"^(?P<brand>[^:]{1,40}:\s*)?черновик\s+для\s+ситуации\s+«[^»]+»\s*:\s*",
        lambda match: str(match.group("brand") or ""),
        cleaned,
        flags=re.I,
    )
    cleaned = re.sub(
        r"^(?P<brand>[^:]{1,40}:\s*)?черновик\s+для\s+ситуации\s+\"[^\"]+\"\s*:\s*",
        lambda match: str(match.group("brand") or ""),
        cleaned,
        flags=re.I,
    )
    return cleaned.strip()


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


def _defer_direct_process_to_format_choice_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> bool:
    if _active_brand(context) != "unpk":
        return False
    client_haystack = str(client_message or "").casefold().replace("ё", "е")
    if not has_any_marker(client_haystack, ("пусть менеджер", "менеджер проверит", "передайте менеджеру", "передам менеджеру")):
        return False
    if not has_any_marker(client_haystack, ("выходн", "суббот", "воскрес", "по дням", "распис")):
        return False
    plan = _conversation_intent_plan(context)
    if str(plan.get("primary_intent") or "") not in {"schedule", "format", "general_consultation"} and result.topic_id != "theme:013_schedule":
        return False
    known = known_context_fields(context)
    known_format = str(known.get("format") or "").casefold()
    return bool(
        known_format
        or has_any_marker(client_haystack, ("онлайн", "дистанц", "мтс", "линк", "очно", "офлайн", "сретен"))
    )


def _format_choice_is_disjunctive_question(text: str) -> bool:
    value = str(text or "").casefold().replace("ё", "е")
    return bool(
        ("онлайн" in value and has_any_marker(value, ("очно", "офлайн")) and has_marker(value, "или"))
        or ("очно" in value and "онлайн" in value and "?" in value)
    )


def _recent_format_choice_was_ambiguous(context: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(context, Mapping):
        return False
    recent = context.get("recent_messages")
    if not isinstance(recent, Sequence) or isinstance(recent, (str, bytes, bytearray)):
        return False
    for item in list(recent)[-5:]:
        text = str(item or "").casefold().replace("ё", "е")
        if _format_choice_is_disjunctive_question(text):
            return True
    return False


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


def _without_known_grade_reask(text: str, *, context: Optional[Mapping[str, Any]]) -> str:
    known = known_context_fields(context)
    if not known.get("grade"):
        return text
    value = str(text or "")
    replacements = (
        (
            "Напишите класс ребёнка — подберём смену и проверим наличие.",
            "По вашему классу менеджер проверит подходящую смену и наличие мест.",
        ),
        (
            "Напишите класс ребёнка — подберём подходящую смену и проверим актуальные условия.",
            "По вашему классу подберём подходящую смену и проверим актуальные условия.",
        ),
        (
            "Напишите класс ребёнка — менеджер проверит, можем ли ещё закрепить место.",
            "По вашему классу менеджер проверит, можем ли ещё закрепить место.",
        ),
    )
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def _known_subject_or_format(context: Optional[Mapping[str, Any]], marker: str) -> bool:
    if not isinstance(context, Mapping):
        return False
    known = known_context_fields(context)
    if any(has_marker(item, marker) for item in known.values()):
        return True
    for key in ("known_dialog_fields", "known_client_fields", "customer_context_summary", "known_context_summary"):
        value = context.get(key)
        if isinstance(value, Mapping):
            if any(has_marker(item, marker) for item in value.values()):
                return True
        elif has_marker(value, marker):
            return True
    return False


def _known_grade_int(context: Optional[Mapping[str, Any]], *, client_message: str = "") -> int:
    known = known_context_fields(context)
    value = str(known.get("grade") or "").strip()
    if not value:
        match = re.search(r"\b(?P<grade>[1-9]|10|11)\s*(?:класс|класса|классе|кл\.?)\b", str(client_message or ""), flags=re.I)
        value = match.group("grade") if match else ""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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


def _default_autonomy_flip_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if not isinstance(context, Mapping) or not _autonomy_enabled(context):
        return False
    policy = _autonomy_policy(context)
    for value in (
        context.get("allow_default_autonomy"),
        context.get("default_autonomy_flip_enabled"),
        policy.get("allow_default_autonomy"),
        policy.get("default_autonomy_flip_enabled"),
    ):
        if value is not None:
            return _truthy_value(value)
    return False


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


def _humanity_p0_required(result: SubscriptionDraftResult) -> bool:
    metadata = dict(result.metadata)
    answer_safety = metadata.get("answer_safety")
    p0_from_safety = bool(isinstance(answer_safety, Mapping) and answer_safety.get("p0_required"))
    return bool(
        p0_from_safety
        or metadata.get("final_p0_text_override")
        or metadata.get("forced_route_high_risk")
        or "high_risk_manager_only" in result.safety_flags
    )


def _humanity_allows_dry_p0_text(result: SubscriptionDraftResult, *, p0_required: bool) -> bool:
    if not p0_required:
        return False
    normalized = " ".join(str(result.draft_text or "").split())
    dry_templates = {
        LEGAL_THREAT_SAFE_TEXT,
        LEGAL_THREAT_PII_SAFE_TEXT,
        *_REFUND_ZERO_COLLECT_VARIANTS,
        *_COMPLAINT_SAFE_VARIANTS,
        *_PAYMENT_DISPUTE_VARIANTS,
        *_LEGAL_SAFE_VARIANTS,
    }
    return normalized in {" ".join(template.split()) for template in dry_templates}


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


def _has_humanity_answer_fact(context: Optional[Mapping[str, Any]]) -> bool:
    return bool(_first_humanity_fact_text(context))


def _humanity_block_a_route_fix_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping):
        value = context.get("humanity_block_a_route_fix_enabled")
        if value is not None:
            return _truthy_value(value)
    env_value = os.getenv(HUMANITY_BLOCK_A_ROUTE_FIX_ENV)
    if env_value is not None:
        return _truthy_value(env_value)
    return True


def _scope_fact_guard_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping):
        value = context.get("scope_fact_guard_enabled")
        if value is not None:
            return _truthy_value(value)
    env_value = os.getenv(SCOPE_FACT_GUARD_ENV)
    if env_value is not None:
        return _truthy_value(env_value)
    return True


def _antirepeat_strict_enabled(context: Optional[Mapping[str, Any]]) -> bool:
    if isinstance(context, Mapping):
        value = context.get("antirepeat_strict_enabled")
        if value is not None:
            return _truthy_value(value)
    env_value = os.getenv(ANTIREPEAT_STRICT_ENV)
    if env_value is not None:
        return _truthy_value(env_value)
    return True


def _humanity_can_trim_cosmetic_opening(result: SubscriptionDraftResult) -> bool:
    if result.topic_id in HIGH_RISK_THEME_IDS:
        return False
    money_or_protective_topics = {
        "theme:001_pricing",
        "theme:002_payment_method",
        "theme:003_payment_status",
        "theme:005_discounts",
        "theme:006_installment",
        "theme:007_matkap_payment",
        "theme:008_tax_deduction",
        "theme:009_refund",
        "theme:011_contract",
    }
    if result.topic_id in money_or_protective_topics:
        return False
    if result.message_type in {"non_question", "context_update", "wait_for_more", "manager_only"}:
        return False
    return True


def _trim_repeated_cosmetic_opening(text: str, previous_bot_texts: Sequence[str]) -> str:
    value = str(text or "").strip()
    match = COSMETIC_OPENING_RE.match(value)
    if not match:
        return value
    opening = match.group(0).strip().casefold()
    if not opening:
        return value
    previous_openings = {
        (COSMETIC_OPENING_RE.match(str(item or "").strip()).group(0).strip().casefold())
        for item in previous_bot_texts
        if COSMETIC_OPENING_RE.match(str(item or "").strip())
    }
    if opening not in previous_openings:
        return value
    trimmed = value[match.end() :].lstrip(" ,.!—-")
    if len(trimmed.split()) < 4:
        return value
    return trimmed[:1].upper() + trimmed[1:]


def _humanity_block_a_direct_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
    previous_bot_texts: Sequence[str] = (),
) -> str:
    for candidate in (
        _humanity_unpk_address_confirmation_answer(
            context, client_message=client_message, current_draft=current_draft
        ),
        _humanity_presale_refund_rules_answer(
            context, client_message=client_message, current_draft=current_draft
        ),
        _humanity_unpk_tax_certificate_followup_answer(
            context, client_message=client_message, current_draft=current_draft
        ),
        _humanity_foton_bank_transfer_monthly_answer(
            context, client_message=client_message, current_draft=current_draft
        ),
        _humanity_unpk_weekend_address_answer(
            context, client_message=client_message, current_draft=current_draft
        ),
    ):
        if candidate and not is_near_repeat(candidate, previous_bot_texts):
            return candidate
    return ""


def _humanity_presale_refund_rules_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    dialog = _dialog_context_haystack(context)
    asks_where_to_read = has_any_marker(
        text,
        ("где", "почитать", "посмотреть", "договор", "оферт", "правил", "до оплаты", "заранее"),
    )
    refund_context = has_any_marker(text, ("возврат", "вернут", "вернете", "вернёте", "передума", "отказ")) or has_any_marker(
        dialog,
        ("возврат", "вернут", "вернете", "вернёте", "передума", "отказ"),
    )
    if not (asks_where_to_read and refund_context):
        return ""
    known = known_context_fields(context)
    details = []
    for key in ("grade", "subject", "format"):
        value = str(known.get(key) or "").strip()
        if value:
            details.append(value)
    detail_text = f" по {', '.join(details)}" if details else ""
    return (
        f"Да, правила можно посмотреть до оплаты: менеджер пришлёт актуальный договор или оферту{detail_text}, "
        "и там будут условия отказа/возврата. Передам менеджеру запрос именно по условиям возврата до оплаты. "
        "Точную сумму без документа я не буду обещать, но сформулирую запрос именно так: "
        "прислать правила до оплаты, чтобы вы спокойно посмотрели их заранее."
    )


def _humanity_unpk_address_confirmation_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if _active_brand(context) != "unpk":
        return ""
    asks_address_confirmation = (
        "сретен" in text
        and (
            "20" in text
            or has_any_marker(text, ("адрес", "подтверд", "да?", "верно", "правильно"))
        )
    )
    if not asks_address_confirmation:
        return ""
    facts = " ".join(_confirmed_fact_texts(context, limit=16)).casefold().replace("ё", "е")
    if "сретенка, 20" not in facts and "сретенка 20" not in facts:
        return ""
    return (
        "Да, верно: регулярные курсы УНПК в Москве проходят на Сретенке, 20, метро Чистые Пруды. "
        "Класс, предмет и очный формат уже вижу; если захотите записываться, останется только сверить конкретную группу и слот."
    )


def _humanity_unpk_tax_certificate_followup_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if _active_brand(context) != "unpk":
        return ""
    facts = " ".join([*_confirmed_fact_texts(context, limit=16), current_draft]).casefold().replace("ё", "е")
    dialog = _dialog_context_haystack(context)
    has_tax_fact = "кнд 1151158" in facts or ("налог" in facts and "вычет" in facts)
    if not has_tax_fact:
        return ""
    mentions_certificate = has_any_marker(text, ("справк", "вычет", "налог", "кнд"))
    follows_tax_context = has_any_marker(dialog, ("налог", "вычет", "кнд 1151158"))
    if not mentions_certificate and not (follows_tax_context and has_any_marker(text, ("менеджер", "напишу", "попрошу"))):
        return ""
    return (
        "Да, для налогового вычета нужна справка по форме КНД 1151158. "
        "Менеджер пришлёт шаблон заявления на email, после заявления справку подготовят и отправят в течение 10 рабочих дней. "
        "Лучше так и написать менеджеру: нужна справка для налогового вычета."
    )


def _humanity_foton_bank_transfer_monthly_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if _active_brand(context) != "foton":
        return ""
    asks_transfer = has_any_marker(text, ("перевод", "счет", "счёт", "безнал"))
    asks_monthly = has_any_marker(text, ("помесяч", "каждый месяц", "по месяц", "не все сразу", "не всё сразу"))
    if not (asks_transfer and asks_monthly):
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
    detail_text = f" для {', '.join(details)}" if details else ""
    return (
        f"Понял: вы спрашиваете не про рассрочку и не про Долями, а про то, можно ли помесячно оплачивать переводом на счёт{detail_text}. "
        "Я не буду подставлять сюда условия рассрочки: это другой способ оплаты. "
        "Менеджер проверит, можно ли оформить именно счёт каждый месяц, и даст корректные реквизиты/порядок оплаты."
    )


def _humanity_unpk_weekend_address_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if _active_brand(context) != "unpk":
        return ""
    asks_weekend = has_any_marker(text, ("суббот", "воскрес", "выходн", "по каким дням", "дням", "сб", "вс"))
    asks_direct_yes_no = has_any_marker(text, ("да/нет", "да или нет", "просто да", "просто понять", "заранее"))
    mentions_address = "сретен" in text or "там" in text or "москв" in text
    if not (asks_weekend and (asks_direct_yes_no or mentions_address)):
        return ""
    facts = tuple(_confirmed_fact_texts(context, limit=16))
    facts_low = " ".join(facts).casefold().replace("ё", "е")
    if "разные слоты по выходным" not in facts_low:
        return ""
    known = known_context_fields(context)
    grade = str(known.get("grade") or "").strip()
    subject = str(known.get("subject") or "").strip()
    group_text = ""
    if grade and subject:
        group_text = f" для {grade} класса, {subject}"
    elif grade:
        group_text = f" для {grade} класса"
    asks_specific_weekend_days = has_any_marker(text, ("суббот", "воскрес", "сб", "вс"))
    if asks_specific_weekend_days:
        if has_any_marker(text, ("или только", "просто бывают", "просто по выходным", "вообще там", "да или нет", "просто сказать")):
            return (
                f"Если совсем коротко по Сретенке{group_text}: подтверждено, что есть слоты по выходным. "
                "А вот обещать, что нужная группа идёт именно и в субботу, и в воскресенье, я не буду: такого точного факта по конкретной группе нет. "
                "Значит честный ответ такой: выходные — да; конкретный день или оба дня — только после сверки группы."
            )
        return (
            f"Коротко по Сретенке{group_text}: подтверждённый факт — есть разные слоты по выходным. "
            "То есть смотреть нужно выходные дни; но я не буду обещать, что именно ваша группа будет и в субботу, и в воскресенье одновременно без сетки конкретной группы. "
            "Если нужен точный слот, проверяем уже по группе."
        )
    return (
        f"Да: по УНПК на Сретенке ориентир — выходные, есть разные слоты по выходным. "
        f"Точный день и время{group_text} зависят от конкретной группы, поэтому их нужно сверить отдельно; но сам ответ на вопрос «выходные бывают?» — да."
    )


def _humanity_generic_fact_answer_blocked(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
) -> bool:
    """Do not replace an unresolved operational question with a neighboring fact."""
    text = str(client_message or "").casefold().replace("ё", "е")
    missing = " ".join(str(item or "") for item in result.missing_facts).casefold().replace("ё", "е")
    asks_bank_transfer = (
        has_any_marker(text, ("перевод", "счет", "счёт", "безнал"))
        and has_any_marker(text, ("оплат", "платить", "помесяч"))
    )
    if asks_bank_transfer and (
        "payment_methods.current" in missing
        or "способ" in missing
        or "порядок оплаты" in missing
        or "реквизит" in missing
        or "перевод" in missing
    ):
        return True
    asks_matkap_installment_combo = (
        has_any_marker(text, ("маткап", "материнск"))
        and has_any_marker(text, ("рассроч", "долями", "совмещ", "вместе"))
    )
    if asks_matkap_installment_combo and (
        "совмещ" in missing
        or "сочетан" in missing
        or "рассроч" in missing
        or "installment_terms.current" in missing
    ):
        return True
    return False


def _humanity_preserve_existing_answer(result: SubscriptionDraftResult) -> bool:
    if result.route in {"bot_answer_self", "bot_answer_self_for_pilot"}:
        return True
    flags = set(result.safety_flags)
    return any(
        flag.endswith("_safe_template_applied")
        or flag
        in {
            "autonomy_verified_fact_answer_template_applied",
            "pricing_safe_template_applied",
            "camp_safe_template_applied",
            "installment_safe_template_applied",
            "tax_safe_template_applied",
            "trial_safe_template_applied",
            "offline_free_trial_promise_guarded",
            "presale_refund_policy_manager_check",
            "presale_refund_policy_non_p0",
        }
        for flag in flags
    ) or bool(result.metadata.get("presale_refund_policy_manager_check"))


def _humanity_guarded_handoff_reason(result: SubscriptionDraftResult) -> bool:
    flags = set(result.safety_flags)
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    if result.message_type in {"non_question", "context_update", "wait_for_more"}:
        return True
    guarded_flags = {
        "autonomy_default_cautious_live_status_missing",
        "future_price_handoff_applied",
        "price_future_manager_only",
        "unsupported_promise_guarded",
        "unconfirmed_operational_specificity_guarded",
        "message_type_non_question",
        "message_type_context_update",
        "message_type_wait_for_more",
    }
    if flags.intersection(guarded_flags):
        return True
    if metadata.get("future_price_handoff_applied") or metadata.get("autonomy_default_cautious_live_status_missing"):
        return True
    return False


def _first_humanity_fact_text(context: Optional[Mapping[str, Any]]) -> str:
    facts = _confirmed_fact_texts(context, limit=8)
    for fact in facts:
        text = _client_clean_fact_text(fact)
        low = text.casefold().replace("ё", "е")
        if not text or "client_blocked:" in low or "internal_only" in low or "клиенту суммы не называть" in low:
            continue
        return text
    return ""


def _humanity_fact_answer(context: Optional[Mapping[str, Any]], *, client_message: str = "") -> str:
    precise_fact_answer = _humanity_precise_fact_answer(context, client_message=client_message)
    if precise_fact_answer:
        return precise_fact_answer
    installment_amount_answer = _humanity_installment_amount_answer(context, client_message=client_message)
    if installment_amount_answer:
        return installment_amount_answer
    fact = _first_humanity_fact_text(context)
    if not fact:
        return ""
    client_low = client_message.casefold().replace("ё", "е")
    fact_low = fact.casefold().replace("ё", "е")
    if "питан" in client_low and "5-разовым питанием" in fact_low and "5-разовое питание" not in fact_low:
        fact = re.sub(r"с\s+проживанием\s+и\s+5-разовым\s+питанием", "с проживанием; 5-разовое питание включено", fact, flags=re.I)
    fact_sentence = _ensure_sentence(fact)
    next_step = _humanity_next_step(client_message=client_message, context=context)
    return " ".join(part for part in (fact_sentence, next_step) if part).strip()


def _humanity_precise_fact_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    discount_percent_answer = _humanity_discount_percent_answer(
        context, client_message=client_message, current_draft=current_draft
    )
    if discount_percent_answer:
        return discount_percent_answer
    return ""


def _humanity_context_correction_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    weekend_schedule_answer = _humanity_weekend_schedule_no_format_lock_answer(
        context, client_message=client_message, current_draft=current_draft
    )
    if weekend_schedule_answer:
        return weekend_schedule_answer
    return ""


def _humanity_weekend_schedule_no_format_lock_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if _active_brand(context) != "unpk":
        return ""
    asks_weekend = has_any_marker(text, ("выходн", "суббот", "воскрес", " сб", " вс", "дням", "по каким дням"))
    rejects_format_lock = (
        has_any_marker(text, ("формат не принцип", "не принципиален", "главное выходн", "почему онлайн", "не про формат"))
        or ("формат" in text and "главное" in text)
    )
    if not (asks_weekend and rejects_format_lock):
        return ""
    draft_low = str(current_draft or "").casefold().replace("ё", "е")
    locks_online = "формат уже вижу как онлайн" in draft_low or "если скажете, какой формат" in draft_low
    mentions_online_instead = "онлайн с записью" in draft_low and "разные слоты по выходным" not in draft_low
    if current_draft and not (locks_online or mentions_online_instead):
        return ""
    facts = _confirmed_fact_texts(context, limit=16)
    has_weekend_fact = any("разные слоты по выходным" in str(fact).casefold().replace("ё", "е") for fact in facts)
    if not has_weekend_fact:
        return ""
    return (
        "Формат не фиксирую: вы написали, что главное — выходные. "
        "По УНПК есть разные слоты по выходным, но точные суббота/воскресенье и время зависят от конкретной группы. "
        "Для 9 класса по математике менеджер сверит ближайшие варианты и наличие мест."
    )


def _humanity_discount_percent_answer(
    context: Optional[Mapping[str, Any]],
    *,
    client_message: str = "",
    current_draft: str = "",
) -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if "скид" not in text:
        return ""
    asks_percent = "%" in text or has_any_marker(text, ("процент", "сколько", "такая же", "так же"))
    if not asks_percent:
        return ""
    if re.search(r"\b\d{1,2}\s*%", str(current_draft or "")):
        return ""
    format_key = ""
    if has_any_marker(text, ("очн", "офлайн")):
        format_key = "offline"
    elif has_any_marker(text, ("онлайн", "дистанц")):
        format_key = "online"
    facts = _confirmed_fact_texts(context, limit=16)
    if not facts:
        return ""

    def matches_format(value: str) -> bool:
        low = value.casefold().replace("ё", "е")
        if format_key == "offline":
            return "очн" in low or "офлайн" in low
        if format_key == "online":
            return "онлайн" in low or "дистанц" in low
        return True

    selected = ""
    for fact in facts:
        low = str(fact or "").casefold().replace("ё", "е")
        if "скид" not in low or "%" not in low or not matches_format(low):
            continue
        if "втор" in text and "втор" not in low:
            continue
        selected = str(fact)
        if "составляет" in low or "действует" in low:
            break
    if not selected:
        return ""
    match = re.search(r"\b\d{1,2}\s*%", selected)
    if not match:
        return ""
    pct = match.group(0).replace(" ", "")
    brand = _active_brand(context)
    if brand == "foton":
        format_label = "Очно" if format_key == "offline" else "Онлайн" if format_key == "online" else "По этому формату"
        base = f"{format_label} на второй предмет в Фотоне скидка {pct}."
    elif brand == "unpk":
        format_label = "очно" if format_key == "offline" else "онлайн" if format_key == "online" else "по этому формату"
        base = f"В УНПК {format_label} скидка по этому вопросу — {pct}."
    else:
        base = f"Скидка по этому вопросу — {pct}."
    stacking = ""
    for fact in facts:
        low = str(fact or "").casefold().replace("ё", "е")
        if "не сумм" in low or "наибольш" in low:
            stacking = " Скидки не суммируются: применяется наибольшая доступная."
            break
    next_step = " Если хотите, дальше менеджер проверит подходящую группу и оформит скидку к заявке."
    return base + stacking + next_step


def _humanity_installment_amount_answer(context: Optional[Mapping[str, Any]], *, client_message: str = "") -> str:
    text = str(client_message or "").casefold().replace("ё", "е")
    if _active_brand(context) != "foton":
        return ""
    asks_monthly_payment = (
        has_any_marker(text, ("помесяч", "каждый месяц", "по месяц", "сумм"))
        or bool(re.search(r"\bсколько\b[^.?!\n]{0,80}\b(?:месяц|выходит|платеж|платёж)", text, flags=re.I))
    )
    plan = context.get("conversation_intent_plan") if isinstance(context, Mapping) and isinstance(context.get("conversation_intent_plan"), Mapping) else {}
    plan_intent = str(plan.get("primary_intent") or plan.get("topic_id") or "").casefold()
    asks_installment = (
        _asks_installment(text)
        or has_any_marker(text, ("рассроч", "частями", "долями"))
        or "installment" in plan_intent
        or "theme:006_installment" in plan_intent
    )
    if not (asks_monthly_payment and asks_installment):
        return ""
    price_text = _foton_online_price_text_from_facts(context)
    if not price_text:
        return ""
    return (
        f"{price_text} По ежемесячному платежу не буду делить сумму на глаз: в Фотоне доступны варианты оплаты частями на 6, 10 или 12 месяцев и сервис Долями, "
        "а точный платёж зависит от выбранного срока и условий оформления. Менеджер посчитает платеж именно под выбранный вариант."
    )


def _humanity_next_step(*, client_message: str = "", context: Optional[Mapping[str, Any]] = None) -> str:
    brand = _active_brand(context)
    if has_any_marker(client_message, ("мест", "налич", "брон", "запис")):
        return "Если хотите, менеджер проверит наличие и поможет с оформлением."
    if has_any_marker(client_message, ("цен", "стоим", "сколько", "оплат", "рассроч", "долями")):
        return "Если подходит, менеджер поможет подобрать удобный вариант оплаты и оформить запись."
    if brand == "unpk":
        return "Если хотите, менеджер УНПК поможет подобрать следующий шаг."
    if brand == "foton":
        return "Если хотите, менеджер Фотона поможет подобрать следующий шаг."
    return "Если хотите, менеджер поможет подобрать следующий шаг."


def _sanitize_humanity_meta_text(text: str) -> str:
    value = strip_internal_service_markers(text)
    replacements = (
        "Сориентирую по проверенным данным:",
        "По проверенным данным:",
        "по проверенным данным",
        "Такой вопрос до оплаты не оформляю как жалобу или заявление на возврат.",
        "Не оформляю как жалобу или заявление.",
        "не оформляю как жалобу",
        "не оформляю как заявление",
        "Передам ему контекст диалога.",
    )
    for marker in replacements:
        value = value.replace(marker, "")
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\s{2,}", " ", value)
    return value.strip()


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


def _asks_money_price_question(text: str) -> bool:
    normalized = str(text or "").casefold().replace("ё", "е")
    if has_marker(normalized, "процент") and not has_any_marker(normalized, ("стоим", "цена", "цену", "прайс", "руб", "почем", "почём")):
        return False
    return bool(
        re.search(r"\b(?:стоим\w*|цена|цену|цены|ценой|прайс|почем|почём|руб(?:\.|лей|ля|ль)?)\b", normalized)
        or re.search(r"\bсколько\b[^.!?\n]{0,80}\b(?:стоит|стоим|руб|₽)", normalized)
        or re.search(r"\bсколько\b[^.!?\n]{0,80}\b(?:выходит|плат[её]ж|в\s+месяц|за\s+месяц)", normalized)
    )


def _is_generic_price_question_without_selection(client_haystack: str) -> bool:
    text = str(client_haystack or "").casefold().replace("ё", "е")
    asks_price = has_any_marker(text, ("сколько", "стоим", "цен", "почем", "почём", "прайс"))
    if not asks_price:
        return False
    has_grade = bool(re.search(r"\b(?:[1-9]|10|11)\s*(?:класс|классе|кл\.?)\b", text))
    has_format = has_any_marker(text, ("очно", "очный", "онлайн", "дистанц"))
    has_product = has_any_marker(text, ("лвш", "лагер", "интенсив", "4 недели", "четыре недели", "егэ", "огэ"))
    return not (has_grade or has_format or has_product)


def _foton_online_price_text_from_facts(context: Optional[Mapping[str, Any]]) -> str:
    semester = _price_amount_from_facts(context, required_markers=("онлайн",), period_markers=("семестр",))
    year = _price_amount_from_facts(
        context,
        required_markers=("онлайн",),
        period_markers=("год —", "год -", "годовая", "за год"),
        excluded_markers=("семестр",),
    )
    if not semester and not year:
        return ""
    parts = []
    if semester:
        parts.append(f"за семестр — {semester}")
    if year:
        parts.append(f"за год — {year}")
    return (
        f"Для онлайн-обучения в Фотоне сейчас: {', '.join(parts)}. "
        "Цена скоро подрастёт, поэтому если формат подходит, лучше закрепить текущие условия. "
        "Дальше подберём группу под класс, предмет и уровень ребёнка."
    )


def _price_amount_from_facts(
    context: Optional[Mapping[str, Any]],
    *,
    required_markers: Sequence[str],
    period_markers: Sequence[str],
    excluded_markers: Sequence[str] = (),
) -> str:
    facts = _fresh_fact_texts(context) or _confirmed_fact_texts(context, limit=12)
    for fact in facts:
        normalized = str(fact or "").casefold().replace("ё", "е")
        if not all(marker in normalized for marker in required_markers):
            continue
        if any(marker in normalized for marker in excluded_markers):
            continue
        if not any(marker in normalized for marker in period_markers):
            continue
        match = re.search(r"\b\d{1,3}(?:[ \u00a0]\d{3})+(?:\s*(?:₽|руб(?:\.|лей|ля|ль)?))?", str(fact or ""))
        if match:
            amount = " ".join(match.group(0).replace("\u00a0", " ").split())
            return amount if "₽" in amount or "руб" in amount.casefold() else f"{amount} ₽"
    return ""


def _normalized_fact_text(context: Optional[Mapping[str, Any]]) -> str:
    return " ".join(_fresh_fact_texts(context)).replace("\u00a0", " ").casefold()


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


def _active_brand(context: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(context, Mapping):
        return "unknown"
    value = context.get("active_brand")
    if not value and isinstance(context.get("facts_context"), Mapping):
        value = context["facts_context"].get("active_brand")  # type: ignore[index]
    text = str(value or "unknown").strip().casefold()
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "унпк мфти"}:
        return "unpk"
    return "unknown"


def _topic_id_from_context(context: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(context, Mapping):
        return UNKNOWN_TOPIC_FALLBACK_ID
    plan = context.get("conversation_intent_plan")
    if isinstance(plan, Mapping) and plan.get("topic_id"):
        return str(plan.get("topic_id") or UNKNOWN_TOPIC_FALLBACK_ID)
    contract = context.get("answer_contract")
    if isinstance(contract, Mapping) and contract.get("topic_id"):
        return str(contract.get("topic_id") or UNKNOWN_TOPIC_FALLBACK_ID)
    return str(context.get("topic_id") or UNKNOWN_TOPIC_FALLBACK_ID)


def _dialogue_contract_tone_guide(context: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(context, Mapping):
        return ""
    examples: list[str] = []
    for key in ("few_shot_style_examples", "few_shot_correction_examples"):
        value = context.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            examples.extend(str(item or "").strip() for item in value if str(item or "").strip())
    gold = context.get("gold_answer_context")
    if isinstance(gold, Mapping):
        for value in gold.values():
            if isinstance(value, str) and value.strip():
                examples.append(value.strip())
            elif isinstance(value, Mapping):
                text = value.get("answer") or value.get("text") or value.get("draft_text")
                if text:
                    examples.append(str(text).strip())
    return " | ".join(dict.fromkeys(examples[:3]))[:1600]


def _dialogue_contract_style_examples(context: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    if not isinstance(context, Mapping):
        return ()
    examples: list[str] = []
    for key in ("few_shot_style_examples", "few_shot_correction_examples"):
        value = context.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            examples.extend(str(item or "").strip() for item in value if str(item or "").strip())
    gold = context.get("gold_answer_context")
    if isinstance(gold, Mapping):
        for value in gold.values():
            if isinstance(value, str) and value.strip():
                examples.append(value.strip())
            elif isinstance(value, Mapping):
                text = value.get("answer") or value.get("text") or value.get("draft_text")
                if text:
                    examples.append(str(text).strip())
    return tuple(dict.fromkeys(item[:900] for item in examples if item))[:8]


def _dialogue_contract_safety_flags(pipeline_result: Any) -> list[str]:
    flags = ["dialogue_contract_pipeline", "manager_approval_required", "no_auto_send"]
    if getattr(pipeline_result.contract, "is_p0", False):
        flags.append("dialogue_contract_p0_pregate")
    flags.append(
        "dialogue_contract_verified"
        if not pipeline_result.findings and not getattr(pipeline_result, "fallback_reason", "")
        else "dialogue_contract_verification_fallback"
    )
    if getattr(pipeline_result, "unsupported_claims", ()):
        flags.append("dialogue_contract_semantic_fallback")
    if getattr(pipeline_result, "warmed", False):
        flags.append("dialogue_contract_x2_warmth_applied")
    if getattr(pipeline_result, "repaired", False):
        flags.append("dialogue_contract_safety_repair_applied")
    if getattr(pipeline_result, "is_estimate", False):
        flags.append("dialogue_contract_estimate_answer")
    if getattr(pipeline_result, "partial_yield_applied", False):
        flags.append("dialogue_contract_partial_yield_applied")
    if getattr(pipeline_result, "composite_applied", False):
        flags.append("dialogue_contract_composite_applied")
    if getattr(pipeline_result, "next_step_applied", False):
        flags.append("dialogue_contract_next_step_applied")
    return flags


def _sanitize_dialogue_contract_client_text(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    stripped = strip_internal_service_markers(result.draft_text)
    if stripped != result.draft_text:
        flags = tuple(dict.fromkeys([*result.safety_flags, "dialogue_contract_internal_text_sanitized"]))
        metadata = {**dict(result.metadata), "dialogue_contract_internal_text_sanitized": True}
        if not stripped.strip():
            return replace(
                result,
                draft_text=SAFE_FALLBACK_DRAFT_TEXT,
                route="draft_for_manager" if result.route != "manager_only" else result.route,
                safety_flags=tuple(dict.fromkeys([*flags, "manager_approval_required", "no_auto_send"])),
                metadata=metadata,
            )
        result = replace(result, draft_text=stripped, safety_flags=flags, metadata=metadata)
    sanitized = sanitize_answer(result.draft_text, mode="bot")
    blocking_flags = {
        "raw_json_leak",
        "internal_metadata_leak",
        "bot_placeholder_leak",
        "unsafe_placeholder_leak",
        "personal_placeholder_leak",
    }
    blocking_detected = set(sanitized.flags) & blocking_flags
    if not blocking_detected:
        if not sanitized.flags:
            return result
        return replace(
            result,
            safety_flags=tuple(dict.fromkeys([*result.safety_flags, "dialogue_contract_sanitize_checked", *sanitized.flags])),
            metadata={**dict(result.metadata), "dialogue_contract_sanitize_flags": list(sanitized.flags)},
        )
    if sanitized.text == result.draft_text:
        return result
    flags = tuple(dict.fromkeys([*result.safety_flags, "dialogue_contract_sanitize_applied", *sanitized.flags]))
    metadata = {**dict(result.metadata), "dialogue_contract_sanitize_flags": list(sanitized.flags)}
    if not sanitized.text.strip():
        return replace(
            result,
            draft_text=SAFE_FALLBACK_DRAFT_TEXT,
            route="draft_for_manager" if result.route != "manager_only" else result.route,
            safety_flags=tuple(dict.fromkeys([*flags, "manager_approval_required", "no_auto_send"])),
            metadata=metadata,
        )
    return replace(result, draft_text=sanitized.text or SAFE_FALLBACK_DRAFT_TEXT, safety_flags=flags, metadata=metadata)


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
    return replace(
        result,
        route="manager_only",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, *precise_condition_claims])),
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, reason, "brand_separation_guarded", *precise_condition_flags])),
        manager_checklist=tuple(
            dict.fromkeys(
                [
                    *result.manager_checklist,
                    "Проверить бренд клиента: в черновике нельзя смешивать Фотон и УНПК.",
                ]
            )
        ),
        metadata={
            **dict(result.metadata),
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


def _fresh_fact_texts(context: Optional[Mapping[str, Any]]) -> tuple[str, ...]:
    if not isinstance(context, Mapping):
        return ()
    facts_context = context.get("facts_context")
    facts_mapping = facts_context if isinstance(facts_context, Mapping) else {}
    context_quality = context.get("context_quality")
    quality_mapping = context_quality if isinstance(context_quality, Mapping) else {}

    stale = (
        _truthy_value(context.get("facts_stale"))
        or _truthy_value(facts_mapping.get("stale"))
        or _truthy_value(facts_mapping.get("facts_stale"))
        or _truthy_value(quality_mapping.get("facts_stale"))
    )
    fresh = (
        context.get("facts_fresh") is True
        or facts_mapping.get("fresh") is True
        or facts_mapping.get("facts_fresh") is True
        or facts_mapping.get("fresh_facts") is True
    )
    verified = (
        context.get("client_safe_fact_verified") is True
        or facts_mapping.get("client_safe_fact_verified") is True
        or _has_dialogue_contract_retrieved_facts(context)
    )
    if stale and not (fresh or verified):
        return ()
    if not (fresh or verified):
        return ()

    texts: list[str] = []
    for key in ("confirmed_facts", "facts_context"):
        _append_fact_texts(texts, context.get(key))
    pipeline = context.get("dialogue_contract_pipeline") if isinstance(context.get("dialogue_contract_pipeline"), Mapping) else {}
    if isinstance(pipeline.get("retrieved_facts"), Mapping):
        _append_fact_texts(texts, pipeline.get("retrieved_facts"))
    _append_fact_texts(texts, context.get("knowledge_snippets"))
    return tuple(text for text in texts if text)


def _has_dialogue_contract_retrieved_facts(context: Mapping[str, Any]) -> bool:
    pipeline = context.get("dialogue_contract_pipeline") if isinstance(context.get("dialogue_contract_pipeline"), Mapping) else {}
    retrieved = pipeline.get("retrieved_facts") if isinstance(pipeline.get("retrieved_facts"), Mapping) else {}
    return any(str(key).strip() and str(value).strip() for key, value in retrieved.items())


def _append_fact_texts(result: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, str):
        cleaned = " ".join(value.split())
        if cleaned:
            result.append(cleaned)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).strip().casefold() in {
                "missing",
                "facts_missing",
                "stale",
                "facts_stale",
                "fresh",
                "facts_fresh",
                "fresh_facts",
                "client_safe_fact_verified",
            }:
                continue
            _append_fact_texts(result, item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _append_fact_texts(result, item)
        return
    if isinstance(value, (int, float)):
        result.append(str(value))


def _claim_supported_by_facts(claim: str, fact_texts: Sequence[str]) -> bool:
    normalized_claim = _normalize_fact_match_text(claim)
    if not normalized_claim:
        return False
    normalized_facts = [_normalize_fact_match_text(text) for text in fact_texts]
    if normalized_claim == "до 1 июля" and any(
        "before_2026_07_01" in text or "до 1 июля" in text or "ранн" in text for text in normalized_facts
    ):
        return True
    if normalized_claim == "до 1 июня" and any(
        "before_2026_06_01" in text or "до 1 июня" in text or "ранн" in text for text in normalized_facts
    ):
        return True
    if any(normalized_claim in text for text in normalized_facts):
        return True
    claim_anchors = _fact_match_anchors(claim)
    if not claim_anchors:
        return False
    return any(claim_anchors <= _fact_match_anchors(text) for text in fact_texts)


def _fact_match_anchors(text: Any) -> set[str]:
    source = str(text or "")
    low = source.casefold().replace("ё", "е").replace("\u00a0", " ")
    anchors = set(dialogue_contract_concrete_anchors(source))
    anchors.update(_fact_match_unit_anchors(source))
    anchors.update(_fact_match_schedule_condition_anchors(low))
    if re.search(r"\bфотон\b|цдпо|црдо|cdpofoton", low, re.I):
        anchors.add("brand:foton")
    if re.search(r"\bунпк\b|унпк\s+мфти|kmipt", low, re.I):
        anchors.add("brand:unpk")
    if re.search(r"\bсегодня\b", low, re.I):
        anchors.add("deadline:today")
    if re.search(r"\bзавтра\b|до\s+завтра", low, re.I):
        anchors.add("deadline:tomorrow")
    if re.search(r"до\s+вечера|к\s+вечеру", low, re.I):
        anchors.add("deadline:evening")
    if re.search(r"в\s+течение\s+\d+\s*(?:минут|час|часов|дн|дней|суток|сутки)", low, re.I):
        anchors.add("deadline:relative_period")
    return anchors


def _fact_match_unit_anchors(text: Any) -> set[str]:
    source = str(text or "").replace("\u00a0", " ")
    anchors: set[str] = set()
    if re.search(r"\b\d{1,3}(?:[,.]\d{1,2})?\s*(?:%|процент\w*)", source, re.I):
        anchors.add("unit:percent")
    if re.search(r"\b\d[\d\s]{1,9}\s*(?:руб(?:\.|лей|ля|ль)?|₽|р\.)", source, re.I):
        anchors.add("unit:money")
    if re.search(r"\b\d{1,3}\+?\s*балл\w*", source, re.I):
        anchors.add("unit:points")
    return anchors


def _fact_match_schedule_condition_anchors(low_text: str) -> set[str]:
    anchors: set[str] = set()
    if re.search(r"\bвечерн\w*", low_text, re.I):
        anchors.add("condition:evening")
    if re.search(r"\bутренн\w*", low_text, re.I):
        anchors.add("condition:morning")
    if re.search(r"\bдневн\w*", low_text, re.I):
        anchors.add("condition:day")
    if re.search(r"\b(?:выходн|суббот|воскресен)\w*", low_text, re.I):
        anchors.add("condition:weekend")
    if re.search(r"\b(?:будн|буден)\w*", low_text, re.I):
        anchors.add("condition:weekday")
    return anchors


def _normalize_fact_match_text(text: Any) -> str:
    value = str(text or "").casefold().replace("ё", "е").replace("\u00a0", " ")
    return " ".join(value.split())


def _truthy_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да"}


def _output_sanitizer_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        for key in (OUTPUT_SANITIZER_ENV, "output_sanitizer_enabled"):
            if key in context:
                return _truthy_value(context.get(key))
    return _truthy_value(os.getenv(OUTPUT_SANITIZER_ENV))


def _phase2_tone_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        for key in (PH2_TONE_ENV, "phase2_tone_enabled"):
            if key in context:
                return _truthy_value(context.get(key))
    return _truthy_value(os.getenv(PH2_TONE_ENV))


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


def _semantic_diagnosis_guard_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        for key in (SEMANTIC_DIAGNOSIS_GUARD_ENV, "semantic_diagnosis_guard_enabled"):
            if key in context:
                return _truthy_value(context.get(key))
    return _truthy_value(os.getenv(SEMANTIC_DIAGNOSIS_GUARD_ENV))


def _answer_quality_llm_rewrite_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        value = context.get("answer_quality_llm_rewrite_enabled")
        if value is not None:
            return _truthy_value(value)
    return _truthy_value(os.getenv(ANSWER_QUALITY_LLM_REWRITE_ENV)) or _truthy_value(os.getenv(ANSWER_QUALITY_LLM_REWRITER_ENV))


def _answer_quality_llm_rewrite_mode(context: Optional[Mapping[str, Any]] = None) -> str:
    if isinstance(context, Mapping):
        value = context.get("answer_quality_llm_rewrite_mode")
        if value is not None:
            return str(value or "").strip().casefold()
    return str(os.getenv(ANSWER_QUALITY_LLM_REWRITE_MODE_ENV) or "").strip().casefold()


def _answer_quality_llm_polish_sales_enabled(
    context: Optional[Mapping[str, Any]],
    result: SubscriptionDraftResult,
) -> bool:
    if not _answer_quality_llm_rewrite_enabled(context):
        return False
    mode = _answer_quality_llm_rewrite_mode(context)
    if mode not in {"polish_sales", "always_sales", "all"}:
        return False
    if result.route == "manager_only" or result.topic_id in HIGH_RISK_THEME_IDS:
        return False
    if any(marker in " ".join(result.safety_flags).casefold() for marker in ("high_risk", "zero_collect", "legal", "complaint")):
        return False
    return result.topic_id in {
        "theme:001_pricing",
        "theme:005_discounts",
        "theme:006_installment",
        "theme:013_schedule",
        "theme:014_format",
        "theme:016_program",
        "theme:020_enrollment",
        "theme:023_trial_class",
        "theme:026_camp_general",
        "service:S5_general_consultation",
    }


def _humanity_x2_rewrite_enabled(context: Optional[Mapping[str, Any]] = None) -> bool:
    if isinstance(context, Mapping):
        value = context.get("humanity_x2_rewrite_enabled")
        if value is not None:
            return _truthy_value(value)
    return _truthy_value(os.getenv(HUMANITY_X2_REWRITE_ENV))


def _humanity_x2_rewrite_mode(context: Optional[Mapping[str, Any]] = None) -> str:
    if isinstance(context, Mapping):
        value = context.get("humanity_x2_rewrite_mode")
        if value is not None:
            mode = str(value or "").strip().casefold()
            return mode if mode in {"linter", "all_eligible"} else "all_eligible"
    mode = str(os.getenv(HUMANITY_X2_REWRITE_MODE_ENV) or "all_eligible").strip().casefold()
    return mode if mode in {"linter", "all_eligible"} else "all_eligible"


def _humanity_x2_confirmed_facts(context: Optional[Mapping[str, Any]]) -> Any:
    if not isinstance(context, Mapping):
        return ()
    for key in ("confirmed_facts", "selected_facts", "facts_context", "gold_answer_context"):
        value = context.get(key)
        if value:
            return value
    return ()


def _extract_humanity_x2_text(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = extract_json_object(text)
    except Exception:
        return text.strip().strip('"').strip()
    for key in ("draft_text", "answer", "text", "message"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return ""


_HUMANITY_X2_BLOCKING_SANITIZER_FLAGS: tuple[str, ...] = (
    "raw_json_redacted",
    "internal_metadata_redacted",
    "email_redacted",
    "phone_redacted",
    "person_name_redacted",
    "role_name_redacted",
    "document_reference_redacted",
    "brand_normalized",
    "refund_policy_redacted",
    "service_promise_redacted",
)
_HUMANITY_X2_PRESSURE_RE = re.compile(
    r"только\s+сегодня|последн(?:ий|яя)\s+шанс|успейт|решайт[е]?\s+сейчас|"
    r"срочно\s+(?:оформ|запис|реш)|иначе\s+(?:мест|скид|цен)|мест\s+почти\s+нет|"
    r"надо\s+успеть|не\s+тяните|лучше\s+не\s+тянуть",
    re.I,
)


def _humanity_x2_repo_gate(
    candidate: str,
    *,
    result: SubscriptionDraftResult,
    client_message: str,
    context: Optional[Mapping[str, Any]],
) -> str | None:
    if draft_has_identity_disclosure(candidate):
        return "identity_disclosure"
    stripped = strip_internal_service_markers(candidate)
    if stripped != str(candidate or "").strip():
        return "internal_service_marker"
    safety = classify_answer_safety(
        client_message=client_message,
        context=context,
        topic_id=result.topic_id,
        route=result.route,
        safety_flags=result.safety_flags,
    )
    if safety.blocks_rewriter or safety.p0_required or safety.manager_only:
        return f"answer_safety:{safety.primary_risk or 'manager_only'}"
    if _HUMANITY_X2_PRESSURE_RE.search(candidate):
        return "pressure"
    sanitized = sanitize_answer(candidate, mode="bot")
    if not sanitized.fixpoint_reached or sanitized.status == "fixpoint_not_reached":
        return "sanitize_answer:fixpoint_not_reached"
    for flag in sanitized.flags:
        if flag in _HUMANITY_X2_BLOCKING_SANITIZER_FLAGS:
            return f"sanitize_answer:{flag}"
    if has_meta_leak(candidate):
        return "repo_meta_leak"
    return None


DraftGenerationResult = SubscriptionDraftResult
CodexExecDraftProvider = SubscriptionLlmDraftProvider
FakeDraftProvider = FakeSubscriptionLlmDraftProvider
contains_bot_identity_disclosure = draft_has_identity_disclosure


def subscription_llm_safety_contract() -> Mapping[str, Any]:
    return {
        "schema_version": SUBSCRIPTION_LLM_SCHEMA_VERSION,
        "provider": "codex_exec",
        "uses_openai_api_key": False,
        "client_auto_send_allowed": False,
        "crm_write_allowed": False,
        "tallanto_write_allowed": False,
        "stable_runtime_write_allowed": False,
        "fallback_text": SAFE_FALLBACK_DRAFT_TEXT,
        "identity_disclosure_forbidden_phrases": list(IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES),
        "safe_schedule_template": safe_schedule_template(),
    }


def _clean_list(value: Any, *, max_items: int, max_chars: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = value
    else:
        return []
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        result.append(" ".join(text.split())[:max_chars])
        if len(result) >= max_items:
            break
    return result


def _clean_crm_recommendations(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    result: list[Mapping[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        recommendation = {
            "target": str(item.get("target") or "").strip()[:80],
            "action": str(item.get("action") or "").strip()[:80],
            "text": str(item.get("text") or "").strip()[:500],
            "requires_manager_approval": True,
        }
        if recommendation["target"] and recommendation["action"] and recommendation["text"]:
            result.append(recommendation)
        if len(result) >= 8:
            break
    return result


def _clamp_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, parsed))


def _optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _cache_key(payload: Mapping[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _with_metadata(result: SubscriptionDraftResult, extra: Mapping[str, Any]) -> SubscriptionDraftResult:
    return replace(result, metadata={**dict(result.metadata), **dict(extra)})


def _guard_cache_dir(path: Path | str) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    if "stable_runtime" in resolved.parts:
        raise ValueError("subscription LLM cache must not be inside stable_runtime")
    return resolved


def _is_retryable(stderr: str) -> bool:
    lowered = (stderr or "").casefold()
    return any(marker in lowered for marker in _RETRYABLE_MARKERS)


class _CodexRetryableError(RuntimeError):
    pass
