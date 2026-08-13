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
    is_near_repeat,
    parse_contract as parse_dialogue_contract,
    verify_output as verify_dialogue_contract_output,
)
from mango_mvp.channels.p0_recall_spec import HARD_P0_CODES
from mango_mvp.channels.text_signals import has_any_marker, has_marker
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
    _direct_path_template_fact_text,
    _fresh_fact_texts,
    _has_dialogue_contract_retrieved_facts,
    _normalize_fact_match_text,
    _p0_model_led_enabled,
    _prose_model_led_enabled,
    _presale_prompt_child_name_value,
    _template_from_kb_enabled,
    _template_from_kb_trace_event,
    _truthy_value,
)

ANSWER_CONTRACT_GREEN_TEMPLATE_REDUCTION_ENV = "TELEGRAM_ANSWER_CONTRACT_GREEN_TEMPLATE_REDUCTION"

SCOPE_FACT_GUARD_ENV = "TELEGRAM_SCOPE_FACT_GUARD"

A_THREAD_ENV = "TELEGRAM_A_THREAD"

PH2_OBJECTION_ENV = "TELEGRAM_PH2_OBJECTION"

PH2_ANXIETY_ENV = "TELEGRAM_PH2_ANXIETY"



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
    "а также сервис Долями. Это относится к очным и онлайн-курсам Фотона. "
    "По обычным курсам также можно обсудить помесячную оплату или оплату за семестр. "
    "Конкретные условия и оформление зависят от выбранного способа оплаты; менеджер поможет подобрать удобный вариант."
)

FOTON_DOLYAMI_SAFE_TEXT = (
    "Да, Долями можно использовать в Фотоне. По точному числу частей и процентам не буду обещать без оформления: "
    "условия зависят от выбранного способа оплаты и платёжного сервиса. Подтверждённо: в Фотоне также доступны варианты "
    "оплаты частями на 6, 10 или 12 месяцев для очных и онлайн-курсов. "
    "Менеджер поможет выбрать и оформить подходящий вариант дистанционно."
)

PROMOCODE_SAFE_TEXT = "Промокодов сейчас нет. Из реальных выгод: при оплате за семестр или за год выходит выгоднее — это уже учтено в прайсе."

UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT = (
    "В УНПК рассрочки нет, это не банковская рассрочка, поэтому одобрение банка не требуется. "
    "Можно платить помесячно, за семестр или за год. "
    "При оплате за семестр действует скидка 10%, за год - 14%. "
    "Если нужно растянуть оплату, менеджер подскажет варианты под вашу ситуацию."
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

FOTON_ONLINE_TRIAL_SAFE_TEXT = (
    "В онлайн-формате Фотона можем прислать вам фрагмент занятия — посмотреть подачу и уровень; оформление проходит дистанционно — приезжать не нужно. "
    "Условия просмотра фрагмента подтвердит менеджер перед записью."
)

FOTON_OFFLINE_FREE_TRIAL_GUARD_TEXT = (
    "По очному формату бесплатное пробное по умолчанию не обещаю. "
    "Очный пробный шаг согласует менеджер при записи: он проверит подходящую группу, филиал и условия. "
    "Запрос передам именно как очный, без подмены на онлайн-фрагмент."
)

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























def _conversation_intent_plan(context: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    plan = context.get("conversation_intent_plan")
    return plan if isinstance(plan, Mapping) else {}

























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
    return (
        f"Вижу {details}. Не буду подставлять предмет или направление, которое вы не называли. "
        "Передам менеджеру ваш исходный вопрос без дополнительных предположений."
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
        include_text_signals=not _p0_model_led_enabled(context),
    )
    codes = tuple(code for code in decision.risk_codes if code in HARD_P0_CODES)
    if _p0_model_led_enabled(context) and "p0_latch" not in (decision.evidence or {}):
        return ()
    return codes








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
