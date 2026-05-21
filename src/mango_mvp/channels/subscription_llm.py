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

from mango_mvp.channels.draft_prompt_builder import (
    IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES,
    build_draft_prompt,
    safe_schedule_template,
    should_force_manager_only,
)
from mango_mvp.question_catalog.classifier import load_valid_theme_and_service_ids


SUBSCRIPTION_LLM_SCHEMA_VERSION = "subscription_llm_draft_v1_2026_05_16"
DEFAULT_CODEX_MODEL = "gpt-5.5"
DEFAULT_CODEX_REASONING_EFFORT = "medium"
SAFE_FALLBACK_DRAFT_TEXT = "Спасибо за сообщение. Передам вопрос менеджеру, он вернется с проверенным ответом."
UNKNOWN_TOPIC_FALLBACK_ID = "service:S2_unclear"
REFUND_ZERO_COLLECT_SAFE_TEXT = (
    "Спасибо, что обратились. Менеджер проверит ваш вопрос, мы рассмотрим ситуацию, и менеджер свяжется в ближайшее время. "
    "Пока ничего дополнительно присылать не нужно."
)
LEGAL_THREAT_SAFE_TEXT = (
    "Ваше обращение принято. Передам его ответственному сотруднику, он свяжется с вами."
)
LEGAL_THREAT_PII_SAFE_TEXT = (
    "Ваше обращение принято. Передам его ответственному сотруднику, он свяжется с вами."
)
COMPLAINT_SAFE_TEXT = "Спасибо, что обратились. Передам вопрос менеджеру: он свяжется с вами в ближайшее время."
RESULT_GUARANTEE_SAFE_TEXT = (
    "Мы не даём и не гарантируем конкретный балл: результат зависит от ученика, регулярности занятий "
    "и самостоятельной работы. Менеджер свяжется, уточнит цель и может показать, какая у нас статистика результатов."
)
ADMISSION_GUARANTEE_SAFE_TEXT = (
    "Мы не даём и не гарантируем поступление: результат зависит от ученика и выбранной траектории подготовки. "
    "Есть статистика: 97% наших учеников поступают в желаемые вузы. Менеджер свяжется и подробно поможет подобрать программу."
)
FORCED_DISCOUNT_SAFE_TEXT = "Передам вопрос менеджеру: он проверит доступные условия и свяжется с вами."
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
UNPK_MFTI_EMPLOYEE_DISCOUNT_TEXT = "Сотрудникам МФТИ действует скидка 10%; нужен подтверждающий документ с места работы. Менеджер проверит документы и условия."
MULTICHILD_DISCOUNT_TEXT = (
    "Да, для детей из многодетной семьи есть скидка 10%; нужно удостоверение многодетной семьи, "
    "даже если учится один ребёнок или два ребёнка. "
    "Скидка не суммируется с другими скидками: применяется наибольшая. Менеджер поможет проверить условия."
)
DISCOUNT_STACKING_SAFE_TEXT = "Скидки не суммируются: применяется наибольшая доступная скидка. Менеджер проверит условия под вашу ситуацию."
FOTON_INSTALLMENT_SAFE_TEXT = (
    "Да, в Фотоне есть рассрочка через Т-Банк. Долями — 4 части без процентов для коротких программ. "
    "Также есть варианты рассрочки на 6 или 12 месяцев; условия индивидуальные. "
    "Решение по заявке принимает банк. Менеджер пришлёт ссылку на оформление."
)
FOTON_DOLYAMI_SAFE_TEXT = (
    "Долями — это оплата на 4 части без процентов, обычно для коротких программ. "
    "Это подходит именно для короткие программы. "
    "Решение о доступности принимает Т-Банк и банк. Менеджер проверит, подходит ли этот вариант для вашего курса."
)
UNPK_OLYMPIAD_PHYSTECH_PRICE_TEXT = (
    "Олимпиадная подготовка Физтех сейчас указана для 9 и 11 классов, занятия проходят в будни. "
    "Менеджер сориентирует по группе и актуальным условиям."
)
UNPK_OLYMPIAD_PHYSTECH_HANDOFF_TEXT = (
    "По олимпиадной подготовке Физтех для этого класса менеджер проверит актуальную возможность и свяжется с вами."
)
UNPK_EGE_INTENSIVE_PRICE_SAFE_TEXT = (
    "Для 10-11 классов есть интенсив подготовки к ЕГЭ: онлайн, 8 недель, живые вебинары, "
    "ручная проверка второй части и пробники. Ориентир: 1 предмет — 18 800 ₽, 2 предмета — 34 400 ₽. "
    "Актуальность текущего набора уточнит менеджер. Подскажите, какой предмет интересует?"
)
FOTON_ONLINE_PRICE_SAFE_TEXT = "Онлайн-обучение в Фотоне: есть варианты оплаты за семестр и за год. Менеджер подскажет актуальную стоимость под ваш класс и курс."
UNPK_GRADES_5_11_PRICE_SAFE_TEXT = (
    "Для 5-11 классов в УНПК есть варианты оплаты за семестр и за год. "
    "Менеджер свяжется и подскажет актуальную стоимость под ваш класс, курс и формат."
)
UNPK_FOUR_WEEKS_NEW_PRICE_SAFE_TEXT = (
    "Курс на 4 недели — ориентир 10 900 ₽, для новых учеников 9 900 ₽. "
    "Точную стоимость под класс и предмет подтвердит менеджер. Подскажите, какой класс и направление интересуют?"
)
PROMOCODE_SAFE_TEXT = "Спасибо, передам менеджеру — он свяжется, уточнит и подскажет актуальные акции и промокоды."
UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT = (
    "Здравствуйте! У нас оплата возможна помесячно, за семестр или за год. "
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
    "По налоговому вычету можно вернуть до 14 300 ₽ в год на одного ребёнка: "
    "это 13% с расходов до 110 000 ₽. За 2023 год и ранее действовал лимит 50 000 ₽ — возврат до 6 500 ₽. "
    "Справку для вычета готовим до 10 рабочих дней; "
    "решение и сроки выплаты остаются на стороне ФНС."
)
TAX_LICENSE_SAFE_TEXT = "Да, есть лицензия на ведение образовательной деятельности. Менеджер поможет подготовить документы для налогового вычета."
UNPK_LVSH_SEATS_SAFE_TEXT = "Обычно группа 12-15 человек. Точные места по смене проверит менеджер и свяжется с вами."
FOTON_LVSH_PRICE_SAFE_TEXT = (
    "Летняя выездная школа в Менделеево: ориентир от 75 000 ₽, полный тариф 98 000 ₽. "
    "В стоимость входит проживание, 5-разовое питание и есть бесплатный трансфер из Москвы. "
    "Подскажите возраст ребёнка — подберём подходящую смену."
)
UNPK_LVSH_PRICE_SAFE_TEXT = (
    "ЛВШ Менделеево: полный тариф 120 000 ₽, со скидкой 25% — 89 900 ₽, минимальная цена — 83 800 ₽. "
    "В стоимость входит проживание и 5-разовое питание. Подскажите возраст ребёнка — подберём смену."
)
FOTON_CITY_CAMP_AUGUST_SAFE_TEXT = "ЛШ Москва Фотон проходит 3-14 августа, адрес: Верхняя Красносельская. Менеджер подскажет детали записи."
UNPK_JUNE_CAMP_HANDOFF_TEXT = "По июньской выездной смене менеджер свяжется и подскажет актуальные смены."
FOTON_LVSH_DATES_SAFE_TEXT = "ЛВШ Менделеево у Фотона: 20-28 июня и 18-26 июля. Менеджер подскажет наличие мест."
UNPK_LVSH_DATES_SAFE_TEXT = "ЛВШ Менделеево у УНПК: 18-26 июля и 15-25 августа. Менеджер подскажет наличие мест."
CONTRACT_ENTITY_SAFE_TEXT = "Менеджер пришлёт информацию, на каком оформлении будет договор, и проверит данные по вашей заявке."
CERTIFICATE_SAFE_TEXT = "Менеджер свяжется и подготовит справку: срок до 10 дней, постараемся раньше. Уточним данные для подготовки."
PII_DOCUMENT_SAFE_TEXT = "Менеджер свяжется и проверит вопрос по документам. Повторно присылать данные в чат не нужно."
TEACHERS_GENERAL_SAFE_TEXT = "У нас преподают специалисты из МФТИ, МГУ, ВШЭ, МИФИ, эксперты ЕГЭ и члены жюри олимпиад. Менеджер подскажет преподавателя по конкретной группе."
TEACHERS_SPECIFIC_SAFE_TEXT = "Имя преподавателя зависит от конкретной группы. У нас преподают специалисты из МФТИ, МГУ, ВШЭ и эксперты; менеджер уточнит преподавателя по группе."
TEACHERS_CHANGE_SAFE_TEXT = "Если педагог не подойдёт, менеджер поможет: при возможности переведём ребёнка в другую группу."
TEACHERS_MENDELEEVO_SAFE_TEXT = "В ЛВШ Менделеево преподают специалисты из МФТИ, МГУ, МИФИ, эксперты, к.т.н. и к.ф.-м.н. Менеджер подскажет состав по группе."
CROSS_BRAND_GENERIC_SAFE_TEXT = "Это отдельные организации, по вашему вопросу сориентирую в рамках текущего учебного центра. Менеджер свяжется и расскажет по нашей программе и наших условиях."
CROSS_BRAND_LICENSE_SAFE_TEXT = "У нас есть лицензия на образовательную деятельность. Менеджер свяжется и подскажет детали по документам."
CROSS_BRAND_PLATFORM_SAFE_TEXT = "В нашем учебном центре онлайн-занятия проходят в МТС Линк / Webinar, доступна запись. Менеджер подскажет детали."
IDENTITY_PROMPT_SAFE_TEXT = (
    "Я помощник менеджера: окажу помощь с вопросом и при необходимости передам коллеге; "
    "менеджер свяжется и уточнит детали."
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
    "По лагерям и выездным школам важно подобрать смену под возраст, предмет и формат. "
    "Напишите, пожалуйста, класс ребёнка и интересующее направление — менеджер проверит актуальные смены и наличие мест."
)
MISSING_GENERAL_HELPFUL_TEXT = (
    "Помогу сориентироваться по обучению. Напишите, пожалуйста, класс ребёнка, предмет и цель: подтянуть школьную программу, "
    "подготовиться к экзамену или олимпиаде. По этим данным менеджер предложит подходящий следующий шаг."
)
INTERNAL_SERVICE_MARKER_RE = re.compile(
    r"\[[^\]\n]{0,220}?(?:\bsource\s*=|\bfreshness\s*=|source:[A-Za-z0-9_:\-]+|fact:[A-Za-z0-9_:\-]+|kc_chunk:[A-Za-z0-9_:\-]+)[^\]\n]{0,260}\]\s*",
    re.I,
)
INTERNAL_SERVICE_TOKEN_RE = re.compile(
    r"\b(?:source|freshness)\s*=\s*[^\s;\],.]+|source:[A-Za-z0-9_:\-]+|fact:[A-Za-z0-9_:\-]+|kc_chunk:[A-Za-z0-9_:\-]+",
    re.I,
)
DRAFT_PLACEHOLDER_RE = re.compile(
    r"\[(?:[^\]\n]{0,80})?(?:вставить|указать|подставить|TODO|проверенн\w+\s+ссылк|актуальн\w+\s+ссылк)(?:[^\]\n]{0,120})?\]",
    re.I,
)
PROMOCODE_DRAFT_RE = re.compile(r"\b(?:LVSH-VEB20|LVSH-KF-10|ABRAMOV|VAGIN)\b", re.I)

ALLOWED_ROUTES = {"draft_for_manager", "manager_only", "blocked", "bot_answer_self", "bot_answer_self_for_pilot"}
AUTONOMOUS_ROUTES = {"bot_answer_self", "bot_answer_self_for_pilot"}
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
    r"\b(?:после\s+1\s+(?:июля|августа)|в\s+август\w*|августовск\w+|после\s+повышени\w*|с\s+сентябр\w*|будущ\w+\s+цен\w*|цена\s+выраст\w*)\b",
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
UNSUPPORTED_PROMISE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b\d{1,3}(?:[,.]\d{1,2})?\s*(?:%|процент\w*)", re.I),
    re.compile(r"\b\d[\d\s\u00a0]{1,9}\s*(?:руб(?:\.|лей|ля|ль)?|₽|р\.)", re.I),
    re.compile(r"\b\d+\s*(?:к|тыс\.?|тысяч)\b", re.I),
    re.compile(
        r"\b(?:до|по)\s+\d{1,2}(?:[./-]\d{1,2}(?:[./-]\d{2,4})?|\s+"
        r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря))",
        re.I,
    ),
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

    def build_draft(
        self,
        client_message: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SubscriptionDraftResult:
        prompt = build_draft_prompt(client_message, context=context)
        result = self.generate_from_prompt(prompt, force_manager_only=should_force_manager_only(context))
        result = apply_unsupported_promise_guard(result, context=context)
        result = apply_payment_confirmation_guard(result, client_message=client_message, context=context)
        result = apply_brand_separation_guard(result, client_message=client_message, context=context)
        result = apply_input_policy_guards(result, client_message=client_message, context=context)
        result = apply_high_risk_content_guards(result, client_message=client_message, context=context)
        return apply_autonomy_matrix_guard(result, client_message=client_message, context=context)

    def generate(self, prompt: str) -> SubscriptionDraftResult:
        return self.generate_from_prompt(prompt)

    def generate_from_prompt(self, prompt: str, *, force_manager_only: bool = False) -> SubscriptionDraftResult:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return safe_fallback_draft(reason="empty_prompt")

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
            return _with_metadata(cached, {"cache_hit": True})

        last_error = "codex_exec_failed"
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self._run_once(prompt_text, force_manager_only=force_manager_only)
            except subprocess.TimeoutExpired:
                return safe_fallback_draft(reason="timeout", metadata={"attempt": attempt, "timeout_sec": self.timeout_sec})
            except FileNotFoundError:
                return safe_fallback_draft(reason="codex_binary_not_found", metadata={"codex_bin": self.codex_bin})
            except _CodexRetryableError as exc:
                last_error = str(exc) or "retryable_codex_error"
                if attempt < self.max_attempts:
                    self.sleep(min(3.0, float(attempt)))
                    continue
                return safe_fallback_draft(reason="codex_retryable_error", metadata={"last_error": last_error})
            except Exception as exc:  # noqa: BLE001
                return safe_fallback_draft(reason="invalid_json_or_codex_error", metadata={"last_error": str(exc)[:400]})
            self._cache_put(cache_key, result)
            return result
        return safe_fallback_draft(reason=last_error)

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
        return guard_identity_disclosure(result)

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
        result = apply_unsupported_promise_guard(result, context=context)
        result = apply_payment_confirmation_guard(result, client_message=client_message, context=context)
        result = apply_brand_separation_guard(result, client_message=client_message, context=context)
        result = apply_input_policy_guards(result, client_message=client_message, context=context)
        result = apply_high_risk_content_guards(result, client_message=client_message, context=context)
        return apply_autonomy_matrix_guard(result, client_message=client_message, context=context)

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
    value = INTERNAL_SERVICE_MARKER_RE.sub("", value)
    value = INTERNAL_SERVICE_TOKEN_RE.sub("", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\s{2,}", " ", value)
    return value.strip()


def draft_has_internal_service_markers(text: str) -> bool:
    value = str(text or "")
    return bool(INTERNAL_SERVICE_MARKER_RE.search(value) or INTERNAL_SERVICE_TOKEN_RE.search(value))


def draft_has_identity_disclosure(text: str) -> bool:
    return bool(find_identity_disclosure_phrases(text))


def find_identity_disclosure_phrases(text: str) -> tuple[str, ...]:
    lowered = str(text or "").casefold()
    return tuple(phrase for phrase in IDENTITY_DISCLOSURE_FORBIDDEN_PHRASES if phrase.casefold() in lowered)


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
        return result
    claims = find_unsupported_numeric_promises(result.draft_text, context=context)
    if not claims:
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
    return replace(
        result,
        route="manager_only",
        forbidden_promises_detected=tuple(dict.fromkeys([*result.forbidden_promises_detected, *claims])),
        safety_flags=flags,
        manager_checklist=checklist,
        metadata={**dict(result.metadata), "unsupported_promises": list(claims)},
    )


def find_unsupported_numeric_promises(
    draft_text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    claims = _extract_numeric_promise_claims(draft_text)
    if not claims:
        return ()
    fact_texts = _fresh_fact_texts(context)
    return tuple(claim for claim in claims if not _claim_supported_by_facts(claim, fact_texts))


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
    markers = set(detect_high_risk_input_markers(client_message, context=context))
    topic = str(result.topic_id or "").strip()
    flags = list(result.safety_flags)
    checklist = list(result.manager_checklist)
    metadata = dict(result.metadata)
    route = result.route
    draft_text = result.draft_text

    if _is_unpk_installment_case(result, client_message=client_message, context=context):
        route = result.route if result.route in AUTONOMOUS_ROUTES else "draft_for_manager"
        draft_text = UNPK_INSTALLMENT_APPROVED_FALLBACK_TEXT
        flags.append("unpk_installment_approved_fallback_applied")
        checklist.append("Проверить, что вопрос относится к УНПК и к оплате частями/по периодам.")
        metadata["unpk_installment_approved_fallback_applied"] = True

    if _is_unpk_zvsh_case(result, client_message=client_message, context=context):
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

    if not cross_brand_guarded() and _is_future_price_case(result, client_message=client_message, context=context):
        route = "manager_only"
        draft_text = "Передам вопрос менеджеру: он проверит актуальную стоимость на нужный период и свяжется с вами."
        flags.extend(("future_price_handoff_applied", "manager_approval_required", "no_auto_send"))
        checklist.append("Будущая цена: не называть суммы после повышения, передать менеджеру.")
        metadata["future_price_handoff_applied"] = True

    terminal_template = "" if cross_brand_guarded() else _terminal_safe_template(result, client_message=client_message, context=context)
    if terminal_template:
        direct_info_template = terminal_template in {
            ADDRESS_UNPK_SAFE_TEXT,
            ADDRESS_UNPK_MOSCOW_REGULAR_SAFE_TEXT,
            CONTACT_FOTON_SAFE_TEXT,
            CONTACT_UNPK_SAFE_TEXT,
            IDENTITY_PROMPT_SAFE_TEXT,
            OFF_TOPIC_FOTON_SAFE_TEXT,
            OFF_TOPIC_UNPK_SAFE_TEXT,
            OFF_TOPIC_GENERIC_SAFE_TEXT,
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

    matkap_template = "" if cross_brand_guarded() or metadata.get("terminal_safe_template_applied") else _matkap_safe_template(result, client_message=client_message, context=context)
    if matkap_template:
        route = "manager_only" if route == "manager_only" else "draft_for_manager"
        draft_text = matkap_template
        flags.append("matkap_safe_template_applied")
        checklist.append("Маткапитал: не обещать одобрение СФР и не принимать региональный маткапитал.")
        metadata["matkap_safe_template_applied"] = True

    tax_template = "" if cross_brand_guarded() or metadata.get("terminal_safe_template_applied") else _tax_safe_template(result, client_message=client_message, context=context)
    if tax_template:
        route = "manager_only" if route == "manager_only" else "draft_for_manager"
        draft_text = tax_template
        flags.append("tax_safe_template_applied")
        checklist.append("Налоговый вычет: не гарантировать возврат от ФНС.")
        metadata["tax_safe_template_applied"] = True

    camp_template = "" if cross_brand_guarded() or metadata.get("terminal_safe_template_applied") else _camp_safe_template(result, client_message=client_message, context=context)
    if camp_template:
        route = "manager_only" if route == "manager_only" else "draft_for_manager"
        draft_text = camp_template
        if topic not in {"theme:026_camp_general", "theme:027_camp_living_conditions", "theme:028_transport_logistics"}:
            topic = "theme:026_camp_general"
            flags.append("camp_topic_normalized")
            metadata["camp_topic_normalized"] = True
        flags.append("camp_safe_template_applied")
        checklist.append("Лагерь: не смешивать смены Фотона и УНПК.")
        metadata["camp_safe_template_applied"] = True

    installment_template = "" if cross_brand_guarded() or metadata.get("terminal_safe_template_applied") or metadata.get("future_price_handoff_applied") else _installment_safe_template(result, client_message=client_message, context=context)
    if installment_template:
        route = "draft_for_manager"
        draft_text = installment_template
        flags.append("installment_safe_template_applied")
        checklist.append("Рассрочка: не сравнивать бренды и не обещать одобрение заявки.")
        metadata["installment_safe_template_applied"] = True

    pricing_template = "" if cross_brand_guarded() or metadata.get("terminal_safe_template_applied") or metadata.get("future_price_handoff_applied") else _pricing_safe_template(result, client_message=client_message, context=context)
    if pricing_template:
        route = "manager_only" if pricing_template == UNPK_OLYMPIAD_PHYSTECH_HANDOFF_TEXT else "draft_for_manager"
        draft_text = pricing_template
        if pricing_template == UNPK_EGE_INTENSIVE_PRICE_SAFE_TEXT and topic != "theme:016_program":
            topic = "theme:016_program"
            flags.append("program_topic_normalized")
            metadata["program_topic_normalized"] = True
        flags.append("pricing_safe_template_applied")
        checklist.append("Прайс: использовать только проверенный брендовый факт или передать менеджеру.")
        metadata["pricing_safe_template_applied"] = True

    discount_template = "" if cross_brand_guarded() or metadata.get("terminal_safe_template_applied") or metadata.get("future_price_handoff_applied") else _discount_safe_template(result, client_message=client_message, context=context)
    if discount_template:
        route = "manager_only" if discount_template == FORCED_DISCOUNT_SAFE_TEXT else "draft_for_manager"
        draft_text = discount_template
        flags.append("discount_safe_template_applied")
        checklist.append("Скидка: не обещать индивидуальные условия без проверки менеджером.")
        metadata["discount_safe_template_applied"] = True

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

    docs_template = (
        ""
        if cross_brand_guarded()
        or metadata.get("terminal_safe_template_applied")
        or metadata.get("tax_safe_template_applied")
        or metadata.get("matkap_safe_template_applied")
        else _docs_safe_template(result, client_message=client_message, context=context)
    )
    if docs_template:
        route = "manager_only" if route == "manager_only" else "draft_for_manager"
        draft_text = docs_template
        flags.append("docs_safe_template_applied")
        if _client_message_contains_pii(client_message):
            flags.append("placeholder_in_draft")
        checklist.append("Документы: не раскрывать юрлица и номера лицензий в черновике.")
        metadata["docs_safe_template_applied"] = True

    teacher_template = (
        ""
        if cross_brand_guarded()
        or metadata.get("terminal_safe_template_applied")
        or metadata.get("result_guarantee_safe_template_applied")
        or metadata.get("admission_guarantee_safe_template_applied")
        else _teacher_safe_template(result, client_message=client_message, context=context)
    )
    if teacher_template:
        route = "manager_only" if route == "manager_only" else "draft_for_manager"
        draft_text = teacher_template
        flags.append("teacher_safe_template_applied")
        checklist.append("Преподаватели: не называть ФИО без привязки к конкретной группе.")
        metadata["teacher_safe_template_applied"] = True

    if (
        not cross_brand_guarded()
        and not metadata.get("terminal_safe_template_applied")
        and not metadata.get("result_guarantee_safe_template_applied")
        and not metadata.get("admission_guarantee_safe_template_applied")
        and not _is_reputation_only_case(markers=markers)
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
        and _is_refund_case(result, markers=markers)
    ):
        route = "manager_only"
        draft_text = REFUND_ZERO_COLLECT_SAFE_TEXT
        flags.extend(("zero_collect_refund_guarded", "high_risk_manager_only"))
        checklist.append("Возврат: не собирать ФИО, договор, оплату, телефон, email, сумму или причину в черновике.")
        metadata["zero_collect_refund_guarded"] = True

    if not metadata.get("zero_collect_legal_guarded") and _is_complaint_case(result, markers=markers):
        if "reputation_threat" in markers and "legal" not in markers:
            topic = "theme:019b_negative_feedback"
        route = "manager_only"
        draft_text = COMPLAINT_SAFE_TEXT
        flags.extend(("complaint_apology_guarded", "high_risk_manager_only"))
        checklist.append("Жалоба: не извиняться от лица компании и не признавать вину в авточерновике.")
        metadata["complaint_apology_guarded"] = True

    missing_fact_template = _missing_fact_helpful_template(
        replace(result, topic_id=topic, route=route, draft_text=draft_text, safety_flags=tuple(dict.fromkeys(flags))),
        client_message=client_message,
        context=context,
        markers=markers,
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

    if route == result.route and draft_text == result.draft_text and tuple(flags) == result.safety_flags:
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

    def demote(route: str, reason: str, checklist_item: str) -> SubscriptionDraftResult:
        flags.append(reason)
        checklist.append(checklist_item)
        metadata[reason] = True
        return replace(
            result,
            route=route,
            safety_flags=tuple(dict.fromkeys(flags)),
            manager_checklist=tuple(dict.fromkeys(checklist)),
            metadata=metadata,
        )

    if markers or is_high_risk_result(result):
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
    if _context_has_missing_fact_signal(context):
        return demote(
            "draft_for_manager",
            "autonomy_default_cautious_missing_facts",
            "Автономный ответ запрещен: есть недостающие факты.",
        )
    if not _has_client_safe_current_fact(context):
        return demote(
            "draft_for_manager",
            "autonomy_default_cautious_unverified_fact",
            "Автономный ответ запрещен: нет факта с флагами client-safe и актуальности.",
        )

    flags.append("autonomy_matrix_passed")
    metadata["autonomy_matrix_passed"] = True
    draft_text = result.draft_text
    if _draft_is_low_value_without_exact_fact(draft_text):
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
    lowered = result.draft_text.casefold()
    leaked = tuple(term for term in forbidden_terms if term in lowered)
    if not leaked:
        return result
    return _brand_guarded_result(result, reason="cross_brand_client_text_blocked", leaked_terms=leaked)


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
    texts = [str(client_message or "")]
    if isinstance(context, Mapping):
        recent = context.get("recent_messages")
        if isinstance(recent, Sequence) and not isinstance(recent, (str, bytes, bytearray)):
            texts.extend(str(item or "") for item in recent[-3:])
        for key in ("risk_flags", "context_warnings", "missing_facts"):
            value = context.get(key)
            if isinstance(value, str):
                texts.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
                texts.extend(str(item or "") for item in value)
    haystack = "\n".join(texts)
    markers = [name for name, pattern in HIGH_RISK_INPUT_PATTERNS if pattern.search(haystack)]
    if LEGAL_CONTEXT_INPUT_RE.search(haystack):
        markers.append("legal")
    return tuple(dict.fromkeys(markers))


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
    return bool(FUTURE_PRICE_INPUT_RE.search(client_text))


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


def _discount_safe_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    active_brand = _active_brand(context)
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
    client_haystack = str(client_message or "").casefold()
    if isinstance(context, Mapping):
        for key in ("risk_flags", "context_warnings"):
            value = context.get(key)
            text = " ".join(str(item or "") for item in value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else str(value or "")
            haystack += " " + text.casefold()
    if "forced_discount" in haystack or ("скидк" in client_haystack and "конкурент" in client_haystack):
        return FORCED_DISCOUNT_SAFE_TEXT
    if "stacking" in haystack or "сложат" in client_haystack or "суммир" in client_haystack:
        return DISCOUNT_STACKING_SAFE_TEXT
    if active_brand == "foton" and (
        "second_subject" in haystack
        or "два предмет" in client_haystack
        or "второй" in client_haystack
        or re.search(r"(физик\w*[^.!?\n]{0,80}математ\w*|математ\w*[^.!?\n]{0,80}физик\w*)", client_haystack)
    ):
        return FOTON_SECOND_SUBJECT_DISCOUNT_TEXT
    if active_brand == "unpk" and (
        "second_subject" in haystack
        or "два предмет" in client_haystack
        or "второй" in client_haystack
        or re.search(r"(физик\w*[^.!?\n]{0,80}математ\w*|математ\w*[^.!?\n]{0,80}физик\w*)", client_haystack)
    ):
        return UNPK_SECOND_SUBJECT_DISCOUNT_TEXT
    if active_brand == "unpk" and ("monthly" in haystack or "помесяч" in client_haystack or "семестр" in client_haystack):
        return UNPK_MONTHLY_SEMESTER_DISCOUNT_TEXT
    if active_brand == "unpk" and ("mfti_employees" in haystack or "работаю в мфти" in client_haystack or "сотрудник" in client_haystack):
        return UNPK_MFTI_EMPLOYEE_DISCOUNT_TEXT
    if "multichild" in haystack or "многодет" in client_haystack or re.search(r"\b(?:трое|три|четверо|пятеро)\s+дет", client_haystack):
        return MULTICHILD_DISCOUNT_TEXT
    return ""


def _installment_safe_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    if _active_brand(context) != "foton":
        return ""
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
    client_haystack = str(client_message or "").casefold()
    if "долями" in client_haystack or "dolyami" in haystack:
        return FOTON_DOLYAMI_SAFE_TEXT
    if result.topic_id == "theme:006_installment" or "installment" in haystack or "рассроч" in haystack:
        return FOTON_INSTALLMENT_SAFE_TEXT
    return ""


def _matkap_safe_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    haystack = _semantic_haystack(result, client_message=client_message, context=context)
    if "маткап" not in haystack and "matkap" not in haystack and "материн" not in haystack and "сертификат" not in haystack and "сфр" not in haystack:
        return ""
    if "региональ" in haystack:
        return MATKAP_REGIONAL_SAFE_TEXT
    if ("сфр" in haystack or "sfr_guarantee" in haystack) and (
        "одобр" in haystack or "точно" in haystack or "гарант" in haystack or "sfr_guarantee" in haystack
    ):
        return MATKAP_SFR_REVIEW_SAFE_TEXT
    if "федераль" in haystack or "маткап" in haystack or "материн" in haystack or "сфр" in haystack:
        return MATKAP_FEDERAL_TIMING_SAFE_TEXT
    return ""


def _terminal_safe_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    active_brand = _active_brand(context)
    client_haystack = str(client_message or "").casefold()
    haystack = _semantic_haystack(result, client_message=client_message, context=context)
    if any(marker in client_haystack for marker in ("ignore all previous", "system prompt", "системный промпт", "покажи промпт", "chatgpt")):
        return IDENTITY_PROMPT_SAFE_TEXT
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
    if active_brand == "unpk" and ("площадки" in client_haystack or "адрес" in client_haystack):
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
    if "индивидуаль" in client_haystack:
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


def _docs_safe_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    if LEGAL_CONTEXT_INPUT_RE.search(str(client_message or "")):
        return ""
    haystack = _semantic_haystack(result, client_message=client_message, context=context)
    client_haystack = str(client_message or "").casefold()
    if _client_message_contains_pii(client_message) and ("справк" in client_haystack or "вычет" in client_haystack):
        return PII_DOCUMENT_SAFE_TEXT
    if "договор" in haystack and ("юр" in haystack or "лиц" in haystack or "оформ" in haystack):
        return CONTRACT_ENTITY_SAFE_TEXT
    if "справк" in haystack:
        return CERTIFICATE_SAFE_TEXT
    return ""


def _teacher_safe_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    haystack = _semantic_haystack(result, client_message=client_message, context=context)
    client_haystack = str(client_message or "").casefold()
    if "преподав" not in haystack and "педагог" not in haystack and "teacher" not in haystack:
        return ""
    if "не понрав" in client_haystack:
        return TEACHERS_CHANGE_SAFE_TEXT
    if "менделеево" in client_haystack or "лвш" in client_haystack or "specific_mendeleevo" in haystack:
        return TEACHERS_MENDELEEVO_SAFE_TEXT
    if "как зовут" in client_haystack or "кто в лобне" in client_haystack or "кто работает" in client_haystack or "specific_lobnya" in haystack or "specific_name" in haystack:
        return TEACHERS_SPECIFIC_SAFE_TEXT
    return TEACHERS_GENERAL_SAFE_TEXT


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
        if (
            any(marker in client_lower for marker in ("когда", "дат", "заезд", "смен"))
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
        return (
            f"Да, сориентирую по проверенным условиям. {fact_sentence} "
            "Если напишете класс ребёнка и удобный формат, подберём самый подходящий вариант оплаты."
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
            cleaned = " ".join(str(item or "").split())
            if cleaned:
                texts.append(cleaned)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            cleaned = " ".join(str(item or "").split())
            if cleaned:
                texts.append(cleaned)
    return tuple(dict.fromkeys(texts[: max(1, limit)]))


def _prefer_format_facts(facts: Sequence[str], *, query: str = "") -> tuple[str, ...]:
    query_text = str(query or "").casefold().replace("ё", "е")
    asks_online = "онлайн" in query_text or "дистанц" in query_text
    asks_offline = "очно" in query_text or "офлайн" in query_text
    preferred: list[str] = []
    for fact in facts:
        normalized = fact.casefold().replace("ё", "е")
        if "учебный год" in normalized or "уровень обучения" in normalized:
            continue
        if asks_online and "очно" in normalized and "онлайн" not in normalized.replace("онлайн-платформа", ""):
            continue
        if asks_offline and "онлайн" in normalized and "очно" not in normalized:
            continue
        if any(marker in normalized for marker in ("онлайн-платформа", "мтс линк", "webinar", "запис", "очно", "онлайн")):
            preferred.append(fact)
    return tuple(preferred)


def _ensure_sentence(text: str) -> str:
    value = " ".join(str(text or "").split()).rstrip()
    if not value:
        return ""
    return value if value.endswith((".", "!", "?")) else f"{value}."


def _tax_safe_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    haystack = _semantic_haystack(result, client_message=client_message, context=context)
    if "налог" not in haystack and "вычет" not in haystack and "фнс" not in haystack:
        return ""
    if "лиценз" in haystack:
        return TAX_LICENSE_SAFE_TEXT
    if any(marker in haystack for marker in ("сумм", "сколько", "верн", "возврат", "13%", "110 000", "14 300")):
        return TAX_AMOUNT_SAFE_TEXT
    if "онлайн" in haystack and ("очн" in haystack or "заоч" in haystack):
        return TAX_ONLINE_FORM_SAFE_TEXT
    if "фнс" in haystack and ("верн" in haystack or "точно" in haystack or "гарант" in haystack):
        return TAX_FNS_REVIEW_SAFE_TEXT
    return ""


def _camp_safe_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    active_brand = _active_brand(context)
    haystack = _semantic_haystack(result, client_message=client_message, context=context)
    client_haystack = str(client_message or "").casefold()
    if not any(marker in (client_haystack + " " + haystack) for marker in ("лагер", "лвш", "лш", "смен", "менделеево")):
        return ""
    price_question = bool(
        "сколько стоит" in client_haystack
        or "стоим" in client_haystack
        or "цен" in client_haystack
        or "почем" in client_haystack
        or "почём" in client_haystack
        or "прайс" in client_haystack
    )
    if price_question and ("менделеево" in client_haystack or "лвш" in client_haystack):
        if active_brand == "foton":
            return FOTON_LVSH_PRICE_SAFE_TEXT
        if active_brand == "unpk":
            return UNPK_LVSH_PRICE_SAFE_TEXT
    if active_brand == "unpk" and ("зимн" in haystack or "звш" in haystack):
        return UNPK_ZVSH_WAITLIST_SAFE_TEXT
    if active_brand == "foton" and ("все смен" in client_haystack or "какие смен" in client_haystack or "лвш" in client_haystack):
        return FOTON_LVSH_DATES_SAFE_TEXT
    if active_brand == "unpk" and ("все смен" in client_haystack or "какие смен" in client_haystack or "лвш" in client_haystack):
        return UNPK_LVSH_DATES_SAFE_TEXT
    if active_brand == "unpk" and ("мест" in client_haystack or ("сколько" in client_haystack and not price_question)):
        return UNPK_LVSH_SEATS_SAFE_TEXT
    if active_brand == "foton" and ("август" in client_haystack or "городск" in client_haystack or "лш" in client_haystack):
        return FOTON_CITY_CAMP_AUGUST_SAFE_TEXT
    if active_brand == "unpk" and "июн" in client_haystack:
        return UNPK_JUNE_CAMP_HANDOFF_TEXT
    return ""


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


def _client_message_contains_pii(client_message: str) -> bool:
    text = str(client_message or "")
    return bool(
        re.search(r"\+?\d[\d\s().-]{7,}\d", text)
        or re.search(r"\bдоговор\w*\s*(?:номер|№)?\s*\d+", text, flags=re.I)
        or re.search(r"\b(?:фио|паспорт|снилс|инн|email|e-mail)\b", text, flags=re.I)
    )


def _pricing_safe_template(
    result: SubscriptionDraftResult,
    *,
    client_message: str = "",
    context: Optional[Mapping[str, Any]] = None,
) -> str:
    active_brand = _active_brand(context)
    client_haystack = str(client_message or "").casefold()
    if active_brand == "foton" and ("онлайн" in client_haystack and ("сколько" in client_haystack or "сто" in client_haystack or "цена" in client_haystack)):
        return FOTON_ONLINE_PRICE_SAFE_TEXT
    if active_brand != "unpk":
        return ""
    haystack = " ".join([client_haystack, result.topic_id, result.broad_group, *result.alternative_themes, *result.context_warnings]).casefold()
    fact_text = _normalized_fact_text(context)
    if (
        "егэ" in client_haystack
        and ("интенсив" in client_haystack or "перед" in client_haystack)
        and ("сколько" in client_haystack or "стоим" in client_haystack or "цен" in client_haystack)
        and "18 800" in fact_text
        and "34 400" in fact_text
    ):
        return UNPK_EGE_INTENSIVE_PRICE_SAFE_TEXT
    if (
        ("4 недели" in client_haystack or "четыре недели" in client_haystack)
        and ("сколько" in client_haystack or "стоим" in client_haystack or "цен" in client_haystack)
        and ("нов" in client_haystack or "ученик" in client_haystack)
        and ("10 900" in fact_text or "10900" in fact_text)
        and ("9 900" in fact_text or "9900" in fact_text)
    ):
        return UNPK_FOUR_WEEKS_NEW_PRICE_SAFE_TEXT
    if "физтех" in haystack and "олимпиад" in haystack:
        if re.search(r"\b(?:9|11)\s*(?:класс|классе|кл\.?)", client_haystack):
            return UNPK_OLYMPIAD_PHYSTECH_PRICE_TEXT
        return UNPK_OLYMPIAD_PHYSTECH_HANDOFF_TEXT
    if (
        ("цен" in client_haystack or "стоим" in client_haystack or "прайс" in client_haystack)
        and re.search(r"\b(?:5|6|7|8|9|10|11)\s*(?:класс|классе|кл\.?)", client_haystack)
    ):
        return UNPK_GRADES_5_11_PRICE_SAFE_TEXT
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
    return "рассроч" in haystack or "частями" in haystack or "помесяч" in haystack


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


def _brand_guarded_result(
    result: SubscriptionDraftResult,
    *,
    reason: str,
    leaked_terms: Sequence[str] = (),
) -> SubscriptionDraftResult:
    return replace(
        result,
        route="manager_only",
        draft_text=SAFE_FALLBACK_DRAFT_TEXT,
        safety_flags=tuple(dict.fromkeys([*result.safety_flags, reason, "brand_separation_guarded"])),
        manager_checklist=tuple(
            dict.fromkeys(
                [
                    *result.manager_checklist,
                    "Проверить бренд клиента: в черновике нельзя смешивать Фотон и УНПК.",
                ]
            )
        ),
        metadata={**dict(result.metadata), reason: True, "forbidden_brand_terms": list(leaked_terms)},
    )


def _extract_numeric_promise_claims(text: str) -> tuple[str, ...]:
    source = str(text or "")
    claims: list[str] = []
    for pattern in UNSUPPORTED_PROMISE_PATTERNS:
        for match in pattern.finditer(source):
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

    if _truthy_value(context.get("facts_stale")) or _truthy_value(facts_mapping.get("stale")) or _truthy_value(facts_mapping.get("facts_stale")):
        return ()
    if _truthy_value(quality_mapping.get("facts_stale")):
        return ()

    fresh = (
        context.get("facts_fresh") is True
        or facts_mapping.get("fresh") is True
        or facts_mapping.get("facts_fresh") is True
        or facts_mapping.get("fresh_facts") is True
    )
    if not fresh:
        return ()

    texts: list[str] = []
    for key in ("confirmed_facts", "facts_context"):
        _append_fact_texts(texts, context.get(key))
    _append_fact_texts(texts, context.get("knowledge_snippets"))
    return tuple(text for text in texts if text)


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
            if str(key).strip().casefold() in {"missing", "facts_missing", "stale", "facts_stale"}:
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
    return any(normalized_claim in _normalize_fact_match_text(text) for text in fact_texts)


def _normalize_fact_match_text(text: Any) -> str:
    value = str(text or "").casefold().replace("ё", "е").replace("\u00a0", " ")
    return " ".join(value.split())


def _truthy_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да"}


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
