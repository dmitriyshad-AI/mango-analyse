from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence

from openai import OpenAI
from sqlalchemy import text, update as sa_update
from sqlalchemy.orm import Session

from mango_mvp.clients.ollama import OllamaClient
from mango_mvp.config import Settings
from mango_mvp.models import CallRecord
from mango_mvp.services.controlled_call_scope import (
    call_artifact_directory,
    require_unique_controlled_call,
    write_call_artifact_bytes,
)
from mango_mvp.quality.non_conversation import detect_non_conversation_signals
from mango_mvp.quality.tenant_text_normalizer import (
    TENANT_TEXT_ENGINE_VERSION,
    normalize_manager_text_with_provenance,
    tenant_ruleset_version,
)
from mango_mvp.services.dialogue_contract import (
    ANALYSIS_SCHEMA_VERSION_V3,
    CALLS_TENANT_ID,
    CLAIM_CONTRACT_VERSION,
    CLAIM_REASON_PREFIX,
    DETECTOR_CONTRACT_VERSION,
    HISTORY_SUMMARY_CONTRACT_VERSION,
    PREFERRED_CHANNELS,
    PRICE_SENSITIVITY_VALUES,
    RESULT_STATUSES,
    RESULT_STATUS_RU,
    ROLE_GUARD_VERSION,
    TIMEZONE_CONTRACT_VERSION,
    DialogueInput,
    apply_role_guard,
    build_display_fields,
    build_dialogue_input,
    call_key_for_record,
    call_record_view,
    canonical_item_key,
    deterministic_claim_id,
    guard_stored_analysis,
    manager_output_sha256,
    # One timezone implementation for Analyse, the summary and the publisher.
    moscow_datetime,
    project_untrusted_analysis,
    review_reasons_ru,
    # One shared implementation: a stage-local copy is how one of the two
    # pipelines keeps leaking the conversation after the other is fixed.
    safe_error_text,
    validate_structured_fields,
    value_sha256,
)
from mango_mvp.services.llm_response_cache import LLMResponseCache
from mango_mvp.services.pipeline_claims import release_stale_pipeline_claims
from mango_mvp.utils.codex_cli import append_codex_service_tier

# The only paths the model may claim, and the only ones the service will
# evidence.  Everything else it might invent has nowhere to land.
CLAIM_FIELD_PATHS: Dict[str, Dict[str, Any]] = {
    "structured_fields.result.status": {"kind": "scalar", "categorical": True},
    "structured_fields.result.detail": {"kind": "scalar"},
    "structured_fields.objections": {"kind": "list", "categorical": True},
    "structured_fields.next_step.action": {"kind": "scalar"},
    "structured_fields.next_step.due": {"kind": "scalar"},
    "structured_fields.interests.products": {"kind": "list"},
    "structured_fields.interests.format": {"kind": "list"},
    "structured_fields.interests.subjects": {"kind": "list"},
    "structured_fields.interests.exam_targets": {"kind": "list"},
    "structured_fields.student.grade_current": {"kind": "scalar"},
    "structured_fields.student.school": {"kind": "scalar"},
    "structured_fields.people.parent_fio": {"kind": "scalar"},
    "structured_fields.people.child_fio": {"kind": "scalar"},
    "structured_fields.contacts.email": {"kind": "scalar"},
    "structured_fields.contacts.preferred_channel": {"kind": "scalar", "categorical": True},
    "structured_fields.commercial.price_sensitivity": {"kind": "scalar", "categorical": True},
    "structured_fields.commercial.budget": {"kind": "scalar"},
    "structured_fields.commercial.discount_interest": {"kind": "scalar", "categorical": True},
}

_ALLOWED_FIELD_PATH_LIST = ", ".join(sorted(CLAIM_FIELD_PATHS))

SYSTEM_PROMPT_V3_HEAD = """Strict analyst for Russian EdTech phone calls.
Return a single-line minified JSON object only. No markdown, comments, or extra keys.

The input is the canonical dialogue: one whole reply per line, prefixed with its
turn id (T0001), its timecode and its speaker. Nothing outside those lines is a fact.

Rules:
- Return exactly two root keys: "structured_fields" and "claim_requests". Any other root key rejects the whole answer.
- Fill a field only when a reply states it. If unsupported, return null or [].
- The deterministic hints in the user message are candidates only. Never invent facts from hints: a hint no reply supports is not a fact.
- All text in Russian except emails and phone numbers.
- Do not write a summary, a story, a comment, a quote, a timecode, a hash or a claim id: the service builds all of them from the dialogue itself.
- Every non-empty value needs its own item in "claim_requests"; a list value needs one item per element.
- A claim request has exactly these keys: "field_path", "item_id", "support_type", "turn_ids".
- "item_id" is the exact list element for a list path and null for a scalar path.
- "support_type" is "explicit" when the referenced reply literally states the fact, otherwise "inferred".
- "turn_ids" holds 1 to 3 distinct turn ids that really appear above.
- Intention is not a fact: "payment_confirmed" needs a reply about a payment that already happened, and agreement to buy is at most "sale_agreed".
- A question about a discount is not an agreed discount, and one mention of price is not a price objection.
- next_step.action must be in Russian.
"""

SYSTEM_PROMPT_V3_NON_CONVERSATION_RULE = """- For long transcripts or multi-turn MANAGER/CLIENT dialogue, do not use result.status "non_conversation" just because words like "абонент", "секретарь", "коллекторская организация", "перезвонить", or company auto-greeting markers appear. Use non_conversation only when the client side is exclusively a system/IVR/voicemail/no-live message and there is no human response.
"""

SYSTEM_PROMPT_V3_TAIL = """
Return exactly these keys:
{
  "structured_fields": {
    "result": {"status": null, "detail": null},
    "people": {"parent_fio": null, "child_fio": null},
    "contacts": {"email": null, "preferred_channel": null},
    "student": {"grade_current": null, "school": null},
    "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
    "commercial": {"price_sensitivity": null, "budget": null, "discount_interest": null},
    "objections": [],
    "next_step": {"action": null, "due": null}
  },
  "claim_requests": []
}

Allowed result.status: %(statuses)s, null.
Allowed contacts.preferred_channel: %(channels)s, null.
Allowed commercial.price_sensitivity: %(price)s, null.
Allowed interests.products: "годовые курсы", "летний лагерь", "интенсив", "индивидуальные занятия".
Allowed field_path: %(paths)s.""" % {
    "statuses": ", ".join(RESULT_STATUSES),
    "channels": ", ".join(PREFERRED_CHANNELS),
    "price": ", ".join(PRICE_SENSITIVITY_VALUES),
    "paths": _ALLOWED_FIELD_PATH_LIST,
}

SYSTEM_PROMPT_FULL = (
    SYSTEM_PROMPT_V3_HEAD
    + SYSTEM_PROMPT_V3_NON_CONVERSATION_RULE
    + "- Do not collapse a meaningful call into a single empty answer: fill every path the dialogue supports.\n"
    + SYSTEM_PROMPT_V3_TAIL
)

SYSTEM_PROMPT_COMPACT = SYSTEM_PROMPT_V3_HEAD + SYSTEM_PROMPT_V3_TAIL

STRONG_NON_CONVERSATION_MARKERS = (
    "продолжение следует",
    "голосовой ассистент",
    "голосовой помощник",
    "я секретарь",
    "на связи я секретарь",
    "ассистент миа",
    "временно попросили отвечать",
    "абонент не может ответить",
    "абонент временно недоступен",
    "абонент занят",
    "вызываемый абонент",
    "номер недоступен",
    "вне зоны действия",
    "телефон выключен",
    "звонок был перенаправлен",
    "оставьте сообщение",
    "после сигнала",
    "отправить бесплатное смс",
    "нажмите 1",
    "целевые финансы",
    "7sky",
    "сервис резерв",
    "актив бизнес консалт",
    "коллекторская организация",
)

WEAK_NON_CONVERSATION_MARKERS = (
    "оставайтесь на линии",
    "дозванивайтесь",
    "дозваниваться",
)

TECHNICAL_CALL_PATTERNS = (
    re.compile(
        r"личн\w* кабинет|не открыва\w*|не работа\w*|ошибк\w*|ссылк\w*|подключ\w*|"
        r"логин|парол\w*|код подтвержден\w*|смс|вебинар|zoom|зум|платформ\w*|"
        r"доступ\w*|тест\b|онлайн[- ]?тест",
        re.I,
    ),
)

SERVICE_CALL_PATTERNS = (
    re.compile(
        r"оплат\w*|счет\w*|чек\w*|договор\w*|расписан\w*|перенос\w*|отмен\w*|возврат\w*|"
        r"заняти\w*|урок\w*|преподавател\w*|куратор\w*|домашн\w*|пробник\w*|срез\w*|"
        r"посещаемост\w*|доступ к урокам|доступ к материалам",
        re.I,
    ),
)

EXISTING_CLIENT_PROGRESS_PATTERNS = (
    re.compile(
        r"обратн\w* связ\w*|как проходит|как вам курс|втор\w* семестр|"
        r"продолж\w* обучен\w*|ранее обучал\w*|уже обуча\w*|результат\w*|"
        r"успеваемост\w*|по текущему курсу|на следующий год",
        re.I,
    ),
)

CALL_TYPE_TAGS = {
    "non_conversation",
    "technical_call",
    "service_call",
    "existing_client_progress",
    "sales_call",
}

# What ``_normalize_analysis`` alone can promise.  A payload only becomes v3
# once the claim contract has actually run against a canonical dialogue, so a
# re-read of an old row keeps saying v2 instead of pretending to have evidence.
LATEST_ANALYSIS_SCHEMA_VERSION = "v2"
# Bumped twice: for the canonical dialogue input (whole replies with a
# ``turn_id``, a timecode and a physical speaker instead of two channel
# monoliths) and for the v3 claim contract.  The prompt bytes changed, so the
# cache key changes with them — an old cached answer was produced from a
# different conversation shape and a different contract.
ANALYZE_PROMPT_VERSION_COMPACT = "v8"
ANALYZE_PROMPT_VERSION_FULL = "v9"
TRANSCRIPT_QUALITY_GUARDRAILS_VERSION = "non_conversation_v4_live_safeguards"
NON_CONVERSATION_ADVISORY_ENV = "TELEGRAM_NON_CONVERSATION_ADVISORY"
TRUE_ENV_VALUES = {"1", "true", "yes", "y", "on", "да"}

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
GRADE_RE = re.compile(r"\b((?:[1-9]|1[0-1]))(?:-?й)?\s*класс(?:а)?\b", re.I)
SPEAKER_LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\]\s*(?P<speaker>[^:]{1,60}):\s*(?P<text>.+)$",
    re.U,
)
SUBJECT_PATTERNS = {
    "математика": re.compile(r"математ\w*", re.I),
    "физика": re.compile(r"физик\w*", re.I),
    "информатика": re.compile(r"информат\w*|python|питон|алгоритм", re.I),
    "русский язык": re.compile(r"русск\w* язык", re.I),
    "химия": re.compile(r"хими\w*", re.I),
    "биология": re.compile(r"биолог\w*", re.I),
    "английский язык": re.compile(r"английск\w*", re.I),
}
FORMAT_PATTERNS = {
    "онлайн": re.compile(r"\bонлайн\b", re.I),
    "оффлайн": re.compile(r"\bоффлайн\b|\bочно\b", re.I),
    "групповой": re.compile(r"\bгрупп\w*", re.I),
    "индивидуальный": re.compile(r"индивидуал\w*", re.I),
    "с проживанием": re.compile(r"с проживанием|проживани[ея]", re.I),
    "без проживания": re.compile(r"без проживания", re.I),
}
PRODUCT_PATTERNS = {
    "годовые курсы": re.compile(r"годов\w* курс|на год|годов\w* программ", re.I),
    "летний лагерь": re.compile(r"летн(?:ий|его) лаг|летн\w* смен|летн\w* школ|выездн\w* школ", re.I),
    "интенсив": re.compile(r"интенсив\w*", re.I),
    "индивидуальные занятия": re.compile(r"индивидуал\w* занят|репетиторств|индивидуальн\w* формат", re.I),
}
EXAM_PATTERNS = {
    "ЕГЭ": re.compile(r"\bегэ\b", re.I),
    "ОГЭ": re.compile(r"\bогэ\b", re.I),
    "олимпиады": re.compile(r"олимпиад\w*", re.I),
}
OBJECTION_PATTERNS = {
    "цена": re.compile(
        r"\bцен(?:а|е|у|ы|ой|ам|ами|ник\w*)\b|\bстоимост\w*\b|\bдорог\w*\b|\bдешев\w*\b|\bбюджет\w*\b",
        re.I,
    ),
    "время": re.compile(r"нет времени|занят\w*|нагрузк\w*|расписан\w*", re.I),
    "доверие": re.compile(r"кто вы|не слышал\w* о вас|отзыв\w*|гаранти", re.I),
    "неактуально": re.compile(r"не актуальн\w*|не интерес\w*|не нужно", re.I),
}
DIALOGUE_DUMP_LINE_RE = re.compile(r"^\[(?:~)?\d{2}:\d{2}(?:\.\d+)?\]\s*", re.M)
ROLE_PREFIX_RE = re.compile(r"^\s*(manager|client|менеджер|клиент)\s*:\s*", re.I | re.M)

# --------------------------------------------------------------------------
# ТЗ-03: the model points at replies, the service writes the evidence.
# --------------------------------------------------------------------------

V3_ROOT_KEYS = frozenset({"structured_fields", "claim_requests"})
CLAIM_REQUEST_KEYS = frozenset({"field_path", "item_id", "support_type", "turn_ids"})
CLAIM_SUPPORT_TYPES = ("explicit", "inferred")
CLAIM_MAX_TURN_REFS = 3
TURN_ID_RE = re.compile(r"^T\d{4}$")

# A discount is only discussed where one of these words is; a payment only
# happened where the reply says it happened, not where it was promised.
DISCOUNT_ANCHOR_RE = re.compile(r"скидк\w*|акци\w*|рассрочк\w*", re.I)
PRICE_ACCEPTED_RE = re.compile(
    r"(?:цен\w*|стоимост\w*)[^.!?…]{0,48}(?:нормальн\w*|устраива\w*|"
    r"подход\w*|приемлем\w*|адекват\w*|не\s+дорог\w*)",
    re.I,
)
PRICE_OBJECTION_RE = re.compile(
    r"(?:цен\w*|стоимост\w*)[^.!?…]{0,48}(?:не\s+устраива\w*|не\s+подход\w*|"
    r"высок\w*|завыш\w*|дорог\w*|слишком)|"
    r"(?:беспокоит|смущает|не\s+устраива\w*|вопрос\s+по)\s+(?:цен\w*|стоимост\w*)|"
    r"(?:дорог\w*|не\s+по\s+карману|не\s+укладыва\w*|нет\s+денег|"
    r"бюджет\w*\s+(?:не\s+позволя\w*|огранич\w*))",
    re.I,
)
CONDITIONAL_COMMITMENT_RE = re.compile(
    r"\b(?:если|только\s+если|при\s+условии|при\s+возможности|в\s+случае|"
    r"после\s+того[, ]+как|возможно|наверное|вероятно|скорее\s+всего|"
    r"может(?:\s+быть)?|мог(?:ла|ли|ло)?\s+бы|"
    r"рассматрива(?:ю|ем|ет)\s+возможность|(?:я|мы)\s+бы\s+хотел(?:а|и)?|"
    r"хотел(?:а|и|о)?\s+бы|можно\s+(?:было\s+)?бы)\b",
    re.I,
)
NEXT_STEP_COMMITMENT_RE = re.compile(
    r"\b(?:нужно|надо|необходимо|давайте|договорил\w*|планир\w*|"
    r"перезвон(?:ю|им|ите|ить)|позвон(?:ю|им|ите|ить)|свяж(?:усь|емся|итесь|аться)|"
    r"отправ(?:лю|им|ьте|ить)|вышл(?:ю|ем|ите|ать)|пришл(?:ю|ем|ите|ать)|"
    r"направ(?:лю|им|ьте|ить)|скин(?:у|ем|ьте|уть)|дожд(?:усь|ёмся|емся|аться)|"
    r"подума(?:ю|ем|ть)|реш(?:у|им|ить)|согласу(?:ю|ем|йте|овать)|"
    r"ожида(?:ю|ем|ть)|жд(?:у|ём|ем))\b",
    re.I,
)
HISTORICAL_CONTEXT_RE = re.compile(
    r"\b(?:раньше|ранее|когда[- ]то|в\s+прошл(?:ом|ый)\s+"
    r"(?:году|месяце|квартале|семестре|сезоне|раз)|"
    r"на\s+прошл(?:ой|ом)\s+(?:неделе|месяце|квартале|курсе|смене)|"
    r"в\s+(?:предыдущем|позапрошлом)\s+году|в\s+(?:19|20)\d{2}\s+году|"
    r"(?:\d+|один|одну|два|две|три|четыре|пять|несколько|полтора|пару)\s+"
    r"(?:дн\w*|недел\w*|месяц\w*|год\w*|смен\w*|сезон\w*|заезд\w*)\s+назад|"
    r"(?:дн\w*|недел\w*|месяц\w*|лет)\s+"
    r"(?:\d+|один|два|три|четыре|пять|несколько)\s+назад|"
    r"(?:день|недел\w*|месяц\w*|год|полгода)\s+назад|давно|"
    r"(?:прошл(?:ой|ою|ым)|позапрошл(?:ой|ою|ым)|минувш(?:ей|ею|им)|"
    r"предыдущ(?:ей|ею|им))\s+(?:осенью|весной|зимой|летом)|"
    r"(?:в|на)\s+(?:предыдущ(?:ей|ем)|позапрошл(?:ой|ом))\s+"
    r"(?:смене|курсе|заезде|сезоне)|"
    r"(?:ещ[её]\s+)?в\s+(?:19|20)\d{2}[- ]?(?:м|ом)?|"
    r"на\s+прошл(?:ом|ый)\s+(?:курсе|лагере|заезде)|"
    r"за\s+прошл(?:ый|ую)\s+(?:курс|смену)|до\s+этого)\b",
    re.I,
)
CURRENT_CONTEXT_RE = re.compile(
    r"\b(?:сейчас|сегодня|теперь|в\s+этот\s+раз|в\s+этом\s+году|"
    r"(?:по|за)\s+текущ(?:ий|ую|ему)|на\s+текущ(?:ий|ую)|"
    r"на\s+этот\s+(?:курс|лагерь|заезд))\b",
    re.I,
)
NEXT_STEP_CLOSED_RE = re.compile(
    r"\b(?:вопрос|задач|просьб|проблем)\w*\s+(?:уже\s+)?"
    r"(?:закрыт\w*|реш[её]н\w*|снят\w*|неактуальн\w*)\b",
    re.I,
)
NEXT_STEP_GLOBAL_CANCELLATION_RE = re.compile(
    NEXT_STEP_CLOSED_RE.pattern
    + r"|\bконтакт\w*\s+(?:полностью\s+)?отмен[её]н\w*\b|"
    r"\b(?:полностью\s+)?отказ(?:ался|алась|ались|ываюсь|ываемся)\s+"
    r"от\s+(?:обучения|курса|лагеря|покупки)\b",
    re.I,
)
NEXT_STEP_ACTION_END_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "Перезвонить клиенту": re.compile(
        r"\b(?:созвон\w*\s+(?:уже\s+)?(?:состоял\w*|заверш[её]н\w*)|"
        r"(?:уже\s+)?(?:перезвонил[аи]?|позвонил[аи]?|созвонил(?:ся|ась|ись))|"
        r"с\s+клиент\w*[^.!?…]{0,16}(?:уже\s+)?связал(?:ся|ась|ись)|"
        r"(?:уже\s+)?связал(?:ся|ась|ись)[^.!?…]{0,16}с\s+клиент\w*|"
        r"(?:перезв[ао]н|созвон|звон)\w*\s+(?:больше\s+)?"
        r"(?:не\s+треб\w*|отмен(?:я\w*|[её]н\w*))|"
        r"звонок\s+уже\s+(?:был|состоял\w*|прош[её]л)|"
        r"отмен\w*[^.!?…]{0,32}(?:перезв[ао]н|созвон|звон)\w*|"
        r"не\s+смог\w*[^.!?…]{0,32}(?:перезв[ао]н|созвон|звон)\w*|"
        r"(?:перезванивать|созваниваться|звонить)\s+(?:больше\s+)?"
        r"не\s+(?:надо|нужно|треб\w*)|"
        r"решил\w*\s+больше\s+не\s+(?:созван|звон)\w*|"
        r"не\s+(?:звон(?:ите|ить)|перезванива(?:йте|ть)))\b",
        re.I,
    ),
    "Отправить материалы": re.compile(
        r"\b(?:(?:материал|информац|презентац|программ|документ)\w*[^.!?…]{0,40}"
        r"(?:уже\s+)?(?:отправлен|выслан|прислан|направлен|"
        r"отправил|выслал|прислал|направил)\w*|"
        r"(?:уже\s+)?(?:отправил|выслал|прислал|направил)[аи]?[^.!?…]{0,40}"
        r"(?:материал|информац|презентац|программ|документ)\w*|"
        r"(?:материал|информац|презентац|программ|документ)\w*[^.!?…]{0,30}"
        r"(?:не\s+нуж\w*|не\s+надо|неактуальн\w*|"
        r"не\s+(?:отправля(?:йте|ть)|присыла(?:йте|ть)|"
        r"высыла(?:йте|ть)|направля(?:йте|ть)))|"
        r"не\s+(?:надо\s+)?(?:отправля(?:йте|ть)|присыла(?:йте|ть))\s+"
        r"(?:материал|информац|презентац|программ|документ)\w*)\b",
        re.I,
    ),
    "Отправить ссылку на оплату": re.compile(
        r"\b(?:ссылк\w*[^.!?…]{0,30}оплат\w*[^.!?…]{0,30}"
        r"(?:уже\s+)?(?:отправлен|выслан|прислан|направлен|"
        r"отправил|выслал|прислал|направил)\w*|"
        r"(?:уже\s+)?(?:отправил|выслал|прислал|направил)[аи]?[^.!?…]{0,40}"
        r"ссылк\w*[^.!?…]{0,20}оплат\w*|"
        r"ссылк\w*[^.!?…]{0,30}оплат\w*[^.!?…]{0,30}"
        r"(?:не\s+нуж\w*|не\s+надо|неактуальн\w*|"
        r"не\s+(?:отправля(?:йте|ть)|присыла(?:йте|ть)|"
        r"высыла(?:йте|ть)|направля(?:йте|ть))))\b",
        re.I,
    ),
    "Дождаться решения клиента": re.compile(
        r"\b(?:решени\w*\s+(?:уже\s+)?принят\w*|"
        r"(?:уже\s+)?определил(?:ся|ась|ись)|"
        r"(?:мы|клиент)\s+(?:уже\s+)?решил\w*|"
        r"решени\w*[^.!?…]{0,24}(?:больше\s+)?(?:не\s+нужно|не\s+надо))\b",
        re.I,
    ),
}
NEXT_STEP_ACTION_END_PATTERNS["Согласовать следующий контакт"] = re.compile(
    NEXT_STEP_ACTION_END_PATTERNS["Перезвонить клиенту"].pattern
    + r"|\bконтакт\w*\s+(?:уже\s+)?согласован\w*\b",
    re.I,
)
NEXT_STEP_DUE_MARKER_RE = re.compile(
    r"\b(?:сегодня|завтра|послезавтра|"
    r"(?:в|на)\s+(?:понедельник|вторник|среду|четверг|пятницу|субботу|воскресенье)|"
    r"(?:до|в|на)\s+\d{1,2}(?:[.:]\d{2}|[./-]\d{1,2}(?:[./-]\d{2,4})?))\b",
    re.I,
)
NEXT_STEP_RESCHEDULE_RE = re.compile(
    r"\bперенес\w*|\bвместо\b|\bне\b[^.!?…]{0,24}\bа\b", re.I
)
BUDGET_CONTEXT_RE = re.compile(
    r"\b(?:мой|наш|у\s+нас)\s+бюджет\b|\bбюджет\w*\b|"
    r"готов\w*\s+(?:потратить|выделить|оплатить)|"
    r"рассчитыва\w*\s+на|мож\w*\s+выделить|по\s+деньгам\b",
    re.I,
)
RESULT_DETAIL_CONTEXT_RE = re.compile(
    r"оплат\w*|плат[её]ж\w*|\bберу\b|покупа\w*|оформ\w*|отказ\w*|"
    r"не\s+(?:интерес\w*|нужн\w*|актуальн\w*)|подума\w*|решил\w*|"
    r"встрет\w*|запис\w*|перезвон\w*|созвон\w*|свяж\w*|"
    r"отправил\w*|выслал\w*|направил\w*|расска\w*|уточн\w*|объясн\w*",
    re.I,
)
PAYMENT_DONE_RE = re.compile(
    r"уже\s+оплат\w*|оплатил\w*|оплачен\w*|платеж\w*\s+прош\w*|платёж\w*\s+прош\w*|"
    r"деньги\s+(?:посту|прош|ушл)\w*|перевёл\s+деньги|перевел\s+деньги|чек\s+пришё?л",
    re.I,
)
PAYMENT_CONFIRMATION_BY_MANAGER_RE = re.compile(
    r"вижу[,;:]?\s+(?:что\s+)?(?:получен\w*|зафиксирован\w*)\s+(?:ваш[ау]\s+)?оплат\w*|"
    r"(?:получили|получен\w*|зафиксирован\w*)\s+(?:ваш[ау]\s+)?оплат\w*|"
    r"оплат\w*\s+(?:прош\w*|поступил\w*|получен\w*|зафиксирован\w*)|"
    r"(?:платеж|платёж|деньги)\w*\s+(?:прош\w*|поступил\w*|получен\w*)|"
    r"чек\s+пришё?л",
    re.I,
)
PAYMENT_QUESTION_RE = re.compile(
    r"\?|\bвы\s+(?:ведь\s+|уже\s+)?оплат\w*|\bоплатил[аи]?\s+ли\b",
    re.I,
)
# Deliberately narrow: a proven direct negation inside the same sentence.  It is
# not a language model and does not pretend to be one — an unclear negation is
# review, not a silent "the fact is fine".
NEGATION_WINDOW_RE = re.compile(
    r"\b(?:не|ни|без)\b[^.!?…]{0,24}$|"
    r"\bнет\b(?!\s*[,;:—-])[^.!?…]{0,24}$",
    re.I,
)
NEGATION_AFTER_ANCHOR_RE = re.compile(r"^[^.!?…]{0,48}\b(?:не|ни|нет|без)\b", re.I)
HYPOTHETICAL_AFTER_ANCHOR_RE = re.compile(r"^[^.!?…]{0,16}\bбы\b", re.I)
PAYMENT_REVERSAL_AFTER_ANCHOR_RE = re.compile(
    r"(?:\b(?:но|однако|хотя|точнее|вернее)\b|[.!?…]\s*(?:ой[,\s]*)?нет\b)[\s\S]{0,120}(?:"
    r"денег\s+(?:нет|не\s+перевод)|"
    r"(?:оплат\w*|плат[её]ж\w*|деньг\w*)\s+(?:ещ[её]\s+|пока\s+)?"
    r"не\s+(?:был\w*|прош\w*|поступ\w*|перев\w*|сдел\w*|получ\w*)|"
    r"(?:ещ[её]\s+|пока\s+)?не\s+(?:оплат\w*|перевод\w*|прош\w*|"
    r"поступ\w*|получ\w*)"
    r")",
    re.I,
)
PAYMENT_DENIAL_TURN_RE = re.compile(
    r"(?:\bнет\b[\s\S]{0,32})?(?:я\s+)?не\s+(?:платил\w*|оплатил\w*|"
    r"оплачивал\w*|переводил\w*)|"
    r"(?:оплат\w*|плат[её]ж\w*|деньг\w*)[^.!?…]{0,40}\b(?:не|нет)\b"
    r"[^.!?…]{0,32}(?:прош\w*|поступ\w*|получ\w*|перев\w*|сдел\w*)",
    re.I,
)
PAYMENT_REVERSAL_TURN_RE = re.compile(
    r"(?:оплат|плат[её]ж|деньг)\w*[^.!?…]{0,48}"
    r"(?:отмен\w*|аннулир\w*|возврат\w*|вернул\w*|возвращ\w*)|"
    r"(?:отмен\w*|аннулир\w*|возврат\w*|вернул\w*|возвращ\w*)"
    r"[^.!?…]{0,48}(?:оплат|плат[её]ж|деньг)\w*",
    re.I,
)
SALE_CANCELLATION_TURN_RE = re.compile(
    r"\bпередумал\w*\b|\bотказыва(?:юсь|емся|ется)\b|"
    r"\bне\s+буд(?:у|ем)\s+(?:брать|покупать|оплачивать|оформлять)\b|"
    r"\b(?:брать|покупать|оплачивать|оформлять)\s+не\s+буд(?:у|ем)\b|"
    r"\bне\s+хоч(?:у|ем)\s+(?:брать|покупать|оплачивать|оформлять)\b",
    re.I,
)
SHORT_AFFIRMATIVE_RE = re.compile(
    r"^\s*(?:да|ага|угу|верно|правильно|конечно|именно|подходит|интересует)"
    r"(?:[\s,.!?…]+(?:да|верно|именно))?[\s.!?…]*$",
    re.I,
)

# Deterministic anchors for the values our own regexes produce.  A claim on one
# of these paths has to point at a reply the very same detector fires on, so a
# model cannot attach "физика" to a reply that never mentions it.
CLAIM_LIST_ANCHORS: Dict[str, Dict[str, "re.Pattern[str]"]] = {
    "structured_fields.interests.subjects": SUBJECT_PATTERNS,
    "structured_fields.interests.products": PRODUCT_PATTERNS,
    "structured_fields.interests.format": FORMAT_PATTERNS,
    "structured_fields.interests.exam_targets": EXAM_PATTERNS,
    "structured_fields.objections": OBJECTION_PATTERNS,
}
# The canonical next-step labels are ours, not the caller's words, so a literal
# search would never find them.  Each one is proven by the reply that made the
# deterministic rule fire — the same rule, pinned to the exact turn.
NEXT_STEP_ANCHORS: Dict[str, "re.Pattern[str]"] = {
    "Перезвонить клиенту": re.compile(r"перезвон\w*|созвон\w*|позвон\w*", re.I),
    "Отправить материалы": re.compile(
        r"отправ\w*|вышл\w*|пришл\w*|направ\w*|скин\w*", re.I
    ),
    "Отправить ссылку на оплату": re.compile(r"ссылк\w*.{0,40}оплат\w*|оплат\w*.{0,40}ссылк\w*", re.I),
    "Дождаться решения клиента": re.compile(r"реш\w*|подума\w*", re.I),
    "Согласовать следующий контакт": re.compile(r"согласу\w*|договорим\w*|свяж\w*", re.I),
}
# Closed categorical values need a reply that carries their marker, so that a
# direct negation in that same reply can be seen and fail the claim.  This is a
# narrow polarity guard, not language understanding: what it cannot decide it
# refuses, and the value goes to review instead of into the report.
RESULT_STATUS_ANCHORS: Dict[str, "re.Pattern[str]"] = {
    "payment_confirmed": PAYMENT_DONE_RE,
    "sale_agreed": re.compile(
        r"\bберу\b|\bпокупа(?:ю|ем)\b|готов(?:а|ы)?\s+(?:купить|оплатить|оформить)|"
        r"соглас(?:ен|на|ны)\s+(?:купить|оплатить|оформить|на\s+(?:курс|лагерь|обучение))|"
        r"давайте\s+оформ",
        re.I,
    ),
    "appointment_agreed": re.compile(
        r"\b(?:встретимся|записываюсь|запиш(?:ите|емся|усь)|записаться|"
        r"прид(?:у|[её]м)|приед(?:у|ем)|давайте\s+запиш\w*)\b",
        re.I,
    ),
    "follow_up_agreed": re.compile(
        r"\b(?:перезвон(?:ю|им|ите|ить)|созвон(?:юсь|имся|иться)|"
        r"свяж(?:усь|емся|итесь|аться)|набер(?:у|ем|ите|ать))\b",
        re.I,
    ),
    "offer_sent": re.compile(
        r"отправил[аи]?|выслал[аи]?|направил[аи]?|скинул[аи]?|"
        r"отправлен\w*|выслан\w*|направлен\w*",
        re.I,
    ),
    "refusal": re.compile(
        r"(?:курс|лагер|обучени|заняти|школ|смен|программ|предложени|покупк|запис)\w*"
        r"[^.!?…]{0,40}(?:не\s+интерес\w*|не\s+нуж\w*|не\s+актуальн\w*)|"
        r"(?:не\s+интерес\w*|не\s+нуж\w*|не\s+актуальн\w*)[^.!?…]{0,40}"
        r"(?:курс|лагер|обучени|заняти|школ|смен|программ|предложени|покупк|запис)\w*|"
        r"^\s*(?:я\s+)?отказываюсь[.!?…]*\s*$|"
        r"отказ\w*\s+от\s+(?:курс|лагер|обучени|заняти|покупк|запис|предложени)\w*|"
        r"(?:покупать|брать|оплачивать|оформлять)\s+не\s+буд(?:у|ем)|"
        r"(?:курс|лагер|обучени|заняти)\w*[^.!?…]{0,24}не\s+подход\w*|"
        r"решил\w*\s+не\s+(?:записыва\w*|покупа\w*|брать|оплачива\w*|оформля\w*)",
        re.I,
    ),
    "no_decision": re.compile(
        r"подума\w*|посовету\w*|решение\s+(?:пока\s+)?не\s+принят\w*|"
        r"(?:ещ[её]\s+)?не\s+(?:решил\w*|определил\w*)",
        re.I,
    ),
    "information_only": re.compile(r"расска\w*|подскаж\w*|объясн\w*|информац\w*|уточн\w*", re.I),
    "non_conversation": re.compile(
        r"автоответчик|недоступ\w*|оставьте сообщение|не может ответить", re.I
    ),
}
CLIENT_DECIDED_RESULT_STATUSES = frozenset(
    {"sale_agreed", "appointment_agreed", "follow_up_agreed", "refusal", "no_decision"}
)
RESULT_OUTCOME_STATUSES = frozenset(
    {"payment_confirmed", *CLIENT_DECIDED_RESULT_STATUSES}
)
RESULT_LATER_OVERRIDES = {
    status: frozenset(CLIENT_DECIDED_RESULT_STATUSES - {status})
    for status in CLIENT_DECIDED_RESULT_STATUSES
}
RESULT_DIRECT_REVERSAL_PATTERNS: Dict[str, "re.Pattern[str]"] = {
    "sale_agreed": SALE_CANCELLATION_TURN_RE,
    "appointment_agreed": re.compile(
        r"\b(?:не\s+(?:приеду|приду|запишусь)|"
        r"(?:отменяю|отменяем|отменили|отменена)\s+запис\w*|"
        r"запис\w*\s+(?:отменяю|отменяем|отменили|отменена))\b",
        re.I,
    ),
    "follow_up_agreed": re.compile(
        r"\b(?:не\s+буд(?:у|ем)\s+(?:звонить|перезванивать|созваниваться|связываться)|"
        r"(?:звонить|перезванивать|созваниваться|связываться)\s+не\s+буд(?:у|ем)|"
        r"(?:перезв[ао]н|созвон|звонок)\w*\s+(?:уже\s+)?"
        r"отмен(?:я\w*|[её]н\w*))\b",
        re.I,
    ),
}
CLIENT_FACT_PATH_PREFIXES = (
    "structured_fields.people.",
    "structured_fields.contacts.",
    "structured_fields.student.",
    "structured_fields.interests.",
    "structured_fields.commercial.",
)
CLIENT_FACT_PATHS = frozenset({"structured_fields.objections"})
# Both the detector's own vocabulary (telegram/whatsapp/email/site) and the
# contract enum, so either value is provable against the same replies.
PREFERRED_CHANNEL_ANCHORS: Dict[str, "re.Pattern[str]"] = {
    "telegram": re.compile(r"телеграм|telegram", re.I),
    "whatsapp": re.compile(r"ватсап|вотсап|whatsapp", re.I),
    "email": re.compile(r"почт\w*|e-?mail", re.I),
    "site": re.compile(r"сайт\w*|заявк\w*", re.I),
    "phone": re.compile(r"позвон\w*|перезвон\w*|телефон\w*", re.I),
    "messenger": re.compile(r"мессендж\w*|телеграм|telegram|ватсап|вотсап|whatsapp", re.I),
}
CLAIM_VALUE_ANCHORS: Dict[str, Dict[str, "re.Pattern[str]"]] = {
    "structured_fields.result.status": RESULT_STATUS_ANCHORS,
    "structured_fields.contacts.preferred_channel": PREFERRED_CHANNEL_ANCHORS,
    "structured_fields.next_step.action": NEXT_STEP_ANCHORS,
}
# Closed enums: the value has to be heard, so an unknown one is never accepted.
# ``next_step.action`` is deliberately absent — its free-text form is legitimate
# and is proven by the reply the model points at.
_VALUE_ANCHOR_REQUIRED_PATHS = frozenset(
    {"structured_fields.result.status", "structured_fields.contacts.preferred_channel"}
)
CLAIM_SCALAR_ANCHORS: Dict[str, "re.Pattern[str]"] = {
    "structured_fields.student.grade_current": GRADE_RE,
    "structured_fields.commercial.price_sensitivity": OBJECTION_PATTERNS["цена"],
    "structured_fields.commercial.discount_interest": DISCOUNT_ANCHOR_RE,
}
# Which validated paths feed which deterministic sentence of the summary.  This
# is what makes the visible конспект traceable part by part instead of carrying
# one flat list of references that fits every sentence and proves none.
SUMMARY_TEMPLATE_FIELDS = (
    ("student_v1", ("structured_fields.people.parent_fio",
                    "structured_fields.people.child_fio",
                    "structured_fields.student.grade_current")),
    ("topics_v1", ("structured_fields.interests.products",
                   "structured_fields.interests.format",
                   "structured_fields.interests.subjects",
                   "structured_fields.interests.exam_targets")),
    ("objections_v1", ("structured_fields.objections",)),
    ("school_v1", ("structured_fields.student.school",)),
    ("commercial_v1", ("structured_fields.commercial.price_sensitivity",
                       "structured_fields.commercial.budget",
                       "structured_fields.commercial.discount_interest")),
    ("result_v1", ("structured_fields.result.status",
                   "structured_fields.result.detail")),
    ("next_step_v1", ("structured_fields.next_step.action",
                      "structured_fields.next_step.due")),
)


class AnalysisContractError(RuntimeError):
    """The model answer is not the v3 contract; it is rejected as a whole."""


def claim_field_reason(field_path: str, item_key: str = "") -> str:
    """Closed review reason; a list element is named by its key, not its text."""
    suffix = f"[{item_key}]" if item_key else ""
    return f"{CLAIM_REASON_PREFIX}:{field_path}{suffix}"


def validate_v3_model_response(payload: Any) -> Dict[str, Any]:
    """Accept exactly the v3 contract, or reject the answer whole.

    Dropping unknown keys instead of rejecting is open by default: the next
    prompt, the next model or a confused retry brings back ``history_summary``,
    a quote or a ``claim_id`` of its own, and a lenient reader would publish it
    as if the service had built it.  So anything outside the contract fails the
    whole attempt, before a single production field is written.
    """
    if not isinstance(payload, Mapping):
        raise AnalysisContractError("analysis response is not a JSON object")
    # ``quality_flags`` is added by us after the provider answered, never by the
    # model, so it is the one key that may be present besides the contract.
    extra = set(payload) - V3_ROOT_KEYS - {"quality_flags"}
    if extra or not V3_ROOT_KEYS.issubset(set(payload)):
        raise AnalysisContractError("analysis response root keys are not the v3 contract")
    try:
        fields = validate_structured_fields(payload.get("structured_fields"), stored=False)
    except ValueError as exc:
        raise AnalysisContractError(str(exc)) from exc
    raw_requests = payload.get("claim_requests")
    if not isinstance(raw_requests, list):
        raise AnalysisContractError("claim_requests is not a list")
    requests: list[Dict[str, Any]] = []
    for item in raw_requests:
        if not isinstance(item, Mapping) or set(item) != CLAIM_REQUEST_KEYS:
            raise AnalysisContractError("claim request keys are not the v3 contract")
        field_path = item.get("field_path")
        if not isinstance(field_path, str) or field_path not in CLAIM_FIELD_PATHS:
            raise AnalysisContractError("claim request field_path is outside the closed list")
        if item.get("support_type") not in CLAIM_SUPPORT_TYPES:
            raise AnalysisContractError("claim request support_type is not explicit/inferred")
        item_id = item.get("item_id")
        if item_id is not None and not isinstance(item_id, str):
            raise AnalysisContractError("claim request item_id must be a string or null")
        turn_ids = item.get("turn_ids")
        if (
            not isinstance(turn_ids, list)
            or not 1 <= len(turn_ids) <= CLAIM_MAX_TURN_REFS
            or not all(isinstance(one, str) and TURN_ID_RE.fullmatch(one) for one in turn_ids)
            or len(set(turn_ids)) != len(turn_ids)
        ):
            raise AnalysisContractError("claim request turn_ids are invalid")
        requests.append(
            {
                "field_path": str(field_path),
                "item_id": item_id,
                "support_type": str(item.get("support_type")),
                "turn_ids": [str(one) for one in turn_ids],
            }
        )
    validated = dict(payload)
    validated["structured_fields"] = fields
    validated["claim_requests"] = requests
    return validated


def _contradicted(text: str, start: int, end: Optional[int] = None) -> bool:
    """A direct negation or a hypothetical marker around the same anchor."""
    raw = str(text or "")
    after = raw[end if end is not None else start :]
    return bool(
        NEGATION_WINDOW_RE.search(raw[:start])
        or NEGATION_AFTER_ANCHOR_RE.search(after)
        or HYPOTHETICAL_AFTER_ANCHOR_RE.search(after)
    )


def _payment_contradicted(text: str, start: int, end: Optional[int] = None) -> bool:
    raw = str(text or "")
    return _contradicted(raw, start, end) or bool(
        PAYMENT_REVERSAL_AFTER_ANCHOR_RE.search(raw[start:])
        or PAYMENT_REVERSAL_TURN_RE.search(raw)
    )


def _historical_claim(text: str, anchor_start: int) -> bool:
    """True when the nearest time marker makes the anchored event historical."""
    raw = str(text or "")
    left = max(raw.rfind(mark, 0, anchor_start) for mark in ".!?…;") + 1
    right_candidates = [
        position for mark in ".!?…;"
        if (position := raw.find(mark, anchor_start)) >= 0
    ]
    right = min(right_candidates, default=len(raw))
    clause = raw[left:right]
    local_anchor = max(0, anchor_start - left)

    def distance(match: re.Match[str]) -> int:
        if match.end() <= local_anchor:
            return local_anchor - match.end()
        if match.start() >= local_anchor:
            return match.start() - local_anchor
        return 0

    markers = [
        (distance(match), 1, "historical")
        for match in HISTORICAL_CONTEXT_RE.finditer(clause)
    ] + [
        (distance(match), 0, "current")
        for match in CURRENT_CONTEXT_RE.finditer(clause)
    ]
    return bool(markers) and min(markers)[2] == "historical"


class _LiteralMatch:
    """A literal hit that answers ``.start()`` like a regex match does."""

    __slots__ = ("_start", "_end")

    def __init__(self, start: int, end: int) -> None:
        self._start = int(start)
        self._end = int(end)

    def start(self) -> int:
        return self._start

    def end(self) -> int:
        return self._end


ANALYSIS_PROMPT_PASSTHROUGH_FLAGS = (
    "analysis_prompt_sha256",
    "dialogue_version",
    "dialogue_source",
    "dialogue_canonical_sha256",
    "dialogue_selected_turn_ids",
    "dialogue_selected_turn_count",
    "dialogue_total_turn_count",
)

# Exact usage is only ever taken from a provider that reported it.  Nothing is
# estimated from characters: a made-up number in a cost report is worse than an
# honest "the provider did not say".
UNAVAILABLE_TOKEN_USAGE: Dict[str, Any] = {
    "source": "unavailable",
    "prompt_tokens": None,
    "completion_tokens": None,
    "total_tokens": None,
}
# The role guard skipped the model on purpose: this is a zero, not a gap.
SKIPPED_TOKEN_USAGE: Dict[str, Any] = {
    **UNAVAILABLE_TOKEN_USAGE,
    "source": "skipped_untrusted_role",
}
# A deterministic pre-model decision (for example, a proven technical
# non-conversation) also costs exactly zero, but it is not an untrusted-role skip.
SKIPPED_DETERMINISTIC_TOKEN_USAGE: Dict[str, Any] = {
    **UNAVAILABLE_TOKEN_USAGE,
    "source": "skipped_deterministic",
}
# The answer came from the local cache, so nothing was spent on this attempt.
CACHE_HIT_TOKEN_USAGE: Dict[str, Any] = {
    **UNAVAILABLE_TOKEN_USAGE,
    "source": "cache_hit",
}
def _model_invocation_error(
    message: str, attempts: Sequence[Mapping[str, Any]]
) -> RuntimeError:
    error = RuntimeError(message)
    error.model_attempts = [dict(item) for item in attempts]  # type: ignore[attr-defined]
    return error


def _analysis_model_attempts(analysis: Any) -> list[Dict[str, Any]]:
    if not isinstance(analysis, Mapping):
        return []
    flags = analysis.get("quality_flags")
    attempts = flags.get("analyze_attempts") if isinstance(flags, Mapping) else None
    return [dict(item) for item in attempts if isinstance(item, Mapping)] if isinstance(attempts, list) else []


def provider_token_usage(usage: Any) -> Dict[str, Any]:
    """Exact provider counters, or an honest "the provider did not say".

    Nothing is derived here — not even a total from a prompt plus a completion.
    A number in a cost report is read as measured; a plausible number that was
    actually computed by us is worse than a visible gap.
    """
    values: Dict[str, Any] = {}
    for key, names in (
        ("prompt_tokens", ("prompt_tokens", "input_tokens", "prompt_eval_count")),
        ("completion_tokens", ("completion_tokens", "output_tokens", "eval_count")),
        ("total_tokens", ("total_tokens",)),
    ):
        value = None
        for name in names:
            candidate = (
                usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
            )
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate >= 0:
                value = candidate
                break
        values[key] = value
    if all(value is None for value in values.values()):
        return dict(UNAVAILABLE_TOKEN_USAGE)
    source = (
        "provider_exact"
        if all(isinstance(value, int) for value in values.values())
        else "provider_partial"
    )
    return {"source": source, **values}


def aggregate_token_usage(stages: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Sum reported stages while keeping exact, partial and unknown distinct."""
    usages = [stage.get("token_usage") for stage in stages]
    if any(not isinstance(usage, Mapping) for usage in usages):
        return dict(UNAVAILABLE_TOKEN_USAGE)
    provider_sources = {"provider", "provider_exact", "provider_partial"}
    paid = [usage for usage in usages if usage.get("source") in provider_sources]
    if any(usage.get("source") == "unavailable" for usage in usages):
        return dict(UNAVAILABLE_TOKEN_USAGE)
    if not paid:
        return dict(CACHE_HIT_TOKEN_USAGE)
    exact_sources = all(
        usage.get("source") in {"provider", "provider_exact"}
        and all(
            isinstance(usage.get(key), int)
            and not isinstance(usage.get(key), bool)
            and usage.get(key) >= 0
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        for usage in paid
    )
    result: Dict[str, Any] = {
        "source": "provider_exact" if exact_sources else "provider_partial"
    }
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [usage.get(key) for usage in paid]
        result[key] = (
            sum(values)
            if all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in values
            )
            else None
        )
    return result

ANALYSIS_PROMPT_TRANSCRIPT_MAX_CHARS_FULL = 10000
ANALYSIS_PROMPT_TRANSCRIPT_HEAD_CHARS_FULL = 7000
ANALYSIS_PROMPT_TRANSCRIPT_TAIL_CHARS_FULL = 2800
ANALYSIS_PROMPT_TRANSCRIPT_MAX_CHARS_COMPACT = 6500
ANALYSIS_PROMPT_TRANSCRIPT_HEAD_CHARS_COMPACT = 4600
ANALYSIS_PROMPT_TRANSCRIPT_TAIL_CHARS_COMPACT = 1600
PROMPT_COMPACTION_FILLER_TOKENS = {
    "ага",
    "алло",
    "да",
    "ладно",
    "понятно",
    "спасибо",
    "угу",
    "хорошо",
    "ясно",
}
PROMPT_COMPACTION_COMMITMENT_TOKENS = {"да", "спасибо"}
PROMPT_COMPACTION_REPEAT_RE = re.compile(
    r"\b(?P<token>ага|алло|да|ладно|понятно|спасибо|угу|хорошо|ясно)\b"
    r"(?:[\s,.;:!?-]+(?P=token)\b)+",
    re.I,
)


def _truthy_env_flag(name: str) -> bool:
    value = os.getenv(name)
    return str(value or "").strip().casefold() in TRUE_ENV_VALUES


def _parse_migration_datetime(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"unsupported datetime value: {type(value)!r}")


def build_analysis_migration_call_snapshot(call: CallRecord) -> Dict[str, Any]:
    return {
        "source_call_id": str(getattr(call, "source_call_id", "") or ""),
        "source_recording_id": str(getattr(call, "source_recording_id", "") or ""),
        "source_file": str(call.source_file or ""),
        "source_filename": str(call.source_filename or ""),
        "phone": str(call.phone or ""),
        "manager_name": str(call.manager_name or ""),
        "direction": str(call.direction or ""),
        "started_at": call.started_at.isoformat() if call.started_at is not None else None,
        "duration_sec": call.duration_sec,
        "transcript_text": str(call.transcript_text or ""),
        "transcript_variants_json": str(call.transcript_variants_json or ""),
    }


def migrate_analysis_payload(
    call_snapshot: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    non_conversation_advisory_enabled: Optional[bool] = None,
) -> Dict[str, Any]:
    call = SimpleNamespace(
        source_call_id=str(call_snapshot.get("source_call_id") or ""),
        source_recording_id=str(call_snapshot.get("source_recording_id") or ""),
        source_file=str(call_snapshot.get("source_file") or ""),
        source_filename=str(call_snapshot.get("source_filename") or ""),
        phone=str(call_snapshot.get("phone") or ""),
        manager_name=str(call_snapshot.get("manager_name") or ""),
        direction=str(call_snapshot.get("direction") or ""),
        started_at=_parse_migration_datetime(call_snapshot.get("started_at")),
        duration_sec=call_snapshot.get("duration_sec"),
        transcript_text=str(call_snapshot.get("transcript_text") or ""),
        transcript_variants_json=str(call_snapshot.get("transcript_variants_json") or ""),
    )
    service = AnalyzeService.__new__(AnalyzeService)
    text = call.transcript_text.strip()
    raw = dict(payload) if isinstance(payload, Mapping) else {}
    normalized = service._normalize_analysis(
        call,
        text,
        raw,
        non_conversation_advisory_enabled=non_conversation_advisory_enabled,
    )
    # Migration re-normalizes an OLD stored payload, written before the role
    # guard existed.  Normalization alone would happily re-derive a next step
    # and a person from it, so the guard runs here too: a payload whose sides
    # are not proven can never be revived by being read again, and one whose
    # dialogue no longer parses is the least trustworthy of all.
    return guard_stored_analysis(call_record_view(call), normalized)


# Every stored field that really changes the prompt, the normalization or the
# identity of an exported artefact.  The whole tuple is the stale guard: if any
# of it moved while the model was running, the answer belongs to another call.
ANALYSIS_INPUT_COLUMNS = (
    "source_call_id",
    "source_recording_id",
    "transcript_variants_json",
    "transcript_text",
    "manager_name",
    "phone",
    "direction",
    "started_at",
    "duration_sec",
    "source_filename",
    "source_file",
)


class _StaleAnalysisClaim(RuntimeError):
    """The lease or the input moved: this answer belongs to nobody."""


def analysis_input_snapshot(record: Any) -> Dict[str, Any]:
    """One immutable read of the input, taken before the model is called."""
    if isinstance(record, Mapping):
        return {name: record.get(name) for name in ANALYSIS_INPUT_COLUMNS}
    return {name: getattr(record, name, None) for name in ANALYSIS_INPUT_COLUMNS}


def analysis_input_identity_sha256(record: Any, prompt_identity: Any = None) -> str:
    """Hash of the exact input *and* prompt identity Analyse used.

    Prompt provider, model and version belong here: the same transcript under a
    different prompt is a different analysis, and a stale guard that ignores it
    would happily accept an answer produced by another configuration.
    """
    snapshot = analysis_input_snapshot(record)
    payload = json.dumps(
        {
            "input": {key: _identity_value(value) for key, value in snapshot.items()},
            "prompt": dict(prompt_identity or {}),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return "" if value is None else str(value)


class AnalyzeService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Optional[OpenAI] = None
        self._ollama_client_instance: Optional[OllamaClient] = None
        self._llm_cache = LLMResponseCache(
            enabled=settings.llm_cache_enabled,
            root_dir=settings.llm_cache_dir,
        )
        self._analysis_attempt_context: Optional[Dict[str, Any]] = None

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _analysis_worker_id() -> str:
        return f"an-{uuid.uuid4().hex[:12]}"

    def _analysis_lease_cutoff(self, now: datetime) -> datetime:
        timeout_sec = max(60, int(self._settings.analyze_lease_timeout_sec))
        return now - timedelta(seconds=timeout_sec)

    def _release_stale_claims(self, session: Session, now: datetime) -> int:
        cutoff = self._analysis_lease_cutoff(now)
        scope = require_unique_controlled_call(session, self._settings)
        scope_sql = (
            " AND source_call_id = :controlled_source_call_id" if scope else ""
        )
        params: dict[str, Any] = {"now": now, "cutoff": cutoff}
        if scope:
            params["controlled_source_call_id"] = scope.source_call_id
        result = session.execute(
            text(
                f"""
                UPDATE call_records
                   SET analysis_status = 'pending',
                       analysis_worker_id = NULL,
                       analysis_claimed_at = NULL,
                       updated_at = :now
                 WHERE analysis_status = 'in_progress'
                   AND (
                        analysis_claimed_at IS NULL
                        OR analysis_claimed_at <= :cutoff
                   )
                   {scope_sql}
                """
            ),
            params,
        )
        return int(result.rowcount or 0)

    def _claim_batch(self, session: Session, limit: int, worker_id: str) -> list[int]:
        if limit <= 0:
            return []
        now = self._utc_now()
        max_attempts = max(1, self._settings.analyze_max_attempts)
        release_stale_pipeline_claims(session, self._settings, now)
        self._release_stale_claims(session, now)
        scope = require_unique_controlled_call(session, self._settings)
        scope_sql = (
            " AND source_call_id = :controlled_source_call_id" if scope else ""
        )
        params: dict[str, Any] = {
            "worker_id": worker_id,
            "now": now,
            "max_attempts": max_attempts,
            "limit": int(limit),
        }
        if scope:
            params["controlled_source_call_id"] = scope.source_call_id
        session.execute(
            text(
                f"""
                UPDATE call_records
                   SET analysis_status = 'in_progress',
                       analysis_worker_id = :worker_id,
                       analysis_claimed_at = :now,
                       updated_at = :now
                 WHERE id IN (
                    SELECT id
                      FROM call_records
                     WHERE transcription_status = 'done'
                       AND resolve_status IN ('done', 'skipped', 'manual')
                       AND dead_letter_stage IS NULL
                       AND analysis_status IN ('pending', 'failed')
                       AND analyze_attempts < :max_attempts
                       AND (next_retry_at IS NULL OR next_retry_at <= :now)
                       AND pipeline_stage IS NULL
                       AND pipeline_worker_id IS NULL
                       AND pipeline_claimed_at IS NULL
                       {scope_sql}
                     ORDER BY id ASC
                     LIMIT :limit
                 )
                """
            ),
            params,
        )
        ids = [
            int(row[0])
            for row in session.execute(
                text(
                    f"""
                    SELECT id
                      FROM call_records
                     WHERE analysis_status = 'in_progress'
                       AND analysis_worker_id = :worker_id
                       {scope_sql}
                     ORDER BY id ASC
                    """
                ),
                params,
            ).all()
        ]
        session.commit()
        return ids

    def _retry_delay(self, attempts: int) -> timedelta:
        base = max(1, self._settings.retry_base_delay_sec)
        multiplier = max(1, 2 ** max(0, attempts - 1))
        return timedelta(seconds=base * multiplier)

    def _openai_client(self) -> OpenAI:
        if not self._settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for openai analyze provider")
        if self._client is None:
            # One local reservation must correspond to one HTTP attempt. Hidden
            # SDK retries would make the durable cost ledger under-count calls.
            self._client = OpenAI(api_key=self._settings.openai_api_key, max_retries=0)
        return self._client

    def _ollama_client(self) -> OllamaClient:
        if self._ollama_client_instance is None:
            self._ollama_client_instance = OllamaClient(self._settings.ollama_base_url)
        return self._ollama_client_instance

    @staticmethod
    def _clean_text(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _clean_list(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            text = cls._clean_text(item)
            if text:
                result.append(text)
        return result

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for item in values:
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(item.strip())
        return result

    @staticmethod
    def _nested_dict(payload: Dict[str, Any], key: str) -> Dict[str, Any]:
        value = payload.get(key)
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _coerce_score(value: Any) -> Optional[int]:
        try:
            score = int(value)
        except (TypeError, ValueError):
            return None
        if score < 0:
            return 0
        if score > 100:
            return 100
        return score

    @staticmethod
    def _priority_from_score(score: int) -> str:
        if score >= 75:
            return "hot"
        if score >= 45:
            return "warm"
        return "cold"

    def _detect_from_patterns(self, text: str, patterns: Dict[str, re.Pattern[str]]) -> list[str]:
        lowered = text.lower()
        detected: list[str] = []
        for label, pattern in patterns.items():
            if pattern.search(lowered):
                detected.append(label)
        return detected

    def _detect_preferred_channel(self, text: str) -> Optional[str]:
        lowered = text.lower()
        if "телеграм" in lowered or "telegram" in lowered:
            return "telegram"
        if "ватсап" in lowered or "whatsapp" in lowered or "вотсап" in lowered:
            return "whatsapp"
        if "почт" in lowered or "email" in lowered or "e-mail" in lowered:
            return "email"
        if "сайт" in lowered or "заявк" in lowered:
            return "site"
        return None

    @staticmethod
    def _extract_email(text: str) -> Optional[str]:
        match = EMAIL_RE.search(text or "")
        if not match:
            return None
        return match.group(0).lower()

    @staticmethod
    def _extract_grade(text: str) -> Optional[str]:
        match = GRADE_RE.search(text or "")
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _extract_evidence(text: str, limit: int = 3) -> list[Dict[str, str]]:
        evidence: list[Dict[str, str]] = []
        for line in (text or "").splitlines():
            match = SPEAKER_LINE_RE.match(line.strip())
            if not match:
                continue
            snippet = (match.group("text") or "").strip()
            if len(snippet) < 12:
                continue
            evidence.append(
                {
                    "speaker": (match.group("speaker") or "").strip(),
                    "ts": (match.group("ts") or "").strip(),
                    "text": snippet[:260],
                }
            )
            if len(evidence) >= limit:
                break
        return evidence

    @staticmethod
    def _parse_object_candidate(text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        try:
            payload = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _extract_json_payload(text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            raise RuntimeError("empty response")
        payload = AnalyzeService._parse_object_candidate(raw)
        if payload is not None:
            return payload

        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        if fence:
            payload = AnalyzeService._parse_object_candidate(fence.group(1))
            if payload is not None:
                return payload

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            payload = AnalyzeService._parse_object_candidate(raw[start : end + 1])
            if payload is not None:
                return payload
        raise RuntimeError("response does not contain JSON object")

    @staticmethod
    def _format_started_at(started_at: Optional[datetime]) -> Optional[str]:
        """Always Moscow, always once: naive stored values are UTC, not local.

        The old branch converted only aware values and left a naive one as-is,
        so the same call was shown in UTC in the database-fed paths and in the
        machine's local zone elsewhere.  There is one contract now, shared with
        the Google publisher.
        """
        moscow = moscow_datetime(started_at)
        return None if moscow is None else moscow.strftime("%d.%m.%Y %H:%M")

    @staticmethod
    def _sentence(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return None
        if compact[-1] not in ".!?":
            compact = f"{compact}."
        return compact

    @staticmethod
    def _looks_like_dialogue_dump(text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        if DIALOGUE_DUMP_LINE_RE.search(raw):
            return True
        if ROLE_PREFIX_RE.search(raw):
            return True
        line_count = len([line for line in raw.splitlines() if line.strip()])
        lowered = raw.lower()
        if line_count >= 3 and (
            "клиент:" in lowered
            or "менеджер" in lowered
            or "manager:" in lowered
            or "client:" in lowered
        ):
            return True
        return False

    @staticmethod
    def _normalize_next_step_action(value: Optional[str]) -> Optional[str]:
        text = (value or "").strip()
        if not text:
            return None
        lowered = text.lower()
        mappings = (
            (r"call back|callback|follow[\s-]?up", "Перезвонить клиенту"),
            (r"wait for decision", "Дождаться решения клиента"),
            (r"schedule|arrange", "Согласовать следующий контакт"),
        )
        for pattern, replacement in mappings:
            if re.search(pattern, lowered):
                return replacement
        if re.search(r"ссылк\w*.*оплат|оплат\w*.*ссылк", lowered):
            return "Отправить ссылку на оплату"
        if re.search(r"(telegram|whatsapp|email|e-mail|телеграм|ватсап|вотсап|почт)", lowered) and re.search(
            r"(написа\w*|отправ\w*|вышл\w*|пришл\w*|направ\w*|скин\w*)",
            lowered,
        ):
            return "Отправить материалы"
        return text

    def _analysis_prompt_profile(self, override: Optional[str] = None) -> str:
        profile = (override or self._settings.analyze_prompt_profile or "").strip().lower()
        if profile not in {"compact", "full"}:
            return "compact"
        return profile

    def _analysis_system_prompt(self, profile: Optional[str] = None) -> str:
        normalized = self._analysis_prompt_profile(profile)
        if normalized == "full":
            return SYSTEM_PROMPT_FULL
        return SYSTEM_PROMPT_COMPACT

    def _analysis_prompt_version(self, profile: Optional[str] = None) -> str:
        normalized = self._analysis_prompt_profile(profile)
        if normalized == "full":
            return ANALYZE_PROMPT_VERSION_FULL
        return ANALYZE_PROMPT_VERSION_COMPACT

    def _analysis_prompt_limits(self, profile: Optional[str] = None) -> tuple[int, int, int]:
        normalized = self._analysis_prompt_profile(profile)
        if normalized == "full":
            return (
                ANALYSIS_PROMPT_TRANSCRIPT_MAX_CHARS_FULL,
                ANALYSIS_PROMPT_TRANSCRIPT_HEAD_CHARS_FULL,
                ANALYSIS_PROMPT_TRANSCRIPT_TAIL_CHARS_FULL,
            )
        return (
            ANALYSIS_PROMPT_TRANSCRIPT_MAX_CHARS_COMPACT,
            ANALYSIS_PROMPT_TRANSCRIPT_HEAD_CHARS_COMPACT,
            ANALYSIS_PROMPT_TRANSCRIPT_TAIL_CHARS_COMPACT,
        )

    @staticmethod
    def _compact_prompt_filler_body(text: str) -> str:
        compact = re.sub(r"\s+", " ", text or "").strip()
        if not compact:
            return ""
        previous = None
        while previous != compact:
            previous = compact
            compact = PROMPT_COMPACTION_REPEAT_RE.sub(lambda match: match.group("token"), compact)
            compact = re.sub(r"\s+([,.;:!?])", r"\1", compact)
            compact = re.sub(r"([,.;:!?])(?=[^\s])", r"\1 ", compact)
            compact = re.sub(r"\s+", " ", compact).strip(" ,")
        return compact

    @staticmethod
    def _filler_only_signature(text: str) -> Optional[str]:
        lowered = (text or "").lower()
        tokens = re.findall(r"[a-zа-яё0-9]+", lowered, flags=re.I)
        if not tokens:
            return None
        if not all(token in PROMPT_COMPACTION_FILLER_TOKENS for token in tokens):
            return None
        if len(tokens) == 1:
            return None
        if any(token in PROMPT_COMPACTION_COMMITMENT_TOKENS for token in tokens):
            return None
        return " ".join(tokens)

    @staticmethod
    def _prompt_speaker_label(speaker: str) -> str:
        lowered = (speaker or "").strip().lower()
        if "менедж" in lowered or "manager" in lowered:
            return "Менеджер"
        if "клиент" in lowered or "client" in lowered:
            return "Клиент"
        return "Спикер"

    def _compact_transcript_for_prompt(
        self,
        text: str,
        profile: Optional[str] = None,
        *,
        apply_compaction: Optional[bool] = None,
    ) -> Dict[str, Any]:
        normalized = self._analysis_prompt_profile(profile)
        original = text or ""
        use_compaction = (
            self._settings.analyze_transcript_compaction_enabled
            if apply_compaction is None
            else bool(apply_compaction)
        )
        compacted = original
        shortened_lines = 0
        deduped_lines = 0
        removed_lines = 0
        timestamp_removed_lines = 0

        if use_compaction and original:
            compacted_lines: list[str] = []
            prev_filler_signature: Optional[str] = None
            prev_speaker: Optional[str] = None
            for raw_line in original.splitlines():
                stripped = raw_line.strip()
                if not stripped:
                    if compacted_lines and compacted_lines[-1] != "":
                        compacted_lines.append("")
                    prev_filler_signature = None
                    prev_speaker = None
                    continue

                match = SPEAKER_LINE_RE.match(stripped)
                speaker = None
                body = stripped
                prefix = ""
                if match:
                    speaker = self._prompt_speaker_label(self._clean_text(match.group("speaker")) or "")
                    body = (match.group("text") or "").strip()
                    prefix = f"{speaker}: "
                    timestamp_removed_lines += 1
                compact_body = self._compact_prompt_filler_body(body)
                if compact_body != body:
                    shortened_lines += 1
                filler_signature = self._filler_only_signature(compact_body)
                if (
                    filler_signature
                    and filler_signature == prev_filler_signature
                    and (speaker or "") == (prev_speaker or "")
                ):
                    deduped_lines += 1
                    removed_lines += 1
                    continue
                rendered = f"{prefix}{compact_body}".strip()
                if rendered:
                    compacted_lines.append(rendered)
                prev_filler_signature = filler_signature
                prev_speaker = speaker

            compacted = "\n".join(compacted_lines).strip()
            if not compacted:
                compacted = original

        compacted = re.sub(r"[ \t]+", " ", compacted)
        prompt_transcript = compacted
        max_chars, head_chars, tail_chars = self._analysis_prompt_limits(normalized)
        truncated = False
        if len(prompt_transcript) > max_chars:
            head = prompt_transcript[:head_chars].rstrip()
            tail = prompt_transcript[-tail_chars:].lstrip()
            prompt_transcript = (
                f"{head}\n\n"
                "[... transcript truncated for prompt budget ...]\n\n"
                f"{tail}"
            )
            truncated = True

        chars_original = len(original)
        chars_compacted = len(compacted)
        chars_prompt = len(prompt_transcript)
        return {
            "profile": normalized,
            "transcript": prompt_transcript,
            "transcript_chars_original": chars_original,
            "transcript_chars_compacted": chars_compacted,
            "transcript_chars_prompt": chars_prompt,
            "transcript_chars_saved": max(0, chars_original - chars_prompt),
            "transcript_compacted": bool(use_compaction and chars_compacted < chars_original),
            "transcript_truncated": truncated,
            "transcript_compaction_removed_lines": removed_lines,
            "transcript_compaction_shortened_lines": shortened_lines,
            "transcript_compaction_deduped_lines": deduped_lines,
            "transcript_prompt_timestamps_removed_lines": timestamp_removed_lines,
        }

    @staticmethod
    def _with_analysis_prompt_quality_flags(
        payload: Dict[str, Any],
        *,
        metrics: Dict[str, Any],
        prompt_version: str,
        cache_hit: bool,
        token_usage: Optional[Mapping[str, Any]] = None,
        provider_attempts: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        merged = dict(payload) if isinstance(payload, dict) else {}
        raw_quality = merged.get("quality_flags")
        quality_flags = dict(raw_quality) if isinstance(raw_quality, dict) else {}
        quality_flags.update(
            {
                "analyze_prompt_profile": metrics.get("profile"),
                "analyze_prompt_version": prompt_version,
                "analyze_prompt_compacted": bool(metrics.get("transcript_compacted")),
                "analyze_prompt_truncated": bool(metrics.get("transcript_truncated")),
                "analyze_llm_cache_hit": bool(cache_hit),
                "analyze_transcript_chars_original": int(metrics.get("transcript_chars_original", 0) or 0),
                "analyze_transcript_chars_compacted": int(metrics.get("transcript_chars_compacted", 0) or 0),
                "analyze_transcript_chars_prompt": int(metrics.get("transcript_chars_prompt", 0) or 0),
                "analyze_transcript_chars_saved": int(metrics.get("transcript_chars_saved", 0) or 0),
                "analyze_prompt_compaction_removed_lines": int(
                    metrics.get("transcript_compaction_removed_lines", 0) or 0
                ),
                "analyze_prompt_compaction_shortened_lines": int(
                    metrics.get("transcript_compaction_shortened_lines", 0) or 0
                ),
                "analyze_prompt_compaction_deduped_lines": int(
                    metrics.get("transcript_compaction_deduped_lines", 0) or 0
                ),
                "analyze_prompt_timestamps_removed_lines": int(
                    metrics.get("transcript_prompt_timestamps_removed_lines", 0) or 0
                ),
            }
        )
        for key in ANALYSIS_PROMPT_PASSTHROUGH_FLAGS:
            if metrics.get(key) is not None:
                quality_flags[key] = metrics[key]
        # A cached answer costs nothing now, whatever the original call cost, so
        # the stored counters of that first call are replaced rather than
        # re-reported as if they had been spent again.
        effective_usage = dict(
            CACHE_HIT_TOKEN_USAGE if cache_hit else (token_usage or UNAVAILABLE_TOKEN_USAGE)
        )
        quality_flags["analyze_token_usage"] = effective_usage
        if cache_hit:
            quality_flags["analyze_provider_attempts"] = []
        else:
            attempts = provider_attempts or ({"token_usage": effective_usage},)
            quality_flags["analyze_provider_attempts"] = [
                {
                    **{
                        key: attempt.get(key)
                        for key in (
                            "attempt_id", "stage", "state", "analysis_source_sha256",
                            "provider", "model", "profile", "prompt_version",
                            "cache_hit", "model_called",
                        )
                        if attempt.get(key) is not None
                    },
                    "token_usage": dict(
                        attempt.get("token_usage")
                        if isinstance(attempt.get("token_usage"), Mapping)
                        else UNAVAILABLE_TOKEN_USAGE
                    )
                }
                for attempt in attempts
                if isinstance(attempt, Mapping)
            ]
        merged["quality_flags"] = quality_flags
        return merged

    @classmethod
    def _prune_prompt_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        pruned: Dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, dict):
                nested = cls._prune_prompt_payload(value)
                if nested:
                    pruned[key] = nested
                continue
            if isinstance(value, list):
                cleaned_items = []
                for item in value:
                    if isinstance(item, dict):
                        nested = cls._prune_prompt_payload(item)
                        if nested:
                            cleaned_items.append(nested)
                    elif item not in (None, "", False):
                        cleaned_items.append(item)
                if cleaned_items:
                    pruned[key] = cleaned_items
                continue
            if value in (None, "", False):
                continue
            pruned[key] = value
        return pruned

    def _dialogue_prompt_metrics(
        self, dialogue: DialogueInput, profile: str
    ) -> Dict[str, Any]:
        """Whole-turn prompt projection with prompt-only filler compaction."""
        max_chars, _head, _tail = self._analysis_prompt_limits(profile)
        raw_full = dialogue.render_for_analysis()
        shortened_turns = 0
        prompt_turns = []
        for turn in dialogue.turns:
            raw_text = str(turn.get("text") or "")
            compact_text = (
                self._compact_prompt_filler_body(raw_text)
                if self._settings.analyze_transcript_compaction_enabled
                else raw_text
            )
            if compact_text and compact_text != raw_text:
                shortened_turns += 1
            prompt_turns.append({**turn, "text": compact_text or raw_text})
        prompt_dialogue = DialogueInput(
            version=dialogue.version,
            source=dialogue.source,
            role_attribution=dialogue.role_attribution,
            turns=tuple(prompt_turns),
            warnings=dialogue.warnings,
            canonical_sha256=dialogue.canonical_sha256,
        )
        full = prompt_dialogue.render_for_analysis()
        selection = prompt_dialogue.render_for_analysis(max_chars=max_chars)
        return {
            "profile": profile,
            "transcript": selection["text"],
            "transcript_chars_original": len(raw_full["text"]),
            "transcript_chars_compacted": len(full["text"]),
            "transcript_chars_prompt": len(selection["text"]),
            "transcript_chars_saved": max(
                0, len(raw_full["text"]) - len(selection["text"])
            ),
            "transcript_compacted": shortened_turns > 0,
            "transcript_truncated": bool(selection["truncated"]),
            "transcript_compaction_removed_lines": 0,
            "transcript_compaction_shortened_lines": shortened_turns,
            "transcript_compaction_deduped_lines": 0,
            "transcript_prompt_timestamps_removed_lines": 0,
            "dialogue_version": dialogue.version,
            "dialogue_source": dialogue.source,
            "dialogue_canonical_sha256": dialogue.canonical_sha256,
            "dialogue_selected_turn_ids": list(selection["selected_turn_ids"]),
            "dialogue_selected_turn_count": int(selection["selected_turn_count"]),
            "dialogue_total_turn_count": int(selection["total_turn_count"]),
        }

    def _analysis_prompt_context(
        self,
        call: CallRecord,
        text: str,
        profile: Optional[str] = None,
        dialogue: Optional[DialogueInput] = None,
    ) -> Dict[str, Any]:
        normalized = self._analysis_prompt_profile(profile)
        started_at = self._format_started_at(call.started_at) or "unknown"
        manager = self._clean_text(call.manager_name) or "unknown"
        phone = self._clean_text(call.phone) or "unknown"
        direction = self._clean_text(call.direction) or "unknown"
        transcript_meta = (
            self._dialogue_prompt_metrics(dialogue, normalized)
            if dialogue is not None and dialogue.turns
            else self._compact_transcript_for_prompt(text, normalized)
        )
        metadata_payload = {
            "source_filename": call.source_filename,
            "started_at": started_at,
            "manager_name": manager,
            "client_phone": phone,
            "direction": direction,
        }
        prompt = (
            "Analyze this call transcript and return JSON only.\n"
            "Call metadata JSON:\n"
            f"{json.dumps(metadata_payload, ensure_ascii=False, separators=(',', ':'))}\n"
        )
        if normalized in {"compact", "full"}:
            hints_payload = self._prune_prompt_payload(self._analysis_rule_hints(call, text))
            prompt += (
                "\nDeterministic hints JSON (may be incomplete; use only if supported by transcript):\n"
                f"{json.dumps(hints_payload, ensure_ascii=False, separators=(',', ':'))}\n"
            )
        label = "Dialogue" if transcript_meta.get("dialogue_version") else "Transcript"
        prompt += "\n" f"{label}:\n{transcript_meta['transcript']}"
        system_prompt = self._analysis_system_prompt(normalized)
        llm_prompt = f"{system_prompt}\n\n{prompt}"
        metrics = dict(transcript_meta)
        metrics["analysis_prompt_sha256"] = hashlib.sha256(
            llm_prompt.encode("utf-8")
        ).hexdigest()
        return {
            "profile": normalized,
            "system_prompt": system_prompt,
            "user_prompt": prompt,
            "llm_prompt": llm_prompt,
            "metrics": metrics,
        }

    def _candidate_next_step_action(self, text: str) -> Optional[str]:
        lowered = (text or "").lower()
        signals = detect_non_conversation_signals(transcript_text=text)
        if signals.should_force_non_conversation or (
            signals.strong_no_live_marker and not signals.protected_live_dialogue and signals.score <= 1
        ):
            return None
        if "перезвон" in lowered or "созвон" in lowered or "позвон" in lowered:
            return "Перезвонить клиенту"
        if re.search(r"ссылк\w*.*оплат|оплат\w*.*ссылк", lowered):
            return "Отправить ссылку на оплату"
        if "отправ" in lowered:
            return "Отправить материалы"
        if "уточн" in lowered:
            return "Уточнить информацию и сообщить клиенту"
        return None

    @staticmethod
    def _has_explicit_sales_signal(
        *,
        raw_sales_signal: bool,
        products: Optional[list[str]] = None,
        formats: Optional[list[str]] = None,
        exam_targets: Optional[list[str]] = None,
    ) -> bool:
        return bool(raw_sales_signal or products or formats or exam_targets)

    def _analysis_rule_hints(self, call: CallRecord, text: str) -> Dict[str, Any]:
        hints: Dict[str, Any] = {
            "target_product_candidates": self._detect_from_patterns(text, PRODUCT_PATTERNS),
            "subject_candidates": self._detect_from_patterns(text, SUBJECT_PATTERNS),
            "format_candidates": self._detect_from_patterns(text, FORMAT_PATTERNS),
            "exam_target_candidates": self._detect_from_patterns(text, EXAM_PATTERNS),
            "objection_candidates": self._detect_from_patterns(text, OBJECTION_PATTERNS),
            "grade_candidate": self._extract_grade(text),
            "email_candidate": self._extract_email(text),
            "preferred_channel_candidate": self._detect_preferred_channel(text),
            "next_step_candidate": self._candidate_next_step_action(text),
            "call_type_candidate": self._detect_call_type(text),
            "non_conversation_candidate": self._is_non_conversation(text),
            "phone_from_filename": self._clean_text(call.phone),
        }
        return hints

    def _analysis_cache_lookup(
        self,
        *,
        provider: str,
        model: str,
        reasoning: str,
        prompt_version: str,
        prompt: str,
    ) -> Optional[Dict[str, Any]]:
        cached = self._llm_cache.get(
            namespace="analyze",
            provider=provider,
            model=model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
        )
        if cached is None:
            return None
        try:
            return validate_v3_model_response(cached)
        except AnalysisContractError:
            # Old or poisoned cache entries are ordinary misses.  They cannot
            # lock a call into repeating a response that the current contract
            # would reject before publication.
            return None

    def _analysis_cache_store(
        self,
        *,
        provider: str,
        model: str,
        reasoning: str,
        prompt_version: str,
        prompt: str,
        response: Dict[str, Any],
    ) -> None:
        validated = validate_v3_model_response(response)
        self._llm_cache.put(
            namespace="analyze",
            provider=provider,
            model=model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
            response=validated,
        )

    def _cache_analysis_after_commit(self, **entry: Any) -> None:
        """Production answers enter cache only after their DB result is durable."""
        context = self._analysis_attempt_context
        if context is None:
            self._analysis_cache_store(**entry)
            return
        context.setdefault("cache_writes", []).append(dict(entry))

    def _flush_analysis_cache_writes(self) -> int:
        context = self._analysis_attempt_context or {}
        failed = 0
        for entry in context.get("cache_writes", []):
            try:
                self._analysis_cache_store(**entry)
            except Exception:  # cache is an optimization, not the source of truth
                failed += 1
        context["cache_writes"] = []
        return failed

    def _should_escalate_full_profile(self, text: str, raw_analysis: Dict[str, Any]) -> bool:
        if not self._settings.analyze_escalate_full_on_ambiguity:
            return False
        if self._analysis_prompt_profile() != "compact":
            return False
        raw = raw_analysis if isinstance(raw_analysis, dict) else {}
        blocks = self._nested_dict(raw, "structured_fields") or self._nested_dict(raw, "crm_blocks")
        summary = self._clean_text(raw.get("history_summary")) or self._clean_text(raw.get("summary"))
        # v3 answers carry no summary at all, so an empty one only means "the
        # model told us nothing" when it also filled no structured field.
        if not blocks and (not summary or self._looks_like_dialogue_dump(summary)):
            return True
        interests = self._nested_dict(blocks, "interests")
        next_step_block = self._nested_dict(blocks, "next_step")
        llm_subjects = self._clean_list(interests.get("subjects"))
        llm_products = self._clean_list(interests.get("products"))
        llm_formats = self._clean_list(interests.get("format"))
        llm_exam_targets = self._clean_list(interests.get("exam_targets"))
        llm_next_step = self._clean_text(next_step_block.get("action")) or self._clean_text(raw.get("next_step"))
        llm_target_product = self._clean_text(raw.get("target_product"))
        heuristic_subjects = self._detect_from_patterns(text, SUBJECT_PATTERNS)
        heuristic_products = self._detect_from_patterns(text, PRODUCT_PATTERNS)
        heuristic_formats = self._detect_from_patterns(text, FORMAT_PATTERNS)
        heuristic_exam_targets = self._detect_from_patterns(text, EXAM_PATTERNS)
        heuristic_next_step = self._candidate_next_step_action(text)
        tags = [str(item).strip().lower() for item in self._clean_list(raw.get("tags"))]
        heuristic_call_type = self._detect_call_type(
            text,
            products=heuristic_products,
            subjects=heuristic_subjects,
            formats=heuristic_formats,
            exam_targets=heuristic_exam_targets,
            next_step_action=heuristic_next_step,
        )
        llm_call_type = self._detect_call_type(
            text,
            tags=tags,
            products=llm_products + ([llm_target_product] if llm_target_product else []),
            subjects=llm_subjects,
            formats=llm_formats,
            exam_targets=llm_exam_targets,
            next_step_action=llm_next_step,
        )
        if heuristic_products and not (llm_target_product or llm_products):
            return True
        if heuristic_subjects and not llm_subjects:
            return True
        if heuristic_next_step and not llm_next_step:
            return True
        if "non_conversation" in tags and heuristic_call_type != "non_conversation":
            return True
        # The v3 contract has no ``tags``; a model that wants to call a real
        # conversation empty says so through ``result.status``, and that claim
        # gets the same second opinion the old tag did.
        if (
            self._clean_text(self._nested_dict(blocks, "result").get("status"))
            == "non_conversation"
            and heuristic_call_type != "non_conversation"
        ):
            return True
        if llm_call_type == "non_conversation" and heuristic_call_type != "non_conversation":
            return True
        return False

    def _has_meaningful_sales_signal(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        lowered = raw.lower()
        products = self._detect_from_patterns(raw, PRODUCT_PATTERNS)
        subjects = self._detect_from_patterns(raw, SUBJECT_PATTERNS)
        formats = self._detect_from_patterns(raw, FORMAT_PATTERNS)
        exam_targets = self._detect_from_patterns(raw, EXAM_PATTERNS)
        lead_interest = bool(
            re.search(
                r"интерес\w*|хотел\w*|хочу|ищ\w*|рассматрива\w*|подобрат\w*|"
                r"запис\w*|узнат\w*|подход\w*|выбрат\w*",
                lowered,
            )
        )
        needs_training = bool(
            re.search(
                r"нуж(?:ен|на|ны)\s+(?:курс\w*|лагер\w*|школ\w*|обучен\w*|заняти\w*|подготов\w*|программ\w*)",
                lowered,
            )
        )
        training_noun = bool(
            re.search(r"курс\w*|лагер\w*|школ\w*|обучен\w*|заняти\w*|подготов\w*|программ\w*", lowered)
        )
        existing_client_context = self._matches_any_pattern(raw, TECHNICAL_CALL_PATTERNS) or self._matches_any_pattern(
            raw, SERVICE_CALL_PATTERNS
        ) or self._matches_any_pattern(raw, EXISTING_CLIENT_PROGRESS_PATTERNS)
        if products:
            return True
        if existing_client_context and not lead_interest and not needs_training:
            return False
        if (lead_interest or needs_training) and (subjects or formats or exam_targets or training_noun):
            return True
        if subjects and formats and training_noun:
            return True
        if exam_targets and (subjects or training_noun or lead_interest):
            return True
        if subjects and self._extract_grade(raw) and (lead_interest or needs_training):
            return True
        return False

    @staticmethod
    def _matches_any_pattern(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
        raw = text or ""
        return any(pattern.search(raw) for pattern in patterns)

    def _has_substantial_dialogue(self, text: str) -> bool:
        raw = (text or "").strip()
        if not raw:
            return False
        word_count = len(re.findall(r"\w+", raw, re.U))
        speaker_markers = len(re.findall(r"^\s*(manager|client|менеджер|клиент)\s*:", raw, re.I | re.M))
        line_count = len([line for line in raw.splitlines() if line.strip()])
        return word_count >= 18 and (speaker_markers >= 1 or line_count >= 3)

    def _detect_call_type(
        self,
        text: str,
        *,
        tags: Optional[list[str]] = None,
        products: Optional[list[str]] = None,
        subjects: Optional[list[str]] = None,
        formats: Optional[list[str]] = None,
        exam_targets: Optional[list[str]] = None,
        next_step_action: Optional[str] = None,
    ) -> str:
        raw = (text or "").strip()
        if not raw:
            return "non_conversation"
        signals = detect_non_conversation_signals(transcript_text=raw)
        if signals.should_force_non_conversation:
            return "non_conversation"
        if signals.label == "manual_review_borderline_live_context" and any(
            str(reason).startswith("safeguard_") for reason in signals.reason_codes
        ):
            return "service_call"
        lowered = raw.lower()
        semantic_tags = {str(item).strip().lower() for item in (tags or []) if str(item).strip()}
        raw_sales_signal = self._has_meaningful_sales_signal(raw)
        explicit_sales_signal = self._has_explicit_sales_signal(
            raw_sales_signal=raw_sales_signal,
            products=products,
            formats=formats,
            exam_targets=exam_targets,
        )
        meaningful_dialogue = self._has_substantial_dialogue(raw)
        technical_signal = self._matches_any_pattern(raw, TECHNICAL_CALL_PATTERNS)
        service_signal = self._matches_any_pattern(raw, SERVICE_CALL_PATTERNS)
        progress_signal = self._matches_any_pattern(raw, EXISTING_CLIENT_PROGRESS_PATTERNS)
        has_followup = bool(self._clean_text(next_step_action))
        has_business_content = explicit_sales_signal or technical_signal or service_signal or progress_signal or (
            has_followup and meaningful_dialogue
        )

        if any(marker in lowered for marker in STRONG_NON_CONVERSATION_MARKERS) and not has_business_content:
            return "non_conversation"
        if len(raw) <= 40 and not meaningful_dialogue and not has_business_content:
            return "non_conversation"

        if "existing_client_progress" in semantic_tags and not explicit_sales_signal:
            return "existing_client_progress"
        if "technical_call" in semantic_tags and not explicit_sales_signal:
            return "technical_call"
        if "service_call" in semantic_tags and not explicit_sales_signal:
            return "service_call"

        if progress_signal and not explicit_sales_signal:
            return "existing_client_progress"
        if technical_signal and not explicit_sales_signal:
            return "technical_call"
        if service_signal and not explicit_sales_signal:
            return "service_call"

        if "non_conversation" in semantic_tags and not raw_sales_signal and not meaningful_dialogue:
            return "non_conversation"
        if explicit_sales_signal:
            return "sales_call"
        if meaningful_dialogue:
            if technical_signal:
                return "technical_call"
            if progress_signal:
                return "existing_client_progress"
            return "service_call"
        if any(marker in lowered for marker in WEAK_NON_CONVERSATION_MARKERS):
            return "non_conversation"
        if len(re.findall(r"\w+", raw, re.U)) < 12:
            return "non_conversation"
        return "service_call"

    def _non_conversation_advisory_call_type(
        self,
        text: str,
        *,
        tags: list[str],
        products: list[str],
        formats: list[str],
        exam_targets: list[str],
        next_step_action: Optional[str],
    ) -> str:
        lowered_tags = {str(item).strip().lower() for item in tags if str(item).strip()}
        if "existing_client_progress" in lowered_tags or self._matches_any_pattern(text, EXISTING_CLIENT_PROGRESS_PATTERNS):
            return "existing_client_progress"
        if "technical_call" in lowered_tags or self._matches_any_pattern(text, TECHNICAL_CALL_PATTERNS):
            return "technical_call"
        if "service_call" in lowered_tags or self._matches_any_pattern(text, SERVICE_CALL_PATTERNS):
            return "service_call"
        if self._has_explicit_sales_signal(
            raw_sales_signal=self._has_meaningful_sales_signal(text),
            products=products,
            formats=formats,
            exam_targets=exam_targets,
        ):
            return "sales_call"
        if self._clean_text(next_step_action) or self._has_substantial_dialogue(text):
            return "service_call"
        return "non_conversation"

    def _build_review_flags(
        self,
        call: CallRecord,
        *,
        text: str,
        call_type: str,
        products: list[str],
        formats: list[str],
        exam_targets: list[str],
        target_product: Optional[str],
        next_step_action: Optional[str],
        history_summary: Optional[str],
    ) -> Dict[str, Any]:
        reasons: list[str] = []
        product_present = bool(products or self._clean_text(target_product))
        summary_lower = (history_summary or "").lower()
        technical_signal = self._matches_any_pattern(text, TECHNICAL_CALL_PATTERNS)
        service_signal = self._matches_any_pattern(text, SERVICE_CALL_PATTERNS)
        progress_signal = self._matches_any_pattern(text, EXISTING_CLIENT_PROGRESS_PATTERNS)
        explicit_sales_signal = self._has_explicit_sales_signal(
            raw_sales_signal=self._has_meaningful_sales_signal(text),
            products=products,
            formats=formats,
            exam_targets=exam_targets,
        )

        if call_type == "sales_call":
            if not product_present and not next_step_action:
                reasons.append("sales_missing_product_and_next_step")
            elif not product_present:
                reasons.append("sales_missing_product")
            elif not next_step_action:
                reasons.append("sales_missing_next_step")
            if (technical_signal or service_signal or progress_signal) and not product_present:
                reasons.append("sales_service_overlap")

        if call_type == "non_conversation" and float(call.duration_sec or 0.0) >= 30:
            reasons.append("long_non_conversation")

        if call_type != "non_conversation" and (
            "нецелевой звонок" in summary_lower or "автоответчик/короткий технический дозвон" in summary_lower
        ):
            reasons.append("legacy_summary_conflict")

        if (
            call_type in {"service_call", "technical_call", "existing_client_progress"}
            and explicit_sales_signal
            and not next_step_action
        ):
            reasons.append("non_sales_with_sales_signal")

        if getattr(call, "resolve_status", None) == "manual":
            reasons.append("resolve_manual_review_required")
        if self._quality_flags_from_call(call).get("secondary_backfill_exhausted"):
            reasons.append("secondary_asr_exhausted_primary_fallback")

        return {
            "needs_review": bool(reasons),
            "review_reasons": self._unique(reasons),
        }

    def _transcript_quality_guardrails(
        self,
        call: CallRecord,
        *,
        text: str,
        history_summary: Optional[str],
        call_type: str,
        products: list[str],
        subjects: list[str],
        objections: list[str],
        next_step_action: Optional[str],
    ) -> Dict[str, Any]:
        signals = detect_non_conversation_signals(
            transcript_text=text,
            history_summary=history_summary or "",
            call_type=call_type,
            next_step=next_step_action or "",
            products=products,
            subjects=subjects,
            objections=objections,
            duration_sec=getattr(call, "duration_sec", None),
        )
        return {
            "version": TRANSCRIPT_QUALITY_GUARDRAILS_VERSION,
            "mode": "dry_run",
            "label": signals.label,
            "score": signals.score,
            "reason_codes": list(signals.reason_codes),
            "strong_no_live_marker": signals.strong_no_live_marker,
            "asr_artifact_marker": signals.asr_artifact_marker,
            "system_no_dialogue_phrase": signals.system_no_dialogue_phrase,
            "risky_keyword_marker": signals.risky_keyword_marker,
            "outbound_voicemail_marker": signals.outbound_voicemail_marker,
            "protected_live_dialogue": signals.protected_live_dialogue,
            "should_force_non_conversation": signals.should_force_non_conversation,
            "requires_manual_review": signals.requires_manual_review,
            "recommended_call_type": signals.recommended_call_type,
            "recommended_contentful": signals.recommended_contentful,
            "recommended_contact_subtype": signals.recommended_contact_subtype,
            "manager_chars": signals.manager_chars,
            "client_chars": signals.client_chars,
            "transcript_chars": signals.transcript_chars,
        }

    def _non_conversation_summary(self, call: CallRecord, *, contact_subtype: Optional[str] = None) -> str:
        # Same rule as the ordinary конспект: the date and the manager belong to
        # their own columns, so this sentence says only what happened.
        _ = call
        reason = "автоответчик, IVR, голосовой ассистент или технический недозвон"
        if contact_subtype == "outbound_voicemail":
            reason = "менеджер оставил сообщение на автоответчике, живого диалога с клиентом не было"
        return f"Содержательного диалога не было: {reason}."

    def _apply_non_conversation_hard_validation(
        self,
        call: CallRecord,
        normalized: Dict[str, Any],
    ) -> Dict[str, Any]:
        quality_flags = normalized.get("quality_flags")
        if not isinstance(quality_flags, dict):
            quality_flags = {}
        if quality_flags.get("non_conversation_advisory"):
            return normalized
        if quality_flags.get("call_type") != "non_conversation":
            return normalized

        existing_structured = (
            normalized.get("structured_fields") if isinstance(normalized.get("structured_fields"), dict) else {}
        )
        existing_contacts = (
            existing_structured.get("contacts") if isinstance(existing_structured.get("contacts"), dict) else {}
        )
        phone_from_filename = self._clean_text(existing_contacts.get("phone_from_filename")) or self._clean_text(
            call.phone
        )

        contact_subtype = self._clean_text(quality_flags.get("transcript_quality_recommended_contact_subtype"))
        summary = self._non_conversation_summary(call, contact_subtype=contact_subtype)
        structured_fields = {
            # The technical category already lives in ``quality_flags.call_type``.
            # Publishing it a second time as a business outcome would need its
            # own reply as evidence, and there is no conversation to quote.
            "result": {"status": None, "detail": None},
            "people": {
                "parent_fio": None,
                "child_fio": None,
            },
            "contacts": {
                "email": None,
                "phone_from_filename": phone_from_filename,
                "preferred_channel": None,
            },
            "student": {
                "grade_current": None,
                "school": None,
            },
            "interests": {
                "products": [],
                "format": [],
                "subjects": [],
                "exam_targets": [],
            },
            "commercial": {
                "price_sensitivity": None,
                "budget": None,
                "discount_interest": None,
            },
            "objections": [],
            "next_step": {
                "action": None,
                "due": None,
            },
            "lead_priority": "cold",
        }

        quality_flags["call_type"] = "non_conversation"
        quality_flags["non_conversation_hard_validation_applied"] = True
        normalized.update(
            {
                "history_summary": summary,
                "structured_fields": structured_fields,
                "history_short": summary,
                "crm_blocks": structured_fields,
                "summary": "Нет содержательного диалога менеджер-клиент для анализа продаж.",
                "interests": [],
                "student_grade": None,
                "target_product": None,
                "personal_offer": None,
                "pain_points": [],
                "budget": None,
                "timeline": None,
                "objections": [],
                "next_step": None,
                "follow_up_score": 0,
                "follow_up_reason": "Нет содержательного диалога менеджер-клиент для анализа продаж.",
                "tags": ["non_conversation"],
                "quality_flags": quality_flags,
            }
        )
        return normalized

    def _clean_history_summary_draft(self, call: CallRecord, draft: str) -> str:
        cleaned = self._clean_text(draft) or ""
        if not cleaned:
            return ""
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        context_stamps = [self._format_started_at(call.started_at) or ""]
        if call.started_at is not None:
            context_stamps.extend(
                [
                    call.started_at.strftime("%d.%m.%Y %H:%M"),
                    call.started_at.strftime("%d.%m.%Y в %H:%M"),
                ]
            )
        context_stamps = [stamp for stamp in dict.fromkeys(context_stamps) if stamp]
        manager_name = self._clean_text(call.manager_name) or ""
        non_empty_sentences = [self._clean_text(sentence) for sentence in sentences if self._clean_text(sentence)]
        pruned: list[str] = []
        skipping_context = len(non_empty_sentences) > 1
        for sentence in non_empty_sentences:
            compact = self._clean_text(sentence)
            if not compact:
                continue
            lowered = compact.lower()
            has_context = any(stamp in compact for stamp in context_stamps) or bool(
                manager_name and manager_name.lower() in lowered
            )
            if skipping_context and has_context:
                for stamp in context_stamps:
                    compact = re.sub(rf"^{re.escape(stamp)}\s*", "", compact, flags=re.I)
                if manager_name:
                    compact = re.sub(
                        rf"^(?:менеджер\s+)?{re.escape(manager_name)}[\s,:-]*",
                        "",
                        compact,
                        flags=re.I,
                    )
                compact = compact.lstrip(" ,.-")
                lowered = compact.lower()
                if not compact or re.fullmatch(r"(?:менеджер\s+)?общал\w*\s+с\s+клиентом\.?", lowered):
                    continue
            skipping_context = False
            pruned.append(compact)
        return re.sub(r"\s+", " ", " ".join(pruned)).strip()

    @staticmethod
    def _summary_mentions_any(text: str, values: list[Optional[str]]) -> bool:
        lowered = (text or "").lower()
        if not lowered:
            return False
        for value in values:
            cleaned = (value or "").strip().lower()
            if cleaned and cleaned in lowered:
                return True
        return False

    @staticmethod
    def _is_empty_budget_value(value: Optional[str]) -> bool:
        lowered = (value or "").strip().lower()
        return not lowered or lowered in {"не указан", "не указано", "нет", "none", "null", "-"}

    def _build_commercial_lines(self, structured_fields: Dict[str, Any]) -> list[str]:
        commercial = self._nested_dict(structured_fields, "commercial")
        price_sensitivity = self._clean_text(commercial.get("price_sensitivity"))
        budget = self._clean_text(commercial.get("budget"))
        discount_interest = commercial.get("discount_interest")
        price_labels = {
            "high": "высокая",
            "medium": "средняя",
            "low": "низкая",
        }
        bits: list[str] = []
        if price_sensitivity in price_labels:
            bits.append(f"чувствительность к цене: {price_labels[price_sensitivity]}")
        if not self._is_empty_budget_value(budget):
            bits.append(f"бюджет: {budget}")
        if discount_interest is True:
            bits.append("интересуется скидками")
        if not bits:
            return []
        return [f"Коммерческий контекст: {'; '.join(bits)}."]

    def _build_school_line(self, structured_fields: Dict[str, Any]) -> Optional[str]:
        student = self._nested_dict(structured_fields, "student")
        school = self._clean_text(student.get("school"))
        if not school:
            return None
        return f"Школа: {school}."

    def _build_lead_priority_line(self, structured_fields: Dict[str, Any]) -> Optional[str]:
        priority = self._clean_text(structured_fields.get("lead_priority"))
        labels = {
            "hot": "горячий",
            "warm": "теплый",
        }
        if priority not in labels:
            return None
        return f"Приоритет лида: {labels[priority]}."

    def _compose_history_summary(
        self,
        call: CallRecord,
        *,
        draft_history_summary: Optional[str],
        summary: Optional[str],
        structured_fields: Dict[str, Any],
        objections: list[str],
        next_step_action: Optional[str],
        due: Optional[str],
        follow_up_reason: Optional[str],
    ) -> str:
        # ТЗ-04 §7.4: no date/manager preamble.  Both already have their own
        # column in the report and their own field in CRM, so repeating them
        # here only pushed the actual content of the call out of the first line
        # a human reads — and did it in UTC, next to a Moscow column.
        people = self._nested_dict(structured_fields, "people")
        contacts = self._nested_dict(structured_fields, "contacts")
        student = self._nested_dict(structured_fields, "student")
        interests = self._nested_dict(structured_fields, "interests")
        result = self._nested_dict(structured_fields, "result")

        child_fio = self._clean_text(people.get("child_fio"))
        parent_fio = self._clean_text(people.get("parent_fio"))
        grade = self._clean_text(student.get("grade_current"))
        student_bits: list[str] = []
        if child_fio:
            student_bits.append(f"ребенок: {child_fio}")
        if parent_fio:
            student_bits.append(f"родитель: {parent_fio}")
        if grade:
            student_bits.append(f"класс: {grade}")

        products = self._clean_list(interests.get("products"))
        formats = self._clean_list(interests.get("format"))
        subjects = self._clean_list(interests.get("subjects"))
        exams = self._clean_list(interests.get("exam_targets"))
        school_line = self._build_school_line(structured_fields)
        commercial_lines = self._build_commercial_lines(structured_fields)
        lead_priority_line = self._build_lead_priority_line(structured_fields)
        result_status = self._clean_text(result.get("status"))
        result_detail = self._clean_text(result.get("detail"))
        result_bits = self._unique(
            [item for item in (RESULT_STATUS_RU.get(result_status or ""), result_detail) if item]
        )
        result_line = f"Результат: {'; '.join(result_bits)}." if result_bits else None
        topic_parts: list[str] = []
        if products:
            topic_parts.append(f"продукты: {', '.join(products)}")
        if formats:
            topic_parts.append(f"формат: {', '.join(formats)}")
        if subjects:
            topic_parts.append(f"предметы: {', '.join(subjects)}")
        if exams:
            topic_parts.append(f"цели: {', '.join(exams)}")

        email = self._clean_text(contacts.get("email"))
        preferred_channel = self._clean_text(contacts.get("preferred_channel"))
        contact_bits: list[str] = []
        if email:
            contact_bits.append(f"email: {email}")
        if preferred_channel:
            contact_bits.append(f"канал: {preferred_channel}")

        cleaned_draft = self._clean_history_summary_draft(call, draft_history_summary or "")
        if cleaned_draft and self._looks_like_dialogue_dump(cleaned_draft):
            cleaned_draft = None
        if cleaned_draft:
            compact_draft = re.sub(r"\s+", " ", cleaned_draft).strip()
            summary_sentence = self._sentence(summary)
            draft_sentences = [item for item in re.split(r"(?<=[.!?])\s+", compact_draft) if item.strip()]
            draft_is_sparse = len(compact_draft) < 180 or len(draft_sentences) < 2
            parts: list[str] = []
            sentence = self._sentence(compact_draft)
            if sentence:
                parts.append(sentence)
            if (
                summary_sentence
                and not self._looks_like_dialogue_dump(summary_sentence)
                and summary_sentence.lower() not in compact_draft.lower()
                and compact_draft.lower() not in summary_sentence.lower()
                and draft_is_sparse
            ):
                parts.append(f"Суть обращения: {summary_sentence}")
            if student_bits and not self._summary_mentions_any(
                compact_draft,
                [child_fio, parent_fio, grade],
            ):
                parts.append(f"Уточнили данные: {'; '.join(student_bits)}.")
            if topic_parts and not self._summary_mentions_any(
                compact_draft,
                products + formats + subjects + exams,
            ):
                parts.append(f"Обсудили: {'; '.join(topic_parts)}.")
            elif not topic_parts:
                if (
                    summary_sentence
                    and not self._looks_like_dialogue_dump(summary_sentence)
                    and not self._summary_mentions_any(compact_draft, [summary_sentence])
                ):
                    parts.append(f"Суть обращения: {summary_sentence}")
            if objections and not self._summary_mentions_any(compact_draft, objections):
                parts.append(f"Ограничения/возражения: {', '.join(objections)}.")
            for extra_line in [school_line, *commercial_lines, lead_priority_line, result_line]:
                if not extra_line:
                    continue
                current_text = " ".join(parts)
                if not self._summary_mentions_any(current_text, [extra_line]):
                    parts.append(extra_line)
            if next_step_action and not any(
                token in compact_draft.lower()
                for token in ("договор", "следующ", "перезвон", "отправ", "созвон", "соедин")
            ):
                agreement = next_step_action
                if due:
                    agreement = f"{agreement} (срок: {due})"
                parts.append(f"Договорились: {agreement}.")
            elif not result_line and follow_up_reason and not self._summary_mentions_any(compact_draft, [follow_up_reason]):
                reason_sentence = self._sentence(follow_up_reason)
                if reason_sentence:
                    parts.append(f"Итог: {reason_sentence}")
            if contact_bits and not self._summary_mentions_any(compact_draft, [email, preferred_channel]):
                parts.append(f"Контакты: {'; '.join(contact_bits)}.")
            compact = re.sub(r"\s+", " ", " ".join(parts)).strip()
            if len(compact) > 32000:
                compact = compact[:31974].rstrip() + " [обрезано по лимиту поля]"
            return compact

        blocks: list[str] = []
        if student_bits:
            blocks.append(f"Уточнили данные: {'; '.join(student_bits)}.")

        if topic_parts:
            blocks.append(f"Обсудили: {'; '.join(topic_parts)}.")
        else:
            summary_sentence = self._sentence(summary)
            if summary_sentence and not self._looks_like_dialogue_dump(summary_sentence):
                blocks.append(f"Суть обращения: {summary_sentence}")

        if objections:
            blocks.append(f"Ограничения/возражения: {', '.join(objections)}.")

        for extra_line in [school_line, *commercial_lines, lead_priority_line, result_line]:
            if extra_line:
                blocks.append(extra_line)

        if next_step_action:
            agreement = next_step_action
            if due:
                agreement = f"{agreement} (срок: {due})"
            blocks.append(f"Договорились: {agreement}.")
        elif not result_line and follow_up_reason:
            reason_sentence = self._sentence(follow_up_reason)
            if reason_sentence:
                blocks.append(f"Итог: {reason_sentence}")

        if contact_bits:
            blocks.append(f"Контакты: {'; '.join(contact_bits)}.")

        compact = re.sub(r"\s+", " ", " ".join(blocks)).strip()
        if len(compact) > 32000:
            compact = compact[:31974].rstrip() + " [обрезано по лимиту поля]"
        return compact

    def _compose_manager_brief(self, structured_fields: Dict[str, Any]) -> str:
        """Short narrative for the ROP table; adjacent columns keep actions/risks."""
        people = self._nested_dict(structured_fields, "people")
        student = self._nested_dict(structured_fields, "student")
        interests = self._nested_dict(structured_fields, "interests")
        commercial = self._nested_dict(structured_fields, "commercial")
        sentences: list[str] = []

        person_bits = self._unique(
            [
                value
                for value in (
                    self._clean_text(people.get("child_fio")),
                    (
                        f"{self._clean_text(student.get('grade_current'))} класс"
                        if self._clean_text(student.get("grade_current"))
                        else None
                    ),
                    self._clean_text(student.get("school")),
                )
                if value
            ]
        )
        if person_bits:
            sentences.append(f"Обращение по ученику: {', '.join(person_bits)}.")

        products = self._clean_list(interests.get("products"))
        subjects = self._clean_list(interests.get("subjects"))
        primary_topic = next(iter(products or subjects), "").casefold()
        focus = [
            item
            for item in self._unique(
                products
                + subjects
                + self._clean_list(interests.get("format"))
                + self._clean_list(interests.get("exam_targets"))
            )
            if item.casefold() != primary_topic
        ]
        if focus:
            sentences.append(f"В разговоре обсуждали: {'; '.join(focus)}.")

        commercial_bits: list[str] = []
        budget = self._clean_text(commercial.get("budget"))
        if not self._is_empty_budget_value(budget):
            commercial_bits.append(f"бюджет {budget}")
        if commercial.get("discount_interest") is True:
            commercial_bits.append("интерес к скидке")
        if commercial_bits:
            sentences.append(f"Коммерческий контекст: {'; '.join(commercial_bits)}.")

        return " ".join(sentences) or "Подтверждённых деталей для краткого конспекта недостаточно."

    def _quality_flags_from_call(self, call: CallRecord) -> Dict[str, Any]:
        flags: Dict[str, Any] = {}
        raw = (call.transcript_variants_json or "").strip()
        if raw:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                mode = payload.get("mode")
                if mode in {"stereo", "mono_or_fallback"}:
                    flags["mode"] = mode
                    flags["mono_fallback"] = mode == "mono_or_fallback"
                secondary_provider = payload.get("secondary_provider")
                if secondary_provider is not None:
                    flags["secondary_provider"] = str(secondary_provider or "") or None
                secondary_backfill_meta = payload.get("secondary_backfill_meta")
                if isinstance(secondary_backfill_meta, dict):
                    flags["secondary_backfill_status"] = str(
                        secondary_backfill_meta.get("status") or ""
                    ).strip() or None
                    flags["secondary_backfill_exhausted"] = bool(
                        secondary_backfill_meta.get("exhausted")
                    )
                warnings = payload.get("warnings")
                if isinstance(warnings, list):
                    clean_warnings = [str(item) for item in warnings if str(item).strip()]
                    flags["warnings_count"] = len(clean_warnings)
                    flags["has_secondary_empty_warning"] = any(
                        "secondary" in item.lower() and "empty" in item.lower()
                        for item in clean_warnings
                    )
        if "mono_fallback" not in flags:
            flags["mono_fallback"] = False
        return flags

    def _analysis_export_paths(self, call: CallRecord) -> Optional[tuple[Path, Path]]:
        export_dir = (self._settings.transcript_export_dir or "").strip()
        if not export_dir:
            return None
        source_path = Path(call.source_file)
        target_dir = call_artifact_directory(
            self._settings,
            export_dir=Path(export_dir),
            source_file=source_path,
            source_call_id=call.source_call_id,
        )
        stem = source_path.stem
        return (
            target_dir / f"{stem}_history_summary.txt",
            target_dir / f"{stem}_structured_fields.json",
        )

    def _export_analysis_files(self, call: CallRecord, analysis: Dict[str, Any]) -> None:
        paths = self._analysis_export_paths(call)
        if not paths:
            return
        summary_path, structured_path = paths
        history_summary = self._clean_text(analysis.get("history_summary"))
        if not history_summary:
            history_summary = self._clean_text(analysis.get("history_short"))
        if not history_summary:
            history_summary = self._clean_text(analysis.get("summary"))
        write_call_artifact_bytes(
            self._settings,
            summary_path,
            ((history_summary or "") + "\n").encode("utf-8"),
        )

        structured_fields = analysis.get("structured_fields")
        if not isinstance(structured_fields, dict):
            structured_fields = self._nested_dict(analysis, "crm_blocks")
        write_call_artifact_bytes(
            self._settings,
            structured_path,
            json.dumps(
                structured_fields,
                ensure_ascii=False,
                indent=2,
            ).encode("utf-8"),
        )

    def _normalize_analysis(
        self,
        call: CallRecord,
        text: str,
        raw_analysis: Dict[str, Any],
        *,
        non_conversation_advisory_enabled: Optional[bool] = None,
    ) -> Dict[str, Any]:
        raw = raw_analysis if isinstance(raw_analysis, dict) else {}
        blocks = self._nested_dict(raw, "crm_blocks")
        if not blocks:
            blocks = self._nested_dict(raw, "structured_fields")
        people = self._nested_dict(blocks, "people")
        contacts = self._nested_dict(blocks, "contacts")
        student = self._nested_dict(blocks, "student")
        interests = self._nested_dict(blocks, "interests")
        commercial = self._nested_dict(blocks, "commercial")
        next_step_block = self._nested_dict(blocks, "next_step")
        result_block = self._nested_dict(blocks, "result")
        # Closed enum or nothing: an unknown outcome is never displayed, and a
        # detail without its own accepted status has nothing to detail.
        result_status = self._clean_text(result_block.get("status"))
        if result_status not in RESULT_STATUSES:
            result_status = None
        result_detail = self._clean_text(result_block.get("detail")) if result_status else None
        llm_sales_signal_sources: list[str] = []
        if self._clean_list(interests.get("products")):
            llm_sales_signal_sources.append("interests.products")
        if self._clean_text(raw.get("target_product")):
            llm_sales_signal_sources.append("target_product")
        if self._clean_text(next_step_block.get("action")) or (
            not isinstance(raw.get("next_step"), dict) and self._clean_text(raw.get("next_step"))
        ):
            llm_sales_signal_sources.append("next_step.action")
        if self._clean_list(blocks.get("objections")) or self._clean_list(raw.get("objections")):
            llm_sales_signal_sources.append("objections")

        summary = (
            self._clean_text(raw.get("summary"))
            or self._clean_text(raw.get("history_summary"))
            or self._clean_text(raw.get("history_short"))
        )
        if summary and self._looks_like_dialogue_dump(summary):
            summary = None
        if not summary:
            transcript_fallback = (text or "").strip()[:600]
            if not self._looks_like_dialogue_dump(transcript_fallback):
                summary = transcript_fallback
        history_short = (
            self._clean_text(raw.get("history_short"))
            or self._clean_text(raw.get("history_summary"))
            or summary
            or ""
        )
        if history_short and self._looks_like_dialogue_dump(history_short):
            history_short = None
        raw_history_summary = self._clean_text(raw.get("history_summary")) or history_short
        if raw_history_summary and self._looks_like_dialogue_dump(raw_history_summary):
            raw_history_summary = None

        target_product = self._clean_text(raw.get("target_product"))
        legacy_interests = self._clean_list(raw.get("interests"))

        products = self._unique(
            self._clean_list(interests.get("products"))
            + ([target_product] if target_product else [])
            + self._detect_from_patterns(text, PRODUCT_PATTERNS)
        )
        formats = self._unique(
            self._clean_list(interests.get("format"))
            + self._detect_from_patterns(text, FORMAT_PATTERNS)
        )
        subjects = self._unique(
            self._clean_list(interests.get("subjects"))
            + self._detect_from_patterns(text, SUBJECT_PATTERNS)
        )
        exam_targets = self._unique(
            self._clean_list(interests.get("exam_targets"))
            + self._detect_from_patterns(text, EXAM_PATTERNS)
        )
        if not target_product and products:
            target_product = products[0]
        if target_product and target_product not in PRODUCT_PATTERNS:
            target_product = None

        grade_current = (
            self._clean_text(student.get("grade_current"))
            or self._clean_text(raw.get("student_grade"))
            or self._extract_grade(text)
        )
        school = self._clean_text(student.get("school"))
        parent_fio = self._clean_text(people.get("parent_fio"))
        child_fio = self._clean_text(people.get("child_fio"))
        budget = self._clean_text(commercial.get("budget")) or self._clean_text(raw.get("budget"))
        timeline = self._clean_text(raw.get("timeline")) or self._clean_text(next_step_block.get("due"))
        due = self._clean_text(next_step_block.get("due")) or timeline
        phone_from_filename = self._clean_text(contacts.get("phone_from_filename")) or self._clean_text(call.phone)
        email = self._clean_text(contacts.get("email")) or self._extract_email(text)
        preferred_channel = self._clean_text(contacts.get("preferred_channel")) or self._detect_preferred_channel(text)
        pain_points = self._unique(self._clean_list(raw.get("pain_points")))
        personal_offer = self._clean_text(raw.get("personal_offer"))

        price_signal = bool(OBJECTION_PATTERNS["цена"].search(text.lower()))
        raw_price_sensitivity = self._clean_text(commercial.get("price_sensitivity"))
        if raw_price_sensitivity in {"high", "medium", "low"}:
            price_sensitivity = raw_price_sensitivity
        elif price_signal:
            price_sensitivity = "high"
        else:
            price_sensitivity = None

        raw_discount_interest = commercial.get("discount_interest")
        if isinstance(raw_discount_interest, bool):
            discount_interest = raw_discount_interest
        else:
            discount_interest = bool(re.search(r"скидк\w*|акци\w*|рассрочк\w*", text.lower()))

        objections = self._unique(
            self._clean_list(blocks.get("objections"))
            + self._clean_list(raw.get("objections"))
            + self._detect_from_patterns(text, OBJECTION_PATTERNS)
        )
        if not price_signal:
            objections = [
                item
                for item in objections
                if not any(token in item.lower() for token in ("цен", "стоим", "дорог", "бюджет"))
            ]
            if price_sensitivity == "high":
                price_sensitivity = None

        next_step_action = self._clean_text(next_step_block.get("action")) or self._clean_text(
            raw.get("next_step")
        )
        next_step_action = self._normalize_next_step_action(next_step_action)
        next_step_signals = detect_non_conversation_signals(
            transcript_text=text,
            duration_sec=getattr(call, "duration_sec", None),
        )
        if non_conversation_advisory_enabled is None:
            non_conversation_advisory_enabled = _truthy_env_flag(NON_CONVERSATION_ADVISORY_ENV)
        pre_llm_non_conversation_advisory = (
            non_conversation_advisory_enabled and next_step_signals.should_force_non_conversation
        )
        allow_heuristic_next_step = not (
            next_step_signals.should_force_non_conversation
            or (
                next_step_signals.strong_no_live_marker
                and not next_step_signals.protected_live_dialogue
                and next_step_signals.score <= 1
            )
        )
        if (
            not next_step_action
            and allow_heuristic_next_step
            and ("перезвон" in text.lower() or "созвон" in text.lower() or "позвон" in text.lower())
        ):
            next_step_action = "Перезвонить клиенту"
        if not next_step_action and allow_heuristic_next_step and "отправ" in text.lower():
            next_step_action = "Отправить материалы"

        score = self._coerce_score(raw.get("follow_up_score"))
        if score is None:
            if self._is_non_conversation(text) and not non_conversation_advisory_enabled:
                score = 0
            elif next_step_action:
                score = 70
            elif objections:
                score = 55
            else:
                score = 60

        raw_lead_priority = self._clean_text(blocks.get("lead_priority"))
        if raw_lead_priority in {"hot", "warm", "cold"}:
            lead_priority = raw_lead_priority
        else:
            lead_priority = self._priority_from_score(score)

        tags = self._unique(self._clean_list(raw.get("tags")))
        detected_call_type = self._detect_call_type(
            text,
            tags=tags,
            products=products,
            subjects=subjects,
            formats=formats,
            exam_targets=exam_targets,
            next_step_action=next_step_action,
        )
        call_type = detected_call_type
        non_conversation_advisory_sources: list[str] = []
        if non_conversation_advisory_enabled and detected_call_type == "non_conversation":
            if pre_llm_non_conversation_advisory:
                non_conversation_advisory_sources.append("pre_llm_guardrail")
            non_conversation_advisory_sources.append("post_llm_detector")
            call_type = self._non_conversation_advisory_call_type(
                text,
                tags=tags,
                products=products,
                formats=formats,
                exam_targets=exam_targets,
                next_step_action=next_step_action,
            )
        tags = [item for item in tags if item.lower() not in CALL_TYPE_TAGS]
        non_conversation_soft_warning_sources = (
            self._unique(llm_sales_signal_sources) if detected_call_type == "non_conversation" else []
        )
        if detected_call_type == "non_conversation" and not non_conversation_advisory_enabled:
            tags.append("non_conversation")
            products = []
            formats = []
            subjects = []
            exam_targets = []
            target_product = None
            grade_current = None
            school = None
            parent_fio = None
            child_fio = None
            email = None
            preferred_channel = None
            budget = None
            timeline = None
            price_sensitivity = None
            discount_interest = None
            objections = []
            next_step_action = None
            due = None
            pain_points = []
            personal_offer = None
            score = 0
            lead_priority = "cold"
        elif call_type != "sales_call":
            tags.append(call_type)
        if non_conversation_advisory_sources:
            tags.append("non_conversation_advisory")

        follow_up_reason = self._clean_text(raw.get("follow_up_reason"))
        if not follow_up_reason:
            if call_type == "non_conversation":
                follow_up_reason = "Нет содержательного диалога."
            elif next_step_action:
                follow_up_reason = "Есть согласованный следующий шаг."
            else:
                follow_up_reason = "Оценка на основе содержания звонка."
        pain_points = self._unique(pain_points + objections)

        legacy_interests_out = self._unique(legacy_interests + products + formats + subjects + exam_targets)
        quality_flags = self._quality_flags_from_call(call)
        raw_quality = raw.get("quality_flags")
        if isinstance(raw_quality, dict):
            quality_flags.update(raw_quality)
        quality_flags["call_type"] = call_type

        evidence: list[Dict[str, Any]] = []
        raw_evidence = raw.get("evidence")
        if isinstance(raw_evidence, list):
            for item in raw_evidence:
                if not isinstance(item, dict):
                    continue
                text_item = self._clean_text(item.get("text"))
                if not text_item:
                    continue
                evidence.append(
                    {
                        "speaker": self._clean_text(item.get("speaker")),
                        "ts": self._clean_text(item.get("ts")),
                        "text": text_item[:260],
                    }
                )
                if len(evidence) >= 5:
                    break
        if not evidence:
            evidence = self._extract_evidence(text, limit=3)

        structured_fields = {
            "result": {
                "status": result_status,
                "detail": result_detail,
            },
            "people": {
                "parent_fio": parent_fio,
                "child_fio": child_fio,
            },
            "contacts": {
                "email": email,
                "phone_from_filename": phone_from_filename,
                "preferred_channel": preferred_channel,
            },
            "student": {
                "grade_current": grade_current,
                "school": school,
            },
            "interests": {
                "products": products,
                "format": formats,
                "subjects": subjects,
                "exam_targets": exam_targets,
            },
            "commercial": {
                "price_sensitivity": price_sensitivity,
                "budget": budget,
                "discount_interest": discount_interest,
            },
            "objections": objections,
            "next_step": {
                "action": next_step_action,
                "due": due,
            },
            "lead_priority": lead_priority,
        }
        history_summary = self._compose_history_summary(
            call,
            draft_history_summary=raw_history_summary,
            summary=summary,
            structured_fields=structured_fields,
            objections=objections,
            next_step_action=next_step_action,
            due=due,
            follow_up_reason=follow_up_reason,
        )
        if not history_short or self._looks_like_dialogue_dump(history_short):
            history_short = history_summary

        transcript_quality_guardrails = self._transcript_quality_guardrails(
            call,
            text=text,
            history_summary=history_summary,
            call_type=call_type,
            products=products,
            subjects=subjects,
            objections=objections,
            next_step_action=next_step_action,
        )
        quality_flags["transcript_quality_guardrails"] = transcript_quality_guardrails
        quality_flags["transcript_quality_guardrails_version"] = transcript_quality_guardrails["version"]
        quality_flags["transcript_quality_guardrails_mode"] = transcript_quality_guardrails["mode"]
        quality_flags["transcript_quality_label"] = transcript_quality_guardrails["label"]
        quality_flags["transcript_quality_score"] = transcript_quality_guardrails["score"]
        quality_flags["transcript_quality_reason_codes"] = transcript_quality_guardrails["reason_codes"]
        quality_flags["transcript_quality_should_force_non_conversation"] = transcript_quality_guardrails[
            "should_force_non_conversation"
        ]
        quality_flags["transcript_quality_requires_manual_review"] = transcript_quality_guardrails[
            "requires_manual_review"
        ]
        quality_flags["transcript_quality_protected_live_dialogue"] = transcript_quality_guardrails[
            "protected_live_dialogue"
        ]
        quality_flags["transcript_quality_recommended_call_type"] = transcript_quality_guardrails[
            "recommended_call_type"
        ]
        quality_flags["transcript_quality_recommended_contact_subtype"] = transcript_quality_guardrails[
            "recommended_contact_subtype"
        ]

        review_flags = self._build_review_flags(
            call,
            text=text,
            call_type=call_type,
            products=products,
            formats=formats,
            exam_targets=exam_targets,
            target_product=target_product,
            next_step_action=next_step_action,
            history_summary=history_summary,
        )
        review_reasons = list(review_flags["review_reasons"])
        needs_review = bool(review_flags["needs_review"])
        if non_conversation_advisory_sources:
            quality_flags["non_conversation_advisory"] = True
            quality_flags["non_conversation_advisory_env"] = NON_CONVERSATION_ADVISORY_ENV
            quality_flags["non_conversation_advisory_sources"] = self._unique(non_conversation_advisory_sources)
            quality_flags["non_conversation_advisory_recommended_call_type"] = "non_conversation"
            quality_flags["non_conversation_advisory_final_call_type"] = call_type
            review_reasons = self._unique(review_reasons + ["non_conversation_advisory"])
            needs_review = True
        if non_conversation_soft_warning_sources:
            quality_flags["non_conversation_soft_warning_llm_sales_signal"] = True
            quality_flags["non_conversation_soft_warning_sources"] = non_conversation_soft_warning_sources
            review_reasons = self._unique(
                review_reasons + ["non_conversation_llm_sales_signal_soft_warning"]
            )
            needs_review = True
        if quality_flags.get("analyze_prompt_truncated"):
            review_reasons = self._unique(review_reasons + ["analyze_prompt_truncated"])
            needs_review = True
        quality_flags["needs_review"] = needs_review
        quality_flags["review_reasons"] = review_reasons

        normalized: Dict[str, Any] = {
            "analysis_schema_version": LATEST_ANALYSIS_SCHEMA_VERSION,
            "history_summary": history_summary,
            "structured_fields": structured_fields,
            "history_short": history_short,
            "crm_blocks": structured_fields,
            "evidence": evidence,
            "quality_flags": quality_flags,
            # Legacy-compatible keys for existing downstream sync.
            "summary": summary,
            "interests": legacy_interests_out,
            "student_grade": grade_current,
            "target_product": target_product,
            "personal_offer": personal_offer,
            "pain_points": pain_points,
            "budget": budget,
            "timeline": timeline,
            "objections": objections,
            "next_step": next_step_action,
            "follow_up_score": score,
            "follow_up_reason": follow_up_reason,
            "tags": tags,
            "needs_review": needs_review,
            "review_reasons": review_reasons,
        }
        return self._apply_non_conversation_hard_validation(call, normalized)

    # ----------------------------------------------------------------------
    # ТЗ-03 §6: every published high-risk value is traced back to one reply.
    # ----------------------------------------------------------------------

    @staticmethod
    def _call_key(call: Any) -> str:
        """The transferable call key a claim id is computed from.

        A local row id may not enter the formula: two tenants or two providers
        would collide on one claim id, and that id is what a future Customer
        Timeline supersedes a value by.  A row with no provider id at all still
        gets a namespaced key, visibly marked as *not* transferable.
        """
        return call_key_for_record(call_record_view(call))

    @staticmethod
    def _anchor_match(field_path: str, value: Any, text: str):
        """Where in this reply the value is actually said, if it is said at all.

        A value produced by one of our own detectors has to be found by that same
        detector; anything else has to appear literally.  A paraphrase the model
        invented matches nothing here, which is exactly the point.
        """
        haystack = str(text or "")
        list_anchors = CLAIM_LIST_ANCHORS.get(field_path)
        if list_anchors is not None:
            pattern = list_anchors.get(str(value))
            return pattern.search(haystack) if pattern is not None else None
        value_anchors = CLAIM_VALUE_ANCHORS.get(field_path)
        if value_anchors is not None:
            canonical = value_anchors.get(str(value))
            if canonical is not None:
                return canonical.search(haystack)
            if field_path in _VALUE_ANCHOR_REQUIRED_PATHS:
                # A closed enum value we do not know how to hear is not a value
                # we may publish from an unverifiable reference.
                return None
        pattern = CLAIM_SCALAR_ANCHORS.get(field_path)
        if pattern is not None:
            return pattern.search(haystack)
        if isinstance(value, bool) or value is None:
            return None
        literal = str(value).strip().lower()
        if len(literal) < 3:
            return None
        index = haystack.lower().find(literal)
        return None if index < 0 else _LiteralMatch(index, index + len(literal))

    @classmethod
    def _turn_supports(cls, field_path: str, value: Any, turn: Mapping[str, Any]) -> bool:
        """Does this exact reply support the value, and is it not negated?"""
        text = str(turn.get("text") or "")
        speaker = str(turn.get("speaker_kind") or "")
        if field_path == "structured_fields.result.status":
            if "?" in text:
                return False
            if value == "payment_confirmed" and speaker == "manager":
                if PAYMENT_QUESTION_RE.search(text):
                    return False
                match = PAYMENT_CONFIRMATION_BY_MANAGER_RE.search(text)
                return bool(match) and not (
                    _payment_contradicted(text, match.start(), match.end())
                    or _historical_claim(text, match.start())
                )
            # Client words are a signal for later finance reconciliation, not
            # proof that the payment reached us.
            if value == "payment_confirmed":
                return False
            if value in CLIENT_DECIDED_RESULT_STATUSES and speaker != "client":
                return False
            if value in CLIENT_DECIDED_RESULT_STATUSES and CONDITIONAL_COMMITMENT_RE.search(text):
                return False
        if (
            field_path in CLIENT_FACT_PATHS
            or field_path.startswith(CLIENT_FACT_PATH_PREFIXES)
        ) and speaker != "client":
            return False
        if field_path.startswith("structured_fields.next_step.") and "?" in text:
            return False
        if field_path.startswith("structured_fields.next_step.") and (
            CONDITIONAL_COMMITMENT_RE.search(text)
            or NEXT_STEP_GLOBAL_CANCELLATION_RE.search(text)
            or bool(
                NEXT_STEP_ACTION_END_PATTERNS.get(str(value))
                and NEXT_STEP_ACTION_END_PATTERNS[str(value)].search(text)
            )
        ):
            return False
        if field_path in {
            "structured_fields.next_step.action",
            "structured_fields.next_step.due",
        } and not NEXT_STEP_COMMITMENT_RE.search(text):
            return False
        if field_path == "structured_fields.result.detail" and not RESULT_DETAIL_CONTEXT_RE.search(text):
            return False
        if field_path == "structured_fields.commercial.budget" and not BUDGET_CONTEXT_RE.search(text):
            return False
        if field_path == "structured_fields.objections" and value == "цена":
            if PRICE_ACCEPTED_RE.search(text) or not PRICE_OBJECTION_RE.search(text):
                return False
        if (
            field_path == "structured_fields.commercial.price_sensitivity"
            and value == "high"
            and PRICE_ACCEPTED_RE.search(text)
        ):
            return False
        match = cls._anchor_match(field_path, value, turn.get("text"))
        if match is None:
            # A real turn id proves only that the reply exists.  It cannot prove
            # arbitrary free text: otherwise the model could attach an invented
            # name, school, date or next step to "Здравствуйте" and the service
            # would manufacture a quote around it.  Closed canonical values use
            # their detector above; every other value must occur literally.
            return False
        if field_path in {
            "structured_fields.result.status",
            "structured_fields.result.detail",
        } and _historical_claim(text, match.start()):
            return False
        if field_path.startswith("structured_fields.next_step.") and _historical_claim(
            text, match.start()
        ):
            return False
        if (
            field_path == "structured_fields.result.status"
            and value in CLIENT_DECIDED_RESULT_STATUSES
        ):
            tail = text[match.end() :]
            reversal = RESULT_DIRECT_REVERSAL_PATTERNS.get(str(value))
            if reversal is not None and reversal.search(tail):
                return False
            for later_status in RESULT_LATER_OVERRIDES.get(str(value), frozenset()):
                later_pattern = RESULT_STATUS_ANCHORS[later_status]
                later_match = later_pattern.search(text, match.end())
                if later_match is None:
                    continue
                if _historical_claim(text, later_match.start()) or _contradicted(
                    text, later_match.start(), later_match.end()
                ):
                    continue
                return False
        if field_path == "structured_fields.result.status" and value == "payment_confirmed":
            return not _payment_contradicted(text, match.start(), match.end())
        return not _contradicted(text, match.start(), match.end())

    @classmethod
    def _claim_refs_support(
        cls,
        field_path: str,
        value: Any,
        refs: Sequence[Mapping[str, Any]],
        ordered: Sequence[Mapping[str, Any]],
    ) -> bool:
        """Validate either direct replies or one adjacent question/yes pair."""
        if not refs or len({str(ref.get("turn_id") or "") for ref in refs}) != len(refs):
            return False
        directly_supported = all(cls._turn_supports(field_path, value, ref) for ref in refs)
        if not directly_supported:
            pair_allowed = (
                field_path in CLIENT_FACT_PATHS
                or field_path.startswith(CLIENT_FACT_PATH_PREFIXES)
                or (
                    field_path == "structured_fields.result.status"
                    and value in {"follow_up_agreed", "appointment_agreed"}
                )
                or field_path in {
                    "structured_fields.next_step.action",
                    "structured_fields.next_step.due",
                }
            )
            if len(refs) != 2 or not pair_allowed:
                return False
            question, answer = refs
            ids = [str(turn.get("turn_id") or "") for turn in ordered]
            try:
                question_index = ids.index(str(question.get("turn_id") or ""))
                answer_index = ids.index(str(answer.get("turn_id") or ""))
            except ValueError:
                return False
            question_text = str(question.get("text") or "")
            if not (
                answer_index == question_index + 1
                and question.get("speaker_kind") == "manager"
                and answer.get("speaker_kind") == "client"
                and "?" in question_text
                and cls._anchor_match(field_path, value, question_text) is not None
                and SHORT_AFFIRMATIVE_RE.fullmatch(str(answer.get("text") or ""))
            ):
                return False
        if (
            field_path == "structured_fields.result.status"
            and value in RESULT_OUTCOME_STATUSES
        ):
            return not any(
                cls._result_contradicted_after(str(value), ref, ordered) for ref in refs
            )
        if field_path.startswith("structured_fields.next_step."):
            if field_path == "structured_fields.next_step.due" and any(
                cls._next_step_due_superseded_after(str(value), ref, ordered)
                for ref in refs
            ):
                return False
            return not any(
                cls._next_step_cancelled_after(str(value), ref, ordered) for ref in refs
            )
        return True

    def _resolve_claim_turns(
        self,
        *,
        field_path: str,
        value: Any,
        item_id: Optional[str],
        request: Optional[Mapping[str, Any]],
        turns: Mapping[str, Mapping[str, Any]],
        selected: Sequence[str],
        ordered: Sequence[Mapping[str, Any]],
    ) -> tuple[list[Mapping[str, Any]], str]:
        """The replies that really prove this value, and where they came from."""
        if request is not None:
            # High-risk means explicit only: an inferred claim is a guess with a
            # reference attached, which is exactly what evidence must not be.
            if request.get("support_type") != "explicit":
                return [], ""
            if item_id is not None and str(request.get("item_id") or "") != str(item_id):
                return [], ""
            refs: list[Mapping[str, Any]] = []
            for turn_id in request.get("turn_ids") or []:
                turn = turns.get(str(turn_id))
                # A reply that was cut out of the prompt was never seen by the
                # model, so a reference to it cannot be an observation.
                if turn is None or str(turn_id) not in set(selected):
                    return [], ""
                refs.append(turn)
            if refs and self._claim_refs_support(field_path, value, refs, ordered):
                return refs[:CLAIM_MAX_TURN_REFS], "model_claim"
            return [], ""
        selected_ids = set(selected)
        for turn in ordered:
            if str(turn.get("turn_id") or "") not in selected_ids:
                continue
            if self._anchor_match(field_path, value, turn.get("text")) is None:
                continue
            if self._claim_refs_support(field_path, value, [turn], ordered):
                return [turn], "deterministic_detector"
        return [], ""

    @staticmethod
    def _result_contradicted_after(
        status: str,
        supporting_turn: Mapping[str, Any],
        ordered: Sequence[Mapping[str, Any]],
    ) -> bool:
        """A later explicit reversal wins over an earlier commercial outcome."""
        turn_id = str(supporting_turn.get("turn_id") or "")
        for index, turn in enumerate(ordered):
            if str(turn.get("turn_id") or "") != turn_id:
                continue
            later = ordered[index + 1 :]
            if status == "payment_confirmed":
                return any(
                    PAYMENT_DENIAL_TURN_RE.search(str(item.get("text") or ""))
                    or PAYMENT_REVERSAL_TURN_RE.search(str(item.get("text") or ""))
                    for item in later
                )
            reversal = RESULT_DIRECT_REVERSAL_PATTERNS.get(status)
            if reversal is not None and any(
                item.get("speaker_kind") == "client"
                and reversal.search(str(item.get("text") or ""))
                for item in later
            ):
                return True
            overriding = RESULT_LATER_OVERRIDES.get(status, frozenset())
            return any(
                AnalyzeService._turn_supports(
                    "structured_fields.result.status", candidate, item
                )
                for item in later
                for candidate in overriding
            )
        return False

    @staticmethod
    def _next_step_cancelled_after(
        action: str,
        supporting_turn: Mapping[str, Any],
        ordered: Sequence[Mapping[str, Any]],
    ) -> bool:
        """A later explicit completion, cancellation or closure voids the step."""
        turn_id = str(supporting_turn.get("turn_id") or "")
        for index, turn in enumerate(ordered):
            if str(turn.get("turn_id") or "") != turn_id:
                continue
            pattern = NEXT_STEP_ACTION_END_PATTERNS.get(str(action))
            return any(
                NEXT_STEP_GLOBAL_CANCELLATION_RE.search(str(item.get("text") or ""))
                or bool(pattern and pattern.search(str(item.get("text") or "")))
                for item in ordered[index + 1 :]
            )
        return False

    @staticmethod
    def _next_step_due_superseded_after(
        due: str,
        supporting_turn: Mapping[str, Any],
        ordered: Sequence[Mapping[str, Any]],
    ) -> bool:
        """A later committed or explicit rescheduled date replaces the old due."""
        turn_id = str(supporting_turn.get("turn_id") or "")
        for index, turn in enumerate(ordered):
            if str(turn.get("turn_id") or "") != turn_id:
                continue
            for item in ordered[index + 1 :]:
                text = str(item.get("text") or "")
                if not NEXT_STEP_DUE_MARKER_RE.search(text):
                    continue
                if not (
                    NEXT_STEP_COMMITMENT_RE.search(text)
                    or NEXT_STEP_RESCHEDULE_RE.search(text)
                ):
                    continue
                if NEXT_STEP_RESCHEDULE_RE.search(text) or due.casefold() not in text.casefold():
                    return True
            return False
        return False

    def _claim_evidence_entries(
        self,
        *,
        call_key: str,
        field_path: str,
        item_key: str,
        value: Any,
        refs: Sequence[Mapping[str, Any]],
        source: str,
        dialogue: DialogueInput,
    ) -> list[Dict[str, Any]]:
        digest = value_sha256(value)
        contract = (
            CLAIM_CONTRACT_VERSION if source == "model_claim" else DETECTOR_CONTRACT_VERSION
        )
        claim_id = deterministic_claim_id(
            call_key=call_key,
            field_path=field_path,
            item_key=item_key,
            digest=digest,
            contract_version=contract,
        )
        return [
            {
                "claim_id": claim_id,
                "field_path": field_path,
                "item_id": item_key or None,
                "evidence_type": "explicit",
                "support_type": "explicit",
                "source": source,
                "contract_version": contract,
                "turn_id": str(turn["turn_id"]),
                # Quote, timecode and speaker are copied from the stored turn by
                # the service; the model never authors any of the three.
                "exact_quote": str(turn["text"]),
                "timecode": str(turn["timecode"]),
                "speaker_kind": str(turn["speaker_kind"]),
                "start_sec": turn["start_sec"],
                "dialogue_sha256": dialogue.canonical_sha256,
                "raw_value": value,
                "value_sha256": digest,
                "validation_status": "valid",
            }
            for turn in refs
        ]

    def _apply_claim_evidence(
        self,
        call: CallRecord,
        analysis: Dict[str, Any],
        dialogue: DialogueInput,
        claim_requests: Any,
    ) -> Dict[str, Any]:
        """Clear every high-risk value that no reply of this call supports.

        Only the failing field — or the failing list element — is removed; its
        proven neighbours survive, because a whole cleared call teaches the
        sales head to ignore the column instead of reading it.
        """
        fields = analysis.get("structured_fields")
        if not isinstance(fields, dict):
            return analysis
        flags = analysis.get("quality_flags") if isinstance(analysis.get("quality_flags"), dict) else {}
        selected = [str(one) for one in (flags.get("dialogue_selected_turn_ids") or [])]
        ordered = list(dialogue.turns)
        # Missing prompt selection is not permission to inspect the entire
        # conversation after the fact.  The model may evidence only the exact
        # turns that the prompt builder recorded as visible.
        turns = {str(turn["turn_id"]): turn for turn in ordered}
        requests: Dict[tuple, Dict[str, Any]] = {}
        for request in claim_requests if isinstance(claim_requests, list) else []:
            if not isinstance(request, Mapping):
                continue
            key = (str(request.get("field_path") or ""), request.get("item_id"))
            requests.setdefault(key, dict(request))
        call_key = self._call_key(call)
        evidence: list[Dict[str, Any]] = []
        reasons: list[str] = []
        for field_path, spec in CLAIM_FIELD_PATHS.items():
            container, name = self._claim_container(fields, field_path)
            if container is None:
                continue
            value = container.get(name)
            if spec["kind"] == "list":
                kept: list[Any] = []
                for item in list(value or []):
                    item_key = canonical_item_key(item)
                    refs, source = self._resolve_claim_turns(
                        field_path=field_path,
                        value=item,
                        item_id=str(item),
                        request=requests.get((field_path, str(item))),
                        turns=turns,
                        selected=selected,
                        ordered=ordered,
                    )
                    if not refs:
                        reasons.append(claim_field_reason(field_path, item_key))
                        continue
                    kept.append(item)
                    evidence.extend(
                        self._claim_evidence_entries(
                            call_key=call_key, field_path=field_path, item_key=item_key,
                            value=item, refs=refs, source=source, dialogue=dialogue,
                        )
                    )
                container[name] = kept
                continue
            # ``False`` is the absence of a fact ("no interest in a discount"),
            # not a fact about the call, so it needs nothing to prove it.
            if value is None or value is False or value == "" or value == []:
                continue
            refs, source = self._resolve_claim_turns(
                field_path=field_path,
                value=value,
                item_id=None,
                request=requests.get((field_path, None)),
                turns=turns,
                selected=selected,
                ordered=ordered,
            )
            if not refs:
                container[name] = None
                reasons.append(claim_field_reason(field_path))
                continue
            evidence.extend(
                self._claim_evidence_entries(
                    call_key=call_key, field_path=field_path, item_key="",
                    value=value, refs=refs, source=source, dialogue=dialogue,
                )
            )
        next_step = fields.get("next_step")
        if (
            isinstance(next_step, dict)
            and not self._clean_text(next_step.get("action"))
            and self._clean_text(next_step.get("due"))
        ):
            next_step["due"] = None
            evidence = [
                item
                for item in evidence
                if item.get("field_path") != "structured_fields.next_step.due"
            ]
            reasons.append(claim_field_reason("structured_fields.next_step.due"))
        analysis["claim_evidence"] = evidence
        analysis["normalized_facts"] = self._normalized_facts(evidence)
        # The legacy free-form quotes have no field binding and must not survive
        # beside the validated claim ledger in a v3 result.
        analysis.pop("evidence", None)
        analysis["analysis_schema_version"] = ANALYSIS_SCHEMA_VERSION_V3
        analysis["claim_contract_version"] = CLAIM_CONTRACT_VERSION
        self._rebuild_from_validated(call, analysis, evidence=evidence, reasons=reasons)
        return analysis

    @staticmethod
    def _claim_container(
        fields: Dict[str, Any], field_path: str
    ) -> tuple[Optional[Dict[str, Any]], str]:
        """``structured_fields.a.b`` → the ``a`` dict and the name ``b``."""
        parts = field_path.split(".")[1:]
        container: Any = fields
        for part in parts[:-1]:
            container = container.get(part) if isinstance(container, dict) else None
        if not isinstance(container, dict):
            return None, ""
        return container, parts[-1]

    @staticmethod
    def _normalized_facts(evidence: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
        """Explainable display copies; the raw value stays untouched in place."""
        facts: list[Dict[str, Any]] = []
        seen: set[tuple[str, Any, str]] = set()
        for entry in evidence:
            raw_value = entry.get("raw_value")
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            result = normalize_manager_text_with_provenance(
                raw_value, tenant_id=CALLS_TENANT_ID
            )
            if not result.changed:
                continue
            key = (str(entry.get("field_path") or ""), entry.get("item_id"), raw_value)
            if key in seen:
                continue
            seen.add(key)
            facts.append(
                {
                    "field_path": entry["field_path"],
                    "item_id": entry.get("item_id"),
                    "claim_id": entry["claim_id"],
                    "raw_value": result.raw_value,
                    "normalized_value": result.normalized_value,
                    "rule_id": result.rule_ids[0],
                    "rule_ids": list(result.rule_ids),
                    "engine_version": result.engine_version,
                    "ruleset_version": result.ruleset_version,
                    "tenant_id": result.tenant_id,
                    "status": result.status,
                }
            )
        return facts

    def _rebuild_from_validated(
        self,
        call: CallRecord,
        analysis: Dict[str, Any],
        *,
        evidence: Sequence[Mapping[str, Any]],
        reasons: Sequence[str],
    ) -> None:
        """Rebuild every visible block from the values that survived validation.

        The summary, the CRM mirror and the legacy top-level keys are three
        copies of the same facts.  Cleaning only ``structured_fields`` would
        leave the other two carrying the unproven claim into Google and AMO, so
        all of them are recomputed here from one source.
        """
        raw_fields = analysis["structured_fields"]
        contacts_raw = raw_fields.get("contacts")
        if isinstance(contacts_raw, dict):
            contacts_raw.setdefault(
                "phone_from_filename", self._clean_text(getattr(call, "phone", None))
            )
        raw_fields.setdefault(
            "lead_priority",
            self._priority_from_score(self._coerce_score(analysis.get("follow_up_score")) or 0),
        )
        fields = build_display_fields(raw_fields, analysis.get("normalized_facts"))
        interests = self._nested_dict(fields, "interests")
        commercial = self._nested_dict(fields, "commercial")
        student = self._nested_dict(fields, "student")
        next_step = self._nested_dict(fields, "next_step")
        objections = self._clean_list(fields.get("objections"))
        action = self._clean_text(next_step.get("action"))
        due = self._clean_text(next_step.get("due"))
        products = self._clean_list(interests.get("products"))
        formats = self._clean_list(interests.get("format"))
        subjects = self._clean_list(interests.get("subjects"))
        exams = self._clean_list(interests.get("exam_targets"))
        result = self._nested_dict(fields, "result")
        if action:
            follow_up_reason = "Есть подтверждённый следующий шаг."
        elif self._clean_text(result.get("status")):
            follow_up_reason = "Исход разговора подтверждён репликами."
        else:
            follow_up_reason = "Выводы основаны только на подтверждённых репликах."
        summary = self._compose_history_summary(
            call,
            draft_history_summary=None,
            summary=None,
            structured_fields=fields,
            objections=objections,
            next_step_action=action,
            due=due,
            follow_up_reason=follow_up_reason,
        )
        manager_brief = self._compose_manager_brief(fields)
        flags = analysis.get("quality_flags") if isinstance(analysis.get("quality_flags"), dict) else {}
        if flags.get("non_conversation_hard_validation_applied"):
            # Already a deterministic, service-written sentence about a call
            # with no dialogue in it.  Recomposing it from the (correctly)
            # empty fields would only make it say less.
            summary = self._clean_text(analysis.get("history_summary")) or summary
        target_product = products[0] if products and products[0] in PRODUCT_PATTERNS else None
        analysis.update(
            {
                "history_summary": summary,
                "history_short": summary,
                "summary": summary,
                "manager_brief": manager_brief,
                "display_fields": fields,
                "crm_blocks": fields,
                "objections": objections,
                "pain_points": self._unique(list(objections)),
                "next_step": action,
                "timeline": due,
                "budget": self._clean_text(commercial.get("budget")),
                "student_grade": self._clean_text(student.get("grade_current")),
                "target_product": target_product,
                "interests": self._unique(products + formats + subjects + exams),
                # A personal offer was free model text with no path of its own,
                # so it can never be evidenced and never survives v3.
                "personal_offer": None,
                "follow_up_reason": follow_up_reason,
                "history_summary_meta": self._history_summary_meta(evidence, reasons),
            }
        )
        if reasons:
            review_reasons = self._unique(
                [str(item) for item in (analysis.get("review_reasons") or [])] + list(reasons)
            )
            analysis["needs_review"] = True
            analysis["review_reasons"] = review_reasons
            analysis["review_reasons_ru"] = review_reasons_ru(review_reasons)
            flags = analysis.get("quality_flags")
            if isinstance(flags, dict):
                flags["needs_review"] = True
                flags["review_reasons"] = review_reasons
        all_reasons = [str(item) for item in (analysis.get("review_reasons") or [])]
        analysis["review_reasons_ru"] = review_reasons_ru(all_reasons) if all_reasons else []

    @staticmethod
    def _history_summary_meta(
        evidence: Sequence[Mapping[str, Any]], reasons: Sequence[str]
    ) -> Dict[str, Any]:
        """Per-sentence provenance: which claims stand behind which template."""
        by_path: Dict[str, list[Mapping[str, Any]]] = {}
        for entry in evidence:
            by_path.setdefault(str(entry.get("field_path") or ""), []).append(entry)
        parts: list[Dict[str, Any]] = []
        for template_id, paths in SUMMARY_TEMPLATE_FIELDS:
            entries = [item for path in paths for item in by_path.get(path, ())]
            if not entries:
                continue
            parts.append(
                {
                    "template_id": template_id,
                    "claim_ids": sorted({str(item["claim_id"]) for item in entries}),
                    "turn_ids": sorted({str(item["turn_id"]) for item in entries}),
                    "source_code": "validated_claim",
                }
            )
        if reasons:
            parts.append(
                {
                    "template_id": "review_notice_v1",
                    "claim_ids": [],
                    "turn_ids": [],
                    "source_code": "review_required",
                }
            )
        return {
            "contract_version": HISTORY_SUMMARY_CONTRACT_VERSION,
            "parts": parts,
        }

    @staticmethod
    def analysis_schema_version(payload: Dict[str, Any]) -> str:
        raw = payload.get("analysis_schema_version")
        if raw is None:
            return "v1"
        value = str(raw).strip().lower()
        return value or "v1"

    def migrate_analysis_payload(self, call: CallRecord, payload: Dict[str, Any]) -> Dict[str, Any]:
        return migrate_analysis_payload(
            build_analysis_migration_call_snapshot(call),
            payload if isinstance(payload, dict) else {},
        )

    def _analysis_model_for_provider(self, provider: str) -> str:
        if provider == "codex_cli":
            return (self._settings.codex_analyze_model or "").strip() or "unknown"
        if provider == "openai":
            return (self._settings.openai_analysis_model or "").strip() or "unknown"
        if provider == "ollama":
            return (self._settings.ollama_model or "").strip() or "unknown"
        if provider == "mock":
            return "mock"
        return provider or "unknown"

    def _build_analysis_meta(
        self,
        analysis: Dict[str, Any],
        *,
        model_called: bool = True,
        token_usage: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Provider identity plus what actually happened, not what was planned.

        The prompt version and profile are read back from the quality flags the
        *executed* path wrote, so a compact request that escalated to ``full``
        is recorded as ``full`` — the assumption made before the call is not
        evidence of which prompt the model finally saw.
        """
        provider = (self._settings.analyze_provider or "").strip().lower() or "mock"
        quality_flags = analysis.get("quality_flags") if isinstance(analysis.get("quality_flags"), dict) else {}
        prompt_version = quality_flags.get("analyze_prompt_version") or self._analysis_prompt_version()
        legacy_cache_hit = bool(quality_flags.get("analyze_llm_cache_hit"))
        model_call_count = int(
            quality_flags.get(
                "analyze_model_call_count", int(bool(model_called) and not legacy_cache_hit)
            )
            or 0
        )
        cache_hit_count = int(
            quality_flags.get(
                "analyze_cache_hit_count",
                int(legacy_cache_hit),
            )
            or 0
        )
        cache_hit = cache_hit_count > 0 and model_call_count == 0
        usage = token_usage
        if usage is None:
            usage = quality_flags.get("analyze_token_usage")
        if cache_hit:
            usage = CACHE_HIT_TOKEN_USAGE
        return {
            "analysis_model": self._analysis_model_for_provider(provider),
            "analysis_provider": provider,
            "analysis_prompt_version": str(prompt_version),
            "analysis_prompt_profile": str(
                quality_flags.get("analyze_prompt_profile") or ""
            ),
            "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model_called": bool(model_called) and model_call_count > 0,
            "model_call_count": model_call_count if model_called else 0,
            "cache_hit": cache_hit,
            "cache_hit_count": cache_hit_count,
            "model_attempts": list(quality_flags.get("analyze_attempts") or []),
            "token_usage": dict(usage or UNAVAILABLE_TOKEN_USAGE),
            # Everything that can change the answer without the transcript
            # changing.  The Google fingerprint reads these, so a contract bump
            # makes an already published row stale instead of silently correct.
            "analysis_schema_version": str(analysis.get("analysis_schema_version") or ""),
            "analysis_input_sha256": str(quality_flags.get("analysis_input_sha256") or ""),
            "analysis_prompt_sha256": str(quality_flags.get("analysis_prompt_sha256") or ""),
            "dialogue_contract_version": str(quality_flags.get("dialogue_version") or ""),
            "dialogue_canonical_sha256": str(
                quality_flags.get("dialogue_canonical_sha256") or ""
            ),
            "role_guard_version": ROLE_GUARD_VERSION,
            "prompt_contract_version": CLAIM_CONTRACT_VERSION,
            "claim_contract_version": CLAIM_CONTRACT_VERSION,
            "detector_contract_version": DETECTOR_CONTRACT_VERSION,
            "history_summary_contract_version": HISTORY_SUMMARY_CONTRACT_VERSION,
            "normalizer_engine_version": TENANT_TEXT_ENGINE_VERSION,
            "normalizer_ruleset_version": tenant_ruleset_version(CALLS_TENANT_ID),
            "normalizer_tenant_id": CALLS_TENANT_ID,
            "timezone_contract_version": TIMEZONE_CONTRACT_VERSION,
            "manager_output_sha256": manager_output_sha256(analysis),
        }

    @staticmethod
    def _bind_analysis_input_metadata(
        analysis: Dict[str, Any], dialogue: DialogueInput, input_sha: str
    ) -> Dict[str, Any]:
        """Bind deterministic and model paths to the same stored-input contract."""
        flags = dict(
            analysis.get("quality_flags")
            if isinstance(analysis.get("quality_flags"), Mapping)
            else {}
        )
        turn_ids = [str(turn.get("turn_id") or "") for turn in dialogue.turns]
        # The source snapshot and the model prompt are different artifacts.
        # Never let provider-returned metadata replace the source identity.
        flags["analysis_input_sha256"] = input_sha
        defaults = {
            "dialogue_version": dialogue.version,
            "dialogue_source": dialogue.source,
            "dialogue_canonical_sha256": dialogue.canonical_sha256,
            "dialogue_turn_count": len(dialogue.turns),
            "dialogue_selected_turn_ids": turn_ids,
            "dialogue_selected_turn_count": len(turn_ids),
            "dialogue_total_turn_count": len(turn_ids),
        }
        for key, value in defaults.items():
            flags.setdefault(key, value)
        analysis["quality_flags"] = flags
        return analysis

    @staticmethod
    def _with_analysis_runtime_metadata(analysis: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(analysis)
        quality_flags = enriched.get("quality_flags") if isinstance(enriched.get("quality_flags"), dict) else {}
        analysis_meta = enriched.get("analysis_meta") if isinstance(enriched.get("analysis_meta"), dict) else {}
        enriched["analyze_model"] = str(analysis_meta.get("analysis_model") or "")
        enriched["analyze_prompt_profile"] = str(quality_flags.get("analyze_prompt_profile") or "")
        enriched["analyze_prompt_truncated"] = bool(quality_flags.get("analyze_prompt_truncated"))
        enriched["analyze_prompt_chars"] = int(quality_flags.get("analyze_transcript_chars_prompt", 0) or 0)
        return enriched

    def _mock_analysis(self, call: CallRecord, text: str) -> Dict[str, Any]:
        _ = call
        lowered = text.lower()
        tags = []
        if "дорого" in lowered or "expensive" in lowered:
            tags.append("price_sensitive")
        if "перезвон" in lowered or "follow" in lowered:
            tags.append("needs_follow_up")
        score = 60 + (15 if "needs_follow_up" in tags else 0)
        return {
            "summary": text[:600],
            "interests": [],
            "student_grade": None,
            "target_product": None,
            "personal_offer": None,
            "pain_points": [],
            "budget": None,
            "timeline": None,
            "objections": [],
            "next_step": "Перезвонить с персональным предложением.",
            "follow_up_score": min(score, 100),
            "follow_up_reason": "MVP mock-анализ по ключевым словам транскрипта.",
            "tags": tags,
        }

    def _is_non_conversation(self, text: str) -> bool:
        return self._detect_call_type(text) == "non_conversation"

    def _non_conversation_analysis(self, signals: Optional[Any] = None) -> Dict[str, Any]:
        reason_codes = list(getattr(signals, "reason_codes", ()) or ())
        contact_subtype = getattr(signals, "recommended_contact_subtype", None)
        return {
            "summary": "Нецелевой звонок: автоответчик/короткий технический дозвон.",
            "interests": [],
            "student_grade": None,
            "target_product": None,
            "personal_offer": None,
            "pain_points": [],
            "budget": None,
            "timeline": None,
            "objections": [],
            "next_step": None,
            "follow_up_score": 0,
            "follow_up_reason": "Нет содержательного диалога менеджер-клиент для анализа продаж.",
            "tags": ["non_conversation"],
            "quality_flags": {
                "pre_llm_non_conversation_gate": True,
                "transcript_quality_guardrails_version": TRANSCRIPT_QUALITY_GUARDRAILS_VERSION,
                "transcript_quality_label": getattr(signals, "label", None),
                "transcript_quality_score": getattr(signals, "score", None),
                "transcript_quality_reason_codes": reason_codes,
                "transcript_quality_should_force_non_conversation": bool(
                    getattr(signals, "should_force_non_conversation", False)
                ),
                "transcript_quality_recommended_call_type": getattr(signals, "recommended_call_type", None),
                "transcript_quality_recommended_contact_subtype": contact_subtype,
            },
        }

    def _openai_analysis(
        self,
        call: CallRecord,
        text: str,
        profile: Optional[str] = None,
        dialogue: Optional[DialogueInput] = None,
    ) -> Dict[str, Any]:
        client = self._openai_client()
        prompt_context = self._analysis_prompt_context(call, text, profile, dialogue)
        prompt = prompt_context["llm_prompt"]
        user_prompt = prompt_context["user_prompt"]
        metrics = prompt_context["metrics"]
        prompt_version = self._analysis_prompt_version(profile)
        reasoning = (
            "temperature=0.1;runtime="
            + self._analysis_runtime_identity_sha256("openai")
        )
        cached = self._analysis_cache_lookup(
            provider="openai",
            model=self._settings.openai_analysis_model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
        )
        if cached is not None:
            return self._with_analysis_prompt_quality_flags(
                cached,
                metrics=metrics,
                prompt_version=prompt_version,
                cache_hit=True,
            )
        identity = {
            "provider": "openai", "model": self._settings.openai_analysis_model,
            "profile": str(profile or self._analysis_prompt_profile()),
            "prompt_version": prompt_version,
        }
        reserved = self._reserve_model_attempt(**identity)
        try:
            response = client.chat.completions.create(
                model=self._settings.openai_analysis_model,
                temperature=0.1,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": prompt_context["system_prompt"]},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            indeterminate = {**identity, **self._finish_model_attempt(
                reserved, state="indeterminate"
            )}
            raise _model_invocation_error(str(exc), [indeterminate]) from exc
        usage = provider_token_usage(getattr(response, "usage", None))
        try:
            content = response.choices[0].message.content if response.choices else None
            if not content:
                raise RuntimeError("OpenAI analysis returned empty content")
            data = json.loads(content)
            if not isinstance(data, dict):
                raise RuntimeError("OpenAI analysis must return object JSON")
        except Exception as exc:
            failed_attempt = {**identity, **self._finish_model_attempt(
                reserved, state="failed", token_usage=usage
            )}
            raise _model_invocation_error(str(exc), [failed_attempt]) from exc
        completed = {**identity, **self._finish_model_attempt(
            reserved, state="completed", token_usage=usage
        )}
        data = self._with_analysis_prompt_quality_flags(
            data,
            metrics=metrics,
            prompt_version=prompt_version,
            cache_hit=False,
            token_usage=usage,
            provider_attempts=[completed],
        )
        self._cache_analysis_after_commit(
            provider="openai",
            model=self._settings.openai_analysis_model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
            response=data,
        )
        return data

    def _ollama_analysis(
        self,
        call: CallRecord,
        text: str,
        profile: Optional[str] = None,
        dialogue: Optional[DialogueInput] = None,
    ) -> Dict[str, Any]:
        client = self._ollama_client()
        prompt_context = self._analysis_prompt_context(call, text, profile, dialogue)
        prompt = prompt_context["llm_prompt"]
        user_prompt = prompt_context["user_prompt"]
        metrics = prompt_context["metrics"]
        prompt_version = self._analysis_prompt_version(profile)
        num_predict = max(200, int(self._settings.analyze_ollama_num_predict))
        reasoning = (
            f"think={self._settings.ollama_think};"
            f"temperature={self._settings.ollama_temperature};"
            f"num_predict={num_predict};runtime="
            f"{self._analysis_runtime_identity_sha256('ollama')}"
        )
        cached = self._analysis_cache_lookup(
            provider="ollama",
            model=self._settings.ollama_model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
        )
        if cached is not None:
            return self._with_analysis_prompt_quality_flags(
                cached,
                metrics=metrics,
                prompt_version=prompt_version,
                cache_hit=True,
            )
        identity = {
            "provider": "ollama", "model": self._settings.ollama_model,
            "profile": str(profile or self._analysis_prompt_profile()),
            "prompt_version": prompt_version,
        }
        reserved = self._reserve_model_attempt(**identity)
        provider_usage: Dict[str, Any] = {}
        try:
            payload = client.generate_json(
                model=self._settings.ollama_model,
                think=self._settings.ollama_think,
                temperature=self._settings.ollama_temperature,
                system_prompt=prompt_context["system_prompt"],
                user_prompt=user_prompt,
                num_predict=num_predict,
                usage_out=provider_usage,
            )
        except Exception as exc:
            indeterminate = {**identity, **self._finish_model_attempt(
                reserved, state="indeterminate"
            )}
            raise _model_invocation_error(str(exc), [indeterminate]) from exc
        usage = provider_token_usage(provider_usage)
        if not isinstance(payload, dict):
            failed_attempt = {**identity, **self._finish_model_attempt(
                reserved, state="failed", token_usage=usage
            )}
            raise _model_invocation_error(
                "Ollama analysis must return object JSON", [failed_attempt]
            )
        completed = {**identity, **self._finish_model_attempt(
            reserved, state="completed", token_usage=usage
        )}
        payload = self._with_analysis_prompt_quality_flags(
            payload,
            metrics=metrics,
            prompt_version=prompt_version,
            cache_hit=False,
            token_usage=usage,
            provider_attempts=[completed],
        )
        self._cache_analysis_after_commit(
            provider="ollama",
            model=self._settings.ollama_model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
            response=payload,
        )
        return payload

    def _codex_cli_analysis(
        self,
        call: CallRecord,
        text: str,
        profile: Optional[str] = None,
        dialogue: Optional[DialogueInput] = None,
    ) -> Dict[str, Any]:
        codex_bin = (self._settings.codex_cli_command or "codex").strip() or "codex"
        if shutil.which(codex_bin) is None:
            raise RuntimeError(f"codex binary is not available: {codex_bin}")

        prompt_context = self._analysis_prompt_context(call, text, profile, dialogue)
        prompt = prompt_context["llm_prompt"]
        metrics = prompt_context["metrics"]
        prompt_version = self._analysis_prompt_version(profile)
        timeout_sec = max(15, int(self._settings.codex_cli_timeout_sec))
        reasoning_effort = (self._settings.codex_reasoning_effort or "").strip().lower()
        reasoning = (
            f"{reasoning_effort};runtime="
            f"{self._analysis_runtime_identity_sha256('codex_cli')}"
        )
        cached = self._analysis_cache_lookup(
            provider="codex_cli",
            model=self._settings.codex_analyze_model,
            reasoning=reasoning,
            prompt_version=prompt_version,
            prompt=prompt,
        )
        if cached is not None:
            return self._with_analysis_prompt_quality_flags(
                cached,
                metrics=metrics,
                prompt_version=prompt_version,
                cache_hit=True,
            )

        identity = {
            "provider": "codex_cli",
            "model": self._settings.codex_analyze_model,
            "profile": str(profile or self._analysis_prompt_profile()),
            "prompt_version": prompt_version,
        }
        reserved = self._reserve_model_attempt(**identity)
        with tempfile.NamedTemporaryFile(
            prefix="mango_codex_analyze_", suffix=".txt"
        ) as out_file:
            cmd = [
                codex_bin, "exec", "--skip-git-repo-check", "--ephemeral",
                "--ignore-user-config", "--sandbox", "read-only", "--model",
                self._settings.codex_analyze_model, "--output-last-message",
                out_file.name,
            ]
            append_codex_service_tier(cmd)
            if reasoning_effort in {"low", "medium", "high"}:
                cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
            cmd.append("-")
            try:
                proc = subprocess.run(
                    cmd, input=prompt, capture_output=True, text=True,
                    check=False, timeout=timeout_sec,
                )
            except Exception as exc:
                indeterminate = {
                    **identity,
                    **self._finish_model_attempt(reserved, state="indeterminate"),
                }
                raise _model_invocation_error(str(exc), [indeterminate]) from exc
            raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore")

        for candidate in (raw, proc.stdout or "", proc.stderr or ""):
            candidate = (candidate or "").strip()
            if not candidate:
                continue
            try:
                payload = self._extract_json_payload(candidate)
            except RuntimeError:
                continue
            if isinstance(payload, dict):
                completed = {
                    **identity,
                    **self._finish_model_attempt(reserved, state="completed"),
                }
                payload = self._with_analysis_prompt_quality_flags(
                    payload, metrics=metrics, prompt_version=prompt_version,
                    cache_hit=False, provider_attempts=[completed],
                )
                self._cache_analysis_after_commit(
                    provider="codex_cli", model=self._settings.codex_analyze_model,
                    reasoning=reasoning, prompt_version=prompt_version,
                    prompt=prompt, response=payload,
                )
                return payload

        failed = {**identity, **self._finish_model_attempt(reserved, state="failed")}
        message = (
            "Codex analysis returned empty content"
            if proc.returncode == 0
            else f"codex exec failed rc={proc.returncode}"
        )
        raise _model_invocation_error(message, [failed])

    def _analyze_text(
        self,
        call: CallRecord,
        text: str,
        dialogue: Optional[DialogueInput] = None,
    ) -> Dict[str, Any]:
        signals = detect_non_conversation_signals(
            transcript_text=text,
            duration_sec=getattr(call, "duration_sec", None),
        )
        non_conversation_advisory_enabled = _truthy_env_flag(NON_CONVERSATION_ADVISORY_ENV)
        if signals.should_force_non_conversation and not non_conversation_advisory_enabled:
            return self._non_conversation_analysis(signals)
        if self._is_non_conversation(text) and not non_conversation_advisory_enabled:
            return self._non_conversation_analysis()
        provider = self._settings.analyze_provider
        profile = self._analysis_prompt_profile()
        if provider == "mock":
            return self._mock_analysis(call, text)
        try:
            if provider == "openai":
                payload = self._openai_analysis(call, text, profile, dialogue)
            elif provider == "ollama":
                payload = self._ollama_analysis(call, text, profile, dialogue)
            elif provider == "codex_cli":
                payload = self._codex_cli_analysis(call, text, profile, dialogue)
            else:
                raise RuntimeError(f"Unsupported ANALYZE_PROVIDER={provider}")
        except Exception as exc:
            if getattr(exc, "model_attempts", None) is not None:
                raise
            raise _model_invocation_error(
                str(exc),
                [
                    {
                        "provider": provider,
                        "model": self._analysis_model_for_provider(provider),
                        "profile": profile,
                        "prompt_version": self._analysis_prompt_version(profile),
                        "cache_hit": False,
                        "model_called": True,
                        "token_usage": dict(UNAVAILABLE_TOKEN_USAGE),
                    }
                ],
            ) from exc
        stages: list[Dict[str, Any]] = []

        def record_stage(answer: Mapping[str, Any], executed_profile: str) -> None:
            flags = answer.get("quality_flags") if isinstance(answer.get("quality_flags"), Mapping) else {}
            cache_hit = bool(flags.get("analyze_llm_cache_hit"))
            usage = flags.get("analyze_token_usage")
            prompt_version = str(flags.get("analyze_prompt_version") or "")
            provider_attempts = flags.get("analyze_provider_attempts")
            if not cache_hit and isinstance(provider_attempts, list) and provider_attempts:
                for attempt in provider_attempts:
                    attempt_usage = (
                        attempt.get("token_usage") if isinstance(attempt, Mapping) else None
                    )
                    stage = {
                        "provider": str(attempt.get("provider") or provider),
                        "model": str(
                            attempt.get("model") or self._analysis_model_for_provider(provider)
                        ),
                        "profile": str(attempt.get("profile") or executed_profile),
                        "prompt_version": str(
                            attempt.get("prompt_version") or prompt_version
                        ),
                        "cache_hit": bool(attempt.get("cache_hit")),
                        "model_called": bool(attempt.get("model_called", True)),
                        "token_usage": dict(attempt_usage)
                        if isinstance(attempt_usage, Mapping)
                        else dict(UNAVAILABLE_TOKEN_USAGE),
                    }
                    for key in (
                        "attempt_id", "stage", "state", "analysis_source_sha256",
                    ):
                        if attempt.get(key) is not None:
                            stage[key] = attempt.get(key)
                    stages.append(stage)
                return
            stages.append(
                {
                    "provider": provider,
                    "model": self._analysis_model_for_provider(provider),
                    "profile": executed_profile,
                    "prompt_version": prompt_version,
                    "cache_hit": cache_hit,
                    "model_called": not cache_hit,
                    "token_usage": dict(usage)
                    if isinstance(usage, Mapping)
                    else dict(UNAVAILABLE_TOKEN_USAGE),
                }
            )

        record_stage(payload, profile)
        if profile == "compact" and self._should_escalate_full_profile(text, payload):
            try:
                if provider == "openai":
                    payload = self._openai_analysis(call, text, "full", dialogue)
                elif provider == "ollama":
                    payload = self._ollama_analysis(call, text, "full", dialogue)
                elif provider == "codex_cli":
                    payload = self._codex_cli_analysis(call, text, "full", dialogue)
            except Exception as exc:
                failed_attempts = list(getattr(exc, "model_attempts", []) or [])
                if not failed_attempts:
                    failed_attempts = [
                        {
                            "provider": provider,
                            "model": self._analysis_model_for_provider(provider),
                            "profile": "full",
                            "prompt_version": self._analysis_prompt_version("full"),
                            "cache_hit": False,
                            "model_called": True,
                            "token_usage": dict(UNAVAILABLE_TOKEN_USAGE),
                        }
                    ]
                raise _model_invocation_error(
                    str(exc), [*stages, *failed_attempts]
                ) from exc
            record_stage(payload, "full")
        flags = dict(payload.get("quality_flags") or {})
        flags.update(
            {
                "analyze_attempts": stages,
                "analyze_model_call_count": sum(
                    1 for stage in stages if stage["model_called"]
                ),
                "analyze_cache_hit_count": sum(
                    1 for stage in stages if stage["cache_hit"]
                ),
                "analyze_token_usage": aggregate_token_usage(stages),
            }
        )
        payload = {**payload, "quality_flags": flags}
        # One gate for all three providers: whatever the model sent is either
        # exactly the v3 contract or it never becomes a production payload.
        return validate_v3_model_response(payload)

    def _analysis_runtime_identity_sha256(self, provider: str) -> str:
        """Hash output-affecting runtime controls without storing endpoint text."""
        provider = str(provider or "").strip().lower()
        controls: Dict[str, Any] = {"provider": provider}
        if provider == "ollama":
            controls.update(
                base_url=str(self._settings.ollama_base_url or "").strip().rstrip("/"),
                think=str(self._settings.ollama_think or "").strip().lower(),
                temperature=float(self._settings.ollama_temperature),
                num_predict=max(200, int(self._settings.analyze_ollama_num_predict)),
            )
        elif provider == "openai":
            controls["temperature"] = 0.1
        elif provider == "codex_cli":
            controls.update(
                command=str(self._settings.codex_cli_command or "codex").strip(),
                reasoning=str(self._settings.codex_reasoning_effort or "").strip().lower(),
            )
        payload = json.dumps(
            controls, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _analysis_prompt_identity(self) -> Dict[str, str]:
        """Provider, model and prompt version: part of the input, not of luck."""
        provider = (self._settings.analyze_provider or "").strip().lower() or "mock"
        return {
            "provider": provider,
            "model": self._analysis_model_for_provider(provider),
            "prompt_version": str(self._analysis_prompt_version()),
            "runtime_sha256": self._analysis_runtime_identity_sha256(provider),
        }

    def _transition_analysis_claim(
        self,
        session: Session,
        *,
        call_id: int,
        worker_id: str,
        snapshot: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> bool:
        """Move the row only if the claim and the whole input never moved.

        The model call can take minutes.  Re-reading the row and then writing it
        is two statements: between them another worker can re-claim the call or
        the input can change, and our session snapshot may not even see that
        foreign commit.  So every outcome — success *and* failure — is one
        conditional UPDATE; the database itself compares the lease and the full
        input snapshot, and ``rowcount != 1`` means the claim is stale and
        nothing of ours was written.  A failure path that wrote unconditionally
        would steal the new owner's lease and mark their work dead.
        """
        session.expunge_all()
        conditions = [
            CallRecord.id == int(call_id),
            CallRecord.analysis_status == "in_progress",
            CallRecord.analysis_worker_id == worker_id,
        ]
        conditions.extend(
            getattr(CallRecord, name) == snapshot.get(name)
            for name in ANALYSIS_INPUT_COLUMNS
        )
        result = session.execute(
            sa_update(CallRecord)
            .where(*conditions)
            .values(**{**dict(values), "updated_at": self._utc_now()})
            .execution_options(synchronize_session=False)
        )
        return int(result.rowcount or 0) == 1

    @staticmethod
    def _read_attempt_ledger(row: CallRecord) -> list[Dict[str, Any]]:
        raw = row.analysis_attempts_json
        try:
            parsed = json.loads(str(raw)) if raw else []
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("analysis attempts ledger is invalid") from exc
        if not isinstance(parsed, list) or any(not isinstance(item, Mapping) for item in parsed):
            raise RuntimeError("analysis attempts ledger is invalid")
        return [dict(item) for item in parsed]

    def _store_analysis_attempt(
        self,
        session: Session,
        *,
        call_id: int,
        snapshot: Mapping[str, Any],
        attempt: Mapping[str, Any],
        replace: bool,
    ) -> bool:
        """CAS append/finalize with commit-ack readback and stable call identity."""
        incoming = dict(attempt)
        attempt_id = str(incoming.get("attempt_id") or "")
        if not attempt_id:
            raise RuntimeError("analysis attempt needs attempt_id")
        stable_identity = {
            name: snapshot.get(name) for name in ("source_call_id", "source_recording_id")
        }

        def exact_readback() -> bool:
            session.expunge_all()
            row = session.get(CallRecord, int(call_id))
            if row is None or any(
                getattr(row, name) != value for name, value in stable_identity.items()
            ):
                return False
            matches = [
                item for item in self._read_attempt_ledger(row)
                if str(item.get("attempt_id") or "") == attempt_id
            ]
            return len(matches) == 1 and matches[0] == incoming

        for _ in range(4):
            session.expunge_all()
            row = session.get(CallRecord, int(call_id))
            if row is None or any(
                getattr(row, name) != value for name, value in stable_identity.items()
            ):
                session.rollback()
                return False
            old_raw = row.analysis_attempts_json
            stored = self._read_attempt_ledger(row)
            stored_ids = [
                str(item.get("attempt_id") or "") for item in stored
                if item.get("attempt_id")
            ]
            if len(stored_ids) != len(set(stored_ids)):
                raise RuntimeError("analysis attempts ledger has duplicate attempt ids")
            positions = [
                index for index, item in enumerate(stored)
                if str(item.get("attempt_id") or "") == attempt_id
            ]
            if len(positions) > 1:
                raise RuntimeError("analysis attempts ledger has duplicate attempt ids")
            if positions and stored[positions[0]] == incoming:
                session.rollback()
                return True
            if positions and not replace:
                raise RuntimeError("analysis attempt id collision")
            if replace and not positions:
                raise RuntimeError("analysis attempt reservation is missing")
            if positions and replace:
                existing = stored[positions[0]]
                immutable = (
                    "attempt_id", "stage", "analysis_source_sha256", "provider",
                    "model", "profile", "prompt_version", "cache_hit",
                )
                if any(existing.get(key) != incoming.get(key) for key in immutable):
                    raise RuntimeError("analysis attempt identity changed during finalize")
                if existing.get("state") != "reserved":
                    raise RuntimeError("analysis attempt was already finalized")
            updated = list(stored)
            if positions:
                updated[positions[0]] = incoming
            else:
                updated.append(incoming)
            result = session.execute(
                text(
                    "UPDATE call_records SET analysis_attempts_json = :new_ledger "
                    "WHERE id = :call_id AND ((analysis_attempts_json = :old_ledger) "
                    "OR (analysis_attempts_json IS NULL AND :old_ledger IS NULL))"
                ),
                {
                    "call_id": int(call_id),
                    "old_ledger": old_raw,
                    "new_ledger": json.dumps(updated, ensure_ascii=False),
                },
            )
            if int(result.rowcount or 0) != 1:
                session.rollback()
                continue
            try:
                session.commit()
                return True
            except Exception as exc:  # commit may succeed while its acknowledgement is lost
                session.rollback()
                if exact_readback():
                    session.rollback()
                    return True
                raise exc
        raise RuntimeError("analysis attempts ledger CAS conflict")

    def _reserve_model_attempt(
        self, *, provider: str, model: str, profile: str, prompt_version: str
    ) -> Optional[Dict[str, Any]]:
        context = self._analysis_attempt_context
        if context is None:
            return None
        context["sequence"] = int(context.get("sequence") or 0) + 1
        entry = {
            "attempt_id": (
                f"{context['worker_id']}:{context['run_attempt']}:{context['sequence']}"
            ),
            "stage": "analyze",
            "state": "reserved",
            "analysis_source_sha256": context["source_sha"],
            "provider": provider,
            "model": model,
            "profile": profile,
            "prompt_version": prompt_version,
            "cache_hit": False,
            "model_called": None,
            "token_usage": dict(UNAVAILABLE_TOKEN_USAGE),
        }
        if not self._store_analysis_attempt(
            context["session"], call_id=context["call_id"],
            snapshot=context["snapshot"], attempt=entry, replace=False,
        ):
            raise _StaleAnalysisClaim("call identity changed before model invocation")
        return entry

    def _finish_model_attempt(
        self,
        reserved: Optional[Mapping[str, Any]],
        *,
        state: str,
        token_usage: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if reserved is None:
            return {
                "state": state,
                "cache_hit": False,
                "model_called": True,
                "token_usage": dict(token_usage or UNAVAILABLE_TOKEN_USAGE),
            }
        context = self._analysis_attempt_context
        if context is None:
            raise RuntimeError("analysis attempt context disappeared")
        completed = {
            **dict(reserved),
            "state": state,
            "model_called": True,
            "token_usage": dict(token_usage or UNAVAILABLE_TOKEN_USAGE),
        }
        if not self._store_analysis_attempt(
            context["session"], call_id=context["call_id"],
            snapshot=context["snapshot"], attempt=completed, replace=True,
        ):
            raise _StaleAnalysisClaim("call identity changed after model invocation")
        return completed

    def _persist_result_attempts(
        self,
        attempts: Sequence[Mapping[str, Any]],
        *,
        default_state: str,
    ) -> list[Dict[str, Any]]:
        context = self._analysis_attempt_context
        if context is None:
            return [dict(item) for item in attempts]
        for item in attempts:
            context["sequence"] = int(context.get("sequence") or 0) + 1
            entry = {
                **dict(item),
                "attempt_id": str(item.get("attempt_id") or (
                    f"{context['worker_id']}:{context['run_attempt']}:"
                    f"{context['sequence']}"
                )),
                "stage": str(item.get("stage") or "analyze"),
                "state": str(item.get("state") or default_state),
                "analysis_source_sha256": str(
                    item.get("analysis_source_sha256") or context["source_sha"]
                ),
                "cache_hit": bool(item.get("cache_hit")),
                "model_called": bool(item.get("model_called")),
                "token_usage": dict(
                    item.get("token_usage")
                    if isinstance(item.get("token_usage"), Mapping)
                    else UNAVAILABLE_TOKEN_USAGE
                ),
            }
            if not self._store_analysis_attempt(
                context["session"], call_id=context["call_id"],
                snapshot=context["snapshot"], attempt=entry, replace=False,
            ):
                raise _StaleAnalysisClaim("call identity changed while saving usage")
        context["session"].expunge_all()
        row = context["session"].get(CallRecord, int(context["call_id"]))
        if row is None:
            raise _StaleAnalysisClaim("call disappeared while saving usage")
        return self._read_attempt_ledger(row)

    def run(self, session: Session, limit: int) -> Dict[str, int]:
        self._analysis_attempt_context = None
        max_attempts = max(1, self._settings.analyze_max_attempts)
        worker_id = self._analysis_worker_id()
        claimed_ids = self._claim_batch(session, limit=limit, worker_id=worker_id)
        success = 0
        failed = 0
        stale = 0
        export_failed = 0
        cache_write_failed = 0
        role_attribution_trusted = 0
        role_attribution_untrusted = 0
        for call_id in claimed_ids:
            call = session.get(CallRecord, call_id)
            if call is None:
                continue
            if call.analysis_status != "in_progress" or call.analysis_worker_id != worker_id:
                continue
            scope = require_unique_controlled_call(session, self._settings)
            if scope and call.source_call_id != scope.source_call_id:
                raise RuntimeError("controlled_call_claim_identity_mismatch")
            attempt = int(call.analyze_attempts or 0) + 1
            # The exact stored input read before the model runs: the same values
            # every conditional transition below compares against.
            snapshot = analysis_input_snapshot(call)
            prompt_identity = self._analysis_prompt_identity()
            committed: Optional[Dict[str, Any]] = None
            self._read_attempt_ledger(call)
            raw_analysis: Dict[str, Any] = {}
            source_sha = ""
            try:
                source_sha = analysis_input_identity_sha256(snapshot, prompt_identity)
                self._analysis_attempt_context = {
                    "session": session,
                    "call_id": call_id,
                    "snapshot": snapshot,
                    "source_sha": source_sha,
                    "worker_id": worker_id,
                    "run_attempt": attempt,
                    "sequence": 0,
                    "cache_writes": [],
                }
                dialogue = build_dialogue_input(snapshot)
                if dialogue.trusted:
                    role_attribution_trusted += 1
                else:
                    role_attribution_untrusted += 1
                # When the model is asked at all it sees only the canonical
                # dialogue; a second, differently cleaned ``transcript_text``
                # would be an untested input path.
                if dialogue.trusted:
                    text_value = dialogue.render()
                    raw_analysis = self._analyze_text(call, text_value, dialogue)
                    analysis = self._normalize_analysis(call, text_value, raw_analysis)
                    analysis = apply_role_guard(analysis, dialogue)
                    # ТЗ-03: only now does the payload become v3 — every
                    # high-risk value that no reply of *this* call supports is
                    # removed, and what stays carries its quote and timecode.
                    analysis = self._apply_claim_evidence(
                        call,
                        analysis,
                        dialogue,
                        raw_analysis.get("claim_requests")
                        if isinstance(raw_analysis, Mapping)
                        else None,
                    )
                    flags = analysis.get("quality_flags")
                    flags = flags if isinstance(flags, Mapping) else {}
                    deterministic = (
                        (self._settings.analyze_provider or "").strip().lower() == "mock"
                        or bool(flags.get("pre_llm_non_conversation_gate"))
                    )
                    model_called = not deterministic
                    token_usage = (
                        SKIPPED_DETERMINISTIC_TOKEN_USAGE if deterministic else None
                    )
                else:
                    # ТЗ-02 R3: with the sides unproven, every role-dependent
                    # answer would be deleted again a moment later.  Asking the
                    # model anyway costs tokens for nothing and leaves its
                    # guesses one refactor away from a neighbouring field, so
                    # neither the model nor the response cache is touched: the
                    # published result is built deterministically from the
                    # dialogue itself.
                    analysis = project_untrusted_analysis(
                        {"analysis_schema_version": ANALYSIS_SCHEMA_VERSION_V3},
                        dialogue,
                    )
                    model_called = False
                    token_usage = SKIPPED_TOKEN_USAGE
                analysis = self._bind_analysis_input_metadata(
                    analysis, dialogue, source_sha
                )
                analysis["analysis_meta"] = self._build_analysis_meta(
                    analysis,
                    model_called=model_called,
                    token_usage=token_usage,
                )
                current_model_attempts = list(
                    analysis["analysis_meta"].get("model_attempts") or []
                )
                all_model_attempts = self._persist_result_attempts(
                    current_model_attempts, default_state="completed"
                )
                analysis["analysis_meta"]["model_attempts"] = all_model_attempts
                if all_model_attempts:
                    analysis["analysis_meta"]["model_call_count"] = sum(
                        bool(item.get("model_called")) for item in all_model_attempts
                    )
                    analysis["analysis_meta"]["cache_hit_count"] = sum(
                        bool(item.get("cache_hit")) for item in all_model_attempts
                    )
                    analysis["analysis_meta"]["token_usage"] = aggregate_token_usage(
                        all_model_attempts
                    )
                analysis["analysis_meta"]["analysis_source_sha256"] = source_sha
                analysis = self._with_analysis_runtime_metadata(analysis)
                if prompt_identity != self._analysis_prompt_identity():
                    raise _StaleAnalysisClaim("prompt identity changed during the call")
                analysis_json = json.dumps(analysis, ensure_ascii=False)
                if not self._transition_analysis_claim(
                    session,
                    call_id=call_id,
                    worker_id=worker_id,
                    snapshot=snapshot,
                    values={
                        "analysis_json": analysis_json,
                        "analysis_status": "done",
                        "analysis_worker_id": None,
                        "analysis_claimed_at": None,
                        "sync_status": "pending",
                        "next_retry_at": None,
                        "dead_letter_stage": None,
                        "last_error": None,
                        "analyze_attempts": attempt,
                    },
                ):
                    raise _StaleAnalysisClaim("claim or input changed during the call")
                try:
                    session.commit()
                except Exception as exc:
                    session.rollback()
                    session.expunge_all()
                    row = session.get(CallRecord, call_id)
                    committed_after_ack_loss = bool(
                        row is not None
                        and analysis_input_snapshot(row) == snapshot
                        and row.analysis_status == "done"
                        and row.analysis_worker_id is None
                        and row.analysis_json == analysis_json
                        and int(row.analyze_attempts or 0) == attempt
                    )
                    if not committed_after_ack_loss:
                        raise exc
                committed = analysis
                cache_write_failed += self._flush_analysis_cache_writes()
                success += 1
            except _StaleAnalysisClaim:
                session.rollback()
                # Provider attempts were reserved and finalized independently of
                # the result lease, so a stolen result cannot erase their cost.
                self._analysis_attempt_context = None
                stale += 1
                continue
            except Exception as exc:  # noqa: BLE001
                session.rollback()
                dead = attempt >= max_attempts
                failed_attempts = list(getattr(exc, "model_attempts", []) or [])
                if not failed_attempts:
                    failed_attempts = _analysis_model_attempts(raw_analysis)
                self._persist_result_attempts(
                    failed_attempts, default_state="failed"
                )
                if not self._transition_analysis_claim(
                    session,
                    call_id=call_id,
                    worker_id=worker_id,
                    snapshot=snapshot,
                    values={
                        "analysis_status": "dead" if dead else "failed",
                        "analysis_worker_id": None,
                        "analysis_claimed_at": None,
                        "dead_letter_stage": "analyze" if dead else None,
                        "next_retry_at": (
                            None if dead else self._utc_now() + self._retry_delay(attempt)
                        ),
                        "last_error": safe_error_text("analyze", exc),
                        "analyze_attempts": attempt,
                    },
                ):
                    session.rollback()
                    self._analysis_attempt_context = None
                    stale += 1
                    continue
                session.commit()
                self._analysis_attempt_context = None
                failed += 1
                continue
            # Files are a projection of the committed row, never of a result a
            # rollback or a lost claim threw away.  The database stays the
            # source of truth: a failed artefact is counted, not propagated,
            # and it never stops the next claimed row.
            try:
                self._export_analysis_files(session.get(CallRecord, call_id), committed)
            except Exception:  # noqa: BLE001
                export_failed += 1
            self._analysis_attempt_context = None
        return {
            "processed": len(claimed_ids),
            "claimed": len(claimed_ids),
            "success": success,
            "failed": failed,
            "stale": stale,
            "export_failed": export_failed,
            "cache_write_failed": cache_write_failed,
            "role_attribution_trusted": role_attribution_trusted,
            "role_attribution_untrusted": role_attribution_untrusted,
            "worker_id": worker_id,
        }
