"""Deterministic guardrails for voicemail/no-live ASR artifacts.

This module is intentionally independent from AnalyzeService.  The first
production use should be a dry-run and regression comparison, not an immediate
rewrite of historical analysis.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable


LABEL_NON_CONVERSATION_HIGH_CONFIDENCE = "non_conversation_high_confidence"
LABEL_MANUAL_REVIEW_PROBABLE_NO_LIVE = "manual_review_probable_no_live"
LABEL_MANUAL_REVIEW_BORDERLINE_LIVE_CONTEXT = "manual_review_borderline_live_context"
LABEL_CONTENTFUL_PROTECTED_LIVE_DIALOGUE = "contentful_protected_live_dialogue"
LABEL_CONTENTFUL_LOW_RISK = "contentful_low_risk"

SYSTEM_NO_DIALOGUE_RE = re.compile(
    r"голосов(?:ая|ой)\s+(?:почта|почтовый\s+ящик)|"
    r"звонок\s+(?:был\s+)?(?:перенаправлен|переведен)(?:\s+на\s+голосов)?|"
    r"остав(?:ить|ьте)\s+сообщени|после\s+звуков(?:ого)?\s+сигнал|"
    r"абонент(?:\s+сейчас)?\s+(?:не\s+может|не\s+отвечает|не\s+ответил|недоступен|временно\s+недоступен)|"
    r"вызываемый\s+абонент|абонент\s+занят|номер\s+(?:недоступен|не\s+отвечает)|"
    r"не\s+отвечает\s+на\s+ваш\s+звонок|не\s+может\s+принять\s+ваш\s+звонок|"
    r"вне\s+зоны\s+действия|находится\s+вне\s+зоны|телефон\s+(?:выключен|занят|разряжен)|"
    r"в\s+данный\s+момент\s+мы\s+не\s+можем\s+ответить|"
    r"продолжаем\s+дозваниваться|оставайтесь\s+на\s+линии|"
    r"недозвон|контакт\s+не\s+состоя|клиент\s+не\s+ответил|"
    r"не\s+удалось\s+(?:связаться|дозвониться|поговорить)|"
    r"живого\s+диалога\s+не\s+было|разговора\s+с\s+клиент[а-я]*\s+не\s+было|"
    r"(?:я\s+)?(?:виртуальн\w+\s+)?секретар[ьяь]\b|на\s+связи\s+я\s+секретар[ьяь]\b|"
    r"вы\s+говорите\s+с\s+секретар[её]м|говорите\s+с\s+секретар[её]м|"
    r"голосов(?:ой|ая)\s+(?:ассистент|помощник)|ассистент\s+(?:миа|мия|ния)|временно\s+попросили\s+отвечать|"
    r"(?:сбербанк[^.]{0,60}(?:голосов|помощник)|(?:голосов|помощник)[^.]{0,60}сбербанк)|"
    r"целевые\s+финансы|7\s*sky|сервис\s+резерв|актив\s+бизнес\s+консалт|коллекторск\w+\s+организац|"
    r"групп[ауы]\s+компан(?:ии|ий)\s+мтс|ооо\s+пко|действующ\w+\s+в\s+интересах|"
    r"отправ(?:ить|ьте)\s+бесплатн\w+\s+смс|нажмите\s+(?:1|2|один|два|цифру)",
    re.I,
)

COMPLIANCE_PREAMBLE_RE = re.compile(
    r"вас\s+приветствует\s+компан|все\s+разговоры\s+записываются|"
    r"ваш\s+звонок\s+очень\s+важен|звонок\s+может\s+быть\s+записан",
    re.I,
)

THIRD_PARTY_IVR_RE = re.compile(
    r"(?:сбербанк|тинькофф|альфа[-\s]?банк|мтс\s+банк|мегафон)[^.]{0,90}"
    r"(?:голосов\w+|помощник|ассистент|вас\s+приветствует|нажмите|кредитн\w+|"
    r"потребительск\w+\s+кредит|коллекторск\w+)|"
    r"(?:голосов\w+|помощник|ассистент|вас\s+приветствует|нажмите)[^.]{0,90}"
    r"(?:сбербанк|тинькофф|альфа[-\s]?банк|мтс\s+банк|мегафон)|"
    r"групп[ауы]\s+компан(?:ии|ий)\s+мтс|"
    r"целевые\s+финансы|7\s*sky|сервис\s+резерв|актив\s+бизнес\s+консалт|"
    r"active\s+business\s+consult|ооо\s+пко|филберт|капусто|екапусто|хартия|"
    r"коллекторск\w+\s+организац|действующ\w+\s+в\s+интересах|"
    r"кредитн(?:ые|ых)\s+каникул|потребительск\w+\s+кредит|"
    r"для\s+(?:улучшения|повышения)\s+качества\s+обслуживания|"
    r"вас\s+приветствует\s+компан",
    re.I,
)

VIRTUAL_SECRETARY_RE = re.compile(
    r"(?:я|это|на\s+связи\s+я|вы\s+говорите\s+с)\s+секретар[ьеё]м?|"
    r"секретар[ьяь]\s+(?:ева|мия|миа|ния)|(?:ева|мия|миа|ния)[,\s]+секретар[ьяь]|"
    r"голосов(?:ой|ая)\s+(?:ассистент|помощник|помощница)|"
    r"ассистент\s+(?:миа|мия|ния)|"
    r"временно\s+попросили\s+отвечать|попросили\s+отвечать\s+на\s+звонки|"
    r"передам\s+(?:ему|ей|абоненту|ваше|ваши)|я\s+вс[её]\s+запишу|"
    r"искусственн\w+\s+интеллект|улучшить\s+свои\s+алгоритм",
    re.I,
)

NO_LIVE_RE = re.compile(
    SYSTEM_NO_DIALOGUE_RE.pattern
    + r"|"
    + r"автоответчик|автоматическ\w+\s+сообщени|робот\s+сообщил|"
    + r"попробуйте\s+перезвонить\s+позднее|нужно\s+перезвонить\s+позже|"
    + r"линия\s+занята|связь\s+оборвалась|звонок\s+сброшен",
    re.I,
)

HARD_NO_LIVE_RE = re.compile(
    r"голосов(?:ая|ой)\s+(?:почта|почтовый\s+ящик)|"
    r"звонок\s+(?:был\s+)?(?:перенаправлен|переведен)(?:\s+на\s+голосов)?|"
    r"остав(?:ить|ьте)\s+сообщени|после\s+звуков(?:ого)?\s+сигнал|"
    r"абонент(?:\s+сейчас)?\s+(?:не\s+может|не\s+отвечает|не\s+ответил|недоступен|временно\s+недоступен)|"
    r"вызываемый\s+абонент|абонент\s+занят|номер\s+(?:недоступен|не\s+отвечает)|"
    r"не\s+отвечает\s+на\s+ваш\s+звонок|не\s+может\s+принять\s+ваш\s+звонок|"
    r"вне\s+зоны\s+действия|находится\s+вне\s+зоны|телефон\s+(?:выключен|занят|разряжен)|"
    r"недозвон|контакт\s+не\s+состоя|клиент\s+не\s+ответил|"
    r"не\s+удалось\s+(?:связаться|дозвониться|поговорить)|"
    r"живого\s+диалога\s+не\s+было|разговора\s+с\s+клиент[а-я]*\s+не\s+было|"
    r"отправ(?:ить|ьте)\s+бесплатн\w+\s+смс",
    re.I,
)

BRIDGE_SYSTEM_RE = re.compile(
    r"продолжаем\s+дозваниваться|оставайтесь\s+на\s+линии",
    re.I,
)

VOICE_MAIL_RE = re.compile(
    r"голосов(?:ая|ой)\s+(?:почта|почтовый\s+ящик)|"
    r"остав(?:ить|ьте)\s+сообщени|после\s+звуков(?:ого)?\s+сигнал",
    re.I,
)

OUTBOUND_VOICEMAIL_RE = re.compile(
    r"голосов(?:ая|ой)\s+(?:почта|почтовый\s+ящик)|"
    r"остав(?:ить|ьте)\s+сообщени|после\s+звуков(?:ого)?\s+сигнал|"
    r"абонент(?:\s+сейчас)?\s+(?:не\s+может|не\s+отвечает|не\s+ответил|недоступен|временно\s+недоступен)|"
    r"вызываемый\s+абонент|абонент\s+занят|номер\s+недоступен|вне\s+зоны\s+действия|телефон\s+выключен|"
    r"попробуйте\s+перезвонить\s+позднее|отправ(?:ить|ьте)\s+бесплатн\w+\s+смс|нажмите\s+1",
    re.I,
)

ASR_ARTIFACT_RE = re.compile(
    r"DimaTorzok|субтитры\s+сделал|редактор\s+субтитров|продолжение\s+следует|"
    r"спасибо\s+за\s+просмотр|динамичн\w+\s+музык|вес[её]л\w+\s+музык|"
    r"тревожн\w+\s+музык|музык\w+\s+играет|телефон\s+звонит|звонок\s+в\s+дверь|"
    r"\bKim(?:\s+Kim){2,}\b|\bOl[áa]\b|\bhola\b|\bvoc[êe]\b|Norske\s+Lagerforskning|"
    r"Thank\s+you\s+for\s+watching|thanks\s+for\s+watching",
    re.I,
)

RISKY_KEYWORD_RE = re.compile(
    r"перезвон|созвон|позвон|почт|email|e-mail|сообщени|недоступ|остав|"
    r"сигнал|автоответ|голосов|сброшен|занят",
    re.I,
)

LIVE_DIALOGUE_RE = re.compile(
    r"клиент\s+(?:сказал|сообщил|подтвердил|спросил|уточнил|интересуется|попросил|согласился|"
    r"сомневается|возражал|отказался|готов|планирует)|"
    r"родител[ья]\s+(?:сказал|сообщил|спросил|уточнил|интересуется|попросил|готов)|"
    r"обсудили|договорились|согласовали|выяснили|уточнили|подобрали|предложил|"
    r"оплат|стоимост|цена|скидк|рассрочк|счет|чек|договор|"
    r"курс|заняти|урок|преподавател|расписан|перенос|"
    r"математ|физик|информат|хими|биолог|английск|русск\w+\s+язык|"
    r"\bегэ\b|\bогэ\b|олимпиад|класс|реб[её]нок|сын|дочь|лагер|смена|интенсив",
    re.I,
)

NEGATIVE_NON_CONTENTFUL_CONTEXT_RE = re.compile(
    r"(?:содержательн\w+|осмысленн\w+)\s+(?:разговор|диалог|обсуждени\w*)[^.]{0,80}\s+не\s+"
    r"(?:было|произош\w*|состоял\w*|состоялос\w*|развил\w*|установлен\w*)|"
    r"разговор\s+не\s+состоял\w*|разговор[а]?\s+по\s+существу\s+не\s+был\w*|"
    r"диалог\s+не\s+развил\w*|"
    r"без\s+сформулированного\s+запроса|"
    r"тема\s+(?:обращения\s+)?не\s+(?:была\s+)?(?:раскрыта|ясна)|"
    r"запрос[^.]{0,80}\s+не\s+(?:выявлен|обсуждал\w*|зафиксирован)|"
    r"интерес[^.]{0,80}\s+не\s+(?:подтвердил\w*|выявлен|зафиксирован)|"
    r"потребност\w*[^.]{0,80}\s+не\s+выявлен\w*|"
    r"не\s+относится\s+к\s+edtech|нецелев\w+|неразборчив\w*",
    re.I,
)

BUSINESS_TERM_RE = re.compile(
    r"оплат|стоимост|цена|скидк|рассрочк|курс|заняти|урок|преподавател|расписан|"
    r"математ|физик|информат|хими|биолог|английск|русск\w+\s+язык|"
    r"\bегэ\b|\bогэ\b|олимпиад|класс|реб[её]нок|сын|дочь|лагер|смена|интенсив",
    re.I,
)

EDTECH_KEYWORD_RE = re.compile(
    r"курс|заняти|урок|преподавател|расписан|"
    r"математ|физик|информат|хими|биолог|английск|русск\w+\s+язык|"
    r"\bегэ\b|\bогэ\b|олимпиад|класс|реб[её]нок|сын|дочь|лагер|смена|интенсив|"
    r"искусственн\w+\s+интеллект|летн\w+\s+школ|выездн\w+\s+школ",
    re.I,
)

TRANSFER_OFFER_RE = re.compile(
    r"(?:соедин|переключ|перевед)[^.]{0,100}(?:коллег|специалист|администратор|менеджер|отдел)|"
    r"(?:коллег|специалист|администратор|менеджер)[^.]{0,100}(?:соедин|переключ|перевед)|"
    r"оставайтесь\s+на\s+линии",
    re.I,
)

CLIENT_TRANSFER_CONSENT_RE = re.compile(
    r"\b(?:давайте|хорошо|да(?:[-\s]?да)?|угу|слушаю|согласн\w*|можно|переключ(?:айте|ите)|соедин(?:яйте|ите))\b",
    re.I,
)

LIVE_OPT_OUT_RE = re.compile(
    r"случайн\w+\s+нажал|ошибочн\w+|не\s+надо|не\s+нужно|не\s+получается|не\s+актуальн\w+|"
    r"не\s+желаете|отказал\w*|спасибо\s+за\s+обратн\w+\s+связ",
    re.I,
)

SERVICE_ATTEMPT_RE = re.compile(
    r"пропущенн\w+\s+вызов|чем\s+могу\s+помочь|вы\s+нам\s+звонили|от\s+вас\s+звон",
    re.I,
)

CLIENT_HUMAN_RESPONSE_RE = re.compile(
    r"\b(?:алло|здравствуйте|добрый\s+день|слушаю|слышу|говорите|да|нет|повторите|"
    r"не\s+слышу|я\s+вас\s+слышу|минут[ауые]?|удобно|неудобно)\b",
    re.I,
)


def _has_live_education_context(combined: str, client_text: str) -> bool:
    return (
        bool(CLIENT_HUMAN_RESPONSE_RE.search(client_text))
        and bool(BUSINESS_TERM_RE.search(combined) or EDTECH_KEYWORD_RE.search(combined))
        and not bool(
            re.search(
                r"голосов\w+\s+помощник|нажмите|кредитн\w+|коллекторск\w+|секретар[ьяь]",
                client_text,
                re.I,
            )
        )
    )

SPEAKER_LINE_RE = re.compile(
    r"^\s*(?:\[[^\]]+\]\s*)?"
    r"(?P<speaker>MANAGER|CLIENT|Менеджер|Клиент|Оператор)\s*:?\s*"
    r"(?P<text>.*)$|"
    r"^\s*(?:\[[^\]]+\]\s*)?"
    r"(?P<speaker_colon>Абонент)\s*:\s*"
    r"(?P<text_colon>.*)$",
    re.I,
)


@dataclass(frozen=True)
class NonConversationSignals:
    label: str
    score: int
    reason_codes: tuple[str, ...]
    strong_no_live_marker: bool
    asr_artifact_marker: bool
    system_no_dialogue_phrase: bool
    risky_keyword_marker: bool
    live_dialogue_evidence_score: int
    protected_live_dialogue: bool
    manager_chars: int
    client_chars: int
    transcript_chars: int
    outbound_voicemail_marker: bool
    should_force_non_conversation: bool
    requires_manual_review: bool
    recommended_call_type: str | None
    recommended_contentful: bool | None
    recommended_contact_subtype: str | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reason_codes"] = list(self.reason_codes)
        return payload


def detect_non_conversation_signals(
    text: str = "",
    *,
    history_summary: str = "",
    transcript_text: str = "",
    call_type: str = "",
    next_step: str = "",
    products: Iterable[str] | str | None = None,
    subjects: Iterable[str] | str | None = None,
    objections: Iterable[str] | str | None = None,
    duration_sec: float | int | None = None,
) -> NonConversationSignals:
    """Classify no-live risk without changing source analysis.

    The rule order is conservative:
    - high-confidence non-conversation only when no-live/artifact signals are
      strong and live-dialogue evidence is not protected;
    - ambiguous rows are manual review, not automatic deletion;
    - service/technical/sales calls with real business dialogue are protected.
    """

    transcript = _clean(transcript_text or text)
    history = _clean(history_summary)
    step = _clean(next_step)
    combined = _join_text(transcript, history, step)
    structured_blob = _join_text(
        *_as_list(products),
        *_as_list(subjects),
        *_as_list(objections),
    )
    manager_text, client_text = _split_speaker_text(transcript)
    manager_chars = len(manager_text)
    client_chars = len(client_text)
    transcript_chars = len(transcript)

    third_party_ivr_raw = bool(THIRD_PARTY_IVR_RE.search(combined))
    virtual_secretary = bool(VIRTUAL_SECRETARY_RE.search(client_text))
    hard_no_live = bool(HARD_NO_LIVE_RE.search(combined))
    live_payment_context = (
        third_party_ivr_raw
        and bool(re.search(r"оплат|ссылк|реквизит|договор|чек|налогов\w+\s+вычет", combined, re.I))
        and bool(CLIENT_HUMAN_RESPONSE_RE.search(client_text))
        and len(client_text) >= 80
        and not bool(re.search(r"вас\s+приветствует|голосов\w+\s+помощник|нажмите|кредитн\w+|коллекторск\w+", client_text, re.I))
    )
    live_education_context = third_party_ivr_raw and _has_live_education_context(combined, client_text)
    third_party_business_dialogue = (
        third_party_ivr_raw
        and ((duration_sec is not None and _safe_float(duration_sec) >= 60) or (duration_sec is None and transcript_chars >= 500))
        and manager_chars >= 120
        and client_chars >= 120
        and not hard_no_live
        and not virtual_secretary
        and not bool(
            re.search(
                r"голосов\w+\s+(?:помощник|ассистент)|"
                r"после\s+звуков(?:ого)?\s+сигнал|остав(?:ить|ьте)\s+сообщени|"
                r"абонент\s+(?:не\s+может|недоступен|не\s+отвечает)",
                client_text,
                re.I,
            )
        )
        and bool(
            CLIENT_HUMAN_RESPONSE_RE.search(client_text)
            or re.search(r"номер|баз[аеуы]|данн\w*|провер|звонк|обращени|удал|почему|не\s+подтверд", client_text, re.I)
        )
    )
    third_party_ivr = third_party_ivr_raw and not (
        live_payment_context or live_education_context or third_party_business_dialogue
    )
    repeated_loop = _has_repeated_phrase_loop(transcript)
    strong_no_live = bool(NO_LIVE_RE.search(combined))
    bridge_system = bool(BRIDGE_SYSTEM_RE.search(combined))
    system_no_dialogue = bool(SYSTEM_NO_DIALOGUE_RE.search(combined))
    asr_artifact = bool(ASR_ARTIFACT_RE.search(combined)) or repeated_loop
    client_asr_artifact = bool(ASR_ARTIFACT_RE.search(client_text)) or repeated_loop
    risky_keyword = bool(RISKY_KEYWORD_RE.search(combined))
    client_has_system_text = bool(SYSTEM_NO_DIALOGUE_RE.search(client_text))
    manager_has_system_text = bool(SYSTEM_NO_DIALOGUE_RE.search(manager_text))
    client_has_business_terms = bool(BUSINESS_TERM_RE.search(client_text)) and not third_party_ivr
    manager_has_business_terms = bool(BUSINESS_TERM_RE.search(manager_text))
    transcript_edtech_hits = _keyword_hit_count(transcript, EDTECH_KEYWORD_RE)
    client_human_response = (
        bool(CLIENT_HUMAN_RESPONSE_RE.search(client_text))
        and not client_has_system_text
        and not third_party_ivr
        and not virtual_secretary
    )
    negative_non_contentful_context = bool(NEGATIVE_NON_CONTENTFUL_CONTEXT_RE.search(history))
    history_has_live_evidence = bool(LIVE_DIALOGUE_RE.search(history)) and not negative_non_contentful_context
    transcript_has_live_evidence = bool(LIVE_DIALOGUE_RE.search(transcript))
    client_has_live_evidence = bool(LIVE_DIALOGUE_RE.search(client_text)) and not client_has_system_text and not third_party_ivr
    structured_has_signal = bool(structured_blob.strip())
    duration = _safe_float(duration_sec)
    outbound_voicemail = (
        bool(OUTBOUND_VOICEMAIL_RE.search(client_text))
        and client_has_system_text
        and not client_has_business_terms
        and (manager_chars >= 40 or manager_has_business_terms)
    )

    score = 0
    reasons: list[str] = []

    if client_chars >= 80 and not client_has_system_text:
        score += 3
        reasons.append("client_turn_80_plus")
    elif client_chars >= 30 and not client_has_system_text:
        score += 1
        reasons.append("client_turn_30_plus")

    if client_has_business_terms:
        score += 2
        reasons.append("client_business_terms")
    if transcript_has_live_evidence and not system_no_dialogue:
        score += 2
        reasons.append("transcript_live_evidence")
    if history_has_live_evidence:
        score += 2
        reasons.append("history_live_evidence")
    if structured_has_signal and not system_no_dialogue:
        score += 1
        reasons.append("structured_fields_present")
    if duration is not None and duration >= 90 and transcript_chars >= 500 and not system_no_dialogue:
        score += 1
        reasons.append("long_call")

    if system_no_dialogue:
        score -= 4
        reasons.append("system_no_dialogue_phrase")
    if negative_non_contentful_context:
        score -= 2
        reasons.append("negative_non_contentful_context")
    if strong_no_live:
        score -= 2
        reasons.append("no_live_marker")
    if asr_artifact:
        score -= 3
        reasons.append("asr_artifact_marker")
    if repeated_loop:
        score -= 3
        reasons.append("asr_loop_marker")
    if third_party_ivr:
        score -= 4
        reasons.append("third_party_ivr")
    if virtual_secretary:
        score -= 3
        reasons.append("virtual_secretary")
    if manager_has_system_text and (client_chars < 30 or client_has_system_text):
        score -= 2
        reasons.append("manager_side_system_leak")
    if transcript_chars < 160 and strong_no_live:
        score -= 1
        reasons.append("short_no_live_transcript")
    if outbound_voicemail:
        score -= 2
        reasons.append("outbound_voicemail")

    bridge_live_context = (
        bridge_system
        and not hard_no_live
        and manager_chars >= 40
        and client_chars >= 30
        and (transcript_has_live_evidence or history_has_live_evidence)
    )
    if bridge_live_context:
        reasons.append("bridge_live_dialogue")

    live_client_signal = (
        client_human_response
        or client_has_business_terms
        or (client_chars >= 30 and history_has_live_evidence and not virtual_secretary and not third_party_ivr)
    )
    transfer_after_live_dialogue = (
        bool(TRANSFER_OFFER_RE.search(manager_text))
        and bool(CLIENT_TRANSFER_CONSENT_RE.search(client_text))
        and (duration is not None and duration >= 40)
        and client_chars >= 20
    )
    long_client_live_safeguard = (
        ((duration is not None and duration > 60) or (duration is None and transcript_chars >= 500))
        and client_chars > 150
        and (
            live_client_signal
            or transcript_has_live_evidence
            or history_has_live_evidence
            or transcript_edtech_hits >= 2
        )
        and not (client_has_system_text and not (client_has_business_terms or history_has_live_evidence))
    )
    edtech_live_safeguard = (
        client_chars > 100
        and transcript_edtech_hits >= 2
        and (live_client_signal or history_has_live_evidence)
        and not virtual_secretary
        and not third_party_ivr
        and not (client_has_system_text and not client_has_business_terms)
    )
    proxy_parent_safeguard = (
        client_chars > 200
        and transcript_edtech_hits >= 1
        and (live_client_signal or history_has_live_evidence)
        and not virtual_secretary
        and not third_party_ivr
        and not (client_has_system_text and not client_has_business_terms)
    )
    sales_live_safeguard = (
        _clean(call_type) == "sales_call"
        and duration is not None
        and duration > 30
        and client_chars > 100
        and (live_client_signal or client_has_business_terms or history_has_live_evidence)
        and not (client_has_system_text and not client_has_business_terms)
    )
    third_party_ivr_after_live_safeguard = (
        (third_party_ivr or third_party_business_dialogue)
        and history_has_live_evidence
        and duration is not None
        and duration > 60
        and client_chars > 200
    )
    live_opt_out_safeguard = (
        _clean(call_type) in {"sales_call", "service_call", "technical_call", "existing_client_progress"}
        and duration is not None
        and duration >= 20
        and (history_has_live_evidence or bool(LIVE_OPT_OUT_RE.search(_join_text(transcript, history))))
        and bool(LIVE_OPT_OUT_RE.search(_join_text(transcript, history)))
        and not third_party_ivr
        and not virtual_secretary
        and not (client_has_system_text and not client_has_business_terms)
    )
    ambiguous_service_attempt_safeguard = (
        _clean(call_type) in {"sales_call", "service_call", "existing_client_progress"}
        and client_chars >= 30
        and bool(SERVICE_ATTEMPT_RE.search(manager_text))
        and (asr_artifact or negative_non_contentful_context or not client_human_response)
        and not (client_asr_artifact and not client_human_response and negative_non_contentful_context)
        and not strong_no_live
        and not system_no_dialogue
        and not third_party_ivr
        and not virtual_secretary
    )
    manual_safeguard = any(
        (
            transfer_after_live_dialogue,
            long_client_live_safeguard,
            edtech_live_safeguard,
            proxy_parent_safeguard,
            sales_live_safeguard,
            third_party_ivr_after_live_safeguard,
            third_party_business_dialogue,
            live_opt_out_safeguard,
            ambiguous_service_attempt_safeguard,
        )
    )
    force_manual_safeguard = any(
        (
            transfer_after_live_dialogue,
            third_party_ivr_after_live_safeguard,
            third_party_business_dialogue,
            live_opt_out_safeguard,
            ambiguous_service_attempt_safeguard,
        )
    )
    if transfer_after_live_dialogue:
        reasons.append("safeguard_transfer_after_live_dialogue")
    if long_client_live_safeguard:
        reasons.append("safeguard_long_client_live_turn")
    if edtech_live_safeguard:
        reasons.append("safeguard_edtech_live_turn")
    if proxy_parent_safeguard:
        reasons.append("safeguard_proxy_parent_live_turn")
    if sales_live_safeguard:
        reasons.append("safeguard_sales_live_turn")
    if third_party_ivr_after_live_safeguard:
        reasons.append("safeguard_third_party_ivr_after_live")
    if third_party_business_dialogue:
        reasons.append("safeguard_third_party_business_dialogue")
    if live_opt_out_safeguard:
        reasons.append("safeguard_live_opt_out")
    if ambiguous_service_attempt_safeguard:
        reasons.append("safeguard_ambiguous_service_attempt")

    protected = bridge_live_context or _is_protected_live_dialogue(
        score=score,
        call_type=call_type,
        system_no_dialogue=system_no_dialogue,
        asr_artifact=asr_artifact,
        strong_no_live=strong_no_live,
        negative_non_contentful_context=negative_non_contentful_context,
        structured_has_signal=structured_has_signal,
        client_chars=client_chars,
        client_has_business_terms=client_has_business_terms,
        transcript_has_live_evidence=transcript_has_live_evidence,
        history_has_live_evidence=history_has_live_evidence,
    )

    high_confidence = False
    if not protected and not manual_safeguard:
        # Automatic rewrite is allowed only for the safest class caught in the
        # audit: explicit no-live/system voicemail plus recognizable ASR junk,
        # explicit system no-live with no client business/live evidence, or
        # outbound voicemail where manager spoke into a system prompt but the
        # client never joined the dialogue.
        client_no_live_only = (
            client_has_system_text
            and strong_no_live
            and not client_has_business_terms
            and not client_has_live_evidence
        )
        manager_system_leak_only = (
            manager_has_system_text
            and not client_has_business_terms
            and not client_has_live_evidence
            and (client_chars < 30 or client_has_system_text)
        )
        artifact_only = (
            asr_artifact
            and not client_has_business_terms
            and not client_has_live_evidence
            and not client_human_response
            and not (transcript_has_live_evidence and not strong_no_live and not system_no_dialogue)
            and not (
                history_has_live_evidence
                and _clean(call_type) in {"sales_call", "service_call", "technical_call", "existing_client_progress"}
                and not strong_no_live
                and not system_no_dialogue
            )
            and (client_chars < 120 or strong_no_live or negative_non_contentful_context)
        )
        high_confidence = (
            (asr_artifact and (strong_no_live or system_no_dialogue))
            or outbound_voicemail
            or third_party_ivr
            or virtual_secretary
            or client_no_live_only
            or manager_system_leak_only
            or artifact_only
            or (
                system_no_dialogue
                and strong_no_live
                and score <= -4
                and not client_has_business_terms
                and not client_has_live_evidence
            )
            or (
                negative_non_contentful_context
                and strong_no_live
                and not client_has_business_terms
                and not client_has_live_evidence
                and client_chars < 40
            )
        )

    if high_confidence:
        label = LABEL_NON_CONVERSATION_HIGH_CONFIDENCE
        should_force = True
        manual = False
        recommended_call_type = "non_conversation"
        recommended_contentful = False
        recommended_contact_subtype = "outbound_voicemail" if outbound_voicemail else "no_live_or_voicemail"
    elif force_manual_safeguard:
        label = LABEL_MANUAL_REVIEW_BORDERLINE_LIVE_CONTEXT
        should_force = False
        manual = True
        recommended_call_type = None
        recommended_contentful = None
        recommended_contact_subtype = "borderline_live_context"
    elif protected:
        label = LABEL_CONTENTFUL_PROTECTED_LIVE_DIALOGUE
        should_force = False
        manual = False
        recommended_call_type = _clean(call_type) or None
        recommended_contentful = True
        recommended_contact_subtype = None
        reasons.append("protected_live_dialogue")
    elif manual_safeguard:
        label = LABEL_MANUAL_REVIEW_BORDERLINE_LIVE_CONTEXT
        should_force = False
        manual = True
        recommended_call_type = None
        recommended_contentful = None
        recommended_contact_subtype = "borderline_live_context"
    elif strong_no_live and score <= 1:
        label = LABEL_MANUAL_REVIEW_PROBABLE_NO_LIVE
        should_force = False
        manual = True
        recommended_call_type = None
        recommended_contentful = None
        recommended_contact_subtype = "probable_no_live"
    elif strong_no_live or (risky_keyword and system_no_dialogue):
        label = LABEL_MANUAL_REVIEW_BORDERLINE_LIVE_CONTEXT
        should_force = False
        manual = True
        recommended_call_type = None
        recommended_contentful = None
        recommended_contact_subtype = "borderline_no_live"
    else:
        label = LABEL_CONTENTFUL_LOW_RISK
        should_force = False
        manual = False
        recommended_call_type = _clean(call_type) or None
        recommended_contentful = None
        recommended_contact_subtype = None

    return NonConversationSignals(
        label=label,
        score=score,
        reason_codes=tuple(dict.fromkeys(reasons)),
        strong_no_live_marker=strong_no_live,
        asr_artifact_marker=asr_artifact,
        system_no_dialogue_phrase=system_no_dialogue,
        risky_keyword_marker=risky_keyword,
        live_dialogue_evidence_score=score,
        protected_live_dialogue=protected,
        manager_chars=manager_chars,
        client_chars=client_chars,
        transcript_chars=transcript_chars,
        outbound_voicemail_marker=outbound_voicemail,
        should_force_non_conversation=should_force,
        requires_manual_review=manual,
        recommended_call_type=recommended_call_type,
        recommended_contentful=recommended_contentful,
        recommended_contact_subtype=recommended_contact_subtype,
    )


def classify_transcript_quality(*args: Any, **kwargs: Any) -> NonConversationSignals:
    return detect_non_conversation_signals(*args, **kwargs)


def blocks_email_from_voice_mail(text: str) -> bool:
    """Return True when `почта` most likely means voice mail, not email."""

    return bool(VOICE_MAIL_RE.search(_clean(text)))


def blocks_system_next_step(text: str) -> bool:
    """Return True for system-generated callback prompts from failed calls."""

    signals = detect_non_conversation_signals(text)
    return signals.should_force_non_conversation or (
        signals.strong_no_live_marker and not signals.protected_live_dialogue and signals.score <= 1
    )


def _is_protected_live_dialogue(
    *,
    score: int,
    call_type: str,
    system_no_dialogue: bool,
    asr_artifact: bool,
    strong_no_live: bool,
    negative_non_contentful_context: bool,
    structured_has_signal: bool,
    client_chars: int,
    client_has_business_terms: bool,
    transcript_has_live_evidence: bool,
    history_has_live_evidence: bool,
) -> bool:
    if negative_non_contentful_context:
        return False
    artifact_tail_after_live_dialogue = (
        asr_artifact
        and (system_no_dialogue or strong_no_live)
        and history_has_live_evidence
        and transcript_has_live_evidence
        and client_chars >= 60
        and (client_has_business_terms or structured_has_signal)
    )
    if artifact_tail_after_live_dialogue:
        return True
    artifact_inside_contentful_dialogue = (
        asr_artifact
        and not system_no_dialogue
        and not strong_no_live
        and history_has_live_evidence
        and _clean(call_type) in {"sales_call", "service_call", "technical_call", "existing_client_progress"}
    )
    if artifact_inside_contentful_dialogue:
        return True
    if system_no_dialogue or asr_artifact:
        return False
    if _clean(call_type) in {"sales_call", "service_call", "technical_call", "existing_client_progress"}:
        if history_has_live_evidence and score >= 2:
            return True
    if score >= 4 and (client_chars >= 30 or transcript_has_live_evidence or history_has_live_evidence):
        return True
    if client_chars >= 80 and client_has_business_terms:
        return True
    if _clean(call_type) in {"sales_call", "service_call", "technical_call", "existing_client_progress"}:
        return score >= 3 and (transcript_has_live_evidence or history_has_live_evidence)
    return False


def _split_speaker_text(text: str) -> tuple[str, str]:
    manager_parts: list[str] = []
    client_parts: list[str] = []
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = SPEAKER_LINE_RE.match(line)
        if match:
            speaker = (match.group("speaker") or match.group("speaker_colon") or "").lower()
            current = "client" if speaker in {"client", "клиент", "абонент"} else "manager"
            line_text = (match.group("text") or match.group("text_colon") or "").strip()
            if not line_text:
                continue
        else:
            line_text = line
        if current == "client":
            client_parts.append(line_text)
        elif current == "manager":
            manager_parts.append(line_text)
    return " ".join(manager_parts), " ".join(client_parts)


def _as_list(value: Iterable[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [_clean(item) for item in value if _clean(item)]


def _join_text(*parts: str) -> str:
    return " ".join(_clean(part) for part in parts if _clean(part))


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _has_repeated_phrase_loop(text: str) -> bool:
    """Detect obvious ASR hallucination loops without classifying normal repeats."""

    words = re.findall(r"[а-яёa-z0-9]{3,}", _clean(text).lower(), re.I)
    if len(words) < 8:
        return False
    # Catch one-word loops and short phrase loops like "папочка попал на меня"
    # repeated several times on silence/noise.
    for size in range(1, 6):
        limit = len(words) - size * 4 + 1
        if limit < 0:
            continue
        for start in range(limit):
            phrase = words[start : start + size]
            if not phrase:
                continue
            repeats = 1
            cursor = start + size
            while cursor + size <= len(words) and words[cursor : cursor + size] == phrase:
                repeats += 1
                cursor += size
            if repeats >= 4:
                return True
    return False


def _keyword_hit_count(text: str, pattern: re.Pattern[str]) -> int:
    return len({match.group(0).lower() for match in pattern.finditer(_clean(text))})


def _safe_float(value: float | int | str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
