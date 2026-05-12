from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CrmTextQualityFinding:
    class_id: str
    risk_type: str
    severity: str
    field: str
    matched_text: str
    reason: str
    row_index: int | None = None


TARGET_CRM_TEXT_FIELDS = (
    "Авто история общения",
    "Последняя AI-сводка",
    "AI-краткая сводка клиента",
    "AI-история общения",
    "AI-активные сделки клиента",
    "AI-учебный контекст Tallanto",
    "AI-финансы Tallanto",
    "AI-важные риски клиента",
    "Краткая история общения",
    "Хронология общения (последние 5 касаний)",
    "Краткое резюме последнего свежего звонка",
    "Краткое резюме разговора",
    "Возражения",
    "Следующий шаг",
    "AI-рекомендованный следующий шаг",
    "AI-сводка по сделке",
    "AI-история по сделке",
    "AI-фактический статус сделки",
    "AI-актуальные возражения",
    "AI-основание рекомендации",
    "AI-качество привязки к сделке",
    "AI-предупреждение по сделке",
    "AI-Tallanto статус по сделке",
)
AUTO_HISTORY_FIELDS = ("Авто история общения", "auto_history", "autoHistory")
OBJECTION_FIELDS = (
    "Возражения",
    "Ограничения/возражения",
    "Актуальные ограничения",
    "objections",
    "current_objections",
)
NEXT_STEP_FIELDS = (
    "Следующий шаг",
    "AI-рекомендованный следующий шаг",
    "recommended_next_step",
    "next_step",
    "next_step_action",
)
FOLLOWUP_DATE_FIELDS = (
    "Рекомендуемая дата следующего контакта",
    "AI-дата следующего касания",
    "recommended_followup_date",
    "follow_up_due_at",
)
LAST_CALL_DATE_FIELDS = (
    "Дата последнего свежего звонка",
    "Дата последнего звонка",
    "last_call_at",
    "latest_call_at",
)
PRIORITY_FIELDS = ("Приоритет лида", "AI-приоритет", "priority", "lead_priority")
PROBABILITY_FIELDS = ("Вероятность продажи, %", "sale_probability_pct", "probability")
LOSS_REASON_FIELDS = (
    "Причина отказа (лид)",
    "Причина отказа (B2C)",
    "AMO причина отказа",
    "loss_reason",
)

ELLIPSIS_RE = re.compile(r"\.\.\.|…")
COUNTED_LABEL_RE = re.compile(r"^(?P<label>[^:|]{1,90}?)\s*:\s*(?P<count>\d{1,5})(?:\s+\w+)?$", re.I)
CONCRETE_DATE_RE = re.compile(
    r"\b\d{4}-\d{1,2}-\d{1,2}\b|"
    r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b|"
    r"\b\d{1,2}\s+"
    r"(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b",
    re.I,
)
MONTH_ONLY_RE = re.compile(
    r"\bв\s+(?:январе|феврале|марте|апреле|мае|июне|июле|августе|сентябре|октябре|ноябре|декабре)\b",
    re.I,
)
WEAK_FILLER_LABELS = {"время", "доверие", "цена", "неудобно"}
STRONG_NEGATIVE_LABEL_RE = re.compile(
    r"\b(?:неактуально|не\s+актуально|отказ(?:ался|алась|ались)?|"
    r"не\s*интересно|неинтересно|не\s+беспокоить|не\s+звонить|закрыть)\b",
    re.I,
)
HISTORICAL_EVIDENCE_RE = re.compile(
    r"\b(?:историческ\w+|ранее|раньше|стар(?:ое|ый|ые)?|было|снято|сняли)\b|"
    r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b",
    re.I,
)
SALES_NEXT_STEP_RE = re.compile(
    r"\b(?:оплат\w*|ссылк\w+\s+на\s+оплат\w*|плат[её]ж\w*|счет\w*|сч[её]т\w*|"
    r"запис\w*|брон\w*|договор\w*|продолж\w*|перезвон\w*|созвон\w*|"
    r"отправ\w*|подтверд\w*)\b",
    re.I,
)
CLOSURE_NEXT_STEP_RE = re.compile(
    r"\b(?:не\s+беспокоить|не\s+звонить|не\s+продолжать|отменить(?:\s+запись)?|"
    r"снять\s+заявк\w+|убрать\s+из\s+списк\w+|оставить\s+клиент\w+\s+без|"
    r"ждать\s+обращени\w+|ожидать\s+решени\w+|ожидать,\s*пока\s+клиент\w+|дождаться|"
    r"клиент\s+сам\s+(?:обратится|свяжется|перезвонит)|"
    r"сам[аи]?\s+(?:свяжется|перезвонит)|закрыть|отказ)\b",
    re.I,
)
VAGUE_NEXT_STEP_RE = re.compile(
    r"\b(?:связаться|созвониться|перезвонить|вернуться|поговорить)\s+(?:позже|потом)\b|"
    r"\b(?:связаться|созвониться|перезвонить|вернуться|поговорить)\s+(?:в|во|ближе\s+к|к)\s+"
    r"(?:январ[юе]|феврал[юе]|март[уе]|апрел[юе]|ма[юе]|июн[юе]|июл[юе]|август[уе]|"
    r"сентябр[юе]|октябр[юе]|ноябр[юе]|декабр[юе]|летом|осенью|зимой|весной)\b|"
    r"\b(?:связаться|созвониться|перезвонить|вернуться|поговорить)\s+через\s+пару\s+недель\b|"
    r"\b(?:летом|осенью|зимой|весной|в\s+следующем\s+году)\b|"
    r"\bвернуться\s+при\s+изменени\w+\s+решени\w+\b|"
    r"\b(?:ждать|ожидать)\s+(?:обращени\w+|обновлени\w+|решени\w+|выбор\w+|"
    r"определени\w+|ответ\w+|обратн\w+\s+связ\w+)\b|"
    r"\b(?:при\s+необходимости|если\s+понадобится|если\s+будет\s+актуально)\b",
    re.I,
)
LOST_LEAD_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"(?:уже\s+)?(?:купил[аи]?|купили|приобр[её]л[аи]?|приобрели|оплатил[аи]?|оплатили)\s+"
    r"(?:программ\w+|курс\w+|обучени\w+|лагер\w+|смен\w+|пут[её]вк\w+)?\s*"
    r"(?:у|в)\s+(?:друг\w+|ин\w+|сторонн\w+)"
    r"|(?:купил[аи]?|купили|приобр[её]л[аи]?|приобрели|оплатил[аи]?|оплатили|выбрал[аи]?|выбрали)\s+"
    r"(?:у\s+)?конкур\w+"
    r"|(?:выбрал[аи]?|выбрали|остановил[аи]?сь|остановились)\s+на\s+"
    r"(?:друг\w+|ин\w+)\s+(?:лагер\w+|школ\w+|центр\w+|курс\w+|программ\w+)"
    r"|(?:уже\s+)?выбрал[а-я]*\s+друг\w+\s+школ\w+"
    r"|от\s+запис\w+\s+отказал[а-я]*"
    r"|интерес\s+закрыт\s+покупк\w+\s+у\s+конкур\w+"
    r"|дальнейш\w+\s+продолжени\w+\s+сделк\w+\s+не\s+требует\w+"
    r"|дальнейш\w+\s+(?:действи\w+|контакт\w+)[^.]{0,80}не\s+согласован\w+"
    r"|потребност\w+[^.]{0,120}не\s+актуальн\w+[^.]{0,80}покупк\w+"
    r")\b",
    re.I,
)
PASSIVE_CUSTOMER_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"(?:клиент\w*|родител\w*)\s+сам[аи]?\s+(?:свяж\w+|обрат\w+|перезвон\w+)"
    r"|сам[аи]?\s+(?:свяж\w+|обрат\w+|перезвон\w+)"
    r"|свяж\w+\s+сам[аи]?"
    r"|когда\s+определ(?:ится|ятся|имся)"
    r"|повторн\w+\s+обращени\w+\s+клиент\w+"
    r"|просил[аи]?\s+не\s+предлагать\s+активн\w+"
    r"|не\s+предлагать\s+активн\w+"
    r"|ждать\s+обращени\w+\s+клиент\w+"
    r"|ожидать\s+обращени\w+\s+клиент\w+"
    r")\b",
    re.I,
)
ACTIVE_CLIENT_LOSS_REASON_RE = re.compile(
    r"\b(?:действующ\w+\s+клиент\w*|текущ\w+\s+клиент\w*|"
    r"действующ\w+\s+ученик\w*|текущ\w+\s+ученик\w*|уже\s+учит\w+|уже\s+занима\w+)\b",
    re.I,
)
EXPLICIT_NO_NEXT_STEP_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"договор[её]нност\w+\s+о\s+следующ\w+\s+шаг\w+\s+не\s+был\w+"
    r"|следующ\w+\s+шаг\w+\s+не\s+(?:был\w+\s+)?(?:согласован|зафиксирован|определ[её]н)"
    r"|клиент\w*\s+отказал[а-я]*\s+от\s+дальнейш\w+\s+(?:интерес\w+|общени\w+|участи\w+)"
    r"|отказал[а-я]*\s+от\s+(?:лвш|летн\w+\s+выездн\w+\s+школ\w+|летн\w+\s+лагер\w+)"
    r"|отказал[а-я]*\s+из-за\s+цен\w+"
    r"|не\s+оставил[а-я]*\s+(?:вопрос\w+|договор[её]нност\w+)"
    r"|дальнейш\w+\s+интерес\w+\s+нет"
    r")\b",
    re.I,
)
WRONG_PERSON_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"контакт\s+не\s+подтвердил[а-я]*"
    r"|путаниц\w+\s+с\s+имен\w+"
    r"|на\s+лини[и]\s+был[ао]?\s+не\s+т[ао]т"
    r"|не\s+тот\s+клиент"
    r"|не\s+та\s+(?:светлана|клиентка|родительница|мама|женщина)"
    r"|обсуждени\w+\s+программ\w+[^.]{0,80}не\s+состоял\w+"
    r")\b",
    re.I,
)
COMPLETED_PAYMENT_SIGNAL_RE = re.compile(
    r"\b(?:"
    r"(?:чек|квитанц\w+|подтверждени\w+\s+оплат\w+)\s+(?:уже\s+)?(?:прислан\w*|отправлен\w*|получен\w*)"
    r"|(?:наличи\w+\s+)?чек\w+[^.]{0,80}(?:оплат\w+|сумм\w+|\d[\d\s]{3,})"
    r"|плат[её]жк\w+\s+оплатил[а-я]*"
    r"|оплатил[а-я]*\s+за\s+[А-ЯЁA-Zа-яёa-z0-9 -]{3,80}"
    r"|оплат\w+[^.]{0,80}(?:на\s+сумм\w+|\d[\d\s]{3,})"
    r"|(?:оплат\w+|плат[её]ж\w+)\s+(?:уже\s+)?(?:внес[её]н\w*|поступил\w*|получен\w*|подтвержд[её]н\w*)"
    r"|сделк\w+\s+(?:закрыт\w+|успешн\w+\s+закрыт\w+|оплачен\w+)"
    r"|запис\w+\s+(?:подтвержден\w+|подтверждена)\s+после\s+оплат\w+"
    r")\b",
    re.I,
)
RELATIVE_STALE_NEXT_STEP_RE = re.compile(
    r"\b(?:"
    r"завтра"
    r"|через\s+(?:полтора|пару|несколько|\d{1,2})\s+(?:час\w+|дн\w+|недел\w+|месяц\w+|год\w+)"
    r"|через\s+год"
    r"|до\s+конца\s+(?:дня|недели|месяца)"
    r")\b",
    re.I,
)
PAYMENT_COLLECTION_NEXT_STEP_RE = re.compile(
    r"\b(?:"
    r"оплат\w+|прислать\s+оплат\w+|отправить\s+чек|прислать\s+чек|"
    r"подписать\s+договор\s+и\s+отправить\s+чек|дождаться\s+оплат\w+|получить\s+оплат\w+"
    r")\b",
    re.I,
)
ACTIVE_STALE_NEXT_STEP_RE = re.compile(
    r"\b(?:"
    r"перезвон\w+|связ\w+|отправ\w+|уточн\w+|соедин\w+|направ\w+|"
    r"подать\s+заявк\w+|провер\w+|выслать|прислать|передать|сообщить"
    r")\b",
    re.I,
)


def detect_crm_text_quality_risks(
    payload: object,
    *,
    analysis_date: str | date | datetime | None = None,
    min_severity: str = "P3",
    compact_max_chars: int = 1200,
    verbose_max_chars: int = 1800,
) -> list[CrmTextQualityFinding]:
    """Detect Stage20 CRM text UX/semantic defects without touching writeback code."""
    allowed = _allowed_severities(min_severity)
    findings: list[CrmTextQualityFinding] = []
    row = payload if isinstance(payload, Mapping) else None

    for field, value in _iter_target_text_fields(payload):
        findings.extend(_detect_ellipsis(field, value))
        findings.extend(_detect_duplicate_label_counts(field, value))

    objection_text = _objection_text(payload)
    next_step = _next_step_text(payload)
    priority = _safe_text(_first_mapping_value(row, PRIORITY_FIELDS)) if row else _extract_labeled_value(_payload_text(payload), PRIORITY_FIELDS)
    probability = _safe_text(_first_mapping_value(row, PROBABILITY_FIELDS)) if row else _extract_labeled_value(_payload_text(payload), PROBABILITY_FIELDS)

    findings.extend(_detect_objection_labels(objection_text, next_step, priority, probability))
    findings.extend(_detect_next_step_quality(next_step, priority, probability))
    findings.extend(_detect_lost_lead_next_step_conflict(_payload_text(payload), next_step, priority, probability))
    findings.extend(_detect_passive_customer_next_step_conflict(_payload_text(payload), next_step, priority, probability))
    findings.extend(_detect_no_next_step_conflict(_payload_text(payload), next_step, priority, probability))
    findings.extend(_detect_wrong_person_conflict(_payload_text(payload), next_step, priority, probability))
    findings.extend(_detect_active_client_loss_reason_conflict(payload, next_step, priority, probability))
    findings.extend(_detect_completed_payment_next_step_conflict(_payload_text(payload), next_step, priority, probability))
    findings.extend(_detect_stale_followup_date(payload, analysis_date=analysis_date, next_step=next_step))
    findings.extend(
        _detect_verbose_manager_ux(
            payload,
            compact_max_chars=compact_max_chars,
            verbose_max_chars=verbose_max_chars,
        )
    )
    findings.extend(_detect_empty_auto_history(payload))
    findings.extend(_detect_cross_field_duplicate_information(payload))

    return [finding for finding in findings if finding.severity in allowed]


def has_blocking_crm_text_quality_risk(payload: object, *, min_severity: str = "P2", **kwargs: Any) -> bool:
    return bool(detect_crm_text_quality_risks(payload, min_severity=min_severity, **kwargs))


def findings_to_risk_counts(findings: Iterable[CrmTextQualityFinding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.risk_type] = counts.get(finding.risk_type, 0) + 1
    return counts


def crm_text_findings_to_counts(findings: Iterable[CrmTextQualityFinding]) -> dict[str, int]:
    return findings_to_risk_counts(findings)


def has_blocking_crm_text_findings(
    findings: Iterable[CrmTextQualityFinding],
    *,
    min_blocking_severity: str = "P2",
) -> bool:
    allowed = _allowed_severities(min_blocking_severity)
    return any(finding.severity in allowed for finding in findings)


def detect_crm_text_quality_batch_risks(
    rows: Sequence[Mapping[str, Any]],
    *,
    analysis_date: str | date | datetime | None = None,
    min_severity: str = "P3",
    uniform_date_ratio: float = 0.8,
    uniform_date_min_count: int = 3,
) -> list[CrmTextQualityFinding]:
    findings: list[CrmTextQualityFinding] = []
    for index, row in enumerate(rows, start=1):
        for finding in detect_crm_text_quality_risks(row, analysis_date=analysis_date, min_severity=min_severity):
            findings.append(_with_row_index(finding, index))

    date_by_index = {
        index: _normalize_date(_first_mapping_value(row, FOLLOWUP_DATE_FIELDS))
        for index, row in enumerate(rows, start=1)
    }
    non_empty_dates = [value for value in date_by_index.values() if value]
    if len(non_empty_dates) < uniform_date_min_count:
        return findings

    date_counts = Counter(non_empty_dates)
    dominant_date, dominant_count = date_counts.most_common(1)[0]
    if dominant_count < uniform_date_min_count or dominant_count / len(non_empty_dates) < uniform_date_ratio:
        return findings

    analysis_date_text = _normalize_date(analysis_date)
    if analysis_date_text and dominant_date != analysis_date_text:
        return findings

    allowed = _allowed_severities(min_severity)
    for index, date_text in date_by_index.items():
        if date_text != dominant_date:
            continue
        next_step = _next_step_text(rows[index - 1])
        if _has_concrete_date(next_step) or _mentions_today(next_step):
            continue
        finding = CrmTextQualityFinding(
            class_id="Q4b",
            risk_type="stale_uniform_followup_date",
            severity="P2",
            field=_first_present_field(rows[index - 1], FOLLOWUP_DATE_FIELDS),
            matched_text=date_text,
            reason="Recommended follow-up date is uniform across the batch without next-step date semantics",
            row_index=index,
        )
        if finding.severity in allowed and finding not in findings:
            findings.append(finding)
    return findings


def _detect_ellipsis(field: str, value: str) -> list[CrmTextQualityFinding]:
    result: list[CrmTextQualityFinding] = []
    for match in ELLIPSIS_RE.finditer(value):
        result.append(
            CrmTextQualityFinding(
                class_id="Q1",
                risk_type="lossy_ellipsis_truncation",
                severity="P1",
                field=field,
                matched_text=_snippet(value, match.start(), match.end()),
                reason="CRM text contains lossy ellipsis truncation",
            )
        )
    return result


def _detect_duplicate_label_counts(field: str, value: str) -> list[CrmTextQualityFinding]:
    segments = _split_label_segments(value)
    raw_seen: dict[str, str] = {}
    counted_seen: dict[str, str] = {}
    result: list[CrmTextQualityFinding] = []

    for segment in segments:
        count_match = COUNTED_LABEL_RE.match(segment)
        if count_match:
            canonical = _canonical_label(count_match.group("label"))
            if canonical in raw_seen:
                result.append(
                    CrmTextQualityFinding(
                        class_id="Q2",
                        risk_type="duplicate_label_and_count",
                        severity="P2",
                        field=field,
                        matched_text=f"{raw_seen[canonical]} | {segment}",
                        reason="CRM text mixes a raw label with the same counted label",
                    )
                )
            counted_seen.setdefault(canonical, segment)
            continue

        canonical = _canonical_label(segment)
        if not canonical:
            continue
        if canonical in counted_seen:
            result.append(
                CrmTextQualityFinding(
                    class_id="Q2",
                    risk_type="duplicate_label_and_count",
                    severity="P2",
                    field=field,
                    matched_text=f"{segment} | {counted_seen[canonical]}",
                    reason="CRM text mixes a raw label with the same counted label",
                )
            )
        raw_seen.setdefault(canonical, segment)

    return _dedupe_findings(result)


def _detect_objection_labels(
    objection_text: str,
    next_step: str,
    priority: str,
    probability: str,
) -> list[CrmTextQualityFinding]:
    if not objection_text:
        return []
    result: list[CrmTextQualityFinding] = []
    for segment in _split_label_segments(objection_text):
        canonical = _canonical_label(segment)
        if canonical in WEAK_FILLER_LABELS:
            result.append(
                CrmTextQualityFinding(
                    class_id="Q3a",
                    risk_type="weak_filler_objection_label",
                    severity="P3",
                    field="Возражения",
                    matched_text=segment,
                    reason="Weak objection labels need evidence text or should be hidden from the CRM card",
                )
            )
        strong_match = STRONG_NEGATIVE_LABEL_RE.search(segment)
        if not strong_match or HISTORICAL_EVIDENCE_RE.search(segment):
            continue
        conflict = _has_sales_context(next_step, priority, probability)
        result.append(
            CrmTextQualityFinding(
                class_id="Q3b",
                risk_type="strong_negative_objection_conflict" if conflict else "strong_negative_objection_label",
                severity="P1" if conflict else "P2",
                field="Возражения",
                matched_text=strong_match.group(0),
                reason=(
                    "Strong negative objection conflicts with current sales context"
                    if conflict
                    else "Strong negative objection needs date/evidence or historical separation"
                ),
            )
        )
    return _dedupe_findings(result)


def _detect_next_step_quality(next_step: str, priority: str, probability: str) -> list[CrmTextQualityFinding]:
    if not next_step:
        return []
    result: list[CrmTextQualityFinding] = []
    closure = CLOSURE_NEXT_STEP_RE.search(next_step)
    if closure:
        result.append(
            CrmTextQualityFinding(
                class_id="Q4a",
                risk_type="closure_next_step_requires_downgrade",
                severity="P1",
                field="Следующий шаг",
                matched_text=closure.group(0),
                reason="Closure/passive next step must downgrade priority or route to manual review",
            )
        )
        if _has_sales_context("", priority, probability):
            result.append(
                CrmTextQualityFinding(
                    class_id="Q4a",
                    risk_type="priority_next_step_conflict",
                    severity="P1",
                    field="Следующий шаг",
                    matched_text=next_step,
                    reason="Closure next step conflicts with warm/hot priority or sales probability",
                )
            )

    vague = VAGUE_NEXT_STEP_RE.search(next_step)
    if vague and not _has_concrete_date(next_step):
        result.append(
            CrmTextQualityFinding(
                class_id="Q4c",
                risk_type="vague_next_step",
                severity="P2",
                field="Следующий шаг",
                matched_text=vague.group(0),
                reason="Next step is not actionable without a concrete date, condition, or owner",
            )
        )
    return _dedupe_findings(result)


def _detect_lost_lead_next_step_conflict(
    text: str,
    next_step: str,
    priority: str,
    probability: str,
) -> list[CrmTextQualityFinding]:
    lost_signal = LOST_LEAD_SIGNAL_RE.search(text)
    if not lost_signal or not _has_sales_context(next_step, priority, probability):
        return []
    return [
        CrmTextQualityFinding(
            class_id="Q4d",
            risk_type="lost_lead_next_step_conflict",
            severity="P1",
            field="Авто история общения",
            matched_text=lost_signal.group(0),
            reason="Lost/competitor-purchase signal conflicts with active sales next step, priority, or probability",
        )
    ]


def _detect_passive_customer_next_step_conflict(
    text: str,
    next_step: str,
    priority: str,
    probability: str,
) -> list[CrmTextQualityFinding]:
    passive_signal = PASSIVE_CUSTOMER_SIGNAL_RE.search(text)
    if not passive_signal or not _has_sales_context(next_step, priority, probability):
        return []
    return [
        CrmTextQualityFinding(
            class_id="Q4e",
            risk_type="passive_customer_next_step_conflict",
            severity="P2",
            field="Авто история общения",
            matched_text=passive_signal.group(0),
            reason="Customer asked to return on their own or avoid active outreach, but CRM still has an active sales next step",
        )
    ]


def _detect_no_next_step_conflict(
    text: str,
    next_step: str,
    priority: str,
    probability: str,
) -> list[CrmTextQualityFinding]:
    no_step_signal = EXPLICIT_NO_NEXT_STEP_SIGNAL_RE.search(text)
    if not no_step_signal or not _has_sales_context(next_step, priority, probability):
        return []
    return [
        CrmTextQualityFinding(
            class_id="Q4f",
            risk_type="explicit_no_next_step_conflict",
            severity="P1",
            field="Авто история общения",
            matched_text=no_step_signal.group(0),
            reason="CRM text says there is no agreed next step or explicit refusal, but an active sales next step remains",
        )
    ]


def _detect_wrong_person_conflict(
    text: str,
    next_step: str,
    priority: str,
    probability: str,
) -> list[CrmTextQualityFinding]:
    wrong_person_signal = WRONG_PERSON_SIGNAL_RE.search(text)
    if not wrong_person_signal:
        return []
    return [
        CrmTextQualityFinding(
            class_id="Q4h",
            risk_type="wrong_person_or_identity_mismatch",
            severity="P1",
            field="Авто история общения",
            matched_text=wrong_person_signal.group(0),
            reason="Wrong-person or identity-mismatch signal must not produce active sales CRM writeback",
        )
    ]


def _detect_active_client_loss_reason_conflict(
    payload: object,
    next_step: str,
    priority: str,
    probability: str,
) -> list[CrmTextQualityFinding]:
    row = payload if isinstance(payload, Mapping) else None
    reason = _safe_text(_first_mapping_value(row, LOSS_REASON_FIELDS)) if row else ""
    signal = ACTIVE_CLIENT_LOSS_REASON_RE.search(reason)
    if not signal:
        return []
    return [
        CrmTextQualityFinding(
            class_id="Q4j",
            risk_type="active_client_loss_reason_requires_entity_resolution",
            severity="P1",
            field=_first_present_field(row, LOSS_REASON_FIELDS),
            matched_text=signal.group(0),
            reason="AMO loss reason says the person is an active client; do not treat this closed lead as a lost sale or active sales target",
        )
    ]


def _detect_completed_payment_next_step_conflict(
    text: str,
    next_step: str,
    priority: str,
    probability: str,
) -> list[CrmTextQualityFinding]:
    payment_signal = COMPLETED_PAYMENT_SIGNAL_RE.search(text)
    if not payment_signal:
        return []
    if PAYMENT_COLLECTION_NEXT_STEP_RE.search(next_step) or _has_sales_context(next_step, priority, probability):
        return [
            CrmTextQualityFinding(
                class_id="Q4g",
                risk_type="completed_payment_next_step_conflict",
                severity="P1",
                field="Авто история общения",
                matched_text=payment_signal.group(0),
                reason="CRM text indicates payment/receipt/closed deal, but next step or sales context still asks to collect payment",
            )
        ]
    return []


def _detect_stale_followup_date(
    payload: object,
    *,
    analysis_date: str | date | datetime | None,
    next_step: str,
) -> list[CrmTextQualityFinding]:
    row = payload if isinstance(payload, Mapping) else None
    followup_raw = _first_mapping_value(row, FOLLOWUP_DATE_FIELDS) if row else _extract_labeled_value(_payload_text(payload), FOLLOWUP_DATE_FIELDS)
    followup_date = _normalize_date(followup_raw)
    analysis_date_text = _normalize_date(analysis_date)
    stale_source = _stale_source_next_step(payload, analysis_date_text=analysis_date_text, next_step=next_step)
    if stale_source:
        return [stale_source]
    if not followup_date or not analysis_date_text or followup_date != analysis_date_text:
        relative = RELATIVE_STALE_NEXT_STEP_RE.search(next_step)
        if relative and not _relative_followup_date_matches(next_step, followup_date, analysis_date_text):
            return [
                CrmTextQualityFinding(
                    class_id="Q4b",
                    risk_type="relative_next_step_date_mismatch",
                    severity="P2",
                    field=_first_present_field(row, FOLLOWUP_DATE_FIELDS) if row else "Рекомендуемая дата следующего контакта",
                    matched_text=f"{relative.group(0)} / {followup_date or 'empty'}",
                    reason="Relative next-step wording must align with the recommended follow-up date or be rewritten",
                )
            ]
        return []
    relative = RELATIVE_STALE_NEXT_STEP_RE.search(next_step)
    if relative and not _relative_followup_date_matches(next_step, followup_date, analysis_date_text):
        return [
            CrmTextQualityFinding(
                class_id="Q4b",
                risk_type="relative_next_step_date_mismatch",
                severity="P2",
                field=_first_present_field(row, FOLLOWUP_DATE_FIELDS) if row else "Рекомендуемая дата следующего контакта",
                matched_text=f"{relative.group(0)} / {followup_date}",
                reason="Relative next-step wording conflicts with the recommended follow-up date",
            )
        ]
    if _has_concrete_date(next_step) or _mentions_today(next_step):
        return []
    if not (CLOSURE_NEXT_STEP_RE.search(next_step) or VAGUE_NEXT_STEP_RE.search(next_step)):
        return []
    return [
        CrmTextQualityFinding(
            class_id="Q4b",
            risk_type="stale_uniform_followup_date",
            severity="P2",
            field=_first_present_field(row, FOLLOWUP_DATE_FIELDS) if row else "Рекомендуемая дата следующего контакта",
            matched_text=followup_date,
            reason="Recommended follow-up date equals the analysis date without next-step date semantics",
        )
    ]


def _stale_source_next_step(
    payload: object,
    *,
    analysis_date_text: str,
    next_step: str,
) -> CrmTextQualityFinding | None:
    if not isinstance(payload, Mapping) or not analysis_date_text or not next_step:
        return None
    last_call_date = _normalize_date(_first_mapping_value(payload, LAST_CALL_DATE_FIELDS))
    if not last_call_date:
        return None
    try:
        analysis_day = datetime.strptime(analysis_date_text, "%Y-%m-%d").date()
        last_call_day = datetime.strptime(last_call_date, "%Y-%m-%d").date()
    except ValueError:
        return None
    age_days = (analysis_day - last_call_day).days
    if age_days < 30:
        return None
    if age_days < 90 and not ACTIVE_STALE_NEXT_STEP_RE.search(next_step):
        return None
    return CrmTextQualityFinding(
        class_id="Q4b",
        risk_type="stale_source_next_step",
        severity="P2",
        field=_first_present_field(payload, LAST_CALL_DATE_FIELDS),
        matched_text=f"{last_call_date} ({age_days}d) -> {next_step}",
        reason="Old source call cannot produce a fresh active sales next step without reactivation/manual review policy",
    )


def _relative_followup_date_matches(next_step: str, followup_date: str, analysis_date_text: str) -> bool:
    if not followup_date or not analysis_date_text:
        return False
    if re.search(r"\bзавтра\b", next_step, re.I):
        try:
            expected = datetime.strptime(analysis_date_text, "%Y-%m-%d").date() + date.resolution
            return followup_date == expected.isoformat()
        except ValueError:
            return False
    if re.search(r"\bчерез\s+год\b", next_step, re.I):
        try:
            current = datetime.strptime(analysis_date_text, "%Y-%m-%d").date()
            followup = datetime.strptime(followup_date, "%Y-%m-%d").date()
            return (followup - current).days >= 300
        except ValueError:
            return False
    if re.search(r"\bчерез\s+(?:полтора|пару|несколько|\d{1,2})\s+час", next_step, re.I):
        return followup_date == analysis_date_text
    return False


def _detect_verbose_manager_ux(
    payload: object,
    *,
    compact_max_chars: int,
    verbose_max_chars: int,
) -> list[CrmTextQualityFinding]:
    field, text = _preferred_auto_history(payload)
    if not text:
        return []
    has_summary_and_chronology = bool(re.search(r"\bсводк\w+\b", text, re.I)) and bool(
        re.search(r"\bхронологи\w+\b", text, re.I)
    )
    if len(text) > verbose_max_chars or (len(text) > compact_max_chars and has_summary_and_chronology):
        return [
            CrmTextQualityFinding(
                class_id="Q5",
                risk_type="verbose_manager_ux",
                severity="P3",
                field=field,
                matched_text=f"{len(text)} chars",
                reason="CRM card is longer than the compact manager UX budget",
            )
        ]
    return []


def _detect_empty_auto_history(payload: object) -> list[CrmTextQualityFinding]:
    if not isinstance(payload, Mapping):
        return []
    for field in AUTO_HISTORY_FIELDS:
        if field in payload and not _safe_text(payload.get(field)):
            return [
                CrmTextQualityFinding(
                    class_id="Q6",
                    risk_type="empty_auto_history",
                    severity="P1",
                    field=field,
                    matched_text="",
                    reason="Post-writeback readback must not contain empty auto history",
                )
            ]
    return []


def _detect_cross_field_duplicate_information(payload: object) -> list[CrmTextQualityFinding]:
    if not isinstance(payload, Mapping):
        return []

    fields = [(field, value) for field, value in _iter_target_text_fields(payload) if len(value) >= 60]
    result: list[CrmTextQualityFinding] = []
    for index, (left_field, left_text) in enumerate(fields):
        for right_field, right_text in fields[index + 1 :]:
            duplicate = _cross_field_duplicate_reason(left_text, right_text)
            if not duplicate:
                continue
            result.append(
                CrmTextQualityFinding(
                    class_id="Q7",
                    risk_type="cross_field_duplicate_information",
                    severity="P2",
                    field=f"{left_field} <-> {right_field}",
                    matched_text=duplicate,
                    reason="Different CRM AI fields must carry unique information, not repeat the same manager-facing text",
                )
            )
    return _dedupe_findings(result)


def _iter_target_text_fields(payload: object) -> Iterable[tuple[str, str]]:
    if isinstance(payload, Mapping):
        seen: set[str] = set()
        for field in TARGET_CRM_TEXT_FIELDS:
            if field in payload:
                seen.add(field)
                value = _safe_text(payload.get(field))
                if value:
                    yield (field, value)
        for field, raw in payload.items():
            if field in seen:
                continue
            if isinstance(raw, str) and ("AI" in str(field) or "истори" in str(field).casefold()):
                value = _safe_text(raw)
                if value:
                    yield (str(field), value)
        return
    value = _safe_text(payload)
    if value:
        yield ("text", value)


def _payload_text(payload: object) -> str:
    if isinstance(payload, Mapping):
        return " ".join(value for _, value in _iter_target_text_fields(payload))
    return _safe_text(payload)


def _objection_text(payload: object) -> str:
    if isinstance(payload, Mapping):
        values = [_safe_text(payload.get(field)) for field in OBJECTION_FIELDS if _safe_text(payload.get(field))]
        return " | ".join(values)
    text = _safe_text(payload)
    return _extract_labeled_value(text, OBJECTION_FIELDS)


def _next_step_text(payload: object) -> str:
    if isinstance(payload, Mapping):
        value = _first_mapping_value(payload, NEXT_STEP_FIELDS)
        if value:
            return _safe_text(value)
    return _extract_labeled_value(_payload_text(payload), NEXT_STEP_FIELDS)


def _preferred_auto_history(payload: object) -> tuple[str, str]:
    if isinstance(payload, Mapping):
        for field in AUTO_HISTORY_FIELDS:
            value = _safe_text(payload.get(field))
            if value:
                return (field, value)
        values = list(_iter_target_text_fields(payload))
        if not values:
            return ("Авто история общения", "")
        return max(values, key=lambda item: len(item[1]))
    return ("text", _safe_text(payload))


def _first_mapping_value(row: Mapping[str, Any] | None, fields: Iterable[str]) -> Any:
    if row is None:
        return ""
    for field in fields:
        value = row.get(field)
        if _safe_text(value):
            return value
    return ""


def _first_present_field(row: Mapping[str, Any] | None, fields: Iterable[str]) -> str:
    if row is None:
        return ""
    for field in fields:
        if field in row:
            return field
    return ""


def _extract_labeled_value(text: str, labels: Iterable[str]) -> str:
    if not text:
        return ""
    label_alternation = "|".join(re.escape(label) for label in labels if label)
    if not label_alternation:
        return ""
    match = re.search(rf"(?:{label_alternation})\s*:\s*(?P<value>[^.\n\r]+)", text, re.I)
    return _safe_text(match.group("value")) if match else ""


def _split_label_segments(value: str) -> list[str]:
    text = re.sub(
        r"\b(?:Возражения|Ограничения/возражения|Актуальные ограничения|"
        r"Продукты интереса|Интерес/продукты|Предметы)\s*:\s*",
        "",
        _safe_text(value),
        flags=re.I,
    )
    raw_parts = re.split(r"\s*[|;\n]\s*|\s*,\s*", text)
    result: list[str] = []
    for part in raw_parts:
        cleaned = part.strip(" .:-")
        if cleaned:
            result.append(cleaned)
    return result


def _canonical_label(value: str) -> str:
    text = _safe_text(value).casefold()
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r":\s*\d{1,5}(?:\s+\w+)?$", "", text)
    text = re.sub(r"[\"'«»“”]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .:-")


def _has_sales_context(next_step: str, priority: str, probability: str) -> bool:
    priority_value = _safe_text(priority).casefold()
    if priority_value in {"warm", "hot", "теплый", "тёплый", "горячий"}:
        return True
    probability_value = _parse_probability(probability)
    if probability_value is not None and probability_value >= 45:
        return True
    return bool(SALES_NEXT_STEP_RE.search(next_step))


def _parse_probability(value: str) -> int | None:
    text = _safe_text(value).replace("%", "")
    if not text:
        return None
    try:
        return int(round(float(text.replace(",", "."))))
    except ValueError:
        return None


def _has_concrete_date(text: str) -> bool:
    return bool(CONCRETE_DATE_RE.search(text))


def _mentions_today(text: str) -> bool:
    return bool(re.search(r"\b(?:сегодня|сегодняшн\w+|завтра)\b", text, re.I)) and not MONTH_ONLY_RE.search(text)


def _normalize_date(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _safe_text(value)
    if not text:
        return ""
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text[:19], fmt).date().isoformat()
        except ValueError:
            continue
    match = re.search(r"\b\d{4}-\d{1,2}-\d{1,2}\b", text)
    if match:
        try:
            return datetime.strptime(match.group(0), "%Y-%m-%d").date().isoformat()
        except ValueError:
            return ""
    return ""


def _snippet(text: str, start: int, end: int, *, radius: int = 45) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right].strip()


def _dedupe_findings(findings: Iterable[CrmTextQualityFinding]) -> list[CrmTextQualityFinding]:
    result: list[CrmTextQualityFinding] = []
    seen: set[tuple[str, str, str, str]] = set()
    for finding in findings:
        key = (finding.class_id, finding.risk_type, finding.field, finding.matched_text)
        if key in seen:
            continue
        seen.add(key)
        result.append(finding)
    return result


def _cross_field_duplicate_reason(left: str, right: str) -> str:
    left_clean = _normalize_for_similarity(left)
    right_clean = _normalize_for_similarity(right)
    if not left_clean or not right_clean:
        return ""

    smaller_text, larger_text = (left_clean, right_clean) if len(left_clean) <= len(right_clean) else (right_clean, left_clean)
    if len(smaller_text) >= 90 and smaller_text in larger_text:
        return _snippet(smaller_text, 0, min(len(smaller_text), 180), radius=0)

    for sentence in _significant_sentences(smaller_text):
        if sentence in larger_text:
            return _snippet(sentence, 0, min(len(sentence), 180), radius=0)

    left_tokens = _significant_tokens(left_clean)
    right_tokens = _significant_tokens(right_clean)
    if len(left_tokens) < 12 or len(right_tokens) < 12:
        return ""
    smaller_tokens, larger_tokens = (
        (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    )
    containment = len(smaller_tokens & larger_tokens) / max(len(smaller_tokens), 1)
    if containment >= 0.82:
        return _snippet(smaller_text, 0, min(len(smaller_text), 180), radius=0)
    return ""


def _normalize_for_similarity(value: str) -> str:
    text = _safe_text(value).casefold()
    text = re.sub(r"\[[a-zа-я0-9_ -]+\]", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[\u2013\u2014]", "-", text)
    text = re.sub(r"[^а-яa-z0-9ё.?!;: -]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _significant_sentences(value: str) -> list[str]:
    result: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+|[;\n\r]+", value):
        text = sentence.strip(" .:-")
        if len(text) >= 70 and len(_significant_tokens(text)) >= 10:
            result.append(text)
    return result


def _significant_tokens(value: str) -> set[str]:
    stopwords = {
        "клиент",
        "клиента",
        "клиенту",
        "клиентка",
        "менеджер",
        "звонок",
        "общение",
        "сделка",
        "сделке",
        "следующий",
        "текущий",
        "актуальный",
        "последний",
        "история",
        "сводка",
    }
    return {
        token
        for token in re.findall(r"[а-яa-z0-9ё]{4,}", value.casefold())
        if token not in stopwords and not token.isdigit()
    }


def _with_row_index(finding: CrmTextQualityFinding, row_index: int) -> CrmTextQualityFinding:
    return CrmTextQualityFinding(
        class_id=finding.class_id,
        risk_type=finding.risk_type,
        severity=finding.severity,
        field=finding.field,
        matched_text=finding.matched_text,
        reason=finding.reason,
        row_index=row_index,
    )


def _allowed_severities(min_severity: str) -> set[str]:
    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    threshold = order.get(min_severity, 3)
    return {severity for severity, rank in order.items() if rank <= threshold}


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


__all__ = [
    "CrmTextQualityFinding",
    "detect_crm_text_quality_batch_risks",
    "detect_crm_text_quality_risks",
    "crm_text_findings_to_counts",
    "findings_to_risk_counts",
    "has_blocking_crm_text_findings",
    "has_blocking_crm_text_quality_risk",
]
