from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class BotSafetyFinding:
    risk_type: str
    severity: str
    matched_text: str
    reason: str


MONEY_RE = re.compile(
    r"(?<!\w)(?:\d{1,3}(?:[\s\u00a0]\d{3})+|\d+[.,]\d+)\s*"
    r"(?:руб(?:\.|лей|ля|ль)?|₽|р\.?|тыс\.?|тысяч\w*)?(?!\w)|"
    r"\b(?:цен[ауыойе]?|стоимост\w*|стоит|оплат[аеуыой]?|плат[её]ж\w*|семестр\w*|год\s+целиком)"
    r"\D{0,35}\d{4,6}\b|"
    r"\bза\s+\d{4,6}\b(?!\s*(?:год(?:а|у|ом|е)?|г\.?))|"
    r"\b(?:пятьдесят|сорок|тридцать|двадцать|десять|пятнадцать|шестьдесят|семьдесят|"
    r"восемьдесят|девяносто|сто)\s+(?:тысяч\w*|рубл\w*)\b",
    re.I,
)
PERCENT_RE = re.compile(r"(?<!\w)\d{1,3}\s*(?:%|процент(?:а|ов)?)(?!\w)", re.I)
PAYMENT_PROVIDER_RE = re.compile(r"\b(?:альфа(?:[-\s]?банк\w*)?|алфа(?:[-\s]?банк\w*)?|сбер(?:банк)?\w*|тинькофф|т-банк\w*|яндекс\s*сплит)\b", re.I)
PHONE_EMAIL_HANDLE_RE = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|"
    r"(?<!\w)@[A-Za-z0-9_]{4,}(?!\w)|"
    r"(?<!\d)(?:\+?\d[\s()\-]*){10,15}(?!\d)",
    re.I,
)
ROLE_NAME_RE = re.compile(
    r"\b(?i:(?:преподавател[ьяюеем]*|педагог[а-яё]*|учител[ьяюеем]*|куратор[а-яё]*|"
    r"методист[а-яё]*|администратор[а-яё]*|наставник[а-яё]*|ученик(?:а|у|ом|е)?|"
    r"реб[её]н(?:ок|ка|ку|ком|ке)|клиент(?:а|у|ом|е)?))\s+"
    r"[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){0,2}\b"
)
SURNAME_ACTION_RE = re.compile(
    r"\b[А-ЯЁ][а-яё]{2,}(?:ов|ев|ёв|ин|ын|ский|цкий|ская|цкая|ова|ева|ёва|ина|енко|юк|ич|"
    r"ович|евич|овна|евна|ична|инична)[а-яё]{0,4}\s+"
    r"(?:вести|вед[её]т|проводит|препода[её]т|курирует)\b|"
    r"\b(?:вести|вед[её]т|проводит|препода[её]т|курирует)\s+"
    r"[А-ЯЁ][а-яё]{2,}(?:ов|ев|ёв|ин|ын|ский|цкий|ская|цкая|ова|ева|ёва|ина|енко|юк|ич|"
    r"ович|евич|овна|евна|ична|инична)[а-яё]{0,4}\b"
)
FAMILY_LABEL_RE = re.compile(r"\b(?i:фамил[иьею][а-яё]*)\s+[А-ЯЁ][а-яё]{2,}\b")
ADDRESS_RE = re.compile(
    r"\b(?:ул\.?|улиц[ауыей]?|проспект[а-яё]*|пр-т|переул(?:ок|ка|ке)?|пер\.?|шоссе|"
    r"бульвар[а-яё]*|площад[ьи]|проезд[а-яё]*|метро|м\.)\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b|"
    r"\b(?:дом|д\.|корпус|корп\.?|к\.|строение|стр\.?|кабинет\w*|каб\.|аудитор\w*)\s*\d+[А-Яа-яA-Za-z]?\b|"
    r"\b(?:Сухарев\w*|Долгопрудн\w*|С?корн[яе]жн\w*|С?коряжн\w*|Пацаев\w*|Майск\w*|КПМ)\b",
    re.I,
)
DEADLINE_RE = re.compile(
    r"\b(?:до|по)\s+(?:конца\s+)?(?:\d{1,2}\s+)?(?:дня|недели|месяца|года|каникул|"
    r"января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b|"
    r"\b(?:до\s+)?(?:\d{1,2}\s*(?:и|,)\s*)*\d{1,2}\s+числа\b|"
    r"\b\d{1,2}[.\/-]\d{1,2}(?:[.\/-]\d{2,4})?\b|"
    r"\b\d{1,2}\s*:\s*\d{2}(?:\s*[-–]\s*\d{1,2}\s*:\s*\d{2})?\b",
    re.I,
)
PROMISE_RE = re.compile(
    r"\b(?:верн[её]м(?:ся)?|перезвон\w*|свяж(?:емся|ется)|напиш\w*|сообщ\w*|"
    r"компенсир\w+|возмест\w+)\b.{0,70}?"
    r"\b(?:сегодня|завтра|до\s+конца|в\s+течение|как\s+только)\b",
    re.I,
)
DOCUMENT_RE = re.compile(r"\b(?:файл|документ|word|pdf|excel|таблиц[ауы])\s+[«\"'][^»\"']{1,80}[»\"']", re.I)
BRAND_ARTIFACT_RE = re.compile(r"\b(?:НПК|МФТИ|ФТИ|КПМ|ЛНПК|МФШТИ|черн(?:ый|ой)\s+центр)\b", re.I)
PLACEHOLDER_REPEAT_RE = re.compile(
    r"\b(актуальную стоимость|актуальные варианты|актуальное окно записи|адрес, который подтвердит менеджер|"
    r"менеджер свяжется с вами после проверки)"
    r"(?:[,.]?\s+\1){1,}",
    re.I,
)


DETECTOR_RULES: tuple[tuple[str, str, re.Pattern[str], str], ...] = (
    ("money_or_terms", "P0", MONEY_RE, "Concrete money amount or commercial term remains"),
    ("percent", "P0", PERCENT_RE, "Concrete percent remains"),
    ("payment_provider", "P1", PAYMENT_PROVIDER_RE, "Concrete payment provider remains"),
    ("contact_data", "P0", PHONE_EMAIL_HANDLE_RE, "Phone, email, or messenger handle remains"),
    ("role_name", "P1", ROLE_NAME_RE, "Named person remains in role context"),
    ("surname_action", "P1", SURNAME_ACTION_RE, "Surname remains near teaching/action verb"),
    ("family_label", "P1", FAMILY_LABEL_RE, "Surname remains after family-name label"),
    ("address_or_room", "P1", ADDRESS_RE, "Concrete location, room, or facility remains"),
    ("deadline", "P2", DEADLINE_RE, "Concrete deadline remains"),
    ("promise", "P2", PROMISE_RE, "Concrete service promise remains"),
    ("document_reference", "P2", DOCUMENT_RE, "Concrete document/file reference remains"),
    ("brand_artifact", "P3", BRAND_ARTIFACT_RE, "Tenant-specific brand/facility artifact remains"),
    ("over_sanitization_cluster_repeat", "P3", PLACEHOLDER_REPEAT_RE, "Repeated generic placeholder cluster"),
)


def detect_bot_safety_risks(text: object, *, min_severity: str = "P3") -> list[BotSafetyFinding]:
    value = _clean(text)
    if not value:
        return []
    allowed = _allowed_severities(min_severity)
    findings: list[BotSafetyFinding] = []
    for risk_type, severity, pattern, reason in DETECTOR_RULES:
        if severity not in allowed:
            continue
        for match in pattern.finditer(value):
            matched = match.group(0).strip()
            if not matched:
                continue
            if risk_type == "promise" and "свяжется с вами после проверки" in matched.lower():
                continue
            findings.append(BotSafetyFinding(risk_type, severity, matched, reason))
    return findings


def has_blocking_bot_safety_risk(text: object, *, min_severity: str = "P1") -> bool:
    return bool(detect_bot_safety_risks(text, min_severity=min_severity))


def findings_to_risk_counts(findings: Iterable[BotSafetyFinding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.risk_type] = counts.get(finding.risk_type, 0) + 1
    return counts


def _allowed_severities(min_severity: str) -> set[str]:
    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    threshold = order.get(min_severity, 3)
    return {severity for severity, rank in order.items() if rank <= threshold}


def _clean(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


__all__ = [
    "BotSafetyFinding",
    "detect_bot_safety_risks",
    "findings_to_risk_counts",
    "has_blocking_bot_safety_risk",
]
