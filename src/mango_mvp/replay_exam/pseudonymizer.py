from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


PHONE_RE = re.compile(
    r"(?<!\d)(?:(?:\+7|8)[\s\-()]*(?:\d[\s\-()]*){10}|7\d{10})(?:@c\.us)?(?!\d)|"
    r"(?<![\w])(?:\+|00)[ \t\u00a0]*[1-9](?:[ \t\u00a0()\-–—]*\d){6,13}(?:@c\.us)?(?![\w])",
    re.I,
)
LABELED_BARE_PHONE_RE = re.compile(
    r"(?P<label>\b(?:тел(?:ефон)?|моб(?:ильный)?|phone|mobile|cell|whats?app|ватсап)\b|\bномер\s+(?:телефона|мобильного)\b|\((?:тел|моб)\))"
    r"(?P<separator>[^\d\n]{0,24})"
    r"(?P<number>(?!\d{4}[./-]\d{1,2}[./-]\d{1,2}(?!\d))(?!\d{1,2}[./-]\d{1,2}[./-]\d{4}(?!\d))"
    r"\d(?:[ \t\u00a0()\-–—]*\d){6,14})(?![\w])",
    re.I,
)
BIRTH_DATE_RE = re.compile(r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-](?:19|20)\d{2}(?!\d)")
BIRTH_DATE_CONTEXT_RE = re.compile(r"\b(?:дата|день)\s+рождения\b|\bрожд\w*\b|(?<!\w)д\.?\s*р\.?(?!\w)", re.I)
CLASS_CONTEXT_RE = re.compile(r"\b(?:\d{1,2}(?:-?й)?\s+)?класс\b", re.I)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
USERNAME_RE = re.compile(r"(?<![\w/])@[A-Za-z][A-Za-z0-9_]{3,32}\b")
LONG_ID_RE = re.compile(r"(?<![\w\[])(?:[a-f0-9]{16,}|[0-9]{16,})(?![\w\]])", re.I)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
CONTRACT_RE = re.compile(
    r"\b(?:договор|контракт)\b\s*№?\s*(?=[A-Za-zА-Яа-я0-9/_-]{3,}\b)(?=[A-Za-zА-Яа-я0-9/_-]*\d)[A-Za-zА-Яа-я0-9/_-]+\b",
    re.I,
)
RU_NAME_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\b")
RU_MIXED_CASE_SURNAME_RE = re.compile(
    r"\b[А-ЯЁ][а-яё]{2,}\s+[а-яё]{3,}(?:ова|ева|ёва|ина|ская|цкая|ский|цкий|ов|ев|ёв|ин)\b"
)
RAW_IDENTIFIER_KEY_RE = re.compile(
    r"^(?:id|from|to|phone|username|senderName|contact_name|task_id|wappi_bot_id|stanzaId|chatId|file_name|filename|file_id|attachment_id)$",
    re.I,
)
SENSITIVE_ID_KEY_RE = re.compile(
    r"(^|_)(?:profile|chat|message|lead|contact|talk|dialog|thread|event|source|dedupe)_?id$|"
    r"^(?:id|profile_id|chat_id|message_id|lead_id|contact_id|talk_id|dialog_id|thread_id|from|to|phone|username|senderName|contact_name|task_id|wappi_bot_id|stanzaId|chatId|file_name|filename|file_id|attachment_id)$",
    re.I,
)
TIMESTAMP_KEY_RE = re.compile(r"(^|_)(?:ts|time|timestamp|created_at|updated_at|date|datetime|ts_masked)$", re.I)
HASH_KEY_RE = re.compile(r"(^|_)(?:hash|digest|sha|sha256)(_|$)", re.I)
PSEUDONYMIZED_ID_RE = re.compile(r"^\[[a-z0-9_]+:id_[a-z2-7]{8,}\]$")

FAKE_NAMES = (
    "Анна Иванова",
    "Мария Петрова",
    "Ольга Смирнова",
    "Ирина Волкова",
    "Елена Соколова",
    "Наталья Кузнецова",
)
PROGRAM_NAME_STOP_PHRASES = (
    'Летняя очная школа "Ирина Волкова"',
    "Летняя очная школа «Ирина Волкова»",
    "Летняя очная школа Ирина Волкова",
    "ФизМат «Формула Физтеха»",
    'ФизМат "Формула Физтеха"',
    "Формула Физтеха",
    "AI Lab",
)


def _has_birth_date_context(text: str) -> bool:
    return bool(
        BIRTH_DATE_CONTEXT_RE.search(text)
        or (
            CLASS_CONTEXT_RE.search(text)
            and (PHONE_RE.search(text) or LABELED_BARE_PHONE_RE.search(text) or "[phone]" in text)
        )
    )


class ReplayPseudonymizer:
    def __init__(self, *, dialog_salt: str) -> None:
        self.dialog_salt = dialog_salt
        self._name_map: dict[str, str] = {}
        self._id_map: dict[tuple[str, str], str] = {}

    def _fake_name(self, original: str) -> str:
        if original not in self._name_map:
            digest = hashlib.sha256(f"{self.dialog_salt}:{original}".encode("utf-8")).digest()
            self._name_map[original] = FAKE_NAMES[digest[0] % len(FAKE_NAMES)]
        return self._name_map[original]

    def text(self, value: str) -> str:
        text = str(value or "")
        mask_birth_dates = _has_birth_date_context(text)
        text = PHONE_RE.sub("[phone]", text)
        text = LABELED_BARE_PHONE_RE.sub(
            lambda match: f"{match.group('label')}{match.group('separator')}[phone]",
            text,
        )
        if mask_birth_dates:
            text = BIRTH_DATE_RE.sub("[date_of_birth]", text)
        text = EMAIL_RE.sub("[email]", text)
        text = USERNAME_RE.sub("[username]", text)
        text = URL_RE.sub("[url]", text)
        text = CONTRACT_RE.sub("договор [contract]", text)
        protected: dict[str, str] = {}
        for index, phrase in enumerate(sorted(PROGRAM_NAME_STOP_PHRASES, key=len, reverse=True)):
            token = f"__REPLAY_PROGRAM_NAME_{index}__"
            if phrase in text:
                protected[token] = phrase
                text = text.replace(phrase, token)
        text = RU_NAME_RE.sub(lambda match: self._fake_name(match.group(0)), text)
        text = RU_MIXED_CASE_SURNAME_RE.sub(lambda match: self._fake_name(match.group(0)), text)
        for token, phrase in protected.items():
            text = text.replace(token, phrase)
        return text

    def id_value(self, key: str, value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        map_key = (str(key or "id").casefold(), raw)
        if map_key not in self._id_map:
            digest_bytes = hashlib.sha256(f"{self.dialog_salt}:{map_key[0]}:{raw}".encode("utf-8")).digest()
            digest = base64.b32encode(digest_bytes).decode("ascii").lower().rstrip("=")[:12]
            self._id_map[map_key] = f"[{map_key[0]}:id_{digest}]"
        return self._id_map[map_key]

    def object(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.object(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.object(item) for item in value)
        if isinstance(value, dict):
            scrubbed: dict[str, Any] = {}
            for key, item in value.items():
                key_s = str(key)
                if SENSITIVE_ID_KEY_RE.search(key_s):
                    if isinstance(item, list):
                        scrubbed[key_s] = [self.id_value(key_s, entry) for entry in item]
                    elif isinstance(item, tuple):
                        scrubbed[key_s] = tuple(self.id_value(key_s, entry) for entry in item)
                    else:
                        scrubbed[key_s] = self.id_value(key_s, item)
                    continue
                scrubbed[key_s] = self.object(item)
            return scrubbed
        return value


def pii_signals(value: Any) -> list[str]:
    return sorted({finding["kind"] for finding in pii_findings(value)})


def _allowlist_values(values: Iterable[str] = ()) -> set[str]:
    allowed: set[str] = set()
    for value in values:
        raw = str(value or "").strip()
        if not raw:
            continue
        allowed.add(raw)
        digits = re.sub(r"\D+", "", raw)
        if digits:
            allowed.add(digits)
            if len(digits) == 11 and digits.startswith("8"):
                allowed.add("7" + digits[1:])
    return allowed


def _is_allowed(raw: str, allowed: set[str]) -> bool:
    text = str(raw or "").strip()
    if not text:
        return True
    if PSEUDONYMIZED_ID_RE.match(text):
        return True
    if text in allowed:
        return True
    digits = re.sub(r"\D+", "", text)
    return bool(digits and digits in allowed)


def pii_findings(value: Any, *, allowlist: Iterable[str] = ()) -> list[dict[str, str]]:
    allowed = _allowlist_values(allowlist)
    findings: list[dict[str, str]] = []

    def add(kind: str, path: str, raw: str) -> None:
        if _is_allowed(raw, allowed):
            return
        findings.append({"kind": kind, "path": path, "value": "[redacted]"})

    def scan_text(text: str, path: str, *, timestamp_context: bool = False) -> None:
        for match in PHONE_RE.finditer(text):
            add("phone", path, match.group(0))
        for match in LABELED_BARE_PHONE_RE.finditer(text):
            add("phone", path, match.group("number"))
        if not timestamp_context and _has_birth_date_context(text):
            for match in BIRTH_DATE_RE.finditer(text):
                add("date_of_birth", path, match.group(0))
        for match in EMAIL_RE.finditer(text):
            add("email", path, match.group(0))
        for match in USERNAME_RE.finditer(text):
            add("username", path, match.group(0))
        for match in URL_RE.finditer(text):
            add("url", path, match.group(0))
        for match in CONTRACT_RE.finditer(text):
            add("contract", path, match.group(0))
        if not timestamp_context:
            for match in LONG_ID_RE.finditer(text):
                add("raw_id", path, match.group(0))

    def walk(item: Any, path: str = "$", *, key: str = "") -> None:
        key_s = str(key or "")
        timestamp_context = bool(TIMESTAMP_KEY_RE.search(key_s) or HASH_KEY_RE.search(key_s))
        if isinstance(item, Mapping):
            for nested_key, nested_value in item.items():
                nested_key_s = str(nested_key)
                nested_path = f"{path}.{nested_key_s}"
                if RAW_IDENTIFIER_KEY_RE.fullmatch(nested_key_s):
                    if nested_value not in (None, "", [], {}, ()):
                        add("raw_id_key", nested_path, str(nested_value))
                elif SENSITIVE_ID_KEY_RE.search(nested_key_s) and isinstance(nested_value, (str, int, float)):
                    raw = str(nested_value)
                    if LONG_ID_RE.search(raw) or PHONE_RE.search(raw) or EMAIL_RE.search(raw) or USERNAME_RE.search(raw):
                        add("raw_id_key", nested_path, raw)
                walk(nested_value, nested_path, key=nested_key_s)
        elif isinstance(item, (list, tuple, set)):
            for index, nested in enumerate(item):
                walk(nested, f"{path}[{index}]", key=key_s)
        elif isinstance(item, str):
            scan_text(item, path, timestamp_context=timestamp_context)
        elif isinstance(item, (int, float)) and not timestamp_context:
            scan_text(str(item), path, timestamp_context=False)

    walk(value)
    return findings


def kb_contact_allowlist(snapshot_path: Path) -> tuple[str, ...]:
    payload = json.loads(snapshot_path.expanduser().read_text(encoding="utf-8"))
    facts = payload.get("facts") or payload.get("facts_registry") or []
    contacts: set[str] = set()

    def collect_texts(item: Any) -> list[str]:
        if isinstance(item, str):
            return [item]
        if isinstance(item, Mapping):
            result: list[str] = []
            for nested in item.values():
                result.extend(collect_texts(nested))
            return result
        if isinstance(item, (list, tuple)):
            result: list[str] = []
            for nested in item:
                result.extend(collect_texts(nested))
            return result
        return []

    for fact in facts:
        if not isinstance(fact, Mapping):
            continue
        text_blob = "\n".join(collect_texts(fact))
        lowered = text_blob.casefold()
        if not any(marker in lowered for marker in ("телефон", "почта", "email", "telegram", "контакт", "писать можно")):
            continue
        contacts.update(match.group(0) for match in PHONE_RE.finditer(text_blob))
        contacts.update(match.group("number") for match in LABELED_BARE_PHONE_RE.finditer(text_blob))
        contacts.update(match.group(0) for match in EMAIL_RE.finditer(text_blob))
        contacts.update(match.group(0) for match in USERNAME_RE.finditer(text_blob))
    return tuple(sorted(contacts))
