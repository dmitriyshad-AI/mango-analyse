from __future__ import annotations

import base64
import hashlib
import re
from typing import Any


PHONE_RE = re.compile(r"(?<!\d)(?:\+7|8)[\s\-()]*(?:\d[\s\-()]*){10}(?!\d)")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
CONTRACT_RE = re.compile(
    r"\b(?:договор|контракт)\b\s*№?\s*(?=[A-Za-zА-Яа-я0-9/_-]{3,}\b)(?=[A-Za-zА-Яа-я0-9/_-]*\d)[A-Za-zА-Яа-я0-9/_-]+\b",
    re.I,
)
RU_NAME_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\b")
RU_MIXED_CASE_SURNAME_RE = re.compile(
    r"\b[А-ЯЁ][а-яё]{2,}\s+[а-яё]{3,}(?:ова|ева|ёва|ина|ская|цкая|ский|цкий|ов|ев|ёв|ин)\b"
)
SENSITIVE_ID_KEY_RE = re.compile(
    r"(^|_)(?:profile|chat|message|lead|contact|talk|dialog|thread|event|source|dedupe)_?id$|"
    r"^(?:profile_id|chat_id|message_id|lead_id|contact_id|talk_id|dialog_id|thread_id)$",
    re.I,
)

FAKE_NAMES = (
    "Анна Иванова",
    "Мария Петрова",
    "Ольга Смирнова",
    "Ирина Волкова",
    "Елена Соколова",
    "Наталья Кузнецова",
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
        text = PHONE_RE.sub("[phone]", text)
        text = EMAIL_RE.sub("[email]", text)
        text = URL_RE.sub("[url]", text)
        text = CONTRACT_RE.sub("договор [contract]", text)
        text = RU_NAME_RE.sub(lambda match: self._fake_name(match.group(0)), text)
        return RU_MIXED_CASE_SURNAME_RE.sub(lambda match: self._fake_name(match.group(0)), text)

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
    text = repr(value)
    signals: list[str] = []
    if PHONE_RE.search(text):
        signals.append("phone")
    if EMAIL_RE.search(text):
        signals.append("email")
    if URL_RE.search(text):
        signals.append("url")
    if CONTRACT_RE.search(text):
        signals.append("contract")
    if re.search(
        r"(?<![\w\[])(?:profile|chat|message|lead|contact|talk|thread|dialog)_id['\"]?\s*[:=]\s*['\"]?(?!\[|wappi_replay_)[A-Za-z0-9_-]{4,}",
        text,
    ):
        signals.append("raw_id")
    return signals
