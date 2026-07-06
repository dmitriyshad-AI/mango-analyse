from __future__ import annotations

import hashlib
import re
from typing import Any


PHONE_RE = re.compile(r"(?<!\d)(?:\+7|8)[\s\-()]*(?:\d[\s\-()]*){10}(?!\d)")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
CONTRACT_RE = re.compile(r"\b(?:договор|контракт)\s*№?\s*[A-Za-zА-Яа-я0-9/_-]{3,}\b", re.I)
RU_NAME_RE = re.compile(r"\b[А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,})?\b")

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
        return RU_NAME_RE.sub(lambda match: self._fake_name(match.group(0)), text)

    def object(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.object(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.object(item) for item in value)
        if isinstance(value, dict):
            return {str(key): self.object(item) for key, item in value.items()}
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
    return signals
