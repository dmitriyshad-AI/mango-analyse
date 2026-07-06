from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import BotReplayResult, ReplayCase
from .pseudonymizer import pii_signals


NUMBER_RE = re.compile(r"(?<![\w/])\d[\d\s.,/-]*\d|\b\d\b")
MANAGER_ROUTES = {"manager_only", "draft_for_manager", "blocked"}


@dataclass(frozen=True)
class MachineGateResult:
    passed: bool
    flags: tuple[str, ...]
    new_numbers: tuple[str, ...] = ()


def extract_numbers(text: str) -> set[str]:
    result: set[str] = set()
    for match in NUMBER_RE.finditer(str(text or "")):
        raw = match.group(0).strip()
        normalized = re.sub(r"\s+", "", raw)
        if normalized:
            result.add(normalized)
    return result


def number_index(texts: Iterable[str]) -> set[str]:
    numbers: set[str] = set()
    for text in texts:
        numbers.update(extract_numbers(text))
    return numbers


def run_machine_gate(
    case: ReplayCase,
    result: BotReplayResult,
    *,
    client_safe_numbers: Sequence[str] = (),
    judge_payloads: Sequence[object] = (),
) -> MachineGateResult:
    flags: list[str] = []
    allowed = set(client_safe_numbers)
    allowed.update(number_index([case.client_message, *(item.text for item in case.prefix_messages)]))
    produced = extract_numbers(result.bot_text)
    new_numbers = tuple(sorted(number for number in produced if number not in allowed))
    if new_numbers:
        flags.append("new_number_unverified")
    brand = case.brand.casefold()
    text = result.bot_text.casefold()
    if brand == "foton" and "унпк" in text:
        flags.append("other_brand")
    if brand == "unpk" and "фотон" in text:
        flags.append("other_brand")
    if pii_signals(result.bot_text):
        flags.append("pii_in_bot_text")
    for payload in judge_payloads:
        if pii_signals(payload):
            flags.append("pii_in_judge_payload")
            break
    if case.expected_p0:
        if result.route not in MANAGER_ROUTES:
            flags.append("p0_route_lost")
        if not set(result.safety_flags).intersection({"conversation_intent_plan_p0", "payment_dispute", "high_risk_manager_only", "no_auto_send"}):
            flags.append("p0_flags_missing")
    return MachineGateResult(passed=not flags, flags=tuple(dict.fromkeys(flags)), new_numbers=new_numbers)
