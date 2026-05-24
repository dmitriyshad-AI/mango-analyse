from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


ANSWER_SAFETY_SCHEMA_VERSION = "answer_safety_v1_2026_05_24"


@dataclass(frozen=True)
class AnswerSafetyDecision:
    risk_codes: tuple[str, ...] = ()
    primary_risk: str = ""
    p0_required: bool = False
    manager_only: bool = False
    zero_collect_required: bool = False
    blocks_autonomy: bool = False
    blocks_rewriter: bool = False
    semantic_non_p0: bool = False
    evidence: Mapping[str, str] | None = None

    def to_json_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": ANSWER_SAFETY_SCHEMA_VERSION,
            "risk_codes": list(self.risk_codes),
            "primary_risk": self.primary_risk,
            "p0_required": self.p0_required,
            "manager_only": self.manager_only,
            "zero_collect_required": self.zero_collect_required,
            "blocks_autonomy": self.blocks_autonomy,
            "blocks_rewriter": self.blocks_rewriter,
            "semantic_non_p0": self.semantic_non_p0,
            "evidence": dict(self.evidence or {}),
        }


REFUND_RE = re.compile(
    r"\bвозв?рат(?!\w*\s+к\s+(?:тем|урок|материал|заняти))\w*"
    r"|\bвозвращ\w*\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)"
    r"|\bверн\w*(?:\s+мне|\s+нам|\s+пожалуйста)?\s+(?:деньги|оплат\w*|плат[её]ж\w*|средств\w*|сумм\w*)"
    r"|\bденьги\s+назад\b"
    r"|\bрасторг\w*\s+договор"
    r"|\bотказ\w*\s+от\s+обучен"
    r"|\bзабрать\s+деньги",
    re.I,
)

LEGAL_RE = re.compile(
    r"\bсуд\b|\bиск\b|претензи\w*|досудеб|роспотребнадзор|прокуратур|адвокат|юрист"
    r"|прав[ао][^.!?\n]{0,60}потребител|защит[а-яё]*\s+прав\s+потребител"
    r"|наруш\w*\s+прав|расторжен\w*\s+договор"
    r"|по\s+закону[^.!?\n]{0,80}(?:обязан|должн|наруш)",
    re.I,
)

COMPLAINT_RE = re.compile(
    r"жалоб(?!а\s+на\s+сайт)\w*|жалуюсь|возмущ\w*|недовол\w*|претензи|конфликт"
    r"|обман|ужасн|плохо\s+учит|плохо\s+пров[её]л|некомпетентн\w*",
    re.I,
)

REPUTATION_RE = re.compile(
    r"отзыв\w*\s+в\s+интернет|всех\s+предупреж\w*|напиш\w*\s+отзыв|остав\w*\s+отзыв",
    re.I,
)

PAYMENT_DISPUTE_RE = re.compile(
    r"(?:оплатил|оплатила|пров[её]л(?:и)?\s+плат[её]ж|списал[иось]*|деньги\s+списал)"
    r"[^.!?\n]{0,100}(?:не\s+вид|не\s+прош|нет\s+оплат|не\s+зачисл|не\s+получ)"
    r"|(?:оплат[ау]\s+не\s+вид|плат[её]ж\s+не\s+(?:прош[её]л|видно|зачисл))",
    re.I,
)

SOFT_NEGATIVE_ONLY_RE = re.compile(
    r"\b(?:подумаю|обсудить|обсудим|с менеджером обсудить|наверное\s+подумаем)\b",
    re.I,
)


def classify_answer_safety(
    *,
    client_message: str = "",
    context: Mapping[str, Any] | None = None,
    topic_id: str = "",
    route: str = "",
    safety_flags: Sequence[str] = (),
    include_recent_client_messages: bool = True,
) -> AnswerSafetyDecision:
    """Single source for P0/high-risk classification.

    The classifier is intentionally conservative for real P0: if topic/flags
    already say high-risk, it stays high-risk. At the same time, a semantic
    non-P0 conversation plan can prevent old false-positive topics from being
    recreated when the current message is harmless.
    """

    current = str(client_message or "")
    current_codes = codes_from_current_message(current)
    texts = [current]
    if include_recent_client_messages and isinstance(context, Mapping):
        recent = context.get("recent_messages")
        if isinstance(recent, Sequence) and not isinstance(recent, (str, bytes, bytearray)):
            for item in recent[-3:]:
                line = str(item or "").strip()
                lowered = line.casefold()
                if lowered.startswith(("клиент:", "client:", "user:")):
                    texts.append(line)
    haystack = "\n".join(texts)
    normalized = _normalize(haystack)
    current_norm = _normalize(current)
    evidence: dict[str, str] = {}
    codes: list[str] = []

    if REFUND_RE.search(haystack):
        codes.append("refund")
        evidence["refund"] = _first_match(REFUND_RE, haystack)
    if LEGAL_RE.search(haystack):
        codes.append("legal")
        evidence["legal"] = _first_match(LEGAL_RE, haystack)
    if _has_complaint_signal(haystack):
        codes.append("complaint")
        evidence["complaint"] = _first_match(COMPLAINT_RE, haystack)
    if REPUTATION_RE.search(haystack):
        codes.append("reputation_threat")
        evidence["reputation_threat"] = _first_match(REPUTATION_RE, haystack)
    if PAYMENT_DISPUTE_RE.search(haystack):
        codes.append("payment_dispute")
        evidence["payment_dispute"] = _first_match(PAYMENT_DISPUTE_RE, haystack)

    plan = _conversation_plan(context)
    plan_primary = str(plan.get("primary_intent") or "").strip()
    plan_risks = _text_list(plan.get("risk_signals"))
    plan_route_bias = str(plan.get("route_bias") or "").strip()
    if plan_primary in {"refund", "legal_threat", "complaint"}:
        codes.append(_risk_code_from_plan_primary(plan_primary))
        evidence.setdefault("conversation_intent_plan", plan_primary)
    for risk in plan_risks:
        if risk:
            codes.append(_normalize_plan_risk(risk))
            evidence.setdefault("conversation_intent_plan_risk", risk)

    flag_text = _normalize(" ".join(str(item or "") for item in safety_flags))
    if any(marker in flag_text for marker in ("zero_collect_refund", "refund_zero_collect")):
        codes.append("refund")
        evidence.setdefault("safety_flags", "refund")
    if any(marker in flag_text for marker in ("zero_collect_legal", "legal_threat", "conversation_intent_plan_p0")):
        codes.append("legal")
        evidence.setdefault("safety_flags", "legal")
    if "complaint_apology" in flag_text:
        codes.append("complaint")
        evidence.setdefault("safety_flags", "complaint")
    if "payment_confirmation" in flag_text:
        codes.append("payment_dispute")
        evidence.setdefault("safety_flags", "payment_dispute")

    topic = str(topic_id or "").strip()
    if topic == "theme:009_refund":
        codes.append("refund")
        evidence.setdefault("topic_id", topic)
    elif topic == "theme:029_legal_question":
        codes.append("legal")
        evidence.setdefault("topic_id", topic)
    elif topic == "theme:019b_negative_feedback":
        codes.append("complaint")
        evidence.setdefault("topic_id", topic)

    semantic_non_p0 = _semantic_non_p0_by_plan(plan, current_norm=current_norm)
    if semantic_non_p0 and not codes_from_current_message(current):
        # Let a clean current-message plan repair stale model topics/flags.
        codes = [code for code in codes if evidence.get("topic_id") != topic and evidence.get("safety_flags") not in {code}]
        evidence.pop("topic_id", None)
        evidence.pop("safety_flags", None)

    codes = tuple(dict.fromkeys(code for code in codes if code))
    primary = _primary_risk(codes, current_codes=current_codes)
    p0 = bool(codes) or plan_route_bias == "manager_only" and plan_primary in {"refund", "legal_threat", "complaint"}
    if semantic_non_p0 and not codes_from_current_message(current):
        p0 = False
        primary = ""
        codes = ()

    return AnswerSafetyDecision(
        risk_codes=codes,
        primary_risk=primary,
        p0_required=p0,
        manager_only=p0 or str(route or "") == "manager_only" and primary in {"refund", "legal", "complaint", "payment_dispute"},
        zero_collect_required=primary in {"refund", "legal", "complaint"},
        blocks_autonomy=p0,
        blocks_rewriter=p0,
        semantic_non_p0=semantic_non_p0,
        evidence=evidence,
    )


def codes_from_current_message(client_message: str) -> tuple[str, ...]:
    text = str(client_message or "")
    result: list[str] = []
    if REFUND_RE.search(text):
        result.append("refund")
    if LEGAL_RE.search(text):
        result.append("legal")
    if _has_complaint_signal(text):
        result.append("complaint")
    if REPUTATION_RE.search(text):
        result.append("reputation_threat")
    if PAYMENT_DISPUTE_RE.search(text):
        result.append("payment_dispute")
    return tuple(dict.fromkeys(result))


def _has_complaint_signal(text: str) -> bool:
    if SOFT_NEGATIVE_ONLY_RE.search(text) and not COMPLAINT_RE.search(text):
        return False
    return bool(COMPLAINT_RE.search(text))


def _conversation_plan(context: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(context, Mapping):
        return {}
    plan = context.get("conversation_intent_plan")
    return plan if isinstance(plan, Mapping) else {}


def _semantic_non_p0_by_plan(plan: Mapping[str, Any], *, current_norm: str) -> bool:
    if not plan:
        return False
    primary = str(plan.get("primary_intent") or "").strip()
    if primary in {"refund", "legal_threat", "complaint"}:
        return False
    risks = _text_list(plan.get("risk_signals"))
    if risks:
        return False
    if codes_from_current_message(current_norm):
        return False
    return primary in {
        "pricing",
        "price_fix",
        "installment",
        "discount",
        "trial",
        "camp",
        "schedule",
        "format",
        "address",
        "document",
        "matkap",
        "tax",
        "general_consultation",
    }


def _risk_code_from_plan_primary(primary: str) -> str:
    return {
        "refund": "refund",
        "legal_threat": "legal",
        "complaint": "complaint",
    }.get(primary, primary)


def _normalize_plan_risk(value: str) -> str:
    text = str(value or "").strip()
    if text == "legal_threat":
        return "legal"
    return text


def _primary_risk(codes: Sequence[str], *, current_codes: Sequence[str] = ()) -> str:
    current_present = set(current_codes)
    if "legal" in current_present:
        return "legal"
    if "refund" in current_present:
        return "refund"
    if "complaint" in current_present:
        return "complaint"
    if "reputation_threat" in current_present:
        return "reputation_threat"
    if "payment_dispute" in current_present:
        return "payment_dispute"
    priority = ("legal", "refund", "complaint", "payment_dispute", "reputation_threat")
    present = set(codes)
    for item in priority:
        if item in present:
            return item
    return str(codes[0]) if codes else ""


def _first_match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return " ".join(match.group(0).split())[:160] if match else ""


def _text_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item or "").strip() for item in value if str(item or "").strip())
    return ()


def _normalize(text: Any) -> str:
    return " ".join(str(text or "").casefold().replace("ё", "е").split())
