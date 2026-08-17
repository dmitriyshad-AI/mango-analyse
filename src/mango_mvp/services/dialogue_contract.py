"""Canonical dialogue contract for Mango calls (ТЗ-01/ТЗ-02, этапы A и B).

One strict parser/projection shared by the Google publisher, Analyse, Resolve
and the offline Excel/AI Office exports: stable ``turn_id``, one source line =
one turn, neutral but *distinguishable* speakers, and a fail-closed role
attribution.  ``Менеджер``/``Клиент`` appear only when Mango itself proved the
channel roles with immutable, call-bound evidence that still matches the stored
dialogue line for line; text heuristics, direction, file name, greeting and
channel order never grant trust.

Fail-closed by construction: a broken ``transcript_variants_json``, a malformed
or empty line, a backwards timecode or a partially readable ``dialogue_lines``
raises :class:`DialogueContractError` instead of silently shrinking the call.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

from mango_mvp.productization.contracts import stable_event_key
from mango_mvp.quality.tenant_text_normalizer import (
    TENANT_TEXT_ENGINE_VERSION,
    normalize_manager_text_with_provenance,
    tenant_ruleset_version,
)

CONTRACT_VERSION = "canonical_dialogue_v1"
ROLE_GUARD_VERSION = "role_guard_v1"
# ТЗ-03/ТЗ-04 contract versions live next to the dialogue they are computed
# from, so Analyse and the Google publisher read one definition instead of two.
ANALYSIS_SCHEMA_VERSION_V3 = "v3"
CLAIM_CONTRACT_VERSION = "analyse_v3_claim_evidence_1"
DETECTOR_CONTRACT_VERSION = "deterministic_detector_v1"
HISTORY_SUMMARY_CONTRACT_VERSION = "history_summary_v3_2"
TIMEZONE_CONTRACT_VERSION = "timezone_msk_v1"
# The production service is single-tenant; the value is explicit and versioned
# rather than implicit, so a tenant-specific ruleset can never fire by default.
CALLS_TENANT_ID = "mango"
CALLS_PROVIDER_ID = "mango_office"
MOSCOW_TZ = ZoneInfo("Europe/Moscow")
RESULT_STATUSES = (
    "information_only", "follow_up_agreed", "appointment_agreed", "offer_sent",
    "sale_agreed", "payment_confirmed", "refusal", "no_decision",
    "non_conversation",
)
RESULT_STATUS_RU = {
    "information_only": "получена или уточнена информация",
    "follow_up_agreed": "согласован повторный контакт",
    "appointment_agreed": "согласована встреча или занятие",
    "offer_sent": "предложение отправлено",
    "sale_agreed": "покупка согласована",
    "payment_confirmed": "менеджер сообщил о получении оплаты",
    "refusal": "клиент отказался",
    "no_decision": "решение пока не принято",
    "non_conversation": "содержательного разговора не было",
}
PREFERRED_CHANNELS = ("phone", "email", "messenger")
PRICE_SENSITIVITY_VALUES = ("low", "medium", "high")
SOURCE_DIALOGUE_LINES = "dialogue_lines"
SOURCE_TRANSCRIPT_FALLBACK = "transcript_text_fallback"
PROVIDER_EVIDENCE_FIELD = "provider_role_evidence"
SOURCE_RECORDING_ID_FIELD = "source_recording_id"
PROVIDER_EVIDENCE_SOURCE = "mango_office"
PROVIDER_PHRASE_ROLES = ("client", "operator")
PROVIDER_ROLE_TO_INTERNAL = {"operator": "manager", "client": "client"}
PROVIDER_SUCCESS_RESULT = 1000
# The only two ways ``operator``/``client`` can sit on the two physical tracks.
PROVIDER_SIDE_ALIGNMENTS = (
    {"operator": "left", "client": "right"},
    {"operator": "right", "client": "left"},
)
# Two independent ASR engines do not preserve punctuation or phrase boundaries.
# Trust therefore needs strong aggregate agreement for both sides and a clear
# win over the inverse assignment; ambiguous calls remain untrusted.
PROVIDER_ALIGNMENT_MIN_SIDE_SCORE = 0.72
PROVIDER_ALIGNMENT_MIN_MARGIN = 0.18
PROVIDER_ALIGNMENT_MIN_TOKEN_COVERAGE = 0.85
PROVIDER_ALIGNMENT_MIN_TURN_COVERAGE = 0.60
TRUSTED_ROLE_STATUSES = frozenset({"confirmed_multi_signal"})
TRUNCATION_MARKER = "[... часть реплик пропущена по лимиту ...]"
# Kept under the old name for the Analyse tests and callers that import it.
ANALYSIS_TRUNCATION_MARKER = TRUNCATION_MARKER

# Closed list: ТЗ-02 §4 plus the provider-evidence and physical-binding codes
# required by the owner decision of 2026-08-16 — a text-derived
# ``confirmed_multi_signal`` is not proof of roles, so every code below can
# only ever remove trust.
ROLE_REASON_CODES = frozenset(
    {
        "role_mapping_missing", "role_mapping_invalid", "role_mapping_status_not_allowed",
        "role_mapping_not_confirmed", "manager_quality_not_allowed", "unsupported_topology",
        "invalid_channel_mapping", "speaker_correction_revoked_trust", "mono_or_unknown",
        "non_conversation_not_applicable", "provider_evidence_missing", "provider_evidence_invalid",
        "provider_evidence_call_mismatch", "provider_evidence_dialogue_mismatch",
        "provider_recording_binding_missing", "provider_recording_binding_mismatch",
        "provider_evidence_no_channel_binding", "provider_evidence_ambiguous_sides",
        "provider_evidence_internal_call", "transcript_text_fallback",
        "unknown_speaker_label", "missing_physical_binding", "empty_dialogue",
        "dialogue_unreadable", "ambiguous_cross_speaker_timecode",
    }
)

# One short Russian sentence per code: the owner and the sales head read this,
# not the code.  Anything unmapped degrades to a safe generic sentence.
ROLE_REASON_RU = {
    "role_mapping_missing": "Mango не прислал разметку дорожек звонка",
    "role_mapping_invalid": "разметка дорожек Mango повреждена",
    "role_mapping_status_not_allowed": "разметка дорожек не подтверждена Mango",
    "role_mapping_not_confirmed": "разметка дорожек помечена как неподтверждённая",
    "manager_quality_not_allowed": "звонок не допущен к оценке качества работы менеджера",
    "unsupported_topology": "сложный звонок: перевод, конференция или дубль дорожек",
    "invalid_channel_mapping": "дорожки не разложены на две разные стороны",
    "speaker_correction_revoked_trust": "модель пыталась переставить говорящего",
    "mono_or_unknown": "запись одноканальная: стороны технически неразличимы",
    "non_conversation_not_applicable": "содержательного разговора в записи нет",
    "provider_evidence_missing": "нет ответа Mango о том, где менеджер, а где клиент",
    "provider_evidence_invalid": "ответ Mango о сторонах разговора не проходит проверку",
    "provider_evidence_call_mismatch": "ответ Mango относится к другому звонку",
    "provider_recording_binding_missing": (
        "исходная запись звонка не связана с идентификатором записи Mango"
    ),
    "provider_recording_binding_mismatch": (
        "расшифровка Mango относится не к той исходной записи звонка"
    ),
    "provider_evidence_dialogue_mismatch": "ответ Mango не совпадает с сохранённым разговором",
    "provider_evidence_no_channel_binding": "по ответу Mango нельзя понять, какая дорожка чья",
    "provider_evidence_ambiguous_sides": "дорожки записи неразличимы: на них один и тот же текст",
    "provider_evidence_internal_call": "внутренний звонок: обе стороны не обычная пара сотрудник-клиент",
    "transcript_text_fallback": "разговор сохранён одним куском без разбивки на реплики",
    "unknown_speaker_label": "в разговоре есть реплика без опознанной стороны",
    "missing_physical_binding": "подпись стороны не привязана к дорожке записи",
    "empty_dialogue": "в записи не нашлось ни одной реплики",
    "dialogue_unreadable": "сохранённый разговор не читается и не проходит разбор",
    "ambiguous_cross_speaker_timecode": (
        "две разные стороны имеют одинаковую отметку времени"
    ),
}

NEUTRAL_SPEAKER_PREFIX = "Спикер"
NEUTRAL_UNDEFINED = "Не определено"
ROLE_SPEAKERS = {"manager": "Менеджер", "client": "Клиент"}
CHANNEL_KINDS = {"left": "channel_left", "right": "channel_right"}
SIDE_LABELS = {
    "дорожка левая": "left",
    "channel_left": "left",
    "left": "left",
    "дорожка правая": "right",
    "channel_right": "right",
    "right": "right",
}
ROLE_LABELS = {"менеджер": "manager", "manager": "manager", "клиент": "client", "client": "client"}
NEUTRAL_LABELS = frozenset({"спикер (не определен)", "спикер (не определён)"})
LINE_RE = re.compile(
    r"^\[(?P<timecode>(?P<approx>~)?(?:(?P<hh>\d{2,}):)?(?P<mm>[0-5]\d):(?P<ss>[0-5]\d(?:\.\d)?))\]\s+"
    r"(?P<label>[^:\[\]]{1,120}?)\s*:\s*(?P<text>.*)$"
)
TECHNICAL_RE = re.compile(r"^(?:source_call_id|sha(?:-?256)?)\s*:", re.IGNORECASE)
PHYSICAL_RE = re.compile(
    r"^(?:CHANNEL_(?:LEFT|RIGHT)|MANAGER|CLIENT|Дорожка\s+(?:левая|правая))\s*:\s*",
    re.IGNORECASE,
)
TEXT_MATCH_RE = re.compile(r"[^\w]+", re.UNICODE)


class DialogueContractError(ValueError):
    """Fail-closed parse error: the caller quarantines the whole call."""


@dataclass(frozen=True)
class DialogueInput:
    version: str
    source: str
    role_attribution: Mapping[str, Any]
    turns: tuple[Mapping[str, Any], ...]
    warnings: tuple[str, ...]
    canonical_sha256: str

    @property
    def trusted(self) -> bool:
        return bool(self.role_attribution.get("trusted"))

    @property
    def needs_review(self) -> bool:
        return not self.trusted

    def render(self, *, max_chars: Optional[int] = None) -> str:
        """Publisher projection: no turn ids, neutral speakers, whole replies.

        ``max_chars`` is the hard cell budget of a spreadsheet.  A conversation
        that does not fit is *not* dropped: whole replies are kept from both
        ends and the gap is marked, so the reader sees that something is
        missing instead of silently reading a shortened call as a complete one.
        """
        return self.select(max_chars=max_chars, with_turn_id=False)["text"]

    def render_for_analysis(self, *, max_chars: Optional[int] = None) -> dict[str, Any]:
        """Whole-turn projection for Analyse; never cuts inside a reply."""
        return self.select(max_chars=max_chars, with_turn_id=True)

    def select(
        self, *, max_chars: Optional[int] = None, with_turn_id: bool
    ) -> dict[str, Any]:
        """One whole-turn budget shared by Analyse and every export."""
        rendered = [
            _render_turn(turn, with_turn_id=with_turn_id) for turn in self.turns
        ]
        kept = _select_turn_indexes(rendered, max_chars)
        parts: list[str] = []
        previous: Optional[int] = None
        for index in kept:
            # A gap at the head, in the middle or at the tail is always visible:
            # nobody may read a cut dialogue as a complete one.
            if index != (0 if previous is None else previous + 1):
                parts.append(TRUNCATION_MARKER)
            parts.append(rendered[index])
            previous = index
        if kept and kept[-1] != len(rendered) - 1:
            parts.append(TRUNCATION_MARKER)
        return {
            "text": "\n".join(parts),
            "selected_turn_ids": [self.turns[index]["turn_id"] for index in kept],
            "selected_turn_count": len(kept),
            "total_turn_count": len(self.turns),
            "truncated": len(kept) != len(self.turns),
        }


def _render_turn(turn: Mapping[str, Any], *, with_turn_id: bool) -> str:
    prefix = f"{turn['turn_id']} " if with_turn_id else ""
    return f"{prefix}{turn['timecode']} {turn['display_speaker']}: {turn['text']}".rstrip()


def _select_turn_indexes(rendered: Sequence[str], max_chars: Optional[int]) -> list[int]:
    """Keep whole turns from both ends until the budget is spent.

    A reply is never cut inside: it is either kept whole or dropped whole.  The
    single truncation marker is paid for up front, so the projection can never
    overrun ``max_chars``.  If not one whole turn fits, the projection fails
    closed instead of handing the model a silently empty prompt.
    """
    if not rendered:
        return []
    total = sum(len(item) for item in rendered) + len(rendered) - 1
    if max_chars is None or max_chars <= 0 or total <= max_chars:
        return list(range(len(rendered)))
    budget = max_chars - len(TRUNCATION_MARKER) - 1
    head: list[int] = []
    tail: list[int] = []
    blocked = {True: False, False: False}
    low, high, used, from_head = 0, len(rendered) - 1, 0, True
    while low <= high and not (blocked[True] and blocked[False]):
        if blocked[from_head]:
            from_head = not from_head
            continue
        index = low if from_head else high
        cost = len(rendered[index]) + 1
        if used + cost > budget:
            # An oversized reply is dropped whole, never shortened in place.
            blocked[from_head] = True
            from_head = not from_head
            continue
        if from_head:
            head.append(index)
            low += 1
        else:
            tail.append(index)
            high -= 1
        used += cost
        from_head = not from_head
    if not head and not tail:
        raise DialogueContractError(
            "dialogue budget fits no whole turn: "
            f"max_chars={max_chars}, "
            f"shortest_edge_turn_chars={min(len(rendered[0]), len(rendered[-1]))}"
        )
    return head + list(reversed(tail))


# --------------------------------------------------------------------------
# Operator-facing error text: one implementation for every pipeline stage.
# --------------------------------------------------------------------------

# Long enough that two different failures never collide in practice, short
# enough to stay readable in a dashboard cell.
SAFE_ERROR_DIGEST_CHARS = 16
_SAFE_ERROR_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.\-]+")


def _safe_error_token(value: Any) -> str:
    """A short technical token; anything else is dropped, never escaped."""
    token = _SAFE_ERROR_TOKEN_RE.sub("_", str(value or "")).strip("_")[:40]
    return token or "unknown"


def safe_error_text(stage: str, exc: BaseException) -> str:
    """``last_error`` that cannot carry one character of the conversation.

    A provider stderr tail, a prompt echo or a merged transcript arrives inside
    the exception message: a name, a phone number, a price, a fragment of the
    prompt.  Truncating that message to N characters does not help, because the
    leak sits at the *front* of it — the first 200 characters of "provider
    echoed the prompt back: …" are exactly the conversation.

    So no part of the message survives.  What is stored is the stage, the
    exception class and an irreversible digest of the message: two failures are
    still comparable ("the same error again") without creating a second log
    that could contain the conversation, and nothing can be read back out.
    """
    digest = hashlib.sha256(str(exc).encode("utf-8", "replace")).hexdigest()
    return (
        f"{_safe_error_token(stage)}: {_safe_error_token(type(exc).__name__)}: "
        f"message_sha256={digest[:SAFE_ERROR_DIGEST_CHARS]}"
    )


def moscow_datetime(value: Optional[datetime]) -> Optional[datetime]:
    """The single UTC→Moscow conversion of the whole pipeline.

    Capture stores naive UTC, so a value without a timezone *is* UTC — reading
    it as local time is how a call drifts by three hours in one export and not
    in the next.  Everything else is converted to UTC first and to Moscow once,
    which makes a second call on an already-converted value a no-op instead of
    a second ``+3``.
    """
    if value is None:
        return None
    utc = (
        value.replace(tzinfo=timezone.utc)
        if value.tzinfo is None
        else value.astimezone(timezone.utc)
    )
    return utc.astimezone(MOSCOW_TZ)


def json_object(value: Any) -> Mapping[str, Any]:
    """Lenient reader kept for legacy identity rendering only."""
    if isinstance(value, Mapping):
        return value
    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def strict_variants(value: Any) -> Mapping[str, Any]:
    """An absent payload is empty; a broken payload is an error, not empty."""
    if isinstance(value, Mapping):
        return value
    raw = "" if value is None else str(value).strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise DialogueContractError("transcript_variants_json is invalid JSON") from exc
    if not isinstance(parsed, Mapping):
        raise DialogueContractError("transcript_variants_json is not an object")
    return parsed


def call_record_view(call: Any) -> dict[str, Any]:
    """Read the three fields the contract needs from an ORM ``CallRecord``."""
    if isinstance(call, Mapping):
        return dict(call)
    return {
        "source_call_id": getattr(call, "source_call_id", ""),
        "source_recording_id": getattr(call, "source_recording_id", ""),
        "source_file": getattr(call, "source_file", ""),
        "transcript_variants_json": getattr(call, "transcript_variants_json", ""),
        "transcript_text": getattr(call, "transcript_text", ""),
    }


def label_side(label: Any) -> Optional[str]:
    return SIDE_LABELS.get(_normalized_label(label))


def label_role(label: Any) -> Optional[str]:
    return ROLE_LABELS.get(_normalized_label(label))


def label_is_neutral(label: Any) -> bool:
    """Our own explicit ``speaker not identified`` label, and nothing else."""
    return str(label or "").strip().lower() in NEUTRAL_LABELS


def _normalized_label(label: Any) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(label or "").strip().lower())


def parse_line(line: Any) -> dict[str, Any]:
    """Strict single-line parser shared by publisher, Analyse and Resolve."""
    raw = str(line or "").strip()
    match = LINE_RE.match(raw)
    if not match:
        raise DialogueContractError("dialogue_lines contains a malformed line")
    text = (match.group("text") or "").strip()
    if not text:
        raise DialogueContractError("dialogue_lines contains a line with empty text")
    return {
        "raw_line": raw,
        "timecode": f"[{match.group('timecode')}]",
        "approximate": bool(match.group("approx")),
        "start_sec": int(match.group("hh") or 0) * 3600
        + int(match.group("mm")) * 60
        + float(match.group("ss")),
        "label": (match.group("label") or "").strip(),
        "text": text,
    }


def parse_dialogue_lines(lines: Any) -> list[dict[str, Any]]:
    """Parse every line or fail: partial loss of a dialogue is forbidden."""
    if not isinstance(lines, list):
        raise DialogueContractError("dialogue_lines is not a list")
    parsed: list[dict[str, Any]] = []
    for raw in lines:
        item = parse_line(raw)
        if parsed and item["start_sec"] < parsed[-1]["start_sec"]:
            raise DialogueContractError("dialogue_lines timecodes go backwards")
        parsed.append(item)
    return parsed


def canonical_provider_phrases(phrases: Any) -> list[dict[str, str]]:
    """Strict chronological ``client``/``operator`` phrases of one recording.

    The documented answer promises a role and a text per phrase, in the order
    they were spoken — and nothing else.  A start time and a left/right channel
    are deliberately *not* required here: demanding a field the official
    envelope never promised is how a hand-written fixture starts proving
    something the real API does not say.  The order of the list is the only
    chronology there is, and it is preserved exactly as received.
    """
    if not isinstance(phrases, list) or not phrases:
        raise DialogueContractError("provider phrases must be a non-empty list")
    canonical: list[dict[str, str]] = []
    for phrase in phrases:
        if isinstance(phrase, Mapping):
            role = phrase.get("role")
            text = phrase.get("text")
        elif (
            isinstance(phrase, Sequence)
            and not isinstance(phrase, (str, bytes))
            and len(phrase) == 2
        ):
            # The public Mango example renders each phrase as
            # ``["operator", "text"]``.  Some captured adapters expose the
            # same two values as named keys; both normalize to one internal
            # shape, while every other sequence still fails closed.
            role, text = phrase
        else:
            raise DialogueContractError("provider phrase is not role plus text")
        if not isinstance(role, str) or role not in PROVIDER_PHRASE_ROLES:
            raise DialogueContractError("provider phrase role is not client/operator")
        if not isinstance(text, str) or not text.strip():
            raise DialogueContractError("provider phrase text is empty")
        canonical.append({"role": role, "text": text.strip()})
    return canonical


def canonical_provider_phrases_sha256(phrases: Any) -> str:
    payload = json.dumps(
        canonical_provider_phrases(phrases),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def provider_names(names: Any) -> dict[str, str]:
    """The documented ``names`` object: ``{"client": ..., "operator": ...}``.

    It is an *object* keyed by the role, not a list of speaker records with a
    channel — that shape was invented by us and could prove a per-side binding
    the API never sent.  Reading it strictly is what makes the invented shape
    fail loudly instead of quietly unlocking names.
    """
    if not isinstance(names, Mapping) or not names:
        raise DialogueContractError("recording_transcripts names is not an object")
    declared: dict[str, str] = {}
    for role, value in names.items():
        key = str(role).strip().lower()
        if not key or key in declared:
            raise DialogueContractError("recording_transcripts name role is duplicated")
        if not isinstance(value, str) or not value.strip():
            raise DialogueContractError("recording_transcripts name is empty")
        declared[key] = value.strip()
    return declared


def is_ordinary_two_party_names(names: Mapping[str, str]) -> bool:
    """One operator and an explicitly external client, not an internal call.

    Mango states plainly that ``operator``/``client`` are roles *of the
    recording*, and that on an internal call both sides can be employees.  Such
    a recording says nothing about how a manager speaks to a customer, so it may
    never reach manager quality scoring.  Mango can label two employees
    ``operator`` and ``client``, so merely different human names are not proof:
    the client must be the documented external marker or a phone number.
    Channel placeholders and two employee names are refused instead of guessed.
    """
    if sorted(names) != sorted(PROVIDER_PHRASE_ROLES):
        return False
    operator = names["operator"].strip()
    client = names["client"].strip()
    if operator.casefold() == client.casefold():
        return False
    if client.casefold() in {"клиент", "client"}:
        return operator.casefold() not in {"канал 1", "канал 2"}
    digits = re.sub(r"\D+", "", client)
    return 10 <= len(digits) <= 15 and operator.casefold() not in {
        "канал 1", "канал 2"
    }


def parse_provider_envelope(payload: Any) -> dict[str, dict[str, Any]]:
    """Strict reader of the official batch ``recording_transcripts`` answer.

    Documented envelope: ``result`` plus ``data``.  The public one-record example
    uses an object; a batch may legitimately contain a list because one request
    carries up to 500 ``recording_id`` values.  Each record carries its own
    ``recording_id``, the ``names`` object and chronological ``phrases``.

    The result is keyed by the recording id the *body* declares.  Nothing is
    reordered and nothing is guessed: a caller that wants one recording asks for
    it by id through :func:`provider_record`, and an answer that does not contain
    it is an error rather than "probably the first one".
    """
    if not isinstance(payload, Mapping):
        raise DialogueContractError("recording_transcripts response is not an object")
    result = payload.get("result")
    if isinstance(result, bool) or not isinstance(result, int):
        raise DialogueContractError("recording_transcripts result code is missing")
    if int(result) != PROVIDER_SUCCESS_RESULT:
        raise DialogueContractError(
            f"recording_transcripts result code is not success: {int(result)}"
        )
    data = payload.get("data")
    if isinstance(data, Mapping):
        entries = [data]
    elif isinstance(data, list) and data:
        entries = data
    else:
        raise DialogueContractError(
            "recording_transcripts data is not a record or non-empty array"
        )
    records: dict[str, dict[str, Any]] = {}
    for record in entries:
        if not isinstance(record, Mapping):
            raise DialogueContractError("recording_transcripts data entry is not an object")
        recording_id = str(record.get("recording_id") or "").strip()
        if not recording_id:
            raise DialogueContractError("recording_transcripts record has no recording_id")
        if recording_id in records:
            raise DialogueContractError("recording_transcripts record is duplicated")
        records[recording_id] = {
            "recording_id": recording_id,
            "names": provider_names(record.get("names")),
            "phrases": canonical_provider_phrases(record.get("phrases")),
        }
    return records


def provider_record(payload: Any, recording_id: Any) -> dict[str, Any]:
    """Exactly the requested record of a batch answer, without reordering."""
    wanted = str(recording_id or "").strip()
    if not wanted:
        raise DialogueContractError("recording_transcripts needs a recording_id to extract")
    record = parse_provider_envelope(payload).get(wanted)
    if record is None:
        raise DialogueContractError("recording_transcripts answer has no such recording")
    return record


def provider_raw_response_record(raw_response: Any, recording_id: Any) -> dict[str, Any]:
    """Re-derive one record from the stored raw body — the only editable-proof."""
    if not isinstance(raw_response, str) or not raw_response.strip():
        raise DialogueContractError("provider raw_response is missing")
    try:
        payload = json.loads(raw_response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise DialogueContractError("provider raw_response is invalid JSON") from exc
    return provider_record(payload, recording_id)


def _counter_overlap_count(first: Sequence[str], second: Sequence[str]) -> int:
    return sum((Counter(first) & Counter(second)).values())


def _multiset_dice(first: Sequence[str], second: Sequence[str]) -> float:
    if not first or not second:
        return 0.0
    return 2.0 * _counter_overlap_count(first, second) / (len(first) + len(second))


def _character_ngrams(value: str, size: int = 3) -> list[str]:
    compact = value.replace(" ", "")
    if len(compact) < size:
        return [compact] if compact else []
    return [compact[index : index + size] for index in range(len(compact) - size + 1)]


def _token_order_score(first: Sequence[str], second: Sequence[str]) -> float:
    """Linear position score for repeated tokens in two ordered streams."""
    if not first or not second:
        return 0.0

    def positions(tokens: Sequence[str]) -> dict[tuple[str, int], float]:
        seen: Counter[str] = Counter()
        denominator = max(1, len(tokens) - 1)
        result: dict[tuple[str, int], float] = {}
        for index, token in enumerate(tokens):
            seen[token] += 1
            result[(token, seen[token])] = index / denominator
        return result

    first_positions = positions(first)
    second_positions = positions(second)
    common = first_positions.keys() & second_positions.keys()
    if not common:
        return 0.0
    return sum(
        1.0 - abs(first_positions[key] - second_positions[key])
        for key in common
    ) / len(common)


def provider_side_alignment_report(
    phrases: Sequence[Mapping[str, Any]], turns: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Measurable, PII-free proof that provider roles match physical tracks."""
    if not phrases or not turns:
        return {"alignment": None, "reason": "provider_evidence_dialogue_mismatch"}
    sides = [str(turn.get("side") or "") for turn in turns]
    roles = [str(phrase.get("role") or "") for phrase in phrases]
    if set(sides) != {"left", "right"} or set(roles) != set(PROVIDER_PHRASE_ROLES):
        return {"alignment": None, "reason": "provider_evidence_no_channel_binding"}

    def normalized_text(items: Sequence[Mapping[str, Any]], key: str, value: str) -> str:
        parts = [
            TEXT_MATCH_RE.sub(" ", str(item.get("text") or "").casefold()).strip()
            for item in items
            if str(item.get(key) or "") == value
        ]
        return " ".join(part for part in parts if part)

    provider_text = {
        role: normalized_text(phrases, "role", role) for role in PROVIDER_PHRASE_ROLES
    }
    side_text = {
        side: normalized_text(turns, "side", side) for side in ("left", "right")
    }
    if not all(provider_text.values()) or not all(side_text.values()):
        return {"alignment": None, "reason": "provider_evidence_dialogue_mismatch"}
    if side_text["left"] == side_text["right"]:
        return {"alignment": None, "reason": "provider_evidence_ambiguous_sides"}

    def similarity(first: str, second: str) -> float:
        # Counter-based Dice scores are linear, monotonic under small edits and
        # do not collapse on long repetitive Russian text like SequenceMatcher's
        # popularity heuristic.  Word order remains represented by trigrams.
        char_score = _multiset_dice(_character_ngrams(first), _character_ngrams(second))
        token_score = _multiset_dice(first.split(), second.split())
        return (char_score + token_score) / 2.0

    provider_tokens = {role: provider_text[role].split() for role in PROVIDER_PHRASE_ROLES}
    side_tokens = {side: side_text[side].split() for side in ("left", "right")}
    turn_tokens = [
        (
            str(turn.get("side") or ""),
            TEXT_MATCH_RE.sub(" ", str(turn.get("text") or "").casefold()).strip().split(),
        )
        for turn in turns
        if str(turn.get("side") or "") in {"left", "right"}
    ]

    def role_runs(values: Sequence[str]) -> list[str]:
        runs: list[str] = []
        for value in values:
            if value and (not runs or runs[-1] != value):
                runs.append(value)
        return runs

    provider_runs = role_runs(roles)
    scored: list[tuple[float, float, dict[str, str], dict[str, Any]]] = []
    for candidate in PROVIDER_SIDE_ALIGNMENTS:
        pair_scores = [
            similarity(provider_text[role], side_text[candidate[role]])
            for role in PROVIDER_PHRASE_ROLES
        ]
        order_scores = [
            _token_order_score(
                provider_tokens[role], side_tokens[candidate[role]]
            )
            for role in PROVIDER_PHRASE_ROLES
        ]
        role_for_side = {side: role for role, side in candidate.items()}
        dialogue_runs = role_runs([role_for_side.get(side, "") for side in sides])
        side_coverages: list[float] = []
        provider_coverages: list[float] = []
        provider_phrase_coverages: list[float] = []
        short_phrase_exact_matches: list[float] = []
        turn_coverages: list[float] = []
        for side in ("left", "right"):
            role = role_for_side[side]
            matched = _counter_overlap_count(side_tokens[side], provider_tokens[role])
            side_coverages.append(matched / max(1, len(side_tokens[side])))
            provider_coverages.append(matched / max(1, len(provider_tokens[role])))
        used_short_turns: set[int] = set()
        for phrase in phrases:
            role = str(phrase.get("role") or "")
            phrase_tokens = (
                TEXT_MATCH_RE.sub(
                    " ", str(phrase.get("text") or "").casefold()
                ).strip().split()
            )
            if not phrase_tokens:
                continue
            matched = _counter_overlap_count(
                phrase_tokens, side_tokens[candidate[role]]
            )
            provider_phrase_coverages.append(matched / len(phrase_tokens))
            if len(phrase_tokens) <= 2:
                matched_index = next(
                    (
                        index
                        for index, (side, tokens) in enumerate(turn_tokens)
                        if index not in used_short_turns
                        and side == candidate[role]
                        and tokens == phrase_tokens
                    ),
                    None,
                )
                short_phrase_exact_matches.append(float(matched_index is not None))
                if matched_index is not None:
                    used_short_turns.add(matched_index)
        for side, tokens in turn_tokens:
            if not tokens:
                continue
            role = role_for_side[side]
            matched = _counter_overlap_count(tokens, provider_tokens[role])
            turn_coverages.append(matched / len(tokens))
        metrics = {
            "min_side_score": round(min(pair_scores), 6),
            "mean_side_score": round(sum(pair_scores) / len(pair_scores), 6),
            "min_side_order_score": round(min(order_scores), 6),
            "min_side_token_coverage": round(min(side_coverages), 6),
            "min_provider_token_coverage": round(min(provider_coverages), 6),
            "min_provider_phrase_coverage": round(
                min(provider_phrase_coverages, default=1.0), 6
            ),
            "min_short_phrase_exact_match": round(
                min(short_phrase_exact_matches, default=1.0), 6
            ),
            "min_substantial_turn_coverage": round(min(turn_coverages, default=1.0), 6),
            "substantial_turn_count": len(turn_coverages),
            "role_run_sequence_equal": dialogue_runs == provider_runs,
        }
        scored.append((min(pair_scores), sum(pair_scores) / len(pair_scores), candidate, metrics))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best, alternative = scored
    report = {
        "alignment": dict(best[2]),
        "reason": None,
        "best": best[3],
        "alternative": alternative[3],
        "mean_score_margin": round(best[1] - alternative[1], 6),
        "thresholds": {
            "min_side_score": PROVIDER_ALIGNMENT_MIN_SIDE_SCORE,
            "min_side_order_score": PROVIDER_ALIGNMENT_MIN_SIDE_SCORE,
            "min_mean_score_margin": PROVIDER_ALIGNMENT_MIN_MARGIN,
            "min_token_coverage": PROVIDER_ALIGNMENT_MIN_TOKEN_COVERAGE,
            "min_provider_phrase_coverage": PROVIDER_ALIGNMENT_MIN_TURN_COVERAGE,
            "min_short_phrase_exact_match": 1.0,
            "min_substantial_turn_coverage": PROVIDER_ALIGNMENT_MIN_TURN_COVERAGE,
        },
    }
    if best[0] < PROVIDER_ALIGNMENT_MIN_SIDE_SCORE:
        report.update(alignment=None, reason="provider_evidence_dialogue_mismatch")
    elif best[1] - alternative[1] < PROVIDER_ALIGNMENT_MIN_MARGIN:
        report.update(alignment=None, reason="provider_evidence_ambiguous_sides")
    elif (
        best[3]["min_side_token_coverage"] < PROVIDER_ALIGNMENT_MIN_TOKEN_COVERAGE
        or best[3]["min_side_order_score"] < PROVIDER_ALIGNMENT_MIN_SIDE_SCORE
        or best[3]["min_provider_token_coverage"] < PROVIDER_ALIGNMENT_MIN_TOKEN_COVERAGE
        or best[3]["min_provider_phrase_coverage"] < PROVIDER_ALIGNMENT_MIN_TURN_COVERAGE
        or best[3]["min_short_phrase_exact_match"] < 1.0
        or best[3]["min_substantial_turn_coverage"] < PROVIDER_ALIGNMENT_MIN_TURN_COVERAGE
        or not best[3]["role_run_sequence_equal"]
    ):
        report.update(alignment=None, reason="provider_evidence_dialogue_mismatch")
    return report


def stored_side_by_role(variants: Mapping[str, Any]) -> dict[str, str]:
    """Physical track of each role exactly as the producer stored it.

    New recordings label every line with its physical track, but dialogues
    written before that carry ``Менеджер``/``Клиент``.  The side of such a line
    is never guessed from the label: it is read from the two
    ``physical_channel`` values stored next to the role texts, and only when
    they really are the two opposite tracks and agree with ``role_mapping``.
    Anything else returns nothing, and the line stays without a physical
    binding — which keeps the call untrusted.
    """
    sides: dict[str, str] = {}
    for role in ("manager", "client"):
        block = variants.get(role)
        if not isinstance(block, Mapping):
            return {}
        side = str(block.get("physical_channel") or "").strip().lower()
        if side not in CHANNEL_KINDS:
            return {}
        sides[role] = side
    if sides["manager"] == sides["client"]:
        return {}
    pair = _channel_pair(variants.get("role_mapping"))
    if pair is not None and {side: role for role, side in sides.items()} != dict(pair):
        # The stored sides and the stored mapping disagree about the same call.
        return {}
    return sides


def _channel_pair(mapping: Any) -> Optional[dict[str, str]]:
    if not isinstance(mapping, Mapping):
        return None
    pair = {side: str(mapping.get(side) or "").lower() for side in ("left", "right")}
    return pair if sorted(pair.values()) == ["client", "manager"] else None


def _provider_evidence_check(
    variants: Mapping[str, Any],
    pair: Optional[Mapping[str, str]],
    source_call_id: str,
    source_recording_id: str,
    turns: Sequence[Mapping[str, Any]],
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Only Mango's own, call-bound, re-derivable channel roles unlock names.

    Everything is recomputed here from the one artefact that cannot be edited
    field by field — the stored raw provider body.  The body must be present,
    must hash to ``raw_response_sha256``, must parse as the official envelope,
    and must still contain the recording the evidence claims.  No convenience
    copy of the same fact is read: a second, independently forgeable field
    (``channels``, a stored ``phrases`` list) is exactly what made inverting one
    value enough to swap the sides.

    The side binding is not read anywhere — it is derived from strong aggregate
    agreement of the two role texts with the two physical tracks.  The M1 pilot
    must calibrate the conservative score on real provider responses before any
    content cutover.

    ponytail: capture does not store this field yet, so production stays
    untrusted by construction.  Ceiling: the M1 1→10 ladder captures a real
    ``/vpbx/queries/recording_transcripts`` answer and proves the derived
    binding against audio before any content cutover.
    """
    evidence = variants.get(PROVIDER_EVIDENCE_FIELD)
    if evidence is None or evidence == {} or evidence == "":
        return "provider_evidence_missing", None
    if not isinstance(evidence, Mapping):
        return "provider_evidence_invalid", None
    if str(evidence.get("provider") or "") != PROVIDER_EVIDENCE_SOURCE:
        return "provider_evidence_invalid", None
    claimed_call_id = str(evidence.get("source_call_id") or "").strip()
    if not source_call_id or claimed_call_id != source_call_id:
        return "provider_evidence_call_mismatch", None
    raw_response = evidence.get("raw_response")
    if not isinstance(raw_response, str) or not raw_response.strip():
        return "provider_evidence_invalid", None
    raw_digest = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
    if str(evidence.get("raw_response_sha256") or "") != raw_digest:
        return "provider_evidence_invalid", None
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(evidence.get("batch_response_sha256") or "")
    ):
        return "provider_evidence_invalid", None
    recording_id = str(evidence.get("recording_id") or "").strip()
    if not recording_id:
        return "provider_evidence_invalid", None
    # The transcript answer and the call manifest are independent inputs.  A
    # raw answer that is internally valid may still describe another recording;
    # only the recording id captured with the source call can bind it here.
    # This value comes from the capture manifest through its own DB column.  A
    # copy inside ``transcript_variants_json`` would be mutable together with
    # the evidence and therefore could not prove that the right recording was
    # attached to this call.
    source_recording_id = str(source_recording_id or "").strip()
    if not source_recording_id:
        return "provider_recording_binding_missing", None
    if source_recording_id != recording_id:
        return "provider_recording_binding_mismatch", None
    try:
        record = provider_raw_response_record(raw_response, recording_id)
        phrases = record["phrases"]
        if str(evidence.get("phrases_sha256") or "") != canonical_provider_phrases_sha256(
            phrases
        ):
            return "provider_evidence_invalid", None
    except DialogueContractError:
        return "provider_evidence_invalid", None
    if not is_ordinary_two_party_names(record["names"]):
        return "provider_evidence_internal_call", None
    alignment_report = provider_side_alignment_report(phrases, turns)
    alignment = alignment_report.get("alignment")
    reason = alignment_report.get("reason")
    if reason is not None or alignment is None:
        return reason or "provider_evidence_dialogue_mismatch", alignment_report
    proven = {
        side: PROVIDER_ROLE_TO_INTERNAL[role] for role, side in alignment.items()
    }
    if proven != dict(pair or {}):
        return "provider_evidence_dialogue_mismatch", alignment_report
    return None, alignment_report


def evaluate_role_attribution(
    variants: Mapping[str, Any],
    *,
    source_call_id: str = "",
    source_recording_id: str = "",
    turns: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    mapping = variants.get("role_mapping")
    pair = _channel_pair(mapping)
    reasons: set[str] = set()
    if not isinstance(mapping, Mapping) or not mapping:
        reasons.add("role_mapping_missing")
    else:
        if str(variants.get("mode") or "") != "stereo":
            reasons.add("mono_or_unknown")
        if str(mapping.get("status") or "") not in TRUSTED_ROLE_STATUSES:
            reasons.add("role_mapping_status_not_allowed")
        if mapping.get("confirmed") is not True:
            reasons.add("role_mapping_not_confirmed")
        if mapping.get("manager_quality_allowed") is not True:
            reasons.add("manager_quality_not_allowed")
        if str(mapping.get("topology") or "") != "simple_two_party":
            reasons.add("unsupported_topology")
        if pair is None:
            reasons.add("invalid_channel_mapping")
    evidence_reason, alignment_report = _provider_evidence_check(
        variants, pair, source_call_id, source_recording_id, turns
    )
    if evidence_reason:
        reasons.add(evidence_reason)
    result = {
        "version": ROLE_GUARD_VERSION,
        "decision": "untrusted" if reasons else "trusted",
        "trusted": not reasons,
        "topology": str(mapping.get("topology") or "") if isinstance(mapping, Mapping) else "",
        "reason_codes": sorted(reasons),
    }
    if alignment_report is not None:
        result["provider_alignment"] = alignment_report
    return result


def _fallback_turns(variants: Mapping[str, Any], record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One honest blob: no line boundaries survived, so no speaker is claimed."""
    full = variants.get("full")
    final = full.get("final") if isinstance(full, Mapping) else None
    cleaned: list[str] = []
    for raw_line in str(final or record.get("transcript_text") or "").strip().splitlines():
        line = raw_line.strip()
        if TECHNICAL_RE.match(line):
            continue
        line = PHYSICAL_RE.sub("", line).strip()
        if line:
            cleaned.append(line)
    text = " ".join(cleaned).strip()
    if not text:
        return []
    return [
        {
            "start_sec": 0.0,
            "timecode": "[00:00.0]",
            "approximate": False,
            "side": "",
            "speaker_key": "",
            "defect": "unknown_speaker_label",
            "text": text,
        }
    ]


def _neutral_speaker_names(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Distinguishable neutral names without claiming a role or a side.

    Production dialogues carry ``Менеджер (Имя)``/``Клиент`` labels that we do
    not believe.  Collapsing all of them to ``Не определено`` would destroy the
    dialogue as a dialogue — the reader could no longer tell two people apart.
    So each speaker whose *distinctness* is established gets a stable letter,
    and the letter says only "another person", never who.

    A letter is a claim in itself, so it is only given where distinctness is
    real: a physical channel, or a label the contract recognises.  A line whose
    speaker is genuinely unknown (``Спикер 1``, or our own explicit "speaker not
    identified") carries no such proof and stays ``Не определено`` — two of them
    may well be the same person, and lettering them would invent two.

    Physical sides keep fixed letters whenever the call has any side at all:
    the left track is ``Спикер A`` in every call, so two rows of the report can
    be compared instead of each using its own numbering.
    """
    ordered: list[str] = ["left", "right"] if any(row["side"] for row in rows) else []
    for row in rows:
        key = row["speaker_key"]
        if key and key not in ordered:
            ordered.append(key)
    return {
        key: f"{NEUTRAL_SPEAKER_PREFIX} {_letter(index)}" for index, key in enumerate(ordered)
    }


def _letter(index: int) -> str:
    return string.ascii_uppercase[index] if index < 26 else f"S{index + 1}"


def build_dialogue_input(record: Mapping[str, Any]) -> DialogueInput:
    """Build the canonical dialogue of one stored call without touching raw data.

    One stored line is one turn: neighbouring replies are never glued together,
    because этап C has to quote an exact ``turn_id`` with its own timecode.
    """
    source_call_id = str(record.get("source_call_id") or "").strip()
    source_recording_id = str(record.get("source_recording_id") or "").strip()
    variants = strict_variants(record.get("transcript_variants_json"))
    lines = variants.get("dialogue_lines")
    source = SOURCE_DIALOGUE_LINES
    rows: list[dict[str, Any]] = []
    if lines is None or (isinstance(lines, list) and not lines):
        source = SOURCE_TRANSCRIPT_FALLBACK
        rows = _fallback_turns(variants, record)
    else:
        stored_sides = stored_side_by_role(variants)
        for item in parse_dialogue_lines(lines):
            side = label_side(item["label"])
            role = label_role(item["label"])
            if not side and role:
                # Legacy line: recover the physical track from what the producer
                # stored, never from the role word itself.
                side = stored_sides.get(role, "")
            defect = "" if side else ("missing_physical_binding" if role else "unknown_speaker_label")
            if side:
                speaker_key = side
            elif role:
                # Not believed as a role, but still a distinct known speaker.
                speaker_key = f"label:{_normalized_label(item['label'])}"
            else:
                # Nothing proves this line belongs to a speaker of its own.
                speaker_key = ""
            rows.append(
                {
                    "start_sec": item["start_sec"],
                    "timecode": item["timecode"],
                    "approximate": item["approximate"],
                    "side": side or "",
                    "speaker_key": speaker_key,
                    "defect": defect,
                    "text": item["text"],
                }
            )

    attribution = evaluate_role_attribution(
        variants,
        source_call_id=source_call_id,
        source_recording_id=source_recording_id,
        turns=rows,
    )
    pair = _channel_pair(variants.get("role_mapping"))
    reasons = set(attribution["reason_codes"])
    reasons.update(row["defect"] for row in rows if row["defect"])
    if any(
        current["start_sec"] == previous["start_sec"]
        and current["speaker_key"]
        and previous["speaker_key"]
        and current["speaker_key"] != previous["speaker_key"]
        for previous, current in zip(rows, rows[1:])
    ):
        reasons.add("ambiguous_cross_speaker_timecode")
    if source == SOURCE_TRANSCRIPT_FALLBACK:
        reasons.add(SOURCE_TRANSCRIPT_FALLBACK)
    if not rows:
        # An empty dialogue is not a proven non-conversation: nobody looked at
        # the audio.  It needs a human, so it is untrusted, not not_applicable.
        reasons.add("empty_dialogue")
    trusted = not reasons
    attribution = {
        **attribution,
        "decision": "trusted" if trusted else "untrusted",
        "trusted": trusted,
        "reason_codes": sorted(reasons),
        "source": source,
    }
    neutral = _neutral_speaker_names(rows)
    turns: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if trusted and pair and row["side"]:
            kind = pair[row["side"]]
            display = ROLE_SPEAKERS[kind]
        else:
            kind = CHANNEL_KINDS[row["side"]] if row["side"] else "unknown"
            display = neutral.get(row["speaker_key"]) or NEUTRAL_UNDEFINED
        turns.append(
            {
                "turn_id": f"T{index:04d}",
                "start_sec": row["start_sec"],
                "timecode": row["timecode"],
                "approximate": row["approximate"],
                "physical_side": row["side"],
                "speaker_kind": kind,
                "display_speaker": display,
                "text": row["text"],
            }
        )
    payload = json.dumps(turns, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return DialogueInput(
        version=CONTRACT_VERSION,
        source=source,
        role_attribution=attribution,
        turns=tuple(turns),
        warnings=tuple(sorted(reasons)),
        canonical_sha256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )


def trusted_role_text(record: Mapping[str, Any], role: str) -> str:
    """Return role text only when this exact call has trusted Mango evidence."""
    if role not in {"manager", "client"}:
        raise ValueError("role must be manager or client")
    try:
        dialogue = build_dialogue_input(record)
    except DialogueContractError:
        return ""
    if not dialogue.trusted:
        return ""
    return "\n".join(
        str(turn.get("text") or "").strip()
        for turn in dialogue.turns
        if turn.get("speaker_kind") == role and str(turn.get("text") or "").strip()
    )


# --------------------------------------------------------------------------
# Fail-closed projection of an analysis whose roles are not proven (ТЗ-02 §6).
# --------------------------------------------------------------------------

UNTRUSTED_PROJECTION_VERSION = "untrusted_analysis_projection_v1"
NEUTRAL_TOPIC_VERSION = "neutral_topic_v1"
UNTRUSTED_SUMMARY = (
    "Стороны разговора не подтверждены технической разметкой Mango. "
    "Кто что сказал, к чему пришли и какой следующий шаг — по этой записи "
    "автоматически не определяется и требует ручной проверки."
)
UNTRUSTED_FOLLOW_UP_REASON = (
    "Роли сторон не подтверждены: оценка и следующий шаг требуют проверки."
)
INVALID_STORED_SUMMARY = (
    "Сохранённый анализ не прошёл текущую проверку качества. "
    "Расшифровка доступна, но коммерческие выводы нужно сформировать заново."
)
# Closed deterministic vocabulary: a neutral topic of the conversation is the
# only thing that survives, and it is derived from the dialogue text by this
# table alone — never from the model's free text.
NEUTRAL_TOPIC_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("математика", ("математик", "алгебр", "геометри")),
    ("физика", ("физик",)),
    ("информатика", ("информатик", "программирован", "python", "питон")),
    ("русский язык", ("русский язык", "русского языка", "русскому языку")),
    ("подготовка к ЕГЭ", ("егэ",)),
    ("подготовка к ОГЭ", ("огэ",)),
    ("олимпиады", ("олимпиад",)),
    ("лагерь", ("лагер",)),
    ("летняя школа", ("летняя школа", "летней школы", "летнюю школу")),
    ("онлайн-формат", ("онлайн", "дистанцион")),
    ("очный формат", ("очно", "очный", "очное", "очная")),
    ("расписание", ("расписан",)),
    ("стоимость и оплата", ("стоимост", "цена", "цену", "оплат", "рассрочк", "скидк")),
    ("документы", ("договор", "справк", "документ")),
)
# Technical telemetry only: no judgement about people, money or next steps.
UNTRUSTED_QUALITY_FLAG_ALLOWLIST = frozenset(
    {
        "mode", "secondary_provider", "secondary_backfill_status",
        "analyze_prompt_profile", "analyze_prompt_version", "analyze_prompt_truncated",
        "analyze_transcript_chars_original", "analyze_transcript_chars_prompt",
        "analysis_input_sha256", "analysis_prompt_sha256", "dialogue_version", "dialogue_source",
        "dialogue_canonical_sha256", "dialogue_turn_count",
        "dialogue_selected_turn_count", "dialogue_total_turn_count",
        "analysis_cache_hit", "analysis_model_called",
    }
)
UNTRUSTED_META_ALLOWLIST = frozenset(
    {
        "analysis_model", "analysis_provider", "analysis_prompt_version",
        "analysis_prompt_profile", "analyzed_at", "analysis_source_sha256",
        "model_called", "model_call_count", "cache_hit", "cache_hit_count",
        # ТЗ-04/ТЗ-05: closed technical version identifiers.  They carry no
        # statement about the call, and the Google fingerprint needs them to
        # notice that a contract moved under an unchanged stored payload.
        "analysis_input_sha256", "analysis_prompt_sha256", "analysis_schema_version",
        "dialogue_contract_version", "dialogue_canonical_sha256",
        "role_guard_version", "prompt_contract_version", "claim_contract_version",
        "detector_contract_version", "history_summary_contract_version",
        "normalizer_engine_version", "normalizer_ruleset_version",
        "normalizer_tenant_id", "timezone_contract_version",
        "manager_output_sha256",
    }
)
TOKEN_USAGE_KEYS = frozenset(
    {"source", "prompt_tokens", "completion_tokens", "total_tokens"}
)
TOKEN_USAGE_SOURCES = frozenset(
    {
        "provider", "unavailable", "skipped_untrusted_role",
        "skipped_deterministic", "cache_hit",
    }
)
_SCALARS = (str, int, float, bool)
# A leading =, +, - or @ turns a cell into a formula in Google Sheets and Excel.
# Even a technical flag is copied into a spreadsheet downstream, so a value that
# would execute there never leaves the projection.
FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def neutral_topics(dialogue: DialogueInput) -> list[str]:
    """Closed deterministic topics of the conversation, in a fixed order."""
    haystack = " ".join(turn["text"] for turn in dialogue.turns).lower()
    return [name for name, terms in NEUTRAL_TOPIC_TERMS if any(t in haystack for t in terms)]


# The closed claim reason codes live next to their Russian sentences so that
# Analyse writes the code and the Google publisher reads it without a second,
# drifting copy of the same vocabulary.
CLAIM_REASON_PREFIX = "claim_evidence_missing_or_invalid"
CLAIM_FIELD_LABELS_RU = {
    "structured_fields.result.status": "исход разговора",
    "structured_fields.result.detail": "детали исхода разговора",
    "structured_fields.objections": "возражение или причина",
    "structured_fields.next_step.action": "следующий шаг",
    "structured_fields.next_step.due": "срок следующего шага",
    "structured_fields.interests.products": "продукт",
    "structured_fields.interests.format": "формат обучения",
    "structured_fields.interests.subjects": "предмет",
    "structured_fields.interests.exam_targets": "цель подготовки",
    "structured_fields.student.grade_current": "класс ученика",
    "structured_fields.student.school": "школа ученика",
    "structured_fields.people.parent_fio": "имя родителя",
    "structured_fields.people.child_fio": "имя ребёнка",
    "structured_fields.contacts.email": "почта клиента",
    "structured_fields.contacts.preferred_channel": "предпочтительный канал связи",
    "structured_fields.commercial.price_sensitivity": "чувствительность к цене",
    "structured_fields.commercial.budget": "бюджет",
    "structured_fields.commercial.discount_interest": "интерес к скидке",
}
ANALYSIS_REASON_RU = {
    "analysis_contract_invalid": (
        "сохранённый анализ сделан по устаревшим или неполным правилам; нужен повторный анализ"
    ),
    "role_attribution_untrusted": (
        "Mango не подтвердил, какая дорожка принадлежит менеджеру, а какая клиенту"
    ),
    "sales_missing_product_and_next_step": (
        "в звонке о продаже не подтверждены продукт и следующий шаг"
    ),
    "sales_missing_product": "в звонке о продаже не подтверждён продукт",
    "sales_missing_next_step": "в звонке о продаже не подтверждён следующий шаг",
    "sales_service_overlap": "продажа смешана с сервисным обращением",
    "long_non_conversation": "длинный звонок помечен как разговор без содержания",
    "legacy_summary_conflict": "конспект противоречит типу звонка",
    "non_sales_with_sales_signal": (
        "в сервисном звонке есть признак продажи, но нет следующего шага"
    ),
    "resolve_manual_review_required": "расшифровка требует ручной проверки",
    "secondary_asr_exhausted_primary_fallback": (
        "вторая расшифровка не получена, использована только первая"
    ),
    "analyze_prompt_truncated": (
        "разговор не поместился в окно анализа целиком; выводы нужно проверить по полной расшифровке"
    ),
}

MODEL_STRUCTURED_KEYS = frozenset(
    {"result", "people", "contacts", "student", "interests", "commercial", "objections", "next_step"}
)
STORED_STRUCTURED_KEYS = MODEL_STRUCTURED_KEYS | {"lead_priority"}
STRUCTURED_CHILD_KEYS = {
    "result": frozenset({"status", "detail"}),
    "people": frozenset({"parent_fio", "child_fio"}),
    "contacts": frozenset({"email", "preferred_channel"}),
    "student": frozenset({"grade_current", "school"}),
    "interests": frozenset({"products", "format", "subjects", "exam_targets"}),
    "commercial": frozenset({"price_sensitivity", "budget", "discount_interest"}),
    "next_step": frozenset({"action", "due"}),
}
STORED_EVIDENCE_KEYS = frozenset(
    {
        "claim_id", "field_path", "item_id", "evidence_type", "support_type",
        "source", "contract_version", "turn_id", "exact_quote", "timecode",
        "speaker_kind", "start_sec", "dialogue_sha256", "raw_value",
        "value_sha256", "validation_status",
    }
)
NORMALIZED_FACT_KEYS = frozenset(
    {
        "field_path", "item_id", "claim_id", "raw_value", "normalized_value",
        "rule_id", "rule_ids", "engine_version", "ruleset_version", "tenant_id",
        "status",
    }
)


class StoredAnalysisContractError(ValueError):
    """A stored answer is stale, incomplete or no longer reproducible."""


def _optional_text(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _text_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
        and len(value) == len({item.casefold() for item in value})
    )


def validate_structured_fields(value: Any, *, stored: bool) -> dict[str, Any]:
    """Strict nested schema shared by provider answers and stored readers."""
    if not isinstance(value, Mapping):
        raise StoredAnalysisContractError("structured_fields is not an object")
    expected = STORED_STRUCTURED_KEYS if stored else MODEL_STRUCTURED_KEYS
    if set(value) != expected:
        raise StoredAnalysisContractError("structured_fields keys are not current")
    fields = copy.deepcopy(dict(value))
    for block, keys in STRUCTURED_CHILD_KEYS.items():
        item = fields.get(block)
        expected_child_keys = (
            keys | {"phone_from_filename"} if stored and block == "contacts" else keys
        )
        if not isinstance(item, Mapping) or set(item) != expected_child_keys:
            raise StoredAnalysisContractError(f"structured_fields.{block} keys are not current")
        fields[block] = dict(item)
    if stored:
        if fields.get("lead_priority") not in {"hot", "warm", "cold"}:
            raise StoredAnalysisContractError("stored lead_priority is invalid")
    scalar_paths = (
        ("result", "detail"), ("people", "parent_fio"), ("people", "child_fio"),
        ("contacts", "email"), ("contacts", "preferred_channel"),
        ("student", "grade_current"), ("student", "school"),
        ("commercial", "price_sensitivity"), ("commercial", "budget"),
        ("next_step", "action"), ("next_step", "due"),
    )
    if stored:
        scalar_paths += (("contacts", "phone_from_filename"),)
    if not all(_optional_text(fields[block].get(name)) for block, name in scalar_paths):
        raise StoredAnalysisContractError("a structured scalar has an invalid type")
    if fields["result"].get("status") is not None and fields["result"].get("status") not in RESULT_STATUSES:
        raise StoredAnalysisContractError("result.status is invalid")
    if fields["contacts"].get("preferred_channel") is not None and fields["contacts"].get("preferred_channel") not in PREFERRED_CHANNELS:
        raise StoredAnalysisContractError("preferred_channel is invalid")
    if fields["commercial"].get("price_sensitivity") is not None and fields["commercial"].get("price_sensitivity") not in PRICE_SENSITIVITY_VALUES:
        raise StoredAnalysisContractError("price_sensitivity is invalid")
    if fields["commercial"].get("discount_interest") not in {True, False, None}:
        raise StoredAnalysisContractError("discount_interest is invalid")
    for name in ("products", "format", "subjects", "exam_targets"):
        if not _text_list(fields["interests"].get(name)):
            raise StoredAnalysisContractError(f"interests.{name} is invalid")
    if not _text_list(fields.get("objections")):
        raise StoredAnalysisContractError("objections is invalid")
    return fields


def canonical_item_key(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def value_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Only the content a manager or a downstream CRM reader can act on is bound to
# this digest. Runtime telemetry is intentionally excluded because it is added
# after the deterministic business projection has been built.
MANAGER_OUTPUT_KEYS = (
    "history_summary", "history_short", "summary", "manager_brief", "structured_fields",
    "display_fields", "crm_blocks", "claim_evidence", "evidence", "normalized_facts",
    "history_summary_meta", "quality_flags", "needs_review", "review_reasons",
    "objections", "pain_points", "next_step", "timeline", "budget",
    "student_grade", "target_product", "interests", "personal_offer",
    "follow_up_reason", "tags", "follow_up_score", "neutral_topics",
    "review_reasons_ru",
    "result", "call_result",
)


def manager_output_sha256(analysis: Mapping[str, Any]) -> str:
    payload = {
        key: analysis[key]
        for key in MANAGER_OUTPUT_KEYS
        if key in analysis
    }
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def call_key_for_record(record: Mapping[str, Any]) -> str:
    source_call_id = str(record.get("source_call_id") or "").strip()
    if source_call_id:
        return stable_event_key(CALLS_TENANT_ID, CALLS_PROVIDER_ID, source_call_id)
    seed = f"{CALLS_TENANT_ID}:{CALLS_PROVIDER_ID}:local:{record.get('source_file') or ''}"
    return "unresolved:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def deterministic_claim_id(
    *, call_key: str, field_path: str, item_key: str, digest: str, contract_version: str
) -> str:
    payload = json.dumps(
        {
            "contract_version": contract_version,
            "call_key": call_key,
            "field_path": field_path,
            "item_key": item_key,
            "value_sha256": digest,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_display_fields(
    structured_fields: Mapping[str, Any], normalized_facts: Any
) -> dict[str, Any]:
    """Rebuild the human-readable copy from raw fields and current rules."""
    fields = validate_structured_fields(structured_fields, stored=True)
    facts = normalized_facts if isinstance(normalized_facts, list) else []
    seen: set[tuple[str, Optional[str], str]] = set()
    for fact in facts:
        if not isinstance(fact, Mapping) or set(fact) != NORMALIZED_FACT_KEYS:
            raise StoredAnalysisContractError("normalized fact keys are not current")
        field_path = str(fact.get("field_path") or "")
        raw_value = fact.get("raw_value")
        normalized_value = fact.get("normalized_value")
        item_id = fact.get("item_id")
        if (
            field_path not in CLAIM_FIELD_LABELS_RU
            or not isinstance(raw_value, str)
            or not isinstance(normalized_value, str)
            or (item_id is not None and not isinstance(item_id, str))
        ):
            raise StoredAnalysisContractError("normalized fact identity is invalid")
        result = normalize_manager_text_with_provenance(raw_value, tenant_id=CALLS_TENANT_ID)
        if not result.changed or (
            normalized_value != result.normalized_value
            or fact.get("engine_version") != TENANT_TEXT_ENGINE_VERSION
            or fact.get("ruleset_version") != tenant_ruleset_version(CALLS_TENANT_ID)
            or fact.get("tenant_id") != CALLS_TENANT_ID
            or fact.get("status") != result.status
            or fact.get("rule_id") != result.rule_ids[0]
            or list(fact.get("rule_ids") or []) != list(result.rule_ids)
        ):
            raise StoredAnalysisContractError("normalized fact is not reproducible")
        key = (field_path, item_id, raw_value)
        if key in seen:
            continue
        seen.add(key)
        parts = field_path.split(".")[1:]
        container: Any = fields
        for part in parts[:-1]:
            container = container.get(part) if isinstance(container, dict) else None
        if not isinstance(container, dict):
            raise StoredAnalysisContractError("normalized fact path is invalid")
        name = parts[-1]
        current = container.get(name)
        if isinstance(current, list):
            expected_item_id = canonical_item_key(raw_value)
            if item_id != expected_item_id or raw_value not in current:
                raise StoredAnalysisContractError("normalized list item does not match raw field")
            container[name] = [normalized_value if item == raw_value else item for item in current]
        else:
            if item_id is not None or current != raw_value:
                raise StoredAnalysisContractError("normalized scalar does not match raw field")
            container[name] = normalized_value
    return fields


def _stored_claim_values(fields: Mapping[str, Any]) -> dict[str, list[tuple[Optional[str], Any]]]:
    values: dict[str, list[tuple[Optional[str], Any]]] = {}
    for field_path in CLAIM_FIELD_LABELS_RU:
        current: Any = fields
        for part in field_path.split(".")[1:]:
            current = current.get(part) if isinstance(current, Mapping) else None
        if isinstance(current, list):
            values[field_path] = [
                (canonical_item_key(item), item) for item in current if item not in (None, "")
            ]
        elif current not in (None, "", False):
            values[field_path] = [(None, current)]
        else:
            values[field_path] = []
    return values


def validate_stored_analysis(
    record: Mapping[str, Any], analysis: Any, dialogue: DialogueInput
) -> dict[str, Any]:
    """Prove that a stored v3 payload still satisfies today's contract."""
    if not isinstance(analysis, Mapping):
        raise StoredAnalysisContractError("analysis is not an object")
    source = copy.deepcopy(dict(analysis))
    if source.get("analysis_schema_version") != ANALYSIS_SCHEMA_VERSION_V3:
        raise StoredAnalysisContractError("analysis schema is stale")
    if source.get("claim_contract_version") != CLAIM_CONTRACT_VERSION:
        raise StoredAnalysisContractError("claim contract is stale")
    meta = source.get("analysis_meta")
    if not isinstance(meta, Mapping):
        raise StoredAnalysisContractError("analysis_meta is missing")
    expected_meta = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION_V3,
        "dialogue_contract_version": CONTRACT_VERSION,
        "dialogue_canonical_sha256": dialogue.canonical_sha256,
        "role_guard_version": ROLE_GUARD_VERSION,
        "prompt_contract_version": CLAIM_CONTRACT_VERSION,
        "claim_contract_version": CLAIM_CONTRACT_VERSION,
        "detector_contract_version": DETECTOR_CONTRACT_VERSION,
        "history_summary_contract_version": HISTORY_SUMMARY_CONTRACT_VERSION,
        "normalizer_engine_version": TENANT_TEXT_ENGINE_VERSION,
        "normalizer_ruleset_version": tenant_ruleset_version(CALLS_TENANT_ID),
        "normalizer_tenant_id": CALLS_TENANT_ID,
        "timezone_contract_version": TIMEZONE_CONTRACT_VERSION,
    }
    if any(meta.get(key) != value for key, value in expected_meta.items()):
        raise StoredAnalysisContractError("analysis_meta contract is stale")
    input_sha = str(meta.get("analysis_input_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", input_sha):
        raise StoredAnalysisContractError("analysis input sha is missing")
    output_sha = str(meta.get("manager_output_sha256") or "")
    if (
        not re.fullmatch(r"[0-9a-f]{64}", output_sha)
        or output_sha != manager_output_sha256(source)
    ):
        raise StoredAnalysisContractError("manager output integrity failed")
    quality = source.get("quality_flags")
    if not isinstance(quality, Mapping) or (
        quality.get("dialogue_canonical_sha256") != dialogue.canonical_sha256
        or quality.get("analysis_input_sha256") != input_sha
    ):
        raise StoredAnalysisContractError("quality metadata does not match analysis")
    dialogue_input = source.get("dialogue_input")
    if not isinstance(dialogue_input, Mapping) or (
        dialogue_input.get("version") != CONTRACT_VERSION
        or dialogue_input.get("canonical_sha256") != dialogue.canonical_sha256
        or dialogue_input.get("turn_count") != len(dialogue.turns)
    ):
        raise StoredAnalysisContractError("stored dialogue identity is stale")

    fields = validate_structured_fields(source.get("structured_fields"), stored=True)
    if source.get("evidence"):
        raise StoredAnalysisContractError("legacy unbound evidence is not allowed")
    evidence = source.get("claim_evidence")
    if not isinstance(evidence, list):
        raise StoredAnalysisContractError("claim_evidence is missing")
    allowed_values = _stored_claim_values(fields)
    turns = {str(turn["turn_id"]): turn for turn in dialogue.turns}
    matched: set[tuple[str, Optional[str]]] = set()
    call_key = call_key_for_record(record)
    evidence_ids: set[str] = set()
    evidence_groups: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    # Lazy import avoids the module cycle while keeping one detector for write
    # and read validation. Import once, not once per claim.
    from mango_mvp.services.analyze import AnalyzeService

    for entry in evidence:
        if not isinstance(entry, Mapping) or set(entry) != STORED_EVIDENCE_KEYS:
            raise StoredAnalysisContractError("claim evidence keys are not current")
        field_path = str(entry.get("field_path") or "")
        item_id = entry.get("item_id")
        raw_value = entry.get("raw_value")
        candidates = allowed_values.get(field_path)
        if candidates is None or (item_id, raw_value) not in candidates:
            raise StoredAnalysisContractError("claim evidence points outside stored fields")
        turn = turns.get(str(entry.get("turn_id") or ""))
        if turn is None or any(
            entry.get(key) != turn.get(source_key)
            for key, source_key in (
                ("exact_quote", "text"), ("timecode", "timecode"),
                ("speaker_kind", "speaker_kind"), ("start_sec", "start_sec"),
            )
        ):
            raise StoredAnalysisContractError("claim evidence does not match the dialogue turn")
        source_kind = entry.get("source")
        contract_version = (
            CLAIM_CONTRACT_VERSION if source_kind == "model_claim" else DETECTOR_CONTRACT_VERSION
        )
        digest = value_sha256(raw_value)
        expected_claim_id = deterministic_claim_id(
            call_key=call_key,
            field_path=field_path,
            item_key=str(item_id or ""),
            digest=digest,
            contract_version=contract_version,
        )
        if (
            source_kind not in {"model_claim", "deterministic_detector"}
            or entry.get("contract_version") != contract_version
            or entry.get("dialogue_sha256") != dialogue.canonical_sha256
            or entry.get("value_sha256") != digest
            or entry.get("claim_id") != expected_claim_id
            or entry.get("evidence_type") != "explicit"
            or entry.get("support_type") != "explicit"
            or entry.get("validation_status") != "valid"
        ):
            raise StoredAnalysisContractError("claim evidence integrity failed")
        evidence_ids.add(str(entry.get("claim_id")))
        evidence_groups.setdefault(str(entry.get("claim_id")), []).append((entry, turn))
        matched.add((field_path, item_id))
    for group in evidence_groups.values():
        first = group[0][0]
        if any(
            entry.get("field_path") != first.get("field_path")
            or entry.get("item_id") != first.get("item_id")
            or entry.get("raw_value") != first.get("raw_value")
            for entry, _turn in group
        ) or not AnalyzeService._claim_refs_support(
            str(first.get("field_path") or ""),
            first.get("raw_value"),
            [turn for _entry, turn in group],
            list(dialogue.turns),
        ):
            raise StoredAnalysisContractError("claim evidence does not support its value")
    required = {
        (path, item_id)
        for path, values in allowed_values.items()
        for item_id, _value in values
    }
    if matched != required:
        raise StoredAnalysisContractError("not every stored claim has evidence")
    normalized_facts = source.get("normalized_facts")
    if not isinstance(normalized_facts, list) or any(
        not isinstance(fact, Mapping) or str(fact.get("claim_id") or "") not in evidence_ids
        for fact in normalized_facts
    ):
        raise StoredAnalysisContractError("normalized facts do not point at proven claims")
    display = build_display_fields(fields, normalized_facts)
    if source.get("display_fields") != display or source.get("crm_blocks") != display:
        raise StoredAnalysisContractError("manager display is stale or not reproducible")
    return source


def claim_review_sentence_ru(code: Any) -> Optional[str]:
    """One readable sentence for a ``claim_evidence_missing_or_invalid`` code.

    The item part of the code is an opaque key by design, so nothing of the
    conversation reaches the report; the reader gets the field, which is what
    tells them what to listen for.
    """
    raw = str(code or "")
    if not raw.startswith(f"{CLAIM_REASON_PREFIX}:"):
        return None
    field_path = raw.split(":", 1)[1].split("[", 1)[0]
    label = CLAIM_FIELD_LABELS_RU.get(field_path)
    if label is None:
        return "в репликах не нашлось подтверждения одного из выводов"
    return f"в репликах не нашлось подтверждения: {label}"


def review_reasons_ru(reason_codes: Sequence[str]) -> list[str]:
    """Russian sentences for the sales head; unknown codes never leak raw."""
    sentences: list[str] = []
    for code in reason_codes:
        sentence = (
            ROLE_REASON_RU.get(str(code))
            or ANALYSIS_REASON_RU.get(str(code))
            or claim_review_sentence_ru(code)
        )
        if sentence is None:
            sentence = "техническая проверка разметки разговора не пройдена"
        if sentence not in sentences:
            sentences.append(sentence)
    return sentences or ["роли сторон разговора не подтверждены"]


MANAGER_EVIDENCE_PATHS = frozenset(
    {
        "structured_fields.result.status",
        "structured_fields.result.detail",
        "structured_fields.objections",
        "structured_fields.next_step.action",
        "structured_fields.next_step.due",
        "structured_fields.commercial.price_sensitivity",
        "structured_fields.commercial.budget",
        "structured_fields.commercial.discount_interest",
    }
)


def manager_claim_evidence_ru(analysis: Mapping[str, Any]) -> str:
    """Exact replies behind the risky fields shown to the sales head."""
    lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    evidence = analysis.get("claim_evidence")
    for entry in evidence if isinstance(evidence, list) else ():
        if not isinstance(entry, Mapping):
            continue
        path = str(entry.get("field_path") or "")
        if path not in MANAGER_EVIDENCE_PATHS:
            continue
        quote = str(entry.get("exact_quote") or "").strip()
        timecode = str(entry.get("timecode") or "").strip()
        turn_id = str(entry.get("turn_id") or "").strip()
        if not quote or not timecode or not turn_id:
            continue
        key = (path, turn_id, quote)
        if key in seen:
            continue
        seen.add(key)
        label = CLAIM_FIELD_LABELS_RU[path]
        lines.append(f"{label.capitalize()} — {turn_id} {timecode}: «{quote}»")
    return "\n".join(lines)


def manager_result_ru(analysis: Mapping[str, Any]) -> str:
    """One shared Russian rendering of the proven v3 call result."""
    structured = analysis.get("structured_fields")
    result = structured.get("result") if isinstance(structured, Mapping) else None
    display = analysis.get("display_fields")
    display_result = display.get("result") if isinstance(display, Mapping) else None
    if not isinstance(result, Mapping):
        return "—"
    label = RESULT_STATUS_RU.get(str(result.get("status") or "").strip())
    if not label:
        return "—"
    label = label[:1].upper() + label[1:]
    detail = (
        str(display_result.get("detail") or "").strip()
        if isinstance(display_result, Mapping)
        else ""
    )
    return f"{label}: {detail}" if detail else label


def is_formula_like(value: Any) -> bool:
    """A value a spreadsheet would execute instead of showing."""
    return isinstance(value, str) and value.strip()[:1] in FORMULA_PREFIXES


def _allowed_scalars(source: Any, allowlist: frozenset) -> dict[str, Any]:
    """Copy only allowlisted keys, and only as scalars or a flat scalar map.

    Telemetry such as ``token_usage`` is legitimately a small flat object, so a
    Mapping survives — but only one level deep and only with scalar values, so
    a payload can never smuggle a nested structure through a technical key.
    """
    values: dict[str, Any] = {}
    if not isinstance(source, Mapping):
        return values
    for key in sorted(allowlist):
        if key not in source:
            continue
        value = source.get(key)
        if is_formula_like(value):
            continue
        if isinstance(value, Mapping):
            values[key] = {
                str(name): item
                for name, item in value.items()
                if (isinstance(item, _SCALARS) or item is None)
                and not is_formula_like(item)
            }
        elif isinstance(value, _SCALARS) or value is None:
            values[key] = value
    return values


def _safe_token_usage(source: Any) -> dict[str, Any]:
    """Keep only exact technical counters; free text can never ride along."""
    if not isinstance(source, Mapping) or set(source) - TOKEN_USAGE_KEYS:
        return {}
    usage_source = source.get("source")
    if usage_source not in TOKEN_USAGE_SOURCES:
        return {}
    result: dict[str, Any] = {"source": usage_source}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = source.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            return {}
        result[key] = value
    return result


def project_untrusted_analysis(
    analysis: Any, dialogue: DialogueInput
) -> dict[str, Any]:
    """Rebuild an analysis from an allowlist — never strip fields from it.

    Deleting known-bad keys is open by default: the next prompt version, the
    next model or an older stored payload brings a key nobody blacklisted, and
    it reaches the sales head as a proven fact about a call whose sides are not
    even established.  So nothing is copied over: this returns a fresh payload
    containing only the technical version, the neutral topics derived from the
    dialogue text, a fixed neutral summary, the role/dialogue contracts, the
    review reasons and a closed list of technical flags.

    Nothing is reconstructed from the transcript: the topics come from a closed
    vocabulary, and no name, number, price or intent is ever recovered.
    """
    source = analysis if isinstance(analysis, Mapping) else {}
    attribution = dict(dialogue.role_attribution)
    reason_codes = [str(code) for code in attribution.get("reason_codes") or []]
    review_reasons = sorted({*reason_codes, "role_attribution_untrusted"})
    topics = neutral_topics(dialogue)
    dialogue_input = {
        "version": dialogue.version,
        "source": dialogue.source,
        "canonical_sha256": dialogue.canonical_sha256,
        "turn_count": len(dialogue.turns),
    }
    quality_flags = _allowed_scalars(
        source.get("quality_flags"), UNTRUSTED_QUALITY_FLAG_ALLOWLIST
    )
    quality_flags.update(
        {
            "role_attribution": attribution,
            "role_attribution_version": attribution.get("version"),
            "role_attribution_decision": attribution.get("decision"),
            "role_attribution_reason_codes": reason_codes,
            "role_attribution_untrusted": True,
            "dialogue_source": dialogue.source,
            "dialogue_canonical_sha256": dialogue.canonical_sha256,
            "dialogue_turn_count": len(dialogue.turns),
            "needs_review": True,
            "review_reasons": review_reasons,
        }
    )
    schema_version = source.get("analysis_schema_version")
    analysis_meta = _allowed_scalars(
        source.get("analysis_meta"), UNTRUSTED_META_ALLOWLIST
    )
    token_usage = _safe_token_usage(
        (source.get("analysis_meta") or {}).get("token_usage")
        if isinstance(source.get("analysis_meta"), Mapping)
        else None
    )
    if token_usage:
        analysis_meta["token_usage"] = token_usage
    return {
        "analysis_schema_version": (
            schema_version if isinstance(schema_version, str) else ""
        ),
        "untrusted_projection_version": UNTRUSTED_PROJECTION_VERSION,
        "neutral_topic_version": NEUTRAL_TOPIC_VERSION,
        "neutral_topics": topics,
        "summary": UNTRUSTED_SUMMARY,
        "manager_brief": UNTRUSTED_SUMMARY,
        "history_summary": UNTRUSTED_SUMMARY,
        "history_short": UNTRUSTED_SUMMARY,
        "follow_up_reason": UNTRUSTED_FOLLOW_UP_REASON,
        "role_attribution": attribution,
        "dialogue_input": dialogue_input,
        "needs_review": True,
        "review_reasons": review_reasons,
        "review_reasons_ru": review_reasons_ru(reason_codes),
        "structured_fields": {},
        "display_fields": {},
        "crm_blocks": {},
        "evidence": [],
        # No proven side means no proven business fact, so there is nothing to
        # evidence and nothing to normalize.  The keys exist so that a reader
        # never has to tell "no claims" apart from "an older payload".
        "claim_evidence": [],
        "normalized_facts": [],
        "tags": [],
        "objections": [],
        "quality_flags": quality_flags,
        "analysis_meta": analysis_meta,
    }


def project_invalid_stored_analysis(
    analysis: Any, dialogue: DialogueInput
) -> dict[str, Any]:
    """Safe row for a trusted call whose saved analysis no longer proves facts."""
    safe = project_untrusted_analysis(analysis, dialogue)
    attribution = dict(dialogue.role_attribution)
    role_reasons = [str(code) for code in attribution.get("reason_codes") or []]
    reasons = list(role_reasons)
    if not dialogue.trusted:
        reasons.append("role_attribution_untrusted")
    reasons.append("analysis_contract_invalid")
    reasons = list(dict.fromkeys(reasons))
    safe["role_attribution"] = attribution
    safe["dialogue_input"] = {
        "version": dialogue.version,
        "source": dialogue.source,
        "canonical_sha256": dialogue.canonical_sha256,
        "turn_count": len(dialogue.turns),
    }
    safe["review_reasons"] = reasons
    safe["review_reasons_ru"] = review_reasons_ru(reasons)
    safe["summary"] = INVALID_STORED_SUMMARY
    safe["manager_brief"] = INVALID_STORED_SUMMARY
    safe["history_summary"] = INVALID_STORED_SUMMARY
    safe["history_short"] = INVALID_STORED_SUMMARY
    safe["follow_up_reason"] = ANALYSIS_REASON_RU["analysis_contract_invalid"]
    flags = safe["quality_flags"]
    flags.update(
        {
            "role_attribution": attribution,
            "role_attribution_version": attribution.get("version"),
            "role_attribution_decision": attribution.get("decision"),
            "role_attribution_reason_codes": role_reasons,
            "role_attribution_untrusted": not dialogue.trusted,
            "needs_review": True,
            "review_reasons": reasons,
            "analysis_contract_invalid": True,
        }
    )
    return safe


def unreadable_dialogue(reason_code: str = "dialogue_unreadable") -> DialogueInput:
    """Fail-closed stand-in for a call whose stored dialogue cannot be parsed.

    A reader that cannot even rebuild the conversation knows strictly less than
    one that can, so it may not fall back to “no dialogue, therefore nothing to
    check”: it gets an empty, untrusted dialogue and the analysis is rebuilt
    from the allowlist like any other unproven call.
    """
    if reason_code not in ROLE_REASON_CODES:
        raise DialogueContractError(f"unknown role reason code: {reason_code}")
    return DialogueInput(
        version=CONTRACT_VERSION,
        source=SOURCE_DIALOGUE_LINES,
        role_attribution={
            "version": ROLE_GUARD_VERSION,
            "decision": "untrusted",
            "trusted": False,
            "topology": "",
            "reason_codes": [reason_code],
            "source": SOURCE_DIALOGUE_LINES,
        },
        turns=(),
        warnings=(reason_code,),
        canonical_sha256=hashlib.sha256(b"[]").hexdigest(),
    )


def guard_stored_analysis(record: Mapping[str, Any], analysis: Any) -> dict[str, Any]:
    """The one entry point every reader of a stored ``analysis_json`` uses.

    Excel, AI Office and the Google publisher all read payloads written before
    the role guard existed.  Letting each of them decide what is safe is how
    a cleaned field comes back to life in exactly one export, so they all call
    this instead: same dialogue rebuild, same allowlist, same result.
    """
    try:
        dialogue = build_dialogue_input(record)
    except DialogueContractError:
        dialogue = unreadable_dialogue()
    if not dialogue.trusted:
        return project_untrusted_analysis(analysis, dialogue)
    try:
        validated = validate_stored_analysis(record, analysis, dialogue)
    except StoredAnalysisContractError:
        return project_invalid_stored_analysis(analysis, dialogue)
    return apply_role_guard(validated, dialogue)


def apply_role_guard(analysis: Any, dialogue: DialogueInput) -> dict[str, Any]:
    """Trusted analysis keeps its content; untrusted is rebuilt from allowlist."""
    if not dialogue.trusted:
        return project_untrusted_analysis(analysis, dialogue)
    guarded = dict(analysis) if isinstance(analysis, Mapping) else {}
    attribution = dict(dialogue.role_attribution)
    guarded["role_attribution"] = attribution
    guarded["dialogue_input"] = {
        "version": dialogue.version,
        "source": dialogue.source,
        "canonical_sha256": dialogue.canonical_sha256,
        "turn_count": len(dialogue.turns),
    }
    quality_flags = dict(guarded.get("quality_flags") or {})
    quality_flags.update(
        {
            "role_attribution": attribution,
            "role_attribution_version": attribution.get("version"),
            "role_attribution_decision": attribution.get("decision"),
            "role_attribution_reason_codes": list(attribution.get("reason_codes") or []),
            "role_attribution_untrusted": False,
            "dialogue_source": dialogue.source,
            "dialogue_canonical_sha256": dialogue.canonical_sha256,
            "dialogue_turn_count": len(dialogue.turns),
        }
    )
    guarded["quality_flags"] = quality_flags
    meta = guarded.get("analysis_meta")
    if isinstance(meta, Mapping) and "manager_output_sha256" in meta:
        guarded["analysis_meta"] = dict(meta)
        guarded["analysis_meta"]["manager_output_sha256"] = manager_output_sha256(
            guarded
        )
    return guarded
