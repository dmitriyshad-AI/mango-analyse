"""One builder of official Mango ``recording_transcripts`` fixtures.

Not a test module (pytest collects ``test_*.py``): it exists so that the five
suites which need proven provider evidence stop hand-writing the envelope.  Five
copies of a hand-written envelope is exactly how an invented shape survives — it
gets fixed in one file and keeps proving roles in the other four.

The shape here is the documented one and nothing else:

* the request carries a serialized array of ``recording_id`` (up to 500);
* the answer is ``result`` plus ``data``: one record object or a batch array;
* a record carries ``recording_id``, ``names`` as the object
  ``{"client": ..., "operator": ...}`` and ``phrases`` as chronological
  ``[role, text]`` pairs.

There is deliberately no per-phrase ``channel`` and no ``start``: the official
answer never promised either, so a fixture may not supply them.  Which physical
track each role sat on is *derived* from the words, by the contract itself.
"""
from __future__ import annotations

import hashlib
import json

from mango_mvp.services.dialogue_contract import (
    PROVIDER_EVIDENCE_FIELD,
    PROVIDER_EVIDENCE_SOURCE,
    canonical_provider_phrases_sha256,
)


SOURCE_CALL_ID = "call-7"
RECORDING_ID = "rec-7"
OPERATOR_NAME = "Оператор Иванов"
CLIENT_NAME = "+79000000000"
# (provider role, physical track, text) — the single source of every artefact
# below, so the stored dialogue and the provider answer can never drift apart.
DEFAULT_TURNS = (
    ("operator", "left", "Добрый день"),
    ("client", "right", "Здравствуйте"),
)
SIDE_LABEL = {"left": "Дорожка левая", "right": "Дорожка правая"}
PROVEN_ROLE_MAPPING = {
    "status": "confirmed_multi_signal",
    "confirmed": True,
    "manager_quality_allowed": True,
    "topology": "simple_two_party",
    "left": "manager",
    "right": "client",
}


def dialogue_lines(turns=DEFAULT_TURNS) -> list[str]:
    """Stored dialogue exactly as the producer writes it: physical tracks only.

    Timecodes are 1, 3, 5, … so that a gap between two replies is visible and a
    test can insert a line between them without renumbering everything.
    """
    return [
        f"[00:{2 * index - 1:02d}.0] {SIDE_LABEL[side]}: {text}"
        for index, (_role, side, text) in enumerate(turns, start=1)
    ]


def names(operator: str = OPERATOR_NAME, client: str = CLIENT_NAME) -> dict[str, str]:
    return {"operator": operator, "client": client}


def record(turns=DEFAULT_TURNS, *, recording_id: str = RECORDING_ID, **name_overrides):
    return {
        "recording_id": recording_id,
        "names": names(**name_overrides),
        "phrases": [[role, text] for role, _side, text in turns],
    }


def envelope(*records, result: int = 1000) -> dict:
    """The documented batch answer; one request may return many recordings."""
    entries = list(records) or [record()]
    return {"result": result, "data": entries[0] if len(entries) == 1 else entries}


def raw_sha(body) -> str:
    return hashlib.sha256(str(body).encode("utf-8")).hexdigest()


def evidence_for_recording(
    turns=DEFAULT_TURNS,
    *,
    source_call_id: str,
    recording_id: str,
):
    """Valid evidence for one explicit call/recording identity."""
    item = record(turns, recording_id=recording_id)
    body = json.dumps(envelope(item), ensure_ascii=False)
    record_body = json.dumps(
        {"result": 1000, "data": item},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "provider": PROVIDER_EVIDENCE_SOURCE,
        "source_call_id": source_call_id,
        "recording_id": recording_id,
        "phrases_sha256": canonical_provider_phrases_sha256(
            [{"role": role, "text": text} for role, _side, text in turns]
        ),
        "raw_response": record_body,
        "raw_response_sha256": raw_sha(record_body),
        "batch_response_sha256": raw_sha(body),
    }


def evidence(turns=DEFAULT_TURNS, *, source_call_id: str = SOURCE_CALL_ID, **overrides):
    """Valid evidence first; a patch then breaks exactly one field.

    Nothing is re-derived from the patch on purpose: a broken field has to reach
    the production guard, not blow up while the fixture is still assembling.
    """
    payload = evidence_for_recording(
        turns, source_call_id=source_call_id, recording_id=RECORDING_ID
    )
    payload.update(overrides)
    return payload


def proven_variants(turns=DEFAULT_TURNS, **mapping) -> dict:
    """A stereo call whose sides Mango really proved, ready to be stored."""
    role_mapping = dict(PROVEN_ROLE_MAPPING)
    role_mapping.update(mapping)
    return {
        "mode": "stereo",
        "role_mapping": role_mapping,
        "dialogue_lines": dialogue_lines(turns),
        PROVIDER_EVIDENCE_FIELD: evidence(turns),
    }


def proven_call(turns=DEFAULT_TURNS, *, call_id: int = 7, **overrides) -> dict:
    payload = {
        "id": call_id,
        "source_call_id": SOURCE_CALL_ID,
        "source_recording_id": RECORDING_ID,
        "transcript_variants_json": json.dumps(
            proven_variants(turns), ensure_ascii=False
        ),
        "transcript_text": "",
    }
    payload.update(overrides)
    return payload
