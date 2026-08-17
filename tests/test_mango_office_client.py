"""Offline contract of the Mango ``recording_transcripts`` reader (этап B/K).

Nothing here touches the network: the method under test only builds a request
body and strictly parses a response the caller already has.

The only shape accepted is the documented envelope — ``result`` plus ``data``,
an **array** of records, each carrying ``recording_id``, the ``names`` object
``{"client": ..., "operator": ...}`` and the chronological ``phrases`` of
``role``/``text``.  Our earlier invented shape (``names`` as a list of speaker
records with a ``channel``, phrases keyed by ``name`` and carrying ``start``) is
rejected: it could prove a per-side binding the real API never sent.

Until a golden response is captured on M1 the whole path stays fail-closed —
these tests pin the contract, not the availability of the method.
"""
from __future__ import annotations

import hashlib
import json

import pytest

from mango_mvp.productization.mango_office_client import (
    MangoOfficeApiError,
    MangoOfficeClient,
    MangoOfficeCredentials,
    RECORDING_TRANSCRIPTS_MAX_IDS,
    RECORDING_TRANSCRIPTS_PATH,
)
from mango_mvp.services import dialogue_contract as contract
from tests import mango_provider_fixture as fx


RECORDING_ID = fx.RECORDING_ID
SOURCE_CALL_ID = fx.SOURCE_CALL_ID
OTHER_RECORDING_ID = "rec-8"
# A three-turn call: the interleaving of the tracks is what binds the roles.
TURNS = (
    ("operator", "left", "Добрый день"),
    ("client", "right", "Здравствуйте"),
    ("operator", "left", "Слушаю вас"),
)


class _Response:
    def __init__(self, payload, *, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self.payload


class _Session:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, data, timeout):
        self.calls.append((url, data, timeout))
        return self.response


def parse(payload, *, source_call_id=SOURCE_CALL_ID, raw_body=None, expected=RECORDING_ID):
    """By default the raw body really is the body the payload was decoded from."""
    if raw_body is None:
        raw_body = json.dumps(payload, ensure_ascii=False)
    return MangoOfficeClient.parse_recording_transcripts_response(
        payload,
        source_call_id=source_call_id,
        raw_body=raw_body,
        expected_recording_id=expected,
    )


def stored_call(evidence, *, source_call_id=SOURCE_CALL_ID, turns=TURNS, lines=None):
    variants = {
        "mode": "stereo",
        "role_mapping": dict(fx.PROVEN_ROLE_MAPPING),
        contract.PROVIDER_EVIDENCE_FIELD: evidence,
        "dialogue_lines": list(fx.dialogue_lines(turns) if lines is None else lines),
    }
    return {
        "source_call_id": source_call_id,
        "source_recording_id": RECORDING_ID,
        "transcript_variants_json": variants,
    }


# --------------------------------------------------------------------------
# Request
# --------------------------------------------------------------------------


def test_request_body_asks_by_recording_id_and_names_the_official_path():
    assert RECORDING_TRANSCRIPTS_PATH == "/vpbx/queries/recording_transcripts"
    assert MangoOfficeClient.build_recording_transcripts_payload(["r1", " r2 "]) == {
        "recording_id": '["r1","r2"]'
    }


@pytest.mark.parametrize("ids", [[], [""], ["r1", "r1"], ["r1", "   "]])
def test_request_body_rejects_empty_or_duplicated_ids(ids):
    with pytest.raises(MangoOfficeApiError):
        MangoOfficeClient.build_recording_transcripts_payload(ids)


@pytest.mark.parametrize("ids", ["r1", None, ["r1", None], ["r1", 2]])
def test_request_body_rejects_non_sequence_and_non_string_ids(ids):
    with pytest.raises(MangoOfficeApiError):
        MangoOfficeClient.build_recording_transcripts_payload(ids)


def test_request_body_respects_the_documented_batch_ceiling():
    ids = [f"r{index}" for index in range(RECORDING_TRANSCRIPTS_MAX_IDS + 1)]

    assert json.loads(
        MangoOfficeClient.build_recording_transcripts_payload(ids[:-1])["recording_id"]
    ) == ids[:-1]
    with pytest.raises(MangoOfficeApiError, match="at most"):
        MangoOfficeClient.build_recording_transcripts_payload(ids)


def test_transport_fetches_one_signed_batch_and_preserves_exact_raw_body():
    payload = fx.envelope(fx.record(TURNS))
    response = _Response(payload)
    session = _Session(response)
    client = MangoOfficeClient(
        credentials=MangoOfficeCredentials(api_key="key", api_salt="salt"),
        base_url="https://mango.example",
        session=session,
        timeout_sec=17,
    )

    decoded, raw_body = client.fetch_recording_transcripts([RECORDING_ID])

    assert decoded == payload
    assert raw_body == response.text
    assert len(session.calls) == 1
    url, form, timeout = session.calls[0]
    assert url == f"https://mango.example{RECORDING_TRANSCRIPTS_PATH}"
    assert timeout == 17
    assert json.loads(form["json"]) == {
        "recording_id": json.dumps([RECORDING_ID], separators=(",", ":"))
    }
    assert set(form) == {"vpbx_api_key", "sign", "json"}


def test_http_error_reports_only_status_path_and_body_digest():
    secret = "Клиент +79990000000 сказал секрет"
    response = _Response({"detail": secret}, status_code=500)
    client = MangoOfficeClient(
        credentials=MangoOfficeCredentials(api_key="key", api_salt="salt"),
        base_url="https://mango.example",
        session=_Session(response),
    )

    with pytest.raises(MangoOfficeApiError) as error:
        client.fetch_recording_transcripts([RECORDING_ID])

    message = str(error.value)
    assert secret not in message
    assert "status=500" in message
    assert "body_sha256=" in message


# --------------------------------------------------------------------------
# Response: the documented one-record and batch envelopes
# --------------------------------------------------------------------------


def test_parser_binds_the_official_envelope_to_the_call_and_the_recording():
    payload = fx.envelope(fx.record(TURNS))
    raw_body = json.dumps(payload, ensure_ascii=False)

    evidence = parse(payload, raw_body=raw_body)

    assert evidence["provider"] == contract.PROVIDER_EVIDENCE_SOURCE
    assert evidence["source_call_id"] == SOURCE_CALL_ID
    # The recording id always comes from the body, never from our request.
    assert evidence["recording_id"] == RECORDING_ID
    # One call row contains only its own record, never the other 499 batch
    # transcripts.  The exact batch remains referenced by its digest.
    assert evidence["raw_response"] != raw_body
    assert json.loads(evidence["raw_response"])["data"]["recording_id"] == RECORDING_ID
    assert evidence["raw_response_sha256"] == hashlib.sha256(
        evidence["raw_response"].encode("utf-8")
    ).hexdigest()
    assert evidence["batch_response_sha256"] == hashlib.sha256(
        raw_body.encode("utf-8")
    ).hexdigest()
    assert evidence["phrases_sha256"] == contract.canonical_provider_phrases_sha256(
        [{"role": role, "text": text} for role, _side, text in TURNS]
    )
    # No second, independently forgeable copy of the side binding is stored.
    assert "channels" not in evidence
    assert "phrases" not in evidence


def test_one_answer_may_describe_many_recordings_and_the_right_one_is_taken():
    """A request carries up to 500 ids, so a batch answer is the normal case."""
    other = fx.record(
        (("operator", "left", "Другой звонок"), ("client", "right", "Ага")),
        recording_id=OTHER_RECORDING_ID,
    )
    payload = fx.envelope(other, fx.record(TURNS))

    mine = parse(payload, expected=RECORDING_ID)
    theirs = parse(payload, expected=OTHER_RECORDING_ID)

    assert mine["recording_id"] == RECORDING_ID
    assert theirs["recording_id"] == OTHER_RECORDING_ID
    assert mine["phrases_sha256"] != theirs["phrases_sha256"]
    assert "Другой звонок" not in mine["raw_response"]
    assert "Добрый день" not in theirs["raw_response"]


def test_parser_never_guesses_which_record_was_meant():
    payload = fx.envelope(fx.record(TURNS))

    with pytest.raises(MangoOfficeApiError, match="is invalid"):
        parse(payload, expected="rec-999")
    for missing in ("", "   ", None):
        with pytest.raises(MangoOfficeApiError):
            parse(payload, expected=missing)


@pytest.mark.parametrize("raw_body", ["{}", '{"result": 1000}', "не json", ""])
def test_parser_refuses_a_body_that_does_not_carry_the_parsed_envelope(raw_body):
    with pytest.raises(MangoOfficeApiError, match="is invalid"):
        parse(fx.envelope(fx.record(TURNS)), raw_body=raw_body)


def test_parser_hash_is_deterministic_and_content_sensitive():
    changed = (*TURNS[:2], ("operator", "left", "Совсем другое"))

    first = parse(fx.envelope(fx.record(TURNS)))
    second = parse(fx.envelope(fx.record(TURNS)))
    other = parse(fx.envelope(fx.record(changed)))

    assert first == second
    assert first["phrases_sha256"] != other["phrases_sha256"]


@pytest.mark.parametrize(
    "payload",
    [
        "текст",
        [],
        {},
        {"result": 1000},
        # The old, unbindable shape: a bare list of phrases and no recording id.
        {"phrases": [{"role": "operator", "text": "a"}]},
        {"result": "1000", "data": [fx.record()]},
        {"result": True, "data": [fx.record()]},
        {"result": 200, "data": [fx.record()]},
        {"result": 1000, "data": []},
        {"result": 1000, "data": [fx.record(), fx.record()]},
        {"result": 1000, "data": [{**fx.record(), "recording_id": ""}]},
        {"result": 1000, "data": ["не запись"]},
    ],
)
def test_parser_rejects_everything_but_the_official_envelope(payload):
    with pytest.raises(MangoOfficeApiError):
        parse(payload)


def test_parser_accepts_public_example_shape_with_object_data_and_phrase_pairs():
    payload = {
        "result": 1000,
        "data": {
            "recording_id": RECORDING_ID,
            "names": {"client": "Клиент", "operator": "Сотрудник"},
            "phrases": [
                ["operator", "Добрый день"],
                ["client", "Здравствуйте"],
            ],
        },
    }

    parsed = parse(payload)

    assert parsed["recording_id"] == RECORDING_ID
    assert parsed["phrases_sha256"] == contract.canonical_provider_phrases_sha256(
        payload["data"]["phrases"]
    )


@pytest.mark.parametrize(
    "names",
    [
        # The invented shape: a list of speaker records carrying a channel.
        [
            {"name": "Оператор", "role": "operator", "channel": "left"},
            {"name": "Клиент", "role": "client", "channel": "right"},
        ],
        {},
        {"operator": ""},
        {"operator": "Иванов", "client": None},
        "Оператор и Клиент",
    ],
)
def test_parser_rejects_a_names_declaration_that_is_not_the_official_object(names):
    payload = {"result": 1000, "data": [{**fx.record(TURNS), "names": names}]}

    with pytest.raises(MangoOfficeApiError):
        parse(payload)


@pytest.mark.parametrize(
    "phrases",
    [
        [],
        "нет",
        [{"role": "operator"}],
        [{"text": "Добрый день"}],
        [{"role": "manager", "text": "Добрый день"}],
        [{"role": "operator", "text": "   "}],
        ["Добрый день"],
        [["operator"]],
        [["operator", "текст", "лишнее"]],
    ],
)
def test_parser_rejects_phrases_that_are_not_role_plus_text(phrases):
    payload = {"result": 1000, "data": [{**fx.record(TURNS), "phrases": phrases}]}

    with pytest.raises(MangoOfficeApiError):
        parse(payload)


@pytest.mark.parametrize("source_call_id", ["", "   ", None])
def test_parser_refuses_evidence_without_a_call_binding(source_call_id):
    with pytest.raises(MangoOfficeApiError):
        parse(fx.envelope(fx.record(TURNS)), source_call_id=source_call_id)


# --------------------------------------------------------------------------
# What the evidence is allowed to unlock
# --------------------------------------------------------------------------


def test_unique_text_alignment_unlocks_the_named_roles_for_this_call_only():
    evidence = parse(fx.envelope(fx.record(TURNS)))

    mine = contract.build_dialogue_input(stored_call(evidence))
    other = contract.build_dialogue_input(
        stored_call(evidence, source_call_id="call-8")
    )

    assert mine.role_attribution["trusted"] is True
    assert mine.render().splitlines() == [
        "[00:01.0] Менеджер: Добрый день",
        "[00:03.0] Клиент: Здравствуйте",
        "[00:05.0] Менеджер: Слушаю вас",
    ]
    assert other.role_attribution["reason_codes"] == ["provider_evidence_call_mismatch"]


def test_two_tracks_carrying_the_same_words_can_never_be_told_apart():
    """Bleed-through / echo: the words prove nothing about who is who."""
    same = (
        ("operator", "left", "Алло"),
        ("client", "right", "Алло"),
    )
    evidence = parse(fx.envelope(fx.record(same)))

    dialogue = contract.build_dialogue_input(stored_call(evidence, turns=same))

    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_ambiguous_sides"
    ]
    assert "Менеджер" not in dialogue.render()


def test_an_internal_call_never_reaches_manager_quality():
    """Mango warns that on an internal call both sides can be employees."""
    payload = {
        "result": 1000,
        "data": [
            {
                **fx.record(TURNS),
                "names": {"operator": "Иванов", "client": "Иванов"},
            }
        ],
    }
    evidence = parse(payload)

    dialogue = contract.build_dialogue_input(stored_call(evidence))

    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_internal_call"
    ]
    assert "Менеджер" not in dialogue.render()


def test_two_different_employee_names_are_still_an_internal_call():
    payload = {
        "result": 1000,
        "data": {
            **fx.record(TURNS),
            "names": {"operator": "Иванов", "client": "Петров"},
        },
    }
    evidence = parse(payload)

    dialogue = contract.build_dialogue_input(stored_call(evidence))

    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_internal_call"
    ]
    assert "Менеджер" not in dialogue.render()


def test_a_one_sided_recording_has_no_binding_to_derive():
    one_sided = (
        ("operator", "left", "Добрый день"),
        ("operator", "left", "Вы меня слышите?"),
    )
    evidence = parse(fx.envelope(fx.record(one_sided)))

    dialogue = contract.build_dialogue_input(stored_call(evidence, turns=one_sided))

    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_no_channel_binding"
    ]


def test_a_swapped_raw_response_after_capture_takes_the_trust_away():
    evidence = parse(fx.envelope(fx.record(TURNS)))
    swapped_turns = tuple(
        ("client" if role == "operator" else "operator", side, text)
        for role, side, text in TURNS
    )
    swapped = json.dumps(fx.envelope(fx.record(swapped_turns)), ensure_ascii=False)
    tampered = {
        **evidence,
        "raw_response": swapped,
        "raw_response_sha256": hashlib.sha256(swapped.encode("utf-8")).hexdigest(),
    }

    dialogue = contract.build_dialogue_input(stored_call(tampered))

    # The re-derived phrases no longer hash to the stored digest.
    assert dialogue.role_attribution["reason_codes"] == ["provider_evidence_invalid"]
    assert "Менеджер" not in dialogue.render()


def test_evidence_rebound_to_another_recording_id_is_refused():
    evidence = parse(fx.envelope(fx.record(TURNS)))
    rebound = {**evidence, "recording_id": OTHER_RECORDING_ID}

    dialogue = contract.build_dialogue_input(stored_call(rebound))

    assert dialogue.role_attribution["reason_codes"] == [
        "provider_recording_binding_mismatch"
    ]


def test_internally_valid_answer_for_another_recording_is_not_rebound_to_the_call():
    other_body = json.dumps(
        fx.envelope(fx.record(TURNS, recording_id=OTHER_RECORDING_ID)),
        ensure_ascii=False,
    )
    evidence = parse(
        json.loads(other_body), raw_body=other_body, expected=OTHER_RECORDING_ID
    )

    dialogue = contract.build_dialogue_input(stored_call(evidence))

    assert dialogue.role_attribution["reason_codes"] == [
        "provider_recording_binding_mismatch"
    ]


@pytest.mark.parametrize(
    "lines",
    [
        ["[00:01.0] Дорожка левая: Совсем другой разговор",
         "[00:03.0] Дорожка правая: Здравствуйте",
         "[00:05.0] Дорожка левая: Слушаю вас"],
        ["[00:01.0] Дорожка правая: Добрый день",
         "[00:03.0] Дорожка левая: Здравствуйте",
         "[00:05.0] Дорожка правая: Слушаю вас"],
        ["[00:01.0] Дорожка левая: Добрый день",
         "[00:03.0] Дорожка правая: Здравствуйте"],
    ],
)
def test_evidence_that_does_not_describe_this_dialogue_is_refused(lines):
    evidence = parse(fx.envelope(fx.record(TURNS)))

    dialogue = contract.build_dialogue_input(stored_call(evidence, lines=lines))

    assert "provider_evidence_dialogue_mismatch" in dialogue.role_attribution[
        "reason_codes"
    ]
    assert "Менеджер" not in dialogue.render()


def test_the_producer_alone_never_makes_a_call_trusted():
    """Without provider evidence a perfect stored mapping proves nothing."""
    variants = {
        "mode": "stereo",
        "role_mapping": dict(fx.PROVEN_ROLE_MAPPING),
        "dialogue_lines": fx.dialogue_lines(TURNS),
    }
    dialogue = contract.build_dialogue_input(
        {"source_call_id": SOURCE_CALL_ID, "transcript_variants_json": variants}
    )

    assert dialogue.role_attribution["reason_codes"] == ["provider_evidence_missing"]
    assert "Менеджер" not in dialogue.render()
