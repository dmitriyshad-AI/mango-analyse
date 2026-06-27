from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import mango_mvp.channels.subscription_llm as subscription_llm
from mango_mvp.channels.dialogue_contract_pipeline import (
    AnswerContract,
    verify_output as verify_dialogue_contract_output,
)
from mango_mvp.channels.fact_venue_scope import FACT_VENUE_SCOPE_ENV, venue_scope_enabled
from mango_mvp.channels.subscription_llm import LLM_RETRIEVE_ENV
from mango_mvp.channels.subscription_llm_parts.direct_path import (
    _direct_path_context_fact_pack,
    _direct_path_fact_conflicts_slots,
    _direct_path_render_fact_block,
    build_direct_path_llm_retriever_prompt,
)
from mango_mvp.channels.subscription_llm_parts.support import (
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS,
)


R4_1_SNAPSHOT = Path(
    "product_data/knowledge_base/kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json"
)


def _write_snapshot(tmp_path: Path, facts: list[Mapping[str, object]]) -> Path:
    path = tmp_path / "venue_scope_snapshot.json"
    path.write_text(json.dumps({"facts": facts}, ensure_ascii=False), encoding="utf-8")
    return path


def _fact(
    key: str,
    text: str,
    *,
    venue: str = "any",
    program_kind: str = "regular",
    brand: str = "unpk",
    fact_type: str = "location",
) -> dict[str, object]:
    return {
        "brand": brand,
        "fact_key": key,
        "fact_type": fact_type,
        "product": program_kind,
        "venue": venue,
        "program_kind": program_kind,
        "allowed_for_client_answer": True,
        "forbidden_for_client": False,
        "internal_only": False,
        "client_safe_text": text,
    }


def _venue_context(snapshot_path: Path) -> dict[str, object]:
    return {
        "active_brand": "unpk",
        "snapshot_path": str(snapshot_path),
        LLM_RETRIEVE_ENV: "1",
        FACT_VENUE_SCOPE_ENV: "1",
        "conversation_intent_plan": {"primary_intent": "address", "answer_topics": ["address"]},
    }


def _pack_text(pack: Mapping[str, object]) -> str:
    facts = pack.get("facts") if isinstance(pack.get("facts"), Mapping) else {}
    meta = pack.get("fact_metadata") if isinstance(pack.get("fact_metadata"), Mapping) else {}
    return _direct_path_render_fact_block(facts, fact_metadata=meta, keys=tuple(str(key) for key in facts))


def test_fact_venue_scope_default_off_and_not_in_pilot_profile(monkeypatch) -> None:
    monkeypatch.delenv(FACT_VENUE_SCOPE_ENV, raising=False)

    assert venue_scope_enabled({}) is False
    assert venue_scope_enabled({DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION}) is False
    assert venue_scope_enabled({FACT_VENUE_SCOPE_ENV: "1"}) is True
    assert venue_scope_enabled({FACT_VENUE_SCOPE_ENV: "0"}) is False
    assert FACT_VENUE_SCOPE_ENV not in DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS
    assert subscription_llm.FACT_VENUE_SCOPE_ENV == FACT_VENUE_SCOPE_ENV


def test_fact_venue_scope_prompt_requests_scope_only_when_enabled() -> None:
    candidate = _fact(
        "unpk.moscow.address",
        "УНПК МФТИ: московские занятия проходят на Сретенке.",
        venue="moscow_regular",
    )

    off_prompt = build_direct_path_llm_retriever_prompt(
        "Где очные занятия?",
        context={},
        candidates=[candidate],
    )
    on_prompt = build_direct_path_llm_retriever_prompt(
        "Где очные занятия?",
        context={FACT_VENUE_SCOPE_ENV: "1"},
        candidates=[candidate],
    )

    assert "requested_scope" not in off_prompt
    assert "venue=moscow_regular" not in off_prompt
    assert "requested_scope" in on_prompt
    assert "venue=moscow_regular" in on_prompt
    assert "не выводи requested_scope из списка фактов" in on_prompt


def test_fact_venue_scope_off_keeps_llm_pack_metadata_unchanged(tmp_path: Path) -> None:
    snapshot = _write_snapshot(
        tmp_path,
        [
            _fact("unpk.moscow.address", "УНПК МФТИ: московские занятия проходят на Сретенке.", venue="moscow_regular"),
            _fact("unpk.lvsh.address", "УНПК МФТИ: ЛВШ проходит в Менделеево.", venue="lvsh_mendeleevo"),
        ],
    )

    pack = _direct_path_context_fact_pack(
        {
            "active_brand": "unpk",
            "snapshot_path": str(snapshot),
            LLM_RETRIEVE_ENV: "1",
        },
        client_message="Где очные занятия в Москве?",
        retriever_fn=lambda _prompt: {
            "requested_scope": "moscow_regular",
            "exact_ids": ["unpk.lvsh.address", "unpk.moscow.address"],
            "adjacent_ids": [],
        },
    )

    assert pack["exact_keys"] == ["unpk.lvsh.address", "unpk.moscow.address"]
    assert "venue_scope" not in pack["llm_retrieve"]
    assert all("venue" not in meta for meta in pack["fact_metadata"].values())


def test_fact_venue_scope_removes_foreign_exact_when_target_fact_present(tmp_path: Path) -> None:
    snapshot = _write_snapshot(
        tmp_path,
        [
            _fact("unpk.moscow.address", "УНПК МФТИ: московские занятия проходят на Сретенке.", venue="moscow_regular"),
            _fact("unpk.lvsh.address", "УНПК МФТИ: ЛВШ проходит в Менделеево.", venue="lvsh_mendeleevo", program_kind="camp_lvsh"),
            _fact("unpk.contacts", "УНПК МФТИ: общий контакт менеджера есть в карточке.", venue="any"),
        ],
    )

    pack = _direct_path_context_fact_pack(
        _venue_context(snapshot),
        client_message="Где очные занятия в Москве?",
        retriever_fn=lambda _prompt: {
            "requested_scope": "moscow_regular",
            "exact_ids": ["unpk.lvsh.address", "unpk.moscow.address"],
            "adjacent_ids": ["unpk.contacts"],
        },
    )

    text = _pack_text(pack)
    assert "unpk.moscow.address" in pack["exact_keys"]
    assert "unpk.lvsh.address" not in pack["facts"]
    assert "Сретенке" in text
    assert "Менделеево" not in text
    assert pack["llm_retrieve"]["venue_scope"]["target_venue_fact_present"] is True
    assert pack["llm_retrieve"]["venue_scope"]["venue_scope_removed_ids"] == ["unpk.lvsh.address"]


def test_fact_venue_scope_demotes_foreign_exact_when_no_target_fact_present(tmp_path: Path) -> None:
    snapshot = _write_snapshot(
        tmp_path,
        [
            _fact("unpk.lvsh.address", "УНПК МФТИ: ЛВШ проходит в Менделеево.", venue="lvsh_mendeleevo", program_kind="camp_lvsh"),
            _fact("unpk.contacts", "УНПК МФТИ: общий контакт менеджера есть в карточке.", venue="any"),
        ],
    )

    pack = _direct_path_context_fact_pack(
        _venue_context(snapshot),
        client_message="Где очные занятия в Москве?",
        retriever_fn=lambda _prompt: {
            "requested_scope": "moscow_regular",
            "exact_ids": ["unpk.lvsh.address"],
            "adjacent_ids": ["unpk.contacts"],
        },
    )

    text = _pack_text(pack)
    assert pack["exact_keys"] == []
    assert "unpk.lvsh.address" in pack["adjacent_keys"]
    assert "[площадка: ЛВШ Менделеево]" in text
    assert pack["llm_retrieve"]["scope_demoted_ids"] == ["unpk.lvsh.address"]
    assert pack["llm_retrieve"]["venue_scope"]["venue_scope_demoted_ids"] == ["unpk.lvsh.address"]


def test_fact_venue_scope_unspecified_does_not_narrow_model_selection(tmp_path: Path) -> None:
    snapshot = _write_snapshot(
        tmp_path,
        [
            _fact("unpk.moscow.address", "УНПК МФТИ: московские занятия проходят на Сретенке.", venue="moscow_regular"),
            _fact("unpk.lvsh.address", "УНПК МФТИ: ЛВШ проходит в Менделеево.", venue="lvsh_mendeleevo", program_kind="camp_lvsh"),
        ],
    )

    pack = _direct_path_context_fact_pack(
        _venue_context(snapshot),
        client_message="Где проходят занятия?",
        retriever_fn=lambda _prompt: {
            "requested_scope": "unspecified",
            "exact_ids": ["unpk.lvsh.address", "unpk.moscow.address"],
            "adjacent_ids": [],
        },
    )

    assert pack["exact_keys"] == ["unpk.lvsh.address", "unpk.moscow.address"]
    assert pack["llm_retrieve"]["venue_scope"]["venue_scope_removed_ids"] == []
    assert pack["llm_retrieve"]["venue_scope"]["venue_scope_demoted_ids"] == []


def test_fact_venue_scope_program_kind_replaces_camp_text_regex() -> None:
    camp_fact = _fact(
        "unpk.lvsh.program",
        "УНПК МФТИ: выездная программа проходит летом.",
        venue="lvsh_mendeleevo",
        program_kind="camp_lvsh",
    )
    regular_fact_with_legacy_word = _fact(
        "unpk.regular.legacy_note",
        "УНПК МФТИ: регулярный курс, в архивной заметке упомянута ЛВШ.",
        venue="moscow_regular",
        program_kind="regular",
    )

    assert _direct_path_fact_conflicts_slots(
        camp_fact,
        {"product": "regular_course"},
        use_structured_program_kind=True,
    )
    assert not _direct_path_fact_conflicts_slots(
        regular_fact_with_legacy_word,
        {"product": "regular_course"},
        use_structured_program_kind=True,
    )


def test_venue_scope_verifier_uses_structural_venue_metadata() -> None:
    question = "Где проходят московские очные занятия?"
    facts = {"unpk.lvsh.address": "УНПК МФТИ: ЛВШ проходит в Менделеево."}
    findings = verify_dialogue_contract_output(
        "УНПК МФТИ: ЛВШ проходит в Менделеево.",
        facts=facts,
        active_brand="unpk",
        contract=AnswerContract(active_brand="unpk", current_question=question, answerability="answer_self"),
        client_message=question,
        context={
            FACT_VENUE_SCOPE_ENV: "1",
            "direct_path": {"llm_retrieve": {"venue_scope": {"requested_scope": "moscow_regular"}}},
            "direct_path_fact_metadata": {
                "unpk.lvsh.address": {"venue": "lvsh_mendeleevo", "program_kind": "camp_lvsh"}
            },
        },
    )

    assert any(
        finding.code == "wrong_intent_fact" and "ЛВШ Менделеево" in finding.detail
        for finding in findings
    )


def test_r4_1_live_snapshot_has_structural_venue_scope_markup() -> None:
    snapshot = json.loads(R4_1_SNAPSHOT.read_text(encoding="utf-8"))
    facts = snapshot["facts"]
    venue_counts: dict[str, int] = {}
    program_counts: dict[str, int] = {}
    for fact in facts:
        if fact.get("venue"):
            venue_counts[str(fact["venue"])] = venue_counts.get(str(fact["venue"]), 0) + 1
        if fact.get("program_kind"):
            program_counts[str(fact["program_kind"])] = program_counts.get(str(fact["program_kind"]), 0) + 1

    assert len(facts) == 1075
    assert venue_counts == {
        "any": 90,
        "moscow_regular": 97,
        "online": 78,
        "lvsh_mendeleevo": 119,
        "dolgoprudny": 16,
    }
    assert program_counts == {
        "regular": 171,
        "camp_lvsh": 117,
        "camp_city": 45,
        "any": 37,
        "olympiad": 30,
    }
