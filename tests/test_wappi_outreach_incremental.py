from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path

import pytest

from scripts import build_wappi_outreach_cases as module


# Независимый канон A:AB — переписан из STRATEGY_SOURCE_DRAFT.md, а не импортирован из модуля.
CANONICAL_COLUMNS = [
    "Ключ чата",                # A
    "Статус",                   # B
    "Теплота",                  # C
    "Бренд и канал",            # D
    "Открыть",                  # E
    "Клиент",                   # F
    "Последний контакт",        # G
    "Что хотел",                # H
    "Покупка/запись 2026/27",   # I
    "Следующий шаг",            # J
    "Что предложить",           # K
    "Почему сейчас",            # L
    "Сообщение 1",              # M
    "Дата 1",                   # N
    "Сообщение 2",              # O
    "Дата 2",                   # P
    "Сообщение 3",              # Q
    "Дата 3",                   # R
    "Сообщение 4",              # S
    "Дата 4",                   # T
    "Сообщение 5",              # U
    "Дата 5",                   # V
    "Подпись",                  # W
    "Решение РОПа",             # X
    "Результат",                # Y
    "Доказательства",           # Z
    "Актуальность",             # AA
    "Проверено на",             # AB
]
CANONICAL_VERDICT_COLUMNS = {
    "status": 1, "heat": 2, "past_interest": 7, "purchase_2026_27": 8, "next_step": 9, "offer": 10,
    "why_now": 11, "message_1": 12, "date_1": 13, "message_2": 14, "date_2": 15, "message_3": 16,
    "date_3": 17, "message_4": 18, "date_4": 19, "message_5": 20, "date_5": 21, "signature": 22,
    "freshness": 26,
}
STAMP = "23.08.2026 12:00"


def _write(path: Path, rows: object) -> Path:
    path.write_text(json.dumps(rows, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _call_row(number: str, moscow: str, phone: str, summary: str, next_step: str, transcript: str) -> dict:
    values = [""] * 16
    values[0], values[1], values[6] = number, moscow, phone
    values[9], values[12], values[15] = summary, next_step, transcript
    return {"values": values}


def _evidence(rows: list[dict]) -> dict:
    return {"schema_version": "wappi_outreach_calls_sheet_v1", "source_sheet": "Звонки", "rows": rows}


def _inputs(tmp_path: Path) -> dict[str, Path | None]:
    candidates = [
        {"profile_id": "p1", "chat_id": "c1", "brand": "foton", "channel": "telegram",
         "peer_names": ["Мама Пети"], "last_inbound_ts": 1787217300, "last_outbound_ts": 1787119500,
         "messages": [{"text": "full chat"}]},
        {"profile_id": "p2", "chat_id": "c2", "brand": "unpk", "channel": "max",
         "peer_names": ["Родитель из чата"], "last_inbound_ts": 0, "last_outbound_ts": 1787119500,
         "amo_contact_id": "20", "messages": []},
    ]
    registry = [
        {"profile_id": "p1", "chat_id": "c1", "status": "LINK_EXTRACTION_GAP", "gap_reason": "saved_cache_gap"},
        {"profile_id": "p2", "chat_id": "c2", "amo_contact_id": "20", "amo_lead_ids": ["l2"]},
    ]
    delta = [{"profile_id": "p1", "chat_id": "c1", "amo_contact_id": "10", "amo_lead_ids": ["l1", "l3"]}]
    contacts = [
        {"entity_id": "10", "customer_id": "cu1", "name": "Ирина", "record": {"_embedded": {"leads": [{"id": "l1"}, {"id": "l3"}]}}},
        {"entity_id": "20", "customer_id": "cu2", "record": {"_embedded": {"leads": [{"id": "l2"}]}}},
    ]
    leads = [
        {"entity_id": "l1", "customer_id": "cu1", "record": {"_embedded": {"contacts": [{"id": "10"}]}}},
        {"entity_id": "l3", "customer_id": "cu1", "record": {"_embedded": {"contacts": [{"id": "10"}]}}},
        {"entity_id": "l2", "customer_id": "cu2", "record": {"_embedded": {"contacts": [{"id": "20"}]}}},
    ]
    events = [{"event_id": "e1", "entity_id": "l1", "customer_id": "cu1"}]
    tallanto = [{"tallanto_id": "t1", "amo_contact_id": "10"}]
    calls = [
        {"call_id": "exact", "amo_contact_id": "10", "amo_contact_ids": ["10"], "match_status": "EXACT_AMO_CONTACT",
         "normalized_phone": "+79990000001", "call_at": "2026-08-20 09:15:30"},
        {"call_id": "family", "amo_contact_ids": ["10", "99"], "match_status": "SHARED_PHONE_CONTEXT",
         "normalized_phone": "+79990000002", "call_at": "2026-08-20 10:00:00"},
    ]
    evidence = _evidence([
        _call_row("31", "2026-08-20 12:15:30", "+7 999 000-00-01", "конспект", "следующий шаг", "расшифровка"),
        _call_row("32", "2026-08-20 13:00:00", "79990000002", "семейный конспект", "перезвонить", "семейная расшифровка"),
    ])
    return {
        "candidates_path": _write(tmp_path / "candidates.json", candidates),
        "wappi_registry_path": _write_jsonl(tmp_path / "registry.jsonl", registry),
        "wappi_link_delta_path": _write_jsonl(tmp_path / "delta.jsonl", delta),
        "amo_contacts_path": _write_jsonl(tmp_path / "contacts.jsonl", contacts),
        "amo_leads_path": _write_jsonl(tmp_path / "leads.jsonl", leads),
        "amo_events_path": _write_jsonl(tmp_path / "events.jsonl", events),
        "tallanto_path": _write_jsonl(tmp_path / "tallanto.jsonl", tallanto),
        "calls_sidecar_path": _write_jsonl(tmp_path / "calls.jsonl", calls),
        "calls_evidence_path": _write(tmp_path / "evidence.json", evidence),
    }


# --------------------------------------------------------------------------- пакеты кейсов


def test_delta_join_retains_multiple_deals_and_call_scopes(tmp_path: Path) -> None:
    result = module.build_case_packets(**_inputs(tmp_path))
    case = result["cases"][0]

    assert case["key"] == "p1:c1"
    assert case["identity"]["amo_contact_ids"] == ["10"]
    assert case["identity"]["status"] == "OVERLAY_EXACT_MULTIPLE_LEADS"
    assert case["identity"]["amo_lead_ids"] == ["l1", "l3"]
    assert [row["entity_id"] for row in case["raw_linked_facts"]["amo_leads"]] == ["l1", "l3"]
    assert case["raw_linked_facts"]["wappi_conversation"]["messages"] == [{"text": "full chat"}]
    assert case["raw_linked_facts"]["wappi_registry"]["gap_reason"] == "saved_cache_gap"
    assert "gap_reason" not in case["raw_linked_facts"]["wappi_effective_link"]

    exact = case["raw_linked_facts"]["calls_exact"]
    family = case["raw_linked_facts"]["calls_family"]
    assert exact[0]["context_only"] is False and exact[0]["link"]["call_id"] == "exact"
    assert family[0]["context_only"] is True and family[0]["link"]["call_id"] == "family"
    assert result["case_count"] == sum(result["link_class_counts"].values()) == 2


def test_repeat_is_byte_identical_and_pagination_is_deterministic(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    first = module.build_case_packets(**inputs, offset=1, limit=1)
    second = module.build_case_packets(**inputs, offset=1, limit=1)

    assert module._json_bytes(first, indent=2) == module._json_bytes(second, indent=2)
    assert [case["key"] for case in first["cases"]] == ["p2:c2"]
    assert first["total_candidates"] == 2

    candidates = json.loads(Path(inputs["candidates_path"]).read_text(encoding="utf-8"))
    inputs["candidates_path"] = _write(tmp_path / "ordered.json", list(reversed(candidates)))
    assert module.build_case_packets(**inputs, limit=1)["cases"][0]["key"] == "p1:c1"


def test_duplicate_candidate_key_fails_closed(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    duplicate = [{"profile_id": "p1", "chat_id": "c1"}, {"profile_id": "p1", "chat_id": "c1"}]
    inputs["candidates_path"] = _write(tmp_path / "duplicate.json", duplicate)

    with pytest.raises(ValueError, match="duplicate candidate key: p1:c1"):
        module.build_case_packets(**inputs)


@pytest.mark.parametrize("payload", [{"unexpected": {"id": 1}}, {"single": "record"}, [1]])
def test_unknown_json_shape_is_rejected_instead_of_becoming_a_record(tmp_path: Path, payload: object) -> None:
    path = _write(tmp_path / "unknown.json", payload)

    with pytest.raises(ValueError, match="unsupported JSON payload|non-object record"):
        module._records(path)


@pytest.mark.parametrize("row", [
    {"profile_id": "", "chat_id": "c1"},
    {"chat_id": "c1"},
    {"profile_id": "p1", "chat_id": ""},
    {"profile_id": "p1"},
    {"profile_id": "  ", "chat_id": "c1"},
])
def test_half_empty_key_never_becomes_a_case(tmp_path: Path, row: dict) -> None:
    """Ключ склеивается из двух половин: пустая половина обязана падать, а не давать ':c1'."""
    inputs = _inputs(tmp_path)
    inputs["candidates_path"] = _write(tmp_path / "half-key.json", [row])

    with pytest.raises(ValueError, match="needs both profile_id and chat_id"):
        module.build_case_packets(**inputs)


def test_scalar_list_field_does_not_shatter_into_characters(tmp_path: Path) -> None:
    """amo_lead_ids строкой — это одна сделка '4021', а не четыре сделки '4','0','2','1'."""
    inputs = _inputs(tmp_path)
    delta = [{"profile_id": "p1", "chat_id": "c1", "amo_contact_id": "10", "amo_lead_ids": "4021",
              "tallanto_ids": "t9"}]
    inputs["wappi_link_delta_path"] = _write_jsonl(tmp_path / "scalar-delta.jsonl", delta)
    calls = [{"call_id": "exact", "amo_contact_ids": "10", "match_status": "EXACT_AMO_CONTACT",
              "normalized_phone": "+79990000001", "call_at": "2026-08-20 09:15:30"}]
    inputs["calls_sidecar_path"] = _write_jsonl(tmp_path / "scalar-calls.jsonl", calls)

    identity = module.build_case_packets(**inputs)["cases"][0]["identity"]

    assert "4021" in identity["reported_amo_lead_ids"]
    assert [item for item in identity["reported_amo_lead_ids"] if len(item) == 1] == []
    # Итоговый класс учитывает и свежие AMO-сделки контакта; главное — строка 4021 не стала четырьмя ID.
    assert identity["status"] == "OVERLAY_EXACT_MULTIPLE_LEADS"
    assert identity["tallanto_ids"] == ["t1"]  # t9 не существует в свежей Tallanto-выгрузке
    assert identity["amo_contact_ids"] == ["10"]


def test_registry_gap_does_not_become_exact_from_stale_candidate(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    candidates = json.loads(Path(inputs["candidates_path"]).read_text(encoding="utf-8"))
    candidates[0]["amo_contact_id"] = "10"
    inputs["candidates_path"] = _write(tmp_path / "stale-candidates.json", candidates)
    inputs["wappi_link_delta_path"] = None

    case = module.build_case_packets(**inputs)["cases"][0]

    assert case["identity"]["status"] == "REGISTRY_GAP"
    assert case["identity"]["amo_contact_ids"] == []
    assert case["raw_linked_facts"]["amo_contacts"] == []


def test_real_link_classes_close_balance(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    candidates = [{"profile_id": "p", "chat_id": name, "messages": []} for name in ("gap", "conflict", "many", "single", "contact", "missing")]
    registry = [
        {"profile_id": "p", "chat_id": "gap", "status": "LINK_EXTRACTION_GAP"},
        {"profile_id": "p", "chat_id": "conflict", "status": "LINK_CONFLICT", "amo_contact_id": "10"},
        {"profile_id": "p", "chat_id": "many", "status": "LINK_EXTRACTED", "lead_link_status": "MULTIPLE_LEADS", "amo_contact_id": "20", "amo_lead_ids": ["l2", "l3"]},
        {"profile_id": "p", "chat_id": "single", "status": "LINK_EXTRACTED", "lead_link_status": "SINGLE_LEAD", "amo_contact_id": "20", "amo_lead_ids": ["l2"]},
        {"profile_id": "p", "chat_id": "contact", "status": "LINK_EXTRACTED", "lead_link_status": "CONTACT_ONLY", "amo_contact_id": "10"},
    ]
    inputs["candidates_path"] = _write(tmp_path / "classes.json", candidates)
    inputs["wappi_registry_path"] = _write_jsonl(tmp_path / "classes-registry.jsonl", registry)
    inputs["wappi_link_delta_path"] = None

    result = module.build_case_packets(**inputs)

    by_chat = {case["chat_id"]: case for case in result["cases"]}
    assert {chat: case["identity"]["status"] for chat, case in by_chat.items()} == {
        "gap": "REGISTRY_GAP", "conflict": "REGISTRY_CONFLICT",
        # У many одна из заявленных сделок принадлежит чужому контакту и отбрасывается.
        "many": "REGISTRY_EXACT_SINGLE_LEAD", "single": "REGISTRY_EXACT_SINGLE_LEAD",
        # У contact свежая AMO-карточка содержит две сделки, поэтому старый CONTACT_ONLY не сохраняется.
        "contact": "REGISTRY_EXACT_MULTIPLE_LEADS", "missing": "REGISTRY_MISSING",
    }
    assert result["case_count"] == sum(result["link_class_counts"].values()) == 6
    many = by_chat["many"]
    assert many["identity"]["amo_lead_ids"] == ["l2"]
    assert many["identity"]["reported_amo_lead_ids"] == ["l2", "l3"]
    assert many["identity"]["amo_lead_record_conflict_ids"] == ["l3"]
    assert [row["entity_id"] for row in many["raw_linked_facts"]["amo_leads"]] == ["l2"]


def test_amo_event_type_prevents_contact_lead_id_collision(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    events = [
        {"event_id": "right", "entity_type": "lead", "entity_id": "l1", "customer_id": "cu1"},
        {"event_id": "wrong", "entity_type": "contact", "entity_id": "l1", "customer_id": "cu1"},
        {"event_id": "family", "entity_type": "lead", "entity_id": "other", "customer_id": "cu1"},
    ]
    inputs["amo_events_path"] = _write_jsonl(tmp_path / "typed-events.jsonl", events)

    case = module.build_case_packets(**inputs)["cases"][0]

    assert [row["event_id"] for row in case["raw_linked_facts"]["amo_events"]] == ["right"]


def test_case_fingerprint_changes_only_for_affected_case(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    before = module.build_case_packets(**inputs)
    leads = [json.loads(line) for line in Path(inputs["amo_leads_path"]).read_text(encoding="utf-8").splitlines()]
    leads[0]["changed_fact"] = True
    inputs["amo_leads_path"] = _write_jsonl(tmp_path / "changed-leads.jsonl", leads)

    after = module.build_case_packets(**inputs)

    assert before["cases"][0]["source_fingerprint"] != after["cases"][0]["source_fingerprint"]
    assert before["cases"][1]["source_fingerprint"] == after["cases"][1]["source_fingerprint"]


def test_registry_missing_excludes_stale_identity_and_semantic_candidate_fields(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    candidate = {
        "profile_id": "p-missing", "chat_id": "c-missing", "brand": "foton", "channel": "telegram",
        "messages": [{"text": "чат должен остаться"}], "amo_contact_id": "FOREIGN_CONTACT",
        "amo_lead_id_pair": "FOREIGN_LEAD", "leads": [{"id": "FOREIGN_LEAD"}],
        "tallanto_id": "FOREIGN_TALLANTO", "pool": "legacy_warm", "staff_test": True,
        "p0_codes": ["legacy_p0"], "optout_phrases": ["legacy_optout"],
    }
    inputs["candidates_path"] = _write(tmp_path / "missing.json", [candidate])
    inputs["wappi_registry_path"] = _write_jsonl(tmp_path / "empty-registry.jsonl", [])
    inputs["wappi_link_delta_path"] = None

    case = module.build_case_packets(**inputs)["cases"][0]
    serialized = json.dumps(case, ensure_ascii=False, sort_keys=True)

    assert case["identity"]["status"] == "REGISTRY_MISSING"
    assert case["raw_linked_facts"]["wappi_conversation"]["messages"] == [{"text": "чат должен остаться"}]
    for forbidden in ("FOREIGN_CONTACT", "FOREIGN_LEAD", "FOREIGN_TALLANTO", "legacy_warm", "legacy_p0", "legacy_optout"):
        assert forbidden not in serialized


def test_overlay_reassignment_cannot_keep_the_old_tallanto_card(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    registry = [{"profile_id": "p1", "chat_id": "c1", "amo_contact_id": "20", "amo_lead_ids": ["l2"],
                 "tallanto_ids": ["old"]}]
    delta = [{"profile_id": "p1", "chat_id": "c1", "amo_contact_id": "10", "amo_lead_ids": ["l1"]}]
    tallanto = [
        {"tallanto_id": "old", "amo_contact_id": "20", "name": "чужая карточка"},
        {"tallanto_id": "current", "amo_contact_id": "10", "name": "текущая карточка"},
    ]
    inputs["wappi_registry_path"] = _write_jsonl(tmp_path / "reassigned-registry.jsonl", registry)
    inputs["wappi_link_delta_path"] = _write_jsonl(tmp_path / "reassigned-delta.jsonl", delta)
    inputs["tallanto_path"] = _write_jsonl(tmp_path / "reassigned-tallanto.jsonl", tallanto)

    case = module.build_case_packets(**inputs)["cases"][0]

    assert [row["tallanto_id"] for row in case["raw_linked_facts"]["tallanto"]] == ["current"]
    assert "old" not in case["identity"]["tallanto_ids"]
    assert "old" not in json.dumps(case["raw_linked_facts"]["wappi_effective_link"], ensure_ascii=False)
    assert "l2" not in json.dumps(case["raw_linked_facts"]["wappi_effective_link"], ensure_ascii=False)


# --------------------------------------------------------------------------- join звонков


def test_calls_evidence_joins_exactly_on_phone_and_moscow_second(tmp_path: Path) -> None:
    result = module.build_case_packets(**_inputs(tmp_path))
    case = result["cases"][0]

    assert case["raw_linked_facts"]["calls_exact"][0]["evidence"] == [{
        "call_id": "exact", "report_row": "31", "call_at_moscow": "2026-08-20 12:15:30",
        "phone": "+7 999 000-00-01", "summary": "конспект", "next_step": "следующий шаг",
        "transcript": "расшифровка",
    }]
    assert case["raw_linked_facts"]["calls_family"][0]["evidence"][0]["call_id"] == "family"
    assert result["calls_evidence_diagnostics"] == {
        "sidecar_rows": 2, "sidecar_unparsed": 0,
        "rows": 2, "unparsed": 0, "matched_exact": 2, "unmatched": 0, "ambiguous": 0,
    }


@pytest.mark.parametrize("call_at", [
    "2026-08-20 09:15:30.123456", "2026-08-20T09:15:30+00:00", "2026-08-20T09:15:30Z",
])
def test_calls_evidence_accepts_real_sidecar_time_formats(tmp_path: Path, call_at: str) -> None:
    inputs = _inputs(tmp_path)
    calls = [
        {
            "call_id": "microseconds", "amo_contact_id": "10", "normalized_phone": "+79990000001",
            "call_at": call_at, "match_status": "EXACT_AMO_CONTACT",
        }
    ]
    inputs["calls_sidecar_path"] = _write_jsonl(tmp_path / "microseconds-calls.jsonl", calls)

    result = module.build_case_packets(**inputs)

    assert result["calls_evidence_diagnostics"] == {
        "sidecar_rows": 1, "sidecar_unparsed": 0,
        "rows": 2, "unparsed": 0, "matched_exact": 1, "unmatched": 1, "ambiguous": 0,
    }
    evidence = result["cases"][0]["raw_linked_facts"]["calls_exact"][0]["evidence"]
    assert evidence[0]["call_id"] == "microseconds"


@pytest.mark.parametrize("row, expected", [
    (_call_row("31", "2026-08-20 12:15:31", "+7 999 000-00-01", "к", "ш", "р"), "unmatched"),  # секунда мимо
    (_call_row("31", "2026-08-20 12:15:30", "+7 999 000-00-09", "к", "ш", "р"), "unmatched"),  # другой телефон
    (_call_row("31", "2026-08-20 09:15:30", "+7 999 000-00-01", "к", "ш", "р"), "unmatched"),  # UTC вместо MSK
    (_call_row("31", "не время", "+7 999 000-00-01", "к", "ш", "р"), "unparsed"),
])
def test_call_evidence_without_exact_match_stays_in_diagnostics(tmp_path: Path, row: dict, expected: str) -> None:
    inputs = _inputs(tmp_path)
    inputs["calls_evidence_path"] = _write(tmp_path / "near-miss.json", _evidence([row]))

    result = module.build_case_packets(**inputs)

    assert result["calls_evidence_diagnostics"][expected] == 1
    assert result["calls_evidence_diagnostics"]["matched_exact"] == 0
    assert result["cases"][0]["raw_linked_facts"]["calls_exact"][0]["evidence"] == []


def test_ambiguous_call_evidence_is_never_attached(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["calls_evidence_path"] = _write(tmp_path / "ambiguous.json", _evidence([
        _call_row("31", "2026-08-20 12:15:30", "+79990000001", "первый конспект", "шаг", "расшифровка A"),
        _call_row("41", "2026-08-20 12:15:30", "+79990000001", "второй конспект", "шаг", "расшифровка B"),
    ]))

    result = module.build_case_packets(**inputs)

    assert result["calls_evidence_diagnostics"] == {
        "sidecar_rows": 2, "sidecar_unparsed": 0,
        "rows": 2, "unparsed": 0, "matched_exact": 0, "unmatched": 0, "ambiguous": 2,
    }
    assert result["cases"][0]["raw_linked_facts"]["calls_exact"][0]["evidence"] == []


def test_two_sidecar_calls_in_one_second_are_ambiguous(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    calls = [
        {"call_id": call_id, "amo_contact_id": "10", "normalized_phone": "+79990000001",
         "call_at": "2026-08-20 09:15:30.000000", "match_status": "EXACT_AMO_CONTACT"}
        for call_id in ("a", "b")
    ]
    inputs["calls_sidecar_path"] = _write_jsonl(tmp_path / "same-second.jsonl", calls)

    result = module.build_case_packets(**inputs)

    assert result["calls_evidence_diagnostics"]["ambiguous"] == 1
    assert all(not call["evidence"] for call in result["cases"][0]["raw_linked_facts"]["calls_exact"])


def test_calls_evidence_rejects_wrong_wrapper_or_width(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["calls_evidence_path"] = _write(tmp_path / "wrong-wrapper.json", {"rows": []})
    with pytest.raises(ValueError, match="calls evidence schema mismatch"):
        module.build_case_packets(**inputs)

    inputs["calls_evidence_path"] = _write(
        tmp_path / "wrong-width.json", _evidence([{"values": [""] * 15}]),
    )
    with pytest.raises(ValueError, match="exactly 16 columns"):
        module.build_case_packets(**inputs)


def test_unclassified_call_is_visible_and_blocks_send(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    calls = [{"call_id": "future", "amo_contact_id": "10", "normalized_phone": "+79990000001",
              "call_at": "2026-08-20 09:15:30.000000", "match_status": "FUTURE_CLASS"}]
    inputs["calls_sidecar_path"] = _write_jsonl(tmp_path / "future-calls.jsonl", calls)
    payload = module.build_case_packets(**inputs)
    case = payload["cases"][0]
    assert case["raw_linked_facts"]["calls_exact"] == []
    assert case["raw_linked_facts"]["calls_family"] == []
    assert case["raw_linked_facts"]["calls_unclassified"][0]["link"]["call_id"] == "future"

    case["identity"].update(status="OVERLAY_EXACT_SINGLE_LEAD", amo_contact_ids=["10"], amo_lead_ids=["1"])
    rows = [{"key": item["key"], "verdict": _verdict()} for item in payload["cases"]]
    rows[0]["verdict"]["status"] = module.SEND_STATUS
    sheet = module.build_sheet_rows(payload, rows, checked_at=STAMP)
    assert sheet["analysis_errors"]["by_key"]["p1:c1"] == "UNSAFE_SEND"
    assert _flat(sheet["rows"][0])[1] == "ПРОВЕРКА"


# --------------------------------------------------------------------------- строки листа


def _verdict(marker: str = "v", **overrides: str) -> dict[str, str]:
    verdict = {name: f"{name}:{marker}" for name in CANONICAL_VERDICT_COLUMNS}
    verdict["status"] = "ПРОВЕРКА"
    verdict["freshness"] = "АКТУАЛЬНО"
    verdict.update(overrides)
    return verdict


def _verdict_rows(tmp_path: Path, marker: str = "v") -> list[dict]:
    result = module.build_case_packets(**_inputs(tmp_path))
    return [{"key": case["key"], "verdict": _verdict(marker)} for case in result["cases"]]


def _sheet(tmp_path: Path, verdict_rows: list[dict] | None = None, **kwargs) -> dict:
    payload = module.build_case_packets(**_inputs(tmp_path))
    rows = _verdict_rows(tmp_path) if verdict_rows is None else verdict_rows
    return module.build_sheet_rows(payload, rows, checked_at=STAMP, **kwargs)


def _flat(row: dict) -> dict[int, str]:
    """Абсолютный индекс колонки -> значение, восстановленный из двух записываемых блоков."""
    cells: dict[int, str] = {}
    for name, values in row["values"].items():
        start = module.WRITE_SLICES[name].start
        cells.update({start + offset: value for offset, value in enumerate(values)})
    return cells


def test_sheet_schema_matches_independent_canon(tmp_path: Path) -> None:
    sheet = _sheet(tmp_path)

    assert sheet["columns"] == CANONICAL_COLUMNS
    assert sheet["column_count"] == 28
    assert module.VERDICT_COLUMNS == CANONICAL_VERDICT_COLUMNS
    assert sheet["write_ranges"] == ["A:W", "Z:AB"]
    assert sheet["row_count"] == 2
    assert sheet["row_count"] == sum(sheet["status_counts"].values())
    assert [row["key"] for row in sheet["rows"]] == ["p1:c1", "p2:c2"]

    for name, index in CANONICAL_VERDICT_COLUMNS.items():
        assert _flat(sheet["rows"][0])[index] == _verdict()[name], name
    # Детерминированные колонки принадлежат коду, а не LLM.
    assert set(CANONICAL_VERDICT_COLUMNS.values()).isdisjoint({0, 3, 4, 5, 6, 23, 24, 25, 27})

    for row in sheet["rows"]:
        assert row["anchor"] == {"column": "A", "value": row["key"]}
        assert len(row["values"]["A:W"]) == 23 and len(row["values"]["Z:AB"]) == 3
        assert set(row["values"]) == {"A:W", "Z:AB"}
        assert _flat(row)[0] == row["key"]


def test_written_blocks_never_carry_the_rop_columns(tmp_path: Path) -> None:
    sheet = _sheet(tmp_path)

    assert sheet["manual_columns"] == {
        "range": "X:Y", "indexes": [23, 24], "headers": ["Решение РОПа", "Результат"],
        "policy": "preserve", "excluded_from_output": True,
        "allowed_decisions": ["", "ОК", "ПРАВИТЬ", "НЕ ПИСАТЬ"],
        "allowed_results": ["", "ОТПРАВЛЕНО-1", "ОТПРАВЛЕНО-2", "ОТПРАВЛЕНО-3",
                            "ОТПРАВЛЕНО-4", "ОТПРАВЛЕНО-5", "ОТВЕТИЛ", "КУПИЛ", "СТОП"],
    }
    for row in sheet["rows"]:
        assert 23 not in _flat(row) and 24 not in _flat(row)
        assert None not in row["values"]["A:W"] and None not in row["values"]["Z:AB"]
    assert "Решение РОПа" not in json.dumps([row["values"] for row in sheet["rows"]], ensure_ascii=False)

    rows = _verdict_rows(tmp_path)
    rows[0]["verdict"]["rop_decision"] = "ОК"
    rejected = _sheet(tmp_path, rows)
    assert rejected["analysis_errors"]["by_key"]["p1:c1"] == "VERDICT_INVALID"
    assert _flat(rejected["rows"][0])[1] == "ПРОВЕРКА"


def test_fake_writer_applying_blocks_preserves_existing_x_y(tmp_path: Path) -> None:
    """Писатель кладёт ровно два блока по якорю A; заполненные РОПом X:Y обязаны выжить."""
    existing = {row["key"]: [f"старое-{index}" for index in range(28)] for row in _sheet(tmp_path)["rows"]}
    for cells in existing.values():
        cells[23], cells[24] = "РОП: писать, ребёнок 9 класс", "оплатил 14.08"

    for row in _sheet(tmp_path)["rows"]:
        cells = existing[row["anchor"]["value"]]
        assert row["anchor"]["column"] == "A"
        for name, values in row["values"].items():
            start = module.WRITE_SLICES[name].start
            cells[start:start + len(values)] = values

    for key, cells in existing.items():
        assert cells[23] == "РОП: писать, ребёнок 9 класс"
        assert cells[24] == "оплатил 14.08"
        assert cells[0] == key and cells[22] == "signature:v" and cells[27] == STAMP
        assert "старое-" not in json.dumps(cells[:23] + cells[25:], ensure_ascii=False)


def test_deterministic_columns_come_from_raw_facts_not_from_llm(tmp_path: Path) -> None:
    sheet = _sheet(tmp_path)
    first, second = _flat(sheet["rows"][0]), _flat(sheet["rows"][1])

    assert first[3] == "Фотон · Telegram" and second[3] == "УНПК МФТИ · Max"
    assert first[4] == "https://educent.amocrm.ru/contacts/detail/10"
    assert first[5] == "Ирина"                     # единственное точное имя AMO
    assert second[5] == "Родитель из чата"          # имени в AMO нет -> peer_names, без выдумки
    assert first[6] == "20.08.2026 12:15 · последним: клиент"
    assert second[6] == "19.08.2026 09:05 · последним: мы"
    assert f"source_fingerprint={sheet['rows'][0]['source_fingerprint']}" in first[25]


@pytest.mark.parametrize("chat_patch, expected", [
    ({"last_inbound_ts": 1787217300000}, "20.08.2026 12:15 · последним: клиент"),
    ({"last_inbound_ts": 0, "last_outbound_ts": None}, ""),          # времени нет — пусто, а не догадка
    ({"last_inbound_ts": "мусор", "last_outbound_ts": 0}, ""),
    ({"last_inbound_ts": 0, "last_outbound_ts": 1787119500}, "19.08.2026 09:05 · последним: мы"),
])
def test_last_contact_cell_is_a_moscow_timestamp_or_nothing(
    tmp_path: Path, chat_patch: dict, expected: str
) -> None:
    inputs = _inputs(tmp_path)
    candidates = json.loads(Path(inputs["candidates_path"]).read_text(encoding="utf-8"))
    candidates[0].update(chat_patch)
    inputs["candidates_path"] = _write(tmp_path / "stamps.json", candidates)
    payload = module.build_case_packets(**inputs)

    sheet = module.build_sheet_rows(payload, _verdict_rows(tmp_path), checked_at=STAMP)

    assert _flat(sheet["rows"][0])[6] == expected


def test_two_amo_names_fall_back_to_the_chat_instead_of_picking_one(tmp_path: Path) -> None:
    """Точного имени нет — берём имя из чата, а не одно из двух возможных."""
    inputs = _inputs(tmp_path)
    contacts = [json.loads(line) for line in Path(inputs["amo_contacts_path"]).read_text(encoding="utf-8").splitlines()]
    contacts.append({**contacts[0], "name": "Ирина Петровна"})
    inputs["amo_contacts_path"] = _write_jsonl(tmp_path / "two-names.jsonl", contacts)
    payload = module.build_case_packets(**inputs)

    cells = _flat(module.build_sheet_rows(payload, _verdict_rows(tmp_path), checked_at=STAMP)["rows"][0])

    assert cells[5] == "Мама Пети"


def test_unknown_brand_or_channel_fails_instead_of_mixing(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    candidates = json.loads(Path(inputs["candidates_path"]).read_text(encoding="utf-8"))
    candidates[0]["brand"] = "?"
    inputs["candidates_path"] = _write(tmp_path / "unknown-brand.json", candidates)
    payload = module.build_case_packets(**inputs)

    with pytest.raises(ValueError, match="unknown brand/channel for p1:c1"):
        module.build_sheet_rows(payload, _verdict_rows(tmp_path), checked_at=STAMP)


def test_amo_link_needs_a_numeric_id(tmp_path: Path) -> None:
    assert module._amo_url({"amo_lead_ids": ["l1"], "amo_contact_ids": ["10"]}) == \
        "https://educent.amocrm.ru/contacts/detail/10"
    assert module._amo_url({"amo_lead_ids": ["4021"], "amo_contact_ids": ["10"]}) == \
        "https://educent.amocrm.ru/leads/detail/4021"
    assert module._amo_url({"amo_lead_ids": ["l1"], "amo_contact_ids": ["cu1"]}) == ""
    assert module._amo_url({"amo_lead_ids": [], "amo_contact_ids": []}) == ""


# --------------------------------------------------------------------------- структурная безопасность


def test_status_and_freshness_are_a_closed_schema(tmp_path: Path) -> None:
    assert module.STATUS_VALUES == (
        "ОТПРАВИТЬ ПОСЛЕ ПРОВЕРКИ ЧАТА", "МЕНЕДЖЕР СЕГОДНЯ", "ПРОВЕРКА", "НЕ ПИСАТЬ",
    )
    assert module.FRESHNESS_VALUES == ("АКТУАЛЬНО", "ОБНОВИТЬ", "СТОП ДЕЙСТВУЕТ")

    for status in module.STATUS_VALUES[1:]:
        rows = _verdict_rows(tmp_path)
        rows[0]["verdict"]["status"] = status
        assert _flat(_sheet(tmp_path, rows)["rows"][0])[1] == status
    for freshness in module.FRESHNESS_VALUES:
        rows = _verdict_rows(tmp_path)
        rows[0]["verdict"]["freshness"] = freshness
        assert _flat(_sheet(tmp_path, rows)["rows"][0])[26] == freshness


@pytest.mark.parametrize("freshness", ["ОБНОВИТЬ", "СТОП ДЕЙСТВУЕТ"])
def test_send_status_requires_current_sources(tmp_path: Path, freshness: str) -> None:
    rows = _verdict_rows(tmp_path)
    rows[0]["verdict"].update(status=module.SEND_STATUS, freshness=freshness)

    sheet = _sheet(tmp_path, rows)

    assert sheet["analysis_errors"]["by_key"]["p1:c1"] == "VERDICT_INVALID"
    assert _flat(sheet["rows"][0])[1] == "ПРОВЕРКА"


@pytest.mark.parametrize("field, value, message", [
    ("status", "ГОТОВО", "status outside schema"),
    ("status", "ПРОВЕРКА ", "status outside schema"),
    ("status", "не писать", "status outside schema"),
    ("freshness", "ЖИВОЙ", "freshness outside schema"),
    ("freshness", "", "freshness outside schema"),
])
def test_value_outside_the_enum_is_quarantined(tmp_path: Path, field: str, value: str, message: str) -> None:
    rows = _verdict_rows(tmp_path)
    rows[0]["verdict"][field] = value

    sheet = _sheet(tmp_path, rows)

    assert sheet["analysis_errors"]["by_key"]["p1:c1"] == "VERDICT_INVALID"
    assert _flat(sheet["rows"][0])[1] == "ПРОВЕРКА"


def test_send_status_requires_exact_single_lead_identity(tmp_path: Path) -> None:
    """p1:c1 — OVERLAY_EXACT_MULTIPLE_LEADS: «отправить» по нему запрещено структурно."""
    rows = _verdict_rows(tmp_path)
    rows[0]["verdict"]["status"] = module.SEND_STATUS

    sheet = _sheet(tmp_path, rows)

    assert sheet["analysis_errors"]["by_key"]["p1:c1"] == "UNSAFE_SEND"
    assert _flat(sheet["rows"][0])[1] == "ПРОВЕРКА"


def test_send_status_is_allowed_on_exact_single_lead(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    delta = [{"profile_id": "p1", "chat_id": "c1", "amo_contact_id": "10", "amo_lead_ids": ["l1"]}]
    inputs["wappi_link_delta_path"] = _write_jsonl(tmp_path / "single-delta.jsonl", delta)
    contacts = [json.loads(line) for line in Path(inputs["amo_contacts_path"]).read_text(encoding="utf-8").splitlines()]
    contacts[0]["record"]["_embedded"]["leads"] = [{"id": "l1"}]
    inputs["amo_contacts_path"] = _write_jsonl(tmp_path / "single-contacts.jsonl", contacts)
    leads = [json.loads(line) for line in Path(inputs["amo_leads_path"]).read_text(encoding="utf-8").splitlines()]
    inputs["amo_leads_path"] = _write_jsonl(tmp_path / "single-leads.jsonl", [row for row in leads if row["entity_id"] != "l3"])
    payload = module.build_case_packets(**inputs)
    rows = [{"key": case["key"], "verdict": _verdict()} for case in payload["cases"]]
    rows[0]["verdict"]["status"] = module.SEND_STATUS

    sheet = module.build_sheet_rows(payload, rows, checked_at=STAMP)

    assert payload["cases"][0]["identity"]["status"] == "OVERLAY_EXACT_SINGLE_LEAD"
    assert payload["cases"][0]["identity"]["amo_records_current"] is True
    assert _flat(sheet["rows"][0])[1] == module.SEND_STATUS


def test_exact_call_without_transcript_evidence_cannot_send(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["wappi_link_delta_path"] = _write_jsonl(tmp_path / "single-delta.jsonl", [
        {"profile_id": "p1", "chat_id": "c1", "amo_contact_id": "10", "amo_lead_ids": ["l1"]},
    ])
    contacts = [json.loads(line) for line in Path(inputs["amo_contacts_path"]).read_text(encoding="utf-8").splitlines()]
    contacts[0]["record"]["_embedded"]["leads"] = [{"id": "l1"}]
    inputs["amo_contacts_path"] = _write_jsonl(tmp_path / "single-contacts.jsonl", contacts)
    inputs["amo_leads_path"] = _write_jsonl(tmp_path / "single-leads.jsonl", [
        {"entity_id": "l1", "customer_id": "cu1", "record": {"_embedded": {"contacts": [{"id": "10"}]}}},
    ])
    inputs["calls_evidence_path"] = _write(tmp_path / "empty-evidence.json", _evidence([]))
    payload = module.build_case_packets(**inputs)
    rows = [{"key": case["key"], "verdict": _verdict()} for case in payload["cases"]]
    rows[0]["verdict"]["status"] = module.SEND_STATUS

    sheet = module.build_sheet_rows(payload, rows, checked_at=STAMP)

    assert sheet["analysis_errors"]["by_key"]["p1:c1"] == "CALL_EVIDENCE_MISSING"
    assert _flat(sheet["rows"][0])[1] == "ПРОВЕРКА"


def test_exact_registry_ids_without_current_amo_records_cannot_send(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs["wappi_link_delta_path"] = _write_jsonl(tmp_path / "id-only.jsonl", [
        {"profile_id": "p1", "chat_id": "c1", "amo_contact_id": "10", "amo_lead_ids": ["l1"]},
    ])
    inputs["amo_contacts_path"] = _write_jsonl(tmp_path / "no-contacts.jsonl", [])
    inputs["amo_leads_path"] = _write_jsonl(tmp_path / "no-leads.jsonl", [])
    payload = module.build_case_packets(**inputs)
    rows = [{"key": case["key"], "verdict": _verdict()} for case in payload["cases"]]
    rows[0]["verdict"]["status"] = module.SEND_STATUS

    assert payload["cases"][0]["identity"]["status"] == "OVERLAY_EXACT_SINGLE_LEAD"
    assert payload["cases"][0]["identity"]["amo_records_current"] is False
    sheet = module.build_sheet_rows(payload, rows, checked_at=STAMP)
    assert sheet["analysis_errors"]["by_key"]["p1:c1"] == "UNSAFE_SEND"
    assert _flat(sheet["rows"][0])[1] == "ПРОВЕРКА"


def test_fabricated_single_lead_label_without_ids_cannot_send(tmp_path: Path) -> None:
    payload = module.build_case_packets(**_inputs(tmp_path))
    payload["cases"][0]["identity"].update(
        status="FABRICATED_EXACT_SINGLE_LEAD", amo_contact_ids=[], amo_lead_ids=[],
    )
    rows = _verdict_rows(tmp_path)
    rows[0]["verdict"]["status"] = module.SEND_STATUS

    sheet = module.build_sheet_rows(payload, rows, checked_at=STAMP)
    assert sheet["analysis_errors"]["by_key"]["p1:c1"] == "UNSAFE_SEND"
    assert _flat(sheet["rows"][0])[1] == "ПРОВЕРКА"


@pytest.mark.parametrize(("manual", "expected"), [
    ({"rop_decision": "", "result": "КУПИЛ"}, "НЕ ПИСАТЬ"),
    ({"rop_decision": "", "result": "СТОП"}, "НЕ ПИСАТЬ"),
    ({"rop_decision": "", "result": "ОТВЕТИЛ"}, "МЕНЕДЖЕР СЕГОДНЯ"),
    ({"rop_decision": "", "result": "ОТПРАВЛЕНО-1"}, "ПРОВЕРКА"),
    ({"rop_decision": "ПРАВИТЬ", "result": ""}, "ПРОВЕРКА"),
    ({"rop_decision": "НЕ ПИСАТЬ", "result": ""}, "НЕ ПИСАТЬ"),
])
def test_manual_campaign_state_cannot_turn_back_into_send(tmp_path: Path, manual: dict, expected: str) -> None:
    payload = module.build_case_packets(**_inputs(tmp_path))
    rows = _verdict_rows(tmp_path)
    rows[0]["verdict"]["status"] = module.SEND_STATUS

    sheet = module.build_sheet_rows(
        payload, rows, checked_at=STAMP, manual_state_rows=[{"key": "p1:c1", **manual}],
    )

    assert _flat(sheet["rows"][0])[1] == expected
    assert sheet["rows"][0]["verdict"]["status"] == module.SEND_STATUS
    assert sheet["rows"][0]["manual_status_override"] is True
    if expected == "НЕ ПИСАТЬ":
        assert _flat(sheet["rows"][0])[26] == "СТОП ДЕЙСТВУЕТ"
    if manual["result"]:
        assert manual["result"] not in _flat(sheet["rows"][0])


def test_manual_state_schema_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="manual state outside schema"):
        _sheet(tmp_path, manual_state_rows=[
            {"key": "p1:c1", "rop_decision": "", "result": "оплатил вчера"},
        ])


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda rows: rows.append(dict(rows[0])), "duplicate verdict key: p1:c1"),
        (lambda rows: rows.append({"key": "p9:c9", "verdict": _verdict()}), "unknown verdict key: 'p9:c9'"),
    ],
)
def test_systemic_verdict_key_errors_fail_closed(tmp_path: Path, mutate, message: str) -> None:
    rows = _verdict_rows(tmp_path)
    mutate(rows)

    with pytest.raises(ValueError, match=message):
        _sheet(tmp_path, rows)


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda rows: rows.pop(0), "VERDICT_MISSING"),
        (lambda rows: rows[0]["verdict"].pop("status"), r"missing=\['status'\]"),
        (lambda rows: rows[0]["verdict"].update(surprise="x"), r"unknown=\['surprise'\]"),
        (lambda rows: rows[0].update(extra="x"), r"must hold exactly key\+verdict"),
        (lambda rows: rows[0].update(verdict="not an object"), "verdict must be an object"),
        (lambda rows: rows[0]["verdict"].update(signature=None), "must be strings"),
    ],
)
def test_one_bad_verdict_is_quarantined_without_stopping_other_rows(tmp_path: Path, mutate, code: str) -> None:
    rows = _verdict_rows(tmp_path)
    mutate(rows)

    sheet = _sheet(tmp_path, rows)

    assert sheet["row_count"] == 2
    assert sheet["analysis_errors"] == {"count": 1, "by_key": {"p1:c1": code if code == "VERDICT_MISSING" else "VERDICT_INVALID"}}
    assert sheet["rows"][0]["analysis_error"] is not None
    assert _flat(sheet["rows"][0])[1] == "ПРОВЕРКА"
    assert _flat(sheet["rows"][0])[12] == ""
    assert _flat(sheet["rows"][0])[26] == "ОБНОВИТЬ"
    assert sheet["rows"][1]["analysis_error"] is None


# --------------------------------------------------------------------------- отпечатки и инкремент


def test_sheet_repeat_is_byte_identical_and_verdict_change_moves_fingerprint(tmp_path: Path) -> None:
    first = _sheet(tmp_path)

    assert module._json_bytes(first, indent=2) == module._json_bytes(_sheet(tmp_path), indent=2)

    rows = _verdict_rows(tmp_path)
    rows[0]["verdict"]["why_now"] = "другой повод"
    changed = _sheet(tmp_path, rows)

    assert changed["rows"][0]["verdict_fingerprint"] != first["rows"][0]["verdict_fingerprint"]
    assert changed["rows"][0]["row_fingerprint"] != first["rows"][0]["row_fingerprint"]
    assert changed["rows"][0]["source_fingerprint"] == first["rows"][0]["source_fingerprint"]
    assert changed["rows"][1]["row_fingerprint"] == first["rows"][1]["row_fingerprint"]


def test_source_change_moves_row_fingerprint_without_touching_verdict(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    before = module.build_sheet_rows(module.build_case_packets(**inputs), _verdict_rows(tmp_path), checked_at=STAMP)
    leads = [json.loads(line) for line in Path(inputs["amo_leads_path"]).read_text(encoding="utf-8").splitlines()]
    leads[0]["changed_fact"] = True
    inputs["amo_leads_path"] = _write_jsonl(tmp_path / "changed-leads.jsonl", leads)

    after = module.build_sheet_rows(module.build_case_packets(**inputs), _verdict_rows(tmp_path), checked_at=STAMP)

    assert after["rows"][0]["source_fingerprint"] != before["rows"][0]["source_fingerprint"]
    assert after["rows"][0]["row_fingerprint"] != before["rows"][0]["row_fingerprint"]
    assert after["rows"][0]["verdict_fingerprint"] == before["rows"][0]["verdict_fingerprint"]


def test_recheck_stamp_is_outside_the_row_fingerprint(tmp_path: Path) -> None:
    first = _sheet(tmp_path)
    payload = module.build_case_packets(**_inputs(tmp_path))
    later = module.build_sheet_rows(payload, _verdict_rows(tmp_path), checked_at="24.08.2026 09:30")

    assert [row["row_fingerprint"] for row in later["rows"]] == [row["row_fingerprint"] for row in first["rows"]]
    assert _flat(later["rows"][0])[27] == "24.08.2026 09:30"
    assert later["checked_at"] == "24.08.2026 09:30"


@pytest.mark.parametrize("stamp", ["", "   ", None])
def test_empty_checked_at_is_rejected(tmp_path: Path, stamp) -> None:
    payload = module.build_case_packets(**_inputs(tmp_path))

    with pytest.raises(ValueError, match="checked_at must be a non-empty stamp"):
        module.build_sheet_rows(payload, _verdict_rows(tmp_path), checked_at=stamp)


def test_unchanged_source_reuses_the_previous_verdict_without_a_new_one(tmp_path: Path) -> None:
    previous = _sheet(tmp_path)

    reused = _sheet(tmp_path, [], previous_sheet=previous)

    assert reused["reuse"] == {"new": 0, "reused": 2, "required_keys": []}
    assert all(row["reused_verdict"] for row in reused["rows"])
    assert [row["row_fingerprint"] for row in reused["rows"]] == [row["row_fingerprint"] for row in previous["rows"]]
    assert reused["rows"][0]["verdict"] == previous["rows"][0]["verdict"]


def test_manual_result_changes_render_without_a_new_llm_verdict(tmp_path: Path) -> None:
    payload = module.build_case_packets(**_inputs(tmp_path))
    previous = _sheet(tmp_path)

    sheet = module.build_sheet_rows(
        payload, [], checked_at=STAMP, previous_sheet=previous,
        manual_state_rows=[{"key": "p1:c1", "rop_decision": "", "result": "КУПИЛ"}],
    )

    assert sheet["reuse"] == {"new": 0, "reused": 2, "required_keys": []}
    assert _flat(sheet["rows"][0])[1] == "НЕ ПИСАТЬ"


def test_manual_terminal_status_stays_safe_when_next_run_forgets_manual_state(tmp_path: Path) -> None:
    payload = module.build_case_packets(**_inputs(tmp_path))
    rows = _verdict_rows(tmp_path)
    rows[0]["verdict"].update(status=module.SEND_STATUS, freshness="АКТУАЛЬНО")
    previous = module.build_sheet_rows(
        payload, rows, checked_at=STAMP,
        manual_state_rows=[{"key": "p1:c1", "rop_decision": "", "result": "КУПИЛ"}],
    )

    repeated = module.build_sheet_rows(payload, [], checked_at=STAMP, previous_sheet=previous)

    assert _flat(previous["rows"][0])[1] == "НЕ ПИСАТЬ"
    assert _flat(repeated["rows"][0])[1] == "НЕ ПИСАТЬ"
    assert _flat(repeated["rows"][0])[26] == "СТОП ДЕЙСТВУЕТ"


def test_manual_terminal_status_survives_source_change_without_manual_state(tmp_path: Path) -> None:
    previous = _sheet(
        tmp_path,
        manual_state_rows=[{"key": "p1:c1", "rop_decision": "", "result": "СТОП"}],
    )
    inputs = _inputs(tmp_path)
    leads = [json.loads(line) for line in Path(inputs["amo_leads_path"]).read_text(encoding="utf-8").splitlines()]
    leads[0]["changed_fact"] = True
    inputs["amo_leads_path"] = _write_jsonl(tmp_path / "changed-leads.jsonl", leads)
    payload = module.build_case_packets(**inputs)
    fresh = [{"key": "p1:c1", "verdict": _verdict("fresh", status=module.SEND_STATUS)}]

    repeated = module.build_sheet_rows(payload, fresh, checked_at=STAMP, previous_sheet=previous)

    assert _flat(repeated["rows"][0])[1] == "НЕ ПИСАТЬ"
    assert _flat(repeated["rows"][0])[26] == "СТОП ДЕЙСТВУЕТ"
    assert repeated["rows"][0]["manual_status_override"] is True


def test_bad_new_verdict_never_falls_back_to_a_reusable_send(tmp_path: Path) -> None:
    payload = module.build_case_packets(**_inputs(tmp_path))
    payload["cases"][0]["identity"].update(
        status="OVERLAY_EXACT_SINGLE_LEAD", amo_contact_ids=["10"], amo_lead_ids=["l1"],
        amo_records_current=True,
    )
    original = _verdict_rows(tmp_path)
    original[0]["verdict"].update(status=module.SEND_STATUS, freshness="АКТУАЛЬНО")
    previous = module.build_sheet_rows(payload, original, checked_at=STAMP)
    broken = [{"key": "p1:c1", "verdict": {"status": "ГОТОВО"}}]

    repeated = module.build_sheet_rows(payload, broken, checked_at=STAMP, previous_sheet=previous)

    assert repeated["analysis_errors"]["by_key"] == {"p1:c1": "VERDICT_INVALID"}
    assert _flat(repeated["rows"][0])[1] == "ПРОВЕРКА"
    assert repeated["rows"][0]["reused_verdict"] is False


def test_changed_source_requires_a_fresh_verdict_for_that_key(tmp_path: Path) -> None:
    previous = _sheet(tmp_path)
    inputs = _inputs(tmp_path)
    leads = [json.loads(line) for line in Path(inputs["amo_leads_path"]).read_text(encoding="utf-8").splitlines()]
    leads[0]["changed_fact"] = True
    inputs["amo_leads_path"] = _write_jsonl(tmp_path / "changed-leads.jsonl", leads)
    payload = module.build_case_packets(**inputs)

    missing = module.build_sheet_rows(payload, [], checked_at=STAMP, previous_sheet=previous)
    assert missing["analysis_errors"]["by_key"] == {"p1:c1": "VERDICT_MISSING"}
    assert [row["reused_verdict"] for row in missing["rows"]] == [False, True]

    fresh = [{"key": "p1:c1", "verdict": _verdict("свежий")}]
    mixed = module.build_sheet_rows(payload, fresh, checked_at=STAMP, previous_sheet=previous)

    assert mixed["reuse"] == {"new": 1, "reused": 1, "required_keys": ["p1:c1"]}
    assert [row["reused_verdict"] for row in mixed["rows"]] == [False, True]
    assert mixed["rows"][0]["verdict"]["why_now"] == "why_now:свежий"
    assert mixed["rows"][1]["verdict"] == previous["rows"][1]["verdict"]


def test_key_absent_from_the_previous_sheet_is_required(tmp_path: Path) -> None:
    previous = _sheet(tmp_path)
    previous["rows"] = [row for row in previous["rows"] if row["key"] != "p2:c2"]
    payload = module.build_case_packets(**_inputs(tmp_path))

    result = module.build_sheet_rows(payload, [], checked_at=STAMP, previous_sheet=previous)
    assert result["analysis_errors"]["by_key"] == {"p2:c2": "VERDICT_MISSING"}
    assert _flat(result["rows"][1])[1] == "ПРОВЕРКА"


def test_sheet_rejects_a_case_payload_from_another_schema(tmp_path: Path) -> None:
    payload = module.build_case_packets(**_inputs(tmp_path))
    payload["schema_version"] = "other_schema"

    with pytest.raises(ValueError, match="case payload schema mismatch"):
        module.build_sheet_rows(payload, _verdict_rows(tmp_path), checked_at=STAMP)


@pytest.mark.parametrize("break_previous, message", [
    (lambda sheet: sheet.update(schema_version="whatever_v0"), "previous sheet schema mismatch"),
    (lambda sheet: sheet["rows"][0].pop("source_fingerprint"), "previous sheet row is unusable"),
    (lambda sheet: sheet["rows"][0].update(key=""), "previous sheet row is unusable"),
    (lambda sheet: sheet["rows"][0]["verdict"].update(status="ГОТОВО"), "status outside schema"),
    (lambda sheet: sheet["rows"][0].pop("verdict"), "verdict must be an object"),
])
def test_unusable_previous_sheet_never_silently_reuses(tmp_path: Path, break_previous, message: str) -> None:
    previous = _sheet(tmp_path)
    break_previous(previous)
    payload = module.build_case_packets(**_inputs(tmp_path))

    with pytest.raises(ValueError, match=message):
        module.build_sheet_rows(payload, [], checked_at=STAMP, previous_sheet=previous)


# --------------------------------------------------------------------------- CLI: входы read-only


def _argv(tmp_path: Path, inputs: dict, **extra: object) -> list[str]:
    argv = ["build_wappi_outreach_cases.py"]
    for name, value in inputs.items():
        if value is not None:
            argv += [f"--{name[:-5].replace('_', '-')}", str(value)]
    for name, value in extra.items():
        if value is not None:
            argv += [f"--{name.replace('_', '-')}", str(value)]
    return argv


def _run(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", argv)
    return module.main()


def test_cli_writes_payload_and_sheet(tmp_path: Path, monkeypatch, capsys) -> None:
    inputs = _inputs(tmp_path)
    verdicts = _write(tmp_path / "verdicts.json", _verdict_rows(tmp_path))
    out, sheet_out = tmp_path / "out" / "cases.json", tmp_path / "out" / "sheet.json"

    assert _run(monkeypatch, _argv(tmp_path, inputs, out=out, verdicts=verdicts,
                                   sheet_out=sheet_out, checked_at=STAMP)) == 0

    sheet = json.loads(sheet_out.read_text(encoding="utf-8"))
    assert json.loads(out.read_text(encoding="utf-8"))["case_count"] == 2
    assert sheet["reuse"] == {"new": 2, "reused": 0, "required_keys": ["p1:c1", "p2:c2"]}
    assert "new=2 reused=0 errors=0" in capsys.readouterr().out
    assert list(tmp_path.glob("**/*.tmp")) == []


def test_cli_reuses_the_previous_sheet(tmp_path: Path, monkeypatch) -> None:
    inputs = _inputs(tmp_path)
    previous = _write(tmp_path / "previous.json", _sheet(tmp_path))
    empty = _write(tmp_path / "no-verdicts.json", [])
    out, sheet_out = tmp_path / "cases.json", tmp_path / "sheet.json"

    assert _run(monkeypatch, _argv(tmp_path, inputs, out=out, verdicts=empty, sheet_out=sheet_out,
                                   checked_at=STAMP, previous_sheet=previous)) == 0

    assert json.loads(sheet_out.read_text(encoding="utf-8"))["reuse"]["reused"] == 2


@pytest.mark.parametrize("collide", ["candidates_path", "amo_leads_path", "calls_evidence_path"])
def test_cli_refuses_to_write_over_an_input(tmp_path: Path, monkeypatch, collide: str) -> None:
    inputs = _inputs(tmp_path)
    target = Path(inputs[collide])
    before = target.read_bytes()

    with pytest.raises(SystemExit):
        _run(monkeypatch, _argv(tmp_path, inputs, out=target))
    # Тот же файл через './' и симлинк — то же самое решение.
    with pytest.raises(SystemExit):
        _run(monkeypatch, _argv(tmp_path, inputs, out=target.parent / "." / target.name))
    case_variant = target.with_name(target.name.upper())
    if case_variant.exists() and case_variant.samefile(target):
        with pytest.raises(SystemExit):
            _run(monkeypatch, _argv(tmp_path, inputs, out=case_variant))

    assert target.read_bytes() == before


def test_cli_refuses_a_previous_sheet_as_its_own_output(tmp_path: Path, monkeypatch) -> None:
    inputs = _inputs(tmp_path)
    previous = _write(tmp_path / "previous.json", _sheet(tmp_path))
    before = previous.read_bytes()
    verdicts = _write(tmp_path / "verdicts.json", [])

    with pytest.raises(SystemExit):
        _run(monkeypatch, _argv(tmp_path, inputs, out=tmp_path / "cases.json", verdicts=verdicts,
                                sheet_out=previous, checked_at=STAMP, previous_sheet=previous))

    assert previous.read_bytes() == before


def test_cli_refuses_one_path_for_both_outputs(tmp_path: Path, monkeypatch) -> None:
    inputs = _inputs(tmp_path)
    verdicts = _write(tmp_path / "verdicts.json", _verdict_rows(tmp_path))
    same = tmp_path / "same.json"

    with pytest.raises(SystemExit):
        _run(monkeypatch, _argv(tmp_path, inputs, out=same, verdicts=verdicts, sheet_out=same, checked_at=STAMP))

    assert not same.exists()

    with pytest.raises(SystemExit):
        _run(monkeypatch, _argv(tmp_path, inputs, out=tmp_path / "result.json", verdicts=verdicts,
                                sheet_out=tmp_path / "RESULT.JSON", checked_at=STAMP))

    same.write_text("old", encoding="utf-8")
    hardlink = tmp_path / "hardlink.json"
    os.link(same, hardlink)
    with pytest.raises(SystemExit):
        _run(monkeypatch, _argv(tmp_path, inputs, out=same, verdicts=verdicts,
                                sheet_out=hardlink, checked_at=STAMP))
    assert same.read_text(encoding="utf-8") == "old"


def test_cli_reports_a_missing_input_before_path_comparison(tmp_path: Path, monkeypatch) -> None:
    inputs = _inputs(tmp_path)
    inputs["amo_events_path"] = tmp_path / "missing-events.jsonl"

    with pytest.raises(SystemExit):
        _run(monkeypatch, _argv(tmp_path, inputs, out=tmp_path / "out.json"))


@pytest.mark.parametrize("extra", [
    {"checked_at": STAMP},
    {"verdicts": "verdicts.json"},
    {"verdicts": "verdicts.json", "checked_at": STAMP},
    {"verdicts": "verdicts.json", "sheet_out": "sheet.json"},
    {"previous_sheet": "previous.json"},
])
def test_cli_refuses_a_half_configured_sheet(tmp_path: Path, monkeypatch, extra: dict) -> None:
    inputs = _inputs(tmp_path)
    _write(tmp_path / "verdicts.json", _verdict_rows(tmp_path))
    _write(tmp_path / "previous.json", _sheet(tmp_path))
    resolved = {name: (tmp_path / value if str(value).endswith(".json") else value)
                for name, value in extra.items()}

    with pytest.raises(SystemExit):
        _run(monkeypatch, _argv(tmp_path, inputs, out=tmp_path / "cases.json", **resolved))


def test_bad_verdict_writes_a_quarantined_row_and_keeps_the_good_row(tmp_path: Path, monkeypatch, capsys) -> None:
    inputs = _inputs(tmp_path)
    rows = _verdict_rows(tmp_path)
    rows[0]["verdict"]["status"] = "ГОТОВО"
    verdicts = _write(tmp_path / "bad-verdicts.json", rows)
    out, sheet_out = tmp_path / "out" / "cases.json", tmp_path / "out" / "sheet.json"

    assert _run(monkeypatch, _argv(tmp_path, inputs, out=out, verdicts=verdicts,
                                  sheet_out=sheet_out, checked_at=STAMP)) == 0

    sheet = json.loads(sheet_out.read_text(encoding="utf-8"))
    assert out.exists() and sheet["row_count"] == 2
    assert sheet["analysis_errors"]["by_key"] == {"p1:c1": "VERDICT_INVALID"}
    assert sheet["rows"][0]["values"]["A:W"][1] == "ПРОВЕРКА"
    assert sheet["rows"][1]["analysis_error"] is None
    assert "errors=1" in capsys.readouterr().out
    assert list(tmp_path.glob("**/*.tmp")) == []


# --------------------------------------------------------------------------- границы модуля


def test_module_imports_nothing_that_could_guess_or_call_out() -> None:
    tree = ast.parse(inspect.getsource(module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise AssertionError("relative import in a standalone CLI")
            imported.add((node.module or "").split(".")[0])

    assert imported == {
        "__future__", "argparse", "collections", "datetime", "hashlib", "json", "os", "pathlib", "typing", "zoneinfo",
    }
    forbidden = {
        "re", "regex", "requests", "urllib", "urllib3", "http", "httpx", "socket", "aiohttp", "ssl",
        "googleapiclient", "gspread", "google", "subprocess", "openpyxl", "xlsxwriter", "mango_mvp", "scripts",
    }
    assert not imported & forbidden
    forbidden_calls = {"eval", "exec", "compile", "__import__"}
    assert not {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)} & forbidden_calls


def test_column_letter_arithmetic_is_gone() -> None:
    assert not hasattr(module, "_column_letter")
    assert not hasattr(module, "_write_ranges")
    assert module.WRITE_RANGES == ("A:W", "Z:AB")
    assert module.WRITE_SLICES == {"A:W": slice(0, 23), "Z:AB": slice(25, 28)}
