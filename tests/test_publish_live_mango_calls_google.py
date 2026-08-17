from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.services import dialogue_contract as contract
from scripts import publish_live_mango_calls_google as publisher
from tests import mango_provider_fixture as fx


def record(**overrides):
    explicit_recording_id = "source_recording_id" in overrides
    payload = {
        "id": 7,
        "source_call_id": "call-7",
        "source_recording_id": fx.RECORDING_ID,
        "started_at": "2026-08-14 09:03:04",
        "phone": "+70000000000",
        "manager_name": "mango_manager_1",
        "direction": "outbound",
        "duration_sec": 142.5,
        "analysis_json": json.dumps(
            {
                "analysis_schema_version": "v3",
                "quality_flags": {"call_type": "sales_call"},
                "target_product": "Математика",
                "history_summary": "Обсудили обучение.",
                "follow_up_reason": "Информация предоставлена",
                "structured_fields": {
                    "objections": ["Цена"],
                    "next_step": {"action": "Перезвонить", "due": "завтра"},
                },
                "review_reasons": [],
            },
            ensure_ascii=False,
        ),
        "transcript_variants_json": json.dumps(
            {
                "role_mapping": {"left": "manager", "right": "client"},
                "dialogue_lines": [
                    "[00:01.0] Дорожка левая: Добрый день",
                    "[00:02.0] Дорожка левая: представлюсь",
                    "[00:03.0] Дорожка правая: Здравствуйте",
                ],
            },
            ensure_ascii=False,
        ),
        "transcript_text": "",
        "analysis_status": "done",
        "sync_status": "pending",
    }
    payload.update(overrides)
    if not explicit_recording_id and payload["source_call_id"] != "call-7":
        payload["source_recording_id"] = f"recording-{payload['source_call_id']}"
    return payload


def projected(**overrides):
    return publisher.call_projection(record(**overrides), {"mango_manager_1": "Иван Иванов"})


# --- A call whose sides Mango itself proved ---------------------------------
# The stored dialogue of ``record()`` line for line, on the same physical sides.
TRUSTED_TURNS = (
    ("operator", "left", "Расскажу про математику"),
    ("client", "right", "Меня интересует математика, беспокоит цена"),
    ("client", "right", "Перезвоните клиенту завтра"),
)


def trusted_variants():
    """``record()``'s own dialogue plus the provider answer that proves it."""
    variants = fx.proven_variants(TRUSTED_TURNS)
    variants["dialogue_lines"] = [
        "[00:01.0] Дорожка левая: Расскажу про математику",
        "[00:02.0] Дорожка правая: Меня интересует математика, беспокоит цена",
        "[00:03.0] Дорожка правая: Перезвоните клиенту завтра",
    ]
    return variants


def current_v3_analysis(raw, *, result=None):
    dialogue = contract.build_dialogue_input(raw)
    fields = {
        "result": result or {"status": None, "detail": None},
        "people": {"parent_fio": None, "child_fio": None},
        "contacts": {"email": None, "preferred_channel": None, "phone_from_filename": None},
        "student": {"grade_current": None, "school": None},
        "interests": {
            "products": [], "format": [], "subjects": ["математика"], "exam_targets": [],
        },
        "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
        "objections": ["цена"],
        "next_step": {"action": "Перезвонить клиенту", "due": "завтра"},
        "lead_priority": "warm",
    }
    claim_specs = [
        ("structured_fields.interests.subjects", "математика", "T0002", True),
        ("structured_fields.objections", "цена", "T0002", True),
        ("structured_fields.next_step.action", "Перезвонить клиенту", "T0003", False),
        ("structured_fields.next_step.due", "завтра", "T0003", False),
    ]
    if fields["result"]["status"]:
        claim_specs.extend(
            [
                ("structured_fields.result.status", fields["result"]["status"], "T0001", False),
                ("structured_fields.result.detail", fields["result"]["detail"], "T0001", False),
            ]
        )
    turns = {turn["turn_id"]: turn for turn in dialogue.turns}
    call_key = contract.call_key_for_record(raw)
    evidence = []
    for field_path, value, turn_id, listed in claim_specs:
        turn = turns[turn_id]
        item_id = contract.canonical_item_key(value) if listed else None
        digest = contract.value_sha256(value)
        evidence.append(
            {
                "claim_id": contract.deterministic_claim_id(
                    call_key=call_key, field_path=field_path,
                    item_key=str(item_id or ""), digest=digest,
                    contract_version=contract.CLAIM_CONTRACT_VERSION,
                ),
                "field_path": field_path, "item_id": item_id,
                "evidence_type": "explicit", "support_type": "explicit",
                "source": "model_claim", "contract_version": contract.CLAIM_CONTRACT_VERSION,
                "turn_id": turn_id, "exact_quote": turn["text"],
                "timecode": turn["timecode"], "speaker_kind": turn["speaker_kind"],
                "start_sec": turn["start_sec"], "dialogue_sha256": dialogue.canonical_sha256,
                "raw_value": value, "value_sha256": digest, "validation_status": "valid",
            }
        )
    input_sha = "a" * 64
    display = json.loads(json.dumps(fields, ensure_ascii=False))
    payload = {
        "analysis_schema_version": contract.ANALYSIS_SCHEMA_VERSION_V3,
        "claim_contract_version": contract.CLAIM_CONTRACT_VERSION,
        "history_summary": "Обсудили обучение.",
        "follow_up_reason": "Информация предоставлена",
        "structured_fields": fields, "display_fields": display, "crm_blocks": display,
        "claim_evidence": evidence, "normalized_facts": [],
        "dialogue_input": {
            "version": contract.CONTRACT_VERSION, "source": dialogue.source,
            "canonical_sha256": dialogue.canonical_sha256, "turn_count": len(dialogue.turns),
        },
        "quality_flags": {
            "call_type": "sales_call", "needs_review": False, "review_reasons": [],
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
        "review_reasons": [], "needs_review": False,
    }
    payload["analysis_meta"]["manager_output_sha256"] = (
        contract.manager_output_sha256(payload)
    )
    return payload


def with_current_output_hash(analysis):
    analysis["analysis_meta"]["manager_output_sha256"] = (
        contract.manager_output_sha256(analysis)
    )
    return analysis


def test_analysis_cost_summary_never_estimates_missing_provider_usage():
    summary = publisher.analysis_cost_summary(
        {
            "exact": {
                "analysis_done": True,
                "model_call_count": 1,
                "cache_hit_count": 0,
                "token_usage": {
                    "source": "provider_exact",
                    "prompt_tokens": 120,
                    "completion_tokens": 30,
                    "total_tokens": 150,
                },
            },
            "partial": {
                "analysis_done": True,
                "analysis_provider": "ollama",
                "analysis_model": "local",
                "analysis_prompt_version": "v8",
                "model_call_count": 1,
                "cache_hit_count": 0,
                "token_usage": {
                    "source": "provider_partial",
                    "prompt_tokens": 80,
                    "completion_tokens": 20,
                    "total_tokens": None,
                },
            },
            "legacy_partial": {
                "analysis_done": True,
                "analysis_provider": "legacy",
                "analysis_model": "legacy",
                "analysis_prompt_version": "v7",
                "model_call_count": 1,
                "cache_hit_count": 0,
                "token_usage": {
                    "source": "provider",
                    "prompt_tokens": 40,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
            },
            "unknown": {
                "analysis_done": True,
                "model_call_count": 2,
                "cache_hit_count": 0,
                "token_usage": {"source": "unavailable"},
            },
            "cache": {
                "analysis_done": True,
                "model_call_count": 0,
                "cache_hit_count": 1,
                "token_usage": {"source": "cache_hit"},
            },
            "skipped": {
                "analysis_done": True,
                "model_call_count": 0,
                "cache_hit_count": 0,
                "token_usage": {"source": "skipped_untrusted_role"},
            },
            "deterministic": {
                "analysis_done": True,
                "model_call_count": 0,
                "cache_hit_count": 0,
                "token_usage": {"source": "skipped_deterministic"},
            },
            "pending": {"analysis_done": False},
        }
    )

    assert summary == {
        "analysis_done": 7,
        "failed_analysis_calls": 0,
        "cost_tracked_calls": 7,
        "model_calls": 5,
        "cache_hits": 1,
        "invalid_attempt_ledger_calls": 0,
        "indeterminate_attempts": 0,
        "provider_usage_calls": 3,
        "provider_exact_calls": 1,
        "provider_partial_calls": 2,
        "usage_unavailable_calls": 1,
        "exact_usage_model_calls": 1,
        "partial_usage_model_calls": 2,
        "unavailable_usage_model_calls": 2,
        "cache_only_calls": 1,
        "skipped_untrusted_role_calls": 1,
        "skipped_deterministic_calls": 1,
        "prompt_tokens": 240,
        "completion_tokens": 50,
        "total_tokens": 150,
        "by_provider_model_prompt": {
            "unknown|unknown|unknown": {
                "model_calls": 3,
                "exact_usage_model_calls": 1,
                "partial_usage_model_calls": 0,
                "unavailable_usage_model_calls": 2,
            },
            "ollama|local|v8": {
                "model_calls": 1,
                "exact_usage_model_calls": 0,
                "partial_usage_model_calls": 1,
                "unavailable_usage_model_calls": 0,
            },
            "legacy|legacy|v7": {
                "model_calls": 1,
                "exact_usage_model_calls": 0,
                "partial_usage_model_calls": 1,
                "unavailable_usage_model_calls": 0,
            },
        },
        "classified_calls": 7,
        "model_usage_balanced": True,
        "identity_balanced": True,
        "balanced": True,
    }


def test_analysis_cost_summary_classifies_each_model_attempt_separately():
    summary = publisher.analysis_cost_summary(
        {
            "mixed": {
                "analysis_done": True,
                "analysis_provider": "openai",
                "analysis_model": "gpt-test",
                "analysis_prompt_version": "full-v1",
                "model_call_count": 2,
                "cache_hit_count": 0,
                "model_attempts": [
                    {
                        "provider": "openai",
                        "model": "gpt-compact",
                        "model_called": True,
                        "prompt_version": "compact-v1",
                        "token_usage": {
                            "source": "provider_exact",
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                        },
                    },
                    {
                        "provider": "codex_cli",
                        "model": "gpt-full",
                        "model_called": True,
                        "prompt_version": "full-v1",
                        "token_usage": {
                            "source": "provider_partial",
                            "prompt_tokens": 200,
                            "completion_tokens": None,
                            "total_tokens": None,
                        },
                    },
                ],
                "token_usage": {
                    "source": "provider_partial",
                    "prompt_tokens": 300,
                    "completion_tokens": 20,
                    "total_tokens": None,
                },
            }
        }
    )

    assert summary["model_calls"] == 2
    assert summary["exact_usage_model_calls"] == 1
    assert summary["partial_usage_model_calls"] == 1
    assert summary["unavailable_usage_model_calls"] == 0
    assert summary["prompt_tokens"] == 300
    assert summary["completion_tokens"] == 20
    assert summary["total_tokens"] == 120
    assert summary["model_usage_balanced"] is True
    assert set(summary["by_provider_model_prompt"]) == {
        "openai|gpt-compact|compact-v1",
        "codex_cli|gpt-full|full-v1",
    }


def test_provider_exact_without_all_counters_is_partial_not_exact():
    summary = publisher.analysis_cost_summary(
        {
            "broken": {
                "analysis_done": True,
                "model_call_count": 1,
                "model_attempts": [
                    {
                        "model_called": True,
                        "token_usage": {
                            "source": "provider_exact",
                            "prompt_tokens": 10,
                            "completion_tokens": None,
                            "total_tokens": None,
                        },
                    }
                ],
                "token_usage": {
                    "source": "provider_exact",
                    "prompt_tokens": 10,
                    "completion_tokens": None,
                    "total_tokens": None,
                },
            }
        }
    )

    assert summary["provider_exact_calls"] == 0
    assert summary["provider_partial_calls"] == 1
    assert summary["exact_usage_model_calls"] == 0
    assert summary["partial_usage_model_calls"] == 1


def test_failed_call_keeps_every_model_attempt_in_the_cost_balance(tmp_path):
    attempts = [
        {
            "attempt_id": f"attempt-{index}",
            "state": "failed",
            "provider": "codex_cli",
            "model": "gpt-test",
            "profile": "compact",
            "prompt_version": "v8",
            "model_called": True,
            "cache_hit": False,
            "token_usage": {
                "source": "unavailable",
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        }
        for index in range(5)
    ]
    db_path = _sqlite(
        tmp_path,
        [
            trusted_record(
                analysis_status="failed",
                analysis_json=None,
                analysis_attempts_json=json.dumps(attempts),
            )
        ],
    )

    calls, identities, errors = publisher.load_calls(db_path, {})
    summary = publisher.analysis_cost_summary(calls)

    assert len(identities) == 1
    assert errors == {}
    assert summary["analysis_done"] == 0
    assert summary["failed_analysis_calls"] == 1
    assert summary["cost_tracked_calls"] == 1
    assert summary["model_calls"] == 5
    assert summary["unavailable_usage_model_calls"] == 5
    assert summary["balanced"] is True


def trusted_record(**overrides):
    updates = dict(overrides)
    analysis_json = updates.pop("analysis_json", None)
    payload = record(transcript_variants_json=json.dumps(trusted_variants(), ensure_ascii=False))
    payload.update(updates)
    payload["analysis_json"] = analysis_json or json.dumps(
        current_v3_analysis(payload), ensure_ascii=False
    )
    return payload


def trusted_projected(**overrides):
    return publisher.call_projection(
        trusted_record(**overrides), {"mango_manager_1": "Иван Иванов"}
    )


def test_database_attempt_ledger_overrides_stale_analysis_meta():
    attempts = [
        {
            "attempt_id": f"attempt-{index}", "stage": "analyze",
            "state": "completed", "provider": "codex_cli", "model": "gpt-test",
            "profile": "compact", "prompt_version": "v8", "model_called": True,
            "cache_hit": False,
            "token_usage": {
                "source": "provider_exact", "prompt_tokens": 10 * index,
                "completion_tokens": index, "total_tokens": 11 * index,
            },
        }
        for index in (1, 2)
    ]
    raw = trusted_record(analysis_attempts_json=json.dumps(attempts))
    analysis = json.loads(raw["analysis_json"])
    analysis["analysis_meta"].update(
        {
            "model_call_count": 1,
            "model_attempts": [attempts[0]],
            "token_usage": attempts[0]["token_usage"],
        }
    )
    raw["analysis_json"] = json.dumps(analysis)

    call = publisher.call_projection(raw, {})
    summary = publisher.analysis_cost_summary({call["call_key"]: call})

    assert call["attempt_ledger_source"] == "database"
    assert call["model_call_count"] == 2
    assert len(call["model_attempts"]) == 2
    assert summary["total_tokens"] == 33
    assert summary["balanced"] is True


def test_empty_database_ledger_cannot_hide_model_work_reported_by_analysis():
    raw = trusted_record(analysis_attempts_json="[]")
    analysis = json.loads(raw["analysis_json"])
    analysis["analysis_meta"].update(
        {
            "model_call_count": 1,
            "model_attempts": [
                {
                    "provider": "codex_cli",
                    "model": "gpt-test",
                    "model_called": True,
                    "cache_hit": False,
                    "token_usage": {"source": "unavailable"},
                }
            ],
        }
    )
    raw["analysis_json"] = json.dumps(analysis, ensure_ascii=False)

    call = publisher.call_projection(raw, {})
    summary = publisher.analysis_cost_summary({call["call_key"]: call})

    assert call["attempt_ledger_source"] == "database"
    assert call["attempt_ledger_valid"] is False
    assert summary["balanced"] is False
    with pytest.raises(RuntimeError, match="cost ledger does not close"):
        publisher.require_closed_analysis_cost(summary)


def test_current_database_ledger_requires_attempt_ids_and_states():
    attempts = [
        {
            "provider": "codex_cli", "model": "gpt-test", "profile": "compact",
            "prompt_version": "v8", "model_called": True, "cache_hit": False,
            "token_usage": {"source": "unavailable"},
        }
    ]
    call = trusted_projected(analysis_attempts_json=json.dumps(attempts))
    summary = publisher.analysis_cost_summary({call["call_key"]: call})

    assert call["attempt_ledger_source"] == "database"
    assert call["attempt_ledger_valid"] is False
    assert summary["balanced"] is False
    with pytest.raises(RuntimeError, match="cost ledger does not close"):
        publisher.require_closed_analysis_cost(summary)


@pytest.mark.parametrize(
    ("state", "second_id"),
    [("reserved", "other-id"), ("indeterminate", "other-id"), ("completed", "same-id")],
)
def test_open_or_duplicate_attempt_ledger_blocks_publication_cost(state, second_id):
    attempts = [
        {
            "attempt_id": "same-id", "stage": "analyze", "state": state,
            "provider": "codex_cli", "model": "gpt-test", "profile": "compact",
            "prompt_version": "v8", "model_called": True, "cache_hit": False,
            "token_usage": {"source": "unavailable"},
        },
        {
            "attempt_id": second_id, "stage": "analyze", "state": "completed",
            "provider": "codex_cli", "model": "gpt-test", "profile": "compact",
            "prompt_version": "v8", "model_called": True, "cache_hit": False,
            "token_usage": {"source": "unavailable"},
        },
    ]
    call = trusted_projected(analysis_attempts_json=json.dumps(attempts))
    summary = publisher.analysis_cost_summary({call["call_key"]: call})

    assert call["attempt_ledger_valid"] is False
    assert summary["balanced"] is False
    with pytest.raises(RuntimeError, match="cost ledger does not close"):
        publisher.require_closed_analysis_cost(summary)


def with_number(call, number=10):
    return publisher.desired_row(call, number)


def state_for(call, row, *, status="verified"):
    return {
        "schema_version": publisher.STATE_SCHEMA,
        "destination_id": "dest",
        "incidents": {},
        "entries": {
            call["call_key"]: {
                "display_number": int(row[0]),
                "status": status,
                "projection_version": publisher.PROJECTION_VERSION,
                "source_fingerprint": call["source_fingerprint"],
                "started_epoch": call["started_epoch"],
                "planned_row_sha256": (
                    publisher.physical_row_hash(row) if status == "reserved" else None
                ),
                "last_verified_row_sha256": publisher.physical_row_hash(row),
                "attempts": 1,
            }
        },
    }


def _owner_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.chmod(path, 0o600)


def _destination(sheet_id=0):
    return f"google_sheets:v1:spreadsheet:{sheet_id}:mango:production"


def _sqlite(tmp_path: Path, records):
    db_path = tmp_path / "calls.sqlite"
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE call_records ("
        "id INTEGER PRIMARY KEY, source_call_id TEXT, source_recording_id TEXT, started_at TEXT, phone TEXT, "
        "manager_name TEXT, direction TEXT, duration_sec REAL, analysis_json TEXT, "
        "analysis_attempts_json TEXT, "
        "transcript_variants_json TEXT, transcript_text TEXT, analysis_status TEXT, "
        "sync_status TEXT, sync_attempts INTEGER NOT NULL DEFAULT 0)"
    )
    columns = (
        "id", "source_call_id", "source_recording_id", "started_at", "phone", "manager_name", "direction",
        "duration_sec", "analysis_json", "analysis_attempts_json", "transcript_variants_json", "transcript_text",
        "analysis_status", "sync_status",
    )
    connection.executemany(
        f"INSERT INTO call_records ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        [tuple(item.get(column) for column in columns) for item in records],
    )
    connection.commit()
    connection.close()
    return db_path


def _decode_cell(cell):
    value = cell.get("userEnteredValue") or {}
    for field in ("stringValue", "numberValue", "boolValue"):
        if field in value:
            return value[field]
    return ""


class FakeLiveGoogleGateway:
    def __init__(self, rows=(), *, title="Звонки", sheet_id=0):
        self.rows = [list(row) for row in rows]
        self.row_heights = [publisher.row_height(row[9]) for row in self.rows]
        self.title = title
        self.sheet_id = sheet_id
        self.batch_calls = 0
        self.values_calls = 0
        self.raise_after_apply = False
        self.on_batch_applied = None
        self.before_values = None

    def sheets(self):
        return [{"title": self.title, "sheetId": self.sheet_id}]

    def values(self, _title):
        self.values_calls += 1
        if self.before_values is not None:
            callback, self.before_values = self.before_values, None
            callback()
        return [list(publisher.LIVE_HEADERS), *[list(row) for row in self.rows]]

    def batch_sheet_requests(self, requests):
        self.batch_calls += 1
        q_values = None
        for request in requests:
            if "updateCells" in request:
                update = request["updateCells"]
                target = update.get("range")
                if target and int(target.get("startColumnIndex", 0)) == 0:
                    start = int(target["startRowIndex"]) - 1
                    decoded = [
                        [_decode_cell(cell) for cell in row.get("values") or ()]
                        for row in update.get("rows") or ()
                    ]
                    self.rows[start:start + len(decoded)] = decoded
                elif int((update.get("start") or {}).get("columnIndex", -1)) == publisher.SORT_KEY_COLUMN_INDEX:
                    q_values = [
                        _decode_cell((row.get("values") or [{}])[0])
                        for row in update.get("rows") or ()
                    ]
            elif "appendCells" in request:
                appended = [
                    [_decode_cell(cell) for cell in row.get("values") or ()]
                    for row in request["appendCells"].get("rows") or ()
                ]
                self.rows.extend(appended)
                self.row_heights.extend([21] * len(appended))
            elif "sortRange" in request:
                assert q_values is not None and len(q_values) == len(self.rows)
                ordered = sorted(zip(q_values, self.rows), key=lambda item: item[0], reverse=True)
                self.rows = [row for _key, row in ordered]
            elif "updateDimensionProperties" in request:
                update = request["updateDimensionProperties"]
                target = update.get("range") or {}
                if target.get("dimension") != "ROWS":
                    continue
                start = int(target["startIndex"]) - 1
                end = int(target["endIndex"]) - 1
                size = int((update.get("properties") or {})["pixelSize"])
                while len(self.row_heights) < end:
                    self.row_heights.append(21)
                self.row_heights[start:end] = [size] * (end - start)
        if self.on_batch_applied is not None:
            self.on_batch_applied(self)
        if self.raise_after_apply:
            self.raise_after_apply = False
            raise TimeoutError("simulated lost Google response")

    def layout(self, _title, _last_row):
        column_metadata = [{} for _ in range(10)]
        column_metadata[9] = {"pixelSize": 320}
        return {
            "sheets": [{
                "data": [{
                    "columnMetadata": column_metadata,
                    "rowMetadata": [
                        {"pixelSize": size} for size in self.row_heights
                    ],
                    "rowData": [
                        {
                            "values": [
                                {
                                    "userEnteredFormat": (
                                        {"wrapStrategy": "WRAP", "verticalAlignment": "TOP"}
                                        if index == 9
                                        else {"wrapStrategy": "CLIP", "verticalAlignment": "TOP"}
                                        if index == publisher.TRANSCRIPT_COLUMN_INDEX
                                        else {}
                                    )
                                }
                                for index in range(publisher.MANAGED_COLUMN_COUNT)
                            ]
                        }
                        for _row in self.rows
                    ],
                }]
            }]
        }


def _harness(tmp_path, monkeypatch, *, records, rows=(), state=None, fake=None, sheet_id=0):
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    os.chmod(private, 0o700)
    db_path = _sqlite(private, records)
    credentials = private / "credentials.json"
    manager = private / "manager.json"
    state_path = private / "state.json"
    lock = private / "publisher.lock"
    config = private / "config.json"
    _owner_json(credentials, {"client_email": "publisher@example.invalid"})
    _owner_json(manager, {"mapping": {"mango_manager_1": "Иван Иванов"}})
    if state is not None:
        _owner_json(state_path, state)
    _owner_json(
        config,
        {
            "schema_version": publisher.CONFIG_SCHEMA,
            "spreadsheet_id": "spreadsheet",
            "sheet_id": sheet_id,
            "sheet_title": "Звонки",
            "working_db": str(db_path),
            "manager_identity": str(manager),
            "credentials": str(credentials),
            "state": str(state_path),
            "lock": str(lock),
            "summary_width_px": 320,
            "batch_limit": 25,
            "expected_code_sha": "a" * 40,
        },
    )
    fake = fake or FakeLiveGoogleGateway(rows, sheet_id=sheet_id)
    monkeypatch.setattr(publisher, "authorized_session", lambda _info: object())
    monkeypatch.setattr(
        publisher, "LiveGoogleGateway", lambda _session, _spreadsheet_id: fake
    )
    monkeypatch.setattr(
        publisher.subprocess,
        "check_output",
        lambda command, text=True: "a" * 40 + "\n" if command[-1] == "HEAD" else "",
    )
    return {
        "config": config,
        "db": db_path,
        "state": state_path,
        "fake": fake,
        "private": private,
    }


def _run_execute(config, *extra):
    return publisher.run(
        [
            "--config", str(config), "--execute", "--confirmation", publisher.CONFIRMATION,
            *extra,
        ]
    )


def _db_status(db_path, call_id=7):
    connection = sqlite3.connect(db_path)
    result = connection.execute(
        "SELECT sync_status,sync_attempts FROM call_records WHERE id=?", (call_id,)
    ).fetchone()
    connection.close()
    return result


def test_live_headers_are_exact_production_contract():
    assert publisher.LIVE_HEADERS == (
        "№", "Дата и время (МСК)", "Менеджер", "Направление", "Длительность",
        "Категория", "Телефон клиента", "Нужна проверка", "Тема",
        "Конспект разговора", "Результат", "Возражение / причина",
        "Следующий шаг", "Срок", "Основание ключевых выводов",
        "Что проверить РОПу", "Полная расшифровка",
    )


def test_repository_launchd_template_is_shadow_only():
    template = (
        Path(__file__).resolve().parents[1]
        / "deploy/mango_calls_live_publisher/com.mango.calls.live.publisher.plist.template"
    ).read_text(encoding="utf-8")

    assert "--config" in template
    assert "--execute" not in template
    assert "PUBLISH_MANGO_CALLS_LIVE" not in template
    assert "StartInterval" not in template


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "0 мин 0 с"), (6.5, "0 мин 7 с"), (142.5, "2 мин 23 с"), (179.49, "2 мин 59 с")],
)
def test_duration_is_half_up_and_human_readable(seconds, expected):
    assert publisher.format_duration(seconds) == expected


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_duration_fails_closed(value):
    with pytest.raises(ValueError):
        publisher.format_duration(value)


def test_projection_converts_naive_utc_to_moscow_once_and_keeps_full_transcript():
    call = projected()
    row = with_number(call)
    assert row[1] == "2026-08-14 12:03:04"
    assert row[4] == "2 мин 23 с"
    # A text-derived role mapping is not proof: physical speakers only.  And one
    # stored line stays one published line, with its own timecode.
    assert row[publisher.TRANSCRIPT_COLUMN_INDEX] == (
        "[00:01.0] Спикер A: Добрый день\n"
        "[00:02.0] Спикер A: представлюсь\n"
        "[00:03.0] Спикер B: Здравствуйте"
    )


def test_a_proven_call_publishes_the_named_sides():
    row = with_number(trusted_projected())
    assert row[publisher.TRANSCRIPT_COLUMN_INDEX] == (
        "[00:01.0] Менеджер: Расскажу про математику\n"
        "[00:02.0] Клиент: Меня интересует математика, беспокоит цена\n"
        "[00:03.0] Клиент: Перезвоните клиенту завтра"
    )
    evidence = row[publisher.LIVE_HEADERS.index("Основание ключевых выводов")]
    assert "Возражение или причина" in evidence
    assert "T0002 [00:02.0]" in evidence
    assert "беспокоит цена" in evidence


def test_projection_converts_only_generated_summary_prefix_to_moscow():
    analysis = json.loads(trusted_record()["analysis_json"])
    analysis["history_summary"] = "14.08.2026 09:03 — Менеджер: обсудили занятие в 15:00"
    row = with_number(
        trusted_projected(
            analysis_json=json.dumps(with_current_output_hash(analysis), ensure_ascii=False)
        )
    )
    assert row[9] == "14.08.2026 12:03 — Менеджер: обсудили занятие в 15:00"


def test_projection_does_not_shift_non_generated_time_inside_summary():
    analysis = json.loads(trusted_record()["analysis_json"])
    analysis["history_summary"] = "Договорились созвониться в 15:00"
    row = with_number(
        trusted_projected(
            analysis_json=json.dumps(with_current_output_hash(analysis), ensure_ascii=False)
        )
    )
    assert row[9] == "Договорились созвониться в 15:00"


def test_projection_uses_one_canonical_summary_field_for_google():
    analysis = json.loads(trusted_record()["analysis_json"])
    analysis["summary"] = "Краткий конспект для рабочей таблицы"
    analysis["history_summary"] = "14.08.2026 09:03 — служебная память"
    row = with_number(
        trusted_projected(
            analysis_json=json.dumps(with_current_output_hash(analysis), ensure_ascii=False)
        )
    )
    assert row[9] == "Краткий конспект для рабочей таблицы"


def test_projection_prefers_the_nonduplicating_manager_brief():
    analysis = json.loads(trusted_record()["analysis_json"])
    analysis["manager_brief"] = "Обращение по ученику 8 класса. Семья сравнивает варианты."
    analysis["summary"] = "Тема; результат; возражение; следующий шаг; срок."
    row = with_number(
        trusted_projected(
            analysis_json=json.dumps(with_current_output_hash(analysis), ensure_ascii=False)
        )
    )

    assert row[9] == "Обращение по ученику 8 класса. Семья сравнивает варианты."


# --- Этап G: an old analysis_json goes through the same fail-closed gate ----


def test_an_old_analysis_of_an_unproven_call_publishes_no_role_dependent_claim():
    """The stored payload predates the role guard; reading it is not trusting it."""
    row = with_number(projected())

    assert row[7] == "Да"                       # Нужна проверка
    assert row[9] == contract.UNTRUSTED_SUMMARY  # Конспект
    assert row[11] == "—"                        # Возражение / причина
    assert row[12] == "—"                        # Следующий шаг
    assert row[13] == "—"                        # Срок
    assert "Перезвонить" not in json.dumps(row, ensure_ascii=False)
    assert "Математика" not in json.dumps(row, ensure_ascii=False)


def test_a_v2_unproven_call_is_visible_as_neutral_instead_of_quarantined():
    analysis = json.loads(record()["analysis_json"])
    analysis["analysis_schema_version"] = "v2"
    analysis["history_summary"] = "Клиент оплатил курс, нужно списать деньги"

    projected_call = publisher.call_projection(
        record(analysis_json=json.dumps(analysis, ensure_ascii=False)), {}
    )
    row = with_number(projected_call)

    assert projected_call["analysis_done"] is True
    assert row[9] == contract.UNTRUSTED_SUMMARY
    assert row[10:14] == ["—", "—", "—", "—"]
    assert "оплатил" not in json.dumps(row, ensure_ascii=False).lower()


def test_a_proven_call_keeps_its_next_step_and_says_no_review_is_needed():
    row = with_number(trusted_projected())

    assert row[7] == "Нет"
    assert row[8] == "математика"
    assert row[11] == "цена"
    assert row[12] == "Перезвонить клиенту"
    assert row[13] == "завтра"


def test_projection_never_publishes_a_due_without_an_action():
    raw = trusted_record()
    analysis = json.loads(raw["analysis_json"])
    for block_name in ("structured_fields", "display_fields", "crm_blocks"):
        analysis[block_name]["next_step"]["action"] = None
    analysis["next_step"] = None
    analysis["claim_evidence"] = [
        item
        for item in analysis["claim_evidence"]
        if item["field_path"] != "structured_fields.next_step.action"
    ]
    raw["analysis_json"] = json.dumps(
        with_current_output_hash(analysis), ensure_ascii=False
    )

    row = with_number(publisher.call_projection(raw, {}))

    assert row[12] == "—"
    assert row[13] == "—"


def test_result_uses_closed_structured_status_and_detail_not_follow_up_reason():
    raw = trusted_record()
    analysis = current_v3_analysis(
        raw,
        result={"status": "information_only", "detail": "математику"},
    )
    analysis["follow_up_reason"] = "Клиент отказался"
    raw["analysis_json"] = json.dumps(
        with_current_output_hash(analysis), ensure_ascii=False
    )
    row = with_number(publisher.call_projection(raw, {}))

    assert row[10] == "Получена или уточнена информация: математику"
    assert "отказ" not in row[10].lower()

    assert publisher.result_text_ru(
        {
            "structured_fields": {"result": {"status": "invented_status"}},
            "display_fields": {"result": {"detail": "Не должно публиковаться"}},
        }
    ) == "—"


def test_the_review_column_is_a_russian_sentence_and_never_a_raw_code():
    untrusted = with_number(projected())[publisher.REVIEW_COLUMN_INDEX]
    analysis = json.loads(trusted_record()["analysis_json"])
    analysis["review_reasons"] = ["sales_missing_next_step", "some_future_code"]
    coded = with_number(
        trusted_projected(
            analysis_json=json.dumps(with_current_output_hash(analysis), ensure_ascii=False)
        )
    )[publisher.REVIEW_COLUMN_INDEX]

    assert "role_attribution_untrusted" not in untrusted
    assert "разметка дорожек не подтверждена Mango" in untrusted
    assert "sales_missing_next_step" not in coded
    assert "some_future_code" not in coded
    assert "в звонке о продаже не подтверждён следующий шаг" in coded
    assert publisher.UNKNOWN_REVIEW_REASON_RU in coded


def test_truncated_analysis_is_visibly_routed_to_review():
    analysis = json.loads(trusted_record()["analysis_json"])
    analysis["needs_review"] = True
    analysis["review_reasons"] = ["analyze_prompt_truncated"]
    analysis["review_reasons_ru"] = contract.review_reasons_ru(
        analysis["review_reasons"]
    )
    analysis["quality_flags"]["needs_review"] = True
    analysis["quality_flags"]["review_reasons"] = ["analyze_prompt_truncated"]
    analysis["quality_flags"]["analyze_prompt_truncated"] = True
    row = with_number(
        trusted_projected(
            analysis_json=json.dumps(with_current_output_hash(analysis), ensure_ascii=False)
        )
    )

    assert row[7] == "Да"
    assert "разговор не поместился в окно анализа целиком" in row[publisher.REVIEW_COLUMN_INDEX]


def test_an_unproven_call_still_gets_a_neutral_topic_from_the_dialogue():
    variants = json.loads(record()["transcript_variants_json"])
    variants["dialogue_lines"] = [
        "[00:01.0] Дорожка левая: Расскажу про подготовку к ЕГЭ по математике",
        "[00:03.0] Дорожка правая: Какая стоимость",
    ]
    row = with_number(
        projected(transcript_variants_json=json.dumps(variants, ensure_ascii=False))
    )

    assert row[8] == "математика; подготовка к ЕГЭ; стоимость и оплата"


def test_transcript_fallback_removes_technical_labels():
    call = projected(
        transcript_variants_json=json.dumps({"full": {"final": "CHANNEL_LEFT: Текст\nsha256: secret"}})
    )
    assert call["tail"][-1] == "[00:00.0] Не определено: Текст"


def test_projection_rejects_an_empty_transcript():
    with pytest.raises(ValueError, match="empty"):
        projected(transcript_variants_json="{}", transcript_text="")


def oversized_variants(reply_chars=6_000, replies=20):
    """A conversation whose whole text is far past one Google cell."""
    return {
        "role_mapping": {"left": "manager", "right": "client"},
        "dialogue_lines": [
            f"[00:{index:02d}.0] Дорожка {'левая' if index % 2 else 'правая'}: "
            f"Реплика {index} " + "я" * reply_chars
            for index in range(1, replies + 1)
        ],
    }


def test_a_call_longer_than_one_google_cell_keeps_its_row():
    """Dropping the row was the worst outcome for the longest conversations."""
    call = projected(
        transcript_variants_json=json.dumps(oversized_variants(), ensure_ascii=False)
    )
    row = with_number(call)
    transcript = row[publisher.TRANSCRIPT_COLUMN_INDEX]

    # The row exists, with its metadata and its summary.
    assert row[1] == "2026-08-14 12:03:04"
    assert row[9] == contract.UNTRUSTED_SUMMARY
    assert len(transcript) <= publisher.MAX_CELL_CHARS
    # Whole replies only, the gap is marked, and the note says where the rest is.
    assert transcript.startswith(publisher.TRANSCRIPT_OVERSIZE_NOTE)
    assert contract.TRUNCATION_MARKER in transcript
    # Every published line is a whole reply of the real dialogue: no reply is
    # cut in the middle, and nothing was invented to fill the cell.
    whole = set(
        contract.build_dialogue_input(
            record(
                transcript_variants_json=json.dumps(
                    oversized_variants(), ensure_ascii=False
                )
            )
        )
        .render()
        .splitlines()
    )
    published = transcript.split("\n")[1:]
    assert published
    for line in published:
        assert line == contract.TRUNCATION_MARKER or line in whole
    # Both ends of the conversation survive.
    assert published[0] in whole and published[-1] in whole
    # The reader is told to check the call, with the reason spelled out.
    assert row[7] == "Да"
    assert publisher.TRANSCRIPT_OVERSIZE_REVIEW_RU in row[publisher.REVIEW_COLUMN_INDEX]


def test_a_single_reply_larger_than_the_cell_still_leaves_a_row():
    call = projected(
        transcript_variants_json=json.dumps(
            {"dialogue_lines": ["[00:01.0] Дорожка левая: " + "я" * 60_000]},
            ensure_ascii=False,
        )
    )
    row = with_number(call)

    assert row[publisher.TRANSCRIPT_COLUMN_INDEX] == publisher.TRANSCRIPT_UNRENDERABLE_NOTE
    assert row[7] == "Да"
    assert publisher.TRANSCRIPT_OVERSIZE_REVIEW_RU in row[publisher.REVIEW_COLUMN_INDEX]


def test_identity_matching_hashes_exactly_the_published_cell():
    """A shortened transcript must still find its own Google row."""
    raw = record(
        transcript_variants_json=json.dumps(oversized_variants(), ensure_ascii=False)
    )
    identity = publisher.call_identity(raw)
    published = publisher.call_projection(raw, {})["tail"][-1]

    assert identity["transcript_sha"] == publisher.hashlib.sha256(
        published.encode("utf-8")
    ).hexdigest()


def test_pending_identity_hashes_the_real_transcript_that_is_published():
    raw = record(analysis_status="pending", analysis_json=None)
    published = publisher.call_projection(raw, {})["tail"][-1]
    identity = publisher.call_identity(raw)

    assert published != publisher.SAFE_PENDING_TRANSCRIPT_RU
    assert identity["transcript_sha"] == publisher.hashlib.sha256(
        published.encode("utf-8")
    ).hexdigest()


def test_a_stated_review_reason_can_never_be_published_as_no_review_needed():
    analysis = json.loads(trusted_record()["analysis_json"])
    analysis["needs_review"] = False
    analysis["review_reasons"] = ["sales_missing_next_step"]
    row = with_number(
        trusted_projected(
            analysis_json=json.dumps(with_current_output_hash(analysis), ensure_ascii=False)
        )
    )

    assert row[publisher.REVIEW_COLUMN_INDEX] != "—"
    assert row[7] == "Да"


@pytest.mark.parametrize(
    ("variants", "expected"),
    [
        (
            {"dialogue_lines": ["[00:01.0] Спикер 1: Текст"]},
            "[00:01.0] Не определено: Текст",
        ),
        (
            {
                "role_mapping": {"left": "manager"},
                "dialogue_lines": ["[00:01.0] Дорожка правая: Текст"],
            },
            "[00:01.0] Спикер B: Текст",
        ),
        (
            {
                "mode": "mono_or_fallback",
                "role_mapping": {"status": "unverified_mono_or_legacy"},
                "dialogue_lines": ["[00:01.0] Спикер (не определен): Текст"],
            },
            "[00:01.0] Не определено: Текст",
        ),
    ],
)
def test_projection_keeps_unknown_and_unmapped_roles_instead_of_dropping_them(
    variants, expected
):
    call = projected(transcript_variants_json=json.dumps(variants, ensure_ascii=False))
    assert call["tail"][-1] == expected


def test_legacy_identity_renderer_still_matches_a_row_written_before_the_contract():
    raw = record(
        transcript_variants_json=json.dumps(
            {
                "role_mapping": {"left": "manager", "right": "client"},
                "dialogue_lines": ["[00:01.0] Дорожка левая: Старый текст"],
            },
            ensure_ascii=False,
        )
    )
    identity = publisher.call_identity(raw)
    legacy = "[00:01.0] Менеджер: Старый текст"
    current = "[00:01.0] Спикер A: Старый текст"
    assert publisher.render_legacy_identity_transcript(raw) == legacy
    assert publisher.call_projection(raw, {})["tail"][-1] == current
    assert identity["transcript_sha"] == publisher.hashlib.sha256(
        current.encode("utf-8")
    ).hexdigest()
    assert identity["legacy_transcript_sha"] == publisher.hashlib.sha256(
        legacy.encode("utf-8")
    ).hexdigest()


def test_legacy_fallback_render_is_byte_identical_to_the_pre_contract_one():
    # The old projection stripped only CHANNEL_*/Дорожка prefixes, so a row
    # written from a MANAGER:/CLIENT: dump still carries those words.  The new
    # contract deliberately strips them, and the two must not be confused.
    raw = record(
        transcript_variants_json=json.dumps({"dialogue_lines": []}, ensure_ascii=False),
        transcript_text="CHANNEL_LEFT: Первое\nsha256: секрет\nMANAGER: Второе\nCLIENT: Третье",
    )
    legacy = publisher.render_legacy_identity_transcript(raw)

    assert legacy == "[00:00.0] Не определено: Первое MANAGER: Второе CLIENT: Третье"
    assert publisher.render_transcript(raw) == (
        "[00:00.0] Не определено: Первое Второе Третье"
    )
    identity = publisher.call_identity(raw)
    assert identity["legacy_transcript_sha"] == publisher.hashlib.sha256(
        legacy.encode("utf-8")
    ).hexdigest()


def test_legacy_identity_renderer_never_raises_on_a_broken_payload():
    for broken in ("not-json", "[1,2]", None, ""):
        raw = record(transcript_variants_json=broken, transcript_text="Текст")
        assert publisher.render_legacy_identity_transcript(raw) == (
            "[00:00.0] Не определено: Текст"
        )


def test_source_change_changes_fingerprint():
    assert projected()["source_fingerprint"] != projected(phone="+71111111111")["source_fingerprint"]


@pytest.mark.parametrize(
    "field",
    (
        "analysis_provider",
        "analysis_model",
        "analysis_prompt_version",
        "analysis_prompt_profile",
        "analysis_input_sha256",
        "analysis_source_sha256",
        "analysis_prompt_sha256",
        "dialogue_canonical_sha256",
        "manager_output_sha256",
    ),
)
def test_analysis_runtime_identity_changes_source_fingerprint(field):
    raw = record()
    analysis = json.loads(raw["analysis_json"])
    analysis["analysis_meta"] = {field: "identity-a"}
    first = publisher.call_projection(
        {**raw, "analysis_json": json.dumps(analysis)}, {"mango_manager_1": "Иван Иванов"}
    )
    analysis["analysis_meta"][field] = "identity-b"
    second = publisher.call_projection(
        {**raw, "analysis_json": json.dumps(analysis)}, {"mango_manager_1": "Иван Иванов"}
    )

    assert first["source_fingerprint"] != second["source_fingerprint"]


def test_telemetry_only_analysis_timestamp_does_not_change_fingerprint():
    baseline = trusted_record()
    changed = json.loads(str(baseline["analysis_json"]))
    changed["analyzed_at"] = "2099-01-01T00:00:00+00:00"

    first = trusted_projected(**baseline)
    second = trusted_projected(
        **{**baseline, "analysis_json": json.dumps(changed, ensure_ascii=False)}
    )

    assert first["source_fingerprint"] == second["source_fingerprint"]


def test_analysis_status_changes_fingerprint_even_when_safe_row_is_identical():
    pending = publisher.call_projection(record(analysis_status="pending"), {})
    failed = publisher.call_projection(record(analysis_status="failed"), {})

    assert pending["tail"] == failed["tail"]
    assert pending["source_fingerprint"] != failed["source_fingerprint"]


def test_unrelated_manager_mapping_does_not_change_fingerprint():
    raw = record()
    first = publisher.call_projection(raw, {"mango_manager_1": "Иван Иванов"})
    second = publisher.call_projection(
        raw,
        {"mango_manager_1": "Иван Иванов", "unrelated_manager": "Другой менеджер"},
    )
    assert first["source_fingerprint"] == second["source_fingerprint"]


def test_sheet_identity_accepts_current_and_minute_legacy_time():
    call = projected()
    current = with_number(call)
    assert publisher.identity_matches(publisher.sheet_identity(current), call)
    legacy = list(current)
    legacy[1] = "2026-08-14 12:03"
    legacy[4] = "142,5"
    assert publisher.identity_matches(publisher.sheet_identity(legacy), call)


def test_sheet_identity_accepts_proven_live_legacy_phone_and_zero_seconds():
    call = projected(started_at="2026-08-14 09:03:37")
    legacy = with_number(call)
    legacy[1] = "14.08.2026 12:03:00"
    legacy[6] = "'+70000000000"
    assert publisher.identity_matches(publisher.sheet_identity(legacy), call)


def test_normalize_values_rejects_headers_gaps_and_q_data():
    header = [*publisher.LIVE_HEADERS]
    row = with_number(projected())
    assert publisher.normalize_values([header, row])[1] == [row]
    with pytest.raises(ValueError, match="header"):
        publisher.normalize_values([["wrong"], row])
    with pytest.raises(ValueError, match="reserved helper"):
        publisher.normalize_values([header, [*row, "leaked"]])
    with pytest.raises(ValueError, match="contiguous"):
        publisher.normalize_values([header, row, [], row])


@pytest.mark.parametrize("nonblank", [0, False])
def test_normalize_values_treats_zero_and_false_as_nonblank(nonblank):
    header = [*publisher.LIVE_HEADERS]
    row = with_number(projected())

    with pytest.raises(ValueError, match="header"):
        publisher.normalize_values([[*header, nonblank], row])
    with pytest.raises(ValueError, match="reserved helper"):
        publisher.normalize_values([header, [*row, nonblank]])

    marker_row = [""] * publisher.MANAGED_COLUMN_COUNT
    marker_row[0] = nonblank
    parsed_rows = publisher.normalize_values([header, marker_row])[1]
    assert parsed_rows == [marker_row]


def test_reconcile_matches_one_call_and_blocks_unknown_or_duplicate():
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    assert publisher.reconcile([row], {call["call_key"]: call}, state)[0] == {call["call_key"]: 0}
    unknown = list(row)
    unknown[6] = "+79999999999"
    with pytest.raises(RuntimeError, match=r"ambiguous.*Google row 2 \(№ 10\)"):
        publisher.reconcile([unknown], {call["call_key"]: call}, state)
    with pytest.raises(RuntimeError, match="Google rows 2 and 3"):
        publisher.reconcile([row, row], {call["call_key"]: call}, state)


def test_reconcile_identifies_a_row_written_by_the_legacy_role_projection():
    call = projected()
    legacy_row = with_number(call)
    legacy_row[publisher.TRANSCRIPT_COLUMN_INDEX] = publisher.render_legacy_identity_transcript(record())
    identities = {call["call_key"]: publisher.call_identity(record())}

    assert publisher.reconcile(
        [legacy_row], identities, publisher.default_state("dest")
    )[0] == {call["call_key"]: 0}


def test_reconcile_physical_validation_error_names_google_row():
    call = projected()
    bad = with_number(call)
    bad[1] = "not-a-date"

    with pytest.raises(
        RuntimeError,
        match=r"reconcile: ValueError: message_sha256=.*Google row 2 \(№ 10\)",
    ):
        publisher.reconcile(
            [bad],
            {call["call_key"]: call},
            publisher.default_state("dest"),
        )


def test_reconcile_blocks_two_calls_with_same_business_identity():
    first = projected()
    second_record = record(id=8, source_call_id="call-8")
    second = publisher.call_projection(second_record, {"mango_manager_1": "Иван Иванов"})
    row = with_number(first)
    with pytest.raises(RuntimeError, match="ambiguous"):
        publisher.reconcile([row], {first["call_key"]: first, second["call_key"]: second}, publisher.default_state("dest"))


def test_reservation_is_stable_and_zero_change_is_zero_write():
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    mapping, _ = publisher.reconcile([row], {call["call_key"]: call}, state)
    _state, selected = publisher.reserve(state, {call["call_key"]: call}, [row], mapping, limit=25)
    assert selected == []

    changed = projected(phone="+71111111111")
    state, selected = publisher.reserve(state, {changed["call_key"]: changed}, [row], mapping, limit=25)
    assert selected == [changed["call_key"]]
    assert state["entries"][changed["call_key"]]["display_number"] == 10
    assert state["entries"][changed["call_key"]]["status"] == "reserved"


def test_missing_call_gets_next_display_number():
    first = projected()
    first_row = with_number(first, 17)
    second = publisher.call_projection(record(id=8, source_call_id="call-8", started_at="2026-08-14 10:00:00"), {})
    state = state_for(first, first_row)
    mapping, _ = publisher.reconcile([first_row], {first["call_key"]: first, second["call_key"]: second}, state)
    state, selected = publisher.reserve(
        state, {first["call_key"]: first, second["call_key"]: second}, [first_row], mapping, limit=25
    )
    assert selected == [second["call_key"]]
    assert state["entries"][second["call_key"]]["display_number"] == 18


def test_deleted_verified_row_is_restored_before_new_and_stale_rows():
    deleted = projected()
    new = publisher.call_projection(
        record(id=8, source_call_id="call-8", started_at="2026-08-14 10:00:00"), {}
    )
    stale = publisher.call_projection(
        record(id=9, source_call_id="call-9", started_at="2026-08-14 11:00:00"), {}
    )
    deleted_row = with_number(deleted, 10)
    stale_row = with_number(stale, 11)
    state = state_for(deleted, deleted_row)
    state["entries"].update(state_for(stale, stale_row)["entries"])
    physical_stale = list(stale_row)
    physical_stale[9] = "Устаревший конспект"
    state, selected = publisher.reserve(
        state,
        {deleted["call_key"]: deleted, new["call_key"]: new, stale["call_key"]: stale},
        [physical_stale],
        {stale["call_key"]: 0},
        limit=1,
    )
    assert selected == [deleted["call_key"]]


def test_missing_verified_row_with_nonpublishable_source_fails_closed():
    call = projected()
    state = state_for(call, with_number(call))
    with pytest.raises(RuntimeError, match="source is not publishable"):
        publisher.verify_sheet_snapshot(
            rows=[], call_to_row={}, identities={}, state=state, recoverable_keys=[]
        )


def test_unknown_result_is_recovered_from_exact_planned_readback():
    call = projected()
    row = with_number(call)
    state = state_for(call, row, status="reserved")
    mapping, _ = publisher.reconcile([row], {call["call_key"]: call}, state)
    assert publisher.applied_reservations(
        state, {call["call_key"]: call}, [row], mapping
    ) == [call["call_key"]]
    assert state["entries"][call["call_key"]]["status"] == "reserved"


def test_google_cells_never_emit_formula_values():
    for prefix in ("=", "+", "-", "@"):
        assert publisher.google_cell(prefix + "danger") == {
            "userEnteredValue": {"stringValue": prefix + "danger"}
        }


def test_height_depends_only_on_summary():
    summary = "Короткий конспект"
    assert publisher.row_height(summary) == publisher.row_height(summary)
    assert publisher.row_height(summary) != publisher.row_height("д" * 900)


def test_height_requests_skip_rows_that_already_have_the_target_height():
    requests = publisher.height_requests(
        0,
        [42, 78, 78, 96],
        current_heights=[42, 42, 78, 96],
    )

    assert len(requests) == 1
    target = requests[0]["updateDimensionProperties"]["range"]
    assert (target["startIndex"], target["endIndex"]) == (2, 3)

    swapped = publisher.height_requests(
        0,
        [78, 42],
        current_heights=[42, 78],
    )
    assert [
        (
            item["updateDimensionProperties"]["range"]["startIndex"],
            item["updateDimensionProperties"]["range"]["endIndex"],
        )
        for item in swapped
    ] == [(1, 2), (2, 3)]


def test_live_gateway_uses_extended_timeout_for_atomic_batch():
    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {}

    class Session:
        timeout = None

        def post(self, _url, **kwargs):
            self.timeout = kwargs["timeout"]
            return Response()

    session = Session()
    publisher.LiveGoogleGateway(session, "sheet").batch_sheet_requests([])
    assert session.timeout == 180


def test_layout_formula_error_names_cell_and_display_number():
    row = with_number(projected())
    fake = FakeLiveGoogleGateway([row])
    payload = fake.layout("Звонки", 2)
    values = payload["sheets"][0]["data"][0]["rowData"][0]["values"]
    values[publisher.TRANSCRIPT_COLUMN_INDEX]["userEnteredValue"] = {
        "formulaValue": '="hidden"'
    }

    with pytest.raises(RuntimeError, match=r"Q2 \(№ 10\)"):
        publisher.verify_layout(payload, [row], 320)


def test_build_batch_is_one_atomic_request_list_with_sort_clear_and_layout():
    first = projected()
    second = publisher.call_projection(record(id=8, source_call_id="call-8", started_at="2026-08-14 10:00:00"), {})
    first_row = with_number(first, 10)
    state = state_for(first, first_row)
    state["entries"][second["call_key"]] = {
        "display_number": 11,
        "status": "reserved",
        "projection_version": publisher.PROJECTION_VERSION,
        "source_fingerprint": second["source_fingerprint"],
        "started_epoch": second["started_epoch"],
        "planned_row_sha256": publisher.physical_row_hash(with_number(second, 11)),
        "last_verified_row_sha256": None,
        "attempts": 1,
    }
    requests, final_rows = publisher.build_batch(
        sheet_id=0,
        rows=[first_row],
        row_to_call={0: first["call_key"]},
        calls={first["call_key"]: first, second["call_key"]: second},
        identities={first["call_key"]: first, second["call_key"]: second},
        state=state,
        selected=[second["call_key"]],
        summary_width_px=320,
    )
    kinds = [next(iter(request)) for request in requests]
    assert kinds.count("appendCells") == 1
    assert kinds.count("sortRange") == 1
    assert kinds.count("repeatCell") >= 3  # helper clear plus summary/transcript formats
    assert final_rows[0][0] == 11
    assert all(len(row) == publisher.MANAGED_COLUMN_COUNT for row in final_rows)


def test_q_tie_breaker_is_display_number_and_stays_exact():
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    requests, _ = publisher.build_batch(
        sheet_id=0,
        rows=[row],
        row_to_call={0: call["call_key"]},
        calls={call["call_key"]: call},
        identities={call["call_key"]: call},
        state=state,
        selected=[call["call_key"]],
        summary_width_px=320,
    )
    q_request = next(
        item["updateCells"]
        for item in requests
        if "updateCells" in item
        and item["updateCells"].get("start", {}).get("columnIndex")
        == publisher.SORT_KEY_COLUMN_INDEX
    )
    actual = q_request["rows"][0]["values"][0]["userEnteredValue"]["numberValue"]
    assert actual == call["started_epoch"] * 1_000_000 + 10
    assert actual < 2**53


def test_run_rejects_wrong_sheet_title_id_pair_before_any_read_or_write(tmp_path, monkeypatch):
    fake = FakeLiveGoogleGateway(title="Звонки", sheet_id=99)
    env = _harness(tmp_path, monkeypatch, records=[record()], fake=fake, sheet_id=0)

    with pytest.raises(RuntimeError, match="title/id"):
        publisher.run(["--config", str(env["config"])])

    assert fake.values_calls == 0
    assert fake.batch_calls == 0


def test_shadow_without_state_checks_real_layout_before_bootstrap(tmp_path, monkeypatch):
    call = projected()
    row = with_number(call)
    fake = FakeLiveGoogleGateway([row])
    env = _harness(tmp_path, monkeypatch, records=[record()], rows=[row], fake=fake)
    valid_layout = fake.layout

    def wrong_height(title, last_row):
        payload = valid_layout(title, last_row)
        payload["sheets"][0]["data"][0]["rowMetadata"][0]["pixelSize"] = 21
        return payload

    fake.layout = wrong_height
    with pytest.raises(RuntimeError, match=r"Google row 2 \(№ 10\)"):
        publisher.run(["--config", str(env["config"])])

    assert fake.batch_calls == 0
    assert not env["state"].exists()


def test_shadow_without_state_checks_newest_first_before_bootstrap(tmp_path, monkeypatch):
    older_record = record()
    newer_record = record(
        id=8,
        source_call_id="call-8",
        started_at="2026-08-14 10:03:04",
        phone="+71111111111",
    )
    older = publisher.call_projection(older_record, {"mango_manager_1": "Иван Иванов"})
    newer = publisher.call_projection(newer_record, {"mango_manager_1": "Иван Иванов"})
    wrong_order = [with_number(older, 10), with_number(newer, 11)]
    fake = FakeLiveGoogleGateway(wrong_order)
    env = _harness(
        tmp_path,
        monkeypatch,
        records=[older_record, newer_record],
        rows=wrong_order,
        fake=fake,
    )

    with pytest.raises(RuntimeError, match="newest-first"):
        publisher.run(["--config", str(env["config"])])

    assert fake.batch_calls == 0
    assert not env["state"].exists()


def test_execute_rejects_explicit_zero_limit_without_state_or_google_change(
    tmp_path, monkeypatch
):
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)
    before = env["state"].read_bytes()

    with pytest.raises(RuntimeError, match="between 1 and 25"):
        _run_execute(env["config"], "--limit", "0")

    assert fake.batch_calls == 0
    assert env["state"].read_bytes() == before


@pytest.mark.parametrize(
    "attempts",
    [
        [{
            "attempt_id": "open", "stage": "analyze", "state": "reserved",
            "provider": "codex_cli", "model": "gpt-test", "profile": "compact",
            "prompt_version": "v8", "model_called": None, "cache_hit": False,
            "token_usage": {"source": "unavailable"},
        }],
        [{
            "provider": "codex_cli", "model": "gpt-test", "profile": "compact",
            "prompt_version": "v8", "model_called": True, "cache_hit": False,
            "token_usage": {"source": "unavailable"},
        }],
    ],
)
def test_invalid_attempt_ledger_blocks_before_the_first_google_batch(
    tmp_path, monkeypatch, attempts
):
    fake = FakeLiveGoogleGateway()
    env = _harness(
        tmp_path, monkeypatch,
        records=[trusted_record(analysis_attempts_json=json.dumps(attempts))],
        state=publisher.default_state(_destination()), fake=fake,
    )

    with pytest.raises(RuntimeError, match="cost ledger does not close"):
        _run_execute(env["config"])

    assert fake.batch_calls == 0


def test_run_refuses_repeated_bootstrap_over_nonempty_state(tmp_path, monkeypatch):
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    state["destination_id"] = _destination()
    fake = FakeLiveGoogleGateway([row])
    env = _harness(
        tmp_path, monkeypatch, records=[record()], rows=[row], state=state, fake=fake
    )

    with pytest.raises(RuntimeError, match="non-empty"):
        publisher.run(
            [
                "--config", str(env["config"]), "--bootstrap", "--confirmation",
                publisher.BOOTSTRAP_CONFIRMATION,
            ]
        )

    assert fake.batch_calls == 0
    assert json.loads(env["state"].read_text(encoding="utf-8"))["entries"] == state["entries"]


def test_run_recovers_applied_timeout_without_duplicate_row(tmp_path, monkeypatch):
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    fake.raise_after_apply = True
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)

    with pytest.raises(TimeoutError, match="lost Google response"):
        _run_execute(env["config"])

    reserved = json.loads(env["state"].read_text(encoding="utf-8"))
    assert next(iter(reserved["entries"].values()))["status"] == "reserved"
    assert len(fake.rows) == 1
    assert fake.batch_calls == 1
    assert _db_status(env["db"]) == ("pending", 0)

    report = _run_execute(env["config"])

    verified = json.loads(env["state"].read_text(encoding="utf-8"))
    assert report["status"] == "no_change"
    assert next(iter(verified["entries"].values()))["status"] == "verified"
    assert len(fake.rows) == 1
    assert fake.batch_calls == 1
    assert _db_status(env["db"]) == ("done", 1)


def test_source_change_after_batch_build_blocks_google_write(tmp_path, monkeypatch):
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)
    real_build = publisher.build_batch

    def mutate_then_return(*args, **kwargs):
        result = real_build(*args, **kwargs)
        with sqlite3.connect(env["db"]) as connection:
            connection.execute(
                "UPDATE call_records SET manager_name='manager_changed_after_build' WHERE id=7"
            )
        return result

    monkeypatch.setattr(publisher, "build_batch", mutate_then_return)

    with pytest.raises(RuntimeError, match="selected source changed"):
        _run_execute(env["config"])

    assert fake.batch_calls == 0
    assert _db_status(env["db"]) == ("pending", 0)


def test_sqlite_source_fence_blocks_analyze_writes_until_google_window_closes(
    tmp_path,
):
    db_path = tmp_path / "fenced.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE calls (id INTEGER PRIMARY KEY, value TEXT)")
        connection.execute("INSERT INTO calls VALUES (1, 'before')")

    with publisher.sqlite_source_write_fence(db_path):
        with sqlite3.connect(db_path, timeout=0.01) as competing:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                competing.execute("UPDATE calls SET value='during' WHERE id=1")

    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE calls SET value='after' WHERE id=1")
        assert connection.execute("SELECT value FROM calls WHERE id=1").fetchone() == (
            "after",
        )


def test_recovery_balance_must_close_before_verified_or_sync(tmp_path, monkeypatch):
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    fake.raise_after_apply = True
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)

    with pytest.raises(TimeoutError, match="lost Google response"):
        _run_execute(env["config"])

    monkeypatch.setattr(
        publisher,
        "require_closed_balance",
        lambda _balance: (_ for _ in ()).throw(RuntimeError("forced recovery balance")),
    )
    with pytest.raises(RuntimeError, match="forced recovery balance"):
        _run_execute(env["config"])

    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    assert next(iter(persisted["entries"].values()))["status"] == "reserved"
    assert _db_status(env["db"]) == ("pending", 0)
    assert fake.batch_calls == 1


def test_no_change_pending_sync_requires_layout_before_done(tmp_path, monkeypatch):
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    state["destination_id"] = _destination()
    fake = FakeLiveGoogleGateway([row])
    env = _harness(
        tmp_path, monkeypatch, records=[record()], rows=[row], state=state, fake=fake
    )

    def broken_layout(_title, _last_row):
        raise RuntimeError("layout is broken")

    fake.layout = broken_layout
    with pytest.raises(RuntimeError, match="layout is broken"):
        _run_execute(env["config"])

    assert _db_status(env["db"]) == ("pending", 0)
    assert fake.batch_calls == 0


def test_run_source_change_between_verified_state_and_sync_never_marks_done(
    tmp_path, monkeypatch
):
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)
    monkeypatch.setattr(
        publisher, "sqlite_source_write_fence", lambda _path: nullcontext()
    )
    original_finalize = publisher.finalize_verified

    def finalize_then_change(*args, **kwargs):
        proofs = original_finalize(*args, **kwargs)
        connection = sqlite3.connect(env["db"])
        connection.execute("UPDATE call_records SET phone=? WHERE id=7", ("+71111111111",))
        connection.commit()
        connection.close()
        return proofs

    monkeypatch.setattr(publisher, "finalize_verified", finalize_then_change)

    with pytest.raises(RuntimeError, match="source changed before sync"):
        _run_execute(env["config"])

    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    assert next(iter(persisted["entries"].values()))["status"] == "verified"
    assert fake.batch_calls == 1
    assert _db_status(env["db"]) == ("pending", 0)


def _break_source_call_id(db_path, call_id=7):
    connection = sqlite3.connect(db_path)
    connection.execute("UPDATE call_records SET source_call_id='' WHERE id=?", (call_id,))
    connection.commit()
    connection.close()


def _change_phone(db_path, phone, call_id=7):
    connection = sqlite3.connect(db_path)
    connection.execute("UPDATE call_records SET phone=? WHERE id=?", (phone, call_id))
    connection.commit()
    connection.close()


def test_source_change_during_google_write_is_compensated_before_sync_done(
    tmp_path, monkeypatch
):
    fake = FakeLiveGoogleGateway()
    env = _harness(
        tmp_path,
        monkeypatch,
        records=[record()],
        state=publisher.default_state(_destination()),
        fake=fake,
    )
    monkeypatch.setattr(
        publisher, "sqlite_source_write_fence", lambda _path: nullcontext()
    )

    def mutate_once(_gateway):
        if fake.batch_calls == 1:
            _change_phone(env["db"], "+71111111111")

    fake.on_batch_applied = mutate_once

    report = _run_execute(env["config"])

    assert report["status"] == "published"
    assert fake.batch_calls == 2
    assert fake.rows[0][publisher.LIVE_HEADERS.index("Телефон клиента")] == "+71111111111"
    assert _db_status(env["db"]) == ("done", 1)


def test_three_source_changes_are_compensated_then_stable_check_allows_sync(
    tmp_path, monkeypatch
):
    fake = FakeLiveGoogleGateway()
    env = _harness(
        tmp_path,
        monkeypatch,
        records=[record()],
        state=publisher.default_state(_destination()),
        fake=fake,
    )
    monkeypatch.setattr(
        publisher, "sqlite_source_write_fence", lambda _path: nullcontext()
    )

    def mutate_three_times(_gateway):
        if fake.batch_calls <= 3:
            _change_phone(env["db"], f"+7111111111{fake.batch_calls}")

    fake.on_batch_applied = mutate_three_times

    report = _run_execute(env["config"])

    assert report["status"] == "published"
    assert fake.batch_calls == 4
    assert fake.rows[0][publisher.LIVE_HEADERS.index("Телефон клиента")] == "+71111111113"
    assert _db_status(env["db"]) == ("done", 1)


def test_fourth_source_change_stops_after_three_compensations(tmp_path, monkeypatch):
    fake = FakeLiveGoogleGateway()
    env = _harness(
        tmp_path,
        monkeypatch,
        records=[record()],
        state=publisher.default_state(_destination()),
        fake=fake,
    )
    monkeypatch.setattr(
        publisher, "sqlite_source_write_fence", lambda _path: nullcontext()
    )

    def keep_mutating(_gateway):
        if fake.batch_calls <= 4:
            _change_phone(env["db"], f"+7111111111{fake.batch_calls}")

    fake.on_batch_applied = keep_mutating

    with pytest.raises(RuntimeError, match="source kept changing"):
        _run_execute(env["config"])

    assert fake.batch_calls == 4
    assert _db_status(env["db"]) == ("pending", 0)


def test_identity_error_in_the_final_reload_blocks_sync_before_any_db_write(
    tmp_path, monkeypatch
):
    """A reload is a new read: it gets the same identity gate as the first one."""
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)
    monkeypatch.setattr(
        publisher, "sqlite_source_write_fence", lambda _path: nullcontext()
    )
    fake.on_batch_applied = lambda _gateway: _break_source_call_id(env["db"])

    with pytest.raises(RuntimeError, match="identity errors block the whole run"):
        _run_execute(env["config"])

    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    assert fake.batch_calls == 1
    assert [item["code"] for item in persisted["incidents"].values()] == [
        "identity_source_call_id_empty"
    ]
    assert _db_status(env["db"]) == ("pending", 0)


def test_identity_error_in_the_recovery_reload_blocks_sync_before_any_db_write(
    tmp_path, monkeypatch
):
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    fake.raise_after_apply = True
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)

    with pytest.raises(TimeoutError, match="lost Google response"):
        _run_execute(env["config"])
    assert _db_status(env["db"]) == ("pending", 0)

    original = publisher.applied_reservations

    def recover_then_break(*args, **kwargs):
        recovered = original(*args, **kwargs)
        _break_source_call_id(env["db"])
        return recovered

    monkeypatch.setattr(publisher, "applied_reservations", recover_then_break)

    with pytest.raises(RuntimeError, match="identity errors block the whole run"):
        _run_execute(env["config"])

    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    assert fake.batch_calls == 1
    assert [item["code"] for item in persisted["incidents"].values()] == [
        "identity_source_call_id_empty"
    ]
    assert _db_status(env["db"]) == ("pending", 0)


def test_run_fails_if_unselected_google_row_changes_concurrently(tmp_path, monkeypatch):
    first_record = record()
    second_record = record(
        id=8,
        source_call_id="call-8",
        started_at="2026-08-14 10:03:04",
        phone="+71111111111",
    )
    first = publisher.call_projection(first_record, {"mango_manager_1": "Иван Иванов"})
    second = publisher.call_projection(second_record, {"mango_manager_1": "Иван Иванов"})
    stale_first = with_number(first, 10)
    stale_first[9] = "Старый конспект"
    second_row = with_number(second, 11)
    state = state_for(first, stale_first)
    state["destination_id"] = _destination()
    state["entries"].update(state_for(second, second_row)["entries"])
    fake = FakeLiveGoogleGateway([second_row, stale_first])

    def change_unselected(gateway):
        for row in gateway.rows:
            if int(row[0]) == 11:
                row[9] = "Параллельная чужая правка"

    fake.on_batch_applied = change_unselected
    env = _harness(
        tmp_path,
        monkeypatch,
        records=[first_record, second_record],
        state=state,
        fake=fake,
    )

    with pytest.raises(RuntimeError, match="unselected Google row changed concurrently"):
        _run_execute(env["config"], "--limit", "1")

    assert fake.batch_calls == 1
    assert _db_status(env["db"], 7) == ("pending", 0)
    assert _db_status(env["db"], 8) == ("pending", 0)


def test_invalid_source_after_prewrite_crash_does_not_block_other_call(
    tmp_path, monkeypatch
):
    original = projected()
    reserved_row = with_number(original, 10)
    state = state_for(original, reserved_row, status="reserved")
    state["destination_id"] = _destination()
    malformed = record(
        transcript_variants_json=json.dumps(
            {"dialogue_lines": ["[00:01.0] Дорожка левая: Валидно", "сломанная строка"]},
            ensure_ascii=False,
        )
    )
    valid = record(
        id=8,
        source_call_id="call-8",
        started_at="2026-08-14 10:03:04",
        phone="+71111111111",
    )
    fake = FakeLiveGoogleGateway()
    env = _harness(
        tmp_path,
        monkeypatch,
        records=[malformed, valid],
        state=state,
        fake=fake,
    )

    report = _run_execute(env["config"], "--limit", "1")

    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    assert report["status"] == "published"
    assert report["published"] == 1
    assert len(fake.rows) == 1
    assert fake.rows[0][publisher.LIVE_HEADERS.index("Телефон клиента")] == "+71111111111"
    assert persisted["entries"][original["call_key"]]["status"] == "reserved"
    assert _db_status(env["db"], 7) == ("pending", 0)
    assert _db_status(env["db"], 8) == ("done", 1)


def test_one_bad_row_yields_one_published_row_one_incident_and_exit_two(
    tmp_path, monkeypatch, capsys
):
    broken = record(
        id=9,
        source_call_id="call-9",
        started_at="2026-08-14 08:00:00",
        phone="+72222222222",
        transcript_variants_json=json.dumps(
            {"dialogue_lines": ["сломанная строка"]}, ensure_ascii=False
        ),
    )
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(
        tmp_path, monkeypatch, records=[record(), broken], state=state, fake=fake
    )

    exit_code = publisher.main(
        [
            "--config", str(env["config"]), "--execute", "--confirmation",
            publisher.CONFIRMATION,
        ]
    )

    report = json.loads(capsys.readouterr().out)
    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    incident = next(iter(persisted["incidents"].values()))
    assert exit_code == 2
    assert report["status"] == "published"
    assert report["published"] == 1
    assert len(fake.rows) == 1
    assert all(row[10] != publisher.SAFE_ERROR_RESULT_RU for row in fake.rows)
    assert report["health"] == {
        "status": "amber", "open_incidents": 1, "sla_breached": 0, "sla_hours": 24,
    }
    assert incident["code"] == "projection_dialogue_line_malformed"
    assert incident["first_seen_at"] and incident["last_seen_at"]
    assert _db_status(env["db"], 7) == ("done", 1)
    assert _db_status(env["db"], 9) == ("pending", 0)


def test_second_unchanged_run_keeps_the_same_incident_and_writes_nothing(
    tmp_path, monkeypatch
):
    broken = record(
        transcript_variants_json=json.dumps(
            {"dialogue_lines": ["сломанная строка"]}, ensure_ascii=False
        )
    )
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[broken], state=state, fake=fake)

    first = _run_execute(env["config"])
    after_first = json.loads(env["state"].read_text(encoding="utf-8"))["incidents"]
    second = _run_execute(env["config"])
    after_second = json.loads(env["state"].read_text(encoding="utf-8"))["incidents"]

    assert first["health"]["status"] == second["health"]["status"] == "amber"
    assert second["health"]["open_incidents"] == 1
    assert list(after_second) == list(after_first)
    assert [item["first_seen_at"] for item in after_second.values()] == [
        item["first_seen_at"] for item in after_first.values()
    ]
    assert first["status"] == "no_change"
    assert second["status"] == "no_change"
    assert first["balance"]["failed_with_incident"] == 1
    assert second["balance"]["failed_with_incident"] == 1
    assert fake.batch_calls == 0
    assert fake.rows == []


def test_healthy_run_exits_zero_and_keeps_no_incident(tmp_path, monkeypatch, capsys):
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)

    exit_code = publisher.main(
        [
            "--config", str(env["config"]), "--execute", "--confirmation",
            publisher.CONFIRMATION,
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert report["health"]["status"] == "green"
    assert json.loads(env["state"].read_text(encoding="utf-8"))["incidents"] == {}


def test_incident_older_than_sla_turns_health_red():
    incidents = {
        "7": {
            "call_key": "",
            "code": "identity_source_call_id_empty",
            "first_seen_at": "2026-08-14T09:00:00+00:00",
            "last_seen_at": "2026-08-16T09:00:00+00:00",
        }
    }
    assert publisher.health_report(incidents, now="2026-08-14T20:00:00+00:00") == {
        "status": "amber", "open_incidents": 1, "sla_breached": 0, "sla_hours": 24,
    }
    assert publisher.health_report(incidents, now="2026-08-16T09:00:00+00:00")["status"] == "red"
    assert publisher.health_report({}, now="2026-08-16T09:00:00+00:00")["status"] == "green"


def test_resolved_incident_disappears_but_repeated_one_keeps_first_seen():
    previous = {
        "7": {
            "call_key": "mango:mango_office:call-7",
            "code": "projection_dialogue_line_malformed",
            "first_seen_at": "2026-08-14T09:00:00+00:00",
            "last_seen_at": "2026-08-14T09:00:00+00:00",
        },
        "8": {
            "call_key": "",
            "code": "identity_source_call_id_empty",
            "first_seen_at": "2026-08-14T09:00:00+00:00",
            "last_seen_at": "2026-08-14T09:00:00+00:00",
        },
    }
    merged = publisher.merge_incidents(
        previous,
        {
            "7": {
                "call_key": "mango:mango_office:call-7",
                "code": "projection_dialogue_line_malformed",
            },
            "9": {"call_key": "", "code": "projection_analysis_json_invalid"},
        },
        now="2026-08-16T09:00:00+00:00",
    )

    assert set(merged) == {"7", "9"}
    assert merged["7"]["first_seen_at"] == "2026-08-14T09:00:00+00:00"
    assert merged["7"]["last_seen_at"] == "2026-08-16T09:00:00+00:00"
    assert merged["9"]["first_seen_at"] == "2026-08-16T09:00:00+00:00"


def test_a_new_code_for_the_same_call_keeps_first_seen_until_it_is_really_fixed():
    key = "mango:mango_office:call-7"
    merged = publisher.merge_incidents(
        {
            key: {
                "call_key": key,
                "code": "projection_dialogue_line_malformed",
                "first_seen_at": "2026-08-14T09:00:00+00:00",
                "last_seen_at": "2026-08-14T09:00:00+00:00",
            }
        },
        {key: {"call_key": key, "code": "projection_analysis_json_invalid"}},
        now="2026-08-16T09:00:00+00:00",
    )
    assert merged[key]["first_seen_at"] == "2026-08-14T09:00:00+00:00"
    assert merged[key]["previous_code"] == "projection_dialogue_line_malformed"
    assert merged[key]["code"] == "projection_analysis_json_invalid"


def test_incident_key_is_stable_de_identified_and_not_the_local_row_id():
    first = publisher.incident_key(record(id=7))
    same_call_moved = publisher.incident_key(record(id=99))
    other_call = publisher.incident_key(record(source_call_id="call-8"))
    without_id = publisher.incident_key(record(source_call_id=""))

    assert first == same_call_moved
    assert first != other_call
    # The sidecar is a durable journal, so the provider call id itself never
    # reaches it: only an irreversible digest of the stable call key.
    assert "call-7" not in first
    assert first.startswith("call:")
    for key in (first, other_call, without_id):
        digest = key.split(":", 1)[1]
        assert len(digest) == 32
        assert all(char in "0123456789abcdef" for char in digest)
    assert without_id.startswith("unresolved:")
    assert publisher.incident_key(record(source_call_id="")) == without_id


def test_no_incident_field_carries_the_raw_provider_call_id(tmp_path, monkeypatch):
    broken = record(
        transcript_variants_json=json.dumps({"dialogue_lines": ["сломанная строка"]})
    )
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[broken], fake=fake)

    publisher.run(["--config", str(env["config"])])
    sidecar = json.loads(Path(env["state"]).read_text(encoding="utf-8"))

    assert sidecar["incidents"]
    dumped = json.dumps(sidecar["incidents"], ensure_ascii=False)
    for personal in ("call-7", "+70000000000", "mango_manager_1", "Добрый день"):
        assert personal not in dumped


def test_a_legacy_sidecar_stops_carrying_the_raw_call_key_when_it_is_read():
    incidents = publisher.validated_incidents(
        {
            "mango:mango_office:call-7": {
                "class": publisher.INCIDENT_CLASS_DATA,
                "call_key": "mango:mango_office:call-7",
                "code": "projection_dialogue_line_malformed",
                "first_seen_at": "2026-08-14T09:00:00+00:00",
                "last_seen_at": "2026-08-14T09:00:00+00:00",
            }
        }
    )
    digest = publisher.call_key_digest("mango:mango_office:call-7")
    entry = incidents["call:" + digest]

    assert "call_key" not in entry
    assert entry["call_digest"] == digest


def test_a_row_without_source_call_id_is_keyed_by_technical_identity_only():
    """Two different broken rows must not collapse into one incident."""
    base = record(source_call_id="")
    same = publisher.incident_key(record(source_call_id=""))

    assert publisher.incident_key(base) == same
    # Every technical coordinate separates two rows that share a start time.
    for changed in (
        {"id": 42},
        {"duration_sec": 300.0},
        {"started_at": "2026-08-14 10:00:00"},
        {"transcript_variants_json": json.dumps({"dialogue_lines": []})},
    ):
        assert publisher.incident_key(record(source_call_id="", **changed)) != same
    # ...and nothing personal is in the key: only a hash of technical fields.
    for personal in ("+70000000000", "mango_manager_1", "Добрый день"):
        assert personal not in publisher.incident_key(record(source_call_id=""))


def test_old_state_v1_data_errors_migrate_without_losing_anything(tmp_path):
    path = tmp_path / "state.json"
    legacy = {
        "schema_version": publisher.STATE_SCHEMA_LEGACY,
        "destination_id": "dest",
        "entries": {},
        "updated_at": "2026-08-14T09:00:00+00:00",
        "data_errors": {
            "7": {
                "call_key": "mango:mango_office:call-7",
                "code": "projection_dialogue_line_malformed",
            },
            "8": {"call_key": "", "code": "identity_source_call_id_empty"},
        },
    }
    _owner_json(path, legacy)

    state = publisher.load_state(path, "dest", required=True)

    # A v1 sidecar is readable and comes back as v2; the owner edits nothing.
    assert publisher.STATE_SCHEMA_LEGACY != publisher.STATE_SCHEMA
    assert state["schema_version"] == publisher.STATE_SCHEMA
    assert publisher.LEGACY_ERROR_FIELD not in state
    assert len(state["incidents"]) == 2
    # The raw provider call key does not survive the migration: it becomes the
    # same digest the current run would produce, so the history is preserved
    # without the sidecar carrying a real call id.
    digest = publisher.call_key_digest("mango:mango_office:call-7")
    stable = "call:" + digest
    assert "mango:mango_office:call-7" not in json.dumps(
        state["incidents"], ensure_ascii=False
    )
    migrated = state["incidents"][stable]
    assert migrated["call_digest"] == digest
    assert migrated["code"] == "projection_dialogue_line_malformed"
    assert migrated["first_seen_at"] == "2026-08-14T09:00:00+00:00"
    assert migrated["first_seen_source"] == "migrated_state_v1_data_errors"
    # A migrated incident that is still open keeps its history on the next run,
    # under the key the current ``incident_key`` produces.
    assert publisher.incident_key(record()) == stable
    kept = publisher.merge_incidents(
        state["incidents"],
        {stable: {"call_digest": digest, "code": "projection_dialogue_line_malformed"}},
        now="2026-08-16T09:00:00+00:00",
    )
    assert kept[stable]["first_seen_at"] == "2026-08-14T09:00:00+00:00"


@pytest.mark.parametrize(
    "incident",
    [
        "not-an-object",
        {"code": "Projection Bad Code", "first_seen_at": "2026-08-14T09:00:00+00:00",
         "last_seen_at": "2026-08-14T09:00:00+00:00"},
        {"code": "projection_dialogue_line_malformed", "first_seen_at": "вчера",
         "last_seen_at": "2026-08-14T09:00:00+00:00"},
        {"code": "projection_dialogue_line_malformed",
         "first_seen_at": "2026-08-14T09:00:00+00:00"},
    ],
)
def test_a_corrupt_incident_stops_the_run_instead_of_being_filtered_out(
    tmp_path, incident
):
    path = tmp_path / "state.json"
    _owner_json(
        path,
        {
            "schema_version": publisher.STATE_SCHEMA,
            "destination_id": "dest",
            "entries": {},
            "incidents": {"mango:mango_office:call-7": incident},
        },
    )

    with pytest.raises(RuntimeError, match="incident"):
        publisher.load_state(path, "dest", required=True)


def test_identity_ambiguity_blocks_the_whole_write_with_exit_one(
    tmp_path, monkeypatch, capsys
):
    call = projected()
    unknown_row = with_number(call)
    unknown_row[6] = "+79999999999"
    state = state_for(call, with_number(call))
    state["destination_id"] = _destination()
    fake = FakeLiveGoogleGateway([unknown_row])
    env = _harness(
        tmp_path, monkeypatch, records=[record()], state=state, fake=fake
    )

    exit_code = publisher.main(
        [
            "--config", str(env["config"]), "--execute", "--confirmation",
            publisher.CONFIRMATION,
        ]
    )

    report = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["error"].startswith("publish: ReconcileError: message_sha256=")
    assert fake.batch_calls == 0
    assert _db_status(env["db"]) == ("pending", 0)


@pytest.mark.parametrize("corruption", ["duplicate_number", "invalid_hash", "invalid_status"])
def test_run_rejects_corrupt_owner_state_without_google_write(
    tmp_path, monkeypatch, corruption
):
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    state["destination_id"] = _destination()
    entry = state["entries"][call["call_key"]]
    if corruption == "duplicate_number":
        duplicate = dict(entry)
        state["entries"]["mango:mango_office:call-8"] = duplicate
    elif corruption == "invalid_hash":
        entry["source_fingerprint"] = "not-a-sha256"
    else:
        entry["status"] = "published"
    fake = FakeLiveGoogleGateway([row])
    env = _harness(
        tmp_path, monkeypatch, records=[record()], state=state, fake=fake
    )

    with pytest.raises(RuntimeError, match="publisher state"):
        _run_execute(env["config"])

    assert fake.batch_calls == 0


def test_run_reports_malformed_dialogue_as_data_error_without_partial_transcript(
    tmp_path, monkeypatch
):
    malformed = record(
        transcript_variants_json=json.dumps(
            {
                "role_mapping": {"left": "manager"},
                "dialogue_lines": [
                    "[00:01.0] Дорожка левая: Валидно",
                    "сломанная строка",
                ],
            },
            ensure_ascii=False,
        )
    )
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[malformed], fake=fake)

    report = publisher.run(["--config", str(env["config"])])

    assert report["status"] == "shadow_ok"
    assert report["analysis_done"] == 0
    assert report["data_errors"] == 1
    assert report["data_error_codes"] == {"projection_dialogue_line_malformed": 1}
    assert fake.rows == []
    assert fake.batch_calls == 0


def test_run_does_not_report_pending_empty_transcript_as_data_error(
    tmp_path, monkeypatch
):
    pending = record(
        analysis_status="pending",
        analysis_json=None,
        transcript_variants_json=None,
        transcript_text=None,
        duration_sec=None,
    )
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[pending], fake=fake)

    report = publisher.run(["--config", str(env["config"])])

    assert report["status"] == "shadow_ok"
    assert report["source_calls"] == 1
    assert report["analysis_done"] == 0
    assert report["data_errors"] == 0
    assert fake.batch_calls == 0


def test_pending_call_is_visible_but_never_marked_sync_done(tmp_path, monkeypatch):
    pending = record(
        analysis_status="pending", analysis_json=None,
        transcript_variants_json=None, transcript_text=None, duration_sec=None,
    )
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[pending], state=state, fake=fake)

    report = _run_execute(env["config"])

    assert report["balance"]["source_calls"] == 1
    assert report["balance"]["analysis_done"] == 0
    assert report["balance"]["balanced"] is True
    assert len(fake.rows[0]) == len(publisher.LIVE_HEADERS)
    assert fake.rows[0][9] == publisher.SAFE_PENDING_SUMMARY_RU
    assert fake.rows[0][10] == publisher.SAFE_PENDING_RESULT_RU
    assert fake.rows[0][publisher.TRANSCRIPT_COLUMN_INDEX] == publisher.SAFE_PENDING_TRANSCRIPT_RU
    assert _db_status(env["db"]) == ("pending", 0)


def test_malformed_done_is_excluded_with_incident_and_never_sync_done(
    tmp_path, monkeypatch
):
    malformed = record(
        transcript_variants_json=json.dumps(
            {"dialogue_lines": ["[00:01.0] Дорожка левая: Валидно", "сломанная строка"]},
            ensure_ascii=False,
        )
    )
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[malformed], state=state, fake=fake)

    report = _run_execute(env["config"])
    persisted = json.loads(env["state"].read_text(encoding="utf-8"))

    assert fake.rows == []
    assert fake.batch_calls == 0
    assert report["status"] == "no_change"
    assert report["balance"]["source_calls"] == 1
    assert report["balance"]["analysis_done"] == 0
    assert report["balance"]["failed_with_incident"] == 1
    assert report["balance"]["balanced"] is True
    assert next(iter(persisted["incidents"].values()))["code"] == (
        "projection_dialogue_line_malformed"
    )
    assert _db_status(env["db"]) == ("pending", 0)


def test_stale_v2_analysis_is_excluded_with_incident_without_business_facts(tmp_path):
    stale = json.loads(trusted_record()["analysis_json"])
    stale.update(
        {
            "analysis_schema_version": "v2",
            "history_summary": "Оплата подтверждена на 100 000 рублей",
            "follow_up_reason": "Срочно продать",
        }
    )
    stale["structured_fields"]["result"] = {
        "status": "payment_confirmed", "detail": "100 000 рублей",
    }
    stale["structured_fields"]["next_step"] = {
        "action": "Списать оплату", "due": "сегодня",
    }
    db_path = _sqlite(
        tmp_path,
        [trusted_record(analysis_json=json.dumps(stale, ensure_ascii=False))],
    )

    calls, identities, errors = publisher.load_calls(db_path, {})
    dumped = json.dumps(errors, ensure_ascii=False)

    assert len(identities) == 1
    assert calls == {}
    assert errors[next(iter(errors))]["code"] == "projection_analysis_contract_invalid"
    assert "100 000" not in dumped and "Списать оплату" not in dumped


@pytest.mark.parametrize("source_call_id", [None, ""])
def test_row_without_source_call_id_blocks_the_run_and_leaves_an_incident(
    tmp_path, monkeypatch, source_call_id
):
    invalid = record(source_call_id=source_call_id)
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[invalid], fake=fake)

    with pytest.raises(RuntimeError, match="identity errors block the whole run"):
        publisher.run(["--config", str(env["config"])])

    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    incident = next(iter(persisted["incidents"].values()))
    assert incident["code"] == "identity_source_call_id_empty"
    assert incident["first_seen_at"] and incident["last_seen_at"]
    assert persisted["health"]["status"] in {"amber", "red"}
    assert fake.batch_calls == 0
    assert fake.values_calls == 0


@pytest.mark.parametrize("source_call_id", ["x" * 600, "bad\ncall\nid"])
def test_unsafe_source_call_id_keeps_one_stable_incident_across_runs(
    tmp_path, monkeypatch, source_call_id
):
    invalid = record(source_call_id=source_call_id)
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[invalid], fake=fake)

    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="identity_source_call_id_invalid"):
            publisher.run(["--config", str(env["config"])])
        persisted = json.loads(env["state"].read_text(encoding="utf-8"))
        if _attempt == 0:
            first_seen = [item["first_seen_at"] for item in persisted["incidents"].values()]

    assert len(persisted["incidents"]) == 1
    assert [item["first_seen_at"] for item in persisted["incidents"].values()] == first_seen
    assert fake.batch_calls == 0


def test_one_identity_error_blocks_the_good_row_too_and_writes_no_batch(
    tmp_path, monkeypatch, capsys
):
    broken = record(id=9, source_call_id="", started_at="2026-08-14 08:00:00")
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(
        tmp_path, monkeypatch, records=[record(), broken], state=state, fake=fake
    )

    exit_code = publisher.main(
        [
            "--config", str(env["config"]), "--execute", "--confirmation",
            publisher.CONFIRMATION,
        ]
    )

    report = json.loads(capsys.readouterr().out)
    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["status"] == "failed"
    assert report["error"].startswith("publish: RuntimeError: message_sha256=")
    assert fake.batch_calls == 0
    assert len(fake.rows) == 0
    assert [item["code"] for item in persisted["incidents"].values()] == [
        "identity_source_call_id_empty"
    ]
    assert _db_status(env["db"], 7) == ("pending", 0)


def test_an_unidentified_physical_row_also_leaves_an_incident(tmp_path, monkeypatch):
    call = projected()
    unknown_row = with_number(call)
    unknown_row[6] = "+79999999999"
    state = state_for(call, with_number(call))
    state["destination_id"] = _destination()
    fake = FakeLiveGoogleGateway([unknown_row])
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)

    with pytest.raises(RuntimeError, match="unidentified_or_ambiguous_physical_row"):
        _run_execute(env["config"])

    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    incident = next(iter(persisted["incidents"].values()))
    assert list(persisted["incidents"])[0].startswith("reconcile:")
    assert incident["code"] == "reconcile_unidentified_or_ambiguous_row"
    assert fake.batch_calls == 0


@pytest.mark.parametrize("analysis_json", [None, "", "not-json", "[]", "null", "{}"])
def test_run_rejects_done_row_without_valid_analysis_object(
    tmp_path, monkeypatch, analysis_json
):
    invalid = record(analysis_json=analysis_json)
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[invalid], fake=fake)

    report = publisher.run(["--config", str(env["config"])])

    assert report["status"] == "shadow_ok"
    assert report["analysis_done"] == 0
    assert report["data_error_codes"] == {"projection_analysis_json_invalid": 1}
    assert fake.batch_calls == 0
    assert _db_status(env["db"]) == ("pending", 0)


@pytest.mark.parametrize("analysis_json", [None, "not-json", "{}"])
def test_malformed_or_empty_done_analysis_is_excluded_with_visible_incident(
    tmp_path, monkeypatch, analysis_json
):
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(
        tmp_path, monkeypatch, records=[record(analysis_json=analysis_json)],
        state=state, fake=fake,
    )

    report = _run_execute(env["config"])

    assert fake.rows == []
    assert fake.batch_calls == 0
    assert report["status"] == "no_change"
    assert report["balance"]["source_calls"] == 1
    assert report["balance"]["analysis_done"] == 0
    assert report["balance"]["failed_with_incident"] == 1
    assert report["balance"]["balanced"] is True
    assert report["data_errors"] == 1
    assert _db_status(env["db"]) == ("pending", 0)


def test_run_allows_only_one_writer_for_same_lock(tmp_path, monkeypatch):
    fake = FakeLiveGoogleGateway()
    entered = threading.Event()
    release = threading.Event()

    def block_first_values():
        entered.set()
        assert release.wait(timeout=5)

    fake.before_values = block_first_values
    env = _harness(tmp_path, monkeypatch, records=[record()], fake=fake)
    first_result = []
    first_error = []

    def first_run():
        try:
            first_result.append(publisher.run(["--config", str(env["config"])]))
        except Exception as exc:  # pragma: no cover - surfaced by assertions below
            first_error.append(exc)

    thread = threading.Thread(target=first_run)
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="publisher is active"):
            publisher.run(["--config", str(env["config"])])
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert first_error == []
    assert first_result[0]["status"] == "shadow_ok"
    assert fake.batch_calls == 0


def test_run_db_busy_fails_before_sync_done(tmp_path, monkeypatch):
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    state["destination_id"] = _destination()
    fake = FakeLiveGoogleGateway([row])
    env = _harness(
        tmp_path, monkeypatch, records=[record()], rows=[row], state=state, fake=fake
    )
    real_connect = sqlite3.connect

    class BusyConnection:
        def __init__(self, inner):
            object.__setattr__(self, "inner", inner)

        def __setattr__(self, name, value):
            if name == "row_factory":
                self.inner.row_factory = value
            else:
                object.__setattr__(self, name, value)

        def execute(self, sql, parameters=()):
            if sql == "BEGIN IMMEDIATE":
                raise sqlite3.OperationalError("database is locked")
            return self.inner.execute(sql, parameters)

        def rollback(self):
            return self.inner.rollback()

        def close(self):
            return self.inner.close()

    def busy_connect(database, *args, **kwargs):
        connection = real_connect(database, *args, **kwargs)
        if isinstance(database, str) and database.startswith("file:"):
            return connection
        return BusyConnection(connection)

    monkeypatch.setattr(publisher.sqlite3, "connect", busy_connect)

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        _run_execute(env["config"])

    assert fake.batch_calls == 0
    assert _db_status(env["db"]) == ("pending", 0)


# --- Этап G: incidents are classed, durable and never silently closed -------


def test_a_data_scan_that_finds_nothing_does_not_close_a_reconcile_incident():
    """Only the scan that can see a problem is allowed to declare it over."""
    state = {
        "incidents": {
            "reconcile:abc": {
                "class": "reconcile",
                "call_key": "",
                "code": "reconcile_duplicate_physical_call",
                "first_seen_at": "2026-08-14T09:00:00+00:00",
                "last_seen_at": "2026-08-14T09:00:00+00:00",
            },
            "mango:mango_office:call-7": {
                "class": "data",
                "call_key": "mango:mango_office:call-7",
                "code": "projection_dialogue_line_malformed",
                "first_seen_at": "2026-08-14T09:00:00+00:00",
                "last_seen_at": "2026-08-14T09:00:00+00:00",
            },
        }
    }

    health = publisher.apply_incidents(state, {})

    assert set(state["incidents"]) == {"reconcile:abc"}
    assert health["open_incidents"] == 1
    assert health["status"] != "green"


def test_a_reconcile_scan_that_succeeds_does_not_close_a_data_incident():
    state = {
        "incidents": {
            "reconcile:abc": {
                "class": "reconcile", "call_key": "", "code": "reconcile_x",
                "first_seen_at": "2026-08-14T09:00:00+00:00",
                "last_seen_at": "2026-08-14T09:00:00+00:00",
            },
            "mango:mango_office:call-7": {
                "class": "data", "call_key": "mango:mango_office:call-7",
                "code": "projection_dialogue_line_malformed",
                "first_seen_at": "2026-08-14T09:00:00+00:00",
                "last_seen_at": "2026-08-14T09:00:00+00:00",
            },
        }
    }

    publisher.apply_incidents(
        state, {}, incident_class=publisher.INCIDENT_CLASS_RECONCILE
    )

    assert set(state["incidents"]) == {"mango:mango_office:call-7"}


def test_two_calls_with_the_same_stable_identity_leave_an_incident_before_stopping(
    tmp_path, monkeypatch
):
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[record(), record(id=8)], fake=fake)

    with pytest.raises(RuntimeError, match="duplicate stable call_key"):
        publisher.run(["--config", str(env["config"])])

    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    incident = next(iter(persisted["incidents"].values()))
    dumped = json.dumps(persisted["incidents"], ensure_ascii=False)
    assert incident["code"] == "duplicate_stable_call_key"
    assert incident["class"] == publisher.INCIDENT_CLASS_DATA
    assert incident["first_seen_at"] and incident["last_seen_at"]
    # De-identified: the key is a hash, never the phone or the transcript.
    assert "+70000000000" not in dumped and "Добрый день" not in dumped
    assert fake.batch_calls == 0
    assert fake.values_calls == 0


def test_two_calls_with_whitespace_equivalent_recording_ids_stop_before_google(
    tmp_path, monkeypatch
):
    fake = FakeLiveGoogleGateway()
    env = _harness(
        tmp_path,
        monkeypatch,
        records=[
            record(source_recording_id="recording-7"),
            record(
                id=8,
                source_call_id="call-8",
                source_recording_id=" recording-7 ",
            ),
        ],
        fake=fake,
    )

    with pytest.raises(RuntimeError, match="duplicate normalized source_recording_id"):
        publisher.run(["--config", str(env["config"])])

    persisted = json.loads(env["state"].read_text(encoding="utf-8"))
    assert next(iter(persisted["incidents"].values()))["code"] == (
        "duplicate_source_recording_id"
    )
    assert fake.batch_calls == 0
    assert fake.values_calls == 0


def test_a_duplicate_identity_incident_keeps_its_first_seen_across_runs(
    tmp_path, monkeypatch
):
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[record(), record(id=8)], fake=fake)

    first_seen = []
    for _attempt in range(2):
        with pytest.raises(RuntimeError, match="duplicate stable call_key"):
            publisher.run(["--config", str(env["config"])])
        persisted = json.loads(env["state"].read_text(encoding="utf-8"))
        first_seen.append(
            [item["first_seen_at"] for item in persisted["incidents"].values()]
        )

    assert len(persisted["incidents"]) == 1
    assert first_seen[0] == first_seen[1]


def test_a_corrupt_incident_class_stops_the_run(tmp_path):
    path = tmp_path / "state.json"
    _owner_json(
        path,
        {
            "schema_version": publisher.STATE_SCHEMA,
            "destination_id": "dest",
            "entries": {},
            "incidents": {
                "mango:mango_office:call-7": {
                    "class": "invented",
                    "code": "projection_dialogue_line_malformed",
                    "first_seen_at": "2026-08-14T09:00:00+00:00",
                    "last_seen_at": "2026-08-14T09:00:00+00:00",
                }
            },
        },
    )

    with pytest.raises(RuntimeError, match="incident class"):
        publisher.load_state(path, "dest", required=True)


def test_a_dry_run_writes_no_google_batch_and_touches_no_working_database(
    tmp_path, monkeypatch
):
    broken = record(
        transcript_variants_json=json.dumps({"dialogue_lines": ["сломанная строка"]})
    )
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[broken], fake=fake)

    report = publisher.run(["--config", str(env["config"])])

    assert report["external_write"] is False
    assert report["data_error_codes"] == {"projection_dialogue_line_malformed": 1}
    assert fake.batch_calls == 0
    # The owner sidecar is the single local journal a dry run may touch.
    assert _db_status(env["db"]) == ("pending", 0)


def test_a_dry_run_incident_survives_the_process(tmp_path, monkeypatch):
    """Losing why a call cannot be published is worse than a local file write.

    A dry run reports the incident in its own output, but the report scrolls
    away.  What the owner needs on the next run is "this call has been broken
    since 14.08", which only a persisted sidecar can say.
    """
    broken = record(
        transcript_variants_json=json.dumps({"dialogue_lines": ["сломанная строка"]})
    )
    env = _harness(tmp_path, monkeypatch, records=[broken])

    first = publisher.run(["--config", str(env["config"])])
    stored = json.loads(Path(env["state"]).read_text(encoding="utf-8"))
    incident = next(iter(stored["incidents"].values()))
    second = publisher.run(["--config", str(env["config"])])
    again = json.loads(Path(env["state"]).read_text(encoding="utf-8"))

    assert first["external_write"] is False and second["external_write"] is False
    assert incident["code"] == "projection_dialogue_line_malformed"
    assert stored["health"]["open_incidents"] == 1
    # The second dry run keeps the moment it first broke and adds no duplicate.
    assert len(again["incidents"]) == 1
    assert next(iter(again["incidents"].values()))["first_seen_at"] == (
        incident["first_seen_at"]
    )


def test_a_failed_run_prints_no_exception_text_on_stdout(tmp_path, monkeypatch, capsys):
    """stdout goes to launchd logs: a message may carry a transcript or a phone."""
    secret = "СЕКРЕТ-МАРКЕР-9F3A"
    env = _harness(tmp_path, monkeypatch, records=[record()])

    def explode(*_args, **_kwargs):
        raise RuntimeError(f"provider echoed the prompt back: {secret}")

    monkeypatch.setattr(publisher, "load_calls", explode)

    exit_code = publisher.main(["--config", str(env["config"])])
    printed = capsys.readouterr().out

    assert exit_code == 1
    assert secret not in printed
    payload = json.loads(printed)
    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"].startswith("publish: RuntimeError: message_sha256=")


# --- Этап D: versioned fingerprint, closed balance, idempotent second run ---


def test_fingerprint_carries_the_analysis_result_and_every_contract_version():
    """A row is stale when the code that interprets it moves, too."""
    raw = record()
    baseline = publisher.call_projection(raw, {"mango_manager_1": "Иван Иванов"})

    for name in (
        "dialogue", "role_guard", "claim", "timezone",
        "normalizer_engine", "normalizer_ruleset", "projection",
    ):
        original = publisher.CONTRACT_FINGERPRINT[name]
        publisher.CONTRACT_FINGERPRINT[name] = f"{original}-bumped"
        try:
            bumped = publisher.call_projection(raw, {"mango_manager_1": "Иван Иванов"})
        finally:
            publisher.CONTRACT_FINGERPRINT[name] = original
        assert bumped["source_fingerprint"] != baseline["source_fingerprint"], name

    # A moved Analyse result moves the fingerprint too, on a proven call where
    # the projection really does publish the analysed content.
    trusted_raw = trusted_record()
    trusted_baseline = publisher.call_projection(
        trusted_raw, {"mango_manager_1": "Иван Иванов"}
    )
    changed = json.loads(trusted_raw["analysis_json"])
    changed["history_summary"] = "Другой конспект того же звонка."
    with_current_output_hash(changed)
    moved = publisher.call_projection(
        {**trusted_raw, "analysis_json": json.dumps(changed, ensure_ascii=False)},
        {"mango_manager_1": "Иван Иванов"},
    )
    assert moved["source_fingerprint"] != trusted_baseline["source_fingerprint"]
    # And an unchanged source still produces the very same fingerprint.
    assert (
        publisher.call_projection(raw, {"mango_manager_1": "Иван Иванов"})[
            "source_fingerprint"
        ]
        == baseline["source_fingerprint"]
    )


def test_every_done_call_lands_in_exactly_one_balance_category():
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    identities = {call["call_key"]: publisher.call_identity(record())}

    balance = publisher.publication_balance(
        identities, {call["call_key"]: call}, {}, state, [row], {call["call_key"]: 0}
    )

    assert balance["balanced"] is True
    assert balance["source_calls"] == 1
    assert balance["analysis_done"] == 1
    assert balance["verified_current"] == 1
    assert sum(balance[name] for name in publisher.BALANCE_CATEGORIES) == 1


def test_a_stale_row_is_published_not_verified_current():
    call = projected()
    row = with_number(call)
    state = state_for(call, row)
    identities = {call["call_key"]: publisher.call_identity(record())}
    # The row is in the sheet and proven, but the source it was built from moved.
    state["entries"][call["call_key"]]["source_fingerprint"] = "0" * 64

    balance = publisher.publication_balance(
        identities, {call["call_key"]: call}, {}, state, [row], {call["call_key"]: 0}
    )

    assert balance["published"] == 1
    assert balance["verified_current"] == 0
    assert balance["balanced"] is True


def test_a_call_that_cannot_be_projected_is_failed_with_incident():
    identity = publisher.call_identity(record())
    call_key = identity["call_key"]
    errors = {
        "call:" + publisher.call_key_digest(call_key): {
            "call_digest": publisher.call_key_digest(call_key),
            "code": "projection_dialogue_line_malformed",
        }
    }

    balance = publisher.publication_balance(
        {call_key: identity}, {}, errors, publisher.default_state(_destination()), [], {}
    )

    assert balance["failed_with_incident"] == 1
    assert balance["source_calls"] == 1
    assert balance["balanced"] is True
    assert balance["verified_current"] == 0


def test_a_missing_projection_without_an_incident_breaks_the_real_balance():
    identity = publisher.call_identity(record())
    call_key = identity["call_key"]

    balance = publisher.publication_balance(
        {call_key: identity}, {}, {}, publisher.default_state(_destination()), [], {}
    )

    assert balance["integrity_violations"]["unaccounted_source_calls"] == 1
    assert balance["balanced"] is False
    with pytest.raises(RuntimeError, match="balance does not close"):
        publisher.require_closed_balance(balance)


def test_orphan_ledger_entry_breaks_balance_without_exposing_its_key():
    call = projected()
    identity = publisher.call_identity(record())
    state = publisher.default_state(_destination())
    state["entries"]["mango:mango_office:orphan"] = {
        "status": "reserved", "display_number": 2,
    }

    balance = publisher.publication_balance(
        {call["call_key"]: identity}, {call["call_key"]: call}, {}, state, [], {}
    )

    assert balance["integrity_violations"]["orphan_state_entries"] == 1
    assert "mango:mango_office:orphan" not in json.dumps(balance)
    with pytest.raises(RuntimeError, match="balance does not close"):
        publisher.require_closed_balance(balance)


def test_an_unbalanced_report_blocks_the_external_write():
    closed = {name: 0 for name in publisher.BALANCE_CATEGORIES}
    closed.update(
        {"source_calls": 2, "analysis_done": 2, "verified_current": 2, "balanced": True}
    )

    # Negative control: a report whose categories add up does not block.
    publisher.require_closed_balance(closed)

    with pytest.raises(RuntimeError, match="balance does not close"):
        publisher.require_closed_balance({**closed, "balanced": False})
    with pytest.raises(RuntimeError, match="balance does not close"):
        # One call is finished but sits in no category at all.
        publisher.require_closed_balance({**closed, "source_calls": 3})


def test_main_returns_nonzero_for_unbalanced_shadow_report(monkeypatch, capsys):
    monkeypatch.setattr(
        publisher,
        "run",
        lambda _argv=None: {
            "status": "shadow_ok",
            "health": {"status": "green"},
            "balance": {"balanced": False},
            "analysis_cost": {"balanced": True},
        },
    )

    assert publisher.main([]) == 2
    assert json.loads(capsys.readouterr().out)["balance"]["balanced"] is False


def test_pending_call_participates_in_the_same_closed_balance():
    raw = record(
        analysis_status="pending", analysis_json=None,
        transcript_variants_json=None, transcript_text=None, duration_sec=None,
    )
    call = publisher.call_projection(raw, {})
    identity = publisher.call_identity(raw)

    balance = publisher.publication_balance(
        {call["call_key"]: identity}, {call["call_key"]: call}, {},
        publisher.default_state(_destination()), [], {},
    )

    assert balance["source_calls"] == 1
    assert balance["analysis_done"] == 0
    assert balance["reserved"] == 1
    assert sum(balance[name] for name in publisher.BALANCE_CATEGORIES) == 1
    publisher.require_closed_balance(balance)


def test_a_second_unchanged_run_issues_no_batch_update_and_no_duplicate(
    tmp_path, monkeypatch
):
    state = publisher.default_state(_destination())
    fake = FakeLiveGoogleGateway()
    env = _harness(tmp_path, monkeypatch, records=[record()], state=state, fake=fake)

    first = _run_execute(env["config"])
    rows_after_first = [list(row) for row in fake.rows]
    second = _run_execute(env["config"])

    assert first["status"] == "published" and first["external_write"] is True
    assert second["status"] == "no_change" and second["external_write"] is False
    assert fake.batch_calls == 1
    assert [list(row) for row in fake.rows] == rows_after_first
    assert len(fake.rows) == 1
    assert second["balance"]["balanced"] is True
    assert second["balance"]["verified_current"] == 1
