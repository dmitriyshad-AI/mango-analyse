from __future__ import annotations

import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from mango_mvp import cli as cli_module
from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.services import dialogue_contract as contract
from mango_mvp.services.export_ai_office import (
    _parse_analysis,
    build_call_insight_payload,
    build_call_insight_payload_for_record,
    push_call_insights,
)
from tests import mango_provider_fixture as fx
from tests.test_dialogue_format import make_settings


# AI Office reads stored payloads through the same fail-closed role guard as
# Analyse and the Google publisher.  A call whose sides Mango proved keeps its
# content; the unproven case has its own tests at the end of this module.
DEFAULT_TURNS = (
    ("client", "right", "Нас интересует математика для 9 класса."),
    ("operator", "left", "Хорошо, я отправлю программу в Telegram."),
)


def proven_variants_json(
    source_call_id: str, turns=DEFAULT_TURNS, *, recording_id: str = fx.RECORDING_ID
) -> str:
    """Stored variants whose provider evidence really describes these lines."""
    variants = fx.proven_variants(turns)
    variants[contract.PROVIDER_EVIDENCE_FIELD] = fx.evidence_for_recording(
        turns, source_call_id=source_call_id, recording_id=recording_id
    )
    return json.dumps(variants, ensure_ascii=False)


def valid_v3_analysis(call: CallRecord) -> dict:
    if not call.source_recording_id:
        call.source_recording_id = fx.RECORDING_ID
    record = contract.call_record_view(call)
    dialogue = contract.build_dialogue_input(record)
    raw_value = "математика"
    turn = next((
        turn
        for turn in dialogue.turns
        if turn["speaker_kind"] == "client"
        and "математик" in str(turn["text"]).casefold()
    ), None)
    item_id = contract.canonical_item_key(raw_value) if turn else ""
    digest = contract.value_sha256(raw_value) if turn else ""
    claim_id = (
        contract.deterministic_claim_id(
            call_key=contract.call_key_for_record(record),
            field_path="structured_fields.interests.subjects",
            item_key=item_id,
            digest=digest,
            contract_version=contract.DETECTOR_CONTRACT_VERSION,
        )
        if turn else ""
    )
    fields = {
        "result": {"status": None, "detail": None},
        "people": {"parent_fio": None, "child_fio": None},
        "contacts": {
            "email": None, "preferred_channel": None, "phone_from_filename": None,
        },
        "student": {"grade_current": None, "school": None},
        "interests": {
            "products": [], "format": [],
            "subjects": [raw_value] if turn else [], "exam_targets": [],
        },
        "commercial": {
            "price_sensitivity": None, "budget": None, "discount_interest": None,
        },
        "objections": [],
        "next_step": {"action": None, "due": None},
        "lead_priority": "warm",
    }
    input_sha = "a" * 64
    meta = {
        "analysis_schema_version": contract.ANALYSIS_SCHEMA_VERSION_V3,
        "analysis_input_sha256": input_sha,
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
    }
    display = contract.build_display_fields(fields, [])
    payload = {
        "analysis_schema_version": contract.ANALYSIS_SCHEMA_VERSION_V3,
        "claim_contract_version": contract.CLAIM_CONTRACT_VERSION,
        "analysis_meta": meta,
        "quality_flags": {
            "dialogue_canonical_sha256": dialogue.canonical_sha256,
            "analysis_input_sha256": input_sha,
        },
        "dialogue_input": {
            "version": dialogue.version,
            "canonical_sha256": dialogue.canonical_sha256,
            "turn_count": len(dialogue.turns),
        },
        "history_summary": "Предмет: математика.",
        "structured_fields": fields,
        "display_fields": display,
        "crm_blocks": display,
        "claim_evidence": ([{
            "claim_id": claim_id,
            "field_path": "structured_fields.interests.subjects",
            "item_id": item_id,
            "evidence_type": "explicit",
            "support_type": "explicit",
            "source": "deterministic_detector",
            "contract_version": contract.DETECTOR_CONTRACT_VERSION,
            "turn_id": turn["turn_id"],
            "exact_quote": turn["text"],
            "timecode": turn["timecode"],
            "speaker_kind": turn["speaker_kind"],
            "start_sec": turn["start_sec"],
            "dialogue_sha256": dialogue.canonical_sha256,
            "raw_value": raw_value,
            "value_sha256": digest,
            "validation_status": "valid",
        }] if turn else []),
        "normalized_facts": [],
    }
    payload["analysis_meta"]["manager_output_sha256"] = (
        contract.manager_output_sha256(payload)
    )
    return payload


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class AIOfficeExportTest(unittest.TestCase):
    def test_parse_legacy_analysis_fails_closed(self) -> None:
        settings = make_settings()
        call = CallRecord(
            id=500,
            source_call_id="mango-500",
            source_file="/tmp/calls/call-500.mp3",
            source_filename="call-500.mp3",
            source_recording_id=fx.RECORDING_ID,
            phone="+79990001122",
            manager_name="Иванов Иван",
            transcript_variants_json=proven_variants_json("mango-500"),
            transcript_text="[00:01.0] Клиент: Нужна математика для 9 класса.",
            analysis_json=json.dumps(
                {
                    "summary": "Клиент интересуется математикой.",
                    "next_step": "Отправить программу",
                    "follow_up_score": 70,
                },
                ensure_ascii=False,
            ),
        )

        parsed = _parse_analysis(call, settings)

        self.assertEqual(parsed.get("analysis_schema_version"), "v2")
        self.assertIn("analysis_contract_invalid", parsed["review_reasons"])
        self.assertEqual(parsed["analysis_meta"], {})

    def test_build_call_insight_payload_for_record_maps_v2_analysis(self) -> None:
        settings = make_settings()
        call = CallRecord(
            id=501,
            source_file="/tmp/calls/call-501.mp3",
            source_filename="call-501.mp3",
            source_call_id="mango-501",
            source_recording_id=fx.RECORDING_ID,
            phone="+79990001122",
            manager_name="Иванов Иван",
            direction="outbound",
            duration_sec=185.0,
            started_at=datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc),
            transcription_status="done",
            resolve_status="done",
            analysis_status="done",
            resolve_quality_score=91.0,
            transcript_variants_json=proven_variants_json("mango-501"),
            transcript_text=(
                "[00:01.0] Клиент: Нас интересует математика для 9 класса.\n"
                "[00:03.0] Менеджер: Хорошо, я отправлю программу в Telegram.\n"
            ),
            analysis_json=json.dumps(
                {
                    "analysis_schema_version": "v2",
                    "history_summary": "19.03.2026 менеджер обсудил курс по математике.",
                    "history_short": "Обсудили курс по математике.",
                    "structured_fields": {
                        "people": {
                            "parent_fio": "Иванова Анна",
                            "child_fio": "Петр Иванов",
                        },
                        "contacts": {
                            "email": "family@example.com",
                            "phone_from_filename": "+79990001122",
                            "preferred_channel": "telegram",
                        },
                        "student": {
                            "grade_current": "9",
                            "school": "Школа 57",
                        },
                        "interests": {
                            "products": ["годовые курсы"],
                            "format": ["онлайн"],
                            "subjects": ["математика"],
                            "exam_targets": ["ОГЭ"],
                        },
                        "commercial": {
                            "price_sensitivity": "medium",
                            "budget": "до 100000",
                            "discount_interest": True,
                        },
                        "objections": ["цена"],
                        "next_step": {
                            "action": "Перезвонить",
                            "due": "на этой неделе",
                        },
                        "lead_priority": "warm",
                    },
                    "follow_up_score": 72,
                    "follow_up_reason": "Есть согласованный следующий шаг.",
                    "personal_offer": "Пробный модуль",
                    "pain_points": ["цена"],
                    "tags": ["follow_up"],
                    "evidence": [
                        {"speaker": "Клиент", "ts": "00:32.1", "text": "Интересует математика."}
                    ],
                    "quality_flags": {"mode": "stereo"},
                },
                ensure_ascii=False,
            ),
        )

        payload = build_call_insight_payload_for_record(call, settings)

        self.assertEqual(payload["schema_version"], "call_insight_v1")
        self.assertEqual(payload["source"]["system"], "mango_analyse")
        self.assertEqual(payload["source"]["call_record_id"], "501")
        self.assertEqual(payload["source"]["source_call_id"], "mango-501")
        self.assertEqual(payload["source"]["started_at"], "2026-03-19T10:00:00Z")
        self.assertEqual(payload["processing"]["resolve_quality_score"], 91.0)
        self.assertIsNone(payload["identity_hints"]["child_fio"])
        self.assertIsNone(payload["identity_hints"]["preferred_channel"])
        self.assertEqual(payload["sales_insight"]["interests"]["subjects"], [])
        self.assertIsNone(payload["sales_insight"]["lead_priority"])
        self.assertIsNone(payload["sales_insight"]["follow_up_score"])
        self.assertIn("analysis_contract_invalid", payload["raw_analysis"]["review_reasons"])
        self.assertEqual(payload["quality_flags"]["mode"], "stereo")

    def test_v3_exports_claim_evidence_without_legacy_evidence(self) -> None:
        call = CallRecord(
            id=503,
            source_call_id="mango-503",
            source_file="/tmp/calls/call-503.mp3",
            source_filename="call-503.mp3",
            transcript_variants_json=proven_variants_json("mango-503"),
            transcript_text="[00:01.0] Клиент: Нас интересует математика.",
        )
        analysis = valid_v3_analysis(call)
        call.analysis_json = json.dumps(analysis, ensure_ascii=False)

        summary = build_call_insight_payload_for_record(call, make_settings())["call_summary"]
        expected = analysis["claim_evidence"][0]

        self.assertNotIn("evidence", summary)
        self.assertEqual(
            summary["claim_evidence"],
            [{
                key: expected[key]
                for key in (
                    "field_path", "turn_id", "exact_quote", "timecode",
                    "speaker_kind", "claim_id",
                )
            }],
        )

    def test_v3_with_legacy_free_form_evidence_fails_closed(self) -> None:
        call = CallRecord(
            id=504,
            source_call_id="mango-504",
            source_file="/tmp/calls/call-504.mp3",
            source_filename="call-504.mp3",
            transcript_variants_json=proven_variants_json("mango-504"),
        )
        analysis = valid_v3_analysis(call)
        analysis["evidence"] = [{"speaker": "Клиент", "text": "устарело"}]
        analysis["analysis_meta"]["manager_output_sha256"] = (
            contract.manager_output_sha256(analysis)
        )
        call.analysis_json = json.dumps(analysis, ensure_ascii=False)

        payload = build_call_insight_payload_for_record(call, make_settings())

        self.assertEqual(payload["call_summary"]["claim_evidence"], [])
        self.assertIn("analysis_contract_invalid", payload["raw_analysis"]["review_reasons"])
        self.assertNotIn("устарело", json.dumps(payload, ensure_ascii=False))

    def test_a_self_consistent_but_false_stored_claim_is_rejected_on_read(self) -> None:
        call = CallRecord(
            id=505,
            source_call_id="mango-505",
            source_file="/tmp/calls/call-505.mp3",
            source_filename="call-505.mp3",
            transcript_variants_json=proven_variants_json("mango-505"),
        )
        analysis = valid_v3_analysis(call)
        raw_value = "физика"
        item_id = contract.canonical_item_key(raw_value)
        digest = contract.value_sha256(raw_value)
        evidence = analysis["claim_evidence"][0]
        evidence.update(
            {
                "item_id": item_id,
                "raw_value": raw_value,
                "value_sha256": digest,
                "claim_id": contract.deterministic_claim_id(
                    call_key=contract.call_key_for_record(contract.call_record_view(call)),
                    field_path=evidence["field_path"],
                    item_key=item_id,
                    digest=digest,
                    contract_version=contract.DETECTOR_CONTRACT_VERSION,
                ),
            }
        )
        for key in ("structured_fields", "display_fields", "crm_blocks"):
            analysis[key]["interests"]["subjects"] = [raw_value]

        guarded = contract.guard_stored_analysis(contract.call_record_view(call), analysis)

        self.assertEqual(guarded["structured_fields"], {})
        self.assertIn("analysis_contract_invalid", guarded["review_reasons"])

    def test_legacy_payload_cannot_smuggle_claim_evidence_through_builder(self) -> None:
        payload = build_call_insight_payload(
            CallRecord(id=504),
            {
                "analysis_schema_version": "v2",
                "history_summary": "Legacy.",
                "claim_evidence": [{
                    "field_path": "structured_fields.next_step.action",
                    "turn_id": "T0001",
                    "exact_quote": "Подложная цитата",
                    "timecode": "[00:01.0]",
                    "speaker_kind": "client",
                    "claim_id": "fake",
                }],
            },
        )

        self.assertEqual(payload["call_summary"]["claim_evidence"], [])

    def test_build_call_insight_payload_for_record_migrates_legacy_analysis(self) -> None:
        settings = make_settings()
        call = CallRecord(
            id=777,
            source_file="/tmp/calls/call-777.mp3",
            source_filename="2026-03-19__10-00-00__79990002233__Леонов Алексей_777.mp3",
            source_call_id="mango-777",
            source_recording_id=fx.RECORDING_ID,
            phone="+79990002233",
            manager_name="Леонов Алексей",
            direction="outbound",
            duration_sec=205.0,
            started_at=datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc),
            transcription_status="done",
            resolve_status="done",
            analysis_status="done",
            transcript_variants_json=proven_variants_json(
                "mango-777",
                turns=(
                    ("client", "right", "Нас интересует информатика для 10 класса."),
                    ("operator", "left", "Хорошо, отправим материалы в Telegram."),
                ),
            ),
            transcript_text=(
                "[00:01.0] Клиент: Нас интересует информатика для 10 класса.\n"
                "[00:03.0] Менеджер: Хорошо, отправим материалы в Telegram.\n"
            ),
            analysis_json=json.dumps(
                {
                    "summary": "Клиент интересуется информатикой.",
                    "next_step": "Отправить материалы в Telegram",
                    "follow_up_score": 70,
                    "tags": ["follow_up"],
                },
                ensure_ascii=False,
            ),
        )

        payload = build_call_insight_payload_for_record(call, settings)

        self.assertEqual(payload["raw_analysis"]["analysis_schema_version"], "v2")
        self.assertIn("analysis_contract_invalid", payload["raw_analysis"]["review_reasons"])
        self.assertIsNone(payload["sales_insight"]["follow_up_score"])
        self.assertIsNone(payload["sales_insight"]["next_step"]["action"])
        self.assertIsNone(payload["identity_hints"]["grade_current"])
        self.assertEqual(payload["sales_insight"]["interests"]["subjects"], [])

    def test_push_call_insights_posts_to_ai_office_and_handles_duplicates(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_ai_office_push_") as td:
            db_path = Path(td) / "export.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                ai_office_api_base_url="https://api.fotonai.online",
                ai_office_api_key="ai-office-token",
                ai_office_timeout_sec=12,
            )
            init_db(settings)
            session_factory = build_session_factory(settings)

            with session_factory() as session:
                session.add_all(
                    [
                        CallRecord(
                            id=901,
                            source_file=str(Path(td) / "call-901.mp3"),
                            source_filename="call-901.mp3",
                            source_call_id="mango-901",
                            phone="+79991110000",
                            manager_name="Анна",
                            transcription_status="done",
                            resolve_status="done",
                            analysis_status="done",
                            analysis_json=json.dumps(
                                {
                                    "analysis_schema_version": "v2",
                                    "history_summary": "Первая карточка.",
                                    "structured_fields": {
                                        "people": {},
                                        "contacts": {},
                                        "student": {},
                                        "interests": {},
                                        "commercial": {},
                                        "objections": [],
                                        "next_step": {},
                                        "lead_priority": "warm",
                                    },
                                    "follow_up_score": 55,
                                },
                                ensure_ascii=False,
                            ),
                        ),
                        CallRecord(
                            id=902,
                            source_file=str(Path(td) / "call-902.mp3"),
                            source_filename="call-902.mp3",
                            source_call_id="mango-902",
                            phone="+79992220000",
                            manager_name="Борис",
                            transcription_status="done",
                            resolve_status="done",
                            analysis_status="done",
                            analysis_json=json.dumps(
                                {
                                    "analysis_schema_version": "v2",
                                    "history_summary": "Вторая карточка.",
                                    "structured_fields": {
                                        "people": {},
                                        "contacts": {},
                                        "student": {},
                                        "interests": {},
                                        "commercial": {},
                                        "objections": [],
                                        "next_step": {},
                                        "lead_priority": "cold",
                                    },
                                    "follow_up_score": 20,
                                },
                                ensure_ascii=False,
                            ),
                        ),
                        CallRecord(
                            id=903,
                            source_file=str(Path(td) / "call-903.mp3"),
                            source_filename="call-903.mp3",
                            source_call_id="mango-903",
                            phone="+79993330000",
                            manager_name="Вера",
                            transcription_status="done",
                            resolve_status="done",
                            analysis_status="pending",
                            analysis_json=json.dumps(
                                {"analysis_schema_version": "v2", "history_summary": "Не должен уйти"},
                                ensure_ascii=False,
                            ),
                        ),
                    ]
                )
                session.flush()
                for call_id in (901, 902):
                    call = session.get(CallRecord, call_id)
                    recording_id = f"recording-{call_id}"
                    call.source_recording_id = recording_id
                    call.transcript_variants_json = proven_variants_json(
                        str(call.source_call_id), recording_id=recording_id
                    )
                    call.transcript_text = "\n".join(fx.dialogue_lines(DEFAULT_TURNS))
                    call.analysis_json = json.dumps(
                        valid_v3_analysis(call), ensure_ascii=False
                    )
                session.commit()

            posted = []

            def fake_post(url, json=None, headers=None, timeout=None):
                posted.append(
                    {
                        "url": url,
                        "json": json,
                        "headers": headers,
                        "timeout": timeout,
                    }
                )
                if json["source"]["call_record_id"] == "901":
                    return _FakeResponse(201, {"insight": {"id": "insight-901"}})
                return _FakeResponse(409, {"detail": "already exists"})

            with patch("mango_mvp.services.export_ai_office.requests.post", side_effect=fake_post):
                with session_factory() as session:
                    result = push_call_insights(
                        session,
                        settings,
                        project_id="project-123",
                        limit=100,
                    )

            self.assertEqual(result["selected"], 2)
            self.assertEqual(result["created"], 1)
            self.assertEqual(result["duplicates"], 1)
            self.assertEqual(result["failed"], 0)
            self.assertNotIn("source_call_id", result["items"][0])
            self.assertNotIn("source_filename", result["items"][0])
            self.assertEqual(len(posted), 2)
            self.assertEqual(
                posted[0]["url"],
                "https://api.fotonai.online/api/projects/project-123/calls/insights",
            )
            self.assertEqual(posted[0]["headers"]["X-API-Key"], "ai-office-token")
            self.assertEqual(posted[0]["timeout"], 12)
            self.assertEqual(posted[0]["json"]["source"]["system"], "mango_analyse")

    def test_push_ai_office_insights_cli_writes_summary_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_ai_office_cli_") as td:
            db_path = Path(td) / "cli.db"
            out_path = Path(td) / "push_result.json"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                ai_office_api_base_url="http://localhost:8001/api",
                ai_office_api_key="token",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        id=1001,
                        source_file=str(Path(td) / "call-1001.mp3"),
                        source_filename="call-1001.mp3",
                        source_call_id="mango-1001",
                        phone="+79994440000",
                        manager_name="Иван",
                        transcription_status="done",
                        resolve_status="done",
                        analysis_status="done",
                        analysis_json=json.dumps(
                            {
                                "analysis_schema_version": "v2",
                                "history_summary": "CLI экспорт.",
                                "structured_fields": {
                                    "people": {},
                                    "contacts": {},
                                    "student": {},
                                    "interests": {},
                                    "commercial": {},
                                    "objections": [],
                                    "next_step": {},
                                    "lead_priority": "warm",
                                },
                                "follow_up_score": 61,
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
                session.flush()
                call = session.get(CallRecord, 1001)
                call.transcript_variants_json = proven_variants_json("mango-1001")
                call.transcript_text = "\n".join(fx.dialogue_lines(DEFAULT_TURNS))
                call.analysis_json = json.dumps(valid_v3_analysis(call), ensure_ascii=False)
                session.commit()

            args = Namespace(
                project_id="project-cli",
                limit=10,
                ids_in=None,
                include_not_done=False,
                dry_run=True,
                out=str(out_path),
            )

            with patch.object(cli_module, "get_settings", return_value=settings):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    rc = cli_module.cmd_push_ai_office_insights(args)

            self.assertEqual(rc, 0)
            payload = json.loads(buffer.getvalue())
            self.assertTrue(out_path.exists())
            self.assertEqual(payload["selected"], 1)
            self.assertEqual(payload["created"], 0)
            self.assertEqual(payload["duplicates"], 0)
            self.assertEqual(payload["failed"], 0)
            self.assertEqual(payload["skipped"], 0)
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["items"][0]["status"], "dry_run")
            saved = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["project_id"], "project-cli")

    def _unproven_call(self) -> CallRecord:
        """The real production shape: channel roles guessed from the words."""
        return CallRecord(
            id=502,
            source_call_id="mango-502",
            source_file="/tmp/calls/call-502.mp3",
            source_filename="call-502.mp3",
            phone="+79990001122",
            manager_name="Иванов Иван",
            direction="outbound",
            duration_sec=185.0,
            started_at=datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc),
            transcription_status="done",
            resolve_status="done",
            analysis_status="done",
            transcript_variants_json=json.dumps(
                {
                    "mode": "stereo",
                    "role_mapping": {
                        "status": "confirmed_multi_signal",
                        "confirmed": True,
                        "manager_quality_allowed": True,
                        "topology": "simple_two_party",
                        "left": "manager",
                        "right": "client",
                    },
                    "dialogue_lines": [
                        "[00:01.0] Дорожка правая: Нас интересует математика.",
                        "[00:03.0] Дорожка левая: Отправлю программу в Telegram.",
                    ],
                },
                ensure_ascii=False,
            ),
            transcript_text="[00:01.0] Клиент: Нас интересует математика.",
            analysis_json=json.dumps(
                {
                    "analysis_schema_version": "v2",
                    "history_summary": "Клиент Иванова Анна согласилась оплатить.",
                    "structured_fields": {
                        "people": {"parent_fio": "Иванова Анна",
                                   "child_fio": "Петр Иванов"},
                        "contacts": {"email": "family@example.com",
                                     "preferred_channel": "telegram"},
                        "student": {"grade_current": "9", "school": "Школа 57"},
                        "interests": {"subjects": ["математика"]},
                        "commercial": {"budget": "до 100000"},
                        "objections": ["цена"],
                        "next_step": {"action": "Перезвонить", "due": "на неделе"},
                        "lead_priority": "warm",
                    },
                    "follow_up_score": 72,
                    "personal_offer": "Пробный модуль",
                    "tags": ["follow_up"],
                    "evidence": [
                        {"speaker": "Клиент", "ts": "00:32.1", "text": "Интересует."}
                    ],
                    "quality_flags": {"mode": "stereo"},
                },
                ensure_ascii=False,
            ),
        )

    def test_an_unproven_call_pushes_no_role_dependent_claim_to_ai_office(self) -> None:
        payload = build_call_insight_payload_for_record(
            self._unproven_call(), make_settings()
        )

        self.assertIsNone(payload["identity_hints"]["parent_fio"])
        self.assertIsNone(payload["identity_hints"]["child_fio"])
        self.assertIsNone(payload["identity_hints"]["email"])
        self.assertIsNone(payload["identity_hints"]["grade_current"])
        self.assertIsNone(payload["identity_hints"]["school"])
        self.assertIsNone(payload["sales_insight"]["next_step"]["action"])
        self.assertIsNone(payload["sales_insight"]["next_step"]["due"])
        self.assertIsNone(payload["sales_insight"]["lead_priority"])
        self.assertIsNone(payload["sales_insight"]["follow_up_score"])
        self.assertIsNone(payload["sales_insight"]["personal_offer"])
        self.assertEqual(payload["sales_insight"]["objections"], [])
        self.assertEqual(payload["sales_insight"]["interests"]["subjects"], [])
        self.assertEqual(payload["call_summary"]["claim_evidence"], [])
        self.assertTrue(payload["raw_analysis"]["needs_review"])

        dumped = json.dumps(payload, ensure_ascii=False)
        for leaked in ("Иванова Анна", "Петр Иванов", "family@example.com",
                       "Школа 57", "Перезвонить", "Пробный модуль", "до 100000"):
            self.assertNotIn(leaked, dumped)

    def test_push_skips_untrusted_analysis_without_network_call(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_ai_office_untrusted_") as td:
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{Path(td) / 'calls.db'}",
                ai_office_api_base_url="https://api.fotonai.online",
                ai_office_api_key="token",
            )
            init_db(settings)
            factory = build_session_factory(settings)
            with factory() as session:
                session.add(self._unproven_call())
                session.commit()

            with patch("mango_mvp.services.export_ai_office.requests.post") as post:
                with factory() as session:
                    report = push_call_insights(
                        session, settings, project_id="project", limit=10
                    )

            self.assertEqual(report["selected"], 1)
            self.assertEqual(report["skipped"], 1)
            self.assertEqual(report["failed"], 0)
            self.assertEqual(report["items"][0]["status"], "skipped_untrusted_analysis")
            post.assert_not_called()

    def test_ai_office_error_report_never_echoes_response_body(self) -> None:
        secret = "клиент +79990000000 просил вернуть 60000"
        with tempfile.TemporaryDirectory(prefix="mango_ai_office_error_") as td:
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{Path(td) / 'calls.db'}",
                ai_office_api_base_url="https://api.fotonai.online",
                ai_office_api_key="token",
            )
            init_db(settings)
            factory = build_session_factory(settings)
            call = CallRecord(
                id=1901,
                source_call_id="mango-1901",
                source_file=str(Path(td) / "call.mp3"),
                source_filename="call.mp3",
                transcript_variants_json=proven_variants_json("mango-1901"),
                transcript_text="\n".join(fx.dialogue_lines(DEFAULT_TURNS)),
                analysis_status="done",
            )
            call.analysis_json = json.dumps(valid_v3_analysis(call), ensure_ascii=False)
            with factory() as session:
                session.add(call)
                session.commit()

            with patch(
                "mango_mvp.services.export_ai_office.requests.post",
                return_value=_FakeResponse(500, {"detail": secret}),
            ):
                with factory() as session:
                    report = push_call_insights(
                        session, settings, project_id="project", limit=10
                    )

            self.assertEqual(report["failed"], 1)
            dumped = json.dumps(report, ensure_ascii=False)
            self.assertNotIn(secret, dumped)
            self.assertNotIn("+79990000000", dumped)
            self.assertNotIn("mango-1901", dumped)
            self.assertNotIn("call.mp3", dumped)


if __name__ == "__main__":
    unittest.main()
