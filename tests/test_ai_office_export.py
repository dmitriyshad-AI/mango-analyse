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
from mango_mvp.services.export_ai_office import (
    _parse_analysis,
    build_call_insight_payload_for_record,
    push_call_insights,
)
from tests.test_dialogue_format import make_settings


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


class AIOfficeExportTest(unittest.TestCase):
    def test_parse_analysis_does_not_backfill_analysis_meta(self) -> None:
        settings = make_settings()
        call = CallRecord(
            id=500,
            source_file="/tmp/calls/call-500.mp3",
            source_filename="call-500.mp3",
            phone="+79990001122",
            manager_name="Иванов Иван",
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
        self.assertNotIn("analysis_meta", parsed)

    def test_build_call_insight_payload_for_record_maps_v2_analysis(self) -> None:
        settings = make_settings()
        call = CallRecord(
            id=501,
            source_file="/tmp/calls/call-501.mp3",
            source_filename="call-501.mp3",
            source_call_id="mango-501",
            phone="+79990001122",
            manager_name="Иванов Иван",
            direction="outbound",
            duration_sec=185.0,
            started_at=datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc),
            transcription_status="done",
            resolve_status="done",
            analysis_status="done",
            resolve_quality_score=91.0,
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
        self.assertEqual(payload["identity_hints"]["child_fio"], "Петр Иванов")
        self.assertEqual(payload["identity_hints"]["preferred_channel"], "telegram")
        self.assertEqual(payload["call_summary"]["history_short"], "Обсудили курс по математике.")
        self.assertEqual(payload["sales_insight"]["interests"]["subjects"], ["математика"])
        self.assertEqual(payload["sales_insight"]["lead_priority"], "warm")
        self.assertEqual(payload["sales_insight"]["follow_up_score"], 72)
        self.assertEqual(payload["quality_flags"]["mode"], "stereo")

    def test_build_call_insight_payload_for_record_migrates_legacy_analysis(self) -> None:
        settings = make_settings()
        call = CallRecord(
            id=777,
            source_file="/tmp/calls/call-777.mp3",
            source_filename="2026-03-19__10-00-00__79990002233__Леонов Алексей_777.mp3",
            source_call_id="mango-777",
            phone="+79990002233",
            manager_name="Леонов Алексей",
            direction="outbound",
            duration_sec=205.0,
            started_at=datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc),
            transcription_status="done",
            resolve_status="done",
            analysis_status="done",
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
        self.assertEqual(payload["sales_insight"]["follow_up_score"], 70)
        self.assertEqual(payload["sales_insight"]["next_step"]["action"], "Отправить материалы")
        self.assertEqual(payload["identity_hints"]["grade_current"], "10")
        self.assertIn("информатика", payload["sales_insight"]["interests"]["subjects"])

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
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["items"][0]["status"], "dry_run")
            saved = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["project_id"], "project-cli")


if __name__ == "__main__":
    unittest.main()
