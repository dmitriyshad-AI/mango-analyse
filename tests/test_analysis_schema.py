from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from mango_mvp import cli as cli_module
from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.services.analyze import AnalyzeService
from mango_mvp.services.sync_amocrm import _build_custom_fields
from tests.test_dialogue_format import make_settings


class AnalysisSchemaTest(unittest.TestCase):
    def test_analysis_schema_version_detection(self) -> None:
        service = AnalyzeService(make_settings())
        self.assertEqual(service.analysis_schema_version({}), "v1")
        self.assertEqual(service.analysis_schema_version({"analysis_schema_version": "V2"}), "v2")

    def test_migrate_analysis_payload_from_v1(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            source_file="/tmp/call2.mp3",
            source_filename="2026-02-10__13-02-47__79161042660__Леонов Алексей_33.mp3",
            phone="+79161042660",
            manager_name="Леонов Алексей",
            transcript_text=(
                "[00:01.0] Клиент: Интересует математика для 10 класса.\n"
                "[00:02.4] Менеджер: Мы можем отправить программу в Telegram.\n"
            ),
        )
        old_payload = {
            "summary": "Клиент спрашивает программу.",
            "next_step": "Отправить программу",
            "follow_up_score": 70,
        }

        migrated = service.migrate_analysis_payload(call, old_payload)
        self.assertEqual(migrated.get("analysis_schema_version"), "v2")
        self.assertEqual(migrated.get("next_step"), "Отправить программу")
        self.assertEqual(migrated["crm_blocks"]["student"]["grade_current"], "10")
        self.assertIn("математика", migrated["crm_blocks"]["interests"]["subjects"])

    def test_normalize_analysis_emits_v2_blocks_and_legacy(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            source_file="/tmp/call.mp3",
            source_filename="2026-02-10__13-02-47__79161042660__Леонов Алексей_33.mp3",
            phone="+79161042660",
            manager_name="Леонов Алексей",
        )
        transcript = (
            "[00:01.2] Клиент: Добрый день, интересует онлайн курс по физике для 9 класса.\n"
            "[00:03.8] Менеджер: Да, подскажите, готовитесь к ОГЭ?\n"
            "[00:04.9] Клиент: Да, но цена важна, если дорого, не потянем.\n"
            "[00:07.1] Менеджер: Можем отправить материалы в Telegram и перезвонить завтра.\n"
        )
        raw = {
            "summary": "Клиент интересуется курсом и просит материалы.",
            "next_step": "Перезвонить завтра",
            "follow_up_score": 72,
            "tags": ["needs_follow_up"],
        }

        analysis = service._normalize_analysis(call, transcript, raw)

        self.assertEqual(analysis.get("analysis_schema_version"), "v2")
        self.assertEqual(analysis.get("summary"), analysis.get("history_short"))
        self.assertIn("менеджер леонов алексей", analysis.get("history_summary", "").lower())
        self.assertIn("Договорились:", analysis.get("history_summary", ""))
        self.assertEqual(analysis.get("structured_fields"), analysis.get("crm_blocks"))
        self.assertEqual(
            analysis["crm_blocks"]["contacts"]["phone_from_filename"],
            "+79161042660",
        )
        self.assertEqual(analysis["crm_blocks"]["student"]["grade_current"], "9")
        self.assertIn("физика", analysis["crm_blocks"]["interests"]["subjects"])
        self.assertIn("ОГЭ", analysis["crm_blocks"]["interests"]["exam_targets"])
        self.assertIn("цена", analysis["crm_blocks"]["objections"])
        self.assertEqual(analysis["crm_blocks"]["next_step"]["action"], "Перезвонить завтра")
        self.assertEqual(analysis["next_step"], "Перезвонить завтра")
        self.assertEqual(analysis["follow_up_score"], 72)
        self.assertGreaterEqual(len(analysis.get("evidence") or []), 1)

    def test_history_summary_rewrites_dialogue_dump_into_crm_story(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            source_file="/tmp/call_story.mp3",
            source_filename="2026-03-08__11-22-33__79161234567__Петрова Анна_44.mp3",
            phone="+79161234567",
            manager_name="Петрова Анна",
            started_at=datetime(2026, 3, 8, 11, 22, 33),
        )
        transcript = (
            "[00:01.1] Клиент: Нужна математика онлайн для 8 класса.\n"
            "[00:04.0] Менеджер: Запишу и отправлю программу на почту.\n"
        )
        raw = {
            "history_summary": transcript,
            "summary": transcript,
            "next_step": "Отправить программу на email и перезвонить завтра",
        }

        analysis = service._normalize_analysis(call, transcript, raw)
        summary_text = analysis.get("history_summary", "")
        self.assertIn("08.03.2026 11:22", summary_text)
        self.assertIn("менеджер Петрова Анна", summary_text)
        self.assertIn("Договорились:", summary_text)
        self.assertNotIn("[00:01.1]", summary_text)

    def test_custom_fields_fallback_to_v2_blocks(self) -> None:
        settings = replace(
            make_settings(),
            amocrm_interests_field_id=1,
            amocrm_student_grade_field_id=2,
            amocrm_target_product_field_id=3,
            amocrm_budget_field_id=4,
            amocrm_timeline_field_id=5,
            amocrm_next_step_field_id=6,
            amocrm_followup_score_field_id=7,
        )
        analysis = {
            "follow_up_score": 83,
            "crm_blocks": {
                "student": {"grade_current": "10"},
                "interests": {
                    "products": ["годовые курсы"],
                    "format": ["онлайн"],
                    "subjects": ["математика", "физика"],
                    "exam_targets": ["ЕГЭ"],
                },
                "commercial": {"budget": "до 120000"},
                "next_step": {"action": "Отправить материалы", "due": "на этой неделе"},
            },
        }

        fields = _build_custom_fields(settings, analysis)
        by_id = {item["field_id"]: item["values"][0]["value"] for item in fields}

        self.assertIn("математика", by_id[1])
        self.assertEqual(by_id[2], "10")
        self.assertEqual(by_id[3], "годовые курсы")
        self.assertEqual(by_id[4], "до 120000")
        self.assertEqual(by_id[5], "на этой неделе")
        self.assertEqual(by_id[6], "Отправить материалы")
        self.assertEqual(by_id[7], "83")

    def test_analyze_exports_history_and_structured_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_export_") as td:
            root = Path(td)
            calls_dir = root / "calls"
            calls_dir.mkdir(parents=True, exist_ok=True)
            source = calls_dir / "a.mp3"
            source.write_bytes(b"")
            export_dir = root / "transcripts"
            db_path = root / "analyze.db"

            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                analyze_provider="mock",
                transcript_export_dir=str(export_dir),
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(source),
                        source_filename=source.name,
                        transcription_status="done",
                        resolve_status="done",
                        analysis_status="pending",
                        transcript_text=(
                            "[00:01.0] Клиент: Интересует математика для 9 класса.\n"
                            "[00:02.4] Менеджер: Отправим программу в Telegram.\n"
                        ),
                        transcript_manager="Отправим программу в Telegram.",
                        transcript_client="Интересует математика для 9 класса.",
                        transcript_variants_json=json.dumps({"mode": "stereo", "warnings": []}, ensure_ascii=False),
                    )
                )
                session.commit()

            service = AnalyzeService(settings)
            with session_factory() as session:
                result = service.run(session, limit=10)
            self.assertEqual(result["success"], 1)

            base = export_dir / calls_dir.name / f"{source.stem}"
            summary_path = base.with_name(f"{source.stem}_history_summary.txt")
            structured_path = base.with_name(f"{source.stem}_structured_fields.json")
            self.assertTrue(summary_path.exists())
            self.assertTrue(structured_path.exists())

            summary_text = summary_path.read_text(encoding="utf-8").strip()
            self.assertTrue(summary_text)
            structured = json.loads(structured_path.read_text(encoding="utf-8"))
            self.assertIsInstance(structured, dict)
            self.assertIn("interests", structured)

    def test_export_crm_fields_cli(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_crm_export_") as td:
            root = Path(td)
            db_path = root / "crm_export.db"
            out_path = root / "crm_fields.csv"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(root / "call.mp3"),
                        source_filename="call.mp3",
                        transcription_status="done",
                        resolve_status="done",
                        analysis_status="done",
                        transcript_text="[00:01.0] Клиент: Интересует математика.",
                        analysis_json=json.dumps(
                            {
                                "analysis_schema_version": "v2",
                                "history_summary": "Клиент интересуется математикой.",
                                "structured_fields": {
                                    "people": {"parent_fio": "Иванова Анна", "child_fio": "Петр Иванов"},
                                    "contacts": {"email": "test@example.com", "preferred_channel": "telegram"},
                                    "student": {"grade_current": "9", "school": "Школа 1"},
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
                                    "next_step": {"action": "Перезвонить", "due": "2026-03-10"},
                                    "lead_priority": "warm",
                                },
                                "follow_up_score": 72,
                                "follow_up_reason": "Нужен контакт на этой неделе",
                                "tags": ["follow_up"],
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
                session.commit()

            old_get_settings = cli_module.get_settings
            try:
                cli_module.get_settings = lambda: settings  # type: ignore[assignment]
                code = cli_module.cmd_export_crm_fields(
                    Namespace(out=str(out_path), limit=1000, only_done=True)
                )
            finally:
                cli_module.get_settings = old_get_settings  # type: ignore[assignment]

            self.assertEqual(code, 0)
            content = out_path.read_text(encoding="utf-8")
            self.assertIn("history_summary", content)
            self.assertIn("Иванова Анна", content)
            self.assertIn("годовые курсы", content)
            self.assertIn("Нужен контакт на этой неделе", content)

    def test_analyze_provider_codex_cli_routes_to_codex_method(self) -> None:
        settings = replace(make_settings(), analyze_provider="codex_cli")
        service = AnalyzeService(settings)
        call = CallRecord(
            source_file="/tmp/codex.mp3",
            source_filename="codex.mp3",
            manager_name="Смирнов Иван",
        )
        text = (
            "Клиент интересуется математикой и информатикой, уточняет формат онлайн и оффлайн, "
            "обсуждает стоимость, просит отправить программу на почту и перезвонить завтра в 18:00."
        )

        with patch.object(service, "_codex_cli_analysis", return_value={"summary": "ok"}) as mocked:
            payload = service._analyze_text(call, text)
        self.assertEqual(payload.get("summary"), "ok")
        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
