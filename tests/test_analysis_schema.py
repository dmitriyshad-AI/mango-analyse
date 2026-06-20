from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from mango_mvp import cli as cli_module
from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.services.analyze import (
    NON_CONVERSATION_ADVISORY_ENV,
    AnalyzeService,
    build_analysis_migration_call_snapshot,
    migrate_analysis_payload,
)
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
        self.assertNotIn("analysis_meta", migrated)

    def test_top_level_migrate_analysis_payload_matches_service_adapter(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            source_file="/tmp/call3.mp3",
            source_filename="2026-02-10__13-02-47__79161042660__Леонов Алексей_33.mp3",
            phone="+79161042660",
            manager_name="Леонов Алексей",
            transcript_text=(
                "[00:01.0] Клиент: Интересует математика для 10 класса.\n"
                "[00:02.4] Менеджер: Мы можем отправить программу в Telegram.\n"
            ),
        )
        payload = {
            "summary": "Клиент спрашивает программу.",
            "next_step": "Отправить программу",
            "follow_up_score": 70,
        }

        service_result = service.migrate_analysis_payload(call, payload)
        top_level_result = migrate_analysis_payload(
            build_analysis_migration_call_snapshot(call),
            payload,
            non_conversation_advisory_enabled=False,
        )

        self.assertEqual(top_level_result, service_result)

    def test_top_level_migrate_analysis_payload_is_process_pool_picklable(self) -> None:
        calls = [
            CallRecord(
                source_file=f"/tmp/call-pool-{idx}.mp3",
                source_filename=f"2026-02-10__13-02-47__7916104266{idx}__Леонов Алексей_33.mp3",
                phone=f"+7916104266{idx}",
                manager_name="Леонов Алексей",
                transcript_text=(
                    "[00:01.0] Клиент: Интересует математика для 10 класса.\n"
                    "[00:02.4] Менеджер: Можно отправить материалы.\n"
                ),
            )
            for idx in range(2)
        ]
        snapshots = [build_analysis_migration_call_snapshot(call) for call in calls]
        payloads = [
            {"summary": "Клиент спрашивает программу.", "next_step": "Отправить программу"},
            {"summary": "Клиент просит материалы.", "next_step": "Отправить материалы"},
        ]

        with ProcessPoolExecutor(max_workers=2) as executor:
            migrated = list(executor.map(migrate_analysis_payload, snapshots, payloads))

        self.assertEqual([item["analysis_schema_version"] for item in migrated], ["v2", "v2"])
        self.assertEqual(migrated[0]["next_step"], "Отправить программу")
        self.assertEqual(migrated[1]["next_step"], "Отправить материалы")

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

    def test_analysis_meta_uses_provider_model_and_actual_prompt_version(self) -> None:
        codex_service = AnalyzeService(
            replace(
                make_settings(),
                analyze_provider="codex_cli",
                codex_analyze_model="gpt-5.4-mini",
            )
        )
        codex_meta = codex_service._build_analysis_meta(
            {"quality_flags": {"analyze_prompt_version": "v7"}}
        )
        self.assertEqual(codex_meta["analysis_provider"], "codex_cli")
        self.assertEqual(codex_meta["analysis_model"], "gpt-5.4-mini")
        self.assertEqual(codex_meta["analysis_prompt_version"], "v7")

        openai_service = AnalyzeService(
            replace(
                make_settings(),
                analyze_provider="openai",
                openai_analysis_model="gpt-4o-mini",
            )
        )
        openai_meta = openai_service._build_analysis_meta({})
        self.assertEqual(openai_meta["analysis_provider"], "openai")
        self.assertEqual(openai_meta["analysis_model"], "gpt-4o-mini")

    def test_analysis_runtime_metadata_is_additive_top_level(self) -> None:
        service = AnalyzeService(
            replace(
                make_settings(),
                analyze_provider="codex_cli",
                codex_analyze_model="gpt-5.4-mini",
            )
        )
        analysis = {
            "history_summary": "Клиент запросил программу.",
            "quality_flags": {
                "analyze_prompt_profile": "compact",
                "analyze_prompt_truncated": True,
                "analyze_transcript_chars_prompt": 1234,
                "analyze_prompt_version": "v7",
            },
        }
        analysis["analysis_meta"] = service._build_analysis_meta(analysis)

        enriched = service._with_analysis_runtime_metadata(analysis)

        self.assertEqual(enriched["analyze_model"], "gpt-5.4-mini")
        self.assertEqual(enriched["analyze_prompt_profile"], "compact")
        self.assertTrue(enriched["analyze_prompt_truncated"])
        self.assertEqual(enriched["analyze_prompt_chars"], 1234)
        self.assertEqual(enriched["quality_flags"], analysis["quality_flags"])
        self.assertEqual(enriched["analysis_meta"], analysis["analysis_meta"])

    def test_normalize_analysis_preserves_prompt_quality_flags(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            source_file="/tmp/call_prompt_metrics.mp3",
            source_filename="2026-02-10__13-02-47__79161042660__Леонов Алексей_34.mp3",
            phone="+79161042660",
            manager_name="Леонов Алексей",
        )
        transcript = (
            "[00:01.2] Клиент: Да, да, да.\n"
            "[00:02.0] Клиент: Да, да, да.\n"
            "[00:03.8] Менеджер: Отправим программу по математике в Telegram.\n"
        )
        raw = {
            "history_summary": "Клиент запросил программу по математике.",
            "structured_fields": {
                "people": {"parent_fio": None, "child_fio": None},
                "contacts": {"email": None, "phone_from_filename": None, "preferred_channel": "telegram"},
                "student": {"grade_current": None, "school": None},
                "interests": {"products": [], "format": [], "subjects": ["математика"], "exam_targets": []},
                "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
                "objections": [],
                "next_step": {"action": "Отправить материалы", "due": None},
                "lead_priority": "warm",
            },
            "quality_flags": {
                "analyze_prompt_profile": "compact",
                "analyze_prompt_compacted": True,
                "analyze_transcript_chars_original": 120,
                "analyze_transcript_chars_prompt": 96,
                "analyze_transcript_chars_saved": 24,
                "analyze_prompt_timestamps_removed_lines": 3,
            },
            "tags": [],
        }

        analysis = service._normalize_analysis(call, transcript, raw)
        self.assertTrue(analysis["quality_flags"]["analyze_prompt_compacted"])
        self.assertEqual(analysis["quality_flags"]["analyze_prompt_profile"], "compact")
        self.assertEqual(analysis["quality_flags"]["analyze_transcript_chars_saved"], 24)
        self.assertEqual(analysis["quality_flags"]["analyze_prompt_timestamps_removed_lines"], 3)

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

    def test_normalize_analysis_reclassifies_false_non_conversation_service_call(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            source_file="/tmp/service.mp3",
            source_filename="2026-03-08__11-22-33__79161234567__Петрова Анна_44.mp3",
            phone="+79161234567",
            manager_name="Петрова Анна",
            started_at=datetime(2026, 3, 8, 11, 22, 33),
        )
        transcript = (
            "[00:01.1] Менеджер: Подскажите, по оплате и расписанию на следующую неделю все удобно?\n"
            "[00:04.0] Клиент: Оплату внесем завтра, а одно занятие нужно перенести.\n"
        )
        raw = {
            "history_summary": "Звонок без содержательного диалога.",
            "structured_fields": {
                "people": {},
                "contacts": {},
                "student": {},
                "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
                "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
                "objections": [],
                "next_step": {"action": "Уточнить информацию и сообщить клиенту", "due": None},
                "lead_priority": "cold",
            },
            "follow_up_score": 0,
            "follow_up_reason": "Нет содержательного диалога.",
            "tags": ["non_conversation"],
        }

        analysis = service._normalize_analysis(call, transcript, raw)
        self.assertNotIn("non_conversation", analysis["tags"])
        self.assertIn("service_call", analysis["tags"])
        self.assertEqual(analysis["quality_flags"]["call_type"], "service_call")
        self.assertEqual(analysis["next_step"], "Уточнить информацию и сообщить клиенту")

    def test_non_conversation_clears_incompatible_fields(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            source_file="/tmp/empty.mp3",
            source_filename="2026-03-08__11-22-33__79161234567__Петрова Анна_44.mp3",
            phone="+79161234567",
            manager_name="Петрова Анна",
            started_at=datetime(2026, 3, 8, 11, 22, 33),
        )
        transcript = "Голосовой ассистент. Оставьте сообщение после сигнала."
        raw = {
            "history_summary": "Автоответчик",
            "structured_fields": {
                "people": {},
                "contacts": {"email": "test@example.com", "preferred_channel": "email"},
                "student": {"grade_current": "8", "school": "Школа 1"},
                "interests": {
                    "products": ["летний лагерь"],
                    "format": ["онлайн"],
                    "subjects": ["математика"],
                    "exam_targets": [],
                },
                "commercial": {"price_sensitivity": "high", "budget": "до 100000", "discount_interest": True},
                "objections": ["цена"],
                "next_step": {"action": "Отправить материалы", "due": "2026-03-10"},
                "lead_priority": "hot",
            },
            "target_product": "летний лагерь",
            "follow_up_score": 80,
            "follow_up_reason": "Надо отправить материалы.",
            "tags": ["non_conversation"],
        }

        analysis = service._normalize_analysis(call, transcript, raw)
        self.assertEqual(analysis["quality_flags"]["call_type"], "non_conversation")
        self.assertEqual(analysis["follow_up_score"], 0)
        self.assertIsNone(analysis["target_product"])
        self.assertEqual(analysis["structured_fields"]["interests"]["products"], [])
        self.assertIsNone(analysis["structured_fields"]["next_step"]["action"])
        self.assertEqual(analysis["tags"], ["non_conversation"])

    def test_non_conversation_hard_validation_rewrites_invented_sales_payload(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            source_file="/tmp/voicemail.mp3",
            source_filename="2026-03-08__11-22-33__79161234567__Петрова Анна_44.mp3",
            phone="+79161234567",
            manager_name="Петрова Анна",
            started_at=datetime(2026, 3, 8, 11, 22, 33),
            duration_sec=24.0,
        )
        transcript = (
            "MANAGER:\n"
            "Добрый день, это Фотон, хотели рассказать про летний лагерь.\n\n"
            "CLIENT:\n"
            "Абонент сейчас не может ответить на ваш звонок. Оставьте сообщение после звукового сигнала."
        )
        raw = {
            "history_summary": "Клиент заинтересовался летним лагерем и попросил отправить ссылку на оплату.",
            "structured_fields": {
                "people": {"parent_fio": "Иванова Анна", "child_fio": "Петр"},
                "contacts": {"email": "test@example.com", "preferred_channel": "telegram"},
                "student": {"grade_current": "7", "school": "Школа 1"},
                "interests": {"products": ["летний лагерь"], "format": [], "subjects": ["математика"], "exam_targets": []},
                "commercial": {"price_sensitivity": "high", "budget": "до 100000", "discount_interest": True},
                "objections": ["цена"],
                "next_step": {"action": "Отправить ссылку на оплату", "due": "завтра"},
                "lead_priority": "hot",
            },
            "target_product": "летний лагерь",
            "follow_up_score": 90,
            "tags": ["sales_call"],
        }

        analysis = service._normalize_analysis(call, transcript, raw)

        self.assertEqual(analysis["quality_flags"]["call_type"], "non_conversation")
        self.assertTrue(analysis["quality_flags"]["non_conversation_hard_validation_applied"])
        self.assertNotIn("летний лагерь", analysis["history_summary"].lower())
        self.assertNotIn("ссылку на оплату", analysis["history_summary"].lower())
        self.assertEqual(analysis["follow_up_score"], 0)
        self.assertIsNone(analysis["next_step"])
        self.assertEqual(analysis["structured_fields"]["interests"]["products"], [])
        self.assertEqual(analysis["structured_fields"]["objections"], [])
        self.assertEqual(analysis["structured_fields"]["contacts"]["phone_from_filename"], "+79161234567")

    def test_non_conversation_advisory_preserves_llm_facts_and_marks_review(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            source_file="/tmp/voicemail.mp3",
            source_filename="2026-03-08__11-22-33__79161234567__Петрова Анна_44.mp3",
            phone="+79161234567",
            manager_name="Петрова Анна",
            started_at=datetime(2026, 3, 8, 11, 22, 33),
            duration_sec=24.0,
        )
        transcript = (
            "MANAGER:\n"
            "Добрый день, это Фотон, хотели рассказать про летний лагерь.\n\n"
            "CLIENT:\n"
            "Абонент сейчас не может ответить на ваш звонок. Оставьте сообщение после звукового сигнала."
        )
        raw = {
            "history_summary": "Клиент заинтересовался летним лагерем и попросил отправить ссылку на оплату.",
            "structured_fields": {
                "people": {"parent_fio": "Иванова Анна", "child_fio": "Петр"},
                "contacts": {"email": "test@example.com", "preferred_channel": "telegram"},
                "student": {"grade_current": "7", "school": "Школа 1"},
                "interests": {"products": ["летний лагерь"], "format": [], "subjects": ["математика"], "exam_targets": []},
                "commercial": {"price_sensitivity": "high", "budget": "до 100000", "discount_interest": True},
                "objections": ["цена"],
                "next_step": {"action": "Отправить ссылку на оплату", "due": "завтра"},
                "lead_priority": "hot",
            },
            "target_product": "летний лагерь",
            "follow_up_score": 90,
            "tags": ["sales_call"],
        }

        with patch.dict("os.environ", {NON_CONVERSATION_ADVISORY_ENV: "1"}):
            analysis = service._normalize_analysis(call, transcript, raw)

        quality_flags = analysis["quality_flags"]
        self.assertEqual(quality_flags["call_type"], "sales_call")
        self.assertTrue(quality_flags["non_conversation_advisory"])
        self.assertEqual(quality_flags["non_conversation_advisory_recommended_call_type"], "non_conversation")
        self.assertIn("pre_llm_guardrail", quality_flags["non_conversation_advisory_sources"])
        self.assertIn("post_llm_detector", quality_flags["non_conversation_advisory_sources"])
        self.assertTrue(quality_flags["needs_review"])
        self.assertIn("non_conversation_advisory", quality_flags["review_reasons"])
        self.assertFalse(quality_flags.get("non_conversation_hard_validation_applied", False))
        self.assertEqual(analysis["target_product"], "летний лагерь")
        self.assertEqual(analysis["structured_fields"]["interests"]["products"], ["летний лагерь"])
        self.assertEqual(analysis["structured_fields"]["next_step"]["action"], "Отправить ссылку на оплату")
        self.assertEqual(analysis["structured_fields"]["people"]["parent_fio"], "Иванова Анна")
        self.assertIn("non_conversation_advisory", analysis["tags"])

    def test_normalize_analysis_marks_sales_without_product_and_next_step_for_review(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            source_file="/tmp/review.mp3",
            source_filename="2026-03-08__11-22-33__79161234567__Петрова Анна_44.mp3",
            phone="+79161234567",
            manager_name="Петрова Анна",
            started_at=datetime(2026, 3, 8, 11, 22, 33),
            duration_sec=46.0,
        )
        transcript = (
            "[00:01.1] Менеджер: Добрый день, подскажите, вас интересует обучение по математике для 8 класса?\n"
            "[00:04.0] Клиент: Да, хотим подобрать курс, но пока просто узнаем подробности и стоимость.\n"
        )
        raw = {
            "history_summary": "Клиент интересуется обучением по математике для 8 класса и узнает подробности.",
            "structured_fields": {
                "people": {},
                "contacts": {},
                "student": {"grade_current": "8", "school": None},
                "interests": {"products": [], "format": [], "subjects": ["математика"], "exam_targets": []},
                "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
                "objections": [],
                "next_step": {"action": None, "due": None},
                "lead_priority": "warm",
            },
            "follow_up_score": 58,
            "follow_up_reason": "Есть интерес к обучению, но конкретный продукт и следующий шаг не зафиксированы.",
            "tags": [],
        }

        analysis = service._normalize_analysis(call, transcript, raw)
        self.assertEqual(analysis["quality_flags"]["call_type"], "sales_call")
        self.assertTrue(analysis["quality_flags"]["needs_review"])
        self.assertIn("sales_missing_product_and_next_step", analysis["quality_flags"]["review_reasons"])
        self.assertTrue(analysis["needs_review"])

    def test_normalize_analysis_marks_long_non_conversation_for_review(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            source_file="/tmp/long_nonconv.mp3",
            source_filename="2026-03-08__11-22-33__79161234567__Петрова Анна_44.mp3",
            phone="+79161234567",
            manager_name="Петрова Анна",
            started_at=datetime(2026, 3, 8, 11, 22, 33),
            duration_sec=42.0,
        )
        transcript = "Голосовой ассистент. Оставьте сообщение после сигнала."
        raw = {
            "history_summary": "Нецелевой звонок: автоответчик/короткий технический дозвон.",
            "tags": ["non_conversation"],
        }

        analysis = service._normalize_analysis(call, transcript, raw)
        self.assertEqual(analysis["quality_flags"]["call_type"], "non_conversation")
        self.assertTrue(analysis["quality_flags"]["needs_review"])
        self.assertIn("long_non_conversation", analysis["quality_flags"]["review_reasons"])

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
            with session_factory() as session:
                call = session.query(CallRecord).one()
                analysis = json.loads(call.analysis_json or "{}")
            meta = analysis.get("analysis_meta")
            self.assertIsInstance(meta, dict)
            self.assertEqual(meta["analysis_provider"], "mock")
            self.assertEqual(meta["analysis_model"], "mock")
            self.assertEqual(meta["analysis_prompt_version"], "v6")
            self.assertTrue(meta["analyzed_at"].endswith("+00:00"))

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
        settings = replace(
            make_settings(),
            analyze_provider="codex_cli",
            analyze_escalate_full_on_ambiguity=False,
        )
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

    def test_analyze_text_pre_llm_gate_skips_codex_for_no_live_call(self) -> None:
        settings = replace(
            make_settings(),
            analyze_provider="codex_cli",
            analyze_escalate_full_on_ambiguity=True,
        )
        service = AnalyzeService(settings)
        call = CallRecord(
            source_file="/tmp/voicemail.mp3",
            source_filename="voicemail.mp3",
            manager_name="Смирнов Иван",
            duration_sec=18.0,
        )
        text = (
            "MANAGER:\n"
            "Добрый день, это учебный центр Фотон.\n\n"
            "CLIENT:\n"
            "Абонент сейчас не может ответить на ваш звонок. Оставьте сообщение после звукового сигнала."
        )

        with patch.object(service, "_codex_cli_analysis", return_value={"summary": "should not be called"}) as mocked:
            payload = service._analyze_text(call, text)

        mocked.assert_not_called()
        self.assertEqual(payload["tags"], ["non_conversation"])
        self.assertEqual(payload["follow_up_score"], 0)
        self.assertTrue(payload["quality_flags"]["pre_llm_non_conversation_gate"])
        self.assertTrue(payload["quality_flags"]["transcript_quality_should_force_non_conversation"])

    def test_analyze_text_advisory_calls_provider_for_no_live_candidate(self) -> None:
        settings = replace(
            make_settings(),
            analyze_provider="codex_cli",
            analyze_escalate_full_on_ambiguity=False,
        )
        service = AnalyzeService(settings)
        call = CallRecord(
            source_file="/tmp/voicemail.mp3",
            source_filename="voicemail.mp3",
            manager_name="Смирнов Иван",
            duration_sec=18.0,
        )
        text = (
            "MANAGER:\n"
            "Добрый день, это учебный центр Фотон.\n\n"
            "CLIENT:\n"
            "Абонент сейчас не может ответить на ваш звонок. Оставьте сообщение после звукового сигнала."
        )

        with patch.dict("os.environ", {NON_CONVERSATION_ADVISORY_ENV: "1"}):
            with patch.object(
                service,
                "_codex_cli_analysis",
                return_value={"summary": "provider called", "target_product": "летний лагерь"},
            ) as mocked:
                payload = service._analyze_text(call, text)

        mocked.assert_called_once()
        self.assertEqual(payload["target_product"], "летний лагерь")

    def test_migrate_analysis_schema_refreshes_export_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_migrate_export_") as td:
            root = Path(td)
            calls_dir = root / "calls"
            calls_dir.mkdir(parents=True, exist_ok=True)
            source = calls_dir / "a.mp3"
            source.write_bytes(b"")
            export_dir = root / "transcripts"
            db_path = root / "migrate.db"

            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                transcript_export_dir=str(export_dir),
            )
            init_db(settings)
            session_factory = build_session_factory(settings)

            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(source),
                        source_filename=source.name,
                        phone="+79990000000",
                        manager_name="Петрова Анна",
                        started_at=datetime(2026, 3, 8, 11, 22, 33),
                        transcription_status="done",
                        resolve_status="done",
                        analysis_status="done",
                        transcript_text=(
                            "[00:01.0] Менеджер: Подскажите, по оплате и расписанию на следующую неделю все удобно?\n"
                            "[00:04.0] Клиент: Оплату внесем завтра, а одно занятие нужно перенести.\n"
                        ),
                        analysis_json=json.dumps(
                            {
                                "analysis_schema_version": "v2",
                                "history_summary": "Звонок без содержательного диалога.",
                                "structured_fields": {
                                    "people": {},
                                    "contacts": {},
                                    "student": {},
                                    "interests": {
                                        "products": [],
                                        "format": [],
                                        "subjects": [],
                                        "exam_targets": [],
                                    },
                                    "commercial": {
                                        "price_sensitivity": None,
                                        "budget": None,
                                        "discount_interest": None,
                                    },
                                    "objections": [],
                                    "next_step": {"action": "Уточнить информацию и сообщить клиенту", "due": None},
                                    "lead_priority": "cold",
                                },
                                "follow_up_score": 0,
                                "follow_up_reason": "Нет содержательного диалога.",
                                "tags": ["non_conversation"],
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
                session.commit()

            old_get_settings = cli_module.get_settings
            try:
                cli_module.get_settings = lambda: settings  # type: ignore[assignment]
                code = cli_module.cmd_migrate_analysis_schema(
                    Namespace(
                        limit=10,
                        target_version="v2",
                        only_done=True,
                        dry_run=False,
                        force=True,
                        workers=2,
                    )
                )
            finally:
                cli_module.get_settings = old_get_settings  # type: ignore[assignment]

            self.assertEqual(code, 0)
            base = export_dir / calls_dir.name / source.stem
            summary_path = base.with_name(f"{source.stem}_history_summary.txt")
            structured_path = base.with_name(f"{source.stem}_structured_fields.json")
            self.assertTrue(summary_path.exists())
            self.assertTrue(structured_path.exists())
            summary_text = summary_path.read_text(encoding="utf-8")
            structured = json.loads(structured_path.read_text(encoding="utf-8"))
            self.assertIn("Петрова Анна", summary_text)
            self.assertEqual(structured["next_step"]["action"], "Уточнить информацию и сообщить клиенту")
            with session_factory() as session:
                call = session.query(CallRecord).one()
                migrated = json.loads(call.analysis_json or "{}")
            self.assertNotIn("analysis_meta", migrated)


if __name__ == "__main__":
    unittest.main()
