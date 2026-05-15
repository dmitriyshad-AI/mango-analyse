from __future__ import annotations

import io
import json
import tempfile
import unittest
from argparse import Namespace
from subprocess import CompletedProcess
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from mango_mvp.cli import cmd_reset_analysis
from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.services.analyze import AnalyzeService, OBJECTION_PATTERNS
from tests.test_dialogue_format import make_settings


class AnalyzeServiceTest(unittest.TestCase):
    def test_compact_prompt_requires_dense_history_summary(self) -> None:
        self.assertIn("dense CRM note", AnalyzeService(make_settings())._analysis_system_prompt("compact"))
        self.assertIn("what the manager clarified/offered/explained", AnalyzeService(make_settings())._analysis_system_prompt("compact"))

    def test_claim_batch_assigns_distinct_calls_per_worker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_claim_") as td:
            db_path = Path(td) / "claim.db"
            settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
            init_db(settings)
            session_factory = build_session_factory(settings)

            with session_factory() as session:
                for idx in range(5):
                    session.add(
                        CallRecord(
                            source_file=str(Path(td) / f"call_{idx}.mp3"),
                            source_filename=f"call_{idx}.mp3",
                            transcription_status="done",
                            resolve_status="done",
                            analysis_status="pending",
                            transcript_text=f"dialogue {idx}",
                        )
                    )
                session.commit()

            service = AnalyzeService(settings)
            with session_factory() as session1, session_factory() as session2:
                claimed1 = service._claim_batch(session1, limit=2, worker_id="w1")
                claimed2 = service._claim_batch(session2, limit=2, worker_id="w2")

            self.assertEqual(len(claimed1), 2)
            self.assertEqual(len(claimed2), 2)
            self.assertTrue(set(claimed1).isdisjoint(set(claimed2)))

            with session_factory() as session:
                rows = session.query(CallRecord).order_by(CallRecord.id.asc()).all()
                claimed_rows = [row for row in rows if row.analysis_status == "in_progress"]
                self.assertEqual(len(claimed_rows), 4)
                self.assertEqual(sum(1 for row in claimed_rows if row.analysis_worker_id == "w1"), 2)
                self.assertEqual(sum(1 for row in claimed_rows if row.analysis_worker_id == "w2"), 2)

    def test_claim_batch_releases_stale_in_progress_rows(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_stale_") as td:
            db_path = Path(td) / "stale.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                analyze_lease_timeout_sec=120,
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            stale_time = datetime.now(timezone.utc) - timedelta(seconds=3600)

            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(Path(td) / "stale.mp3"),
                        source_filename="stale.mp3",
                        transcription_status="done",
                        resolve_status="done",
                        analysis_status="in_progress",
                        analysis_worker_id="old-worker",
                        analysis_claimed_at=stale_time,
                        transcript_text="реальный диалог",
                    )
                )
                session.commit()

            service = AnalyzeService(settings)
            with session_factory() as session:
                claimed = service._claim_batch(session, limit=1, worker_id="new-worker")

            self.assertEqual(len(claimed), 1)
            with session_factory() as session:
                row = session.query(CallRecord).one()
                self.assertEqual(row.analysis_status, "in_progress")
                self.assertEqual(row.analysis_worker_id, "new-worker")
                self.assertIsNotNone(row.analysis_claimed_at)

    def test_dialogue_dump_detection_handles_manager_client_labels(self) -> None:
        service = AnalyzeService(make_settings())
        text = "MANAGER:\nДобрый день.\n\nCLIENT:\nЗдравствуйте."
        self.assertTrue(service._looks_like_dialogue_dump(text))

    def test_extract_json_payload_handles_python_style_dict(self) -> None:
        service = AnalyzeService(make_settings())
        payload = service._extract_json_payload("{'ok': True, 'value': 'test'}")
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["value"], "test")

    def test_price_objection_pattern_does_not_match_centr(self) -> None:
        self.assertIsNone(OBJECTION_PATTERNS["цена"].search("учебный центр"))

    def test_normalize_next_step_action_rewrites_english_phrase(self) -> None:
        service = AnalyzeService(make_settings())
        self.assertEqual(
            service._normalize_next_step_action("Call back with personalized offer."),
            "Перезвонить клиенту",
        )

    def test_normalize_next_step_action_canonicalizes_send_to_channel(self) -> None:
        service = AnalyzeService(make_settings())
        self.assertEqual(
            service._normalize_next_step_action("Написать клиенту в Telegram и выслать программу на email."),
            "Отправить материалы",
        )

    def test_non_conversation_does_not_drop_meaningful_transfer_call(self) -> None:
        service = AnalyzeService(make_settings())
        text = (
            "MANAGER:\n"
            "Учебный центр здравствуйте чем могу помочь? Тогда я сейчас соединю с коллегами, "
            "оставайтесь на линии пожалуйста. На июль хочу двоих детей записать.\n\n"
            "CLIENT:\n"
            "Добрый день, хотела записать двоих детей в летний лагерь. Мне сказали, что есть "
            "еще места на июль."
        )
        self.assertFalse(service._is_non_conversation(text))

    def test_detect_call_type_marks_technical_call_not_non_conversation(self) -> None:
        service = AnalyzeService(make_settings())
        text = (
            "MANAGER:\n"
            "Добрый день, подскажите, у вас открывается личный кабинет и онлайн-тест?\n\n"
            "CLIENT:\n"
            "Нет, ссылка не работает, выдает ошибку, помогите подключиться."
        )
        self.assertEqual(service._detect_call_type(text), "technical_call")
        self.assertFalse(service._is_non_conversation(text))

    def test_detect_call_type_marks_service_call_not_non_conversation(self) -> None:
        service = AnalyzeService(make_settings())
        text = (
            "MANAGER:\n"
            "Звоню уточнить оплату и расписание на следующую неделю.\n\n"
            "CLIENT:\n"
            "Да, оплату внесем завтра, а одно занятие нужно перенести."
        )
        self.assertEqual(service._detect_call_type(text), "service_call")
        self.assertFalse(service._is_non_conversation(text))

    def test_detect_call_type_marks_virtual_secretary_as_non_conversation(self) -> None:
        service = AnalyzeService(make_settings())
        text = (
            "MANAGER:\n"
            "Добрый день.\n\n"
            "CLIENT:\n"
            "На связи я секретарь, временно попросили отвечать на звонки. "
            "Абонент сейчас не может ответить."
        )
        self.assertEqual(service._detect_call_type(text), "non_conversation")
        self.assertTrue(service._is_non_conversation(text))

    def test_normalize_analysis_attaches_transcript_quality_guardrails_high_confidence(self) -> None:
        service = AnalyzeService(make_settings())
        text = (
            "MANAGER:\n"
            "Добрый день.\n\n"
            "CLIENT:\n"
            "Звонок был перенаправлен на голосовой почтовый ящик. "
            "Оставьте сообщение после звукового сигнала. Продолжение следует."
        )
        call = CallRecord(
            source_file="/tmp/voicemail.mp3",
            source_filename="voicemail.mp3",
            duration_sec=25,
            transcript_text=text,
        )

        analysis = service._normalize_analysis(call, text, {})
        quality = analysis["quality_flags"]
        guardrails = quality["transcript_quality_guardrails"]

        self.assertEqual(guardrails["mode"], "dry_run")
        self.assertEqual(guardrails["label"], "non_conversation_high_confidence")
        self.assertTrue(guardrails["should_force_non_conversation"])
        self.assertEqual(guardrails["recommended_call_type"], "non_conversation")
        self.assertEqual(quality["transcript_quality_label"], "non_conversation_high_confidence")
        self.assertTrue(quality["transcript_quality_should_force_non_conversation"])

    def test_normalize_analysis_marks_outbound_voicemail_subtype(self) -> None:
        service = AnalyzeService(make_settings())
        text = (
            "MANAGER:\n"
            "Добрый день, это учебный центр Фотон. Оставляю информацию по курсу подготовки к ЕГЭ "
            "по математике, перезвоните нам, пожалуйста.\n\n"
            "CLIENT:\n"
            "Абонент сейчас не может ответить на ваш звонок. Оставьте сообщение после звукового сигнала."
        )
        call = CallRecord(
            source_file="/tmp/outbound_voicemail.mp3",
            source_filename="outbound_voicemail.mp3",
            duration_sec=45,
            transcript_text=text,
        )

        analysis = service._normalize_analysis(call, text, {"call_type": "sales_call", "tags": ["sales_call"]})
        guardrails = analysis["quality_flags"]["transcript_quality_guardrails"]

        self.assertEqual(guardrails["label"], "non_conversation_high_confidence")
        self.assertTrue(guardrails["should_force_non_conversation"])
        self.assertTrue(guardrails["outbound_voicemail_marker"])
        self.assertEqual(guardrails["recommended_contact_subtype"], "outbound_voicemail")
        self.assertIn("outbound_voicemail", guardrails["reason_codes"])

    def test_normalize_analysis_quality_guardrails_protect_live_service_words(self) -> None:
        service = AnalyzeService(make_settings())
        text = (
            "MANAGER:\n"
            "Добрый день, я отправлю чек на почту и завтра перезвоню.\n\n"
            "CLIENT:\n"
            "Да, чек нужен на почту. Оплату внесли, но ссылка на занятие не работает, "
            "помогите с доступом и расписанием."
        )
        call = CallRecord(
            source_file="/tmp/live_service.mp3",
            source_filename="live_service.mp3",
            duration_sec=180,
            transcript_text=text,
        )

        analysis = service._normalize_analysis(call, text, {})
        guardrails = analysis["quality_flags"]["transcript_quality_guardrails"]

        self.assertEqual(guardrails["mode"], "dry_run")
        self.assertEqual(guardrails["label"], "contentful_protected_live_dialogue")
        self.assertTrue(guardrails["protected_live_dialogue"])
        self.assertFalse(guardrails["should_force_non_conversation"])
        self.assertNotEqual(analysis["quality_flags"]["call_type"], "non_conversation")

    def test_normalize_analysis_quality_guardrails_force_clear_no_live(self) -> None:
        service = AnalyzeService(make_settings())
        text = (
            "MANAGER:\n"
            "Добрый день, учебный центр.\n\n"
            "CLIENT:\n"
            "Абонент сейчас не может ответить на ваш звонок. Попробуйте перезвонить позднее."
        )
        call = CallRecord(
            source_file="/tmp/borderline.mp3",
            source_filename="borderline.mp3",
            duration_sec=55,
            transcript_text=text,
        )

        analysis = service._normalize_analysis(call, text, {})
        quality = analysis["quality_flags"]
        guardrails = quality["transcript_quality_guardrails"]

        self.assertEqual(guardrails["mode"], "dry_run")
        self.assertEqual(guardrails["label"], "non_conversation_high_confidence")
        self.assertFalse(guardrails["requires_manual_review"])
        self.assertTrue(guardrails["should_force_non_conversation"])
        self.assertEqual(guardrails["recommended_call_type"], "non_conversation")
        self.assertEqual(guardrails["recommended_contact_subtype"], "no_live_or_voicemail")
        self.assertEqual(quality["call_type"], "non_conversation")
        self.assertTrue(quality["non_conversation_hard_validation_applied"])
        self.assertIsNone(analysis["next_step"])
        self.assertEqual(analysis["follow_up_score"], 0)

    def test_detect_call_type_marks_existing_client_progress_not_sales_with_subjects(self) -> None:
        service = AnalyzeService(make_settings())
        text = (
            "MANAGER:\n"
            "Звоню собрать обратную связь по текущему курсу по математике и физике.\n\n"
            "CLIENT:\n"
            "По физике всё нравится, по математике есть пробелы, но в целом продолжаем обучение."
        )
        self.assertEqual(service._detect_call_type(text, subjects=["математика", "физика"]), "existing_client_progress")

    def test_existing_client_feedback_with_subject_and_grade_is_not_sales_signal(self) -> None:
        service = AnalyzeService(make_settings())
        text = (
            "Менеджер позвонил собрать обратную связь по текущему обучению по математике для 8 класса. "
            "Клиент сообщил, что продолжают занятия и замечаний по курсу нет."
        )
        self.assertFalse(service._has_meaningful_sales_signal(text))

    def test_codex_cli_analysis_retries_empty_last_message(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_retry_") as td:
            service = AnalyzeService(
                replace(
                    make_settings(),
                    analyze_provider="codex_cli",
                    codex_analyze_model="gpt-5.4-mini",
                    llm_cache_enabled=True,
                    llm_cache_dir=str(Path(td) / "llm-cache"),
                )
            )
            prompt_payload = {
                "analysis_schema_version": "v2",
                "history_summary": "28.01.2026 11:11 менеджер Клычева Дарья общался с клиентом. Клиент интересуется летней выездной школой на август. Менеджер сообщил про скидку 20% до 1 февраля и предложил отправить информацию на почту. Клиент попросил прислать материалы и сообщил, что при возможности оплатит участие. Следующий шаг: отправить информацию по программе и дождаться решения клиента.",
                "structured_fields": {
                    "people": {"parent_fio": None, "child_fio": None},
                    "contacts": {"email": None, "phone_from_filename": None, "preferred_channel": "email"},
                    "student": {"grade_current": None, "school": None},
                    "interests": {"products": ["летний лагерь"], "format": [], "subjects": [], "exam_targets": []},
                    "commercial": {"price_sensitivity": "medium", "budget": None, "discount_interest": True},
                    "objections": [],
                    "next_step": {"action": "Отправить материалы", "due": None},
                    "lead_priority": "warm",
                },
                "target_product": "летний лагерь",
                "tags": [],
            }

            class DummyCall:
                source_filename = "test.mp3"
                started_at = None
                manager_name = "Менеджер"
                phone = "+70000000000"
                direction = "unknown"

            state = {"calls": 0}

            def fake_run(cmd, capture_output, text, check, timeout, input=None):
                state["calls"] += 1
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                if state["calls"] == 1:
                    out_path.write_text("", encoding="utf-8")
                    return CompletedProcess(
                        cmd,
                        1,
                        stdout="",
                        stderr="Warning: no last agent message; wrote empty content to output",
                    )
                out_path.write_text(json.dumps(prompt_payload, ensure_ascii=False), encoding="utf-8")
                return CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("mango_mvp.services.analyze.shutil.which", return_value="/usr/bin/codex"):
                with patch("mango_mvp.services.analyze.subprocess.run", side_effect=fake_run):
                    with patch("mango_mvp.services.analyze.time.sleep", return_value=None):
                        payload = service._codex_cli_analysis(
                            DummyCall(),
                            "MANAGER:\nДобрый день.\nCLIENT:\nЗдравствуйте.",
                        )

            self.assertEqual(state["calls"], 2)
            self.assertEqual(payload["target_product"], "летний лагерь")

    def test_codex_cli_analysis_uses_response_cache_on_repeat(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_cache_") as td:
            service = AnalyzeService(
                replace(
                    make_settings(),
                    analyze_provider="codex_cli",
                    codex_analyze_model="gpt-5.4-mini",
                    llm_cache_enabled=True,
                    llm_cache_dir=str(Path(td) / "llm-cache"),
                )
            )
            payload = {
                "analysis_schema_version": "v2",
                "history_summary": "Клиент интересуется математикой и просит выслать материалы.",
                "structured_fields": {
                    "people": {"parent_fio": None, "child_fio": None},
                    "contacts": {"email": None, "phone_from_filename": None, "preferred_channel": "telegram"},
                    "student": {"grade_current": "8", "school": None},
                    "interests": {"products": ["годовые курсы"], "format": ["онлайн"], "subjects": ["математика"], "exam_targets": []},
                    "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
                    "objections": [],
                    "next_step": {"action": "Отправить материалы", "due": None},
                    "lead_priority": "warm",
                },
                "target_product": "годовые курсы",
                "tags": [],
            }

            class DummyCall:
                source_filename = "test.mp3"
                started_at = None
                manager_name = "Менеджер"
                phone = "+70000000000"
                direction = "unknown"

            state = {"calls": 0}

            def fake_run(cmd, capture_output, text, check, timeout, input=None):
                state["calls"] += 1
                self.assertIn("--model", cmd)
                self.assertIn("gpt-5.4-mini", cmd)
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("mango_mvp.services.analyze.shutil.which", return_value="/usr/bin/codex"):
                with patch("mango_mvp.services.analyze.subprocess.run", side_effect=fake_run):
                    first = service._codex_cli_analysis(
                        DummyCall(),
                        "MANAGER:\nДобрый день.\nCLIENT:\nИнтересует математика, пришлите материалы.",
                    )
                    second = service._codex_cli_analysis(
                        DummyCall(),
                        "MANAGER:\nДобрый день.\nCLIENT:\nИнтересует математика, пришлите материалы.",
                    )

            self.assertEqual(state["calls"], 1)
            self.assertFalse(first["quality_flags"]["analyze_llm_cache_hit"])
            self.assertTrue(second["quality_flags"]["analyze_llm_cache_hit"])
            self.assertEqual(first["target_product"], second["target_product"])
            self.assertEqual(first["history_summary"], second["history_summary"])
            self.assertEqual(
                first["quality_flags"]["analyze_transcript_chars_prompt"],
                second["quality_flags"]["analyze_transcript_chars_prompt"],
            )

    def test_compact_prompt_includes_hints_and_compact_contract(self) -> None:
        service = AnalyzeService(replace(make_settings(), analyze_provider="codex_cli"))

        class DummyCall:
            source_filename = "2026-03-05__16-03-46__Тропов Олег__79269136368_5.mp3"
            started_at = None
            manager_name = "Тропов Олег"
            phone = "+79269136368"
            direction = "outbound"

        text = (
            "MANAGER:\n"
            "Собираем обратную связь по курсу физики во втором семестре.\n\n"
            "CLIENT:\n"
            "Все нравится, будут ли пробники и дополнительные срезы?"
        )
        system_prompt = service._analysis_system_prompt("compact")
        user_prompt = service._analysis_user_prompt(DummyCall(), text, "compact")

        self.assertIn("Return exactly these keys", system_prompt)
        self.assertNotIn("evidence", system_prompt.lower())
        self.assertNotIn("follow_up_score", system_prompt)
        self.assertNotIn("follow_up_reason", system_prompt)
        self.assertIn("single-line minified JSON object", system_prompt)
        self.assertIn("Deterministic hints", user_prompt)
        self.assertIn("subject_candidates", user_prompt)
        self.assertNotIn('"target_product_candidates":[]', user_prompt)

    def test_prompt_compaction_reduces_filler_without_losing_sales_content(self) -> None:
        service = AnalyzeService(make_settings())
        transcript = (
            "[00:00.1] Клиент: Да, да, да, да.\n"
            "[00:00.6] Клиент: Да, да, да, да.\n"
            "[00:01.1] Менеджер: Хорошо, хорошо, спасибо, спасибо.\n"
            "[00:03.0] Клиент: Нас интересует летний лагерь по математике, пришлите ссылку на оплату.\n"
        )

        compacted = service._compact_transcript_for_prompt(transcript, "compact")
        baseline = service._compact_transcript_for_prompt(transcript, "compact", apply_compaction=False)

        self.assertLess(compacted["transcript_chars_prompt"], baseline["transcript_chars_prompt"])
        self.assertTrue(compacted["transcript_compacted"])
        self.assertEqual(compacted["transcript_compaction_removed_lines"], 0)
        self.assertGreater(compacted["transcript_compaction_shortened_lines"], 0)
        self.assertGreater(compacted["transcript_prompt_timestamps_removed_lines"], 0)
        self.assertIn("летний лагерь", compacted["transcript"])
        self.assertIn("ссылку на оплату", compacted["transcript"])
        self.assertNotIn("[00:00.1]", compacted["transcript"])
        self.assertEqual(compacted["transcript"].count("Клиент: Да"), 2)

    def test_non_conversation_llm_sales_signal_adds_soft_warning_without_preserving_fields(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            source_file="voicemail.mp3",
            source_filename="voicemail.mp3",
            manager_name="Иван",
            phone="+79990000000",
            direction="outbound",
            duration_sec=20,
        )
        text = (
            "MANAGER:\n"
            "Добрый день, это учебный центр Фотон.\n\n"
            "CLIENT:\n"
            "Абонент сейчас не может ответить на ваш звонок. "
            "Оставьте сообщение после звукового сигнала."
        )
        raw = {
            "history_summary": "Клиент интересуется курсом.",
            "structured_fields": {
                "interests": {"products": ["летний лагерь"]},
                "objections": ["цена"],
                "next_step": {"action": "Отправить ссылку на оплату"},
            },
            "target_product": "летний лагерь",
        }

        result = service._normalize_analysis(call, text, raw)
        quality_flags = result["quality_flags"]
        structured_fields = result["structured_fields"]

        self.assertEqual(quality_flags["call_type"], "non_conversation")
        self.assertTrue(quality_flags["non_conversation_soft_warning_llm_sales_signal"])
        self.assertEqual(
            quality_flags["non_conversation_soft_warning_sources"],
            ["interests.products", "target_product", "next_step.action", "objections"],
        )
        self.assertTrue(quality_flags["needs_review"])
        self.assertIn("non_conversation_llm_sales_signal_soft_warning", quality_flags["review_reasons"])
        self.assertEqual(structured_fields["interests"]["products"], [])
        self.assertIsNone(structured_fields["next_step"]["action"])
        self.assertEqual(structured_fields["objections"], [])
        self.assertIsNone(result["target_product"])

    def test_prompt_compaction_can_be_disabled(self) -> None:
        settings = replace(make_settings(), analyze_transcript_compaction_enabled=False)
        service = AnalyzeService(settings)
        transcript = (
            "[00:00.1] Клиент: Да, да, да.\n"
            "[00:00.6] Клиент: Да, да, да.\n"
            "[00:03.0] Клиент: Нас интересует летний лагерь.\n"
        )

        metrics = service._compact_transcript_for_prompt(transcript, "compact")

        self.assertFalse(metrics["transcript_compacted"])
        self.assertEqual(metrics["transcript_compaction_removed_lines"], 0)
        self.assertEqual(metrics["transcript_prompt_timestamps_removed_lines"], 0)
        self.assertIn("[00:00.6] Клиент: Да, да, да.", metrics["transcript"])

    def test_analyze_text_escalates_compact_to_full_when_product_missing(self) -> None:
        settings = replace(
            make_settings(),
            analyze_provider="codex_cli",
            analyze_prompt_profile="compact",
            analyze_escalate_full_on_ambiguity=True,
        )
        service = AnalyzeService(settings)

        class DummyCall:
            source_filename = "call.mp3"
            started_at = None
            manager_name = "Менеджер"
            phone = "+70000000000"
            direction = "unknown"

        compact_payload = {
            "analysis_schema_version": "v2",
            "history_summary": "Клиент интересуется лагерем.",
            "structured_fields": {
                "people": {"parent_fio": None, "child_fio": None},
                "contacts": {"email": None, "phone_from_filename": None, "preferred_channel": None},
                "student": {"grade_current": None, "school": None},
                "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
                "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
                "objections": [],
                "next_step": {"action": None, "due": None},
                "lead_priority": "warm",
            },
            "target_product": None,
            "follow_up_score": 60,
            "follow_up_reason": "Есть интерес.",
            "tags": [],
        }
        full_payload = {
            **compact_payload,
            "structured_fields": {
                **compact_payload["structured_fields"],
                "interests": {"products": ["летний лагерь"], "format": [], "subjects": [], "exam_targets": []},
            },
            "target_product": "летний лагерь",
        }
        observed_profiles: list[str] = []

        def fake_codex(call, text, profile=None):
            observed_profiles.append(profile or "compact")
            return compact_payload if (profile or "compact") == "compact" else full_payload

        with patch.object(service, "_codex_cli_analysis", side_effect=fake_codex):
            payload = service._analyze_text(
                DummyCall(),
                "MANAGER:\nУ нас есть летний лагерь.\nCLIENT:\nРасскажите подробнее про лагерь.",
            )

        self.assertEqual(observed_profiles, ["compact", "full"])
        self.assertEqual(payload["target_product"], "летний лагерь")

    def test_analyze_text_escalates_compact_when_false_non_conversation_tagged(self) -> None:
        settings = replace(
            make_settings(),
            analyze_provider="codex_cli",
            analyze_prompt_profile="compact",
            analyze_escalate_full_on_ambiguity=True,
        )
        service = AnalyzeService(settings)

        class DummyCall:
            source_filename = "call.mp3"
            started_at = None
            manager_name = "Менеджер"
            phone = "+70000000000"
            direction = "unknown"

        compact_payload = {
            "analysis_schema_version": "v2",
            "history_summary": "Звонок без содержательного диалога.",
            "structured_fields": {
                "people": {"parent_fio": None, "child_fio": None},
                "contacts": {"email": None, "phone_from_filename": None, "preferred_channel": None},
                "student": {"grade_current": None, "school": None},
                "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
                "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
                "objections": [],
                "next_step": {"action": None, "due": None},
                "lead_priority": "cold",
            },
            "target_product": None,
            "follow_up_score": 0,
            "follow_up_reason": "Нет содержательного диалога.",
            "tags": ["non_conversation"],
        }
        full_payload = {
            **compact_payload,
            "history_summary": "Менеджер помог клиенту восстановить доступ к онлайн-тесту и договорился отправить инструкцию.",
            "structured_fields": {
                **compact_payload["structured_fields"],
                "next_step": {"action": "Отправить материалы", "due": None},
            },
            "follow_up_score": 40,
            "follow_up_reason": "Есть сервисный следующий шаг.",
            "tags": ["technical_call"],
        }
        observed_profiles: list[str] = []

        def fake_codex(call, text, profile=None):
            observed_profiles.append(profile or "compact")
            return compact_payload if (profile or "compact") == "compact" else full_payload

        with patch.object(service, "_codex_cli_analysis", side_effect=fake_codex):
            payload = service._analyze_text(
                DummyCall(),
                "MANAGER:\nПодскажите, открывается ли онлайн-тест?\nCLIENT:\nНет, ссылка не работает, нужна инструкция.",
            )

        self.assertEqual(observed_profiles, ["compact", "full"])
        self.assertEqual(payload["tags"], ["technical_call"])

    def test_compose_history_summary_does_not_duplicate_opening(self) -> None:
        service = AnalyzeService(make_settings())

        class DummyCall:
            started_at = datetime(2026, 1, 28, 11, 11)
            manager_name = "Клычева Дарья"

        summary = service._compose_history_summary(
            DummyCall(),
            draft_history_summary=(
                "28.01.2026 11:11 менеджер Клычева Дарья общался с клиентом. "
                "Клиент попросил отправить материалы на почту."
            ),
            summary=None,
            structured_fields={},
            objections=[],
            next_step_action="Отправить материалы",
            due=None,
            follow_up_reason=None,
        )
        self.assertEqual(summary.count("менеджер Клычева Дарья"), 1)

    def test_compose_history_summary_strips_duplicate_datetime_context(self) -> None:
        service = AnalyzeService(make_settings())

        class DummyCall:
            started_at = datetime(2026, 1, 23, 9, 0)
            manager_name = "Клычева Дарья"

        summary = service._compose_history_summary(
            DummyCall(),
            draft_history_summary=(
                "23.01.2026 09:00 менеджер Клычева Дарья общался с клиентом. "
                "23.01.2026 в 09:00 клиент уточнил детали по курсу и попросил выслать материалы."
            ),
            summary=None,
            structured_fields={},
            objections=[],
            next_step_action="Отправить материалы",
            due=None,
            follow_up_reason=None,
        )
        self.assertEqual(summary.count("23.01.2026 09:00"), 1)
        self.assertNotIn("23.01.2026 в 09:00", summary)

    def test_compose_history_summary_enriches_sparse_mini_draft(self) -> None:
        service = AnalyzeService(make_settings())

        class DummyCall:
            started_at = datetime(2026, 1, 23, 9, 0)
            manager_name = "Клычева Дарья"

        summary = service._compose_history_summary(
            DummyCall(),
            draft_history_summary="Клиент интересуется курсом по математике.",
            summary="Менеджер объяснил формат годового обучения и пообещал отправить программу в Telegram.",
            structured_fields={
                "student": {"grade_current": "8"},
                "interests": {
                    "products": ["годовые курсы"],
                    "subjects": ["математика"],
                    "format": ["онлайн"],
                    "exam_targets": [],
                },
                "contacts": {"preferred_channel": "telegram"},
            },
            objections=["цена"],
            next_step_action="Отправить материалы",
            due=None,
            follow_up_reason=None,
        )
        self.assertIn("Суть обращения: Менеджер объяснил формат годового обучения", summary)
        self.assertIn("класс: 8", summary)
        self.assertIn("продукты: годовые курсы", summary)
        self.assertIn("Ограничения/возражения: цена.", summary)
        self.assertIn("Договорились: Отправить материалы.", summary)

    def test_compose_history_summary_keeps_long_tail_without_ellipsis_cut(self) -> None:
        service = AnalyzeService(make_settings())

        class DummyCall:
            started_at = datetime(2026, 1, 23, 9, 0)
            manager_name = "Клычева Дарья"

        long_tail = " ".join(f"фрагмент{i}" for i in range(250))
        summary = service._compose_history_summary(
            DummyCall(),
            draft_history_summary=f"Клиент подробно обсуждал программу. {long_tail}",
            summary=None,
            structured_fields={},
            objections=[],
            next_step_action="Отправить материалы",
            due=None,
            follow_up_reason=None,
        )
        self.assertIn("фрагмент249", summary)
        self.assertFalse(summary.endswith("..."))

    def test_normalize_analysis_filters_price_objection_without_signal(self) -> None:
        service = AnalyzeService(make_settings())

        class DummyCall:
            started_at = datetime(2026, 1, 28, 11, 41)
            manager_name = "Козлова Екатерина"
            phone = "+79103549764"
            direction = "unknown"
            source_file = "a.mp3"
            source_filename = "a.mp3"
            transcript_variants_json = None

        text = (
            "MANAGER:\n"
            "Добрый день, вас беспокоит учебный центр по поводу обучения по информатике.\n\n"
            "CLIENT:\n"
            "Мы приняли положительное решение, просто сейчас неудобно разговаривать."
        )
        raw = {
            "history_summary": "Клиент подтвердил решение, но попросил перезвонить позже.",
            "structured_fields": {
                "people": {},
                "contacts": {},
                "student": {},
                "interests": {"subjects": ["информатика"]},
                "commercial": {"price_sensitivity": "high"},
                "objections": ["цена", "Неудобно разговаривать в момент звонка"],
                "next_step": {"action": "Перезвонить клиенту", "due": "29.01.2026"},
                "lead_priority": "hot",
            },
            "follow_up_score": 80,
            "follow_up_reason": "Есть согласованный следующий шаг.",
            "tags": [],
        }
        normalized = service._normalize_analysis(DummyCall(), text, raw)
        self.assertEqual(normalized["structured_fields"]["objections"], ["Неудобно разговаривать в момент звонка"])
        self.assertIsNone(normalized["structured_fields"]["commercial"]["price_sensitivity"])


class ResetAnalysisCliTest(unittest.TestCase):
    def test_reset_analysis_moves_done_back_to_pending(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_reset_analysis_") as td:
            db_path = Path(td) / "reset.db"
            settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
            init_db(settings)
            session_factory = build_session_factory(settings)

            with session_factory() as session:
                session.add_all(
                    [
                        CallRecord(
                            source_file=str(Path(td) / "a.mp3"),
                            source_filename="a.mp3",
                            transcription_status="done",
                            resolve_status="done",
                            analysis_status="done",
                            analysis_json=json.dumps({"history_summary": "old"}, ensure_ascii=False),
                            dead_letter_stage=None,
                            last_error="analyze: old",
                        ),
                        CallRecord(
                            source_file=str(Path(td) / "b.mp3"),
                            source_filename="b.mp3",
                            transcription_status="done",
                            resolve_status="manual",
                            analysis_status="done",
                            analysis_json=json.dumps({"history_summary": "manual resolve"}, ensure_ascii=False),
                            dead_letter_stage="resolve",
                        ),
                    ]
                )
                session.commit()

            args = Namespace(
                limit=100,
                statuses="done",
                only_terminal_resolve=True,
                only_analysis_dead_letter=True,
                clear_json=True,
                clear_error=True,
            )

            with patch("mango_mvp.cli.get_settings", return_value=settings):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cmd_reset_analysis(args)

            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["updated"], 1)

            with session_factory() as session:
                rows = session.query(CallRecord).order_by(CallRecord.id.asc()).all()
                self.assertEqual(rows[0].analysis_status, "pending")
                self.assertIsNone(rows[0].analysis_json)
                self.assertIsNone(rows[0].last_error)
                self.assertEqual(rows[1].analysis_status, "done")

    def test_reset_analysis_clears_in_progress_claim_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_reset_analysis_claim_") as td:
            db_path = Path(td) / "reset_claim.db"
            settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
            init_db(settings)
            session_factory = build_session_factory(settings)

            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(Path(td) / "a.mp3"),
                        source_filename="a.mp3",
                        transcription_status="done",
                        resolve_status="done",
                        analysis_status="in_progress",
                        analysis_worker_id="worker-1",
                        analysis_claimed_at=datetime.now(timezone.utc),
                        analysis_json=json.dumps({"history_summary": "old"}, ensure_ascii=False),
                    )
                )
                session.commit()

            args = Namespace(
                limit=100,
                statuses="in_progress",
                only_terminal_resolve=True,
                only_analysis_dead_letter=False,
                clear_json=True,
                clear_error=True,
            )

            with patch("mango_mvp.cli.get_settings", return_value=settings):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cmd_reset_analysis(args)

            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["updated"], 1)

            with session_factory() as session:
                row = session.query(CallRecord).one()
                self.assertEqual(row.analysis_status, "pending")
                self.assertIsNone(row.analysis_worker_id)
                self.assertIsNone(row.analysis_claimed_at)
                self.assertIsNone(row.analysis_json)
