from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import unittest
from argparse import Namespace
from subprocess import CompletedProcess
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import hashlib

from sqlalchemy import text as sa_text

from mango_mvp.cli import cmd_reset_analysis
from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.clients.ollama import OllamaClient
from mango_mvp.services.analyze import (
    AnalysisContractError,
    AnalyzeService,
    OBJECTION_PATTERNS,
    aggregate_token_usage,
    analysis_input_identity_sha256,
    analysis_input_snapshot,
    provider_token_usage,
    validate_v3_model_response,
)
from mango_mvp.services.dialogue_contract import (
    ANALYSIS_TRUNCATION_MARKER,
    HISTORY_SUMMARY_CONTRACT_VERSION,
    UNTRUSTED_SUMMARY,
    build_dialogue_input,
    call_record_view,
    guard_stored_analysis,
    apply_role_guard,
    moscow_datetime,
)
from mango_mvp.services.llm_response_cache import LLMResponseCache
from tests import mango_provider_fixture as fx
from tests.test_dialogue_format import make_settings


def _v3_fields(**overrides: Any) -> dict[str, Any]:
    """The empty v3 ``structured_fields`` object, patched field by field."""
    fields: dict[str, Any] = {
        "result": {"status": None, "detail": None},
        "people": {"parent_fio": None, "child_fio": None},
        "contacts": {"email": None, "preferred_channel": None},
        "student": {"grade_current": None, "school": None},
        "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
        "commercial": {"price_sensitivity": None, "budget": None, "discount_interest": None},
        "objections": [],
        "next_step": {"action": None, "due": None},
    }
    fields.update(overrides)
    return fields


def _v3_answer(claim_requests=None, **overrides: Any) -> dict[str, Any]:
    return {
        "structured_fields": _v3_fields(**overrides),
        "claim_requests": list(claim_requests or []),
    }


class AnalyzeServiceTest(unittest.TestCase):
    def test_compact_prompt_asks_only_for_fields_and_claims(self) -> None:
        prompt = AnalyzeService(make_settings())._analysis_system_prompt("compact")

        # ТЗ-03: the model fills fields and points at replies; the summary,
        # the quote, the timecode and the claim id are built by the service.
        self.assertIn('Return exactly two root keys: "structured_fields" and "claim_requests"', prompt)
        self.assertIn('"field_path", "item_id", "support_type", "turn_ids"', prompt)
        self.assertIn("structured_fields.next_step.action", prompt)
        self.assertIn("the service builds all of them from the dialogue itself", prompt)
        # The model is no longer asked for a конспект of its own at all.
        self.assertNotIn("dense CRM note", prompt)
        self.assertNotIn("history_summary", prompt)

    def test_full_prompt_blocks_long_dialogue_as_autoresponder_shortcut(self) -> None:
        prompt = AnalyzeService(make_settings())._analysis_system_prompt("full")

        self.assertIn("long transcripts", prompt)
        self.assertIn("multi-turn MANAGER/CLIENT dialogue", prompt)
        self.assertIn("client side is exclusively a system/IVR/voicemail/no-live message", prompt)

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

    def test_claim_batch_accepts_manual_resolve_for_review(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_manual_") as td:
            db_path = Path(td) / "manual.db"
            settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                call = CallRecord(
                    source_file=str(Path(td) / "manual.mp3"),
                    source_filename="manual.mp3",
                    transcription_status="done",
                    resolve_status="manual",
                    analysis_status="pending",
                    transcript_text="Менеджер: Здравствуйте.\nКлиент: Добрый день.",
                )
                session.add(call)
                session.commit()
                call_id = int(call.id)

            with session_factory() as session:
                claimed = AnalyzeService(settings)._claim_batch(
                    session,
                    limit=1,
                    worker_id="manual-review",
                )

            self.assertEqual(claimed, [call_id])

    def test_review_flags_keep_manual_and_exhausted_uncertainty(self) -> None:
        service = AnalyzeService(make_settings())
        call = CallRecord(
            resolve_status="manual",
            duration_sec=60.0,
            transcript_variants_json=json.dumps(
                {
                    "mode": "stereo",
                    "secondary_backfill_meta": {
                        "provider": "gigaam",
                        "attempts": 2,
                        "status": "exhausted",
                        "exhausted": True,
                    },
                }
            ),
        )

        flags = service._build_review_flags(
            call,
            text="Менеджер: Здравствуйте. Клиент: Добрый день.",
            call_type="service_call",
            products=[],
            formats=[],
            exam_targets=[],
            target_product=None,
            next_step_action=None,
            history_summary=None,
        )

        self.assertTrue(flags["needs_review"])
        self.assertIn("resolve_manual_review_required", flags["review_reasons"])
        self.assertIn(
            "secondary_asr_exhausted_primary_fallback",
            flags["review_reasons"],
        )

    def test_claim_batch_never_skips_missing_resolve_state(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_resolve_gate_") as td:
            db_path = Path(td) / "claim.db"
            settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE call_records (
                        id INTEGER PRIMARY KEY,
                        transcription_status TEXT,
                        resolve_status TEXT,
                        dead_letter_stage TEXT,
                        analysis_status TEXT,
                        analyze_attempts INTEGER DEFAULT 0,
                        next_retry_at TEXT,
                        analysis_worker_id TEXT,
                        analysis_claimed_at TEXT,
                        updated_at TEXT
                    )
                    """
                )
                connection.execute(
                    """
                    INSERT INTO call_records (
                        id, transcription_status, resolve_status,
                        analysis_status, analyze_attempts
                    ) VALUES (1, 'done', NULL, 'pending', 0)
                    """
                )
            session_factory = build_session_factory(settings)

            service = AnalyzeService(settings)
            with session_factory() as session:
                claimed = service._claim_batch(
                    session,
                    limit=1,
                    worker_id="must-not-claim",
                )

            self.assertEqual(claimed, [])
            with sqlite3.connect(db_path) as connection:
                state = connection.execute(
                    "SELECT resolve_status, analysis_status FROM call_records"
                ).fetchone()
            self.assertEqual(state, (None, "pending"))

    def test_claim_batch_never_crosses_live_pipeline_lease(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_pipeline_gate_") as td:
            db_path = Path(td) / "claim.db"
            settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(Path(td) / "call.mp3"),
                        source_filename="call.mp3",
                        transcription_status="done",
                        resolve_status="done",
                        analysis_status="pending",
                        pipeline_stage="resolve",
                        pipeline_worker_id="live-worker",
                        pipeline_claimed_at=datetime.now(timezone.utc),
                        transcript_text="synthetic dialogue",
                    )
                )
                session.commit()

            service = AnalyzeService(settings)
            with session_factory() as session:
                claimed = service._claim_batch(
                    session,
                    limit=1,
                    worker_id="must-not-claim",
                )

            self.assertEqual(claimed, [])
            with session_factory() as session:
                row = session.query(CallRecord).one()
                self.assertEqual(row.analysis_status, "pending")
                self.assertEqual(row.pipeline_worker_id, "live-worker")

    def test_claim_batch_recovers_stale_or_blank_orphan_pipeline_lease(
        self,
    ) -> None:
        cases = (
            ("resolve", "old-worker", "2020-01-01T00:00:00+00:00"),
            ("transcribe", "old-worker", "2020-01-01T00:00:00+00:00"),
            ("", "", ""),
            (" \t", " \n", None),
            (None, "old-worker", "2020-01-01T00:00:00+00:00"),
        )
        for index, (stage, worker, claimed_at) in enumerate(cases):
            with self.subTest(index=index):
                with tempfile.TemporaryDirectory(
                    prefix="mango_analyze_orphan_lease_"
                ) as td:
                    db_path = Path(td) / "claim.db"
                    settings = replace(
                        make_settings(),
                        database_url=f"sqlite:///{db_path}",
                    )
                    init_db(settings)
                    session_factory = build_session_factory(settings)
                    with session_factory() as session:
                        session.add(
                            CallRecord(
                                source_file=str(Path(td) / "call.mp3"),
                                source_filename="call.mp3",
                                transcription_status="done",
                                resolve_status="done",
                                analysis_status="pending",
                                transcript_text="synthetic dialogue",
                            )
                        )
                        session.commit()
                    with sqlite3.connect(db_path) as connection:
                        connection.execute(
                            """
                            UPDATE call_records
                               SET pipeline_stage=?, pipeline_worker_id=?,
                                   pipeline_claimed_at=?
                            """,
                            (stage, worker, claimed_at),
                        )

                    service = AnalyzeService(settings)
                    with session_factory() as session:
                        claimed = service._claim_batch(
                            session,
                            limit=1,
                            worker_id="new-worker",
                        )

                    self.assertEqual(claimed, [1])
                    with sqlite3.connect(db_path) as connection:
                        state = connection.execute(
                            """
                            SELECT pipeline_stage, pipeline_worker_id,
                                   pipeline_claimed_at, analysis_status
                              FROM call_records
                            """
                        ).fetchone()
                    self.assertEqual(state, (None, None, None, "in_progress"))

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

    def test_detect_call_type_keeps_third_party_live_dialogue_contentful(self) -> None:
        service = AnalyzeService(make_settings())
        text = (
            "MANAGER:\n"
            "Добрый день. На наш учебный центр несколько раз поступают обратные звонки, "
            "хотим понять, почему ваш номер отображается в пропущенных и можно ли убрать его из базы. "
            "Я перечислю номера, которые видим у себя, а вы проверьте, пожалуйста.\n\n"
            "CLIENT:\n"
            "Здравствуйте, ООО ПКО Актив Бизнес Консалт, я вас слышу. По указанным номерам данных в базе нет. "
            "Назовите еще раз последние цифры, мы проверим обращение и передадим ответственному сотруднику. "
            "Если звонки повторятся, попросите клиента обратиться с того номера, на который они приходят."
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

    def test_codex_cli_analysis_never_retries_an_unmetered_failure(self) -> None:
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
            prompt_payload = _v3_answer(
                interests={
                    "products": ["летний лагерь"],
                    "format": [],
                    "subjects": [],
                    "exam_targets": [],
                },
                contacts={"email": None, "preferred_channel": "email"},
            )

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
                # A second call would already be the defect under test: the
                # reserved attempt must map to exactly one external call.
                out_path.write_text(
                    json.dumps(prompt_payload, ensure_ascii=False), encoding="utf-8"
                )
                return CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("mango_mvp.services.analyze.shutil.which", return_value="/usr/bin/codex"):
                with patch("mango_mvp.services.analyze.subprocess.run", side_effect=fake_run):
                    with self.assertRaises(RuntimeError) as raised:
                        service._codex_cli_analysis(
                            DummyCall(),
                            "MANAGER:\nДобрый день.\nCLIENT:\nЗдравствуйте.",
                        )

            self.assertEqual(state["calls"], 1)
            self.assertEqual(len(raised.exception.model_attempts), 1)

    def test_codex_cli_failure_reports_every_real_retry(self) -> None:
        service = AnalyzeService(
            replace(make_settings(), analyze_provider="codex_cli")
        )

        class DummyCall:
            source_filename = "test.mp3"
            started_at = None
            manager_name = "Менеджер"
            phone = "+70000000000"
            direction = "unknown"

        calls = 0

        def fake_run(cmd, capture_output, text, check, timeout, input=None):
            nonlocal calls
            calls += 1
            Path(cmd[cmd.index("--output-last-message") + 1]).write_text(
                "", encoding="utf-8"
            )
            return CompletedProcess(
                cmd, 1, stdout="", stderr="no last agent message"
            )

        with patch(
            "mango_mvp.services.analyze.shutil.which", return_value="/usr/bin/codex"
        ), patch(
            "mango_mvp.services.analyze.subprocess.run", side_effect=fake_run
        ), patch("mango_mvp.services.analyze.time.sleep", return_value=None):
            with self.assertRaises(RuntimeError) as raised:
                service._codex_cli_analysis(
                    DummyCall(), "MANAGER:\nДобрый день.\nCLIENT:\nЗдравствуйте."
                )

        self.assertEqual(calls, 1)
        self.assertEqual(len(raised.exception.model_attempts), 1)
        self.assertTrue(
            all(item["model_called"] for item in raised.exception.model_attempts)
        )

    def test_ollama_cache_identity_includes_generation_controls(self) -> None:
        call = CallRecord(source_file="/tmp/call.mp3", source_filename="call.mp3")
        context = {
            "llm_prompt": "prompt",
            "user_prompt": "user",
            "system_prompt": "system",
            "metrics": {"profile": "compact"},
        }
        identities = []

        for base_url, temperature, num_predict in (
            ("http://127.0.0.1:11434", 0.1, 400),
            ("http://127.0.0.1:11434", 0.2, 400),
            ("http://127.0.0.1:11434", 0.2, 800),
            ("http://127.0.0.1:11435", 0.2, 800),
        ):
            service = AnalyzeService(
                replace(
                    make_settings(),
                    analyze_provider="ollama",
                    ollama_base_url=base_url,
                    ollama_temperature=temperature,
                    analyze_ollama_num_predict=num_predict,
                )
            )
            with patch.object(
                service, "_analysis_prompt_context", return_value=context
            ), patch.object(
                service,
                "_analysis_cache_lookup",
                return_value={"structured_fields": {}, "claim_requests": []},
            ) as lookup:
                service._ollama_analysis(call, "dialogue", "compact")
            identities.append(lookup.call_args.kwargs["reasoning"])

        self.assertEqual(len(set(identities)), 4)
        self.assertIn("temperature=0.1", identities[0])
        self.assertIn("num_predict=800", identities[2])

    def test_runtime_controls_are_part_of_the_analysis_source_identity(self) -> None:
        call = _trusted_dialogue_call()
        identities = []
        for base_url, temperature in (
            ("http://127.0.0.1:11434", 0.0),
            ("http://127.0.0.1:11435", 0.0),
            ("http://127.0.0.1:11435", 0.2),
        ):
            service = AnalyzeService(
                replace(
                    make_settings(), analyze_provider="ollama",
                    ollama_base_url=base_url, ollama_temperature=temperature,
                )
            )
            identities.append(
                analysis_input_identity_sha256(
                    call, service._analysis_prompt_identity()
                )
            )

        self.assertEqual(len(set(identities)), 3)

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
            payload = _v3_answer(
                student={"grade_current": "8", "school": None},
                interests={
                    "products": ["годовые курсы"],
                    "format": ["онлайн"],
                    "subjects": ["математика"],
                    "exam_targets": [],
                },
            )

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
                self.assertIn("--ignore-user-config", cmd)
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
            self.assertEqual(first["structured_fields"], second["structured_fields"])
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
        user_prompt = str(service._analysis_prompt_context(DummyCall(), text, "compact")["user_prompt"])

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
            "structured_fields": _v3_fields(),
            "claim_requests": [],
        }
        full_payload = {
            "structured_fields": _v3_fields(
                interests={
                    "products": ["летний лагерь"], "format": [],
                    "subjects": [], "exam_targets": [],
                }
            ),
            "claim_requests": [
                {
                    "field_path": "structured_fields.interests.products",
                    "item_id": "летний лагерь",
                    "support_type": "explicit",
                    "turn_ids": ["T0001"],
                }
            ],
        }
        observed_profiles: list[str] = []
        observed_dialogues: list[Any] = []

        def fake_codex(call, text, profile=None, dialogue=None):
            observed_profiles.append(profile or "compact")
            observed_dialogues.append(dialogue)
            return compact_payload if (profile or "compact") == "compact" else full_payload

        dialogue = build_dialogue_input(call_record_view(_dialogue_call()))

        with patch.object(service, "_codex_cli_analysis", side_effect=fake_codex):
            payload = service._analyze_text(
                DummyCall(),
                "MANAGER:\nУ нас есть летний лагерь.\nCLIENT:\nРасскажите подробнее про лагерь.",
                dialogue,
            )

        self.assertEqual(observed_profiles, ["compact", "full"])
        # The canonical dialogue reaches the provider on both the compact call
        # and the escalated full one, unchanged.
        self.assertEqual(observed_dialogues, [dialogue, dialogue])
        self.assertEqual(
            payload["structured_fields"]["interests"]["products"], ["летний лагерь"]
        )

    def test_analyze_text_escalates_compact_when_false_non_conversation_claimed(self) -> None:
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

        # v3 has no ``tags``: a model that wants to call a live conversation
        # empty has to say so through ``result.status``, and that claim gets the
        # very same second opinion the old tag used to get.
        compact_payload = {
            "structured_fields": _v3_fields(
                result={"status": "non_conversation", "detail": None}
            ),
            "claim_requests": [],
        }
        full_payload = {
            "structured_fields": _v3_fields(
                next_step={"action": "Отправить материалы", "due": None}
            ),
            "claim_requests": [
                {
                    "field_path": "structured_fields.next_step.action",
                    "item_id": None,
                    "support_type": "explicit",
                    "turn_ids": ["T0001"],
                }
            ],
        }
        observed_profiles: list[str] = []
        observed_dialogues: list[Any] = []

        def fake_codex(call, text, profile=None, dialogue=None):
            observed_profiles.append(profile or "compact")
            observed_dialogues.append(dialogue)
            return compact_payload if (profile or "compact") == "compact" else full_payload

        with patch.object(service, "_codex_cli_analysis", side_effect=fake_codex):
            payload = service._analyze_text(
                DummyCall(),
                "MANAGER:\nПодскажите, открывается ли онлайн-тест?\nCLIENT:\nНет, ссылка не работает, нужна инструкция.",
            )

        self.assertEqual(observed_profiles, ["compact", "full"])
        self.assertEqual(observed_dialogues, [None, None])
        self.assertEqual(
            payload["structured_fields"]["next_step"]["action"], "Отправить материалы"
        )
        self.assertIsNone(payload["structured_fields"]["result"]["status"])

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
        # ТЗ-04 §7.4: the date and the manager have their own columns, so the
        # конспект carries neither — not once, not twice.
        self.assertNotIn("Клычева Дарья", summary)
        self.assertNotIn("28.01.2026", summary)
        self.assertIn("Клиент попросил отправить материалы на почту.", summary)

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
        self.assertEqual(summary.count("23.01.2026 09:00"), 0)
        self.assertNotIn("23.01.2026 в 09:00", summary)
        self.assertIn("клиент уточнил детали по курсу", summary)

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

    def test_manager_brief_is_readable_and_does_not_repeat_adjacent_columns(self) -> None:
        service = AnalyzeService(make_settings())
        brief = service._compose_manager_brief(
            _v3_fields(
                result={"status": "no_decision", "detail": "семья сравнивает варианты"},
                student={"grade_current": "8", "school": None},
                interests={
                    "products": ["летняя школа"],
                    "subjects": ["математика"],
                    "format": ["очно"],
                    "exam_targets": [],
                },
                objections=["цена"],
                next_step={"action": "Отправить договор", "due": "завтра"},
            )
        )

        self.assertIn("8 класс", brief)
        self.assertNotIn("летняя школа", brief)
        self.assertIn("математика", brief)
        self.assertNotIn("семья сравнивает варианты", brief)
        self.assertNotIn("цена", brief)
        self.assertNotIn("Отправить договор", brief)
        self.assertNotIn("завтра", brief)

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

    def test_reset_analysis_does_not_treat_legacy_null_resolve_as_terminal(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_reset_analysis_legacy_") as td:
            db_path = Path(td) / "reset.db"
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
                        analysis_status="done",
                        analysis_json=json.dumps(
                            {"history_summary": "must remain"},
                            ensure_ascii=False,
                        ),
                    )
                )
                session.commit()
            with sqlite3.connect(db_path) as connection:
                create_sql = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE name='call_records'"
                ).fetchone()[0]
                connection.execute("ALTER TABLE call_records RENAME TO legacy_source")
                connection.execute(
                    create_sql.replace(
                        "resolve_status VARCHAR(16) NOT NULL",
                        "resolve_status VARCHAR(16)",
                    )
                )
                columns = [
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(call_records)")
                ]
                selected = [
                    "NULL" if column == "resolve_status" else column
                    for column in columns
                ]
                connection.execute(
                    f"INSERT INTO call_records ({','.join(columns)}) "
                    f"SELECT {','.join(selected)} FROM legacy_source"
                )
                connection.execute("DROP TABLE legacy_source")

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
            self.assertEqual(json.loads(out.getvalue())["updated"], 0)
            with sqlite3.connect(db_path) as connection:
                state = connection.execute(
                    "SELECT resolve_status, analysis_status, analysis_json "
                    "FROM call_records"
                ).fetchone()
            self.assertEqual(
                state,
                (
                    None,
                    "done",
                    json.dumps(
                        {"history_summary": "must remain"},
                        ensure_ascii=False,
                    ),
                ),
            )

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


DIALOGUE_VARIANTS = {
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
        "[00:01.0] Дорожка левая: A1 добрый день, подскажу по математике",
        "[00:02.0] Дорожка правая: B1 здравствуйте, нужен курс ЕГЭ",
        "[00:03.0] Дорожка левая: A2 записываю, пришлю ссылку на оплату",
    ],
}


# The same three replies as the official ``recording_transcripts`` answer would
# describe them: a provider role, a physical track and the text.
DIALOGUE_TURNS = (
    ("operator", "left", "A1 добрый день, подскажу по математике"),
    ("client", "right", "B1 здравствуйте, нужен курс ЕГЭ"),
    ("operator", "left", "A2 записываю, пришлю ссылку на оплату"),
)


def _dialogue_call(**overrides):
    payload = {
        "source_file": "/tmp/mango/call.mp3",
        "source_filename": "call.mp3",
        "source_call_id": "call-7",
        "source_recording_id": fx.RECORDING_ID,
        "transcription_status": "done",
        "resolve_status": "done",
        "analysis_status": "pending",
        "transcript_text": "MANAGER:\nA1 A2\n\nCLIENT:\nB1",
        "transcript_variants_json": json.dumps(DIALOGUE_VARIANTS, ensure_ascii=False),
    }
    payload.update(overrides)
    return CallRecord(**payload)


def _trusted_dialogue_call(**overrides):
    """The same call, plus the Mango answer that really proves its sides."""
    variants = json.loads(json.dumps(DIALOGUE_VARIANTS, ensure_ascii=False))
    variants[fx.PROVIDER_EVIDENCE_FIELD] = fx.evidence(
        DIALOGUE_TURNS, source_call_id="call-7"
    )
    payload = {"transcript_variants_json": json.dumps(variants, ensure_ascii=False)}
    payload.update(overrides)
    return _dialogue_call(**payload)


class CanonicalDialogueAnalyseInputTest(unittest.TestCase):
    """Этап B: Analyse gets the chronological dialogue, never two monoliths."""

    def test_model_prompt_carries_a1_b1_a2_in_order_with_turn_ids(self) -> None:
        service = AnalyzeService(make_settings())
        call = _dialogue_call()
        dialogue = build_dialogue_input(call_record_view(call))

        context = service._analysis_prompt_context(
            call, dialogue.render(), "compact", dialogue
        )

        transcript = context["user_prompt"].split("Dialogue:\n", 1)[1]
        self.assertEqual(
            transcript.splitlines(),
            [
                "T0001 [00:01.0] Спикер A: A1 добрый день, подскажу по математике",
                "T0002 [00:02.0] Спикер B: B1 здравствуйте, нужен курс ЕГЭ",
                "T0003 [00:03.0] Спикер A: A2 записываю, пришлю ссылку на оплату",
            ],
        )
        self.assertNotIn("MANAGER:", context["user_prompt"])
        self.assertNotIn("Менеджер", transcript)

    def test_prompt_records_the_sha_of_the_exact_full_prompt(self) -> None:
        service = AnalyzeService(make_settings())
        call = _dialogue_call()
        dialogue = build_dialogue_input(call_record_view(call))

        context = service._analysis_prompt_context(
            call, dialogue.render(), "compact", dialogue
        )
        metrics = context["metrics"]

        self.assertEqual(
            metrics["analysis_prompt_sha256"],
            hashlib.sha256(context["llm_prompt"].encode("utf-8")).hexdigest(),
        )
        self.assertEqual(metrics["dialogue_canonical_sha256"], dialogue.canonical_sha256)
        self.assertEqual(
            metrics["dialogue_selected_turn_ids"], ["T0001", "T0002", "T0003"]
        )
        self.assertEqual(metrics["dialogue_total_turn_count"], 3)
        self.assertFalse(metrics["transcript_truncated"])

    def test_canonical_dialogue_compacts_only_filler_inside_existing_turns(self) -> None:
        variants = json.loads(json.dumps(DIALOGUE_VARIANTS, ensure_ascii=False))
        variants["dialogue_lines"] = [
            "[00:01.0] Дорожка левая: Да, да, да. Расскажу про математику",
            "[00:02.0] Дорожка правая: Хорошо, хорошо. Пришлите ссылку на оплату",
        ]
        service = AnalyzeService(make_settings())
        call = _dialogue_call(
            transcript_variants_json=json.dumps(variants, ensure_ascii=False)
        )
        dialogue = build_dialogue_input(call_record_view(call))
        canonical_before = dialogue.canonical_sha256

        metrics = service._dialogue_prompt_metrics(dialogue, "compact")

        self.assertTrue(metrics["transcript_compacted"])
        self.assertEqual(metrics["transcript_compaction_shortened_lines"], 2)
        self.assertEqual(metrics["transcript_compaction_removed_lines"], 0)
        self.assertEqual(metrics["dialogue_selected_turn_ids"], ["T0001", "T0002"])
        self.assertIn("T0001 [00:01.0]", metrics["transcript"])
        self.assertIn("T0002 [00:02.0]", metrics["transcript"])
        self.assertIn("математику", metrics["transcript"])
        self.assertIn("ссылку на оплату", metrics["transcript"])
        self.assertNotIn("Да, да, да", metrics["transcript"])
        self.assertEqual(dialogue.canonical_sha256, canonical_before)
        self.assertIn("Да, да, да", dialogue.render_for_analysis()["text"])

    def test_long_dialogue_is_cut_only_on_turn_boundaries(self) -> None:
        variants = json.loads(json.dumps(DIALOGUE_VARIANTS, ensure_ascii=False))
        variants["dialogue_lines"] = [
            "[00:{:02d}.0] Дорожка {}: Реплика {} {}".format(
                index, "левая" if index % 2 else "правая", index, "текст " * 200
            )
            for index in range(1, 13)
        ]
        service = AnalyzeService(make_settings())
        call = _dialogue_call(
            transcript_variants_json=json.dumps(variants, ensure_ascii=False)
        )
        dialogue = build_dialogue_input(call_record_view(call))

        metrics = service._dialogue_prompt_metrics(dialogue, "compact")
        whole = set(dialogue.render_for_analysis()["text"].splitlines())

        self.assertTrue(metrics["transcript_truncated"])
        self.assertLess(metrics["dialogue_selected_turn_count"], 12)
        for line in metrics["transcript"].splitlines():
            self.assertTrue(
                line == ANALYSIS_TRUNCATION_MARKER or line in whole,
                f"prompt line is not a whole turn: {line!r}",
            )

    def test_prompt_flags_reach_quality_flags(self) -> None:
        service = AnalyzeService(make_settings())
        call = _dialogue_call()
        dialogue = build_dialogue_input(call_record_view(call))
        context = service._analysis_prompt_context(
            call, dialogue.render(), "compact", dialogue
        )

        merged = service._with_analysis_prompt_quality_flags(
            {}, metrics=context["metrics"], prompt_version="v6", cache_hit=False
        )

        flags = merged["quality_flags"]
        self.assertEqual(flags["dialogue_source"], "dialogue_lines")
        self.assertEqual(
            flags["analysis_prompt_sha256"], context["metrics"]["analysis_prompt_sha256"]
        )


class RoleAttributionGuardTest(unittest.TestCase):
    """ТЗ-02: untrusted roles leave no role-dependent claim in analysis_json."""

    def _analysis(self):
        return {
            "structured_fields": {
                "people": {"parent_fio": "Иванова Мария", "child_fio": "Иванов Пётр"},
                "contacts": {
                    "email": "mama@example.com",
                    "phone_from_filename": "+70000000000",
                    "preferred_channel": "telegram",
                },
                "student": {"grade_current": "11", "school": "Лицей 1"},
                "interests": {
                    "products": ["Курс ЕГЭ"], "subjects": ["Математика"],
                    "format": [], "exam_targets": [],
                },
                "objections": ["Цена"],
                "next_step": {"action": "Отправить ссылку на оплату", "due": "завтра"},
                "lead_priority": "hot",
            },
            "crm_blocks": {
                "people": {"parent_fio": "Иванова Мария", "child_fio": "Иванов Пётр"},
                "contacts": {"email": "mama@example.com", "preferred_channel": "telegram"},
                "student": {"grade_current": "11", "school": "Лицей 1"},
                "next_step": {"action": "Отправить ссылку на оплату", "due": "завтра"},
            },
            "summary": "Менеджер пообещал прислать ссылку, клиент согласился оплатить.",
            "history_summary": "Клиент попросил счёт, менеджер отправил договор.",
            "history_short": "Клиент попросил счёт.",
            "evidence": [{"speaker": "Менеджер", "ts": "00:01.0", "text": "Пришлю ссылку"}],
            "next_step": "Отправить ссылку на оплату",
            "timeline": "завтра",
            "student_grade": "11",
            "personal_offer": "Скидка 10% клиенту",
            "target_product": "Курс ЕГЭ",
            "objections": ["Цена"],
            "follow_up_reason": "Клиент готов оплатить.",
            "needs_review": False,
            "review_reasons": [],
            "quality_flags": {
                "needs_review": False, "review_reasons": [], "call_type": "sales_call",
            },
        }

    def _untrusted_dialogue(self):
        return build_dialogue_input(call_record_view(_dialogue_call()))

    def _trusted_dialogue(self):
        """Provider evidence that really describes *this* stored dialogue."""
        return build_dialogue_input(call_record_view(_trusted_dialogue_call()))

    def test_untrusted_roles_clear_every_role_dependent_field(self) -> None:
        dialogue = self._untrusted_dialogue()
        self.assertFalse(dialogue.role_attribution["trusted"])

        guarded = apply_role_guard(self._analysis(), dialogue)

        # The payload is rebuilt from an allowlist, not stripped key by key:
        # the blocks that used to carry role-dependent claims are simply gone.
        self.assertEqual(guarded["structured_fields"], {})
        self.assertEqual(guarded["crm_blocks"], {})
        self.assertEqual(guarded["evidence"], [])
        self.assertEqual(guarded["objections"], [])
        self.assertEqual(guarded["tags"], [])
        for field in ("next_step", "timeline", "student_grade", "personal_offer",
                      "target_product"):
            self.assertNotIn(field, guarded)
        # The dangerous raw model candidate is not kept anywhere.
        payload = json.dumps(guarded, ensure_ascii=False)
        for leaked in (
            "Иванова Мария", "Иванов Пётр", "mama@example.com", "telegram",
            "Лицей 1", "Отправить ссылку на оплату", "пообещал", "Скидка 10%",
            "Курс ЕГЭ", "Математика", "hot", "Цена",
        ):
            self.assertNotIn(leaked, payload)

    def test_untrusted_roles_keep_only_a_deterministic_neutral_topic(self) -> None:
        guarded = apply_role_guard(
            self._analysis(), self._untrusted_dialogue()
        )

        # The topic comes from the closed vocabulary applied to the dialogue
        # text, never from the model's own words about the call.
        self.assertEqual(
            guarded["neutral_topics"],
            ["математика", "подготовка к ЕГЭ", "стоимость и оплата"],
        )
        self.assertEqual(guarded["summary"], guarded["history_summary"])
        self.assertEqual(guarded["history_short"], guarded["history_summary"])
        self.assertNotIn("Менеджер", guarded["summary"])

    def test_untrusted_roles_agree_between_top_object_and_quality_flags(self) -> None:
        guarded = apply_role_guard(
            self._analysis(), self._untrusted_dialogue()
        )
        flags = guarded["quality_flags"]

        self.assertTrue(guarded["needs_review"])
        self.assertTrue(flags["needs_review"])
        self.assertIn("role_attribution_untrusted", guarded["review_reasons"])
        self.assertEqual(flags["review_reasons"], guarded["review_reasons"])
        self.assertTrue(flags["role_attribution_untrusted"])
        self.assertEqual(flags["role_attribution_decision"], "untrusted")
        self.assertEqual(
            flags["role_attribution_reason_codes"],
            guarded["role_attribution"]["reason_codes"],
        )
        self.assertEqual(
            guarded["dialogue_input"]["canonical_sha256"],
            flags["dialogue_canonical_sha256"],
        )

    def test_trusted_roles_keep_the_role_dependent_answer(self) -> None:
        dialogue = self._trusted_dialogue()
        self.assertTrue(dialogue.role_attribution["trusted"])

        guarded = apply_role_guard(self._analysis(), dialogue)

        self.assertEqual(guarded["next_step"], "Отправить ссылку на оплату")
        self.assertEqual(
            guarded["structured_fields"]["people"]["parent_fio"], "Иванова Мария"
        )
        self.assertFalse(guarded["quality_flags"]["role_attribution_untrusted"])
        self.assertFalse(guarded["needs_review"])


class AnalyzeStaleResultGuardTest(unittest.TestCase):
    """Этап B/G: a lost lease or a changed input never overwrites a newer row."""

    def _prepare(self, td):
        db_path = Path(td) / "stale.db"
        settings = replace(
            make_settings(),
            database_url=f"sqlite:///{db_path}",
            transcript_export_dir=str(Path(td) / "export"),
        )
        init_db(settings)
        session_factory = build_session_factory(settings)
        with session_factory() as session:
            # A proven call on purpose: only a trusted dialogue reaches the
            # model at all, so this is the row where the race actually exists.
            session.add(
                _trusted_dialogue_call(source_file=str(Path(td) / "call.mp3"))
            )
            session.commit()
        return settings, session_factory

    @staticmethod
    def _export_files(settings):
        root = Path(settings.transcript_export_dir)
        return sorted(path.name for path in root.rglob("*") if path.is_file())

    def test_lost_lease_rejects_the_write_and_exports_no_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_stale_") as td:
            settings, session_factory = self._prepare(td)
            service = AnalyzeService(settings)

            def steal(_self, _call, _text, _dialogue=None):
                with session_factory() as thief:
                    thief.execute(
                        sa_text(
                            "UPDATE call_records SET analysis_worker_id = 'other-worker'"
                        )
                    )
                    thief.commit()
                return {"summary": "нормальный ответ"}

            with patch.object(AnalyzeService, "_analyze_text", steal):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            self.assertEqual(result["success"], 0)
            with session_factory() as session:
                row = session.query(CallRecord).one()
                self.assertEqual(row.analysis_status, "in_progress")
                self.assertEqual(row.analysis_worker_id, "other-worker")
                self.assertIsNone(row.analysis_json)
            self.assertEqual(self._export_files(settings), [])

    def test_changed_input_rejects_the_write_and_exports_no_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_changed_") as td:
            settings, session_factory = self._prepare(td)
            service = AnalyzeService(settings)

            def mutate(_self, _call, _text, _dialogue=None):
                with session_factory() as writer:
                    writer.execute(
                        sa_text(
                            "UPDATE call_records SET transcript_text = 'другой текст'"
                        )
                    )
                    writer.commit()
                return {"summary": "ответ по старому входу"}

            with patch.object(AnalyzeService, "_analyze_text", mutate):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            self.assertEqual(result["success"], 0)
            with session_factory() as session:
                row = session.query(CallRecord).one()
                self.assertIsNone(row.analysis_json)
                self.assertEqual(row.analysis_status, "in_progress")
            self.assertEqual(self._export_files(settings), [])

    @staticmethod
    def _race_after_model_result(session_factory, sql, params=None):
        """Commit a foreign change in the gap between the model and the write.

        ``_with_analysis_runtime_metadata`` is the last step before the
        conditional finalization, so patching it reproduces exactly the window
        that a separate check-then-write pair cannot cover.
        """
        original = AnalyzeService._with_analysis_runtime_metadata

        def racing(analysis):
            with session_factory() as other:
                other.execute(sa_text(sql), params or {})
                other.commit()
            return original(analysis)

        return patch.object(
            AnalyzeService, "_with_analysis_runtime_metadata", staticmethod(racing)
        )

    def test_worker_stolen_after_model_result_is_rejected_by_the_conditional_update(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_race_worker_") as td:
            settings, session_factory = self._prepare(td)
            service = AnalyzeService(settings)

            with self._race_after_model_result(
                session_factory,
                "UPDATE call_records SET analysis_worker_id = 'other-worker'",
            ):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            self.assertEqual(result["success"], 0)
            with session_factory() as session:
                row = session.query(CallRecord).one()
                self.assertEqual(row.analysis_worker_id, "other-worker")
                self.assertEqual(row.analysis_status, "in_progress")
                self.assertIsNone(row.analysis_json)
                self.assertIsNone(row.last_error)
                self.assertEqual(row.sync_status, "pending")
            self.assertEqual(self._export_files(settings), [])

    def test_commit_ack_loss_keeps_one_attempt_and_returns_success(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_ack_loss_") as td:
            settings, session_factory = self._prepare(td)
            service = AnalyzeService(replace(settings, analyze_provider="codex_cli"))
            usage = {
                "source": "provider_exact", "prompt_tokens": 10,
                "completion_tokens": 2, "total_tokens": 12,
            }
            answer = _v3_answer()
            answer["quality_flags"] = {
                "analyze_attempts": [{
                    "provider": "codex_cli", "model": settings.codex_analyze_model,
                    "profile": "compact", "prompt_version": "v8",
                    "model_called": True, "cache_hit": False, "token_usage": usage,
                }],
                "analyze_model_call_count": 1,
                "analyze_cache_hit_count": 0,
                "analyze_token_usage": usage,
            }

            with patch.object(AnalyzeService, "_analyze_text", return_value=answer):
                with session_factory() as session:
                    real_commit = session.commit
                    injected = {"done": False}

                    def lose_final_ack():
                        real_commit()
                        with sqlite3.connect(Path(td) / "stale.db") as connection:
                            status = connection.execute(
                                "SELECT analysis_status FROM call_records"
                            ).fetchone()[0]
                        if status == "done" and not injected["done"]:
                            injected["done"] = True
                            raise RuntimeError("commit acknowledgement lost")

                    session.commit = lose_final_ack
                    result = service.run(session, limit=1)

            self.assertEqual(result["success"], 1)
            self.assertEqual(result["failed"], 0)
            self.assertEqual(result["stale"], 0)
            with session_factory() as session:
                row = session.query(CallRecord).one()
                attempts = json.loads(row.analysis_attempts_json)
            self.assertEqual(row.analysis_status, "done")
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["state"], "completed")
            self.assertEqual(attempts[0]["token_usage"], usage)

    def test_model_answer_enters_cache_only_after_analysis_commit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_cache_order_") as td:
            settings, session_factory = self._prepare(td)
            settings = replace(
                settings, analyze_provider="codex_cli", llm_cache_enabled=True,
                llm_cache_dir=str(Path(td) / "cache"),
                analyze_escalate_full_on_ambiguity=False,
            )
            service = AnalyzeService(settings)
            observed_statuses = []

            def fake_run(cmd, **_kwargs):
                Path(cmd[cmd.index("--output-last-message") + 1]).write_text(
                    json.dumps(_v3_answer(), ensure_ascii=False), encoding="utf-8"
                )
                return CompletedProcess(cmd, 0, stdout="", stderr="")

            def guarded_store(**_kwargs):
                with session_factory() as audit_session:
                    observed_statuses.append(
                        audit_session.query(CallRecord).one().analysis_status
                    )

            with patch("mango_mvp.services.analyze.shutil.which", return_value="/bin/codex"), patch(
                "mango_mvp.services.analyze.subprocess.run", side_effect=fake_run
            ), patch.object(service, "_analysis_cache_store", side_effect=guarded_store):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["success"], 1)
            self.assertEqual(observed_statuses, ["done"])
            with session_factory() as session:
                attempt = json.loads(
                    session.query(CallRecord).one().analysis_attempts_json
                )[0]
            self.assertEqual(attempt["state"], "completed")

    def test_stolen_lease_appends_exact_attempt_only_to_the_technical_ledger(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_stale_usage_") as td:
            settings, session_factory = self._prepare(td)
            settings = replace(settings, analyze_provider="codex_cli")
            service = AnalyzeService(settings)
            previous = {"attempt_id": "previous:1", "model_called": True}
            usage = {
                "source": "provider_exact",
                "prompt_tokens": 17,
                "completion_tokens": 5,
                "total_tokens": 22,
            }
            current = {
                "provider": "codex_cli",
                "model": settings.codex_analyze_model,
                "profile": "compact",
                "prompt_version": service._analysis_prompt_version(),
                "cache_hit": False,
                "model_called": True,
                "token_usage": usage,
            }
            answer = _v3_answer()
            answer["quality_flags"] = {
                "analyze_attempts": [current],
                "analyze_model_call_count": 1,
                "analyze_cache_hit_count": 0,
                "analyze_token_usage": usage,
                "analyze_prompt_profile": "compact",
                "analyze_prompt_version": service._analysis_prompt_version(),
            }
            with session_factory() as session:
                row = session.query(CallRecord).one()
                row.analysis_attempts_json = json.dumps([previous])
                session.commit()

            stolen_result = json.dumps({"owner": "other-worker"})
            stolen_updated_at = "2026-08-17 12:34:56"
            with patch.object(AnalyzeService, "_analyze_text", return_value=answer):
                with self._race_after_model_result(
                    session_factory,
                    """UPDATE call_records
                       SET analysis_worker_id = 'other-worker',
                           analysis_json = :analysis_json,
                           updated_at = :updated_at""",
                    {"analysis_json": stolen_result, "updated_at": stolen_updated_at},
                ):
                    with session_factory() as session:
                        result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            with sqlite3.connect(Path(td) / "stale.db") as connection:
                stored = connection.execute(
                    """SELECT analysis_worker_id, analysis_status, analysis_json,
                              analysis_attempts_json, analyze_attempts, updated_at
                         FROM call_records"""
                ).fetchone()
            attempts = json.loads(stored[3])
            self.assertEqual(stored[:3], ("other-worker", "in_progress", stolen_result))
            self.assertEqual(stored[4:], (0, stolen_updated_at))
            self.assertEqual(attempts[0], previous)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[1]["token_usage"], usage)
            self.assertTrue(attempts[1]["attempt_id"])
            self.assertEqual(len(attempts[1]["analysis_source_sha256"]), 64)

            # Replaying the same CAS append is idempotent and does not touch time.
            with session_factory() as session:
                row = session.query(CallRecord).one()
                self.assertTrue(
                    service._store_analysis_attempt(
                        session,
                        call_id=row.id,
                        snapshot=analysis_input_snapshot(row),
                        attempt=attempts[1],
                        replace=False,
                    )
                )
            with sqlite3.connect(Path(td) / "stale.db") as connection:
                replayed = connection.execute(
                    "SELECT analysis_attempts_json, updated_at FROM call_records"
                ).fetchone()
            self.assertEqual(json.loads(replayed[0]), attempts)
            self.assertEqual(replayed[1], stolen_updated_at)

    def test_stolen_lease_with_changed_input_keeps_only_the_cost_ledger(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_stale_input_usage_") as td:
            settings, session_factory = self._prepare(td)
            settings = replace(settings, analyze_provider="codex_cli")
            service = AnalyzeService(settings)
            usage = {
                "source": "provider_exact",
                "prompt_tokens": 17,
                "completion_tokens": 5,
                "total_tokens": 22,
            }
            answer = _v3_answer()
            answer["quality_flags"] = {
                "analyze_attempts": [{"model_called": True, "token_usage": usage}],
                "analyze_model_call_count": 1,
                "analyze_cache_hit_count": 0,
            }

            with patch.object(AnalyzeService, "_analyze_text", return_value=answer):
                with self._race_after_model_result(
                    session_factory,
                    """UPDATE call_records
                       SET analysis_worker_id = 'other-worker',
                           transcript_text = 'changed input'""",
                ):
                    with session_factory() as session:
                        result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            with session_factory() as session:
                row = session.query(CallRecord).one()
                self.assertEqual(row.analysis_worker_id, "other-worker")
                self.assertEqual(row.transcript_text, "changed input")
                attempts = json.loads(row.analysis_attempts_json)
                self.assertEqual(len(attempts), 1)
                self.assertEqual(attempts[0]["state"], "completed")
                self.assertEqual(attempts[0]["token_usage"], usage)
                self.assertIsNone(row.analysis_json)

    def test_transcript_changed_after_model_result_is_rejected_and_kept_intact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_race_input_") as td:
            settings, session_factory = self._prepare(td)
            service = AnalyzeService(settings)
            changed = json.dumps(
                {"dialogue_lines": ["[00:09.0] Дорожка левая: Новое"]},
                ensure_ascii=False,
            )

            with self._race_after_model_result(
                session_factory,
                "UPDATE call_records SET transcript_variants_json = :value",
                {"value": changed},
            ):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            self.assertEqual(result["success"], 0)
            with session_factory() as session:
                row = session.query(CallRecord).one()
                # Neither the old nor the new value was overwritten by our answer.
                self.assertEqual(row.transcript_variants_json, changed)
                self.assertIsNone(row.analysis_json)
                self.assertEqual(row.analysis_status, "in_progress")
                self.assertEqual(row.analysis_worker_id[:3], "an-")
            self.assertEqual(self._export_files(settings), [])

    def test_any_prompt_or_export_identity_change_rejects_the_write(self) -> None:
        """Not only the transcript: everything the prompt or the file name uses.

        ``manager_name``, ``phone``, ``direction`` and ``started_at`` are part
        of the prompt metadata, and ``source_file``/``source_filename`` decide
        where the artefact is written.  A guard that watched only the transcript
        would happily attach an answer to a call that is no longer the one it
        was produced for.
        """
        columns = {
            "manager_name": "'Другой менеджер'",
            "phone": "'+79990000000'",
            "direction": "'inbound'",
            "started_at": "'2026-08-16 10:00:00'",
            "source_filename": "'other.mp3'",
            "source_file": "'/tmp/mango/other.mp3'",
            "duration_sec": "123.5",
        }
        for column, value in columns.items():
            with self.subTest(column=column):
                with tempfile.TemporaryDirectory(
                    prefix=f"mango_analyze_{column}_"
                ) as td:
                    settings, session_factory = self._prepare(td)
                    service = AnalyzeService(settings)

                    with self._race_after_model_result(
                        session_factory,
                        f"UPDATE call_records SET {column} = {value}",
                    ):
                        with session_factory() as session:
                            result = service.run(session, limit=1)

                    self.assertEqual(result["stale"], 1)
                    self.assertEqual(result["success"], 0)
                    with session_factory() as session:
                        row = session.query(CallRecord).one()
                        self.assertIsNone(row.analysis_json)
                        self.assertEqual(row.analysis_status, "in_progress")
                    self.assertEqual(self._export_files(settings), [])

    def test_a_stolen_lease_on_the_failure_path_never_marks_the_new_owner_dead(
        self,
    ) -> None:
        """The error path is a write too, so it is conditional as well."""
        with tempfile.TemporaryDirectory(prefix="mango_analyze_fail_race_") as td:
            settings, session_factory = self._prepare(td)
            service = AnalyzeService(settings)

            def steal_then_fail(_self, _call, _text, _dialogue=None):
                with session_factory() as thief:
                    thief.execute(
                        sa_text(
                            "UPDATE call_records SET analysis_worker_id = 'other-worker'"
                        )
                    )
                    thief.commit()
                raise RuntimeError("провайдер недоступен")

            with patch.object(AnalyzeService, "_analyze_text", steal_then_fail):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            self.assertEqual(result["failed"], 0)
            with session_factory() as session:
                row = session.query(CallRecord).one()
                self.assertEqual(row.analysis_worker_id, "other-worker")
                self.assertEqual(row.analysis_status, "in_progress")
                self.assertIsNone(row.last_error)
                self.assertIsNone(row.dead_letter_stage)

    def test_paid_failed_attempt_survives_a_stolen_lease_without_touching_owner_fields(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_fail_usage_race_") as td:
            settings, session_factory = self._prepare(td)
            settings = replace(settings, analyze_provider="codex_cli")
            service = AnalyzeService(settings)
            usage = {
                "source": "provider_exact",
                "prompt_tokens": 19,
                "completion_tokens": 3,
                "total_tokens": 22,
            }
            paid_attempt = {
                "provider": "codex_cli",
                "model": settings.codex_analyze_model,
                "profile": "compact",
                "prompt_version": service._analysis_prompt_version(),
                "cache_hit": False,
                "model_called": True,
                "token_usage": usage,
            }
            stolen_result = json.dumps({"owner": "other-worker"})
            stolen_updated_at = "2026-08-17 13:45:00"

            def steal_then_fail(_self, _call, _text, _dialogue=None):
                with session_factory() as thief:
                    thief.execute(
                        sa_text(
                            """UPDATE call_records
                               SET analysis_worker_id = 'other-worker',
                                   analysis_json = :analysis_json,
                                   updated_at = :updated_at"""
                        ),
                        {"analysis_json": stolen_result, "updated_at": stolen_updated_at},
                    )
                    thief.commit()
                error = RuntimeError("provider failed after paid call")
                error.model_attempts = [paid_attempt]
                raise error

            with patch.object(AnalyzeService, "_analyze_text", steal_then_fail):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            self.assertEqual(result["failed"], 0)
            with sqlite3.connect(Path(td) / "stale.db") as connection:
                stored = connection.execute(
                    """SELECT analysis_worker_id, analysis_status, analysis_json,
                              analysis_attempts_json, analyze_attempts, last_error,
                              dead_letter_stage, updated_at
                         FROM call_records"""
                ).fetchone()
            attempts = json.loads(stored[3])
            self.assertEqual(stored[:3], ("other-worker", "in_progress", stolen_result))
            self.assertEqual(stored[4:], (0, None, None, stolen_updated_at))
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["token_usage"], usage)
            self.assertTrue(attempts[0]["attempt_id"])
            self.assertEqual(attempts[0]["state"], "failed")
            self.assertEqual(len(attempts[0]["analysis_source_sha256"]), 64)

    def test_a_provider_error_is_stored_without_leaking_the_conversation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_error_text_") as td:
            settings, session_factory = self._prepare(td)
            service = AnalyzeService(settings)
            secret = "клиент Мария Иванова, телефон +79990000000, оплата 60000"

            def leaky(_self, _call, _text, _dialogue=None):
                raise RuntimeError(f"provider echoed the prompt back: {secret} " * 20)

            with patch.object(AnalyzeService, "_analyze_text", leaky):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["failed"], 1)
            with session_factory() as session:
                row = session.query(CallRecord).one()
            self.assertLessEqual(len(row.last_error), 260)
            self.assertNotIn("+79990000000", row.last_error)
            self.assertIn("RuntimeError", row.last_error)
            # Not only the phone: the leak is at the *front* of the message, so
            # the name, the price and the echoed prompt must be gone as well —
            # a bounded prefix of this message would have kept all three.
            self.assertNotIn("Мария", row.last_error)
            self.assertNotIn("Иванова", row.last_error)
            self.assertNotIn("provider echoed the prompt", row.last_error)
            self.assertNotIn("клиент", row.last_error.lower())
            self.assertIn("message_sha256=", row.last_error)
            # The digest is hex, so a decimal price is only checked against the
            # readable part — a hash cannot be searched for a chosen substring.
            self.assertNotIn("60000", row.last_error.split("message_sha256=")[0])

    def test_a_failed_export_is_counted_and_never_stops_the_batch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_export_fail_") as td:
            settings, session_factory = self._prepare(td)
            with session_factory() as session:
                session.add(
                    _dialogue_call(
                        source_file=str(Path(td) / "call2.mp3"),
                        source_filename="call2.mp3",
                        source_call_id="call-8",
                        source_recording_id="rec-8",
                    )
                )
                session.commit()
            service = AnalyzeService(settings)
            original = AnalyzeService._export_analysis_files

            def flaky(self, call, analysis):
                if str(call.source_filename) == "call.mp3":
                    raise OSError("export target is read-only")
                return original(self, call, analysis)

            with patch.object(AnalyzeService, "_export_analysis_files", flaky):
                with session_factory() as session:
                    result = service.run(session, limit=2)

            # The artefact failed; the analysis is still committed for both.
            self.assertEqual(result["success"], 2)
            self.assertEqual(result["export_failed"], 1)
            self.assertEqual(result["stale"], 0)
            with session_factory() as session:
                statuses = sorted(
                    row.analysis_status for row in session.query(CallRecord).all()
                )
            self.assertEqual(statuses, ["done", "done"])

    def test_unchanged_lease_and_input_commit_and_export(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_ok_") as td:
            settings, session_factory = self._prepare(td)
            service = AnalyzeService(settings)

            with session_factory() as session:
                result = service.run(session, limit=1)

            self.assertEqual(result["success"], 1)
            self.assertEqual(result["stale"], 0)
            self.assertEqual(result["role_attribution_trusted"], 1)
            self.assertEqual(result["role_attribution_untrusted"], 0)
            with session_factory() as session:
                row = session.query(CallRecord).one()
                self.assertEqual(row.analysis_status, "done")
                analysis = json.loads(row.analysis_json)
            self.assertTrue(analysis["role_attribution"]["trusted"])
            self.assertFalse(analysis["quality_flags"]["role_attribution_untrusted"])
            self.assertTrue(self._export_files(settings))

    def test_an_unproven_call_commits_and_exports_without_calling_the_model(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_analyze_untrusted_ok_") as td:
            settings, session_factory = self._prepare(td)
            with session_factory() as session:
                session.query(CallRecord).delete()
                session.add(
                    _dialogue_call(source_file=str(Path(td) / "call.mp3"))
                )
                session.commit()
            service = AnalyzeService(settings)

            with session_factory() as session:
                result = service.run(session, limit=1)

            self.assertEqual(result["success"], 1)
            self.assertEqual(result["role_attribution_trusted"], 0)
            self.assertEqual(result["role_attribution_untrusted"], 1)
            with session_factory() as session:
                row = session.query(CallRecord).one()
                self.assertEqual(row.analysis_status, "done")
                analysis = json.loads(row.analysis_json)
            self.assertFalse(analysis["role_attribution"]["trusted"])
            self.assertTrue(analysis["quality_flags"]["role_attribution_untrusted"])
            self.assertIn("role_attribution_untrusted", analysis["review_reasons"])
            self.assertTrue(self._export_files(settings))


class AnalyzeUntrustedCostTest(unittest.TestCase):
    """ТЗ-02 R3: an unproven call costs zero tokens and zero cache lookups.

    Asking the model and then deleting its role-dependent answer is not merely
    wasteful — the deleted guess lives in the process next to the fields that do
    get published, one refactor away from leaking into a neighbouring key.  So
    the model is not asked at all, and the published result is built
    deterministically from the dialogue itself.
    """

    def _run(self, call, prefix):
        """Run one call with every model and cache entry point tripwired."""
        touched: list[str] = []
        with tempfile.TemporaryDirectory(prefix=prefix) as td:
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{Path(td) / 'cost.db'}",
                transcript_export_dir=str(Path(td) / "export"),
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            call.source_file = str(Path(td) / "call.mp3")
            with session_factory() as session:
                session.add(call)
                session.commit()
            service = AnalyzeService(settings)

            def tripwire(name):
                def fail(*_args, **_kwargs):
                    touched.append(name)
                    raise AssertionError(name + " must not be reached")

                return fail

            patches = [
                patch.object(AnalyzeService, name, tripwire(name))
                for name in (
                    "_openai_analysis",
                    "_ollama_analysis",
                    "_codex_cli_analysis",
                    "_mock_analysis",
                    "_analysis_cache_lookup",
                    "_analysis_cache_store",
                )
            ]
            for item in patches:
                item.start()
            try:
                with session_factory() as session:
                    result = service.run(session, limit=1)
            finally:
                for item in reversed(patches):
                    item.stop()
            with session_factory() as session:
                row = session.query(CallRecord).one()
                analysis = json.loads(row.analysis_json or "null")
        return result, analysis, touched

    def test_an_unproven_call_never_reaches_the_provider_or_the_cache(self) -> None:
        result, analysis, touched = self._run(_dialogue_call(), "mango_cost_untrusted_")

        self.assertEqual(touched, [])
        self.assertEqual(result["success"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertFalse(analysis["role_attribution"]["trusted"])
        self.assertEqual(analysis["summary"], UNTRUSTED_SUMMARY)
        self.assertIn("role_attribution_untrusted", analysis["review_reasons"])
        self.assertTrue(analysis["needs_review"])
        self.assertEqual(analysis["structured_fields"], {})

    def test_the_skipped_call_is_reported_as_skipped_and_not_as_unknown(self) -> None:
        _result, analysis, _touched = self._run(
            _dialogue_call(), "mango_cost_untrusted_meta_"
        )
        meta = analysis["analysis_meta"]

        self.assertIs(meta["model_called"], False)
        self.assertIs(meta["cache_hit"], False)
        self.assertEqual(meta["token_usage"]["source"], "skipped_untrusted_role")
        self.assertIsNone(meta["token_usage"]["prompt_tokens"])
        self.assertIsNone(meta["token_usage"]["completion_tokens"])
        self.assertIsNone(meta["token_usage"]["total_tokens"])
        # Identity is still recorded: a skipped call stays auditable.
        self.assertEqual(meta["analysis_provider"], "mock")
        self.assertTrue(meta["analysis_prompt_version"])
        self.assertEqual(len(meta["analysis_source_sha256"]), 64)

    def test_a_proven_call_still_goes_through_the_normal_model_path(self) -> None:
        """Negative control: the skip is about trust, not about the code path."""
        seen: list[str] = []

        def spy(_self, _call, text, _dialogue=None):
            seen.append(text)
            return {"summary": "ответ модели"}

        with tempfile.TemporaryDirectory(prefix="mango_cost_trusted_") as td:
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{Path(td) / 'cost.db'}",
                transcript_export_dir=str(Path(td) / "export"),
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    _trusted_dialogue_call(source_file=str(Path(td) / "call.mp3"))
                )
                session.commit()
            with patch.object(AnalyzeService, "_analyze_text", spy):
                with session_factory() as session:
                    result = AnalyzeService(settings).run(session, limit=1)

        self.assertEqual(result["success"], 1)
        self.assertEqual(len(seen), 1)
        # And the model saw the canonical dialogue, with the proven sides named.
        self.assertIn("Менеджер:", seen[0])
        self.assertIn("Клиент:", seen[0])


class AnalyzePromptIdentityTest(unittest.TestCase):
    """The recorded prompt identity is the one that actually ran."""

    def test_an_escalated_analysis_records_full_and_not_the_compact_guess(
        self,
    ) -> None:
        settings = replace(make_settings(), analyze_provider="openai")
        service = AnalyzeService(settings)
        call = _trusted_dialogue_call()
        dialogue = build_dialogue_input(call_record_view(call))
        profiles: list[str] = []

        def fake(self, _call, _text, profile=None, _dialogue=None):
            resolved = self._analysis_prompt_profile(profile)
            profiles.append(resolved)
            return self._with_analysis_prompt_quality_flags(
                _v3_answer(),
                metrics={"profile": resolved},
                prompt_version=self._analysis_prompt_version(resolved),
                cache_hit=False,
            )

        with patch.object(AnalyzeService, "_openai_analysis", fake), patch.object(
            AnalyzeService, "_should_escalate_full_profile", lambda *_a, **_k: True
        ):
            payload = service._analyze_text(call, dialogue.render(), dialogue)

        meta = service._build_analysis_meta(payload)

        # The first request was compact; escalation re-asked with ``full``.
        self.assertEqual(profiles, ["compact", "full"])
        self.assertEqual(meta["analysis_prompt_profile"], "full")
        self.assertEqual(
            meta["analysis_prompt_version"], service._analysis_prompt_version("full")
        )
        self.assertIs(meta["model_called"], True)
        self.assertEqual(meta["model_call_count"], 2)
        self.assertEqual(meta["cache_hit_count"], 0)
        self.assertEqual(len(meta["model_attempts"]), 2)
        self.assertIs(meta["cache_hit"], False)
        self.assertEqual(meta["token_usage"]["source"], "unavailable")


class ClaimEvidenceContractTest(unittest.TestCase):
    """Этап C: a published high-risk value always points at a real reply."""

    def _service(self):
        return AnalyzeService(make_settings())

    def _call_and_dialogue(self):
        call = _trusted_dialogue_call()
        return call, build_dialogue_input(call_record_view(call))

    @staticmethod
    def _selected(*turn_ids):
        return {"quality_flags": {"dialogue_selected_turn_ids": list(turn_ids)}}

    def test_valid_model_claim_gets_a_service_written_quote_and_timecode(self) -> None:
        service = self._service()
        call, dialogue = self._call_and_dialogue()
        analysis = {
            "structured_fields": _v3_fields(
                next_step={"action": "Отправить ссылку на оплату", "due": None}
            ),
            **self._selected("T0001", "T0002", "T0003"),
        }

        guarded = service._apply_claim_evidence(
            call,
            analysis,
            dialogue,
            [
                {
                    "field_path": "structured_fields.next_step.action",
                    "item_id": None,
                    "support_type": "explicit",
                    "turn_ids": ["T0003"],
                }
            ],
        )

        self.assertEqual(
            guarded["structured_fields"]["next_step"]["action"],
            "Отправить ссылку на оплату",
        )
        entry = next(
            item
            for item in guarded["claim_evidence"]
            if item["field_path"] == "structured_fields.next_step.action"
        )
        self.assertEqual(entry["turn_id"], "T0003")
        self.assertEqual(entry["exact_quote"], dialogue.turns[2]["text"])
        self.assertEqual(entry["timecode"], dialogue.turns[2]["timecode"])
        self.assertEqual(entry["speaker_kind"], dialogue.turns[2]["speaker_kind"])
        self.assertEqual(entry["dialogue_sha256"], dialogue.canonical_sha256)
        self.assertEqual(entry["evidence_type"], "explicit")
        self.assertEqual(guarded["analysis_schema_version"], "v3")

    def test_missing_foreign_and_inferred_references_all_fail_closed(self) -> None:
        cases = (
            ("no claim at all", None),
            (
                "reference to a reply this call never had",
                {
                    "field_path": "structured_fields.next_step.action",
                    "item_id": None,
                    "support_type": "explicit",
                    "turn_ids": ["T0099"],
                },
            ),
            (
                "inferred instead of explicit",
                {
                    "field_path": "structured_fields.next_step.action",
                    "item_id": None,
                    "support_type": "inferred",
                    "turn_ids": ["T0003"],
                },
            ),
        )
        for label, request in cases:
            with self.subTest(label):
                service = self._service()
                call, dialogue = self._call_and_dialogue()
                analysis = {
                    "structured_fields": _v3_fields(
                        next_step={"action": "Выставить счёт на 50 000", "due": "завтра"}
                    ),
                    **self._selected("T0001", "T0002", "T0003"),
                }

                guarded = service._apply_claim_evidence(
                    call, analysis, dialogue, [request] if request else []
                )

                self.assertIsNone(guarded["structured_fields"]["next_step"]["action"])
                self.assertIsNone(guarded["structured_fields"]["next_step"]["due"])
                self.assertTrue(guarded["needs_review"])
                self.assertIn(
                    "claim_evidence_missing_or_invalid:structured_fields.next_step.action",
                    guarded["review_reasons"],
                )
                self.assertNotIn("50 000", json.dumps(guarded, ensure_ascii=False))

    def test_reference_to_a_turn_cut_out_of_the_prompt_is_not_evidence(self) -> None:
        service = self._service()
        call, dialogue = self._call_and_dialogue()
        analysis = {
            "structured_fields": _v3_fields(
                people={"parent_fio": "Иванова Мария", "child_fio": None}
            ),
            # T0003 exists in the call but never reached the model.
            **self._selected("T0001", "T0002"),
        }

        guarded = service._apply_claim_evidence(
            call,
            analysis,
            dialogue,
            [
                {
                    "field_path": "structured_fields.people.parent_fio",
                    "item_id": None,
                    "support_type": "explicit",
                    "turn_ids": ["T0003"],
                }
            ],
        )

        self.assertIsNone(guarded["structured_fields"]["people"]["parent_fio"])
        self.assertEqual(guarded["claim_evidence"], [])

    def test_an_unrelated_existing_turn_cannot_prove_invented_free_text(self) -> None:
        service = self._service()
        call, dialogue = self._call_and_dialogue()
        analysis = {
            "structured_fields": _v3_fields(
                people={"parent_fio": "Иванова Мария", "child_fio": None},
                student={"grade_current": None, "school": "Лицей 1535"},
                next_step={"action": "Позвонить директору завтра", "due": "18.08.2026"},
            ),
            **self._selected("T0001", "T0002", "T0003"),
        }
        requests = [
            {
                "field_path": path,
                "item_id": None,
                "support_type": "explicit",
                # T0002 is real, selected and trusted, but says only that a course
                # is needed; it contains none of the values above.
                "turn_ids": ["T0002"],
            }
            for path in (
                "structured_fields.people.parent_fio",
                "structured_fields.student.school",
                "structured_fields.next_step.action",
                "structured_fields.next_step.due",
            )
        ]

        guarded = service._apply_claim_evidence(
            call, analysis, dialogue, requests
        )

        payload = json.dumps(guarded, ensure_ascii=False)
        for invented in (
            "Иванова Мария", "Лицей 1535", "Позвонить директору завтра", "18.08.2026"
        ):
            self.assertNotIn(invented, payload)
        self.assertEqual(guarded["claim_evidence"], [])
        self.assertTrue(guarded["needs_review"])

    def test_one_unproven_list_item_drops_alone(self) -> None:
        service = self._service()
        turns = (
            ("operator", "left", "Добрый день, расскажу про программу"),
            ("client", "right", "Нужна математика для подготовки"),
            ("operator", "left", "Пришлю ссылку на оплату"),
        )
        dialogue = build_dialogue_input(fx.proven_call(turns))
        call = SimpleNamespace(
            source_call_id=fx.SOURCE_CALL_ID,
            started_at=None,
            manager_name="Менеджер",
        )
        analysis = {
            "structured_fields": _v3_fields(
                interests={
                    "products": [],
                    "format": [],
                    "exam_targets": [],
                    # "математика" is really said by the client; "химия" is not.
                    "subjects": ["математика", "химия"],
                }
            ),
            **self._selected("T0001", "T0002", "T0003"),
        }

        guarded = service._apply_claim_evidence(call, analysis, dialogue, [])

        self.assertEqual(
            guarded["structured_fields"]["interests"]["subjects"], ["математика"]
        )
        self.assertTrue(guarded["needs_review"])
        self.assertTrue(
            any(
                reason.startswith(
                    "claim_evidence_missing_or_invalid:"
                    "structured_fields.interests.subjects["
                )
                for reason in guarded["review_reasons"]
            )
        )

    def test_an_explicit_denial_cannot_confirm_a_payment(self) -> None:
        turns = (
            ("operator", "left", "Скажите, вы уже оплатили курс?"),
            ("client", "right", "Нет, я не оплатил, ещё думаю"),
        )
        service = self._service()
        dialogue = build_dialogue_input(fx.proven_call(turns))
        call = SimpleNamespace(
            source_call_id=fx.SOURCE_CALL_ID,
            started_at=None,
            manager_name="Менеджер",
        )
        analysis = {
            "structured_fields": _v3_fields(
                result={"status": "payment_confirmed", "detail": None}
            ),
            **self._selected("T0001", "T0002"),
        }

        guarded = service._apply_claim_evidence(
            call,
            analysis,
            dialogue,
            [
                {
                    "field_path": "structured_fields.result.status",
                    "item_id": None,
                    "support_type": "explicit",
                    "turn_ids": ["T0002"],
                }
            ],
        )

        self.assertIsNone(guarded["structured_fields"]["result"]["status"])
        self.assertIn(
            "claim_evidence_missing_or_invalid:structured_fields.result.status",
            guarded["review_reasons"],
        )

        # The manager's question is not proof either, even though the words
        # "вы уже оплатили" match the old payment detector exactly.
        analysis["structured_fields"]["result"]["status"] = "payment_confirmed"
        guarded = service._apply_claim_evidence(
            call,
            analysis,
            dialogue,
            [
                {
                    "field_path": "structured_fields.result.status",
                    "item_id": None,
                    "support_type": "explicit",
                    "turn_ids": ["T0001"],
                }
            ],
        )
        self.assertIsNone(guarded["structured_fields"]["result"]["status"])

    def test_a_hypothetical_payment_followed_by_no_money_is_not_payment(self) -> None:
        dialogue = build_dialogue_input(
            fx.proven_call(
                (
                    ("operator", "left", "Вы готовы оплатить?"),
                    ("client", "right", "Оплатил бы сразу, но пока денег нет"),
                )
            )
        )

        self.assertFalse(
            self._service()._turn_supports(
                "structured_fields.result.status",
                "payment_confirmed",
                dialogue.turns[1],
            )
        )

    def test_post_anchor_negation_and_positive_price_do_not_invert_meaning(self) -> None:
        service = self._service()
        self.assertFalse(
            service._turn_supports(
                "structured_fields.commercial.discount_interest",
                True,
                {"speaker_kind": "client", "text": "Скидки меня не интересуют"},
            )
        )
        self.assertFalse(
            service._turn_supports(
                "structured_fields.commercial.price_sensitivity",
                "high",
                {"speaker_kind": "client", "text": "Цена нормальная, всё устраивает"},
            )
        )

    def test_payment_requires_manager_confirmation_and_rejects_later_reversal(self) -> None:
        service = self._service()
        self.assertFalse(
            service._turn_supports(
                "structured_fields.result.status",
                "payment_confirmed",
                {"speaker_kind": "client", "text": "Я уже оплатил курс"},
            )
        )

        for text in (
            "Оплата получена. Хотя деньги ещё не поступили.",
            "Оплата получена. Точнее, платёж ещё не прошёл.",
            "Оплата получена. Ой, нет, ещё не прошла.",
        ):
            with self.subTest(text=text):
                self.assertFalse(
                    service._turn_supports(
                        "structured_fields.result.status",
                        "payment_confirmed",
                        {"speaker_kind": "manager", "text": text},
                    )
                )

    def test_adjacent_payment_denial_blocks_confirmation_but_neutral_reply_does_not(self) -> None:
        service = self._service()
        call = SimpleNamespace(
            source_call_id=fx.SOURCE_CALL_ID,
            started_at=None,
            manager_name="Менеджер",
        )

        def resolved(client_text):
            dialogue = build_dialogue_input(
                fx.proven_call(
                    (
                        ("operator", "left", "Вижу, оплата получена"),
                        ("client", "right", client_text),
                    )
                )
            )
            return service._resolve_claim_turns(
                field_path="structured_fields.result.status",
                value="payment_confirmed",
                item_id=None,
                request={
                    "field_path": "structured_fields.result.status",
                    "item_id": None,
                    "support_type": "explicit",
                    "turn_ids": ["T0001"],
                },
                turns={str(turn["turn_id"]): turn for turn in dialogue.turns},
                selected=["T0001", "T0002"],
                ordered=list(dialogue.turns),
            )

        self.assertEqual(resolved("Нет, я не платил"), ([], ""))
        refs, source = resolved("Спасибо, увидел")
        self.assertEqual([ref["turn_id"] for ref in refs], ["T0001"])
        self.assertEqual(source, "model_claim")

    def test_payment_denial_after_an_intermediate_reply_still_blocks_confirmation(self) -> None:
        service = self._service()
        dialogue = build_dialogue_input(
            fx.proven_call(
                (
                    ("operator", "left", "Вижу, оплата получена"),
                    ("client", "right", "Спасибо, сейчас проверю"),
                    ("client", "right", "Нет, я не оплатил"),
                )
            )
        )

        refs, source = service._resolve_claim_turns(
            field_path="structured_fields.result.status",
            value="payment_confirmed",
            item_id=None,
            request={
                "field_path": "structured_fields.result.status",
                "item_id": None,
                "support_type": "explicit",
                "turn_ids": ["T0001"],
            },
            turns={str(turn["turn_id"]): turn for turn in dialogue.turns},
            selected=["T0001", "T0002", "T0003"],
            ordered=list(dialogue.turns),
        )

        self.assertEqual((refs, source), ([], ""))

    def test_payment_reversal_blocks_an_earlier_confirmation(self) -> None:
        service = self._service()
        for reversal in (
            "Платёж отменён банком.",
            "Оплату вернули клиенту.",
        ):
            dialogue = build_dialogue_input(
                fx.proven_call(
                    (
                        ("operator", "left", "Вижу, оплата получена"),
                        ("operator", "left", reversal),
                        ("client", "right", "Понял"),
                    )
                )
            )
            with self.subTest(reversal=reversal):
                self.assertFalse(
                    service._claim_refs_support(
                        "structured_fields.result.status",
                        "payment_confirmed",
                        [dialogue.turns[0]],
                        list(dialogue.turns),
                    )
                )

        self.assertTrue(
            service._turn_supports(
                "structured_fields.result.status",
                "payment_confirmed",
                {"speaker_kind": "manager", "text": "Вижу, оплата получена."},
            )
        )

    def test_payment_denial_late_in_the_dialogue_still_blocks_confirmation(self) -> None:
        service = self._service()
        dialogue = build_dialogue_input(
            fx.proven_call(
                (
                    ("operator", "left", "Вижу, оплата получена"),
                    ("client", "right", "Спасибо, сейчас посмотрю"),
                    ("operator", "left", "Проверьте личный кабинет"),
                    ("client", "right", "Секунду"),
                    ("client", "right", "Нет, я не оплачивал"),
                )
            )
        )
        refs, source = service._resolve_claim_turns(
            field_path="structured_fields.result.status",
            value="payment_confirmed",
            item_id=None,
            request={
                "field_path": "structured_fields.result.status",
                "item_id": None,
                "support_type": "explicit",
                "turn_ids": ["T0001"],
            },
            turns={str(turn["turn_id"]): turn for turn in dialogue.turns},
            selected=[str(turn["turn_id"]) for turn in dialogue.turns],
            ordered=list(dialogue.turns),
        )

        self.assertEqual((refs, source), ([], ""))

    def test_later_client_cancellation_blocks_earlier_sale_agreement(self) -> None:
        service = self._service()
        dialogue = build_dialogue_input(
            fx.proven_call(
                (
                    ("client", "right", "Да, беру курс и готов оплатить"),
                    ("operator", "left", "Тогда оформляю документы"),
                    ("client", "right", "Нет, я передумала, не буду покупать"),
                )
            )
        )
        refs, source = service._resolve_claim_turns(
            field_path="structured_fields.result.status",
            value="sale_agreed",
            item_id=None,
            request={
                "field_path": "structured_fields.result.status",
                "item_id": None,
                "support_type": "explicit",
                "turn_ids": ["T0001"],
            },
            turns={str(turn["turn_id"]): turn for turn in dialogue.turns},
            selected=["T0001", "T0002", "T0003"],
            ordered=list(dialogue.turns),
        )

        self.assertEqual((refs, source), ([], ""))

    def test_manager_question_plus_short_client_yes_supports_interest_only_as_pair(self) -> None:
        service = self._service()

        for answer, accepted in (("Да.", True), ("Нет", False), ("Наверное", False)):
            with self.subTest(answer=answer):
                dialogue = build_dialogue_input(
                    fx.proven_call(
                        (
                            ("operator", "left", "Вас интересует математика?"),
                            ("client", "right", answer),
                        )
                    )
                )
                refs, source = service._resolve_claim_turns(
                    field_path="structured_fields.interests.subjects",
                    value="математика",
                    item_id="математика",
                    request={
                        "field_path": "structured_fields.interests.subjects",
                        "item_id": "математика",
                        "support_type": "explicit",
                        "turn_ids": ["T0001", "T0002"],
                    },
                    turns={str(turn["turn_id"]): turn for turn in dialogue.turns},
                    selected=["T0001", "T0002"],
                    ordered=list(dialogue.turns),
                )
                self.assertEqual(bool(refs), accepted)
                self.assertEqual(source, "model_claim" if accepted else "")

    def test_manager_follow_up_question_plus_client_yes_is_an_explicit_agreement(self) -> None:
        service = self._service()
        for answer, accepted in (("Да.", True), ("Нет.", False)):
            dialogue = build_dialogue_input(
                fx.proven_call(
                    (
                        ("operator", "left", "Перезвоним вам в пятницу?"),
                        ("client", "right", answer),
                    )
                )
            )
            for field_path, value in (
                ("structured_fields.result.status", "follow_up_agreed"),
                ("structured_fields.next_step.action", "Перезвонить клиенту"),
                ("structured_fields.next_step.due", "в пятницу"),
            ):
                with self.subTest(answer=answer, field_path=field_path):
                    self.assertIs(
                        service._claim_refs_support(
                            field_path, value, list(dialogue.turns), list(dialogue.turns)
                        ),
                        accepted,
                    )

    def test_missing_selected_turn_list_never_opens_the_whole_dialogue(self) -> None:
        service = self._service()
        dialogue = build_dialogue_input(
            fx.proven_call(
                (("operator", "left", "Вижу, оплата получена"),
                 ("client", "right", "Спасибо"))
            )
        )
        call = SimpleNamespace(
            source_call_id=fx.SOURCE_CALL_ID,
            started_at=None,
            manager_name="Менеджер",
        )
        guarded = service._apply_claim_evidence(
            call,
            {"structured_fields": _v3_fields(
                result={"status": "payment_confirmed", "detail": None}
            ), "quality_flags": {}},
            dialogue,
            [{
                "field_path": "structured_fields.result.status",
                "item_id": None,
                "support_type": "explicit",
                "turn_ids": ["T0001"],
            }],
        )

        self.assertIsNone(guarded["structured_fields"]["result"]["status"])
        self.assertFalse(
            service._turn_supports(
                "structured_fields.result.status",
                "payment_confirmed",
                {
                    "speaker_kind": "manager",
                    "text": "Вижу, оплата прошла. Ой, нет, платёж не прошёл.",
                },
            )
        )

    def test_deterministic_evidence_cannot_use_a_turn_cut_from_the_prompt(self) -> None:
        service = self._service()
        dialogue = build_dialogue_input(
            fx.proven_call(
                (
                    ("client", "right", "Здравствуйте"),
                    ("operator", "left", "Вижу, получена ваша оплата"),
                )
            )
        )
        refs, source = service._resolve_claim_turns(
            field_path="structured_fields.result.status",
            value="payment_confirmed",
            item_id=None,
            request=None,
            turns={str(turn["turn_id"]): turn for turn in dialogue.turns},
            selected=["T0001"],
            ordered=list(dialogue.turns),
        )

        self.assertEqual(refs, [])
        self.assertEqual(source, "")

    def test_every_pre_llm_non_conversation_path_reports_zero_model_work(self) -> None:
        payload = self._service()._non_conversation_analysis()

        self.assertTrue(payload["quality_flags"]["pre_llm_non_conversation_gate"])

    def test_manager_question_and_manager_name_cannot_prove_client_facts(self) -> None:
        service = self._service()
        self.assertFalse(
            service._turn_supports(
                "structured_fields.result.status",
                "sale_agreed",
                {
                    "speaker_kind": "manager",
                    "text": "Оформляем запись прямо сейчас?",
                },
            )
        )
        self.assertFalse(
            service._turn_supports(
                "structured_fields.people.parent_fio",
                "Ольга",
                {
                    "speaker_kind": "manager",
                    "text": "Здравствуйте, меня зовут Ольга, учебный центр.",
                },
            )
        )

    def test_result_anchors_do_not_invert_clear_customer_meaning(self) -> None:
        service = self._service()
        cases = (
            ("sale_agreed", "Оформление документов занимает два дня", False),
            ("no_decision", "Я уже решил, беру курс", False),
            ("refusal", "Я не буду отказываться, мне интересно", False),
            ("sale_agreed", "Я беру курс и готов оплатить", True),
            ("no_decision", "Я ещё не решил, мне нужно подумать", True),
        )
        for status, text, expected in cases:
            with self.subTest(status=status, text=text):
                self.assertIs(
                    service._turn_supports(
                        "structured_fields.result.status",
                        status,
                        {"speaker_kind": "client", "text": text},
                    ),
                    expected,
                )

    def test_historical_events_do_not_become_current_results_or_next_steps(self) -> None:
        service = self._service()
        cases = (
            (
                "structured_fields.result.status",
                "appointment_agreed",
                "client",
                "Я уже записывалась к вам в прошлом году.",
                False,
            ),
            (
                "structured_fields.result.status",
                "payment_confirmed",
                "manager",
                "В прошлом году получили оплату за прошлый курс.",
                False,
            ),
            (
                "structured_fields.result.status",
                "appointment_agreed",
                "client",
                "Два года назад записывалась к вам на пробное занятие.",
                False,
            ),
            (
                "structured_fields.result.status",
                "payment_confirmed",
                "manager",
                "Месяц назад получили оплату за тот курс.",
                False,
            ),
            (
                "structured_fields.result.status",
                "sale_agreed",
                "client",
                "В 2024 году покупала у вас курс.",
                False,
            ),
            (
                "structured_fields.result.status",
                "appointment_agreed",
                "client",
                "Несколько недель назад записывалась на консультацию.",
                False,
            ),
            (
                "structured_fields.result.status",
                "follow_up_agreed",
                "client",
                "Лет пять назад мы уже созванивались.",
                False,
            ),
            (
                "structured_fields.result.status",
                "follow_up_agreed",
                "client",
                "Прошлой осенью договорились, что я перезвоню.",
                False,
            ),
            (
                "structured_fields.result.status",
                "follow_up_agreed",
                "client",
                "Прошлым летом договорились, что я перезвоню.",
                False,
            ),
            (
                "structured_fields.result.status",
                "appointment_agreed",
                "client",
                "На предыдущей смене договорились, я приеду.",
                False,
            ),
            (
                "structured_fields.result.status",
                "follow_up_agreed",
                "client",
                "Ещё в 2024-м договорились: я перезвоню.",
                False,
            ),
            (
                "structured_fields.result.status",
                "appointment_agreed",
                "client",
                "Пару смен назад договорились, я приеду.",
                False,
            ),
            (
                "structured_fields.result.status",
                "follow_up_agreed",
                "client",
                "Три сезона назад договорились, я перезвоню.",
                False,
            ),
            (
                "structured_fields.result.status",
                "appointment_agreed",
                "client",
                "В предыдущем сезоне договорились, я приеду.",
                False,
            ),
            (
                "structured_fields.result.status",
                "appointment_agreed",
                "client",
                "Если получится, я приеду.",
                False,
            ),
            (
                "structured_fields.result.status",
                "follow_up_agreed",
                "client",
                "Если будет удобно, я перезвоню.",
                False,
            ),
            (
                "structured_fields.result.status",
                "no_decision",
                "client",
                "Мы всё обсудили, решение принято.",
                False,
            ),
            (
                "structured_fields.result.status",
                "sale_agreed",
                "client",
                "На прошлой неделе покупала другой курс.",
                False,
            ),
            (
                "structured_fields.next_step.action",
                "Перезвонить клиенту",
                "manager",
                "Вчера планировали перезвонить, но вопрос уже закрыт.",
                False,
            ),
            (
                "structured_fields.result.status",
                "sale_agreed",
                "client",
                "Раньше сомневалась, но сейчас беру этот курс.",
                True,
            ),
            (
                "structured_fields.result.status",
                "payment_confirmed",
                "manager",
                "Вчера получили вашу оплату за текущий курс.",
                True,
            ),
            (
                "structured_fields.result.status",
                "appointment_agreed",
                "client",
                "В прошлом году сомневалась, а сегодня записываюсь на курс.",
                True,
            ),
            (
                "structured_fields.next_step.action",
                "Перезвонить клиенту",
                "manager",
                "Договорились: я перезвоню завтра.",
                True,
            ),
            (
                "structured_fields.result.status",
                "follow_up_agreed",
                "client",
                "Договорились, я перезвоню завтра.",
                True,
            ),
        )
        for path, value, speaker, text, expected in cases:
            with self.subTest(path=path, text=text):
                self.assertIs(
                    service._turn_supports(
                        path, value, {"speaker_kind": speaker, "text": text}
                    ),
                    expected,
                )

    def test_later_completion_or_refusal_cancels_an_earlier_next_step(self) -> None:
        service = self._service()
        support = {
            "turn_id": "T0001",
            "speaker_kind": "manager",
            "text": "Договорились: я перезвоню завтра.",
        }
        cancelled = (
            {"turn_id": "T0002", "speaker_kind": "client", "text": "Не звоните, мы отказались."},
            {"turn_id": "T0002", "speaker_kind": "manager", "text": "Мы уже перезвонили и решили вопрос."},
            {"turn_id": "T0002", "speaker_kind": "manager", "text": "Созвон уже состоялся."},
            {"turn_id": "T0002", "speaker_kind": "client", "text": "Решили больше не созваниваться."},
            {"turn_id": "T0002", "speaker_kind": "manager", "text": "Перезвон больше не требуется."},
            {"turn_id": "T0002", "speaker_kind": "manager", "text": "Контакт отменён."},
        )
        for later in cancelled:
            with self.subTest(text=later["text"]):
                self.assertFalse(
                    service._claim_refs_support(
                        "structured_fields.next_step.action",
                        "Перезвонить клиенту",
                        [support],
                        [support, later],
                    )
                )
        self.assertTrue(
            service._claim_refs_support(
                "structured_fields.next_step.action",
                "Перезвонить клиенту",
                [support],
                [support, {"turn_id": "T0002", "speaker_kind": "client", "text": "Хорошо, буду ждать."}],
            )
        )
        self.assertTrue(
            service._claim_refs_support(
                "structured_fields.next_step.action",
                "Перезвонить клиенту",
                [support],
                [
                    support,
                    {
                        "turn_id": "T0002",
                        "speaker_kind": "manager",
                        "text": "Материалы уже отправили.",
                    },
                ],
            )
        )

    def test_completed_action_is_detected_in_normal_russian_word_order(self) -> None:
        service = self._service()
        cases = (
            ("Перезвонить клиенту", "Договорились, перезвоню завтра.", "Созвон завершён."),
            ("Отправить материалы", "Договорились, отправлю материалы завтра.", "Материалы уже отправили."),
            (
                "Отправить ссылку на оплату",
                "Договорились, отправлю ссылку на оплату завтра.",
                "Ссылку на оплату уже отправили.",
            ),
            (
                "Дождаться решения клиента",
                "Договорились, дождёмся решения клиента до пятницы.",
                "Клиент уже решил.",
            ),
            (
                "Согласовать следующий контакт",
                "Договорились, согласуем следующий контакт завтра.",
                "Контакт уже согласован.",
            ),
        )
        for action, first_text, later_text in cases:
            support = {"turn_id": "T0001", "speaker_kind": "manager", "text": first_text}
            later = {"turn_id": "T0002", "speaker_kind": "manager", "text": later_text}
            with self.subTest(action=action):
                self.assertFalse(
                    service._claim_refs_support(
                        "structured_fields.next_step.action",
                        action,
                        [support],
                        [support, later],
                    )
                )

    def test_action_cancellation_does_not_cancel_a_different_action(self) -> None:
        service = self._service()
        cases = (
            (
                "Перезвонить клиенту",
                "Договорились, перезвоню завтра.",
                "Материалы присылать не надо, но позвоните.",
                True,
            ),
            (
                "Отправить материалы",
                "Договорились, отправлю материалы завтра.",
                "Звонить не надо, но материалы пришлите.",
                True,
            ),
            (
                "Отправить материалы",
                "Договорились, отправлю материалы завтра.",
                "Материалы мне не нужны.",
                False,
            ),
            (
                "Отправить материалы",
                "Договорились, отправлю материалы завтра.",
                "Презентацию не присылайте.",
                False,
            ),
            (
                "Отправить ссылку на оплату",
                "Договорились, отправлю ссылку на оплату завтра.",
                "Ссылку на оплату не присылайте.",
                False,
            ),
            (
                "Отправить материалы",
                "Договорились, отправлю материалы завтра.",
                "Ссылку на оплату не присылайте, а материалы нужны.",
                True,
            ),
            (
                "Отправить ссылку на оплату",
                "Договорились, отправлю ссылку на оплату завтра.",
                "Презентацию не присылайте, а ссылку на оплату жду.",
                True,
            ),
            (
                "Перезвонить клиенту",
                "Договорились, перезвоню завтра.",
                "Перезванивать больше не требуется.",
                False,
            ),
            (
                "Перезвонить клиенту",
                "Договорились, перезвоню завтра.",
                "Мы отменяем завтрашний звонок.",
                False,
            ),
            (
                "Перезвонить клиенту",
                "Договорились, перезвоню завтра.",
                "Не смогу завтра перезвонить.",
                False,
            ),
        )
        for action, first_text, later_text, expected in cases:
            support = {"turn_id": "T0001", "speaker_kind": "manager", "text": first_text}
            later = {"turn_id": "T0002", "speaker_kind": "client", "text": later_text}
            with self.subTest(action=action, later=later_text):
                self.assertIs(
                    service._claim_refs_support(
                        "structured_fields.next_step.action",
                        action,
                        [support],
                        [support, later],
                    ),
                    expected,
                )

    def test_later_committed_due_replaces_the_old_due(self) -> None:
        service = self._service()
        support = {
            "turn_id": "T0001",
            "speaker_kind": "manager",
            "text": "Договорились, перезвоню завтра.",
        }
        for later_text in (
            "Перенесём звонок на пятницу.",
            "Давайте не завтра, а в пятницу.",
        ):
            later = {
                "turn_id": "T0002",
                "speaker_kind": "manager",
                "text": later_text,
            }
            with self.subTest(later=later_text):
                self.assertFalse(
                    service._claim_refs_support(
                        "structured_fields.next_step.due",
                        "завтра",
                        [support],
                        [support, later],
                    )
                )

    def test_completed_or_cancelled_action_clears_its_due_atomically(self) -> None:
        cases = (
            (
                "Отправить материалы",
                "Договорились, отправлю материалы завтра.",
                "Материалы уже отправили.",
            ),
            (
                "Отправить материалы",
                "Договорились, отправлю материалы завтра.",
                "Презентацию не присылайте.",
            ),
            (
                "Отправить ссылку на оплату",
                "Договорились, отправлю ссылку на оплату завтра.",
                "Ссылку на оплату не присылайте.",
            ),
            (
                "Отправить материалы",
                "Договорились, отправлю материалы завтра.",
                "Программу уже отправили.",
            ),
            (
                "Отправить материалы",
                "Договорились, отправлю материалы завтра.",
                "Документы уже отправили.",
            ),
            (
                "Перезвонить клиенту",
                "Договорились, перезвоню завтра.",
                "С клиентом уже связались.",
            ),
            (
                "Перезвонить клиенту",
                "Договорились, перезвоню завтра.",
                "Перезвон отменяется.",
            ),
            (
                "Перезвонить клиенту",
                "Договорились, перезвоню завтра.",
                "Звонок уже был.",
            ),
        )
        for action, support_text, later_text in cases:
            turns = (
                ("operator", "left", support_text),
                ("client", "right", later_text),
                ("operator", "left", "Понял."),
            )
            dialogue = build_dialogue_input(fx.proven_call(turns))
            call = SimpleNamespace(
                source_call_id=fx.SOURCE_CALL_ID,
                source_recording_id=fx.RECORDING_ID,
                started_at=None,
                manager_name="Менеджер",
                phone=None,
            )
            analysis = {
                "structured_fields": _v3_fields(
                    next_step={"action": action, "due": "завтра"}
                ),
                **self._selected("T0001", "T0002", "T0003"),
            }
            requests = [
                {
                    "field_path": field_path,
                    "item_id": None,
                    "support_type": "explicit",
                    "turn_ids": ["T0001"],
                }
                for field_path in (
                    "structured_fields.next_step.action",
                    "structured_fields.next_step.due",
                )
            ]

            with self.subTest(action=action, later=later_text):
                guarded = self._service()._apply_claim_evidence(
                    call, analysis, dialogue, requests
                )
                self.assertIsNone(guarded["structured_fields"]["next_step"]["action"])
                self.assertIsNone(guarded["structured_fields"]["next_step"]["due"])
                self.assertIsNone(guarded["timeline"])
                self.assertFalse(
                    any(
                        item["field_path"] == "structured_fields.next_step.due"
                        for item in guarded["claim_evidence"]
                    )
                )
    def test_latest_explicit_customer_decision_wins(self) -> None:
        service = self._service()
        cases = (
            ("sale_agreed", "Я беру курс.", "Я отказываюсь, курс не нужен."),
            ("refusal", "Мне курс не нужен.", "Передумала, сейчас беру курс."),
            ("no_decision", "Мне нужно подумать.", "Я решила, беру курс."),
            ("appointment_agreed", "Запишите меня на консультацию.", "Я отказываюсь."),
            ("follow_up_agreed", "Я перезвоню завтра.", "Я решила, беру курс."),
            ("appointment_agreed", "Я приеду завтра.", "Не приеду, отменяю запись."),
            ("follow_up_agreed", "Я перезвоню завтра.", "Звонить не буду."),
            ("sale_agreed", "Я беру курс.", "Стоп, я ещё не решила."),
            ("refusal", "Курс мне не нужен.", "Я всё-таки подумаю."),
            ("sale_agreed", "Я беру курс.", "Нет, всё-таки я подумаю."),
            ("sale_agreed", "Я беру курс.", "Нет, всё-таки курс не нужен."),
            ("no_decision", "Я ещё подумаю.", "Покупать не будем."),
            ("no_decision", "Я ещё подумаю.", "Курс нам не подходит."),
            ("no_decision", "Я ещё подумаю.", "Решили не записываться."),
            ("sale_agreed", "Я беру курс.", "Всё-таки брать не будем."),
        )
        for status, first_text, later_text in cases:
            support = {"turn_id": "T0001", "speaker_kind": "client", "text": first_text}
            later = {"turn_id": "T0002", "speaker_kind": "client", "text": later_text}
            with self.subTest(status=status):
                self.assertFalse(
                    service._claim_refs_support(
                        "structured_fields.result.status",
                        status,
                        [support],
                        [support, later],
                    )
                )

    def test_introductory_no_does_not_negate_the_decision_that_follows(self) -> None:
        service = self._service()
        cases = (
            ("no_decision", "Нет, всё-таки я подумаю."),
            ("refusal", "Нет, всё-таки курс не нужен."),
            ("sale_agreed", "Нет, я беру курс."),
        )
        for status, text in cases:
            with self.subTest(status=status, text=text):
                self.assertTrue(
                    service._turn_supports(
                        "structured_fields.result.status",
                        status,
                        {"speaker_kind": "client", "text": text},
                    )
                )

        same_turn = (
            ("sale_agreed", "Я беру курс. Стоп, я ещё не решила."),
            ("refusal", "Курс мне не нужен. Я всё-таки подумаю."),
            ("no_decision", "Я подумала и решила: беру курс."),
            ("appointment_agreed", "Я приеду, но отменяю запись."),
            ("follow_up_agreed", "Я перезвоню, хотя звонить не буду."),
            ("no_decision", "Я подумаю, но покупать не будем."),
            ("no_decision", "Я подумаю, курс нам не подходит."),
            ("no_decision", "Я подумаю, но решили не записываться."),
        )
        for status, text in same_turn:
            with self.subTest(status=status, same_turn=text):
                self.assertFalse(
                    service._turn_supports(
                        "structured_fields.result.status",
                        status,
                        {"speaker_kind": "client", "text": text},
                    )
                )

    def test_refusal_marker_is_not_negated_by_its_own_wording(self) -> None:
        service = self._service()
        for text in ("Курс мне не нужен.", "Мне курс не нужен, я отказываюсь."):
            with self.subTest(text=text):
                self.assertTrue(
                    service._turn_supports(
                        "structured_fields.result.status",
                        "refusal",
                        {"speaker_kind": "client", "text": text},
                    )
                )

        for text in (
            "Мне не нужна рассрочка, оплачу сразу.",
            "Скидка не нужна, цена устраивает.",
        ):
            with self.subTest(non_refusal=text):
                self.assertFalse(
                    service._turn_supports(
                        "structured_fields.result.status",
                        "refusal",
                        {"speaker_kind": "client", "text": text},
                    )
                )

    def test_conditional_next_step_is_not_a_commitment(self) -> None:
        service = self._service()
        cases = (
            "Если захотите, отправлю материалы.",
            "Возможно, отправлю материалы.",
            "При возможности отправлю материалы.",
            "Мы могли бы отправить материалы.",
            "Может, отправлю материалы.",
            "Наверное, отправлю материалы.",
            "Хотел бы отправить материалы завтра.",
            "Можно было бы отправить материалы завтра.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(
                    service._turn_supports(
                        "structured_fields.next_step.action",
                        "Отправить материалы",
                        {"speaker_kind": "manager", "text": text},
                    )
                )
        self.assertTrue(
            service._turn_supports(
                "structured_fields.next_step.action",
                "Отправить материалы",
                {"speaker_kind": "manager", "text": "Точно отправлю материалы."},
            )
        )

    def test_modal_customer_phrases_are_not_decisions(self) -> None:
        service = self._service()
        cases = (
            ("follow_up_agreed", "Возможно, я перезвоню."),
            ("follow_up_agreed", "Мы могли бы созвониться."),
            ("appointment_agreed", "При возможности я приеду."),
            ("appointment_agreed", "Рассматриваю возможность записаться."),
            ("follow_up_agreed", "Рассматриваем возможность перезвонить."),
            ("appointment_agreed", "Я бы хотела записаться."),
            ("follow_up_agreed", "Может, я перезвоню."),
            ("follow_up_agreed", "Наверное, я перезвоню."),
            ("follow_up_agreed", "Хотел бы перезвонить завтра."),
            ("follow_up_agreed", "Можно было бы перезвонить завтра."),
        )
        for status, text in cases:
            with self.subTest(status=status, text=text):
                self.assertFalse(
                    service._turn_supports(
                        "structured_fields.result.status",
                        status,
                        {"speaker_kind": "client", "text": text},
                    )
                )
        self.assertTrue(
            service._turn_supports(
                "structured_fields.result.status",
                "follow_up_agreed",
                {"speaker_kind": "client", "text": "Точно, я перезвоню."},
            )
        )

    def test_historical_result_detail_is_not_published_as_current(self) -> None:
        service = self._service()
        cases = (
            "В прошлом году получили вашу оплату.",
            "В предыдущем году получили вашу оплату.",
            "Позапрошлой зимой получили вашу оплату.",
            "Минувшей зимой получили вашу оплату.",
        )
        for text in cases:
            with self.subTest(text=text):
                self.assertFalse(
                    service._turn_supports(
                        "structured_fields.result.status",
                        "payment_confirmed",
                        {"speaker_kind": "manager", "text": text},
                    )
                )
                self.assertFalse(
                    service._turn_supports(
                        "structured_fields.result.detail",
                        text,
                        {"speaker_kind": "manager", "text": text},
                    )
                )

    def test_high_risk_claims_require_their_own_business_context(self) -> None:
        service = self._service()
        rejected = (
            (
                "structured_fields.result.status",
                "sale_agreed",
                "client",
                "Готова купить, если будет скидка.",
            ),
            (
                "structured_fields.objections",
                "цена",
                "client",
                "Цена нас полностью устраивает.",
            ),
            (
                "structured_fields.next_step.action",
                "Отправить материалы",
                "client",
                "Вы уже отправили материалы.",
            ),
            (
                "structured_fields.commercial.budget",
                "50 000",
                "client",
                "Цена курса 50 000 рублей.",
            ),
            (
                "structured_fields.result.detail",
                "50 000",
                "client",
                "Цена курса 50 000 рублей.",
            ),
            (
                "structured_fields.next_step.due",
                "завтра",
                "client",
                "Завтра у ребёнка контрольная.",
            ),
        )
        for path, value, speaker, text in rejected:
            with self.subTest(path=path, text=text):
                self.assertFalse(
                    service._turn_supports(
                        path, value, {"speaker_kind": speaker, "text": text}
                    )
                )

        accepted = (
            (
                "structured_fields.commercial.budget",
                "50 000",
                "client",
                "Наш бюджет — 50 000 рублей.",
            ),
            (
                "structured_fields.next_step.due",
                "завтра",
                "manager",
                "Я перезвоню завтра.",
            ),
            (
                "structured_fields.objections",
                "цена",
                "client",
                "Цена слишком высокая.",
            ),
        )
        for path, value, speaker, text in accepted:
            with self.subTest(path=path, text=text):
                self.assertTrue(
                    service._turn_supports(
                        path, value, {"speaker_kind": speaker, "text": text}
                    )
                )

    def test_prompt_truncation_always_requires_human_review(self) -> None:
        service = self._service()
        call = _trusted_dialogue_call()
        normalized = service._normalize_analysis(
            call,
            "Менеджер: расскажу про курс. Клиент: нужна математика.",
            {"quality_flags": {"analyze_prompt_truncated": True}},
        )

        self.assertTrue(normalized["needs_review"])
        self.assertIn("analyze_prompt_truncated", normalized["review_reasons"])

    def test_manager_may_confirm_received_payment_but_not_ask_about_it(self) -> None:
        service = self._service()
        self.assertTrue(
            service._turn_supports(
                "structured_fields.result.status",
                "payment_confirmed",
                {"speaker_kind": "manager", "text": "Вижу, получена ваша оплата."},
            )
        )
        self.assertFalse(
            service._turn_supports(
                "structured_fields.result.status",
                "payment_confirmed",
                {"speaker_kind": "manager", "text": "Вы уже оплатили?"},
            )
        )

    def test_manager_display_is_normalized_while_the_proven_raw_value_stays(self) -> None:
        turns = (
            ("operator", "left", "Рассказываю: мы представляем МПК МФТИ"),
            ("client", "right", "Здравствуйте"),
        )
        service = self._service()
        dialogue = build_dialogue_input(fx.proven_call(turns))
        call = SimpleNamespace(
            source_call_id=fx.SOURCE_CALL_ID,
            source_file="call.mp3",
            started_at=None,
            manager_name="Менеджер",
            phone=None,
        )
        analysis = {
            "structured_fields": _v3_fields(
                result={"status": "information_only", "detail": "МПК МФТИ"}
            ),
            **self._selected("T0001", "T0002"),
        }

        guarded = service._apply_claim_evidence(
            call,
            analysis,
            dialogue,
            [
                {
                    "field_path": "structured_fields.result.detail",
                    "item_id": None,
                    "support_type": "explicit",
                    "turn_ids": ["T0001"],
                }
            ],
        )

        self.assertEqual(guarded["structured_fields"]["result"]["detail"], "МПК МФТИ")
        self.assertEqual(guarded["display_fields"]["result"]["detail"], "УНПК МФТИ")
        self.assertEqual(guarded["crm_blocks"], guarded["display_fields"])
        self.assertIn("УНПК МФТИ", guarded["history_summary"])
        self.assertNotIn("МПК МФТИ", guarded["history_summary"])

    def test_claim_id_is_deterministic_and_moves_with_the_value(self) -> None:
        service = self._service()
        call, dialogue = self._call_and_dialogue()

        def run(action, *, record=call, current_dialogue=dialogue):
            analysis = {
                "structured_fields": _v3_fields(
                    next_step={"action": action, "due": None}
                ),
                **self._selected("T0001", "T0002", "T0003"),
            }
            guarded = service._apply_claim_evidence(
                record,
                analysis,
                current_dialogue,
                [
                    {
                        "field_path": "structured_fields.next_step.action",
                        "item_id": None,
                        "support_type": "explicit",
                        "turn_ids": ["T0003"],
                    }
                ],
            )
            return [
                item["claim_id"]
                for item in guarded["claim_evidence"]
                if item["field_path"] == "structured_fields.next_step.action"
            ]

        first = run("Отправить ссылку на оплату")
        again = run("Отправить ссылку на оплату")
        # Also provable against T0003 ("пришлю"), so this compares two real ids.
        changed = run("Отправить материалы")
        other_variants = json.loads(json.dumps(DIALOGUE_VARIANTS, ensure_ascii=False))
        other_variants[fx.PROVIDER_EVIDENCE_FIELD] = fx.evidence(
            DIALOGUE_TURNS, source_call_id="call-8"
        )
        other_call = _dialogue_call(
            source_call_id="call-8",
            source_recording_id=fx.RECORDING_ID,
            transcript_variants_json=json.dumps(other_variants, ensure_ascii=False),
        )
        other_dialogue = build_dialogue_input(call_record_view(other_call))
        other_call_id = run(
            "Отправить ссылку на оплату",
            record=other_call,
            current_dialogue=other_dialogue,
        )

        self.assertTrue(first)
        self.assertTrue(changed)
        self.assertEqual(first, again)
        self.assertNotEqual(first, changed)
        self.assertNotEqual(first, other_call_id)

    def test_the_summary_is_rebuilt_from_the_values_that_survived(self) -> None:
        service = self._service()
        turns = (
            ("operator", "left", "Добрый день, расскажу про программу"),
            ("client", "right", "Нужна математика для подготовки"),
            ("operator", "left", "Пришлю ссылку на оплату"),
        )
        dialogue = build_dialogue_input(fx.proven_call(turns))
        call = SimpleNamespace(
            source_call_id=fx.SOURCE_CALL_ID,
            source_file="call.mp3",
            started_at=None,
            manager_name="Менеджер",
        )
        analysis = {
            "structured_fields": _v3_fields(
                next_step={"action": "Выставить счёт на 50 000", "due": None},
                interests={
                    "products": [],
                    "format": [],
                    "exam_targets": [],
                    "subjects": ["математика"],
                },
            ),
            "follow_up_reason": "Оценка на основе содержания звонка.",
            "history_summary": "Клиент согласился оплатить 50 000 рублей.",
            "history_short": "Клиент согласился оплатить.",
            "summary": "Клиент согласился оплатить.",
            "crm_blocks": {"next_step": {"action": "Выставить счёт на 50 000"}},
            "next_step": "Выставить счёт на 50 000",
            **self._selected("T0001", "T0002", "T0003"),
        }

        guarded = service._apply_claim_evidence(call, analysis, dialogue, [])

        payload = json.dumps(guarded, ensure_ascii=False)
        # The unproven number is gone from every copy of the same fact.
        self.assertNotIn("50 000", payload)
        self.assertIsNone(guarded["next_step"])
        self.assertEqual(guarded["crm_blocks"], guarded["structured_fields"])
        self.assertEqual(guarded["history_summary"], guarded["history_short"])
        self.assertEqual(
            guarded["follow_up_reason"],
            "Выводы основаны только на подтверждённых репликах.",
        )
        self.assertIn("математика", guarded["history_summary"])
        self.assertEqual(
            guarded["history_summary_meta"]["contract_version"],
            HISTORY_SUMMARY_CONTRACT_VERSION,
        )
        self.assertTrue(
            any(
                part["template_id"] == "topics_v1" and part["claim_ids"]
                for part in guarded["history_summary_meta"]["parts"]
            )
        )


class V3ResponseContractTest(unittest.TestCase):
    """Этап C: anything that is not the v3 answer never becomes a payload."""

    def test_a_free_summary_or_an_authored_quote_rejects_the_whole_answer(self) -> None:
        cases = (
            ("legacy root keys", {"history_summary": "…", "tags": []}),
            (
                "model wrote its own summary",
                {**_v3_answer(), "history_summary": "Клиент согласился."},
            ),
            (
                "model authored the quote",
                {
                    "structured_fields": _v3_fields(),
                    "claim_requests": [
                        {
                            "field_path": "structured_fields.next_step.action",
                            "item_id": None,
                            "support_type": "explicit",
                            "turn_ids": ["T0001"],
                            "quote_text": "я пришлю ссылку",
                        }
                    ],
                },
            ),
            (
                "model invented a claim id",
                {
                    "structured_fields": _v3_fields(),
                    "claim_requests": [
                        {
                            "field_path": "structured_fields.next_step.action",
                            "item_id": None,
                            "support_type": "explicit",
                            "claim_id": "deadbeef",
                            "turn_ids": ["T0001"],
                        }
                    ],
                },
            ),
            (
                "path outside the closed list",
                {
                    "structured_fields": _v3_fields(),
                    "claim_requests": [
                        {
                            "field_path": "structured_fields.lead_priority",
                            "item_id": None,
                            "support_type": "explicit",
                            "turn_ids": ["T0001"],
                        }
                    ],
                },
            ),
            (
                "more than three references",
                {
                    "structured_fields": _v3_fields(),
                    "claim_requests": [
                        {
                            "field_path": "structured_fields.next_step.action",
                            "item_id": None,
                            "support_type": "explicit",
                            "turn_ids": ["T0001", "T0002", "T0003", "T0004"],
                        }
                    ],
                },
            ),
            (
                "malformed turn id",
                {
                    "structured_fields": _v3_fields(),
                    "claim_requests": [
                        {
                            "field_path": "structured_fields.next_step.action",
                            "item_id": None,
                            "support_type": "explicit",
                            "turn_ids": ["turn-3"],
                        }
                    ],
                },
            ),
            (
                "missing nested block",
                {
                    "structured_fields": {
                        key: value
                        for key, value in _v3_fields().items()
                        if key != "commercial"
                    },
                    "claim_requests": [],
                },
            ),
            (
                "extra nested key",
                _v3_answer(
                    contacts={
                        "email": None,
                        "preferred_channel": None,
                        "phone_from_filename": "+70000000000",
                    }
                ),
            ),
            (
                "wrong list type",
                _v3_answer(objections="цена"),
            ),
            (
                "invalid closed enum",
                _v3_answer(
                    result={"status": "probably_paid", "detail": None}
                ),
            ),
        )
        for label, payload in cases:
            with self.subTest(label):
                with self.assertRaises(AnalysisContractError):
                    validate_v3_model_response(payload)

    def test_the_contract_answer_is_accepted_unchanged(self) -> None:
        answer = _v3_answer(
            [
                {
                    "field_path": "structured_fields.objections",
                    "item_id": "цена",
                    "support_type": "explicit",
                    "turn_ids": ["T0002"],
                }
            ]
        )

        validated = validate_v3_model_response(answer)

        self.assertEqual(validated["structured_fields"], answer["structured_fields"])
        self.assertEqual(validated["claim_requests"], answer["claim_requests"])


class MoscowTimeContractTest(unittest.TestCase):
    """Этап C: one conversion, naive is UTC, and it holds at the boundaries."""

    def test_naive_value_is_utc_and_crosses_the_day_and_the_year(self) -> None:
        # 22:30 UTC on 31 December is already 01:30 on 1 January in Moscow.
        self.assertEqual(
            AnalyzeService._format_started_at(datetime(2025, 12, 31, 22, 30)),
            "01.01.2026 01:30",
        )
        self.assertEqual(
            AnalyzeService._format_started_at(datetime(2026, 3, 17, 23, 10)),
            "18.03.2026 02:10",
        )

    def test_aware_values_convert_once_and_only_once(self) -> None:
        utc = datetime(2026, 3, 17, 23, 10, tzinfo=timezone.utc)

        first = moscow_datetime(utc)
        second = moscow_datetime(first)

        self.assertEqual(first, second)
        self.assertEqual(first.strftime("%d.%m.%Y %H:%M"), "18.03.2026 02:10")
        self.assertEqual(AnalyzeService._format_started_at(utc), "18.03.2026 02:10")


class AnalysisMetaAndUsageTest(unittest.TestCase):
    """Этап D: versions, cache telemetry and never an estimated token count."""

    def test_meta_records_every_contract_version_and_the_input_sha(self) -> None:
        service = AnalyzeService(make_settings())
        call = _trusted_dialogue_call()
        dialogue = build_dialogue_input(call_record_view(call))
        context = service._analysis_prompt_context(
            call, dialogue.render(), "compact", dialogue
        )
        analysis = service._with_analysis_prompt_quality_flags(
            {"analysis_schema_version": "v3"},
            metrics=context["metrics"],
            prompt_version="v8",
            cache_hit=False,
        )

        analysis = service._bind_analysis_input_metadata(
            analysis, dialogue, "b" * 64
        )
        meta = service._build_analysis_meta(analysis)

        self.assertEqual(meta["analysis_input_sha256"], "b" * 64)
        self.assertEqual(
            meta["analysis_prompt_sha256"], context["metrics"]["analysis_prompt_sha256"]
        )
        self.assertEqual(meta["analysis_schema_version"], "v3")
        self.assertEqual(meta["dialogue_contract_version"], "canonical_dialogue_v1")
        self.assertEqual(meta["dialogue_canonical_sha256"], dialogue.canonical_sha256)
        self.assertEqual(meta["role_guard_version"], "role_guard_v1")
        self.assertEqual(meta["prompt_contract_version"], "analyse_v3_claim_evidence_1")
        self.assertEqual(meta["normalizer_engine_version"], "tenant_text_engine_v1")
        self.assertEqual(meta["normalizer_ruleset_version"], "tenant_ru_v1")
        self.assertEqual(meta["normalizer_tenant_id"], "mango")
        self.assertEqual(meta["timezone_contract_version"], "timezone_msk_v1")

    def test_prompt_identity_is_part_of_the_analysis_input_sha(self) -> None:
        call = _trusted_dialogue_call()

        first = analysis_input_identity_sha256(
            call,
            {"provider": "codex_cli", "model": "gpt-a", "prompt_version": "v8"},
        )
        second = analysis_input_identity_sha256(
            call,
            {"provider": "codex_cli", "model": "gpt-a", "prompt_version": "v9"},
        )

        self.assertNotEqual(first, second)

    def test_ollama_reports_exact_native_token_counts(self) -> None:
        client = OllamaClient("http://127.0.0.1:11434")
        usage: dict[str, Any] = {}
        with patch.object(
            client,
            "_post",
            return_value={
                "response": "{}",
                "prompt_eval_count": 17,
                "eval_count": 5,
            },
        ):
            payload = client.generate_json(
                model="test",
                system_prompt="system",
                user_prompt="user",
                think=None,
                temperature=0,
                usage_out=usage,
            )

        self.assertEqual(payload, {})
        self.assertEqual(usage, {"prompt_eval_count": 17, "eval_count": 5})
        self.assertEqual(
            provider_token_usage(usage),
            {
                "source": "provider_partial",
                "prompt_tokens": 17,
                "completion_tokens": 5,
                "total_tokens": None,
            },
        )

    def test_provider_usage_is_copied_and_never_estimated(self) -> None:
        reported = provider_token_usage(
            SimpleNamespace(prompt_tokens=1200, completion_tokens=300, total_tokens=1500)
        )
        silent = provider_token_usage(None)
        partial = provider_token_usage(SimpleNamespace(prompt_tokens=1200))

        self.assertEqual(
            reported,
            {
                "source": "provider_exact",
                "prompt_tokens": 1200,
                "completion_tokens": 300,
                "total_tokens": 1500,
            },
        )
        self.assertEqual(
            silent,
            {
                "source": "unavailable",
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
        )
        # A total the provider did not report is left empty, not computed.
        self.assertIsNone(partial["total_tokens"])
        self.assertEqual(partial["prompt_tokens"], 1200)
        self.assertEqual(partial["source"], "provider_partial")

    def test_token_usage_carries_only_closed_technical_keys(self) -> None:
        service = AnalyzeService(make_settings())
        analysis = service._with_analysis_prompt_quality_flags(
            {},
            metrics={"profile": "compact"},
            prompt_version="v8",
            cache_hit=False,
            token_usage=provider_token_usage(SimpleNamespace(prompt_tokens=10)),
        )

        usage = service._build_analysis_meta(analysis)["token_usage"]

        self.assertEqual(
            set(usage), {"source", "prompt_tokens", "completion_tokens", "total_tokens"}
        )
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.assertTrue(usage[key] is None or isinstance(usage[key], int))

    def test_untrusted_roles_call_neither_the_model_nor_the_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_untrusted_") as td:
            db_path = Path(td) / "untrusted.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                analyze_provider="codex_cli",
                llm_cache_enabled=True,
                llm_cache_dir=str(Path(td) / "cache"),
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(_dialogue_call(source_file=str(Path(td) / "call.mp3")))
                session.commit()
            service = AnalyzeService(settings)
            counters = {"model": 0, "cache": 0}

            def no_model(*_args, **_kwargs):
                counters["model"] += 1
                raise AssertionError("the model must not be called for untrusted roles")

            def no_cache(*_args, **_kwargs):
                counters["cache"] += 1
                return None

            with patch.object(AnalyzeService, "_analyze_text", no_model), patch.object(
                AnalyzeService, "_analysis_cache_lookup", no_cache
            ):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["success"], 1)
            self.assertEqual(counters, {"model": 0, "cache": 0})
            with session_factory() as session:
                stored = json.loads(session.query(CallRecord).one().analysis_json)
            self.assertIs(stored["analysis_meta"]["model_called"], False)
            self.assertEqual(
                stored["analysis_meta"]["token_usage"]["source"],
                "skipped_untrusted_role",
            )
            self.assertEqual(stored["analysis_schema_version"], "v3")
            self.assertEqual(stored["claim_evidence"], [])

    def test_failed_model_attempts_are_persisted_without_replacing_analysis(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_failed_usage_") as td:
            db_path = Path(td) / "failed.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                analyze_provider="codex_cli",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    _trusted_dialogue_call(source_file=str(Path(td) / "call.mp3"))
                )
                session.commit()
            attempts = [
                {
                    "provider": "codex_cli",
                    "model": settings.codex_analyze_model,
                    "profile": "compact",
                    "prompt_version": "v8",
                    "cache_hit": False,
                    "model_called": True,
                    "token_usage": {
                        "source": "unavailable",
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                    },
                }
                for _ in range(5)
            ]
            failure = RuntimeError("provider failed")
            failure.model_attempts = attempts

            with patch.object(AnalyzeService, "_analyze_text", side_effect=failure):
                with session_factory() as session:
                    result = AnalyzeService(settings).run(session, limit=1)

            self.assertEqual(result["failed"], 1)
            with session_factory() as session:
                row = session.query(CallRecord).one()
                self.assertEqual(row.analysis_status, "failed")
                self.assertIsNone(row.analysis_json)
                self.assertEqual(len(json.loads(row.analysis_attempts_json)), 5)

    def test_trusted_nonconversation_skips_model_and_roundtrips_current_contract(self) -> None:
        turns = (
            ("operator", "left", "Добрый день."),
            (
                "client",
                "right",
                "Абонент сейчас не может ответить. Оставьте сообщение после сигнала.",
            ),
        )
        with tempfile.TemporaryDirectory(prefix="mango_nonconversation_meta_") as td:
            db_path = Path(td) / "nonconversation.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                analyze_provider="codex_cli",
                llm_cache_enabled=True,
                llm_cache_dir=str(Path(td) / "cache"),
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(Path(td) / "call.mp3"),
                        source_filename="call.mp3",
                        source_call_id=fx.SOURCE_CALL_ID,
                        source_recording_id=fx.RECORDING_ID,
                        transcription_status="done",
                        resolve_status="done",
                        analysis_status="pending",
                        duration_sec=20,
                        transcript_variants_json=json.dumps(
                            fx.proven_variants(turns), ensure_ascii=False
                        ),
                    )
                )
                session.commit()

            with patch.object(
                AnalyzeService,
                "_codex_cli_analysis",
                side_effect=AssertionError("model must not be called"),
            ), patch.object(
                AnalyzeService,
                "_analysis_cache_lookup",
                side_effect=AssertionError("cache must not be read"),
            ):
                with session_factory() as session:
                    result = AnalyzeService(settings).run(session, limit=1)

            self.assertEqual(result["success"], 1)
            with session_factory() as session:
                row = session.query(CallRecord).one()
                stored = json.loads(row.analysis_json)
                guarded = guard_stored_analysis(call_record_view(row), stored)
            self.assertFalse(stored["analysis_meta"]["model_called"])
            self.assertEqual(
                stored["analysis_meta"]["token_usage"]["source"],
                "skipped_deterministic",
            )
            self.assertNotIn("analysis_contract_invalid", guarded["review_reasons"])


class AnalysisCacheIdentityTest(unittest.TestCase):
    """Этап D: the cache answers only for the exact provider/model/prompt."""

    @staticmethod
    def _key():
        return {
            "namespace": "analyze",
            "provider": "codex_cli",
            "model": "gpt-5.4-mini",
            "reasoning": "low",
            "prompt_version": "v8",
            "prompt": "PROMPT-A",
        }

    def test_a_stored_answer_is_returned_for_the_same_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_cache_ok_") as td:
            cache = LLMResponseCache(enabled=True, root_dir=Path(td) / "cache")
            cache.put(**self._key(), response={"structured_fields": {}})

            self.assertEqual(cache.get(**self._key()), {"structured_fields": {}})

    def test_token_usage_sums_only_when_every_paid_stage_is_exact(self) -> None:
        exact = aggregate_token_usage(
            [
                {"token_usage": {"source": "provider_exact", "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}},
                {"token_usage": {"source": "provider_exact", "prompt_tokens": 20, "completion_tokens": 6, "total_tokens": 26}},
            ]
        )
        self.assertEqual(
            exact,
            {"source": "provider_exact", "prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
        )
        partial = aggregate_token_usage(
            [
                {"token_usage": {"source": "provider_exact", "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}},
                {"token_usage": {"source": "provider_partial", "prompt_tokens": 20, "completion_tokens": 6, "total_tokens": None}},
            ]
        )
        self.assertEqual(partial["source"], "provider_partial")
        self.assertEqual(partial["prompt_tokens"], 30)
        self.assertIsNone(partial["total_tokens"])
        legacy_partial = aggregate_token_usage(
            [
                {"token_usage": {"source": "provider", "prompt_tokens": 10, "completion_tokens": 4, "total_tokens": None}},
            ]
        )
        self.assertEqual(legacy_partial["source"], "provider_partial")
        unknown = aggregate_token_usage(
            [
                {"token_usage": exact},
                {"token_usage": {"source": "unavailable", "prompt_tokens": None, "completion_tokens": None, "total_tokens": None}},
            ]
        )
        self.assertEqual(unknown["source"], "unavailable")

        malformed_exact = aggregate_token_usage(
            [
                {
                    "token_usage": {
                        "source": "provider_exact",
                        "prompt_tokens": 10,
                        "completion_tokens": None,
                        "total_tokens": None,
                    }
                }
            ]
        )
        self.assertEqual(malformed_exact["source"], "provider_partial")
        self.assertIsNone(malformed_exact["completion_tokens"])
        self.assertIsNone(unknown["total_tokens"])

    def test_tampered_metadata_is_a_miss_not_a_hit(self) -> None:
        cases = (
            ("provider", "openai"),
            ("model", "some-cheaper-model"),
            ("prompt_version", "v1"),
            ("input_sha256", "0" * 64),
        )
        for field, value in cases:
            with self.subTest(field):
                with tempfile.TemporaryDirectory(prefix="mango_cache_bad_") as td:
                    cache = LLMResponseCache(enabled=True, root_dir=Path(td) / "cache")
                    cache.put(**self._key(), response={"structured_fields": {}})
                    path = next((Path(td) / "cache").rglob("*.json"))
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload[field] = value
                    path.write_text(
                        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
                    )

                    self.assertIsNone(cache.get(**self._key()))

    def test_a_legacy_or_poisoned_cached_answer_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_cache_contract_") as td:
            settings = replace(
                make_settings(),
                llm_cache_enabled=True,
                llm_cache_dir=str(Path(td) / "cache"),
            )
            service = AnalyzeService(settings)
            key = self._key()
            service._llm_cache.put(
                **key,
                response={"summary": "Старый ответ без v3-контракта"},
            )

            service_key = {name: value for name, value in key.items() if name != "namespace"}
            self.assertIsNone(service._analysis_cache_lookup(**service_key))

    def test_an_invalid_provider_answer_is_never_written_to_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_cache_store_contract_") as td:
            settings = replace(
                make_settings(),
                llm_cache_enabled=True,
                llm_cache_dir=str(Path(td) / "cache"),
            )
            service = AnalyzeService(settings)
            service_key = {
                name: value for name, value in self._key().items() if name != "namespace"
            }

            with self.assertRaises(AnalysisContractError):
                service._analysis_cache_store(
                    **service_key,
                    response={"summary": "Ответ вне v3-контракта"},
                )
            self.assertEqual(list((Path(td) / "cache").rglob("*.json")), [])

    def test_the_key_moves_with_input_model_and_prompt_version_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_cache_key_") as td:
            cache = LLMResponseCache(enabled=True, root_dir=Path(td) / "cache")
            cache.put(**self._key(), response={"structured_fields": {}})

            self.assertIsNone(cache.get(**{**self._key(), "prompt": "PROMPT-B"}))
            self.assertIsNone(cache.get(**{**self._key(), "model": "other-model"}))
            self.assertIsNone(cache.get(**{**self._key(), "prompt_version": "v9"}))
            self.assertIsNotNone(cache.get(**self._key()))
            self.assertEqual(len(list((Path(td) / "cache").rglob("*.json"))), 1)

    def test_a_second_identical_analysis_calls_no_model(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_cache_repeat_") as td:
            settings = replace(
                make_settings(),
                analyze_provider="codex_cli",
                codex_analyze_model="gpt-5.4-mini",
                llm_cache_enabled=True,
                llm_cache_dir=str(Path(td) / "cache"),
            )
            service = AnalyzeService(settings)
            call = _trusted_dialogue_call()
            dialogue = build_dialogue_input(call_record_view(call))
            state = {"calls": 0}

            def fake_run(cmd, capture_output, text, check, timeout, input=None):
                state["calls"] += 1
                Path(cmd[cmd.index("--output-last-message") + 1]).write_text(
                    json.dumps(_v3_answer(), ensure_ascii=False), encoding="utf-8"
                )
                return CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch(
                "mango_mvp.services.analyze.shutil.which", return_value="/bin/codex"
            ):
                with patch(
                    "mango_mvp.services.analyze.subprocess.run", side_effect=fake_run
                ):
                    first = service._codex_cli_analysis(
                        call, dialogue.render(), "compact", dialogue
                    )
                    second = service._codex_cli_analysis(
                        call, dialogue.render(), "compact", dialogue
                    )

            self.assertEqual(state["calls"], 1)
            self.assertFalse(first["quality_flags"]["analyze_llm_cache_hit"])
            self.assertTrue(second["quality_flags"]["analyze_llm_cache_hit"])
            self.assertEqual(
                second["quality_flags"]["analyze_token_usage"]["source"], "cache_hit"
            )
            self.assertIs(service._build_analysis_meta(second)["model_called"], False)

    def test_provider_result_without_durable_ledger_never_enters_cache(self) -> None:
        service = AnalyzeService(
            replace(make_settings(), analyze_provider="codex_cli", llm_cache_enabled=True)
        )
        call = _trusted_dialogue_call()
        dialogue = build_dialogue_input(call_record_view(call))
        service._analysis_attempt_context = {
            "session": object(), "call_id": 1, "snapshot": {},
            "source_sha": "a" * 64, "worker_id": "worker",
            "run_attempt": 1, "sequence": 0, "cache_writes": [],
        }
        ledger_calls = {"count": 0}

        def ledger(*_args, **_kwargs):
            ledger_calls["count"] += 1
            if ledger_calls["count"] == 1:
                return True
            raise RuntimeError("ledger finalize was not durable")

        def fake_run(cmd, **_kwargs):
            Path(cmd[cmd.index("--output-last-message") + 1]).write_text(
                json.dumps(_v3_answer(), ensure_ascii=False), encoding="utf-8"
            )
            return CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("mango_mvp.services.analyze.shutil.which", return_value="/bin/codex"), patch(
            "mango_mvp.services.analyze.subprocess.run", side_effect=fake_run
        ), patch.object(service, "_store_analysis_attempt", side_effect=ledger), patch.object(
            service, "_analysis_cache_store"
        ) as cache_store:
            with self.assertRaisesRegex(RuntimeError, "ledger finalize"):
                service._codex_cli_analysis(
                    call, dialogue.render(), "compact", dialogue
                )

        cache_store.assert_not_called()
        self.assertEqual(service._analysis_attempt_context["cache_writes"], [])

    def test_a_normalizer_or_publisher_bump_does_not_re_ask_the_model(self) -> None:
        service = AnalyzeService(make_settings())
        call = _trusted_dialogue_call()
        dialogue = build_dialogue_input(call_record_view(call))

        prompt = service._analysis_prompt_context(
            call, dialogue.render(), "compact", dialogue
        )["llm_prompt"]

        # The paid prompt is the dialogue and the contract, nothing else: no
        # normalizer ruleset version and no Google projection version reach it,
        # so bumping either cannot invalidate a single cached answer.
        for version in (
            "tenant_text_engine_v1",
            "tenant_ru_v1",
            "mango_calls_live_google_projection",
        ):
            self.assertNotIn(version, prompt)
