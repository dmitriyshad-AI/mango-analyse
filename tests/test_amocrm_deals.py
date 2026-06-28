from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import tempfile
from datetime import datetime, timezone
import dataclasses
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from urllib import error as url_error, parse as url_parse

from mango_mvp.amocrm_runtime import deal_llm as deal_llm_module
from mango_mvp.amocrm_runtime.deal_dossier import build_deal_dossier
from mango_mvp.amocrm_runtime.deal_llm import DealLLMAnalyzer
from mango_mvp.amocrm_runtime.phone_context import PhoneContext
from mango_mvp.amocrm_runtime import deals as deals_module
from mango_mvp.amocrm_runtime import amo_integration


class AmoCrmDealAnalysisTest(unittest.TestCase):
    def test_build_deal_dossier_includes_transcript_context_from_source_db(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_deal_dossier_") as td:
            db_path = Path(td) / "calls.db"
            con = sqlite3.connect(db_path)
            con.execute(
                """
                CREATE TABLE call_records (
                    id INTEGER PRIMARY KEY,
                    source_filename TEXT,
                    started_at TEXT,
                    manager_name TEXT,
                    duration_sec REAL,
                    transcript_text TEXT,
                    transcript_variants_json TEXT,
                    analysis_json TEXT,
                    resolve_status TEXT,
                    analysis_status TEXT
                )
                """
            )
            con.execute(
                """
                INSERT INTO call_records (
                    source_filename, started_at, manager_name, duration_sec,
                    transcript_text, transcript_variants_json, analysis_json,
                    resolve_status, analysis_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "2026-04-10__10-00-00__Иванов Иван__79990001122.mp3",
                    "2026-04-10 10:00:00",
                    "Иванов Иван",
                    180.0,
                    "[00:01.0] Клиент: Мы не отказываемся, вернемся после экзаменов.\n[00:04.0] Менеджер: Хорошо, я поставлю follow-up.",
                    '{"variant_b": {"transcript_text": "альтернативный вариант gigaam"}}',
                    '{"history_summary": "Клиент просит вернуться после экзаменов."}',
                    "done",
                    "done",
                ),
            )
            con.commit()
            con.close()

            phone_context = PhoneContext(
                phone="+79990001122",
                source_dir=td,
                contact_row={
                    "Всего звонков в истории": "3",
                    "Звонков с полным анализом": "3",
                    "Незакрытых звонков в истории": "0",
                    "Полная история проанализирована": "Да",
                    "Последний свежий звонок": "2026-04-10 10:00:00",
                    "Последний свежий звонок проанализирован": "Да",
                    "Менеджер последнего свежего звонка": "Иванов Иван",
                    "Краткое резюме последнего свежего звонка": "Клиент не отказался, просит вернуться позже.",
                    "Тип последнего свежего звонка": "sales_call",
                    "Краткая история общения": "Клиент интересуется курсом и просит вернуться после экзаменов.",
                    "Хронология общения (последние 5 касаний)": "10.04.2026 — Иванов Иван: вернуться после экзаменов",
                    "Рекомендуемый продукт": "годовые курсы",
                    "Следующий шаг": "Перезвонить после экзаменов",
                    "Рекомендуемая дата следующего контакта": "2026-04-20",
                    "Приоритет лида": "warm",
                    "ID Tallanto": "T-100",
                    "Статус матчинга Tallanto": "exact_phone_single",
                },
                call_rows=[
                    {
                        "ID звонка": "101",
                        "Дата и время звонка": "2026-04-10 10:00:00",
                        "Менеджер": "Иванов Иван",
                        "Направление звонка": "outbound",
                        "Длительность, сек": "180",
                        "Тип звонка": "sales_call",
                        "Свежий период": "Да",
                        "Статус Resolve": "done",
                        "Статус Analyze": "done",
                        "Краткое резюме разговора": "Клиент попросил вернуться после экзаменов.",
                        "Возражения": "сроки",
                        "Следующий шаг": "Перезвонить после экзаменов",
                        "Рекомендуемая дата следующего контакта": "2026-04-20",
                        "Имя исходного файла": "2026-04-10__10-00-00__Иванов Иван__79990001122.mp3",
                        "Источник лучшего статуса": str(db_path),
                    }
                ],
                call_ids=["101"],
                first_call_at="2026-04-10 10:00:00",
                last_call_at="2026-04-10 10:00:00",
                manager_history=["Иванов Иван"],
                interest_summary="годовые курсы",
                objections_summary="сроки",
                current_sales_temperature="warm",
                recommended_next_step="Перезвонить после экзаменов",
                follow_up_due_at="2026-04-20",
                history_summary="Клиент интересуется курсом и просит вернуться позже.",
                chronology="10.04.2026 — Иванов Иван: вернуться после экзаменов",
                tallanto_id="T-100",
                tallanto_match_status="exact_phone_single",
            )

            dossier = build_deal_dossier(
                phone_context=phone_context,
                contact={"id": 1, "name": "Ивановы"},
                lead={"id": 10, "name": "Сделка", "pipeline_id": 100, "status_id": 143},
                notes=[{"id": 11, "params": {"text": "Клиент попросил вернуться после экзаменов"}}],
                tasks=[{"id": 12, "text": "Перезвонить", "responsible_user_id": 77}],
                pipeline_name="Сделки B2C",
                status_name="Закрыто и не реализовано",
                user_map={77: "Иванов Иван"},
            )

            self.assertEqual(dossier["lead"]["id"], 10)
            self.assertEqual(len(dossier["call_history"]), 1)
            self.assertEqual(len(dossier["transcript_context"]), 1)
            self.assertIn("после экзаменов", dossier["transcript_context"][0]["transcript_excerpt"].lower())
            self.assertIn("variant_b", dossier["transcript_context"][0]["variant_overview"]["available_variants"])
            self.assertEqual(dossier["notes"][0]["text"], "Клиент попросил вернуться после экзаменов")
            self.assertEqual(dossier["tasks"][0]["responsible_user_name"], "Иванов Иван")

    def test_deal_llm_normalize_response_falls_back_to_manual_review(self) -> None:
        analyzer = DealLLMAnalyzer()
        normalized = analyzer._normalize_response(
            {
                "close_verdict": "something_wrong",
                "premature_close_risk": "panic",
                "close_reason_summary": "Недостаточно данных.",
                "recommended_next_step": "Проверить вручную",
                "confidence": 0.2,
                "needs_manual_review": False,
                "evidence_signals": ["Клиент не отказался окончательно"],
            }
        )
        self.assertEqual(normalized["close_verdict"], "manual_review")
        self.assertEqual(normalized["premature_close_risk"], "manual_review")
        self.assertTrue(normalized["needs_manual_review"])
        self.assertEqual(normalized["confidence"], 0.2)

    def test_deal_llm_codex_cli_uses_scrubbed_env(self) -> None:
        analyzer = DealLLMAnalyzer()
        analyzer._settings = SimpleNamespace(
            codex_cli_path="codex",
            crm_analysis_model="gpt-5.5",
            crm_analysis_reasoning_effort="low",
            crm_analysis_timeout_seconds=15,
        )
        analyzer._cache_lookup = lambda **kwargs: None
        analyzer._cache_store = lambda **kwargs: None
        seen: dict[str, Any] = {}

        def fake_run(cmd, **kwargs):
            seen["env"] = dict(kwargs["env"])
            output_path = Path(cmd[cmd.index("--output-last-message") + 1])
            output_path.write_text(
                json.dumps(
                    {
                        "close_verdict": "manual_review",
                        "premature_close_risk": "manual_review",
                        "close_reason_summary": "Недостаточно данных.",
                        "recommended_next_step": "Проверить вручную",
                        "confidence": 0.7,
                        "needs_manual_review": True,
                        "evidence_signals": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with tempfile.TemporaryDirectory(prefix="mango_deal_llm_test_home_") as td:
            with patch.object(analyzer, "_prepare_runtime_codex_home", return_value=td), patch.object(
                deal_llm_module.shutil, "which", return_value="/bin/codex"
            ), patch.object(deal_llm_module.subprocess, "run", fake_run), patch.dict(
                os.environ,
                {
                    "PATH": "/bin",
                    "HOME": "/home/test",
                    "OPENAI_API_KEY": "openai",
                    "AMO_TOKEN": "amo",
                    "WAPPI_SECRET": "wappi",
                    "CRM_AMO_API_TOKEN": "crm",
                    "AI_OFFICE_API_KEY": "office",
                },
                clear=False,
            ):
                normalized = analyzer._analyze_codex_cli(prompt="Верни JSON")

        self.assertEqual(normalized["close_verdict"], "manual_review")
        self.assertEqual(seen["env"]["CODEX_HOME"], str(Path(td).resolve()))
        self.assertEqual(seen["env"]["PATH"], "/bin")
        self.assertEqual(seen["env"]["HOME"], "/home/test")
        self.assertNotIn("OPENAI_API_KEY", seen["env"])
        self.assertNotIn("AMO_TOKEN", seen["env"])
        self.assertNotIn("WAPPI_SECRET", seen["env"])
        self.assertNotIn("CRM_AMO_API_TOKEN", seen["env"])
        self.assertNotIn("AI_OFFICE_API_KEY", seen["env"])

    def test_build_dossier_and_analysis_passes_active_brand_to_dossier(self) -> None:
        phone_context = PhoneContext(
            phone="+79990001122",
            source_dir="",
            contact_row={},
            call_rows=[],
            call_ids=[],
            first_call_at="",
            last_call_at="",
            manager_history=[],
            interest_summary="",
            objections_summary="",
            current_sales_temperature="warm",
            recommended_next_step="Перезвонить",
            follow_up_due_at="2026-04-20",
            history_summary="",
            chronology="",
            tallanto_id="",
            tallanto_match_status="",
        )
        candidate = deals_module.LeadCandidate(
            contact_id=1,
            lead_id=10,
            score=100,
            confidence=1.0,
            reason="test",
            lead={"id": 10},
        )
        captured: dict[str, Any] = {}

        def fake_build_deal_dossier(**kwargs):
            captured.update(kwargs)
            return {"schema_version": "test_dossier"}

        with patch.object(deals_module, "fetch_lead", return_value={"id": 10, "pipeline_id": 100, "status_id": 143}), patch.object(
            deals_module, "fetch_lead_notes", return_value=[]
        ), patch.object(deals_module, "fetch_lead_tasks", return_value=[]), patch.object(
            deals_module, "build_deal_dossier", side_effect=fake_build_deal_dossier
        ):
            deals_module._build_dossier_and_analysis(
                object(),
                phone_context=phone_context,
                candidate=candidate,
                contact={"id": 1, "name": "Ивановы"},
                pipelines=[{"id": 100, "name": "Сделки B2C", "statuses": [{"id": 143, "name": "Закрыто"}]}],
                users=[],
                active_brand="foton",
            )

        self.assertEqual(captured["active_brand"], "foton")

    def test_prepare_writeback_payload_does_not_include_ai_office_by_default(self) -> None:
        payload = deals_module._prepare_writeback_payload(
            {
                "close_verdict": "follow_up_needed",
                "premature_close_risk": "medium",
                "close_reason_summary": "Клиент просит вернуться позже.",
                "recommended_next_step": "Перезвонить клиенту",
                "follow_up_due_at": "2026-04-20",
                "deal_summary": "Есть шанс на возврат.",
                "status_id": 143,
                "latest_call_summary": "Клиент просит вернуться позже.",
                "history_summary": "Интерес к курсу сохраняется.",
                "chronology": "10.04.2026 — попросил перезвонить",
                "objections_summary": "сроки",
                "ai_office": "служебная строка",
            }
        )
        self.assertIn("AI-вердикт по закрытию", payload)
        self.assertIn("AI-сводка по сделке", payload)
        self.assertNotIn("AI office", payload)
        self.assertEqual(payload["AI-вердикт по закрытию"], "Нужен follow-up")
        self.assertEqual(payload["AI-risk: premature close"], "Средний")

    def test_prepare_writeback_payload_for_open_deal_only_uses_context_fields(self) -> None:
        payload = deals_module._prepare_writeback_payload(
            {
                "close_verdict": "follow_up_needed",
                "premature_close_risk": "low",
                "close_reason_summary": "Сделка живая.",
                "recommended_next_step": "Отправить материалы",
                "follow_up_due_at": "2026-04-20",
                "deal_summary": "Клиент в работе.",
                "status_id": 124,
                "latest_call_summary": "Клиент ждет материалы.",
                "history_summary": "Ранее обсуждали программу.",
                "chronology": "12.04.2026 — ждут материалы",
                "objections_summary": "цена",
            }
        )
        self.assertNotIn("AI-вердикт по закрытию", payload)
        self.assertNotIn("AI-risk: premature close", payload)
        self.assertNotIn("AI-основание вердикта", payload)
        self.assertEqual(payload["AI-рекомендованный следующий шаг"], "Отправить материалы")
        self.assertIn("Полная история общения", payload["AI-сводка по сделке"])

    def test_prepare_writeback_payload_for_closed_valid_deal_is_empty(self) -> None:
        payload = deals_module._prepare_writeback_payload(
            {
                "close_verdict": "closed_valid",
                "premature_close_risk": "no_risk",
                "close_reason_summary": "Закрыта корректно.",
                "recommended_next_step": "",
                "follow_up_due_at": None,
                "deal_summary": "Корректно закрытая сделка",
                "status_id": 143,
            }
        )
        self.assertEqual(payload, {})

    def test_write_analysis_to_lead_rejects_blocked_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "write-back is blocked"):
            deals_module.write_analysis_to_lead(
                None,  # type: ignore[arg-type]
                analysis={
                    "matched_lead_id": 123,
                    "writeback_blockers": ["shadow_mode"],
                    "close_verdict": "follow_up_needed",
                },
            )

    def test_finalize_analysis_marks_shadow_mode_as_non_writeable(self) -> None:
        heuristic = {
            "close_verdict": "closed_valid",
            "premature_close_risk": "no_risk",
            "match_confidence": 0.9,
            "analysis_source": "heuristic",
        }
        llm = {
            "close_verdict": "follow_up_needed",
            "premature_close_risk": "medium",
            "confidence": 0.88,
            "needs_manual_review": False,
            "conflict_flags": [],
        }
        with patch.object(deals_module, "_analysis_mode", return_value="llm_shadow"):
            final, _, comparison = deals_module._finalize_analysis(
                heuristic_analysis=heuristic,
                llm_analysis=llm,
            )
        self.assertEqual(final["analysis_source"], "llm")
        self.assertFalse(final["writeback_allowed"])
        self.assertIn("shadow_mode", final["writeback_blockers"])
        self.assertTrue(comparison["verdict_changed"])

    def test_finalize_analysis_allows_business_contradictions_in_conflict_flags(self) -> None:
        heuristic = {
            "close_verdict": "closed_valid",
            "premature_close_risk": "low",
            "match_confidence": 0.94,
            "analysis_source": "heuristic",
        }
        llm = {
            "close_verdict": "reopen_recommended",
            "premature_close_risk": "high",
            "confidence": 0.91,
            "needs_manual_review": False,
            "conflict_flags": [
                "loss_reason_conflicts_with_call_history",
                "agreed_next_step_but_lead_closed",
            ],
        }
        with patch.object(deals_module, "_analysis_mode", return_value="llm_primary"):
            final, _, comparison = deals_module._finalize_analysis(
                heuristic_analysis=heuristic,
                llm_analysis=llm,
            )
        self.assertEqual(final["analysis_source"], "llm")
        self.assertTrue(final["writeback_allowed"])
        self.assertEqual(final["writeback_blockers"], [])
        self.assertTrue(comparison["severe_conflict"])

    def test_finalize_analysis_blocks_true_ambiguity_conflict_flags(self) -> None:
        heuristic = {
            "close_verdict": "follow_up_needed",
            "premature_close_risk": "medium",
            "match_confidence": 0.95,
            "analysis_source": "heuristic",
        }
        llm = {
            "close_verdict": "follow_up_needed",
            "premature_close_risk": "medium",
            "confidence": 0.92,
            "needs_manual_review": False,
            "conflict_flags": ["несколько одинаково вероятных сделок"],
        }
        with patch.object(deals_module, "_analysis_mode", return_value="llm_primary"):
            final, _, _ = deals_module._finalize_analysis(
                heuristic_analysis=heuristic,
                llm_analysis=llm,
            )
        self.assertFalse(final["writeback_allowed"])
        self.assertIn("llm_conflict_flags", final["writeback_blockers"])

    def test_finalize_analysis_blocks_low_confidence_severe_heuristic_llm_conflict(self) -> None:
        heuristic = {
            "close_verdict": "closed_valid",
            "premature_close_risk": "no_risk",
            "match_confidence": 0.94,
            "analysis_source": "heuristic",
        }
        llm = {
            "close_verdict": "reopen_recommended",
            "premature_close_risk": "high",
            "confidence": 0.71,
            "needs_manual_review": False,
            "conflict_flags": [],
        }
        with patch.object(deals_module, "_analysis_mode", return_value="llm_primary"):
            final, _, comparison = deals_module._finalize_analysis(
                heuristic_analysis=heuristic,
                llm_analysis=llm,
            )
        self.assertFalse(final["writeback_allowed"])
        self.assertIn("heuristic_llm_conflict", final["writeback_blockers"])
        self.assertTrue(comparison["severe_conflict"])

    def test_build_recent_closed_queue_respects_max_leads_on_manual_review_branch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_deal_queue_") as td:
            run_dir = Path(td)
            now = datetime.now(timezone.utc).isoformat()
            recent_leads = [
                {
                    "id": 101,
                    "name": "Lead 101",
                    "pipeline_id": 100,
                    "status_id": 143,
                    "closed_at": now,
                    "_embedded": {"contacts": [{"id": 201}]},
                },
                {
                    "id": 102,
                    "name": "Lead 102",
                    "pipeline_id": 100,
                    "status_id": 143,
                    "closed_at": now,
                    "_embedded": {"contacts": [{"id": 202}]},
                },
                {
                    "id": 103,
                    "name": "Lead 103",
                    "pipeline_id": 100,
                    "status_id": 143,
                    "closed_at": now,
                    "_embedded": {"contacts": [{"id": 203}]},
                },
            ]
            pipelines = [
                {
                    "id": 100,
                    "name": "Сделки B2C",
                    "_embedded": {"statuses": [{"id": 143, "name": "Закрыто и не реализовано"}]},
                }
            ]

            with (
                patch.object(deals_module, "_queue_dir", return_value=run_dir),
                patch.object(deals_module, "fetch_pipelines_with_statuses", return_value=pipelines),
                patch.object(deals_module, "fetch_users", return_value=[]),
                patch.object(deals_module, "fetch_recent_leads", return_value=recent_leads),
                patch.object(deals_module, "_default_target_pipeline_ids", return_value={100}),
                patch.object(deals_module, "_choose_best_phone_context_for_contact", return_value=(None, None, {})),
            ):
                summary = deals_module.build_recent_closed_queue(
                    None,  # type: ignore[arg-type]
                    days_back=30,
                    max_leads=1,
                )

            self.assertEqual(summary["analyzed"], 1)
            self.assertEqual(summary["manual_review"], 1)
            latest = json.loads((run_dir / "latest_run.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["analyzed"], 1)
            rows = json.loads(Path(summary["files"]["all_results_json"]).read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["matched_lead_id"], 101)

    def test_build_custom_fields_values_clips_long_text_fields(self) -> None:
        catalog = [
            {"id": 1, "name": "AI-основание вердикта", "type": "text"},
            {"id": 2, "name": "AI-сводка по сделке", "type": "text"},
            {"id": 3, "name": "AI-вердикт по закрытию", "type": "text"},
        ]
        payload = {
            "AI-основание вердикта": "А" * 300,
            "AI-сводка по сделке": "Б" * 260,
            "AI-вердикт по закрытию": "reopen_recommended",
        }
        values = amo_integration.build_custom_fields_values(payload, catalog)
        by_id = {item["field_id"]: item["values"][0]["value"] for item in values}
        self.assertEqual(len(by_id[1]), 255)
        self.assertTrue(str(by_id[1]).endswith(" [сжато]"))
        self.assertNotIn("...", str(by_id[1]))
        self.assertEqual(len(by_id[2]), 255)
        self.assertEqual(by_id[3], "reopen_recommended")

    def test_build_custom_fields_values_converts_date_time_fields(self) -> None:
        catalog = [
            {"id": 1, "name": "AI-дата обновления сделки", "type": "date_time"},
            {"id": 2, "name": "AI-дата следующего касания", "type": "date"},
        ]
        payload = {
            "AI-дата обновления сделки": "2026-05-13T11:57:33+00:00",
            "AI-дата следующего касания": "2026-05-15",
        }

        values = amo_integration.build_custom_fields_values(payload, catalog)
        by_id = {item["field_id"]: item["values"][0]["value"] for item in values}

        self.assertIsInstance(by_id[1], int)
        self.assertEqual(by_id[1], 1778673453)
        self.assertEqual(by_id[2], 1778803200)

    def test_fetch_leads_batch_uses_amo_filter_id_contract_and_preserves_input_order(self) -> None:
        captured_urls: list[str] = []

        def fake_http_request(**kwargs):
            captured_urls.append(kwargs["url"])
            return {"_embedded": {"leads": [{"id": 22, "name": "Lead 22"}, {"id": 11, "name": "Lead 11"}]}}

        with patch.object(
            amo_integration,
            "resolve_amo_access_context",
            return_value=amo_integration.AmoAccessContext(
                account_base_url="https://educent.amocrm.ru",
                access_token="token",
                token_source="oauth",
                connection=None,
            ),
        ), patch.object(amo_integration, "_amo_http_request", side_effect=fake_http_request):
            leads = amo_integration.fetch_leads_batch(
                None,  # type: ignore[arg-type]
                lead_ids=[11, 22],
                with_fields="contacts",
            )

        query = url_parse.parse_qs(url_parse.urlparse(captured_urls[0]).query)
        self.assertEqual(query["filter[id][]"], ["11", "22"])
        self.assertEqual(query["with"], ["contacts"])
        self.assertEqual(query["limit"], ["2"])
        self.assertEqual([lead["id"] for lead in leads], [11, 22])

    def test_resolve_target_lead_batch_fetch_is_flagged_and_keeps_selected_lead(self) -> None:
        phone_context = PhoneContext(
            phone="+79990001122",
            source_dir="",
            contact_row={},
            call_rows=[{"Тип звонка": "sales_call"}],
            call_ids=[],
            first_call_at="2026-04-10 10:00:00",
            last_call_at="2026-04-10 10:00:00",
            manager_history=["Иванов Иван"],
            interest_summary="",
            objections_summary="",
            current_sales_temperature="warm",
            recommended_next_step="",
            follow_up_due_at="",
            history_summary="",
            chronology="",
            tallanto_id="",
            tallanto_match_status="",
        )
        contact = {"id": 123, "name": "Контакт", "_embedded": {"leads": [{"id": 11}, {"id": 22}]}}
        pipelines = [
            {
                "id": 100,
                "name": "Сделки B2C",
                "_embedded": {"statuses": [{"id": 143, "name": "Закрыто и не реализовано"}]},
            }
        ]
        users = [{"id": 1, "name": "Иванов Иван"}]
        leads_by_id = {
            11: {
                "id": 11,
                "name": "Best",
                "pipeline_id": 100,
                "status_id": 143,
                "responsible_user_id": 1,
                "updated_at": "2026-04-10 10:00:00",
            },
            22: {
                "id": 22,
                "name": "Older",
                "pipeline_id": 100,
                "status_id": 143,
                "responsible_user_id": 0,
                "updated_at": "2025-01-01 10:00:00",
            },
        }

        def run_resolve(*, batch_enabled: bool | None) -> tuple[dict[str, Any], int, int]:
            fetch_one_calls = 0
            fetch_batch_calls = 0

            def fake_fetch_lead(session, *, lead_id, with_fields="contacts"):
                nonlocal fetch_one_calls
                fetch_one_calls += 1
                return leads_by_id[int(lead_id)]

            def fake_fetch_leads_batch(session, *, lead_ids, with_fields="contacts"):
                nonlocal fetch_batch_calls
                fetch_batch_calls += 1
                return [leads_by_id[int(lead_id)] for lead_id in lead_ids]

            env_patch = {} if batch_enabled is None else {"AMO_LEADS_BATCH_FETCH": "1" if batch_enabled else "0"}
            with patch.dict("os.environ", env_patch), patch.object(
                deals_module,
                "get_phone_context",
                return_value=phone_context,
            ), patch.object(
                deals_module,
                "search_contacts_by_phone",
                return_value=[contact],
            ), patch.object(
                deals_module,
                "fetch_pipelines_with_statuses",
                return_value=pipelines,
            ), patch.object(
                deals_module,
                "fetch_users",
                return_value=users,
            ), patch.object(
                deals_module,
                "fetch_lead",
                side_effect=fake_fetch_lead,
            ), patch.object(
                deals_module,
                "fetch_leads_batch",
                side_effect=fake_fetch_leads_batch,
            ), patch.object(
                deals_module,
                "fetch_related_leads",
                side_effect=AssertionError("embedded leads should skip related-leads fetch"),
            ):
                if batch_enabled is None:
                    os.environ.pop("AMO_LEADS_BATCH_FETCH", None)
                result = deals_module.resolve_target_lead(None, phone="+79990001122")  # type: ignore[arg-type]
            return result, fetch_one_calls, fetch_batch_calls

        off_result, off_single_calls, off_batch_calls = run_resolve(batch_enabled=False)
        on_result, on_single_calls, on_batch_calls = run_resolve(batch_enabled=True)
        default_result, default_single_calls, default_batch_calls = run_resolve(batch_enabled=None)

        self.assertEqual(off_result["candidates"], on_result["candidates"])
        self.assertEqual(off_result["selected"], on_result["selected"])
        self.assertEqual(off_result["selected"]["lead_id"], 11)
        self.assertEqual(off_single_calls, 2)
        self.assertEqual(off_batch_calls, 0)
        self.assertEqual(on_single_calls, 0)
        self.assertEqual(on_batch_calls, 1)
        self.assertEqual(default_result["candidates"], on_result["candidates"])
        self.assertEqual(default_result["selected"], on_result["selected"])
        self.assertEqual(default_single_calls, 0)
        self.assertEqual(default_batch_calls, 1)

    def test_write_analysis_to_lead_safe_mode_skips_nonempty_fields(self) -> None:
        with patch.object(
            deals_module,
            "fetch_lead",
            return_value={
                "id": 777,
                "custom_fields_values": [
                    {
                        "field_name": "AI-сводка по сделке",
                        "values": [{"value": "Старое ручное значение"}],
                    },
                    {
                        "field_name": "AI-рекомендованный следующий шаг",
                        "values": [{"value": ""}],
                    },
                ],
            },
        ), patch.object(
            deals_module,
            "send_lead_custom_field_update",
            return_value={"entity_id": 777, "updated_fields": ["AI-рекомендованный следующий шаг"]},
        ) as send_mock:
            result = deals_module.write_analysis_to_lead(
                None,  # type: ignore[arg-type]
                analysis={
                    "matched_lead_id": 777,
                    "writeback_blockers": [],
                    "close_verdict": "follow_up_needed",
                    "premature_close_risk": "medium",
                    "recommended_next_step": "Перезвонить",
                    "follow_up_due_at": "2026-04-20",
                    "deal_summary": "Новая AI-сводка",
                    "status_id": 124,
                    "latest_call_summary": "Контекст клиента",
                },
            )

        field_payload = send_mock.call_args.kwargs["field_payload"]
        self.assertEqual(result["status"], "written")
        self.assertEqual(
            field_payload,
            {
                "AI-рекомендованный следующий шаг": "Перезвонить",
                "AI-дата следующего касания": "2026-04-20",
            },
        )
        self.assertEqual(result["skipped_fields"], ["AI-сводка по сделке"])

    def test_write_analysis_to_lead_safe_mode_returns_skipped_when_everything_is_nonempty(self) -> None:
        with patch.object(
            deals_module,
            "fetch_lead",
            return_value={
                "id": 778,
                "custom_fields_values": [
                    {
                        "field_name": "AI-рекомендованный следующий шаг",
                        "values": [{"value": "Старое значение"}],
                    },
                    {
                        "field_name": "AI-сводка по сделке",
                        "values": [{"value": "Старый контекст"}],
                    },
                    {
                        "field_name": "AI-дата следующего касания",
                        "values": [{"value": "2026-04-18"}],
                    },
                ],
            },
        ), patch.object(
            deals_module,
            "send_lead_custom_field_update",
        ) as send_mock:
            result = deals_module.write_analysis_to_lead(
                None,  # type: ignore[arg-type]
                analysis={
                    "matched_lead_id": 778,
                    "writeback_blockers": [],
                    "close_verdict": "follow_up_needed",
                    "premature_close_risk": "medium",
                    "recommended_next_step": "Новое значение",
                    "follow_up_due_at": "2026-04-20",
                    "deal_summary": "Новый контекст",
                    "status_id": 124,
                    "latest_call_summary": "Контекст клиента",
                },
            )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["reason"], "safe_mode_prevented_overwrite")
        send_mock.assert_not_called()

    def test_send_contact_custom_field_update_skips_tallanto_identity_fields(self) -> None:
        catalog = [
            {"id": 1, "name": "Id Tallanto", "type": "text"},
            {"id": 2, "name": "Филиал Tallanto", "type": "multiselect"},
            {"id": 3, "name": "AI-приоритет", "type": "text"},
            {"id": 4, "name": "Авто история общения", "type": "textarea"},
        ]
        with patch.object(
            amo_integration,
            "resolve_amo_access_context",
            return_value=amo_integration.AmoAccessContext(
                account_base_url="https://educent.amocrm.ru",
                access_token="token",
                token_source="oauth",
                connection=None,
            ),
        ), patch.object(
            amo_integration,
            "fetch_contact_field_catalog",
            return_value=catalog,
        ), patch.object(
            amo_integration,
            "_amo_http_request",
            return_value={"id": 123},
        ) as request_mock:
            result = amo_integration.send_contact_custom_field_update(
                None,  # type: ignore[arg-type]
                contact_id=123,
                field_payload={
                    "Id Tallanto": "T-1",
                    "Филиал Tallanto": "Онлайн",
                    "AI-приоритет": "warm",
                    "Авто история общения": "Полная история",
                },
            )

        body = request_mock.call_args.kwargs["body"]
        field_ids = [item["field_id"] for item in body["custom_fields_values"]]
        self.assertEqual(field_ids, [3, 4])
        self.assertEqual(result["updated_fields"], ["AI-приоритет", "Авто история общения"])

    def test_refresh_connection_marks_reauthorization_required_on_revoked_token(self) -> None:
        connection = SimpleNamespace(
            id="conn-1",
            client_id="cid",
            client_secret="secret",
            refresh_token="refresh",
            redirect_uri="https://example.test/callback",
            account_base_url="https://educent.amocrm.ru",
            access_token="dead-token",
            expires_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            status="active",
            last_error=None,
        )

        class _Scalars:
            def one(self) -> SimpleNamespace:
                return connection

        class _Session:
            def __init__(self) -> None:
                self.flushed = False

            def scalars(self, *_args, **_kwargs) -> _Scalars:
                return _Scalars()

            def flush(self) -> None:
                self.flushed = True

        session = _Session()

        with patch.object(
            amo_integration,
            "_exchange_token",
            side_effect=amo_integration.AmoIntegrationError(
                "HTTP 401 from amoCRM: Token has been revoked",
                status_code=401,
            ),
        ):
            with self.assertRaises(amo_integration.AmoIntegrationError):
                amo_integration.refresh_connection_tokens(session, connection)

        self.assertEqual(connection.status, "reauthorization_required")
        self.assertIn("Token has been revoked", connection.last_error)
        self.assertTrue(session.flushed)

    def test_status_marks_stale_oauth_connection_as_not_connected(self) -> None:
        connection = SimpleNamespace(
            access_token="token",
            refresh_token="refresh",
            client_id="cid",
            client_secret="secret",
            account_base_url="https://educent.amocrm.ru",
            account_subdomain="educent",
            authorized_at=None,
            expires_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            last_error=None,
            contact_field_catalog=[],
            contact_field_catalog_synced_at=None,
            status="active",
        )
        with patch.object(amo_integration, "get_active_connection", return_value=connection), patch.object(
            amo_integration, "build_external_oauth_setup", return_value={}
        ), patch.object(amo_integration, "fetch_lead_field_catalog", return_value=[]):
            payload = amo_integration.get_amo_connection_status(SimpleNamespace())
        self.assertFalse(payload["connected"])
        self.assertEqual(payload["status"], "token_stale")

    def test_status_does_not_refresh_amo_when_connection_is_awaiting_callback(self) -> None:
        connection = SimpleNamespace(
            access_token="old-token",
            refresh_token="old-refresh",
            client_id="cid",
            client_secret="secret",
            account_base_url="https://educent.amocrm.ru",
            account_subdomain="educent",
            authorized_at=None,
            expires_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
            last_error=None,
            contact_field_catalog=[],
            contact_field_catalog_synced_at=None,
            status="awaiting_callback",
        )
        with patch.object(amo_integration, "get_active_connection", return_value=connection), patch.object(
            amo_integration, "build_external_oauth_setup", return_value={}
        ), patch.object(amo_integration, "fetch_lead_field_catalog") as fetch_mock:
            payload = amo_integration.get_amo_connection_status(SimpleNamespace())

        fetch_mock.assert_not_called()
        self.assertFalse(payload["connected"])
        self.assertEqual(payload["status"], "awaiting_callback")
        self.assertEqual(payload["lead_field_sync_error"], "skipped: amoCRM connection is not active")

    def test_fetch_recent_leads_applies_closed_from_filter(self) -> None:
        with patch.object(
            amo_integration,
            "resolve_amo_access_context",
            return_value=amo_integration.AmoAccessContext(
                account_base_url="https://educent.amocrm.ru",
                access_token="token",
                token_source="oauth",
                connection=None,
            ),
        ), patch.object(
            amo_integration,
            "_paged_embedded_items",
            return_value=[],
        ) as paged_mock:
            amo_integration.fetch_recent_leads(
                None,  # type: ignore[arg-type]
                closed_from_ts=1773542400,
                limit_per_page=25,
            )

        initial_url = paged_mock.call_args.kwargs["initial_url"]
        self.assertIn("filter%5Bclosed_at%5D%5Bfrom%5D=1773542400", initial_url)
        self.assertIn("limit=25", initial_url)
        self.assertIn("with=contacts", initial_url)

    def test_active_client_loss_reason_is_treated_as_closed_valid(self) -> None:
        phone_context = PhoneContext(
            phone="+79990001122",
            source_dir="",
            contact_row={},
            call_rows=[],
            call_ids=[],
            first_call_at="2026-04-10 10:00:00",
            last_call_at="2026-04-10 10:00:00",
            manager_history=["Иванов Иван"],
            interest_summary="",
            objections_summary="",
            current_sales_temperature="warm",
            recommended_next_step="",
            follow_up_due_at="",
            history_summary="Клиент продолжает учиться.",
            chronology="",
            tallanto_id="",
            tallanto_match_status="",
        )
        candidate = deals_module.LeadCandidate(
            contact_id=123,
            lead_id=456,
            score=95,
            confidence=0.95,
            reason="test",
            lead={},
        )
        lead = {
            "id": 456,
            "name": "Сделка 456",
            "pipeline_id": 8938034,
            "status_id": 143,
            "responsible_user_id": 1,
            "created_at": "2026-04-01 10:00:00",
            "updated_at": "2026-04-10 10:00:00",
            "closed_at": "2026-04-10 10:00:00",
            "closest_task_at": None,
            "custom_fields_values": [
                {
                    "field_name": "Причина отказа (лид)",
                    "values": [{"value": "Действующий клиент"}],
                }
            ],
        }
        pipelines = [
            {"id": 8938034, "name": "Лиды", "statuses": [{"id": 143, "name": "Закрыто и не реализовано"}]}
        ]
        users = [{"id": 1, "name": "Иванов Иван"}]

        analysis = deals_module._analysis_from_selected_lead(
            SimpleNamespace(),
            phone_context=phone_context,
            candidate=candidate,
            contact={"id": 123},
            pipelines=pipelines,
            users=users,
            lead=lead,
            notes=[],
            tasks=[],
        )

        self.assertEqual(analysis["close_verdict"], "closed_valid")
        self.assertEqual(analysis["premature_close_risk"], "no_risk")
        self.assertIn("Действующий клиент", analysis["close_reason_summary"])

    def test_amo_http_request_retries_transient_url_errors(self) -> None:
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b'{"ok": true}'

        calls = {"count": 0}

        def _urlopen(*_args, **_kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                raise url_error.URLError("temporary handshake timeout")
            return _Response()

        with patch.object(amo_integration.url_request, "urlopen", side_effect=_urlopen), patch.object(
            amo_integration.time, "sleep", return_value=None
        ):
            payload = amo_integration._amo_http_request(method="GET", url="https://educent.amocrm.ru/api/v4/account")

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(calls["count"], 3)

    def test_amo_http_request_uses_configured_timeout_by_default(self) -> None:
        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                return b'{"ok": true}'

        seen: dict[str, object] = {}

        def _urlopen(*_args, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            return _Response()

        patched_settings = dataclasses.replace(
            amo_integration.settings,
            crm_amo_http_timeout_seconds=7,
        )

        with patch.object(amo_integration.url_request, "urlopen", side_effect=_urlopen), patch.object(
            amo_integration,
            "settings",
            patched_settings,
        ):
            payload = amo_integration._amo_http_request(method="GET", url="https://educent.amocrm.ru/api/v4/account")

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(seen["timeout"], 7)


if __name__ == "__main__":
    unittest.main()
