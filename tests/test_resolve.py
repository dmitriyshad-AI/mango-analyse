from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from sqlalchemy import text as sa_text

from mango_mvp.customer_timeline.calls_two_processes import (
    CallsTwoProcessesConfig,
    worker_environment,
)
from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.services import resolve as resolve_module
from mango_mvp.services.dialogue_contract import (
    PROVIDER_EVIDENCE_FIELD,
    build_dialogue_input,
)
from mango_mvp.services.resolve import ResolveService
from mango_mvp.services.transcribe import TranscribeService
from tests.test_dialogue_format import make_settings


class ResolveServiceTest(unittest.TestCase):
    def test_merge_pair_with_codex_uses_response_cache_on_repeat(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_cache_") as td:
            service = ResolveService(
                replace(
                    make_settings(),
                    resolve_llm_provider="codex_cli",
                    codex_resolve_model="gpt-5.4",
                    llm_cache_enabled=True,
                    llm_cache_dir=str(Path(td) / "llm-cache"),
                )
            )
            payload = {
                "merged_text": "Здравствуйте, расскажите подробнее.",
                "selection": "MIX",
                "confidence": 0.91,
                "notes": "merged",
            }
            state = {"calls": 0}

            def fake_run(cmd, capture_output, text, check, timeout):
                state["calls"] += 1
                self.assertIn("--model", cmd)
                self.assertIn("gpt-5.4", cmd)
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return CompletedProcess(cmd, 0, stdout="", stderr="tokens used\n710\n")

            with patch("mango_mvp.services.resolve.shutil.which", return_value="/usr/bin/codex"):
                with patch("mango_mvp.services.resolve.subprocess.run", side_effect=fake_run):
                    first = service._merge_pair_with_llm(
                        speaker_label="Менеджер",
                        variant_a="Здравствуйте, расскажите.",
                        variant_b="Здравствуйте, расскажите подробнее.",
                        context="",
                    )
                    second = service._merge_pair_with_llm(
                        speaker_label="Менеджер",
                        variant_a="Здравствуйте, расскажите.",
                        variant_b="Здравствуйте, расскажите подробнее.",
                        context="",
                    )

            self.assertEqual(state["calls"], 1)
            self.assertEqual(first["merged_text"], second["merged_text"])
            self.assertEqual(first["selection"], second["selection"])
            self.assertEqual(first.get("tokens_used_actual"), 710)

    def test_timed_line_parser_accepts_hours_and_rejects_invalid_seconds(self) -> None:
        service = ResolveService(make_settings())
        parsed = service._parse_timed_line("[01:00:00.0] Клиент: Продолжаем разговор.")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["ts_sec"], 3600.0)
        self.assertEqual(service._parse_timed_line("[100:00:00.0] Клиент: Запись.")["ts_sec"], 360000.0)
        self.assertIsNone(service._parse_timed_line("[00:99.9] Клиент: Ошибка часов."))
        self.assertIsNone(service._parse_timed_line("[01:99:00.0] Клиент: Ошибка минут."))

    def test_physical_track_lines_remain_distinct_without_inventing_roles(self) -> None:
        service = ResolveService(make_settings())
        lines = [
            "[00:01.0] Дорожка левая: Первый вариант.",
            "[00:01.0] Дорожка правая: Второй вариант.",
        ]

        rows = service._parse_dialogue_lines(
            CallRecord(source_file="a.mp3", source_filename="a.mp3"), lines
        )

        self.assertEqual([row[1] for row in rows], ["channel_left", "channel_right"])
        self.assertEqual(
            service._line_metrics(rows)["same_ts_cross_speaker_events"], 1
        )

    def test_proven_physical_channels_drive_role_quality_metrics(self) -> None:
        service = ResolveService(make_settings())
        lines = [
            "[00:01.0] Дорожка левая: Отправлю договор.",
            "[00:02.0] Дорожка правая: Хорошо, спасибо.",
        ]
        call = CallRecord(
            source_file="a.mp3",
            source_filename="a.mp3",
            transcript_manager="Отправлю договор.",
            transcript_client="Хорошо, спасибо.",
            transcript_variants_json=json.dumps(
                {
                    "dialogue_lines": lines,
                    "manager": {"physical_channel": "left"},
                    "client": {"physical_channel": "right"},
                },
                ensure_ascii=False,
            ),
        )

        self.assertEqual(service._load_dialogue_lines_from_export(call), lines)
        rows = service._parse_dialogue_lines(call, lines)
        self.assertEqual([row[1] for row in rows], ["manager", "client"])
        self.assertEqual(service._line_metrics(rows)["max_same_speaker_run"], 1)

    def test_dialogue_lines_are_loaded_from_variants_before_export_file(self) -> None:
        lines = ["[00:01.0] Менеджер (Иван): Добрый день.", "[00:02.0] Клиент: Здравствуйте."]
        call = CallRecord(
            source_file="calls/a.mp3",
            source_filename="a.mp3",
            transcript_manager="Добрый день.",
            transcript_client="Здравствуйте.",
            transcript_variants_json=json.dumps({"dialogue_lines": lines}, ensure_ascii=False),
        )
        self.assertEqual(ResolveService(make_settings())._load_dialogue_lines_from_export(call), lines)

    def test_dialogue_export_file_must_match_stored_role_texts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_sidecar_") as td:
            export_dir = Path(td) / "transcripts"
            path = export_dir / "calls" / "a_text.txt"
            path.parent.mkdir(parents=True)
            call = CallRecord(
                source_file="calls/a.mp3",
                source_filename="a.mp3",
                transcript_manager="Оплата подтверждена.",
                transcript_client="Ждём договор.",
                transcript_variants_json="{}",
            )
            service = ResolveService(replace(make_settings(), transcript_export_dir=str(export_dir)))
            valid = ["[00:01.0] Менеджер: Оплата подтверждена.", "[00:02.0] Клиент: Ждём договор."]
            path.write_text("\n".join(valid) + "\n", encoding="utf-8")
            self.assertEqual(service._load_dialogue_lines_from_export(call), valid)

            path.write_text("[00:01.0] Менеджер: Ждём договор.\n[00:02.0] Клиент: Оплата подтверждена.\n", encoding="utf-8")
            self.assertEqual(service._load_dialogue_lines_from_export(call), [])

    def test_dialogue_resolve_rejects_partially_parsed_lines(self) -> None:
        service = ResolveService(make_settings())
        call = CallRecord(source_file="a.mp3", source_filename="a.mp3")
        payload = {
            "mode": "stereo",
            "manager": {"physical_channel": "left"},
            "client": {"physical_channel": "right"},
        }
        lines = ["[00:01.0] Менеджер: Текст.", "[00:02.0 Клиент: Важная реплика."]
        self.assertIsNone(service._build_dialogue_resolve_payload(call, payload, lines))

    def test_dialogue_resolve_rejects_an_empty_direct_line(self) -> None:
        service = ResolveService(make_settings())
        call = CallRecord(source_file="a.mp3", source_filename="a.mp3")

        self.assertEqual(
            service._parse_dialogue_lines(
                call, ["[00:01.0] Дорожка левая: Текст.", ""]
            ),
            [],
        )

    def test_dialogue_resolve_from_sidecar_revokes_manager_quality(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_sidecar_quality_") as td:
            export_dir = Path(td) / "transcripts"
            path = export_dir / "calls" / "a_text.txt"
            path.parent.mkdir(parents=True)
            path.write_text(
                "[00:01.0] Менеджер: Оплата подтверждена.\n"
                "[00:02.0] Клиент: Ждём договор.\n",
                encoding="utf-8",
            )
            payload = {
                "mode": "stereo",
                "call_topology": "simple_two_party",
                "role_mapping": {
                    "confirmed": True,
                    "manager_quality_allowed": True,
                    "topology": "simple_two_party",
                },
                "manager": {"physical_channel": "left"},
                "client": {"physical_channel": "right"},
            }
            call = CallRecord(
                source_file="calls/a.mp3",
                source_filename="a.mp3",
                transcript_manager="Оплата подтверждена.",
                transcript_client="Ждём договор.",
                transcript_variants_json=json.dumps(payload, ensure_ascii=False),
            )
            service = ResolveService(replace(
                make_settings(),
                transcript_export_dir=str(export_dir),
                resolve_llm_provider="codex_cli",
            ))
            service._run_dialogue_llm = lambda request: {
                "turns": [
                    {
                        "turn_id": turn["turn_id"],
                        "speaker": turn["speaker"],
                        "final_text": turn["baseline_text"],
                    }
                    for turn in request["turns"]
                ]
            }

            candidate = service._resolve_dialogue_with_llm(call, payload)

            self.assertIsNotNone(candidate)
            mapping = json.loads(candidate["transcript_variants_json"])["role_mapping"]
            self.assertFalse(mapping["confirmed"])
            self.assertFalse(mapping["manager_quality_allowed"])
            self.assertEqual(mapping["status"], "mutable_sidecar_timing")

    def test_postfilter_persists_final_lines_in_variants(self) -> None:
        service = ResolveService(make_settings())
        call = CallRecord(source_file="a.mp3", source_filename="a.mp3")
        candidate = {
            "name": "baseline",
            "dialogue_lines": [
                "[00:10.0] Менеджер (Иван): Добрый день.",
                "[00:10.0] Клиент: Здравствуйте.",
            ],
            "transcript_variants_json": json.dumps({"mode": "stereo"}),
        }
        result = service._maybe_postfilter_candidate_dialogue(call, candidate)
        stored = json.loads(result["transcript_variants_json"])["dialogue_lines"]
        self.assertEqual(stored, result["dialogue_lines"])
        self.assertIn("[00:10.0] Клиент:", stored[1])

    def test_postfilter_keeps_mutable_sidecar_provenance_and_revokes_roles(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_sidecar_provenance_") as td:
            export_dir = Path(td) / "transcripts"
            path = export_dir / "calls" / "a_text.txt"
            path.parent.mkdir(parents=True)
            path.write_text(
                "[00:01.0] Менеджер: Первый ответ.\n[00:20.0] Клиент: Первый вопрос.\n",
                encoding="utf-8",
            )
            payload = {
                "mode": "stereo",
                "call_topology": "simple_two_party",
                "role_mapping": {
                    "confirmed": True,
                    "manager_quality_allowed": True,
                    "topology": "simple_two_party",
                },
                "manager": {"physical_channel": "left"},
                "client": {"physical_channel": "right"},
            }
            call = CallRecord(
                source_file="calls/a.mp3",
                source_filename="a.mp3",
                transcript_manager="Первый ответ.",
                transcript_client="Первый вопрос.",
                transcript_variants_json=json.dumps(payload, ensure_ascii=False),
            )
            service = ResolveService(replace(make_settings(), transcript_export_dir=str(export_dir)))

            result = service._maybe_postfilter_candidate_dialogue(
                call, service._candidate_from_call(call)
            )

            stored = json.loads(result["transcript_variants_json"])
            self.assertEqual(stored["dialogue_lines_source"], "mutable_sidecar")
            self.assertFalse(stored["role_mapping"]["confirmed"])
            self.assertFalse(stored["role_mapping"]["manager_quality_allowed"])

            call.transcript_variants_json = result["transcript_variants_json"]
            repeated = service._maybe_postfilter_candidate_dialogue(
                call, service._candidate_from_call(call)
            )
            repeated_payload = json.loads(repeated["transcript_variants_json"])
            self.assertEqual(repeated_payload["dialogue_lines_source"], "mutable_sidecar")
            self.assertFalse(repeated_payload["role_mapping"]["confirmed"])

    def test_mutated_sidecar_invalidates_a_prepared_candidate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_mutated_sidecar_") as td:
            export_dir = Path(td) / "transcripts"
            path = export_dir / "calls" / "a_text.txt"
            path.parent.mkdir(parents=True)
            original = [
                "[00:01.0] Менеджер: Первый ответ.",
                "[00:02.0] Клиент: Первый вопрос.",
            ]
            path.write_text("\n".join(original) + "\n", encoding="utf-8")
            call = CallRecord(
                source_file="calls/a.mp3",
                source_filename="a.mp3",
                transcript_manager="Первый ответ.",
                transcript_client="Первый вопрос.",
                transcript_variants_json="{}",
            )
            service = ResolveService(
                replace(make_settings(), transcript_export_dir=str(export_dir))
            )
            candidate = {
                "meta": {
                    "source_artifact_sha256": service._dialogue_lines_sha256(original)
                }
            }

            self.assertTrue(service._candidate_source_is_current(call, candidate))
            path.write_text(
                "[00:01.0] Менеджер: Подменённый ответ.\n"
                "[00:02.0] Клиент: Первый вопрос.\n",
                encoding="utf-8",
            )
            self.assertFalse(service._candidate_source_is_current(call, candidate))

    def test_postfilter_persists_lines_when_no_adjustment_is_needed(self) -> None:
        service = ResolveService(make_settings())
        lines = ["[00:01.0] Менеджер (Иван): Добрый день."]
        result = service._maybe_postfilter_candidate_dialogue(
            CallRecord(source_file="a.mp3", source_filename="a.mp3"),
            {
                "name": "baseline",
                "dialogue_lines": lines,
                "transcript_variants_json": json.dumps({"mode": "stereo"}),
            },
        )
        self.assertEqual(json.loads(result["transcript_variants_json"])["dialogue_lines"], lines)

    def test_dialogue_llm_cannot_swap_exact_turns_or_known_roles(self) -> None:
        service = ResolveService(make_settings())
        input_payload = {
            "turns": [
                {"turn_id": 1, "ts_sec": 10.0, "speaker": "manager", "baseline_text": "Вопрос", "approximate": False, "flags": ["same_ts_cross"]},
                {"turn_id": 2, "ts_sec": 10.0, "speaker": "client", "baseline_text": "Ответ", "approximate": False, "flags": ["same_ts_cross"]},
            ],
            "role_variants": {},
        }
        llm_payload = {
            "turns": [
                {"turn_id": 1, "speaker": "client", "final_text": "Вопрос", "swap_with_next": True},
                {"turn_id": 2, "speaker": "manager", "final_text": "Ответ", "swap_with_next": False},
            ]
        }
        normalized = service._normalize_dialogue_result(input_payload, llm_payload)
        self.assertEqual([turn["turn_id"] for turn in normalized["turns"]], [1, 2])
        self.assertEqual([turn["speaker"] for turn in normalized["turns"]], ["manager", "client"])
        self.assertEqual(normalized["swaps_applied"], 0)

        variants = {
            "mode": "stereo",
            "role_mapping": {"confirmed": False, "status": "unverified_legacy_channel_order"},
            "manager": {"physical_channel": "left", "variant_a_segments": [{"start": 10.0, "text": "Вопрос"}], "variant_b_segments": [{"start": 10.0, "text": "Вопрос."}]},
            "client": {"physical_channel": "right", "variant_a_segments": [{"start": 10.0, "text": "Ответ"}], "variant_b_segments": [{"start": 10.0, "text": "Ответ."}]},
        }
        candidate = service._dialogue_turns_to_candidate(
            CallRecord(source_file="a.mp3", source_filename="a.mp3", manager_name="Иван"),
            variants,
            normalized,
            provider="test",
        )
        stored = json.loads(candidate["transcript_variants_json"])
        self.assertEqual(candidate["transcript_manager"], "Вопрос")
        self.assertEqual(candidate["transcript_client"], "Ответ")
        self.assertEqual(
            candidate["dialogue_lines"],
            [
                "[00:10.0] Дорожка левая: Вопрос",
                "[00:10.0] Дорожка правая: Ответ",
            ],
        )
        self.assertNotIn("Менеджер", "\n".join(candidate["dialogue_lines"]))
        self.assertNotIn("Клиент", "\n".join(candidate["dialogue_lines"]))
        self.assertEqual(stored["manager"]["physical_channel"], "left")
        self.assertEqual(stored["client"]["physical_channel"], "right")
        self.assertEqual(len(stored["manager"]["variant_b_segments"]), 1)
        self.assertEqual(len(stored["client"]["variant_b_segments"]), 1)
        self.assertFalse(stored["role_mapping"]["confirmed"])

    def test_model_speaker_correction_revokes_confirmed_role_mapping(self) -> None:
        service = ResolveService(make_settings())
        normalized = service._normalize_dialogue_result(
            {
                "turns": [{"turn_id": 1, "ts_sec": 10.0, "speaker": "unknown", "baseline_text": "Вопрос", "approximate": False}],
                "role_variants": {},
            },
            {"turns": [{"turn_id": 1, "speaker": "manager", "final_text": "Вопрос"}]},
        )
        candidate = service._dialogue_turns_to_candidate(
            CallRecord(source_file="a.mp3", source_filename="a.mp3", manager_name="Иван"),
            {
                "mode": "stereo",
                "call_topology": "simple_two_party",
                "role_mapping": {"confirmed": True, "manager_quality_allowed": True, "topology": "simple_two_party"},
                "manager": {"physical_channel": "left"},
                "client": {"physical_channel": "right"},
            },
            normalized,
            provider="test",
        )
        mapping = json.loads(candidate["transcript_variants_json"])["role_mapping"]
        self.assertFalse(mapping["confirmed"])
        self.assertFalse(mapping["manager_quality_allowed"])

    def test_short_calls_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_skip_") as td:
            db_path = Path(td) / "resolve.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                resolve_min_duration_sec=30,
                resolve_llm_provider="off",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(Path(td) / "a.mp3"),
                        source_filename="a.mp3",
                        duration_sec=12.0,
                        transcription_status="done",
                        resolve_status="pending",
                        analysis_status="pending",
                        transcript_text="MANAGER:\nЗдравствуйте\n\nCLIENT:\nДобрый день",
                        transcript_manager="Здравствуйте",
                        transcript_client="Добрый день",
                        transcript_variants_json=json.dumps(
                            {
                                "mode": "stereo",
                                "warnings": [],
                                "manager": {"variant_a": "Здравствуйте", "variant_b": "", "final": "Здравствуйте"},
                                "client": {"variant_a": "Добрый день", "variant_b": "", "final": "Добрый день"},
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
                session.commit()

            service = ResolveService(settings)
            with session_factory() as session:
                result = service.run(session, limit=10)
            self.assertEqual(result["processed"], 1)
            self.assertEqual(result["skipped_short"], 1)
            self.assertEqual(result["success"], 1)

            with session_factory() as session:
                call = session.query(CallRecord).first()
                assert call is not None
                self.assertEqual(call.resolve_status, "skipped")
                self.assertEqual(call.resolve_quality_score, 100.0)
                self.assertIsNotNone(call.resolve_json)

    def test_export_manual_review_queue(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_queue_") as td:
            db_path = Path(td) / "queue.db"
            out_path = Path(td) / "manual_queue.csv"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                resolve_llm_provider="off",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(Path(td) / "b.mp3"),
                        source_filename="b.mp3",
                        duration_sec=180.0,
                        transcription_status="done",
                        resolve_status="manual",
                        analysis_status="pending",
                        resolve_quality_score=52.0,
                        resolve_json=json.dumps(
                            {
                                "decision": "manual_review_required",
                                "chosen": {"name": "baseline", "score": 52, "reasons": ["same_ts_cross=3"]},
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
                session.commit()

            service = ResolveService(settings)
            with session_factory() as session:
                result = service.export_manual_review_queue(session, out_path=out_path, limit=100)
            self.assertEqual(result["exported"], 1)
            self.assertTrue(out_path.exists())
            content = out_path.read_text(encoding="utf-8")
            self.assertIn("manual_review_required", content)
            self.assertIn("same_ts_cross=3", content)

    def test_export_failed_resolve_queue(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_failed_queue_") as td:
            db_path = Path(td) / "failed_queue.db"
            out_path = Path(td) / "failed_resolve_queue.csv"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(Path(td) / "c.mp3"),
                        source_filename="c.mp3",
                        transcription_status="done",
                        resolve_status="failed",
                        analysis_status="pending",
                        resolve_attempts=1,
                        last_error="resolve: test failure",
                    )
                )
                session.commit()

            service = ResolveService(settings)
            with session_factory() as session:
                result = service.export_failed_resolve_queue(session, out_path=out_path, limit=100)
            self.assertEqual(result["exported"], 1)
            content = out_path.read_text(encoding="utf-8")
            self.assertIn("failed", content)
            self.assertIn("resolve: test failure", content)

    def test_aggressive_rescue_runs_for_risky_ordering(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_risky_") as td:
            db_path = Path(td) / "resolve_risky.db"
            export_dir = Path(td) / "transcripts"
            source_dir = Path(td) / "calls"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_file = source_dir / "risky.mp3"
            source_file.write_bytes(b"")

            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                transcript_export_dir=str(export_dir),
                resolve_llm_provider="off",
                resolve_accept_score=0,
                resolve_aggressive_rescue_for_risky=True,
                resolve_risky_same_ts_threshold=1,
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(source_file),
                        source_filename=source_file.name,
                        duration_sec=180.0,
                        transcription_status="done",
                        resolve_status="pending",
                        analysis_status="pending",
                        transcript_text="MANAGER:\nДобрый день\n\nCLIENT:\nЗдравствуйте",
                        transcript_manager="Добрый день",
                        transcript_client="Здравствуйте",
                        transcript_variants_json=json.dumps(
                            {
                                "mode": "stereo",
                                "warnings": [],
                                "manager": {"variant_a": "Добрый день", "variant_b": "", "final": "Добрый день"},
                                "client": {"variant_a": "Здравствуйте", "variant_b": "", "final": "Здравствуйте"},
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
                session.commit()

            target_dir = export_dir / source_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "risky_text.txt").write_text(
                "\n".join(
                    [
                        "[00:10.0] Менеджер (Иван): Добрый день.",
                        "[00:10.0] Клиент: Здравствуйте.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ResolveService(settings)
            rescue_called = {"value": False}

            def fake_rescue(_: CallRecord) -> dict:
                rescue_called["value"] = True
                return {
                    "name": "rescue",
                    "transcript_manager": "Добрый день",
                    "transcript_client": "Здравствуйте",
                    "transcript_text": "MANAGER:\nДобрый день\n\nCLIENT:\nЗдравствуйте",
                    "dialogue_lines": [
                        "[00:10.0] Менеджер (Иван): Добрый день.",
                        "[00:10.2] Клиент: Здравствуйте.",
                    ],
                    "transcript_variants_json": json.dumps(
                        {
                            "mode": "stereo",
                            "warnings": [],
                        },
                        ensure_ascii=False,
                    ),
                    "meta": {"provider": "fake_rescue"},
                }

            service._run_rescue_asr = fake_rescue  # type: ignore[method-assign]
            with session_factory() as session:
                result = service.run(session, limit=10)

            self.assertEqual(result["processed"], 1)
            self.assertEqual(result["failed"], 0)
            self.assertTrue(rescue_called["value"])
            self.assertEqual(result["rescue_used"], 1)

    def test_rescue_provider_none_disables_rescue_asr(self) -> None:
        service = ResolveService(
            replace(
                make_settings(),
                resolve_rescue_provider="none",
            )
        )

        self.assertEqual(service._rescue_provider(), "")
        self.assertIsNone(service._run_rescue_asr(CallRecord(source_file="a.mp3", source_filename="a.mp3")))

    def test_rescue_asr_cannot_replace_independent_provider_evidence(self) -> None:
        settings = replace(
            make_settings(),
            resolve_rescue_provider="mlx",
            resolve_rescue_dual_enabled=False,
        )
        service = ResolveService(settings)
        original_evidence = {"recording_id": "recording-original", "tracks": []}
        call = CallRecord(
            source_file="a.mp3",
            source_filename="a.mp3",
            source_recording_id="recording-original",
            transcript_variants_json=json.dumps(
                {
                    PROVIDER_EVIDENCE_FIELD: original_evidence,
                    "provider_capture_manifest_sha256": "a" * 64,
                },
                ensure_ascii=False,
            ),
        )

        class FakeRescue:
            @staticmethod
            def _transcribe_call(_call):
                return {
                    "transcript_variants_json": json.dumps(
                        {
                            PROVIDER_EVIDENCE_FIELD: {
                                "recording_id": "recording-rebound",
                                "tracks": [],
                            }
                        },
                        ensure_ascii=False,
                    )
                }

        service._rescue_service_cache[("mlx", False)] = FakeRescue()
        result = service._run_rescue_asr(call)
        payload = json.loads(result["transcript_variants_json"])

        self.assertEqual(payload[PROVIDER_EVIDENCE_FIELD], original_evidence)
        self.assertEqual(payload["provider_capture_manifest_sha256"], "a" * 64)

    def test_rescue_asr_cannot_introduce_provider_evidence(self) -> None:
        settings = replace(
            make_settings(),
            resolve_rescue_provider="mlx",
            resolve_rescue_dual_enabled=False,
        )
        service = ResolveService(settings)
        call = CallRecord(
            source_file="a.mp3",
            source_filename="a.mp3",
            transcript_variants_json="{}",
        )

        class FakeRescue:
            @staticmethod
            def _transcribe_call(_call):
                return {
                    "transcript_variants_json": json.dumps(
                        {
                            PROVIDER_EVIDENCE_FIELD: {
                                "recording_id": "recording-injected",
                            },
                            "provider_capture_manifest_sha256": "b" * 64,
                        }
                    )
                }

        service._rescue_service_cache[("mlx", False)] = FakeRescue()
        result = service._run_rescue_asr(call)
        payload = json.loads(result["transcript_variants_json"])

        self.assertNotIn(PROVIDER_EVIDENCE_FIELD, payload)
        self.assertNotIn("provider_capture_manifest_sha256", payload)

    def test_llm_runs_for_risky_ordering_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_llm_risky_") as td:
            db_path = Path(td) / "resolve_llm_risky.db"
            export_dir = Path(td) / "transcripts"
            source_dir = Path(td) / "calls"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_file = source_dir / "risky_llm.mp3"
            source_file.write_bytes(b"")

            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                transcript_export_dir=str(export_dir),
                resolve_llm_provider="codex_cli",
                resolve_llm_for_risky=True,
                resolve_llm_trigger_score=75,
                resolve_accept_score=50,
                resolve_risky_same_ts_threshold=1,
                resolve_aggressive_rescue_for_risky=False,
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(source_file),
                        source_filename=source_file.name,
                        duration_sec=180.0,
                        transcription_status="done",
                        resolve_status="pending",
                        analysis_status="pending",
                        transcript_text="MANAGER:\nДобрый день\n\nCLIENT:\nЗдравствуйте",
                        transcript_manager="Добрый день",
                        transcript_client="Здравствуйте",
                        transcript_variants_json=json.dumps(
                            {
                                "mode": "stereo",
                                "warnings": [],
                                "manager": {"variant_a": "Добрый день", "variant_b": "Добрый день", "final": "Добрый день"},
                                "client": {"variant_a": "Здравствуйте", "variant_b": "Здравствуйте", "final": "Здравствуйте"},
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
                session.commit()

            target_dir = export_dir / source_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "risky_llm_text.txt").write_text(
                "\n".join(
                    [
                        "[00:10.0] Менеджер (Иван): Добрый день.",
                        "[00:10.0] Клиент: Здравствуйте.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            service = ResolveService(settings)
            llm_called = {"value": False}

            def fake_llm(_: CallRecord, payload: dict) -> dict:
                llm_called["value"] = True
                return {
                    "name": "llm",
                    "transcript_manager": "Добрый день",
                    "transcript_client": "Здравствуйте",
                    "transcript_text": "MANAGER:\nДобрый день\n\nCLIENT:\nЗдравствуйте",
                    "dialogue_lines": [
                        "[00:10.0] Менеджер (Иван): Добрый день.",
                        "[00:10.1] Клиент: Здравствуйте.",
                    ],
                    "transcript_variants_json": json.dumps(payload, ensure_ascii=False),
                    "meta": {"provider": "fake_llm"},
                }

            service._resolve_with_llm = fake_llm  # type: ignore[method-assign]

            with session_factory() as session:
                result = service.run(session, limit=10)

            self.assertEqual(result["processed"], 1)
            self.assertEqual(result["failed"], 0)
            self.assertTrue(llm_called["value"])
            self.assertEqual(result["llm_used"], 1)

    def test_resolve_llm_provider_off_returns_none_immediately(self) -> None:
        service = ResolveService(replace(make_settings(), resolve_llm_provider="off"))
        call = CallRecord(
            source_file="a.mp3",
            source_filename="a.mp3",
            transcript_text="MANAGER:\nДобрый день\n\nCLIENT:\nЗдравствуйте",
            transcript_manager="Добрый день",
            transcript_client="Здравствуйте",
        )
        payload = {
            "mode": "mono_or_fallback",
            "full": {
                "variant_a": "Добрый день. Здравствуйте.",
                "variant_b": "Добрый день, здравствуйте.",
                "final": "Добрый день. Здравствуйте.",
            },
        }

        self.assertIsNone(service._resolve_with_llm(call, payload))

    def test_resolve_llm_off_does_not_increment_llm_used(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_llm_off_") as td:
            db_path = Path(td) / "resolve_llm_off.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                resolve_llm_provider="off",
                resolve_llm_trigger_score=101,
                resolve_accept_score=0,
                resolve_rescue_provider="off",
                resolve_aggressive_rescue_for_risky=False,
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(Path(td) / "off.mp3"),
                        source_filename="off.mp3",
                        duration_sec=180.0,
                        transcription_status="done",
                        resolve_status="pending",
                        analysis_status="pending",
                        transcript_text="MANAGER:\nДобрый день\n\nCLIENT:\nЗдравствуйте",
                        transcript_manager="Добрый день",
                        transcript_client="Здравствуйте",
                        transcript_variants_json=json.dumps(
                            {
                                "mode": "stereo",
                                "warnings": [],
                                "manager": {
                                    "variant_a": "Добрый день",
                                    "variant_b": "Добрый, день",
                                    "final": "Добрый день",
                                },
                                "client": {
                                    "variant_a": "Здравствуйте",
                                    "variant_b": "Здравствуйте",
                                    "final": "Здравствуйте",
                                },
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
                session.commit()

            service = ResolveService(settings)
            with session_factory() as session:
                result = service.run(session, limit=10)

            self.assertEqual(result["processed"], 1)
            self.assertEqual(result["llm_used"], 0)
            self.assertEqual(result["failed"], 0)

    def test_worker_environment_does_not_switch_the_resolve_llm_on(self) -> None:
        """M1 regression: the pipeline env forced ``codex_cli`` on every stage."""
        with tempfile.TemporaryDirectory(prefix="mango_worker_env_") as td:
            root = Path(td)
            timeline_root = root / "timeline"
            timeline_root.mkdir()
            config = CallsTwoProcessesConfig(
                pipeline_root=root / "pipeline",
                timeline_db=timeline_root / "timeline.sqlite",
                timeline_allowed_root=timeline_root,
                python_executable=Path(sys.executable),
                codex_binary=Path(sys.executable),
                codex_home_root=root / "codex",
            )

            environment = worker_environment(config)

            self.assertEqual(environment["RESOLVE_LLM_PROVIDER"], "off")
            # Negative control: Analyse is still the stage that may call a model.
            self.assertEqual(environment["ANALYZE_PROVIDER"], "codex_cli")

    def test_resolve_llm_off_makes_zero_provider_subprocess_calls(self) -> None:
        """The counter, not the label: nothing may reach an LLM binary."""
        settings = replace(
            make_settings(),
            resolve_llm_provider="off",
            resolve_rescue_provider="off",
        )
        service = ResolveService(settings)
        call = CallRecord(
            source_file="a.mp3",
            source_filename="a.mp3",
            transcript_text="MANAGER:\nДобрый день\n\nCLIENT:\nЗдравствуйте",
            transcript_manager="Добрый день",
            transcript_client="Здравствуйте",
        )
        payload = {
            "mode": "stereo",
            "manager": {
                "variant_a": "Добрый день",
                "variant_b": "Добрый, день",
                "final": "Добрый день",
            },
            "client": {
                "variant_a": "Здравствуйте",
                "variant_b": "Здравствуйте!",
                "final": "Здравствуйте",
            },
        }
        calls = {"subprocess": 0}

        def counting_run(*_args, **_kwargs):
            calls["subprocess"] += 1
            raise AssertionError("the resolve LLM must not be launched")

        with patch.object(resolve_module.subprocess, "run", counting_run):
            merged = service._merge_pair_with_llm(
                speaker_label="Менеджер",
                variant_a="Клиент спрашивал про летний лагерь по математике",
                variant_b="Клиент интересовался годовым курсом физики зимой",
                context="",
            )
            self.assertIsNone(service._resolve_with_llm(call, payload))

        self.assertEqual(calls["subprocess"], 0)
        self.assertEqual(merged["provider"], "rule")
        self.assertEqual(merged["notes"], "resolve_llm_provider_off")

    def test_score_candidate_does_not_fallback_to_export_for_missing_dialogue_lines(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_no_export_fallback_") as td:
            export_dir = Path(td) / "transcripts"
            source_dir = Path(td) / "calls"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_file = source_dir / "score.mp3"
            source_file.write_bytes(b"")

            settings = replace(
                make_settings(),
                transcript_export_dir=str(export_dir),
            )
            service = ResolveService(settings)
            call = CallRecord(
                id=1,
                source_file=str(source_file),
                source_filename=source_file.name,
                duration_sec=180.0,
                transcript_text="MANAGER:\nДобрый день\n\nCLIENT:\nЗдравствуйте",
                transcript_manager="Добрый день",
                transcript_client="Здравствуйте",
            )

            target_dir = export_dir / source_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "score_text.txt").write_text(
                "\n".join(
                    [
                        "[00:10.0] Менеджер (Иван): Добрый день.",
                        "[00:10.0] Клиент: Здравствуйте.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            quality = service._score_candidate(
                call,
                call.transcript_text or "",
                call.transcript_manager,
                call.transcript_client,
                {"mode": "stereo", "warnings": []},
                dialogue_lines=None,
            )
            self.assertNotIn("same_ts_cross=1", quality["reasons"])
            self.assertEqual(
                int(quality.get("signals", {}).get("same_ts_cross_speaker_events", 0) or 0),
                0,
            )

    def test_resolve_with_llm_uses_dialogue_level_candidate_for_stereo(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_dialogue_level_") as td:
            export_dir = Path(td) / "transcripts"
            source_dir = Path(td) / "calls"
            source_dir.mkdir(parents=True, exist_ok=True)
            source_file = source_dir / "dialogue.mp3"
            source_file.write_bytes(b"")

            settings = replace(
                make_settings(),
                transcript_export_dir=str(export_dir),
                resolve_llm_provider="codex_cli",
            )
            service = ResolveService(settings)
            call = CallRecord(
                id=1,
                source_file=str(source_file),
                source_filename=source_file.name,
                manager_name="Иван",
                duration_sec=180.0,
                transcript_text="MANAGER:\nЗдравствуйте как вам удобно\n\nCLIENT:\nДа, слушаю хорошо",
                transcript_manager="Здравствуйте как вам удобно",
                transcript_client="Да, слушаю хорошо",
                transcript_variants_json=json.dumps(
                    {
                        "mode": "stereo",
                        "warnings": [],
                        "primary_provider": "mlx",
                        "secondary_provider": "gigaam",
                        "merge_provider": "codex_cli",
                        "role_mapping": {
                            "left": "manager",
                            "right": "client",
                            "confirmed": True,
                            "manager_quality_allowed": True,
                        },
                        "manager": {
                            "physical_channel": "left",
                            "variant_a": "Здравствуйте как вам удобно",
                            "variant_b": "Здравствуйте, как вам удобно",
                            "final": "Здравствуйте как вам удобно",
                        },
                        "client": {
                            "physical_channel": "right",
                            "variant_a": "Да, слушаю хорошо",
                            "variant_b": "Да, слушаю. Хорошо.",
                            "final": "Да, слушаю хорошо",
                        },
                    },
                    ensure_ascii=False,
                ),
            )

            target_dir = export_dir / source_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "dialogue_text.txt").write_text(
                "\n".join(
                    [
                        "[00:01.0] Дорожка левая: Здравствуйте как вам удобно",
                        "[00:01.0] Дорожка правая: Да, слушаю хорошо",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            def fake_dialogue_runner(payload: dict) -> dict:
                self.assertEqual(payload.get("schema_version"), "dialogue_resolve_v1")
                turns = payload.get("turns") or []
                self.assertEqual(len(turns), 2)
                return {
                    "schema_version": "dialogue_resolve_result_v1",
                    "turns": [
                        {
                            "turn_id": 1,
                            "speaker": "manager",
                            "final_text": "Здравствуйте, как вам удобно?",
                            "selection": "B",
                            "drop": False,
                            "swap_with_next": False,
                            "confidence": 0.9,
                            "notes": "",
                        },
                        {
                            "turn_id": 2,
                            "speaker": "client",
                            "final_text": "Да, слушаю. Хорошо.",
                            "selection": "B",
                            "drop": False,
                            "swap_with_next": False,
                            "confidence": 0.85,
                            "notes": "",
                        },
                    ],
                    "warnings": [],
                    "global_notes": "",
                    "_llm_meta": {
                        "llm_tokens_used_actual": 901,
                        "llm_duration_sec": 8.7,
                    },
                }

            service._run_dialogue_llm = fake_dialogue_runner  # type: ignore[method-assign]

            candidate = service._resolve_with_llm(
                call,
                json.loads(call.transcript_variants_json or "{}"),
            )

            self.assertIsNotNone(candidate)
            assert candidate is not None
            self.assertEqual(candidate["name"], "llm")
            self.assertEqual(candidate["meta"]["resolve_mode"], "dialogue_level")
            self.assertEqual(candidate["meta"]["llm_tokens_used_actual"], 901)
            self.assertEqual(candidate["meta"]["llm_duration_sec"], 8.7)
            self.assertEqual(
                candidate["dialogue_lines"],
                [
                    "[00:01.0] Дорожка левая: Здравствуйте, как вам удобно?",
                    "[00:01.0] Дорожка правая: Да, слушаю. Хорошо.",
                ],
            )
            self.assertIn("Здравствуйте, как вам удобно?", candidate["transcript_text"])
            self.assertIn("Да, слушаю. Хорошо.", candidate["transcript_text"])

    def test_openai_provider_without_key_falls_back_to_rule(self) -> None:
        settings = replace(
            make_settings(),
            openai_api_key=None,
            resolve_llm_provider="openai",
        )
        service = ResolveService(settings)
        merged = service._merge_pair_with_llm(
            speaker_label="Менеджер",
            variant_a="Добрый день",
            variant_b="Добрый, день",
            context="",
        )
        self.assertEqual(merged.get("provider"), "rule_fallback")
        self.assertIn("openai_failed", str(merged.get("notes", "")))
        self.assertTrue(str(merged.get("merged_text", "")).strip())

    def test_codex_provider_without_binary_falls_back_to_rule(self) -> None:
        settings = replace(
            make_settings(),
            resolve_llm_provider="codex_cli",
            codex_cli_command="codex",
        )
        service = ResolveService(settings)
        with patch("mango_mvp.services.resolve.shutil.which", return_value=None):
            merged = service._merge_pair_with_llm(
                speaker_label="Менеджер",
                variant_a="Добрый день",
                variant_b="Добрый, день",
                context="",
            )
        self.assertEqual(merged.get("provider"), "rule_fallback")
        self.assertIn("codex_cli_failed", str(merged.get("notes", "")))
        self.assertTrue(str(merged.get("merged_text", "")).strip())


class ResolveSharedContractTest(unittest.TestCase):
    """Этап B/E: one parser, and no model may move a turn to another side."""

    def test_resolve_uses_the_shared_line_parser(self) -> None:
        service = ResolveService(make_settings())
        parsed = service._parse_timed_line("[~00:05] Спикер (не определен): Текст")

        self.assertEqual(parsed["ts_sec"], 5.0)
        self.assertTrue(parsed["approximate"])
        self.assertEqual(parsed["role"], "unknown")
        self.assertEqual(parsed["text"], "Текст")
        physical = service._parse_timed_line("[00:01.0] Дорожка левая: Текст")
        self.assertEqual(physical["role"], "channel_left")
        self.assertIsNone(service._parse_timed_line("[00:01.0] Клиент:"))
        self.assertIsNone(service._parse_timed_line("сломанная строка"))

    def test_one_unparsable_line_drops_the_whole_dialogue_not_a_part_of_it(self) -> None:
        service = ResolveService(make_settings())
        call = CallRecord(source_file="a.mp3", source_filename="a.mp3")
        good = [
            "[00:01.0] Менеджер (Иван): Добрый день.",
            "[00:03.0] Клиент: Здравствуйте.",
        ]

        self.assertEqual(len(service._parse_dialogue_lines(call, good)), 2)
        self.assertEqual(
            service._parse_dialogue_lines(call, [good[0], "сломанная строка", good[1]]),
            [],
        )
        self.assertEqual(
            service._parse_dialogue_lines(call, [good[0], "[00:05.0] Клиент:"]), []
        )

    def test_model_speaker_correction_is_rejected_and_never_changes_a_side(self) -> None:
        service = ResolveService(make_settings())
        input_payload = {
            "turns": [
                {"turn_id": 1, "ts_sec": 1.0, "speaker": "unknown",
                 "baseline_text": "Первая", "approximate": False},
                {"turn_id": 2, "ts_sec": 2.0, "speaker": "manager",
                 "baseline_text": "Вторая", "approximate": False},
            ],
            "role_variants": {},
        }
        llm_payload = {
            "turns": [
                {"turn_id": 1, "speaker": "manager", "final_text": "Первая"},
                {"turn_id": 2, "speaker": "client", "final_text": "Вторая"},
            ]
        }

        normalized = service._normalize_dialogue_result(input_payload, llm_payload)

        self.assertEqual(
            [turn["speaker"] for turn in normalized["turns"]], ["unknown", "manager"]
        )
        self.assertEqual(normalized["speaker_corrections"], 0)
        self.assertEqual(normalized["speaker_corrections_rejected"], 2)
        self.assertIn("speaker_change_rejected:1", normalized["warnings"])
        self.assertIn("speaker_change_rejected:2", normalized["warnings"])

    def test_model_cannot_move_a_turn_between_physical_tracks(self) -> None:
        service = ResolveService(make_settings())
        normalized = service._normalize_dialogue_result(
            {
                "turns": [
                    {"turn_id": 1, "ts_sec": 1.0, "speaker": "channel_left",
                     "baseline_text": "Первая", "approximate": False}
                ],
                "role_variants": {},
            },
            {"turns": [
                {"turn_id": 1, "speaker": "channel_right", "final_text": "Первая"}
            ]},
        )
        candidate = service._dialogue_turns_to_candidate(
            CallRecord(source_file="a.mp3", source_filename="a.mp3"),
            {"mode": "stereo", "manager": {}, "client": {}},
            normalized,
            provider="test",
        )

        self.assertEqual(normalized["turns"][0]["speaker"], "channel_left")
        self.assertEqual(normalized["speaker_corrections_rejected"], 1)
        self.assertIn(
            "Дорожка левая: Первая", candidate["dialogue_lines"][0]
        )

    def test_a_rejected_correction_never_reaches_dialogue_lines(self) -> None:
        service = ResolveService(make_settings())
        normalized = service._normalize_dialogue_result(
            {
                "turns": [
                    {"turn_id": 1, "ts_sec": 1.0, "speaker": "unknown",
                     "baseline_text": "Первая", "approximate": False}
                ],
                "role_variants": {},
            },
            {"turns": [{"turn_id": 1, "speaker": "manager", "final_text": "Первая"}]},
        )
        candidate = service._dialogue_turns_to_candidate(
            CallRecord(source_file="a.mp3", source_filename="a.mp3", manager_name="Иван"),
            {
                "mode": "stereo",
                "role_mapping": {
                    "confirmed": True, "manager_quality_allowed": True,
                    "topology": "simple_two_party", "left": "manager", "right": "client",
                    "status": "confirmed_multi_signal",
                },
            },
            normalized,
            provider="test",
        )
        stored = json.loads(candidate["transcript_variants_json"])

        self.assertEqual(candidate["dialogue_lines"], ["[00:01.0] Спикер (не определен): Первая"])
        self.assertNotIn("Менеджер", candidate["dialogue_lines"][0])
        self.assertEqual(stored["dialogue_resolve"]["speaker_corrections"], 0)
        self.assertEqual(stored["dialogue_resolve"]["speaker_corrections_rejected"], 1)
        self.assertEqual(stored["role_mapping"]["status"], "model_speaker_correction")
        self.assertFalse(stored["role_mapping"]["confirmed"])

    def test_a_rejected_correction_keeps_the_call_untrusted_for_analyse(self) -> None:
        service = ResolveService(make_settings())
        normalized = service._normalize_dialogue_result(
            {
                "turns": [
                    {"turn_id": 1, "ts_sec": 1.0, "speaker": "unknown",
                     "baseline_text": "Первая", "approximate": False}
                ],
                "role_variants": {},
            },
            {"turns": [{"turn_id": 1, "speaker": "manager", "final_text": "Первая"}]},
        )
        candidate = service._dialogue_turns_to_candidate(
            CallRecord(source_file="a.mp3", source_filename="a.mp3", manager_name="Иван"),
            {
                "mode": "stereo",
                "role_mapping": {
                    "confirmed": True, "manager_quality_allowed": True,
                    "topology": "simple_two_party", "left": "manager", "right": "client",
                    "status": "confirmed_multi_signal",
                },
            },
            normalized,
            provider="test",
        )
        dialogue = build_dialogue_input(
            {
                "source_call_id": "call-7",
                "transcript_variants_json": candidate["transcript_variants_json"],
            }
        )

        self.assertFalse(dialogue.role_attribution["trusted"])
        self.assertNotIn("Менеджер", dialogue.render())


class ResolveOrderAndSpeakerGuardTest(unittest.TestCase):
    """Этап F: the model may propose, but chronology and sides never move."""

    def test_the_prompt_forbids_changing_the_speaker_or_the_order(self) -> None:
        from mango_mvp.services.resolve import DIALOGUE_RESOLVE_SYSTEM_PROMPT as prompt

        self.assertIn("You must NOT reorder turns", prompt)
        self.assertIn("You must NOT change speaker", prompt)
        self.assertIn("swap_with_next must always be false", prompt)
        # The old prompt invited exactly the two edits the runtime now rejects.
        self.assertNotIn("You may set swap_with_next=true only when", prompt)
        self.assertNotIn("Change speaker only if", prompt)

    def test_a_requested_swap_is_rejected_and_the_chronology_never_moves(self) -> None:
        service = ResolveService(make_settings())
        input_payload = {
            "turns": [
                {"turn_id": 1, "ts_sec": 10.0, "speaker": "manager",
                 "baseline_text": "Первая", "approximate": True},
                {"turn_id": 2, "ts_sec": 10.0, "speaker": "client",
                 "baseline_text": "Вторая", "approximate": True},
            ],
            "role_variants": {},
        }
        llm_payload = {
            "turns": [
                {"turn_id": 1, "final_text": "Первая", "swap_with_next": True},
                {"turn_id": 2, "final_text": "Вторая", "swap_with_next": False},
            ]
        }

        normalized = service._normalize_dialogue_result(input_payload, llm_payload)

        # Approximate timings used to be enough to let the model reorder.
        self.assertEqual([turn["turn_id"] for turn in normalized["turns"]], [1, 2])
        self.assertEqual(
            [turn["final_text"] for turn in normalized["turns"]], ["Первая", "Вторая"]
        )
        self.assertEqual(normalized["swaps_applied"], 0)
        self.assertEqual(normalized["swap_requests_rejected"], 1)
        self.assertIn("swap_rejected:1", normalized["warnings"])
        self.assertFalse(any(turn["swap_with_next"] for turn in normalized["turns"]))

    def test_a_malformed_line_invalidates_the_whole_dialogue_for_resolve(self) -> None:
        service = ResolveService(make_settings())
        call = CallRecord(source_file="a.mp3", source_filename="a.mp3")
        good = [
            "[00:01.0] Менеджер (Иван): Добрый день.",
            "[00:03.0] Клиент: Здравствуйте.",
        ]

        self.assertEqual(len(service._parse_dialogue_lines(call, good)), 2)
        for broken in (
            [good[0], "сломанная строка", good[1]],
            [good[0], "[00:05.0] Клиент:"],
            ["[00:05.0] Клиент: Позже", "[00:01.0] Менеджер: Раньше"],
        ):
            with self.subTest(broken=broken):
                self.assertEqual(service._parse_dialogue_lines(call, broken), [])

    def test_resolve_keeps_no_second_speaker_regex_of_its_own(self) -> None:
        import mango_mvp.services.resolve as resolve_module

        self.assertFalse(hasattr(resolve_module, "TIMED_LINE_RE"))
        self.assertFalse(hasattr(resolve_module, "RESOLVE_SPEAKER_LABEL_RE"))


class ResolveStaleResultGuardTest(unittest.TestCase):
    """Этап A: a lost lease or a changed input never overwrites a newer row."""

    def setUp(self) -> None:
        """Offline sentinel: this class measures the stale guard, never ASR.

        ``resolve_rescue_provider`` was left at its default here, so a baseline
        score below the accept threshold sent the guard through the real rescue
        provider and imported ``mlx_whisper`` — a measurement bug of the test
        that also broke the offline boundary of the run.  Configuration alone
        would fix it silently; these two sentinels make a regression loud.

        Neither of them enables anything: rescue is asserted to be disabled by
        configuration and returns nothing, and any attempt to actually
        transcribe audio fails the test instead of loading a model.
        """
        def forbidden_transcribe(_self, _call):
            raise AssertionError(
                "stale guard tests must never run ASR: rescue leaked into the run"
            )

        def rescue_must_stay_disabled(inner_self, _call):
            assert inner_self._rescue_provider() == "", (
                "stale guard tests must keep resolve_rescue_provider disabled"
            )
            return None

        for patcher in (
            patch.object(TranscribeService, "_transcribe_call", forbidden_transcribe),
            patch.object(ResolveService, "_run_rescue_asr", rescue_must_stay_disabled),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _prepare(self, td, **overrides):
        db_path = Path(td) / "stale.db"
        options = {
            "database_url": f"sqlite:///{db_path}",
            "resolve_llm_provider": "off",
            # Offline by construction, not by luck: rescue ASR is a second
            # provider that would run a real model on a fixture path.
            "resolve_rescue_provider": "none",
            "transcript_export_dir": str(Path(td) / "export"),
            **overrides,
        }
        settings = replace(make_settings(), **options)
        init_db(settings)
        session_factory = build_session_factory(settings)
        with session_factory() as session:
            session.add(
                CallRecord(
                    source_call_id="call-7",
                    source_file=str(Path(td) / "a.mp3"),
                    source_filename="a.mp3",
                    duration_sec=120.0,
                    transcription_status="done",
                    resolve_status="pending",
                    analysis_status="pending",
                    transcript_text="MANAGER:\nЗдравствуйте\n\nCLIENT:\nДобрый день",
                    transcript_manager="Здравствуйте",
                    transcript_client="Добрый день",
                    transcript_variants_json=json.dumps(
                        {
                            "mode": "stereo",
                            "warnings": [],
                            "manager": {"variant_a": "Здравствуйте", "variant_b": "",
                                        "final": "Здравствуйте"},
                            "client": {"variant_a": "Добрый день", "variant_b": "",
                                       "final": "Добрый день"},
                        },
                        ensure_ascii=False,
                    ),
                )
            )
            session.commit()
        return settings, session_factory

    def test_progress_payload_does_not_expose_source_filename(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_progress_") as td:
            settings, session_factory = self._prepare(td)
            with session_factory() as session:
                call = session.query(CallRecord).one()
                call.duration_sec = 1.0
                session.commit()

            events = []
            with session_factory() as session:
                ResolveService(settings).run_with_progress(
                    session,
                    limit=1,
                    progress_callback=events.append,
                )

            self.assertTrue(any(event.get("call_id") for event in events))
            self.assertFalse(any("source_filename" in event for event in events))
            self.assertNotIn("a.mp3", json.dumps(events, ensure_ascii=False))

    @staticmethod
    def _export_files(settings):
        root = Path(settings.transcript_export_dir)
        if not root.exists():
            return []
        return sorted(path.name for path in root.rglob("*") if path.is_file())

    @staticmethod
    def _race_before_commit(session_factory, sql, params=None):
        """Commit a foreign change in the gap between scoring and the write."""
        original = ResolveService._build_resolve_payload

        def racing(self, **kwargs):
            with session_factory() as other:
                other.execute(sa_text(sql), params or {})
                other.commit()
            return original(self, **kwargs)

        return patch.object(ResolveService, "_build_resolve_payload", racing)

    def test_a_stolen_lease_rejects_the_write_and_exports_no_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_stale_") as td:
            settings, session_factory = self._prepare(td)
            service = ResolveService(settings)

            with self._race_before_commit(
                session_factory,
                "UPDATE call_records SET pipeline_worker_id = 'other-worker'",
            ):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            self.assertEqual(result["success"], 0)
            self.assertEqual(result["failed"], 0)
            with session_factory() as session:
                call = session.query(CallRecord).one()
                self.assertEqual(call.resolve_status, "in_progress")
                self.assertEqual(call.pipeline_worker_id, "other-worker")
                self.assertIsNone(call.resolve_json)
            self.assertEqual(self._export_files(settings), [])

    def test_a_changed_input_rejects_the_write_and_exports_no_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_changed_") as td:
            settings, session_factory = self._prepare(td)
            service = ResolveService(settings)

            with self._race_before_commit(
                session_factory,
                "UPDATE call_records SET transcript_text = 'другой текст'",
            ):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            self.assertEqual(result["success"], 0)
            with session_factory() as session:
                call = session.query(CallRecord).one()
                # Neither the old nor the new value was overwritten by our answer.
                self.assertEqual(call.transcript_text, "другой текст")
                self.assertIsNone(call.resolve_json)
                self.assertEqual(call.resolve_status, "in_progress")
            self.assertEqual(self._export_files(settings), [])

    def test_a_stale_sidecar_releases_its_own_unchanged_claim(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_sidecar_stale_") as td:
            settings, session_factory = self._prepare(td)
            service = ResolveService(settings)

            with patch.object(
                ResolveService, "_candidate_source_is_current", return_value=False
            ):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            self.assertEqual(result["success"], 0)
            with session_factory() as session:
                call = session.query(CallRecord).one()
                self.assertEqual(call.resolve_status, "pending")
                self.assertIsNone(call.pipeline_stage)
                self.assertIsNone(call.pipeline_worker_id)
                self.assertIsNone(call.pipeline_claimed_at)
                self.assertEqual(call.resolve_attempts, 0)
            self.assertEqual(self._export_files(settings), [])

    def test_the_input_snapshot_covers_the_channel_count_too(self) -> None:
        """Mono vs stereo decides which candidates Resolve may build at all.

        A re-ingest that flips it while the rescue ASR or the dialogue LLM is
        running makes our answer describe a call that no longer exists, so it
        belongs in the stale guard like the transcript itself.
        """
        self.assertIn("channels", resolve_module.RESOLVE_INPUT_COLUMNS)
        with tempfile.TemporaryDirectory(prefix="mango_resolve_channels_") as td:
            settings, session_factory = self._prepare(td)
            service = ResolveService(settings)

            with self._race_before_commit(
                session_factory, "UPDATE call_records SET channels = 1"
            ):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            self.assertEqual(result["success"], 0)
            with session_factory() as session:
                call = session.query(CallRecord).one()
                self.assertIsNone(call.resolve_json)
                self.assertEqual(call.resolve_status, "in_progress")
            self.assertEqual(self._export_files(settings), [])

    def test_a_stolen_lease_on_the_failure_path_leaves_the_new_owner_alone(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_fail_race_") as td:
            settings, session_factory = self._prepare(td)
            service = ResolveService(settings)

            def steal_then_fail(_self, *_args, **_kwargs):
                with session_factory() as thief:
                    thief.execute(
                        sa_text(
                            "UPDATE call_records SET pipeline_worker_id = 'other-worker'"
                        )
                    )
                    thief.commit()
                raise RuntimeError("scoring blew up")

            with patch.object(ResolveService, "_score_candidate", steal_then_fail):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 1)
            self.assertEqual(result["failed"], 0)
            with session_factory() as session:
                call = session.query(CallRecord).one()
                self.assertEqual(call.resolve_status, "in_progress")
                self.assertEqual(call.pipeline_worker_id, "other-worker")
                self.assertIsNone(call.last_error)

    def test_a_provider_error_is_stored_without_leaking_the_conversation(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_error_") as td:
            settings, session_factory = self._prepare(td)
            service = ResolveService(settings)

            def leaky(_self, *_args, **_kwargs):
                raise RuntimeError(
                    "provider echoed the transcript: "
                    "клиент Мария Иванова, телефон +79990000000, оплата 60000 " * 30
                )

            with patch.object(ResolveService, "_score_candidate", leaky):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["failed"], 1)
            with session_factory() as session:
                call = session.query(CallRecord).one()
            self.assertLessEqual(len(call.last_error), 260)
            self.assertNotIn("+79990000000", call.last_error)
            self.assertIn("RuntimeError", call.last_error)
            # Not only the phone: the leak is at the *front* of the message, so
            # the name, the price and the echoed transcript must be gone too —
            # a bounded prefix of this message would have kept all three.
            self.assertNotIn("Мария", call.last_error)
            self.assertNotIn("Иванова", call.last_error)
            self.assertNotIn("provider echoed the transcript", call.last_error)
            self.assertNotIn("клиент", call.last_error.lower())
            self.assertIn("message_sha256=", call.last_error)
            # The digest is hex, so a decimal price is only checked against the
            # readable part — a hash cannot be searched for a chosen substring.
            self.assertNotIn("60000", call.last_error.split("message_sha256=")[0])

    def test_a_failed_export_is_counted_and_the_row_stays_committed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_export_") as td:
            # A trigger above any score forces the improved-candidate path, which
            # is the only one that writes a transcript file.
            settings, session_factory = self._prepare(
                td, resolve_llm_trigger_score=101, resolve_accept_score=50
            )
            service = ResolveService(settings)

            def better_candidate(_self, call, _payload):
                return {
                    "name": "llm",
                    "transcript_manager": "Здравствуйте, слушаю вас",
                    "transcript_client": "Добрый день, нужен курс",
                    "transcript_text": "MANAGER:\nЗдравствуйте, слушаю вас"
                                       "\n\nCLIENT:\nДобрый день, нужен курс",
                    "dialogue_lines": [
                        "[00:01.0] Менеджер (Иван): Здравствуйте, слушаю вас",
                        "[00:03.0] Клиент: Добрый день, нужен курс",
                    ],
                    "transcript_variants_json": json.dumps(
                        {"mode": "stereo", "warnings": []}, ensure_ascii=False
                    ),
                    "meta": {"mode": "stereo", "provider": "test"},
                }

            def broken_export(_self, _call, _result):
                raise OSError("export target is read-only")

            with patch.object(ResolveService, "_resolve_with_llm", better_candidate), \
                    patch.object(
                        ResolveService, "_score_candidate",
                        lambda _self, *_a, **_k: {
                            "score": 100, "reasons": ["clean"], "signals": {}
                        },
                    ), \
                    patch(
                        "mango_mvp.services.transcribe.TranscribeService."
                        "_export_transcript_file",
                        broken_export,
                    ):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["success"], 1)
            self.assertEqual(result["export_failed"], 1)
            self.assertEqual(result["stale"], 0)
            with session_factory() as session:
                call = session.query(CallRecord).one()
                self.assertEqual(call.resolve_status, "done")
                self.assertEqual(call.transcript_manager, "Здравствуйте, слушаю вас")

    def test_resolve_lease_stays_held_until_the_export_finishes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_export_lease_") as td:
            settings, session_factory = self._prepare(
                td, resolve_llm_trigger_score=101, resolve_accept_score=50
            )
            service = ResolveService(settings)

            def better_candidate(_self, call, _payload):
                return {
                    "name": "llm",
                    "transcript_manager": "Здравствуйте",
                    "transcript_client": "Добрый день",
                    "transcript_text": "MANAGER:\nЗдравствуйте\n\nCLIENT:\nДобрый день",
                    "dialogue_lines": [
                        "[00:01.0] Менеджер: Здравствуйте",
                        "[00:03.0] Клиент: Добрый день",
                    ],
                    "transcript_variants_json": json.dumps(
                        {"mode": "stereo", "warnings": []}, ensure_ascii=False
                    ),
                    "meta": {"mode": "stereo", "provider": "test"},
                }

            observed = {}

            def inspect_export(_self, _call, _result):
                with session_factory() as observer:
                    current = observer.query(CallRecord).one()
                    observed["stage"] = current.pipeline_stage
                    observed["worker"] = current.pipeline_worker_id

            with patch.object(ResolveService, "_resolve_with_llm", better_candidate), patch.object(
                ResolveService,
                "_score_candidate",
                lambda _self, *_a, **_k: {
                    "score": 100,
                    "reasons": ["clean"],
                    "signals": {},
                },
            ), patch(
                "mango_mvp.services.transcribe.TranscribeService._export_transcript_file",
                inspect_export,
            ):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["success"], 1)
            self.assertEqual(observed["stage"], "resolve")
            self.assertTrue(observed["worker"])
            with session_factory() as session:
                current = session.query(CallRecord).one()
                self.assertIsNone(current.pipeline_stage)
                self.assertIsNone(current.pipeline_worker_id)

    def test_identity_change_after_commit_blocks_export_before_path_use(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_export_identity_") as td:
            settings, session_factory = self._prepare(
                td, resolve_llm_trigger_score=101, resolve_accept_score=50
            )
            service = ResolveService(settings)

            def better_candidate(_self, _call, _payload):
                return {
                    "name": "llm",
                    "transcript_manager": "Здравствуйте",
                    "transcript_client": "Добрый день",
                    "transcript_text": "MANAGER:\nЗдравствуйте\n\nCLIENT:\nДобрый день",
                    "dialogue_lines": [
                        "[00:01.0] Менеджер: Здравствуйте",
                        "[00:03.0] Клиент: Добрый день",
                    ],
                    "transcript_variants_json": json.dumps(
                        {"mode": "stereo", "warnings": []}, ensure_ascii=False
                    ),
                    "meta": {"mode": "stereo", "provider": "test"},
                }

            original_transition = ResolveService._transition_resolve_export_claim
            raced = False

            def change_identity_before_export(inner_self, session, **kwargs):
                nonlocal raced
                if not kwargs["release"] and not raced:
                    raced = True
                    with session_factory() as other:
                        other.execute(
                            sa_text(
                                "UPDATE call_records "
                                "SET source_call_id='other-call', "
                                "source_recording_id='other-recording', "
                                "source_file=:source_file"
                            ),
                            {"source_file": str(Path(td) / "other.mp3")},
                        )
                        other.commit()
                return original_transition(inner_self, session, **kwargs)

            with patch.object(ResolveService, "_resolve_with_llm", better_candidate), patch.object(
                ResolveService,
                "_score_candidate",
                lambda _self, *_a, **_k: {
                    "score": 100,
                    "reasons": ["clean"],
                    "signals": {},
                },
            ), patch.object(
                ResolveService,
                "_transition_resolve_export_claim",
                change_identity_before_export,
            ), patch(
                "mango_mvp.services.transcribe.TranscribeService._export_transcript_file"
            ) as export_mock:
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["success"], 1)
            self.assertEqual(result["stale"], 1)
            self.assertEqual(result["export_failed"], 0)
            export_mock.assert_not_called()
            with session_factory() as session:
                current = session.query(CallRecord).one()
                self.assertEqual(current.source_call_id, "other-call")
                self.assertEqual(current.source_recording_id, "other-recording")
                self.assertEqual(current.source_file, str(Path(td) / "other.mp3"))
                self.assertEqual(current.resolve_status, "done")
                self.assertEqual(current.pipeline_stage, "resolve")
            self.assertEqual(self._export_files(settings), [])

    def test_result_commit_refreshes_lease_before_export(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_export_heartbeat_") as td:
            settings, session_factory = self._prepare(
                td, resolve_llm_trigger_score=101, resolve_accept_score=50
            )
            service = ResolveService(settings)

            def better_candidate(_self, _call, _payload):
                return {
                    "name": "llm",
                    "transcript_manager": "Здравствуйте",
                    "transcript_client": "Добрый день",
                    "transcript_text": "MANAGER:\nЗдравствуйте\n\nCLIENT:\nДобрый день",
                    "dialogue_lines": [
                        "[00:01.0] Менеджер: Здравствуйте",
                        "[00:03.0] Клиент: Добрый день",
                    ],
                    "transcript_variants_json": json.dumps(
                        {"mode": "stereo", "warnings": []}, ensure_ascii=False
                    ),
                    "meta": {"mode": "stereo", "provider": "test"},
                }

            original_payload = ResolveService._build_resolve_payload

            def age_lease_before_result_commit(inner_self, **kwargs):
                with session_factory() as other:
                    other.execute(
                        sa_text(
                            "UPDATE call_records "
                            "SET pipeline_claimed_at='2000-01-01 00:00:00'"
                        )
                    )
                    other.commit()
                return original_payload(inner_self, **kwargs)

            observed = {}

            def inspect_export(_self, _call, _result):
                with session_factory() as observer:
                    current = observer.query(CallRecord).one()
                    observed["claimed_at"] = current.pipeline_claimed_at
                    observed["stage"] = current.pipeline_stage

            with patch.object(ResolveService, "_resolve_with_llm", better_candidate), patch.object(
                ResolveService,
                "_score_candidate",
                lambda _self, *_a, **_k: {
                    "score": 100,
                    "reasons": ["clean"],
                    "signals": {},
                },
            ), patch.object(
                ResolveService, "_build_resolve_payload", age_lease_before_result_commit
            ), patch(
                "mango_mvp.services.transcribe.TranscribeService._export_transcript_file",
                inspect_export,
            ):
                with session_factory() as session:
                    result = service.run(session, limit=1)

            self.assertEqual(result["success"], 1)
            self.assertEqual(result["stale"], 0)
            self.assertEqual(observed["stage"], "resolve")
            self.assertIsNotNone(observed["claimed_at"])
            self.assertGreater(observed["claimed_at"].year, 2000)

    def test_an_unchanged_lease_and_input_commit_and_export(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_resolve_ok_") as td:
            settings, session_factory = self._prepare(td)
            service = ResolveService(settings)

            with session_factory() as session:
                result = service.run(session, limit=1)

            self.assertEqual(result["stale"], 0)
            self.assertEqual(result["export_failed"], 0)
            self.assertEqual(result["failed"], 0)
            with session_factory() as session:
                call = session.query(CallRecord).one()
                self.assertIn(call.resolve_status, {"done", "manual"})
                self.assertIsNone(call.pipeline_worker_id)
                self.assertIsNotNone(call.resolve_json)

    def test_pipeline_worker_ids_are_unique_even_inside_one_clock_tick(self) -> None:
        identifiers = {
            ResolveService._pipeline_worker_id("resolve") for _ in range(100)
        }

        self.assertEqual(len(identifiers), 100)
        self.assertTrue(all(value.startswith("resolve-") for value in identifiers))


if __name__ == "__main__":
    unittest.main()
