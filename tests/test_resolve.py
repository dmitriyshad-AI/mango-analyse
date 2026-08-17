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
from mango_mvp import config as config_module
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
            service._run_dialogue_llm = lambda request, **_kwargs: {
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

            # The default is asserted against a cleaned environment: an ambient flag of
            # the host must never be what makes this green.
            with patch.dict("os.environ", {"RESOLVE_SEMANTIC_MERGE_MODE": ""}):
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

            def fake_dialogue_runner(payload: dict, **_kwargs) -> dict:
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


# One stereo call in five turns.  The manager side is heard twice: Whisper as
# MANAGER_A, GigaAM as MANAGER_B — the same 36 words, 16 of them heard as other
# words (including the name), the price identical.  The client side is heard the
# same way twice, so it is eligible but never divergent.
MANAGER_TURN_1 = "Добрый день, меня зовут Анна, я звоню по вашей заявке на подготовительный курс."
MANAGER_TURN_2 = (
    "Мы обсуждали расписание занятий в субботу и стоимость обучения 47250 рублей за первый семестр."
)
MANAGER_TURN_3 = "Хорошо, значит тогда я отправляю договор вам на почту."
CLIENT_TURN_1 = (
    "Здравствуйте, да, я оставляла заявку, хотела узнать сколько стоит обучение "
    "и когда начинаются занятия у вас в центре."
)
CLIENT_TURN_2 = "А также можно ли оплатить частями после первого урока?"
MANAGER_A = f"{MANAGER_TURN_1} {MANAGER_TURN_2} {MANAGER_TURN_3}"
MANAGER_B = (
    "Добрый вечер, меня зовут Ольга, я пишу про вашу заявку на вводный курс. "
    "Мы уточняли график уроков в воскресенье и цену учебы 47250 рублей за первый семестр. "
    "Ладно, значит тогда я отправлю контракт вам на почту."
)
CLIENT_SAME = f"{CLIENT_TURN_1} {CLIENT_TURN_2}"
# The one edit of this fixture that every guard check lets through: one word replaced in
# place by a near-identical word of the second ASR variant, in a turn that carries no
# number, no date, no name cue, no negation and no hedge.
MANAGER_TURN_3_FIXED = MANAGER_TURN_3.replace("отправляю", "отправлю")


def semantic_settings(**overrides):
    values = {
        "resolve_llm_provider": "codex_cli",
        "resolve_semantic_merge_mode": "selective",
        "llm_cache_enabled": False,
    }
    values.update(overrides)
    return replace(make_settings(), **values)


def semantic_input(
    *,
    manager_a: str = MANAGER_A,
    manager_b: str = MANAGER_B,
    client_a: str = CLIENT_SAME,
    client_b: str = CLIENT_SAME,
):
    """The existing dialogue_resolve_v1 payload, as _build_dialogue_resolve_payload builds it."""
    baselines = [
        ("manager", MANAGER_TURN_1),
        ("client", CLIENT_TURN_1),
        ("manager", MANAGER_TURN_2),
        ("client", CLIENT_TURN_2),
        ("manager", MANAGER_TURN_3),
    ]
    return {
        "schema_version": "dialogue_resolve_v1",
        "call_id": 77,
        "source_filename": "2026-08-17_Анна_Петрова.mp3",
        "manager_name": "Анна Петрова",
        "mode": "stereo",
        "role_variants": {
            "manager": {"variant_a": manager_a, "variant_b": manager_b, "baseline_text": manager_a},
            "client": {"variant_a": client_a, "variant_b": client_b, "baseline_text": client_a},
        },
        "turns": [
            {
                "turn_id": index + 1,
                "ts_sec": float(index + 1),
                "speaker": speaker,
                "baseline_text": text,
                "approximate": False,
                "flags": [],
            }
            for index, (speaker, text) in enumerate(baselines)
        ],
    }


def model_turns(projection, replacements=None, *, drops=(), warnings=None, global_notes=""):
    """Model answer in the existing dialogue format: same turn_id set, final_text."""
    replacements = replacements or {}
    payload = {
        "turns": [
            {
                "turn_id": turn["turn_id"],
                "speaker": turn["speaker"],
                "final_text": replacements.get(turn["turn_id"], turn["baseline_text"]),
                "drop": turn["turn_id"] in drops,
            }
            for turn in projection["turns"]
        ]
    }
    if warnings is not None:
        payload["warnings"] = warnings
    if global_notes:
        payload["global_notes"] = global_notes
    return payload


def guard_projection(*, variant_b: str = "", conflict: bool = False, glossary=None, role: str = "manager"):
    """The minimal shape the output guard reads, so a G-check can be attacked alone."""
    return {
        "editable_roles": [role],
        "semantic_merge": {"numeric_conflict": {role: conflict}},
        "role_variants": {role: {"variant_a": "", "variant_b": variant_b}},
        "glossary": glossary or [],
    }


def model_answer(replacements=None):
    """The same answer, written the way the CLI writes it: every turn, final_text."""
    replacements = replacements or {}
    texts = {
        1: MANAGER_TURN_1,
        2: CLIENT_TURN_1,
        3: MANAGER_TURN_2,
        4: CLIENT_TURN_2,
        5: MANAGER_TURN_3,
    }
    return {
        "turns": [
            {"turn_id": turn_id, "final_text": replacements.get(turn_id, text)}
            for turn_id, text in texts.items()
        ]
    }


class ResolveSemanticMergeGateTest(unittest.TestCase):
    """ТЗ §3: who is escalated, and — much more often — who is not."""

    def test_matching_variants_escalate_nothing_and_call_nothing(self) -> None:
        service = ResolveService(semantic_settings())

        projection = service._semantic_selective_input(
            semantic_input(manager_b=MANAGER_A)
        )

        self.assertIsNone(projection)
        block = service._semantic_merge_last
        self.assertTrue(block["eligible"])
        self.assertFalse(block["escalated"])
        self.assertEqual(block["model_calls"], 0)
        self.assertEqual(block["escalation_reasons"], [])

    def test_two_signals_below_threshold_escalate_only_that_side(self) -> None:
        service = ResolveService(semantic_settings())

        projection = service._semantic_selective_input(semantic_input())

        self.assertEqual(projection["editable_roles"], ["manager"])
        block = service._semantic_merge_last
        self.assertEqual(block["escalation_reasons"], ["side_divergent:manager"])
        signals = block["signals"]["manager"]
        self.assertLess(signals["dice_tokens"], 0.82)
        self.assertLess(signals["dice_char3"], 0.88)
        self.assertEqual(signals["len_ratio"], 1.0)
        self.assertEqual(block["signals"]["client"]["dice_tokens"], 1.0)

    def test_one_signal_below_threshold_does_not_escalate(self) -> None:
        service = ResolveService(semantic_settings())
        # 36 vs 43 words: only len_ratio falls below its threshold, the dice do not.
        longer = MANAGER_A + " и еще раз до новых скорых встреч"

        projection = service._semantic_selective_input(
            semantic_input(manager_b=longer, client_a="", client_b="")
        )

        self.assertIsNone(projection)
        signals = service._semantic_merge_last["signals"]["manager"]
        self.assertLess(signals["len_ratio"], 0.85)
        self.assertGreaterEqual(signals["dice_tokens"], 0.82)
        self.assertGreaterEqual(signals["dice_char3"], 0.88)
        self.assertFalse(service._semantic_merge_last["escalated"])

    def test_hard_length_loss_escalates_on_its_own_signal(self) -> None:
        service = ResolveService(semantic_settings())

        projection = service._semantic_selective_input(
            semantic_input(manager_a=f"{MANAGER_A} {MANAGER_A}", manager_b=MANAGER_A)
        )

        self.assertEqual(projection["editable_roles"], ["manager"])
        block = service._semantic_merge_last
        self.assertEqual(block["escalation_reasons"], ["hard_length_loss:manager"])
        self.assertLess(block["signals"]["manager"]["len_ratio"], 0.60)

    def test_a_short_side_never_blocks_the_eligible_other_side(self) -> None:
        service = ResolveService(semantic_settings())

        projection = service._semantic_selective_input(
            semantic_input(client_a="да хорошо спасибо", client_b="да ладно спасибо")
        )

        self.assertEqual(projection["editable_roles"], ["manager"])
        self.assertEqual(sorted(service._semantic_merge_last["signals"]), ["manager"])

    def test_numeric_conflict_is_reported_but_never_escalates_by_itself(self) -> None:
        service = ResolveService(semantic_settings())

        projection = service._semantic_selective_input(
            semantic_input(
                manager_b=MANAGER_A,
                client_a=f"{CLIENT_SAME} за 47250 рублей",
                client_b=f"{CLIENT_SAME} за 47 250 рублей",
            )
        )

        self.assertIsNone(projection)
        block = service._semantic_merge_last
        self.assertTrue(block["numeric_conflict"]["client"])
        self.assertFalse(block["escalated"])

    def test_the_projection_carries_no_identity_and_no_new_schema(self) -> None:
        service = ResolveService(semantic_settings())
        payload = semantic_input()

        projection = service._semantic_selective_input(payload)

        for key in ("call_id", "source_filename", "manager_name", "mode"):
            self.assertNotIn(key, projection)
            self.assertIn(key, payload)
        self.assertEqual(projection["schema_version"], "dialogue_resolve_v1")
        self.assertEqual([turn["turn_id"] for turn in projection["turns"]], [1, 2, 3, 4, 5])
        self.assertEqual([turn["baseline_text"] for turn in projection["turns"]],
                         [turn["baseline_text"] for turn in payload["turns"]])

    def test_a_turn_travels_in_five_fields_and_carries_no_label(self) -> None:
        service = ResolveService(semantic_settings())
        payload = semantic_input()
        for turn in payload["turns"]:
            # What _build_dialogue_resolve_payload really puts there today.
            turn.update({"ts_label": "00:07.0", "speaker_label": "Анна Петрова", "flags": ["same_ts_cross"]})

        projection = service._semantic_selective_input(payload)

        for turn in projection["turns"]:
            self.assertEqual(sorted(turn), ["approximate", "baseline_text", "speaker", "ts_sec", "turn_id"])
        self.assertNotIn("Анна Петрова", json.dumps(projection, ensure_ascii=False))

    def test_an_unconfirmed_role_never_reaches_a_model(self) -> None:
        service = ResolveService(semantic_settings())
        payload = semantic_input()
        payload["turns"][1]["speaker"] = "channel_left"

        projection = service._semantic_selective_input(payload)

        self.assertIsNone(projection)
        block = service._semantic_merge_last
        self.assertEqual(block["fallback_reason"], "unconfirmed_roles")
        self.assertEqual(block["model_calls"], 0)
        self.assertFalse(block["escalated"])
        self.assertEqual(block["signals"], {})

    def test_an_unknown_role_blocks_the_call_as_well(self) -> None:
        service = ResolveService(semantic_settings())
        payload = semantic_input()
        payload["turns"][4]["speaker"] = "unknown"

        self.assertIsNone(service._semantic_selective_input(payload))
        self.assertEqual(service._semantic_merge_last["fallback_reason"], "unconfirmed_roles")

    def test_the_projection_drops_every_field_nobody_downstream_reads(self) -> None:
        service = ResolveService(semantic_settings())
        payload = semantic_input()
        payload.update({"duration_sec": 12.5, "providers": {"primary": "whisper"},
                        "quality_hints": {"warnings": ["дословная цитата менеджера"]}})

        projection = service._semantic_selective_input(payload)

        self.assertEqual(sorted(projection), ["editable_roles", "glossary", "role_variants",
                                              "schema_version", "semantic_merge", "turns"])
        # Only the escalated side travels, and its duplicated role baseline does not.
        self.assertEqual(sorted(projection["role_variants"]), ["manager"])
        self.assertEqual(sorted(projection["role_variants"]["manager"]), ["variant_a", "variant_b"])
        self.assertNotIn("numeric_conflict", projection["semantic_merge"])
        self.assertNotIn("дословная цитата менеджера",
                         json.dumps(projection, ensure_ascii=False))

    def test_telemetry_holds_numbers_versions_and_codes_only(self) -> None:
        service = ResolveService(semantic_settings())

        service._semantic_selective_input(semantic_input())
        dumped = json.dumps(service._semantic_merge_last, ensure_ascii=False)

        for secret in ("Анна", "Петрова", "заявке", "47250", "2026-08-17"):
            self.assertNotIn(secret, dumped)
        versions = service._semantic_merge_last["versions"]
        self.assertEqual(versions["prompt"], "resolve_semantic_guard_v1")
        self.assertEqual(versions["schema_in"], "dialogue_resolve_v1+semantic_merge_v1")
        self.assertEqual(versions["normalizer"], "tenant_text_engine_v1/tenant_ru_v1")
        self.assertEqual(versions["reasoning"], "medium")

    def test_the_same_input_twice_gives_the_same_telemetry(self) -> None:
        service = ResolveService(semantic_settings())

        service._semantic_selective_input(semantic_input())
        first = json.loads(json.dumps(service._semantic_merge_last, ensure_ascii=False))
        service._semantic_selective_input(semantic_input())

        self.assertEqual(service._semantic_merge_last, first)

    def test_the_glossary_is_the_live_normalizer_and_nothing_else(self) -> None:
        service = ResolveService(semantic_settings())

        projection = service._semantic_selective_input(
            semantic_input(manager_b=f"{MANAGER_B} центр МПК МФТИ")
        )

        self.assertEqual(
            projection["glossary"],
            [{"alias": "МПК МФТИ", "canonical": "УНПК МФТИ",
              "rule_id": "brand.unpk_mfti.known_alias"}],
        )

    def test_a_foreign_tenant_ruleset_leaves_the_glossary_empty(self) -> None:
        service = ResolveService(semantic_settings())

        with patch(
            "mango_mvp.quality.tenant_text_normalizer.TENANT_TEXT_RULESET_VERSIONS", {}
        ):
            projection = service._semantic_selective_input(
                semantic_input(manager_b=f"{MANAGER_B} центр МПК МФТИ")
            )

        self.assertEqual(projection["glossary"], [])
        self.assertEqual(
            service._semantic_merge_last["versions"]["normalizer"], "tenant_text_engine_v1/"
        )

    def test_a_real_foreign_tenant_id_gets_no_glossary_either(self) -> None:
        service = ResolveService(semantic_settings(controlled_call_tenant_id="another_customer"))

        projection = service._semantic_selective_input(
            semantic_input(manager_b=f"{MANAGER_B} центр МПК МФТИ")
        )

        self.assertEqual(projection["glossary"], [])
        self.assertEqual(
            service._semantic_merge_last["versions"]["normalizer"], "tenant_text_engine_v1/"
        )

    def test_the_controlled_tenant_of_the_scope_is_the_one_used(self) -> None:
        service = ResolveService(semantic_settings(controlled_call_tenant_id="mango"))

        projection = service._semantic_selective_input(
            semantic_input(manager_b=f"{MANAGER_B} центр МПК МФТИ")
        )

        self.assertEqual([item["alias"] for item in projection["glossary"]], ["МПК МФТИ"])


class ResolveSemanticMergeGuardTest(unittest.TestCase):
    """ТЗ §5: what the model returns is a proposal; the guard decides."""

    def _normalize(self, replacements, *, projection=None, service=None, **kwargs):
        service = service or ResolveService(semantic_settings())
        projection = projection or service._semantic_selective_input(semantic_input())
        result = service._normalize_dialogue_result(
            projection, model_turns(projection, replacements, **kwargs)
        )
        return service, projection, result

    @staticmethod
    def _text(result, turn_id):
        return next(
            turn["final_text"] for turn in result["turns"] if turn["turn_id"] == turn_id
        )

    def test_a_supported_word_from_the_second_asr_is_accepted(self) -> None:
        service, _, result = self._normalize({5: MANAGER_TURN_3_FIXED})

        self.assertEqual(self._text(result, 5), MANAGER_TURN_3_FIXED)
        block = service._semantic_merge_last
        self.assertEqual(block["turns_changed_proposed"], 1)
        self.assertEqual(block["turns_changed_accepted"], 1)
        self.assertEqual(block["turns_reset"], {})

    def test_g0_a_turn_of_a_non_editable_role_is_reset(self) -> None:
        service, _, result = self._normalize(
            {
                5: MANAGER_TURN_3_FIXED,
                4: "А также можно ли оплатить частями после первого занятия?",
            }
        )

        self.assertEqual(self._text(result, 4), CLIENT_TURN_2)
        block = service._semantic_merge_last
        self.assertEqual(block["turns_reset"], {"non_editable_role_change": 1})
        # The other turn of the same call survives its neighbour's reset.
        self.assertEqual(block["turns_changed_accepted"], 1)

    def test_g1_a_changed_digit_is_reset(self) -> None:
        service, _, result = self._normalize(
            {3: MANAGER_TURN_2.replace("47250", "47 250")}
        )

        self.assertEqual(self._text(result, 3), MANAGER_TURN_2)
        self.assertEqual(service._semantic_merge_last["turns_reset"], {"numeric_change": 1})

    def test_g1_a_turn_holding_a_price_is_frozen_whole(self) -> None:
        # Not just the number: a turn with a fact in it is not edited at all (ТЗ §12b).
        service, _, result = self._normalize(
            {3: MANAGER_TURN_2.replace("обсуждали", "уточняли")}
        )

        self.assertEqual(self._text(result, 3), MANAGER_TURN_2)
        self.assertEqual(service._semantic_merge_last["turns_reset"], {"numeric_change": 1})

    def test_g1_under_a_numeric_conflict_a_numeric_turn_is_frozen(self) -> None:
        service = ResolveService(semantic_settings())
        projection = service._semantic_selective_input(
            semantic_input(manager_b=MANAGER_B.replace("47250", "4725"))
        )

        _, _, result = self._normalize(
            {3: MANAGER_TURN_2.replace("обсуждали", "уточняли")},
            projection=projection,
            service=service,
        )

        self.assertTrue(service._semantic_merge_last["numeric_conflict"]["manager"])
        self.assertEqual(self._text(result, 3), MANAGER_TURN_2)
        self.assertEqual(service._semantic_merge_last["turns_reset"], {"numeric_change": 1})

    def test_g2_a_turn_carrying_a_negation_is_frozen_whole(self) -> None:
        service, _, result = self._normalize(
            {5: MANAGER_TURN_3.replace("я отправляю", "я не отправляю")}
        )

        self.assertEqual(self._text(result, 5), MANAGER_TURN_3)
        self.assertEqual(
            service._semantic_merge_last["turns_reset"], {"negation_frozen_turn": 1}
        )

    def test_g3_a_word_from_neither_variant_is_reset(self) -> None:
        service, _, result = self._normalize(
            {5: MANAGER_TURN_3.replace("отправляю", "отправляем")}
        )

        self.assertEqual(self._text(result, 5), MANAGER_TURN_3)
        self.assertEqual(service._semantic_merge_last["turns_reset"], {"unsupported_token": 1})

    def test_g4_a_name_is_not_corrected_even_when_variant_b_spells_it(self) -> None:
        service, projection, result = self._normalize(
            {1: MANAGER_TURN_1.replace("Анна", "Ольга")}
        )

        self.assertIn("Ольга", projection["role_variants"]["manager"]["variant_b"])
        self.assertEqual(self._text(result, 1), MANAGER_TURN_1)
        # The name cue freezes the turn before the name check even runs (ТЗ §12a).
        self.assertEqual(
            service._semantic_merge_last["turns_reset"], {"name_cue_frozen_turn": 1}
        )

    def test_g4_a_new_capitalised_token_is_reset(self) -> None:
        service, _, result = self._normalize(
            {5: MANAGER_TURN_3.replace("Хорошо", "Ольга")}
        )

        self.assertEqual(self._text(result, 5), MANAGER_TURN_3)
        self.assertEqual(service._semantic_merge_last["turns_reset"], {"proper_name_change": 1})

    def test_g5_a_shortened_turn_is_reset(self) -> None:
        service, _, result = self._normalize(
            {5: "Хорошо, значит тогда я отправляю."}
        )

        self.assertEqual(self._text(result, 5), MANAGER_TURN_3)
        self.assertEqual(service._semantic_merge_last["turns_reset"], {"length_loss_reset": 1})

    def test_an_added_phrase_is_reset_by_the_growth_limit(self) -> None:
        service, _, result = self._normalize(
            {5: "Хорошо, значит тогда я отправляю договор вам на почту вашу заявку на курс."}
        )

        self.assertEqual(self._text(result, 5), MANAGER_TURN_3)
        self.assertEqual(service._semantic_merge_last["turns_reset"], {"length_growth_reset": 1})

    def test_cosmetics_return_the_exact_baseline_before_any_counter(self) -> None:
        service, _, result = self._normalize({3: MANAGER_TURN_2.lower().replace(",", "")})

        self.assertEqual(self._text(result, 3), MANAGER_TURN_2)
        block = service._semantic_merge_last
        self.assertEqual(block["turns_changed_proposed"], 0)
        self.assertEqual(block["turns_reset"], {})
        self.assertEqual(block["turns_changed_accepted"], 0)

    def test_a_drop_is_ignored_under_the_guard_even_for_an_artifact(self) -> None:
        service = ResolveService(semantic_settings())
        payload = semantic_input()
        payload["turns"][1]["flags"] = ["artifact_candidate"]
        projection = service._semantic_selective_input(payload)

        result = service._normalize_dialogue_result(
            projection, model_turns(projection, {5: MANAGER_TURN_3_FIXED}, drops=(2,))
        )

        self.assertEqual([turn["turn_id"] for turn in result["turns"]], [1, 2, 3, 4, 5])
        self.assertIn("drop_ignored:2", result["warnings"])
        self.assertEqual(self._text(result, 2), CLIENT_TURN_1)
        # The accepted neighbour edit is not lost together with the refused drop.
        self.assertEqual(self._text(result, 5), MANAGER_TURN_3_FIXED)
        self.assertEqual(service._semantic_merge_last["turns_changed_accepted"], 1)

    def test_the_model_free_text_is_not_stored_under_the_guard(self) -> None:
        _, _, result = self._normalize(
            {5: MANAGER_TURN_3_FIXED},
            warnings=["клиент назвал сумму 47250"],
            global_notes="менеджер Анна Петрова говорила быстро",
        )

        self.assertEqual(result["global_notes"], "")
        self.assertNotIn("клиент назвал сумму 47250", result["warnings"])

    def test_the_guard_moves_no_role_no_timecode_and_no_variant(self) -> None:
        service = ResolveService(semantic_settings())
        payload = semantic_input()
        projection = service._semantic_selective_input(payload)
        before = json.dumps(projection, ensure_ascii=False, sort_keys=True)

        result = service._normalize_dialogue_result(
            projection, model_turns(projection, {5: MANAGER_TURN_3_FIXED})
        )

        self.assertEqual(json.dumps(projection, ensure_ascii=False, sort_keys=True), before)
        self.assertEqual(
            [(turn["turn_id"], turn["speaker"], turn["ts_sec"]) for turn in result["turns"]],
            [(turn["turn_id"], turn["speaker"], turn["ts_sec"]) for turn in payload["turns"]],
        )

    def test_an_editable_input_without_guard_state_fails_closed(self) -> None:
        service = ResolveService(semantic_settings())
        projection = service._semantic_selective_input(semantic_input())
        answer = model_turns(projection, {5: MANAGER_TURN_3_FIXED})
        service._semantic_merge_last = None

        with self.assertRaises(RuntimeError):
            service._normalize_dialogue_result(projection, answer)

    def test_the_default_dialogue_path_runs_without_any_guard(self) -> None:
        service = ResolveService(replace(make_settings(), resolve_llm_provider="codex_cli"))
        payload = semantic_input()

        result = service._normalize_dialogue_result(
            payload,
            model_turns(payload, {1: "Совершенно новый текст реплики."},
                        warnings=["модельное предупреждение"], global_notes="заметка"),
        )

        self.assertEqual(self._text(result, 1), "Совершенно новый текст реплики.")
        self.assertIsNone(service._semantic_merge_last)
        self.assertIn("модельное предупреждение", result["warnings"])
        self.assertEqual(result["global_notes"], "заметка")


class ResolveSemanticGuardAdversarialTest(unittest.TestCase):
    """ТЗ §12a: the guard checks attacked one by one, on their own inputs."""

    def setUp(self) -> None:
        self.service = ResolveService(semantic_settings())

    def _code(self, baseline, final, **kwargs):
        return self.service._semantic_guard_reset_code(
            guard_projection(**kwargs), "manager", baseline, final
        )

    def test_swapping_two_numbers_is_a_numeric_change(self) -> None:
        code = self._code(
            "скидка 30 процентов и рассрочка на 20 месяцев",
            "скидка 20 процентов и рассрочка на 30 месяцев",
            variant_b="скидка 20 процентов и рассрочка на 30 месяцев",
        )

        self.assertEqual(code, "numeric_change")

    def test_a_spelled_out_number_is_frozen_like_a_digit(self) -> None:
        code = self._code(
            "рассрочка на сорок семь месяцев доступна",
            "рассрочка на сорок восемь месяцев доступна",
            variant_b="рассрочка на сорок восемь месяцев доступна",
        )

        self.assertEqual(code, "numeric_change")

    def test_a_sum_unit_is_frozen_too(self) -> None:
        code = self._code(
            "оплата принимается рублями в кассе центра",
            "оплата принимается процентами в кассе центра",
            variant_b="оплата принимается процентами в кассе центра",
        )

        self.assertEqual(code, "numeric_change")

    def test_a_moved_negation_does_not_pass_by_keeping_the_count(self) -> None:
        code = self._code(
            "я не отправлю договор сегодня вечером",
            "я отправлю договор не сегодня вечером",
            variant_b="я отправлю договор не сегодня вечером",
        )

        self.assertEqual(code, "negation_frozen_turn")

    def test_an_added_negation_is_refused_as_well(self) -> None:
        code = self._code(
            "мы отправим договор сегодня вечером",
            "мы не отправим договор сегодня вечером",
            variant_b="мы не отправим договор сегодня вечером",
        )

        self.assertEqual(code, "negation_frozen_turn")

    def test_a_glued_negation_is_refused_too(self) -> None:
        code = self._code(
            "к сожалению это невозможно исправить в договоре",
            "к сожалению это возможно исправить в договоре",
            variant_b="к сожалению это возможно исправить в договоре",
        )

        self.assertEqual(code, "negation_frozen_turn")

    def test_a_dropped_qualifier_is_refused(self) -> None:
        code = self._code(
            "договор придет примерно в субботу вечером на почту после обеда",
            "договор придет в субботу вечером на почту после обеда",
            variant_b="договор придет в субботу вечером на почту после обеда",
        )

        self.assertEqual(code, "qualifier_removed")

    def test_a_month_is_frozen_like_a_number(self) -> None:
        code = self._code(
            "встреча назначена на пятого января в центре",
            "встреча назначена на пятого февраля в центре",
            variant_b="встреча назначена на пятого февраля в центре",
        )

        self.assertEqual(code, "numeric_change")

    def test_a_currency_is_frozen_like_a_number(self) -> None:
        code = self._code(
            "перевод на триста долларов уже поступил",
            "перевод на триста евро уже поступил",
            variant_b="перевод на триста евро уже поступил",
        )

        self.assertEqual(code, "numeric_change")

    def test_ordinary_words_are_not_numbers(self) -> None:
        # The old prefix regex read дверь, семья and семестр as numerals and froze
        # every turn holding them; the exact forms below are the closed class.
        for word in ("дверь", "семья", "семя", "семинар", "семестр", "сотрудник", "сотовый",
                     "пятно", "стая", "however", "стоимость", "договор"):
            self.assertIsNone(resolve_module.SEMANTIC_NUMERAL_RE.fullmatch(word), word)
        for word in ("сорок", "семь", "восемь", "пятого", "триста", "рублей",
                     "рублями", "долларов", "евро", "процентами", "первый"):
            self.assertIsNotNone(resolve_module.SEMANTIC_NUMERAL_RE.fullmatch(word), word)

        for word in ("летний", "деньги", "годовой", "именно"):
            self.assertIsNone(resolve_module.SEMANTIC_FACT_RE.fullmatch(word), word)
            self.assertIsNone(resolve_module.SEMANTIC_NAME_CUE_RE.fullmatch(word), word)
        for word in ("января", "пятницу", "утром", "неделю", "месяцев", "лет"):
            self.assertIsNotNone(resolve_module.SEMANTIC_FACT_RE.fullmatch(word), word)

    def test_a_turn_without_a_fact_is_still_editable(self) -> None:
        code = self._code(
            "мы обсудим ваш семинар и ответим на вопросы",
            "мы обсудим ваш семинар и ответим на вопрос",
            variant_b="мы обсудим ваш семинар и ответим на вопрос",
        )

        self.assertIsNone(code)

    def test_a_lowercased_name_is_refused(self) -> None:
        code = self._code(
            "Анна отправит договор на почту",
            "анна вышлет договор на почту",
            variant_b="анна вышлет договор на почту",
        )

        self.assertEqual(code, "proper_name_change")

    def test_a_quoted_replacement_name_is_refused(self) -> None:
        code = self._code(
            "Анна отправит договор на почту",
            "«Ольга» вышлет договор на почту",
            variant_b="Ольга вышлет договор на почту",
        )

        self.assertEqual(code, "punctuation_change")

    def test_a_hyphenated_second_name_is_refused(self) -> None:
        code = self._code(
            "Анна отправит договор на почту",
            "Анна-Ольга отправит договор на почту",
            variant_b="Ольга отправит договор на почту",
        )

        self.assertEqual(code, "punctuation_change")

    def test_a_name_cue_freezes_the_turn(self) -> None:
        code = self._code(
            "меня зовут Анна я отправлю договор",
            "меня зовут Анна я вышлю договор",
            variant_b="меня зовут Анна я вышлю договор",
        )

        self.assertEqual(code, "name_cue_frozen_turn")

    def test_a_pure_word_reorder_is_refused(self) -> None:
        code = self._code(
            "договор придет вам на почту после оплаты",
            "после оплаты договор придет вам на почту",
            variant_b="после оплаты договор придет вам на почту",
        )

        self.assertEqual(code, "order_change")

    def test_losing_a_third_of_the_words_is_refused(self) -> None:
        baseline = "договор придет вам на почту после оплаты и подтверждения заявки"
        code = self._code(baseline, "договор придет вам на почту после оплаты", variant_b=baseline)

        self.assertEqual(code, "length_loss_reset")

    def test_a_far_word_pair_is_refused_even_when_the_variant_holds_it(self) -> None:
        for baseline, final in (
            ("все прошло хорошо и клиент остался доволен", "все прошло плохо и клиент остался доволен"),
            ("мы стараемся закрыть заявку в этот раз", "мы гарантируем закрыть заявку в этот раз"),
            ("передайте пожалуйста марине эти документы", "передайте пожалуйста ирине эти документы"),
            ("передайте пожалуйста сергею эти документы", "передайте пожалуйста андрею эти документы"),
            ("курс включает практические занятия", "курс исключает практические занятия"),
            ("участие платно для всех учеников", "участие бесплатно для всех учеников"),
            ("нужно оплатить обучение до начала", "нужно доплатить обучение до начала"),
        ):
            self.assertEqual(self._code(baseline, final, variant_b=final), "token_distance_reset", final)

    def test_an_inserted_word_is_refused_by_the_edit_shape(self) -> None:
        baseline = "договор придет вам на почту после оплаты и подтверждения заявки"
        final = "договор придет вам на почту сразу после оплаты и подтверждения заявки"

        self.assertEqual(self._code(baseline, final, variant_b=final), "edit_shape_rejected")

    def test_a_phrase_carried_over_from_another_turn_is_refused(self) -> None:
        baseline = "договор придет вам на почту после оплаты"
        final = "договор придет вам на почту после оплаты и подтверждения заявки"

        self.assertEqual(self._code(baseline, final, variant_b=final), "length_growth_reset")

    def test_the_glossary_alias_is_replaced_and_nothing_else_is(self) -> None:
        glossary = [{"alias": "МПК МФТИ", "canonical": "УНПК МФТИ",
                     "rule_id": "brand.unpk_mfti.known_alias"}]

        self.assertIsNone(self._code(
            "Мы ждем вас в центре МПК МФТИ на консультации",
            "Мы ждем вас в центре УНПК МФТИ на консультации",
            glossary=glossary,
        ))
        # The same turn may not carry a second edit under the dictionary exception: without
        # it the alias itself is a lost baseline capital, which is exactly a name change.
        self.assertEqual(self._code(
            "Мы ждем вас в центре МПК МФТИ на консультации",
            "Мы ждем вас в центре УНПК МФТИ на консультацию",
            glossary=glossary,
            variant_b="УНПК МФТИ на консультацию",
        ), "proper_name_change")
        self.assertEqual(self._code(
            "Анна, это МПК МФТИ, звоню по заявке.",
            "анна, это УНПК МФТИ, звоню по заявке.",
            glossary=glossary,
        ), "proper_name_change")
        self.assertEqual(self._code(
            "Анна, это МПК МФТИ, звоню по заявке.",
            "Анна это УНПК МФТИ. звоню по заявке?",
            glossary=glossary,
        ), "punctuation_change")

    def test_the_glossary_alias_works_in_its_own_declension(self) -> None:
        code = self._code(
            "приглашаем ребенка на летнюю ночную школу в центре",
            "приглашаем ребенка на летнюю очную школу в центре",
            glossary=[{"alias": "летнюю ночную школу", "canonical": "летнюю очную школу",
                       "rule_id": "product.summer_school.asr_artifact"}],
        )

        self.assertIsNone(code)

    def test_replacing_every_word_by_a_close_one_still_fails_on_support(self) -> None:
        # The second floor: shape and spelling pass, multiset support does not.
        baseline, final = "договоры уточняли пометки", "договора уточнили пометке"

        self.assertEqual(self._code(baseline, final, variant_b=final), "low_support_reset")
        with patch.object(resolve_module, "SEMANTIC_GUARD_MIN_SUPPORT", 0.0):
            self.assertIsNone(self._code(baseline, final, variant_b=final))

    def test_a_supported_single_word_fix_survives_every_check(self) -> None:
        code = self._code(
            "договор отправляем вам на почту после подтверждения",
            "договор отправляет вам на почту после подтверждения",
            variant_b="договор отправляет вам на почту после подтверждения",
        )

        self.assertIsNone(code)


class ResolveSemanticMergeCallTest(unittest.TestCase):
    """ТЗ §7: one call per escalated call, one fallback for every accident."""

    def _stereo_call(self, td):
        export_dir = Path(td) / "transcripts"
        path = export_dir / "calls" / "a_text.txt"
        path.parent.mkdir(parents=True)
        path.write_text(
            f"[00:01.0] Менеджер: {MANAGER_TURN_1}\n"
            f"[00:02.0] Клиент: {CLIENT_TURN_1}\n"
            f"[00:03.0] Менеджер: {MANAGER_TURN_2}\n"
            f"[00:04.0] Клиент: {CLIENT_TURN_2}\n"
            f"[00:05.0] Менеджер: {MANAGER_TURN_3}\n",
            encoding="utf-8",
        )
        payload = {
            "mode": "stereo",
            "call_topology": "simple_two_party",
            "role_mapping": {"confirmed": True, "manager_quality_allowed": True},
            "manager": {
                "physical_channel": "left",
                "variant_a": MANAGER_A,
                "variant_b": MANAGER_B,
                "final": MANAGER_A,
            },
            "client": {
                "physical_channel": "right",
                "variant_a": CLIENT_SAME,
                "variant_b": CLIENT_SAME,
                "final": CLIENT_SAME,
            },
        }
        call = CallRecord(
            source_file="calls/a.mp3",
            source_filename="a.mp3",
            transcript_manager=MANAGER_A,
            transcript_client=CLIENT_SAME,
            transcript_variants_json=json.dumps(payload, ensure_ascii=False),
        )
        return call, payload, export_dir

    def _service(self, export_dir, **overrides):
        return ResolveService(semantic_settings(
            transcript_export_dir=str(export_dir), **overrides
        ))

    def test_both_diverging_sides_travel_in_one_single_call(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_one_call_") as td:
            call, payload, export_dir = self._stereo_call(td)
            payload["client"]["variant_b"] = MANAGER_B
            call.transcript_variants_json = json.dumps(payload, ensure_ascii=False)
            service = self._service(export_dir)
            seen = []

            def runner(request, *, selective=False):
                seen.append((request, selective))
                return model_turns(request)

            service._run_dialogue_llm = runner  # type: ignore[method-assign]

            candidate = service._resolve_dialogue_with_llm(call, payload)

            self.assertEqual(len(seen), 1)
            self.assertTrue(seen[0][1])
            self.assertEqual(seen[0][0]["editable_roles"], ["manager", "client"])
            # Nothing survived the guard, so the call still ends on baseline.
            self.assertIsNone(candidate)
            self.assertEqual(
                service._semantic_merge_last["fallback_reason"], "no_accepted_edits"
            )

    def test_matching_variants_reach_no_model_at_all(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_no_call_") as td:
            call, payload, export_dir = self._stereo_call(td)
            payload["manager"]["variant_b"] = MANAGER_A
            call.transcript_variants_json = json.dumps(payload, ensure_ascii=False)
            service = self._service(export_dir)
            state = {"calls": 0}
            pair_calls = []

            def runner(request, **kwargs):
                state["calls"] += 1
                return model_turns(request)

            service._run_dialogue_llm = runner  # type: ignore[method-assign]
            service._merge_pair_with_llm = lambda **kw: pair_calls.append(kw)

            self.assertIsNone(service._resolve_with_llm(call, payload))
            self.assertEqual(state["calls"], 0)
            self.assertEqual(pair_calls, [])
            self.assertFalse(service._semantic_merge_last["escalated"])

    def test_a_failed_call_is_baseline_and_never_a_per_role_merge(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_fallback_") as td:
            call, payload, export_dir = self._stereo_call(td)
            service = self._service(export_dir)

            def boom(request, *, selective=False):
                raise RuntimeError("codex exec failed rc=1: timeout")

            pair_calls = []
            service._run_dialogue_llm = boom  # type: ignore[method-assign]
            service._merge_pair_with_llm = lambda **kw: pair_calls.append(kw)

            self.assertIsNone(service._resolve_with_llm(call, payload))
            self.assertEqual(pair_calls, [])
            reason = service._semantic_merge_last["fallback_reason"]
            # safe_error_text: the stage, the class and a digest — never the message.
            self.assertTrue(reason.startswith("resolve_semantic_merge: RuntimeError:"))
            self.assertNotIn("codex exec failed", reason)
            self.assertFalse(service._semantic_merge_last["applied"])

    def test_a_broken_turn_id_set_is_baseline_too(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_turnids_") as td:
            call, payload, export_dir = self._stereo_call(td)
            service = self._service(export_dir)
            pair_calls = []
            service._run_dialogue_llm = lambda request, **k: {
                "turns": [{"turn_id": 1, "final_text": MANAGER_TURN_1}]
            }
            service._merge_pair_with_llm = lambda **kw: pair_calls.append(kw)

            self.assertIsNone(service._resolve_with_llm(call, payload))
            self.assertEqual(pair_calls, [])
            self.assertTrue(
                service._semantic_merge_last["fallback_reason"].startswith(
                    "resolve_semantic_merge: RuntimeError:"
                )
            )

    def test_mono_never_reaches_the_model_under_selective(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_mono_") as td:
            call, payload, export_dir = self._stereo_call(td)
            payload["mode"] = "mono_or_fallback"
            payload["full"] = {"variant_a": MANAGER_A, "variant_b": MANAGER_B, "final": MANAGER_A}
            service = self._service(export_dir)
            pair_calls = []
            service._merge_pair_with_llm = lambda **kw: pair_calls.append(kw)

            self.assertIsNone(service._resolve_with_llm(call, payload))
            self.assertEqual(pair_calls, [])
            self.assertIsNone(service._semantic_merge_last)

    def test_more_than_half_the_edits_reset_drops_the_whole_candidate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_reject_") as td:
            call, payload, export_dir = self._stereo_call(td)
            service = self._service(export_dir)
            service._run_dialogue_llm = lambda request, **k: model_turns(
                request,
                {
                    1: MANAGER_TURN_1.replace("Анна", "Ольга"),
                    2: CLIENT_TURN_1.replace("центре", "центрах"),
                },
            )

            candidate = service._resolve_dialogue_with_llm(call, payload)

            block = service._semantic_merge_last
            self.assertIsNone(candidate)
            self.assertEqual(block["fallback_reason"], "reject_rate_exceeded")
            self.assertEqual(block["turns_changed_proposed"], 2)
            self.assertEqual(block["turns_changed_accepted"], 0)

    def test_an_accepted_edit_reaches_the_candidate_and_the_telemetry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_applied_") as td:
            call, payload, export_dir = self._stereo_call(td)
            service = self._service(export_dir)
            service._run_dialogue_llm = lambda request, **k: model_turns(
                request, {5: MANAGER_TURN_3_FIXED}
            )

            candidate = service._resolve_dialogue_with_llm(call, payload)

            self.assertIsNotNone(candidate)
            self.assertIn(MANAGER_TURN_3_FIXED, candidate["dialogue_lines"][4])
            self.assertIn(CLIENT_TURN_1, candidate["dialogue_lines"][1])
            self.assertEqual(len(candidate["dialogue_lines"]), 5)
            self.assertTrue(candidate["dialogue_lines"][0].startswith("[00:01.0] Менеджер:"))
            self.assertTrue(candidate["dialogue_lines"][1].startswith("[00:02.0] Клиент:"))
            stored = json.loads(candidate["transcript_variants_json"])
            self.assertFalse(stored["role_mapping"]["confirmed"])
            self.assertFalse(stored["role_mapping"]["manager_quality_allowed"])
            self.assertEqual(stored["role_mapping"]["status"], "mutable_sidecar_timing")
            self.assertEqual(stored["manager"]["variant_a"], MANAGER_A)
            self.assertEqual(stored["manager"]["variant_b"], MANAGER_B)
            block = service._semantic_merge_last
            self.assertIsNone(block["fallback_reason"])
            self.assertEqual(block["turns_changed_accepted"], 1)
            # Built is not applied: only _choose_best() decides that (ТЗ §12a).
            self.assertFalse(block["applied"])

    def test_an_error_inside_the_gate_is_baseline_and_not_a_failed_stage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_gate_boom_") as td:
            call, payload, export_dir = self._stereo_call(td)
            service = self._service(export_dir)
            pair_calls = []
            service._merge_pair_with_llm = lambda **kw: pair_calls.append(kw)
            service._run_dialogue_llm = lambda request, **k: model_turns(request)

            def boom(_variants, _tenant):
                raise RuntimeError("normalizer ruleset unavailable")

            service._semantic_glossary = boom  # type: ignore[method-assign]

            self.assertIsNone(service._resolve_with_llm(call, payload))
            self.assertEqual(pair_calls, [])
            self.assertTrue(
                service._semantic_merge_last["fallback_reason"].startswith(
                    "resolve_semantic_merge: RuntimeError:"
                )
            )

    def test_a_broken_candidate_build_is_baseline_and_not_a_failed_stage(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_build_boom_") as td:
            call, payload, export_dir = self._stereo_call(td)
            service = self._service(export_dir)
            pair_calls = []
            service._merge_pair_with_llm = lambda **kw: pair_calls.append(kw)
            service._run_dialogue_llm = lambda request, **k: model_turns(
                request, {5: MANAGER_TURN_3_FIXED}
            )

            def boom(*_args, **_kwargs):
                raise RuntimeError("candidate assembly failed")

            # The fail-soft covers everything after the gate, not only the model call.
            service._dialogue_turns_to_candidate = boom  # type: ignore[method-assign]

            self.assertIsNone(service._resolve_with_llm(call, payload))
            self.assertEqual(pair_calls, [])
            self.assertTrue(
                service._semantic_merge_last["fallback_reason"].startswith(
                    "resolve_semantic_merge: RuntimeError:"
                )
            )

    def test_a_gate_that_dies_before_its_state_still_ends_on_baseline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_gate_dead_") as td:
            call, payload, export_dir = self._stereo_call(td)
            service = self._service(export_dir)
            pair_calls = []
            service._merge_pair_with_llm = lambda **kw: pair_calls.append(kw)

            def boom(_payload):
                raise RuntimeError("gate math failed")

            service._semantic_selective_input = boom  # type: ignore[method-assign]

            self.assertIsNone(service._resolve_with_llm(call, payload))
            self.assertEqual(pair_calls, [])
            self.assertIsNone(service._semantic_merge_last)

    def test_the_cache_answers_the_repeat_and_a_retuned_threshold_does_not(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_cache_") as td:
            call, payload, export_dir = self._stereo_call(td)
            service = self._service(
                export_dir, llm_cache_enabled=True, llm_cache_dir=str(Path(td) / "cache")
            )
            answer = model_answer({5: MANAGER_TURN_3_FIXED})
            state = {"calls": 0}

            def fake_run(cmd, capture_output, text, check, timeout):
                state["calls"] += 1
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                out_path.write_text(json.dumps(answer, ensure_ascii=False), encoding="utf-8")
                return CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("mango_mvp.services.resolve.shutil.which", return_value="/usr/bin/codex"):
                with patch("mango_mvp.services.resolve.subprocess.run", side_effect=fake_run):
                    self.assertIsNotNone(service._resolve_dialogue_with_llm(call, payload))
                    first = service._semantic_merge_last
                    self.assertEqual(first["model_calls"], 1)
                    self.assertFalse(first["cache_hit"])

                    self.assertIsNotNone(service._resolve_dialogue_with_llm(call, payload))
                    repeat = service._semantic_merge_last
                    self.assertEqual(state["calls"], 1)
                    self.assertTrue(repeat["cache_hit"])
                    self.assertEqual(repeat["model_calls"], 0)

                    with patch.object(resolve_module, "SEMANTIC_GUARD_MIN_SUPPORT", 0.55):
                        self.assertIsNotNone(
                            service._resolve_dialogue_with_llm(call, payload)
                        )
                    self.assertEqual(state["calls"], 2)
                    self.assertFalse(service._semantic_merge_last["cache_hit"])

    def test_the_selective_prompt_and_its_own_reasoning_depth_reach_the_cli(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_prompt_") as td:
            call, payload, export_dir = self._stereo_call(td)
            service = self._service(export_dir, codex_resolve_reasoning_effort="high")
            seen = {}

            def fake_run(cmd, capture_output, text, check, timeout):
                seen["cmd"] = list(cmd)
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                out_path.write_text(
                    json.dumps(model_answer(), ensure_ascii=False), encoding="utf-8"
                )
                return CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("mango_mvp.services.resolve.shutil.which", return_value="/usr/bin/codex"):
                with patch("mango_mvp.services.resolve.subprocess.run", side_effect=fake_run):
                    service._resolve_dialogue_with_llm(call, payload)

            self.assertIn('model_reasoning_effort="high"', seen["cmd"])
            self.assertIn(resolve_module.RESOLVE_EDIT_SYSTEM_PROMPT, seen["cmd"][-1])
            self.assertNotIn('"call_id"', seen["cmd"][-1])
            self.assertNotIn('"manager_name"', seen["cmd"][-1])


class ResolveSemanticMergeDefaultTest(unittest.TestCase):
    """ТЗ §10: with the default mode nothing above exists."""

    def test_the_default_mode_is_off_and_adds_no_block_to_resolve_json(self) -> None:
        service = ResolveService(make_settings())

        self.assertFalse(service._semantic_merge_selective())
        payload = service._build_resolve_payload(
            duration_sec=1.0,
            decision="accept_baseline",
            baseline={"quality": {"score": 90, "reasons": []}},
            llm_candidate=None,
            rescue_candidate=None,
            chosen=None,
        )

        self.assertNotIn("semantic_merge", payload)

    def test_the_selective_block_is_added_only_when_the_mode_ran(self) -> None:
        service = ResolveService(semantic_settings())
        service._semantic_selective_input(semantic_input())

        payload = service._build_resolve_payload(
            duration_sec=1.0,
            decision="accept_baseline",
            baseline={"quality": {"score": 90, "reasons": []}},
            llm_candidate=None,
            rescue_candidate=None,
            chosen=None,
        )

        self.assertEqual(payload["semantic_merge"]["mode"], "selective")

    def test_applied_follows_the_chosen_candidate_and_nothing_else(self) -> None:
        service = ResolveService(semantic_settings())
        service._semantic_selective_input(semantic_input())
        service._semantic_merge_last["turns_changed_accepted"] = 1
        common = dict(
            duration_sec=1.0,
            baseline={"quality": {"score": 90, "reasons": []}},
            llm_candidate={"quality": {"score": 91, "reasons": []}, "meta": {}},
            rescue_candidate=None,
        )

        lost = service._build_resolve_payload(
            decision="accept_baseline", chosen={"name": "baseline", "quality": {"score": 90}}, **common
        )
        won = service._build_resolve_payload(
            decision="accept_llm", chosen={"name": "llm", "quality": {"score": 91}}, **common
        )
        # Won the comparison but scored below the acceptance threshold: nothing is written,
        # so nothing is applied either (ТЗ §12b).
        manual = service._build_resolve_payload(
            decision="manual_review_required", chosen={"name": "llm", "quality": {"score": 40}}, **common
        )

        self.assertFalse(lost["semantic_merge"]["applied"])
        self.assertTrue(won["semantic_merge"]["applied"])
        self.assertFalse(manual["semantic_merge"]["applied"])

    def test_the_worker_hands_a_transport_only_to_the_selective_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_env_") as td:
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

            # Both new variables are stated by the worker, never inherited silently, so
            # the ambient shell of the test host cannot decide the default.
            with patch.dict(
                "os.environ",
                {"RESOLVE_SEMANTIC_MERGE_MODE": "", "CODEX_RESOLVE_REASONING_EFFORT": ""},
            ):
                env = worker_environment(config)
            self.assertEqual(env["RESOLVE_LLM_PROVIDER"], "off")
            self.assertEqual(env["RESOLVE_SEMANTIC_MERGE_MODE"], "off")
            self.assertEqual(env["CODEX_RESOLVE_REASONING_EFFORT"], "medium")

            with patch.dict(
                "os.environ",
                {"RESOLVE_SEMANTIC_MERGE_MODE": "Selective",
                 "CODEX_RESOLVE_REASONING_EFFORT": "High"},
            ):
                env = worker_environment(config)
            self.assertEqual(env["RESOLVE_LLM_PROVIDER"], "codex_cli")
            self.assertEqual(env["RESOLVE_SEMANTIC_MERGE_MODE"], "selective")
            self.assertEqual(env["CODEX_RESOLVE_REASONING_EFFORT"], "high")

    def test_the_mode_and_the_resolve_depth_come_from_the_environment(self) -> None:
        config_module.get_settings.cache_clear()
        try:
            with patch.dict(
                "os.environ",
                {
                    "RESOLVE_SEMANTIC_MERGE_MODE": "Selective",
                    "CODEX_RESOLVE_REASONING_EFFORT": "High",
                },
            ):
                settings = config_module.get_settings()
            self.assertEqual(settings.resolve_semantic_merge_mode, "selective")
            self.assertEqual(settings.codex_resolve_reasoning_effort, "high")
            config_module.get_settings.cache_clear()
            with patch.dict(
                "os.environ",
                {"RESOLVE_SEMANTIC_MERGE_MODE": "", "CODEX_RESOLVE_REASONING_EFFORT": ""},
            ):
                defaults = config_module.get_settings()
            self.assertEqual(defaults.resolve_semantic_merge_mode, "off")
            self.assertEqual(defaults.codex_resolve_reasoning_effort, "medium")
        finally:
            config_module.get_settings.cache_clear()


class ResolveSemanticMergeLiveRunTest(unittest.TestCase):
    """ТЗ §12b: the same mechanism through ResolveService.run(), not only its helpers."""

    DIALOGUE = (
        f"[00:01.0] Менеджер: {MANAGER_TURN_1}\n"
        f"[00:02.0] Клиент: {CLIENT_TURN_1}\n"
        f"[00:03.0] Менеджер: {MANAGER_TURN_2}\n"
        f"[00:04.0] Клиент: {CLIENT_TURN_2}\n"
        f"[00:05.0] Менеджер: {MANAGER_TURN_3}\n"
    )

    def _prepare(self, td, **overrides):
        export_dir = Path(td) / "export"
        settings = replace(make_settings(), **{
            "database_url": f"sqlite:///{Path(td) / 'run.db'}",
            "resolve_llm_provider": "codex_cli",
            "resolve_semantic_merge_mode": "selective",
            "resolve_rescue_provider": "none",
            "llm_cache_enabled": False,
            "transcript_export_dir": str(export_dir),
            **overrides,
        })
        init_db(settings)
        session_factory = build_session_factory(settings)
        with session_factory() as session:
            # Two calls in one run: the first diverges between the two ASR variants, the
            # second is heard identically twice and must reach no model at all.
            for name, manager_b in (("a", MANAGER_B), ("b", MANAGER_A)):
                path = export_dir / "calls" / f"{name}_text.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(self.DIALOGUE, encoding="utf-8")
                session.add(CallRecord(
                    source_call_id=f"call-{name}",
                    source_file=f"calls/{name}.mp3",
                    source_filename=f"{name}.mp3",
                    duration_sec=120.0,
                    transcription_status="done",
                    resolve_status="pending",
                    analysis_status="pending",
                    transcript_text=f"MANAGER:\n{MANAGER_A}\n\nCLIENT:\n{CLIENT_SAME}",
                    transcript_manager=MANAGER_A,
                    transcript_client=CLIENT_SAME,
                    transcript_variants_json=json.dumps({
                        "mode": "stereo",
                        "call_topology": "simple_two_party",
                        "role_mapping": {"confirmed": True, "manager_quality_allowed": True},
                        "manager": {"physical_channel": "left", "variant_a": MANAGER_A,
                                    "variant_b": manager_b, "final": MANAGER_A},
                        "client": {"physical_channel": "right", "variant_a": CLIENT_SAME,
                                   "variant_b": CLIENT_SAME, "final": CLIENT_SAME},
                    }, ensure_ascii=False),
                ))
            session.commit()
        return settings, session_factory

    @staticmethod
    def _blocks(session_factory):
        with session_factory() as session:
            return {
                call.source_call_id: json.loads(call.resolve_json or "{}")
                for call in session.query(CallRecord).all()
            }

    def test_a_run_escalates_only_the_diverging_call_and_leaks_nothing(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_run_") as td:
            settings, session_factory = self._prepare(td)
            service = ResolveService(settings)
            seen = []

            def fake_run(cmd, capture_output, text, check, timeout):
                request = json.loads(cmd[-1].split("Call dialogue payload JSON:\n", 1)[1])
                seen.append(request)
                out_path = Path(cmd[cmd.index("--output-last-message") + 1])
                out_path.write_text(
                    json.dumps(model_turns(request, {5: MANAGER_TURN_3_FIXED}), ensure_ascii=False),
                    encoding="utf-8",
                )
                return CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("mango_mvp.services.resolve.shutil.which", return_value="/usr/bin/codex"):
                with patch("mango_mvp.services.resolve.subprocess.run", side_effect=fake_run):
                    with session_factory() as session:
                        result = service.run(session, limit=2)

            self.assertEqual(len(seen), 1)
            self.assertEqual(result["semantic_eligible"], 2)
            self.assertEqual(result["semantic_escalated"], 1)
            self.assertEqual(result["semantic_model_calls"], 1)
            self.assertEqual(result["semantic_turns_accepted"], 1)
            self.assertEqual(result["semantic_cache_hit"], 0)

            payloads = self._blocks(session_factory)
            diverging = payloads["call-a"]["semantic_merge"]
            identical = payloads["call-b"]["semantic_merge"]
            self.assertEqual(diverging["model_calls"], 1)
            self.assertEqual(diverging["turns_changed_accepted"], 1)
            self.assertEqual(diverging["applied"], payloads["call-a"]["decision"] == "accept_llm")
            # The second call carries its own counters, not the first call's ones.
            self.assertEqual(identical["model_calls"], 0)
            self.assertEqual(identical["turns_changed_accepted"], 0)
            self.assertFalse(identical["escalated"])
            self.assertIsNone(identical["fallback_reason"])
            self.assertEqual(identical["turns_reset"], {})

    def test_stale_candidates_do_not_enter_semantic_run_totals(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_stale_") as td:
            settings, session_factory = self._prepare(td)
            service = ResolveService(settings)

            def fake_run(cmd, capture_output, text, check, timeout):
                request = json.loads(cmd[-1].split("Call dialogue payload JSON:\n", 1)[1])
                Path(cmd[cmd.index("--output-last-message") + 1]).write_text(
                    json.dumps(model_turns(request, {5: MANAGER_TURN_3_FIXED}), ensure_ascii=False),
                    encoding="utf-8",
                )
                return CompletedProcess(cmd, 0, stdout="", stderr="")

            with patch("mango_mvp.services.resolve.shutil.which", return_value="/usr/bin/codex"), \
                    patch("mango_mvp.services.resolve.subprocess.run", side_effect=fake_run), \
                    patch.object(service, "_candidate_source_is_current", return_value=False):
                with session_factory() as session:
                    result = service.run(session, limit=2)

            self.assertEqual(result["stale"], 2)
            self.assertFalse(any(key.startswith("semantic_") for key in result))

    def test_the_default_run_calls_no_model_and_writes_no_block(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_semantic_run_off_") as td:
            settings, session_factory = self._prepare(
                td, resolve_semantic_merge_mode="off", resolve_llm_provider="off"
            )
            service = ResolveService(settings)

            def runner(request, **_kwargs):
                raise AssertionError("the default mode may not reach a model")

            service._run_dialogue_llm = runner  # type: ignore[method-assign]
            with session_factory() as session:
                result = service.run(session, limit=2)

            self.assertNotIn("semantic_model_calls", result)
            self.assertNotIn("semantic_escalated", result)
            for payload in self._blocks(session_factory).values():
                self.assertNotIn("semantic_merge", payload)


if __name__ == "__main__":
    unittest.main()
