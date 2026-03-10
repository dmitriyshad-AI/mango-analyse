from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.services.resolve import ResolveService
from tests.test_dialogue_format import make_settings


class ResolveServiceTest(unittest.TestCase):
    def test_same_ts_postfilter_adjusts_cross_speaker_timecodes(self) -> None:
        service = ResolveService(make_settings())
        lines = [
            "[00:10.0] Менеджер (Иван): Добрый день.",
            "[00:10.0] Клиент: Здравствуйте.",
            "[00:11.5] Менеджер (Иван): Подскажите класс.",
        ]
        fixed = service._postfilter_same_ts_dialogue_lines(lines)
        self.assertEqual(int(fixed["adjusted"]), 1)
        out_lines = fixed["dialogue_lines"]
        self.assertIn("[00:10.0] Менеджер (Иван):", out_lines[0])
        self.assertIn("[00:10.1] Клиент:", out_lines[1])

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
                resolve_llm_provider="off",
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


if __name__ == "__main__":
    unittest.main()
