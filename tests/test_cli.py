from __future__ import annotations

import io
import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from mango_mvp.cli import cmd_export_pilot_bundle, cmd_prepare_resolve_pilot
from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from tests.test_dialogue_format import make_settings


class PrepareResolvePilotCliTest(unittest.TestCase):
    def test_prepare_resolve_pilot_selects_real_calls_only(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_prepare_resolve_pilot_") as td:
            db_path = Path(td) / "pilot.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                dual_transcribe_enabled=True,
                transcribe_provider="mlx",
                secondary_transcribe_provider="gigaam",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)

            with session_factory() as session:
                session.add_all(
                    [
                        CallRecord(
                            source_file=str(Path(td) / "2026-03-01__10-00-00__79990000000__Иванов Иван_1.mp3"),
                            source_filename="2026-03-01__10-00-00__79990000000__Иванов Иван_1.mp3",
                            duration_sec=180.0,
                            transcription_status="done",
                            resolve_status="done",
                            analysis_status="done",
                            sync_status="pending",
                            transcript_text="MANAGER:\nЗдравствуйте\n\nCLIENT:\nДа",
                            transcript_manager="Здравствуйте",
                            transcript_client="Да",
                            transcript_variants_json=json.dumps(
                                {
                                    "mode": "stereo",
                                    "primary_provider": "mlx",
                                    "secondary_provider": "gigaam",
                                    "manager": {
                                        "variant_a": "Здравствуйте",
                                        "variant_b": "Здравствуйте",
                                        "final": "Здравствуйте",
                                    },
                                    "client": {
                                        "variant_a": "Да",
                                        "variant_b": "Да",
                                        "final": "Да",
                                    },
                                },
                                ensure_ascii=False,
                            ),
                            analysis_json=json.dumps({"history_summary": "old"}, ensure_ascii=False),
                            resolve_json=json.dumps({"decision": "accept_baseline"}, ensure_ascii=False),
                            resolve_quality_score=88.0,
                        ),
                        CallRecord(
                            source_file=str(Path(td) / "test-8000Hz-le-1ch.wav"),
                            source_filename="test-8000Hz-le-1ch.wav",
                            duration_sec=180.0,
                            transcription_status="done",
                            resolve_status="done",
                            analysis_status="done",
                            transcript_text="test",
                            transcript_variants_json=json.dumps({"mode": "stereo"}, ensure_ascii=False),
                        ),
                        CallRecord(
                            source_file=str(Path(td) / "2026-03-02__10-00-00__79990000001__Петров Петр_2.mp3"),
                            source_filename="2026-03-02__10-00-00__79990000001__Петров Петр_2.mp3",
                            duration_sec=12.0,
                            transcription_status="done",
                            resolve_status="done",
                            analysis_status="done",
                            transcript_text="short",
                            transcript_variants_json=json.dumps(
                                {
                                    "mode": "stereo",
                                    "primary_provider": "mlx",
                                    "secondary_provider": "gigaam",
                                },
                                ensure_ascii=False,
                            ),
                        ),
                        CallRecord(
                            source_file=str(Path(td) / "2026-03-03__10-00-00__79990000002__Сидоров Сидор_3.mp3"),
                            source_filename="2026-03-03__10-00-00__79990000002__Сидоров Сидор_3.mp3",
                            duration_sec=180.0,
                            transcription_status="done",
                            resolve_status="done",
                            analysis_status="done",
                            transcript_text="needs second asr",
                            transcript_variants_json=json.dumps(
                                {
                                    "mode": "stereo",
                                    "primary_provider": "mlx",
                                    "secondary_provider": "gigaam",
                                    "manager": {
                                        "variant_a": "Здравствуйте",
                                        "variant_b": None,
                                        "final": "Здравствуйте",
                                    },
                                    "client": {
                                        "variant_a": "Да",
                                        "variant_b": None,
                                        "final": "Да",
                                    },
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    ]
                )
                session.commit()

            args = Namespace(
                limit=10,
                seed=7,
                statuses="done,manual",
                min_duration_sec=None,
                ids_in=None,
                ids_out=str(Path(td) / "pilot_ids.txt"),
                include_tests=False,
                dry_run=False,
            )

            with patch("mango_mvp.cli.get_settings", return_value=settings):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cmd_prepare_resolve_pilot(args)

            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["selected"], 1)
            self.assertEqual(payload["updated"], 1)
            self.assertEqual(payload["blocked_secondary"], 1)
            self.assertEqual(payload["skipped_tests"], 1)
            self.assertEqual(payload["skipped_short"], 1)
            self.assertEqual(len(payload["selected_ids"]), 1)

            with session_factory() as session:
                rows = session.query(CallRecord).order_by(CallRecord.id.asc()).all()
                real = rows[0]
                self.assertEqual(real.resolve_status, "pending")
                self.assertEqual(real.resolve_attempts, 0)
                self.assertEqual(real.analysis_status, "pending")
                self.assertIsNone(real.analysis_json)
                self.assertIsNone(real.resolve_json)
                self.assertIsNone(real.resolve_quality_score)
                self.assertEqual(rows[1].resolve_status, "done")
                self.assertEqual(rows[2].resolve_status, "done")
                self.assertEqual(rows[3].resolve_status, "done")

            ids_file = Path(td) / "pilot_ids.txt"
            self.assertTrue(ids_file.exists())
            self.assertTrue(ids_file.read_text(encoding="utf-8").strip())

    def test_prepare_resolve_pilot_can_reuse_id_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_prepare_resolve_ids_") as td:
            db_path = Path(td) / "pilot_ids.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)

            with session_factory() as session:
                session.add_all(
                    [
                        CallRecord(
                            id=101,
                            source_file=str(Path(td) / "2026-03-01__10-00-00__79990000000__Иванов Иван_1.mp3"),
                            source_filename="2026-03-01__10-00-00__79990000000__Иванов Иван_1.mp3",
                            duration_sec=180.0,
                            transcription_status="done",
                            resolve_status="done",
                            analysis_status="done",
                            transcript_text="ok",
                            transcript_variants_json=json.dumps({"mode": "stereo"}, ensure_ascii=False),
                        ),
                        CallRecord(
                            id=102,
                            source_file=str(Path(td) / "2026-03-01__10-00-00__79990000001__Петров Петр_2.mp3"),
                            source_filename="2026-03-01__10-00-00__79990000001__Петров Петр_2.mp3",
                            duration_sec=180.0,
                            transcription_status="failed",
                            resolve_status="done",
                            analysis_status="done",
                            transcript_text="bad",
                            transcript_variants_json=json.dumps({"mode": "stereo"}, ensure_ascii=False),
                        ),
                    ]
                )
                session.commit()

            ids_path = Path(td) / "ids.txt"
            ids_path.write_text("101\n102\n999\n", encoding="utf-8")

            args = Namespace(
                limit=10,
                seed=42,
                statuses="done,manual",
                min_duration_sec=30.0,
                ids_in=str(ids_path),
                ids_out=None,
                include_tests=False,
                dry_run=False,
            )

            with patch("mango_mvp.cli.get_settings", return_value=settings):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cmd_prepare_resolve_pilot(args)

            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["selected"], 1)
            self.assertEqual(payload["updated"], 1)
            self.assertEqual(payload["missing_ids"], [999])
            self.assertEqual(payload["skipped_ids"], [102])

    def test_export_pilot_bundle_writes_variants_and_merge_snapshot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_export_pilot_bundle_") as td:
            db_path = Path(td) / "pilot_export.db"
            export_dir = Path(td) / "transcripts"
            calls_dir = Path(td) / "calls"
            calls_dir.mkdir(parents=True, exist_ok=True)
            source_file = calls_dir / "2026-03-01__10-00-00__79990000000__Иванов Иван_1.mp3"
            source_file.write_bytes(b"")

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
                        id=201,
                        source_file=str(source_file),
                        source_filename=source_file.name,
                        manager_name="Иванов Иван",
                        duration_sec=180.0,
                        transcription_status="done",
                        resolve_status="done",
                        analysis_status="pending",
                        transcript_text="MANAGER:\nЗдравствуйте\n\nCLIENT:\nДа, слушаю",
                        transcript_manager="Здравствуйте",
                        transcript_client="Да, слушаю",
                        transcript_variants_json=json.dumps(
                            {
                                "mode": "stereo",
                                "primary_provider": "mlx",
                                "secondary_provider": "gigaam",
                                "manager": {
                                    "variant_a": "Здравствуйте",
                                    "variant_b": "Здравствуйте!",
                                    "final": "Здравствуйте",
                                },
                                "client": {
                                    "variant_a": "Да, слушаю",
                                    "variant_b": "Да, слушаю внимательно",
                                    "final": "Да, слушаю",
                                },
                            },
                            ensure_ascii=False,
                        ),
                        resolve_json=json.dumps({"decision": "accept_baseline"}, ensure_ascii=False),
                    )
                )
                session.commit()

            target_dir = export_dir / calls_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / f"{source_file.stem}_text.txt").write_text(
                "[00:01.0] Менеджер (Иванов Иван): Здравствуйте.\n[00:02.0] Клиент: Да, слушаю.\n",
                encoding="utf-8",
            )

            ids_path = Path(td) / "ids.txt"
            ids_path.write_text("201\n", encoding="utf-8")
            out_dir = Path(td) / "bundle"

            args = Namespace(
                ids_in=str(ids_path),
                out=str(out_dir),
                label="initial",
            )

            with patch("mango_mvp.cli.get_settings", return_value=settings):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cmd_export_pilot_bundle(args)

            self.assertEqual(rc, 0)
            raw = out.getvalue()
            payload = json.loads(raw[raw.rfind("{") :])
            self.assertEqual(payload["exported"], 1)
            call_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
            self.assertEqual(len(call_dirs), 1)
            call_dir = call_dirs[0]
            self.assertTrue((call_dir / "01_mlx.txt").exists())
            self.assertTrue((call_dir / "02_gigaam.txt").exists())
            self.assertTrue((call_dir / "03_initial_merge.txt").exists())
            self.assertTrue((call_dir / "metadata.json").exists())
            self.assertTrue((call_dir / "resolve.json").exists())
            mlx_text = (call_dir / "01_mlx.txt").read_text(encoding="utf-8")
            gigaam_text = (call_dir / "02_gigaam.txt").read_text(encoding="utf-8")
            merge_text = (call_dir / "03_initial_merge.txt").read_text(encoding="utf-8")
            self.assertIn("Менеджер (Иванов Иван)", mlx_text)
            self.assertIn("Да, слушаю внимательно", gigaam_text)
            self.assertIn("[00:01.0]", merge_text)
            self.assertTrue((out_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
