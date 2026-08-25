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

from mango_mvp.cli import _current_call_text, _provider_variants_for_export, cmd_export_pilot_bundle, cmd_prepare_resolve_pilot, cmd_sync
from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.services.dialogue_contract import DialogueContractError
from mango_mvp.services.sync_amocrm import LEGACY_AMOCRM_SYNC_DISABLED_MESSAGE
from mango_mvp.services.worker import normalize_pipeline_stages
from tests.test_dialogue_format import make_settings


class PrepareResolvePilotCliTest(unittest.TestCase):
    def test_invalid_provider_evidence_never_exports_business_role_labels(self) -> None:
        payload = {"mode": "stereo", "primary_provider": "mlx", "secondary_provider": "gigaam", "provider_role_evidence": {"provider": "broken"},
                   "role_mapping": {"status": "confirmed_multi_signal", "confirmed": True, "manager_quality_allowed": True, "topology": "simple_two_party", "left": "manager", "right": "client"},
                   "dialogue_lines": ["[00:01.0] Дорожка левая: LEFT CLIENT", "[00:02.0] Дорожка правая: RIGHT MANAGER"],
                   "manager": {"physical_channel": "left", "variant_a": "LEFT CLIENT", "variant_b": "LEFT B"}, "client": {"physical_channel": "right", "variant_a": "RIGHT MANAGER", "variant_b": "RIGHT B"}}
        call = CallRecord(source_call_id="call-a", source_recording_id="recording-a", source_file="a.mp3", source_filename="a.mp3", transcript_variants_json=json.dumps(payload, ensure_ascii=False))
        exported = "\n".join(_provider_variants_for_export(call).values())
        self.assertNotIn("Менеджер", exported); self.assertNotIn("Клиент:", exported)
        self.assertIn("Спикер A", exported)
        self.assertIn("Спикер A", _current_call_text(call, None))

    def test_untrusted_model_orientation_never_exports_business_role_labels(self) -> None:
        payload = {"mode": "stereo", "primary_provider": "mlx", "role_mapping": {
            "status": "confirmed_model_channel_orientation", "confirmed": True,
            "manager_quality_allowed": True, "topology": "simple_two_party",
            "left": "manager", "right": "client", "trust_source": "model_channel_orientation",
            "evidence": ["model_channel_orientation"], "confidence": 0.99,
            "model_orientation": {"provider": "codex_cli", "roles": ["manager", "client"],
                "confidence": 0.99, "ordinary_two_party": True}},
            "dialogue_lines": ["[00:01.0] Дорожка левая: LEFT", "[00:02.0] Дорожка правая: RIGHT"],
            "manager": {"physical_channel": "left", "variant_a": "LEFT"},
            "client": {"physical_channel": "right", "variant_a": "RIGHT"}}
        call = CallRecord(source_file="a.mp3", source_filename="a.mp3",
                          transcript_variants_json=json.dumps(payload, ensure_ascii=False))
        exported = "\n".join(_provider_variants_for_export(call).values())
        self.assertNotIn("Менеджер", exported)
        self.assertNotIn("Клиент:", exported)
        self.assertIn("Спикер A", exported)

    def test_broken_contract_neutralizes_legacy_role_headers(self) -> None:
        variants = {"mode": "stereo", "primary_provider": "mlx",
                    "role_mapping": {},
                    "manager": {"physical_channel": "left", "variant_a": "LEFT"},
                    "client": {"physical_channel": "right", "variant_a": "RIGHT"}}
        call = CallRecord(source_file="a.mp3", source_filename="a.mp3",
                          transcript_variants_json=json.dumps(variants),
                          transcript_text="[00:01.0] Менеджер:\nLEFT\n\n[00:02.0] Клиент:\nRIGHT")
        with patch("mango_mvp.cli.build_dialogue_input",
                   side_effect=DialogueContractError("broken")):
            text = _current_call_text(call, None)
            exported = "\n".join(_provider_variants_for_export(call).values())
        self.assertNotIn("MANAGER:", text)
        self.assertNotIn("CLIENT:", text)
        self.assertIn("Спикер A:", text)
        self.assertIn("Спикер B:", text)
        self.assertIn("Спикер A:", exported)

    def test_untrusted_export_uses_physical_sides_not_legacy_role_headers(self) -> None:
        payload = {
            "mode": "stereo", "primary_provider": "mlx", "role_mapping": {},
            "dialogue_lines": [
                "[00:01.0] Дорожка левая: LEFT CLIENT",
                "[00:02.0] Дорожка правая: RIGHT MANAGER",
            ],
            "manager": {"physical_channel": "right", "variant_a": "RIGHT MANAGER"},
            "client": {"physical_channel": "left", "variant_a": "LEFT CLIENT"},
        }
        call = CallRecord(
            source_file="a.mp3", source_filename="a.mp3",
            transcript_variants_json=json.dumps(payload, ensure_ascii=False),
            transcript_text="MANAGER:\nRIGHT MANAGER\n\nCLIENT:\nLEFT CLIENT",
        )
        text = _current_call_text(call, None)
        self.assertIn("Спикер A: LEFT CLIENT", text)
        self.assertIn("Спикер B: RIGHT MANAGER", text)

    def test_broken_variants_fall_back_to_both_legacy_text_columns(self) -> None:
        call = CallRecord(
            source_file="a.mp3", source_filename="a.mp3",
            transcript_variants_json="{", transcript_text="",
            transcript_manager="FIRST", transcript_client="SECOND",
        )
        text = _current_call_text(call, None)
        self.assertIn("Спикер A:\nFIRST", text)
        self.assertIn("Спикер B:\nSECOND", text)

    def test_stereo_without_role_mapping_never_exports_business_roles(self) -> None:
        payload = {"mode": "stereo", "primary_provider": "mlx",
                   "manager": {"physical_channel": "left", "variant_a": "LEFT"},
                   "client": {"physical_channel": "right", "variant_a": "RIGHT"}}
        call = CallRecord(source_file="a.mp3", source_filename="a.mp3",
                          transcript_variants_json=json.dumps(payload),
                          transcript_manager="LEFT", transcript_client="RIGHT")
        exported = "\n".join(_provider_variants_for_export(call).values())
        self.assertNotIn("Менеджер", exported)
        self.assertNotIn("Клиент:", exported)
        self.assertIn("Спикер A", exported)

    def test_invalid_variants_json_neutralizes_named_legacy_headers(self) -> None:
        for raw in ("{", "[]", "null", "{}", '{"mode":"unknown"}'):
            call = CallRecord(source_file="a.mp3", source_filename="a.mp3",
                              transcript_variants_json=raw,
                              transcript_text="[00:01.0] MANAGER (Иван): LEFT\n[00:02.0] Клиент (Анна): RIGHT")
            text = _current_call_text(call, None)
            self.assertIn("Спикер A:", text)
            self.assertIn("Спикер B:", text)
            self.assertNotIn("MANAGER", text)
            self.assertNotIn("Клиент", text)

    def test_duplicate_physical_side_keeps_both_neutral_tracks(self) -> None:
        payload = {"mode": "stereo", "primary_provider": "mlx",
                   "manager": {"physical_channel": "left", "variant_a": "FIRST TRACK"},
                   "client": {"physical_channel": "left", "variant_a": "SECOND TRACK"}}
        call = CallRecord(source_file="a.mp3", source_filename="a.mp3",
                          transcript_variants_json=json.dumps(payload))
        exported = "\n".join(_provider_variants_for_export(call).values())
        self.assertIn("Спикер A:\nFIRST TRACK", exported)
        self.assertIn("Спикер B:\nSECOND TRACK", exported)

    def test_trusted_dialogue_without_stored_physical_pair_keeps_both_tracks(self) -> None:
        payload = {"mode": "stereo", "primary_provider": "mlx",
                   "manager": {"variant_a": "FIRST TRACK"},
                   "client": {"variant_a": "SECOND TRACK"}}
        call = CallRecord(source_file="a.mp3", source_filename="a.mp3",
                          transcript_variants_json=json.dumps(payload))
        trusted = type("Dialogue", (), {"trusted": True, "turns": ()})()
        with patch("mango_mvp.cli.build_dialogue_input", return_value=trusted):
            exported = "\n".join(_provider_variants_for_export(call).values())
        self.assertIn("Спикер A:\nFIRST TRACK", exported)
        self.assertIn("Спикер B:\nSECOND TRACK", exported)

    def test_worker_default_stages_exclude_legacy_sync(self) -> None:
        self.assertEqual(
            normalize_pipeline_stages(None),
            ["transcribe", "backfill-second-asr", "resolve", "analyze"],
        )
        self.assertEqual(
            normalize_pipeline_stages(["transcribe", "sync"]),
            ["transcribe", "sync"],
        )

    def test_cmd_sync_requires_explicit_legacy_opt_in(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_sync_disabled_") as td:
            db_path = Path(td) / "sync_disabled.db"
            settings = make_settings()
            settings = replace(settings, database_url=f"sqlite:///{db_path}")
            init_db(settings)

            with patch("mango_mvp.cli.get_settings", return_value=settings):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cmd_sync(Namespace(limit=10))

            self.assertEqual(rc, 2)
            payload = json.loads(out.getvalue())
            self.assertFalse(payload["ok"])
            self.assertIn(LEGACY_AMOCRM_SYNC_DISABLED_MESSAGE, payload["error"])

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
            self.assertIn("Спикер A", mlx_text)
            self.assertNotIn("Менеджер", mlx_text)
            self.assertIn("Да, слушаю внимательно", gigaam_text)
            self.assertIn("Спикер A", merge_text)
            self.assertNotIn("Менеджер", merge_text)
            self.assertTrue((out_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
