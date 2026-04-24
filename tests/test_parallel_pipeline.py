from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from mango_mvp.cli import cmd_stats
from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.services.pipeline_claims import release_stale_pipeline_claims
from mango_mvp.services.resolve import ResolveService
from mango_mvp.services.transcribe import TranscribeService
from tests.test_dialogue_format import make_settings


def _stereo_payload(
    *,
    primary_provider: str = "mlx",
    secondary_provider: str = "gigaam",
    manager_a: str = "Здравствуйте",
    client_a: str = "Да",
    manager_b: str | None = "Здравствуйте",
    client_b: str | None = "Да",
    exhausted: bool = False,
) -> str:
    payload: dict[str, object] = {
        "mode": "stereo",
        "primary_provider": primary_provider,
        "secondary_provider": secondary_provider,
        "manager": {
            "variant_a": manager_a,
            "variant_b": manager_b,
            "final": manager_a,
        },
        "client": {
            "variant_a": client_a,
            "variant_b": client_b,
            "final": client_a,
        },
    }
    if exhausted:
        payload["secondary_backfill_meta"] = {
            "provider": secondary_provider,
            "attempts": 2,
            "status": "failed",
            "exhausted": True,
            "last_error": "mock exhausted",
        }
    return json.dumps(payload, ensure_ascii=False)


class ParallelPipelineClaimsTest(unittest.TestCase):
    def test_transcribe_claims_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_parallel_tr_claims_") as td:
            db_path = Path(td) / "claims.db"
            settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                for idx in range(3):
                    session.add(
                        CallRecord(
                            source_file=str(Path(td) / f"call_{idx}.mp3"),
                            source_filename=f"call_{idx}.mp3",
                            transcription_status="pending",
                            resolve_status="pending",
                            analysis_status="pending",
                            sync_status="pending",
                        )
                    )
                session.commit()

            service = TranscribeService(settings)
            with session_factory() as session:
                first = service._claim_transcribe_batch(session, limit=2, worker_id="w1")
                second = service._claim_transcribe_batch(session, limit=2, worker_id="w2")
                state = service.count_primary_queue_state(session)

            self.assertEqual(len(first), 2)
            self.assertEqual(len(second), 1)
            self.assertTrue(set(first).isdisjoint(second))
            self.assertEqual(state["ready_pending"], 0)
            self.assertEqual(state["in_progress"], 3)

    def test_secondary_backfill_counts_split_pending_and_in_progress(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_parallel_backfill_") as td:
            db_path = Path(td) / "backfill.db"
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
                            source_file=str(Path(td) / "fresh.mp3"),
                            source_filename="fresh.mp3",
                            transcription_status="done",
                            resolve_status="pending",
                            analysis_status="pending",
                            sync_status="pending",
                            transcript_variants_json=_stereo_payload(
                                secondary_provider="",
                                manager_b=None,
                                client_b=None,
                            ),
                        ),
                        CallRecord(
                            source_file=str(Path(td) / "working.mp3"),
                            source_filename="working.mp3",
                            transcription_status="done",
                            resolve_status="pending",
                            analysis_status="pending",
                            sync_status="pending",
                            pipeline_stage="backfill-second-asr",
                            pipeline_worker_id="bf-1",
                            pipeline_claimed_at=datetime.now(timezone.utc),
                            transcript_variants_json=_stereo_payload(manager_b=None, client_b=None),
                        ),
                        CallRecord(
                            source_file=str(Path(td) / "retry.mp3"),
                            source_filename="retry.mp3",
                            transcription_status="done",
                            resolve_status="pending",
                            analysis_status="pending",
                            sync_status="pending",
                            transcript_variants_json=_stereo_payload(
                                secondary_provider="gigaam",
                                manager_b=None,
                                client_b=None,
                            ),
                        ),
                        CallRecord(
                            source_file=str(Path(td) / "exhausted.mp3"),
                            source_filename="exhausted.mp3",
                            transcription_status="done",
                            resolve_status="pending",
                            analysis_status="pending",
                            sync_status="pending",
                            transcript_variants_json=_stereo_payload(
                                secondary_provider="gigaam",
                                manager_b=None,
                                client_b=None,
                                exhausted=True,
                            ),
                        ),
                    ]
                )
                session.commit()

            service = TranscribeService(settings)
            with session_factory() as session:
                summary = service.count_secondary_backfill_pending(session)

            self.assertTrue(summary["enabled"])
            self.assertEqual(summary["pending"], 2)
            self.assertEqual(summary["retry_pending"], 1)
            self.assertEqual(summary["in_progress"], 1)
            self.assertEqual(summary["exhausted"], 1)

    def test_resolve_requeues_calls_waiting_for_second_asr_without_counting_processed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_parallel_resolve_wait_") as td:
            db_path = Path(td) / "resolve_wait.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                dual_transcribe_enabled=True,
                transcribe_provider="mlx",
                secondary_transcribe_provider="gigaam",
                resolve_llm_provider="off",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(Path(td) / "wait.mp3"),
                        source_filename="wait.mp3",
                        duration_sec=120.0,
                        transcription_status="done",
                        resolve_status="pending",
                        analysis_status="pending",
                        sync_status="pending",
                        transcript_text="MANAGER:\nЗдравствуйте\n\nCLIENT:\nДа",
                        transcript_manager="Здравствуйте",
                        transcript_client="Да",
                        transcript_variants_json=_stereo_payload(
                            secondary_provider="gigaam",
                            manager_b=None,
                            client_b=None,
                        ),
                    )
                )
                session.commit()

            service = ResolveService(settings)
            with session_factory() as session:
                result = service.run(session, limit=10)

            self.assertEqual(result["processed"], 0)
            self.assertEqual(result["success"], 0)
            self.assertEqual(result["failed"], 0)
            with session_factory() as session:
                call = session.query(CallRecord).one()
                self.assertEqual(call.resolve_status, "pending")
                self.assertIsNone(call.pipeline_stage)
                self.assertIsNone(call.pipeline_worker_id)
                self.assertIsNone(call.pipeline_claimed_at)

    def test_release_stale_pipeline_claims_resets_transcribe_and_resolve(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_parallel_stale_") as td:
            db_path = Path(td) / "stale.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                pipeline_lease_timeout_sec=60,
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            stale_at = datetime.now(timezone.utc) - timedelta(hours=2)
            with session_factory() as session:
                session.add_all(
                    [
                        CallRecord(
                            source_file=str(Path(td) / "stale_tr.mp3"),
                            source_filename="stale_tr.mp3",
                            transcription_status="in_progress",
                            pipeline_stage="transcribe",
                            pipeline_worker_id="tr-old",
                            pipeline_claimed_at=stale_at,
                            resolve_status="pending",
                            analysis_status="pending",
                            sync_status="pending",
                        ),
                        CallRecord(
                            source_file=str(Path(td) / "stale_rs.mp3"),
                            source_filename="stale_rs.mp3",
                            transcription_status="done",
                            resolve_status="in_progress",
                            pipeline_stage="resolve",
                            pipeline_worker_id="rs-old",
                            pipeline_claimed_at=stale_at,
                            analysis_status="pending",
                            sync_status="pending",
                        ),
                    ]
                )
                session.commit()

            with session_factory() as session:
                released = release_stale_pipeline_claims(session, settings)
                session.commit()

            self.assertEqual(released, 2)
            with session_factory() as session:
                calls = session.query(CallRecord).order_by(CallRecord.id.asc()).all()
                self.assertEqual(calls[0].transcription_status, "pending")
                self.assertIsNone(calls[0].pipeline_stage)
                self.assertEqual(calls[1].resolve_status, "pending")
                self.assertIsNone(calls[1].pipeline_stage)

    def test_stats_report_queue_and_lease_fields(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_parallel_stats_") as td:
            db_path = Path(td) / "stats.db"
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
                            source_file=str(Path(td) / "tr_in_progress.mp3"),
                            source_filename="tr_in_progress.mp3",
                            transcription_status="in_progress",
                            pipeline_stage="transcribe",
                            pipeline_worker_id="tr-1",
                            pipeline_claimed_at=datetime.now(timezone.utc),
                            resolve_status="pending",
                            analysis_status="pending",
                            sync_status="pending",
                        ),
                        CallRecord(
                            source_file=str(Path(td) / "resolve_ready.mp3"),
                            source_filename="resolve_ready.mp3",
                            transcription_status="done",
                            resolve_status="pending",
                            analysis_status="pending",
                            sync_status="pending",
                            transcript_text="MANAGER:\nЗдравствуйте\n\nCLIENT:\nДа",
                            transcript_manager="Здравствуйте",
                            transcript_client="Да",
                            transcript_variants_json=_stereo_payload(),
                        ),
                        CallRecord(
                            source_file=str(Path(td) / "resolve_blocked.mp3"),
                            source_filename="resolve_blocked.mp3",
                            transcription_status="done",
                            resolve_status="pending",
                            analysis_status="pending",
                            sync_status="pending",
                            transcript_text="MANAGER:\nЗдравствуйте\n\nCLIENT:\nДа",
                            transcript_manager="Здравствуйте",
                            transcript_client="Да",
                            transcript_variants_json=_stereo_payload(manager_b=None, client_b=None),
                        ),
                        CallRecord(
                            source_file=str(Path(td) / "backfill_working.mp3"),
                            source_filename="backfill_working.mp3",
                            transcription_status="done",
                            resolve_status="pending",
                            analysis_status="pending",
                            sync_status="pending",
                            pipeline_stage="backfill-second-asr",
                            pipeline_worker_id="bf-1",
                            pipeline_claimed_at=datetime.now(timezone.utc),
                            transcript_variants_json=_stereo_payload(manager_b=None, client_b=None),
                        ),
                    ]
                )
                session.commit()

            with patch("mango_mvp.cli.get_settings", return_value=settings):
                out = io.StringIO()
                with redirect_stdout(out):
                    rc = cmd_stats(None)

            self.assertEqual(rc, 0)
            payload = json.loads(out.getvalue())
            self.assertIn("transcribe_queue", payload)
            self.assertIn("resolve_queue", payload)
            self.assertIn("pipeline_stage_leases", payload)
            self.assertEqual(payload["transcribe_queue"]["in_progress"], 1)
            self.assertEqual(payload["resolve_queue"]["ready_pending"], 1)
            self.assertEqual(payload["resolve_queue"]["blocked_waiting_secondary"], 2)
            self.assertEqual(payload["secondary_asr_backfill"]["in_progress"], 1)
            self.assertEqual(payload["pipeline_stage_leases"]["transcribe"], 1)
            self.assertEqual(payload["pipeline_stage_leases"]["backfill-second-asr"], 1)

    def test_resolve_queue_handles_naive_retry_timestamps(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_parallel_naive_retry_") as td:
            db_path = Path(td) / "naive_retry.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                dual_transcribe_enabled=True,
                transcribe_provider="mlx",
                secondary_transcribe_provider="gigaam",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            future_retry = datetime.utcnow() + timedelta(minutes=10)
            with session_factory() as session:
                session.add(
                    CallRecord(
                        source_file=str(Path(td) / "naive.mp3"),
                        source_filename="naive.mp3",
                        transcription_status="done",
                        resolve_status="pending",
                        analysis_status="pending",
                        sync_status="pending",
                        next_retry_at=future_retry,
                        transcript_text="MANAGER:\nЗдравствуйте\n\nCLIENT:\nДа",
                        transcript_manager="Здравствуйте",
                        transcript_client="Да",
                        transcript_variants_json=_stereo_payload(),
                    )
                )
                session.commit()

            service = ResolveService(settings)
            with session_factory() as session:
                summary = service.count_queue_state(session)

            self.assertEqual(summary["ready_pending"], 0)
            self.assertEqual(summary["blocked_waiting_secondary"], 0)
