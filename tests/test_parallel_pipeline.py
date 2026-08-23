from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from mango_mvp.cli import cmd_stats
from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.services.pipeline_claims import release_stale_pipeline_claims
from mango_mvp.productization.mango_calls_service_contract import (
    has_dual_asr_or_exception,
    ready_row_is_complete,
    resolve_row_is_complete,
)
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
    def test_selective_gigaam_skips_only_high_confidence_non_conversation(self) -> None:
        service = TranscribeService(make_settings())
        call = CallRecord(
            source_call_id="voicemail-primary",
            source_filename="voicemail.mp3",
            duration_sec=20.0,
        )
        payload = {
            "mode": "stereo",
            "primary_provider": "mlx",
            "manager": {"variant_a": "Здравствуйте.", "physical_channel": "left"},
            "client": {
                "variant_a": "Абонент сейчас не может ответить. Оставьте сообщение после звукового сигнала.",
                "physical_channel": "right",
            },
        }

        with patch.dict(os.environ, {"GIGAAM_POLICY": "selective_non_conversation_v1"}):
            updated = service._apply_selective_gigaam_policy(call, payload)

        self.assertEqual(updated["secondary_asr_policy"]["decision"], "skipped")
        self.assertIn(
            "high_confidence_non_conversation",
            updated["secondary_asr_policy"]["reason_codes"],
        )
        self.assertTrue(updated["dual_asr_exception"]["approved"])
        self.assertEqual(
            service.secondary_backfill_state_from_payload(
                updated,
                secondary_provider="gigaam",
            ),
            "not_needed",
        )

    def test_selective_gigaam_requires_contentful_call_even_with_high_wpm(self) -> None:
        service = TranscribeService(make_settings())
        call = CallRecord(
            source_call_id="low-primary",
            source_filename="low.mp3",
            duration_sec=60.0,
        )
        payload = {
            "mode": "stereo",
            "primary_provider": "mlx",
            "manager": {"variant_a": " ".join(["расписание"] * 70)},
            "client": {"variant_a": " ".join(["оплата"] * 70)},
            "dual_asr_exception": {
                "approved": True,
                "reason": "selective_rescue_v1:primary_text_sufficient",
                "approved_by": "owner_policy:selective_rescue_v1",
                "approved_at": "2026-08-01T00:00:00+00:00",
            },
        }

        with patch.dict(os.environ, {"GIGAAM_POLICY": "selective_non_conversation_v1"}):
            updated = service._apply_selective_gigaam_policy(call, payload)

        self.assertEqual(updated["secondary_asr_policy"]["decision"], "required")
        self.assertIn(
            "gigaam_required_for_contentful_or_ambiguous_call",
            updated["secondary_asr_policy"]["reason_codes"],
        )
        self.assertNotIn("dual_asr_exception", updated)
        self.assertEqual(
            service.secondary_backfill_state_from_payload(
                updated,
                secondary_provider="gigaam",
            ),
            "fresh",
        )

    def test_selective_gigaam_quality_sample_still_runs_second_asr(self) -> None:
        service = TranscribeService(make_settings())
        payload = {
            "mode": "stereo",
            "primary_provider": "mlx",
            "manager": {"variant_a": "Здравствуйте.", "physical_channel": "left"},
            "client": {
                "variant_a": "Абонент недоступен. Оставьте сообщение после звукового сигнала.",
                "physical_channel": "right",
            },
        }
        updated = payload
        with patch.dict(os.environ, {"GIGAAM_POLICY": "selective_non_conversation_v1"}):
            for idx in range(1000):
                call = CallRecord(
                    source_call_id=f"quality-sample-{idx}",
                    duration_sec=20.0,
                )
                candidate = service._apply_selective_gigaam_policy(call, payload)
                if candidate["secondary_asr_policy"]["shadow_quality_sample"]:
                    updated = candidate
                    break

        self.assertTrue(updated["secondary_asr_policy"]["shadow_quality_sample"])
        self.assertEqual(updated["secondary_asr_policy"]["decision"], "required")
        self.assertNotIn("dual_asr_exception", updated)
        self.assertEqual(
            service.secondary_backfill_state_from_payload(
                updated,
                secondary_provider="gigaam",
            ),
            "fresh",
        )

    def test_selective_gigaam_preserves_explicit_owner_exception(self) -> None:
        service = TranscribeService(make_settings())
        call = CallRecord(source_call_id="owner-approved", duration_sec=60.0)
        owner_exception = {
            "approved": True,
            "reason": "owner approved single-ASR exception",
            "approved_by": "owner",
            "approved_at": "2026-08-01T00:00:00+00:00",
        }
        payload = {
            "mode": "stereo",
            "primary_provider": "mlx",
            "manager": {"variant_a": "Информация по расписанию курса."},
            "client": {"variant_a": "Да, я изучу и отвечу позднее."},
            "dual_asr_exception": owner_exception,
        }

        with patch.dict(os.environ, {"GIGAAM_POLICY": "selective_non_conversation_v1"}):
            updated = service._apply_selective_gigaam_policy(call, payload)

        self.assertEqual(updated["dual_asr_exception"], owner_exception)
        self.assertEqual(
            service.secondary_backfill_state_from_payload(
                updated,
                secondary_provider="gigaam",
            ),
            "not_needed",
        )

    def test_selective_gigaam_policy_is_off_by_default(self) -> None:
        service = TranscribeService(make_settings())
        payload = {"mode": "mono_or_fallback", "full": {"variant_a": "текст"}}
        call = CallRecord(source_call_id="default", duration_sec=60.0)

        with patch.dict(os.environ, {}, clear=True):
            updated = service._apply_selective_gigaam_policy(call, payload)

        self.assertIs(updated, payload)

    def test_secondary_claim_heartbeat_requires_exact_owner(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_backfill_heartbeat_") as td:
            db_path = Path(td) / "heartbeat.db"
            settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                call = CallRecord(
                    source_file=str(Path(td) / "call.mp3"),
                    source_filename="call.mp3",
                    transcription_status="done",
                    pipeline_stage="backfill-second-asr",
                    pipeline_worker_id="owner",
                    pipeline_claimed_at=datetime.now(timezone.utc),
                )
                session.add(call)
                session.commit()
                call_id = int(call.id)

            with session_factory() as session:
                TranscribeService._renew_secondary_claim(
                    session,
                    call_id=call_id,
                    worker_id="owner",
                )
                with self.assertRaisesRegex(RuntimeError, "secondary_asr_lease_lost"):
                    TranscribeService._renew_secondary_claim(
                        session,
                        call_id=call_id,
                        worker_id="other",
                    )

    def test_selective_backfill_claims_only_one_call_per_cycle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_single_backfill_claim_") as td:
            db_path = Path(td) / "single.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                dual_transcribe_enabled=True,
                transcribe_provider="mlx",
                secondary_transcribe_provider="gigaam",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            service = TranscribeService(settings)
            with session_factory() as session:
                with patch.object(
                    service,
                    "_claim_secondary_backfill_batch",
                    return_value=[],
                ) as claim:
                    with patch.dict(
                        os.environ,
                        {"GIGAAM_POLICY": "selective_non_conversation_v1"},
                    ):
                        service.backfill_secondary_asr(session, limit=10)

        self.assertEqual(claim.call_args.kwargs["limit"], 1)

    def test_default_backfill_preserves_requested_limit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_default_backfill_claim_") as td:
            db_path = Path(td) / "default.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                dual_transcribe_enabled=True,
                transcribe_provider="mlx",
                secondary_transcribe_provider="gigaam",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            service = TranscribeService(settings)
            with session_factory() as session:
                with patch.object(
                    service,
                    "_claim_secondary_backfill_batch",
                    return_value=[],
                ) as claim:
                    with patch.dict(os.environ, {"GIGAAM_POLICY": "all"}):
                        service.backfill_secondary_asr(session, limit=10)

        self.assertEqual(claim.call_args.kwargs["limit"], 10)

    def test_required_policy_claims_empty_primary_fail_open(self) -> None:
        payload = {
            "mode": "stereo",
            "primary_provider": "mlx",
            "manager": {"variant_a": "", "physical_channel": "left"},
            "client": {"variant_a": "да", "physical_channel": "right"},
            "secondary_asr_policy": {
                "schema": "selective_non_conversation_v1",
                "decision": "required",
            },
        }

        self.assertEqual(
            TranscribeService.secondary_backfill_state_from_payload(
                payload,
                secondary_provider="gigaam",
            ),
            "fresh",
        )

    def test_missing_one_primary_stereo_channel_is_a_rescue_candidate(self) -> None:
        payload = {
            "mode": "stereo",
            "primary_provider": "mlx",
            "manager": {"variant_a": "", "physical_channel": "left"},
            "client": {"variant_a": "да", "physical_channel": "right"},
        }

        self.assertEqual(
            TranscribeService.secondary_backfill_state_from_payload(
                payload,
                secondary_provider="gigaam",
            ),
            "fresh",
        )

    def test_secondary_backfill_rejects_non_text_primary_variant(self) -> None:
        state = TranscribeService.secondary_backfill_state_from_payload(
            {
                "mode": "mono_or_fallback",
                "primary_provider": "mlx",
                "secondary_provider": "gigaam",
                "full": {"variant_a": {"bad": 1}, "variant_b": ""},
            },
            secondary_provider="gigaam",
        )

        self.assertEqual(state, "not_needed")

    def test_secondary_backfill_skips_valid_audited_dual_asr_exception(self) -> None:
        state = TranscribeService.secondary_backfill_state_from_payload(
            {
                "mode": "mono_or_fallback",
                "primary_provider": "mlx",
                "secondary_provider": "gigaam",
                "full": {"variant_a": "primary", "variant_b": ""},
                "dual_asr_exception": {
                    "approved": True,
                    "reason": "synthetic audited exception",
                    "approved_by": "owner",
                    "approved_at": "2026-07-01T00:00:00+00:00",
                },
            },
            secondary_provider="gigaam",
        )

        self.assertEqual(state, "not_needed")

    def test_secondary_backfill_does_not_skip_unapproved_exception(self) -> None:
        state = TranscribeService.secondary_backfill_state_from_payload(
            {
                "mode": "mono_or_fallback",
                "primary_provider": "mlx",
                "secondary_provider": "gigaam",
                "full": {"variant_a": "primary", "variant_b": ""},
                "dual_asr_exception": {
                    "approved": False,
                    "reason": "not approved",
                    "approved_by": "owner",
                    "approved_at": "2026-07-01T00:00:00+00:00",
                },
            },
            secondary_provider="gigaam",
        )

        self.assertEqual(state, "retry")

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

    def test_pipeline_worker_ids_are_process_unique(self) -> None:
        worker_ids = {
            TranscribeService._pipeline_worker_id("bf") for _index in range(100)
        }

        self.assertEqual(len(worker_ids), 100)
        self.assertTrue(
            all(worker_id.startswith(f"bf-{os.getpid()}-") for worker_id in worker_ids)
        )

    def test_parallel_secondary_claims_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_parallel_bf_claims_") as td:
            db_path = Path(td) / "claims.db"
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
                for idx in range(4):
                    session.add(
                        CallRecord(
                            source_file=str(Path(td) / f"call_{idx}.mp3"),
                            source_filename=f"call_{idx}.mp3",
                            transcription_status="done",
                            resolve_status="pending",
                            analysis_status="pending",
                            sync_status="pending",
                            transcript_variants_json=_stereo_payload(
                                manager_b=None,
                                client_b=None,
                            ),
                        )
                    )
                session.commit()

            barrier = threading.Barrier(2)

            def claim(worker_id: str) -> list[int]:
                service = TranscribeService(settings)
                with session_factory() as session:
                    barrier.wait(timeout=2)
                    return service._claim_secondary_backfill_batch(
                        session,
                        limit=1,
                        worker_id=worker_id,
                        secondary_provider="gigaam",
                    )

            with ThreadPoolExecutor(max_workers=2) as pool:
                first_future = pool.submit(claim, "bf-1")
                second_future = pool.submit(claim, "bf-2")
                first = first_future.result(timeout=5)
                second = second_future.result(timeout=5)

            self.assertGreaterEqual(len(first) + len(second), 1)
            self.assertTrue(set(first).isdisjoint(second))
            with session_factory() as session:
                owners = dict(
                    session.query(CallRecord.id, CallRecord.pipeline_worker_id)
                    .filter(CallRecord.pipeline_stage == "backfill-second-asr")
                    .all()
                )
            self.assertEqual(set(owners), set(first + second))
            self.assertTrue(set(owners.values()).issubset({"bf-1", "bf-2"}))

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

    def test_secondary_backfill_prioritizes_retry_over_fresh(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_backfill_retry_first_") as td:
            db_path = Path(td) / "retry_first.db"
            settings = replace(make_settings(), database_url=f"sqlite:///{db_path}")
            init_db(settings)
            session_factory = build_session_factory(settings)
            with session_factory() as session:
                session.add_all(
                    [
                        CallRecord(
                            source_file=str(Path(td) / "fresh.mp3"),
                            source_filename="fresh.mp3",
                            transcription_status="done",
                            transcript_variants_json=_stereo_payload(
                                secondary_provider="", manager_b=None, client_b=None
                            ),
                        ),
                        CallRecord(
                            source_file=str(Path(td) / "retry.mp3"),
                            source_filename="retry.mp3",
                            transcription_status="done",
                            transcript_variants_json=_stereo_payload(
                                manager_b="Здравствуйте", client_b=None
                            ),
                        ),
                    ]
                )
                session.commit()
                retry_id = int(
                    session.query(CallRecord)
                    .filter(CallRecord.source_filename == "retry.mp3")
                    .one()
                    .id
                )

            service = TranscribeService(settings)
            with session_factory() as session:
                claimed = service._claim_secondary_backfill_batch(
                    session,
                    limit=1,
                    worker_id="retry-owner",
                    secondary_provider="gigaam",
                )

            self.assertEqual(claimed, [retry_id])

    def test_exhausted_secondary_fallback_is_explicit_and_ready(self) -> None:
        service = TranscribeService(make_settings())
        payload = json.loads(
            _stereo_payload(manager_b="Здравствуйте", client_b=None, exhausted=True)
        )

        updated = service._apply_exhausted_secondary_exception(
            payload,
            secondary_provider="gigaam",
        )
        row = {
            "transcription_status": "done",
            "transcript_variants_json": json.dumps(updated, ensure_ascii=False),
            "resolve_status": "manual",
            "analysis_status": "done",
            "analysis_json": json.dumps({"needs_review": True}),
        }

        self.assertTrue(has_dual_asr_or_exception(row))
        self.assertTrue(ready_row_is_complete(row))

    def test_manual_resolve_is_terminal_only_with_boolean_review_flag(self) -> None:
        base = {
            "transcription_status": "done",
            "transcript_variants_json": _stereo_payload(),
            "resolve_status": "manual",
            "analysis_status": "done",
        }

        for analysis in ({}, {"needs_review": False}, {"needs_review": "true"}):
            row = {**base, "analysis_json": json.dumps(analysis)}
            self.assertFalse(resolve_row_is_complete(row))
            self.assertFalse(ready_row_is_complete(row))

        row = {**base, "analysis_json": json.dumps({"needs_review": True})}
        self.assertTrue(resolve_row_is_complete(row))
        self.assertTrue(ready_row_is_complete(row))

    def test_second_secondary_failure_becomes_terminal_review_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_backfill_exhausted_") as td:
            db_path = Path(td) / "exhausted.db"
            settings = replace(
                make_settings(),
                database_url=f"sqlite:///{db_path}",
                dual_transcribe_enabled=True,
                transcribe_provider="mlx",
                secondary_transcribe_provider="gigaam",
            )
            init_db(settings)
            session_factory = build_session_factory(settings)
            variants = json.loads(
                _stereo_payload(
                    manager_a="Здравствуйте",
                    client_a="",
                    manager_b="Здравствуйте",
                    client_b=None,
                )
            )
            variants["secondary_asr_policy"] = {
                "schema": "selective_non_conversation_v1",
                "decision": "required",
            }
            with session_factory() as session:
                call = CallRecord(
                    source_file=str(Path(td) / "call.mp3"),
                    source_filename="call.mp3",
                    transcription_status="done",
                    resolve_status="pending",
                    analysis_status="pending",
                    sync_status="pending",
                    transcript_text="[00:00.0] Менеджер: Здравствуйте.",
                    transcript_variants_json=json.dumps(variants, ensure_ascii=False),
                )
                session.add(call)
                session.commit()
                call_id = int(call.id)

            service = TranscribeService(settings)
            with session_factory() as session, patch.object(
                service,
                "_backfill_secondary_only",
                side_effect=RuntimeError("secondary still unavailable"),
            ):
                report = service.backfill_secondary_asr(session, limit=1)

            with session_factory() as session:
                call = session.get(CallRecord, call_id)
                assert call is not None
                payload = json.loads(call.transcript_variants_json or "{}")
                row = {
                    "transcript_variants_json": call.transcript_variants_json,
                }

            self.assertEqual(report["exhausted"], 1)
            self.assertTrue(payload["secondary_backfill_meta"]["exhausted"])
            self.assertTrue(has_dual_asr_or_exception(row))
            self.assertIsNone(call.pipeline_stage)

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
