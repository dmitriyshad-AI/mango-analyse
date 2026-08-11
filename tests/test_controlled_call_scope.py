from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import mango_mvp.services.controlled_call_scope as controlled_scope_module
from mango_mvp.config import Settings
from mango_mvp.customer_timeline.calls_two_processes import (
    CallsTwoProcessesConfig,
    controlled_call_database_snapshot,
    controlled_stage_report,
    controlled_worker_authority_environment,
    parse_worker_stage_metrics,
    process_lease,
    run_controlled_one,
    stage_subprocess_command,
    worker_environment,
)
from mango_mvp.db import build_session_factory, init_db
from mango_mvp.models import CallRecord
from mango_mvp.productization.mango_calls_service_contract import current_git_sha
from mango_mvp.services.analyze import AnalyzeService
from mango_mvp.services.controlled_call_scope import (
    CONTROLLED_CALL_ALLOWLIST_SCHEMA,
    controlled_audio_input_path,
    enforce_controlled_cli_command,
    enforce_controlled_worker_stages,
    load_controlled_call_scope,
)
from mango_mvp.services.resolve import ResolveService
from mango_mvp.services.transcribe import TranscribeService
from mango_mvp.services.worker import run_worker
from tests.test_dialogue_format import make_settings


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clean_git_for_scope_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        controlled_scope_module,
        "git_worktree_is_clean",
        lambda _root: True,
    )


def _write_allowlist(
    root: Path,
    *,
    source_call_ids: list[str] | None = None,
    tenant_id: str = "foton",
    host_id: str = "m1-host",
    code_sha: str | None = None,
) -> tuple[Path, str]:
    owner_local = root / ".mango_local" / "controlled-one"
    owner_local.mkdir(parents=True, mode=0o700, exist_ok=True)
    path = owner_local / "allowlist.json"
    payload = {
        "schema_version": CONTROLLED_CALL_ALLOWLIST_SCHEMA,
        "source_call_ids": source_call_ids if source_call_ids is not None else ["TARGET"],
        "target_record_id": 2,
        "source_audio_sha256": hashlib.sha256(
            b"synthetic-target-audio"
        ).hexdigest(),
        "source_audio_size_bytes": len(b"synthetic-target-audio"),
        "tenant_id": tenant_id,
        "code_sha": code_sha or current_git_sha(REPO_ROOT),
        "host_id": host_id,
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(raw)
    path.chmod(0o600)
    return path, hashlib.sha256(raw).hexdigest()


def _controlled_settings(
    root: Path,
    *,
    database_url: str = "sqlite:///test.db",
    source_call_ids: list[str] | None = None,
) -> Settings:
    path, sha256 = _write_allowlist(root, source_call_ids=source_call_ids)
    host_id_path = root / ".mango_local" / "controlled-one" / "host_id"
    host_id_path.write_text("m1-host\n", encoding="utf-8")
    host_id_path.chmod(0o600)
    audio_snapshot_path = (
        root / ".mango_local" / "controlled-one" / "input.wav"
    )
    audio_snapshot_path.write_bytes(b"synthetic-target-audio")
    audio_snapshot_path.chmod(0o600)
    return replace(
        make_settings(),
        database_url=database_url,
        calls_processing_scope="controlled_1",
        controlled_call_allowlist_path=str(path),
        controlled_call_allowlist_sha256=sha256,
        controlled_call_tenant_id="foton",
        controlled_call_code_sha=current_git_sha(REPO_ROOT),
        controlled_call_host_id="m1-host",
        controlled_call_host_id_path=str(host_id_path),
        controlled_call_audio_snapshot_path=str(audio_snapshot_path),
        controlled_call_audio_snapshot_sha256=hashlib.sha256(
            b"synthetic-target-audio"
        ).hexdigest(),
        controlled_call_audio_snapshot_size_bytes=len(
            b"synthetic-target-audio"
        ),
    )


def _controlled_authority_ok(config, *_args, **_kwargs) -> dict[str, object]:
    return {
        "ok": True,
        "active_host_id": "m1-host",
        "verified_cutover_manifest_sha256": hashlib.sha256(
            config.cutover_manifest_file.read_bytes()
        ).hexdigest(),
    }


def _row_state(call: CallRecord) -> tuple[object, ...]:
    values = (
        getattr(call, name)
        for name in (
            "transcription_status",
            "resolve_status",
            "analysis_status",
            "pipeline_stage",
            "pipeline_worker_id",
            "pipeline_claimed_at",
            "analysis_worker_id",
            "analysis_claimed_at",
            "transcribe_attempts",
            "resolve_attempts",
            "analyze_attempts",
            "transcript_text",
            "transcript_variants_json",
            "resolve_json",
            "analysis_json",
            "dead_letter_stage",
            "last_error",
            "updated_at",
        )
    )
    return tuple(
        value.replace(tzinfo=None).isoformat()
        if isinstance(value, datetime)
        else value
        for value in values
    )


def test_allowlist_is_owner_only_exact_and_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    settings = _controlled_settings(tmp_path)

    scope = load_controlled_call_scope(settings)

    assert scope is not None
    assert scope.source_call_id == "TARGET"
    assert scope.allowlist_sha256 == settings.controlled_call_allowlist_sha256

    monkeypatch.setattr(
        controlled_scope_module,
        "git_worktree_is_clean",
        lambda _root: False,
    )
    with pytest.raises(RuntimeError, match="runtime_code_mismatch"):
        load_controlled_call_scope(settings)
    monkeypatch.setattr(
        controlled_scope_module,
        "git_worktree_is_clean",
        lambda _root: True,
    )

    path = Path(settings.controlled_call_allowlist_path or "")
    path.chmod(0o644)
    with pytest.raises(RuntimeError, match="owner_only_0600"):
        load_controlled_call_scope(settings)


@pytest.mark.parametrize("ids", [[], ["A", "B"], [""], [" TARGET"]])
def test_allowlist_rejects_not_exactly_one_canonical_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ids: list[str],
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    settings = _controlled_settings(tmp_path, source_call_ids=ids)

    with pytest.raises(RuntimeError):
        load_controlled_call_scope(settings)


def test_allowlist_rejects_digest_symlink_and_hardlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    settings = _controlled_settings(tmp_path)
    path = Path(settings.controlled_call_allowlist_path or "")

    wrong_digest = replace(
        settings,
        controlled_call_allowlist_sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="sha256_mismatch"):
        load_controlled_call_scope(wrong_digest)

    real = path.with_name("real.json")
    path.replace(real)
    path.symlink_to(real)
    with pytest.raises(RuntimeError, match="unsafe_or_missing"):
        load_controlled_call_scope(settings)
    path.unlink()
    os.link(real, path)
    with pytest.raises(RuntimeError, match="owner_only_0600"):
        load_controlled_call_scope(settings)


def test_allowlist_rejects_actual_host_mismatch_before_database_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    settings = _controlled_settings(tmp_path)
    host_id_path = Path(settings.controlled_call_host_id_path or "")
    host_id_path.write_text("another-m1\n", encoding="utf-8")
    host_id_path.chmod(0o600)

    with pytest.raises(RuntimeError, match="actual_host_mismatch"):
        load_controlled_call_scope(settings)


def test_controlled_audio_snapshot_requires_owner_only_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    settings = _controlled_settings(tmp_path)
    snapshot_path = Path(settings.controlled_call_audio_snapshot_path or "")
    snapshot_path.chmod(0o644)

    with pytest.raises(RuntimeError, match="owner_only_0600"):
        controlled_audio_input_path(
            settings,
            record_id=2,
            source_call_id="TARGET",
            source_file=tmp_path / "target.wav",
        )


def test_controlled_claims_and_stale_recovery_never_touch_other_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    stale = datetime.now(timezone.utc) - timedelta(hours=2)

    cases = (
        (
            "transcribe",
            dict(transcription_status="pending", resolve_status="pending", analysis_status="pending"),
            dict(
                transcription_status="in_progress",
                resolve_status="pending",
                analysis_status="pending",
                pipeline_stage="transcribe",
                pipeline_worker_id="old",
                pipeline_claimed_at=stale,
            ),
        ),
        (
            "backfill",
            dict(
                transcription_status="done",
                resolve_status="pending",
                analysis_status="pending",
                transcript_variants_json=json.dumps(
                    {
                        "mode": "mono_or_fallback",
                        "primary_provider": "mlx",
                        "secondary_provider": "gigaam",
                        "full": {"variant_a": "target", "variant_b": ""},
                    }
                ),
            ),
            dict(
                transcription_status="done",
                resolve_status="pending",
                analysis_status="pending",
                pipeline_stage="backfill-second-asr",
                pipeline_worker_id="old",
                pipeline_claimed_at=stale,
                transcript_variants_json=json.dumps(
                    {
                        "mode": "mono_or_fallback",
                        "primary_provider": "mlx",
                        "secondary_provider": "gigaam",
                        "full": {"variant_a": "other", "variant_b": ""},
                    }
                ),
            ),
        ),
        (
            "resolve",
            dict(
                transcription_status="done",
                resolve_status="pending",
                analysis_status="pending",
                transcript_text="MANAGER:\nhello\nCLIENT:\nhello",
            ),
            dict(
                transcription_status="done",
                resolve_status="in_progress",
                analysis_status="pending",
                pipeline_stage="resolve",
                pipeline_worker_id="old",
                pipeline_claimed_at=stale,
                transcript_text="other",
            ),
        ),
        (
            "analyze",
            dict(
                transcription_status="done",
                resolve_status="done",
                analysis_status="pending",
                transcript_text="target",
            ),
            dict(
                transcription_status="done",
                resolve_status="done",
                analysis_status="in_progress",
                analysis_worker_id="old",
                analysis_claimed_at=stale,
                transcript_text="other",
            ),
        ),
    )
    for stage, target_fields, other_fields in cases:
        db_path = tmp_path / f"{stage}.sqlite"
        settings = _controlled_settings(
            tmp_path,
            database_url=f"sqlite:///{db_path}",
        )
        settings = replace(
            settings,
            dual_transcribe_enabled=True,
            transcribe_provider="mlx",
            secondary_transcribe_provider="gigaam",
        )
        init_db(settings)
        session_factory = build_session_factory(settings)
        with session_factory() as session:
            other = CallRecord(
                source_file=str(tmp_path / f"{stage}-other.wav"),
                source_filename=f"{stage}-other.wav",
                source_call_id="OTHER",
                sync_status="pending",
                **other_fields,
            )
            target = CallRecord(
                source_file=str(tmp_path / f"{stage}-target.wav"),
                source_filename=f"{stage}-target.wav",
                source_call_id="TARGET",
                sync_status="pending",
                **target_fields,
            )
            session.add_all([other, target])
            session.commit()
            other_id = int(other.id)
            target_id = int(target.id)
            before = _row_state(other)

        with session_factory() as session:
            if stage == "transcribe":
                claimed = TranscribeService(settings)._claim_transcribe_batch(
                    session, limit=1, worker_id="new"
                )
            elif stage == "backfill":
                claimed = TranscribeService(settings)._claim_secondary_backfill_batch(
                    session,
                    limit=1,
                    worker_id="new",
                    secondary_provider="gigaam",
                )
            elif stage == "resolve":
                claimed = ResolveService(settings)._claim_batch(
                    session, limit=1, worker_id="new"
                )
            else:
                claimed = AnalyzeService(settings)._claim_batch(
                    session, limit=1, worker_id="new"
                )

        assert claimed == [target_id]
        with session_factory() as session:
            assert _row_state(session.get(CallRecord, other_id)) == before


def test_controlled_worker_drains_only_target_then_becomes_idle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(
        controlled_scope_module,
        "_validate_controlled_runtime_settings",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        controlled_scope_module,
        "_validate_controlled_run_authority",
        lambda *_args, **_kwargs: None,
    )
    db_path = tmp_path / "worker.sqlite"
    settings = _controlled_settings(
        tmp_path,
        database_url=f"sqlite:///{db_path}",
    )
    settings = replace(
        settings,
        transcript_export_dir=str(
            tmp_path / ".mango_local" / "controlled-one" / "transcripts"
        ),
        transcribe_provider="mock",
        dual_transcribe_enabled=True,
        secondary_transcribe_provider="mock-secondary",
        resolve_llm_provider="none",
        analyze_provider="mock",
    )
    original_transcribe = TranscribeService._transcribe_file_with_meta

    def synthetic_secondary(
        service: TranscribeService,
        path: Path,
        provider: str,
    ) -> dict[str, object]:
        if provider == "mock-secondary":
            return {
                "text": f"[secondary mock transcript for {path.name}]",
                "segments": None,
            }
        return original_transcribe(service, path, provider)

    monkeypatch.setattr(
        TranscribeService,
        "_transcribe_file_with_meta",
        synthetic_secondary,
    )
    init_db(settings)
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        session.add_all(
            [
                CallRecord(
                    source_file=str(tmp_path / "other.wav"),
                    source_filename="other.wav",
                    source_call_id="OTHER",
                    transcription_status="pending",
                    resolve_status="pending",
                    analysis_status="pending",
                    sync_status="pending",
                ),
                CallRecord(
                    source_file=str(tmp_path / "target.wav"),
                    source_filename="target.wav",
                    source_call_id="TARGET",
                    transcription_status="pending",
                    resolve_status="pending",
                    analysis_status="pending",
                    sync_status="pending",
                ),
            ]
        )
        session.commit()
        other = session.query(CallRecord).filter_by(source_call_id="OTHER").one()
        other_before = _row_state(other)

    stages = ("transcribe", "backfill-second-asr", "resolve", "analyze")
    first = {
        stage: run_worker(
            settings,
            stage_limit=1,
            once=False,
            stages=[stage],
            poll_sec=1,
            max_idle_cycles=1,
        )["totals"][stage]
        for stage in stages
    }
    second = {
        stage: run_worker(
            settings,
            stage_limit=1,
            once=False,
            stages=[stage],
            poll_sec=1,
            max_idle_cycles=1,
        )["totals"][stage]
        for stage in stages
    }

    assert first["transcribe"]["processed"] == 1
    assert first["backfill-second-asr"]["processed"] in {0, 1}
    assert first["resolve"]["processed"] == 1
    assert first["analyze"]["processed"] == 1
    assert all(
        second[stage]["processed"] == 0
        for stage in stages
    )
    with session_factory() as session:
        target = session.query(CallRecord).filter_by(source_call_id="TARGET").one()
        other = session.query(CallRecord).filter_by(source_call_id="OTHER").one()
        assert target.transcription_status == "done"
        assert target.resolve_status in {"done", "skipped"}
        assert target.analysis_status == "done"
        assert _row_state(other) == other_before
    assert not list(
        (
            tmp_path
            / ".mango_local"
            / "controlled-one"
            / "transcripts"
        ).rglob("*other*")
    )


def test_controlled_transcript_export_cannot_collide_with_other_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    db_path = tmp_path / "artifact-collision.sqlite"
    export_dir = (
        tmp_path / ".mango_local" / "controlled-one" / "transcripts"
    )
    settings = replace(
        _controlled_settings(
            tmp_path,
            database_url=f"sqlite:///{db_path}",
        ),
        transcript_export_dir=str(export_dir),
        transcribe_provider="mock",
    )
    init_db(settings)
    audio_dir = tmp_path / "same-parent"
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        session.add_all(
            [
                CallRecord(
                    source_file=str(audio_dir / "collision.mp3"),
                    source_filename="collision.mp3",
                    source_call_id="OTHER",
                    transcription_status="pending",
                    resolve_status="pending",
                    analysis_status="pending",
                    sync_status="pending",
                ),
                CallRecord(
                    source_file=str(audio_dir / "collision.wav"),
                    source_filename="collision.wav",
                    source_call_id="TARGET",
                    transcription_status="pending",
                    resolve_status="pending",
                    analysis_status="pending",
                    sync_status="pending",
                ),
            ]
        )
        session.commit()
    legacy_dir = export_dir / audio_dir.name
    legacy_dir.mkdir(parents=True)
    other_text = legacy_dir / "collision_text.txt"
    other_variants = legacy_dir / "collision_variants.json"
    other_text.write_text("other-text", encoding="utf-8")
    other_variants.write_text("other-variants", encoding="utf-8")

    with session_factory() as session:
        result = TranscribeService(settings).run(session, limit=1)

    assert result["processed"] == 1
    assert other_text.read_text(encoding="utf-8") == "other-text"
    assert other_variants.read_text(encoding="utf-8") == "other-variants"
    controlled_dir = (
        export_dir
        / "controlled_1"
        / hashlib.sha256(b"TARGET").hexdigest()
    )
    assert (controlled_dir / "collision_text.txt").is_file()
    assert (controlled_dir / "collision_variants.json").is_file()
    assert (controlled_dir / "collision_text.txt").stat().st_mode & 0o777 == 0o600


def test_service_transcript_export_still_creates_fresh_directory(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "service-export.sqlite"
    export_dir = tmp_path / "fresh-export"
    settings = replace(
        make_settings(),
        database_url=f"sqlite:///{db_path}",
        transcript_export_dir=str(export_dir),
        transcribe_provider="mock",
    )
    init_db(settings)
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        session.add(
            CallRecord(
                source_file=str(tmp_path / "fresh-audio" / "one.wav"),
                source_filename="one.wav",
                source_call_id="SERVICE-ONE",
                transcription_status="pending",
                resolve_status="pending",
                analysis_status="pending",
                sync_status="pending",
            )
        )
        session.commit()

    with session_factory() as session:
        result = TranscribeService(settings).run(session, limit=1)

    assert result["success"] == 1
    assert (export_dir / "fresh-audio" / "one_text.txt").is_file()


def test_controlled_transcript_export_rejects_symlinked_call_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    db_path = tmp_path / "artifact-symlink.sqlite"
    export_dir = (
        tmp_path / ".mango_local" / "controlled-one" / "transcripts"
    )
    settings = replace(
        _controlled_settings(
            tmp_path,
            database_url=f"sqlite:///{db_path}",
        ),
        transcript_export_dir=str(export_dir),
        transcribe_provider="mock",
    )
    init_db(settings)
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        session.add_all(
            [
                CallRecord(
                    source_file=str(tmp_path / "other.wav"),
                    source_filename="other.wav",
                    source_call_id="OTHER",
                    transcription_status="pending",
                    resolve_status="pending",
                    analysis_status="pending",
                    sync_status="pending",
                ),
                CallRecord(
                    source_file=str(tmp_path / "target.wav"),
                    source_filename="target.wav",
                    source_call_id="TARGET",
                    transcription_status="pending",
                    resolve_status="pending",
                    analysis_status="pending",
                    sync_status="pending",
                ),
            ]
        )
        session.commit()

    victim_dir = tmp_path / ".mango_local" / "controlled-one" / "victim"
    victim_dir.mkdir(mode=0o700)
    victim = victim_dir / "target_text.txt"
    victim.write_text("must-not-change", encoding="utf-8")
    controlled_root = export_dir / "controlled_1"
    controlled_root.mkdir(parents=True, mode=0o700)
    controlled_root.chmod(0o700)
    target_dir = controlled_root / hashlib.sha256(b"TARGET").hexdigest()
    target_dir.symlink_to(victim_dir, target_is_directory=True)

    with session_factory() as session:
        result = TranscribeService(settings).run(session, limit=1)

    assert result["processed"] == 1
    assert result["failed"] == 1
    assert victim.read_text(encoding="utf-8") == "must-not-change"


@pytest.mark.parametrize("target_count", [0, 2])
def test_controlled_worker_rejects_unknown_or_duplicate_database_id_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_count: int,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    db_path = tmp_path / f"target-count-{target_count}.sqlite"
    settings = _controlled_settings(
        tmp_path,
        database_url=f"sqlite:///{db_path}",
    )
    init_db(settings)
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        session.add(
            CallRecord(
                source_file="other.wav",
                source_filename="other.wav",
                source_call_id="OTHER",
                transcription_status="pending",
                resolve_status="pending",
                analysis_status="pending",
                sync_status="pending",
            )
        )
        for index in range(target_count):
            session.add(
                CallRecord(
                    source_file=f"target-{index}.wav",
                    source_filename=f"target-{index}.wav",
                    source_call_id="TARGET",
                    transcription_status="pending",
                    resolve_status="pending",
                    analysis_status="pending",
                    sync_status="pending",
                )
            )
        session.commit()
        before = [_row_state(call) for call in session.query(CallRecord).order_by(CallRecord.id)]

    with session_factory() as session:
        with pytest.raises(RuntimeError, match="database_match_must_be_exactly_one"):
            TranscribeService(settings)._claim_transcribe_batch(
                session,
                limit=1,
                worker_id="must-not-run",
            )

    with session_factory() as session:
        after = [_row_state(call) for call in session.query(CallRecord).order_by(CallRecord.id)]
    assert after == before


def test_controlled_worker_rejects_replaced_target_record_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    db_path = tmp_path / "replaced-target.sqlite"
    settings = _controlled_settings(
        tmp_path,
        database_url=f"sqlite:///{db_path}",
    )
    init_db(settings)
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        session.add_all(
            [
                CallRecord(
                    source_file="other.wav",
                    source_filename="other.wav",
                    source_call_id="OTHER",
                    transcription_status="pending",
                    resolve_status="pending",
                    analysis_status="pending",
                    sync_status="pending",
                ),
                CallRecord(
                    source_file="target.wav",
                    source_filename="target.wav",
                    source_call_id="TARGET",
                    transcription_status="pending",
                    resolve_status="pending",
                    analysis_status="pending",
                    sync_status="pending",
                ),
            ]
        )
        session.commit()
        before = [
            _row_state(call)
            for call in session.query(CallRecord).order_by(CallRecord.id)
        ]

    with sqlite3.connect(db_path) as con:
        con.execute(
            "UPDATE call_records SET id=99 WHERE source_call_id='TARGET'"
        )

    with session_factory() as session:
        with pytest.raises(
            RuntimeError,
            match="database_match_must_be_exactly_one",
        ):
            TranscribeService(settings)._claim_transcribe_batch(
                session,
                limit=1,
                worker_id="must-not-run",
            )

    with session_factory() as session:
        after = [
            _row_state(call)
            for call in session.query(CallRecord).order_by(CallRecord.id)
        ]
    assert after == before


def test_controlled_scope_forbids_broad_cli_and_sync_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    settings = _controlled_settings(tmp_path)

    for command in (
        "ingest",
        "run-all",
        "sync",
        "requeue-dead",
        "requeue",
        "reset-analysis",
        "reset-transcribe",
        "init-db",
        "transcribe",
        "backfill-second-asr",
        "resolve",
        "analyze",
    ):
        with pytest.raises(RuntimeError, match="forbids_cli_command"):
            enforce_controlled_cli_command(settings, command)
    with pytest.raises(RuntimeError, match="forbids_worker_stages"):
        enforce_controlled_worker_stages(
            settings,
            ["transcribe", "sync"],
            stage_limit=1,
        )


def test_service_worker_environment_clears_inherited_controlled_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MANGO_CALLS_PROCESSING_SCOPE", "controlled_1")
    monkeypatch.setenv("MANGO_CALLS_CONTROLLED_ALLOWLIST_PATH", "/tmp/stale")
    monkeypatch.delenv("LLM_CACHE_ENABLED", raising=False)
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    timeline_root = tmp_path / "timeline"
    timeline_root.mkdir()
    config = CallsTwoProcessesConfig(
        pipeline_root=tmp_path / "pipeline",
        timeline_db=timeline_root / "timeline.sqlite",
        timeline_allowed_root=timeline_root,
        python_executable=Path(os.sys.executable),
        codex_binary=Path(os.sys.executable),
        codex_home_root=tmp_path / "codex",
    )

    environment = worker_environment(config)

    assert environment["MANGO_CALLS_PROCESSING_SCOPE"] == "service"
    assert environment["MANGO_CALLS_CONTROLLED_ALLOWLIST_PATH"] == ""
    assert environment["MANGO_CALLS_CONTROLLED_ALLOWLIST_SHA256"] == ""
    assert environment["MANGO_CALLS_CONTROLLED_TENANT_ID"] == ""
    assert environment["MANGO_CALLS_CONTROLLED_CODE_SHA"] == ""
    assert environment["MANGO_CALLS_CONTROLLED_HOST_ID"] == ""
    assert environment["MANGO_CALLS_CONTROLLED_HOST_ID_PATH"] == ""
    assert environment["MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_PATH"] == ""
    assert environment["MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_SHA256"] == ""
    assert "LLM_CACHE_ENABLED" not in environment
    assert "LLM_CACHE_DIR" not in environment


def test_controlled_worker_stays_direct_child_of_orchestrator(
    tmp_path: Path,
) -> None:
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    command = [str(config.python_executable), "-m", "mango_mvp.cli"]

    assert stage_subprocess_command(config, command) == command


def _controlled_runtime_config(
    root: Path,
    allowlist_path: Path,
    allowlist_sha256: str,
) -> CallsTwoProcessesConfig:
    owner_local = root / ".mango_local"
    host_id_path = owner_local / "pipeline" / "state" / "host_id"
    host_id_path.parent.mkdir(parents=True, exist_ok=True)
    host_id_path.write_text("m1-host\n", encoding="utf-8")
    host_id_path.chmod(0o600)
    cutover_path = owner_local / "pipeline" / "state" / "cutover_manifest.json"
    cutover_path.write_text("{}\n", encoding="utf-8")
    cutover_path.chmod(0o600)
    timeline_root = root / "timeline-staging"
    timeline_root.mkdir(parents=True, exist_ok=True)
    return CallsTwoProcessesConfig(
        pipeline_root=owner_local / "pipeline",
        timeline_db=timeline_root / "timeline.sqlite",
        timeline_allowed_root=timeline_root,
        python_executable=Path(os.sys.executable),
        codex_binary=Path(os.sys.executable),
        codex_home_root=owner_local / "codex-runtime",
        tenant_id="foton",
        min_free_gib=1,
        stage_limit=1,
        processing_scope="controlled_1",
        controlled_call_allowlist_path=allowlist_path,
        controlled_call_allowlist_sha256=allowlist_sha256,
        expected_code_sha=current_git_sha(REPO_ROOT),
        expected_active_host_id="m1-host",
        expected_previous_host_id="old-mac-host",
        host_id_path=host_id_path,
        require_cutover_authority=True,
        strict_ready_provenance=True,
    )


def _strict_worker_settings(settings: Settings, *, stage: str) -> Settings:
    return replace(
        settings,
        transcribe_provider="mlx",
        dual_transcribe_enabled=stage != "transcribe",
        secondary_transcribe_provider=(
            "gigaam" if stage != "transcribe" else None
        ),
        gigaam_model="v2_rnnt",
        dual_merge_provider="rule",
        mono_role_assignment_mode="rule",
        resolve_llm_provider="codex_cli",
        resolve_dialogue_mode="dialogue",
        resolve_rescue_provider=None,
        resolve_rescue_dual_enabled=False,
        analyze_provider="codex_cli",
        split_stereo_channels=True,
        llm_cache_enabled=False,
    )


def test_controlled_worker_disables_inherited_llm_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_CACHE_ENABLED", "1")
    monkeypatch.setenv("LLM_CACHE_DIR", "/tmp/inherited-customer-cache")
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )

    environment = worker_environment(config)

    assert environment["LLM_CACHE_ENABLED"] == "0"
    assert environment["LLM_CACHE_DIR"] == str(
        config.pipeline_root / "state" / "controlled_llm_cache_disabled"
    )


def test_controlled_worker_requires_fresh_orchestrator_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("MANGO_STRICT_ASR_RUNTIME", "1")
    monkeypatch.setenv("MANGO_CODEX_SERVICE_TIER", "flex")
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    settings = _strict_worker_settings(
        _controlled_settings(tmp_path),
        stage="transcribe",
    )

    with pytest.raises(RuntimeError, match="run_authority_missing"):
        enforce_controlled_worker_stages(
            settings,
            ["transcribe"],
            stage_limit=1,
        )

    with process_lease(config.pipeline_lock, stale_seconds=60):
        with controlled_worker_authority_environment(
            config,
            stage="transcribe",
            run_id="20260811T210000Z_controlled_test",
        ) as authority_env:
            authorized = replace(
                settings,
                controlled_call_run_authority_path=authority_env[
                    "MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_PATH"
                ],
                controlled_call_run_authority_sha256=authority_env[
                    "MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_SHA256"
                ],
            )
            monkeypatch.setattr(
                controlled_scope_module.os,
                "getppid",
                lambda: os.getpid(),
            )
            enforce_controlled_worker_stages(
                authorized,
                ["transcribe"],
                stage_limit=1,
            )

    with pytest.raises(RuntimeError, match="unsafe_or_missing"):
        enforce_controlled_worker_stages(
            authorized,
            ["transcribe"],
            stage_limit=1,
        )


def test_controlled_worker_ticket_uses_verified_cutover_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    verified_digest = hashlib.sha256(
        config.cutover_manifest_file.read_bytes()
    ).hexdigest()

    def proof_then_change_manifest(config_value, *_args, **_kwargs):
        config_value.cutover_manifest_file.write_text(
            '{"changed":true}', encoding="utf-8"
        )
        config_value.cutover_manifest_file.chmod(0o600)
        return {
            "ok": True,
            "active_host_id": "m1-host",
            "verified_cutover_manifest_sha256": verified_digest,
        }

    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        proof_then_change_manifest,
    )

    with process_lease(config.pipeline_lock, stale_seconds=60):
        with controlled_worker_authority_environment(
            config,
            stage="transcribe",
            run_id="20260811T210000Z_digest_test",
        ) as environment:
            ticket = json.loads(
                Path(
                    environment[
                        "MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_PATH"
                    ]
                ).read_text(encoding="utf-8")
            )

    assert ticket["cutover_manifest_sha256"] == verified_digest
    assert ticket["cutover_manifest_sha256"] != hashlib.sha256(
        config.cutover_manifest_file.read_bytes()
    ).hexdigest()


def test_controlled_worker_rejects_wrong_provider_and_stage_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("MANGO_STRICT_ASR_RUNTIME", "1")
    monkeypatch.setenv("MANGO_CODEX_SERVICE_TIER", "flex")
    settings = _controlled_settings(tmp_path)

    with pytest.raises(RuntimeError, match="stage_limit_one"):
        enforce_controlled_worker_stages(
            settings,
            ["transcribe"],
            stage_limit=2,
        )
    with pytest.raises(RuntimeError, match="transcribe_provider"):
        enforce_controlled_worker_stages(
            settings,
            ["transcribe"],
            stage_limit=1,
        )
    cached = replace(
        _strict_worker_settings(settings, stage="transcribe"),
        llm_cache_enabled=True,
    )
    with pytest.raises(RuntimeError, match="llm_cache_must_be_disabled"):
        enforce_controlled_worker_stages(
            cached,
            ["transcribe"],
            stage_limit=1,
        )


def _init_controlled_runtime_db(config: CallsTwoProcessesConfig) -> None:
    config.working_audio_dir.mkdir(parents=True, exist_ok=True)
    other_audio = config.working_audio_dir / "other.wav"
    target_audio = config.working_audio_dir / "target.wav"
    other_audio.write_bytes(b"synthetic-other-audio")
    target_audio.write_bytes(b"synthetic-target-audio")
    settings = replace(
        make_settings(),
        database_url=f"sqlite:///{config.working_db}",
    )
    init_db(settings)
    session_factory = build_session_factory(settings)
    with session_factory() as session:
        session.add_all(
            [
                CallRecord(
                    source_file=str(other_audio),
                    source_filename="other.wav",
                    source_call_id="OTHER",
                    transcription_status="pending",
                    resolve_status="pending",
                    analysis_status="pending",
                    sync_status="pending",
                ),
                CallRecord(
                    source_file=str(target_audio),
                    source_filename="target.wav",
                    source_call_id="TARGET",
                    transcription_status="pending",
                    resolve_status="pending",
                    analysis_status="pending",
                    sync_status="pending",
                ),
            ]
        )
        session.commit()


@pytest.mark.parametrize("unsafe_kind", ["missing", "empty", "symlink", "outside"])
def test_controlled_snapshot_rejects_unusable_pending_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    target = config.working_audio_dir / "target.wav"
    if unsafe_kind == "missing":
        target.unlink()
    elif unsafe_kind == "empty":
        target.write_bytes(b"")
    elif unsafe_kind == "symlink":
        real = target.with_name("target-real.wav")
        target.replace(real)
        target.symlink_to(real)
    else:
        outside = tmp_path / "outside.wav"
        target.replace(outside)
        with sqlite3.connect(config.working_db) as con:
            con.execute(
                "UPDATE call_records SET source_file=? WHERE source_call_id='TARGET'",
                (str(outside),),
            )

    with pytest.raises(RuntimeError):
        controlled_call_database_snapshot(
            config.working_db,
            "TARGET",
            working_audio_dir=config.working_audio_dir,
            require_source_audio=True,
        )


def test_controlled_snapshot_rejects_missing_audio_after_transcription_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    (config.working_audio_dir / "target.wav").unlink()
    with sqlite3.connect(config.working_db) as con:
        con.execute(
            "UPDATE call_records SET transcription_status='done' "
            "WHERE source_call_id='TARGET'"
        )

    with pytest.raises(RuntimeError):
        controlled_call_database_snapshot(
            config.working_db,
            "TARGET",
            working_audio_dir=config.working_audio_dir,
            require_source_audio=True,
        )


def test_controlled_snapshot_accepts_pipeline_hardlinked_audio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    target = config.working_audio_dir / "target.wav"
    capture_copy = config.recordings_dir / "target.wav"
    capture_copy.parent.mkdir(parents=True, exist_ok=True)
    os.link(target, capture_copy)

    snapshot = controlled_call_database_snapshot(
        config.working_db,
        "TARGET",
        working_audio_dir=config.working_audio_dir,
        require_source_audio=True,
    )

    assert snapshot["target"]["source_audio"]["ready"] is True


def _synthetic_controlled_runner(
    config: CallsTwoProcessesConfig,
    calls: list[str],
):
    def runner(command: list[str], env: dict[str, str], _cwd: Path) -> dict[str, object]:
        stage = command[command.index("--stages") + 1]
        assert env["LLM_CACHE_ENABLED"] == "0"
        assert env["TMPDIR"] == env["MANGO_CODEX_PROCESS_TMPDIR"]
        authority_path = Path(env["MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_PATH"])
        assert authority_path.is_file()
        assert hashlib.sha256(authority_path.read_bytes()).hexdigest() == env[
            "MANGO_CALLS_CONTROLLED_RUN_AUTHORITY_SHA256"
        ]
        calls.append(stage)
        processed = 0
        with sqlite3.connect(config.working_db) as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                "SELECT * FROM call_records WHERE source_call_id='TARGET'"
            ).fetchone()
            assert row is not None
            if stage == "transcribe" and row["transcription_status"] != "done":
                con.execute(
                    """
                    UPDATE call_records
                       SET transcription_status='done', resolve_status='pending',
                           transcript_text='synthetic target dialogue',
                           transcript_variants_json=?
                     WHERE source_call_id='TARGET'
                    """,
                    (
                        json.dumps(
                            {
                                "mode": "mono_or_fallback",
                                "primary_provider": "mlx",
                                "secondary_provider": "gigaam",
                                "full": {
                                    "variant_a": "synthetic target dialogue",
                                    "variant_b": "",
                                },
                            }
                        ),
                    ),
                )
                processed = 1
            elif stage == "backfill-second-asr":
                variants = json.loads(str(row["transcript_variants_json"] or "{}"))
                full = variants.get("full") if isinstance(variants, dict) else None
                if isinstance(full, dict) and not str(full.get("variant_b") or ""):
                    full["variant_b"] = "synthetic target dialogue"
                    con.execute(
                        "UPDATE call_records SET transcript_variants_json=? "
                        "WHERE source_call_id='TARGET'",
                        (json.dumps(variants),),
                    )
                    processed = 1
            elif stage == "resolve" and row["resolve_status"] not in {"done", "skipped"}:
                con.execute(
                    "UPDATE call_records SET resolve_status='done', resolve_json='{}' "
                    "WHERE source_call_id='TARGET'"
                )
                processed = 1
            elif stage == "analyze" and row["analysis_status"] != "done":
                con.execute(
                    "UPDATE call_records SET analysis_status='done', "
                    "analysis_json='{\"summary\":\"synthetic\"}' "
                    "WHERE source_call_id='TARGET'"
                )
                processed = 1
        return {
            "rc": 0,
            "command": f"worker:{stage}",
            "metrics": {
                "processed": processed,
                "success": processed,
                "failed": 0,
                "cycles": 2 if processed else 1,
                "runtime_receipt": {
                    "provider_invocations": (
                        {"mlx": 1}
                        if stage == "transcribe" and processed
                        else {"gigaam": 1}
                        if stage == "backfill-second-asr" and processed
                        else {}
                    ),
                    "mlx_cache_release_attempts": (
                        1 if stage == "transcribe" and processed else 0
                    ),
                    "mlx_cache_release_successes": (
                        1 if stage == "transcribe" and processed else 0
                    ),
                },
            },
        }

    return runner


def test_controlled_one_report_proves_non_target_invariant_and_zero_repeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    calls: list[str] = []
    runner = _synthetic_controlled_runner(config, calls)

    first = run_controlled_one(config, command_runner=runner)
    second = run_controlled_one(config, command_runner=runner)

    assert first["status"] == "ok"
    assert first["machine_result_ready_for_human_review"] is True
    assert first["execution_class"] == "transitioned_to_ready"
    assert first["fresh_asr_sequence_proven"] is True
    assert first["pilot_transition_proven"] is True
    assert first["non_target_rows_unchanged"] is True
    assert first["controlled_1_human_pass"] is False
    assert first["business_pass"] is False
    assert first["runtime_pass"] is False
    assert [item["metrics"]["processed"] for item in first["stages"]] == [1, 1, 1, 1]
    assert second["status"] == "ok"
    assert second["execution_class"] == "idempotent_noop"
    assert second["fresh_asr_sequence_proven"] is False
    assert second["pilot_transition_proven"] is False
    assert second["safety"]["runs_asr"] is False
    assert second["safety"]["runs_resolve_analyze"] is False
    assert [item["metrics"]["processed"] for item in second["stages"]] == [0, 0, 0, 0]
    assert second["before"]["target_row_sha256"] == second["after"]["target_row_sha256"]
    assert Path(str(first["report_path"])).stat().st_mode & 0o777 == 0o600
    assert calls == [*list(("transcribe", "backfill-second-asr", "resolve", "analyze"))] * 2


def test_controlled_one_stops_before_next_stage_when_allowlist_drifts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    called: list[str] = []

    def drifting_runner(command: list[str], _env: dict[str, str], _cwd: Path) -> dict[str, object]:
        stage = command[command.index("--stages") + 1]
        called.append(stage)
        allowlist_path.write_text("{}", encoding="utf-8")
        allowlist_path.chmod(0o600)
        return {"rc": 0, "command": f"worker:{stage}", "metrics": {"processed": 0}}

    report = run_controlled_one(config, command_runner=drifting_runner)

    assert report["status"] == "failed"
    assert called == ["transcribe"]


@pytest.mark.parametrize(
    "bad_metrics",
    [
        {},
        {"processed": 1, "success": 0, "failed": 1},
        {"processed": 2, "success": 2, "failed": 0},
    ],
)
def test_controlled_one_stops_after_invalid_zero_exit_stage_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_metrics: dict[str, int],
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    called: list[str] = []
    inner = _synthetic_controlled_runner(config, called)

    def invalid_metrics_runner(
        command: list[str],
        env: dict[str, str],
        cwd: Path,
    ) -> dict[str, object]:
        result = inner(command, env, cwd)
        if command[command.index("--stages") + 1] == "backfill-second-asr":
            result["metrics"] = bad_metrics
        return result

    report = run_controlled_one(config, command_runner=invalid_metrics_runner)

    assert report["status"] == "failed"
    assert called == ["transcribe", "backfill-second-asr"]
    assert report["stages"][-1]["controlled_stage_contract_ok"] is False


def test_real_worker_log_parser_preserves_controlled_stage_receipts(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "worker.log"
    payload = {
        "totals": {
            "transcribe": {
                "processed": 1,
                "success": 1,
                "failed": 0,
            }
        },
        "runtime_receipts": {
            "transcribe": {
                "provider_invocations": {"mlx": 1},
                "mlx_cache_release_attempts": 1,
                "mlx_cache_release_successes": 1,
            }
        },
        "cycles": 2,
        "idle_cycles": 1,
        "stop_reason": "idle",
    }
    log_path.write_text(
        "progress: stage started\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n  12345678  maximum resident set size\n",
        encoding="utf-8",
    )

    metrics = parse_worker_stage_metrics(log_path, "transcribe")

    assert metrics["processed"] == 1
    assert metrics["success"] == 1
    assert metrics["failed"] == 0
    assert metrics["runtime_receipt"]["provider_invocations"] == {"mlx": 1}
    assert metrics["runtime_receipt"]["mlx_cache_release_successes"] == 1


def test_real_worker_log_parser_rejects_malformed_counts(
    tmp_path: Path,
) -> None:
    base = {
        "totals": {
            "transcribe": {
                "processed": 1,
                "success": 1,
                "failed": 0,
            }
        },
        "runtime_receipts": {
            "transcribe": {
                "provider_invocations": {"mlx": 1},
                "mlx_cache_release_attempts": 1,
                "mlx_cache_release_successes": 1,
            }
        },
        "cycles": 2,
        "idle_cycles": 1,
        "stop_reason": "idle",
    }
    variants: list[dict[str, object]] = []
    for path, bad_value in (
        (("totals", "transcribe", "processed"), -1),
        (("totals", "transcribe", "success"), True),
        (("totals", "transcribe", "failed"), "0"),
        (("runtime_receipts", "transcribe", "provider_invocations", "mlx"), True),
        (("runtime_receipts", "transcribe", "mlx_cache_release_attempts"), -1),
        (("cycles",), False),
    ):
        payload = json.loads(json.dumps(base))
        cursor: dict[str, object] = payload
        for key in path[:-1]:
            child = cursor[key]
            assert isinstance(child, dict)
            cursor = child
        cursor[path[-1]] = bad_value
        variants.append(payload)

    for index, payload in enumerate(variants):
        log_path = tmp_path / f"bad-worker-{index}.log"
        log_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        assert parse_worker_stage_metrics(log_path, "transcribe") == {}


def test_real_worker_log_parser_requires_complete_runtime_receipt(
    tmp_path: Path,
) -> None:
    base = {
        "totals": {
            "transcribe": {
                "processed": 1,
                "success": 1,
                "failed": 0,
            }
        },
        "runtime_receipts": {
            "transcribe": {
                "provider_invocations": {"mlx": 1},
                "mlx_cache_release_attempts": 1,
                "mlx_cache_release_successes": 1,
            }
        },
        "cycles": 2,
        "idle_cycles": 1,
        "stop_reason": "idle",
    }
    variants: list[dict[str, object]] = []

    without_receipts = json.loads(json.dumps(base))
    del without_receipts["runtime_receipts"]
    variants.append(without_receipts)

    without_stage_receipt = json.loads(json.dumps(base))
    without_stage_receipt["runtime_receipts"] = {}
    variants.append(without_stage_receipt)

    without_receipt_counter = json.loads(json.dumps(base))
    del without_receipt_counter["runtime_receipts"]["transcribe"][
        "mlx_cache_release_successes"
    ]
    variants.append(without_receipt_counter)

    zero_cycles = json.loads(json.dumps(base))
    zero_cycles["cycles"] = 0
    variants.append(zero_cycles)

    impossible_idle_cycles = json.loads(json.dumps(base))
    impossible_idle_cycles["idle_cycles"] = 3
    variants.append(impossible_idle_cycles)

    for index, payload in enumerate(variants):
        log_path = tmp_path / f"incomplete-worker-{index}.log"
        log_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metrics = parse_worker_stage_metrics(log_path, "transcribe")
        assert metrics == {}


def test_controlled_stage_contract_rejects_missing_or_boolean_rc(
    tmp_path: Path,
) -> None:
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    metrics = {"processed": 0, "success": 0, "failed": 0}

    missing = controlled_stage_report(config, {"metrics": metrics})
    boolean = controlled_stage_report(
        config,
        {"rc": False, "metrics": metrics},
    )

    assert missing["rc"] == 65
    assert boolean["rc"] == 65


def test_controlled_one_fails_if_source_audio_changes_during_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    inner = _synthetic_controlled_runner(config, [])

    def mutating_runner(
        command: list[str],
        env: dict[str, str],
        cwd: Path,
    ) -> dict[str, object]:
        result = inner(command, env, cwd)
        if command[command.index("--stages") + 1] == "transcribe":
            (config.working_audio_dir / "target.wav").write_bytes(
                b"mutated-after-preflight"
            )
        return result

    report = run_controlled_one(config, command_runner=mutating_runner)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "controlled_one_exception:RuntimeError"
    assert report["pilot_transition_proven"] is False


def test_controlled_one_asr_uses_private_snapshot_during_source_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    source = config.working_audio_dir / "target.wav"
    original = source.read_bytes()
    base_settings = _controlled_settings(tmp_path)
    observed_inputs: list[bytes] = []
    inner = _synthetic_controlled_runner(config, [])

    def swapping_runner(
        command: list[str],
        env: dict[str, str],
        cwd: Path,
    ) -> dict[str, object]:
        stage = command[command.index("--stages") + 1]
        if stage in {"transcribe", "backfill-second-asr"}:
            source.write_bytes(b"wrong-customer-audio")
            settings = replace(
                base_settings,
                controlled_call_audio_snapshot_path=env[
                    "MANGO_CALLS_CONTROLLED_AUDIO_SNAPSHOT_PATH"
                ],
                controlled_call_audio_snapshot_sha256=env[
                    "MANGO_CALLS_CONTROLLED_AUDIO_SNAPSHOT_SHA256"
                ],
                controlled_call_audio_snapshot_size_bytes=int(
                    env["MANGO_CALLS_CONTROLLED_AUDIO_SNAPSHOT_SIZE_BYTES"]
                ),
            )
            observed_inputs.append(
                controlled_audio_input_path(
                    settings,
                    record_id=2,
                    source_call_id="TARGET",
                    source_file=source,
                ).read_bytes()
            )
            source.write_bytes(original)
        return inner(command, env, cwd)

    report = run_controlled_one(config, command_runner=swapping_runner)

    assert report["status"] == "ok"
    assert observed_inputs == [original, original]
    assert report["asr_input_snapshot"]["sha256"] == hashlib.sha256(
        original
    ).hexdigest()


def test_controlled_one_preserves_transition_evidence_when_snapshot_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    called: list[str] = []
    inner = _synthetic_controlled_runner(config, called)
    injected = False

    def cleanup_breaking_runner(
        command: list[str],
        env: dict[str, str],
        cwd: Path,
    ) -> dict[str, object]:
        nonlocal injected
        if not injected:
            snapshot_path = Path(
                env["MANGO_CALLS_CONTROLLED_AUDIO_SNAPSHOT_PATH"]
            )
            leftover = snapshot_path.parent / "unexpected-owner-file"
            leftover.write_bytes(b"synthetic-cleanup-blocker")
            leftover.chmod(0o600)
            injected = True
        return inner(command, env, cwd)

    report = run_controlled_one(
        config,
        command_runner=cleanup_breaking_runner,
    )

    assert report["status"] == "failed"
    assert report["stop_reason"] == (
        "controlled_call_audio_snapshot_cleanup_failed"
    )
    assert report["execution_class"] == "transitioned_to_ready"
    assert report["machine_result_ready_for_human_review"] is True
    assert report["pilot_transition_proven"] is False
    assert report["after"]["target"]["ready_for_human_review"] is True
    assert called == [
        "transcribe",
        "backfill-second-asr",
        "resolve",
        "analyze",
    ]
    assert len(report["stages"]) == 4
    assert report["asr_input_snapshot_cleanup"] == {
        "ok": False,
        "snapshot_integrity_ok": True,
        "snapshot_removed": True,
        "run_directory_removed": False,
        "errors": ["controlled_call_audio_snapshot_cleanup_failed"],
    }


def test_controlled_one_reports_snapshot_tamper_without_losing_stage_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    called: list[str] = []
    inner = _synthetic_controlled_runner(config, called)

    def tampering_runner(
        command: list[str],
        env: dict[str, str],
        cwd: Path,
    ) -> dict[str, object]:
        result = inner(command, env, cwd)
        if command[command.index("--stages") + 1] == "analyze":
            snapshot_path = Path(
                env["MANGO_CALLS_CONTROLLED_AUDIO_SNAPSHOT_PATH"]
            )
            snapshot_path.write_bytes(b"synthetic-tamper")
            snapshot_path.chmod(0o600)
        return result

    report = run_controlled_one(config, command_runner=tampering_runner)

    assert report["status"] == "failed"
    assert report["stop_reason"] == (
        "controlled_call_audio_snapshot_cleanup_failed"
    )
    assert report["execution_class"] == "transitioned_to_ready"
    assert report["pilot_transition_proven"] is False
    assert len(report["stages"]) == 4
    assert report["after"]["target"]["ready_for_human_review"] is True
    assert report["asr_input_snapshot_cleanup"] == {
        "ok": False,
        "snapshot_integrity_ok": False,
        "snapshot_removed": True,
        "run_directory_removed": True,
        "errors": ["controlled_call_audio_snapshot_changed_during_run"],
    }


def test_controlled_one_preserves_evidence_on_snapshot_integrity_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    import mango_mvp.customer_timeline.calls_two_processes as calls_runtime

    original_inspect = calls_runtime.inspect_stable_regular_file

    def failing_snapshot_inspect(path: Path, **kwargs: object):
        if "controlled_runs" in path.parts:
            raise OSError("synthetic cleanup I/O failure")
        return original_inspect(path, **kwargs)

    monkeypatch.setattr(
        calls_runtime,
        "inspect_stable_regular_file",
        failing_snapshot_inspect,
    )
    called: list[str] = []

    report = run_controlled_one(
        config,
        command_runner=_synthetic_controlled_runner(config, called),
    )

    assert report["status"] == "failed"
    assert report["pilot_transition_proven"] is False
    assert len(report["stages"]) == 4
    assert report["after"]["target"]["ready_for_human_review"] is True
    assert report["asr_input_snapshot_cleanup"] == {
        "ok": False,
        "snapshot_integrity_ok": False,
        "snapshot_removed": True,
        "run_directory_removed": True,
        "errors": ["controlled_call_audio_snapshot_integrity_unproven"],
    }


def test_snapshot_cleanup_oserror_does_not_mask_primary_stage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    import mango_mvp.customer_timeline.calls_two_processes as calls_runtime

    original_inspect = calls_runtime.inspect_stable_regular_file

    def failing_snapshot_inspect(path: Path, **kwargs: object):
        if "controlled_runs" in path.parts:
            raise OSError("synthetic cleanup I/O failure")
        return original_inspect(path, **kwargs)

    monkeypatch.setattr(
        calls_runtime,
        "inspect_stable_regular_file",
        failing_snapshot_inspect,
    )

    def failing_runner(
        _command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> dict[str, object]:
        raise RuntimeError("controlled_primary_failure")

    report = run_controlled_one(config, command_runner=failing_runner)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "controlled_one_exception:RuntimeError"
    assert report["diagnostic"] == {
        "type": "RuntimeError",
        "code": "controlled_primary_failure",
    }


def test_controlled_one_rejects_audio_changed_after_allowlist_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    (config.working_audio_dir / "target.wav").write_bytes(
        b"changed-after-allowlist"
    )
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    called: list[str] = []

    report = run_controlled_one(
        config,
        command_runner=_synthetic_controlled_runner(config, called),
    )

    assert report["status"] == "failed"
    assert report["stop_reason"] == "controlled_one_exception:RuntimeError"
    assert report["diagnostic"]["code"] == (
        "controlled_one_allowlist_target_binding_mismatch"
    )
    assert called == []


def test_controlled_one_stops_on_stale_private_audio_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    stale = (
        config.pipeline_root
        / "state"
        / "controlled_runs"
        / "interrupted"
    )
    stale.mkdir(parents=True, mode=0o700)
    (stale / "input.wav").write_bytes(b"stale-private-audio")
    (stale / "input.wav").chmod(0o600)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    called: list[str] = []

    report = run_controlled_one(
        config,
        command_runner=_synthetic_controlled_runner(config, called),
    )

    assert report["status"] == "failed"
    assert report["stop_reason"] == "controlled_one_exception:RuntimeError"
    assert report["diagnostic"]["code"] == (
        "controlled_call_audio_snapshot_stale_artifacts_present"
    )
    assert called == []


def test_controlled_one_does_not_claim_fresh_pilot_without_asr_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    inner = _synthetic_controlled_runner(config, [])

    def receipt_free_runner(
        command: list[str],
        env: dict[str, str],
        cwd: Path,
    ) -> dict[str, object]:
        result = inner(command, env, cwd)
        metrics = result.get("metrics")
        assert isinstance(metrics, dict)
        metrics["runtime_receipt"] = {
            "provider_invocations": {},
            "mlx_cache_release_attempts": 0,
            "mlx_cache_release_successes": 0,
        }
        return result

    report = run_controlled_one(config, command_runner=receipt_free_runner)

    assert report["status"] == "ok"
    assert report["execution_class"] == "transitioned_to_ready"
    assert report["fresh_asr_sequence_proven"] is False
    assert report["pilot_transition_proven"] is False
    assert report["safety"]["runs_asr"] is False


def test_controlled_one_rejects_zero_metrics_when_target_row_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    allowlist_path, allowlist_sha256 = _write_allowlist(tmp_path)
    config = _controlled_runtime_config(
        tmp_path,
        allowlist_path,
        allowlist_sha256,
    )
    _init_controlled_runtime_db(config)
    monkeypatch.setattr(
        "mango_mvp.customer_timeline.calls_two_processes.controlled_read_only_cutover_authority_report",
        _controlled_authority_ok,
    )
    first = run_controlled_one(
        config,
        command_runner=_synthetic_controlled_runner(config, []),
    )
    assert first["status"] == "ok"

    changed = False

    def false_zero_runner(
        command: list[str],
        _env: dict[str, str],
        _cwd: Path,
    ) -> dict[str, object]:
        nonlocal changed
        stage = command[command.index("--stages") + 1]
        if not changed:
            with sqlite3.connect(config.working_db) as con:
                con.execute(
                    "UPDATE call_records SET manager_name='changed' "
                    "WHERE source_call_id='TARGET'"
                )
            changed = True
        return {
            "rc": 0,
            "command": f"worker:{stage}",
            "metrics": {
                "processed": 0,
                "success": 0,
                "failed": 0,
                "runtime_receipt": {
                    "provider_invocations": {},
                    "mlx_cache_release_attempts": 0,
                    "mlx_cache_release_successes": 0,
                },
            },
        }

    report = run_controlled_one(config, command_runner=false_zero_runner)

    assert report["status"] == "failed"
    assert report["stop_reason"] == "controlled_one_exception:RuntimeError"
    assert report["pilot_transition_proven"] is False
