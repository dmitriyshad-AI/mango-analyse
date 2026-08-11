from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sqlite3
import stat
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import pytest

from mango_mvp.customer_timeline import calls_two_processes as calls
from mango_mvp.productization.capture_staging import (
    CaptureManifestStore,
    load_capture_recovery,
    record_capture_recovery,
)
from scripts import export_daily_mango_calls_resolve as daily_export
from scripts import relocate_mango_calls_pipeline as relocation


@pytest.mark.parametrize(
    "inventory_option",
    ("--inventory-root", "--inventory-out", "--verify-inventory"),
)
def test_relocation_cli_rejects_inventory_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    inventory_option: str,
) -> None:
    with pytest.raises(SystemExit) as raised:
        relocation.parse_args(
            [
                "--pipeline-root",
                str(tmp_path / "transfer"),
                "--old-root",
                str(tmp_path / "source"),
                "--new-root",
                str(tmp_path / "target"),
                "--source-inventory",
                str(tmp_path / "source_inventory.json"),
                "--dry-run",
                inventory_option,
                str(tmp_path / "foreign_mode_value"),
            ]
        )

    assert raised.value.code == 2
    assert "relocation mode cannot be combined with inventory options" in capsys.readouterr().err


def _manifest_row(
    event: str,
    status: str,
    *,
    old_root: Path,
    local_audio_path: str | None = None,
    canonical_audio_path: str | None = None,
    recording_paths: list[str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": "capture_manifest_v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tenant_id": "synthetic",
        "provider": "mango_office",
        "event_key": event,
        "provider_call_id": event,
        "recording_id": f"recording-{event}",
        "recording_ids": [f"recording-{event}"],
        "started_at": "2026-01-01T00:00:00+00:00",
        "ended_at": "2026-01-01T00:01:00+00:00",
        "direction": "incoming",
        "manager_ref": f"negative control {old_root}/must-not-change",
        "status": status,
        "note_path": f"{old_root}/unknown-field-must-not-change",
    }
    if local_audio_path is not None:
        row["local_audio_path"] = local_audio_path
    if canonical_audio_path is not None:
        row["canonical_audio_path"] = canonical_audio_path
    if recording_paths is not None:
        row["recording_paths"] = recording_paths
        row["recording_ids"] = [f"recording-{event}-{index}" for index in range(len(recording_paths))]
        row["recording_assets"] = [
            {
                "recording_id": row["recording_ids"][index],
                "path": value,
                "size_bytes": Path(value).stat().st_size,
                "checksum_sha256": _sha(Path(value)),
            }
            for index, value in enumerate(recording_paths)
        ]
    return row


def _write_sqlite(path: Path, source_files: list[str], *, journal_mode: str, old_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute(f"PRAGMA journal_mode={journal_mode}").fetchone()[0].casefold() == journal_mode
        connection.executescript(
            """
            CREATE TABLE call_records (
                id INTEGER PRIMARY KEY,
                source_file TEXT NOT NULL UNIQUE,
                source_filename TEXT NOT NULL,
                source_call_id TEXT,
                audio_codec TEXT,
                sample_rate INTEGER,
                channels INTEGER,
                duration_sec REAL,
                phone TEXT,
                manager_name TEXT,
                direction TEXT,
                started_at TEXT,
                transcription_status TEXT NOT NULL,
                resolve_status TEXT NOT NULL,
                analysis_status TEXT NOT NULL,
                sync_status TEXT NOT NULL,
                transcribe_attempts INTEGER NOT NULL,
                resolve_attempts INTEGER NOT NULL,
                analyze_attempts INTEGER NOT NULL,
                sync_attempts INTEGER NOT NULL,
                pipeline_stage TEXT,
                pipeline_worker_id TEXT,
                pipeline_claimed_at TEXT,
                analysis_worker_id TEXT,
                analysis_claimed_at TEXT,
                next_retry_at TEXT,
                dead_letter_stage TEXT,
                transcript_manager TEXT,
                transcript_client TEXT,
                transcript_text TEXT,
                transcript_variants_json TEXT,
                resolve_json TEXT,
                resolve_quality_score REAL,
                analysis_json TEXT,
                amocrm_contact_id INTEGER,
                amocrm_lead_id INTEGER,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE audit_notes (id INTEGER PRIMARY KEY, source_file TEXT, note TEXT);
            """
        )
        for index, source_file in enumerate(source_files, start=1):
            done = index == 1
            transcript = "MANAGER: Добрый день\nCLIENT: Нужна программа курса" if done else None
            variants = (
                json.dumps(
                    {
                        "call_topology": "simple_two_party",
                        "role_mapping": {
                            "confirmed": True,
                            "manager_quality_allowed": True,
                            "topology": "simple_two_party",
                        },
                        "manager": {
                            "physical_channel": "left",
                            "variant_a": "Добрый день",
                            "variant_b": "Добрый день",
                        },
                        "client": {
                            "physical_channel": "right",
                            "variant_a": "Нужна программа курса",
                            "variant_b": "Нужна программа курса",
                        },
                        "dialogue_lines": [
                            "[00:00.000] Менеджер: Добрый день",
                            "[00:01.000] Клиент: Нужна программа курса",
                        ],
                        "primary_provider": "mlx",
                        "secondary_provider": "gigaam",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if done
                else None
            )
            resolve_json = json.dumps({"decision": "resolved", "confidence": 0.99}, sort_keys=True) if done else None
            analysis_json = (
                json.dumps(
                    {
                        "call_type": "sales_call",
                        "history_summary": "Менеджер обсудил программу курса.",
                        "next_step_action": "Отправить программу.",
                        "quality_flags": {},
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if done
                else None
            )
            connection.execute(
                """
                INSERT INTO call_records (
                    id, source_file, source_filename, source_call_id,
                    audio_codec, sample_rate, channels, duration_sec, phone,
                    manager_name, direction, started_at,
                    transcription_status, resolve_status, analysis_status, sync_status,
                    transcribe_attempts, resolve_attempts, analyze_attempts, sync_attempts,
                    pipeline_stage, transcript_manager, transcript_client, transcript_text,
                    transcript_variants_json, resolve_json, resolve_quality_score, analysis_json,
                    amocrm_contact_id, amocrm_lead_id, last_error, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, ?, 'mp3', 16000, 1, 60.5, ?, ?, 'inbound',
                    '2026-01-01T00:00:00+00:00', ?, ?, ?, 'pending', ?, ?, ?, 0,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    '2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00'
                )
                """,
                (
                    index,
                    source_file,
                    Path(source_file).name,
                    f"call-{chr(96 + index)}",
                    f"+7000000000{index}",
                    f"manager-{index}",
                    "done" if done else "pending",
                    "done" if done else "pending",
                    "done" if done else "pending",
                    2 if done else 1,
                    1 if done else 0,
                    1 if done else 0,
                    None if done else "transcribe",
                    "Добрый день" if done else None,
                    "Нужна программа курса" if done else None,
                    transcript,
                    variants,
                    resolve_json,
                    0.99 if done else None,
                    analysis_json,
                    1000 + index if done else None,
                    2000 + index if done else None,
                    f"control {old_root}/do-not-change",
                ),
            )
        connection.execute(
            "INSERT INTO audit_notes(source_file, note) VALUES (?, ?)",
            (f"{old_root}/unrelated/table/value.mp3", f"control {old_root}/text"),
        )
        connection.commit()
        assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        if journal_mode == "wal":
            assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    if journal_mode == "wal":
        for suffix in ("-wal", "-shm"):
            Path(str(path) + suffix).unlink(missing_ok=True)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _storage(path: Path) -> Mapping[str, Mapping[str, int]]:
    result: dict[str, Mapping[str, int]] = {}
    for label, candidate in (("db", path), ("wal", Path(str(path) + "-wal"))):
        if candidate.is_file():
            path_stat = candidate.stat()
            result[label] = {"size_bytes": path_stat.st_size, "mtime_ns": path_stat.st_mtime_ns}
    return result


def _sqlite_business_snapshot(path: Path) -> tuple[Any, ...]:
    with closing(sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)) as connection:
        schema = tuple(
            connection.execute(
                """
                SELECT type, name, tbl_name, COALESCE(sql, '')
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            )
        )
        tables: list[tuple[str, tuple[str, ...], tuple[tuple[Any, ...], ...]]] = []
        for (table_name,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ):
            quoted = '"' + str(table_name).replace('"', '""') + '"'
            cursor = connection.execute(f"SELECT * FROM {quoted}")
            columns = tuple(str(item[0]) for item in cursor.description or ())
            source_index = columns.index("source_file") if table_name == "call_records" else -1
            rows = []
            for raw in cursor:
                row = list(raw)
                if source_index >= 0:
                    row[source_index] = "<allowed-relocated-source-file>"
                rows.append(tuple(row))
            tables.append((str(table_name), columns, tuple(sorted(rows, key=repr))))
        return schema, tuple(tables)


def _publication_hash(ready_db: Path, working_db: Path) -> str:
    rows, _pending = daily_export.merged_day_rows(
        ready_db,
        working_db,
        date(2026, 1, 1),
        {"manager-1": "Синтетический менеджер", "manager-2": "Синтетический менеджер"},
        sealed_only=True,
    )
    return daily_export.publication_content_sha256(rows)


def _snapshot(root: Path) -> tuple[tuple[Any, ...], ...]:
    result: list[tuple[Any, ...]] = []
    for path in sorted((root, *root.rglob("*")), key=lambda item: str(item.relative_to(root))):
        relative = "." if path == root else path.relative_to(root).as_posix()
        path_stat = os.lstat(path)
        if stat.S_ISREG(path_stat.st_mode):
            result.append((relative, "file", path_stat.st_size, path_stat.st_mtime_ns, stat.S_IMODE(path_stat.st_mode), _sha(path)))
        elif stat.S_ISDIR(path_stat.st_mode):
            result.append((relative, "dir", path_stat.st_mtime_ns, stat.S_IMODE(path_stat.st_mode)))
        elif stat.S_ISLNK(path_stat.st_mode):
            result.append((relative, "symlink", os.readlink(path)))
        else:
            result.append((relative, "special", stat.S_IFMT(path_stat.st_mode)))
    return tuple(result)


def _private_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _source_inventory(pipeline: Path, output: Path, old_root: Path) -> Path:
    inventory = dict(relocation.build_inventory(pipeline))
    inventory["source_root"] = str(old_root)
    _private_json(output, inventory)
    return output


@pytest.fixture()
def synthetic_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Mapping[str, Any]:
    home = tmp_path / "home"
    local = home / ".mango_local"
    local.mkdir(parents=True, mode=0o700)
    local.chmod(0o700)
    monkeypatch.setenv("HOME", str(home))
    source = tmp_path / "source_mac" / "mango_calls_two_processes"
    pipeline = local / "mango_calls_transfers" / "generation-1"
    for relative in (
        "capture/recordings",
        "working/audio",
        "drop",
        "locks",
        "reports",
        "state",
    ):
        (source / relative).mkdir(parents=True, mode=0o755)

    for relative, payload in (
        ("capture/recordings/a.mp3", b"capture-a"),
        ("capture/recordings/b.mp3", b"capture-b"),
        ("working/audio/a.mp3", b"working-a"),
        ("working/audio/b.mp3", b"working-b"),
    ):
        target = source / relative
        target.write_bytes(payload)
        target.chmod(0o644)

    rows = [
        _manifest_row(
            "call-a",
            "downloaded",
            old_root=source,
            local_audio_path=str(source / "capture/recordings/a.mp3"),
        ),
        _manifest_row(
            "call-multi",
            "multiple_recordings_needs_review",
            old_root=source,
            recording_paths=[
                str(source / "capture/recordings/a.mp3"),
                str(source / "capture/recordings/b.mp3"),
            ],
        ),
        _manifest_row(
            "call-duplicate",
            "duplicate_recording",
            old_root=source,
            canonical_audio_path=str(source / "capture/recordings/a.mp3"),
        ),
        _manifest_row(
            "call-failed",
            "failed",
            old_root=source,
            local_audio_path=str(source / "capture/recordings/missing.mp3"),
        ),
    ]
    complete_prefix = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )
    torn_tail = b'{"schema_version":"capture_manifest_v1","created_at":"2026-08-08T12:00:00Z","tenant_id":"synthetic","provider":"mango_office","event_key":"unfinished","provider_call_id":"unfinished","started_at":"2026-08-08T12:00:00Z","direction":"incoming","status":"downlo\xd0'
    capture_manifest = source / relocation.CAPTURE_REL
    capture_manifest.write_bytes(complete_prefix + torn_tail)
    capture_manifest.chmod(0o644)

    source_files = [
        str(source / "working/audio/a.mp3"),
        str(source / "working/audio/b.mp3"),
    ]
    working_db = source / relocation.WORKING_DB_REL
    ready_db = source / relocation.READY_DB_REL
    _write_sqlite(working_db, source_files, journal_mode="wal", old_root=source)
    _write_sqlite(ready_db, source_files, journal_mode="delete", old_root=source)
    working_db.chmod(0o644)
    ready_db.chmod(0o644)
    Path(str(working_db) + "-wal").write_bytes(b"")
    Path(str(working_db) + "-shm").write_bytes(b"\0" * 32768)
    ready_manifest = {
        "schema_version": "mango_calls_two_processes_v1",
        "status": "ready",
        "published_at": "2026-08-08T00:00:00+00:00",
        "ready_db": str(source / relocation.READY_DB_REL),
        "sha256": _sha(ready_db),
        "size_bytes": ready_db.stat().st_size,
        "ready_mtime_ns": ready_db.stat().st_mtime_ns,
        "quick_check": "ok",
        "counts": {"total": 2, "pending": 2},
        "source_storage": _storage(working_db),
        "code_sha": "82208ad1e2c95ca0c8476ec3e9b88268ebb3d455",
    }
    _private_json(source / relocation.READY_MANIFEST_REL, ready_manifest)
    (source / relocation.READY_MANIFEST_REL).chmod(0o644)
    _private_json(
        source / relocation.CURSOR_REL,
        {
            "schema_version": "mango_api_freshness_v1",
            "until": "2026-08-08T00:00:00+00:00",
            "mango_enumeration_complete": True,
            "manifest_end_offset": len(complete_prefix),
            "manifest_snapshot_sha256": hashlib.sha256(complete_prefix).hexdigest(),
        },
    )
    (source / relocation.CURSOR_REL).chmod(0o644)
    (source / "locks/README.txt").write_text("synthetic, no process locks yet\n", encoding="utf-8")
    (source / "reports/control.json").write_text(
        json.dumps({"message": f"do not rewrite {source}/reports/control"}) + "\n",
        encoding="utf-8",
    )
    for path in source.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file() and path.name != capture_manifest.name:
            path.chmod(0o644)
    source.chmod(0o755)

    transfer_parent = pipeline.parent
    transfer_parent.mkdir(parents=True, mode=0o700)
    transfer_parent.chmod(0o700)
    source_inventory = transfer_parent / "generation-1.source_inventory.json"
    relocation.write_inventory(source, source_inventory)
    source_snapshot = _snapshot(source)
    shutil.copytree(source, pipeline, copy_function=shutil.copy2)
    assert relocation.verify_inventory(source_inventory, pipeline)["status"] == "verified"
    return {
        "home": home,
        "local": local,
        "source": source,
        "source_snapshot": source_snapshot,
        "pipeline": pipeline,
        "old": source,
        "new": local / "mango_calls_two_processes",
        "source_inventory": source_inventory,
        "torn_tail": torn_tail,
    }


def _relocate(fixture: Mapping[str, Any], *, execute: bool, checkpoint=lambda _name: None) -> Mapping[str, Any]:
    return relocation.relocate_pipeline(
        fixture["pipeline"],
        fixture["old"],
        fixture["new"],
        fixture["source_inventory"],
        execute=execute,
        confirmation=relocation.CONFIRM_VALUE if execute else "",
        checkpoint=checkpoint,
    )


def test_capture_tail_uses_the_runtime_recoverability_predicate() -> None:
    recoverable = b'{"event_key":"unfinished","status":"downlo\xd0'
    rows, tail = relocation.split_capture_rows(recoverable)
    assert rows == []
    assert tail == recoverable

    for corrupt in (b'{garbage\xd0', b'{"event_key": ??? \xd0'):
        with pytest.raises(relocation.RelocationError, match="invalid UTF-8"):
            relocation.split_capture_rows(corrupt)


def test_inventory_build_and_verify_are_root_independent_and_private(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    source = synthetic_pipeline["pipeline"]
    copied = synthetic_pipeline["local"] / "inventory-copy"
    shutil.copytree(source, copied, copy_function=shutil.copy2)
    output_parent = synthetic_pipeline["local"] / "private"
    output_parent.mkdir(mode=0o700)
    output = output_parent / "inventory.json"

    inventory = relocation.write_inventory(source, output)
    assert inventory["source_root"] == str(source)
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    verified = relocation.verify_inventory(output, copied)
    assert verified["status"] == "verified"

    fractional_mtime_seen = False
    copied_paths = sorted(
        (copied, *copied.rglob("*")),
        key=lambda item: (item.is_dir(), len(item.parts)),
    )
    for path in copied_paths:
        path_stat = path.stat()
        rounded = (path_stat.st_mtime_ns // 1_000_000_000) * 1_000_000_000
        fractional_mtime_seen = fractional_mtime_seen or rounded != path_stat.st_mtime_ns
        os.utime(path, ns=(path_stat.st_atime_ns, rounded))
    assert fractional_mtime_seen
    assert relocation.verify_inventory(output, copied)["status"] == "verified"

    mode_target = copied / "capture/recordings/b.mp3"
    mode_target.chmod(0o600)
    with pytest.raises(relocation.RelocationError, match="inventories differ"):
        relocation.verify_inventory(output, copied)
    mode_target.chmod(0o644)

    (copied / "capture/recordings/a.mp3").write_bytes(b"tampered")
    with pytest.raises(relocation.RelocationError, match="inventories differ"):
        relocation.verify_inventory(output, copied)


def test_selective_inventory_copies_exact_files_and_omits_only_strict_ready_audio(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    source = synthetic_pipeline["source"]
    private = synthetic_pipeline["local"] / "selective-contract"
    private.mkdir(mode=0o700)
    inventory_path = private / "inventory.json"
    files_from_path = private / "files-from.nul"

    inventory = relocation.write_selective_inventory(
        source, inventory_path, files_from_path
    )
    files_from = files_from_path.read_bytes()
    selected = [value.decode("utf-8") for value in files_from.split(b"\0") if value]

    assert "." not in selected
    assert selected == sorted(set(selected))
    assert relocation.CURSOR_REL in selected
    assert "working/audio/a.mp3" not in selected
    assert "working/audio/b.mp3" in selected
    assert "capture/recordings/a.mp3" in selected
    assert inventory["selection"]["omitted_audio"] == [
        next(
            item
            for item in relocation.build_inventory(source)["files"]
            if item["relative_path"] == "working/audio/a.mp3"
        )
    ]
    assert relocation.verify_selective_source(
        inventory_path, source, files_from_path
    )["status"] == "source_verified"

    target = synthetic_pipeline["local"] / "selective-copy"
    target.mkdir(mode=0o700)
    for relative in selected:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    assert relocation.verify_inventory(inventory_path, target)["status"] == "verified"

    tampered = json.loads(inventory_path.read_text(encoding="utf-8"))
    tampered["selection"]["omitted_audio"][0]["sha256"] = "0" * 64
    _private_json(inventory_path, tampered)
    with pytest.raises(relocation.RelocationError, match="omitted-audio digest"):
        relocation.verify_selective_source(inventory_path, source, files_from_path)


def test_selective_inventory_refuses_missing_unfinished_audio(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    source = synthetic_pipeline["source"]
    (source / "working/audio/b.mp3").unlink()

    with pytest.raises(relocation.RelocationError, match="unfinished or multi audio"):
        relocation.build_selective_inventory(source)


def test_selective_inventory_retains_unreferenced_audio_after_capture_crash(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    source = synthetic_pipeline["source"]
    capture_orphan = source / "capture/recordings/orphan-after-download.mp3"
    working_orphan = source / "working/audio/orphan-before-ingest-commit.mp3"
    capture_orphan.write_bytes(b"capture orphan must survive")
    working_orphan.write_bytes(b"working orphan must survive")

    _inventory, files_from = relocation.build_selective_inventory(source)
    selected = {
        value.decode("utf-8") for value in files_from.split(b"\0") if value
    }

    assert "capture/recordings/orphan-after-download.mp3" in selected
    assert "working/audio/orphan-before-ingest-commit.mp3" in selected


def test_inventory_documents_stay_out_of_external_git_and_cloud_locations(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    local = synthetic_pipeline["local"]
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)

    with pytest.raises(relocation.RelocationError, match="outside its allowed root"):
        relocation.write_inventory(pipeline, outside / "inventory.json")

    with pytest.raises(relocation.RelocationError, match="forbidden metadata location"):
        relocation.write_inventory(
            pipeline,
            local / "Yandex.Disk.localized" / "inventory.json",
        )

    repository = local / "inventory-repository"
    repository.mkdir(mode=0o700)
    (repository / ".git").mkdir(mode=0o700)
    with pytest.raises(relocation.RelocationError, match="outside Git"):
        relocation.write_inventory(pipeline, repository / "inventory.json")

    outside_link = local / "inventory-link"
    outside_link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(relocation.RelocationError, match="symlink component"):
        relocation.write_inventory(pipeline, outside_link / "inventory.json")

    external_inventory = outside / "source_inventory.json"
    shutil.copy2(synthetic_pipeline["source_inventory"], external_inventory)
    external_inventory.chmod(0o600)
    with pytest.raises(relocation.RelocationError, match="outside its allowed root"):
        relocation.verify_inventory(external_inventory, pipeline)
    with pytest.raises(relocation.RelocationError, match="outside its allowed root"):
        relocation.relocate_pipeline(
            pipeline,
            synthetic_pipeline["old"],
            synthetic_pipeline["new"],
            external_inventory,
            execute=False,
            confirmation="",
        )

    external_alias = outside / "source_inventory_alias.json"
    os.link(synthetic_pipeline["source_inventory"], external_alias)
    with pytest.raises(relocation.RelocationError, match="single-link regular file"):
        relocation.verify_inventory(synthetic_pipeline["source_inventory"], pipeline)

    synthetic_pipeline["local"].chmod(0o755)
    try:
        with pytest.raises(relocation.RelocationError, match="owner-only permissions"):
            relocation.write_inventory(
                pipeline,
                synthetic_pipeline["local"] / "private" / "mode_guard.json",
            )
    finally:
        synthetic_pipeline["local"].chmod(0o700)


def test_inventory_rejects_a_directory_changed_after_scandir(
    synthetic_pipeline: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    real_scandir = os.scandir
    mutated = False

    class MutatingScandir:
        def __init__(self, descriptor: int) -> None:
            self._iterator = real_scandir(descriptor)

        def __enter__(self):
            return self._iterator.__enter__()

        def __exit__(self, exc_type, exc, traceback):
            nonlocal mutated
            result = self._iterator.__exit__(exc_type, exc, traceback)
            if not mutated:
                mutated = True
                (pipeline / "added-after-scandir.txt").write_bytes(b"race")
            return result

    monkeypatch.setattr(relocation.os, "scandir", MutatingScandir)
    with pytest.raises(relocation.RelocationError, match="changed during inventory"):
        relocation.build_inventory(pipeline)


def test_dry_run_is_strictly_read_only_and_plans_exact_fields(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    for path in sorted(
        (pipeline, *pipeline.rglob("*")),
        key=lambda item: (item.is_dir(), len(item.parts)),
    ):
        path_stat = path.stat()
        os.utime(
            path,
            ns=(
                path_stat.st_atime_ns,
                (path_stat.st_mtime_ns // 1_000_000_000) * 1_000_000_000,
            ),
        )
    before = _snapshot(synthetic_pipeline["local"])
    state_parent = synthetic_pipeline["local"] / "mango_calls_relocation_state"

    report = _relocate(synthetic_pipeline, execute=False)

    assert report["status"] == "dry_run"
    assert report["capture"] == {
        "rows": 4,
        "changed_rows": 4,
        "changed_paths": 7,
        "omitted_historical_ready_assets": 0,
        "incomplete_tail_preserved": True,
        "tail_size_bytes": len(synthetic_pipeline["torn_tail"]),
        "tail_sha256": hashlib.sha256(synthetic_pipeline["torn_tail"]).hexdigest(),
    }
    assert report["working"]["updates"] == 2
    assert report["ready"]["updates"] == 2
    assert _snapshot(synthetic_pipeline["local"]) == before
    assert not state_parent.exists()
    assert Path(str(synthetic_pipeline["pipeline"] / relocation.WORKING_DB_REL) + "-wal").read_bytes() == b""
    assert len(Path(str(synthetic_pipeline["pipeline"] / relocation.WORKING_DB_REL) + "-shm").read_bytes()) == 32768


def test_selective_transfer_omits_only_historical_ready_audio(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    old = synthetic_pipeline["old"]
    capture_path = pipeline / relocation.CAPTURE_REL
    ready_row = _manifest_row(
        "call-a",
        "downloaded",
        old_root=old,
        local_audio_path=str(old / "capture/recordings/a.mp3"),
    )
    pending_row = _manifest_row(
        "call-b",
        "downloaded",
        old_root=old,
        local_audio_path=str(old / "capture/recordings/b.mp3"),
    )
    capture_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in (ready_row, pending_row)
        ),
        encoding="utf-8",
    )
    (pipeline / "capture/recordings/a.mp3").unlink()
    (pipeline / "working/audio/a.mp3").unlink()
    _source_inventory(pipeline, synthetic_pipeline["source_inventory"], old)

    dry_run = _relocate(synthetic_pipeline, execute=False)

    assert dry_run["status"] == "dry_run"
    assert dry_run["capture"]["omitted_historical_ready_assets"] == 1
    assert dry_run["working"]["omitted_historical_ready_assets"] == 1
    assert dry_run["ready"]["omitted_historical_ready_assets"] == 1
    assert _relocate(synthetic_pipeline, execute=True)["status"] == "relocated"
    assert not (pipeline / "capture/recordings/a.mp3").exists()
    assert not (pipeline / "working/audio/a.mp3").exists()
    assert (pipeline / "capture/recordings/b.mp3").is_file()
    assert (pipeline / "working/audio/b.mp3").is_file()
    pipeline.rename(synthetic_pipeline["new"])
    assert relocation.relocate_pipeline(
        synthetic_pipeline["new"],
        old,
        synthetic_pipeline["new"],
        synthetic_pipeline["source_inventory"],
        execute=True,
        confirmation=relocation.CONFIRM_VALUE,
    )["status"] == "already_relocated"


def test_selective_transfer_still_requires_unfinished_audio(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    old = synthetic_pipeline["old"]
    (pipeline / "working/audio/b.mp3").unlink()
    _source_inventory(pipeline, synthetic_pipeline["source_inventory"], old)

    with pytest.raises(relocation.RelocationError, match="target asset is missing or empty"):
        _relocate(synthetic_pipeline, execute=False)


@pytest.mark.parametrize("artifact", ["capture", "working", "ready", "ready_manifest"])
def test_first_plan_rejects_any_path_already_below_new_root(
    synthetic_pipeline: Mapping[str, Any],
    artifact: str,
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    old = synthetic_pipeline["old"]
    new = synthetic_pipeline["new"]
    ready_manifest_path = pipeline / relocation.READY_MANIFEST_REL

    if artifact == "capture":
        capture = pipeline / relocation.CAPTURE_REL
        capture.write_bytes(
            capture.read_bytes().replace(str(old).encode(), str(new).encode(), 1)
        )
    elif artifact == "working":
        working_db = pipeline / relocation.WORKING_DB_REL
        for suffix in ("-wal", "-shm"):
            Path(str(working_db) + suffix).unlink(missing_ok=True)
        with closing(sqlite3.connect(working_db)) as connection:
            connection.execute(
                "UPDATE call_records SET source_file=? WHERE id=1",
                (str(new / "working/audio/a.mp3"),),
            )
            connection.commit()
            assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
        Path(str(working_db) + "-wal").write_bytes(b"")
        Path(str(working_db) + "-shm").write_bytes(b"\0" * 32768)
        ready_manifest = json.loads(ready_manifest_path.read_text(encoding="utf-8"))
        ready_manifest["source_storage"] = _storage(working_db)
        _private_json(ready_manifest_path, ready_manifest)
        ready_manifest_path.chmod(0o644)
    elif artifact == "ready":
        ready_db = pipeline / relocation.READY_DB_REL
        with closing(sqlite3.connect(ready_db)) as connection:
            connection.execute(
                "UPDATE call_records SET source_file=? WHERE id=1",
                (str(new / "working/audio/a.mp3"),),
            )
            connection.commit()
        ready_manifest = json.loads(ready_manifest_path.read_text(encoding="utf-8"))
        ready_manifest.update(
            {
                "sha256": _sha(ready_db),
                "size_bytes": ready_db.stat().st_size,
                "ready_mtime_ns": ready_db.stat().st_mtime_ns,
            }
        )
        _private_json(ready_manifest_path, ready_manifest)
        ready_manifest_path.chmod(0o644)
    else:
        ready_manifest = json.loads(ready_manifest_path.read_text(encoding="utf-8"))
        ready_manifest["ready_db"] = str(new / relocation.READY_DB_REL)
        _private_json(ready_manifest_path, ready_manifest)
        ready_manifest_path.chmod(0o644)

    _source_inventory(pipeline, synthetic_pipeline["source_inventory"], old)
    before = _snapshot(synthetic_pipeline["local"])
    with pytest.raises(relocation.RelocationError, match="before the first durable plan"):
        _relocate(synthetic_pipeline, execute=False)
    assert _snapshot(synthetic_pipeline["local"]) == before
    assert not (synthetic_pipeline["local"] / "mango_calls_relocation_state").exists()


def test_read_only_source_preflight_is_allowed_but_execute_outside_local_is_not(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    source_before = _snapshot(synthetic_pipeline["source"])
    local_before = _snapshot(synthetic_pipeline["local"])

    report = relocation.relocate_pipeline(
        synthetic_pipeline["source"],
        synthetic_pipeline["source"],
        synthetic_pipeline["new"],
        synthetic_pipeline["source_inventory"],
        execute=False,
        confirmation="",
    )
    assert report["status"] == "dry_run"
    assert _snapshot(synthetic_pipeline["source"]) == source_before
    assert _snapshot(synthetic_pipeline["local"]) == local_before

    with pytest.raises(relocation.RelocationError, match="outside its allowed root"):
        relocation.relocate_pipeline(
            synthetic_pipeline["source"],
            synthetic_pipeline["source"],
            synthetic_pipeline["new"],
            synthetic_pipeline["source_inventory"],
            execute=True,
            confirmation=relocation.CONFIRM_VALUE,
        )
    assert _snapshot(synthetic_pipeline["source"]) == source_before


def test_execute_rewrites_only_runtime_paths_reseals_and_second_run_is_exact_noop(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    source_inventory_before = synthetic_pipeline["source_inventory"].read_bytes()
    old_report = (pipeline / "reports/control.json").read_bytes()
    business_before = {
        relative: _sqlite_business_snapshot(pipeline / relative)
        for relative in (relocation.WORKING_DB_REL, relocation.READY_DB_REL)
    }
    publication_before = _publication_hash(
        pipeline / relocation.READY_DB_REL,
        pipeline / relocation.WORKING_DB_REL,
    )

    report = _relocate(synthetic_pipeline, execute=True)
    assert report["status"] == "relocated"
    assert synthetic_pipeline["source_inventory"].read_bytes() == source_inventory_before
    assert _snapshot(synthetic_pipeline["source"]) == synthetic_pipeline["source_snapshot"]
    assert (pipeline / "reports/control.json").read_bytes() == old_report
    assert {
        relative: _sqlite_business_snapshot(pipeline / relative)
        for relative in (relocation.WORKING_DB_REL, relocation.READY_DB_REL)
    } == business_before
    assert _publication_hash(
        pipeline / relocation.READY_DB_REL,
        pipeline / relocation.WORKING_DB_REL,
    ) == publication_before

    manifest_raw = (pipeline / relocation.CAPTURE_REL).read_bytes()
    assert manifest_raw.endswith(synthetic_pipeline["torn_tail"])
    rows, tail = relocation.split_capture_rows(manifest_raw)
    assert tail == synthetic_pipeline["torn_tail"]
    payloads = [payload for _raw, _newline, payload in rows]
    assert all(str(synthetic_pipeline["new"]) in json.dumps(row) for row in payloads)
    assert all(row["manager_ref"] == f"negative control {synthetic_pipeline['old']}/must-not-change" for row in payloads)
    assert all(row["note_path"] == f"{synthetic_pipeline['old']}/unknown-field-must-not-change" for row in payloads)

    for db_relative in (relocation.WORKING_DB_REL, relocation.READY_DB_REL):
        with closing(sqlite3.connect(
            f"file:{pipeline / db_relative}?mode=ro&immutable=1",
            uri=True,
        )) as connection:
            sources = [str(row[0]) for row in connection.execute("SELECT source_file FROM call_records ORDER BY id")]
            assert sources == [
                str(synthetic_pipeline["new"] / "working/audio/a.mp3"),
                str(synthetic_pipeline["new"] / "working/audio/b.mp3"),
            ]
            assert connection.execute("SELECT last_error FROM call_records WHERE id=1").fetchone()[0] == f"control {synthetic_pipeline['old']}/do-not-change"
            assert connection.execute("SELECT source_file FROM audit_notes").fetchone()[0] == f"{synthetic_pipeline['old']}/unrelated/table/value.mp3"
            assert connection.execute("PRAGMA quick_check").fetchone()[0] == "ok"
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

    ready_manifest = json.loads((pipeline / relocation.READY_MANIFEST_REL).read_text(encoding="utf-8"))
    ready_db = pipeline / relocation.READY_DB_REL
    assert ready_manifest["ready_db"] == str(synthetic_pipeline["new"] / relocation.READY_DB_REL)
    assert ready_manifest["sha256"] == _sha(ready_db)
    assert ready_manifest["size_bytes"] == ready_db.stat().st_size
    assert ready_manifest["ready_mtime_ns"] == ready_db.stat().st_mtime_ns
    assert ready_manifest["source_storage"] == _storage(pipeline / relocation.WORKING_DB_REL)
    assert ready_manifest["counts"] == {"total": 2, "pending": 2}
    assert ready_manifest["published_at"] == "2026-08-08T00:00:00+00:00"
    assert ready_manifest["integrity_check"] == "ok"

    for path in (pipeline, *pipeline.rglob("*")):
        path_stat = os.lstat(path)
        if stat.S_ISDIR(path_stat.st_mode):
            assert stat.S_IMODE(path_stat.st_mode) == 0o700
        elif stat.S_ISREG(path_stat.st_mode):
            assert stat.S_IMODE(path_stat.st_mode) == 0o600

    before_repeat = _snapshot(synthetic_pipeline["local"])
    repeated = _relocate(synthetic_pipeline, execute=True)
    assert repeated["status"] == "already_relocated"
    assert repeated["changes"] == 0
    assert _snapshot(synthetic_pipeline["local"]) == before_repeat
    assert calls.call_db_has_open_work(pipeline / relocation.WORKING_DB_REL) is True


def test_sqlite_trigger_cannot_change_business_fields_during_relocation(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    working_db = pipeline / relocation.WORKING_DB_REL
    with closing(sqlite3.connect(working_db)) as connection:
        connection.executescript(
            """
            CREATE TRIGGER corrupt_business_fields
            AFTER UPDATE OF source_file ON call_records
            BEGIN
                UPDATE call_records
                SET transcription_status='done', transcribe_attempts=99,
                    analysis_json='{"history_summary":"CORRUPTED"}'
                WHERE id=NEW.id;
            END;
            """
        )
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    for suffix in ("-wal", "-shm"):
        Path(str(working_db) + suffix).unlink(missing_ok=True)
    ready_manifest_path = pipeline / relocation.READY_MANIFEST_REL
    ready_manifest = json.loads(ready_manifest_path.read_text(encoding="utf-8"))
    ready_manifest["source_storage"] = _storage(working_db)
    _private_json(ready_manifest_path, ready_manifest)
    ready_manifest_path.chmod(0o644)
    _source_inventory(pipeline, synthetic_pipeline["source_inventory"], synthetic_pipeline["old"])
    business_before = _sqlite_business_snapshot(working_db)

    with pytest.raises(relocation.RelocationError, match="business fields changed"):
        _relocate(synthetic_pipeline, execute=True)

    assert _sqlite_business_snapshot(working_db) == business_before


def test_sqlite_cross_row_trigger_cannot_escape_full_path_vector(
    tmp_path: Path,
) -> None:
    cwd_before = (os.stat(".").st_dev, os.stat(".").st_ino)
    source = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(source)) as connection:
        connection.executescript(
            """
            CREATE TABLE call_records (
                id INTEGER PRIMARY KEY,
                source_file TEXT NOT NULL UNIQUE
            );
            INSERT INTO call_records VALUES (1, 'old-a');
            INSERT INTO call_records VALUES (2, 'new-b');
            CREATE TRIGGER corrupt_other_source_path
            AFTER UPDATE OF source_file ON call_records
            WHEN NEW.id=1
            BEGIN
                UPDATE call_records SET source_file='third-root-evil' WHERE id=2;
            END;
            """
        )
    source_before = source.read_bytes()
    staging_path = tmp_path / "staging"
    staging_path.mkdir(mode=0o700)
    staging = relocation._pin_directory(
        staging_path,
        label="synthetic staging",
        private=True,
    )
    try:
        with pytest.raises(relocation.RelocationError, match="full-table exact readback"):
            relocation._stage_sqlite(
                source,
                staging,
                "staged.sqlite",
                [(1, "old-a", "new-a")],
                [(1, "new-a"), (2, "new-b")],
                journal_mode="delete",
            )
    finally:
        staging.close()
    assert source.read_bytes() == source_before
    assert list(staging_path.iterdir()) == []
    assert (os.stat(".").st_dev, os.stat(".").st_ino) == cwd_before


def test_sqlite_connections_force_memory_temp_storage() -> None:
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.execute("PRAGMA temp_store=FILE")
        relocation.configure_sqlite_memory_temp(connection)
        assert connection.execute("PRAGMA temp_store").fetchone()[0] == 2


def test_sqlite_staging_ignores_cloud_directed_tmpdir(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cwd_before = (os.stat(".").st_dev, os.stat(".").st_ino)
    cloud_tmp = tmp_path / "Yandex.Disk.localized" / "tmp"
    cloud_tmp.mkdir(parents=True, mode=0o700)
    cloud_before = _snapshot(cloud_tmp)
    monkeypatch.setattr(relocation.tempfile, "tempdir", str(cloud_tmp))
    monkeypatch.setenv("SQLITE_TMPDIR", str(cloud_tmp))
    monkeypatch.setenv("TMPDIR", str(cloud_tmp))

    assert _relocate(synthetic_pipeline, execute=True)["status"] == "relocated"
    assert _snapshot(cloud_tmp) == cloud_before
    assert (os.stat(".").st_dev, os.stat(".").st_ino) == cwd_before


def test_sqlite_staging_repairs_main_mode_before_reopen_under_strict_umask(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(source)) as connection:
        connection.executescript(
            """
            CREATE TABLE call_records (
                id INTEGER PRIMARY KEY,
                source_file TEXT NOT NULL UNIQUE
            );
            INSERT INTO call_records VALUES (1, 'old-a');
            """
        )
    staging_path = tmp_path / "staging"
    staging_path.mkdir(mode=0o700)
    staging = relocation._pin_directory(
        staging_path,
        label="synthetic staging",
        private=True,
    )
    previous_umask = os.umask(0o777)
    try:
        relocation._stage_sqlite(
            source,
            staging,
            "staged.sqlite",
            [(1, "old-a", "new-a")],
            [(1, "new-a")],
            journal_mode="delete",
        )
    finally:
        os.umask(previous_umask)
        staging.close()
    assert stat.S_IMODE((staging_path / "staged.sqlite").stat().st_mode) == 0o600


@pytest.mark.parametrize("checkpoint_name", [f"after_replace:{relative}" for relative in relocation.ARTIFACT_ORDER])
def test_crash_after_each_replace_resumes_to_exact_result(
    synthetic_pipeline: Mapping[str, Any],
    checkpoint_name: str,
) -> None:
    fired = False

    def crash(name: str) -> None:
        nonlocal fired
        if name == checkpoint_name and not fired:
            fired = True
            raise RuntimeError(f"synthetic crash at {name}")

    with pytest.raises(RuntimeError, match="synthetic crash"):
        _relocate(synthetic_pipeline, execute=True, checkpoint=crash)
    assert fired

    resumed = _relocate(synthetic_pipeline, execute=True)
    assert resumed["status"] == "relocated"
    snapshot = _snapshot(synthetic_pipeline["local"])
    assert _relocate(synthetic_pipeline, execute=True)["status"] == "already_relocated"
    assert _snapshot(synthetic_pipeline["local"]) == snapshot


def test_preplan_sqlite_debris_is_rebuilt_before_durable_plan(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    state_dir = (
        synthetic_pipeline["local"]
        / "mango_calls_relocation_state"
        / relocation.state_key(synthetic_pipeline["old"], synthetic_pipeline["new"])
    )
    staging = state_dir / "staging"
    staging.mkdir(parents=True, mode=0o700)
    state_dir.parent.chmod(0o700)
    state_dir.chmod(0o700)
    staging.chmod(0o700)
    for name in ("working.sqlite", "working.sqlite-wal", "working.sqlite-shm"):
        path = staging / name
        path.write_bytes(b"synthetic pre-plan crash debris")
        path.chmod(0o600)
    assert not (state_dir / "plan.json").exists()

    assert _relocate(synthetic_pipeline, execute=True)["status"] == "relocated"
    assert (state_dir / "complete.json").is_file()
    assert list(staging.iterdir()) == []


def test_resume_rejects_a_staging_symlink_without_consuming_external_artifacts(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
) -> None:
    def crash_after_plan(name: str) -> None:
        if name == "after_plan":
            raise RuntimeError("synthetic crash after plan")

    with pytest.raises(RuntimeError, match="synthetic crash after plan"):
        _relocate(synthetic_pipeline, execute=True, checkpoint=crash_after_plan)

    state_dir = (
        synthetic_pipeline["local"]
        / "mango_calls_relocation_state"
        / relocation.state_key(synthetic_pipeline["old"], synthetic_pipeline["new"])
    )
    staging = state_dir / "staging"
    outside = tmp_path / "resume-staging-outside"
    staging.rename(outside)
    staging.symlink_to(outside, target_is_directory=True)
    outside_before = _snapshot(outside)

    with pytest.raises(relocation.RelocationError, match="staging directory is unsafe"):
        _relocate(synthetic_pipeline, execute=True)

    assert _snapshot(outside) == outside_before


def test_torn_tail_survives_relocation_and_real_process_a_recovers_it(
    synthetic_pipeline: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _relocate(synthetic_pipeline, execute=True)
    synthetic_pipeline["pipeline"].rename(synthetic_pipeline["new"])
    pipeline = synthetic_pipeline["new"]
    manifest = pipeline / relocation.CAPTURE_REL
    raw_before = manifest.read_bytes()
    rows_before, tail_before = relocation.split_capture_rows(raw_before)
    assert tail_before == synthetic_pipeline["torn_tail"]

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def poll_call_history(self, **_kwargs: Any) -> list[Mapping[str, Any]]:
            return []

    monkeypatch.setenv("MANGO_OFFICE_API_KEY", "synthetic-key")
    monkeypatch.setenv("MANGO_OFFICE_API_SALT", "synthetic-salt")
    monkeypatch.setattr(calls, "MangoOfficeClient", FakeClient)
    monkeypatch.setattr(calls, "MangoRecordingDownloader", FakeClient)
    monkeypatch.setattr(calls, "disk_preflight", lambda _config: {"ok": True})
    monkeypatch.setattr(
        calls,
        "environment_preflight",
        lambda _config, **_kwargs: {"ok": True, "codex_network_ok": False},
    )
    config = calls.CallsTwoProcessesConfig(
        pipeline_root=pipeline,
        timeline_db=synthetic_pipeline["local"] / "timeline" / "timeline.sqlite",
        timeline_allowed_root=synthetic_pipeline["local"] / "timeline",
        python_executable=Path(os.sys.executable),
        codex_binary=Path("/usr/bin/false"),
        codex_home_root=synthetic_pipeline["local"] / "codex-home",
        min_free_gib=1,
    )
    result = calls.run_process_a(
        config,
        since="2026-08-08T11:00:00+00:00",
        until="2026-08-08T12:00:00+00:00",
        skip_workers=True,
    )
    assert result["status"] == "partial", result
    assert result["stop_reason"] == "capture_manifest_tail_incomplete"
    assert not manifest.read_bytes().endswith(tail_before)
    ledger = load_capture_recovery(manifest.with_name(f".{manifest.name}.recovery.json"))
    assert ledger["status"] == "resolved"
    assert ledger["acknowledged_incident_sha256"]
    recovered_store = CaptureManifestStore(manifest)
    assert len(recovered_store.read_entries()) >= len(rows_before)
    assert recovered_store.incomplete_trailing_records == 0

    with closing(sqlite3.connect(f"file:{config.working_db}?mode=ro&immutable=1", uri=True)) as connection:
        pending = connection.execute(
            """
            SELECT id, source_call_id, source_file, transcribe_attempts,
                   resolve_attempts, analyze_attempts
            FROM call_records
            WHERE transcription_status = 'pending'
            ORDER BY id LIMIT 1
            """
        ).fetchone()
        assert pending is not None
        pending_id, pending_call_id = int(pending[0]), str(pending[1])
        pending_source = Path(str(pending[2]))
        attempts_before = tuple(int(value) for value in pending[3:])
    assert pending_source.is_file()

    commands: list[tuple[list[str], Mapping[str, str]]] = []

    def fake_command(command, environment, _cwd):
        commands.append((list(command), dict(environment)))
        if "--stages" in command:
            stage = command[command.index("--stages") + 1]
            with closing(sqlite3.connect(config.working_db)) as connection:
                if stage == "transcribe":
                    payload = {
                        "mode": "mono_or_fallback",
                        "primary_provider": "mlx",
                        "secondary_provider": "",
                        "full": {"variant_a": "синтетический Whisper", "variant_b": ""},
                    }
                    cursor = connection.execute(
                        """
                        UPDATE call_records
                        SET transcription_status='done', transcript_text=?,
                            transcript_variants_json=?, transcribe_attempts=transcribe_attempts+1,
                            pipeline_stage='backfill-second-asr'
                        WHERE id=? AND source_call_id=? AND transcription_status='pending'
                        """,
                        ("синтетический Whisper", json.dumps(payload, ensure_ascii=False), pending_id, pending_call_id),
                    )
                elif stage == "backfill-second-asr":
                    payload = {
                        "mode": "mono_or_fallback",
                        "primary_provider": "mlx",
                        "secondary_provider": "gigaam",
                        "full": {
                            "variant_a": "синтетический Whisper",
                            "variant_b": "синтетический GigaAM",
                        },
                    }
                    cursor = connection.execute(
                        """
                        UPDATE call_records
                        SET transcript_variants_json=?, pipeline_stage='resolve'
                        WHERE id=? AND source_call_id=? AND transcription_status='done'
                        """,
                        (json.dumps(payload, ensure_ascii=False), pending_id, pending_call_id),
                    )
                elif stage == "resolve":
                    cursor = connection.execute(
                        """
                        UPDATE call_records
                        SET resolve_status='done', resolve_json=?,
                            resolve_attempts=resolve_attempts+1, pipeline_stage='analyze'
                        WHERE id=? AND source_call_id=? AND transcription_status='done'
                        """,
                        (json.dumps({"decision": "resolved"}), pending_id, pending_call_id),
                    )
                else:
                    assert stage == "analyze"
                    cursor = connection.execute(
                        """
                        UPDATE call_records
                        SET analysis_status='done', analysis_json=?,
                            analyze_attempts=analyze_attempts+1, pipeline_stage=NULL
                        WHERE id=? AND source_call_id=? AND resolve_status='done'
                        """,
                        (
                            json.dumps(
                                {
                                    "call_type": "synthetic",
                                    "history_summary": "Синтетическая запись продолжена после переноса.",
                                },
                                ensure_ascii=False,
                            ),
                            pending_id,
                            pending_call_id,
                        ),
                    )
                assert cursor.rowcount == 1
                connection.commit()
        return {"rc": 0, "command": calls.compact_command_name(command), "log_path": "synthetic"}

    monkeypatch.setattr(
        calls,
        "environment_preflight",
        lambda _config, **_kwargs: {"ok": True, "codex_network_ok": True},
    )
    continued = calls.run_process_a(
        config,
        skip_capture=True,
        command_runner=fake_command,
    )
    stages = [
        command[command.index("--stages") + 1]
        for command, _environment in commands
        if "--stages" in command
    ]
    assert stages == ["transcribe", "backfill-second-asr", "resolve", "analyze"], continued
    assert continued["status"] == "partial", continued
    assert continued["stop_reason"] == "stage10_consistency_not_proven"
    with closing(sqlite3.connect(f"file:{config.working_db}?mode=ro", uri=True)) as connection:
        completed = connection.execute(
            """
            SELECT source_call_id, transcription_status, resolve_status, analysis_status,
                   transcribe_attempts, resolve_attempts, analyze_attempts
            FROM call_records WHERE id=?
            """,
            (pending_id,),
        ).fetchone()
        assert completed == (
            pending_call_id,
            "done",
            "done",
            "done",
            attempts_before[0] + 1,
            attempts_before[1] + 1,
            attempts_before[2] + 1,
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM call_records WHERE source_call_id=?",
            (pending_call_id,),
        ).fetchone()[0] == 1
    assert calls.call_db_has_open_work(config.working_db) is False


def test_unresolved_capture_recovery_blocks_before_any_write(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    manifest = pipeline / relocation.CAPTURE_REL
    raw = manifest.read_bytes()
    prefix = raw[: -len(synthetic_pipeline["torn_tail"])]
    record_capture_recovery(
        manifest.with_name(f".{manifest.name}.recovery.json"),
        synthetic_pipeline["torn_tail"],
        prefix,
    )
    _source_inventory(pipeline, synthetic_pipeline["source_inventory"], synthetic_pipeline["old"])
    before = _snapshot(synthetic_pipeline["local"])
    with pytest.raises(relocation.RelocationError, match="unresolved capture recovery"):
        _relocate(synthetic_pipeline, execute=False)
    assert _snapshot(synthetic_pipeline["local"]) == before


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_symlink_and_special_tree_entries_fail_before_write(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
    kind: str,
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    target = pipeline / "capture" / f"unsafe-{kind}"
    if kind == "symlink":
        outside = tmp_path / "outside"
        outside.write_text("outside", encoding="utf-8")
        target.symlink_to(outside)
    else:
        os.mkfifo(target)
    before_source = synthetic_pipeline["source_inventory"].read_bytes()
    with pytest.raises(relocation.RelocationError, match="unsupported entries"):
        relocation.build_inventory(pipeline)
    assert synthetic_pipeline["source_inventory"].read_bytes() == before_source


@pytest.mark.parametrize("kind", ["symlinked_working", "external_wal_hardlink"])
def test_precheckpoint_inventory_rejects_sqlite_path_escape_without_mutation(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
    kind: str,
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    working = pipeline / "working"
    database = working / "mango_calls_pipeline.sqlite"

    if kind == "symlinked_working":
        outside = tmp_path / "outside-working"
        working.rename(outside)
        working.symlink_to(outside, target_is_directory=True)
        protected = outside / database.name
        expected = protected.read_bytes()
        error = "unsupported entries"
    else:
        wal = Path(f"{database}-wal")
        wal.write_bytes(b"synthetic-wal-must-not-change")
        protected = tmp_path / "outside-wal-hardlink"
        os.link(wal, protected)
        expected = protected.read_bytes()
        error = "hard links with aliases outside the tree"

    with pytest.raises(relocation.RelocationError, match=error):
        relocation.build_inventory(pipeline)

    assert protected.read_bytes() == expected


def test_third_root_and_prefix_trap_are_rejected_without_mutation(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    manifest = pipeline / relocation.CAPTURE_REL
    raw = manifest.read_bytes()
    evil_root = f"{synthetic_pipeline['old']}_evil"
    manifest.write_bytes(raw.replace(str(synthetic_pipeline["old"]).encode(), evil_root.encode(), 1))
    source_inventory = _source_inventory(pipeline, synthetic_pipeline["source_inventory"], synthetic_pipeline["old"])
    assert source_inventory == synthetic_pipeline["source_inventory"]
    before = _snapshot(synthetic_pipeline["local"])
    with pytest.raises(relocation.RelocationError, match="before the first durable plan"):
        _relocate(synthetic_pipeline, execute=False)
    assert _snapshot(synthetic_pipeline["local"]) == before


def test_execute_token_and_busy_process_lock_fail_without_pipeline_writes(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    before = _snapshot(pipeline)
    with pytest.raises(relocation.RelocationError, match="execute requires"):
        relocation.relocate_pipeline(
            pipeline,
            synthetic_pipeline["old"],
            synthetic_pipeline["new"],
            synthetic_pipeline["source_inventory"],
            execute=True,
            confirmation="wrong",
        )
    assert _snapshot(pipeline) == before

    lock = pipeline / "locks/process_a.lock"
    lock.write_text("{}", encoding="utf-8")
    lock.chmod(0o600)
    _source_inventory(pipeline, synthetic_pipeline["source_inventory"], synthetic_pipeline["old"])
    with lock.open("rb") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked_before = _snapshot(pipeline)
        with pytest.raises(relocation.RelocationError, match="process lock is busy"):
            _relocate(synthetic_pipeline, execute=False)
        assert _snapshot(pipeline) == locked_before


@pytest.mark.parametrize("outside_exists", [True, False])
def test_relocation_lock_symlink_cannot_escape_owner_only_state(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
    outside_exists: bool,
) -> None:
    state_dir = (
        synthetic_pipeline["local"]
        / "mango_calls_relocation_state"
        / relocation.state_key(synthetic_pipeline["old"], synthetic_pipeline["new"])
    )
    state_dir.mkdir(parents=True, mode=0o700)
    state_dir.parent.chmod(0o700)
    state_dir.chmod(0o700)
    outside = tmp_path / "outside-lock"
    if outside_exists:
        outside.write_bytes(b"must stay unchanged")
        outside.chmod(0o644)
    (state_dir / "relocation.lock").symlink_to(outside)

    with pytest.raises(relocation.RelocationError, match="relocation lock is unsafe"):
        _relocate(synthetic_pipeline, execute=True)

    if outside_exists:
        assert outside.read_bytes() == b"must stay unchanged"
        assert stat.S_IMODE(outside.stat().st_mode) == 0o644
    else:
        assert not outside.exists()


def test_state_directory_swap_after_open_cannot_redirect_writes(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
) -> None:
    pipeline_before = _snapshot(synthetic_pipeline["pipeline"])
    state_dir = (
        synthetic_pipeline["local"]
        / "mango_calls_relocation_state"
        / relocation.state_key(synthetic_pipeline["old"], synthetic_pipeline["new"])
    )
    displaced = state_dir.with_name(f"{state_dir.name}.displaced")
    outside = tmp_path / "outside-state"
    outside.mkdir(mode=0o700)
    sentinel = outside / "sentinel.txt"
    sentinel.write_bytes(b"must stay unchanged")
    sentinel.chmod(0o600)
    outside_before = _snapshot(outside)

    def swap(name: str) -> None:
        if name != "after_state_open":
            return
        state_dir.rename(displaced)
        state_dir.symlink_to(outside, target_is_directory=True)

    with pytest.raises(relocation.RelocationError, match="no longer bound"):
        _relocate(synthetic_pipeline, execute=True, checkpoint=swap)

    assert _snapshot(outside) == outside_before
    assert _snapshot(synthetic_pipeline["pipeline"]) == pipeline_before
    assert not (outside / "relocation.lock").exists()
    assert not (outside / "plan.json").exists()


def test_hardlink_swap_before_permission_repair_never_chmods_external_inode(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    victim = pipeline / "reports/control.json"
    outside = tmp_path / "outside-hardlink"
    outside.write_bytes(victim.read_bytes())
    outside.chmod(0o644)
    outside_bytes = outside.read_bytes()
    outside_mode = stat.S_IMODE(outside.stat().st_mode)

    def swap(name: str) -> None:
        if name != "before_permissions":
            return
        victim.unlink()
        os.link(outside, victim)

    with pytest.raises(relocation.RelocationError, match="hard links"):
        _relocate(synthetic_pipeline, execute=True, checkpoint=swap)

    assert outside.read_bytes() == outside_bytes
    assert stat.S_IMODE(outside.stat().st_mode) == outside_mode
    state_dir = (
        synthetic_pipeline["local"]
        / "mango_calls_relocation_state"
        / relocation.state_key(synthetic_pipeline["old"], synthetic_pipeline["new"])
    )
    assert not (state_dir / "complete.json").exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_new_tree_entry_before_permission_repair_cannot_enter_completion_manifest(
    synthetic_pipeline: Mapping[str, Any],
    kind: str,
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    injected = pipeline / "reports" / f"injected-{kind}"

    def add_entry(name: str) -> None:
        if name != "before_permissions":
            return
        if kind == "file":
            injected.write_bytes(b"not part of the verified source")
            injected.chmod(0o644)
        else:
            injected.mkdir(mode=0o755)

    with pytest.raises(relocation.RelocationError, match="tree changed across permission repair"):
        _relocate(synthetic_pipeline, execute=True, checkpoint=add_entry)

    state_dir = (
        synthetic_pipeline["local"]
        / "mango_calls_relocation_state"
        / relocation.state_key(synthetic_pipeline["old"], synthetic_pipeline["new"])
    )
    assert injected.exists()
    assert not (state_dir / "complete.json").exists()


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_new_tree_entry_during_artifact_commit_cannot_enter_permission_baseline(
    synthetic_pipeline: Mapping[str, Any],
    kind: str,
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    injected = pipeline / "reports" / f"injected-after-replace-{kind}"

    def add_entry(name: str) -> None:
        if name != f"after_replace:{relocation.CAPTURE_REL}":
            return
        if kind == "file":
            injected.write_bytes(b"not part of durable before inventory")
            injected.chmod(0o600)
        else:
            injected.mkdir(mode=0o700)

    with pytest.raises(relocation.RelocationError, match="gained or lost entries"):
        _relocate(synthetic_pipeline, execute=True, checkpoint=add_entry)

    state_dir = (
        synthetic_pipeline["local"]
        / "mango_calls_relocation_state"
        / relocation.state_key(synthetic_pipeline["old"], synthetic_pipeline["new"])
    )
    assert injected.exists()
    assert not (state_dir / "complete.json").exists()


def test_permission_repair_preserves_supported_internal_hardlinks(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    first = pipeline / "reports/internal-hardlink-a.txt"
    second = pipeline / "reports/internal-hardlink-b.txt"
    first.write_bytes(b"internal alias")
    first.chmod(0o644)
    os.link(first, second)
    _source_inventory(
        pipeline,
        synthetic_pipeline["source_inventory"],
        synthetic_pipeline["old"],
    )

    assert _relocate(synthetic_pipeline, execute=True)["status"] == "relocated"
    assert first.stat().st_ino == second.stat().st_ino
    assert stat.S_IMODE(first.stat().st_mode) == 0o600
    assert stat.S_IMODE(second.stat().st_mode) == 0o600


def test_staging_symlink_cannot_escape_owner_only_state(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
) -> None:
    state_dir = (
        synthetic_pipeline["local"]
        / "mango_calls_relocation_state"
        / relocation.state_key(synthetic_pipeline["old"], synthetic_pipeline["new"])
    )
    state_dir.mkdir(parents=True, mode=0o700)
    state_dir.parent.chmod(0o700)
    state_dir.chmod(0o700)
    outside = tmp_path / "outside-staging"
    outside.mkdir(mode=0o755)
    sentinel = outside / "capture.jsonl"
    sentinel.write_bytes(b"must stay unchanged")
    sentinel.chmod(0o644)
    outside_before = _snapshot(outside)
    (state_dir / "staging").symlink_to(outside, target_is_directory=True)

    with pytest.raises(relocation.RelocationError, match="staging directory is unsafe"):
        _relocate(synthetic_pipeline, execute=True)

    assert _snapshot(outside) == outside_before


def test_staging_hardlink_sidecar_cannot_modify_an_external_file(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
) -> None:
    state_dir = (
        synthetic_pipeline["local"]
        / "mango_calls_relocation_state"
        / relocation.state_key(synthetic_pipeline["old"], synthetic_pipeline["new"])
    )
    staging = state_dir / "staging"
    staging.mkdir(parents=True, mode=0o700)
    state_dir.parent.chmod(0o700)
    state_dir.chmod(0o700)
    staging.chmod(0o700)
    outside = tmp_path / "outside-sidecar"
    outside.write_bytes(b"must stay unchanged")
    outside.chmod(0o644)
    os.link(outside, staging / "working.sqlite-shm")
    outside_bytes = outside.read_bytes()
    outside_mode = stat.S_IMODE(outside.stat().st_mode)

    with pytest.raises(relocation.RelocationError, match="unsafe planned artifact"):
        _relocate(synthetic_pipeline, execute=True)

    assert outside.read_bytes() == outside_bytes
    assert stat.S_IMODE(outside.stat().st_mode) == outside_mode


def test_sqlite_sidecar_and_missing_required_asset_fail_closed(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    sidecar = Path(str(pipeline / relocation.WORKING_DB_REL) + "-wal")
    sidecar.write_bytes(b"not a real WAL")
    _source_inventory(pipeline, synthetic_pipeline["source_inventory"], synthetic_pipeline["old"])
    before = _snapshot(pipeline)
    with pytest.raises(relocation.RelocationError, match="active sidecar"):
        _relocate(synthetic_pipeline, execute=False)
    assert _snapshot(pipeline) == before

    sidecar.write_bytes(b"")
    (pipeline / "capture/recordings/a.mp3").unlink()
    _source_inventory(pipeline, synthetic_pipeline["source_inventory"], synthetic_pipeline["old"])
    with pytest.raises(relocation.RelocationError, match="target asset is missing"):
        _relocate(synthetic_pipeline, execute=False)


def test_immutable_sqlite_check_accepts_clean_wal_without_creating_sidecars(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    working_db = pipeline / relocation.WORKING_DB_REL
    ready_db = pipeline / relocation.READY_DB_REL
    for suffix in relocation.SQLITE_SIDECAR_SUFFIXES:
        Path(f"{working_db}{suffix}").unlink(missing_ok=True)
        Path(f"{ready_db}{suffix}").unlink(missing_ok=True)
    before = _snapshot(pipeline)

    result = relocation.check_sqlite_files([working_db, ready_db])

    assert result == {
        "status": "sqlite_checks_ok",
        "databases": 2,
        "quick_check": "ok",
        "integrity_check": "ok",
    }
    assert _snapshot(pipeline) == before
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(relocation.RelocationError, match="missing"):
        relocation.check_sqlite_files([missing])
    assert not missing.exists()
    encoded = tmp_path / "encoded ?#%20.sqlite"
    shutil.copy2(ready_db, encoded)
    uri = relocation.immutable_sqlite_uri(encoded)
    assert "%3F" in uri and "%23" in uri and "%2520" in uri
    assert relocation.check_sqlite_files([encoded])["status"] == "sqlite_checks_ok"
    assert not any(Path(f"{encoded}{suffix}").exists() for suffix in relocation.SQLITE_SIDECAR_SUFFIXES)


def test_immutable_sqlite_check_rejects_committed_active_wal_without_mutation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "active.sqlite"
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA journal_mode=WAL").fetchone()[0].casefold() == "wal"
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("CREATE TABLE probe(value TEXT)")
        connection.commit()
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
        connection.execute("INSERT INTO probe VALUES ('committed-in-wal')")
        connection.commit()
        wal = Path(f"{database}-wal")
        assert wal.stat().st_size > 0
        before = {
            path.name: path.read_bytes()
            for path in (database, wal, Path(f"{database}-shm"))
            if path.exists()
        }

        with pytest.raises(relocation.RelocationError, match="active WAL"):
            relocation.check_sqlite_files([database])
        assert relocation.main(["--check-sqlite", str(database)]) == 2
        assert json.loads(capsys.readouterr().err)["status"] == "failed"
        assert {
            path.name: path.read_bytes()
            for path in (database, wal, Path(f"{database}-shm"))
            if path.exists()
        } == before
    finally:
        connection.close()


def test_immutable_sqlite_check_rejects_rollback_journal_without_mutation(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    database = pipeline / relocation.READY_DB_REL
    journal = Path(f"{database}-journal")
    journal.unlink(missing_ok=True)
    journal.write_bytes(b"synthetic pending rollback")
    before = _snapshot(pipeline)

    with pytest.raises(relocation.RelocationError, match="rollback journal"):
        relocation.check_sqlite_files([database])

    assert _snapshot(pipeline) == before


@pytest.mark.parametrize("kind", ["mode", "hardlink"])
@pytest.mark.parametrize("window", ["sqlite_checks", "after_sidecars"])
def test_immutable_sqlite_check_rejects_main_database_metadata_drift(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    window: str,
) -> None:
    database = synthetic_pipeline["pipeline"] / relocation.READY_DB_REL
    before = os.lstat(database)
    before_bytes = database.read_bytes()
    alias = tmp_path / "database-hardlink.sqlite"
    changed_mode = stat.S_IMODE(before.st_mode) ^ stat.S_IRGRP

    def drift(path: Path) -> None:
        assert path == database
        if kind == "mode":
            os.chmod(path, changed_mode)
        else:
            os.link(path, alias)

    def drift_during_checks(path: Path) -> Mapping[str, str]:
        drift(path)
        return {"quick_check": "ok", "integrity_check": "ok"}

    if window == "sqlite_checks":
        monkeypatch.setattr(relocation, "sqlite_checks", drift_during_checks)
    else:
        original_sidecar_snapshot = relocation.sqlite_sidecar_snapshot
        sidecar_calls = 0

        def drift_after_second_sidecar_snapshot(path: Path) -> Mapping[str, tuple[Any, ...]]:
            nonlocal sidecar_calls
            result = original_sidecar_snapshot(path)
            sidecar_calls += 1
            if sidecar_calls == 2:
                drift(path)
            return result

        monkeypatch.setattr(relocation, "sqlite_sidecar_snapshot", drift_after_second_sidecar_snapshot)
    try:
        with pytest.raises(relocation.RelocationError, match="changed during immutable checks"):
            relocation.check_sqlite_files([database])

        after = os.lstat(database)
        assert database.read_bytes() == before_bytes
        assert after.st_size == before.st_size
        assert after.st_mtime_ns == before.st_mtime_ns
        if kind == "mode":
            assert stat.S_IMODE(after.st_mode) == changed_mode
        else:
            assert after.st_nlink == before.st_nlink + 1
    finally:
        alias.unlink(missing_ok=True)
        os.chmod(database, stat.S_IMODE(before.st_mode))


def test_immutable_sqlite_check_rejects_wal_created_during_final_database_hash(
    synthetic_pipeline: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = synthetic_pipeline["pipeline"] / relocation.READY_DB_REL
    wal = Path(f"{database}-wal")
    wal.unlink(missing_ok=True)
    database_before = database.read_bytes()
    original_read = relocation._read_regular_bytes
    database_reads = 0

    def create_wal_after_final_database_read(path: Path, *, label: str) -> bytes:
        nonlocal database_reads
        result = original_read(path, label=label)
        if path == database and label == "checked SQLite":
            database_reads += 1
            if database_reads == 2:
                wal.write_bytes(b"synthetic active WAL")
        return result

    monkeypatch.setattr(relocation, "_read_regular_bytes", create_wal_after_final_database_read)
    try:
        with pytest.raises(relocation.RelocationError, match="active WAL"):
            relocation.check_sqlite_files([database])

        assert database.read_bytes() == database_before
        assert wal.read_bytes() == b"synthetic active WAL"
    finally:
        wal.unlink(missing_ok=True)


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_immutable_sqlite_check_rejects_unsafe_existing_sidecar(
    synthetic_pipeline: Mapping[str, Any],
    tmp_path: Path,
    kind: str,
) -> None:
    database = synthetic_pipeline["pipeline"] / relocation.READY_DB_REL
    sidecar = Path(f"{database}-wal")
    sidecar.unlink(missing_ok=True)
    outside = tmp_path / f"outside-{kind}"
    outside.write_bytes(b"")
    if kind == "symlink":
        sidecar.symlink_to(outside)
        error = "symlink component"
    else:
        os.link(outside, sidecar)
        error = "single-link regular file"
    before = outside.read_bytes()

    with pytest.raises(relocation.RelocationError, match=error):
        relocation.check_sqlite_files([database])

    assert outside.read_bytes() == before


def test_immutable_sqlite_check_cli_is_exclusive_and_reports_success(
    synthetic_pipeline: Mapping[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    databases = [
        pipeline / relocation.WORKING_DB_REL,
        pipeline / relocation.READY_DB_REL,
    ]
    arguments = ["--check-sqlite", *(str(path) for path in databases)]

    assert relocation.main(arguments) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "sqlite_checks_ok",
        "databases": 2,
        "quick_check": "ok",
        "integrity_check": "ok",
    }
    with pytest.raises(SystemExit) as raised:
        relocation.parse_args([*arguments, "--dry-run"])
    assert raised.value.code == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_stale_delete_ready_sidecars_are_verified_preserved_and_not_read(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    pipeline = synthetic_pipeline["pipeline"]
    ready_db = pipeline / relocation.READY_DB_REL
    stale_wal = Path(str(ready_db) + "-wal")
    stale_shm = Path(str(ready_db) + "-shm")
    stale_wal.write_bytes(b"")
    stale_shm.write_bytes(b"S" * 32768)
    _source_inventory(pipeline, synthetic_pipeline["source_inventory"], synthetic_pipeline["old"])
    dry_before = _snapshot(pipeline)

    assert _relocate(synthetic_pipeline, execute=False)["status"] == "dry_run"
    assert _snapshot(pipeline) == dry_before
    assert _relocate(synthetic_pipeline, execute=True)["status"] == "relocated"
    assert stale_wal.read_bytes() == b""
    assert stale_shm.read_bytes() == b"S" * 32768
    assert stat.S_IMODE(stale_wal.stat().st_mode) == 0o600
    assert stat.S_IMODE(stale_shm.stat().st_mode) == 0o600
    assert relocation.sqlite_journal_mode(pipeline / relocation.WORKING_DB_REL) == "wal"
    assert relocation.sqlite_journal_mode(ready_db) == "delete"


def test_completion_marker_follows_atomic_transfer_to_new_root(
    synthetic_pipeline: Mapping[str, Any],
) -> None:
    assert _relocate(synthetic_pipeline, execute=True)["status"] == "relocated"
    transfer = synthetic_pipeline["pipeline"]
    new_root = synthetic_pipeline["new"]
    before_move = _snapshot(transfer)
    transfer.rename(new_root)

    repeated = relocation.relocate_pipeline(
        new_root,
        synthetic_pipeline["old"],
        new_root,
        synthetic_pipeline["source_inventory"],
        execute=True,
        confirmation=relocation.CONFIRM_VALUE,
    )
    assert repeated["status"] == "already_relocated"
    assert _snapshot(new_root) == before_move
