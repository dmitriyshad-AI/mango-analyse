from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from mango_mvp.productization.mail_archive import (
    CANONICAL_MAIL_ARCHIVE_DB,
    CANONICAL_MAIL_ARCHIVE_ROOT,
    CANONICAL_MAIL_ARCHIVE_SCHEMA_VERSION,
    CANONICAL_MAIL_IDENTITY_DB,
    MAIL_ARCHIVE_SCHEMA_VERSION,
    assert_canonical_mail_archive_ready,
    canonical_mail_archive_dbs,
    existing_tallanto_identity_dbs,
)
from scripts import build_customer_timeline_nightly_dv2_sources as builder
from scripts import run_customer_timeline_mail_download as download
from scripts import run_customer_timeline_mail_import as mail_import


def _write_archive(
    path: Path,
    *,
    sha: str,
    schema: str = MAIL_ARCHIVE_SCHEMA_VERSION,
    stamped: bool = True,
    messages: bool = True,
    meta: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as con:
        if messages:
            con.execute(
                "CREATE TABLE messages (sha256 TEXT PRIMARY KEY, message_date_iso TEXT, subject TEXT, "
                "message_kind TEXT, mailbox TEXT, extracted_text_path TEXT, updated_at TEXT, first_ingested_at TEXT)"
            )
            con.execute(
                "INSERT INTO messages VALUES (?, ?, '', 'external', 'INBOX', '', ?, ?)",
                (
                    sha,
                    "2026-07-22T10:00:00+00:00",
                    "2026-07-22T10:00:00+00:00",
                    "2026-07-22T10:00:00+00:00",
                ),
            )
        if meta:
            con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            con.execute("INSERT INTO meta VALUES ('schema_version', ?)", (schema,))
            if stamped:
                con.execute("INSERT INTO meta VALUES ('updated_at', '2026-07-22T10:00:00+00:00')")


def _identity_db(root: Path) -> Path:
    path = root / CANONICAL_MAIL_IDENTITY_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(path).close()
    return path


def test_canonical_resolver_and_builder_cover_all_archive_parts(tmp_path: Path) -> None:
    root = tmp_path / "Mango_Data"
    paths = (
        root / CANONICAL_MAIL_ARCHIVE_DB,
        root / CANONICAL_MAIL_ARCHIVE_ROOT / "incoming/regru_edu/inbox/mail_archive.sqlite",
        root / CANONICAL_MAIL_ARCHIVE_ROOT / "incoming/regru_edu/sent/mail_archive.sqlite",
    )
    _write_archive(paths[0], sha="a" * 64, schema=CANONICAL_MAIL_ARCHIVE_SCHEMA_VERSION)
    _write_archive(paths[1], sha="a" * 64)
    _write_archive(paths[2], sha="b" * 64)

    resolved = canonical_mail_archive_dbs(root)
    report = builder.build_mail_increment(
        root,
        out_jsonl=tmp_path / "out/mail.jsonl",
        manifest_path=tmp_path / "out/manifest.json",
        since=datetime(2026, 7, 1, tzinfo=timezone.utc),
        text_limit=1200,
    )

    assert resolved == paths
    assert report["rows_written"] == 2
    assert len(report["archive_readiness"]["databases"]) == 3
    rows = [json.loads(line) for line in (tmp_path / "out/mail.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["message_sha256"] for row in rows} == {"a" * 64, "b" * 64}
    assert (tmp_path / "out/mail.jsonl").stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(("schema", "stamped"), (("unknown", True), (MAIL_ARCHIVE_SCHEMA_VERSION, False)))
def test_archive_readiness_rejects_unknown_or_unstamped_archive(
    tmp_path: Path, schema: str, stamped: bool
) -> None:
    path = tmp_path / "mail.sqlite"
    _write_archive(path, sha="a" * 64, schema=schema, stamped=stamped)
    with pytest.raises(ValueError):
        assert_canonical_mail_archive_ready((path,))


@pytest.mark.parametrize(("messages", "meta"), ((False, True), (True, False)))
def test_archive_readiness_rejects_missing_required_table(
    tmp_path: Path, messages: bool, meta: bool
) -> None:
    path = tmp_path / "mail.sqlite"
    _write_archive(path, sha="a" * 64, messages=messages, meta=meta)
    with pytest.raises(ValueError, match="misses tables"):
        assert_canonical_mail_archive_ready((path,))


def test_identity_resolver_uses_only_existing_files_and_fails_when_empty(tmp_path: Path) -> None:
    root = tmp_path / "Mango_Data"
    identity = _identity_db(root)
    assert existing_tallanto_identity_dbs(root) == (identity,)
    with pytest.raises(FileNotFoundError, match="no Tallanto identity DB"):
        existing_tallanto_identity_dbs(tmp_path / "empty")


def test_download_reuses_the_canonical_root_constant() -> None:
    assert download.CANONICAL_RELATIVE_ROOT == CANONICAL_MAIL_ARCHIVE_ROOT


def test_manual_import_passes_only_existing_identity_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "Mango_Data"
    identity = _identity_db(root)
    captured = {}

    monkeypatch.setattr(mail_import, "DEFAULT_MAIL_DATA_ROOT", root)
    monkeypatch.setattr(
        mail_import,
        "run_mail_link_enrich",
        lambda config: captured.setdefault("config", config) or {},
    )
    mail_import.enrich_mail_links(
        timeline_db=tmp_path / "timeline.sqlite",
        allowed_root=tmp_path,
        out_dir=tmp_path / "out",
    )

    assert captured["config"].tallanto_identity_dbs == (identity,)
