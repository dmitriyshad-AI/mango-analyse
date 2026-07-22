from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts import run_customer_timeline_stage3_maintenance as stage3_cli


def _seed_count_tables(db_path: Path) -> None:
    with sqlite3.connect(db_path) as con:
        for table, rows in (("derived_signals", 2), ("customer_objections_v1", 1), ("family_links_v1", 3)):
            con.execute(f"CREATE TABLE {table} (id INTEGER)")
            con.executemany(f"INSERT INTO {table} VALUES (?)", [(index,) for index in range(rows)])


@pytest.mark.parametrize(("extra_args", "expected_apply"), [([], False), (["--apply"], True)])
def test_stage3_cli_writes_table_counts_and_requires_explicit_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected_apply: bool,
) -> None:
    db_path = tmp_path / "customer_timeline.sqlite"
    _seed_count_tables(db_path)
    captured = {}

    def fake_run(config):  # type: ignore[no-untyped-def]
        captured["config"] = config
        return {"mode": "dry_run"}

    monkeypatch.setattr(stage3_cli, "run_stage3_maintenance", fake_run)

    canonical_calls = tmp_path / "canonical_calls.sqlite"
    canonical_calls.touch()

    assert stage3_cli.main(
        [
            "--db-path",
            str(db_path),
            "--output",
            str(tmp_path / "report"),
            "--canonical-calls-db",
            str(canonical_calls),
            "--as-of",
            "2026-07-22T18:00:00Z",
            *extra_args,
        ]
    ) == 0

    assert captured["config"].apply is expected_apply
    assert captured["config"].canonical_calls_db_path == canonical_calls
    assert captured["config"].signal_as_of.isoformat() == "2026-07-22T18:00:00+00:00"
    report = json.loads((tmp_path / "report" / stage3_cli.REPORT_NAME).read_text(encoding="utf-8"))
    assert report["mode"] == "dry_run"
    assert report["table_counts"] == {
        "derived_signals": 2,
        "customer_objections_v1": 1,
        "family_links_v1": 3,
    }


def test_stage3_cli_refuses_prod_like_db_before_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prod_path = tmp_path / "customer_timeline_prod_20260722" / "customer_timeline.sqlite"
    prod_path.parent.mkdir()
    prod_path.touch()
    monkeypatch.setattr(stage3_cli, "run_stage3_maintenance", lambda config: pytest.fail("runner must not start"))

    with pytest.raises(ValueError, match="prod timeline"):
        stage3_cli.main(["--db-path", str(prod_path), "--output", str(tmp_path / "report"), "--apply"])
