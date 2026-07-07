from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "customer_timeline_publish_snapshot_v3"
DEFAULT_COUNT_TABLES = (
    "customer_identities",
    "identity_links",
    "customer_opportunities",
    "timeline_events",
    "timeline_event_fts",
    "bot_context_chunks",
    "timeline_conflicts",
    "derived_signals",
    "ingestion_cursors",
)
SIDE_SUFFIXES = ("-wal", "-shm", "-journal")


class PublishSnapshotError(RuntimeError):
    pass


@dataclass(frozen=True)
class PublishConfig:
    raw: Mapping[str, Any]
    path: Path

    @property
    def package_name(self) -> str:
        return str(self.raw.get("package_name") or self.path.stem)

    @property
    def staging_db(self) -> Path:
        return Path(str(self.raw["staging_db"])).expanduser().resolve(strict=False)

    @property
    def prod_db(self) -> Path:
        return Path(str(self.raw["prod_db"])).expanduser().resolve(strict=False)

    @property
    def snapshot_root(self) -> Path:
        return Path(str(self.raw["snapshot_root"])).expanduser().resolve(strict=False)

    @property
    def tenant_id(self) -> str:
        return str(self.raw.get("tenant_id") or "foton")

    @property
    def required_free_copies(self) -> int:
        return int(self.raw.get("required_free_copies") or 3)

    @property
    def count_tables(self) -> tuple[str, ...]:
        values = self.raw.get("count_tables") or DEFAULT_COUNT_TABLES
        return tuple(str(item) for item in values)

    @property
    def readers(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(item for item in (self.raw.get("readers") or ()) if isinstance(item, Mapping))

    @property
    def control_customers(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(item for item in (self.raw.get("control_customers") or ()) if isinstance(item, Mapping))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_config(path: Path) -> PublishConfig:
    cfg_path = path.expanduser().resolve(strict=False)
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "publish_snapshot_config_v1":
        raise PublishSnapshotError("config schema_version must be publish_snapshot_config_v1")
    for key in ("staging_db", "prod_db", "snapshot_root"):
        if not payload.get(key):
            raise PublishSnapshotError(f"missing config key: {key}")
    return PublishConfig(raw=payload, path=cfg_path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sqlite_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)


def quick_check(path: Path) -> str:
    with sqlite_ro(path) as con:
        return str(con.execute("PRAGMA quick_check").fetchone()[0])


def foreign_key_check(path: Path) -> list[tuple[Any, ...]]:
    with sqlite_ro(path) as con:
        return [tuple(row) for row in con.execute("PRAGMA foreign_key_check").fetchall()]


def user_version(path: Path) -> int:
    with sqlite_ro(path) as con:
        return int(con.execute("PRAGMA user_version").fetchone()[0])


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    row = con.execute("SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name=?", (table,)).fetchone()
    return row is not None


def table_counts(path: Path, tables: Sequence[str] = DEFAULT_COUNT_TABLES) -> dict[str, int]:
    result: dict[str, int] = {}
    with sqlite_ro(path) as con:
        for table in tables:
            if table_exists(con, table):
                result[table] = int(con.execute(f"SELECT COUNT(*) FROM {quote_ident(table)}").fetchone()[0])
    return result


def schema_signature(path: Path) -> dict[str, str]:
    with sqlite_ro(path) as con:
        rows = con.execute(
            """
            SELECT type, name, COALESCE(sql, '') AS sql
            FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ).fetchall()
    return {f"{row[0]}:{row[1]}": str(row[2]) for row in rows}


def schema_diff(left: Path, right: Path) -> Mapping[str, Any]:
    a = schema_signature(left)
    b = schema_signature(right)
    keys = sorted(set(a) | set(b))
    changed = [key for key in keys if a.get(key) != b.get(key)]
    return {
        "left": str(left),
        "right": str(right),
        "changed_count": len(changed),
        "only_left": [key for key in keys if key in a and key not in b],
        "only_right": [key for key in keys if key in b and key not in a],
        "changed": [key for key in changed if key in a and key in b],
    }


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def sidecar_paths(db_path: Path) -> list[Path]:
    return [Path(str(db_path) + suffix) for suffix in SIDE_SUFFIXES]


def remove_sidecars(db_path: Path, *, execute: bool) -> list[str]:
    removed: list[str] = []
    for path in sidecar_paths(db_path):
        if path.exists():
            removed.append(str(path))
            if execute:
                path.unlink()
    return removed


def wal_checkpoint_truncate(db_path: Path) -> Mapping[str, Any]:
    con = sqlite3.connect(str(db_path), timeout=30)
    try:
        row = tuple(con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone() or ())
    finally:
        con.close()
    wal = Path(str(db_path) + "-wal")
    wal_size = wal.stat().st_size if wal.exists() else 0
    ok = len(row) >= 1 and int(row[0]) == 0 and wal_size == 0
    if not ok:
        raise PublishSnapshotError(f"wal checkpoint failed: row={row}, wal_size={wal_size}")
    return {"row": row, "wal_path": str(wal), "wal_size": wal_size}


def vacuum_into(source_db: Path, target_db: Path) -> None:
    if target_db.exists():
        raise PublishSnapshotError(f"snapshot DB already exists: {target_db}")
    target_db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(source_db), timeout=30)
    try:
        con.execute(f"VACUUM INTO {sql_literal(str(target_db))}")
    finally:
        con.close()


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def disk_report(path: Path, required_bytes: int) -> Mapping[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "free_bytes": usage.free,
        "required_bytes": required_bytes,
        "ok": usage.free >= required_bytes,
    }


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: int = 120,
    execute: bool = True,
) -> Mapping[str, Any]:
    rendered = [str(item) for item in command]
    if not execute:
        return {"command": rendered, "cwd": str(cwd) if cwd else None, "skipped": True, "rc": None}
    proc = subprocess.run(
        rendered,
        cwd=str(cwd) if cwd else None,
        env={**os.environ, **dict(env or {})},
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return {
        "command": rendered,
        "cwd": str(cwd) if cwd else None,
        "rc": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def render_template(value: str, variables: Mapping[str, Any]) -> str:
    out = value
    for key, raw in variables.items():
        out = out.replace("{{" + key + "}}", str(raw))
    return out


def render_command(command: Sequence[str], variables: Mapping[str, Any]) -> list[str]:
    return [render_template(str(part), variables) for part in command]


def lsof_holders(path: Path) -> list[str]:
    proc = subprocess.run(["lsof", str(path)], text=True, capture_output=True)
    if proc.returncode not in (0, 1):
        return [f"lsof_error rc={proc.returncode}: {proc.stderr.strip()}"]
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return lines[1:] if len(lines) > 1 else []


def git_head(worktree: Path) -> str | None:
    proc = subprocess.run(["git", "-C", str(worktree), "rev-parse", "HEAD"], text=True, capture_output=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_status_short(worktree: Path) -> str | None:
    proc = subprocess.run(["git", "-C", str(worktree), "status", "--short"], text=True, capture_output=True)
    return proc.stdout if proc.returncode == 0 else None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def report_base(config: PublishConfig, command: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "config": str(config.path),
        "package_name": config.package_name,
        "generated_at": utc_now().isoformat(),
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)


def finish_cli(report: Mapping[str, Any], out: Path | None, *, ok: bool) -> int:
    if out:
        write_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 1

