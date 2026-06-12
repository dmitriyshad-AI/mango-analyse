#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_profile.builder import CustomerProfileBuilder, CustomerProfileBuildOptions
from mango_mvp.customer_profile.store import sha256_file


DEFAULT_SOURCE_ROOT = Path("/Users/dmitrijfabarisov/Projects/Mango analyse/product_data/customer_profiles/tz12_working_batch3")
DEFAULT_MASTER_CALLS_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db"
)
DEFAULT_BLACKLIST = Path(
    "/Users/dmitrijfabarisov/Yandex.Disk.localized/OpenClaw/Actual Mango Tests/analyze_rerun_20260611/blacklist_77.txt"
)
FIELD_COVERAGE = (
    "parent_name",
    "child_name",
    "grade",
    "subject",
    "format",
    "target_product",
    "next_step",
    "objection",
    "tallanto_balance",
    "tallanto_group",
    "payment_fact",
    "amo_stage",
    "amo_status",
    "child_slot_merge_candidate",
)
CHILD_FIELDS = ("child_name", "grade", "subject", "child_slot_merge_candidate")


@dataclass(frozen=True)
class Tz16ProfileBuildConfig:
    source_root: Path
    out_root: Path
    master_calls_db: Path
    blacklist_path: Path
    tenant_id: str = "foton"
    micro_count: int = 5


def build_tz16_profiles_v7(config: Tz16ProfileBuildConfig) -> Mapping[str, Any]:
    config = Tz16ProfileBuildConfig(
        source_root=config.source_root.expanduser().resolve(strict=False),
        out_root=config.out_root.expanduser().resolve(strict=False),
        master_calls_db=config.master_calls_db.expanduser().resolve(strict=False),
        blacklist_path=config.blacklist_path.expanduser().resolve(strict=False),
        tenant_id=config.tenant_id,
        micro_count=config.micro_count,
    )
    started = time.perf_counter()
    config.out_root.mkdir(parents=True, exist_ok=True)
    old_profiles_db = config.source_root / "customer_profiles.sqlite"
    source_timeline_db = config.source_root / "customer_timeline.sqlite"
    copied_timeline_db = config.out_root / "customer_timeline.sqlite"
    full_profiles_db = config.out_root / "customer_profiles.sqlite"
    idempotence_profiles_db = config.out_root / "customer_profiles_idempotence.sqlite"
    micro_profiles_db = config.out_root / "customer_profiles_micro.sqlite"

    before_hash = hash_directory(config.source_root)
    copy_timeline(source_timeline_db, copied_timeline_db)
    micro_ids = select_micro_customer_ids(copied_timeline_db, tenant_id=config.tenant_id, limit=config.micro_count)
    micro_report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=copied_timeline_db,
            profiles_db=micro_profiles_db,
            master_calls_db=config.master_calls_db,
            tenant_id=config.tenant_id,
            customer_ids=tuple(micro_ids),
            build_id="tz16_micro_v7",
        )
    ).build()
    full_started = time.perf_counter()
    full_report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=copied_timeline_db,
            profiles_db=full_profiles_db,
            master_calls_db=config.master_calls_db,
            tenant_id=config.tenant_id,
            build_id="tz16_profiles_v7",
        )
    ).build()
    full_seconds = round(time.perf_counter() - full_started, 3)
    repeat_report = CustomerProfileBuilder(
        CustomerProfileBuildOptions(
            timeline_db=copied_timeline_db,
            profiles_db=idempotence_profiles_db,
            master_calls_db=config.master_calls_db,
            tenant_id=config.tenant_id,
            build_id="tz16_profiles_v7_repeat",
        )
    ).build()
    old_metrics = profile_metrics(old_profiles_db)
    new_metrics = profile_metrics(full_profiles_db)
    idempotence = {
        "content_signature_equal": profile_content_signature(full_profiles_db) == profile_content_signature(idempotence_profiles_db),
        "first_profiles_built": full_report.get("profiles_built"),
        "second_profiles_built": repeat_report.get("profiles_built"),
        "first_fields_written": full_report.get("fields_written"),
        "second_fields_written": repeat_report.get("fields_written"),
    }
    after_hash = hash_directory(config.source_root)
    summary = {
        "schema_version": "tz16_profiles_v7_build_v1",
        "source_root": str(config.source_root),
        "out_root": str(config.out_root),
        "timeline_db_copied": str(copied_timeline_db),
        "profiles_db": str(full_profiles_db),
        "micro_profiles_db": str(micro_profiles_db),
        "idempotence_profiles_db": str(idempotence_profiles_db),
        "source_tz12_hash_before": before_hash,
        "source_tz12_hash_after": after_hash,
        "source_tz12_unchanged": before_hash == after_hash,
        "source_timeline_sha256": before_hash.get("files", {}).get("customer_timeline.sqlite", {}).get("sha256"),
        "copied_timeline_sha256": sha256_file(copied_timeline_db),
        "micro": {
            "customer_count": len(micro_ids),
            "customer_ids_sha256": sha256_text("\n".join(micro_ids)),
            "report": micro_report,
        },
        "full_build": full_report,
        "full_build_seconds": full_seconds,
        "idempotence": idempotence,
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "analysis_counts": analysis_counts(config.master_calls_db, config.blacklist_path),
        "anonymized_examples": anonymized_examples(full_profiles_db, limit=5),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "safety": {
            "write_crm": False,
            "write_tallanto": False,
            "write_amo": False,
            "run_asr": False,
            "run_resolve_analyze": False,
            "source_tz12_opened_for_write": False,
        },
    }
    write_json(config.out_root / "summary.json", summary)
    write_json(config.out_root / "source_hash_before.json", before_hash)
    write_json(config.out_root / "source_hash_after.json", after_hash)
    write_json(config.out_root / "anonymized_examples.json", summary["anonymized_examples"])
    return summary


def copy_timeline(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == source.stat().st_size and sha256_file(target) == sha256_file(source):
        return
    shutil.copy2(source, target)


def select_micro_customer_ids(timeline_db: Path, *, tenant_id: str, limit: int) -> list[str]:
    con = sqlite3.connect(f"file:{timeline_db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            """
            SELECT ci.customer_id, COUNT(te.event_id) AS event_count, MAX(te.event_at) AS last_event_at
            FROM customer_identities ci
            LEFT JOIN timeline_events te ON te.customer_id = ci.customer_id AND te.tenant_id = ci.tenant_id
            WHERE ci.tenant_id = ?
              AND COALESCE(ci.primary_phone, '') <> ''
            GROUP BY ci.customer_id
            ORDER BY event_count DESC, last_event_at DESC, ci.customer_id
            LIMIT ?
            """,
            (tenant_id, int(limit)),
        ).fetchall()
    finally:
        con.close()
    return [str(row[0]) for row in rows]


def profile_metrics(profiles_db: Path) -> Mapping[str, Any]:
    con = sqlite3.connect(f"file:{profiles_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        profile_count = scalar_int(con, "SELECT COUNT(*) FROM customer_profiles")
        fields_total = scalar_int(con, "SELECT COUNT(*) FROM profile_fields")
        active_fields_total = scalar_int(con, "SELECT COUNT(*) FROM profile_fields WHERE superseded_by = ''")
        superseded_fields = scalar_int(con, "SELECT COUNT(*) FROM profile_fields WHERE superseded_by <> ''")
        coverage = {
            field: scalar_int(
                con,
                "SELECT COUNT(DISTINCT profile_id) FROM profile_fields WHERE field = ? AND superseded_by = ''",
                (field,),
            )
            for field in FIELD_COVERAGE
        }
        profiles_with_2plus_children = scalar_int(
            con,
            f"""
            SELECT COUNT(*) FROM (
              SELECT profile_id, COUNT(DISTINCT child_key) AS child_count
              FROM profile_fields
              WHERE superseded_by = ''
                AND child_key <> ''
                AND field IN ({",".join("?" for _ in CHILD_FIELDS)})
              GROUP BY profile_id
              HAVING child_count >= 2
            )
            """,
            CHILD_FIELDS,
        )
        merge_candidate_profiles = scalar_int(
            con,
            """
            SELECT COUNT(DISTINCT profile_id)
            FROM profile_fields
            WHERE field = 'child_slot_merge_candidate' AND superseded_by = ''
            """,
        )
        merge_candidate_markers = scalar_int(
            con,
            "SELECT COUNT(*) FROM profile_fields WHERE field = 'child_slot_merge_candidate' AND superseded_by = ''",
        )
        field_counts = dict(
            (str(row["field"]), int(row["count"]))
            for row in con.execute(
                """
                SELECT field, COUNT(*) AS count
                FROM profile_fields
                WHERE superseded_by = ''
                GROUP BY field
                ORDER BY count DESC, field
                """
            ).fetchall()
        )
        superseded_by_field = dict(
            (str(row["field"]), int(row["count"]))
            for row in con.execute(
                """
                SELECT field, COUNT(*) AS count
                FROM profile_fields
                WHERE superseded_by <> ''
                GROUP BY field
                ORDER BY count DESC, field
                """
            ).fetchall()
        )
    finally:
        con.close()
    return {
        "profile_count": profile_count,
        "fields_total": fields_total,
        "active_fields_total": active_fields_total,
        "superseded_fields": superseded_fields,
        "coverage_profiles_by_field": coverage,
        "profiles_with_2plus_children": profiles_with_2plus_children,
        "merge_candidate_profiles": merge_candidate_profiles,
        "merge_candidate_markers": merge_candidate_markers,
        "active_field_counts": field_counts,
        "superseded_by_field": superseded_by_field,
    }


def analysis_counts(master_calls_db: Path, blacklist_path: Path) -> Mapping[str, Any]:
    blacklist_ids = read_id_set(blacklist_path)
    con = sqlite3.connect(f"file:{master_calls_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        table = "canonical_calls" if table_exists(con, "canonical_calls") else "call_records"
        id_column = "canonical_call_id" if has_column(con, table, "canonical_call_id") else "id"
        total_done = scalar_int(con, f"SELECT COUNT(*) FROM {table} WHERE analysis_status = 'done' AND analysis_json IS NOT NULL")
        valid_json = scalar_int(
            con,
            f"SELECT COUNT(*) FROM {table} WHERE analysis_status = 'done' AND analysis_json IS NOT NULL AND json_valid(analysis_json)",
        )
        v7 = scalar_int(
            con,
            f"""
            SELECT COUNT(*) FROM {table}
            WHERE analysis_status = 'done'
              AND analysis_json IS NOT NULL
              AND json_valid(analysis_json)
              AND json_extract(analysis_json, '$.analysis_meta.analysis_prompt_version') = 'v7'
            """,
        )
        non_v7 = scalar_int(
            con,
            f"""
            SELECT COUNT(*) FROM {table}
            WHERE analysis_status = 'done'
              AND analysis_json IS NOT NULL
              AND (
                NOT json_valid(analysis_json)
                OR COALESCE(json_extract(analysis_json, '$.analysis_meta.analysis_prompt_version'), '') <> 'v7'
              )
            """,
        )
        blacklist_present = count_ids(con, table, id_column, blacklist_ids)
        blacklist_v7 = count_ids(
            con,
            table,
            id_column,
            blacklist_ids,
            extra_where="json_valid(analysis_json) AND json_extract(analysis_json, '$.analysis_meta.analysis_prompt_version') = 'v7'",
        )
    finally:
        con.close()
    return {
        "master_calls_db": str(master_calls_db),
        "blacklist_path": str(blacklist_path),
        "analysis_done_with_json": total_done,
        "analysis_done_valid_json": valid_json,
        "analysis_prompt_version_v7": v7,
        "analysis_prompt_version_not_v7": non_v7,
        "blacklist_ids_loaded": len(blacklist_ids),
        "blacklist_ids_present_in_master": blacklist_present,
        "blacklist_ids_with_v7": blacklist_v7,
        "blacklist_ids_preserved_old": blacklist_present - blacklist_v7,
    }


def count_ids(
    con: sqlite3.Connection,
    table: str,
    id_column: str,
    ids: set[str],
    *,
    extra_where: str = "",
) -> int:
    if not ids:
        return 0
    total = 0
    sorted_ids = sorted(ids, key=lambda item: int(item) if item.isdigit() else item)
    for offset in range(0, len(sorted_ids), 500):
        batch = sorted_ids[offset : offset + 500]
        placeholders = ",".join("?" for _ in batch)
        where = f"CAST({id_column} AS TEXT) IN ({placeholders})"
        if extra_where:
            where = f"{where} AND {extra_where}"
        total += scalar_int(con, f"SELECT COUNT(*) FROM {table} WHERE {where}", tuple(batch))
    return total


def anonymized_examples(profiles_db: Path, *, limit: int) -> list[Mapping[str, Any]]:
    con = sqlite3.connect(f"file:{profiles_db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    examples: list[Mapping[str, Any]] = []
    try:
        rows = con.execute(
            """
            SELECT profile_id, source_event_count, primary_phone
            FROM customer_profiles
            ORDER BY source_event_count DESC, profile_id
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        for index, row in enumerate(rows, start=1):
            fields = con.execute(
                """
                SELECT field, child_key, brand, source_system, LENGTH(value) AS value_len
                FROM profile_fields
                WHERE profile_id = ? AND superseded_by = ''
                ORDER BY field, child_key, source_system
                """,
                (row["profile_id"],),
            ).fetchall()
            field_names = sorted({str(item["field"]) for item in fields})
            child_slots = sorted({str(item["child_key"]) for item in fields if str(item["child_key"] or "")})
            brands = sorted({str(item["brand"]) for item in fields if str(item["brand"] or "")})
            source_systems = sorted({str(item["source_system"]) for item in fields if str(item["source_system"] or "")})
            examples.append(
                {
                    "example_id": f"profile_example_{index}",
                    "profile_hash": sha256_text(str(row["profile_id"]))[:12],
                    "has_phone": bool(str(row["primary_phone"] or "")),
                    "source_event_count": int(row["source_event_count"] or 0),
                    "active_field_count": len(fields),
                    "active_field_names": field_names,
                    "child_slot_count": len(child_slots),
                    "brand_set": brands,
                    "source_systems": source_systems,
                    "max_value_len": max((int(item["value_len"] or 0) for item in fields), default=0),
                }
            )
    finally:
        con.close()
    return examples


def profile_content_signature(profiles_db: Path) -> str:
    con = sqlite3.connect(f"file:{profiles_db}?mode=ro", uri=True)
    try:
        digest = hashlib.sha256()
        for query in (
            """
            SELECT profile_id, tenant_id, primary_phone, display_name, source_event_count, COALESCE(last_event_at, '')
            FROM customer_profiles
            ORDER BY profile_id
            """,
            """
            SELECT field_id, profile_id, field, value, child_key, brand, source_system, source_ref,
                   event_at, quote, superseded_by
            FROM profile_fields
            ORDER BY field_id
            """,
        ):
            for row in con.execute(query):
                digest.update(json.dumps(tuple(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                digest.update(b"\n")
        return digest.hexdigest()
    finally:
        con.close()


def hash_directory(root: Path) -> Mapping[str, Any]:
    files: dict[str, Mapping[str, Any]] = {}
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        files[path.name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    payload = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"root": str(root), "files": files, "combined_sha256": sha256_text(payload)}


def read_id_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def has_column(con: sqlite3.Connection, table: str, column: str) -> bool:
    return any(str(row[1]) == column for row in con.execute(f"PRAGMA table_info({table})").fetchall())


def scalar_int(con: sqlite3.Connection, query: str, params: Sequence[Any] = ()) -> int:
    return int(con.execute(query, tuple(params)).fetchone()[0] or 0)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TZ-16 build customer profiles on v7 summaries.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--master-calls-db", type=Path, default=DEFAULT_MASTER_CALLS_DB)
    parser.add_argument("--blacklist-path", type=Path, default=DEFAULT_BLACKLIST)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--micro-count", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_tz16_profiles_v7(
        Tz16ProfileBuildConfig(
            source_root=args.source_root,
            out_root=args.out_root,
            master_calls_db=args.master_calls_db,
            blacklist_path=args.blacklist_path,
            tenant_id=args.tenant_id,
            micro_count=args.micro_count,
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
