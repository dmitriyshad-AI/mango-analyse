from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from mango_mvp.customer_timeline.safe_copy import safe_copy_prod_snapshot
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore, json_dumps


FTS_REBUILD_PORT_THRESHOLD_SECONDS = 600.0


def measure_customer_timeline_window(
    source_db: Path | str,
    out_dir: Path | str,
    *,
    batch_size: int = 1000,
    retries: int = 3,
) -> Mapping[str, Any]:
    """Measure D/J maintenance costs on a bounded local copy.

    The source database is copied with the WAL-safe file protocol. SQLite writes,
    FTS rebuild, checkpoint, and content-key backfill happen only on the copied
    database under `out_dir`.
    """

    output = Path(out_dir).expanduser().resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    copy_path = output / "customer_timeline_window_measurement.sqlite"
    copy_manifest = safe_copy_prod_snapshot(source_db, copy_path, retries=retries)
    timings: dict[str, float] = {}

    started = time.monotonic()
    store = CustomerTimelineSQLiteStore(copy_path, allowed_root=output)
    timings["schema_bootstrap_seconds"] = time.monotonic() - started
    try:
        missing_before = store.count_missing_timeline_email_content_keys()
        started = time.monotonic()
        backfill = store.backfill_timeline_event_content_keys(batch_size=batch_size)
        timings["content_key_backfill_seconds"] = time.monotonic() - started
        missing_after = store.count_missing_timeline_email_content_keys()

        started = time.monotonic()
        store._rebuild_fts_indexes()  # noqa: SLF001 - maintenance measurement of the store's own rebuild path.
        store._con.commit()  # noqa: SLF001
        timings["fts_rebuild_seconds"] = time.monotonic() - started

        started = time.monotonic()
        checkpoint_rows = [tuple(row) for row in store._con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()]  # noqa: SLF001
        timings["copy_wal_checkpoint_truncate_seconds"] = time.monotonic() - started
        quick_check = str(store._con.execute("PRAGMA quick_check").fetchone()[0])  # noqa: SLF001
        summary = store.summary()
    finally:
        store.close()

    report = {
        "schema_version": "customer_timeline_window_measurement_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_db": str(Path(source_db).expanduser().resolve(strict=True)),
        "copy_db": str(copy_path),
        "copy_manifest": copy_manifest,
        "timings": timings,
        "content_key_backfill": {
            "missing_before": missing_before,
            "missing_after": missing_after,
            **dict(backfill),
        },
        "checkpoint_rows": checkpoint_rows,
        "quick_check": quick_check,
        "summary_counts": summary["counts"],
        "fba43e4_decision": {
            "threshold_seconds": FTS_REBUILD_PORT_THRESHOLD_SECONDS,
            "port_required": timings["fts_rebuild_seconds"] > FTS_REBUILD_PORT_THRESHOLD_SECONDS,
            "reason": "fts_rebuild_over_threshold"
            if timings["fts_rebuild_seconds"] > FTS_REBUILD_PORT_THRESHOLD_SECONDS
            else "fts_rebuild_within_threshold",
        },
        "safety": {
            "source_sqlite_opened": False,
            "source_checkpoint": False,
            "writes_target": "copy_only",
        },
    }
    report_path = output / "customer_timeline_window_measurement.json"
    report_path.write_text(json_dumps(report), encoding="utf-8")
    return {**report, "report_path": str(report_path)}


__all__ = ["FTS_REBUILD_PORT_THRESHOLD_SECONDS", "measure_customer_timeline_window"]
