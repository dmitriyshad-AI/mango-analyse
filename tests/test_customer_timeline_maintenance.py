from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.customer_timeline import CustomerTimelineSQLiteStore
from mango_mvp.customer_timeline.maintenance import measure_customer_timeline_window

from tests.test_customer_timeline_store import email_event, identity


def test_measure_customer_timeline_window_runs_only_on_safe_copy(tmp_path: Path) -> None:
    source = tmp_path / "source" / "customer_timeline.sqlite"
    source.parent.mkdir()
    customer = identity()
    store = CustomerTimelineSQLiteStore(source, allowed_root=tmp_path)
    store.upsert_customer(customer)
    store.upsert_event(email_event(customer, source_id="mail-1"))
    store.close()
    source_bytes_before = source.read_bytes()

    out_dir = tmp_path / "measure"
    report = measure_customer_timeline_window(source, out_dir, batch_size=1, retries=1)

    assert source.read_bytes() == source_bytes_before
    assert Path(str(report["copy_db"])).exists()
    assert Path(str(report["report_path"])).exists()
    assert report["safety"] == {
        "source_sqlite_opened": False,
        "source_checkpoint": False,
        "writes_target": "copy_only",
    }
    assert report["quick_check"] == "ok"
    assert report["content_key_backfill"]["missing_after"] == 0
    assert report["fba43e4_decision"]["port_required"] is False
    stored = json.loads(Path(str(report["report_path"])).read_text(encoding="utf-8"))
    assert stored["copy_manifest"]["source_sha256"] == stored["copy_manifest"]["target_sha256"]
