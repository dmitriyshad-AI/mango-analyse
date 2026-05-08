from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts import mango_office_quarantine_test_ingest


def test_quarantine_test_ingest_script_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "audit.json"
    fake_report = {
        "summary": {"validation_ok": True, "db_call_records": 1},
        "audit": {
            "blocked": 0,
            "blocked_reasons": {},
            "warnings": 0,
            "warning_reasons": {},
            "status_counts": {},
            "direction_counts": {},
            "db_call_records": 1,
            "metadata_rows": 1,
            "audio_files": 1,
            "current_call_records_model_gaps": [],
        },
    }

    with patch.object(
        mango_office_quarantine_test_ingest,
        "run_quarantine_test_ingest",
        return_value=fake_report,
    ):
        rc = mango_office_quarantine_test_ingest.main(
            [
                "--audio-dir",
                str(tmp_path / "audio"),
                "--metadata-csv",
                str(tmp_path / "metadata.csv"),
                "--db",
                str(tmp_path / "out" / "test.sqlite"),
                "--out-root",
                str(tmp_path / "out"),
                "--out",
                str(out),
                "--replace",
            ]
        )

    assert rc == 0
    assert out.exists()
