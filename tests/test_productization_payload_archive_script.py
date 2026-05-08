from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts import mango_office_payload_archive


def test_payload_archive_script_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "payload_archive_audit.json"
    fake_report = {
        "summary": {"validation_ok": True, "archived_entries": 1},
        "audit": {
            "archive_root": str(tmp_path / "archive"),
            "archived_entries": 1,
            "archive_files": 1,
            "archive_file_rows": 1,
            "sidecar_rows": 1,
            "sidecar_refs_present": 1,
            "blocked": 0,
            "blocked_reasons": {},
            "warnings": 0,
            "warning_reasons": {},
            "source_kind_counts": {},
            "tenant_provider_counts": {"foton|mango": 1},
        },
    }

    with patch.object(
        mango_office_payload_archive,
        "archive_mango_payloads_and_update_sidecar",
        return_value=fake_report,
    ):
        rc = mango_office_payload_archive.main(
            [
                "--db",
                str(tmp_path / "test.sqlite"),
                "--metadata-csv",
                str(tmp_path / "metadata.csv"),
                "--source-payload",
                str(tmp_path / "source.jsonl"),
                "--archive-root",
                str(tmp_path / "archive"),
                "--out-root",
                str(tmp_path),
                "--out",
                str(out),
                "--replace",
            ]
        )

    assert rc == 0
    assert out.exists()
