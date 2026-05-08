from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts import mango_office_provider_metadata_sidecar


def test_provider_metadata_sidecar_script_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "provider_metadata_audit.json"
    fake_report = {
        "summary": {"validation_ok": True, "sidecar_rows": 1},
        "audit": {
            "table_name": "provider_call_metadata",
            "metadata_rows": 1,
            "call_records": 1,
            "sidecar_rows": 1,
            "blocked": 0,
            "blocked_reasons": {},
            "warnings": 0,
            "warning_reasons": {},
            "tenant_provider_counts": {"foton|mango": 1},
            "manager_extension_counts": {"101": 1},
            "known_gaps": [],
        },
    }

    with patch.object(
        mango_office_provider_metadata_sidecar,
        "install_provider_metadata_sidecar",
        return_value=fake_report,
    ):
        rc = mango_office_provider_metadata_sidecar.main(
            [
                "--db",
                str(tmp_path / "test.sqlite"),
                "--metadata-csv",
                str(tmp_path / "metadata.csv"),
                "--out-root",
                str(tmp_path),
                "--out",
                str(out),
                "--replace",
            ]
        )

    assert rc == 0
    assert out.exists()
