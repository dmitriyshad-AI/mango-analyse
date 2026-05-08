from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from scripts import mango_office_manager_identity_map


def test_manager_identity_script_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "manager_identity_audit.json"
    fake_report = {
        "summary": {"validation_ok": True, "manager_extensions": 1},
        "audit": {
            "table_name": "manager_identity_map",
            "view_name": "provider_call_metadata_with_manager",
            "manager_extensions": 1,
            "sidecar_rows": 1,
            "view_rows": 1,
            "mapped_mango_users": 1,
            "missing_mango_users": 0,
            "crm_owner_matched": 1,
            "crm_owner_unmatched": 0,
            "calls_with_mango_user": 1,
            "calls_with_crm_owner": 1,
            "crm_owner_unmatched_call_count": 0,
            "blocked": 0,
            "blocked_reasons": {},
            "warnings": 0,
            "warning_reasons": {},
            "mapping_status_counts": {"mapped_mango_user": 1},
            "crm_match_status_counts": {"matched_email": 1},
            "manager_call_counts": {"101": 1},
            "manual_review_items": [],
        },
    }

    with patch.object(
        mango_office_manager_identity_map,
        "install_manager_identity_map",
        return_value=fake_report,
    ):
        rc = mango_office_manager_identity_map.main(
            [
                "--db",
                str(tmp_path / "test.sqlite"),
                "--mango-users",
                str(tmp_path / "mango_users.json"),
                "--amo-users",
                str(tmp_path / "amo_users.json"),
                "--out-root",
                str(tmp_path),
                "--out",
                str(out),
                "--csv-out",
                str(tmp_path / "manager_identity.csv"),
                "--replace",
            ]
        )

    assert rc == 0
    assert out.exists()
