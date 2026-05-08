from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from mango_mvp.productization.manager_identity import (
    MANAGER_IDENTITY_TABLE,
    MANAGER_IDENTITY_VIEW,
    install_manager_identity_map,
)
from mango_mvp.productization.provider_metadata import install_provider_metadata_sidecar
from tests.test_productization_provider_metadata import build_disposable_db


def make_manager_row(audio_dir: Path, event_id: str, manager_extension: str) -> dict:
    filename = f"2026-05-07__09-00-00__79990000000__mango_{manager_extension}_{event_id}.mp3"
    audio_path = audio_dir / filename
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake-mp3")
    return {
        "filename": filename,
        "target_audio_path": str(audio_path),
        "phone": "+79990000000",
        "client_phone": "+79990000000",
        "manager": manager_extension,
        "manager_name": f"mango_{manager_extension}",
        "started_at": "2026-05-07T09:00:00+03:00",
        "start_time": "2026-05-07T09:00:00+03:00",
        "direction": "inbound",
        "call_id": event_id,
        "record_id": f"REC-{event_id}",
        "recording_id": f"REC-{event_id}",
        "event_key": f"foton:mango:{event_id}",
        "provider_call_id": event_id,
        "duration_sec": "10.0",
        "checksum_sha256": "a" * 64,
        "source_size_bytes": "8",
        "source": "mango_api_capture",
        "tenant_id": "foton",
        "provider": "mango",
    }


def write_mango_users(path: Path, users: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "users": [
            {
                "general": {
                    "name": user["name"],
                    "email": user.get("email"),
                    "department": user.get("department"),
                    "position": user.get("position"),
                },
                "telephony": {"extension": user["extension"]},
            }
            for user in users
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_amo_users(path: Path, users: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(users, ensure_ascii=False), encoding="utf-8")


def build_db_with_provider_sidecar(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path, Path]:
    db_path, metadata_csv, out_root = build_disposable_db(tmp_path, rows)
    install_provider_metadata_sidecar(
        db_path=db_path,
        metadata_csv_path=metadata_csv,
        out_allowed_root=out_root,
        replace_existing=True,
    )
    return db_path, metadata_csv, out_root


def test_manager_identity_map_installs_rows_view_and_audit(tmp_path: Path) -> None:
    audio_dir = tmp_path / "quarantine" / "audio"
    rows = [
        make_manager_row(audio_dir, "CALL-1", "101"),
        make_manager_row(audio_dir, "CALL-2", "101"),
        make_manager_row(audio_dir, "CALL-3", "102"),
    ]
    db_path, _metadata_csv, out_root = build_db_with_provider_sidecar(tmp_path, rows)
    mango_users_path = tmp_path / "config" / "mango_users.json"
    amo_users_path = tmp_path / "config" / "amo_users.json"
    write_mango_users(
        mango_users_path,
        [
            {"extension": "101", "name": "Анна Менеджер", "email": "anna@example.com"},
            {"extension": "102", "name": "Олег Менеджер", "email": "oleg@example.com"},
        ],
    )
    write_amo_users(amo_users_path, [{"id": 9001, "name": "Анна Менеджер", "email": "anna@example.com"}])

    report = install_manager_identity_map(
        db_path=db_path,
        mango_users_path=mango_users_path,
        amo_users_path=amo_users_path,
        out_allowed_root=out_root,
        replace_existing=True,
        csv_out=out_root / "manager_identity.csv",
    )

    assert report["summary"]["validation_ok"] is True
    assert report["summary"]["manager_extensions"] == 2
    assert report["summary"]["sidecar_rows"] == 3
    assert report["summary"]["view_rows"] == 3
    assert report["summary"]["mapped_mango_users"] == 2
    assert report["summary"]["missing_mango_users"] == 0
    assert report["summary"]["crm_owner_matched"] == 1
    assert report["summary"]["crm_owner_unmatched"] == 1
    assert report["summary"]["calls_with_crm_owner"] == 2
    assert report["audit"]["crm_owner_unmatched_call_count"] == 1
    assert report["audit"]["crm_match_status_counts"] == {"matched_email": 1, "unmatched": 1}
    assert report["audit"]["manual_review_items"][0]["manager_extension"] == "102"

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        view_rows = con.execute(
            f"""
            SELECT manager_extension, manager_display_name, manager_crm_owner_id
            FROM {MANAGER_IDENTITY_VIEW}
            ORDER BY provider_call_id
            """
        ).fetchall()
        table_rows = con.execute(f"SELECT count(*) FROM {MANAGER_IDENTITY_TABLE}").fetchone()[0]

    assert table_rows == 2
    assert [row["manager_display_name"] for row in view_rows] == [
        "Анна Менеджер",
        "Анна Менеджер",
        "Олег Менеджер",
    ]
    assert [row["manager_crm_owner_id"] for row in view_rows] == [9001, 9001, None]


def test_manager_identity_map_blocks_missing_mango_user(tmp_path: Path) -> None:
    audio_dir = tmp_path / "quarantine" / "audio"
    row = make_manager_row(audio_dir, "CALL-1", "103")
    db_path, _metadata_csv, out_root = build_db_with_provider_sidecar(tmp_path, [row])
    mango_users_path = tmp_path / "config" / "mango_users.json"
    write_mango_users(mango_users_path, [])

    report = install_manager_identity_map(
        db_path=db_path,
        mango_users_path=mango_users_path,
        out_allowed_root=out_root,
        replace_existing=True,
    )

    assert report["summary"]["validation_ok"] is False
    assert report["summary"]["blocked"] == 1
    assert report["audit"]["blocked_reasons"]["missing_mango_users"] == 1
    assert report["audit"]["manual_review_items"][0]["reason"] == "missing_mango_user"


def test_manager_identity_map_is_idempotent_update(tmp_path: Path) -> None:
    audio_dir = tmp_path / "quarantine" / "audio"
    row = make_manager_row(audio_dir, "CALL-1", "101")
    db_path, _metadata_csv, out_root = build_db_with_provider_sidecar(tmp_path, [row])
    mango_users_path = tmp_path / "config" / "mango_users.json"
    write_mango_users(mango_users_path, [{"extension": "101", "name": "Анна Менеджер", "email": "anna@example.com"}])

    first = install_manager_identity_map(db_path, mango_users_path, out_root, replace_existing=True)
    second = install_manager_identity_map(db_path, mango_users_path, out_root, replace_existing=False)

    assert first["summary"]["replaced_existing_table"] is True
    assert second["summary"]["replaced_existing_table"] is False
    assert second["summary"]["manager_extensions"] == 1
    assert second["summary"]["view_rows"] == 1

    with sqlite3.connect(db_path) as con:
        table_rows = con.execute(f"SELECT count(*) FROM {MANAGER_IDENTITY_TABLE}").fetchone()[0]
    assert table_rows == 1


def test_manager_identity_map_refuses_runtime_db_name(tmp_path: Path) -> None:
    db_path = tmp_path / "mango_mvp.db"
    db_path.write_bytes(b"not-real-db")
    mango_users_path = tmp_path / "mango_users.json"
    write_mango_users(mango_users_path, [])

    with pytest.raises(ValueError, match="runtime-looking DB"):
        install_manager_identity_map(
            db_path=db_path,
            mango_users_path=mango_users_path,
            out_allowed_root=tmp_path,
        )


def test_manager_identity_map_refuses_db_outside_allowed_root(tmp_path: Path) -> None:
    db_path = tmp_path / "outside" / "test.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_bytes(b"not-real-db")
    mango_users_path = tmp_path / "mango_users.json"
    write_mango_users(mango_users_path, [])

    with pytest.raises(ValueError, match="allowed root"):
        install_manager_identity_map(
            db_path=db_path,
            mango_users_path=mango_users_path,
            out_allowed_root=tmp_path / "allowed",
        )
