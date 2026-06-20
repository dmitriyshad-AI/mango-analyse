from __future__ import annotations

from pathlib import Path

import pytest

from mango_mvp.customer_timeline import full_memory_ingest as fmi


def _write_required_inputs(tmp_path: Path) -> tuple[Path, tuple[Path, Path]]:
    identity = tmp_path / "identity.sqlite"
    identity.write_bytes(b"not used by faked mail stage")
    event1 = tmp_path / "events1.jsonl"
    event2 = tmp_path / "events2.jsonl"
    event1.write_text("", encoding="utf-8")
    event2.write_text("", encoding="utf-8")
    return identity, (event1, event2)


def test_full_memory_ingest_refuses_to_use_production_db_as_test_target(tmp_path: Path) -> None:
    identity, events = _write_required_inputs(tmp_path)
    test_out = tmp_path / "target"
    config = fmi.FullMemoryIngestConfig(
        project_root=tmp_path,
        production_db=test_out / "customer_timeline.sqlite",
        test_out_root=test_out,
        identity_db=identity,
        event_jsonl_paths=events,
    )

    with pytest.raises(RuntimeError, match="must not be the appointed production DB"):
        fmi.run_full_memory_test_procedure(config)


def test_full_memory_ingest_test_copy_runs_stages_in_order_and_never_production_apply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity, events = _write_required_inputs(tmp_path)
    config = fmi.FullMemoryIngestConfig(
        project_root=tmp_path,
        production_db=tmp_path / "prod" / "customer_timeline.sqlite",
        test_out_root=tmp_path / "testcopy",
        identity_db=identity,
        event_jsonl_paths=events,
    )
    calls: list[str] = []

    def fake_dry_run(_mail_config: object) -> dict[str, object]:
        calls.append("mail_dry_run")
        return {"mode": "dry_run", "counts": {"would_create_events": 0}}

    def fake_canonical(_config: object) -> dict[str, object]:
        calls.append("canonical")
        return {"summary": {"source_event_counts": {}}, "paths": {}, "safety": {}}

    def fake_apply(_mail_config: object, *, backup_manifest_path: Path) -> dict[str, object]:
        calls.append(f"mail_apply:{backup_manifest_path.name}")
        return {"mode": "apply", "counts": {"created_events": 0, "selected_new_events": 0}}

    def fake_restore(_mail_config: object, *, backup_manifest_path: Path) -> dict[str, object]:
        calls.append(f"restore:{backup_manifest_path.name}")
        return {"mode": "restore", "backup_manifest_path": str(backup_manifest_path)}

    monkeypatch.setattr(fmi, "dry_run_stage2_mail_ingest", fake_dry_run)
    monkeypatch.setattr(fmi, "run_canonical_import", fake_canonical)
    monkeypatch.setattr(fmi, "apply_stage2_mail_ingest", fake_apply)
    monkeypatch.setattr(fmi, "restore_timeline_backup", fake_restore)

    report = fmi.run_full_memory_test_procedure(config)

    assert calls == [
        "mail_dry_run",
        "canonical",
        "mail_apply:backup_manifest.json",
        "canonical",
        "mail_apply:backup_manifest.json",
        "restore:backup_manifest.json",
    ]
    assert report["production_target"]["appointed_db"].endswith("prod/customer_timeline.sqlite")
    assert report["production_target"]["apply_performed"] is False
    assert report["validation"]["backup_created_before_first_importer"] is True
    assert report["validation"]["production_apply_not_performed"] is True
    assert Path(report["backup"]["manifest_path"]).exists()
