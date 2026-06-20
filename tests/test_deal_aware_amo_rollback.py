from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from mango_mvp.deal_aware.amo_rollback import (
    ROLLBACK_CONFIRMATION,
    RetryPolicy,
    build_pre_write_snapshot_rows,
    call_with_retries,
    load_snapshot_rows,
    load_successful_rollback_keys,
    rollback_decision,
    rollback_summary,
    run_rollback,
    write_rollback_outputs,
)
from mango_mvp.deal_aware.deal_text_builder import DEAL_AI_FIELDS
from scripts.write_deal_aware_amo_fields import run_live_write


class FakeAmoError(ValueError):
    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class FakeSession:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def commit(self) -> None:
        self.events.append("commit")

    def rollback(self) -> None:
        self.events.append("rollback")

    def close(self) -> None:
        self.events.append("close")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_row() -> dict[str, str]:
    row = {field: f"{field}: безопасный тестовый текст" for field in DEAL_AI_FIELDS}
    row.update(
        {
            "review_id": "deal-stage5-00001",
            "selected_deal_id": "123",
            "stage5_decision": "allow_stage6_dry_run",
            "stage5_warning_gate_count": "0",
            "AI-приоритет сделки": "warm",
            "candidate_phone_count": "1",
            "tallanto_context_status": "exact_phone_single",
        }
    )
    row["AI-дата следующего касания"] = "2026-05-15"
    row["AI-дата обновления сделки"] = "2026-05-13T11:57:33+00:00"
    return row


def _field_catalog() -> dict[str, object]:
    return {
        "synced_at": "2026-05-13T11:22:02+00:00",
        "fields": [
            {
                "id": 1000 + index,
                "name": field,
                "type": "date_time" if field == "AI-дата обновления сделки" else "textarea",
                "is_api_only": False,
            }
            for index, field in enumerate(DEAL_AI_FIELDS)
        ],
    }


def _lead_with_values(values: dict[str, str]) -> dict[str, object]:
    fields = []
    for index, (field, value) in enumerate(values.items(), start=1000):
        fields.append(
            {
                "field_id": index,
                "field_name": field,
                "values": [{"value": value}],
            }
        )
    return {"id": 123, "custom_fields_values": fields}


def _live_files(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    input_csv = tmp_path / "candidates.csv"
    stage5_summary = tmp_path / "stage5_summary.json"
    field_catalog = tmp_path / "lead_field_catalog_cache.json"
    approval = tmp_path / "operator_approval.json"
    out = tmp_path / "out"
    _write_csv(input_csv, [_candidate_row()])
    stage5_summary.write_text("{}", encoding="utf-8")
    field_catalog.write_text(json.dumps(_field_catalog(), ensure_ascii=False), encoding="utf-8")
    approval.write_text(
        json.dumps({"input": str(input_csv), "expected_written": 1}, ensure_ascii=False),
        encoding="utf-8",
    )
    return input_csv, stage5_summary, field_catalog, approval, out


def test_pre_write_snapshot_rows_capture_old_and_new_values(tmp_path: Path) -> None:
    payload = {"AI-сводка по сделке": "новая сводка"}
    rows = build_pre_write_snapshot_rows(
        batch_id="batch1",
        input_csv=tmp_path / "input.csv",
        input_sha256="abc",
        row_index=7,
        review_id="review-7",
        lead_id="123",
        payload=payload,
        current_lead=_lead_with_values({"AI-сводка по сделке": "старая сводка"}),
        field_catalog=_field_catalog()["fields"],  # type: ignore[index]
        operator_approval_path=tmp_path / "approval.json",
        snapshot_taken_at="2026-05-15T12:00:00+00:00",
    )

    assert rows[0]["old_value"] == "старая сводка"
    assert rows[0]["new_value"] == "новая сводка"
    assert rows[0]["field_id"] == "1000"
    assert rows[0]["operator_approval_path"].endswith("approval.json")


def test_live_write_saves_snapshot_before_patch(tmp_path: Path) -> None:
    input_csv, stage5_summary, field_catalog, approval, out = _live_files(tmp_path)
    events: list[str] = []

    def snapshot_writer(path: Path, rows: list[dict[str, object]]) -> None:
        events.append("snapshot")
        assert path == out
        assert len(rows) == len(DEAL_AI_FIELDS)

    def fetcher(session: FakeSession, *, lead_id: int) -> dict[str, object]:
        events.append("fetch")
        return _lead_with_values({field: f"old {field}" for field in DEAL_AI_FIELDS})

    def updater(session: FakeSession, *, lead_id: int, field_payload: dict[str, object]) -> dict[str, object]:
        events.append("patch")
        return {"updated_fields": sorted(field_payload)}

    code = run_live_write(
        input_csv=input_csv,
        stage5_summary=stage5_summary,
        field_catalog_cache=field_catalog,
        out_root=out,
        live_confirmation="WRITE_AMO_DEAL_AWARE_LIVE",
        expected_written=1,
        operator_approval=approval,
        analysis_date="2026-05-13",
        delay_ms=0,
        session_factory=lambda: FakeSession(events),
        preflight_func=lambda session: (True, ""),
        fetch_lead_func=fetcher,
        send_update_func=updater,
        snapshot_writer=snapshot_writer,
    )

    assert code == 0
    assert events[:5] == ["fetch", "snapshot", "fetch", "patch", "commit"]


def test_live_write_blocks_patch_when_current_value_changed_after_snapshot(tmp_path: Path) -> None:
    input_csv, stage5_summary, field_catalog, approval, out = _live_files(tmp_path)
    events: list[str] = []
    fetch_values = [
        {field: f"old {field}" for field in DEAL_AI_FIELDS},
        {field: f"manual {field}" for field in DEAL_AI_FIELDS},
    ]

    def fetcher(session: FakeSession, *, lead_id: int) -> dict[str, object]:
        events.append("fetch")
        return _lead_with_values(fetch_values.pop(0))

    def updater(session: FakeSession, *, lead_id: int, field_payload: dict[str, object]) -> dict[str, object]:
        events.append("patch")
        return {"updated_fields": sorted(field_payload)}

    code = run_live_write(
        input_csv=input_csv,
        stage5_summary=stage5_summary,
        field_catalog_cache=field_catalog,
        out_root=out,
        live_confirmation="WRITE_AMO_DEAL_AWARE_LIVE",
        expected_written=None,
        operator_approval=approval,
        analysis_date="2026-05-13",
        delay_ms=0,
        session_factory=lambda: FakeSession(events),
        preflight_func=lambda session: (True, ""),
        fetch_lead_func=fetcher,
        send_update_func=updater,
    )
    report = json.loads((out / "live_write_report.json").read_text(encoding="utf-8"))

    assert code == 0
    assert "patch" not in events
    assert report["rows"][0]["status"] == "skipped"
    assert "clobber_protected" in report["rows"][0]["reason"]


def test_live_write_skips_patch_when_snapshot_fails(tmp_path: Path) -> None:
    input_csv, stage5_summary, field_catalog, approval, out = _live_files(tmp_path)
    events: list[str] = []

    def snapshot_writer(path: Path, rows: list[dict[str, object]]) -> None:
        events.append("snapshot")
        raise OSError("disk full")

    code = run_live_write(
        input_csv=input_csv,
        stage5_summary=stage5_summary,
        field_catalog_cache=field_catalog,
        out_root=out,
        live_confirmation="WRITE_AMO_DEAL_AWARE_LIVE",
        expected_written=1,
        operator_approval=approval,
        analysis_date="2026-05-13",
        delay_ms=0,
        session_factory=lambda: FakeSession(events),
        preflight_func=lambda session: (True, ""),
        fetch_lead_func=lambda session, *, lead_id: _lead_with_values({field: "old" for field in DEAL_AI_FIELDS}),
        send_update_func=lambda session, *, lead_id, field_payload: events.append("patch") or {"updated_fields": []},
        snapshot_writer=snapshot_writer,
    )
    report = json.loads((out / "live_write_report.json").read_text(encoding="utf-8"))

    assert code == 1
    assert "patch" not in events
    assert report["rows"][0]["status"] == "failed"
    assert report["rows"][0]["reason"].startswith("snapshot_failed")


def test_snapshot_rows_are_written_to_jsonl_and_csv(tmp_path: Path) -> None:
    input_csv, stage5_summary, field_catalog, approval, out = _live_files(tmp_path)
    code = run_live_write(
        input_csv=input_csv,
        stage5_summary=stage5_summary,
        field_catalog_cache=field_catalog,
        out_root=out,
        live_confirmation="WRITE_AMO_DEAL_AWARE_LIVE",
        expected_written=1,
        operator_approval=approval,
        analysis_date="2026-05-13",
        delay_ms=0,
        session_factory=lambda: FakeSession([]),
        preflight_func=lambda session: (True, ""),
        fetch_lead_func=lambda session, *, lead_id: _lead_with_values({field: "old" for field in DEAL_AI_FIELDS}),
        send_update_func=lambda session, *, lead_id, field_payload: {"updated_fields": sorted(field_payload)},
    )

    assert code == 0
    assert (out / "pre_write_snapshot.jsonl").exists()
    assert (out / "pre_write_snapshot.csv").exists()
    assert (out / "rollback_manifest.json").exists()
    assert len(load_snapshot_rows(out / "pre_write_snapshot.jsonl")) == len(DEAL_AI_FIELDS)


def test_live_write_uses_delay_after_successful_patch(tmp_path: Path) -> None:
    input_csv, stage5_summary, field_catalog, approval, out = _live_files(tmp_path)
    sleeps: list[float] = []

    code = run_live_write(
        input_csv=input_csv,
        stage5_summary=stage5_summary,
        field_catalog_cache=field_catalog,
        out_root=out,
        live_confirmation="WRITE_AMO_DEAL_AWARE_LIVE",
        expected_written=1,
        operator_approval=approval,
        analysis_date="2026-05-13",
        batch_size=1,
        delay_ms=250,
        sleep_func=sleeps.append,
        session_factory=lambda: FakeSession([]),
        preflight_func=lambda session: (True, ""),
        fetch_lead_func=lambda session, *, lead_id: _lead_with_values({field: "old" for field in DEAL_AI_FIELDS}),
        send_update_func=lambda session, *, lead_id, field_payload: {"updated_fields": sorted(field_payload)},
    )

    assert code == 0
    assert sleeps == [0.25]


def test_rollback_dry_run_restores_only_unchanged_current_values() -> None:
    snapshot = {
        "lead_id": "123",
        "field_name": "AI-сводка по сделке",
        "field_id": "1000",
        "old_value": "старая сводка",
        "new_value": "новая сводка",
    }
    decision = rollback_decision(snapshot, _lead_with_values({"AI-сводка по сделке": "новая сводка"}))

    assert decision["rollback_status"] == "dry_run_ready"
    assert decision["old_value"] == "старая сводка"


def test_rollback_blocks_field_changed_by_manager() -> None:
    snapshot = {
        "lead_id": "123",
        "field_name": "AI-сводка по сделке",
        "old_value": "старая сводка",
        "new_value": "новая сводка",
    }
    decision = rollback_decision(snapshot, _lead_with_values({"AI-сводка по сделке": "ручная правка"}))

    assert decision["rollback_status"] == "skipped"
    assert decision["reason"] == "current_value_changed_after_write"


def test_rollback_empty_old_value_requires_manual_restore() -> None:
    snapshot = {
        "lead_id": "123",
        "field_name": "AI-сводка по сделке",
        "old_value": "",
        "new_value": "новая сводка",
    }
    decision = rollback_decision(snapshot, _lead_with_values({"AI-сводка по сделке": "новая сводка"}))

    assert decision["rollback_status"] == "manual_restore_required"


def test_rollback_apply_requires_separate_confirmation() -> None:
    with pytest.raises(ValueError, match=ROLLBACK_CONFIRMATION):
        run_rollback(
            snapshot_rows=[{"lead_id": "123", "field_name": "AI-сводка по сделке", "old_value": "old", "new_value": "new"}],
            fetch_lead=lambda lead_id: _lead_with_values({"AI-сводка по сделке": "new"}),
            send_update=lambda **kwargs: {},
            apply=True,
            confirmation="WRITE_AMO_DEAL_AWARE_LIVE",
        )


def test_rollback_apply_touches_only_snapshot_field() -> None:
    calls: list[dict[str, object]] = []
    rows = run_rollback(
        snapshot_rows=[{"lead_id": "123", "field_name": "AI-сводка по сделке", "old_value": "old", "new_value": "new"}],
        fetch_lead=lambda lead_id: _lead_with_values({"AI-сводка по сделке": "new", "AI-приоритет сделки": "hot"}),
        send_update=lambda **kwargs: calls.append(kwargs) or {"ok": True},
        apply=True,
        confirmation=ROLLBACK_CONFIRMATION,
        retry_policy=RetryPolicy(max_retries=0, delay_ms=0, sleep_func=lambda seconds: None),
    )

    assert rows[0]["rollback_status"] == "restored"
    assert calls == [{"lead_id": 123, "field_payload": {"AI-сводка по сделке": "old"}}]


def test_rollback_apply_can_restore_contact_snapshot() -> None:
    calls: list[dict[str, object]] = []
    snapshot = {
        "entity_type": "contact",
        "entity_id": "777",
        "lead_id": "777",
        "field_name": "Авто история общения",
        "old_value": "old",
        "new_value": "new",
    }

    rows = run_rollback(
        snapshot_rows=[snapshot],
        fetch_lead=lambda lead_id: pytest.fail("lead fetch must not be used for contact snapshot"),
        fetch_entity=lambda entity_type, entity_id: _lead_with_values({"Авто история общения": "new"}),
        send_entity_update=lambda **kwargs: calls.append(kwargs) or {"ok": True},
        apply=True,
        confirmation=ROLLBACK_CONFIRMATION,
        retry_policy=RetryPolicy(max_retries=0, delay_ms=0, sleep_func=lambda seconds: None),
    )

    assert rows[0]["rollback_status"] == "restored"
    assert calls == [{"entity_type": "contact", "entity_id": 777, "field_payload": {"Авто история общения": "old"}}]


def test_retry_retries_429_and_5xx() -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise FakeAmoError("temporary", status_code=429)
        return "ok"

    result = call_with_retries(flaky, retry_policy=RetryPolicy(max_retries=3, delay_ms=100, sleep_func=sleeps.append))

    assert result == "ok"
    assert attempts["count"] == 3
    assert sleeps == [0.1, 0.2]


def test_retry_retries_5xx_limited_times() -> None:
    attempts = {"count": 0}
    sleeps: list[float] = []

    def server_error() -> str:
        attempts["count"] += 1
        raise FakeAmoError("server unavailable", status_code=503)

    with pytest.raises(FakeAmoError):
        call_with_retries(server_error, retry_policy=RetryPolicy(max_retries=2, delay_ms=100, sleep_func=sleeps.append))

    assert attempts["count"] == 3
    assert sleeps == [0.1, 0.2]


def test_retry_does_not_retry_permanent_4xx() -> None:
    attempts = {"count": 0}

    def bad_request() -> str:
        attempts["count"] += 1
        raise FakeAmoError("bad request", status_code=400)

    with pytest.raises(FakeAmoError):
        call_with_retries(bad_request, retry_policy=RetryPolicy(max_retries=3, delay_ms=100, sleep_func=lambda seconds: None))

    assert attempts["count"] == 1


def test_rollback_resume_skips_already_successful_rows() -> None:
    snapshot = {"lead_id": "123", "field_name": "AI-сводка по сделке", "old_value": "old", "new_value": "new"}
    key = "123|AI-сводка по сделке|"
    rows = run_rollback(
        snapshot_rows=[snapshot],
        fetch_lead=lambda lead_id: pytest.fail("fetch must be skipped"),
        resume_success_keys={key},
    )

    assert rows[0]["rollback_status"] == "skipped"
    assert rows[0]["reason"] == "resume_success_already_processed"


def test_rollback_resume_does_not_treat_dry_run_ready_as_success(tmp_path: Path) -> None:
    report = tmp_path / "rollback_dry_run_report.json"
    report.write_text(
        json.dumps(
            {
                "rows": [
                    {"snapshot_key": "dry-run-key", "rollback_status": "dry_run_ready"},
                    {"snapshot_key": "restored-key", "rollback_status": "restored"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert load_successful_rollback_keys(report) == {"restored-key"}


def test_rollback_resume_state_ignores_legacy_state_without_success_statuses(tmp_path: Path) -> None:
    state = tmp_path / "rollback_resume_state.json"
    state.write_text(json.dumps({"successful_keys": ["legacy-dry-run-key"]}), encoding="utf-8")

    assert load_successful_rollback_keys(state) == set()


def test_rollback_resume_state_records_only_restored_rows(tmp_path: Path) -> None:
    rows = [
        {"snapshot_key": "dry-run-key", "rollback_status": "dry_run_ready"},
        {"snapshot_key": "restored-key", "rollback_status": "restored"},
    ]
    summary = rollback_summary(rows=rows, snapshot_path=tmp_path / "snapshot.jsonl", apply=True, max_rollback_rows=None)

    write_rollback_outputs(tmp_path, rows=rows, summary=summary, apply=True)

    state_payload = json.loads((tmp_path / "rollback_resume_state.json").read_text(encoding="utf-8"))
    assert state_payload["successful_statuses"] == ["restored"]
    assert state_payload["successful_keys"] == ["restored-key"]
    assert load_successful_rollback_keys(tmp_path / "rollback_resume_state.json") == {"restored-key"}


def test_rollback_apply_after_dry_run_report_still_restores(tmp_path: Path) -> None:
    snapshot = {"lead_id": "123", "field_name": "AI-сводка по сделке", "old_value": "old", "new_value": "new"}
    dry_run_report = tmp_path / "rollback_dry_run_report.json"
    dry_run_report.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "snapshot_key": "123|AI-сводка по сделке|",
                        "rollback_status": "dry_run_ready",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    rows = run_rollback(
        snapshot_rows=[snapshot],
        fetch_lead=lambda lead_id: _lead_with_values({"AI-сводка по сделке": "new"}),
        send_update=lambda **kwargs: calls.append(kwargs) or {"ok": True},
        apply=True,
        confirmation=ROLLBACK_CONFIRMATION,
        retry_policy=RetryPolicy(max_retries=0, delay_ms=0, sleep_func=lambda seconds: None),
        resume_success_keys=load_successful_rollback_keys(dry_run_report),
    )

    assert rows[0]["rollback_status"] == "restored"
    assert calls == [{"lead_id": 123, "field_payload": {"AI-сводка по сделке": "old"}}]
