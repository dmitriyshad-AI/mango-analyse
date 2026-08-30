from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_strict_s100_current.py"
SPEC = importlib.util.spec_from_file_location("audit_strict_s100_current", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _report(*, strict_ready: bool = True, foreign_events: int = 0) -> dict:
    return {
        "database": {"unchanged": True, "sidecars_absent": True},
        "summary": {
            "context_uniqueness": {"duplicate_chunk_event_pairs": 0},
            "checks": {
                "formal": {
                    "row_count_exact": True, "customer_hashes_unique": True,
                    "status_values_supported": True, "field_states_explicit": True,
                    "canaries_passed": True,
                },
                "data": {
                    "all_customers_found": True, "all_context_chunks_owner_exact": True,
                    "source_denominators_closed": True, "database_unchanged": True,
                },
                "semantic": {
                    "no_duplicate_chunk_event_pairs": True,
                    "all_dossier_rows_provenance_exact": True,
                    "structured_responsible_and_due_provenance_valid": True,
                    "closed_pairs_valid": True, "empty_dispositions_explicit": True,
                    "storage_and_dossier_visibility_separated": True,
                    "active_deals_and_attendance_projection_exact": True,
                    "calls_absent_from_bot_context": True,
                },
                "runtime": {
                    "query_only_immutable": True, "database_unchanged": True,
                    "sidecars_absent": True, "external_writes_zero": True,
                },
            },
        },
        "rows": [
            {
                "customer_sha256": f"{index:064x}",
                "found": True,
                "action_status": "active" if index == 0 else "empty",
                "fields": {
                    **{
                        name: {"state": "absent", "count": 0}
                        for name in ("family", "money", "signals", "objections", "chronology")
                    },
                    **{
                        name: {
                            "projection_matches_storage": True,
                            "dossier_displayed": {"state": "absent", "count": 0},
                        }
                        for name in ("active_deals", "attendance")
                    },
                },
                "strict_validation": {
                    "ready": strict_ready if index == 0 else True,
                    "validator_facts": {"product_readiness_ready": index == 0},
                },
                "context": {
                    "owner_validation": {"foreign_event_owners": foreign_events if index == 0 else 0},
                },
            }
            for index in range(100)
        ],
    }


SAFETY = {
    "network_denied": True,
    "network_attempts": 0,
    "unexpected_subprocesses": 0,
    "adapter_unchanged": True,
    "database_unchanged": True,
}


def test_current_regrade_ignores_obsolete_ready_quota_but_rejects_false_ready() -> None:
    passed = MODULE._current_regrade(_report(), SAFETY)
    failed = MODULE._current_regrade(_report(strict_ready=False), SAFETY)

    assert passed["product_ready_count"] == 1
    assert passed["false_ready_count"] == 0
    assert passed["verdict"] == "PASS"
    assert passed["verdicts"]["business"] == "NOT_EVALUATED_REQUIRES_30_HUMAN_DOSSIERS"
    assert failed["false_ready_count"] == 1
    assert failed["verdict"] == "FAIL"


def test_current_regrade_rejects_foreign_context_owner() -> None:
    result = MODULE._current_regrade(_report(foreign_events=1), SAFETY)

    assert result["critical_error_reason_counts"] == {"foreign_event_owners": 1}
    assert result["verdicts"]["data"] == "FAIL"
    assert result["verdict"] == "FAIL"


def test_current_regrade_preserves_upstream_semantic_failure() -> None:
    report = _report()
    report["summary"]["checks"]["semantic"]["closed_pairs_valid"] = False

    result = MODULE._current_regrade(report, SAFETY)

    assert result["checks"]["semantic"]["closed_pairs_valid"] is False
    assert result["verdicts"]["semantic"] == "FAIL"


def test_compatibility_keeps_current_eventless_pairs_and_sql_functions(tmp_path: Path) -> None:
    harness = SimpleNamespace(
        _active_deals_field=object(),
        _expected_dossier_sources=lambda _con, **_kwargs: {},
        _open_immutable_read_api=object(),
    )
    restore = MODULE._install_compatibility(harness, SCRIPT.parents[1])

    from mango_mvp.customer_timeline import freshness

    try:
        assert freshness.EVENTLESS_BOT_CONTEXT_ALLOWED_PAIRS == {
            ("customer_timeline_bot_safe_summary", "bot_safe_summary"),
            ("customer_purchases_v1", "purchase_history"),
        }
        db = tmp_path / "immutable.sqlite"
        sqlite3.connect(db).close()

        class Store:
            def __init__(self, db_path: Path, **_kwargs: object) -> None:
                self.db_path = db_path

        class Api:
            def __init__(self, store: Store) -> None:
                self.store = store

        api = harness._open_immutable_read_api(Store, Api, db)
        con = api.store._connect()
        try:
            assert con.execute("PRAGMA query_only").fetchone()[0] == 1
            assert con.execute(
                "SELECT mango_tz_at_or_before('2026-08-29T00:00:00+00:00',"
                "'2026-08-30T00:00:00+00:00')"
            ).fetchone()[0] == 1
            assert con.execute("SELECT _mango_timeline_record_digest('{}')").fetchone()[0]
        finally:
            con.close()
    finally:
        restore()


def test_compatibility_keeps_frozen_source_oracle_canaries_green() -> None:
    raw_path = os.environ.get("CUSTOMER_TIMELINE_FROZEN_S100_HARNESS")
    if not raw_path:
        pytest.skip("frozen release harness is supplied only by the release job")
    harness_path = Path(raw_path).expanduser().resolve(strict=True)
    assert MODULE._sha_file(harness_path) == MODULE.HARNESS_SHA256
    harness = MODULE._load(harness_path)
    restore = MODULE._install_compatibility(harness, SCRIPT.parents[1])
    try:
        canaries = harness._fixture_canaries(
            MODULE.datetime.fromisoformat("2026-08-29T13:47:04.071151+00:00")
        )
        assert canaries["passed"] is True
        canary = canaries["dossier_source_provenance"]["storage_oracle"]
        assert canary["baseline_accepted"] is True
        assert canary["foreign_rows_rejected"] is True
        assert canary["future_rows_rejected"] is True
        assert canary["mutation_rejected"] is True
    finally:
        restore()


def test_identity_conflict_allows_only_zero_business_projection() -> None:
    report = _report()
    row = report["rows"][0]
    row["reason_code"] = "identity_conflict_open"
    row["fields"] = {
        name: {
            "state": "conflict", "reason_code": "identity_conflict_open",
            "count": 0, "projection_matches_storage": False,
        }
        for name in ("family", "money", "signals", "objections", "chronology")
    }
    row["fields"].update({
        name: {
            "projection_matches_storage": False,
            "dossier_displayed": {
                "state": "conflict", "reason_code": "identity_conflict_open", "count": 0,
            },
        }
        for name in ("active_deals", "attendance")
    })

    assert MODULE._current_regrade(report, SAFETY)["verdict"] == "PASS"
    row["fields"]["money"]["count"] = 1
    assert MODULE._current_regrade(report, SAFETY)["verdict"] == "FAIL"

    row["fields"] = {}
    assert MODULE._current_regrade(report, SAFETY)["verdict"] == "FAIL"


def test_current_regrade_rejects_database_sha_change() -> None:
    safety = {**SAFETY, "database_unchanged": False}

    result = MODULE._current_regrade(_report(), safety)

    assert result["checks"]["runtime"]["database_unchanged"] is False
    assert result["verdicts"]["runtime"] == "FAIL"
    assert result["verdict"] == "FAIL"


def test_current_report_body_removes_contradictory_legacy_summary() -> None:
    report = _report()
    report["summary"]["overall_go"] = False
    report["summary"]["verdicts"] = {"semantic": "FAIL", "business": "FAIL"}

    result = MODULE._current_report_body(report, SAFETY)

    assert result["schema_version"].endswith("_v2")
    assert "summary" not in result
    assert result["s100_current_tz_verdict"]["verdict"] == "PASS"
