from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest
from openpyxl import load_workbook


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_reconcile_stalled_deals_checks_open_deals_by_customer(tmp_path: Path) -> None:
    mod = _load_script("reconcile_stalled_deals_with_amo")
    db = _f9_db(tmp_path)
    client = FakeAmoClient({"lead-1": {"id": 1, "status_id": 142, "pipeline_id": 10, "closed_at": 123}})

    report = mod.reconcile(
        timeline_db=db,
        limit=10,
        mcp_env=tmp_path / "missing.env",
        page_limit=50,
        sleep_sec=0,
        client=client,
    )

    assert report["status"] == "checked"
    assert report["customers_selected"] == 1
    assert report["open_opportunities_checked"] == 1
    assert report["customers_changed"] == 1
    assert report["snapshot_stale"] is True
    assert report["rows"][0]["reason"] == "live_lead_closed"
    assert report["rows"][0]["lead_hash"] != "lead-1"
    assert client.calls == 1


def test_reconcile_stalled_deals_is_unavailable_without_env(tmp_path: Path) -> None:
    mod = _load_script("reconcile_stalled_deals_with_amo")
    db = _f9_db(tmp_path)

    report = mod.reconcile(
        timeline_db=db,
        limit=10,
        mcp_env=tmp_path / "missing.env",
        page_limit=50,
        sleep_sec=0,
    )

    assert report["status"] == "unavailable"
    assert report["reason"] == "mcp_env_missing"
    assert report["customers_selected"] == 1
    assert report["customers_checked"] == 0


def test_wave0_manager_lists_write_actuality_header(tmp_path: Path) -> None:
    mod = _load_script("build_wave0_manager_lists")
    db = _f9_db(tmp_path)
    reconcile = tmp_path / ".codex_local" / "reconcile.json"
    reconcile.parent.mkdir(parents=True)
    reconcile.write_text(
        json.dumps(
            {
                "status": "checked",
                "generated_at": "2026-07-03T12:00:00+00:00",
                "customers_checked": 10,
                "customers_changed": 2,
                "snapshot_stale": True,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / ".codex_local" / "wave0.xlsx"

    summary = mod.build_workbook(
        timeline_db=db,
        out_xlsx=out,
        allowed_root=tmp_path,
        reconcile_json=reconcile,
        limit_per_sheet=50,
    )

    assert summary["quick_check"] == "ok"
    assert summary["rows_by_sheet"]["Зависшие факт LTV"] == 1
    wb = load_workbook(out, read_only=True)
    ws = wb["Зависшие факт LTV"]
    assert "2 расхождений из 10" in ws["A2"].value
    assert "amocrm_snapshot=2026-07-03T04:18:25+00:00" in ws["A2"].value


def test_wave0_manager_lists_pii_output_must_stay_under_codex_local(tmp_path: Path) -> None:
    mod = _load_script("build_wave0_manager_lists")
    db = _f9_db(tmp_path)

    with pytest.raises(ValueError, match=".codex_local"):
        mod.build_workbook(
            timeline_db=db,
            out_xlsx=tmp_path / "docs" / "wave0.xlsx",
            allowed_root=tmp_path,
            reconcile_json=None,
            limit_per_sheet=50,
        )


def test_refresh_manager_views_is_off_by_default(tmp_path: Path) -> None:
    mod = _load_script("refresh_manager_views")

    report = mod.refresh_manager_views(
        timeline_db=tmp_path / "missing.sqlite",
        allowed_root=tmp_path,
        out_dir=tmp_path / ".codex_local" / "review" / "manager_views",
        reconcile_json=None,
        customer_ids_file=None,
        canonical_calls_db=None,
        limit=10,
        run=False,
    )

    assert report["status"] == "skipped_off_by_default"
    assert report["writes"] == []


def test_refresh_manager_views_run_calls_both_builders(tmp_path: Path, monkeypatch) -> None:
    mod = _load_script("refresh_manager_views")
    calls = []

    def fake_wave0(**kwargs):
        calls.append(("wave0", kwargs))
        return {"out_xlsx": str(kwargs["out_xlsx"]), "rows_by_sheet": {}}

    def fake_dossier(**kwargs):
        calls.append(("dossier", kwargs))
        return {"out_xlsx": str(kwargs["out_xlsx"]), "customers": 0}

    monkeypatch.setattr(mod, "build_wave0_workbook", fake_wave0)
    monkeypatch.setattr(mod, "build_manager_dossier_workbook", fake_dossier)

    report = mod.refresh_manager_views(
        timeline_db=tmp_path / "timeline.sqlite",
        allowed_root=tmp_path,
        out_dir=tmp_path / ".codex_local" / "review" / "manager_views",
        reconcile_json=None,
        customer_ids_file=None,
        canonical_calls_db=None,
        limit=10,
        run=True,
    )

    assert report["status"] == "built"
    assert [item[0] for item in calls] == ["wave0", "dossier"]
    assert ".codex_local" in report["wave0"]["out_xlsx"]
    assert ".codex_local" in report["dossier"]["out_xlsx"]


def test_refresh_manager_views_run_rejects_non_codex_local_out_dir_before_mkdir(tmp_path: Path) -> None:
    mod = _load_script("refresh_manager_views")
    unsafe = tmp_path / "docs" / "manager_views"

    with pytest.raises(ValueError, match=".codex_local"):
        mod.refresh_manager_views(
            timeline_db=tmp_path / "timeline.sqlite",
            allowed_root=tmp_path,
            out_dir=unsafe,
            reconcile_json=None,
            customer_ids_file=None,
            canonical_calls_db=None,
            limit=10,
            run=True,
        )

    assert not (tmp_path / "docs").exists()


class FakeAmoClient:
    def __init__(self, leads: dict[str, dict]):
        self.leads = leads
        self.calls = 0

    def amo_api_get(self, *, path, params=None, limit=50):
        self.calls += 1
        lead_id = str((params or {}).get("filter[id]") or "")
        lead = self.leads.get(lead_id)
        return {"_embedded": {"leads": [lead] if lead else []}}


def _f9_db(tmp_path: Path) -> Path:
    db = tmp_path / "timeline.sqlite"
    with sqlite3.connect(db) as con:
        con.executescript(
            """
            CREATE TABLE derived_signals (
              signal_id TEXT,
              tenant_id TEXT,
              customer_id TEXT,
              opportunity_id TEXT,
              event_id TEXT,
              signal_type TEXT,
              severity TEXT,
              status TEXT,
              expires_at TEXT,
              confidence REAL,
              requires_manager_review INTEGER,
              created_at TEXT,
              record_hash TEXT,
              record_json TEXT
            );
            CREATE TABLE customer_purchases_v1 (
              tenant_id TEXT,
              customer_id TEXT,
              period TEXT,
              money_kind TEXT,
              total_in REAL,
              total_out REAL,
              deals_cnt INTEGER,
              last_purchase_at TEXT,
              sources_json TEXT,
              computability TEXT,
              code_version TEXT
            );
            CREATE TABLE customer_identities (
              tenant_id TEXT,
              customer_id TEXT,
              identity_status TEXT,
              display_name TEXT,
              primary_phone TEXT,
              primary_email TEXT,
              record_json TEXT
            );
            CREATE TABLE customer_opportunities (
              opportunity_id TEXT,
              tenant_id TEXT,
              customer_id TEXT,
              opportunity_type TEXT,
              source_system TEXT,
              source_id TEXT,
              title TEXT,
              status TEXT,
              opened_at TEXT,
              closed_at TEXT,
              confidence REAL,
              record_hash TEXT,
              record_json TEXT
            );
            CREATE TABLE timeline_events (
              event_id TEXT,
              dedupe_key TEXT,
              tenant_id TEXT,
              customer_id TEXT,
              opportunity_id TEXT,
              event_type TEXT,
              event_at TEXT,
              source_system TEXT,
              source_id TEXT,
              source_ref TEXT,
              direction TEXT,
              match_status TEXT,
              confidence REAL,
              importance INTEGER,
              subject TEXT,
              text_preview TEXT,
              summary TEXT,
              created_at TEXT,
              record_hash TEXT,
              record_json TEXT,
              content_key TEXT,
              superseded_by TEXT
            );
            """
        )
        con.execute(
            "INSERT INTO customer_identities VALUES (?,?,?,?,?,?,?)",
            ("foton", "customer:1", "strong", "Клиент", "+70000000000", "client@example.com", "{}"),
        )
        con.execute(
            "INSERT INTO customer_purchases_v1 VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("foton", "customer:1", "all_time", "fact", 100000, 0, 1, "2026-01-01", "[]", "computed", "v1"),
        )
        con.execute(
            "INSERT INTO derived_signals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "signal:1",
                "foton",
                "customer:1",
                None,
                None,
                "deal_stalling",
                "medium",
                "active",
                None,
                0.8,
                1,
                "2026-07-01T00:00:00+00:00",
                "h",
                json.dumps({"recommended_action": "Проверить сделку", "evidence_text": "Пауза 14+ дней"}, ensure_ascii=False),
            ),
        )
        con.execute(
            "INSERT INTO derived_signals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "signal:2",
                "foton",
                "customer:1",
                None,
                None,
                "season_return_candidate",
                "medium",
                "active",
                None,
                0.8,
                1,
                "2026-07-01T00:00:00+00:00",
                "h",
                json.dumps({"recommended_action": "Вернуться к клиенту", "evidence_text": "Сезон"}, ensure_ascii=False),
            ),
        )
        con.execute(
            "INSERT INTO customer_opportunities VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("opp:1", "foton", "customer:1", "amo_deal", "amocrm_snapshot", "lead-1", "Сделка", "12345", "2026-01-01", "", 0.8, "h", "{}"),
        )
        con.execute(
            "INSERT INTO customer_opportunities VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("opp:2", "foton", "customer:1", "amo_deal", "amocrm_snapshot", "lead-closed", "Закрытая", "143", "2026-01-01", "", 0.8, "h", "{}"),
        )
        con.execute(
            "INSERT INTO timeline_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "event:1",
                "d",
                "foton",
                "customer:1",
                None,
                "amo_snapshot",
                "2026-07-03T04:18:25+00:00",
                "amocrm_snapshot",
                "1",
                "",
                "system",
                "strong",
                1.0,
                1,
                "",
                "",
                "summary",
                "2026-07-03T04:18:25+00:00",
                "h",
                "{}",
                "",
                None,
            ),
        )
    return db
