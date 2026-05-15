from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from scripts import run_crm_writeback_quality_gate as gate
from scripts.run_crm_writeback_quality_gate import _c12_overlap_summary, _field_level_findings, _metadata_findings


def test_quality_gate_blocks_truncated_crm_text() -> None:
    row = {
        "Краткая история общения": "Клиент интересуется курсом.",
        "Краткое резюме последнего свежего звонка": "Резюме обрывается...",
    }

    findings = _field_level_findings(row)

    assert {finding.risk_type for finding in findings} == {"truncated_crm_text"}


def test_quality_gate_blocks_internal_ellipsis_not_only_suffix() -> None:
    row = {
        "Краткая история общения": "Клиент интересуется курсом.",
        "Хронология общения (последние 5 касаний)": "01.05.2026: менеджер начал говорить и не... затем строка продолжилась",
    }

    findings = _field_level_findings(row)

    assert {finding.risk_type for finding in findings} >= {"truncated_crm_text"}


def test_quality_gate_blocks_empty_short_history() -> None:
    findings = _field_level_findings({"Краткая история общения": ""})

    assert {finding.risk_type for finding in findings} >= {"empty_crm_history"}


def test_quality_gate_blocks_service_context_metadata_for_live_ready() -> None:
    findings = _metadata_findings(
        {
            "Тип последнего свежего звонка": "service_call",
            "AMO contact IDs": "123",
            "CRM writeback policy": "live_update_ready",
        }
    )

    assert {finding.risk_type for finding in findings} == {"service_or_existing_client_live_writeback"}


def test_quality_gate_blocks_orphan_or_ambiguous_amo_contact_metadata() -> None:
    findings = _metadata_findings(
        {
            "Тип последнего свежего звонка": "sales_call",
            "AMO contact IDs": "",
        }
    )

    assert {finding.risk_type for finding in findings} >= {"amo_orphan_or_ambiguous_contact"}


def test_quality_gate_reports_c12_history_overlap_as_soft_counter() -> None:
    summary = _c12_overlap_summary(
        [
            {
                "Краткое резюме последнего свежего звонка": "Клиент интересуется летней школой по физике и просит перезвонить.",
                "Краткая история общения": "Клиент интересуется летней школой по физике и просит перезвонить.",
                "Хронология общения (последние 5 касаний)": "Клиент интересуется летней школой по физике и просит перезвонить.",
            }
        ]
    )

    assert summary["summary_history_overlap_rows"] == 1
    assert summary["history_chronology_overlap_rows"] == 1
    assert summary["blocking"] is False


def _write_gate_input(path: Path) -> None:
    rows = [
        {
            "Телефон клиента": "+79990000000",
            "Краткое резюме последнего свежего звонка": "Клиент интересуется курсом.",
            "Краткая история общения": "Клиент интересуется курсом и ожидает звонок менеджера.",
            "Хронология общения (последние 5 касаний)": "01.05.2026: клиент оставил заявку.",
            "Возражения": "",
            "Следующий шаг": "Менеджеру перезвонить клиенту.",
            "История общения Tallanto": "",
            "Тип последнего свежего звонка": "sales_call",
            "AMO contact IDs": "123",
            "CRM writeback policy": "live_update_ready",
        }
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_tenant_config(path: Path) -> None:
    payload = {
        "schema_version": "tenant_config_v1",
        "tenant_id": "foton",
        "business": {"industry": "edtech"},
        "crm": {"target_fields": ["Авто история общения"], "protected_fields": ["Id Tallanto"]},
        "privacy": {"phone_in_ai_text": "redact"},
        "quality": {"crm_detector_min_severity": "P2"},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _patch_gate_to_pass_business_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "detect_crm_writeback_quality_risks", lambda *args, **kwargs: [])
    monkeypatch.setattr(gate, "detect_crm_text_quality_risks", lambda *args, **kwargs: [])
    monkeypatch.setattr(gate, "detect_crm_text_quality_batch_risks", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        gate,
        "scan_crm_writeback_population_recall",
        lambda *args, **kwargs: {"summary": {"passed_for_live": True}},
    )
    monkeypatch.setattr(gate, "write_population_recall_outputs", lambda *args, **kwargs: {})


def test_quality_gate_warns_on_tenant_config_pin_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gate_to_pass_business_checks(monkeypatch)
    input_csv = tmp_path / "input.csv"
    tenant_config = tmp_path / "tenant_config_v1.json"
    out_root = tmp_path / "warn"
    _write_gate_input(input_csv)
    _write_tenant_config(tenant_config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_crm_writeback_quality_gate.py",
            "--input",
            str(input_csv),
            "--out-root",
            str(out_root),
            "--tenant-config",
            str(tenant_config),
            "--tenant-config-pin-mode",
            "warn",
        ],
    )

    exit_code = gate.main()
    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert summary["passed"] is True
    assert summary["tenant_config_pin_mode"] == "warn"
    assert summary["tenant_config_pin"]["passed"] is False
    assert summary["tenant_config_pin"]["reason"] == "tenant_config_sha256_mismatch"


def test_quality_gate_strict_blocks_tenant_config_pin_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_gate_to_pass_business_checks(monkeypatch)
    input_csv = tmp_path / "input.csv"
    tenant_config = tmp_path / "tenant_config_v1.json"
    out_root = tmp_path / "strict"
    _write_gate_input(input_csv)
    _write_tenant_config(tenant_config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_crm_writeback_quality_gate.py",
            "--input",
            str(input_csv),
            "--out-root",
            str(out_root),
            "--tenant-config",
            str(tenant_config),
            "--tenant-config-pin-mode",
            "strict",
        ],
    )

    exit_code = gate.main()
    summary = json.loads((out_root / "summary.json").read_text(encoding="utf-8"))

    assert exit_code == 1
    assert summary["passed"] is False
    assert summary["tenant_config_pin_mode"] == "strict"
    assert summary["tenant_config_pin"]["passed"] is False
    assert summary["tenant_config_pin"]["reason"] == "tenant_config_sha256_mismatch"
