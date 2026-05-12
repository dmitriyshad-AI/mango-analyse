from __future__ import annotations

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
