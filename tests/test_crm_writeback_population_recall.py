from __future__ import annotations

from pathlib import Path

from mango_mvp.quality.crm_writeback_population_recall import (
    scan_crm_writeback_population_recall,
    write_population_recall_outputs,
)


TEXT_FIELDS = ("Краткая история общения",)


def test_population_counter_counts_unique_rows_not_matches() -> None:
    rows = [
        {
            "Телефон клиента": "+79990000000",
            "Краткая история общения": "Нецелевой звонок. Повторно: нецелевое обращение.",
        }
    ]

    result = scan_crm_writeback_population_recall(rows, text_fields=TEXT_FIELDS)

    assert result["summary"]["class_marker_prevalence_rows"] == 1
    assert result["summary"]["by_marker"]["hp_self_label_necelevoi"] >= 1


def test_population_counter_reports_detector_gap_for_high_precision_marker() -> None:
    rows = [
        {
            "Телефон клиента": "+79990000001",
            "Краткая история общения": "Клиент сообщил, что заявка не относится к учебному центру.",
        }
    ]

    result = scan_crm_writeback_population_recall(rows, text_fields=TEXT_FIELDS)

    assert result["summary"]["high_precision_marker_uncovered_rows"] == 0
    assert result["summary"]["passed_for_live"] is True


def test_population_counter_tracks_wrong_person_identity_mismatch() -> None:
    rows = [
        {
            "Телефон клиента": "+79990000005",
            "Краткая история общения": "Контакт не подтвердился: на линии была не та Светлана, обсуждение программы, интереса к продукту и следующих шагов не состоялось.",
        }
    ]

    result = scan_crm_writeback_population_recall(rows, text_fields=TEXT_FIELDS)

    assert result["summary"]["by_marker"]["hp_wrong_person_or_identity_mismatch"] >= 1
    assert result["summary"]["high_precision_marker_uncovered_rows"] == 0
    assert result["summary"]["passed_for_live"] is True


def test_population_counter_review_markers_do_not_block_live_by_themselves() -> None:
    rows = [
        {
            "Телефон клиента": "+79990000002",
            "Краткая история общения": "Обучение не обсуждалось подробно, но клиент просил перезвонить по заявке.",
        }
    ]

    result = scan_crm_writeback_population_recall(rows, text_fields=TEXT_FIELDS)

    assert result["summary"]["review_marker_uncovered_rows"] == 1
    assert result["summary"]["passed_for_live"] is True


def test_population_counter_does_not_flag_mts_link_as_carrier() -> None:
    rows = [
        {
            "Телефон клиента": "+79990000003",
            "Краткая история общения": "Клиент обсуждал онлайн-занятия через МТС Линк и расписание курса по физике.",
        }
    ]

    result = scan_crm_writeback_population_recall(rows, text_fields=TEXT_FIELDS)

    assert result["summary"]["class_marker_prevalence_rows"] == 0


def test_population_counter_writes_outputs(tmp_path: Path) -> None:
    result = scan_crm_writeback_population_recall(
        [{"Телефон клиента": "+79990000004", "Краткая история общения": "Нецелевой звонок."}],
        text_fields=TEXT_FIELDS,
    )

    outputs = write_population_recall_outputs(tmp_path, result)

    assert Path(outputs["summary_json"]).exists()
    assert Path(outputs["marker_hits_csv"]).exists()
    assert Path(outputs["marker_uncovered_csv"]).exists()
