from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping, Optional

import pytest

from scripts.run_telegram_stage6_kb_eval import (
    run_stage6_kb_eval,
    stage6_kb_eval_safety_contract,
)


class RecordingProvider:
    def __init__(self) -> None:
        self.contexts: list[Mapping[str, Any]] = []

    def build_draft(self, client_message: str, *, context: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        self.contexts.append(dict(context or {}))
        return {
            "message_type": "question",
            "broad_group": "commercial",
            "topic_id": "theme:001_pricing",
            "confidence_theme": 0.9,
            "confidence_group": 0.9,
            "route": "draft_for_manager",
            "draft_text": "Здравствуйте! Менеджер сверит условия по выбранной программе и подскажет следующий шаг.",
            "manager_checklist": ["Проверить программу клиента"],
            "missing_facts": [],
            "safety_flags": ["manager_approval_required", "no_auto_send"],
            "context_used": ["knowledge_snippets"],
            "context_warnings": [],
        }


def test_stage6_kb_eval_writes_enriched_outputs_and_passes_snapshot_context(tmp_path: Path) -> None:
    input_path = tmp_path / "private_dialog_threads.jsonl"
    snapshot_path = tmp_path / "kc_snapshot.json"
    baseline_path = tmp_path / "baseline.csv"
    out_dir = tmp_path / "out"
    write_jsonl(input_path, [dialog_record("d1", "Какая цена курса и можно оплатить частями?"), dialog_record("d2", "Когда будет ссылка на занятие?")])
    write_snapshot(snapshot_path)
    write_baseline_csv(baseline_path, [("d1", "m2"), ("d2", "m2")])
    provider = RecordingProvider()

    result = run_stage6_kb_eval(
        input_path=input_path,
        snapshot_path=snapshot_path,
        out_dir=out_dir,
        baseline_csv_path=baseline_path,
        provider_mode="fake",
        provider=provider,
        expected_dialogs=2,
    )

    assert result.rows_total == 2
    assert result.used_kb_context == 2
    assert Path(result.enriched_csv_path).exists()
    assert Path(result.enriched_xlsx_path).exists()
    assert Path(result.comparison_csv_path).exists()
    assert Path(result.comparison_summary_path).exists()
    with zipfile.ZipFile(result.enriched_xlsx_path) as xlsx:
        assert "xl/workbook.xml" in xlsx.namelist()

    rows = read_csv(Path(result.enriched_csv_path))
    assert rows[0]["snapshot_run_id"] == "20260517_night_v1"
    assert rows[0]["selected_chunk_count"] == "2"
    assert rows[0]["used_kb_context"] == "True"
    assert "Стоимость и оплата" in rows[0]["knowledge_snippets"]
    assert provider.contexts[0]["facts_context"]["snapshot_run_id"] == "20260517_night_v1"
    assert provider.contexts[0]["knowledge_snippets"]
    assert provider.contexts[0]["pilot_context_safety"]["send_client_message"] is False


def test_stage6_kb_eval_fake_provider_builds_comparison_summary(tmp_path: Path) -> None:
    input_path = tmp_path / "private_dialog_threads.jsonl"
    snapshot_path = tmp_path / "kc_snapshot.json"
    baseline_path = tmp_path / "baseline.csv"
    out_dir = tmp_path / "out"
    write_jsonl(input_path, [dialog_record("d1", "Подскажите, какие курсы входят в программу?")])
    write_snapshot(snapshot_path)
    write_baseline_csv(baseline_path, [("d1", "m2")], draft_text="Здравствуйте! Уточним и вернемся.")

    result = run_stage6_kb_eval(
        input_path=input_path,
        snapshot_path=snapshot_path,
        out_dir=out_dir,
        baseline_csv_path=baseline_path,
        provider_mode="fake",
        expected_dialogs=1,
    )

    comparison = read_csv(Path(result.comparison_csv_path))
    summary = Path(result.comparison_summary_path).read_text(encoding="utf-8")

    assert comparison[0]["draft_became_more_substantive"] == "True"
    assert comparison[0]["empty_clarification_reduced"] == "True"
    assert "client_send: false" in summary
    assert "write_stable_runtime: false" in summary
    assert result.became_more_substantive == 1


def test_stage6_kb_eval_requires_fixed_sample_size(tmp_path: Path) -> None:
    input_path = tmp_path / "private_dialog_threads.jsonl"
    snapshot_path = tmp_path / "kc_snapshot.json"
    baseline_path = tmp_path / "baseline.csv"
    write_jsonl(input_path, [dialog_record("d1", "Какая цена?")])
    write_snapshot(snapshot_path)
    write_baseline_csv(baseline_path, [("d1", "m2")])

    with pytest.raises(ValueError, match="fixed sample"):
        run_stage6_kb_eval(
            input_path=input_path,
            snapshot_path=snapshot_path,
            out_dir=tmp_path / "out",
            baseline_csv_path=baseline_path,
            provider_mode="fake",
            expected_dialogs=20,
        )


def test_stage6_kb_eval_blocks_stable_runtime_output(tmp_path: Path) -> None:
    input_path = tmp_path / "private_dialog_threads.jsonl"
    snapshot_path = tmp_path / "kc_snapshot.json"
    baseline_path = tmp_path / "baseline.csv"
    write_jsonl(input_path, [dialog_record("d1", "Какая цена?")])
    write_snapshot(snapshot_path)
    write_baseline_csv(baseline_path, [("d1", "m2")])

    with pytest.raises(ValueError, match="stable_runtime"):
        run_stage6_kb_eval(
            input_path=input_path,
            snapshot_path=snapshot_path,
            out_dir=Path("stable_runtime") / "stage6_kb_eval",
            baseline_csv_path=baseline_path,
            provider_mode="fake",
            expected_dialogs=1,
        )


def test_stage6_kb_eval_safety_contract_is_read_only() -> None:
    safety = stage6_kb_eval_safety_contract()

    assert safety["live_telegram"] is False
    assert safety["client_send"] is False
    assert safety["write_crm"] is False
    assert safety["write_tallanto"] is False
    assert safety["write_stable_runtime"] is False
    assert safety["run_asr"] is False
    assert safety["run_resolve_analyze"] is False


def dialog_record(dialog_id: str, target_text: str) -> dict[str, Any]:
    return {
        "schema_version": "telegram_pilot_eval_pack_v1",
        "run_id": "run_01",
        "dialog_id": dialog_id,
        "message_count": 2,
        "client_message_count": 1,
        "manager_message_count": 1,
        "topic_ids": [],
        "manager_only": False,
        "useful_feedback": False,
        "messages": [
            {
                "message_id": "m1",
                "date": "2026-05-17T09:00:00+00:00",
                "direction": "manager",
                "text": "Здравствуйте, чем помочь?",
                "has_media": False,
            },
            {
                "message_id": "m2",
                "date": "2026-05-17T09:01:00+00:00",
                "direction": "client",
                "text": target_text,
                "has_media": False,
            },
        ],
    }


def write_snapshot(path: Path) -> None:
    payload = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "20260517_night_v1",
        "sources": [
            {
                "source_id": "source:price",
                "title": "Прайс 2026/2027",
                "source_kind": "local_docx",
                "path": "snapshot://source:price",
                "fact_types": ["price", "payment_methods"],
                "freshness_status": "fresh_verified",
                "usable_for_precise_answer": True,
            },
            {
                "source_id": "source:schedule",
                "title": "Расписание",
                "source_kind": "local_docx",
                "path": "snapshot://source:schedule",
                "fact_types": ["schedule"],
                "freshness_status": "fresh_verified",
                "usable_for_precise_answer": True,
            },
        ],
        "chunks": [
            {
                "chunk_id": "chunk:price",
                "source_id": "source:price",
                "title": "Стоимость и оплата",
                "text": "Для вопросов о цене менеджер сверяет программу, класс и формат обучения перед ответом.",
                "fact_types": ["price", "payment_methods"],
                "freshness_status": "fresh_verified",
            },
            {
                "chunk_id": "chunk:schedule",
                "source_id": "source:schedule",
                "title": "Расписание и доступ",
                "text": "По расписанию и ссылкам менеджер проверяет конкретную группу и личный кабинет клиента.",
                "fact_types": ["schedule"],
                "freshness_status": "fresh_verified",
            },
        ],
        "manager_answer_patterns": [
            {
                "pattern_id": "pattern:next_step",
                "related_theme_ids": ["theme:001_pricing"],
                "safe_pattern": "Сначала подтвердить программу и класс, затем дать клиенту понятный следующий шаг.",
            }
        ],
        "safety": {
            "crm_write": False,
            "tallanto_write": False,
            "client_send": False,
            "stable_runtime_write": False,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_baseline_csv(path: Path, rows: list[tuple[str, str]], *, draft_text: str = "Здравствуйте! Уточним и вернемся.") -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["dialog_id", "target_message_id", "topic_id", "route", "draft_text"],
        )
        writer.writeheader()
        for dialog_id, message_id in rows:
            writer.writerow(
                {
                    "dialog_id": dialog_id,
                    "target_message_id": message_id,
                    "topic_id": "theme:001_pricing",
                    "route": "draft_for_manager",
                    "draft_text": draft_text,
                }
            )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))
