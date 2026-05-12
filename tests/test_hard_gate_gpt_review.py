from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.quality import hard_gate_gpt_review
from mango_mvp.quality.hard_gate_gpt_review import (
    HardGateGptReviewConfig,
    run_hard_gate_gpt_review,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _task(idx: int, risk: str = "critical") -> dict:
    return {
        "audit_id": f"hgate_full_{idx:06d}",
        "task_id": f"hard_gate_gpt::calls.db::{idx}",
        "risk_level": risk,
        "audit_goal": "Проверить no-live.",
        "call": {"source_filename": f"call_{idx}.mp3"},
        "current_state": {"call_type": "sales_call", "history_summary_excerpt": "old"},
        "proposed_change": {"new_call_type": "non_conversation"},
        "guardrail": {"reason_codes": ["no_live_marker"]},
        "transcript": {"full_text": "CLIENT: Абонент сейчас не может ответить. Оставьте сообщение."},
    }


def test_run_hard_gate_gpt_review_dry_run_writes_reviews(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "tasks.jsonl"
    _write_jsonl(input_jsonl, [_task(1), _task(2, "high")])

    summary = run_hard_gate_gpt_review(
        HardGateGptReviewConfig(
            input_jsonl=input_jsonl,
            out_root=tmp_path / "out",
            project_root=tmp_path,
            dry_run=True,
        )
    )

    assert summary["totals"]["input_tasks"] == 2
    assert summary["totals"]["reviews_written"] == 2
    assert summary["totals"]["dry_run_reviews"] == 2
    assert summary["counts"]["by_decision"] == {"safe_apply": 2}
    assert (tmp_path / "out" / "reviews.jsonl").read_text(encoding="utf-8").count("\n") == 2


def test_run_hard_gate_gpt_review_uses_batch_provider_and_existing_skip(tmp_path: Path, monkeypatch) -> None:
    input_jsonl = tmp_path / "tasks.jsonl"
    _write_jsonl(input_jsonl, [_task(1), _task(2), _task(3)])
    batch_sizes: list[int] = []

    def fake_call(config, tasks, project_root):
        batch_sizes.append(len(tasks))
        return {
            task["task_id"]: {
                "decision": "safe_apply" if task["audit_id"] != "hgate_full_000002" else "keep_current",
                "confidence": 0.97,
                "reason_ru": "test",
                "evidence": ["evidence"],
            }
            for task in tasks
        }

    monkeypatch.setattr(hard_gate_gpt_review, "_call_codex_batch", fake_call)

    first = run_hard_gate_gpt_review(
        HardGateGptReviewConfig(
            input_jsonl=input_jsonl,
            out_root=tmp_path / "out",
            project_root=tmp_path,
            batch_size=2,
            workers=1,
        )
    )
    second = run_hard_gate_gpt_review(
        HardGateGptReviewConfig(
            input_jsonl=input_jsonl,
            out_root=tmp_path / "out",
            project_root=tmp_path,
            batch_size=2,
            workers=1,
        )
    )

    assert batch_sizes == [2, 1]
    assert first["counts"]["by_decision"] == {"safe_apply": 2, "keep_current": 1}
    assert second["totals"]["skipped_existing"] == 3
    assert second["totals"]["provider_calls"] == 0


def test_copy_codex_file_if_fresher_overwrites_stale_copy(tmp_path: Path) -> None:
    source = tmp_path / "source_auth.json"
    target = tmp_path / "target_auth.json"
    source.write_text("fresh-token", encoding="utf-8")
    target.write_text("stale", encoding="utf-8")

    hard_gate_gpt_review._copy_codex_file_if_fresher(source, target)

    assert target.read_text(encoding="utf-8") == "fresh-token"
