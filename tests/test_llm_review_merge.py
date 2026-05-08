from __future__ import annotations

import json
from pathlib import Path

from scripts.merge_pilot_sales_moment_llm_reviews import merge_review_roots


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_merge_review_roots_deduplicates_and_reports_completion(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "input.jsonl"
    _write_jsonl(input_jsonl, [{"id": "pilot-00001"}, {"id": "pilot-00002"}])

    root_a = tmp_path / "root_a"
    root_b = tmp_path / "root_b"
    row_one = {"moment_id": "pilot-00001", "overall_quality_score": 80, "phone": "79000000001", "provider": "codex_cli"}
    row_two = {"moment_id": "pilot-00002", "overall_quality_score": 40, "phone": "79000000002", "provider": "codex_cli"}
    _write_jsonl(root_a / "reviews.jsonl", [row_one])
    _write_jsonl(root_b / "reviews.jsonl", [row_one, row_two])

    summary = merge_review_roots(
        project_root=tmp_path,
        input_jsonl=input_jsonl,
        out_root=tmp_path / "merged",
        review_roots=[root_a, root_b],
        model="gpt-5.5",
        reasoning_effort="medium",
    )

    assert summary["totals"]["input_items"] == 2
    assert summary["totals"]["reviews_written"] == 2
    assert summary["quality"]["low_quality_count_lt_55"] == 1
    assert summary["merge"]["duplicate_same"] == 1
    assert summary["merge"]["complete"] is True
    assert (tmp_path / "merged" / "reviews.csv").exists()
