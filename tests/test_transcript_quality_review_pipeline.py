from __future__ import annotations

import csv
import json
from pathlib import Path

from mango_mvp.quality.transcript_quality_claude_package import ClaudePackageConfig, build_transcript_quality_claude_package
from mango_mvp.quality.transcript_quality_consensus import ConsensusConfig, build_transcript_quality_consensus
from mango_mvp.quality.transcript_quality_escalation import EscalationConfig, build_transcript_quality_escalation
from mango_mvp.quality.transcript_quality_llm_review import (
    TranscriptQualityLLMReviewConfig,
    build_batch_review_prompt,
    extract_batch_review_payloads,
    run_transcript_quality_llm_review,
    select_review_tasks,
)
from mango_mvp.quality.transcript_quality_review_validator import ReviewValidatorConfig, validate_transcript_quality_reviews


def _task(idx: int, *, bucket: str, call_type: str, contentful: bool, text: str = "Автоответчик. Оставьте сообщение.") -> dict:
    return {
        "task_id": f"task-{idx:03d}",
        "schema": "transcript_quality_disputed_review_v1",
        "call": {
            "id": idx,
            "source_filename": f"call-{idx}.mp3",
            "started_at": "2026-01-01 10:00:00",
            "duration_sec": 30,
            "manager_name": "Manager",
            "phone": f"+7999{idx:07d}",
        },
        "guardrail": {
            "review_bucket": bucket,
            "current_call_type": call_type,
            "current_contentful": contentful,
            "label": "manual_review_probable_no_live",
            "score": -6,
            "reason_codes": "system_no_dialogue_phrase|no_live_marker",
            "should_force_non_conversation": False,
            "requires_manual_review": True,
            "protected_live_dialogue": False,
        },
        "current_analysis": {"history_summary": "", "next_step": "", "products": [], "objections": []},
        "transcript_text": text,
        "asr_variants": {},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_select_review_tasks_stratifies_by_bucket_and_call_type() -> None:
    tasks = [_task(i, bucket="llm_review_non_contentful_probable_no_live", call_type="non_conversation", contentful=False) for i in range(10)]
    tasks += [_task(100 + i, bucket="llm_review_contentful_auto_fix_conflict", call_type="service_call", contentful=True) for i in range(10)]

    selected = select_review_tasks(tasks, limit=6, offset=0, strategy="stratified")

    groups = {f"{row['guardrail']['review_bucket']}::{row['guardrail']['current_call_type']}" for row in selected}
    assert len(selected) == 6
    assert len(groups) == 2


def test_batch_prompt_and_extractor_validate_task_ids() -> None:
    tasks = [_task(1, bucket="a", call_type="unknown", contentful=False), _task(2, bucket="b", call_type="service_call", contentful=True)]
    prompt = build_batch_review_prompt(tasks)
    assert "task-001" in prompt
    payload = {
        "reviews": [
            {
                "task_id": "task-001",
                "review_schema_version": "v1",
                "decision": "force_non_conversation",
                "confidence": 0.9,
                "reason": "auto",
                "evidence": ["автоответчик"],
                "safe_to_auto_apply": True,
                "recommended_call_type": "non_conversation",
            },
            {
                "task_id": "task-002",
                "review_schema_version": "v1",
                "decision": "human_review_required",
                "confidence": 0.5,
                "reason": "risk",
                "evidence": ["service"],
                "safe_to_auto_apply": False,
                "recommended_call_type": "service_call",
            },
        ]
    }
    by_id = extract_batch_review_payloads(payload, ["task-001", "task-002"])
    assert set(by_id) == {"task-001", "task-002"}


def test_full_dry_run_pipeline_builds_validation_escalation_claude_and_consensus(tmp_path: Path) -> None:
    tasks = [
        _task(1, bucket="llm_review_non_contentful_probable_no_live", call_type="non_conversation", contentful=False),
        _task(2, bucket="llm_review_contentful_auto_fix_conflict", call_type="service_call", contentful=True),
        _task(3, bucket="human_review_sales_call_conflict", call_type="sales_call", contentful=True),
    ]
    input_jsonl = tmp_path / "tasks.jsonl"
    _write_jsonl(input_jsonl, tasks)

    review_root = tmp_path / "mini_review"
    summary = run_transcript_quality_llm_review(
        TranscriptQualityLLMReviewConfig(
            project_root=tmp_path,
            input_jsonl=input_jsonl,
            out_root=review_root,
            limit=0,
            sample_strategy="first",
            dry_run=True,
            force=True,
        )
    )
    assert summary["totals"]["reviews_written"] == 3

    validation_root = tmp_path / "validation"
    validation = validate_transcript_quality_reviews(
        ReviewValidatorConfig(
            tasks_jsonl=review_root / "selected_tasks.jsonl",
            reviews_jsonl=review_root / "reviews.jsonl",
            out_root=validation_root,
            review_tier="mini",
        )
    )
    assert validation["totals"]["escalation_tasks"] >= 2
    assert (validation_root / "escalation_tasks.jsonl").exists()

    escalation_root = tmp_path / "escalation"
    escalation = build_transcript_quality_escalation(
        EscalationConfig(validation_root=validation_root, out_root=escalation_root, limit=2)
    )
    assert escalation["selected_tasks"] == 2
    assert (escalation_root / "run_escalation_gpt55.sh").exists()

    claude_root = tmp_path / "claude"
    claude = build_transcript_quality_claude_package(
        ClaudePackageConfig(
            tasks_jsonl=review_root / "selected_tasks.jsonl",
            mini_reviews_jsonl=review_root / "reviews.jsonl",
            validation_root=validation_root,
            out_root=claude_root,
        )
    )
    assert claude["claude_items"] >= 2
    assert "Claude Audit Prompt" in (claude_root / "CLAUDE_AUDIT_PROMPT.md").read_text(encoding="utf-8")

    consensus_root = tmp_path / "consensus"
    consensus = build_transcript_quality_consensus(
        ConsensusConfig(
            tasks_jsonl=review_root / "selected_tasks.jsonl",
            mini_reviews_jsonl=review_root / "reviews.jsonl",
            mini_validation_root=validation_root,
            out_root=consensus_root,
        )
    )
    assert consensus["tasks"] == 3
    rows = _read_csv(consensus_root / "consensus.csv")
    assert len(rows) == 3
    assert any(row["consensus_route"] == "claude_audit_required" for row in rows)


def test_consensus_allows_claude_safe_force_on_high_risk_task(tmp_path: Path) -> None:
    tasks = [_task(1, bucket="llm_review_contentful_auto_fix_conflict", call_type="sales_call", contentful=True)]
    tasks[0]["guardrail"]["reason_codes"] = "system_no_dialogue_phrase|no_live_marker|outbound_voicemail"
    input_jsonl = tmp_path / "tasks.jsonl"
    mini_reviews = tmp_path / "mini_reviews.jsonl"
    claude_reviews = tmp_path / "claude_reviews.jsonl"
    validation_root = tmp_path / "validation"
    validation_root.mkdir()

    _write_jsonl(input_jsonl, tasks)
    _write_jsonl(
        mini_reviews,
        [
            {
                "task_id": "task-001",
                "decision": "force_non_conversation",
                "confidence": 0.97,
                "reason": "voicemail",
                "evidence": "voicemail",
                "safe_to_auto_apply": True,
                "recommended_call_type": "non_conversation",
            }
        ],
    )
    _write_jsonl(
        claude_reviews,
        [
            {
                "task_id": "task-001",
                "claude_decision": "force_non_conversation",
                "claude_confidence": 0.96,
                "claude_reason": "manager left outbound voicemail; no live client dialogue",
                "claude_evidence": ["client side is voicemail"],
                "safe_to_auto_apply": True,
                "recommended_call_type": "non_conversation",
            }
        ],
    )
    (validation_root / "validated_reviews.csv").write_text("task_id,validator_route\n", encoding="utf-8")

    consensus_root = tmp_path / "consensus"
    summary = build_transcript_quality_consensus(
        ConsensusConfig(
            tasks_jsonl=input_jsonl,
            mini_reviews_jsonl=mini_reviews,
            mini_validation_root=validation_root,
            out_root=consensus_root,
            claude_reviews_jsonl=claude_reviews,
        )
    )

    assert summary["counts"]["by_route"] == {"auto_apply_force_non_conversation": 1}
    rows = _read_csv(consensus_root / "consensus.csv")
    assert rows[0]["final_source"] == "claude"
    assert rows[0]["consensus_route"] == "auto_apply_force_non_conversation"
