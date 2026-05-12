#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mango_mvp.quality.transcript_quality_claude_package import ClaudePackageConfig, build_transcript_quality_claude_package
from mango_mvp.quality.transcript_quality_consensus import ConsensusConfig, build_transcript_quality_consensus
from mango_mvp.quality.transcript_quality_escalation import EscalationConfig, build_transcript_quality_escalation
from mango_mvp.quality.transcript_quality_llm_review import TranscriptQualityLLMReviewConfig, run_transcript_quality_llm_review
from mango_mvp.quality.transcript_quality_review_validator import ReviewValidatorConfig, validate_transcript_quality_reviews


def main() -> int:
    args = parse_args()
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    mini_root = out_root / "mini_review"
    mini_summary = run_transcript_quality_llm_review(
        TranscriptQualityLLMReviewConfig(
            project_root=args.project_root,
            input_jsonl=args.input_jsonl,
            out_root=mini_root,
            provider="codex_cli",
            model=args.mini_model,
            reasoning_effort=args.reasoning_effort,
            limit=args.limit,
            offset=args.offset,
            sample_strategy=args.sample_strategy,
            batch_size=args.mini_batch_size,
            workers=args.mini_workers,
            dry_run=args.dry_run_mini,
            force=args.force,
            timeout_sec=args.timeout_sec,
            codex_home=args.codex_home,
            cache_enabled=not args.no_cache,
        )
    )

    mini_validation_root = out_root / "mini_validation"
    mini_validation_summary = validate_transcript_quality_reviews(
        ReviewValidatorConfig(
            tasks_jsonl=mini_root / "selected_tasks.jsonl",
            reviews_jsonl=mini_root / "reviews.jsonl",
            out_root=mini_validation_root,
            review_tier="mini",
        )
    )

    escalation_root = out_root / "escalation_gpt55"
    escalation_summary = build_transcript_quality_escalation(
        EscalationConfig(
            validation_root=mini_validation_root,
            out_root=escalation_root,
            model=args.advanced_model,
            reasoning_effort=args.reasoning_effort,
            limit=args.advanced_limit,
        )
    )

    advanced_root = out_root / "gpt55_review"
    advanced_validation_root = out_root / "gpt55_validation"
    advanced_summary = None
    advanced_validation_summary = None
    advanced_reviews_path: Path | None = None
    if args.run_advanced:
        advanced_summary = run_transcript_quality_llm_review(
            TranscriptQualityLLMReviewConfig(
                project_root=args.project_root,
                input_jsonl=escalation_root / "escalation_tasks.jsonl",
                out_root=advanced_root,
                provider="codex_cli",
                model=args.advanced_model,
                reasoning_effort=args.reasoning_effort,
                limit=0,
                sample_strategy="first",
                batch_size=args.advanced_batch_size,
                workers=args.advanced_workers,
                dry_run=args.dry_run_advanced,
                force=args.force,
                timeout_sec=args.timeout_sec,
                codex_home=args.codex_home,
                cache_enabled=not args.no_cache,
            )
        )
        advanced_reviews_path = advanced_root / "reviews.jsonl"
        advanced_validation_summary = validate_transcript_quality_reviews(
            ReviewValidatorConfig(
                tasks_jsonl=advanced_root / "selected_tasks.jsonl",
                reviews_jsonl=advanced_reviews_path,
                out_root=advanced_validation_root,
                review_tier="advanced",
            )
        )

    claude_root = out_root / "claude_audit_package"
    claude_validation_root = advanced_validation_root if advanced_reviews_path is not None else mini_validation_root
    claude_summary = build_transcript_quality_claude_package(
        ClaudePackageConfig(
            tasks_jsonl=mini_root / "selected_tasks.jsonl",
            mini_reviews_jsonl=mini_root / "reviews.jsonl",
            validation_root=claude_validation_root,
            out_root=claude_root,
            advanced_reviews_jsonl=advanced_reviews_path,
            limit=args.claude_limit,
        )
    )

    consensus_root = out_root / "consensus"
    consensus_summary = build_transcript_quality_consensus(
        ConsensusConfig(
            tasks_jsonl=mini_root / "selected_tasks.jsonl",
            mini_reviews_jsonl=mini_root / "reviews.jsonl",
            mini_validation_root=mini_validation_root,
            out_root=consensus_root,
            advanced_reviews_jsonl=advanced_reviews_path,
        )
    )

    summary = {
        "out_root": str(out_root),
        "mini_review": mini_summary,
        "mini_validation": mini_validation_summary,
        "escalation": escalation_summary,
        "advanced_review": advanced_summary,
        "advanced_validation": advanced_validation_summary,
        "claude_package": claude_summary,
        "consensus": consensus_summary,
    }
    (out_root / "pipeline_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transcript-quality review pipeline: mini -> validator -> GPT-5.5 -> Claude package -> consensus.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-strategy", default="stratified", choices=["stratified", "first"])
    parser.add_argument("--mini-model", default="gpt-5.4-mini")
    parser.add_argument("--advanced-model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--advanced-batch-size", type=int, default=5)
    parser.add_argument("--mini-workers", type=int, default=6)
    parser.add_argument("--advanced-workers", type=int, default=4)
    parser.add_argument("--advanced-limit", type=int, default=0)
    parser.add_argument("--claude-limit", type=int, default=300)
    parser.add_argument("--run-advanced", action="store_true")
    parser.add_argument("--dry-run-mini", action="store_true")
    parser.add_argument("--dry-run-advanced", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--codex-home", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
