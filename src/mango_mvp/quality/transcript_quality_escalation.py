from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.quality.transcript_quality_llm_review import (
    DEFAULT_ESCALATION_MODEL,
    DEFAULT_REASONING_EFFORT,
    read_jsonl,
    write_jsonl,
)
from mango_mvp.quality.transcript_quality_llm_review import write_csv


@dataclass(frozen=True)
class EscalationConfig:
    validation_root: Path
    out_root: Path
    model: str = DEFAULT_ESCALATION_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    limit: int = 0


def build_transcript_quality_escalation(config: EscalationConfig) -> dict[str, Any]:
    validation_root = config.validation_root.resolve()
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    source = validation_root / "escalation_tasks.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"Missing escalation tasks: {source}")
    tasks = read_jsonl(source)
    selected = tasks[: max(0, config.limit)] if config.limit else tasks
    outputs = {
        "escalation_tasks_jsonl": out_root / "escalation_tasks.jsonl",
        "run_command_sh": out_root / "run_escalation_gpt55.sh",
        "summary_json": out_root / "summary.json",
    }
    write_jsonl(outputs["escalation_tasks_jsonl"], selected)
    write_csv(out_root / "escalation_tasks.csv", [_task_row(task) for task in selected])
    command = f'''#!/bin/zsh
set -euo pipefail
cd "{Path.cwd()}"
PYTHONPATH=src python3 scripts/run_transcript_quality_llm_review.py \\
  --project-root . \\
  --input-jsonl "{outputs['escalation_tasks_jsonl']}" \\
  --out-root "{out_root / 'gpt55_review'}" \\
  --provider codex_cli \\
  --model "{config.model}" \\
  --reasoning-effort "{config.reasoning_effort}" \\
  --limit 0 \\
  --sample-strategy first \\
  --batch-size 5 \\
  --workers 4 \\
  --force
'''
    outputs["run_command_sh"].write_text(command, encoding="utf-8")
    outputs["run_command_sh"].chmod(0o755)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_root": str(validation_root),
        "source_tasks": len(tasks),
        "selected_tasks": len(selected),
        "recommended_model": config.model,
        "recommended_reasoning_effort": config.reasoning_effort,
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _task_row(task: dict[str, Any]) -> dict[str, Any]:
    guardrail = task.get("guardrail") if isinstance(task.get("guardrail"), dict) else {}
    call = task.get("call") if isinstance(task.get("call"), dict) else {}
    return {
        "task_id": task.get("task_id") or task.get("id") or "",
        "review_bucket": guardrail.get("review_bucket", ""),
        "current_call_type": guardrail.get("current_call_type", ""),
        "current_contentful": guardrail.get("current_contentful", ""),
        "source_filename": call.get("source_filename", ""),
        "started_at": call.get("started_at", ""),
        "duration_sec": call.get("duration_sec", ""),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GPT-5.5 escalation package for transcript-quality reviews.")
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_ESCALATION_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> EscalationConfig:
    return EscalationConfig(
        validation_root=args.validation_root,
        out_root=args.out_root,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        limit=args.limit,
    )
