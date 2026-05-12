from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.quality.transcript_quality_llm_review import read_jsonl, write_csv, write_jsonl


@dataclass(frozen=True)
class ClaudePackageConfig:
    tasks_jsonl: Path
    mini_reviews_jsonl: Path
    validation_root: Path
    out_root: Path
    advanced_reviews_jsonl: Path | None = None
    limit: int = 0


def build_transcript_quality_claude_package(config: ClaudePackageConfig) -> dict[str, Any]:
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = {_task_id(task): task for task in read_jsonl(config.tasks_jsonl) if _task_id(task)}
    mini = {_clean(row.get("task_id")): row for row in read_jsonl(config.mini_reviews_jsonl) if _clean(row.get("task_id"))}
    advanced = {}
    if config.advanced_reviews_jsonl and config.advanced_reviews_jsonl.exists():
        advanced = {_clean(row.get("task_id")): row for row in read_jsonl(config.advanced_reviews_jsonl) if _clean(row.get("task_id"))}

    candidate_ids = _candidate_task_ids(config.validation_root, tasks)
    if config.limit:
        candidate_ids = candidate_ids[: max(0, config.limit)]
    items = []
    for task_id in candidate_ids:
        task = tasks.get(task_id)
        if not task:
            continue
        items.append(build_claude_item(task, mini.get(task_id, {}), advanced.get(task_id, {})))

    outputs = {
        "items_jsonl": out_root / "claude_audit_items.jsonl",
        "items_csv": out_root / "claude_audit_items.csv",
        "prompt_md": out_root / "CLAUDE_AUDIT_PROMPT.md",
        "readme_md": out_root / "README_FOR_CLAUDE.md",
        "summary_json": out_root / "summary.json",
    }
    write_jsonl(outputs["items_jsonl"], items)
    write_csv(outputs["items_csv"], [flatten_claude_item(item) for item in items])
    outputs["prompt_md"].write_text(claude_prompt(), encoding="utf-8")
    outputs["readme_md"].write_text(readme_text(outputs), encoding="utf-8")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks_jsonl": str(config.tasks_jsonl.resolve()),
        "mini_reviews_jsonl": str(config.mini_reviews_jsonl.resolve()),
        "advanced_reviews_jsonl": str(config.advanced_reviews_jsonl.resolve()) if config.advanced_reviews_jsonl else "",
        "validation_root": str(config.validation_root.resolve()),
        "claude_items": len(items),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _candidate_task_ids(validation_root: Path, tasks: dict[str, dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for name in ("escalation_queue.csv", "human_or_claude_required.csv", "invalid_reviews.csv"):
        path = validation_root / name
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                task_id = _clean(row.get("task_id"))
                if task_id and task_id not in ids:
                    ids.append(task_id)
    if not ids:
        ids = list(tasks)
    return ids


def build_claude_item(task: dict[str, Any], mini: dict[str, Any], advanced: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": _task_id(task),
        "task": task,
        "mini_review": mini,
        "advanced_review": advanced,
        "claude_required_output": {
            "task_id": _task_id(task),
            "claude_decision": "keep_current_analysis | force_non_conversation | reanalyze_required | human_review_required",
            "claude_confidence": 0.0,
            "claude_reason": "короткое объяснение на русском",
            "claude_evidence": ["признаки из transcript/current_analysis"],
            "safe_to_auto_apply": False,
            "recommended_call_type": "non_conversation | sales_call | service_call | technical_call | existing_client_progress | unknown",
        },
    }


def flatten_claude_item(item: dict[str, Any]) -> dict[str, Any]:
    task = item.get("task") if isinstance(item.get("task"), dict) else {}
    guardrail = task.get("guardrail") if isinstance(task.get("guardrail"), dict) else {}
    call = task.get("call") if isinstance(task.get("call"), dict) else {}
    mini = item.get("mini_review") if isinstance(item.get("mini_review"), dict) else {}
    advanced = item.get("advanced_review") if isinstance(item.get("advanced_review"), dict) else {}
    return {
        "task_id": item.get("task_id", ""),
        "review_bucket": guardrail.get("review_bucket", ""),
        "current_call_type": guardrail.get("current_call_type", ""),
        "source_filename": call.get("source_filename", ""),
        "mini_decision": mini.get("decision", ""),
        "mini_confidence": mini.get("confidence", ""),
        "advanced_decision": advanced.get("decision", ""),
        "advanced_confidence": advanced.get("confidence", ""),
        "transcript_excerpt": str(task.get("transcript_text", ""))[:1200],
    }


def claude_prompt() -> str:
    return """# Claude Audit Prompt: Transcript Quality Review

Проверь решения GPT по спорным расшифровкам звонков. Главная цель: не допустить, чтобы автоответчик, отсутствие живого диалога или ASR-мусор попадали в CRM/ROP/KB как содержательный разговор.

Для каждого `claude_audit_items.jsonl` верни JSONL, один объект на строку:

```json
{
  "task_id": "...",
  "claude_decision": "keep_current_analysis | force_non_conversation | reanalyze_required | human_review_required",
  "claude_confidence": 0.0,
  "claude_reason": "короткое объяснение на русском",
  "claude_evidence": ["признаки из transcript/current_analysis"],
  "safe_to_auto_apply": false,
  "recommended_call_type": "non_conversation | sales_call | service_call | technical_call | existing_client_progress | unknown"
}
```

Правила:
- Не затирай реальный sales/service/technical диалог, если он есть.
- Если transcript явно автоответчик/системная фраза/нет живого диалога, выбирай `force_non_conversation`.
- Если менеджер оставил содержательное сообщение, но клиентская сторона — автоответчик/голосовая почта/виртуальный секретарь/IVR, это всё равно `force_non_conversation`: клиент не участвовал в диалоге, а прослушивание сообщения не подтверждено.
- К no-live маркерам относятся: голосовая почта, абонент недоступен/занят/вне зоны действия, звонок перенаправлен, отправить бесплатное смс, нажмите 1, я секретарь, голосовой ассистент/помощник, ассистент Миа, вас приветствует компания, все разговоры записываются.
- Если разговор есть, но текущий analysis неверный, выбирай `reanalyze_required`.
- Если GPT-mini и GPT-advanced расходятся или уверенность низкая, выбирай `human_review_required`.
- Не используй аудио; работай только с расшифровками и текущим анализом.
"""


def readme_text(outputs: dict[str, Path]) -> str:
    return (
        "# Claude Audit Package\n\n"
        "Файлы для проверки:\n\n"
        f"- `{outputs['items_jsonl'].name}`: задачи с transcript, текущим analysis и решениями GPT.\n"
        f"- `{outputs['items_csv'].name}`: краткая таблица для просмотра.\n"
        f"- `{outputs['prompt_md'].name}`: промт, который нужно дать Claude.\n\n"
        "Ожидаемый результат от Claude: JSONL с решениями по схеме из промта.\n"
    )


def _task_id(task: dict[str, Any]) -> str:
    return _clean(task.get("task_id") or task.get("id"))


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Claude audit package for transcript-quality review decisions.")
    parser.add_argument("--tasks-jsonl", type=Path, required=True)
    parser.add_argument("--mini-reviews-jsonl", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--advanced-reviews-jsonl", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ClaudePackageConfig:
    return ClaudePackageConfig(
        tasks_jsonl=args.tasks_jsonl,
        mini_reviews_jsonl=args.mini_reviews_jsonl,
        validation_root=args.validation_root,
        out_root=args.out_root,
        advanced_reviews_jsonl=args.advanced_reviews_jsonl,
        limit=args.limit,
    )
