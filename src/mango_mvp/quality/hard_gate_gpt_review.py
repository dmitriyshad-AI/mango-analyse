from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REVIEW_SCHEMA_VERSION = "hard_gate_gpt_review_v1"
PROMPT_VERSION = "hard_gate_gpt_review_prompt_v1"
DECISIONS = {"safe_apply", "keep_current", "manual_review"}


@dataclass(frozen=True)
class HardGateGptReviewConfig:
    input_jsonl: Path
    out_root: Path
    project_root: Path = Path(".")
    model: str = "gpt-5.5"
    reasoning_effort: str = "medium"
    limit: int = 0
    offset: int = 0
    batch_size: int = 8
    workers: int = 2
    timeout_sec: int = 600
    force: bool = False
    dry_run: bool = False
    codex_cli_command: str = "codex"
    codex_home: Path | None = None


def run_hard_gate_gpt_review(config: HardGateGptReviewConfig) -> dict[str, Any]:
    project_root = config.project_root.expanduser().resolve()
    input_path = _resolve_path(config.input_jsonl, project_root)
    out_root = _resolve_path(config.out_root, project_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = _select_tasks(_read_jsonl(input_path), limit=config.limit, offset=config.offset)
    existing = {} if config.force else _load_existing(out_root / "reviews.jsonl")

    reviews_by_task: dict[str, dict[str, Any]] = dict(existing)
    errors: list[dict[str, Any]] = []
    pending = [task for task in tasks if _task_id(task) and _task_id(task) not in reviews_by_task]
    skipped_existing = len(tasks) - len(pending)
    provider_calls = 0
    fallback_single_calls = 0
    dry_run_reviews = 0

    if config.dry_run:
        for task in pending:
            review = _normalize_review(_deterministic_payload(task), task, config)
            reviews_by_task[_task_id(task)] = review
            dry_run_reviews += 1
        _persist(out_root, reviews_by_task, errors)
    else:
        batches = _batches(pending, max(1, int(config.batch_size or 1)))
        max_workers = max(1, int(config.workers or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_review_batch, config, batch, project_root): batch for batch in batches}
            for future in as_completed(futures):
                batch = futures[future]
                provider_calls += 1
                try:
                    payloads = future.result()
                    for task in batch:
                        task_id = _task_id(task)
                        reviews_by_task[task_id] = _normalize_review(payloads[task_id], task, config)
                except Exception as batch_exc:  # noqa: BLE001
                    if len(batch) == 1:
                        errors.append({"task_id": _task_id(batch[0]), "error": f"{type(batch_exc).__name__}: {batch_exc}"})
                    else:
                        for task in batch:
                            task_id = _task_id(task)
                            try:
                                fallback_single_calls += 1
                                payloads = _call_codex_batch(config, [task], project_root)
                                reviews_by_task[task_id] = _normalize_review(payloads[task_id], task, config)
                            except Exception as single_exc:  # noqa: BLE001
                                errors.append(
                                    {
                                        "task_id": task_id,
                                        "error": (
                                            f"batch_failed={type(batch_exc).__name__}: {batch_exc}; "
                                            f"single_failed={type(single_exc).__name__}: {single_exc}"
                                        ),
                                    }
                                )
                _persist(out_root, reviews_by_task, errors)

    expected_ids = [_task_id(task) for task in tasks if _task_id(task)]
    for task_id in expected_ids:
        if task_id not in reviews_by_task:
            errors.append({"task_id": task_id, "error": "missing_review_after_run"})

    reviews = _sorted_reviews(reviews_by_task.values())
    outputs = _write_outputs(out_root, tasks, reviews, errors)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "hard_gate_gpt_review",
        "input_jsonl": str(input_path),
        "project_root": str(project_root),
        "config": {
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "limit": config.limit,
            "offset": config.offset,
            "batch_size": max(1, int(config.batch_size or 1)),
            "workers": max(1, int(config.workers or 1)),
            "timeout_sec": config.timeout_sec,
            "force": config.force,
            "dry_run": config.dry_run,
            "codex_home": str(config.codex_home.expanduser().resolve()) if config.codex_home else "",
        },
        "totals": {
            "input_tasks": len(tasks),
            "reviews_written": len(reviews),
            "errors": len(errors),
            "skipped_existing": skipped_existing,
            "provider_calls": provider_calls,
            "fallback_single_calls": fallback_single_calls,
            "dry_run_reviews": dry_run_reviews,
        },
        "counts": {
            "by_decision": dict(Counter(row.get("decision", "") for row in reviews).most_common()),
            "by_risk_level": dict(Counter(row.get("risk_level", "") for row in reviews).most_common()),
            "by_model": dict(Counter(row.get("model", "") for row in reviews).most_common()),
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "HARD_GATE_GPT_REVIEW_REPORT.md").write_text(_markdown_report(summary), encoding="utf-8")
    return summary


def _review_batch(config: HardGateGptReviewConfig, tasks: list[dict[str, Any]], project_root: Path) -> dict[str, dict[str, Any]]:
    return _call_codex_batch(config, tasks, project_root)


def _call_codex_batch(
    config: HardGateGptReviewConfig,
    tasks: list[dict[str, Any]],
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    if shutil.which(config.codex_cli_command) is None:
        raise RuntimeError(f"codex binary is not available: {config.codex_cli_command}")
    prompt = _batch_prompt(tasks)
    schema = _batch_schema()
    with tempfile.NamedTemporaryFile(prefix="hard_gate_gpt_review_", suffix=".json") as out_file, tempfile.NamedTemporaryFile(
        prefix="hard_gate_gpt_review_schema_",
        suffix=".json",
        mode="w",
        encoding="utf-8",
    ) as schema_file:
        schema_file.write(json.dumps(schema, ensure_ascii=False))
        schema_file.flush()
        cmd = _codex_command(config, project_root=project_root, output_path=Path(out_file.name), schema_path=Path(schema_file.name))
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(60, int(config.timeout_sec)),
            env=_codex_env(config),
        )
        raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore").strip()
    payload = None
    for candidate in (raw, proc.stdout or "", proc.stderr or ""):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            payload = _extract_json_object(candidate)
            break
        except Exception:
            continue
    if payload is None:
        stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
        raise RuntimeError(f"codex exec returned no JSON; rc={proc.returncode}; stderr_tail={stderr_tail[0]}")
    return _extract_reviews(payload, [_task_id(task) for task in tasks])


def _batch_prompt(tasks: list[dict[str, Any]]) -> str:
    payload = {
        "instructions": {
            "role": "Ты эксперт по качеству анализа звонков отдела продаж.",
            "task": "Для каждого input реши, можно ли безопасно заменить старый анализ на non_conversation.",
            "language": "Русский.",
            "critical_rules": [
                "safe_apply допустим только если нет живого содержательного диалога менеджер-клиент.",
                "Автоответчик, voicemail, IVR, виртуальный секретарь, голосовой помощник, занято, недоступен, нажмите 1, вне зоны действия = safe_apply, если клиент не вступал в диалог.",
                "Если менеджер оставил содержательное сообщение на автоответчике, но клиент не участвовал в диалоге, это safe_apply.",
                "Если внутри записи есть живой разговор о курсах, оплате, расписании, возражениях, следующем шаге или обратной связи по обучению, выбирай keep_current.",
                "Если звонок выглядит живым, но ASR противоречивый или данных мало, выбирай manual_review.",
                "Не добавляй task_id, которых нет во входе. Верни ровно один review на каждый task_id.",
            ],
            "decisions": sorted(DECISIONS),
            "return_json_schema": {
                "reviews": [
                    {
                        "task_id": "same as input task_id",
                        "decision": "safe_apply | keep_current | manual_review",
                        "confidence": 0.0,
                        "reason_ru": "короткое объяснение на русском",
                        "evidence": ["цитаты/признаки из transcript"],
                    }
                ]
            },
        },
        "inputs": [{"task_id": _task_id(task), "input": task} for task in tasks],
    }
    return (
        "Верни строго один JSON object по схеме. Не используй markdown. "
        "Не запускай shell-команды. Работай только с prompt.\n\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _batch_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reviews"],
        "properties": {
            "reviews": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["task_id", "decision", "confidence", "reason_ru", "evidence"],
                    "properties": {
                        "task_id": {"type": "string"},
                        "decision": {"type": "string", "enum": sorted(DECISIONS)},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason_ru": {"type": "string"},
                        "evidence": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        },
    }


def _codex_command(config: HardGateGptReviewConfig, *, project_root: Path, output_path: Path, schema_path: Path) -> list[str]:
    cmd = [
        config.codex_cli_command,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--cd",
        str(project_root),
        "--model",
        config.model,
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
    ]
    if config.reasoning_effort:
        cmd.extend(["-c", f'model_reasoning_effort="{config.reasoning_effort}"'])
    cmd.append("-")
    return cmd


def _codex_env(config: HardGateGptReviewConfig) -> dict[str, str]:
    env = os.environ.copy()
    if config.codex_home is not None:
        env["CODEX_HOME"] = str(config.codex_home.expanduser().resolve())
    elif not env.get("CODEX_HOME"):
        env["CODEX_HOME"] = str(_prepare_default_codex_home())
    return env


def _prepare_default_codex_home() -> Path:
    target = Path("/private/tmp/mango_codex_home_worker")
    target.mkdir(parents=True, exist_ok=True)
    (target / "sessions").mkdir(parents=True, exist_ok=True)
    source = Path.home() / ".codex"
    for name in ("auth.json", "config.toml", "installation_id", "models_cache.json", ".codex-global-state.json"):
        src = source / name
        dst = target / name
        _copy_codex_file_if_fresher(src, dst)
    try:
        target.chmod(0o700)
    except OSError:
        pass
    for name in ("auth.json", "config.toml"):
        try:
            (target / name).chmod(0o600)
        except OSError:
            pass
    return target


def _copy_codex_file_if_fresher(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    try:
        should_copy = not dst.exists()
        if not should_copy:
            src_stat = src.stat()
            dst_stat = dst.stat()
            should_copy = src_stat.st_mtime_ns > dst_stat.st_mtime_ns or src_stat.st_size != dst_stat.st_size
        if should_copy:
            shutil.copy2(src, dst)
    except OSError:
        pass


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("payload must be JSON object")
    return data


def _extract_reviews(payload: dict[str, Any], expected_ids: list[str]) -> dict[str, dict[str, Any]]:
    expected = [task_id for task_id in expected_ids if task_id]
    expected_set = set(expected)
    rows = payload.get("reviews")
    if not isinstance(rows, list):
        raise ValueError("payload must contain reviews array")
    by_id: dict[str, dict[str, Any]] = {}
    unexpected: list[str] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"review row {idx} must be object")
        task_id = _clean(row.get("task_id"))
        if task_id not in expected_set:
            unexpected.append(task_id)
            continue
        if task_id in by_id:
            raise ValueError(f"duplicate task_id: {task_id}")
        cleaned = dict(row)
        cleaned.pop("task_id", None)
        by_id[task_id] = cleaned
    missing = [task_id for task_id in expected if task_id not in by_id]
    if missing or unexpected:
        raise ValueError(f"task_id mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return by_id


def _normalize_review(payload: dict[str, Any], task: dict[str, Any], config: HardGateGptReviewConfig) -> dict[str, Any]:
    decision = _clean(payload.get("decision"))
    if decision not in DECISIONS:
        decision = "manual_review"
    return {
        "audit_id": _clean(task.get("audit_id")),
        "task_id": _task_id(task),
        "risk_level": _clean(task.get("risk_level")),
        "decision": decision,
        "confidence": _clamp_float(payload.get("confidence"), default=0.0),
        "reason_ru": _clean(payload.get("reason_ru") or payload.get("reason")),
        "evidence": " | ".join(_string_list(payload.get("evidence"))),
        "provider": "dry_run" if config.dry_run else "codex_cli",
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }


def _deterministic_payload(task: dict[str, Any]) -> dict[str, Any]:
    text = _clean((_dict(task.get("transcript")).get("full_text")))
    no_live = any(marker in text.lower() for marker in ("автоответчик", "не может ответить", "номер не отвечает", "занят"))
    return {
        "decision": "safe_apply" if no_live else "manual_review",
        "confidence": 0.5,
        "reason_ru": "dry-run placeholder, not an LLM decision",
        "evidence": [text[:200] if text else "no transcript"],
    }


def _select_tasks(tasks: list[dict[str, Any]], *, limit: int, offset: int) -> list[dict[str, Any]]:
    sliced = tasks[max(0, int(offset or 0)) :]
    if int(limit or 0) <= 0:
        return sliced
    return sliced[: max(0, int(limit))]


def _batches(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {_clean(row.get("task_id")): row for row in _read_jsonl(path) if _clean(row.get("task_id"))}


def _persist(out_root: Path, reviews_by_task: dict[str, dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    reviews = _sorted_reviews(reviews_by_task.values())
    _write_jsonl(out_root / "reviews.jsonl", reviews)
    _write_csv(out_root / "reviews.csv", reviews)
    _write_csv(out_root / "errors.csv", errors)


def _write_outputs(
    out_root: Path,
    tasks: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Path]:
    outputs = {
        "selected_tasks_jsonl": out_root / "selected_tasks.jsonl",
        "reviews_jsonl": out_root / "reviews.jsonl",
        "reviews_csv": out_root / "reviews.csv",
        "errors_csv": out_root / "errors.csv",
        "summary_json": out_root / "summary.json",
        "report_markdown": out_root / "HARD_GATE_GPT_REVIEW_REPORT.md",
    }
    _write_jsonl(outputs["selected_tasks_jsonl"], tasks)
    _write_jsonl(outputs["reviews_jsonl"], reviews)
    _write_csv(outputs["reviews_csv"], reviews)
    _write_csv(outputs["errors_csv"], errors)
    return outputs


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Hard Gate GPT Review Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Input: `{summary['input_jsonl']}`",
        f"- Model: `{summary['config']['model']}`",
        f"- Reasoning: `{summary['config']['reasoning_effort']}`",
        f"- Input tasks: `{summary['totals']['input_tasks']}`",
        f"- Reviews written: `{summary['totals']['reviews_written']}`",
        f"- Errors: `{summary['totals']['errors']}`",
        f"- Provider calls: `{summary['totals']['provider_calls']}`",
        f"- Fallback single calls: `{summary['totals']['fallback_single_calls']}`",
        "",
        "## Decisions",
    ]
    for key, value in summary["counts"]["by_decision"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Risk Levels"])
    for key, value in summary["counts"]["by_risk_level"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Outputs"])
    for key, path in summary["outputs"].items():
        lines.append(f"- `{key}`: `{path}`")
    return "\n".join(lines) + "\n"


def _sorted_reviews(rows: Any) -> list[dict[str, Any]]:
    return sorted((dict(row) for row in rows), key=lambda row: (_clean(row.get("audit_id")), _clean(row.get("task_id"))))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_path(path: Path, project_root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (project_root / expanded).resolve()


def _task_id(task: dict[str, Any]) -> str:
    return _clean(task.get("task_id"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean(item) for item in value if _clean(item)]
    cleaned = _clean(value)
    return [cleaned] if cleaned else []


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def _clamp_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(_clean(value))
    except ValueError:
        return default
    return max(0.0, min(1.0, parsed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT review for hard-gate non-conversation candidates.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument("--codex-home")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> HardGateGptReviewConfig:
    return HardGateGptReviewConfig(
        input_jsonl=Path(args.input_jsonl),
        out_root=Path(args.out_root),
        project_root=Path(args.project_root),
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        limit=args.limit,
        offset=args.offset,
        batch_size=args.batch_size,
        workers=args.workers,
        timeout_sec=args.timeout_sec,
        force=bool(args.force),
        dry_run=bool(args.dry_run),
        codex_cli_command=args.codex_cli_command,
        codex_home=Path(args.codex_home) if args.codex_home else None,
    )


__all__ = [
    "HardGateGptReviewConfig",
    "run_hard_gate_gpt_review",
    "config_from_args",
    "parse_args",
]
