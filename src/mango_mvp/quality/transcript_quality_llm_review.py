from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.services.llm_response_cache import LLMResponseCache
from mango_mvp.utils.codex_cli import append_codex_service_tier


PROMPT_VERSION = "transcript_quality_llm_review_v2"
REVIEW_SCHEMA_VERSION = "transcript_quality_review_v1"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_ESCALATION_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "medium"
DECISIONS = {
    "keep_current_analysis",
    "force_non_conversation",
    "reanalyze_required",
    "human_review_required",
}
CALL_TYPES = {
    "non_conversation",
    "sales_call",
    "service_call",
    "technical_call",
    "existing_client_progress",
    "unknown",
}


@dataclass(frozen=True)
class TranscriptQualityLLMReviewConfig:
    project_root: Path
    input_jsonl: Path
    out_root: Path
    provider: str = "codex_cli"
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    limit: int = 1000
    offset: int = 0
    sample_strategy: str = "stratified"
    batch_size: int = 8
    workers: int = 6
    dry_run: bool = False
    force: bool = False
    cache_enabled: bool = True
    cache_dir: Path = Path(".cache/llm_responses")
    timeout_sec: int = 240
    codex_cli_command: str = "codex"
    codex_home: Path | None = None


def run_transcript_quality_llm_review(config: TranscriptQualityLLMReviewConfig) -> dict[str, Any]:
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    all_tasks = read_jsonl(config.input_jsonl)
    selected = select_review_tasks(
        all_tasks,
        limit=config.limit,
        offset=config.offset,
        strategy=config.sample_strategy,
    )
    selected_by_id = {_task_id(task): task for task in selected if _task_id(task)}
    existing = load_existing_reviews(out_root / "reviews.jsonl") if not config.force else {}
    cache = LLMResponseCache(enabled=config.cache_enabled, root_dir=config.cache_dir)

    reviews_by_id: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    skipped_existing = 0
    cache_hits = 0
    dry_run_reviews = 0
    provider_calls = 0
    fallback_single_calls = 0

    pending: list[dict[str, Any]] = []
    for task in selected:
        task_id = _task_id(task)
        if not task_id:
            errors.append({"task_id": "", "error": "missing task_id"})
            continue
        if task_id in existing:
            reviews_by_id[task_id] = existing[task_id]
            skipped_existing += 1
            continue
        prompt = build_single_review_prompt(task)
        cached = cache.get(
            namespace="transcript_quality_llm_review",
            provider=config.provider,
            model=config.model,
            reasoning=config.reasoning_effort,
            prompt_version=PROMPT_VERSION,
            prompt=prompt,
        )
        if cached is not None:
            reviews_by_id[task_id] = flatten_review(task, normalize_review_payload(cached, task, config), config)
            cache_hits += 1
            continue
        pending.append(task)

    if config.dry_run:
        for task in pending:
            payload = deterministic_review_payload(task)
            reviews_by_id[_task_id(task)] = flatten_review(task, normalize_review_payload(payload, task, config), config)
            dry_run_reviews += 1
    else:
        batches = make_batches(pending, normalized_batch_size(config))
        max_workers = max(1, int(config.workers or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(review_batch, config, batch): batch for batch in batches}
            for future in as_completed(futures):
                batch = futures[future]
                provider_calls += 1
                try:
                    payload_by_id = future.result()
                    for task in batch:
                        task_id = _task_id(task)
                        payload = payload_by_id[task_id]
                        cache.put(
                            namespace="transcript_quality_llm_review",
                            provider=config.provider,
                            model=config.model,
                            reasoning=config.reasoning_effort,
                            prompt_version=PROMPT_VERSION,
                            prompt=build_single_review_prompt(task),
                            response=payload,
                        )
                        reviews_by_id[task_id] = flatten_review(task, normalize_review_payload(payload, task, config), config)
                except Exception as batch_exc:  # noqa: BLE001
                    if len(batch) == 1:
                        errors.append({"task_id": _task_id(batch[0]), "error": f"{type(batch_exc).__name__}: {batch_exc}"})
                        continue
                    for task in batch:
                        task_id = _task_id(task)
                        try:
                            fallback_single_calls += 1
                            payload = call_review_provider(config, [task])[_task_id(task)]
                            cache.put(
                                namespace="transcript_quality_llm_review",
                                provider=config.provider,
                                model=config.model,
                                reasoning=config.reasoning_effort,
                                prompt_version=PROMPT_VERSION,
                                prompt=build_single_review_prompt(task),
                                response=payload,
                            )
                            reviews_by_id[task_id] = flatten_review(task, normalize_review_payload(payload, task, config), config)
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
                persist_outputs(out_root, sorted_reviews(list(reviews_by_id.values())), errors)

    missing_review_ids = [task_id for task_id in selected_by_id if task_id not in reviews_by_id]
    for task_id in missing_review_ids:
        errors.append({"task_id": task_id, "error": "missing_review_after_run"})

    reviews = sorted_reviews(list(reviews_by_id.values()))
    summary = build_review_summary(
        config,
        all_tasks=all_tasks,
        selected=selected,
        reviews=reviews,
        errors=errors,
        skipped_existing=skipped_existing,
        cache_hits=cache_hits,
        dry_run_reviews=dry_run_reviews,
        provider_calls=provider_calls,
        fallback_single_calls=fallback_single_calls,
    )
    outputs = write_review_outputs(out_root, summary, selected, reviews, errors)
    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_no}: {exc}") from exc
            if isinstance(payload, dict):
                items.append(payload)
    return items


def select_review_tasks(tasks: list[dict[str, Any]], *, limit: int, offset: int, strategy: str) -> list[dict[str, Any]]:
    sliced = tasks[max(0, offset) :]
    target = len(sliced) if int(limit or 0) == 0 else max(0, int(limit or 0))
    if target <= 0:
        return []
    if strategy == "first":
        return sliced[:target]
    if strategy != "stratified":
        raise ValueError(f"Unsupported sample strategy: {strategy}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in sliced:
        guardrail = _dict(task.get("guardrail"))
        key = f"{guardrail.get('review_bucket') or 'unknown'}::{guardrail.get('current_call_type') or 'unknown'}"
        groups[key].append(task)
    keys = sorted(groups, key=lambda key: (-len(groups[key]), key))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(selected) < target and keys:
        progressed = False
        for key in list(keys):
            bucket = groups[key]
            if not bucket:
                keys.remove(key)
                continue
            task = bucket.pop(0)
            task_id = _task_id(task)
            if task_id and task_id not in seen:
                seen.add(task_id)
                selected.append(task)
                progressed = True
                if len(selected) >= target:
                    break
        if not progressed:
            break
    return selected


def build_single_review_prompt(task: dict[str, Any]) -> str:
    return build_batch_review_prompt([task])


def build_batch_review_prompt(tasks: list[dict[str, Any]]) -> str:
    payload = {
        "instructions": {
            "role": "Ты эксперт по качеству расшифровок звонков отдела продаж и CRM-аналитики.",
            "task": (
                "Для каждого input реши, можно ли доверять текущему анализу звонка или звонок надо считать "
                "non_conversation/no-live/ASR artifact. Используй только transcript/current_analysis/guardrail."
            ),
            "language": "Русский.",
            "critical_rules": [
                "Не придумывай содержательный разговор, если transcript похож на автоответчик, системную фразу, тишину или ASR-мусор.",
                "Если менеджер оставил даже содержательное сообщение, но клиентская сторона — автоответчик/голосовая почта/виртуальный секретарь/IVR, это no-live контакт: выбирай force_non_conversation и recommended_call_type=non_conversation.",
                "К no-live маркерам относятся: голосовая почта, абонент недоступен/занят/вне зоны действия, звонок перенаправлен, отправить бесплатное смс, нажмите 1, я секретарь, голосовой ассистент/помощник, ассистент Миа, вас приветствует компания, все разговоры записываются.",
                "Если есть реальный диалог менеджер-клиент с вопросами/ответами/следующим шагом, не затирай его в non_conversation из-за одной artifact-фразы.",
                "Для sales_call/service_call/technical_call будь консервативен: при сомнениях выбирай human_review_required или reanalyze_required.",
                "decision force_non_conversation допустим только при явных признаках отсутствия живого содержательного диалога.",
                "Верни ровно один review на каждый task_id, не добавляй лишние task_id.",
            ],
            "decisions": sorted(DECISIONS),
            "call_types": sorted(CALL_TYPES),
            "return_json_schema": {
                "reviews": [
                    {
                        "task_id": "same as input task_id",
                        **review_json_schema_hint(),
                    }
                ]
            },
        },
        "inputs": [{"task_id": _task_id(task), "input": task} for task in tasks],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def review_json_schema_hint() -> dict[str, Any]:
    return {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "decision": "keep_current_analysis | force_non_conversation | reanalyze_required | human_review_required",
        "confidence": 0.0,
        "reason": "короткое объяснение на русском",
        "evidence": ["признаки из transcript/current_analysis"],
        "safe_to_auto_apply": False,
        "recommended_call_type": "non_conversation | sales_call | service_call | technical_call | existing_client_progress | unknown",
    }


def batch_review_output_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reviews"],
        "properties": {
            "reviews": {
                "type": "array",
                "minItems": 1,
                "items": review_payload_json_schema(include_task_id=True),
            }
        },
    }


def review_payload_json_schema(*, include_task_id: bool) -> dict[str, Any]:
    required = [
        "review_schema_version",
        "decision",
        "confidence",
        "reason",
        "evidence",
        "safe_to_auto_apply",
        "recommended_call_type",
    ]
    properties: dict[str, Any] = {
        "review_schema_version": {"type": "string"},
        "decision": {"type": "string", "enum": sorted(DECISIONS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "safe_to_auto_apply": {"type": "boolean"},
        "recommended_call_type": {"type": "string", "enum": sorted(CALL_TYPES)},
    }
    if include_task_id:
        required = ["task_id", *required]
        properties = {"task_id": {"type": "string"}, **properties}
    return {"type": "object", "additionalProperties": False, "required": required, "properties": properties}


def review_batch(config: TranscriptQualityLLMReviewConfig, tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return call_review_provider(config, tasks)


def call_review_provider(config: TranscriptQualityLLMReviewConfig, tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if config.provider != "codex_cli":
        raise RuntimeError(f"Unsupported transcript-quality review provider: {config.provider}")
    payload = call_codex_cli_json(
        config,
        (
            "Верни строго один JSON object по переданной схеме. "
            "Поле reviews должно содержать ровно один review на каждый входной task_id. "
            "Не используй markdown. Не запускай shell-команды. Работай только с prompt.\n\n"
            + build_batch_review_prompt(tasks)
        ),
        batch_review_output_json_schema(),
    )
    return extract_batch_review_payloads(payload, [_task_id(task) for task in tasks])


def call_codex_cli_json(config: TranscriptQualityLLMReviewConfig, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    codex_bin = (config.codex_cli_command or "codex").strip() or "codex"
    if shutil.which(codex_bin) is None:
        raise RuntimeError(f"codex binary is not available: {codex_bin}")
    with tempfile.NamedTemporaryFile(prefix="transcript_quality_review_", suffix=".json") as out_file, tempfile.NamedTemporaryFile(
        prefix="transcript_quality_review_schema_",
        suffix=".json",
        mode="w",
        encoding="utf-8",
    ) as schema_file:
        schema_file.write(json.dumps(schema, ensure_ascii=False))
        schema_file.flush()
        cmd = build_codex_cli_command(config, output_path=Path(out_file.name), schema_path=Path(schema_file.name))
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            check=False,
            timeout=max(30, int(config.timeout_sec)),
            env=codex_cli_env(config),
        )
        raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore").strip()
    for candidate in (raw, proc.stdout or "", proc.stderr or ""):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            return extract_json_object(candidate)
        except Exception:
            continue
    stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
    raise RuntimeError(f"codex exec returned no JSON review; rc={proc.returncode}; stderr_tail={stderr_tail[0]}")


def build_codex_cli_command(config: TranscriptQualityLLMReviewConfig, *, output_path: Path, schema_path: Path) -> list[str]:
    cmd = [
        (config.codex_cli_command or "codex").strip() or "codex",
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--cd",
        str(config.project_root.resolve()),
        "--model",
        config.model,
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
    ]
    append_codex_service_tier(cmd)
    reasoning = (config.reasoning_effort or "").strip().lower()
    if reasoning:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning}"'])
    cmd.append("-")
    return cmd


def codex_cli_env(config: TranscriptQualityLLMReviewConfig) -> dict[str, str]:
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
    for name in (
        "auth.json",
        "config.toml",
        "installation_id",
        "models_cache.json",
        ".codex-global-state.json",
    ):
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


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Review payload must be a JSON object")
    return data


def extract_batch_review_payloads(payload: dict[str, Any], expected_ids: list[str]) -> dict[str, dict[str, Any]]:
    expected = [task_id for task_id in expected_ids if task_id]
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("Batch input contains duplicate task_id values")
    rows = payload.get("reviews")
    if not isinstance(rows, list):
        raise ValueError("Batch review payload must contain reviews array")
    by_id: dict[str, dict[str, Any]] = {}
    unexpected: list[str] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Batch review row {idx} must be a JSON object")
        task_id = _clean(row.get("task_id"))
        if not task_id:
            raise ValueError(f"Batch review row {idx} has empty task_id")
        if task_id not in expected_set:
            unexpected.append(task_id)
            continue
        if task_id in by_id:
            raise ValueError(f"Batch review has duplicate task_id: {task_id}")
        cleaned = dict(row)
        cleaned.pop("task_id", None)
        by_id[task_id] = cleaned
    missing = [task_id for task_id in expected if task_id not in by_id]
    if unexpected or missing:
        parts = []
        if missing:
            parts.append(f"missing={missing[:10]}")
        if unexpected:
            parts.append(f"unexpected={unexpected[:10]}")
        raise ValueError("Batch review task_id mismatch: " + "; ".join(parts))
    return by_id


def deterministic_review_payload(task: dict[str, Any]) -> dict[str, Any]:
    guardrail = _dict(task.get("guardrail"))
    bucket = _clean(guardrail.get("review_bucket"))
    call_type = _clean(guardrail.get("current_call_type")) or "unknown"
    reason_codes = _clean(guardrail.get("reason_codes"))
    transcript = _clean(task.get("transcript_text"))
    has_no_live = any(marker in reason_codes for marker in ("no_live_marker", "system_no_dialogue_phrase", "asr_artifact_marker"))
    has_outbound_voicemail = "outbound_voicemail" in reason_codes
    contentful = bool(guardrail.get("current_contentful")) or call_type in {"sales_call", "service_call", "technical_call", "existing_client_progress"}

    if has_outbound_voicemail:
        decision = "force_non_conversation"
        confidence = 0.9
        safe = False  # dry-run should never be applied automatically.
        recommended = "non_conversation"
    elif "borderline" in bucket or call_type == "sales_call":
        decision = "human_review_required"
        confidence = 0.62
        safe = False
        recommended = call_type
    elif not contentful and has_no_live:
        decision = "force_non_conversation"
        confidence = 0.88
        safe = False  # dry-run should never be applied automatically.
        recommended = "non_conversation"
    elif contentful and has_no_live:
        decision = "reanalyze_required"
        confidence = 0.72
        safe = False
        recommended = call_type
    else:
        decision = "human_review_required"
        confidence = 0.55
        safe = False
        recommended = call_type

    evidence = []
    if reason_codes:
        evidence.append(f"guardrail reason_codes: {reason_codes}")
    if transcript:
        evidence.append(transcript[:240])
    return {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "decision": decision,
        "confidence": confidence,
        "reason": "dry-run heuristic placeholder; требуется live LLM-review перед применением",
        "evidence": evidence[:3] or ["dry-run placeholder"],
        "safe_to_auto_apply": safe,
        "recommended_call_type": recommended if recommended in CALL_TYPES else "unknown",
    }


def normalize_review_payload(payload: dict[str, Any], task: dict[str, Any], config: TranscriptQualityLLMReviewConfig) -> dict[str, Any]:
    decision = _clean(payload.get("decision"))
    if decision not in DECISIONS:
        decision = "human_review_required"
    recommended = _clean(payload.get("recommended_call_type"))
    if recommended not in CALL_TYPES:
        recommended = _clean(_dict(task.get("guardrail")).get("current_call_type")) or "unknown"
    if recommended not in CALL_TYPES:
        recommended = "unknown"
    return {
        "review_schema_version": _clean(payload.get("review_schema_version")) or REVIEW_SCHEMA_VERSION,
        "decision": decision,
        "confidence": _clamp_float(payload.get("confidence"), 0.0, 1.0, 0.0),
        "reason": _clean(payload.get("reason")),
        "evidence": _string_list(payload.get("evidence")),
        "safe_to_auto_apply": _is_true(payload.get("safe_to_auto_apply")),
        "recommended_call_type": recommended,
        "provider": "dry_run" if config.dry_run else config.provider,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "prompt_version": PROMPT_VERSION,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }


def flatten_review(task: dict[str, Any], review: dict[str, Any], config: TranscriptQualityLLMReviewConfig) -> dict[str, Any]:
    guardrail = _dict(task.get("guardrail"))
    call = _dict(task.get("call"))
    return {
        "task_id": _task_id(task),
        "provider": review.get("provider", ""),
        "model": review.get("model", ""),
        "reasoning_effort": review.get("reasoning_effort", ""),
        "prompt_version": review.get("prompt_version", ""),
        "review_schema_version": review.get("review_schema_version", ""),
        "decision": review.get("decision", ""),
        "confidence": review.get("confidence", ""),
        "safe_to_auto_apply": review.get("safe_to_auto_apply", False),
        "recommended_call_type": review.get("recommended_call_type", ""),
        "reason": review.get("reason", ""),
        "evidence": " | ".join(_string_list(review.get("evidence"))),
        "call_id": call.get("id", ""),
        "source_filename": call.get("source_filename", ""),
        "started_at": call.get("started_at", ""),
        "duration_sec": call.get("duration_sec", ""),
        "manager_name": call.get("manager_name", ""),
        "phone": call.get("phone", ""),
        "review_bucket": guardrail.get("review_bucket", ""),
        "current_call_type": guardrail.get("current_call_type", ""),
        "current_contentful": guardrail.get("current_contentful", ""),
        "guardrail_label": guardrail.get("label", ""),
        "guardrail_score": guardrail.get("score", ""),
        "guardrail_reason_codes": guardrail.get("reason_codes", ""),
        "should_force_non_conversation": guardrail.get("should_force_non_conversation", ""),
        "requires_manual_review": guardrail.get("requires_manual_review", ""),
        "protected_live_dialogue": guardrail.get("protected_live_dialogue", ""),
        "reviewed_at": review.get("reviewed_at", ""),
    }


def build_review_summary(
    config: TranscriptQualityLLMReviewConfig,
    *,
    all_tasks: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    skipped_existing: int,
    cache_hits: int,
    dry_run_reviews: int,
    provider_calls: int,
    fallback_single_calls: int,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(config.input_jsonl.resolve()),
        "config": {
            "provider": config.provider,
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "limit": config.limit,
            "offset": config.offset,
            "sample_strategy": config.sample_strategy,
            "batch_size": normalized_batch_size(config),
            "workers": max(1, int(config.workers or 1)),
            "dry_run": config.dry_run,
            "force": config.force,
            "cache_enabled": config.cache_enabled,
            "codex_home": str(config.codex_home.resolve()) if config.codex_home else "",
        },
        "totals": {
            "input_tasks": len(all_tasks),
            "selected_tasks": len(selected),
            "reviews_written": len(reviews),
            "errors": len(errors),
            "skipped_existing": skipped_existing,
            "cache_hits": cache_hits,
            "dry_run_reviews": dry_run_reviews,
            "provider_calls": provider_calls,
            "fallback_single_calls": fallback_single_calls,
        },
        "counts": {
            "by_decision": dict(Counter(row.get("decision", "") for row in reviews).most_common()),
            "by_review_bucket": dict(Counter(row.get("review_bucket", "") for row in reviews).most_common()),
            "by_current_call_type": dict(Counter(row.get("current_call_type", "") for row in reviews).most_common()),
            "by_model": dict(Counter(row.get("model", "") for row in reviews).most_common()),
        },
        "audit_notes": [
            "This stage never writes to the runtime DB.",
            "Dry-run reviews are structural placeholders only and must not be applied.",
            "Validator and consensus stages are required before any staged apply.",
        ],
    }


def write_review_outputs(
    out_root: Path,
    summary: dict[str, Any],
    selected: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> dict[str, Path]:
    outputs = {
        "selected_tasks_jsonl": out_root / "selected_tasks.jsonl",
        "reviews_jsonl": out_root / "reviews.jsonl",
        "reviews_csv": out_root / "reviews.csv",
        "errors_csv": out_root / "errors.csv",
        "summary_json": out_root / "summary.json",
    }
    write_jsonl(outputs["selected_tasks_jsonl"], selected)
    write_jsonl(outputs["reviews_jsonl"], reviews)
    write_csv(outputs["reviews_csv"], reviews)
    write_csv(outputs["errors_csv"], errors)
    xlsx_path = out_root / "transcript_quality_llm_review.xlsx"
    try:
        write_xlsx(xlsx_path, summary, reviews, errors)
        outputs["xlsx"] = xlsx_path
    except Exception as exc:  # noqa: BLE001
        (out_root / "xlsx_error.txt").write_text(str(exc), encoding="utf-8")
    return outputs


def persist_outputs(out_root: Path, reviews: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    write_jsonl(out_root / "reviews.jsonl", reviews)
    write_csv(out_root / "reviews.csv", reviews)
    write_csv(out_root / "errors.csv", errors)


def write_xlsx(path: Path, summary: dict[str, Any], reviews: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    import pandas as pd

    summary_rows: list[dict[str, Any]] = []
    for section in ("totals",):
        for key, value in summary.get(section, {}).items():
            summary_rows.append({"section": section, "metric": key, "value": value})
    for group, counts in summary.get("counts", {}).items():
        for key, value in counts.items():
            summary_rows.append({"section": group, "metric": key, "value": value})
    for note in summary.get("audit_notes", []):
        summary_rows.append({"section": "audit_note", "metric": "note", "value": note})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(reviews).to_excel(writer, sheet_name="Reviews", index=False)
        pd.DataFrame(errors).to_excel(writer, sheet_name="Errors", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column_cells in sheet.columns:
                max_len = 0
                col = column_cells[0].column_letter
                for cell in column_cells[:200]:
                    max_len = max(max_len, len(str(cell.value or "")))
                sheet.column_dimensions[col].width = min(max(max_len + 2, 10), 64)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def load_existing_reviews(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {_clean(row.get("task_id")): row for row in read_jsonl(path) if _clean(row.get("task_id"))}


def sorted_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(reviews, key=lambda row: str(row.get("task_id") or ""))


def make_batches(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(batch_size or 1))
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


def normalized_batch_size(config: TranscriptQualityLLMReviewConfig) -> int:
    try:
        return max(1, min(10, int(config.batch_size)))
    except (TypeError, ValueError):
        return 1


def _task_id(task: dict[str, Any]) -> str:
    return _clean(task.get("task_id") or task.get("id"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean(item) for item in value if _clean(item)]
    cleaned = _clean(value)
    return [cleaned] if cleaned else []


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "да"}


def _clamp_float(value: Any, lo: float, hi: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(lo, min(hi, number))


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run transcript-quality LLM review on JSONL tasks.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--provider", default="codex_cli")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-strategy", default="stratified", choices=["stratified", "first"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/llm_responses"))
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument("--codex-home", type=Path, default=None)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TranscriptQualityLLMReviewConfig:
    return TranscriptQualityLLMReviewConfig(
        project_root=args.project_root,
        input_jsonl=args.input_jsonl,
        out_root=args.out_root,
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        limit=args.limit,
        offset=args.offset,
        sample_strategy=args.sample_strategy,
        batch_size=args.batch_size,
        workers=args.workers,
        dry_run=args.dry_run,
        force=args.force,
        cache_enabled=not args.no_cache,
        cache_dir=args.cache_dir,
        timeout_sec=args.timeout_sec,
        codex_cli_command=args.codex_cli_command,
        codex_home=args.codex_home,
    )
