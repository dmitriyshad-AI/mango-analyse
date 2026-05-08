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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.services.llm_response_cache import LLMResponseCache


PROMPT_VERSION = "sales_moment_llm_review_v1"
REVIEW_SCHEMA_VERSION = "v1"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "medium"
SIGNAL_TAXONOMY = {
    "price_question",
    "price_objection",
    "discount_or_installment_question",
    "schedule_question",
    "format_question_online_offline",
    "location_question",
    "teacher_question",
    "program_question",
    "level_fit_question",
    "exam_or_olympiad_goal",
    "trust_question",
    "competitor_comparison",
    "child_motivation_concern",
    "parent_decision_delay",
    "spouse_or_family_approval",
    "not_relevant_now",
    "already_learning_elsewhere",
    "ready_to_pay",
    "materials_request",
    "callback_request",
    "technical_or_access_issue",
    "existing_client_progress",
    "complaint_or_service_risk",
    "payment_or_contract_service",
    "unknown",
}
STAGE_TAXONOMY = {
    "new_request",
    "discovery",
    "offer_explained",
    "price_discussion",
    "objection_handling",
    "materials_sent",
    "decision_wait",
    "payment_intent",
    "paid_or_enrolled",
    "existing_client_service",
    "reactivation",
    "lost_or_stalled",
    "retention_or_expansion",
    "unknown",
}
RUBRIC_KEYS = (
    "factual_correctness",
    "completeness",
    "persuasiveness",
    "personalization",
    "objection_handling",
    "next_step_clarity",
    "empathy_tone",
    "sales_discipline",
)


@dataclass(frozen=True)
class LLMReviewConfig:
    project_root: Path
    input_jsonl: Path
    out_root: Path
    provider: str = "openai"
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    limit: int = 160
    offset: int = 0
    sample_strategy: str = "stratified"
    dry_run: bool = False
    force: bool = False
    cache_enabled: bool = True
    cache_dir: Path = Path(".cache/llm_responses")
    timeout_sec: int = 180
    codex_cli_command: str = "codex"
    codex_home: Path | None = None
    codex_batch_size: int = 5


def run_pilot_sales_moment_llm_review(config: LLMReviewConfig) -> dict[str, Any]:
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    all_items = read_jsonl(config.input_jsonl)
    selected = select_review_items(all_items, limit=config.limit, offset=config.offset, strategy=config.sample_strategy)
    existing = load_existing_reviews(out_root / "reviews.jsonl") if not config.force else {}
    cache = LLMResponseCache(enabled=config.cache_enabled, root_dir=config.cache_dir)

    reviews: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    skipped_existing = 0
    cache_hits = 0
    dry_run_count = 0
    provider_stats: dict[str, int] = {
        "single_provider_calls": 0,
        "codex_batch_provider_calls": 0,
        "codex_batch_fallback_single_calls": 0,
    }

    def persist_progress() -> None:
        write_incremental_review_outputs(out_root, sorted_reviews(reviews), errors)

    def add_review(item: dict[str, Any], payload: dict[str, Any]) -> None:
        normalized = normalize_review_payload(payload, item, config)
        reviews.append(flatten_review(item, normalized, config))

    def cache_put(item: dict[str, Any], payload: dict[str, Any]) -> None:
        cache.put(
            namespace="sales_moment_llm_review",
            provider=config.provider,
            model=config.model,
            reasoning=config.reasoning_effort,
            prompt_version=PROMPT_VERSION,
            prompt=build_review_prompt(item),
            response=payload,
        )

    batch_items: list[dict[str, Any]] = []

    def flush_codex_batch() -> None:
        nonlocal provider_stats
        if not batch_items:
            return
        current_batch = list(batch_items)
        batch_items.clear()
        try:
            provider_stats["codex_batch_provider_calls"] += 1
            payload_by_id = call_codex_cli_batch_review_provider(config, current_batch)
            for batch_item in current_batch:
                item_id = _clean(batch_item.get("id"))
                payload = payload_by_id[item_id]
                cache_put(batch_item, payload)
                add_review(batch_item, payload)
            persist_progress()
        except Exception as batch_exc:  # noqa: BLE001
            if len(current_batch) == 1:
                item_id = _clean(current_batch[0].get("id"))
                errors.append({"id": item_id, "error": f"{type(batch_exc).__name__}: {batch_exc}"})
                persist_progress()
                return
            for batch_item in current_batch:
                item_id = _clean(batch_item.get("id"))
                try:
                    provider_stats["codex_batch_fallback_single_calls"] += 1
                    payload = call_review_provider(config, build_review_prompt(batch_item))
                    cache_put(batch_item, payload)
                    add_review(batch_item, payload)
                except Exception as single_exc:  # noqa: BLE001
                    errors.append(
                        {
                            "id": item_id,
                            "error": (
                                f"batch_failed={type(batch_exc).__name__}: {batch_exc}; "
                                f"single_failed={type(single_exc).__name__}: {single_exc}"
                            ),
                        }
                    )
                persist_progress()

    for item in selected:
        moment_id = _clean(item.get("id"))
        if moment_id in existing:
            reviews.append(existing[moment_id])
            skipped_existing += 1
            continue
        try:
            prompt = build_review_prompt(item)
            cached = cache.get(
                namespace="sales_moment_llm_review",
                provider=config.provider,
                model=config.model,
                reasoning=config.reasoning_effort,
                prompt_version=PROMPT_VERSION,
                prompt=prompt,
            )
            if cached is not None:
                payload = cached
                cache_hits += 1
            elif config.dry_run:
                payload = deterministic_review_payload(item)
                dry_run_count += 1
            else:
                if should_use_codex_batch(config):
                    batch_items.append(item)
                    if len(batch_items) >= normalized_codex_batch_size(config):
                        flush_codex_batch()
                    continue
                provider_stats["single_provider_calls"] += 1
                payload = call_review_provider(config, prompt)
                cache_put(item, payload)
            add_review(item, payload)
            persist_progress()
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": moment_id, "error": f"{type(exc).__name__}: {exc}"})
            persist_progress()

    flush_codex_batch()
    reviews = sorted_reviews(reviews)
    summary = build_review_summary(
        config,
        all_items,
        selected,
        reviews,
        errors,
        skipped_existing,
        cache_hits,
        dry_run_count,
        provider_stats,
    )
    outputs = write_review_outputs(out_root, summary, reviews, errors)
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
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL line {line_no}: {exc}") from exc
            if isinstance(item, dict):
                items.append(item)
    return items


def select_review_items(items: list[dict[str, Any]], *, limit: int, offset: int, strategy: str) -> list[dict[str, Any]]:
    sliced = items[max(offset, 0) :]
    if limit == 0:
        target = len(sliced)
    else:
        target = max(limit, 0)
    if target <= 0:
        return []
    if strategy == "first":
        return sliced[:target]
    if strategy != "stratified":
        raise ValueError(f"Unsupported sample strategy: {strategy}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in sliced:
        ctx = _dict(item.get("chain_context"))
        seed = _dict(item.get("deterministic_seed"))
        key = f"{ctx.get('extraction_use_case') or 'unknown'}::{seed.get('customer_signal_label') or 'unknown'}"
        groups[key].append(item)
    ordered_keys = sorted(groups, key=lambda key: (-len(groups[key]), key))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    while len(selected) < target and ordered_keys:
        progressed = False
        for key in list(ordered_keys):
            bucket = groups[key]
            if not bucket:
                ordered_keys.remove(key)
                continue
            item = bucket.pop(0)
            item_id = _clean(item.get("id"))
            if item_id and item_id not in seen:
                seen.add(item_id)
                selected.append(item)
                progressed = True
                if len(selected) >= target:
                    break
        if not progressed:
            break
    return selected


def build_review_prompt(item: dict[str, Any]) -> str:
    payload = {
        "instructions": {
            "role": "Ты эксперт по продажам EdTech и аудиту звонков отдела продаж.",
            "task": "Оцени один sales moment. Используй только данные из input. Не придумывай факты.",
            "language": "Русский.",
            "critical_rules": [
                "Не оценивай ответ менеджера лучше только потому, что итог клиента положительный.",
                "Если transcript противоречит deterministic_seed, доверяй transcript и history_summary.",
                "Отделяй сервисный вопрос действующего клиента от новой продажи.",
                "Идеальный ответ должен быть применим менеджером в реальном звонке, без канцелярита.",
            ],
            "signal_taxonomy": sorted(SIGNAL_TAXONOMY),
            "stage_taxonomy": sorted(STAGE_TAXONOMY),
            "rubric": {key: "integer 0-100" for key in RUBRIC_KEYS},
            "return_json_schema": review_json_schema_hint(),
        },
        "input": item,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def build_batch_review_prompt(items: list[dict[str, Any]]) -> str:
    payload = {
        "instructions": {
            "role": "Ты эксперт по продажам EdTech и аудиту звонков отдела продаж.",
            "task": (
                "Оцени каждый sales moment из inputs. Используй только данные конкретного input. "
                "Верни ровно один review на каждый moment_id."
            ),
            "language": "Русский.",
            "critical_rules": [
                "Не оценивай ответ менеджера лучше только потому, что итог клиента положительный.",
                "Если transcript противоречит deterministic_seed, доверяй transcript и history_summary.",
                "Отделяй сервисный вопрос действующего клиента от новой продажи.",
                "Не пропускай moment_id и не добавляй moment_id, которых нет во входе.",
                "Идеальный ответ должен быть применим менеджером в реальном звонке, без канцелярита.",
            ],
            "signal_taxonomy": sorted(SIGNAL_TAXONOMY),
            "stage_taxonomy": sorted(STAGE_TAXONOMY),
            "rubric": {key: "integer 0-100" for key in RUBRIC_KEYS},
            "return_json_schema": {
                "reviews": [
                    {
                        "moment_id": "same as input id",
                        **review_json_schema_hint(),
                    }
                ]
            },
        },
        "inputs": [{"moment_id": _clean(item.get("id")), "input": item} for item in items],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def review_json_schema_hint() -> dict[str, Any]:
    return {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "customer_question": "string",
        "customer_signal_type": "taxonomy value",
        "hidden_sales_stage": "taxonomy value",
        "manager_answer": "string",
        "rubric_scores": {key: 0 for key in RUBRIC_KEYS},
        "overall_quality_score": 0,
        "what_manager_did_well": ["string"],
        "what_manager_missed": ["string"],
        "ideal_reaction": "string",
        "ideal_answer_example": "string",
        "risk_flags": ["string"],
        "avoid_using_when": "string",
        "evidence_quotes": {"customer_quote": "string", "manager_quote": "string"},
        "extraction_confidence": 0.0,
    }


def call_review_provider(config: LLMReviewConfig, prompt: str) -> dict[str, Any]:
    if config.provider == "codex_cli":
        return call_codex_cli_review_provider(config, prompt)
    if config.provider != "openai":
        raise RuntimeError(f"Unsupported provider for live review: {config.provider}")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for live OpenAI review. Use --dry-run for local audit.")
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    instructions = "Return one valid JSON object only. No markdown."
    try:
        response = client.responses.create(
            model=config.model,
            instructions=instructions,
            input=prompt,
            reasoning={"effort": config.reasoning_effort},
            text={"format": {"type": "json_object"}},
            timeout=config.timeout_sec,
        )
        content = getattr(response, "output_text", "") or _response_output_text(response)
    except Exception:
        # Compatibility fallback for models/accounts where Responses JSON mode is unavailable.
        response = client.chat.completions.create(
            model=config.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt},
            ],
            timeout=config.timeout_sec,
        )
        content = response.choices[0].message.content if response.choices else ""
    if not content:
        raise RuntimeError("Review provider returned empty content")
    return extract_json_object(content)


def call_codex_cli_review_provider(config: LLMReviewConfig, prompt: str) -> dict[str, Any]:
    return call_codex_cli_json(
        config,
        (
            "Верни строго один JSON object по переданной схеме. "
            "Не используй markdown. Не запускай shell-команды. "
            "Работай только с данными prompt.\n\n"
            + prompt
        ),
        review_output_json_schema(),
    )


def call_codex_cli_batch_review_provider(config: LLMReviewConfig, items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not items:
        return {}
    payload = call_codex_cli_json(
        config,
        (
            "Верни строго один JSON object по переданной схеме. "
            "Поле reviews должно содержать ровно один review на каждый входной moment_id. "
            "Не используй markdown. Не запускай shell-команды. "
            "Работай только с данными prompt.\n\n"
            + build_batch_review_prompt(items)
        ),
        batch_review_output_json_schema(),
    )
    return extract_batch_review_payloads(payload, [_clean(item.get("id")) for item in items])


def call_codex_cli_json(config: LLMReviewConfig, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    codex_bin = (config.codex_cli_command or "codex").strip() or "codex"
    if shutil.which(codex_bin) is None:
        raise RuntimeError(f"codex binary is not available: {codex_bin}")
    with tempfile.NamedTemporaryFile(prefix="sales_moment_review_", suffix=".json") as out_file, tempfile.NamedTemporaryFile(
        prefix="sales_moment_review_schema_",
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


def build_codex_cli_command(config: LLMReviewConfig, *, output_path: Path, schema_path: Path) -> list[str]:
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
    reasoning = (config.reasoning_effort or "").strip().lower()
    if reasoning:
        cmd.extend(["-c", f'model_reasoning_effort="{reasoning}"'])
    cmd.append("-")
    return cmd


def codex_cli_env(config: LLMReviewConfig) -> dict[str, str]:
    env = os.environ.copy()
    if config.codex_home is not None:
        env["CODEX_HOME"] = str(config.codex_home.expanduser().resolve())
    return env


def review_output_json_schema() -> dict[str, Any]:
    return review_payload_json_schema(include_moment_id=False)


def batch_review_output_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["reviews"],
        "properties": {
            "reviews": {
                "type": "array",
                "minItems": 1,
                "items": review_payload_json_schema(include_moment_id=True),
            }
        },
    }


def review_payload_json_schema(*, include_moment_id: bool) -> dict[str, Any]:
    rubric_properties = {key: {"type": "integer", "minimum": 0, "maximum": 100} for key in RUBRIC_KEYS}
    required = [
        "review_schema_version",
        "customer_question",
        "customer_signal_type",
        "hidden_sales_stage",
        "manager_answer",
        "rubric_scores",
        "overall_quality_score",
        "what_manager_did_well",
        "what_manager_missed",
        "ideal_reaction",
        "ideal_answer_example",
        "risk_flags",
        "avoid_using_when",
        "evidence_quotes",
        "extraction_confidence",
    ]
    properties: dict[str, Any] = {
        "review_schema_version": {"type": "string"},
        "customer_question": {"type": "string"},
        "customer_signal_type": {"type": "string", "enum": sorted(SIGNAL_TAXONOMY)},
        "hidden_sales_stage": {"type": "string", "enum": sorted(STAGE_TAXONOMY)},
        "manager_answer": {"type": "string"},
        "rubric_scores": {
            "type": "object",
            "additionalProperties": False,
            "required": list(RUBRIC_KEYS),
            "properties": rubric_properties,
        },
        "overall_quality_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "what_manager_did_well": {"type": "array", "items": {"type": "string"}},
        "what_manager_missed": {"type": "array", "items": {"type": "string"}},
        "ideal_reaction": {"type": "string"},
        "ideal_answer_example": {"type": "string"},
        "risk_flags": {"type": "array", "items": {"type": "string"}},
        "avoid_using_when": {"type": "string"},
        "evidence_quotes": {
            "type": "object",
            "additionalProperties": False,
            "required": ["customer_quote", "manager_quote"],
            "properties": {
                "customer_quote": {"type": "string"},
                "manager_quote": {"type": "string"},
            },
        },
        "extraction_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    }
    if include_moment_id:
        required = ["moment_id", *required]
        properties = {"moment_id": {"type": "string"}, **properties}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def deterministic_review_payload(item: dict[str, Any]) -> dict[str, Any]:
    seed = _dict(item.get("deterministic_seed"))
    chain = _dict(item.get("chain_context"))
    call = _dict(item.get("call_context"))
    signal = map_seed_signal(seed.get("customer_signal_label"))
    stage = map_seed_stage(seed.get("hidden_sales_stage"), chain.get("extraction_use_case"))
    base_score = _clamp_int(seed.get("manager_response_quality_score"), 0, 100, 55)
    rubric = {
        "factual_correctness": min(100, base_score + 10),
        "completeness": base_score,
        "persuasiveness": max(0, base_score - 5),
        "personalization": base_score if call.get("history_summary") else max(0, base_score - 10),
        "objection_handling": base_score if signal in {"price_question", "price_objection", "schedule_question", "trust_question"} else max(0, base_score - 5),
        "next_step_clarity": 80 if "следующ" in str(seed.get("manager_answer_or_reaction") or "").lower() or "перезвон" in str(seed.get("manager_answer_or_reaction") or "").lower() else max(0, base_score - 10),
        "empathy_tone": max(0, base_score - 5),
        "sales_discipline": base_score,
    }
    return {
        "review_schema_version": REVIEW_SCHEMA_VERSION,
        "customer_question": _clean(seed.get("customer_question_or_need")),
        "customer_signal_type": signal,
        "hidden_sales_stage": stage,
        "manager_answer": _clean(seed.get("manager_answer_or_reaction")),
        "rubric_scores": rubric,
        "overall_quality_score": int(round(sum(rubric.values()) / len(rubric))),
        "what_manager_did_well": ["deterministic_seed: зафиксирована основная реакция менеджера"],
        "what_manager_missed": ["dry_run: требуется LLM-review для экспертной проверки"],
        "ideal_reaction": _clean(seed.get("ideal_manager_reaction")),
        "ideal_answer_example": _clean(seed.get("ideal_answer_template")),
        "risk_flags": ["dry_run_not_llm_review"],
        "avoid_using_when": "Не использовать как финальную экспертную оценку без live LLM-review.",
        "evidence_quotes": {
            "customer_quote": _clean(seed.get("customer_question_or_need"))[:300],
            "manager_quote": _clean(seed.get("manager_answer_or_reaction"))[:300],
        },
        "extraction_confidence": 0.45,
    }


def normalize_review_payload(payload: dict[str, Any], item: dict[str, Any], config: LLMReviewConfig) -> dict[str, Any]:
    seed = _dict(item.get("deterministic_seed"))
    normalized: dict[str, Any] = {}
    normalized["review_schema_version"] = _clean(payload.get("review_schema_version")) or REVIEW_SCHEMA_VERSION
    normalized["customer_question"] = _clean(payload.get("customer_question")) or _clean(seed.get("customer_question_or_need"))
    normalized["customer_signal_type"] = normalize_taxonomy(payload.get("customer_signal_type"), SIGNAL_TAXONOMY, map_seed_signal(seed.get("customer_signal_label")))
    normalized["hidden_sales_stage"] = normalize_taxonomy(payload.get("hidden_sales_stage"), STAGE_TAXONOMY, map_seed_stage(seed.get("hidden_sales_stage"), _dict(item.get("chain_context")).get("extraction_use_case")))
    normalized["manager_answer"] = _clean(payload.get("manager_answer")) or _clean(seed.get("manager_answer_or_reaction"))
    rubric_raw = _dict(payload.get("rubric_scores"))
    normalized["rubric_scores"] = {key: _clamp_int(rubric_raw.get(key), 0, 100, 0) for key in RUBRIC_KEYS}
    if not any(normalized["rubric_scores"].values()):
        normalized["rubric_scores"] = deterministic_review_payload(item)["rubric_scores"]
    normalized["overall_quality_score"] = _clamp_int(
        payload.get("overall_quality_score"),
        0,
        100,
        int(round(sum(normalized["rubric_scores"].values()) / len(normalized["rubric_scores"]))),
    )
    normalized["what_manager_did_well"] = _string_list(payload.get("what_manager_did_well"))
    normalized["what_manager_missed"] = _string_list(payload.get("what_manager_missed"))
    normalized["ideal_reaction"] = _clean(payload.get("ideal_reaction")) or _clean(seed.get("ideal_manager_reaction"))
    normalized["ideal_answer_example"] = _clean(payload.get("ideal_answer_example")) or _clean(seed.get("ideal_answer_template"))
    normalized["risk_flags"] = _string_list(payload.get("risk_flags"))
    normalized["avoid_using_when"] = _clean(payload.get("avoid_using_when"))
    quotes = _dict(payload.get("evidence_quotes"))
    normalized["evidence_quotes"] = {
        "customer_quote": _clean(quotes.get("customer_quote")),
        "manager_quote": _clean(quotes.get("manager_quote")),
    }
    normalized["extraction_confidence"] = _clamp_float(payload.get("extraction_confidence"), 0.0, 1.0, 0.0)
    normalized["provider"] = "dry_run" if config.dry_run else config.provider
    normalized["model"] = config.model
    normalized["reasoning_effort"] = config.reasoning_effort
    normalized["prompt_version"] = PROMPT_VERSION
    return normalized


def flatten_review(item: dict[str, Any], review: dict[str, Any], config: LLMReviewConfig) -> dict[str, Any]:
    chain = _dict(item.get("chain_context"))
    call = _dict(item.get("call_context"))
    seed = _dict(item.get("deterministic_seed"))
    rubric = _dict(review.get("rubric_scores"))
    quotes = _dict(review.get("evidence_quotes"))
    row = {
        "moment_id": item.get("id", ""),
        "provider": review.get("provider", ""),
        "model": review.get("model", ""),
        "reasoning_effort": review.get("reasoning_effort", ""),
        "review_schema_version": review.get("review_schema_version", ""),
        "phone": chain.get("phone", ""),
        "source_filename": call.get("source_filename", ""),
        "started_at": call.get("started_at", ""),
        "manager_name": call.get("manager_name", ""),
        "call_type": call.get("call_type", ""),
        "extraction_use_case": chain.get("extraction_use_case", ""),
        "final_outcome_label": chain.get("final_outcome_label", ""),
        "outcome_confidence_tier": chain.get("outcome_confidence_tier", ""),
        "deterministic_signal": seed.get("customer_signal_label", ""),
        "llm_customer_signal_type": review.get("customer_signal_type", ""),
        "deterministic_stage": seed.get("hidden_sales_stage", ""),
        "llm_hidden_sales_stage": review.get("hidden_sales_stage", ""),
        "customer_question": review.get("customer_question", ""),
        "manager_answer": review.get("manager_answer", ""),
        "overall_quality_score": review.get("overall_quality_score", ""),
        "extraction_confidence": review.get("extraction_confidence", ""),
        "what_manager_did_well": " | ".join(_string_list(review.get("what_manager_did_well"))),
        "what_manager_missed": " | ".join(_string_list(review.get("what_manager_missed"))),
        "ideal_reaction": review.get("ideal_reaction", ""),
        "ideal_answer_example": review.get("ideal_answer_example", ""),
        "risk_flags": " | ".join(_string_list(review.get("risk_flags"))),
        "avoid_using_when": review.get("avoid_using_when", ""),
        "customer_quote": quotes.get("customer_quote", ""),
        "manager_quote": quotes.get("manager_quote", ""),
        "history_summary": call.get("history_summary", ""),
    }
    for key in RUBRIC_KEYS:
        row[f"rubric_{key}"] = rubric.get(key, "")
    return row


def build_review_summary(
    config: LLMReviewConfig,
    all_items: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    skipped_existing: int,
    cache_hits: int,
    dry_run_count: int,
    provider_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    scores = [_clamp_int(row.get("overall_quality_score"), 0, 100, 0) for row in reviews]
    provider_stats = provider_stats or {}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_jsonl": str(config.input_jsonl.resolve()),
        "config": {
            "provider": config.provider,
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "codex_cli_command": config.codex_cli_command if config.provider == "codex_cli" else "",
            "codex_home": str(config.codex_home.resolve()) if config.provider == "codex_cli" and config.codex_home else "",
            "codex_batch_size": normalized_codex_batch_size(config) if config.provider == "codex_cli" else 1,
            "limit": config.limit,
            "offset": config.offset,
            "sample_strategy": config.sample_strategy,
            "dry_run": config.dry_run,
            "force": config.force,
            "cache_enabled": config.cache_enabled,
        },
        "totals": {
            "input_items": len(all_items),
            "selected_items": len(selected),
            "reviews_written": len(reviews),
            "errors": len(errors),
            "skipped_existing": skipped_existing,
            "cache_hits": cache_hits,
            "dry_run_reviews": dry_run_count,
            "unique_phones": len({row.get("phone") for row in reviews if row.get("phone")}),
            "single_provider_calls": int(provider_stats.get("single_provider_calls", 0)),
            "codex_batch_provider_calls": int(provider_stats.get("codex_batch_provider_calls", 0)),
            "codex_batch_fallback_single_calls": int(provider_stats.get("codex_batch_fallback_single_calls", 0)),
        },
        "quality": {
            "avg_overall_quality_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "low_quality_count_lt_55": sum(1 for score in scores if score < 55),
            "high_quality_count_gte_75": sum(1 for score in scores if score >= 75),
        },
        "counts": {
            "by_use_case": dict(Counter(row.get("extraction_use_case", "") for row in reviews).most_common()),
            "by_signal": dict(Counter(row.get("llm_customer_signal_type", "") for row in reviews).most_common()),
            "by_stage": dict(Counter(row.get("llm_hidden_sales_stage", "") for row in reviews).most_common()),
            "by_outcome": dict(Counter(row.get("final_outcome_label", "") for row in reviews).most_common()),
            "by_provider": dict(Counter(row.get("provider", "") for row in reviews).most_common()),
            "top_managers": dict(Counter(row.get("manager_name", "") for row in reviews if row.get("manager_name")).most_common(30)),
            "risk_flags": dict(_count_pipe_values(row.get("risk_flags", "") for row in reviews).most_common()),
        },
        "audit_notes": [
            "Dry-run rows are structural placeholders, not live LLM expert reviews.",
            "Use --provider codex_cli --model gpt-5.5 --reasoning-effort medium without --dry-run to use the local Codex CLI subscription.",
            "Use --provider openai --model gpt-5.5 --reasoning-effort medium without --dry-run after OPENAI_API_KEY is available.",
            "Stratified sampling avoids reviewing only the first reactivation-heavy rows.",
        ],
    }


def write_review_outputs(out_root: Path, summary: dict[str, Any], reviews: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Path]:
    paths = {
        "reviews_jsonl": out_root / "reviews.jsonl",
        "reviews_csv": out_root / "reviews.csv",
        "errors_csv": out_root / "errors.csv",
        "summary_json": out_root / "summary.json",
    }
    with paths["reviews_jsonl"].open("w", encoding="utf-8") as fh:
        for row in reviews:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(paths["reviews_csv"], reviews)
    write_csv(paths["errors_csv"], errors)
    xlsx_path = out_root / "sales_moment_llm_review.xlsx"
    try:
        write_xlsx(xlsx_path, summary, reviews, errors)
        paths["xlsx"] = xlsx_path
    except Exception as exc:  # noqa: BLE001
        (out_root / "xlsx_error.txt").write_text(str(exc), encoding="utf-8")
    return paths


def write_incremental_review_outputs(out_root: Path, reviews: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    with (out_root / "reviews.jsonl").open("w", encoding="utf-8") as fh:
        for row in reviews:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(out_root / "reviews.csv", reviews)
    write_csv(out_root / "errors.csv", errors)


def sorted_reviews(reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(reviews, key=lambda row: str(row.get("moment_id") or ""))


def write_xlsx(path: Path, summary: dict[str, Any], reviews: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    import pandas as pd

    summary_rows: list[dict[str, Any]] = []
    for section in ("totals", "quality"):
        for key, value in summary.get(section, {}).items():
            summary_rows.append({"section": section, "metric": key, "value": value})
    for group, counts in summary.get("counts", {}).items():
        for label, count in counts.items():
            summary_rows.append({"section": group, "metric": label, "value": count})
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


def load_existing_reviews(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {_clean(row.get("moment_id")): row for row in rows if _clean(row.get("moment_id"))}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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
    expected = [item_id for item_id in expected_ids if item_id]
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("Batch input contains duplicate moment_id values")
    rows = payload.get("reviews")
    if not isinstance(rows, list):
        raise ValueError("Batch review payload must contain reviews array")
    by_id: dict[str, dict[str, Any]] = {}
    unexpected: list[str] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Batch review row {idx} must be a JSON object")
        moment_id = _clean(row.get("moment_id"))
        if not moment_id:
            raise ValueError(f"Batch review row {idx} has empty moment_id")
        if moment_id not in expected_set:
            unexpected.append(moment_id)
            continue
        if moment_id in by_id:
            raise ValueError(f"Batch review has duplicate moment_id: {moment_id}")
        cleaned = dict(row)
        cleaned.pop("moment_id", None)
        by_id[moment_id] = cleaned
    missing = [item_id for item_id in expected if item_id not in by_id]
    if unexpected or missing:
        parts = []
        if missing:
            parts.append(f"missing={missing[:10]}")
        if unexpected:
            parts.append(f"unexpected={unexpected[:10]}")
        raise ValueError("Batch review moment_id mismatch: " + "; ".join(parts))
    return by_id


def normalized_codex_batch_size(config: LLMReviewConfig) -> int:
    try:
        return max(1, min(10, int(config.codex_batch_size)))
    except (TypeError, ValueError):
        return 1


def should_use_codex_batch(config: LLMReviewConfig) -> bool:
    return config.provider == "codex_cli" and not config.dry_run and normalized_codex_batch_size(config) > 1


def _response_output_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks)


def normalize_taxonomy(value: Any, allowed: set[str], fallback: str) -> str:
    text = _clean(value).lower()
    if text in allowed:
        return text
    return fallback if fallback in allowed else "unknown"


def map_seed_signal(value: Any) -> str:
    text = _clean(value)
    return {
        "price_or_payment": "price_question",
        "schedule_or_format_constraint": "schedule_question",
        "trust_or_quality_question": "trust_question",
        "payment_service": "payment_or_contract_service",
        "refusal_or_cooling": "not_relevant_now",
        "next_year_interest": "parent_decision_delay",
        "product_interest": "program_question",
        "service_or_existing_client": "existing_client_progress",
    }.get(text, "unknown")


def map_seed_stage(value: Any, use_case: Any = "") -> str:
    text = _clean(value)
    mapped = {
        "reactivation_after_lost_deal": "reactivation",
        "service_retention_or_expansion": "existing_client_service",
        "loss_or_churn_path": "lost_or_stalled",
        "success_path_validation": "paid_or_enrolled",
        "objection_handling": "objection_handling",
        "first_contact_or_need_discovery": "discovery",
        "decision_or_follow_up": "decision_wait",
        "nurture_and_qualification": "discovery",
    }.get(text)
    if mapped:
        return mapped
    if _clean(use_case) == "reactivation_revenue":
        return "reactivation"
    return "unknown"


def _count_pipe_values(values: Iterable[Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for value in values:
        for item in str(value or "").split("|"):
            cleaned = _clean(item)
            if cleaned:
                counter[cleaned] += 1
    return counter


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean(item) for item in value if _clean(item)]
    cleaned = _clean(value)
    return [cleaned] if cleaned else []


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        parsed = int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM rubric review for pilot sales moments.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--input-jsonl", default="stable_runtime/pilot_sales_moments_20260507/llm_sales_moment_input.jsonl")
    parser.add_argument("--out-root", default="stable_runtime/pilot_sales_moment_llm_review_20260507_calibration")
    parser.add_argument("--provider", default="openai", choices=["openai", "codex_cli"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--limit", type=int, default=160, help="0 means all remaining rows")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-strategy", default="stratified", choices=["stratified", "first"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-dir", default=".cache/llm_responses")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--codex-cli-command", default="codex")
    parser.add_argument("--codex-home", default="")
    parser.add_argument("--codex-batch-size", type=int, default=5, help="Codex CLI reviews per exec call, clamped to 1..10")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> LLMReviewConfig:
    project_root = Path(args.project_root).expanduser().resolve()
    return LLMReviewConfig(
        project_root=project_root,
        input_jsonl=(project_root / args.input_jsonl).resolve(),
        out_root=(project_root / args.out_root).resolve(),
        provider=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        limit=int(args.limit),
        offset=int(args.offset),
        sample_strategy=args.sample_strategy,
        dry_run=bool(args.dry_run),
        force=bool(args.force),
        cache_enabled=not bool(args.no_cache),
        cache_dir=(project_root / args.cache_dir).resolve(),
        timeout_sec=int(args.timeout_sec),
        codex_cli_command=str(args.codex_cli_command or "codex"),
        codex_home=(project_root / str(args.codex_home)).resolve() if str(args.codex_home or "").strip() else None,
        codex_batch_size=int(args.codex_batch_size),
    )


__all__ = [
    "LLMReviewConfig",
    "batch_review_output_json_schema",
    "build_codex_cli_command",
    "build_batch_review_prompt",
    "build_review_prompt",
    "codex_cli_env",
    "config_from_args",
    "call_codex_cli_batch_review_provider",
    "call_codex_cli_review_provider",
    "deterministic_review_payload",
    "extract_batch_review_payloads",
    "flatten_review",
    "map_seed_signal",
    "map_seed_stage",
    "normalize_review_payload",
    "normalized_codex_batch_size",
    "parse_args",
    "read_jsonl",
    "review_output_json_schema",
    "should_use_codex_batch",
    "run_pilot_sales_moment_llm_review",
    "select_review_items",
]
