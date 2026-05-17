#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence

from mango_mvp.channels.pilot_context import build_pilot_context
from mango_mvp.channels.subscription_llm import (
    SubscriptionDraftResult,
    SubscriptionLlmDraftProvider,
    apply_brand_separation_guard,
    apply_input_policy_guards,
    apply_payment_confirmation_guard,
    apply_unsupported_promise_guard,
    normalize_subscription_draft_payload,
)
from mango_mvp.knowledge_base.fact_registry import FRESHNESS_FRESH, classify_fact_types
from mango_mvp.knowledge_base.kc_context import build_kc_context
from mango_mvp.question_catalog.classifier import load_valid_theme_and_service_ids

try:
    from mango_mvp.channels.telegram_pilot_context_builder import build_telegram_pilot_context_from_snapshot
except Exception:  # pragma: no cover - compatibility path for local parallel work
    build_telegram_pilot_context_from_snapshot = None


DEFAULT_BASELINE = Path(
    ".codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/"
    "llm_drafts_stage6_taxonomy_20260517/stage6_llm_drafts_for_manual_review.csv"
)
DEFAULT_INPUT = Path(".codex_local/telegram_pilot/eval_packs/20260517_contextual_layer_smoke/private_dialog_threads.jsonl")
DEFAULT_SNAPSHOT = Path("product_data/knowledge_base/kb_night_20260517_v1/kc_snapshot_20260517_night_v1.json")
DEFAULT_OUT_DIR = Path("product_data/knowledge_base/kb_night_20260517_v1")
STAGE6_KB_EVAL_SCHEMA_VERSION = "telegram_stage6_kb_eval_v1"
FRESH_STATUSES = {"fresh", "fresh_verified"}
SUBSTANTIVE_RE = re.compile(r"\?|подскаж|можно|когда|сколько|стоим|цен|оплат|возврат|распис|ссылк|доступ|курс|программ", re.I)
TRIVIAL_RE = re.compile(r"^(спасибо.*|ок|окей|хорошо|да|нет|поняла|понял|ясно)[.!?\s]*$", re.I)
HIGH_RISK_RE = re.compile(r"возврат|вернуть\s+деньги|суд|иск|претензи|роспотребнадзор|жалоб|нарушили\s+права", re.I)
EMPTY_CLARIFICATION_RE = re.compile(r"\b(уточним|проверим|передам|верн[её]мся|свяжемся)\b", re.I)
WORD_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{4,}")
STAGE6_DRAFT_FALLBACK_TEXT = (
    "Здравствуйте! Передам вопрос менеджеру: он сверит актуальные данные и вернется с ответом."
)
STAGE6_INTERNAL_METADATA_BLOCK_RE = re.compile(
    r"\[[^\]\n]{0,260}?"
    r"(?:\b(?:source|source_id|fact|fact_id|freshness|freshness_status)\s*[:=]"
    r"|(?:source|fact|kc_chunk):[A-Za-z0-9_:\-]+)"
    r"[^\]\n]{0,260}\]\s*",
    re.I,
)
STAGE6_INTERNAL_METADATA_TOKEN_RE = re.compile(
    r"\b(?:source|source_id|fact|fact_id|freshness|freshness_status)\s*[:=]\s*[^\s;\],.]+"
    r"|(?:source|fact|kc_chunk):[A-Za-z0-9_:\-]+",
    re.I,
)
STAGE6_FORBIDDEN_DRAFT_MARKER_RE = re.compile(
    r"\b(?:source_id|fact_id|freshness_status|AMO|Tallanto|GPT|Claude|Codex)\b"
    r"|\bя\s+(?:бот|ии)\b",
    re.I,
)
STAGE6_BRAND_FORBIDDEN_TERMS = {
    "foton": ("унпк", "унпк мфти", "ано дпо", "ноу унпк", "kmipt.ru"),
    "unpk": ("фотон", "цдпо", "црдо", "cdpofoton", "т-банк", "долями"),
}


@dataclass(frozen=True)
class Stage6KbEvalResult:
    rows_total: int
    used_kb_context: int
    became_more_substantive: int
    enriched_csv_path: str
    enriched_xlsx_path: str
    comparison_csv_path: str
    comparison_summary_path: str
    summary_json_path: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


class FakeStage6Provider:
    def build_draft(self, client_message: str, *, context: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        snippets = list((context or {}).get("knowledge_snippets") or [])
        suffix = " На проверке есть материалы базы знаний." if snippets else ""
        return {
            "message_type": "question",
            "broad_group": "support",
            "topic_id": "service:S3_other_or_low_confidence",
            "confidence_theme": 0.72,
            "confidence_group": 0.8,
            "route": "draft_for_manager",
            "draft_text": (
                "Здравствуйте! Вижу ваш вопрос. Менеджер сверит актуальные данные по вашей ситуации "
                f"и подскажет следующий шаг.{suffix}"
            ),
            "manager_checklist": ["Проверить актуальные факты перед отправкой клиенту"],
            "missing_facts": list((context or {}).get("missing_facts") or []),
            "safety_flags": ["manager_approval_required", "no_auto_send"],
            "context_used": ["knowledge_snippets"] if snippets else [],
            "context_warnings": list((context or {}).get("context_warnings") or []),
        }


def stage6_kb_eval_safety_contract() -> dict[str, bool]:
    return {
        "live_telegram": False,
        "client_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_stable_runtime": False,
        "run_asr": False,
        "run_resolve_analyze": False,
    }


def run_stage6_kb_eval(
    *,
    input_path: str | Path = DEFAULT_INPUT,
    snapshot_path: str | Path = DEFAULT_SNAPSHOT,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    baseline_csv_path: str | Path = DEFAULT_BASELINE,
    provider_mode: str = "fake",
    provider: Any | None = None,
    model: str = "gpt-5.5",
    reasoning_effort: str = "xhigh",
    timeout_sec: int = 240,
    expected_dialogs: int | None = 20,
) -> Stage6KbEvalResult:
    output_root = Path(out_dir)
    if "stable_runtime" in output_root.resolve().parts:
        raise ValueError("Refusing to write Stage 6 KB eval under stable_runtime")
    dialogs = read_jsonl(Path(input_path))
    if expected_dialogs is not None and len(dialogs) != expected_dialogs:
        raise ValueError(
            f"Stage 6 KB eval requires a fixed sample of {expected_dialogs} dialogs; got {len(dialogs)}"
        )
    snapshot = read_json(Path(snapshot_path))
    baseline_rows = read_csv(Path(baseline_csv_path)) if Path(baseline_csv_path).exists() else []
    baseline_by_dialog = {str(row.get("dialog_id") or ""): row for row in baseline_rows}
    draft_provider = provider or FakeStage6Provider()

    enriched_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for dialog in dialogs:
        dialog_id = str(dialog.get("dialog_id") or dialog.get("id") or "")
        baseline = baseline_by_dialog.get(dialog_id, {})
        message = select_target_client_message(dialog, baseline)
        client_message = str(message.get("text") or "")
        topic_id = str(baseline.get("topic_id") or baseline.get("theme_id") or "service:S3_other_or_low_confidence")
        context = build_telegram_pilot_context_from_snapshot(
            client_message,
            kc_snapshot=snapshot,
            topic_id=topic_id,
            required_fact_keys=required_fact_keys_for_topic(topic_id, client_message),
            rop_policy={"topic_id": topic_id, "bot_permission": "draft_for_manager"},
            recent_messages=recent_messages_for_dialog(dialog, target_message_id=str(message.get("message_id") or "")),
        )
        prompt_context = dict(context.to_prompt_context())
        prompt_context.setdefault("pilot_context_safety", stage6_kb_eval_safety_contract() | {"send_client_message": False})
        facts_context = dict(prompt_context.get("facts_context") or {})
        facts_context.setdefault("snapshot_run_id", snapshot_run_id(snapshot))
        prompt_context["facts_context"] = facts_context
        snippets = list(prompt_context.get("knowledge_snippets") or [])
        snippets = supplement_snapshot_snippets(snapshot, snippets, max_snippets=2)
        prompt_context["knowledge_snippets"] = snippets
        response = normalize_provider_response(draft_provider.build_draft(client_message, context=prompt_context))
        selected_count = len(snippets)
        used_kb = selected_count > 0
        old_draft = str(baseline.get("draft_text") or "")
        new_draft = str(response.get("draft_text") or "")
        more_substantive = is_more_substantive(old_draft, new_draft)
        empty_reduced = bool(EMPTY_CLARIFICATION_RE.search(old_draft)) and not bool(
            EMPTY_CLARIFICATION_RE.search(new_draft)
        )
        enriched_rows.append(
            {
                "dialog_id": dialog_id,
                "target_message_id": message.get("message_id") or "",
                "client_message": client_message,
                "snapshot_run_id": snapshot_run_id(snapshot),
                "selected_chunk_count": selected_count,
                "used_kb_context": used_kb,
                "knowledge_snippets": "\n".join(snippets),
                "facts_context_json": json.dumps(facts_context, ensure_ascii=False, sort_keys=True),
                "message_type": response.get("message_type", ""),
                "topic_id": response.get("topic_id", ""),
                "route": response.get("route", ""),
                "draft_text": new_draft,
                "manager_checklist": "|".join(response.get("manager_checklist") or []),
                "safety_flags": "|".join(response.get("safety_flags") or []),
            }
        )
        comparison_rows.append(
            {
                "dialog_id": dialog_id,
                "target_message_id": message.get("message_id") or "",
                "old_draft_len": len(old_draft),
                "new_draft_len": len(new_draft),
                "draft_became_more_substantive": more_substantive,
                "empty_clarification_reduced": empty_reduced,
                "used_kb_context": used_kb,
                "unsupported_numeric_promises": 0,
                "requires_manual_review": response.get("route") == "manager_only",
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    enriched_csv = output_root / "stage6_kb_enriched_drafts.csv"
    enriched_xlsx = output_root / "stage6_kb_enriched_drafts.xlsx"
    comparison_csv = output_root / "stage6_before_after_comparison.csv"
    comparison_md = output_root / "stage6_before_after_summary.md"
    summary_json = output_root / "stage6_eval_summary.json"
    write_csv(enriched_csv, enriched_rows)
    try_write_xlsx(enriched_rows, enriched_xlsx)
    write_csv(comparison_csv, comparison_rows)
    result = Stage6KbEvalResult(
        rows_total=len(enriched_rows),
        used_kb_context=sum(1 for row in enriched_rows if row["used_kb_context"]),
        became_more_substantive=sum(1 for row in comparison_rows if row["draft_became_more_substantive"]),
        enriched_csv_path=str(enriched_csv),
        enriched_xlsx_path=str(enriched_xlsx),
        comparison_csv_path=str(comparison_csv),
        comparison_summary_path=str(comparison_md),
        summary_json_path=str(summary_json),
    )
    comparison_md.write_text(render_stage6_summary(result, comparison_rows), encoding="utf-8")
    summary_payload = {
        **result.to_json_dict(),
        "schema_version": STAGE6_KB_EVAL_SCHEMA_VERSION,
        "safety": stage6_kb_eval_safety_contract(),
        "provider_mode": provider_mode,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "timeout_sec": timeout_sec,
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run safe Stage 6 KB enriched dry evaluation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline", "--baseline-csv", dest="baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--provider", choices=("fake", "codex"), default="fake")
    parser.add_argument("--active-brand", choices=("foton", "unpk", "unknown"), default="foton")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--expected-dialogs", type=int, default=20)
    args = parser.parse_args()

    result = run_stage6_kb_eval(
        input_path=args.input,
        snapshot_path=args.snapshot,
        out_dir=args.out_dir,
        baseline_csv_path=args.baseline,
        active_brand=args.active_brand,
        provider_mode=args.provider,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        timeout_sec=args.timeout_sec,
        expected_dialogs=args.expected_dialogs,
    )
    print(json.dumps(result.to_json_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


class DraftProvider(Protocol):
    def build_draft(self, client_message: str, *, context: Optional[Mapping[str, Any]] = None) -> Any:
        ...


@dataclass(frozen=True)
class Stage6KbEvalResult:
    out_dir: str
    enriched_csv_path: str
    enriched_xlsx_path: str
    comparison_csv_path: str
    comparison_summary_path: str
    rows_total: int
    snapshot_path: str
    snapshot_run_id: str
    provider_mode: str
    used_kb_context: int
    became_more_substantive: int
    empty_clarification_reduced: int
    invalid_topic_ids: int
    unsupported_numeric_promises: int
    brand_separation_violation: int
    high_risk_route_relaxed: int
    baseline_manager_only_relaxed: int
    baseline_manager_only_preserved: int
    errors: int

    def to_json_dict(self) -> Mapping[str, Any]:
        return {"schema_version": STAGE6_KB_EVAL_SCHEMA_VERSION, **asdict(self), "safety": stage6_kb_eval_safety_contract()}


def run_stage6_kb_eval(
    *,
    input_path: Path | str,
    snapshot_path: Path | str,
    out_dir: Path | str,
    baseline_csv_path: Path | str | None = None,
    active_brand: str = "foton",
    provider_mode: str = "fake",
    provider: DraftProvider | None = None,
    model: str = "gpt-5.5",
    reasoning_effort: str = "xhigh",
    timeout_sec: int = 240,
    expected_dialogs: int = 20,
) -> Stage6KbEvalResult:
    target = Path(out_dir)
    guard_safe_output_dir(target)
    dialogs = read_jsonl(Path(input_path))
    if expected_dialogs > 0 and len(dialogs) != expected_dialogs:
        raise ValueError(f"Stage 6 fixed sample must contain exactly {expected_dialogs} dialogs, got {len(dialogs)}")

    snapshot = load_snapshot(Path(snapshot_path))
    baseline = read_baseline_index(Path(baseline_csv_path)) if baseline_csv_path else {}
    target.mkdir(parents=True, exist_ok=True)
    draft_provider = provider or build_draft_provider(
        provider_mode=provider_mode,
        model=model,
        reasoning_effort=reasoning_effort,
        timeout_sec=timeout_sec,
        cache_dir=target / "llm_cache",
    )

    enriched_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for run_index, dialog in enumerate(dialogs, start=1):
        old_row = find_baseline_for_dialog(dialog, baseline)
        prepared = prepare_eval_dialog(dialog, snapshot=snapshot, baseline=old_row, active_brand=active_brand)
        result = build_guarded_draft(draft_provider, prepared["current_client_message"], context=prepared["pilot_context"])
        enriched = build_stage6_enriched_row(prepared, result.to_json_dict(), provider_mode=provider_mode, run_index=run_index)
        enriched = enforce_stage6_route_safety(enriched, old_row)
        comparison = build_stage6_comparison_row(enriched, old_row)
        enriched_rows.append(enriched)
        comparison_rows.append(comparison)

    enriched_csv = target / "stage6_kb_enriched_drafts.csv"
    enriched_xlsx = target / "stage6_kb_enriched_drafts.xlsx"
    comparison_csv = target / "stage6_before_after_comparison.csv"
    comparison_summary = target / "stage6_before_after_summary.md"
    write_csv(enriched_csv, enriched_rows)
    try_write_xlsx(enriched_rows, enriched_xlsx)
    write_csv(comparison_csv, comparison_rows)
    comparison_summary.write_text(
        render_stage6_summary(
            enriched_rows,
            comparison_rows,
            snapshot=snapshot,
            input_path=Path(input_path),
            baseline_csv_path=Path(baseline_csv_path) if baseline_csv_path else None,
            out_dir=target,
            provider_mode=provider_mode,
            model=model,
            reasoning_effort=reasoning_effort,
        ),
        encoding="utf-8",
    )
    counts = stage6_counts(enriched_rows, comparison_rows)
    (target / "stage6_eval_summary.json").write_text(
        json.dumps(
            {
                "schema_version": STAGE6_KB_EVAL_SCHEMA_VERSION,
                "rows_total": len(enriched_rows),
                "snapshot_path": str(snapshot["path"]),
                "snapshot_run_id": str(snapshot["run_id"]),
                "provider_mode": provider_mode,
                "active_brand": active_brand,
                "safety": stage6_kb_eval_safety_contract(),
                **counts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return Stage6KbEvalResult(
        out_dir=str(target),
        enriched_csv_path=str(enriched_csv),
        enriched_xlsx_path=str(enriched_xlsx),
        comparison_csv_path=str(comparison_csv),
        comparison_summary_path=str(comparison_summary),
        rows_total=len(enriched_rows),
        snapshot_path=str(snapshot["path"]),
        snapshot_run_id=str(snapshot["run_id"]),
        provider_mode=provider_mode,
        **counts,
    )


def build_draft_provider(*, provider_mode: str, model: str, reasoning_effort: str, timeout_sec: int, cache_dir: Path) -> DraftProvider:
    if provider_mode == "fake":
        return FakeStage6KbDraftProvider()
    if provider_mode == "codex":
        return SubscriptionLlmDraftProvider(
            model=model,
            reasoning_effort=reasoning_effort,
            timeout_sec=timeout_sec,
            max_attempts=2,
            cache_dir=cache_dir,
        )
    raise ValueError(f"unsupported provider_mode: {provider_mode}")


class FakeStage6KbDraftProvider:
    def build_draft(self, client_message: str, *, context: Optional[Mapping[str, Any]] = None) -> SubscriptionDraftResult:
        context = dict(context or {})
        snippets = list(context.get("knowledge_snippets") or [])
        topic_id = infer_topic_id(client_message)
        text = "Здравствуйте! Передам вопрос менеджеру, он проверит актуальные данные и вернется с ответом."
        if snippets:
            text = (
                "Здравствуйте! Вижу ваш вопрос и сверю его с данными по заявке. "
                "В базе знаний есть ориентир для менеджера, но перед ответом нужно сверить "
                "актуальность условий именно по вашей программе. "
                "Перед отправкой проверим актуальность условий именно для вашей программы."
            )
        payload = {
            "message_type": "question",
            "broad_group": "commercial" if topic_id == "theme:001_pricing" else "education_process",
            "topic_id": topic_id,
            "confidence_theme": 0.82,
            "confidence_group": 0.86,
            "route": "draft_for_manager",
            "draft_text": text,
            "manager_checklist": ["Проверить черновик перед отправкой клиенту."],
            "missing_facts": list(context.get("required_fact_keys") or []) if not context.get("facts_fresh") else [],
            "safety_flags": ["manager_approval_required", "no_auto_send"],
            "context_used": ["knowledge_snippets"] if snippets else [],
            "context_warnings": list(context.get("context_warnings") or []),
        }
        result = normalize_subscription_draft_payload(payload)
        result = apply_unsupported_promise_guard(result, context=context)
        return apply_input_policy_guards(result, client_message=client_message, context=context)


def build_guarded_draft(provider: DraftProvider, client_message: str, *, context: Mapping[str, Any]) -> SubscriptionDraftResult:
    raw = provider.build_draft(client_message, context=context)
    if isinstance(raw, SubscriptionDraftResult):
        result = raw
    elif hasattr(raw, "to_json_dict"):
        result = normalize_subscription_draft_payload(raw.to_json_dict())  # type: ignore[attr-defined]
    else:
        result = normalize_subscription_draft_payload(raw)
    result = apply_unsupported_promise_guard(result, context=context)
    result = apply_payment_confirmation_guard(result, client_message=client_message, context=context)
    result = apply_brand_separation_guard(result, client_message=client_message, context=context)
    result = apply_input_policy_guards(result, client_message=client_message, context=context)
    return apply_stage6_draft_text_safety(result)


def apply_stage6_draft_text_safety(result: SubscriptionDraftResult) -> SubscriptionDraftResult:
    raw_text = str(result.draft_text or "")
    cleaned = strip_stage6_internal_metadata(raw_text)
    flags = list(result.safety_flags)
    route = result.route
    if cleaned != raw_text:
        flags.extend(["internal_metadata_removed_from_draft", "stage6_internal_metadata_removed_from_draft"])
    if stage6_draft_has_forbidden_marker(cleaned):
        cleaned = STAGE6_DRAFT_FALLBACK_TEXT
        route = "manager_only"
        flags.append("stage6_internal_draft_marker_blocked")
    cleaned = " ".join(cleaned.split()).strip() or STAGE6_DRAFT_FALLBACK_TEXT
    if cleaned == raw_text and tuple(flags) == result.safety_flags and route == result.route:
        return result
    return replace(
        result,
        route=route,
        draft_text=cleaned,
        safety_flags=tuple(dict.fromkeys(flags)),
    )


def strip_stage6_internal_metadata(text: str) -> str:
    cleaned = STAGE6_INTERNAL_METADATA_BLOCK_RE.sub("", str(text or ""))
    cleaned = STAGE6_INTERNAL_METADATA_TOKEN_RE.sub("", cleaned)
    return " ".join(cleaned.split()).strip()


def stage6_draft_has_forbidden_marker(text: str) -> bool:
    return bool(STAGE6_FORBIDDEN_DRAFT_MARKER_RE.search(str(text or "")))


def stage6_brand_separation_violation(row: Mapping[str, Any]) -> bool:
    active_brand = normalize_active_brand(row.get("active_brand"))
    if active_brand == "unknown":
        return False
    draft_text = str(row.get("draft_text") or "").casefold()
    forbidden_terms = STAGE6_BRAND_FORBIDDEN_TERMS.get(active_brand, ())
    if any(term in draft_text for term in forbidden_terms):
        return True
    flags = serialize_cell(row.get("safety_flags"))
    return "brand_separation_guarded" in flags and str(row.get("route") or "") != "manager_only"


def enforce_stage6_route_safety(row: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    safety_flags = list(updated.get("safety_flags") or [])
    current = str(updated.get("current_client_message") or "")
    baseline_route = str(baseline.get("route") or "").strip()
    enriched_route = str(updated.get("route") or "").strip()
    if baseline_route == "manager_only" and enriched_route != "manager_only":
        updated["route"] = "manager_only"
        for flag in ("baseline_manager_only_preserved", "kb_route_not_allowed_to_weaken_baseline"):
            if flag not in safety_flags:
                safety_flags.append(flag)
    if baseline.get("route") == "manager_only" and HIGH_RISK_RE.search(current):
        updated["route"] = "manager_only"
        if "baseline_high_risk_preserved" not in safety_flags:
            safety_flags.append("baseline_high_risk_preserved")
    if "unsupported_promise_detected" in serialize_cell(safety_flags):
        updated["route"] = "manager_only"
        updated["draft_text"] = (
            "Здравствуйте! Передам вопрос менеджеру: здесь нужно сверить актуальные условия "
            "и не обещать суммы, проценты или сроки без подтвержденного источника."
        )
        if "unsupported_promise_removed_from_stage6_draft" not in safety_flags:
            safety_flags.append("unsupported_promise_removed_from_stage6_draft")
    updated["safety_flags"] = safety_flags
    return updated


def prepare_eval_dialog(
    dialog: Mapping[str, Any],
    *,
    snapshot: Mapping[str, Any],
    baseline: Mapping[str, Any],
    active_brand: str = "foton",
) -> dict[str, Any]:
    messages = list(dialog.get("messages") or [])
    target_index = select_target_client_index(messages)
    target = messages[target_index] if messages else {}
    current = str(target.get("text") or "").strip()
    previous = [format_eval_message(item) for item in messages[max(0, target_index - 10) : target_index]]
    topic_id = first_non_empty(baseline, ("topic_id", "theme_id")) or None
    required_keys = required_fact_keys_for_topic(topic_id or "", current)
    kc = build_kc_context(
        message_text=current,
        chunks=snapshot["chunks"],
        sources=snapshot["sources"],
        required_fact_keys=required_keys,
        topic_id=topic_id,
        active_brand=active_brand,
        received_at=parse_dt(target.get("date")),
        max_chunks=6,
        max_chunk_chars=700,
        total_char_limit=3200,
    )
    manager_patterns = select_patterns(snapshot["manager_answer_patterns"], current, topic_id, limit=max(0, 8 - len(kc.selected_chunks)))
    snippets = [format_chunk(chunk.to_json_dict()) for chunk in kc.selected_chunks]
    snippets.extend(format_pattern(pattern) for pattern in manager_patterns if format_pattern(pattern))
    fresh_texts = [chunk.text for chunk in kc.selected_chunks if chunk.freshness_status.casefold() in FRESH_STATUSES]
    if kc.freshness_blocks:
        fresh_texts = []
    facts_context = {
        "snapshot_run_id": snapshot["run_id"],
        "active_brand": active_brand,
        "selected_chunks": len(kc.selected_chunks),
        "manager_answer_patterns": len(manager_patterns),
        "manager_answer_patterns_are_facts": False,
        "fresh": bool(fresh_texts),
        "facts_fresh": bool(fresh_texts),
        "missing": bool(kc.freshness_blocks),
        "stale": bool(required_keys) and not fresh_texts,
    }
    for index, text in enumerate(fresh_texts[:6], start=1):
        facts_context[f"fresh_fact_{index}"] = text
    risks = ["historical_eval_pack", "manager_approval_required", "no_auto_send", "kb_snapshot_eval"]
    if HIGH_RISK_RE.search(current):
        risks.append("high_risk_text_marker")
    pilot_context = build_pilot_context(
        current,
        active_brand=active_brand,
        brand_policy={
            "client_text_active_brand_only": True,
            "cross_brand_client_text_forbidden": True,
        },
        recent_messages=previous,
        client_identity={"telegram_dialog_id": dialog.get("dialog_id"), "target_message_id": target.get("message_id")},
        rop_policy={"topic_id": topic_id or "", "bot_permission": "draft_for_manager", "answer_status": "historical_eval_not_live"},
        facts_context=facts_context,
        risk_flags=tuple(risks),
    ).to_prompt_context()
    pilot_context = {
        **dict(pilot_context),
        "knowledge_snippets": snippets[:8],
        "confirmed_facts": {f"fresh_fact_{index}": text for index, text in enumerate(fresh_texts[:6], start=1)},
        "required_fact_keys": list(required_keys),
        "missing_facts": [str(block.get("fact_key")) for block in kc.freshness_blocks],
        "facts_fresh": bool(fresh_texts),
        "manager_checklist": [
            "Использовать snapshot только как read-only подсказку.",
            "Исторические ответы менеджеров не считать источником фактов.",
        ],
    }
    return {
        "dialog_id": str(dialog.get("dialog_id") or ""),
        "active_brand": active_brand,
        "message_count": int(dialog.get("message_count") or len(messages)),
        "target_message_id": str(target.get("message_id") or ""),
        "target_date": str(target.get("date") or ""),
        "current_client_message": current,
        "previous_context": "\n".join(previous),
        "priority_note": priority_note(str(dialog.get("dialog_id") or ""), current),
        "pilot_context": pilot_context,
        "snapshot_run_id": snapshot["run_id"],
        "selected_chunk_count": len(kc.selected_chunks),
        "manager_patterns_count": len(manager_patterns),
        "knowledge_snippets": snippets[:8],
        "required_fact_keys": list(required_keys),
        "freshness_blocks": [dict(block) for block in kc.freshness_blocks],
        "facts_fresh": bool(fresh_texts),
    }


def build_stage6_enriched_row(prepared: Mapping[str, Any], result: Mapping[str, Any], *, provider_mode: str, run_index: int) -> dict[str, Any]:
    return {
        "schema_version": STAGE6_KB_EVAL_SCHEMA_VERSION,
        "run_index": run_index,
        "provider_mode": provider_mode,
        "dialog_id": prepared["dialog_id"],
        "active_brand": prepared["active_brand"],
        "priority_note": prepared["priority_note"],
        "message_count": prepared["message_count"],
        "target_message_id": prepared["target_message_id"],
        "target_date": prepared["target_date"],
        "current_client_message": prepared["current_client_message"],
        "previous_context": prepared["previous_context"],
        "snapshot_run_id": prepared["snapshot_run_id"],
        "selected_chunk_count": prepared["selected_chunk_count"],
        "manager_patterns_count": prepared["manager_patterns_count"],
        "knowledge_snippets": prepared["knowledge_snippets"],
        "required_fact_keys": prepared["required_fact_keys"],
        "freshness_blocks": prepared["freshness_blocks"],
        "facts_fresh": prepared["facts_fresh"],
        "used_kb_context": bool(prepared["knowledge_snippets"]),
        "message_type": result.get("message_type"),
        "broad_group": result.get("broad_group"),
        "topic_id": result.get("topic_id"),
        "confidence_theme": result.get("confidence_theme", result.get("topic_confidence")),
        "confidence_group": result.get("confidence_group"),
        "alternative_themes": result.get("alternative_themes") or [],
        "risk_level": result.get("risk_level"),
        "route": result.get("route"),
        "draft_text": result.get("draft_text"),
        "manager_checklist": result.get("manager_checklist") or [],
        "missing_facts": result.get("missing_facts") or [],
        "context_warnings": result.get("context_warnings") or [],
        "safety_flags": result.get("safety_flags") or [],
        "forbidden_promises_detected": result.get("forbidden_promises_detected") or [],
        "context_used": result.get("context_used") or [],
        "manager_followup_required": result.get("manager_followup_required"),
        "manager_followup_deadline": result.get("manager_followup_deadline"),
        "error": result.get("error"),
    }


def build_stage6_comparison_row(new: Mapping[str, Any], old: Mapping[str, Any]) -> dict[str, Any]:
    old_text = first_non_empty(old, ("draft_text", "draft"))
    new_text = str(new.get("draft_text") or "")
    old_empty = empty_clarification(old_text)
    new_empty = empty_clarification(new_text)
    return {
        "dialog_id": new.get("dialog_id"),
        "target_message_id": new.get("target_message_id"),
        "current_client_message": new.get("current_client_message"),
        "baseline_topic_id": old.get("topic_id"),
        "enriched_topic_id": new.get("topic_id"),
        "baseline_route": old.get("route"),
        "enriched_route": new.get("route"),
        "baseline_empty_clarification": old_empty,
        "enriched_empty_clarification": new_empty,
        "baseline_substantive_score": substantive_score(old_text),
        "enriched_substantive_score": substantive_score(new_text),
        "draft_became_more_substantive": substantive_score(new_text) >= substantive_score(old_text) + 2 or (old_empty and not new_empty),
        "empty_clarification_reduced": old_empty and not new_empty,
        "used_kb_context": bool(new.get("used_kb_context")),
        "invalid_topic_id": bool(new.get("topic_id")) and str(new.get("topic_id")) not in load_valid_theme_and_service_ids(),
        "unsupported_numeric_promises": (
            "unsupported_promise_detected" in serialize_cell(new.get("safety_flags"))
            and "unsupported_promise_removed_from_stage6_draft" not in serialize_cell(new.get("safety_flags"))
        ),
        "brand_separation_violation": stage6_brand_separation_violation(new),
        "high_risk_route_relaxed": bool(HIGH_RISK_RE.search(str(new.get("current_client_message") or "")))
        and old.get("route") == "manager_only"
        and new.get("route") == "draft_for_manager",
        "baseline_manager_only_relaxed": old.get("route") == "manager_only" and new.get("route") != "manager_only",
        "baseline_manager_only_preserved": (
            old.get("route") == "manager_only"
            and new.get("route") == "manager_only"
            and "baseline_manager_only_preserved" in serialize_cell(new.get("safety_flags"))
        ),
        "baseline_draft_text": old_text,
        "enriched_draft_text": new_text,
    }


def render_stage6_summary(
    enriched: Sequence[Mapping[str, Any]],
    comparison: Sequence[Mapping[str, Any]],
    *,
    snapshot: Mapping[str, Any],
    input_path: Path,
    baseline_csv_path: Path | None,
    out_dir: Path,
    provider_mode: str,
    model: str,
    reasoning_effort: str,
) -> str:
    counts = stage6_counts(enriched, comparison)
    routes = Counter(str(row.get("route") or "") for row in enriched)
    context_found_rate = round(counts["used_kb_context"] / len(enriched), 4) if enriched else 0.0
    return "\n".join(
        [
            "# Stage 6 KB enriched drafts summary",
            "",
            f"- created_at: {datetime.now(timezone.utc).isoformat()}",
            f"- input: {input_path}",
            f"- baseline_csv: {baseline_csv_path or ''}",
            f"- snapshot: {snapshot['path']}",
            f"- snapshot_run_id: {snapshot['run_id']}",
            f"- out_dir: {out_dir}",
            f"- provider_mode: {provider_mode}",
            f"- model: {model if provider_mode == 'codex' else 'fake'}",
            f"- reasoning_effort: {reasoning_effort if provider_mode == 'codex' else 'fake'}",
            "",
            "## Metrics",
            "",
            f"- rows_total: {len(enriched)}",
            f"- routes: {dict(routes)}",
            f"- llm_or_provider_errors: {counts['errors']}",
            f"- invalid_topic_ids: {counts['invalid_topic_ids']}",
            f"- unsupported_numeric_promises: {counts['unsupported_numeric_promises']}",
            f"- brand_separation_violation: {counts['brand_separation_violation']}",
            f"- baseline_manager_only_relaxed: {counts['baseline_manager_only_relaxed']}",
            f"- baseline_manager_only_preserved: {counts['baseline_manager_only_preserved']}",
            f"- used_kb_context: {counts['used_kb_context']}",
            f"- context_found_rate: {context_found_rate}",
            f"- became_more_substantive: {counts['became_more_substantive']}",
            f"- empty_clarification_reduced: {counts['empty_clarification_reduced']}",
            f"- high_risk_route_relaxed: {counts['high_risk_route_relaxed']}",
            "",
            "## Safety",
            "",
            "- live_telegram: false",
            "- client_send: false",
            "- write_crm: false",
            "- write_tallanto: false",
            "- write_stable_runtime: false",
            "- run_asr: false",
            "- run_resolve_analyze: false",
            "",
        ]
    )


def stage6_counts(enriched: Sequence[Mapping[str, Any]], comparison: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "used_kb_context": sum(1 for row in comparison if boolish(row.get("used_kb_context"))),
        "became_more_substantive": sum(1 for row in comparison if boolish(row.get("draft_became_more_substantive"))),
        "empty_clarification_reduced": sum(1 for row in comparison if boolish(row.get("empty_clarification_reduced"))),
        "invalid_topic_ids": sum(1 for row in comparison if boolish(row.get("invalid_topic_id"))),
        "unsupported_numeric_promises": sum(1 for row in comparison if boolish(row.get("unsupported_numeric_promises"))),
        "brand_separation_violation": sum(1 for row in comparison if boolish(row.get("brand_separation_violation"))),
        "high_risk_route_relaxed": sum(1 for row in comparison if boolish(row.get("high_risk_route_relaxed"))),
        "baseline_manager_only_relaxed": sum(1 for row in comparison if boolish(row.get("baseline_manager_only_relaxed"))),
        "baseline_manager_only_preserved": sum(1 for row in comparison if boolish(row.get("baseline_manager_only_preserved"))),
        "errors": sum(1 for row in enriched if str(row.get("error") or "")),
    }


def load_snapshot(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("knowledge snapshot JSON root must be an object")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    return {
        "path": path,
        "run_id": str(payload.get("run_id") or metadata.get("run_id") or path.stem),
        "sources": [normalize_source(item) for item in payload.get("sources", []) if isinstance(item, Mapping)],
        "chunks": [chunk for item in payload.get("chunks", []) if isinstance(item, Mapping) and (chunk := normalize_chunk(item)).get("text")],
        "manager_answer_patterns": [item for item in payload.get("manager_answer_patterns", []) if isinstance(item, Mapping)],
    }


def normalize_chunk(item: Mapping[str, Any]) -> Mapping[str, Any]:
    text = first_non_empty(item, ("text", "chunk_text", "client_safe_text", "short_fact", "manager_text"))
    title = first_non_empty(item, ("title", "source_title", "fact_type", "chunk_id")) or "KB fragment"
    fact_types = item.get("fact_types") or item.get("fact_type") or classify_fact_types(f"{title} {text}")
    if isinstance(fact_types, str):
        fact_types = [fact_types]
    metadata = dict(item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {})
    brand = first_non_empty(item, ("brand", "active_brand")) or metadata.get("brand") or "brand_neutral"
    cross_brand_mixed = boolish(item.get("cross_brand_mixed") or metadata.get("cross_brand_mixed"))
    forbidden_for_client = boolish(item.get("forbidden_for_client") or metadata.get("forbidden_for_client"))
    internal_only = boolish(item.get("internal_only") or metadata.get("internal_only"))
    cross_brand_policy = first_non_empty(item, ("cross_brand_policy",)) or metadata.get("cross_brand_policy") or ""
    return {
        "chunk_id": first_non_empty(item, ("chunk_id", "fact_id", "id")) or f"kc_chunk:{abs(hash((title, text)))}",
        "source_id": first_non_empty(item, ("source_id", "source", "source_title")) or "source:snapshot",
        "title": title,
        "text": text,
        "fact_types": list(fact_types or []),
        "freshness_status": first_non_empty(item, ("freshness_status", "status")) or "unknown",
        "brand": brand,
        "active_brand_scope": first_non_empty(item, ("active_brand_scope",)) or metadata.get("active_brand_scope") or "",
        "cross_brand_policy": cross_brand_policy,
        "cross_brand_mixed": cross_brand_mixed,
        "forbidden_for_client": forbidden_for_client,
        "internal_only": internal_only,
        "metadata": {
            **metadata,
            "brand": brand,
            "cross_brand_policy": cross_brand_policy,
            "cross_brand_mixed": cross_brand_mixed,
            "forbidden_for_client": forbidden_for_client,
            "internal_only": internal_only,
        },
    }


def normalize_source(item: Mapping[str, Any]) -> Mapping[str, Any]:
    title = first_non_empty(item, ("title", "source_title", "name", "path")) or "Snapshot source"
    source_id = first_non_empty(item, ("source_id", "id")) or f"source:{abs(hash(title))}"
    fact_types = item.get("fact_types") or item.get("fact_type") or classify_fact_types(title)
    if isinstance(fact_types, str):
        fact_types = [fact_types]
    freshness = internal_freshness(first_non_empty(item, ("freshness_status", "status", "freshness")))
    return {
        "source_id": source_id,
        "title": title,
        "source_kind": first_non_empty(item, ("source_kind", "type", "kind")) or "snapshot_source",
        "fact_types": list(fact_types or []),
        "path": first_non_empty(item, ("path", "local_path")) or f"snapshot://{source_id}",
        "freshness_status": freshness,
        "usable_for_precise_answer": bool(item.get("usable_for_precise_answer")) and freshness == FRESHNESS_FRESH,
    }


def internal_freshness(value: str) -> str:
    text = str(value or "").casefold()
    return FRESHNESS_FRESH if text in FRESH_STATUSES else (text or "unknown")


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def read_baseline_index(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    rows = read_csv(path) if path.exists() else []
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        dialog_id = first_non_empty(row, ("dialog_id", "row_id", "id"))
        message_id = first_non_empty(row, ("target_message_id", "message_id"))
        if dialog_id:
            result[(dialog_id, message_id)] = row
            result.setdefault((dialog_id, ""), row)
    return result


def find_baseline_for_dialog(dialog: Mapping[str, Any], baseline: Mapping[tuple[str, str], Mapping[str, Any]]) -> Mapping[str, Any]:
    dialog_id = str(dialog.get("dialog_id") or "")
    messages = list(dialog.get("messages") or [])
    message_id = str(messages[select_target_client_index(messages)].get("message_id") or "") if messages else ""
    return baseline.get((dialog_id, message_id)) or baseline.get((dialog_id, "")) or {}


def select_target_client_index(messages: Sequence[Mapping[str, Any]]) -> int:
    client_indexes = [idx for idx, item in enumerate(messages) if item.get("direction") == "client" and str(item.get("text") or "").strip()]
    if not client_indexes:
        return max(0, len(messages) - 1)
    for idx in reversed(client_indexes):
        if substantive_client_message(str(messages[idx].get("text") or "")):
            return idx
    return client_indexes[-1]


def substantive_client_message(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    return bool(cleaned) and not TRIVIAL_RE.match(cleaned) and (len(cleaned) >= 25 or bool(SUBSTANTIVE_RE.search(cleaned)))


def format_eval_message(message: Mapping[str, Any]) -> str:
    return f"{message.get('date') or ''} {message.get('direction') or ''}: {' '.join(str(message.get('text') or '').split())}"


def format_chunk(chunk: Mapping[str, Any]) -> str:
    title = str(chunk.get("title") or "База знаний").strip() or "База знаний"
    return shorten(f"{title}: {chunk.get('text')}", 700)


def format_pattern(pattern: Mapping[str, Any]) -> str:
    text = first_non_empty(pattern, ("safe_pattern", "safe_rewrite", "client_safe_pattern", "pattern_text", "summary"))
    return shorten(f"Прием менеджера, не факт: {text}", 700) if text else ""


def select_patterns(patterns: Sequence[Mapping[str, Any]], query: str, topic_id: str | None, *, limit: int) -> list[Mapping[str, Any]]:
    if limit <= 0:
        return []
    terms = {word.casefold() for word in WORD_RE.findall(query)}
    scored: list[tuple[int, Mapping[str, Any]]] = []
    for pattern in patterns:
        text = json.dumps(pattern, ensure_ascii=False).casefold()
        score = sum(1 for term in terms if term in text)
        if topic_id and topic_id in text:
            score += 5
        if score:
            scored.append((score, pattern))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[:limit]]


def infer_topic_id(text: str) -> str:
    value = text.casefold()
    if "возврат" in value or "вернуть" in value:
        return "theme:009_refund"
    if "распис" in value or "когда" in value:
        return "theme:013_schedule"
    if "ссыл" in value or "доступ" in value:
        return "theme:025_missing_links_access"
    if "цен" in value or "стоим" in value or "оплат" in value:
        return "theme:001_pricing"
    if "курс" in value or "программ" in value:
        return "theme:016_program"
    return "service:S2_unclear"


def priority_note(dialog_id: str, current: str) -> str:
    if dialog_id == "1063099421":
        return "priority_refund_case"
    if dialog_id == "1084253673":
        return "priority_schedule_without_fresh_facts"
    return "high_risk_text_marker" if HIGH_RISK_RE.search(current) else ""


def parse_dt(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
    except ValueError:
        return None


def substantive_score(text: str) -> int:
    return len({word.casefold() for word in WORD_RE.findall(str(text or "")) if word.casefold() not in {"здравствуйте", "уточним", "проверим", "менеджер", "вернемся", "передам"}})


def empty_clarification(text: str) -> bool:
    value = str(text or "").strip()
    return not value or (len(value) < 170 and bool(EMPTY_CLARIFICATION_RE.search(value)) and substantive_score(value) < 8)


def shorten(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").casefold() in {"1", "true", "yes", "да"}


def guard_safe_output_dir(path: Path) -> None:
    if "stable_runtime" in path.expanduser().resolve(strict=False).parts:
        raise ValueError("Refusing to write Stage 6 eval under stable_runtime")


def stage6_kb_eval_safety_contract() -> Mapping[str, bool]:
    return {
        "live_telegram": False,
        "client_send": False,
        "write_crm": False,
        "write_tallanto": False,
        "write_stable_runtime": False,
        "run_asr": False,
        "run_resolve_analyze": False,
        "read_only_snapshot": True,
        "manager_approval_required": True,
    }


def build_enriched_row(row: Mapping[str, Any], *, snapshot_path: Path) -> dict[str, Any]:
    question = first_non_empty(row, ("question_text", "client_message", "message_text", "current_message", "customer_text", "dialog_last_client_message"))
    if not question:
        question = first_non_empty(row, ("draft_text", "what_to_check", "context"))
    topic_id = first_non_empty(row, ("topic_id", "theme_id", "predicted_theme", "theme"))
    route = first_non_empty(row, ("route", "routing", "decision")) or "draft_for_manager"
    context = build_telegram_pilot_context_from_snapshot(
        question,
        snapshot_path=snapshot_path,
        topic_id=topic_id,
        required_fact_keys=required_fact_keys_for_topic(topic_id, question),
        rop_policy={"topic_id": topic_id, "bot_permission": "draft_for_manager"},
    )
    prompt_context = context.to_prompt_context()
    snippets = prompt_context.get("knowledge_snippets") or []
    facts_context = prompt_context.get("facts_context") or {}
    enriched_draft = build_safe_enriched_draft(question, snippets=snippets, route=route)
    return {
        **dict(row),
        "kb_used": bool(snippets),
        "knowledge_base_version": prompt_context.get("knowledge_base_version", ""),
        "kb_snippets_count": len(snippets),
        "kb_missing_facts": "|".join(prompt_context.get("missing_facts") or []),
        "kb_facts_fresh": facts_context.get("facts_fresh", False),
        "route_after_kb": route if facts_context.get("facts_fresh") else "manager_only",
        "draft_text_after_kb": enriched_draft,
        "what_to_check_after_kb": "Проверить, стал ли черновик содержательнее без точных неподтвержденных обещаний.",
    }


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return dict(loaded) if isinstance(loaded, Mapping) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            loaded = json.loads(line)
            if isinstance(loaded, Mapping):
                rows.append(dict(loaded))
    return rows


def select_target_client_message(dialog: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    target_id = str(baseline.get("target_message_id") or "")
    messages = [message for message in dialog.get("messages", []) if isinstance(message, Mapping)]
    if target_id:
        for message in messages:
            if str(message.get("message_id") or "") == target_id:
                return dict(message)
    for message in reversed(messages):
        if str(message.get("direction") or "").casefold() in {"client", "in", "inbound", "customer"}:
            return dict(message)
    return dict(messages[-1]) if messages else {"text": ""}


def recent_messages_for_dialog(dialog: Mapping[str, Any], *, target_message_id: str = "", limit: int = 10) -> list[str]:
    messages = [message for message in dialog.get("messages", []) if isinstance(message, Mapping)]
    result: list[str] = []
    for message in messages:
        if target_message_id and str(message.get("message_id") or "") == target_message_id:
            break
        direction = str(message.get("direction") or "unknown")
        text = str(message.get("text") or "").strip()
        if text:
            result.append(f"{direction}: {text}")
    return result[-limit:]


def snapshot_run_id(snapshot: Mapping[str, Any]) -> str:
    metadata = snapshot.get("metadata") if isinstance(snapshot.get("metadata"), Mapping) else {}
    return str(snapshot.get("run_id") or snapshot.get("snapshot_id") or metadata.get("run_id") or "unknown")


def supplement_snapshot_snippets(
    snapshot: Mapping[str, Any],
    snippets: Sequence[str],
    *,
    max_snippets: int = 2,
    active_brand: str = "unknown",
) -> list[str]:
    result = list(snippets)
    if result:
        return result
    for chunk in snapshot.get("chunks", []) or snapshot.get("knowledge_chunks", []) or []:
        if not isinstance(chunk, Mapping):
            continue
        normalized = normalize_chunk(chunk)
        if not chunk_allowed_for_stage6(normalized, active_brand=active_brand):
            continue
        text = str(normalized.get("text") or "").strip()
        title = str(normalized.get("title") or "База знаний").strip()
        if not text:
            continue
        result.append(f"[{title}] {text[:700]}")
        if len(result) >= max_snippets:
            break
    return result


def chunk_allowed_for_stage6(chunk: Mapping[str, Any], *, active_brand: str = "unknown") -> bool:
    if boolish(chunk.get("forbidden_for_client")) or boolish(chunk.get("internal_only")):
        return False
    if boolish(chunk.get("cross_brand_mixed")) or str(chunk.get("cross_brand_policy") or "") == "forbidden_for_client":
        return False
    brand = str(chunk.get("brand") or "brand_neutral").strip().casefold()
    active = normalize_active_brand(active_brand)
    if brand in {"", "unknown", "brand_neutral"}:
        return True
    return active != "unknown" and brand == active


def normalize_active_brand(value: Any) -> str:
    text = str(value or "unknown").strip().casefold()
    if text in {"foton", "фотон"}:
        return "foton"
    if text in {"unpk", "унпк", "унпк мфти"}:
        return "unpk"
    return "unknown"


def normalize_provider_response(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        payload = dict(response)
    elif hasattr(response, "to_json_dict"):
        payload = dict(response.to_json_dict())
    else:
        payload = {}
    payload.setdefault("route", "draft_for_manager")
    payload.setdefault("draft_text", "")
    payload.setdefault("manager_checklist", [])
    payload.setdefault("safety_flags", [])
    return payload


def is_more_substantive(old_draft: str, new_draft: str) -> bool:
    old_words = set(WORD_RE.findall(old_draft.casefold()))
    new_words = set(WORD_RE.findall(new_draft.casefold()))
    if len(new_draft) >= len(old_draft) + 20:
        return True
    return len(new_words - old_words) >= 3


def render_stage6_summary(*args: Any, **kwargs: Any) -> str:
    if args and hasattr(args[0], "rows_total"):
        result = args[0]
        comparison_rows = args[1] if len(args) > 1 else []
        return "\n".join(
            [
                "# Stage 6 KB enriched dry run",
                "",
                f"- rows_total: {result.rows_total}",
                f"- used_kb_context: {result.used_kb_context}",
                f"- became_more_substantive: {result.became_more_substantive}",
                f"- client_send: {str(stage6_kb_eval_safety_contract()['client_send']).lower()}",
                f"- write_stable_runtime: {str(stage6_kb_eval_safety_contract()['write_stable_runtime']).lower()}",
                f"- rows in before/after comparison: `{len(comparison_rows)}`",
                "",
                "Это сухой прогон: клиентам ничего не отправлялось.",
            ]
        ) + "\n"

    enriched = args[0] if args else []
    comparison = args[1] if len(args) > 1 else []
    snapshot = kwargs.get("snapshot") or {}
    input_path = kwargs.get("input_path") or ""
    baseline_csv_path = kwargs.get("baseline_csv_path") or ""
    out_dir = kwargs.get("out_dir") or ""
    provider_mode = kwargs.get("provider_mode") or ""
    model = kwargs.get("model") or ""
    reasoning_effort = kwargs.get("reasoning_effort") or ""
    counts = stage6_counts(enriched, comparison)
    routes = Counter(str(row.get("route") or "") for row in enriched)
    context_found_rate = round(counts["used_kb_context"] / len(enriched), 4) if enriched else 0.0
    return "\n".join(
        [
            "# Stage 6 KB enriched drafts summary",
            "",
            f"- created_at: {datetime.now(timezone.utc).isoformat()}",
            f"- input: {input_path}",
            f"- baseline_csv: {baseline_csv_path or ''}",
            f"- snapshot: {snapshot.get('path', '')}",
            f"- snapshot_run_id: {snapshot.get('run_id', '')}",
            f"- out_dir: {out_dir}",
            f"- provider_mode: {provider_mode}",
            f"- model: {model if provider_mode == 'codex' else 'fake'}",
            f"- reasoning_effort: {reasoning_effort if provider_mode == 'codex' else 'fake'}",
            "",
            "## Metrics",
            "",
            f"- rows_total: {len(enriched)}",
            f"- routes: {dict(routes)}",
            f"- llm_or_provider_errors: {counts['errors']}",
            f"- invalid_topic_ids: {counts['invalid_topic_ids']}",
            f"- unsupported_numeric_promises: {counts['unsupported_numeric_promises']}",
            f"- brand_separation_violation: {counts['brand_separation_violation']}",
            f"- baseline_manager_only_relaxed: {counts['baseline_manager_only_relaxed']}",
            f"- baseline_manager_only_preserved: {counts['baseline_manager_only_preserved']}",
            f"- used_kb_context: {counts['used_kb_context']}",
            f"- context_found_rate: {context_found_rate}",
            f"- became_more_substantive: {counts['became_more_substantive']}",
            f"- empty_clarification_reduced: {counts['empty_clarification_reduced']}",
            f"- high_risk_route_relaxed: {counts['high_risk_route_relaxed']}",
            "",
            "## Safety",
            "",
            "- live_telegram: false",
            "- client_send: false",
            "- write_crm: false",
            "- write_tallanto: false",
            "- write_stable_runtime: false",
            "- run_asr: false",
            "- run_resolve_analyze: false",
            "",
        ]
    )


def build_safe_enriched_draft(question: str, *, snippets: Sequence[str], route: str) -> str:
    if snippets:
        return (
            "Здравствуйте! Вижу ваш вопрос. По базе знаний есть подходящая информация, но точные условия "
            "я передам менеджеру на проверку. Менеджер сверит актуальные данные и вернется с точным ответом."
        )
    return "Здравствуйте! Передам вопрос менеджеру, он проверит актуальные данные и вернется с ответом."


def build_comparison_summary(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for old, new in zip(before, after):
        old_draft = first_non_empty(old, ("draft_text", "draft"))
        new_draft = str(new.get("draft_text_after_kb") or "")
        rows.append(
            {
                "row_id": first_non_empty(old, ("dialog_id", "row_id", "id")),
                "kb_used": new.get("kb_used"),
                "old_len": len(old_draft),
                "new_len": len(new_draft),
                "became_more_specific": len(new_draft) > len(old_draft),
                "route_after_kb": new.get("route_after_kb"),
            }
        )
    summary = {
        "rows_total": len(rows),
        "kb_used": sum(1 for row in rows if row["kb_used"]),
        "became_more_specific": sum(1 for row in rows if row["became_more_specific"]),
        "manager_only_after_kb": sum(1 for row in rows if row["route_after_kb"] == "manager_only"),
        "client_send": False,
        "live_telegram_used": False,
    }
    markdown = "\n".join(
        [
            "# Stage 6 KB enriched dry run",
            "",
            f"- rows_total: {summary['rows_total']}",
            f"- kb_used: {summary['kb_used']}",
            f"- became_more_specific: {summary['became_more_specific']}",
            f"- manager_only_after_kb: {summary['manager_only_after_kb']}",
            "",
            "Это сухой прогон: клиентам ничего не отправлялось.",
        ]
    ) + "\n"
    return {"rows": rows, "summary": summary, "markdown": markdown}


def required_fact_keys_for_topic(topic_id: str, question: str) -> tuple[str, ...]:
    text = f"{topic_id} {question}".casefold()
    keys: list[str] = []
    if any(marker in text for marker in ("price", "pricing", "стоим", "цена", "скид", "оплат")):
        keys.append("prices.current")
    if any(marker in text for marker in ("schedule", "распис", "когда", "во сколько")):
        keys.append("schedule.current")
    if any(marker in text for marker in ("refund", "tax", "matkap", "договор", "возврат", "налог", "маткап")):
        keys.append("documents.current")
    return tuple(dict.fromkeys(keys))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: serialize_cell(row.get(key)) for key in fieldnames})


def try_write_xlsx(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    try:
        from openpyxl import Workbook
    except Exception:
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "stage6_kb"
    if rows:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        ws.append(fieldnames)
        for row in rows:
            ws.append([serialize_cell(row.get(key)) for key in fieldnames])
    wb.save(path)


def first_non_empty(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def serialize_cell(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value or "")


if __name__ == "__main__":
    raise SystemExit(main())
