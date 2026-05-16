from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.question_catalog.builder import (
    CatalogBuildConfig,
    attach_template_ids,
    build_answer_templates,
    build_question_classes,
    build_summary,
    default_config,
    discover_fact_sources,
    write_outputs,
)
from mango_mvp.question_catalog.classifier import get_theme_metadata, validate_against_taxonomy
from mango_mvp.question_catalog.codex_full_run import CODEX_FULL_RUN_SCHEMA_VERSION, read_jsonl
from mango_mvp.question_catalog.contracts import (
    ANSWER_STATUS_DRAFT_NEEDS_REVIEW,
    ANSWER_STATUS_MANAGER_ONLY,
    ANSWER_STATUS_NOT_CUSTOMER_QUESTION,
    ANSWER_STATUS_NOT_ENOUGH_CONTEXT,
    ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT,
    BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK,
    BOT_PERMISSION_DRAFT_ONLY,
    BOT_PERMISSION_MANAGER_ONLY,
    BOT_PERMISSION_NOT_ALLOWED,
    QuestionItem,
    stable_question_class_id,
)
from mango_mvp.question_catalog.safety import guard_question_catalog_output_path


REBUILD_SCHEMA_VERSION = "question_catalog_rebuild_from_codex_predictions_v1_2026_05_16"


def rebuild_catalog_from_predictions(
    *,
    project_root: Path | str,
    items_path: Path | str,
    predictions_path: Path | str,
    out_root: Path | str,
    tenant_id: str = "foton",
    since: datetime | None = None,
    require_all_predictions: bool = True,
) -> dict[str, Any]:
    project = Path(project_root).resolve()
    output_root = guard_question_catalog_output_path(Path(out_root), project_root=project)
    if any(part.casefold() == "stable_runtime" for part in output_root.parts):
        raise ValueError(f"question catalog rebuild output must not be under stable_runtime: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_items = read_question_items_jsonl(Path(items_path))
    predictions = load_predictions(Path(predictions_path))
    calibrated_items = apply_predictions_to_items(
        source_items,
        predictions,
        tenant_id=tenant_id,
        require_all_predictions=require_all_predictions,
    )
    config = default_config(project, output_root)
    config = CatalogBuildConfig(
        project_root=project,
        out_root=output_root,
        tenant_id=tenant_id,
        since=since or datetime(2025, 1, 1, tzinfo=timezone.utc),
        calls_enriched_reviews=config.calls_enriched_reviews,
        telegram_messages_jsonl=config.telegram_messages_jsonl,
        mail_archive_root=config.mail_archive_root,
        fact_source_roots=config.fact_source_roots,
    )
    classes = build_question_classes(calibrated_items, tenant_id=tenant_id)
    templates = build_answer_templates(classes, tenant_id=tenant_id)
    classes = attach_template_ids(classes, templates)
    fact_sources = discover_fact_sources(config.fact_source_roots)
    summary = build_summary(
        config=config,
        items=calibrated_items,
        classes=classes,
        templates=templates,
        fact_sources=fact_sources,
        source_reports=[
            {
                "source": "codex_full_v2_predictions",
                "items_path": str(items_path),
                "predictions_path": str(predictions_path),
                "question_items": len(calibrated_items),
                "predictions": len(predictions),
            }
        ],
    )
    summary["rebuild"] = {
        "schema_version": REBUILD_SCHEMA_VERSION,
        "prediction_schema_version": CODEX_FULL_RUN_SCHEMA_VERSION,
        "items_path": str(items_path),
        "predictions_path": str(predictions_path),
        "require_all_predictions": require_all_predictions,
        "question_items": len(calibrated_items),
        "question_classes": len(classes),
        "missing_predictions": max(0, len(source_items) - len(predictions)),
    }
    outputs = write_outputs(
        output_root,
        items=calibrated_items,
        classes=classes,
        templates=templates,
        fact_sources=fact_sources,
        source_reports=summary["source_reports"],
        summary=summary,
    )
    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    (output_root / "question_catalog_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def read_question_items_jsonl(path: Path) -> list[QuestionItem]:
    items: list[QuestionItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number}: question item must be JSON object")
            items.append(question_item_from_json(payload))
    return items


def question_item_from_json(payload: Mapping[str, Any]) -> QuestionItem:
    occurred_at = _parse_datetime(payload.get("occurred_at"))
    return QuestionItem(
        tenant_id=str(payload.get("tenant_id") or "foton"),
        source_channel=str(payload.get("source_channel") or ""),
        source_ref=str(payload.get("source_ref") or ""),
        customer_text_redacted=str(payload.get("customer_text_redacted") or ""),
        question_class_id=str(payload.get("question_class_id") or ""),
        occurred_at=occurred_at,
        manager_text_redacted=payload.get("manager_text_redacted"),
        intent=str(payload.get("intent") or "other"),
        product=payload.get("product"),
        grade=payload.get("grade"),
        subject=payload.get("subject"),
        format=payload.get("format"),
        price_related=bool(payload.get("price_related")),
        schedule_related=bool(payload.get("schedule_related")),
        documents_related=bool(payload.get("documents_related")),
        safety_flags=tuple(payload.get("safety_flags") or ()),
        answer_evidence_status=str(payload.get("answer_evidence_status") or ANSWER_STATUS_DRAFT_NEEDS_REVIEW),
        answer_source=payload.get("answer_source"),
        requires_dynamic_facts=bool(payload.get("requires_dynamic_facts")),
        dynamic_fact_types=tuple(payload.get("dynamic_fact_types") or ()),
        fact_freshness_required=payload.get("fact_freshness_required"),
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {},
        question_item_id=str(payload.get("question_item_id") or ""),
    )


def load_predictions(path: Path) -> dict[str, Mapping[str, Any]]:
    rows = read_jsonl(path)
    predictions: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        question_item_id = str(row.get("question_item_id") or "").strip()
        theme_id = str(row.get("predicted_theme_id") or "").strip()
        if not question_item_id:
            raise ValueError("prediction row missing question_item_id")
        if question_item_id in predictions:
            raise ValueError(f"duplicate prediction for {question_item_id}")
        validate_against_taxonomy(theme_id)
        predictions[question_item_id] = dict(row)
    return predictions


def apply_predictions_to_items(
    items: Sequence[QuestionItem],
    predictions: Mapping[str, Mapping[str, Any]],
    *,
    tenant_id: str = "foton",
    require_all_predictions: bool = True,
) -> list[QuestionItem]:
    calibrated: list[QuestionItem] = []
    missing: list[str] = []
    for item in items:
        question_item_id = str(item.question_item_id or "")
        prediction = predictions.get(question_item_id)
        if prediction is None:
            missing.append(question_item_id)
            if require_all_predictions:
                continue
            calibrated.append(item)
            continue
        calibrated.append(apply_prediction_to_item(item, prediction, tenant_id=tenant_id))
    if missing and require_all_predictions:
        raise ValueError(f"missing predictions for {len(missing)} question items; first={missing[:5]}")
    return calibrated


def apply_prediction_to_item(item: QuestionItem, prediction: Mapping[str, Any], *, tenant_id: str = "foton") -> QuestionItem:
    theme_id = str(prediction.get("predicted_theme_id") or "").strip()
    validate_against_taxonomy(theme_id)
    theme = get_theme_metadata(theme_id)
    required_facts = tuple(str(value) for value in (theme.get("required_facts") or ()) if str(value).strip())
    default_permission = str(theme.get("default_bot_permission") or "draft_for_manager")
    answer_status, bot_permission = policy_for_theme(theme_id, default_permission=default_permission, required_facts=required_facts)
    class_id = stable_question_class_id(tenant_id=tenant_id, class_key=theme_id)
    metadata = dict(item.metadata)
    metadata.update(
        {
            "previous_question_class_id": item.question_class_id,
            "previous_theme_id": item.metadata.get("theme_id") or item.intent,
            "theme_id": theme_id,
            "theme_name": theme.get("theme_name") or theme.get("service_name") or theme_id,
            "business_block": theme.get("business_block") or "Служебные категории",
            "question_subclass": theme.get("theme_name") or theme.get("service_name") or theme_id,
            "question_subclass_key": theme_id,
            "required_fact_keys": list(required_facts),
            "answer_status": answer_status,
            "bot_permission": bot_permission,
            "manager_handoff_reason": theme.get("escalation_rule") or theme.get("routing_rule") or "",
            "fact_freshness_policy": "Проверять актуальный источник перед ответом." if required_facts else "",
            "fallback_when_fact_missing": "Передать менеджеру и не называть конкретные условия." if required_facts else "",
            "forbidden_promises": list(theme.get("forbidden_promises") or ()),
            "classification_method": prediction.get("classification_method") or "codex_cli_full_v2",
            "llm_confidence": prediction.get("confidence"),
            "llm_reasoning": prediction.get("reasoning") or "",
            "llm_model": prediction.get("model") or "",
            "llm_reasoning_effort": prediction.get("reasoning_effort") or "",
            "llm_prompt_version": prediction.get("prompt_version") or "",
            "llm_taxonomy_sha256": prediction.get("taxonomy_sha256") or "",
            "llm_batch_id": prediction.get("batch_id") or "",
            "llm_response_sha256": prediction.get("response_sha256") or "",
        }
    )
    fact_types = tuple(_fact_type_from_key(value) for value in required_facts)
    return replace(
        item,
        question_class_id=class_id,
        intent=theme_id,
        answer_evidence_status=answer_status,
        requires_dynamic_facts=bool(required_facts),
        dynamic_fact_types=tuple(dict.fromkeys(value for value in fact_types if value)),
        fact_freshness_required="manual_current_fact_check" if required_facts else item.fact_freshness_required,
        metadata=metadata,
    )


def policy_for_theme(theme_id: str, *, default_permission: str, required_facts: Sequence[str]) -> tuple[str, str]:
    if theme_id == "service:S1_non_question":
        return ANSWER_STATUS_NOT_CUSTOMER_QUESTION, BOT_PERMISSION_NOT_ALLOWED
    if theme_id == "service:S2_unclear":
        return ANSWER_STATUS_NOT_ENOUGH_CONTEXT, BOT_PERMISSION_NOT_ALLOWED
    if default_permission == "manager_only":
        return ANSWER_STATUS_MANAGER_ONLY, BOT_PERMISSION_MANAGER_ONLY
    if required_facts or default_permission == "answer_after_fact_check":
        return ANSWER_STATUS_TEMPLATE_NEEDS_CURRENT_FACT, BOT_PERMISSION_ALLOWED_AFTER_FACT_CHECK
    return ANSWER_STATUS_DRAFT_NEEDS_REVIEW, BOT_PERMISSION_DRAFT_ONLY


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _fact_type_from_key(value: str) -> str:
    head = str(value or "").split(".", 1)[0].strip()
    if head.endswith("s"):
        head = head[:-1]
    return head

