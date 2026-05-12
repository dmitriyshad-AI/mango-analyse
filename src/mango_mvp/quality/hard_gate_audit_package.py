from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LIMIT = 200
DEFAULT_SEED = "hard_gate_non_conversation_audit_2026_05"


@dataclass(frozen=True)
class HardGateAuditPackageConfig:
    candidates_csv: Path
    out_root: Path
    project_root: Path = Path(".")
    limit: int = DEFAULT_LIMIT
    seed: str = DEFAULT_SEED


def build_hard_gate_audit_package(config: HardGateAuditPackageConfig) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    out_root = _resolve_under_project(config.out_root, project_root)
    out_root.mkdir(parents=True, exist_ok=True)

    candidates = _read_csv(_resolve_under_project(config.candidates_csv, project_root))
    selected = select_stratified_candidates(candidates, limit=config.limit, seed=config.seed)
    all_strata = Counter(_stratum_key(row) for row in candidates)
    selected_strata = Counter(_stratum_key(row) for row in selected)
    items = [
        _build_audit_item(
            candidate,
            idx=idx,
            project_root=project_root,
            stratum_total=all_strata[_stratum_key(candidate)],
            stratum_sampled=selected_strata[_stratum_key(candidate)],
        )
        for idx, candidate in enumerate(selected, start=1)
    ]

    outputs = {
        "items_jsonl": out_root / "audit_items.jsonl",
        "items_csv": out_root / "audit_items_preview.csv",
        "strata_summary_csv": out_root / "strata_summary.csv",
        "private_mapping_csv": out_root / "PRIVATE_mapping_for_apply_do_not_edit.csv",
        "prompt_ru_md": out_root / "AUDIT_PROMPT_RU.md",
        "readme_md": out_root / "README_FOR_CLAUDE_AND_GPT.md",
        "decisions_template_jsonl": out_root / "decisions_template.jsonl",
        "expected_output_schema_json": out_root / "expected_output_schema.json",
        "summary_json": out_root / "summary.json",
    }
    _write_jsonl(outputs["items_jsonl"], items)
    _write_csv(outputs["items_csv"], [_preview_row(item) for item in items])
    _write_csv(outputs["private_mapping_csv"], [_private_mapping_row(item) for item in items])
    _write_csv(outputs["strata_summary_csv"], _strata_summary(candidates, selected))
    outputs["prompt_ru_md"].write_text(_audit_prompt(), encoding="utf-8")
    outputs["readme_md"].write_text(_readme(outputs), encoding="utf-8")
    _write_jsonl(outputs["decisions_template_jsonl"], [_auditor_required_output(item["audit_id"]) for item in items])
    outputs["expected_output_schema_json"].write_text(
        json.dumps(_expected_output_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "audit_package_read_only",
        "candidates_csv": str(_resolve_under_project(config.candidates_csv, project_root)),
        "project_root": str(project_root),
        "limit": config.limit,
        "seed": config.seed,
        "total_candidates": len(candidates),
        "selected": len(items),
        "selected_by_current_call_type": dict(Counter(item["current_state"]["call_type"] for item in items)),
        "selected_by_subtype": dict(Counter(item["guardrail"]["recommended_contact_subtype"] for item in items)),
        "selected_by_month": dict(Counter(item["call"]["month"] for item in items)),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def select_stratified_candidates(
    rows: list[dict[str, str]],
    *,
    limit: int = DEFAULT_LIMIT,
    seed: str = DEFAULT_SEED,
) -> list[dict[str, str]]:
    target = min(max(0, int(limit)), len(rows))
    if target == 0:
        return []

    by_stratum: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_stratum[_stratum_key(row)].append(row)

    quotas = _stratum_quotas(by_stratum, target)
    selected: list[dict[str, str]] = []
    selected_keys: set[str] = set()

    for stratum, quota in sorted(quotas.items()):
        bucket = list(by_stratum[stratum])
        bucket.sort(key=lambda row: _stable_hash(row, seed=f"{seed}::{stratum}"))
        for row in bucket[:quota]:
            key = _candidate_key(row)
            selected_keys.add(key)
            selected.append(row)

    if len(selected) < target:
        leftovers = [row for row in rows if _candidate_key(row) not in selected_keys]
        leftovers.sort(key=lambda row: _stable_hash(row, seed=f"{seed}::fill"))
        for row in leftovers[: target - len(selected)]:
            selected.append(row)

    selected.sort(key=lambda row: (_clean(row.get("month")), _clean(row.get("current_call_type")), _stable_hash(row, seed)))
    return selected[:target]


def _stratum_quotas(by_stratum: dict[tuple[str, str, str], list[dict[str, str]]], target: int) -> dict[tuple[str, str, str], int]:
    if target <= 0:
        return {}
    strata = {key: len(value) for key, value in by_stratum.items() if value}
    if not strata:
        return {}

    if target < len(strata):
        ranked = sorted(strata, key=lambda key: (strata[key], key), reverse=True)
        return {key: 1 for key in ranked[:target]}

    quotas = {key: 1 for key in strata}
    remaining = target - len(strata)
    if remaining <= 0:
        return quotas

    extras = _largest_remainder_allocation(
        {key: max(size - 1, 0) for key, size in strata.items()},
        remaining,
    )
    for key, extra in extras.items():
        quotas[key] += extra
    return quotas


def _largest_remainder_allocation(weights: dict[tuple[str, str, str], int], target: int) -> dict[tuple[str, str, str], int]:
    if target <= 0 or not weights:
        return {}
    total_weight = sum(max(0, weight) for weight in weights.values())
    if total_weight <= 0:
        return {key: 0 for key in weights}

    non_empty = [key for key, weight in weights.items() if weight > 0]
    allocation = {key: 0 for key in weights}
    exact: dict[tuple[str, str, str], float] = {
        key: target * (weights[key] / total_weight)
        for key in non_empty
    }
    for key, value in exact.items():
        allocation[key] = int(math.floor(value))
    used = sum(allocation.values())
    remainder = max(0, target - used)
    ranked = sorted(non_empty, key=lambda key: (exact[key] - math.floor(exact[key]), weights[key], key), reverse=True)
    for key in ranked[:remainder]:
        allocation[key] += 1
    return allocation


def _build_audit_item(
    candidate: dict[str, str],
    *,
    idx: int,
    project_root: Path,
    stratum_total: int,
    stratum_sampled: int,
) -> dict[str, Any]:
    row = _fetch_call_row(candidate, project_root=project_root)
    current_analysis = _safe_json_object(row.get("analysis_json"))
    transcript = _transcript_payload(row)
    return {
        "audit_id": f"hgate200_{idx:04d}",
        "task_id": f"hard_gate::{_clean(candidate.get('db'))}::{_clean(candidate.get('id'))}",
        "audit_goal": (
            "Проверить, безопасно ли исправить старый анализ звонка на non_conversation "
            "и очистить продажные поля."
        ),
        "call": {
            "db": _clean(candidate.get("db")),
            "call_record_id": _clean(candidate.get("id")),
            "source_filename": _clean(candidate.get("source_filename")),
            "started_at": _clean(candidate.get("started_at")),
            "month": _clean(candidate.get("month")),
            "phone": _clean(candidate.get("phone")),
            "manager_name": _clean(candidate.get("manager_name")),
            "duration_sec": _clean(candidate.get("duration_sec")),
        },
        "stratum": {
            "month": _clean(candidate.get("month")),
            "current_call_type": _clean(candidate.get("current_call_type")),
            "recommended_contact_subtype": _clean(candidate.get("guardrail_recommended_contact_subtype")),
            "total_candidates_in_stratum": stratum_total,
            "sampled_candidates_in_stratum": stratum_sampled,
            "sampling_weight": round(stratum_total / stratum_sampled, 6) if stratum_sampled else None,
        },
        "current_state": {
            "call_type": _clean(candidate.get("current_call_type")),
            "follow_up_score": _clean(candidate.get("current_follow_up_score")),
            "next_step": _clean(candidate.get("current_next_step")),
            "products": _split_list(candidate.get("current_products")),
            "subjects": _split_list(candidate.get("current_subjects")),
            "objections": _split_list(candidate.get("current_objections")),
            "history_summary_excerpt": _clean(candidate.get("current_history_summary_excerpt")),
            "analysis": _analysis_for_audit(current_analysis),
        },
        "proposed_change": {
            "decision_under_test": "safe_apply_non_conversation_hard_gate",
            "new_call_type": _clean(candidate.get("normalized_call_type")) or "non_conversation",
            "new_follow_up_score": _clean(candidate.get("normalized_follow_up_score")),
            "new_next_step": _clean(candidate.get("normalized_next_step")),
            "new_products": _split_list(candidate.get("normalized_products")),
            "new_subjects": _split_list(candidate.get("normalized_subjects")),
            "new_objections": _split_list(candidate.get("normalized_objections")),
            "new_history_summary_excerpt": _clean(candidate.get("normalized_history_summary_excerpt")),
            "fields_to_clear": [
                "follow_up_score",
                "next_step",
                "products",
                "subjects",
                "objections",
                "commercial",
                "lead_priority",
            ],
        },
        "guardrail": {
            "label": _clean(candidate.get("guardrail_label")),
            "score": _clean(candidate.get("guardrail_score")),
            "reason_codes": _split_list(candidate.get("guardrail_reason_codes")),
            "should_force_non_conversation": _clean(candidate.get("guardrail_should_force_non_conversation")),
            "requires_manual_review": _clean(candidate.get("guardrail_requires_manual_review")),
            "protected_live_dialogue": _clean(candidate.get("guardrail_protected_live_dialogue")),
            "recommended_contact_subtype": _clean(candidate.get("guardrail_recommended_contact_subtype")),
        },
        "transcript": transcript,
        "auditor_required_output": _auditor_required_output(f"hgate200_{idx:04d}"),
    }


def _fetch_call_row(candidate: dict[str, str], *, project_root: Path) -> dict[str, Any]:
    db_path = _resolve_under_project(Path(_clean(candidate.get("db"))), project_root)
    call_id = int(_clean(candidate.get("id")) or 0)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            """
            select
                id,
                source_file,
                source_filename,
                phone,
                manager_name,
                duration_sec,
                started_at,
                transcript_manager,
                transcript_client,
                transcript_text,
                transcript_variants_json,
                resolve_json,
                analysis_json
            from call_records
            where id = ?
            """,
            (call_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Candidate not found in DB: {db_path} id={call_id}")
        return dict(row)
    finally:
        con.close()


def _transcript_payload(row: dict[str, Any]) -> dict[str, Any]:
    manager = _clean(row.get("transcript_manager"))
    client = _clean(row.get("transcript_client"))
    final_text = _clean(row.get("transcript_text"))
    if not final_text and (manager or client):
        final_text = "\n\n".join(part for part in [f"MANAGER:\n{manager}" if manager else "", f"CLIENT:\n{client}" if client else ""] if part)
    return {
        "final_text": final_text,
        "manager_text": manager,
        "client_text": client,
        "final_text_chars": len(final_text),
        "manager_text_chars": len(manager),
        "client_text_chars": len(client),
    }


def _analysis_for_audit(payload: dict[str, Any]) -> dict[str, Any]:
    quality_flags = _dict(payload.get("quality_flags"))
    structured = _dict(payload.get("structured_fields"))
    interests = _dict(structured.get("interests"))
    next_step = structured.get("next_step")
    return {
        "analysis_schema_version": payload.get("analysis_schema_version"),
        "history_summary": payload.get("history_summary"),
        "call_type": quality_flags.get("call_type"),
        "follow_up_score": payload.get("follow_up_score"),
        "target_product": payload.get("target_product"),
        "next_step": payload.get("next_step") or next_step,
        "products": interests.get("products"),
        "subjects": interests.get("subjects"),
        "objections": payload.get("objections") or structured.get("objections"),
        "tags": payload.get("tags"),
        "quality_flags": quality_flags,
    }


def _auditor_required_output(audit_id: str) -> dict[str, Any]:
    return {
        "audit_id": audit_id,
        "decision": "safe_apply | keep_current | manual_review",
        "confidence": 0.0,
        "reason_ru": "коротко объяснить решение",
        "evidence": ["1-3 цитаты/признака из transcript/current_state"],
        "is_live_customer_dialogue": False,
        "recommended_call_type": "non_conversation | sales_call | service_call | technical_call | existing_client_progress | unknown",
    }


def _strata_summary(all_rows: list[dict[str, str]], selected: list[dict[str, str]]) -> list[dict[str, Any]]:
    all_counter: Counter[tuple[str, str, str]] = Counter()
    selected_counter: Counter[tuple[str, str, str]] = Counter()
    for row in all_rows:
        all_counter[_stratum_key(row)] += 1
    for row in selected:
        selected_counter[_stratum_key(row)] += 1
    keys = sorted(set(all_counter) | set(selected_counter))
    return [
        {
            "month": key[0],
            "current_call_type": key[1],
            "recommended_contact_subtype": key[2],
            "total_candidates": all_counter[key],
            "sampled": selected_counter[key],
        }
        for key in keys
    ]


def _preview_row(item: dict[str, Any]) -> dict[str, Any]:
    transcript = _dict(item.get("transcript"))
    call = _dict(item.get("call"))
    current = _dict(item.get("current_state"))
    proposed = _dict(item.get("proposed_change"))
    guardrail = _dict(item.get("guardrail"))
    return {
        "audit_id": item.get("audit_id"),
        "month": call.get("month"),
        "current_call_type": current.get("call_type"),
        "recommended_contact_subtype": guardrail.get("recommended_contact_subtype"),
        "source_filename": call.get("source_filename"),
        "started_at": call.get("started_at"),
        "duration_sec": call.get("duration_sec"),
        "manager_name": call.get("manager_name"),
        "guardrail_reason_codes": "|".join(guardrail.get("reason_codes") or []),
        "current_history_summary_excerpt": current.get("history_summary_excerpt"),
        "proposed_call_type": proposed.get("new_call_type"),
        "proposed_history_summary_excerpt": proposed.get("new_history_summary_excerpt"),
        "transcript_excerpt": _excerpt(transcript.get("final_text"), 1800),
    }


def _private_mapping_row(item: dict[str, Any]) -> dict[str, Any]:
    call = _dict(item.get("call"))
    return {
        "audit_id": item.get("audit_id"),
        "task_id": item.get("task_id"),
        "db": call.get("db"),
        "call_record_id": call.get("call_record_id"),
        "source_filename": call.get("source_filename"),
        "phone": call.get("phone"),
        "started_at": call.get("started_at"),
        "manager_name": call.get("manager_name"),
    }


def _audit_prompt() -> str:
    return """# Аудит hard gate non_conversation на 200 кандидатах

Проверь файл `audit_items.jsonl`. Каждый объект — один звонок, который наш новый hard gate предлагает исправить: старый анализ был sales/service/technical/existing_client_progress, а новый deterministic hard gate предлагает перевести звонок в `non_conversation` и очистить продажные поля.

## Главная задача

Нужно оценить, безопасно ли автоматически применить это исправление.

## Верни результат

Верни JSONL: один объект на строку, строго в таком формате:

```json
{
  "audit_id": "hgate200_0001",
  "decision": "safe_apply | keep_current | manual_review",
  "confidence": 0.0,
  "reason_ru": "короткое объяснение на русском",
  "evidence": ["1-3 коротких признака или цитаты из transcript/current_state"],
  "is_live_customer_dialogue": false,
  "recommended_call_type": "non_conversation | sales_call | service_call | technical_call | existing_client_progress | unknown"
}
```

## Решения

- `safe_apply`: в расшифровке нет живого клиентского диалога; это автоответчик, голосовая почта, IVR, виртуальный секретарь, абонент недоступен/занят/вне зоны, короткий технический дозвон без участия клиента или менеджер говорит в автоответчик. Исправление на `non_conversation` безопасно.
- `keep_current`: в расшифровке есть реальный живой диалог менеджера и клиента; нельзя переводить в `non_conversation`.
- `manual_review`: по тексту нельзя уверенно решить: ASR сильно испорчен, признаки противоречат друг другу, есть фрагменты возможного живого диалога, но он неочевиден.

## Важные правила

- Содержательное сообщение менеджера на автоответчик всё равно считай `non_conversation`: клиент не участвовал в диалоге, факт прослушивания не подтвержден.
- Голосовые ассистенты и виртуальные секретари тоже `non_conversation`: «я секретарь», «голосовой ассистент», «ассистент Миа», «вас приветствует компания», «все разговоры записываются», «нажмите 1».
- Системные фразы оператора связи тоже `non_conversation`: «абонент недоступен», «вне зоны действия», «оставьте сообщение», «после звукового сигнала», «отправить бесплатное SMS».
- Если есть живой диалог про оплату, Сбер/Альфа/QR/банк, это не IVR само по себе. Такие звонки нельзя автоматически отправлять в `non_conversation`.
- Если звучит «оставайтесь на линии» или «продолжаем дозваниваться», но дальше есть живой разговор, это не `non_conversation`.
- Не используй аудио. Проверка делается только по расшифровке и текущему анализу.
"""


def _readme(outputs: dict[str, Path]) -> str:
    return (
        "# Audit package для Claude/GPT\n\n"
        "Цель: проверить 200 стратифицированных кандидатов, которых hard gate предлагает перевести в `non_conversation`.\n\n"
        "Файлы:\n\n"
        f"- `{outputs['items_jsonl'].name}`: основной файл для Claude/GPT, содержит полные расшифровки и текущий/предлагаемый анализ.\n"
        f"- `{outputs['items_csv'].name}`: preview-таблица для быстрого просмотра человеком.\n"
        f"- `{outputs['strata_summary_csv'].name}`: контроль стратификации по месяцу, старому типу звонка и подтипу.\n"
        f"- `{outputs['private_mapping_csv'].name}`: локальная карта audit_id -> db/id/phone для последующего применения решений. Не редактировать.\n"
        f"- `{outputs['prompt_ru_md'].name}`: промт для Claude или GPT.\n\n"
        f"- `{outputs['decisions_template_jsonl'].name}`: шаблон ожидаемого ответа по всем audit_id.\n"
        f"- `{outputs['expected_output_schema_json'].name}`: машинно-читаемая схема ожидаемого ответа.\n\n"
        "Как использовать:\n\n"
        "1. Дай модели файл `AUDIT_PROMPT_RU.md` и `audit_items.jsonl`.\n"
        "2. Попроси вернуть JSONL с решениями по каждому `audit_id`.\n"
        "3. Сохрани ответ модели отдельным файлом, например `claude_decisions.jsonl` или `gpt_decisions.jsonl`.\n"
    )


def _expected_output_schema() -> dict[str, Any]:
    return {
        "type": "jsonl",
        "one_object_per_line": True,
        "required_fields": {
            "audit_id": "string, one of audit_items.jsonl audit_id",
            "decision": "safe_apply | keep_current | manual_review",
            "confidence": "number between 0.0 and 1.0",
            "reason_ru": "short Russian explanation",
            "evidence": "array of 1-3 strings from transcript/current_state",
            "is_live_customer_dialogue": "boolean",
            "recommended_call_type": (
                "non_conversation | sales_call | service_call | technical_call | "
                "existing_client_progress | unknown"
            ),
        },
        "decision_meaning": {
            "safe_apply": "Можно безопасно применить hard gate и перевести звонок в non_conversation.",
            "keep_current": "Есть живой содержательный диалог; hard gate применять нельзя.",
            "manual_review": "Нужна дополнительная проверка: transcript или evidence неоднозначны.",
        },
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stable_hash(row: dict[str, str], seed: str) -> str:
    value = f"{seed}|{_candidate_key(row)}"
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _candidate_key(row: dict[str, str]) -> str:
    return f"{_clean(row.get('db'))}::{_clean(row.get('id'))}::{_clean(row.get('source_filename'))}"


def _stratum_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        _clean(row.get("month")) or "unknown_month",
        _clean(row.get("current_call_type")) or "unknown_type",
        _clean(row.get("guardrail_recommended_contact_subtype")) or "unknown_subtype",
    )


def _safe_json_object(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        payload = json.loads(str(raw or "{}"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _split_list(value: Any) -> list[str]:
    text = _clean(value)
    if not text:
        return []
    return [part.strip() for part in text.replace(";", "|").split("|") if part.strip()]


def _excerpt(value: Any, limit: int) -> str:
    text = _clean(value)
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def _resolve_under_project(path: Path, project_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a stratified Claude/GPT audit package for hard-gate candidates.")
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> HardGateAuditPackageConfig:
    return HardGateAuditPackageConfig(
        candidates_csv=args.candidates_csv,
        out_root=args.out_root,
        project_root=args.project_root,
        limit=args.limit,
        seed=args.seed,
    )
