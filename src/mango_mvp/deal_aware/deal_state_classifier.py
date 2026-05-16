from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.deal_aware.deal_attribution import CONFIDENCE_LOW_WARNING_THRESHOLD, CONFIDENCE_MEDIUM_THRESHOLD
from mango_mvp.deal_aware.stage1_snapshot import quote_ident, read_csv, safe_text, stringify, write_csv


SCHEMA_VERSION = "deal_aware_stage3_deal_state_v1"


ACTIVE_STATUS_MARKERS = (
    "перспектива",
    "в работе",
    "ожидание оплаты",
    "принимают решение",
    "переговоры",
    "заключение договора",
    "запись в группу",
)
WON_STATUS_MARKERS = ("оплата получена", "успешно")
CONTACT_PROBLEM_MARKERS = ("проблема с контактом", "недозвон")
LOST_STATUS_MARKERS = ("закрыто и не реализовано",)

LOSS_DUPLICATE_MARKERS = ("дубль", "объедин")
LOSS_EXISTING_CLIENT_MARKERS = ("действующий клиент", "выпускник")
LOSS_NOISE_MARKERS = ("спам", "не квал", "не оставлял заявку", "тест", "жуковский")
LOSS_REJECTED_MARKERS = (
    "не актуально",
    "дорого",
    "конкурент",
    "не подходит",
    "репетитор",
    "не подошло",
    "архив",
)
PAYMENT_NEXT_STEP_MARKERS = ("оплат", "ссылк", "платеж", "договор", "чек")


@dataclass(frozen=True)
class DealStatePaths:
    stage2_attribution_root: Path
    amo_live_snapshot_root: Path
    out_root: Path


def build_deal_state_classifier(paths: DealStatePaths) -> dict[str, Any]:
    paths.out_root.mkdir(parents=True, exist_ok=True)
    links = read_csv(paths.stage2_attribution_root / "deal_call_links.csv")
    deals = read_csv(paths.amo_live_snapshot_root / "amo_deals_snapshot.csv")
    tasks = read_csv(paths.amo_live_snapshot_root / "amo_tasks_snapshot.csv")
    stage2_summary = load_json(paths.stage2_attribution_root / "summary.json")
    live_summary = load_json(paths.amo_live_snapshot_root / "summary.json")

    deal_by_id = {safe_text(row.get("lead_id")): row for row in deals if safe_text(row.get("lead_id"))}
    task_meta_by_deal = aggregate_tasks(tasks)
    deal_state_rows = [
        classify_deal_state(deal, task_meta_by_deal.get(safe_text(deal.get("lead_id")), {}))
        for deal in deals
    ]
    deal_state_by_id = {row["deal_id"]: row for row in deal_state_rows if row.get("deal_id")}
    policy_rows = [
        classify_call_policy(
            link,
            deal_by_id=deal_by_id,
            deal_state_by_id=deal_state_by_id,
            task_meta_by_deal=task_meta_by_deal,
        )
        for link in links
    ]

    stage4_candidates = [row for row in policy_rows if row.get("safe_for_stage4_generation") == "Да"]
    stage4_deal_candidates = build_stage4_deal_candidates(stage4_candidates)
    manual_review = [row for row in policy_rows if row.get("stage3_bucket") == "manual_review"]
    blocked = [row for row in policy_rows if row.get("stage3_bucket") == "blocked"]
    distribution = distribution_rows(policy_rows)

    outputs = {
        "deal_state_csv": paths.out_root / "deal_state_by_deal.csv",
        "call_policy_csv": paths.out_root / "deal_call_writeback_policy.csv",
        "stage4_candidates_csv": paths.out_root / "deal_stage4_generation_candidates.csv",
        "stage4_deal_candidates_csv": paths.out_root / "deal_stage4_deal_candidates.csv",
        "manual_review_csv": paths.out_root / "deal_stage3_manual_review.csv",
        "blocked_csv": paths.out_root / "deal_stage3_blocked.csv",
        "distribution_csv": paths.out_root / "deal_stage3_distribution.csv",
        "sqlite": paths.out_root / "deal_aware_stage3_deal_state.sqlite",
        "summary_json": paths.out_root / "summary.json",
        "readme": paths.out_root / "README.md",
    }
    write_csv(outputs["deal_state_csv"], deal_state_rows)
    write_csv(outputs["call_policy_csv"], policy_rows)
    write_csv(outputs["stage4_candidates_csv"], stage4_candidates)
    write_csv(outputs["stage4_deal_candidates_csv"], stage4_deal_candidates)
    write_csv(outputs["manual_review_csv"], manual_review)
    write_csv(outputs["blocked_csv"], blocked)
    write_csv(outputs["distribution_csv"], distribution)
    write_sqlite(
        outputs["sqlite"],
        {
            "deal_state_by_deal": deal_state_rows,
            "deal_call_writeback_policy": policy_rows,
            "stage4_candidates": stage4_candidates,
            "stage4_deal_candidates": stage4_deal_candidates,
            "manual_review": manual_review,
            "blocked": blocked,
            "distribution": distribution,
        },
    )
    summary = build_summary(
        paths=paths,
        links=links,
        deals=deals,
        tasks=tasks,
        policy_rows=policy_rows,
        stage4_candidates=stage4_candidates,
        stage4_deal_candidates=stage4_deal_candidates,
        manual_review=manual_review,
        blocked=blocked,
        stage2_summary=stage2_summary,
        live_summary=live_summary,
        outputs=outputs,
    )
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["readme"].write_text(render_readme(summary), encoding="utf-8")
    return summary


def classify_deal_state(deal: dict[str, str], task_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    task_meta = task_meta or {}
    deal_id = safe_text(deal.get("lead_id"))
    status_name = safe_text(deal.get("status_name"))
    loss_reason = normalize_loss_reason(deal.get("loss_reason"))
    closed_at = safe_text(deal.get("closed_at"))
    status_text = status_name.casefold()
    loss_text = loss_reason.casefold()

    if contains_any(status_text, WON_STATUS_MARKERS):
        state = "won_paid"
        bucket = "context_only"
        policy = "allow_context_only_paid_or_success"
        reason = "Сделка уже оплачена или успешно закрыта; нельзя генерировать продающий следующий шаг."
    elif contains_any(status_text, LOST_STATUS_MARKERS) or (closed_at and loss_reason):
        state, bucket, policy, reason = classify_lost_deal(loss_text)
    elif contains_any(status_text, CONTACT_PROBLEM_MARKERS):
        state = "contact_problem_or_no_answer"
        bucket = "manual_review"
        policy = "manual_review_contact_problem"
        reason = "Сделка в статусе недозвона/проблемы контакта; автоматическая запись может закрепить неверный следующий шаг."
    elif contains_any(status_text, ACTIVE_STATUS_MARKERS):
        state = "active_sales"
        bucket = "full_active"
        policy = "allow_full_active_deal_context"
        reason = "Сделка открыта и находится в рабочем коммерческом статусе."
    else:
        state = "unknown_state"
        bucket = "manual_review"
        policy = "manual_review_unknown_deal_state"
        reason = "Статус сделки не попал в явно разрешённые классы."

    open_tasks = int_or_zero(task_meta.get("open_task_count"))
    overdue_tasks = int_or_zero(task_meta.get("overdue_task_count"))
    return {
        "deal_id": deal_id,
        "deal_name": safe_text(deal.get("lead_name")),
        "pipeline_id": safe_text(deal.get("pipeline_id")),
        "pipeline_name": safe_text(deal.get("pipeline_name")),
        "status_id": safe_text(deal.get("status_id")),
        "status_name": status_name,
        "loss_reason": loss_reason,
        "closed_at": closed_at,
        "responsible_user_name": safe_text(deal.get("responsible_user_name")),
        "linked_contact_ids": safe_text(deal.get("linked_contact_ids")),
        "deal_state_class": state,
        "deal_state_bucket": bucket,
        "deal_state_policy": policy,
        "deal_state_reason_ru": reason,
        "open_task_count": open_tasks,
        "overdue_task_count": overdue_tasks,
        "latest_open_task_at": safe_text(task_meta.get("latest_open_task_at")),
        "latest_open_task_text": safe_text(task_meta.get("latest_open_task_text"))[:240],
    }


def classify_lost_deal(loss_text: str) -> tuple[str, str, str, str]:
    if contains_any(loss_text, LOSS_DUPLICATE_MARKERS):
        return (
            "closed_duplicate",
            "manual_review",
            "manual_review_duplicate_or_merged_deal",
            "Сделка закрыта как дубль; нужно искать реальную активную сделку/карточку, а не писать сюда.",
        )
    if contains_any(loss_text, LOSS_EXISTING_CLIENT_MARKERS):
        return (
            "closed_existing_client",
            "manual_review",
            "manual_review_existing_client_redirect",
            "Сделка закрыта как действующий клиент/выпускник; вероятно работа ведётся в другой сделке или карточке.",
        )
    if contains_any(loss_text, LOSS_NOISE_MARKERS):
        return (
            "closed_noise_or_wrong_request",
            "blocked",
            "block_noise_or_wrong_request",
            "Причина отказа относится к мусорному/нецелевому классу; автоматическая запись в сделку запрещена.",
        )
    if contains_any(loss_text, LOSS_REJECTED_MARKERS):
        return (
            "closed_rejected_or_not_actual",
            "blocked",
            "block_closed_rejected_historical_only",
            "Сделка закрыта по коммерческому отказу/неактуальности; можно хранить историю в контакте, но не оживлять сделку автоматически.",
        )
    return (
        "closed_lost_other",
        "manual_review",
        "manual_review_closed_lost_other",
        "Сделка закрыта как нереализованная, но причина не классифицирована надёжно.",
    )


def classify_call_policy(
    link: dict[str, str],
    *,
    deal_by_id: dict[str, dict[str, str]],
    deal_state_by_id: dict[str, dict[str, Any]],
    task_meta_by_deal: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stage2_decision = safe_text(link.get("attribution_decision"))
    deal_id = safe_text(link.get("selected_deal_id"))
    base = {
        **link,
        "stage3_bucket": "",
        "stage3_decision": "",
        "stage3_reason_ru": "",
        "deal_writeback_mode": "none",
        "safe_for_stage4_generation": "Нет",
        "safe_for_live_deal_writeback_now": "Нет",
        "stage3_risk_flags": "",
        "open_task_count": "0",
        "overdue_task_count": "0",
        "latest_open_task_at": "",
        "latest_open_task_text": "",
    }

    state = deal_state_by_id.get(deal_id) if deal_id else None
    task_meta = task_meta_by_deal.get(deal_id, {}) if deal_id else {}
    if state:
        base = base | {
            "deal_state_class": safe_text(state.get("deal_state_class")),
            "deal_state_bucket": safe_text(state.get("deal_state_bucket")),
            "deal_state_policy": safe_text(state.get("deal_state_policy")),
            "deal_state_reason_ru": safe_text(state.get("deal_state_reason_ru")),
            "open_task_count": stringify(task_meta.get("open_task_count", 0)),
            "overdue_task_count": stringify(task_meta.get("overdue_task_count", 0)),
            "latest_open_task_at": safe_text(task_meta.get("latest_open_task_at")),
            "latest_open_task_text": safe_text(task_meta.get("latest_open_task_text"))[:240],
        }

    if stage2_decision == "manual_review_single_terminal_deal_candidate" and state:
        state_bucket = safe_text(state.get("deal_state_bucket"))
        score = float_or_zero(link.get("confidence_score"))
        if state_bucket == "context_only" and score >= CONFIDENCE_MEDIUM_THRESHOLD:
            risk_flags = policy_risk_flags(link, state)
            return base | {
                "stage3_bucket": "allow",
                "stage3_decision": "allow_stage4_context_only_paid_deal_generation",
                "stage3_reason_ru": "Оплаченная/успешная сделка; Stage 4 должен писать только контекст и сервисный следующий шаг, без продажи.",
                "deal_writeback_mode": "context_only_paid_or_success",
                "safe_for_stage4_generation": "Да",
                "stage3_risk_flags": " | ".join(risk_flags),
            }
        if state_bucket == "context_only":
            return base | {
                "stage3_bucket": "manual_review",
                "stage3_decision": "manual_review_low_stage2_confidence",
                "stage3_reason_ru": f"Оплаченная/успешная сделка найдена, но confidence ниже порога {CONFIDENCE_MEDIUM_THRESHOLD:.2f}.",
            }
        if state_bucket == "blocked":
            return base | {
                "stage3_bucket": "blocked",
                "stage3_decision": safe_text(state.get("deal_state_policy")),
                "stage3_reason_ru": safe_text(state.get("deal_state_reason_ru")),
            }
        return base | {
            "stage3_bucket": "manual_review",
            "stage3_decision": safe_text(state.get("deal_state_policy")) or "manual_review_terminal_deal_candidate",
            "stage3_reason_ru": safe_text(state.get("deal_state_reason_ru")),
        }

    if stage2_decision != "linked_single_deal_candidate":
        return base | {
            "stage3_bucket": "blocked" if stage2_decision.startswith("skipped") else "manual_review",
            "stage3_decision": f"not_eligible_stage2_{stage2_decision or 'unknown'}",
            "stage3_reason_ru": "Stage 2 не дал единственную надёжную открытую связку звонок-сделка.",
        }
    if not deal_id or deal_id not in deal_by_id:
        return base | {
            "stage3_bucket": "manual_review",
            "stage3_decision": "manual_review_missing_live_deal_row",
            "stage3_reason_ru": "В Stage 2 есть selected_deal_id, но такой сделки нет в свежем AMO snapshot.",
        }

    state = state or classify_deal_state(deal_by_id[deal_id], task_meta_by_deal.get(deal_id, {}))
    score = float_or_zero(link.get("confidence_score"))
    risk_flags = policy_risk_flags(link, state)
    task_meta = task_meta_by_deal.get(deal_id, {})
    base = base | {
        "deal_state_class": safe_text(state.get("deal_state_class")),
        "deal_state_bucket": safe_text(state.get("deal_state_bucket")),
        "deal_state_policy": safe_text(state.get("deal_state_policy")),
        "deal_state_reason_ru": safe_text(state.get("deal_state_reason_ru")),
        "open_task_count": stringify(task_meta.get("open_task_count", 0)),
        "overdue_task_count": stringify(task_meta.get("overdue_task_count", 0)),
        "latest_open_task_at": safe_text(task_meta.get("latest_open_task_at")),
        "latest_open_task_text": safe_text(task_meta.get("latest_open_task_text"))[:240],
        "stage3_risk_flags": " | ".join(risk_flags),
    }

    if score < CONFIDENCE_MEDIUM_THRESHOLD:
        return base | {
            "stage3_bucket": "manual_review",
            "stage3_decision": "manual_review_low_stage2_confidence",
            "stage3_reason_ru": f"Связка звонок-сделка есть, но confidence ниже порога {CONFIDENCE_MEDIUM_THRESHOLD:.2f}.",
        }

    state_bucket = safe_text(state.get("deal_state_bucket"))
    state_policy = safe_text(state.get("deal_state_policy"))
    if state_bucket == "full_active":
        return base | {
            "stage3_bucket": "allow",
            "stage3_decision": "allow_stage4_full_active_deal_generation",
            "stage3_reason_ru": "Открытая рабочая сделка; можно передавать в Stage 4 для генерации deal-aware полей.",
            "deal_writeback_mode": "full_active",
            "safe_for_stage4_generation": "Да",
        }
    if state_bucket == "context_only":
        return base | {
            "stage3_bucket": "allow",
            "stage3_decision": "allow_stage4_context_only_paid_deal_generation",
            "stage3_reason_ru": "Оплаченная/успешная сделка; Stage 4 должен писать только контекст и сервисный следующий шаг, без продажи.",
            "deal_writeback_mode": "context_only_paid_or_success",
            "safe_for_stage4_generation": "Да",
        }
    if state_bucket == "blocked":
        return base | {
            "stage3_bucket": "blocked",
            "stage3_decision": state_policy,
            "stage3_reason_ru": safe_text(state.get("deal_state_reason_ru")),
        }
    return base | {
        "stage3_bucket": "manual_review",
        "stage3_decision": state_policy or "manual_review_unknown_deal_state",
        "stage3_reason_ru": safe_text(state.get("deal_state_reason_ru")) or "Требуется ручная проверка состояния сделки.",
    }


def aggregate_tasks(tasks: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    now = datetime.now(timezone.utc)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for task in tasks:
        deal_id = safe_text(task.get("entity_id"))
        if deal_id:
            grouped[deal_id].append(task)

    result: dict[str, dict[str, Any]] = {}
    for deal_id, rows in grouped.items():
        open_rows = [row for row in rows if not is_true(row.get("is_completed"))]
        overdue_rows = [
            row for row in open_rows if (dt := parse_dt(row.get("complete_till"))) is not None and dt < now
        ]
        latest = max(open_rows, key=lambda row: safe_text(row.get("complete_till")), default={})
        result[deal_id] = {
            "task_count": len(rows),
            "open_task_count": len(open_rows),
            "overdue_task_count": len(overdue_rows),
            "latest_open_task_at": safe_text(latest.get("complete_till")),
            "latest_open_task_text": safe_text(latest.get("text") or latest.get("result")),
        }
    return result


def policy_risk_flags(link: dict[str, str], state: dict[str, Any]) -> list[str]:
    flags = []
    next_step = safe_text(link.get("call_next_step")).casefold()
    state_class = safe_text(state.get("deal_state_class"))
    if state_class == "won_paid" and contains_any(next_step, PAYMENT_NEXT_STEP_MARKERS):
        flags.append("paid_deal_has_payment_next_step_in_call")
    if int_or_zero(state.get("overdue_task_count")) > 0:
        flags.append("deal_has_overdue_open_tasks")
    if float_or_zero(link.get("confidence_score")) < CONFIDENCE_LOW_WARNING_THRESHOLD:
        flags.append("stage2_confidence_low")
    return flags


def distribution_rows(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: Counter[tuple[str, str, str]] = Counter(
        (
            safe_text(row.get("stage3_bucket")),
            safe_text(row.get("stage3_decision")),
            safe_text(row.get("deal_writeback_mode")),
        )
        for row in policy_rows
    )
    return [
        {
            "stage3_bucket": bucket,
            "stage3_decision": decision,
            "deal_writeback_mode": mode,
            "rows": count,
        }
        for (bucket, decision, mode), count in sorted(grouped.items())
    ]


def build_stage4_deal_candidates(stage4_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage4_candidates:
        deal_id = safe_text(row.get("selected_deal_id"))
        if deal_id:
            grouped[deal_id].append(row)

    result = []
    for deal_id, rows in grouped.items():
        rows.sort(key=lambda row: safe_text(row.get("started_at")))
        latest = rows[-1]
        modes = {safe_text(row.get("deal_writeback_mode")) for row in rows}
        mode = "full_active" if "full_active" in modes else "context_only_paid_or_success"
        phones = sorted({safe_text(row.get("phone")) for row in rows if safe_text(row.get("phone"))})
        managers = sorted({safe_text(row.get("manager_name")) for row in rows if safe_text(row.get("manager_name"))})
        risk_flags = sorted(
            {
                flag
                for row in rows
                for flag in safe_text(row.get("stage3_risk_flags")).split(" | ")
                if flag
            }
        )
        result.append(
            {
                "selected_deal_id": deal_id,
                "selected_deal_name": safe_text(latest.get("selected_deal_name")),
                "selected_pipeline_name": safe_text(latest.get("selected_pipeline_name")),
                "selected_status_name": safe_text(latest.get("selected_status_name")),
                "selected_loss_reason": safe_text(latest.get("selected_loss_reason")),
                "deal_writeback_mode": mode,
                "candidate_call_count": len(rows),
                "candidate_phone_count": len(phones),
                "phones": " | ".join(phones),
                "managers": " | ".join(managers),
                "first_call_at": safe_text(rows[0].get("started_at")),
                "last_call_at": safe_text(latest.get("started_at")),
                "latest_call_id": safe_text(latest.get("call_id")),
                "latest_call_next_step": safe_text(latest.get("call_next_step")),
                "latest_call_summary": safe_text(latest.get("call_summary"))[:900],
                "stage3_risk_flags": " | ".join(risk_flags),
                "safe_for_stage4_generation": "Да",
            }
        )
    result.sort(key=lambda row: (row["deal_writeback_mode"], row["last_call_at"], row["selected_deal_id"]), reverse=True)
    return result


def build_summary(
    *,
    paths: DealStatePaths,
    links: list[dict[str, str]],
    deals: list[dict[str, str]],
    tasks: list[dict[str, str]],
    policy_rows: list[dict[str, Any]],
    stage4_candidates: list[dict[str, Any]],
    stage4_deal_candidates: list[dict[str, Any]],
    manual_review: list[dict[str, Any]],
    blocked: list[dict[str, Any]],
    stage2_summary: dict[str, Any],
    live_summary: dict[str, Any],
    outputs: dict[str, Path],
) -> dict[str, Any]:
    bucket_counts = Counter(safe_text(row.get("stage3_bucket")) for row in policy_rows)
    decision_counts = Counter(safe_text(row.get("stage3_decision")) for row in policy_rows)
    mode_counts = Counter(safe_text(row.get("deal_writeback_mode")) for row in policy_rows)
    state_counts = Counter(safe_text(row.get("deal_state_class")) for row in policy_rows if row.get("deal_state_class"))
    risk_counts = Counter(
        flag
        for row in policy_rows
        for flag in safe_text(row.get("stage3_risk_flags")).split(" | ")
        if flag
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {
            "stage2_attribution_root": str(paths.stage2_attribution_root),
            "amo_live_snapshot_root": str(paths.amo_live_snapshot_root),
        },
        "safety": {
            "read_only": True,
            "write_amo": False,
            "write_tallanto": False,
            "run_asr": False,
            "run_resolve_analyze": False,
        },
        "coverage": {
            "call_policy_rows": len(policy_rows),
            "amo_deals_seen": len(deals),
            "amo_tasks_seen": len(tasks),
            "stage2_linked_rows": int_or_zero((stage2_summary.get("coverage") or {}).get("linked_rows")),
            "stage4_generation_candidates": len(stage4_candidates),
            "stage4_deal_candidates": len(stage4_deal_candidates),
            "manual_review_rows": len(manual_review),
            "blocked_rows": len(blocked),
            "allow_full_active_rows": sum(row.get("deal_writeback_mode") == "full_active" for row in stage4_candidates),
            "allow_context_only_paid_rows": sum(
                row.get("deal_writeback_mode") == "context_only_paid_or_success" for row in stage4_candidates
            ),
        },
        "bucket_counts": dict(bucket_counts.most_common()),
        "decision_counts": dict(decision_counts.most_common()),
        "mode_counts": dict(mode_counts.most_common()),
        "deal_state_counts": dict(state_counts.most_common()),
        "risk_counts": dict(risk_counts.most_common()),
        "readiness": {
            "deal_state_classifier_built": True,
            "safe_to_write_deal_fields": False,
            "safe_for_stage4_generation_rows": len(stage4_candidates),
            "requires_stage4_deal_text_builder": True,
            "requires_claude_or_rop_audit_before_live_deal_writeback": True,
        },
        "amo_live_snapshot": {
            "connected_before": bool(((live_summary.get("connection") or {}).get("before") or {}).get("connected")),
            "connected_after": bool(((live_summary.get("connection") or {}).get("after") or {}).get("connected")),
            "contacts_seen": int_or_zero((live_summary.get("fetch") or {}).get("contacts_seen")),
            "leads_seen": int_or_zero((live_summary.get("fetch") or {}).get("leads_seen")),
            "tasks_seen": int_or_zero((live_summary.get("fetch") or {}).get("tasks_seen")),
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }


def render_readme(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    return "\n".join(
        [
            "# Deal-Aware Stage 3 Deal State Classifier",
            "",
            "Read-only classifier. No AMO/Tallanto writes.",
            "",
            "## Coverage",
            "",
            f"- call policy rows: {coverage['call_policy_rows']}",
            f"- Stage 4 call-level generation candidates: {coverage['stage4_generation_candidates']}",
            f"- Stage 4 deal-level candidates: {coverage['stage4_deal_candidates']}",
            f"- full active candidates: {coverage['allow_full_active_rows']}",
            f"- context-only paid/success candidates: {coverage['allow_context_only_paid_rows']}",
            f"- manual review rows: {coverage['manual_review_rows']}",
            f"- blocked rows: {coverage['blocked_rows']}",
            "",
            "## Outputs",
            "",
            *[f"- `{key}`: `{path}`" for key, path in summary["outputs"].items()],
            "",
        ]
    )


def write_sqlite(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        for table, rows in tables.items():
            if not rows:
                con.execute(f'CREATE TABLE "{table}" (empty TEXT)')
                continue
            columns = sorted({key for row in rows for key in row.keys()})
            con.execute(f'CREATE TABLE "{table}" ({", ".join(f"{quote_ident(col)} TEXT" for col in columns)})')
            placeholders = ", ".join("?" for _ in columns)
            con.executemany(
                f'INSERT INTO "{table}" ({", ".join(quote_ident(col) for col in columns)}) VALUES ({placeholders})',
                [[stringify(row.get(col)) for col in columns] for row in rows],
            )
        con.commit()
    finally:
        con.close()


def normalize_loss_reason(value: Any) -> str:
    return " ".join(safe_text(value).split())


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def is_true(value: Any) -> bool:
    return safe_text(value).casefold() in {"1", "true", "yes", "да"}


def parse_dt(value: Any) -> datetime | None:
    text = safe_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def int_or_zero(value: Any) -> int:
    try:
        return int(float(safe_text(value).replace(",", ".")))
    except ValueError:
        return 0


def float_or_zero(value: Any) -> float:
    try:
        return float(safe_text(value).replace(",", "."))
    except ValueError:
        return 0.0


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
