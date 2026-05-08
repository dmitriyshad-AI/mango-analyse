from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from mango_mvp.amocrm_runtime.agent_models import AgentAction, AgentActionPolicy, AgentRun
from mango_mvp.amocrm_runtime.models import utc_now
from mango_mvp.amocrm_runtime.phone_context import PhoneContext, get_all_known_phones, get_phone_context

ACTION_LEVELS = {"L1", "L2", "L3", "L4"}
SAFE_DRY_RUN_MODE = "dry_run"

ActionHandler = Callable[["ActionProposal"], dict[str, Any]]


@dataclass(frozen=True)
class ActionProposal:
    action_type: str
    target_system: str = "internal"
    entity_type: str = "unknown"
    entity_id: Optional[str] = None
    title: str = ""
    summary: str = ""
    rationale: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    idempotency_key: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ActionProposal":
        payload = data.get("payload")
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        return cls(
            action_type=_safe_text(data.get("action_type")),
            target_system=_safe_text(data.get("target_system")) or "internal",
            entity_type=_safe_text(data.get("entity_type")) or "unknown",
            entity_id=_safe_text(data.get("entity_id")) or None,
            title=_safe_text(data.get("title")),
            summary=_safe_text(data.get("summary")),
            rationale=_safe_text(data.get("rationale")),
            payload=payload,
            confidence=_safe_float_or_none(data.get("confidence")),
            idempotency_key=_safe_text(data.get("idempotency_key")) or None,
        )

    def validate(self) -> None:
        if not self.action_type:
            raise ValueError("action_type is required")


@dataclass(frozen=True)
class ActionExecutionResult:
    action: AgentAction
    created: bool
    duplicate: bool = False


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float_or_none(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _utc_now_iso() -> str:
    return utc_now().isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {
            str(k): _jsonable(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def build_idempotency_key(proposal: ActionProposal) -> str:
    if proposal.idempotency_key:
        return proposal.idempotency_key
    seed = {
        "action_type": proposal.action_type,
        "target_system": proposal.target_system,
        "entity_type": proposal.entity_type,
        "entity_id": proposal.entity_id,
        "payload": _jsonable(proposal.payload),
    }
    raw = json.dumps(seed, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"agent:{proposal.action_type}:{digest[:48]}"


def default_action_policies() -> list[dict[str, Any]]:
    return [
        {
            "action_type": "update_contact_ai_context",
            "autonomy_level": "L1",
            "description": "Обновить безопасные AI-поля контакта: история, приоритет, следующий шаг, сводка.",
        },
        {
            "action_type": "update_lead_ai_context",
            "autonomy_level": "L1",
            "description": "Обновить безопасные AI-поля сделки без смены статуса и финансовых условий.",
        },
        {
            "action_type": "create_amo_follow_up_task",
            "autonomy_level": "L2",
            "requires_notification": True,
            "description": "Поставить менеджеру задачу follow-up в amoCRM по понятному следующему шагу.",
        },
        {
            "action_type": "notify_rop_hot_lead",
            "autonomy_level": "L2",
            "requires_notification": True,
            "description": "Уведомить РОПа о горячем клиенте или просроченном важном касании.",
        },
        {
            "action_type": "send_daily_digest",
            "autonomy_level": "L2",
            "requires_notification": True,
            "description": "Отправить управленческий дайджест в служебный канал.",
        },
        {
            "action_type": "draft_client_message",
            "autonomy_level": "L3",
            "requires_approval": True,
            "description": "Подготовить черновик сообщения клиенту; отправка только после подтверждения человека.",
        },
        {
            "action_type": "recommend_lead_reopen",
            "autonomy_level": "L3",
            "requires_approval": True,
            "description": "Предложить вернуть сделку в работу; фактическая смена стадии требует подтверждения.",
        },
        {
            "action_type": "change_lead_status",
            "autonomy_level": "L3",
            "requires_approval": True,
            "description": "Изменить статус сделки только через очередь подтверждения.",
        },
        {
            "action_type": "direct_client_message",
            "autonomy_level": "L4",
            "description": "Прямое сообщение клиенту без подтверждения запрещено.",
        },
        {
            "action_type": "close_lead",
            "autonomy_level": "L4",
            "description": "Закрытие сделки агентом запрещено.",
        },
        {
            "action_type": "delete_or_merge_contact",
            "autonomy_level": "L4",
            "description": "Удаление или объединение контактов агентом запрещено.",
        },
        {
            "action_type": "update_tallanto_financials",
            "autonomy_level": "L4",
            "description": "Изменение финансовых/учебных данных Tallanto агентом запрещено.",
        },
    ]


def ensure_default_action_policies(session: Session) -> dict[str, Any]:
    created = 0
    present = 0
    for seed in default_action_policies():
        action_type = seed["action_type"]
        policy = session.scalars(
            select(AgentActionPolicy).where(AgentActionPolicy.action_type == action_type)
        ).first()
        if policy is None:
            policy = AgentActionPolicy(
                action_type=action_type,
                autonomy_level=seed.get("autonomy_level", "L3"),
                enabled=bool(seed.get("enabled", True)),
                dry_run_only=bool(seed.get("dry_run_only", False)),
                requires_notification=bool(seed.get("requires_notification", False)),
                requires_approval=bool(seed.get("requires_approval", False)),
                description=_safe_text(seed.get("description")),
                conditions=dict(seed.get("conditions") or {}),
            )
            session.add(policy)
            created += 1
        else:
            present += 1
            # Existing policies may be intentionally customized; only backfill empty descriptions.
            if not _safe_text(policy.description):
                policy.description = _safe_text(seed.get("description"))
    session.flush()
    return {
        "created": created,
        "present": present,
        "total_defaults": len(default_action_policies()),
    }


def resolve_action_policy(session: Session, action_type: str) -> AgentActionPolicy:
    policy = session.scalars(
        select(AgentActionPolicy).where(AgentActionPolicy.action_type == action_type)
    ).first()
    if policy is not None:
        return policy
    policy = AgentActionPolicy(
        action_type=action_type,
        autonomy_level="L3",
        enabled=True,
        requires_approval=True,
        description="Безопасная политика по умолчанию для неизвестного типа действия: только через подтверждение.",
    )
    session.add(policy)
    session.flush()
    return policy


def create_agent_run(
    session: Session,
    *,
    run_type: str,
    trigger: str = "manual",
    mode: str = SAFE_DRY_RUN_MODE,
    actor: Optional[str] = None,
    source: Optional[str] = None,
    summary: Optional[dict[str, Any]] = None,
) -> AgentRun:
    run = AgentRun(
        run_type=_safe_text(run_type) or "manual",
        trigger=_safe_text(trigger) or "manual",
        mode=_safe_text(mode) or SAFE_DRY_RUN_MODE,
        actor=_safe_text(actor) or None,
        source=_safe_text(source) or None,
        status="running",
        summary=dict(summary or {}),
        metrics={},
    )
    session.add(run)
    session.flush()
    return run


def finish_agent_run(
    session: Session,
    run: AgentRun,
    *,
    status: str = "completed",
    error: Optional[str] = None,
    extra_metrics: Optional[dict[str, Any]] = None,
) -> AgentRun:
    metrics = summarize_run(session, run.id)
    if extra_metrics:
        metrics.update(extra_metrics)
    run.status = status
    run.error = _safe_text(error) or None
    run.metrics = metrics
    run.finished_at = utc_now()
    session.flush()
    return run


def _policy_level(policy: AgentActionPolicy) -> str:
    level = _safe_text(policy.autonomy_level).upper()
    return level if level in ACTION_LEVELS else "L3"


def _decision_for_action(
    *,
    proposal: ActionProposal,
    policy: AgentActionPolicy,
    mode: str,
    handlers: Optional[dict[str, ActionHandler]],
) -> tuple[str, list[str], dict[str, Any]]:
    level = _policy_level(policy)
    normalized_mode = _safe_text(mode) or SAFE_DRY_RUN_MODE
    blockers: list[str] = []
    preview = {
        "mode": normalized_mode,
        "autonomy_level": level,
        "policy_enabled": bool(policy.enabled),
        "dry_run_only": bool(policy.dry_run_only),
        "requires_approval": bool(policy.requires_approval or level == "L3"),
        "requires_notification": bool(policy.requires_notification),
        "decision_at": _utc_now_iso(),
        "will_call_external_system": False,
    }

    if not policy.enabled:
        blockers.append("policy_disabled")
        return "blocked_by_policy", blockers, preview
    if level == "L4":
        blockers.append("level_l4_forbidden")
        return "blocked_by_policy", blockers, preview
    if policy.dry_run_only and normalized_mode != SAFE_DRY_RUN_MODE:
        blockers.append("policy_dry_run_only")
        return "blocked_by_policy", blockers, preview
    if normalized_mode == SAFE_DRY_RUN_MODE:
        return "dry_run", blockers, preview
    if level == "L3" or policy.requires_approval:
        return "queued_for_approval", blockers, preview
    if not handlers or proposal.action_type not in handlers:
        blockers.append("live_handler_not_registered")
        return "blocked_no_handler", blockers, preview
    preview["will_call_external_system"] = True
    return "ready_for_execution", blockers, preview


def execute_action(
    session: Session,
    *,
    run: AgentRun,
    proposal: ActionProposal,
    mode: str = SAFE_DRY_RUN_MODE,
    handlers: Optional[dict[str, ActionHandler]] = None,
) -> ActionExecutionResult:
    proposal.validate()
    policy = resolve_action_policy(session, proposal.action_type)
    idempotency_key = build_idempotency_key(proposal)
    existing = session.scalars(
        select(AgentAction).where(AgentAction.idempotency_key == idempotency_key)
    ).first()
    if existing is not None:
        existing.seen_count = int(existing.seen_count or 0) + 1
        existing.last_seen_at = utc_now()
        result = dict(existing.result or {})
        result["last_duplicate_run_id"] = run.id
        result["last_duplicate_seen_at"] = _utc_now_iso()
        existing.result = result
        session.flush()
        return ActionExecutionResult(action=existing, created=False, duplicate=True)

    status, blockers, preview = _decision_for_action(
        proposal=proposal,
        policy=policy,
        mode=mode,
        handlers=handlers,
    )
    action = AgentAction(
        run_id=run.id,
        idempotency_key=idempotency_key,
        action_type=proposal.action_type,
        autonomy_level=_policy_level(policy),
        mode=_safe_text(mode) or SAFE_DRY_RUN_MODE,
        status=status,
        target_system=proposal.target_system,
        entity_type=proposal.entity_type,
        entity_id=proposal.entity_id,
        title=proposal.title,
        summary=proposal.summary,
        rationale=proposal.rationale,
        payload=_jsonable(proposal.payload),
        preview_payload=preview,
        blockers=blockers,
        confidence=proposal.confidence,
        requires_approval=bool(policy.requires_approval or _policy_level(policy) == "L3"),
        requires_notification=bool(policy.requires_notification),
    )
    session.add(action)
    session.flush()

    if status == "ready_for_execution" and handlers and proposal.action_type in handlers:
        try:
            action.result = _jsonable(handlers[proposal.action_type](proposal))
            action.status = "executed"
            action.executed_at = utc_now()
        except Exception as exc:  # pragma: no cover - concrete live handlers are added later.
            action.status = "failed"
            action.blockers = [*blockers, "handler_error"]
            action.result = {"error": str(exc)}
    session.flush()
    return ActionExecutionResult(action=action, created=True, duplicate=False)


def run_action_preview(
    session: Session,
    *,
    proposals: list[ActionProposal],
    run_type: str,
    trigger: str = "manual",
    actor: Optional[str] = None,
    source: Optional[str] = None,
    mode: str = SAFE_DRY_RUN_MODE,
) -> dict[str, Any]:
    ensure_default_action_policies(session)
    run = create_agent_run(
        session,
        run_type=run_type,
        trigger=trigger,
        mode=mode,
        actor=actor,
        source=source,
        summary={"proposal_count": len(proposals)},
    )
    created = 0
    duplicates = 0
    actions: list[AgentAction] = []
    try:
        for proposal in proposals:
            result = execute_action(session, run=run, proposal=proposal, mode=mode)
            actions.append(result.action)
            if result.created:
                created += 1
            if result.duplicate:
                duplicates += 1
        finish_agent_run(
            session,
            run,
            extra_metrics={
                "proposal_count": len(proposals),
                "created_actions": created,
                "duplicate_actions": duplicates,
            },
        )
    except Exception as exc:
        finish_agent_run(session, run, status="failed", error=str(exc))
        raise
    return {
        "run": serialize_run(run),
        "actions": [serialize_action(action) for action in actions],
        "digest": render_run_digest(session, run.id),
    }


def summarize_run(session: Session, run_id: str) -> dict[str, Any]:
    actions = list(session.scalars(select(AgentAction).where(AgentAction.run_id == run_id)).all())
    by_status = Counter(action.status for action in actions)
    by_level = Counter(action.autonomy_level for action in actions)
    by_type = Counter(action.action_type for action in actions)
    blockers = Counter(blocker for action in actions for blocker in (action.blockers or []))
    return {
        "actions_total": len(actions),
        "by_status": dict(sorted(by_status.items())),
        "by_autonomy_level": dict(sorted(by_level.items())),
        "by_action_type": dict(sorted(by_type.items())),
        "blockers": dict(sorted(blockers.items())),
    }


def serialize_policy(policy: AgentActionPolicy) -> dict[str, Any]:
    return {
        "id": policy.id,
        "action_type": policy.action_type,
        "autonomy_level": policy.autonomy_level,
        "enabled": policy.enabled,
        "dry_run_only": policy.dry_run_only,
        "requires_notification": policy.requires_notification,
        "requires_approval": policy.requires_approval,
        "description": policy.description,
        "conditions": policy.conditions or {},
        "created_at": policy.created_at.isoformat() if policy.created_at else None,
        "updated_at": policy.updated_at.isoformat() if policy.updated_at else None,
    }


def serialize_run(run: AgentRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "run_type": run.run_type,
        "trigger": run.trigger,
        "mode": run.mode,
        "status": run.status,
        "actor": run.actor,
        "source": run.source,
        "summary": run.summary or {},
        "metrics": run.metrics or {},
        "error": run.error,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def serialize_action(action: AgentAction) -> dict[str, Any]:
    return {
        "id": action.id,
        "run_id": action.run_id,
        "idempotency_key": action.idempotency_key,
        "action_type": action.action_type,
        "autonomy_level": action.autonomy_level,
        "mode": action.mode,
        "status": action.status,
        "target_system": action.target_system,
        "entity_type": action.entity_type,
        "entity_id": action.entity_id,
        "title": action.title,
        "summary": action.summary,
        "rationale": action.rationale,
        "payload": action.payload or {},
        "preview_payload": action.preview_payload or {},
        "result": action.result or {},
        "blockers": action.blockers or [],
        "confidence": action.confidence,
        "requires_approval": action.requires_approval,
        "requires_notification": action.requires_notification,
        "seen_count": action.seen_count,
        "last_seen_at": action.last_seen_at.isoformat() if action.last_seen_at else None,
        "created_at": action.created_at.isoformat() if action.created_at else None,
        "executed_at": action.executed_at.isoformat() if action.executed_at else None,
    }


def render_run_digest(session: Session, run_id: str, *, max_items: int = 12) -> str:
    run = session.get(AgentRun, run_id)
    if run is None:
        return "Агентский запуск не найден."
    actions = list(
        session.scalars(
            select(AgentAction).where(AgentAction.run_id == run_id).order_by(AgentAction.created_at.asc())
        ).all()
    )
    metrics = summarize_run(session, run_id)
    lines = [
        f"Агентский запуск: {run.run_type}",
        f"Режим: {run.mode}; статус: {run.status}; действий: {metrics['actions_total']}",
    ]
    if metrics["by_autonomy_level"]:
        levels = ", ".join(f"{key}: {value}" for key, value in metrics["by_autonomy_level"].items())
        lines.append(f"Уровни автономии: {levels}")
    if metrics["by_status"]:
        statuses = ", ".join(f"{key}: {value}" for key, value in metrics["by_status"].items())
        lines.append(f"Статусы: {statuses}")
    if metrics["blockers"]:
        blockers = ", ".join(f"{key}: {value}" for key, value in metrics["blockers"].items())
        lines.append(f"Блокеры: {blockers}")
    if actions:
        lines.append("Ключевые действия:")
        for action in actions[:max_items]:
            title = action.title or action.summary or action.action_type
            lines.append(
                f"- [{action.autonomy_level}/{action.status}] {title} "
                f"({action.entity_type}:{action.entity_id or 'n/a'})"
            )
    return "\n".join(lines)


def _parse_date(value: Any) -> Optional[date]:
    text = _safe_text(value)
    if not text:
        return None
    normalized = text.replace("T", " ")
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(normalized[:19], fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _is_warm_or_hot(temperature: str) -> bool:
    normalized = temperature.casefold()
    return any(marker in normalized for marker in ("hot", "warm", "горяч", "тепл", "тёпл", "высок", "сред"))


def _is_hot(temperature: str) -> bool:
    normalized = temperature.casefold()
    return any(marker in normalized for marker in ("hot", "горяч", "высок"))


def _compact(text: str, limit: int = 700) -> str:
    normalized = " ".join(_safe_text(text).split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."


def _context_payload(ctx: PhoneContext) -> dict[str, Any]:
    latest_summary = ""
    if ctx.contact_row:
        latest_summary = _safe_text(ctx.contact_row.get("Краткое резюме последнего свежего звонка"))
    if not latest_summary and ctx.call_rows:
        latest_summary = _safe_text(ctx.call_rows[0].get("Краткое резюме разговора"))
    return {
        "phone": ctx.phone,
        "source_dir": ctx.source_dir,
        "call_count": len(ctx.call_rows),
        "first_call_at": ctx.first_call_at,
        "last_call_at": ctx.last_call_at,
        "manager_history": ctx.manager_history,
        "tallanto_id": ctx.tallanto_id,
        "tallanto_match_status": ctx.tallanto_match_status,
        "fields": {
            "Авто история общения": ctx.history_summary,
            "AI-приоритет": ctx.current_sales_temperature,
            "AI-рекомендованный следующий шаг": ctx.recommended_next_step,
            "Последняя AI-сводка": latest_summary or ctx.history_summary,
        },
    }


def build_morning_scan_proposals(
    *,
    phones: Optional[list[str]] = None,
    limit: int = 100,
    today: Optional[date] = None,
) -> list[ActionProposal]:
    today = today or datetime.now(timezone.utc).date()
    action_limit = max(1, int(limit))
    phone_candidates = phones if phones is not None else get_all_known_phones()
    proposals: list[ActionProposal] = []

    for raw_phone in phone_candidates:
        if len(proposals) >= action_limit:
            break
        ctx = get_phone_context(raw_phone)
        if ctx is None:
            continue
        temperature = _safe_text(ctx.current_sales_temperature)
        next_step = _safe_text(ctx.recommended_next_step)
        history = _safe_text(ctx.history_summary)
        due_date = _parse_date(ctx.follow_up_due_at)
        due_soon = bool(due_date and due_date <= today + timedelta(days=2))
        warm_or_hot = _is_warm_or_hot(temperature)
        hot = _is_hot(temperature)
        actionable = warm_or_hot or due_soon or bool(next_step)
        if not actionable:
            continue

        payload = _context_payload(ctx)
        proposals.append(
            ActionProposal(
                action_type="update_contact_ai_context",
                target_system="amocrm",
                entity_type="contact_phone",
                entity_id=ctx.phone,
                title=f"Обновить AI-контекст контакта {ctx.phone}",
                summary=_compact(history or next_step or "Есть обновленный контактный контекст."),
                rationale=(
                    "Контакт попал в утренний scan из-за теплого/горячего приоритета, "
                    "даты касания или явного следующего шага."
                ),
                payload=payload,
                confidence=0.82 if warm_or_hot and next_step else 0.68,
            )
        )
        if len(proposals) >= action_limit:
            break

        if next_step and (due_soon or warm_or_hot):
            task_text = _compact(
                f"Связаться с клиентом {ctx.phone}. Следующий шаг: {next_step}. "
                f"Контекст: {history}",
                limit=900,
            )
            proposals.append(
                ActionProposal(
                    action_type="create_amo_follow_up_task",
                    target_system="amocrm",
                    entity_type="contact_phone",
                    entity_id=ctx.phone,
                    title=f"Поставить follow-up задачу по {ctx.phone}",
                    summary=task_text,
                    rationale="Есть понятный следующий шаг и контакт теплый/горячий либо дата касания уже близко.",
                    payload={
                        "phone": ctx.phone,
                        "due_at": due_date.isoformat() if due_date else today.isoformat(),
                        "text": task_text,
                        "source_dir": ctx.source_dir,
                    },
                    confidence=0.86 if due_soon else 0.76,
                )
            )
        if len(proposals) >= action_limit:
            break

        if hot:
            proposals.append(
                ActionProposal(
                    action_type="notify_rop_hot_lead",
                    target_system="telegram",
                    entity_type="contact_phone",
                    entity_id=ctx.phone,
                    title=f"Сообщить РОПу о горячем клиенте {ctx.phone}",
                    summary=_compact(next_step or history or "Горячий клиент требует контроля."),
                    rationale="Высокий/горячий приоритет должен попасть в управленческий контроль.",
                    payload={
                        "phone": ctx.phone,
                        "priority": temperature,
                        "next_step": next_step,
                        "last_call_at": ctx.last_call_at,
                        "history_summary": history,
                    },
                    confidence=0.84,
                )
            )

    return proposals[:action_limit]
