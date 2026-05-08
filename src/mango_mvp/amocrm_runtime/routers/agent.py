from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from mango_mvp.amocrm_runtime.agent_runtime import (
    ActionProposal,
    build_morning_scan_proposals,
    ensure_default_action_policies,
    render_run_digest,
    run_action_preview,
    serialize_action,
    serialize_policy,
    serialize_run,
    summarize_run,
)
from mango_mvp.amocrm_runtime.auth import AuthContext, require_api_key
from mango_mvp.amocrm_runtime.db import get_db
from mango_mvp.amocrm_runtime.agent_models import AgentAction, AgentActionPolicy, AgentRun

router = APIRouter(prefix="/agent", tags=["AI agent runtime"])


@router.post("/policies/bootstrap")
def agent_policies_bootstrap(
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_api_key),
) -> dict[str, Any]:
    try:
        summary = ensure_default_action_policies(db)
        db.commit()
        return {
            "status": "ok",
            "actor": auth.actor,
            "summary": summary,
        }
    except Exception:
        db.rollback()
        raise


@router.get("/policies")
def agent_policies_list(
    db: Session = Depends(get_db),
    _: AuthContext = Depends(require_api_key),
) -> dict[str, Any]:
    policies = list(
        db.scalars(select(AgentActionPolicy).order_by(AgentActionPolicy.action_type.asc())).all()
    )
    return {
        "status": "ok",
        "count": len(policies),
        "policies": [serialize_policy(policy) for policy in policies],
    }


@router.post("/actions/preview")
def agent_actions_preview(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_api_key),
) -> dict[str, Any]:
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list):
        raise HTTPException(status_code=400, detail="actions must be a list")
    if any(not isinstance(item, dict) for item in raw_actions):
        raise HTTPException(status_code=400, detail="each action must be an object")
    try:
        proposals = [ActionProposal.from_mapping(item) for item in raw_actions]
        result = run_action_preview(
            db,
            proposals=proposals,
            run_type=str(payload.get("run_type") or "manual_action_preview"),
            trigger="api",
            actor=auth.actor,
            source=str(payload.get("source") or "api:/agent/actions/preview"),
            mode="dry_run",
        )
        db.commit()
        return {"status": "ok", **result}
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        db.rollback()
        raise


@router.post("/morning-scan/preview")
def agent_morning_scan_preview(
    payload: dict[str, Any] = Body(default_factory=dict),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_api_key),
) -> dict[str, Any]:
    phones = payload.get("phones")
    if phones is not None and not isinstance(phones, list):
        raise HTTPException(status_code=400, detail="phones must be a list when provided")
    limit = int(payload.get("limit") or 100)
    limit = max(1, min(limit, 500))
    today_raw = payload.get("today")
    today = None
    if today_raw:
        try:
            today = datetime.fromisoformat(str(today_raw)).date()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="today must be ISO date") from exc
    try:
        proposals = build_morning_scan_proposals(
            phones=[str(item) for item in phones] if phones is not None else None,
            limit=limit,
            today=today,
        )
        result = run_action_preview(
            db,
            proposals=proposals,
            run_type="morning_sales_scan_preview",
            trigger="api",
            actor=auth.actor,
            source="canonical_export",
            mode="dry_run",
        )
        db.commit()
        return {
            "status": "ok",
            "proposal_count": len(proposals),
            **result,
        }
    except FileNotFoundError as exc:
        db.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        db.rollback()
        raise


@router.get("/runs/{run_id}")
def agent_run_read(
    run_id: str,
    include_actions: bool = Query(default=True),
    db: Session = Depends(get_db),
    _: AuthContext = Depends(require_api_key),
) -> dict[str, Any]:
    run: Optional[AgentRun] = db.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="agent run not found")
    actions = []
    if include_actions:
        action_rows = list(
            db.scalars(
                select(AgentAction)
                .where(AgentAction.run_id == run_id)
                .order_by(AgentAction.created_at.asc())
            ).all()
        )
        actions = [serialize_action(action) for action in action_rows]
    return {
        "status": "ok",
        "run": serialize_run(run),
        "metrics": summarize_run(db, run_id),
        "digest": render_run_digest(db, run_id),
        "actions": actions,
    }
