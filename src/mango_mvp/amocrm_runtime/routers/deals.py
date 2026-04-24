from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from mango_mvp.amocrm_runtime.amo_integration import AmoIntegrationError, fetch_lead_field_catalog, search_contacts_by_phone
from mango_mvp.amocrm_runtime.auth import require_api_key
from mango_mvp.amocrm_runtime.db import get_db
from mango_mvp.amocrm_runtime.deals import (
    analyze_by_phone,
    build_recent_closed_queue,
    get_latest_queue_snapshot,
    resolve_target_lead,
    write_analysis_to_lead,
)

router = APIRouter(prefix="/integrations/amocrm", tags=["amoCRM deals"])


@router.get(
    "/contacts/by-phone",
    dependencies=[Depends(require_api_key)],
)
def amo_contacts_by_phone(
    phone: str = Query(..., min_length=5),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
) -> dict:
    try:
        contacts = search_contacts_by_phone(db, phone=phone, limit=limit)
        return {
            "status": "ok",
            "phone": phone,
            "count": len(contacts),
            "contacts": contacts,
        }
    except AmoIntegrationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get(
    "/leads/by-phone",
    dependencies=[Depends(require_api_key)],
)
def amo_leads_by_phone(
    phone: str = Query(..., min_length=5),
    call_at: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
) -> dict:
    try:
        return resolve_target_lead(db, phone=phone, call_at=call_at)
    except AmoIntegrationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post(
    "/lead-fields/sync",
    dependencies=[Depends(require_api_key)],
)
def amo_lead_fields_sync(
    db: Session = Depends(get_db),
) -> dict:
    try:
        fields = fetch_lead_field_catalog(db, force_refresh=True)
        return {
            "status": "ok",
            "summary": "Каталог полей сделок amoCRM синхронизирован.",
            "field_count": len(fields),
            "fields": fields,
        }
    except AmoIntegrationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post(
    "/deals/dossier-by-phone",
    dependencies=[Depends(require_api_key)],
)
def amo_deal_dossier_by_phone(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
) -> dict:
    phone = str(payload.get("phone") or "").strip()
    call_at = str(payload.get("call_at") or "").strip() or None
    if not phone:
        raise HTTPException(status_code=400, detail="phone is required")
    try:
        result = analyze_by_phone(db, phone=phone, call_at=call_at)
        return {
            "status": "ok",
            "phone": phone,
            "analysis_mode": result.get("analysis_mode"),
            "selected": result.get("selected"),
            "dossier": result.get("dossier"),
        }
    except AmoIntegrationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post(
    "/deals/analyze-by-phone",
    dependencies=[Depends(require_api_key)],
)
def amo_deal_analyze_by_phone(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
) -> dict:
    phone = str(payload.get("phone") or "").strip()
    call_at = str(payload.get("call_at") or "").strip() or None
    if not phone:
        raise HTTPException(status_code=400, detail="phone is required")
    try:
        return analyze_by_phone(db, phone=phone, call_at=call_at)
    except AmoIntegrationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post(
    "/deals/writeback",
    dependencies=[Depends(require_api_key)],
)
def amo_deal_writeback(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
) -> dict:
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        raise HTTPException(status_code=400, detail="analysis object is required")
    try:
        result = write_analysis_to_lead(db, analysis=analysis)
        db.commit()
        return {
            "status": "ok",
            "summary": "AI-результат записан в custom fields сделки amoCRM.",
            "result": result,
        }
    except (AmoIntegrationError, ValueError) as exc:
        db.rollback()
        status_code = exc.status_code if isinstance(exc, AmoIntegrationError) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.post(
    "/deals/queue/build",
    dependencies=[Depends(require_api_key)],
)
def amo_deal_queue_build(
    payload: dict = Body(default_factory=dict),
    db: Session = Depends(get_db),
) -> dict:
    days_back = payload.get("days_back")
    apply_writeback = bool(payload.get("apply_writeback", False))
    max_leads = payload.get("max_leads")
    try:
        summary = build_recent_closed_queue(
            db,
            days_back=int(days_back) if days_back is not None else None,
            apply_writeback=apply_writeback,
            max_leads=int(max_leads) if max_leads is not None else None,
        )
        db.commit()
        return summary
    except (AmoIntegrationError, ValueError) as exc:
        db.rollback()
        status_code = exc.status_code if isinstance(exc, AmoIntegrationError) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.get(
    "/deals/queue/latest",
    dependencies=[Depends(require_api_key)],
)
def amo_deal_queue_latest() -> dict:
    return get_latest_queue_snapshot()
