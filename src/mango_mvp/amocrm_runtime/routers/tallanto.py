from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, Depends, HTTPException, Query

from mango_mvp.amocrm_runtime.auth import require_api_key
from mango_mvp.amocrm_runtime.config import get_settings
from mango_mvp.amocrm_runtime.tallanto_api import TallantoApiClient, TallantoApiError, build_tallanto_api_config
from mango_mvp.amocrm_runtime.tallanto_export import (
    DEFAULT_DISCOVERY_MODULES,
    DEFAULT_ENUM_OPTIONS,
    export_tallanto_schema_bundle,
)


router = APIRouter(prefix="/integrations/tallanto", tags=["Tallanto integration"])
settings = get_settings()


def _client() -> TallantoApiClient:
    return TallantoApiClient(build_tallanto_api_config())


@router.get(
    "/health",
    dependencies=[Depends(require_api_key)],
)
def tallanto_health() -> dict:
    try:
        return {
            "status": "ok",
            "summary": "Tallanto API доступен.",
            "payload": _client().healthcheck(),
        }
    except TallantoApiError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get(
    "/modules",
    dependencies=[Depends(require_api_key)],
)
def tallanto_modules() -> dict:
    try:
        payload = _client().list_possible_modules()
        return {
            "status": "ok",
            "modules": payload,
        }
    except TallantoApiError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get(
    "/fields",
    dependencies=[Depends(require_api_key)],
)
def tallanto_fields(
    module: str = Query(..., min_length=1),
) -> dict:
    try:
        payload = _client().list_possible_fields(module)
        return {
            "status": "ok",
            "module": module,
            "fields": payload,
        }
    except TallantoApiError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post(
    "/schema/export",
    dependencies=[Depends(require_api_key)],
)
def tallanto_schema_export(
    payload: dict = Body(default_factory=dict),
) -> dict:
    try:
        output_dir = str(payload.get("output_dir") or "").strip()
        destination = (
            Path(output_dir).expanduser()
            if output_dir
            else (Path(settings.source_workspace_root) / "stable_runtime" / "tallanto_runtime")
        )
        modules = payload.get("modules") or list(DEFAULT_DISCOVERY_MODULES)
        enum_options = payload.get("enum_options") or list(DEFAULT_ENUM_OPTIONS)
        written = export_tallanto_schema_bundle(
            _client(),
            output_dir=destination,
            modules=modules,
            enum_options=enum_options,
        )
        return {
            "status": "ok",
            "summary": "Схема Tallanto выгружена.",
            "files": written,
        }
    except TallantoApiError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get(
    "/contact/by-phone",
    dependencies=[Depends(require_api_key)],
)
def tallanto_contact_by_phone(
    phone: str = Query(..., min_length=5),
    limit: int = Query(10, ge=1, le=50),
) -> dict:
    try:
        contacts = _client().search_contacts_by_phone(
            phone,
            max_records=limit,
        )
        return {
            "status": "ok",
            "phone": phone,
            "count": len(contacts),
            "contacts": contacts,
        }
    except TallantoApiError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get(
    "/opportunities/by-contact",
    dependencies=[Depends(require_api_key)],
)
def tallanto_opportunities_by_contact(
    contact_id: str = Query(..., min_length=1),
    limit: int = Query(100, ge=1, le=500),
) -> dict:
    try:
        opportunities = _client().opportunities_by_contact(contact_id, max_records=limit)
        return {
            "status": "ok",
            "contact_id": contact_id,
            "count": len(opportunities),
            "opportunities": opportunities,
        }
    except TallantoApiError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get(
    "/context/by-phone",
    dependencies=[Depends(require_api_key)],
)
def tallanto_context_by_phone(
    phone: str = Query(..., min_length=5),
    max_contacts: int = Query(10, ge=1, le=50),
    max_related_records: int = Query(100, ge=1, le=500),
) -> dict:
    try:
        payload = _client().build_contact_context(
            phone,
            max_contacts=max_contacts,
            max_related_records=max_related_records,
        )
        return {
            "status": "ok",
            **payload,
        }
    except TallantoApiError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get(
    "/context/by-contact-id",
    dependencies=[Depends(require_api_key)],
)
def tallanto_context_by_contact_id(
    contact_id: str = Query(..., min_length=1),
    max_related_records: int = Query(100, ge=1, le=500),
) -> dict:
    try:
        payload = _client().build_contact_context_by_contact_id(
            contact_id,
            max_related_records=max_related_records,
        )
        return {
            "status": "ok",
            **payload,
        }
    except TallantoApiError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
