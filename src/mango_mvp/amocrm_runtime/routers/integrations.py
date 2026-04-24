from __future__ import annotations

import json
from urllib import parse as url_parse

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from mango_mvp.amocrm_runtime.amo_integration import (
    AmoIntegrationError,
    exchange_callback_code,
    fetch_contact_field_catalog,
    get_active_connection,
    get_amo_connection_status,
    record_external_secrets,
    refresh_connection_tokens,
    resolve_amo_access_context,
)
from mango_mvp.amocrm_runtime.auth import require_api_key
from mango_mvp.amocrm_runtime.db import get_db
from mango_mvp.amocrm_runtime.schemas import (
    AmoContactFieldSyncResponse,
    AmoIntegrationRefreshResponse,
    AmoIntegrationSecretsWebhookResponse,
    AmoIntegrationStatusRead,
)


router = APIRouter(prefix="/integrations/amocrm", tags=["amoCRM integration"])


async def _parse_request_payload(request: Request) -> dict:
    payload: dict = {}
    for key, value in request.query_params.multi_items():
        payload[key] = value

    raw_body = await request.body()
    if not raw_body:
        return payload

    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        try:
            decoded = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            decoded = {}
        if isinstance(decoded, dict):
            payload.update(decoded)
            nested = decoded.get("data")
            if isinstance(nested, dict):
                payload.update(nested)
        return payload

    decoded_body = raw_body.decode("utf-8", errors="replace")
    form_payload = url_parse.parse_qs(decoded_body, keep_blank_values=True)
    for key, values in form_payload.items():
        payload[key] = values[-1] if values else ""
    return payload


def _callback_html(summary: str, *, success: bool, account_base_url: str | None = None) -> str:
    title = "amoCRM подключена" if success else "amoCRM авторизация не завершена"
    accent = "#0b7a4b" if success else "#9f3020"
    account_line = (
        f"<p><strong>Аккаунт:</strong> {account_base_url}</p>" if account_base_url else ""
    )
    return f"""<!doctype html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
    <style>
      body {{
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background: #f5f3ee;
        color: #1e2428;
      }}
      main {{
        max-width: 720px;
        margin: 64px auto;
        padding: 32px;
        border-radius: 24px;
        background: #fffdf7;
        box-shadow: 0 24px 80px rgba(31, 39, 43, 0.12);
      }}
      h1 {{
        margin: 0 0 12px;
        font-size: 28px;
      }}
      .accent {{
        display: inline-block;
        padding: 8px 12px;
        border-radius: 999px;
        background: {accent};
        color: white;
        font-weight: 600;
        margin-bottom: 20px;
      }}
      p {{
        line-height: 1.55;
      }}
      code {{
        background: #f0ebe0;
        padding: 2px 6px;
        border-radius: 8px;
      }}
    </style>
  </head>
  <body>
    <main>
      <div class="accent">{title}</div>
      <h1>{title}</h1>
      <p>{summary}</p>
      {account_line}
      <p>Если вы запускали подключение из AI Office, окно можно закрыть и вернуться в интерфейс.</p>
      <p>Дальше проверьте статус на маршруте <code>/api/integrations/amocrm/status</code>.</p>
    </main>
  </body>
</html>"""


@router.api_route("/secrets", methods=["GET", "POST"], response_model=AmoIntegrationSecretsWebhookResponse)
async def amo_external_secrets(
    request: Request,
    db: Session = Depends(get_db),
) -> AmoIntegrationSecretsWebhookResponse:
    try:
        payload = await _parse_request_payload(request)
        connection, summary = record_external_secrets(db, payload=payload)
        db.commit()
        return AmoIntegrationSecretsWebhookResponse(
            status=connection.status,
            summary=summary,
        )
    except AmoIntegrationError as exc:
        db.rollback()
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.get("/callback", response_class=HTMLResponse)
async def amo_external_callback(
    request: Request,
    db: Session = Depends(get_db),
) -> HTMLResponse:
    code = str(request.query_params.get("code") or "").strip()
    state = str(request.query_params.get("state") or "").strip() or None
    referer = str(request.query_params.get("referer") or "").strip() or None
    if not code:
        return HTMLResponse(
            _callback_html("В callback не пришёл параметр code.", success=False),
            status_code=400,
        )

    try:
        connection, summary = exchange_callback_code(
            db,
            code=code,
            state=state,
            referer=referer,
        )
        db.commit()
        return HTMLResponse(
            _callback_html(summary, success=True, account_base_url=connection.account_base_url),
            status_code=200,
        )
    except AmoIntegrationError as exc:
        db.commit()
        return HTMLResponse(
            _callback_html(str(exc), success=False, account_base_url=referer),
            status_code=exc.status_code,
        )


@router.get(
    "/status",
    response_model=AmoIntegrationStatusRead,
    dependencies=[Depends(require_api_key)],
)
def amo_integration_status(
    db: Session = Depends(get_db),
) -> AmoIntegrationStatusRead:
    return AmoIntegrationStatusRead.model_validate(get_amo_connection_status(db))


@router.post(
    "/refresh",
    response_model=AmoIntegrationRefreshResponse,
    dependencies=[Depends(require_api_key)],
)
def amo_integration_refresh(
    db: Session = Depends(get_db),
) -> AmoIntegrationRefreshResponse:
    try:
        context = resolve_amo_access_context(db)
        connection = get_active_connection(db)
        if connection is None and context.token_source == "env":
            return AmoIntegrationRefreshResponse(
                status="direct_token",
                summary="Runtime использует direct token fallback; refresh для этого режима не требуется.",
                expires_at=None,
            )
        if connection is None:
            raise AmoIntegrationError("AMO integration is not connected yet.", status_code=409)
        connection = refresh_connection_tokens(db, connection)
        db.commit()
        return AmoIntegrationRefreshResponse(
            status=connection.status,
            summary="Токен amoCRM обновлён.",
            expires_at=connection.expires_at,
        )
    except AmoIntegrationError as exc:
        db.rollback()
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc


@router.post(
    "/contact-fields/sync",
    response_model=AmoContactFieldSyncResponse,
    dependencies=[Depends(require_api_key)],
)
def amo_contact_fields_sync(
    db: Session = Depends(get_db),
) -> AmoContactFieldSyncResponse:
    try:
        fields = fetch_contact_field_catalog(db, force_refresh=True)
        connection = get_active_connection(db)
        db.commit()
        return AmoContactFieldSyncResponse(
            status="ok",
            summary="Каталог полей контактов amoCRM синхронизирован.",
            field_count=len(fields),
            synced_at=connection.contact_field_catalog_synced_at if connection is not None else None,
        )
    except AmoIntegrationError as exc:
        db.rollback()
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
