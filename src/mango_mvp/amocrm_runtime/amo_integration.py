from __future__ import annotations

import json
import socket
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html import escape as html_escape
from pathlib import Path
from typing import Any, Optional
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request

from sqlalchemy import select
from sqlalchemy.orm import Session

from mango_mvp.amocrm_runtime.config import get_settings
from mango_mvp.amocrm_runtime.models import AmoIntegrationConnection, utc_now
from mango_mvp.utils.phone import normalize_phone


settings = get_settings()
AMO_CONTACT_FIELD_CACHE_TTL = timedelta(hours=12)
DEFAULT_REQUIRED_AMO_CONTACT_FIELDS = (
    "Id Tallanto",
    "Филиал Tallanto",
    "Авто история общения",
    "Статус матчинга",
    "AI-приоритет",
    "AI-рекомендованный следующий шаг",
    "Последняя AI-сводка",
)
DEFAULT_REQUIRED_AMO_LEAD_FIELDS = (
    "AI-вердикт по закрытию",
    "AI-risk: premature close",
    "AI-основание вердикта",
    "AI-рекомендованный следующий шаг",
    "AI-дата следующего касания",
    "AI-сводка по сделке",
)
CONTACT_WRITE_PROTECTED_FIELDS = frozenset(
    {
        "Id Tallanto",
        "Филиал Tallanto",
    }
)


class AmoIntegrationError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class AmoAccessContext:
    account_base_url: str
    access_token: str
    token_source: str
    connection: Optional[AmoIntegrationConnection]


def _normalize_base_url(raw_value: Optional[str]) -> Optional[str]:
    candidate = str(raw_value or "").strip()
    if not candidate:
        return None

    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = url_parse.urlparse(candidate)
    if not parsed.netloc and parsed.path:
        parsed = url_parse.urlparse(f"https://{parsed.path}")
    if not parsed.netloc:
        return None
    scheme = parsed.scheme or "https"
    return f"{scheme}://{parsed.netloc}"


def _account_subdomain(account_base_url: Optional[str]) -> Optional[str]:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        return None
    host = url_parse.urlparse(normalized).netloc
    if not host:
        return None
    return host.split(".", 1)[0]


def _resolve_scopes() -> list[str]:
    scopes = [scope.strip() for scope in settings.crm_amo_oauth_scopes if scope.strip()]
    return scopes or ["crm"]


def _resolve_redirect_uri() -> Optional[str]:
    return str(settings.crm_amo_oauth_redirect_uri or "").strip() or None


def _resolve_secrets_uri() -> Optional[str]:
    return str(settings.crm_amo_oauth_secrets_uri or "").strip() or None


def _resolve_account_base_url_hint() -> Optional[str]:
    direct_hint = _normalize_base_url(settings.crm_amo_oauth_account_base_url)
    if direct_hint:
        return direct_hint
    return _normalize_base_url(settings.crm_amo_base_url)


def _amo_http_request(
    *,
    method: str,
    url: str,
    headers: Optional[dict[str, str]] = None,
    body: Optional[dict[str, Any]] = None,
    timeout_seconds: Optional[int] = None,
) -> dict[str, Any]:
    effective_timeout_seconds = max(1, int(timeout_seconds or settings.crm_amo_http_timeout_seconds))
    payload = None
    request_headers = {
        "Accept": "application/json",
    }
    if headers:
        request_headers.update(headers)
    if body is not None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    request = url_request.Request(
        url,
        data=payload,
        headers=request_headers,
        method=method.upper(),
    )
    attempts = 4
    retry_delay_seconds = 1.5
    for attempt in range(1, attempts + 1):
        try:
            with url_request.urlopen(request, timeout=effective_timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                if not raw.strip():
                    return {}
                decoded = json.loads(raw)
                if isinstance(decoded, dict):
                    return decoded
                return {"data": decoded}
        except url_error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise AmoIntegrationError(
                f"HTTP {exc.code} from amoCRM: {details or exc.reason}",
                status_code=502,
            ) from exc
        except (url_error.URLError, TimeoutError, socket.timeout, ssl.SSLError) as exc:
            if attempt >= attempts:
                reason = getattr(exc, "reason", exc)
                raise AmoIntegrationError(
                    f"Failed to reach amoCRM: {reason}",
                    status_code=502,
                ) from exc
            time.sleep(retry_delay_seconds * attempt)
        except json.JSONDecodeError as exc:
            raise AmoIntegrationError(
                f"Invalid JSON response from amoCRM endpoint {url}.",
                status_code=502,
            ) from exc


def amo_api_request(
    session: Session,
    *,
    method: str,
    path_or_url: str,
    params: Optional[dict[str, Any]] = None,
    body: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    context = resolve_amo_access_context(session)
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        url = path_or_url
    else:
        normalized_base = _normalize_base_url(context.account_base_url)
        url = url_parse.urljoin(f"{normalized_base.rstrip('/')}/", path_or_url.lstrip("/"))
    if params:
        query = url_parse.urlencode(params, doseq=True)
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}{query}"
    return _amo_http_request(
        method=method,
        url=url,
        headers={"Authorization": f"Bearer {context.access_token}"},
        body=body,
    )


def _paged_embedded_items(
    session: Session,
    *,
    initial_url: str,
    embedded_key: str,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    next_url: Optional[str] = initial_url
    while next_url:
        payload = amo_api_request(session, method="GET", path_or_url=next_url)
        embedded = payload.get("_embedded") if isinstance(payload, dict) else {}
        page_items = embedded.get(embedded_key) if isinstance(embedded, dict) else []
        if isinstance(page_items, list):
            items.extend(item for item in page_items if isinstance(item, dict))
        next_link = None
        links = payload.get("_links") if isinstance(payload, dict) else {}
        if isinstance(links, dict):
            next_meta = links.get("next")
            if isinstance(next_meta, dict):
                next_link = next_meta.get("href")
        next_url = next_link
    return items


def _token_endpoint(account_base_url: str) -> str:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)
    return f"{normalized.rstrip('/')}/oauth2/access_token"


def _contacts_custom_fields_endpoint(account_base_url: str) -> str:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)
    return f"{normalized.rstrip('/')}/api/v4/contacts/custom_fields?limit=50"


def _contact_update_endpoint(account_base_url: str, contact_id: int) -> str:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)
    return f"{normalized.rstrip('/')}/api/v4/contacts/{contact_id}"


def _contact_entity_endpoint(account_base_url: str, contact_id: int) -> str:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)
    return f"{normalized.rstrip('/')}/api/v4/contacts/{contact_id}"


def _contacts_search_endpoint(account_base_url: str) -> str:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)
    return f"{normalized.rstrip('/')}/api/v4/contacts"


def _lead_entity_endpoint(account_base_url: str, lead_id: int) -> str:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)
    return f"{normalized.rstrip('/')}/api/v4/leads/{lead_id}"


def _lead_update_endpoint(account_base_url: str, lead_id: int) -> str:
    return _lead_entity_endpoint(account_base_url, lead_id)


def _lead_notes_endpoint(account_base_url: str, lead_id: int) -> str:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)
    return f"{normalized.rstrip('/')}/api/v4/leads/{lead_id}/notes"


def _leads_collection_endpoint(account_base_url: str) -> str:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)
    return f"{normalized.rstrip('/')}/api/v4/leads"


def _lead_custom_fields_endpoint(account_base_url: str) -> str:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)
    return f"{normalized.rstrip('/')}/api/v4/leads/custom_fields?limit=50"


def _tasks_endpoint(account_base_url: str) -> str:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)
    return f"{normalized.rstrip('/')}/api/v4/tasks"


def _pipelines_endpoint(account_base_url: str) -> str:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)
    return f"{normalized.rstrip('/')}/api/v4/leads/pipelines?with=statuses"


def _users_endpoint(account_base_url: str) -> str:
    normalized = _normalize_base_url(account_base_url)
    if not normalized:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)
    return f"{normalized.rstrip('/')}/api/v4/users"


def _pick_first_non_empty(payload: dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            value = value[-1] if value else None
        candidate = str(value or "").strip()
        if candidate:
            return candidate
    return None


def _extract_state(payload: dict[str, Any]) -> Optional[str]:
    return _pick_first_non_empty(payload, "state", "request_state")


def _extract_account_base_url(payload: dict[str, Any], *, fallback: Optional[str] = None) -> Optional[str]:
    direct = _pick_first_non_empty(
        payload,
        "referer",
        "base_url",
        "account_base_url",
        "account_url",
    )
    if direct:
        return _normalize_base_url(direct)
    subdomain = _pick_first_non_empty(payload, "subdomain")
    if subdomain:
        return _normalize_base_url(f"https://{subdomain}.amocrm.ru")
    return _normalize_base_url(fallback)


def _select_connection(
    session: Session,
    *,
    state: Optional[str] = None,
    client_id: Optional[str] = None,
    account_base_url: Optional[str] = None,
) -> Optional[AmoIntegrationConnection]:
    if state:
        connection = session.scalars(
            select(AmoIntegrationConnection)
            .where(AmoIntegrationConnection.state == state)
            .order_by(AmoIntegrationConnection.updated_at.desc())
        ).first()
        if connection is not None:
            return connection

    if client_id:
        connection = session.scalars(
            select(AmoIntegrationConnection)
            .where(AmoIntegrationConnection.client_id == client_id)
            .order_by(AmoIntegrationConnection.updated_at.desc())
        ).first()
        if connection is not None:
            return connection

    if account_base_url:
        normalized = _normalize_base_url(account_base_url)
        connection = session.scalars(
            select(AmoIntegrationConnection)
            .where(AmoIntegrationConnection.account_base_url == normalized)
            .order_by(AmoIntegrationConnection.updated_at.desc())
        ).first()
        if connection is not None:
            return connection
    return None


def _ensure_connection(
    session: Session,
    *,
    state: Optional[str],
    client_id: Optional[str],
    account_base_url: Optional[str],
) -> AmoIntegrationConnection:
    existing = _select_connection(
        session,
        state=state,
        client_id=client_id,
        account_base_url=account_base_url,
    )
    if existing is not None:
        return existing

    connection = AmoIntegrationConnection(
        state=state,
        account_base_url=_normalize_base_url(account_base_url),
        account_subdomain=_account_subdomain(account_base_url),
        redirect_uri=_resolve_redirect_uri(),
        secrets_uri=_resolve_secrets_uri(),
        scopes=_resolve_scopes(),
        status="pending",
    )
    session.add(connection)
    session.flush()
    return connection


def build_external_oauth_setup() -> dict[str, Any]:
    redirect_uri = _resolve_redirect_uri()
    secrets_uri = _resolve_secrets_uri()
    scopes = _resolve_scopes()
    data_scopes = ",".join(scopes)
    button_snippet = None
    if redirect_uri and secrets_uri:
        button_snippet = (
            '<script class="amocrm_oauth" charset="utf-8" '
            f'data-name="{html_escape(settings.crm_amo_oauth_name)}" '
            f'data-description="{html_escape(settings.crm_amo_oauth_description)}" '
            f'data-redirect_uri="{html_escape(redirect_uri)}" '
            f'data-secrets_uri="{html_escape(secrets_uri)}" '
            f'data-logo="{html_escape(settings.crm_amo_oauth_logo_url or "")}" '
            f'data-scopes="{html_escape(data_scopes)}" '
            'data-title="Подключить amoCRM" '
            'data-mode="popup" '
            'src="https://www.amocrm.ru/auth/button.min.js"></script>'
        )
    return {
        "integration_mode": "external",
        "redirect_uri": redirect_uri,
        "secrets_uri": secrets_uri,
        "scopes": scopes,
        "integration_name": settings.crm_amo_oauth_name,
        "integration_description": settings.crm_amo_oauth_description,
        "logo_url": settings.crm_amo_oauth_logo_url,
        "account_base_url_hint": _resolve_account_base_url_hint(),
        "button_snippet": button_snippet,
    }


def record_external_secrets(
    session: Session,
    *,
    payload: dict[str, Any],
) -> tuple[AmoIntegrationConnection, str]:
    client_id = _pick_first_non_empty(payload, "client_id", "integration_id", "id", "client_uuid")
    client_secret = _pick_first_non_empty(
        payload,
        "client_secret",
        "secret_key",
        "integration_secret_key",
        "secret",
    )
    if not client_id or not client_secret:
        raise AmoIntegrationError(
            "Webhook secrets payload must include client_id and client_secret (or secret_key).",
            status_code=400,
        )

    state = _extract_state(payload)
    account_base_url = _extract_account_base_url(payload, fallback=_resolve_account_base_url_hint())
    connection = _ensure_connection(
        session,
        state=state,
        client_id=client_id,
        account_base_url=account_base_url,
    )
    connection.client_id = client_id
    connection.client_secret = client_secret
    connection.state = state or connection.state
    connection.account_base_url = account_base_url or connection.account_base_url
    connection.account_subdomain = _account_subdomain(connection.account_base_url)
    connection.redirect_uri = _resolve_redirect_uri()
    connection.secrets_uri = _resolve_secrets_uri()
    connection.scopes = _resolve_scopes()
    connection.last_secrets_payload = payload
    connection.last_error = None
    connection.status = "awaiting_callback"
    session.flush()
    return (
        connection,
        "Секреты внешней amoCRM интеграции сохранены. Можно завершать авторизацию и принимать callback.",
    )


def _exchange_token(
    *,
    account_base_url: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    grant_type: str,
    code: Optional[str] = None,
    refresh_token: Optional[str] = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": grant_type,
        "redirect_uri": redirect_uri,
    }
    if code:
        body["code"] = code
    if refresh_token:
        body["refresh_token"] = refresh_token
    return _amo_http_request(
        method="POST",
        url=_token_endpoint(account_base_url),
        body=body,
    )


def _apply_token_payload(
    connection: AmoIntegrationConnection,
    *,
    payload: dict[str, Any],
    account_base_url: Optional[str],
    callback_payload: Optional[dict[str, Any]] = None,
) -> None:
    access_token = _pick_first_non_empty(payload, "access_token")
    refresh_token = _pick_first_non_empty(payload, "refresh_token")
    token_type = _pick_first_non_empty(payload, "token_type")
    expires_in_raw = payload.get("expires_in")

    if not access_token or not refresh_token:
        raise AmoIntegrationError("amoCRM token response did not include access_token/refresh_token.", status_code=502)

    expires_at = None
    if expires_in_raw is not None:
        try:
            expires_at = utc_now() + timedelta(seconds=int(expires_in_raw))
        except (TypeError, ValueError):
            expires_at = None

    connection.account_base_url = _normalize_base_url(account_base_url) or connection.account_base_url
    connection.account_subdomain = _account_subdomain(connection.account_base_url)
    connection.access_token = access_token
    connection.refresh_token = refresh_token
    connection.token_type = token_type or "Bearer"
    connection.expires_at = expires_at
    connection.authorized_at = utc_now()
    connection.status = "active"
    connection.last_error = None
    if callback_payload is not None:
        connection.last_callback_payload = callback_payload


def exchange_callback_code(
    session: Session,
    *,
    code: str,
    state: Optional[str],
    referer: Optional[str],
) -> tuple[AmoIntegrationConnection, str]:
    account_base_url = _extract_account_base_url(
        {"referer": referer} if referer else {},
        fallback=_resolve_account_base_url_hint(),
    )
    connection = _ensure_connection(
        session,
        state=state,
        client_id=None,
        account_base_url=account_base_url,
    )
    if not connection.client_id or not connection.client_secret:
        connection.last_callback_payload = {
            "code": code,
            "state": state,
            "referer": referer,
        }
        connection.last_error = "Callback получен раньше secrets webhook: нет client_id/client_secret."
        connection.status = "awaiting_secrets"
        session.flush()
        raise AmoIntegrationError(
            "Callback received before secrets webhook. Repeat authorization after secrets arrive.",
            status_code=409,
        )

    redirect_uri = connection.redirect_uri or _resolve_redirect_uri()
    if not redirect_uri:
        raise AmoIntegrationError("CRM_AMO_OAUTH_REDIRECT_URI is not configured.", status_code=503)
    if not account_base_url:
        raise AmoIntegrationError("Unable to determine amoCRM account base URL from callback.", status_code=400)

    token_payload = _exchange_token(
        account_base_url=account_base_url,
        client_id=connection.client_id,
        client_secret=connection.client_secret,
        grant_type="authorization_code",
        code=code,
        redirect_uri=redirect_uri,
    )
    _apply_token_payload(
        connection,
        payload=token_payload,
        account_base_url=account_base_url,
        callback_payload={
            "code": code,
            "state": state,
            "referer": referer,
        },
    )
    session.flush()
    return (
        connection,
        f"Авторизация amoCRM завершена для {connection.account_base_url}. Токены сохранены.",
    )


def _looks_like_revoked_token_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "token has been revoked" in message or "invalid_grant" in message


def refresh_connection_tokens(
    session: Session,
    connection: AmoIntegrationConnection,
) -> AmoIntegrationConnection:
    locked_connection = session.scalars(
        select(AmoIntegrationConnection)
        .where(AmoIntegrationConnection.id == connection.id)
        .with_for_update()
    ).one()
    if not _token_is_stale(locked_connection):
        return locked_connection

    if not locked_connection.client_id or not locked_connection.client_secret:
        raise AmoIntegrationError("AMO connection does not have client credentials yet.", status_code=409)
    if not locked_connection.refresh_token:
        raise AmoIntegrationError("AMO connection does not have refresh_token.", status_code=409)

    redirect_uri = locked_connection.redirect_uri or _resolve_redirect_uri()
    if not redirect_uri:
        raise AmoIntegrationError("CRM_AMO_OAUTH_REDIRECT_URI is not configured.", status_code=503)
    account_base_url = locked_connection.account_base_url or _resolve_account_base_url_hint()
    if not account_base_url:
        raise AmoIntegrationError("AMO account base URL is not configured.", status_code=503)

    try:
        token_payload = _exchange_token(
            account_base_url=account_base_url,
            client_id=locked_connection.client_id,
            client_secret=locked_connection.client_secret,
            grant_type="refresh_token",
            refresh_token=locked_connection.refresh_token,
            redirect_uri=redirect_uri,
        )
        _apply_token_payload(
            locked_connection,
            payload=token_payload,
            account_base_url=account_base_url,
        )
        session.flush()
        return locked_connection
    except AmoIntegrationError as exc:
        locked_connection.last_error = str(exc)
        locked_connection.status = "reauthorization_required" if _looks_like_revoked_token_error(exc) else "refresh_error"
        session.flush()
        raise


def _token_is_stale(connection: AmoIntegrationConnection) -> bool:
    if not connection.access_token:
        return True
    if connection.expires_at is None:
        return False
    return connection.expires_at <= (utc_now() + timedelta(minutes=2))


def get_active_connection(session: Session) -> Optional[AmoIntegrationConnection]:
    account_hint = _resolve_account_base_url_hint()
    if account_hint:
        connection = _select_connection(session, account_base_url=account_hint)
        if connection is not None:
            return connection
    return session.scalars(
        select(AmoIntegrationConnection).order_by(AmoIntegrationConnection.updated_at.desc())
    ).first()


def resolve_amo_access_context(session: Session) -> AmoAccessContext:
    env_token = str(settings.crm_amo_api_token or "").strip()
    env_base = _normalize_base_url(settings.crm_amo_base_url)
    if env_token and env_base:
        return AmoAccessContext(
            account_base_url=env_base,
            access_token=env_token,
            token_source="env",
            connection=None,
        )

    connection = get_active_connection(session)
    if connection is None:
        raise AmoIntegrationError(
            "AMO integration is not connected yet. Complete external OAuth first.",
            status_code=409,
        )

    if _token_is_stale(connection):
        refresh_connection_tokens(session, connection)

    if not connection.access_token or not connection.account_base_url:
        raise AmoIntegrationError(
            "AMO integration does not have a usable access token yet.",
            status_code=409,
        )

    return AmoAccessContext(
        account_base_url=connection.account_base_url,
        access_token=connection.access_token,
        token_source="oauth",
        connection=connection,
    )


def _follow_next_link(base_url: str, next_href: str) -> str:
    if next_href.startswith("http://") or next_href.startswith("https://"):
        return next_href
    normalized_base = _normalize_base_url(base_url)
    return url_parse.urljoin(f"{normalized_base.rstrip('/')}/", next_href.lstrip("/"))


def _flatten_contact_field_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "code": item.get("code"),
        "type": item.get("type"),
        "group_id": item.get("group_id"),
        "is_predefined": bool(item.get("is_predefined")),
        "is_api_only": bool(item.get("is_api_only")),
    }


def fetch_contact_field_catalog(session: Session, *, force_refresh: bool = False) -> list[dict[str, Any]]:
    context = resolve_amo_access_context(session)
    connection = context.connection
    if (
        not force_refresh
        and connection is not None
        and connection.contact_field_catalog
        and connection.contact_field_catalog_synced_at is not None
        and connection.contact_field_catalog_synced_at >= (utc_now() - AMO_CONTACT_FIELD_CACHE_TTL)
    ):
        return list(connection.contact_field_catalog)

    fields: list[dict[str, Any]] = []
    next_url = _contacts_custom_fields_endpoint(context.account_base_url)
    headers = {"Authorization": f"Bearer {context.access_token}"}

    while next_url:
        payload = _amo_http_request(method="GET", url=next_url, headers=headers)
        embedded = payload.get("_embedded") if isinstance(payload, dict) else {}
        items = embedded.get("custom_fields") if isinstance(embedded, dict) else []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    fields.append(_flatten_contact_field_item(item))

        links = payload.get("_links") if isinstance(payload, dict) else {}
        next_link = None
        if isinstance(links, dict):
            next_meta = links.get("next")
            if isinstance(next_meta, dict):
                next_link = next_meta.get("href")
        next_url = _follow_next_link(context.account_base_url, next_link) if next_link else None

    if connection is not None:
        connection.contact_field_catalog = fields
        connection.contact_field_catalog_synced_at = utc_now()
        connection.last_error = None
        session.flush()
    return fields


def _find_field_meta(field_catalog: list[dict[str, Any]], field_name: str) -> Optional[dict[str, Any]]:
    normalized_target = field_name.strip().casefold()
    for item in field_catalog:
        if str(item.get("name") or "").strip().casefold() == normalized_target:
            return item
    for item in field_catalog:
        if str(item.get("code") or "").strip().casefold() == normalized_target:
            return item
    return None


def _field_values(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        result = []
        for item in value:
            if item is None:
                continue
            result.append({"value": item})
        return result
    return [{"value": value}]


def _normalize_field_value_for_meta(meta: dict[str, Any], value: Any) -> Any:
    field_type = str(meta.get("type") or "").strip().casefold()
    if isinstance(value, list):
        return [_normalize_field_value_for_meta(meta, item) for item in value]
    if isinstance(value, str) and field_type in {"date", "date_time", "birthday"}:
        timestamp = _parse_amo_date_value(value)
        if timestamp is not None:
            return timestamp
    if isinstance(value, str) and field_type == "text" and len(value) > 255:
        suffix = " [сжато]"
        budget = max(20, 255 - len(suffix))
        candidate = value[:budget].rstrip()
        word_boundary = max(candidate.rfind(" "), candidate.rfind(","), candidate.rfind(";"), candidate.rfind("."))
        if word_boundary >= int(budget * 0.55):
            candidate = candidate[:word_boundary].rstrip(" ,;.")
        return f"{candidate}{suffix}"
    return value


def _parse_amo_date_value(value: str) -> int | None:
    text = value.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                parsed = None  # type: ignore[assignment]
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return int(parsed.timestamp())


def build_custom_fields_values(
    field_payload: dict[str, Any],
    field_catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    missing: list[str] = []
    custom_fields_values: list[dict[str, Any]] = []
    for field_name, value in field_payload.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue

        meta = _find_field_meta(field_catalog, field_name)
        if meta is None or meta.get("id") is None:
            missing.append(field_name)
            continue
        normalized_value = _normalize_field_value_for_meta(meta, value)
        custom_fields_values.append(
            {
                "field_id": int(meta["id"]),
                "values": _field_values(normalized_value),
            }
        )

    if missing:
        raise AmoIntegrationError(
            f"AMO custom fields not found in account: {', '.join(sorted(missing))}. "
            "Sync contacts/custom_fields and verify field names.",
            status_code=409,
        )
    return custom_fields_values


def send_contact_custom_field_update(
    session: Session,
    *,
    contact_id: int,
    field_payload: dict[str, Any],
) -> dict[str, Any]:
    context = resolve_amo_access_context(session)
    field_catalog = fetch_contact_field_catalog(session)
    sanitized_payload = {
        field_name: value
        for field_name, value in field_payload.items()
        if field_name not in CONTACT_WRITE_PROTECTED_FIELDS
    }
    custom_fields_values = build_custom_fields_values(sanitized_payload, field_catalog)
    if not custom_fields_values:
        raise AmoIntegrationError("No AMO custom field values were prepared for update.", status_code=400)

    result = _amo_http_request(
        method="PATCH",
        url=_contact_update_endpoint(context.account_base_url, contact_id),
        headers={"Authorization": f"Bearer {context.access_token}"},
        body={
            "id": int(contact_id),
            "custom_fields_values": custom_fields_values,
        },
    )
    return {
        "mode": "amo_api",
        "account_base_url": context.account_base_url,
        "entity_type": "contact",
        "entity_id": int(contact_id),
        "updated_fields": sorted(sanitized_payload.keys()),
        "custom_fields_values": custom_fields_values,
        "amo_response": result,
    }


def _flatten_lead_field_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "name": item.get("name"),
        "code": item.get("code"),
        "type": item.get("type"),
        "group_id": item.get("group_id"),
        "is_predefined": bool(item.get("is_predefined")),
        "is_api_only": bool(item.get("is_api_only")),
    }


def _lead_field_cache_path() -> Path:
    cache_dir = Path(settings.crm_amo_deal_queue_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "lead_field_catalog_cache.json"


def _read_lead_field_cache() -> tuple[list[dict[str, Any]], Optional[datetime]]:
    path = _lead_field_cache_path()
    if not path.exists():
        return [], None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [], None
    fields = payload.get("fields")
    synced_at_raw = payload.get("synced_at")
    synced_at = None
    if isinstance(synced_at_raw, str) and synced_at_raw:
        try:
            synced_at = datetime.fromisoformat(synced_at_raw)
        except ValueError:
            synced_at = None
    if not isinstance(fields, list):
        return [], synced_at
    return [item for item in fields if isinstance(item, dict)], synced_at


def _write_lead_field_cache(fields: list[dict[str, Any]]) -> None:
    path = _lead_field_cache_path()
    payload = {
        "synced_at": utc_now().isoformat(),
        "fields": fields,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_lead_field_catalog(session: Session, *, force_refresh: bool = False) -> list[dict[str, Any]]:
    cached_fields, synced_at = _read_lead_field_cache()
    if (
        not force_refresh
        and cached_fields
        and synced_at is not None
        and synced_at >= (utc_now() - AMO_CONTACT_FIELD_CACHE_TTL)
    ):
        return cached_fields

    fields: list[dict[str, Any]] = []
    next_url = _lead_custom_fields_endpoint(resolve_amo_access_context(session).account_base_url)
    while next_url:
        payload = amo_api_request(session, method="GET", path_or_url=next_url)
        embedded = payload.get("_embedded") if isinstance(payload, dict) else {}
        items = embedded.get("custom_fields") if isinstance(embedded, dict) else []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    fields.append(_flatten_lead_field_item(item))
        next_link = None
        links = payload.get("_links") if isinstance(payload, dict) else {}
        if isinstance(links, dict):
            next_meta = links.get("next")
            if isinstance(next_meta, dict):
                next_link = next_meta.get("href")
        next_url = next_link
    _write_lead_field_cache(fields)
    return fields


def send_lead_custom_field_update(
    session: Session,
    *,
    lead_id: int,
    field_payload: dict[str, Any],
) -> dict[str, Any]:
    context = resolve_amo_access_context(session)
    field_catalog = fetch_lead_field_catalog(session)
    custom_fields_values = build_custom_fields_values(field_payload, field_catalog)
    if not custom_fields_values:
        raise AmoIntegrationError("No AMO lead custom field values were prepared for update.", status_code=400)
    result = _amo_http_request(
        method="PATCH",
        url=_lead_update_endpoint(context.account_base_url, lead_id),
        headers={"Authorization": f"Bearer {context.access_token}"},
        body={
            "id": int(lead_id),
            "custom_fields_values": custom_fields_values,
        },
    )
    return {
        "mode": "amo_api",
        "account_base_url": context.account_base_url,
        "entity_type": "lead",
        "entity_id": int(lead_id),
        "updated_fields": sorted(field_payload.keys()),
        "custom_fields_values": custom_fields_values,
        "amo_response": result,
    }


def _contact_phones(contact: dict[str, Any]) -> list[str]:
    phones: list[str] = []
    for item in contact.get("custom_fields_values") or []:
        if not isinstance(item, dict):
            continue
        field_code = str(item.get("field_code") or "").strip().upper()
        field_name = str(item.get("field_name") or "").strip().casefold()
        if field_code != "PHONE" and "тел" not in field_name and "phone" not in field_name:
            continue
        for value_item in item.get("values") or []:
            if not isinstance(value_item, dict):
                continue
            normalized = normalize_phone(value_item.get("value"))
            if normalized:
                phones.append(normalized)
    unique: list[str] = []
    seen: set[str] = set()
    for phone in phones:
        if phone in seen:
            continue
        seen.add(phone)
        unique.append(phone)
    return unique


def search_contacts_by_phone(
    session: Session,
    *,
    phone: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    normalized_phone = normalize_phone(phone)
    if not normalized_phone:
        return []
    payload = amo_api_request(
        session,
        method="GET",
        path_or_url=_contacts_search_endpoint(resolve_amo_access_context(session).account_base_url),
        params={
            "query": normalized_phone[-10:],
            "limit": max(1, min(limit, 50)),
            "with": "leads",
        },
    )
    contacts = (payload.get("_embedded") or {}).get("contacts") or []
    return [
        contact
        for contact in contacts
        if normalized_phone in _contact_phones(contact)
    ]


def fetch_contact(
    session: Session,
    *,
    contact_id: int,
    with_fields: Optional[str] = None,
) -> dict[str, Any]:
    params = {"with": with_fields} if with_fields else None
    return amo_api_request(
        session,
        method="GET",
        path_or_url=_contact_entity_endpoint(resolve_amo_access_context(session).account_base_url, contact_id),
        params=params,
    )


def fetch_related_leads(
    session: Session,
    *,
    contact_id: int,
    limit: int = 250,
) -> list[dict[str, Any]]:
    params = {"filter[contacts]": int(contact_id), "limit": max(1, min(limit, 250))}
    url = _leads_collection_endpoint(resolve_amo_access_context(session).account_base_url)
    query = url_parse.urlencode(params)
    return _paged_embedded_items(
        session,
        initial_url=f"{url}?{query}",
        embedded_key="leads",
    )


def fetch_lead(
    session: Session,
    *,
    lead_id: int,
    with_fields: Optional[str] = "contacts",
) -> dict[str, Any]:
    params = {"with": with_fields} if with_fields else None
    return amo_api_request(
        session,
        method="GET",
        path_or_url=_lead_entity_endpoint(resolve_amo_access_context(session).account_base_url, lead_id),
        params=params,
    )


def fetch_lead_notes(
    session: Session,
    *,
    lead_id: int,
    limit: int = 250,
) -> list[dict[str, Any]]:
    url = _lead_notes_endpoint(resolve_amo_access_context(session).account_base_url, lead_id)
    query = url_parse.urlencode({"limit": max(1, min(limit, 250))})
    return _paged_embedded_items(
        session,
        initial_url=f"{url}?{query}",
        embedded_key="notes",
    )


def fetch_lead_tasks(
    session: Session,
    *,
    lead_id: int,
    limit: int = 250,
) -> list[dict[str, Any]]:
    url = _tasks_endpoint(resolve_amo_access_context(session).account_base_url)
    query = url_parse.urlencode(
        {
            "filter[entity_type]": "leads",
            "filter[entity_id]": int(lead_id),
            "limit": max(1, min(limit, 250)),
        }
    )
    return _paged_embedded_items(
        session,
        initial_url=f"{url}?{query}",
        embedded_key="tasks",
    )


def fetch_pipelines_with_statuses(session: Session) -> list[dict[str, Any]]:
    payload = amo_api_request(
        session,
        method="GET",
        path_or_url=_pipelines_endpoint(resolve_amo_access_context(session).account_base_url),
    )
    embedded = payload.get("_embedded") if isinstance(payload, dict) else {}
    items = embedded.get("pipelines") if isinstance(embedded, dict) else None
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def fetch_users(session: Session) -> list[dict[str, Any]]:
    url = f"{_users_endpoint(resolve_amo_access_context(session).account_base_url)}?limit=250"
    return _paged_embedded_items(session, initial_url=url, embedded_key="users")


def fetch_recent_leads(
    session: Session,
    *,
    closed_from_ts: Optional[int] = None,
    limit_per_page: int = 250,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "limit": max(1, min(limit_per_page, 250)),
        "with": "contacts",
    }
    if closed_from_ts is not None:
        params["filter[closed_at][from]"] = int(closed_from_ts)
    query = url_parse.urlencode(params)
    url = f"{_leads_collection_endpoint(resolve_amo_access_context(session).account_base_url)}?{query}"
    return _paged_embedded_items(session, initial_url=url, embedded_key="leads")


def get_amo_connection_status(session: Session) -> dict[str, Any]:
    connection = get_active_connection(session)
    setup = build_external_oauth_setup()
    env_base = _normalize_base_url(settings.crm_amo_base_url)
    env_token = str(settings.crm_amo_api_token or "").strip()
    env_connected = bool(env_base and env_token)
    connection_status = connection.status if connection is not None else None
    token_stale = _token_is_stale(connection) if connection is not None else False
    connection_can_sync_lead_fields = bool(
        connection is not None
        and connection.access_token
        and connection_status == "active"
        and not token_stale
    )
    lead_fields: list[dict[str, Any]] = []
    lead_field_error: Optional[str] = None
    if env_connected or connection_can_sync_lead_fields:
        try:
            lead_fields = fetch_lead_field_catalog(session)
        except Exception as exc:
            lead_field_error = str(exc)
    else:
        lead_field_error = "skipped: amoCRM connection is not active"
    lead_field_names = {str(item.get("name") or "").strip() for item in lead_fields if isinstance(item, dict)}
    required_lead_present = [
        field_name for field_name in DEFAULT_REQUIRED_AMO_LEAD_FIELDS if field_name in lead_field_names
    ]
    required_lead_missing = [
        field_name for field_name in DEFAULT_REQUIRED_AMO_LEAD_FIELDS if field_name not in lead_field_names
    ]
    status_payload: dict[str, Any] = {
        **setup,
        "connected": env_connected,
        "status": "direct_token" if env_connected else "not_connected",
        "account_base_url": env_base or _resolve_account_base_url_hint(),
        "account_subdomain": _account_subdomain(env_base or _resolve_account_base_url_hint()),
        "client_id_present": False,
        "client_secret_present": False,
        "access_token_present": bool(env_token),
        "refresh_token_present": False,
        "authorized_at": None,
        "expires_at": None,
        "last_error": None,
        "contact_field_catalog_synced_at": None,
        "contact_field_count": 0,
        "required_contact_fields_present": [],
        "required_contact_fields_missing": list(DEFAULT_REQUIRED_AMO_CONTACT_FIELDS),
        "lead_field_count": len(lead_fields),
        "required_lead_fields_present": required_lead_present,
        "required_lead_fields_missing": required_lead_missing,
        "lead_field_sync_error": lead_field_error,
        "token_source": "env" if env_connected else None,
    }
    if connection is None:
        return status_payload

    field_catalog = connection.contact_field_catalog or []
    field_names = {str(item.get("name") or "").strip() for item in field_catalog if isinstance(item, dict)}
    required_present = [
        field_name for field_name in DEFAULT_REQUIRED_AMO_CONTACT_FIELDS if field_name in field_names
    ]
    required_missing = [
        field_name for field_name in DEFAULT_REQUIRED_AMO_CONTACT_FIELDS if field_name not in field_names
    ]
    connected = bool(connection.access_token) and not token_stale and connection_status not in {
        "reauthorization_required",
        "refresh_error",
        "awaiting_callback",
        "awaiting_secrets",
        "pending",
    }
    if token_stale and connection_status == "active":
        connection_status = "token_stale"
    return {
        **setup,
        "connected": connected,
        "status": connection_status,
        "account_base_url": connection.account_base_url,
        "account_subdomain": connection.account_subdomain,
        "client_id_present": bool(connection.client_id),
        "client_secret_present": bool(connection.client_secret),
        "access_token_present": bool(connection.access_token),
        "refresh_token_present": bool(connection.refresh_token),
        "authorized_at": connection.authorized_at,
        "expires_at": connection.expires_at,
        "last_error": connection.last_error,
        "contact_field_catalog_synced_at": connection.contact_field_catalog_synced_at,
        "contact_field_count": len(field_catalog),
        "required_contact_fields_present": required_present,
        "required_contact_fields_missing": required_missing,
        "lead_field_count": len(lead_fields),
        "required_lead_fields_present": required_lead_present,
        "required_lead_fields_missing": required_lead_missing,
        "lead_field_sync_error": lead_field_error,
        "token_source": "oauth",
    }
