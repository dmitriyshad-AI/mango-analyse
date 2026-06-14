from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Optional

from fastapi import Header, HTTPException, Query, Request, status

from mango_mvp.amocrm_runtime.config import get_settings


settings = get_settings()


@dataclass(frozen=True)
class AuthContext:
    api_key_id: str
    role: str
    actor: str


DEFAULT_DEV_CONTEXT = AuthContext(
    api_key_id="local-dev",
    role="Director",
    actor="director",
)


def _parse_api_key_specs() -> dict[str, AuthContext]:
    principals: dict[str, AuthContext] = {}
    for spec in settings.api_keys:
        raw = spec.strip()
        if not raw:
            continue

        parts = raw.split(":", 2)
        if len(parts) < 2:
            continue
        token = parts[0].strip()
        role = parts[1].strip()
        actor = parts[2].strip() if len(parts) == 3 and parts[2].strip() else role.lower()
        if not token or not role:
            continue
        principals[token] = AuthContext(api_key_id=token[-6:], role=role, actor=actor)

    if settings.api_key and settings.api_key not in principals:
        principals[settings.api_key] = AuthContext(
            api_key_id=settings.api_key[-6:],
            role="Director",
            actor="director",
        )
    return principals


AUTH_PRINCIPALS = _parse_api_key_specs()
DEV_AUTH_ENV_NAMES = ("AI_OFFICE_ALLOW_DEV_AUTH_CONTEXT", "MANGO_API_DEV_CONTEXT")
LOCAL_DEV_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _resolve_stream_token_secret() -> bytes:
    if settings.stream_token_secret:
        return settings.stream_token_secret.encode("utf-8")

    seed_parts: list[str] = []
    if settings.api_key:
        seed_parts.append(settings.api_key)
    seed_parts.extend(settings.api_keys)
    if seed_parts:
        digest = hashlib.sha256("|".join(seed_parts).encode("utf-8")).hexdigest()
        return digest.encode("utf-8")

    return secrets.token_hex(32).encode("utf-8")


STREAM_TOKEN_SECRET = _resolve_stream_token_secret()


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padded = value + ("=" * (-len(value) % 4))
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _sign_stream_payload(encoded_payload: str) -> str:
    signature = hmac.new(
        STREAM_TOKEN_SECRET,
        encoded_payload.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(signature)


def issue_stream_token(
    *,
    project_id: str,
    auth: AuthContext,
) -> tuple[str, datetime]:
    now_ts = int(time.time())
    ttl_seconds = max(15, min(settings.stream_token_ttl_seconds, 900))
    exp_ts = now_ts + ttl_seconds
    payload = {
        "v": 1,
        "project_id": project_id,
        "exp": exp_ts,
        "role": auth.role,
        "actor": auth.actor,
        "api_key_id": auth.api_key_id,
    }
    encoded_payload = _b64url_encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    signature = _sign_stream_payload(encoded_payload)
    token = f"{encoded_payload}.{signature}"
    expires_at = datetime.fromtimestamp(exp_ts, tz=timezone.utc)
    return token, expires_at


def verify_stream_token(token: str, *, project_id: str) -> AuthContext:
    parts = token.split(".", 1)
    if len(parts) != 2:
        raise ValueError("invalid stream token format")
    encoded_payload, encoded_signature = parts

    expected_signature = _sign_stream_payload(encoded_payload)
    if not secrets.compare_digest(encoded_signature, expected_signature):
        raise ValueError("invalid stream token signature")

    try:
        payload_raw = _b64url_decode(encoded_payload).decode("utf-8")
        payload = json.loads(payload_raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid stream token payload") from exc

    if str(payload.get("project_id", "")) != project_id:
        raise ValueError("invalid stream token project")

    exp_value = payload.get("exp")
    try:
        exp_ts = int(exp_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid stream token expiration") from exc

    now_ts = int(time.time())
    if exp_ts <= now_ts:
        raise ValueError("stream token expired")

    role = str(payload.get("role", "")).strip()
    actor = str(payload.get("actor", "")).strip()
    api_key_id = str(payload.get("api_key_id", "stream")).strip() or "stream"
    if not role or not actor:
        raise ValueError("invalid stream token subject")

    return AuthContext(api_key_id=api_key_id, role=role, actor=actor)


def _env_flag_enabled(*names: str) -> bool:
    for name in names:
        value = os.getenv(name)
        if value and value.strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _is_local_request(request: Request) -> bool:
    client = request.client
    host = client.host if client else ""
    return host in LOCAL_DEV_HOSTS


def _allow_dev_auth_context(request: Request) -> bool:
    return _env_flag_enabled(*DEV_AUTH_ENV_NAMES) and _is_local_request(request)


def get_auth_context(request: Request) -> AuthContext:
    context = getattr(request.state, "auth_context", None)
    if isinstance(context, AuthContext):
        return context
    if not AUTH_PRINCIPALS and _allow_dev_auth_context(request):
        request.state.auth_context = DEFAULT_DEV_CONTEXT
        return DEFAULT_DEV_CONTEXT
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
    )


def ensure_role(auth: AuthContext, *allowed_roles: str) -> None:
    if not allowed_roles:
        return
    if auth.role in allowed_roles:
        return
    allowed = ", ".join(allowed_roles)
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=f"Role '{auth.role}' is not allowed for this action. Allowed roles: {allowed}.",
    )


def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    api_key: Optional[str] = Query(default=None),
    stream_token: Optional[str] = Query(default=None),
) -> AuthContext:
    if not AUTH_PRINCIPALS:
        if _allow_dev_auth_context(request):
            request.state.auth_context = DEFAULT_DEV_CONTEXT
            return DEFAULT_DEV_CONTEXT
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
        )

    provided = x_api_key or api_key
    if provided is not None:
        for token, principal in AUTH_PRINCIPALS.items():
            if secrets.compare_digest(provided, token):
                request.state.auth_context = principal
                return principal

    if stream_token:
        is_stream_endpoint = request.url.path.endswith("/events/stream")
        project_id = request.path_params.get("project_id")
        if not is_stream_endpoint or not project_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key.",
            )
        try:
            principal = verify_stream_token(stream_token, project_id=project_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key.",
            ) from None
        request.state.auth_context = principal
        return principal

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid API key.",
    )
