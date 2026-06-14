from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from mango_mvp.amocrm_runtime import auth
from mango_mvp.amocrm_runtime.config import get_settings


def _request(host: str, path: str = "/api/integrations/amocrm/status") -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [],
            "query_string": b"",
            "client": (host, 54321),
            "server": ("testserver", 80),
            "scheme": "http",
        }
    )


def test_empty_keys_do_not_grant_director_context_for_non_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "AUTH_PRINCIPALS", {})
    monkeypatch.delenv("AI_OFFICE_ALLOW_DEV_AUTH_CONTEXT", raising=False)
    monkeypatch.delenv("MANGO_API_DEV_CONTEXT", raising=False)

    with pytest.raises(HTTPException) as exc:
        auth.require_api_key(_request("203.0.113.10"))

    assert exc.value.status_code == 401


def test_empty_keys_require_explicit_dev_escape_even_on_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "AUTH_PRINCIPALS", {})
    monkeypatch.delenv("AI_OFFICE_ALLOW_DEV_AUTH_CONTEXT", raising=False)
    monkeypatch.delenv("MANGO_API_DEV_CONTEXT", raising=False)

    with pytest.raises(HTTPException) as exc:
        auth.require_api_key(_request("127.0.0.1"))

    assert exc.value.status_code == 401


def test_explicit_dev_escape_is_localhost_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth, "AUTH_PRINCIPALS", {})
    monkeypatch.setenv("AI_OFFICE_ALLOW_DEV_AUTH_CONTEXT", "1")

    assert auth.require_api_key(_request("127.0.0.1")) == auth.DEFAULT_DEV_CONTEXT
    with pytest.raises(HTTPException) as exc:
        auth.require_api_key(_request("203.0.113.10"))
    assert exc.value.status_code == 401


def test_valid_api_key_still_authenticates(monkeypatch: pytest.MonkeyPatch) -> None:
    principal = auth.AuthContext(api_key_id="secret", role="Director", actor="director")
    monkeypatch.setattr(auth, "AUTH_PRINCIPALS", {"secret-token": principal})

    assert auth.require_api_key(_request("203.0.113.10"), x_api_key="secret-token") == principal


def test_amocrm_runtime_defaults_are_not_public_or_wildcard_cors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("AI_OFFICE_CORS_ALLOW_ORIGINS", raising=False)

    settings = get_settings()

    assert settings.api_host == "127.0.0.1"
    assert settings.api_cors_allow_origins == ()


def test_cors_origins_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_OFFICE_CORS_ALLOW_ORIGINS", "https://api.fotonai.online, https://app.example")

    settings = get_settings()

    assert settings.api_cors_allow_origins == ("https://api.fotonai.online", "https://app.example")
