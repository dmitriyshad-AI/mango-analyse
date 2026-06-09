from __future__ import annotations

import json
import os
import socket
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request


AMO_WAPPI_ENV_FILE = Path.home() / ".mango_secrets" / "amo_wappi.env"
AMO_WAPPI_CONFIG_PATH_ENV = "AMO_WAPPI_CONFIG_PATH"
DEFAULT_AMO_WAPPI_CONFIG_PATH = Path.home() / ".mango_secrets" / "amo_wappi_phase1.json"
WAPPI_DEFAULT_BASE_URL = "https://wappi.pro"
VALID_BRANDS = frozenset({"foton", "unpk"})
DRAFT_NOTE_MARKER = "ЧЕРНОВИК БОТА, не отправлено"
MOSCOW_TZ = ZoneInfo("Europe/Moscow")
MAX_DRAFT_NOTE_TEXT_CHARS = 6000

JsonTransport = Callable[..., Mapping[str, Any]]


class AmoWappiPhase1Error(RuntimeError):
    pass


class AmoWappiConfigError(AmoWappiPhase1Error):
    pass


class AmoWappiWriteBlocked(AmoWappiPhase1Error):
    pass


class AmoWappiHttpError(AmoWappiPhase1Error):
    pass


def load_env_file(path: Path | str = AMO_WAPPI_ENV_FILE, *, override: bool = False) -> dict[str, str]:
    env_path = Path(path).expanduser()
    if not env_path.exists():
        raise AmoWappiConfigError(f"Env file not found: {env_path}")
    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if not key:
            continue
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def _normalize_base_url(value: str, *, default: str = "") -> str:
    candidate = str(value or default or "").strip()
    if not candidate:
        raise AmoWappiConfigError("Base URL is required.")
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = url_parse.urlparse(candidate)
    if not parsed.netloc and parsed.path:
        parsed = url_parse.urlparse(f"https://{parsed.path}")
    if not parsed.netloc:
        raise AmoWappiConfigError(f"Invalid base URL: {value!r}")
    return f"{parsed.scheme or 'https'}://{parsed.netloc}".rstrip("/")


def _json_http_request(
    *,
    method: str,
    url: str,
    headers: Optional[Mapping[str, str]] = None,
    json_body: Any = None,
    timeout_seconds: int = 25,
) -> Mapping[str, Any]:
    payload = None
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(dict(headers))
    if json_body is not None:
        payload = json.dumps(json_body, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = url_request.Request(url, data=payload, headers=request_headers, method=method.upper())
    try:
        with url_request.urlopen(request, timeout=max(1, int(timeout_seconds))) as response:
            raw = response.read().decode("utf-8")
    except url_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AmoWappiHttpError(f"HTTP {exc.code}: {detail or exc.reason}") from exc
    except (url_error.URLError, TimeoutError, socket.timeout, ssl.SSLError) as exc:
        reason = getattr(exc, "reason", exc)
        raise AmoWappiHttpError(f"Request failed: {reason}") from exc
    if not raw.strip():
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AmoWappiHttpError("Invalid JSON response.") from exc
    if isinstance(decoded, Mapping):
        return decoded
    return {"data": decoded}


def _read_env(env: Mapping[str, str], *keys: str, required: bool = True) -> str:
    for key in keys:
        value = str(env.get(key) or "").strip()
        if value:
            return value
    if required:
        raise AmoWappiConfigError(f"Missing required env variable: {'/'.join(keys)}")
    return ""


def _normalize_brand(value: Any) -> str:
    brand = str(value or "").strip().casefold()
    if brand not in VALID_BRANDS:
        raise AmoWappiConfigError(f"Unknown brand: {value!r}")
    return brand


@dataclass(frozen=True)
class AmoWappiPhase1Config:
    profile_brand_map: Mapping[str, str] = field(default_factory=dict)
    profile_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    allowed_test_lead_ids: frozenset[str] = field(default_factory=frozenset)
    manager_edit_log_path: Path = Path("runs/amo_wappi_phase1/manager_edits.jsonl")

    @classmethod
    def from_file(cls, path: Path | str | None = None) -> "AmoWappiPhase1Config":
        config_path = Path(path or os.getenv(AMO_WAPPI_CONFIG_PATH_ENV) or DEFAULT_AMO_WAPPI_CONFIG_PATH).expanduser()
        if not config_path.exists():
            raise AmoWappiConfigError(f"Config file not found: {config_path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise AmoWappiConfigError("AMO/Wappi config must be a JSON object.")
        profiles_raw = payload.get("profiles") or payload.get("profile_brand_map") or {}
        if not isinstance(profiles_raw, Mapping):
            raise AmoWappiConfigError("profiles/profile_brand_map must be an object.")
        profile_brand_map: dict[str, str] = {}
        profile_metadata: dict[str, Mapping[str, Any]] = {}
        for profile_id, raw in profiles_raw.items():
            key = str(profile_id or "").strip()
            if not key:
                continue
            if isinstance(raw, Mapping):
                brand = _normalize_brand(raw.get("brand"))
                profile_metadata[key] = dict(raw)
            else:
                brand = _normalize_brand(raw)
                profile_metadata[key] = {"brand": brand}
            profile_brand_map[key] = brand
        allowed = frozenset(str(item).strip() for item in payload.get("allowed_test_lead_ids", []) if str(item).strip())
        log_path = Path(str(payload.get("manager_edit_log_path") or "runs/amo_wappi_phase1/manager_edits.jsonl")).expanduser()
        return cls(
            profile_brand_map=profile_brand_map,
            profile_metadata=profile_metadata,
            allowed_test_lead_ids=allowed,
            manager_edit_log_path=log_path,
        )

    def brand_for_profile(self, profile_id: str) -> str:
        key = str(profile_id or "").strip()
        brand = self.profile_brand_map.get(key)
        if not brand:
            raise AmoWappiConfigError(f"Wappi profile_id is not mapped to a brand: {key!r}")
        return brand

    def require_note_allowed(self, lead_id: int | str) -> str:
        normalized = str(lead_id).strip()
        if normalized not in self.allowed_test_lead_ids:
            raise AmoWappiWriteBlocked(f"AMO draft-note write blocked: lead_id {normalized} is not in allowlist.")
        return normalized


@dataclass(frozen=True)
class AmoClientConfig:
    base_url: str
    access_token: str
    timeout_seconds: int = 25

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AmoClientConfig":
        source = env or os.environ
        base_url = _normalize_base_url(_read_env(source, "AMO_WAPPI_AMO_BASE_URL", "AMOCRM_BASE_URL"))
        token = _read_env(source, "AMO_WAPPI_AMO_ACCESS_TOKEN", "AMOCRM_ACCESS_TOKEN", "AMO_ACCESS_TOKEN")
        timeout = int(source.get("AMO_WAPPI_HTTP_TIMEOUT_SEC", "25") or "25")
        return cls(base_url=base_url, access_token=token, timeout_seconds=timeout)


class AmoPhase1Client:
    def __init__(self, config: AmoClientConfig, *, transport: JsonTransport | None = None) -> None:
        self.config = config
        self.transport = transport

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json_body: Any = None,
    ) -> Mapping[str, Any]:
        url = url_parse.urljoin(f"{self.config.base_url.rstrip('/')}/", path.lstrip("/"))
        if params:
            query = url_parse.urlencode({key: value for key, value in params.items() if value not in (None, "")}, doseq=True)
            if query:
                url = f"{url}{'&' if '?' in url else '?'}{query}"
        headers = {"Authorization": f"Bearer {self.config.access_token}"}
        if self.transport is not None:
            return self.transport(
                method=method.upper(),
                url=url,
                headers=headers,
                json_body=json_body,
                timeout_seconds=self.config.timeout_seconds,
            )
        return _json_http_request(
            method=method,
            url=url,
            headers=headers,
            json_body=json_body,
            timeout_seconds=self.config.timeout_seconds,
        )

    def list_pipelines(self) -> Mapping[str, Any]:
        return self._request("GET", "/api/v4/leads/pipelines", params={"with": "statuses"})

    def get_lead(self, lead_id: int | str, *, with_contacts: bool = True) -> Mapping[str, Any]:
        params = {"with": "contacts"} if with_contacts else None
        return self._request("GET", f"/api/v4/leads/{int(lead_id)}", params=params)

    def list_contacts(self, *, query: str = "", limit: int = 50) -> Mapping[str, Any]:
        params: dict[str, Any] = {"limit": max(1, min(int(limit), 250))}
        if query:
            params["query"] = query
        return self._request("GET", "/api/v4/contacts", params=params)

    def get_contact(self, contact_id: int | str, *, with_leads: bool = True) -> Mapping[str, Any]:
        params = {"with": "leads"} if with_leads else None
        return self._request("GET", f"/api/v4/contacts/{int(contact_id)}", params=params)

    def add_draft_note_to_test_lead(
        self,
        lead_id: int | str,
        *,
        config: AmoWappiPhase1Config,
        draft_text: str,
        brand: str,
        profile_id: str = "",
        route: str = "",
        safety_flags: Sequence[str] = (),
        outgoing_visibility_note: str = "",
        created_at: datetime | None = None,
    ) -> Mapping[str, Any]:
        allowed_lead_id = config.require_note_allowed(lead_id)
        note_text = build_draft_note_text(
            draft_text=draft_text,
            brand=brand,
            profile_id=profile_id,
            route=route,
            safety_flags=safety_flags,
            outgoing_visibility_note=outgoing_visibility_note,
            created_at=created_at,
        )
        body = [{"note_type": "common", "params": {"text": note_text}}]
        return self._request("POST", f"/api/v4/leads/{int(allowed_lead_id)}/notes", json_body=body)


@dataclass(frozen=True)
class WappiClientConfig:
    base_url: str
    telegram_token: str = ""
    max_token: str = ""
    timeout_seconds: int = 25

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "WappiClientConfig":
        source = env or os.environ
        base_url = _normalize_base_url(str(source.get("WAPPI_API_BASE_URL") or WAPPI_DEFAULT_BASE_URL), default=WAPPI_DEFAULT_BASE_URL)
        token = str(source.get("WAPPI_API_TOKEN") or "").strip()
        telegram = str(source.get("WAPPI_TELEGRAM_TOKEN") or token).strip()
        max_token = str(source.get("WAPPI_MAX_TOKEN") or token).strip()
        timeout = int(source.get("AMO_WAPPI_HTTP_TIMEOUT_SEC", "25") or "25")
        return cls(base_url=base_url, telegram_token=telegram, max_token=max_token, timeout_seconds=timeout)


class WappiPhase1Client:
    def __init__(self, config: WappiClientConfig, *, transport: JsonTransport | None = None) -> None:
        self.config = config
        self.transport = transport

    def _token_for_channel(self, channel: str) -> str:
        normalized = str(channel or "").strip().casefold()
        if normalized == "telegram":
            token = self.config.telegram_token
        elif normalized == "max":
            token = self.config.max_token
        else:
            raise AmoWappiConfigError(f"Unknown Wappi channel: {channel!r}")
        if not token:
            raise AmoWappiConfigError(f"Wappi {normalized} token is not configured.")
        return token

    def list_profiles(self, channel: str) -> list[Mapping[str, Any]]:
        normalized = str(channel or "").strip().casefold()
        path = "/tapi/profile/all/get" if normalized == "telegram" else "/maxapi/profile/all/get"
        token = self._token_for_channel(normalized)
        url = url_parse.urljoin(f"{self.config.base_url.rstrip('/')}/", path.lstrip("/"))
        headers = {"Authorization": token}
        if self.transport is not None:
            payload = self.transport(method="GET", url=url, headers=headers, json_body=None, timeout_seconds=self.config.timeout_seconds)
        else:
            payload = _json_http_request(method="GET", url=url, headers=headers, timeout_seconds=self.config.timeout_seconds)
        return _extract_wappi_profiles(payload, channel=normalized)

    def list_all_profiles(self) -> list[Mapping[str, Any]]:
        profiles: list[Mapping[str, Any]] = []
        if self.config.telegram_token:
            profiles.extend(self.list_profiles("telegram"))
        if self.config.max_token:
            profiles.extend(self.list_profiles("max"))
        return profiles

    def list_telegram_chats(
        self,
        *,
        profile_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
        show_all: bool = False,
    ) -> Mapping[str, Any]:
        params = {
            "profile_id": str(profile_id),
            "limit": max(1, min(int(limit), 100)),
            "offset": max(0, int(offset)),
            "order": str(order or "desc"),
            "show_all": "true" if show_all else "false",
        }
        return self._request_telegram("GET", "/tapi/sync/chats/get", params=params)

    def get_telegram_chat_messages(
        self,
        *,
        profile_id: str,
        chat_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
        mark_all: bool = False,
    ) -> Mapping[str, Any]:
        params = {
            "profile_id": str(profile_id),
            "chat_id": str(chat_id),
            "limit": max(1, min(int(limit), 100)),
            "offset": max(0, int(offset)),
            "order": str(order or "desc"),
            "mark_all": "true" if mark_all else "false",
        }
        return self._request_telegram("GET", "/tapi/sync/messages/get", params=params)

    def _request_telegram(self, method: str, path: str, *, params: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        token = self._token_for_channel("telegram")
        url = url_parse.urljoin(f"{self.config.base_url.rstrip('/')}/", path.lstrip("/"))
        if params:
            query = url_parse.urlencode({key: value for key, value in params.items() if value not in (None, "")}, doseq=True)
            if query:
                url = f"{url}?{query}"
        headers = {"Authorization": token}
        if self.transport is not None:
            return self.transport(
                method=method.upper(),
                url=url,
                headers=headers,
                json_body=None,
                timeout_seconds=self.config.timeout_seconds,
            )
        return _json_http_request(method=method, url=url, headers=headers, timeout_seconds=self.config.timeout_seconds)


def _extract_wappi_profiles(payload: Mapping[str, Any], *, channel: str) -> list[Mapping[str, Any]]:
    candidates: Any = payload.get("profiles")
    if candidates is None:
        candidates = payload.get("data")
    if isinstance(candidates, Mapping):
        candidates = candidates.get("profiles") or candidates.get("items") or candidates.get("data")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)):
        return []
    result: list[Mapping[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        profile_id = str(item.get("profile_id") or item.get("id") or item.get("profileId") or "").strip()
        if not profile_id:
            continue
        normalized = dict(item)
        normalized["profile_id"] = profile_id
        normalized["channel"] = channel
        result.append(normalized)
    return result


def build_draft_note_text(
    *,
    draft_text: str,
    brand: str,
    profile_id: str = "",
    route: str = "",
    safety_flags: Sequence[str] = (),
    outgoing_visibility_note: str = "",
    created_at: datetime | None = None,
) -> str:
    timestamp = (created_at or datetime.now(timezone.utc)).astimezone(MOSCOW_TZ).isoformat()
    brand_value = _normalize_brand(brand)
    route_value = str(route or "").strip()
    flags = tuple(str(item).strip() for item in safety_flags if str(item).strip())
    if route_value in {"bot_answer_self", "bot_answer_self_for_pilot"}:
        route_line = "Маршрут: черновик (бот считает безопасным)"
    elif route_value == "draft_for_manager":
        route_line = "Маршрут: черновик (проверить)"
    elif route_value in {"manager_only", "blocked"}:
        route_line = "Маршрут: бот передал менеджеру"
    elif route_value:
        route_line = f"Маршрут: {route_value}"
    else:
        route_line = ""
    parts = [
        DRAFT_NOTE_MARKER,
        f"Бренд: {brand_value}",
        f"Время: {timestamp} (Europe/Moscow)",
    ]
    if str(profile_id or "").strip():
        parts.append(f"Wappi profile_id: {str(profile_id).strip()}")
    if route_line:
        parts.append(route_line)
    if flags:
        parts.append("Флаги безопасности: " + ", ".join(flags))
    if str(outgoing_visibility_note or "").strip():
        parts.append(str(outgoing_visibility_note).strip())
    parts.extend(["", str(draft_text or "").strip()])
    text = "\n".join(parts).strip()
    if len(text) <= MAX_DRAFT_NOTE_TEXT_CHARS:
        return text
    marker = "\n\n[Черновик усечён до 6000 символов]"
    return text[: MAX_DRAFT_NOTE_TEXT_CHARS - len(marker)].rstrip() + marker


@dataclass(frozen=True)
class ManagerEditLogRecord:
    lead_id: str
    brand: str
    profile_id: str
    bot_draft_text: str
    chat_id: str = ""
    message_id: str = ""
    matched_message_id: str = ""
    draft_route: str = ""
    match_class: str = ""
    ratio: float | None = None
    draft_ts: str = ""
    sent_ts: str = ""
    window_closed: bool = False
    manager_sent_text: str = ""
    reason_codes: tuple[str, ...] = ()
    created_at: str = ""

    def as_json(self) -> dict[str, Any]:
        ratio_value = None if self.ratio is None else float(self.ratio)
        return {
            "schema_version": "amo_wappi_manager_edit_log_v2_2026_06_10",
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
            "lead_id": str(self.lead_id),
            "brand": _normalize_brand(self.brand),
            "profile_id": str(self.profile_id or ""),
            "chat_id": str(self.chat_id or ""),
            "message_id": str(self.message_id or ""),
            "matched_message_id": str(self.matched_message_id or ""),
            "draft_route": str(self.draft_route or ""),
            "match_class": str(self.match_class or ""),
            "ratio": ratio_value,
            "draft_ts": str(self.draft_ts or ""),
            "sent_ts": str(self.sent_ts or ""),
            "window_closed": bool(self.window_closed),
            "bot_draft_text": str(self.bot_draft_text or ""),
            "manager_sent_text": str(self.manager_sent_text or ""),
            "reason_codes": list(self.reason_codes),
        }


def append_manager_edit_log(path: Path | str, record: ManagerEditLogRecord) -> None:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.as_json(), ensure_ascii=False) + "\n")
