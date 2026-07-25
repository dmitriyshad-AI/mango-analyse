from __future__ import annotations

import hashlib
import json
import os
import socket
import ssl
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request


AMO_WAPPI_ENV_FILE = Path.home() / ".mango_secrets" / "amo_wappi.env"
AI_OFFICE_ENV_FILE = Path.home() / ".mango_secrets" / "ai_office.env"
AMO_WAPPI_CONFIG_PATH_ENV = "AMO_WAPPI_CONFIG_PATH"
DEFAULT_AMO_WAPPI_CONFIG_PATH = Path.home() / ".mango_secrets" / "amo_wappi_phase1.json"
WAPPI_DEFAULT_BASE_URL = "https://wappi.pro"
AI_OFFICE_DEFAULT_BASE_URL = "https://api.fotonai.online"
VALID_BRANDS = frozenset({"foton", "unpk"})
DRAFT_NOTE_MARKER = "ЧЕРНОВИК БОТА, не отправлено"
MOSCOW_TZ = ZoneInfo("Europe/Moscow")
MAX_DRAFT_NOTE_TEXT_CHARS = 6000

JsonTransport = Callable[..., Mapping[str, Any]]


class AmoNoteReadClient(Protocol):
    def amo_api_get(
        self,
        *,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        limit: int = 50,
    ) -> Mapping[str, Any]:
        ...


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
        chat_id: str = "",
        message_id: str = "",
        route: str = "",
        safety_flags: Sequence[str] = (),
        outgoing_visibility_note: str = "",
        created_at: datetime | None = None,
    ) -> Mapping[str, Any]:
        del chat_id, message_id
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
class AiOfficeClientConfig:
    base_url: str
    api_key: str
    timeout_seconds: int = 25

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AiOfficeClientConfig":
        source = env or os.environ
        base_url = _normalize_base_url(
            str(source.get("AI_OFFICE_API_BASE_URL") or AI_OFFICE_DEFAULT_BASE_URL),
            default=AI_OFFICE_DEFAULT_BASE_URL,
        )
        api_key = _read_env(source, "AI_OFFICE_API_KEY", "CRM_SERVER_API_KEY")
        timeout = int(source.get("AI_OFFICE_TIMEOUT_SEC", source.get("AMO_WAPPI_HTTP_TIMEOUT_SEC", "25")) or "25")
        return cls(base_url=base_url, api_key=api_key, timeout_seconds=timeout)


class AiOfficeAmoNoteClient:
    def __init__(
        self,
        config: AiOfficeClientConfig,
        *,
        transport: JsonTransport | None = None,
        read_client: AmoNoteReadClient | None = None,
    ) -> None:
        self.config = config
        self.transport = transport
        self.read_client = read_client

    def _request(self, method: str, path: str, *, json_body: Any = None) -> Mapping[str, Any]:
        url = url_parse.urljoin(f"{self.config.base_url.rstrip('/')}/", path.lstrip("/"))
        headers = {"X-API-Key": self.config.api_key, "User-Agent": "mango-draft-loop/1.0"}
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

    def add_draft_note_to_test_lead(
        self,
        lead_id: int | str,
        *,
        config: AmoWappiPhase1Config,
        draft_text: str,
        brand: str,
        profile_id: str = "",
        chat_id: str = "",
        message_id: str = "",
        route: str = "",
        safety_flags: Sequence[str] = (),
        outgoing_visibility_note: str = "",
        created_at: datetime | None = None,
    ) -> Mapping[str, Any]:
        del config
        return self._add_draft_note(
            entity_type="lead",
            entity_id=lead_id,
            draft_text=draft_text,
            brand=brand,
            profile_id=profile_id,
            chat_id=chat_id,
            message_id=message_id,
            route=route,
            safety_flags=safety_flags,
            outgoing_visibility_note=outgoing_visibility_note,
            created_at=created_at,
        )

    def add_draft_note_to_contact(
        self,
        contact_id: int | str,
        *,
        config: AmoWappiPhase1Config,
        draft_text: str,
        brand: str,
        profile_id: str = "",
        chat_id: str = "",
        message_id: str = "",
        route: str = "",
        safety_flags: Sequence[str] = (),
        outgoing_visibility_note: str = "",
        created_at: datetime | None = None,
    ) -> Mapping[str, Any]:
        del config
        return self._add_draft_note(
            entity_type="contact",
            entity_id=contact_id,
            draft_text=draft_text,
            brand=brand,
            profile_id=profile_id,
            chat_id=chat_id,
            message_id=message_id,
            route=route,
            safety_flags=safety_flags,
            outgoing_visibility_note=outgoing_visibility_note,
            created_at=created_at,
        )

    def _add_draft_note(
        self,
        *,
        entity_type: str,
        entity_id: int | str,
        draft_text: str,
        brand: str,
        profile_id: str,
        chat_id: str,
        message_id: str,
        route: str,
        safety_flags: Sequence[str],
        outgoing_visibility_note: str,
        created_at: datetime | None,
    ) -> Mapping[str, Any]:
        if entity_type not in {"lead", "contact"}:
            raise AmoWappiWriteBlocked("AMO draft-note entity must be lead or contact.")
        try:
            allowed_entity_id = str(int(str(entity_id).strip()))
        except (TypeError, ValueError) as exc:
            raise AmoWappiWriteBlocked(f"AMO draft-note write requires a positive numeric {entity_type}_id.") from exc
        if int(allowed_entity_id) <= 0:
            raise AmoWappiWriteBlocked(f"AMO draft-note write requires a positive numeric {entity_type}_id.")
        marker = draft_note_idempotency_marker(
            profile_id=profile_id,
            chat_id=chat_id,
            message_id=message_id,
        )
        existing_note_id = self._existing_note_id(entity_type, allowed_entity_id, marker)
        if existing_note_id:
            return {
                "status": "ok",
                f"{entity_type}_id": int(allowed_entity_id),
                "note_id": existing_note_id,
                "deduplicated": True,
                "idempotency_marker": marker,
            }
        note_text = build_draft_note_text(
            draft_text=draft_text,
            brand=brand,
            profile_id=profile_id,
            idempotency_marker=marker,
            route=route,
            safety_flags=safety_flags,
            outgoing_visibility_note=outgoing_visibility_note,
            created_at=created_at,
        )
        response = self._request(
            "POST",
            f"/api/integrations/amocrm/{entity_type}s/{int(allowed_entity_id)}/notes",
            json_body={"text": note_text},
        )
        try:
            response_note_id = int(response.get("note_id"))
        except (TypeError, ValueError) as exc:
            raise AmoWappiWriteBlocked("AI Office draft-note response has no valid note_id.") from exc
        if response_note_id <= 0:
            raise AmoWappiWriteBlocked("AI Office draft-note response has no valid note_id.")
        confirmed_note_id = self._existing_note_id(entity_type, allowed_entity_id, marker)
        if confirmed_note_id is None:
            raise AmoWappiWriteBlocked("AMO draft-note POST succeeded but readback did not confirm the note.")
        if confirmed_note_id != response_note_id:
            raise AmoWappiWriteBlocked("AMO draft-note readback note_id differs from the POST response.")
        return {
            **dict(response),
            "status": "ok",
            f"{entity_type}_id": int(allowed_entity_id),
            "note_id": confirmed_note_id,
            "deduplicated": False,
            "idempotency_marker": marker,
        }

    def find_existing_draft_note(
        self,
        *,
        entity_type: str,
        entity_id: int | str,
        profile_id: str,
        chat_id: str,
        message_id: str,
    ) -> Mapping[str, Any]:
        if entity_type not in {"lead", "contact"}:
            raise AmoWappiWriteBlocked("AMO draft-note entity must be lead or contact.")
        try:
            allowed_entity_id = str(int(str(entity_id).strip()))
        except (TypeError, ValueError) as exc:
            raise AmoWappiWriteBlocked(f"AMO draft-note readback requires a positive numeric {entity_type}_id.") from exc
        if int(allowed_entity_id) <= 0:
            raise AmoWappiWriteBlocked(f"AMO draft-note readback requires a positive numeric {entity_type}_id.")
        marker = draft_note_idempotency_marker(
            profile_id=profile_id,
            chat_id=chat_id,
            message_id=message_id,
        )
        note_id = self._existing_note_id(entity_type, allowed_entity_id, marker)
        return {
            "status": "found" if note_id else "missing",
            "note_id": note_id,
            "idempotency_marker": marker,
        }

    def _existing_note_id(self, entity_type: str, entity_id: str, marker: str) -> int | None:
        if self.read_client is None:
            raise AmoWappiWriteBlocked("AMO draft-note readback is required before write.")
        page_limit = 50
        for page in range(1, 21):
            query: dict[str, Any] = {"order[id]": "desc"}
            if page > 1:
                query["page"] = page
            payload = self.read_client.amo_api_get(
                path=f"{entity_type}s/{int(entity_id)}/notes",
                params=query,
                limit=page_limit,
            )
            embedded = payload.get("_embedded") if isinstance(payload, Mapping) else None
            notes = embedded.get("notes") if isinstance(embedded, Mapping) else payload.get("notes")
            if not isinstance(notes, Sequence) or isinstance(notes, (str, bytes, bytearray)):
                raise AmoWappiWriteBlocked("AMO draft-note readback returned an invalid notes payload.")
            for item in notes:
                if not isinstance(item, Mapping):
                    raise AmoWappiWriteBlocked("AMO draft-note readback returned an invalid note row.")
                params = item.get("params") if isinstance(item.get("params"), Mapping) else {}
                if marker not in str(params.get("text") or item.get("text") or ""):
                    continue
                try:
                    note_id = int(item.get("id"))
                except (TypeError, ValueError) as exc:
                    raise AmoWappiWriteBlocked("AMO draft-note readback returned an invalid note_id.") from exc
                if note_id <= 0:
                    raise AmoWappiWriteBlocked("AMO draft-note readback returned an invalid note_id.")
                return note_id
            links = payload.get("_links") if isinstance(payload, Mapping) else None
            if links is not None and not isinstance(links, Mapping):
                raise AmoWappiWriteBlocked("AMO draft-note readback returned invalid pagination links.")
            if isinstance(links, Mapping):
                if "next" not in links:
                    return None
                if not isinstance(links.get("next"), Mapping):
                    raise AmoWappiWriteBlocked("AMO draft-note readback returned an invalid next-page link.")
            elif len(notes) < page_limit:
                return None
        raise AmoWappiWriteBlocked("AMO draft-note readback exceeded the pagination limit.")


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

    def find_amocrm_contact(
        self,
        *,
        channel: str,
        chat_id: str,
        phone: str,
        username: str,
        platform: str,
        crm_id: str,
        profile_uuid: str,
        manager: str = "",
        avito_user_id: str = "",
        avito_user_hash: str = "",
    ) -> Mapping[str, Any]:
        del manager
        token = self._token_for_channel(channel)
        normalized_crm_id = str(crm_id or "").strip()
        if not normalized_crm_id:
            raise AmoWappiConfigError("Wappi AMO crm_id is required.")
        headers = {"Authorization": token}
        body = {
            "chat_id": str(chat_id or "").strip(),
            "phone": str(phone or "").strip(),
            "username": str(username or "").strip(),
            "avito_user_id": str(avito_user_id or "").strip(),
            "avito_user_hash": str(avito_user_hash or "").strip(),
            "platform": str(platform or "").strip(),
            "crm_id": normalized_crm_id,
            "profile_uuid": str(profile_uuid or "").strip(),
            "token": token,
        }
        if not body["chat_id"] or not body["platform"] or not body["profile_uuid"]:
            raise AmoWappiConfigError("Wappi chat_id, platform and profile_uuid are required.")
        url = url_parse.urljoin(
            f"{self.config.base_url.rstrip('/')}/",
            "amocrm/contact/find",
        )
        parsed_url = url_parse.urlparse(url)
        if parsed_url.scheme != "https" or parsed_url.netloc.casefold() != "wappi.pro":
            raise AmoWappiConfigError("Wappi AMO contact lookup requires https://wappi.pro.")
        if self.transport is not None:
            return self.transport(
                method="POST",
                url=url,
                headers=headers,
                json_body=body,
                timeout_seconds=self.config.timeout_seconds,
            )
        return _json_http_request(
            method="POST",
            url=url,
            headers=headers,
            json_body=body,
            timeout_seconds=self.config.timeout_seconds,
        )

    def find_amocrm_contact_for_dialog(
        self,
        *,
        channel: str,
        dialog: Mapping[str, Any],
        profile: Mapping[str, Any],
        crm_id: str,
    ) -> Mapping[str, Any]:
        lookup = wappi_amocrm_lookup_fields(channel, dialog)
        if not lookup["chat_id"]:
            raise AmoWappiConfigError("Wappi dialog does not contain one authoritative peer id")
        return self.find_amocrm_contact(
            channel=channel,
            chat_id=lookup["chat_id"],
            phone=lookup["phone"],
            username=lookup["username"],
            platform=str(profile.get("platform") or channel).strip(),
            crm_id=crm_id,
            profile_uuid=str(profile.get("uuid") or profile.get("profile_id") or "").strip(),
            manager=lookup["manager_id"],
            avito_user_id=str(dialog.get("avito_user_id") or "").strip(),
            avito_user_hash=str(dialog.get("avito_user_hash") or "").strip(),
        )

    def list_telegram_chats(
        self,
        *,
        profile_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
        show_all: bool = False,
    ) -> Mapping[str, Any]:
        return self.list_chats(channel="telegram", profile_id=profile_id, limit=limit, offset=offset, order=order, show_all=show_all)

    def list_chats(
        self,
        *,
        channel: str,
        profile_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
        show_all: bool = False,
    ) -> Mapping[str, Any]:
        normalized = str(channel or "").strip().casefold()
        path = "/tapi/sync/chats/get" if normalized == "telegram" else "/maxapi/sync/chats/get"
        params = {
            "profile_id": str(profile_id),
            "limit": max(1, min(int(limit), 100)),
            "offset": max(0, int(offset)),
            "order": str(order or "desc"),
            "show_all": "true" if show_all else "false",
        }
        return self._request_channel(normalized, "GET", path, params=params)

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
        return self.get_chat_messages(
            channel="telegram",
            profile_id=profile_id,
            chat_id=chat_id,
            limit=limit,
            offset=offset,
            order=order,
            mark_all=mark_all,
        )

    def get_chat_messages(
        self,
        *,
        channel: str,
        profile_id: str,
        chat_id: str,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
        mark_all: bool = False,
    ) -> Mapping[str, Any]:
        normalized = str(channel or "").strip().casefold()
        path = "/tapi/sync/messages/get" if normalized == "telegram" else "/maxapi/sync/messages/get"
        params = {
            "profile_id": str(profile_id),
            "chat_id": str(chat_id),
            "limit": max(1, min(int(limit), 100)),
            "offset": max(0, int(offset)),
            "order": str(order or "desc"),
            "mark_all": "true" if mark_all else "false",
        }
        return self._request_channel(normalized, "GET", path, params=params)

    def _request_telegram(self, method: str, path: str, *, params: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        return self._request_channel("telegram", method, path, params=params)

    def _request_channel(self, channel: str, method: str, path: str, *, params: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        token = self._token_for_channel(channel)
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


def wappi_amocrm_lookup_fields(channel: str, dialog: Mapping[str, Any]) -> Mapping[str, str]:
    normalized = str(channel or "").strip().casefold()
    user = dialog.get("user") if isinstance(dialog.get("user"), Mapping) else {}
    chat_id = str(dialog.get("id") or dialog.get("chat_id") or dialog.get("chatId") or "").strip()
    phone = str(dialog.get("phone") or user.get("Phone") or user.get("phone") or "").strip()
    username = str(user.get("Username") or user.get("username") or dialog.get("username") or "").strip()
    if normalized == "max" and str(dialog.get("type") or "").strip().upper() == "DIALOG":
        max_user_id = str(dialog.get("max_user_id") or "").strip()
        peers = tuple(
            item
            for item in (dialog.get("participants") or ())
            if isinstance(item, Mapping) and not item.get("is_me")
        )
        if max_user_id:
            chat_id = max_user_id
        elif len(peers) == 1 and str(peers[0].get("user_id") or "").strip():
            chat_id = str(peers[0]["user_id"]).strip()
        # The widget falls back to the room id when MAX has no separate peer id.
    manager = dialog.get("manager") if isinstance(dialog.get("manager"), Mapping) else {}
    return {
        "chat_id": chat_id,
        "phone": phone,
        "username": username,
        "manager_id": str(manager.get("id") or dialog.get("manager_id") or "").strip(),
    }


def build_draft_note_text(
    *,
    draft_text: str,
    brand: str,
    profile_id: str = "",
    idempotency_marker: str = "",
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
    meta_parts = [
        DRAFT_NOTE_MARKER,
        f"Бренд: {brand_value}",
        f"Время: {timestamp} (Europe/Moscow)",
    ]
    if str(profile_id or "").strip():
        meta_parts.append(f"Wappi profile_id: {str(profile_id).strip()}")
    if str(idempotency_marker or "").strip():
        meta_parts.append(f"Mango draft id: {str(idempotency_marker).strip()}")
    if route_line:
        meta_parts.append(route_line)
    if flags:
        meta_parts.append("Флаги безопасности: " + ", ".join(flags))
    if str(outgoing_visibility_note or "").strip():
        meta_parts.append(str(outgoing_visibility_note).strip())
    draft_value = str(draft_text or "").strip()
    meta_text = "\n".join(meta_parts).strip()
    text = "\n\n".join(part for part in (draft_value, meta_text) if part).strip()
    if len(text) <= MAX_DRAFT_NOTE_TEXT_CHARS:
        return text
    marker = "\n\n[Черновик усечён до 6000 символов]"
    suffix = f"\n\n{meta_text}" if meta_text else ""
    draft_limit = MAX_DRAFT_NOTE_TEXT_CHARS - len(marker) - len(suffix)
    if draft_limit <= 0:
        return text[: MAX_DRAFT_NOTE_TEXT_CHARS - len(marker)].rstrip() + marker
    return draft_value[:draft_limit].rstrip() + marker + suffix


def draft_note_idempotency_marker(*, profile_id: str, chat_id: str, message_id: str) -> str:
    parts = tuple(str(item or "").strip() for item in (profile_id, chat_id, message_id))
    if not all(parts):
        raise AmoWappiWriteBlocked("AMO draft-note idempotency requires profile_id, chat_id and message_id.")
    digest = hashlib.sha256("\t".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"mango-draft:{digest}"


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
