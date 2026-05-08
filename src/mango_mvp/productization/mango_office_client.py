from __future__ import annotations

import hashlib
import json
import csv
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Any, Mapping, Optional, Protocol, Sequence


DEFAULT_MANGO_BASE_URL = "https://app.mango-office.ru"
DEFAULT_STATS_FIELDS = (
    "records,start,finish,answer,from_extension,from_number,"
    "to_extension,to_number,disconnect_reason,line_number,location,entry_id"
)
DEFAULT_STATS_FIELD_NAMES = tuple(field.strip() for field in DEFAULT_STATS_FIELDS.split(","))


class HttpResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> Any:
        ...


class HttpSession(Protocol):
    def post(
        self,
        url: str,
        data: Mapping[str, str],
        timeout: int,
    ) -> HttpResponse:
        ...


@dataclass(frozen=True)
class MangoOfficeCredentials:
    api_key: str
    api_salt: str

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("Mango Office API key must not be empty")
        if not self.api_salt.strip():
            raise ValueError("Mango Office API salt must not be empty")


@dataclass(frozen=True)
class MangoSignedForm:
    json_body: str
    sign: str
    form_data: Mapping[str, str]


class MangoOfficeClient:
    def __init__(
        self,
        credentials: MangoOfficeCredentials,
        base_url: str = DEFAULT_MANGO_BASE_URL,
        session: Optional[HttpSession] = None,
        timeout_sec: int = 30,
    ) -> None:
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.timeout_sec = timeout_sec

    def build_signed_form(self, payload: Mapping[str, Any]) -> MangoSignedForm:
        json_body = _json_dumps(payload)
        sign_source = f"{self.credentials.api_key}{json_body}{self.credentials.api_salt}"
        sign = hashlib.sha256(sign_source.encode("utf-8")).hexdigest()
        return MangoSignedForm(
            json_body=json_body,
            sign=sign,
            form_data={
                "vpbx_api_key": self.credentials.api_key,
                "sign": sign,
                "json": json_body,
            },
        )

    def create_stats_request(
        self,
        since: datetime,
        until: datetime,
        fields: str = DEFAULT_STATS_FIELDS,
    ) -> Any:
        payload = build_stats_request_payload(since=since, until=until, fields=fields)
        return self.post_command("/vpbx/stats/request", payload)

    def fetch_stats_result(self, request_token: Mapping[str, Any]) -> Any:
        return self.post_command("/vpbx/stats/result", request_token)

    def poll_call_history(
        self,
        since: datetime,
        until: datetime,
        fields: str = DEFAULT_STATS_FIELDS,
    ) -> Sequence[Mapping[str, Any]]:
        request_token = self.create_stats_request(since=since, until=until, fields=fields)
        if not isinstance(request_token, Mapping):
            raise MangoOfficeApiError("stats/request returned non-object response")
        result = self.fetch_stats_result(request_token)
        return extract_stats_rows(result)

    def post_command(self, path: str, payload: Mapping[str, Any]) -> Any:
        signed = self.build_signed_form(payload)
        session = self.session or _RequestsSession()
        response = session.post(
            f"{self.base_url}{path}",
            data=signed.form_data,
            timeout=self.timeout_sec,
        )
        if response.status_code >= 400:
            raise MangoOfficeApiError(
                f"Mango Office API HTTP {response.status_code}: {response.text[:500]}"
            )
        return _decode_json_response(response)


class MangoOfficeApiError(RuntimeError):
    pass


def build_stats_request_payload(
    since: datetime,
    until: datetime,
    fields: str = DEFAULT_STATS_FIELDS,
) -> Mapping[str, Any]:
    if since.tzinfo is None or since.utcoffset() is None:
        raise ValueError("since must be timezone-aware")
    if until.tzinfo is None or until.utcoffset() is None:
        raise ValueError("until must be timezone-aware")
    if until <= since:
        raise ValueError("until must be later than since")

    return {
        "date_from": str(int(since.timestamp())),
        "date_to": str(int(until.timestamp())),
        "from": {"extension": "", "number": ""},
        "to": {"extension": "", "number": ""},
        "fields": fields,
    }


def extract_stats_rows(result: Any) -> Sequence[Mapping[str, Any]]:
    if isinstance(result, str):
        return parse_stats_csv(result, DEFAULT_STATS_FIELD_NAMES)
    if isinstance(result, list):
        return tuple(_ensure_mapping_rows(result))
    if isinstance(result, Mapping):
        for key in ("data", "result", "records", "calls", "items"):
            value = result.get(key)
            if isinstance(value, list):
                return tuple(_ensure_mapping_rows(value))
        if _looks_like_single_call_row(result):
            return (dict(result),)
    raise MangoOfficeApiError("Could not locate call history rows in Mango stats/result response")


def parse_stats_csv(text: str, fields: Sequence[str]) -> Sequence[Mapping[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return ()

    rows = []
    reader = csv.reader(StringIO(stripped), delimiter=";")
    for row in reader:
        if not row:
            continue
        if len(row) != len(fields):
            raise MangoOfficeApiError(
                f"Mango stats/result CSV row has {len(row)} fields, expected {len(fields)}"
            )
        rows.append(dict(zip(fields, row)))
    return tuple(rows)


def _ensure_mapping_rows(rows: Sequence[Any]) -> Sequence[Mapping[str, Any]]:
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise MangoOfficeApiError("Mango stats/result contains a non-object row")
        result.append(dict(row))
    return tuple(result)


def _looks_like_single_call_row(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in ("entry_id", "call_id", "start", "finish", "records"))


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _decode_json_response(response: HttpResponse) -> Any:
    try:
        return response.json()
    except Exception as exc:
        try:
            return json.loads(response.text)
        except json.JSONDecodeError as json_exc:
            return response.text


class _RequestsSession:
    def __init__(self) -> None:
        import requests

        self._requests = requests

    def post(self, url: str, data: Mapping[str, str], timeout: int) -> HttpResponse:
        return self._requests.post(url, data=data, timeout=timeout)
