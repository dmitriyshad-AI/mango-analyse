from __future__ import annotations

import hashlib
import json
import csv
import time
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from mango_mvp.services.dialogue_contract import (
    DialogueContractError,
    PROVIDER_EVIDENCE_SOURCE,
    canonical_provider_phrases_sha256,
    provider_raw_response_record,
    provider_record,
)


DEFAULT_MANGO_BASE_URL = "https://app.mango-office.ru"
RECORDING_TRANSCRIPTS_PATH = "/vpbx/queries/recording_transcripts"
# The documented ceiling of one ``recording_transcripts`` request.
RECORDING_TRANSCRIPTS_MAX_IDS = 500
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
        stats_result_poll_attempts: int = 13,
        stats_result_poll_interval_sec: float = 5.0,
        sleeper: Optional[Callable[[float], None]] = None,
    ) -> None:
        if stats_result_poll_attempts < 1:
            raise ValueError("stats_result_poll_attempts must be positive")
        if stats_result_poll_interval_sec < 0:
            raise ValueError("stats_result_poll_interval_sec must be non-negative")
        self.credentials = credentials
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.timeout_sec = timeout_sec
        self.stats_result_poll_attempts = stats_result_poll_attempts
        self.stats_result_poll_interval_sec = stats_result_poll_interval_sec
        self.sleeper = sleeper or time.sleep

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
        response = self._post_response("/vpbx/stats/result", request_token)
        if response.status_code == 204:
            raise MangoOfficeStatsPending("stats/result is not ready")
        return _decode_json_response(response)

    def poll_call_history(
        self,
        since: datetime,
        until: datetime,
        fields: str = DEFAULT_STATS_FIELDS,
    ) -> Sequence[Mapping[str, Any]]:
        request_token = self.create_stats_request(since=since, until=until, fields=fields)
        if not isinstance(request_token, Mapping):
            raise MangoOfficeApiError("stats/request returned non-object response")
        for attempt in range(self.stats_result_poll_attempts):
            try:
                result = self.fetch_stats_result(request_token)
            except MangoOfficeStatsPending:
                if attempt + 1 >= self.stats_result_poll_attempts:
                    raise MangoOfficeApiError(
                        "stats/result did not become ready before the polling deadline"
                    ) from None
                self.sleeper(self.stats_result_poll_interval_sec)
                continue
            return extract_stats_rows(result)
        raise AssertionError("unreachable Mango stats polling state")

    @staticmethod
    def build_recording_transcripts_payload(
        recording_ids: Sequence[str],
    ) -> Mapping[str, Any]:
        """Pure request body for ``/vpbx/queries/recording_transcripts``.

        No network call: этап B only prepares and validates the contract.  The
        real availability of the method is checked by the M1 1→10 ladder.
        """
        if isinstance(recording_ids, (str, bytes)) or not isinstance(recording_ids, Sequence):
            raise MangoOfficeApiError("recording_transcripts needs a sequence of recording ids")
        if any(not isinstance(item, str) for item in recording_ids):
            raise MangoOfficeApiError("recording_transcripts recording ids must be strings")
        cleaned = [item.strip() for item in recording_ids]
        if not cleaned or any(not item for item in cleaned):
            raise MangoOfficeApiError("recording_transcripts needs non-empty recording ids")
        if len(set(cleaned)) != len(cleaned):
            raise MangoOfficeApiError("recording_transcripts recording ids are duplicated")
        if len(cleaned) > RECORDING_TRANSCRIPTS_MAX_IDS:
            raise MangoOfficeApiError(
                "recording_transcripts accepts at most "
                f"{RECORDING_TRANSCRIPTS_MAX_IDS} recording ids"
            )
        # Mango's current VPBX API PDF documents a JSON *string* containing
        # the array, not a native nested array.  Keep this odd wire contract in
        # one place so callers cannot sign a different body by accident.
        return {
            "recording_id": json.dumps(
                cleaned, ensure_ascii=False, separators=(",", ":")
            )
        }

    def fetch_recording_transcripts(
        self, recording_ids: Sequence[str]
    ) -> tuple[Any, str]:
        """Read one documented batch and preserve the exact response body.

        This method only performs transport.  A caller must still bind every
        returned record to its capture metadata with
        :meth:`parse_recording_transcripts_response` before storing evidence.
        """
        response = self._post_response(
            RECORDING_TRANSCRIPTS_PATH,
            self.build_recording_transcripts_payload(recording_ids),
        )
        raw_body = str(response.text)
        return _decode_json_response(response), raw_body

    @staticmethod
    def parse_recording_transcripts_response(
        payload: Any,
        *,
        source_call_id: str,
        raw_body: str,
        expected_recording_id: str,
    ) -> Mapping[str, Any]:
        """Strict offline reader of the official ``recording_transcripts`` answer.

        The documented envelope is ``result`` plus ``data``.  Mango's public
        one-record example uses an object, while batch responses may contain a
        list; the dialogue contract accepts only those two explicit forms.  The
        envelope grammar itself lives in the dialogue contract so that capture
        and the role guard can never drift apart.

        ``expected_recording_id`` is required and is used only to *select* the
        record: the answer is searched for exactly that id and nothing is
        reordered or picked by position.  The value written into the evidence is
        the id the body itself declares — a request parameter is not proof of
        what came back, so the two are compared rather than copied.

        The returned evidence is bound to one ``source_call_id`` and carries
        only that recording's canonical response.  The full batch body may
        contain other clients and must be stored once in owner-only capture
        storage under ``batch_response_sha256``; it must never be copied into
        every call row.  A ``raw_body`` that does not re-derive the same record
        is refused here.

        The response is NOT assumed to carry a left/right channel binding — the
        documented answer never promised one.  The dialogue contract derives the
        binding from the phrases themselves, so this method deliberately stores
        no ``channels`` field: a second copy of that fact is a second thing to
        forge.  Until the M1 1→10 ladder captures a real answer, the whole path
        stays fail-closed.
        """
        call_id = str(source_call_id or "").strip()
        if not call_id:
            raise MangoOfficeApiError("recording_transcripts evidence needs a source_call_id")
        expected = str(expected_recording_id or "").strip()
        if not expected:
            raise MangoOfficeApiError(
                "recording_transcripts evidence needs an expected recording_id"
            )
        raw_response = str(raw_body)
        try:
            record = provider_raw_response_record(raw_response, expected)
            if record != provider_record(payload, expected):
                raise DialogueContractError(
                    "raw response body does not carry the parsed response"
                )
        except DialogueContractError as exc:
            raise MangoOfficeApiError(f"recording_transcripts response is invalid: {exc}") from exc
        if record["recording_id"] != expected:
            raise MangoOfficeApiError(
                "recording_transcripts response describes another recording"
            )
        record_response = json.dumps(
            {"result": 1000, "data": record},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "provider": PROVIDER_EVIDENCE_SOURCE,
            "source_call_id": call_id,
            "recording_id": record["recording_id"],
            "phrases_sha256": canonical_provider_phrases_sha256(record["phrases"]),
            "raw_response": record_response,
            "raw_response_sha256": hashlib.sha256(
                record_response.encode("utf-8")
            ).hexdigest(),
            "batch_response_sha256": hashlib.sha256(
                raw_response.encode("utf-8")
            ).hexdigest(),
        }

    def post_command(self, path: str, payload: Mapping[str, Any]) -> Any:
        return _decode_json_response(self._post_response(path, payload))

    def _post_response(
        self, path: str, payload: Mapping[str, Any]
    ) -> HttpResponse:
        signed = self.build_signed_form(payload)
        session = self.session or _RequestsSession()
        response = session.post(
            f"{self.base_url}{path}",
            data=signed.form_data,
            timeout=self.timeout_sec,
        )
        if response.status_code >= 400:
            body_sha256 = hashlib.sha256(
                str(response.text or "").encode("utf-8")
            ).hexdigest()
            raise MangoOfficeApiError(
                "Mango Office API request failed: "
                f"status={response.status_code} path={path} body_sha256={body_sha256}"
            )
        return response


class MangoOfficeApiError(RuntimeError):
    pass


class MangoOfficeStatsPending(MangoOfficeApiError):
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
