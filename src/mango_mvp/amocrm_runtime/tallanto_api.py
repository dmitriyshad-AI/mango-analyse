from __future__ import annotations

import json
import socket
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request

from mango_mvp.amocrm_runtime.config import get_settings
from mango_mvp.utils.phone import normalize_phone


settings = get_settings()


class TallantoApiError(ValueError):
    def __init__(self, message: str, *, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class TallantoApiConfig:
    base_url: str
    api_token: str
    rest_path: str = "/service/api/rest.php"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_base_url(raw_value: str) -> str:
    candidate = str(raw_value or "").strip()
    if not candidate:
        raise TallantoApiError("Tallanto base URL is required.", status_code=503)
    if "://" not in candidate:
        candidate = f"https://{candidate}"
    parsed = url_parse.urlparse(candidate)
    if not parsed.netloc and parsed.path:
        parsed = url_parse.urlparse(f"https://{parsed.path}")
    if not parsed.netloc:
        raise TallantoApiError("Tallanto base URL is invalid.", status_code=503)
    scheme = parsed.scheme or "https"
    return f"{scheme}://{parsed.netloc}"


def build_tallanto_api_config() -> TallantoApiConfig:
    base_url = _normalize_base_url(str(settings.crm_tallanto_base_url or "").strip())
    api_token = str(settings.crm_tallanto_api_token or "").strip()
    if not api_token:
        raise TallantoApiError("CRM_TALLANTO_API_TOKEN is not configured.", status_code=503)
    rest_path = str(settings.crm_tallanto_student_path or "").strip() or "/service/api/rest.php"
    if "{student_id}" in rest_path:
        rest_path = "/service/api/rest.php"
    return TallantoApiConfig(base_url=base_url, api_token=api_token, rest_path=rest_path)


def _http_json_request(
    *,
    method: str,
    url: str,
    headers: Optional[dict[str, str]] = None,
    form_items: Optional[list[tuple[str, str]]] = None,
    timeout_seconds: int = 25,
) -> dict[str, Any]:
    payload = None
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    if form_items is not None:
        payload = url_parse.urlencode(form_items, doseq=True).encode("utf-8")
        request_headers["Content-Type"] = "application/x-www-form-urlencoded"

    request = url_request.Request(
        url,
        data=payload,
        headers=request_headers,
        method=method.upper(),
    )
    attempts = 4
    retry_delay_seconds = 2.0
    for attempt in range(1, attempts + 1):
        try:
            with url_request.urlopen(request, timeout=timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                if not raw.strip():
                    return {}
                decoded = json.loads(raw)
                if isinstance(decoded, dict):
                    return decoded
                return {"data": decoded}
        except url_error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            should_retry = exc.code in {429, 500, 502, 503, 504}
            if should_retry and attempt < attempts:
                time.sleep(retry_delay_seconds * attempt)
                continue
            raise TallantoApiError(
                f"HTTP {exc.code} from Tallanto: {details or exc.reason}",
                status_code=502,
            ) from exc
        except (url_error.URLError, TimeoutError, socket.timeout, ssl.SSLError) as exc:
            if attempt >= attempts:
                reason = getattr(exc, "reason", exc)
                raise TallantoApiError(
                    f"Failed to reach Tallanto: {reason}",
                    status_code=502,
                ) from exc
            time.sleep(retry_delay_seconds * attempt)
        except json.JSONDecodeError as exc:
            raise TallantoApiError(
                f"Invalid JSON response from Tallanto endpoint {url}.",
                status_code=502,
            ) from exc


def _append_query_items(url: str, query_items: Iterable[tuple[str, str]]) -> str:
    encoded_query = url_parse.urlencode(list(query_items), doseq=True)
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{encoded_query}"


def _build_url(base_url: str, path: str) -> str:
    normalized_base = base_url.rstrip("/") + "/"
    normalized_path = path.lstrip("/")
    return url_parse.urljoin(normalized_base, normalized_path)


def _dedupe_dicts(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        record_id = str(record.get("id") or "").strip()
        signature = record_id or json.dumps(record, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        result.append(record)
    return result


def _extract_record_list(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("entry_list", "records", "result", "data"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return [item for item in value.values() if isinstance(item, dict)]

    if any(key in payload for key in ("id", "phone_mobile", "phone_work", "first_name", "last_name", "contact_id")):
        return [payload]

    nested_records = [item for item in payload.values() if isinstance(item, dict) and ("id" in item or "name" in item)]
    if nested_records:
        return nested_records
    return []


def _build_phone_candidates(value: str) -> list[str]:
    normalized_value = str(value or "").strip()
    digits = "".join(char for char in normalized_value if char.isdigit())
    variants = [
        normalized_value,
        normalized_value.replace(" ", ""),
        digits,
        normalize_phone(normalized_value) or "",
        normalize_phone(digits) or "",
        f"8{digits[1:]}" if len(digits) == 11 and digits.startswith("7") else "",
        f"7{digits}" if len(digits) == 10 else "",
        f"+7{digits}" if len(digits) == 10 else "",
    ]
    result: list[str] = []
    seen: set[str] = set()
    for item in variants:
        candidate = str(item or "").strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        result.append(candidate)
    return result


def _is_not_found_error(exc: Exception) -> bool:
    message = str(exc).casefold()
    return "entry does not exist" in message or "not find by id" in message


class TallantoApiClient:
    CONTACT_PHONE_FIELDS = ("phone_mobile", "phone_work", "phone_home", "phone_other", "phone")

    def __init__(self, config: TallantoApiConfig):
        self.config = TallantoApiConfig(
            base_url=_normalize_base_url(config.base_url),
            api_token=str(config.api_token or "").strip(),
            rest_path=str(config.rest_path or "").strip() or "/service/api/rest.php",
        )
        if not self.config.api_token:
            raise TallantoApiError("Tallanto API token is required.", status_code=503)

    @property
    def endpoint_url(self) -> str:
        return _build_url(self.config.base_url, self.config.rest_path)

    def _headers(self, *, authenticated: bool) -> dict[str, str]:
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        if authenticated:
            headers["X-Auth-Token"] = self.config.api_token
        return headers

    @staticmethod
    def _normalize_select_fields(select_fields: Iterable[str] | None) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for field in select_fields or ():
            candidate = str(field or "").strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            result.append(candidate)
        return result

    def request(
        self,
        *,
        method_name: str,
        module: Optional[str] = None,
        http_method: str = "GET",
        query_items: Optional[list[tuple[str, str]]] = None,
        form_items: Optional[list[tuple[str, str]]] = None,
        select_fields: Iterable[str] | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        final_query_items: list[tuple[str, str]] = []
        if str(method_name).strip():
            final_query_items.append(("method", method_name))
        if module:
            final_query_items.append(("module", module))
        if query_items:
            final_query_items.extend(query_items)
        for field_name in self._normalize_select_fields(select_fields):
            final_query_items.append(("select_fields[]", field_name))
        request_url = _append_query_items(self.endpoint_url, final_query_items)
        return _http_json_request(
            method=http_method,
            url=request_url,
            headers=self._headers(authenticated=authenticated),
            form_items=form_items,
        )

    def list_possible_methods(self) -> dict[str, Any]:
        return self.request(method_name="", authenticated=False)

    def list_possible_modules(self) -> dict[str, Any]:
        return self.request(method_name="list_possible_modules")

    def list_possible_fields(self, module: str) -> dict[str, Any]:
        return self.request(method_name="list_possible_fields", module=module)

    def list_possible_fields_doc(self, module: str) -> str:
        request_url = _append_query_items(
            self.endpoint_url,
            [("method", "list_possible_fields_doc"), ("module", module)],
        )
        request = url_request.Request(
            request_url,
            headers=self._headers(authenticated=True),
            method="GET",
        )
        try:
            with url_request.urlopen(request, timeout=25) as response:
                return response.read().decode("utf-8")
        except url_error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise TallantoApiError(
                f"HTTP {exc.code} from Tallanto: {details or exc.reason}",
                status_code=502,
            ) from exc
        except url_error.URLError as exc:
            raise TallantoApiError(
                f"Failed to reach Tallanto: {exc.reason}",
                status_code=502,
            ) from exc

    def list_enum_values(self, options: Iterable[str]) -> dict[str, Any]:
        query_items = [("options[]", str(option).strip()) for option in options if str(option).strip()]
        return self.request(method_name="list_enum_values", query_items=query_items)

    def get_entry_by_id(
        self,
        *,
        module: str,
        entry_id: str,
        select_fields: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        return self.request(
            method_name="get_entry_by_id",
            module=module,
            query_items=[("id", str(entry_id).strip())],
            select_fields=select_fields,
        )

    def get_entry_by_fields(
        self,
        *,
        module: str,
        field_values: dict[str, Any],
        select_fields: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        query_items = [
            (f"fields_values[{field_name}]", str(field_value))
            for field_name, field_value in field_values.items()
            if field_value is not None and str(field_name).strip()
        ]
        return self.request(
            method_name="get_entry_by_fields",
            module=module,
            query_items=query_items,
            select_fields=select_fields,
        )

    def get_entry_list(
        self,
        *,
        module: str,
        select_fields: Iterable[str] | None = None,
        field_values: Optional[dict[str, Any]] = None,
        query: Optional[str] = None,
        order_by: Optional[str] = None,
        offset: int = 0,
    ) -> dict[str, Any]:
        form_items: list[tuple[str, str]] = [("offset", str(max(0, int(offset))))]
        if query:
            form_items.append(("query", query))
        if order_by:
            form_items.append(("order_by", order_by))
        for field_name, field_value in (field_values or {}).items():
            if field_value is None or not str(field_name).strip():
                continue
            form_items.append((f"fields_values[{field_name}]", str(field_value)))
        return self.request(
            method_name="get_entry_list",
            module=module,
            http_method="POST",
            form_items=form_items,
            select_fields=select_fields,
        )

    def iter_entry_list(
        self,
        *,
        module: str,
        select_fields: Iterable[str] | None = None,
        field_values: Optional[dict[str, Any]] = None,
        query: Optional[str] = None,
        order_by: Optional[str] = None,
        max_records: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        offset = 0
        while True:
            payload = self.get_entry_list(
                module=module,
                select_fields=select_fields,
                field_values=field_values,
                query=query,
                order_by=order_by,
                offset=offset,
            )
            entry_list = payload.get("entry_list")
            if not isinstance(entry_list, list) or not entry_list:
                break
            for entry in entry_list:
                if not isinstance(entry, dict):
                    continue
                collected.append(entry)
                if max_records is not None and len(collected) >= max_records:
                    return collected[:max_records]
            next_offset = payload.get("next_offset")
            if next_offset in (None, "", offset):
                break
            offset = int(next_offset)
        return collected

    def search_contacts_by_phone(
        self,
        phone: str,
        *,
        select_fields: Iterable[str] | None = None,
        max_records: int = 20,
    ) -> list[dict[str, Any]]:
        candidates = _build_phone_candidates(phone)
        records: list[dict[str, Any]] = []
        first_error: TallantoApiError | None = None
        for field_name in self.CONTACT_PHONE_FIELDS:
            for candidate in candidates:
                try:
                    payload = self.get_entry_by_fields(
                        module="Contact",
                        field_values={field_name: candidate},
                        select_fields=select_fields,
                    )
                except TallantoApiError as exc:
                    if _is_not_found_error(exc):
                        continue
                    if first_error is None:
                        first_error = exc
                    if records:
                        continue
                    continue
                records.extend(_extract_record_list(payload))
                if len(records) >= max_records:
                    return _dedupe_dicts(records)[:max_records]
        deduped = _dedupe_dicts(records)[:max_records]
        if deduped:
            return deduped
        if first_error is not None:
            raise first_error
        return deduped

    def contact_by_id(
        self,
        contact_id: str,
        *,
        select_fields: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        payload = self.get_entry_by_id(
            module="Contact",
            entry_id=str(contact_id).strip(),
            select_fields=select_fields,
        )
        records = _extract_record_list(payload)
        return records[0] if records else {}

    def opportunities_by_contact(
        self,
        contact_id: str,
        *,
        max_records: int = 100,
    ) -> list[dict[str, Any]]:
        return self.iter_entry_list(
            module="Opportunity",
            field_values={"contact_id": contact_id},
            max_records=max_records,
        )

    def requests_by_contact(
        self,
        contact_id: str,
        *,
        max_records: int = 100,
    ) -> list[dict[str, Any]]:
        return self.iter_entry_list(
            module="Request",
            field_values={"contact_id": contact_id},
            max_records=max_records,
        )

    def finances_by_contact(
        self,
        contact_id: str,
        *,
        max_records: int = 100,
    ) -> list[dict[str, Any]]:
        return self.iter_entry_list(
            module="most_finances",
            field_values={"contact_id": contact_id},
            max_records=max_records,
        )

    def course_relations_by_contact(
        self,
        contact_id: str,
        *,
        max_records: int = 100,
    ) -> list[dict[str, Any]]:
        return self.iter_entry_list(
            module="CoursesContactsRelationship",
            field_values={"contact_id": contact_id},
            max_records=max_records,
        )

    def class_relations_by_contact(
        self,
        contact_id: str,
        *,
        max_records: int = 100,
    ) -> list[dict[str, Any]]:
        return self.iter_entry_list(
            module="ClassContactsRelationship",
            field_values={"contact_id": contact_id},
            max_records=max_records,
        )

    def build_contact_context(
        self,
        phone: str,
        *,
        max_contacts: int = 10,
        max_related_records: int = 100,
    ) -> dict[str, Any]:
        contacts = self.search_contacts_by_phone(phone, max_records=max_contacts)
        contexts: list[dict[str, Any]] = []
        for contact in contacts:
            contact_id = str(contact.get("id") or "").strip()
            if not contact_id:
                continue
            contexts.append(
                {
                    "contact": contact,
                    "opportunities": self.opportunities_by_contact(contact_id, max_records=max_related_records),
                    "requests": self.requests_by_contact(contact_id, max_records=max_related_records),
                    "finances": self.finances_by_contact(contact_id, max_records=max_related_records),
                    "course_relations": self.course_relations_by_contact(contact_id, max_records=max_related_records),
                    "class_relations": self.class_relations_by_contact(contact_id, max_records=max_related_records),
                }
            )
        return {
            "generated_at": _iso_now(),
            "base_url": self.config.base_url,
            "phone": normalize_phone(phone) or str(phone).strip(),
            "contacts_found": len(contexts),
            "contexts": contexts,
        }

    def build_contact_context_by_contact_id(
        self,
        contact_id: str,
        *,
        max_related_records: int = 100,
    ) -> dict[str, Any]:
        contact = self.contact_by_id(contact_id)
        if not contact:
            return {
                "generated_at": _iso_now(),
                "base_url": self.config.base_url,
                "contact_id": str(contact_id).strip(),
                "contacts_found": 0,
                "contexts": [],
            }
        resolved_contact_id = str(contact.get("id") or contact_id).strip()
        return {
            "generated_at": _iso_now(),
            "base_url": self.config.base_url,
            "contact_id": resolved_contact_id,
            "contacts_found": 1,
            "contexts": [
                {
                    "contact": contact,
                    "opportunities": self.opportunities_by_contact(resolved_contact_id, max_records=max_related_records),
                    "requests": self.requests_by_contact(resolved_contact_id, max_records=max_related_records),
                    "finances": self.finances_by_contact(resolved_contact_id, max_records=max_related_records),
                    "course_relations": self.course_relations_by_contact(resolved_contact_id, max_records=max_related_records),
                    "class_relations": self.class_relations_by_contact(resolved_contact_id, max_records=max_related_records),
                }
            ],
        }

    def healthcheck(self) -> dict[str, Any]:
        return {
            "base_url": self.config.base_url,
            "rest_path": self.config.rest_path,
            "generated_at": _iso_now(),
            "modules": self.list_possible_modules(),
        }


__all__ = [
    "TallantoApiClient",
    "TallantoApiConfig",
    "TallantoApiError",
    "build_tallanto_api_config",
]
