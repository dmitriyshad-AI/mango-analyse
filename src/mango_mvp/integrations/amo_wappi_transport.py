from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping
from urllib import parse as url_parse


JsonTransport = Callable[..., Mapping[str, Any]]


class TransportDenied(RuntimeError):
    pass


_AMO_LEAD_RE = re.compile(r"^/api/v4/leads/\d+$")
_AMO_LEAD_NOTES_RE = re.compile(r"^/api/v4/leads/\d+/notes$")
_AMO_CONTACT_RE = re.compile(r"^/api/v4/contacts/\d+$")


@dataclass(frozen=True)
class SafeTransportPolicy:
    allowed_hosts: frozenset[str] = field(default_factory=lambda: frozenset({"wappi.pro", "educent.amocrm.ru"}))

    def assert_allowed(self, *, method: str, url: str) -> None:
        parsed = url_parse.urlparse(str(url or ""))
        host = str(parsed.netloc or "").casefold()
        if host not in self.allowed_hosts:
            raise TransportDenied(f"HTTP denied: host is not allowlisted: {host}")
        normalized_method = str(method or "").upper()
        path = parsed.path or "/"
        query = url_parse.parse_qs(parsed.query, keep_blank_values=True)
        if host == "wappi.pro":
            self._assert_wappi_allowed(method=normalized_method, path=path, query=query)
            return
        self._assert_amo_allowed(method=normalized_method, path=path)

    def _assert_wappi_allowed(self, *, method: str, path: str, query: Mapping[str, list[str]]) -> None:
        if method != "GET":
            raise TransportDenied(f"Wappi HTTP denied: method {method} is not read-only.")
        if path in {"/tapi/profile/all/get", "/maxapi/profile/all/get", "/tapi/sync/chats/get"}:
            return
        if path == "/tapi/sync/messages/get":
            mark_values = [str(item).casefold() for item in query.get("mark_all", []) if str(item).strip()]
            if any(value not in {"0", "false", "no", "off"} for value in mark_values):
                raise TransportDenied("Wappi HTTP denied: mark_all must be false or omitted.")
            return
        raise TransportDenied(f"Wappi HTTP denied: path is not allowlisted: {path}")

    def _assert_amo_allowed(self, *, method: str, path: str) -> None:
        if method == "GET" and (
            path == "/api/v4/leads/pipelines"
            or path == "/api/v4/contacts"
            or _AMO_LEAD_RE.match(path)
            or _AMO_CONTACT_RE.match(path)
        ):
            return
        if method == "POST" and _AMO_LEAD_NOTES_RE.match(path):
            return
        raise TransportDenied(f"AMO HTTP denied: {method} {path} is not allowlisted.")


class DefaultDenyTransport:
    def __init__(self, inner: JsonTransport, *, policy: SafeTransportPolicy | None = None) -> None:
        self.inner = inner
        self.policy = policy or SafeTransportPolicy()

    def __call__(self, **kwargs: Any) -> Mapping[str, Any]:
        self.policy.assert_allowed(method=str(kwargs.get("method") or ""), url=str(kwargs.get("url") or ""))
        return self.inner(**kwargs)

