from __future__ import annotations

"""Shared, PII-safe HTTP error categorization and bounded-retry policy.

D3: every read-only HTTP client in this codebase (Tallanto REST API,
AMO MCP connector) hand-rolled its own retry loop. This module gives them one
shared, testable policy instead of a second/third copy of the same logic:

- classify a failure into a small closed set of safe categories (never a raw
  URL, query string, token, or response body -- only status/HTTP class/
  timeout/rc), so any caller can put the category into a step report or
  proof without a PII/secret-leak review each time;
- decide whether a given category is retryable at all (429/5xx/timeout/
  transient connection errors only -- 401/403 and structurally invalid
  responses are never retryable, regardless of attempts remaining);
- compute a bounded backoff delay so every caller retries with the same
  capped, monotonically-bounded schedule instead of ad-hoc sleeps.

This is intentionally not an HTTP client: it has no I/O of its own. Existing
callers (mango_mvp.amocrm_runtime.tallanto_api, mango_mvp.existing_clients.
amo_step1_snapshot) keep doing their own transport; they only delegate the
"is this retryable, what do I call it safely" decision here.
"""

from dataclasses import dataclass
from typing import Optional


RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# curl-style transports: connect timeout, SSL connect error, recv error, network unreachable.
RETRYABLE_CURL_RETURN_CODES = frozenset({28, 35, 52, 56})


@dataclass(frozen=True)
class HttpErrorCategory:
    """A safe-to-log classification of one HTTP-layer failure.

    ``category`` is always one of a small closed vocabulary below -- never a
    raw exception message, URL, or response body -- so it is always safe to
    place directly into a step report/proof.
    """

    category: str
    retryable: bool
    status_code: Optional[int] = None
    rc: Optional[int] = None

    def to_json_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "retryable": self.retryable,
            "status_code": self.status_code,
            "rc": self.rc,
        }


def categorize_status_code(status_code: int) -> HttpErrorCategory:
    """Classify an HTTP response status code into a safe category.

    401/403 are always ``auth`` and never retryable, no matter how many
    attempts remain -- callers must fail loud/partial immediately instead of
    burning retry budget on a credential problem that will not fix itself.
    """
    code = int(status_code)
    if code == 429:
        return HttpErrorCategory("rate_limited", retryable=True, status_code=code)
    if code in {401, 403}:
        return HttpErrorCategory("auth", retryable=False, status_code=code)
    if code == 404:
        return HttpErrorCategory("not_found", retryable=False, status_code=code)
    if 500 <= code <= 599:
        return HttpErrorCategory("server_error", retryable=True, status_code=code)
    if 400 <= code <= 499:
        return HttpErrorCategory("client_error", retryable=False, status_code=code)
    return HttpErrorCategory("unknown_status", retryable=False, status_code=code)


def categorize_timeout() -> HttpErrorCategory:
    return HttpErrorCategory("timeout", retryable=True)


def categorize_connection_error() -> HttpErrorCategory:
    return HttpErrorCategory("connection_error", retryable=True)


def categorize_invalid_response() -> HttpErrorCategory:
    """Structurally invalid response (bad JSON, missing required fields).

    Never retryable: retrying an endpoint that returned malformed data will
    not usually produce valid data, and looping on it burns the retry budget
    that a genuinely transient failure needs.
    """
    return HttpErrorCategory("invalid_response", retryable=False)


def categorize_curl_return_code(rc: int) -> HttpErrorCategory:
    code = int(rc)
    if code in RETRYABLE_CURL_RETURN_CODES:
        return HttpErrorCategory("connection_error", retryable=True, rc=code)
    return HttpErrorCategory("transport_error", retryable=False, rc=code)


def should_retry(category: HttpErrorCategory, *, attempt: int, max_attempts: int) -> bool:
    """Whether attempt number ``attempt`` (1-based, the attempt that just
    failed) should be followed by another try. Bounded by both the
    category's own retryability and the caller's max_attempts -- a retryable
    category never retries forever.
    """
    if not category.retryable:
        return False
    return attempt < max(1, int(max_attempts))


def bounded_backoff_seconds(attempt: int, *, base: float = 1.0, cap: float = 20.0) -> float:
    """Monotonically increasing, capped backoff for attempt N (1-based)."""
    if attempt < 1:
        attempt = 1
    return min(cap, base * attempt)


def safe_error_summary(category: HttpErrorCategory, *, source: str) -> str:
    """A short, constant-shape message safe to store in any report/proof.

    Deliberately excludes the request URL, query string, auth headers, and
    response body -- only the source label (a fixed identifier the caller
    chooses, e.g. "tallanto_api"/"amo_mcp"), the safe category, and the
    numeric status/rc (never PII) ever appear.
    """
    parts = [f"source={source}", f"category={category.category}"]
    if category.status_code is not None:
        parts.append(f"status={category.status_code}")
    if category.rc is not None:
        parts.append(f"rc={category.rc}")
    return "http_error " + " ".join(parts)


__all__ = [
    "RETRYABLE_STATUS_CODES",
    "RETRYABLE_CURL_RETURN_CODES",
    "HttpErrorCategory",
    "categorize_status_code",
    "categorize_timeout",
    "categorize_connection_error",
    "categorize_invalid_response",
    "categorize_curl_return_code",
    "should_retry",
    "bounded_backoff_seconds",
    "safe_error_summary",
]
