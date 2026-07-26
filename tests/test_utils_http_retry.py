from __future__ import annotations

from mango_mvp.utils.http_retry import (
    bounded_backoff_seconds,
    categorize_connection_error,
    categorize_curl_return_code,
    categorize_invalid_response,
    categorize_status_code,
    categorize_timeout,
    safe_error_summary,
    should_retry,
)


def test_categorize_status_code_rate_limited_is_retryable():
    category = categorize_status_code(429)
    assert category.category == "rate_limited"
    assert category.retryable is True


def test_categorize_status_code_server_error_is_retryable():
    for code in (500, 502, 503, 504):
        category = categorize_status_code(code)
        assert category.category == "server_error", code
        assert category.retryable is True


def test_categorize_status_code_auth_never_retries():
    for code in (401, 403):
        category = categorize_status_code(code)
        assert category.category == "auth"
        assert category.retryable is False


def test_categorize_status_code_not_found_never_retries():
    category = categorize_status_code(404)
    assert category.category == "not_found"
    assert category.retryable is False


def test_categorize_status_code_generic_client_error_never_retries():
    category = categorize_status_code(400)
    assert category.category == "client_error"
    assert category.retryable is False


def test_categorize_timeout_and_connection_error_are_retryable():
    assert categorize_timeout().retryable is True
    assert categorize_timeout().category == "timeout"
    assert categorize_connection_error().retryable is True
    assert categorize_connection_error().category == "connection_error"


def test_categorize_invalid_response_never_retries():
    category = categorize_invalid_response()
    assert category.retryable is False
    assert category.category == "invalid_response"


def test_categorize_curl_return_code_transient_vs_structural():
    assert categorize_curl_return_code(28).retryable is True  # connect timeout
    assert categorize_curl_return_code(28).category == "connection_error"
    assert categorize_curl_return_code(99).retryable is False
    assert categorize_curl_return_code(99).category == "transport_error"


def test_should_retry_respects_max_attempts_bound():
    category = categorize_status_code(500)
    assert should_retry(category, attempt=1, max_attempts=4) is True
    assert should_retry(category, attempt=3, max_attempts=4) is True
    assert should_retry(category, attempt=4, max_attempts=4) is False


def test_should_retry_never_true_for_non_retryable_category_regardless_of_budget():
    category = categorize_status_code(401)
    assert should_retry(category, attempt=1, max_attempts=100) is False


def test_bounded_backoff_seconds_is_monotonic_and_capped():
    values = [bounded_backoff_seconds(attempt, base=1.0, cap=5.0) for attempt in range(1, 10)]
    assert values == sorted(values)
    assert max(values) == 5.0
    assert bounded_backoff_seconds(0, base=1.0, cap=5.0) == 1.0


def test_safe_error_summary_never_includes_url_or_body():
    category = categorize_status_code(500)
    summary = safe_error_summary(category, source="tallanto_api")
    assert "https://" not in summary
    assert "token" not in summary.casefold()
    assert "status=500" in summary
    assert "category=server_error" in summary
    assert "source=tallanto_api" in summary
