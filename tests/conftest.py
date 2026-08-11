from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

from mango_mvp.productization.mango_office_client import DEFAULT_STATS_FIELDS


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def dual_strict_source(
    source: Mapping[str, object],
    *,
    call_keys: Sequence[str],
    calls_by_day: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Build deterministic synthetic dual-enumeration source evidence."""

    result = json.loads(json.dumps(source))
    rolling_since = result.get("rolling_since") or result["since"]
    result["rolling_since"] = rolling_since
    canonical_keys = sorted(set(call_keys))
    canonical_days = {
        key: sorted(set(values)) for key, values in sorted(calls_by_day.items())
    }
    raw_intervals = list(result.get("covered_intervals") or [])
    rolling = [
        dict(interval)
        for interval in raw_intervals
        if interval.get("scope", "rolling_authority")
        == "rolling_authority"
    ]
    auxiliary = [
        dict(interval)
        for interval in raw_intervals
        if interval.get("scope") == "recovery_auxiliary"
    ]
    if not rolling:
        rolling = [
            {
                "since": rolling_since,
                "until": result["until"],
                "result_complete": True,
                "scope": "rolling_authority",
            }
        ]
    for index, interval in enumerate(rolling):
        interval["scope"] = "rolling_authority"
        interval["result_complete"] = True
        interval["rows"] = len(canonical_keys) if index == 0 else 0
    chunks = [
        {
            "since": interval["since"],
            "until": interval["until"],
            "result_complete": True,
            "rows": interval["rows"],
        }
        for interval in rolling
    ]
    pass_body = {
        "rolling_since": rolling_since,
        "until": result["until"],
        "requests": len(chunks),
        "raw_rows": len(canonical_keys),
        "chunks": chunks,
        "call_key_multiset": canonical_keys,
        "call_key_multiset_sha256": _digest(canonical_keys),
        "raw_rows_sha256": _digest({"synthetic_rows": canonical_keys}),
        "call_keys": canonical_keys,
        "normalized_unique_count": len(canonical_keys),
        "call_keys_sha256": _digest(canonical_keys),
        "calls_by_moscow_day": canonical_days,
        "calls_by_moscow_day_sha256": _digest(canonical_days),
        "event_digest_sha256": _digest(
            {"call_keys": canonical_keys, "calls_by_day": canonical_days}
        ),
    }
    comparison = {
        "raw_rows_equal": True,
        "call_key_multiset_equal": True,
        "call_key_multiset_sha256_equal": True,
        "raw_rows_sha256_equal": True,
        "normalized_unique_count_equal": True,
        "call_keys_equal": True,
        "call_keys_sha256_equal": True,
        "calls_by_moscow_day_equal": True,
        "calls_by_moscow_day_sha256_equal": True,
        "event_digest_sha256_equal": True,
        "chunk_geometry_equal": True,
    }
    result["covered_intervals"] = [
        *({**interval, "authority_pass": 1} for interval in rolling),
        *({**interval, "authority_pass": 2} for interval in rolling),
        *auxiliary,
    ]
    result["requests"] = len(result["covered_intervals"])
    result["enumeration_consistency_ok"] = True
    result["dual_enumeration"] = {
        "schema_version": "mango_exact_dual_enumeration_v1",
        "normalization_version": "mango_rows_call_day_v1",
        "tenant_id": "foton",
        "base_url": "https://app.mango-office.ru",
        "fields_sha256": _digest(DEFAULT_STATS_FIELDS),
        "rolling_since": rolling_since,
        "until": result["until"],
        "passes_required": 2,
        "passes_completed": 2,
        "passes": [
            {"pass_id": "primary", **pass_body},
            {"pass_id": "verification", **pass_body},
        ],
        "comparison": comparison,
        "enumeration_consistency_ok": True,
        "mismatch_reason": "",
    }
    return result


def dualize_strict_enumeration(
    evidence: Mapping[str, object],
) -> dict[str, object]:
    result = json.loads(json.dumps(evidence))
    call_keys = list(result.get("call_keys") or [])
    calls_by_day = dict(result.get("calls_by_moscow_day") or {})
    source = dual_strict_source(
        result["mango_enumeration_source"],
        call_keys=call_keys,
        calls_by_day=calls_by_day,
    )
    auxiliary_rows = sum(
        int(interval.get("rows") or 0)
        for interval in source["covered_intervals"]
        if interval.get("scope") == "recovery_auxiliary"
    )
    result["mango_enumeration_source"] = source
    result["enumeration_consistency_ok"] = True
    result["api_requests"] = source["requests"]
    result["api_authoritative_rows_total"] = len(call_keys) * 2
    result["api_auxiliary_rows_total"] = auxiliary_rows
    result["api_rows_total"] = len(call_keys) * 2 + auxiliary_rows
    result["api_events_total"] = len(call_keys)
    return result
