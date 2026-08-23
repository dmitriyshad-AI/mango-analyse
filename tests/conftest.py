from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Mapping, Sequence

from mango_mvp.productization.mango_office_client import DEFAULT_STATS_FIELDS
from mango_mvp.customer_timeline.calls_two_processes import (
    build_mango_official_list_proof,
)


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
    window_start = datetime.fromisoformat(str(rolling_since))
    window_end = datetime.fromisoformat(str(result["until"]))
    split = window_start + (window_end - window_start) / 3
    verification_chunks = [
        {
            "since": window_start.isoformat(),
            "until": split.isoformat(),
            "result_complete": True,
            "rows": len(canonical_keys),
        },
        {
            "since": split.isoformat(),
            "until": window_end.isoformat(),
            "result_complete": True,
            "rows": 0,
        },
    ]
    pass_body = {
        "rolling_since": rolling_since,
        "until": result["until"],
        "requests": len(chunks),
        "raw_rows": len(canonical_keys),
        "chunks": chunks,
        "partition_sha256": _digest(
            [{"since": item["since"], "until": item["until"]} for item in chunks]
        ),
        "recordable_unique_rows": len(canonical_keys),
        "without_recording_rows": 0,
        "proven_duplicate_rows": 0,
        "quarantined_rows": 0,
        "error_rows": 0,
        "unexplained_rows": 0,
        "raw_balance_ok": True,
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
        "normalized_unique_count_equal": True,
        "call_keys_equal": True,
        "call_keys_sha256_equal": True,
        "calls_by_moscow_day_equal": True,
        "calls_by_moscow_day_sha256_equal": True,
        "event_digest_sha256_equal": True,
        "primary_raw_balance_ok": True,
        "verification_raw_balance_ok": True,
        "partition_sha256_different": True,
        "official_list_equal": True,
    }
    official_pages = [
        *(
            {
                "offset": 0,
                "rows": len(canonical_keys),
                "entry_ids": canonical_keys,
                "entry_ids_sha256": _digest(canonical_keys),
                "buckets": [
                    {
                        "period": None,
                        "declared_total_calls_count": len(canonical_keys),
                        "rows": len(canonical_keys),
                        "entry_ids": canonical_keys,
                        "entry_ids_sha256": _digest(canonical_keys),
                    }
                ],
                "status": "complete",
            }
            for _ in [None]
            if canonical_keys
        ),
        {
            "offset": len(canonical_keys),
            "rows": 0,
            "entry_ids": [],
            "entry_ids_sha256": _digest([]),
            "buckets": [],
            "status": "complete",
        },
    ]
    official_list = build_mango_official_list_proof(
        call_ids=canonical_keys,
        page_receipts=official_pages,
        since=window_start,
        until=window_end,
    )
    result["covered_intervals"] = [
        *({**interval, "authority_pass": 1} for interval in rolling),
        *(
            {**interval, "scope": "rolling_authority", "authority_pass": 2}
            for interval in verification_chunks
        ),
        *auxiliary,
    ]
    result["requests"] = len(result["covered_intervals"])
    result["enumeration_consistency_ok"] = True
    dual_proof = {
        "schema_version": "mango_exact_dual_enumeration_v3",
        "normalization_version": "mango_rows_call_day_v2",
        "tenant_id": "foton",
        "base_url": "https://app.mango-office.ru",
        "fields_sha256": _digest(DEFAULT_STATS_FIELDS),
        "rolling_since": rolling_since,
        "until": result["until"],
        "proof_run_id": "synthetic-proof-run-v1",
        "observed_at": "2026-08-12T00:00:00+00:00",
        "passes_required": 2,
        "passes_completed": 2,
        "passes": [
            {"pass_id": "primary", **pass_body},
            {
                "pass_id": "verification",
                **pass_body,
                "requests": len(verification_chunks),
                "chunks": verification_chunks,
                "partition_sha256": _digest(
                    [
                        {"since": item["since"], "until": item["until"]}
                        for item in verification_chunks
                    ]
                ),
            },
        ],
        "official_list": official_list,
        "comparison": comparison,
        "enumeration_consistency_ok": True,
        "mismatch_reason": "",
    }
    dual_proof["proof_sha256"] = _digest(dual_proof)
    result["dual_enumeration"] = dual_proof
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


def ready_capture_proof(
    source: Mapping[str, object],
    *,
    zero_by_day: Mapping[str, int],
) -> tuple[dict[str, object], str]:
    """Build the exact ready-v3 capture projection for strict fixtures."""

    intervals = [dict(item) for item in source["covered_intervals"]]
    dual = source["dual_enumeration"]
    primary = dual["passes"][0]
    authoritative_rows = sum(
        int(item.get("rows") or 0)
        for item in intervals
        if item.get("scope") == "rolling_authority"
    )
    auxiliary_rows = sum(
        int(item.get("rows") or 0)
        for item in intervals
        if item.get("scope") == "recovery_auxiliary"
    )
    projection = {
        "mango_enumeration_complete": True,
        "mango_enumeration_source": {
            key: source.get(key)
            for key in (
                "mode",
                "since",
                "rolling_since",
                "until",
                "cursor",
                "pages",
                "pagination",
                "requests",
                "catch_up",
                "enumeration_consistency_ok",
                "dual_enumeration",
            )
        }
        | {
            "covered_intervals": [
                {
                    key: item.get(key)
                    for key in (
                        "since",
                        "until",
                        "result_complete",
                        "rows",
                        "scope",
                        "authority_pass",
                    )
                }
                for item in intervals
            ]
        },
        "call_keys": list(primary["call_keys"]),
        "calls_by_moscow_day": dict(primary["calls_by_moscow_day"]),
        "independent_zero_enumerations_by_day": dict(zero_by_day),
        "api_requests": source["requests"],
        "api_rows_total": authoritative_rows + auxiliary_rows,
        "api_authoritative_rows_total": authoritative_rows,
        "api_auxiliary_rows_total": auxiliary_rows,
        "api_events_total": len(primary["call_keys"]),
    }
    return projection, _digest(projection)
