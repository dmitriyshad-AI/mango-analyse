from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_customer_memory_shadow import (
    infer_shadow_brand,
    require_staging_db_path,
    shadow_manual_review_flags,
    shadow_safety_violations,
    summarize_shadow_rows,
)


def test_infer_shadow_brand_prefers_a2v3_brand() -> None:
    brand, source = infer_shadow_brand(
        "unpk",
        [{"relevance_tags": ["bot_safe", "structured", "foton"]}],
    )

    assert (brand, source) == ("unpk", "a2v3_customer_brand")


def test_infer_shadow_brand_uses_single_bot_context_brand_when_customer_unknown() -> None:
    brand, source = infer_shadow_brand(
        "unknown",
        [{"relevance_tags": ["bot_safe", "structured", "foton"]}],
    )

    assert (brand, source) == ("foton", "single_bot_context_brand")


def test_infer_shadow_brand_keeps_ambiguous_bot_context_unresolved() -> None:
    brand, source = infer_shadow_brand(
        "",
        [
            {"relevance_tags": ["bot_safe", "structured", "foton"]},
            {"relevance_tags": ["bot_safe", "structured", "unpk"]},
        ],
    )

    assert (brand, source) == ("unknown", "unresolved")


def test_summarize_shadow_rows_counts_safety_findings() -> None:
    summary = summarize_shadow_rows(
        [
            {
                "active_brand": "foton",
                "brand_source": "a2v3_customer_brand",
                "shadow_enabled": True,
                "shadow_found": True,
                "route_text_shadow_only": True,
                "prompt_pii_findings": [],
                "prompt_has_service_id": False,
                "shadow_warnings": [],
            },
            {
                "active_brand": "unknown",
                "brand_source": "unresolved",
                "shadow_enabled": True,
                "shadow_found": False,
                    "route_text_shadow_only": True,
                    "prompt_pii_findings": ["email"],
                    "prompt_has_service_id": True,
                    "prompt_text": "Безопасные bot_context-фрагменты:",
                    "shadow_warnings": ["active_brand_not_supported"],
                },
            ]
        )

    assert summary["total_customers"] == 2
    assert summary["enabled"] == 2
    assert summary["found"] == 1
    assert summary["shadow_only"] == 2
    assert summary["prompt_pii_hits"] == 1
    assert summary["prompt_service_id_hits"] == 1
    assert summary["by_brand"] == {"foton": 1, "unknown": 1}
    assert summary["warnings"] == {"active_brand_not_supported": 1}
    assert summary["safety_violations_total"] == 3
    assert summary["safety_violations"] == {
        "empty_shadow_contains_memory_items": 1,
        "prompt_pii": 1,
        "prompt_service_id": 1,
    }


def test_shadow_safety_violations_block_cross_brand_and_raw_ids() -> None:
    reasons = shadow_safety_violations(
        {
            "active_brand": "foton",
            "shadow_found": True,
            "prompt_text": "Бренд: Фотон. Бренд: УНПК. customer:abc record_json={}",
            "prompt_pii_findings": [],
            "prompt_has_service_id": False,
        }
    )

    assert set(reasons) == {
        "prompt_cross_brand",
        "prompt_brand_mismatch",
        "prompt_debug_or_raw_id",
        "prompt_service_id",
    }


def test_shadow_manual_review_flags_mark_temporal_memory_without_fail() -> None:
    row = {
        "active_brand": "foton",
        "shadow_found": True,
        "prompt_text": "Бренд: Фотон. Интерес: курсы на 2026 август.",
        "prompt_pii_findings": [],
        "prompt_has_service_id": False,
    }

    assert shadow_safety_violations(row) == ()
    assert shadow_manual_review_flags(row) == ("temporal_marker_in_memory",)


def test_require_staging_db_path_rejects_non_staging_paths() -> None:
    with pytest.raises(ValueError):
        require_staging_db_path(Path("/tmp/customer_timeline.sqlite"))

    require_staging_db_path(
        Path("/Users/dmitrijfabarisov/Projects/Mango_email_pipeline_restore/.codex_local/staging/customer_timeline_staging.sqlite")
    )
