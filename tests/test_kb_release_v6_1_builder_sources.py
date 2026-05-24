from __future__ import annotations

from pathlib import Path

from scripts.build_kb_release_v6_1_team_answers import (
    DEFAULT_SOURCE_OUT,
    apply_release_manifest,
    load_release_manifest,
    validate_source_overlay,
)
from scripts.build_kb_release_v3_from_claude_handoff import build_post_filter_registry


def test_v6_1_builder_uses_release_manifest_and_validates_yaml_sources() -> None:
    manifest = load_release_manifest(DEFAULT_SOURCE_OUT)

    validate_source_overlay(DEFAULT_SOURCE_OUT, manifest)
    assert manifest["source_mutation_policy"] == "read_only_yaml_sources_no_business_patches_in_python"
    assert "gold_answers_v3" in manifest["source_files"]


def test_v6_1_builder_has_no_business_patch_functions() -> None:
    source = Path("scripts/build_kb_release_v6_1_team_answers.py").read_text(encoding="utf-8")

    forbidden = [
        "def patch_sources",
        "def patch_foton_facts",
        "def patch_unpk_facts",
        "def patch_brand_rules",
        "def patch_bot_policy",
        "def patch_foton_installment_client_terms",
        "write_yaml(",
        "pricing[\"current_price\"]",
        "contacts[\"vk\"]",
    ]
    for marker in forbidden:
        assert marker not in source


def test_release_manifest_can_drive_control_numbers_without_business_literals_in_builder(monkeypatch) -> None:
    from scripts import build_kb_release_v3_from_claude_handoff as kb_builder

    monkeypatch.setattr(kb_builder, "CONTROL_NUMBERS", ("100", "200", "300"))
    monkeypatch.setattr(kb_builder, "BUILDER_VERSION", "before")
    monkeypatch.setattr(kb_builder, "FRESHNESS_CHECK_DATE", "before")
    manifest = {
        "control_numbers": {"remove": ["200"], "add": ["400"]},
        "builder_version": "test_builder",
        "freshness_check_date": "2026-05-24",
        "source_files": {},
    }

    apply_release_manifest(manifest)

    assert kb_builder.CONTROL_NUMBERS == ("100", "300", "400")
    assert kb_builder.BUILDER_VERSION == "test_builder"
    assert kb_builder.FRESHNESS_CHECK_DATE == "2026-05-24"


def test_post_filter_registry_keeps_installment_terms_brand_scoped() -> None:
    registry = build_post_filter_registry(
        {
            "brand_rules": {
                "forbidden_client_mentions": {
                    "when_active_brand_is_foton": {"blocked_terms": ["УНПК"]},
                    "when_active_brand_is_unpk": {"blocked_terms": ["Фотон", "Т-Банк", "Долями"]},
                },
                "forbidden_client_phrasings": ["Фотон и УНПК — наши две компании"],
            },
            "bot_policy": {"post_filter_draft_text": {"forbidden_in_any_brand": ["автоматический ответ"]}},
        }
    )

    assert "Долями" not in registry["global_phrases"]
    assert "Т-Банк" not in registry["global_phrases"]
    assert "Долями" in registry["phrases_by_active_brand"]["unpk"]
    assert "Т-Банк" in registry["phrases_by_active_brand"]["unpk"]
    assert "Долями" not in registry["phrases_by_active_brand"]["foton"]
    assert "global_phrases" in registry["matcher_fields"]
    assert "phrases_by_active_brand" in registry["matcher_fields"]
