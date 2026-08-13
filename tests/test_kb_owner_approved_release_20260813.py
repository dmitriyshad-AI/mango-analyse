from __future__ import annotations

import json
import hashlib
from datetime import date
from pathlib import Path

from mango_mvp.channels.subscription_llm import SubscriptionDraftResult, apply_authoritative_output_gate
from mango_mvp.channels.subscription_llm_parts.direct_path import (
    ASSUMED_SCOPE_GUARD_ENV,
    LLM_RETRIEVE_ENV,
    RETRIEVER_MODEL_DRIVEN_ENV,
    _direct_path_context_fact_pack,
)
from mango_mvp.channels.telegram_pilot_context_builder import build_telegram_pilot_context_from_snapshot
from mango_mvp.knowledge_base.fact_registry import fact_runtime_time_ok
from mango_mvp.knowledge_base.price_axes_catalog import build_price_axes_catalog, select_price
from mango_mvp.knowledge_base.product_existence_axes_catalog import (
    build_product_existence_axes_catalog,
    verify_product_format_exists,
)
from scripts.run_kb_semantic_review import DEFAULT_RELEASE_DIR
from scripts.run_kb_semantic_review import run_kb_semantic_review
from mango_mvp.graphify_structural import list_structured_files


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "product_data/knowledge_base/kb_release_20260813_v6_8_owner_approved"
SNAPSHOT = RELEASE / "kb_release_v3_snapshot.json"


def facts() -> list[dict]:
    return list(json.loads(SNAPSHOT.read_text(encoding="utf-8"))["facts"])


def test_owner_release_has_closed_current_business_balance() -> None:
    records = facts()
    allowed = [fact for fact in records if fact.get("allowed_for_client_answer") is True]

    assert sum(str(fact.get("fact_key", "")).startswith("owner_schedule_2026_27.") for fact in records) == 178
    assert all(
        fact.get("freshness_check_date")
        or (fact.get("structured_value") or {}).get("freshness_check_date")
        for fact in allowed
    )
    assert not [
        fact
        for fact in allowed
        if str(fact.get("valid_from") or "") <= "2026-08-13"
        and not fact_runtime_time_ok(fact, today=date(2026, 8, 13))
    ]
    old_fragments = ("93 100", "114 000", "20-28 июня", "3-14 августа")
    assert not [fact for fact in allowed if any(value in str(fact.get("client_safe_text") or "") for value in old_fragments)]


def test_current_release_is_the_default_semantic_and_graphify_source() -> None:
    assert DEFAULT_RELEASE_DIR == Path("product_data/knowledge_base/kb_release_20260813_v6_8_owner_approved")
    paths = list_structured_files(ROOT)
    assert paths
    assert all("kb_release_20260813_v6_8_owner_approved" in str(path) for path in paths)


def test_generated_release_provenance_is_portable() -> None:
    generated_roots = (
        RELEASE,
        RELEASE.parent / f"{RELEASE.name}_sources",
    )
    leaked = [
        path
        for root in generated_roots
        for path in root.rglob("*")
        if path.is_file()
        and "/.codex_workers/" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert leaked == []


def test_default_semantic_review_is_independent_of_current_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    report = run_kb_semantic_review(DEFAULT_RELEASE_DIR, today=date(2026, 8, 13))
    assert report["semantic_pass"] is True
    assert report["facts_total"] == 806


def test_retired_2026_summer_sales_text_is_absent_from_runtime_surfaces() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    surfaces = "\n".join(
        (
            json.dumps(snapshot.get("bot_policy") or {}, ensure_ascii=False),
            (RELEASE / "bot_policy.yaml").read_text(encoding="utf-8"),
            "\n".join(
                str(fact.get("client_safe_text") or "")
                for fact in snapshot["facts"]
                if fact.get("allowed_for_client_answer") is True
            ),
        )
    )

    for retired in ("93 100", "98 000", "114 000", "120 000", "130 000", "20-28 июня", "3-14 августа"):
        assert retired not in surfaces
    for current in ("99 450", "89 100", "93 600", "99 750", "3-10 января 2027"):
        assert current in surfaces


def test_generic_camp_existence_comes_only_from_owner_2027_facts(monkeypatch) -> None:
    monkeypatch.setenv("MANGO_EVALUATION_DATE", "2026-08-13")
    entries = build_product_existence_axes_catalog(facts())["entries"]
    positives = [
        entry
        for entry in entries
        if entry["product_family"] == "camp" and entry["existence_status"] == "exists"
    ]

    assert positives
    assert all(entry["source_fact_key"].startswith("owner_2026_08_13.") for entry in positives)


def test_owner_prices_reach_axis_catalog_without_hardcoded_overrides(monkeypatch) -> None:
    monkeypatch.setenv("MANGO_EVALUATION_DATE", "2026-08-13")
    catalog = build_price_axes_catalog(facts())

    assert select_price(catalog, brand="foton", grade=9, subject="math", format="online", period="year")["entry"]["amount"] == 57000
    assert select_price(catalog, brand="unpk", grade=9, subject="informatics", format="online", period="year", schedule="weekend")["entry"]["amount"] == 59000
    assert select_price(catalog, brand="unpk", grade=9, subject="informatics", format="online", period="year", schedule="weekday")["entry"]["amount"] == 69900


def test_unpk_chemistry_negative_fact_stays_narrow(monkeypatch) -> None:
    monkeypatch.setenv("MANGO_EVALUATION_DATE", "2026-08-13")
    catalog = build_product_existence_axes_catalog(facts())

    assert verify_product_format_exists(catalog, brand="unpk", grade=10, subject="chemistry", format="offline")["status"] == "not_offered"
    assert verify_product_format_exists(catalog, brand="unpk", subject="chemistry")["status"] == "unknown"
    assert verify_product_format_exists(catalog, brand="unpk", grade=10, subject="chemistry", format="online")["status"] == "unknown"


def test_unpk_offline_price_is_known_before_it_takes_effect(monkeypatch) -> None:
    monkeypatch.setenv("MANGO_EVALUATION_DATE", "2026-08-13")
    catalog = build_price_axes_catalog(facts())
    assert any(
        entry["source_fact_key"] == "owner_2026_08_13.unpk.regular.offline.5_11.year.82000"
        for entry in catalog["entries"]
    )
    selected = select_price(
        catalog, brand="unpk", grade=8, format="offline", period="year"
    )
    assert selected["entry"]["amount"] == 82000
    assert "после 15 августа" in selected["entry"]["client_safe_text"]


def test_referral_facts_come_only_from_owner_approved_layer() -> None:
    records = facts()
    referral = [
        fact
        for fact in records
        if fact.get("allowed_for_client_answer") is True
        and (
            "referral" in str(fact.get("fact_key") or "").casefold()
            or "приведи друга" in str(fact.get("client_safe_text") or "").casefold()
            or "рефераль" in str(fact.get("client_safe_text") or "").casefold()
        )
    ]

    assert {str(fact.get("fact_key")) for fact in referral} == {
        "owner_2026_08_13.foton.referral.offline",
        "owner_2026_08_13.foton.referral.online",
        "owner_2026_08_13.unpk.referral",
    }


def test_zvsh_2027_dates_and_price_reach_bot_context(monkeypatch) -> None:
    monkeypatch.setenv("MANGO_EVALUATION_DATE", "2026-08-13")
    context = build_telegram_pilot_context_from_snapshot(
        "Когда и сколько стоит ЗВШ 2027?",
        snapshot_path=SNAPSHOT,
        active_brand="unpk",
        topic_id="program",
        required_fact_keys=("programs.current", "prices.current", "schedule.current"),
    ).to_prompt_context()

    confirmed = "\n".join(str(value) for value in context["confirmed_facts"].values())
    assert context.get("missing_facts", []) == []
    assert "93 600" in confirmed
    assert "3-10 января 2027" in confirmed


def test_owner_facts_have_auditable_manifest_provenance() -> None:
    manifest = RELEASE.parent / f"{RELEASE.name}_sources" / "release_manifest.yaml"
    expected_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    owner_facts = [
        fact
        for fact in facts()
        if str(fact.get("fact_key") or "").startswith(("owner_2026_08_13.", "owner_schedule_2026_27."))
    ]

    assert owner_facts
    assert all(Path(str(fact["source_path"])).name == "release_manifest.yaml" for fact in owner_facts)
    assert all(fact["source_sha256"] == expected_sha for fact in owner_facts)
    assert all(fact["usable_for_precise_answer"] is True for fact in owner_facts)


def test_retired_summer_2026_has_only_closed_enrollment_facts() -> None:
    summer = [
        fact
        for fact in facts()
        if "summer" in str(fact.get("fact_key") or "").casefold()
        and fact.get("allowed_for_client_answer") is True
    ]

    assert {fact["fact_key"] for fact in summer} == {
        "owner_2026_08_13.foton.summer_schools_2026.enrollment_closed",
        "owner_2026_08_13.unpk.summer_schools_2026.enrollment_closed",
    }
    assert all("набор" in fact["client_safe_text"].casefold() and "заверш" in fact["client_safe_text"].casefold() for fact in summer)


def test_old_unpk_weekend_program_fact_is_removed() -> None:
    assert not [
        fact
        for fact in facts()
        if str(fact.get("fact_key") or "").startswith(
            "kb_v6_6_client_safe_facts_2026_06_08.annual_online_courses_math_physics_5_11_weekend_2026_27."
        )
    ]


def test_wrong_owner_price_is_not_preserved_in_manager_draft() -> None:
    result = apply_authoritative_output_gate(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="ЗВШ пройдёт 3-14 января 2027 года. Цена — 93 100 руб.",
            metadata={"direct_path": {"enabled": True, "direct_path_attempted": True}},
        ),
        client_message="Когда ЗВШ и сколько стоит?",
        context={
            "active_brand": "unpk",
            "confirmed_facts": {
                "dates": "ЗВШ предварительно пройдёт 3-10 января 2027 года.",
                "price": "Текущая стоимость — 93 600 руб., полная — 117 000 руб.",
            },
        },
    )

    assert result.route in {"draft_for_manager", "manager_only"}
    assert "93 100" not in result.draft_text
    assert "3-14" not in result.draft_text


def test_model_driven_retriever_delivers_owner_facts_for_non_keyword_questions() -> None:
    cases = (
        (
            "Что за реферальная программа УНПК и сколько выплачивают?",
            "owner_2026_08_13.unpk.referral",
            "discount",
            "5 000",
        ),
        (
            "Есть ли очная химия в УНПК для 10-11 класса?",
            "owner_2026_08_13.unpk.regular.offline.10_11.chemistry.not_offered",
            "program",
            "химии нет",
        ),
        (
            "Набор в летнюю школу 2026 ещё идёт?",
            "owner_2026_08_13.unpk.summer_schools_2026.enrollment_closed",
            "program",
            "набор на летние школы 2026 года завершён",
        ),
    )
    for question, fact_key, fact_type, expected in cases:
        pack = _direct_path_context_fact_pack(
            {
                "active_brand": "unpk",
                "snapshot_path": str(SNAPSHOT),
                LLM_RETRIEVE_ENV: "1",
                ASSUMED_SCOPE_GUARD_ENV: "1",
                RETRIEVER_MODEL_DRIVEN_ENV: "1",
            },
            client_message=question,
            retriever_fn=lambda _prompt, key=fact_key, kind=fact_type: {
                "needed_facts": [
                    {
                        "theme": "product_information",
                        "fact_type": kind,
                        "brand": "unpk",
                        "why_needed": "прямой ответ на вопрос клиента",
                        "importance": "required",
                    }
                ],
                "exact_ids": [key],
                "adjacent_ids": [],
            },
        )

        selected = "\n".join(str(pack["facts"][key]) for key in pack["exact_keys"])
        assert pack["selected_category"] == "llm_retrieve"
        assert fact_key in pack["exact_keys"]
        assert expected.casefold() in selected.casefold()
