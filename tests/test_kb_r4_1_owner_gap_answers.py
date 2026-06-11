import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
R4_1_RELEASE = PROJECT_ROOT / "product_data" / "knowledge_base" / "kb_release_20260612_v6_7_staging_r4_1"
R4_1_SNAPSHOT = R4_1_RELEASE / "kb_release_v3_snapshot.json"


def _facts() -> dict[tuple[str, str], dict]:
    payload = json.loads(R4_1_SNAPSHOT.read_text(encoding="utf-8"))
    return {
        (str(item.get("brand") or ""), str(item.get("fact_key") or "")): item
        for item in payload["facts"]
    }


def _client(facts: dict[tuple[str, str], dict], brand: str, fact_key: str) -> str:
    return str(facts[(brand, fact_key)].get("client_safe_text") or "")


def test_kb_r4_1_owner_gap_release_is_built_but_not_default() -> None:
    build_result = json.loads((R4_1_RELEASE / "v6_1_build_result.json").read_text(encoding="utf-8"))
    assert build_result["build_result"]["quality_passed"] is True
    assert build_result["semantic_pass"] is True

    pipeline = (PROJECT_ROOT / "src" / "mango_mvp" / "channels" / "dialogue_contract_pipeline.py").read_text(
        encoding="utf-8"
    )
    runner = (PROJECT_ROOT / "scripts" / "run_telegram_dynamic_client_sim.py").read_text(encoding="utf-8")
    assert "kb_release_20260611_v6_7_staging_r4/kb_release_v3_snapshot.json" in pipeline
    assert "kb_release_20260611_v6_7_staging_r4/kb_release_v3_snapshot.json" in runner
    assert "kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json" not in pipeline
    assert "kb_release_20260612_v6_7_staging_r4_1/kb_release_v3_snapshot.json" not in runner


def test_kb_r4_1_owner_gap_client_facts_for_both_brands() -> None:
    facts = _facts()
    for brand in ("foton", "unpk"):
        individual = _client(facts, brand, f"r4_1_owner_2026_06_12.{brand}.individual_lessons_request")
        assert "Индивидуальные занятия возможны по запросу" in individual
        assert "стоимость подскажет менеджер" in individual
        assert facts[(brand, f"r4_1_owner_2026_06_12.{brand}.individual_lessons_request")]["route_policy"] == "draft_for_manager"

        trial = _client(facts, brand, f"r4_1_owner_2026_06_12.{brand}.acquaintance_mechanics")
        assert "фрагмент онлайн-занятия" in trial
        assert "Первые занятия можно использовать" in trial
        assert "по согласованию с менеджером" in trial
        assert "Постоянного бесплатного формата нет" in trial
        assert "пробного периода" not in trial
        assert "пробной недели" not in trial

        midyear = _client(facts, brand, f"r4_1_owner_2026_06_12.{brand}.midyear_entry_payment_and_records")
        assert "Присоединиться можно в течение года" in midyear
        assert "с января — 50%" in midyear
        assert "Заявление на перерасчёт не требуется" in midyear

        transfer = _client(facts, brand, f"r4_1_owner_2026_06_12.{brand}.funds_transfer_and_makeup")
        assert "Оплаченные средства не сгорают" in transfer
        assert "по согласованию с менеджером" in transfer

        group_size = _client(facts, brand, f"r4_1_owner_2026_06_12.{brand}.regular_group_size")
        assert "Очные группы — от 3 до 12 человек" in group_size
        assert "Онлайн-группы — 10-20 человек" in group_size


def test_kb_r4_1_city_school_tariffs_are_current_and_brand_separated() -> None:
    facts = _facts()
    all_client_text = "\n".join(str(item.get("client_safe_text") or "") for item in facts.values())

    assert "37 500" not in all_client_text
    assert "37500" not in all_client_text
    assert "Премиум 10" not in all_client_text
    assert "Премиум+" not in all_client_text

    foton = _client(facts, "foton", "r4_1_owner_2026_06_12.foton.city_summer_school_tariffs")
    assert "«База» — 34 300 ₽" in foton
    assert "«База + половина факультативного блока» — 49 000 ₽" in foton
    assert "«База + полный факультативный блок» — 59 000 ₽" in foton

    plus_half = _client(facts, "foton", "ls_city_2026_foton.moscow_foton.prices.plus_half")
    assert "49 000 ₽" in plus_half
    assert "половина факультативного блока" in plus_half
    assert "полный факультативный блок" not in plus_half

    unpk = _client(facts, "unpk", "r4_1_owner_2026_06_12.unpk.city_summer_school_tariffs")
    assert "«База» — 39 500 ₽" in unpk
    assert "«База + факультативный блок» — 59 500 ₽" in unpk
    assert "«База + факультативный блок с индивидуальным обучением» — 99 500 ₽" in unpk


def test_kb_r4_1_internal_owner_gap_facts_are_not_client_safe() -> None:
    facts = _facts()
    for brand in ("foton", "unpk"):
        for suffix in (
            "matkap_refund_to_sfr_internal",
            "nonpayment_second_semester_freeze_internal",
            "oge_ege_mock_exams_internal",
        ):
            fact = facts[(brand, f"r4_1_owner_2026_06_12.{brand}.{suffix}")]
            assert fact["route_policy"] == "manager_handoff_only"
            assert fact.get("client_safe_text") == ""

    product = facts[("foton", "r4_1_owner_2026_06_12.foton.new_online_oge_ege_math_product_internal")]
    assert product["route_policy"] == "manager_handoff_only"
    assert product.get("client_safe_text") == ""
