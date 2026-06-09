import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V67_RELEASE = PROJECT_ROOT / "product_data" / "knowledge_base" / "kb_release_20260610_v6_7_staging"
V67_FACTS = V67_RELEASE / "facts_registry.jsonl"


def _facts_by_brand_key() -> dict[tuple[str, str], dict]:
    facts: dict[tuple[str, str], dict] = {}
    for line in V67_FACTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        facts[(str(item.get("brand") or ""), str(item.get("fact_key") or ""))] = item
    return facts


def _client_text(facts: dict[tuple[str, str], dict], brand: str, key: str) -> str:
    return str(facts[(brand, key)].get("client_safe_text") or "")


def test_kb_v67_processes_keep_platform_lvsh_and_promo_decisions_client_safe():
    facts = _facts_by_brand_key()

    for brand in ("foton", "unpk"):
        availability = _client_text(facts, brand, "lvsh_mendeleevo_2026.availability_2026.client_safe_text")
        assert "места распроданы" in availability
        assert "лист ожидания" in availability
        assert "городская очная школа" in availability
        assert "онлайн-смен" not in availability.casefold()

        platform = _client_text(facts, brand, "online_platform.name")
        assert "SohoLMS" in platform

        promo = _client_text(facts, brand, f"processes_2026_06_10.{brand}.marketing_codes_absent")
        assert "Промокодов сейчас нет" in promo
        assert "учтено в прайсе" in promo
        assert "%" not in promo

        payment = _client_text(facts, brand, f"processes_2026_06_10.{brand}.payment_general")
        assert "эквайринг Альфа-Банка" in payment


def test_kb_v67_client_safe_texts_do_not_regress_to_removed_process_terms():
    facts = _facts_by_brand_key()
    all_client_text = "\n".join(str(item.get("client_safe_text") or "") for item in facts.values())

    forbidden = (
        "МТС Линк",
        "МТС-Link",
        "МТС Link",
        "Webinar",
        "бывший Webinar",
        "почти распрод",
        "живой менеджер",
        "живой сотрудник",
        "акции и промокоды",
        "онлайн-смена",
        "онлайн смена",
    )
    for phrase in forbidden:
        assert phrase not in all_client_text


def test_kb_v67_default_snapshot_not_switched_yet():
    pipeline = (PROJECT_ROOT / "src" / "mango_mvp" / "channels" / "dialogue_contract_pipeline.py").read_text(
        encoding="utf-8"
    )
    assert "kb_release_20260608_v6_6_staging/kb_release_v3_snapshot.json" in pipeline
    assert "kb_release_20260610_v6_7_staging/kb_release_v3_snapshot.json" not in pipeline

