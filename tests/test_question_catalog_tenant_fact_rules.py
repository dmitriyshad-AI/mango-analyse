from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.productization.tenant_config import load_tenant_config


def test_question_catalog_fact_registry_contains_rop_fact_rules() -> None:
    payload = json.loads(Path("product_data/question_catalog/current_fact_source_registry.json").read_text(encoding="utf-8"))
    facts = payload["tenant_fact_rules"]["facts"]

    assert facts["installment_terms"]["eligible_legal_entity"] == ["Фотон"]
    assert facts["matkap_procedure"]["accepted_types"] == ["federal"]
    assert "regional" not in facts["matkap_procedure"]["accepted_types"]
    assert facts["trial_class"]["allowed_formats"] == ["онлайн"]
    assert "очно" not in facts["trial_class"]["allowed_formats"]


def test_question_catalog_tenant_config_contains_same_rop_fact_rules() -> None:
    result = load_tenant_config("product_data/question_catalog/tenant_config_foton_question_catalog_v1.json")

    assert result is not None
    facts = result.config["facts"]
    assert facts["installment_terms"]["eligible_legal_entity"] == ["Фотон"]
    assert facts["matkap_procedure"]["accepted_types"] == ["federal"]
    assert facts["trial_class"]["allowed_formats"] == ["онлайн"]
