from __future__ import annotations

from mango_mvp.channels import dialogue_contract_pipeline as legacy_dcp
from mango_mvp.channels import humanity_guards as legacy_humanity
from mango_mvp.channels import output_verification_floor as floor
from mango_mvp.channels import pilot_profile_runtime
from mango_mvp.channels import subscription_llm_parts as facade
from mango_mvp.channels.subscription_llm_parts import policy_routing, post_layers, provider, support


_HUMANITY_EXPORTS = {
    "_META_CLIENT_MARKERS",
    "_META_FACT_PHRASE_RE",
    "_WORD_CHARS",
    "_has_meta_fact_phrase",
    "_norm",
    "_tokens",
    "has_meta_leak",
    "is_near_repeat",
    "repeat_ratio",
}


def test_floor_manifest_has_one_compatibility_owner() -> None:
    assert len(floor.__all__) == 175
    assert "DIALOGUE_CONTRACT_PIPELINE_ENV" not in floor.__all__
    assert "new_concrete_anchors" not in floor.__all__

    for name in floor.__all__:
        compatibility_module = legacy_humanity if name in _HUMANITY_EXPORTS else legacy_dcp
        assert getattr(compatibility_module, name) is getattr(floor, name), name


def test_live_consumers_import_the_floor_directly() -> None:
    assert provider.dialogue_contract_p0_pre_gate is floor.p0_pre_gate
    assert provider.parse_dialogue_contract is floor.parse_contract
    assert provider.verify_dialogue_contract_output is floor.verify_output
    assert post_layers.dialogue_contract_p0_pre_gate is floor.p0_pre_gate
    assert post_layers.verify_dialogue_contract_output is floor.verify_output
    assert post_layers.has_meta_leak is floor.has_meta_leak
    assert post_layers.is_near_repeat is floor.is_near_repeat
    assert policy_routing.parse_dialogue_contract is floor.parse_contract
    assert policy_routing.verify_dialogue_contract_output is floor.verify_output
    assert policy_routing.is_near_repeat is floor.is_near_repeat
    assert support.dialogue_contract_concrete_anchors is floor.concrete_anchors
    assert pilot_profile_runtime.autonomy_scope_precision_enabled is floor.autonomy_scope_precision_enabled
    assert facade.DIRECT_PATH_PILOT_CONFIG_ENV is floor.DIRECT_PATH_PILOT_CONFIG_ENV
    assert facade.DIRECT_PATH_PILOT_CONFIG_VERSION is floor.DIRECT_PATH_PILOT_CONFIG_VERSION
    assert facade.NUMBER_GATE_SCOPE_AWARE_ENV is floor.NUMBER_GATE_SCOPE_AWARE_ENV
    assert facade._clamp_float is floor._clamp_float


def test_floor_keeps_p0_brand_meta_and_repeat_behavior() -> None:
    assert floor.p0_pre_gate("Оплатил, доступа нет.", context={}) == "payment_dispute"
    assert floor.p0_pre_gate("Ребёнок расстроился после занятия, как ему помочь?", context={}) is None

    findings = floor.verify_output("В УНПК занятия проходят очно.", facts={}, active_brand="foton")
    assert "brand_leak" in {finding.code for finding in findings}

    internal = "Автономный ответ не требуется; передам менеджеру контекст диалога."
    assert floor.has_meta_leak(internal)
    assert not floor.has_meta_leak("Передам вопрос менеджеру, он ответит по сути.")

    previous = "По физике очно для 9 класса семестр стоит 44 600 рублей."
    assert floor.is_near_repeat(previous, [previous])
    assert floor.repeat_ratio(previous, previous) == 1.0
