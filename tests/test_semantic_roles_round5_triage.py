from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mango_mvp.channels.answer_plan import build_answer_plan
from mango_mvp.channels.semantic_roles import tag_message_roles


ROUND5 = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/v8_holdout_fresh_round5_2026-05-25.jsonl")

EXPECT: dict[str, dict[str, Any]] = {
    "hold5_foton_p0_refund_dispute_01": {"turn": 1, "refund_frame": "dispute", "p0": True, "route": "manager_only"},
    "hold5_foton_refund_presale_peredumayu_02": {"turn": 1, "refund_frame": "presale_policy", "p0": False, "route": "bot_answer_self"},
    "hold5_foton_word_zapis_recording_03": {"turn": 0, "enrollment_vs_recording": "recording"},
    "hold5_foton_word_perevod_money_04": {"turn": 0, "transfer_sense": "money"},
    "hold5_foton_matkap_rassrochka_forbid_05": {"turn": 1, "payment_source": "matkap", "forbidden": "matkap+installment"},
    "hold5_foton_multitopic_price_installment_06": {"turn": 0, "topics_any": ["price", "installment"]},
    "hold5_foton_trial_offline_none_07": {"turn": 0, "training_format": "ochno", "topics_any": ["trial"]},
    "hold5_foton_dolyami_no_parts_08": {"turn": 0, "payment_method": "dolyami"},
    "hold5_foton_discount_second_subject_online_09": {"turn": 0, "training_format": "online", "topics_any": ["discount"]},
    "hold5_foton_assumed_need_10": {"turn": 0, "training_format": "", "payment_source": "", "topics_any": ["price"], "no_assumed": True},
    "hold5_foton_identity_11": {"turn": 0, "topics_any": ["identity"]},
    "hold5_foton_city_vs_camp_12": {"turn": 0, "topics_any": ["camp"]},
    "hold5_unpk_p0_complaint_latch_13": {"turn": 1, "external_p0": True, "p0": True, "route": "manager_only"},
    "hold5_unpk_refund_presale_notlike_14": {"turn": 1, "refund_frame": "presale_policy", "p0": False, "route": "bot_answer_self"},
    "hold5_unpk_word_zapis_enroll_15": {"turn": 0, "enrollment_vs_recording": "enroll"},
    "hold5_unpk_word_perevod_manager_16": {"turn": 0, "transfer_sense": "manager"},
    "hold5_unpk_matkap_sfr_17": {"turn": 0, "payment_source": "matkap", "forbidden": "matkap+installment"},
    "hold5_unpk_tax_deduction_18": {"turn": 0, "payment_source": "tax_deduction", "topics_any": ["tax"]},
    "hold5_unpk_online_olymp_vs_regular_19": {"turn": 0, "training_format": "online"},
    "hold5_unpk_multitopic_format_schedule_20": {"turn": 0, "topics_any": ["format", "schedule"], "format_is_question": True},
    "hold5_unpk_discount_year_14_21": {"turn": 0, "topics_any": ["discount"]},
    "hold5_unpk_over_handoff_addr_22": {"turn": 0, "asks_place": True, "topics_any": ["address"]},
    "hold5_unpk_no_bank_rassrochka_23": {"turn": 0, "payment_method": "rassrochka"},
    "hold5_unpk_assumed_need_24": {"turn": 0, "topics_any": ["price"], "no_assumed": True},
}


def _load_personas() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with ROUND5.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("type") == "persona":
                out[item["dialog_id"]] = item
    return out


def test_round5_roles_triage_has_no_structural_failures() -> None:
    personas = _load_personas()
    failures: list[str] = []

    for dialog_id, expected in EXPECT.items():
        persona = personas.get(dialog_id)
        if not persona:
            failures.append(f"{dialog_id}: persona missing")
            continue
        message = persona["behaviors"][expected["turn"]]
        roles = tag_message_roles(message)
        plan = build_answer_plan(roles, external_p0=expected.get("external_p0", False))

        for field in ("refund_frame", "enrollment_vs_recording", "transfer_sense", "payment_source", "payment_method", "training_format", "asks_place"):
            if field in expected and getattr(roles, field) != expected[field]:
                failures.append(f"{dialog_id}: {field} expected {expected[field]!r}, got {getattr(roles, field)!r}")
        if "p0" in expected and plan.p0_required != expected["p0"]:
            failures.append(f"{dialog_id}: p0 expected {expected['p0']!r}, got {plan.p0_required!r}")
        if "route" in expected and plan.route != expected["route"]:
            failures.append(f"{dialog_id}: route expected {expected['route']!r}, got {plan.route!r}")
        if "forbidden" in expected and expected["forbidden"] not in plan.forbidden_pairs:
            failures.append(f"{dialog_id}: missing forbidden pair {expected['forbidden']!r}")
        for topic in expected.get("topics_any", []):
            if topic not in roles.topics and topic not in plan.answer_topics:
                failures.append(f"{dialog_id}: missing topic {topic!r}, roles={roles.topics}, plan={plan.answer_topics}")
        if expected.get("no_assumed") and (roles.training_format or roles.payment_source or roles.payment_method):
            failures.append(f"{dialog_id}: assumed axis fmt={roles.training_format!r}, source={roles.payment_source!r}, method={roles.payment_method!r}")
        if expected.get("format_is_question") and roles.training_format in {"online", "ochno"}:
            failures.append(f"{dialog_id}: disjunctive format question latched {roles.training_format!r}")

    assert failures == []
