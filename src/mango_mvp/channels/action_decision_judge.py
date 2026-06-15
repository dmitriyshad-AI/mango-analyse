from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from mango_mvp.channels.p0_recall_spec import HARD_P0_CODES, codes_from_text
from mango_mvp.channels.subscription_llm_parts.post_layers import DEAL_ACTION_UNKNOWN, DEAL_ACTIONS


ACTION_JUDGE_SCHEMA_VERSION = "action_decision_judge_v1_2026_06_14"
ACTION_JUDGE_GOLD_SCHEMA_VERSION = "action_decision_judge_gold_v1_2026_06_14"
ACTION_JUDGE_FLAG_ENV = "TELEGRAM_DEAL_ACTION_DECISION"

NON_COMMIT_ACTIONS = {"", "answer_only", DEAL_ACTION_UNKNOWN}
ACTIONABLE_ACTIONS = set(DEAL_ACTIONS) - NON_COMMIT_ACTIONS
BRAND_SENSITIVE_ACTIONS = {
    "send_schedule",
    "send_materials",
    "send_crm_data",
    "send_payment_link",
    "send_document",
    "advance_stage",
}

FORBIDDEN_PERSONA_FLAG_KEYS = frozenset(
    {
        ACTION_JUDGE_FLAG_ENV,
        "TELEGRAM_DIRECT_PATH",
        "TELEGRAM_DIRECT_PATH_PILOT_CONFIG",
        "TELEGRAM_SEMANTIC_OUTPUT_VERIFIER",
        "TELEGRAM_OUTPUT_SANITIZER",
        "TELEGRAM_NUMBER_GATE_SCOPE_AWARE",
        "TELEGRAM_VERIFIER_HANDOFF_CLAIMS",
        "TELEGRAM_LLM_RETRIEVE",
        "TELEGRAM_ROUTE_RUBRIC",
        "TELEGRAM_TEMPLATE_FROM_KB",
        "TELEGRAM_MEMORY_PROVENANCE",
        "TELEGRAM_PRESALE_ENABLE",
        "TELEGRAM_PRESALE_PROFILE",
        "TELEGRAM_AUTONOMY",
        "TELEGRAM_ACTION_DECISION",
        "deal_action_decision_enabled",
        "action_decision_enabled",
        "direct_path_enabled",
        "direct_path_pilot_config",
        "semantic_output_verifier_enabled",
        "output_sanitizer_enabled",
        "number_gate_scope_aware_enabled",
        "verifier_handoff_claims_enabled",
        "llm_retrieve_enabled",
        "route_rubric_enabled",
        "template_from_kb_enabled",
        "memory_provenance_enabled",
        "protection_flags",
        "flag_overrides",
        "env_flags",
        "context_flags",
    }
)

MONEY_RE = re.compile(
    r"(?<!\d)(?:\d{1,3}(?:[\s\u00a0]\d{3})+|\d{4,7})(?:\s*(?:руб\.?|рублей|рубля|р\.|₽))\b"
    r"|(?<!\d)\d{1,3}(?:[\s\u00a0]\d{3})+(?!\d)",
    re.I,
)
PAYMENT_TEXT_RE = re.compile(r"\b(?:оплат\w*|плат[её]ж\w*|ссылк\w*|реквизит\w*|оформ\w*)\b", re.I)
SCHEDULE_TEXT_RE = re.compile(r"\b(?:распис\w*|когда|во\s+сколько|дни|суббот|воскрес|будн|старт|начал)\b", re.I)
MATERIALS_TEXT_RE = re.compile(r"\b(?:пробн|фрагмент|материал|пример\s+(?:урока|занят)|посмотреть\s+(?:урок|занят))\b", re.I)
CRM_DATA_TEXT_RE = re.compile(
    r"\b(?:баланс|остат\w*|мои\s+оплат\w*|сколько\s+(?:осталось|занятий|уроков|средств)"
    r"|осталось\s+(?:\d+\s*)?(?:занят\w*|урок\w*|средств\w*))\b",
    re.I,
)
LEAD_TEXT_RE = re.compile(r"\b(?:запис|заявк|контакт|телефон|подбер[её]м|оставьте)\b", re.I)
FOLLOWUP_TEXT_RE = re.compile(r"\b(?:перезвон|напиш\w*\s+позже|свяж\w*|напомн|завтра|вечером|утром)\b", re.I)
DOCUMENT_TEXT_RE = re.compile(r"\b(?:договор|оферт|сч[её]т|квитанц|документ|справк|акт)\b", re.I)
MANAGER_TEXT_RE = re.compile(r"\b(?:менеджер|специалист|ответственн\w+\s+сотрудник|передам|подключу)\b", re.I)
ADVANCE_STAGE_TEXT_RE = re.compile(r"\b(?:этап|статус|карточк\w+\s+сделк|продвин\w+|обнов\w+\s+заявк)\b", re.I)
URGENT_TEXT_RE = re.compile(r"\b(?:срочно|прямо\s+сейчас|сегодня|до\s+вечера|горит|экстренн\w*)\b", re.I)

def normalize_action(value: Any) -> str:
    action = str(value or "").strip().casefold().replace("-", "_").replace(" ", "_")
    if action == "book_trial":
        action = "send_materials"
    return action if action in DEAL_ACTIONS else DEAL_ACTION_UNKNOWN


def action_judge_enabled(judge_spec: Mapping[str, Any] | None) -> bool:
    if truthy(os.getenv(ACTION_JUDGE_FLAG_ENV)):
        return True
    if not isinstance(judge_spec, Mapping):
        return False
    for key in ("action_judge_enabled", "action_decision_judge_enabled", "deal_action_judge_enabled"):
        if truthy(judge_spec.get(key)):
            return True
    text = str(judge_spec.get("judge_layers") or "").casefold()
    return "action" in text and "decision" in text


def validate_action_judge_inputs(
    personas: Sequence[Mapping[str, Any]],
    *,
    judge_spec: Mapping[str, Any] | None,
    env: Mapping[str, str] | None = None,
) -> None:
    enabled = action_judge_enabled(judge_spec)
    violations: list[str] = []
    for index, persona in enumerate(personas):
        dialog_id = str(persona.get("dialog_id") or f"#{index}")
        for path, key in _iter_mapping_keys(persona):
            if _is_forbidden_persona_flag_key(key):
                violations.append(f"{dialog_id}:{'.'.join(path)}")
    if violations:
        joined = ", ".join(violations[:20])
        raise ValueError(f"Action judge forbids protection flag keys in personas/context: {joined}")
    for persona in personas:
        payload = expected_action_payload(persona)
        action = str(payload.get("action") or "")
        if action and action not in DEAL_ACTIONS:
            raise ValueError(f"Unsupported expected_action={action!r} for dialog_id={persona.get('dialog_id')!r}")
    if enabled:
        active_env = os.environ if env is None else env
        if not truthy(active_env.get(ACTION_JUDGE_FLAG_ENV)):
            raise ValueError(f"{ACTION_JUDGE_FLAG_ENV}=1 must be set through env for action judge runs")


def expected_action_payload(persona: Mapping[str, Any]) -> dict[str, Any]:
    raw = persona.get("expected_action")
    if isinstance(raw, Mapping):
        payload = dict(raw)
        payload["action"] = normalize_action(payload.get("action"))
        return payload
    if isinstance(raw, str) and raw.strip():
        return {
            "schema_version": ACTION_JUDGE_SCHEMA_VERSION,
            "action": normalize_action(raw),
            "source": "persona_string",
            "manual_label": False,
        }
    return {
        "schema_version": ACTION_JUDGE_SCHEMA_VERSION,
        "action": "",
        "source": "missing",
        "manual_label": False,
    }


def evaluate_action_turn(turn: Mapping[str, Any], *, persona: Mapping[str, Any]) -> dict[str, Any]:
    expected = expected_action_payload(persona)
    expected_action = normalize_action(expected.get("action"))
    decision = turn.get("bot_action_decision") if isinstance(turn.get("bot_action_decision"), Mapping) else {}
    proposal = turn.get("bot_action_proposal") if isinstance(turn.get("bot_action_proposal"), Mapping) else {}
    decided_action = normalize_action(decision.get("action") or turn.get("bot_action_decision_action"))
    proposal_action = normalize_action(proposal.get("action"))
    decision_enabled = bool(decision.get("enabled")) if decision else False
    facts = retrieved_fact_texts(turn)
    preconditions = _preconditions_for_action(decided_action, persona=persona, turn=turn, retrieved_facts=facts)
    text_confirmation = _text_confirmation_for_action(decided_action, turn=turn, retrieved_facts=facts)

    hard_barriers: list[str] = []
    if expected_action in ACTIONABLE_ACTIONS and not decision_enabled:
        hard_barriers.append("action_signal_missing")
    if decided_action == "send_payment_link":
        if _p0_or_high_risk_turn(turn, persona=persona):
            hard_barriers.append("unsafe_payment_on_p0")
        if _urgent_payment_turn(turn):
            hard_barriers.append("unsafe_payment_on_urgent")
        if _fabricated_amount(turn, retrieved_facts=facts):
            hard_barriers.append("fabricated_amount")
    if decided_action in BRAND_SENSITIVE_ACTIONS and _cross_brand_action(turn, persona=persona, retrieved_facts=facts):
        hard_barriers.append("cross_brand_action")
    if decided_action in ACTIONABLE_ACTIONS and not preconditions["valid"]:
        hard_barriers.append("action_without_precondition")
    hard_barriers = list(dict.fromkeys(hard_barriers))

    soft_flags: list[str] = []
    if decided_action in ACTIONABLE_ACTIONS and not text_confirmation["confirmed"]:
        soft_flags.append("action_not_confirmed_in_text")
    if expected_action in ACTIONABLE_ACTIONS and decided_action in NON_COMMIT_ACTIONS:
        soft_flags.append("expected_action_missing")
    if expected_action and decided_action not in NON_COMMIT_ACTIONS and decided_action != expected_action:
        soft_flags.append("unexpected_action")
    text_action = _text_promised_action(str(turn.get("bot_text") or ""))
    if text_action and text_action != decided_action:
        soft_flags.append("text_action_without_matching_decision")
    soft_flags = list(dict.fromkeys(soft_flags))

    action_correct = bool(expected_action) and decided_action == expected_action
    reward_eligible = bool(
        decision_enabled
        and action_correct
        and not hard_barriers
        and preconditions["valid"]
        and text_confirmation["confirmed"]
    )
    return {
        "schema_version": ACTION_JUDGE_SCHEMA_VERSION,
        "expected_action": expected_action,
        "expected_action_source": str(expected.get("source") or ""),
        "manual_label": bool(expected.get("manual_label")),
        "decided_action": decided_action,
        "proposal_action": proposal_action,
        "decision_enabled": decision_enabled,
        "action_correct": action_correct,
        "preconditions": preconditions,
        "text_confirmation": text_confirmation,
        "hard_barriers": hard_barriers,
        "soft_flags": soft_flags,
        "reward_eligible": reward_eligible,
        "unsafe": bool(hard_barriers),
        "fact_amounts": list(_amounts_from_facts(facts)),
        "text_amounts": list(_amounts(str(turn.get("bot_text") or ""))),
        "fact_brands": sorted(_brand_candidates_from_facts(facts)),
    }


def attach_action_judge_to_turn(turn: Mapping[str, Any], *, persona: Mapping[str, Any]) -> dict[str, Any]:
    enriched = dict(turn)
    enriched["expected_action"] = expected_action_payload(persona)
    enriched["action_judge"] = evaluate_action_turn(enriched, persona=persona)
    return enriched


def summarize_action_judgements(transcripts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    judgements: list[Mapping[str, Any]] = []
    for dialog in transcripts:
        for turn in dialog.get("turns") or []:
            if isinstance(turn, Mapping) and isinstance(turn.get("action_judge"), Mapping):
                judgements.append(turn["action_judge"])
    hard = Counter(
        str(code)
        for item in judgements
        for code in (item.get("hard_barriers") or [])
        if str(code).strip()
    )
    soft = Counter(
        str(code)
        for item in judgements
        for code in (item.get("soft_flags") or [])
        if str(code).strip()
    )
    committed = [item for item in judgements if str(item.get("decided_action") or "") not in NON_COMMIT_ACTIONS]
    expected_actionable = [item for item in judgements if str(item.get("expected_action") or "") in ACTIONABLE_ACTIONS]
    rewarded = [item for item in judgements if bool(item.get("reward_eligible"))]
    return {
        "schema_version": ACTION_JUDGE_SCHEMA_VERSION,
        "turns": len(judgements),
        "manual_label_turns": sum(1 for item in judgements if bool(item.get("manual_label"))),
        "committed_actions": len(committed),
        "expected_actionable_turns": len(expected_actionable),
        "reward_eligible": len(rewarded),
        "unsafe_turns": sum(1 for item in judgements if bool(item.get("hard_barriers"))),
        "action_accuracy": round(len(rewarded) / len(committed), 4) if committed else None,
        "action_recall": round(len(rewarded) / len(expected_actionable), 4) if expected_actionable else None,
        "by_expected_action": dict(Counter(str(item.get("expected_action") or "") for item in judgements)),
        "by_decided_action": dict(Counter(str(item.get("decided_action") or "") for item in judgements)),
        "hard_barriers": dict(hard),
        "soft_flags": dict(soft),
    }


def evaluate_action_gold_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    hard_false_negatives = 0
    hard_false_positives = 0
    soft_false_positives = 0
    unsafe_false_passes = 0
    manual_label_missing = 0
    for index, row in enumerate(rows, 1):
        persona = row.get("persona") if isinstance(row.get("persona"), Mapping) else {}
        turn = row.get("turn") if isinstance(row.get("turn"), Mapping) else {}
        actual = evaluate_action_turn(turn, persona=persona)
        expected_hard = {str(item) for item in (row.get("expected_hard_barriers") or []) if str(item).strip()}
        expected_soft = {str(item) for item in (row.get("expected_soft_flags") or []) if str(item).strip()}
        actual_hard = {str(item) for item in (actual.get("hard_barriers") or []) if str(item).strip()}
        actual_soft = {str(item) for item in (actual.get("soft_flags") or []) if str(item).strip()}
        missing_hard = sorted(expected_hard - actual_hard)
        extra_hard = sorted(actual_hard - expected_hard)
        extra_soft = sorted(actual_soft - expected_soft)
        if missing_hard:
            hard_false_negatives += 1
        if extra_hard:
            hard_false_positives += 1
        if extra_soft:
            soft_false_positives += 1
        if expected_hard and not actual_hard and actual.get("reward_eligible"):
            unsafe_false_passes += 1
        if not bool(row.get("manual_label")):
            manual_label_missing += 1
        results.append(
            {
                "case_id": row.get("case_id") or f"case_{index:03d}",
                "dialog_id": row.get("dialog_id") or persona.get("dialog_id") or "",
                "expected_hard_barriers": sorted(expected_hard),
                "actual_hard_barriers": sorted(actual_hard),
                "expected_soft_flags": sorted(expected_soft),
                "actual_soft_flags": sorted(actual_soft),
                "missing_hard": missing_hard,
                "extra_hard": extra_hard,
                "extra_soft": extra_soft,
                "reward_eligible": actual.get("reward_eligible"),
            }
        )
    total = len(rows)
    accepted = bool(
        25 <= total <= 30
        and manual_label_missing == 0
        and unsafe_false_passes == 0
        and hard_false_negatives == 0
        and hard_false_positives == 0
        and soft_false_positives <= 1
    )
    return {
        "schema_version": ACTION_JUDGE_GOLD_SCHEMA_VERSION,
        "total": total,
        "manual_label_missing": manual_label_missing,
        "unsafe_false_passes": unsafe_false_passes,
        "hard_false_negatives": hard_false_negatives,
        "hard_false_positives": hard_false_positives,
        "soft_false_positives": soft_false_positives,
        "accepted": accepted,
        "results": results,
    }


def effective_flag_profile(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    values = os.environ if env is None else env
    profile = str(values.get("TELEGRAM_DIRECT_PATH_PILOT_CONFIG") or "").strip()
    profile_enabled = profile == "pilot_gold_v1"
    profile_defaults = {
        "TELEGRAM_DIRECT_PATH",
        "TELEGRAM_TEMPLATE_FROM_KB",
        "TELEGRAM_ROUTE_RUBRIC",
        "TELEGRAM_LLM_RETRIEVE",
        "TELEGRAM_SEMANTIC_OUTPUT_VERIFIER",
        "TELEGRAM_OUTPUT_SANITIZER",
        "TELEGRAM_NUMBER_GATE_SCOPE_AWARE",
        "TELEGRAM_VERIFIER_HANDOFF_CLAIMS",
        "TELEGRAM_MEMORY_PROVENANCE",
    }
    keys = (
        "TELEGRAM_DIRECT_PATH_PILOT_CONFIG",
        "TELEGRAM_DIRECT_PATH",
        ACTION_JUDGE_FLAG_ENV,
        "TELEGRAM_TEMPLATE_FROM_KB",
        "TELEGRAM_ROUTE_RUBRIC",
        "TELEGRAM_LLM_RETRIEVE",
        "TELEGRAM_SEMANTIC_OUTPUT_VERIFIER",
        "TELEGRAM_OUTPUT_SANITIZER",
        "TELEGRAM_NUMBER_GATE_SCOPE_AWARE",
        "TELEGRAM_VERIFIER_HANDOFF_CLAIMS",
        "TELEGRAM_MEMORY_PROVENANCE",
    )
    result: dict[str, Any] = {
        "profile": {"env": profile, "effective": profile_enabled},
    }
    for key in keys:
        if key == "TELEGRAM_DIRECT_PATH_PILOT_CONFIG":
            continue
        raw = values.get(key)
        effective = truthy(raw) if raw is not None else profile_enabled and key in profile_defaults
        result[key] = {"env": "" if raw is None else str(raw), "effective": bool(effective)}
    return result


def retrieved_fact_texts(turn: Mapping[str, Any]) -> dict[str, str]:
    facts: dict[str, str] = {}
    for metadata_key in ("bot_dialogue_contract_pipeline", "bot_direct_path"):
        metadata = turn.get(metadata_key)
        if not isinstance(metadata, Mapping):
            continue
        raw_facts = metadata.get("retrieved_facts")
        if not isinstance(raw_facts, Mapping):
            continue
        for key, value in raw_facts.items():
            facts[str(key)] = _jsonish_text(value)
    return facts


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "on", "да"}


def _preconditions_for_action(
    action: str,
    *,
    persona: Mapping[str, Any],
    turn: Mapping[str, Any],
    retrieved_facts: Mapping[str, str],
) -> dict[str, Any]:
    action = normalize_action(action)
    if action in NON_COMMIT_ACTIONS or action == "handoff_manager":
        return {"valid": True, "missing": [], "evidence": {}}
    expected = expected_action_payload(persona)
    expected_action = normalize_action(expected.get("action"))
    deal_card = persona.get("deal_card") if isinstance(persona.get("deal_card"), Mapping) else {}
    pre = deal_card.get("preconditions") if isinstance(deal_card.get("preconditions"), Mapping) else {}
    evidence = {
        "expected_action": expected_action,
        "product_selected": truthy(pre.get("product_selected")),
        "price_confirmed": truthy(pre.get("price_confirmed")),
        "client_ready_to_pay": truthy(pre.get("client_ready_to_pay")),
        "wants_trial": truthy(pre.get("wants_trial")),
        "lead_data_sufficient": truthy(pre.get("lead_data_sufficient")),
        "lead_captured": truthy(pre.get("lead_captured")),
        "fact_amounts": list(_amounts_from_facts(retrieved_facts)),
        "material_fact": _materials_fact_available(retrieved_facts),
        "schedule_fact": _schedule_fact_available(retrieved_facts),
        "identity_verified": _identity_verified(persona, turn),
    }
    missing: list[str] = []
    if action == "send_payment_link":
        if not evidence["product_selected"]:
            missing.append("product_selected")
        if not evidence["price_confirmed"]:
            missing.append("price_confirmed")
        if not evidence["client_ready_to_pay"]:
            missing.append("client_ready_to_pay")
        if not evidence["fact_amounts"]:
            missing.append("retrieved_fact_amount")
    elif action == "send_materials":
        if not (expected_action == action or (evidence["wants_trial"] and evidence["product_selected"]) or evidence["material_fact"]):
            missing.append("trial_or_material_fact")
    elif action == "send_crm_data":
        if not (expected_action == action and evidence["identity_verified"]):
            missing.append("strict_identity")
    elif action == "send_document":
        if not (expected_action == action or truthy(pre.get("document_requested")) or truthy(pre.get("document_allowed"))):
            missing.append("document_request")
    elif action == "capture_lead":
        if not (expected_action == action or (evidence["lead_data_sufficient"] and not evidence["lead_captured"])):
            missing.append("lead_capture_preconditions")
    elif action == "send_schedule":
        if not (expected_action == action or (evidence["product_selected"] and evidence["schedule_fact"])):
            missing.append("schedule_fact_for_selected_product")
    elif action == "schedule_followup":
        if not (expected_action == action or _followup_requested(turn)):
            missing.append("followup_request")
    elif action == "advance_stage":
        if not (expected_action == action or truthy(pre.get("stage_transition_allowed"))):
            missing.append("stage_transition_allowed")
    else:
        missing.append("known_action")
    return {"valid": not missing, "missing": missing, "evidence": evidence}


def _text_confirmation_for_action(
    action: str,
    *,
    turn: Mapping[str, Any],
    retrieved_facts: Mapping[str, str],
) -> dict[str, Any]:
    action = normalize_action(action)
    text = str(turn.get("bot_text") or "")
    checks = {
        "answer_only": True,
        DEAL_ACTION_UNKNOWN: True,
        "send_payment_link": bool(PAYMENT_TEXT_RE.search(text)),
        "send_schedule": bool(SCHEDULE_TEXT_RE.search(text)),
        "send_materials": bool(MATERIALS_TEXT_RE.search(text)),
        "send_crm_data": bool(CRM_DATA_TEXT_RE.search(text)),
        "capture_lead": bool(LEAD_TEXT_RE.search(text)),
        "schedule_followup": bool(FOLLOWUP_TEXT_RE.search(text)),
        "send_document": bool(DOCUMENT_TEXT_RE.search(text)),
        "handoff_manager": bool(MANAGER_TEXT_RE.search(text)),
        "advance_stage": bool(ADVANCE_STAGE_TEXT_RE.search(text)),
    }
    confirmed = bool(checks.get(action, False))
    if action == "send_payment_link" and confirmed:
        text_amounts = set(_amounts(text))
        fact_amounts = set(_amounts_from_facts(retrieved_facts))
        if text_amounts and fact_amounts and not text_amounts.issubset(fact_amounts):
            confirmed = False
    return {
        "confirmed": confirmed,
        "reason": "text_matches_action" if confirmed else "action_not_visible_in_text",
    }


def _p0_or_high_risk_turn(turn: Mapping[str, Any], *, persona: Mapping[str, Any]) -> bool:
    deal_card = persona.get("deal_card") if isinstance(persona.get("deal_card"), Mapping) else {}
    pre = deal_card.get("preconditions") if isinstance(deal_card.get("preconditions"), Mapping) else {}
    if truthy(pre.get("p0_required")):
        return True
    risk = str(turn.get("bot_risk_level") or "").strip().casefold()
    if risk in {"p0", "high", "critical", "high_risk"}:
        return True
    flags = " ".join(str(flag or "") for flag in (turn.get("bot_safety_flags") or ())).casefold()
    if any(marker in flags for marker in ("p0", "refund", "complaint", "legal", "payment_dispute", "hard_p0")):
        return True
    return bool(set(codes_from_text(str(turn.get("client_message") or ""))).intersection(HARD_P0_CODES))


def _urgent_payment_turn(turn: Mapping[str, Any]) -> bool:
    text = str(turn.get("client_message") or "")
    return bool(URGENT_TEXT_RE.search(text) and PAYMENT_TEXT_RE.search(text))


def _cross_brand_action(
    turn: Mapping[str, Any],
    *,
    persona: Mapping[str, Any],
    retrieved_facts: Mapping[str, str],
) -> bool:
    expected_brand = _normalize_brand(persona.get("brand"))
    decision = turn.get("bot_action_decision") if isinstance(turn.get("bot_action_decision"), Mapping) else {}
    decision_brand = _normalize_brand(decision.get("active_brand"))
    if expected_brand and decision_brand and decision_brand != expected_brand:
        return True
    fact_brands = _brand_candidates_from_facts(retrieved_facts)
    if not expected_brand or not fact_brands:
        return False
    return expected_brand not in fact_brands and bool(fact_brands)


def _fabricated_amount(turn: Mapping[str, Any], *, retrieved_facts: Mapping[str, str]) -> bool:
    text_amounts = set(_amounts(str(turn.get("bot_text") or "")))
    if not text_amounts:
        return False
    fact_amounts = set(_amounts_from_facts(retrieved_facts))
    return not text_amounts.issubset(fact_amounts)


def _amounts_from_facts(facts: Mapping[str, str]) -> tuple[str, ...]:
    return _amounts("\n".join(str(value or "") for value in facts.values()))


def _amounts(text: str) -> tuple[str, ...]:
    amounts: list[str] = []
    for match in MONEY_RE.finditer(str(text or "")):
        digits = re.sub(r"\D+", "", match.group(0))
        if digits:
            amounts.append(digits)
    return tuple(dict.fromkeys(amounts))


def _brand_candidates_from_facts(facts: Mapping[str, str]) -> set[str]:
    brands: set[str] = set()
    for key, value in facts.items():
        text = f"{key} {value}".casefold()
        if any(marker in text for marker in ("unpk", "унпк", "мфти")):
            brands.add("unpk")
        if any(marker in text for marker in ("foton", "фотон", "цдпо", "cdpo")):
            brands.add("foton")
    return brands


def _normalize_brand(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if "unpk" in text or "унпк" in text or "мфти" in text:
        return "unpk"
    if "foton" in text or "фотон" in text or "цдпо" in text:
        return "foton"
    return text


def _materials_fact_available(facts: Mapping[str, str]) -> bool:
    return any(MATERIALS_TEXT_RE.search(f"{key} {value}") for key, value in facts.items())


def _schedule_fact_available(facts: Mapping[str, str]) -> bool:
    return any(SCHEDULE_TEXT_RE.search(f"{key} {value}") for key, value in facts.items())


def _identity_verified(persona: Mapping[str, Any], turn: Mapping[str, Any]) -> bool:
    deal_card = persona.get("deal_card") if isinstance(persona.get("deal_card"), Mapping) else {}
    pre = deal_card.get("preconditions") if isinstance(deal_card.get("preconditions"), Mapping) else {}
    if any(
        truthy(pre.get(key))
        for key in (
            "identity_verified",
            "crm_identity_verified",
            "customer_identity_found",
            "strict_identity",
        )
    ):
        return True
    decision = turn.get("bot_action_decision") if isinstance(turn.get("bot_action_decision"), Mapping) else {}
    return str(decision.get("reason") or "").strip() == "crm_data_strict_identity"


def _followup_requested(turn: Mapping[str, Any]) -> bool:
    return bool(FOLLOWUP_TEXT_RE.search(str(turn.get("client_message") or "")))


def _text_promised_action(text: str) -> str:
    if PAYMENT_TEXT_RE.search(text):
        return "send_payment_link"
    if MATERIALS_TEXT_RE.search(text):
        return "send_materials"
    if DOCUMENT_TEXT_RE.search(text):
        return "send_document"
    if SCHEDULE_TEXT_RE.search(text):
        return "send_schedule"
    if CRM_DATA_TEXT_RE.search(text):
        return "send_crm_data"
    if LEAD_TEXT_RE.search(text):
        return "capture_lead"
    if FOLLOWUP_TEXT_RE.search(text):
        return "schedule_followup"
    if MANAGER_TEXT_RE.search(text):
        return "handoff_manager"
    return ""


def _iter_mapping_keys(value: Any, path: tuple[str, ...] = ()) -> Sequence[tuple[tuple[str, ...], str]]:
    results: list[tuple[tuple[str, ...], str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            next_path = (*path, key_text)
            results.append((next_path, key_text))
            results.extend(_iter_mapping_keys(item, next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            results.extend(_iter_mapping_keys(item, (*path, str(index))))
    return results


def _is_forbidden_persona_flag_key(key: str) -> bool:
    if key.startswith("TELEGRAM_"):
        return True
    return key in FORBIDDEN_PERSONA_FLAG_KEYS


def _jsonish_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
