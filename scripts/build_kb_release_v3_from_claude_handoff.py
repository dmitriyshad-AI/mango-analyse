#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyYAML is required to build kb_release_v3") from exc


SCHEMA_VERSION = "kb_release_v3_snapshot_2026_05_18"
FACT_SCHEMA_VERSION = "kb_release_v3_fact_v1"
SOURCE_SCHEMA_VERSION = "kb_release_v3_source_v1"
BUILDER_VERSION = "kb_release_v3_builder_2026_05_18_v1"

DEFAULT_RUN_ID = "kb_release_20260518_v3"
DEFAULT_HANDOFF_DIR = Path("claude_to_codex_v3_handoff_2026-05-17")
DEFAULT_OUT_DIR = Path("product_data/knowledge_base/kb_release_20260518_v3")
DEFAULT_HANDOFF_OUT_DIR = Path("product_data/knowledge_base/kb_release_20260518_v3_handoff_for_claude_and_team")

SOURCE_FILES: dict[str, dict[str, str]] = {
    "brand_rules": {
        "filename": "brand_rules.yaml",
        "source_id": "claude_layer_v3:brand_rules",
        "kind": "claude_yaml",
        "brand": "brand_neutral",
    },
    "bot_policy": {
        "filename": "bot_policy.yaml",
        "source_id": "claude_layer_v3:bot_policy",
        "kind": "claude_yaml",
        "brand": "brand_neutral",
    },
    "facts_for_bot_FOTON": {
        "filename": "facts_for_bot_FOTON.yaml",
        "source_id": "claude_layer_v3:facts_for_bot_FOTON",
        "kind": "claude_yaml",
        "brand": "foton",
    },
    "facts_for_bot_UNPK": {
        "filename": "facts_for_bot_UNPK.yaml",
        "source_id": "claude_layer_v3:facts_for_bot_UNPK",
        "kind": "claude_yaml",
        "brand": "unpk",
    },
    "facts_internal_only": {
        "filename": "facts_internal_only.yaml",
        "source_id": "claude_layer_v3:facts_internal_only",
        "kind": "claude_yaml",
        "brand": "internal",
    },
    "open_questions": {
        "filename": "OPEN_QUESTIONS_FOR_TEAM.md",
        "source_id": "claude_layer_v3:open_questions",
        "kind": "claude_markdown",
        "brand": "brand_neutral",
    },
    "rebuild_requirements": {
        "filename": "REBUILD_REQUIREMENTS_FROM_CLAUDE.md",
        "source_id": "claude_layer_v3:rebuild_requirements",
        "kind": "claude_markdown",
        "brand": "brand_neutral",
    },
    "changelog_final": {
        "filename": "CHANGELOG_FINAL.md",
        "source_id": "claude_layer_v3:changelog_final",
        "kind": "claude_markdown",
        "brand": "brand_neutral",
    },
    "changelog_team_answers": {
        "filename": "CHANGELOG_v3_after_team_answers.md",
        "source_id": "claude_layer_v3:changelog_team_answers",
        "kind": "claude_markdown",
        "brand": "brand_neutral",
    },
    "readme": {
        "filename": "README.md",
        "source_id": "claude_layer_v3:readme",
        "kind": "claude_markdown",
        "brand": "brand_neutral",
    },
}

CONTROL_NUMBERS = (
    "44600",
    "74500",
    "49000",
    "82000",
    "29750",
    "47250",
    "98000",
    "75000",
    "120000",
    "89900",
    "83800",
    "16900",
    "27720",
    "18800",
    "34400",
    "3900",
    "6900",
    "23000",
    "18900",
    "94500",
    "33000",
    "50000",
    "11900",
    "56500",
    "94000",
)

SKIP_SCALAR_KEYS = {
    "schema_version",
    "generated_at",
    "active_brand_scope",
    "brand",
    "source",
    "source_title",
    "freshness_status",
    "status",
    "internal_only",
    "internal_only_for_number",
    "client_facing",
    "bot_route",
}
FORBIDDEN_KEYS = {
    "forbidden_to_say",
    "forbidden_to_ask",
    "forbidden_in_client",
    "forbidden_phrasings",
    "forbidden_in_this_response",
}
INTERNAL_PATH_MARKERS = {
    "legal_entities",
    "legal_entities_full_map",
    "crm_brand_groups",
    "cross_brand_handoff_notes",
    "cross_brand_sources_inventory",
    "internal_processes",
    "oc_script_brand_routing",
    "returns_analytics_25_26",
    "signatories_2026",
    "refund_withholdings_2026",
    "vat_exemption",
    "source_coverage_audit_2026_05_17",
}
INTERNAL_KEY_MARKERS = (
    "_internal",
    "internal_",
    "manager_use",
    "manager_visible_text",
    "manager_checklist",
    "heuristics_for_manager",
)
LICENSE_PRIVATE_KEYS = {"number", "date", "holder", "license_basis_internal", "legal_entity_internal"}
CLIENT_ALLOWED_STATUSES = {"verified", "document_verified", "fresh_verified", "current", "waiting_list"}
CLIENT_BLOCKED_STATUSES = {
    "needs_owner_confirmation",
    "discontinued",
    "do_not_use",
    "conflicting",
    "internal_only",
    "dynamic_needs_check",
}
BRAND_LABELS = {
    "foton": "Фотон",
    "unpk": "УНПК",
    "brand_neutral": "Общее правило",
    "internal": "Внутренне",
}
FACT_TYPE_TOPICS = {
    "price": "01_pricing",
    "discount": "02_discounts",
    "promocode": "03_promocodes",
    "deadline": "04_deadlines",
    "camp_lvsh": "05_lvsh",
    "camp_zvsh": "06_zvsh",
    "camp_city": "07_city_camp",
    "program": "08_program",
    "intensive": "09_intensive",
    "installment": "10_installment",
    "tax": "11_tax_deduction",
    "matkap": "12_matkap",
    "documents": "13_documents",
    "contact": "14_contacts",
    "location": "15_locations",
    "teacher": "16_teachers",
    "refund": "17_refund",
    "policy": "18_policy",
}
GLOBAL_CLIENT_FORBIDDEN_PATTERNS = (
    "source_id",
    "fact_id",
    "freshness",
    "AMO",
    "Tallanto",
    "CRM",
    "GPT",
    "Claude",
    "Codex",
    "ChatGPT",
    "я бот",
    "я ИИ",
    "я нейросеть",
    "Л035",
    "50Л01",
    "№77753",
    "70369",
    "АНО ДПО",
    "НОУ УНПК",
    "ООО «ЦДПО",
    "ООО ЦДПО",
    "ООО «ЦРДО",
    "ООО ЦРДО",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build kb_release_20260518_v3 from Claude handoff v3.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--handoff-out-dir", type=Path, default=DEFAULT_HANDOFF_OUT_DIR)
    args = parser.parse_args(argv)

    result = build_kb_release_v3(
        run_id=args.run_id,
        handoff_dir=args.handoff_dir,
        out_dir=args.out_dir,
        handoff_out_dir=args.handoff_out_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_kb_release_v3(
    *,
    run_id: str = DEFAULT_RUN_ID,
    handoff_dir: Path = DEFAULT_HANDOFF_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    handoff_out_dir: Path = DEFAULT_HANDOFF_OUT_DIR,
) -> Mapping[str, Any]:
    handoff_root = handoff_dir.expanduser().resolve(strict=False)
    out_root = guard_output_dir(out_dir)
    team_root = guard_output_dir(handoff_out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    team_root.mkdir(parents=True, exist_ok=True)

    handoff = load_handoff(handoff_root)
    sources = build_source_registry(handoff_root)
    source_lookup = {str(source["source_id"]): source for source in sources}
    post_filter = build_post_filter_registry(handoff)

    facts: list[dict[str, Any]] = []
    facts.extend(
        facts_from_payload(
            handoff["facts_for_bot_FOTON"],
            origin_key="facts_for_bot_FOTON",
            default_brand="foton",
            source=source_lookup["claude_layer_v3:facts_for_bot_FOTON"],
        )
    )
    facts.extend(
        facts_from_payload(
            handoff["facts_for_bot_UNPK"],
            origin_key="facts_for_bot_UNPK",
            default_brand="unpk",
            source=source_lookup["claude_layer_v3:facts_for_bot_UNPK"],
        )
    )
    facts.extend(
        facts_from_payload(
            handoff["facts_internal_only"],
            origin_key="facts_internal_only",
            default_brand="internal",
            source=source_lookup["claude_layer_v3:facts_internal_only"],
            force_internal=True,
        )
    )
    facts.extend(build_policy_facts(handoff, source_lookup))
    facts.extend(build_manual_decision_facts(source_lookup))
    facts = attach_source_details(dedupe_facts(facts), source_lookup=source_lookup)
    facts = sorted(facts, key=lambda item: (str(item.get("brand")), str(item.get("product")), str(item.get("fact_key"))))

    approval_queue = build_approval_queue_v3(facts)
    snapshot = build_snapshot_v3(
        run_id=run_id,
        sources=sources,
        facts=facts,
        approval_queue=approval_queue,
        post_filter=post_filter,
        brand_rules=normalize_brand_rules(handoff["brand_rules"]),
        bot_policy=normalize_bot_policy(handoff["bot_policy"]),
    )
    quality = build_quality_report(snapshot, approval_queue=approval_queue)
    snapshot["quality_summary"] = {
        "quality_passed": quality["quality_passed"],
        "blocking_failures": quality["blocking_failures"],
        "control_numbers_missing": quality["control_numbers"]["missing"],
    }

    write_outputs(
        out_root,
        team_root,
        snapshot=snapshot,
        sources=sources,
        facts=facts,
        approval_queue=approval_queue,
        post_filter=post_filter,
        quality=quality,
        brand_rules=snapshot["brand_rules"],
        bot_policy=snapshot["bot_policy"],
    )

    return {
        "run_id": run_id,
        "out_dir": str(out_root),
        "handoff_out_dir": str(team_root),
        "snapshot_path": str(out_root / "kb_release_v3_snapshot.json"),
        "facts_total": len(facts),
        "client_allowed_facts": sum(1 for fact in facts if fact.get("allowed_for_client_answer")),
        "source_registry_total": len(sources),
        "approval_queue_items": len(approval_queue),
        "quality_passed": quality["quality_passed"],
        "blocking_failures": quality["blocking_failures"],
        "control_numbers_missing": quality["control_numbers"]["missing"],
    }


def load_handoff(handoff_root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, meta in SOURCE_FILES.items():
        path = handoff_root / meta["filename"]
        if meta["filename"].endswith((".yaml", ".yml")):
            payload[key] = load_yaml(path)
        else:
            payload[key] = path.read_text(encoding="utf-8") if path.exists() else ""
    return payload


def build_source_registry(handoff_root: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for key, meta in SOURCE_FILES.items():
        path = handoff_root / meta["filename"]
        sha = sha256_file(path) if path.exists() else ""
        sources.append(
            {
                "schema_version": SOURCE_SCHEMA_VERSION,
                "source_id": meta["source_id"],
                "source_kind": meta["kind"],
                "title": f"{meta['filename']} (Claude layer v3)",
                "path": str(path),
                "url": "",
                "sha256": sha,
                "source_sha256": sha,
                "brand": meta["brand"],
                "freshness_status": "fresh_verified" if key in {"open_questions", "changelog_team_answers"} else "document_verified",
                "source_status": "read" if path.exists() else "missing",
                "source_role": "claude_v3_handoff_source",
                "read_status": "read" if path.exists() else "missing",
                "usable_for_precise_answer": key.startswith("facts_for_bot"),
                "requires_manager_confirmation": False,
            }
        )
    return sources


def facts_from_payload(
    payload: Mapping[str, Any],
    *,
    origin_key: str,
    default_brand: str,
    source: Mapping[str, Any],
    force_internal: bool = False,
) -> list[dict[str, Any]]:
    context = {
        "origin_key": origin_key,
        "brand": default_brand,
        "status": "verified",
        "freshness_status": "document_verified",
        "product": "",
        "route_policy": "",
        "linked_open_question": "",
        "internal_only": force_internal,
        "force_internal": force_internal,
        "number_fields_internal": False,
    }
    records: list[dict[str, Any]] = []
    for key, value in payload.items():
        if key in {"schema_version", "generated_at", "active_brand_scope", "brand"}:
            continue
        records.extend(walk_value(value, path=(str(key),), context=context, source=source))
    return records


def walk_value(
    value: Any,
    *,
    path: tuple[str, ...],
    context: Mapping[str, Any],
    source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        current = dict(context)
        current["brand"] = normalize_brand(value.get("brand") or current.get("brand"))
        status = clean_text(value.get("status") or current.get("status") or "verified")
        freshness = clean_text(value.get("freshness_status") or status or current.get("freshness_status"))
        current["status"] = status
        current["freshness_status"] = normalize_freshness(freshness)
        current["product"] = clean_text(value.get("product") or current.get("product"))
        current["route_policy"] = clean_text(value.get("bot_route") or current.get("route_policy"))
        current["linked_open_question"] = infer_linked_question(
            value.get("open_question") or value.get("open_question_for_rop_q3") or current.get("linked_open_question"),
            path,
        )
        if truthy(value.get("internal_only")) or truthy(value.get("client_facing")) is False and "client_facing" in value:
            current["internal_only"] = True
        current["number_fields_internal"] = truthy(value.get("internal_only_for_number"))

        records: list[dict[str, Any]] = []
        for key, item in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_KEYS:
                continue
            if key_text in SKIP_SCALAR_KEYS:
                continue
            child_context = dict(current)
            if is_internal_child(path, key_text, item, current):
                child_context["internal_only"] = True
            records.extend(walk_value(item, path=(*path, key_text), context=child_context, source=source))
        return records

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        records: list[dict[str, Any]] = []
        range_mode = bool(path and "range" in path[-1])
        for index, item in enumerate(value):
            suffix = ("min" if index == 0 else "max") if range_mode and index < 2 else str(index + 1)
            records.extend(walk_value(item, path=(*path, suffix), context=context, source=source))
        return records

    return [make_fact(path=path, value=value, context=context, source=source)]


def is_internal_child(path: tuple[str, ...], key: str, item: Any, context: Mapping[str, Any]) -> bool:
    normalized_path = ".".join(path + (key,)).casefold()
    if context.get("force_internal") or context.get("internal_only"):
        return True
    if path and path[0] in INTERNAL_PATH_MARKERS:
        return True
    if path and path[0] == "licenses" and key != "client_safe_summary":
        return True
    if context.get("number_fields_internal") and key in LICENSE_PRIVATE_KEYS:
        return True
    if any(marker in key.casefold() for marker in INTERNAL_KEY_MARKERS):
        return True
    if "teacher" in normalized_path or "prepodavat" in normalized_path or "teachers" in normalized_path:
        return True
    if "refund" in normalized_path or "vozvrat" in normalized_path:
        return True
    if "vat" in normalized_path or "nds" in normalized_path or "ндс" in normalized_path:
        return True
    if key in LICENSE_PRIVATE_KEYS:
        return True
    if isinstance(item, Mapping) and truthy(item.get("internal_only")):
        return True
    return False


def make_fact(
    *,
    path: tuple[str, ...],
    value: Any,
    context: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    fact_key = ".".join(safe_id(part) for part in path)
    brand = normalize_brand(context.get("brand"))
    status = normalize_status(context.get("status") or context.get("freshness_status"))
    freshness = normalize_freshness(context.get("freshness_status") or status)
    fact_type = infer_fact_type(path, value)
    if is_deadline_fact(path):
        fact_type = "deadline"
    product = clean_text(context.get("product") or infer_product(path, fact_type))
    internal_only = bool(context.get("internal_only") or should_force_internal(path, value))
    linked_open_question = infer_linked_question(context.get("linked_open_question"), path)
    if linked_open_question == "q14_closed" and brand != "unpk":
        linked_open_question = ""
    structured_value = build_structured_value(path, value, fact_type=fact_type)
    fact_text = render_fact_text(path, value, brand=brand, fact_type=fact_type, structured_value=structured_value)
    route_policy = infer_route_policy(path, fact_type=fact_type, status=status, internal_only=internal_only)
    risk_level = infer_risk_level(path, fact_type=fact_type)
    allowed, safety_reasons = client_allowed(
        fact_text,
        brand=brand,
        status=status,
        freshness=freshness,
        fact_type=fact_type,
        route_policy=route_policy,
        internal_only=internal_only,
        path=path,
    )
    usable_precise = bool(allowed and freshness not in {"dynamic_needs_check", "waiting_list"} and status != "waiting_list")
    requires_confirmation = bool(not allowed or status in CLIENT_BLOCKED_STATUSES or freshness == "dynamic_needs_check")
    source_id = str(source.get("source_id") or "")
    fact_id = f"fact:v3:{brand}:{safe_id(':'.join(path))}:{sha256_text(f'{fact_key}|{value}|{source_id}')[:10]}"
    client_safe_text = fact_text if allowed else ""
    manager_text = fact_text
    if safety_reasons:
        manager_text = f"{fact_text} [client_blocked: {', '.join(safety_reasons)}]"

    return {
        "schema_version": FACT_SCHEMA_VERSION,
        "fact_id": fact_id,
        "fact_key": fact_key,
        "fact_type": fact_type,
        "fact_types": [fact_type],
        "title": humanize_path(path),
        "fact_text": fact_text,
        "short_fact": humanize_path(path, max_parts=4),
        "client_safe_text": client_safe_text,
        "manager_check_text": manager_text,
        "internal_text": fact_text if internal_only else "",
        "brand": brand,
        "active_brand_scope": active_brand_scope(brand, internal_only=internal_only),
        "cross_brand_policy": "forbidden_for_client" if internal_only else "active_brand_only",
        "cross_brand_mixed": is_cross_brand_text(fact_text),
        "product": product,
        "source_id": source_id,
        "source_title": str(source.get("title") or ""),
        "source_path": str(source.get("path") or ""),
        "source_url": str(source.get("url") or ""),
        "source_sha256": str(source.get("source_sha256") or source.get("sha256") or ""),
        "source_status": str(source.get("source_status") or ""),
        "freshness_status": freshness,
        "verification_status": status,
        "structured_value": structured_value,
        "valid_from": "",
        "valid_until": structured_value.get("valid_until", ""),
        "verified_by": "",
        "verified_at": "",
        "owner_role": owner_role_for_fact(path, fact_type),
        "allowed_for_client_answer": allowed,
        "usable_for_precise_answer": usable_precise,
        "requires_manager_confirmation": requires_confirmation,
        "requires_amo_check": fact_type == "payment_status",
        "requires_tallanto_check": fact_type == "payment_status",
        "route_policy": route_policy,
        "risk_level": risk_level,
        "related_theme_ids": related_theme_ids(fact_type),
        "linked_open_question": linked_open_question,
        "forbidden_promises": forbidden_promises_for_fact_type(fact_type),
        "forbidden_client_mentions": forbidden_mentions_for_brand(brand),
        "forbidden_for_client": bool(not allowed),
        "internal_only": internal_only,
        "safety_block_reasons": safety_reasons,
        "notes": notes_for_fact(path, status, allowed),
        "record_type": "fact",
    }


def build_policy_facts(handoff: Mapping[str, Any], source_lookup: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    bot_policy = handoff["bot_policy"]
    brand_rules = handoff["brand_rules"]
    source_policy = source_lookup["claude_layer_v3:bot_policy"]
    source_rules = source_lookup["claude_layer_v3:brand_rules"]
    facts: list[dict[str, Any]] = []

    approved = bot_policy.get("approved_phrases_themes_11_12_17") if isinstance(bot_policy, Mapping) else {}
    if isinstance(approved, Mapping):
        for theme_key, payload in approved.items():
            if not isinstance(payload, Mapping):
                continue
            for brand, bot_key in (("foton", "foton_bot"), ("unpk", "unpk_bot")):
                text = payload.get(bot_key)
                if not text:
                    continue
                facts.append(
                    make_manual_fact(
                        fact_key=f"bot_policy.approved_phrases.{theme_key}.{brand}",
                        fact_type="documents" if "certificate" in theme_key or "contract" in theme_key else "teacher",
                        brand=brand,
                        product=theme_key,
                        fact_text=str(text),
                        client_safe_text=str(text),
                        source=source_policy,
                        status="verified",
                        route_policy=str(payload.get("bot_route") or "bot_answer_self_for_pilot"),
                        linked_open_question=theme_key,
                    )
                )

    relationship = {}
    if isinstance(brand_rules, Mapping):
        relationship = brand_rules.get("approved_brand_relationship_answer") or {}
    if isinstance(relationship, Mapping) and relationship.get("approved_response"):
        for brand in ("foton", "unpk"):
            facts.append(
                make_manual_fact(
                    fact_key=f"brand_rules.approved_brand_relationship_answer.{brand}",
                    fact_type="policy",
                    brand=brand,
                    product="brand_relationship",
                    fact_text=str(relationship["approved_response"]),
                    client_safe_text=str(relationship["approved_response"]),
                    source=source_rules,
                    status="verified",
                    route_policy="bot_answer_self_for_pilot",
                    linked_open_question="q3_closed",
                )
            )

    refund = (((bot_policy.get("theme_routes") or {}).get("refund") or {}) if isinstance(bot_policy, Mapping) else {})
    if isinstance(refund, Mapping) and refund.get("bot_phrase"):
        for brand in ("foton", "unpk"):
            facts.append(
                make_manual_fact(
                    fact_key=f"bot_policy.theme_routes.refund.bot_phrase.{brand}",
                    fact_type="refund",
                    brand=brand,
                    product="refund",
                    fact_text=str(refund["bot_phrase"]),
                    client_safe_text="",
                    source=source_policy,
                    status="verified",
                    route_policy="manager_only",
                    linked_open_question="q12_closed",
                    internal_only=False,
                    requires_manager_confirmation=True,
                )
            )
    return facts


def build_manual_decision_facts(source_lookup: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    req_source = source_lookup["claude_layer_v3:rebuild_requirements"]
    changelog_source = source_lookup["claude_layer_v3:changelog_team_answers"]
    internal_source = source_lookup["claude_layer_v3:facts_internal_only"]
    manual_specs = [
        {
            "fact_key": "team_answers.q11.modular_courses_m9_m11.discontinued",
            "fact_type": "program",
            "brand": "foton",
            "product": "modular_courses_m9_m11",
            "fact_text": "Модульных курсов М9/М11 сейчас нет; это были прошлые интенсивы. Не предлагать как действующий продукт.",
            "source": changelog_source,
            "status": "discontinued",
            "route_policy": "manager_handoff_only",
            "linked_open_question": "q11_closed",
        },
        {
            "fact_key": "team_answers.q11.modular_courses_m9_m11.old_price_range.min",
            "fact_type": "price",
            "brand": "foton",
            "product": "modular_courses_m9_m11",
            "fact_text": "Устаревший нижний край цены модульных М9/М11: 18 900 ₽. Не использовать как клиентский факт.",
            "source": req_source,
            "status": "discontinued",
            "route_policy": "manager_handoff_only",
            "linked_open_question": "q11_closed",
            "structured_value": {"amount": 18900, "currency": "RUB", "range_bound": "min", "do_not_use_as_current_price": True},
        },
        {
            "fact_key": "team_answers.q11.modular_courses_m9_m11.old_price_range.max",
            "fact_type": "price",
            "brand": "foton",
            "product": "modular_courses_m9_m11",
            "fact_text": "Устаревший верхний край цены модульных М9/М11: 94 500 ₽. Не использовать как клиентский факт.",
            "source": req_source,
            "status": "discontinued",
            "route_policy": "manager_handoff_only",
            "linked_open_question": "q11_closed",
            "structured_value": {"amount": 94500, "currency": "RUB", "range_bound": "max", "do_not_use_as_current_price": True},
        },
        {
            "fact_key": "team_answers.q15.unpk_online_other_classes.manager_handoff",
            "fact_type": "price",
            "brand": "unpk",
            "product": "online_olympiad_phystech_9_and_11",
            "fact_text": "По онлайн-направлениям УНПК вне олимпиадной подготовки Физтех для 9 и 11 классов точные условия должен проверить менеджер.",
            "source": changelog_source,
            "status": "verified",
            "route_policy": "manager_handoff_only",
            "linked_open_question": "q15_closed",
            "structured_value": {"scope_exception": "other_unpk_online_classes_require_manager"},
        },
        {
            "fact_key": "team_answers.q9.installment_6_12_manager_only",
            "fact_type": "installment",
            "brand": "foton",
            "product": "installment",
            "fact_text": "Рассрочка на 6 и 12 месяцев есть, но условия индивидуальны, зависят от банка и могут менять стоимость курса; вопрос передаётся менеджеру.",
            "source": changelog_source,
            "status": "verified",
            "route_policy": "manager_handoff_only",
            "linked_open_question": "q9_closed",
            "structured_value": {"term_months_options": [6, 12], "requires_bank_check": True},
        },
        {
            "fact_key": "team_answers.crm_tallanto.active_brand_from_telegram_bot",
            "fact_type": "policy",
            "brand": "internal",
            "product": "brand_detection",
            "fact_text": "Активный бренд для Telegram-пилота задаётся выбранным ботом/каналом входа, а не AMO или Tallanto.",
            "source": internal_source,
            "status": "verified",
            "route_policy": "manager_handoff_only",
            "linked_open_question": "q5_it_closed",
            "internal_only": True,
        },
        {
            "fact_key": "team_answers.telegram.two_separate_bots",
            "fact_type": "policy",
            "brand": "internal",
            "product": "telegram_architecture",
            "fact_text": "Целевая архитектура Telegram-пилота: два отдельных бота, отдельно Фотон и отдельно УНПК.",
            "source": internal_source,
            "status": "verified",
            "route_policy": "manager_handoff_only",
            "linked_open_question": "q4_2_it_closed",
            "internal_only": True,
        },
    ]
    return [make_manual_fact(**spec) for spec in manual_specs]


def make_manual_fact(
    *,
    fact_key: str,
    fact_type: str,
    brand: str,
    product: str,
    fact_text: str,
    source: Mapping[str, Any],
    status: str = "verified",
    route_policy: str = "bot_answer_self_for_pilot",
    linked_open_question: str = "",
    client_safe_text: str | None = None,
    structured_value: Mapping[str, Any] | None = None,
    internal_only: bool = False,
    requires_manager_confirmation: bool | None = None,
) -> dict[str, Any]:
    clean_brand = normalize_brand(brand)
    freshness = normalize_freshness(status)
    safe_client_text = fact_text if client_safe_text is None else client_safe_text
    allowed, safety_reasons = client_allowed(
        safe_client_text or fact_text,
        brand=clean_brand,
        status=status,
        freshness=freshness,
        fact_type=fact_type,
        route_policy=route_policy,
        internal_only=internal_only,
        path=tuple(fact_key.split(".")),
    )
    if client_safe_text == "":
        allowed = False
    fact_id = f"fact:v3:{clean_brand}:{safe_id(fact_key)}:{sha256_text(f'{fact_key}|{fact_text}')[:10]}"
    return {
        "schema_version": FACT_SCHEMA_VERSION,
        "fact_id": fact_id,
        "fact_key": fact_key,
        "fact_type": fact_type,
        "fact_types": [fact_type],
        "title": humanize_path(tuple(fact_key.split("."))),
        "fact_text": fact_text,
        "short_fact": humanize_path(tuple(fact_key.split(".")), max_parts=4),
        "client_safe_text": safe_client_text if allowed else "",
        "manager_check_text": fact_text,
        "internal_text": fact_text if internal_only else "",
        "brand": clean_brand,
        "active_brand_scope": active_brand_scope(clean_brand, internal_only=internal_only),
        "cross_brand_policy": "forbidden_for_client" if internal_only else "active_brand_only",
        "cross_brand_mixed": is_cross_brand_text(fact_text),
        "product": product,
        "source_id": str(source.get("source_id") or ""),
        "source_title": str(source.get("title") or ""),
        "source_path": str(source.get("path") or ""),
        "source_url": str(source.get("url") or ""),
        "source_sha256": str(source.get("source_sha256") or source.get("sha256") or ""),
        "source_status": str(source.get("source_status") or ""),
        "freshness_status": freshness,
        "verification_status": normalize_status(status),
        "structured_value": dict(structured_value or {}),
        "valid_from": "",
        "valid_until": "",
        "verified_by": "",
        "verified_at": "",
        "owner_role": owner_role_for_fact(tuple(fact_key.split(".")), fact_type),
        "allowed_for_client_answer": allowed,
        "usable_for_precise_answer": bool(allowed and freshness not in {"dynamic_needs_check", "waiting_list"}),
        "requires_manager_confirmation": bool(requires_manager_confirmation if requires_manager_confirmation is not None else not allowed),
        "requires_amo_check": False,
        "requires_tallanto_check": False,
        "route_policy": route_policy,
        "risk_level": infer_risk_level(tuple(fact_key.split(".")), fact_type=fact_type),
        "related_theme_ids": related_theme_ids(fact_type),
        "linked_open_question": linked_open_question,
        "forbidden_promises": forbidden_promises_for_fact_type(fact_type),
        "forbidden_client_mentions": forbidden_mentions_for_brand(clean_brand),
        "forbidden_for_client": not allowed,
        "internal_only": internal_only,
        "safety_block_reasons": safety_reasons,
        "notes": notes_for_fact(tuple(fact_key.split(".")), status, allowed),
        "record_type": "fact",
    }


def build_post_filter_registry(handoff: Mapping[str, Any]) -> dict[str, Any]:
    phrases: list[str] = []
    for key in ("brand_rules", "bot_policy", "facts_for_bot_FOTON", "facts_for_bot_UNPK", "facts_internal_only"):
        collect_forbidden_phrases(handoff.get(key), phrases)
    unique = sorted({clean_text(phrase, max_chars=500) for phrase in phrases if clean_text(phrase)})
    return {
        "schema_version": "kb_release_v3_post_filter_v1",
        "source": "Claude v3 handoff forbidden_to_say + brand rules + bot policy",
        "violation_action": "manager_only",
        "violation_flag": "brand_separation_violation",
        "phrases": unique,
        "phrases_total": len(unique),
    }


def collect_forbidden_phrases(value: Any, result: list[str]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in FORBIDDEN_KEYS or str(key) in {
                "blocked_terms",
                "forbidden_client_phrasings",
                "forbidden_in_any_brand",
                "examples_blocked",
            }:
                result.extend(flatten_scalars(item))
            else:
                collect_forbidden_phrases(item, result)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            collect_forbidden_phrases(item, result)


def flatten_scalars(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        out: list[str] = []
        for item in value.values():
            out.extend(flatten_scalars(item))
        return out
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        out: list[str] = []
        for item in value:
            out.extend(flatten_scalars(item))
        return out
    return [clean_text(value, max_chars=500)] if value is not None else []


def build_approval_queue_v3(facts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for fact in facts:
        item_type = approval_item_type(fact)
        if not should_include_in_approval_queue(fact, item_type):
            continue
        fact_id = str(fact.get("fact_id") or "")
        topic = FACT_TYPE_TOPICS.get(item_type, FACT_TYPE_TOPICS.get(str(fact.get("fact_type")), "99_other"))
        queue.append(
            {
                "priority": approval_priority(fact, item_type),
                "approval_item_id": f"approve:{safe_id(fact_id)}",
                "item_type": item_type,
                "topic": topic,
                "fact_id_ref": fact_id,
                "brand": fact.get("brand"),
                "product": fact.get("product"),
                "manager_text": fact.get("manager_check_text") or fact.get("fact_text"),
                "suggested_decision": suggested_decision(fact),
                "rop_question": rop_question(item_type, fact),
                "source_id": fact.get("source_id"),
                "linked_open_question": fact.get("linked_open_question"),
                "risk_notes": risk_notes(fact),
            }
        )
    return sorted(queue, key=lambda item: (str(item["priority"]), str(item["topic"]), str(item["approval_item_id"])))


def should_include_in_approval_queue(fact: Mapping[str, Any], item_type: str) -> bool:
    if item_type in {
        "price",
        "discount",
        "promocode",
        "deadline",
        "camp_lvsh",
        "camp_zvsh",
        "camp_city",
        "program",
        "intensive",
        "installment",
        "tax",
        "matkap",
        "documents",
        "contact",
        "teacher",
        "refund",
        "policy",
    }:
        return True
    return bool(fact.get("requires_manager_confirmation") or fact.get("linked_open_question"))


def build_snapshot_v3(
    *,
    run_id: str,
    sources: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    approval_queue: Sequence[Mapping[str, Any]],
    post_filter: Mapping[str, Any],
    brand_rules: Mapping[str, Any],
    bot_policy: Mapping[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    chunks = build_chunks(facts)
    return {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "run_id": run_id,
        "generated_at": now,
        "metadata": {
            "purpose": "v3 KB rebuild from Claude handoff with atomic facts and strict brand separation.",
            "input_handoff": str(DEFAULT_HANDOFF_DIR),
            "autonomous_client_answer_policy": "active_brand_only_and_policy_filtered",
            "crm_tallanto_brand_detection": "not_authoritative_for_client_answer",
            "telegram_target_architecture": "two_separate_bots",
        },
        "summary": {
            "sources_total": len(sources),
            "facts_total": len(facts),
            "chunks_total": len(chunks),
            "client_allowed_facts": sum(1 for fact in facts if fact.get("allowed_for_client_answer")),
            "usable_for_precise_answer": sum(1 for fact in facts if fact.get("usable_for_precise_answer")),
            "approval_queue_items": len(approval_queue),
            "facts_by_brand": dict(Counter(str(fact.get("brand") or "") for fact in facts)),
            "facts_by_type": dict(Counter(str(fact.get("fact_type") or "") for fact in facts)),
        },
        "safety": {
            "send_client_message": False,
            "crm_write": False,
            "tallanto_write": False,
            "stable_runtime_write": False,
            "asr_run": False,
            "resolve_analyze_run": False,
            "active_brand_required_for_precise_answer": True,
            "cross_brand_client_text_forbidden": True,
            "forbidden_to_say_is_post_filter_only": True,
            "internal_only_for_number_blocks_client_text": True,
            "two_separate_telegram_bots": True,
        },
        "sources": list(sources),
        "source_registry": list(sources),
        "facts": list(facts),
        "facts_registry": list(facts),
        "chunks": chunks,
        "knowledge_chunks": chunks,
        "approval_queue": list(approval_queue),
        "post_filter": dict(post_filter),
        "brand_rules": dict(brand_rules),
        "bot_policy": dict(bot_policy),
    }


def build_chunks(facts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for fact in facts:
        text = clean_text(fact.get("client_safe_text") or fact.get("manager_check_text") or fact.get("fact_text"), max_chars=1600)
        if not text:
            continue
        chunks.append(
            {
                "schema_version": "kb_release_v3_chunk_v1",
                "chunk_id": f"kc_chunk:v3:{safe_id(fact.get('fact_id'))}",
                "source_id": fact.get("source_id"),
                "title": fact.get("title"),
                "text": text,
                "fact_types": list(fact.get("fact_types") or [fact.get("fact_type")]),
                "freshness_status": fact.get("freshness_status"),
                "bot_permission": "bot_answer_self_or_draft" if fact.get("allowed_for_client_answer") else "internal_or_manager_only",
                "forbidden_for_client": bool(fact.get("forbidden_for_client")),
                "requires_manager_confirmation": bool(fact.get("requires_manager_confirmation")),
                "usable_for_precise_answer": bool(fact.get("usable_for_precise_answer")),
                "brand": fact.get("brand"),
                "active_brand_scope": fact.get("active_brand_scope"),
                "cross_brand_policy": fact.get("cross_brand_policy"),
                "cross_brand_mixed": bool(fact.get("cross_brand_mixed")),
                "product": fact.get("product"),
                "source_title": fact.get("source_title"),
                "metadata": {
                    "fact_id": fact.get("fact_id"),
                    "source_title": fact.get("source_title"),
                    "brand": fact.get("brand"),
                    "product": fact.get("product"),
                    "route_policy": fact.get("route_policy"),
                    "source_role": "kb_release_v3_atomic_fact",
                },
            }
        )
    return chunks


def build_quality_report(snapshot: Mapping[str, Any], *, approval_queue: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    facts = [fact for fact in snapshot.get("facts", []) if isinstance(fact, Mapping)]
    sources = [source for source in snapshot.get("sources", []) if isinstance(source, Mapping)]
    source_ids = {str(source.get("source_id") or "") for source in sources}
    fact_blob = json.dumps(facts, ensure_ascii=False, sort_keys=True)
    source_orphans = sorted({str(fact.get("source_id") or "") for fact in facts if str(fact.get("source_id") or "") not in source_ids})
    empty_fact_text = [str(fact.get("fact_id")) for fact in facts if not clean_text(fact.get("fact_text"))]
    allowed_violations = [
        {
            "fact_id": fact.get("fact_id"),
            "brand": fact.get("brand"),
            "client_safe_text": fact.get("client_safe_text"),
            "reasons": client_safety_violations(str(fact.get("client_safe_text") or ""), str(fact.get("brand") or "")),
        }
        for fact in facts
        if fact.get("allowed_for_client_answer")
        and client_safety_violations(str(fact.get("client_safe_text") or ""), str(fact.get("brand") or ""))
    ]
    forbidden_to_say_hits = [
        str(fact.get("fact_id"))
        for fact in facts
        if "forbidden_to_say" in json.dumps(fact, ensure_ascii=False, sort_keys=True)
    ]
    allowed_license_hits = [
        str(fact.get("fact_id"))
        for fact in facts
        if fact.get("allowed_for_client_answer")
        and re.search(r"(Л035|50Л01|№77753|70369|АНО ДПО)", str(fact.get("client_safe_text") or ""), re.IGNORECASE)
    ]
    control_found = [number for number in CONTROL_NUMBERS if number in fact_blob]
    control_missing = [number for number in CONTROL_NUMBERS if number not in fact_blob]
    approval_counts = Counter(str(item.get("item_type") or "") for item in approval_queue)
    checks = {
        "all_fact_source_ids_exist": not source_orphans,
        "all_claude_sources_have_sha256": all(source.get("source_sha256") for source in sources if str(source.get("source_kind")) in {"claude_yaml", "claude_markdown"}),
        "control_numbers_present": not control_missing,
        "no_empty_fact_text": not empty_fact_text,
        "forbidden_to_say_not_in_facts": not forbidden_to_say_hits,
        "allowed_client_text_has_no_license_numbers": not allowed_license_hits,
        "allowed_client_text_passes_brand_safety": not allowed_violations,
        "approval_queue_has_required_columns": approval_queue_has_required_columns(approval_queue),
        "approval_queue_has_400_plus_items": len(approval_queue) >= 400,
        "approval_queue_has_business_types": required_approval_types_present(approval_counts),
        "brand_scope_has_foton_and_unpk_facts": {"foton", "unpk"} <= {str(fact.get("brand") or "") for fact in facts},
        "post_filter_has_phrases": bool((snapshot.get("post_filter") or {}).get("phrases")),
        "two_separate_bots_recorded": ((snapshot.get("safety") or {}).get("two_separate_telegram_bots") is True),
    }
    blocking_failures = [key for key, ok in checks.items() if not ok]
    return {
        "schema_version": "kb_release_v3_quality_report_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quality_passed": not blocking_failures,
        "blocking_failures": blocking_failures,
        "checks": checks,
        "summary": {
            "facts_total": len(facts),
            "client_allowed_facts": sum(1 for fact in facts if fact.get("allowed_for_client_answer")),
            "source_registry_total": len(sources),
            "approval_queue_items": len(approval_queue),
            "approval_queue_by_type": dict(approval_counts),
        },
        "control_numbers": {
            "expected": list(CONTROL_NUMBERS),
            "found": control_found,
            "missing": control_missing,
        },
        "details": {
            "orphan_source_ids": source_orphans,
            "empty_fact_text_fact_ids": empty_fact_text[:100],
            "allowed_client_text_violations": allowed_violations[:100],
            "forbidden_to_say_fact_ids": forbidden_to_say_hits[:100],
            "allowed_license_fact_ids": allowed_license_hits[:100],
        },
        "stage6": {
            "status": "not_run_by_builder",
            "note": "Сборщик готовит v3 snapshot и fixtures-compatible поля; Stage 6 запускается отдельным безопасным тестовым контуром.",
        },
    }


def approval_queue_has_required_columns(queue: Sequence[Mapping[str, Any]]) -> bool:
    required = {
        "priority",
        "approval_item_id",
        "item_type",
        "topic",
        "fact_id_ref",
        "brand",
        "product",
        "manager_text",
        "suggested_decision",
        "rop_question",
        "source_id",
        "linked_open_question",
        "risk_notes",
    }
    return bool(queue) and all(required <= set(item) for item in queue)


def required_approval_types_present(counts: Mapping[str, int]) -> bool:
    required = {"price", "discount", "promocode", "deadline", "camp_lvsh", "program", "installment", "tax", "matkap"}
    return required <= {key for key, value in counts.items() if value}


def write_outputs(
    out_root: Path,
    team_root: Path,
    *,
    snapshot: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    approval_queue: Sequence[Mapping[str, Any]],
    post_filter: Mapping[str, Any],
    quality: Mapping[str, Any],
    brand_rules: Mapping[str, Any],
    bot_policy: Mapping[str, Any],
) -> None:
    write_json(out_root / "kb_release_v3_snapshot.json", snapshot)
    write_json(out_root / "source_registry.json", {"items": list(sources)})
    write_csv(out_root / "source_registry.csv", sources)
    write_jsonl(out_root / "facts_registry.jsonl", facts)
    write_csv(out_root / "facts_registry.csv", facts)
    write_yaml(out_root / "facts_registry.yaml", {"schema_version": "facts_registry_v3", "items": list(facts)})
    write_csv(out_root / "knowledge_chunks.csv", snapshot.get("knowledge_chunks") or [])
    write_yaml(out_root / "brand_rules.yaml", brand_rules)
    write_yaml(out_root / "bot_policy.yaml", bot_policy)
    write_json(out_root / "post_filter_registry.json", post_filter)
    write_csv(out_root / "approval_queue_for_rop_v3.csv", approval_queue)
    write_json(out_root / "quality_report.json", quality)
    (out_root / "QUALITY_REPORT.md").write_text(render_quality_report_md(quality), encoding="utf-8")
    (out_root / "README.md").write_text(render_readme(snapshot, quality), encoding="utf-8")

    write_json(team_root / "kb_release_v3_snapshot.json", snapshot)
    write_yaml(team_root / "facts_registry.yaml", {"schema_version": "facts_registry_v3", "items": list(facts)})
    write_json(team_root / "source_registry.json", {"items": list(sources)})
    write_csv(team_root / "approval_queue_for_rop_v3.csv", approval_queue)
    write_json(team_root / "quality_report.json", quality)
    (team_root / "QUALITY_REPORT.md").write_text(render_quality_report_md(quality), encoding="utf-8")
    (team_root / "README_FOR_CLAUDE_AND_TEAM.md").write_text(render_team_handoff_readme(snapshot, quality), encoding="utf-8")
    (team_root / "claude_handoff_response.md").write_text(render_claude_handoff_response(snapshot, quality), encoding="utf-8")


def render_quality_report_md(quality: Mapping[str, Any]) -> str:
    lines = [
        "# QUALITY_REPORT kb_release_20260518_v3",
        "",
        f"quality_passed: `{quality.get('quality_passed')}`",
        "",
        "## Summary",
    ]
    for key, value in (quality.get("summary") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks"])
    for key, value in (quality.get("checks") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Control Numbers"])
    lines.append(f"- found: `{len((quality.get('control_numbers') or {}).get('found') or [])}`")
    missing = (quality.get("control_numbers") or {}).get("missing") or []
    lines.append(f"- missing: `{missing}`")
    lines.extend(["", "## Blocking Failures"])
    failures = quality.get("blocking_failures") or []
    lines.extend([f"- {item}" for item in failures] or ["- none"])
    lines.extend(["", "## Stage 6"])
    stage6 = quality.get("stage6") or {}
    lines.append(f"- status: `{stage6.get('status')}`")
    lines.append(f"- note: {stage6.get('note')}")
    return "\n".join(lines) + "\n"


def render_readme(snapshot: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    summary = snapshot.get("summary") or {}
    return (
        "# kb_release_20260518_v3\n\n"
        "V3 база знаний собрана из `claude_to_codex_v3_handoff_2026-05-17`.\n\n"
        "Главные правила: каждый числовой бизнес-факт развернут в отдельную запись, "
        "`forbidden_to_say` вынесен в post-filter, внутренние номера лицензий не попадают в `client_safe_text`.\n\n"
        f"- facts_total: `{summary.get('facts_total')}`\n"
        f"- client_allowed_facts: `{summary.get('client_allowed_facts')}`\n"
        f"- approval_queue_items: `{summary.get('approval_queue_items')}`\n"
        f"- quality_passed: `{quality.get('quality_passed')}`\n"
    )


def render_team_handoff_readme(snapshot: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    return (
        "# Handoff kb_release_20260518_v3\n\n"
        "Папка содержит v3 snapshot, facts_registry, source_registry, approval queue и quality report.\n\n"
        "Сборка не запускала ASR, Resolve+Analyze, live write в AMO/CRM/Tallanto и не меняла `stable_runtime`.\n\n"
        f"Quality passed: `{quality.get('quality_passed')}`.\n"
        f"Facts: `{(snapshot.get('summary') or {}).get('facts_total')}`.\n"
    )


def render_claude_handoff_response(snapshot: Mapping[str, Any], quality: Mapping[str, Any]) -> str:
    control = quality.get("control_numbers") or {}
    return (
        "# Claude handoff response\n\n"
        "Codex v3 builder rebuilt the KB from the v3 handoff folder.\n\n"
        "- Nested numeric YAML values are expanded as atomic facts.\n"
        "- Every fact source_id is present in source_registry.\n"
        "- forbidden_to_say is not imported as facts; it is stored in post_filter_registry.\n"
        "- internal_only_for_number keeps license numbers out of client_safe_text.\n"
        "- q14/q15 are represented with narrowed verified scopes.\n\n"
        f"Control numbers missing: `{control.get('missing')}`.\n"
        f"Quality passed: `{quality.get('quality_passed')}`.\n"
    )


def normalize_brand_rules(value: Mapping[str, Any]) -> dict[str, Any]:
    rules = dict(value)
    rules["schema_version"] = str(rules.get("schema_version") or "brand_rules_v3")
    rules["client_default_relationship_answer"] = "Это отдельные организации, по вашему вопросу сориентирую в рамках текущего учебного центра."
    rules["client_text_rule"] = "Клиентский текст использует только факты active_brand; CRM/Tallanto не задают бренд для клиента."
    rules["automatic_previous_cooperation_phrase_allowed"] = False
    rules["two_separate_telegram_bots"] = True
    rules["brand_values"] = ["foton", "unpk", "brand_neutral", "internal"]
    return rules


def normalize_bot_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    policy = dict(value)
    policy["schema_version"] = str(policy.get("schema_version") or "bot_policy_v3")
    policy["crm_tallanto_brand_detection_for_client_answer"] = False
    policy["telegram_active_brand_source"] = "telegram_bot_or_channel"
    policy["two_separate_telegram_bots"] = True
    policy["bot_answer_self_rollout"] = {
        "enabled_for_staff_tests": True,
        "enabled_for_loyal_prepared_clients": True,
        "enabled_for_public_traffic": False,
    }
    return policy


def attach_source_details(
    facts: Sequence[Mapping[str, Any]],
    *,
    source_lookup: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fact in facts:
        item = dict(fact)
        source = source_lookup.get(str(item.get("source_id") or ""))
        if source:
            item["source_title"] = item.get("source_title") or source.get("title") or ""
            item["source_path"] = item.get("source_path") or source.get("path") or ""
            item["source_url"] = item.get("source_url") or source.get("url") or ""
            item["source_sha256"] = item.get("source_sha256") or source.get("source_sha256") or source.get("sha256") or ""
            item["source_status"] = item.get("source_status") or source.get("source_status") or ""
        result.append(item)
    return result


def dedupe_facts(facts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for fact in facts:
        key = (
            str(fact.get("brand") or ""),
            str(fact.get("fact_key") or ""),
            str(fact.get("fact_type") or ""),
            sha256_text(str(fact.get("fact_text") or ""))[:12],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(fact))
    return result


def render_fact_text(
    path: tuple[str, ...],
    value: Any,
    *,
    brand: str,
    fact_type: str,
    structured_value: Mapping[str, Any],
) -> str:
    if "client_safe_text" in path and isinstance(value, str):
        return clean_text(value, max_chars=1400)
    label = humanize_path(path)
    brand_label = BRAND_LABELS.get(brand, brand)
    rendered_value = render_value(path, value, fact_type=fact_type, structured_value=structured_value)
    return clean_text(f"{brand_label}: {label} — {rendered_value}.", max_chars=1400)


def render_value(
    path: tuple[str, ...],
    value: Any,
    *,
    fact_type: str,
    structured_value: Mapping[str, Any],
) -> str:
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if structured_value.get("currency") == "RUB":
            return format_rub(value)
        if structured_value.get("unit") == "percent":
            return f"{format_number(value)}%"
        return format_number(value)
    text = clean_text(value, max_chars=1000)
    pct = percent_from_path(path)
    if pct is not None and text:
        return f"{pct}%: {text}"
    return text


def build_structured_value(path: tuple[str, ...], value: Any, *, fact_type: str) -> dict[str, Any]:
    structured: dict[str, Any] = {"path": ".".join(path), "raw_value": value}
    numeric = numeric_value(value)
    if numeric is not None:
        if fact_type == "price" or is_money_path(path):
            structured["amount"] = int(numeric) if numeric == int(numeric) else numeric
            structured["currency"] = "RUB"
        elif percent_from_path(path) is not None or "%" in clean_text(value):
            structured["percentage"] = numeric
            structured["unit"] = "percent"
        elif is_month_path(path):
            structured["months"] = int(numeric)
        elif is_day_path(path):
            structured["days"] = int(numeric)
        else:
            structured["number"] = int(numeric) if numeric == int(numeric) else numeric
        if path and path[-1] in {"min", "max"}:
            structured["range_bound"] = path[-1]
    pct = percent_from_path(path)
    if pct is not None:
        structured["percentage"] = pct
        structured["unit"] = "percent"
    valid_until = valid_until_from_path(path)
    if valid_until:
        structured["valid_until"] = valid_until
    classes = classes_from_path(path)
    if classes:
        structured["classes"] = classes
    fmt = format_from_path(path)
    if fmt:
        structured["format"] = fmt
    period = period_from_path(path)
    if period:
        structured["period"] = period
    return structured


def infer_fact_type(path: tuple[str, ...], value: Any) -> str:
    text = ".".join(path).casefold()
    last = path[-1].casefold() if path else ""
    if "deadline" in text or "valid_until" in text:
        return "deadline"
    if "discount" in text or "cashback" in text or "pct_" in text:
        return "discount"
    if "promo" in text or "promocode" in text:
        return "promocode"
    if "prices" in text or "pricing" in text or is_money_path(path):
        return "price"
    if "installment" in text or "rassrochka" in text or "dolyami" in text or "payment_options" in text:
        return "installment"
    if "matkap" in text:
        return "matkap"
    if "tax" in text or "deduction" in text or "вычет" in clean_text(value).casefold():
        return "tax"
    if "certificate" in text or "licenses" in text or "legal_entities" in text or "contract" in text or "document" in text:
        return "documents"
    if "refund" in text or "return" in text or "withholding" in text:
        return "refund"
    if "lvsh" in text:
        return "camp_lvsh"
    if "zvsh" in text:
        return "camp_zvsh"
    if "ls_city" in text:
        return "camp_city"
    if "intensive" in text or "intensives" in text:
        return "intensive"
    if "teacher" in text or last == "education":
        return "teacher"
    if "contact" in text or last in {"phone", "email", "telegram", "whatsapp", "website", "vk"}:
        return "contact"
    if "location" in text or "address" in text or "metro" in text:
        return "location"
    if "academic_year" in text or "schedule" in text or "start_date" in text or "dates" in text or "smeny" in text:
        return "deadline" if is_deadline_fact(path) else "program"
    if "policy" in text or "brand_rules" in text:
        return "policy"
    return "program"


def infer_product(path: tuple[str, ...], fact_type: str) -> str:
    if not path:
        return "general"
    text = ".".join(path).casefold()
    if "prices_regular_2026_27" in text:
        return "regular_courses_2026_27"
    if "individual_lessons" in text:
        return "individual_lessons"
    if "modular_courses" in text:
        return "modular_courses_m9_m11"
    if "lvsh_mendeleevo" in text:
        return "lvsh_mendeleevo_2026"
    if "zvsh_mendeleevo" in text:
        return "zvsh_mendeleevo"
    if "ls_city" in text:
        return "city_camp_2026"
    if "intensive" in text:
        return "intensives_2026"
    if "fiztech_olympiad" in text or "online_olympiad_phystech" in text:
        return "online_olympiad_phystech"
    if "preschool_patsayeva" in text:
        return "preschool_patsayeva"
    if "matkap" in text:
        return "matkap"
    if "tax" in text:
        return "tax_deduction"
    if "certificate" in text:
        return "certificates"
    if "installment" in text or "payment_options" in text:
        return "installment"
    return safe_id(path[0])


def is_deadline_fact(path: tuple[str, ...]) -> bool:
    text = ".".join(path).casefold()
    return any(marker in text for marker in ("deadline", "start_date", "start_dates", "valid_until", "dates", "smeny"))


def should_force_internal(path: tuple[str, ...], value: Any) -> bool:
    text = ".".join(path).casefold()
    if path and path[0] in INTERNAL_PATH_MARKERS:
        return True
    if path and path[0] == "licenses" and path[-1] != "client_safe_summary":
        return True
    if any(marker in text for marker in ("note_internal", "legal_entity_internal", "responsible_person_internal", "license_basis_internal")):
        return True
    if "teachers" in text or "teacher_" in text:
        return True
    value_text = clean_text(value).casefold()
    if "certificates" in text and ("lead_time_days" in text or "client_safe_text" in text):
        return True
    if "certificate_lead_time_days" in text or (
        "client_safe_text" in path and ("3 рабочих дня" in value_text or "3 рабочих дней" in value_text)
    ):
        return True
    return False


def client_allowed(
    text: str,
    *,
    brand: str,
    status: str,
    freshness: str,
    fact_type: str,
    route_policy: str,
    internal_only: bool,
    path: tuple[str, ...],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    normalized_status = normalize_status(status)
    path_text = ".".join(path).casefold()
    if internal_only:
        reasons.append("internal_only")
    if brand not in {"foton", "unpk", "brand_neutral"}:
        reasons.append("invalid_client_brand")
    if normalized_status in CLIENT_BLOCKED_STATUSES or freshness in CLIENT_BLOCKED_STATUSES:
        reasons.append(f"blocked_status:{normalized_status or freshness}")
    if "modular_courses_m9_m11" in path_text:
        reasons.append("discontinued_product")
    if "rassrochka_6_12_months" in path_text:
        reasons.append("individual_bank_terms")
    if fact_type in {"refund"} or route_policy in {"manager_only", "manager_handoff_only"}:
        reasons.append("manager_only_route")
    if fact_type in {"teacher"} and "approved_phrases" not in path_text:
        reasons.append("teacher_names_internal")
    if fact_type == "promocode" and ("teacher" in path_text or "flocktory" in path_text):
        reasons.append("promocode_needs_marketing_confirmation")
    reasons.extend(client_safety_violations(text, brand))
    if normalized_status in CLIENT_ALLOWED_STATUSES and not reasons:
        return True, []
    if normalized_status == "waiting_list" and not reasons:
        return True, []
    if normalized_status not in CLIENT_ALLOWED_STATUSES:
        reasons.append(f"not_client_allowed_status:{normalized_status}")
    return False, sorted(set(reasons))


def client_safety_violations(text: str, brand: str) -> list[str]:
    if not text:
        return []
    violations: list[str] = []
    lowered = text.casefold().replace("ё", "е")
    for pattern in GLOBAL_CLIENT_FORBIDDEN_PATTERNS:
        if pattern.casefold().replace("ё", "е") in lowered:
            violations.append(f"global_forbidden:{pattern}")
    for stale_certificate_phrase in ("3 рабочих дня", "3 рабочих дней", "тип справки", "работа / налоговая / иное"):
        if stale_certificate_phrase in lowered:
            violations.append(f"stale_certificate_phrase:{stale_certificate_phrase}")
    has_foton = any(term in lowered for term in ("фотон", "цдпо", "црдо", "cdpofoton"))
    has_unpk = any(term in lowered for term in ("унпк", "ано дпо", "ноу унпк", "kmipt"))
    if has_foton and has_unpk:
        violations.append("cross_brand_text")
    if brand == "foton":
        for term in ("унпк", "ано дпо", "ноу унпк", "kmipt.ru", "@unpk_mipt", "+7 (495) 150-81-51", "8 (800) 500-81-51"):
            if term in lowered:
                violations.append(f"other_brand_term:{term}")
    if brand == "unpk":
        for term in ("фотон", "цдпо", "црдо", "cdpofoton.ru", "edu@cdpofoton.ru", "@unpkmfti", "т-банк", "долями"):
            if term in lowered:
                violations.append(f"other_brand_term:{term}")
    return sorted(set(violations))


def infer_route_policy(path: tuple[str, ...], *, fact_type: str, status: str, internal_only: bool) -> str:
    path_text = ".".join(path).casefold()
    if internal_only:
        return "manager_handoff_only"
    if fact_type == "refund":
        return "manager_only"
    if "rassrochka_6_12_months" in path_text:
        return "manager_handoff_only"
    if normalize_status(status) in {"needs_owner_confirmation", "discontinued", "dynamic_needs_check"}:
        return "manager_handoff_only"
    if fact_type in {"price", "discount", "installment", "matkap", "tax", "documents", "camp_zvsh", "policy"}:
        return "bot_answer_self_for_pilot"
    return "draft_for_manager"


def infer_risk_level(path: tuple[str, ...], *, fact_type: str) -> str:
    text = ".".join(path).casefold()
    if fact_type == "refund" or any(marker in text for marker in ("complaint", "legal_threat", "withholding")):
        return "high"
    if fact_type in {"installment", "documents", "policy"}:
        return "medium"
    return "low"


def infer_linked_question(value: Any, path: tuple[str, ...]) -> str:
    text = f"{value or ''} {'.'.join(path)}".casefold()
    if "q14" in text or "offline_5_11_class" in text and "prices_regular" in text:
        return "q14_closed" if "unpk" in text or "prices_regular" in text else clean_text(value)
    if "q15" in text or "online_olympiad_phystech" in text:
        return "q15_closed"
    if "q11" in text or "modular_courses" in text:
        return "q11_closed"
    if "q13" in text or "zvsh" in text:
        return "q13_closed_waiting_list"
    if "q9" in text or "dolyami" in text or "rassrochka_6_12" in text:
        return "q9_closed"
    if "teacher_promo" in text or "flocktory" in text:
        return "promocodes_needs_marketing_confirmation"
    if "needs_owner_confirmation" in text or "уточнить" in text:
        return "owner_confirmation_required"
    return clean_text(value)


def normalize_status(value: Any) -> str:
    text = clean_text(value).casefold()
    mapping = {
        "verified": "verified",
        "document_verified": "document_verified",
        "fresh_verified": "fresh_verified",
        "current": "current",
        "waiting_list": "waiting_list",
        "needs_owner_confirmation": "needs_owner_confirmation",
        "dynamic_needs_check": "dynamic_needs_check",
        "discontinued": "discontinued",
        "internal_only": "internal_only",
        "false": "verified",
        "true": "verified",
    }
    return mapping.get(text, text or "verified")


def normalize_freshness(value: Any) -> str:
    status = normalize_status(value)
    mapping = {
        "verified": "document_verified",
        "current": "fresh_verified",
        "waiting_list": "waiting_list",
        "needs_owner_confirmation": "needs_owner_confirmation",
        "dynamic_needs_check": "dynamic_needs_check",
        "discontinued": "discontinued",
        "internal_only": "internal_only",
    }
    return mapping.get(status, status or "document_verified")


def normalize_brand(value: Any) -> str:
    text = clean_text(value).casefold().replace("ё", "е")
    if text in {"foton", "unpk", "brand_neutral", "internal"}:
        return text
    if "foton" in text or "фотон" in text or "цдпо" in text or "црдо" in text:
        return "foton"
    if "unpk" in text or "унпк" in text or "мфти" in text or "ано дпо" in text:
        return "unpk"
    if text in {"both", "all", "оба", "общий"}:
        return "brand_neutral"
    return "brand_neutral" if not text else "internal"


def active_brand_scope(brand: str, *, internal_only: bool = False) -> str:
    if internal_only or brand == "internal":
        return "internal_only"
    if brand == "foton":
        return "foton_bot"
    if brand == "unpk":
        return "unpk_bot"
    return "brand_neutral"


def humanize_path(path: tuple[str, ...], *, max_parts: int | None = None) -> str:
    parts = list(path[-max_parts:] if max_parts else path)
    return " / ".join(part.replace("_", " ") for part in parts)


def numeric_value(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = clean_text(value)
    match = re.search(r"(?<!\d)(\d+(?:[ \u00a0]\d{3})+|\d+(?:[.,]\d+)?)(?!\d)", text)
    if not match:
        return None
    raw = match.group(1).replace(" ", "").replace("\u00a0", "").replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def percent_from_path(path: tuple[str, ...]) -> int | None:
    for part in reversed(path):
        match = re.fullmatch(r"pct_(\d+)", part.casefold())
        if match:
            return int(match.group(1))
    return None


def is_money_path(path: tuple[str, ...]) -> bool:
    text = ".".join(path).casefold()
    markers = (
        "price",
        "pricing",
        "cashback",
        "deposit",
        "tariff",
        "semester",
        "year",
        "four_weeks",
        "lesson_",
        "session_",
        "package_",
        "tier_",
        "main_full",
        "main_min",
        "return",
        "withholding",
    )
    return any(marker in text for marker in markers)


def is_month_path(path: tuple[str, ...]) -> bool:
    return any("month" in part.casefold() for part in path)


def is_day_path(path: tuple[str, ...]) -> bool:
    return any("day" in part.casefold() or "days" in part.casefold() for part in path)


def valid_until_from_path(path: tuple[str, ...]) -> str:
    text = ".".join(path)
    match = re.search(r"before_(\d{4})_(\d{2})_(\d{2})", text)
    if not match:
        return ""
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def classes_from_path(path: tuple[str, ...]) -> str:
    text = ".".join(path).casefold()
    if "5_11" in text:
        return "5-11"
    if "3_4" in text:
        return "3-4"
    if "1_4" in text:
        return "1-4"
    if "9_and_11" in text:
        return "9 и 11"
    if "10_11" in text:
        return "10-11"
    if "8_9" in text:
        return "8-9"
    return ""


def format_from_path(path: tuple[str, ...]) -> str:
    text = ".".join(path).casefold()
    if "offline" in text:
        return "offline"
    if "online" in text:
        return "online"
    return ""


def period_from_path(path: tuple[str, ...]) -> str:
    text = ".".join(path).casefold()
    for marker in ("semester", "year", "four_weeks", "month", "week", "lesson", "session"):
        if marker in text:
            return marker
    return ""


def format_rub(value: int | float) -> str:
    return f"{format_number(value)} ₽"


def format_number(value: int | float) -> str:
    if isinstance(value, float) and value != int(value):
        return f"{value:,.2f}".replace(",", " ").replace(".00", "")
    return f"{int(value):,}".replace(",", " ")


def related_theme_ids(fact_type: str) -> list[str]:
    mapping = {
        "price": ["theme:001_pricing"],
        "discount": ["theme:005_discounts"],
        "promocode": ["theme:005_discounts"],
        "deadline": ["theme:013_schedule"],
        "camp_lvsh": ["theme:021_camps"],
        "camp_zvsh": ["theme:021_camps"],
        "camp_city": ["theme:021_camps"],
        "program": ["theme:016_program_content"],
        "intensive": ["theme:017_intensives"],
        "installment": ["theme:006_installment"],
        "tax": ["theme:008_tax_deduction"],
        "matkap": ["theme:007_matkap_payment"],
        "documents": ["theme:012_certificates"],
        "contact": ["theme:014_contacts"],
        "location": ["theme:015_location_address"],
        "teacher": ["theme:017_teachers"],
        "refund": ["theme:010_refund"],
        "policy": ["theme:000_policy"],
    }
    return mapping.get(fact_type, ["service:S2_unclear"])


def forbidden_promises_for_fact_type(fact_type: str) -> list[str]:
    base = ["Не обещать условия вне утвержденного факта."]
    if fact_type == "installment":
        base.append("Не обещать одобрение банка или неизменную конечную стоимость.")
    if fact_type in {"tax", "matkap"}:
        base.append("Не обещать одобрение СФР/ФНС или конкретный возврат денег.")
    if fact_type == "discount":
        base.append("Не суммировать скидки; применять наибольшую доступную.")
    if fact_type == "refund":
        base.append("Не объяснять клиенту расчет возврата; передать менеджеру.")
    return base


def forbidden_mentions_for_brand(brand: str) -> list[str]:
    if brand == "foton":
        return ["УНПК", "АНО ДПО", "НОУ УНПК", "kmipt.ru", "@unpk_mipt"]
    if brand == "unpk":
        return ["Фотон", "ЦДПО", "ЦРДО", "cdpofoton.ru", "Т-Банк", "Долями"]
    return []


def owner_role_for_fact(path: tuple[str, ...], fact_type: str) -> str:
    if fact_type in {"price", "discount", "promocode", "installment", "refund"}:
        return "РОП/бухгалтерия"
    if fact_type in {"tax", "matkap", "documents"}:
        return "Бухгалтерия/операционный менеджер"
    if "teacher" in ".".join(path).casefold():
        return "РОП/академический руководитель"
    return "РОП/ответственный владелец факта"


def notes_for_fact(path: tuple[str, ...], status: str, allowed: bool) -> str:
    notes: list[str] = ["Импортировано из Claude v3 handoff."]
    text = ".".join(path).casefold()
    if "certificate_lead_time_days" in text:
        notes.append("Старый срок из YAML не используется в клиентском ответе; актуальная фраза 10 дней взята из bot_policy.")
    if "modular_courses" in text:
        notes.append("Продукт discontinued; старые цены только для регрессии и аудита.")
    if status in {"needs_owner_confirmation", "discontinued", "dynamic_needs_check"} or not allowed:
        notes.append("Перед клиентским использованием нужна проверка менеджером/РОПом.")
    return " ".join(notes)


def approval_item_type(fact: Mapping[str, Any]) -> str:
    key = str(fact.get("fact_key") or "").casefold()
    fact_type = str(fact.get("fact_type") or "")
    if "deadline" in key or "start_date" in key or "dates" in key or "smeny" in key:
        return "deadline"
    if "lvsh" in key:
        return "camp_lvsh"
    if "zvsh" in key:
        return "camp_zvsh"
    return fact_type


def approval_priority(fact: Mapping[str, Any], item_type: str) -> str:
    if fact.get("risk_level") == "high" or item_type in {"price", "discount", "promocode", "installment", "tax", "matkap"}:
        return "P0"
    if fact.get("requires_manager_confirmation") or fact.get("linked_open_question"):
        return "P1"
    return "P2"


def suggested_decision(fact: Mapping[str, Any]) -> str:
    if fact.get("internal_only"):
        return "keep_internal_only"
    status = str(fact.get("verification_status") or fact.get("freshness_status") or "")
    if status in {"needs_owner_confirmation", "dynamic_needs_check"}:
        return "needs_owner_confirmation_before_client_use"
    if status == "discontinued":
        return "do_not_offer_to_client"
    if fact.get("allowed_for_client_answer"):
        return "approve_for_client_answer_after_rop_review"
    return "review_before_client_use"


def rop_question(item_type: str, fact: Mapping[str, Any]) -> str:
    if item_type == "price":
        return "Подтверждаете эту цену и область применения для бота?"
    if item_type == "discount":
        return "Подтверждаете скидку и правило, что скидки не суммируются?"
    if item_type == "promocode":
        return "Можно ли использовать этот промокод в клиентском ответе или оставить внутренним?"
    if item_type == "installment":
        return "Какую часть условия рассрочки можно говорить клиенту без ручной проверки?"
    if item_type in {"tax", "matkap", "documents"}:
        return "Подтверждаете клиентскую формулировку без раскрытия юрлица и номеров?"
    return "Можно ли использовать этот факт в ответе клиенту текущего бренда?"


def risk_notes(fact: Mapping[str, Any]) -> str:
    notes = [str(fact.get("notes") or "")]
    if fact.get("safety_block_reasons"):
        notes.append(f"safety: {fact.get('safety_block_reasons')}")
    return " ".join(part for part in notes if part).strip()


def is_cross_brand_text(text: str) -> bool:
    lowered = clean_text(text, max_chars=3000).casefold().replace("ё", "е")
    has_foton = any(marker in lowered for marker in ("фотон", "цдпо", "црдо", "cdpofoton"))
    has_unpk = any(marker in lowered for marker in ("унпк", "ано дпо", "ноу унпк", "kmipt"))
    return bool(has_foton and has_unpk)


def guard_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if "stable_runtime" in resolved.parts:
        raise ValueError("kb_release_v3 output must not be inside stable_runtime")
    return resolved


def load_yaml(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if loaded is not None else {}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (Mapping, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def clean_text(value: Any, *, max_chars: int = 500) -> str:
    text = " ".join(str(value or "").replace("\u00a0", " ").split())
    return text[:max_chars]


def safe_id(value: Any) -> str:
    text = clean_text(value, max_chars=200).casefold().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[:120] or "item"


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).casefold() in {"1", "true", "yes", "y", "да", "истина"}


if __name__ == "__main__":
    raise SystemExit(main())
