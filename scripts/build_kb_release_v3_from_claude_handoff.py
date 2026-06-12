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
BUILDER_VERSION = "kb_release_v3_builder_2026_05_20_v4"
FRESHNESS_CHECK_DATE = "2026-05-20"
DEPRECATED_DIRECT_BUILDER_WARNING = (
    "DEPRECATED: direct use of build_kb_release_v3_from_claude_handoff.py is forbidden for current KB releases; "
    "use scripts/build_kb_release_v6_1_team_answers.py with release_manifest.yaml."
)

DEFAULT_RUN_ID = "kb_release_20260520_v4"
DEFAULT_HANDOFF_DIR = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/kb_release_v3_2026-05-19")
DEFAULT_OUT_DIR = Path("product_data/knowledge_base/kb_release_20260520_v4")
DEFAULT_HANDOFF_OUT_DIR = Path("product_data/knowledge_base/kb_release_20260520_v4_handoff_for_claude_and_team")

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
        "filename": "07_OPEN_QUESTIONS_AND_GAPS.md",
        "source_id": "claude_layer_v3:open_questions",
        "kind": "claude_markdown",
        "brand": "brand_neutral",
    },
    "rebuild_requirements": {
        "filename": "UPDATE_TASK_FOR_AGENT_2026-05-19.md",
        "source_id": "claude_layer_v3:rebuild_requirements",
        "kind": "claude_markdown",
        "brand": "brand_neutral",
    },
    "changelog_final": {
        "filename": "CHANGELOG_HUMAN.md",
        "source_id": "claude_layer_v3:changelog_final",
        "kind": "claude_markdown",
        "brand": "brand_neutral",
    },
    "changelog_team_answers": {
        "filename": "UPDATE_REPORT_2026-05-19.md",
        "source_id": "claude_layer_v3:changelog_team_answers",
        "kind": "claude_markdown",
        "brand": "brand_neutral",
    },
    "readme": {
        "filename": "00_START_HERE.md",
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
CLIENT_SAFE_PATH_MARKERS = ("refund_presale_policy",)
MANIFEST_MANUAL_DECISION_FACT_OVERRIDES: tuple[Mapping[str, Any], ...] = ()
MANIFEST_STRUCTURED_METADATA_RULES: tuple[Mapping[str, Any], ...] = ()
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
CONFIRMATION_STATUSES = {"needs_owner_confirmation", "dynamic_needs_check"}
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
    "course_parameter": "19_course_parameters",
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
    "раньше сотрудничали",
    "были одно",
    "наш партнёр",
    "наш партнер",
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

NON_MONEY_NUMERIC_PATH_MARKERS = (
    "total_lessons",
    "weekly_lessons",
    "daily_hours",
    "semester_1_weeks",
    "semester_2_weeks",
    "daily_pairs",
    "pair_duration_minutes",
    "duration_weeks",
    "experience_years",
    "classes",
    "programs",
    "schedule",
    "start",
    "signup_deadline",
    "payment_deadline",
    "retroactive_years",
    "lead_time_days",
)
MONEY_LEAF_KEYS = {
    "semester",
    "year",
    "four_weeks",
    "four_weeks_new",
    "lesson",
    "session",
    "package",
    "individual",
    "group",
    "main_full",
    "main_min",
    "online",
    "offline",
    "cashback",
    "deposit",
    "price",
    "amount",
    "min",
    "max",
}


def main(argv: Sequence[str] | None = None) -> int:
    print(DEPRECATED_DIRECT_BUILDER_WARNING, file=sys.stderr)
    parser = argparse.ArgumentParser(description="Deprecated direct builder. Use build_kb_release_v6_1_team_answers.py.")
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
    facts = remove_manifest_deleted_facts(facts)
    facts.extend(build_manual_decision_facts(source_lookup))
    facts = attach_source_details(dedupe_facts(facts), source_lookup=source_lookup)
    facts = ensure_fact_refresh_dates(facts)
    facts = enrich_phase2_structured_metadata(facts)
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
        path = resolve_handoff_source_path(handoff_root, meta["filename"])
        if meta["filename"].endswith((".yaml", ".yml")):
            payload[key] = load_yaml(path)
        else:
            payload[key] = path.read_text(encoding="utf-8") if path.exists() else ""
    return payload


def build_source_registry(handoff_root: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for key, meta in SOURCE_FILES.items():
        path = resolve_handoff_source_path(handoff_root, meta["filename"])
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


def resolve_handoff_source_path(handoff_root: Path, filename: str) -> Path:
    direct = handoff_root / filename
    if direct.exists():
        return direct
    nested_facts = handoff_root / "facts" / filename
    if nested_facts.exists():
        return nested_facts
    return direct


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
    if should_skip_fact_path(path, value):
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
        if range_mode and len(value) >= 2 and numeric_value(value[0]) is not None and numeric_value(value[1]) is not None:
            return [make_fact(path=path, value={"min": value[0], "max": value[1]}, context=context, source=source)]
        for index, item in enumerate(value):
            suffix = ("min" if index == 0 else "max") if range_mode and index < 2 else str(index + 1)
            records.extend(walk_value(item, path=(*path, suffix), context=context, source=source))
        return records

    normalized_value = normalize_value_for_fact(path, value)
    return [make_fact(path=path, value=normalized_value, context=context, source=source)]


def should_skip_fact_path(path: tuple[str, ...], value: Any) -> bool:
    text = f"{'.'.join(path)} {clean_text(value, max_chars=1000)}".casefold().replace("ё", "е")
    if "долями плюс" in text and "3/6/10" in text:
        return True
    is_container = isinstance(value, Mapping) or (
        isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
    )
    if not is_container and (
        "certificate_lead_time_days" in ".".join(path).casefold()
        or "3 рабочих дня" in text
        or "3 рабочих дней" in text
    ):
        return True
    if "тип справки" in text or "работа / налоговая / иное" in text:
        return True
    return False


def normalize_value_for_fact(path: tuple[str, ...], value: Any) -> Any:
    text = ".".join(path).casefold()
    if "brand_link_question.approved_response" in text:
        return "Это отдельные организации, по вашему вопросу сориентирую в рамках текущего учебного центра."
    return value


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
    if any(marker in normalized_path for marker in CLIENT_SAFE_PATH_MARKERS):
        return False
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
    product = canonical_product_key(path, product)
    internal_only = bool(context.get("internal_only") or should_force_internal(path, value))
    linked_open_question = infer_linked_question(context.get("linked_open_question"), path)
    if linked_open_question == "q14_closed" and brand != "unpk":
        linked_open_question = ""
    structured_value = build_structured_value(path, value, fact_type=fact_type)
    structured_value["freshness_check_date"] = FRESHNESS_CHECK_DATE
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
    bot_template_required = bool(allowed and direct_text_requires_template(fact_text, fact_type=fact_type))
    usable_precise = bool(allowed and freshness not in {"dynamic_needs_check", "waiting_list"} and status != "waiting_list")
    requires_confirmation = requires_manager_confirmation_for_fact(
        status=status,
        freshness=freshness,
        route_policy=route_policy,
        safety_reasons=safety_reasons,
    )
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
        "manager_display_text": fact_text,
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
        "freshness_check_date": FRESHNESS_CHECK_DATE,
        "verified_by": "",
        "verified_at": "",
        "owner_role": owner_role_for_fact(path, fact_type),
        "allowed_for_client_answer": allowed,
        "usable_for_precise_answer": usable_precise,
        "bot_template_required": bot_template_required,
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
                if str(theme_key) == "theme_12_certificate":
                    text = "Менеджер подготовит справку и пришлёт в течение 10 дней, постараемся раньше."
                    structured_value = {"days": 10, "raw_value": {"lead_time_days": 10}}
                else:
                    structured_value = None
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
                        structured_value=structured_value,
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
                    requires_manager_confirmation=False,
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
    specs_by_key = {str(spec.get("fact_key") or ""): dict(spec) for spec in manual_specs}
    removed = manifest_removed_fact_keys()
    for brand, fact_key in removed:
        spec = specs_by_key.get(fact_key)
        if spec and normalize_brand(str(spec.get("brand") or "")) == brand:
            specs_by_key.pop(fact_key, None)
    for override in MANIFEST_MANUAL_DECISION_FACT_OVERRIDES:
        if not isinstance(override, Mapping):
            continue
        if truthy(override.get("remove_from_release")):
            continue
        fact_key = str(override.get("fact_key") or "").strip()
        source_key = str(override.get("source_key") or "").strip()
        if not fact_key or not source_key or source_key not in source_lookup:
            continue
        spec = dict(override)
        spec.pop("source_key", None)
        spec["source"] = source_lookup[source_key]
        specs_by_key[fact_key] = spec
    return [make_manual_fact(**spec) for spec in specs_by_key.values()]


def manifest_removed_fact_keys() -> set[tuple[str, str]]:
    removed: set[tuple[str, str]] = set()
    for override in MANIFEST_MANUAL_DECISION_FACT_OVERRIDES:
        if not isinstance(override, Mapping) or not truthy(override.get("remove_from_release")):
            continue
        fact_key = str(override.get("fact_key") or "").strip()
        brand = normalize_brand(str(override.get("brand") or ""))
        if fact_key and brand:
            removed.add((brand, fact_key))
    return removed


def remove_manifest_deleted_facts(facts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    removed = manifest_removed_fact_keys()
    if not removed:
        return [dict(fact) for fact in facts]
    return [
        dict(fact)
        for fact in facts
        if (normalize_brand(str(fact.get("brand") or "")), str(fact.get("fact_key") or "")) not in removed
    ]


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
    structured = dict(structured_value or {})
    structured.setdefault("path", fact_key)
    structured["freshness_check_date"] = FRESHNESS_CHECK_DATE
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
        "manager_display_text": fact_text,
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
        "structured_value": structured,
        "valid_from": "",
        "valid_until": "",
        "freshness_check_date": FRESHNESS_CHECK_DATE,
        "verified_by": "",
        "verified_at": "",
        "owner_role": owner_role_for_fact(tuple(fact_key.split(".")), fact_type),
        "allowed_for_client_answer": allowed,
        "usable_for_precise_answer": bool(allowed and freshness not in {"dynamic_needs_check", "waiting_list"}),
        "requires_manager_confirmation": bool(
            requires_manager_confirmation
            if requires_manager_confirmation is not None
            else requires_manager_confirmation_for_fact(
                status=status,
                freshness=freshness,
                route_policy=route_policy,
                safety_reasons=safety_reasons,
            )
        ),
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
    global_unique, descriptions = normalize_filter_phrases(collect_global_forbidden_phrases(handoff))
    foton_unique, foton_descriptions = normalize_filter_phrases(collect_brand_forbidden_phrases(handoff, "foton"))
    unpk_unique, unpk_descriptions = normalize_filter_phrases(collect_brand_forbidden_phrases(handoff, "unpk"))
    descriptions = sorted({*descriptions, *foton_descriptions, *unpk_descriptions})
    regex_patterns = [
        r"(?<!не\s)скидки\s+суммируются",
        r"\bоплат(?:ите|и)\s+(?:сейчас|сразу|сегодня)\b",
        r"\bгарантир(?:уем|ую|уется|овать)\s+(?:поступление|результат|место|зачисление)\b",
    ]
    return {
        "schema_version": "kb_release_v3_post_filter_v2",
        "source": "Claude v3 handoff forbidden_to_say + brand rules + bot policy",
        "violation_action": "manager_only",
        "violation_flag": "brand_separation_violation",
        "matcher_fields": ["phrases", "global_phrases", "phrases_by_active_brand", "regex_patterns"],
        "human_only_fields": ["pattern_descriptions"],
        "phrases": global_unique,
        "phrases_total": len(global_unique),
        "global_phrases": global_unique,
        "global_phrases_total": len(global_unique),
        "phrases_by_active_brand": {
            "foton": foton_unique,
            "unpk": unpk_unique,
        },
        "phrases_by_active_brand_total": {
            "foton": len(foton_unique),
            "unpk": len(unpk_unique),
        },
        "regex_patterns": regex_patterns,
        "regex_patterns_total": len(regex_patterns),
        "pattern_descriptions": descriptions,
        "pattern_descriptions_total": len(descriptions),
    }


def collect_global_forbidden_phrases(handoff: Mapping[str, Any]) -> list[str]:
    phrases: list[str] = []
    bot_policy = handoff.get("bot_policy")
    if isinstance(bot_policy, Mapping):
        post_filter = bot_policy.get("post_filter_draft_text")
        if isinstance(post_filter, Mapping):
            phrases.extend(flatten_scalars(post_filter.get("forbidden_in_any_brand")))
            phrases.extend(flatten_scalars(post_filter.get("examples_blocked")))
    collect_forbidden_phrases(handoff.get("facts_internal_only"), phrases)
    brand_rules = handoff.get("brand_rules")
    if isinstance(brand_rules, Mapping):
        brand_rules_without_active_blocks = {
            key: value for key, value in brand_rules.items() if str(key) != "forbidden_client_mentions"
        }
        collect_forbidden_phrases(brand_rules_without_active_blocks, phrases)
    return phrases


def collect_brand_forbidden_phrases(handoff: Mapping[str, Any], active_brand: str) -> list[str]:
    phrases: list[str] = []
    brand_rules = handoff.get("brand_rules")
    if not isinstance(brand_rules, Mapping):
        brand_rules = {}
    mentions = brand_rules.get("forbidden_client_mentions")
    if isinstance(mentions, Mapping):
        active_block = mentions.get(f"when_active_brand_is_{active_brand}")
        if isinstance(active_block, Mapping):
            phrases.extend(flatten_scalars(active_block.get("blocked_terms")))
    source_key = "facts_for_bot_FOTON" if active_brand == "foton" else "facts_for_bot_UNPK"
    collect_forbidden_phrases(handoff.get(source_key), phrases)
    bot_policy = handoff.get("bot_policy")
    if isinstance(bot_policy, Mapping):
        theme_routes = bot_policy.get("theme_routes")
        if isinstance(theme_routes, Mapping):
            for route in theme_routes.values():
                if isinstance(route, Mapping):
                    collect_forbidden_phrases(route.get(f"{active_brand}_specific"), phrases)
        post_filter = bot_policy.get("post_filter_draft_text")
        if isinstance(post_filter, Mapping):
            collect_forbidden_phrases(post_filter.get(f"forbidden_when_active_brand_{active_brand}"), phrases)
    return phrases


def normalize_filter_phrases(phrases: Sequence[str]) -> tuple[list[str], list[str]]:
    literals: list[str] = []
    pattern_descriptions: list[str] = []
    for phrase in phrases:
        normalized = clean_text(phrase, max_chars=500)
        if not normalized:
            continue
        split = split_matchable_filter_phrase(normalized)
        if split:
            literals.extend(split)
        elif is_filter_pattern_description(normalized):
            pattern_descriptions.append(normalized)
        else:
            literals.append(normalized)
    return sorted({phrase for phrase in literals if phrase}), sorted({phrase for phrase in pattern_descriptions if phrase})


def split_matchable_filter_phrase(phrase: str) -> list[str]:
    lowered = phrase.casefold().replace("ё", "е")
    if lowered == "у них есть места / цены / условия":
        return ["У них есть места", "У них есть цены", "У них есть условия"]
    return []


def is_filter_pattern_description(phrase: str) -> bool:
    lowered = phrase.casefold().replace("ё", "е")
    return lowered.startswith("любое ") or lowered in {"работа / налоговая / иное"}


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
        decision = suggested_decision(fact)
        priority = approval_priority(fact, item_type, decision)
        queue.append(
            {
                "priority": priority,
                "approval_item_id": f"approve:{safe_id(fact_id)}",
                "item_type": item_type,
                "topic": topic,
                "fact_id_ref": fact_id,
                "brand": fact.get("brand"),
                "product": fact.get("product"),
                "manager_text": fact.get("manager_check_text") or fact.get("fact_text"),
                "suggested_decision": decision,
                "rop_question": rop_question(item_type, fact, decision),
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
        "course_parameter",
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
    invalid_weekly_frequency_facts = [
        str(fact.get("fact_id"))
        for fact in facts
        if invalid_weekly_frequency_fact(fact)
    ]
    raw_text_number_grounding_findings = [
        finding
        for fact in facts
        for finding in text_number_grounding_findings_for_fact(fact)
    ]
    grounded_number_index = grounded_number_index_for_facts(facts)
    text_number_grounding_findings: list[dict[str, Any]] = []
    text_number_global_match_suspects: list[dict[str, Any]] = []
    for finding in raw_text_number_grounding_findings:
        matches = same_brand_global_fact_matches(finding, grounded_number_index)
        if matches:
            suspect = dict(finding)
            suspect["reason"] = "global_fact_match_requires_review"
            suspect["matched_fact_keys"] = matches[:5]
            text_number_global_match_suspects.append(suspect)
        else:
            text_number_grounding_findings.append(finding)
    field_range_findings = [
        finding
        for fact in facts
        for finding in field_range_findings_for_fact(fact)
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
        "weekly_frequency_is_plausible": not invalid_weekly_frequency_facts,
        "text_number_grounded": not text_number_grounding_findings,
        "field_ranges_ok": not field_range_findings,
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
            "invalid_weekly_frequency_fact_ids": invalid_weekly_frequency_facts[:100],
            "text_number_grounding_findings": text_number_grounding_findings[:100],
            "text_number_global_match_suspects": text_number_global_match_suspects[:100],
            "field_range_findings": field_range_findings[:100],
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


def invalid_weekly_frequency_fact(fact: Mapping[str, Any]) -> bool:
    key = str(fact.get("fact_key") or "").casefold()
    if "weekly_lessons" not in key:
        return False
    text = " ".join(
        str(fact.get(field) or "")
        for field in ("fact_text", "client_safe_text", "manager_check_text", "manager_display_text")
    )
    if re.search(r"\b20\d{2}\s+раз(?:а)?\s+в\s+недел", text, re.I):
        return True
    structured = fact.get("structured_value")
    if isinstance(structured, Mapping):
        weeks = numeric_value(structured.get("weeks"))
        if weeks is not None and (weeks < 1 or weeks > 7):
            return True
        count = numeric_value(structured.get("count"))
        unit = str(structured.get("unit") or "").casefold()
        if unit == "lessons" and count is not None and count > 20:
            return True
    return False


_MONEY_CLAIM_RE = re.compile(r"(?<!\d)(\d[\d \u00a0]{2,})(?:\s*(?:₽|руб\.?|рубл(?:ей|я|ь)?))", re.I)
_PERCENT_CLAIM_RE = re.compile(r"(?<!\d)(\d{1,3}(?:[.,]\d+)?)\s*%")
_WEEKLY_FREQUENCY_CLAIM_RE = re.compile(r"(?<!\d)(\d{1,4})\s+раз(?:а)?\s+в\s+недел", re.I)
_MONTH_CLAIM_RE = re.compile(r"(?<!\d)(\d{1,3})\s+месяц", re.I)
_DAY_CLAIM_RE = re.compile(r"(?<!\d)(\d{1,3})\s+(?:рабоч(?:их|ие)\s+)?дн", re.I)
_MINUTE_CLAIM_RE = re.compile(r"(?<!\d)(\d{1,3})\s+мин", re.I)


def text_number_grounding_findings_for_fact(fact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Find client-visible business numbers that are not grounded in this fact.

    This is intentionally conservative: dates, classes, phones, URLs and bare
    academic years are ignored here. They are useful context but too noisy for a
    blocking build gate. Business numbers are blocked only when the same fact's
    structured_value or raw_value cannot support the numeric claim.
    """

    if not fact.get("allowed_for_client_answer"):
        return []
    structured = fact.get("structured_value")
    structured_map = structured if isinstance(structured, Mapping) else {}
    raw_value = structured_map.get("raw_value", fact.get("value"))
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for field in ("client_safe_text", "fact_text"):
        text = clean_text(fact.get(field) or "")
        if not text:
            continue
        for claim in business_number_claims(text):
            if claim_supported_by_same_fact(claim, structured_map, raw_value, path=str(fact.get("fact_key") or "")):
                continue
            key = (field, str(claim["kind"]), str(claim["normalized"]))
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                {
                    "fact_id": fact.get("fact_id"),
                    "fact_key": fact.get("fact_key"),
                    "brand": fact.get("brand"),
                    "field": field,
                    "kind": claim["kind"],
                    "claim": claim["text"],
                    "value": claim["normalized"],
                    "reason": "number_not_grounded_in_same_fact",
                }
            )
    return findings


def grounded_number_index_for_facts(facts: Sequence[Mapping[str, Any]]) -> Mapping[tuple[str, str, str], frozenset[str]]:
    index: dict[tuple[str, str, str], set[str]] = {}
    for fact in facts:
        if not fact.get("allowed_for_client_answer"):
            continue
        structured = fact.get("structured_value")
        structured_map = structured if isinstance(structured, Mapping) else {}
        raw_value = structured_map.get("raw_value", fact.get("value"))
        fact_key = str(fact.get("fact_key") or "")
        brand = str(fact.get("brand") or "").casefold()
        for field in ("client_safe_text", "fact_text"):
            text = clean_text(fact.get(field) or "")
            if not text:
                continue
            for claim in business_number_claims(text):
                if not claim_supported_by_same_fact(claim, structured_map, raw_value, path=fact_key):
                    continue
                key = (brand, str(claim.get("kind") or ""), str(claim.get("normalized") or ""))
                index.setdefault(key, set()).add(fact_key)
    return {key: frozenset(values) for key, values in index.items()}


def same_brand_global_fact_matches(
    finding: Mapping[str, Any],
    grounded_number_index: Mapping[tuple[str, str, str], frozenset[str]],
) -> list[str]:
    key = (
        str(finding.get("brand") or "").casefold(),
        str(finding.get("kind") or ""),
        str(finding.get("value") or ""),
    )
    current_fact_key = str(finding.get("fact_key") or "")
    return sorted(fact_key for fact_key in grounded_number_index.get(key, frozenset()) if fact_key != current_fact_key)


def business_number_claims(text: str) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    for kind, regex in (
        ("money", _MONEY_CLAIM_RE),
        ("percentage", _PERCENT_CLAIM_RE),
        ("weekly_frequency", _WEEKLY_FREQUENCY_CLAIM_RE),
        ("months", _MONTH_CLAIM_RE),
        ("days", _DAY_CLAIM_RE),
        ("minutes", _MINUTE_CLAIM_RE),
    ):
        for match in regex.finditer(str(text or "")):
            normalized = normalize_numeric_claim(match.group(1))
            if normalized is None:
                continue
            claims.append({"kind": kind, "text": match.group(0), "normalized": normalized})
    return claims


def normalize_numeric_claim(value: object) -> float | None:
    raw = str(value or "").replace(" ", "").replace("\u00a0", "").replace(",", ".")
    try:
        number = float(raw)
    except ValueError:
        return None
    return int(number) if number == int(number) else number


def claim_supported_by_same_fact(
    claim: Mapping[str, Any],
    structured: Mapping[str, Any],
    raw_value: object,
    *,
    path: str = "",
) -> bool:
    kind = str(claim.get("kind") or "")
    value = normalize_numeric_claim(claim.get("normalized"))
    if value is None:
        return True
    if claim_supported_by_structured_value(kind, value, structured):
        return True
    if claim_supported_by_path(kind, value, path):
        return True
    return claim_supported_by_raw_value(kind, value, raw_value)


def claim_supported_by_structured_value(kind: str, value: float, structured: Mapping[str, Any]) -> bool:
    if not structured:
        return False
    candidate_keys_by_kind = {
        "money": ("amount", "amount_min", "amount_max", "limit", "price"),
        "percentage": ("percentage",),
        "weekly_frequency": ("weeks",),
        "months": ("months",),
        "days": ("days",),
    }
    if kind == "minutes":
        return str(structured.get("unit") or "").casefold() == "minutes" and _same_number(value, structured.get("count"))
    for key in candidate_keys_by_kind.get(kind, ()):
        if _same_number(value, structured.get(key)):
            return True
    return False


def claim_supported_by_path(kind: str, value: float, path: str) -> bool:
    text = str(path or "").casefold()
    if kind == "minutes":
        return bool(re.search(rf"(?<!\d){re.escape(format_number_for_match(value))}\s*min\b", text, re.I))
    return False


def claim_supported_by_raw_value(kind: str, value: float, raw_value: object) -> bool:
    if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        return _same_number(value, raw_value)
    return claim_supported_by_raw_text(kind, value, clean_text(raw_value or ""))


def claim_supported_by_raw_text(kind: str, value: float, raw_text: str) -> bool:
    text = clean_text(raw_text).casefold().replace("\u00a0", " ")
    if not text:
        return False
    for claim in business_number_claims(text):
        if str(claim.get("kind") or "") == kind and _same_number(value, claim.get("normalized")):
            return True
    number = format_number_for_match(value)
    if kind == "money":
        return bool(re.search(rf"(?<!\d){re.escape(number)}(?:\s*(?:₽|руб|рубл))", text, re.I))
    if kind == "percentage":
        return bool(re.search(rf"(?<!\d){re.escape(number)}\s*%", text, re.I))
    if kind == "weekly_frequency":
        return bool(re.search(rf"(?<!\d){re.escape(number)}\s+раз(?:а)?\s+в\s+недел", text, re.I))
    if kind == "months":
        return bool(re.search(rf"(?<!\d){re.escape(number)}\s+месяц", text, re.I))
    if kind == "days":
        return bool(re.search(rf"(?<!\d){re.escape(number)}\s+(?:рабоч(?:их|ие)\s+)?дн", text, re.I))
    if kind == "minutes":
        return bool(re.search(rf"(?<!\d){re.escape(number)}\s+мин", text, re.I))
    return False


def format_number_for_match(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value).replace(".", ",")


def _same_number(left: object, right: object) -> bool:
    left_num = normalize_numeric_claim(left)
    right_num = normalize_numeric_claim(right)
    return left_num is not None and right_num is not None and abs(float(left_num) - float(right_num)) < 0.0001


def field_range_findings_for_fact(fact: Mapping[str, Any]) -> list[dict[str, Any]]:
    structured = fact.get("structured_value")
    if not isinstance(structured, Mapping):
        return []
    findings: list[dict[str, Any]] = []
    fact_id = fact.get("fact_id")
    fact_key = str(fact.get("fact_key") or "")
    path = str(structured.get("path") or fact_key)

    def add(field: str, value: object, reason: str) -> None:
        findings.append({"fact_id": fact_id, "fact_key": fact_key, "field": field, "value": value, "reason": reason})

    weeks = numeric_value(structured.get("weeks"))
    if weeks is not None:
        upper = 7 if "weekly_lessons" in path.casefold() else 52
        if weeks < 1 or weeks > upper:
            add("weeks", structured.get("weeks"), f"expected_1_to_{upper}")
    percentage = numeric_value(structured.get("percentage"))
    if percentage is not None and (percentage < 0 or percentage > 100):
        add("percentage", structured.get("percentage"), "expected_0_to_100")
    months = numeric_value(structured.get("months"))
    if months is not None and (months < 1 or months > 120):
        add("months", structured.get("months"), "expected_1_to_120")
    days = numeric_value(structured.get("days"))
    if days is not None and (days < 1 or days > 365):
        add("days", structured.get("days"), "expected_1_to_365")
    unit = str(structured.get("unit") or "").casefold()
    count = numeric_value(structured.get("count"))
    if count is not None:
        if unit == "minutes" and (count < 20 or count > 300):
            add("count", structured.get("count"), "minutes_expected_20_to_300")
        elif unit == "hours" and (count < 1 or count > 24):
            add("count", structured.get("count"), "hours_expected_1_to_24")
        elif unit == "pairs" and (count < 1 or count > 20):
            add("count", structured.get("count"), "pairs_expected_1_to_20")
        elif unit == "lessons":
            upper = 7 if "weekly_lessons" in path.casefold() else 500
            if count < 1 or count > upper:
                add("count", structured.get("count"), f"lessons_expected_1_to_{upper}")
    for field in ("amount", "amount_min", "amount_max"):
        amount = numeric_value(structured.get(field))
        if amount is not None and amount <= 0:
            add(field, structured.get(field), "money_expected_positive")
    classes = structured.get("classes")
    if isinstance(classes, Sequence) and not isinstance(classes, (str, bytes, bytearray)):
        for item in classes:
            cls = numeric_value(item)
            if cls is not None and (cls < 1 or cls > 11):
                add("classes", item, "class_expected_1_to_11")
    return findings


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
    write_jsonl(team_root / "facts_registry.jsonl", facts)
    write_csv(team_root / "facts_registry.csv", facts)
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


def ensure_fact_refresh_dates(facts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fact in facts:
        item = dict(fact)
        structured_value = item.get("structured_value")
        structured = dict(structured_value) if isinstance(structured_value, Mapping) else {}
        path_text = str(structured.get("path") or item.get("fact_key") or "")
        path = tuple(part for part in path_text.split(".") if part)
        valid_until = str(item.get("valid_until") or structured.get("valid_until") or "").strip()
        if not valid_until:
            valid_until = valid_until_from_path(path)
        structured["valid_until"] = valid_until
        item["valid_until"] = valid_until
        item["structured_value"] = structured
        result.append(item)
    return result


def enrich_phase2_structured_metadata(facts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fact in facts:
        item = dict(fact)
        fact_key = str(item.get("fact_key") or "")
        brand = str(item.get("brand") or "")
        structured = dict(item.get("structured_value") or {})
        applies_to = dict(structured.get("applies_to") or {})

        for rule in MANIFEST_STRUCTURED_METADATA_RULES:
            if not _structured_metadata_rule_matches(rule, fact_key=fact_key, brand=brand):
                continue
            rule_applies_to = rule.get("applies_to") if isinstance(rule, Mapping) else {}
            if isinstance(rule_applies_to, Mapping):
                for key, value in rule_applies_to.items():
                    applies_to.setdefault(str(key), value)
            for key, value in rule.items():
                if key in {"brand", "fact_key", "fact_key_prefix", "applies_to"}:
                    continue
                structured.setdefault(str(key), value)

        if applies_to:
            structured["applies_to"] = applies_to
            item["structured_value"] = structured
        result.append(item)
    return result


def _structured_metadata_rule_matches(rule: Mapping[str, Any], *, fact_key: str, brand: str) -> bool:
    rule_brand = str(rule.get("brand") or "").strip()
    if rule_brand and rule_brand != brand:
        return False
    exact_key = str(rule.get("fact_key") or "").strip()
    if exact_key:
        return fact_key == exact_key
    prefix = str(rule.get("fact_key_prefix") or "").strip()
    return bool(prefix and fact_key.startswith(prefix))


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
    if any(part == "client_safe_text" or part.endswith("_client_safe_text") for part in path) and isinstance(value, str):
        return clean_text(value, max_chars=1400)
    contextual = render_contextual_fact_text(
        path,
        value,
        brand=brand,
        fact_type=fact_type,
        structured_value=structured_value,
    )
    if contextual:
        return contextual
    label = readable_fact_label(path, fact_type=fact_type)
    brand_label = BRAND_LABELS.get(brand, brand)
    rendered_value = render_value(path, value, fact_type=fact_type, structured_value=structured_value)
    return clean_text(f"{brand_label}: {label} — {rendered_value}.", max_chars=1400)


def render_contextual_fact_text(
    path: tuple[str, ...],
    value: Any,
    *,
    brand: str,
    fact_type: str,
    structured_value: Mapping[str, Any],
) -> str:
    text = ".".join(path).casefold()
    brand_label = BRAND_LABELS.get(brand, brand)
    raw_value = structured_value.get("raw_value", value)
    rendered_value = render_value(path, value, fact_type=fact_type, structured_value=structured_value)
    pct_value = structured_value.get("percentage") or numeric_value(raw_value)
    pct_text = f"{format_number(pct_value)}%" if pct_value is not None else rendered_value

    if "licenses.client_safe_summary" in text:
        return f"{brand_label}: у учебного центра есть лицензия на образовательную деятельность."

    objection = render_objection_response_text(path, raw_value, brand_label=brand_label)
    if objection:
        return objection

    social = render_social_proof_text(path, raw_value, brand_label=brand_label)
    if social:
        return social

    discount = render_discount_text(path, raw_value, rendered_value=rendered_value, brand_label=brand_label)
    if discount:
        return discount

    intensive = render_intensive_text(path, raw_value, brand_label=brand_label)
    if intensive:
        return intensive

    academic = render_academic_year_text(path, raw_value, brand_label=brand_label)
    if academic:
        return academic

    lvsh = render_lvsh_text(path, raw_value, rendered_value=rendered_value, brand_label=brand_label)
    if lvsh:
        return lvsh
    if "zvsh_mendeleevo.dates_2026_27" in text:
        return f"{brand_label}: даты ЗВШ Менделеево на 2026/27 учебный год пока не определены."

    if "installment.services.dolyami.parts" in text:
        return f"{brand_label}: сервис «Долями» делит оплату на {format_number(numeric_value(raw_value) or 4)} равные части."
    if "installment.services.dolyami.interest_for_client" in text:
        return f"{brand_label}: при оплате через «Долями» для клиента указана переплата 0%."
    if text.endswith("installment.services.rassrochka.term_months"):
        return f"{brand_label}: срок рассрочки может составлять {rendered_value} месяцев."
    if "installment.services.rassrochka.limit" in text:
        return f"{brand_label}: лимит рассрочки по текущим данным — {rendered_value}."
    if "installment.services.rassrochka.signing" in text:
        return f"{brand_label}: подписание документов по рассрочке проходит по СМС."
    if "installment.provider" in text:
        return f"{brand_label}: рассрочка оформляется через Т-Банк; перед ответом клиенту менеджер проверяет актуальные условия."
    if "installment.refresh_frequency" in text:
        return f"{brand_label}: условия рассрочки через Т-Банк нужно обновлять и перепроверять минимум раз в квартал."
    if "installment.services.dolyami.best_for" in text:
        return f"{brand_label}: сервис «Долями» подходит для коротких программ, например интенсивов и летних школ."
    if "installment.services.rassrochka.decision_time" in text:
        return f"{brand_label}: предварительное решение по рассрочке обычно занимает {rendered_value}; перед обещанием срока нужна проверка."
    if "installment.services.rassrochka.no_interest_for_client" in text:
        return f"{brand_label}: рассрочка указана как вариант без переплаты для клиента, но условия нужно проверить перед оформлением."

    if "tax_deduction.certificate_form" in text:
        return f"{brand_label}: для налогового вычета используется справка по форме {clean_text(raw_value)}."
    if "tax_deduction.limits.2024_and_later.max_expenses_per_child" in text:
        return f"{brand_label}: по расходам с 2024 года лимит расходов на обучение ребёнка для налогового вычета — {format_rub(numeric_value(raw_value) or 110000)}."
    if "tax_deduction.limits.2024_and_later.max_return_per_child" in text:
        return f"{brand_label}: по расходам с 2024 года можно вернуть до {format_rub(numeric_value(raw_value) or 14300)} за ребёнка в год."
    if "tax_deduction.limits.2023_and_earlier.max_expenses_per_child" in text:
        return f"{brand_label}: по расходам за 2023 год и раньше лимит расходов на обучение ребёнка для налогового вычета — {format_rub(numeric_value(raw_value) or 50000)}."
    if "tax_deduction.limits.2023_and_earlier.max_return_per_child" in text:
        return f"{brand_label}: по расходам за 2023 год и раньше можно вернуть до {format_rub(numeric_value(raw_value) or 6500)} за ребёнка в год."
    if "tax_deduction.retroactive_years" in text:
        return f"{brand_label}: налоговый вычет можно оформить за последние {format_number(numeric_value(raw_value) or 3)} года."
    if "tax_deduction.fns_review_months" in text:
        return f"{brand_label}: налоговая обычно рассматривает декларацию до {format_number(numeric_value(raw_value) or 3)} месяцев."
    if "tax_deduction.payment_after_approval_months" in text:
        return f"{brand_label}: после одобрения налоговой выплата обычно занимает до {format_number(numeric_value(raw_value) or 1)} месяца."

    if "payment_options.available_schedules" in text and "discount_extra" in text:
        if ".monthly." in text:
            return f"{brand_label}: при помесячной оплате дополнительная скидка не применяется."
        if ".semester." in text:
            return f"{brand_label}: при оплате за семестр действует дополнительная скидка {rendered_value}."
        if ".year." in text:
            return f"{brand_label}: при оплате за год действует дополнительная скидка {rendered_value}."

    if "prices_regular_2026_27.online_5_11_class_regular" in text:
        if text.endswith(".semester"):
            return f"{brand_label}: онлайн-курсы для 5-11 классов, формат 2 раза в неделю по 90 минут, семестр — {rendered_value}."
        if text.endswith(".year"):
            return f"{brand_label}: онлайн-курсы для 5-11 классов, формат 2 раза в неделю по 90 минут, год — {rendered_value}."
        if text.endswith(".weekly_lessons"):
            return f"{brand_label}: онлайн-курсы для 5-11 классов проходят {clean_text(raw_value)}."
        if text.endswith(".pair_duration_minutes"):
            return f"{brand_label}: занятие онлайн-курса для 5-11 классов длится {rendered_value}."

    if "matkap.child_age.sertificate_owner_min" in text:
        return f"{brand_label}: материнский капитал можно использовать, если ребёнку, на которого оформлен сертификат, исполнилось {format_number(numeric_value(raw_value) or 3)} года."
    if "matkap.child_age.student_max" in text:
        return f"{brand_label}: обучение с оплатой материнским капиталом возможно для ученика до {format_number(numeric_value(raw_value) or 25)} лет."
    if "matkap.timeline.sfr_review_days" in text:
        return f"{brand_label}: СФР рассматривает заявление на оплату материнским капиталом до {format_number(numeric_value(raw_value) or 10)} рабочих дней."
    if "matkap.timeline.transfer_days" in text:
        return f"{brand_label}: после одобрения СФР перевод средств обычно занимает до {format_number(numeric_value(raw_value) or 5)} рабочих дней."
    if "matkap.required_docs" in text:
        return f"{brand_label}: для оформления материнского капитала может понадобиться документ: {clean_text(value)}."
    if "matkap.timeline.total_max_days" in text:
        return f"{brand_label}: весь цикл проверки и перевода материнского капитала может занимать до {format_number(numeric_value(raw_value) or 15)} рабочих дней."

    if "ls_city_2026" in text and "discounts.already_paid_2026_27" in text:
        return f"{brand_label}: для городского летнего лагеря действует скидка {pct_text} для семей, уже оплативших обучение на 2026/27 год."
    if "ls_city_2026" in text and "discounts.former_student_same_brand" in text:
        return f"{brand_label}: для городского летнего лагеря действует скидка {pct_text} для учеников, которые ранее учились в этом же учебном центре."
    if "ls_city_2026" in text and "discounts.multichild" in text:
        return f"{brand_label}: для городского летнего лагеря действует скидка {pct_text} для многодетных семей."
    if "ls_city_2026" in text and "discounts.mfti_employees" in text:
        return f"{brand_label}: для городского летнего лагеря действует скидка {pct_text} для сотрудников МФТИ."
    if "ls_city_2026" in text and "discounts.refer_a_friend" in text:
        tariff = "расширенный тариф" if "advanced" in text else "базовый тариф" if "base" in text else "тариф"
        return (
            f"{brand_label}: для городского летнего лагеря по программе «приведи друга», {tariff}, "
            f"кэшбэк {rendered_value} ₽; условие: после завершения смены другом."
        )
    if "ls_city_2026" in text and "free_morning_club.from_day" in text:
        return f"{brand_label}: утренний клуб в городском летнем лагере доступен со {format_number(numeric_value(raw_value) or 2)}-го дня смены."

    if "loyal_customers_camps" in text:
        pct = percent_from_path(path)
        condition = clean_text(value)
        if pct and condition:
            return f"{brand_label}: скидка для постоянных участников лагерей — {pct}%, условие: {condition}."
    if "refer_a_friend" in text and "cashback" in text:
        condition = refer_a_friend_condition(path, brand=brand)
        return f"{brand_label}: по программе «приведи друга» кэшбэк {rendered_value}; условие: {condition}."
    if "discounts.active_student_to_summer_camp" in text:
        return f"{brand_label}: для действующих учеников этого учебного центра доступна скидка {pct_text} на любой летний лагерь."
    if "second_subject.offline.example" in text:
        return f"{brand_label}: при очном обучении на второй предмет действует скидка 20%; условие: ученик добавляет второй предмет."
    if "second_subject.online.example" in text:
        return f"{brand_label}: при онлайн-обучении на второй предмет действует скидка 30%; условие: ученик добавляет второй предмет."
    if "second_subject.offline" in text:
        return f"{brand_label}: при очном обучении скидка на второй предмет составляет {rendered_value}."
    if "second_subject.online" in text:
        return f"{brand_label}: при онлайн-обучении скидка на второй предмет составляет {rendered_value}."
    if "discounts.mfti_employees.pct" in text:
        return f"{brand_label}: для сотрудников МФТИ действует скидка {rendered_value}."
    if "discounts.mfti_employees.note" in text:
        return f"{brand_label}: для сотрудников МФТИ действует скидка 10%; другие значения перед ответом нужно проверить у ответственного менеджера."
    if "discounts.multichild.rule" in text:
        return f"{brand_label}: для многодетных семей действует скидка 10%; скидки не суммируются, применяется наибольшая доступная скидка."
    if "discounts.monthly_payment" in text:
        return f"{brand_label}: при помесячной оплате действует скидка {rendered_value}."
    if "discounts.stacking_rule" in text:
        return f"{brand_label}: если клиенту доступно несколько скидок, они не суммируются; применяется наибольшая доступная скидка."

    if "online_platform.levels" in text:
        return f"{brand_label}: на онлайн-платформе есть уровень обучения: {clean_text(value)}."
    if "online_platform.recording" in text:
        return (
            f"{brand_label}: на онлайн-платформе доступны записи занятий."
            if bool(value)
            else f"{brand_label}: по текущим данным записи занятий на онлайн-платформе не указаны."
        )
    if "free_trial_offline.offer" in text:
        return (
            f"{brand_label}: по бесплатному пробному очному занятию менеджер подскажет доступный вариант; "
            "лимит и возможность записи проверяются с куратором филиала."
        )

    if "academic_year_2026_27.weekly_lessons" in text:
        return f"{brand_label}: в учебном году 2026/27 занятия проходят {format_number(weekly_lessons_value(path, raw_value) or 1)} раз в неделю."
    if "lvsh_mendeleevo_2026.accommodation.meals_per_day" in text:
        return f"{brand_label}: в ЛВШ Менделеево предусмотрено {format_number(numeric_value(raw_value) or 5)} приёмов пищи в день."
    if "lvsh_mendeleevo_2026.pricing_2026.main_with_25_pct_discount" in text:
        return f"{brand_label}: при применении скидки 25% стоимость ЛВШ Менделеево составляет {rendered_value}."
    if "matkap.rule_federal_only" in text:
        return f"{brand_label}: по текущим правилам можно использовать федеральный материнский капитал."

    return ""


def render_discount_text(path: tuple[str, ...], value: Any, *, rendered_value: str, brand_label: str) -> str:
    text = ".".join(path).casefold()
    if "discount" not in text and "cashback" not in text:
        return ""

    if "loyal_customers_lvsh_only.applies_to" in text:
        return f"{brand_label}: скидка постоянным участникам ЛВШ применяется к продукту «{clean_text(value)}»."
    if "loyal_customers_lvsh_only.not_applies_to" in text:
        return f"{brand_label}: скидка постоянным участникам ЛВШ не применяется к продукту «{clean_text(value)}»."
    if "loyal_customers_lvsh_only.condition_client_text" in text:
        return f"{brand_label}: скидка постоянным участникам ЛВШ действует при условии: {clean_text(value)}."
    if "loyal_customers_lvsh_only.tiers.pct_" in text:
        pct = percent_from_path(path)
        if pct:
            return f"{brand_label}: скидка постоянным участникам ЛВШ — {pct}%; условие: {clean_text(value)}."

    if "discounts.multichild.pct" in text:
        return f"{brand_label}: скидка для многодетных семей — {rendered_value}; условие: подтвердить статус многодетной семьи."
    if "discounts.multichild.note" in text:
        return (
            f"{brand_label}: скидка для многодетных семей действует по статусу семьи, "
            "а не по количеству детей на оплате; условие: подтвердить статус многодетной семьи. "
            "Скидки не суммируются, применяется наибольшая доступная скидка."
        )

    if "lvsh_mendeleevo_2026.pricing_2026.discount_tiers_early_booking.tiers" in text:
        return (
            f"{brand_label}: по раннему бронированию ЛВШ Менделеево указана ступень скидки {rendered_value}; "
            "актуальный срок применения и доступность ступени проверяет менеджер."
        )
    if "lvsh_mendeleevo_2026.pricing_2026.discount_tiers_early_booking.type" in text:
        return (
            f"{brand_label}: для ЛВШ Менделеево используется динамическая система раннего бронирования; "
            "актуальную ступень скидки проверяет менеджер."
        )

    return ""


def render_objection_response_text(path: tuple[str, ...], value: Any, *, brand_label: str) -> str:
    text = ".".join(path).casefold()
    if "objection_responses" not in text:
        return ""
    raw = clean_text(value)
    if not raw:
        return ""
    topic = "возражение клиента"
    if "inconvenient_time" in text:
        topic = "возражение о неудобном времени"
    elif "too_expensive_camp" in text:
        topic = "возражение о стоимости лагеря"
    elif "too_expensive_course" in text:
        topic = "возражение о стоимости курса"
    elif "brand_link_question" in text:
        topic = "вопрос о связи брендов"
    elif "is_it_bot" in text:
        topic = "вопрос о том, кто отвечает клиенту"
    if raw.casefold().replace("ё", "е") == "скидки суммируются":
        return (
            f"{brand_label}: не использовать этот черновик без проверки РОПа; "
            "в источнике есть противоречивая фраза «Скидки суммируются», а действующее правило говорит, что скидки не суммируются."
        )
    if "приведи друга" in raw.casefold() and "условие:" not in raw.casefold():
        raw = f"{raw.rstrip('.')}; условие: проверить по программе «приведи друга» перед отправкой"
    return f"{raw.rstrip('.')}."


def render_social_proof_text(path: tuple[str, ...], value: Any, *, brand_label: str) -> str:
    text = ".".join(path).casefold()
    if "results_social_proof" not in text:
        return ""
    if "ege_avg_above_country_pts" in text:
        return f"{brand_label}: средний результат ЕГЭ у учеников выше среднего по стране на {format_number(numeric_value(value) or 25)} баллов."
    if "oge_grade_5_pct" in text:
        return f"{brand_label}: по текущим данным {format_number(numeric_value(value) or 92)}% учеников получили оценку 5 на ОГЭ."
    if "university_admission_pct" in text:
        return f"{brand_label}: по текущим данным {format_number(numeric_value(value) or 97)}% выпускников поступили в вузы."
    if "total_alumni" in text:
        return f"{brand_label}: по текущим данным через обучение прошло более {format_number(numeric_value(value) or 100000)} учеников."
    if "total_students_currently" in text:
        return f"{brand_label}: по текущим данным сейчас обучается {clean_text(value)} учеников."
    if "industry_rating_2025" in text:
        return f"{brand_label}: в рейтинге 2025 года указан статус «{clean_text(value)}»."
    if "olymp_fiztech_2024_winners" in text:
        return f"{brand_label}: в 2024 году указано {format_number(numeric_value(value) or 14)} победителей и призёров олимпиады «Физтех»."
    return ""


def render_intensive_text(path: tuple[str, ...], value: Any, *, brand_label: str) -> str:
    text = ".".join(path).casefold()
    if "intensives_2026" not in text:
        return ""
    rendered = render_simple_value(value)
    if "available_in_foton" in text or "available_in_unpk" in text:
        return f"{brand_label}: этот интенсив в текущем бренде {'доступен' if bool(value) else 'не проводится'}."
    if "duration_weeks" in text:
        return f"{brand_label}: длительность интенсива 2026 — {format_number(numeric_value(value) or 0)} недель."
    if "group_size" in text:
        return f"{brand_label}: размер группы на интенсиве 2026 — {rendered} человек."
    if ".classes" in text:
        return f"{brand_label}: интенсив 2026 рассчитан на {rendered} классы."
    if ".subjects." in text:
        return f"{brand_label}: среди предметов интенсива 2026 есть {rendered}."
    if ".includes." in text:
        return f"{brand_label}: в интенсив 2026 входит: {rendered}."
    if "bot_behavior_when_asked" in text:
        return f"{brand_label}: по вопросу об интенсиве менеджер подберёт подходящие варианты и свяжется в течение рабочего дня."
    return ""


def render_academic_year_text(path: tuple[str, ...], value: Any, *, brand_label: str) -> str:
    text = ".".join(path).casefold()
    if "academic_year_2026_27" not in text:
        return ""
    rendered = render_simple_value(value)
    if "semester_1_weeks" in text:
        return f"{brand_label}: в первом семестре 2026/27 учебного года {rendered} недель."
    if "semester_2_weeks" in text:
        return f"{brand_label}: во втором семестре 2026/27 учебного года {rendered} недель."
    return ""


def render_lvsh_text(path: tuple[str, ...], value: Any, *, rendered_value: str, brand_label: str) -> str:
    text = ".".join(path).casefold()
    if "lvsh_mendeleevo_2026" not in text:
        return ""
    rendered = render_simple_value(value)
    if ".smeny_2026." in text and ".freshness" in text:
        return f"{brand_label}: даты смен ЛВШ Менделеево требуют ручной проверки перед ответом клиенту."
    if "accommodation.air_conditioning" in text:
        return f"{brand_label}: в ЛВШ Менделеево по текущим данным есть кондиционирование."
    if ".directions." in text and ".classes" in text:
        direction = "физико-математического направления" if ".fizmat." in text else "IT-направления"
        return f"{brand_label}: ЛВШ Менделеево для {direction} рассчитана на {rendered} классы."
    if ".directions." in text and ".subjects." in text:
        direction = "физико-математического направления" if ".fizmat." in text else "IT-направления"
        return f"{brand_label}: в ЛВШ Менделеево для {direction} есть предмет «{rendered}»."
    if "program.group_size" in text:
        return f"{brand_label}: размер группы в ЛВШ Менделеево — {rendered} человек."
    if "pricing_2026.deposit" in text:
        return f"{brand_label}: депозит для ЛВШ Менделеево — {rendered_value}."
    if "pricing_2026.main_full" in text:
        return f"{brand_label}: полная стоимость ЛВШ Менделеево — {rendered_value}."
    if "pricing_2026.main_min" in text:
        return f"{brand_label}: минимальная стоимость ЛВШ Менделеево по текущим условиям — {rendered_value}."
    if "not_to_confuse_with" in text:
        return f"{brand_label}: городской лагерь и выездная ЛВШ Менделеево — разные продукты."
    return ""


def render_simple_value(value: Any) -> str:
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return format_number(value)
    return clean_text(value)


def refer_a_friend_condition(path: tuple[str, ...], *, brand: str) -> str:
    text = ".".join(path).casefold()
    if "camp_base_tariff" in text or "camp_advanced_tariff" in text:
        return "после завершения смены другом"
    if brand == "unpk":
        return "после 1 семестра обучения друга"
    if "offline" in text:
        return "после 5 занятий друга"
    if "online" in text:
        return "после семестра обучения друга"
    return "после выполнения условия программы"


def render_value(
    path: tuple[str, ...],
    value: Any,
    *,
    fact_type: str,
    structured_value: Mapping[str, Any],
) -> str:
    if is_range_mapping(value):
        amount_min, amount_max = range_bounds(value)
        return f"диапазон от {format_rub(amount_min)} до {format_rub(amount_max)}"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if structured_value.get("currency") == "RUB":
            return format_rub(value)
        if structured_value.get("unit") == "percent":
            return f"{format_number(value)}%"
        if structured_value.get("unit") == "minutes":
            return f"{format_number(value)} минут"
        if structured_value.get("unit") == "hours":
            return f"{format_number(value)} часов"
        if structured_value.get("unit") == "pairs":
            return f"{format_number(value)} пары"
        if structured_value.get("unit") == "lessons":
            return f"{format_number(value)} занятий"
        return format_number(value)
    text = clean_text(value, max_chars=1000).replace(" / ", " или ")
    is_url_or_handle = bool(
        re.search(r"(?:https?://|www\.|\b[a-z0-9-]+\.(?:ru|com|org|net)/\S+)", text, re.I)
        or text.startswith("@")
    )
    if "@" not in text and not is_url_or_handle:
        text = text.replace("_", " ")
    if text == "non stacking largest applies":
        return "скидки не суммируются, применяется наибольшая доступная скидка"
    if text == "used for crm only":
        return "только для внутренней CRM-метки"
    pct = percent_from_path(path)
    if pct is not None and text:
        return f"{pct}%: {text}"
    return text


def readable_fact_label(path: tuple[str, ...], *, fact_type: str) -> str:
    text = ".".join(path).casefold()
    labels: list[str] = []
    if "prices_regular_2026_27" in text:
        labels.append("цены на 2026/27 учебный год")
    elif "lvsh_mendeleevo" in text:
        labels.append("ЛВШ Менделеево")
    elif "ls_city_2026" in text:
        labels.append("городской летний лагерь")
    elif "individual_lessons" in text:
        labels.append("индивидуальные занятия")
    elif "intensives_2026" in text:
        labels.append("интенсивы 2026")
    elif "online_platform" in text:
        labels.append("онлайн-платформа")
    elif "academic_year_2026_27" in text:
        labels.append("учебный год 2026/27")
    elif "discount" in text:
        labels.append("скидка")
    elif "tax_deduction" in text:
        labels.append("налоговый вычет")
    elif "matkap" in text:
        labels.append("материнский капитал")
    elif "certificates" in text or "certificate" in text:
        labels.append("справки и документы")
    elif "location" in text or "address" in text or "metro" in text:
        labels.append("адрес и место занятий")
    elif "contacts" in text:
        labels.append("контакты")
    elif "installment" in text or "payment_options" in text:
        labels.append("рассрочка и оплата")
    else:
        labels.append(humanize_path(path, max_parts=2).replace("/", ",").replace("_", " "))

    classes = classes_from_path(path)
    if classes:
        labels.append(f"{classes} класс")
    fmt = format_from_path(path)
    if fmt == "offline":
        labels.append("очно")
    elif fmt == "online":
        labels.append("онлайн")
    include_price_validity_in_client_text = fact_type != "price"
    if include_price_validity_in_client_text and "before_2026_07_01" in text:
        labels.append("до 01.07.2026")
    if include_price_validity_in_client_text and "after_2026_07_01" in text:
        labels.append("после 01.07.2026")
    if include_price_validity_in_client_text and "before_2026_08_01" in text:
        labels.append("до 01.08.2026")
    if include_price_validity_in_client_text and "after_2026_08_01" in text:
        labels.append("после 01.08.2026")
    if include_price_validity_in_client_text and "before_2026_04_07" in text:
        labels.append("до 07.04.2026")
    if include_price_validity_in_client_text and "after_2026_04_07" in text:
        labels.append("после 07.04.2026")
    if "moscow" in text:
        labels.append("Москва")
    if "dolgoprudny" in text:
        labels.append("Долгопрудный")
    if "refer_a_friend" in text:
        labels.append("программа «приведи друга»")
    if "second_subject" in text:
        labels.append("скидка на второй предмет")
    if "mfti_employees" in text:
        labels.append("для сотрудников МФТИ")
    if "monthly_payment" in text:
        labels.append("при помесячной оплате")
    if "stacking" in text:
        labels.append("правило суммирования скидок")

    last = path[-1].casefold() if path else ""
    leaf_labels = {
        "name": "название",
        "dates": "даты",
        "start_date": "дата старта",
        "base": "базовый вариант",
        "plus_full": "расширенный вариант на полный день",
        "plus_half": "расширенный вариант на полдня",
        "plus_factultative": "вариант с факультативом",
        "plus_individual": "вариант с индивидуальными занятиями",
        "lesson_45min": "занятие 45 минут",
        "session_90min": "занятие 90 минут",
        "package_5_sessions": "пакет 5 занятий",
        "one_block": "один блок",
        "one_subject": "один предмет",
        "two_subjects": "два предмета",
        "semester": "семестр",
        "year": "год",
        "semester_range": "семестр",
        "year_range": "год",
        "four_weeks": "4 недели",
        "four_weeks_new": "4 недели для новых учеников",
        "total_lessons": "количество занятий за год",
        "weekly_lessons": "занятий в неделю",
        "daily_hours": "длительность занятия",
        "daily_pairs": "пар в день",
        "pair_duration_minutes": "длительность пары",
        "semester_1_weeks": "недель в первом семестре",
        "semester_2_weeks": "недель во втором семестре",
        "classes": "классы",
        "product": "продукт",
        "schedule": "расписание",
        "start": "старт занятий",
        "retroactive_years": "за сколько прошлых лет можно подать",
        "cashback": "кэшбэк",
        "condition": "условие",
        "rule": "правило",
    }
    if last in leaf_labels:
        labels.append(leaf_labels[last])
    return ", ".join(dict.fromkeys(part for part in labels if part))


def build_structured_value(path: tuple[str, ...], value: Any, *, fact_type: str) -> dict[str, Any]:
    structured: dict[str, Any] = {"path": ".".join(path), "raw_value": value}
    if is_range_mapping(value):
        amount_min, amount_max = range_bounds(value)
        structured["amount_min"] = amount_min
        structured["amount_max"] = amount_max
        structured["currency"] = "RUB"
        structured["range_id"] = ".".join(path)
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
    weekly_lessons = weekly_lessons_value(path, value)
    if weekly_lessons is not None:
        structured["weeks"] = weekly_lessons
        numeric = None
    else:
        numeric = numeric_value(value)
    if numeric is not None:
        path_percentage = percent_from_path(path)
        value_percentage = percentage_value(value)
        if path_percentage is not None:
            structured["percentage"] = path_percentage
            structured["unit"] = "percent"
        elif value_percentage is not None:
            structured["percentage"] = int(value_percentage) if value_percentage == int(value_percentage) else value_percentage
            structured["unit"] = "percent"
        elif is_month_path(path) and has_month_value(value):
            structured["months"] = int(numeric)
        elif is_day_path(path):
            structured["days"] = int(numeric)
        elif is_week_path(path) and weekly_lessons is None:
            structured["weeks"] = int(numeric)
        elif is_duration_minutes_path(path):
            structured["count"] = int(numeric)
            structured["unit"] = "minutes"
        elif is_duration_hours_path(path):
            structured["count"] = int(numeric)
            structured["unit"] = "hours"
        elif is_pair_count_path(path):
            structured["count"] = int(numeric)
            structured["unit"] = "pairs"
        elif is_lesson_count_path(path):
            structured["count"] = int(numeric)
            structured["unit"] = "lessons"
        elif is_class_path(path):
            structured["classes_raw"] = clean_text(value)
        elif fact_type == "price" or is_money_path(path):
            structured["amount"] = int(numeric) if numeric == int(numeric) else numeric
            structured["currency"] = "RUB"
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
    if "tax" in text or "deduction" in text or "вычет" in clean_text(value).casefold():
        return "tax"
    if any(marker in text for marker in CLIENT_SAFE_PATH_MARKERS):
        return "policy"
    if "refund" in text or "return" in text or "withholding" in text:
        return "refund"
    if is_non_money_numeric_path(path):
        return "course_parameter"
    if "academic_year" in text or "schedule" in text or "start_date" in text or "dates" in text or "smeny" in text:
        return "deadline" if is_deadline_fact(path) else "program"
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
    if "certificate" in text or "licenses" in text or "legal_entities" in text or "contract" in text or "document" in text:
        return "documents"
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
    if "online_olympiad_phystech_9_and_11" in text:
        return "online_olympiad_phystech_classes_9_11"
    if "fiztech_olympiad" in text:
        return "fiztech_olympiad_general"
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


def canonical_product_key(path: tuple[str, ...], product: str) -> str:
    text = ".".join(path).casefold()
    if "online_olympiad_phystech_9_and_11" in text:
        return "online_olympiad_phystech_classes_9_11"
    if "fiztech_olympiad" in text:
        return "fiztech_olympiad_general"
    return product


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
    if "installment.services.rassrochka.limit" in text:
        return True
    if "installment.services.rassrochka.term_months" in text:
        return True
    if "term_months_confirmed_by_dmitry" in text:
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
    if "installment.services.rassrochka.limit" in path_text:
        reasons.append("installment_limit_internal")
    if "installment.services.rassrochka.term_months" in path_text:
        reasons.append("installment_term_months_internal")
    if "term_months_confirmed_by_dmitry" in path_text:
        reasons.append("installment_confirmation_date_not_term")
    if fact_type in {"refund"} or route_policy in {"manager_only", "manager_handoff_only"}:
        reasons.append("manager_only_route")
    if fact_type in {"teacher"} and "approved_phrases" not in path_text:
        reasons.append("teacher_names_internal")
    if fact_type == "promocode" or "promo_codes" in path_text or "promocode" in path_text:
        reasons.append("promocode_removed_from_bot")
    if re.search(r"(?<!не\s)скидки\s+суммируются", text.casefold().replace("ё", "е")):
        reasons.append("discount_stacking_contradiction")
    reasons.extend(client_safety_violations(text, brand))
    if normalized_status in CLIENT_ALLOWED_STATUSES and not reasons:
        return True, []
    if normalized_status == "waiting_list" and not reasons:
        return True, []
    if normalized_status not in CLIENT_ALLOWED_STATUSES:
        reasons.append(f"not_client_allowed_status:{normalized_status}")
    return False, sorted(set(reasons))


def direct_text_requires_template(text: str, *, fact_type: str) -> bool:
    if fact_type in {"contact"}:
        return False
    if fact_type in {"price", "discount", "installment"}:
        return True
    stripped = " ".join(text.split())
    match = re.search(r"—\s*([^—.]{1,42})\.$", stripped)
    if not match:
        return False
    tail = match.group(1).strip()
    if any(unit in tail.casefold() for unit in ("руб", "₽", "%", "месяц", "недел", "дн", "занят", "человек")):
        return True
    return bool(re.fullmatch(r"(?:да|нет|[0-9]+(?:[.,][0-9]+)?|[0-9]+[–-][0-9]+|[А-Яа-яA-Za-z ]{1,24})", tail, re.IGNORECASE))


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
    has_unpk = any(term in lowered for term in ("унпк", "ано дпо", "ноу унпк", "kmipt", "unpk"))
    if has_foton and has_unpk:
        violations.append("cross_brand_text")
    if brand == "foton":
        for term in (
            "унпк",
            "ано дпо",
            "ноу унпк",
            "kmipt.ru",
            "@unpk_mipt",
            "@unpkmfti",
            "@unpk mipt",
            "unpkmfti",
            "+7 (495) 150-81-51",
            "8 (800) 500-81-51",
        ):
            if term in lowered:
                violations.append(f"other_brand_term:{term}")
    if brand == "unpk":
        for term in ("фотон", "цдпо", "црдо", "cdpofoton.ru", "edu@cdpofoton.ru", "@unpkmfti", "т-банк", "долями"):
            if term in lowered:
                violations.append(f"other_brand_term:{term}")
    return sorted(set(violations))


def requires_manager_confirmation_for_fact(
    *,
    status: str,
    freshness: str,
    route_policy: str,
    safety_reasons: Sequence[str],
) -> bool:
    normalized_status = normalize_status(status)
    normalized_freshness = normalize_freshness(freshness)
    if normalized_status in CONFIRMATION_STATUSES or normalized_freshness in CONFIRMATION_STATUSES:
        return True
    if route_policy in {"manager_only", "manager_handoff_only"} and any(
        "dynamic" in reason or "confirmation" in reason or "owner" in reason for reason in safety_reasons
    ):
        return True
    return False


def infer_route_policy(path: tuple[str, ...], *, fact_type: str, status: str, internal_only: bool) -> str:
    path_text = ".".join(path).casefold()
    if internal_only:
        return "manager_handoff_only"
    if fact_type == "refund":
        return "manager_only"
    if "rassrochka_6_12_months" in path_text:
        return "manager_handoff_only"
    if "discount_tiers_early_booking" in path_text:
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
    if text.startswith("approved_by_dmitry"):
        return "verified"
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


def percentage_value(value: Any) -> float | None:
    text = clean_text(value)
    match = re.search(r"(?<!\d)(\d{1,3}(?:[.,]\d+)?)\s*%", text)
    if not match:
        return None
    raw = match.group(1).replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def has_month_value(value: Any) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return True
    return bool(re.search(r"(?<!\d)\d{1,3}\s*(?:мес\.?|месяц)", clean_text(value), re.I))


def weekly_lessons_value(path: tuple[str, ...], value: Any) -> int | None:
    if "weekly_lessons" not in ".".join(path).casefold():
        return None
    text = clean_text(value).casefold().replace("ё", "е")
    match = re.search(r"(?<!\d)([1-7])\s+раз(?:а)?\s+в\s+недел", text, re.I)
    if match:
        return int(match.group(1))
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 1 <= int(value) <= 7:
        return int(value)
    return None


def is_range_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and numeric_value(value.get("min")) is not None and numeric_value(value.get("max")) is not None


def range_bounds(value: Mapping[str, Any]) -> tuple[int | float, int | float]:
    lower = numeric_value(value.get("min"))
    upper = numeric_value(value.get("max"))
    if lower is None or upper is None:
        raise ValueError("range value must contain numeric min and max")
    if lower > upper:
        lower, upper = upper, lower
    lower_out: int | float = int(lower) if lower == int(lower) else lower
    upper_out: int | float = int(upper) if upper == int(upper) else upper
    return lower_out, upper_out


def percent_from_path(path: tuple[str, ...]) -> int | None:
    for part in reversed(path):
        match = re.fullmatch(r"pct_(\d+)", part.casefold())
        if match:
            return int(match.group(1))
    return None


def is_money_path(path: tuple[str, ...]) -> bool:
    text = ".".join(path).casefold()
    if is_non_money_numeric_path(path):
        return False
    last = path[-1].casefold() if path else ""
    if "prices_regular_2026_27" in text:
        return last in MONEY_LEAF_KEYS or last.endswith("_range")
    markers = (
        "price",
        "pricing",
        "cost",
        "cashback",
        "deposit",
        "tariff",
        "four_weeks",
        "lesson_price",
        "session_price",
        "package_price",
        "tier_price",
        "main_full",
        "main_min",
        "max_return",
        "amount_rub",
    )
    return any(marker in text for marker in markers)


def is_non_money_numeric_path(path: tuple[str, ...]) -> bool:
    text = ".".join(path).casefold()
    return any(marker in text for marker in NON_MONEY_NUMERIC_PATH_MARKERS)


def is_month_path(path: tuple[str, ...]) -> bool:
    return any("month" in part.casefold() for part in path)


def is_day_path(path: tuple[str, ...]) -> bool:
    return any("day" in part.casefold() or "days" in part.casefold() for part in path)


def is_week_path(path: tuple[str, ...]) -> bool:
    return any("week" in part.casefold() or "weeks" in part.casefold() for part in path)


def is_lesson_count_path(path: tuple[str, ...]) -> bool:
    text = ".".join(path).casefold()
    return any(marker in text for marker in ("total_lessons", "weekly_lessons"))


def is_duration_minutes_path(path: tuple[str, ...]) -> bool:
    return "pair_duration_minutes" in ".".join(path).casefold()


def is_duration_hours_path(path: tuple[str, ...]) -> bool:
    return "daily_hours" in ".".join(path).casefold()


def is_pair_count_path(path: tuple[str, ...]) -> bool:
    return "daily_pairs" in ".".join(path).casefold()


def is_class_path(path: tuple[str, ...]) -> bool:
    return any("class" == part.casefold() or "classes" == part.casefold() or "programs" == part.casefold() for part in path)


def valid_until_from_path(path: tuple[str, ...]) -> str:
    text = ".".join(path)
    match = re.search(r"before_(\d{4})_(\d{2})_(\d{2})", text)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month}-{day}"
    lowered = text.casefold()
    if "prices_regular_2026_27" in lowered or "academic_year_2026_27" in lowered:
        return "2027-08-31"
    if "individual_lessons" in lowered:
        return "2027-08-31"
    if "ls_city_2026" in lowered or "lvsh_mendeleevo_2026" in lowered:
        return "2026-08-31"
    if "intensives_2026" in lowered:
        return "2026-12-31"
    if "zvsh_mendeleevo" in lowered:
        return "2027-08-31"
    if "online_platform" in lowered or "objection_responses" in lowered:
        return "2027-08-31"
    if "results_social_proof" in lowered:
        return "2026-12-31"
    if "matkap" in lowered or "tax_deduction" in lowered or "certificates" in lowered:
        return "2026-12-31"
    if "fiztech_olympiad" in lowered or "preschool_patsayeva" in lowered:
        return "2027-08-31"
    if (
        lowered.startswith("team_answers.")
        or "teacher_promo_codes" in lowered
        or "cross_brand_handoff_notes" in lowered
        or "crm_brand_groups" in lowered
    ):
        return "2026-12-31"
    if lowered.startswith("discounts.") or ".discounts." in lowered:
        return "2027-08-31"
    if "installment" in lowered or "payment_options" in lowered:
        return "2027-08-31"
    return "2026-12-31"


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
        "course_parameter": ["theme:016_program_content"],
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
        return ["УНПК", "АНО ДПО", "НОУ УНПК", "kmipt.ru", "@unpk_mipt", "@unpkmfti", "@unpk mipt"]
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
    if fact_type == "course_parameter":
        return "program"
    if "deadline" in key or "start_date" in key or "dates" in key or "smeny" in key:
        return "deadline"
    if "lvsh" in key:
        return "camp_lvsh"
    if "zvsh" in key:
        return "camp_zvsh"
    return fact_type


def approval_priority(fact: Mapping[str, Any], item_type: str, decision: str) -> str:
    if decision == "keep_internal_only":
        if fact.get("risk_level") == "high" or fact.get("linked_open_question"):
            return "P1"
        return "P2"
    if decision == "do_not_offer_to_client":
        return "P1"
    if item_type in {"price", "discount", "promocode", "installment", "tax", "matkap"}:
        return "P0"
    if decision == "needs_owner_confirmation_before_client_use":
        return "P1"
    if fact.get("risk_level") == "high":
        return "P1"
    if fact.get("requires_manager_confirmation") or fact.get("linked_open_question"):
        return "P1"
    return "P2"


def suggested_decision(fact: Mapping[str, Any]) -> str:
    status = str(fact.get("verification_status") or fact.get("freshness_status") or "")
    if status == "discontinued":
        return "do_not_offer_to_client"
    if fact.get("internal_only"):
        return "keep_internal_only"
    if status in {"needs_owner_confirmation", "dynamic_needs_check"}:
        return "needs_owner_confirmation_before_client_use"
    if fact.get("safety_block_reasons"):
        return "review_before_client_use"
    if fact.get("allowed_for_client_answer"):
        return "approve_for_client_answer_after_rop_review"
    return "review_before_client_use"


def rop_question(item_type: str, fact: Mapping[str, Any], decision: str) -> str:
    snippet = short_question_fact_text(fact)
    brand = BRAND_LABELS.get(str(fact.get("brand") or ""), str(fact.get("brand") or ""))
    product = clean_text(fact.get("product") or "", max_chars=80).replace("_", " ")
    prefix = f"{brand}"
    if product:
        prefix = f"{prefix}, {product}"
    if decision == "keep_internal_only":
        return f"Оставляем только для менеджера: {prefix}? Проверьте, не нужна ли клиентская короткая версия. Факт: {snippet}"
    if decision == "do_not_offer_to_client":
        return f"Подтверждаете, что это нельзя предлагать клиенту: {prefix}? Факт: {snippet}"
    if decision == "needs_owner_confirmation_before_client_use":
        return f"Кто должен подтвердить перед ответом клиенту: {prefix}? Факт: {snippet}"
    if item_type == "price":
        return f"Подтверждаете цену, период действия и область применения: {prefix}? Факт: {snippet}"
    if item_type == "discount":
        return f"Подтверждаете скидку и правило применения без суммирования: {prefix}? Факт: {snippet}"
    if item_type == "promocode":
        return f"Можно ли показывать этот промокод клиенту или оставить внутренним: {prefix}? Факт: {snippet}"
    if item_type == "installment":
        return f"Какую часть условия рассрочки можно говорить без ручной проверки: {prefix}? Факт: {snippet}"
    if item_type in {"tax", "matkap", "documents"}:
        return f"Подтверждаете клиентскую формулировку без лишних реквизитов: {prefix}? Факт: {snippet}"
    if item_type in {"deadline", "camp_lvsh", "camp_zvsh", "camp_city", "program", "intensive"}:
        return f"Подтверждаете актуальность программы, сроков и ограничений: {prefix}? Факт: {snippet}"
    if item_type in {"contact", "location"}:
        return f"Подтверждаете контакт или адрес для клиентского ответа: {prefix}? Факт: {snippet}"
    return f"Можно ли использовать этот факт в ответе клиенту текущего бренда: {prefix}? Факт: {snippet}"


def short_question_fact_text(fact: Mapping[str, Any]) -> str:
    text = clean_text(
        fact.get("client_safe_text") or fact.get("manager_check_text") or fact.get("fact_text") or fact.get("fact_key"),
        max_chars=180,
    )
    text = re.sub(r"\[[^\]]*client_blocked:[^\]]*\]", "", text).strip()
    return text or clean_text(fact.get("fact_key") or "", max_chars=180)


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
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
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
