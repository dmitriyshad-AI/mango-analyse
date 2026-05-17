#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
    raise SystemExit("PyYAML is required to build kb_release_v2") from exc


SCHEMA_VERSION = "kb_release_v2_snapshot_2026_05_17"
FACT_SCHEMA_VERSION = "kb_release_v2_fact_v1"
BUILDER_VERSION = "kb_release_v2_builder_2026_05_17_v1"

DEFAULT_RUN_ID = "kb_release_20260517_v2"
DEFAULT_OUT_DIR = Path("product_data/knowledge_base/kb_release_20260517_v2")
DEFAULT_AGENT_PACK_DIR = Path("product_data/knowledge_base/kb_release_20260517_v2_agent_pack")
DEFAULT_CODEX_V1_AGENT_PACK = Path("product_data/knowledge_base/kb_release_20260517_v1_agent_pack")
DEFAULT_CLAUDE_LAYER_DIR = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/kb_release_v2_claude_layer_2026-05-17")
DEFAULT_CLAUDE_FINAL_DIR = Path("Claude Mango_Bot_Knowledge_Base_FINAL_2026-05-17")
DEFAULT_TZ_PATH = Path("docs/KB_RELEASE_V2_CLAUDE_CODEX_INTEGRATION_TZ_2026-05-17.md")

ALLOWED_BRANDS = {"foton", "unpk", "brand_neutral", "internal"}
CLIENT_ALLOWED_FRESHNESS = {"fresh_verified", "document_verified"}
BRAND_NAMES = {
    "foton": ("фотон", "цдпо", "црдо", "cdpofoton"),
    "unpk": ("унпк", "мфти", "ано дпо", "ноу", "kmipt"),
}
FACT_TYPE_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("price", ("price", "pricing", "prices", "стоим", "цен", "тариф", "руб", "₽")),
    ("discount", ("discount", "discounts", "скид", "промокод", "кэшбек", "акци")),
    ("installment", ("installment", "рассроч", "долями", "т-банк")),
    ("matkap", ("matkap", "маткап", "материн")),
    ("tax", ("tax", "налог", "вычет", "ндфл")),
    ("documents", ("certificate", "certificates", "document", "договор", "справ", "квитанц", "лиценз")),
    ("schedule", ("schedule", "smen", "смен", "распис", "дат", "время")),
    ("program", ("program", "format", "course", "lvsh", "лагер", "лвш", "интенсив", "программ")),
    ("location", ("location", "address", "адрес", "локац", "контакт")),
    ("payment_status", ("payment_status", "оплат", "платеж", "чек")),
)
DYNAMIC_FACT_TYPES = {"price", "discount", "schedule", "program", "installment"}
INTERNAL_SECTION_KEYS = {"legal_entities", "internal_kc_processes", "crm_brand_groups"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build kb_release_v2 by integrating Claude layer and Codex v1 KB.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--agent-pack-dir", type=Path, default=DEFAULT_AGENT_PACK_DIR)
    parser.add_argument("--codex-v1-agent-pack", type=Path, default=DEFAULT_CODEX_V1_AGENT_PACK)
    parser.add_argument("--claude-layer-dir", type=Path, default=DEFAULT_CLAUDE_LAYER_DIR)
    parser.add_argument("--claude-final-dir", type=Path, default=DEFAULT_CLAUDE_FINAL_DIR)
    parser.add_argument("--tz-path", type=Path, default=DEFAULT_TZ_PATH)
    args = parser.parse_args(argv)

    result = build_kb_release_v2(
        run_id=args.run_id,
        out_dir=args.out_dir,
        agent_pack_dir=args.agent_pack_dir,
        codex_v1_agent_pack=args.codex_v1_agent_pack,
        claude_layer_dir=args.claude_layer_dir,
        claude_final_dir=args.claude_final_dir,
        tz_path=args.tz_path,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_kb_release_v2(
    *,
    run_id: str = DEFAULT_RUN_ID,
    out_dir: Path = DEFAULT_OUT_DIR,
    agent_pack_dir: Path = DEFAULT_AGENT_PACK_DIR,
    codex_v1_agent_pack: Path = DEFAULT_CODEX_V1_AGENT_PACK,
    codex_snapshot_path: Path | None = None,
    snapshot_path: Path | None = None,
    v1_snapshot_path: Path | None = None,
    codex_source_inventory_path: Path | None = None,
    source_inventory_path: Path | None = None,
    inventory_path: Path | None = None,
    claude_yaml_path: Path | None = None,
    claude_facts_path: Path | None = None,
    facts_yaml_path: Path | None = None,
    yaml_path: Path | None = None,
    claude_layer_dir: Path = DEFAULT_CLAUDE_LAYER_DIR,
    claude_final_dir: Path = DEFAULT_CLAUDE_FINAL_DIR,
    tz_path: Path = DEFAULT_TZ_PATH,
) -> Mapping[str, Any]:
    out_root = guard_output_dir(out_dir)
    agent_root = guard_output_dir(agent_pack_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    agent_root.mkdir(parents=True, exist_ok=True)

    snapshot_file = codex_snapshot_path or snapshot_path or v1_snapshot_path or (
        codex_v1_agent_pack / "kc_snapshot_kb_release_20260517_v1.json"
    )
    inventory_file = codex_source_inventory_path or source_inventory_path or inventory_path
    codex_snapshot = load_json(Path(snapshot_file))
    codex_sources = load_source_registry(codex_v1_agent_pack, inventory_path=inventory_file)
    claude = load_claude_layer(
        claude_layer_dir=claude_layer_dir,
        claude_final_dir=claude_final_dir,
        claude_yaml_path=claude_yaml_path or claude_facts_path or facts_yaml_path or yaml_path,
    )

    sources = build_v2_sources(
        codex_sources=codex_sources,
        claude=claude,
        claude_layer_dir=claude_layer_dir,
        claude_final_dir=claude_final_dir,
        tz_path=tz_path,
    )
    source_lookup = {str(source["source_id"]): source for source in sources}
    source_matcher = build_source_matcher(sources)

    facts = build_v2_facts(claude=claude, source_matcher=source_matcher, tz_path=tz_path)
    facts = ensure_policy_decision_facts(facts, source_matcher=source_matcher, tz_path=tz_path)
    facts = attach_source_details(facts, source_lookup=source_lookup)
    facts = dedupe_facts(facts)
    chunks = build_v2_chunks(facts=facts, codex_snapshot=codex_snapshot)

    theme_mapping = build_theme_mapping_stub(claude)
    approval_queue = build_approval_queue_v2(facts)
    snapshot = build_snapshot_v2(
        run_id=run_id,
        sources=sources,
        facts=facts,
        chunks=chunks,
        codex_snapshot=codex_snapshot,
        claude=claude,
        theme_mapping=theme_mapping,
    )
    quality = build_quality_report(snapshot, approval_queue=approval_queue)

    write_outputs(
        out_root,
        agent_root,
        snapshot=snapshot,
        sources=sources,
        facts=facts,
        chunks=chunks,
        approval_queue=approval_queue,
        quality=quality,
        brand_rules=normalize_brand_rules(claude.get("brand_rules") or {}),
        bot_policy=normalize_bot_policy(claude.get("bot_policy") or {}),
        theme_mapping=theme_mapping,
    )
    return {
        "out_dir": str(out_root),
        "agent_pack_dir": str(agent_root),
        "snapshot_path": str(out_root / "kb_release_v2_snapshot.json"),
        "facts_total": len(facts),
        "client_allowed_facts": sum(1 for fact in facts if fact.get("allowed_for_client_answer")),
        "cross_brand_internal_facts": sum(1 for fact in facts if fact.get("cross_brand_mixed")),
        "brands": Counter(str(fact.get("brand") or "") for fact in facts),
        "quality_passed": quality["quality_passed"],
    }


def load_claude_layer(
    *,
    claude_layer_dir: Path = DEFAULT_CLAUDE_LAYER_DIR,
    claude_final_dir: Path = DEFAULT_CLAUDE_FINAL_DIR,
    claude_yaml_path: Path | None = None,
    claude_facts_path: Path | None = None,
    facts_yaml_path: Path | None = None,
    yaml_path: Path | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    layer = Path(claude_layer_dir)
    final = Path(claude_final_dir)
    direct_yaml = claude_yaml_path or claude_facts_path or facts_yaml_path or yaml_path or path
    payload: dict[str, Any] = {
        "layer_dir": str(layer),
        "final_dir": str(final),
        "brand_facts": {},
        "loaded_files": [],
        "missing_files": [],
        "load_errors": {},
    }
    if direct_yaml is not None:
        direct_path = Path(direct_yaml)
        raw = load_yaml(direct_path)
        raw_mapping = raw if isinstance(raw, Mapping) else {}
        payload["raw"] = raw_mapping
        payload["legacy_facts"] = raw_mapping
        payload["top_level_sections"] = list(raw_mapping.keys())
        payload["loaded_files"].append(str(direct_path))
        return payload
    for key, filename in (
        ("brand_rules", "brand_rules.yaml"),
        ("bot_policy", "bot_policy.yaml"),
        ("facts_internal_only", "facts_internal_only.yaml"),
    ):
        path = layer / filename
        if path.exists():
            payload[key] = load_yaml(path)
            payload["loaded_files"].append(str(path))
        else:
            payload["missing_files"].append(str(path))
    for brand, filename in (("foton", "facts_for_bot_FOTON.yaml"), ("unpk", "facts_for_bot_UNPK.yaml")):
        path = layer / filename
        if path.exists():
            loaded = load_yaml(path, tolerate_errors=True)
            if isinstance(loaded, Mapping) and loaded.get("__load_error__"):
                payload["load_errors"][str(path)] = loaded["__load_error__"]
                payload["brand_facts"][brand] = {}
            else:
                payload["brand_facts"][brand] = loaded
            payload["loaded_files"].append(str(path))
        else:
            payload["missing_files"].append(str(path))
    legacy_path = final / "02_FOR_BOT_AND_CODEX" / "facts_for_bot.yaml"
    if legacy_path.exists():
        payload["legacy_facts"] = load_yaml(legacy_path, tolerate_errors=True)
        payload["loaded_files"].append(str(legacy_path))
    else:
        payload["legacy_facts"] = {}
        payload["missing_files"].append(str(legacy_path))
    return payload


def normalize_v2_fact(
    *,
    raw_fact: Mapping[str, Any] | None = None,
    fact: Mapping[str, Any] | None = None,
    record: Mapping[str, Any] | None = None,
    source_record: Mapping[str, Any] | None = None,
    fact_key: str = "",
    title: str = "",
    fact_text: str = "",
    brand: str = "",
    source_id: str = "",
    source_title: str = "",
    source_path: str = "",
    source_url: str = "",
    source_sha256: str = "",
    freshness_status: str = "document_verified",
    fact_type: str | None = None,
    product: str = "",
    route_policy: str = "draft_for_manager",
    risk_level: str = "low",
    related_theme_ids: Sequence[str] = (),
    internal_only: bool = False,
    cross_brand_mixed: bool = False,
    cross_brand_policy: str = "",
    owner_role: str = "РОП/ответственный владелец факта",
    notes: str = "",
) -> dict[str, Any]:
    raw = raw_fact or fact or record or {}
    if raw:
        fact_key = fact_key or str(raw.get("fact_key") or raw.get("key") or raw.get("id") or "test.fact")
        title = title or str(raw.get("title") or raw.get("name") or fact_key)
        fact_text = fact_text or str(
            raw.get("fact_text") or raw.get("client_safe_text") or raw.get("text") or raw.get("description") or ""
        )
        brand = brand or str(raw.get("brand") or "brand_neutral")
        fact_type = fact_type or str(raw.get("fact_type") or "")
        product = product or str(raw.get("product") or "")
        source_title = source_title or str(raw.get("source") or raw.get("source_title") or "")
        freshness_status = str(raw.get("freshness_status") or raw.get("status") or freshness_status)
    if source_record:
        source_id = source_id or str(source_record.get("source_id") or "")
        source_title = source_title or str(source_record.get("source_title") or source_record.get("title") or "")
        source_path = source_path or str(source_record.get("source_path") or source_record.get("path") or "")
        source_url = source_url or str(source_record.get("source_url") or source_record.get("url") or "")
        source_sha256 = source_sha256 or str(source_record.get("source_sha256") or source_record.get("sha256") or "")
    clean_brand = normalize_brand(brand)
    clean_fact_type = fact_type or infer_fact_type(f"{fact_key} {title} {fact_text}")
    clean_freshness = normalize_freshness(freshness_status)
    detected_cross_brand = bool(cross_brand_mixed or is_cross_brand_text(f"{title} {fact_text}"))
    if detected_cross_brand and clean_brand in {"brand_neutral", "internal"}:
        clean_brand = "internal"
    if clean_brand == "brand_neutral" and not is_brand_neutral_candidate(f"{title} {fact_text}"):
        clean_brand = "internal"
        internal_only = True
    if clean_brand == "brand_neutral" and clean_fact_type in {"matkap", "tax"}:
        clean_brand = "internal"
        internal_only = True
    clean_cross_policy = cross_brand_policy or ("forbidden_for_client" if detected_cross_brand else "brand_neutral_allowed")
    forbidden_for_client = bool(
        internal_only or detected_cross_brand or clean_brand == "internal" or clean_cross_policy == "forbidden_for_client"
    )
    dynamic = clean_fact_type in DYNAMIC_FACT_TYPES and clean_freshness == "dynamic_needs_check"
    allowed = (
        clean_freshness in CLIENT_ALLOWED_FRESHNESS
        and not forbidden_for_client
        and clean_brand in {"foton", "unpk", "brand_neutral"}
        and clean_fact_type not in {"matkap", "tax", "payment_status"}
    )
    requires_confirmation = not allowed or dynamic
    fact_id = f"fact:v2:{clean_brand}:{safe_id(fact_key)}:{sha256_text(f'{title}|{fact_text}|{source_id}')[:10]}"
    text = clean_text(fact_text, max_chars=1200)
    return {
        "schema_version": FACT_SCHEMA_VERSION,
        "fact_id": fact_id,
        "fact_key": clean_text(fact_key, max_chars=160),
        "fact_type": clean_fact_type,
        "fact_types": [clean_fact_type],
        "title": clean_text(title, max_chars=180),
        "fact_text": text,
        "short_fact": clean_text(title, max_chars=180),
        "client_safe_text": "" if forbidden_for_client else text,
        "manager_check_text": text,
        "brand": clean_brand,
        "active_brand_scope": active_brand_scope(clean_brand, internal_only=internal_only),
        "cross_brand_policy": clean_cross_policy,
        "cross_brand_mixed": bool(detected_cross_brand),
        "product": product or infer_product(fact_key),
        "source_id": source_id,
        "source_title": source_title,
        "source_path": source_path,
        "source_url": source_url,
        "source_sha256": source_sha256,
        "source_status": "read" if source_path or source_url else "unknown",
        "freshness_status": clean_freshness,
        "verification_status": clean_freshness,
        "valid_from": "",
        "valid_until": "",
        "verified_by": "",
        "verified_at": "",
        "owner_role": owner_role,
        "allowed_for_client_answer": bool(allowed),
        "usable_for_precise_answer": bool(allowed),
        "requires_manager_confirmation": bool(requires_confirmation),
        "requires_amo_check": clean_fact_type == "payment_status",
        "requires_tallanto_check": clean_fact_type == "payment_status",
        "route_policy": route_policy,
        "risk_level": risk_level,
        "related_theme_ids": list(related_theme_ids or theme_ids_for_fact_type(clean_fact_type)),
        "forbidden_promises": forbidden_promises_for_fact_type(clean_fact_type),
        "forbidden_client_mentions": forbidden_mentions_for_brand(clean_brand),
        "forbidden_for_client": forbidden_for_client,
        "internal_only": bool(internal_only),
        "notes": notes,
        "record_type": "fact",
    }


def build_v2_sources(
    *,
    codex_sources: Sequence[Mapping[str, Any]],
    claude: Mapping[str, Any],
    claude_layer_dir: Path,
    claude_final_dir: Path,
    tz_path: Path,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in codex_sources:
        source = dict(record)
        source_id = clean_text(source.get("source_id") or source.get("id"))
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        source["schema_version"] = "kb_release_v2_source_v1"
        source.setdefault("source_status", source.get("read_status") or source.get("processing_status") or "unknown")
        sources.append(source)
    for path_text in claude.get("loaded_files") or ():
        path = Path(path_text)
        source_id = f"source:claude_layer:{safe_id(path.stem)}:{sha256_text(str(path))[:10]}"
        if source_id in seen:
            continue
        seen.add(source_id)
        sources.append(
            {
                "schema_version": "kb_release_v2_source_v1",
                "source_id": source_id,
                "title": path.name,
                "source_kind": "claude_layer_yaml" if path.suffix.lower() in {".yaml", ".yml"} else "claude_layer_doc",
                "source_role": "claude_curated_layer",
                "path": str(path),
                "url": "",
                "source_sha256": sha256_file(path) if path.exists() else "",
                "sha256": sha256_file(path) if path.exists() else "",
                "source_status": "read" if path.exists() else "missing",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": False,
                "requires_manager_confirmation": True,
            }
        )
    for path in (claude_layer_dir, claude_final_dir, tz_path):
        if not path.exists():
            continue
        source_id = f"source:local:{safe_id(path.name)}:{sha256_text(str(path))[:10]}"
        if source_id in seen:
            continue
        seen.add(source_id)
        sources.append(
            {
                "schema_version": "kb_release_v2_source_v1",
                "source_id": source_id,
                "title": path.name,
                "source_kind": "local_reference",
                "source_role": "implementation_reference",
                "path": str(path),
                "url": "",
                "source_sha256": sha256_file(path) if path.is_file() else "",
                "sha256": sha256_file(path) if path.is_file() else "",
                "source_status": "read",
                "freshness_status": "document_verified",
                "usable_for_precise_answer": False,
                "requires_manager_confirmation": True,
            }
        )
    return sources


def build_v2_facts(*, claude: Mapping[str, Any], source_matcher: Mapping[str, Mapping[str, Any]], tz_path: Path) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for brand, payload in (claude.get("brand_facts") or {}).items():
        if isinstance(payload, Mapping):
            facts.extend(facts_from_brand_payload(payload, brand=brand, source_matcher=source_matcher))
    internal_payload = claude.get("facts_internal_only")
    if isinstance(internal_payload, Mapping):
        facts.extend(facts_from_internal_payload(internal_payload, source_matcher=source_matcher))
    legacy_payload = claude.get("legacy_facts")
    if isinstance(legacy_payload, Mapping):
        facts.extend(facts_from_legacy_payload(legacy_payload, source_matcher=source_matcher))
    return facts


def facts_from_brand_payload(
    payload: Mapping[str, Any],
    *,
    brand: str,
    source_matcher: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, value in payload.items():
        if key in {"schema_version", "generated_at", "active_brand_scope"}:
            continue
        records.extend(flatten_fact_section(value, path=(str(key),), brand=brand, source_matcher=source_matcher))
    return records


def facts_from_internal_payload(
    payload: Mapping[str, Any],
    *,
    source_matcher: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, value in payload.items():
        if key in {"schema_version", "generated_at"}:
            continue
        records.extend(
            flatten_fact_section(
                value,
                path=(str(key),),
                brand="internal",
                source_matcher=source_matcher,
                internal_only=True,
                cross_brand_policy="forbidden_for_client",
            )
        )
    return records


def facts_from_legacy_payload(
    payload: Mapping[str, Any],
    *,
    source_matcher: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key, value in payload.items():
        if key in {"schema_version", "generated_at"}:
            continue
        internal = key in {"critical_rules", "legal_entities", "internal_kc_processes", "crm_brand_groups"}
        records.extend(
            flatten_fact_section(
                value,
                path=(str(key),),
                brand="internal" if internal else "brand_neutral",
                source_matcher=source_matcher,
                internal_only=internal,
                cross_brand_policy="forbidden_for_client" if internal else "",
            )
        )
    return records


def flatten_fact_section(
    value: Any,
    *,
    path: tuple[str, ...],
    brand: str,
    source_matcher: Mapping[str, Mapping[str, Any]],
    internal_only: bool = False,
    cross_brand_policy: str = "",
    inherited_source_hint: str = "",
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        section_internal = internal_only or truthy(value.get("internal_only")) or path[0] in INTERNAL_SECTION_KEYS
        section_status = normalize_freshness(value.get("freshness_status") or value.get("status") or "document_verified")
        section_source = clean_text(value.get("source") or value.get("source_title") or inherited_source_hint)
        section_brand = normalize_brand(value.get("brand") or brand)
        if is_fact_leaf(value):
            text = render_fact_text(value)
            return [
                make_fact_from_path(
                    path=path,
                    text=text,
                    brand=section_brand,
                    status=section_status,
                    source_hint=section_source,
                    source_matcher=source_matcher,
                    internal_only=section_internal,
                    cross_brand_policy=cross_brand_policy,
                )
            ]
        records: list[dict[str, Any]] = []
        for key, item in value.items():
            if key in {"status", "freshness_status", "source", "source_title", "brand", "internal_only", "owner", "product"}:
                continue
            key_text = str(key)
            next_brand = key_text if key_text.casefold() in {"foton", "unpk"} else section_brand
            next_path = path if key_text == "brands" else (*path, key_text)
            records.extend(
                flatten_fact_section(
                    item,
                    path=next_path,
                    brand=next_brand,
                    source_matcher=source_matcher,
                    internal_only=section_internal,
                    cross_brand_policy=cross_brand_policy,
                    inherited_source_hint=section_source,
                )
            )
        return records
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        records: list[dict[str, Any]] = []
        for index, item in enumerate(value, 1):
            records.extend(
                flatten_fact_section(
                    item,
                    path=(*path, str(index)),
                    brand=brand,
                    source_matcher=source_matcher,
                    internal_only=internal_only,
                    cross_brand_policy=cross_brand_policy,
                    inherited_source_hint=inherited_source_hint,
                )
            )
        return records
    return [
        make_fact_from_path(
            path=path,
            text=clean_text(value, max_chars=1200),
            brand=brand,
            status="document_verified",
            source_hint=inherited_source_hint,
            source_matcher=source_matcher,
            internal_only=internal_only,
            cross_brand_policy=cross_brand_policy,
        )
    ]


def make_fact_from_path(
    *,
    path: tuple[str, ...],
    text: str,
    brand: str,
    status: str,
    source_hint: str,
    source_matcher: Mapping[str, Mapping[str, Any]],
    internal_only: bool,
    cross_brand_policy: str,
) -> dict[str, Any]:
    key = ".".join(safe_id(part) for part in path if part)
    title = " / ".join(path)
    fact_type = infer_fact_type(f"{key} {title} {text}")
    clean_brand = normalize_brand(brand)
    cross_mixed = bool(is_cross_brand_text(f"{title} {text}") and clean_brand in {"internal", "brand_neutral"})
    source = match_source(source_hint or title, source_matcher)
    return normalize_v2_fact(
        fact_key=key,
        title=title,
        fact_text=text,
        brand=clean_brand,
        fact_type=fact_type,
        product=infer_product(key),
        source_id=str(source.get("source_id") or ""),
        source_title=str(source.get("title") or source.get("source_title") or source_hint or "Claude layer"),
        source_path=str(source.get("path") or ""),
        source_url=str(source.get("url") or source.get("google_drive_url") or ""),
        source_sha256=str(source.get("source_sha256") or source.get("sha256") or ""),
        freshness_status=status,
        related_theme_ids=theme_ids_for_fact_type(fact_type),
        internal_only=internal_only or cross_mixed,
        cross_brand_mixed=cross_mixed,
        cross_brand_policy=cross_brand_policy or ("forbidden_for_client" if cross_mixed else "brand_neutral_allowed"),
        notes="Импортировано из Claude layer.",
    )


def is_fact_leaf(value: Mapping[str, Any]) -> bool:
    if "brands" in value:
        return False
    scalar_count = sum(1 for item in value.values() if not isinstance(item, (Mapping, list, tuple)))
    return scalar_count >= 2 and any(key in value for key in ("status", "brand", "source", "client_safe_text", "approved_phrase"))


def render_fact_text(value: Mapping[str, Any]) -> str:
    preferred = value.get("client_safe_text") or value.get("approved_phrase") or value.get("description") or value.get("rule")
    if preferred:
        return clean_text(preferred, max_chars=1200)
    parts: list[str] = []
    for key, item in value.items():
        if key in {"status", "freshness_status", "source", "source_title", "brand", "internal_only"}:
            continue
        if isinstance(item, (Mapping, list, tuple)):
            continue
        parts.append(f"{key}: {item}")
    return clean_text("; ".join(parts), max_chars=1200)


def ensure_policy_decision_facts(
    facts: Sequence[Mapping[str, Any]],
    *,
    source_matcher: Mapping[str, Mapping[str, Any]],
    tz_path: Path,
) -> list[dict[str, Any]]:
    result = [dict(fact) for fact in facts]
    source = {
        "source_id": f"source:codex_tz:{safe_id(tz_path.stem)}:{sha256_text(str(tz_path))[:10]}",
        "title": tz_path.name,
        "path": str(tz_path),
        "source_sha256": sha256_file(tz_path) if tz_path.exists() else "",
    }
    existing = {(str(f.get("brand")), str(f.get("product")), str(f.get("fact_type"))) for f in result}
    decisions = (
        ("foton", "matkap", "matkap", "Материнский капитал доступен в рамках бренда Фотон при проверке актуальных брендовых документов."),
        ("unpk", "matkap", "matkap", "Материнский капитал доступен в рамках бренда УНПК МФТИ при проверке актуальных брендовых документов."),
        ("foton", "tax", "tax", "Налоговый вычет доступен в рамках бренда Фотон при проверке актуальных брендовых документов."),
        ("unpk", "tax", "tax", "Налоговый вычет доступен в рамках бренда УНПК МФТИ при проверке актуальных брендовых документов."),
        ("foton", "lvsh_mendeleevo", "program", "ЛВШ Менделеево есть в бренде Фотон; использовать только Фотон-смены, цены и документы."),
        ("unpk", "lvsh_mendeleevo", "program", "ЛВШ Менделеево есть в бренде УНПК МФТИ; использовать только УНПК-смены, цены и документы."),
    )
    for brand, product, fact_type, text in decisions:
        if (brand, product, fact_type) in existing:
            continue
        result.append(
            normalize_v2_fact(
                fact_key=f"{product}.policy_decision",
                title=f"{product} / {brand} / policy decision",
                fact_text=text,
                brand=brand,
                fact_type=fact_type,
                product=product,
                source_id=str(source["source_id"]),
                source_title=str(source["title"]),
                source_path=str(source["path"]),
                source_sha256=str(source["source_sha256"]),
                freshness_status="document_verified",
                notes="Добавлено из утвержденного решения Дмитрия в ТЗ.",
            )
        )
    return result


def attach_source_details(facts: Sequence[Mapping[str, Any]], *, source_lookup: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for fact in facts:
        item = dict(fact)
        source = source_lookup.get(str(item.get("source_id") or ""))
        if source:
            item["source_title"] = item.get("source_title") or source.get("title") or source.get("source_title") or ""
            item["source_path"] = item.get("source_path") or source.get("path") or ""
            item["source_url"] = item.get("source_url") or source.get("url") or source.get("google_drive_url") or ""
            item["source_sha256"] = item.get("source_sha256") or source.get("source_sha256") or source.get("sha256") or ""
        result.append(item)
    return result


def dedupe_facts(facts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for fact in facts:
        item = dict(fact)
        key = (
            str(item.get("brand") or ""),
            str(item.get("fact_key") or ""),
            str(item.get("fact_type") or ""),
            sha256_text(str(item.get("fact_text") or item.get("manager_check_text") or ""))[:12],
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def build_v2_chunks(*, facts: Sequence[Mapping[str, Any]], codex_snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for fact in facts:
        text = clean_text(fact.get("client_safe_text") or fact.get("manager_check_text") or fact.get("fact_text"), max_chars=1400)
        if not text:
            continue
        chunks.append(
            {
                "schema_version": "kb_release_v2_chunk_v1",
                "chunk_id": f"kc_chunk:v2:{safe_id(str(fact.get('fact_id')))}",
                "source_id": fact.get("source_id"),
                "title": fact.get("title"),
                "text": text,
                "fact_types": list(fact.get("fact_types") or [fact.get("fact_type")]),
                "freshness_status": fact.get("freshness_status"),
                "bot_permission": "internal_only" if fact.get("forbidden_for_client") else "bot_answer_self_or_draft",
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
                    "source_title": fact.get("source_title"),
                    "brand": fact.get("brand"),
                    "product": fact.get("product"),
                    "cross_brand_policy": fact.get("cross_brand_policy"),
                    "cross_brand_mixed": bool(fact.get("cross_brand_mixed")),
                    "bot_permission": "internal_only" if fact.get("forbidden_for_client") else "bot_answer_self_or_draft",
                    "source_role": "approved_fact_candidate",
                },
            }
        )
    # Keep manager patterns and question templates available as style, but still not as facts.
    for chunk in (codex_snapshot.get("chunks") or codex_snapshot.get("knowledge_chunks") or [])[:400]:
        if not isinstance(chunk, Mapping):
            continue
        copied = dict(chunk)
        copied.setdefault("schema_version", "kc_knowledge_chunk_v1")
        metadata = dict(copied.get("metadata") or {})
        metadata.setdefault("source_role", copied.get("source_role") or "legacy_style_context")
        copied["metadata"] = metadata
        copied.setdefault("brand", copied.get("brand") or metadata.get("brand") or "brand_neutral")
        copied.setdefault("cross_brand_policy", "forbidden_for_client" if is_cross_brand_text(str(copied.get("text") or "")) else "brand_neutral_allowed")
        copied.setdefault("cross_brand_mixed", is_cross_brand_text(str(copied.get("text") or "")))
        if copied["cross_brand_mixed"]:
            copied["forbidden_for_client"] = True
            copied["bot_permission"] = "internal_only"
        chunks.append(copied)
    return chunks


def build_theme_mapping_stub(claude: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "current_theme_id": "",
            "current_theme_title": "",
            "claude_v5_number": "",
            "claude_v5_title": "",
            "mapping_status": "v5_not_imported_as_truth",
            "route_change": "none",
            "requires_code_change": "no",
            "requires_test_change": "no",
            "notes": "Опросник v5 использовать только для анализа до исправления Дмитрием.",
        }
    ]


def build_approval_queue_v2(facts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for fact in facts:
        if fact.get("allowed_for_client_answer") and not fact.get("cross_brand_mixed"):
            continue
        queue.append(
            {
                "priority": "P0" if fact.get("cross_brand_mixed") else "P1",
                "approval_item_id": fact.get("fact_id"),
                "brand": fact.get("brand"),
                "item_type": fact.get("fact_type"),
                "source_title": fact.get("source_title"),
                "manager_text": fact.get("manager_check_text"),
                "suggested_decision": "internal_only" if fact.get("cross_brand_mixed") else "review_before_client_use",
                "rop_question": "Можно ли использовать этот факт в ответе клиенту текущего бренда?",
                "bot_permission_after_approval": "allowed_for_client_answer",
                "risk_notes": fact.get("notes"),
            }
        )
    return queue


def build_snapshot_v2(
    *,
    run_id: str,
    sources: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    codex_snapshot: Mapping[str, Any],
    claude: Mapping[str, Any],
    theme_mapping: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "run_id": run_id,
        "generated_at": now,
        "metadata": {
            "purpose": "Интеграция Claude смыслового слоя и Codex безопасного машинного слоя.",
            "autonomous_client_answer_policy": "allowed_only_after_active_brand_filter_policy_route_and_checks",
            "brand_policy": "client_text_active_brand_only",
            "rop_v5_policy": "analysis_only_not_imported_as_truth",
        },
        "summary": {
            "sources_total": len(sources),
            "facts_total": len(facts),
            "chunks_total": len(chunks),
            "client_allowed_facts": sum(1 for fact in facts if fact.get("allowed_for_client_answer")),
            "usable_for_precise_answer": sum(1 for fact in facts if fact.get("usable_for_precise_answer")),
            "facts_by_brand": dict(Counter(str(fact.get("brand") or "") for fact in facts)),
            "facts_by_type": dict(Counter(str(fact.get("fact_type") or "") for fact in facts)),
            "cross_brand_internal_facts": sum(1 for fact in facts if fact.get("cross_brand_mixed")),
            "codex_v1_historical_questions": ((codex_snapshot.get("summary") or {}).get("historical_question_items_total") or 0),
            "codex_v1_answer_templates": ((codex_snapshot.get("summary") or {}).get("answer_templates_total") or 0),
            "codex_v1_manager_patterns": ((codex_snapshot.get("summary") or {}).get("manager_patterns_total") or 0),
        },
        "safety": {
            "send_client_message": False,
            "crm_write": False,
            "tallanto_write": False,
            "stable_runtime_write": False,
            "active_brand_required_for_precise_answer": True,
            "cross_brand_client_text_forbidden": True,
            "rop_v5_imported_as_truth": False,
            "bot_answer_self_allowed_only_in_test_rollout": True,
        },
        "sources": list(sources),
        "facts": list(facts),
        "chunks": list(chunks),
        "knowledge_chunks": list(chunks),
        "brand_rules": normalize_brand_rules(claude.get("brand_rules") or {}),
        "bot_policy": normalize_bot_policy(claude.get("bot_policy") or {}),
        "theme_mapping": list(theme_mapping),
        "codex_v1_summary": codex_snapshot.get("summary") or {},
    }


def build_quality_report(snapshot: Mapping[str, Any], *, approval_queue: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    facts = [fact for fact in snapshot.get("facts", []) if isinstance(fact, Mapping)]
    failures: list[str] = []
    if any(fact.get("brand") == "both" for fact in facts):
        failures.append("brand_both_is_forbidden")
    if any(fact.get("allowed_for_client_answer") and fact.get("cross_brand_mixed") for fact in facts):
        failures.append("cross_brand_fact_allowed_for_client")
    if any(fact.get("allowed_for_client_answer") and fact.get("brand") not in {"foton", "unpk", "brand_neutral"} for fact in facts):
        failures.append("invalid_client_brand_allowed")
    if any(
        fact.get("brand") == "brand_neutral" and fact.get("fact_type") in {"matkap", "tax"}
        for fact in facts
    ):
        failures.append("matkap_tax_must_be_brand_specific_or_internal")
    if any(
        fact.get("brand") == "brand_neutral"
        and not is_brand_neutral_candidate(f"{fact.get('title', '')} {fact.get('fact_text', '')}")
        for fact in facts
    ):
        failures.append("brand_neutral_contains_brand_specific_text")
    checks = {
        "no_brand_both": "brand_both_is_forbidden" not in failures,
        "cross_brand_mixed_internal_only": "cross_brand_fact_allowed_for_client" not in failures,
        "has_foton_facts": any(fact.get("brand") == "foton" for fact in facts),
        "has_unpk_facts": any(fact.get("brand") == "unpk" for fact in facts),
        "has_matkap_both_brands": has_fact_for_both_brands(facts, "matkap"),
        "has_tax_both_brands": has_fact_for_both_brands(facts, "tax"),
        "has_lvsh_mendeleevo_both_brands": has_product_for_both_brands(facts, "lvsh_mendeleevo"),
        "matkap_tax_not_brand_neutral": "matkap_tax_must_be_brand_specific_or_internal" not in failures,
        "brand_neutral_is_safe": "brand_neutral_contains_brand_specific_text" not in failures,
        "rop_v5_not_imported_as_truth": (snapshot.get("metadata") or {}).get("rop_v5_policy") == "analysis_only_not_imported_as_truth",
        "approval_queue_exists": bool(approval_queue),
    }
    if not all(checks.values()):
        for key, ok in checks.items():
            if not ok:
                failures.append(key)
    return {
        "schema_version": "kb_release_v2_quality_report_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "quality_passed": not failures,
        "blocking_failures": failures,
        "checks": checks,
        "interpretation": "v2 готов как кандидат для тестового Telegram-пилота после прохождения тестов.",
        "manual_review_required": {
            "approval_queue_items": len(approval_queue),
            "cross_brand_internal_facts": sum(1 for fact in facts if fact.get("cross_brand_mixed")),
        },
    }


def write_outputs(
    out_root: Path,
    agent_root: Path,
    *,
    snapshot: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    facts: Sequence[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    approval_queue: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    brand_rules: Mapping[str, Any],
    bot_policy: Mapping[str, Any],
    theme_mapping: Sequence[Mapping[str, Any]],
) -> None:
    write_json(out_root / "kb_release_v2_snapshot.json", snapshot)
    write_json(out_root / "source_registry.json", {"items": list(sources)})
    write_csv(out_root / "source_registry.csv", sources)
    write_jsonl(out_root / "facts_registry.jsonl", facts)
    write_csv(out_root / "facts_registry.csv", facts)
    write_yaml(out_root / "facts_registry.yaml", {"schema_version": "facts_registry_v2", "items": list(facts)})
    write_yaml(out_root / "brand_rules.yaml", brand_rules)
    write_yaml(out_root / "bot_policy.yaml", bot_policy)
    write_csv(out_root / "approval_queue_for_rop_v2.csv", approval_queue)
    write_csv(out_root / "rop_policy_v5_theme_mapping.csv", theme_mapping)
    write_json(out_root / "quality_report.json", quality)
    (out_root / "QUALITY_REPORT.md").write_text(render_quality_report_md(quality), encoding="utf-8")
    (out_root / "rop_policy_v5_import_summary.md").write_text(render_v5_summary_md(theme_mapping), encoding="utf-8")
    (out_root / "README.md").write_text(render_readme(), encoding="utf-8")

    write_json(agent_root / "kb_release_v2_snapshot.json", snapshot)
    write_yaml(agent_root / "facts_registry.yaml", {"schema_version": "facts_registry_v2", "items": list(facts)})
    write_yaml(agent_root / "brand_rules.yaml", brand_rules)
    write_yaml(agent_root / "bot_policy.yaml", bot_policy)
    for name, content in agent_pack_docs().items():
        (agent_root / name).write_text(content, encoding="utf-8")


def normalize_brand_rules(value: Mapping[str, Any]) -> dict[str, Any]:
    rules = dict(value)
    rules["schema_version"] = str(rules.get("schema_version") or "brand_rules_v2")
    rules["client_default_relationship_answer"] = "Это отдельные организации, по вашему вопросу сориентирую в рамках текущего учебного центра."
    rules["automatic_previous_cooperation_phrase_allowed"] = False
    rules["brand_values"] = ["foton", "unpk", "brand_neutral", "internal"]
    rules["client_text_rule"] = "Показывать клиенту только факты активного бренда."
    rules.setdefault("forbidden_client_phrasings", [])
    return rules


def normalize_bot_policy(value: Mapping[str, Any]) -> dict[str, Any]:
    policy = dict(value)
    policy["schema_version"] = str(policy.get("schema_version") or "bot_policy_v2")
    policy["rop_v5_imported_as_truth"] = False
    policy["bot_answer_self_rollout"] = {
        "enabled_for_staff_tests": True,
        "enabled_for_loyal_prepared_clients": True,
        "enabled_for_public_traffic": False,
    }
    policy["brand_decisions"] = {
        "matkap_available_for": ["foton", "unpk"],
        "tax_deduction_available_for": ["foton", "unpk"],
        "lvsh_mendeleevo_available_for": ["foton", "unpk"],
        "client_text_active_brand_only": True,
    }
    return policy


def load_source_registry(codex_v1_agent_pack: Path, *, inventory_path: Path | None = None) -> list[Mapping[str, Any]]:
    if inventory_path is not None:
        path = Path(inventory_path)
        if path.exists() and path.suffix.lower() == ".json":
            loaded = load_json(path)
            if isinstance(loaded, Mapping) and isinstance(loaded.get("items"), Sequence):
                return [dict(item) for item in loaded["items"] if isinstance(item, Mapping)]
            if isinstance(loaded, Sequence):
                return [dict(item) for item in loaded if isinstance(item, Mapping)]
        if path.exists() and path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                return list(csv.DictReader(file))
    json_path = codex_v1_agent_pack / "source_inventory.json"
    csv_path = codex_v1_agent_pack / "source_inventory.csv"
    if json_path.exists():
        loaded = load_json(json_path)
        if isinstance(loaded, Mapping) and isinstance(loaded.get("items"), Sequence):
            return [dict(item) for item in loaded["items"] if isinstance(item, Mapping)]
        if isinstance(loaded, Sequence):
            return [dict(item) for item in loaded if isinstance(item, Mapping)]
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))
    return []


def build_source_matcher(sources: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    matcher: dict[str, Mapping[str, Any]] = {}
    for source in sources:
        title = normalize_match_text(source.get("title") or source.get("source_title") or "")
        path = normalize_match_text(source.get("path") or "")
        url = normalize_match_text(source.get("url") or source.get("google_drive_url") or "")
        source_id = normalize_match_text(source.get("source_id") or "")
        for key in (title, path, url, source_id):
            if key:
                matcher.setdefault(key, source)
    return matcher


def match_source(hint: str, source_matcher: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    normalized = normalize_match_text(hint)
    if not normalized:
        return {"source_id": "source:claude_layer:unknown", "title": "Claude layer"}
    for key, source in source_matcher.items():
        if normalized in key or key in normalized:
            return source
    for marker in ("фотон", "foton"):
        if marker in normalized:
            for key, source in source_matcher.items():
                if marker in key and ("цен" in key or "price" in key):
                    return source
    for marker in ("унпк", "unpk"):
        if marker in normalized:
            for key, source in source_matcher.items():
                if marker in key and ("цен" in key or "price" in key):
                    return source
    return {"source_id": f"source:claude_layer:{safe_id(hint)}", "title": hint or "Claude layer"}


def normalize_freshness(value: Any) -> str:
    text = clean_text(value).casefold()
    mapping = {
        "verified": "document_verified",
        "fresh": "fresh_verified",
        "fresh_verified": "fresh_verified",
        "document_verified": "document_verified",
        "needs_owner_confirmation": "needs_owner_confirmation",
        "dynamic_needs_check": "dynamic_needs_check",
        "internal_only": "internal_only",
        "conflicting": "conflicting",
        "do_not_use": "do_not_use",
        "needs_fresh_doc": "dynamic_needs_check",
        "verified_outdated": "dynamic_needs_check",
    }
    return mapping.get(text, "document_verified" if text else "document_verified")


def normalize_brand(value: Any) -> str:
    text = clean_text(value).casefold().replace("ё", "е")
    if text in ALLOWED_BRANDS:
        return text
    if "foton" in text or "фотон" in text or "цдпо" in text or "црдо" in text:
        return "foton"
    if "unpk" in text or "унпк" in text or "мфти" in text or "ано" in text:
        return "unpk"
    if text in {"both", "оба", "all"}:
        return "brand_neutral"
    return "internal" if text == "internal_only" else "brand_neutral"


def active_brand_scope(brand: str, *, internal_only: bool = False) -> str:
    if internal_only or brand == "internal":
        return "internal_only"
    if brand == "foton":
        return "foton_bot"
    if brand == "unpk":
        return "unpk_bot"
    return "brand_neutral"


def infer_fact_type(text: str) -> str:
    normalized = clean_text(text).casefold().replace("ё", "е")
    for fact_type, markers in FACT_TYPE_MARKERS:
        if any(marker in normalized for marker in markers):
            return fact_type
    return "program"


def infer_product(key: str) -> str:
    text = clean_text(key).casefold()
    if "mendeleevo" in text or "lvsh" in text or "лвш" in text:
        return "lvsh_mendeleevo"
    if "matkap" in text or "маткап" in text:
        return "matkap"
    if "tax" in text or "налог" in text:
        return "tax"
    if "installment" in text or "рассроч" in text:
        return "installment"
    if "price" in text or "стоим" in text:
        return "regular_courses"
    if "intensive" in text or "интенсив" in text:
        return "intensive"
    return text.split(".")[0] if text else "general"


def theme_ids_for_fact_type(fact_type: str) -> list[str]:
    return {
        "price": ["theme:001_pricing"],
        "discount": ["theme:005_discounts"],
        "installment": ["theme:006_installment"],
        "matkap": ["theme:007_matkap_payment"],
        "tax": ["theme:008_tax_deduction"],
        "documents": ["theme:012_certificates"],
        "schedule": ["theme:013_schedule"],
        "program": ["theme:016_program_content"],
        "payment_status": ["theme:003_payment_status"],
        "location": ["theme:015_location_address"],
    }.get(fact_type, ["service:S2_unclear"])


def forbidden_promises_for_fact_type(fact_type: str) -> list[str]:
    base = ["Не обещать условия вне утвержденного факта."]
    if fact_type == "installment":
        base.append("Не обещать одобрение банка.")
    if fact_type in {"tax", "matkap"}:
        base.append("Не обещать одобрение СФР/ФНС или конкретный возврат денег.")
    if fact_type == "discount":
        base.append("Не обещать индивидуальную скидку без менеджера.")
    return base


def forbidden_mentions_for_brand(brand: str) -> list[str]:
    if brand == "foton":
        return ["УНПК", "УНПК МФТИ", "АНО ДПО", "НОУ УНПК", "kmipt.ru", "@unpk_mipt"]
    if brand == "unpk":
        return ["Фотон", "ЦДПО", "ЦРДО", "cdpofoton.ru", "Т-Банк рассрочка", "Долями"]
    return []


def is_cross_brand_text(text: str) -> bool:
    normalized = clean_text(text, max_chars=3000).casefold().replace("ё", "е")
    has_foton = any(marker in normalized for marker in BRAND_NAMES["foton"])
    has_unpk = any(marker in normalized for marker in BRAND_NAMES["unpk"])
    sensitive = any(marker in normalized for marker in ("цен", "сто", "скид", "рассроч", "договор", "лиценз", "услов", "перевод"))
    return bool(has_foton and has_unpk and sensitive)


def is_brand_neutral_candidate(text: str) -> bool:
    normalized = clean_text(text, max_chars=3000).casefold().replace("ё", "е")
    forbidden = (
        "фотон",
        "унпк",
        "мфти",
        "цдпо",
        "црдо",
        "ано дпо",
        "ноу",
        "kmipt",
        "cdpofoton",
        "т-банк",
        "долями",
        "руб",
        "₽",
    )
    return not any(marker in normalized for marker in forbidden)


def has_fact_for_both_brands(facts: Sequence[Mapping[str, Any]], fact_type: str) -> bool:
    brands = {str(fact.get("brand") or "") for fact in facts if str(fact.get("fact_type") or "") == fact_type}
    return {"foton", "unpk"} <= brands


def has_product_for_both_brands(facts: Sequence[Mapping[str, Any]], product: str) -> bool:
    brands = {str(fact.get("brand") or "") for fact in facts if str(fact.get("product") or "") == product}
    return {"foton", "unpk"} <= brands


def guard_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if "stable_runtime" in resolved.parts:
        raise ValueError("kb_release_v2 output must not be inside stable_runtime")
    return resolved


def load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path, *, tolerate_errors: bool = False) -> Any:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        if tolerate_errors:
            return {"__load_error__": f"{type(exc).__name__}: {exc}"}
        raise
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
        for key in row.keys():
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


def render_quality_report_md(quality: Mapping[str, Any]) -> str:
    checks = quality.get("checks") if isinstance(quality.get("checks"), Mapping) else {}
    lines = [
        "# Quality report kb_release_v2",
        "",
        f"quality_passed: `{quality.get('quality_passed')}`",
        "",
        "## Checks",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: `{value}`")
    failures = quality.get("blocking_failures") or []
    lines.extend(["", "## Blocking failures"])
    lines.extend([f"- {item}" for item in failures] or ["- none"])
    return "\n".join(lines) + "\n"


def render_v5_summary_md(theme_mapping: Sequence[Mapping[str, Any]]) -> str:
    return (
        "# RoP v5 import summary\n\n"
        "Опросник v5 не импортирован как готовая политика. Он используется только для анализа до исправления Дмитрием.\n\n"
        f"Строк mapping: `{len(theme_mapping)}`.\n"
    )


def render_readme() -> str:
    return (
        "# kb_release_20260517_v2\n\n"
        "Интегрированная база знаний Claude + Codex для тестового Telegram-пилота.\n\n"
        "Главное правило: клиентский текст использует только факты активного бренда. "
        "Смешанные брендовые фрагменты доступны только внутренне.\n"
    )


def agent_pack_docs() -> Mapping[str, str]:
    return {
        "README_FOR_EMPLOYEES_RU.md": "# База знаний v2\n\nПакет предназначен для ИИ-помощников и сотрудников. Клиентские ответы фильтруются по активному бренду.\n",
        "AI_AGENT_PROMPT_RU.md": "# Инструкция ИИ-агенту\n\nИспользуй `kb_release_v2_snapshot.json`. Никогда не смешивай Фотон и УНПК в клиентском ответе.\n",
        "AI_AGENT_USAGE_RU.md": "# Как пользоваться\n\nПередавай агенту snapshot, facts_registry, brand_rules и bot_policy. Для клиента используй только active_brand.\n",
        "SAFETY_RULES_RU.md": "# Правила безопасности\n\nCross-brand информация только внутренняя. Возврат, жалоба, суд и претензия — менеджеру.\n",
        "SHORT_CONTEXT_RU.md": "База v2: факты разделены по брендам; маткапитал, налоговый вычет и ЛВШ доступны обоим брендам через отдельные брендовые факты.\n",
        "BOT_USAGE_CONTRACT.md": "# Контракт использования\n\nБез active_brand нельзя давать точные брендовые условия. Клиенту ничего не отправлять без разрешённого режима.\n",
        "FILES_GUIDE_RU.md": "# Файлы\n\n`kb_release_v2_snapshot.json` — основной снимок. `facts_registry.yaml` — факты. `brand_rules.yaml` — правила брендов. `bot_policy.yaml` — маршруты.\n",
    }


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def safe_id(value: Any) -> str:
    text = clean_text(value).casefold().replace("ё", "е")
    text = re.sub(r"[^0-9a-zа-я]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")[:90] or "item"


def clean_text(value: Any, *, max_chars: int = 500) -> str:
    text = " ".join(str(value or "").replace("\u00a0", " ").split())
    return text[:max_chars]


def normalize_match_text(value: Any) -> str:
    return clean_text(value, max_chars=1000).casefold().replace("ё", "е")


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да", "истина"}


if __name__ == "__main__":
    raise SystemExit(main())
