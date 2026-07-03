from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from mango_mvp.customer_profile.builder import child_name_keys, normalized_name_tokens
from mango_mvp.customer_timeline.ids import normalize_key, stable_digest, stable_prefixed_id
from mango_mvp.customer_timeline.store import guard_customer_timeline_sqlite_path


FAMILY_GRAPH_SCHEMA_VERSION = "family_graph_v1"
FAMILY_GRAPH_RUN_KIND = "family_graph_v1"
FAMILY_GRAPH_ACTOR = "family_graph_v1_builder"
VALID_BRANDS = {"foton", "unpk", "unknown"}
CHILD_RELEVANT_RE = re.compile(
    r"\b(?:реб[её]нок|дет[еи]|сын|дочь|дочка|ученик|ученица|класс|егэ|огэ|"
    r"математик|физик|информатик|русск|английск|курс|заняти|школ)\w*\b",
    re.I,
)
INITIALS_RE = re.compile(r"\b[А-ЯЁA-Z]\.\s*[А-ЯЁA-Z]\.", re.U)
EMAIL_OR_PHONE_RE = re.compile(r"@|\+?\d[\d\s().-]{5,}")
NON_CHILD_NAME_RE = re.compile(
    r"\b(?:преподавател|куратор|менеджер|администратор|школа|центр|родител|мама|папа|"
    r"бабушка|дедушка|семья|клиент)\w*\b",
    re.I,
)


@dataclass(frozen=True)
class FamilyGraphConfig:
    timeline_db: Path
    allowed_root: Path
    out_path: Optional[Path] = None
    profiles_db: Optional[Path] = None
    tenant_id: str = "foton"
    apply: bool = False
    customer_ids: tuple[str, ...] = ()


@dataclass
class ChildEvidence:
    source_system: str
    source_ref: str
    event_at: str
    name: str = ""
    grade: str = ""
    subject: str = ""
    brand: str = "unknown"
    quote: str = ""


@dataclass
class ChildGroup:
    name_key: str
    names: set[str] = field(default_factory=set)
    grades: set[str] = field(default_factory=set)
    subjects: set[str] = field(default_factory=set)
    brands: Counter[str] = field(default_factory=Counter)
    evidence: list[ChildEvidence] = field(default_factory=list)
    suspicious_reasons: set[str] = field(default_factory=set)

    @property
    def canonical_name(self) -> str:
        if not self.names:
            return ""
        return sorted(self.names, key=lambda value: (len(value), value.casefold()))[0]

    @property
    def brand(self) -> str:
        known = {key: count for key, count in self.brands.items() if key in {"foton", "unpk"} and count > 0}
        if len(known) == 1:
            return next(iter(known))
        return "unknown"

    @property
    def source_refs(self) -> tuple[str, ...]:
        return tuple(sorted({item.source_ref for item in self.evidence if item.source_ref}))


@dataclass(frozen=True)
class CustomerContext:
    customer_id: str
    tenant_id: str
    identity_status: str
    display_name: str
    primary_phone: str
    primary_email: str
    shared_family_phone: bool
    parent_name_keys: frozenset[str]


def build_family_graph(config: FamilyGraphConfig) -> Mapping[str, Any]:
    tenant_id = normalize_key(config.tenant_id, "tenant_id")
    db_path = _guard_db(config.timeline_db, apply=config.apply)
    generated_at = _stable_generated_at(db_path)
    with _connect(db_path, write=config.apply) as con:
        customers = _load_customers(con, tenant_id=tenant_id, customer_ids=config.customer_ids)
        shared_customers = _shared_family_phone_customers(con, tenant_id=tenant_id)
        contexts = {
            customer_id: CustomerContext(
                customer_id=customer_id,
                tenant_id=tenant_id,
                identity_status=str(row["identity_status"] or ""),
                display_name=str(row["display_name"] or ""),
                primary_phone=str(row["primary_phone"] or ""),
                primary_email=str(row["primary_email"] or ""),
                shared_family_phone=customer_id in shared_customers,
                parent_name_keys=frozenset(),
            )
            for customer_id, row in customers.items()
        }
        evidence_by_customer: dict[str, list[ChildEvidence]] = defaultdict(list)
        profile_report: Mapping[str, Any] = {"profiles_db": None, "fields_seen": 0, "mapped_fields": 0}
        if config.profiles_db:
            profile_report, profile_evidence, parent_keys = _load_profile_evidence(
                con,
                profiles_db=Path(config.profiles_db),
                tenant_id=tenant_id,
                customer_ids=set(customers),
            )
            for customer_id, items in profile_evidence.items():
                evidence_by_customer[customer_id].extend(items)
            for customer_id, keys in parent_keys.items():
                if customer_id in contexts:
                    ctx = contexts[customer_id]
                    contexts[customer_id] = CustomerContext(
                        customer_id=ctx.customer_id,
                        tenant_id=ctx.tenant_id,
                        identity_status=ctx.identity_status,
                        display_name=ctx.display_name,
                        primary_phone=ctx.primary_phone,
                        primary_email=ctx.primary_email,
                        shared_family_phone=ctx.shared_family_phone,
                        parent_name_keys=frozenset(keys),
                    )
        mail_report, mail_evidence = _load_mail_fact_evidence(con, tenant_id=tenant_id, customer_ids=set(customers))
        for customer_id, items in mail_evidence.items():
            evidence_by_customer[customer_id].extend(items)

        family_rows, groups_by_customer = _build_family_rows(contexts, evidence_by_customer, generated_at=generated_at)
        event_rows = _build_event_attributions(con, tenant_id=tenant_id, groups_by_customer=groups_by_customer, contexts=contexts, generated_at=generated_at)
        opportunity_rows = _build_opportunity_attributions(
            con,
            tenant_id=tenant_id,
            groups_by_customer=groups_by_customer,
            contexts=contexts,
            generated_at=generated_at,
        )
        summary = _summary(
            tenant_id=tenant_id,
            db_path=db_path,
            generated_at=generated_at,
            profile_report=profile_report,
            mail_report=mail_report,
            family_rows=family_rows,
            event_rows=event_rows,
            opportunity_rows=opportunity_rows,
            apply=config.apply,
        )
        if config.apply:
            _ensure_schema(con)
            _replace_rows(con, "family_links_v1", tenant_id, family_rows)
            _replace_rows(con, "event_child_attribution_v1", tenant_id, event_rows)
            _replace_rows(con, "opportunity_child_attribution_v1", tenant_id, opportunity_rows)
            _record_run(con, tenant_id=tenant_id, generated_at=generated_at, summary=summary)
            con.commit()
            summary = {**summary, "write_applied": True, "quick_check": con.execute("PRAGMA quick_check").fetchone()[0]}
    if config.out_path:
        out_path = _guard_out(config.out_path, config.allowed_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_json_dumps(summary), encoding="utf-8")
    return summary


def _guard_db(path: Path, *, apply: bool) -> Path:
    resolved = guard_customer_timeline_sqlite_path(path)
    if "customer_timeline_prod_20260621" in resolved.parts:
        raise ValueError("family graph must not open prod timeline DB")
    if apply and (".codex_local" not in resolved.parts or "staging" not in resolved.parts):
        raise ValueError("family graph apply requires DB under .codex_local/staging")
    return resolved


def _guard_out(path: Path, allowed_root: Path) -> Path:
    allowed = Path(allowed_root).expanduser().resolve(strict=False)
    resolved = Path(path).expanduser().resolve(strict=False)
    if not _is_relative_to(resolved, allowed):
        raise ValueError(f"output path must be under allowed root: {resolved}")
    return resolved


def _connect(path: Path, *, write: bool) -> sqlite3.Connection:
    if write:
        con = sqlite3.connect(path, timeout=30)
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA busy_timeout = 30000")
    else:
        uri = f"{path.as_uri()}?mode=ro&immutable=1"
        con = sqlite3.connect(uri, uri=True, timeout=15)
        con.execute("PRAGMA query_only = ON")
    con.row_factory = sqlite3.Row
    return con


def _load_customers(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_ids: Sequence[str],
) -> dict[str, sqlite3.Row]:
    params: list[Any] = [tenant_id]
    filter_sql = ""
    if customer_ids:
        placeholders = ",".join("?" for _ in customer_ids)
        filter_sql = f" AND customer_id IN ({placeholders})"
        params.extend(customer_ids)
    rows = con.execute(
        f"""
        SELECT customer_id, tenant_id, identity_status, display_name, primary_phone, primary_email
        FROM customer_identities
        WHERE tenant_id = ?{filter_sql}
        ORDER BY customer_id
        """,
        tuple(params),
    ).fetchall()
    return {str(row["customer_id"]): row for row in rows}


def _shared_family_phone_customers(con: sqlite3.Connection, *, tenant_id: str) -> set[str]:
    result: set[str] = set()
    rows = con.execute(
        """
        SELECT record_json
        FROM timeline_conflicts
        WHERE tenant_id = ?
          AND conflict_type = 'shared_family_phone'
          AND status = 'open'
        """,
        (tenant_id,),
    ).fetchall()
    for row in rows:
        payload = _json_loads(row["record_json"])
        refs = payload.get("entity_refs") if isinstance(payload.get("entity_refs"), list) else []
        for ref in refs:
            text = str(ref or "")
            if text.startswith("customer:customer:"):
                result.add(text.removeprefix("customer:"))
            elif text.startswith("customer:"):
                result.add(text)
    return result


def _load_profile_evidence(
    con: sqlite3.Connection,
    *,
    profiles_db: Path,
    tenant_id: str,
    customer_ids: set[str],
) -> tuple[Mapping[str, Any], dict[str, list[ChildEvidence]], dict[str, set[str]]]:
    if not profiles_db.exists():
        return {"profiles_db": str(profiles_db), "missing": True, "fields_seen": 0, "mapped_fields": 0}, {}, {}
    con.execute("ATTACH DATABASE ? AS family_profile_source", (str(profiles_db),))
    try:
        phone_to_customers = _phone_to_unique_customer(con, tenant_id=tenant_id)
        profile_to_customer: dict[str, str] = {}
        profile_rows = con.execute(
            """
            SELECT profile_id, tenant_id, primary_phone
            FROM family_profile_source.customer_profiles
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        ).fetchall()
        for row in profile_rows:
            profile_id = str(row["profile_id"])
            if profile_id in customer_ids:
                profile_to_customer[profile_id] = profile_id
                continue
            phone_customer = phone_to_customers.get(str(row["primary_phone"] or ""))
            if phone_customer:
                profile_to_customer[profile_id] = phone_customer
        if not profile_to_customer:
            return {"profiles_db": str(profiles_db), "fields_seen": 0, "mapped_fields": 0, "mapped_profiles": 0}, {}, {}
        placeholders = ",".join("?" for _ in profile_to_customer)
        rows = con.execute(
            f"""
            SELECT profile_id, field, value, child_key, brand, source_system, source_ref, event_at, quote
            FROM family_profile_source.profile_fields
            WHERE profile_id IN ({placeholders})
              AND superseded_by = ''
              AND field IN ('child_name', 'grade', 'subject', 'parent_name')
            ORDER BY profile_id, child_key, event_at, source_ref
            """,
            tuple(profile_to_customer),
        ).fetchall()
    finally:
        con.execute("DETACH DATABASE family_profile_source")

    evidence_by_slot: dict[tuple[str, str], list[sqlite3.Row]] = defaultdict(list)
    parent_keys: dict[str, set[str]] = defaultdict(set)
    fields_seen = 0
    mapped_fields = 0
    for row in rows:
        fields_seen += 1
        customer_id = profile_to_customer.get(str(row["profile_id"]))
        if not customer_id:
            continue
        mapped_fields += 1
        if str(row["field"]) == "parent_name":
            parent_keys[customer_id].update(_safe_name_keys(str(row["value"] or "")))
            continue
        child_key = str(row["child_key"] or "")
        if not child_key:
            continue
        evidence_by_slot[(customer_id, child_key)].append(row)

    evidence: dict[str, list[ChildEvidence]] = defaultdict(list)
    for (customer_id, _child_key), slot_rows in evidence_by_slot.items():
        names = [str(row["value"] or "").strip() for row in slot_rows if str(row["field"]) == "child_name" and str(row["value"] or "").strip()]
        grades = [str(row["value"] or "").strip() for row in slot_rows if str(row["field"]) == "grade" and str(row["value"] or "").strip()]
        subjects = [str(row["value"] or "").strip() for row in slot_rows if str(row["field"]) == "subject" and str(row["value"] or "").strip()]
        source = slot_rows[-1]
        evidence[customer_id].append(
            ChildEvidence(
                source_system=str(source["source_system"] or "customer_profile"),
                source_ref=str(source["source_ref"] or source["profile_id"]),
                event_at=str(source["event_at"] or ""),
                name=names[-1] if names else "",
                grade=grades[-1] if grades else "",
                subject="; ".join(_dedupe(subjects)),
                brand=_normalize_brand(source["brand"]),
                quote=str(source["quote"] or ""),
            )
        )
    return (
        {
            "profiles_db": str(profiles_db),
            "fields_seen": fields_seen,
            "mapped_fields": mapped_fields,
            "mapped_profiles": len(set(profile_to_customer.values())),
            "profile_rows": len(profile_rows),
        },
        evidence,
        parent_keys,
    )


def _phone_to_unique_customer(con: sqlite3.Connection, *, tenant_id: str) -> dict[str, str]:
    rows = con.execute(
        """
        SELECT primary_phone, COUNT(DISTINCT customer_id) AS n, MIN(customer_id) AS customer_id
        FROM customer_identities
        WHERE tenant_id = ? AND COALESCE(primary_phone, '') != ''
        GROUP BY primary_phone
        HAVING n = 1
        """,
        (tenant_id,),
    ).fetchall()
    return {str(row["primary_phone"]): str(row["customer_id"]) for row in rows}


def _load_mail_fact_evidence(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_ids: set[str],
) -> tuple[Mapping[str, Any], dict[str, list[ChildEvidence]]]:
    if not _table_exists(con, "a2v3_mail_event_facts"):
        return {"table": "a2v3_mail_event_facts", "missing": True, "student_name_rows": 0}, {}
    rows = con.execute(
        """
        SELECT event_id, customer_id, student_name, grade, subject_area, email_brand
        FROM a2v3_mail_event_facts
        WHERE tenant_id = ?
          AND customer_id IS NOT NULL
          AND customer_id != ''
          AND (
            COALESCE(student_name, '') != ''
            OR COALESCE(grade, '') != ''
            OR COALESCE(subject_area, '') != ''
          )
        ORDER BY customer_id, event_id
        """,
        (tenant_id,),
    ).fetchall()
    evidence: dict[str, list[ChildEvidence]] = defaultdict(list)
    rows_with_student = 0
    for row in rows:
        customer_id = str(row["customer_id"] or "")
        if customer_id not in customer_ids:
            continue
        name = str(row["student_name"] or "").strip()
        if name:
            rows_with_student += 1
            evidence[customer_id].append(
                ChildEvidence(
                    source_system="a2v3_mail_event_facts",
                    source_ref=str(row["event_id"]),
                    event_at="",
                    name=name,
                    grade=str(row["grade"] or ""),
                    subject=str(row["subject_area"] or ""),
                    brand=_normalize_brand(row["email_brand"]),
                )
            )
    return {
        "table": "a2v3_mail_event_facts",
        "fact_rows_with_child_fields": len(rows),
        "student_name_rows_mapped": rows_with_student,
    }, evidence


def _build_family_rows(
    contexts: Mapping[str, CustomerContext],
    evidence_by_customer: Mapping[str, Sequence[ChildEvidence]],
    *,
    generated_at: str,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    groups_by_customer: dict[str, list[dict[str, Any]]] = {}
    for customer_id, context in contexts.items():
        groups = _child_groups_for_customer(context, evidence_by_customer.get(customer_id, ()))
        groups_by_customer[customer_id] = []
        valid_groups = [group for group in groups.values() if not group.suspicious_reasons and group.canonical_name]
        identity_risks = _identity_risks(context)
        for group in sorted(groups.values(), key=lambda item: (item.canonical_name.casefold(), item.name_key)):
            child_key = _child_key(customer_id, group.name_key)
            status, confidence, reason = _family_confidence(group, valid_groups=valid_groups, identity_risks=identity_risks)
            payload = {
                "schema_version": FAMILY_GRAPH_SCHEMA_VERSION,
                "tenant_id": context.tenant_id,
                "family_id": _family_id(context.tenant_id, customer_id),
                "customer_id": customer_id,
                "child_key": child_key,
                "canonical_name": group.canonical_name,
                "name_variants": sorted(group.names),
                "grades": sorted(group.grades),
                "subjects": sorted(group.subjects),
                "brand": group.brand,
                "status": status,
                "confidence": confidence,
                "reason": reason,
                "suspicious_reasons": sorted(group.suspicious_reasons),
                "identity_risks": identity_risks,
                "source_refs": list(group.source_refs[:12]),
                "evidence_count": len(group.evidence),
                "created_at": generated_at,
            }
            row = {
                "tenant_id": context.tenant_id,
                "family_id": payload["family_id"],
                "customer_id": customer_id,
                "child_key": child_key,
                "canonical_name": group.canonical_name,
                "name_variants_json": json.dumps(payload["name_variants"], ensure_ascii=False, sort_keys=True),
                "grades_json": json.dumps(payload["grades"], ensure_ascii=False, sort_keys=True),
                "subjects_json": json.dumps(payload["subjects"], ensure_ascii=False, sort_keys=True),
                "brand": group.brand,
                "status": status,
                "confidence": confidence,
                "reason": reason,
                "source_refs_json": json.dumps(payload["source_refs"], ensure_ascii=False, sort_keys=True),
                "evidence_count": len(group.evidence),
                "created_at": generated_at,
                "record_hash": stable_digest(payload),
                "record_json": _json_dumps(payload),
            }
            rows.append(row)
            groups_by_customer[customer_id].append({**payload, **{"child_key": child_key}})
    return rows, groups_by_customer


def _child_groups_for_customer(context: CustomerContext, evidence_items: Sequence[ChildEvidence]) -> dict[str, ChildGroup]:
    groups: dict[str, ChildGroup] = {}
    for item in evidence_items:
        if not item.name.strip():
            continue
        name_key = _name_key(item.name)
        if not name_key:
            continue
        group = groups.setdefault(name_key, ChildGroup(name_key=name_key))
        group.names.add(item.name.strip())
        if item.grade.strip():
            group.grades.add(item.grade.strip())
        for subject in _split_subjects(item.subject):
            group.subjects.add(subject)
        group.brands[_normalize_brand(item.brand)] += 1
        group.evidence.append(item)
        group.suspicious_reasons.update(_suspicious_name_reasons(item.name, parent_name_keys=context.parent_name_keys))
    return groups


def _identity_risks(context: CustomerContext) -> list[str]:
    risks: list[str] = []
    if context.identity_status != "strong":
        risks.append(f"identity_status:{context.identity_status or 'unknown'}")
    if context.shared_family_phone:
        risks.append("shared_family_phone")
    return risks


def _family_confidence(
    group: ChildGroup,
    *,
    valid_groups: Sequence[ChildGroup],
    identity_risks: Sequence[str],
) -> tuple[str, str, str]:
    if group.suspicious_reasons:
        return "excluded", "low", "suspicious_child_name"
    if identity_risks:
        return "ambiguous", "low", "identity_risk"
    if len(valid_groups) == 1:
        return "confident", "high", "single_child_family"
    if len(valid_groups) > 1:
        return "needs_review", "medium", "multiple_child_candidates"
    return "unknown", "low", "no_valid_child"


def _build_event_attributions(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    groups_by_customer: Mapping[str, Sequence[Mapping[str, Any]]],
    contexts: Mapping[str, CustomerContext],
    generated_at: str,
) -> list[dict[str, Any]]:
    customer_ids = [customer_id for customer_id, groups in groups_by_customer.items() if groups]
    if not customer_ids:
        return []
    rows: list[dict[str, Any]] = []
    for chunk in _chunks(customer_ids, 900):
        placeholders = ",".join("?" for _ in chunk)
        events = con.execute(
            f"""
            SELECT event_id, tenant_id, customer_id, opportunity_id, event_type, subject, text_preview, summary, record_json
            FROM timeline_events
            WHERE tenant_id = ?
              AND customer_id IN ({placeholders})
              AND superseded_by IS NULL
            ORDER BY customer_id, event_at, event_id
            """,
            (tenant_id, *chunk),
        ).fetchall()
        for event in events:
            customer_id = str(event["customer_id"])
            attribution = _attribute_text(
                groups_by_customer.get(customer_id, ()),
                _event_text(event),
                context=contexts[customer_id],
                object_kind="event",
                event_type=str(event["event_type"] or ""),
            )
            if not attribution:
                continue
            payload = {
                "schema_version": FAMILY_GRAPH_SCHEMA_VERSION,
                "tenant_id": tenant_id,
                "event_id": str(event["event_id"]),
                "customer_id": customer_id,
                **attribution,
                "created_at": generated_at,
            }
            rows.append(_attribution_row("event_child_attribution_v1", payload))
    return rows


def _build_opportunity_attributions(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    groups_by_customer: Mapping[str, Sequence[Mapping[str, Any]]],
    contexts: Mapping[str, CustomerContext],
    generated_at: str,
) -> list[dict[str, Any]]:
    customer_ids = [customer_id for customer_id, groups in groups_by_customer.items() if groups]
    if not customer_ids:
        return []
    rows: list[dict[str, Any]] = []
    for chunk in _chunks(customer_ids, 900):
        placeholders = ",".join("?" for _ in chunk)
        opportunities = con.execute(
            f"""
            SELECT opportunity_id, tenant_id, customer_id, title, status, record_json
            FROM customer_opportunities
            WHERE tenant_id = ?
              AND customer_id IN ({placeholders})
            ORDER BY customer_id, opened_at, opportunity_id
            """,
            (tenant_id, *chunk),
        ).fetchall()
        for opp in opportunities:
            customer_id = str(opp["customer_id"])
            attribution = _attribute_text(
                groups_by_customer.get(customer_id, ()),
                " ".join(str(opp[key] or "") for key in ("title", "status", "record_json")),
                context=contexts[customer_id],
                object_kind="opportunity",
                event_type="opportunity",
            )
            if not attribution:
                continue
            payload = {
                "schema_version": FAMILY_GRAPH_SCHEMA_VERSION,
                "tenant_id": tenant_id,
                "opportunity_id": str(opp["opportunity_id"]),
                "customer_id": customer_id,
                **attribution,
                "created_at": generated_at,
            }
            rows.append(_attribution_row("opportunity_child_attribution_v1", payload))
    return rows


def _attribute_text(
    groups: Sequence[Mapping[str, Any]],
    text: str,
    *,
    context: CustomerContext,
    object_kind: str,
    event_type: str,
) -> Optional[dict[str, Any]]:
    usable = [group for group in groups if group.get("status") in {"confident", "needs_review"}]
    if not usable:
        return None
    identity_risks = _identity_risks(context)
    normalized = _normalize_match_text(text)
    matches = [
        group
        for group in usable
        if any(_name_mentioned(normalized, value) for value in [group.get("canonical_name", ""), *group.get("name_variants", [])])
    ]
    if identity_risks:
        return {
            "child_key": "",
            "status": "ambiguous",
            "confidence": "low",
            "reason": "identity_risk",
            "matched_names": [group.get("canonical_name", "") for group in matches],
        }
    if len(matches) == 1:
        return {
            "child_key": str(matches[0].get("child_key") or ""),
            "status": "matched",
            "confidence": "medium" if len(usable) > 1 else "high",
            "reason": "unique_child_name_mention",
            "matched_names": [str(matches[0].get("canonical_name") or "")],
        }
    if len(matches) > 1:
        return {
            "child_key": "",
            "status": "ambiguous",
            "confidence": "low",
            "reason": "multiple_child_name_mentions",
            "matched_names": [str(group.get("canonical_name") or "") for group in matches],
        }
    if len(usable) == 1 and usable[0].get("confidence") == "high":
        return {
            "child_key": str(usable[0].get("child_key") or ""),
            "status": "matched",
            "confidence": "high",
            "reason": "single_child_family",
            "matched_names": [],
        }
    if _child_relevant_text(normalized, event_type=event_type, object_kind=object_kind):
        return {
            "child_key": "",
            "status": "ambiguous",
            "confidence": "low",
            "reason": "child_relevant_but_no_unique_name",
            "matched_names": [],
        }
    return None


def _attribution_row(table: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    key_name = "event_id" if table == "event_child_attribution_v1" else "opportunity_id"
    return {
        "tenant_id": str(payload["tenant_id"]),
        key_name: str(payload[key_name]),
        "customer_id": str(payload["customer_id"]),
        "child_key": str(payload.get("child_key") or ""),
        "status": str(payload["status"]),
        "confidence": str(payload["confidence"]),
        "reason": str(payload["reason"]),
        "evidence_json": json.dumps({"matched_names": payload.get("matched_names", [])}, ensure_ascii=False, sort_keys=True),
        "created_at": str(payload["created_at"]),
        "record_hash": stable_digest(payload),
        "record_json": _json_dumps(payload),
    }


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS family_links_v1 (
          tenant_id TEXT NOT NULL,
          family_id TEXT NOT NULL,
          customer_id TEXT NOT NULL,
          child_key TEXT NOT NULL,
          canonical_name TEXT NOT NULL,
          name_variants_json TEXT NOT NULL,
          grades_json TEXT NOT NULL,
          subjects_json TEXT NOT NULL,
          brand TEXT NOT NULL,
          status TEXT NOT NULL,
          confidence TEXT NOT NULL,
          reason TEXT NOT NULL,
          source_refs_json TEXT NOT NULL,
          evidence_count INTEGER NOT NULL,
          created_at TEXT NOT NULL,
          record_hash TEXT NOT NULL,
          record_json TEXT NOT NULL,
          PRIMARY KEY (tenant_id, customer_id, child_key)
        );
        CREATE INDEX IF NOT EXISTS ix_family_links_v1_customer
          ON family_links_v1(tenant_id, customer_id, status, confidence);
        CREATE TABLE IF NOT EXISTS event_child_attribution_v1 (
          tenant_id TEXT NOT NULL,
          event_id TEXT NOT NULL,
          customer_id TEXT NOT NULL,
          child_key TEXT NOT NULL,
          status TEXT NOT NULL,
          confidence TEXT NOT NULL,
          reason TEXT NOT NULL,
          evidence_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          record_hash TEXT NOT NULL,
          record_json TEXT NOT NULL,
          PRIMARY KEY (tenant_id, event_id)
        );
        CREATE INDEX IF NOT EXISTS ix_event_child_attr_customer
          ON event_child_attribution_v1(tenant_id, customer_id, status, confidence);
        CREATE TABLE IF NOT EXISTS opportunity_child_attribution_v1 (
          tenant_id TEXT NOT NULL,
          opportunity_id TEXT NOT NULL,
          customer_id TEXT NOT NULL,
          child_key TEXT NOT NULL,
          status TEXT NOT NULL,
          confidence TEXT NOT NULL,
          reason TEXT NOT NULL,
          evidence_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          record_hash TEXT NOT NULL,
          record_json TEXT NOT NULL,
          PRIMARY KEY (tenant_id, opportunity_id)
        );
        CREATE INDEX IF NOT EXISTS ix_opportunity_child_attr_customer
          ON opportunity_child_attribution_v1(tenant_id, customer_id, status, confidence);
        CREATE TABLE IF NOT EXISTS family_graph_runs_v1 (
          tenant_id TEXT NOT NULL,
          run_kind TEXT NOT NULL,
          generated_at TEXT NOT NULL,
          summary_json TEXT NOT NULL,
          record_hash TEXT NOT NULL,
          PRIMARY KEY (tenant_id, run_kind)
        );
        """
    )


def _replace_rows(con: sqlite3.Connection, table: str, tenant_id: str, rows: Sequence[Mapping[str, Any]]) -> None:
    con.execute(f"DELETE FROM {table} WHERE tenant_id = ?", (tenant_id,))
    if not rows:
        return
    columns = list(rows[0].keys())
    placeholders = ",".join("?" for _ in columns)
    con.executemany(
        f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
        [tuple(row[column] for column in columns) for row in rows],
    )


def _record_run(con: sqlite3.Connection, *, tenant_id: str, generated_at: str, summary: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": FAMILY_GRAPH_SCHEMA_VERSION,
        "tenant_id": tenant_id,
        "run_kind": FAMILY_GRAPH_RUN_KIND,
        "generated_at": generated_at,
        "summary": dict(summary),
    }
    con.execute(
        """
        INSERT INTO family_graph_runs_v1 (tenant_id, run_kind, generated_at, summary_json, record_hash)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(tenant_id, run_kind) DO UPDATE SET
          generated_at = excluded.generated_at,
          summary_json = excluded.summary_json,
          record_hash = excluded.record_hash
        """,
        (tenant_id, FAMILY_GRAPH_RUN_KIND, generated_at, _json_dumps(payload), stable_digest(payload)),
    )


def _summary(
    *,
    tenant_id: str,
    db_path: Path,
    generated_at: str,
    profile_report: Mapping[str, Any],
    mail_report: Mapping[str, Any],
    family_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    opportunity_rows: Sequence[Mapping[str, Any]],
    apply: bool,
) -> Mapping[str, Any]:
    return {
        "schema_version": FAMILY_GRAPH_SCHEMA_VERSION,
        "tenant_id": tenant_id,
        "timeline_db": str(db_path),
        "generated_at": generated_at,
        "write_applied": bool(apply),
        "llm_calls_total": 0,
        "profile_source": dict(profile_report),
        "mail_source": dict(mail_report),
        "family_links_total": len(family_rows),
        "family_status_counts": _counts(row["status"] for row in family_rows),
        "family_confidence_counts": _counts(row["confidence"] for row in family_rows),
        "customers_with_family_links": len({str(row["customer_id"]) for row in family_rows}),
        "event_attributions_total": len(event_rows),
        "event_status_counts": _counts(row["status"] for row in event_rows),
        "event_reason_counts": _counts(row["reason"] for row in event_rows),
        "opportunity_attributions_total": len(opportunity_rows),
        "opportunity_status_counts": _counts(row["status"] for row in opportunity_rows),
    }


def _event_text(row: sqlite3.Row) -> str:
    return " ".join(str(row[key] or "") for key in ("subject", "text_preview", "summary"))


def _child_relevant_text(text: str, *, event_type: str, object_kind: str) -> bool:
    if CHILD_RELEVANT_RE.search(text):
        return True
    if object_kind == "opportunity" and event_type == "opportunity":
        return True
    return event_type in {"mango_call", "email_message", "amo_deal_stage", "tallanto_payment", "tallanto_abonement"}


def _name_mentioned(normalized_text: str, name: Any) -> bool:
    keys = _safe_name_keys(str(name or ""))
    if not keys:
        return False
    text_tokens = set(normalized_name_tokens(normalized_text))
    canonical_text_tokens = set(text_tokens)
    for token in tuple(text_tokens):
        canonical_text_tokens.update(_safe_name_keys(token))
    return any(key in canonical_text_tokens for key in keys)


def _name_key(value: str) -> str:
    keys = _safe_name_keys(value)
    if len(keys) == 1:
        return next(iter(keys))
    normalized = "_".join(normalized_name_tokens(value))
    if normalized:
        return normalized[:80]
    return ""


def _safe_name_keys(value: str) -> set[str]:
    try:
        return set(child_name_keys({value}))
    except Exception:
        return set()


def _suspicious_name_reasons(value: str, *, parent_name_keys: frozenset[str]) -> set[str]:
    text = str(value or "").strip()
    reasons: set[str] = set()
    if not text:
        reasons.add("empty_name")
    if len(text) > 80:
        reasons.add("too_long_name")
    if EMAIL_OR_PHONE_RE.search(text):
        reasons.add("contact_value_not_name")
    if INITIALS_RE.search(text):
        reasons.add("initials_possible_adult_or_teacher")
    if NON_CHILD_NAME_RE.search(text):
        reasons.add("role_or_non_child_token")
    keys = _safe_name_keys(text)
    if keys and parent_name_keys and keys & set(parent_name_keys):
        reasons.add("same_as_parent_name")
    return reasons


def _split_subjects(value: str) -> list[str]:
    parts = re.split(r"[;,/|]+", str(value or ""))
    return _dedupe(part.strip() for part in parts if part.strip())


def _normalize_match_text(value: Any) -> str:
    return str(value or "").replace("ё", "е").casefold()


def _normalize_brand(value: Any) -> str:
    text = normalize_key(str(value or "unknown"), "brand")
    return text if text in VALID_BRANDS else "unknown"


def _stable_generated_at(db_path: Path) -> str:
    with _connect(db_path, write=False) as con:
        rows = con.execute(
            """
            SELECT MAX(value) FROM (
              SELECT MAX(created_at) AS value FROM timeline_events
              UNION ALL SELECT MAX(created_at) FROM bot_context_chunks
              UNION ALL SELECT MAX(created_at) FROM derived_signals
            )
            """
        ).fetchone()
    return str(rows[0] or "1970-01-01T00:00:00+00:00")


def _family_id(tenant_id: str, customer_id: str) -> str:
    return stable_prefixed_id("family", {"tenant_id": tenant_id, "customer_id": customer_id}, length=32)


def _child_key(customer_id: str, name_key: str) -> str:
    digest = sha256(f"{customer_id}:{name_key}".encode("utf-8")).hexdigest()[:16]
    return f"child:{digest}"


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counter = Counter(str(value or "") for value in values)
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def _table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _json_loads(value: Any) -> Any:
    try:
        return json.loads(str(value or ""))
    except json.JSONDecodeError:
        return {}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build deterministic customer family graph v1 on staging timeline DB.")
    parser.add_argument("--timeline-db", required=True, type=Path)
    parser.add_argument("--allowed-root", required=True, type=Path)
    parser.add_argument("--profiles-db", type=Path)
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--customer-id", action="append", default=[])
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    summary = build_family_graph(
        FamilyGraphConfig(
            timeline_db=args.timeline_db,
            allowed_root=args.allowed_root,
            out_path=args.out,
            profiles_db=args.profiles_db,
            tenant_id=args.tenant_id,
            apply=bool(args.apply),
            customer_ids=tuple(args.customer_id or ()),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
