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
from mango_mvp.customer_profile.contracts import has_explicit_brand_conflict
from mango_mvp.customer_timeline.ids import normalize_email, normalize_key, stable_digest, stable_prefixed_id
from mango_mvp.customer_timeline.store import (
    authoritative_tallanto_student_owners,
    customer_timeline_readonly_uri,
    guard_customer_timeline_sqlite_path,
)
from mango_mvp.utils.phone import normalize_phone


FAMILY_GRAPH_SCHEMA_VERSION = "family_graph_v1"
FAMILY_GRAPH_RUN_KIND = "family_graph_v1"
FAMILY_GRAPH_ACTOR = "family_graph_v1_builder"
VALID_BRANDS = {"foton", "unpk", "unknown"}
TALLANTO_CHILD_EVENT_TYPES = {"tallanto_student_snapshot", "tallanto_payment", "tallanto_abonement", "tallanto_attendance"}
CHILD_RELEVANT_RE = re.compile(
    r"\b(?:реб[её]нок|дет[еи]|сын|дочь|дочка|ученик|ученица|класс|егэ|огэ|"
    r"математик|физик|информатик|русск|английск|курс|заняти|школ)\w*\b",
    re.I,
)
OTHER_CHILD_REFERENCE_RE = re.compile(
    r"\b(?:младш\w*|старш\w*|втор\w*|друг\w*|ещ[её]\s+од(?:ин|на))\s+"
    r"(?:реб[её]н\w*|сын\w*|доч\w*|брат\w*|сестр\w*)\b",
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
    tallanto_student_id: str = ""
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
        return sorted(self.names, key=_canonical_child_name_sort_key)[0]

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
    family_id: str


@dataclass(frozen=True)
class FamilyAssignment:
    family_id: str
    status: str
    confidence: str
    reason: str
    created_at: str


def build_family_graph(config: FamilyGraphConfig) -> Mapping[str, Any]:
    tenant_id = normalize_key(config.tenant_id, "tenant_id")
    db_path = _guard_db(config.timeline_db, apply=config.apply)
    generated_at = _stable_generated_at(db_path)
    with _connect(db_path, write=config.apply) as con:
        if config.apply:
            _ensure_schema(con)
        existing_family_links = (
            int(con.execute("SELECT COUNT(*) FROM family_links_v1 WHERE tenant_id = ?", (tenant_id,)).fetchone()[0])
            if _table_exists(con, "family_links_v1")
            else 0
        )
        profiles_db_available = bool(config.profiles_db and Path(config.profiles_db).expanduser().is_file())
        preserve_child_graph = not profiles_db_available and existing_family_links > 0
        # ponytail: the persisted family graph is one global derived view; partial writes
        # can split a family or erase rows outside the selection, so apply always rebuilds all.
        selected_customer_ids = () if config.apply else config.customer_ids
        customers = _load_customers(con, tenant_id=tenant_id, customer_ids=selected_customer_ids)
        exact_tallanto_owners = authoritative_tallanto_student_owners(con, tenant_id)
        assignments = _resolve_family_assignments(
            con,
            tenant_id=tenant_id,
            customer_ids=tuple(customers),
            generated_at=generated_at,
            exact_tallanto_owners=exact_tallanto_owners,
        )
        conflict_reconciliation = _reconcile_contact_conflicts(con, tenant_id, assignments, generated_at, config.apply)
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
                family_id=assignments[customer_id].family_id,
            )
            for customer_id, row in customers.items()
        }
        evidence_by_customer: dict[str, list[ChildEvidence]] = defaultdict(list)
        stable_child_keys = _load_stable_tallanto_child_keys(con, tenant_id=tenant_id)
        has_new_child_evidence = False
        persisted_groups = _load_persisted_family_groups(con, tenant_id=tenant_id) if preserve_child_graph else {}
        tallanto_snapshot_customers: set[str] = set()
        for row in con.execute(
            "SELECT customer_id,source_id,source_ref,event_at,record_json,match_status FROM timeline_events "
            "WHERE tenant_id=? AND source_system='tallanto_snapshot' "
            "AND event_type='tallanto_student_snapshot' "
            "AND superseded_by IS NULL AND customer_id IS NOT NULL",
            (tenant_id,),
        ):
            student_id = str(row["source_id"] or "")
            customer_id = str(row["customer_id"])
            if student_id in exact_tallanto_owners:
                if exact_tallanto_owners[student_id] != customer_id:
                    continue
            elif str(row["match_status"] or "") not in {"strong_unique", "manual"}:
                continue
            event_record = _json_loads(row["record_json"]).get("record") or {}
            payload = event_record.get("payload") or {}
            name = str(payload.get("display_name") or "").strip()
            if not name or customer_id not in contexts:
                continue
            tallanto_snapshot_customers.add(customer_id)
            evidence_by_customer[customer_id].append(
                ChildEvidence(
                    source_system="tallanto_snapshot",
                    source_ref=str(row["source_ref"] or ""),
                    event_at=str(row["event_at"] or ""),
                    tallanto_student_id=student_id,
                    name=name,
                    grade=str(payload.get("student_type") or ""),
                    subject=str(payload.get("subjects") or ""),
                    brand=str(event_record.get("brand") or "unknown"),
                )
            )
            has_new_child_evidence = True
        current_tallanto_student_ids = {item.tallanto_student_id for items in evidence_by_customer.values() for item in items if item.tallanto_student_id}
        for customer_id, groups in persisted_groups.items():
            if customer_id in tallanto_snapshot_customers:
                continue
            for group in groups:
                group_student_ids = set(group.get("tallanto_student_ids", ()))
                if group_student_ids & current_tallanto_student_ids:
                    continue
                evidence_by_customer[customer_id].append(
                    ChildEvidence(
                        source_system="persisted_family_graph",
                        source_ref=f"family_graph:{group['child_key']}",
                        event_at=generated_at,
                        tallanto_student_id=(next(iter(group_student_ids)) if len(group_student_ids) == 1 else ""),
                        name=str(group["canonical_name"]),
                        grade="; ".join(group["grades"]),
                        subject="; ".join(group["subjects"]),
                        brand=str(group["brand"]),
                    )
                )
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
                has_new_child_evidence = has_new_child_evidence or bool(items)
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
                        family_id=ctx.family_id,
                    )
        mail_report, mail_evidence = _load_mail_fact_evidence(con, tenant_id=tenant_id, customer_ids=set(customers))
        for customer_id, items in mail_evidence.items():
            evidence_by_customer[customer_id].extend(items)
            has_new_child_evidence = has_new_child_evidence or bool(items)
        for customer_id, brand in _load_customer_amo_brand_overrides(
            con,
            tenant_id=tenant_id,
            customer_ids=set(customers),
        ).items():
            for item in evidence_by_customer.get(customer_id, ()):
                if brand == "unknown" or _normalize_brand(item.brand) == "unknown":
                    item.brand = brand

        if preserve_child_graph and has_new_child_evidence:
            preserve_child_graph = False

        member_rows = _build_family_member_rows(tenant_id, assignments, generated_at=generated_at)
        if preserve_child_graph:
            family_rows = []
            groups_by_customer = _load_persisted_family_groups(con, tenant_id=tenant_id)
        else:
            family_rows, groups_by_customer = _build_family_rows(
                contexts,
                evidence_by_customer,
                stable_child_keys=stable_child_keys,
                generated_at=generated_at,
            )
        groups_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for owner_id, groups in groups_by_customer.items():
            groups_by_family[assignments[owner_id].family_id].extend(groups)
        scoped_groups = {
            customer_id: groups_by_family[assignment.family_id]
            for customer_id, assignment in assignments.items()
        }
        event_rows = _build_event_attributions(con, tenant_id=tenant_id, groups_by_customer=scoped_groups, contexts=contexts, generated_at=generated_at)
        opportunity_rows = _build_opportunity_attributions(
            con,
            tenant_id=tenant_id,
            groups_by_customer=scoped_groups,
            contexts=contexts,
            generated_at=generated_at,
        )
        summary = _summary(
            tenant_id=tenant_id,
            db_path=db_path,
            generated_at=generated_at,
            profile_report=profile_report,
            mail_report=mail_report,
            member_rows=member_rows,
            family_rows=family_rows,
            event_rows=event_rows,
            opportunity_rows=opportunity_rows,
            existing_family_links=existing_family_links,
            preserve_child_graph=preserve_child_graph,
            apply=config.apply,
        )
        summary = {**summary, "contact_conflict_reconciliation": conflict_reconciliation}
        if config.apply:
            _replace_rows(
                con,
                "family_members_v1",
                tenant_id,
                [{key: value for key, value in row.items() if key != "schema_version"} for row in member_rows],
            )
            if not preserve_child_graph:
                _replace_rows(con, "family_links_v1", tenant_id, family_rows)
            _replace_rows(con, "event_child_attribution_v1", tenant_id, event_rows)
            _replace_rows(con, "opportunity_child_attribution_v1", tenant_id, opportunity_rows)
            _record_run(con, tenant_id=tenant_id, generated_at=generated_at, summary=summary)
            con.commit()
            # The nightly service runs one full quick_check immediately before
            # publish. Repeating it here scans the 11 GB DB and both FTS indexes
            # after every derived rebuild without adding another safety boundary.
            summary = {**summary, "write_applied": True, "quick_check": "deferred_to_nightly_service"}
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
        con = sqlite3.connect(customer_timeline_readonly_uri(path), uri=True, timeout=15)
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
        filter_sql = f" AND customer.customer_id IN ({placeholders})"
        params.extend(customer_ids)
    rows = con.execute(
        f"""
        SELECT customer_id, tenant_id, identity_status, display_name, primary_phone, primary_email
        FROM customer_identities AS customer
        WHERE tenant_id = ?{filter_sql}
          AND (
            EXISTS (
              SELECT 1 FROM timeline_events AS event
              WHERE event.tenant_id=customer.tenant_id AND event.customer_id=customer.customer_id
                AND event.superseded_by IS NULL
            )
            OR EXISTS (
              SELECT 1 FROM identity_links AS link
              WHERE link.tenant_id=customer.tenant_id AND link.customer_id=customer.customer_id
            )
            OR EXISTS (
              SELECT 1 FROM customer_opportunities AS opportunity
              WHERE opportunity.tenant_id=customer.tenant_id AND opportunity.customer_id=customer.customer_id
            )
          )
        ORDER BY customer.customer_id
        """,
        tuple(params),
    ).fetchall()
    return {str(row["customer_id"]): row for row in rows}


def _resolve_family_assignments(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_ids: Sequence[str],
    generated_at: str,
    exact_tallanto_owners: Mapping[str, Optional[str]],
) -> dict[str, FamilyAssignment]:
    customers = set(customer_ids)
    existing_rows = (
        con.execute(
            "SELECT customer_id, family_id, created_at FROM family_members_v1 WHERE tenant_id = ?",
            (tenant_id,),
        ).fetchall()
        if _table_exists(con, "family_members_v1")
        else ()
    )
    existing = {str(row["customer_id"]): str(row["family_id"]) for row in existing_rows}
    created_at = {str(row["customer_id"]): str(row["created_at"] or generated_at) for row in existing_rows}
    tallanto_rows = con.execute(
        """
        SELECT customer_id, source_id AS link_value, match_status AS match_class
        FROM timeline_events
        WHERE tenant_id = ? AND source_system = 'tallanto_snapshot'
          AND event_type = 'tallanto_student_snapshot'
          AND customer_id IS NOT NULL AND customer_id != '' AND superseded_by IS NULL
          AND (
            EXISTS (SELECT 1 FROM identity_links AS exact_link
              WHERE exact_link.tenant_id=timeline_events.tenant_id AND exact_link.customer_id=timeline_events.customer_id
                AND exact_link.link_type='tallanto_student_id' AND exact_link.link_value=timeline_events.source_id
                AND exact_link.match_class IN ('strong_unique','manual'))
            OR NOT EXISTS (SELECT 1 FROM identity_links AS exact_link
              WHERE exact_link.tenant_id=timeline_events.tenant_id AND exact_link.customer_id=timeline_events.customer_id
                AND exact_link.link_type='tallanto_student_id' AND exact_link.match_class IN ('strong_unique','manual'))
          )
        UNION ALL
        SELECT link.customer_id, link.link_value, link.match_class
        FROM identity_links AS link
        WHERE link.tenant_id = ? AND link.source_system = 'tallanto_snapshot'
          AND link.link_type = 'tallanto_student_id'
          AND link.customer_id IS NOT NULL AND link.customer_id != ''
          AND NOT EXISTS (
            SELECT 1 FROM timeline_events AS event
            WHERE event.tenant_id = link.tenant_id
              AND event.source_system = 'tallanto_snapshot'
              AND event.event_type = 'tallanto_student_snapshot'
              AND event.source_id = link.link_value
              AND event.customer_id = link.customer_id
              AND event.superseded_by IS NULL
          )
        """,
        (tenant_id, tenant_id),
    ).fetchall()
    ids_by_customer: dict[str, set[str]] = defaultdict(set)
    customers_by_id: dict[str, set[str]] = defaultdict(set)
    unsafe: set[str] = set()
    for row in tallanto_rows:
        customer_id = str(row["customer_id"])
        student_id = str(row["link_value"] or "")
        if not student_id:
            continue
        customers_by_id[student_id].add(customer_id)
        if customer_id not in customers:
            continue
        ids_by_customer[customer_id].add(student_id)
        if student_id in exact_tallanto_owners:
            if exact_tallanto_owners[student_id] != customer_id:
                unsafe.add(customer_id)
        elif str(row["match_class"] or "") not in {"strong_unique", "manual"}:
            unsafe.add(customer_id)
    for owners in customers_by_id.values():
        if len(owners) != 1:
            unsafe.update(customers & owners)
    for row in con.execute(
        "SELECT record_json FROM timeline_conflicts "
        "WHERE tenant_id = ? AND conflict_type = 'tallanto_identity_conflict' AND status = 'open'",
        (tenant_id,),
    ):
        refs = _json_loads(row["record_json"]).get("entity_refs", [])
        for ref in refs if isinstance(refs, list) else ():
            if str(ref).startswith("tallanto_student_id:"):
                unsafe.update(customers & customers_by_id[str(ref).split(":", 1)[1]])
    amo_contact_owners: dict[str, set[str]] = defaultdict(set)
    for row in con.execute(
        """
        SELECT customer_id, link_value
        FROM identity_links
        WHERE tenant_id=? AND source_system='amocrm_snapshot'
          AND link_type='amo_contact_id' AND match_class IN ('strong_unique','manual')
          AND customer_id IS NOT NULL AND customer_id != '' AND link_value != ''
        """,
        (tenant_id,),
    ):
        amo_contact_owners[str(row["link_value"])].add(str(row["customer_id"]))
    shared_amo_contact_customers = {
        customer_id
        for owners in amo_contact_owners.values()
        if len(owners) > 1
        for customer_id in owners & customers
    }
    unsafe.update(shared_amo_contact_customers)
    eligible = set(ids_by_customer) - unsafe
    parents = {customer_id: customer_id for customer_id in eligible}
    parent_keys_by_customer: dict[str, set[str]] = defaultdict(set)
    first_names_by_customer: dict[str, set[str]] = defaultdict(set)
    last_names_by_customer: dict[str, set[str]] = defaultdict(set)
    student_types_by_customer: dict[str, set[str]] = defaultdict(set)
    child_keys_by_customer: dict[str, set[str]] = defaultdict(set)
    current_contacts_by_customer: dict[str, set[tuple[str, str]]] = defaultdict(set)
    exact_contacts_by_customer: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in con.execute(
        """
        WITH ranked AS (
          SELECT customer_id, source_id,
                 json_extract(record_json, '$.record.payload.parent_fio') AS parent_fio,
                 json_extract(record_json, '$.record.payload.first_name') AS first_name,
                 json_extract(record_json, '$.record.payload.last_name') AS last_name,
                 json_extract(record_json, '$.record.payload.student_type') AS student_type,
                 json_extract(record_json, '$.record.payload.primary_phone') AS primary_phone,
                 json_extract(record_json, '$.record.payload.phone_extra') AS phone_extra,
                 json_extract(record_json, '$.record.payload.primary_email') AS primary_email,
                 json_extract(record_json, '$.record.payload.email_extra') AS email_extra,
                 ROW_NUMBER() OVER (
                   PARTITION BY customer_id, source_id ORDER BY event_at DESC, event_id DESC
                 ) AS row_number
          FROM timeline_events
          WHERE tenant_id = ? AND source_system = 'tallanto_snapshot'
            AND event_type = 'tallanto_student_snapshot'
            AND customer_id IS NOT NULL AND customer_id != '' AND superseded_by IS NULL
        )
        SELECT customer_id, source_id, parent_fio, first_name, last_name, student_type,
               primary_phone, phone_extra, primary_email, email_extra
        FROM ranked WHERE row_number = 1
        """,
        (tenant_id,),
    ):
        customer_id = str(row["customer_id"])
        parent_key = _parent_identity_key(str(row["parent_fio"] or ""))
        if parent_key:
            parent_keys_by_customer[customer_id].add(parent_key)
        source_id = str(row["source_id"] or "")
        exact_card = exact_tallanto_owners.get(source_id) == customer_id
        if exact_card:
            first_name_tokens = normalized_name_tokens(str(row["first_name"] or ""))
            first_name = " ".join(first_name_tokens)
            last_name = " ".join(normalized_name_tokens(str(row["last_name"] or "")))
            student_type = str(row["student_type"] or "").strip().casefold()
            if first_name:
                first_names_by_customer[customer_id].add(first_name)
                given_name = first_name_tokens[0]
                child_keys_by_customer[customer_id].update(
                    child_name_keys({given_name}) or {given_name}
                )
            if last_name:
                last_names_by_customer[customer_id].add(last_name)
            if student_type:
                student_types_by_customer[customer_id].add(student_type)
        for value in (row["primary_phone"], *str(row["phone_extra"] or "").split("|")):
            if normalized := normalize_phone(value):
                current_contacts_by_customer[customer_id].add(("phone", normalized))
                if exact_card:
                    exact_contacts_by_customer[customer_id].add(("phone", normalized))
        for value in (row["primary_email"], *str(row["email_extra"] or "").split("|")):
            if normalized := normalize_email(value):
                current_contacts_by_customer[customer_id].add(("email", normalized))
                if exact_card:
                    exact_contacts_by_customer[customer_id].add(("email", normalized))

    def find(customer_id: str) -> str:
        while parents[customer_id] != customer_id:
            parents[customer_id] = parents[parents[customer_id]]
            customer_id = parents[customer_id]
        return customer_id

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    by_parent_identity: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for customer_id, contacts in current_contacts_by_customer.items():
        parent_keys = parent_keys_by_customer.get(customer_id, set())
        if customer_id in eligible and len(parent_keys) == 1:
            for link_type, link_value in contacts:
                by_parent_identity[(link_type, link_value, next(iter(parent_keys)))].add(customer_id)
    for members in by_parent_identity.values():
        # ponytail: larger groups are likely shared/system contacts; review them manually.
        if 2 <= len(members) <= 8:
            anchor = min(members)
            for member in members:
                union(anchor, member)

    by_shared_phone_email: dict[tuple[str, str], set[str]] = defaultdict(set)
    for customer_id, contacts in exact_contacts_by_customer.items():
        phones = {value for kind, value in contacts if kind == "phone"}
        emails = {value for kind, value in contacts if kind == "email"}
        for phone in phones:
            for email in emails:
                by_shared_phone_email[(phone, email)].add(customer_id)
    root_members: dict[str, set[str]] = defaultdict(set)
    for customer_id in eligible:
        root_members[find(customer_id)].add(customer_id)
    attachment_cores: dict[str, set[str]] = defaultdict(set)
    for members in by_shared_phone_email.values():
        if not (2 <= len(members) <= 8 and members <= eligible):
            continue
        student_ids = set().union(*(ids_by_customer[member] for member in members))
        if len(student_ids) != len(members) or any(len(ids_by_customer[member]) != 1 for member in members):
            continue
        roots = {find(member) for member in members}
        cores = {root for root in roots if len(root_members[root]) >= 2}
        singletons = {member for member in members if len(root_members[find(member)]) == 1}
        if len(cores) != 1 or len(singletons) != 1:
            continue
        core = next(iter(cores))
        if root_members[core] | singletons != members:
            continue
        first_names = [first_names_by_customer[member] for member in members]
        last_names = [last_names_by_customer[member] for member in members]
        student_types = [student_types_by_customer[member] for member in members]
        if not all(len(values) == 1 for values in (*first_names, *last_names, *student_types)):
            continue
        if len(set().union(*last_names)) != 1 or len(set().union(*first_names)) != len(members):
            continue
        if len(set().union(*student_types)) != len(members):
            continue
        child_key_sets = [child_keys_by_customer[member] for member in sorted(members)]
        if any(
            left & right
            for index, left in enumerate(child_key_sets)
            for right in child_key_sets[index + 1 :]
        ):
            continue
        attachment_cores[next(iter(singletons))].add(core)
    singletons_by_core: dict[str, set[str]] = defaultdict(set)
    for singleton, cores in attachment_cores.items():
        if len(cores) == 1:
            singletons_by_core[next(iter(cores))].add(singleton)
    strict_contact_family_members = {
        singleton
        for core, singletons in singletons_by_core.items()
        if len(root_members[core]) + len(singletons) <= 8
        for singleton in singletons
    }
    for singleton in sorted(strict_contact_family_members):
        union(singleton, next(iter(attachment_cores[singleton])))

    identity_roots: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for identity_key, members in by_parent_identity.items():
        identity_roots[identity_key].update(find(member) for member in members)
    tallanto_customers = {
        str(row["customer_id"])
        for row in con.execute(
            """
            SELECT DISTINCT customer_id FROM identity_links
            WHERE tenant_id=? AND link_type='tallanto_student_id'
              AND customer_id IS NOT NULL AND customer_id != ''
            """,
            (tenant_id,),
        )
    }
    amo_contact_ids_by_customer: dict[str, set[str]] = defaultdict(set)
    for row in con.execute(
        """
        SELECT customer_id, link_value
        FROM identity_links
        WHERE tenant_id=? AND source_system='amocrm_snapshot'
          AND link_type='amo_contact_id' AND match_class IN ('strong_unique','manual')
          AND customer_id IS NOT NULL AND customer_id != ''
        """,
        (tenant_id,),
    ):
        customer_id = str(row["customer_id"])
        if (
            customer_id in customers
            and customer_id not in tallanto_customers
            and customer_id not in unsafe
        ):
            amo_contact_ids_by_customer[customer_id].add(str(row["link_value"]))
    amo_contact_customers = {
        customer_id
        for customer_id, values in amo_contact_ids_by_customer.items()
        if len(values) == 1
    }
    tallanto_children_by_amo_contact: dict[str, set[str]] = defaultdict(set)
    for row in con.execute(
        """
        SELECT customer_id, link_value
        FROM identity_links
        WHERE tenant_id=? AND source_system='tallanto_snapshot'
          AND link_type='amo_contact_id' AND match_class='ambiguous'
          AND json_extract(record_json, '$.evidence.relationship')='family_amo_contact'
          AND customer_id IS NOT NULL AND customer_id != ''
        """,
        (tenant_id,),
    ):
        child_id = str(row["customer_id"])
        if child_id in eligible:
            tallanto_children_by_amo_contact[str(row["link_value"])].add(child_id)
    amo_identity_values: dict[str, set[tuple[str, str]]] = defaultdict(set)
    direct_amo_family_members: set[str] = set()
    if amo_contact_customers:
        placeholders = ",".join("?" for _ in amo_contact_customers)
        for row in con.execute(
            f"""
            SELECT customer_id, link_type, link_value
            FROM identity_links
            WHERE tenant_id=? AND customer_id IN ({placeholders})
              AND source_system='amocrm_snapshot' AND link_type IN ('phone','email')
              AND match_class IN ('strong_unique','manual','ambiguous') AND link_value != ''
            """,
            (tenant_id, *sorted(amo_contact_customers)),
        ):
            amo_identity_values[str(row["customer_id"])].add(
                (str(row["link_type"]), str(row["link_value"]))
            )
        for row in con.execute(
            f"""
            SELECT customer_id, display_name
            FROM customer_identities
            WHERE tenant_id=? AND customer_id IN ({placeholders})
            """,
            (tenant_id, *sorted(amo_contact_customers)),
        ):
            customer_id = str(row["customer_id"])
            direct_children = {
                child_id
                for amo_contact_id in amo_contact_ids_by_customer.get(customer_id, set())
                for child_id in tallanto_children_by_amo_contact.get(amo_contact_id, set())
            }
            if direct_children:
                parents[customer_id] = customer_id
                eligible.add(customer_id)
                direct_amo_family_members.add(customer_id)
                for child_id in sorted(direct_children):
                    direct_amo_family_members.add(child_id)
                    union(customer_id, child_id)
                continue
            parent_key = _parent_identity_key(str(row["display_name"] or ""))
            roots = {
                root
                for link_type, link_value in amo_identity_values.get(customer_id, set())
                for root in identity_roots.get((link_type, link_value, parent_key), set())
            }
            if parent_key and len(roots) == 1:
                parents[customer_id] = customer_id
                eligible.add(customer_id)
                union(customer_id, next(iter(roots)))
    attached_amo_parents = amo_contact_customers & eligible

    components: dict[str, set[str]] = defaultdict(set)
    for customer_id in eligible:
        components[find(customer_id)].add(customer_id)
    assignments: dict[str, FamilyAssignment] = {}
    for members in components.values():
        roots = {existing[member] for member in members if member in existing}
        non_amo_roots = {
            existing[member]
            for member in members
            if member in existing and member not in attached_amo_parents
        }
        safe_amo_attach = bool(attached_amo_parents & members) and len(non_amo_roots) == 1
        exact_tallanto_merge = members <= set(ids_by_customer)
        exact_amo_family_merge = members <= direct_amo_family_members
        if len(roots) > 1 and not (safe_amo_attach or exact_tallanto_merge or exact_amo_family_merge):
            for member in members:
                assignments[member] = FamilyAssignment(
                    existing.get(member, _family_id(tenant_id, member)),
                    "conflict",
                    "low",
                    "conflicting_persisted_family_roots",
                    created_at.get(member, generated_at),
                )
            continue
        multi = len(members) > 1
        root = (
            min(non_amo_roots or roots)
            if multi and (non_amo_roots or roots)
            else _family_id(tenant_id, min(members))
        )
        for member in members:
            assignments[member] = FamilyAssignment(
                root,
                "confident" if multi else "singleton",
                "high" if multi else "medium",
                (
                    "exact_tallanto_amo_family_reference"
                    if member in direct_amo_family_members
                    else "exact_amo_parent_name_and_phone_or_email"
                    if member in attached_amo_parents
                    else "exact_tallanto_shared_contacts_family_core"
                    if member in strict_contact_family_members
                    else "exact_tallanto_parent_name_and_phone_or_email"
                )
                if multi
                else "single_customer_family",
                created_at.get(member, generated_at) if existing.get(member) == root else generated_at,
            )

    for customer_id in customers - set(assignments):
        conflict = customer_id in unsafe
        amo_conflict = customer_id in shared_amo_contact_customers
        root = (
            existing.get(customer_id, _family_id(tenant_id, customer_id))
            if amo_conflict
            else _family_id(tenant_id, customer_id)
        )
        assignments[customer_id] = FamilyAssignment(
            root,
            "conflict" if conflict else "singleton",
            "low" if conflict else "medium",
            (
                "shared_amo_contact_across_customers"
                if amo_conflict
                else "tallanto_student_id_conflict"
            )
            if conflict
            else "single_customer_family",
            created_at.get(customer_id, generated_at) if existing.get(customer_id) == root else generated_at,
        )
    return assignments


def _reconcile_contact_conflicts(
    con: sqlite3.Connection, tenant_id: str, assignments: Mapping[str, FamilyAssignment], generated_at: str, apply: bool
) -> Mapping[str, Any]:
    """Recheck stale contact conflicts against current exact Tallanto cards."""
    contact_owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    phone_hash_owners: dict[str, set[str]] = defaultdict(set)
    cards: dict[str, set[tuple[str, frozenset[str]]]] = defaultdict(set)
    for row in con.execute(
        "SELECT event.customer_id,event.source_id,event.record_json FROM timeline_events event JOIN identity_links exact "
        "ON exact.tenant_id=event.tenant_id AND exact.customer_id=event.customer_id AND exact.link_type='tallanto_student_id' "
        "AND exact.link_value=event.source_id AND exact.match_class IN ('strong_unique','manual') WHERE event.tenant_id=? "
        "AND event.source_system='tallanto_snapshot' AND event.event_type='tallanto_student_snapshot' "
        "AND event.superseded_by IS NULL AND NOT EXISTS (SELECT 1 FROM identity_links other WHERE other.tenant_id=exact.tenant_id "
        "AND other.link_type=exact.link_type AND other.link_value=exact.link_value AND other.customer_id!=exact.customer_id "
        "AND other.match_class IN ('strong_unique','manual'))",
        (tenant_id,)
    ):
        customer_id, student_id = str(row["customer_id"]), str(row["source_id"] or "")
        payload = (_json_loads(row["record_json"]).get("record") or {}).get("payload") or {}
        first_name_tokens = normalized_name_tokens(
            str(payload.get("first_name") or payload.get("display_name") or "")
        )
        given_name = first_name_tokens[0] if first_name_tokens else ""
        name_keys = frozenset(child_name_keys({given_name}) or ({given_name} if given_name else set()))
        if name_keys:
            cards[customer_id].add((student_id, name_keys))
        for value in (payload.get("primary_phone"), *str(payload.get("phone_extra") or "").split("|")):
            if phone := normalize_phone(value):
                contact_owners[("phone", phone)].add(customer_id)
                phone_hash_owners[stable_digest({"phone": phone})[:16]].add(customer_id)
        for value in (payload.get("primary_email"), *str(payload.get("email_extra") or "").split("|")):
            if email := normalize_email(value):
                contact_owners[("email", email)].add(customer_id)

    conflicted_students: set[str] = set()
    for row in con.execute(
        "SELECT record_json FROM timeline_conflicts WHERE tenant_id=? AND conflict_type='tallanto_identity_conflict' AND status='open'",
        (tenant_id,),
    ):
        for ref in _json_loads(row["record_json"]).get("entity_refs", ()):
            if str(ref).startswith("tallanto_student_id:"):
                conflicted_students.add(str(ref).split(":", 1)[1])

    def confirmed_family(customer_ids: set[str]) -> str:
        if len(customer_ids) < 2:
            return ""
        member_cards = [cards.get(customer_id, set()) for customer_id in customer_ids]
        if any(len(items) != 1 for items in member_cards):
            return ""
        student_ids = {next(iter(items))[0] for items in member_cards}
        name_sets = [next(iter(items))[1] for items in member_cards]
        names_overlap = any(
            left & right
            for index, left in enumerate(name_sets)
            for right in name_sets[index + 1 :]
        )
        if len(student_ids) != len(customer_ids) or names_overlap or student_ids & conflicted_students:
            return ""
        family_ids = {
            assignment.family_id for customer_id in customer_ids
            if (assignment := assignments.get(customer_id)) and assignment.status == "confident" and assignment.confidence == "high"
        }
        return next(iter(family_ids)) if len(family_ids) == 1 else ""

    def all_in_family(customer_ids: set[str], family_id: str) -> bool:
        return bool(customer_ids) and all(
            (assignment := assignments.get(customer_id)) and assignment.family_id == family_id
            and assignment.status == "confident" and assignment.confidence == "high"
            for customer_id in customer_ids
        )

    checked = resolved = reopened = 0
    reasons: Counter[str] = Counter(); types: Counter[str] = Counter()
    rows = con.execute(
        "SELECT conflict_id,conflict_type,status,record_json FROM timeline_conflicts "
        "WHERE tenant_id=? AND conflict_type IN ('shared_family_phone','ambiguous_identity') AND (status='open' OR "
        "(status='resolved' AND json_extract(record_json,'$.metadata.resolved_by')=?))",
        (tenant_id, FAMILY_GRAPH_ACTOR),
    ).fetchall()
    for row in rows:
        is_open = str(row["status"]) == "open"
        checked += int(is_open)
        payload = _json_loads(row["record_json"])
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        if metadata.get("identity_repair") == "current_source_conflict":
            continue
        entity_refs = tuple(str(raw) for raw in payload.get("entity_refs", ()))
        refs = {raw for raw in entity_refs if raw.startswith("customer:")}
        customer_ids = {ref[len("customer:"):] if ref.startswith("customer:customer:") else ref for ref in refs}
        reason = ""
        if str(row["conflict_type"]) == "shared_family_phone":
            expected = {str(value) for value in metadata.get("tallanto_student_ids", ()) if value} or {
                ref.split(":", 1)[1]
                for ref in entity_refs
                if ref.startswith(("tallanto_student:", "tallanto_student_id:"))
            }
            current = {student_id for customer_id in customer_ids for student_id, _ in cards.get(customer_id, ())}
            family_id = confirmed_family(customer_ids)
            owners = phone_hash_owners.get(str(metadata.get("phone_hash") or ""), set())
            if family_id and current == expected and len(expected) >= 2 and all_in_family(owners, family_id):
                reason = "confirmed_single_family"
        else:
            identifiers = metadata.get("identifiers")
            if not isinstance(identifiers, list):
                identifiers = [ref for ref in entity_refs if ref.startswith(("phone:", "email:"))]
            parsed: list[tuple[str, str]] = []
            if isinstance(identifiers, list):
                for raw in identifiers:
                    kind, separator, value = str(raw).partition(":")
                    normalized = normalize_phone(value) if kind == "phone" else normalize_email(value) if kind == "email" else ""
                    if separator and normalized:
                        parsed.append((kind, normalized))
            valid_identifiers = isinstance(identifiers, list) and len(parsed) == len(identifiers) > 0
            owner_sets = [contact_owners.get(key, set()) for key in parsed]
            unique_owners = set().union(*owner_sets) if owner_sets else set()
            if valid_identifiers and all(len(owners) == 1 for owners in owner_sets) and len(unique_owners) == 1 and unique_owners <= customer_ids:
                reason = "contact_no_longer_shared"
            elif valid_identifiers and all(owner_sets):
                all_customers = customer_ids | set().union(*owner_sets)
                family_id = confirmed_family(customer_ids)
                if family_id and all_in_family(all_customers, family_id):
                    reason = "confirmed_single_family"
        if not reason:
            if not is_open:
                reopened += 1
                if apply:
                    payload.update(status="open", resolved_at=None)
                    payload["metadata"] = {key: value for key, value in metadata.items() if key not in {"resolved_by", "resolution_reason"}}
                    con.execute(
                        "UPDATE timeline_conflicts SET status='open',resolved_at=NULL,record_hash=?,record_json=? WHERE conflict_id=?",
                        (stable_digest(payload), _json_dumps(payload), str(row["conflict_id"])),
                    )
            continue
        if not is_open:
            continue
        resolved += 1
        reasons[reason] += 1; types[str(row["conflict_type"])] += 1
        if apply:
            payload.update(status="resolved", resolved_at=generated_at)
            payload["metadata"] = {**metadata, "resolved_by": FAMILY_GRAPH_ACTOR, "resolution_reason": reason}
            con.execute(
                "UPDATE timeline_conflicts SET status='resolved',resolved_at=?,record_hash=?,record_json=? WHERE conflict_id=?",
                (generated_at, stable_digest(payload), _json_dumps(payload), str(row["conflict_id"])),
            )
    return {"checked": checked, "resolved": resolved, "reopened": reopened, "remaining_open": checked - resolved + reopened, "resolved_reason_counts": dict(sorted(reasons.items())), "resolved_type_counts": dict(sorted(types.items()))}
def _parent_identity_key(value: str) -> str:
    tokens = sorted(normalized_name_tokens(value))
    return "|".join(tokens) if len(tokens) >= 2 else ""


def _build_family_member_rows(
    tenant_id: str,
    assignments: Mapping[str, FamilyAssignment],
    *,
    generated_at: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for customer_id, assignment in sorted(assignments.items()):
        payload = {
            "schema_version": FAMILY_GRAPH_SCHEMA_VERSION,
            "tenant_id": tenant_id,
            "family_id": assignment.family_id,
            "customer_id": customer_id,
            "membership_status": assignment.status,
            "confidence": assignment.confidence,
            "reason": assignment.reason,
            "created_at": assignment.created_at,
            "updated_at": generated_at,
        }
        rows.append({**payload, "record_hash": stable_digest(payload), "record_json": _json_dumps(payload)})
    return rows


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
    stable_child_keys: Mapping[str, str],
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
            tallanto_student_ids = sorted(
                {
                    item.tallanto_student_id
                    for item in group.evidence
                    if item.source_system == "tallanto_snapshot" and item.tallanto_student_id
                }
            )
            child_key = (
                stable_child_keys.get(tallanto_student_ids[0], _tallanto_child_key(context.tenant_id, tallanto_student_ids[0]))
                if len(tallanto_student_ids) == 1
                else _child_key(context.family_id, group.name_key)
            )
            status, confidence, reason = _family_confidence(group, valid_groups=valid_groups, identity_risks=identity_risks)
            payload = {
                "schema_version": FAMILY_GRAPH_SCHEMA_VERSION,
                "tenant_id": context.tenant_id,
                "family_id": context.family_id,
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
                "tallanto_student_ids": tallanto_student_ids,
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


def _load_persisted_family_groups(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in con.execute(
        """
        SELECT family_id,customer_id,child_key,canonical_name,name_variants_json,
               grades_json,subjects_json,brand,status,confidence,reason,
               source_refs_json,record_json
        FROM family_links_v1 WHERE tenant_id=?
        ORDER BY customer_id,child_key
        """,
        (tenant_id,),
    ):
        groups[str(row["customer_id"])].append(
            {
                "family_id": str(row["family_id"]),
                "customer_id": str(row["customer_id"]),
                "child_key": str(row["child_key"]),
                "canonical_name": str(row["canonical_name"] or ""),
                "name_variants": _json_list(row["name_variants_json"]),
                "grades": _json_list(row["grades_json"]),
                "subjects": _json_list(row["subjects_json"]),
                "brand": str(row["brand"] or "unknown"),
                "status": str(row["status"] or "unknown"),
                "confidence": str(row["confidence"] or "low"),
                "reason": str(row["reason"] or "persisted_family_graph"),
                "source_refs": _json_list(row["source_refs_json"]),
                "tallanto_student_ids": _json_list(
                    _json_loads(row["record_json"]).get("tallanto_student_ids", [])
                ),
            }
        )
    return groups


def _load_stable_tallanto_child_keys(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
) -> dict[str, str]:
    if not (_table_exists(con, "family_links_v1") and _table_exists(con, "event_child_attribution_v1")):
        return {}
    candidates: dict[str, set[str]] = defaultdict(set)
    students_by_key: dict[str, set[str]] = defaultdict(set)
    for row in con.execute(
        """
        SELECT event.source_id AS student_id, attribution.child_key, child.record_json AS child_record_json
        FROM timeline_events AS event
        JOIN event_child_attribution_v1 AS attribution
          ON attribution.tenant_id=event.tenant_id AND attribution.event_id=event.event_id
        JOIN family_links_v1 AS child
          ON child.tenant_id=attribution.tenant_id
         AND child.customer_id=attribution.customer_id AND child.child_key=attribution.child_key
        WHERE event.tenant_id=? AND event.source_system='tallanto_snapshot'
          AND event.event_type='tallanto_student_snapshot'
          AND event.match_status IN ('strong_unique','manual')
          AND event.superseded_by IS NULL
          AND attribution.status='matched' AND attribution.child_key!=''
        """,
        (tenant_id,),
    ):
        student_id = str(row["student_id"])
        child_student_ids = set(_json_list(_json_loads(row["child_record_json"]).get("tallanto_student_ids", [])))
        if student_id not in child_student_ids:
            continue
        key = str(row["child_key"])
        candidates[student_id].add(key)
        students_by_key[key].add(student_id)
    return {
        student_id: next(iter(keys))
        for student_id, keys in candidates.items()
        if len(keys) == 1 and len(students_by_key[next(iter(keys))]) == 1
    }


def _child_groups_for_customer(context: CustomerContext, evidence_items: Sequence[ChildEvidence]) -> dict[str, ChildGroup]:
    groups: dict[str, ChildGroup] = {}
    for item in evidence_items:
        if not item.name.strip():
            continue
        name_key = _name_key(item.name)
        if not name_key:
            continue
        if item.source_system == "tallanto_snapshot" and item.tallanto_student_id:
            name_key = f"{name_key}|{item.tallanto_student_id}"
        group = groups.setdefault(name_key, ChildGroup(name_key=name_key))
        group.names.add(item.name.strip())
        if item.grade.strip():
            group.grades.add(item.grade.strip())
        for subject in _split_subjects(item.subject):
            group.subjects.add(subject)
        group.brands[_normalize_brand(item.brand)] += 1
        group.evidence.append(item)
        group.suspicious_reasons.update(_suspicious_name_reasons(item.name, parent_name_keys=context.parent_name_keys))
    result = _merge_safe_child_name_variants(groups)
    _mark_weak_competing_child_candidates(result.values(), identity_risk=bool(_identity_risks(context)))
    return result


def _load_customer_amo_brand_overrides(
    con: sqlite3.Connection,
    *,
    tenant_id: str,
    customer_ids: set[str],
) -> Mapping[str, str]:
    brands: dict[str, set[str]] = defaultdict(set)
    conflicts: set[str] = set()
    for chunk in _chunks(sorted(customer_ids), 900):
        placeholders = ",".join("?" for _ in chunk)
        for row in con.execute(
            f"SELECT customer_id,record_json FROM customer_opportunities "
            f"WHERE tenant_id=? AND customer_id IN ({placeholders})",
            (tenant_id, *chunk),
        ):
            record = _json_loads(row["record_json"])
            customer_id = str(row["customer_id"])
            if has_explicit_brand_conflict(record):
                conflicts.add(customer_id)
                continue
            product_context = record.get("product_context") if isinstance(record, Mapping) else None
            brand = _normalize_brand(product_context.get("brand") if isinstance(product_context, Mapping) else "")
            if brand in {"foton", "unpk"}:
                brands[customer_id].add(brand)
        for row in con.execute(
            f"SELECT customer_id,record_json FROM timeline_events "
            f"WHERE tenant_id=? AND source_system='amocrm_snapshot' "
            f"AND customer_id IN ({placeholders}) AND superseded_by IS NULL",
            (tenant_id, *chunk),
        ):
            if has_explicit_brand_conflict(_json_loads(row["record_json"])):
                conflicts.add(str(row["customer_id"]))
    return {
        customer_id: "unknown" if customer_id in conflicts else next(iter(brands[customer_id]))
        for customer_id in brands.keys() | conflicts
        if customer_id in conflicts or len(brands[customer_id]) == 1
    }


def _merge_safe_child_name_variants(groups: Mapping[str, ChildGroup]) -> dict[str, ChildGroup]:
    keys = sorted(groups)
    parents = {key: key for key in keys}
    tallanto_ids = {
        key: {item.tallanto_student_id for item in group.evidence if item.tallanto_student_id} for key, group in groups.items()
    }
    ambiguous_bridges = _ambiguous_patronymic_bridge_keys(groups)

    def find(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if tallanto_ids[root_left] and tallanto_ids[root_right] and tallanto_ids[root_left] != tallanto_ids[root_right]:
            return
        canonical = min((root_left, root_right))
        other = root_right if canonical == root_left else root_left
        parents[other] = canonical
        tallanto_ids[canonical].update(tallanto_ids[other])

    for left_index, left_key in enumerate(keys):
        for right_key in keys[left_index + 1 :]:
            if left_key in ambiguous_bridges or right_key in ambiguous_bridges:
                continue
            if _child_groups_have_matching_full_names(groups[left_key], groups[right_key]):
                union(left_key, right_key)

    # One-token aliases ("Дан", "Даня") are merged only if exactly one
    # already-known multi-token child group can explain the alias.
    for key in keys:
        if not _group_has_only_single_token_names(groups[key]):
            continue
        candidates = {
            find(other_key)
            for other_key in keys
            if find(other_key) != find(key)
            and not _group_has_only_single_token_names(groups[other_key])
            and _single_token_group_matches_multi_name(groups[key], groups[other_key])
        }
        if len(candidates) == 1:
            union(key, next(iter(candidates)))

    merged: dict[str, ChildGroup] = {}
    for key in keys:
        root = find(key)
        target = merged.setdefault(root, ChildGroup(name_key=root))
        _absorb_child_group(target, groups[key])

    result: dict[str, ChildGroup] = {}
    for group in merged.values():
        if group.name_key in ambiguous_bridges:
            group.suspicious_reasons.add("ambiguous_patronymic_bridge")
        group.name_key = _name_key(group.canonical_name) or group.name_key
        while group.name_key in result:
            group.name_key = f"{group.name_key}_dup"
        result[group.name_key] = group
    return result


def _ambiguous_patronymic_bridge_keys(groups: Mapping[str, ChildGroup]) -> set[str]:
    result: set[str] = set()
    for key, group in groups.items():
        bridge_options = _group_patronymic_bridge_options(group)
        if not bridge_options:
            continue
        leftover_candidates: list[frozenset[str]] = []
        for other_key, other in groups.items():
            if other_key == key:
                continue
            for name in other.names:
                options = _name_token_options(name)
                if len(options) < 3 or not _token_options_subset_match(bridge_options, options, min_matches=2):
                    continue
                leftovers = [
                    token_options
                    for token_options in options
                    if not any(_token_option_sets_match(token_options, bridge_token) for bridge_token in bridge_options)
                ]
                leftover_candidates.extend(leftovers[:1])
        if len(leftover_candidates) < 2:
            continue
        if any(
            not _token_option_sets_match(left, right)
            for index, left in enumerate(leftover_candidates)
            for right in leftover_candidates[index + 1 :]
        ):
            result.add(key)
    return result


def _group_patronymic_bridge_options(group: ChildGroup) -> list[frozenset[str]]:
    candidates: list[list[frozenset[str]]] = []
    for name in group.names:
        options = _name_token_options(name)
        if len(options) == 2 and _token_option_looks_like_patronymic(options[-1]):
            candidates.append(options)
    if len(candidates) != 1:
        return []
    return candidates[0]


def _token_option_looks_like_patronymic(options: frozenset[str]) -> bool:
    return any(token.endswith(("ич", "вич", "ович", "евич", "ична", "вна", "овна", "евна")) for token in options)


def _absorb_child_group(target: ChildGroup, source: ChildGroup) -> None:
    target.names.update(source.names)
    target.grades.update(source.grades)
    target.subjects.update(source.subjects)
    target.brands.update(source.brands)
    target.evidence.extend(source.evidence)
    target.suspicious_reasons.update(source.suspicious_reasons)


def _child_groups_have_matching_full_names(left: ChildGroup, right: ChildGroup) -> bool:
    if left.suspicious_reasons or right.suspicious_reasons:
        return False
    if _distinct_tallanto_children(left, right):
        return False
    return any(_full_child_names_match(left_name, right_name) for left_name in left.names for right_name in right.names)


def _distinct_tallanto_children(left: ChildGroup, right: ChildGroup) -> bool:
    left_ids = {item.tallanto_student_id for item in left.evidence if item.tallanto_student_id}
    right_ids = {item.tallanto_student_id for item in right.evidence if item.tallanto_student_id}
    return bool(left_ids and right_ids and left_ids.isdisjoint(right_ids))


def _full_child_names_match(left: str, right: str) -> bool:
    left_tokens = _name_token_options(left)
    right_tokens = _name_token_options(right)
    if len(left_tokens) < 2 or len(right_tokens) < 2:
        return False
    shorter, longer = (left_tokens, right_tokens) if len(left_tokens) <= len(right_tokens) else (right_tokens, left_tokens)
    return _token_options_subset_match(shorter, longer, min_matches=2)


def _token_options_subset_match(shorter: Sequence[frozenset[str]], longer: Sequence[frozenset[str]], *, min_matches: int) -> bool:
    used: set[int] = set()
    matches = 0
    for options in shorter:
        match_index = next(
            (
                index
                for index, candidate in enumerate(longer)
                if index not in used and _token_option_sets_match(options, candidate)
            ),
            None,
        )
        if match_index is None:
            return False
        used.add(match_index)
        matches += 1
    return matches >= min_matches


def _token_option_sets_match(left: frozenset[str], right: frozenset[str]) -> bool:
    if left & right:
        return True
    return any(_token_spelling_variant(left_token, right_token) for left_token in left for right_token in right)


def _token_spelling_variant(left: str, right: str) -> bool:
    if len(left) < 4 or len(right) < 4:
        return False
    distance = _levenshtein_distance(left, right)
    if left[:3] == right[:3]:
        return distance <= 2
    if left[:2] == right[:2] and (left.endswith(_RUSSIAN_SURNAME_ENDINGS) or right.endswith(_RUSSIAN_SURNAME_ENDINGS)):
        return distance <= 3
    if (
        left[0] == right[0]
        and left.endswith(_RUSSIAN_SURNAME_ENDINGS)
        and right.endswith(_RUSSIAN_SURNAME_ENDINGS)
    ):
        return distance <= 2
    return False


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (0 if left_char == right_char else 1),
                )
            )
        previous = current
    return previous[-1]


def _group_has_only_single_token_names(group: ChildGroup) -> bool:
    return bool(group.names) and all(len(normalized_name_tokens(name)) == 1 for name in group.names)


def _single_token_group_matches_multi_name(single: ChildGroup, multi: ChildGroup) -> bool:
    if single.suspicious_reasons or multi.suspicious_reasons:
        return False
    if _distinct_tallanto_children(single, multi):
        return False
    single_options = [_name_token_options(name)[0] for name in single.names if len(_name_token_options(name)) == 1]
    multi_options = [options for name in multi.names for options in _name_token_options(name)]
    return bool(single_options and multi_options) and any(
        _token_option_sets_match(left, right) for left in single_options for right in multi_options
    )


_LOCAL_CHILD_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "дан": ("даниил", "данил"),
    "данил": ("даниил", "данил"),
    "даниил": ("даниил", "данил"),
    "даня": ("даниил", "данил"),
    "дениил": ("даниил", "данил"),
    "елизавета": ("елизавета",),
    "лиза": ("елизавета",),
}
_RUSSIAN_SURNAME_ENDINGS = ("ов", "ова", "ев", "ева", "ин", "ина", "ын", "ына")


def _name_token_options(value: str) -> list[frozenset[str]]:
    options: list[frozenset[str]] = []
    for token in normalized_name_tokens(value):
        token_options = {token}
        token_options.update(_safe_name_keys(token))
        token_options.update(_LOCAL_CHILD_NAME_ALIASES.get(token, ()))
        options.append(frozenset(token_options))
    return options


_NULLISH_CHILD_NAMES = {"null", "none", "nan", "unknown", "не указан", "неизвестно"}


def _canonical_child_name_sort_key(value: str) -> tuple[bool, bool, bool, int, str]:
    text = str(value or "").strip()
    tokens = normalized_name_tokens(text)
    return (_is_nullish_child_name(text), len(tokens) < 2, not bool(tokens), len(text), text.casefold())


def _mark_weak_competing_child_candidates(groups: Iterable[ChildGroup], *, identity_risk: bool) -> None:
    candidates = [group for group in groups if not group.suspicious_reasons and group.canonical_name]
    if len(candidates) < 2:
        return
    has_non_mail_group = any(not _group_is_mail_only(group) for group in candidates)
    stronger_groups = [group for group in candidates if len(group.evidence) >= 2]
    for group in candidates:
        if _group_is_mail_only(group) and has_non_mail_group:
            group.suspicious_reasons.add("weak_mail_only_child_candidate")
            continue
        if (
            identity_risk
            and stronger_groups
            and len(group.evidence) == 1
            and (
                (_group_has_only_single_token_names(group) and not group.grades)
                or _group_traits_covered_by_stronger_candidate(group, stronger_groups)
            )
        ):
            group.suspicious_reasons.add("weak_one_off_child_candidate")


def _group_is_mail_only(group: ChildGroup) -> bool:
    return bool(group.evidence) and all(item.source_system == "a2v3_mail_event_facts" for item in group.evidence)


def _group_traits_covered_by_stronger_candidate(group: ChildGroup, stronger_groups: Sequence[ChildGroup]) -> bool:
    if not group.grades and not group.subjects:
        return False
    group_grades = {_normalize_trait(value) for value in group.grades}
    group_subjects = {_normalize_trait(value) for value in group.subjects}
    for stronger in stronger_groups:
        if stronger is group:
            continue
        stronger_grades = {_normalize_trait(value) for value in stronger.grades}
        stronger_subjects = {_normalize_trait(value) for value in stronger.subjects}
        grades_covered = not group_grades or bool(group_grades & stronger_grades)
        subjects_covered = not group_subjects or bool(group_subjects & stronger_subjects)
        if grades_covered and subjects_covered:
            return True
    return False


def _normalize_trait(value: str) -> str:
    return str(value or "").strip().replace("ё", "е").casefold()


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
    customer_ids = sorted(contexts)
    if not customer_ids:
        return []
    rows: list[dict[str, Any]] = []
    for chunk in _chunks(customer_ids, 900):
        placeholders = ",".join("?" for _ in chunk)
        events = con.execute(
            f"""
            SELECT event_id, tenant_id, customer_id, opportunity_id, event_type, source_id, source_ref,
                   match_status, subject, text_preview, summary, record_json
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
            groups = groups_by_customer.get(customer_id, ())
            exact_group = _exact_tallanto_child_group(event, groups)
            exact_student_id = _event_tallanto_student_id(event)
            attribution = (
                {
                    "child_customer_id": str(exact_group.get("customer_id") or customer_id),
                    "child_key": str(exact_group.get("child_key") or ""),
                    "status": "matched",
                    "confidence": "high",
                    "reason": "exact_tallanto_identity",
                    "matched_names": [str(exact_group.get("canonical_name") or "")],
                }
                if exact_group
                else {
                    "child_key": "",
                    "status": "ambiguous",
                    "confidence": "low",
                    "reason": (
                        "tallanto_student_id_not_in_family"
                        if exact_student_id
                        else "missing_exact_tallanto_student_id"
                    ),
                    "matched_names": [],
                }
                if str(event["event_type"] or "") in TALLANTO_CHILD_EVENT_TYPES
                else _attribute_text(
                    groups,
                    _event_text(event),
                    context=contexts[customer_id],
                    object_kind="event",
                    event_type=str(event["event_type"] or ""),
                )
            )
            if not attribution:
                continue
            child_customer_id = str(attribution.pop("child_customer_id", customer_id))
            payload = {
                "schema_version": FAMILY_GRAPH_SCHEMA_VERSION,
                "tenant_id": tenant_id,
                "event_id": str(event["event_id"]),
                "customer_id": child_customer_id,
                **attribution,
                "created_at": generated_at,
            }
            rows.append(_attribution_row("event_child_attribution_v1", payload))
    return rows


def _exact_tallanto_child_group(
    event: sqlite3.Row,
    groups: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    usable = [group for group in groups if group.get("status") in {"confident", "needs_review", "ambiguous"}]
    student_id = _event_tallanto_student_id(event)
    matches = [group for group in usable if student_id in group.get("tallanto_student_ids", ())]
    return matches[0] if len(matches) == 1 else None


def _event_tallanto_student_id(event: sqlite3.Row) -> str:
    if str(event["event_type"] or "") not in TALLANTO_CHILD_EVENT_TYPES:
        return ""
    if str(event["event_type"]) == "tallanto_student_snapshot":
        source_id = str(event["source_id"] or "")
        customer_id = str(event["customer_id"] or "")
        if (
            source_id == f"tallanto:{customer_id}"
            and str(event["source_ref"] or "") == f"master_contact:{customer_id}:tallanto"
        ):
            return ""
        return source_id
    record = _json_loads(event["record_json"]).get("record") or {}
    if str(event["event_type"]) in {"tallanto_payment", "tallanto_abonement"}:
        return str(record.get("contact_id") or "")
    logical_key = record.get("logical_key") or {}
    return str(record.get("tallanto_student_id") or logical_key.get("tallanto_id") or "")


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
            child_customer_id = str(attribution.pop("child_customer_id", customer_id))
            payload = {
                "schema_version": FAMILY_GRAPH_SCHEMA_VERSION,
                "tenant_id": tenant_id,
                "opportunity_id": str(opp["opportunity_id"]),
                "customer_id": child_customer_id,
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
    if len(matches) == 1 and OTHER_CHILD_REFERENCE_RE.search(text):
        return {
            "child_key": "",
            "status": "ambiguous",
            "confidence": "low",
            "reason": "named_child_plus_other_child_reference",
            "matched_names": [str(matches[0].get("canonical_name") or "")],
        }
    if len(matches) == 1:
        return {
            "child_customer_id": str(matches[0].get("customer_id") or context.customer_id),
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
            "child_customer_id": str(usable[0].get("customer_id") or context.customer_id),
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
        CREATE TABLE IF NOT EXISTS family_members_v1 (
          tenant_id TEXT NOT NULL,
          family_id TEXT NOT NULL,
          customer_id TEXT NOT NULL,
          membership_status TEXT NOT NULL,
          confidence TEXT NOT NULL,
          reason TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          record_hash TEXT NOT NULL,
          record_json TEXT NOT NULL,
          PRIMARY KEY (tenant_id, customer_id)
        );
        CREATE INDEX IF NOT EXISTS ix_family_members_v1_family
          ON family_members_v1(tenant_id, family_id, customer_id);
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
    columns = {str(row["name"]) for row in con.execute("PRAGMA table_info(family_members_v1)")}
    if "updated_at" not in columns:
        con.execute("ALTER TABLE family_members_v1 ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")


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
    member_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    event_rows: Sequence[Mapping[str, Any]],
    opportunity_rows: Sequence[Mapping[str, Any]],
    existing_family_links: int,
    preserve_child_graph: bool,
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
        "family_members_total": len(member_rows),
        "family_members_write_applied": bool(apply),
        "families_total": len({str(row["family_id"]) for row in member_rows}),
        "family_membership_status_counts": _counts(row["membership_status"] for row in member_rows),
        "multi_customer_families": len(
            {
                str(row["family_id"])
                for row in member_rows
                if str(row["membership_status"]) == "confident"
            }
        ),
        "family_links_total": len(family_rows),
        "existing_family_links": int(existing_family_links),
        "child_graph_preserved_without_profiles": bool(preserve_child_graph),
        "child_graph_write_applied": bool(apply and not preserve_child_graph),
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
    normalized_tokens = normalized_name_tokens(value)
    if len(normalized_tokens) >= 2:
        return "_".join(normalized_tokens)[:80]
    keys = _safe_name_keys(value)
    if len(keys) == 1:
        return next(iter(keys))
    normalized = "_".join(normalized_tokens)
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
    if _is_nullish_child_name(text):
        reasons.add("empty_or_null_name")
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


def _is_nullish_child_name(value: str) -> bool:
    return str(value or "").strip().casefold().replace("ё", "е") in _NULLISH_CHILD_NAMES


def _split_subjects(value: str) -> list[str]:
    parts = re.split(r"[;,/|]+", str(value or ""))
    return _dedupe(part.strip() for part in parts if part.strip())


def _normalize_match_text(value: Any) -> str:
    return str(value or "").replace("ё", "е").casefold()


def _normalize_brand(value: Any) -> str:
    text = normalize_key(str(value or "unknown"), "brand")
    return text if text in VALID_BRANDS else "unknown"


def _stable_generated_at(db_path: Path) -> str:
    latest_allowed = datetime.now(timezone.utc).isoformat()
    with _connect(db_path, write=False) as con:
        rows = con.execute(
            """
            SELECT MAX(value) FROM (
              SELECT MAX(created_at) AS value FROM timeline_events
              UNION ALL SELECT MAX(created_at) FROM bot_context_chunks
              UNION ALL SELECT MAX(created_at) FROM derived_signals
            )
            WHERE value <= ?
            """,
            (latest_allowed,),
        ).fetchone()
    return str(rows[0] or "1970-01-01T00:00:00+00:00")


def _family_id(tenant_id: str, customer_id: str) -> str:
    return stable_prefixed_id("family", {"tenant_id": tenant_id, "customer_id": customer_id}, length=32)


def _child_key(customer_id: str, name_key: str) -> str:
    digest = sha256(f"{customer_id}:{name_key}".encode("utf-8")).hexdigest()[:16]
    return f"child:{digest}"


def _tallanto_child_key(tenant_id: str, student_id: str) -> str:
    return _child_key(tenant_id, f"tallanto:{student_id}")


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


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    parsed = _json_loads(value)
    return parsed if isinstance(parsed, list) else []


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
