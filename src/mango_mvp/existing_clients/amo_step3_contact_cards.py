from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from mango_mvp.customer_profile.crm_summary import ProfileRecord, load_profiles_for_summary, mask_phone
from mango_mvp.existing_clients.amo_step1_snapshot import AmoMcpClient, guard_output_root
from mango_mvp.utils.phone import normalize_phone


AMO_STEP3_SCHEMA_VERSION = "tz14_amo_step3_contact_cards_v1"
CONTACT_CARD_FIELD_NAME = "ИИ: профиль клиента"
DEFAULT_PROFILES_DB = Path("product_data/customer_profiles/tz12_working_batch3/customer_profiles.sqlite")
DEFAULT_AMO_SNAPSHOT_DB = Path("product_data/customer_profiles/tz14_amo_step1_full_20260612/amo_step1_snapshot.sqlite")
DEFAULT_OUT_ROOT = Path("product_data/customer_profiles/tz14_amo_step3_stage_a")
PROTECTED_CONTACT_FIELDS = {"Телефон", "ФИО", "Email", "Id Tallanto", "Филиал Tallanto"}
ALLOWED_CONTACT_CARD_FIELD_TYPES = {"textarea", "text", "multitext"}
MAX_CARD_CHARS = 1800


@dataclass(frozen=True)
class ContactCardOptions:
    project_root: Path
    out_root: Path = DEFAULT_OUT_ROOT
    profiles_db: Path = DEFAULT_PROFILES_DB
    amo_snapshot_db: Path = DEFAULT_AMO_SNAPSHOT_DB
    client: AmoMcpClient | None = None
    stage_a_families: int = 20
    generated_at: datetime | None = None


def build_contact_card_stage_a(options: ContactCardOptions) -> Mapping[str, Any]:
    project_root = options.project_root.expanduser().resolve(strict=False)
    out_root = _resolve_out_root(project_root, options.out_root)
    guard_output_root(project_root=project_root, out_root=out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    generated_at = options.generated_at or datetime.now(timezone.utc)
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise ValueError("generated_at must be timezone-aware")
    if options.stage_a_families < 1:
        raise ValueError("stage_a_families must be positive")

    field_check = check_contact_card_field(options.client) if options.client else {"status": "not_checked", "reason": "client_not_provided"}
    candidates, skip_counts = select_stage_a_families(
        profiles_db=options.profiles_db,
        amo_snapshot_db=options.amo_snapshot_db,
        limit=options.stage_a_families,
        generated_at=generated_at,
    )
    rows: list[dict[str, str]] = []
    findings: list[dict[str, str]] = []
    for candidate in candidates:
        row, row_findings = build_contact_card_row(candidate, generated_at=generated_at)
        rows.append(row)
        findings.extend(row_findings)
    outputs = write_contact_card_outputs(
        out_root=out_root,
        rows=rows,
        findings=findings,
        field_check=field_check,
    )
    summary = {
        "schema_version": AMO_STEP3_SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "out_root": str(out_root),
        "profiles_db": str(options.profiles_db),
        "amo_snapshot_db": str(options.amo_snapshot_db),
        "read_only": True,
        "write_crm": False,
        "field_name": CONTACT_CARD_FIELD_NAME,
        "field_check": field_check,
        "stage_a_families_requested": options.stage_a_families,
        "stage_a_families_selected": len({row["family_hash"] for row in rows}),
        "card_rows": len(rows),
        "finding_rows": len(findings),
        "skip_counts": skip_counts,
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    _write_json(outputs["summary_json"], summary)
    return summary


def check_contact_card_field(client: AmoMcpClient) -> Mapping[str, Any]:
    payload = client.amo_api_get(path="contacts/custom_fields", params={}, limit=50)
    fields = (payload.get("_embedded") or {}).get("custom_fields") if isinstance(payload, Mapping) else []
    matches = [field for field in fields if isinstance(field, Mapping) and str(field.get("name") or "").strip() == CONTACT_CARD_FIELD_NAME]
    if not matches:
        return {"status": "missing", "field_name": CONTACT_CARD_FIELD_NAME}
    field = matches[0]
    field_type = str(field.get("type") or "").strip()
    return {
        "status": "ok" if field_type in ALLOWED_CONTACT_CARD_FIELD_TYPES else "wrong_type",
        "field_name": CONTACT_CARD_FIELD_NAME,
        "field_id": str(field.get("id") or ""),
        "field_type": field_type,
        "is_api_only": bool(field.get("is_api_only")),
    }


def select_stage_a_families(
    *,
    profiles_db: Path,
    amo_snapshot_db: Path,
    limit: int,
    generated_at: datetime,
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    profile_rows = _profile_rows(profiles_db)
    phone_profile_counts = _phone_profile_counts(profile_rows)
    contacts_by_phone = _amo_contacts_by_phone(amo_snapshot_db)
    skip_counts = {
        "missing_phone": 0,
        "phone_2plus_profiles": 0,
        "no_amo_contact": 0,
        "no_live_contact": 0,
    }
    candidates: list[Mapping[str, Any]] = []
    selected_profiles: set[str] = set()
    for row in profile_rows:
        if len(selected_profiles) >= limit:
            break
        profile_id = str(row["profile_id"])
        phone = normalize_phone(str(row["primary_phone"] or ""))
        if not phone:
            skip_counts["missing_phone"] += 1
            continue
        if phone_profile_counts.get(phone, 0) != 1:
            skip_counts["phone_2plus_profiles"] += 1
            continue
        contacts = contacts_by_phone.get(phone, [])
        if not contacts:
            skip_counts["no_amo_contact"] += 1
            continue
        live_contacts = [item for item in contacts if item["has_active_lead"] or item["has_tallanto_link"]]
        if not live_contacts:
            skip_counts["no_live_contact"] += 1
            continue
        profiles = load_profiles_for_summary(profiles_db, profile_id=profile_id)
        if not profiles:
            continue
        profile = profiles[0]
        family_hash = _family_hash(profile.profile_id, phone, generated_at.date().isoformat())
        candidates.append(
            {
                "profile": profile,
                "phone": phone,
                "phone_masked": mask_phone(phone),
                "contacts": tuple(live_contacts),
                "family_hash": family_hash,
            }
        )
        selected_profiles.add(profile_id)
    return candidates, skip_counts


def build_contact_card_row(candidate: Mapping[str, Any], *, generated_at: datetime) -> tuple[dict[str, str], list[dict[str, str]]]:
    profile = candidate["profile"]
    if not isinstance(profile, ProfileRecord):
        raise TypeError("candidate profile must be ProfileRecord")
    contacts = tuple(candidate["contacts"])
    card_text = render_contact_card(profile, generated_at=generated_at, include_family=True)
    findings = contact_card_findings(card_text)
    payload_hash = _sha256(card_text)
    status = "dry_run" if not findings else "blocked"
    reason = "live_write_not_enabled" if status == "dry_run" else "quality_gate_findings"
    row = {
        "family_hash": str(candidate["family_hash"]),
        "profile_id": profile.profile_id,
        "contact_ids": " | ".join(str(item["contact_id"]) for item in contacts),
        "field_name": CONTACT_CARD_FIELD_NAME,
        "status": status,
        "reason": reason,
        "card_text": card_text,
        "payload_sha256": payload_hash,
        "live_write": "false",
        "generated_at": generated_at.isoformat(timespec="seconds"),
    }
    return row, [
        {
            "family_hash": row["family_hash"],
            "profile_id": profile.profile_id,
            "risk_type": finding,
            "status": "blocked",
        }
        for finding in findings
    ]


def render_contact_card(profile: ProfileRecord, *, generated_at: datetime, include_family: bool = True) -> str:
    fields = tuple(row for row in profile.active_fields if not str(row.get("superseded_by") or ""))
    child_line = _children_line(fields)
    parent = _latest_value(fields, "parent_name") or profile.display_name
    family_line = f"Семья: родитель {parent or 'не указан'}." if include_family else ""
    next_step = _latest_value(fields, "next_step") or "нет активных договоренностей"
    objections = _latest_value(fields, "objection") or "нет активных возражений"
    tallanto = _tallanto_status(fields)
    lines = [
        f"Ученик: {child_line}.",
    ]
    if family_line:
        lines.append(family_line)
    lines.extend(
        [
            f"Договоренность семьи: {next_step}.",
            f"Возражения: {objections}.",
            f"Tallanto-статус: {tallanto}.",
            f"ИИ: dry-run v1, обновлено {generated_at.date().isoformat()}; проверьте перед использованием.",
        ]
    )
    return _clip("\n".join(_sanitize_line(line) for line in lines if line), MAX_CARD_CHARS)


def contact_card_findings(text: str) -> list[str]:
    findings: list[str] = []
    if len(text) > MAX_CARD_CHARS:
        findings.append("card_too_long")
    if _PHONE_RE.search(text):
        findings.append("raw_phone_in_card")
    protected = sorted(field for field in PROTECTED_CONTACT_FIELDS if field in text and field != CONTACT_CARD_FIELD_NAME)
    if protected:
        findings.append("protected_field_name_in_card")
    if "[Фотон]" in text and "[УНПК]" in text:
        findings.append("mixed_brand_markers")
    return findings


def write_contact_card_outputs(
    *,
    out_root: Path,
    rows: Sequence[Mapping[str, str]],
    findings: Sequence[Mapping[str, str]],
    field_check: Mapping[str, Any],
) -> Mapping[str, Path]:
    outputs = {
        "contact_card_dry_run_csv": out_root / "contact_card_dry_run.csv",
        "findings_csv": out_root / "contact_card_findings.csv",
        "field_check_json": out_root / "amo_contact_field_check.json",
        "summary_json": out_root / "summary.json",
    }
    _write_csv(outputs["contact_card_dry_run_csv"], rows)
    _write_csv(outputs["findings_csv"], findings)
    _write_json(outputs["field_check_json"], field_check)
    return outputs


def _profile_rows(profiles_db: Path) -> list[sqlite3.Row]:
    with _connect_read_only(profiles_db) as con:
        con.row_factory = sqlite3.Row
        return con.execute(
            """
            SELECT profile_id, COALESCE(primary_phone, '') AS primary_phone,
                   COALESCE(last_event_at, '') AS last_event_at
            FROM customer_profiles
            ORDER BY COALESCE(last_event_at, '') DESC, profile_id
            """
        ).fetchall()


def _phone_profile_counts(profile_rows: Sequence[sqlite3.Row]) -> dict[str, int]:
    counts: dict[str, set[str]] = {}
    for row in profile_rows:
        phone = normalize_phone(str(row["primary_phone"] or ""))
        if phone:
            counts.setdefault(phone, set()).add(str(row["profile_id"]))
    return {phone: len(profile_ids) for phone, profile_ids in counts.items()}


def _amo_contacts_by_phone(amo_snapshot_db: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    with _connect_read_only(amo_snapshot_db) as con:
        con.row_factory = sqlite3.Row
        for row in con.execute(
            """
            SELECT contact_id, phones_json, tallanto_ids_json, lead_ids_json, active_lead_ids_json,
                   has_active_lead, has_tallanto_link
            FROM contacts
            ORDER BY contact_id
            """
        ):
            phones = _json_list(row["phones_json"])
            record = {
                "contact_id": str(row["contact_id"]),
                "tallanto_ids": _json_list(row["tallanto_ids_json"]),
                "lead_ids": _json_list(row["lead_ids_json"]),
                "active_lead_ids": _json_list(row["active_lead_ids_json"]),
                "has_active_lead": bool(row["has_active_lead"]),
                "has_tallanto_link": bool(row["has_tallanto_link"]),
            }
            for phone in phones:
                normalized = normalize_phone(str(phone))
                if normalized:
                    result.setdefault(normalized, []).append(record)
    return result


def _children_line(fields: Sequence[Mapping[str, Any]]) -> str:
    child_keys = sorted(
        {
            str(row.get("child_key") or "child_1")
            for row in fields
            if str(row.get("field") or "") in {"child_name", "grade", "subject", "target_product"}
        }
    )
    if not child_keys:
        return "нет активных данных"
    children: list[str] = []
    for child_key in child_keys[:4]:
        name = _latest_value(fields, "child_name", child_key=child_key) or "ребенок"
        grade = _latest_value(fields, "grade", child_key=child_key)
        subject = _latest_value(fields, "subject", child_key=child_key) or _latest_value(fields, "target_product", child_key=child_key)
        brand = _latest_brand(fields, child_key=child_key)
        parts = [name]
        if grade:
            parts.append(f"{grade} кл")
        if subject:
            parts.append(subject)
        if brand:
            parts.append(brand)
        children.append(" - ".join(parts))
    return "; ".join(children)


def _tallanto_status(fields: Sequence[Mapping[str, Any]]) -> str:
    balance = _latest_value(fields, "tallanto_balance")
    group = _latest_value(fields, "tallanto_group")
    if balance or group:
        parts = []
        if group:
            parts.append(f"группа {group}")
        if balance:
            parts.append(f"баланс {balance}")
        return "; ".join(parts)
    return "нет активных данных в профиле"


def _latest_value(fields: Sequence[Mapping[str, Any]], field: str, *, child_key: str | None = None) -> str:
    rows = [
        row
        for row in fields
        if str(row.get("field") or "") == field and (child_key is None or str(row.get("child_key") or "child_1") == child_key)
    ]
    rows.sort(key=lambda row: str(row.get("event_at") or ""), reverse=True)
    return _safe_text(rows[0].get("value")) if rows else ""


def _latest_brand(fields: Sequence[Mapping[str, Any]], *, child_key: str) -> str:
    rows = [row for row in fields if str(row.get("child_key") or "child_1") == child_key]
    rows.sort(key=lambda row: str(row.get("event_at") or ""), reverse=True)
    for row in rows:
        brand = _safe_text(row.get("brand")).casefold()
        if brand == "foton":
            return "[Фотон]"
        if brand == "unpk":
            return "[УНПК]"
    return "[—]" if rows else ""


def _sanitize_line(value: str) -> str:
    text = _safe_text(value)
    text = re.sub(r"\s+", " ", text)
    return text


def _resolve_out_root(project_root: Path, out_root: Path) -> Path:
    out_root = out_root.expanduser()
    if not out_root.is_absolute():
        out_root = project_root / out_root
    return out_root.resolve(strict=False)


def _connect_read_only(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=False)
    uri = f"file:{quote(str(resolved), safe='/:')}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _json_list(value: Any) -> list[Any]:
    try:
        payload = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _family_hash(profile_id: str, phone: str, salt: str) -> str:
    return _sha256(f"{profile_id}:{phone}:{salt}")[:16]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clip(value: str, max_chars: int) -> str:
    text = _safe_text(value)
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


_PHONE_RE = re.compile(r"(?<!\d)(?:\+?7|8|9)(?:[\s().-]*\d){9,14}(?!\d)")
