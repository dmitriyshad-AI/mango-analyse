#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mango_mvp.customer_timeline.store import json_dumps


DEFAULT_CRM_CALL = Path("/Users/dmitrijfabarisov/Projects/Mango analyse/audits/_inbox/mcp_tools/crm_call.sh")
DEFAULT_BATCH_SIZE = 20
PAID_STATUSES = {"Оплата получена", "Успешно", "won", "success", "paid"}


def main() -> int:
    args = parse_args()
    db_path = Path(args.timeline_db).expanduser()
    out_path = Path(args.out).expanduser()
    allowed_root = Path(args.allowed_root).expanduser().resolve(strict=False)
    _guard_output(out_path, allowed_root)
    _guard_readonly_db(db_path)
    _guard_staging_path(out_path, allowed_root, label="output")
    _guard_staging_path(db_path, allowed_root, label="timeline DB")
    lead_ids = _paid_amo_lead_ids(db_path, tenant_id=args.tenant_id)
    leads, missing = fetch_amo_prices(
        lead_ids,
        crm_call=Path(args.crm_call).expanduser(),
        batch_size=args.batch_size,
    )
    report = {
        "schema_version": "stage5_amo_price_readonly_snapshot_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tenant_id": args.tenant_id,
        "source": {
            "tool": "crm_call.sh",
            "tool_mode": "read_only",
            "tool_path": str(Path(args.crm_call).expanduser()),
        },
        "safety": {
            "crm_write": False,
            "tallanto_write": False,
            "raw_payload_persisted": False,
            "pii_fields_persisted": False,
        },
        "selection": {
            "paid_amo_leads_in_staging": len(lead_ids),
            "fetched": len(leads),
            "missing": len(missing),
            "missing_ids": missing[:50],
            "missing_ids_truncated": max(0, len(missing) - 50),
        },
        "amo_leads": leads,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json_dumps(report), encoding="utf-8")
    print(json_dumps({"out": str(out_path), "selection": report["selection"], "safety": report["safety"]}))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch read-only AMO lead prices for stage5 staging money ingest.")
    parser.add_argument("--timeline-db", required=True, help="Staging customer_timeline SQLite DB")
    parser.add_argument("--allowed-root", required=True, help="Output must stay under this root")
    parser.add_argument("--out", required=True, help="Safe-projection JSON output path under allowed-root")
    parser.add_argument("--tenant-id", default="foton")
    parser.add_argument("--crm-call", default=str(DEFAULT_CRM_CALL))
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args()


def fetch_amo_prices(lead_ids: Sequence[int], *, crm_call: Path, batch_size: int) -> tuple[list[Mapping[str, Any]], list[int]]:
    if not crm_call.exists():
        raise FileNotFoundError(f"crm_call.sh not found: {crm_call}")
    if batch_size < 1 or batch_size > 250:
        raise ValueError("batch-size must be between 1 and 250")
    result: dict[int, Mapping[str, Any]] = {}
    for chunk in _chunks(tuple(lead_ids), batch_size):
        args = {
            "path": "leads",
            "params": {"filter[id][]": list(chunk)},
            "limit": len(chunk),
        }
        payload = _crm_call(crm_call, "amo_api_get", args)
        for lead in _extract_leads(payload):
            lead_id = _as_int(lead.get("id"))
            if lead_id is None:
                continue
            result[lead_id] = _safe_lead_projection(lead)
    missing = [lead_id for lead_id in lead_ids if lead_id not in result]
    return [result[lead_id] for lead_id in lead_ids if lead_id in result], missing


def _paid_amo_lead_ids(db_path: Path, *, tenant_id: str) -> tuple[int, ...]:
    uri = f"{db_path.resolve(strict=False).as_uri()}?mode=ro&immutable=1"
    with sqlite3.connect(uri, uri=True) as con:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        rows = con.execute(
            """
            SELECT DISTINCT source_id
            FROM customer_opportunities
            WHERE tenant_id = ?
              AND source_system = 'amocrm_snapshot'
              AND source_id GLOB '[0-9]*'
              AND (
                status IN ('Оплата получена', 'Успешно')
                OR lower(coalesce(status, '')) IN ('won', 'success', 'paid')
              )
            ORDER BY CAST(source_id AS INTEGER)
            """,
            (tenant_id,),
        ).fetchall()
    return tuple(int(row["source_id"]) for row in rows)


def _crm_call(crm_call: Path, tool: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
    completed = subprocess.run(
        ["bash", str(crm_call), "call", tool, json.dumps(args, ensure_ascii=False, separators=(",", ":"))],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"crm_call.sh {tool} failed with rc={completed.returncode}: {completed.stderr[:300]}")
    data = json.loads(completed.stdout)
    content = (data.get("result") or {}).get("content") or []
    text = content[0].get("text") if content and isinstance(content[0], Mapping) else None
    if not text:
        return data
    parsed = json.loads(str(text), strict=False)
    if isinstance(parsed, Mapping):
        return parsed
    raise ValueError(f"unexpected crm_call payload for {tool}")


def _extract_leads(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    embedded = payload.get("_embedded")
    if isinstance(embedded, Mapping):
        leads = embedded.get("leads")
        if isinstance(leads, list):
            return tuple(item for item in leads if isinstance(item, Mapping))
    leads = payload.get("leads")
    if isinstance(leads, list):
        return tuple(item for item in leads if isinstance(item, Mapping))
    if payload.get("id"):
        return (payload,)
    return ()


def _safe_lead_projection(lead: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "id": _as_int(lead.get("id")),
        "price": _as_number(lead.get("price")),
        "status_id": _as_int(lead.get("status_id")),
        "status_name": _safe_text(lead.get("status_name")),
        "pipeline_id": _as_int(lead.get("pipeline_id")),
        "pipeline_name": _safe_text(lead.get("pipeline_name")),
        "created_at": _as_int(lead.get("created_at")),
        "updated_at": _as_int(lead.get("updated_at")),
        "closed_at": _as_int(lead.get("closed_at")),
    }


def _guard_output(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"output must stay under allowed-root: {allowed_root}") from exc
    if any("customer_timeline_prod_" in part for part in resolved.parts):
        raise ValueError(f"refusing to write under prod timeline path: {resolved}")


def _guard_readonly_db(path: Path) -> None:
    resolved = path.resolve(strict=False)
    if any("customer_timeline_prod_" in part for part in resolved.parts):
        raise ValueError(f"refusing to fetch stage5 money from prod timeline path: {resolved}")
    if not path.exists():
        raise FileNotFoundError(f"timeline DB does not exist: {path}")


def _guard_staging_path(path: Path, allowed_root: Path, *, label: str) -> None:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(f"{label} must stay under allowed-root: {allowed_root}") from exc
    parts = tuple(part.casefold() for part in resolved.parts)
    if not any(part == ".codex_local" and parts[index + 1] == "staging" for index, part in enumerate(parts[:-1])):
        raise ValueError(f"{label} must stay under .codex_local/staging: {resolved}")


def _chunks(values: Sequence[int], size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(values[index : index + size]) for index in range(0, len(values), size))


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_number(value: Any) -> int | float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _safe_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


if __name__ == "__main__":
    raise SystemExit(main())
