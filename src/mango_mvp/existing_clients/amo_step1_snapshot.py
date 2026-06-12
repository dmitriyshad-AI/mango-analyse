from __future__ import annotations

import csv
import json
import re
import sqlite3
import subprocess
import sys
import socket
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from mango_mvp.utils.phone import normalize_phone


AMO_STEP1_SCHEMA_VERSION = "tz14_amo_step1_snapshot_v1"
DEFAULT_PAGE_LIMIT = 50
DEFAULT_SLEEP_SEC = 1.05
DEFAULT_TIMEOUT_SECONDS = 25
DEFAULT_MAX_RETRIES = 3
DEFAULT_USER_AGENT = "foton-tz14-d4-step1/1.0"
DEFAULT_TRANSPORT = "urllib"
DEFAULT_ENV_PATH = Path("~/.mango_secrets/foton_crm_readonly_mcp_connector.env").expanduser()
DEFAULT_OUT_ROOT = Path("product_data/customer_profiles/tz14_amo_step1_snapshot")
CLOSED_STATUS_IDS = {"142", "143"}

_DIMINUTIVES = {
    "вова": "владимир",
    "володя": "владимир",
    "ваня": "иван",
    "даня": "даниил",
    "дима": "дмитрий",
    "димка": "дмитрий",
    "женя": "евгений",
    "катя": "екатерина",
    "кирилл": "кирилл",
    "леша": "алексей",
    "лёша": "алексей",
    "миша": "михаил",
    "никита": "никита",
    "петя": "петр",
    "пётр": "петр",
    "рома": "роман",
    "соня": "софья",
    "софия": "софья",
    "таня": "татьяна",
}


@dataclass(frozen=True)
class AmoMcpConfig:
    connector_url: str
    bearer_token: str
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    user_agent: str = DEFAULT_USER_AGENT
    transport: str = DEFAULT_TRANSPORT


@dataclass(frozen=True)
class AmoContactRecord:
    contact_id: str
    name: str
    name_key: str
    parent_name: str
    parent_key: str
    phones: tuple[str, ...]
    tallanto_ids: tuple[str, ...]
    lead_ids: tuple[str, ...]
    active_lead_ids: tuple[str, ...]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    has_active_lead: bool
    has_tallanto_link: bool


class AmoMcpError(RuntimeError):
    pass


class AmoMcpClient:
    def __init__(self, config: AmoMcpConfig) -> None:
        self.config = config
        self.calls = 0

    def amo_api_get(
        self,
        *,
        path: str,
        params: Optional[Mapping[str, Any]] = None,
        limit: int = DEFAULT_PAGE_LIMIT,
    ) -> Mapping[str, Any]:
        if limit < 1 or limit > 50:
            raise ValueError("TZ-14 Step 1 AMO reads must use limit 1..50")
        payload = {
            "jsonrpc": "2.0",
            "id": self.calls + 1,
            "method": "tools/call",
            "params": {
                "name": "amo_api_get",
                "arguments": {
                    "path": path,
                    "params": dict(params or {}),
                    "limit": limit,
                },
            },
        }
        body = self._post_json_rpc(payload)

        if body.get("error"):
            raise AmoMcpError(f"MCP JSON-RPC error: {body['error']}")
        result = body.get("result") if isinstance(body, Mapping) else None
        if not isinstance(result, Mapping):
            raise AmoMcpError("MCP result is missing")
        if result.get("isError"):
            text = _tool_text(result)
            raise AmoMcpError(f"MCP tool error: {text[:300]}")
        text = _tool_text(result)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as exc:
            if "[TRUNCATED:" in text:
                raise AmoMcpError("MCP tool response was truncated; lower --page-limit and rerun") from exc
            raise AmoMcpError(f"MCP tool returned invalid JSON text at char {exc.pos}") from exc
        if not isinstance(decoded, Mapping):
            raise AmoMcpError("MCP tool returned non-object payload")
        return decoded

    def _post_json_rpc(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.config.transport == "curl":
            return self._post_json_rpc_curl(payload)
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        attempts = max(1, self.config.max_retries + 1)
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                self.config.connector_url,
                data=raw,
                headers={
                    "Authorization": f"Bearer {self.config.bearer_token}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": self.config.user_agent,
                },
                method="POST",
            )
            self.calls += 1
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < attempts:
                    retry_after = _parse_retry_after(exc.headers.get("Retry-After"))
                    time.sleep(retry_after)
                    continue
                raise AmoMcpError(f"MCP HTTP {exc.code}: {detail[:300]}") from exc
            except urllib.error.URLError as exc:
                if attempt < attempts:
                    time.sleep(min(2.0, 0.5 * attempt))
                    continue
                raise AmoMcpError(f"MCP connection failed: {exc.reason}") from exc
            except socket.timeout as exc:
                if attempt < attempts:
                    time.sleep(min(2.0, 0.5 * attempt))
                    continue
                raise AmoMcpError("MCP connection timed out") from exc
            except json.JSONDecodeError as exc:
                raise AmoMcpError("MCP returned invalid JSON") from exc
        raise AmoMcpError("MCP request failed after retries")

    def _post_json_rpc_curl(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        attempts = max(1, self.config.max_retries + 1)
        for attempt in range(1, attempts + 1):
            self.calls += 1
            completed = subprocess.run(
                [
                    "curl",
                    "-sS",
                    "--max-time",
                    str(self.config.timeout_seconds),
                    "-w",
                    "\n%{http_code}",
                    "-X",
                    "POST",
                    self.config.connector_url,
                    "-H",
                    f"Authorization: Bearer {self.config.bearer_token}",
                    "-H",
                    "Content-Type: application/json",
                    "-H",
                    "Accept: application/json",
                    "-H",
                    f"User-Agent: {self.config.user_agent}",
                    "--data-binary",
                    "@-",
                ],
                input=raw,
                capture_output=True,
                check=False,
            )
            stdout = completed.stdout.decode("utf-8", errors="replace")
            stderr = completed.stderr.decode("utf-8", errors="replace")
            body_text, _, status_text = stdout.rpartition("\n")
            status_code = int(status_text) if status_text.isdigit() else 0
            if completed.returncode == 0 and status_code == 200:
                try:
                    return json.loads(body_text)
                except json.JSONDecodeError as exc:
                    raise AmoMcpError("MCP returned invalid JSON") from exc
            if status_code == 429 and attempt < attempts:
                time.sleep(min(10.0, 1.0 + attempt))
                continue
            if completed.returncode in {28, 35, 52, 56} and attempt < attempts:
                time.sleep(min(2.0, 0.5 * attempt))
                continue
            detail = body_text[:300] or stderr[:300] or f"curl exit {completed.returncode}"
            raise AmoMcpError(f"MCP curl HTTP {status_code}: {detail}")
        raise AmoMcpError("MCP curl request failed after retries")


def read_mcp_env(path: Path = DEFAULT_ENV_PATH) -> AmoMcpConfig:
    values: dict[str, str] = {}
    for raw_line in path.expanduser().read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    connector_url = values.get("CONNECTOR_URL", "").strip()
    bearer_token = values.get("BEARER_TOKEN", "").strip()
    if not connector_url or not bearer_token:
        raise ValueError(f"{path} must contain CONNECTOR_URL and BEARER_TOKEN")
    return AmoMcpConfig(connector_url=connector_url, bearer_token=bearer_token)


def build_amo_step1_snapshot(
    *,
    project_root: Path,
    out_root: Path = DEFAULT_OUT_ROOT,
    client: AmoMcpClient,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    sleep_sec: float = DEFAULT_SLEEP_SEC,
    max_contacts: Optional[int] = None,
    max_leads: Optional[int] = None,
    max_pages: Optional[int] = None,
    progress_every: int = 0,
    checkpoint: bool = True,
    generated_at: Optional[datetime] = None,
) -> Mapping[str, Any]:
    project_root = project_root.expanduser().resolve(strict=False)
    out_root = out_root.expanduser()
    if not out_root.is_absolute():
        out_root = project_root / out_root
    out_root = out_root.resolve(strict=False)
    guard_output_root(project_root=project_root, out_root=out_root)
    if page_limit < 1 or page_limit > DEFAULT_PAGE_LIMIT:
        raise ValueError("page_limit must be between 1 and 50")
    if sleep_sec < 0:
        raise ValueError("sleep_sec must not be negative")
    out_root.mkdir(parents=True, exist_ok=True)
    now = generated_at or datetime.now(timezone.utc)

    pipelines = client.amo_api_get(path="leads/pipelines", params={"with": "statuses"}, limit=page_limit)
    contacts, contact_pages = fetch_amo_collection(
        client,
        path="contacts",
        embedded_key="contacts",
        params={"with": "leads"},
        page_limit=page_limit,
        sleep_sec=sleep_sec,
        max_items=max_contacts,
        max_pages=max_pages,
        progress_every=progress_every,
        checkpoint_root=out_root / "_checkpoint" if checkpoint else None,
        collection_name="contacts",
    )
    leads, lead_pages = fetch_amo_collection(
        client,
        path="leads",
        embedded_key="leads",
        params={"with": "contacts"},
        page_limit=page_limit,
        sleep_sec=sleep_sec,
        max_items=max_leads,
        max_pages=max_pages,
        progress_every=progress_every,
        checkpoint_root=out_root / "_checkpoint" if checkpoint else None,
        collection_name="leads",
    )
    records, lead_index = build_contact_records(contacts=contacts, leads=leads, pipelines=pipelines)
    analysis = analyze_contacts(records=records, generated_at=now)
    outputs = write_step1_outputs(
        out_root=out_root,
        generated_at=now,
        contacts=contacts,
        leads=leads,
        pipelines=pipelines,
        records=records,
        lead_index=lead_index,
        analysis=analysis,
    )
    summary = {
        "schema_version": AMO_STEP1_SCHEMA_VERSION,
        "generated_at": now.isoformat(timespec="seconds"),
        "out_root": str(out_root),
        "read_only": True,
        "write_crm": False,
        "mcp_tool": "amo_api_get",
        "page_limit": page_limit,
        "sleep_sec": sleep_sec,
        "contacts_seen": len(contacts),
        "leads_seen": len(leads),
        "contact_pages": contact_pages,
        "lead_pages": lead_pages,
        "api_calls": client.calls,
        "counts": analysis["counts"],
        "outputs": {name: str(path) for name, path in outputs.items()},
        "policy": {
            "raw_pii_under_ignored_product_data": True,
            "live_write_executed": False,
            "duplicate_merge_is_manual_only": True,
            "active_duplicate_filter": "active_lead_or_tallanto_link_present",
        },
    }
    _write_json(outputs["summary_json"], summary)
    return summary


def fetch_amo_collection(
    client: AmoMcpClient,
    *,
    path: str,
    embedded_key: str,
    params: Mapping[str, Any],
    page_limit: int,
    sleep_sec: float,
    max_items: Optional[int],
    max_pages: Optional[int],
    progress_every: int = 0,
    checkpoint_root: Optional[Path] = None,
    collection_name: str = "",
) -> tuple[list[Mapping[str, Any]], int]:
    checkpoint = _load_checkpoint(
        checkpoint_root=checkpoint_root,
        collection_name=collection_name,
        path=path,
        embedded_key=embedded_key,
        params=params,
        page_limit=page_limit,
        max_items=max_items,
        max_pages=max_pages,
    )
    if checkpoint.get("complete"):
        return list(checkpoint["items"]), int(checkpoint["pages"])
    items = list(checkpoint.get("items") or [])
    page = int(checkpoint.get("next_page") or 1)
    while True:
        if max_pages is not None and page > max_pages:
            break
        if max_items is not None and len(items) >= max_items:
            break
        payload = client.amo_api_get(path=path, params={**params, "page": page}, limit=page_limit)
        embedded = payload.get("_embedded") if isinstance(payload, Mapping) else {}
        page_items = embedded.get(embedded_key) if isinstance(embedded, Mapping) else []
        if not isinstance(page_items, list) or not page_items:
            break
        for item in page_items:
            if isinstance(item, Mapping):
                items.append(dict(item))
                if max_items is not None and len(items) >= max_items:
                    break
        _save_checkpoint(
            checkpoint_root=checkpoint_root,
            collection_name=collection_name,
            path=path,
            embedded_key=embedded_key,
            params=params,
            page_limit=page_limit,
            max_items=max_items,
            max_pages=max_pages,
            items=items,
            last_page=page,
            complete=False,
        )
        page += 1
        if progress_every > 0 and (page - 1) % progress_every == 0:
            print(f"{path}: pages={page - 1} items={len(items)}", file=sys.stderr, flush=True)
        next_link = _next_link(payload)
        if not next_link:
            break
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    pages = page - 1
    _save_checkpoint(
        checkpoint_root=checkpoint_root,
        collection_name=collection_name,
        path=path,
        embedded_key=embedded_key,
        params=params,
        page_limit=page_limit,
        max_items=max_items,
        max_pages=max_pages,
        items=items,
        last_page=pages,
        complete=True,
    )
    return items, pages


def _load_checkpoint(
    *,
    checkpoint_root: Optional[Path],
    collection_name: str,
    path: str,
    embedded_key: str,
    params: Mapping[str, Any],
    page_limit: int,
    max_items: Optional[int],
    max_pages: Optional[int],
) -> dict[str, Any]:
    if checkpoint_root is None or not collection_name:
        return {"items": [], "next_page": 1, "pages": 0, "complete": False}
    items_path, state_path = _checkpoint_paths(checkpoint_root, collection_name)
    expected_key = _checkpoint_key(
        path=path,
        embedded_key=embedded_key,
        params=params,
        page_limit=page_limit,
        max_items=max_items,
        max_pages=max_pages,
    )
    state = _read_json_if_exists(state_path)
    if state.get("key") != expected_key:
        return {"items": [], "next_page": 1, "pages": 0, "complete": False}
    items = _read_jsonl(items_path)
    complete = bool(state.get("complete"))
    pages = int(state.get("last_page") or 0)
    return {
        "items": items,
        "next_page": pages + 1,
        "pages": pages,
        "complete": complete,
    }


def _save_checkpoint(
    *,
    checkpoint_root: Optional[Path],
    collection_name: str,
    path: str,
    embedded_key: str,
    params: Mapping[str, Any],
    page_limit: int,
    max_items: Optional[int],
    max_pages: Optional[int],
    items: Sequence[Mapping[str, Any]],
    last_page: int,
    complete: bool,
) -> None:
    if checkpoint_root is None or not collection_name:
        return
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    items_path, state_path = _checkpoint_paths(checkpoint_root, collection_name)
    key = _checkpoint_key(
        path=path,
        embedded_key=embedded_key,
        params=params,
        page_limit=page_limit,
        max_items=max_items,
        max_pages=max_pages,
    )
    state = _read_json_if_exists(state_path)
    previous_count = int(state.get("item_count") or 0) if state.get("key") == key else 0
    mode = "a" if previous_count > 0 else "w"
    new_items = list(items)[previous_count:]
    if new_items or mode == "w":
        with items_path.open(mode, encoding="utf-8") as handle:
            for item in new_items:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    _write_json(
        state_path,
        {
            "key": key,
            "last_page": last_page,
            "item_count": len(items),
            "complete": complete,
        },
    )


def _checkpoint_paths(root: Path, collection_name: str) -> tuple[Path, Path]:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", collection_name)
    return root / f"{safe_name}.jsonl", root / f"{safe_name}_state.json"


def _checkpoint_key(
    *,
    path: str,
    embedded_key: str,
    params: Mapping[str, Any],
    page_limit: int,
    max_items: Optional[int],
    max_pages: Optional[int],
) -> Mapping[str, Any]:
    return {
        "path": path,
        "embedded_key": embedded_key,
        "params": dict(sorted((str(key), value) for key, value in params.items())),
        "page_limit": page_limit,
        "max_items": max_items,
        "max_pages": max_pages,
    }


def build_contact_records(
    *,
    contacts: Sequence[Mapping[str, Any]],
    leads: Sequence[Mapping[str, Any]],
    pipelines: Mapping[str, Any],
) -> tuple[list[AmoContactRecord], dict[str, Mapping[str, Any]]]:
    closed_status_ids = closed_status_ids_from_pipelines(pipelines)
    lead_index = {_safe_id(lead.get("id")): lead for lead in leads if _safe_id(lead.get("id"))}
    contact_to_leads: dict[str, set[str]] = defaultdict(set)
    for lead in leads:
        lead_id = _safe_id(lead.get("id"))
        for ref in embedded_items(lead, "contacts"):
            contact_id = _safe_id(ref.get("id"))
            if lead_id and contact_id:
                contact_to_leads[contact_id].add(lead_id)

    records: list[AmoContactRecord] = []
    for contact in contacts:
        contact_id = _safe_id(contact.get("id"))
        if not contact_id:
            continue
        lead_ids = set(contact_to_leads.get(contact_id, set()))
        for ref in embedded_items(contact, "leads"):
            lead_id = _safe_id(ref.get("id"))
            if lead_id:
                lead_ids.add(lead_id)
                lead_index.setdefault(lead_id, ref)
        active_leads = tuple(sorted(lead_id for lead_id in lead_ids if lead_is_active(lead_index.get(lead_id), closed_status_ids)))
        tallanto_ids = tuple(sorted(set(field_values(contact, names=("Id Tallanto",)))))
        phones = tuple(sorted(set(extract_contact_phones(contact))))
        name = _safe_text(contact.get("name"))
        parent_name = first_field_value(contact, names=("ФИО Родителя", "ФИО родителя"))
        records.append(
            AmoContactRecord(
                contact_id=contact_id,
                name=name,
                name_key=normalize_person_key(name),
                parent_name=parent_name,
                parent_key=normalize_person_key(parent_name),
                phones=phones,
                tallanto_ids=tallanto_ids,
                lead_ids=tuple(sorted(lead_ids)),
                active_lead_ids=active_leads,
                created_at=epoch_to_dt(contact.get("created_at")),
                updated_at=epoch_to_dt(contact.get("updated_at")),
                has_active_lead=bool(active_leads),
                has_tallanto_link=bool(tallanto_ids),
            )
        )
    return records, lead_index


def analyze_contacts(*, records: Sequence[AmoContactRecord], generated_at: datetime) -> Mapping[str, Any]:
    by_phone: dict[str, list[AmoContactRecord]] = defaultdict(list)
    for record in records:
        for phone in record.phones:
            by_phone[phone].append(record)

    duplicate_rows: list[dict[str, str]] = []
    common_rows: list[dict[str, str]] = []
    multi_child_rows: list[dict[str, str]] = []
    ambiguous_rows: list[dict[str, str]] = []
    now_7d = generated_at - timedelta(days=7)

    for phone, phone_records in sorted(by_phone.items()):
        unique_contacts = _unique_records(phone_records)
        by_name: dict[str, list[AmoContactRecord]] = defaultdict(list)
        for record in unique_contacts:
            by_name[record.name_key or f"contact:{record.contact_id}"].append(record)

        for name_key, name_records in sorted(by_name.items()):
            if len(name_records) < 2:
                continue
            live_records = [item for item in name_records if item.has_active_lead or item.has_tallanto_link]
            duplicate_rows.append(_duplicate_row(phone, name_key, name_records, live_records, now_7d))

        distinct_name_keys = sorted({record.name_key for record in unique_contacts if record.name_key})
        if len(distinct_name_keys) < 2:
            continue
        parent_keys = sorted({record.parent_key for record in unique_contacts if record.parent_key})
        row = _phone_family_row(phone, unique_contacts, distinct_name_keys, parent_keys)
        if len(parent_keys) > 1:
            common_rows.append({**row, "review_class": "possible_common_phone_distinct_parents"})
        elif len(parent_keys) == 1:
            multi_child_rows.append({**row, "review_class": "multi_child_same_parent"})
        else:
            ambiguous_rows.append({**row, "review_class": "ambiguous_missing_parent"})

    counts = {
        "contacts_with_phone": sum(1 for record in records if record.phones),
        "contacts_without_phone": sum(1 for record in records if not record.phones),
        "unique_phones": len(by_phone),
        "phones_with_2plus_contacts": sum(1 for rows in by_phone.values() if len({row.contact_id for row in rows}) >= 2),
        "duplicate_groups_total": len(duplicate_rows),
        "duplicate_contact_rows_total": sum(int(row["contact_count"]) for row in duplicate_rows),
        "live_duplicate_groups": sum(1 for row in duplicate_rows if row["live_status"] == "live_candidate"),
        "live_duplicate_contact_rows": sum(int(row["contact_count"]) for row in duplicate_rows if row["live_status"] == "live_candidate"),
        "weekly_duplicate_groups": sum(1 for row in duplicate_rows if row["created_last_7d_contact_ids"]),
        "weekly_duplicate_new_contacts": sum(
            len([item for item in row["created_last_7d_contact_ids"].split(" | ") if item]) for row in duplicate_rows
        ),
        "possible_common_phone_groups": len(common_rows),
        "multi_child_family_groups": len(multi_child_rows),
        "ambiguous_missing_parent_groups": len(ambiguous_rows),
    }
    return {
        "counts": counts,
        "duplicate_rows": duplicate_rows,
        "common_rows": common_rows,
        "multi_child_rows": multi_child_rows,
        "ambiguous_rows": ambiguous_rows,
    }


def write_step1_outputs(
    *,
    out_root: Path,
    generated_at: datetime,
    contacts: Sequence[Mapping[str, Any]],
    leads: Sequence[Mapping[str, Any]],
    pipelines: Mapping[str, Any],
    records: Sequence[AmoContactRecord],
    lead_index: Mapping[str, Mapping[str, Any]],
    analysis: Mapping[str, Any],
) -> Mapping[str, Path]:
    outputs = {
        "contacts_jsonl": out_root / "amo_contacts_raw.jsonl",
        "leads_jsonl": out_root / "amo_leads_raw.jsonl",
        "pipelines_json": out_root / "amo_pipelines.json",
        "snapshot_sqlite": out_root / "amo_step1_snapshot.sqlite",
        "duplicate_candidates_csv": out_root / "duplicate_candidates.csv",
        "common_phone_review_csv": out_root / "common_phone_review.csv",
        "multi_child_families_csv": out_root / "multi_child_families.csv",
        "ambiguous_phone_review_csv": out_root / "ambiguous_phone_review.csv",
        "summary_json": out_root / "summary.json",
    }
    _write_jsonl(outputs["contacts_jsonl"], contacts)
    _write_jsonl(outputs["leads_jsonl"], leads)
    _write_json(outputs["pipelines_json"], pipelines)
    write_snapshot_sqlite(outputs["snapshot_sqlite"], generated_at=generated_at, records=records, lead_index=lead_index)
    _write_csv(outputs["duplicate_candidates_csv"], analysis["duplicate_rows"])
    _write_csv(outputs["common_phone_review_csv"], analysis["common_rows"])
    _write_csv(outputs["multi_child_families_csv"], analysis["multi_child_rows"])
    _write_csv(outputs["ambiguous_phone_review_csv"], analysis["ambiguous_rows"])
    return outputs


def write_snapshot_sqlite(
    path: Path,
    *,
    generated_at: datetime,
    records: Sequence[AmoContactRecord],
    lead_index: Mapping[str, Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    try:
        con.execute("DROP TABLE IF EXISTS metadata")
        con.execute("DROP TABLE IF EXISTS leads")
        con.execute("DROP TABLE IF EXISTS contacts")
        con.execute(
            """
            CREATE TABLE contacts (
              contact_id TEXT PRIMARY KEY,
              name TEXT,
              name_key TEXT,
              parent_name TEXT,
              parent_key TEXT,
              phones_json TEXT,
              tallanto_ids_json TEXT,
              lead_ids_json TEXT,
              active_lead_ids_json TEXT,
              has_active_lead INTEGER,
              has_tallanto_link INTEGER,
              created_at TEXT,
              updated_at TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE leads (
              lead_id TEXT PRIMARY KEY,
              payload_json TEXT
            )
            """
        )
        con.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
        for record in records:
            con.execute(
                """
                INSERT INTO contacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.contact_id,
                    record.name,
                    record.name_key,
                    record.parent_name,
                    record.parent_key,
                    json.dumps(record.phones, ensure_ascii=False),
                    json.dumps(record.tallanto_ids, ensure_ascii=False),
                    json.dumps(record.lead_ids, ensure_ascii=False),
                    json.dumps(record.active_lead_ids, ensure_ascii=False),
                    int(record.has_active_lead),
                    int(record.has_tallanto_link),
                    record.created_at.isoformat() if record.created_at else "",
                    record.updated_at.isoformat() if record.updated_at else "",
                ),
            )
        for lead_id, payload in lead_index.items():
            con.execute("INSERT OR REPLACE INTO leads VALUES (?, ?)", (lead_id, json.dumps(payload, ensure_ascii=False, sort_keys=True)))
        con.execute("INSERT INTO metadata VALUES (?, ?)", ("schema_version", AMO_STEP1_SCHEMA_VERSION))
        con.execute("INSERT INTO metadata VALUES (?, ?)", ("generated_at", generated_at.isoformat(timespec="seconds")))
        con.commit()
    finally:
        con.close()


def _parse_retry_after(value: Optional[str]) -> float:
    try:
        parsed = float(value or "")
    except (TypeError, ValueError):
        return 1.0
    return max(0.1, min(10.0, parsed))


def guard_output_root(*, project_root: Path, out_root: Path) -> None:
    try:
        relative = out_root.relative_to(project_root)
    except ValueError:
        return
    allowed = Path("product_data/customer_profiles")
    if not str(relative).startswith(str(allowed)):
        raise ValueError("TZ-14 AMO snapshot output under the project must be inside product_data/customer_profiles/")


def closed_status_ids_from_pipelines(payload: Mapping[str, Any]) -> set[str]:
    closed = set(CLOSED_STATUS_IDS)
    pipelines = (payload.get("_embedded") or {}).get("pipelines") if isinstance(payload, Mapping) else []
    if not isinstance(pipelines, list):
        return closed
    for pipeline in pipelines:
        statuses = ((pipeline.get("_embedded") or {}).get("statuses") or []) if isinstance(pipeline, Mapping) else []
        for status in statuses:
            name = _safe_text(status.get("name")).casefold()
            status_id = _safe_id(status.get("id"))
            if status_id and ("закрыто" in name or "успеш" in name):
                closed.add(status_id)
    return closed


def lead_is_active(lead: Optional[Mapping[str, Any]], closed_status_ids: set[str]) -> bool:
    if not isinstance(lead, Mapping):
        return False
    status_id = _safe_id(lead.get("status_id"))
    if status_id in closed_status_ids:
        return False
    if _safe_text(lead.get("closed_at")):
        return False
    if bool(lead.get("is_deleted")):
        return False
    return True


def extract_contact_phones(contact: Mapping[str, Any]) -> list[str]:
    phones: list[str] = []
    for field in contact.get("custom_fields_values") or []:
        if not isinstance(field, Mapping):
            continue
        field_name = _safe_text(field.get("field_name")).casefold()
        field_code = _safe_text(field.get("field_code")).casefold()
        if field_code != "phone" and "тел" not in field_name:
            continue
        for item in field.get("values") or []:
            if not isinstance(item, Mapping):
                continue
            phone = normalize_phone(_safe_text(item.get("value")))
            if phone and phone not in phones:
                phones.append(phone)
    return phones


def field_values(contact: Mapping[str, Any], *, names: Sequence[str]) -> list[str]:
    wanted = {name.casefold() for name in names}
    values: list[str] = []
    for field in contact.get("custom_fields_values") or []:
        if not isinstance(field, Mapping):
            continue
        field_name = _safe_text(field.get("field_name")).casefold()
        field_code = _safe_text(field.get("field_code")).casefold()
        if field_name not in wanted and field_code not in wanted:
            continue
        for item in field.get("values") or []:
            if not isinstance(item, Mapping):
                continue
            value = _safe_text(item.get("value"))
            if value:
                values.append(value)
    return values


def first_field_value(contact: Mapping[str, Any], *, names: Sequence[str]) -> str:
    values = field_values(contact, names=names)
    return values[0] if values else ""


def normalize_person_key(value: str) -> str:
    text = _safe_text(value).replace("ё", "е").replace("Ё", "Е").casefold()
    tokens = re.findall(r"[a-zа-я0-9]+", text)
    normalized: list[str] = []
    for token in tokens:
        if token.isdigit() or len(token) <= 1:
            continue
        normalized.append(_DIMINUTIVES.get(token, token))
    return " ".join(sorted(dict.fromkeys(normalized)))


def epoch_to_dt(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


def embedded_items(entity: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    embedded = entity.get("_embedded")
    if not isinstance(embedded, Mapping):
        return []
    items = embedded.get(key)
    return [item for item in items if isinstance(item, Mapping)] if isinstance(items, list) else []


def _tool_text(result: Mapping[str, Any]) -> str:
    content = result.get("content")
    if not isinstance(content, list) or not content or not isinstance(content[0], Mapping):
        raise AmoMcpError("MCP content is missing")
    return str(content[0].get("text") or "")


def _next_link(payload: Mapping[str, Any]) -> str:
    links = payload.get("_links")
    if not isinstance(links, Mapping):
        return ""
    next_item = links.get("next")
    if not isinstance(next_item, Mapping):
        return ""
    return _safe_text(next_item.get("href"))


def _duplicate_row(
    phone: str,
    name_key: str,
    records: Sequence[AmoContactRecord],
    live_records: Sequence[AmoContactRecord],
    cutoff: datetime,
) -> dict[str, str]:
    created_last_7d = [record.contact_id for record in records if record.created_at and record.created_at >= cutoff]
    return {
        "phone": phone,
        "child_name_key": name_key,
        "contact_count": str(len(records)),
        "contact_ids": " | ".join(record.contact_id for record in records),
        "contact_names": " | ".join(record.name for record in records),
        "parent_keys": " | ".join(sorted({record.parent_key for record in records if record.parent_key})),
        "lead_ids": " | ".join(sorted({lead_id for record in records for lead_id in record.lead_ids})),
        "active_lead_ids": " | ".join(sorted({lead_id for record in records for lead_id in record.active_lead_ids})),
        "tallanto_ids": " | ".join(sorted({item for record in records for item in record.tallanto_ids})),
        "live_status": "live_candidate" if live_records else "archive_or_unconfirmed",
        "created_last_7d_contact_ids": " | ".join(created_last_7d),
    }


def _phone_family_row(
    phone: str,
    records: Sequence[AmoContactRecord],
    distinct_name_keys: Sequence[str],
    parent_keys: Sequence[str],
) -> dict[str, str]:
    return {
        "phone": phone,
        "contact_count": str(len(records)),
        "contact_ids": " | ".join(record.contact_id for record in records),
        "child_name_keys": " | ".join(distinct_name_keys),
        "parent_keys": " | ".join(parent_keys),
        "active_lead_ids": " | ".join(sorted({lead_id for record in records for lead_id in record.active_lead_ids})),
        "tallanto_ids": " | ".join(sorted({item for record in records for item in record.tallanto_ids})),
    }


def _unique_records(records: Sequence[AmoContactRecord]) -> list[AmoContactRecord]:
    by_id: dict[str, AmoContactRecord] = {}
    for record in records:
        by_id.setdefault(record.contact_id, record)
    return [by_id[key] for key in sorted(by_id)]


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_id(value: Any) -> str:
    return _safe_text(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_json_if_exists(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.exists():
        return []
    rows: list[Mapping[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, Mapping):
                rows.append(payload)
    return rows


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
