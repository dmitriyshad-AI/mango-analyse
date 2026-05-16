from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.question_catalog.contracts import (
    BOT_PERMISSION_MANAGER_ONLY,
    SOURCE_CALL,
    QuestionItem,
)


SCHEMA_VERSION = "question_catalog_source_index_v1"


def safe_text(value: Any) -> str:
    return "" if value is None else " ".join(str(value).split()).strip()


def split_tokens(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(split_tokens(item))
        return list(dict.fromkeys(result))
    text = safe_text(value)
    if not text:
        return []
    return [part.strip() for part in text.replace(",", "|").replace(";", "|").split("|") if part.strip()]


def build_source_index_rows(items: Iterable[QuestionItem | dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {
            "theme_id": set(),
            "service_id": set(),
            "policy_status": set(),
            "bot_allowed_mode": set(),
            "risk_flags": set(),
        }
    )
    for item in items:
        payload = item.to_json_dict() if isinstance(item, QuestionItem) else item
        if safe_text(payload.get("source_channel")) != SOURCE_CALL:
            continue
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        call_id = safe_text(metadata.get("call_id") or metadata.get("source_id_raw"))
        if not call_id:
            continue
        theme_id = safe_text(metadata.get("theme_id") or payload.get("intent"))
        if theme_id.startswith("service:"):
            grouped[call_id]["service_id"].add(theme_id)
        elif theme_id:
            grouped[call_id]["theme_id"].add(theme_id)
        status = safe_text(metadata.get("answer_status") or payload.get("answer_evidence_status"))
        permission = safe_text(metadata.get("bot_permission"))
        if status:
            grouped[call_id]["policy_status"].add(status)
        if permission:
            grouped[call_id]["bot_allowed_mode"].add(permission)
        if permission == BOT_PERMISSION_MANAGER_ONLY:
            grouped[call_id]["risk_flags"].add("manager_only")
        for fact_type in payload.get("dynamic_fact_types") or []:
            grouped[call_id]["risk_flags"].add(f"fact:{safe_text(fact_type)}")
    rows = []
    for call_id, values in sorted(grouped.items()):
        rows.append(
            {
                "call_id": call_id,
                "theme_ids": " | ".join(sorted(values["theme_id"])),
                "service_ids": " | ".join(sorted(values["service_id"])),
                "policy_statuses": " | ".join(sorted(values["policy_status"])),
                "bot_allowed_modes": " | ".join(sorted(values["bot_allowed_mode"])),
                "risk_flags": " | ".join(sorted(values["risk_flags"])),
            }
        )
    return rows


def build_source_index(items: Iterable[QuestionItem | dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    result = {}
    for row in build_source_index_rows(items):
        result.update(rows_to_index([row]))
    return result


def rows_to_index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    result = {}
    for row in rows:
        call_id = safe_text(row.get("call_id"))
        result[call_id] = {
            "theme_ids": split_tokens(row.get("theme_ids")),
            "service_ids": split_tokens(row.get("service_ids")),
            "policy_statuses": split_tokens(row.get("policy_statuses")),
            "bot_allowed_modes": split_tokens(row.get("bot_allowed_modes")),
            "risk_flags": split_tokens(row.get("risk_flags")),
        }
    return result


def write_source_index(out_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "question_catalog_source_index.csv"
    json_path = out_root / "question_catalog_source_index.json"
    fieldnames = ["call_id", "theme_ids", "service_ids", "policy_statuses", "bot_allowed_modes", "risk_flags"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    index = rows_to_index(rows)
    payload = {"schema_version": SCHEMA_VERSION, "rows": rows, "index": index}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path), "rows": len(rows)}


def load_source_index(path: Path) -> dict[str, dict[str, list[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("index"), dict):
        return {
            safe_text(call_id): {
                "theme_ids": split_tokens(value.get("theme_ids")),
                "service_ids": split_tokens(value.get("service_ids")),
                "policy_statuses": split_tokens(value.get("policy_statuses")),
                "bot_allowed_modes": split_tokens(value.get("bot_allowed_modes")),
                "risk_flags": split_tokens(value.get("risk_flags")),
            }
            for call_id, value in payload["index"].items()
            if isinstance(value, dict)
        }
    return rows_to_index(payload.get("rows", []) if isinstance(payload, dict) else [])
