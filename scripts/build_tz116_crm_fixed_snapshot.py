#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MAIN_ROOT = Path("/Users/dmitrijfabarisov/Projects/Mango analyse")
DEFAULT_CRM_CALL = DEFAULT_MAIN_ROOT / "audits/_inbox/mcp_tools/crm_call.sh"
DEFAULT_LEADS = DEFAULT_MAIN_ROOT / "product_data/customer_profiles/tz14_amo_step1_full_20260612/amo_leads_raw.jsonl"
DEFAULT_PIPELINES = DEFAULT_MAIN_ROOT / "product_data/customer_profiles/tz14_amo_step1_full_20260612/amo_pipelines.json"

SKIP_LOSS_MARKERS = ("дубль", "спам", "не оставлял", "тест")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    audit_dir = Path(args.audit_dir).expanduser().resolve()
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    leads = list(read_jsonl(Path(args.leads).expanduser().resolve()))
    pipeline_map, status_map = load_pipeline_meta(Path(args.pipelines).expanduser().resolve())
    selected = select_cases(leads, per_brand=int(args.per_brand), pipeline_ids=parse_int_set(args.pipeline_ids))

    cases: list[dict[str, Any]] = []
    redacted_rows: list[dict[str, Any]] = []
    rpc_errors = 0
    for lead in selected:
        lead_id = int(lead.get("id") or 0)
        brand = classify_brand(lead)
        pipeline_id = int(lead.get("pipeline_id") or 0)
        status_id = int(lead.get("status_id") or 0)
        pipeline_name = pipeline_map.get(pipeline_id, "")
        status_name = status_map.get((pipeline_id, status_id), "")
        loss_reason = loss_reason_summary(lead)

        pause_sec = max(1.05, float(args.sleep_sec))
        compact_rpc = call_crm(args.crm_call, "amo_get_lead", {"lead_id": lead_id}, raw_dir / f"lead_{lead_id}_compact.rpc.json")
        time.sleep(pause_sec)
        raw_rpc = call_crm(
            args.crm_call,
            "amo_api_get",
            {"path": f"leads/{lead_id}", "params": {"with": "contacts"}, "limit": 1},
            raw_dir / f"lead_{lead_id}_raw.rpc.json",
        )
        time.sleep(pause_sec)
        notes_rpc = call_crm(
            args.crm_call,
            "amo_api_get",
            {"path": f"leads/{lead_id}/notes", "limit": int(args.notes_limit)},
            raw_dir / f"lead_{lead_id}_notes.rpc.json",
        )
        time.sleep(pause_sec)
        tasks_rpc = call_crm(
            args.crm_call,
            "amo_api_get",
            {
                "path": "tasks",
                "params": {"filter[entity_type]": "leads", "filter[entity_id]": lead_id},
                "limit": int(args.tasks_limit),
            },
            raw_dir / f"lead_{lead_id}_tasks.rpc.json",
        )
        time.sleep(pause_sec)
        if any(item.get("_rpc_error") for item in (compact_rpc, raw_rpc, notes_rpc, tasks_rpc)):
            rpc_errors += 1

        heuristic = build_heuristic(
            lead=lead,
            brand=brand,
            pipeline_name=pipeline_name,
            status_name=status_name,
            loss_reason=loss_reason,
        )
        dossier = {
            "dossier_schema_version": "tz116_crm_fixed_snapshot_v1",
            "brand": brand,
            "lead": {
                "id": lead_id,
                "name": safe_text(lead.get("name")),
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline_name,
                "status_id": status_id,
                "status_name": status_name,
                "responsible_user_id": int(lead.get("responsible_user_id") or 0),
                "created_at": lead.get("created_at"),
                "updated_at": lead.get("updated_at"),
                "closed_at": lead.get("closed_at"),
                "loss_reason": loss_reason,
                "custom_fields": custom_fields_map(lead),
            },
            "live_readonly": {
                "amo_get_lead": compact_rpc,
                "amo_api_get_lead": raw_rpc,
                "notes": notes_rpc,
                "tasks": tasks_rpc,
            },
            "measurement_scope": {
                "source": "read-only AMO via crm_call.sh",
                "writes_enabled": False,
                "brand_isolation": brand,
            },
        }
        case = {
            "case_id": f"tz116_{brand}_{lead_id}",
            "brand": brand,
            "lead_id": lead_id,
            "heuristic_analysis": heuristic,
            "dossier": dossier,
        }
        cases.append(case)
        redacted_rows.append(
            {
                "case_id": case["case_id"],
                "brand": brand,
                "lead_id": lead_id,
                "pipeline_id": pipeline_id,
                "pipeline_name": pipeline_name,
                "status_id": status_id,
                "status_name": status_name,
                "loss_reason_summary": loss_reason,
                "heuristic_verdict": heuristic["close_verdict"],
                "heuristic_risk": heuristic["premature_close_risk"],
            }
        )

    cases_path = out_dir / "crm_fixed_snapshot_cases.jsonl"
    write_jsonl(cases_path, cases)
    write_csv(audit_dir / "crm_fixed_snapshot_manifest_redacted.csv", redacted_rows)
    summary = {
        "schema_version": "tz116_crm_fixed_snapshot_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_total": len(cases),
        "brand_counts": dict(Counter(row["brand"] for row in redacted_rows)),
        "rpc_errors": rpc_errors,
        "cases_path": str(cases_path),
        "audit_manifest": str(audit_dir / "crm_fixed_snapshot_manifest_redacted.csv"),
        "raw_dir": str(raw_dir),
        "safety": {
            "read_only_crm_call": True,
            "writes_amo": False,
            "writes_tallanto": False,
            "token_saved": False,
            "raw_pii_dir_ignored": ".codex_local" in out_dir.parts,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (audit_dir / "summary_redacted.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                yield payload


def select_cases(leads: Sequence[Mapping[str, Any]], *, per_brand: int, pipeline_ids: set[int]) -> list[dict[str, Any]]:
    by_brand: dict[str, list[dict[str, Any]]] = {"foton": [], "unpk": []}
    for lead in leads:
        if int(lead.get("status_id") or 0) != 143:
            continue
        if not lead.get("closed_at"):
            continue
        pipeline_id = int(lead.get("pipeline_id") or 0)
        if pipeline_ids and pipeline_id not in pipeline_ids:
            continue
        brand = classify_brand(lead)
        if brand not in by_brand:
            continue
        loss_reason = loss_reason_summary(lead).casefold()
        if any(marker in loss_reason for marker in SKIP_LOSS_MARKERS):
            continue
        by_brand[brand].append(dict(lead))
    selected: list[dict[str, Any]] = []
    for brand in ("foton", "unpk"):
        candidates = sorted(
            by_brand[brand],
            key=lambda item: (-int(item.get("closed_at") or 0), int(item.get("id") or 0)),
        )
        selected.extend(candidates[:per_brand])
    if len(selected) != per_brand * 2:
        raise SystemExit(f"not enough closed CRM cases selected: expected={per_brand * 2} actual={len(selected)}")
    return selected


def classify_brand(lead: Mapping[str, Any]) -> str:
    text = f"{safe_text(lead.get('name'))} {json.dumps(lead.get('custom_fields_values') or [], ensure_ascii=False)}".casefold()
    matched: set[str] = set()
    if any(marker in text for marker in ("унпк", "unpk", "мфти", "kmipt")):
        matched.add("unpk")
    if any(marker in text for marker in ("фотон", "foton", "цдпо", "cdpo", "cdpofoton")):
        matched.add("foton")
    if len(matched) == 1:
        return next(iter(matched))
    if len(matched) > 1:
        return "conflict"
    return "unknown"


def custom_fields_map(lead: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for field in lead.get("custom_fields_values") or []:
        if not isinstance(field, dict):
            continue
        name = safe_text(field.get("field_name"))
        values = [
            safe_text(value.get("value"))
            for value in field.get("values") or []
            if isinstance(value, dict) and safe_text(value.get("value"))
        ]
        if name and values:
            result[name] = " | ".join(dict.fromkeys(values))
    return result


def loss_reason_summary(lead: Mapping[str, Any]) -> str:
    fields = custom_fields_map(lead)
    values = [
        value
        for key, value in fields.items()
        if "причина отказа" in key.casefold() or "loss" in key.casefold()
    ]
    embedded = lead.get("_embedded") if isinstance(lead.get("_embedded"), dict) else {}
    for item in embedded.get("loss_reason") or []:
        if isinstance(item, dict) and safe_text(item.get("name")):
            values.append(safe_text(item.get("name")))
    return " | ".join(dict.fromkeys(values))


def build_heuristic(
    *,
    lead: Mapping[str, Any],
    brand: str,
    pipeline_name: str,
    status_name: str,
    loss_reason: str,
) -> dict[str, Any]:
    reason = loss_reason.casefold()
    verdict = "closed_valid"
    risk = "no_risk"
    next_step = ""
    if "действующ" in reason:
        summary = "Закрыто как действующий клиент; автоматическое переоткрытие не требуется."
    elif "недозвон" in reason or "нет связи" in reason or "архив" in reason:
        verdict = "follow_up_needed"
        risk = "medium"
        next_step = "Проверить, было ли качественное касание после закрытия; при живом интересе вернуть follow-up."
        summary = "Закрытие связано с недозвоном или отсутствием связи, поэтому риск преждевременного закрытия требует проверки."
    elif "дорого" in reason or "не актуально" in reason or "не подходит" in reason:
        verdict = "manual_review"
        risk = "manual_review"
        next_step = "Проверить контекст отказа и наличие альтернативного предложения."
        summary = "Причина отказа требует ручной проверки: без истории общения нельзя отличить окончательный отказ от отложенного интереса."
    else:
        verdict = "manual_review"
        risk = "manual_review"
        next_step = "Проверить сделку вручную перед любым решением."
        summary = "Недостаточно структурных сигналов для уверенного автоматического вывода."
    return {
        "analysis_schema_version": "tz116_heuristic_snapshot_v1",
        "analysis_source": "heuristic",
        "analysis_mode": "heuristic",
        "matched_lead_id": int(lead.get("id") or 0),
        "match_confidence": 0.78,
        "match_reason": "fixed_closed_deal_snapshot",
        "brand": brand,
        "pipeline_id": int(lead.get("pipeline_id") or 0),
        "pipeline_name": pipeline_name,
        "status_id": int(lead.get("status_id") or 0),
        "status_name": status_name,
        "loss_reason_summary": loss_reason,
        "lead_name": safe_text(lead.get("name")),
        "lead_closed_at": lead.get("closed_at"),
        "close_verdict": verdict,
        "premature_close_risk": risk,
        "close_reason_summary": summary,
        "recommended_next_step": next_step,
        "manager_action_summary": next_step,
        "confidence": 0.78,
        "needs_manual_review": verdict == "manual_review",
        "evidence_signals": [],
        "conflict_flags": [],
    }


def call_crm(crm_call: str, tool: str, payload: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    cmd = ["bash", str(Path(crm_call).expanduser().resolve()), "call", tool, json.dumps(dict(payload), ensure_ascii=False)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=90)
    raw = proc.stdout or proc.stderr or ""
    output_path.write_text(raw, encoding="utf-8")
    if proc.returncode != 0:
        return {"_rpc_error": True, "returncode": proc.returncode, "stderr_tail": "\n".join((proc.stderr or "").splitlines()[-5:])}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"_rpc_error": True, "returncode": proc.returncode, "raw_preview": raw[:500]}
    return parsed if isinstance(parsed, dict) else {"payload": parsed}


def load_pipeline_meta(path: Path) -> tuple[dict[int, str], dict[tuple[int, int], str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pipelines = ((payload.get("_embedded") or {}).get("pipelines") or []) if isinstance(payload, dict) else []
    pipeline_map: dict[int, str] = {}
    status_map: dict[tuple[int, int], str] = {}
    for pipeline in pipelines:
        if not isinstance(pipeline, dict):
            continue
        pipeline_id = int(pipeline.get("id") or 0)
        pipeline_map[pipeline_id] = safe_text(pipeline.get("name"))
        for status in ((pipeline.get("_embedded") or {}).get("statuses") or []):
            if isinstance(status, dict):
                status_map[(pipeline_id, int(status.get("id") or 0))] = safe_text(status.get("name"))
    return pipeline_map, status_map


def parse_int_set(value: Any) -> set[int]:
    result: set[int] = set()
    for part in str(value or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            result.add(int(part))
        except ValueError:
            continue
    return result


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TZ-116 A: build fixed read-only CRM snapshot for offline Codex shadow measurement.")
    parser.add_argument("--leads", default=str(DEFAULT_LEADS))
    parser.add_argument("--pipelines", default=str(DEFAULT_PIPELINES))
    parser.add_argument("--crm-call", default=str(DEFAULT_CRM_CALL))
    parser.add_argument("--out-dir", default=".codex_local/tz116_crm_fixed_snapshot")
    parser.add_argument("--audit-dir", default="audits/_inbox/tz116_crm_fixed_snapshot")
    parser.add_argument("--per-brand", type=int, default=12)
    parser.add_argument("--pipeline-ids", default="8938034,10408062")
    parser.add_argument("--notes-limit", type=int, default=25)
    parser.add_argument("--tasks-limit", type=int, default=25)
    parser.add_argument("--sleep-sec", type=float, default=1.1)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
