#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.utils.phone import normalize_phone


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_DB = (
    PROJECT_ROOT / "stable_runtime" / "canonical_master_20260516_after_mango_update_v1" / "canonical_calls_master.db"
)
DEFAULT_PREVIOUS_CHAINS = (
    PROJECT_ROOT
    / "stable_runtime"
    / "insight_readiness_report_after_quality_backfill_20260510_v1"
    / "client_chains.csv"
)
DEFAULT_OUT_ROOT = PROJECT_ROOT / "stable_runtime" / "insight_readiness_report_after_mango_update_20260516_v1"
EXPORT_SCRIPT = PROJECT_ROOT / "scripts" / "build_post_backfill_amo_ready_export.py"

CHAIN_HEADERS = [
    "client_key",
    "phone",
    "first_seen_at",
    "last_seen_at",
    "first_year",
    "last_year",
    "years",
    "months_count",
    "touch_count",
    "touch_bucket",
    "contentful_call_count",
    "non_conversation_count",
    "sales_call_count",
    "service_call_count",
    "technical_call_count",
    "existing_client_progress_count",
    "dominant_call_type",
    "manager_count",
    "managers",
    "next_step_count",
    "needs_review_count",
    "products_top",
    "subjects_top",
    "objections_top",
    "has_tallanto_match",
    "tallanto_ids_count",
    "tallanto_ids",
    "tallanto_student_types",
    "tallanto_branches",
    "tallanto_history_terms",
    "has_amo_link",
    "amo_contact_ids_count",
    "amo_lead_ids_count",
    "amo_contact_ids",
    "amo_lead_ids",
    "amo_statuses",
    "amo_verdicts",
    "outcome_source",
    "outcome_availability",
    "sample_stratum",
    "utility_score",
    "example_latest_summary",
]

CALL_HEADERS = [
    "source_filename",
    "source_db",
    "started_at",
    "year",
    "month",
    "phone",
    "client_key",
    "manager_name",
    "duration_sec",
    "call_type",
    "contentful",
    "lead_priority",
    "follow_up_score",
    "needs_review",
    "products",
    "subjects",
    "formats",
    "exam_targets",
    "objections",
    "next_step",
    "has_tallanto_match",
    "has_amo_link",
    "history_summary",
]

EXTERNAL_CHAIN_FIELDS = [
    "has_tallanto_match",
    "tallanto_ids_count",
    "tallanto_ids",
    "tallanto_student_types",
    "tallanto_branches",
    "tallanto_history_terms",
    "has_amo_link",
    "amo_contact_ids_count",
    "amo_lead_ids_count",
    "amo_contact_ids",
    "amo_lead_ids",
    "amo_statuses",
    "amo_verdicts",
    "outcome_source",
    "outcome_availability",
    "sample_stratum",
    "utility_score",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build phone-chain readiness files directly from a canonical DB.")
    parser.add_argument("--canonical-db", default=str(DEFAULT_CANONICAL_DB))
    parser.add_argument("--previous-client-chains", default=str(DEFAULT_PREVIOUS_CHAINS))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--fresh-from", default="2025-01-01")
    parser.add_argument("--fresh-to", default="2026-05-31")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_export_module() -> Any:
    spec = importlib.util.spec_from_file_location("build_post_backfill_amo_ready_export", EXPORT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {EXPORT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _parse_dt(value: Any) -> datetime | None:
    text = _safe_text(value).replace("T", " ")
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def _split_parts(value: Any) -> list[str]:
    return [part.strip(" ,;") for part in _safe_text(value).replace("\n", " | ").split("|") if part.strip(" ,;")]


def _top_counter(values: list[str], *, limit: int = 8) -> str:
    counter = Counter(value for value in values if value)
    return " | ".join(f"{value}: {count}" for value, count in counter.most_common(limit))


def _unique_join(values: list[str], *, limit: int | None = None) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = _safe_text(value)
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
        if limit is not None and len(out) >= limit:
            break
    return " | ".join(out)


def _touch_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 7:
        return "4-7"
    return "8+"


def _bool_text(value: Any) -> str:
    return "true" if bool(value) else "false"


def main() -> int:
    args = _parse_args()
    canonical_db = Path(args.canonical_db).expanduser().resolve()
    previous_chains = Path(args.previous_client_chains).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    if out_root.exists() and any(out_root.iterdir()) and not args.force:
        raise SystemExit(f"Output root exists and is not empty. Use --force: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    export = _load_export_module()
    call_rows = export._load_call_rows(canonical_db, args.fresh_from, args.fresh_to)
    previous_by_phone = {
        normalize_phone(row.get("phone")): row for row in _read_csv(previous_chains) if normalize_phone(row.get("phone"))
    }

    by_phone: dict[str, list[dict[str, Any]]] = defaultdict(list)
    call_terminal_rows: list[dict[str, Any]] = []
    for row in call_rows:
        phone = normalize_phone(row.get("Телефон клиента"))
        if not phone:
            continue
        dt = _parse_dt(row.get("Дата и время звонка"))
        prev = previous_by_phone.get(phone, {})
        contentful = row.get("Содержательный звонок") == "Да"
        terminal = {
            "source_filename": row.get("Имя исходного файла"),
            "source_db": row.get("Источник лучшего статуса"),
            "started_at": row.get("Дата и время звонка"),
            "year": dt.strftime("%Y") if dt else "",
            "month": dt.strftime("%Y-%m") if dt else "",
            "phone": phone,
            "client_key": phone,
            "manager_name": row.get("Менеджер"),
            "duration_sec": row.get("Длительность, сек"),
            "call_type": row.get("Тип звонка"),
            "contentful": _bool_text(contentful),
            "lead_priority": row.get("Приоритет лида"),
            "follow_up_score": row.get("Вероятность продажи, %"),
            "needs_review": _bool_text(row.get("Нужна ручная проверка") == "Да"),
            "products": row.get("Продукты интереса") or row.get("Рекомендуемый продукт"),
            "subjects": row.get("Предметы интереса"),
            "formats": row.get("Формат обучения"),
            "exam_targets": row.get("Целевые экзамены"),
            "objections": row.get("Возражения"),
            "next_step": row.get("Следующий шаг"),
            "has_tallanto_match": prev.get("has_tallanto_match", "false"),
            "has_amo_link": prev.get("has_amo_link", "false"),
            "history_summary": row.get("Краткое резюме разговора"),
        }
        call_terminal_rows.append(terminal)
        by_phone[phone].append(row)

    chain_rows: list[dict[str, Any]] = []
    for phone, rows in sorted(by_phone.items()):
        rows_sorted = sorted(
            rows,
            key=lambda item: (_parse_dt(item.get("Дата и время звонка")) or datetime.min, _safe_text(item.get("Имя исходного файла"))),
        )
        contentful = [row for row in rows_sorted if row.get("Содержательный звонок") == "Да"]
        dates = [_parse_dt(row.get("Дата и время звонка")) for row in rows_sorted if _parse_dt(row.get("Дата и время звонка"))]
        managers = _unique_join([_safe_text(row.get("Менеджер")) for row in contentful], limit=8)
        products: list[str] = []
        subjects: list[str] = []
        objections: list[str] = []
        next_step_count = 0
        for row in contentful:
            products.extend(_split_parts(row.get("Продукты интереса")))
            products.extend(_split_parts(row.get("Рекомендуемый продукт")))
            subjects.extend(_split_parts(row.get("Предметы интереса")))
            objections.extend(_split_parts(row.get("Возражения")))
            if _safe_text(row.get("Следующий шаг")):
                next_step_count += 1
        call_type_counter = Counter(_safe_text(row.get("Тип звонка")) for row in rows_sorted)
        prev = previous_by_phone.get(phone, {})
        chain = {
            "client_key": phone,
            "phone": phone,
            "first_seen_at": min(dates).isoformat(sep=" ") if dates else "",
            "last_seen_at": max(dates).isoformat(sep=" ") if dates else "",
            "first_year": min(dates).strftime("%Y") if dates else "",
            "last_year": max(dates).strftime("%Y") if dates else "",
            "years": _unique_join([dt.strftime("%Y") for dt in dates]),
            "months_count": len({dt.strftime("%Y-%m") for dt in dates}),
            "touch_count": len(rows_sorted),
            "touch_bucket": _touch_bucket(len(rows_sorted)),
            "contentful_call_count": len(contentful),
            "non_conversation_count": sum(1 for row in rows_sorted if row.get("Тип звонка") == "non_conversation"),
            "sales_call_count": sum(1 for row in rows_sorted if row.get("Тип звонка") == "sales_call"),
            "service_call_count": sum(1 for row in rows_sorted if row.get("Тип звонка") == "service_call"),
            "technical_call_count": sum(1 for row in rows_sorted if row.get("Тип звонка") == "technical_call"),
            "existing_client_progress_count": sum(1 for row in rows_sorted if row.get("Тип звонка") == "existing_client_progress"),
            "dominant_call_type": call_type_counter.most_common(1)[0][0] if call_type_counter else "",
            "manager_count": len({_safe_text(row.get("Менеджер")) for row in contentful if _safe_text(row.get("Менеджер"))}),
            "managers": managers,
            "next_step_count": next_step_count,
            "needs_review_count": sum(1 for row in contentful if row.get("Нужна ручная проверка") == "Да"),
            "products_top": _top_counter(products, limit=8),
            "subjects_top": _top_counter(subjects, limit=8),
            "objections_top": _top_counter(objections, limit=8),
            "example_latest_summary": next(
                (_safe_text(row.get("Краткое резюме разговора")) for row in reversed(rows_sorted) if _safe_text(row.get("Краткое резюме разговора"))),
                "",
            ),
        }
        for field in EXTERNAL_CHAIN_FIELDS:
            chain[field] = prev.get(field, "")
        if not chain.get("has_tallanto_match"):
            chain["has_tallanto_match"] = "false"
        if not chain.get("has_amo_link"):
            chain["has_amo_link"] = "false"
        chain_rows.append(chain)

    _write_csv(out_root / "calls_terminal_analyzed.csv", CALL_HEADERS, call_terminal_rows)
    _write_csv(out_root / "client_chains.csv", CHAIN_HEADERS, chain_rows)
    summary = {
        "schema_version": "insight_readiness_from_canonical_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "canonical_db": str(canonical_db),
        "previous_client_chains": str(previous_chains),
        "calls_terminal_analyzed_rows": len(call_terminal_rows),
        "client_chains_rows": len(chain_rows),
        "unique_phones": len(chain_rows),
        "contentful_calls": sum(1 for row in call_terminal_rows if row["contentful"] == "true"),
        "non_contentful_calls": sum(1 for row in call_terminal_rows if row["contentful"] != "true"),
        "phones_with_old_external_chain": sum(
            1
            for row in chain_rows
            if _safe_text(row.get("has_amo_link")).casefold() == "true"
            or _safe_text(row.get("has_tallanto_match")).casefold() == "true"
        ),
        "safety": {
            "read_only_canonical": True,
            "crm_writes": False,
            "tallanto_writes": False,
            "deleted_files": False,
        },
        "outputs": {
            "calls_terminal_analyzed_csv": str(out_root / "calls_terminal_analyzed.csv"),
            "client_chains_csv": str(out_root / "client_chains.csv"),
            "summary_json": str(out_root / "summary.json"),
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
