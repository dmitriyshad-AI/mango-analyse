from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.insights.phone_identity import client_key_for_phone, normalize_phone, phones_from_text

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}
TERMINAL_RESOLVE = {"done", "skipped"}
NON_CONTENTFUL_TYPES = {"", "non_conversation", "null", "none"}
FILENAME_DATE_RE = re.compile(r"^(20\d\d)-([01]\d)-([0-3]\d)__([0-2]\d)-([0-5]\d)-([0-5]\d)")


@dataclass
class CallCandidate:
    source_filename: str
    source_db: str
    source_db_id: int | None
    source_file: str
    started_at: datetime | None
    month: str
    year: str
    phone_key: str | None
    manager_name: str
    duration_sec: float
    transcription_status: str
    resolve_status: str
    analysis_status: str
    call_type: str
    lead_priority: str
    follow_up_score: int | None
    needs_review: bool
    products: list[str]
    subjects: list[str]
    formats: list[str]
    exam_targets: list[str]
    objections: list[str]
    next_step: str
    history_summary: str
    transcript_chars: int
    analysis_chars: int
    amocrm_contact_id: str
    amocrm_lead_id: str
    score: tuple[int, str] = field(default_factory=lambda: (0, ""))

    @property
    def contentful(self) -> bool:
        return self.call_type.lower() not in NON_CONTENTFUL_TYPES


@dataclass
class TallantoMatch:
    tallanto_ids: set[str] = field(default_factory=set)
    student_types: set[str] = field(default_factory=set)
    branches: set[str] = field(default_factory=set)
    responsible: set[str] = field(default_factory=set)
    history_terms: Counter[str] = field(default_factory=Counter)
    history_samples: list[str] = field(default_factory=list)


@dataclass
class AmoMatch:
    contact_ids: set[str] = field(default_factory=set)
    lead_ids: set[str] = field(default_factory=set)
    statuses: set[str] = field(default_factory=set)
    verdicts: set[str] = field(default_factory=set)
    tallanto_ids: set[str] = field(default_factory=set)
    rows: int = 0


@dataclass
class ReadinessConfig:
    project_root: Path
    coverage_root: Path
    source_dir: Path
    out_root: Path
    start_date: date
    end_date: date
    tallanto_contacts: Path | None
    amo_deal_analysis_root: Path | None
    sample_limit: int


def build_insight_readiness_report(config: ReadinessConfig) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    excluded = _read_name_list(config.coverage_root / "excluded_no_asr.txt")
    source_names = _source_audio_names(config.source_dir, config.start_date, config.end_date) - excluded
    db_paths = _read_included_db_paths(config.coverage_root / "included_dbs.tsv", project_root)

    calls_by_name, scan_stats = _select_best_calls(project_root, db_paths, source_names)
    calls = [calls_by_name[name] for name in sorted(calls_by_name)]
    missing_terminal = sorted(source_names - set(calls_by_name))

    tallanto_index = load_tallanto_index(config.tallanto_contacts) if config.tallanto_contacts else {}
    amo_index = load_amo_deal_index(config.amo_deal_analysis_root) if config.amo_deal_analysis_root else {}

    chain_rows = _build_chain_rows(calls, tallanto_index, amo_index)
    call_rows = [_call_row(call, tallanto_index, amo_index) for call in calls]
    sample_rows = _build_pilot_sample(chain_rows, config.sample_limit)
    summaries = _build_summaries(calls, chain_rows, sample_rows, missing_terminal, scan_stats, config)

    outputs = _write_outputs(out_root, summaries, call_rows, chain_rows, sample_rows)
    summaries["outputs"] = {key: str(path) for key, path in outputs.items()}
    (out_root / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def load_tallanto_index(path: Path | None) -> dict[str, TallantoMatch]:
    if path is None or not path.exists():
        return {}
    index: dict[str, TallantoMatch] = defaultdict(TallantoMatch)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            phones: list[str] = []
            for key in ("phone_parent", "phone_extra", "phones_joined"):
                phones.extend(phones_from_text(row.get(key)))
            for phone in _unique(phones):
                match = index[phone]
                _add_clean(match.tallanto_ids, row.get("tallanto_id"))
                _add_clean(match.student_types, row.get("student_type"))
                _add_clean(match.branches, row.get("branch"))
                _add_clean(match.responsible, row.get("responsible"))
                history = _clean(row.get("history_raw"))
                if history:
                    if len(match.history_samples) < 3:
                        match.history_samples.append(history[:500])
                    for label, pattern in {
                        "payment_terms": r"оплат|счет|сч[её]т|чек|договор",
                        "refusal_terms": r"отказ|не актуаль|неинтерес|не интерес",
                        "active_learning_terms": r"занима|учится|обуча|групп|курс",
                        "thinking_terms": r"дума|подума|реша|совет",
                    }.items():
                        if re.search(pattern, history, flags=re.I):
                            match.history_terms[label] += 1
    return dict(index)


def load_amo_deal_index(root: Path | None) -> dict[str, AmoMatch]:
    if root is None or not root.exists():
        return {}
    index: dict[str, AmoMatch] = defaultdict(AmoMatch)
    for path in sorted(root.glob("*/all_results.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                phone = normalize_phone(row.get("Телефон"))
                if not phone:
                    continue
                match = index[phone]
                match.rows += 1
                _add_clean(match.contact_ids, row.get("ID контакта amoCRM"))
                _add_clean(match.lead_ids, row.get("ID сделки amoCRM"))
                _add_clean(match.statuses, row.get("Статус"))
                _add_clean(match.verdicts, row.get("AI-вердикт"))
                _add_clean(match.tallanto_ids, row.get("Tallanto ID"))
    return dict(index)


def _select_best_calls(project_root: Path, db_paths: list[Path], source_names: set[str]) -> tuple[dict[str, CallCandidate], dict[str, Any]]:
    best: dict[str, CallCandidate] = {}
    scanned_rows = 0
    full_terminal_rows = 0
    db_stats: list[dict[str, Any]] = []
    errors: list[str] = []
    for db_path in db_paths:
        if not db_path.exists():
            continue
        try:
            # Plain path open is intentionally used here. SQLite URI mode can fail
            # on local macOS paths with spaces or WAL sidecar state, while the
            # default connection opens the same DB reliably. This script never
            # writes, so read discipline is enforced by code rather than URI mode.
            con = sqlite3.connect(str(db_path), timeout=15)
            con.row_factory = sqlite3.Row
            if not _has_call_records(con):
                con.close()
                continue
            local_rows = local_hits = local_full = 0
            columns = {row[1] for row in con.execute("pragma table_info(call_records)")}
            select_columns = [
                "id", "source_filename", "source_file", "started_at", "phone", "manager_name", "duration_sec",
                "transcription_status", "resolve_status", "analysis_status", "dead_letter_stage", "transcript_text",
                "analysis_json", "amocrm_contact_id", "amocrm_lead_id", "updated_at",
            ]
            available = [col for col in select_columns if col in columns]
            sql = f"select {', '.join(available)} from call_records where source_filename is not null and source_filename != ''"
            for row in con.execute(sql):
                local_rows += 1
                scanned_rows += 1
                name = _clean(row["source_filename"])
                if name not in source_names:
                    continue
                local_hits += 1
                trs = _norm(row["transcription_status"])
                rs = _norm(row["resolve_status"])
                ans = _norm(row["analysis_status"])
                if not (trs == "done" and rs in TERMINAL_RESOLVE and ans == "done"):
                    continue
                local_full += 1
                full_terminal_rows += 1
                candidate = _candidate_from_row(project_root, db_path, row)
                old = best.get(name)
                if old is None or candidate.score > old.score:
                    best[name] = candidate
            con.close()
            db_stats.append({
                "db": _rel(db_path, project_root),
                "rows_scanned": local_rows,
                "source_hits": local_hits,
                "full_terminal_hits": local_full,
            })
        except sqlite3.Error as exc:
            errors.append(f"{_rel(db_path, project_root)}: {exc}")
    return best, {
        "db_count": len(db_paths),
        "dbs_scanned": len(db_stats),
        "rows_scanned": scanned_rows,
        "full_terminal_rows": full_terminal_rows,
        "db_stats": db_stats,
        "errors": errors,
    }


def _candidate_from_row(project_root: Path, db_path: Path, row: sqlite3.Row) -> CallCandidate:
    name = _clean(row["source_filename"])
    analysis_raw = _clean(row["analysis_json"])
    analysis = _safe_json(analysis_raw)
    structured = _dict(analysis.get("structured_fields")) or _dict(analysis.get("crm_blocks"))
    quality = _dict(analysis.get("quality_flags"))
    interests = _dict(_dict(structured.get("interests")))
    commercial = _dict(_dict(structured.get("commercial")))
    next_step = _dict(_dict(structured.get("next_step")))
    call_type = _clean(quality.get("call_type")) or "unknown"
    started_at = _parse_datetime(row["started_at"]) or _parse_datetime_from_filename(name)
    month = started_at.strftime("%Y-%m") if started_at else "unknown"
    year = started_at.strftime("%Y") if started_at else "unknown"
    phone = normalize_phone(row["phone"])
    if not phone:
        contacts = _dict(structured.get("contacts"))
        phone = normalize_phone(contacts.get("phone_from_filename"))
    if not phone:
        phones = phones_from_text(name)
        phone = phones[0] if phones else None
    transcript = _clean(row["transcript_text"])
    lead_priority = _clean(structured.get("lead_priority")) or _clean(analysis.get("lead_priority"))
    follow_up_score = _as_int(analysis.get("follow_up_score"))
    needs_review = bool(quality.get("needs_review") or analysis.get("needs_review"))
    updated_at = _clean(row["updated_at"])
    score = (
        1000000
        + (20000 if analysis_raw else 0)
        + (5000 if call_type not in NON_CONTENTFUL_TYPES else 0)
        + min(len(analysis_raw), 200000) // 10
        + min(len(transcript), 50000) // 10,
        updated_at,
    )
    return CallCandidate(
        source_filename=name,
        source_db=_rel(db_path, project_root),
        source_db_id=_as_int(row["id"]),
        source_file=_clean(row["source_file"]),
        started_at=started_at,
        month=month,
        year=year,
        phone_key=phone,
        manager_name=_clean(row["manager_name"]),
        duration_sec=float(row["duration_sec"] or 0.0),
        transcription_status=_norm(row["transcription_status"]),
        resolve_status=_norm(row["resolve_status"]),
        analysis_status=_norm(row["analysis_status"]),
        call_type=call_type,
        lead_priority=lead_priority,
        follow_up_score=follow_up_score,
        needs_review=needs_review,
        products=_string_list(interests.get("products")) + _string_list(analysis.get("target_product")),
        subjects=_string_list(interests.get("subjects")),
        formats=_string_list(interests.get("format")),
        exam_targets=_string_list(interests.get("exam_targets")),
        objections=_string_list(structured.get("objections")) + _string_list(analysis.get("objections")),
        next_step=_clean(next_step.get("action")) or _clean(analysis.get("next_step")),
        history_summary=_clean(analysis.get("history_summary")) or _clean(analysis.get("summary")),
        transcript_chars=len(transcript),
        analysis_chars=len(analysis_raw),
        amocrm_contact_id=_clean(row["amocrm_contact_id"]),
        amocrm_lead_id=_clean(row["amocrm_lead_id"]),
        score=score,
    )


def _build_chain_rows(calls: list[CallCandidate], tallanto_index: dict[str, TallantoMatch], amo_index: dict[str, AmoMatch]) -> list[dict[str, Any]]:
    grouped: dict[str, list[CallCandidate]] = defaultdict(list)
    for call in calls:
        if call.phone_key:
            grouped[call.phone_key].append(call)
    rows: list[dict[str, Any]] = []
    for phone, items in grouped.items():
        items.sort(key=lambda c: c.started_at or datetime.min)
        contentful = [c for c in items if c.contentful]
        call_types = Counter(c.call_type for c in items)
        contentful_types = Counter(c.call_type for c in contentful)
        managers = {c.manager_name for c in items if c.manager_name}
        years = {c.year for c in items if c.year != "unknown"}
        months = {c.month for c in items if c.month != "unknown"}
        amo_ids = {c.amocrm_contact_id for c in items if c.amocrm_contact_id}
        lead_ids = {c.amocrm_lead_id for c in items if c.amocrm_lead_id}
        tallanto = tallanto_index.get(phone)
        amo = amo_index.get(phone)
        if amo:
            amo_ids.update(amo.contact_ids)
            lead_ids.update(amo.lead_ids)
        products = _counter_from_lists(c.products for c in items)
        subjects = _counter_from_lists(c.subjects for c in items)
        objections = _counter_from_lists(c.objections for c in items)
        next_steps = sum(1 for c in items if c.next_step)
        first = items[0].started_at
        last = items[-1].started_at
        touch_bucket = _touch_bucket(len(items))
        outcome_sources: list[str] = []
        if tallanto:
            outcome_sources.append("tallanto_match")
        if amo_ids or lead_ids or amo:
            outcome_sources.append("amo_link")
        utility = _utility_score(items, tallanto, amo, call_types, contentful_types, managers)
        row = {
            "client_key": client_key_for_phone(phone),
            "phone": phone,
            "first_seen_at": _dt_to_str(first),
            "last_seen_at": _dt_to_str(last),
            "first_year": first.strftime("%Y") if first else "unknown",
            "last_year": last.strftime("%Y") if last else "unknown",
            "years": _join_sorted(years),
            "months_count": len(months),
            "touch_count": len(items),
            "touch_bucket": touch_bucket,
            "contentful_call_count": len(contentful),
            "non_conversation_count": call_types.get("non_conversation", 0),
            "sales_call_count": call_types.get("sales_call", 0),
            "service_call_count": call_types.get("service_call", 0),
            "technical_call_count": call_types.get("technical_call", 0),
            "existing_client_progress_count": call_types.get("existing_client_progress", 0),
            "dominant_call_type": contentful_types.most_common(1)[0][0] if contentful_types else call_types.most_common(1)[0][0],
            "manager_count": len(managers),
            "managers": _join_sorted(managers),
            "next_step_count": next_steps,
            "needs_review_count": sum(1 for c in items if c.needs_review),
            "products_top": _counter_join(products, 5),
            "subjects_top": _counter_join(subjects, 5),
            "objections_top": _counter_join(objections, 5),
            "has_tallanto_match": bool(tallanto),
            "tallanto_ids_count": len(tallanto.tallanto_ids) if tallanto else 0,
            "tallanto_ids": _join_sorted(tallanto.tallanto_ids if tallanto else []),
            "tallanto_student_types": _join_sorted(tallanto.student_types if tallanto else []),
            "tallanto_branches": _join_sorted(tallanto.branches if tallanto else []),
            "tallanto_history_terms": _counter_join(tallanto.history_terms if tallanto else Counter(), 10),
            "has_amo_link": bool(amo_ids or lead_ids or amo),
            "amo_contact_ids_count": len(amo_ids),
            "amo_lead_ids_count": len(lead_ids),
            "amo_contact_ids": _join_sorted(amo_ids),
            "amo_lead_ids": _join_sorted(lead_ids),
            "amo_statuses": _join_sorted(amo.statuses if amo else []),
            "amo_verdicts": _join_sorted(amo.verdicts if amo else []),
            "outcome_source": ", ".join(outcome_sources) if outcome_sources else "unknown",
            "outcome_availability": "known_proxy" if outcome_sources else "unknown",
            "sample_stratum": _sample_stratum(items, tallanto, amo, contentful_types),
            "utility_score": utility,
            "example_latest_summary": items[-1].history_summary[:600],
        }
        rows.append(row)
    rows.sort(key=lambda r: (-int(r["utility_score"]), r["phone"]))
    return rows


def _call_row(call: CallCandidate, tallanto_index: dict[str, TallantoMatch], amo_index: dict[str, AmoMatch]) -> dict[str, Any]:
    tallanto = tallanto_index.get(call.phone_key or "")
    amo = amo_index.get(call.phone_key or "")
    return {
        "source_filename": call.source_filename,
        "source_db": call.source_db,
        "started_at": _dt_to_str(call.started_at),
        "year": call.year,
        "month": call.month,
        "phone": call.phone_key or "",
        "client_key": client_key_for_phone(call.phone_key),
        "manager_name": call.manager_name,
        "duration_sec": round(call.duration_sec, 3),
        "call_type": call.call_type,
        "contentful": call.contentful,
        "lead_priority": call.lead_priority,
        "follow_up_score": call.follow_up_score if call.follow_up_score is not None else "",
        "needs_review": call.needs_review,
        "products": " | ".join(_unique(call.products)),
        "subjects": " | ".join(_unique(call.subjects)),
        "formats": " | ".join(_unique(call.formats)),
        "exam_targets": " | ".join(_unique(call.exam_targets)),
        "objections": " | ".join(_unique(call.objections)),
        "next_step": call.next_step,
        "has_tallanto_match": bool(tallanto),
        "has_amo_link": bool(call.amocrm_contact_id or call.amocrm_lead_id or amo),
        "history_summary": call.history_summary,
    }


def _build_summaries(
    calls: list[CallCandidate],
    chain_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    missing_terminal: list[str],
    scan_stats: dict[str, Any],
    config: ReadinessConfig,
) -> dict[str, Any]:
    phone_calls = [c for c in calls if c.phone_key]
    no_phone_calls = [c for c in calls if not c.phone_key]
    contentful = [c for c in calls if c.contentful]
    chains = chain_rows
    chains_with_content = [r for r in chains if int(r["contentful_call_count"]) > 0]
    by_year: dict[str, Counter[str]] = defaultdict(Counter)
    for call in calls:
        c = by_year[call.year]
        c["calls"] += 1
        c["contentful_calls"] += int(call.contentful)
        c["no_phone_calls"] += int(not call.phone_key)
        c[f"call_type:{call.call_type}"] += 1
    for row in chains:
        for year in re.split(r"\s*\|\s*", str(row["years"] or "unknown")):
            if not year:
                continue
            c = by_year[year]
            c["chains_touching_year"] += 1
            c["chains_with_tallanto"] += int(bool(row["has_tallanto_match"]))
            c["chains_with_amo"] += int(bool(row["has_amo_link"]))
            c["chains_with_known_proxy_outcome"] += int(row["outcome_availability"] == "known_proxy")
    call_type_counts = Counter(c.call_type for c in calls)
    chain_touch_counts = Counter(str(row["touch_bucket"]) for row in chains)
    sample_strata = Counter(str(row["sample_stratum"]) for row in chains)
    manager_counts = Counter(c.manager_name for c in contentful if c.manager_name)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "coverage_root": str(config.coverage_root.resolve()),
        "date_window": {"start": config.start_date.isoformat(), "end": config.end_date.isoformat()},
        "totals": {
            "terminal_analyzed_calls": len(calls),
            "terminal_missing_from_coverage_source": len(missing_terminal),
            "calls_with_phone": len(phone_calls),
            "calls_without_phone": len(no_phone_calls),
            "unique_client_phones": len(chains),
            "client_chains_with_contentful_calls": len(chains_with_content),
            "contentful_calls": len(contentful),
            "non_conversation_calls": call_type_counts.get("non_conversation", 0),
            "chains_with_tallanto_match": sum(1 for row in chains if row["has_tallanto_match"]),
            "chains_with_amo_link": sum(1 for row in chains if row["has_amo_link"]),
            "chains_with_tallanto_or_amo_proxy_outcome": sum(1 for row in chains if row["outcome_availability"] == "known_proxy"),
            "multi_touch_chains": sum(1 for row in chains if int(row["touch_count"]) >= 2),
            "long_chains_4_plus": sum(1 for row in chains if int(row["touch_count"]) >= 4),
            "pilot_sample_rows": len(sample_rows),
        },
        "call_type_counts": dict(call_type_counts.most_common()),
        "chain_touch_buckets": dict(chain_touch_counts.most_common()),
        "sample_strata_counts": dict(sample_strata.most_common()),
        "top_contentful_managers": manager_counts.most_common(30),
        "by_year": {year: dict(counter) for year, counter in sorted(by_year.items())},
        "scan_stats": scan_stats,
        "notes": [
            "AMO was introduced operationally in 2026, so 2025 AMO gaps are not treated as data-quality failures.",
            "Tallanto match is a historical/customer-context proxy, not a guaranteed payment label.",
            "Outcome availability here means AMO or Tallanto context exists; exact paid/lost labels require the next outcome-linker phase.",
        ],
    }


def _build_pilot_sample(chain_rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    quotas = {
        "tallanto_matched_contentful": 120,
        "amo_linked_2026": 90,
        "multi_touch_contentful": 100,
        "sales_call": 80,
        "service_or_existing": 60,
        "unknown_outcome_contentful": 50,
        "technical_or_review": 40,
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in chain_rows:
        by_stratum[str(row["sample_stratum"])].append(row)
    for stratum, quota in quotas.items():
        for row in sorted(by_stratum.get(stratum, []), key=lambda r: (-int(r["utility_score"]), r["phone"]))[:quota]:
            key = str(row["client_key"])
            if key in seen:
                continue
            seen.add(key)
            selected.append(dict(row, sample_reason=stratum))
            if len(selected) >= limit:
                return selected
    for row in chain_rows:
        key = str(row["client_key"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(dict(row, sample_reason="top_utility_fill"))
        if len(selected) >= limit:
            break
    return selected


def _write_outputs(
    out_root: Path,
    summary: dict[str, Any],
    call_rows: list[dict[str, Any]],
    chain_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
) -> dict[str, Path]:
    paths = {
        "calls_csv": out_root / "calls_terminal_analyzed.csv",
        "client_chains_csv": out_root / "client_chains.csv",
        "pilot_sample_csv": out_root / "pilot_stratified_sample.csv",
        "summary_json": out_root / "summary.json",
    }
    _write_csv(paths["calls_csv"], call_rows)
    _write_csv(paths["client_chains_csv"], chain_rows)
    _write_csv(paths["pilot_sample_csv"], sample_rows)
    xlsx_path = out_root / "insight_readiness_report.xlsx"
    try:
        _write_xlsx(xlsx_path, summary, call_rows, chain_rows, sample_rows)
        paths["xlsx"] = xlsx_path
    except Exception as exc:  # noqa: BLE001
        (out_root / "xlsx_error.txt").write_text(str(exc), encoding="utf-8")
    return paths


def _write_xlsx(path: Path, summary: dict[str, Any], call_rows: list[dict[str, Any]], chain_rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for key, value in summary.get("totals", {}).items():
        summary_rows.append({"metric": key, "value": value})
    for note in summary.get("notes", []):
        summary_rows.append({"metric": "note", "value": note})
    by_year_rows = []
    for year, values in summary.get("by_year", {}).items():
        row = {"year": year}
        row.update(values)
        by_year_rows.append(row)
    call_type_rows = [{"call_type": k, "count": v} for k, v in summary.get("call_type_counts", {}).items()]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(by_year_rows).to_excel(writer, sheet_name="By Year", index=False)
        pd.DataFrame(call_type_rows).to_excel(writer, sheet_name="Call Types", index=False)
        pd.DataFrame(chain_rows[:5000]).to_excel(writer, sheet_name="Client Chains", index=False)
        pd.DataFrame(sample_rows).to_excel(writer, sheet_name="Pilot Sample", index=False)
        pd.DataFrame(call_rows[:5000]).to_excel(writer, sheet_name="Calls Sample", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column_cells in sheet.columns:
                max_len = 0
                col = column_cells[0].column_letter
                for cell in column_cells[:200]:
                    max_len = max(max_len, len(str(cell.value or "")))
                sheet.column_dimensions[col].width = min(max(max_len + 2, 10), 52)


def _read_included_db_paths(path: Path, project_root: Path) -> list[Path]:
    paths: list[Path] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            raw = _clean(row.get("db"))
            if raw:
                p = Path(raw)
                paths.append(p if p.is_absolute() else (project_root / p).resolve())
    return paths


def _read_name_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _source_audio_names(source_dir: Path, start: date, end: date) -> set[str]:
    names: set[str] = set()
    for path in source_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        dt = _parse_datetime_from_filename(path.name)
        if dt is None:
            continue
        if start <= dt.date() <= end:
            names.add(path.name)
    return names


def _has_call_records(con: sqlite3.Connection) -> bool:
    return bool(con.execute("select 1 from sqlite_master where type='table' and name='call_records'").fetchone())


def _utility_score(items: list[CallCandidate], tallanto: TallantoMatch | None, amo: AmoMatch | None, call_types: Counter[str], contentful_types: Counter[str], managers: set[str]) -> int:
    score = 0
    score += min(len(items), 8) * 4
    score += min(sum(contentful_types.values()), 8) * 8
    score += min(call_types.get("sales_call", 0), 5) * 12
    score += min(call_types.get("service_call", 0), 5) * 5
    score += min(call_types.get("technical_call", 0), 5) * 3
    score += 20 if tallanto else 0
    score += 20 if amo else 0
    score += 8 if len(managers) >= 2 else 0
    score += 10 if len(items) >= 4 else 0
    score += 10 if any(c.year == "2026" for c in items) else 0
    score += min(sum(1 for c in items if c.next_step), 5) * 4
    score += min(sum(1 for c in items if c.objections), 5) * 4
    if sum(contentful_types.values()) == 0:
        score -= 80
    return score


def _sample_stratum(items: list[CallCandidate], tallanto: TallantoMatch | None, amo: AmoMatch | None, contentful_types: Counter[str]) -> str:
    has_content = sum(contentful_types.values()) > 0
    if tallanto and has_content:
        return "tallanto_matched_contentful"
    if amo and any(c.year == "2026" for c in items):
        return "amo_linked_2026"
    if len(items) >= 2 and has_content:
        return "multi_touch_contentful"
    if contentful_types.get("sales_call", 0):
        return "sales_call"
    if contentful_types.get("service_call", 0) or contentful_types.get("existing_client_progress", 0):
        return "service_or_existing"
    if contentful_types.get("technical_call", 0):
        return "technical_or_review"
    if has_content:
        return "unknown_outcome_contentful"
    return "low_value_non_conversation"


def _touch_bucket(count: int) -> str:
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 7:
        return "4-7"
    if count <= 15:
        return "8-15"
    return "16+"


def _counter_from_lists(values: Iterable[list[str]]) -> Counter[str]:
    c: Counter[str] = Counter()
    for items in values:
        for item in items:
            cleaned = _clean(item)
            if cleaned:
                c[cleaned] += 1
    return c


def _counter_join(counter: Counter[str], limit: int) -> str:
    return " | ".join(f"{k}: {v}" for k, v in counter.most_common(limit))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_clean(item) for item in value if _clean(item)]
    cleaned = _clean(value)
    return [cleaned] if cleaned else []


def _unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean(value)
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _add_clean(target: set[str], value: Any) -> None:
    cleaned = _clean(value)
    if cleaned:
        target.add(cleaned)


def _join_sorted(values: Iterable[str]) -> str:
    return " | ".join(sorted(str(value) for value in values if str(value).strip()))


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return re.sub(r"\s+", " ", text)


def _norm(value: Any) -> str:
    return _clean(value).lower()


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    text = _clean(value)
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%d.%m.%Y %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _parse_datetime_from_filename(name: str) -> datetime | None:
    match = FILENAME_DATE_RE.match(name)
    if not match:
        return None
    y, m, d, hh, mm, ss = match.groups()
    return datetime(int(y), int(m), int(d), int(hh), int(mm), int(ss))


def _dt_to_str(value: datetime | None) -> str:
    return value.isoformat(sep=" ") if value else ""


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build readiness report for sales insight / knowledge-base extraction.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--coverage-root", default="stable_runtime/final_processing_coverage_report_20260507_v5")
    parser.add_argument("--source-dir", default="2026-03-09--26")
    parser.add_argument("--out-root", default="stable_runtime/insight_readiness_report_20260507")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-05-31")
    parser.add_argument("--tallanto-contacts", default="stable_runtime/tallanto_snapshot_20260331/tallanto_contacts_normalized.csv")
    parser.add_argument("--amo-deal-analysis-root", default="stable_runtime/amocrm_runtime/deal_analysis")
    parser.add_argument("--sample-limit", type=int, default=500)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ReadinessConfig:
    project_root = Path(args.project_root).expanduser().resolve()
    return ReadinessConfig(
        project_root=project_root,
        coverage_root=(project_root / args.coverage_root).resolve(),
        source_dir=(project_root / args.source_dir).resolve(),
        out_root=(project_root / args.out_root).resolve(),
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        tallanto_contacts=(project_root / args.tallanto_contacts).resolve() if args.tallanto_contacts else None,
        amo_deal_analysis_root=(project_root / args.amo_deal_analysis_root).resolve() if args.amo_deal_analysis_root else None,
        sample_limit=int(args.sample_limit),
    )
