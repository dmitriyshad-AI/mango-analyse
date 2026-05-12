from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from mango_mvp.insights.sanitizers import has_brand_risk, has_money_or_terms_risk, has_personal_data_risk


NO_LIVE_RE = re.compile(
    r"абонент(?:\s+сейчас)?\s+(?:не\s+может|не\s+отвечает|не\s+ответил|недоступен|временно\s+недоступен)|"
    r"вызываемый\s+абонент|абонент\s+занят|голосов(?:ая|ой)\s+(?:почта|почтовый\s+ящик)|"
    r"остав(?:ить|ьте)\s+сообщени|после\s+звуков(?:ого)?\s+сигнал|недозвон|"
    r"живого\s+диалога\s+не\s+было|разговора\s+с\s+клиент[а-я]*\s+не\s+было|"
    r"контакт\s+не\s+состоя|клиент\s+не\s+ответил|не\s+удалось\s+(?:связаться|дозвониться|поговорить)",
    re.I,
)

ASR_ARTIFACT_RE = re.compile(
    r"DimaTorzok|субтитры\s+сделал|продолжение\s+следует|\bKim(?:\s+Kim){2,}\b|"
    r"\bOl[áa]\b|Norske\s+Lagerforskning|Thank\s+you\s+for\s+watching|Редактор\s+субтитров",
    re.I,
)

MONEY_OR_TERMS_RE = re.compile(
    r"(?:\d[\d\s]{2,}\s*(?:руб|₽)|\b\d+\s*%|скидк\w*|рассрочк\w*|возврат\w*|брон[ьи]\w*|"
    r"до\s+\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря))",
    re.I,
)

BRAND_RISK_RE = re.compile(
    r"\b(?:НПК|ОНПК|МПК|УНФК|НП\s*К|О\s*Н\s*П\s*К)\s*М\s*[ФШ]\s*[ТД]\s*[ИI]?\b|"
    r"черн(?:ый|ой)\s+центр|чеб[её]н?центр|чебноцентр|вечерний\s+центр",
    re.I,
)

BOT_READY_STATUSES = {"ready_for_bot_draft", "needs_rop_validation"}


@dataclass(frozen=True)
class BaselineConfig:
    project_root: Path
    readiness_root: Path
    kb_root: Path
    rop_root: Path
    out_root: Path
    suspicious_sample_limit: int = 500


def build_transcript_quality_baseline(config: BaselineConfig) -> dict[str, Any]:
    project_root = config.project_root.resolve()
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    readiness_calls = config.readiness_root / "calls_terminal_analyzed.csv"
    readiness_chains = config.readiness_root / "client_chains.csv"
    readiness_summary = config.readiness_root / "summary.json"
    kb_reviews = config.kb_root / "enriched_reviews.csv"
    kb_summary = config.kb_root / "summary.json"
    rop_validation = config.rop_root / "rop_validation.csv"
    rop_summary = config.rop_root / "summary.json"

    input_files = [
        readiness_calls,
        readiness_chains,
        readiness_summary,
        kb_reviews,
        kb_summary,
        rop_validation,
        rop_summary,
    ]
    _require_files(input_files)

    readiness_metrics, suspicious_rows, monthly_rows = _readiness_metrics(readiness_calls, config.suspicious_sample_limit)
    chain_metrics = _chain_metrics(readiness_chains)
    kb_metrics = _kb_metrics(kb_reviews)
    rop_metrics = _rop_metrics(rop_validation)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "source_paths": {
            "readiness_root": str(config.readiness_root),
            "kb_root": str(config.kb_root),
            "rop_root": str(config.rop_root),
        },
        "input_fingerprints": [_fingerprint(path, project_root) for path in input_files],
        "readiness_summary": _load_json(readiness_summary),
        "kb_summary": _load_json(kb_summary),
        "rop_summary": _load_json(rop_summary),
        "readiness_metrics": readiness_metrics,
        "chain_metrics": chain_metrics,
        "kb_metrics": kb_metrics,
        "rop_metrics": rop_metrics,
        "baseline_risks": _baseline_risks(readiness_metrics, kb_metrics, rop_metrics),
        "outputs": {},
    }

    outputs = {
        "summary_json": out_root / "summary.json",
        "baseline_markdown": out_root / "BASELINE_REPORT.md",
        "readiness_monthly_call_types_csv": out_root / "readiness_monthly_call_types.csv",
        "suspicious_contentful_sample_csv": out_root / "suspicious_contentful_sample.csv",
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["baseline_markdown"].write_text(_markdown_report(summary), encoding="utf-8")
    _write_csv(outputs["readiness_monthly_call_types_csv"], monthly_rows)
    _write_csv(outputs["suspicious_contentful_sample_csv"], suspicious_rows)

    summary["outputs"] = {key: str(path) for key, path in outputs.items()}
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _readiness_metrics(path: Path, sample_limit: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    total = 0
    contentful_count = 0
    suspicious_contentful = 0
    no_live_history = 0
    asr_artifact_history = 0
    suspicious_with_next_step = 0
    false_email_from_voice_mail = 0
    call_types: Counter[str] = Counter()
    contentful_call_types: Counter[str] = Counter()
    monthly: dict[str, Counter[str]] = {}
    suspicious_by_call_type: Counter[str] = Counter()
    suspicious_by_month: Counter[str] = Counter()
    managers_suspicious: Counter[str] = Counter()
    sample: list[dict[str, Any]] = []

    for row in _read_csv(path):
        total += 1
        call_type = _clean(row.get("call_type")) or "unknown"
        month = _clean(row.get("month")) or "unknown"
        manager = _clean(row.get("manager_name")) or "unknown"
        contentful = _is_true(row.get("contentful"))
        history = _clean(row.get("history_summary"))
        next_step = _clean(row.get("next_step"))
        joined = " ".join([history, next_step])
        has_no_live = bool(NO_LIVE_RE.search(joined))
        has_artifact = bool(ASR_ARTIFACT_RE.search(joined))

        call_types[call_type] += 1
        monthly.setdefault(month, Counter())["total"] += 1
        monthly[month][call_type] += 1
        if contentful:
            contentful_count += 1
            contentful_call_types[call_type] += 1
            monthly[month]["contentful"] += 1
        else:
            monthly[month]["non_contentful"] += 1

        if has_no_live:
            no_live_history += 1
        if has_artifact:
            asr_artifact_history += 1
        if contentful and (has_no_live or has_artifact):
            suspicious_contentful += 1
            suspicious_by_call_type[call_type] += 1
            suspicious_by_month[month] += 1
            managers_suspicious[manager] += 1
            if next_step:
                suspicious_with_next_step += 1
            if len(sample) < sample_limit:
                sample.append(
                    {
                        "source_filename": _clean(row.get("source_filename")),
                        "source_db": _clean(row.get("source_db")),
                        "started_at": _clean(row.get("started_at")),
                        "month": month,
                        "manager_name": manager,
                        "phone": _clean(row.get("phone")),
                        "call_type": call_type,
                        "contentful": str(contentful),
                        "follow_up_score": _clean(row.get("follow_up_score")),
                        "next_step": next_step,
                        "reason": _reason(has_no_live=has_no_live, has_artifact=has_artifact),
                        "history_summary": history,
                    }
                )
        if contentful and re.search(r"голосов\w*\s+почт", joined, re.I) and re.search(r"канал:\s*(?:email|электронная\s+почта)", history, re.I):
            false_email_from_voice_mail += 1

    monthly_rows = [
        {
            "month": month,
            **{key: counts[key] for key in sorted(counts)},
        }
        for month, counts in sorted(monthly.items())
    ]
    metrics = {
        "terminal_analyzed_calls": total,
        "contentful_calls": contentful_count,
        "non_contentful_calls": total - contentful_count,
        "call_type_counts": dict(call_types.most_common()),
        "contentful_call_type_counts": dict(contentful_call_types.most_common()),
        "history_no_live_marker_calls": no_live_history,
        "history_asr_artifact_marker_calls": asr_artifact_history,
        "suspicious_contentful_by_history": suspicious_contentful,
        "suspicious_contentful_with_next_step": suspicious_with_next_step,
        "false_email_from_voice_mail_candidates": false_email_from_voice_mail,
        "suspicious_contentful_by_call_type": dict(suspicious_by_call_type.most_common()),
        "suspicious_contentful_by_month_top": suspicious_by_month.most_common(15),
        "suspicious_contentful_by_manager_top": managers_suspicious.most_common(20),
    }
    return metrics, sample, monthly_rows


def _chain_metrics(path: Path) -> dict[str, Any]:
    total = 0
    touch_buckets: Counter[str] = Counter()
    sample_strata: Counter[str] = Counter()
    outcome_availability: Counter[str] = Counter()
    contentful_counts: Counter[str] = Counter()
    for row in _read_csv(path):
        total += 1
        touch_buckets[_clean(row.get("touch_bucket")) or "unknown"] += 1
        sample_strata[_clean(row.get("sample_stratum")) or "unknown"] += 1
        outcome_availability[_clean(row.get("outcome_availability")) or "unknown"] += 1
        contentful_counts[_bucket_int(row.get("contentful_call_count"))] += 1
    return {
        "client_chains": total,
        "touch_buckets": dict(touch_buckets.most_common()),
        "sample_strata": dict(sample_strata.most_common()),
        "outcome_availability": dict(outcome_availability.most_common()),
        "contentful_call_count_buckets": dict(contentful_counts.most_common()),
    }


def _kb_metrics(path: Path) -> dict[str, Any]:
    total = 0
    answer_patterns: Counter[str] = Counter()
    commercial_usefulness: Counter[str] = Counter()
    bot_statuses: Counter[str] = Counter()
    quality_bands: Counter[str] = Counter()
    no_live_revenue_risk = 0
    no_live_bot_ready = 0
    raw_ideal_brand_risk = 0
    raw_ideal_money_terms = 0
    manager_ideal_brand_risk = 0
    manager_ideal_money_terms = 0
    bot_safe_money_terms = 0
    bot_safe_brand_risk = 0
    bot_safe_personal_data_risk = 0
    sanitizer_blocked = 0
    for row in _read_csv(path):
        total += 1
        pattern = _clean(row.get("answer_pattern")) or "unknown"
        usefulness = _clean(row.get("commercial_usefulness")) or "unknown"
        bot_status = _clean(row.get("bot_seed_status")) or "unknown"
        raw_ideal = _clean(row.get("ideal_answer_example"))
        manager_ideal = _clean(row.get("ideal_answer_manager_sanitized")) or raw_ideal
        bot_safe = _clean(row.get("bot_safe_answer")) or raw_ideal
        bot_safety_status = _clean(row.get("bot_safety_status"))
        answer_patterns[pattern] += 1
        commercial_usefulness[usefulness] += 1
        bot_statuses[bot_status] += 1
        quality_bands[_clean(row.get("quality_band")) or "unknown"] += 1
        if pattern == "no_live_contact_or_voicemail" and usefulness == "revenue_leakage_risk":
            no_live_revenue_risk += 1
        if pattern == "no_live_contact_or_voicemail" and bot_status in BOT_READY_STATUSES:
            no_live_bot_ready += 1
        if bot_status in BOT_READY_STATUSES and has_money_or_terms_risk(bot_safe):
            bot_safe_money_terms += 1
        if has_brand_risk(raw_ideal):
            raw_ideal_brand_risk += 1
        if has_money_or_terms_risk(raw_ideal):
            raw_ideal_money_terms += 1
        if has_brand_risk(manager_ideal):
            manager_ideal_brand_risk += 1
        if has_money_or_terms_risk(manager_ideal):
            manager_ideal_money_terms += 1
        if bot_status in BOT_READY_STATUSES and has_brand_risk(bot_safe):
            bot_safe_brand_risk += 1
        if bot_status in BOT_READY_STATUSES and has_personal_data_risk(bot_safe):
            bot_safe_personal_data_risk += 1
        if bot_safety_status == "blocked_unresolved_safety_risk":
            sanitizer_blocked += 1
    return {
        "reviews": total,
        "answer_pattern_counts": dict(answer_patterns.most_common()),
        "commercial_usefulness_counts": dict(commercial_usefulness.most_common()),
        "bot_seed_status_counts": dict(bot_statuses.most_common()),
        "quality_band_counts": dict(quality_bands.most_common()),
        "no_live_revenue_risk": no_live_revenue_risk,
        "no_live_bot_ready_or_validation": no_live_bot_ready,
        "bot_ready_money_or_terms": bot_safe_money_terms,
        "bot_safe_answer_money_or_terms": bot_safe_money_terms,
        "bot_safe_answer_brand_risk": bot_safe_brand_risk,
        "bot_safe_answer_personal_data_risk": bot_safe_personal_data_risk,
        "manager_ideal_answer_brand_risk": manager_ideal_brand_risk,
        "manager_ideal_answer_money_or_terms": manager_ideal_money_terms,
        "raw_ideal_answer_brand_risk": raw_ideal_brand_risk,
        "raw_ideal_answer_money_or_terms": raw_ideal_money_terms,
        "ideal_answer_brand_risk": manager_ideal_brand_risk,
        "ideal_answer_money_or_terms": manager_ideal_money_terms,
        "sanitizer_blocked": sanitizer_blocked,
    }


def _rop_metrics(path: Path) -> dict[str, Any]:
    total = 0
    categories: Counter[str] = Counter()
    priorities: Counter[str] = Counter()
    no_live_or_artifact = 0
    p0_no_live_or_artifact = 0
    revenue_no_live_or_artifact = 0
    bot_candidate_money_terms = 0
    bot_safe_brand_risk = 0
    bot_safe_personal_data_risk = 0
    manager_ideal_brand_risk = 0
    manager_ideal_money_terms = 0
    for row in _read_csv(path):
        total += 1
        category = _clean(row.get("Категория проверки")) or "unknown"
        priority = _clean(row.get("Приоритет")) or "unknown"
        ideal = _clean(row.get("Идеальный ответ"))
        manager_ideal = _clean(row.get("Идеальный ответ для менеджера")) or ideal
        bot_safe = _clean(row.get("Безопасный ответ для бота")) or ideal
        joined = " ".join(_clean(value) for value in row.values())
        suspicious = bool(NO_LIVE_RE.search(joined) or ASR_ARTIFACT_RE.search(joined))
        categories[category] += 1
        priorities[priority] += 1
        if suspicious:
            no_live_or_artifact += 1
        if "P0" in priority and suspicious:
            p0_no_live_or_artifact += 1
        if "Риск потери выручки" in category and suspicious:
            revenue_no_live_or_artifact += 1
        if "бот" in category.lower() and has_money_or_terms_risk(bot_safe):
            bot_candidate_money_terms += 1
        if "бот" in category.lower() and has_brand_risk(bot_safe):
            bot_safe_brand_risk += 1
        if "бот" in category.lower() and has_personal_data_risk(bot_safe):
            bot_safe_personal_data_risk += 1
        if has_brand_risk(manager_ideal):
            manager_ideal_brand_risk += 1
        if has_money_or_terms_risk(manager_ideal):
            manager_ideal_money_terms += 1
    return {
        "rows": total,
        "category_counts": dict(categories.most_common()),
        "priority_counts": dict(priorities.most_common()),
        "no_live_or_artifact_rows": no_live_or_artifact,
        "p0_no_live_or_artifact": p0_no_live_or_artifact,
        "revenue_risk_no_live_or_artifact": revenue_no_live_or_artifact,
        "bot_candidate_money_or_terms": bot_candidate_money_terms,
        "bot_safe_answer_brand_risk": bot_safe_brand_risk,
        "bot_safe_answer_personal_data_risk": bot_safe_personal_data_risk,
        "ideal_answer_brand_risk": manager_ideal_brand_risk,
        "ideal_answer_money_or_terms": manager_ideal_money_terms,
    }


def _baseline_risks(readiness: dict[str, Any], kb: dict[str, Any], rop: dict[str, Any]) -> dict[str, Any]:
    return {
        "readiness_suspicious_contentful": readiness["suspicious_contentful_by_history"],
        "readiness_suspicious_with_next_step": readiness["suspicious_contentful_with_next_step"],
        "readiness_false_email_from_voice_mail_candidates": readiness["false_email_from_voice_mail_candidates"],
        "kb_no_live_revenue_risk": kb["no_live_revenue_risk"],
        "kb_raw_ideal_answer_brand_risk": kb["raw_ideal_answer_brand_risk"],
        "kb_raw_ideal_answer_money_or_terms": kb["raw_ideal_answer_money_or_terms"],
        "kb_bot_ready_money_or_terms": kb["bot_ready_money_or_terms"],
        "kb_ideal_answer_brand_risk": kb["ideal_answer_brand_risk"],
        "kb_bot_safe_answer_brand_risk": kb["bot_safe_answer_brand_risk"],
        "kb_bot_safe_answer_personal_data_risk": kb["bot_safe_answer_personal_data_risk"],
        "rop_p0_no_live_or_artifact": rop["p0_no_live_or_artifact"],
        "rop_revenue_risk_no_live_or_artifact": rop["revenue_risk_no_live_or_artifact"],
        "rop_bot_candidate_money_or_terms": rop["bot_candidate_money_or_terms"],
        "rop_bot_safe_answer_brand_risk": rop["bot_safe_answer_brand_risk"],
        "rop_bot_safe_answer_personal_data_risk": rop["bot_safe_answer_personal_data_risk"],
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    risks = summary["baseline_risks"]
    readiness = summary["readiness_metrics"]
    kb = summary["kb_metrics"]
    rop = summary["rop_metrics"]
    lines = [
        "# Transcript Quality Baseline v1",
        "",
        f"Generated at: `{summary['generated_at']}`",
        "",
        "## Readiness",
        "",
        f"- Terminal analyzed calls: `{readiness['terminal_analyzed_calls']}`",
        f"- Contentful calls: `{readiness['contentful_calls']}`",
        f"- Non-contentful calls: `{readiness['non_contentful_calls']}`",
        f"- Suspicious contentful by history: `{readiness['suspicious_contentful_by_history']}`",
        f"- Suspicious contentful with next step: `{readiness['suspicious_contentful_with_next_step']}`",
        f"- False email from voice mail candidates: `{readiness['false_email_from_voice_mail_candidates']}`",
        "",
        "## Knowledge Base",
        "",
        f"- Reviews: `{kb['reviews']}`",
        f"- No-live revenue risk: `{kb['no_live_revenue_risk']}`",
        f"- Raw ideal answer brand risk before sanitizer: `{kb['raw_ideal_answer_brand_risk']}`",
        f"- Raw ideal answer money/terms before sanitizer: `{kb['raw_ideal_answer_money_or_terms']}`",
        f"- Bot-ready rows with money/terms: `{kb['bot_ready_money_or_terms']}`",
        f"- Bot-safe brand risk: `{kb['bot_safe_answer_brand_risk']}`",
        f"- Bot-safe personal data risk: `{kb['bot_safe_answer_personal_data_risk']}`",
        f"- Manager ideal answer brand risk: `{kb['ideal_answer_brand_risk']}`",
        "",
        "## ROP Pack",
        "",
        f"- Rows: `{rop['rows']}`",
        f"- P0 no-live/artifact rows: `{rop['p0_no_live_or_artifact']}`",
        f"- Revenue risk no-live/artifact rows: `{rop['revenue_risk_no_live_or_artifact']}`",
        f"- Bot-candidate rows with money/terms: `{rop['bot_candidate_money_or_terms']}`",
        f"- Bot-safe brand risk: `{rop['bot_safe_answer_brand_risk']}`",
        f"- Bot-safe personal data risk: `{rop['bot_safe_answer_personal_data_risk']}`",
        "",
        "## Baseline Risks",
        "",
    ]
    for key, value in risks.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Input Fingerprints",
            "",
        ]
    )
    for item in summary["input_fingerprints"]:
        lines.append(f"- `{item['path']}`: size `{item['size_bytes']}`, sha256 `{item['sha256']}`")
    lines.append("")
    return "\n".join(lines)


def _read_csv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        yield from csv.DictReader(fh)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required baseline inputs: " + ", ".join(missing))


def _fingerprint(path: Path, project_root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        rel = path.resolve().relative_to(project_root)
    except ValueError:
        rel = path.resolve()
    return {
        "path": str(rel),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _clean(value: Any) -> str:
    return str(value or "").replace("\x00", "").strip()


def _is_true(value: Any) -> bool:
    return _clean(value).lower() in {"1", "true", "yes", "да"}


def _bucket_int(value: Any) -> str:
    try:
        number = int(float(_clean(value)))
    except ValueError:
        return "unknown"
    if number <= 0:
        return "0"
    if number == 1:
        return "1"
    if number <= 3:
        return "2-3"
    if number <= 7:
        return "4-7"
    return "8+"


def _reason(*, has_no_live: bool, has_artifact: bool) -> str:
    reasons = []
    if has_no_live:
        reasons.append("no_live_marker")
    if has_artifact:
        reasons.append("asr_artifact_marker")
    return "|".join(reasons)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build transcript quality v1 baseline report.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--readiness-root", type=Path, required=True)
    parser.add_argument("--kb-root", type=Path, required=True)
    parser.add_argument("--rop-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--suspicious-sample-limit", type=int, default=500)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> BaselineConfig:
    return BaselineConfig(
        project_root=args.project_root,
        readiness_root=args.readiness_root,
        kb_root=args.kb_root,
        rop_root=args.rop_root,
        out_root=args.out_root,
        suspicious_sample_limit=args.suspicious_sample_limit,
    )


__all__ = [
    "BaselineConfig",
    "build_transcript_quality_baseline",
    "config_from_args",
    "parse_args",
]
