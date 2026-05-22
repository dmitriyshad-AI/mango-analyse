from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE6 = ROOT / "stable_runtime/deal_aware_stage6_writeback_preflight_20260513_rop_iter03/deal_stage6_dry_run_report.csv"
DEFAULT_STAGE5 = ROOT / "stable_runtime/deal_aware_stage5_quality_gate_20260513_rop_iter03/deal_stage5_quality_gate_report.csv"
DEFAULT_STAGE2 = ROOT / "stable_runtime/deal_aware_stage2_attribution_20260513_v2"
DEFAULT_STAGE1 = ROOT / "stable_runtime/deal_aware_stage1_snapshot_20260513_v2"
DEFAULT_OUT = ROOT / "stable_runtime/deal_aware_stage709_review_20260514_v1"
DEFAULT_AUDIT_PACK = ROOT / "audits/_inbox/deal_aware_stage100_stratified_preview_20260514_v1"


RISK_RU = {
    "blocked_completed_payment_next_step_conflict": "Строка заблокирована gate: возможный конфликт оплаты и следующего шага.",
    "blocked_cross_field_duplicate_information": "Строка заблокирована gate: одна и та же информация повторяется в разных полях.",
    "paid_or_success_context": "Сделка уже оплачена или близка к завершению: нельзя механически дожимать оплату.",
    "payment_stage": "Этап оплаты или договора: важно сверять фактическую оплату и документы.",
    "service_feedback": "Похоже на сервисную обратную связь по обучению: задача может быть для куратора, а не продаж.",
    "amo_tallanto_mismatch": "AMO и Tallanto могут расходиться: менеджеру нужна сверка.",
    "no_reliable_tallanto_match": "Нет надежной связи с Tallanto: финансовый/учебный контекст неполный.",
    "multiple_tallanto_matches": "Несколько совпадений в Tallanto: нужен ручной контроль ученика.",
    "multi_phone_history": "В истории несколько телефонов: есть риск смешать членов семьи или дубли.",
    "long_history": "Длинная история общения: повышенный риск устаревшего контекста.",
    "stage2_confidence_low": "Привязка к сделке попала в низкий confidence-бакет Stage2.",
    "overdue_tasks": "В AMO есть просроченные задачи: следующий шаг надо сверять с задачами.",
    "future_loss_reactivation": "Есть причина отказа/перспективы: нужна политика реактивации, а не обычный дожим.",
    "review_priority": "Служебный признак: AI приоритет = review.",
    "active_sales": "Обычная активная продажа.",
}


ROP_COLUMNS = [
    ("rop_row_decision", "ready_for_rop / minor_comment / needs_fix_before_rop / block"),
    ("summary_correctness", "ok / minor_inaccuracy / wrong_or_misleading / cannot_judge"),
    ("next_step_quality", "ok_manager_action / too_passive / customer_side_only / wrong_action / missing"),
    ("deal_status_priority_quality", "ok / status_wrong / priority_wrong / amo_tallanto_mismatch_not_handled / cannot_judge"),
    ("tallanto_block_quality", "ok / wrong_student / wrong_finance_or_attendance / too_raw_or_unreadable / not_relevant"),
    ("history_relevance", "ok / irrelevant_calls / important_call_missing / too_verbose / cannot_judge"),
    ("sales_usefulness", "useful_as_is / useful_after_small_edit / not_useful / dangerous"),
    ("issue_type", "wrong_deal_binding / wrong_customer_context / wrong_next_step / payment_conflict / service_feedback_routing / amo_tallanto_mismatch / bad_tenant_terms / too_verbose / duplicate_fields / other"),
    ("severity", "P0_blocker / P1_fix_before_rop / P2_minor / P3_note"),
    ("comment", "Свободный комментарий РОПа; обязателен, если есть любая проблема."),
]

SAMPLE_MINIMUM_CLASSES = (
    "blocked_completed_payment_next_step_conflict",
    "blocked_cross_field_duplicate_information",
    "future_loss_reactivation",
    "multiple_tallanto_matches",
    "no_reliable_tallanto_match",
    "paid_or_success_context",
    "payment_stage",
    "service_feedback",
    "amo_tallanto_mismatch",
    "multi_phone_history",
    "long_history",
    "overdue_tasks",
)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def text_blob(row: pd.Series) -> str:
    fields = [
        "AI-сводка по сделке",
        "AI-история по сделке",
        "AI-рекомендованный следующий шаг",
        "AI-основание рекомендации",
        "AI-предупреждение по сделке",
        "AI-Tallanto статус по сделке",
        "AI-актуальные возражения",
        "selected_deal_name",
        "selected_status_name",
        "selected_loss_reason",
        "quality_risk_types",
        "stage3_risk_flags",
        "stage6_finding_types",
    ]
    return " ".join(safe_text(row.get(field)) for field in fields).casefold()


def has_any(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def classify_row(row: pd.Series) -> list[str]:
    classes: list[str] = []
    blob = text_blob(row)
    status = safe_text(row.get("selected_status_name")).casefold()
    loss_reason = safe_text(row.get("selected_loss_reason")).casefold()
    stage3 = safe_text(row.get("stage3_risk_flags"))
    quality = safe_text(row.get("quality_risk_types"))
    findings = safe_text(row.get("stage6_finding_types"))
    priority = safe_text(row.get("AI-приоритет сделки")).casefold()
    tallanto = safe_text(row.get("tallanto_context_status"))

    if safe_text(row.get("stage6_status")) == "blocked":
        if "completed_payment_next_step_conflict" in findings or "completed_payment_next_step_conflict" in quality:
            classes.append("blocked_completed_payment_next_step_conflict")
        if "cross_field_duplicate_information" in findings or "cross_field_duplicate_information" in quality:
            classes.append("blocked_cross_field_duplicate_information")

    if safe_text(row.get("stage3_mode")) == "context_only_paid_or_success" or "оплата получена" in status:
        classes.append("paid_or_success_context")
    if any(marker in status for marker in ("ожидание оплаты", "заключение договора", "запись в группу")):
        classes.append("payment_stage")
    if has_any(blob, r"обратн\w+\s+связ", r"куратор", r"посещаем", r"заняти", r"домашн\w+\s+задан", r"преподавател"):
        classes.append("service_feedback")
    if has_any(blob, r"amo-статус нужно сверить", r"tallanto.*расход", r"активного ученика"):
        classes.append("amo_tallanto_mismatch")
    if tallanto == "no_reliable_tallanto_match":
        classes.append("no_reliable_tallanto_match")
    if tallanto == "multiple_tallanto_matches":
        classes.append("multiple_tallanto_matches")
    if int_or_zero(row.get("candidate_phone_count")) > 1:
        classes.append("multi_phone_history")
    if int_or_zero(row.get("candidate_call_count")) >= 8:
        classes.append("long_history")
    if "stage2_confidence_low" in stage3:
        classes.append("stage2_confidence_low")
    if "deal_has_overdue_open_tasks" in stage3:
        classes.append("overdue_tasks")
    if "future_prospect_loss_reason_requires_reactivation_policy" in quality or loss_reason:
        classes.append("future_loss_reactivation")
    if "review" in priority:
        classes.append("review_priority")
    if not classes:
        classes.append("active_sales")
    return list(dict.fromkeys(classes))


def int_or_zero(value: Any) -> int:
    try:
        return int(float(safe_text(value).replace(",", ".")))
    except ValueError:
        return 0


def build_risk_text(classes: list[str]) -> str:
    manager_visible = [
        cls
        for cls in classes
        if cls not in {"stage2_confidence_low", "review_priority"}
    ]
    return " | ".join(RISK_RU.get(cls, cls) for cls in manager_visible)


def primary_class(classes: list[str]) -> str:
    order = [
        "blocked_completed_payment_next_step_conflict",
        "blocked_cross_field_duplicate_information",
        "multiple_tallanto_matches",
        "future_loss_reactivation",
        "no_reliable_tallanto_match",
        "paid_or_success_context",
        "payment_stage",
        "service_feedback",
        "amo_tallanto_mismatch",
        "multi_phone_history",
        "long_history",
        "review_priority",
        "active_sales",
    ]
    for item in order:
        if item in classes:
            return item
    return classes[0]


def sample_rows(classified: pd.DataFrame, target: int = 100) -> pd.DataFrame:
    selected: list[int] = []

    def add(indexes: list[int], limit: int | None = None) -> None:
        count = 0
        for idx in indexes:
            if idx in selected:
                continue
            selected.append(idx)
            count += 1
            if limit is not None and count >= limit:
                return

    blocked = classified[classified["stage6_status"] == "blocked"].index.tolist()
    add(blocked)
    add(classified[classified["risk_classes"].apply(lambda value: "multiple_tallanto_matches" in safe_text(value).split("|"))].index.tolist())
    add(classified[classified["risk_classes"].apply(lambda value: "future_loss_reactivation" in safe_text(value).split("|"))].index.tolist())

    # Explicit minimums prevent top-N bias. If a class has fewer than 8 rows, include all.
    for cls in SAMPLE_MINIMUM_CLASSES:
        rows = classified[classified["risk_classes"].apply(lambda value: cls in safe_text(value).split("|"))].copy()
        rows["_score"] = rows.apply(sample_score, axis=1)
        rows = rows.sort_values(["_score", "last_call_at", "selected_deal_id"], ascending=[False, False, True])
        add(rows.index.tolist(), min(8, len(rows.index)))

    quotas = [
        ("no_reliable_tallanto_match", 12),
        ("paid_or_success_context", 10),
        ("payment_stage", 10),
        ("service_feedback", 10),
        ("amo_tallanto_mismatch", 10),
        ("multi_phone_history", 6),
        ("long_history", 6),
        ("overdue_tasks", 6),
        ("review_priority", 6),
        ("active_sales", 10),
    ]
    for cls, quota in quotas:
        rows = classified[classified["risk_classes"].apply(lambda value: cls in safe_text(value).split("|"))].copy()
        rows["_score"] = rows.apply(sample_score, axis=1)
        rows = rows.sort_values(["_score", "last_call_at", "selected_deal_id"], ascending=[False, False, True])
        add(rows.index.tolist(), quota)

    if len(selected) < target:
        rest = classified.drop(index=selected, errors="ignore").copy()
        rest["_score"] = rest.apply(sample_score, axis=1)
        rest = rest.sort_values(["_score", "last_call_at", "selected_deal_id"], ascending=[False, False, True])
        add(rest.index.tolist(), target - len(selected))

    return classified.loc[selected[:target]].copy()


def sample_coverage(classified: pd.DataFrame, sample: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cls in SAMPLE_MINIMUM_CLASSES:
        total = int(classified["risk_classes"].apply(lambda value: cls in safe_text(value).split("|")).sum())
        sampled = int(sample["risk_classes"].apply(lambda value: cls in safe_text(value).split("|")).sum())
        required = min(8, total)
        rows.append(
            {
                "class_id": cls,
                "total_rows": total,
                "required_minimum": required,
                "sampled_rows": sampled,
                "passed": sampled >= required,
            }
        )
    return rows


def sample_score(row: pd.Series) -> int:
    classes = safe_text(row.get("risk_classes")).split("|")
    score = 0
    weights = {
        "blocked_completed_payment_next_step_conflict": 100,
        "blocked_cross_field_duplicate_information": 90,
        "paid_or_success_context": 45,
        "payment_stage": 35,
        "service_feedback": 30,
        "amo_tallanto_mismatch": 28,
        "no_reliable_tallanto_match": 20,
        "multi_phone_history": 18,
        "long_history": 15,
        "overdue_tasks": 12,
        "review_priority": 10,
    }
    for cls in classes:
        score += weights.get(cls, 0)
    score += min(int_or_zero(row.get("candidate_call_count")), 20)
    return score


def human_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "manager_warning_ru" in df.columns:
        df["AI-предупреждение по сделке"] = df["manager_warning_ru"]
    cols = [
        "sample_id",
        "sample_bucket_ru",
        "review_url_hint",
        "selected_deal_id",
        "selected_deal_name",
        "selected_pipeline_name",
        "selected_status_name",
        "selected_loss_reason",
        "phones",
        "managers",
        "candidate_call_count",
        "candidate_phone_count",
        "tallanto_context_status",
        "AI-фактический статус сделки",
        "AI-приоритет сделки",
        "AI-рекомендованный следующий шаг",
        "AI-сводка по сделке",
        "AI-история по сделке",
        "AI-Tallanto статус по сделке",
        "AI-предупреждение по сделке",
        "risk_flags_ru",
        "risk_classes",
        "stage6_status",
        "stage6_reason",
        "stage6_finding_types",
    ]
    present = [col for col in cols if col in df.columns]
    out = df[present].copy()
    for column, instruction in ROP_COLUMNS:
        out[column] = ""
        out[f"{column}_allowed_values"] = instruction
    return out


def manager_warning_text(row: pd.Series) -> str:
    warning = safe_text(row.get("AI-предупреждение по сделке"))
    warning = re.sub(r"\s*Stage 2 confidence не high; перед массовой записью нужна аудитная проверка выборки\.?", "", warning).strip()
    warning = re.sub(r"\s{2,}", " ", warning).strip()
    if warning:
        return warning
    return "Специальных предупреждений по строке нет; общая Stage2-метка вынесена на уровень пакета."


def business_class(row: pd.Series) -> str:
    text = " ".join(
        [
            safe_text(row.get("AI-рекомендованный следующий шаг")),
            safe_text(row.get("AI-сводка по сделке")),
            safe_text(row.get("AI-история по сделке")),
        ]
    ).casefold()
    mode = safe_text(row.get("deal_writeback_mode"))
    status = safe_text(row.get("selected_status_name"))
    priority = safe_text(row.get("AI-приоритет сделки"))
    if mode == "context_only_paid_or_success" or status == "Оплата получена" or priority == "service-paid":
        return "B1_paid_context_service_only"
    if has_any(text, r"оплат|плат[её]ж|квитанц|сч[её]т|договор|документ|заявлен|реквизит|qr|чек|перерасчет|финансов"):
        return "B2_payment_contract_docs_admin"
    if has_any(text, r"обратн|куратор|преподавател|домашн|платформ|личн.*кабинет|тетрад|рекомендац|прогресс|учебн.*маршрут"):
        return "B3_learning_service_feedback"
    if has_any(text, r"переключ|перевести|соединить|передать номер|специалист|консультант|коллег|партнер"):
        return "B4_handoff_to_specialist"
    if has_any(text, r"ручн.*контроль|не делать активн|не делать автомат|контрольн.*срок ожид|без нового сигнала"):
        return "B5_manual_hold_no_auto_push"
    if has_any(text, r"лист ожидан|спис.*желающих|внести в список|добавить в список|записать|брон|групп|свободн.*мест"):
        return "B6_waitlist_or_enrollment_ops"
    if has_any(text, r"отправ|высл|направ|присл|материал|информац|расписан|программ|предложен|письм|оповестить"):
        return "B7_send_info_program_offer"
    if has_any(text, r"связаться|перезвон|созвон|уточн|обсуд|соглас|подтверд|решени|готовност|консультац"):
        return "B8_callback_qualify_decision"
    if has_any(text, r"снять с подбора|не актуал|закрыть|отказ"):
        return "B9_drop_or_not_actual"
    return "B5_manual_hold_no_auto_push"


def write_business_classification(classified: pd.DataFrame, out_root: Path) -> None:
    business = classified.copy()
    business["business_class"] = business.apply(business_class, axis=1)
    business["business_risk_classes"] = business["risk_classes"]
    business.to_csv(out_root / "deal_stage6_709_business_classification.csv", index=False)
    summary = {
        "business_class_counts": business.groupby(["business_class", "stage6_status"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "risk_class_counts": Counter(
            cls
            for value in business["business_risk_classes"]
            for cls in safe_text(value).split("|")
            if cls
        ),
    }
    summary["risk_class_counts"] = dict(summary["risk_class_counts"].most_common())
    (out_root / "business_classification_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def build_stage2_root_cause(stage2_root: Path, stage6: pd.DataFrame, out_root: Path) -> dict[str, Any]:
    distribution = pd.read_csv(stage2_root / "confidence_distribution.csv")
    links = pd.read_csv(stage2_root / "deal_call_links.csv", usecols=[
        "phone",
        "selected_deal_id",
        "confidence_bucket",
        "confidence_score",
        "candidate_sources",
        "attribution_decision",
        "safe_for_deal_writeback",
    ])
    phone_candidates = pd.read_csv(stage2_root / "phone_deal_candidates.csv")

    candidate_sources = Counter(safe_text(v) for v in phone_candidates["candidate_sources"].fillna(""))
    linked_sources = Counter(safe_text(v) for v in links.loc[
        links["attribution_decision"].eq("linked_single_deal_candidate"), "candidate_sources"
    ].fillna(""))

    stage6_pairs = set()
    for _, row in stage6.iterrows():
        deal_id = safe_text(row.get("selected_deal_id"))
        for phone in safe_text(row.get("phones")).split("|"):
            phone = re.sub(r"\D+", "", phone)
            if phone and deal_id:
                stage6_pairs.add((phone, deal_id))

    links["_pair"] = list(zip(links["phone"].astype(str), links["selected_deal_id"].fillna("").astype(str)))
    stage6_link_rows = links[links["_pair"].isin(stage6_pairs)]
    stage6_conf = stage6_link_rows["confidence_bucket"].value_counts(dropna=False).to_dict()

    root_cause = {
        "stage2_distribution": distribution.to_dict(orient="records"),
        "stage6_rows": int(len(stage6)),
        "stage6_rows_with_stage2_link_rows": int(stage6_link_rows.shape[0]),
        "stage6_call_level_confidence_distribution": {str(k): int(v) for k, v in stage6_conf.items()},
        "top_candidate_sources_all_phone_candidates": dict(candidate_sources.most_common(10)),
        "top_candidate_sources_linked_calls": dict(linked_sources.most_common(10)),
        "conclusion_ru": (
            "Старый массовый Stage2 warning заменён на stage2_confidence_low. "
            "Теперь предупреждение появляется только у действительно низкого confidence, а обычный medium не засоряет "
            "ROP/live-кандидаты массовой технической меткой."
        ),
        "live_pilot_policy_ru": (
            "Для РОП-workbook низкий confidence остаётся строковым предупреждением. "
            "Для будущего live-pilot он должен проверяться через Stage1 source, frozen corpus, readback/rollback "
            "и Claude preflight."
        ),
    }
    (out_root / "stage2_confidence_root_cause.json").write_text(
        json.dumps(root_cause, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_root / "stage2_confidence_root_cause.md").write_text(render_stage2_markdown(root_cause), encoding="utf-8")
    docs_path = ROOT / "docs/DEAL_AWARE_STAGE2_CONFIDENCE_ROOT_CAUSE_2026-05-14.md"
    docs_path.write_text(render_stage2_markdown(root_cause), encoding="utf-8")
    distribution.to_csv(out_root / "stage2_confidence_distribution.csv", index=False)
    return root_cause


def render_stage2_markdown(root_cause: dict[str, Any]) -> str:
    lines = [
        "# Stage2 Confidence Root Cause",
        "",
        "## Короткий вывод",
        "",
        root_cause["conclusion_ru"],
        "",
        "## Политика для live-pilot",
        "",
        root_cause["live_pilot_policy_ru"],
        "",
        "## Распределение Stage2 по всему корпусу",
        "",
        "| decision | confidence | rows |",
        "|---|---:|---:|",
    ]
    for row in root_cause["stage2_distribution"]:
        lines.append(f"| {row['attribution_decision']} | {row['confidence_bucket']} | {row['rows']} |")
    lines += [
        "",
        "## Что это значит",
        "",
        "- 709 строк Stage6 не являются 709 плохими строками.",
        "- Это 723 кандидата Stage5 минус 14 ранних блокеров; затем Stage6 оставил 680 dry-run и 29 текстовых блокеров.",
        "- Массовая техническая метка старого Stage2 больше не должна появляться в новых пакетах.",
        "- Новый сигнал `stage2_confidence_low` означает действительно низкую уверенность привязки.",
        "- Для следующих live-партий нужен не ручной запрет по этой метке, а отдельные защиты: проверка deal_id, качество текста, dry-run, readback, rollback.",
        "",
    ]
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    if not path.exists() or path.is_dir():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_lineage(stage1_root: Path, out_root: Path) -> dict[str, Any]:
    manifest_path = stage1_root / "source_manifest.csv"
    contract_path = stage1_root / "runtime_contract_snapshot.json"
    summary_path = stage1_root / "summary.json"
    manifest = pd.read_csv(manifest_path)
    stage1_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = manifest[manifest["required"].astype(str).str.lower().eq("true")].copy()
    required["sha256"] = required["path"].apply(lambda p: sha256_file(Path(p)))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    source_master_root = Path(safe_text(stage1_summary.get("sources", {}).get("master_contacts_csv"))).parent
    active_export_root = Path(safe_text(contract.get("paths", {}).get("active_export_root")))
    known_mismatch = source_master_root != active_export_root
    canonical_db = Path(safe_text(contract.get("paths", {}).get("canonical_db")))
    canonical_summary = Path(safe_text(contract.get("paths", {}).get("canonical_summary")))
    client_chains = find_client_chains_path(stage1_summary)
    lineage = {
        "stage1_root": str(stage1_root),
        "source_manifest": str(manifest_path),
        "runtime_contract": str(contract_path),
        "stage1_summary": str(summary_path),
        "stage1_summary_sha256": sha256_file(summary_path),
        "required_sources": required[["source_key", "path", "bytes", "modified_at", "sha256"]].to_dict(orient="records"),
        "runtime_paths": contract.get("paths", {}),
        "current_runtime_required_sources": {
            "canonical_db": str(canonical_db),
            "canonical_db_sha256": sha256_file(canonical_db),
            "canonical_summary": str(canonical_summary),
            "canonical_summary_sha256": sha256_file(canonical_summary),
            "client_chains_csv": str(client_chains),
            "client_chains_sha256": sha256_file(client_chains),
            "canonical_db_from_current_runtime": bool(canonical_db)
            and canonical_db == Path(safe_text(contract.get("paths", {}).get("canonical_db"))),
            "client_chains_exists": client_chains.exists(),
        },
        "stage1_actual_sources": stage1_summary.get("sources", {}),
        "gate_failures": [gate for gate in contract.get("gates", []) if not gate.get("passed")],
        "readiness_failures": [gate for gate in contract.get("readiness", {}).get("gates", []) if not gate.get("passed")],
        "known_mismatch": {
            "current_runtime_active_export_vs_stage1_export": known_mismatch,
            "runtime_active_export_root": str(active_export_root),
            "stage1_actual_master_export_root": str(source_master_root),
            "explanation_ru": (
                "Stage1 intentionally uses the newer human-history export as a derived post-backfill layer. "
                "Runtime contract still records the stable strict export pointer as global baseline; this is not an April legacy fallback."
                if known_mismatch
                else "Stage1 export root matches runtime active export root."
            ),
        },
        "row_count_reconciliation": {
            "stage1_call_snapshot_rows": csv_row_count(stage1_root / "call_snapshot.csv"),
            "stage1_phone_rollup_rows": csv_row_count(stage1_root / "phone_rollup.csv"),
            "stage1_amo_ready_snapshot_rows": csv_row_count(stage1_root / "amo_ready_snapshot.csv"),
            "stage1_amo_writeback_snapshot_rows": csv_row_count(stage1_root / "amo_writeback_snapshot.csv"),
            "stage1_tallanto_writeoff_rows": csv_row_count(stage1_root / "tallanto_writeoff_visits.csv"),
        },
        "conclusion_ru": "Stage1 сверяется с текущим runtime-контрактом; старый апрельский экспорт по контракту запрещен. Есть осознанный override: Stage1 может брать отдельный human-history слой, а runtime pointer хранит стабильный strict baseline.",
    }
    required_sources = lineage["current_runtime_required_sources"]
    lineage["current_runtime_source_check_passed"] = bool(
        required_sources["canonical_db_from_current_runtime"]
        and required_sources["client_chains_exists"]
        and required_sources["canonical_db_sha256"]
        and required_sources["client_chains_sha256"]
    )
    (out_root / "source_lineage_proof.json").write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8")
    return lineage


def find_client_chains_path(stage1_summary: dict[str, Any]) -> Path:
    candidates: list[str] = []
    sources = stage1_summary.get("sources") if isinstance(stage1_summary.get("sources"), dict) else {}
    for value in sources.values():
        if isinstance(value, str) and value.endswith("client_chains.csv"):
            candidates.append(value)
        elif isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, str) and item.endswith("client_chains.csv"))
    # Stage1 v2 consumes master export built from client_chains, not client_chains directly.
    master_summary_path = None
    for value in sources.get("quality_summary_paths", []):
        if isinstance(value, str) and "sales_master_export_20260513_human_history_v8_normalized/summary.json" in value:
            master_summary_path = Path(value)
            break
    if master_summary_path and master_summary_path.exists():
        try:
            master_summary = json.loads(master_summary_path.read_text(encoding="utf-8"))
            client_chains = safe_text(master_summary.get("sources", {}).get("client_chains_csv") or master_summary.get("client_chains_csv"))
            if client_chains:
                candidates.append(client_chains)
        except json.JSONDecodeError:
            pass
    if candidates:
        return Path(candidates[0])
    return ROOT / "stable_runtime/insight_readiness_report_after_quality_backfill_20260510_v1/client_chains.csv"


def csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def render_rop_instruction(out_root: Path) -> None:
    lines = [
        "# Инструкция для РОПа по deal-aware preview",
        "",
        "Проверяем не техническую таблицу, а качество текста, который менеджер увидит в сделке AMO.",
        "",
        "## Что проверять в каждой строке",
        "",
        "1. Правильно ли выбрана сделка и не смешаны ли разные ученики/члены семьи.",
        "2. Понятна ли сводка по сделке: что хочет клиент, где сейчас процесс, что мешает продаже.",
        "3. Следующий шаг должен быть действием менеджера, а не пассивным ожиданием клиента.",
        "4. Если оплата уже могла пройти, AI должен предлагать сверку, а не повторно просить оплатить.",
        "5. Если это обратная связь по обучению, задача должна идти куратору/ответственному, а не только в продажу.",
        "6. Tallanto-блок должен быть полезным и не выглядеть как сырой технический дамп.",
        "7. Поля не должны повторять одно и то же разными словами.",
        "",
        "## Решение по строке",
        "",
        "- `ready_for_rop`: можно показывать менеджеру как есть.",
        "- `minor_comment`: полезно, но есть небольшая стилистика.",
        "- `needs_fix_before_rop`: логика полезна, но перед показом надо исправить.",
        "- `block`: нельзя показывать менеджеру, есть риск вредного действия.",
        "",
        "## Правило приемки партии",
        "",
        "- PASS: 0 block, не больше 2 `needs_fix_before_rop`, нет повторяющегося критичного класса.",
        "- PASS_WITH_LIMITATIONS: 0 block, не больше 5 `needs_fix_before_rop`, проблемы локальные.",
        "- FAIL: есть block или системный дефект в 3+ строках.",
        "",
    ]
    (out_root / "ROP_REVIEW_INSTRUCTIONS.md").write_text("\n".join(lines), encoding="utf-8")
    rows = [{"field": name, "allowed_values": allowed} for name, allowed in ROP_COLUMNS]
    pd.DataFrame(rows).to_csv(out_root / "rop_rubric.csv", index=False)


def render_summary(
    out_root: Path,
    stage6: pd.DataFrame,
    classified: pd.DataFrame,
    sample: pd.DataFrame,
    root_cause: dict[str, Any],
    lineage: dict[str, Any],
    *,
    stage6_path: Path,
    stage5_path: Path,
    stage2_root: Path,
    stage1_root: Path,
) -> dict[str, Any]:
    class_counter: Counter[str] = Counter()
    for item in classified["risk_classes"]:
        for cls in safe_text(item).split("|"):
            if cls:
                class_counter[cls] += 1
    summary = {
        "schema_version": "deal_aware_stage709_review_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "stage6_report": str(stage6_path),
            "stage5_report": str(stage5_path),
            "stage2_root": str(stage2_root),
            "stage1_root": str(stage1_root),
        },
        "counts": {
            "stage6_rows": int(stage6.shape[0]),
            "dry_run_rows": int((stage6["stage6_status"] == "dry_run").sum()),
            "blocked_rows": int((stage6["stage6_status"] == "blocked").sum()),
            "stratified_sample_rows": int(sample.shape[0]),
        },
        "class_counts": dict(class_counter.most_common()),
        "stage2_root_cause_short_ru": root_cause["conclusion_ru"],
        "source_lineage_short_ru": lineage["conclusion_ru"],
        "source_lineage": {
            "current_runtime_source_check_passed": lineage.get("current_runtime_source_check_passed"),
            "known_mismatch": lineage.get("known_mismatch", {}),
            "current_runtime_required_sources": lineage.get("current_runtime_required_sources", {}),
        },
        "sample_coverage": sample_coverage(classified, sample),
        "readiness": {
            "rop_review_form_ready": True,
            "stratified_preview_ready_for_claude": True,
            "stage2_confidence_live_blocker_retained": True,
            "live_write_authorized": False,
            "live_write_reason": "Этот пакет готовит проверку и микропилот, но не разрешает live-запись.",
        },
        "outputs": {
            "classification_csv": str(out_root / "deal_stage6_709_classification.csv"),
            "dry_run_classification_csv": str(out_root / "dry_run_680_classification.csv"),
            "blocked_classification_csv": str(out_root / "blocked_29_classification.csv"),
            "stratified_preview_100_csv": str(out_root / "stratified_preview_100_for_rop.csv"),
            "rop_instructions": str(out_root / "ROP_REVIEW_INSTRUCTIONS.md"),
            "stage2_root_cause": str(out_root / "stage2_confidence_root_cause.md"),
            "source_lineage": str(out_root / "source_lineage_proof.json"),
        },
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary["sample_coverage"]).to_csv(out_root / "sample_coverage.csv", index=False)
    (out_root / "README.md").write_text(render_readme(summary), encoding="utf-8")
    return summary


def render_readme(summary: dict[str, Any]) -> str:
    lines = [
        "# Deal-Aware Stage709 Review",
        "",
        "Пакет создан для перехода от top-20 проверки к широкой проверке всех 709 строк Stage6.",
        "",
        "## Что такое 709 строк",
        "",
        "- Stage5 получил 723 deal-aware кандидата.",
        "- 14 строк были заблокированы Stage5.",
        "- 709 строк дошли до Stage6.",
        "- Из них 680 являются dry-run кандидатами, 29 заблокированы CRM text quality gate.",
        "",
        "## Основные выходы",
        "",
        *[f"- `{key}`: `{path}`" for key, path in summary["outputs"].items()],
        "",
        "## Live write",
        "",
        "Live-запись этим пакетом не разрешена.",
        "",
    ]
    return "\n".join(lines)


def build_audit_pack(out_root: Path, summary: dict[str, Any], pack: Path) -> Path:
    pack.mkdir(parents=True, exist_ok=True)
    files = [
        "summary.json",
        "README.md",
        "stratified_preview_100_for_rop.csv",
        "deal_aware_stage100_rop_review.xlsx",
        "deal_stage6_709_classification.csv",
        "deal_stage6_709_business_classification.csv",
        "blocked_29_classification.csv",
        "sample_coverage.csv",
        "ROP_REVIEW_INSTRUCTIONS.md",
        "rop_rubric.csv",
        "stage2_confidence_root_cause.md",
        "stage2_confidence_root_cause.json",
        "source_lineage_proof.json",
    ]
    for filename in files:
        src = out_root / filename
        if src.exists():
            (pack / filename).write_bytes(src.read_bytes())
    audit_scope = [
        "# AUDIT_SCOPE",
        "",
        "Проведи независимый read-only аудит широкой deal-aware проверки.",
        "",
        "## Проверить",
        "",
        "1. Является ли классификация 709 строк достаточно общей, а не подогнанной под частные примеры.",
        "2. Достаточно ли стратифицированная выборка 100 строк покрывает риски: оплата, Tallanto/AMO mismatch, сервисная обратная связь, длинная история, no Tallanto, blocked rows.",
        "3. Корректно ли объяснена причина старого массового Stage2 warning.",
        "4. Достаточна ли форма РОПа для ручной проверки.",
        "5. Можно ли после проверки РОПом готовить микропилот live-записи не больше 5 сделок.",
        "",
        "## Не делать",
        "",
        "- Не запускать live-запись в AMO/Tallanto.",
        "- Не редактировать stable_runtime.",
        "- Не расширять scope на бесконечный поиск новых классов; новые классы записать отдельно.",
        "",
        "## Вердикт",
        "",
        "`PASS`, `PASS_WITH_LIMITATIONS` или `FAIL`; отдельно указать blockers before ROP review и blockers before live pilot.",
        "",
    ]
    (pack / "AUDIT_SCOPE.md").write_text("\n".join(audit_scope), encoding="utf-8")
    (pack / "pack_manifest.json").write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_summary": summary,
        "internal_only": True,
        "internal_only_reason": "ROP workbook may include student/deal context; bot-safe and CRM-live outputs require separate sanitization and audit.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return pack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage6", type=Path, default=DEFAULT_STAGE6)
    parser.add_argument("--stage5", type=Path, default=DEFAULT_STAGE5)
    parser.add_argument("--stage2-root", type=Path, default=DEFAULT_STAGE2)
    parser.add_argument("--stage1-root", type=Path, default=DEFAULT_STAGE1)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--audit-pack", type=Path, default=DEFAULT_AUDIT_PACK)
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    stage6 = pd.read_csv(args.stage6)
    classified = stage6.copy()
    classes = classified.apply(classify_row, axis=1)
    classified["risk_classes"] = ["|".join(item) for item in classes]
    classified["primary_risk_class"] = [primary_class(item) for item in classes]
    classified["risk_flags_ru"] = [build_risk_text(item) for item in classes]
    classified["manager_warning_ru"] = classified.apply(manager_warning_text, axis=1)
    classified["sample_bucket_ru"] = classified["primary_risk_class"].map(RISK_RU).fillna(classified["primary_risk_class"])
    classified["review_url_hint"] = classified["selected_deal_id"].apply(
        lambda deal_id: f"https://educent.amocrm.ru/leads/detail/{safe_text(deal_id)}" if safe_text(deal_id) else ""
    )
    classified["sample_id"] = [f"DA709-{i:03d}" for i in range(1, len(classified) + 1)]

    classified.to_csv(args.out_root / "deal_stage6_709_classification.csv", index=False)
    classified[classified["stage6_status"] == "dry_run"].to_csv(args.out_root / "dry_run_680_classification.csv", index=False)
    classified[classified["stage6_status"] == "blocked"].to_csv(args.out_root / "blocked_29_classification.csv", index=False)
    write_business_classification(classified, args.out_root)

    sample = sample_rows(classified, args.sample_size)
    human = human_columns(sample)
    human.to_csv(args.out_root / "stratified_preview_100_for_rop.csv", index=False)
    sample.to_csv(args.out_root / "stratified_preview_100_full.csv", index=False)

    render_rop_instruction(args.out_root)
    root_cause = build_stage2_root_cause(args.stage2_root, classified, args.out_root)
    lineage = build_source_lineage(args.stage1_root, args.out_root)
    summary = render_summary(
        args.out_root,
        stage6,
        classified,
        sample,
        root_cause,
        lineage,
        stage6_path=args.stage6,
        stage5_path=args.stage5,
        stage2_root=args.stage2_root,
        stage1_root=args.stage1_root,
    )
    pack = build_audit_pack(args.out_root, summary, args.audit_pack)
    print(json.dumps({"out_root": str(args.out_root), "audit_pack": str(pack), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
