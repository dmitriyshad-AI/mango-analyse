#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_RELEASE_DIR = Path("product_data/knowledge_base/kb_release_20260518_v3_handoff_for_claude_and_team")
SEMANTIC_REVIEW_SCHEMA_VERSION = "kb_semantic_review_v1"

NON_MONEY_PATH_MARKERS = (
    "total_lessons",
    "weekly_lessons",
    "daily_hours",
    "semester_1_weeks",
    "semester_2_weeks",
    "daily_pairs",
    "pair_duration_minutes",
    "duration_weeks",
    "experience_years",
    "retroactive_years",
    "lead_time_days",
    "certificate_lead_time_days",
)
GLOBAL_FORBIDDEN_CLIENT_MARKERS = (
    "source_id",
    "fact_id",
    "freshness_status",
    "AMO",
    "Tallanto",
    "GPT",
    "Claude",
    "Codex",
    "ChatGPT",
    "я бот",
    "я ИИ",
    "нейросеть",
    "искусственный интеллект",
    "раньше сотрудничали",
    "были одно",
    "наш партнёр",
    "наш партнер",
)
FOTON_FORBIDDEN_CLIENT_MARKERS = (
    "УНПК",
    "АНО ДПО",
    "НОУ УНПК",
    "kmipt.ru",
    "@unpk_mipt",
    "@unpkmfti",
)
UNPK_FORBIDDEN_CLIENT_MARKERS = (
    "Фотон",
    "ЦДПО",
    "ЦРДО",
    "cdpofoton.ru",
    "edu@cdpofoton.ru",
    "Т-Банк",
    "Долями",
)
STALE_CERTIFICATE_MARKERS = (
    "3 рабочих дня",
    "3 рабочих дней",
    "тип справки",
    "работа / налоговая / иное",
)
TECHNICAL_ENGLISH_CLIENT_RE = re.compile(
    r"\b(?:prices?|lesson|session|package|base|plus|one\s+block|one\s+subject|two\s+subjects|"
    r"after\s+20\d{2}|before\s+20\d{2}|moscow|dolgoprudny|location|start\s+date|"
    r"online\s+platform|free\s+morning\s+club|factultative)\b",
    re.IGNORECASE,
)
GENERIC_ROP_QUESTIONS = {
    "Можно ли использовать этот факт в ответе клиенту текущего бренда?",
    "Подтверждаете эту цену и область применения для бота?",
}
MONEY_FACT_TYPES = {"price", "discount", "promocode", "installment", "tax", "matkap", "refund"}
NON_MONEY_RUB_FACT_TYPES = {"course_parameter", "deadline", "program", "documents", "contact", "location", "teacher", "policy"}


@dataclass(frozen=True)
class Finding:
    severity: str
    check_id: str
    message: str
    item_id: str = ""
    evidence: str = ""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run semantic sanity review for a KB release folder.")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args(argv)

    report = run_kb_semantic_review(args.release_dir, out_dir=args.out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["semantic_pass"] else 1


def run_kb_semantic_review(release_dir: str | Path, *, out_dir: str | Path | None = None) -> dict[str, Any]:
    release_root = Path(release_dir).expanduser().resolve(strict=False)
    snapshot = load_snapshot(release_root)
    facts = load_facts(release_root, snapshot)
    approval_queue = load_approval_queue(release_root)

    findings: list[Finding] = []
    findings.extend(review_facts(facts))
    findings.extend(review_approval_queue(approval_queue))
    findings.extend(review_snapshot(snapshot, facts=facts))

    counts_by_severity = Counter(finding.severity for finding in findings)
    blocking = [finding for finding in findings if finding.severity in {"P0", "P1"}]
    report = {
        "schema_version": SEMANTIC_REVIEW_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_dir": str(release_root),
        "snapshot_path": str(release_root / "kb_release_v3_snapshot.json"),
        "formal_quality_passed": bool((snapshot.get("quality_summary") or {}).get("quality_passed")),
        "semantic_pass": not blocking,
        "facts_total": len(facts),
        "approval_queue_items": len(approval_queue),
        "findings_total": len(findings),
        "blocking_findings": len(blocking),
        "findings_by_severity": dict(counts_by_severity),
        "findings": [asdict(finding) for finding in findings],
    }

    if out_dir is not None:
        out_root = guard_output_dir(Path(out_dir))
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "semantic_review.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (out_root / "semantic_review.md").write_text(render_markdown(report), encoding="utf-8")

    return report


def review_facts(facts: Sequence[Mapping[str, Any]]) -> list[Finding]:
    findings: list[Finding] = []
    for fact in facts:
        fact_id = str(fact.get("fact_id") or fact.get("fact_key") or "")
        fact_type = str(fact.get("fact_type") or "")
        brand = str(fact.get("brand") or "")
        structured = as_mapping(fact.get("structured_value"))
        text = str(fact.get("client_safe_text") or "")
        fact_key = str(fact.get("fact_key") or "")
        path = str(structured.get("path") or fact_key)
        allowed = is_true(fact.get("allowed_for_client_answer")) and bool(text.strip())

        if allowed and fact_type == "price":
            amount = structured.get("amount")
            if isinstance(amount, (int, float)) and amount < 3000:
                findings.append(
                    Finding(
                        "P0",
                        "implausible_low_client_price",
                        "Клиентская цена выглядит неправдоподобно низкой.",
                        item_id=fact_id,
                        evidence=f"{amount} RUB | {text[:220]}",
                    )
                )
            amount_min = structured.get("amount_min")
            amount_max = structured.get("amount_max")
            if isinstance(amount_min, (int, float)) and amount_min < 3000:
                findings.append(
                    Finding(
                        "P0",
                        "implausible_low_price_range",
                        "Нижняя граница клиентского диапазона цены выглядит неправдоподобно низкой.",
                        item_id=fact_id,
                        evidence=f"{amount_min}-{amount_max} RUB | {text[:220]}",
                    )
                )

        if allowed and fact_type in NON_MONEY_RUB_FACT_TYPES and structured.get("currency") == "RUB":
            findings.append(
                Finding(
                    "P0",
                    "non_money_fact_has_rub",
                    "Недежный факт получил валюту RUB.",
                    item_id=fact_id,
                    evidence=f"{fact_type} | {path} | {text[:220]}",
                )
            )

        if allowed and any(marker in path.casefold() for marker in NON_MONEY_PATH_MARKERS):
            if fact_type == "price" or structured.get("currency") == "RUB":
                findings.append(
                    Finding(
                        "P0",
                        "non_money_path_became_price",
                        "Учебный параметр или срок распознан как цена.",
                        item_id=fact_id,
                        evidence=f"{fact_type} | {path} | {text[:220]}",
                    )
                )

        if ".range.min" in fact_key or ".range.max" in fact_key:
            if not structured.get("do_not_use_as_current_price"):
                findings.append(
                    Finding(
                        "P0",
                        "split_current_price_range",
                        "Текущий диапазон цены разорван на отдельные min/max факты.",
                        item_id=fact_id,
                        evidence=f"{fact_key} | {text[:220]}",
                    )
                )

        if structured.get("amount_min") is not None or structured.get("amount_max") is not None:
            if structured.get("amount_min") is None or structured.get("amount_max") is None:
                findings.append(
                    Finding("P0", "incomplete_price_range", "Диапазон цены содержит только одну границу.", item_id=fact_id)
                )
            elif structured["amount_min"] > structured["amount_max"]:
                findings.append(
                    Finding(
                        "P0",
                        "reversed_price_range",
                        "Нижняя граница диапазона больше верхней.",
                        item_id=fact_id,
                        evidence=f"{structured['amount_min']} > {structured['amount_max']}",
                    )
                )

        if allowed:
            findings.extend(review_client_text(text, brand=brand, item_id=fact_id))

        if allowed and "theme_12_certificate" in fact_key:
            for marker in ("ФИО плательщика", "ФИО ребёнка", "за какой период"):
                if marker.casefold() in text.casefold():
                    findings.append(
                        Finding(
                            "P0",
                            "certificate_unconfirmed_field_request",
                            "Тема справок просит поле, которое не подтверждено текущей политикой.",
                            item_id=fact_id,
                            evidence=marker,
                        )
                    )

        if "online_olympiad_phystech_9_and_11" in fact_key:
            product = str(fact.get("product") or "")
            if product != "online_olympiad_phystech_classes_9_11":
                findings.append(
                    Finding(
                        "P1",
                        "phystech_product_collapsed",
                        "Онлайн Физтех 9/11 схлопнулся с общей олимпиадной подготовкой.",
                        item_id=fact_id,
                        evidence=product,
                    )
                )

    return findings


def review_client_text(text: str, *, brand: str, item_id: str) -> list[Finding]:
    findings: list[Finding] = []
    lowered = text.casefold().replace("ё", "е")

    for marker in GLOBAL_FORBIDDEN_CLIENT_MARKERS:
        if marker.casefold().replace("ё", "е") in lowered:
            findings.append(
                Finding(
                    "P0",
                    "forbidden_client_marker",
                    "Клиентский текст содержит запрещенный служебный или брендовый маркер.",
                    item_id=item_id,
                    evidence=marker,
                )
            )

    for marker in STALE_CERTIFICATE_MARKERS:
        if marker.casefold().replace("ё", "е") in lowered:
            findings.append(
                Finding(
                    "P0",
                    "stale_certificate_phrase",
                    "Клиентский текст содержит старую или запрещенную формулировку по справкам.",
                    item_id=item_id,
                    evidence=marker,
                )
            )

    if re.search(r"\b[a-z]+_[a-z0-9_]+\b", text) or " / " in text:
        findings.append(
            Finding(
                "P1",
                "machine_text_in_client_fact",
                "Клиентский текст выглядит как технический артефакт.",
                item_id=item_id,
                evidence=text[:220],
            )
        )
    if TECHNICAL_ENGLISH_CLIENT_RE.search(text):
        findings.append(
            Finding(
                "P1",
                "technical_english_in_client_fact",
                "Клиентский текст содержит английский технический фрагмент из ключа данных.",
                item_id=item_id,
                evidence=text[:220],
            )
        )

    if brand == "foton":
        for marker in FOTON_FORBIDDEN_CLIENT_MARKERS:
            if marker.casefold().replace("ё", "е") in lowered:
                findings.append(
                    Finding(
                        "P0",
                        "cross_brand_foton_client_text",
                        "Клиентский текст Фотона содержит маркер УНПК.",
                        item_id=item_id,
                        evidence=marker,
                    )
                )
    if brand == "unpk":
        for marker in UNPK_FORBIDDEN_CLIENT_MARKERS:
            if marker.casefold().replace("ё", "е") in lowered:
                findings.append(
                    Finding(
                        "P0",
                        "cross_brand_unpk_client_text",
                        "Клиентский текст УНПК содержит маркер Фотона или условия Фотона.",
                        item_id=item_id,
                        evidence=marker,
                    )
                )

    return findings


def review_approval_queue(rows: Sequence[Mapping[str, Any]]) -> list[Finding]:
    findings: list[Finding] = []
    questions = [str(row.get("rop_question") or "") for row in rows]
    unique_questions = len(set(questions))

    for index, row in enumerate(rows, start=2):
        item_id = str(row.get("approval_item_id") or f"row:{index}")
        priority = str(row.get("priority") or "")
        decision = str(row.get("suggested_decision") or "")
        question = str(row.get("rop_question") or "")
        if priority == "P0" and decision == "keep_internal_only":
            findings.append(
                Finding(
                    "P0",
                    "p0_keep_internal_only",
                    "Внутренний факт не должен быть P0 для РОПа.",
                    item_id=item_id,
                    evidence=question[:220],
                )
            )
        if question in GENERIC_ROP_QUESTIONS:
            findings.append(
                Finding(
                    "P1",
                    "generic_rop_question",
                    "Вопрос РОПу слишком общий и не проверяет конкретный факт.",
                    item_id=item_id,
                    evidence=question,
                )
            )
        if decision == "keep_internal_only" and not keep_internal_question_matches(question):
            findings.append(
                Finding(
                    "P1",
                    "rop_question_mismatch_keep_internal",
                    "Решение keep_internal_only не отражено в вопросе РОПу.",
                    item_id=item_id,
                    evidence=question[:220],
                )
            )

    if len(rows) >= 50 and unique_questions < min(100, max(20, len(rows) // 4)):
        findings.append(
            Finding(
                "P1",
                "low_rop_question_variety",
                "В очереди РОПа слишком мало уникальных вопросов, это похоже на шаблонный дамп.",
                evidence=f"unique={unique_questions}, total={len(rows)}",
            )
        )

    return findings


def keep_internal_question_matches(question: str) -> bool:
    lowered = question.casefold().replace("ё", "е")
    return any(
        marker in lowered
        for marker in (
            "внутрен",
            "только для менеджера",
            "оставляем только",
            "оставить только",
            "не говорит его клиенту",
            "не показывать клиенту",
        )
    )


def review_snapshot(snapshot: Mapping[str, Any], *, facts: Sequence[Mapping[str, Any]]) -> list[Finding]:
    findings: list[Finding] = []
    q15_products = {
        str(fact.get("product") or "")
        for fact in facts
        if "online_olympiad_phystech_9_and_11" in str(fact.get("fact_key") or "")
    }
    general_products = {
        str(fact.get("product") or "")
        for fact in facts
        if str(fact.get("fact_key") or "").startswith("fiztech_olympiad.")
    }
    if q15_products and q15_products != {"online_olympiad_phystech_classes_9_11"}:
        findings.append(
            Finding(
                "P1",
                "q15_product_scope_wrong",
                "Q15 онлайн Физтех 9/11 имеет неверный product scope.",
                evidence=", ".join(sorted(q15_products)),
            )
        )
    if general_products and general_products != {"fiztech_olympiad_general"}:
        findings.append(
            Finding(
                "P1",
                "general_phystech_product_scope_wrong",
                "Общая олимпиадная подготовка Физтех имеет неверный product scope.",
                evidence=", ".join(sorted(general_products)),
            )
        )
    if (snapshot.get("quality_summary") or {}).get("quality_passed") is not True:
        findings.append(
            Finding(
                "P1",
                "formal_quality_not_passed",
                "Формальная quality_summary не пройдена; semantic review не может заменить обычные проверки.",
            )
        )
    return findings


def load_snapshot(release_root: Path) -> Mapping[str, Any]:
    path = release_root / "kb_release_v3_snapshot.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping snapshot at {path}")
    return payload


def load_facts(release_root: Path, snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = [
        release_root / "facts_registry.jsonl",
        release_root.parent / "kb_release_20260518_v3" / "facts_registry.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return [
                item
                for item in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
                if isinstance(item, Mapping)
            ]
    for key in ("facts", "facts_registry"):
        value = snapshot.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    raise FileNotFoundError(f"No facts registry found under {release_root}")


def load_approval_queue(release_root: Path) -> list[Mapping[str, Any]]:
    path = release_root / "approval_queue_for_rop_v3.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def render_markdown(report: Mapping[str, Any]) -> str:
    findings = [item for item in report.get("findings", []) if isinstance(item, Mapping)]
    lines = [
        "# KB Semantic Review",
        "",
        f"- created_at: `{report.get('created_at')}`",
        f"- release_dir: `{report.get('release_dir')}`",
        f"- formal_quality_passed: `{report.get('formal_quality_passed')}`",
        f"- semantic_pass: `{report.get('semantic_pass')}`",
        f"- facts_total: `{report.get('facts_total')}`",
        f"- approval_queue_items: `{report.get('approval_queue_items')}`",
        f"- findings_total: `{report.get('findings_total')}`",
        f"- blocking_findings: `{report.get('blocking_findings')}`",
        f"- findings_by_severity: `{report.get('findings_by_severity')}`",
        "",
    ]
    if not findings:
        lines.extend(["## Findings", "", "Нет блокеров или предупреждений по текущим смысловым правилам."])
        return "\n".join(lines) + "\n"

    lines.extend(["## Findings", ""])
    for item in findings:
        lines.append(
            f"- `{item.get('severity')}` `{item.get('check_id')}` "
            f"{item.get('message')} item=`{item.get('item_id')}` evidence=`{item.get('evidence')}`"
        )
    return "\n".join(lines) + "\n"


def guard_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if "stable_runtime" in resolved.parts:
        raise ValueError("Semantic review output must not be inside stable_runtime")
    return resolved


def as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, Mapping) else {}
    return {}


def is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"true", "1", "yes", "y", "да"}
    return False


if __name__ == "__main__":
    raise SystemExit(main())
