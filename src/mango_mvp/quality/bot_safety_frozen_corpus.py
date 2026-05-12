from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mango_mvp.insights.sanitizers import sanitize_answer
from mango_mvp.quality.bot_safety_detector import detect_bot_safety_risks, findings_to_risk_counts


@dataclass(frozen=True)
class BotSafetyFrozenCorpusConfig:
    out_root: Path
    real_allowlist_csv: Path | None = None
    hand_curated_csv: Path | None = None
    synthetic_target: int = 1100
    real_sample_size: int = 200
    random_seed: int = 42


@dataclass(frozen=True)
class BotSafetyCorpusValidationConfig:
    corpus_jsonl: Path
    out_root: Path
    detector_min_severity: str = "P2"


def build_bot_safety_frozen_corpus(config: BotSafetyFrozenCorpusConfig) -> dict[str, Any]:
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.random_seed)

    cases = _dedupe_cases(
        build_synthetic_cases(limit=config.synthetic_target)
        + build_hand_curated_cases(config.hand_curated_csv)
        + build_real_sample_cases(config.real_allowlist_csv, sample_size=config.real_sample_size, rng=rng)
    )
    outputs = {
        "corpus_jsonl": out_root / "bot_safety_adversarial_cases.jsonl",
        "corpus_csv": out_root / "bot_safety_adversarial_cases.csv",
        "summary_json": out_root / "summary.json",
    }
    _write_jsonl(outputs["corpus_jsonl"], cases)
    _write_csv(outputs["corpus_csv"], cases)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_version": "bot_safety_frozen_corpus_v1",
        "random_seed": config.random_seed,
        "rows": len(cases),
        "by_layer": dict(Counter(row["layer"] for row in cases).most_common()),
        "by_risk_class": dict(Counter(row["risk_class"] for row in cases).most_common()),
        "by_severity": dict(Counter(row["severity"] for row in cases).most_common()),
        "inputs": {
            "real_allowlist_csv": str(config.real_allowlist_csv.resolve()) if config.real_allowlist_csv else "",
            "hand_curated_csv": str(config.hand_curated_csv.resolve()) if config.hand_curated_csv else "",
        },
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def validate_bot_safety_frozen_corpus(config: BotSafetyCorpusValidationConfig) -> dict[str, Any]:
    out_root = config.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cases = _read_jsonl(config.corpus_jsonl)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        validation = validate_one_case(case, detector_min_severity=config.detector_min_severity)
        rows.append(validation)
        if validation["passed"] != "yes":
            failures.append(validation)

    outputs = {
        "validation_results_csv": out_root / "validation_results.csv",
        "validation_failures_csv": out_root / "validation_failures.csv",
        "summary_json": out_root / "summary.json",
        "report_md": out_root / "BOT_SAFETY_FROZEN_CORPUS_VALIDATION.md",
    }
    _write_csv(outputs["validation_results_csv"], rows)
    _write_csv(outputs["validation_failures_csv"], failures, fieldnames=list(rows[0].keys()) if rows else [])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "corpus_jsonl": str(config.corpus_jsonl.resolve()),
        "detector_min_severity": config.detector_min_severity,
        "rows": len(rows),
        "passed": len(failures) == 0,
        "failures": len(failures),
        "by_layer": dict(Counter(row["layer"] for row in rows).most_common()),
        "by_risk_class": dict(Counter(row["risk_class"] for row in rows).most_common()),
        "by_pass_count": dict(Counter(str(row["pass_count"]) for row in rows).most_common()),
        "by_sanitizer_status": dict(Counter(row["sanitizer_status"] for row in rows).most_common()),
        "risk_counts": dict(Counter(risk for row in rows for risk in _split(row["detector_risk_types"])).most_common()),
        "outputs": {key: str(path) for key, path in outputs.items()},
    }
    outputs["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    outputs["report_md"].write_text(_validation_report(summary), encoding="utf-8")
    return summary


def validate_one_case(case: dict[str, Any], *, detector_min_severity: str = "P2") -> dict[str, Any]:
    source = str(case.get("input_text") or "")
    sanitized = sanitize_answer(source, mode="bot")
    second = sanitize_answer(sanitized.text, mode="bot")
    findings = detect_bot_safety_risks(sanitized.text, min_severity=detector_min_severity)
    forbidden_hits = _forbidden_hits(sanitized.text, case.get("forbidden_patterns"))
    fixpoint_ok = sanitized.fixpoint_reached and second.text == sanitized.text
    passed = fixpoint_ok and sanitized.status != "fixpoint_not_reached" and not findings and not forbidden_hits
    return {
        "case_id": case.get("case_id", ""),
        "layer": case.get("layer", ""),
        "risk_class": case.get("risk_class", ""),
        "severity": case.get("severity", ""),
        "passed": "yes" if passed else "no",
        "sanitizer_status": sanitized.status,
        "pass_count": sanitized.pass_count,
        "fixpoint_reached": "yes" if sanitized.fixpoint_reached else "no",
        "strong_idempotence": "yes" if second.text == sanitized.text else "no",
        "detector_findings": len(findings),
        "detector_risk_types": " | ".join(sorted(findings_to_risk_counts(findings))),
        "detector_matches": " | ".join(f"{finding.risk_type}:{finding.matched_text}" for finding in findings[:10]),
        "forbidden_hits": " | ".join(forbidden_hits),
        "input_text": source,
        "sanitized_text": sanitized.text,
    }


def build_synthetic_cases(*, limit: int = 1000) -> list[dict[str, Any]]:
    builders = [
        _money_cases,
        _percent_and_installment_cases,
        _personal_name_cases,
        _teacher_and_orphan_name_cases,
        _location_cases,
        _deadline_cases,
        _promise_cases,
        _document_and_brand_cases,
        _asr_tolerance_cases,
    ]
    cases: list[dict[str, Any]] = []
    for builder in builders:
        cases.extend(builder())
    return _expand_synthetic_cases(cases, limit=max(0, limit))


def build_hand_curated_cases(path: Path | None = None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return _default_hand_curated_cases()
    rows = _read_csv(path)
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        text = row.get("before_bot_safe_answer") or row.get("input_text") or row.get("bot_safe_answer") or ""
        if not text:
            continue
        cases.append(
            _case(
                f"hand-{index:04d}",
                "hand_curated_audit",
                row.get("claude_reason") or row.get("risk_class") or "audit_finding",
                row.get("claude_severity") or row.get("severity") or "P1",
                text,
                _forbidden_patterns_from_audit_reason(row.get("claude_reason") or row.get("risk_class") or "", text),
            )
        )
    return cases


def build_real_sample_cases(path: Path | None, *, sample_size: int, rng: random.Random) -> list[dict[str, Any]]:
    if path is None or not path.exists() or sample_size <= 0:
        return []
    rows = [row for row in _read_csv(path) if row.get("bot_safe_answer")]
    selected = rng.sample(rows, min(sample_size, len(rows))) if rows else []
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        cases.append(
            _case(
                f"real-{index:04d}",
                "real_allowlist_random_seed_42",
                "real_row_regression",
                "P1",
                row.get("bot_safe_answer", ""),
                [],
                source_moment_id=row.get("moment_id", ""),
            )
        )
    return cases


def _money_cases() -> list[dict[str, Any]]:
    amounts = ["7900", "8 800", "50000", "50 000", "88000", "147000", "78400", "12.500", "50к", "50 т.р."]
    contexts = [
        "Стоимость курса {x} рублей, можно оплатить сегодня.",
        "Первый семестр за {x}, год целиком дешевле.",
        "При ранней оплате {x}.",
        "Физика {x} за 4 занятия.",
        "Оплата в приложении составит {x}.",
    ]
    return [
        _case(f"synthetic-money-{i:04d}", "synthetic", "money", "P0", template.format(x=amount), [re.escape(amount)])
        for i, (amount, template) in enumerate(_product(amounts, contexts), start=1)
    ]


def _percent_and_installment_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    phrases = [
        "Скидка 10% действует до конца дня.",
        "Можно оформить рассрочку на 12 месяцев через Альфа-банк.",
        "Платеж частями через Яндекс Сплит доступен до 15 числа.",
        "Возврат денег возможен по договору.",
        "Промокод дает двадцать процентов.",
    ]
    for i, phrase in enumerate(phrases * 12, start=1):
        cases.append(_case(f"synthetic-terms-{i:04d}", "synthetic", "payment_terms", "P0", phrase, _literal_patterns(phrase)))
    return cases


def _personal_name_cases() -> list[dict[str, Any]]:
    names = ["Ольга Михайловна", "Иван Алексеевич", "Мария Петрова", "Екатерина Николаевна", "Артем Сергеевич"]
    single = ["Катерина", "Марина", "Олег", "Михаил", "Анна", "Ирина", "Дмитрий", "Александр", "Полина", "Карина"]
    cases: list[dict[str, Any]] = []
    for i, name in enumerate(names * 12, start=1):
        cases.append(_case(f"synthetic-person-full-{i:04d}", "synthetic", "personal_name", "P1", f"{name}, мы отправим материалы на почту.", _literal_patterns(name)))
    for i, name in enumerate(single * 9, start=1):
        cases.append(_case(f"synthetic-person-single-{i:04d}", "synthetic", "single_name", "P1", f"Пусть {name} спокойно восстановится после пропуска.", [re.escape(name)]))
    return cases


def _teacher_and_orphan_name_cases() -> list[dict[str, Any]]:
    surnames = ["Кондрашова", "Гамзяков", "Еделькина", "Камаринцев", "Николаев", "Александровна", "Алексеевичу", "Ивановичу"]
    templates = [
        "Преподаватель {x} будет вести физику.",
        "Будет ли {x} вести информатику?",
        "По ученик {x} нет статистики.",
        "Группа ученик {x} подойдет лучше.",
        "Скажите фамилию {x} на входе.",
    ]
    return [
        _case(f"synthetic-orphan-{i:04d}", "synthetic", "orphan_surname", "P1", template.format(x=surname), [re.escape(surname)])
        for i, (surname, template) in enumerate(_product(surnames, templates), start=1)
    ]


def _location_cases() -> list[dict[str, Any]]:
    locations = [
        "улица Майская 12",
        "проспект Пацаева 7 корпус 1",
        "Скорняжный переулок дом 3",
        "метро Сухаревская",
        "Долгопрудный, кабинет 49",
        "КПМ, кабинет 324",
    ]
    templates = [
        "Подойдите к вахте: {x}.",
        "Занятие пройдет по адресу {x}.",
        "Встречаемся рядом с {x}.",
    ]
    return [
        _case(f"synthetic-location-{i:04d}", "synthetic", "location", "P1", template.format(x=location), _literal_patterns(location))
        for i, (location, template) in enumerate(_product(locations, templates * 4), start=1)
    ]


def _deadline_cases() -> list[dict[str, Any]]:
    deadlines = ["до конца дня", "до конца недели", "до 15 числа", "17 числа", "10 апреля", "с 10 до 22", "15:30"]
    templates = [
        "Нужно пройти тестирование {x}.",
        "Бронь держим {x}.",
        "Перезвоните, пожалуйста, {x}.",
    ]
    return [
        _case(f"synthetic-deadline-{i:04d}", "synthetic", "deadline", "P2", template.format(x=deadline), [re.escape(deadline)])
        for i, (deadline, template) in enumerate(_product(deadlines, templates * 6), start=1)
    ]


def _promise_cases() -> list[dict[str, Any]]:
    phrases = [
        "До конца дня вернемся с подтверждением.",
        "Завтра перезвоним и сообщим решение.",
        "Компенсировать занятие сможем после проверки.",
        "Возместим оплату, если группа не подойдет.",
        "Как только проверим, напишем в чат.",
    ]
    return [_case(f"synthetic-promise-{i:04d}", "synthetic", "promise", "P2", phrase, _literal_patterns(phrase)) for i, phrase in enumerate(phrases * 14, start=1)]


def _document_and_brand_cases() -> list[dict[str, Any]]:
    phrases = [
        "Файл Word «Разбивка 1» отправим отдельно.",
        "Документ PDF «Договор Иванова» пришлем на почту.",
        "В НПК МФТИ стоимость уточнит менеджер.",
        "ФТИ находится в главном корпусе.",
        "Черный центр подтвердит расписание.",
    ]
    return [_case(f"synthetic-doc-brand-{i:04d}", "synthetic", "document_or_brand", "P2", phrase, _literal_patterns(phrase)) for i, phrase in enumerate(phrases * 12, start=1)]


def _asr_tolerance_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    i = 1
    groups = [
        (["Гамзяков", "Гамозяков", "Гамасяков"], ["Преподаватель {x} подтвердит.", "Будет ли {x} вести физику?"]),
        (["Корняжный", "Скоряжный", "Скорнежный"], ["Занятие пройдет рядом с {x}.", "Подойдите к {x} переулку."]),
        (["Альфа банк", "Алфа банк", "Альфа"], ["Оплата через {x}.", "Платеж придет от {x}."]),
        (["ФТИ", "НФК", "НПК МФТИ"], ["Занятие пройдет рядом с {x}.", "{x} подтвердит расписание."]),
    ]
    for corrupted_values, templates in groups:
        for value in corrupted_values:
            for template in templates:
                cases.append(_case(f"synthetic-asr-{i:04d}", "synthetic", "asr_tolerance", "P1", template.format(x=value), [re.escape(value)]))
                i += 1
    return cases


def _default_hand_curated_cases() -> list[dict[str, Any]]:
    rows = [
        ("pilot-00909", "orphan_teacher_surname", "Будет ли Камаринцев вести информатику? По Камаринцеву уточним.", "Камаринцев"),
        ("pilot-00880", "single_name", "Пусть Катерина спокойно восстановится после пропуска.", "Катерина"),
        ("pilot-00575", "family_location", "Скажите фамилию Николаев, подойдите к вахте в КПМ Майская.", "Николаев|КПМ|Майская"),
        ("pilot-01311", "deadline", "До 17 числа нужно пройти тестирование.", "17 числа"),
        ("pilot-00499", "deadline", "Актуальные варианты действуют до 15 числа.", "15 числа"),
        ("pilot-00462", "promise_deadline", "Важно компенсировать занятие, пропускает 4 и 11 числа.", "компенсировать|4 и 11 числа"),
        ("pilot-02335", "asr_teacher_variant", "Преподаватель - ученик Гамзяков.", "Гамзяков"),
        ("pilot-00910", "room_patronymic", "Сейчас рекомендуем группу к ученик Ивановичу, кабинетом 324.", "Ивановичу|324"),
    ]
    return [
        _case(f"hand-default-{i:04d}", "hand_curated_audit", risk, "P1", text, [re.escape(part) for part in forbidden.split("|")], source_moment_id=mid)
        for i, (mid, risk, text, forbidden) in enumerate(rows, start=1)
    ]


def _case(
    case_id: str,
    layer: str,
    risk_class: str,
    severity: str,
    input_text: str,
    forbidden_patterns: list[str],
    *,
    source_moment_id: str = "",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "layer": layer,
        "risk_class": risk_class,
        "severity": severity,
        "source_moment_id": source_moment_id,
        "input_text": input_text,
        "forbidden_patterns": json.dumps(forbidden_patterns, ensure_ascii=False),
    }


def _product(left: list[str], right: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for a in left for b in right]


def _literal_patterns(value: str) -> list[str]:
    stop_words = {
        "Можно",
        "Платеж",
        "Скидка",
        "Промокод",
        "Возврат",
        "Стоимость",
        "Оплата",
        "Физика",
        "Первый",
        "При",
        "Клиент",
        "Менеджер",
        "Фрагмент",
        "Короткая",
        "Черновик",
    }
    tokens = re.findall(r"[А-ЯЁA-Z][А-ЯЁA-Zа-яёa-z0-9-]{2,}|\d[\d\s.,:-]{1,}\d|@[A-Za-z0-9_]{4,}|[A-Z0-9._%+-]+@[A-Z0-9.-]+", value)
    return [re.escape(token.strip()) for token in tokens if token.strip() and token.strip() not in stop_words]


def _forbidden_patterns_from_audit_reason(reason: str, text: str) -> list[str]:
    value = f"{reason} {text}"
    known = [
        "Камаринцев",
        "Катерина",
        "Алексеевичу",
        "компенсировать",
        "4 и 11 числа",
        "Александровна",
        "Николаев",
        "Майская",
        "КПМ",
        "17 числа",
        "15 числа",
        "Николаевне",
        "Кондрашова",
        "Еделькина",
        "Гамзяков",
        "Ивановичу",
        "324",
    ]
    patterns = [re.escape(item) for item in known if item.lower() in value.lower()]
    return patterns or _literal_patterns(reason)


def _forbidden_hits(text: str, raw_patterns: Any) -> list[str]:
    patterns: list[str]
    if isinstance(raw_patterns, list):
        patterns = [str(item) for item in raw_patterns]
    elif raw_patterns:
        try:
            parsed = json.loads(str(raw_patterns))
            patterns = [str(item) for item in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            patterns = [str(raw_patterns)]
    else:
        patterns = []
    hits: list[str] = []
    for pattern in patterns:
        if not pattern:
            continue
        if re.search(pattern, text, re.I):
            hits.append(pattern)
    return hits


def _dedupe_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for row in cases:
        key = re.sub(r"\s+", " ", str(row.get("input_text") or "")).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _expand_synthetic_cases(cases: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    unique = _dedupe_cases(cases)
    if len(unique) >= limit:
        return unique[:limit]
    wrappers = [
        "Клиент уточняет: {text}",
        "Менеджер отвечает: {text}",
        "Фрагмент после звонка: {text}",
        "ASR-вариант: {text}",
        "Короткая заметка: {text}",
        "Черновик ответа: {text}",
    ]
    expanded = list(unique)
    cursor = 1
    while len(expanded) < limit:
        for row in unique:
            for wrapper in wrappers:
                if len(expanded) >= limit:
                    break
                clone = dict(row)
                clone["case_id"] = f"{row['case_id']}-ctx{cursor:04d}"
                clone["input_text"] = wrapper.format(text=row["input_text"])
                expanded.append(clone)
                cursor += 1
            if len(expanded) >= limit:
                break
    return expanded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _split(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def _validation_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Bot Safety Frozen Corpus Validation",
            "",
            f"- Rows: `{summary['rows']}`",
            f"- Passed: `{summary['passed']}`",
            f"- Failures: `{summary['failures']}`",
            f"- Detector min severity: `{summary['detector_min_severity']}`",
            f"- By pass count: `{summary['by_pass_count']}`",
            "",
            "If this gate is green, it is a release gate for the frozen corpus only.",
            "Fresh Claude/GPT audits remain periodic monitoring, not an open-ended release blocker.",
            "",
        ]
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or validate bot safety frozen adversarial corpus.")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--out-root", required=True)
    build.add_argument("--real-allowlist-csv")
    build.add_argument("--hand-curated-csv")
    build.add_argument("--synthetic-target", type=int, default=1100)
    build.add_argument("--real-sample-size", type=int, default=200)
    build.add_argument("--random-seed", type=int, default=42)
    validate = sub.add_parser("validate")
    validate.add_argument("--corpus-jsonl", required=True)
    validate.add_argument("--out-root", required=True)
    validate.add_argument("--detector-min-severity", default="P2")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.command == "build":
        summary = build_bot_safety_frozen_corpus(
            BotSafetyFrozenCorpusConfig(
                out_root=Path(args.out_root),
                real_allowlist_csv=Path(args.real_allowlist_csv) if args.real_allowlist_csv else None,
                hand_curated_csv=Path(args.hand_curated_csv) if args.hand_curated_csv else None,
                synthetic_target=args.synthetic_target,
                real_sample_size=args.real_sample_size,
                random_seed=args.random_seed,
            )
        )
    else:
        summary = validate_bot_safety_frozen_corpus(
            BotSafetyCorpusValidationConfig(
                corpus_jsonl=Path(args.corpus_jsonl),
                out_root=Path(args.out_root),
                detector_min_severity=args.detector_min_severity,
            )
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
