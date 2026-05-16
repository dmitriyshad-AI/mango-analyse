from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CATALOG_ROOT = Path("product_data/question_catalog")
DEFAULT_BUILD_DATE = "2026-05-14"
DEFAULT_ROW_LIMIT = 120

BOT_ACTION_OPTIONS = (
    "готовый шаблон только после утверждения РОПом",
    "после проверки актуального факта и утверждения РОПом",
    "только черновик для менеджера",
    "только менеджер",
    "нельзя утверждать, сначала дробить класс",
)

BLOCKING_SPLIT_CODES = {"wide_class_block_until_split", "thematic_fallback_needs_split"}
EXCLUDED_ANSWER_STATUSES = {"not_customer_question", "not_enough_context"}
EXCLUDED_SUBCLASS_MARKERS = ("короткий обрывок",)

FORBIDDEN_ANSWER_TOKENS = (
    "актуальное окно записи",
    "актуальные варианты",
    "{documents}",
    "{schedule}",
    "{program}",
    "{price}",
)

TOPIC_EXPECTATIONS: tuple[tuple[re.Pattern[str], tuple[str, ...], str], ...] = (
    (re.compile(r"налог|ндфл|вычет", re.I), ("налогового вычета", "документов для налогового"), "tax_document_topic_mismatch"),
    (re.compile(r"адрес|площадк|метро", re.I), ("адрес", "площадк", "онлайн-ссыл"), "location_topic_mismatch"),
    (re.compile(r"личн\w+\s+кабинет|логин|парол|доступ|техническ", re.I), ("доступ", "личный кабинет", "логин", "пароль", "техничес"), "tech_topic_mismatch"),
    (re.compile(r"способ\s+оплат|как\s+оплат|qr|куар|сбп|карт", re.I), ("способ оплаты", "qr", "счет", "квитанц", "оплат"), "payment_method_topic_mismatch"),
    (re.compile(r"сумм\w+\s+к\s+оплат|сколько\s+доплат|задолж|частич|часть курса", re.I), ("сумм", "остат", "платеж", "оплат"), "payment_amount_topic_mismatch"),
    (re.compile(r"статус|ожидани|подтверждени|прош[её]л|зачисл|поступил", re.I), ("статус оплаты", "платеж", "прошла ли оплата"), "payment_status_topic_mismatch"),
    (re.compile(r"тестир|распредел|уров|результат|обратн", re.I), ("тестирован", "результ", "распредел", "групп"), "placement_or_feedback_topic_mismatch"),
    (re.compile(r"справк|\bформа\b|бланк|какие\s+документ|оформлен|ошибк.*документ|распечат|подпис|отправ", re.I), ("документ", "справк", "форма", "бланк", "подпис"), "document_topic_mismatch"),
    (re.compile(r"расписан|день|время|график|даты", re.I), ("расписан", "дни и время", "график"), "schedule_topic_mismatch"),
    (re.compile(r"формат|онлайн|очно|дистанцион", re.I), ("формат", "онлайн", "очный", "смешанный"), "format_topic_mismatch"),
    (re.compile(r"летн|лагер|смен|выездн", re.I), ("летней школе", "смене", "выездной"), "summer_program_topic_mismatch"),
    (re.compile(r"продолж|следующ\w+\s+год|решени\w+\s+сем", re.I), ("продолж", "следующий год", "обучен"), "continuation_topic_mismatch"),
    (re.compile(r"договор|юрид|лиценз|оферт|претенз", re.I), ("договор", "юрид", "документ"), "legal_topic_mismatch"),
    (re.compile(r"программ|предмет|математик|физик|информатик|егэ|огэ|олимпиад|курс", re.I), ("программ", "предмет", "курс"), "program_topic_mismatch"),
)

MAT_CAPITAL_RE = re.compile(
    r"материнск|мат\s*капитал|маткапит|пенсионн\w+\s+фонд|сертификат\w*\W{0,40}(?:материнск|семейн)",
    re.I,
)
REFUND_RE = re.compile(
    r"\b(?:"
    r"возврат\w*|"
    r"перерасч[её]т\w*|"
    r"вернуть\s+(?:деньги|средства|оплату|сумм\w*)|"
    r"верн[её]те\s+(?:деньги|средства|оплату|сумм\w*|эти|нам|мне)|"
    r"(?:деньги|средства|оплат\w+|сумм\w+)\s+верн\w+|"
    r"возмещен\w*"
    r")\b",
    re.I,
)
TAX_REFUND_CONTEXT_RE = re.compile(
    r"(?:"
    r"возврат\w*\W{0,40}(?:ндфл|налогов\w+\s+вычет|вычет)|"
    r"(?:ндфл|налогов\w+\s+вычет|вычет)\W{0,40}возврат\w*"
    r")",
    re.I,
)
LEGAL_RE = re.compile(r"юрид|лиценз|оферт|претенз|договор", re.I)
PAYMENT_RE = re.compile(r"оплат|плат[её]ж|квитанц|чек|сч[её]т|счет|qr|куар", re.I)
TAX_DOC_RE = re.compile(r"налог|ндфл|вычет", re.I)
PRICE_RE = re.compile(r"цен|стоимост|сколько\s+стоит|сумм", re.I)
SCHEDULE_RE = re.compile(r"расписан|день|время|график|когда\s+занят|даты", re.I)
PROGRAM_RE = re.compile(r"программ|предмет|математик|физик|информатик|егэ|огэ|олимпиад|курс", re.I)
TRIAL_RE = re.compile(r"пробн|диагностик|тестов", re.I)
LOCATION_RE = re.compile(r"адрес|площадк|метро|очно|онлайн|город", re.I)
FORMAT_RE = re.compile(r"формат|онлайн|очно|дистанцион", re.I)
SUMMER_PROGRAM_RE = re.compile(r"летн|лагер|смен|выездн", re.I)
DISCOUNT_RE = re.compile(r"скидк|акци|рассроч|частями|льгот", re.I)
TECH_RE = re.compile(r"доступ|ссылк|личн\w+\s+кабинет|не\s+пришл|техническ", re.I)

NOISE_RE = re.compile(
    r"^(?:угу|ага|да|нет|ок|окей|хорошо|спасибо)[\s.!?,]*$|"
    r"начало переадресованного|отправлено с iphone|промокод на следующую покупку|электронная копия чека",
    re.I,
)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def int_value(value: Any) -> int:
    text = safe_text(value)
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", safe_text(value).casefold()).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_pipe_list(value: str) -> list[str]:
    return [part.strip() for part in safe_text(value).split("|") if part.strip()]


def text_for_risk(row: dict[str, str]) -> str:
    return " ".join(
        [
            safe_text(row.get("question_subclass")),
            safe_text(row.get("examples_for_rop")),
            safe_text(row.get("examples_redacted")),
        ]
    )


def narrow_question_text(row: dict[str, str]) -> str:
    return text_for_risk(row)


def class_text(row: dict[str, str]) -> str:
    return " ".join(
        [
            safe_text(row.get("canonical_question")),
            safe_text(row.get("parent_question_class")),
            safe_text(row.get("question_subclass")),
            safe_text(row.get("examples_for_rop")),
            safe_text(row.get("examples_redacted")),
        ]
    )


def refund_match_count(text: str) -> int:
    source = safe_text(text)
    count = 0
    for match in REFUND_RE.finditer(source):
        window = source[max(0, match.start() - 50) : match.end() + 50]
        if TAX_REFUND_CONTEXT_RE.search(window):
            continue
        count += 1
    return count


def match_example_count(row: dict[str, str], pattern: re.Pattern[str]) -> int:
    examples: list[str] = []
    for field in ("examples_for_rop", "examples_redacted"):
        examples.extend(parse_pipe_list(row.get(field, "")))
    seen: set[str] = set()
    count = 0
    for example in examples:
        normalized = normalize_text(example)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if pattern.search(example):
            count += 1
    return count


def matcap_primary_intent(row: dict[str, str]) -> bool:
    subclass = safe_text(row.get("question_subclass"))
    if MAT_CAPITAL_RE.search(subclass):
        return True
    return match_example_count(row, MAT_CAPITAL_RE) >= 2


def matcap_mentioned(row: dict[str, str]) -> bool:
    return matcap_primary_intent(row) or bool(MAT_CAPITAL_RE.search(text_for_risk(row)))


def tax_doc_primary_intent(row: dict[str, str]) -> bool:
    subclass = safe_text(row.get("question_subclass"))
    return bool(TAX_DOC_RE.search(subclass))


def refund_example_count(row: dict[str, str]) -> int:
    examples: list[str] = []
    for field in ("examples_for_rop", "examples_redacted"):
        examples.extend(parse_pipe_list(row.get(field, "")))
    seen: set[str] = set()
    count = 0
    for example in examples:
        normalized = normalize_text(example)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        if refund_match_count(example):
            count += 1
    return count


def refund_primary_intent(row: dict[str, str]) -> bool:
    """True only when refund looks like the main narrow class, not a stray example."""
    subclass = safe_text(row.get("question_subclass"))
    if refund_match_count(subclass):
        return True
    return refund_example_count(row) >= 2


def refund_mentioned(row: dict[str, str]) -> bool:
    return refund_primary_intent(row) or refund_match_count(text_for_risk(row)) > 0


def review_row_has_refund_intent(row: dict[str, str]) -> bool:
    subclass = safe_text(row.get("Узкий класс"))
    if refund_match_count(subclass):
        return True
    examples = parse_pipe_list(row.get("Реальные примеры вопросов", ""))
    return sum(1 for example in examples if refund_match_count(example)) >= 2


def answer_mentions_refund(answer: str) -> bool:
    return bool(re.search(r"\b(?:возврат\w*|перерасч[её]т\w*)\b", safe_text(answer), re.I))


def topic_mismatch_code(parent: str, subclass: str, answer: str) -> str:
    normalized_answer = normalize_text(answer)
    context = f"{parent} {subclass}"
    for subclass_re, answer_markers, code in TOPIC_EXPECTATIONS:
        if code == "payment_status_topic_mismatch" and "оплат" not in normalize_text(context):
            continue
        if subclass_re.search(subclass) and not any(marker in normalized_answer for marker in answer_markers):
            return code
    return ""


def is_noise_text(text: str) -> bool:
    normalized = normalize_text(text)
    return not normalized or bool(NOISE_RE.search(normalized))


def is_excluded_class(row: dict[str, str]) -> bool:
    if safe_text(row.get("answer_status")) in EXCLUDED_ANSWER_STATUSES:
        return True
    subclass = normalize_text(row.get("question_subclass", ""))
    return any(marker in subclass for marker in EXCLUDED_SUBCLASS_MARKERS)


def high_risk_groups(row: dict[str, str]) -> list[str]:
    text = class_text(row)
    groups: list[str] = []
    if matcap_mentioned(row):
        groups.append("материнский капитал")
    if refund_mentioned(row):
        groups.append("возврат / перерасчет")
    if LEGAL_RE.search(text):
        groups.append("договор / юридический вопрос")
    return groups


def load_quality_blockers(catalog_root: Path) -> dict[str, str]:
    path = catalog_root / "answer_quality_check_report.json"
    if not path.exists():
        return {}
    report = load_json(path)
    blockers: dict[str, str] = {}
    for finding in report.get("findings", []):
        class_id = safe_text(finding.get("question_class_id"))
        code = safe_text(finding.get("code"))
        if class_id and code:
            blockers[class_id] = code
    return blockers


def load_priority_rows(catalog_root: Path) -> dict[str, dict[str, str]]:
    path = catalog_root / "rop_review_priority_top100.csv"
    result: dict[str, dict[str, str]] = {}
    if not path.exists():
        return result
    for row in read_csv(path):
        class_id = safe_text(row.get("ID класса"))
        if class_id and class_id not in result:
            result[class_id] = row
        canonical = safe_text(row.get("Класс вопроса"))
        if canonical and canonical not in result:
            result[canonical] = row
    return result


def load_fact_sources(catalog_root: Path) -> dict[str, list[str]]:
    path = catalog_root / "current_fact_source_registry.json"
    if not path.exists():
        return {}
    registry = load_json(path)
    by_type: dict[str, list[str]] = defaultdict(list)
    for source in registry.get("sources", []):
        source_id = safe_text(source.get("source_id"))
        source_path = Path(safe_text(source.get("path"))).name
        status = safe_text(source.get("approval_status")) or "unknown"
        label = f"{source_id}: {source_path} ({status})"
        for fact_type in source.get("fact_types", []):
            by_type[safe_text(fact_type)].append(label)
    return dict(by_type)


def load_items_by_class(catalog_root: Path, class_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    path = catalog_root / "customer_question_items.jsonl"
    items_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            class_id = safe_text(item.get("question_class_id"))
            if class_id in class_ids:
                items_by_class[class_id].append(item)
    return dict(items_by_class)


def priority_score(row: dict[str, str], priority_by_canonical: dict[str, dict[str, str]]) -> int:
    priority = (
        priority_by_canonical.get(safe_text(row.get("question_class_id")))
        or priority_by_canonical.get(safe_text(row.get("canonical_question")))
        or {}
    )
    if priority:
        return int_value(priority.get("Приоритетный балл")) or 1000 - int_value(priority.get("Место"))
    count = int_value(row.get("count_total"))
    bonus = 250 if high_risk_groups(row) else 0
    return count + bonus


def select_review_classes(
    catalog_root: Path,
    *,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> list[dict[str, str]]:
    classes = [row for row in read_csv(catalog_root / "customer_question_classes.csv") if not is_excluded_class(row)]
    priority_by_canonical = load_priority_rows(catalog_root)

    selected: dict[str, dict[str, str]] = {}

    def add(row: dict[str, str]) -> None:
        selected.setdefault(safe_text(row.get("question_class_id")), row)

    for row in sorted(classes, key=lambda item: (-int_value(item.get("count_total")), safe_text(item.get("canonical_question")))):
        if high_risk_groups(row) and int_value(row.get("count_total")) >= 10:
            add(row)

    for row in sorted(classes, key=lambda item: -priority_score(item, priority_by_canonical)):
        if len(selected) >= row_limit:
            break
        add(row)

    return sorted(
        selected.values(),
        key=lambda item: (
            -priority_score(item, priority_by_canonical),
            -int_value(item.get("count_total")),
            safe_text(item.get("canonical_question")),
        ),
    )


def item_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return (
        safe_text(metadata.get("customer_text_for_rop"))
        or safe_text(item.get("customer_text_redacted"))
        or safe_text(item.get("manager_text_redacted"))
    )


def manager_text(item: dict[str, Any]) -> str:
    return safe_text(item.get("manager_text_redacted"))


def sanitize_historical_answer(text: str) -> str:
    cleaned = safe_text(text)
    if not cleaned:
        return "нет ответа в истории"
    replacements = {
        "актуальное окно записи": "[актуальная дата или окно записи]",
        "актуальную стоимость": "[актуальная стоимость]",
        "актуальная стоимость": "[актуальная стоимость]",
        "актуальные варианты": "[актуальные варианты]",
        "действующие правила изменения или отмены услуги": "[актуальные правила изменения или отмены услуги]",
    }
    for needle, replacement in replacements.items():
        cleaned = re.sub(rf"(?<!\[){re.escape(needle)}(?!\])", replacement, cleaned, flags=re.I)
    return cleaned


def historical_answer_for_class(items: list[dict[str, Any]]) -> tuple[str, str]:
    for item in items:
        answer = sanitize_historical_answer(manager_text(item))
        if answer and answer != "нет ответа в истории":
            reason = "исторический ответ нужен только как контекст, РОП должен проверить его заново"
            if normalize_text(answer) in {"нет.", "нет"}:
                reason = "слишком короткий ответ; нельзя утверждать как шаблон"
            elif any(token in normalize_text(answer) for token in FORBIDDEN_ANSWER_TOKENS):
                reason = "в ответе был технический мусор или устаревший placeholder"
            return answer, reason
    return "нет ответа в истории", "нет сохраненного ответа менеджера"


def examples_for_class(row: dict[str, str], items: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    seen: set[str] = set()
    examples: list[str] = []
    for item in sorted(items, key=lambda value: safe_text(value.get("occurred_at"))):
        text = item_text(item)
        normalized = normalize_text(text)
        if not normalized or normalized in seen or is_noise_text(text):
            continue
        seen.add(normalized)
        examples.append(text)
        if len(examples) >= limit:
            return examples
    for text in parse_pipe_list(row.get("examples_for_rop", "")):
        normalized = normalize_text(text)
        if normalized and normalized not in seen and not is_noise_text(text):
            seen.add(normalized)
            examples.append(text)
        if len(examples) >= limit:
            break
    return examples


def infer_fact_keys(row: dict[str, str], groups: Iterable[str]) -> list[str]:
    explicit = [value.split(".")[0] for value in parse_pipe_list(row.get("required_fact_keys", ""))]
    text = narrow_question_text(row)
    inferred = list(dict.fromkeys(explicit))
    if PRICE_RE.search(text) and "price" not in inferred:
        inferred.append("price")
    if SCHEDULE_RE.search(text) and "schedule" not in inferred:
        inferred.append("schedule")
    if LOCATION_RE.search(text) and "location" not in inferred:
        inferred.append("location")
    if DISCOUNT_RE.search(text):
        for key in ("discount", "installment"):
            if key not in inferred:
                inferred.append(key)
    if PROGRAM_RE.search(text) and "program" not in inferred:
        inferred.append("program")
    if TRIAL_RE.search(text) and "trial" not in inferred:
        inferred.append("trial")
    if "материнский капитал" in groups and "documents" not in inferred:
        inferred.append("documents")
    if LEGAL_RE.search(text) and "documents" not in inferred:
        inferred.append("documents")
    if PAYMENT_RE.search(text) and "documents" not in inferred:
        inferred.append("documents")
    return inferred


def fact_source_label(fact_keys: list[str], fact_sources: dict[str, list[str]]) -> str:
    labels: list[str] = []
    for key in fact_keys:
        labels.extend(fact_sources.get(key, [])[:2] or [f"{key}: источник нужно назначить вручную"])
    return " | ".join(dict.fromkeys(labels))


def blocked_answer() -> tuple[str, str, str]:
    return (
        "Этот класс пока нельзя утверждать как ответ. Внутри смешаны разные вопросы, поэтому сначала РОП должен разбить его на узкие классы. Клиенту безопасно ответить только так: «Передам вопрос менеджеру, он уточнит ваш конкретный случай и вернется с точным ответом».",
        "класс заблокирован проверкой качества до дробления",
        "нельзя утверждать, сначала дробить класс",
    )


def mixed_refund_answer() -> tuple[str, str, str]:
    return (
        "Этот класс пока нельзя утверждать как единый ответ. Внутри есть вопрос про возврат или перерасчет, но весь класс не сводится только к нему. Безопасный ответ клиенту: «Передам вопрос менеджеру, он проверит ваш конкретный случай и вернется с точным ответом».",
        "внутри класса смешаны возврат/перерасчет и другие темы; сначала нужно дробление",
        "нельзя утверждать, сначала дробить класс",
    )


def mixed_matcap_answer() -> tuple[str, str, str]:
    return (
        "Этот класс пока нельзя утверждать как единый ответ. Внутри есть вопрос про материнский капитал, но весь класс не сводится только к нему. Безопасный ответ клиенту: «Передам вопрос менеджеру, он проверит ваш конкретный случай и вернется с точным ответом».",
        "внутри класса смешан материнский капитал и другие темы; сначала нужно дробление",
        "нельзя утверждать, сначала дробить класс",
    )


def proposed_answer(row: dict[str, str], *, blocker_code: str, fact_keys: list[str], groups: list[str]) -> tuple[str, str, str, str, str, str]:
    text = narrow_question_text(row)
    subclass = safe_text(row.get("question_subclass"))
    parent = safe_text(row.get("parent_question_class"))

    if blocker_code in BLOCKING_SPLIT_CODES:
        answer, why, action = blocked_answer()
        return answer, why, action, "высокий", "Разбить класс на 3-5 узких вопросов; не утверждать единый ответ.", "да"

    if re.search(r"ручной разбор|развернутый пересказ|без уточненного подкласса", subclass, re.I):
        answer, why, action = blocked_answer()
        return answer, why, action, "высокий", "Разбить широкий или пересказанный класс на узкие вопросы перед утверждением.", "да"

    if matcap_mentioned(row) and not matcap_primary_intent(row):
        answer, why, action = mixed_matcap_answer()
        return answer, why, action, "высокий", "Разбить класс: отдельно материнский капитал, отдельно расписание, оплата, документы или формат.", "да"

    if matcap_primary_intent(row):
        return (
            "Оплату материнским капиталом нужно проверить по вашему случаю: важны [тип сертификата], [регион], [курс/предмет], [ФИО ученика], договор и лицензия образовательной организации. Менеджер проверит условия и подготовит или поправит договор для подачи, если ваш вариант подходит.",
            "маткапитал зависит от типа сертификата, региона, договора и лицензии; нельзя обещать оплату без проверки",
            "только черновик для менеджера",
            "высокий",
            "Проверить регион, тип сертификата, договор, лицензию и формулировку для клиента.",
            "нет",
        )

    if re.search(r"заявлен|отмен|перенос|изменени[ея]\s+услов", subclass, re.I):
        return (
            "Вопрос об отмене, переносе или изменении условий нужно проверять по договору, датам, оплатам и правилам программы. Менеджер проверит [договор], [период обучения], [оплаты] и скажет, какие документы или заявления нужны.",
            "отмена и перенос зависят от договора, дат и правил программы",
            "только менеджер",
            "высокий",
            "Проверить договор, даты, платежи, заявление и допустимую формулировку ответа.",
            "нет",
        )

    if tax_doc_primary_intent(row):
        return (
            "Для справки или документов для налогового вычета нужно проверить [ФИО ученика], [ФИО плательщика], [год оплаты], [договор], [чеки/квитанции] и нужную форму заявления. После проверки менеджер подготовит или отправит правильный комплект документов.",
            "налоговый вычет — это запрос документов, а не возврат оплаты клиенту",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить год, плательщика, ученика, договор, чеки/квитанции и актуальную форму справки.",
            "нет",
        )

    if refund_mentioned(row) and not refund_primary_intent(row):
        answer, why, action = mixed_refund_answer()
        return answer, why, action, "высокий", "Разбить класс: отдельно возврат/перерасчет, отдельно оплата, расписание или документы.", "да"

    if refund_primary_intent(row):
        return (
            "Вопрос по возврату или перерасчету нужно проверить по договору, датам оплаты, посещениям и актуальным правилам. Я передам ситуацию менеджеру: он проверит [договор], [период обучения], [сумму оплаты] и вернется с точным решением без предварительных обещаний.",
            "возвраты и перерасчеты нельзя обещать без проверки документов и правил",
            "только менеджер",
            "высокий",
            "Проверить договор, даты, сумму, посещения и допустимую формулировку ответа.",
            "нет",
        )

    if re.search(r"срок\s+оплат|до\s+какого\s+числа|дедлайн", subclass, re.I):
        return (
            "Срок оплаты нужно проверить по программе и карточке клиента: [курс/предмет], [период], [сумма], [дата оплаты] и условия бронирования места. После проверки менеджер назовет точный срок и безопасный способ оплаты.",
            "срок оплаты зависит от программы, периода, брони и карточки клиента",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить срок оплаты, сумму, бронь места, период и способ оплаты.",
            "нет",
        )

    if re.search(r"статус|ожидани|подтверждени|прош[её]л|зачисл|поступил", subclass, re.I) and ("оплат" in parent or "оплат" in subclass):
        return (
            "Статус оплаты нужно проверить по карточке клиента и платежам: [ФИО ученика], [дата оплаты], [сумма], [способ оплаты] и подтверждение платежа. После проверки менеджер подтвердит, прошла ли оплата, или скажет, что нужно прислать.",
            "клиент спрашивает, дошел ли платеж; нельзя отвечать без сверки оплат",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить платежи, дату, сумму, способ оплаты и карточку клиента.",
            "нет",
        )

    if re.search(r"письм|копи|подтверждени", subclass, re.I):
        return (
            "Нужно проверить, какое письмо, копия или подтверждение требуется клиенту: [документ/ссылка/подтверждение], [ФИО ученика], [курс/группа] и куда отправить. После проверки менеджер отправит нужный файл или подтверждение.",
            "письма и подтверждения зависят от конкретного документа, ссылки или действия",
            "только черновик для менеджера",
            "средний",
            "Проверить тип письма или подтверждения, адрес отправки и карточку клиента.",
            "нет",
        )

    if re.search(r"частич|задолж|период|часть курса|регулярн\w+\s+курс", subclass, re.I) and "оплат" in parent:
        return (
            "Оплату за период, часть курса или задолженность нужно сверить с карточкой клиента: [ФИО ученика], [курс/предмет], [период], уже внесенные платежи и остаток. После проверки менеджер назовет точную сумму и безопасный способ оплаты.",
            "частичная оплата и задолженность зависят от истории платежей клиента",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить период, остаток, внесенные платежи, скидки и способ оплаты.",
            "нет",
        )

    if re.search(r"способ\s+оплат|как\s+оплат|qr|куар|сбп|карт", subclass, re.I):
        return (
            "Способ оплаты нужно подобрать под ваш случай: [курс/предмет], [период оплаты], [сумма] и удобный вариант оплаты. Менеджер проверит, доступна ли оплата по QR, ссылке, счету, квитанции или другому способу, и отправит корректную инструкцию.",
            "клиент спрашивает, как оплатить, а не просит готовый документ",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить доступные способы оплаты, сумму, период и нужный платежный документ.",
            "нет",
        )

    if re.search(r"личн\w+\s+кабинет|логин|парол|доступ|не\s+открыва|техническ", subclass, re.I):
        return (
            "Проверю доступ. Пришлите, пожалуйста, [ФИО ученика], [курс/группа] и что именно не открывается: личный кабинет, логин, пароль, ссылка на занятие, запись или материал. После проверки отправим корректный доступ или подключим технического специалиста.",
            "технический вопрос требует идентификации ученика и конкретной проблемы",
            "только черновик для менеджера",
            "средний",
            "Проверить доступ, группу, ссылку и ответственного за техническую поддержку.",
            "нет",
        )

    if re.search(r"тестир|распредел|уров|результат|обратн", subclass, re.I):
        return (
            "По тестированию, распределению или результатам нужно проверить [ФИО ученика], [класс], [предмет], дату тестирования и текущую группу. После проверки менеджер объяснит результат, группу или следующий шаг по обучению.",
            "результаты и распределение зависят от данных ученика и проверки менеджера",
            "только черновик для менеджера",
            "средний",
            "Проверить ученика, предмет, дату тестирования, группу и комментарий преподавателя или менеджера.",
            "нет",
        )

    if re.search(r"справк|\bформа\b|бланк|какие\s+документ|оформлен|ошибк.*документ|распечат|подпис|отправ", subclass, re.I):
        return (
            "По документам нужно уточнить, какой именно файл нужен: [договор/справка/форма/бланк/согласие], для какой цели, на кого оформлять и куда отправить. После проверки менеджер подготовит документ или даст точную инструкцию по заполнению, подписи и отправке.",
            "документы зависят от цели, типа файла и данных клиента",
            "только черновик для менеджера",
            "средний",
            "Проверить тип документа, цель, данные ученика/плательщика, способ подписи и адрес отправки.",
            "нет",
        )

    if SUMMER_PROGRAM_RE.search(subclass):
        return (
            "По летней школе, смене или выездной программе нужно проверить [даты смены], [формат/локацию], [программу], [стоимость], [свободные места] и [документы для участия]. После проверки можно дать клиенту точные условия и следующий шаг по записи.",
            "летние программы объединяют стоимость, даты, место, документы и наличие мест; нельзя отвечать только ценой",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить актуальные даты, место, стоимость, программу, документы и наличие мест.",
            "нет",
        )

    if FORMAT_RE.search(subclass):
        return (
            "Формат обучения зависит от [курс/предмет], [класс], [группа] и периода. Я проверю, доступен ли онлайн, очный или смешанный формат, и вернусь с точным вариантом: [актуальный формат обучения].",
            "формат нельзя выводить из старых сообщений; нужен актуальный источник по группе или программе",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить актуальный формат обучения для программы, класса и группы.",
            "нет",
        )

    if LOCATION_RE.search(subclass):
        return (
            "Адрес или площадка зависят от [курс/предмет], [группа], [формат] и [период]. Я проверю актуальный адрес или онлайн-ссылку и пришлю точный вариант: [актуальный адрес/ссылка/формат].",
            "адрес и площадка должны браться из актуального расписания или карточки группы",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить актуальный адрес, формат, группу и период.",
            "нет",
        )

    if SCHEDULE_RE.search(subclass):
        return (
            "Расписание зависит от [курс/предмет], [класс], [формат] и группы. Я проверю актуальное расписание и предложу варианты: [актуальные дни и время]. Если нужен конкретный преподаватель или площадка, проверю это отдельно.",
            "расписание нельзя выдумывать; нужен свежий файл или карточка группы",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить актуальное расписание, группу, формат и площадку.",
            "нет",
        )

    if LEGAL_RE.search(subclass) or LEGAL_RE.search(parent):
        return (
            "Вопрос по договору, документам или юридическим условиям нужно проверять по актуальной версии документов. Я передам менеджеру: он проверит [актуальный договор], [курс/предмет], [ФИО ученика] и вернется с точной формулировкой.",
            "договорные и юридические формулировки нельзя утверждать автоматически",
            "только менеджер",
            "высокий",
            "Проверить актуальную версию договора и допустимую формулировку ответа.",
            "нет",
        )

    if PROGRAM_RE.search(subclass):
        return (
            "По программе важно понять [класс], [предмет], [цель обучения] и текущий уровень ученика. Я проверю актуальное описание курса и подберу подходящий вариант: [актуальная программа/группа].",
            "программа зависит от класса, предмета и цели; нужен актуальный учебный контекст",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить программу, класс, предмет, уровень и подходящую группу.",
            "нет",
        )

    if re.search(r"продолж|следующ\w+\s+год|решени\w+\s+сем", subclass, re.I):
        return (
            "По продолжению обучения на следующий период нужно проверить [ФИО ученика], текущую программу, результаты/обратную связь, подходящую группу на следующий год, [стоимость] и условия записи. После проверки менеджер предложит конкретный вариант продолжения и следующий шаг.",
            "продление обучения зависит от текущей истории ученика, группы, программы и актуальных условий",
            "только черновик для менеджера",
            "средний",
            "Проверить текущую программу, результаты, группу на следующий год, стоимость и условия записи.",
            "нет",
        )

    if re.search(r"назначени[еяи]\s+плат[её]ж", text, re.I):
        return (
            "В назначении платежа обычно указывают: «Оплата за обучение по [курс/предмет] за [период оплаты], [ФИО ученика]». Если оплата идет по маткапиталу или по договору, нужно использовать формулировку из [актуальный договор или инструкция по оплате].",
            "клиенту нужна практическая формулировка назначения платежа с подстановкой курса и ФИО ученика",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Подтвердить точную формулировку назначения платежа для разных способов оплаты.",
            "нет",
        )

    if re.search(r"сумм\w+\s+к\s+оплат|сколько\s+доплат|задолженн", subclass, re.I):
        return (
            "Сумму к оплате нужно проверить по карточке клиента: [ФИО ученика], [курс/предмет], [период оплаты], уже внесенные платежи и действующие скидки. После проверки менеджер назовет точную сумму: [сумма к оплате] и отправит подходящий способ оплаты.",
            "клиент спрашивает не общий прайс, а свою конкретную сумму с учетом оплат и скидок",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить карточку клиента, внесенные оплаты, скидки, период и сумму к оплате.",
            "нет",
        )

    if re.search(r"квитанц|чек|сч[её]т|счет|ссылк\w+\s+для\s+оплат|qr|куар", text, re.I):
        return (
            "Да, подготовим и отправим платежный документ для оплаты. Чтобы он был корректным, нужно проверить [ФИО ученика], [курс/предмет], [период оплаты], [сумму] и нужный формат документа: квитанция, счет, чек или ссылка на оплату.",
            "клиент просит платежный документ; нужно не обсуждать возврат, а отправить или проверить документ",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить, какой платежный документ реально нужен и где его брать.",
            "нет",
        )

    if SUMMER_PROGRAM_RE.search(subclass):
        return (
            "По летней школе, смене или выездной программе нужно проверить [даты смены], [формат/локацию], [программу], [стоимость], [свободные места] и [документы для участия]. После проверки можно дать клиенту точные условия и следующий шаг по записи.",
            "летние программы объединяют стоимость, даты, место, документы и наличие мест; нельзя отвечать только ценой",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить актуальные даты, место, стоимость, программу, документы и наличие мест.",
            "нет",
        )

    if FORMAT_RE.search(subclass):
        return (
            "Формат обучения зависит от [курс/предмет], [класс], [группа] и периода. Я проверю, доступен ли онлайн, очный или смешанный формат, и вернусь с точным вариантом: [актуальный формат обучения].",
            "формат нельзя выводить из старых сообщений; нужен актуальный источник по группе или программе",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить актуальный формат обучения для программы, класса и группы.",
            "нет",
        )

    if LOCATION_RE.search(subclass):
        return (
            "Адрес или площадка зависят от [курс/предмет], [группа], [формат] и [период]. Я проверю актуальный адрес или онлайн-ссылку и пришлю точный вариант: [актуальный адрес/ссылка/формат].",
            "адрес и площадка должны браться из актуального расписания или карточки группы",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить актуальный адрес, формат, группу и период.",
            "нет",
        )

    if SCHEDULE_RE.search(subclass):
        return (
            "Расписание зависит от [курс/предмет], [класс], [формат] и группы. Я проверю актуальное расписание и предложу варианты: [актуальные дни и время]. Если нужен конкретный преподаватель или площадка, проверю это отдельно.",
            "расписание нельзя выдумывать; нужен свежий файл или карточка группы",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить актуальное расписание, группу, формат и площадку.",
            "нет",
        )

    if re.search(r"распечат|подпис|отправ", subclass, re.I):
        return (
            "По документу нужно проверить, какой именно файл требуется: [договор/согласие/заявление], кто подписывает, нужен ли скан или оригинал, и куда его отправить. После проверки менеджер даст точную инструкцию: [распечатать/подписать/отправить].",
            "клиенту нужна инструкция по действию с документом, а не общий ответ",
            "только черновик для менеджера",
            "средний",
            "Проверить тип документа, способ подписи, адрес отправки и нужен ли оригинал.",
            "нет",
        )

    if PRICE_RE.search(text):
        return (
            "Стоимость зависит от [курс/предмет], [класс], [формат обучения] и [период]. Я проверю актуальный файл цен и вернусь с точной суммой: [актуальная цена]. Если есть скидка или рассрочка, отдельно проверю [актуальные условия].",
            "цены и условия меняются, поэтому нужен актуальный источник факта",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить текущую цену, скидки, рассрочку и формат обучения.",
            "нет",
        )

    if SCHEDULE_RE.search(text):
        return (
            "Расписание зависит от [курс/предмет], [класс], [формат] и группы. Я проверю актуальное расписание и предложу варианты: [актуальные дни и время]. Если нужен конкретный преподаватель или площадка, проверю это отдельно.",
            "расписание нельзя выдумывать; нужен свежий файл или карточка группы",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить актуальное расписание, группу, формат и площадку.",
            "нет",
        )

    if DISCOUNT_RE.search(text):
        return (
            "Скидка, акция, рассрочка или оплата частями зависят от [программа], [период], [количество предметов] и актуальных правил. Я проверю [актуальные условия скидок/рассрочки] и вернусь с точным вариантом.",
            "финансовые условия требуют актуального правила, иначе высок риск неверного обещания",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить актуальные правила скидок, рассрочки и ограничения.",
            "нет",
        )

    if LEGAL_RE.search(text):
        return (
            "Вопрос по договору, документам или юридическим условиям нужно проверять по актуальной версии документов. Я передам менеджеру: он проверит [актуальный договор], [курс/предмет], [ФИО ученика] и вернется с точной формулировкой.",
            "договорные и юридические формулировки нельзя утверждать автоматически",
            "только менеджер",
            "высокий",
            "Проверить актуальную версию договора и допустимую формулировку ответа.",
            "нет",
        )

    if TECH_RE.search(subclass):
        return (
            "Проверю доступ или ссылку. Пришлите, пожалуйста, [ФИО ученика], [курс/группа] и что именно не открывается: личный кабинет, ссылка на занятие, запись или материал. После проверки отправим корректную ссылку или подключим технического специалиста.",
            "технический вопрос требует идентификации ученика и конкретной проблемы",
            "только черновик для менеджера",
            "средний",
            "Проверить доступ, группу, ссылку и ответственного за техническую поддержку.",
            "нет",
        )

    if LOCATION_RE.search(subclass):
        return (
            "Формат и адрес зависят от [курс/предмет], [группа] и [период]. Я проверю актуальную площадку или онлайн-формат и пришлю точный вариант: [актуальный адрес/ссылка/формат].",
            "адрес и формат нужно брать из актуального расписания или карточки группы",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить актуальный адрес, формат, группу и период.",
            "нет",
        )

    if PROGRAM_RE.search(text):
        return (
            "По программе важно понять [класс], [предмет], [цель обучения] и текущий уровень ученика. Я проверю актуальное описание курса и подберу подходящий вариант: [актуальная программа/группа].",
            "программа зависит от класса, предмета и цели; нужен актуальный учебный контекст",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить программу, класс, предмет, уровень и подходящую группу.",
            "нет",
        )

    if TRIAL_RE.search(text):
        return (
            "Пробное или диагностическое занятие зависит от [курс/предмет], [класс] и свободных мест. Я проверю актуальные условия и предложу ближайший вариант: [актуальная дата/формат пробного занятия].",
            "условия пробного занятия зависят от текущих групп и правил записи",
            "после проверки актуального факта и утверждения РОПом",
            "средний",
            "Проверить правила пробного занятия и доступные слоты.",
            "нет",
        )

    return (
        "Чтобы ответить точно, нужно уточнить [курс/предмет], [класс], [ФИО ученика] и цель обращения. Я передам вопрос менеджеру, он проверит карточку клиента и вернется с конкретным ответом.",
        "в классе недостаточно надежных признаков для готового ответа",
        "только черновик для менеджера",
        "средний",
        "Проверить, можно ли выделить более узкий класс и какой факт нужен для ответа.",
        "нет",
    )


def build_review_rows(catalog_root: Path, *, row_limit: int = DEFAULT_ROW_LIMIT) -> list[dict[str, str]]:
    selected = select_review_classes(catalog_root, row_limit=row_limit)
    quality_blockers = load_quality_blockers(catalog_root)
    fact_sources = load_fact_sources(catalog_root)
    priority_by_canonical = load_priority_rows(catalog_root)
    items_by_class = load_items_by_class(catalog_root, {safe_text(row.get("question_class_id")) for row in selected})

    rows: list[dict[str, str]] = []
    for index, row in enumerate(selected, start=1):
        class_id = safe_text(row.get("question_class_id"))
        groups = high_risk_groups(row)
        blocker_code = quality_blockers.get(class_id, "")
        fact_keys = infer_fact_keys(row, groups)
        answer, why_answer, bot_action, risk, rop_check, multi_topic = proposed_answer(
            row,
            blocker_code=blocker_code,
            fact_keys=fact_keys,
            groups=groups,
        )
        items = items_by_class.get(class_id, [])
        historical, historical_reason = historical_answer_for_class(items)
        priority = priority_by_canonical.get(class_id) or priority_by_canonical.get(safe_text(row.get("canonical_question")), {})
        examples = examples_for_class(row, items)
        source_counts = {
            "звонки": int_value(row.get("count_calls")),
            "telegram": int_value(row.get("count_telegram")),
            "почта": int_value(row.get("count_email")),
        }
        rows.append(
            {
                "Номер": str(index),
                "Место в топ-100": safe_text(priority.get("Место")),
                "Приоритетный балл": str(priority_score(row, priority_by_canonical)),
                "ID класса": class_id,
                "Крупный класс": safe_text(row.get("parent_question_class")),
                "Узкий класс": safe_text(row.get("question_subclass")),
                "Класс вопроса": safe_text(row.get("canonical_question")),
                "Вопрос клиента простым языком": _plain_question(row),
                "Реальные примеры вопросов": " | ".join(examples),
                "Исторический ответ менеджера (не утверждено)": historical,
                "Почему исторический ответ не готов": historical_reason,
                "Предлагаемый ответ": answer,
                "Почему ответ такой": why_answer,
                "Что бот может делать": bot_action,
                "Нужные актуальные факты": " | ".join(fact_keys) or "нет",
                "Источник факта": fact_source_label(fact_keys, fact_sources) if fact_keys else "не требуется",
                "Какие данные подставить": _placeholder_list(answer),
                "Риск ошибки": risk,
                "Что должен проверить РОП": rop_check,
                "Много тем в одном вопросе": multi_topic,
                "Блокер качества": blocker_code,
                "Группы риска": " | ".join(groups) or "нет",
                "Всего вопросов": safe_text(row.get("count_total")),
                "Источники": json.dumps(source_counts, ensure_ascii=False),
                "Решение РОПа": "",
                "Исправленный ответ РОПа": "",
                "Комментарий РОПа": "",
            }
        )
    return rows


def _plain_question(row: dict[str, str]) -> str:
    canonical = safe_text(row.get("canonical_question"))
    if not canonical:
        return "Уточнить вопрос клиента"
    return f"Клиент спрашивает: {canonical}."


def _placeholder_list(answer: str) -> str:
    placeholders = re.findall(r"\[[^\]]+\]", answer)
    return " | ".join(dict.fromkeys(placeholders)) or "нет"


def audit_review_rows(rows: list[dict[str, str]], *, min_rows: int = 100) -> dict[str, Any]:
    findings: list[dict[str, str]] = []

    def add(severity: str, code: str, row: dict[str, str] | None = None, evidence: str = "") -> None:
        findings.append(
            {
                "severity": severity,
                "code": code,
                "row_number": safe_text(row.get("Номер")) if row else "",
                "question_class_id": safe_text(row.get("ID класса")) if row else "",
                "evidence": evidence,
            }
        )

    if len(rows) < min_rows:
        add("p0", "not_enough_rows", evidence=f"rows={len(rows)} min_rows={min_rows}")

    seen_classes: set[str] = set()
    for row in rows:
        class_id = row["ID класса"]
        if class_id in seen_classes:
            add("p1", "duplicate_question_class", row, class_id)
        seen_classes.add(class_id)

        answer = row["Предлагаемый ответ"]
        answer_norm = normalize_text(answer)
        class_and_examples = " ".join([row["Класс вопроса"], row["Узкий класс"], row["Реальные примеры вопросов"]])
        risk_text = " ".join([row["Узкий класс"], row["Реальные примеры вопросов"]])

        if not answer:
            add("p0", "empty_proposed_answer", row)
        for token in FORBIDDEN_ANSWER_TOKENS:
            if token in answer_norm:
                add("p0", "forbidden_token_in_proposed_answer", row, token)
        if MAT_CAPITAL_RE.search(class_and_examples) and re.match(r"^\s*нет\b", answer, re.I):
            add("p0", "mat_capital_short_no", row, answer[:80])
        if (
            answer_mentions_refund(answer)
            and row["Что бот может делать"] != "нельзя утверждать, сначала дробить класс"
            and not review_row_has_refund_intent(row)
        ):
            add("p0", "refund_token_without_refund_intent", row, answer[:120])
        if row["Что бот может делать"] not in BOT_ACTION_OPTIONS:
            add("p1", "bot_action_enum_violation", row, row["Что бот может делать"])
        if row["Блокер качества"] in BLOCKING_SPLIT_CODES and row["Что бот может делать"] != "нельзя утверждать, сначала дробить класс":
            add("p1", "blocked_class_marked_ready", row, row["Блокер качества"])
        if row["Что бот может делать"] != "нельзя утверждать, сначала дробить класс":
            mismatch_code = topic_mismatch_code(safe_text(row.get("Крупный класс")), row["Узкий класс"], answer)
            if mismatch_code:
                add("p1", mismatch_code, row, answer[:120])
        if (
            row["Нужные актуальные факты"] != "нет"
            and row["Что бот может делать"] != "нельзя утверждать, сначала дробить класс"
            and "[" not in answer
        ):
            add("p1", "placeholder_missing_for_fact_dependent_class", row, row["Нужные актуальные факты"])
        if row["Нужные актуальные факты"] != "нет" and not row["Источник факта"]:
            add("p1", "fact_source_unpinned", row, row["Нужные актуальные факты"])
        if "Исторический ответ менеджера" not in " ".join(row.keys()):
            add("p1", "historical_answer_field_missing", row)
        sentence_count = max(1, len(re.findall(r"[.!?](?:\s|$)", answer)))
        if sentence_count > 6:
            add("p2", "answer_too_long", row, f"sentences={sentence_count}")

    source_totals = Counter()
    for row in rows:
        try:
            source_totals.update(json.loads(row["Источники"]))
        except json.JSONDecodeError:
            add("p2", "source_breakdown_not_json", row, row["Источники"])
    if source_totals.get("звонки", 0) == 0 or source_totals.get("telegram", 0) == 0:
        add("p2", "weak_channel_coverage", evidence=json.dumps(source_totals, ensure_ascii=False))

    counts = Counter(item["severity"] for item in findings)
    verdict = "pass" if counts.get("p0", 0) == 0 and counts.get("p1", 0) == 0 else "blocked"
    return {
        "schema_version": "question_answer_quality_review_audit_v1",
        "verdict": verdict,
        "row_count": len(rows),
        "findings_by_severity": dict(sorted(counts.items())),
        "findings": findings,
        "bot_action_options": list(BOT_ACTION_OPTIONS),
    }


def build_pack(
    catalog_root: Path,
    output_csv: Path,
    output_summary: Path,
    *,
    row_limit: int = DEFAULT_ROW_LIMIT,
    iteration: str,
) -> dict[str, Any]:
    rows = build_review_rows(catalog_root, row_limit=row_limit)
    audit = audit_review_rows(rows, min_rows=min(100, row_limit))
    write_csv(output_csv, rows)
    high_risk_counts = Counter()
    for row in rows:
        for group in parse_pipe_list(row["Группы риска"]):
            if group != "нет":
                high_risk_counts[group] += 1
    summary = {
        "schema_version": "question_answer_quality_review_pack_v1",
        "build_date": DEFAULT_BUILD_DATE,
        "iteration": iteration,
        "catalog_root": str(catalog_root),
        "outputs": {"csv": str(output_csv), "summary_json": str(output_summary)},
        "totals": {
            "rows": len(rows),
            "high_risk_groups": dict(high_risk_counts),
            "bot_actions": dict(Counter(row["Что бот может делать"] for row in rows)),
        },
        "audit": audit,
    }
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
