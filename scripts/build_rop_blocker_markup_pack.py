#!/usr/bin/env python3
"""Build a ROP markup pack for broad question classes that still block approval."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CATALOG_ROOT = Path("product_data/question_catalog")
DEFAULT_BUILD_DATE = "2026-05-13"
DEFAULT_LIMIT_PER_CLASS = 80

DECISION_OPTIONS = (
    "утвердить существующий класс",
    "перенести в новый узкий класс",
    "много тем в одном вопросе",
    "недостаточно контекста",
    "служебный шум",
    "только менеджер",
    "нужен актуальный факт",
    "исключить из базы",
)

BOT_PERMISSION_OPTIONS = (
    "можно отвечать готовым шаблоном без правок",
    "можно отвечать после проверки факта",
    "только черновик для менеджера",
    "только менеджер",
    "нельзя отвечать",
)

FACT_OPTIONS = (
    "нет",
    "цена",
    "расписание",
    "документы",
    "скидка",
    "оплата",
    "договор",
    "юридическое решение",
    "другое",
)

NOISE_PATTERNS = (
    "начало переадресованного",
    "отправлено с iphone",
    "пересланное сообщение",
    "image: картинка подарка",
    "электронная копия чека",
    "промокод на следующую покупку",
    "ваш подарок за покупку",
)

PURE_ACK_RE = re.compile(r"^(?:угу|ага|да|нет|ок|окей|хорошо|понял[аи]?|спасибо|добрый день|здравствуйте)[\s.!?,]*$", re.I)
QUESTION_MARKER_RE = re.compile(
    r"\?|(?:как|что|где|когда|какие|какой|какая|куда|зачем|почему|можно|нуж(?:но|ен|на|ны)|надо|подскажите|уточните|пришлите|отправьте|есть ли|возможно ли)\b",
    re.I,
)
SANITIZER_PLACEHOLDER_REPLACEMENTS = (
    ("актуальное окно записи", "[актуальная дата или окно записи]"),
    ("актуальную стоимость", "[актуальная стоимость]"),
    ("актуальная стоимость", "[актуальная стоимость]"),
    ("актуальные варианты", "[актуальные варианты]"),
    ("актуальный адрес", "[актуальный адрес]"),
    ("актуальное расписание", "[актуальное расписание]"),
    ("действующие правила изменения или отмены услуги", "[актуальные правила изменения или отмены услуги]"),
)


@dataclass(frozen=True)
class BlockedClass:
    question_class_id: str
    canonical_question: str
    parent_question_class: str
    question_subclass: str
    count_total: int
    count_calls: int
    count_telegram: int
    count_email: int
    blocker_code: str
    blocker_reason: str
    top100_place: str = ""


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _int_value(value: Any) -> int:
    text = _safe_text(value)
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _top100_blocked_names(top100_path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in _read_csv(top100_path):
        approval = _safe_text(row.get("Можно ли утверждать сейчас"))
        if not approval.startswith("нет"):
            continue
        class_id = _safe_text(row.get("ID класса"))
        canonical = _safe_text(row.get("Класс вопроса"))
        if not canonical and not class_id:
            continue
        payload = {
            "top100_place": _safe_text(row.get("Место")),
            "blocker_reason": _safe_text(row.get("Причина блокировки утверждения")),
        }
        if class_id:
            result[class_id] = payload
        if canonical:
            result[canonical] = payload
    return result


def collect_blocked_classes(catalog_root: Path, include_all_quality_blockers: bool = False) -> list[BlockedClass]:
    classes_path = catalog_root / "customer_question_classes.csv"
    top100_path = catalog_root / "rop_review_priority_top100.csv"
    quality_path = catalog_root / "answer_quality_check_report.json"

    classes_by_id = {_safe_text(row.get("question_class_id")): row for row in _read_csv(classes_path)}
    top100_blocked = _top100_blocked_names(top100_path)
    quality = _load_json(quality_path)

    selected_ids: set[str] = set()
    reason_by_id: dict[str, tuple[str, str]] = {}
    for finding in quality.get("findings", []):
        if _safe_text(finding.get("severity")).lower() != "p1":
            continue
        class_id = _safe_text(finding.get("question_class_id"))
        canonical = _safe_text(finding.get("canonical_question"))
        top100_match = top100_blocked.get(class_id) or top100_blocked.get(canonical) or {}
        if include_all_quality_blockers or top100_match:
            selected_ids.add(class_id)
            reason_by_id[class_id] = (
                _safe_text(finding.get("code")),
                top100_match.get("blocker_reason") or _safe_text(finding.get("code")),
            )

    blocked: list[BlockedClass] = []
    for class_id in selected_ids:
        row = classes_by_id.get(class_id)
        if not row:
            continue
        canonical = _safe_text(row.get("canonical_question"))
        code, reason = reason_by_id.get(class_id, ("", ""))
        blocked.append(
            BlockedClass(
                question_class_id=class_id,
                canonical_question=canonical,
                parent_question_class=_safe_text(row.get("parent_question_class")),
                question_subclass=_safe_text(row.get("question_subclass")),
                count_total=_int_value(row.get("count_total")),
                count_calls=_int_value(row.get("count_calls")),
                count_telegram=_int_value(row.get("count_telegram")),
                count_email=_int_value(row.get("count_email")),
                blocker_code=code,
                blocker_reason=reason,
                top100_place=(top100_blocked.get(class_id) or top100_blocked.get(canonical) or {}).get("top100_place", ""),
            )
        )

    return sorted(
        blocked,
        key=lambda item: (
            0 if item.top100_place else 1,
            _int_value(item.top100_place) if item.top100_place else 9999,
            -item.count_total,
            item.canonical_question,
        ),
    )


def _load_items_by_class(items_path: Path, class_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    items_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with items_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            class_id = _safe_text(item.get("question_class_id"))
            if class_id in class_ids:
                items_by_class[class_id].append(item)
    return dict(items_by_class)


def is_noise_example(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return True
    if any(marker in normalized for marker in NOISE_PATTERNS):
        return True
    if PURE_ACK_RE.fullmatch(normalized):
        return True
    if len(normalized) < 25 and not QUESTION_MARKER_RE.search(normalized):
        return True
    return False


def sanitize_manager_draft(text: str) -> str:
    cleaned = _safe_text(text)
    if not cleaned:
        return "нет ответа в истории"
    for needle, replacement in SANITIZER_PLACEHOLDER_REPLACEMENTS:
        cleaned = re.sub(rf"(?<!\[){re.escape(needle)}(?!\])", replacement, cleaned, flags=re.I)
    return cleaned


def suggest_bucket(text: str, canonical_question: str = "") -> tuple[str, str]:
    normalized = _normalize_text(text)
    canonical_normalized = _normalize_text(canonical_question)
    markers = {
        "оплата": ("оплат", "счет", "квитанц", "чек", "возврат", "деньг", "стоимост"),
        "договор": ("договор", "оферт", "подпис", "анкета"),
        "документы": ("документ", "справк", "письм", "подтвержд", "заявлен"),
        "скидка": ("скидк", "акци", "промокод", "льгот"),
        "расписание": ("расписан", "день", "время", "график", "смен"),
        "юридическое": ("юрид", "лиценз", "ип", "ооо", "возражен", "претенз"),
    }
    matched = [name for name, words in markers.items() if any(word in normalized for word in words)]

    if len(text.strip()) < 18 or re.fullmatch(r"[\W\d_а-яёА-ЯЁ]{0,20}", text.strip()):
        return "недостаточно контекста", "текст слишком короткий или похож на обрывок"
    if is_noise_example(text):
        return "служебная пересылка / технический текст", "похож на служебную часть письма или пересылки"
    if re.search(r"\b(?:материнск\w*|мат\s*капитал\w*|маткапитал\w*|маткапит\w*|региональн\w+\s+мат)\b", normalized):
        if "договор" in normalized:
            return (
                "материнский капитал: договор для оплаты обучения",
                "вопрос про договор/статус договора для оплаты материнским капиталом",
            )
        if "региональ" in normalized:
            return (
                "материнский капитал: статус оплаты региональным маткапиталом",
                "вопрос про региональный маткапитал; нельзя отвечать «нет» без проверки региона и документов",
            )
        return (
            "материнский капитал: оплата обучения",
            "вопрос про оплату обучения материнским капиталом",
        )
    if re.search(r"\bназначени[еяи]\s+плат[её]ж", normalized):
        return (
            "оплата: назначение платежа",
            "нужно подставить курс/предмет и ФИО ученика",
        )
    if "квитанц" in normalized and "оплат" in normalized:
        return (
            "оплата: отправить квитанцию для оплаты",
            "клиент просит квитанцию для оплаты; нужно отправить документ",
        )
    if re.search(r"\b(?:ссылк\w+\s+для\s+оплат|сч[её]т\s+на\s+оплат|счет\s+на\s+оплат|qr|куар)", normalized):
        return "оплата: счет или ссылка для оплаты", "клиент просит платежный документ или ссылку"
    if re.search(r"\b(?:чек|квитанц)\b", normalized):
        return "оплата: чек или квитанция", "есть маркеры чека/квитанции"
    if len(set(matched)) >= 2:
        return "много тем в одном вопросе", "в одном примере найдено несколько тем: " + ", ".join(matched)
    if re.search(r"\b(?:возврат|верн[её]те|вернуть\s+(?:деньги|средства)|возмести|возмещен)", normalized):
        return "оплата: возврат денег", "есть маркеры возврата"
    if "счет" in normalized and "оплат" in normalized:
        return "оплата: счет на оплату", "есть маркеры счета и оплаты"
    if "рассроч" in normalized or "частями" in normalized:
        return "оплата: рассрочка или оплата частями", "есть маркеры рассрочки/частичной оплаты"
    if "договор" in normalized:
        return "договор: отправка, подписание или правки", "есть маркер договора"
    if "справк" in normalized:
        return "документы: справка или подтверждение", "есть маркер справки/подтверждения"
    if "письм" in normalized or "почт" in normalized:
        return "письма: отправка или получение письма", "есть маркеры письма/почты"
    if "юрид" in normalized or "лиценз" in normalized:
        return "юридический вопрос: требуется менеджер", "есть юридические маркеры"
    if "скидк" in normalized or "акци" in normalized:
        return "скидка: условия или размер скидки", "есть маркеры скидки"
    if "оплат" in normalized:
        return "оплата: общий вопрос, требуется уточнение", "есть маркер оплаты, но нет точного подкласса"
    if "документ" in normalized:
        return "документы: общий вопрос, требуется уточнение", "есть маркер документов, но нет точного подкласса"
    if canonical_normalized and any(word in canonical_normalized for word in ("оплата", "документы", "договор", "юридические")):
        return "ручная классификация РОПом", "в тексте клиента нет надежных маркеров; название класса не использовалось для автоподсказки"
    return "ручная классификация РОПом", "авто-подсказка не уверена"


def _item_text_for_rop(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return (
        _safe_text(metadata.get("customer_text_for_rop"))
        or _safe_text(item.get("customer_text_redacted"))
        or _safe_text(item.get("manager_text_redacted"))
    )


def build_markup_rows(
    blocked_classes: Iterable[BlockedClass],
    items_by_class: dict[str, list[dict[str, Any]]],
    limit_per_class: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    row_number = 1
    for blocked in blocked_classes:
        seen_texts: set[str] = set()
        selected = 0
        source_counter: Counter[str] = Counter()
        sorted_items = sorted(
            items_by_class.get(blocked.question_class_id, []),
            key=lambda item: (
                _safe_text(item.get("source_channel")) != "telegram",
                _safe_text(item.get("source_channel")) != "email",
                _safe_text(item.get("occurred_at")),
                _safe_text(item.get("question_item_id")),
            ),
        )
        for item in sorted_items:
            text = _item_text_for_rop(item)
            normalized = _normalize_text(text)
            if not normalized or normalized in seen_texts:
                continue
            if is_noise_example(text):
                continue
            if selected >= limit_per_class:
                break
            seen_texts.add(normalized)
            source_counter.update([_safe_text(item.get("source_channel")) or "unknown"])
            suggested_bucket, suggested_reason = suggest_bucket(text, blocked.canonical_question)
            rows.append(
                {
                    "Номер": str(row_number),
                    "Место в топ-100": blocked.top100_place,
                    "ID класса": blocked.question_class_id,
                    "Текущий класс-блокер": blocked.canonical_question,
                    "Крупный класс": blocked.parent_question_class,
                    "Текущий подкласс": blocked.question_subclass,
                    "Причина блокировки": blocked.blocker_reason,
                    "Код блокировки": blocked.blocker_code,
                    "Всего вопросов в классе": str(blocked.count_total),
                    "Звонки в классе": str(blocked.count_calls),
                    "Telegram в классе": str(blocked.count_telegram),
                    "Почта в классе": str(blocked.count_email),
                    "Источник примера": _safe_text(item.get("source_channel")),
                    "Дата примера": _safe_text(item.get("occurred_at")),
                    "Пример вопроса клиента": text,
                    "Исторический ответ менеджера (не утверждено)": sanitize_manager_draft(_safe_text(item.get("manager_text_redacted"))),
                    "Авто-подсказка узкого класса": suggested_bucket,
                    "Почему такая подсказка": suggested_reason,
                    "Решение РОПа": "",
                    "Новый узкий класс": "",
                    "Новый узкий класс 2, если в вопросе несколько тем": "",
                    "Новый узкий класс 3, если нужен": "",
                    "Можно ли боту отвечать": "",
                    "Нужен актуальный факт": "",
                    "Идеальный ответ или шаблон РОПа": "",
                    "Комментарий РОПа": "",
                }
            )
            row_number += 1
            selected += 1
        blocked_summary = {
            "Номер": str(row_number),
            "Место в топ-100": blocked.top100_place,
            "ID класса": blocked.question_class_id,
            "Текущий класс-блокер": blocked.canonical_question,
            "Крупный класс": blocked.parent_question_class,
            "Текущий подкласс": blocked.question_subclass,
            "Причина блокировки": blocked.blocker_reason,
            "Код блокировки": blocked.blocker_code,
            "Всего вопросов в классе": str(blocked.count_total),
            "Звонки в классе": str(blocked.count_calls),
            "Telegram в классе": str(blocked.count_telegram),
            "Почта в классе": str(blocked.count_email),
            "Источник примера": "ИТОГО ПО КЛАССУ",
            "Дата примера": "",
            "Пример вопроса клиента": f"Выше выбрано примеров: {selected}; источники: {dict(source_counter)}",
            "Исторический ответ менеджера (не утверждено)": "",
            "Авто-подсказка узкого класса": "",
            "Почему такая подсказка": "",
            "Решение РОПа": "",
            "Новый узкий класс": "",
            "Новый узкий класс 2, если в вопросе несколько тем": "",
            "Новый узкий класс 3, если нужен": "",
            "Можно ли боту отвечать": "",
            "Нужен актуальный факт": "",
            "Идеальный ответ или шаблон РОПа": "",
            "Комментарий РОПа": "",
        }
        rows.append(blocked_summary)
        row_number += 1
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    path: Path,
    blocked_classes: list[BlockedClass],
    rows: list[dict[str, str]],
    *,
    catalog_root: Path,
    limit_per_class: int,
) -> None:
    examples_by_class = Counter(row["ID класса"] for row in rows if row.get("Источник примера") != "ИТОГО ПО КЛАССУ")
    summary = {
        "schema_version": "rop_blocker_markup_pack_v1",
        "build_date": DEFAULT_BUILD_DATE,
        "catalog_root": str(catalog_root),
        "limit_per_class": limit_per_class,
        "blocked_classes": [
            {
                "question_class_id": item.question_class_id,
                "canonical_question": item.canonical_question,
                "top100_place": item.top100_place,
                "blocker_code": item.blocker_code,
                "count_total": item.count_total,
                "selected_examples": examples_by_class[item.question_class_id],
            }
            for item in blocked_classes
        ],
        "totals": {
            "blocked_classes": len(blocked_classes),
            "markup_rows": len(rows),
            "example_rows": sum(examples_by_class.values()),
        },
        "decision_options": list(DECISION_OPTIONS),
        "bot_permission_options": list(BOT_PERMISSION_OPTIONS),
        "fact_options": list(FACT_OPTIONS),
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def write_guide(path: Path, csv_path: Path, summary_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Инструкция для разметки широких классов вопросов РОПом

Дата: {DEFAULT_BUILD_DATE}

## Зачем нужен файл

Текущий каталог вопросов уже собран из звонков, Telegram и почты, но несколько крупных классов остаются слишком широкими. Если сейчас дать боту такие классы без проверки, он будет иногда отвечать не на тот вопрос или смешивать оплату, договор, документы и возврат.

Основной файл для работы: `{csv_path}`

Техническая сводка: `{summary_path}`

## Кто должен размечать

Финальную разметку должен подтвердить РОП или назначенный им сильный старший менеджер. Codex может предложить подсказки и сгруппировать примеры, но не должен сам утверждать коммерческие правила, юридические ответы, возвраты, скидки, договоры и точные формулировки для клиентов.

## Что заполнить в каждой строке

1. `Решение РОПа` - выбрать одно из значений:
   - утвердить существующий класс;
   - перенести в новый узкий класс;
   - много тем в одном вопросе;
   - недостаточно контекста;
   - служебный шум;
   - только менеджер;
   - нужен актуальный факт;
   - исключить из базы.
2. `Новый узкий класс` - коротко назвать точный смысл вопроса, например `возврат денег после оплаты`, `чек или квитанция`, `договор: где подписать`.
3. Если в одной строке реально несколько тем, заполнить `Новый узкий класс 2, если в вопросе несколько тем` и при необходимости `Новый узкий класс 3, если нужен`.
4. `Можно ли боту отвечать` - указать один из вариантов:
   - можно отвечать готовым шаблоном без правок;
   - можно отвечать после проверки факта;
   - только черновик для менеджера;
   - только менеджер;
   - нельзя отвечать.
5. `Нужен актуальный факт` - указать, нужен ли свежий факт из файла цен, расписания, документов, скидок, оплаты или договора.
6. `Идеальный ответ или шаблон РОПа` - написать ответ, который можно будет превратить в утвержденный шаблон. Если нужны актуальные данные, лучше писать не число вручную, а placeholder: `[актуальная цена из файла]`, `[актуальная дата старта]`, `[актуальный адрес]`.
7. `Комментарий РОПа` - любые уточнения: что проверить, кому передавать, какой риск есть.

## Как читать технические заглушки

`[CLIENT_NAME]`, `[телефон скрыт]`, `[email скрыт]` - это не текст для клиента, а скрытые персональные данные. Их нельзя копировать в утвержденный шаблон ответа. Если в черновике ответа встречаются placeholders в квадратных скобках, например `[актуальная стоимость]`, это означает, что бот должен брать значение из актуального файла фактов, а не выдумывать его.

Колонка `Исторический ответ менеджера (не утверждено)` показывает, что реально или приблизительно отвечал менеджер в истории общения. Это не эталонный ответ. Если там стоит короткое `Нет.`, сломанный текст или устаревшая информация, РОП должен исправить это в колонке `Идеальный ответ или шаблон РОПа`.

## Как использовать результат

После разметки мы импортируем файл обратно в проект, создадим новые узкие классы, пересоберем таблицы ответов, добавим тесты качества и снова отдадим результат на аудит.

## Важно

Строки `ИТОГО ПО КЛАССУ` не нужно размечать. Они нужны только как визуальная граница между большими классами.
""",
        encoding="utf-8",
    )


def build_pack(
    catalog_root: Path,
    output_csv: Path,
    output_summary: Path,
    output_guide: Path,
    *,
    limit_per_class: int,
    include_all_quality_blockers: bool = False,
) -> dict[str, Any]:
    blocked_classes = collect_blocked_classes(catalog_root, include_all_quality_blockers=include_all_quality_blockers)
    class_ids = {item.question_class_id for item in blocked_classes}
    items_by_class = _load_items_by_class(catalog_root / "customer_question_items.jsonl", class_ids)
    rows = build_markup_rows(blocked_classes, items_by_class, limit_per_class)
    _write_csv(output_csv, rows)
    _write_summary(output_summary, blocked_classes, rows, catalog_root=catalog_root, limit_per_class=limit_per_class)
    write_guide(output_guide, output_csv, output_summary)
    return {
        "blocked_classes": len(blocked_classes),
        "rows": len(rows),
        "csv": str(output_csv),
        "summary": str(output_summary),
        "guide": str(output_guide),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", type=Path, default=DEFAULT_CATALOG_ROOT)
    parser.add_argument("--date", default=DEFAULT_BUILD_DATE)
    parser.add_argument("--limit-per-class", type=int, default=DEFAULT_LIMIT_PER_CLASS)
    parser.add_argument(
        "--include-all-quality-blockers",
        action="store_true",
        help="Include all P1 quality blockers, not only blocked classes from the ROP top-100 file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    suffix = "all_p1_blockers" if args.include_all_quality_blockers else "top100_blockers"
    output_csv = args.catalog_root / f"rop_blocker_markup_{suffix}_{args.date}.csv"
    output_summary = args.catalog_root / f"rop_blocker_markup_{suffix}_{args.date}.summary.json"
    output_guide = Path("docs") / f"ROP_BLOCKER_MARKUP_GUIDE_{args.date}.md"
    result = build_pack(
        args.catalog_root,
        output_csv,
        output_summary,
        output_guide,
        limit_per_class=args.limit_per_class,
        include_all_quality_blockers=args.include_all_quality_blockers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
