from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("product_data/question_catalog/question_answer_quality_review_2026-05-14_final.csv")
DEFAULT_OUTPUT = Path("product_data/question_catalog/rop_bot_policy_questionnaire_2026-05-14.csv")
DEFAULT_SUMMARY = Path("product_data/question_catalog/rop_bot_policy_questionnaire_2026-05-14.summary.json")

DECISION_OPTIONS = (
    "бот может ответить сам",
    "бот может ответить только после проверки актуального факта",
    "бот только собирает данные и передает менеджеру",
    "только менеджер",
    "тему нужно раздробить на более узкие случаи",
)

FACT_LABELS = {
    "documents": "документы и договоры",
    "schedule": "расписание",
    "price": "цены",
    "program": "программы и предметы",
    "location": "адреса и формат",
    "discount": "скидки",
    "installment": "рассрочка",
    "trial": "пробные занятия",
}

BLOCK_ORDER = {
    "Оплата, чеки, счета": 10,
    "Возвраты, перерасчеты, отмена": 20,
    "Материнский капитал": 30,
    "Налоговый вычет и справки": 40,
    "Документы, справки, договоры": 50,
    "Расписание, формат, адреса": 60,
    "Цены, скидки, рассрочка": 70,
    "Летние школы, лагеря, смены": 80,
    "Программы, уровни, результаты": 90,
    "Доступ, ссылки, личный кабинет": 100,
    "Продолжение обучения": 110,
    "Юридические и партнерские вопросы": 120,
    "Прочие и широкие вопросы": 999,
}

DISPLAY_REPLACEMENTS = {
    "[REFUND_POLICY]": "[правила возврата]",
    "[PAYMENT_OPTIONS]": "[способы оплаты]",
    "[CURRENT_DEADLINE]": "[актуальный срок]",
    "[CURRENT_PRICE]": "[актуальная стоимость]",
    "[CLIENT_NAME]": "[имя клиента]",
    "[COMPANY_NAME]": "[название компании]",
    "[PHONE]": "[телефон]",
    "[EMAIL]": "[почта]",
}

QUESTIONNAIRE_COLUMNS = [
    "Номер",
    "Блок для РОПа",
    "Тема, которую нужно утвердить",
    "Почему это важно",
    "Как клиенты спрашивают",
    "Черновик безопасного ответа",
    "Рекомендация системы",
    "Вопрос 1. Что разрешаем боту?",
    "Вопрос 2. Какую фразу бот может сказать клиенту?",
    "Вопрос 3. Какие данные бот обязан спросить или проверить?",
    "Вопрос 4. Какие обещания боту запрещены?",
    "Вопрос 5. Где брать актуальные факты?",
    "Вопрос 6. Нужно ли дробить тему?",
    "Вопрос 7. Кому передавать, если бот не отвечает сам?",
    "Ответ РОПа: разрешение боту",
    "Ответ РОПа: утвержденная формулировка",
    "Ответ РОПа: обязательные данные",
    "Ответ РОПа: запреты",
    "Ответ РОПа: дробление темы",
    "Комментарий РОПа",
]


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


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", safe_text(value).casefold()).strip()


def clean_display_text(value: str) -> str:
    text = safe_text(value)
    for needle, replacement in DISPLAY_REPLACEMENTS.items():
        text = text.replace(needle, replacement)
    return text


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=QUESTIONNAIRE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def short_examples(value: str, *, limit: int = 3, max_chars: int = 520) -> str:
    examples: list[str] = []
    seen: set[str] = set()
    for part in safe_text(value).split("|"):
        text = re.sub(r"\s+", " ", clean_display_text(part)).strip()
        if not text:
            continue
        key = normalize(text)
        if key in seen:
            continue
        seen.add(key)
        examples.append(text)
        if len(examples) >= limit:
            break
    joined = " | ".join(examples)
    if len(joined) > max_chars:
        return joined[: max_chars - 1].rstrip() + "..."
    return joined


def russian_fact_list(value: str) -> str:
    labels: list[str] = []
    for raw in safe_text(value).split("|"):
        key = raw.strip().split(".")[0]
        if not key or key == "нет":
            continue
        labels.append(FACT_LABELS.get(key, key))
    return ", ".join(dict.fromkeys(labels)) or "не указано"


def business_block(parent: str) -> str:
    text = normalize(parent)
    if "оплат" in text or "чек" in text or "счет" in text:
        return "Оплата, чеки, счета"
    if "возврат" in text or "перерасчет" in text:
        return "Возвраты, перерасчеты, отмена"
    if "изменение" in text or "отмена" in text:
        return "Возвраты, перерасчеты, отмена"
    if "налог" in text:
        return "Налоговый вычет и справки"
    if "письма" in text or "справки" in text or "документ" in text or "договор" in text:
        return "Документы, справки, договоры"
    if "распис" in text or "формат" in text or "адрес" in text:
        return "Расписание, формат, адреса"
    if "стоимость" in text or "скид" in text:
        return "Цены, скидки, рассрочка"
    if "лагерь" in text or "смен" in text or "поезд" in text:
        return "Летние школы, лагеря, смены"
    if "доступ" in text or "ссылка" in text or "кабинет" in text:
        return "Доступ, ссылки, личный кабинет"
    if "продолжение" in text or "решение семьи" in text:
        return "Продолжение обучения"
    if "программ" in text or "материал" in text or "уровень" in text or "возраст" in text:
        return "Программы, уровни, результаты"
    if "юрид" in text or "партнер" in text:
        return "Юридические и партнерские вопросы"
    return "Прочие и широкие вопросы"


def business_block_for_row(row: dict[str, str]) -> str:
    text = normalize(
        " ".join(
            [
                row.get("Узкий класс", ""),
                row.get("Группы риска", ""),
            ]
        )
    )
    if "материн" in text:
        return "Материнский капитал"
    if "возврат" in text or "перерасчет" in text:
        return "Возвраты, перерасчеты, отмена"
    return business_block(row.get("Крупный класс", ""))


def recommendation(action: str) -> str:
    if action == "после проверки актуального факта и утверждения РОПом":
        return "Предварительно: бот отвечает после проверки факта. Нужно утверждение РОПа и актуальный источник."
    if action == "только черновик для менеджера":
        return "Предварительно: только черновик для менеджера. Бот сам клиенту не отправляет."
    if action == "только менеджер":
        return "Предварительно: только менеджер. Бот не отвечает по сути, а собирает данные и передает."
    if action == "нельзя утверждать, сначала дробить класс":
        return "Предварительно: нужно раздробить тему. Единый ответ утверждать нельзя."
    return "Предварительно: нужно решение РОПа."


def rop_task(action: str) -> str:
    if action == "нельзя утверждать, сначала дробить класс":
        return "Решите, на какие 2-5 более узких случаев разделить эту тему. До дробления не утверждайте единый ответ."
    if action == "только менеджер":
        return "Решите, какую короткую безопасную фразу бот может сказать до передачи менеджеру."
    if action == "только черновик для менеджера":
        return "Решите, может ли бот готовить черновик менеджеру и какие данные он обязан собрать."
    return "Решите, можно ли боту отвечать после проверки актуальных фактов, и утвердите текст."


def forbidden_promises(row: dict[str, str]) -> str:
    text = normalize(" ".join([row.get("Крупный класс", ""), row.get("Узкий класс", ""), row.get("Группы риска", "")]))
    rules: list[str] = []
    if "возврат" in text or "перерасчет" in text:
        rules.append("не обещать возврат, срок возврата, отсутствие штрафа или сумму без проверки договора и оплат")
    if "материн" in text:
        rules.append("не обещать, что материнский капитал точно примут, без проверки сертификата, региона, договора и лицензии")
    if "договор" in text or "юрид" in text:
        rules.append("не трактовать договор и не давать юридических обещаний без менеджера")
    if "оплат" in text or "чек" in text or "счет" in text:
        rules.append("не подтверждать оплату и не называть личную сумму без сверки карточки клиента")
    if "распис" in text or "адрес" in text or "формат" in text:
        rules.append("не называть дни, время, адрес или формат без актуального расписания")
    if "стоимость" in text or "скид" in text or "рассроч" in text:
        rules.append("не называть цену, скидку или рассрочку без актуального файла условий")
    if "доступ" in text or "кабинет" in text or "ссылка" in text:
        rules.append("не просить пароль и не отправлять доступ без проверки ученика и группы")
    if not rules:
        rules.append("не давать точные обещания без проверки карточки клиента и актуальных материалов")
    return "; ".join(dict.fromkeys(rules))


def required_data(row: dict[str, str]) -> str:
    placeholders = safe_text(row.get("Какие данные подставить"))
    if placeholders and placeholders != "нет":
        return placeholders
    facts = safe_text(row.get("Нужные актуальные факты"))
    if facts and facts != "нет":
        return f"проверить актуальные факты: {russian_fact_list(facts)}"
    return "минимум: ФИО ученика, класс, предмет/курс, цель обращения"


def fact_sources(row: dict[str, str]) -> str:
    facts = safe_text(row.get("Нужные актуальные факты"))
    if facts == "нет":
        return "актуальный источник не требуется, но нужно проверить карточку клиента при персональном вопросе"
    return f"указать актуальный файл, таблицу или ответственного за факты: {russian_fact_list(facts)}"


def should_split_question(row: dict[str, str]) -> str:
    if row.get("Что бот может делать") == "нельзя утверждать, сначала дробить класс":
        return "Да. Напишите, на какие более узкие случаи разделить эту тему."
    if row.get("Много тем в одном вопросе") == "да":
        return "Вероятно да. Проверьте, не смешаны ли разные ситуации."
    if int_value(row.get("Всего вопросов")) >= 150:
        return "Проверьте. Обращений много, возможно внутри есть несколько разных сценариев."
    return "Если правила отличаются по ситуациям, укажите подклассы. Если нет, напишите «не дробить»."


def owner_hint(row: dict[str, str]) -> str:
    text = normalize(" ".join([row.get("Крупный класс", ""), row.get("Узкий класс", ""), row.get("Группы риска", "")]))
    if "возврат" in text or "договор" in text or "юрид" in text or "материн" in text:
        return "менеджер + РОП, при необходимости бухгалтерия/администратор документов"
    if "оплат" in text or "чек" in text or "счет" in text:
        return "менеджер, при необходимости бухгалтерия"
    if "распис" in text or "адрес" in text or "формат" in text:
        return "менеджер/администратор расписания"
    if "доступ" in text or "кабинет" in text or "ссылка" in text:
        return "менеджер + технический ответственный"
    return "менеджер или РОП по ситуации"


def row_sort_key(row: dict[str, str]) -> tuple[str, int, str]:
    block = business_block_for_row(row)
    return (
        f"{BLOCK_ORDER.get(block, 999):03d}_{block}",
        -int_value(row.get("Приоритетный балл")),
        safe_text(row.get("Класс вопроса")),
    )


def build_questionnaire_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for index, row in enumerate(sorted(source_rows, key=row_sort_key), start=1):
        block = business_block_for_row(row)
        total = int_value(row.get("Всего вопросов"))
        risk = safe_text(row.get("Риск ошибки")) or "не указан"
        groups = safe_text(row.get("Группы риска")) or "нет"
        result.append(
            {
                "Номер": str(index),
                "Блок для РОПа": block,
                "Тема, которую нужно утвердить": safe_text(row.get("Класс вопроса")),
                "Почему это важно": f"{total} обращений; риск ошибки: {risk}; группы риска: {groups}.",
                "Как клиенты спрашивают": short_examples(row.get("Реальные примеры вопросов", "")),
                "Черновик безопасного ответа": clean_display_text(row.get("Предлагаемый ответ", "")),
                "Рекомендация системы": recommendation(safe_text(row.get("Что бот может делать"))),
                "Вопрос 1. Что разрешаем боту?": " / ".join(DECISION_OPTIONS),
                "Вопрос 2. Какую фразу бот может сказать клиенту?": rop_task(safe_text(row.get("Что бот может делать"))),
                "Вопрос 3. Какие данные бот обязан спросить или проверить?": required_data(row),
                "Вопрос 4. Какие обещания боту запрещены?": forbidden_promises(row),
                "Вопрос 5. Где брать актуальные факты?": fact_sources(row),
                "Вопрос 6. Нужно ли дробить тему?": should_split_question(row),
                "Вопрос 7. Кому передавать, если бот не отвечает сам?": owner_hint(row),
                "Ответ РОПа: разрешение боту": "",
                "Ответ РОПа: утвержденная формулировка": "",
                "Ответ РОПа: обязательные данные": "",
                "Ответ РОПа: запреты": "",
                "Ответ РОПа: дробление темы": "",
                "Комментарий РОПа": "",
            }
        )
    return result


def build_summary(rows: list[dict[str, str]], output_csv: Path, source: Path) -> dict[str, Any]:
    blocks = Counter(row["Блок для РОПа"] for row in rows)
    recommendations = Counter(row["Рекомендация системы"] for row in rows)
    return {
        "schema_version": "rop_bot_policy_questionnaire_v1",
        "source": str(source),
        "output_csv": str(output_csv),
        "rows": len(rows),
        "blocks": dict(blocks),
        "system_recommendations": dict(recommendations),
        "decision_options": list(DECISION_OPTIONS),
        "purpose": "Понятный опросник для РОПа: утвердить права бота, безопасные формулировки, обязательные данные, запреты и дробление тем.",
    }


def build_questionnaire(source: Path = DEFAULT_SOURCE, output_csv: Path = DEFAULT_OUTPUT, summary_path: Path = DEFAULT_SUMMARY) -> dict[str, Any]:
    source_rows = read_csv(source)
    rows = build_questionnaire_rows(source_rows)
    write_csv(output_csv, rows)
    summary = build_summary(rows, output_csv, source)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
