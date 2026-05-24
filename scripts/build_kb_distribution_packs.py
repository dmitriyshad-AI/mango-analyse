#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


DEFAULT_RELEASE_DIR = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_handoff_for_claude_and_team")
DEFAULT_FULL_RELEASE_DIR = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers")
DEFAULT_SMOKE_DIR = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_smoke_not_run")
DEFAULT_EMPLOYEE_OUT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_employee_pack")
DEFAULT_BOT_OUT = Path("product_data/knowledge_base/kb_release_20260520_v6_3_team_answers_bot_pack")
PACK_SCHEMA_VERSION = "kb_distribution_packs_v1"

BRAND_LABELS = {
    "foton": "Фотон",
    "unpk": "УНПК МФТИ",
    "brand_neutral": "общее",
    "internal": "внутреннее",
}
FACT_TYPE_LABELS = {
    "price": "цены",
    "discount": "скидки",
    "installment": "рассрочка и варианты оплаты",
    "matkap": "материнский капитал",
    "tax": "налоговый вычет",
    "documents": "документы и справки",
    "contact": "контакты",
    "location": "адреса",
    "deadline": "даты и сроки",
    "program": "программы",
    "course_parameter": "параметры занятий",
    "camp_lvsh": "ЛВШ",
    "camp_city": "городской лагерь",
    "camp_zvsh": "ЗВШ",
    "intensive": "интенсивы",
    "teacher": "преподаватели",
    "policy": "правила ответа",
    "refund": "возвраты",
    "promocode": "промокоды",
}

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build human and bot distribution packs from KB v3.2.")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--full-release-dir", type=Path, default=DEFAULT_FULL_RELEASE_DIR)
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_SMOKE_DIR)
    parser.add_argument("--employee-out", type=Path, default=DEFAULT_EMPLOYEE_OUT)
    parser.add_argument("--bot-out", type=Path, default=DEFAULT_BOT_OUT)
    args = parser.parse_args(argv)

    result = build_distribution_packs(
        release_dir=args.release_dir,
        full_release_dir=args.full_release_dir,
        smoke_dir=args.smoke_dir,
        employee_out=args.employee_out,
        bot_out=args.bot_out,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_distribution_packs(
    *,
    release_dir: Path,
    full_release_dir: Path = DEFAULT_FULL_RELEASE_DIR,
    smoke_dir: Path,
    employee_out: Path,
    bot_out: Path,
) -> Mapping[str, Any]:
    release = release_dir.expanduser().resolve(strict=False)
    full_release = full_release_dir.expanduser().resolve(strict=False)
    employees = prepare_output_dir(employee_out)
    bot = prepare_output_dir(bot_out)
    facts = load_facts(release)
    snapshot = load_json(release / "kb_release_v3_snapshot.json")
    quality = load_json(release / "quality_report.json")
    semantic = load_json(release / "semantic_review.json")
    smoke = load_smoke_summaries(smoke_dir)
    brand_forbidden = extract_brand_forbidden(load_yaml_mapping(full_release / "brand_rules.yaml"))

    build_employee_pack(
        employees,
        facts=facts,
        snapshot=snapshot,
        quality=quality,
        semantic=semantic,
        smoke=smoke,
        release_dir=release,
        full_release_dir=full_release,
        brand_forbidden=brand_forbidden,
    )
    build_bot_pack(
        bot,
        release=release,
        full_release=full_release,
        facts=facts,
        snapshot=snapshot,
        quality=quality,
        semantic=semantic,
        smoke=smoke,
        brand_forbidden=brand_forbidden,
    )
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "employee_pack": str(employees),
        "bot_pack": str(bot),
        "facts_total": len(facts),
        "client_safe_facts": sum(1 for fact in facts if truthy(fact.get("allowed_for_client_answer"))),
        "employee_files": sorted(path.name for path in employees.iterdir() if path.is_file()),
        "bot_files": sorted(path.name for path in bot.iterdir() if path.is_file()),
    }


def build_employee_pack(
    out: Path,
    *,
    facts: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    quality: Mapping[str, Any],
    semantic: Mapping[str, Any],
    smoke: Mapping[str, Any],
    release_dir: Path,
    full_release_dir: Path,
    brand_forbidden: Mapping[str, Sequence[str]],
) -> None:
    allowed = [fact for fact in facts if truthy(fact.get("allowed_for_client_answer"))]
    manager_only = [fact for fact in facts if not truthy(fact.get("allowed_for_client_answer"))]
    write_markdown(out / "START_HERE.md", render_employee_start(snapshot, quality, semantic, smoke, allowed, manager_only))
    write_markdown(out / "README.md", render_employee_start(snapshot, quality, semantic, smoke, allowed, manager_only))
    write_markdown(out / "BRAND_ISOLATION.md", render_brand_isolation())
    write_markdown(out / "MANAGER_ONLY.md", render_manager_only(manager_only))
    write_markdown(out / "FOR_AI_AGENTS.md", render_employee_ai_agent_instructions())
    write_markdown(out / "FOTON.md", render_brand_employee_doc("foton", allowed, brand_forbidden))
    write_markdown(out / "UNPK.md", render_brand_employee_doc("unpk", allowed, brand_forbidden))
    gold_answers = extract_gold_answers(snapshot)
    if gold_answers:
        write_markdown(out / "GOLD_ANSWERS.md", render_gold_answers_markdown(gold_answers, audience="employees"))
    write_markdown(out / "FAQ_EXAMPLES.md", render_smoke_examples())
    write_markdown(out / "SEMANTIC_REVIEW.md", render_pack_semantic_review(quality, semantic, smoke, audience="employees"))
    write_csv(out / "CLIENT_SAFE_FACTS_FOTON.csv", fact_rows(fact for fact in allowed if fact.get("brand") == "foton"))
    write_csv(out / "CLIENT_SAFE_FACTS_UNPK.csv", fact_rows(fact for fact in allowed if fact.get("brand") == "unpk"))
    write_csv(out / "MANAGER_ONLY_FACTS.csv", fact_rows(manager_only))
    write_json(
        out / "manifest.json",
        manifest_payload(
            "employee",
            facts,
            quality,
            semantic,
            smoke,
            release_dir=release_dir,
            full_release_dir=full_release_dir,
        ),
    )


def build_bot_pack(
    out: Path,
    *,
    release: Path,
    full_release: Path,
    facts: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any],
    quality: Mapping[str, Any],
    semantic: Mapping[str, Any],
    smoke: Mapping[str, Any],
    brand_forbidden: Mapping[str, Sequence[str]],
) -> None:
    allowed = [fact for fact in facts if truthy(fact.get("allowed_for_client_answer"))]
    manager_only = [fact for fact in facts if not truthy(fact.get("allowed_for_client_answer"))]
    write_markdown(out / "README_FOR_BOT.md", render_bot_readme(snapshot, quality, semantic, smoke))
    write_markdown(out / "BOT_USAGE_CONTRACT.md", render_bot_contract())
    write_markdown(out / "ACTIVE_BRAND_RULES.md", render_active_brand_rules(brand_forbidden))
    write_markdown(out / "SEMANTIC_REVIEW.md", render_pack_semantic_review(quality, semantic, smoke, audience="bot"))
    write_jsonl(out / "client_safe_facts_foton.jsonl", [fact for fact in allowed if fact.get("brand") == "foton"])
    write_jsonl(out / "client_safe_facts_unpk.jsonl", [fact for fact in allowed if fact.get("brand") == "unpk"])
    write_jsonl(out / "manager_only_or_internal_facts.jsonl", manager_only)
    write_json(out / "bot_template_registry.json", build_bot_template_registry(allowed))
    write_json(out / "bot_fact_index.json", build_fact_index(facts))
    gold_answers = extract_gold_answers(snapshot)
    if gold_answers:
        write_json(out / "bot_gold_answers.json", gold_answers)
        write_yaml(out / "gold_answer_rules.yaml", gold_answers)
        write_markdown(out / "GOLD_ANSWERS_FOR_BOT.md", render_gold_answers_markdown(gold_answers, audience="bot"))
    write_json(
        out / "manifest.json",
        manifest_payload(
            "bot",
            facts,
            quality,
            semantic,
            smoke,
            release_dir=release,
            full_release_dir=full_release,
        ),
    )
    for filename in (
        "facts_registry.jsonl",
        "facts_registry.yaml",
        "facts_registry.csv",
        "source_registry.json",
        "quality_report.json",
        "semantic_review.json",
        "post_filter_registry.json",
        "bot_policy.yaml",
        "brand_rules.yaml",
    ):
        source = release / filename
        fallback = full_release / filename
        if source.exists():
            shutil.copy2(source, out / filename)
        elif fallback.exists():
            shutil.copy2(fallback, out / filename)


def render_employee_start(
    snapshot: Mapping[str, Any],
    quality: Mapping[str, Any],
    semantic: Mapping[str, Any],
    smoke: Mapping[str, Any],
    allowed: Sequence[Mapping[str, Any]],
    manager_only: Sequence[Mapping[str, Any]],
) -> str:
    counts = Counter(str(fact.get("brand") or "") for fact in allowed)
    return "\n".join(
        [
            "# База знаний для сотрудников",
            "",
            "Эта папка нужна, чтобы менеджер или его ИИ-помощник не начинал с чистого листа.",
            "Она содержит проверенные факты, правила ответа и границы безопасности для Фотона и УНПК МФТИ.",
            "",
            "## Как пользоваться",
            "",
            "1. Сначала выберите бренд клиента: Фотон или УНПК МФТИ.",
            "2. Откройте только файл нужного бренда: `FOTON.md` или `UNPK.md`.",
            "3. Если вопрос про возврат, жалобу, суд, оплату, документы или смешение брендов, проверьте `MANAGER_ONLY.md`.",
            "4. Если отдаёте папку ИИ-агенту, приложите `FOR_AI_AGENTS.md` вместе с файлом нужного бренда.",
            "5. Не копируйте клиенту служебные таблицы целиком. Это источник фактов, а не готовый скрипт для массовой отправки.",
            "",
            "## Статус качества",
            "",
            f"- run_id: `{snapshot.get('run_id')}`",
            f"- formal_pass: `{quality.get('quality_passed')}`",
            f"- semantic_pass: `{semantic.get('semantic_pass')}`",
            f"- client_safe_facts: `{len(allowed)}`",
            f"- manager_only_or_internal_facts: `{len(manager_only)}`",
            f"- Фотон client-safe facts: `{counts.get('foton', 0)}`",
            f"- УНПК client-safe facts: `{counts.get('unpk', 0)}`",
            f"- Smoke 50: `{smoke.get('short_status', 'см. manifest')}`",
            "",
            "## Главный принцип",
            "",
            "Один ответ говорит только от имени одного бренда. Фотон не объясняет условия УНПК, УНПК не объясняет условия Фотона.",
            "",
        ]
    )


def render_brand_employee_doc(
    brand: str,
    facts: Sequence[Mapping[str, Any]],
    brand_forbidden: Mapping[str, Sequence[str]],
) -> str:
    brand_facts = [fact for fact in facts if fact.get("brand") == brand]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for fact in brand_facts:
        grouped[str(fact.get("fact_type") or "other")].append(fact)
    lines = [
        f"# {BRAND_LABELS[brand]}: факты для работы",
        "",
        "Используйте только в рамках этого бренда. Не добавляйте условия другого учебного центра.",
        "",
        "## Быстрые запреты",
        "",
    ]
    for marker in brand_forbidden.get(brand, ()):
        lines.append(f"- Не писать клиенту: `{marker}`")
    lines.extend(["", "## Факты по разделам", ""])
    for fact_type in sorted(grouped, key=lambda item: FACT_TYPE_LABELS.get(item, item)):
        items = sorted(grouped[fact_type], key=lambda fact: str(fact.get("client_safe_text") or fact.get("fact_key") or ""))
        lines.extend([f"### {FACT_TYPE_LABELS.get(fact_type, fact_type)}", ""])
        for fact in items:
            text = clean_text(fact.get("client_safe_text") or fact.get("fact_text"))
            source = clean_text(fact.get("source_title") or fact.get("source_id"), limit=120)
            freshness = clean_text(fact.get("freshness_status") or "")
            route = clean_text(fact.get("route_policy") or "")
            lines.append(f"- {text}")
            lines.append(f"  Источник: {source}. Статус: `{freshness}`. Маршрут: `{route}`.")
        lines.append("")
    return "\n".join(lines)


def render_manager_only(facts: Sequence[Mapping[str, Any]]) -> str:
    grouped = Counter(str(fact.get("fact_type") or "other") for fact in facts)
    examples = sorted(facts, key=lambda fact: (str(fact.get("fact_type")), str(fact.get("brand")), str(fact.get("fact_key"))))[:120]
    lines = [
        "# Что не отдавать клиенту напрямую",
        "",
        "Эти факты и темы используются для проверки менеджером, но не являются готовым клиентским ответом.",
        "",
        "## Всегда осторожно",
        "",
        "- возврат денег и расторжение договора;",
        "- жалобы, угрозы суда, прокуратуры, Роспотребнадзора;",
        "- оплата, если нет совпадающего подтверждения в AMO и Tallanto;",
        "- юридические реквизиты и внутренние названия юрлиц;",
        "- промокоды;",
        "- любые сравнения Фотона и УНПК;",
        "- преподаватели по ФИО, если это не утверждено отдельным правилом.",
        "",
        "## Сколько таких фактов в базе",
        "",
    ]
    for fact_type, count in grouped.most_common():
        lines.append(f"- {FACT_TYPE_LABELS.get(fact_type, fact_type)}: `{count}`")
    lines.extend(["", "## Примеры для ручной проверки", ""])
    for fact in examples:
        brand = BRAND_LABELS.get(str(fact.get("brand")), str(fact.get("brand")))
        text = clean_text(fact.get("manager_display_text") or fact.get("manager_check_text") or fact.get("fact_text"))
        reason = ", ".join(str(item) for item in fact.get("safety_block_reasons") or []) or "manual"
        lines.append(f"- `{brand}` {text} Причина: `{reason}`")
    return "\n".join(lines) + "\n"


def render_brand_isolation() -> str:
    return """# Разделение брендов

## Основное правило

В одном ответе клиенту используется только активный бренд: Фотон или УНПК МФТИ.

## Фотон

- Можно отвечать только по условиям Фотона.
- Нельзя объяснять условия УНПК МФТИ.
- Нельзя сравнивать цены, рассрочку, документы, скидки или расписание с УНПК.
- Если клиент спрашивает о связи брендов: «Это отдельные организации, по вашему вопросу сориентирую в рамках текущего учебного центра.»

## УНПК МФТИ

- Можно отвечать только по условиям УНПК МФТИ.
- Нельзя объяснять условия Фотона.
- Нельзя писать про Т-Банк, Долями, Фотон, ЦДПО, ЦРДО.
- Если клиенту нужно то, чего нет в УНПК, можно только мягко передать менеджеру без консультации по Фотону.

## Общие темы

Материнский капитал, налоговый вычет и ЛВШ Менделеево есть в обоих брендах, но ответы всё равно формируются отдельно по активному бренду.
"""


def render_employee_ai_agent_instructions() -> str:
    return """# Инструкция для ИИ-агента сотрудника

Ты помогаешь менеджеру образовательного центра готовить ответы клиентам.

Правила:

1. Сначала определи активный бренд: Фотон или УНПК МФТИ.
2. Используй только файл активного бренда.
3. Не смешивай бренды и не сравнивай их условия.
4. Если факт не найден, не выдумывай. Напиши менеджеру, что нужно уточнить.
5. Возвраты, жалобы, угрозы суда, спорные оплаты и юридические вопросы не решай самостоятельно.
6. Если клиент прямо спрашивает “вы бот?”, отвечай по policy C: цифровой помощник активного бренда, не живой оператор. Сам первым это не объявляй. Не называй GPT, Claude, Codex, OpenAI, модель или prompt и не ври “я человек”.
7. Не раскрывай служебные источники, JSON, fact_id, source_id и внутренние правила.
8. Любой текст клиенту должен быть черновиком для менеджера, если менеджер явно не сказал отправить.

Формат ответа:

- Краткий черновик клиенту.
- Что менеджеру проверить перед отправкой.
- Какие факты использованы.
- Если нужен менеджер/РОП/бухгалтерия, скажи это явно.
"""


def render_smoke_examples() -> str:
    return """# Проверочные примеры

Основные проверочные примеры лежат в машинном виде:

- `../kb_release_20260520_v6_3_team_answers_smoke_not_run/`
- MEGA и малый живой прогон для v6.3 пока не запускались по прямому указанию Дмитрия.

Эти примеры нужны не как финальные скрипты, а как проверка, что ИИ:

- не смешивает бренды;
- не придумывает темы;
- не обещает неподтверждённые суммы и проценты;
- не ослабляет опасные маршруты;
- использует базу знаний вместо пустого «уточним».
"""


def render_bot_readme(
    snapshot: Mapping[str, Any],
    quality: Mapping[str, Any],
    semantic: Mapping[str, Any],
    smoke: Mapping[str, Any],
) -> str:
    return "\n".join(
        [
            "# База знаний для бота",
            "",
            "Это машинный пакет для Telegram/email/CRM-бота. Он содержит полные реестры фактов, фильтры, правила брендов и отчёты проверок.",
            "",
            "## Статус",
            "",
            f"- run_id: `{snapshot.get('run_id')}`",
            f"- formal_pass: `{quality.get('quality_passed')}`",
            f"- semantic_pass: `{semantic.get('semantic_pass')}`",
            f"- blocking_findings: `{semantic.get('blocking_findings')}`",
            f"- smoke_status: `{smoke.get('short_status', '')}`",
            "",
            "## Главные файлы",
            "",
            "- `client_safe_facts_foton.jsonl` — факты, которые можно использовать для Фотона.",
            "- `client_safe_facts_unpk.jsonl` — факты, которые можно использовать для УНПК.",
            "- `manager_only_or_internal_facts.jsonl` — факты только для проверки менеджером/внутренней логики.",
            "- `facts_registry.jsonl` — полный реестр с разрешениями и маршрутами.",
            "- `post_filter_registry.json` — запретные фразы и фильтры.",
            "- `bot_template_registry.json` — обязательные шаблоны для фактов, которые нельзя подставлять дословно.",
            "- `bot_fact_index.json` — компактный индекс по брендам, типам и маршрутам.",
            "- `BOT_USAGE_CONTRACT.md` — правила использования в боте.",
            "",
            "Бот не должен отправлять сообщения клиентам напрямую на первом этапе. Он готовит черновик и показывает менеджеру.",
            "",
        ]
    )


def render_bot_contract() -> str:
    return """# Контракт использования базы ботом

1. Активный бренд задаётся входным каналом: отдельный бот Фотона или отдельный бот УНПК.
2. Перед ответом выбрать только facts активного бренда.
3. `allowed_for_client_answer=true` не равно «можно отправить без менеджера». На первом этапе это источник для черновика.
4. `manager_only_or_internal_facts.jsonl` нельзя цитировать клиенту.
5. Если нужен факт из AMO/Tallanto, бот делает только read-only проверку.
6. Если AMO и Tallanto противоречат друг другу, маршрут `manager_only`.
7. Если в черновике появились чужой бренд, неподтверждённая сумма, процент, дата или обещание результата, маршрут `manager_only`.
8. Промокоды не использовать в клиентском ответе.
9. Возвраты, жалобы, угрозы суда и спорные оплаты — только менеджеру.
10. Все ответы в пилоте идут как черновик в служебный чат менеджера.
11. Бот не отправляет сообщения клиенту напрямую, пока Дмитрий отдельно не подтвердит режим автоответов.
12. Бот не подставляет `client_safe_text` дословно в сообщение клиенту или менеджеру. Он использует факт как источник смысла и собирает нормальную фразу из утверждённого шаблона, `structured_value.raw_value` и контекста вопроса.
13. Если у факта стоит `bot_template_required=true`, его нельзя использовать как готовую фразу: сначала нужно собрать человеческий ответ через шаблон с явным смыслом числа, срока, цены или условия.
14. Все факты с `bot_template_required=true` должны иметь запись в `bot_template_registry.json`. Если записи нет или шаблон не подходит к вопросу клиента, маршрут `manager_only`.
15. `pattern_descriptions` в `post_filter_registry.json` — это человекочитаемые пояснения, а не строки для поиска. Матчеры используют `global_phrases`, `phrases_by_active_brand[active_brand]` и `regex_patterns`; поле `phrases` оставлено как совместимая копия глобальных фраз.
16. Для показа менеджеру использовать `manager_display_text`; `manager_check_text` может содержать служебные причины блокировки.
"""


def render_active_brand_rules(brand_forbidden: Mapping[str, Sequence[str]]) -> str:
    foton_block = "\n".join(f"- {item}" for item in brand_forbidden.get("foton", ())) or "- см. `brand_rules.yaml`"
    unpk_block = "\n".join(f"- {item}" for item in brand_forbidden.get("unpk", ())) or "- см. `brand_rules.yaml`"
    return f"""# Правила active_brand

## active_brand=foton

Использовать:

- `client_safe_facts_foton.jsonl`
- brand `foton`
- общие правила только если они не называют УНПК как условие клиента

Блокировать:

{foton_block}

## active_brand=unpk

Использовать:

- `client_safe_facts_unpk.jsonl`
- brand `unpk`
- общие правила только если они не называют Фотон как условие клиента

Блокировать:

{unpk_block}
"""


def render_pack_semantic_review(
    quality: Mapping[str, Any],
    semantic: Mapping[str, Any],
    smoke: Mapping[str, Any],
    *,
    audience: str,
) -> str:
    verdict = "PASS_WITH_NOTES" if semantic.get("semantic_pass") and semantic.get("findings_total") else "PASS"
    return "\n".join(
        [
            f"# Semantic review: {audience}",
            "",
            f"Verdict: `{verdict}`",
            "",
            "## What passed",
            "",
            f"- formal_pass: `{quality.get('quality_passed')}`",
            f"- semantic_pass: `{semantic.get('semantic_pass')}`",
            f"- blocking_findings: `{semantic.get('blocking_findings')}`",
            f"- smoke50: `{smoke.get('short_status', '')}`",
            "",
            "## Non-blocking risks",
            "",
            "- Для широкого запуска нужно продолжать ручную проверку редких и спорных фактов, даже если автоматический gate зелёный.",
            "- Пакет для сотрудников облегчает работу, но не заменяет решение РОПа/менеджера по спорным кейсам.",
            "- Пакет для бота рассчитан на режим черновиков, а не автономную отправку клиенту.",
            "",
            "## Required controls",
            "",
            "- Перепроверять чувствительные к дате факты перед широким запуском.",
            "- Не смешивать бренды.",
            "- Все новые смысловые ошибки переводить в тест, фильтр или чек-лист.",
            "",
        ]
    )


def render_manager_summary(facts: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(str(fact.get("fact_type") or "other") for fact in facts)
    return "\n".join(f"- {FACT_TYPE_LABELS.get(key, key)}: `{value}`" for key, value in counts.most_common())


def manifest_payload(
    package_type: str,
    facts: Sequence[Mapping[str, Any]],
    quality: Mapping[str, Any],
    semantic: Mapping[str, Any],
    smoke: Mapping[str, Any],
    *,
    release_dir: Path,
    full_release_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": PACK_SCHEMA_VERSION,
        "package_type": package_type,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_release": str(release_dir),
        "source_full_release": str(full_release_dir),
        "facts_total": len(facts),
        "client_safe_facts_total": sum(1 for fact in facts if truthy(fact.get("allowed_for_client_answer"))),
        "facts_by_brand": dict(Counter(str(fact.get("brand") or "") for fact in facts)),
        "formal_pass": bool(quality.get("quality_passed")),
        "semantic_pass": bool(semantic.get("semantic_pass")),
        "semantic_blocking_findings": semantic.get("blocking_findings"),
        "smoke": smoke,
        "safety": {
            "client_auto_send": False,
            "crm_write": False,
            "tallanto_write": False,
            "active_brand_required": True,
        },
    }


def build_fact_index(facts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    index: dict[str, Any] = {
        "schema_version": "bot_fact_index_v1",
        "brands": {},
        "route_policies": {},
        "fact_types": {},
    }
    for fact in facts:
        brand = str(fact.get("brand") or "unknown")
        fact_type = str(fact.get("fact_type") or "other")
        route = str(fact.get("route_policy") or "unknown")
        item = {
            "fact_id": fact.get("fact_id"),
            "fact_key": fact.get("fact_key"),
            "fact_type": fact_type,
            "allowed_for_client_answer": bool(fact.get("allowed_for_client_answer")),
            "route_policy": route,
            "risk_level": fact.get("risk_level"),
            "freshness_status": fact.get("freshness_status"),
            "valid_until": fact.get("valid_until"),
            "freshness_check_date": fact.get("freshness_check_date"),
            "bot_template_required": bool(fact.get("bot_template_required")),
        }
        index["brands"].setdefault(brand, []).append(item)
        index["route_policies"].setdefault(route, []).append(str(fact.get("fact_id")))
        index["fact_types"].setdefault(fact_type, []).append(str(fact.get("fact_id")))
    return index


def build_bot_template_registry(facts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    template_facts = [
        fact
        for fact in facts
        if truthy(fact.get("allowed_for_client_answer")) and truthy(fact.get("bot_template_required"))
    ]
    templates = []
    for fact in sorted(template_facts, key=lambda item: (str(item.get("brand")), str(item.get("fact_key")))):
        templates.append(
            {
                "fact_id": fact.get("fact_id"),
                "fact_key": fact.get("fact_key"),
                "brand": fact.get("brand"),
                "fact_type": fact.get("fact_type"),
                "template_id": template_id_for_fact(fact),
                "template_text": template_text_for_fact(fact),
                "fallback_route": "manager_only",
                "client_send": False,
            }
        )
    return {
        "schema_version": "bot_template_registry_v1",
        "description": "Шаблоны для client-safe фактов, где дословная подстановка текста запрещена.",
        "facts_requiring_template_total": len(template_facts),
        "templates_total": len(templates),
        "fallback_route_if_missing": "manager_only",
        "templates": templates,
    }


def template_id_for_fact(fact: Mapping[str, Any]) -> str:
    fact_type = str(fact.get("fact_type") or "fact")
    return f"template:{fact_type}:contextual_answer_v1"


def template_text_for_fact(fact: Mapping[str, Any]) -> str:
    fact_type = str(fact.get("fact_type") or "")
    if fact_type == "discount":
        return (
            "Ответить только если из факта понятны размер и условие скидки. "
            "Сказать: «По этой программе действует [размер], условие: [условие]. Перед оформлением менеджер проверит актуальность». "
            "Если условия нет — manager_only."
        )
    if fact_type in {"price", "installment"}:
        return (
            "Ответить только с активным брендом, точной суммой/диапазоном, периодом и valid_until. "
            "Если не хватает периода, формата или даты актуальности — manager_only."
        )
    if fact_type in {"deadline", "program", "camp_lvsh", "camp_city", "camp_zvsh", "intensive"}:
        return (
            "Собрать фразу с названием программы, датой/условием и оговоркой «по текущим данным». "
            "Не обещать место, группу или зачисление без проверки."
        )
    return (
        "Собрать человеческую фразу из structured_value и контекста вопроса. "
        "Не копировать client_safe_text дословно. Если смысл числа или условия неясен — manager_only."
    )


def extract_gold_answers(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    policy = snapshot.get("bot_policy")
    if not isinstance(policy, Mapping):
        return {}
    gold = policy.get("gold_answers_v3")
    return dict(gold) if isinstance(gold, Mapping) else {}


def render_gold_answers_markdown(gold_answers: Mapping[str, Any], *, audience: str) -> str:
    lines = [
        "# Gold-ответы v3",
        "",
        "Это эталон качества ответа: тон, структура и проверенные границы. Это не скрипт для дословного копирования.",
        "",
        f"- Статус: `{gold_answers.get('status', '')}`",
        f"- Источник: `{gold_answers.get('source_docx', '')}`",
        f"- Назначение: `{gold_answers.get('use_as', '')}`",
        f"- Аудитория пакета: `{audience}`",
        "",
        "## Общие правила",
        "",
    ]
    for item in gold_answers.get("global_rules") or []:
        lines.append(f"- {clean_text(item)}")
    confirmed = gold_answers.get("confirmed_rules")
    if isinstance(confirmed, Mapping):
        lines.extend(["", "## Подтверждённые правила", ""])
        for key, value in confirmed.items():
            lines.append(f"- `{key}`: {clean_text(value, limit=500)}")
    topics = gold_answers.get("topics")
    if isinstance(topics, Mapping):
        lines.extend(["", "## Эталоны по темам", ""])
        for topic, payload in topics.items():
            lines.extend([f"### {topic}", ""])
            if isinstance(payload, Mapping):
                for brand, record in payload.items():
                    lines.append(f"#### {BRAND_LABELS.get(str(brand), str(brand))}")
                    if isinstance(record, Mapping):
                        example = clean_text(record.get("gold_answer_example"), limit=1200)
                        if example:
                            lines.append("")
                            lines.append(example)
                        must_include = record.get("must_include") or []
                        if must_include:
                            lines.append("")
                            lines.append("Должно быть: " + ", ".join(f"`{clean_text(item, limit=80)}`" for item in must_include))
                        must_not = record.get("must_not_include") or []
                        if must_not:
                            lines.append("Нельзя: " + ", ".join(f"`{clean_text(item, limit=80)}`" for item in must_not))
                    lines.append("")
    return "\n".join(lines)


def fact_rows(facts: Sequence[Mapping[str, Any]] | Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fact in facts:
        rows.append(
            {
                "brand": fact.get("brand", ""),
                "fact_type": fact.get("fact_type", ""),
                "route_policy": fact.get("route_policy", ""),
                "risk_level": fact.get("risk_level", ""),
                "client_safe_text": fact.get("client_safe_text", ""),
                "manager_display_text": fact.get("manager_display_text", ""),
                "manager_check_text": fact.get("manager_check_text", ""),
                "source_title": fact.get("source_title", ""),
                "freshness_status": fact.get("freshness_status", ""),
                "valid_until": fact.get("valid_until", ""),
                "freshness_check_date": fact.get("freshness_check_date", ""),
                "fact_id": fact.get("fact_id", ""),
            }
        )
    return rows


def load_facts(root: Path) -> list[Mapping[str, Any]]:
    path = root / "facts_registry.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    snapshot = load_json(root / "kb_release_v3_snapshot.json")
    return [item for item in snapshot.get("facts", []) if isinstance(item, Mapping)]


def load_smoke_summaries(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for brand in ("FOTON", "UNPK"):
        path = root / brand / "stage6_eval_summary.json"
        if not path.exists():
            path = root / brand.lower() / "stage6_eval_summary.json"
        result[brand.lower()] = load_json(path) if path.exists() else {}
    result["short_status"] = (
        f"FOTON rows={result.get('foton', {}).get('rows_total', 0)}, "
        f"UNPK rows={result.get('unpk', {}).get('rows_total', 0)}, "
        f"errors={result.get('foton', {}).get('errors', 0) + result.get('unpk', {}).get('errors', 0)}, "
        f"brand_violations={result.get('foton', {}).get('brand_separation_violation', 0) + result.get('unpk', {}).get('brand_separation_violation', 0)}"
    )
    return result


def prepare_output_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if "stable_runtime" in resolved.parts:
        raise ValueError("Distribution packs must not be written under stable_runtime")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def extract_brand_forbidden(brand_rules: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    mentions = brand_rules.get("forbidden_client_mentions")
    if not isinstance(mentions, Mapping):
        return {"foton": (), "unpk": ()}
    result: dict[str, tuple[str, ...]] = {}
    for brand in ("foton", "unpk"):
        block = mentions.get(f"when_active_brand_is_{brand}")
        terms = block.get("blocked_terms") if isinstance(block, Mapping) else ()
        if isinstance(terms, Sequence) and not isinstance(terms, (str, bytes, bytearray)):
            result[brand] = tuple(str(term) for term in terms if str(term).strip())
        else:
            result[brand] = ()
    return result


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def truthy(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "да"}
    return False


def clean_text(value: Any, *, limit: int = 1000) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


if __name__ == "__main__":
    raise SystemExit(main())
