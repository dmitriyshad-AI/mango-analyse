#!/usr/bin/env python3
"""Собрать real_questions_20260531.jsonl: simulator_spec + judge_spec + персоны.

Переиспользует спеки из targeted_riskzones (targeted_pravka5 в проекте нет).
Добивает маскировку (2 пропущенных имени), чистит служебные плейсхолдеры,
выкидывает вырожденные не-вопросы. Персона — по шаблону ТЗ.
"""
from __future__ import annotations
import json, os, re

SRC_SPECS = "product_data/telegram_dynamic_test_sets/targeted_riskzones_2026_05_26.jsonl"
SEL = os.environ.get("WQ_SEL", "/tmp/wadb/wq_selected.json")
OUT = os.environ.get("WQ_OUT", "/tmp/wadb/real_questions_20260531.jsonl")

SLUG = {
    "олимпиады": ("olymp", "олимпиады"),
    "лагерь_лвш": ("camp", "лагерь/ЛВШ"),
    "документы_справка_вычет": ("docs", "документы/справка/вычет"),
    "формат_онлайн_очно": ("format", "формат онлайн/очно"),
    "расписание": ("schedule", "расписание"),
    "цена_оплата": ("price", "цена/оплата"),
    "программа_уровень": ("program", "программа/уровень"),
}
FOTON_FORBIDDEN = ["УНПК", "МФТИ", "kmipt", "Сретенка", "Пацаева", "Долгопрудный"]
UNPK_FORBIDDEN = ["Фотон", "ЦДПО", "ЦРДО", "Скорняжный", "cdpofoton", "Долями", "Т-Банк"]

# Финальная ручная маскировка пропущенных имён (вычитано глазами).
MANUAL_NAME = {"Вадиму": "[имя]", "Романом": "[имя]"}

# Вырожденные не-вопросы / слишком контекст-зависимые — выкинуть (по сигнатуре).
DROP_SIGNATURES = [
    "Вот по этому?",
    "Да, пришлите, пожалуйста, ссылку",
    "Если другого времени нет- убираем",
    'удобен?" Суббота',
    'расписание." В Долгопрудном',
]


def clean(q: str) -> str:
    q = q.replace("The media is missing", " ").replace("the media is missing", " ")
    for a, b in MANUAL_NAME.items():
        q = re.sub(rf"\b{a}\b", b, q)
    return re.sub(r"\s+", " ", q).strip()


def main():
    sim = judge = None
    with open(SRC_SPECS, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("type") == "simulator_spec" and sim is None:
                sim = o
            elif o.get("type") == "judge_spec" and judge is None:
                judge = o
    assert sim and judge, "не нашёл simulator_spec/judge_spec"

    selected = json.load(open(SEL, encoding="utf-8"))
    personas = []
    seq = {}
    dropped = 0
    for s in selected:
        q = clean(s["q"])
        if any(sig in q for sig in DROP_SIGNATURES) or len(q) < 12:
            dropped += 1
            continue
        if "[имя]" in q and len(re.sub(r"\[имя\]", "", q).strip()) < 12:
            dropped += 1  # остался по сути только маркер имени
            continue
        brand = s["brand"]; topic = s["topic"]
        slug, goal = SLUG[topic]
        k = (brand, slug); seq[k] = seq.get(k, 0) + 1
        personas.append({
            "type": "persona",
            "brand": brand,
            "category": "real_question",
            "dialog_id": f"real_{brand}_{slug}_{seq[k]:02d}",
            "max_turns": 2,
            "expected_route": "bot_answer_self",
            "persona": "real client question",
            "goal": goal,
            "held_facts": {},
            "behaviors": [q],
            "success_criteria": "ответил по существу из факта, если факт есть; ушёл к менеджеру только если факта нет/P0",
            "fail_criteria": "ушёл к менеджеру при наличии факта; выдумал; смешал бренды",
            "brand_forbidden": FOTON_FORBIDDEN if brand == "foton" else UNPK_FORBIDDEN,
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(sim, ensure_ascii=False) + "\n")
        fh.write(json.dumps(judge, ensure_ascii=False) + "\n")
        for p in personas:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    from collections import Counter
    byb = Counter(p["brand"] for p in personas)
    bytopic = Counter((p["brand"], p["goal"]) for p in personas)
    print(f"Выкинуто вырожденных: {dropped}")
    print(f"Персон всего: {len(personas)} (foton {byb['foton']}, unpk {byb['unpk']})")
    print("Разбивка бренд×тема:")
    for (b, g), n in sorted(bytopic.items()):
        print(f"  {b:5s} {g:26s} {n}")
    # самопроверка ПДн в итоговом файле
    import re as _re
    txt = open(OUT, encoding="utf-8").read()
    behav = " ".join(json.loads(l)["behaviors"][0] for l in txt.splitlines() if json.loads(l).get("type") == "persona")
    phones = _re.findall(r"\+?\d[\d\s\-()]{8,}\d", behav)
    emails = _re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", behav)
    print(f"\nПДн-самопроверка behaviors: телефонов={len(phones)}, email={len(emails)}, плейсхолдеров [имя]={behav.count('[имя]')}")
    print("Файл:", OUT)


if __name__ == "__main__":
    main()
