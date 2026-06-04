#!/usr/bin/env python3
"""Отбор сбалансированного разнообразного набора из пула вопросов. + печать для ручной проверки."""
from __future__ import annotations
import json, os, re, sys

POOL = os.environ.get("WQ_POOL", "/tmp/wadb/wq_pool.json")
SEL = os.environ.get("WQ_SEL", "/tmp/wadb/wq_selected.json")

CAPS = {
    "foton": {"олимпиады": 11, "расписание": 14, "документы_справка_вычет": 15, "лагерь_лвш": 15,
              "формат_онлайн_очно": 16, "цена_оплата": 16, "программа_уровень": 16},
    "unpk":  {"олимпиады": 13, "расписание": 13, "документы_справка_вычет": 15, "лагерь_лвш": 15,
              "формат_онлайн_очно": 15, "цена_оплата": 15, "программа_уровень": 16},
}
ORDER = ["олимпиады", "лагерь_лвш", "документы_справка_вычет", "формат_онлайн_очно",
         "расписание", "цена_оплата", "программа_уровень"]


def coarse(q: str) -> str:
    return re.sub(r"[^а-яё0-9]", "", q.lower())[:22]


def pick(items, cap):
    # доп. дедуп по первым 22 символам, затем равномерный разброс по длине
    seen, uniq = set(), []
    for it in items:
        k = coarse(it["q"])
        if k in seen:
            continue
        seen.add(k); uniq.append(it)
    if len(uniq) <= cap:
        return uniq
    uniq.sort(key=lambda x: x["len"])  # разброс по длине
    step = len(uniq) / cap
    return [uniq[int(i * step)] for i in range(cap)]


def main():
    pool = json.load(open(POOL, encoding="utf-8"))
    selected = []
    for brand in ("foton", "unpk"):
        for topic in ORDER:
            items = pool.get(brand, {}).get(topic, [])
            chosen = pick(items, CAPS[brand][topic])
            for it in chosen:
                selected.append({"brand": brand, "topic": topic, "q": it["q"], "flag_name": it["flag_name"]})
    json.dump(selected, open(SEL, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # печать для ручной проверки
    n = 0
    cur = None
    for s in selected:
        key = (s["brand"], s["topic"])
        if key != cur:
            cur = key
            print(f"\n### {s['brand']} / {s['topic']}")
        n += 1
        mark = " ⚑" if s["flag_name"] else ""
        print(f"{n:3d}.{mark} {s['q']}")
    fo = sum(1 for s in selected if s["brand"] == "foton")
    un = sum(1 for s in selected if s["brand"] == "unpk")
    print(f"\nИТОГО: {len(selected)} (foton {fo}, unpk {un}); ⚑ к проверке: {sum(1 for s in selected if s['flag_name'])}")
    print("Сохранено:", SEL)


if __name__ == "__main__":
    main()
