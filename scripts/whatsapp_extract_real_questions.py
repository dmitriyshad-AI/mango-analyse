#!/usr/bin/env python3
"""Извлечь реальные клиентские вопросы для прогона бота. Read-only + маскировка ПДн.

Фильтры: чёткий бренд чата (foton/unpk), период 2024-2025, сообщение-вопрос (15-160 симв.),
одна из 7 тем. Маскирует телефоны/email/ФИО/адреса. Дедуплицирует близкие.
Выводит пул JSON (сгруппирован по бренду+теме) в outputs для ручного отбора.
"""
from __future__ import annotations
import sqlite3, os, re, json, sys
from collections import defaultdict, Counter

DB = os.environ.get("MANGO_DB", "/tmp/wadb/whatsapp_chats.sqlite")
if not os.path.exists(DB):
    DB = "product_data/transcripts/whatsapp_chats.sqlite"
OUT = os.environ.get("WQ_POOL", "/tmp/wadb/wq_pool.json")

# Темы в порядке приоритета (специфичные → общие), только 7 нужных.
TOPICS = [
    ("олимпиады", ("олимпиад", "физтех", "перечнев", "всош", "оммо", "росатом", "курчатов")),
    ("лагерь_лвш", ("лвш", "лагер", "смен", "менделеево", "выездн", "прожив", "путёвк", "путевк", "лш ")),
    ("документы_справка_вычет", ("справк", "вычет", "договор", "квитанц", "маткап", "материнск", "чек", "документ", "снилс", "оферт")),
    ("формат_онлайн_очно", ("онлайн", "очно", "дистанц", "вебинар", "формат")),
    ("расписание", ("расписан", "по каким дням", "во сколько", "когда занят", "когда начнут", "когда старт", "время занят", "какие дни", "сколько занятий")),
    ("цена_оплата", ("цена", "стоим", "сколько стоит", "оплат", "руб", "рассроч", "помесячн", "скидк", "долями", "частями")),
    ("программа_уровень", ("программ", "уровень", "класс", "подойдёт", "подойдет", "сложн", "групп", "предмет", "тест", "подготов", "репетит", "курс")),
]
QSTART = re.compile(r"^(как|какой|какая|какие|каком|сколько|когда|где|можно ли|можно|есть ли|"
                    r"нужно ли|надо ли|почему|что|а\s|подскаж|уточн|правда ли|будет ли|подойд)", re.I)

# Слова, которые НЕ маскируем как имя (бренды/места/акронимы в нижнем регистре-хвостом).
WHITELIST = {"Москва", "Долгопрудный", "Менделеево", "Фотон", "Физтех", "Красносельская",
             "Сретенка", "Институтский", "Пацаева", "Скорняжный", "Ховрино", "Питон"}


def mask(t: str) -> str:
    t = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[email]", t)               # email
    t = re.sub(r"\+?\d[\d\s\-()]{8,}\d", "[телефон]", t)                  # телефоны
    # ФИО: 2-3 подряд слова с заглавной кириллицей
    t = re.sub(r"\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){1,2}\b",
               lambda m: m.group(0) if any(w in WHITELIST for w in m.group(0).split()) else "[имя]", t)
    # имя после триггеров (одиночное)
    t = re.sub(r"(зовут|реб[её]нок|ребёнка|ребенка|сын[аоу]?|доч[ьику]+|внук[аи]?|внучк[аиу]+)\s+[А-ЯЁ][а-яё]+",
               r"\1 [имя]", t, flags=re.I)
    # адрес
    t = re.sub(r"(ул\.\s*[А-ЯЁ][а-яё]+|улиц[аеуы]\s+[А-ЯЁ][а-яё]+|пер\.\s*[А-ЯЁ]\w+|просп\w*\s+[А-ЯЁ]\w+)",
               "[адрес]", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def flag_possible_name(t: str) -> bool:
    # одиночное заглавное кириллическое слово не в начале и не в whitelist → возможно имя
    for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b[А-ЯЁ][а-яё]{2,}\b", t):
        w = m.group(0)
        if w in WHITELIST or m.start() == 0:
            continue
        # пропускаем частые не-имена
        if w in {"Здравствуйте", "Добрый", "Подскажите", "Скажите", "Спасибо", "День",
                 "Можно", "Есть", "Как", "Доброе", "Олимпиад", "Если"}:
            continue
        return True
    return False


def topic_of(low: str):
    for name, ms in TOPICS:
        if any(x in low for x in ms):
            return name
    return None


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); cur = con.cursor()
    rows = cur.execute("""SELECT m.text, ch.brand_hint, m.ts
                          FROM messages m JOIN chats ch ON ch.chat_id=m.chat_id
                          WHERE m.role='client' AND m.is_service_message=0
                            AND ch.brand_hint IN ('foton','unpk')
                            AND m.ts >= '2024-01' AND m.ts < '2026-01'""").fetchall()
    con.close()
    pool = defaultdict(lambda: defaultdict(list))   # brand -> topic -> [items]
    seen = defaultdict(set)                          # (brand,topic) -> normalized keys
    counts = Counter()
    raw_total = 0
    for text, brand, ts in rows:
        t = (text or "").strip()
        if not (15 <= len(t) <= 160):
            continue
        low = t.lower()
        if "?" not in t and not QSTART.match(low):
            continue
        topic = topic_of(low)
        if not topic:
            continue
        raw_total += 1
        masked = mask(t)
        if len(masked) < 12:
            continue
        norm = re.sub(r"[^а-яё0-9]", "", masked.lower())[:60]
        if norm in seen[(brand, topic)]:
            continue
        seen[(brand, topic)].add(norm)
        pool[brand][topic].append({"q": masked, "flag_name": flag_possible_name(masked), "len": len(masked)})
        counts[(brand, topic)] += 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(pool, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"Кандидатов после фильтра (до дедупа): {raw_total}")
    print(f"После дедупа всего: {sum(counts.values())}")
    print(f"\n{'тема':28s} | foton | unpk")
    topics = [t for t, _ in TOPICS]
    for tp in topics:
        print(f"{tp:28s} | {counts[('foton',tp)]:5d} | {counts[('unpk',tp)]:4d}")
    print(f"{'ИТОГО':28s} | {sum(counts[('foton',t)] for t in topics):5d} | {sum(counts[('unpk',t)] for t in topics):4d}")
    flagged = sum(1 for b in pool for tp in pool[b] for it in pool[b][tp] if it["flag_name"])
    print(f"\nПомечено flag_name (возможно осталось имя, для ручной проверки): {flagged}")
    print(f"Пул сохранён: {OUT}")


if __name__ == "__main__":
    main()
