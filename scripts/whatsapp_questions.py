#!/usr/bin/env python3
"""Частые реальные ВОПРОСЫ клиентов в WhatsApp по темам и бренду. Read-only.

Для поиска дыр в KB: что клиенты реально спрашивают. Берём только клиентские
сообщения-вопросы (есть «?» или вопросительный зачин). Темы — расширенный набор.
"""
from __future__ import annotations
import sqlite3, os, re, json
from collections import Counter, defaultdict

DB = os.environ.get("MANGO_DB", "/tmp/wadb/whatsapp_chats.sqlite")

QSTART = re.compile(r"^(как|какой|какая|какие|сколько|когда|где|можно ли|можно|есть ли|"
                    r"нужно ли|надо ли|почему|что|а\s|подскаж|уточн|правда ли|будет ли)", re.I)

TOPICS = (
    ("цена/оплата", ("цена", "стоим", "сколько стоит", "оплат", "руб", "ценник")),
    ("рассрочка/долями", ("рассроч", "долями", "частями", "помесячно", "в кредит")),
    ("скидки", ("скидк", "дешевл", "акци", "льгот", "многодетн")),
    ("расписание/время", ("расписан", "по каким дням", "во сколько", "когда занят", "время")),
    ("формат онлайн/очно", ("онлайн", "очно", "дистанц", "вебинар")),
    ("пробное/запись", ("пробн", "записать", "как записа", "заявк", "оформ")),
    ("документы/справка/вычет", ("справк", "вычет", "договор", "квитанц", "маткап", "чек")),
    ("лагерь/ЛВШ/смены", ("лвш", "лагер", "смен", "менделеево", "прожив", "питан", "путёвк")),
    ("преподаватели", ("преподавател", "учител", "педагог", "кто ведёт", "кто ведет")),
    ("программа/уровень", ("программ", "уровень", "класс", "подойдёт", "подойдет", "сложн", "групп")),
    ("олимпиады/Физтех", ("олимпиад", "физтех", "перечнев", "поступл", "егэ", "огэ")),
    ("возврат/перенос", ("возврат", "вернуть", "перенос", "пропуск", "отработ", "заморозк")),
    ("результат/гарантии", ("гаранти", "результат", "поступит ли", "балл")),
    ("контакты/адрес", ("адрес", "как добраться", "метро", "телефон", "позвонить", "где наход")),
)


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); cur = con.cursor()
    rows = cur.execute("""SELECT m.text, ch.brand_hint FROM messages m JOIN chats ch ON ch.chat_id=m.chat_id
                          WHERE m.role='client' AND m.is_service_message=0""").fetchall()
    by_topic = Counter()
    by_topic_brand = defaultdict(lambda: defaultdict(Counter))  # topic -> brand -> Counter
    examples = defaultdict(list)
    q_total = 0
    for text, brand in rows:
        t = (text or "").strip()
        low = t.lower()
        is_q = ("?" in t) or bool(QSTART.match(low))
        if not is_q or len(t) < 8:
            continue
        q_total += 1
        b = brand if brand in ("foton", "unpk") else ("mixed" if brand == "mixed" else "null")
        for name, ms in TOPICS:
            if any(x in low for x in ms):
                by_topic[name] += 1
                by_topic_brand[name][b]["n"] += 1
                if len(examples[name]) < 6 and 12 <= len(t) <= 160:
                    examples[name].append((b, re.sub(r"\s+", " ", t)))
    con.close()
    out = {"questions_total": q_total, "by_topic": {}}
    for name, _ in TOPICS:
        out["by_topic"][name] = {
            "total": by_topic[name],
            "by_brand": {b: by_topic_brand[name][b]["n"] for b in ("foton", "unpk", "mixed", "null") if by_topic_brand[name][b]["n"]},
            "examples": examples[name],
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
