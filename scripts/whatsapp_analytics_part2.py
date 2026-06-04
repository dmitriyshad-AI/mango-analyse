#!/usr/bin/env python3
"""Этап 4, часть 2 (детерминированная). Read-only.

Два оставшихся отчёта:
- whatsapp_channel_comparison  — объём WhatsApp (клиентские сообщения) vs звонки
  по месяцам + пересечение клиентов по телефону. Показывает, где WhatsApp —
  ЕДИНСТВЕННЫЙ сигнал (2024, до запуска телефонии) и сколько контактов уникальны.
- whatsapp_manager_tone        — тон исходящих менеджеров (длина, emoji, приветствия,
  вопросы, «спасибо/пожалуйста») для калибровки черновиков X2. Разбивка по бренду чата.

Звонки и WhatsApp НЕ смешиваются по бренду в клиентских ответах — это внутренняя
аналитика. Бренд берётся только со стороны WhatsApp (где размечен однозначно).
"""
from __future__ import annotations
import sqlite3, os, re, csv, sys, json
from collections import Counter, defaultdict

DB = os.environ.get("MANGO_DB", "/tmp/wadb/whatsapp_chats.sqlite")
CALLS = os.environ.get("MANGO_CALLS", "")
OUT = os.environ.get("MANGO_REPORT_DIR", "/tmp/wadb/reports")
DATE = "2026-05-29"
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿←-⇿⬀-⯿]")
GREET = re.compile(r"^(здравствуй|добр(ый|ое|ый день|ое утро|ый вечер)|привет|доброго)", re.I)
THANKS = re.compile(r"спасибо|благодар", re.I)
PLEASE = re.compile(r"пожалуйста|пож\.", re.I)


def norm_phone(s: str) -> str:
    d = re.sub(r"\D", "", s or "")
    if len(d) == 11 and d.startswith("8"):
        d = "7" + d[1:]
    if len(d) == 10:
        d = "7" + d
    return "+" + d if d else ""


def main():
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); cur = con.cursor()

    # ---------- WhatsApp: клиентские сообщения по месяцам + телефоны ----------
    wa_month_client = Counter()
    for ts, in cur.execute("SELECT ts FROM messages WHERE role='client' AND is_service_message=0"):
        mo = ts[:7] if ts and len(ts) >= 7 else ""
        if mo:
            wa_month_client[mo] += 1
    wa_phones = {r[0] for r in cur.execute("SELECT client_phone FROM chats WHERE client_phone!=''")}

    # ---------- Звонки: по месяцам + телефоны ----------
    call_month = Counter(); call_phones = set(); call_min = "9999"; call_max = ""
    if CALLS and os.path.exists(CALLS):
        with open(CALLS, encoding="utf-8-sig", newline="") as fh:
            rd = csv.DictReader(fh)
            for row in rd:
                dt = (row.get("Дата и время звонка") or "").strip()
                mo = dt[:7]
                if mo:
                    call_month[mo] += 1
                    call_min = min(call_min, mo); call_max = max(call_max, mo)
                ph = norm_phone(row.get("Телефон клиента", ""))
                if ph:
                    call_phones.add(ph)
    overlap = wa_phones & call_phones
    wa_only = wa_phones - call_phones

    # объединённый диапазон месяцев для таблицы
    months = sorted(set(wa_month_client) | set(call_month))
    months = [m for m in months if "2024-01" <= m <= "2026-05"]

    with open(f"{OUT}/whatsapp_channel_comparison_{DATE}.md", "w", encoding="utf-8") as f:
        f.write("# Сравнение каналов: WhatsApp vs звонки. Claude #2, %s\n\n" % DATE)
        f.write("Детерминированный подсчёт по месяцам. Внутренняя аналитика (не клиентский ответ), бренды не смешиваются: бренд — только со стороны WhatsApp.\n\n")
        f.write(f"Диапазон звонков в выгрузке: **{call_min}–{call_max}**. ")
        f.write("До старта телефонии WhatsApp — единственный письменный след клиента.\n\n")
        f.write("## Объём по месяцам\n\n| месяц | WhatsApp (клиентских сообщений) | звонков | комментарий |\n|---|---|---|---|\n")
        for m in months:
            wa = wa_month_client.get(m, 0); ca = call_month.get(m, 0)
            note = ""
            if ca == 0 and wa > 0:
                note = "только WhatsApp"
            elif wa > 0 and ca > 0:
                note = "оба канала"
            f.write(f"| {m} | {wa} | {ca} | {note} |\n")
        f.write("\n## Пересечение клиентов по телефону\n\n")
        f.write(f"- WhatsApp-телефонов: **{len(wa_phones)}**\n")
        f.write(f"- Телефонов в звонках: **{len(call_phones)}**\n")
        f.write(f"- Пересечение (есть и там, и там): **{len(overlap)}**\n")
        f.write(f"- Только в WhatsApp (нет ни одного звонка): **{len(wa_only)}** "
                f"({round(100*len(wa_only)/max(1,len(wa_phones)))}% WhatsApp-клиентов)\n\n")
        f.write("**Вывод для timeline:** WhatsApp добавляет письменный контекст там, где звонков нет "
                "(весь 2024 и значимая доля клиентов без единого звонка). Это не дубль телефонии, а "
                "самостоятельный источник для customer timeline.\n")

    # ---------- Тон менеджеров ----------
    rows = cur.execute("""SELECT m.text, ch.brand_hint FROM messages m JOIN chats ch ON ch.chat_id=m.chat_id
                          WHERE m.role='manager' AND m.is_service_message=0""").fetchall()
    # Плейсхолдеры экспорта WhatsApp — не контент менеджера, исключаем из тона.
    PLACEHOLDER = re.compile(r"^(the media is missing|<media omitted>|this message was deleted|"
                             r"you deleted this message|null|image omitted|video omitted|"
                             r"audio omitted|sticker omitted|gif omitted)\b", re.I)
    REPLY_PREFIX = re.compile(r"^replying to this message\s*", re.I)
    skipped_placeholder = 0
    agg = defaultdict(lambda: {"n": 0, "chars": 0, "emoji": 0, "q": 0, "greet": 0, "thanks": 0, "please": 0})
    openings = defaultdict(Counter)
    for text, brand in rows:
        t = REPLY_PREFIX.sub("", (text or "").strip()).strip()
        if not t or PLACEHOLDER.match(t):
            skipped_placeholder += 1
            continue
        key = brand if brand in ("foton", "unpk") else "all"
        for k in (key, "ALL"):
            a = agg[k]
            a["n"] += 1; a["chars"] += len(t)
            if EMOJI.search(t): a["emoji"] += 1
            if "?" in t: a["q"] += 1
            if GREET.search(t): a["greet"] += 1
            if THANKS.search(t): a["thanks"] += 1
            if PLEASE.search(t): a["please"] += 1
        first = re.sub(r"\s+", " ", t)[:32]
        openings[key][first] += 1

    def pct(x, n): return f"{round(100*x/max(1,n))}%"

    with open(f"{OUT}/whatsapp_manager_tone_{DATE}.md", "w", encoding="utf-8") as f:
        f.write("# Тон менеджеров в WhatsApp (исходящие). Claude #2, %s\n\n" % DATE)
        f.write("Детерминированные метрики по сообщениям менеджера (`You`). Материал для калибровки тона черновиков X2: бот должен попадать в привычную клиенту манеру.\n\n")
        f.write("Все менеджеры в выгрузке обезличены как `You` — метрики агрегатные, без атрибуции к человеку.\n\n")
        f.write("## Метрики\n\n| срез | сообщений | сред. длина (симв.) | с emoji | с вопросом | приветствие | «спасибо» | «пожалуйста» |\n|---|---|---|---|---|---|---|---|\n")
        for k, label in (("ALL", "все"), ("unpk", "УНПК"), ("foton", "Фотон"), ("all", "без бренда")):
            a = agg.get(k)
            if not a or a["n"] == 0:
                continue
            f.write(f"| {label} | {a['n']} | {round(a['chars']/a['n'])} | {pct(a['emoji'],a['n'])} | "
                    f"{pct(a['q'],a['n'])} | {pct(a['greet'],a['n'])} | {pct(a['thanks'],a['n'])} | {pct(a['please'],a['n'])} |\n")
        f.write("\n## Частые начала сообщений (по бренду)\n\n")
        for k, label in (("unpk", "УНПК"), ("foton", "Фотон")):
            f.write(f"### {label}\n\n")
            for phrase, n in openings[k].most_common(8):
                f.write(f"- «{phrase}…» — {n}\n")
            f.write("\n")
        f.write("## Вывод для калибровки X2\n\n")
        f.write("Черновики бота держать в этой манере: приветствие в начале нового диалога, вежливые «пожалуйста/спасибо», "
                "умеренные emoji, короткие сообщения. Точные пороги (длина, доля emoji) — в таблице выше; "
                "бренд-срезы не смешивать между собой.\n")

    con.close()
    print(json.dumps({
        "call_range": [call_min, call_max], "call_total_months": len(call_month),
        "wa_phones": len(wa_phones), "call_phones": len(call_phones),
        "overlap": len(overlap), "wa_only": len(wa_only),
        "manager_msgs_total": agg["ALL"]["n"],
        "skipped_placeholder": skipped_placeholder,
        "reports_dir": OUT,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
