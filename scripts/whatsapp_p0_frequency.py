#!/usr/bin/env python3
"""P0-частота по бренду и месяцу + базовая доля P0. Read-only, детерминированно.

Вход для калибровки Волны 1a: насколько редки реальные P0 среди клиентских сообщений,
как они распределены по брендам и сезону. Маркеры — те же выверенные регексы, что в
whatsapp_analytics.py (границы слова, без переловли «суд» внутри «обсудим»).
"""
from __future__ import annotations
import sqlite3, os, re, json
from collections import Counter, defaultdict

DB = os.environ.get("MANGO_DB", "/tmp/wadb/whatsapp_chats.sqlite")
OUT = os.environ.get("MANGO_REPORT_DIR", "/tmp/wadb/reports")
DATE = "2026-05-29"

P0 = {
    "refund": re.compile(
        r"верн\w*\s+деньг|вернуть\s+деньг|требую\s+возврат|хочу\s+возврат|деньги\s+назад|"
        r"расторг\w*\s+договор|расторгнуть|вернуть\s+оплат|возврат\w*\s+(?:денег|средств|оплат)", re.I),
    "legal": re.compile(
        r"\bв\s+суд\b|\bсуд(?:\b|е\b|а\b|ы\b)|прокуратур|роспотреб|претензи|адвокат|юрист|незаконн|"
        r"защит\w*\s+прав\s+потреб", re.I),
    "complaint": re.compile(
        r"жалоб|пожалуюсь|\bобман|мошенн|недовол|возмущ|ужасн\w*\s+(?:обслуж|отнош|качеств)|"
        r"плохо\s+(?:учит|объясн|провел)|некомпетент", re.I),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); cur = con.cursor()
    rows = cur.execute("""SELECT m.ts, m.text, ch.brand_hint
                          FROM messages m JOIN chats ch ON ch.chat_id=m.chat_id
                          WHERE m.role='client' AND m.is_service_message=0""").fetchall()
    total_client = len(rows)
    by_month = defaultdict(Counter)     # month -> P0 class counts
    by_brand = defaultdict(Counter)     # brand -> P0 class counts
    any_p0 = 0
    brand_client = Counter()
    for ts, text, brand in rows:
        mo = ts[:7] if ts and len(ts) >= 7 else "—"
        b = brand if brand in ("foton", "unpk", "mixed") else "null"
        brand_client[b] += 1
        hit = False
        for cls, pat in P0.items():
            if pat.search(text or ""):
                by_month[mo][cls] += 1
                by_brand[b][cls] += 1
                hit = True
        if hit:
            any_p0 += 1
    con.close()

    months = [m for m in sorted(by_month) if m != "—"]
    base_rate = round(100 * any_p0 / max(1, total_client), 3)

    with open(f"{OUT}/whatsapp_p0_frequency_by_brand_season_{DATE}.md", "w", encoding="utf-8") as f:
        f.write("# Частота P0 по бренду и сезону. Claude #2, %s\n\n" % DATE)
        f.write("Детерминированный подсчёт по клиентским сообщениям. Вход для калибровки порогов Волны 1a.\n\n")
        f.write(f"## Базовая доля P0\n\nКлиентских сообщений всего: **{total_client}**. "
                f"С хотя бы одним P0-маркером: **{any_p0}** = **{base_rate}%**.\n\n")
        f.write("> Вывод для Волны 1a: P0 — редкое событие. При такой базовой доле детектор с высокой "
                "полнотой даст много ложных срабатываний, если точность маркеров низкая. "
                "Поэтому маркеры P0 должны быть точными (границы слова), а пограничные случаи — на менеджера.\n\n")
        f.write("## P0 по бренду (бренд чата)\n\n| бренд | клиентских сообщений | refund | complaint | legal |\n|---|---|---|---|---|\n")
        for b in ("foton", "unpk", "mixed", "null"):
            c = by_brand[b]
            f.write(f"| {b} | {brand_client[b]} | {c['refund']} | {c['complaint']} | {c['legal']} |\n")
        f.write("\n> Бренды не смешиваются: это внутренняя статистика, не клиентский ответ.\n\n")
        f.write("## P0 по месяцам\n\n| месяц | refund | complaint | legal | всего P0 |\n|---|---|---|---|---|\n")
        for m in months:
            c = by_month[m]
            tot = c["refund"] + c["complaint"] + c["legal"]
            if tot == 0:
                continue
            f.write(f"| {m} | {c['refund']} | {c['complaint']} | {c['legal']} | {tot} |\n")

    print(json.dumps({"total_client": total_client, "any_p0": any_p0, "base_rate_pct": base_rate,
                      "by_brand": {b: dict(by_brand[b]) for b in ("foton", "unpk", "mixed", "null")},
                      "months_with_p0": len(months)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
