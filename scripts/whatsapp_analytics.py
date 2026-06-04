#!/usr/bin/env python3
"""Сезонная аналитика WhatsApp (Этап 4, детерминированная часть). Read-only.

Генерирует markdown-отчёты из whatsapp_chats.sqlite БЕЗ LLM (только подсчёты по
реальным сообщениям — без выдумки):
- whatsapp_seasonal_topics  (темы по месяцам/брендам, фокус июнь-окт 2024/2025)
- whatsapp_real_p0_phrasings (реальные формулировки претензий клиентов + сверка)
- whatsapp_brand_leaks_managers (статистика бренд-смешения в исходящих)
Остальные два отчёта (сравнение с звонками, тон менеджеров) — следующая сессия.
"""
from __future__ import annotations
import sqlite3, os, re, json
from collections import Counter, defaultdict

DB = os.environ.get("MANGO_DB", "/tmp/wadb/whatsapp_chats.sqlite")
OUT = os.environ.get("MANGO_REPORT_DIR", "/tmp/wadb/reports")
DATE = "2026-05-29"

TOPIC_MARKERS = (
    ("цена/оплата/рассрочка", ("цена", "стоим", "сколько стоит", "рассроч", "оплат", "скидк")),
    ("расписание/дни", ("расписан", "по каким дням", "во сколько", "когда занят", "время занят")),
    ("лагерь/ЛВШ/смены", ("лвш", "лагер", "смен", "менделеево", "выездн", "прожив", "питан")),
    ("формат онлайн/очно", ("онлайн", "очно", "вебинар", "дистанц")),
    ("пробное/запись/заявка", ("пробн", "записать", "заявк", "оформ", "как записа")),
    ("документы/справка/вычет", ("справк", "договор", "документ", "вычет", "маткап", "квитанц")),
    ("олимпиады/Физтех", ("олимпиад", "физтех", "перечнев", "столичн")),
    ("возврат/претензия", ("возврат", "верн", "претензи", "жалоб")),
)
# Регексы с границами слова, чтобы «суд» не ловило «обсудить/посуда» и т.п.
P0_MARKERS = {
    "refund (возврат/требование)": re.compile(
        r"верн\w*\s+деньг|вернуть\s+деньг|требую\s+возврат|хочу\s+возврат|деньги\s+назад|"
        r"расторг\w*\s+договор|расторгнуть|вернуть\s+оплат|возврат\w*\s+(?:денег|средств|оплат)", re.I),
    "legal (юр.угроза)": re.compile(
        r"\bв\s+суд\b|\bсуд(?:\b|е\b|а\b|ы\b)|прокуратур|роспотреб|претензи|адвокат|юрист|незаконн|"
        r"защит\w*\s+прав\s+потреб", re.I),
    "complaint (жалоба/обман)": re.compile(
        r"жалоб|пожалуюсь|\bобман|мошенн|недовол|возмущ|ужасн\w*\s+(?:обслуж|отнош|качеств)|"
        r"плохо\s+(?:учит|объясн|провел)|некомпетент", re.I),
    "payment_dispute (спор оплаты)": re.compile(
        r"дважды\s+списал|списали\s+дважды|двойн\w*\s+списан|не\s+списал|оспор\w*\s+(?:плат|операц)|"
        r"чарджб|ошибочно\s+списал", re.I),
}


def month_of(ts: str) -> str:
    return ts[:7] if ts and len(ts) >= 7 else ""


def main():
    os.makedirs(OUT, exist_ok=True)
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); cur = con.cursor()

    # ---------- 1. СЕЗОННЫЕ ТЕМЫ ----------
    rows = cur.execute("SELECT m.ts, m.role, m.brand_hint, m.text, ch.brand_hint FROM messages m JOIN chats ch ON ch.chat_id=m.chat_id WHERE m.is_service_message=0").fetchall()
    topics_by_month = defaultdict(Counter)        # месяц -> темы (клиентские)
    topics_by_brand = defaultdict(Counter)
    msgs_by_month = Counter()
    for ts, role, mb, text, cb in rows:
        mo = month_of(ts); low = (text or "").lower()
        if not mo:
            continue
        msgs_by_month[mo] += 1
        if role == "client":
            for name, ms in TOPIC_MARKERS:
                if any(x in low for x in ms):
                    topics_by_month[mo][name] += 1
                    if cb in ("foton", "unpk"):
                        topics_by_brand[cb][name] += 1
    focus = [m for m in sorted(topics_by_month) if (("2024-06" <= m <= "2024-10") or ("2025-06" <= m <= "2025-10"))]
    with open(f"{OUT}/whatsapp_seasonal_topics_{DATE}.md", "w", encoding="utf-8") as f:
        f.write("# Сезонные темы WhatsApp (клиентские сообщения). Claude #2, %s\n\n" % DATE)
        f.write("Детерминированный подсчёт по маркерам тем в КЛИЕНТСКИХ сообщениях (без LLM, без выдумки). Бренды не смешиваются: разбивка по бренду чата.\n\n")
        f.write("## Фокус: июнь–октябрь 2024 и 2025\n\n| месяц | всего сообщений | топ-темы (клиент) |\n|---|---|---|\n")
        for m in focus:
            top = ", ".join(f"{k} ({v})" for k, v in topics_by_month[m].most_common(4))
            f.write(f"| {m} | {msgs_by_month[m]} | {top or '—'} |\n")
        f.write("\n## Темы по бренду (весь период, клиентские)\n\n")
        for b in ("foton", "unpk"):
            f.write(f"### {b}\n\n")
            for k, v in topics_by_brand[b].most_common(8):
                f.write(f"- {k}: {v}\n")
            f.write("\n")
        f.write("> Примечание: 2024 (июнь-окт) — единственный канал без звонков; 2025 — есть параллель со звонками (сравнение — отдельный отчёт).\n")

    # ---------- 2. РЕАЛЬНЫЕ P0-ФОРМУЛИРОВКИ ----------
    p0_examples = defaultdict(list); p0_counts = Counter()
    for ts, role, mb, text, cb in rows:
        if role != "client":
            continue
        for cls, pat in P0_MARKERS.items():
            if pat.search(text or ""):
                p0_counts[cls] += 1
                if len(p0_examples[cls]) < 12:
                    snippet = re.sub(r"\s+", " ", text).strip()[:160]
                    p0_examples[cls].append(snippet)
    with open(f"{OUT}/whatsapp_real_p0_phrasings_{DATE}.md", "w", encoding="utf-8") as f:
        f.write("# Реальные P0-формулировки клиентов в WhatsApp. Claude #2, %s\n\n" % DATE)
        f.write("Извлечено из КЛИЕНТСКИХ сообщений по маркерам. Сверка с `p0_recall_spec.py` — материал для калибровки Волны 1a.\n\n")
        for cls in P0_MARKERS:
            f.write(f"## {cls} — найдено {p0_counts[cls]}\n\n")
            for ex in p0_examples[cls]:
                f.write(f"- «{ex}»\n")
            f.write("\n")
        f.write("## Вывод для Волны 1a\n\nРеальные обороты сверить с маркерами `REFUND_RE/LEGAL_RE/COMPLAINT_RE/PAYMENT_DISPUTE_RE`; новые формулировки, не покрытые регексами, — кандидаты в held-out gold (см. TZ_wave_1a_FINAL_v2).\n")

    # ---------- 3. БРЕНД-СМЕШЕНИЕ ----------
    brand_dist = Counter(r[0] for r in cur.execute("SELECT brand_hint FROM chats"))
    mixed_examples = []
    for cid, in cur.execute("SELECT chat_id FROM chats WHERE brand_hint='mixed' LIMIT 15"):
        txt = " ".join(r[0] for r in cur.execute("SELECT text FROM messages WHERE chat_id=? AND role='manager' AND is_service_message=0", (cid,)))
        low = txt.lower()
        u = [m for m in ("унпк", "мфти", "менделеево", "kmipt", "сретенк", "пацаев", "долгопрудн") if m in low]
        ftn = [m for m in ("фотон", "цдпо", "црдо", "скорняжн", "cdpofoton", "долями") if m in low]
        mixed_examples.append((cid, u, ftn))
    with open(f"{OUT}/whatsapp_brand_leaks_managers_{DATE}.md", "w", encoding="utf-8") as f:
        f.write("# Бренд-смешение в исходящих WhatsApp менеджеров. Claude #2, %s\n\n" % DATE)
        f.write("Чаты, где менеджер в переписке употребил маркеры ОБОИХ брендов. Материал для тренинга команды (CLAUDE.md: бренды не смешиваются в клиентском ответе).\n\n")
        f.write("## Распределение бренд-разметки чатов\n\n")
        for k in ("unpk", "foton", "mixed", None):
            f.write(f"- {k}: {brand_dist.get(k, 0)}\n")
        f.write(f"\n**Бренд-смешение (mixed): {brand_dist.get('mixed', 0)} чатов** — менеджер в одном диалоге упоминал оба бренда (адреса/контакты/площадки).\n\n")
        f.write("Примечание: в выгрузке все менеджеры обезличены как `You` — атрибуция к конкретному менеджеру невозможна без доп.данных.\n\n")
        f.write("## Примеры mixed-чатов (какие маркеры обоих брендов встретились)\n\n| chat_id | маркеры УНПК | маркеры Фотон |\n|---|---|---|\n")
        for cid, u, ftn in mixed_examples:
            f.write(f"| {cid} | {', '.join(u) or '—'} | {', '.join(ftn) or '—'} |\n")

    con.close()
    print(json.dumps({"brand_dist": {str(k): v for k, v in brand_dist.items()},
                      "p0_counts": dict(p0_counts),
                      "focus_months": focus,
                      "reports_dir": OUT}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
