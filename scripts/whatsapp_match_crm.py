#!/usr/bin/env python3
"""Матчинг WhatsApp-чатов с CRM по телефону (Этап 2). Read-only по CRM.

Источник истины — master_contacts_ru.csv (колонка «Телефон клиента»).
chat_id у 98% чатов = телефон, поэтому матчим по нормализованному номеру.
confidence: high = телефон + ФИО из CRM найдено в тексте чата; medium = только телефон.
"""
from __future__ import annotations
import csv, sqlite3, os, re, sys, json
from datetime import datetime, timezone

DB = os.environ.get("MANGO_DB", "/tmp/wadb/whatsapp_chats.sqlite")
MASTER = os.environ.get("MANGO_MASTER_CONTACTS")
OUT_DIR = os.environ.get("MANGO_OUT_DIR", "/tmp/wadb")
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

PHONE_COL = "Телефон клиента"
NAME_COLS = ("ФИО ребенка", "ФИО родителя", "ФИО родителя Tallanto")
TALLANTO_COL = "ID Tallanto"
BRANCH_COL = "Филиал Tallanto"


def norm_phone(s: str) -> str:
    d = re.sub(r"\D", "", s or "")
    if len(d) == 11 and d.startswith("8"):
        d = "7" + d[1:]
    if len(d) == 10:
        d = "7" + d
    return "+" + d if d else ""


def name_tokens(*names) -> set[str]:
    toks = set()
    for n in names:
        for t in re.split(r"\s+", (n or "").strip().lower()):
            if len(t) >= 4:
                toks.add(t)
    return toks


def main():
    if not MASTER or not os.path.exists(MASTER):
        print("MASTER not found:", MASTER); sys.exit(1)
    crm: dict[str, dict] = {}
    with open(MASTER, encoding="utf-8-sig", newline="") as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            ph = norm_phone(row.get(PHONE_COL, ""))
            if not ph:
                continue
            crm[ph] = {
                "tallanto_id": (row.get(TALLANTO_COL) or "").strip(),
                "branch": (row.get(BRANCH_COL) or "").strip(),
                "names": name_tokens(*[row.get(c, "") for c in NAME_COLS]),
            }
    print("CRM контактов с телефоном:", len(crm))

    con = sqlite3.connect(DB); cur = con.cursor()
    chats = cur.execute("SELECT chat_id, client_phone FROM chats").fetchall()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    matched = 0; high = 0; review_rows = []
    cur.execute("DELETE FROM crm_match")
    for cid, ph in chats:
        ph = ph or norm_phone(cid)
        info = crm.get(ph)
        if not info:
            continue
        matched += 1
        # текст чата для сверки имени
        txt = " ".join(r[0] for r in cur.execute(
            "SELECT text FROM messages WHERE chat_id=? AND is_service_message=0", (cid,)).fetchall()).lower()
        conf = "medium"; basis = "phone"
        if info["names"] and any(t in txt for t in info["names"]):
            conf = "high"; basis = "phone+name"; high += 1
        cur.execute(
            "INSERT OR REPLACE INTO crm_match(chat_id,primary_phone,tallanto_id,match_confidence,match_basis,matched_at) VALUES(?,?,?,?,?,?)",
            (cid, ph, info["tallanto_id"], conf, basis, now))
        cur.execute("UPDATE chats SET status='matched' WHERE chat_id=?", (cid,))
        if conf != "high":
            review_rows.append({"chat_id": cid, "primary_phone": ph,
                                 "tallanto_id": info["tallanto_id"], "branch": info["branch"],
                                 "match_confidence": conf, "match_basis": basis})
    con.commit()
    total = len(chats)
    report = {"whatsapp_chats": total, "crm_contacts": len(crm),
              "matched": matched, "matched_high": high, "matched_medium": matched - high,
              "unmatched": total - matched,
              "match_ratio": round(matched / total, 4) if total else 0}
    con.close()
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(report, open(os.path.join(OUT_DIR, "whatsapp_crm_match_report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    review_path = os.path.join(OUT_DIR, "whatsapp_crm_match_review.csv")
    if review_rows:
        with open(review_path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(review_rows[0].keys())); w.writeheader(); w.writerows(review_rows[:5000])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
