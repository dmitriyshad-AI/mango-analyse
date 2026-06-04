#!/usr/bin/env python3
"""Нормализатор all_whatsapp_chats.txt -> whatsapp_chats.sqlite (Этап 1).

Read-only по источнику. Берём ТОЛЬКО чаты с ID-телефоном [78]\\d{10}
(по решению Дмитрия: длинные WhatsApp-ID, иностранные номера, реклама и
чат «WhatsApp Calls» игнорируются). PII хранится открыто (БД локальна).
Бренд — только по однозначным маркерам; оба бренда -> mixed; иначе null.
"""
from __future__ import annotations
import re, sqlite3, json, os, sys
from collections import Counter

ROOT = os.environ.get("MANGO_ROOT", os.getcwd())
SRC = os.path.join(ROOT, "all_whatsapp_chats.txt")
# БД строим в локальном каталоге (SQLite не работает на сетевых FUSE-маунтах),
# затем копируем в product_data/transcripts. Override: MANGO_DB_DIR.
DB_DIR = os.environ.get("MANGO_DB_DIR", os.path.join(ROOT, "product_data", "transcripts"))
DB = os.path.join(DB_DIR, "whatsapp_chats.sqlite")
REPORT = os.path.join(DB_DIR, "whatsapp_normalize_report.json")

PHONE_ID = re.compile(r"^[78]\d{10}$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
TIME = re.compile(r"^\d{1,2}:\d{2}$")
AUTHOR_NUM = re.compile(r"^\d{1,16}$")
SERVICE_TEXT = "Not supported WhatsApp internal message"
HEADER = re.compile(r"^(Whatsapp - |Chat history with )")
END = "End of History"
APACHE = ("Portions of this page", "Apache 2.0 License", "shared by Google",
          "created and", "used according to terms", "work")
AD_SIG = re.compile(r"whatsapp business|каталог|приветственн|персонализ|"
                    r"увеличить продажи|заметк[аи].{0,30}клиент|умн[оы].{0,3}привет|"
                    r"делитесь информацией|автоматическ.{0,20}сообщени", re.I)

# Однозначные бренд-маркеры (выверены по CLAUDE.md/KB). Пацаева — адрес УНПК.
UNPK = ("унпк", "мфти", "менделеево", "kmipt", "сретенк", "институтск",
        "долгопрудн", "пацаев")
FOTON = ("фотон", "цдпо", "црдо", "скорняжн", "cdpofoton", "долями")


def norm_phone(s: str) -> str:
    d = re.sub(r"\D", "", s or "")
    if len(d) == 11 and d.startswith("8"):
        d = "7" + d[1:]
    if len(d) == 10:
        d = "7" + d
    return "+" + d if d else ""


def brand_of(text: str) -> str | None:
    low = (text or "").lower()
    u = any(m in low for m in UNPK)
    f = any(m in low for m in FOTON)
    if u and f:
        return "mixed"
    if u:
        return "unpk"
    if f:
        return "foton"
    return None


def is_ad(text: str) -> bool:
    return bool(text.startswith("*") and AD_SIG.search(text)) or bool(AD_SIG.search(text) and len(text) > 120 and text.count("\n") >= 0 and ("🚀" in text or "💬" in text or "🛍" in text or "📝" in text))


def parse_chat(cid: str, body: str):
    """-> list[dict] сообщений."""
    msgs = []
    cur_date = cur_time = ""
    cur_author = None
    buf: list[str] = []

    def flush():
        nonlocal buf
        if not buf:
            return
        text = "\n".join(buf).strip()
        buf = []
        if not text:
            return
        role = "manager" if cur_author == "You" else "client"
        service = 0
        if text == SERVICE_TEXT:
            role, service = "service", 1
        elif is_ad(text):
            role, service = "service", 1
        ts = (cur_date + "T" + cur_time) if (cur_date and cur_time) else (cur_date or "")
        msgs.append({
            "ts": ts, "role": role, "text": text,
            "brand_hint": None if service else brand_of(text),
            "is_service_message": service,
        })

    for raw in body.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line == END:
            break
        if HEADER.match(line):
            continue
        if line in APACHE or line.startswith("Portions of this page"):
            continue
        if DATE.match(line):
            flush(); cur_date = line; continue
        if TIME.match(line):
            flush(); cur_time = line; continue
        if line == "You" or line == cid or AUTHOR_NUM.match(line):
            flush(); cur_author = line; continue
        buf.append(line)
    flush()
    return msgs


def main():
    text = open(SRC, encoding="utf-8", errors="replace").read()
    parts = re.split(r"^===== CHAT: (.+?) =====$", text, flags=re.M)
    # parts[0] = преамбула; далее пары (id, body)
    os.makedirs(DB_DIR, exist_ok=True)
    if os.path.exists(DB):
        os.remove(DB)
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE chats(chat_id TEXT PRIMARY KEY, first_ts TEXT, last_ts TEXT,
      message_count INTEGER, brand_hint TEXT, brand_confidence REAL,
      client_phone TEXT, client_name TEXT, status TEXT);
    CREATE TABLE messages(msg_id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT,
      ts TEXT, role TEXT, text TEXT, brand_hint TEXT, is_service_message INTEGER,
      FOREIGN KEY(chat_id) REFERENCES chats(chat_id));
    CREATE TABLE crm_match(chat_id TEXT PRIMARY KEY, primary_phone TEXT, tallanto_id TEXT,
      match_confidence TEXT, match_basis TEXT, matched_at TEXT);
    CREATE TABLE analyses(analysis_id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT,
      analysis_type TEXT, result_json TEXT, created_at TEXT);
    CREATE INDEX idx_msg_chat ON messages(chat_id);
    CREATE INDEX idx_msg_ts ON messages(ts);
    CREATE INDEX idx_chat_phone ON chats(client_phone);
    CREATE INDEX idx_match_phone ON crm_match(primary_phone);
    """)

    stats = Counter()
    total_chats = 0
    chat_rows = []
    msg_rows = []
    for i in range(1, len(parts), 2):
        cid = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        total_chats += 1
        if not PHONE_ID.match(cid):
            stats["skipped_non_phone"] += 1
            continue
        stats["kept_phone_chats"] += 1
        msgs = parse_chat(cid, body)
        if not msgs:
            stats["empty_after_parse"] += 1
        ts_list = [m["ts"] for m in msgs if m["ts"]]
        nonservice = [m for m in msgs if not m["is_service_message"]]
        # бренд чата = из текстов (mixed > unpk/foton > null)
        bset = {m["brand_hint"] for m in nonservice if m["brand_hint"]}
        if "mixed" in bset or ("unpk" in bset and "foton" in bset):
            brand = "mixed"
        elif "unpk" in bset:
            brand = "unpk"
        elif "foton" in bset:
            brand = "foton"
        else:
            brand = None
        branded = [m for m in nonservice if m["brand_hint"]]
        bconf = round(len(branded) / max(1, len(nonservice)), 3)
        stats[f"brand_{brand}"] += 1
        chat_rows.append((cid, min(ts_list) if ts_list else "", max(ts_list) if ts_list else "",
                          len(nonservice), brand, bconf, norm_phone(cid), None, "raw"))
        for m in msgs:
            msg_rows.append((cid, m["ts"], m["role"], m["text"], m["brand_hint"], m["is_service_message"]))
            stats[f"role_{m['role']}"] += 1
        if len(chat_rows) >= 2000:
            cur.executemany("INSERT INTO chats VALUES(?,?,?,?,?,?,?,?,?)", chat_rows)
            cur.executemany("INSERT INTO messages(chat_id,ts,role,text,brand_hint,is_service_message) VALUES(?,?,?,?,?,?)", msg_rows)
            con.commit(); chat_rows.clear(); msg_rows.clear()
    if chat_rows:
        cur.executemany("INSERT INTO chats VALUES(?,?,?,?,?,?,?,?,?)", chat_rows)
    if msg_rows:
        cur.executemany("INSERT INTO messages(chat_id,ts,role,text,brand_hint,is_service_message) VALUES(?,?,?,?,?,?)", msg_rows)
    con.commit()
    n_chats = cur.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
    n_msgs = cur.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    con.close()
    report = {"total_chat_blocks": total_chats, "db_chats": n_chats, "db_messages": n_msgs,
              "stats": dict(stats), "db_path": DB}
    json.dump(report, open(REPORT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
