#!/usr/bin/env python3
"""Этап 4.5 — верификация качества WhatsApp-пайплайна. Read-only.

Печатает факты для ручного суждения (без авто-вердиктов):
1) целостность БД;
2) разбор бренд-доказательств — особенно риск ложных меток по «фотон» (термин физики)
   и «долями» (обычное «частями»);
3) корректность матчинга (primary_phone == norm(chat_id));
4) контракт безопасности провайдера.
"""
from __future__ import annotations
import sqlite3, os, re, json, random
from collections import Counter, defaultdict

DB = os.environ.get("MANGO_DB", "/tmp/wadb/whatsapp_chats.sqlite")
random.seed(42)

UNPK = ("унпк", "мфти", "менделеево", "kmipt", "сретенк", "институтск", "долгопрудн", "пацаев")
FOTON = ("фотон", "цдпо", "црдо", "скорняжн", "cdpofoton", "долями")


def norm_phone(s: str) -> str:
    d = re.sub(r"\D", "", s or "")
    if len(d) == 11 and d.startswith("8"):
        d = "7" + d[1:]
    if len(d) == 10:
        d = "7" + d
    return "+" + d if d else ""


def markers_in(text: str, markers):
    low = (text or "").lower()
    return [m for m in markers if m in low]


def snippet_around(text: str, marker: str, pad: int = 45) -> str:
    low = text.lower(); i = low.find(marker)
    if i < 0:
        return ""
    a = max(0, i - pad); b = min(len(text), i + len(marker) + pad)
    return re.sub(r"\s+", " ", text[a:b]).strip()


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); cur = con.cursor()
    out = {}

    # ---------- 1) ЦЕЛОСТНОСТЬ ----------
    n_chats = cur.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
    n_msgs = cur.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    orphans = cur.execute("SELECT COUNT(*) FROM messages m LEFT JOIN chats c ON c.chat_id=m.chat_id WHERE c.chat_id IS NULL").fetchone()[0]
    empty_ts = cur.execute("SELECT COUNT(*) FROM messages WHERE ts='' OR ts IS NULL").fetchone()[0]
    roles = dict(cur.execute("SELECT role, COUNT(*) FROM messages GROUP BY role").fetchall())
    svc = dict(cur.execute("SELECT is_service_message, COUNT(*) FROM messages GROUP BY is_service_message").fetchall())
    bad_phone = cur.execute("SELECT COUNT(*) FROM chats WHERE client_phone NOT LIKE '+7%'").fetchone()[0]
    # чаты без единого содержательного сообщения
    empty_chats = cur.execute("SELECT COUNT(*) FROM chats WHERE message_count=0").fetchone()[0]
    out["integrity"] = {"chats": n_chats, "messages": n_msgs, "orphan_messages": orphans,
                        "empty_ts_messages": empty_ts, "roles": roles, "is_service": svc,
                        "chats_phone_not_+7": bad_phone, "chats_zero_content": empty_chats}

    # ---------- 2) БРЕНД-ДОКАЗАТЕЛЬСТВА ----------
    rows = cur.execute("""SELECT ch.chat_id, ch.brand_hint,
                                 GROUP_CONCAT(CASE WHEN m.is_service_message=0 THEN m.text END, ' ')
                          FROM chats ch LEFT JOIN messages m ON m.chat_id=ch.chat_id
                          GROUP BY ch.chat_id""").fetchall()
    foton_evidence_combo = Counter()      # какие именно foton-маркеры дают foton/mixed
    unpk_evidence_combo = Counter()
    foton_only_photon = []                # foton-метка, где доказательство ТОЛЬКО «фотон»
    foton_only_dolyami = []               # ТОЛЬКО «долями»
    mixed_foton_side_weak = []            # mixed, где foton-сторона только «фотон»/«долями»
    label_recheck_mismatch = 0
    for cid, label, alltext in rows:
        alltext = alltext or ""
        fm = markers_in(alltext, FOTON)
        um = markers_in(alltext, UNPK)
        # пересчёт метки «с нуля» — сверка с тем, что в БД
        recomputed = "mixed" if (fm and um) else ("foton" if fm else ("unpk" if um else None))
        if recomputed != label:
            label_recheck_mismatch += 1
        if fm:
            foton_evidence_combo["+".join(sorted(fm))] += 1
        if um:
            unpk_evidence_combo["+".join(sorted(um))] += 1
        if label == "foton":
            if fm == ["фотон"]:
                foton_only_photon.append((cid, snippet_around(alltext, "фотон")))
            elif fm == ["долями"]:
                foton_only_dolyami.append((cid, snippet_around(alltext, "долями")))
        if label == "mixed" and set(fm) <= {"фотон", "долями"}:
            mk = "фотон" if "фотон" in fm else "долями"
            mixed_foton_side_weak.append((cid, mk, snippet_around(alltext, mk)))
    out["brand"] = {
        "label_recheck_mismatch": label_recheck_mismatch,
        "foton_evidence_top": foton_evidence_combo.most_common(12),
        "unpk_evidence_top": unpk_evidence_combo.most_common(12),
        "foton_only_by_фотон_count": len(foton_only_photon),
        "foton_only_by_долями_count": len(foton_only_dolyami),
        "mixed_weak_foton_side_count": len(mixed_foton_side_weak),
        "sample_foton_only_фотон": random.sample(foton_only_photon, min(8, len(foton_only_photon))),
        "sample_foton_only_долями": random.sample(foton_only_dolyami, min(8, len(foton_only_dolyami))),
        "sample_mixed_weak": random.sample(mixed_foton_side_weak, min(8, len(mixed_foton_side_weak))),
    }

    # ---------- 3) МАТЧИНГ ----------
    mism = cur.execute("SELECT COUNT(*) FROM crm_match WHERE primary_phone NOT LIKE '+7%'").fetchone()[0]
    conf = dict(cur.execute("SELECT match_confidence, COUNT(*) FROM crm_match GROUP BY match_confidence").fetchall())
    # сверка primary_phone == norm(chat_id)
    pp = cur.execute("SELECT chat_id, primary_phone FROM crm_match").fetchall()
    phone_mismatch = sum(1 for cid, ph in pp if norm_phone(cid) != ph)
    out["matching"] = {"matched_total": len(pp), "confidence": conf,
                       "primary_phone_not_+7": mism, "phone_vs_chatid_mismatch": phone_mismatch}

    con.close()
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
