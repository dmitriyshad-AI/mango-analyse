#!/usr/bin/env python3
"""Случайная выборка реальных клиентских ВОПРОСОВ для СМЫСЛОВОГО аудита покрытия.
Упор на расшифровки звонков. Маскирует ПДн. Выход: _coverage_sample.json."""
from __future__ import annotations
import sqlite3, json, re, os, random
random.seed(20260610)

WA_DB="/tmp/wadb/whatsapp_chats.sqlite"
CALLS_DB="stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db"
TG_JSON="TP UNPK DataExport_2026-05-21/result.json"
OUT="D1_audit_backlog/kb_intake_20260610/_coverage_sample.json"

UNPK_M=("унпк","мфти","менделеево","kmipt","сретенк","институтск","долгопрудн","пацаев")
FOTON_M=("фотон","цдпо","црдо","скорняжн","cdpofoton","долями")
QSTART=re.compile(r"^(как|что|где|когда|куда|можно|какой|какие|каком|сколько|подскаж|а\s|почему|нужно|надо|есть\s+ли|правда\s+ли|будет\s+ли|подойд|с\s+какого|во\s+сколько|почём)", re.I)

def mask(t):
    t=re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+","<почта>",t)
    t=re.sub(r"\+?\d[\d\s\-()]{8,}\d","<тел>",t)
    t=re.sub(r"\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){1,2}\b","<имя>",t)
    t=re.sub(r"(зовут|реб[её]нок|ребёнка|ребенка|сын\w*|доч\w*)\s+[А-ЯЁ][а-яё]+", r"\1 <имя>", t, flags=re.I)
    t=re.sub(r"\b\d{1,2}\s*(класс|лет)\b", r"<N> \1", t)
    return re.sub(r"\s+"," ",t).strip()

def brand_calls(text):
    low=(text or "").lower(); u=any(m in low for m in UNPK_M); f=any(m in low for m in FOTON_M)
    return "unpk" if u and not f else "foton" if f and not u else "unknown"

def is_q(t):
    return ("?" in t) or bool(QSTART.match(t.lower()))

sample=[]; idc=0
# ---- Звонки: предложения-вопросы из СЛУЧАЙНЫХ звонков (упор) ----
c=sqlite3.connect(f"file:{CALLS_DB}?mode=ro",uri=True)
SENT=re.compile(r"[^.!?…]{30,300}[?]")  # явные вопросы, длиннее → самодостаточнее
pool=[]
for tc,tm in c.execute("SELECT transcript_client,transcript_manager FROM canonical_calls WHERE has_transcript_text=1 ORDER BY RANDOM() LIMIT 7000"):
    if not tc: continue
    br=brand_calls((tm or "")+" "+(tc or ""))
    for seg in SENT.findall(tc):
        s=seg.strip()
        if 40<=len(s)<=240 and is_q(s):
            pool.append((br,mask(s)))
c.close()
# дедуп грубый
seen=set(); upool=[]
for br,s in pool:
    k=re.sub(r"[^а-яё]","",s.lower())[:45]
    if k in seen: continue
    seen.add(k); upool.append((br,s))
for br,s in random.sample(upool, min(150,len(upool))):
    idc+=1; sample.append({"id":idc,"chan":"call","brand":br,"q":s})

# ---- WhatsApp ----
c=sqlite3.connect(f"file:{WA_DB}?mode=ro",uri=True)
wa=[]
for text,bh in c.execute("SELECT m.text,ch.brand_hint FROM messages m JOIN chats ch ON ch.chat_id=m.chat_id WHERE m.role='client' AND m.is_service_message=0"):
    t=(text or "").strip()
    if 15<=len(t)<=220 and is_q(t):
        wa.append((bh if bh in("foton","unpk") else "unknown", mask(t)))
c.close()
seen=set(); uwa=[]
for br,s in wa:
    k=re.sub(r"[^а-яё]","",s.lower())[:45]
    if k in seen: continue
    seen.add(k); uwa.append((br,s))
for br,s in random.sample(uwa, min(100,len(uwa))):
    idc+=1; sample.append({"id":idc,"chan":"wa","brand":br,"q":s})

# ---- Telegram УНПК ----
d=json.load(open(TG_JSON,encoding="utf-8")); tg=[]
for ch in d.get("chats",{}).get("list",[]):
    for mm in ch.get("messages",[]):
        if mm.get("type")!="message" or str(mm.get("from"))=="УНПК МФТИ": continue
        txt=mm.get("text")
        if isinstance(txt,list): txt="".join(x if isinstance(x,str) else x.get("text","") for x in txt)
        t=(txt or "").strip()
        if 15<=len(t)<=220 and is_q(t): tg.append(("unpk",mask(t)))
seen=set(); utg=[]
for br,s in tg:
    k=re.sub(r"[^а-яё]","",s.lower())[:45]
    if k in seen: continue
    seen.add(k); utg.append((br,s))
for br,s in random.sample(utg, min(50,len(utg))):
    idc+=1; sample.append({"id":idc,"chan":"tg","brand":br,"q":s})

os.makedirs(os.path.dirname(OUT),exist_ok=True)
json.dump(sample,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
import collections
ch=collections.Counter(x["chan"] for x in sample)
print("выборка:",dict(ch),"всего",len(sample))
print("OUT:",OUT)
print("\n=== 12 примеров (звонки) для контроля качества ===")
for x in [s for s in sample if s["chan"]=="call"][:12]:
    print(f"  [{x['brand']}] {x['q']}")
