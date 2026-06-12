#!/usr/bin/env python3
"""Полная выгрузка ВСЕХ клиентских вопросов: WhatsApp + Telegram УНПК + звонки.
Маскирует ПДн. Пишет инкрементально в JSONL (источник истины — файл)."""
from __future__ import annotations
import sqlite3, json, re, os, sys, collections
WA_DB="/tmp/wadb/whatsapp_chats.sqlite"
CALLS_DB="stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db"
TG_JSON="TP UNPK DataExport_2026-05-21/result.json"
OUT="D1_audit_backlog/kb_intake_20260610/questions_all_clients_2026-06-10.jsonl"

UNPK_M=("унпк","мфти","менделеево","kmipt","сретенк","институтск","долгопрудн","пацаев","лобн")
FOTON_M=("фотон","цдпо","црдо","скорняжн","cdpofoton","долями","красносельск")
QSTART=re.compile(r"^(как|что|где|когда|куда|можно|какой|какие|каком|сколько|подскаж|а\s|почему|нужно|надо|есть\s+ли|правда\s+ли|будет\s+ли|подойд|с\s+какого|во\s+сколько|почём)",re.I)
def isq(t): return ("?" in t) or bool(QSTART.match(t.lower()))
def mask(t):
    t=re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+","<почта>",t)
    t=re.sub(r"\+?\d[\d\s\-()]{8,}\d","<тел>",t)
    t=re.sub(r"\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){1,2}\b","<имя>",t)
    t=re.sub(r"(зовут|реб[её]нок|ребёнка|ребенка|сын\w*|доч\w*)\s+[А-ЯЁ][а-яё]+",r"\1 <имя>",t,flags=re.I)
    t=re.sub(r"\b\d{1,2}\s*(класс|лет)\b",r"<N> \1",t)
    return re.sub(r"\s+"," ",t).strip()
def brand_calls(text):
    low=(text or "").lower(); u=any(m in low for m in UNPK_M); f=any(m in low for m in FOTON_M)
    return "unpk" if u and not f else "foton" if f and not u else "unknown"

f=open(OUT,"w",encoding="utf-8",buffering=1)  # построчная буферизация — переживёт обрыв
cnt=collections.Counter()
seen=collections.defaultdict(set)
def emit(src,brand,q):
    k=re.sub(r"[^а-яё0-9]","",q.lower())[:60]
    if k in seen[src]: cnt[src+"_dup"]+=1; return
    seen[src].add(k)
    f.write(json.dumps({"src":src,"brand":brand,"q":q[:240]},ensure_ascii=False)+"\n")
    cnt[src]+=1

# WhatsApp
if os.path.exists(WA_DB):
    c=sqlite3.connect(f"file:{WA_DB}?mode=ro",uri=True)
    for text,bh in c.execute("SELECT m.text,ch.brand_hint FROM messages m JOIN chats ch ON ch.chat_id=m.chat_id WHERE m.role='client' AND m.is_service_message=0"):
        t=(text or "").strip()
        if len(t)>=8 and isq(t): emit("wa", bh if bh in("foton","unpk") else "unknown", mask(t))
    c.close()
print("WA done:",cnt["wa"],flush=True)
# Telegram
if os.path.exists(TG_JSON):
    d=json.load(open(TG_JSON,encoding="utf-8"))
    for ch in d.get("chats",{}).get("list",[]):
        for mm in ch.get("messages",[]):
            if mm.get("type")!="message" or str(mm.get("from"))=="УНПК МФТИ": continue
            txt=mm.get("text")
            if isinstance(txt,list): txt="".join(x if isinstance(x,str) else x.get("text","") for x in txt)
            t=(txt or "").strip()
            if len(t)>=8 and isq(t): emit("tg","unpk",mask(t))
print("TG done:",cnt["tg"],flush=True)
# Calls (предложения-вопросы клиента)
if os.path.exists(CALLS_DB):
    SENT=re.compile(r"[^.!?…]{10,300}[?]")  # явные вопросы клиента
    c=sqlite3.connect(f"file:{CALLS_DB}?mode=ro",uri=True)
    n=0
    for tc,tm in c.execute("SELECT transcript_client,transcript_manager FROM canonical_calls WHERE has_transcript_text=1"):
        if not tc: continue
        n+=1
        br=brand_calls((tm or "")+" "+(tc or ""))
        for seg in SENT.findall(tc):
            s=seg.strip()
            if 10<=len(s)<=240: emit("call",br,mask(s))
    c.close()
    print("calls scanned:",n,flush=True)
f.close()
print("ИТОГО:",dict(cnt),flush=True)
