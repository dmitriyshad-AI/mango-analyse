#!/usr/bin/env python3
"""Случайная выборка 3000 реальных клиентских вопросов для смыслового аудита (масштаб).
Звонки 1600 (упор), WhatsApp 1000, Telegram УНПК 400. Маскирует ПДн. Выход + split на 6."""
from __future__ import annotations
import sqlite3, json, re, os, random, collections
random.seed(30303030)

WA_DB="/tmp/wadb/whatsapp_chats.sqlite"
CALLS_DB="stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db"
TG_JSON="TP UNPK DataExport_2026-05-21/result.json"
OUTDIR="D1_audit_backlog/kb_intake_20260610"
OUT=f"{OUTDIR}/_coverage_sample_3000.json"

UNPK_M=("унпк","мфти","менделеево","kmipt","сретенк","институтск","долгопрудн","пацаев","лобн")
FOTON_M=("фотон","цдпо","црдо","скорняжн","cdpofoton","долями","красносельск")
QSTART=re.compile(r"^(как|что|где|когда|куда|можно|какой|какие|каком|сколько|подскаж|а\s|почему|нужно|надо|есть\s+ли|правда\s+ли|будет\s+ли|подойд|с\s+какого|во\s+сколько|почём)", re.I)

def mask(t):
    t=re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+","<почта>",t)
    t=re.sub(r"\+?\d[\d\s\-()]{8,}\d","<тел>",t)
    t=re.sub(r"\b[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]{2,}){1,2}\b","<имя>",t)
    t=re.sub(r"(зовут|реб[её]нок|ребёнка|ребенка|сын\w*|доч\w*)\s+[А-ЯЁ][а-яё]+", r"\1 <имя>", t, flags=re.I)
    t=re.sub(r"\b\d{1,2}\s*(класс|лет)\b", r"<N> \1", t)
    return re.sub(r"\s+"," ",t).strip()[:190]

def brand_calls(text):
    low=(text or "").lower(); u=any(m in low for m in UNPK_M); f=any(m in low for m in FOTON_M)
    return "unpk" if u and not f else "foton" if f and not u else "unknown"

def is_q(t):
    return ("?" in t) or bool(QSTART.match(t.lower()))

def dedup(pairs):
    seen=set(); out=[]
    for br,s in pairs:
        k=re.sub(r"[^а-яё0-9]","",s.lower())[:55]
        if k in seen: continue
        seen.add(k); out.append((br,s))
    return out

sample=[]; idc=0

# ---- Звонки 1600 ----
c=sqlite3.connect(f"file:{CALLS_DB}?mode=ro",uri=True)
SENT=re.compile(r"[^.!?…]{30,300}[?]")
pool=[]
for tc,tm in c.execute("SELECT transcript_client,transcript_manager FROM canonical_calls WHERE has_transcript_text=1 ORDER BY RANDOM() LIMIT 28000"):
    if not tc: continue
    br=brand_calls((tm or "")+" "+(tc or ""))
    for seg in SENT.findall(tc):
        s=seg.strip()
        if 40<=len(s)<=240 and is_q(s): pool.append((br,mask(s)))
    if len(pool)>14000: break
c.close()
pool=dedup(pool)
for br,s in random.sample(pool, min(1600,len(pool))):
    idc+=1; sample.append({"id":idc,"chan":"call","brand":br,"q":s})

# ---- WhatsApp 1000 ----
c=sqlite3.connect(f"file:{WA_DB}?mode=ro",uri=True)
wa=[]
for text,bh in c.execute("SELECT m.text,ch.brand_hint FROM messages m JOIN chats ch ON ch.chat_id=m.chat_id WHERE m.role='client' AND m.is_service_message=0"):
    t=(text or "").strip()
    if 15<=len(t)<=240 and is_q(t):
        wa.append((bh if bh in("foton","unpk") else "unknown", mask(t)))
c.close()
wa=dedup(wa)
for br,s in random.sample(wa, min(1000,len(wa))):
    idc+=1; sample.append({"id":idc,"chan":"wa","brand":br,"q":s})

# ---- Telegram УНПК 400 ----
d=json.load(open(TG_JSON,encoding="utf-8")); tg=[]
for ch in d.get("chats",{}).get("list",[]):
    for mm in ch.get("messages",[]):
        if mm.get("type")!="message" or str(mm.get("from"))=="УНПК МФТИ": continue
        txt=mm.get("text")
        if isinstance(txt,list): txt="".join(x if isinstance(x,str) else x.get("text","") for x in txt)
        t=(txt or "").strip()
        if 15<=len(t)<=240 and is_q(t): tg.append(("unpk",mask(t)))
tg=dedup(tg)
for br,s in random.sample(tg, min(400,len(tg))):
    idc+=1; sample.append({"id":idc,"chan":"tg","brand":br,"q":s})

random.shuffle(sample)
for i,x in enumerate(sample,1): x["id"]=i  # перенумеровать после шафла
json.dump(sample,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=0)
# split на 6 файлов по id-диапазонам
N=len(sample); per=(N+5)//6
for k in range(6):
    a=k*per+1; b=min((k+1)*per,N)
    batch=[x for x in sample if a<=x["id"]<=b]
    json.dump(batch,open(f"{OUTDIR}/_sample3000_batch{k+1}.json","w",encoding="utf-8"),ensure_ascii=False,indent=0)
ch=collections.Counter(x["chan"] for x in sample)
print("всего:",N,dict(ch),"| per-batch:",per)
print("батчи: _sample3000_batch1..6.json")
