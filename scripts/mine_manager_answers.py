#!/usr/bin/env python3
"""Майнинг реальных ОТВЕТОВ менеджеров по процессам. Read-only + маскировка ПДн.

Исходящие сообщения менеджеров (WhatsApp manager / Telegram оператор УНПК / call manager)
по каждому процессу = реальный client-facing порядок действий. Группировка по брендам.
Выводит JSON с обезличенными сниппетами для синтеза черновиков.
"""
from __future__ import annotations
import sqlite3, json, re, os, sys, collections
sys.path.insert(0, "scripts")
from process_questions_scan import P, NEW, UNPK_M, FOTON_M, brand_calls, mask

WA_DB="/tmp/wadb/whatsapp_chats.sqlite"
CALLS_DB="stable_runtime/canonical_master_20260523_audio_working_store_v1/canonical_calls_master.db"
TG_JSON="TP UNPK DataExport_2026-05-21/result.json"
OUT="/tmp/wadb/mgr_answers.json"

ALL=P+NEW
COMP=[(pid,cat,lab,re.compile(rx,re.I)) for pid,cat,lab,rx in ALL]
# процедурные маркеры — менеджер ОБЪЯСНЯЕТ порядок (а не просто упоминает тему)
PROC_CUE=re.compile(r"нужно|необходимо|после оплат|пришл[её]м|отправ\w+|оформля|для этого|сначала|"
                    r"затем|чтобы\s+(записа|оформ|оплат|получ)|вышл\w+|сформиру|в течение|можно\s+(оплат|записа|получ|оформ)|"
                    r"доступ|ссылк|квитанц|заявлен|реквизит|подключ|зайти|войти|кабинет|шаг|порядок|"
                    r"договор|справк|вычет|маткапитал|рассроч|долям|возврат|перенос|тест|расписан|лагер|смен", re.I)


def clean_snip(t):
    t=re.sub(r"^(replying to.*?\")","",t,flags=re.I)
    return mask(t)[:240]


def main():
    # pid -> brand -> list snippets
    res=collections.defaultdict(lambda: collections.defaultdict(list))
    seen=collections.defaultdict(set)

    def add(pid,brand,t):
        s=clean_snip(t)
        if len(s)<25: return
        k=re.sub(r"[^а-яё]","",s.lower())[:50]
        if k in seen[(pid,brand)]: return
        seen[(pid,brand)].add(k)
        if len(res[pid][brand])<10: res[pid][brand].append(s)

    # WhatsApp manager
    if os.path.exists(WA_DB):
        c=sqlite3.connect(f"file:{WA_DB}?mode=ro",uri=True)
        for text,bh in c.execute("""SELECT m.text,ch.brand_hint FROM messages m JOIN chats ch ON ch.chat_id=m.chat_id
                                     WHERE m.role='manager' AND m.is_service_message=0"""):
            t=(text or "").strip()
            if len(t)<30 or not PROC_CUE.search(t): continue
            brand=bh if bh in ("foton","unpk") else "unk"
            for pid,cat,lab,rx in COMP:
                if rx.search(t): add(pid,brand,t)
        c.close()

    # Telegram оператор УНПК
    if os.path.exists(TG_JSON):
        d=json.load(open(TG_JSON,encoding="utf-8"))
        for ch in d.get("chats",{}).get("list",[]):
            for mm in ch.get("messages",[]):
                if mm.get("type")!="message" or str(mm.get("from"))!="УНПК МФТИ": continue
                txt=mm.get("text")
                if isinstance(txt,list): txt="".join(x if isinstance(x,str) else x.get("text","") for x in txt)
                t=(txt or "").strip()
                if len(t)<30 or not PROC_CUE.search(t): continue
                for pid,cat,lab,rx in COMP:
                    if rx.search(t): add(pid,"unpk",t)

    # Звонки — manager канал (выборка: процедуры повторяются, репрезентативной хватит)
    if os.path.exists(CALLS_DB):
        lim=int(os.environ.get("CALLS_LIMIT","20000"))
        c=sqlite3.connect(f"file:{CALLS_DB}?mode=ro",uri=True)
        for tm,tc in c.execute("SELECT transcript_manager,transcript_client FROM canonical_calls WHERE has_transcript_text=1 LIMIT ?",(lim,)):
            t=(tm or "").strip()
            if len(t)<30 or not PROC_CUE.search(t): continue
            brand=brand_calls((tm or "")+" "+(tc or ""))
            for pid,cat,lab,rx in COMP:
                # пропускаем процесс, если по бренду уже набрано
                if len(res[pid][brand])>=10: continue
                m=rx.search(t)
                if m:
                    i=m.start()
                    add(pid,brand,t[max(0,i-40):i+200])
        c.close()

    out={"processes":[]}
    for pid,cat,lab,rx in ALL:
        out["processes"].append({"pid":pid,"category":cat,"label":lab,
            "by_brand":{b:res[pid][b] for b in ("foton","unpk","unk") if res[pid][b]}})
    json.dump(out,open(OUT,"w",encoding="utf-8"),ensure_ascii=False,indent=1)
    print("Сниппетов собрано (pid: foton/unpk/unk):")
    for p in out["processes"]:
        bb=p["by_brand"]
        print(f"  {p['pid']:4s} {p['label'][:34]:34s} f={len(bb.get('foton',[]))} u={len(bb.get('unpk',[]))} unk={len(bb.get('unk',[]))}")
    print("JSON:",OUT)


if __name__=="__main__":
    main()
