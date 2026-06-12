#!/usr/bin/env python3
"""Из полной выгрузки оставить только СОДЕРЖАТЕЛЬНЫЕ УНИКАЛЬНЫЕ вопросы.
Убирает: бэкчэннел («да?/правильно?»), ASR-каша, слишком короткие, без содержания; глобальный дедуп."""
import json, re, collections
RAW="D1_audit_backlog/kb_intake_20260610/questions_all_clients_2026-06-10.jsonl"
OUT="D1_audit_backlog/kb_intake_20260610/questions_clean_2026-06-10.jsonl"

BACKCHANNEL=re.compile(r"^(да|нет|ага|угу|алло|так|ну|и|а|это|вот|хорошо|ладно|понятно|ясно|спасибо|благодар\w*|правильно|верно|точно|конечно|ок|окей|здравствуйте|добр\w+ (день|утро|вечер)|приветствую|привет|momentik|минут\w*)[\s,.\-—?!)(]*$",re.I)
TAGQ=re.compile(r"^.{0,20}[,\s]*(да|правильно|верно|так|нет|ага|угу)\s*\?+$",re.I)
GARBLE=re.compile(r"\b(\w{1,4})(\s+\1){2,}\b",re.I)
CONTENT=re.compile(r"курс|занят|цен|стоим|стоит|оплат|плат\b|расписан|график|лагер|смен|лвш|выездн|предмет|математ|физик|информат|русск|программ|олимпиад|физтех|класс|возраст|онлайн|очно|формат|договор|оферт|справк|вычет|маткап|материнск|рассроч|долям|скидк|кэшб|тест|тестир|групп|уровень|преподавател|педагог|учител|адрес|метро|запис|ссылк|платформ|кабинет|логин|пароль|возврат|вернуть|перенос|заморозк|пропуск|пробн|сертификат|трансфер|прожив|питан|реквизит|\bчек|квитанц|счет|счёт|перезвон|телефон|почт|вебинар|мтс|линк|soho|демо|интенсив|егэ|огэ|балл|поступ|результат|консультац|менеджер|куратор|документ|анкет|полис|медсправк|ваучер|каникул|переход|перевод",re.I)
QWORD=re.compile(r"\b(как|что|где|когда|куда|сколько|какой|какая|какие|каком|можно|есть ли|нужно|почему|во сколько|с какого|до какого|кто|чем|подойд\w+|подскаж)",re.I)

def substantive(src,q):
    t=q.strip()
    low=t.lower()
    low=re.sub(r"^(replying to .*?\"|the media is missing)\s*","",low).strip()
    t2=re.sub(r"<[^>]+>"," ",t)  # убрать плейсхолдеры для оценки
    if not low: return False
    if BACKCHANNEL.match(low): return False
    if TAGQ.match(low) and not CONTENT.search(low): return False
    if GARBLE.search(low): return False
    alpha=sum(ch.isalpha() for ch in t2);
    if alpha/max(1,len(t2))<0.5: return False
    nwords=len(re.findall(r"[a-zа-яё]{2,}",low))
    minlen=12 if src in("wa","tg") else 35
    if len(t)<minlen or nwords<(3 if src in("wa","tg") else 5): return False
    if src=="call":
        # звонки: требуем реальную предметную тему (иначе разговорный шум)
        if re.match(r"^(ну|то есть|значит|вот\b|а вот|хорошо|ладно|короче|это самое|допустим|просто|кстати|слушайте)\b",low) and len(t)<55:
            return False
        if not CONTENT.search(low): return False
        if nwords<6: return False
        return True
    # мессенджеры (чище): контент-токен ИЛИ вопрос-слово с достаточной длиной
    if CONTENT.search(low): return True
    if QWORD.search(low) and nwords>=4: return True
    if "?" in t and len(t)>=40 and nwords>=6: return True
    return False

seen=set(); cnt=collections.Counter()
f=open(OUT,"w",encoding="utf-8")
for l in open(RAW,encoding="utf-8",errors="replace"):
    try: o=json.loads(l)
    except: continue
    src,q=o["src"],o["q"]
    if not substantive(src,q): cnt[src+"_drop"]+=1; continue
    key=re.sub(r"[^а-яё0-9a-z]","",q.lower())[:70]
    if key in seen: cnt[src+"_dup"]+=1; continue
    seen.add(key); f.write(json.dumps(o,ensure_ascii=False)+"\n"); cnt[src]+=1
f.close()
tot=cnt["wa"]+cnt["tg"]+cnt["call"]
print(f"ОСТАВЛЕНО содержательных уникальных: {tot}")
print(f"  WhatsApp {cnt['wa']} (отсев {cnt['wa_drop']}, дубли {cnt['wa_dup']})")
print(f"  Telegram {cnt['tg']} (отсев {cnt['tg_drop']}, дубли {cnt['tg_dup']})")
print(f"  Звонки   {cnt['call']} (отсев {cnt['call_drop']}, дубли {cnt['call_dup']})")
print("файл:",OUT)
