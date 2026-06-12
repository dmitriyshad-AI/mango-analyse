#!/usr/bin/env python3
"""Свести 6 файлов пер-вопросных вердиктов, проверить целостность, посчитать честно,
собрать аудируемый _verdicts_all.jsonl и выдать случайные строки на ручную проверку."""
import json, re, os, random, collections
random.seed(7)
D="D1_audit_backlog/kb_intake_20260610"

# вопросы (id -> chan/brand/q)
q={}
for k in range(1,7):
    for x in json.load(open(f"{D}/_sample3000_batch{k}.json",encoding="utf-8")):
        q[x["id"]]={"chan":x["chan"],"brand":x["brand"],"q":x["q"]}

# вердикты
V={}
bad=0
for k in range(1,7):
    p=f"{D}/_verdicts_batch{k}.jsonl"
    for line in open(p,encoding="utf-8"):
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except: bad+=1; continue
        if "id" in o: V[o["id"]]=o

ids=set(V)&set(q)
print("вопросов:",len(q),"| вердиктов распознано:",len(V),"| совпало:",len(ids),"| битых строк:",bad)
miss=sorted(set(range(1,3001))-set(V))
print("пропущенные id:",len(miss), miss[:10])

def norm_status(s):
    s=(s or "").lower()
    for v in("полное","частичное","p0","фрагмент"):
        if v in s: return v
    if "нет" in s: return "НЕТ"
    return s
def gap_of(o):
    g=(o.get("gap") or "").upper()
    m=re.search(r"G\d+",g)
    if m: return m.group(0)
    m=re.search(r"G\d+",(o.get("note") or "").upper())  # подсказка в note
    return m.group(0) if m else ""

st=collections.Counter(); st_ch=collections.defaultdict(collections.Counter)
gap=collections.Counter()
live_signal=0
out=[]
for i in sorted(ids):
    o=V[i]; s=norm_status(o.get("status"))
    ch=q[i]["chan"]
    st[s]+=1; st_ch[s][ch]+=1
    g=gap_of(o)
    if s=="НЕТ" and g: gap[g]+=1
    # сигнал "живой/оперативный" по note независимо от статуса
    note=(o.get("note") or "").lower()+" "+g.lower()
    if g=="G1" or re.search(r"опер|live|места?\b|прошла ли|статус|сегодня|завтра|набрал|во сколько", note):
        live_signal+=1
    out.append({**q[i],"id":i,"status":s,"gap":g,"note":o.get("note","")})

json.dump(out,open(f"{D}/_verdicts_all.jsonl","w",encoding="utf-8"),ensure_ascii=False)  # компактный комбайн
# заодно построчный JSONL (удобнее аудиту)
with open(f"{D}/_verdicts_all.jsonl","w",encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r,ensure_ascii=False)+"\n")

print("\n=== СТАТУСЫ (всего) ===")
for s in("полное","частичное","НЕТ","p0","фрагмент"):
    print(f"  {s:10s} {st[s]:5d}   call={st_ch[s]['call']} wa={st_ch[s]['wa']} tg={st_ch[s]['tg']}")
denom=st["полное"]+st["частичное"]+st["НЕТ"]
print(f"\nЗнаменатель (полное+частичное+НЕТ) = {denom}")
print(f"  полное      {st['полное']/denom*100:.0f}%")
print(f"  частичное   {st['частичное']/denom*100:.0f}%")
print(f"  НЕТ         {st['НЕТ']/denom*100:.0f}%")
print(f"  бот может ответить (полное+частичное) {(st['полное']+st['частичное'])/denom*100:.0f}%")
print(f"\nСигнал 'живой/оперативный' (по note, незав. от статуса): ~{live_signal} ({live_signal/len(ids)*100:.0f}% выборки)")
print("\n=== Категории дыр (status=НЕТ) ===")
for g,n in gap.most_common(): print(f"  {g}: {n}")

print("\n=== 30 СЛУЧАЙНЫХ вердиктов на ручную проверку ===")
for r in random.sample(out, 30):
    print(f"[{r['id']} {r['chan']}/{r['brand']}] {r['status']}{('/'+r['gap']) if r['gap'] else ''} — {r['q'][:90]} || {r['note'][:50]}")
print("\nкомбайн: _verdicts_all.jsonl (3000 строк, аудируемо)")
