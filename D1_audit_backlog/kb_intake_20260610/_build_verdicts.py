#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, os

SRC = "_part2_batch1.jsonl"
OUT = "_part2_verdicts_batch1.jsonl"

src = {}
order = []
with open(SRC, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        o = json.loads(line)
        src[o["id"]] = o
        order.append(o["id"])

DEC = {}

def s(i, status, action, gap, note):
    DEC[i] = (status, action, gap, note)

seg_path = "_verdict_segments.py"
with open(seg_path, encoding="utf-8") as f:
    code = f.read()
exec(compile(code, seg_path, "exec"))

missing = [i for i in order if i not in DEC]
if missing:
    raise SystemExit("MISSING decisions for ids: %s (count %d)" % (missing[:30], len(missing)))

def trunc(q):
    return q[:110]

with open(OUT, "w", encoding="utf-8") as f:
    for i in order:
        o = src[i]
        status, action, gap, note = DEC[i]
        rec = {
            "id": i,
            "brand": o["brand"],
            "q": trunc(o["q"]),
            "status": status,
            "bot_action": action,
            "gap": gap,
            "note": note,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

from collections import Counter
st = Counter(DEC[i][0] for i in order)
ac = Counter(DEC[i][1] for i in order)
gp = Counter(DEC[i][2] for i in order if DEC[i][1] == "defer_manager")
print("TOTAL", len(order))
print("status", dict(st))
print("action", dict(ac))
print("defer_manager gap breakdown", dict(gp))
print("defer_manager total", sum(1 for i in order if DEC[i][1]=="defer_manager"))
