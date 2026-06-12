#!/usr/bin/env python3
"""Преобразует processes_client_safe_drafts_2026-06-09.md в красивый md с таблицами
Фотон|УНПК для каждого процесса, затем pandoc -> docx (вызывается отдельно)."""
import re, sys

SRC = sys.argv[1]
OUT = sys.argv[2]
lines = open(SRC, encoding="utf-8").read().split("\n")

out = []
i = 0
n = len(lines)


def esc_cell(t):
    return t.replace("|", "\\|").strip()


def flush_table(foton, unpk):
    if foton and unpk:
        out.append("")
        out.append("| **ФОТОН** | **УНПК** |")
        out.append("|:--|:--|")
        out.append(f"| {esc_cell(foton)} | {esc_cell(unpk)} |")
        out.append("")
    elif foton:
        out.append("")
        out.append(f"**Фотон:** {foton}")
        out.append("")
    elif unpk:
        out.append("")
        out.append(f"**УНПК:** {unpk}")
        out.append("")


while i < n:
    line = lines[i]
    if line.startswith("### "):
        # процессный блок
        out.append(line)
        i += 1
        foton = unpk = both = note = conflict = None
        while i < n and not lines[i].startswith("### ") and not lines[i].startswith("## ") and not lines[i].startswith("# ") and not lines[i].startswith("---"):
            l = lines[i].strip()
            if l.startswith("**Фотон"):
                foton = re.sub(r"^\*\*Фотон[^:]*:\*\*\s*", "", l)
            elif l.startswith("**УНПК"):
                unpk = re.sub(r"^\*\*УНПК[^:]*:\*\*\s*", "", l)
            elif l.startswith("**Оба"):
                both = re.sub(r"^\*\*Оба[^:]*:\*\*\s*", "", l)
            elif l.startswith("**Semantic-note"):
                note = re.sub(r"^\*\*Semantic-note:\*\*\s*", "", l)
            elif l.startswith("**⚠"):
                conflict = re.sub(r"^\*\*[^:]*:\*\*\s*", "", l)
            i += 1
        flush_table(foton, unpk)
        if both:
            out.append(f"**Оба бренда (раздельно):** {both}")
            out.append("")
        if conflict:
            out.append(f"> ⚠ **КОНФЛИКТ ИСТОЧНИКОВ:** {conflict}")
            out.append("")
        if note:
            out.append(f"*Заметка: {note}*")
            out.append("")
        continue
    out.append(line)
    i += 1

open(OUT, "w", encoding="utf-8").write("\n".join(out))
print("written", OUT, "lines", len(out))
