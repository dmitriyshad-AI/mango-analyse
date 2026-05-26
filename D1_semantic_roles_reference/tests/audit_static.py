from __future__ import annotations

"""Детерминированные статические аудиты (без вызова модели):
  A. Литеральная бренд-утечка — в client-safe тексте одного бренда не должно быть
     литералов ДРУГОГО бренда (имена/домены/телефоны/адреса/Долями/Т-Банк).
  B. Числа вне снимка — цены/проценты в client-facing драфтах (few-shot), которых
     нет в выверенном снимке v2 для этого бренда → кандидаты на устаревание/выдумку.

Оба аудита — про СТАТИКУ (артефакты, которые бот использует), не про живую генерацию.
gold_answers НЕ сканируем на утечку: там оба бренда по дизайну (справочник правил).

Запуск:
  python3 D1_semantic_roles_reference/tests/audit_static.py
"""

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.join(os.path.dirname(_HERE), "reference")
sys.path.insert(0, _REF)
from semantic_roles import has_marker  # noqa: E402

REPO = "/sessions/confident-sleepy-darwin/mnt/Mango analyse"
FOTON_FACTS = f"{REPO}/Mango_Bot_KB_FINAL_v6_3_2026-05-20/01_bot_pack/client_safe_facts_foton.jsonl"
UNPK_FACTS = f"{REPO}/Mango_Bot_KB_FINAL_v6_3_2026-05-20/01_bot_pack/client_safe_facts_unpk.jsonl"
TEMPLATES = f"{REPO}/Mango_Bot_KB_FINAL_v6_3_2026-05-20/01_bot_pack/bot_template_registry.json"
FEWSHOT = f"{REPO}/Claude Cowork Space/03_corpus/few_shot_warm_answers_2026-05-23.yaml"
SNAP_V2 = "/sessions/confident-sleepy-darwin/mnt/Foton/v8_dynamic_client_sim_2026-05-25_v2.jsonl"

# Термины ЧУЖОГО бренда (то, чего быть НЕ должно в client-safe тексте данного бренда).
# Делим на словарные (по границе слова) и подстрочные (домены/хендлы/с дефисом).
FOTON_FORBIDDEN_WORD = ["УНПК", "АНО ДПО", "НОУ УНПК", "Сретенка", "Лобня", "Пацаева", "Институтский", "Долгопрудный"]
FOTON_FORBIDDEN_SUB = ["kmipt.ru", "edu@kmipt.ru", "kmipt.tallanto.com", "@unpk_mipt", "@unpkmfti"]
UNPK_FORBIDDEN_WORD = ["Фотон", "ЦДПО", "ЦРДО", "Долями", "Т-Банк"]
UNPK_FORBIDDEN_SUB = ["cdpofoton.ru", "edu@cdpofoton.ru", "@cdpoFoton", "foton_edu", "vk.ru/foton_edu"]


def leak_hits(text: str, words, subs):
    hits = []
    low = str(text or "")
    for w in words:
        if has_marker(low, w):
            hits.append(w)
    lc = low.casefold()
    for s in subs:
        if s.casefold() in lc:
            hits.append(s)
    return hits


def audit_brand_leak():
    print("## A. ЛИТЕРАЛЬНАЯ БРЕНД-УТЕЧКА")
    total = 0
    for path, brand, words, subs in (
        (FOTON_FACTS, "foton", FOTON_FORBIDDEN_WORD, FOTON_FORBIDDEN_SUB),
        (UNPK_FACTS, "unpk", UNPK_FORBIDDEN_WORD, UNPK_FORBIDDEN_SUB),
    ):
        n = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                txt = d.get("client_safe_text") or ""
                hits = leak_hits(txt, words, subs)
                if hits:
                    n += 1
                    print(f"   [LEAK {brand}] {d.get('fact_key')}: {hits} | «{txt[:90]}»")
        total += n
        print(f"   client_safe_facts_{brand}: утечек {n}")
    # снимок v2 (правда судьи)
    with open(SNAP_V2, encoding="utf-8") as fh:
        fh.readline()
        judge = json.loads(fh.readline())
    snap = judge.get("confirmed_facts_snapshot", {})
    for brand, words, subs in (("foton", FOTON_FORBIDDEN_WORD, FOTON_FORBIDDEN_SUB),
                               ("unpk", UNPK_FORBIDDEN_WORD, UNPK_FORBIDDEN_SUB)):
        n = 0
        for fact in snap.get(brand, []):
            hits = leak_hits(fact, words, subs)
            if hits:
                n += 1
                print(f"   [LEAK snapshot {brand}] {hits} | «{fact[:90]}»")
        total += n
        print(f"   snapshot_v2[{brand}]: утечек {n}")
    print(f"   ИТОГО утечек: {total}\n")
    return total


_NUM = re.compile(r"\d[\d  ]*\d|\d")
_PCT = re.compile(r"(\d+)\s*%")


def price_numbers(text: str):
    """Цена-подобные числа: суммы >= 1000 (нормализованы) и проценты."""
    out = set()
    s = str(text or "")
    for m in _NUM.finditer(s):
        norm = re.sub(r"[  ]", "", m.group(0))
        if norm.isdigit() and int(norm) >= 1000:
            out.add(norm)
    for m in _PCT.finditer(s):
        out.add(m.group(1) + "%")
    return out


def audit_numbers_vs_snapshot():
    print("## B. ЧИСЛА ВНЕ СНИМКА (few-shot / client-facing драфты)")
    with open(SNAP_V2, encoding="utf-8") as fh:
        fh.readline()
        judge = json.loads(fh.readline())
    snap = judge.get("confirmed_facts_snapshot", {})
    snap_nums = {b: set() for b in ("foton", "unpk")}
    for b in snap_nums:
        for fact in snap.get(b, []):
            snap_nums[b] |= price_numbers(fact)

    try:
        import yaml  # type: ignore
    except Exception:
        print("   PyYAML недоступен — пропускаю few-shot число-аудит\n")
        return 0
    with open(FEWSHOT, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    # обойти дерево, привязывая бренд из ближайшего ключа/поля
    candidates = []
    cover = {"strings": 0, "numbers": 0}

    def walk(node, brand):
        if isinstance(node, dict):
            b = node.get("brand") if isinstance(node.get("brand"), str) else brand
            for k, v in node.items():
                kb = b
                kl = str(k).casefold()
                if "foton" in kl or "фотон" in kl:
                    kb = "foton"
                elif "unpk" in kl or "унпк" in kl:
                    kb = "unpk"
                walk(v, kb)
        elif isinstance(node, list):
            for it in node:
                walk(it, brand)
        elif isinstance(node, str) and brand in ("foton", "unpk") and len(node) > 30:
            nums = price_numbers(node)
            cover["strings"] += 1
            cover["numbers"] += len(nums)
            bad = {n for n in nums if n not in snap_nums[brand]}
            if bad:
                candidates.append((brand, sorted(bad), node[:110]))

    walk(data, "")
    print(f"   покрытие: просканировано текстов few-shot {cover['strings']}, цена-чисел {cover['numbers']}")
    for brand, bad, txt in candidates:
        print(f"   [num? {brand}] {bad} | «{txt}»")
    print(f"   кандидатов (число не из снимка): {len(candidates)}")
    print("   (часть может быть безобидной — телефоны/года; смотреть только цены/проценты)\n")
    return len(candidates)


def main() -> int:
    print("=== СТАТИЧЕСКИЕ АУДИТЫ (детерминированные, без модели) ===\n")
    leaks = audit_brand_leak()
    nums = audit_numbers_vs_snapshot()
    print(f"ВЕРДИКТ: бренд-утечек {leaks} (должно быть 0); число-кандидатов {nums} (ручной разбор)")
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
