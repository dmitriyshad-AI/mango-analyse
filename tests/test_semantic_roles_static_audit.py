from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from mango_mvp.channels.semantic_roles import has_marker


REPO = Path("/Users/dmitrijfabarisov/Projects/Mango analyse")
FOTON_FACTS = REPO / "Mango_Bot_KB_FINAL_v6_3_2026-05-20/01_bot_pack/client_safe_facts_foton.jsonl"
UNPK_FACTS = REPO / "Mango_Bot_KB_FINAL_v6_3_2026-05-20/01_bot_pack/client_safe_facts_unpk.jsonl"
FEWSHOT = REPO / "Claude Cowork Space/03_corpus/few_shot_warm_answers_2026-05-23.yaml"
SNAP_V2 = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/v8_dynamic_client_sim_2026-05-25_v2.jsonl")

FOTON_FORBIDDEN_WORD = ["УНПК", "АНО ДПО", "НОУ УНПК", "Сретенка", "Лобня", "Пацаева", "Институтский", "Долгопрудный"]
FOTON_FORBIDDEN_SUB = ["kmipt.ru", "edu@kmipt.ru", "kmipt.tallanto.com", "@unpk_mipt", "@unpkmfti"]
UNPK_FORBIDDEN_WORD = ["Фотон", "ЦДПО", "ЦРДО", "Долями", "Т-Банк"]
UNPK_FORBIDDEN_SUB = ["cdpofoton.ru", "edu@cdpofoton.ru", "@cdpoFoton", "foton_edu", "vk.ru/foton_edu"]

NUM_RE = re.compile(r"\d[\d \u00a0]*\d|\d")
PCT_RE = re.compile(r"(\d+)\s*%")


def _leak_hits(text: str, words: list[str], subs: list[str]) -> list[str]:
    hits: list[str] = []
    value = str(text or "")
    for word in words:
        if has_marker(value, word):
            hits.append(word)
    low = value.casefold()
    for sub in subs:
        if sub.casefold() in low:
            hits.append(sub)
    return hits


def _price_numbers(text: str) -> set[str]:
    out: set[str] = set()
    value = str(text or "")
    for match in NUM_RE.finditer(value):
        norm = re.sub(r"[ \u00a0]", "", match.group(0))
        if norm.isdigit() and int(norm) >= 1000:
            out.add(norm)
    for match in PCT_RE.finditer(value):
        out.add(f"{match.group(1)}%")
    return out


def _load_snapshot() -> dict[str, list[str]]:
    with SNAP_V2.open(encoding="utf-8") as fh:
        next(fh)
        return json.loads(next(fh)).get("confirmed_facts_snapshot", {})


def test_client_safe_facts_and_snapshot_have_no_literal_brand_leaks() -> None:
    failures: list[str] = []
    for path, brand, words, subs in (
        (FOTON_FACTS, "foton", FOTON_FORBIDDEN_WORD, FOTON_FORBIDDEN_SUB),
        (UNPK_FACTS, "unpk", UNPK_FORBIDDEN_WORD, UNPK_FORBIDDEN_SUB),
    ):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                item = json.loads(line)
                hits = _leak_hits(item.get("client_safe_text") or "", words, subs)
                if hits:
                    failures.append(f"{brand}:{item.get('fact_key')}: {hits}")

    snapshot = _load_snapshot()
    for brand, words, subs in (("foton", FOTON_FORBIDDEN_WORD, FOTON_FORBIDDEN_SUB), ("unpk", UNPK_FORBIDDEN_WORD, UNPK_FORBIDDEN_SUB)):
        for fact in snapshot.get(brand, []):
            hits = _leak_hits(fact, words, subs)
            if hits:
                failures.append(f"snapshot:{brand}:{hits}:{fact[:90]}")

    assert failures == []


def test_fewshot_numbers_are_present_in_v8_snapshot() -> None:
    snapshot = _load_snapshot()
    snapshot_numbers = {
        brand: set().union(*(_price_numbers(fact) for fact in facts)) if facts else set()
        for brand, facts in snapshot.items()
    }
    data = yaml.safe_load(FEWSHOT.read_text(encoding="utf-8"))
    failures: list[str] = []

    def walk(node: Any, brand: str = "") -> None:
        if isinstance(node, dict):
            current = node.get("brand") if isinstance(node.get("brand"), str) else brand
            for key, value in node.items():
                local_brand = current
                key_text = str(key).casefold()
                if "foton" in key_text or "фотон" in key_text:
                    local_brand = "foton"
                elif "unpk" in key_text or "унпк" in key_text:
                    local_brand = "unpk"
                walk(value, local_brand)
        elif isinstance(node, list):
            for item in node:
                walk(item, brand)
        elif isinstance(node, str) and brand in {"foton", "unpk"} and len(node) > 30:
            bad = sorted(_price_numbers(node) - snapshot_numbers.get(brand, set()))
            if bad:
                failures.append(f"{brand}:{bad}:{node[:110]}")

    walk(data)

    assert failures == []
