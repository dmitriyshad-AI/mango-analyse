from __future__ import annotations

"""Recall-проба: измеряет % попадания факта-ОТВЕТА в confirmed_facts на папке прогона.

Это численная метрика слоя извлечения (как «отправил бы» для качества). Детерминированно,
без модели. Запускать на dynamic_dialog_transcripts.jsonl любого прогона round-5.

Банк проб: вопросы round-5, для которых факт-ответ ТОЧНО есть в рантайм client-safe базе
(проверено Claude 2026-05-25). Если факт не в confirmed_facts на ходе вопроса — промах
извлечения (recall miss), а НЕ пробел знаний.

Запуск:
  python3 D1_semantic_roles_reference/tests/recall_probe.py <папка_прогона>
  (по умолчанию — последний round-5)
"""

import json
import os
import sys

DEFAULT_RUN = "/Users/dmitrijfabarisov/Projects/Mango analyse/audits/_inbox/holdout_round5_2026-05-25_033336"
# в песочнице путь монтируется иначе:
_ALT = "/sessions/confident-sleepy-darwin/mnt/Mango analyse/audits/_inbox/holdout_round5_2026-05-25_033336"

# (dialog_id, подстрока в реплике клиента, ожидаемая подстрока в confirmed_facts)
# expect — допускаются альтернативы через "|" (любое совпадение = OK), чтобы probe мерил
# РЕАЛЬНЫЙ recall факта-ответа, а не одну формулировку. (Правка 2026-05-25: запись урока в
# базе формулируется «доступны для пересмотра», а не только «сохраняются» — было ложное MISS.)
PROBES = [
    ("hold5_unpk_matkap_sfr_17", "сфр", "10 рабоч"),
    ("hold5_unpk_over_handoff_addr_22", "москв", "сретен"),
    ("hold5_foton_word_zapis_recording_03", "запис", "сохран|пересмотр|записи занят|записи урок"),
    ("hold5_unpk_discount_year_14_21", "за год", "14"),
    ("hold5_unpk_online_olymp_vs_regular_19", "10 класс", "9 и 11"),
    ("hold5_foton_trial_offline_none_07", "пробн", "фрагмент"),
    ("hold5_foton_city_vs_camp_12", "без прожив", "городск"),
    ("hold5_unpk_multitopic_format_schedule_20", "дням", "выходн"),
]


def cf_text(turn) -> str:
    out = []
    for x in (turn.get("bot_confirmed_facts") or []):
        if isinstance(x, dict):
            out.append(x.get("client_safe_text") or x.get("fact_text") or "")
        else:
            out.append(str(x))
    for x in (turn.get("bot_knowledge_snippets") or []):
        out.append(str(x))
    return " ".join(out).lower()


def main() -> int:
    run = sys.argv[1] if len(sys.argv) > 1 else (DEFAULT_RUN if os.path.isdir(DEFAULT_RUN) else _ALT)
    path = os.path.join(run, "dynamic_dialog_transcripts.jsonl")
    if not os.path.isfile(path):
        print("не найден:", path)
        return 2
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                d = json.loads(line)
                rows[d.get("dialog_id")] = d

    hit = 0
    tot = 0
    print(f"=== RECALL-проба: {os.path.basename(run)} ===")
    for did, needle, expect in PROBES:
        d = rows.get(did)
        if not d:
            print(f"  [нет диалога] {did}")
            continue
        turn = None
        for t in d.get("turns", []):
            if needle in (t.get("client_message") or "").lower():
                turn = t
                break
        if turn is None:
            print(f"  [needle не найден] {did}")
            continue
        tot += 1
        cf = cf_text(turn)
        present = any(alt in cf for alt in expect.lower().split("|"))
        hit += 1 if present else 0
        print(f"  {'OK ' if present else 'MISS'} {did}: факт-ответ '{expect}' {'в confirmed' if present else 'НЕ извлёкся'}")
    pct = 100 * hit / max(tot, 1)
    print(f"\nRECALL извлечения: {hit}/{tot} = {pct:.0f}%  (цель слоя извлечения: >= 90%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
