from __future__ import annotations

"""Офлайн-симуляция полного v8 (322 персоны) для слоя ролей — БЕЗ вызова модели.

Оракула на каждую из 322 персон нет, поэтому это АНОМАЛИЯ-ДЕТЕКТОР: прогоняем
распознаватель+политику по каждой клиентской реплике и ловим структурные
нарушения инвариантов (то, что почти наверняка баг логики слоя ролей,
независимо от ожидаемого ответа). Каждый кандидат — повод прочитать реплику,
а не приговор. Тон/факты/парафразы — НЕ здесь (это round-5 на gpt-5.5).

Запуск:
  python3 D1_semantic_roles_reference/tests/sim_full_v8.py
"""

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.join(os.path.dirname(_HERE), "reference")
sys.path.insert(0, _REF)

from semantic_roles import tag_message_roles            # noqa: E402
from decision_policy import build_answer_plan            # noqa: E402
from held_state import HeldState, update_held            # noqa: E402

V8 = "/sessions/confident-sleepy-darwin/mnt/Foton/v8_dynamic_client_sim_2026-05-25_v2.jsonl"

# Широкие сигналы для инвариантов (намеренно ШИРЕ, чем продакшен-маркеры, —
# чтобы поймать пропуск распознавателя).
_REFUND_CUE = re.compile(r"возврат|верн[уои]\w*\s+(?:деньг|оплат|средств|сумм)|деньги\s+назад|расторг|отказ\w*\s+от\s+обуч|забрать\s+деньг", re.I)
_DEMAND = re.compile(r"верните|требую|хочу\s+(?:возврат|вернуть|деньги\s+назад)|немедленно\s+верн", re.I)
_PAID = re.compile(r"оплатил|оплатила|уже\s+оплат|после\s+оплат|списал|заключил\w*\s+договор|мы\s+платил|я\s+платил|с\s+меня\s+сн", re.I)
_PRESALE = re.compile(r"\bесли\b|вдруг|до\s+начал|до\s+оплат|перед\s+оплат|заранее|передума|не\s+понрав|не\s+подойд|какие\s+услови|какие\s+правил", re.I)
_ZAPIS = re.compile(r"запис|оформ", re.I)
_TRANSFER = re.compile(r"перевод|перевест|перевед|переведите|переключите", re.I)


def main() -> int:
    personas = []
    with open(V8, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") == "persona":
                personas.append(d)

    cats: dict[str, list[str]] = {}
    n_msgs = 0

    def flag(cat, did, msg, extra=""):
        cats.setdefault(cat, []).append(f"[{did}] «{msg}»{(' — ' + extra) if extra else ''}")

    for p in personas:
        did = p.get("dialog_id", "?")
        held = HeldState()  # многоходовость: держим состояние по ходам персоны
        for msg in p.get("behaviors", []):
            text = str(msg or "")
            if not text.strip():
                continue
            n_msgs += 1
            roles = tag_message_roles(text, context=held.tagger_context())
            plan = build_answer_plan(roles, external_p0=held.p0_latched)
            held = update_held(held, text, roles, p0_required=plan.p0_required)

            refund_cue = bool(_REFUND_CUE.search(text))
            demand = bool(_DEMAND.search(text))
            paid = bool(_PAID.search(text))
            presale = bool(_PRESALE.search(text))

            # INV1: упоминание возврата есть, а роль его не увидела
            if refund_cue and roles.refund_frame == "none":
                flag("refund_missed", did, text)
            # INV2: помечен спором, хотя рамка явно предпродажная (нет требования/оплаты)
            if roles.refund_frame == "dispute" and presale and not demand and not paid:
                flag("refund_false_dispute", did, text, f"ev={roles.evidence.get('refund_frame')}")
            # INV3: инвариант запрета (код обязан ставить пару при matkap)
            if roles.payment_source == "matkap" and "matkap+installment" not in plan.forbidden_pairs:
                flag("matkap_invariant_break", did, text)
            # INV4: слово «запись/оформ» есть, а ось не разрешена
            if _ZAPIS.search(text) and roles.enrollment_vs_recording == "":
                flag("zapis_unresolved", did, text)
            # INV5: слово «перевод/переключите» есть, а смысл не разрешён
            if _TRANSFER.search(text) and roles.transfer_sense == "":
                flag("transfer_unresolved", did, text)
            # INV6: реплика-вопрос про возврат-спор, но не P0 (страховка пола безопасности)
            if roles.refund_frame == "dispute" and not plan.p0_required:
                flag("dispute_not_p0", did, text)

    print(f"=== СИМУЛЯЦИЯ v8: персон {len(personas)}, реплик клиента {n_msgs} ===")
    order = ["refund_missed", "refund_false_dispute", "dispute_not_p0",
             "matkap_invariant_break", "zapis_unresolved", "transfer_unresolved"]
    total = 0
    for cat in order:
        items = cats.get(cat, [])
        total += len(items)
        print(f"\n## {cat}: {len(items)}")
        for s in items[:12]:
            print("   ", s)
        if len(items) > 12:
            print(f"    … ещё {len(items) - 12}")
    print(f"\nИТОГО кандидатов-аномалий: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
