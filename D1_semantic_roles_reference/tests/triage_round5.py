from __future__ import annotations

"""Офлайн-триаж round-5: прогон распознавателя ролей + политики по клиентским
репликам всех 24 персон holdout round-5 — БЕЗ вызова модели.

Что ловит: структурные решения слоя ролей (presale vs dispute, маршрут P0,
мультитема, запрет маткапитал×рассрочка, смысл «запись»/«перевод», переразметка
осей). Что НЕ ловит: тон, парафразы, фактические числа, генерацию — это round-5
на gpt-5.5 у Кодекса.

Запуск:
  python3 D1_semantic_roles_reference/tests/triage_round5.py
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.join(os.path.dirname(_HERE), "reference")
sys.path.insert(0, _REF)

from semantic_roles import tag_message_roles            # noqa: E402
from decision_policy import build_answer_plan            # noqa: E402

ROUND5 = "/sessions/confident-sleepy-darwin/mnt/Foton/v8_holdout_fresh_round5_2026-05-25.jsonl"

# Ожидания по структуре (только то, что входит в зону ролей).
# turn — индекс ключевой реплики в persona.behaviors.
# external_p0 — имитируем срабатывание ВНЕШНЕГО детектора (complaint/legal/...),
#   которого нет в слое ролей, чтобы проверить, что политика его не ослабляет.
# scope_note — класс вне зоны ролей (проверяется генерацией/фактами/др. модулем).
EXPECT = {
 "hold5_foton_p0_refund_dispute_01": dict(turn=1, refund_frame="dispute", p0=True, route="manager_only"),
 "hold5_foton_refund_presale_peredumayu_02": dict(turn=1, refund_frame="presale_policy", p0=False, route="bot_answer_self"),
 "hold5_foton_word_zapis_recording_03": dict(turn=0, enrollment_vs_recording="recording"),
 "hold5_foton_word_perevod_money_04": dict(turn=0, transfer_sense="money"),
 "hold5_foton_matkap_rassrochka_forbid_05": dict(turn=1, payment_source="matkap", forbidden="matkap+installment"),
 "hold5_foton_multitopic_price_installment_06": dict(turn=0, topics_any=["price", "installment"]),
 "hold5_foton_trial_offline_none_07": dict(turn=0, training_format="ochno", topics_any=["trial"], scope_note="бесплатность очного пробного — факт/генерация, не роли"),
 "hold5_foton_dolyami_no_parts_08": dict(turn=0, payment_method="dolyami", scope_note="«не называть число частей» — правило генерации"),
 "hold5_foton_discount_second_subject_online_09": dict(turn=0, training_format="online", topics_any=["discount"], scope_note="процент второго предмета — факт"),
 "hold5_foton_assumed_need_10": dict(turn=0, training_format="", payment_source="", topics_any=["price"], no_assumed=True),
 "hold5_foton_identity_11": dict(turn=0, topics_any=["identity"]),
 "hold5_foton_city_vs_camp_12": dict(turn=0, topics_any=["camp"], scope_note="город vs выезд — camp_scope_signals, вне слоя ролей"),
 "hold5_unpk_p0_complaint_latch_13": dict(turn=1, external_p0=True, p0=True, route="manager_only", scope_note="жалоба детектится COMPLAINT_RE, НЕ слоем ролей"),
 "hold5_unpk_refund_presale_notlike_14": dict(turn=1, refund_frame="presale_policy", p0=False, route="bot_answer_self"),
 "hold5_unpk_word_zapis_enroll_15": dict(turn=0, enrollment_vs_recording="enroll"),
 "hold5_unpk_word_perevod_manager_16": dict(turn=0, transfer_sense="manager"),
 "hold5_unpk_matkap_sfr_17": dict(turn=0, payment_source="matkap", forbidden="matkap+installment"),
 "hold5_unpk_tax_deduction_18": dict(turn=0, payment_source="tax_deduction", topics_any=["tax"]),
 "hold5_unpk_online_olymp_vs_regular_19": dict(turn=0, training_format="online", scope_note="олимпиадный онлайн только 9/11 — scope/факт"),
 "hold5_unpk_multitopic_format_schedule_20": dict(turn=0, topics_any=["format", "schedule"], format_is_question=True),
 "hold5_unpk_discount_year_14_21": dict(turn=0, topics_any=["discount"], scope_note="14%/10% — факт"),
 "hold5_unpk_over_handoff_addr_22": dict(turn=0, asks_place=True, topics_any=["address"], scope_note="over_handoff — поведение генерации"),
 "hold5_unpk_no_bank_rassrochka_23": dict(turn=0, payment_method="rassrochka", scope_note="«нет банковской рассрочки + не упоминать Долями» — факт/бренд"),
 "hold5_unpk_assumed_need_24": dict(turn=0, topics_any=["price"], no_assumed=True),
}


def load_personas():
    out = {}
    with open(ROUND5, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("type") == "persona":
                out[d["dialog_id"]] = d
    return out


def main() -> int:
    personas = load_personas()
    fails = []
    findings = []
    notes = []
    checked = 0

    for did, exp in EXPECT.items():
        p = personas.get(did)
        if not p:
            fails.append(f"{did}: персона не найдена в round-5")
            continue
        beh = p["behaviors"]
        turn = exp["turn"]
        msg = beh[turn]
        roles = tag_message_roles(msg)
        plan = build_answer_plan(roles, external_p0=exp.get("external_p0", False))
        checked += 1

        def fail(detail):
            fails.append(f"{did} | «{msg}» → {detail}")

        for field in ("refund_frame", "enrollment_vs_recording", "transfer_sense",
                      "payment_source", "payment_method", "training_format", "asks_place"):
            if field in exp:
                actual = getattr(roles, field)
                if actual != exp[field]:
                    fail(f"{field}: ждали {exp[field]!r}, получили {actual!r}")

        if "p0" in exp and plan.p0_required != exp["p0"]:
            fail(f"p0_required: ждали {exp['p0']}, получили {plan.p0_required}")
        if "route" in exp and plan.route != exp["route"]:
            fail(f"route: ждали {exp['route']}, получили {plan.route}")
        if "forbidden" in exp and exp["forbidden"] not in plan.forbidden_pairs:
            fail(f"нет запрета {exp['forbidden']}; forbidden={plan.forbidden_pairs}")
        for t in exp.get("topics_any", []):
            if t not in roles.topics and t not in plan.answer_topics:
                fail(f"тема {t} отсутствует; topics={list(roles.topics)}")
        if exp.get("no_assumed") and (roles.training_format or roles.payment_source or roles.payment_method):
            fail(f"подставлена ось без запроса клиента: fmt={roles.training_format} src={roles.payment_source} method={roles.payment_method}")

        # спец-проверка: дизъюнктный вопрос о формате не должен латчить один формат
        if exp.get("format_is_question") and roles.training_format in ("online", "ochno"):
            findings.append(f"{did} | «{msg}» → training_format={roles.training_format!r}, "
                            f"но клиент СПРАШИВАЕТ формат (онлайн ИЛИ очно), ось не должна латчиться")

        if exp.get("scope_note"):
            notes.append(f"{did}: {exp['scope_note']}")

    print(f"=== ТРИАЖ round-5: проверено персон {checked}/24 ===")
    print(f"СТРУКТУРНЫХ ПРОВАЛОВ: {len(fails)}")
    for f in fails:
        print("  [FAIL]", f)
    print(f"\nНАХОДКИ (требуют правки распознавателя): {len(findings)}")
    for f in findings:
        print("  [BUG]", f)
    print(f"\nВНЕ ЗОНЫ РОЛЕЙ (проверит round-5/факты, не слой ролей): {len(notes)}")
    for n in notes:
        print("  [scope]", n)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
