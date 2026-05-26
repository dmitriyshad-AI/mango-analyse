from __future__ import annotations

"""Самодостаточный прогон тестов референс-модуля (без pytest).

Запуск:
    python3 D1_semantic_roles_reference/tests/test_semantic_roles.py

Тесты проверяют ровно те классы, что всплыли на round-4:
  A. омонимы по смыслу слова (очно/точной, запись, перевод);
  B. типизированный возврат (presale vs dispute), вкл. провальный кейс;
  C. мультитему;
  D. запрет смешения осей оплаты (маткапитал × рассрочка);
  E. шаблон только как fallback.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REF = os.path.join(os.path.dirname(_HERE), "reference")
sys.path.insert(0, _REF)

from semantic_roles import tag_message_roles  # noqa: E402
from decision_policy import build_answer_plan  # noqa: E402


_PASS = 0
_FAIL = 0
_FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
    else:
        _FAIL += 1
        _FAILURES.append(f"[FAIL] {name}: {detail}")


def run_fixtures() -> None:
    path = os.path.join(os.path.dirname(_HERE), "fixtures", "word_trap_cases.jsonl")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            cid = case["id"]
            roles = tag_message_roles(case["msg"])
            plan = build_answer_plan(roles)

            for field_name, expected in (case.get("expect") or {}).items():
                actual = getattr(roles, field_name)
                check(f"{cid}:{field_name}", actual == expected, f"ждали {expected!r}, получили {actual!r}")

            for topic in case.get("expect_topics_present", []):
                check(f"{cid}:topic+{topic}", topic in plan.answer_topics or topic in roles.topics,
                      f"тема {topic} отсутствует; topics={roles.topics}")

            if "expect_topic_present" in case:
                t = case["expect_topic_present"]
                check(f"{cid}:topic+{t}", t in roles.topics, f"тема {t} отсутствует; topics={roles.topics}")

            if "expect_topic_absent" in case:
                t = case["expect_topic_absent"]
                check(f"{cid}:topic-{t}", t not in roles.topics, f"лишняя тема {t}; topics={roles.topics}")

            if "expect_forbidden_pair" in case:
                pair = case["expect_forbidden_pair"]
                check(f"{cid}:forbid:{pair}", pair in plan.forbidden_pairs,
                      f"нет запрета {pair}; forbidden={plan.forbidden_pairs}")


def run_policy_unit() -> None:
    # E. шаблон только как fallback
    empty = tag_message_roles("")
    plan_empty = build_answer_plan(empty, substantive_answer_present=False)
    check("template_fallback_when_nothing", plan_empty.template_allowed is True,
          f"template_allowed={plan_empty.template_allowed}")

    roles_price = tag_message_roles("Сколько стоит курс?")
    plan_price = build_answer_plan(roles_price, substantive_answer_present=True)
    check("template_blocked_when_substantive", plan_price.template_allowed is False,
          f"template_allowed={plan_price.template_allowed}")
    check("price_topic_present", "price" in plan_price.answer_topics,
          f"topics={plan_price.answer_topics}")

    # B. presale возврат → НЕ p0; dispute → p0
    presale = build_answer_plan(tag_message_roles("А если передумаю до начала, деньги вернут?"))
    check("presale_not_p0", presale.p0_required is False, f"p0={presale.p0_required}")
    check("presale_route_self", presale.route == "bot_answer_self", f"route={presale.route}")

    dispute = build_answer_plan(tag_message_roles("Верните деньги, я уже оплатил"))
    check("dispute_is_p0", dispute.p0_required is True, f"p0={dispute.p0_required}")
    check("dispute_route_manager", dispute.route == "manager_only", f"route={dispute.route}")

    # внешний P0 не ослабляется presale-логикой
    ext = build_answer_plan(tag_message_roles("Сколько стоит курс?"), external_p0=True)
    check("external_p0_preserved", ext.p0_required is True, f"p0={ext.p0_required}")


def main() -> int:
    run_fixtures()
    run_policy_unit()
    print(f"PASS={_PASS}  FAIL={_FAIL}")
    for f in _FAILURES:
        print(f)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
