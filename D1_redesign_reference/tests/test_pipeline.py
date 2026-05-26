"""Тесты ПОЛНОГО потока v2 на моке (без живых вызовов).
Запуск: PYTHONPATH=../reference python3 -m pytest test_pipeline.py  (или простой раннер)."""

from pipeline import run_pipeline, Toggles, build_draft_prompt
from dialogue_understanding import parse_contract, AnswerContract, Slot, Subquestion

CATALOG = ["price.online", "recording.access", "schedule.exact_day"]
STORE = {
    "foton": {
        "price.online": "семестр — 29 750 ₽, год — 47 250 ₽",
        "recording.access": "запись доступна на платформе МТС Линк после урока",
    },
    "unpk": {"price.online": "семестр — 41 800 ₽"},
}


def conv(last, prior=()):
    msgs = [{"role": "client" if i % 2 == 0 else "bot", "text": t} for i, t in enumerate(prior)]
    msgs.append({"role": "client", "text": last})
    return msgs


def uf(d):
    return lambda p: d


# --- контракт-план: подвопросы, слоты-с-источником, client_state ---
def test_contract_subquestions_and_slots():
    c = parse_contract({
        "current_question": "цена и запись",
        "client_state": "сравнивает цену",
        "subquestions": [
            {"text": "цена онлайн", "answerable": "self", "needed_fact_keys": ["price.online"], "next_step": "подобрать группу"},
            {"text": "будет ли запись", "answerable": "self", "needed_fact_keys": ["recording.access"]},
        ],
        "known_slots": {"class": {"value": "9", "source": "client_turn_1"}, "subject": {"value": "физика"}},
        "answerability": "answer_self", "confidence": 0.8,
    }, active_brand="foton")
    assert len(c.subquestions) == 2 and c.all_needed_fact_keys() == ("price.online", "recording.access")
    # слот с источником — утверждаемый; без источника — нет
    assert c.assertable_slots() == {"class": "9"} and c.unsourced_slots() == ("subject",)
    assert c.client_state == "сравнивает цену"


def test_draft_prompt_excludes_unsourced_slots():
    c = parse_contract({"current_question": "x", "subquestions": [{"text": "x", "answerable": "self"}],
                        "known_slots": {"class": {"value": "9", "source": "client_turn_1"}, "subject": {"value": "физика"}},
                        "answerability": "answer_self"}, active_brand="foton")
    p = build_draft_prompt(conversation=conv("x"), contract=c, facts={}, missing=())
    assert "9" in p and "физика" not in p  # неисточниковый слот не попадает в промпт → не утверждается


# --- базовый happy ---
def test_happy():
    out = run_pipeline(
        conversation=conv("сколько онлайн?"), active_brand="foton", fact_store=STORE, catalog=CATALOG,
        understand_fn=uf({"current_question": "цена онлайн", "answerability": "answer_self",
                          "subquestions": [{"text": "цена онлайн", "answerable": "self", "needed_fact_keys": ["price.online"]}]}),
        draft_fn=lambda p: "По онлайну: семестр — 29 750 ₽, год — 47 250 ₽. Подобрать группу?",
        faithfulness_fn=lambda p: {"unsupported": []},
    )
    assert out.route == "bot_answer_self" and not out.hard_findings and "29 750" in out.draft_text


def test_p0_pregate():
    out = run_pipeline(conversation=conv("я оплатил, занятий нет, верните деньги"), active_brand="foton",
                       fact_store=STORE, catalog=CATALOG, understand_fn=uf({"answerability": "answer_self"}),
                       draft_fn=lambda p: "цена")
    assert out.manager_only and "Приняли обращение" in out.draft_text


def test_contract_manager_only():
    out = run_pipeline(conversation=conv("дайте живого человека"), active_brand="unpk", fact_store=STORE, catalog=CATALOG,
                       understand_fn=uf({"answerability": "manager_only"}), draft_fn=lambda p: "x")
    assert out.manager_only and out.fallback_reason == "contract_manager_only"


# --- §11.3 семантическая верность ловит НЕ-числовую выдумку ---
def test_semantic_faithfulness_catches_nonnumeric_then_repairs():
    out = run_pipeline(
        conversation=conv("когда занятия?"), active_brand="unpk", fact_store=STORE, catalog=CATALOG,
        understand_fn=uf({"current_question": "дни занятий", "answerability": "answer_self",
                          "subquestions": [{"text": "дни", "answerable": "self", "needed_fact_keys": ["schedule.exact_day"]}]}),
        draft_fn=lambda p: "Занятия по будням вечером.",   # выдумка, чисел нет
        faithfulness_fn=lambda p: {"unsupported": ["занятия по будням вечером"]} if "будням" in p else {"unsupported": []},
        repair_fn=lambda p: "Точные дни уточнит менеджер по вашей группе.",
    )
    assert out.route == "bot_answer_self" and out.repaired and "будням" not in out.draft_text


def test_semantic_faithfulness_fallback_if_not_repaired():
    out = run_pipeline(
        conversation=conv("когда занятия?"), active_brand="unpk", fact_store=STORE, catalog=CATALOG,
        understand_fn=uf({"current_question": "дни", "answerability": "answer_self",
                          "subquestions": [{"text": "дни", "answerable": "self"}]}),
        draft_fn=lambda p: "Занятия по будням вечером.",
        faithfulness_fn=lambda p: {"unsupported": ["занятия по будням вечером"]},
        repair_fn=lambda p: "Занятия по будням вечером, точно.",  # не чинит
    )
    assert out.route == "draft_for_manager" and out.fallback_reason == "hard_verification_failed"


def test_semantic_toggle_off_skips_check():
    out = run_pipeline(
        conversation=conv("когда?"), active_brand="unpk", fact_store=STORE, catalog=CATALOG,
        understand_fn=uf({"current_question": "дни", "answerability": "answer_self",
                          "subquestions": [{"text": "дни", "answerable": "self"}]}),
        draft_fn=lambda p: "Занятия по будням. Подобрать группу?",
        faithfulness_fn=lambda p: {"unsupported": ["занятия по будням"]},  # сработал бы, но тумблер выключен
        toggles=Toggles(semantic_faithfulness=False),
    )
    assert out.route == "bot_answer_self" and not out.unsupported_claims


# --- §12 form-check → X2-тепло, с ре-проверкой ---
def test_form_check_triggers_warmth():
    out = run_pipeline(
        conversation=conv("цена онлайн?"), active_brand="foton", fact_store=STORE, catalog=CATALOG,
        understand_fn=uf({"current_question": "цена", "client_state": "сомневается", "answerability": "answer_self",
                          "subquestions": [{"text": "цена", "answerable": "self", "needed_fact_keys": ["price.online"]}]}),
        draft_fn=lambda p: "По проверенным данным семестр 29 750 ₽.",   # штамп-зачин + нет шага
        faithfulness_fn=lambda p: {"unsupported": []},
        warmth_fn=lambda p: "Конечно! Онлайн семестр — 29 750 ₽. Подобрать удобную группу?",
    )
    assert out.warmed and "По проверенным данным" not in out.draft_text and "29 750" in out.draft_text


def test_warmth_reverified_keeps_original_if_warm_breaks_facts():
    out = run_pipeline(
        conversation=conv("цена онлайн?"), active_brand="foton", fact_store=STORE, catalog=CATALOG,
        understand_fn=uf({"current_question": "цена", "answerability": "answer_self",
                          "subquestions": [{"text": "цена", "answerable": "self", "needed_fact_keys": ["price.online"]}]}),
        draft_fn=lambda p: "По проверенным данным семестр 29 750 ₽.",
        faithfulness_fn=lambda p: {"unsupported": []},
        warmth_fn=lambda p: "Ура! Сейчас всего 19 900 ₽!",   # тепло, но выдумка цены
    )
    # тёплый кандидат не прошёл ре-проверку → остаётся верифицированный исходный
    assert not out.warmed and "29 750" in out.draft_text and "19 900" not in out.draft_text


# --- §risk1: семантическая проверка FAIL-CLOSED (сбой/мусор → не PASS, а fallback) ---
def _base_self(q="цена"):
    return {"current_question": q, "answerability": "answer_self",
            "subquestions": [{"text": q, "answerable": "self", "needed_fact_keys": ["price.online"]}]}


def test_semantic_unavailable_on_exception_fail_closed():
    def boom(p):
        raise RuntimeError("faithfulness down")
    out = run_pipeline(
        conversation=conv("цена онлайн?"), active_brand="foton", fact_store=STORE, catalog=CATALOG,
        understand_fn=uf(_base_self()), draft_fn=lambda p: "Семестр 29 750 ₽. Подобрать группу?",
        faithfulness_fn=boom,
    )
    assert out.route == "draft_for_manager" and out.fallback_reason == "semantic_check_unavailable" and not out.manager_only


def test_semantic_unavailable_on_garbage_fail_closed():
    out = run_pipeline(
        conversation=conv("цена онлайн?"), active_brand="foton", fact_store=STORE, catalog=CATALOG,
        understand_fn=uf(_base_self()), draft_fn=lambda p: "Семестр 29 750 ₽. Подобрать группу?",
        faithfulness_fn=lambda p: "не json и не список",   # мусор → недоступно
    )
    assert out.fallback_reason == "semantic_check_unavailable"


# --- бренд-утечка → fallback ---
def test_brand_leak_fallback():
    out = run_pipeline(
        conversation=conv("цена онлайн?"), active_brand="foton", fact_store=STORE, catalog=CATALOG,
        understand_fn=uf({"current_question": "цена", "answerability": "answer_self",
                          "subquestions": [{"text": "цена", "answerable": "self", "needed_fact_keys": ["price.online"]}]}),
        draft_fn=lambda p: "У нас 29 750 ₽, в УНПК дешевле.",
        faithfulness_fn=lambda p: {"unsupported": []},
        repair_fn=lambda p: "У нас 29 750 ₽, в УНПК дешевле.",
    )
    assert out.route == "draft_for_manager" and out.fallback_reason == "hard_verification_failed"
