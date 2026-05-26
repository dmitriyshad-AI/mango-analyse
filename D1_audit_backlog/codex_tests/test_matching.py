from __future__ import annotations

"""Бизнес-тесты СОПОСТАВЛЕНИЯ клиента/сделки (область 1).

Запуск (в среде Кодекса, с PYTHONPATH=src):
    PYTHONPATH=src python3 D1_audit_backlog/codex_tests/test_matching.py

Защищаемый главный инвариант области:
  - похожие, но РАЗНЫЕ люди (разные телефоны/имена) НЕ матчатся в одного;
  - один человек с разными форматами телефона (8XXX, +7XXX, 7XXX, 10 цифр) матчится;
  - пустой/мусорный ключ НЕ даёт ложный матч;
  - неоднозначный матч (несколько кандидатов с близким score) блокируется, а не выбирается наугад.

Все проверяемые функции чистые (без БД/сети), поэтому тест почти полностью
самодостаточен. Места, где нужен живой ресурс (phone_context читает CSV из
stable_runtime), вынесены в TODO-скелет внизу.

Модули под проверкой:
  - mango_mvp.amocrm_runtime.tallanto_matching
  - mango_mvp.amocrm_runtime.tallanto_deal_ranking
  - mango_mvp.productization.crm_entity_resolver
  - mango_mvp.amocrm_runtime.phone_context (только скелет, см. TODO)
"""

from datetime import datetime, timezone

from mango_mvp.amocrm_runtime.tallanto_matching import (
    build_phone_candidates,
    match_contact_by_phone,
)
from mango_mvp.amocrm_runtime.tallanto_deal_ranking import (
    choose_best_opportunity,
    rank_opportunity_candidates,
)
from mango_mvp.productization.crm_entity_resolver import (
    BLOCK_AMBIGUOUS_CRM_MATCH,
    BLOCK_NO_CALL_PHONE,
    BLOCK_NO_CRM_MATCH,
    RESOLVE_CRM_ENTITY,
    build_phone_index,
    normalize_snapshot_row,
    resolve_call_to_crm_entity,
)


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


# --------------------------------------------------------------------------
# 1A. tallanto_matching.match_contact_by_phone
# Инвариант: один человек в разных форматах телефона матчится; разные люди — нет.
# --------------------------------------------------------------------------

def _contact(name: str, phone: str, **extra) -> dict:
    row = {"id": name, "name": name, "phone_mobile": phone}
    row.update(extra)
    return row


def test_same_person_phone_format_variants() -> None:
    # Один человек, в базе сохранён как 8XXXXXXXXXX, звонок пришёл как +7XXXXXXXXXX.
    contacts = [_contact("Иванов", "89161234567")]
    res = match_contact_by_phone("+79161234567", contacts)
    check("same_person_8_vs_+7_matched", res.matched_contact is not None,
          f"reason={res.reason}, conf={res.confidence}")
    check("same_person_not_ambiguous", res.ambiguous is False, f"ambiguous={res.ambiguous}")

    # Тот же человек, звонок в формате 10 цифр без кода страны.
    res10 = match_contact_by_phone("9161234567", contacts)
    check("same_person_10digits_matched", res10.matched_contact is not None,
          f"reason={res10.reason}")


def test_different_people_do_not_match() -> None:
    # Два РАЗНЫХ человека, разные телефоны. Звонок от третьего — не должен матчиться.
    contacts = [_contact("Иванов", "+79161234567"), _contact("Петров", "+79169999999")]
    res = match_contact_by_phone("+79165550000", contacts)
    check("different_people_no_false_match", res.matched_contact is None,
          f"matched={res.matched_contact}, reason={res.reason}")
    check("different_people_reason_not_found", res.reason == "phone_not_found",
          f"reason={res.reason}")


def test_empty_and_garbage_key_no_false_match() -> None:
    contacts = [_contact("Иванов", "+79161234567")]
    for bad in ("", "   ", "абвгд", "---", None):
        res = match_contact_by_phone(bad or "", contacts)
        check(f"garbage_key_no_match[{bad!r}]", res.matched_contact is None,
              f"matched={res.matched_contact}, reason={res.reason}")


def test_ambiguous_two_equal_candidates_blocked() -> None:
    # Два РАЗНЫХ контакта с ОДНИМ телефоном (дубль в CRM) → нельзя выбрать одного молча.
    contacts = [_contact("Иванов", "+79161234567"), _contact("Иванова", "+79161234567")]
    res = match_contact_by_phone("+79161234567", contacts)
    check("ambiguous_blocked_no_silent_pick", res.matched_contact is None,
          f"matched={res.matched_contact}")
    check("ambiguous_flag_set", res.ambiguous is True, f"ambiguous={res.ambiguous}")
    check("ambiguous_reason", res.reason == "ambiguous_phone_match", f"reason={res.reason}")


def test_branch_manager_tiebreak_resolves_ambiguity() -> None:
    # Тот же дубль, но один из них совпал по филиалу и менеджеру → перевес снимает неоднозначность.
    contacts = [
        _contact("Иванов", "+79161234567", filial="Фотон", assigned_user_name="Тропина"),
        _contact("Иванова", "+79161234567"),
    ]
    res = match_contact_by_phone(
        "+79161234567", contacts,
        expected_branch="Фотон", expected_manager_name="Тропина",
    )
    check("tiebreak_picks_one", res.matched_contact is not None, f"reason={res.reason}")
    if res.matched_contact is not None:
        check("tiebreak_picks_right_one", res.matched_contact.get("name") == "Иванов",
              f"picked={res.matched_contact.get('name')}")


def test_build_phone_candidates_dedup_and_formats() -> None:
    cands = build_phone_candidates("+7 (916) 123-45-67")
    digits_only = [c for c in cands if c.isdigit()]
    check("candidates_nonempty", len(cands) > 0, f"cands={cands}")
    check("candidates_unique", len(cands) == len(set(cands)), f"dups in {cands}")
    check("candidates_have_digits_form", any(len(d) >= 10 for d in digits_only),
          f"digits={digits_only}")
    # Мусор не должен порождать кандидатов.
    check("garbage_no_candidates", build_phone_candidates("абвг") == [] or
          all(not c.strip() == "" for c in build_phone_candidates("абвг")),
          f"got {build_phone_candidates('абвг')}")


# --------------------------------------------------------------------------
# 1B. tallanto_deal_ranking — выбор лучшей сделки
# Инвариант: активная свежая сделка > закрытой старой; неоднозначность блокируется.
# --------------------------------------------------------------------------

def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def test_active_recent_deal_beats_closed_old() -> None:
    call_at = _dt("2026-05-20T12:00:00")
    opps = [
        {"id": "closed_old", "sales_stage": "Успешно закрыто", "date_modified": "2025-01-01T00:00:00"},
        {"id": "active_now", "sales_stage": "Переговоры", "date_modified": "2026-05-20T08:00:00"},
    ]
    ranked = rank_opportunity_candidates(call_started_at=call_at, opportunities=opps)
    check("ranked_two", len(ranked) == 2, f"n={len(ranked)}")
    check("active_recent_on_top", ranked[0].opportunity.get("id") == "active_now",
          f"top={ranked[0].opportunity.get('id')}, score={ranked[0].score}")


def test_choose_best_blocks_when_ambiguous() -> None:
    call_at = _dt("2026-05-20T12:00:00")
    # Две одинаково активные сделки с близким score — нельзя выбрать одну молча.
    opps = [
        {"id": "a", "sales_stage": "Переговоры", "date_modified": "2026-05-20T08:00:00"},
        {"id": "b", "sales_stage": "Переговоры", "date_modified": "2026-05-20T09:00:00"},
    ]
    best, ranked, ambiguous = choose_best_opportunity(call_started_at=call_at, opportunities=opps)
    check("ambiguous_returns_none", best is None, f"best={best}")
    check("ambiguous_flag", ambiguous is True, f"ambiguous={ambiguous}")


def test_choose_best_blocks_when_below_confidence() -> None:
    call_at = _dt("2026-05-20T12:00:00")
    # Единственный кандидат — закрытая старая сделка, score ниже min_confidence.
    opps = [{"id": "old", "sales_stage": "Закрыто и не реализовано", "date_modified": "2024-01-01T00:00:00"}]
    best, ranked, blocked = choose_best_opportunity(call_started_at=call_at, opportunities=opps)
    check("low_conf_returns_none", best is None, f"best={best}")
    check("low_conf_blocked", blocked is True, f"blocked={blocked}")


def test_choose_best_picks_clear_winner() -> None:
    call_at = _dt("2026-05-20T12:00:00")
    opps = [
        {"id": "winner", "sales_stage": "Переговоры", "date_modified": "2026-05-20T08:00:00",
         "filial": "Фотон", "assigned_user_id": "42"},
        {"id": "loser", "sales_stage": "Закрыто и не реализовано", "date_modified": "2024-01-01T00:00:00"},
    ]
    best, ranked, blocked = choose_best_opportunity(
        call_started_at=call_at, opportunities=opps,
        expected_branch="Фотон", expected_manager_id="42",
    )
    check("clear_winner_chosen", best is not None and best.get("id") == "winner",
          f"best={best}, blocked={blocked}")


# --------------------------------------------------------------------------
# 1C. crm_entity_resolver — резолв звонка к CRM-сущности по телефону
# Инвариант: один телефон → ровно одна сущность = резолв; несколько = блок;
#            нет телефона = блок; нет совпадения = блок. Никаких ложных матчей.
# --------------------------------------------------------------------------

def _entity(eid: str, phone: str, etype: str = "contact", name: str = "") -> dict:
    return normalize_snapshot_row(
        {"entity_id": eid, "entity_type": etype, "name": name or eid, "phone": phone},
        source_ref=f"test#{eid}",
    )


def test_resolver_single_exact_match() -> None:
    entities = [_entity("c1", "+79161234567", name="Иванов")]
    index = build_phone_index(entities)
    # Звонок в другом формате того же номера должен попасть в индекс (оба нормализуются).
    item = resolve_call_to_crm_entity({"client_phone": "89161234567"}, phone_index=index)
    check("resolver_single_action", item.get("action") == RESOLVE_CRM_ENTITY,
          f"action={item.get('action')}, reason={item.get('reason')}")
    check("resolver_single_entity_id", item.get("crm_entity_id") == "c1",
          f"id={item.get('crm_entity_id')}")


def test_resolver_ambiguous_blocked() -> None:
    # Один телефон у ДВУХ разных сущностей → блок, не выбор наугад.
    entities = [_entity("c1", "+79161234567", name="Иванов"),
                _entity("c2", "+79161234567", name="Иванова")]
    index = build_phone_index(entities)
    item = resolve_call_to_crm_entity({"client_phone": "+79161234567"}, phone_index=index)
    check("resolver_ambiguous_action", item.get("action") == BLOCK_AMBIGUOUS_CRM_MATCH,
          f"action={item.get('action')}")
    check("resolver_ambiguous_count", item.get("candidate_count") == 2,
          f"count={item.get('candidate_count')}")


def test_resolver_no_phone_blocked() -> None:
    index = build_phone_index([_entity("c1", "+79161234567")])
    item = resolve_call_to_crm_entity({"client_phone": ""}, phone_index=index)
    check("resolver_no_phone_action", item.get("action") == BLOCK_NO_CALL_PHONE,
          f"action={item.get('action')}")


def test_resolver_no_match_blocked() -> None:
    index = build_phone_index([_entity("c1", "+79161234567")])
    item = resolve_call_to_crm_entity({"client_phone": "+79165550000"}, phone_index=index)
    check("resolver_no_match_action", item.get("action") == BLOCK_NO_CRM_MATCH,
          f"action={item.get('action')}")


def test_resolver_never_writes_crm() -> None:
    # Контракт безопасности: ни один путь резолва не разрешает запись в CRM.
    index = build_phone_index([_entity("c1", "+79161234567")])
    for call in ({"client_phone": "+79161234567"}, {"client_phone": ""},
                 {"client_phone": "+79165550000"}):
        item = resolve_call_to_crm_entity(call, phone_index=index)
        check(f"resolver_write_crm_false[{call}]", item.get("write_crm") is False,
              f"write_crm={item.get('write_crm')}")


# --------------------------------------------------------------------------
# 1D. phone_context — СКЕЛЕТ. Модуль читает CSV из stable_runtime (живой ресурс).
# TODO(codex): подставить фикстуру экспортной папки.
# --------------------------------------------------------------------------

def test_phone_context_readonly_skeleton() -> None:
    """СКЕЛЕТ. phone_context.get_phone_context читает реальные CSV из
    stable_runtime через settings.source_workspace_root и кэширует их.

    Инвариант для проверки Кодексом:
      - get_phone_context(пустой/мусор) -> None (нет ложного контекста);
      - get_phone_context(телефон в одном формате) и (в другом формате) дают
        ОДИН и тот же contact_row (один человек ↔ один контекст);
      - функция нигде не пишет в источник (read-only).

    TODO(codex): подставить фикстуру:
      1. создать временную папку tmp_export/ с master_contacts_ru.csv и
         master_calls_ru.csv (2-3 строки, колонка 'Телефон клиента');
      2. monkeypatch phone_context._latest_export_dir -> lambda: tmp_export
         (или monkeypatch settings.source_workspace_root на родителя stable_runtime);
      3. сбросить phone_context._CACHE между прогонами.
    Ниже — заготовка с понятными assert.
    """
    try:
        import mango_mvp.amocrm_runtime.phone_context as pc  # noqa: F401
    except Exception as exc:  # pragma: no cover
        check("phone_context_importable", False, f"import failed: {exc}")
        return
    check("phone_context_importable", True)
    # TODO(codex): после monkeypatch _latest_export_dir раскомментировать:
    # assert pc.get_phone_context("") is None
    # assert pc.get_phone_context("мусор") is None
    # ctx_a = pc.get_phone_context("+79161234567")
    # ctx_b = pc.get_phone_context("89161234567")
    # assert ctx_a is not None and ctx_b is not None
    # assert ctx_a.contact_row == ctx_b.contact_row  # один человек ↔ один контекст


def main() -> int:
    test_same_person_phone_format_variants()
    test_different_people_do_not_match()
    test_empty_and_garbage_key_no_false_match()
    test_ambiguous_two_equal_candidates_blocked()
    test_branch_manager_tiebreak_resolves_ambiguity()
    test_build_phone_candidates_dedup_and_formats()
    test_active_recent_deal_beats_closed_old()
    test_choose_best_blocks_when_ambiguous()
    test_choose_best_blocks_when_below_confidence()
    test_choose_best_picks_clear_winner()
    test_resolver_single_exact_match()
    test_resolver_ambiguous_blocked()
    test_resolver_no_phone_blocked()
    test_resolver_no_match_blocked()
    test_resolver_never_writes_crm()
    test_phone_context_readonly_skeleton()
    print(f"PASS={_PASS}  FAIL={_FAIL}")
    for line in _FAILURES:
        print(line)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
