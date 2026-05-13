from __future__ import annotations

from mango_mvp.quality.amo_loss_reason_policy import classify_amo_loss_reason, primary_amo_loss_reason_policy


def _categories(reason: str) -> set[str]:
    return {policy.category for policy in classify_amo_loss_reason(reason)}


def test_classifies_active_client_reason() -> None:
    policy = primary_amo_loss_reason_policy("Действующий клиент")

    assert policy is not None
    assert policy.category == "active_client_entity_resolution"
    assert policy.risk_type == "active_client_loss_reason_requires_entity_resolution"


def test_classifies_duplicate_reasons() -> None:
    assert "duplicate_entity_resolution" in _categories("Дубль")
    assert "duplicate_entity_resolution" in _categories("Недозвон | Дубль (объединены карточки)")


def test_classifies_closed_lost_reasons() -> None:
    assert "lost_or_not_actual" in _categories("Дорого")
    assert "lost_or_not_actual" in _categories("Не актуально")
    assert "lost_or_not_actual" in _categories("Ушел к конкурентам | Не актуально")
    assert "lost_or_not_actual" in _categories("Не актуально | Выбрали репетитора")


def test_classifies_no_contact_archive_reasons() -> None:
    assert "no_contact_archive" in _categories("Архив (нет связи)")
    assert "no_contact_archive" in _categories("Недозвон")


def test_classifies_no_application_and_out_of_scope_reasons() -> None:
    assert "no_application_wrong_direction" in _categories("Не оставлял заявку")
    assert "not_qualified_or_out_of_scope" in _categories("Не квал | Не актуально")
    assert "not_qualified_or_out_of_scope" in _categories("Жуковский")
    assert "not_qualified_or_out_of_scope" in _categories("Не подходит формат")
    assert "not_qualified_or_out_of_scope" in _categories("ШД Жако")


def test_classifies_full_amo_reason_catalog_additions() -> None:
    assert "invalid_or_test_no_action" in _categories("Спам")
    assert "invalid_or_test_no_action" in _categories("Тест")
    assert "future_prospect_reactivation" in _categories("Перспектива (не подошло расписание)")
    assert "future_prospect_reactivation" in _categories("Перспектива")
    assert "company_side_unavailable" in _categories("Закрыли группу (мы)")
    assert "refund_or_postsale_service_review" in _categories("Возврат")
    assert "graduate_or_alumni" in _categories("Выпускник")


def test_classifies_ambiguous_other_reason() -> None:
    assert "ambiguous_other_manual_review" in _categories("Другое")
