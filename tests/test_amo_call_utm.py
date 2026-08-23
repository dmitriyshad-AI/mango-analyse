from __future__ import annotations

import pytest

from mango_mvp.integrations.amo_call_utm import AmoCallUtmError, AmoCallUtmResolver


def field(code, *values):
    return {
        "field_code": code,
        "values": [{"value": value} for value in values],
    }


def contact(contact_id, phone, *lead_ids, extra_fields=()):
    return {
        "id": contact_id,
        "custom_fields_values": [field("PHONE", phone), *extra_fields],
        "_embedded": {"leads": [{"id": lead_id} for lead_id in lead_ids]},
    }


def lead(lead_id, created_at, contact_id, *custom_fields, deleted=False):
    return {
        "id": lead_id,
        "created_at": created_at,
        "is_deleted": deleted,
        "custom_fields_values": list(custom_fields),
        "_embedded": {"contacts": [{"id": contact_id}]},
    }


class FakeAmo:
    def __init__(self, *, contacts=(), leads=None, notes=None, contact_pages=None):
        self.contacts = list(contacts)
        self.leads = leads or {}
        self.notes = notes or {}
        self.contact_pages = contact_pages
        self.calls = []

    def amo_api_get(self, *, path, params=None, limit=50):
        params = dict(params or {})
        self.calls.append((path, params, limit))
        if path == "contacts":
            page = int(params.get("page") or 1)
            items = (
                self.contact_pages[page - 1]
                if self.contact_pages is not None and page <= len(self.contact_pages)
                else self.contacts if page == 1 else []
            )
            has_next = self.contact_pages is not None and page < len(self.contact_pages)
            return {
                "_embedded": {"contacts": items},
                "_links": {"next": {"href": "next"}} if has_next else {},
            }
        if path.startswith("contacts/"):
            return self.contacts[0]
        if path == "leads":
            ids = {int(item) for item in params.get("filter[id][]", ())}
            return {
                "_embedded": {
                    "leads": [self.leads[lead_id] for lead_id in ids if lead_id in self.leads]
                },
                "_links": {},
            }
        if path.endswith("/notes"):
            lead_id = int(path.split("/")[1])
            return {"_embedded": {"notes": self.notes.get(lead_id, [])}, "_links": {}}
        if path.startswith("leads/"):
            return self.leads[int(path.split("/")[1])]
        raise AssertionError(path)


def common_note(text, created_at=1_500):
    return {"note_type": "common", "created_at": created_at, "params": {"text": text}}


def test_full_utm_and_exact_url_are_preserved_without_truncation():
    phone = "+79000000000"
    long_term = "слово" * 500
    client = FakeAmo(
        contacts=[contact(1, phone, 10)],
        leads={
            10: lead(
                10, 1_000, 1,
                field("UTM_SOURCE", "yandex", "partner"), field("UTM_MEDIUM", "cpc"),
                field("UTM_CAMPAIGN", "{712019462}"),
                field("UTM_CONTENT", "1914362696234242275"),
                field("UTM_TERM", long_term),
            )
        },
        notes={10: [common_note("ФИО: скрыто\nurl: https://kmipt.ru/courses/EGE_10/Informatika_10/")]},
    )

    result = AmoCallUtmResolver(client).resolve(phone, 2_000)

    assert result == (
        "Предполагаемая связь с заявкой: точный телефон; "
        "ближайшая заявка создана за 16 мин до звонка\n"
        "UTM ниже — снимок карточки AMO при первом сопоставлении, а не "
        "исторические данные на момент звонка\n"
        "utm_source: yandex\nutm_source: partner\nutm_medium: cpc\nutm_campaign: {712019462}\n"
        "utm_content: 1914362696234242275\nutm_term: " + long_term +
        "\nurl: https://kmipt.ru/courses/EGE_10/Informatika_10/"
    )
    assert "..." not in result


def test_nearest_lead_is_selected_before_checking_whether_utm_exist():
    phone = "+79000000000"
    client = FakeAmo(
        contacts=[contact(1, phone, 10, 20)],
        leads={
            10: lead(10, 1_000, 1, field("UTM_SOURCE", "old-yandex")),
            20: lead(20, 1_900, 1),
        },
        notes={20: [common_note("url: https://kmipt.ru/new/")]},
    )

    result = AmoCallUtmResolver(client).resolve(phone, 2_000)

    assert "old-yandex" not in result
    assert "UTM отсутствуют в связанной заявке" in result
    assert result.endswith("url: https://kmipt.ru/new/")


def test_future_old_tied_and_multiple_contact_cases_fail_closed():
    phone = "+79000000000"
    future = FakeAmo(
        contacts=[contact(1, phone, 10)],
        leads={10: lead(10, 2_001, 1)},
    )
    assert "до звонка" in AmoCallUtmResolver(future).resolve(phone, 2_000)

    old = FakeAmo(
        contacts=[contact(1, phone, 10)],
        leads={10: lead(10, 1, 1)},
    )
    assert "старше 30 дней" in AmoCallUtmResolver(old).resolve(
        phone, 31 * 24 * 60 * 60
    )

    tied = FakeAmo(
        contacts=[contact(1, phone, 10, 20)],
        leads={10: lead(10, 1_000, 1), 20: lead(20, 1_000, 1)},
    )
    assert "несколько заявок" in AmoCallUtmResolver(tied).resolve(phone, 2_000)

    duplicate = FakeAmo(
        contacts=[contact(1, phone, 10), contact(2, phone, 20)],
    )
    assert "несколькими контактами" in AmoCallUtmResolver(duplicate).resolve(phone, 2_000)


def test_phone_search_is_exact_paginated_and_cached_per_run():
    phone = "+79000000000"
    fuzzy = contact(7, "+79999999999", 70, extra_fields=[field("TELEGRAM", phone)])
    exact = contact(1, phone, 10, 20)
    client = FakeAmo(
        contact_pages=[[fuzzy], [exact]],
        leads={
            10: lead(10, 1_000, 1, field("UTM_SOURCE", "first")),
            20: lead(20, 2_000, 1, field("UTM_SOURCE", "second")),
        },
        notes={10: [], 20: []},
    )
    resolver = AmoCallUtmResolver(client)

    assert "utm_source: first" in resolver.resolve(phone, 1_500)
    assert "utm_source: second" in resolver.resolve(phone, 2_500)

    contact_queries = [call for call in client.calls if call[0] == "contacts"]
    lead_queries = [call for call in client.calls if call[0] == "leads"]
    assert len(contact_queries) == 2
    assert len(lead_queries) == 1
    assert lead_queries[0][1]["filter[id][]"] == ["10", "20"]


def test_repeated_contact_pages_merge_all_linked_leads_before_selection():
    phone = "+79000000000"
    client = FakeAmo(
        contact_pages=[[contact(1, phone, 10)], [contact(1, phone, 20)]],
        leads={
            10: lead(10, 1_000, 1, field("UTM_SOURCE", "old")),
            20: lead(20, 1_900, 1, field("UTM_SOURCE", "nearest")),
        },
        notes={20: []},
    )

    result = AmoCallUtmResolver(client).resolve(phone, 2_000)

    assert "utm_source: nearest" in result
    assert "utm_source: old" not in result


def test_partial_repeated_contact_without_phone_still_contributes_linked_leads():
    phone = "+79000000000"
    partial_same_contact = {"id": 1, "_embedded": {"leads": [{"id": 20}]}}
    client = FakeAmo(
        contact_pages=[[contact(1, phone, 10)], [partial_same_contact]],
        leads={
            10: lead(10, 1_000, 1, field("UTM_SOURCE", "old")),
            20: lead(20, 1_900, 1, field("UTM_SOURCE", "nearest")),
        },
        notes={20: []},
    )

    result = AmoCallUtmResolver(client).resolve(phone, 2_000)

    assert "utm_source: nearest" in result
    assert "utm_source: old" not in result


def test_missing_url_is_not_replaced_with_referrer():
    phone = "+79000000000"
    client = FakeAmo(
        contacts=[contact(1, phone, 10)],
        leads={10: lead(10, 1_000, 1, field("UTM_SOURCE", "yandex"), field("REFERRER", "https://amo.invalid/"))},
        notes={10: [common_note("referrer: https://amo.invalid/")]},
    )

    result = AmoCallUtmResolver(client).resolve(phone, 2_000)

    assert "amo.invalid" not in result
    assert result.endswith("Страница заявки не сохранена в AMO")


def test_deleted_or_not_reverse_linked_lead_cannot_supply_utm():
    phone = "+79000000000"
    client = FakeAmo(
        contacts=[contact(1, phone, 10, 20, 30)],
        leads={
            10: lead(10, 1_900, 2, field("UTM_SOURCE", "other-contact")),
            20: lead(20, 1_800, 1, field("UTM_SOURCE", "deleted"), deleted=True),
            30: lead(30, 1_700, 1, field("UTM_SOURCE", "valid")),
        },
        notes={30: []},
    )

    result = AmoCallUtmResolver(client).resolve(phone, 2_000)

    assert "utm_source: valid" in result
    assert "other-contact" not in result
    assert "deleted" not in result


def test_incomplete_linked_lead_snapshot_fails_closed():
    phone = "+79000000000"

    class PartialLeadAmo(FakeAmo):
        def amo_api_get(self, *, path, params=None, limit=50):
            payload = super().amo_api_get(path=path, params=params, limit=limit)
            if path == "leads":
                payload["_embedded"]["leads"] = payload["_embedded"]["leads"][:1]
            return payload

    client = PartialLeadAmo(
        contacts=[contact(1, phone, 10, 20)],
        leads={
            10: lead(10, 1_000, 1, field("UTM_SOURCE", "old")),
            20: lead(20, 1_900, 1, field("UTM_SOURCE", "new")),
        },
    )

    with pytest.raises(AmoCallUtmError, match="incomplete linked lead snapshot"):
        AmoCallUtmResolver(client).resolve(phone, 2_000)


def test_linked_lead_without_created_at_cannot_make_older_lead_win():
    phone = "+79000000000"
    partial_nearest = lead(20, 1_900, 1, field("UTM_SOURCE", "nearest"))
    partial_nearest.pop("created_at")
    client = FakeAmo(
        contacts=[contact(1, phone, 10, 20)],
        leads={
            10: lead(10, 1_000, 1, field("UTM_SOURCE", "old")),
            20: partial_nearest,
        },
    )

    with pytest.raises(AmoCallUtmError, match="incomplete linked lead snapshot"):
        AmoCallUtmResolver(client).resolve(phone, 2_000)


def test_empty_contact_page_with_next_link_fails_closed():
    class PartialContactPageAmo(FakeAmo):
        def amo_api_get(self, *, path, params=None, limit=50):
            if path == "contacts":
                return {"_embedded": {"contacts": []}, "_links": {"next": {"href": "next"}}}
            return super().amo_api_get(path=path, params=params, limit=limit)

    with pytest.raises(AmoCallUtmError, match="empty contact page with next"):
        AmoCallUtmResolver(PartialContactPageAmo()).resolve("+79000000000", 2_000)


def test_empty_note_page_with_next_link_fails_closed():
    phone = "+79000000000"

    class PartialNotePageAmo(FakeAmo):
        def amo_api_get(self, *, path, params=None, limit=50):
            if path.endswith("/notes"):
                return {"_embedded": {"notes": []}, "_links": {"next": {"href": "next"}}}
            return super().amo_api_get(path=path, params=params, limit=limit)

    client = PartialNotePageAmo(
        contacts=[contact(1, phone, 10)],
        leads={10: lead(10, 1_000, 1, field("UTM_SOURCE", "yandex"))},
    )
    with pytest.raises(AmoCallUtmError, match="empty note page with next"):
        AmoCallUtmResolver(client).resolve(phone, 2_000)


def test_pagination_and_linked_lead_count_have_hard_limits():
    phone = "+79000000000"
    endless = FakeAmo(contact_pages=[[contact(1, "+79999999999")] for _ in range(21)])
    with pytest.raises(AmoCallUtmError, match="page limit"):
        AmoCallUtmResolver(endless).resolve(phone, 2_000)

    excessive = FakeAmo(contacts=[contact(1, phone, *range(1, 502))])
    with pytest.raises(AmoCallUtmError, match="too many linked leads"):
        AmoCallUtmResolver(excessive).resolve(phone, 2_000)


def test_oversized_utm_block_is_rejected_without_silent_truncation():
    phone = "+79000000000"
    client = FakeAmo(
        contacts=[contact(1, phone, 10)],
        leads={10: lead(10, 1_000, 1, field("UTM_TERM", "x" * 50_001))},
        notes={10: []},
    )

    with pytest.raises(AmoCallUtmError, match="Google cell limit"):
        AmoCallUtmResolver(client).resolve(phone, 2_000)


def test_direct_lead_id_wins_and_is_not_rejected_only_for_age():
    phone = "+79000000000"
    client = FakeAmo(
        contacts=[contact(1, phone, 10, 20)],
        leads={
            10: lead(10, 1, 1, field("UTM_SOURCE", "direct")),
            20: lead(20, 1_900, 1, field("UTM_SOURCE", "nearest")),
        },
        notes={10: [common_note("url: https://kmipt.ru/direct/", created_at=1)]},
    )

    result = AmoCallUtmResolver(client).resolve(
        phone, 40 * 24 * 60 * 60, lead_id=10
    )

    assert result.startswith("Связь с заявкой: прямая из записи звонка")
    assert "utm_source: direct" in result
    assert "nearest" not in result


def test_direct_lead_id_without_matching_phone_is_not_trusted():
    phone = "+79000000000"
    client = FakeAmo(
        contacts=[contact(1, "+79999999999", 10)],
        leads={10: lead(10, 1_000, 1, field("UTM_SOURCE", "wrong"))},
    )

    result = AmoCallUtmResolver(client).resolve(phone, 2_000, lead_id=10)

    assert "телефон не подтверждает" in result
    assert "wrong" not in result


def test_direct_lead_without_created_at_is_retryable_integrity_failure():
    phone = "+79000000000"
    partial = lead(10, 1_000, 1, field("UTM_SOURCE", "must-not-cache"))
    partial.pop("created_at")
    client = FakeAmo(
        contacts=[contact(1, phone, 10)],
        leads={10: partial},
    )

    with pytest.raises(AmoCallUtmError, match="incomplete direct lead snapshot"):
        AmoCallUtmResolver(client).resolve(phone, 2_000, lead_id=10)


def test_future_or_ambiguous_landing_url_is_not_silently_attributed():
    phone = "+79000000000"
    future = FakeAmo(
        contacts=[contact(1, phone, 10)],
        leads={10: lead(10, 1_000, 1, field("UTM_SOURCE", "yandex"))},
        notes={10: [common_note("url: https://kmipt.ru/future/", created_at=2_001)]},
    )
    assert AmoCallUtmResolver(future).resolve(phone, 2_000).endswith(
        "Страница заявки не сохранена в AMO"
    )

    ambiguous = FakeAmo(
        contacts=[contact(1, phone, 10)],
        leads={10: lead(10, 1_000, 1)},
        notes={10: [
            common_note("url: https://kmipt.ru/one/", created_at=1_100),
            common_note("url: https://kmipt.ru/two/", created_at=1_200),
        ]},
    )
    with pytest.raises(AmoCallUtmError, match="multiple historical"):
        AmoCallUtmResolver(ambiguous).resolve(phone, 2_000)
