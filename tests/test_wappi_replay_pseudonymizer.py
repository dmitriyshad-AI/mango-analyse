from __future__ import annotations

from mango_mvp.replay_exam.pseudonymizer import ReplayPseudonymizer, pii_signals


def test_pseudonymizer_masks_contacts_and_names_recursively() -> None:
    pseudonymizer = ReplayPseudonymizer(dialog_salt="d1")
    payload = {
        "body": "Мария Иванова, телефон +7 999 123-45-67, email test@example.com, telegram @real_user, договор ABC-42, сайт https://example.com",
        "nested": ["Мария Иванова просит документы"],
    }
    scrubbed = pseudonymizer.object(payload)

    text = repr(scrubbed)
    assert "+7 999" not in text
    assert "test@example.com" not in text
    assert "@real_user" not in text
    assert "https://example.com" not in text
    assert "ABC-42" not in text
    assert "Мария Иванова" not in text
    assert "[phone]" in text
    assert "[email]" in text
    assert "[username]" in text
    assert pii_signals(scrubbed) == []


def test_pseudonymizer_masks_international_phones_and_birth_dates() -> None:
    raw = (
        "Анна Иванова, 6 класс, +971 50 000-00-00, дата 01.02.2014; "
        "резерв 00 44 20 0000 0000; телефон ученика (ОАЭ): 971500000000."
    )

    assert pii_signals(raw) == ["date_of_birth", "phone"]
    scrubbed = ReplayPseudonymizer(dialog_salt="international").text(raw)

    assert scrubbed.count("[phone]") == 3
    assert "[date_of_birth]" in scrubbed
    assert pii_signals(scrubbed) == []


def test_phone_mask_preserves_non_phone_numbers_and_schedule() -> None:
    raw = "М9, М11, ЕГЭ 2026, цены 47 250 и 94 500, 2025/26, 12:15-14:15, занятие 01.09.2026 для 6 класса."

    assert ReplayPseudonymizer(dialog_salt="numbers").text(raw) == raw
    assert pii_signals(raw) == []


def test_labeled_phone_mask_preserves_date_like_number() -> None:
    raw = "Телефон будет известен 2026-07-19."

    assert ReplayPseudonymizer(dialog_salt="date").text(raw) == raw
    assert pii_signals(raw) == []


def test_pseudonymizer_masks_mixed_case_child_name_from_pilot() -> None:
    pseudonymizer = ReplayPseudonymizer(dialog_salt="pilot")

    scrubbed = pseudonymizer.text("Записала Сашу кибирева в лагерь.")

    assert "Сашу кибирева" not in scrubbed
    assert pii_signals(scrubbed) == []


def test_pseudonymizer_preserves_program_names_from_stop_list() -> None:
    pseudonymizer = ReplayPseudonymizer(dialog_salt="programs")

    scrubbed = pseudonymizer.text('Интересует Летняя очная школа "Ирина Волкова" и ФизМат «Формула Физтеха».')

    assert 'Летняя очная школа "Ирина Волкова"' in scrubbed
    assert "ФизМат «Формула Физтеха»" in scrubbed
    assert pii_signals(scrubbed) == []


def test_pseudonymizer_still_masks_same_words_as_person_name_outside_program_context() -> None:
    pseudonymizer = ReplayPseudonymizer(dialog_salt="person")

    scrubbed = pseudonymizer.text("Ирина Волкова написала про документы.")

    assert "Ирина Волкова" not in scrubbed
    assert pii_signals(scrubbed) == []


def test_pii_signals_do_not_treat_generic_contract_words_as_contract_ids() -> None:
    assert pii_signals("договорные документы и договор на лагерь") == []
    assert pii_signals("договор ABC-42") == ["contract"]


def test_pseudonymizer_uses_stable_fake_name_per_dialog() -> None:
    pseudonymizer = ReplayPseudonymizer(dialog_salt="same")
    one = pseudonymizer.text("Анна Петрова написала")
    two = pseudonymizer.text("Анна Петрова ответила")
    assert one.split()[0] == two.split()[0]


def test_pseudonymizer_replaces_wappi_and_amo_ids_recursively() -> None:
    pseudonymizer = ReplayPseudonymizer(dialog_salt="ids")
    payload = {
        "id": "116000000000000001",
        "profile_id": "ec2eed50-b55f",
        "chat_id": "79001234567",
        "message_id": "msg-real-1",
        "raw": {
            "lead_id": "123456",
            "contact_id": "654321",
            "reply_message": {"file_name": "6d72888888888888.pdf"},
            "events": [{"talk_id": "talk-77", "thread_id": "thread-88"}],
        },
    }

    scrubbed = pseudonymizer.object(payload)
    text = repr(scrubbed)

    assert "116000000000000001" not in text
    assert "ec2eed50-b55f" not in text
    assert "79001234567" not in text
    assert "msg-real-1" not in text
    assert "123456" not in text
    assert "654321" not in text
    assert "6d72888888888888.pdf" not in text
    assert pii_signals(scrubbed) == []
