from __future__ import annotations

from mango_mvp.replay_exam.pseudonymizer import ReplayPseudonymizer, pii_signals


def test_pseudonymizer_masks_contacts_and_names_recursively() -> None:
    pseudonymizer = ReplayPseudonymizer(dialog_salt="d1")
    payload = {
        "body": "Мария Иванова, телефон +7 999 123-45-67, email test@example.com, договор ABC-42, сайт https://example.com",
        "nested": ["Мария Иванова просит документы"],
    }
    scrubbed = pseudonymizer.object(payload)

    text = repr(scrubbed)
    assert "+7 999" not in text
    assert "test@example.com" not in text
    assert "https://example.com" not in text
    assert "ABC-42" not in text
    assert "Мария Иванова" not in text
    assert "[phone]" in text
    assert "[email]" in text
    assert pii_signals(scrubbed) == []


def test_pseudonymizer_masks_mixed_case_child_name_from_pilot() -> None:
    pseudonymizer = ReplayPseudonymizer(dialog_salt="pilot")

    scrubbed = pseudonymizer.text("Записала Сашу кибирева в лагерь.")

    assert "Сашу кибирева" not in scrubbed
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
        "profile_id": "ec2eed50-b55f",
        "chat_id": "79001234567",
        "message_id": "msg-real-1",
        "raw": {
            "lead_id": "123456",
            "contact_id": "654321",
            "events": [{"talk_id": "talk-77", "thread_id": "thread-88"}],
        },
    }

    scrubbed = pseudonymizer.object(payload)
    text = repr(scrubbed)

    assert "ec2eed50-b55f" not in text
    assert "79001234567" not in text
    assert "msg-real-1" not in text
    assert "123456" not in text
    assert "654321" not in text
    assert pii_signals(scrubbed) == []
