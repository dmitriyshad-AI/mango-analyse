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


def test_pseudonymizer_uses_stable_fake_name_per_dialog() -> None:
    pseudonymizer = ReplayPseudonymizer(dialog_salt="same")
    one = pseudonymizer.text("Анна Петрова написала")
    two = pseudonymizer.text("Анна Петрова ответила")
    assert one.split()[0] == two.split()[0]
