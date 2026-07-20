from __future__ import annotations

import json
from pathlib import Path

from mango_mvp.replay_exam.pseudonymizer import kb_contact_allowlist, pii_findings, pii_signals
from mango_mvp.replay_exam.pii_scan import scan_paths


def test_pii_scanner_catches_bare_phone_cus_username_and_long_ids() -> None:
    findings = pii_findings(
        {
            "text": "Пишите 79001234567, 79001234567@c.us или @realmanager.",
            "payload": {"chatId": "12345678901234567890", "from": "79001234567@c.us"},
            "safe": {"from_me": True, "ts_masked": "masked_1234567890s"},
        }
    )

    kinds = {finding["kind"] for finding in findings}
    assert {"phone", "username", "raw_id", "raw_id_key"}.issubset(kinds)
    assert "from_me" not in {finding["path"] for finding in findings}
    assert not any(finding["path"].endswith("ts_masked") for finding in findings)


def test_pii_scanner_catches_international_phone_and_contextual_birth_date() -> None:
    findings = pii_findings(
        {"text": "6 класс, +1 202 555 0100, дата рождения 01.02.2014; телефон ученика (ОАЭ): 971500000000"}
    )

    assert {finding["kind"] for finding in findings} == {"phone", "date_of_birth"}


def test_pii_scanner_allows_kb_public_contacts(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "facts": [
                    {
                        "fact_id": "contacts",
                        "client_safe_text": (
                            "Контакты школы: +7 900 123-45-67, email info@example.ru, "
                            "telegram @school_help, сайт https://school.example/contacts"
                        ),
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    allowlist = kb_contact_allowlist(snapshot)
    findings = pii_findings(
        "Контакты: +7 900 123-45-67, info@example.ru, @school_help, https://school.example/contacts",
        allowlist=allowlist,
    )

    assert findings == []
    assert {finding["kind"] for finding in pii_findings("https://client.example/private", allowlist=allowlist)} == {"url"}


def test_pseudonymized_ids_are_not_pii_signals() -> None:
    assert pii_signals({"dialog_id": "[dialog_id:id_aaaaaaaaaaaa]", "profile_id": "[profile_id:id_bbbbbbbbbbbb]"}) == []


def test_hash_values_are_not_raw_id_findings_but_plain_ids_are() -> None:
    digest_with_phone_shape = "abcde79001234567" + "a" * 48
    assert pii_findings({"draft_before_hash": digest_with_phone_shape, "sha256": "b" * 64}) == []
    assert {finding["kind"] for finding in pii_findings({"draft_before_hash": "phone 79001234567"})} == {"phone"}
    assert {finding["kind"] for finding in pii_findings({"chat_id": "12345678901234567890"})} == {"raw_id_key", "raw_id"}


def test_scan_paths_reads_jsonl_and_reports_source(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"prefix_messages": [{"from_me": False, "text": "Мой номер 79001234567", "ts_masked": "masked_000000s"}]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    findings = scan_paths([path])

    assert findings
    assert findings[0]["source"].endswith("cases.jsonl:1")
