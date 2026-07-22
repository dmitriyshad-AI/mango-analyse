from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "normalize_tallanto_contacts.py"
    spec = importlib.util.spec_from_file_location("normalize_tallanto_contacts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalizer_emits_canonical_identity_and_excludes_freeform_history() -> None:
    mod = _load_script()
    frame = pd.DataFrame(
        [
            {
                "ID": "42",
                "Имя": "Иван",
                "Фамилия": "Петров",
                "Тел. (родителя)": "8 (916) 111-22-33",
                "Тел. (доп.)": "",
                "E-mail": " Parent@Example.com ",
                "Другой E-mail": "",
                "ФИО родителя": "Родитель",
                "Тип ученика": "9 класс",
                "Филиал": "Онлайн",
                "Интересы": "Математика",
                "Предмет №1": "Математика",
                "История общения": "секретная свободная история",
                "Карточка учащегося": "внутренняя заметка",
            }
        ]
    )

    rows = mod.normalize_contacts(frame, snapshot_at="2026-07-10T00:00:00+03:00")

    assert rows[0]["primary_phone"] == "+79161112233"
    assert rows[0]["primary_email"] == "parent@example.com"
    assert rows[0]["display_name"] == "Иван Петров"
    assert rows[0]["match_class"] == "strong_unique"
    assert rows[0]["snapshot_at"] == "2026-07-10T00:00:00+03:00"
    assert "history_raw" not in rows[0]
    assert "История общения" not in rows[0]
    assert "Карточка учащегося" not in rows[0]


def test_normalizer_merges_duplicate_tallanto_rows_without_losing_extra_email() -> None:
    mod = _load_script()
    frame = pd.DataFrame(
        [
            {"ID": "student-1", "Имя": "Иван", "E-mail": "main@example.com", "Другой E-mail": "one@example.com"},
            {"ID": "student-1", "Имя": "Иван", "E-mail": "main@example.com", "Другой E-mail": "two@example.com"},
        ]
    )

    rows = mod.normalize_contacts(frame, snapshot_at="2026-07-10T03:07:00+03:00")

    assert len(rows) == 1
    assert rows[0]["tallanto_id"] == "student-1"
    assert set(rows[0]["email_extra"].split(" | ")) == {"one@example.com", "two@example.com"}


def test_normalizer_uses_real_dates_and_keeps_duplicate_phones() -> None:
    mod = _load_script()
    frame = pd.DataFrame(
        [
            {"ID": "student-1", "Имя": "Старое", "Дата изменения": "31.12.2025", "Тел. (родителя)": "+7 916 111-11-11"},
            {"ID": "student-1", "Имя": "Новое", "Дата изменения": "01.01.2026", "Тел. (родителя)": "+7 916 222-22-22"},
        ]
    )

    rows = mod.normalize_contacts(frame, snapshot_at="2026-07-10T03:07:00+03:00")

    assert rows[0]["display_name"] == "Новое"
    assert rows[0]["primary_phone"] == "+79162222222"
    assert rows[0]["phone_extra"] == "+79161111111"


def test_normalizer_rejects_output_outside_staging(tmp_path: Path) -> None:
    mod = _load_script()

    try:
        mod._guard_staging_out_root(tmp_path / "exports")
    except ValueError as exc:
        assert ".codex_local/staging" in str(exc)
    else:
        raise AssertionError("unsafe output root accepted")
