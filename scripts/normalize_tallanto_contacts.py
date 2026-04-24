from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _normalize_phone(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    digits = re.sub(r"\D+", "", text)
    if not digits:
        return None
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    if len(digits) == 10:
        digits = "7" + digits
    if len(digits) == 11 and digits.startswith("7"):
        return f"+{digits}"
    return None


def _normalize_email(value: Any) -> str | None:
    text = _clean_text(value)
    if not text:
        return None
    return text.lower()


def _join_nonempty(parts: list[str | None], sep: str = " | ") -> str | None:
    values = [part for part in parts if part]
    return sep.join(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Tallanto Contacts.xls into a clean CSV snapshot.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path, engine="calamine")
    normalized_rows: list[dict[str, Any]] = []

    subject_cols = [col for col in df.columns if str(col).startswith("Предмет №")]

    for _, row in df.iterrows():
        phone_parent = _normalize_phone(row.get("Тел. (родителя)"))
        phone_extra = _normalize_phone(row.get("Тел. (доп.)"))
        email = _normalize_email(row.get("E-mail"))
        alt_email = _normalize_email(row.get("Другой E-mail"))

        subjects = [_clean_text(row.get(col)) for col in subject_cols]
        full_name = _join_nonempty([_clean_text(row.get("Имя")), _clean_text(row.get("Фамилия"))], sep=" ")

        normalized_rows.append(
            {
                "tallanto_id": _clean_text(row.get("ID")),
                "first_name": _clean_text(row.get("Имя")),
                "last_name": _clean_text(row.get("Фамилия")),
                "contact_full_name": full_name,
                "parent_fio": _clean_text(row.get("ФИО родителя")),
                "phone_parent": phone_parent,
                "phone_extra": phone_extra,
                "phones_joined": _join_nonempty([phone_parent, phone_extra]),
                "email": email,
                "alt_email": alt_email,
                "any_email": email or alt_email,
                "responsible": _clean_text(row.get("Ответственный(ая)")),
                "student_type": _clean_text(row.get("Тип ученика")),
                "interests_raw": _clean_text(row.get("Интересы")),
                "history_raw": _clean_text(row.get("История общения")),
                "branch": _clean_text(row.get("Филиал")),
                "barcode_text": _clean_text(row.get("Текстовое значение штрихкода")),
                "source": _clean_text(row.get("Источник")),
                "subjects_joined": _join_nonempty(subjects, sep=", "),
                "created_at": _clean_text(row.get("Дата создания")),
                "updated_at": _clean_text(row.get("Дата изменения")),
            }
        )

    out_csv = out_root / "tallanto_contacts_normalized.csv"
    pd.DataFrame(normalized_rows).to_csv(out_csv, index=False, encoding="utf-8")

    summary = {
        "input": str(input_path),
        "rows": len(normalized_rows),
        "output_csv": str(out_csv),
        "rows_with_parent_phone": sum(1 for row in normalized_rows if row["phone_parent"]),
        "rows_with_extra_phone": sum(1 for row in normalized_rows if row["phone_extra"]),
        "rows_with_any_phone": sum(1 for row in normalized_rows if row["phone_parent"] or row["phone_extra"]),
        "rows_with_email": sum(1 for row in normalized_rows if row["email"]),
        "rows_with_alt_email": sum(1 for row in normalized_rows if row["alt_email"]),
        "rows_with_parent_fio": sum(1 for row in normalized_rows if row["parent_fio"]),
        "rows_with_history": sum(1 for row in normalized_rows if row["history_raw"]),
        "rows_with_subjects": sum(1 for row in normalized_rows if row["subjects_joined"]),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
