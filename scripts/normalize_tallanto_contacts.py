from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
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


def normalize_contacts(df: pd.DataFrame, *, snapshot_at: str) -> list[dict[str, Any]]:
    subject_cols = [col for col in df.columns if str(col).startswith("Предмет №")]
    normalized_rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        phone_parent = _normalize_phone(row.get("Тел. (родителя)"))
        phone_extra = _normalize_phone(row.get("Тел. (доп.)"))
        email = _normalize_email(row.get("E-mail"))
        alt_email = _normalize_email(row.get("Другой E-mail"))
        subjects = [_clean_text(row.get(col)) for col in subject_cols]
        display_name = _join_nonempty(
            [_clean_text(row.get("Имя")), _clean_text(row.get("Фамилия"))],
            sep=" ",
        )
        primary_phone = phone_parent or phone_extra
        primary_email = email or alt_email
        normalized_rows.append(
            {
                "tallanto_id": _clean_text(row.get("ID")),
                "display_name": display_name,
                "first_name": _clean_text(row.get("Имя")),
                "last_name": _clean_text(row.get("Фамилия")),
                "parent_fio": _clean_text(row.get("ФИО родителя")),
                "primary_phone": primary_phone,
                "phone_extra": phone_extra if phone_extra != primary_phone else None,
                "primary_email": primary_email,
                "email_extra": alt_email if alt_email != primary_email else None,
                "responsible": _clean_text(row.get("Ответственный(ая)")),
                "student_type": _clean_text(row.get("Тип ученика")),
                "interests": _clean_text(row.get("Интересы")),
                "branch": _clean_text(row.get("Филиал")),
                "subjects": _join_nonempty(subjects, sep=", "),
                "source": _clean_text(row.get("Источник")),
                "created_at": _clean_text(row.get("Дата создания")),
                "updated_at": _clean_text(row.get("Дата изменения")),
                "snapshot_at": snapshot_at,
                "match_class": "strong_unique" if primary_phone or primary_email else "unmatched",
            }
        )
    return _dedupe_contacts(normalized_rows)


def _dedupe_contacts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    without_id: list[dict[str, Any]] = []
    for row in rows:
        tallanto_id = row.get("tallanto_id")
        if not tallanto_id:
            without_id.append(row)
            continue
        grouped.setdefault(str(tallanto_id), []).append(row)

    result: list[dict[str, Any]] = []
    for tallanto_id in sorted(grouped):
        candidates = grouped[tallanto_id]
        selected = dict(
            max(
                candidates,
                key=lambda row: (
                    _timestamp_sort_key(row.get("updated_at")),
                    _timestamp_sort_key(row.get("created_at")),
                    json.dumps(row, ensure_ascii=False, sort_keys=True),
                ),
            )
        )
        emails = sorted(
            {
                str(value)
                for row in candidates
                for value in (row.get("primary_email"), row.get("email_extra"))
                if value
            }
        )
        primary_email = selected.get("primary_email")
        selected["email_extra"] = " | ".join(email for email in emails if email != primary_email) or None
        phones = sorted(
            {
                str(value)
                for row in candidates
                for value in (row.get("primary_phone"), row.get("phone_extra"))
                if value
            }
        )
        primary_phone = selected.get("primary_phone")
        selected["phone_extra"] = " | ".join(phone for phone in phones if phone != primary_phone) or None
        result.append(selected)
    result.extend(without_id)
    return result


def _timestamp_sort_key(value: Any) -> int:
    text = str(value or "").strip()
    parsed = pd.to_datetime(
        text,
        utc=True,
        errors="coerce",
        dayfirst=bool(re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{4}(?:\s+.*)?", text)),
    )
    return int(parsed.value) if not pd.isna(parsed) else -1


def _guard_staging_out_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    parts = resolved.parts
    if not any(parts[index : index + 2] == (".codex_local", "staging") for index in range(len(parts) - 1)):
        raise ValueError("Tallanto normalized output must stay under .codex_local/staging")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Tallanto Contacts.xls into a clean CSV snapshot.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--snapshot-at", required=True, help="ISO timestamp of the Tallanto export.")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_root = _guard_staging_out_root(Path(args.out_root))
    out_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    out_root.chmod(0o700)

    snapshot_at = datetime.fromisoformat(args.snapshot_at.replace("Z", "+00:00")).isoformat()
    engine = "calamine" if input_path.suffix.casefold() == ".xls" else None
    df = pd.read_excel(input_path, engine=engine)
    normalized_rows = normalize_contacts(df, snapshot_at=snapshot_at)

    out_csv = out_root / "tallanto_contacts_normalized.csv"
    pd.DataFrame(normalized_rows).to_csv(out_csv, index=False, encoding="utf-8")
    out_csv.chmod(0o600)
    out_jsonl = out_root / "tallanto_contacts_normalized.jsonl"
    with out_jsonl.open("w", encoding="utf-8", newline="") as handle:
        for row in normalized_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    out_jsonl.chmod(0o600)

    summary = {
        "input": str(input_path),
        "source_rows": len(df),
        "rows": len(normalized_rows),
        "duplicate_rows_removed": len(df) - len(normalized_rows),
        "output_csv": str(out_csv),
        "output_jsonl": str(out_jsonl),
        "snapshot_at": snapshot_at,
        "rows_with_phone": sum(1 for row in normalized_rows if row["primary_phone"]),
        "rows_with_email": sum(1 for row in normalized_rows if row["primary_email"]),
        "rows_with_parent_fio": sum(1 for row in normalized_rows if row["parent_fio"]),
        "rows_without_identity": sum(1 for row in normalized_rows if row["match_class"] == "unmatched"),
        "rows_with_subjects": sum(1 for row in normalized_rows if row["subjects"]),
        "excluded_columns": ["История общения", "Карточка учащегося", "Баланс", "Пополнено на сумму", "Потраченные деньги"],
    }
    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.chmod(0o600)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
