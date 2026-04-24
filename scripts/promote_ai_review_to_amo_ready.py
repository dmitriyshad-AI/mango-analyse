from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required for this script") from exc


AMO_EXPORT_HEADERS = [
    "Телефон клиента",
    "ID Tallanto",
    "Статус матчинга Tallanto",
    "ФИО родителя",
    "ФИО ребенка",
    "Email",
    "Ответственный Tallanto",
    "Тип ученика Tallanto",
    "Филиал Tallanto",
    "Дата последнего свежего звонка",
    "Менеджер последнего свежего звонка",
    "Краткое резюме последнего свежего звонка",
    "Тип последнего свежего звонка",
    "Краткая история общения",
    "Хронология общения (последние 5 касаний)",
    "Продукты интереса",
    "Рекомендуемый продукт",
    "Возражения",
    "Следующий шаг",
    "Рекомендуемая дата следующего контакта",
    "Приоритет лида",
    "Вероятность продажи, %",
    "История общения Tallanto",
    "Готово к записи в AMO",
    "Причина статуса AMO",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote AI-review contacts into AMO-ready export.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--out-root", required=True)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: Iterable[dict[str, str]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def write_xlsx(path: Path, rows: list[dict[str, str]], sheet_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name=sheet_name, index=False)
        worksheet = writer.sheets[sheet_name]
        if hasattr(worksheet, "set_column"):
            for idx, column in enumerate(frame.columns):
                values = [str(column)] + [str(value) for value in frame[column].fillna("")]
                width = min(max(len(v) for v in values) + 2, 80)
                worksheet.set_column(idx, idx, width)


def build_amo_row(contact: dict[str, str]) -> dict[str, str]:
    return {
        "Телефон клиента": contact.get("Телефон клиента", ""),
        "ID Tallanto": contact.get("ID Tallanto", ""),
        "Статус матчинга Tallanto": contact.get("Статус матчинга Tallanto", ""),
        "ФИО родителя": contact.get("ФИО родителя", ""),
        "ФИО ребенка": contact.get("ФИО ребенка", ""),
        "Email": contact.get("Email", ""),
        "Ответственный Tallanto": contact.get("Ответственный Tallanto", ""),
        "Тип ученика Tallanto": contact.get("Тип ученика Tallanto", ""),
        "Филиал Tallanto": contact.get("Филиал Tallanto", ""),
        "Дата последнего свежего звонка": contact.get("Последний свежий звонок", ""),
        "Менеджер последнего свежего звонка": contact.get("Менеджер последнего свежего звонка", ""),
        "Краткое резюме последнего свежего звонка": contact.get("Краткое резюме последнего свежего звонка", ""),
        "Тип последнего свежего звонка": contact.get("Тип последнего свежего звонка", ""),
        "Краткая история общения": contact.get("Краткая история общения", ""),
        "Хронология общения (последние 5 касаний)": contact.get("Хронология общения (последние 5 касаний)", ""),
        "Продукты интереса": contact.get("Продукты интереса", ""),
        "Рекомендуемый продукт": contact.get("Рекомендуемый продукт", ""),
        "Возражения": contact.get("Возражения", ""),
        "Следующий шаг": contact.get("Следующий шаг", ""),
        "Рекомендуемая дата следующего контакта": contact.get("Рекомендуемая дата следующего контакта", ""),
        "Приоритет лида": contact.get("Приоритет лида", ""),
        "Вероятность продажи, %": contact.get("Вероятность продажи, %", ""),
        "История общения Tallanto": "",
        "Готово к записи в AMO": contact.get("Готово к записи в AMO", ""),
        "Причина статуса AMO": contact.get("Причина статуса AMO", ""),
    }


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    review_root = out_root / "review_queues"
    out_root.mkdir(parents=True, exist_ok=True)
    review_root.mkdir(parents=True, exist_ok=True)

    source_contacts_csv = source_root / "master_contacts_ru.csv"
    source_calls_csv = source_root / "master_calls_ru.csv"
    source_calls_xlsx = source_root / "master_calls_ru.xlsx"
    source_tallanto_review_csv = source_root / "tallanto_review_candidates_ru.csv"
    source_summary = source_root / "summary.json"

    contacts = read_csv_rows(source_contacts_csv)
    contact_headers = list(contacts[0].keys()) if contacts else []

    promoted_count = 0
    for row in contacts:
        if (row.get("Нужна ручная проверка") or "").strip() == "Да":
            row["Нужна ручная проверка"] = "Нет"
            row["Готово к записи в AMO"] = "Да"
            row["Причина статуса AMO"] = "готово к записи в AMO после снятия AI-review"
            promoted_count += 1

    amo_rows = [build_amo_row(row) for row in contacts if (row.get("Готово к записи в AMO") or "").strip() == "Да"]
    review_rows = [row for row in contacts if (row.get("Нужна ручная проверка") or "").strip() == "Да"]
    tallanto_issue_rows = [
        row for row in contacts if (row.get("Статус матчинга Tallanto") or "").strip() != "exact_phone_single"
    ]
    history_not_finished_rows = [
        row for row in contacts if (row.get("Полная история проанализирована") or "").strip() != "Да"
    ]

    write_csv(out_root / "master_contacts_ru.csv", contacts, contact_headers)
    write_xlsx(out_root / "master_contacts_ru.xlsx", contacts, "Contacts")

    write_csv(out_root / "amo_export_ready_ru.csv", amo_rows, AMO_EXPORT_HEADERS)
    write_xlsx(out_root / "amo_export_ready_ru.xlsx", amo_rows, "AMO_Ready")

    write_csv(review_root / "ai_review_contacts_current.csv", review_rows, contact_headers)
    write_xlsx(review_root / "ai_review_contacts_current.xlsx", review_rows, "AI_Review")

    write_csv(review_root / "tallanto_match_issues_current.csv", tallanto_issue_rows, contact_headers)
    write_xlsx(review_root / "tallanto_match_issues_current.xlsx", tallanto_issue_rows, "Tallanto_Issues")

    write_csv(review_root / "history_not_finished_contacts_current.csv", history_not_finished_rows, contact_headers)
    write_xlsx(review_root / "history_not_finished_contacts_current.xlsx", history_not_finished_rows, "History_Open")

    shutil.copy2(source_calls_csv, out_root / "master_calls_ru.csv")
    if source_calls_xlsx.exists():
        shutil.copy2(source_calls_xlsx, out_root / "master_calls_ru.xlsx")
    else:
        write_xlsx(out_root / "master_calls_ru.xlsx", read_csv_rows(source_calls_csv), "Calls")
    shutil.copy2(source_tallanto_review_csv, out_root / "tallanto_review_candidates_ru.csv")

    calls_df = pd.read_csv(source_calls_csv, dtype=str).fillna("")
    contacts_df = pd.DataFrame(contacts).fillna("")
    amo_df = pd.DataFrame(amo_rows).fillna("")
    tallanto_df = pd.DataFrame(tallanto_issue_rows).fillna("")
    history_df = pd.DataFrame(history_not_finished_rows).fillna("")
    review_df = pd.DataFrame(review_rows).fillna("")

    workbook_path = out_root / "master_export_pack_ru.xlsx"
    with pd.ExcelWriter(workbook_path) as writer:
        for sheet_name, frame in [
            ("Contacts", contacts_df),
            ("Calls", calls_df),
            ("AMO_Ready", amo_df),
            ("Tallanto_Issues", tallanto_df),
            ("History_Open", history_df),
            ("AI_Review", review_df),
        ]:
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            if hasattr(worksheet, "set_column"):
                for idx, column in enumerate(frame.columns):
                    values = [str(column)] + [str(value) for value in frame[column].astype(str).fillna("")]
                    width = min(max(len(v) for v in values) + 2, 80)
                    worksheet.set_column(idx, idx, width)

    with source_summary.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    summary["source_root"] = str(source_root)
    summary["master_contacts_rows"] = len(contacts)
    summary["amo_export_ready_rows"] = len(amo_rows)
    summary["amo_ready_count"] = len(amo_rows)
    summary["promoted_from_ai_review"] = promoted_count
    summary["remaining_ai_review_rows"] = len(review_rows)
    summary["remaining_tallanto_issue_rows"] = len(tallanto_issue_rows)
    summary["remaining_history_not_finished_rows"] = len(history_not_finished_rows)
    summary["output_files"] = {
        "master_calls_csv": str(out_root / "master_calls_ru.csv"),
        "master_contacts_csv": str(out_root / "master_contacts_ru.csv"),
        "amo_export_ready_csv": str(out_root / "amo_export_ready_ru.csv"),
        "tallanto_review_csv": str(out_root / "tallanto_review_candidates_ru.csv"),
        "workbook_xlsx": str(workbook_path),
    }
    with (out_root / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "out_root": str(out_root),
                "master_contacts_rows": len(contacts),
                "promoted_from_ai_review": promoted_count,
                "amo_export_ready_rows": len(amo_rows),
                "remaining_ai_review_rows": len(review_rows),
                "remaining_tallanto_issue_rows": len(tallanto_issue_rows),
                "remaining_history_not_finished_rows": len(history_not_finished_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
