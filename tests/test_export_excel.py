from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path

from mango_mvp.models import CallRecord
from mango_mvp.services.export_excel import (
    build_call_rows,
    build_contact_rows,
    write_workbook,
)


class ExportExcelTest(unittest.TestCase):
    def test_build_contact_rows_aggregates_by_phone(self) -> None:
        analysis_1 = {
            "analysis_schema_version": "v2",
            "history_summary": "Первый звонок.",
            "structured_fields": {
                "people": {"parent_fio": "Иванова Мария", "child_fio": "Иванов Кирилл"},
                "contacts": {
                    "email": "maria@example.com",
                    "phone_from_filename": "79990000000",
                    "preferred_channel": "telegram",
                },
                "student": {"grade_current": "8", "school": None},
                "interests": {
                    "products": ["годовые курсы"],
                    "format": ["онлайн"],
                    "subjects": ["математика"],
                    "exam_targets": [],
                },
                "commercial": {"price_sensitivity": "medium", "budget": None, "discount_interest": True},
                "objections": ["цена"],
                "next_step": {"action": "Отправить материалы", "due": None},
                "lead_priority": "warm",
            },
            "follow_up_score": 70,
            "follow_up_reason": "Есть согласованный следующий шаг.",
            "tags": [],
            "quality_flags": {
                "mode": "stereo",
                "secondary_provider": "gigaam",
                "call_type": "sales_call",
                "needs_review": False,
                "review_reasons": [],
            },
        }
        analysis_2 = {
            "analysis_schema_version": "v2",
            "history_summary": "Второй звонок.",
            "structured_fields": {
                "people": {"parent_fio": "Иванова Мария", "child_fio": "Иванов Кирилл"},
                "contacts": {
                    "email": "maria@example.com",
                    "phone_from_filename": "79990000000",
                    "preferred_channel": "telegram",
                },
                "student": {"grade_current": "8", "school": None},
                "interests": {
                    "products": ["летний лагерь"],
                    "format": ["оффлайн"],
                    "subjects": ["математика", "физика"],
                    "exam_targets": [],
                },
                "commercial": {"price_sensitivity": "low", "budget": None, "discount_interest": False},
                "objections": [],
                "next_step": {"action": "Перезвонить клиенту", "due": "2026-03-25"},
                "lead_priority": "hot",
            },
            "target_product": "летний лагерь",
            "follow_up_score": 90,
            "follow_up_reason": "Клиент заинтересован, нужно быстро вернуться.",
            "tags": [],
            "quality_flags": {
                "mode": "stereo",
                "secondary_provider": "gigaam",
                "call_type": "sales_call",
                "needs_review": True,
                "review_reasons": ["sales_missing_next_step"],
            },
        }

        calls = [
            CallRecord(
                id=1,
                source_file="/tmp/a.mp3",
                source_filename="a.mp3",
                phone="79990000000",
                manager_name="Менеджер 1",
                duration_sec=120.0,
                started_at=datetime(2026, 3, 20, 10, 0, 0),
                analysis_json=json.dumps(analysis_1, ensure_ascii=False),
            ),
            CallRecord(
                id=2,
                source_file="/tmp/b.mp3",
                source_filename="b.mp3",
                phone="79990000000",
                manager_name="Менеджер 2",
                duration_sec=180.0,
                started_at=datetime(2026, 3, 21, 11, 0, 0),
                analysis_json=json.dumps(analysis_2, ensure_ascii=False),
            ),
        ]

        call_rows = build_call_rows(calls)
        self.assertEqual(len(call_rows), 2)
        self.assertEqual(call_rows[0]["recommended_followup_date"], "2026-03-22")
        self.assertEqual(call_rows[1]["recommended_followup_date"], "2026-03-25")
        self.assertEqual(call_rows[0]["call_type"], "sales_call")
        self.assertEqual(call_rows[1]["needs_review"], True)
        self.assertEqual(call_rows[1]["review_reasons"], "sales_missing_next_step")

        contact_rows = build_contact_rows(call_rows)
        self.assertEqual(len(contact_rows), 1)
        row = contact_rows[0]
        self.assertEqual(row["phone"], "79990000000")
        self.assertEqual(row["calls_count"], 2)
        self.assertEqual(row["latest_manager_name"], "Менеджер 2")
        self.assertEqual(row["recommended_product"], "летний лагерь")
        self.assertEqual(row["lead_priority"], "hot")
        self.assertEqual(row["latest_call_type"], "sales_call")
        self.assertEqual(row["needs_review"], True)
        self.assertEqual(row["review_reasons_latest"], "sales_missing_next_step")
        self.assertIn("годовые курсы", row["interests_products"])
        self.assertIn("летний лагерь", row["interests_products"])

    def test_write_workbook_creates_valid_xlsx(self) -> None:
        calls_rows = [
            {
                "id": 1,
                "started_at": "2026-03-21 11:00:00",
                "phone": "79990000000",
                "manager_name": "Менеджер",
                "duration_sec": 180.0,
                "source_filename": "call.mp3",
                "source_file": "/tmp/call.mp3",
                "history_summary": "Тестовый конспект.",
                "parent_fio": "Иванова Мария",
                "child_fio": "Иванов Кирилл",
                "email": "maria@example.com",
                "preferred_channel": "telegram",
                "grade_current": "8",
                "school": "",
                "interests_products": "годовые курсы",
                "interests_format": "онлайн",
                "interests_subjects": "математика",
                "exam_targets": "",
                "recommended_product": "годовые курсы",
                "price_sensitivity": "medium",
                "budget": "",
                "discount_interest": "True",
                "objections": "цена",
                "next_step_action": "Отправить материалы",
                "next_step_due_raw": "",
                "lead_priority": "warm",
                "sale_probability_pct": 70,
                "sale_probability_reason": "Есть согласованный следующий шаг.",
                "recommended_followup_date": "2026-03-23",
                "recommended_followup_reason": "После отправки материалов оптимален follow-up через 2 дня.",
                "call_type": "sales_call",
                "needs_review": False,
                "review_reasons": "",
                "quality_mode": "stereo",
                "secondary_provider": "gigaam",
                "secondary_backfill_status": "",
                "tags": "",
                "analysis_schema_version": "v2",
            }
        ]
        contacts_rows = [
            {
                "contact_key": "79990000000",
                "phone": "79990000000",
                "calls_count": 1,
                "first_call_at": "2026-03-21 11:00:00",
                "last_call_at": "2026-03-21 11:00:00",
                "latest_manager_name": "Менеджер",
                "latest_history_summary": "Тестовый конспект.",
                "parent_fio": "Иванова Мария",
                "child_fio": "Иванов Кирилл",
                "email": "maria@example.com",
                "preferred_channel": "telegram",
                "grade_current": "8",
                "interests_products": "годовые курсы",
                "interests_format": "онлайн",
                "interests_subjects": "математика",
                "exam_targets": "",
                "recommended_product": "годовые курсы",
                "lead_priority": "warm",
                "sale_probability_pct": 70,
                "sale_probability_reason": "Есть согласованный следующий шаг.",
                "recommended_followup_date": "2026-03-23",
                "recommended_followup_reason": "После отправки материалов оптимален follow-up через 2 дня.",
                "latest_call_type": "sales_call",
                "needs_review": False,
                "review_reasons_latest": "",
                "last_next_step_action": "Отправить материалы",
                "last_next_step_due_raw": "",
                "objections_latest": "цена",
                "source_call_ids": "1",
            }
        ]

        with tempfile.TemporaryDirectory(prefix="mango_xlsx_") as td:
            out_path = write_workbook(
                Path(td) / "sales_workbook.xlsx",
                calls_rows=calls_rows,
                contacts_rows=contacts_rows,
            )
            self.assertTrue(out_path.exists())
            with zipfile.ZipFile(out_path, "r") as zf:
                names = set(zf.namelist())
                self.assertIn("xl/workbook.xml", names)
                self.assertIn("xl/worksheets/sheet1.xml", names)
                self.assertIn("xl/worksheets/sheet2.xml", names)
                sheet1 = zf.read("xl/worksheets/sheet1.xml").decode("utf-8")
                sheet2 = zf.read("xl/worksheets/sheet2.xml").decode("utf-8")
                workbook_text = sheet1 + "\n" + sheet2
                if "xl/sharedStrings.xml" in names:
                    shared_strings = zf.read("xl/sharedStrings.xml").decode("utf-8")
                    workbook_text += "\n" + shared_strings
                self.assertIn("Тестовый конспект.", workbook_text)
                self.assertIn("79990000000", workbook_text)
                self.assertIn("<worksheet", sheet1)
                self.assertIn("<worksheet", sheet2)

    def test_build_call_rows_repairs_mojibake_manager_and_filename_for_export(self) -> None:
        analysis = {
            "analysis_schema_version": "v2",
            "history_summary": "26.03.2026 менеджер КЂлз•Ґ† Д†амп обсудил с клиентом программу.",
            "structured_fields": {
                "people": {},
                "contacts": {},
                "student": {},
                "interests": {"products": [], "format": [], "subjects": [], "exam_targets": []},
                "commercial": {},
                "objections": [],
                "next_step": {"action": None, "due": None},
                "lead_priority": "warm",
            },
            "quality_flags": {"call_type": "sales_call"},
            "tags": [],
        }
        calls = [
            CallRecord(
                id=1,
                source_file="/tmp/raw.mp3",
                source_filename="2026-03-09__10-52-01__КЂлз•Ґ† Д†амп__79801983922.mp3",
                phone="79801983922",
                manager_name="КЂлз•Ґ† Д†амп",
                duration_sec=60.0,
                started_at=datetime(2026, 3, 9, 10, 52, 1),
                analysis_json=json.dumps(analysis, ensure_ascii=False),
            ),
        ]
        row = build_call_rows(calls)[0]
        self.assertEqual(row["manager_name"], "Клычева Дарья")
        self.assertIn("Клычева Дарья", row["history_summary"])
        self.assertEqual(
            row["source_filename"],
            "2026-03-09__10-52-01__Клычева Дарья__79801983922.mp3",
        )
