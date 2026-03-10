from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from mango_mvp.db import Base
from mango_mvp.models import CallRecord
from mango_mvp.services.ingest import ingest_from_directory, parse_filename_metadata


class IngestFilenameParseTest(unittest.TestCase):
    def test_parse_filename_manager_then_phone(self) -> None:
        meta = parse_filename_metadata(
            "2026-02-24__10-26-25__Тютюнник Александр__79037311027_1181.mp3"
        )
        self.assertEqual(meta["phone"], "+79037311027")
        self.assertEqual(meta["manager_name"], "Тютюнник Александр")
        self.assertEqual(meta["source_call_id"], "1181")
        self.assertIsNotNone(meta["started_at"])

    def test_parse_filename_phone_then_manager(self) -> None:
        meta = parse_filename_metadata(
            "2026-03-04__13-34-08__79854106566__Леонов Алексей_182.mp3"
        )
        self.assertEqual(meta["phone"], "+79854106566")
        self.assertEqual(meta["manager_name"], "Леонов Алексей")
        self.assertEqual(meta["source_call_id"], "182")
        self.assertIsNotNone(meta["started_at"])

    def test_parse_filename_internal_call_without_phone(self) -> None:
        meta = parse_filename_metadata(
            "2026-02-18__10-13-57__Тютюнник Александр__Коршунова Анастасия_1702.mp3"
        )
        self.assertIsNone(meta["phone"])
        self.assertEqual(meta["manager_name"], "Тютюнник Александр")
        self.assertEqual(meta["source_call_id"], "1702")

    def test_ingest_uses_filename_metadata_fallback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_ingest_filename_") as td:
            root = Path(td)
            audio = (
                root
                / "2026-02-24__10-26-25__Тютюнник Александр__79037311027_1181.mp3"
            )
            audio.write_bytes(b"")

            engine = create_engine("sqlite:///:memory:", future=True)
            Base.metadata.create_all(bind=engine)
            with Session(engine, future=True) as session:
                result = ingest_from_directory(session, root)
                self.assertEqual(result["inserted"], 1)
                row = session.scalars(select(CallRecord)).first()
                self.assertIsNotNone(row)
                assert row is not None
                self.assertEqual(row.phone, "+79037311027")
                self.assertEqual(row.manager_name, "Тютюнник Александр")
                self.assertEqual(row.source_call_id, "1181")
                self.assertIsNotNone(row.started_at)


if __name__ == "__main__":
    unittest.main()
