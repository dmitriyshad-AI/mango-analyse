from __future__ import annotations

import tempfile
import unittest
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from mango_mvp.cli import cmd_ingest
from mango_mvp.db import Base
from mango_mvp.models import CallRecord
from mango_mvp.services.ingest import ingest_from_directory, parse_filename_metadata
from mango_mvp.utils.filename_repair import repair_manager_name


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
        self.assertEqual(meta["manager_name"], "Леонов Алексей")
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

    def test_ingest_isolates_one_corrupt_audio_and_keeps_good_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_ingest_isolation_") as td:
            root = Path(td)
            good = root / "2026-07-13__10-00-00__79000000001__Менеджер_1.mp3"
            bad = root / "2026-07-13__10-01-00__79000000002__Менеджер_2.mp3"
            good.write_bytes(b"good")
            bad.write_bytes(b"bad")
            engine = create_engine("sqlite:///:memory:", future=True)
            Base.metadata.create_all(bind=engine)

            def fake_probe(path: Path) -> dict:
                if path.name == bad.name:
                    raise ValueError("corrupt audio")
                return {"codec_name": "mp3", "duration_sec": 1.0}

            with Session(engine, future=True) as session, patch(
                "mango_mvp.services.ingest.probe_audio",
                side_effect=fake_probe,
            ):
                result = ingest_from_directory(session, root)
                rows = list(session.scalars(select(CallRecord)).all())

            self.assertEqual(result["processed"], 2)
            self.assertEqual(result["inserted"], 1)
            self.assertEqual(result["failed"], 1)
            self.assertEqual(result["failure_types"], {"ValueError": 1})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].source_file, str(good.resolve()))

    def test_ingest_does_not_mask_database_failure_as_bad_audio(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_ingest_db_failure_") as td:
            root = Path(td)
            (root / "call.mp3").write_bytes(b"audio")
            engine = create_engine("sqlite:///:memory:", future=True)
            Base.metadata.create_all(bind=engine)
            error = OperationalError("SELECT", {}, RuntimeError("database unavailable"))
            with Session(engine, future=True) as session, patch.object(
                session,
                "scalar",
                side_effect=error,
            ):
                with pytest.raises(OperationalError):
                    ingest_from_directory(session, root)

    def test_ingest_does_not_mask_system_io_failure_as_bad_audio(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_ingest_io_failure_") as td:
            root = Path(td)
            (root / "call.mp3").write_bytes(b"audio")
            engine = create_engine("sqlite:///:memory:", future=True)
            Base.metadata.create_all(bind=engine)
            with Session(engine, future=True) as session, patch(
                "mango_mvp.services.ingest.probe_audio",
                side_effect=OSError("disk unavailable"),
            ):
                with pytest.raises(OSError, match="disk unavailable"):
                    ingest_from_directory(session, root)

    def test_cli_ingest_fails_loud_when_every_file_failed(self) -> None:
        result = {
            "processed": 2,
            "inserted": 0,
            "skipped": 0,
            "failed": 2,
            "failure_types": {"ValueError": 2},
        }
        factory = unittest.mock.MagicMock()
        with patch("mango_mvp.cli.get_settings"), patch(
            "mango_mvp.cli.build_session_factory",
            return_value=factory,
        ), patch("mango_mvp.cli.ingest_from_directory", return_value=result):
            rc = cmd_ingest(SimpleNamespace(recordings_dir=".", metadata_csv=None, limit=None))

        self.assertEqual(rc, 1)

    def test_parse_filename_repairs_mojibake_manager_name(self) -> None:
        meta = parse_filename_metadata(
            "2026-03-09__10-52-01__КЂлз•Ґ† Д†амп__79801983922.mp3"
        )
        self.assertEqual(meta["phone"], "+79801983922")
        self.assertEqual(meta["manager_name"], "Клычева Дарья")

    def test_repair_manager_name_uses_extended_archive_mapping(self) -> None:
        self.assertEqual(repair_manager_name("КЃѓЃв•Ґ† ЕҐ†"), "Копотева Ева")
        self.assertEqual(repair_manager_name("Шђл£Ђ•Ґ† ПЃЂ®≠†"), "Шмыглева Полина")
        self.assertEqual(repair_manager_name("Леонова Анна"), "Тропина Анна")

    def test_repair_manager_name_normalizes_unicode(self) -> None:
        repaired = repair_manager_name("Л•Ѓ≠ЃҐ АЂ•™б•©")
        self.assertEqual(repaired, "Леонов Алексей")
        self.assertEqual(unicodedata.normalize("NFC", repaired or ""), repaired)


if __name__ == "__main__":
    unittest.main()
