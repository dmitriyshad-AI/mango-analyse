from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mango_mvp.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(args: list[str], env_overrides: dict[str, str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT / "src"))
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-m", "mango_mvp.cli", *args],
        cwd=str(cwd or PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class SmokePipelineTest(unittest.TestCase):
    def test_get_settings_parses_float_env_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "STEREO_OVERLAP_SIMILARITY_THRESHOLD": "0.93",
                "MONO_ROLE_ASSIGNMENT_MIN_CONFIDENCE": "0.71",
                "MONO_ROLE_ASSIGNMENT_LLM_THRESHOLD": "0.77",
                "OLLAMA_TEMPERATURE": "0.15",
            },
            clear=False,
        ):
            get_settings.cache_clear()
            settings = get_settings()
            self.assertAlmostEqual(settings.stereo_overlap_similarity_threshold, 0.93)
            self.assertAlmostEqual(settings.mono_role_assignment_min_confidence, 0.71)
            self.assertAlmostEqual(settings.mono_role_assignment_llm_threshold, 0.77)
            self.assertAlmostEqual(settings.ollama_temperature, 0.15)
            get_settings.cache_clear()

    def test_init_db_creates_missing_sqlite_parent_dir(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_smoke_init_db_") as td:
            root = Path(td)
            db_path = root / "nested" / "db" / "test_pipeline.db"
            self.assertFalse(db_path.parent.exists())

            result = run_cli(["init-db"], {"DATABASE_URL": f"sqlite:///{db_path}"})
            self.assertEqual(
                result.returncode,
                0,
                msg=f"Command failed: init-db\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )
            self.assertTrue(db_path.parent.exists())
            self.assertTrue(db_path.exists())

    def test_ingest_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_smoke_ingest_") as td:
            root = Path(td)
            rec_dir = root / "calls"
            rec_dir.mkdir(parents=True, exist_ok=True)
            (rec_dir / "a.mp3").write_bytes(b"")
            (rec_dir / "b.wav").write_bytes(b"")
            db_path = root / "ingest.db"

            env = {"DATABASE_URL": f"sqlite:///{db_path}"}
            for args in (
                ["init-db"],
                ["ingest", "--recordings-dir", str(rec_dir)],
                ["ingest", "--recordings-dir", str(rec_dir)],
            ):
                result = run_cli(args, env)
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"Command failed: {args}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
                )

            with sqlite3.connect(str(db_path)) as conn:
                total = conn.execute("SELECT COUNT(*) FROM call_records").fetchone()[0]
                distinct_files = conn.execute(
                    "SELECT COUNT(DISTINCT source_file) FROM call_records"
                ).fetchone()[0]
            self.assertEqual(total, 2)
            self.assertEqual(distinct_files, 2)

    def test_retry_and_dead_letter_cycle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="mango_smoke_retry_") as td:
            root = Path(td)
            rec_dir = root / "calls"
            rec_dir.mkdir(parents=True, exist_ok=True)
            (rec_dir / "x.mp3").write_bytes(b"")
            db_path = root / "retry.db"

            env = {
                "DATABASE_URL": f"sqlite:///{db_path}",
                "TRANSCRIBE_PROVIDER": "openai",
                "ANALYZE_PROVIDER": "mock",
                "TRANSCRIBE_MAX_ATTEMPTS": "2",
                "RETRY_BASE_DELAY_SEC": "0",
            }

            for args in (
                ["init-db"],
                ["ingest", "--recordings-dir", str(rec_dir)],
                ["worker", "--once", "--stage-limit", "10"],
            ):
                result = run_cli(args, env)
                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"Command failed: {args}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
                )

            with sqlite3.connect(str(db_path)) as conn:
                conn.execute("UPDATE call_records SET next_retry_at=NULL")
                conn.commit()

            result = run_cli(["worker", "--once", "--stage-limit", "10"], env)
            self.assertEqual(
                result.returncode,
                0,
                msg=f"Command failed: second worker cycle\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )

            with sqlite3.connect(str(db_path)) as conn:
                status, resolve_status, analysis_status, sync_status, stage, attempts = conn.execute(
                    "SELECT transcription_status, resolve_status, analysis_status, sync_status, dead_letter_stage, transcribe_attempts "
                    "FROM call_records LIMIT 1"
                ).fetchone()

            self.assertEqual(status, "dead")
            self.assertEqual(resolve_status, "skipped")
            self.assertEqual(analysis_status, "failed")
            self.assertEqual(sync_status, "failed")
            self.assertEqual(stage, "transcribe")
            self.assertEqual(attempts, 2)

            result = run_cli(["requeue-dead", "--stage", "all", "--limit", "100"], env)
            self.assertEqual(
                result.returncode,
                0,
                msg=f"Command failed: requeue-dead\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )

            with sqlite3.connect(str(db_path)) as conn:
                status, stage, attempts = conn.execute(
                    "SELECT transcription_status, dead_letter_stage, transcribe_attempts "
                    "FROM call_records LIMIT 1"
                ).fetchone()

            self.assertEqual(status, "pending")
            self.assertIsNone(stage)
            self.assertEqual(attempts, 0)

    def test_stable_runtime_rebuild_smoke(self) -> None:
        rebuild = PROJECT_ROOT / "stable_runtime" / "rebuild_snapshot.sh"
        self.assertTrue(rebuild.exists(), "stable_runtime/rebuild_snapshot.sh is missing")
        stable_python = PROJECT_ROOT / "stable_runtime" / "venv_stable" / "bin" / "python"
        stable_runtime_ready = subprocess.run(
            [
                str(stable_python),
                "-c",
                "import sqlalchemy, mango_mvp.cli",
            ],
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
            text=True,
            capture_output=True,
            check=False,
        ) if stable_python.exists() else None
        if stable_runtime_ready is None or stable_runtime_ready.returncode != 0:
            self.skipTest("stable_runtime/venv_stable fixture is not install-complete in this worktree")

        env = os.environ.copy()
        env["MANGO_STABLE_SMOKE_ONLY"] = "1"
        result = subprocess.run(
            [str(rebuild)],
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"Stable rebuild smoke failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
