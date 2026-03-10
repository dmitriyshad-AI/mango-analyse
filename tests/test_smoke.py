from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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
                status, stage, attempts = conn.execute(
                    "SELECT transcription_status, dead_letter_stage, transcribe_attempts "
                    "FROM call_records LIMIT 1"
                ).fetchone()

            self.assertEqual(status, "dead")
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
