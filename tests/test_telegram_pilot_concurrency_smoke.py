import json

import pytest

from scripts import run_telegram_pilot_concurrency_smoke as smoke


def test_fake_concurrency_smoke_writes_store_and_summary(tmp_path):
    out_dir = tmp_path / "out"
    rc = smoke.main(
        [
            "--out-dir",
            str(out_dir),
            "--snapshot",
            str(tmp_path / "missing_snapshot.json"),
            "--requests",
            "8",
            "--concurrency",
            "4",
            "--mode",
            "fake",
        ]
    )

    assert rc == 0
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["totals"]["requests"] == 8
    assert summary["totals"]["completed"] == 8
    assert summary["totals"]["errors"] == 0
    assert summary["totals"]["concurrency"] == 4
    assert summary["safety"]["telegram_messages_sent"] is False
    assert (out_dir / "telegram_pilot_concurrency_smoke.sqlite").exists()
    assert len((out_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines()) == 8


def test_concurrency_smoke_refuses_stable_runtime_output(tmp_path):
    with pytest.raises(ValueError, match="stable_runtime"):
        smoke.main(["--out-dir", str(tmp_path / "stable_runtime" / "load")])
