import json
from pathlib import Path

import pytest

from scripts import build_memory_measure_scenarios as scenario_builder
from scripts import run_memory_measure_off_on as off_on


def test_memory_measure_scenario_payload_has_no_raw_pii_and_keeps_resolver_id(tmp_path: Path) -> None:
    candidate = scenario_builder.Candidate(
        customer_id="customer:test-memory",
        calls=2,
        emails=1,
        opportunities=1,
        statuses={"В работе"},
        brand_summaries={"foton": "Бренд: Фотон. Стадия: В работе. Интерес: онлайн. Следующий шаг: уточнить расписание."},
        amo_lead_id="5001",
        amo_contact_id="7001",
        phone_ref="sha256:abcdef1234567890",
    )

    rows, report = scenario_builder.build_scenario_rows(
        {"customer:test-memory": candidate},
        timeline_db=tmp_path / "customer_timeline.sqlite",
        per_brand=1,
        include_dual_neg=False,
    )
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    personas = [row for row in rows if row.get("type") == "persona"]

    assert report["personas"] == 1
    assert personas[0]["bot_safe_customer_id"] == "customer:test-memory"
    assert personas[0]["phone_ref"] == "sha256:abcdef1234567890"
    assert scenario_builder.find_raw_pii(payload) == []
    assert "+79991234567" not in payload
    assert "test@example.com" not in payload


def test_memory_measure_off_on_commands_use_same_set_and_flip_only_memory_env(tmp_path: Path) -> None:
    commands = off_on.build_commands(
        scenarios=tmp_path / "scenarios.jsonl",
        snapshot=tmp_path / "snapshot.json",
        timeline_db=tmp_path / "customer_timeline.sqlite",
        out_root=tmp_path / "out",
        parallel=4,
        judge_prompt_version="v9.1",
    )

    off = commands["off"]
    on = commands["on"]
    assert off["argv"][:-1] == on["argv"][:-1]
    assert off["env"]["TELEGRAM_BOT_SAFE_CRM_CONTEXT"] == "0"
    assert on["env"]["TELEGRAM_BOT_SAFE_CRM_CONTEXT"] == "1"
    assert on["env"]["TELEGRAM_BOT_SAFE_CRM_CONTEXT_DB"] == str(tmp_path / "customer_timeline.sqlite")
    assert "--parallel" in on["argv"]
    assert "4" in on["argv"]
    assert "--judge-prompt-version" in on["argv"]
    assert "v9.1" in on["argv"]


def test_memory_measure_execute_is_blocked_until_streams_ready(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(off_on.READY_ENV, raising=False)

    with pytest.raises(RuntimeError, match="Streams 1-2"):
        off_on.main(
            [
                "--scenarios",
                str(tmp_path / "scenarios.jsonl"),
                "--snapshot",
                str(tmp_path / "snapshot.json"),
                "--timeline-db",
                str(tmp_path / "customer_timeline.sqlite"),
                "--out-root",
                str(tmp_path / "out"),
                "--execute",
            ]
        )
