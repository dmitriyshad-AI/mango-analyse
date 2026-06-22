from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mango_mvp.channels.subscription_llm import SubscriptionDraftResult
from scripts.run_telegram_public_pilot_bots import BrandBotConfig
from scripts.check_public_bot_live import (
    ExpectedOnlinePrices,
    LiveCheckTurn,
    apply_runtime_env,
    expected_online_prices_from_snapshot,
    llm_fallback_detected,
    public_bot_heartbeat_status,
    retrieved_fact_keys,
    smoke_context_overrides,
    validate_turns,
    zero_drafts_alert,
)
from mango_mvp.channels.pilot_profile_runtime import DIRECT_PATH_PILOT_CONFIG_ENV, DIRECT_PATH_PILOT_CONFIG_VERSION


def test_retrieved_fact_keys_collects_nested_direct_path_metadata() -> None:
    result = SubscriptionDraftResult(
        route="bot_answer_self_for_pilot",
        draft_text="Ответ",
        context_used=("fact.context",),
        metadata={
            "direct_path": {
                "fact_pack": {
                    "facts": {"fact.a": "A"},
                    "exact_keys": ["fact.b"],
                    "llm_retrieve": {"supplemented_exact_ids": ["fact.c"]},
                }
            }
        },
    )

    assert retrieved_fact_keys(result) == ("fact.a", "fact.b", "fact.c", "fact.context")


def test_live_check_validation_catches_memory_and_fact_failures() -> None:
    turns = [
        LiveCheckTurn(
            name="physics_online",
            input_text="онлайн",
            answer_text="Какой класс?",
            route="draft_for_manager",
            safety_flags=(),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=(),
            known_slots={},
        )
    ]

    failures = validate_turns(turns)

    reasons = {item["reason"] for item in failures}
    assert {"retrieved_facts", "memory_known_slots", "no_grade_reask"}.issubset(reasons)


def test_live_check_validation_accepts_required_public_bot_behaviour() -> None:
    turns = [
        LiveCheckTurn(
            name="greeting",
            input_text="привет",
            answer_text="Здравствуйте! Расскажите, какой класс и предмет интересуют?",
            route="bot_answer_self_for_pilot",
            safety_flags=(),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=(),
            known_slots={},
        ),
        LiveCheckTurn(
            name="physics_first",
            input_text="8 класс физика",
            answer_text="Есть очный и онлайн-формат.",
            route="bot_answer_self_for_pilot",
            safety_flags=(),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=("schedule",),
            known_slots={},
        ),
        LiveCheckTurn(
            name="physics_online",
            input_text="онлайн",
            answer_text="Онлайн-группа: воскресенье 14:30–16:30, старт 20.09.2026. Семестр — 29 750 ₽, год — 47 250 ₽.",
            route="bot_answer_self_for_pilot",
            safety_flags=(),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=("schedule", "price"),
            known_slots={"grade": "8", "subject": "физика"},
        ),
        LiveCheckTurn(
            name="cross_brand",
            input_text="вы же одна контора с УНПК?",
            answer_text="Фотон и УНПК — разные бренды, по Фотону подскажу отдельно.",
            route="manager_only",
            safety_flags=("cross_brand",),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=(),
            known_slots={},
        ),
        LiveCheckTurn(
            name="p0_double_charge",
            input_text="двойное списание!",
            answer_text="Передам менеджеру.",
            route="manager_only",
            safety_flags=("p0_deferral",),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=(),
            known_slots={},
        ),
        LiveCheckTurn(
            name="pii_capture",
            input_text="запишите: Иванов Пётр 8-900-123-45-67",
            answer_text="Записала данные ребёнка, менеджер свяжется.",
            route="draft_for_manager",
            safety_flags=(),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=(),
            known_slots={},
        ),
    ]

    assert validate_turns(turns, expected_online_prices=ExpectedOnlinePrices(("29 750", "29750"), ("47 250", "47250"))) == ()


def test_live_check_validation_uses_brand_prices_from_snapshot(tmp_path) -> None:
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        """
        {
          "facts": [
            {
              "brand": "unpk",
              "allowed_for_client_answer": true,
              "client_safe_text": "УНПК МФТИ: годовые онлайн-курсы по математике и физике для 5–11 классов проходят по выходным. Стоимость: семестр — 37 000 ₽, год — 59 000 ₽.",
              "structured_value": {"format": "online", "classes": "5-11"}
            },
            {
              "brand": "foton",
              "allowed_for_client_answer": true,
              "client_safe_text": "Фотон: цены на 2026/27 учебный год, 5-11 класс, онлайн, семестр — 29 750 ₽.",
              "structured_value": {"format": "online", "classes": "5-11"}
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    prices = expected_online_prices_from_snapshot(snapshot, "unpk")
    turns = [
        LiveCheckTurn(
            name="physics_online",
            input_text="онлайн",
            answer_text="Онлайн-группа: воскресенье 14:30-16:30, старт 20.09. Семестр — 37 000 ₽, год — 59 000 ₽.",
            route="bot_answer_self_for_pilot",
            safety_flags=(),
            error="",
            llm_fallback=False,
            retrieved_fact_keys=("schedule", "price"),
            known_slots={"grade": "8", "subject": "физика"},
        )
    ]

    assert prices == ExpectedOnlinePrices(("37 000", "37000"), ("59 000", "59000"))
    assert validate_turns(turns, expected_online_prices=prices) == ()


def test_llm_fallback_detected_uses_flags_error_and_text() -> None:
    assert llm_fallback_detected(SubscriptionDraftResult(route="manager_only", draft_text="", safety_flags=("llm_fallback",)), "")
    assert llm_fallback_detected(SubscriptionDraftResult(route="manager_only", draft_text="", error="timeout"), "")
    assert not llm_fallback_detected(
        SubscriptionDraftResult(route="manager_only", draft_text="", error="authoritative_output_gate_blocked"),
        "Чтобы не ошибиться, передам вопрос менеджеру.",
    )


def test_zero_drafts_alert_missing_store_is_non_fatal(tmp_path) -> None:
    result = zero_drafts_alert(tmp_path / "missing.sqlite", now=datetime(2026, 6, 12, 12, tzinfo=timezone.utc))

    assert result["drafts_since"] is None
    assert result["alert"] is False


def test_apply_runtime_env_does_not_force_profile(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(DIRECT_PATH_PILOT_CONFIG_ENV, raising=False)

    apply_runtime_env({}, snapshot_path=tmp_path / "snapshot.json")

    assert DIRECT_PATH_PILOT_CONFIG_ENV not in os.environ


def _write_bot_heartbeat(path: Path, *, last_cycle_at: datetime, profile: str = DIRECT_PATH_PILOT_CONFIG_VERSION, guards=None) -> None:
    payload = {
        "schema_version": "public_pilot_bot_heartbeat_v2_2026_06_21",
        "status": "polling",
        "last_cycle_at": last_cycle_at.isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "brands": ["foton"],
        "event": "heartbeat",
        "effective_profile": profile,
        "draft_path": "direct_path" if profile else "dialogue_contract_pipeline",
        "active_guards": guards
        if guards is not None
        else {
            "presale_safety": True,
            "presale_pii_memory": True,
            "pii_relation_stopwords": True,
            "verifier_handoff_claims": True,
        },
        "summary": {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_public_bot_heartbeat_status_accepts_fresh_polling_heartbeat(tmp_path) -> None:
    now = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    path = tmp_path / "heartbeat.json"
    _write_bot_heartbeat(path, last_cycle_at=now - timedelta(seconds=20))

    status = public_bot_heartbeat_status(path, brand="foton", now=now, process_alive_fn=lambda _pid: True)

    assert status["ok"] is True
    assert status["effective_profile"] == DIRECT_PATH_PILOT_CONFIG_VERSION
    assert status["active_guards"]["presale_safety"] is True


def test_public_bot_heartbeat_status_rejects_stale_heartbeat(tmp_path) -> None:
    now = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    path = tmp_path / "heartbeat.json"
    _write_bot_heartbeat(path, last_cycle_at=now - timedelta(seconds=181))

    status = public_bot_heartbeat_status(path, brand="foton", now=now, stale_after_seconds=180, process_alive_fn=lambda _pid: True)

    assert status["ok"] is False
    assert any(item["reason"] == "public_bot_heartbeat_stale" for item in status["failures"])


def test_public_bot_heartbeat_status_rejects_profile_and_guard_off(tmp_path) -> None:
    now = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    path = tmp_path / "heartbeat.json"
    _write_bot_heartbeat(
        path,
        last_cycle_at=now,
        profile="",
        guards={
            "presale_safety": True,
            "presale_pii_memory": False,
            "pii_relation_stopwords": True,
            "verifier_handoff_claims": True,
        },
    )

    status = public_bot_heartbeat_status(path, brand="foton", now=now, process_alive_fn=lambda _pid: True)

    assert status["ok"] is False
    reasons = {(item["reason"], item.get("guard")) for item in status["failures"]}
    assert ("public_bot_profile_off", None) in reasons
    assert ("public_bot_guard_off", "presale_pii_memory") in reasons


def test_smoke_force_profile_uses_local_context_without_process_env(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv(DIRECT_PATH_PILOT_CONFIG_ENV, raising=False)
    config = BrandBotConfig(
        brand="foton",
        token="token",
        display_name="Фотон",
        snapshot_path=tmp_path / "snapshot.json",
        store_enabled=False,
    )

    overrides = smoke_context_overrides(config, smoke_force_profile=True)

    assert overrides == {DIRECT_PATH_PILOT_CONFIG_ENV: DIRECT_PATH_PILOT_CONFIG_VERSION}
    assert DIRECT_PATH_PILOT_CONFIG_ENV not in os.environ
