from __future__ import annotations

"""Бизнес-тесты PRODUCTIZATION (область 4): ASR-гейт, изоляция арендаторов, идемпотентный capture.

Запуск (в среде Кодекса, с PYTHONPATH=src):
    PYTHONPATH=src python3 D1_audit_backlog/codex_tests/test_productization.py

Защищаемые инварианты области:
  - ASR/исполнение НЕ запускается без approval-record:
      * execution_allowed всегда False в этом гейте;
      * без approval_ref гейт выдаёт BLOCK_ASR_EXECUTION_PENDING_APPROVAL;
      * job_plan.hard_guards.run_asr = False и план в режиме dry-run;
  - tenant isolation: строки без tenant_id в защищаемых таблицах блокируют валидацию,
    данные одного арендатора не отдаются в контексте другого (через tenant-фильтр);
  - capture идемпотентен: один и тот же event_key не ставится в очередь дважды.

ASR-гейт и tenant-isolation читают файлы/БД (живые ресурсы) — они вынесены в
TODO-скелеты с фейками. Чистые помощники (verify_technical_blockers, build_actions,
build_job_plan, safety_contract) и CapturePlanner тестируются напрямую.

Модули под проверкой:
  - mango_mvp.productization.asr_execution_approval_gate
  - mango_mvp.productization.tenant_isolation
  - mango_mvp.productization.capture (CapturePlanner — идемпотентность)
"""

from mango_mvp.productization import asr_execution_approval_gate as gate
from mango_mvp.productization import tenant_isolation as ti
from datetime import datetime, timezone

from mango_mvp.productization.capture import (
    CaptureAction,
    CapturePlanner,
    InMemorySeenCallStore,
)
from mango_mvp.productization.contracts import TelephonyCallEvent, TenantRef


_PASS = 0
_FAIL = 0
_FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
    else:
        _FAIL += 1
        _FAILURES.append(f"[FAIL] {name}: {detail}")


# --------------------------------------------------------------------------
# 4A. ASR approval gate — исполнение без approval-record невозможно
# --------------------------------------------------------------------------

def _ready_verify_summary() -> dict:
    # Технически "готовый" pack: схема ок, валидация ок, нет блоков, manifest==ready.
    from mango_mvp.productization.asr_worker_pack_verifier import (
        ASR_WORKER_PACK_VERIFY_SCHEMA_VERSION,
    )
    return {
        "schema_version": ASR_WORKER_PACK_VERIFY_SCHEMA_VERSION,
        "validation_ok": True,
        "blocked": 0,
        "manifest_rows": 5,
        "ready_items": 5,
    }


def _ready_readiness() -> dict:
    return {
        "ready_for_worker": True,
        "worker_may_run_asr": False,
        "requires_explicit_runtime_target_approval": True,
    }


def _ready_safety() -> dict:
    return {
        "read_only": True,
        "runtime_db_writes": False,
        "stable_runtime_writes": False,
        "run_asr": False,
        "run_ra": False,
        "write_crm": False,
        "write_tallanto": False,
    }


def test_gate_no_technical_blockers_when_ready() -> None:
    reasons = gate.verify_technical_blockers(_ready_verify_summary(), _ready_readiness(), _ready_safety())
    check("ready_pack_no_blockers", reasons == [], f"unexpected blockers: {reasons}")


def test_gate_blocks_when_safety_allows_asr() -> None:
    # Если источник вдруг разрешает run_asr — это технический блокер.
    bad_safety = dict(_ready_safety())
    bad_safety["run_asr"] = True
    reasons = gate.verify_technical_blockers(_ready_verify_summary(), _ready_readiness(), bad_safety)
    check("safety_run_asr_is_blocker", "source_safety_run_asr_must_be_false" in reasons,
          f"reasons={reasons}")


def test_gate_blocks_when_readiness_allows_asr() -> None:
    bad_readiness = dict(_ready_readiness())
    bad_readiness["worker_may_run_asr"] = True
    reasons = gate.verify_technical_blockers(_ready_verify_summary(), bad_readiness, _ready_safety())
    check("readiness_allows_asr_is_blocker",
          "readiness_gate_unexpectedly_allows_asr" in reasons, f"reasons={reasons}")


def test_gate_action_blocks_without_approval() -> None:
    # Готовый pack, НО approval_present=False → блок ожидания approval, исполнение запрещено.
    actions = gate.build_actions(readiness_ok=True, technical_reasons=[],
                                 approval_present=False, approval_ref=None)
    check("no_approval_blocks", actions[0]["action"] == "BLOCK_ASR_EXECUTION_PENDING_APPROVAL",
          f"action={actions[0]['action']}")
    check("no_approval_execution_false", actions[0]["execution_allowed"] is False)


def test_gate_action_with_approval_still_dry_run() -> None:
    # Даже с approval_ref гейт НЕ исполняет ASR — только планирует dry-run.
    actions = gate.build_actions(readiness_ok=True, technical_reasons=[],
                                 approval_present=True, approval_ref="approval#1")
    check("with_approval_plan_action",
          actions[0]["action"] == "PLAN_ASR_EXECUTION_APPROVAL_RECORDED_DRY_RUN",
          f"action={actions[0]['action']}")
    check("with_approval_execution_false", actions[0]["execution_allowed"] is False,
          "approval не должен включать исполнение в этом гейте")


def test_gate_action_blocks_when_not_ready() -> None:
    actions = gate.build_actions(readiness_ok=False, technical_reasons=["x"],
                                 approval_present=True, approval_ref="approval#1")
    check("not_ready_blocks_even_with_approval",
          actions[0]["action"] == "BLOCK_ASR_EXECUTION_PACK_NOT_READY",
          f"action={actions[0]['action']}")


def test_gate_job_plan_hard_guards_off() -> None:
    plan = gate.build_job_plan(verify_audit_path=__import__("pathlib").Path("/tmp/x.json"),
                               verify_summary=_ready_verify_summary(),
                               readiness=_ready_readiness(),
                               approval_ref="approval#1", status="approval_recorded_dry_run")
    check("job_plan_execution_false", plan["execution_allowed"] is False)
    guards = plan["hard_guards"]
    for key in ("run_asr", "run_ra", "write_runtime_db", "write_crm", "write_tallanto",
                "touch_stable_runtime", "download_audio", "copy_audio"):
        check(f"hard_guard_{key}_false", guards.get(key) is False, f"{key}={guards.get(key)}")
    check("job_plan_dry_run_mode", plan["mode"] == "approval_gate_dry_run", f"mode={plan['mode']}")


def test_gate_safety_contract_no_writes() -> None:
    c = gate.safety_contract()
    for key in ("run_asr", "run_ra", "write_crm", "write_tallanto",
                "runtime_db_writes", "stable_runtime_writes", "downloads_audio"):
        check(f"gate_safety_{key}_false", c.get(key) is False, f"{key}={c.get(key)}")


# --------------------------------------------------------------------------
# 4B. tenant_isolation — чистые помощники + СКЕЛЕТ для живой БД
# --------------------------------------------------------------------------

def test_tenant_safe_name_sanitizes() -> None:
    # tenant_id с разделителями пути не должен превращаться в обход директорий.
    check("safe_name_strips_slash", "/" not in ti.safe_name("foo/../bar"),
          f"got={ti.safe_name('foo/../bar')}")
    check("safe_name_strips_dots", ".." not in ti.safe_name("../etc"),
          f"got={ti.safe_name('../etc')}")
    check("safe_name_keeps_alnum", ti.safe_name("foton_unpk_1") == "foton_unpk_1",
          f"got={ti.safe_name('foton_unpk_1')}")
    check("safe_name_empty_fallback", ti.safe_name("") == "unknown_tenant",
          f"got={ti.safe_name('')}")


def test_tenant_required_tables_policy() -> None:
    # Инвариант: tenant_id обязателен для строк звонков/инбокса/маппинга владельцев.
    required = {"product_calls", "capture_inbox_items", "tenant_manager_owner_map"}
    check("tenant_scoped_tables_include_required", required <= set(ti.TENANT_SCOPED_TABLES),
          f"scoped={ti.TENANT_SCOPED_TABLES}")


def test_tenant_isolation_db_skeleton() -> None:
    """СКЕЛЕТ. build_tenant_isolation_report открывает product.sqlite в режиме mode=ro.

    Инвариант для проверки Кодексом:
      - строки без tenant_id в product_calls/capture_inbox_items/tenant_manager_owner_map
        дают blocked>0 и validation_ok=False (нельзя отдавать «бесхозные» строки);
      - запрос одного арендатора (через tenant-фильтр в read_api/product_api) НЕ
        возвращает строки другого арендатора.

    TODO(codex): подставить фикстуру:
      1. собрать временную product.sqlite с таблицами tenants, product_calls
         (две строки с разными tenant_id + одна без tenant_id);
      2. вызвать build_tenant_isolation_report(product_root=tmp, product_db_path=db);
      3. assert report['summary']['blocked'] >= 1 и validation_ok is False для строки без tenant_id;
      4. отдельно проверить, что выборка по tenant_id='t1' не содержит строк t2.
    """
    # Контракт безопасности доступен без БД — проверяем его прямо сейчас.
    c = ti.safety_contract()
    check("tenant_safety_read_only_db", c.get("read_only_db") is True)
    for key in ("runtime_db_writes", "stable_runtime_writes", "run_asr", "write_crm", "write_tallanto"):
        check(f"tenant_safety_{key}_false", c.get(key) is False, f"{key}={c.get(key)}")
    # TODO(codex): добавить позитивный/негативный путь с реальной product.sqlite.


# --------------------------------------------------------------------------
# 4C. CapturePlanner — идемпотентность по event_key
# --------------------------------------------------------------------------

def _event(call_id: str, recording_url: str | None = "https://rec/1",
           tenant_id: str = "t1") -> TelephonyCallEvent:
    # event_key — производное свойство от tenant+provider+provider_call_id,
    # поэтому идемпотентность завязана именно на этот тройной ключ.
    return TelephonyCallEvent(
        tenant=TenantRef(tenant_id=tenant_id),
        provider="mango",
        provider_call_id=call_id,
        started_at=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 5, 20, 12, 5, tzinfo=timezone.utc),
        recording_url=recording_url,
    )


def test_capture_enqueues_new_event_once() -> None:
    planner = CapturePlanner(InMemorySeenCallStore())
    d1 = planner.plan_event(_event("call-1"))
    check("first_seen_enqueued", d1.action == CaptureAction.ENQUEUE_SHADOW_CAPTURE,
          f"action={d1.action}")


def test_capture_idempotent_same_event_key() -> None:
    # Инвариант: тот же event_key второй раз → SKIP_DUPLICATE (нет двойной обработки).
    planner = CapturePlanner(InMemorySeenCallStore())
    planner.plan_event(_event("call-1"))
    d2 = planner.plan_event(_event("call-1"))
    check("second_seen_skipped", d2.action == CaptureAction.SKIP_DUPLICATE,
          f"action={d2.action}, reason={d2.reason}")


def test_capture_batch_dedup() -> None:
    planner = CapturePlanner(InMemorySeenCallStore())
    events = [_event("a"), _event("b"), _event("a")]  # 'a' встречается дважды
    decisions = planner.plan_batch(events)
    enqueued = [d for d in decisions if d.action == CaptureAction.ENQUEUE_SHADOW_CAPTURE]
    skipped = [d for d in decisions if d.action == CaptureAction.SKIP_DUPLICATE]
    check("batch_enqueues_unique_only", len(enqueued) == 2, f"enqueued={len(enqueued)}")
    check("batch_skips_duplicate", len(skipped) == 1, f"skipped={len(skipped)}")


def test_capture_skips_without_recording() -> None:
    # require_recording=True (по умолчанию): без аудио — не ставить в очередь.
    planner = CapturePlanner(InMemorySeenCallStore())
    d = planner.plan_event(_event("no-rec", recording_url=None))
    check("no_recording_skipped", d.action == CaptureAction.SKIP_NO_RECORDING,
          f"action={d.action}")


def test_capture_preseeded_store_skips() -> None:
    # Если event_key уже известен (например, из прошлого запуска) — пропуск.
    # event_key — производный хэш, поэтому сидим стор именно вычисленным ключом.
    known = _event("known")
    planner = CapturePlanner(InMemorySeenCallStore(initial_keys={known.event_key}))
    d = planner.plan_event(known)
    check("preseeded_skipped", d.action == CaptureAction.SKIP_DUPLICATE, f"action={d.action}")


def main() -> int:
    test_gate_no_technical_blockers_when_ready()
    test_gate_blocks_when_safety_allows_asr()
    test_gate_blocks_when_readiness_allows_asr()
    test_gate_action_blocks_without_approval()
    test_gate_action_with_approval_still_dry_run()
    test_gate_action_blocks_when_not_ready()
    test_gate_job_plan_hard_guards_off()
    test_gate_safety_contract_no_writes()
    test_tenant_safe_name_sanitizes()
    test_tenant_required_tables_policy()
    test_tenant_isolation_db_skeleton()
    test_capture_enqueues_new_event_once()
    test_capture_idempotent_same_event_key()
    test_capture_batch_dedup()
    test_capture_skips_without_recording()
    test_capture_preseeded_store_skips()
    print(f"PASS={_PASS}  FAIL={_FAIL}")
    for line in _FAILURES:
        print(line)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
