"""БЛОК 4 acceptance tests: "правильная память для всех личных Wappi-диалогов".

Covers the 8 required scenarios from the ТЗ. Each test drives real production
entry points (build_widget_resolver / AmoWappiDraftLoop / build_bot_safe_crm_context /
the direct-path prompt builder) with synthetic fixtures -- no staging/live/external
API calls, no PII. DB fixtures reuse the existing helpers in
tests.test_bot_safe_runtime_context and tests.test_draft_loop instead of
re-implementing the seeding/fake-client plumbing.

Mapping:
  1. test_exact_widget_pair_gets_family_scoped_memory
  2. test_generic_auto_pair_gets_memory_after_strong_amo_check
  3. test_contact_id_change_blocks_old_memory_as_conflict
  4. test_shared_family_phone_and_ambiguous_identity_block_memory
  5. test_wrong_brand_blocks_memory_at_resolver_and_draft_loop
  6. test_two_children_in_one_family_keep_separate_records
  7. test_client_text_never_contains_phone_email_foreign_name_or_internal_id
  8. test_old_bot_safe_chunk_is_not_served_as_fresh
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scripts.run_amo_wappi_draft_loop as runner
from mango_mvp.customer_timeline.bot_safe_runtime_context import (
    BOT_SAFE_CRM_CONTEXT_ENV,
    BotSafeLookup,
    build_bot_safe_crm_context,
)
from mango_mvp.customer_timeline.bot_safe_summary import _freshness_score_for_source_date
from mango_mvp.customer_timeline.contracts import (
    CustomerIdentity,
    CustomerOpportunity,
    IdentityLink,
    IdentityLinkType,
    IdentityStatus,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.channels.subscription_llm_parts.direct_path import _build_direct_path_prompt
from mango_mvp.integrations.draft_loop import DraftLoopConfig, DraftLoopKey, DraftLoopPair, DraftLoopProfile

from tests.test_bot_safe_direct_path_context import _context
from tests.test_bot_safe_runtime_context import NOW, _seed_bot_safe_timeline, _seed_family_rows
from tests.test_draft_loop import FakeAmo, FakeBot, _config, _loop, _message
from tests.test_run_amo_wappi_draft_loop import _lead


def _snapshot(tmp_path: Path) -> Path:
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps({"schema_version": "kc_knowledge_snapshot_v1", "run_id": "block4", "facts": [], "chunks": []}),
        encoding="utf-8",
    )
    return path


# 1. Exact wappi_amo_widget pair -> resolves in Timeline -> build_bot_safe_crm_context
#    hands the direct path the right FAMILY's allowed chunks (not just "some memory").
def test_exact_widget_pair_gets_family_scoped_memory(tmp_path: Path, monkeypatch) -> None:
    db_path, customer_id = _seed_bot_safe_timeline(tmp_path)
    _seed_family_rows(db_path, customer_id=customer_id)
    monkeypatch.setenv(BOT_SAFE_CRM_CONTEXT_ENV, "1")
    key = DraftLoopKey("profile-foton", "chat-1")
    pair = DraftLoopPair(
        key=key,
        lead_id="5001",
        contact_id="7001",
        expected_brand="foton",
        source="wappi_amo_widget",
    )
    config = DraftLoopConfig(
        profiles={"profile-foton": DraftLoopProfile("profile-foton", "foton", "telegram")},
        pairs={key: pair},
    )
    build_context = runner.build_context_builder(
        _snapshot(tmp_path),
        draft_config=config,
        customer_timeline_db=db_path,
        customer_timeline_allowed_root=tmp_path,
    )

    context = build_context(key, (), "Что у вас есть для моего ребёнка?", "foton")

    raw = json.dumps(context.get("read_only_customer_context"), ensure_ascii=False)
    assert "класс: 8" in raw
    assert "предметы: физика" in raw
    assert customer_id not in raw
    assert "botsafe:" not in raw


# 2. An auto-discovered pair receives memory only after the canonical Timeline
#    resolver independently proves its AMO identity strong and unique.
def test_generic_auto_pair_gets_memory_after_strong_amo_check(tmp_path: Path, monkeypatch) -> None:
    db_path, customer_id = _seed_bot_safe_timeline(tmp_path)
    monkeypatch.setenv(BOT_SAFE_CRM_CONTEXT_ENV, "1")
    key = DraftLoopKey("profile-foton", "chat-1")
    auto_pair = DraftLoopPair(key=key, lead_id="5001", contact_id="7001", expected_brand="foton", source="auto")
    cfg = _config(tmp_path, pairs={key: auto_pair})
    build_context = runner.build_context_builder(
        _snapshot(tmp_path),
        draft_config=cfg,
        customer_timeline_db=db_path,
        customer_timeline_allowed_root=tmp_path,
    )
    bot = FakeBot()
    amo = FakeAmo()
    loop = _loop(tmp_path, messages=[_message("m1")], pairs={key: auto_pair}, bot=bot, amo=amo)
    loop.context_builder = build_context
    loop.trusted_auto_customer_context_builder = build_context

    summary = loop.run_once(dry_run=False)

    assert summary["bot_calls"] == 1  # the dialog is NOT dropped/excluded
    assert len(bot.calls) == 1
    context = bot.calls[0]["context"]
    assert context["read_only_customer_context"]["found"] is True
    assert customer_id not in json.dumps(context, ensure_ascii=False)
    assert amo.notes, "a neutral draft / manager-review note must still be written"


# 3. A widget candidate whose contact_id differs from the already-paired contact_id
#    is an identity conflict: the cycle must not hand the (possibly wrong-person)
#    old memory to the model at all.
def test_contact_id_change_blocks_old_memory_as_conflict(tmp_path: Path, monkeypatch) -> None:
    db_path, customer_id = _seed_bot_safe_timeline(tmp_path)
    monkeypatch.setenv(BOT_SAFE_CRM_CONTEXT_ENV, "1")
    key = DraftLoopKey("profile-foton", "chat-1")
    old_pair = DraftLoopPair(key=key, lead_id="5001", contact_id="7001", expected_brand="foton", source="wappi_amo_widget")
    cfg = _config(tmp_path, pairs={key: old_pair})
    build_context = runner.build_context_builder(
        _snapshot(tmp_path),
        draft_config=cfg,
        customer_timeline_db=db_path,
        customer_timeline_allowed_root=tmp_path,
    )
    bot = FakeBot()
    loop = _loop(
        tmp_path,
        messages=[_message("m1")],
        pairs={key: old_pair},
        bot=bot,
        auto_resolver=lambda **_kwargs: {
            "status": "matched",
            "source": "wappi_amo_widget",
            "lead_id": "",
            "contact_id": "9999",  # a different AMO contact than the persisted pair
            "match_key": "wappi_widget_contact",
        },
    )
    loop.context_builder = build_context

    summary = loop.run_once(dry_run=True)

    assert summary["bot_calls"] == 0
    assert bot.calls == []  # the old customer's memory was never handed to the model
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row["event"] == "auto_pair_identity_conflict" for row in rows)
    assert any(
        row.get("auto_candidate", {}).get("reason") == "auto_pair_contact_changed"
        for row in rows
        if row.get("event") == "pair_missing"
    )


# 4. A shared family phone (open timeline_conflicts row) and a genuinely ambiguous
#    identity link (the same amo_contact_id claimed by two customer records) must
#    both block memory rather than silently guessing.
def test_shared_family_phone_and_ambiguous_identity_block_memory(tmp_path: Path) -> None:
    db_path, customer_id = _seed_bot_safe_timeline(tmp_path)
    _seed_family_rows(db_path, customer_id=customer_id)
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.record_conflict(
            "foton",
            conflict_type="shared_family_phone",
            entity_refs=(f"customer:{customer_id}",),
        )

    blocked = build_bot_safe_crm_context(
        timeline_db=db_path,
        allowed_root=tmp_path,
        active_brand="foton",
        lookup=BotSafeLookup(tenant_id="foton", amo_lead_id="5001", amo_contact_id="7001"),
    )

    assert blocked["timeline_context"]["family_dossier"]["context_blocked"] is True
    assert "онлайн-курс" not in blocked["summary"]

    ambiguous_db = tmp_path / "ambiguous" / "customer_timeline.sqlite"
    ambiguous_db.parent.mkdir()
    store2 = CustomerTimelineSQLiteStore(ambiguous_db, allowed_root=ambiguous_db.parent)
    for suffix in ("a", "b"):
        person = CustomerIdentity(tenant_id="foton", identity_status=IdentityStatus.STRONG, customer_id=f"customer:shared-{suffix}")
        store2.upsert_customer(person)
        store2.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=person.customer_id,
                link_type=IdentityLinkType.AMO_CONTACT_ID,
                link_value="9001",
                source_system="amocrm_snapshot",
                source_ref=f"contact:9001:{suffix}",
            )
        )
    store2.close()

    ambiguous = build_bot_safe_crm_context(
        timeline_db=ambiguous_db,
        allowed_root=ambiguous_db.parent,
        active_brand="foton",
        lookup=BotSafeLookup(tenant_id="foton", amo_contact_id="9001"),
    )

    assert ambiguous["found"] is False
    assert "ambiguous_identity" in ambiguous["warnings"]


# 5. The widget resolver's brand guard rejects the pairing when the profile's brand
#    does not match the only active lead's organization -- and the draft loop never
#    reaches a bot call (hence never a memory leak across brands) for that cycle.
def test_wrong_brand_blocks_memory_at_resolver_and_draft_loop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AMO_WAPPI_CRM_ID", "crm-1")

    class WappiClient:
        def list_all_profiles(self):
            return [{"profile_id": "profile-foton", "platform": "tg", "uuid": "profile-uuid"}]

        def find_amocrm_contact_for_dialog(self, **_kwargs):
            return {"contact": {"id": 2002}, "leads": [{"id": 1001, "status_id": 123}]}

    class AmoReadClient:
        def amo_api_get(self, **_kwargs):
            return _lead("1001", org="УНПК МФТИ", contacts=("2002",))

    result = runner.build_widget_resolver(WappiClient(), AmoReadClient())(
        key=DraftLoopKey("profile-foton", "chat-1"),
        profile=DraftLoopProfile("profile-foton", "foton", "telegram"),
        dialog={"id": "chat-1", "type": "user"},
        messages=(),
        message=None,
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "wappi_widget_brand_conflict"

    bot = FakeBot()
    loop = _loop(tmp_path, messages=[_message("m1")], bot=bot, auto_resolver=lambda **_kwargs: dict(result))

    summary = loop.run_once(dry_run=True)

    assert summary["bot_calls"] == 0
    assert bot.calls == []


# 6. Two children of the same family must stay separate child records: each lookup
#    surfaces only its own attributed child's grade/subject, never the sibling's.
def test_two_children_in_one_family_keep_separate_records(tmp_path: Path) -> None:
    db_path, parent_id = _seed_bot_safe_timeline(tmp_path)
    _seed_family_rows(db_path, customer_id=parent_id, second_child=True, subjects=("физика",))
    second_child_id = "customer:second-child"
    with CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path) as store:
        store.upsert_customer(CustomerIdentity(tenant_id="foton", customer_id=second_child_id, identity_status=IdentityStatus.STRONG))
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=parent_id,
                link_type=IdentityLinkType.AMO_LEAD_ID,
                link_value="5002",
                source_system="amocrm_snapshot",
                source_ref="lead:5002",
            )
        )
        opp_child1 = CustomerOpportunity(
            tenant_id="foton", customer_id=parent_id, opportunity_type="amo_deal",
            source_system="amocrm_snapshot", source_id="5001", status="active", product_context={"brand": "foton"},
        )
        opp_child2 = CustomerOpportunity(
            tenant_id="foton", customer_id=second_child_id, opportunity_type="amo_deal",
            source_system="amocrm_snapshot", source_id="5002", status="active", product_context={"brand": "foton"},
        )
        store.upsert_opportunity(opp_child1)
        store.upsert_opportunity(opp_child2)
    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT INTO opportunity_child_attribution_v1 VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("foton", opp_child1.opportunity_id, parent_id, "child:1", "matched", "high", "exact lead", "{}", NOW.isoformat(), "attr-child1", "{}"),
        )
        con.execute(
            "INSERT INTO opportunity_child_attribution_v1 VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            ("foton", opp_child2.opportunity_id, second_child_id, "child:2", "matched", "high", "exact lead", "{}", NOW.isoformat(), "attr-child2", "{}"),
        )

    child1_context = build_bot_safe_crm_context(
        timeline_db=db_path, allowed_root=tmp_path, active_brand="foton",
        lookup=BotSafeLookup(tenant_id="foton", amo_lead_id="5001", amo_contact_id="7001"),
    )
    child2_context = build_bot_safe_crm_context(
        timeline_db=db_path, allowed_root=tmp_path, active_brand="foton",
        lookup=BotSafeLookup(tenant_id="foton", amo_lead_id="5002", amo_contact_id="7001"),
    )

    assert child1_context["timeline_context"]["family_dossier"]["child_scope"] == "lead_attributed"
    assert "класс: 8" in child1_context["summary"] and "физика" in child1_context["summary"]
    assert "математика" not in child1_context["summary"] and "класс: 9" not in child1_context["summary"]

    assert child2_context["timeline_context"]["family_dossier"]["child_scope"] == "lead_attributed"
    assert "класс: 9" in child2_context["summary"] and "математика" in child2_context["summary"]
    assert "физика" not in child2_context["summary"] and "класс: 8" not in child2_context["summary"]


# 7. Phone, email, a third party's name (e.g. the curator's) and an internal service
#    id must never reach the text actually assembled for the client, even if a raw
#    memory chunk happens to contain all four at once. Safe surrounding content from
#    other chunks must still come through (nothing is over-blocked).
def test_client_text_never_contains_phone_email_foreign_name_or_internal_id() -> None:
    context = _context(
        flag=True,
        include_unknown=False,
        extra_items=[
            {
                "chunk_id": "chunk-leaky",
                "chunk_type": "bot_safe_summary",
                "text": (
                    "Фотон: куратор Мария Смирнова обновила карточку customer:abcdef0123456789ab. "
                    "Звоните на +7 999 123-45-67 или пишите на family@example.com."
                ),
                "relevance_tags": ["bot_safe", "structured", "foton"],
                "allowed_for_bot": True,
                "requires_manager_review": False,
            }
        ],
    )

    prompt = _build_direct_path_prompt("Что у вас есть по моему вопросу?", context=context, facts={"fact:1": "Безопасный факт"})

    for forbidden in ("+7 999 123-45-67", "family@example.com", "Мария Смирнова", "customer:abcdef0123456789ab"):
        assert forbidden not in prompt
    # the safe, unrelated memory chunk is not collaterally dropped:
    assert "Фотон: клиент уже спрашивал про онлайн-курс" in prompt


# 8. freshness_score must reflect the actual source date instead of an eternal 1.0,
#    and an old bot-safe chunk must be explicitly marked historical in the direct
#    path prompt rather than presented as an equally-fresh fact.
def test_old_bot_safe_chunk_is_not_served_as_fresh() -> None:
    now = datetime.now(timezone.utc)

    recent_score = _freshness_score_for_source_date(now - timedelta(days=3), now=now)
    old_score = _freshness_score_for_source_date(now - timedelta(days=400), now=now)
    unknown_score = _freshness_score_for_source_date(None, now=now)

    assert recent_score == 1.0
    assert old_score == 0.2  # not the old eternal 1.0
    assert old_score < recent_score
    assert unknown_score == 0.5

    context = _context(
        flag=True,
        include_unknown=False,
        extra_items=[
            {
                "chunk_id": "chunk-fresh",
                "chunk_type": "bot_safe_summary",
                "text": "Фотон: обсуждали формат занятий на этой неделе.",
                "event_at": now.isoformat(),
                "freshness_score": 0.95,
                "relevance_tags": ["bot_safe", "structured", "foton"],
                "allowed_for_bot": True,
                "requires_manager_review": False,
            },
            {
                "chunk_id": "chunk-old",
                "chunk_type": "bot_safe_summary",
                "text": "Фотон: клиент год назад спрашивал про пробное занятие.",
                "event_at": (now - timedelta(days=400)).isoformat(),
                "freshness_score": 0.15,
                "relevance_tags": ["bot_safe", "structured", "foton"],
                "allowed_for_bot": True,
                "requires_manager_review": False,
            },
        ],
    )

    prompt = _build_direct_path_prompt("Что у нас по занятиям?", context=context, facts={"fact:1": "Безопасный факт"})

    fresh_line = next(line for line in prompt.splitlines() if "обсуждали формат занятий" in line)
    old_line = next(line for line in prompt.splitlines() if "год назад спрашивал" in line)
    assert "[историческая запись]" not in fresh_line
    assert "[историческая запись]" in old_line
    assert "нельзя называть клиенту актуальные цены, даты, расписание или наличие мест" in prompt
