from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.customer_timeline.bot_safe_runtime_context import (
    BotSafeLookup,
    bot_safe_crm_context_enabled,
    build_bot_safe_crm_context,
)
from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    CustomerIdentity,
    IdentityLink,
    IdentityLinkType,
    IdentityStatus,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore


NOW = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)


def test_bot_safe_crm_context_default_off() -> None:
    assert bot_safe_crm_context_enabled(None) is False
    assert bot_safe_crm_context_enabled("") is False
    assert bot_safe_crm_context_enabled("1") is True


def test_bot_safe_crm_context_reads_only_allowed_active_brand_chunks(tmp_path: Path) -> None:
    db_path, customer_id = _seed_bot_safe_timeline(tmp_path)

    context = build_bot_safe_crm_context(
        timeline_db=db_path,
        allowed_root=tmp_path,
        active_brand="foton",
        lookup=BotSafeLookup(tenant_id="foton", amo_lead_id="5001", amo_contact_id="7001"),
    )

    raw = json.dumps(context, ensure_ascii=False)
    assert context["found"] is True
    assert "Фотон: клиент уже спрашивал про онлайн-курс" in raw
    assert "Без бренда: клиент ранее уточнял удобный формат" in raw
    assert "УНПК: клиент интересовался выездной школой" not in raw
    assert customer_id not in raw
    assert "botsafe:" not in raw
    assert "chunk-foton" not in raw
    assert "Отправить телефон менеджера" not in raw
    assert "Спорный шаг не выводить" not in raw
    assert context["timeline_context"]["safety"]["customer_profile_included"] is False
    items = context["timeline_context"]["bot_context"]["items"]
    assert {item["text"]: item["next_step_status"] for item in items} == {
        "Фотон: клиент уже спрашивал про онлайн-курс. Следующий шаг: отправить расписание.": "active",
        "Без бренда: клиент ранее уточнял удобный формат.": "needs_manager_review",
    }


def test_bot_safe_crm_context_can_resolve_explicit_customer_id_for_measurements(tmp_path: Path) -> None:
    db_path, customer_id = _seed_bot_safe_timeline(tmp_path)

    context = build_bot_safe_crm_context(
        timeline_db=db_path,
        allowed_root=tmp_path,
        active_brand="unpk",
        lookup=BotSafeLookup(tenant_id="foton", customer_id=customer_id),
    )

    raw = json.dumps(context, ensure_ascii=False)
    assert context["found"] is True
    assert "УНПК: клиент интересовался выездной школой" in raw
    assert "Без бренда: клиент ранее уточнял удобный формат" in raw
    assert "Фотон: клиент уже спрашивал про онлайн-курс" not in raw


def test_bot_safe_crm_context_blocks_ambiguous_identity(tmp_path: Path) -> None:
    db_path, _customer_id = _seed_bot_safe_timeline(tmp_path, duplicate_lead=True)

    context = build_bot_safe_crm_context(
        timeline_db=db_path,
        allowed_root=tmp_path,
        active_brand="foton",
        lookup=BotSafeLookup(tenant_id="foton", amo_lead_id="5001"),
    )

    assert context["found"] is False
    assert "ambiguous_identity" in context["warnings"]


def test_bot_safe_crm_context_drops_chunks_with_pii(tmp_path: Path) -> None:
    db_path, _customer_id = _seed_bot_safe_timeline(tmp_path, pii_chunk=True)

    context = build_bot_safe_crm_context(
        timeline_db=db_path,
        allowed_root=tmp_path,
        active_brand="foton",
        lookup=BotSafeLookup(tenant_id="foton", amo_lead_id="5001"),
    )

    raw = json.dumps(context, ensure_ascii=False)
    assert context["found"] is True
    assert "Фотон: клиент уже спрашивал про онлайн-курс" in raw
    assert "edu@example.com" not in raw
    assert "+79991234567" not in raw


def test_bot_safe_crm_context_opens_read_only_db_under_path_with_spaces(tmp_path: Path) -> None:
    db_path, customer_id = _seed_bot_safe_timeline(tmp_path / "path with spaces")

    context = build_bot_safe_crm_context(
        timeline_db=db_path,
        allowed_root=db_path.parent,
        active_brand="foton",
        lookup=BotSafeLookup(tenant_id="foton", customer_id=customer_id),
    )

    assert context["found"] is True
    assert "Фотон: клиент уже спрашивал про онлайн-курс" in context["summary"]


def test_bot_safe_crm_context_keeps_multiline_dossier_up_to_new_runtime_limit(tmp_path: Path) -> None:
    db_path, customer_id = _seed_bot_safe_timeline(tmp_path)
    long_text = (
        "Бренд: Фотон.\n"
        "Обсуждали:\n"
        "- " + "клиент уточнял онлайн-формат " * 30 + "\n"
        "Интерес / возражения:\n"
        "- интерес к физике ОГЭ.\n"
        "Договорённость / следующий шаг:\n"
        "- семья сравнит варианты."
    )
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    store.upsert_bot_context_chunk(
        BotContextChunk(
            tenant_id="foton",
            customer_id=customer_id,
            chunk_id="chunk-long-foton",
            chunk_type="bot_safe_summary",
            text=long_text,
            source_system="customer_timeline_bot_safe_summary",
            source_ref=f"botsafe:{customer_id}:foton:long",
            event_at=NOW,
            freshness_score=1.0,
            relevance_tags=("bot_safe", "structured", "foton"),
            allowed_for_bot=True,
            requires_manager_review=False,
            metadata={"next_step": {"status": "active"}},
        )
    )
    store.close()

    context = build_bot_safe_crm_context(
        timeline_db=db_path,
        allowed_root=tmp_path,
        active_brand="foton",
        lookup=BotSafeLookup(tenant_id="foton", customer_id=customer_id),
        limit=5,
    )

    raw = json.dumps(context, ensure_ascii=False)
    assert context["found"] is True
    assert "Интерес / возражения:" in raw
    assert "Договорённость / следующий шаг:" in raw
    assert "семья сравнит варианты" in raw
    assert len(context["summary"]) > 700


def _seed_bot_safe_timeline(
    tmp_path: Path,
    *,
    duplicate_lead: bool = False,
    pii_chunk: bool = False,
) -> tuple[Path, str]:
    db_path = tmp_path / "customer_timeline.sqlite"
    store = CustomerTimelineSQLiteStore(db_path, allowed_root=tmp_path)
    customer = CustomerIdentity(
        tenant_id="foton",
        identity_status=IdentityStatus.STRONG,
        customer_id="customer:test-foton",
        display_name="Safe Test",
        created_at=NOW,
        updated_at=NOW,
    )
    store.upsert_customer(customer)
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=customer.customer_id,
            link_type=IdentityLinkType.AMO_LEAD_ID,
            link_value="5001",
            source_system="amo",
            source_ref="lead:5001",
        )
    )
    store.upsert_identity_link(
        IdentityLink(
            tenant_id="foton",
            customer_id=customer.customer_id,
            link_type=IdentityLinkType.AMO_CONTACT_ID,
            link_value="7001",
            source_system="amo",
            source_ref="contact:7001",
        )
    )
    if duplicate_lead:
        other = CustomerIdentity(
            tenant_id="foton",
            identity_status=IdentityStatus.STRONG,
            customer_id="customer:test-other",
            created_at=NOW,
            updated_at=NOW,
        )
        store.upsert_customer(other)
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id=other.customer_id,
                link_type=IdentityLinkType.AMO_LEAD_ID,
                link_value="5001",
                source_system="amo",
                source_ref="lead:5001:duplicate",
            )
        )
    for chunk in (
        BotContextChunk(
            tenant_id="foton",
            customer_id=customer.customer_id,
            chunk_id="chunk-foton",
            chunk_type="bot_safe_summary",
            text="Фотон: клиент уже спрашивал про онлайн-курс. Следующий шаг: отправить расписание.",
            source_system="customer_timeline_bot_safe_summary",
            source_ref=f"botsafe:{customer.customer_id}:foton",
            event_at=NOW,
            freshness_score=1.0,
            relevance_tags=("bot_safe", "structured", "foton"),
            allowed_for_bot=True,
            requires_manager_review=False,
            metadata={"next_step": {"status": "active", "display_text": "Отправить телефон менеджера +79991234567"}},
        ),
        BotContextChunk(
            tenant_id="foton",
            customer_id=customer.customer_id,
            chunk_id="chunk-unpk",
            chunk_type="bot_safe_summary",
            text="УНПК: клиент интересовался выездной школой.",
            source_system="customer_timeline_bot_safe_summary",
            source_ref=f"botsafe:{customer.customer_id}:unpk",
            event_at=NOW,
            freshness_score=1.0,
            relevance_tags=("bot_safe", "structured", "unpk"),
            allowed_for_bot=True,
            requires_manager_review=False,
            metadata={"next_step": {"status": "active"}},
        ),
        BotContextChunk(
            tenant_id="foton",
            customer_id=customer.customer_id,
            chunk_id="chunk-unknown",
            chunk_type="bot_safe_summary",
            text="Без бренда: клиент ранее уточнял удобный формат.",
            source_system="customer_timeline_bot_safe_summary",
            source_ref=f"botsafe:{customer.customer_id}:unknown",
            event_at=NOW,
            freshness_score=1.0,
            relevance_tags=("bot_safe", "structured", "unknown"),
            allowed_for_bot=True,
            requires_manager_review=False,
            metadata={"next_step": {"status": "needs_manager_review", "display_text": "Спорный шаг не выводить"}},
        ),
    ):
        store.upsert_bot_context_chunk(chunk)
    if pii_chunk:
        store.upsert_bot_context_chunk(
            BotContextChunk(
                tenant_id="foton",
                customer_id=customer.customer_id,
                chunk_id="chunk-pii",
                chunk_type="bot_safe_summary",
                text="Фотон: телефон +79991234567, почта edu@example.com.",
                source_system="customer_timeline_bot_safe_summary",
                source_ref=f"botsafe:{customer.customer_id}:foton:pii",
                event_at=NOW,
                relevance_tags=("bot_safe", "structured", "foton"),
                allowed_for_bot=True,
                requires_manager_review=False,
            )
        )
    store.close()
    return db_path, customer.customer_id
