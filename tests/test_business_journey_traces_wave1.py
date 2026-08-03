"""Business journeys that end at the real manager-note formatter."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import scripts.run_amo_wappi_draft_loop as draft_runner
from mango_mvp.channels.output_verification_floor import verify_output
from mango_mvp.channels.subscription_llm import SubscriptionDraftResult
from mango_mvp.channels.subscription_llm_parts.provider import SubscriptionLlmDraftProvider
from mango_mvp.customer_timeline.bot_safe_runtime_context import BOT_SAFE_CRM_CONTEXT_ENV
from mango_mvp.customer_timeline.contracts import (
    BotContextChunk,
    CustomerIdentity,
    IdentityLink,
    IdentityLinkType,
    IdentityStatus,
)
from mango_mvp.customer_timeline.store import CustomerTimelineSQLiteStore
from mango_mvp.integrations.amo_wappi_phase1 import (
    DRAFT_NOTE_MARKER,
    AiOfficeAmoNoteClient,
    AiOfficeClientConfig,
)
from mango_mvp.integrations.draft_loop import AmoWappiDraftLoop, DraftLoopKey, DraftLoopPair

from tests.test_bot_safe_runtime_context import _seed_bot_safe_timeline, _seed_family_rows
from tests.test_draft_loop import _SendTrapWappi, _config, _message


_NO_AUX_MODEL_FLAGS = {
    "TELEGRAM_DIRECT_PATH_PILOT_CONFIG": "",
    "TELEGRAM_LLM_RETRIEVE": "0",
    "TELEGRAM_SEMANTIC_OUTPUT_VERIFIER": "0",
    "TELEGRAM_DIRECT_SLOT_TOPIC_SHADOW": "0",
    "TELEGRAM_SEMANTIC_FRAME_SHADOW": "0",
    "TELEGRAM_SEMANTIC_FRAME_POSTHOC_SHADOW": "0",
    "TELEGRAM_SEMANTIC_FRAME_DECISION_SHADOW": "0",
    "TELEGRAM_SEMANTIC_FRAME_SELF_ANSWER_SHADOW": "0",
    "TELEGRAM_SEMANTIC_FRAME_EXISTENCE_PROOF_SHADOW": "0",
    "TELEGRAM_SEMANTIC_FRAME_PROOF_RECONCILIATION_SHADOW": "0",
    "TELEGRAM_INTENT_MODEL_LED": "0",
    "TELEGRAM_ROUTE_RUBRIC": "0",
}


class _HermeticProvider(SubscriptionLlmDraftProvider):
    def __init__(self, result: SubscriptionDraftResult) -> None:
        self.external_calls = 0
        self.aux_calls = 0

        def deny_external_runner(*_args: Any, **_kwargs: Any) -> Any:
            self.external_calls += 1
            raise AssertionError("business journey attempted an external model runner")

        super().__init__(runner=deny_external_runner)
        self.result = result
        self.calls = 0
        self.last_prompt = ""

    def _direct_path_draft_runner(self, prompt: str) -> SubscriptionDraftResult:
        self.calls += 1
        self.last_prompt = prompt
        return self.result

    def _run_prompt_text(self, *_args: Any, **_kwargs: Any) -> str:
        self.aux_calls += 1
        raise AssertionError("business journey attempted an unstubbed auxiliary model call")


def _write_safe_snapshot(tmp_path: Path) -> Path:
    snapshot = {
        "schema_version": "kc_knowledge_snapshot_v1",
        "run_id": "business-journey",
        "facts": [
            {
                "brand": "foton",
                "fact_id": "fact:v3:foton:processes_2026_06_10_foton_personal_cabinet:8a8fdc5709",
                "fact_key": "processes_2026_06_10.foton.personal_cabinet",
                "fact_type": "process",
                "product": "customer_processes_2026_06_10",
                "allowed_for_client_answer": True,
                "forbidden_for_client": False,
                "internal_only": False,
                "requires_manager_confirmation": False,
                "verification_status": "verified",
                "freshness_status": "document_verified",
                "freshness_check_date": "2026-06-12",
                "valid_until": "2026-12-31",
                "owner_role": "РОП/ответственный владелец факта",
                "source_sha256": "37cbe5207f307f006f78c66ef3a08c62f9c3b5906c15bbfd9908129457cc6c91",
                "source_title": "UPDATE_REPORT_2026-05-19.md (Claude layer v3)",
                "client_safe_text": (
                    "В личном кабинете учебной платформы доступны занятия, материалы и записи "
                    "онлайн-уроков. С лета 2026 новые онлайн-занятия проходят на SohoLMS; курсы, "
                    "начавшиеся до лета 2026, завершаются в MTS-Link. Если пароль забыт, его "
                    "восстанавливают через форму восстановления."
                ),
            },
            {
                "brand": "unpk",
                "fact_key": "unpk.format.offline",
                "fact_type": "format",
                "product": "regular_course",
                "allowed_for_client_answer": True,
                "forbidden_for_client": False,
                "internal_only": False,
                "client_safe_text": "УНПК: занятия проходят очно.",
            },
        ],
        "chunks": [],
    }
    path = tmp_path / "business_journey_snapshot.json"
    path.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    return path


def _capturing_note_client(captured: dict[str, Any]) -> AiOfficeAmoNoteClient:
    def transport(**kwargs: Any) -> Mapping[str, Any]:
        captured.setdefault("requests", []).append(dict(kwargs))
        return {"status": "ok", "note_id": 9001}

    class ReadbackClient:
        def amo_api_get(self, **_kwargs: Any) -> Mapping[str, Any]:
            requests = captured.get("requests")
            if not isinstance(requests, list) or not requests:
                return {"_embedded": {"notes": []}}
            request = requests[-1]
            return {
                "_embedded": {
                    "notes": [
                        {
                            "id": 9001,
                            "params": {"text": str(request["json_body"]["text"])},
                        }
                    ]
                }
            }

    return AiOfficeAmoNoteClient(
        AiOfficeClientConfig(base_url="https://api.fotonai.online", api_key="synthetic"),
        transport=transport,
        read_client=ReadbackClient(),
    )


def _semantic_frame(*, is_p0: bool = False) -> Mapping[str, Any]:
    return {
        "source": "inline",
        "schema_version": "semantic_frame_v1_2026_07_01",
        "requested_action": "answer_question",
        "risk_class": "high" if is_p0 else "safe",
        "answerability": "manager_only" if is_p0 else "answer_self",
        "must_handoff": is_p0,
        "confidence": 0.99,
        "is_p0": is_p0,
        "p0_kind": "payment_dispute" if is_p0 else "none",
    }


def _seed_other_customer_memory(timeline_db: Path, *, allowed_root: Path) -> str:
    marker = "Фотон: другой клиент просил подготовить олимпиадный план."
    now = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)
    with CustomerTimelineSQLiteStore(timeline_db, allowed_root=allowed_root) as store:
        store.upsert_customer(
            CustomerIdentity(
                tenant_id="foton",
                identity_status=IdentityStatus.STRONG,
                customer_id="customer:other-same-brand",
                display_name="Other Customer",
                created_at=now,
                updated_at=now,
            )
        )
        store.upsert_identity_link(
            IdentityLink(
                tenant_id="foton",
                customer_id="customer:other-same-brand",
                link_type=IdentityLinkType.AMO_CONTACT_ID,
                link_value="7002",
                source_system="amocrm_snapshot",
                source_ref="contact:7002",
            )
        )
        store.upsert_bot_context_chunk(
            BotContextChunk(
                tenant_id="foton",
                customer_id="customer:other-same-brand",
                chunk_id="chunk-other-same-brand",
                chunk_type="bot_safe_summary",
                text=marker,
                source_system="customer_timeline_bot_safe_summary",
                source_ref="botsafe:other-same-brand:foton",
                event_at=now,
                relevance_tags=("bot_safe", "structured", "foton"),
                allowed_for_bot=True,
                requires_manager_review=False,
                metadata={"brand_context_authorized": True},
            )
        )
    return marker


def _run_journey(
    tmp_path: Path,
    *,
    message: str,
    provider: _HermeticProvider,
    context_builder: Any,
    config: Any,
) -> tuple[Mapping[str, Any], str]:
    key = DraftLoopKey("profile-foton", "chat-1")
    captured: dict[str, Any] = {}
    loop = AmoWappiDraftLoop(
        config=config,
        wappi_client=_SendTrapWappi(
            {"profile-foton": [{"id": "chat-1", "type": "user"}]},
            {("profile-foton", "chat-1"): [_message("m1", text=message)]},
        ),
        amo_client=_capturing_note_client(captured),
        bot_provider=provider,
        context_builder=context_builder,
        now_fn=lambda: datetime.fromtimestamp(1200, tz=timezone.utc),
        code_identity={"code_root": "synthetic", "git_sha": "synthetic"},
    )

    summary = loop.run_once(dry_run=False)

    requests = captured["requests"]
    assert len(requests) == 1
    request = requests[0]
    assert request["method"] == "POST"
    assert request["url"].endswith("/api/integrations/amocrm/leads/5001/notes")
    assert "/messages" not in request["url"]
    return summary, str(request["json_body"]["text"])


def _journey_config(tmp_path: Path) -> Any:
    key = DraftLoopKey("profile-foton", "chat-1")
    return _config(
        tmp_path,
        pairs={
            key: DraftLoopPair(
                key=key,
                lead_id="5001",
                contact_id="7001",
                expected_brand="foton",
                source="wappi_amo_widget",
            )
        },
    )


def test_exact_memory_and_active_brand_fact_reach_real_manager_note(tmp_path: Path, monkeypatch) -> None:
    timeline_db, customer_id = _seed_bot_safe_timeline(tmp_path)
    other_customer_marker = _seed_other_customer_memory(timeline_db, allowed_root=tmp_path)
    _seed_family_rows(timeline_db, customer_id=customer_id)
    snapshot = _write_safe_snapshot(tmp_path)
    config = _journey_config(tmp_path)
    monkeypatch.setenv(BOT_SAFE_CRM_CONTEXT_ENV, "1")
    base_context_builder = draft_runner.build_context_builder(
        snapshot,
        draft_config=config,
        customer_timeline_db=timeline_db,
        customer_timeline_allowed_root=tmp_path,
    )

    def context_builder(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        context = dict(base_context_builder(*args, **kwargs))
        context.update(_NO_AUX_MODEL_FLAGS)
        context["conversation_intent_plan"] = {
            "primary_intent": "format",
            "answer_topics": ["format"],
        }
        return context

    provider = _HermeticProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            topic_id="theme:002_format",
            draft_text=(
                "Учту, что вопрос по физике. "
                "В личном кабинете доступны занятия, материалы и записи онлайн-уроков."
            ),
            metadata={"semantic_frame": _semantic_frame()},
        )
    )

    summary, note_text = _run_journey(
        tmp_path,
        message="Что доступно в личном кабинете именно для нашего ребёнка?",
        provider=provider,
        context_builder=context_builder,
        config=config,
    )

    assert summary["processed"] == 1 and set(summary["message_outcomes"].values()) == {"note_written"}
    assert provider.aux_calls == 0 and provider.external_calls == 0
    assert "класс: 8" in provider.last_prompt and "физика" in provider.last_prompt
    assert (
        "В личном кабинете учебной платформы доступны занятия, материалы и записи онлайн-уроков."
        in provider.last_prompt
    )
    assert "УНПК" not in provider.last_prompt and customer_id not in provider.last_prompt
    assert other_customer_marker not in provider.last_prompt
    assert note_text.startswith("Учту, что вопрос по физике.")
    assert "В личном кабинете доступны занятия, материалы и записи онлайн-уроков." in note_text
    assert DRAFT_NOTE_MARKER in note_text
    assert "не отправлено" in note_text.casefold()
    assert "Бренд: foton" in note_text
    assert "Маршрут: черновик (бот считает безопасным)" in note_text
    assert "Память клиента: подключена." in note_text
    assert "УНПК" not in note_text and customer_id not in note_text
    assert other_customer_marker not in note_text


def test_model_detected_payment_dispute_reaches_manager_note_without_refund_promise(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = _write_safe_snapshot(tmp_path)
    config = _journey_config(tmp_path)
    monkeypatch.setenv(BOT_SAFE_CRM_CONTEXT_ENV, "0")
    base_context_builder = draft_runner.build_context_builder(snapshot, draft_config=config)

    def context_builder(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        context = dict(base_context_builder(*args, **kwargs))
        context.update(_NO_AUX_MODEL_FLAGS)
        context["TELEGRAM_P0_MODEL_LED"] = "1"
        context["TELEGRAM_DIRECT_PATH_MODEL_P0"] = "1"
        return context

    provider = _HermeticProvider(
        SubscriptionDraftResult(
            route="bot_answer_self_for_pilot",
            draft_text="Вернём вам деньги сегодня.",
            risk_level="high",
            metadata={
                "semantic_frame": _semantic_frame(is_p0=True),
                "direct_path_model_p0": {
                    "is_p0": True,
                    "risk_level": "high",
                    "p0_kind": "payment_dispute",
                    "model_reason": "клиент оспаривает изменение оплаченных условий",
                },
            },
        )
    )

    summary, note_text = _run_journey(
        tmp_path,
        message="Где написано, что скидка слетает?",
        provider=provider,
        context_builder=context_builder,
        config=config,
    )

    assert summary["processed"] == 1 and set(summary["message_outcomes"].values()) == {"note_written"}
    assert provider.aux_calls == 0 and provider.external_calls == 0
    assert "Где написано, что скидка слетает?" in provider.last_prompt
    assert "Маршрут: бот передал менеджеру" in note_text
    assert "no_auto_send" in note_text and "manager_approval_required" in note_text
    assert "direct_path_model_p0_payment_dispute" in note_text
    assert "передам" in note_text.casefold() and "менеджер" in note_text.casefold()
    assert DRAFT_NOTE_MARKER in note_text
    assert "не отправлено" in note_text.casefold()
    draft_text, marker, _metadata = note_text.partition(f"\n\n{DRAFT_NOTE_MARKER}")
    assert marker
    assert "p0_promise" not in {
        finding.code
        for finding in verify_output(
            draft_text,
            facts={},
            active_brand="foton",
            client_message="Где написано, что скидка слетает?",
        )
    }
