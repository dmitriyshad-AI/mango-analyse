from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from mango_mvp.amocrm_runtime import models as _models  # noqa: F401
from mango_mvp.amocrm_runtime.agent_runtime import (
    ActionProposal,
    build_morning_scan_proposals,
    create_agent_run,
    ensure_default_action_policies,
    execute_action,
    finish_agent_run,
    render_run_digest,
    run_action_preview,
)
from mango_mvp.amocrm_runtime.auth import DEFAULT_DEV_CONTEXT, require_api_key
from mango_mvp.amocrm_runtime.db import Base, get_db
from mango_mvp.amocrm_runtime.main import app as runtime_app
from mango_mvp.amocrm_runtime.agent_models import AgentAction, AgentActionPolicy
from mango_mvp.amocrm_runtime.phone_context import PhoneContext
from mango_mvp.amocrm_runtime.routers.agent import router as agent_router


class AgentRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine, future=True)
        self.session = self.Session()

    def tearDown(self) -> None:
        self.session.close()
        self.engine.dispose()

    def test_default_policies_are_idempotent_and_safe(self) -> None:
        first = ensure_default_action_policies(self.session)
        second = ensure_default_action_policies(self.session)
        self.session.commit()

        self.assertGreaterEqual(first["created"], 10)
        self.assertEqual(second["created"], 0)

        policies = list(self.session.scalars(select(AgentActionPolicy)).all())
        self.assertEqual(len(policies), first["total_defaults"])
        by_type = {policy.action_type: policy for policy in policies}
        self.assertEqual(by_type["close_lead"].autonomy_level, "L4")
        self.assertTrue(by_type["draft_client_message"].requires_approval)
        self.assertTrue(by_type["create_amo_follow_up_task"].requires_notification)

    def test_dry_run_logs_action_without_external_effect(self) -> None:
        ensure_default_action_policies(self.session)
        run = create_agent_run(self.session, run_type="unit", mode="dry_run", actor="tester")
        result = execute_action(
            self.session,
            run=run,
            proposal=ActionProposal(
                action_type="update_contact_ai_context",
                target_system="amocrm",
                entity_type="contact_phone",
                entity_id="+79000000000",
                title="Обновить контакт",
                payload={"phone": "+79000000000", "fields": {"AI-приоритет": "warm"}},
                confidence=0.8,
            ),
            mode="dry_run",
        )
        finish_agent_run(self.session, run)
        self.session.commit()

        self.assertTrue(result.created)
        self.assertEqual(result.action.status, "dry_run")
        self.assertEqual(result.action.autonomy_level, "L1")
        self.assertFalse(result.action.preview_payload["will_call_external_system"])
        self.assertEqual(run.metrics["actions_total"], 1)
        self.assertIn("L1", run.metrics["by_autonomy_level"])

    def test_idempotency_key_prevents_duplicate_actions(self) -> None:
        ensure_default_action_policies(self.session)
        run = create_agent_run(self.session, run_type="unit", mode="dry_run")
        proposal = ActionProposal(
            action_type="create_amo_follow_up_task",
            target_system="amocrm",
            entity_type="contact_phone",
            entity_id="+79000000001",
            payload={"phone": "+79000000001", "text": "Перезвонить"},
        )

        first = execute_action(self.session, run=run, proposal=proposal, mode="dry_run")
        second = execute_action(self.session, run=run, proposal=proposal, mode="dry_run")
        self.session.commit()

        actions = list(self.session.scalars(select(AgentAction)).all())
        self.assertEqual(len(actions), 1)
        self.assertTrue(first.created)
        self.assertFalse(second.created)
        self.assertTrue(second.duplicate)
        self.assertEqual(second.action.id, first.action.id)
        self.assertEqual(second.action.seen_count, 2)

    def test_l3_live_actions_go_to_approval_queue(self) -> None:
        ensure_default_action_policies(self.session)
        run = create_agent_run(self.session, run_type="unit", mode="live")
        result = execute_action(
            self.session,
            run=run,
            proposal=ActionProposal(
                action_type="draft_client_message",
                target_system="telegram",
                entity_type="contact_phone",
                entity_id="+79000000002",
                payload={"draft": "Здравствуйте"},
            ),
            mode="live",
        )
        self.session.commit()

        self.assertEqual(result.action.status, "queued_for_approval")
        self.assertTrue(result.action.requires_approval)
        self.assertEqual(result.action.autonomy_level, "L3")

    def test_l4_live_actions_are_blocked_by_policy(self) -> None:
        ensure_default_action_policies(self.session)
        run = create_agent_run(self.session, run_type="unit", mode="live")
        result = execute_action(
            self.session,
            run=run,
            proposal=ActionProposal(
                action_type="close_lead",
                target_system="amocrm",
                entity_type="lead",
                entity_id="123",
                payload={"status": "lost"},
            ),
            mode="live",
        )
        self.session.commit()

        self.assertEqual(result.action.status, "blocked_by_policy")
        self.assertIn("level_l4_forbidden", result.action.blockers)

    def test_unknown_live_action_defaults_to_l3_approval(self) -> None:
        run = create_agent_run(self.session, run_type="unit", mode="live")
        result = execute_action(
            self.session,
            run=run,
            proposal=ActionProposal(
                action_type="new_future_action",
                target_system="internal",
                entity_type="lead",
                entity_id="456",
            ),
            mode="live",
        )
        self.session.commit()

        self.assertEqual(result.action.status, "queued_for_approval")
        self.assertEqual(result.action.autonomy_level, "L3")
        self.assertTrue(result.action.requires_approval)

    def test_run_action_preview_returns_digest(self) -> None:
        result = run_action_preview(
            self.session,
            proposals=[
                ActionProposal(
                    action_type="update_contact_ai_context",
                    target_system="amocrm",
                    entity_type="contact_phone",
                    entity_id="+79000000003",
                    title="Обновить AI-контекст",
                    payload={"phone": "+79000000003"},
                )
            ],
            run_type="unit_preview",
            actor="tester",
        )
        self.session.commit()

        self.assertEqual(result["run"]["status"], "completed")
        self.assertIn("Агентский запуск", result["digest"])
        self.assertEqual(result["run"]["metrics"]["created_actions"], 1)

    def test_morning_scan_builds_contact_task_and_rop_proposals(self) -> None:
        ctx = PhoneContext(
            phone="+79000000004",
            source_dir="/tmp/export",
            contact_row={"Краткое резюме последнего свежего звонка": "Клиент готов обсуждать оплату."},
            call_rows=[],
            call_ids=[],
            first_call_at="2026-04-20 10:00:00",
            last_call_at="2026-04-28 12:00:00",
            manager_history=["Иванов"],
            interest_summary="ЕГЭ математика",
            objections_summary="Цена",
            current_sales_temperature="Горячий",
            recommended_next_step="Позвонить и обсудить рассрочку",
            follow_up_due_at="2026-04-29",
            history_summary="Клиент заинтересован, просил рассрочку.",
            chronology="28.04 звонок",
            tallanto_id="T-1",
            tallanto_match_status="matched",
        )

        with patch("mango_mvp.amocrm_runtime.agent_runtime.get_all_known_phones", return_value=[ctx.phone]), patch(
            "mango_mvp.amocrm_runtime.agent_runtime.get_phone_context", return_value=ctx
        ):
            proposals = build_morning_scan_proposals(today=date(2026, 4, 29), limit=10)

        self.assertEqual(
            [proposal.action_type for proposal in proposals],
            ["update_contact_ai_context", "create_amo_follow_up_task", "notify_rop_hot_lead"],
        )
        self.assertEqual(proposals[1].payload["due_at"], "2026-04-29")


class AgentRuntimeApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite://",
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine, future=True)

        def override_db():
            session = self.Session()
            try:
                yield session
            finally:
                session.close()

        self.app = FastAPI()
        self.app.include_router(agent_router, prefix="/api")
        self.app.dependency_overrides[get_db] = override_db
        self.app.dependency_overrides[require_api_key] = lambda: DEFAULT_DEV_CONTEXT
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()
        self.engine.dispose()

    def test_actions_preview_endpoint_records_dry_run(self) -> None:
        response = self.client.post(
            "/api/agent/actions/preview",
            json={
                "run_type": "api_unit",
                "actions": [
                    {
                        "action_type": "update_contact_ai_context",
                        "target_system": "amocrm",
                        "entity_type": "contact_phone",
                        "entity_id": "+79000000005",
                        "title": "API preview",
                        "payload": {"phone": "+79000000005"},
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["run"]["run_type"], "api_unit")
        self.assertEqual(payload["actions"][0]["status"], "dry_run")

        run_id = payload["run"]["id"]
        detail = self.client.get(f"/api/agent/runs/{run_id}")
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["metrics"]["actions_total"], 1)

    def test_main_runtime_app_does_not_mount_agent_router_by_default(self) -> None:
        paths = {getattr(route, "path", "") for route in runtime_app.routes}
        self.assertNotIn("/api/agent/actions/preview", paths)


if __name__ == "__main__":
    unittest.main()
