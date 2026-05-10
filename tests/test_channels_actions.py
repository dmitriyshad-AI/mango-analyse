from __future__ import annotations

from datetime import datetime, timezone

import pytest

from mango_mvp.amocrm_runtime.agent_runtime import ActionProposal
from mango_mvp.channels import (
    ACTION_CREATE_FOLLOW_UP_TASK,
    ACTION_DRAFT_CLIENT_MESSAGE,
    ACTION_HANDOFF_TO_MANAGER,
    ACTION_MARK_MANUAL_REVIEW,
    ACTION_NOTIFY_ROP_HOT_LEAD,
    ACTION_REQUEST_CRM_CONTEXT,
    ChannelMessage,
    ChannelSession,
    build_channel_recommended_actions,
    channel_action_policy,
    default_channel_action_policy_map,
    recommended_action_to_agent_proposal,
    recommended_actions_to_agent_proposals,
)


NOW = datetime(2026, 5, 9, 14, 0, tzinfo=timezone.utc)


def message(text: str) -> ChannelMessage:
    return ChannelMessage(
        channel="site_chat",
        channel_message_id="m1",
        channel_thread_id="thread-1",
        channel_user_id="visitor-1",
        direction="inbound",
        text=text,
        received_at=NOW,
    )


def test_default_channel_action_policies_are_safe() -> None:
    policies = default_channel_action_policy_map()

    assert policies[ACTION_REQUEST_CRM_CONTEXT]["autonomy_level"] == "L1"
    assert policies[ACTION_CREATE_FOLLOW_UP_TASK]["autonomy_level"] == "L2"
    assert policies[ACTION_DRAFT_CLIENT_MESSAGE]["autonomy_level"] == "L3"
    assert policies[ACTION_HANDOFF_TO_MANAGER]["requires_approval"] is True
    assert all(policy["live_execution_allowed"] is False for policy in policies.values())


def test_unknown_channel_action_defaults_to_l3_approval() -> None:
    policy = channel_action_policy("new_future_action")

    assert policy.autonomy_level == "L3"
    assert policy.requires_approval is True
    assert policy.live_execution_allowed is False


def test_build_channel_recommended_actions_baseline_includes_draft_and_context_request() -> None:
    msg = message("Здравствуйте, расскажите про подготовку к ЕГЭ")
    session = ChannelSession.from_message(msg)

    actions = build_channel_recommended_actions(
        message=msg,
        session=session,
        draft_id="draft-1",
        draft_text="Здравствуйте! Уточним детали.",
    )

    assert [action.action_type for action in actions] == [
        ACTION_DRAFT_CLIENT_MESSAGE,
        ACTION_REQUEST_CRM_CONTEXT,
    ]
    assert actions[0].requires_approval is True
    assert actions[1].requires_approval is False
    assert actions[1].payload["live_send_enabled"] is False


def test_build_channel_recommended_actions_detects_handoff_followup_commercial_and_hot_signals() -> None:
    msg = message("Срочно позвоните, хочу оплатить, но нужна скидка и договор")
    session = ChannelSession.from_message(msg)

    actions = build_channel_recommended_actions(
        message=msg,
        session=session,
        draft_id="draft-1",
        draft_text="Здравствуйте! Передадим менеджеру.",
        context={"priority": "hot", "follow_up_due_at": "2026-05-10"},
    )
    by_type = {action.action_type: action for action in actions}

    assert ACTION_DRAFT_CLIENT_MESSAGE in by_type
    assert ACTION_REQUEST_CRM_CONTEXT in by_type
    assert ACTION_HANDOFF_TO_MANAGER in by_type
    assert ACTION_CREATE_FOLLOW_UP_TASK in by_type
    assert ACTION_MARK_MANUAL_REVIEW in by_type
    assert ACTION_NOTIFY_ROP_HOT_LEAD in by_type
    assert by_type[ACTION_CREATE_FOLLOW_UP_TASK].payload["due_at"] == "2026-05-10"
    assert by_type[ACTION_MARK_MANUAL_REVIEW].payload["policy"]["requires_approval"] is True


def test_recommended_action_to_agent_proposal_preserves_idempotency_and_policy() -> None:
    msg = message("Сколько стоит курс?")
    session = ChannelSession.from_message(msg)
    action = build_channel_recommended_actions(
        message=msg,
        session=session,
        draft_id="draft-1",
        draft_text="Здравствуйте! Уточним детали.",
    )[0]

    proposal = recommended_action_to_agent_proposal(action)

    assert isinstance(proposal, ActionProposal)
    assert proposal.action_type == action.action_type
    assert proposal.target_system == action.target_system
    assert proposal.entity_id == action.entity_id
    assert proposal.idempotency_key == action.idempotency_key
    assert proposal.payload["channel_action_policy"]["autonomy_level"] == "L3"
    assert proposal.payload["channel_action_policy"]["live_execution_allowed"] is False


def test_recommended_actions_to_agent_proposals_maps_all_actions() -> None:
    msg = message("Позвоните менеджеру завтра")
    session = ChannelSession.from_message(msg)
    actions = build_channel_recommended_actions(
        message=msg,
        session=session,
        draft_id="draft-1",
        draft_text="Здравствуйте! Уточним детали.",
    )

    proposals = recommended_actions_to_agent_proposals(actions)

    assert len(proposals) == len(actions)
    assert tuple(proposal.action_type for proposal in proposals) == tuple(action.action_type for action in actions)


def test_channel_action_policy_refuses_live_execution_allowed() -> None:
    from mango_mvp.channels.actions import ChannelActionPolicy

    with pytest.raises(ValueError, match="must not allow live execution"):
        ChannelActionPolicy(
            action_type="unsafe_live_send",
            autonomy_level="L4",
            requires_approval=False,
            live_execution_allowed=True,
        )
