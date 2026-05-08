from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from mango_mvp.amocrm_runtime.db import Base
from mango_mvp.amocrm_runtime.models import generate_id, utc_now


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    run_type: Mapped[str] = mapped_column(String(64), index=True)
    trigger: Mapped[str] = mapped_column(String(64), default="manual", index=True)
    mode: Mapped[str] = mapped_column(String(32), default="dry_run", index=True)
    status: Mapped[str] = mapped_column(String(32), default="running", index=True)
    actor: Mapped[Optional[str]] = mapped_column(String(120), nullable=True, index=True)
    source: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class AgentActionPolicy(Base):
    __tablename__ = "agent_action_policies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    action_type: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    autonomy_level: Mapped[str] = mapped_column(String(8), default="L3", index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    dry_run_only: Mapped[bool] = mapped_column(Boolean, default=False)
    requires_notification: Mapped[bool] = mapped_column(Boolean, default=False)
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=False)
    description: Mapped[str] = mapped_column(Text, default="")
    conditions: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class AgentAction(Base):
    __tablename__ = "agent_actions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    run_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("agent_runs.id"),
        nullable=True,
        index=True,
    )
    idempotency_key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    action_type: Mapped[str] = mapped_column(String(120), index=True)
    autonomy_level: Mapped[str] = mapped_column(String(8), default="L3", index=True)
    mode: Mapped[str] = mapped_column(String(32), default="dry_run", index=True)
    status: Mapped[str] = mapped_column(String(40), default="proposed", index=True)
    target_system: Mapped[str] = mapped_column(String(64), default="internal", index=True)
    entity_type: Mapped[str] = mapped_column(String(64), default="unknown", index=True)
    entity_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(500), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    rationale: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    preview_payload: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    blockers: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    requires_approval: Mapped[bool] = mapped_column(Boolean, default=False)
    requires_notification: Mapped[bool] = mapped_column(Boolean, default=False)
    seen_count: Mapped[int] = mapped_column(Integer, default=1)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
