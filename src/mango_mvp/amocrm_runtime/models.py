from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from mango_mvp.amocrm_runtime.db import Base


def generate_id() -> str:
    return str(uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AmoIntegrationConnection(Base):
    __tablename__ = "amo_integration_connections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_id)
    integration_mode: Mapped[str] = mapped_column(String(32), default="external")
    status: Mapped[str] = mapped_column(String(50), default="pending")
    state: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    account_base_url: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    account_subdomain: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    client_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    client_secret: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    redirect_uri: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    secrets_uri: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    scopes: Mapped[list[str]] = mapped_column(JSON, default=list)
    access_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    refresh_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    token_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    authorized_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_secrets_payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    last_callback_payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    contact_field_catalog: Mapped[Optional[list[dict]]] = mapped_column(JSON, nullable=True)
    contact_field_catalog_synced_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
