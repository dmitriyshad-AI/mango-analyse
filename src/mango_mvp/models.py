from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from mango_mvp.db import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class CallRecord(Base):
    __tablename__ = "call_records"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_file: Mapped[str] = mapped_column(String(1024), unique=True, index=True)
    source_filename: Mapped[str] = mapped_column(String(255), index=True)
    source_call_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)

    audio_codec: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    channels: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    phone: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    manager_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True, index=True)
    direction: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    transcription_status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    resolve_status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    analysis_status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    sync_status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    transcribe_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    resolve_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    analyze_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    sync_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    next_retry_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    dead_letter_stage: Mapped[Optional[str]] = mapped_column(String(16), nullable=True, index=True)

    transcript_manager: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transcript_client: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transcript_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    transcript_variants_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolve_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolve_quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    analysis_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    amocrm_contact_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    amocrm_lead_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)

    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
