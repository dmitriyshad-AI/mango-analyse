"""Quality gates and audit helpers for transcript-derived layers."""

from mango_mvp.quality.non_conversation import (
    LABEL_CONTENTFUL_LOW_RISK,
    LABEL_CONTENTFUL_PROTECTED_LIVE_DIALOGUE,
    LABEL_MANUAL_REVIEW_BORDERLINE_LIVE_CONTEXT,
    LABEL_MANUAL_REVIEW_PROBABLE_NO_LIVE,
    LABEL_NON_CONVERSATION_HIGH_CONFIDENCE,
    NonConversationSignals,
    blocks_email_from_voice_mail,
    blocks_system_next_step,
    classify_transcript_quality,
    detect_non_conversation_signals,
)

__all__ = [
    "LABEL_CONTENTFUL_LOW_RISK",
    "LABEL_CONTENTFUL_PROTECTED_LIVE_DIALOGUE",
    "LABEL_MANUAL_REVIEW_BORDERLINE_LIVE_CONTEXT",
    "LABEL_MANUAL_REVIEW_PROBABLE_NO_LIVE",
    "LABEL_NON_CONVERSATION_HIGH_CONFIDENCE",
    "NonConversationSignals",
    "blocks_email_from_voice_mail",
    "blocks_system_next_step",
    "classify_transcript_quality",
    "detect_non_conversation_signals",
]
