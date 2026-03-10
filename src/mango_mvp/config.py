from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    database_url: str
    openai_api_key: Optional[str]
    transcribe_provider: str
    dual_transcribe_enabled: bool
    secondary_transcribe_provider: Optional[str]
    dual_merge_provider: str
    openai_merge_model: str
    codex_merge_model: str
    codex_cli_command: str
    codex_cli_timeout_sec: int
    dual_merge_similarity_threshold: float
    analyze_provider: str
    openai_transcribe_model: str
    mlx_whisper_model: str
    mlx_condition_on_previous_text: bool
    mlx_word_timestamps: bool
    gigaam_model: str
    gigaam_device: str
    gigaam_segment_sec: int
    openai_analysis_model: str
    analyze_ollama_num_predict: int
    ollama_base_url: str
    ollama_model: str
    ollama_think: str
    ollama_temperature: float
    transcribe_language: Optional[str]
    transcript_export_dir: Optional[str]
    split_stereo_channels: bool
    stereo_overlap_similarity_threshold: float
    stereo_overlap_min_chars: int
    mono_role_assignment_mode: str
    mono_role_assignment_min_confidence: float
    mono_role_assignment_llm_threshold: float
    openai_role_assign_model: str
    max_workers: int
    transcribe_max_attempts: int
    resolve_max_attempts: int
    analyze_max_attempts: int
    sync_max_attempts: int
    resolve_min_duration_sec: int
    resolve_llm_trigger_score: int
    resolve_accept_score: int
    resolve_llm_provider: str
    resolve_llm_for_risky: bool
    resolve_rescue_provider: Optional[str]
    resolve_rescue_dual_enabled: bool
    resolve_postfilter_same_ts: bool
    resolve_risky_same_ts_threshold: int
    resolve_aggressive_rescue_for_risky: bool
    retry_base_delay_sec: int
    worker_poll_sec: int
    worker_max_idle_cycles: int
    amocrm_base_url: Optional[str]
    amocrm_access_token: Optional[str]
    amocrm_refresh_token: Optional[str]
    amocrm_client_id: Optional[str]
    amocrm_client_secret: Optional[str]
    amocrm_redirect_uri: Optional[str]
    amocrm_token_cache_path: str
    amocrm_interests_field_id: Optional[int]
    amocrm_student_grade_field_id: Optional[int]
    amocrm_target_product_field_id: Optional[int]
    amocrm_personal_offer_field_id: Optional[int]
    amocrm_budget_field_id: Optional[int]
    amocrm_timeline_field_id: Optional[int]
    amocrm_next_step_field_id: Optional[int]
    amocrm_followup_score_field_id: Optional[int]
    amocrm_task_type_id: Optional[int]
    amocrm_task_responsible_user_id: Optional[int]
    sync_dry_run: bool
    follow_up_task_threshold: int


def _optional_int(raw: Optional[str]) -> Optional[int]:
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    return Settings(
        database_url=os.getenv("DATABASE_URL", "sqlite:///mango_mvp.db"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        transcribe_provider=os.getenv("TRANSCRIBE_PROVIDER", "mock").strip().lower(),
        dual_transcribe_enabled=_bool_env("DUAL_TRANSCRIBE_ENABLED", False),
        secondary_transcribe_provider=(
            os.getenv("SECONDARY_TRANSCRIBE_PROVIDER", "").strip().lower() or None
        ),
        dual_merge_provider=os.getenv("DUAL_MERGE_PROVIDER", "codex_cli").strip().lower(),
        openai_merge_model=os.getenv("OPENAI_MERGE_MODEL", "gpt-4o-mini").strip(),
        codex_merge_model=os.getenv("CODEX_MERGE_MODEL", "gpt-5.4").strip(),
        codex_cli_command=os.getenv("CODEX_CLI_COMMAND", "codex").strip(),
        codex_cli_timeout_sec=_int_env("CODEX_CLI_TIMEOUT_SEC", 120),
        dual_merge_similarity_threshold=_float_env("DUAL_MERGE_SIMILARITY_THRESHOLD", 0.985),
        analyze_provider=os.getenv("ANALYZE_PROVIDER", "codex_cli").strip().lower(),
        openai_transcribe_model=os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe"),
        mlx_whisper_model=os.getenv(
            "MLX_WHISPER_MODEL", "mlx-community/whisper-large-v3-mlx"
        ),
        mlx_condition_on_previous_text=_bool_env("MLX_CONDITION_ON_PREVIOUS_TEXT", False),
        mlx_word_timestamps=_bool_env("MLX_WORD_TIMESTAMPS", True),
        gigaam_model=os.getenv("GIGAAM_MODEL", "v2_rnnt").strip(),
        gigaam_device=os.getenv("GIGAAM_DEVICE", "cpu").strip().lower(),
        gigaam_segment_sec=_int_env("GIGAAM_SEGMENT_SEC", 20),
        openai_analysis_model=os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-4o-mini"),
        analyze_ollama_num_predict=_int_env("ANALYZE_OLLAMA_NUM_PREDICT", 500),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "gpt-oss:20b").strip(),
        ollama_think=os.getenv("OLLAMA_THINK", "medium").strip().lower(),
        ollama_temperature=_float_env("OLLAMA_TEMPERATURE", 0.0),
        transcribe_language=os.getenv("TRANSCRIBE_LANGUAGE"),
        transcript_export_dir=(os.getenv("TRANSCRIPT_EXPORT_DIR", "transcripts").strip() or None),
        split_stereo_channels=_bool_env("SPLIT_STEREO_CHANNELS", True),
        stereo_overlap_similarity_threshold=_float_env(
            "STEREO_OVERLAP_SIMILARITY_THRESHOLD", 0.97
        ),
        stereo_overlap_min_chars=_int_env("STEREO_OVERLAP_MIN_CHARS", 80),
        mono_role_assignment_mode=os.getenv("MONO_ROLE_ASSIGNMENT_MODE", "off").strip().lower(),
        mono_role_assignment_min_confidence=_float_env(
            "MONO_ROLE_ASSIGNMENT_MIN_CONFIDENCE", 0.62
        ),
        mono_role_assignment_llm_threshold=_float_env(
            "MONO_ROLE_ASSIGNMENT_LLM_THRESHOLD", 0.72
        ),
        openai_role_assign_model=os.getenv("OPENAI_ROLE_ASSIGN_MODEL", "gpt-4o-mini").strip(),
        max_workers=_int_env("MAX_WORKERS", 4),
        transcribe_max_attempts=_int_env("TRANSCRIBE_MAX_ATTEMPTS", 3),
        resolve_max_attempts=_int_env("RESOLVE_MAX_ATTEMPTS", 2),
        analyze_max_attempts=_int_env("ANALYZE_MAX_ATTEMPTS", 3),
        sync_max_attempts=_int_env("SYNC_MAX_ATTEMPTS", 3),
        resolve_min_duration_sec=_int_env("RESOLVE_MIN_DURATION_SEC", 30),
        resolve_llm_trigger_score=_int_env("RESOLVE_LLM_TRIGGER_SCORE", 75),
        resolve_accept_score=_int_env("RESOLVE_ACCEPT_SCORE", 75),
        resolve_llm_provider=os.getenv("RESOLVE_LLM_PROVIDER", "codex_cli").strip().lower(),
        resolve_llm_for_risky=_bool_env("RESOLVE_LLM_FOR_RISKY", False),
        resolve_rescue_provider=(os.getenv("RESOLVE_RESCUE_PROVIDER", "").strip().lower() or None),
        resolve_rescue_dual_enabled=_bool_env("RESOLVE_RESCUE_DUAL_ENABLED", False),
        resolve_postfilter_same_ts=_bool_env("RESOLVE_POSTFILTER_SAME_TS", True),
        resolve_risky_same_ts_threshold=_int_env("RESOLVE_RISKY_SAME_TS_THRESHOLD", 2),
        resolve_aggressive_rescue_for_risky=_bool_env(
            "RESOLVE_AGGRESSIVE_RESCUE_FOR_RISKY", True
        ),
        retry_base_delay_sec=_int_env("RETRY_BASE_DELAY_SEC", 30),
        worker_poll_sec=_int_env("WORKER_POLL_SEC", 10),
        worker_max_idle_cycles=_int_env("WORKER_MAX_IDLE_CYCLES", 30),
        amocrm_base_url=os.getenv("AMOCRM_BASE_URL"),
        amocrm_access_token=os.getenv("AMOCRM_ACCESS_TOKEN"),
        amocrm_refresh_token=os.getenv("AMOCRM_REFRESH_TOKEN"),
        amocrm_client_id=os.getenv("AMOCRM_CLIENT_ID"),
        amocrm_client_secret=os.getenv("AMOCRM_CLIENT_SECRET"),
        amocrm_redirect_uri=os.getenv("AMOCRM_REDIRECT_URI"),
        amocrm_token_cache_path=os.getenv("AMOCRM_TOKEN_CACHE_PATH", ".amocrm_tokens.json"),
        amocrm_interests_field_id=_optional_int(os.getenv("AMOCRM_INTERESTS_FIELD_ID")),
        amocrm_student_grade_field_id=_optional_int(
            os.getenv("AMOCRM_STUDENT_GRADE_FIELD_ID")
        ),
        amocrm_target_product_field_id=_optional_int(
            os.getenv("AMOCRM_TARGET_PRODUCT_FIELD_ID")
        ),
        amocrm_personal_offer_field_id=_optional_int(
            os.getenv("AMOCRM_PERSONAL_OFFER_FIELD_ID")
        ),
        amocrm_budget_field_id=_optional_int(os.getenv("AMOCRM_BUDGET_FIELD_ID")),
        amocrm_timeline_field_id=_optional_int(os.getenv("AMOCRM_TIMELINE_FIELD_ID")),
        amocrm_next_step_field_id=_optional_int(os.getenv("AMOCRM_NEXT_STEP_FIELD_ID")),
        amocrm_followup_score_field_id=_optional_int(
            os.getenv("AMOCRM_FOLLOWUP_SCORE_FIELD_ID")
        ),
        amocrm_task_type_id=_optional_int(os.getenv("AMOCRM_TASK_TYPE_ID")),
        amocrm_task_responsible_user_id=_optional_int(
            os.getenv("AMOCRM_TASK_RESPONSIBLE_USER_ID")
        ),
        sync_dry_run=_bool_env("SYNC_DRY_RUN", True),
        follow_up_task_threshold=_int_env("FOLLOW_UP_TASK_THRESHOLD", 70),
    )
