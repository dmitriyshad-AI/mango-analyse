import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


def _normalize_database_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


@dataclass(frozen=True)
class Settings:
    api_key: Optional[str]
    api_keys: Tuple[str, ...]
    stream_token_ttl_seconds: int
    stream_token_secret: Optional[str]
    database_url: str
    db_pool_size: int
    db_max_overflow: int
    db_pool_timeout_seconds: int
    db_pool_recycle_seconds: int
    redis_url: str
    api_host: str
    api_port: int
    runtime_root: str
    source_workspace_root: str
    task_container_driver: str
    task_container_image: str
    task_container_image_build_context: str
    task_container_image_dockerfile: str
    task_container_image_auto_build: bool
    task_container_workdir: str
    task_container_network: str
    task_container_name_prefix: str
    task_container_codex_command: str
    task_container_codex_sandbox: str
    task_container_codex_home_host_path: Optional[str]
    task_container_codex_home_container_path: str
    task_container_codex_home_runtime_path: str
    task_container_codex_home_copy_allowlist: Tuple[str, ...]
    task_container_env_passthrough: Tuple[str, ...]
    codex_cli_path: str
    codex_worker_mode: str
    director_auto_run_enabled: bool
    director_auto_max_attempts: int
    director_heartbeat_enabled: bool
    director_heartbeat_poll_seconds: int
    director_heartbeat_max_dispatch_per_tick: int
    director_stale_run_grace_seconds: int
    director_stale_run_auto_retry_window_seconds: int
    codex_model: Optional[str]
    codex_execution_timeout_seconds: int
    crm_tallanto_mode: str
    crm_tallanto_base_url: Optional[str]
    crm_tallanto_api_token: Optional[str]
    crm_tallanto_student_path: str
    crm_amo_mode: str
    crm_amo_base_url: Optional[str]
    crm_amo_api_token: Optional[str]
    crm_amo_http_timeout_seconds: int
    crm_amo_upsert_path: str
    crm_amo_contact_field_map: Optional[str]
    crm_amo_lead_field_map: Optional[str]
    crm_amo_target_pipeline_ids: Tuple[str, ...]
    crm_amo_recent_closed_days: int
    crm_amo_deal_queue_dir: str
    crm_amo_oauth_redirect_uri: Optional[str]
    crm_amo_oauth_secrets_uri: Optional[str]
    crm_amo_oauth_scopes: Tuple[str, ...]
    crm_amo_oauth_name: str
    crm_amo_oauth_description: str
    crm_amo_oauth_logo_url: Optional[str]
    crm_amo_oauth_account_base_url: Optional[str]
    crm_analysis_mode: str
    crm_analysis_provider: str
    crm_analysis_model: Optional[str]
    crm_analysis_reasoning_effort: str
    crm_analysis_timeout_seconds: int
    crm_analysis_llm_cache_enabled: bool
    crm_analysis_llm_cache_dir: str
    crm_analysis_max_transcript_calls: int
    crm_analysis_transcript_excerpt_chars: int
    crm_analysis_write_ai_office_field: bool
    crm_amo_deal_writeback_safe_mode: bool
    agent_runtime_enabled: bool


def _normalize_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve())


def _default_source_workspace_root() -> str:
    repo_candidate = Path(__file__).resolve().parents[3]
    if any((repo_candidate / marker).exists() for marker in ("package.json", "docker-compose.yml", ".git")):
        return str(repo_candidate)
    return str(Path(__file__).resolve().parents[1])


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_csv(value: Optional[str]) -> Tuple[str, ...]:
    if not value:
        return tuple()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def get_settings() -> Settings:
    source_workspace_root = os.getenv("SOURCE_WORKSPACE_ROOT")
    if not source_workspace_root:
        source_workspace_root = _default_source_workspace_root()
    else:
        normalized_source_root = _normalize_path(source_workspace_root)
        if Path(normalized_source_root).exists():
            source_workspace_root = normalized_source_root
        else:
            source_workspace_root = _default_source_workspace_root()

    return Settings(
        api_key=os.getenv("AI_OFFICE_API_KEY") or None,
        api_keys=_parse_csv(os.getenv("AI_OFFICE_API_KEYS")),
        stream_token_ttl_seconds=int(os.getenv("AI_OFFICE_STREAM_TOKEN_TTL_SECONDS", "120")),
        stream_token_secret=os.getenv("AI_OFFICE_STREAM_TOKEN_SECRET") or None,
        database_url=_normalize_database_url(
            os.getenv("DATABASE_URL", "sqlite:///./ai_office.db")
        ),
        db_pool_size=max(1, int(os.getenv("DB_POOL_SIZE", "20"))),
        db_max_overflow=max(0, int(os.getenv("DB_MAX_OVERFLOW", "20"))),
        db_pool_timeout_seconds=max(1, int(os.getenv("DB_POOL_TIMEOUT_SECONDS", "30"))),
        db_pool_recycle_seconds=max(30, int(os.getenv("DB_POOL_RECYCLE_SECONDS", "1800"))),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        api_host=os.getenv("API_HOST", "0.0.0.0"),
        api_port=int(os.getenv("API_PORT", "8000")),
        runtime_root=_normalize_path(os.getenv("RUNTIME_ROOT", "./runtime")),
        source_workspace_root=str(Path(source_workspace_root).resolve()),
        task_container_driver=os.getenv("TASK_CONTAINER_DRIVER", "process"),
        task_container_image=os.getenv("TASK_CONTAINER_IMAGE", "ai-office-task-runner:latest"),
        task_container_image_build_context=_normalize_path(
            os.getenv("TASK_CONTAINER_IMAGE_BUILD_CONTEXT", source_workspace_root)
        ),
        task_container_image_dockerfile=os.getenv(
            "TASK_CONTAINER_IMAGE_DOCKERFILE",
            "apps/task-runner/Dockerfile",
        ),
        task_container_image_auto_build=_parse_bool(
            os.getenv("TASK_CONTAINER_IMAGE_AUTO_BUILD"),
            True,
        ),
        task_container_workdir=os.getenv("TASK_CONTAINER_WORKDIR", "/task"),
        task_container_network=os.getenv("TASK_CONTAINER_NETWORK", "none"),
        task_container_name_prefix=os.getenv("TASK_CONTAINER_NAME_PREFIX", "ai-office-task"),
        task_container_codex_command=os.getenv("TASK_CONTAINER_CODEX_COMMAND", "codex"),
        task_container_codex_sandbox=os.getenv(
            "TASK_CONTAINER_CODEX_SANDBOX",
            "workspace-write",
        ),
        task_container_codex_home_host_path=(
            os.getenv("TASK_CONTAINER_CODEX_HOME_HOST_PATH") or None
        ),
        task_container_codex_home_container_path=os.getenv(
            "TASK_CONTAINER_CODEX_HOME_CONTAINER_PATH",
            "/task/.codex-source",
        ),
        task_container_codex_home_runtime_path=os.getenv(
            "TASK_CONTAINER_CODEX_HOME_RUNTIME_PATH",
            "/task/.codex",
        ),
        task_container_codex_home_copy_allowlist=_parse_csv(
            os.getenv(
                "TASK_CONTAINER_CODEX_HOME_COPY_ALLOWLIST",
                "auth.json,config.toml,AGENTS.md,rules,skills,models_cache.json",
            )
        ),
        task_container_env_passthrough=_parse_csv(
            os.getenv(
                "TASK_CONTAINER_ENV_PASSTHROUGH",
                "",
            )
        ),
        codex_cli_path=os.getenv("CODEX_CLI_PATH", shutil.which("codex") or "codex"),
        codex_worker_mode=os.getenv("CODEX_WORKER_MODE", "mock"),
        director_auto_run_enabled=_parse_bool(
            os.getenv("DIRECTOR_AUTO_RUN_ENABLED"),
            True,
        ),
        director_auto_max_attempts=max(
            1,
            int(os.getenv("DIRECTOR_AUTO_MAX_ATTEMPTS", "3")),
        ),
        director_heartbeat_enabled=_parse_bool(
            os.getenv("DIRECTOR_HEARTBEAT_ENABLED"),
            True,
        ),
        director_heartbeat_poll_seconds=max(
            1,
            int(os.getenv("DIRECTOR_HEARTBEAT_POLL_SECONDS", "5")),
        ),
        director_heartbeat_max_dispatch_per_tick=max(
            1,
            int(os.getenv("DIRECTOR_HEARTBEAT_MAX_DISPATCH_PER_TICK", "3")),
        ),
        director_stale_run_grace_seconds=max(
            10,
            int(os.getenv("DIRECTOR_STALE_RUN_GRACE_SECONDS", "120")),
        ),
        director_stale_run_auto_retry_window_seconds=max(
            0,
            int(os.getenv("DIRECTOR_STALE_RUN_AUTO_RETRY_WINDOW_SECONDS", "300")),
        ),
        codex_model=os.getenv("CODEX_MODEL") or None,
        codex_execution_timeout_seconds=int(
            os.getenv("CODEX_EXECUTION_TIMEOUT_SECONDS", "900")
        ),
        crm_tallanto_mode=os.getenv("CRM_TALLANTO_MODE", "mock"),
        crm_tallanto_base_url=os.getenv("CRM_TALLANTO_BASE_URL") or None,
        crm_tallanto_api_token=os.getenv("CRM_TALLANTO_API_TOKEN") or None,
        crm_tallanto_student_path=os.getenv(
            "CRM_TALLANTO_STUDENT_PATH",
            "/service/api/rest.php",
        ),
        crm_amo_mode=os.getenv("CRM_AMO_MODE", "mock"),
        crm_amo_base_url=os.getenv("CRM_AMO_BASE_URL") or None,
        crm_amo_api_token=os.getenv("CRM_AMO_API_TOKEN") or None,
        crm_amo_http_timeout_seconds=max(
            5,
            int(os.getenv("CRM_AMO_HTTP_TIMEOUT_SECONDS", "15")),
        ),
        crm_amo_upsert_path=os.getenv(
            "CRM_AMO_UPSERT_PATH",
            "/contacts/{entity_id}",
        ),
        crm_amo_contact_field_map=os.getenv("CRM_AMO_CONTACT_FIELD_MAP") or None,
        crm_amo_lead_field_map=os.getenv("CRM_AMO_LEAD_FIELD_MAP") or None,
        crm_amo_target_pipeline_ids=_parse_csv(os.getenv("CRM_AMO_TARGET_PIPELINE_IDS")),
        crm_amo_recent_closed_days=max(
            1,
            int(os.getenv("CRM_AMO_RECENT_CLOSED_DAYS", "90")),
        ),
        crm_amo_deal_queue_dir=_normalize_path(
            os.getenv(
                "CRM_AMO_DEAL_QUEUE_DIR",
                str(Path(source_workspace_root) / "stable_runtime" / "amocrm_runtime" / "deal_analysis"),
            )
        ),
        crm_amo_oauth_redirect_uri=os.getenv("CRM_AMO_OAUTH_REDIRECT_URI") or None,
        crm_amo_oauth_secrets_uri=os.getenv("CRM_AMO_OAUTH_SECRETS_URI") or None,
        crm_amo_oauth_scopes=_parse_csv(os.getenv("CRM_AMO_OAUTH_SCOPES", "crm")),
        crm_amo_oauth_name=os.getenv("CRM_AMO_OAUTH_NAME", "AI Office"),
        crm_amo_oauth_description=os.getenv(
            "CRM_AMO_OAUTH_DESCRIPTION",
            "Интеграция AI Office для безопасной записи данных в amoCRM.",
        ),
        crm_amo_oauth_logo_url=os.getenv("CRM_AMO_OAUTH_LOGO_URL") or None,
        crm_amo_oauth_account_base_url=os.getenv("CRM_AMO_OAUTH_ACCOUNT_BASE_URL") or None,
        crm_analysis_mode=os.getenv("CRM_ANALYSIS_MODE", "heuristic"),
        crm_analysis_provider=os.getenv("CRM_ANALYSIS_PROVIDER", "codex_cli"),
        crm_analysis_model=(
            os.getenv("CRM_ANALYSIS_MODEL")
            or os.getenv("CODEX_MODEL")
            or "gpt-5.4"
        ),
        crm_analysis_reasoning_effort=os.getenv("CRM_ANALYSIS_REASONING_EFFORT", "medium"),
        crm_analysis_timeout_seconds=max(
            15,
            int(os.getenv("CRM_ANALYSIS_TIMEOUT_SECONDS", "300")),
        ),
        crm_analysis_llm_cache_enabled=_parse_bool(
            os.getenv("CRM_ANALYSIS_LLM_CACHE_ENABLED"),
            True,
        ),
        crm_analysis_llm_cache_dir=_normalize_path(
            os.getenv(
                "CRM_ANALYSIS_LLM_CACHE_DIR",
                str(Path(source_workspace_root) / "stable_runtime" / "amocrm_runtime" / "llm_cache"),
            )
        ),
        crm_analysis_max_transcript_calls=max(
            1,
            int(os.getenv("CRM_ANALYSIS_MAX_TRANSCRIPT_CALLS", "8")),
        ),
        crm_analysis_transcript_excerpt_chars=max(
            500,
            int(os.getenv("CRM_ANALYSIS_TRANSCRIPT_EXCERPT_CHARS", "2200")),
        ),
        crm_analysis_write_ai_office_field=_parse_bool(
            os.getenv("CRM_ANALYSIS_WRITE_AI_OFFICE_FIELD"),
            False,
        ),
        crm_amo_deal_writeback_safe_mode=_parse_bool(
            os.getenv("CRM_AMO_DEAL_WRITEBACK_SAFE_MODE"),
            True,
        ),
        agent_runtime_enabled=_parse_bool(
            os.getenv("AI_OFFICE_AGENT_RUNTIME_ENABLED"),
            False,
        ),
    )
