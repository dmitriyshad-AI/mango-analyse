from __future__ import annotations

import csv
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from openai import OpenAI
from sqlalchemy import func, select, text, update as sa_update
from sqlalchemy.orm import Session

from mango_mvp.clients.ollama import OllamaClient
from mango_mvp.config import Settings
from mango_mvp.models import CallRecord
from mango_mvp.services.controlled_call_scope import (
    call_artifact_directory,
    read_call_artifact_text,
    require_unique_controlled_call,
)
from mango_mvp.services.dialogue_contract import (
    DialogueContractError,
    PROVIDER_EVIDENCE_FIELD,
    label_is_neutral as dialogue_label_is_neutral,
    label_role as dialogue_label_role,
    label_side as dialogue_label_side,
    parse_dialogue_lines as parse_shared_dialogue_lines,
    parse_line as parse_dialogue_line,
    # One shared implementation: a stage-local copy is how one of the two
    # pipelines keeps leaking the conversation after the other is fixed.
    safe_error_text,
    stored_side_by_role,
)
from mango_mvp.services.llm_response_cache import LLMResponseCache
from mango_mvp.services.pipeline_claims import release_stale_pipeline_claims
from mango_mvp.services.transcribe import TranscribeService
from mango_mvp.utils.codex_cli import append_codex_service_tier


RESOLVE_SYSTEM_PROMPT = """You improve one speaker transcript from two ASR variants.
Rules:
1) Keep meaning strictly from variants A and B only.
2) Never invent facts, names, emails, phone numbers or dates.
3) Keep natural Russian punctuation and casing.
4) Keep concise and readable utterance style.
Return strict JSON only:
{
  "merged_text": "...",
  "selection": "A|B|MIX",
  "confidence": 0.0-1.0,
  "notes": "short reason"
}
Return a single-line minified JSON object."""

DIALOGUE_RESOLVE_SYSTEM_PROMPT = """You improve a turn-by-turn Russian sales phone call dialogue.
Rules:
1) Use only information from the provided baseline turns and role variants. Do not invent facts.
2) Preserve the set of turn_id values. Do not add new turns.
3) Keep ts_sec unchanged. Do not rewrite timestamps.
4) final_text must stay close to baseline/variant wording. If uncertain, keep baseline_text.
5) You may set drop=true only for obvious artifact, exact echo, or duplicated garbage.
6) You must NOT reorder turns. swap_with_next must always be false: only the recording decides who spoke first.
7) You must NOT change speaker. Echo the baseline speaker of the turn unchanged. Only the telephony channel markup decides who spoke, never the words.
8) If you believe a speaker or the order is wrong, say so in notes. Do not act on it.
9) Return strict JSON only:
{
  "schema_version": "dialogue_resolve_result_v1",
  "turns": [
    {
      "turn_id": 1,
      "speaker": "same as baseline",
      "final_text": "...",
      "selection": "A|B|MIX|BASELINE",
      "drop": false,
      "swap_with_next": false,
      "confidence": 0.0,
      "notes": ""
    }
  ],
  "warnings": [],
  "global_notes": ""
}
Return a single-line minified JSON object. No markdown, no extra keys."""

RESOLVE_PAIR_PROMPT_VERSION = "v2"
RESOLVE_DIALOGUE_PROMPT_VERSION = "v2"


WORD_RE = re.compile(r"\S+", flags=re.UNICODE)
ARTIFACT_RE = re.compile(r"продолжение следует|голосовой ассистент|абонент недоступен", re.I)


def _clamp_score(value: int) -> int:
    return max(0, min(100, int(value)))


# Every stored field Resolve reads to build its answer, plus the identity of
# the artefact it exports.  The whole tuple is the stale guard: if any of it
# moved while ASR/LLM/rescue were running, our answer describes another call.
RESOLVE_INPUT_COLUMNS = (
    "source_call_id",
    "source_recording_id",
    # Mono vs stereo decides which candidates Resolve is even allowed to build,
    # so a re-ingest that changes it invalidates the answer in flight.
    "channels",
    "transcript_variants_json",
    "transcript_text",
    "transcript_manager",
    "transcript_client",
    "manager_name",
    "phone",
    "direction",
    "started_at",
    "duration_sec",
    "source_filename",
    "source_file",
)

def resolve_input_snapshot(record: Any) -> Dict[str, Any]:
    """One immutable read of the input, taken before any provider runs."""
    if isinstance(record, Mapping):
        return {name: record.get(name) for name in RESOLVE_INPUT_COLUMNS}
    return {name: getattr(record, name, None) for name in RESOLVE_INPUT_COLUMNS}


class ResolveService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._transcribe_helper = TranscribeService(settings)
        self._ollama_client: Optional[OllamaClient] = None
        self._openai_client: Optional[OpenAI] = None
        self._rescue_service_cache: Dict[Tuple[str, bool], TranscribeService] = {}
        self._llm_cache = LLMResponseCache(
            enabled=settings.llm_cache_enabled,
            root_dir=settings.llm_cache_dir,
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _retry_delay(self, attempts: int) -> timedelta:
        base = max(1, self._settings.retry_base_delay_sec)
        multiplier = max(1, 2 ** max(0, attempts - 1))
        return timedelta(seconds=base * multiplier)

    @staticmethod
    def _is_retry_due(next_retry_at: Optional[datetime], now: datetime) -> bool:
        if next_retry_at is None:
            return True
        retry_at = next_retry_at
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return retry_at <= now

    @staticmethod
    def _pipeline_worker_id(prefix: str) -> str:
        return f"{prefix}-{os.getpid()}-{uuid.uuid4().hex}"

    def _claim_batch(self, session: Session, limit: int, worker_id: str) -> list[int]:
        if limit <= 0:
            return []
        now = self._utc_now()
        max_attempts = max(1, self._settings.resolve_max_attempts)
        release_stale_pipeline_claims(session, self._settings, now)
        scope = require_unique_controlled_call(session, self._settings)
        scope_sql = (
            " AND source_call_id = :controlled_source_call_id" if scope else ""
        )
        params: dict[str, Any] = {
            "worker_id": worker_id,
            "now": now,
            "max_attempts": max_attempts,
            "limit": int(limit),
        }
        if scope:
            params["controlled_source_call_id"] = scope.source_call_id
        session.execute(
            text(
                f"""
                UPDATE call_records
                   SET resolve_status = 'in_progress',
                       pipeline_stage = 'resolve',
                       pipeline_worker_id = :worker_id,
                       pipeline_claimed_at = :now,
                       updated_at = :now
                 WHERE id IN (
                    SELECT id
                      FROM call_records
                     WHERE transcription_status = 'done'
                       AND dead_letter_stage IS NULL
                       AND resolve_status IN ('pending', 'failed')
                       AND resolve_attempts < :max_attempts
                       AND (next_retry_at IS NULL OR next_retry_at <= :now)
                       AND pipeline_stage IS NULL
                       {scope_sql}
                     ORDER BY id ASC
                     LIMIT :limit
                 )
                """
            ),
            params,
        )
        ids = [
            int(row[0])
            for row in session.execute(
                text(
                    f"""
                    SELECT id
                      FROM call_records
                     WHERE resolve_status = 'in_progress'
                       AND pipeline_stage = 'resolve'
                       AND pipeline_worker_id = :worker_id
                       {scope_sql}
                     ORDER BY id ASC
                    """
                ),
                params,
            ).all()
        ]
        session.commit()
        return ids

    def count_queue_state(self, session: Session) -> Dict[str, int]:
        now = self._utc_now()
        scope = require_unique_controlled_call(session, self._settings)
        candidate_query = (
            select(CallRecord)
            .where(CallRecord.transcription_status == "done")
            .where(CallRecord.dead_letter_stage.is_(None))
            .where(CallRecord.resolve_status.in_(["pending", "failed"]))
            .where(CallRecord.resolve_attempts < max(1, self._settings.resolve_max_attempts))
            .order_by(CallRecord.id.asc())
        )
        progress_query = (
            select(func.count(CallRecord.id))
            .where(CallRecord.resolve_status == "in_progress")
            .where(CallRecord.pipeline_stage == "resolve")
        )
        if scope:
            candidate_query = candidate_query.where(
                CallRecord.source_call_id == scope.source_call_id
            )
            progress_query = progress_query.where(
                CallRecord.source_call_id == scope.source_call_id
            )
        candidate_calls = session.scalars(candidate_query).all()
        ready = 0
        blocked_waiting_secondary = 0
        for call in candidate_calls:
            if self._waiting_for_secondary_asr(call):
                blocked_waiting_secondary += 1
            elif self._is_retry_due(call.next_retry_at, now):
                ready += 1
        in_progress = int(
            session.scalar(progress_query)
            or 0
        )
        return {
            "ready_pending": ready,
            "blocked_waiting_secondary": blocked_waiting_secondary,
            "in_progress": in_progress,
        }

    def _ollama(self) -> OllamaClient:
        if self._ollama_client is None:
            self._ollama_client = OllamaClient(self._settings.ollama_base_url)
        return self._ollama_client

    def _openai(self) -> OpenAI:
        if not self._settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for resolve_llm_provider=openai")
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self._settings.openai_api_key)
        return self._openai_client

    @staticmethod
    def _safe_json(raw: str) -> Dict[str, Any]:
        value = (raw or "").strip()
        if not value:
            return {}
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    @staticmethod
    def _get_warnings(payload: Dict[str, Any]) -> List[str]:
        warnings = payload.get("warnings")
        if not isinstance(warnings, list):
            return []
        return [str(item).strip() for item in warnings if str(item).strip()]

    @staticmethod
    def _extract_merge_confidences(payload: Dict[str, Any]) -> List[float]:
        values: List[float] = []
        for section in ("manager", "client", "full"):
            block = payload.get(section)
            if not isinstance(block, dict):
                continue
            meta = block.get("merge_meta")
            if not isinstance(meta, dict):
                continue
            try:
                conf = float(meta.get("confidence"))
            except (TypeError, ValueError):
                continue
            values.append(max(0.0, min(1.0, conf)))
        return values

    def _secondary_asr_required(self) -> bool:
        primary = (self._settings.transcribe_provider or "").strip().lower()
        secondary = (self._settings.secondary_transcribe_provider or "").strip().lower()
        return bool(
            self._settings.dual_transcribe_enabled
            and secondary
            and secondary != primary
        )

    def _waiting_for_secondary_asr(self, call: CallRecord) -> bool:
        if not self._secondary_asr_required():
            return False
        secondary = (self._settings.secondary_transcribe_provider or "").strip().lower()
        payload = self._safe_json(call.transcript_variants_json or "")
        if not payload:
            return True
        return self._transcribe_helper._call_needs_secondary_backfill(
            call,
            secondary_provider=secondary,
        )

    def _dialogue_export_path(self, call: CallRecord) -> Optional[Path]:
        export_dir = (self._settings.transcript_export_dir or "").strip()
        if not export_dir:
            return None
        source_path = Path(call.source_file)
        return call_artifact_directory(
            self._settings,
            export_dir=Path(export_dir),
            source_file=source_path,
            source_call_id=call.source_call_id,
        ) / f"{source_path.stem}_text.txt"

    def _load_dialogue_lines_from_export(self, call: CallRecord) -> List[str]:
        payload = self._safe_json(call.transcript_variants_json or "")
        physical_roles = self._physical_role_map(payload)
        stored = payload.get("dialogue_lines")
        if isinstance(stored, list) and stored:
            lines = [str(line).strip() for line in stored if str(line).strip()]
        else:
            path = self._dialogue_export_path(call)
            if not path or not path.exists():
                return []
            lines = [
                line.strip()
                for line in read_call_artifact_text(
                    self._settings,
                    path,
                    errors="ignore",
                ).splitlines()
                if line.strip()
            ]
        parsed = [self._parse_timed_line(line) for line in lines]
        if any(item is None for item in parsed):
            return []
        manager = " ".join(
            str(item["text"])
            for item in parsed
            if item and physical_roles.get(str(item["role"]), item["role"]) == "manager"
        )
        client = " ".join(
            str(item["text"])
            for item in parsed
            if item and physical_roles.get(str(item["role"]), item["role"]) == "client"
        )
        normalize = self._transcribe_helper._normalize_artifact_text
        if (normalize(manager), normalize(client)) != (
            normalize(call.transcript_manager or ""),
            normalize(call.transcript_client or ""),
        ):
            return []
        return lines

    @staticmethod
    def _accepted_role(label: str) -> Optional[str]:
        """Read the shared contract vocabulary without assigning an unproven role."""
        role = dialogue_label_role(label)
        if role is not None:
            return role
        side = dialogue_label_side(label)
        if side is not None:
            return f"channel_{side}"
        return "unknown" if dialogue_label_is_neutral(label) else None

    @staticmethod
    def _physical_role_map(payload: Mapping[str, Any]) -> Dict[str, str]:
        """Map proven physical Mango channels to business roles, or nothing."""
        manager = payload.get("manager") if isinstance(payload.get("manager"), Mapping) else {}
        client = payload.get("client") if isinstance(payload.get("client"), Mapping) else {}
        manager_side = str(manager.get("physical_channel") or "").strip().lower()
        client_side = str(client.get("physical_channel") or "").strip().lower()
        if (
            manager_side not in {"left", "right"}
            or client_side not in {"left", "right"}
            or manager_side == client_side
        ):
            return {}
        return {
            f"channel_{manager_side}": "manager",
            f"channel_{client_side}": "client",
        }

    @classmethod
    def _parse_timed_line(cls, line: str) -> Optional[Dict[str, Any]]:
        """Shared contract grammar plus the contract's own label vocabulary."""
        try:
            parsed = parse_dialogue_line(line)
        except DialogueContractError:
            return None
        speaker = str(parsed["label"])
        role = cls._accepted_role(speaker)
        if role is None:
            return None
        return {
            "ts_sec": float(parsed["start_sec"]),
            "approximate": bool(parsed["approximate"]),
            "speaker_label": speaker,
            "role": role,
            "text": str(parsed["text"]),
            "raw_line": str(parsed["raw_line"]),
        }

    def _parse_dialogue_lines(
        self,
        call: CallRecord,
        dialogue_lines: Optional[List[str]],
        *,
        allow_export_fallback: bool = False,
    ) -> List[Tuple[float, str, str]]:
        """Parse every line or return nothing: partial parse loss is forbidden.

        The grammar and the ordering check come from the shared contract, so a
        line Resolve accepts and a line the publisher accepts are the same line.
        One unreadable line invalidates the whole dialogue: scoring or merging
        the surviving part is exactly how a reply disappears without a reason.
        """
        lines: List[str] = []
        if dialogue_lines is not None:
            lines = [str(line).strip() for line in dialogue_lines]
        elif allow_export_fallback:
            path = self._dialogue_export_path(call)
            if path and path.exists():
                lines = [
                    line.strip()
                    for line in read_call_artifact_text(
                        self._settings,
                        path,
                        errors="ignore",
                    ).splitlines()
                    if line.strip()
                ]
        if not lines:
            return []
        try:
            parsed_lines = parse_shared_dialogue_lines(lines)
        except DialogueContractError:
            return []
        physical_roles = self._physical_role_map(
            self._safe_json(call.transcript_variants_json or "")
        )
        rows: List[Tuple[float, str, str]] = []
        for item in parsed_lines:
            role = self._accepted_role(str(item["label"]))
            if role is None:
                return []
            role = physical_roles.get(role, role)
            rows.append((float(item["start_sec"]), role, str(item["text"]).strip()))
        return rows

    @staticmethod
    def _dialogue_lines_sha256(lines: Sequence[str]) -> str:
        raw = json.dumps(list(lines), ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _candidate_source_is_current(
        self, call: CallRecord, candidate: Mapping[str, Any]
    ) -> bool:
        meta = candidate.get("meta")
        expected = (
            str(meta.get("source_artifact_sha256") or "")
            if isinstance(meta, Mapping)
            else ""
        )
        if not expected:
            return True
        current = self._load_dialogue_lines_from_export(call)
        return bool(current) and self._dialogue_lines_sha256(current) == expected

    def _maybe_postfilter_candidate_dialogue(
        self,
        call: CallRecord,
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        name = str(candidate.get("name") or "")
        lines = candidate.get("dialogue_lines")
        loaded_from_sidecar = False
        if (not isinstance(lines, list) or not lines) and name == "baseline":
            lines = self._load_dialogue_lines_from_export(call)
            loaded_from_sidecar = bool(lines)
        if not isinstance(lines, list) or not lines:
            return candidate

        meta = candidate.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        dialogue_lines_source = str(meta.get("dialogue_lines_source") or "")
        if loaded_from_sidecar:
            dialogue_lines_source = "mutable_sidecar"
        if dialogue_lines_source:
            meta["dialogue_lines_source"] = dialogue_lines_source
        if dialogue_lines_source == "mutable_sidecar":
            meta["source_artifact_sha256"] = self._dialogue_lines_sha256(lines)
        rows_before = self._parse_dialogue_lines(call, lines, allow_export_fallback=False)
        if rows_before:
            before_metrics = self._line_metrics(rows_before)
            meta["same_ts_events_before_postfilter"] = int(
                before_metrics.get("same_ts_cross_speaker_events", 0) or 0
            )

        candidate["dialogue_lines"] = lines
        candidate["meta"] = meta

        payload = self._safe_json(str(candidate.get("transcript_variants_json") or ""))
        if payload:
            payload["dialogue_lines"] = candidate["dialogue_lines"]
            if dialogue_lines_source:
                payload["dialogue_lines_source"] = dialogue_lines_source
            if dialogue_lines_source == "mutable_sidecar":
                role_mapping = payload.get("role_mapping")
                if isinstance(role_mapping, dict):
                    role_mapping.update({
                        "confirmed": False,
                        "manager_quality_allowed": False,
                        "status": "mutable_sidecar_timing",
                    })
            candidate["transcript_variants_json"] = json.dumps(payload, ensure_ascii=False)
        return candidate

    @staticmethod
    def _candidate_same_ts_events(candidate: Optional[Dict[str, Any]]) -> int:
        if not candidate:
            return 0
        meta = candidate.get("meta")
        if isinstance(meta, dict):
            try:
                before = int(meta.get("same_ts_events_before_postfilter") or 0)
            except (TypeError, ValueError):
                before = 0
            if before > 0:
                return before
        quality = candidate.get("quality")
        if not isinstance(quality, dict):
            return 0
        signals = quality.get("signals")
        if not isinstance(signals, dict):
            return 0
        try:
            return int(signals.get("same_ts_cross_speaker_events") or 0)
        except (TypeError, ValueError):
            return 0

    def _is_ordering_risky(self, *candidates: Optional[Dict[str, Any]]) -> bool:
        threshold = max(1, int(self._settings.resolve_risky_same_ts_threshold))
        return any(self._candidate_same_ts_events(item) >= threshold for item in candidates if item)

    def _is_payload_risky_for_llm(
        self,
        payload: Dict[str, Any],
        quality: Optional[Dict[str, Any]],
    ) -> bool:
        warning_text = " | ".join(self._get_warnings(payload)).lower()
        if warning_text:
            risky_tokens = (
                "same_ts",
                "sequence_fix",
                "time_fix",
                "channels_too_similar",
                "mono_role_assign",
            )
            if any(token in warning_text for token in risky_tokens):
                return True

        seq = payload.get("stereo_sequence_fix")
        if isinstance(seq, dict) and int(seq.get("swapped_adjacent_pairs") or 0) > 0:
            return True

        time_fix = payload.get("stereo_time_fix")
        if isinstance(time_fix, dict) and int(time_fix.get("monotonic_adjusted_lines") or 0) > 0:
            return True

        postfilter = payload.get("resolve_same_ts_postfilter")
        if isinstance(postfilter, dict):
            adjusted = int(postfilter.get("adjusted_lines") or 0)
            if adjusted >= max(1, int(self._settings.resolve_risky_same_ts_threshold)):
                return True

        if isinstance(quality, dict):
            signals = quality.get("signals")
            if isinstance(signals, dict):
                same_ts = int(signals.get("same_ts_cross_speaker_events") or 0)
                near_dup = int(signals.get("near_dup_pairs") or 0)
                if same_ts >= max(1, int(self._settings.resolve_risky_same_ts_threshold)):
                    return True
                if near_dup > 0:
                    return True
        return False

    @staticmethod
    def _line_metrics(rows: List[Tuple[float, str, str]]) -> Dict[str, Any]:
        same_ts_cross = 0
        near_dup_pairs = 0
        max_run = 0
        run = 0
        prev_ts: Optional[float] = None
        prev_role: Optional[str] = None
        words = 0

        for idx, (ts, role, text) in enumerate(rows):
            words += len(WORD_RE.findall(text))
            if prev_ts is not None and abs(ts - prev_ts) <= 1e-6 and prev_role != role:
                same_ts_cross += 1

            if prev_role == role:
                run += 1
            else:
                run = 1
            max_run = max(max_run, run)

            if idx > 0:
                _, p_role, p_text = rows[idx - 1]
                if role != p_role and len(p_text) >= 24 and len(text) >= 24:
                    ratio = difflib.SequenceMatcher(None, p_text, text).ratio()
                    if ratio >= 0.92:
                        near_dup_pairs += 1

            prev_ts = ts
            prev_role = role

        return {
            "lines": len(rows),
            "words": words,
            "same_ts_cross_speaker_events": same_ts_cross,
            "near_dup_pairs": near_dup_pairs,
            "max_same_speaker_run": max_run,
        }

    def _score_candidate(
        self,
        call: CallRecord,
        transcript_text: str,
        transcript_manager: Optional[str],
        transcript_client: Optional[str],
        variants_payload: Dict[str, Any],
        dialogue_lines: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        score = 100
        reasons: List[str] = []
        signals: Dict[str, Any] = {}

        duration = float(call.duration_sec or 0.0)
        mode = str(variants_payload.get("mode") or "")
        warnings = self._get_warnings(variants_payload)
        warning_text = " | ".join(warnings).lower()

        if mode == "mono_or_fallback":
            score -= 24
            reasons.append("mono_or_fallback")
        if warnings:
            score -= min(20, len(warnings) * 3)
            reasons.append(f"warnings={len(warnings)}")
        if "channels_too_similar" in warning_text:
            score -= 12
            reasons.append("channels_too_similar")
        if "primary_empty" in warning_text:
            score -= 18
            reasons.append("primary_empty")
        if "secondary" in warning_text and "empty" in warning_text:
            score -= 8
            reasons.append("secondary_empty")

        for key, step in (
            ("stereo_crosstalk_dedupe", 2),
            ("stereo_echo_dedupe", 2),
            ("dialogue_artifact_filter", 2),
        ):
            block = variants_payload.get(key)
            if isinstance(block, dict):
                dropped = int(block.get("dropped_lines") or 0)
                if dropped > 0:
                    penalty = min(12, dropped * step)
                    score -= penalty
                    reasons.append(f"{key}_dropped={dropped}")
                    signals[key] = dropped

        seq = variants_payload.get("stereo_sequence_fix")
        if isinstance(seq, dict):
            swapped = int(seq.get("swapped_adjacent_pairs") or 0)
            if swapped > 0:
                score -= min(10, swapped * 3)
                reasons.append(f"sequence_swapped={swapped}")
                signals["sequence_swapped"] = swapped

        time_fix = variants_payload.get("stereo_time_fix")
        if isinstance(time_fix, dict):
            adjusted = int(time_fix.get("monotonic_adjusted_lines") or 0)
            if adjusted > 0:
                score -= min(10, adjusted * 2)
                reasons.append(f"time_adjusted={adjusted}")
                signals["time_adjusted"] = adjusted

        confidences = self._extract_merge_confidences(variants_payload)
        if confidences:
            avg_conf = sum(confidences) / float(len(confidences))
            signals["avg_merge_confidence"] = round(avg_conf, 4)
            if avg_conf < 0.5:
                score -= 15
                reasons.append("low_merge_confidence")
            elif avg_conf < 0.65:
                score -= 8
                reasons.append("medium_merge_confidence")

        manager_text = (transcript_manager or "").strip()
        client_text = (transcript_client or "").strip()
        full_text = (transcript_text or "").strip()
        lowered = full_text.lower()

        if ARTIFACT_RE.search(lowered):
            score -= 15
            reasons.append("artifact_phrase")

        if manager_text and client_text:
            if len(manager_text) < 20 or len(client_text) < 20:
                score -= 12
                reasons.append("one_role_too_short")
            ratio = len(manager_text) / float(max(1, len(client_text)))
            signals["manager_client_len_ratio"] = round(ratio, 3)
            if ratio > 8.0 or ratio < 0.125:
                score -= 10
                reasons.append("role_length_imbalance")
        elif not full_text:
            score -= 30
            reasons.append("empty_transcript")

        words = len(WORD_RE.findall(full_text))
        signals["words"] = words
        if duration >= 120 and words < 30:
            score -= 20
            reasons.append("too_few_words_for_duration")
        elif duration >= 60 and words < 20:
            score -= 12
            reasons.append("few_words")

        rows = self._parse_dialogue_lines(call, dialogue_lines, allow_export_fallback=False)
        if rows:
            metrics = self._line_metrics(rows)
            signals.update(metrics)
            same_ts = int(metrics.get("same_ts_cross_speaker_events", 0))
            near_dup = int(metrics.get("near_dup_pairs", 0))
            max_run = int(metrics.get("max_same_speaker_run", 0))
            if same_ts > 0:
                score -= min(18, same_ts * 2)
                reasons.append(f"same_ts_cross={same_ts}")
            if near_dup > 0:
                score -= min(16, near_dup * 8)
                reasons.append(f"near_dup_pairs={near_dup}")
            if max_run >= 12:
                score -= min(15, (max_run - 11) * 2)
                reasons.append(f"long_speaker_run={max_run}")

        score = _clamp_score(score)
        if not reasons:
            reasons.append("clean")
        return {"score": score, "reasons": reasons, "signals": signals}

    @staticmethod
    def _copy_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        return json.loads(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _selection(merged_text: str, a: str, b: str) -> str:
        ma = merged_text.strip()
        if ma == a.strip() and ma:
            return "A"
        if ma == b.strip() and ma:
            return "B"
        return "MIX"

    @staticmethod
    def _rule_merge(a: str, b: str, helper: TranscribeService) -> str:
        aa = (a or "").strip()
        bb = (b or "").strip()
        if not bb:
            return aa
        if not aa:
            return bb
        return helper._merge_texts(aa, bb)

    def _merge_pair_with_llm(
        self,
        *,
        speaker_label: str,
        variant_a: str,
        variant_b: str,
        context: str,
    ) -> Dict[str, Any]:
        a = (variant_a or "").strip()
        b = (variant_b or "").strip()
        if not b:
            return {
                "merged_text": a,
                "selection": "A",
                "confidence": 1.0 if a else 0.0,
                "provider": "single",
                "notes": "variant_b_empty",
            }
        if not a:
            return {
                "merged_text": b,
                "selection": "B",
                "confidence": 0.6 if b else 0.0,
                "provider": "single",
                "notes": "variant_a_empty",
            }

        similarity = difflib.SequenceMatcher(None, a, b).ratio()
        if similarity >= self._settings.dual_merge_similarity_threshold:
            return {
                "merged_text": a,
                "selection": "A",
                "confidence": 0.95,
                "provider": "skip_high_similarity",
                "notes": f"similarity={similarity:.4f}",
                "similarity": round(similarity, 4),
            }

        provider = (self._settings.resolve_llm_provider or "").strip().lower()
        if provider not in {"ollama", "openai", "codex_cli"}:
            merged = self._rule_merge(a, b, self._transcribe_helper)
            return {
                "merged_text": merged,
                "selection": self._selection(merged, a, b),
                "confidence": 0.72,
                "provider": "rule",
                "notes": "resolve_llm_provider_off",
                "similarity": round(similarity, 4),
            }
        user_prompt = (
            f"Speaker: {speaker_label}\n"
            f"Context (other side, optional):\n{(context or '').strip()[:1200]}\n\n"
            f"Variant A:\n{a}\n\n"
            f"Variant B:\n{b}"
        )
        prompt = f"{RESOLVE_SYSTEM_PROMPT}\n\n{user_prompt}"
        reasoning_effort = (self._settings.codex_reasoning_effort or "").strip().lower()
        cached = self._llm_cache.get(
            namespace="resolve_pair_merge",
            provider=provider,
            model=self._settings.openai_merge_model if provider == "openai" else (
                self._settings.codex_resolve_model if provider == "codex_cli" else self._settings.ollama_model
            ),
            reasoning=(
                "temperature=0.0"
                if provider == "openai"
                else (reasoning_effort if provider == "codex_cli" else f"think={self._settings.ollama_think}")
            ),
            prompt_version=RESOLVE_PAIR_PROMPT_VERSION,
            prompt=prompt,
        )
        if cached is not None:
            cached = dict(cached)
            cached["similarity"] = round(similarity, 4)
            return cached
        try:
            if provider == "openai":
                response = self._openai().chat.completions.create(
                    model=self._settings.openai_merge_model,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": RESOLVE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content if response.choices else None
                if not content:
                    raise RuntimeError("empty content")
                payload = json.loads(content)
            elif provider == "codex_cli":
                codex_bin = (self._settings.codex_cli_command or "codex").strip() or "codex"
                if shutil.which(codex_bin) is None:
                    raise RuntimeError(f"codex binary is not available: {codex_bin}")
                timeout_sec = max(15, int(self._settings.codex_cli_timeout_sec))
                with tempfile.NamedTemporaryFile(
                    prefix="mango_resolve_codex_", suffix=".txt"
                ) as out_file:
                    cmd = [
                        codex_bin,
                        "exec",
                        "--skip-git-repo-check",
                        "--ephemeral",
                        "--sandbox",
                        "read-only",
                        "--model",
                        self._settings.codex_resolve_model,
                        "--output-last-message",
                        out_file.name,
                    ]
                    append_codex_service_tier(cmd)
                    if reasoning_effort in {"low", "medium", "high"}:
                        cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
                    cmd.append(
                        prompt
                    )
                    started_at = time.time()
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=timeout_sec,
                    )
                    elapsed_sec = time.time() - started_at
                    if proc.returncode != 0:
                        stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
                        raise RuntimeError(
                            f"codex exec failed rc={proc.returncode}: {stderr_tail[0].strip()}"
                        )
                    raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore")
                payload = self._transcribe_helper._extract_json_payload(raw)
            else:
                payload = self._ollama().generate_json(
                    model=self._settings.ollama_model,
                    think=self._settings.ollama_think,
                    temperature=self._settings.ollama_temperature,
                    system_prompt=RESOLVE_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    num_predict=900,
                )
            merged = str(payload.get("merged_text", "")).strip()
            if not merged:
                raise RuntimeError("empty merged_text")
            selection = str(payload.get("selection", "MIX")).strip().upper()
            if selection not in {"A", "B", "MIX"}:
                selection = self._selection(merged, a, b)
            try:
                confidence = float(payload.get("confidence"))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            result = {
                "merged_text": merged,
                "selection": selection,
                "confidence": confidence,
                "provider": provider,
                "notes": str(payload.get("notes", "")).strip(),
                "similarity": round(similarity, 4),
            }
            if provider == "codex_cli":
                result["tokens_used_actual"] = self._transcribe_helper._parse_codex_tokens_used(proc.stderr or "")
                result["duration_sec"] = round(elapsed_sec, 3)
            self._llm_cache.put(
                namespace="resolve_pair_merge",
                provider=provider,
                model=self._settings.openai_merge_model if provider == "openai" else (
                    self._settings.codex_resolve_model if provider == "codex_cli" else self._settings.ollama_model
                ),
                reasoning=(
                    "temperature=0.0"
                    if provider == "openai"
                    else (reasoning_effort if provider == "codex_cli" else f"think={self._settings.ollama_think}")
                ),
                prompt_version=RESOLVE_PAIR_PROMPT_VERSION,
                prompt=prompt,
                response=result,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            merged = self._rule_merge(a, b, self._transcribe_helper)
            return {
                "merged_text": merged,
                "selection": self._selection(merged, a, b),
                "confidence": 0.62,
                "provider": "rule_fallback",
                "notes": f"{provider}_failed: {exc}",
                "similarity": round(similarity, 4),
            }

    def _dialogue_resolve_provider(self) -> str:
        mode = (self._settings.resolve_dialogue_mode or "").strip().lower()
        if mode not in {"dialogue", "legacy"}:
            mode = "dialogue"
        if mode != "dialogue":
            return "rule"
        provider = (self._settings.resolve_llm_provider or "").strip().lower()
        if provider in {"ollama", "openai", "codex_cli"}:
            return provider
        return "rule"

    def _build_dialogue_resolve_payload(
        self,
        call: CallRecord,
        variants_payload: Dict[str, Any],
        baseline_dialogue_lines: List[str],
    ) -> Optional[Dict[str, Any]]:
        mode = str(variants_payload.get("mode") or "")
        if mode != "stereo":
            return None
        manager = variants_payload.get("manager")
        client = variants_payload.get("client")
        if not isinstance(manager, dict) or not isinstance(client, dict):
            return None

        parsed_turns: List[Dict[str, Any]] = []
        previous_ts: Optional[float] = None
        previous_role: Optional[str] = None
        for idx, raw in enumerate(baseline_dialogue_lines, start=1):
            parsed = self._parse_timed_line(raw)
            if parsed is None:
                return None
            role = str(parsed.get("role") or "unknown")
            text = str(parsed.get("text") or "").strip()
            flags: List[str] = []
            ts_sec = float(parsed.get("ts_sec") or 0.0)
            if previous_ts is not None and previous_role is not None:
                if role != previous_role and abs(ts_sec - previous_ts) <= 1e-6:
                    flags.append("same_ts_cross")
            if ARTIFACT_RE.search(text.lower()):
                flags.append("artifact_candidate")
            parsed_turns.append(
                {
                    "turn_id": idx,
                    "ts_sec": round(ts_sec, 3),
                    "ts_label": self._transcribe_helper._format_timecode(
                        ts_sec,
                        approximate=bool(parsed.get("approximate")),
                    ).strip("[]"),
                    "speaker": role,
                    "speaker_label": str(parsed.get("speaker_label") or "").strip(),
                    "baseline_text": text,
                    "approximate": bool(parsed.get("approximate")),
                    "flags": flags,
                }
            )
            previous_ts = ts_sec
            previous_role = role

        if not parsed_turns:
            return None

        metrics = self._line_metrics(
            [
                (
                    float(turn["ts_sec"]),
                    str(turn["speaker"]),
                    str(turn["baseline_text"]),
                )
                for turn in parsed_turns
            ]
        )
        manager_name = (
            (call.manager_name or "").strip()
            or self._transcribe_helper._extract_manager_name_from_filename(call.source_filename)
        )
        return {
            "schema_version": "dialogue_resolve_v1",
            "call_id": int(call.id or 0),
            "source_filename": call.source_filename,
            "manager_name": manager_name,
            "mode": mode,
            "duration_sec": round(float(call.duration_sec or 0.0), 3),
            "providers": {
                "primary": variants_payload.get("primary_provider"),
                "secondary": variants_payload.get("secondary_provider"),
                "merge_provider": variants_payload.get("merge_provider"),
            },
            "role_variants": {
                "manager": {
                    "variant_a": str(manager.get("variant_a") or "").strip(),
                    "variant_b": str(manager.get("variant_b") or "").strip(),
                    "baseline_text": str(manager.get("final") or call.transcript_manager or "").strip(),
                },
                "client": {
                    "variant_a": str(client.get("variant_a") or "").strip(),
                    "variant_b": str(client.get("variant_b") or "").strip(),
                    "baseline_text": str(client.get("final") or call.transcript_client or "").strip(),
                },
            },
            "turns": parsed_turns,
            "quality_hints": {
                "same_ts_cross": int(metrics.get("same_ts_cross_speaker_events", 0) or 0),
                "near_dup_pairs": int(metrics.get("near_dup_pairs", 0) or 0),
                "warnings": self._get_warnings(variants_payload),
            },
        }

    def _dialogue_turn_output_prompt(self, input_payload: Dict[str, Any]) -> str:
        return (
            "Call dialogue payload JSON:\n"
            + json.dumps(input_payload, ensure_ascii=False, indent=2)
        )

    def _run_dialogue_llm(
        self,
        input_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        provider = self._dialogue_resolve_provider()
        if provider == "rule":
            raise RuntimeError("dialogue-level LLM is disabled")
        user_prompt = self._dialogue_turn_output_prompt(input_payload)
        prompt = f"{DIALOGUE_RESOLVE_SYSTEM_PROMPT}\n\n{user_prompt}"
        reasoning_effort = (self._settings.codex_reasoning_effort or "").strip().lower()
        cached = self._llm_cache.get(
            namespace="resolve_dialogue",
            provider=provider,
            model=self._settings.openai_merge_model if provider == "openai" else (
                self._settings.codex_resolve_model if provider == "codex_cli" else self._settings.ollama_model
            ),
            reasoning=(
                "temperature=0.0"
                if provider == "openai"
                else (reasoning_effort if provider == "codex_cli" else f"think={self._settings.ollama_think}")
            ),
            prompt_version=RESOLVE_DIALOGUE_PROMPT_VERSION,
            prompt=prompt,
        )
        if cached is not None:
            return cached
        if provider == "openai":
            response = self._openai().chat.completions.create(
                model=self._settings.openai_merge_model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": DIALOGUE_RESOLVE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content if response.choices else None
            if not content:
                raise RuntimeError("empty content")
            payload = json.loads(content)
        elif provider == "codex_cli":
            codex_bin = (self._settings.codex_cli_command or "codex").strip() or "codex"
            if shutil.which(codex_bin) is None:
                raise RuntimeError(f"codex binary is not available: {codex_bin}")
            timeout_sec = max(15, int(self._settings.codex_cli_timeout_sec))
            with tempfile.NamedTemporaryFile(
                prefix="mango_resolve_dialogue_codex_", suffix=".txt"
            ) as out_file:
                cmd = [
                    codex_bin,
                    "exec",
                    "--skip-git-repo-check",
                    "--ephemeral",
                    "--sandbox",
                    "read-only",
                    "--model",
                    self._settings.codex_resolve_model,
                    "--output-last-message",
                    out_file.name,
                ]
                append_codex_service_tier(cmd)
                if reasoning_effort in {"low", "medium", "high"}:
                    cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
                cmd.append(prompt)
                started_at = time.time()
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=timeout_sec,
                )
                elapsed_sec = time.time() - started_at
                if proc.returncode != 0:
                    stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
                    raise RuntimeError(
                        f"codex exec failed rc={proc.returncode}: {stderr_tail[0].strip()}"
                    )
                raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore")
            payload = self._transcribe_helper._extract_json_payload(raw)
            payload["_llm_meta"] = {
                "llm_tokens_used_actual": self._transcribe_helper._parse_codex_tokens_used(proc.stderr or ""),
                "llm_duration_sec": round(elapsed_sec, 3),
            }
        else:
            payload = self._ollama().generate_json(
                model=self._settings.ollama_model,
                think=self._settings.ollama_think,
                temperature=self._settings.ollama_temperature,
                system_prompt=DIALOGUE_RESOLVE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                num_predict=max(1600, len(input_payload.get("turns") or []) * 120),
            )
        if not isinstance(payload, dict):
            raise RuntimeError("dialogue resolve payload is not an object")
        self._llm_cache.put(
            namespace="resolve_dialogue",
            provider=provider,
            model=self._settings.openai_merge_model if provider == "openai" else (
                self._settings.codex_resolve_model if provider == "codex_cli" else self._settings.ollama_model
            ),
            reasoning=(
                "temperature=0.0"
                if provider == "openai"
                else (reasoning_effort if provider == "codex_cli" else f"think={self._settings.ollama_think}")
            ),
            prompt_version=RESOLVE_DIALOGUE_PROMPT_VERSION,
            prompt=prompt,
            response=payload,
        )
        return payload

    def _normalize_dialogue_result(
        self,
        input_payload: Dict[str, Any],
        llm_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        input_turns = input_payload.get("turns")
        if not isinstance(input_turns, list) or not input_turns:
            raise RuntimeError("dialogue resolve input has no turns")
        output_turns = llm_payload.get("turns")
        if not isinstance(output_turns, list):
            raise RuntimeError("dialogue resolve output has no turns array")

        input_by_id: Dict[int, Dict[str, Any]] = {}
        for turn in input_turns:
            try:
                turn_id = int(turn.get("turn_id"))
            except (TypeError, ValueError):
                raise RuntimeError("dialogue resolve input contains invalid turn_id")
            input_by_id[turn_id] = dict(turn)

        output_by_id: Dict[int, Dict[str, Any]] = {}
        for raw in output_turns:
            if not isinstance(raw, dict):
                raise RuntimeError("dialogue resolve output turn is not object")
            try:
                turn_id = int(raw.get("turn_id"))
            except (TypeError, ValueError):
                raise RuntimeError("dialogue resolve output contains invalid turn_id")
            if turn_id not in input_by_id:
                raise RuntimeError(f"dialogue resolve output contains unknown turn_id={turn_id}")
            if turn_id in output_by_id:
                raise RuntimeError(f"dialogue resolve output duplicated turn_id={turn_id}")
            output_by_id[turn_id] = raw

        if set(output_by_id) != set(input_by_id):
            raise RuntimeError("dialogue resolve output turn_id set mismatch")

        role_variants = input_payload.get("role_variants")
        if not isinstance(role_variants, dict):
            role_variants = {}

        normalized: List[Dict[str, Any]] = []
        warnings: List[str] = []
        speaker_corrections_rejected = 0
        drops_requested = 0
        for input_turn in input_turns:
            turn_id = int(input_turn["turn_id"])
            out_turn = output_by_id[turn_id]
            role = str(input_turn.get("speaker") or "unknown")
            requested_role = str(out_turn.get("speaker") or role).strip().lower()
            turn_flags = {
                str(flag).strip().lower()
                for flag in input_turn.get("flags", [])
                if str(flag).strip()
            }
            if requested_role not in {
                "manager", "client", "unknown", "channel_left", "channel_right",
            }:
                requested_role = role
            if requested_role != role:
                # The model may never move a turn to another physical side or
                # role: only Mango's own channel markup decides who spoke.  The
                # rejected candidate is not stored anywhere.
                speaker_corrections_rejected += 1
                warnings.append(f"speaker_change_rejected:{turn_id}")

            baseline_text = str(input_turn.get("baseline_text") or "").strip()
            role_block = role_variants.get(role) if isinstance(role_variants.get(role), dict) else {}
            ref_lengths = [
                len(baseline_text),
                len(str(role_block.get("variant_a") or "").strip()),
                len(str(role_block.get("variant_b") or "").strip()),
                len(str(role_block.get("baseline_text") or "").strip()),
            ]
            max_ref_len = max(1, max(ref_lengths))
            final_text = " ".join(str(out_turn.get("final_text") or "").split()).strip()
            if not final_text and not bool(out_turn.get("drop")):
                final_text = baseline_text
            if len(final_text) > max_ref_len * 3 + 80:
                warnings.append(f"oversize_text_reset:{turn_id}")
                final_text = baseline_text

            drop = bool(out_turn.get("drop"))
            if drop:
                drops_requested += 1
                drop_allowed = "artifact_candidate" in turn_flags or "echo_candidate" in turn_flags
                if not drop_allowed:
                    warnings.append(f"drop_ignored:{turn_id}")
                    drop = False

            selection = str(out_turn.get("selection") or "BASELINE").strip().upper()
            if selection not in {"A", "B", "MIX", "BASELINE"}:
                selection = "BASELINE"

            try:
                confidence = float(out_turn.get("confidence"))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            normalized.append(
                {
                    "turn_id": turn_id,
                    "ts_sec": float(input_turn.get("ts_sec") or 0.0),
                    "approximate": bool(input_turn.get("approximate")),
                    "speaker": role,
                    "baseline_text": baseline_text,
                    "final_text": final_text or baseline_text,
                    "selection": selection,
                    "drop": drop,
                    "swap_with_next": bool(out_turn.get("swap_with_next")),
                    "confidence": confidence,
                    "notes": str(out_turn.get("notes") or "").strip(),
                }
            )

        # Chronology belongs to the recording, not to the model.  A requested
        # swap is counted and warned about, and the order never moves: reading
        # the reply of the other side as an answer is the same class of error as
        # naming the wrong speaker, and here it would be invisible afterwards.
        swap_requests_rejected = 0
        for turn in normalized:
            if bool(turn.get("swap_with_next")):
                swap_requests_rejected += 1
                warnings.append(f"swap_rejected:{turn['turn_id']}")
                turn["swap_with_next"] = False

        kept_turns = [
            turn
            for turn in normalized
            if not bool(turn.get("drop")) and str(turn.get("final_text") or "").strip()
        ]
        if not kept_turns:
            raise RuntimeError("dialogue resolve dropped all turns")

        if len(kept_turns) < max(1, len(input_turns) // 3):
            raise RuntimeError("dialogue resolve dropped too many turns")

        global_notes = str(llm_payload.get("global_notes") or "").strip()
        raw_warnings = llm_payload.get("warnings")
        if isinstance(raw_warnings, list):
            for item in raw_warnings:
                text = str(item).strip()
                if text:
                    warnings.append(text)

        return {
            "turns": kept_turns,
            "warnings": warnings,
            "global_notes": global_notes,
            # Applied swaps and applied speaker corrections are both structurally
            # impossible now; the counters of rejected attempts are what revoke
            # role trust downstream.
            "swaps_applied": 0,
            "swap_requests_rejected": swap_requests_rejected,
            "drops_requested": drops_requested,
            "speaker_corrections": 0,
            "speaker_corrections_rejected": speaker_corrections_rejected,
        }

    def _dialogue_turns_to_candidate(
        self,
        call: CallRecord,
        variants_payload: Dict[str, Any],
        normalized_result: Dict[str, Any],
        *,
        provider: str,
        llm_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        side_by_role = stored_side_by_role(variants_payload)
        role_by_side = {side: role for role, side in side_by_role.items()}
        manager_parts: List[str] = []
        client_parts: List[str] = []
        dialogue_lines: List[str] = []
        for turn in normalized_result.get("turns", []):
            speaker = str(turn.get("speaker") or "unknown")
            text = str(turn.get("final_text") or "").strip()
            if not text:
                continue
            ts_sec = float(turn.get("ts_sec") or 0.0)
            approximate = bool(turn.get("approximate"))
            side = (
                speaker.removeprefix("channel_")
                if speaker in {"channel_left", "channel_right"}
                else side_by_role.get(speaker)
            )
            role = speaker if speaker in {"manager", "client"} else role_by_side.get(side)
            if role == "manager":
                manager_parts.append(text)
            elif role == "client":
                client_parts.append(text)
            speaker_label = {
                "left": "Дорожка левая",
                "right": "Дорожка правая",
            }.get(side, "Спикер (не определен)")
            dialogue_lines.append(
                f"{self._transcribe_helper._format_timecode(ts_sec, approximate=approximate)} {speaker_label}: {text}"
            )

        manager_text = " ".join(manager_parts).strip()
        client_text = " ".join(client_parts).strip()
        if manager_text or client_text:
            transcript_text = f"MANAGER:\n{manager_text}\n\nCLIENT:\n{client_text}"
        else:
            transcript_text = "\n".join(dialogue_lines).strip()

        payload = self._copy_payload(variants_payload)
        if int(normalized_result.get("speaker_corrections_rejected") or 0):
            role_mapping = payload.get("role_mapping")
            if isinstance(role_mapping, dict):
                role_mapping.update({
                    "confirmed": False,
                    "manager_quality_allowed": False,
                    "status": "model_speaker_correction",
                })
        warnings = self._get_warnings(payload)
        for item in normalized_result.get("warnings", []):
            text = str(item).strip()
            if text and text not in warnings:
                warnings.append(text)
        payload["warnings"] = warnings
        payload["resolve"] = {
            "provider": provider,
            "mode": "stereo_dialogue",
            "applied": True,
            "swaps_applied": int(normalized_result.get("swaps_applied") or 0),
            "speaker_corrections": int(normalized_result.get("speaker_corrections") or 0),
        }
        payload["dialogue_resolve"] = {
            "schema_version": "dialogue_resolve_result_v1",
            "turns_kept": len(dialogue_lines),
            "swaps_applied": int(normalized_result.get("swaps_applied") or 0),
            "drops_requested": int(normalized_result.get("drops_requested") or 0),
            "speaker_corrections": int(normalized_result.get("speaker_corrections") or 0),
            "speaker_corrections_rejected": int(
                normalized_result.get("speaker_corrections_rejected") or 0
            ),
            "swap_requests_rejected": int(
                normalized_result.get("swap_requests_rejected") or 0
            ),
            "warnings": normalized_result.get("warnings", []),
            "global_notes": str(normalized_result.get("global_notes") or "").strip(),
        }
        payload["dialogue_lines"] = dialogue_lines
        if isinstance(llm_meta, dict) and llm_meta:
            payload["dialogue_resolve"]["llm_meta"] = llm_meta
        manager_block = payload.get("manager")
        if isinstance(manager_block, dict):
            manager_block["final"] = manager_text
        client_block = payload.get("client")
        if isinstance(client_block, dict):
            client_block["final"] = client_text

        return {
            "name": "llm",
            "transcript_manager": manager_text,
            "transcript_client": client_text,
            "transcript_text": transcript_text,
            "dialogue_lines": dialogue_lines,
            "transcript_variants_json": json.dumps(payload, ensure_ascii=False),
            "meta": {
                "mode": "stereo",
                "provider": provider,
                "resolve_mode": "dialogue_level",
                "swaps_applied": int(normalized_result.get("swaps_applied") or 0),
                "speaker_corrections": int(normalized_result.get("speaker_corrections") or 0),
                **(llm_meta if isinstance(llm_meta, dict) else {}),
            },
        }

    def _resolve_dialogue_with_llm(
        self,
        call: CallRecord,
        variants_payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        provider = self._dialogue_resolve_provider()
        if provider == "rule":
            return None
        stored_lines = variants_payload.get("dialogue_lines")
        from_sidecar = variants_payload.get("dialogue_lines_source") == "mutable_sidecar" or not (
            isinstance(stored_lines, list)
            and any(str(line).strip() for line in stored_lines)
        )
        baseline_dialogue_lines = self._load_dialogue_lines_from_export(call)
        input_payload = self._build_dialogue_resolve_payload(
            call,
            variants_payload,
            baseline_dialogue_lines,
        )
        if not input_payload:
            return None
        raw_result = self._run_dialogue_llm(input_payload)
        llm_meta = raw_result.get("_llm_meta") if isinstance(raw_result.get("_llm_meta"), dict) else None
        normalized_result = self._normalize_dialogue_result(input_payload, raw_result)
        if from_sidecar:
            role_mapping = variants_payload.get("role_mapping")
            if isinstance(role_mapping, dict):
                role_mapping.update({
                    "confirmed": False,
                    "manager_quality_allowed": False,
                    "status": "mutable_sidecar_timing",
                })
        candidate = self._dialogue_turns_to_candidate(
            call,
            variants_payload,
            normalized_result,
            provider=f"{provider}_dialogue",
            llm_meta=llm_meta,
        )
        if from_sidecar:
            candidate["meta"]["dialogue_lines_source"] = "mutable_sidecar"
            candidate["meta"]["source_artifact_sha256"] = (
                self._dialogue_lines_sha256(baseline_dialogue_lines)
            )
        return candidate

    def _resolve_with_llm(
        self,
        call: CallRecord,
        variants_payload: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        payload = self._copy_payload(variants_payload)
        mode = str(payload.get("mode") or "")
        if mode not in {"stereo", "mono_or_fallback"}:
            return None

        llm_provider = (self._settings.resolve_llm_provider or "").strip().lower()
        if llm_provider not in {"ollama", "openai", "codex_cli"}:
            return None
        contextual_provider = f"{llm_provider}_contextual"

        if mode == "stereo":
            dialogue_candidate = self._resolve_dialogue_with_llm(call, payload)
            if dialogue_candidate is not None:
                return dialogue_candidate

            manager = payload.get("manager")
            client = payload.get("client")
            if not isinstance(manager, dict) or not isinstance(client, dict):
                return None

            manager_a = str(manager.get("variant_a") or "").strip()
            manager_b = str(manager.get("variant_b") or "").strip()
            client_a = str(client.get("variant_a") or "").strip()
            client_b = str(client.get("variant_b") or "").strip()
            if not manager_b and not client_b:
                return None

            manager_ctx = str(client.get("final") or call.transcript_client or "").strip()
            client_ctx = str(manager.get("final") or call.transcript_manager or "").strip()
            manager_merge = self._merge_pair_with_llm(
                speaker_label="Менеджер",
                variant_a=manager_a,
                variant_b=manager_b,
                context=manager_ctx,
            )
            client_merge = self._merge_pair_with_llm(
                speaker_label="Клиент",
                variant_a=client_a,
                variant_b=client_b,
                context=client_ctx,
            )
            manager_text = str(manager_merge.get("merged_text") or "").strip()
            client_text = str(client_merge.get("merged_text") or "").strip()
            transcript_text = f"MANAGER:\n{manager_text}\n\nCLIENT:\n{client_text}"

            manager["resolved"] = manager_merge
            manager["final"] = manager_text
            client["resolved"] = client_merge
            client["final"] = client_text
            payload["resolve"] = {
                "provider": contextual_provider,
                "mode": "stereo_per_role",
                "applied": True,
            }
            llm_tokens_used_actual = sum(
                int(item.get("tokens_used_actual") or 0)
                for item in (manager_merge, client_merge)
                if isinstance(item, dict)
            )
            llm_duration_sec = round(
                sum(float(item.get("duration_sec") or 0.0) for item in (manager_merge, client_merge) if isinstance(item, dict)),
                3,
            )
            return {
                "name": "llm",
                "transcript_manager": manager_text,
                "transcript_client": client_text,
                "transcript_text": transcript_text,
                "dialogue_lines": payload.get("dialogue_lines"),
                "transcript_variants_json": json.dumps(payload, ensure_ascii=False),
                "meta": {
                    "mode": "stereo",
                    "provider": contextual_provider,
                    "llm_tokens_used_actual": llm_tokens_used_actual or None,
                    "llm_duration_sec": llm_duration_sec,
                },
            }

        full = payload.get("full")
        if not isinstance(full, dict):
            return None
        full_a = str(full.get("variant_a") or "").strip()
        full_b = str(full.get("variant_b") or "").strip()
        if not full_b:
            return None

        full_merge = self._merge_pair_with_llm(
            speaker_label="Полный звонок",
            variant_a=full_a,
            variant_b=full_b,
            context="",
        )
        resolved_text = str(full_merge.get("merged_text") or "").strip()
        if not resolved_text:
            return None

        full["resolved"] = full_merge
        full["final"] = resolved_text
        payload["resolve"] = {
            "provider": contextual_provider,
            "mode": "mono_full",
            "applied": True,
        }
        return {
            "name": "llm",
            "transcript_manager": call.transcript_manager,
            "transcript_client": call.transcript_client,
            "transcript_text": resolved_text,
            "dialogue_lines": None,
            "transcript_variants_json": json.dumps(payload, ensure_ascii=False),
            "meta": {
                "mode": "mono_or_fallback",
                "provider": contextual_provider,
                "llm_tokens_used_actual": int(full_merge.get("tokens_used_actual") or 0) or None,
                "llm_duration_sec": round(float(full_merge.get("duration_sec") or 0.0), 3),
            },
        }

    def _rescue_provider(self) -> str:
        configured = (self._settings.resolve_rescue_provider or "").strip().lower()
        if configured:
            if configured in {"none", "off", "disabled", "disable", "false", "0"}:
                return ""
            return configured
        primary = (self._settings.transcribe_provider or "").strip().lower()
        secondary = (self._settings.secondary_transcribe_provider or "").strip().lower()
        if secondary and secondary != primary:
            return secondary
        if primary == "mlx":
            return "gigaam"
        if primary == "gigaam":
            return "mlx"
        return "mlx"

    def _run_rescue_asr(self, call: CallRecord) -> Optional[Dict[str, Any]]:
        provider = self._rescue_provider()
        if not provider:
            return None
        dual = bool(self._settings.resolve_rescue_dual_enabled)
        cache_key = (provider, dual)
        service = self._rescue_service_cache.get(cache_key)
        if service is None:
            rescue_settings = replace(
                self._settings,
                transcribe_provider=provider,
                dual_transcribe_enabled=dual,
                secondary_transcribe_provider=None,
                dual_merge_provider="rule",
            )
            service = TranscribeService(rescue_settings)
            self._rescue_service_cache[cache_key] = service
        result = service._transcribe_call(call)
        # Rescue rebuilds ASR variants, but it must not erase or replace the
        # independently captured Mango evidence.  The dialogue guard will still
        # compare that evidence with the rescued turns and fail closed on drift.
        original = self._safe_json(call.transcript_variants_json or "")
        rescued = self._safe_json(str(result.get("transcript_variants_json") or ""))
        if PROVIDER_EVIDENCE_FIELD in original:
            rescued[PROVIDER_EVIDENCE_FIELD] = original[PROVIDER_EVIDENCE_FIELD]
        else:
            rescued.pop(PROVIDER_EVIDENCE_FIELD, None)
        if "provider_capture_manifest_sha256" in original:
            rescued["provider_capture_manifest_sha256"] = original[
                "provider_capture_manifest_sha256"
            ]
        else:
            rescued.pop("provider_capture_manifest_sha256", None)
        result["transcript_variants_json"] = json.dumps(rescued, ensure_ascii=False)
        result["name"] = "rescue"
        result["meta"] = {
            "provider": provider,
            "dual": dual,
        }
        return result

    def _candidate_from_call(self, call: CallRecord) -> Dict[str, Any]:
        payload = self._safe_json(call.transcript_variants_json or "")
        stored = payload.get("dialogue_lines")
        has_stored_lines = isinstance(stored, list) and any(str(line).strip() for line in stored)
        dialogue_lines = self._load_dialogue_lines_from_export(call)
        declared_source = str(payload.get("dialogue_lines_source") or "")
        dialogue_lines_source = (
            declared_source
            if declared_source in {"stored", "mutable_sidecar"}
            else "stored" if has_stored_lines else "mutable_sidecar" if dialogue_lines else "none"
        )
        meta = {
            "provider": "baseline",
            "dialogue_lines_source": dialogue_lines_source,
        }
        if dialogue_lines_source == "mutable_sidecar":
            meta["source_artifact_sha256"] = self._dialogue_lines_sha256(dialogue_lines)
        return {
            "name": "baseline",
            "transcript_manager": call.transcript_manager,
            "transcript_client": call.transcript_client,
            "transcript_text": call.transcript_text or "",
            "dialogue_lines": dialogue_lines,
            "transcript_variants_json": call.transcript_variants_json or "{}",
            "meta": meta,
        }

    @staticmethod
    def _choose_best(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        priority = {"llm": 3, "rescue": 2, "baseline": 1}
        return sorted(
            candidates,
            key=lambda item: (
                int(item.get("quality", {}).get("score", 0)),
                priority.get(str(item.get("name")), 0),
            ),
            reverse=True,
        )[0]

    def _build_resolve_payload(
        self,
        *,
        duration_sec: float,
        decision: str,
        baseline: Dict[str, Any],
        llm_candidate: Optional[Dict[str, Any]],
        rescue_candidate: Optional[Dict[str, Any]],
        chosen: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "version": "v1",
            "duration_sec": round(float(duration_sec), 3),
            "decision": decision,
            "baseline": {
                "score": int(baseline.get("quality", {}).get("score", 0)),
                "reasons": baseline.get("quality", {}).get("reasons", []),
            },
            "llm": None,
            "rescue": None,
            "chosen": None,
            "ts_utc": self._utc_now().isoformat(),
        }
        if llm_candidate:
            payload["llm"] = {
                "score": int(llm_candidate.get("quality", {}).get("score", 0)),
                "reasons": llm_candidate.get("quality", {}).get("reasons", []),
                "meta": llm_candidate.get("meta", {}),
            }
        if rescue_candidate:
            payload["rescue"] = {
                "score": int(rescue_candidate.get("quality", {}).get("score", 0)),
                "reasons": rescue_candidate.get("quality", {}).get("reasons", []),
                "meta": rescue_candidate.get("meta", {}),
            }
        if chosen:
            payload["chosen"] = {
                "name": chosen.get("name"),
                "score": int(chosen.get("quality", {}).get("score", 0)),
                "reasons": chosen.get("quality", {}).get("reasons", []),
                "meta": chosen.get("meta", {}),
            }
        return payload

    def _transition_resolve_claim(
        self,
        session: Session,
        *,
        call_id: int,
        worker_id: str,
        snapshot: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> bool:
        """Move the row only if the claim and the whole input never moved.

        Rescue ASR and the dialogue LLM can take minutes.  Reading the row and
        then writing it back is two statements: in between, the lease can expire
        and be re-claimed, or the transcript can be replaced by a secondary ASR
        backfill — and our own session may not even see that foreign commit.  So
        every outcome (done, manual, skipped, failed, waiting) is one
        conditional UPDATE, and the database compares both the lease and the
        full input snapshot.  ``rowcount != 1`` means the claim is stale and
        nothing of ours was written; the caller must not export a file either.
        """
        session.expunge_all()
        conditions = [
            CallRecord.id == int(call_id),
            CallRecord.resolve_status == "in_progress",
            CallRecord.pipeline_stage == "resolve",
            CallRecord.pipeline_worker_id == worker_id,
        ]
        conditions.extend(
            getattr(CallRecord, name) == snapshot.get(name)
            for name in RESOLVE_INPUT_COLUMNS
        )
        result = session.execute(
            sa_update(CallRecord)
            .where(*conditions)
            .values(**{**dict(values), "updated_at": self._utc_now()})
            .execution_options(synchronize_session=False)
        )
        return int(result.rowcount or 0) == 1

    def _transition_resolve_export_claim(
        self,
        session: Session,
        *,
        call_id: int,
        worker_id: str,
        source_call_id: Any,
        source_recording_id: Any,
        source_file: Any,
        resolve_json: Any,
        transcript_text: Any,
        transcript_variants_json: Any,
        release: bool,
    ) -> bool:
        """Refresh or release only the exact call/result owned by this worker."""
        session.expunge_all()
        now = self._utc_now()
        values = (
            {
                "pipeline_stage": None,
                "pipeline_worker_id": None,
                "pipeline_claimed_at": None,
                "updated_at": now,
            }
            if release
            else {"pipeline_claimed_at": now, "updated_at": now}
        )
        result = session.execute(
            sa_update(CallRecord)
            .where(
                CallRecord.id == int(call_id),
                CallRecord.resolve_status == "done",
                CallRecord.pipeline_stage == "resolve",
                CallRecord.pipeline_worker_id == worker_id,
                CallRecord.source_call_id == source_call_id,
                CallRecord.source_recording_id == source_recording_id,
                CallRecord.source_file == source_file,
                CallRecord.resolve_json == resolve_json,
                CallRecord.transcript_text == transcript_text,
                CallRecord.transcript_variants_json == transcript_variants_json,
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        return int(result.rowcount or 0) == 1

    def run(self, session: Session, limit: int) -> Dict[str, int]:
        return self.run_with_progress(session, limit=limit, progress_callback=None)

    def run_with_progress(
        self,
        session: Session,
        limit: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, int]:
        worker_id = self._pipeline_worker_id("rs")
        claimed_ids = self._claim_batch(session, limit=limit, worker_id=worker_id)
        max_attempts = max(1, self._settings.resolve_max_attempts)

        success = 0
        failed = 0
        manual = 0
        skipped = 0
        llm_used = 0
        rescue_used = 0
        handled = 0
        stale = 0
        export_failed = 0

        def _emit_progress(payload: Dict[str, Any]) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(payload)
            except Exception:
                return

        def _report(idx: int, *, outcome: str, call_id: int, error: str) -> None:
            _emit_progress(
                {
                    "stage": "resolve",
                    "current": idx,
                    "total": len(claimed_ids),
                    "success": success,
                    "failed": failed,
                    "manual": manual,
                    "skipped_short": skipped,
                    "llm_used": llm_used,
                    "rescue_used": rescue_used,
                    "stale": stale,
                    "export_failed": export_failed,
                    "status": outcome,
                    "call_id": call_id,
                    "error": error,
                }
            )

        _emit_progress(
            {
                "stage": "resolve",
                "current": 0,
                "total": len(claimed_ids),
                "success": 0,
                "failed": 0,
                "manual": 0,
                "skipped_short": 0,
                "llm_used": 0,
                "rescue_used": 0,
                "stale": 0,
                "export_failed": 0,
            }
        )

        for idx, call_id in enumerate(claimed_ids, start=1):
            call = session.get(CallRecord, call_id)
            if call is None:
                continue
            if call.resolve_status != "in_progress" or call.pipeline_stage != "resolve":
                continue
            scope = require_unique_controlled_call(session, self._settings)
            if scope and call.source_call_id != scope.source_call_id:
                raise RuntimeError("controlled_call_claim_identity_mismatch")
            # The exact stored input, read before any provider runs: the values
            # every conditional transition below compares against.
            snapshot = resolve_input_snapshot(call)

            if self._waiting_for_secondary_asr(call):
                wait_retry_sec = max(10, min(int(self._settings.worker_poll_sec or 10), 60))
                if self._transition_resolve_claim(
                    session,
                    call_id=call_id,
                    worker_id=worker_id,
                    snapshot=snapshot,
                    values={
                        "resolve_status": "pending",
                        "pipeline_stage": None,
                        "pipeline_worker_id": None,
                        "pipeline_claimed_at": None,
                        "next_retry_at": self._utc_now()
                        + timedelta(seconds=wait_retry_sec),
                    },
                ):
                    session.commit()
                else:
                    session.rollback()
                    stale += 1
                continue

            attempt = int(call.resolve_attempts or 0) + 1
            handled += 1
            outcome = "success"
            error_text = ""
            # Nothing is written to the ORM row: the whole result is collected
            # here and applied by one conditional UPDATE below.
            values: Dict[str, Any] = {"resolve_attempts": attempt}
            export_payload: Optional[Dict[str, Any]] = None
            counted = {"success": 0, "manual": 0}
            try:
                duration = float(call.duration_sec or 0.0)
                if duration > 0.0 and duration < float(self._settings.resolve_min_duration_sec):
                    values.update(
                        {
                            "resolve_status": "skipped",
                            "resolve_quality_score": 100.0,
                            "resolve_json": json.dumps(
                                {
                                    "version": "v1",
                                    "decision": "skip_short_call",
                                    "duration_sec": round(duration, 3),
                                    "min_duration_sec": int(
                                        self._settings.resolve_min_duration_sec
                                    ),
                                    "ts_utc": self._utc_now().isoformat(),
                                },
                                ensure_ascii=False,
                            ),
                            "analysis_status": "pending",
                            "sync_status": "pending",
                            "next_retry_at": None,
                            "last_error": None,
                            "pipeline_stage": None,
                            "pipeline_worker_id": None,
                            "pipeline_claimed_at": None,
                        }
                    )
                    outcome = "skipped_short"
                    if not self._transition_resolve_claim(
                        session,
                        call_id=call_id,
                        worker_id=worker_id,
                        snapshot=snapshot,
                        values=values,
                    ):
                        session.rollback()
                        stale += 1
                        continue
                    session.commit()
                    skipped += 1
                    success += 1
                    _report(
                        idx,
                        outcome=outcome,
                        call_id=call_id,
                        error=error_text,
                    )
                    continue

                baseline = self._candidate_from_call(call)
                baseline = self._maybe_postfilter_candidate_dialogue(call, baseline)
                baseline_payload = self._safe_json(str(baseline["transcript_variants_json"]))
                baseline["quality"] = self._score_candidate(
                    call,
                    str(baseline.get("transcript_text") or ""),
                    baseline.get("transcript_manager"),
                    baseline.get("transcript_client"),
                    baseline_payload,
                    dialogue_lines=baseline.get("dialogue_lines"),
                )

                llm_candidate: Optional[Dict[str, Any]] = None
                rescue_candidate: Optional[Dict[str, Any]] = None
                accept_threshold = int(self._settings.resolve_accept_score)
                llm_trigger = int(self._settings.resolve_llm_trigger_score)
                baseline_score = int(baseline["quality"]["score"])
                baseline_risky = self._is_ordering_risky(baseline) or self._is_payload_risky_for_llm(
                    baseline_payload,
                    baseline.get("quality") if isinstance(baseline.get("quality"), dict) else None,
                )
                llm_trigger_reason: Optional[str] = None

                if baseline_score < llm_trigger:
                    llm_trigger_reason = "low_score"
                elif self._settings.resolve_llm_for_risky and baseline_risky:
                    llm_trigger_reason = "risky_ordering_or_timing"

                if llm_trigger_reason is not None:
                    llm_candidate = self._resolve_with_llm(call, baseline_payload)
                    if llm_candidate is not None:
                        llm_candidate = self._maybe_postfilter_candidate_dialogue(call, llm_candidate)
                        llm_payload = self._safe_json(str(llm_candidate["transcript_variants_json"]))
                        llm_candidate["quality"] = self._score_candidate(
                            call,
                            str(llm_candidate.get("transcript_text") or ""),
                            llm_candidate.get("transcript_manager"),
                            llm_candidate.get("transcript_client"),
                            llm_payload,
                            dialogue_lines=llm_candidate.get("dialogue_lines"),
                        )
                        llm_meta = llm_candidate.get("meta")
                        if not isinstance(llm_meta, dict):
                            llm_meta = {}
                        llm_meta["trigger"] = llm_trigger_reason
                        llm_candidate["meta"] = llm_meta
                        llm_used += 1

                llm_score = int(llm_candidate.get("quality", {}).get("score", 0)) if llm_candidate else -1
                should_run_rescue = max(baseline_score, llm_score) < accept_threshold
                if (
                    not should_run_rescue
                    and self._settings.resolve_aggressive_rescue_for_risky
                    and self._is_ordering_risky(baseline, llm_candidate)
                ):
                    should_run_rescue = True

                if should_run_rescue:
                    rescue_candidate = self._run_rescue_asr(call)
                    if rescue_candidate is not None:
                        rescue_candidate = self._maybe_postfilter_candidate_dialogue(call, rescue_candidate)
                        rescue_payload = self._safe_json(
                            str(rescue_candidate.get("transcript_variants_json") or "{}")
                        )
                        rescue_candidate["quality"] = self._score_candidate(
                            call,
                            str(rescue_candidate.get("transcript_text") or ""),
                            rescue_candidate.get("transcript_manager"),
                            rescue_candidate.get("transcript_client"),
                            rescue_payload,
                            dialogue_lines=rescue_candidate.get("dialogue_lines"),
                        )
                        rescue_used += 1

                candidates = [baseline]
                if llm_candidate:
                    candidates.append(llm_candidate)
                if rescue_candidate:
                    candidates.append(rescue_candidate)
                best = self._choose_best(candidates)
                best_score = int(best.get("quality", {}).get("score", 0))
                best_name = str(best.get("name") or "baseline")
                if not self._candidate_source_is_current(call, best):
                    # The sidecar moved, but the database lease may still be
                    # ours.  Release that exact unchanged claim immediately;
                    # otherwise the call stays invisible until lease expiry.
                    if self._transition_resolve_claim(
                        session,
                        call_id=call_id,
                        worker_id=worker_id,
                        snapshot=snapshot,
                        values={
                            "resolve_status": "pending",
                            "pipeline_stage": None,
                            "pipeline_worker_id": None,
                            "pipeline_claimed_at": None,
                            "next_retry_at": None,
                        },
                    ):
                        session.commit()
                    else:
                        session.rollback()
                    stale += 1
                    continue
                if best_score >= accept_threshold:
                    if best_name != "baseline":
                        values["transcript_manager"] = best.get("transcript_manager")
                        values["transcript_client"] = best.get("transcript_client")
                        values["transcript_text"] = str(best.get("transcript_text") or "")
                    if isinstance(best.get("transcript_variants_json"), str):
                        values["transcript_variants_json"] = str(
                            best.get("transcript_variants_json") or "{}"
                        )
                    if best_name != "baseline":
                        # The exported file is a projection of the committed row.
                        # It is written only after the conditional UPDATE proves
                        # the row is still ours — a stale worker must not leave a
                        # transcript file describing a result nobody stored.
                        export_payload = {
                            "transcript_manager": values.get(
                                "transcript_manager", snapshot["transcript_manager"]
                            ),
                            "transcript_client": values.get(
                                "transcript_client", snapshot["transcript_client"]
                            ),
                            "transcript_text": values.get(
                                "transcript_text", snapshot["transcript_text"]
                            )
                            or "",
                            "dialogue_lines": best.get("dialogue_lines"),
                            "transcript_variants_json": values.get(
                                "transcript_variants_json",
                                snapshot["transcript_variants_json"],
                            )
                            or "{}",
                        }
                    decision = f"accept_{best_name}"
                    values.update(
                        {
                            "resolve_status": "done",
                            "analysis_status": "pending",
                            "sync_status": "pending",
                        }
                    )
                    counted["success"] = 1
                    outcome = "done"
                else:
                    decision = "manual_review_required"
                    values["resolve_status"] = "manual"
                    counted["manual"] = 1
                    outcome = "manual"

                values.update(
                    {
                        "resolve_quality_score": float(best_score),
                        "resolve_json": json.dumps(
                            self._build_resolve_payload(
                                duration_sec=duration,
                                decision=decision,
                                baseline=baseline,
                                llm_candidate=llm_candidate,
                                rescue_candidate=rescue_candidate,
                                chosen=best,
                            ),
                            ensure_ascii=False,
                        ),
                        "next_retry_at": None,
                        "dead_letter_stage": None,
                        "last_error": None,
                    }
                )
                if export_payload is None:
                    values.update(
                        {
                            "pipeline_stage": None,
                            "pipeline_worker_id": None,
                            "pipeline_claimed_at": None,
                        }
                    )
                else:
                    # Resolve may have spent most of the lease in ASR/LLM.  The
                    # committed done row needs a fresh lease for its file export.
                    values["pipeline_claimed_at"] = self._utc_now()
                if not self._transition_resolve_claim(
                    session,
                    call_id=call_id,
                    worker_id=worker_id,
                    snapshot=snapshot,
                    values=values,
                ):
                    # Somebody else owns this row now: leave it exactly as it is
                    # and export nothing.
                    session.rollback()
                    stale += 1
                    continue
                session.commit()
                success += counted["success"]
                manual += counted["manual"]
            except Exception as exc:  # noqa: BLE001
                session.rollback()
                dead = attempt >= max_attempts
                if not self._transition_resolve_claim(
                    session,
                    call_id=call_id,
                    worker_id=worker_id,
                    snapshot=snapshot,
                    values={
                        "resolve_attempts": attempt,
                        "resolve_status": "dead" if dead else "failed",
                        "dead_letter_stage": "resolve" if dead else None,
                        "next_retry_at": (
                            None if dead else self._utc_now() + self._retry_delay(attempt)
                        ),
                        "last_error": safe_error_text("resolve", exc),
                        "pipeline_stage": None,
                        "pipeline_worker_id": None,
                        "pipeline_claimed_at": None,
                    },
                ):
                    session.rollback()
                    stale += 1
                    continue
                session.commit()
                failed += 1
                outcome = "failed"
                error_text = safe_error_text("resolve", exc)
                export_payload = None
            if export_payload is not None:
                export_claim = {
                    "source_call_id": snapshot["source_call_id"],
                    "source_recording_id": snapshot["source_recording_id"],
                    "source_file": snapshot["source_file"],
                    "resolve_json": values["resolve_json"],
                    "transcript_text": values.get(
                        "transcript_text", snapshot["transcript_text"]
                    ),
                    "transcript_variants_json": values.get(
                        "transcript_variants_json",
                        snapshot["transcript_variants_json"],
                    ),
                }
                # This UPDATE both validates the immutable export path and holds
                # the row write-lock until the atomic local file write finishes.
                if not self._transition_resolve_export_claim(
                    session,
                    call_id=call_id,
                    worker_id=worker_id,
                    **export_claim,
                    release=False,
                ):
                    session.rollback()
                    stale += 1
                else:
                    committed = session.get(CallRecord, call_id, populate_existing=True)
                    try:
                        self._transcribe_helper._export_transcript_file(
                            committed, export_payload
                        )
                    except Exception:  # noqa: BLE001
                        export_failed += 1
                    finally:
                        released = self._transition_resolve_export_claim(
                            session,
                            call_id=call_id,
                            worker_id=worker_id,
                            **export_claim,
                            release=True,
                        )
                        session.commit() if released else session.rollback()
                        if not released:
                            stale += 1
            _report(
                idx,
                outcome=outcome,
                call_id=call_id,
                error=error_text,
            )
        return {
            "processed": handled,
            "success": success,
            "failed": failed,
            "manual": manual,
            "skipped_short": skipped,
            "llm_used": llm_used,
            "rescue_used": rescue_used,
            "stale": stale,
            "export_failed": export_failed,
            "worker_id": worker_id,
        }

    def export_manual_review_queue(
        self,
        session: Session,
        *,
        out_path: Path,
        limit: int,
    ) -> Dict[str, Any]:
        calls = session.scalars(
            select(CallRecord)
            .where(CallRecord.resolve_status == "manual")
            .order_by(CallRecord.resolve_quality_score.asc(), CallRecord.id.asc())
            .limit(limit)
        ).all()
        rows: List[Dict[str, Any]] = []
        for call in calls:
            payload = self._safe_json(call.resolve_json or "")
            chosen = payload.get("chosen") if isinstance(payload.get("chosen"), dict) else {}
            rows.append(
                {
                    "id": call.id,
                    "source_filename": call.source_filename,
                    "source_file": call.source_file,
                    "manager_name": call.manager_name,
                    "phone": call.phone,
                    "duration_sec": round(float(call.duration_sec or 0.0), 3),
                    "resolve_quality_score": call.resolve_quality_score,
                    "decision": payload.get("decision"),
                    "chosen_name": chosen.get("name"),
                    "chosen_score": chosen.get("score"),
                    "reasons": "; ".join(chosen.get("reasons") or []),
                    "last_error": call.last_error,
                }
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()
        if suffix == ".csv":
            headers = [
                "id",
                "source_filename",
                "source_file",
                "manager_name",
                "phone",
                "duration_sec",
                "resolve_quality_score",
                "decision",
                "chosen_name",
                "chosen_score",
                "reasons",
                "last_error",
            ]
            with out_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
        elif suffix == ".jsonl":
            with out_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"exported": len(rows), "out": str(out_path.resolve())}

    def export_failed_resolve_queue(
        self,
        session: Session,
        *,
        out_path: Path,
        limit: int,
    ) -> Dict[str, Any]:
        calls = session.scalars(
            select(CallRecord)
            .where(CallRecord.resolve_status.in_(["failed", "dead"]))
            .order_by(CallRecord.resolve_status.asc(), CallRecord.id.asc())
            .limit(limit)
        ).all()
        rows: List[Dict[str, Any]] = []
        for call in calls:
            rows.append(
                {
                    "id": call.id,
                    "source_filename": call.source_filename,
                    "source_file": call.source_file,
                    "manager_name": call.manager_name,
                    "phone": call.phone,
                    "duration_sec": round(float(call.duration_sec or 0.0), 3),
                    "resolve_status": call.resolve_status,
                    "resolve_attempts": int(call.resolve_attempts or 0),
                    "next_retry_at": call.next_retry_at.isoformat() if call.next_retry_at else None,
                    "dead_letter_stage": call.dead_letter_stage,
                    "last_error": call.last_error,
                }
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()
        if suffix == ".csv":
            headers = [
                "id",
                "source_filename",
                "source_file",
                "manager_name",
                "phone",
                "duration_sec",
                "resolve_status",
                "resolve_attempts",
                "next_retry_at",
                "dead_letter_stage",
                "last_error",
            ]
            with out_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(rows)
        elif suffix == ".jsonl":
            with out_path.open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

        return {"exported": len(rows), "out": str(out_path.resolve())}
