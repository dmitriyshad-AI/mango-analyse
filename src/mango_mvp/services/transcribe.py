from __future__ import annotations

import difflib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from openai import OpenAI
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from mango_mvp.clients.ollama import OllamaClient
from mango_mvp.config import Settings
from mango_mvp.models import CallRecord
from mango_mvp.utils.audio import split_stereo_to_mono

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)
MERGE_ALLOWED_PROVIDERS = {"primary", "rule", "openai", "ollama", "codex_cli"}
ROLE_ASSIGN_ALLOWED_MODES = {"off", "rule", "openai_selective", "ollama_selective"}
MERGE_SYSTEM_PROMPT = """You merge two ASR transcript variants for the same speaker in one phone call.
Rules:
1) Use only information from variants A and B. Do not invent facts.
2) Keep chronology and intent. Remove obvious ASR loops and garbage.
3) If uncertain, prefer A.
4) Return strict JSON with keys:
   - merged_text (string)
   - selection (one of: A, B, MIX)
   - confidence (number 0..1)
   - notes (string)
No markdown, no extra keys."""
CODEX_MERGE_PROMPT_TEMPLATE = """You merge two ASR transcript variants for the same speaker in one phone call.
Rules:
1) Use only information from variants A and B. Do not invent facts.
2) Keep chronology and intent. Remove obvious ASR loops and garbage.
3) If uncertain, prefer A.
4) Return strict JSON with keys:
   - merged_text (string)
   - selection (one of: A, B, MIX)
   - confidence (number 0..1)
   - notes (string)
No markdown, no extra keys.

Speaker: {speaker_label}

Variant A:
{variant_a}

Variant B:
{variant_b}
"""
ROLE_ASSIGN_SYSTEM_PROMPT = """You classify each utterance in one Russian sales phone call as either manager or client.
Rules:
1) Keep chronological order and number of turns unchanged.
2) Use only provided text, do not invent content.
3) Return strict JSON with keys:
   - roles: array of strings (each exactly "manager" or "client")
   - confidence: number 0..1
   - notes: string
No markdown, no extra keys."""

MANAGER_CUES: dict[str, float] = {
    "учебный центр": 3.0,
    "вас беспокоит": 3.0,
    "меня зовут": 2.5,
    "подскажите": 1.8,
    "вам удобно": 1.8,
    "заявк": 1.6,
    "курс": 1.4,
    "занят": 1.3,
    "скидк": 1.2,
    "оплат": 1.1,
    "приглашени": 1.1,
    "добрый день": 0.8,
}
CLIENT_CUES: dict[str, float] = {
    "сколько стоит": 2.6,
    "можно": 1.6,
    "интересует": 1.6,
    "я оплачив": 1.5,
    "я не поняла": 1.5,
    "не поняла": 1.4,
    "а у вас": 1.3,
    "пришлите": 1.2,
    "на почту": 1.2,
    "на телефон": 1.2,
    "спасибо": 0.6,
}
ARTIFACT_ONLY_PHRASES = {
    "продолжение следует",
}
SECONDARY_BACKFILL_MAX_ATTEMPTS = 2


class TranscribeService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._client: Optional[OpenAI] = None
        self._ollama_client_instance: Optional[OllamaClient] = None
        self._gigaam_model: Any = None

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _retry_delay(self, attempts: int) -> timedelta:
        base = max(1, self._settings.retry_base_delay_sec)
        multiplier = max(1, 2 ** max(0, attempts - 1))
        return timedelta(seconds=base * multiplier)

    @staticmethod
    def _safe_json_dict(raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, str):
            return {}
        text = raw.strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _secondary_backfill_meta(payload: Dict[str, Any]) -> Dict[str, Any]:
        meta = payload.get("secondary_backfill_meta")
        return meta if isinstance(meta, dict) else {}

    def _secondary_backfill_attempts(
        self,
        payload: Dict[str, Any],
        *,
        secondary_provider: str,
    ) -> int:
        meta = self._secondary_backfill_meta(payload)
        if str(meta.get("provider") or "").strip() == secondary_provider:
            try:
                return max(0, int(meta.get("attempts", 0) or 0))
            except (TypeError, ValueError):
                return 0

        cached_secondary = str(payload.get("secondary_provider") or "").strip()
        if cached_secondary != secondary_provider:
            return 0
        # Legacy payload without retry metadata: if the target secondary provider
        # is already present but slot(s) are still incomplete, treat it as one
        # attempt already spent so we only allow one extra retry.
        return 1

    def _secondary_backfill_state_from_payload(
        self,
        payload: Dict[str, Any],
        *,
        secondary_provider: str,
    ) -> str:
        if not payload:
            return "not_needed"

        meta = self._secondary_backfill_meta(payload)
        if (
            str(meta.get("provider") or "").strip() == secondary_provider
            and bool(meta.get("exhausted"))
        ):
            return "exhausted"

        mode = str(payload.get("mode") or "").strip()
        cached_secondary = str(payload.get("secondary_provider") or "").strip()
        secondary_matches = cached_secondary == secondary_provider

        def _slot_has_primary(slot: str) -> bool:
            block = payload.get(slot)
            if not isinstance(block, dict):
                return False
            return bool(str(block.get("variant_a") or "").strip())

        def _slot_missing(slot: str) -> bool:
            if not _slot_has_primary(slot):
                return False
            block = payload.get(slot)
            if not isinstance(block, dict):
                return False
            secondary_text = str(block.get("variant_b") or "").strip()
            if not secondary_matches:
                return True
            return not bool(secondary_text)

        needs_backfill = False
        if mode == "stereo":
            if _slot_has_primary("manager") and _slot_has_primary("client"):
                needs_backfill = _slot_missing("manager") or _slot_missing("client")
        elif mode == "mono_or_fallback":
            if _slot_has_primary("full"):
                needs_backfill = _slot_missing("full")
        if not needs_backfill:
            return "not_needed"
        if secondary_matches:
            return "retry"
        return "fresh"

    def _apply_secondary_backfill_meta(
        self,
        payload: Dict[str, Any],
        *,
        secondary_provider: str,
        attempts: int,
        status: str,
        exhausted: bool,
        error: str = "",
    ) -> Dict[str, Any]:
        updated = dict(payload)
        updated["secondary_backfill_meta"] = {
            "provider": secondary_provider,
            "attempts": int(max(0, attempts)),
            "status": status,
            "exhausted": bool(exhausted),
            "last_error": error.strip(),
            "last_attempt_utc": self._utc_now().isoformat(),
        }
        return updated

    @staticmethod
    def _clear_secondary_backfill_meta(
        payload: Dict[str, Any],
        *,
        secondary_provider: str,
    ) -> Dict[str, Any]:
        updated = dict(payload)
        meta = updated.get("secondary_backfill_meta")
        if not isinstance(meta, dict):
            return updated
        if str(meta.get("provider") or "").strip() != secondary_provider:
            return updated
        updated.pop("secondary_backfill_meta", None)
        return updated

    @staticmethod
    def _variant_text_for_provider(
        block: Any,
        *,
        provider: str,
        cached_primary_provider: str,
        cached_secondary_provider: str,
        fallback_to_variant_a: bool = False,
    ) -> str:
        if not isinstance(block, dict):
            return ""
        if provider and provider == cached_primary_provider:
            return str(block.get("variant_a") or "").strip()
        if provider and provider == cached_secondary_provider:
            return str(block.get("variant_b") or "").strip()
        if (
            fallback_to_variant_a
            and not str(block.get("variant_b") or "").strip()
            and str(block.get("variant_a") or "").strip()
        ):
            return str(block.get("variant_a") or "").strip()
        return ""

    def _cached_variant_candidate(
        self,
        call: CallRecord,
        *,
        slot: str,  # manager | client | full
        provider: str,
        primary_provider: str,
    ) -> Optional[Dict[str, Any]]:
        payload = self._safe_json_dict(call.transcript_variants_json)
        if not payload:
            return None

        mode = str(payload.get("mode") or "").strip()
        if slot in {"manager", "client"} and mode != "stereo":
            return None
        if slot == "full" and mode != "mono_or_fallback":
            return None

        block = payload.get(slot)
        cached_primary_provider = str(payload.get("primary_provider") or "").strip()
        cached_secondary_provider = str(payload.get("secondary_provider") or "").strip()
        cached_text = self._variant_text_for_provider(
            block,
            provider=provider,
            cached_primary_provider=cached_primary_provider,
            cached_secondary_provider=cached_secondary_provider,
            fallback_to_variant_a=(provider == primary_provider),
        )
        if not cached_text:
            return None
        return {
            "text": cached_text,
            "segments": None,
            "error": None,
            "cached": True,
        }

    def _openai_client(self) -> OpenAI:
        if not self._settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for openai-based providers")
        if self._client is None:
            self._client = OpenAI(api_key=self._settings.openai_api_key)
        return self._client

    def _ollama_client(self) -> OllamaClient:
        if self._ollama_client_instance is None:
            self._ollama_client_instance = OllamaClient(self._settings.ollama_base_url)
        return self._ollama_client_instance

    def _get_gigaam_model(self) -> Any:
        if self._gigaam_model is not None:
            return self._gigaam_model
        try:
            from gigaam import load_model
        except ImportError as exc:
            raise RuntimeError(
                "gigaam is not installed in current Python environment. "
                "Use Python >=3.10 and install gigaam to enable SECONDARY_TRANSCRIBE_PROVIDER=gigaam."
            ) from exc

        self._gigaam_model = load_model(
            self._settings.gigaam_model,
            device=self._settings.gigaam_device,
            fp16_encoder=False,
        )
        return self._gigaam_model

    @staticmethod
    def _looks_like_phone(value: str) -> bool:
        normalized = re.sub(r"\s+", "", value)
        return bool(re.fullmatch(r"\+?\d{7,15}", normalized))

    def _extract_manager_name_from_filename(self, source_filename: str) -> str:
        stem = Path(source_filename).stem
        parts = stem.split("__")
        if len(parts) < 4:
            return "Неизвестный менеджер"

        left = parts[2].strip()
        right = parts[3].strip()
        if "_" in right:
            right = right.rsplit("_", 1)[0].strip()

        for candidate in (left, right):
            if candidate and not self._looks_like_phone(candidate):
                return candidate
        return "Неизвестный менеджер"

    def _segment_words_to_turns(
        self,
        segment: Dict[str, Any],
    ) -> list[tuple[float, str]]:
        words = segment.get("words")
        if not isinstance(words, list):
            return []

        turns: list[tuple[float, str]] = []
        current_tokens: list[str] = []
        current_start: Optional[float] = None
        last_end: Optional[float] = None

        for word in words:
            if not isinstance(word, dict):
                continue
            token = str(word.get("word", "")).strip()
            if not token:
                continue

            start_raw = word.get("start")
            end_raw = word.get("end")
            try:
                start = float(start_raw) if start_raw is not None else None
            except (TypeError, ValueError):
                start = None
            try:
                end = float(end_raw) if end_raw is not None else None
            except (TypeError, ValueError):
                end = None

            if start is None:
                if last_end is not None:
                    start = last_end
                elif current_start is not None:
                    start = current_start
                else:
                    start = float(segment.get("start") or 0.0)
            if end is None:
                end = start

            gap = (
                float(start) - float(last_end)
                if last_end is not None
                else 0.0
            )
            should_break = (
                bool(current_tokens)
                and (
                    gap > 0.85
                    or len(current_tokens) >= 24
                    or (
                        current_tokens[-1].endswith((".", "!", "?"))
                        and len(current_tokens) >= 4
                    )
                )
            )
            if should_break and current_start is not None:
                raw_text = " ".join(current_tokens)
                text = self._detokenize(self._tokenize(raw_text))
                if text:
                    turns.append((max(0.0, float(current_start)), text))
                current_tokens = []
                current_start = None

            if current_start is None:
                current_start = max(0.0, float(start))
            current_tokens.append(token)
            last_end = float(end)

        if current_tokens and current_start is not None:
            raw_text = " ".join(current_tokens)
            text = self._detokenize(self._tokenize(raw_text))
            if text:
                turns.append((max(0.0, float(current_start)), text))

        return turns

    def _segments_to_timeline(
        self, raw_segments: Any, speaker: str
    ) -> list[tuple[float, int, str, str]]:
        if not isinstance(raw_segments, list):
            return []
        timeline: list[tuple[float, int, str, str]] = []
        order = 0
        for idx, segment in enumerate(raw_segments):
            if not isinstance(segment, dict):
                continue
            word_turns = self._segment_words_to_turns(segment)
            if word_turns:
                for start, text in word_turns:
                    timeline.append((max(0.0, start), order, speaker, text))
                    order += 1
                continue
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            start_raw = segment.get("start")
            try:
                start = float(start_raw)
            except (TypeError, ValueError):
                continue
            timeline.append((max(0.0, start), 10_000 + idx, speaker, " ".join(text.split())))
        return timeline

    @staticmethod
    def _format_timecode(seconds: float, approximate: bool = False) -> str:
        total_ms = max(0, int(seconds * 1000.0))
        hours, rem_ms = divmod(total_ms, 3_600_000)
        minutes, rem_ms = divmod(rem_ms, 60_000)
        secs, ms = divmod(rem_ms, 1000)
        prefix = "~" if approximate else ""
        if hours > 0:
            if approximate:
                return f"[{prefix}{hours:02d}:{minutes:02d}:{secs:02d}]"
            return f"[{prefix}{hours:02d}:{minutes:02d}:{secs:02d}.{ms // 100}]"
        if approximate:
            return f"[{prefix}{minutes:02d}:{secs:02d}]"
        return f"[{prefix}{minutes:02d}:{secs:02d}.{ms // 100}]"

    @staticmethod
    def _estimate_turn_starts(turns_count: int, call_duration_sec: Optional[float]) -> list[float]:
        if turns_count <= 0:
            return []
        if call_duration_sec is not None and call_duration_sec > 0:
            span = max(float(call_duration_sec), float(turns_count))
            step = span / turns_count
            return [idx * step for idx in range(turns_count)]
        return [idx * 4.0 for idx in range(turns_count)]

    def _build_dialogue_lines(
        self,
        manager_name: str,
        manager_segments: Any,
        client_segments: Any,
        manager_fallback_text: str = "",
        client_fallback_text: str = "",
        call_duration_sec: Optional[float] = None,
    ) -> list[str]:
        timeline = self._segments_to_timeline(
            manager_segments, f"Менеджер ({manager_name})"
        ) + self._segments_to_timeline(client_segments, "Клиент")
        if timeline:
            timeline.sort(key=lambda x: (x[0], x[1]))
            lines: list[str] = []
            prev_start = 0.0
            for start, _, speaker, text in timeline:
                safe_start = max(prev_start, float(start))
                lines.append(f"{self._format_timecode(safe_start)} {speaker}: {text}")
                prev_start = safe_start
            return lines

        manager_sentences = self._split_sentences(manager_fallback_text)
        client_sentences = self._split_sentences(client_fallback_text)
        if not manager_sentences and not client_sentences:
            return []

        turns: list[tuple[str, str]] = []
        i = 0
        j = 0
        manager_turn = True
        while i < len(manager_sentences) or j < len(client_sentences):
            if manager_turn and i < len(manager_sentences):
                turns.append((f"Менеджер ({manager_name})", manager_sentences[i]))
                i += 1
            elif (not manager_turn) and j < len(client_sentences):
                turns.append(("Клиент", client_sentences[j]))
                j += 1
            elif i < len(manager_sentences):
                turns.append((f"Менеджер ({manager_name})", manager_sentences[i]))
                i += 1
            elif j < len(client_sentences):
                turns.append(("Клиент", client_sentences[j]))
                j += 1
            manager_turn = not manager_turn
        starts = self._estimate_turn_starts(len(turns), call_duration_sec)
        return [
            f"{self._format_timecode(starts[idx], approximate=True)} {speaker}: {text}"
            for idx, (speaker, text) in enumerate(turns)
        ]

    @staticmethod
    def _timecode_to_seconds(token: str) -> float:
        raw = (token or "").strip()
        if raw.startswith("~"):
            raw = raw[1:]
        parts = raw.split(":")
        try:
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return float(minutes * 60) + seconds
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return float(hours * 3600 + minutes * 60) + seconds
        except (TypeError, ValueError):
            return 0.0
        return 0.0

    def _parse_dialogue_line(self, line: str) -> Optional[Dict[str, Any]]:
        match = re.match(r"^\[(?P<time>[^\]]+)\]\s+(?P<speaker>[^:]+):\s*(?P<text>.*)$", line.strip())
        if not match:
            return None
        speaker = match.group("speaker").strip()
        text = match.group("text").strip()
        if speaker.startswith("Менеджер"):
            role = "manager"
        elif speaker == "Клиент":
            role = "client"
        else:
            role = "other"
        return {
            "timecode": match.group("time"),
            "start": self._timecode_to_seconds(match.group("time")),
            "speaker": speaker,
            "role": role,
            "text": text,
            "line": line.strip(),
        }

    def _role_text_fit_score(self, role: str, text: str) -> float:
        manager_score, client_score = self._score_role_cues(text)
        if role == "manager":
            cue_fit = manager_score - client_score
        elif role == "client":
            cue_fit = client_score - manager_score
        else:
            cue_fit = 0.0
        tokenized = self._tokenize(text)
        word_count = sum(1 for token in tokenized if WORD_RE.fullmatch(token))
        chunk_penalty, _, _ = self._chunk_quality(tokenized)
        length_bonus = min(2.5, word_count / 12.0)
        return cue_fit * 1.5 + length_bonus - chunk_penalty

    def _pick_crosstalk_drop_index(
        self,
        left: Dict[str, Any],
        right: Dict[str, Any],
    ) -> int:
        left_score = self._role_text_fit_score(str(left.get("role", "")), str(left.get("text", "")))
        right_score = self._role_text_fit_score(str(right.get("role", "")), str(right.get("text", "")))
        if abs(left_score - right_score) >= 0.2:
            return int(right["index"]) if left_score > right_score else int(left["index"])

        left_start = float(left.get("start", 0.0))
        right_start = float(right.get("start", 0.0))
        if abs(left_start - right_start) >= 0.15:
            return int(right["index"]) if left_start < right_start else int(left["index"])

        left_len = len(str(left.get("text", "")))
        right_len = len(str(right.get("text", "")))
        if left_len != right_len:
            return int(right["index"]) if left_len > right_len else int(left["index"])
        return int(right["index"])

    def _dedupe_stereo_cross_talk(self, dialogue_lines: list[str]) -> Dict[str, Any]:
        if not dialogue_lines:
            return {
                "dialogue_lines": dialogue_lines,
                "manager_text": "",
                "client_text": "",
                "dropped": 0,
                "pairs_checked": 0,
            }

        parsed: list[Dict[str, Any]] = []
        for idx, line in enumerate(dialogue_lines):
            item = self._parse_dialogue_line(line)
            if item is None:
                item = {
                    "timecode": "",
                    "start": float(idx),
                    "speaker": "",
                    "role": "other",
                    "text": "",
                    "line": str(line).strip(),
                }
            item["index"] = idx
            parsed.append(item)

        keep = [True] * len(parsed)
        dropped = 0
        pairs_checked = 0
        max_time_diff_sec = 1.6
        min_chars = 12
        similarity_threshold = 0.92

        for i, left in enumerate(parsed):
            if not keep[i]:
                continue
            left_role = str(left.get("role", ""))
            if left_role not in {"manager", "client"}:
                continue
            left_start = float(left.get("start", 0.0))
            left_text = str(left.get("text", "")).strip()
            if len(left_text) < min_chars:
                continue
            for j in range(i + 1, min(len(parsed), i + 4)):
                if not keep[j]:
                    continue
                right = parsed[j]
                right_role = str(right.get("role", ""))
                if right_role not in {"manager", "client"}:
                    continue
                if left_role == right_role:
                    continue
                right_start = float(right.get("start", 0.0))
                if abs(right_start - left_start) > max_time_diff_sec:
                    if right_start > left_start:
                        break
                    continue
                right_text = str(right.get("text", "")).strip()
                if len(right_text) < min_chars:
                    continue
                pairs_checked += 1
                if self._similarity_ratio(left_text, right_text) < similarity_threshold:
                    continue
                drop_idx = self._pick_crosstalk_drop_index(left, right)
                if keep[drop_idx]:
                    keep[drop_idx] = False
                    dropped += 1
                if drop_idx == i:
                    break

        if dropped == 0:
            return {
                "dialogue_lines": dialogue_lines,
                "manager_text": "",
                "client_text": "",
                "dropped": 0,
                "pairs_checked": pairs_checked,
            }

        cleaned_lines = [parsed[idx]["line"] for idx, flag in enumerate(keep) if flag]
        manager_parts: list[str] = []
        client_parts: list[str] = []
        for idx, flag in enumerate(keep):
            if not flag:
                continue
            role = str(parsed[idx].get("role", ""))
            text = str(parsed[idx].get("text", "")).strip()
            if not text:
                continue
            if role == "manager":
                manager_parts.append(text)
            elif role == "client":
                client_parts.append(text)

        return {
            "dialogue_lines": cleaned_lines,
            "manager_text": " ".join(manager_parts).strip(),
            "client_text": " ".join(client_parts).strip(),
            "dropped": dropped,
            "pairs_checked": pairs_checked,
        }

    @staticmethod
    def _normalize_artifact_text(text: str) -> str:
        lowered = (text or "").strip().lower().replace("ё", "е")
        lowered = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
        return " ".join(lowered.split())

    def _is_artifact_only_text(self, text: str) -> bool:
        normalized = self._normalize_artifact_text(text)
        if not normalized:
            return False
        tokens = normalized.split()
        if len(tokens) > 5:
            return False
        return normalized in ARTIFACT_ONLY_PHRASES

    def _drop_artifact_only_lines(self, dialogue_lines: list[str]) -> Dict[str, Any]:
        if not dialogue_lines:
            return {"dialogue_lines": dialogue_lines, "dropped": 0}

        out_lines: list[str] = []
        dropped = 0
        for line in dialogue_lines:
            parsed = self._parse_dialogue_line(line)
            if parsed is not None and self._is_artifact_only_text(str(parsed.get("text", ""))):
                dropped += 1
                continue
            out_lines.append(line)
        return {"dialogue_lines": out_lines, "dropped": dropped}

    def _dedupe_adjacent_cross_speaker_echo(self, dialogue_lines: list[str]) -> Dict[str, Any]:
        if len(dialogue_lines) < 2:
            return {"dialogue_lines": dialogue_lines, "dropped": 0, "pairs_checked": 0}

        parsed: list[Optional[Dict[str, Any]]] = [
            self._parse_dialogue_line(line) for line in dialogue_lines
        ]
        keep = [True] * len(dialogue_lines)
        dropped = 0
        checked = 0
        for idx in range(len(parsed) - 1):
            if not keep[idx]:
                continue
            left = parsed[idx]
            right = parsed[idx + 1]
            if left is None or right is None:
                continue
            left_role = str(left.get("role", ""))
            right_role = str(right.get("role", ""))
            if left_role not in {"manager", "client"} or right_role not in {"manager", "client"}:
                continue
            if left_role == right_role:
                continue

            left_text = str(left.get("text", "")).strip()
            right_text = str(right.get("text", "")).strip()
            if min(len(left_text), len(right_text)) < 12:
                continue

            left_start = float(left.get("start", 0.0))
            right_start = float(right.get("start", 0.0))
            if abs(right_start - left_start) > 1.6:
                continue

            checked += 1
            if self._similarity_ratio(left_text, right_text) < 0.95:
                continue

            left_item = dict(left)
            right_item = dict(right)
            left_item["index"] = idx
            right_item["index"] = idx + 1
            drop_idx = self._pick_crosstalk_drop_index(left_item, right_item)
            if keep[drop_idx]:
                keep[drop_idx] = False
                dropped += 1

        cleaned_lines = [dialogue_lines[idx] for idx, flag in enumerate(keep) if flag]
        return {
            "dialogue_lines": cleaned_lines,
            "dropped": dropped,
            "pairs_checked": checked,
        }

    def _rebuild_role_texts_from_dialogue_lines(self, dialogue_lines: list[str]) -> tuple[str, str]:
        manager_parts: list[str] = []
        client_parts: list[str] = []
        for line in dialogue_lines:
            parsed = self._parse_dialogue_line(line)
            if parsed is None:
                continue
            text = str(parsed.get("text", "")).strip()
            if not text:
                continue
            role = str(parsed.get("role", ""))
            if role == "manager":
                manager_parts.append(text)
            elif role == "client":
                client_parts.append(text)
        return " ".join(manager_parts).strip(), " ".join(client_parts).strip()

    @staticmethod
    def _is_question_like(text: str) -> bool:
        lowered = (text or "").lower()
        if "?" in lowered:
            return True
        question_cues = (
            "подскажите",
            "какой",
            "какая",
            "какие",
            "когда",
            "сколько",
            "можно",
            "можете",
            "интересует",
            "удобно",
        )
        return any(cue in lowered for cue in question_cues)

    @staticmethod
    def _is_short_answer_like(text: str) -> bool:
        compact = " ".join((text or "").lower().split())
        if not compact or "?" in compact:
            return False
        tokens = WORD_RE.findall(compact)
        if not tokens:
            return False
        if len(tokens) > 6:
            return False
        answer_starts = (
            "да",
            "нет",
            "угу",
            "ага",
            "хорошо",
            "понятно",
            "спасибо",
            "девятый",
            "десятый",
            "одиннадцатый",
        )
        if any(compact.startswith(prefix) for prefix in answer_starts):
            return True
        return any(token.isdigit() for token in tokens)

    def _should_swap_adjacent_dialogue_turns(
        self,
        left: Dict[str, Any],
        right: Dict[str, Any],
    ) -> bool:
        left_role = str(left.get("role", ""))
        right_role = str(right.get("role", ""))
        if left_role == right_role:
            return False
        if left_role not in {"manager", "client"} or right_role not in {"manager", "client"}:
            return False

        left_text = str(left.get("text", "")).strip()
        right_text = str(right.get("text", "")).strip()
        if not left_text or not right_text:
            return False

        left_start = float(left.get("start", 0.0))
        right_start = float(right.get("start", 0.0))
        if abs(right_start - left_start) > 0.9:
            return False
        # Never swap when the right turn is clearly later; this keeps timeline monotonic.
        if right_start - left_start > 0.05:
            return False

        left_question = self._is_question_like(left_text)
        right_question = self._is_question_like(right_text)
        left_short_answer = self._is_short_answer_like(left_text)
        right_short_answer = self._is_short_answer_like(right_text)

        # Typical swap pattern:
        # client short answer appears before manager question because timestamps are nearly equal.
        if left_role == "client" and right_role == "manager":
            if left_short_answer and right_question:
                return True
            if left_short_answer and right_text.lower().startswith("а подскажите"):
                return True

        # Reverse case for inbound questions:
        # manager short answer accidentally appears before client question.
        if left_role == "manager" and right_role == "client":
            if left_short_answer and right_question:
                return True
            if left_short_answer and right_text.lower().startswith("а подскажите"):
                return True

        return False

    def _resequence_dialogue_lines(self, dialogue_lines: list[str]) -> Dict[str, Any]:
        if len(dialogue_lines) < 2:
            return {"dialogue_lines": dialogue_lines, "swapped": 0}

        parsed: list[Optional[Dict[str, Any]]] = [
            self._parse_dialogue_line(line) for line in dialogue_lines
        ]
        lines = dialogue_lines[:]
        swapped = 0
        max_passes = 2
        for _ in range(max_passes):
            changed = False
            idx = 0
            while idx < len(lines) - 1:
                left = parsed[idx]
                right = parsed[idx + 1]
                if left is None or right is None:
                    idx += 1
                    continue
                if self._should_swap_adjacent_dialogue_turns(left, right):
                    lines[idx], lines[idx + 1] = lines[idx + 1], lines[idx]
                    parsed[idx], parsed[idx + 1] = parsed[idx + 1], parsed[idx]
                    swapped += 1
                    changed = True
                    idx += 2
                    continue
                idx += 1
            if not changed:
                break
        return {"dialogue_lines": lines, "swapped": swapped}

    def _enforce_monotonic_dialogue_lines(self, dialogue_lines: list[str]) -> Dict[str, Any]:
        if not dialogue_lines:
            return {"dialogue_lines": dialogue_lines, "adjusted": 0}

        adjusted = 0
        prev_start = 0.0
        out_lines: list[str] = []
        for line in dialogue_lines:
            parsed = self._parse_dialogue_line(line)
            if parsed is None:
                out_lines.append(line)
                continue
            start = float(parsed.get("start", 0.0))
            safe_start = max(prev_start, start)
            if safe_start > start + 1e-6:
                adjusted += 1
            prev_start = safe_start
            approximate = str(parsed.get("timecode", "")).startswith("~")
            speaker = str(parsed.get("speaker", "")).strip()
            text = str(parsed.get("text", "")).strip()
            out_lines.append(
                f"{self._format_timecode(safe_start, approximate=approximate)} {speaker}: {text}"
            )
        return {"dialogue_lines": out_lines, "adjusted": adjusted}

    def _build_mono_turns(
        self,
        full_segments: Any,
        full_fallback_text: str = "",
        call_duration_sec: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        timeline = self._segments_to_timeline(full_segments, "Спикер (не определен)")
        if timeline:
            timeline.sort(key=lambda x: (x[0], x[1]))
            return [
                {"start": start, "approximate": False, "text": text}
                for start, _, _, text in timeline
            ]

        sentences = self._split_sentences(full_fallback_text)
        if not sentences:
            return []
        starts = self._estimate_turn_starts(len(sentences), call_duration_sec)
        return [
            {"start": starts[idx], "approximate": True, "text": sentence}
            for idx, sentence in enumerate(sentences)
        ]

    def _build_mono_dialogue_lines_from_turns(
        self,
        turns: list[dict[str, Any]],
        speaker_label: str,
    ) -> list[str]:
        lines: list[str] = []
        for turn in turns:
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            start = float(turn.get("start", 0.0))
            approximate = bool(turn.get("approximate", False))
            lines.append(f"{self._format_timecode(start, approximate=approximate)} {speaker_label}: {text}")
        return lines

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        compact = " ".join((text or "").split())
        if not compact:
            return []
        parts = re.split(r"(?<=[.!?])\s+", compact)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _normalize_token(token: str) -> str:
        return token.lower() if WORD_RE.fullmatch(token) else token

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_RE.findall(text or "")

    @staticmethod
    def _detokenize(tokens: list[str]) -> str:
        text = " ".join(tokens).strip()
        text = re.sub(r"\s+([.,!?;:%)\]}»])", r"\1", text)
        text = re.sub(r"([(\[{«])\s+", r"\1", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def _chunk_quality(self, tokens: list[str]) -> tuple[float, int, float]:
        words = [t.lower() for t in tokens if WORD_RE.fullmatch(t)]
        if not words:
            return 10.0, 0, 1.0
        max_run = 1
        cur = 1
        prev = words[0]
        for word in words[1:]:
            if word == prev:
                cur += 1
            else:
                cur = 1
                prev = word
            if cur > max_run:
                max_run = cur
        max_freq = max(Counter(words).values())
        repetition_ratio = max_freq / len(words)
        score = 0.0
        if max_run > 4:
            score += (max_run - 4) * 1.7
        if repetition_ratio > 0.18:
            score += (repetition_ratio - 0.18) * 35.0
        if len(words) < 3:
            score += 0.5
        return score, max_run, repetition_ratio

    def _is_suspicious_chunk(self, tokens: list[str]) -> bool:
        score, max_run, repetition_ratio = self._chunk_quality(tokens)
        return score >= 4.0 or max_run >= 8 or repetition_ratio >= 0.45

    def _pick_better_chunk(self, primary_chunk: list[str], secondary_chunk: list[str]) -> list[str]:
        p_score, _, _ = self._chunk_quality(primary_chunk)
        s_score, _, _ = self._chunk_quality(secondary_chunk)
        if abs(p_score - s_score) < 0.8:
            return primary_chunk
        return primary_chunk if p_score < s_score else secondary_chunk

    def _merge_texts(self, primary_text: str, secondary_text: str) -> str:
        primary_text = (primary_text or "").strip()
        secondary_text = (secondary_text or "").strip()
        if not primary_text:
            return secondary_text
        if not secondary_text:
            return primary_text

        primary_tokens = self._tokenize(primary_text)
        secondary_tokens = self._tokenize(secondary_text)
        if not primary_tokens:
            return secondary_text
        if not secondary_tokens:
            return primary_text

        primary_norm = [self._normalize_token(t) for t in primary_tokens]
        secondary_norm = [self._normalize_token(t) for t in secondary_tokens]
        matcher = difflib.SequenceMatcher(a=primary_norm, b=secondary_norm, autojunk=False)
        merged: list[str] = []
        for tag, a0, a1, b0, b1 in matcher.get_opcodes():
            if tag == "equal":
                merged.extend(primary_tokens[a0:a1])
                continue
            if tag == "replace":
                merged.extend(
                    self._pick_better_chunk(primary_tokens[a0:a1], secondary_tokens[b0:b1])
                )
                continue
            if tag == "delete":
                chunk = primary_tokens[a0:a1]
                if not self._is_suspicious_chunk(chunk):
                    merged.extend(chunk)
                continue
            if tag == "insert":
                chunk = secondary_tokens[b0:b1]
                if not self._is_suspicious_chunk(chunk):
                    merged.extend(chunk)

        merged_text = self._detokenize(merged)
        if not merged_text:
            p_score, _, _ = self._chunk_quality(primary_tokens)
            s_score, _, _ = self._chunk_quality(secondary_tokens)
            return primary_text if p_score <= s_score else secondary_text

        m_score, _, _ = self._chunk_quality(self._tokenize(merged_text))
        p_score, _, _ = self._chunk_quality(primary_tokens)
        s_score, _, _ = self._chunk_quality(secondary_tokens)
        best_original = primary_text if p_score <= s_score else secondary_text
        if m_score > min(p_score, s_score) + 1.2:
            return best_original
        return merged_text

    @staticmethod
    def _similarity_ratio(a: str, b: str) -> float:
        a_norm = " ".join((a or "").lower().split())
        b_norm = " ".join((b or "").lower().split())
        if not a_norm and not b_norm:
            return 1.0
        if not a_norm or not b_norm:
            return 0.0
        return difflib.SequenceMatcher(a=a_norm, b=b_norm, autojunk=False).ratio()

    def _should_fallback_to_mono_from_stereo(
        self,
        manager_text: str,
        client_text: str,
    ) -> tuple[bool, float]:
        manager_compact = " ".join((manager_text or "").split())
        client_compact = " ".join((client_text or "").split())
        if not manager_compact or not client_compact:
            return False, 0.0
        similarity = self._similarity_ratio(manager_compact, client_compact)
        if manager_compact == client_compact and len(manager_compact) >= 20:
            return True, similarity
        min_chars = max(1, self._settings.stereo_overlap_min_chars)
        if min(len(manager_compact), len(client_compact)) < min_chars:
            return False, similarity
        threshold = min(1.0, max(0.0, self._settings.stereo_overlap_similarity_threshold))
        return similarity >= threshold, similarity

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _score_role_cues(self, text: str) -> tuple[float, float]:
        lowered = (text or "").lower()
        manager_score = 0.0
        client_score = 0.0
        for cue, weight in MANAGER_CUES.items():
            if cue in lowered:
                manager_score += weight
        for cue, weight in CLIENT_CUES.items():
            if cue in lowered:
                client_score += weight
        if "?" in text:
            client_score += 0.2
        if "подскажите" in lowered:
            manager_score += 0.4
        if lowered.startswith("алло"):
            client_score += 0.2
        return manager_score, client_score

    def _build_role_texts_and_lines(
        self,
        turns: list[dict[str, Any]],
        roles: list[str],
        manager_name: str,
    ) -> tuple[str, str, list[str]]:
        manager_label = f"Менеджер ({manager_name})"
        manager_parts: list[str] = []
        client_parts: list[str] = []
        lines: list[str] = []
        for turn, role in zip(turns, roles):
            text = str(turn.get("text", "")).strip()
            if not text:
                continue
            start = float(turn.get("start", 0.0))
            approximate = bool(turn.get("approximate", False))
            if role == "manager":
                label = manager_label
                manager_parts.append(text)
            else:
                label = "Клиент"
                client_parts.append(text)
            lines.append(f"{self._format_timecode(start, approximate=approximate)} {label}: {text}")
        manager_text = " ".join(manager_parts).strip()
        client_text = " ".join(client_parts).strip()
        return manager_text, client_text, lines

    def _assign_roles_rule_based(
        self,
        turns: list[dict[str, Any]],
        manager_name: str,
    ) -> Optional[Dict[str, Any]]:
        if not turns:
            return None
        raw_roles: list[str] = []
        cue_margins: list[float] = []
        covered = 0
        manager_total = 0.0
        client_total = 0.0
        for turn in turns:
            text = str(turn.get("text", "")).strip()
            m_score, c_score = self._score_role_cues(text)
            manager_total += m_score
            client_total += c_score
            if m_score == 0.0 and c_score == 0.0:
                raw_roles.append("unknown")
                continue
            covered += 1
            margin = abs(m_score - c_score) / (m_score + c_score + 1e-6)
            cue_margins.append(margin)
            raw_roles.append("manager" if m_score >= c_score else "client")

        global_bias = "manager" if manager_total >= client_total else "client"
        roles = raw_roles[:]
        for idx, role in enumerate(roles):
            if role != "unknown":
                continue
            prev_role = next((roles[j] for j in range(idx - 1, -1, -1) if roles[j] != "unknown"), None)
            next_role = next(
                (roles[j] for j in range(idx + 1, len(roles)) if roles[j] != "unknown"),
                None,
            )
            if prev_role and next_role and prev_role == next_role:
                roles[idx] = "client" if prev_role == "manager" else "manager"
            elif prev_role:
                roles[idx] = "client" if prev_role == "manager" else "manager"
            elif next_role:
                roles[idx] = "client" if next_role == "manager" else "manager"
            else:
                roles[idx] = global_bias

        manager_text, client_text, dialogue_lines = self._build_role_texts_and_lines(
            turns, roles, manager_name
        )
        manager_turns = sum(1 for role in roles if role == "manager")
        client_turns = sum(1 for role in roles if role == "client")
        has_both_roles = manager_turns > 0 and client_turns > 0
        coverage_ratio = covered / max(1, len(turns))
        mean_margin = (sum(cue_margins) / len(cue_margins)) if cue_margins else 0.0
        confidence = self._clamp_01(0.20 + 0.45 * coverage_ratio + 0.35 * mean_margin)
        if not has_both_roles:
            confidence = min(confidence, 0.45)
        return {
            "manager_text": manager_text,
            "client_text": client_text,
            "dialogue_lines": dialogue_lines,
            "meta": {
                "provider": "rule",
                "confidence": confidence,
                "notes": f"coverage={coverage_ratio:.2f}; margin={mean_margin:.2f}",
                "has_both_roles": has_both_roles,
                "roles": roles,
            },
        }

    def _assign_roles_with_openai(
        self,
        turns: list[dict[str, Any]],
        manager_name: str,
    ) -> Dict[str, Any]:
        if not turns:
            raise RuntimeError("mono role assignment has no turns")
        numbered_turns: list[str] = []
        for idx, turn in enumerate(turns, start=1):
            text = str(turn.get("text", "")).strip()
            start = float(turn.get("start", 0.0))
            approximate = bool(turn.get("approximate", False))
            timecode = self._format_timecode(start, approximate=approximate)
            numbered_turns.append(f"{idx}. {timecode} {text}")

        client = self._openai_client()
        response = client.chat.completions.create(
            model=self._settings.openai_role_assign_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ROLE_ASSIGN_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Manager name from call metadata: {manager_name}\n"
                        "Turns:\n"
                        + "\n".join(numbered_turns)
                    ),
                },
            ],
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise RuntimeError("OpenAI role assignment returned empty content")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI role assignment payload is not object")
        roles_raw = payload.get("roles")
        if not isinstance(roles_raw, list):
            raise RuntimeError("OpenAI role assignment must return roles array")
        if len(roles_raw) != len(turns):
            raise RuntimeError(
                f"OpenAI role assignment roles length mismatch: {len(roles_raw)} != {len(turns)}"
            )
        roles: list[str] = []
        for item in roles_raw:
            value = str(item).strip().lower()
            if value not in {"manager", "client"}:
                raise RuntimeError(f"OpenAI role assignment invalid role: {value}")
            roles.append(value)

        manager_text, client_text, dialogue_lines = self._build_role_texts_and_lines(
            turns, roles, manager_name
        )
        has_both_roles = bool(manager_text and client_text)
        confidence_raw = payload.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = self._clamp_01(confidence)
        if not has_both_roles:
            confidence = min(confidence, 0.45)
        return {
            "manager_text": manager_text,
            "client_text": client_text,
            "dialogue_lines": dialogue_lines,
            "meta": {
                "provider": "openai",
                "confidence": confidence,
                "notes": str(payload.get("notes", "")).strip(),
                "has_both_roles": has_both_roles,
                "roles": roles,
            },
        }

    def _assign_roles_with_ollama(
        self,
        turns: list[dict[str, Any]],
        manager_name: str,
    ) -> Dict[str, Any]:
        if not turns:
            raise RuntimeError("mono role assignment has no turns")
        numbered_turns: list[str] = []
        for idx, turn in enumerate(turns, start=1):
            text = str(turn.get("text", "")).strip()
            start = float(turn.get("start", 0.0))
            approximate = bool(turn.get("approximate", False))
            timecode = self._format_timecode(start, approximate=approximate)
            numbered_turns.append(f"{idx}. {timecode} {text}")

        client = self._ollama_client()
        payload = client.generate_json(
            model=self._settings.ollama_model,
            think=self._settings.ollama_think,
            temperature=self._settings.ollama_temperature,
            system_prompt=ROLE_ASSIGN_SYSTEM_PROMPT,
            user_prompt=(
                f"Manager name from call metadata: {manager_name}\n"
                "Turns:\n"
                + "\n".join(numbered_turns)
            ),
            num_predict=max(320, len(turns) * 24),
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Ollama role assignment payload is not object")
        roles_raw = payload.get("roles")
        if not isinstance(roles_raw, list):
            raise RuntimeError("Ollama role assignment must return roles array")
        if len(roles_raw) != len(turns):
            raise RuntimeError(
                f"Ollama role assignment roles length mismatch: {len(roles_raw)} != {len(turns)}"
            )
        roles: list[str] = []
        for item in roles_raw:
            value = str(item).strip().lower()
            if value not in {"manager", "client"}:
                raise RuntimeError(f"Ollama role assignment invalid role: {value}")
            roles.append(value)

        manager_text, client_text, dialogue_lines = self._build_role_texts_and_lines(
            turns, roles, manager_name
        )
        has_both_roles = bool(manager_text and client_text)
        confidence_raw = payload.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = self._clamp_01(confidence)
        if not has_both_roles:
            confidence = min(confidence, 0.45)
        return {
            "manager_text": manager_text,
            "client_text": client_text,
            "dialogue_lines": dialogue_lines,
            "meta": {
                "provider": "ollama",
                "confidence": confidence,
                "notes": str(payload.get("notes", "")).strip(),
                "has_both_roles": has_both_roles,
                "roles": roles,
            },
        }

    def _assign_roles_for_mono(
        self,
        turns: list[dict[str, Any]],
        manager_name: str,
        warnings: list[str],
    ) -> Optional[Dict[str, Any]]:
        mode = (self._settings.mono_role_assignment_mode or "off").strip().lower()
        if mode not in ROLE_ASSIGN_ALLOWED_MODES:
            warnings.append(
                f"mono_role_assign: unsupported mode={mode}; fallback=off"
            )
            mode = "off"
        if mode == "off":
            return None

        min_conf = self._clamp_01(self._settings.mono_role_assignment_min_confidence)
        llm_threshold = self._clamp_01(self._settings.mono_role_assignment_llm_threshold)
        rule_result = self._assign_roles_rule_based(turns, manager_name)
        if mode == "rule":
            if (
                rule_result
                and rule_result["meta"].get("has_both_roles")
                and float(rule_result["meta"].get("confidence", 0.0)) >= min_conf
            ):
                return rule_result
            warnings.append("mono_role_assign: rule confidence too low; keep mono")
            return None

        # openai_selective/ollama_selective: use rule result directly if it is already reliable.
        if (
            rule_result
            and rule_result["meta"].get("has_both_roles")
            and float(rule_result["meta"].get("confidence", 0.0)) >= llm_threshold
        ):
            rule_result["meta"]["provider"] = "rule_high_conf"
            return rule_result

        llm_result: Optional[Dict[str, Any]] = None
        if mode == "openai_selective":
            if self._settings.openai_api_key:
                try:
                    llm_result = self._assign_roles_with_openai(turns, manager_name)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"mono_role_assign: openai_failed: {exc}")
            else:
                warnings.append(
                    "mono_role_assign: OPENAI_API_KEY missing for openai_selective"
                )
        elif mode == "ollama_selective":
            try:
                llm_result = self._assign_roles_with_ollama(turns, manager_name)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"mono_role_assign: ollama_failed: {exc}")

        if (
            llm_result
            and llm_result["meta"].get("has_both_roles")
            and float(llm_result["meta"].get("confidence", 0.0)) >= min_conf
        ):
            return llm_result

        if (
            rule_result
            and rule_result["meta"].get("has_both_roles")
            and float(rule_result["meta"].get("confidence", 0.0)) >= min_conf
        ):
            rule_result["meta"]["provider"] = "rule_fallback"
            return rule_result

        warnings.append("mono_role_assign: kept mono due to low confidence")
        return None

    def _merge_with_openai(
        self,
        primary_text: str,
        secondary_text: str,
        *,
        speaker_label: str,
    ) -> Dict[str, Any]:
        client = self._openai_client()
        response = client.chat.completions.create(
            model=self._settings.openai_merge_model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": MERGE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Speaker: {speaker_label}\n\n"
                        "Variant A:\n"
                        f"{primary_text}\n\n"
                        "Variant B:\n"
                        f"{secondary_text}"
                    ),
                },
            ],
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise RuntimeError("OpenAI merge returned empty content")
        payload = json.loads(content)
        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI merge response is not JSON object")
        merged_text = str(payload.get("merged_text", "")).strip()
        if not merged_text:
            raise RuntimeError("OpenAI merge returned empty merged_text")
        selection = str(payload.get("selection", "MIX")).strip().upper()
        if selection not in {"A", "B", "MIX"}:
            selection = "MIX"
        confidence_raw = payload.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return {
            "text": merged_text,
            "selection": selection,
            "confidence": confidence,
            "notes": str(payload.get("notes", "")).strip(),
            "provider": "openai",
        }

    def _merge_with_ollama(
        self,
        primary_text: str,
        secondary_text: str,
        *,
        speaker_label: str,
    ) -> Dict[str, Any]:
        client = self._ollama_client()
        payload = client.generate_json(
            model=self._settings.ollama_model,
            think=self._settings.ollama_think,
            temperature=self._settings.ollama_temperature,
            system_prompt=MERGE_SYSTEM_PROMPT,
            user_prompt=(
                f"Speaker: {speaker_label}\n\n"
                "Variant A:\n"
                f"{primary_text}\n\n"
                "Variant B:\n"
                f"{secondary_text}"
            ),
            num_predict=900,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Ollama merge response is not JSON object")
        merged_text = str(payload.get("merged_text", "")).strip()
        if not merged_text:
            raise RuntimeError("Ollama merge returned empty merged_text")
        selection = str(payload.get("selection", "MIX")).strip().upper()
        if selection not in {"A", "B", "MIX"}:
            selection = "MIX"
        confidence_raw = payload.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return {
            "text": merged_text,
            "selection": selection,
            "confidence": confidence,
            "notes": str(payload.get("notes", "")).strip(),
            "provider": "ollama",
        }

    @staticmethod
    def _extract_json_payload(text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            raise RuntimeError("empty response")
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
        if fence:
            payload = json.loads(fence.group(1))
            if isinstance(payload, dict):
                return payload

        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(raw[start : end + 1])
            if isinstance(payload, dict):
                return payload
        raise RuntimeError("response does not contain JSON object")

    def _merge_with_codex_cli(
        self,
        primary_text: str,
        secondary_text: str,
        *,
        speaker_label: str,
    ) -> Dict[str, Any]:
        codex_bin = (self._settings.codex_cli_command or "codex").strip() or "codex"
        if shutil.which(codex_bin) is None:
            raise RuntimeError(f"codex binary is not available: {codex_bin}")

        prompt = CODEX_MERGE_PROMPT_TEMPLATE.format(
            speaker_label=speaker_label,
            variant_a=primary_text,
            variant_b=secondary_text,
        )
        timeout_sec = max(15, int(self._settings.codex_cli_timeout_sec))
        with tempfile.NamedTemporaryFile(prefix="mango_codex_merge_", suffix=".txt") as out_file:
            cmd = [
                codex_bin,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--model",
                self._settings.codex_merge_model,
                "--output-last-message",
                out_file.name,
            ]
            reasoning_effort = (self._settings.codex_reasoning_effort or "").strip().lower()
            if reasoning_effort in {"low", "medium", "high"}:
                cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
            cmd.append(prompt)
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
            if proc.returncode != 0:
                stderr_tail = (proc.stderr or "").strip().splitlines()[-1:] or [""]
                raise RuntimeError(
                    f"codex exec failed rc={proc.returncode}: {stderr_tail[0].strip()}"
                )
            raw = Path(out_file.name).read_text(encoding="utf-8", errors="ignore")

        payload = self._extract_json_payload(raw)
        merged_text = str(payload.get("merged_text", "")).strip()
        if not merged_text:
            raise RuntimeError("Codex merge returned empty merged_text")
        selection = str(payload.get("selection", "MIX")).strip().upper()
        if selection not in {"A", "B", "MIX"}:
            selection = "MIX"
        confidence_raw = payload.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        return {
            "text": merged_text,
            "selection": selection,
            "confidence": confidence,
            "notes": str(payload.get("notes", "")).strip(),
            "provider": "codex_cli",
        }

    def _merge_variant_pair(
        self,
        primary_text: str,
        secondary_text: str,
        *,
        speaker_label: str,
    ) -> Dict[str, Any]:
        a = (primary_text or "").strip()
        b = (secondary_text or "").strip()
        if not b:
            return {
                "text": a,
                "selection": "A",
                "confidence": 1.0 if a else 0.0,
                "provider": "primary",
                "notes": "secondary_empty",
                "similarity": 0.0,
            }
        if not a:
            return {
                "text": b,
                "selection": "B",
                "confidence": 0.6,
                "provider": "secondary_only",
                "notes": "primary_empty",
                "similarity": 0.0,
            }

        similarity = self._similarity_ratio(a, b)
        threshold = self._settings.dual_merge_similarity_threshold
        if similarity >= threshold:
            return {
                "text": a,
                "selection": "A",
                "confidence": 0.95,
                "provider": "skip_high_similarity",
                "notes": f"similarity={similarity:.4f}",
                "similarity": similarity,
            }

        merge_provider = self._settings.dual_merge_provider
        if merge_provider not in MERGE_ALLOWED_PROVIDERS:
            merge_provider = "rule"
        if merge_provider == "primary":
            return {
                "text": a,
                "selection": "A",
                "confidence": 0.7,
                "provider": "primary",
                "notes": "dual_merge_provider=primary",
                "similarity": similarity,
            }
        if merge_provider == "rule":
            merged = self._merge_texts(a, b)
            choice = "MIX"
            if merged == a:
                choice = "A"
            elif merged == b:
                choice = "B"
            return {
                "text": merged,
                "selection": choice,
                "confidence": 0.75,
                "provider": "rule",
                "notes": "",
                "similarity": similarity,
            }
        if merge_provider == "ollama":
            try:
                merged = self._merge_with_ollama(
                    primary_text=a, secondary_text=b, speaker_label=speaker_label
                )
                merged["similarity"] = similarity
                return merged
            except Exception as exc:  # noqa: BLE001
                merged = self._merge_texts(a, b)
                choice = "MIX"
                if merged == a:
                    choice = "A"
                elif merged == b:
                    choice = "B"
                return {
                    "text": merged,
                    "selection": choice,
                    "confidence": 0.6,
                    "provider": "rule_fallback",
                    "notes": f"ollama_merge_failed: {exc}",
                    "similarity": similarity,
                }
        if merge_provider == "codex_cli":
            try:
                merged = self._merge_with_codex_cli(
                    primary_text=a, secondary_text=b, speaker_label=speaker_label
                )
                merged["similarity"] = similarity
                return merged
            except Exception as exc:  # noqa: BLE001
                merged = self._merge_texts(a, b)
                choice = "MIX"
                if merged == a:
                    choice = "A"
                elif merged == b:
                    choice = "B"
                return {
                    "text": merged,
                    "selection": choice,
                    "confidence": 0.6,
                    "provider": "rule_fallback",
                    "notes": f"codex_cli_merge_failed: {exc}",
                    "similarity": similarity,
                }

        try:
            merged = self._merge_with_openai(
                primary_text=a, secondary_text=b, speaker_label=speaker_label
            )
            merged["similarity"] = similarity
            return merged
        except Exception as exc:  # noqa: BLE001
            # In batch mode we should not fail a call if merge LLM is transiently unavailable.
            merged = self._merge_texts(a, b)
            choice = "MIX"
            if merged == a:
                choice = "A"
            elif merged == b:
                choice = "B"
            return {
                "text": merged,
                "selection": choice,
                "confidence": 0.6,
                "provider": "rule_fallback",
                "notes": f"openai_merge_failed: {exc}",
                "similarity": similarity,
            }

    def _try_transcribe_file_with_meta(self, path: Path, provider: str) -> Dict[str, Any]:
        try:
            result = self._transcribe_file_with_meta(path, provider=provider)
        except Exception as exc:  # noqa: BLE001
            return {"text": "", "segments": None, "error": str(exc)}
        text = str(result.get("text", "")).strip()
        segments = result.get("segments")
        return {
            "text": text,
            "segments": segments if isinstance(segments, list) else None,
            "error": None,
        }

    def _transcribe_file_gigaam(self, path: Path) -> Dict[str, Any]:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            raise RuntimeError("ffmpeg is required for gigaam transcribe provider")

        model = self._get_gigaam_model()
        segment_sec = max(5, self._settings.gigaam_segment_sec)
        temp_dir = Path(tempfile.mkdtemp(prefix="mango_mvp_gigaam_"))
        pattern = temp_dir / "chunk_%03d.wav"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(path),
            "-f",
            "segment",
            "-segment_time",
            str(segment_sec),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(pattern),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"gigaam chunking failed: {result.stderr.strip()}")

            chunks = sorted(temp_dir.glob("chunk_*.wav"))
            if not chunks:
                raise RuntimeError("gigaam chunking produced no chunks")

            parts: list[str] = []
            segments: list[dict[str, Any]] = []
            for idx, chunk in enumerate(chunks):
                chunk_text = str(model.transcribe(str(chunk))).strip()
                if not chunk_text:
                    continue
                normalized_text = " ".join(chunk_text.split())
                parts.append(normalized_text)
                segments.append(
                    {
                        "start": float(idx * segment_sec),
                        "end": float((idx + 1) * segment_sec),
                        "text": normalized_text,
                    }
                )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        final_text = " ".join(parts).strip()
        if not final_text:
            raise RuntimeError("gigaam returned empty text")
        return {"text": final_text, "segments": segments}

    def _export_transcript_file(self, call: CallRecord, result: Dict[str, Any]) -> None:
        export_dir = self._settings.transcript_export_dir
        if not export_dir:
            return

        source_path = Path(call.source_file)
        manager_name = self._extract_manager_name_from_filename(call.source_filename)
        manager_text = (result.get("transcript_manager") or "").strip()
        client_text = (result.get("transcript_client") or "").strip()
        full_text = (result.get("transcript_text") or "").strip()
        dialogue_lines = result.get("dialogue_lines")

        if isinstance(dialogue_lines, list) and dialogue_lines:
            normalized_lines = [str(line).strip() for line in dialogue_lines if str(line).strip()]
            if manager_text or client_text:
                body = "\n".join(normalized_lines) + "\n"
            else:
                body = (
                    "Примечание: каналы не разделены, поэтому спикер отмечен как единый поток.\n"
                    + "\n".join(normalized_lines)
                    + "\n"
                )
        elif manager_text or client_text:
            body = (
                f"Менеджер ({manager_name}):\n{manager_text or '[нет распознанной речи]'}\n\n"
                f"Клиент:\n{client_text or '[нет распознанной речи]'}\n"
            )
        else:
            body = (
                f"Менеджер ({manager_name}):\n[каналы не разделены]\n\n"
                "Клиент:\n[каналы не разделены]\n\n"
                "Транскрипт (без разделения каналов):\n"
                f"{full_text}\n"
            )

        target_dir = Path(export_dir) / source_path.parent.name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{source_path.stem}_text.txt"
        target_path.write_text(body, encoding="utf-8")

        variants_json = result.get("transcript_variants_json")
        if isinstance(variants_json, str) and variants_json.strip():
            variants_path = target_dir / f"{source_path.stem}_variants.json"
            variants_path.write_text(variants_json, encoding="utf-8")

    def _transcribe_file_with_meta(self, path: Path, provider: str) -> Dict[str, Any]:
        if provider == "mock":
            return {"text": f"[mock transcript for {path.name}]", "segments": None}
        if provider == "gigaam":
            return self._transcribe_file_gigaam(path)
        if provider == "mlx":
            try:
                import mlx_whisper
            except ImportError as exc:
                raise RuntimeError(
                    "mlx-whisper is not installed. Run: python3 -m pip install mlx-whisper"
                ) from exc

            kwargs = {"path_or_hf_repo": self._settings.mlx_whisper_model}
            kwargs["condition_on_previous_text"] = (
                self._settings.mlx_condition_on_previous_text
            )
            kwargs["word_timestamps"] = self._settings.mlx_word_timestamps
            if self._settings.transcribe_language:
                kwargs["language"] = self._settings.transcribe_language
            try:
                result = mlx_whisper.transcribe(str(path), **kwargs)
            except TypeError:
                # Keep compatibility with older mlx-whisper argument signatures.
                kwargs.pop("language", None)
                kwargs.pop("word_timestamps", None)
                result = mlx_whisper.transcribe(str(path), **kwargs)
            text = result.get("text") if isinstance(result, dict) else None
            if not text:
                raise RuntimeError("mlx-whisper returned empty text")
            segments = result.get("segments") if isinstance(result, dict) else None
            return {
                "text": text.strip(),
                "segments": segments if isinstance(segments, list) else None,
            }
        if provider != "openai":
            raise RuntimeError(f"Unsupported TRANSCRIBE_PROVIDER={provider}")

        client = self._openai_client()
        kwargs = {"model": self._settings.openai_transcribe_model}
        if self._settings.transcribe_language:
            kwargs["language"] = self._settings.transcribe_language
        with path.open("rb") as f:
            result = client.audio.transcriptions.create(file=f, **kwargs)  # type: ignore[arg-type]
        text = getattr(result, "text", None)
        if not text:
            raise RuntimeError("OpenAI transcription returned empty text")
        return {"text": text, "segments": None}

    def _transcribe_call(self, call: CallRecord) -> Dict[str, Any]:
        path = Path(call.source_file)
        primary_provider = self._settings.transcribe_provider
        secondary_provider = self._settings.secondary_transcribe_provider
        warnings: list[str] = []
        dual_enabled = (
            self._settings.dual_transcribe_enabled
            and bool(secondary_provider)
            and secondary_provider != primary_provider
        )
        if (
            self._settings.split_stereo_channels
            and (call.channels == 2 or call.channels is None)
            and path.suffix.lower() in {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
        ):
            split = split_stereo_to_mono(path)
            if split:
                left, right, temp_dir = split
                try:
                    manager_primary = self._cached_variant_candidate(
                        call,
                        slot="manager",
                        provider=primary_provider,
                        primary_provider=primary_provider,
                    ) or self._try_transcribe_file_with_meta(
                        left, provider=primary_provider
                    )
                    client_primary = self._cached_variant_candidate(
                        call,
                        slot="client",
                        provider=primary_provider,
                        primary_provider=primary_provider,
                    ) or self._try_transcribe_file_with_meta(
                        right, provider=primary_provider
                    )
                    manager_secondary: Optional[Dict[str, Any]] = None
                    client_secondary: Optional[Dict[str, Any]] = None
                    if dual_enabled and secondary_provider:
                        manager_secondary = self._cached_variant_candidate(
                            call,
                            slot="manager",
                            provider=secondary_provider,
                            primary_provider=primary_provider,
                        ) or self._try_transcribe_file_with_meta(
                            left, provider=secondary_provider
                        )
                        client_secondary = self._cached_variant_candidate(
                            call,
                            slot="client",
                            provider=secondary_provider,
                            primary_provider=primary_provider,
                        ) or self._try_transcribe_file_with_meta(
                            right, provider=secondary_provider
                        )
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

                manager_primary_text = str(manager_primary["text"]).strip()
                client_primary_text = str(client_primary["text"]).strip()
                for label, payload in (
                    ("manager_primary", manager_primary),
                    ("client_primary", client_primary),
                    ("manager_secondary", manager_secondary or {}),
                    ("client_secondary", client_secondary or {}),
                ):
                    if bool((payload or {}).get("cached")):
                        warnings.append(f"{label}: reused_cached_variant")
                for label, payload in (
                    ("manager_primary", manager_primary),
                    ("client_primary", client_primary),
                    ("manager_secondary", manager_secondary or {}),
                    ("client_secondary", client_secondary or {}),
                ):
                    error = str(payload.get("error") or "").strip()
                    if error:
                        warnings.append(f"{label}: {error}")
                if not manager_primary_text and not client_primary_text:
                    # Channel split can occasionally produce silent/failed mono branches;
                    # in that case fallback to full-file transcription below.
                    if not warnings:
                        warnings.append("stereo_split: both channels returned empty text")
                    pass
                else:
                    should_fallback_to_mono, stereo_similarity = (
                        self._should_fallback_to_mono_from_stereo(
                            manager_primary_text, client_primary_text
                        )
                    )
                    if should_fallback_to_mono:
                        warnings.append(
                            "stereo_split: channels_too_similar "
                            f"(similarity={stereo_similarity:.4f}); fallback_to_full"
                        )
                        pass
                    else:
                        manager_secondary_text = (
                            str((manager_secondary or {}).get("text", "")).strip()
                            if dual_enabled
                            else ""
                        )
                        client_secondary_text = (
                            str((client_secondary or {}).get("text", "")).strip()
                            if dual_enabled
                            else ""
                        )

                        manager_merge = self._merge_variant_pair(
                            manager_primary_text,
                            manager_secondary_text,
                            speaker_label="Менеджер",
                        )
                        client_merge = self._merge_variant_pair(
                            client_primary_text,
                            client_secondary_text,
                            speaker_label="Клиент",
                        )
                        manager_text = str(manager_merge["text"]).strip()
                        client_text = str(client_merge["text"]).strip()
                        if not manager_text and not client_text:
                            raise RuntimeError("Both stereo channels produced empty transcript text")

                        manager_name = self._extract_manager_name_from_filename(
                            call.source_filename
                        )
                        dialogue_lines = self._build_dialogue_lines(
                            manager_name,
                            manager_primary.get("segments"),
                            client_primary.get("segments"),
                            manager_fallback_text=manager_text,
                            client_fallback_text=client_text,
                            call_duration_sec=call.duration_sec,
                        )
                        crosstalk_dedupe = self._dedupe_stereo_cross_talk(dialogue_lines)
                        dropped_cross_talk = int(crosstalk_dedupe.get("dropped", 0) or 0)
                        if dropped_cross_talk > 0:
                            dialogue_lines = crosstalk_dedupe["dialogue_lines"]
                            clean_manager_text = str(
                                crosstalk_dedupe.get("manager_text", "")
                            ).strip()
                            clean_client_text = str(
                                crosstalk_dedupe.get("client_text", "")
                            ).strip()
                            if clean_manager_text and clean_client_text:
                                manager_text = clean_manager_text
                                client_text = clean_client_text
                            warnings.append(
                                "stereo_dedupe: dropped_cross_talk_lines="
                                f"{dropped_cross_talk}"
                            )
                        echo_dedupe = self._dedupe_adjacent_cross_speaker_echo(dialogue_lines)
                        dropped_echo = int(echo_dedupe.get("dropped", 0) or 0)
                        if dropped_echo > 0:
                            dialogue_lines = echo_dedupe["dialogue_lines"]
                            warnings.append(
                                "stereo_echo_dedupe: dropped_adjacent_echo_lines="
                                f"{dropped_echo}"
                            )
                        artifact_filter = self._drop_artifact_only_lines(dialogue_lines)
                        dropped_artifacts = int(artifact_filter.get("dropped", 0) or 0)
                        if dropped_artifacts > 0:
                            dialogue_lines = artifact_filter["dialogue_lines"]
                            warnings.append(
                                "dialogue_artifact_filter: dropped_lines="
                                f"{dropped_artifacts}"
                            )
                        sequence_reorder = self._resequence_dialogue_lines(dialogue_lines)
                        swapped_lines = int(sequence_reorder.get("swapped", 0) or 0)
                        if swapped_lines > 0:
                            dialogue_lines = sequence_reorder["dialogue_lines"]
                            warnings.append(
                                "stereo_sequence_fix: swapped_adjacent_pairs="
                                f"{swapped_lines}"
                            )
                        monotonic_fix = self._enforce_monotonic_dialogue_lines(dialogue_lines)
                        monotonic_adjusted = int(monotonic_fix.get("adjusted", 0) or 0)
                        if monotonic_adjusted > 0:
                            dialogue_lines = monotonic_fix["dialogue_lines"]
                            warnings.append(
                                "stereo_time_fix: monotonic_adjusted_lines="
                                f"{monotonic_adjusted}"
                            )
                        rebuilt_manager, rebuilt_client = self._rebuild_role_texts_from_dialogue_lines(
                            dialogue_lines
                        )
                        if rebuilt_manager and rebuilt_client:
                            manager_text = rebuilt_manager
                            client_text = rebuilt_client
                        combined = f"MANAGER:\n{manager_text}\n\nCLIENT:\n{client_text}"
                        variants_payload = {
                            "mode": "stereo",
                            "primary_provider": primary_provider,
                            "secondary_provider": secondary_provider if dual_enabled else None,
                            "merge_provider": self._settings.dual_merge_provider if dual_enabled else "primary",
                            "manager": {
                                "variant_a": manager_primary_text,
                                "variant_b": manager_secondary_text if dual_enabled else None,
                                "final": manager_text,
                                "merge_meta": manager_merge,
                            },
                            "client": {
                                "variant_a": client_primary_text,
                                "variant_b": client_secondary_text if dual_enabled else None,
                                "final": client_text,
                                "merge_meta": client_merge,
                            },
                            "stereo_similarity": stereo_similarity,
                            "stereo_crosstalk_dedupe": {
                                "applied": dropped_cross_talk > 0,
                                "dropped_lines": dropped_cross_talk,
                                "pairs_checked": int(crosstalk_dedupe.get("pairs_checked", 0) or 0),
                            },
                            "stereo_echo_dedupe": {
                                "applied": dropped_echo > 0,
                                "dropped_lines": dropped_echo,
                                "pairs_checked": int(echo_dedupe.get("pairs_checked", 0) or 0),
                            },
                            "dialogue_artifact_filter": {
                                "applied": dropped_artifacts > 0,
                                "dropped_lines": dropped_artifacts,
                            },
                            "stereo_sequence_fix": {
                                "applied": swapped_lines > 0,
                                "swapped_adjacent_pairs": swapped_lines,
                            },
                            "stereo_time_fix": {
                                "applied": monotonic_adjusted > 0,
                                "monotonic_adjusted_lines": monotonic_adjusted,
                            },
                            "warnings": warnings,
                        }
                        return {
                            "transcript_manager": manager_text,
                            "transcript_client": client_text,
                            "transcript_text": combined,
                            "dialogue_lines": dialogue_lines,
                            "transcript_variants_json": json.dumps(
                                variants_payload, ensure_ascii=False
                            ),
                        }

        full_primary = self._cached_variant_candidate(
            call,
            slot="full",
            provider=primary_provider,
            primary_provider=primary_provider,
        ) or self._try_transcribe_file_with_meta(path, provider=primary_provider)
        full_primary_text = str(full_primary["text"]).strip()
        full_secondary_text = ""
        full_secondary: Optional[Dict[str, Any]] = None
        if dual_enabled and secondary_provider:
            full_secondary = self._cached_variant_candidate(
                call,
                slot="full",
                provider=secondary_provider,
                primary_provider=primary_provider,
            ) or self._try_transcribe_file_with_meta(path, provider=secondary_provider)
            full_secondary_text = str(full_secondary.get("text", "")).strip()
        for label, payload in (
            ("full_primary", full_primary),
            ("full_secondary", full_secondary or {}),
        ):
            if bool((payload or {}).get("cached")):
                warnings.append(f"{label}: reused_cached_variant")
        for label, payload in (
            ("full_primary", full_primary),
            ("full_secondary", full_secondary or {}),
        ):
            error = str(payload.get("error") or "").strip()
            if error:
                warnings.append(f"{label}: {error}")
        full_merge = self._merge_variant_pair(
            full_primary_text,
            full_secondary_text,
            speaker_label="Полный звонок",
        )
        full_text = str(full_merge["text"]).strip()
        if not full_text:
            details = "; ".join(warnings) if warnings else "no provider errors captured"
            raise RuntimeError(f"All transcript variants are empty for full-file mode ({details})")
        dialogue_source_segments = full_primary.get("segments")
        if not isinstance(dialogue_source_segments, list) or not dialogue_source_segments:
            dialogue_source_segments = (full_secondary or {}).get("segments")
        mono_turns = self._build_mono_turns(
            dialogue_source_segments,
            full_fallback_text=full_text,
            call_duration_sec=call.duration_sec,
        )
        dialogue_lines = self._build_mono_dialogue_lines_from_turns(
            mono_turns, "Спикер (не определен)"
        )
        manager_name = self._extract_manager_name_from_filename(call.source_filename)
        role_assignment = self._assign_roles_for_mono(
            mono_turns,
            manager_name=manager_name,
            warnings=warnings,
        )
        transcript_manager: Optional[str] = None
        transcript_client: Optional[str] = None
        transcript_text = full_text
        if role_assignment:
            transcript_manager = str(role_assignment.get("manager_text") or "").strip()
            transcript_client = str(role_assignment.get("client_text") or "").strip()
            if transcript_manager and transcript_client:
                dialogue_lines = role_assignment.get("dialogue_lines") or dialogue_lines
                transcript_text = (
                    f"MANAGER:\n{transcript_manager}\n\nCLIENT:\n{transcript_client}"
                )
        artifact_filter = self._drop_artifact_only_lines(dialogue_lines)
        dropped_artifacts = int(artifact_filter.get("dropped", 0) or 0)
        if dropped_artifacts > 0:
            dialogue_lines = artifact_filter["dialogue_lines"]
            warnings.append(
                "dialogue_artifact_filter: dropped_lines="
                f"{dropped_artifacts}"
            )
            rebuilt_manager, rebuilt_client = self._rebuild_role_texts_from_dialogue_lines(
                dialogue_lines
            )
            if rebuilt_manager and rebuilt_client:
                transcript_manager = rebuilt_manager
                transcript_client = rebuilt_client
                transcript_text = (
                    f"MANAGER:\n{transcript_manager}\n\nCLIENT:\n{transcript_client}"
                )
        variants_payload = {
            "mode": "mono_or_fallback",
            "primary_provider": primary_provider,
            "secondary_provider": secondary_provider if dual_enabled else None,
            "merge_provider": self._settings.dual_merge_provider if dual_enabled else "primary",
            "full": {
                "variant_a": full_primary_text,
                "variant_b": full_secondary_text if dual_enabled else None,
                "final": full_text,
                "merge_meta": full_merge,
            },
            "role_assignment": {
                "applied": bool(role_assignment and transcript_manager and transcript_client),
                "mode": self._settings.mono_role_assignment_mode,
                "meta": (role_assignment or {}).get("meta"),
            },
            "dialogue_artifact_filter": {
                "applied": dropped_artifacts > 0,
                "dropped_lines": dropped_artifacts,
            },
            "warnings": warnings,
        }
        return {
            "transcript_manager": transcript_manager,
            "transcript_client": transcript_client,
            "transcript_text": transcript_text,
            "dialogue_lines": dialogue_lines,
            "transcript_variants_json": json.dumps(variants_payload, ensure_ascii=False),
        }

    def _call_needs_secondary_backfill(
        self,
        call: CallRecord,
        *,
        secondary_provider: str,
    ) -> bool:
        payload = self._safe_json_dict(call.transcript_variants_json)
        state = self._secondary_backfill_state_from_payload(
            payload,
            secondary_provider=secondary_provider,
        )
        return state in {"fresh", "retry"}

    def backfill_secondary_asr(
        self,
        session: Session,
        limit: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, int]:
        primary_provider = (self._settings.transcribe_provider or "").strip().lower()
        secondary_provider = (self._settings.secondary_transcribe_provider or "").strip().lower()
        if not secondary_provider:
            return {
                "processed": 0,
                "success": 0,
                "failed": 0,
                "scanned_done": 0,
                "skipped_config": 1,
            }
        if secondary_provider == primary_provider:
            return {
                "processed": 0,
                "success": 0,
                "failed": 0,
                "scanned_done": 0,
                "skipped_config": 1,
            }

        done_calls = session.scalars(
            select(CallRecord)
            .where(CallRecord.dead_letter_stage.is_(None))
            .where(CallRecord.transcription_status == "done")
            .order_by(CallRecord.id.asc())
        ).all()

        fresh_candidates: list[CallRecord] = []
        retry_candidates: list[CallRecord] = []
        scanned_done = 0
        for call in done_calls:
            scanned_done += 1
            state = self._secondary_backfill_state_from_payload(
                self._safe_json_dict(call.transcript_variants_json),
                secondary_provider=secondary_provider,
            )
            if state == "fresh":
                fresh_candidates.append(call)
                if len(fresh_candidates) >= limit:
                    break
            elif state == "retry" and len(retry_candidates) < limit:
                retry_candidates.append(call)

        candidates = list(fresh_candidates)
        if len(candidates) < limit:
            candidates.extend(retry_candidates[: limit - len(candidates)])

        success = 0
        failed = 0
        partial = 0
        exhausted = 0
        total = len(candidates)

        def _assign_transcribe_result(call: CallRecord, result: Dict[str, Any]) -> None:
            self._export_transcript_file(call, result)
            call.transcript_manager = result["transcript_manager"]
            call.transcript_client = result["transcript_client"]
            call.transcript_text = result["transcript_text"]
            call.transcript_variants_json = result.get("transcript_variants_json")
            call.transcription_status = "done"
            call.resolve_status = "pending"
            call.resolve_attempts = 0
            call.resolve_json = None
            call.resolve_quality_score = None
            call.analysis_status = "pending"
            call.sync_status = "pending"
            call.next_retry_at = None
            call.dead_letter_stage = None

        def _emit_progress(payload: Dict[str, Any]) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(payload)
            except Exception:
                return

        _emit_progress(
            {
                "stage": "backfill_second_asr",
                "current": 0,
                "total": total,
                "success": 0,
                "failed": 0,
                "partial": 0,
                "exhausted": 0,
            }
        )

        for idx, call in enumerate(candidates, start=1):
            current_payload = self._safe_json_dict(call.transcript_variants_json)
            attempts = self._secondary_backfill_attempts(
                current_payload,
                secondary_provider=secondary_provider,
            ) + 1
            outcome = "success"
            error_text = ""
            try:
                result = self._transcribe_call(call)
                result_payload = self._safe_json_dict(result.get("transcript_variants_json"))
                backfill_state = self._secondary_backfill_state_from_payload(
                    result_payload,
                    secondary_provider=secondary_provider,
                )
                if backfill_state in {"fresh", "retry"}:
                    is_exhausted = attempts >= SECONDARY_BACKFILL_MAX_ATTEMPTS
                    result_payload = self._apply_secondary_backfill_meta(
                        result_payload or current_payload,
                        secondary_provider=secondary_provider,
                        attempts=attempts,
                        status="partial",
                        exhausted=is_exhausted,
                        error="secondary_variant_still_missing",
                    )
                    result["transcript_variants_json"] = json.dumps(result_payload, ensure_ascii=False)
                    _assign_transcribe_result(call, result)
                    call.last_error = (
                        f"backfill-second-asr: secondary variant still missing for {secondary_provider}"
                    )
                    if is_exhausted:
                        exhausted += 1
                        outcome = "exhausted"
                    else:
                        partial += 1
                        outcome = "partial"
                else:
                    result_payload = self._clear_secondary_backfill_meta(
                        result_payload or current_payload,
                        secondary_provider=secondary_provider,
                    )
                    result["transcript_variants_json"] = json.dumps(result_payload, ensure_ascii=False)
                    _assign_transcribe_result(call, result)
                    call.last_error = None
                    success += 1
            except Exception as exc:  # noqa: BLE001
                is_exhausted = attempts >= SECONDARY_BACKFILL_MAX_ATTEMPTS
                failed += 1
                outcome = "failed"
                error_text = str(exc)
                updated_payload = self._apply_secondary_backfill_meta(
                    current_payload,
                    secondary_provider=secondary_provider,
                    attempts=attempts,
                    status="failed",
                    exhausted=is_exhausted,
                    error=error_text,
                )
                call.transcript_variants_json = json.dumps(updated_payload, ensure_ascii=False)
                # Keep existing successful transcript intact when selective backfill fails.
                call.transcription_status = "done"
                call.last_error = f"backfill-second-asr: {exc}"
                if is_exhausted:
                    exhausted += 1
                    outcome = "exhausted"
            session.add(call)
            _emit_progress(
                {
                    "stage": "backfill_second_asr",
                    "current": idx,
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "partial": partial,
                    "exhausted": exhausted,
                    "status": outcome,
                    "call_id": call.id,
                    "source_filename": call.source_filename,
                    "error": error_text,
                }
            )

        session.commit()
        return {
            "processed": total,
            "success": success,
            "failed": failed,
            "partial": partial,
            "exhausted": exhausted,
            "scanned_done": scanned_done,
        }

    def count_secondary_backfill_pending(self, session: Session) -> Dict[str, Any]:
        primary_provider = (self._settings.transcribe_provider or "").strip().lower()
        secondary_provider = (self._settings.secondary_transcribe_provider or "").strip().lower()
        if not secondary_provider or secondary_provider == primary_provider:
            return {
                "enabled": False,
                "primary_provider": primary_provider,
                "secondary_provider": secondary_provider or None,
                "pending": 0,
                "retry_pending": 0,
                "exhausted": 0,
            }

        done_calls = session.scalars(
            select(CallRecord)
            .where(CallRecord.dead_letter_stage.is_(None))
            .where(CallRecord.transcription_status == "done")
            .where(CallRecord.transcript_variants_json.is_not(None))
            .order_by(CallRecord.id.asc())
        ).all()

        pending = 0
        retry_pending = 0
        exhausted = 0
        for call in done_calls:
            state = self._secondary_backfill_state_from_payload(
                self._safe_json_dict(call.transcript_variants_json),
                secondary_provider=secondary_provider,
            )
            if state in {"fresh", "retry"}:
                pending += 1
                if state == "retry":
                    retry_pending += 1
            elif state == "exhausted":
                exhausted += 1

        return {
            "enabled": True,
            "primary_provider": primary_provider,
            "secondary_provider": secondary_provider,
            "pending": pending,
            "retry_pending": retry_pending,
            "exhausted": exhausted,
        }

    def run(
        self,
        session: Session,
        limit: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, int]:
        now = self._utc_now()
        max_attempts = max(1, self._settings.transcribe_max_attempts)
        calls = session.scalars(
            select(CallRecord)
            .where(CallRecord.dead_letter_stage.is_(None))
            .where(CallRecord.transcription_status.in_(["pending", "failed"]))
            .where(CallRecord.transcribe_attempts < max_attempts)
            .where(or_(CallRecord.next_retry_at.is_(None), CallRecord.next_retry_at <= now))
            .order_by(CallRecord.id.asc())
            .limit(limit)
        ).all()

        success = 0
        failed = 0
        total = len(calls)

        def _emit_progress(payload: Dict[str, Any]) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(payload)
            except Exception:
                # Progress telemetry should never break the main transcription flow.
                return

        _emit_progress(
            {
                "stage": "transcribe",
                "current": 0,
                "total": total,
                "success": 0,
                "failed": 0,
            }
        )

        for idx, call in enumerate(calls, start=1):
            call.transcribe_attempts = int(call.transcribe_attempts or 0) + 1
            attempt = call.transcribe_attempts
            outcome = "success"
            error_text = ""
            try:
                result = self._transcribe_call(call)
                self._export_transcript_file(call, result)
                call.transcript_manager = result["transcript_manager"]
                call.transcript_client = result["transcript_client"]
                call.transcript_text = result["transcript_text"]
                call.transcript_variants_json = result.get("transcript_variants_json")
                call.transcription_status = "done"
                call.resolve_status = "pending"
                call.resolve_attempts = 0
                call.resolve_json = None
                call.resolve_quality_score = None
                call.analysis_status = "pending"
                call.sync_status = "pending"
                call.next_retry_at = None
                call.dead_letter_stage = None
                call.last_error = None
                success += 1
            except Exception as exc:  # noqa: BLE001
                call.last_error = f"transcribe: {exc}"
                if attempt >= max_attempts:
                    call.transcription_status = "dead"
                    call.dead_letter_stage = "transcribe"
                    call.next_retry_at = None
                else:
                    call.transcription_status = "failed"
                    call.next_retry_at = self._utc_now() + self._retry_delay(attempt)
                failed += 1
                outcome = "failed"
                error_text = str(exc)
            session.add(call)
            _emit_progress(
                {
                    "stage": "transcribe",
                    "current": idx,
                    "total": total,
                    "success": success,
                    "failed": failed,
                    "status": outcome,
                    "call_id": call.id,
                    "source_filename": call.source_filename,
                    "error": error_text,
                }
            )

        session.commit()
        return {"processed": len(calls), "success": success, "failed": failed}
