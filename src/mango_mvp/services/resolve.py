from __future__ import annotations

import csv
import difflib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from mango_mvp.clients.ollama import OllamaClient
from mango_mvp.config import Settings
from mango_mvp.models import CallRecord
from mango_mvp.services.transcribe import TranscribeService


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
}"""

DIALOGUE_RESOLVE_SYSTEM_PROMPT = """You improve a turn-by-turn Russian sales phone call dialogue.
Rules:
1) Use only information from the provided baseline turns and role variants. Do not invent facts.
2) Preserve the set of turn_id values. Do not add new turns.
3) Keep ts_sec unchanged. Do not rewrite timestamps.
4) final_text must stay close to baseline/variant wording. If uncertain, keep baseline_text.
5) You may set drop=true only for obvious artifact, exact echo, or duplicated garbage.
6) You may set swap_with_next=true only when two adjacent turns are clearly in the wrong order.
7) Speaker should normally stay unchanged. Change speaker only if the baseline speaker is clearly wrong.
8) Return strict JSON only:
{
  "schema_version": "dialogue_resolve_result_v1",
  "turns": [
    {
      "turn_id": 1,
      "speaker": "manager|client|unknown",
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
No markdown, no extra keys."""


TIMED_LINE_RE = re.compile(
    r"^\[(?P<approx>~)?(?P<mm>\d{2}):(?P<ss>\d{2}(?:\.\d)?)\]\s+"
    r"(?P<speaker>Менеджер(?:\s*\([^)]+\))?|Клиент|Спикер\s*\(не определен\)):\s*(?P<text>.*)$"
)
WORD_RE = re.compile(r"\S+", flags=re.UNICODE)
ARTIFACT_RE = re.compile(r"продолжение следует|голосовой ассистент|абонент недоступен", re.I)


def _clamp_score(value: int) -> int:
    return max(0, min(100, int(value)))


class ResolveService:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._transcribe_helper = TranscribeService(settings)
        self._ollama_client: Optional[OllamaClient] = None
        self._openai_client: Optional[OpenAI] = None
        self._rescue_service_cache: Dict[Tuple[str, bool], TranscribeService] = {}

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _retry_delay(self, attempts: int) -> timedelta:
        base = max(1, self._settings.retry_base_delay_sec)
        multiplier = max(1, 2 ** max(0, attempts - 1))
        return timedelta(seconds=base * multiplier)

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
        return Path(export_dir) / source_path.parent.name / f"{source_path.stem}_text.txt"

    def _load_dialogue_lines_from_export(self, call: CallRecord) -> List[str]:
        path = self._dialogue_export_path(call)
        if not path or not path.exists():
            return []
        return [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]

    @staticmethod
    def _parse_timed_line(line: str) -> Optional[Dict[str, Any]]:
        match = TIMED_LINE_RE.match(str(line).strip())
        if not match:
            return None
        mm = int(match.group("mm"))
        ss = float(match.group("ss"))
        ts = mm * 60.0 + ss
        speaker = (match.group("speaker") or "").strip()
        if speaker.startswith("Менеджер"):
            role = "manager"
        elif speaker.startswith("Клиент"):
            role = "client"
        else:
            role = "unknown"
        return {
            "ts_sec": ts,
            "approximate": bool(match.group("approx")),
            "speaker_label": speaker,
            "role": role,
            "text": (match.group("text") or "").strip(),
            "raw_line": str(line).strip(),
        }

    def _parse_dialogue_lines(
        self,
        call: CallRecord,
        dialogue_lines: Optional[List[str]],
        *,
        allow_export_fallback: bool = False,
    ) -> List[Tuple[float, str, str]]:
        rows: List[Tuple[float, str, str]] = []
        lines: List[str] = []
        if dialogue_lines:
            lines = [str(line).strip() for line in dialogue_lines if str(line).strip()]
        elif allow_export_fallback:
            path = self._dialogue_export_path(call)
            if path and path.exists():
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for raw in lines:
            parsed = self._parse_timed_line(raw)
            if parsed is None:
                continue
            rows.append(
                (
                    float(parsed.get("ts_sec", 0.0)),
                    str(parsed.get("role") or "unknown"),
                    str(parsed.get("text") or "").strip(),
                )
            )
        return rows

    @staticmethod
    def _format_timecode(seconds: float, approximate: bool = False) -> str:
        total_ms = max(0, int(round(seconds * 1000.0)))
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

    def _postfilter_same_ts_dialogue_lines(self, dialogue_lines: List[str]) -> Dict[str, Any]:
        if not dialogue_lines:
            return {"dialogue_lines": dialogue_lines, "adjusted": 0}

        out: List[str] = []
        adjusted = 0
        prev_ts: Optional[float] = None
        prev_role: Optional[str] = None
        min_step_sec = 0.1

        for raw in dialogue_lines:
            line = str(raw).strip()
            match = TIMED_LINE_RE.match(line)
            if not match:
                out.append(line)
                continue

            mm = int(match.group("mm"))
            ss = float(match.group("ss"))
            ts = mm * 60.0 + ss
            approximate = bool(match.group("approx"))
            speaker = (match.group("speaker") or "").strip()
            text = (match.group("text") or "").strip()
            role = "unknown"
            if speaker.startswith("Менеджер"):
                role = "manager"
            elif speaker.startswith("Клиент"):
                role = "client"

            safe_ts = ts
            if prev_ts is not None and safe_ts < prev_ts - 1e-6:
                safe_ts = prev_ts + min_step_sec
            if (
                self._settings.resolve_postfilter_same_ts
                and prev_ts is not None
                and prev_role is not None
                and role != prev_role
                and abs(safe_ts - prev_ts) <= 1e-6
            ):
                safe_ts = prev_ts + min_step_sec

            if abs(safe_ts - ts) > 1e-6:
                adjusted += 1

            prev_ts = safe_ts
            prev_role = role
            out.append(f"{self._format_timecode(safe_ts, approximate=approximate)} {speaker}: {text}")

        return {"dialogue_lines": out, "adjusted": adjusted}

    def _maybe_postfilter_candidate_dialogue(
        self,
        call: CallRecord,
        candidate: Dict[str, Any],
    ) -> Dict[str, Any]:
        name = str(candidate.get("name") or "")
        lines = candidate.get("dialogue_lines")
        if (not isinstance(lines, list) or not lines) and name == "baseline":
            lines = self._load_dialogue_lines_from_export(call)
        if not isinstance(lines, list) or not lines:
            return candidate

        meta = candidate.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        rows_before = self._parse_dialogue_lines(call, lines, allow_export_fallback=False)
        if rows_before:
            before_metrics = self._line_metrics(rows_before)
            meta["same_ts_events_before_postfilter"] = int(
                before_metrics.get("same_ts_cross_speaker_events", 0) or 0
            )

        fixed = self._postfilter_same_ts_dialogue_lines(lines)
        adjusted = int(fixed.get("adjusted") or 0)
        if adjusted <= 0:
            candidate["dialogue_lines"] = lines
            candidate["meta"] = meta
            return candidate

        candidate["dialogue_lines"] = fixed["dialogue_lines"]
        meta["same_ts_postfilter_adjusted_lines"] = adjusted
        rows_after = self._parse_dialogue_lines(
            call,
            fixed["dialogue_lines"],
            allow_export_fallback=False,
        )
        if rows_after:
            after_metrics = self._line_metrics(rows_after)
            meta["same_ts_events_after_postfilter"] = int(
                after_metrics.get("same_ts_cross_speaker_events", 0) or 0
            )
        candidate["meta"] = meta

        payload = self._safe_json(str(candidate.get("transcript_variants_json") or ""))
        if payload:
            warnings = self._get_warnings(payload)
            marker = f"resolve_same_ts_postfilter: adjusted_lines={adjusted}"
            if marker not in warnings:
                warnings.append(marker)
            payload["warnings"] = warnings
            payload["resolve_same_ts_postfilter"] = {
                "applied": True,
                "adjusted_lines": adjusted,
            }
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
                        self._settings.codex_merge_model,
                        "--output-last-message",
                        out_file.name,
                    ]
                    reasoning_effort = (self._settings.codex_reasoning_effort or "").strip().lower()
                    if reasoning_effort in {"low", "medium", "high"}:
                        cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
                    cmd.append(
                        (
                            f"{RESOLVE_SYSTEM_PROMPT}\n\n"
                            f"{user_prompt}"
                        )
                    )
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
            return {
                "merged_text": merged,
                "selection": selection,
                "confidence": confidence,
                "provider": provider,
                "notes": str(payload.get("notes", "")).strip(),
                "similarity": round(similarity, 4),
            }
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
                continue
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
                    "ts_label": self._format_timecode(
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
                    self._settings.codex_merge_model,
                    "--output-last-message",
                    out_file.name,
                ]
                reasoning_effort = (self._settings.codex_reasoning_effort or "").strip().lower()
                if reasoning_effort in {"low", "medium", "high"}:
                    cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
                cmd.append(f"{DIALOGUE_RESOLVE_SYSTEM_PROMPT}\n\n{user_prompt}")
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
            payload = self._transcribe_helper._extract_json_payload(raw)
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
        speaker_corrections = 0
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
            if requested_role not in {"manager", "client", "unknown"}:
                requested_role = role
            if requested_role != role:
                if role == "unknown" or "same_ts_cross" in turn_flags:
                    role = requested_role
                    speaker_corrections += 1
                else:
                    warnings.append(f"speaker_change_ignored:{turn_id}")

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

        swaps_applied = 0
        ordered = normalized[:]
        idx = 0
        while idx < len(ordered) - 1:
            current = ordered[idx]
            if not bool(current.get("swap_with_next")):
                idx += 1
                continue
            if bool(ordered[idx + 1].get("swap_with_next")):
                warnings.append(f"swap_chain_ignored:{current['turn_id']}")
                current["swap_with_next"] = False
                idx += 1
                continue
            ordered[idx], ordered[idx + 1] = ordered[idx + 1], ordered[idx]
            swaps_applied += 1
            idx += 2

        kept_turns = [turn for turn in ordered if not bool(turn.get("drop")) and str(turn.get("final_text") or "").strip()]
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
            "swaps_applied": swaps_applied,
            "drops_requested": drops_requested,
            "speaker_corrections": speaker_corrections,
        }

    def _dialogue_turns_to_candidate(
        self,
        call: CallRecord,
        variants_payload: Dict[str, Any],
        normalized_result: Dict[str, Any],
        *,
        provider: str,
    ) -> Dict[str, Any]:
        manager_name = (
            (call.manager_name or "").strip()
            or self._transcribe_helper._extract_manager_name_from_filename(call.source_filename)
        )
        manager_label = f"Менеджер ({manager_name})"
        manager_parts: List[str] = []
        client_parts: List[str] = []
        dialogue_lines: List[str] = []
        for turn in normalized_result.get("turns", []):
            role = str(turn.get("speaker") or "unknown")
            text = str(turn.get("final_text") or "").strip()
            if not text:
                continue
            ts_sec = float(turn.get("ts_sec") or 0.0)
            approximate = bool(turn.get("approximate"))
            if role == "manager":
                speaker_label = manager_label
                manager_parts.append(text)
            elif role == "client":
                speaker_label = "Клиент"
                client_parts.append(text)
            else:
                speaker_label = "Спикер (не определен)"
            dialogue_lines.append(
                f"{self._format_timecode(ts_sec, approximate=approximate)} {speaker_label}: {text}"
            )

        manager_text = " ".join(manager_parts).strip()
        client_text = " ".join(client_parts).strip()
        if manager_text or client_text:
            transcript_text = f"MANAGER:\n{manager_text}\n\nCLIENT:\n{client_text}"
        else:
            transcript_text = "\n".join(dialogue_lines).strip()

        payload = self._copy_payload(variants_payload)
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
            "warnings": normalized_result.get("warnings", []),
            "global_notes": str(normalized_result.get("global_notes") or "").strip(),
        }
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
        baseline_dialogue_lines = self._load_dialogue_lines_from_export(call)
        input_payload = self._build_dialogue_resolve_payload(
            call,
            variants_payload,
            baseline_dialogue_lines,
        )
        if not input_payload:
            return None
        raw_result = self._run_dialogue_llm(input_payload)
        normalized_result = self._normalize_dialogue_result(input_payload, raw_result)
        return self._dialogue_turns_to_candidate(
            call,
            variants_payload,
            normalized_result,
            provider=f"{provider}_dialogue",
        )

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
            llm_provider = "rule"
        contextual_provider = f"{llm_provider}_contextual" if llm_provider != "rule" else "rule"

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
            return {
                "name": "llm",
                "transcript_manager": manager_text,
                "transcript_client": client_text,
                "transcript_text": transcript_text,
                "dialogue_lines": None,
                "transcript_variants_json": json.dumps(payload, ensure_ascii=False),
                "meta": {
                    "mode": "stereo",
                    "provider": contextual_provider,
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
            },
        }

    def _rescue_provider(self) -> str:
        configured = (self._settings.resolve_rescue_provider or "").strip().lower()
        if configured:
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
        result["name"] = "rescue"
        result["meta"] = {
            "provider": provider,
            "dual": dual,
        }
        return result

    def _candidate_from_call(self, call: CallRecord) -> Dict[str, Any]:
        return {
            "name": "baseline",
            "transcript_manager": call.transcript_manager,
            "transcript_client": call.transcript_client,
            "transcript_text": call.transcript_text or "",
            "dialogue_lines": self._load_dialogue_lines_from_export(call),
            "transcript_variants_json": call.transcript_variants_json or "{}",
            "meta": {"provider": "baseline"},
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

    def run(self, session: Session, limit: int) -> Dict[str, int]:
        return self.run_with_progress(session, limit=limit, progress_callback=None)

    def run_with_progress(
        self,
        session: Session,
        limit: int,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, int]:
        now = self._utc_now()
        max_attempts = max(1, self._settings.resolve_max_attempts)
        candidate_calls = session.scalars(
            select(CallRecord)
            .where(CallRecord.transcription_status == "done")
            .where(CallRecord.dead_letter_stage.is_(None))
            .where(CallRecord.resolve_status.in_(["pending", "failed"]))
            .where(CallRecord.resolve_attempts < max_attempts)
            .where(or_(CallRecord.next_retry_at.is_(None), CallRecord.next_retry_at <= now))
            .order_by(CallRecord.id.asc())
        ).all()
        calls: List[CallRecord] = []
        for call in candidate_calls:
            if self._waiting_for_secondary_asr(call):
                continue
            calls.append(call)
            if len(calls) >= limit:
                break

        success = 0
        failed = 0
        manual = 0
        skipped = 0
        llm_used = 0
        rescue_used = 0

        def _emit_progress(payload: Dict[str, Any]) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(payload)
            except Exception:
                return

        _emit_progress(
            {
                "stage": "resolve",
                "current": 0,
                "total": len(calls),
                "success": 0,
                "failed": 0,
                "manual": 0,
                "skipped_short": 0,
                "llm_used": 0,
                "rescue_used": 0,
            }
        )

        for idx, call in enumerate(calls, start=1):
            call.resolve_attempts = int(call.resolve_attempts or 0) + 1
            attempt = call.resolve_attempts
            outcome = "success"
            error_text = ""
            try:
                duration = float(call.duration_sec or 0.0)
                if duration > 0.0 and duration < float(self._settings.resolve_min_duration_sec):
                    call.resolve_status = "skipped"
                    call.resolve_quality_score = 100.0
                    call.resolve_json = json.dumps(
                        {
                            "version": "v1",
                            "decision": "skip_short_call",
                            "duration_sec": round(duration, 3),
                            "min_duration_sec": int(self._settings.resolve_min_duration_sec),
                            "ts_utc": self._utc_now().isoformat(),
                        },
                        ensure_ascii=False,
                    )
                    call.analysis_status = "pending"
                    call.sync_status = "pending"
                    call.next_retry_at = None
                    call.last_error = None
                    skipped += 1
                    success += 1
                    session.add(call)
                    outcome = "skipped_short"
                    _emit_progress(
                        {
                            "stage": "resolve",
                            "current": idx,
                            "total": len(calls),
                            "success": success,
                            "failed": failed,
                            "manual": manual,
                            "skipped_short": skipped,
                            "llm_used": llm_used,
                            "rescue_used": rescue_used,
                            "status": outcome,
                            "call_id": call.id,
                            "source_filename": call.source_filename,
                            "error": error_text,
                        }
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
                best_meta = best.get("meta")
                if not isinstance(best_meta, dict):
                    best_meta = {}
                postfilter_adjusted = int(best_meta.get("same_ts_postfilter_adjusted_lines") or 0)

                if best_score >= accept_threshold:
                    if best_name != "baseline":
                        call.transcript_manager = best.get("transcript_manager")
                        call.transcript_client = best.get("transcript_client")
                        call.transcript_text = str(best.get("transcript_text") or "")
                    if isinstance(best.get("transcript_variants_json"), str):
                        call.transcript_variants_json = str(best.get("transcript_variants_json") or "{}")

                    should_export = best_name != "baseline" or postfilter_adjusted > 0
                    if should_export:
                        self._transcribe_helper._export_transcript_file(
                            call,
                            {
                                "transcript_manager": call.transcript_manager,
                                "transcript_client": call.transcript_client,
                                "transcript_text": call.transcript_text or "",
                                "dialogue_lines": best.get("dialogue_lines"),
                                "transcript_variants_json": call.transcript_variants_json or "{}",
                            },
                        )
                    decision = f"accept_{best_name}"
                    call.resolve_status = "done"
                    call.analysis_status = "pending"
                    call.sync_status = "pending"
                    success += 1
                    outcome = "done"
                else:
                    decision = "manual_review_required"
                    call.resolve_status = "manual"
                    manual += 1
                    outcome = "manual"

                call.resolve_quality_score = float(best_score)
                call.resolve_json = json.dumps(
                    self._build_resolve_payload(
                        duration_sec=duration,
                        decision=decision,
                        baseline=baseline,
                        llm_candidate=llm_candidate,
                        rescue_candidate=rescue_candidate,
                        chosen=best,
                    ),
                    ensure_ascii=False,
                )
                call.next_retry_at = None
                call.dead_letter_stage = None
                call.last_error = None
            except Exception as exc:  # noqa: BLE001
                call.last_error = f"resolve: {exc}"
                if attempt >= max_attempts:
                    call.resolve_status = "dead"
                    call.dead_letter_stage = "resolve"
                    call.next_retry_at = None
                else:
                    call.resolve_status = "failed"
                    call.next_retry_at = self._utc_now() + self._retry_delay(attempt)
                failed += 1
                outcome = "failed"
                error_text = str(exc)
            session.add(call)
            _emit_progress(
                {
                    "stage": "resolve",
                    "current": idx,
                    "total": len(calls),
                    "success": success,
                    "failed": failed,
                    "manual": manual,
                    "skipped_short": skipped,
                    "llm_used": llm_used,
                    "rescue_used": rescue_used,
                    "status": outcome,
                    "call_id": call.id,
                    "source_filename": call.source_filename,
                    "error": error_text,
                }
            )

        session.commit()
        return {
            "processed": len(calls),
            "success": success,
            "failed": failed,
            "manual": manual,
            "skipped_short": skipped,
            "llm_used": llm_used,
            "rescue_used": rescue_used,
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
