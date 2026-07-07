from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable

from .models import BotReplayResult, ReplayCase


@dataclass(frozen=True)
class PairedJudgePayload:
    payload: dict[str, object]
    hidden_key: dict[str, str]


def _case_prefix_payload(case: ReplayCase) -> list[dict[str, object]]:
    return [
        {"from_me": message.from_me, "text": message.text, "ts_masked": message.ts_masked}
        for message in case.prefix_messages
    ]


def _facts_digest(case: ReplayCase) -> str:
    payload = {
        "brand": case.brand,
        "segment": case.segment,
        "expected_p0": case.expected_p0,
        "metadata_keys": sorted(str(key) for key in case.metadata),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def build_replay_judge_payload(
    case: ReplayCase,
    baseline: BotReplayResult,
    candidate: BotReplayResult,
    *,
    seed: str,
    swap: bool | None = None,
) -> PairedJudgePayload:
    if swap is None:
        digest = hashlib.sha256(f"{seed}:{case.turn_id}".encode("utf-8")).digest()
        swap = bool(digest[0] % 2)
    first_name, second_name = ("candidate", "baseline") if swap else ("baseline", "candidate")
    first = candidate if swap else baseline
    second = baseline if swap else candidate
    payload = {
        "judge_version": "replay_judge_v1",
        "payload_schema": "replay_judge_payload_v4",
        "metric": "chat_only_replay",
        "exam_id": case.turn_id,
        "turn_index": case.turn_index,
        "contour": case.contour or case.brand,
        "dialog_key_masked": case.dialog_key_masked or case.dialog_id,
        "brand": case.brand,
        "prefix_messages": _case_prefix_payload(case),
        "client_message": case.client_message,
        "client_safe_facts_digest": _facts_digest(case),
        "answer_a": {"route": first.route, "text": first.bot_text},
        "answer_b": {"route": second.route, "text": second.bot_text},
        "rubric": [
            "Would a manager likely send this without edits?",
            "Does it avoid unsupported numbers, wrong brand, P0 weakening, and private data?",
            "Does it answer the client's actual question rather than a nearby one?",
        ],
    }
    return PairedJudgePayload(payload=payload, hidden_key={"answer_a": first_name, "answer_b": second_name})


def build_balanced_replay_judge_payloads(
    rows: Iterable[tuple[ReplayCase, BotReplayResult, BotReplayResult]],
    *,
    seed: str,
) -> list[PairedJudgePayload]:
    ordered = sorted(rows, key=lambda item: (item[0].turn_id, item[0].dialog_id))
    return [
        build_replay_judge_payload(case, baseline, candidate, seed=seed, swap=bool(index % 2))
        for index, (case, baseline, candidate) in enumerate(ordered)
    ]


def serialize_judge_payload(payload: PairedJudgePayload) -> str:
    return json.dumps(payload.payload, ensure_ascii=False, sort_keys=True)
