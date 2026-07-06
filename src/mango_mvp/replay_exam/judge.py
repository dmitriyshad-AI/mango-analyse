from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .models import BotReplayResult, ReplayCase


@dataclass(frozen=True)
class PairedJudgePayload:
    payload: dict[str, object]
    hidden_key: dict[str, str]


def build_replay_judge_payload(case: ReplayCase, baseline: BotReplayResult, candidate: BotReplayResult, *, seed: str) -> PairedJudgePayload:
    digest = hashlib.sha256(f"{seed}:{case.turn_id}".encode("utf-8")).digest()
    swap = bool(digest[0] % 2)
    first_name, second_name = ("candidate", "baseline") if swap else ("baseline", "candidate")
    first = candidate if swap else baseline
    second = baseline if swap else candidate
    payload = {
        "judge_version": "replay_judge_v1",
        "metric": "chat_only_replay",
        "dialog_id": case.dialog_id,
        "turn_id": case.turn_id,
        "brand": case.brand,
        "client_message": case.client_message,
        "manager_reference": case.manager_reference,
        "answer_a": {"route": first.route, "text": first.bot_text},
        "answer_b": {"route": second.route, "text": second.bot_text},
        "rubric": [
            "Would a manager likely send this without edits?",
            "Does it avoid unsupported numbers, wrong brand, P0 weakening, and private data?",
            "Does it answer the client's actual question rather than a nearby one?",
        ],
    }
    return PairedJudgePayload(payload=payload, hidden_key={"answer_a": first_name, "answer_b": second_name})


def serialize_judge_payload(payload: PairedJudgePayload) -> str:
    return json.dumps(payload.payload, ensure_ascii=False, sort_keys=True)
