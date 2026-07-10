#!/usr/bin/env python3
"""Print the ADR-003 one-commit rollback plan for a red M1 class.

The script is intentionally read-only: it does not edit files or change env.
It is a rehearsed checklist generator for the post-M1 decision point.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


OLD_READING_CLASSES = "sense_seats,slots_gsf,off_topic,intent_actions,live_status_read"
OLD_APPLY_CLASSES = "live_status_read/conversation_intent_plan"


@dataclass(frozen=True)
class RedSwitch:
    key: str
    env: str
    default_file: str
    default_symbol: str
    reading_remove: tuple[str, ...] = ()
    apply_remove: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        env_disable = {} if "/" in self.env else {self.env: "0"}
        patch_steps = [
            f"Remove {self.env} from {self.default_symbol} in {self.default_file}.",
        ]
        if self.reading_remove:
            patch_steps.append(
                "Remove reading classes from PILOT_PROFILE_DEFAULT_READING_CLASSES: "
                + ", ".join(self.reading_remove)
                + "."
            )
        if self.apply_remove:
            patch_steps.append(
                "Remove apply classes from PILOT_PROFILE_DEFAULT_APPLY_CLASSES: "
                + ", ".join(self.apply_remove)
                + "."
            )
        return {
            "key": self.key,
            "env_disable": env_disable,
            "old_profile_overlay": {
                "TELEGRAM_SEMANTIC_READING_CLASSES": OLD_READING_CLASSES,
                "TELEGRAM_READING_APPLY_CLASSES": OLD_APPLY_CLASSES,
            },
            "one_commit_patch_steps": patch_steps,
            "validation": [
                "Run profile matrix tests for the touched flag.",
                "Run the focused failing M1 class fixture locally without LLM if available.",
                "Run: PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_adr003_semantic_reading_e3_runner.py tests/test_semantic_reading.py tests/test_subscription_llm_draft_provider.py tests/test_dialogue_memory.py",
            ],
            "do_not_change": [
                "P0 floors",
                "brand floors",
                "payment/PII hard gates",
                "M1 package a246ece2 artifacts",
            ],
            "notes": list(self.notes),
        }


SWITCHES: dict[str, RedSwitch] = {
    "fact_select_frame": RedSwitch(
        key="fact_select_frame",
        env="TELEGRAM_FACT_SELECT_FRAME",
        default_file="src/mango_mvp/channels/subscription_llm_parts/support.py",
        default_symbol="DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS",
        reading_remove=("fact_select_read",),
        notes=("If only apply is red but trace is useful, remove the env flag first; remove reading only when Fable says trace itself is unsafe.",),
    ),
    "tone_close_frame_veto": RedSwitch(
        key="tone_close_frame_veto",
        env="TELEGRAM_TONE_CLOSE_FRAME_VETO",
        default_file="src/mango_mvp/channels/subscription_llm_parts/support.py",
        default_symbol="DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS",
    ),
    "p0_model_led": RedSwitch(
        key="p0_model_led",
        env="TELEGRAM_P0_MODEL_LED",
        default_file="src/mango_mvp/channels/subscription_llm_parts/support.py",
        default_symbol="DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS",
    ),
    "prose_model_led": RedSwitch(
        key="prose_model_led",
        env="TELEGRAM_PROSE_MODEL_LED",
        default_file="src/mango_mvp/channels/subscription_llm_parts/support.py",
        default_symbol="DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS",
    ),
    "payment_refund_dispute_split": RedSwitch(
        key="payment_refund_dispute_split",
        env="TELEGRAM_PAYMENT_REFUND_DISPUTE_SPLIT",
        default_file="src/mango_mvp/channels/subscription_llm_parts/support.py",
        default_symbol="DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS",
    ),
    "seats_default_open": RedSwitch(
        key="seats_default_open",
        env="TELEGRAM_SEATS_DEFAULT_OPEN",
        default_file="src/mango_mvp/channels/subscription_llm_parts/support.py",
        default_symbol="DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS",
    ),
    "p0_latch_autorelease_v2": RedSwitch(
        key="p0_latch_autorelease_v2",
        env="TELEGRAM_P0_LATCH_AUTORELEASE_V2",
        default_file="src/mango_mvp/channels/dialogue_memory.py",
        default_symbol="MEMORY_PROFILE_DEFAULT_ON_FLAGS",
    ),
    "route_templates": RedSwitch(
        key="route_templates",
        env="TELEGRAM_SEMANTIC_READING_CLASSES/TELEGRAM_READING_APPLY_CLASSES",
        default_file="src/mango_mvp/channels/subscription_llm_parts/semantic_reading.py",
        default_symbol="PILOT_PROFILE_DEFAULT_READING_CLASSES/PILOT_PROFILE_DEFAULT_APPLY_CLASSES",
        reading_remove=("route_templates",),
        apply_remove=("route_templates/autonomy_matrix",),
    ),
    "reask_read": RedSwitch(
        key="reask_read",
        env="TELEGRAM_SEMANTIC_READING_CLASSES/TELEGRAM_READING_APPLY_CLASSES",
        default_file="src/mango_mvp/channels/subscription_llm_parts/semantic_reading.py",
        default_symbol="PILOT_PROFILE_DEFAULT_READING_CLASSES/PILOT_PROFILE_DEFAULT_APPLY_CLASSES",
        reading_remove=("reask_read",),
        apply_remove=("reask_read/final_text",),
    ),
    "roles_read": RedSwitch(
        key="roles_read",
        env="TELEGRAM_SEMANTIC_READING_CLASSES/TELEGRAM_READING_APPLY_CLASSES",
        default_file="src/mango_mvp/channels/subscription_llm_parts/semantic_reading.py",
        default_symbol="PILOT_PROFILE_DEFAULT_READING_CLASSES/PILOT_PROFILE_DEFAULT_APPLY_CLASSES",
        reading_remove=("roles_read",),
        apply_remove=("roles_read/refund_tax",),
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Print ADR-003 red-switch rollback plan.")
    parser.add_argument("key", nargs="?", choices=sorted(SWITCHES), help="Class/flag to disable.")
    parser.add_argument("--list", action="store_true", help="List supported keys.")
    args = parser.parse_args()

    if args.list or not args.key:
        print("\n".join(sorted(SWITCHES)))
        return 0
    print(json.dumps(SWITCHES[args.key].to_payload(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
