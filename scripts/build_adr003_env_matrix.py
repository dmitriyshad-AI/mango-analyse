#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from mango_mvp.channels.subscription_llm_parts.reliable_answerer import RELIABLE_ANSWERER_STEP1_ENV
from mango_mvp.channels.subscription_llm_parts.semantic_reading import SEMANTIC_READING_CLASSES_ENV
from mango_mvp.channels.subscription_llm_parts.support import (
    DIRECT_PATH_PILOT_CONFIG_ENV,
    DIRECT_PATH_PILOT_CONFIG_VERSION,
    DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS,
)
from mango_mvp.channels.subscription_llm_parts.direct_path import SEMANTIC_FRAME_SHADOW_ENV


def build_markdown() -> str:
    profile_flags = tuple(DIRECT_PATH_PILOT_PROFILE_DEFAULT_ON_FLAGS)
    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# ADR003 E3 env matrix",
        "",
        f"Generated at: `{created_at}`",
        "",
        "## A. Profile default-on flags",
        "",
        f"Profile env: `{DIRECT_PATH_PILOT_CONFIG_ENV}={DIRECT_PATH_PILOT_CONFIG_VERSION}`",
        "",
        "| flag | leg | expected effect |",
        "|---|---|---|",
    ]
    for flag in profile_flags:
        lines.append(f"| `{flag}` | B and ON | enabled by pilot profile |")
    lines.extend(
        [
            "",
            "## B. Production-parity env outside profile",
            "",
            "| flag | leg | expected effect |",
            "|---|---|---|",
            f"| `{RELIABLE_ANSWERER_STEP1_ENV}=1` | B and ON | keep reliable answerer parity for `sense_seats` measurement |",
            f"| `{SEMANTIC_FRAME_SHADOW_ENV}=1` | B and ON | provide the same inline SemanticFrame payload for readers |",
            "",
            "## C. ON-only reading delta",
            "",
            "| flag | leg | expected effect | negative controls |",
            "|---|---|---|---|",
            f"| `{SEMANTIC_READING_CLASSES_ENV}=sense_seats,off_topic,slots_gsf,intent_actions,<target>` | ON only | enable the current profile readers plus one target reader for measurement | P0, brand, metadata-only off-topic, slot leak tests |",
            "",
            "## Notes",
            "",
            "- `B` and `ON` both use `pilot_gold_v1` and reliable answerer.",
            "- Profile defaults now include `sense_seats,off_topic,slots_gsf,intent_actions`; ON must preserve them when adding a target class.",
            "- `ON` differs from `B` only by the additional target class in `TELEGRAM_SEMANTIC_READING_CLASSES`.",
            "- No profile tuple changes are made by this stage.",
            "- Live bot, P0 floor/preblock and legacy deletion are out of scope.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/ADR003_E3_ENV_MATRIX.md"))
    args = parser.parse_args()
    text = build_markdown()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
