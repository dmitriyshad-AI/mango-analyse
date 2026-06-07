from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


def iter_shadow_rows(path: Path) -> Iterable[dict[str, Any]]:
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            dialog = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
        dialog_id = str(dialog.get("dialog_id") or dialog.get("id") or "")
        turns = dialog.get("turns") if isinstance(dialog.get("turns"), list) else []
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            turn_no = turn.get("turn")
            events = _shadow_events(turn)
            for event in events:
                yield {
                    "dialog_id": dialog_id,
                    "turn": turn_no,
                    "site": str(event.get("site") or "unknown"),
                    "available": bool(event.get("available")),
                    "unsupported": list(event.get("unsupported") or []),
                    "verdicts": _event_verdicts(event),
                }


def summarize_file(path: Path) -> dict[str, Any]:
    rows = list(iter_shadow_rows(path))
    by_site: Counter[str] = Counter()
    by_verdict: Counter[str] = Counter()
    unavailable_by_site: Counter[str] = Counter()
    for row in rows:
        site = str(row["site"])
        by_site[site] += 1
        if not row["available"]:
            unavailable_by_site[site] += 1
        verdicts = row["verdicts"] or ["no_claims"]
        for verdict in verdicts:
            by_verdict[f"{site}:{verdict}"] += 1
    return {
        "rows": rows,
        "by_site": dict(sorted(by_site.items())),
        "by_verdict": dict(sorted(by_verdict.items())),
        "unavailable_by_site": dict(sorted(unavailable_by_site.items())),
    }


def render_summary(summary: Mapping[str, Any]) -> str:
    lines = ["site\tverdicts\tavailable\tdialog_id\tturn"]
    for row in summary.get("rows") or []:
        verdicts = ",".join(row.get("verdicts") or ["no_claims"])
        lines.append(
            "\t".join(
                [
                    str(row.get("site") or "unknown"),
                    verdicts,
                    str(bool(row.get("available"))).lower(),
                    str(row.get("dialog_id") or ""),
                    str(row.get("turn") or ""),
                ]
            )
        )
    lines.append("")
    lines.append("Counters by site:")
    for site, count in (summary.get("by_site") or {}).items():
        lines.append(f"- {site}: {count}")
    lines.append("Counters by site/verdict:")
    for key, count in (summary.get("by_verdict") or {}).items():
        lines.append(f"- {key}: {count}")
    lines.append("Unavailable by site:")
    for site, count in (summary.get("unavailable_by_site") or {}).items():
        lines.append(f"- {site}: {count}")
    return "\n".join(lines)


def _shadow_events(turn: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    direct = turn.get("bot_faithfulness_shadow")
    if isinstance(direct, list):
        return [event for event in direct if isinstance(event, Mapping)]
    pipeline = turn.get("bot_dialogue_contract_pipeline")
    if isinstance(pipeline, Mapping):
        nested = pipeline.get("faithfulness_shadow")
        if isinstance(nested, list):
            return [event for event in nested if isinstance(event, Mapping)]
    return []


def _event_verdicts(event: Mapping[str, Any]) -> list[str]:
    verdicts: list[str] = []
    raw = event.get("verdicts")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            verdict = str(item.get("verdict") or "").strip()
            if verdict:
                verdicts.append(verdict)
    unsupported = event.get("unsupported")
    if isinstance(unsupported, list) and unsupported and not verdicts:
        verdicts.append("unsupported")
    return verdicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize TELEGRAM_FAITHFULNESS_SHADOW events from dynamic transcripts.")
    parser.add_argument("transcripts", type=Path, help="Path to dynamic_dialog_transcripts.jsonl")
    args = parser.parse_args(argv)
    print(render_summary(summarize_file(args.transcripts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
