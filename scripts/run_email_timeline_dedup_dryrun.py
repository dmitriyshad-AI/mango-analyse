#!/usr/bin/env python3
"""Read-only dry-run for content duplicate e-mail events in customer_timeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mango_mvp.customer_timeline.ids import stable_digest, stable_prefixed_id  # noqa: E402
from mango_mvp.customer_timeline.store import (  # noqa: E402
    normalize_email_content_text,
    parse_datetime,
    timeline_email_content_key,
)


DEFAULT_PROD_DB = Path(
    "/Users/dmitrijfabarisov/Projects/Mango analyse/"
    "product_data/customer_timeline/customer_timeline_prod_20260621/customer_timeline.sqlite"
)
DEFAULT_REPORT = Path("/Users/dmitrijfabarisov/Claude Projects/Foton/dedup_email_dryrun.md")
DEFAULT_LOCAL_RAW = PROJECT_ROOT / ".codex_local/email_timeline_dedup/dedup_email_dryrun_raw.json"


@dataclass(frozen=True)
class EmailRow:
    event_id: str
    tenant_id: str
    customer_id: str | None
    event_at: str
    source_system: str
    source_id: str
    subject: str | None
    text_preview: str | None
    summary: str | None
    created_at: str


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(value: Any, length: int = 12) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:length]


def diagnostic_content_key(row: EmailRow) -> str | None:
    if row.customer_id:
        return timeline_email_content_key(
            tenant_id=row.tenant_id,
            customer_id=row.customer_id,
            event_type="email_message",
            event_at=row.event_at,
            subject=row.subject,
            summary=row.summary,
        )
    normalized_summary = normalize_email_content_text(row.summary)
    if not normalized_summary:
        return None
    event_dt = parse_datetime(row.event_at, "event_at")
    minute = event_dt.astimezone(timezone.utc).replace(second=0, microsecond=0).isoformat()
    return stable_prefixed_id(
        "email_content_none",
        {
            "tenant_id": row.tenant_id,
            "customer_id": "__none__",
            "event_at_minute": minute,
            "subject": normalize_email_content_text(row.subject),
            "summary_sha256": stable_digest({"summary": normalized_summary}),
        },
        length=40,
    )


def group_preview_key(row: EmailRow) -> str:
    return stable_digest({"text_preview": normalize_email_content_text(row.text_preview)})


def read_rows(db_path: Path) -> tuple[list[EmailRow], Mapping[str, Any], str]:
    uri = f"{db_path.resolve(strict=True).as_uri()}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA query_only = ON")
        quick_check = str(con.execute("PRAGMA quick_check").fetchone()[0])
        rows = [
            EmailRow(
                event_id=str(row["event_id"]),
                tenant_id=str(row["tenant_id"]),
                customer_id=str(row["customer_id"]) if row["customer_id"] else None,
                event_at=str(row["event_at"]),
                source_system=str(row["source_system"]),
                source_id=str(row["source_id"]),
                subject=row["subject"],
                text_preview=row["text_preview"],
                summary=row["summary"],
                created_at=str(row["created_at"]),
            )
            for row in con.execute(
                """
                SELECT event_id, tenant_id, customer_id, event_at, source_system, source_id,
                       subject, text_preview, summary, created_at
                FROM timeline_events
                WHERE event_type = 'email_message'
                """
            ).fetchall()
        ]
        meta = {
            "quick_check": quick_check,
            "timeline_event_count": int(con.execute("SELECT count(*) FROM timeline_events").fetchone()[0]),
            "email_event_count": len(rows),
            "email_direction_counts": {
                str(row["direction"]): int(row["total"])
                for row in con.execute(
                    """
                    SELECT direction, count(*) AS total
                    FROM timeline_events
                    WHERE event_type = 'email_message'
                    GROUP BY direction
                    ORDER BY direction
                    """
                ).fetchall()
            },
            "email_source_system_counts": {
                str(row["source_system"]): int(row["total"])
                for row in con.execute(
                    """
                    SELECT source_system, count(*) AS total
                    FROM timeline_events
                    WHERE event_type = 'email_message'
                    GROUP BY source_system
                    ORDER BY source_system
                    """
                ).fetchall()
            },
            "email_empty_subject": int(
                con.execute(
                    """
                    SELECT count(*)
                    FROM timeline_events
                    WHERE event_type = 'email_message'
                      AND (subject IS NULL OR trim(subject) = '')
                    """
                ).fetchone()[0]
            ),
            "email_empty_summary": int(
                con.execute(
                    """
                    SELECT count(*)
                    FROM timeline_events
                    WHERE event_type = 'email_message'
                      AND (summary IS NULL OR trim(summary) = '')
                    """
                ).fetchone()[0]
            ),
        }
        return rows, meta, uri
    finally:
        con.close()


def canonical_and_extras(rows: list[EmailRow]) -> tuple[EmailRow, list[EmailRow]]:
    ordered = sorted(rows, key=lambda item: (item.created_at, item.event_id))
    return ordered[0], ordered[1:]


def classify_groups(rows: Iterable[EmailRow]) -> Mapping[str, Any]:
    keyed: dict[tuple[str, str], list[EmailRow]] = defaultdict(list)
    mixed_preview_by_key: dict[str, set[str]] = defaultdict(set)
    skipped_no_key = 0
    for row in rows:
        key = diagnostic_content_key(row)
        if not key:
            skipped_no_key += 1
            continue
        preview_key = group_preview_key(row)
        keyed[(key, preview_key)].append(row)
        mixed_preview_by_key[key].add(preview_key)

    duplicate_groups = [items for items in keyed.values() if len(items) > 1]
    attributed_groups = [items for items in duplicate_groups if items[0].customer_id]
    none_groups = [items for items in duplicate_groups if not items[0].customer_id]
    mixed_preview_groups = sum(1 for previews in mixed_preview_by_key.values() if len(previews) > 1)

    return {
        "skipped_no_key": skipped_no_key,
        "attributed_groups": attributed_groups,
        "none_groups": none_groups,
        "mixed_preview_groups": mixed_preview_groups,
    }


def distribution(groups: Iterable[list[EmailRow]]) -> Mapping[int, int]:
    return dict(sorted(Counter(len(items) for items in groups).items()))


def group_summary(items: list[EmailRow]) -> Mapping[str, Any]:
    canonical, extras = canonical_and_extras(items)
    subject_norm = normalize_email_content_text(canonical.subject)
    summary_norm = normalize_email_content_text(canonical.summary)
    text_norm = normalize_email_content_text(canonical.text_preview)
    joined_probe = " ".join((subject_norm, summary_norm, text_norm))
    return {
        "size": len(items),
        "extra_rows": len(extras),
        "tenant_id": canonical.tenant_id,
        "customer_hash": short_hash(canonical.customer_id) if canonical.customer_id else None,
        "minute": parse_datetime(canonical.event_at, "event_at").replace(second=0, microsecond=0).isoformat(),
        "canonical_event_hash": short_hash(canonical.event_id),
        "subject_hash": short_hash(subject_norm),
        "summary_hash": short_hash(summary_norm),
        "text_preview_hash": short_hash(text_norm),
        "source_systems": dict(sorted(Counter(item.source_system for item in items).items())),
        "looks_like_web_form": any(marker in joined_probe for marker in ("web", "веб", "форма", "заявк")),
    }


def render_markdown(
    *,
    db_path: Path,
    uri: str,
    before_sha: str,
    after_sha: str,
    meta: Mapping[str, Any],
    classified: Mapping[str, Any],
) -> str:
    attributed = list(classified["attributed_groups"])
    none_groups = list(classified["none_groups"])
    attributed_rows = sum(len(items) for items in attributed)
    none_rows = sum(len(items) for items in none_groups)
    attributed_extra = sum(len(items) - 1 for items in attributed)
    none_extra = sum(len(items) - 1 for items in none_groups)
    web_form_none = [items for items in none_groups if group_summary(items)["looks_like_web_form"]]
    top_attributed = sorted(attributed, key=lambda items: (-len(items), items[0].created_at, items[0].event_id))[:7]
    top_none = sorted(none_groups, key=lambda items: (-len(items), items[0].created_at, items[0].event_id))[:7]
    generated_at = datetime.now(timezone.utc).isoformat()

    lines = [
        "# Content dedup email timeline dry-run",
        "",
        f"- generated_at: `{generated_at}`",
        f"- db_path: `{db_path}`",
        "- open_mode: `mode=ro&immutable=1`, `PRAGMA query_only=ON`",
        f"- sqlite_uri_masked: `{uri.replace(str(db_path), '<prod-db>')}`",
        f"- quick_check: `{meta['quick_check']}`",
        f"- prod_sha256_before: `{before_sha}`",
        f"- prod_sha256_after: `{after_sha}`",
        f"- prod_sha256_unchanged: `{before_sha == after_sha}`",
        "",
        "## Summary",
        "",
        f"- timeline_events_total: {meta['timeline_event_count']}",
        f"- email_events_total: {meta['email_event_count']}",
        f"- email_direction_counts: `{json.dumps(meta['email_direction_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- email_source_system_counts: `{json.dumps(meta['email_source_system_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- email_empty_subject: {meta['email_empty_subject']}",
        f"- email_empty_summary: {meta['email_empty_summary']}",
        f"- skipped_no_content_key_empty_summary: {classified['skipped_no_key']}",
        f"- attributed_duplicate_groups_candidate_for_future_soft_cleanup: {len(attributed)}",
        f"- attributed_duplicate_rows: {attributed_rows}",
        f"- attributed_extra_rows_candidate: {attributed_extra}",
        f"- none_customer_duplicate_groups_report_only: {len(none_groups)}",
        f"- none_customer_duplicate_rows_report_only: {none_rows}",
        f"- none_customer_extra_rows_report_only: {none_extra}",
        f"- none_customer_web_form_like_groups_report_only: {len(web_form_none)}",
        f"- mixed_text_preview_groups_not_counted_as_duplicates: {classified['mixed_preview_groups']}",
        "",
        "## Size distributions",
        "",
        f"- attributed_group_sizes: `{json.dumps(distribution(attributed), ensure_ascii=False, sort_keys=True)}`",
        f"- none_customer_group_sizes: `{json.dumps(distribution(none_groups), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Top attributed duplicate groups",
        "",
        "| # | size | extra | tenant | customer_hash | minute | canonical_event_hash | subject_hash | summary_hash | preview_hash | source_systems |",
        "|---:|---:|---:|---|---|---|---|---|---|---|---|",
    ]
    for index, items in enumerate(top_attributed, 1):
        item = group_summary(items)
        rendered = dict(item)
        rendered["source_systems"] = json.dumps(item["source_systems"], ensure_ascii=False, sort_keys=True)
        lines.append(
            "| {index} | {size} | {extra_rows} | `{tenant_id}` | `{customer_hash}` | `{minute}` | "
            "`{canonical_event_hash}` | `{subject_hash}` | `{summary_hash}` | `{text_preview_hash}` | `{source_systems}` |".format(
                index=index,
                **rendered,
            )
        )
    lines.extend(
        [
            "",
            "## Top None-customer duplicate groups: report only, not cleanup",
            "",
            "Эти группы нельзя схлопывать автоматически: без `customer_id` одинаковая web-форма в одну секунду может быть несколькими разными лидами.",
            "",
            "| # | size | extra | tenant | minute | canonical_event_hash | subject_hash | summary_hash | preview_hash | web_form_like | source_systems |",
            "|---:|---:|---:|---|---|---|---|---|---|---|---|",
        ]
    )
    for index, items in enumerate(top_none, 1):
        item = group_summary(items)
        rendered = dict(item)
        rendered["source_systems"] = json.dumps(item["source_systems"], ensure_ascii=False, sort_keys=True)
        lines.append(
            "| {index} | {size} | {extra_rows} | `{tenant_id}` | `{minute}` | `{canonical_event_hash}` | "
            "`{subject_hash}` | `{summary_hash}` | `{text_preview_hash}` | {looks_like_web_form} | `{source_systems}` |".format(
                index=index,
                **rendered,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Part 1 only reports duplicates; it does not mutate prod DB.",
            "- Future cleanup candidates are only attributed groups with identical `text_preview` inside the content key.",
            "- None-customer/web-form-like groups stay report-only until a separate human decision.",
            "- Part 3 prod cleanup was not run.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_local_raw(path: Path, classified: Mapping[str, Any], meta: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def serialize_groups(groups: list[list[EmailRow]]) -> list[Mapping[str, Any]]:
        result = []
        for items in groups:
            canonical, extras = canonical_and_extras(items)
            result.append(
                {
                    "summary": group_summary(items),
                    "canonical": canonical.__dict__,
                    "extras": [item.__dict__ for item in extras],
                }
            )
        return result

    payload = {
        "meta": dict(meta),
        "attributed_groups": serialize_groups(list(classified["attributed_groups"])),
        "none_customer_groups": serialize_groups(list(classified["none_groups"])),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_PROD_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--local-raw", type=Path, default=DEFAULT_LOCAL_RAW)
    args = parser.parse_args(argv)

    db_path = args.db.expanduser().resolve(strict=True)
    before_sha = file_sha256(db_path)
    rows, meta, uri = read_rows(db_path)
    classified = classify_groups(rows)
    after_sha = file_sha256(db_path)
    if before_sha != after_sha:
        raise RuntimeError("prod DB sha256 changed during read-only dry-run")

    report = render_markdown(
        db_path=db_path,
        uri=uri,
        before_sha=before_sha,
        after_sha=after_sha,
        meta=meta,
        classified=classified,
    )
    args.report.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.report.expanduser().write_text(report, encoding="utf-8")
    write_local_raw(args.local_raw.expanduser(), classified, meta)
    print(json.dumps(
        {
            "report": str(args.report.expanduser()),
            "local_raw": str(args.local_raw.expanduser()),
            "email_events_total": meta["email_event_count"],
            "attributed_duplicate_groups": len(classified["attributed_groups"]),
            "none_customer_duplicate_groups": len(classified["none_groups"]),
            "prod_sha256_unchanged": before_sha == after_sha,
        },
        ensure_ascii=False,
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
