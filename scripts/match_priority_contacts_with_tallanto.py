from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def _clean(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_dt(value: Any) -> pd.Timestamp | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return pd.to_datetime(text, dayfirst=True, errors="coerce")
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact-phone Tallanto match tables for priority AI contacts.")
    parser.add_argument("--priority-csv", required=True)
    parser.add_argument("--tallanto-csv", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    priority_csv = Path(args.priority_csv).expanduser().resolve()
    tallanto_csv = Path(args.tallanto_csv).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    priority = pd.read_csv(priority_csv, dtype=str)
    tallanto = pd.read_csv(tallanto_csv, dtype=str)

    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    status_counter: Counter[str] = Counter()

    for _, pr in priority.iterrows():
        phone = _clean(pr.get("phone"))
        matches = tallanto[(tallanto["phone_parent"] == phone) | (tallanto["phone_extra"] == phone)].copy()
        if matches.empty:
            status = "no_exact_phone_match"
            summary_rows.append(
                {
                    **pr.to_dict(),
                    "match_status": status,
                    "match_candidates_count": 0,
                    "matched_tallanto_id": None,
                    "matched_parent_fio": None,
                    "matched_responsible": None,
                    "matched_student_type": None,
                    "matched_branch": None,
                    "matched_history_raw": None,
                    "matched_updated_at": None,
                }
            )
            status_counter[status] += 1
            continue

        matches["updated_at_ts"] = matches["updated_at"].map(_parse_dt)
        matches = matches.sort_values(
            by=["updated_at_ts", "history_raw", "email"],
            ascending=[False, False, False],
            na_position="last",
        )
        status = "exact_phone_single" if len(matches) == 1 else "exact_phone_multiple"
        status_counter[status] += 1

        best = matches.iloc[0]
        summary_rows.append(
            {
                **pr.to_dict(),
                "match_status": status,
                "match_candidates_count": int(len(matches)),
                "matched_tallanto_id": _clean(best.get("tallanto_id")),
                "matched_parent_fio": _clean(best.get("parent_fio")),
                "matched_responsible": _clean(best.get("responsible")),
                "matched_student_type": _clean(best.get("student_type")),
                "matched_branch": _clean(best.get("branch")),
                "matched_history_raw": _clean(best.get("history_raw")),
                "matched_updated_at": _clean(best.get("updated_at")),
            }
        )

        for rank, (_, cand) in enumerate(matches.iterrows(), 1):
            candidate_rows.append(
                {
                    "phone": phone,
                    "priority_rank": pr.get("priority_rank"),
                    "match_status": status,
                    "candidate_rank": rank,
                    "tallanto_id": _clean(cand.get("tallanto_id")),
                    "parent_fio": _clean(cand.get("parent_fio")),
                    "contact_full_name": _clean(cand.get("contact_full_name")),
                    "phone_parent": _clean(cand.get("phone_parent")),
                    "phone_extra": _clean(cand.get("phone_extra")),
                    "email": _clean(cand.get("email")),
                    "alt_email": _clean(cand.get("alt_email")),
                    "responsible": _clean(cand.get("responsible")),
                    "student_type": _clean(cand.get("student_type")),
                    "interests_raw": _clean(cand.get("interests_raw")),
                    "history_raw": _clean(cand.get("history_raw")),
                    "branch": _clean(cand.get("branch")),
                    "created_at": _clean(cand.get("created_at")),
                    "updated_at": _clean(cand.get("updated_at")),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    candidates_df = pd.DataFrame(candidate_rows)

    summary_path = out_root / "top100_priority_tallanto_match_summary.csv"
    candidates_path = out_root / "top100_priority_tallanto_match_candidates.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    candidates_df.to_csv(candidates_path, index=False, encoding="utf-8")

    manifest = {
        "priority_csv": str(priority_csv),
        "tallanto_csv": str(tallanto_csv),
        "summary_csv": str(summary_path),
        "candidates_csv": str(candidates_path),
        "priority_contacts": int(len(priority_df := priority)),
        "match_status_counts": dict(status_counter),
        "matched_contacts": int(sum(1 for row in summary_rows if row["match_candidates_count"] > 0)),
        "unmatched_contacts": int(sum(1 for row in summary_rows if row["match_candidates_count"] == 0)),
    }
    (out_root / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
