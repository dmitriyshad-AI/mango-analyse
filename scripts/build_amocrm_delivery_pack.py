from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _clean(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _amo_write_status(match_status: str | None) -> str:
    if match_status == "exact_phone_single":
        return "ready_exact_phone"
    if match_status == "exact_phone_multiple":
        return "review_multiple_matches"
    return "review_no_match"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build merged AI+Tallanto review pack and amo-ready CSV.")
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--match-summary-csv", required=True)
    parser.add_argument("--match-candidates-csv", required=True)
    parser.add_argument("--tallanto-csv", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()

    workbook = Path(args.workbook).expanduser().resolve()
    match_summary_csv = Path(args.match_summary_csv).expanduser().resolve()
    match_candidates_csv = Path(args.match_candidates_csv).expanduser().resolve()
    tallanto_csv = Path(args.tallanto_csv).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    contacts = pd.read_excel(workbook, sheet_name="Contacts", engine="openpyxl", dtype=str)
    calls = pd.read_excel(workbook, sheet_name="Calls", engine="openpyxl", dtype=str)
    match_summary = pd.read_csv(match_summary_csv, dtype=str)
    match_candidates = pd.read_csv(match_candidates_csv, dtype=str)
    tallanto = pd.read_csv(tallanto_csv, dtype=str)

    phones = set(contacts["phone"].dropna().astype(str).str.strip())
    match_summary = match_summary[match_summary["phone"].isin(phones)].copy()
    match_candidates = match_candidates[match_candidates["phone"].isin(phones)].copy()

    tallanto_best = tallanto[[
        "tallanto_id",
        "contact_full_name",
        "parent_fio",
        "phone_parent",
        "phone_extra",
        "email",
        "alt_email",
        "responsible",
        "student_type",
        "interests_raw",
        "history_raw",
        "branch",
        "created_at",
        "updated_at",
    ]].copy()
    tallanto_best = tallanto_best.rename(
        columns={
            "contact_full_name": "tallanto_contact_full_name",
            "parent_fio": "tallanto_parent_fio",
            "phone_parent": "tallanto_phone_parent",
            "phone_extra": "tallanto_phone_extra",
            "email": "tallanto_email",
            "alt_email": "tallanto_alt_email",
            "responsible": "tallanto_responsible_raw",
            "student_type": "tallanto_student_type_raw",
            "interests_raw": "tallanto_interests_raw",
            "history_raw": "tallanto_history_raw_raw",
            "branch": "tallanto_branch_raw",
            "created_at": "tallanto_created_at",
            "updated_at": "tallanto_updated_at_raw",
        }
    )

    merged = contacts.merge(
        match_summary,
        on="phone",
        how="left",
        suffixes=("_ai", "_priority"),
    )
    merged = merged.merge(
        tallanto_best,
        left_on="matched_tallanto_id",
        right_on="tallanto_id",
        how="left",
    )
    merged["amo_write_status"] = merged["match_status"].map(_amo_write_status)
    merged["amo_write_reason"] = merged["match_status"].map(
        {
            "exact_phone_single": "exact phone match, safe for first-pass write",
            "exact_phone_multiple": "multiple Tallanto candidates for same phone, review before write",
            "no_exact_phone_match": "no exact Tallanto phone match, review before write",
        }
    )
    merged["source_system"] = "mango_ai_recent_top20"

    amo_input = pd.DataFrame(
        {
            "amo_write_status": merged["amo_write_status"],
            "amo_write_reason": merged["amo_write_reason"],
            "phone": merged["phone"],
            "contact_key": merged["contact_key"],
            "ai_parent_fio": merged["parent_fio"],
            "ai_child_fio": merged["child_fio"],
            "ai_email": merged["email"],
            "ai_latest_manager_name": merged["latest_manager_name_ai"],
            "ai_latest_call_type": merged["latest_call_type"],
            "ai_latest_history_summary": merged["latest_history_summary"],
            "ai_interests_products": merged["interests_products"],
            "ai_recommended_product": merged["recommended_product_ai"],
            "ai_lead_priority": merged["lead_priority_ai"],
            "ai_sale_probability_pct": merged["sale_probability_pct"],
            "ai_recommended_followup_date": merged["recommended_followup_date"],
            "ai_last_next_step_action": merged["last_next_step_action"],
            "ai_objections_latest": merged["objections_latest"],
            "ai_source_call_ids": merged["source_call_ids"],
            "tallanto_id": merged["matched_tallanto_id"],
            "tallanto_match_status": merged["match_status"],
            "tallanto_match_candidates_count": merged["match_candidates_count"],
            "tallanto_parent_fio": merged["matched_parent_fio"].combine_first(merged["tallanto_parent_fio"]),
            "tallanto_contact_full_name": merged["tallanto_contact_full_name"],
            "tallanto_phone_parent": merged["tallanto_phone_parent"],
            "tallanto_phone_extra": merged["tallanto_phone_extra"],
            "tallanto_email": merged["tallanto_email"],
            "tallanto_alt_email": merged["tallanto_alt_email"],
            "tallanto_responsible": merged["matched_responsible"].combine_first(merged["tallanto_responsible_raw"]),
            "tallanto_student_type": merged["matched_student_type"].combine_first(merged["tallanto_student_type_raw"]),
            "tallanto_branch": merged["matched_branch"].combine_first(merged["tallanto_branch_raw"]),
            "tallanto_history_raw": merged["matched_history_raw"].combine_first(merged["tallanto_history_raw_raw"]),
            "tallanto_updated_at": merged["matched_updated_at"].combine_first(merged["tallanto_updated_at_raw"]),
        }
    )

    merged_summary_csv = out_root / "top20_ai_tallanto_merged_contacts.csv"
    candidates_csv = out_root / "top20_tallanto_match_candidates.csv"
    amo_input_csv = out_root / "top20_amocrm_input.csv"
    delivery_xlsx = out_root / "top20_delivery_pack.xlsx"

    merged.to_csv(merged_summary_csv, index=False, encoding="utf-8")
    match_candidates.to_csv(candidates_csv, index=False, encoding="utf-8")
    amo_input.to_csv(amo_input_csv, index=False, encoding="utf-8")

    with pd.ExcelWriter(delivery_xlsx, engine="openpyxl") as writer:
        contacts.to_excel(writer, sheet_name="AI_Contacts", index=False)
        calls.to_excel(writer, sheet_name="AI_Calls", index=False)
        merged.to_excel(writer, sheet_name="AI_Tallanto_Merged", index=False)
        match_candidates.to_excel(writer, sheet_name="Tallanto_Candidates", index=False)
        amo_input.to_excel(writer, sheet_name="AMO_Input", index=False)

    summary = {
        "workbook": str(workbook),
        "contacts_rows": int(len(contacts)),
        "calls_rows": int(len(calls)),
        "merged_contacts_csv": str(merged_summary_csv),
        "candidates_csv": str(candidates_csv),
        "amo_input_csv": str(amo_input_csv),
        "delivery_pack_xlsx": str(delivery_xlsx),
        "amo_write_status_counts": amo_input["amo_write_status"].fillna("unknown").value_counts().to_dict(),
        "match_status_counts": merged["match_status"].fillna("missing").value_counts().to_dict(),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
