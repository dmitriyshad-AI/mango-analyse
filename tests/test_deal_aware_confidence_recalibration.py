from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from mango_mvp.deal_aware.deal_attribution import confidence_bucket, single_candidate_confidence


PHASE1_STAGE2_ROOT = Path("stable_runtime/deal_aware_stage2_attribution_20260514_selector_fix_phase1")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_confidence_thresholds_fit_real_phase1_linked_distribution() -> None:
    links_path = PHASE1_STAGE2_ROOT / "deal_call_links.csv"
    candidates_path = PHASE1_STAGE2_ROOT / "phone_deal_candidates.csv"
    if not links_path.exists() or not candidates_path.exists():
        pytest.skip("Phase1 real-data Stage2 artifact is not available in this checkout.")

    candidates = {
        (row.get("phone", ""), row.get("deal_id", "")): row
        for row in _read_csv(candidates_path)
        if row.get("phone") and row.get("deal_id")
    }
    linked = [
        row
        for row in _read_csv(links_path)
        if row.get("attribution_decision") == "linked_single_deal_candidate" and row.get("selected_deal_id")
    ]

    assert len(linked) >= 4326

    buckets = Counter()
    missing = 0
    for row in linked:
        candidate = candidates.get((row.get("phone", ""), row.get("selected_deal_id", "")))
        if not candidate:
            missing += 1
            continue
        buckets[confidence_bucket(single_candidate_confidence(candidate))] += 1

    assert missing == 0
    total = sum(buckets.values())
    shares = {bucket: buckets[bucket] / total for bucket in ("high", "medium", "low")}

    assert 0.25 <= shares["high"] <= 0.40
    assert 0.40 <= shares["medium"] <= 0.60
    # The real-data selector-fix artifact sits near the upper edge; keep the
    # guard broad enough to catch drift without making the test depend on a
    # sub-percent local runtime fluctuation.
    assert 0.10 <= shares["low"] <= 0.27
