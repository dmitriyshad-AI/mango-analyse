from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from mango_mvp.deal_aware.deal_attribution import confidence_bucket


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deal_aware_confidence_phase2_linked_scores.csv"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_confidence_thresholds_fit_real_phase2_linked_distribution_fixture() -> None:
    rows = _read_csv(FIXTURE)

    assert len(rows) >= 4326

    buckets = Counter()
    for row in rows:
        score = float(row["confidence_score"])
        bucket = confidence_bucket(score)
        assert bucket == row["expected_bucket"]
        buckets[bucket] += 1

    total = sum(buckets.values())
    shares = {bucket: buckets[bucket] / total for bucket in ("high", "medium", "low")}

    assert 0.25 <= shares["high"] <= 0.40
    assert 0.40 <= shares["medium"] <= 0.60
    assert 0.10 <= shares["low"] <= 0.25
