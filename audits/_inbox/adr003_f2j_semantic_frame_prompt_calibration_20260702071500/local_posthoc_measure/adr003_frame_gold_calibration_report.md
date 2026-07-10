# ADR-003 Frame Gold Calibration

- Acceptance: `needs_review`
- Labeled rows: `13`
- Compared rows: `12`
- Skipped rows: `1`
- Missing transcript rows: `0`
- Must-handoff accuracy: `0.3333`
- Too cautious: `8`
- Too confident: `0`
- Current over-handoff candidates: `1`
- Safe self candidates: `8`

## Per-field Accuracy

- `answerability`: `0.3077` (4/13)
- `must_handoff`: `0.3333` (4/12)
- `requested_action`: `0.9231` (12/13)
- `risk_class`: `0.3077` (4/13)

## Confidence Buckets

- `0.00-0.59`: rows=0, must_handoff_accuracy=None, too_cautious=0, too_confident=0
- `0.60-0.79`: rows=0, must_handoff_accuracy=None, too_cautious=0, too_confident=0
- `0.80-0.89`: rows=8, must_handoff_accuracy=0.0, too_cautious=8, too_confident=0
- `0.90-1.00`: rows=4, must_handoff_accuracy=1.0, too_cautious=0, too_confident=0
- `missing`: rows=0, must_handoff_accuracy=None, too_cautious=0, too_confident=0

## Blocking Notes

- Frame remains too cautious on safe/self rows; active autonomy needs calibration before Ф3.
- Some gold rows are unclear/not comparable.
