# ADR-003 Frame Gold Calibration

- Acceptance: `needs_review`
- Labeled rows: `79`
- Compared rows: `77`
- Skipped rows: `2`
- Missing transcript rows: `0`
- Must-handoff accuracy: `0.8701`
- Too cautious: `10`
- Too confident: `0`
- Current over-handoff candidates: `11`
- Safe self candidates: `32`

## Per-field Accuracy

- `answerability`: `0.8481` (67/79)
- `must_handoff`: `0.8701` (67/77)
- `requested_action`: `0.7595` (60/79)
- `risk_class`: `0.7848` (62/79)

## Confidence Buckets

- `0.00-0.59`: rows=0, must_handoff_accuracy=None, too_cautious=0, too_confident=0
- `0.60-0.79`: rows=0, must_handoff_accuracy=None, too_cautious=0, too_confident=0
- `0.80-0.89`: rows=19, must_handoff_accuracy=0.5789, too_cautious=8, too_confident=0
- `0.90-1.00`: rows=58, must_handoff_accuracy=0.9655, too_cautious=2, too_confident=0
- `missing`: rows=0, must_handoff_accuracy=None, too_cautious=0, too_confident=0

## Blocking Notes

- Frame remains too cautious on safe/self rows; active autonomy needs calibration before Ф3.
- Some gold rows are unclear/not comparable.
