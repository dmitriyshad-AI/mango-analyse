# ADR-003 Frame Gold Calibration

- Acceptance: `needs_review`
- Labeled rows: `79`
- Compared rows: `77`
- Skipped rows: `2`
- Missing transcript rows: `0`
- Must-handoff accuracy: `0.9351`
- Too cautious: `5`
- Too confident: `0`
- Current over-handoff candidates: `21`
- Safe self candidates: `32`

## Per-field Accuracy

- `answerability`: `0.9114` (72/79)
- `must_handoff`: `0.9351` (72/77)
- `requested_action`: `0.8228` (65/79)
- `risk_class`: `0.8608` (68/79)

## Confidence Buckets

- `0.00-0.59`: rows=0, must_handoff_accuracy=None, too_cautious=0, too_confident=0
- `0.60-0.79`: rows=0, must_handoff_accuracy=None, too_cautious=0, too_confident=0
- `0.80-0.89`: rows=11, must_handoff_accuracy=0.6364, too_cautious=4, too_confident=0
- `0.90-1.00`: rows=66, must_handoff_accuracy=0.9848, too_cautious=1, too_confident=0
- `missing`: rows=0, must_handoff_accuracy=None, too_cautious=0, too_confident=0

## Blocking Notes

- Frame remains too cautious on safe/self rows; active autonomy needs calibration before Ф3.
- Some gold rows are unclear/not comparable.
