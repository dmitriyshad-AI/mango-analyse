# ADR-003 Frame Gold Calibration

- Acceptance: `needs_review`
- Labeled rows: `75`
- Compared rows: `73`
- Skipped rows: `2`
- Missing transcript rows: `0`
- Must-handoff accuracy: `0.6027`
- Too cautious: `29`
- Too confident: `0`
- Current over-handoff candidates: `21`
- Safe self candidates: `32`

## Per-field Accuracy

- `answerability`: `0.0133` (1/75)
- `must_handoff`: `0.6027` (44/73)
- `requested_action`: `0.8133` (61/75)
- `risk_class`: `0.4933` (37/75)

## Confidence Buckets

- `0.00-0.59`: rows=0, must_handoff_accuracy=None, too_cautious=0, too_confident=0
- `0.60-0.79`: rows=0, must_handoff_accuracy=None, too_cautious=0, too_confident=0
- `0.80-0.89`: rows=40, must_handoff_accuracy=0.475, too_cautious=21, too_confident=0
- `0.90-1.00`: rows=33, must_handoff_accuracy=0.7576, too_cautious=8, too_confident=0
- `missing`: rows=0, must_handoff_accuracy=None, too_cautious=0, too_confident=0

## Blocking Notes

- Frame remains too cautious on safe/self rows; active autonomy needs calibration before Ф3.
- Some gold rows are unclear/not comparable.
