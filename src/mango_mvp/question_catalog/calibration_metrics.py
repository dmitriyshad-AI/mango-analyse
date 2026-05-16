from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ThemeScore:
    label: str
    support: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ClassificationMetrics:
    total: int
    correct: int
    accuracy: float
    macro_f1: float
    label_count: int
    per_label: tuple[ThemeScore, ...]

    def worst_recall(self, limit: int = 10) -> tuple[ThemeScore, ...]:
        return tuple(sorted(self.per_label, key=lambda item: (item.recall, -item.support, item.label))[:limit])


def compute_classification_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    true_field: str = "human_label",
    pred_field: str = "predicted_theme_id",
) -> ClassificationMetrics:
    pairs: list[tuple[str, str]] = []
    for row in rows:
        true_label = str(row.get(true_field) or "").strip()
        pred_label = str(row.get(pred_field) or "").strip()
        if true_label and pred_label:
            pairs.append((true_label, pred_label))

    labels = sorted({true for true, _pred in pairs})
    scores: list[ThemeScore] = []
    for label in labels:
        tp = sum(1 for true, pred in pairs if true == label and pred == label)
        fp = sum(1 for true, pred in pairs if true != label and pred == label)
        fn = sum(1 for true, pred in pairs if true == label and pred != label)
        support = sum(1 for true, _pred in pairs if true == label)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        scores.append(
            ThemeScore(
                label=label,
                support=support,
                true_positive=tp,
                false_positive=fp,
                false_negative=fn,
                precision=precision,
                recall=recall,
                f1=f1,
            )
        )

    total = len(pairs)
    correct = sum(1 for true, pred in pairs if true == pred)
    macro_f1 = _safe_div(sum(item.f1 for item in scores), len(scores))
    return ClassificationMetrics(
        total=total,
        correct=correct,
        accuracy=_safe_div(correct, total),
        macro_f1=macro_f1,
        label_count=len(scores),
        per_label=tuple(scores),
    )


def validate_labeled_rows(rows: Iterable[Mapping[str, Any]], valid_labels: set[str]) -> list[str]:
    errors: list[str] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows, start=2):
        question_id = str(row.get("question_id") or "").strip()
        if not question_id:
            errors.append(f"row {index}: missing question_id")
        elif question_id in seen_ids:
            errors.append(f"row {index}: duplicate question_id {question_id}")
        seen_ids.add(question_id)
        label = str(row.get("human_label") or "").strip()
        if not label:
            errors.append(f"row {index}: missing human_label")
        elif label not in valid_labels:
            errors.append(f"row {index}: invalid human_label {label}")
    return errors


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
