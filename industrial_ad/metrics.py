from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class ClassificationMetrics:
    total: int
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.true_positive + self.true_negative) / self.total

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        if denominator == 0:
            return 0.0
        return self.true_positive / denominator

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        if denominator == 0:
            return 0.0
        return self.true_positive / denominator

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        if denominator == 0:
            return 0.0
        return 2 * self.precision * self.recall / denominator

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "total": self.total,
            "true_positive": self.true_positive,
            "true_negative": self.true_negative,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def evaluate_binary_predictions(
    expected: Sequence[bool],
    predicted: Sequence[bool],
) -> ClassificationMetrics:
    if len(expected) != len(predicted):
        raise ValueError("expected and predicted must have the same length")
    tp = tn = fp = fn = 0
    for exp, pred in zip(expected, predicted):
        if exp and pred:
            tp += 1
        elif not exp and not pred:
            tn += 1
        elif not exp and pred:
            fp += 1
        else:
            fn += 1
    return ClassificationMetrics(
        total=len(expected),
        true_positive=tp,
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
    )


def threshold_scores(scores: Iterable[float], threshold: float) -> List[bool]:
    return [score >= threshold for score in scores]
