"""Metric primitives for the evaluation harness.

Detection suites report precision/recall/F1 rather than raw accuracy: with a guardrail,
missing an attack and over-blocking a customer are different failures and a single
accuracy number hides which one is happening.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryMetrics:
    """Confusion-matrix summary for a binary detector."""

    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int

    @property
    def total(self) -> int:
        return self.true_positive + self.false_positive + self.false_negative + self.true_negative

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        return (self.true_positive + self.true_negative) / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "true_negative": self.true_negative,
        }


def binary_metrics(outcomes: list[tuple[bool, bool]]) -> BinaryMetrics:
    """Build metrics from (expected, predicted) pairs."""
    tp = sum(1 for e, p in outcomes if e and p)
    fp = sum(1 for e, p in outcomes if not e and p)
    fn = sum(1 for e, p in outcomes if e and not p)
    tn = sum(1 for e, p in outcomes if not e and not p)
    return BinaryMetrics(tp, fp, fn, tn)


def recall_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    """Fraction of the expected documents that appear in the top-k results."""
    if not expected:
        return 0.0
    top = set(retrieved[:k])
    return sum(1 for doc in expected if doc in top) / len(expected)


def reciprocal_rank(expected: list[str], retrieved: list[str]) -> float:
    """1/rank of the first expected document, or 0 if none was retrieved."""
    for position, doc in enumerate(retrieved, start=1):
        if doc in expected:
            return 1.0 / position
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
