from dataclasses import dataclass

from eval.judge import JudgeResult

MIN_FAITHFULNESS = 4.0
MIN_CORRECTNESS_RATE = 0.7
MAX_HALLUCINATION_RATE = 0.2
HALLUCINATION_THRESHOLD = 2


@dataclass
class EvalSummary:
    mean_faithfulness: float
    mean_relevance: float
    correctness_rate: float
    hallucination_rate: float
    n_items: int


def compute_summary(results: list[JudgeResult]) -> EvalSummary:
    n = len(results)
    if n == 0:
        return EvalSummary(0.0, 0.0, 0.0, 0.0, 0)
    mean_faithfulness = sum(r.faithfulness for r in results) / n
    mean_relevance = sum(r.relevance for r in results) / n
    correctness_rate = sum(1 for r in results if r.correctness) / n
    hallucination_rate = sum(1 for r in results if r.faithfulness <= HALLUCINATION_THRESHOLD) / n
    return EvalSummary(mean_faithfulness, mean_relevance, correctness_rate, hallucination_rate, n)


def check_thresholds(summary: EvalSummary) -> list[str]:
    violations = []
    if summary.mean_faithfulness < MIN_FAITHFULNESS:
        violations.append(f"mean_faithfulness {summary.mean_faithfulness:.2f} < {MIN_FAITHFULNESS}")
    if summary.correctness_rate < MIN_CORRECTNESS_RATE:
        violations.append(f"correctness_rate {summary.correctness_rate:.2f} < {MIN_CORRECTNESS_RATE}")
    if summary.hallucination_rate > MAX_HALLUCINATION_RATE:
        violations.append(f"hallucination_rate {summary.hallucination_rate:.2f} > {MAX_HALLUCINATION_RATE}")
    return violations
