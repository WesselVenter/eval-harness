from dataclasses import dataclass, field

from eval.judge import JudgeVerdict

HALLUCINATION_THRESHOLD = 0.5
DEFAULT_PASS_THRESHOLD = 0.7

MAX_HALLUCINATION_RATE = 0.2
MIN_PASS_RATE = 0.7
MIN_MEAN_CORRECTNESS = 0.6


@dataclass
class EvalSummary:
    n_items: int
    hallucination_rate: float
    mean_faithfulness: float
    mean_correctness: float
    mean_retrieval_quality: float
    correctness_by_category: dict[str, float] = field(default_factory=dict)
    pass_rate: float = 0.0
    failed_questions: list[str] = field(default_factory=list)


def is_hallucination(verdict: JudgeVerdict) -> bool:
    return verdict.faithfulness.score < HALLUCINATION_THRESHOLD


def passes_threshold(verdict: JudgeVerdict, threshold: float = DEFAULT_PASS_THRESHOLD) -> bool:
    return (
        verdict.faithfulness.score >= threshold
        and verdict.correctness.score >= threshold
        and verdict.retrieval_quality.score >= threshold
    )


def compute_summary(
    verdicts: list[JudgeVerdict],
    categories: list[str],
    threshold: float = DEFAULT_PASS_THRESHOLD,
) -> EvalSummary:
    n = len(verdicts)
    if n == 0:
        return EvalSummary(
            n_items=0,
            hallucination_rate=0.0,
            mean_faithfulness=0.0,
            mean_correctness=0.0,
            mean_retrieval_quality=0.0,
        )
    if len(categories) != n:
        raise ValueError("categories must be the same length as verdicts")

    hallucination_count = sum(1 for v in verdicts if is_hallucination(v))
    mean_faithfulness = sum(v.faithfulness.score for v in verdicts) / n
    mean_correctness = sum(v.correctness.score for v in verdicts) / n
    mean_retrieval_quality = sum(v.retrieval_quality.score for v in verdicts) / n

    scores_by_category: dict[str, list[float]] = {}
    for verdict, category in zip(verdicts, categories):
        scores_by_category.setdefault(category, []).append(verdict.correctness.score)
    correctness_by_category = {
        category: sum(scores) / len(scores) for category, scores in scores_by_category.items()
    }

    pass_flags = [passes_threshold(v, threshold) for v in verdicts]
    failed_questions = [v.question for v, passed in zip(verdicts, pass_flags) if not passed]

    return EvalSummary(
        n_items=n,
        hallucination_rate=hallucination_count / n,
        mean_faithfulness=mean_faithfulness,
        mean_correctness=mean_correctness,
        mean_retrieval_quality=mean_retrieval_quality,
        correctness_by_category=correctness_by_category,
        pass_rate=sum(pass_flags) / n,
        failed_questions=failed_questions,
    )


def check_thresholds(summary: EvalSummary) -> list[str]:
    violations = []
    if summary.hallucination_rate > MAX_HALLUCINATION_RATE:
        violations.append(
            f"hallucination_rate {summary.hallucination_rate:.2f} > {MAX_HALLUCINATION_RATE}"
        )
    if summary.pass_rate < MIN_PASS_RATE:
        violations.append(f"pass_rate {summary.pass_rate:.2f} < {MIN_PASS_RATE}")
    if summary.mean_correctness < MIN_MEAN_CORRECTNESS:
        violations.append(
            f"mean_correctness {summary.mean_correctness:.2f} < {MIN_MEAN_CORRECTNESS}"
        )
    return violations
