from eval.judge import JudgeVerdict, ScoredAspect


def _verdict(question, faithfulness, correctness, retrieval_quality):
    return JudgeVerdict(
        question=question,
        faithfulness=ScoredAspect(score=faithfulness, justification=""),
        correctness=ScoredAspect(score=correctness, justification=""),
        retrieval_quality=ScoredAspect(score=retrieval_quality, justification=""),
    )


def test_compute_summary_averages_and_rates():
    from eval.metrics import compute_summary

    verdicts = [
        _verdict("Q1", faithfulness=1.0, correctness=1.0, retrieval_quality=1.0),
        _verdict("Q2", faithfulness=0.0, correctness=0.0, retrieval_quality=0.0),
    ]

    summary = compute_summary(verdicts, categories=["factual", "factual"])

    assert summary.n_items == 2
    assert summary.mean_faithfulness == 0.5
    assert summary.mean_correctness == 0.5
    assert summary.mean_retrieval_quality == 0.5
    assert summary.hallucination_rate == 0.5
    assert summary.pass_rate == 0.5
    assert summary.failed_questions == ["Q2"]


def test_compute_summary_correctness_by_category():
    from eval.metrics import compute_summary

    verdicts = [
        _verdict("Q1", faithfulness=1.0, correctness=1.0, retrieval_quality=1.0),
        _verdict("Q2", faithfulness=1.0, correctness=0.5, retrieval_quality=1.0),
        _verdict("Q3", faithfulness=1.0, correctness=0.0, retrieval_quality=1.0),
    ]

    summary = compute_summary(verdicts, categories=["factual", "factual", "edge_case"])

    assert summary.correctness_by_category == {"factual": 0.75, "edge_case": 0.0}


def test_compute_summary_empty_results():
    from eval.metrics import compute_summary

    summary = compute_summary([], categories=[])

    assert summary.n_items == 0
    assert summary.mean_faithfulness == 0.0
    assert summary.correctness_by_category == {}


def test_is_hallucination_below_threshold():
    from eval.metrics import is_hallucination

    verdict = _verdict("Q", faithfulness=0.4, correctness=1.0, retrieval_quality=1.0)
    assert is_hallucination(verdict) is True

    verdict = _verdict("Q", faithfulness=0.5, correctness=1.0, retrieval_quality=1.0)
    assert is_hallucination(verdict) is False


def test_passes_threshold_requires_all_dimensions_above_threshold():
    from eval.metrics import passes_threshold

    verdict = _verdict("Q", faithfulness=0.8, correctness=0.6, retrieval_quality=0.8)
    assert passes_threshold(verdict, threshold=0.7) is False
    assert passes_threshold(verdict, threshold=0.5) is True


def test_check_thresholds_passes_when_within_bounds():
    from eval.metrics import EvalSummary, check_thresholds

    summary = EvalSummary(
        n_items=5,
        hallucination_rate=0.0,
        mean_faithfulness=0.9,
        mean_correctness=0.9,
        mean_retrieval_quality=0.9,
        pass_rate=0.9,
    )
    assert check_thresholds(summary) == []


def test_check_thresholds_reports_violations():
    from eval.metrics import EvalSummary, check_thresholds

    summary = EvalSummary(
        n_items=5,
        hallucination_rate=0.5,
        mean_faithfulness=0.3,
        mean_correctness=0.3,
        mean_retrieval_quality=0.3,
        pass_rate=0.2,
    )
    violations = check_thresholds(summary)
    assert len(violations) == 3
