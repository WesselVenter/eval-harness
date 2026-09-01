from eval.judge import JudgeResult


def test_compute_summary_averages_and_rates():
    from eval.metrics import compute_summary

    results = [
        JudgeResult(faithfulness=5, relevance=5, correctness=True, reasoning=""),
        JudgeResult(faithfulness=1, relevance=3, correctness=False, reasoning=""),
    ]

    summary = compute_summary(results)

    assert summary.mean_faithfulness == 3.0
    assert summary.mean_relevance == 4.0
    assert summary.correctness_rate == 0.5
    assert summary.hallucination_rate == 0.5
    assert summary.n_items == 2


def test_compute_summary_empty_results():
    from eval.metrics import compute_summary

    summary = compute_summary([])
    assert summary.n_items == 0
    assert summary.mean_faithfulness == 0.0


def test_check_thresholds_passes_when_within_bounds():
    from eval.metrics import check_thresholds, EvalSummary

    summary = EvalSummary(
        mean_faithfulness=4.5, mean_relevance=4.5, correctness_rate=0.9, hallucination_rate=0.0, n_items=5
    )
    assert check_thresholds(summary) == []


def test_check_thresholds_reports_violations():
    from eval.metrics import check_thresholds, EvalSummary

    summary = EvalSummary(
        mean_faithfulness=2.0, mean_relevance=4.0, correctness_rate=0.3, hallucination_rate=0.5, n_items=5
    )
    violations = check_thresholds(summary)
    assert len(violations) == 3
