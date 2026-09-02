from eval.compare_runs import compare_runs


def _item(question, faithfulness, correctness, retrieval_quality, item_id="q1"):
    return {
        "id": item_id,
        "question": question,
        "faithfulness": {"score": faithfulness, "justification": "x"},
        "correctness": {"score": correctness, "justification": "x"},
        "retrieval_quality": {"score": retrieval_quality, "justification": "x"},
    }


def _run(items, hallucination_rate):
    return {"summary": {"hallucination_rate": hallucination_rate}, "items": items}


def test_compare_runs_classifies_improved_and_regressed():
    run_a = _run(
        [
            _item("Q1?", 1.0, 0.5, 1.0, item_id="q1"),
            _item("Q2?", 1.0, 1.0, 1.0, item_id="q2"),
            _item("Q3?", 1.0, 0.8, 1.0, item_id="q3"),
        ],
        hallucination_rate=0.1,
    )
    run_b = _run(
        [
            _item("Q1?", 1.0, 1.0, 1.0, item_id="q1"),
            _item("Q2?", 1.0, 1.0, 1.0, item_id="q2"),
            _item("Q3?", 1.0, 0.2, 1.0, item_id="q3"),
        ],
        hallucination_rate=0.0,
    )

    diff = compare_runs(run_a, run_b)

    assert [e["question"] for e in diff["improved"]] == ["Q1?"]
    assert [e["question"] for e in diff["regressed"]] == ["Q3?"]
    assert [e["question"] for e in diff["unchanged"]] == ["Q2?"]
    assert diff["hallucination_rate_delta"] == -0.1


def test_compare_runs_flags_questions_only_in_one_run():
    run_a = _run([_item("Q1?", 1.0, 1.0, 1.0)], hallucination_rate=0.0)
    run_b = _run([_item("Q2?", 1.0, 1.0, 1.0)], hallucination_rate=0.0)

    diff = compare_runs(run_a, run_b)

    assert diff["only_in_a"] == ["Q1?"]
    assert diff["only_in_b"] == ["Q2?"]
    assert diff["improved"] == []
    assert diff["regressed"] == []
