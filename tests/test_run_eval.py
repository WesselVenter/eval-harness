import json

from app.models import Chunk
from eval.dataset import GoldenItem
from eval.judge import JudgeVerdict, ScoredAspect


def _verdict(question):
    return JudgeVerdict(
        question=question,
        faithfulness=ScoredAspect(score=1.0, justification="ok"),
        correctness=ScoredAspect(score=1.0, justification="ok"),
        retrieval_quality=ScoredAspect(score=1.0, justification="ok"),
    )


def test_run_eval_orchestrates_retrieve_generate_judge(monkeypatch):
    import eval.run_eval as run_eval_module

    items = [
        GoldenItem(question="Q1?", expected_answer="A1", expected_sources=["a.md"], category="factual"),
        GoldenItem(question="Q2?", expected_answer="A2", expected_sources=["a.md"], category="factual"),
    ]
    chunk = Chunk(id="a::0", text="text", source="a.md", chunk_index=0)

    monkeypatch.setattr(run_eval_module, "load_golden_dataset", lambda path: items)
    monkeypatch.setattr(run_eval_module, "retrieve", lambda query, mode, k: [chunk])
    monkeypatch.setattr(run_eval_module, "generate_answer", lambda question, chunks: f"answer to {question}")
    monkeypatch.setattr(
        run_eval_module,
        "judge_answer",
        lambda question, context_chunks, generated_answer, expected_answer, expected_sources, category: _verdict(
            question
        ),
    )

    summary, per_item = run_eval_module.run_eval()

    assert summary.n_items == 2
    assert summary.mean_faithfulness == 1.0
    assert len(per_item) == 2
    assert per_item[0]["id"] == "q1"
    assert per_item[0]["generated_answer"] == "answer to Q1?"
    assert per_item[0]["sources"] == ["a.md"]
    assert per_item[0]["category"] == "factual"
    assert per_item[0]["faithfulness"]["score"] == 1.0
    assert per_item[0]["passed"] is True


def test_save_results_writes_json(tmp_path):
    from eval.metrics import EvalSummary
    from eval.run_eval import save_results

    summary = EvalSummary(
        n_items=1,
        hallucination_rate=0.0,
        mean_faithfulness=1.0,
        mean_correctness=1.0,
        mean_retrieval_quality=1.0,
        pass_rate=1.0,
    )
    per_item = [{"id": "q1", "generated_answer": "A1"}]

    path = save_results(summary, per_item, timestamp="20260101T000000Z", results_dir=tmp_path)

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["n_items"] == 1
    assert data["items"][0]["id"] == "q1"


def _score(value):
    return {"score": value, "justification": "x"}


def test_find_low_scoring_flags_items_below_threshold():
    from eval.run_eval import find_low_scoring

    per_item = [
        {
            "id": "q1",
            "question": "Q1?",
            "faithfulness": _score(1.0),
            "correctness": _score(0.4),
            "retrieval_quality": _score(1.0),
        },
        {
            "id": "q2",
            "question": "Q2?",
            "faithfulness": _score(1.0),
            "correctness": _score(1.0),
            "retrieval_quality": _score(1.0),
        },
        {
            "id": "q3",
            "question": "Q3?",
            "faithfulness": _score(0.2),
            "correctness": _score(0.1),
            "retrieval_quality": _score(1.0),
        },
    ]

    flagged = find_low_scoring(per_item)

    assert [f["id"] for f in flagged] == ["q1", "q3"]
    assert flagged[0]["low_metrics"] == ["correctness"]
    assert flagged[1]["low_metrics"] == ["faithfulness", "correctness"]


def test_print_summary_lists_low_scoring_questions(capsys):
    from eval.metrics import EvalSummary
    from eval.run_eval import print_summary

    summary = EvalSummary(
        n_items=1,
        hallucination_rate=0.0,
        mean_faithfulness=1.0,
        mean_correctness=0.3,
        mean_retrieval_quality=1.0,
        pass_rate=0.0,
    )
    per_item = [
        {
            "id": "q1",
            "question": "Why is the sky blue?",
            "faithfulness": _score(1.0),
            "correctness": _score(0.3),
            "retrieval_quality": _score(1.0),
        }
    ]

    print_summary(summary, per_item)

    out = capsys.readouterr().out
    assert "Questions scoring below 0.5" in out
    assert "Why is the sky blue?" in out
    assert "correctness" in out
