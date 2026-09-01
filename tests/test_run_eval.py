import json

from app.models import Chunk
from eval.dataset import GoldenItem
from eval.judge import JudgeResult


def test_run_eval_orchestrates_retrieve_generate_judge(monkeypatch):
    import eval.run_eval as run_eval_module

    items = [
        GoldenItem(id="q1", question="Q1?", expected_answer="A1"),
        GoldenItem(id="q2", question="Q2?", expected_answer="A2"),
    ]
    chunk = Chunk(id="a::0", text="text", source="a.md", chunk_index=0)

    monkeypatch.setattr(run_eval_module, "load_golden_dataset", lambda path: items)
    monkeypatch.setattr(run_eval_module, "retrieve", lambda query, mode, k: [chunk])
    monkeypatch.setattr(run_eval_module, "generate_answer", lambda question, chunks: f"answer to {question}")
    monkeypatch.setattr(
        run_eval_module,
        "judge_answer",
        lambda question, context_chunks, generated_answer, expected_answer: JudgeResult(
            faithfulness=5, relevance=5, correctness=True, reasoning="ok"
        ),
    )

    summary, per_item = run_eval_module.run_eval()

    assert summary.n_items == 2
    assert summary.mean_faithfulness == 5.0
    assert len(per_item) == 2
    assert per_item[0]["id"] == "q1"
    assert per_item[0]["generated_answer"] == "answer to Q1?"
    assert per_item[0]["sources"] == ["a.md"]


def test_save_results_writes_json(tmp_path):
    from eval.metrics import EvalSummary
    from eval.run_eval import save_results

    summary = EvalSummary(
        mean_faithfulness=5.0, mean_relevance=5.0, correctness_rate=1.0, hallucination_rate=0.0, n_items=1
    )
    per_item = [{"id": "q1", "generated_answer": "A1"}]

    path = save_results(summary, per_item, timestamp="20260101T000000Z", results_dir=tmp_path)

    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["summary"]["n_items"] == 1
    assert data["items"][0]["id"] == "q1"
