import json
import time
from dataclasses import asdict
from pathlib import Path

from app.generate import generate_answer
from app.retrieve import retrieve
from eval.dataset import DEFAULT_DATASET_PATH, load_golden_dataset
from eval.judge import JudgeVerdict, judge_answer
from eval.metrics import EvalSummary, check_thresholds, compute_summary, passes_threshold

RESULTS_DIR = Path("results")


def run_eval(
    golden_path: Path = DEFAULT_DATASET_PATH, mode: str = "hybrid", k: int = 4
) -> tuple[EvalSummary, list[dict]]:
    items = load_golden_dataset(golden_path)
    verdicts: list[JudgeVerdict] = []
    categories: list[str] = []
    per_item = []
    for index, item in enumerate(items, start=1):
        chunks = retrieve(item.question, mode=mode, k=k)
        answer = generate_answer(item.question, chunks)
        verdict = judge_answer(
            item.question,
            chunks,
            answer,
            item.expected_answer,
            expected_sources=item.expected_sources,
            category=item.category,
        )
        verdicts.append(verdict)
        categories.append(item.category)
        per_item.append(
            {
                "id": f"q{index}",
                "question": item.question,
                "category": item.category,
                "generated_answer": answer,
                "expected_answer": item.expected_answer,
                "sources": [c.source for c in chunks],
                "expected_sources": item.expected_sources,
                "faithfulness": verdict.faithfulness.model_dump(),
                "correctness": verdict.correctness.model_dump(),
                "retrieval_quality": verdict.retrieval_quality.model_dump(),
                "passed": passes_threshold(verdict),
            }
        )
    summary = compute_summary(verdicts, categories)
    return summary, per_item


def save_results(
    summary: EvalSummary, per_item: list[dict], timestamp: str, results_dir: Path = RESULTS_DIR
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"eval_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": asdict(summary), "items": per_item}, f, indent=2)
    return path


LOW_SCORE_THRESHOLD = 0.5

METRIC_KEYS = ("faithfulness", "correctness", "retrieval_quality")


def find_low_scoring(per_item: list[dict], threshold: float = LOW_SCORE_THRESHOLD) -> list[dict]:
    flagged = []
    for item in per_item:
        low_metrics = [m for m in METRIC_KEYS if item[m]["score"] < threshold]
        if low_metrics:
            flagged.append({"id": item["id"], "question": item["question"], "low_metrics": low_metrics})
    return flagged


def print_summary(summary: EvalSummary, per_item: list[dict] | None = None) -> None:
    print(f"Items evaluated:       {summary.n_items}")
    print(f"Mean faithfulness:     {summary.mean_faithfulness:.2f}")
    print(f"Mean correctness:      {summary.mean_correctness:.2f}")
    print(f"Mean retrieval qual.:  {summary.mean_retrieval_quality:.2f}")
    print(f"Hallucination rate:    {summary.hallucination_rate:.2%}")
    print(f"Pass rate:             {summary.pass_rate:.2%}")
    if summary.correctness_by_category:
        print("Correctness by category:")
        for category, score in sorted(summary.correctness_by_category.items()):
            print(f"  {category:15s} {score:.2f}")
    if per_item is not None:
        low_scoring = find_low_scoring(per_item)
        if low_scoring:
            print(f"\nQuestions scoring below {LOW_SCORE_THRESHOLD} on any metric:")
            for entry in low_scoring:
                metrics = ", ".join(entry["low_metrics"])
                print(f"  [{entry['id']}] ({metrics}) {entry['question']}")


def main(argv: list[str] | None = None) -> int:
    summary, per_item = run_eval()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = save_results(summary, per_item, timestamp)
    print_summary(summary, per_item)
    print(f"\nResults written to {path}")
    violations = check_thresholds(summary)
    if violations:
        print("\nTHRESHOLD VIOLATIONS:")
        for v in violations:
            print(f"  - {v}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
