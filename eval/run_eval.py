import json
import time
from dataclasses import asdict
from pathlib import Path

from app.generate import generate_answer
from app.retrieve import retrieve
from eval.dataset import DEFAULT_DATASET_PATH, load_golden_dataset
from eval.judge import JudgeResult, judge_answer
from eval.metrics import EvalSummary, check_thresholds, compute_summary

RESULTS_DIR = Path("results")


def run_eval(
    golden_path: Path = DEFAULT_DATASET_PATH, mode: str = "hybrid", k: int = 4
) -> tuple[EvalSummary, list[dict]]:
    items = load_golden_dataset(golden_path)
    judge_results: list[JudgeResult] = []
    per_item = []
    for item in items:
        chunks = retrieve(item.question, mode=mode, k=k)
        answer = generate_answer(item.question, chunks)
        verdict = judge_answer(item.question, chunks, answer, item.expected_answer)
        judge_results.append(verdict)
        per_item.append(
            {
                "id": item.id,
                "question": item.question,
                "generated_answer": answer,
                "expected_answer": item.expected_answer,
                "sources": [c.source for c in chunks],
                **asdict(verdict),
            }
        )
    summary = compute_summary(judge_results)
    return summary, per_item


def save_results(
    summary: EvalSummary, per_item: list[dict], timestamp: str, results_dir: Path = RESULTS_DIR
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"eval_{timestamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"summary": asdict(summary), "items": per_item}, f, indent=2)
    return path


def print_summary(summary: EvalSummary) -> None:
    print(f"Items evaluated:     {summary.n_items}")
    print(f"Mean faithfulness:   {summary.mean_faithfulness:.2f}")
    print(f"Mean relevance:      {summary.mean_relevance:.2f}")
    print(f"Correctness rate:    {summary.correctness_rate:.2%}")
    print(f"Hallucination rate:  {summary.hallucination_rate:.2%}")


def main(argv: list[str] | None = None) -> int:
    summary, per_item = run_eval()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = save_results(summary, per_item, timestamp)
    print_summary(summary)
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
