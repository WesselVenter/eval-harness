import json
import sys
from pathlib import Path

METRIC_KEYS = ("faithfulness", "correctness", "retrieval_quality")


def load_run(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _item_scores(item: dict) -> dict[str, float]:
    return {metric: item[metric]["score"] for metric in METRIC_KEYS}


def compare_runs(run_a: dict, run_b: dict) -> dict:
    items_a = {item["question"]: item for item in run_a["items"]}
    items_b = {item["question"]: item for item in run_b["items"]}

    improved = []
    regressed = []
    unchanged = []
    only_in_a = sorted(set(items_a) - set(items_b))
    only_in_b = sorted(set(items_b) - set(items_a))

    for question in sorted(set(items_a) & set(items_b)):
        item_a, item_b = items_a[question], items_b[question]
        scores_a, scores_b = _item_scores(item_a), _item_scores(item_b)
        delta = sum(scores_b.values()) - sum(scores_a.values())
        entry = {
            "id": item_b["id"],
            "question": question,
            "scores_a": scores_a,
            "scores_b": scores_b,
            "delta": delta,
        }
        if delta > 1e-9:
            improved.append(entry)
        elif delta < -1e-9:
            regressed.append(entry)
        else:
            unchanged.append(entry)

    improved.sort(key=lambda e: e["delta"], reverse=True)
    regressed.sort(key=lambda e: e["delta"])

    hallucination_a = run_a["summary"]["hallucination_rate"]
    hallucination_b = run_b["summary"]["hallucination_rate"]

    return {
        "improved": improved,
        "regressed": regressed,
        "unchanged": unchanged,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "hallucination_rate_a": hallucination_a,
        "hallucination_rate_b": hallucination_b,
        "hallucination_rate_delta": hallucination_b - hallucination_a,
    }


def _format_scores(scores: dict[str, float]) -> str:
    return ", ".join(f"{k[0]}={v:.2f}" for k, v in scores.items())


def print_comparison(diff: dict, label_a: str, label_b: str) -> None:
    print(f"Comparing {label_a} -> {label_b}\n")

    print(
        f"Hallucination rate:    {diff['hallucination_rate_a']:.2%} -> "
        f"{diff['hallucination_rate_b']:.2%} "
        f"({diff['hallucination_rate_delta']:+.2%})"
    )

    print(f"\nImproved ({len(diff['improved'])}):")
    for entry in diff["improved"]:
        print(
            f"  [{entry['id']}] {entry['question']}\n"
            f"      {_format_scores(entry['scores_a'])} -> {_format_scores(entry['scores_b'])} "
            f"(delta {entry['delta']:+.2f})"
        )

    print(f"\nRegressed ({len(diff['regressed'])}):")
    for entry in diff["regressed"]:
        print(
            f"  [{entry['id']}] {entry['question']}\n"
            f"      {_format_scores(entry['scores_a'])} -> {_format_scores(entry['scores_b'])} "
            f"(delta {entry['delta']:+.2f})"
        )

    print(f"\nUnchanged: {len(diff['unchanged'])}")

    if diff["only_in_a"]:
        print(f"\nOnly in {label_a} (missing from {label_b}):")
        for question in diff["only_in_a"]:
            print(f"  {question}")

    if diff["only_in_b"]:
        print(f"\nOnly in {label_b} (missing from {label_a}):")
        for question in diff["only_in_b"]:
            print(f"  {question}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("Usage: python -m eval.compare_runs <results_a.json> <results_b.json>")
        return 2

    path_a, path_b = Path(argv[0]), Path(argv[1])
    run_a, run_b = load_run(path_a), load_run(path_b)
    diff = compare_runs(run_a, run_b)
    print_comparison(diff, path_a.name, path_b.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
