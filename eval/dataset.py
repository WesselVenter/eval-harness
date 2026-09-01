import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_PATH = Path("data/golden_dataset.jsonl")


@dataclass
class GoldenItem:
    id: str
    question: str
    expected_answer: str
    expected_source_ids: list[str] | None = None


def load_golden_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[GoldenItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            items.append(
                GoldenItem(
                    id=data["id"],
                    question=data["question"],
                    expected_answer=data["expected_answer"],
                    expected_source_ids=data.get("expected_source_ids"),
                )
            )
    return items
