import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_PATH = Path("data/golden_dataset.jsonl")

REQUIRED_FIELDS = ("question", "expected_answer", "expected_sources", "category")


@dataclass
class GoldenItem:
    question: str
    expected_answer: str
    expected_sources: list[str]
    category: str


def load_golden_dataset(path: Path = DEFAULT_DATASET_PATH) -> list[GoldenItem]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            for field in REQUIRED_FIELDS:
                if field not in data:
                    raise ValueError(
                        f"Missing required field '{field}' on line {line_number} of {path}"
                    )
            items.append(
                GoldenItem(
                    question=data["question"],
                    expected_answer=data["expected_answer"],
                    expected_sources=data["expected_sources"],
                    category=data["category"],
                )
            )
    return items
