from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    score: float = 0.0
