import json
from pathlib import Path

from ..base import DataClass


class MultiSourceQAItem(DataClass):
    def __init__(self, id: str, question: str, options: dict[str, str], answer: str, answer_idx: str) -> None:
        self.id = id
        self.question = question
        self.options = options
        self.answer = answer
        self.answer_idx = answer_idx

    def concepts(self) -> list[str]:
        return list(self.options.values())

    @classmethod
    def load(cls, file_path: str = "questions.jsonl") -> list["MultiSourceQAItem"]:
        path = Path(file_path)
        if not path.is_absolute() and not path.exists():
            path = Path(__file__).resolve().parent / file_path
        with path.open(encoding="utf-8") as f:
            return [cls(id=r["id"], question=r["question"], options=r["options"], answer=r["answer"],
                        answer_idx=r["answer_idx"]) for r in (json.loads(line) for line in f)]
