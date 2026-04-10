import json
from pathlib import Path

from ..base import DataClass


class MedQAItem(DataClass):
    def __init__(self, question: str, options: dict[str, str], answer: str, answer_idx: str, meta_info: str) -> None:
        self.question = question
        self.options = options
        self.answer = answer
        self.answer_idx = answer_idx
        self.meta_info = meta_info

    def concepts(self) -> list[str]:
        return list(self.options.values())

    @classmethod
    def load(cls, file_path: str = "medqa_30.jsonl") -> list["MedQAItem"]:
        path = Path(file_path)
        if not path.is_absolute() and not path.exists():
            path = Path(__file__).resolve().parent / file_path
        with path.open(encoding="utf-8") as f:
            return [cls(question=r["question"], options=r["options"], answer=r["answer"],
                        answer_idx=r["answer_idx"], meta_info=r["meta_info"]) for r in (json.loads(line) for line in f)]
