import json
from pathlib import Path


def _load_jsonl_records(filename: str) -> str:
    path = Path(__file__).resolve().parent / filename
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    formatted_records = []
    for record in records:
        lines = []
        for key, value in record.items():
            if isinstance(value, list):
                rendered_value = ", ".join(str(item) for item in value)
            else:
                rendered_value = str(value)
            lines.append(f"{key}: {rendered_value}")
        formatted_records.append("\n".join(lines))
    return "\n\n".join(formatted_records)
