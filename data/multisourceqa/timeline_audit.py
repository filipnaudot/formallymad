from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent


def read_jsonl(name: str) -> list[dict]:
    path = DATA_DIR / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    slack = read_jsonl("slack.jsonl")
    company = read_jsonl("company.jsonl")
    calendar = read_jsonl("calendar.jsonl")

    now = max(pd.to_datetime(row["timestamp"], utc=True) for row in slack)
    docs_after_now = [row["id"]
                      for row in company
                      if pd.to_datetime(row["last_updated"], utc=True) > now]
    calendar_not_future = [row["id"]
                           for row in calendar
                           if pd.to_datetime(row["start"], utc=True) <= now]

    print(f"now: {now.isoformat()}")
    print("slack span:",
          min(pd.to_datetime(row["timestamp"], utc=True) for row in slack).isoformat(),
          "to",
          now.isoformat())
    print("company span:",
          min(pd.to_datetime(row["last_updated"], utc=True) for row in company).date(),
          "to",
          max(pd.to_datetime(row["last_updated"], utc=True) for row in company).date())
    print("calendar span:",
          min(pd.to_datetime(row["start"], utc=True) for row in calendar).isoformat(),
          "to",
          max(pd.to_datetime(row["start"], utc=True) for row in calendar).isoformat())
    print("docs_after_now:", docs_after_now or "ok")
    print("calendar_not_future:", calendar_not_future or "ok")


if __name__ == "__main__":
    main()
