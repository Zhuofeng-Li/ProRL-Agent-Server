#!/usr/bin/env python3
"""Prepare GAIA tasks for Polar + QUEST training.

uv run python prepare_data.py [--input path/to/gaia.jsonl] [--output quest_gaia_tasks.jsonl]

Input JSONL format (QUEST-style GAIA):
  {"question": "...", "answer": "...", "task_id": "..."}

Output JSONL format (Polar/Slime):
  {
    "prompt":    [{"role": "user", "content": "<question>"}],
    "label":     "",   # reward comes from quest_answer evaluator
    "metadata":  {"question": ..., "ground_truth": ..., "task_id": ...}
  }

You can source the input file from QUEST's bundled GAIA subset:
  https://github.com/OSU-NLP-Group/QUEST/raw/main/evaluation/gaia/gaia-text-only-103.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_INPUT = Path(__file__).parent / "gaia-text-only-103.jsonl"
DEFAULT_OUTPUT = Path(__file__).parent / "quest_gaia_tasks.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def row_for_item(item: dict) -> dict:
    question = item.get("question") or item.get("Question", "")
    # GAIA uses "answer" or "Final answer" depending on subset version.
    ground_truth = (
        item.get("answer")
        or item.get("Final answer")
        or item.get("ground_truth")
        or ""
    )
    task_id = item.get("task_id") or item.get("id") or ""
    return {
        "prompt": [{"role": "user", "content": question.strip()}],
        "label": "",
        "metadata": {
            "question": question.strip(),
            "ground_truth": str(ground_truth).strip(),
            "task_id": task_id,
        },
    }


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            "Download it from:\n"
            "  https://github.com/OSU-NLP-Group/QUEST/raw/main/evaluation/gaia/gaia-text-only-103.jsonl"
        )
    items = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    rows = [row_for_item(item) for item in items]
    args.output.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n")
    print(f"Wrote {len(rows)} tasks to {args.output}")


if __name__ == "__main__":
    main()
