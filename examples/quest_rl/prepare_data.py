#!/usr/bin/env python3
"""Prepare QUEST RL data for Polar + Slime training.

Default source:
  osunlp/QUEST-RL-Data, split=train

Output rows keep the Slime fields used by ``slime_bridge``:
  prompt, label, metadata
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

DEFAULT_DATASET = "osunlp/QUEST-RL-Data"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "quest_rl_objective_train.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--category",
        default="objective",
        choices=["objective", "open-ended", "all"],
        help="Default objective because QUEST open-ended rewards need a judge-LLM chain.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows; 0 means all.")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Only used with --backend datasets.",
    )
    parser.add_argument(
        "--backend",
        default="hf_rows",
        choices=["hf_rows", "datasets"],
        help="hf_rows avoids pyarrow/datasets and works in lightweight envs.",
    )
    return parser.parse_args()


def _parse_maybe_dict(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return raw


def row_for_item(item: dict[str, Any], index: int) -> dict[str, Any]:
    prompt = item.get("prompt") or []
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    reward_model = _parse_maybe_dict(item.get("reward_model"))
    extra_info = _parse_maybe_dict(item.get("extra_info"))
    if isinstance(extra_info, dict):
        extra_info.setdefault("index", index)
    else:
        extra_info = {"raw": extra_info, "index": index}

    return {
        "prompt": prompt,
        "label": "",
        "metadata": {
            "data_source": item.get("data_source", "deepresearch_tasks"),
            "reward_model": reward_model,
            "extra_info": extra_info,
            "rl_task_category": item.get("rl_task_category", ""),
        },
    }


def iter_hf_rows(dataset: str, split: str, *, page_size: int = 100):
    offset = 0
    config = "default"
    while True:
        query = urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            }
        )
        with urlopen(f"https://datasets-server.huggingface.co/rows?{query}", timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rows = payload.get("rows", [])
        if not rows:
            break
        for wrapped in rows:
            yield wrapped.get("row", wrapped)
        offset += len(rows)
        if len(rows) < page_size:
            break


def iter_dataset_rows(args: argparse.Namespace):
    if args.backend == "hf_rows":
        return iter_hf_rows(args.dataset, args.split)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Either install datasets/pyarrow or use "
            "`--backend hf_rows` (the default)."
        ) from exc
    return iter(load_dataset(args.dataset, split=args.split, streaming=args.streaming))


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for item in iter_dataset_rows(args):
        category = str(item.get("rl_task_category", ""))
        if args.category != "all" and category != args.category:
            continue
        rows.append(row_for_item(item, len(rows)))
        if args.limit and len(rows) >= args.limit:
            break

    if not rows:
        raise RuntimeError(f"No rows selected from {args.dataset} split={args.split}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n")
    print(f"Wrote {len(rows)} QUEST RL {args.category} rows to {args.output}")


if __name__ == "__main__":
    main()
