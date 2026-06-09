#!/usr/bin/env python3
"""Standalone QUEST-style ReAct research agent runner.

uv run python runner.py --task '{"question": "..."}' --output /dir --model gpt-4

Written to /polar/session/quest_runner.py by QuestHarness.setup() and executed
inside the Polar container.  Only depends on: openai, requests, tiktoken.
No QUEST repo clone required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

SYSTEM_PROMPT = """\
You are a deep research assistant. Conduct thorough, multi-source investigations.
Use the available tools to gather information before answering.
When you have enough information, enclose your final answer in <answer>...</answer> tags.\
"""

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def _tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web for up-to-date information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of search queries.",
                        }
                    },
                    "required": ["queries"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "visit",
                "description": "Fetch the content of a URL and return a summary.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url":  {"type": "string", "description": "URL to visit."},
                        "goal": {"type": "string", "description": "What to extract from the page."},
                    },
                    "required": ["url", "goal"],
                },
            },
        },
    ]


def _search(queries: list[str], serper_key: str) -> str:
    results: list[str] = []
    for q in queries[:3]:  # cap at 3 queries per call
        try:
            resp = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                json={"q": q, "num": 5},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            snippets = [
                f"{item.get('title', '')}: {item.get('snippet', '')}"
                for item in data.get("organic", [])
            ]
            results.append(f"[{q}]\n" + "\n".join(snippets))
        except Exception as exc:
            results.append(f"[{q}] search error: {exc}")
    return "\n\n".join(results)


def _visit(url: str, goal: str, jina_keys: str) -> str:
    # Try Jina reader first; fall back to a plain GET + strip HTML.
    key = jina_keys.split(",")[0].strip() if jina_keys else ""
    try:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.text[:8000]
    except Exception:
        try:
            resp = requests.get(url, timeout=15)
            # Very lightweight HTML strip — no heavy deps required.
            text = re.sub(r"<[^>]+>", " ", resp.text)
            text = re.sub(r"\s+", " ", text)[:8000]
        except Exception as exc:
            return f"visit error: {exc}"
    return f"[goal: {goal}]\n{text}"


def _dispatch_tool(name: str, args: dict, serper_key: str, jina_keys: str) -> str:
    if name == "search":
        return _search(args.get("queries", []), serper_key)
    if name == "visit":
        return _visit(args.get("url", ""), args.get("goal", ""), jina_keys)
    return f"unknown tool: {name}"


# ---------------------------------------------------------------------------
# ReAct loop
# ---------------------------------------------------------------------------

def run_agent(
    question: str,
    *,
    client,
    model: str,
    max_turns: int,
    serper_key: str,
    jina_keys: str,
) -> tuple[str, list[dict], str]:
    """Run the ReAct loop.  Returns (prediction, messages, termination_reason)."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    tools = _tool_specs()

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message
        messages.append(msg.model_dump(exclude_unset=True))

        # Check for <answer> tag in text content (model may answer without tool call).
        if msg.content:
            m = re.search(r"<answer>(.*?)</answer>", msg.content, re.DOTALL)
            if m:
                return m.group(1).strip(), messages, "answer"

        # Execute tool calls if present.
        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            tool_results: list[dict] = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                result = _dispatch_tool(tc.function.name, args, serper_key, jina_keys)
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
            messages.extend(tool_results)
            continue

        # Model stopped without a tool call and without <answer> — treat as final.
        if msg.content:
            return msg.content.strip(), messages, "stop"
        break

    return "", messages, f"max_turns_{max_turns}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_client(base_url: str, api_key: str):
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key=api_key or "polar", timeout=300.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task",   required=True, help="JSON string with 'question' field")
    parser.add_argument("--output", required=True, help="Output directory for iter1.jsonl")
    parser.add_argument("--model",  default="model")
    parser.add_argument("--max_turns", type=int, default=int(os.getenv("MAX_LLM_CALL_PER_RUN", "30")))
    args = parser.parse_args()

    task = json.loads(args.task)
    question = task.get("question", "")
    if not question:
        sys.exit("task JSON must have a 'question' key")

    base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8100/v1")
    api_key  = os.environ.get("OPENAI_API_KEY", "polar")
    serper_key = os.environ.get("SERPER_KEY_ID", "")
    jina_keys  = os.environ.get("JINA_API_KEYS", "")

    client = _build_client(base_url, api_key)
    prediction, messages, termination = run_agent(
        question,
        client=client,
        model=args.model,
        max_turns=args.max_turns,
        serper_key=serper_key,
        jina_keys=jina_keys,
    )

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "question":    question,
        "prediction":  prediction,
        "termination": termination,
        "messages":    messages,
        "num_rounds":  len([m for m in messages if m.get("role") == "assistant"]),
    }
    # Matches QUEST's naming convention: iter{rollout_idx}.jsonl (1-indexed).
    (out_dir / "iter1.jsonl").write_text(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[quest] termination={termination} prediction={prediction[:80]!r}")


if __name__ == "__main__":
    main()
