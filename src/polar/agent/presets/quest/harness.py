"""QUEST ReAct research agent harness for Polar.

Deploys the self-contained runner (quest/runner.py) into the container at
setup time — no external repo clone required.  The runner speaks directly to
the Polar gateway via the standard OPENAI_BASE_URL / OPENAI_API_KEY env vars.

Usage in polar_config.yaml:

    agent:
      harness: "quest"
      model_name: "your-model"
      settings:
        max_turns: 30     # max LLM calls per episode (default 30)
      env:
        SERPER_KEY_ID: "..."   # Serper search API key
        JINA_API_KEYS:  "..."  # Jina reader key(s), comma-separated
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from polar.agent.base import BaseHarness
from polar.runtime.base import BaseRuntime, RUNTIME_AGENT_LOG_DIR, RUNTIME_SESSION_DIR
from polar.runtime.models import ExecInput

# Where the runner script lives inside the container — outside the workspace
# so it stays out of any git diff the evaluator might collect.
_RUNNER_DST = f"{RUNTIME_SESSION_DIR}/quest_runner.py"
# QUEST-compatible output dir; evaluator globs for iter1.jsonl beneath it.
_OUTPUT_DIR = f"{RUNTIME_AGENT_LOG_DIR}/quest_output"
# Minimal deps — no QUEST repo needed.
_DEPS = "openai requests"

# Read runner source once at import time so the harness is self-contained.
_RUNNER_SRC = (Path(__file__).parent / "runner.py").read_text()


class QuestHarness(BaseHarness):
    """Deploy and run the self-contained QUEST ReAct runner inside Polar."""

    async def setup(self, runtime: BaseRuntime) -> None:
        escaped = shlex.quote(_RUNNER_SRC)
        await runtime.exec(f"printf '%s' {escaped} > {_RUNNER_DST}")
        await runtime.exec(f"pip install --quiet {_DEPS} 2>&1 | tail -3 || true")

    def run_steps(self, instruction: str) -> list[ExecInput]:
        task_json = shlex.quote(json.dumps({"question": instruction}))
        model = shlex.quote(self.model_name or "model")
        max_turns = int(self.settings.get("max_turns", self.settings.get("max_llm_calls", 30)))

        command = (
            f"python {_RUNNER_DST} "
            f"--task {task_json} "
            f"--output {_OUTPUT_DIR} "
            f"--model {model} "
            f"--max_turns {max_turns} "
            f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/quest.txt"
        )
        return [ExecInput(command=command, env=self.env)]
