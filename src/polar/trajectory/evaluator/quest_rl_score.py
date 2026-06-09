"""``quest_rl_score`` evaluator — grade QUEST RL tasks with QUEST's reward code."""

from __future__ import annotations

from copy import deepcopy
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory

_QUEST_OUTPUT_SUBDIR = "quest_output"


class QuestRlScoreEvaluator(BaseTrajectoryEvaluator):
    """Call QUEST DeepResearch ``compute_score`` on the agent final answer."""

    def __init__(
        self,
        *,
        quest_root: str,
        data_source: str = "deepresearch_tasks",
        reward_model: dict[str, Any] | str | None = None,
        extra_info: dict[str, Any] | str | None = None,
        eval_scripts_dir: str | None = None,
        use_default_scoring: bool = False,
        enable_inline_citation_score: bool = False,
        base_score_weight: float = 1.0,
        eval_llm_nodes_conf: str | None = None,
        eval_llm_model: str = "default",
    ) -> None:
        self.quest_root = Path(quest_root).expanduser().resolve()
        self.data_source = data_source
        self.reward_model = self._coerce_mapping(reward_model)
        self.extra_info = self._coerce_mapping(extra_info)
        self.eval_scripts_dir = (
            str(Path(eval_scripts_dir).expanduser().resolve())
            if eval_scripts_dir
            else str(self.quest_root / "training_scripts/rl/recipe/deepresearch/eval_scripts")
        )
        self.use_default_scoring = use_default_scoring
        self.enable_inline_citation_score = enable_inline_citation_score
        self.base_score_weight = float(base_score_weight)
        self.eval_llm_nodes_conf = eval_llm_nodes_conf
        self.eval_llm_model = eval_llm_model

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        session_dir: Path = runtime["session_dir"]
        output_root = session_dir / "logs" / "agent" / _QUEST_OUTPUT_SUBDIR

        prediction, raw_response, source_file = self._read_prediction(output_root)
        if not prediction and not raw_response:
            return EvalResult(
                outcome_reward=0.0,
                metadata={"error": "no_prediction", "output_root": str(output_root)},
            )

        reward_module = self._load_reward_module()
        reward_model = deepcopy(self.reward_model)
        ground_truth = self._coerce_mapping(reward_model.get("ground_truth"))
        extra_info = deepcopy(self.extra_info)
        task_id = (
            ground_truth.get("task_id")
            or reward_model.get("task_id")
            or extra_info.get("task_id")
            or extra_info.get("original_task_id")
            or "unknown"
        )
        ground_truth.setdefault("task_id", task_id)
        extra_info.setdefault("task_id", task_id)
        if raw_response:
            extra_info.setdefault("full_response", raw_response)

        result = await reward_module.compute_score(
            data_source=self.data_source,
            solution_str=raw_response or prediction or "",
            ground_truth=ground_truth,
            extra_info=extra_info,
            eval_scripts_dir=self.eval_scripts_dir,
            use_default_scoring=self.use_default_scoring,
            enable_inline_citation_score=self.enable_inline_citation_score,
            base_score_weight=self.base_score_weight,
            eval_llm_nodes_conf=self.eval_llm_nodes_conf,
            eval_llm_model=self.eval_llm_model,
        )
        score = float(result.get("score", 0.0) or 0.0)
        return EvalResult(
            outcome_reward=score,
            metadata={
                "prediction": prediction,
                "score": score,
                "quest_reward": result,
                "task_id": task_id,
                "source_file": str(source_file) if source_file else None,
            },
        )

    def _load_reward_module(self) -> Any:
        rl_root = self.quest_root / "training_scripts/rl"
        deepresearch_dir = rl_root / "recipe/deepresearch"
        reward_path = deepresearch_dir / "reward.py"
        if not reward_path.exists():
            raise FileNotFoundError(f"QUEST reward.py not found: {reward_path}")
        for path in (str(rl_root), str(deepresearch_dir)):
            if path not in sys.path:
                sys.path.insert(0, path)
        spec = importlib.util.spec_from_file_location("quest_deepresearch_reward", reward_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import QUEST reward module from {reward_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _read_prediction(self, output_root: Path) -> tuple[str | None, str | None, Path | None]:
        if not output_root.exists():
            return None, None, None
        for path in sorted(output_root.rglob("iter1.jsonl")):
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    prediction = record.get("prediction", "")
                    raw_response = self._raw_response_from_messages(record.get("messages", []))
                    if prediction or raw_response:
                        return str(prediction).strip(), raw_response, path
            except (OSError, json.JSONDecodeError):
                continue
        return None, None, None

    @staticmethod
    def _raw_response_from_messages(messages: Any) -> str | None:
        if not isinstance(messages, list):
            return None
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
        return None

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return {}
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(raw)
                except (SyntaxError, ValueError):
                    return {"raw": raw}
            return dict(parsed) if isinstance(parsed, dict) else {"value": parsed}
        return {"value": value}
