"""``quest_answer`` evaluator — grade by comparing QUEST agent prediction to ground truth.

QUEST writes rollout output to:
  {output_base}/{model_basename}/{dataset_basename}/iter1.jsonl

Each line is a JSON record with a ``prediction`` field containing the agent's
final answer.  This evaluator finds the first ``iter1.jsonl`` under the
quest_output sub-directory of the agent log dir, reads the prediction, and
returns reward 1.0 when it matches the expected ``ground_truth``.

Config schema (EvaluatorSpec.config):
  ground_truth (str, required) — expected answer; use template syntax to
      inject from sample metadata, e.g. ``"{sample.metadata.ground_truth}"``.
  normalize    (bool, default True) — fold to lowercase + collapse whitespace
      before comparing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from polar.trajectory.evaluator.base import BaseTrajectoryEvaluator
from polar.trajectory.models import EvalResult, Trajectory

_QUEST_OUTPUT_SUBDIR = "quest_output"


class QuestAnswerEvaluator(BaseTrajectoryEvaluator):
    """Match QUEST agent prediction against a ground-truth string."""

    def __init__(
        self,
        *,
        ground_truth: str,
        normalize: bool = True,
    ) -> None:
        if not ground_truth:
            raise ValueError("quest_answer requires a non-empty 'ground_truth'")
        self.ground_truth = ground_truth
        self.normalize = normalize

    async def evaluate(self, trajectory: Trajectory, **runtime: Any) -> EvalResult:
        session_dir: Path = runtime["session_dir"]
        output_root = session_dir / "logs" / "agent" / _QUEST_OUTPUT_SUBDIR

        prediction, source_file = self._read_prediction(output_root)

        if prediction is None:
            return EvalResult(
                outcome_reward=0.0,
                metadata={
                    "error": "no_prediction",
                    "output_root": str(output_root),
                    "ground_truth": self.ground_truth,
                },
            )

        gt = self._norm(self.ground_truth) if self.normalize else self.ground_truth
        pred = self._norm(prediction) if self.normalize else prediction
        matched = pred == gt

        return EvalResult(
            outcome_reward=1.0 if matched else 0.0,
            metadata={
                "prediction": prediction,
                "ground_truth": self.ground_truth,
                "matched": matched,
                "source_file": str(source_file),
            },
        )

    def _read_prediction(self, output_root: Path) -> tuple[str | None, Path | None]:
        """Glob for the first iter1.jsonl written by QUEST and return its prediction."""
        if not output_root.exists():
            return None, None
        # QUEST writes iter1.jsonl (1-indexed rollout) when roll_out_count=1.
        candidates = sorted(output_root.rglob("iter1.jsonl"))
        for path in candidates:
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    pred = record.get("prediction", "")
                    if isinstance(pred, str) and pred.strip():
                        return pred.strip(), path
            except (OSError, json.JSONDecodeError):
                continue
        return None, None

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())
