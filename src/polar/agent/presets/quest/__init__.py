"""QUEST ReAct research agent — harness package for Polar.

All QUEST evaluation logic lives in polar.trajectory.evaluator.quest_answer,
following the same layout as every other built-in evaluator.

Usage in polar_config.yaml:

    agent:
      harness: "quest"          # short name registered in polar.agent.factory
      # or: import_path: "polar.agent.presets.quest:QuestHarness"

    evaluator:
      strategy: "quest_answer"  # registered in polar.trajectory.registry
      config:
        ground_truth: "{sample.metadata.ground_truth}"
"""

from polar.agent.presets.quest.harness import QuestHarness

__all__ = ["QuestHarness"]
