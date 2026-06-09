# QUEST RL Slime GRPO

End-to-end training example: run the Polar QUEST agent on the released
`osunlp/QUEST-RL-Data` objective split, score rollouts with QUEST's own
DeepResearch reward code, and train through Slime async GRPO.

This example is intentionally close to `examples/swegym_slime_grpo`: Polar owns
agent rollouts, Slime owns SGLang engines and weight sync, and the bridge turns
Polar trajectories into Slime samples.

## Prerequisites

Install Polar with QUEST data/reward dependencies:

```bash
uv pip install -e ".[quest]"
```

Install Slime, Megatron-LM, SGLang patches, and convert the Qwen3.5-4B
checkpoint the same way as the SWE-Gym example:

```bash
PREPARE_IMAGES=0 bash examples/swegym_slime_grpo/launch_e2e.sh
```

Or use your existing checkouts with:

```bash
export SLIME_DIR=/path/to/slime
export MEGATRON_DIR=/path/to/Megatron-LM
export REF_LOAD=/path/to/Qwen3.5-4B_torch_dist
```

The QUEST repo must be available locally. By default the script uses
`../QUEST` from the Polar repo root:

```bash
export QUEST_ROOT=/fsx-alignment/home/zhuofeng/QUEST
```

Fill the search/visit and eval LLM credentials expected by QUEST. Minimum for
agent tools:

```bash
export SERPER_KEY_ID=...
export JINA_API_KEYS=...
```

Most objective verifier scripts call an eval LLM. Configure it via the QUEST
env variables or `QUEST_ROOT/training_scripts/rl/recipe/deepresearch/config/eval_llm_nodes.conf`.

## Quick Start

Prepare the objective training JSONL:

```bash
python examples/quest_rl/prepare_data.py
```

Start Polar services, Ray, Slime, and training:

```bash
bash examples/quest_rl/run.sh
```

Useful smoke-test knobs:

```bash
python examples/quest_rl/prepare_data.py --limit 8
ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=2 NUM_EPOCH=1 bash examples/quest_rl/run.sh
```

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Downloads `osunlp/QUEST-RL-Data`, filters objective rows, writes Slime JSONL |
| `polar_config.yaml` | QUEST agent runtime + QUEST reward evaluator config |
| `topology.yaml` | Polar rollout/gateway topology pointing to Slime's SGLang router |
| `run.sh` | Launches Polar services, Ray, and Slime async GRPO |

## Notes

- Default training model args reuse `examples/swegym_slime_grpo/model_args.sh`
  for `Qwen/Qwen3.5-4B`. Set `MODEL_ARGS_SCRIPT` when training another model.
- The evaluator strategy is `quest_rl_score`; it dynamically imports
  `QUEST_ROOT/training_scripts/rl/recipe/deepresearch/reward.py`.
- Open-ended QUEST RL rows are not enabled by default because they require a
  separate rubric judge LLM chain. Use `prepare_data.py --category open-ended`
  only after configuring the open-ended eval environment.
