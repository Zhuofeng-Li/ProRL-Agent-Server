#!/usr/bin/env bash
# Async GRPO training on QUEST RL data via Polar + Slime.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/tmp/quest_rl}"
mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/logs"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

detect_host_ip() {
    "${PYTHON_BIN}" - <<'PY'
import socket
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("8.8.8.8", 80))
    print(sock.getsockname()[0])
    sock.close()
except Exception:
    print("127.0.0.1")
PY
}

is_path_like() {
    case "$1" in
        /*|./*|../*|~*) return 0 ;;
        *) return 1 ;;
    esac
}

SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
QUEST_ROOT="${QUEST_ROOT:-$(cd -- "${PROJECT_ROOT}/../QUEST" && pwd)}"
QUEST_EVAL_SCRIPTS_DIR="${QUEST_EVAL_SCRIPTS_DIR:-${QUEST_ROOT}/training_scripts/rl/recipe/deepresearch/eval_scripts}"
QUEST_RUNTIME_IMAGE="${QUEST_RUNTIME_IMAGE:-python:3.11-slim}"

if [ ! -f "${SLIME_DIR}/train_async.py" ]; then
    echo "ERROR: Slime not found at ${SLIME_DIR}" >&2
    echo "  git clone --branch v0.2.4 --depth 1 https://github.com/THUDM/slime.git ${SLIME_DIR}" >&2
    exit 1
fi
if [ ! -d "${MEGATRON_DIR}/megatron" ]; then
    echo "ERROR: Megatron-LM not found at ${MEGATRON_DIR}" >&2
    echo "  git clone https://github.com/NVIDIA/Megatron-LM.git ${MEGATRON_DIR}" >&2
    exit 1
fi
if [ ! -f "${QUEST_ROOT}/training_scripts/rl/recipe/deepresearch/reward.py" ]; then
    echo "ERROR: QUEST repo not found at ${QUEST_ROOT}" >&2
    exit 1
fi
if [ ! -d "${QUEST_EVAL_SCRIPTS_DIR}" ]; then
    echo "ERROR: QUEST eval scripts not found at ${QUEST_EVAL_SCRIPTS_DIR}" >&2
    exit 1
fi
"${PYTHON_BIN}" - <<'PY' || {
import aiohttp  # noqa: F401
PY
    echo "ERROR: missing QUEST runtime deps. Run: uv pip install -e '.[quest]'" >&2
    exit 1
}

bash "${PROJECT_ROOT}/scripts/patch/patch_slime.sh" "${SLIME_DIR}"

HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist}"
RUN_ID="${RUN_ID:-quest-rl-$(date -u +%Y%m%dT%H%M%SZ)}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/tmp/ckpt/quest_rl_qwen35_4b}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${RUN_ID}}"
mkdir -p "${SAVE_DIR}"

if is_path_like "${HF_CHECKPOINT}" && [ ! -e "${HF_CHECKPOINT}" ]; then
    echo "ERROR: HF checkpoint not found at ${HF_CHECKPOINT}" >&2
    exit 1
fi
if [ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: Megatron torch_dist checkpoint not found at ${REF_LOAD}" >&2
    echo "  Run: bash examples/swegym_slime_grpo/convert_weights.sh" >&2
    exit 1
fi
if [ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="${SAVE_DIR}"
else
    LOAD_DIR="${REF_LOAD}"
fi

# Qwen3.5-4B Megatron args are shared with the SWE-Gym training example.
# Override MODEL_ARGS_SCRIPT when training a different architecture.
MODEL_ARGS_SCRIPT="${MODEL_ARGS_SCRIPT:-${PROJECT_ROOT}/examples/swegym_slime_grpo/model_args.sh}"
# shellcheck source=/dev/null
source "${MODEL_ARGS_SCRIPT}"

PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/quest_rl_objective_train.jsonl}"
if [ ! -f "${PROMPT_DATA}" ]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py" --output "${PROMPT_DATA}"
fi

SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-9000}"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-$(detect_host_ip)}"
export SGLANG_ROUTER_BASE_URL="${SGLANG_ROUTER_BASE_URL:-http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}}"
export QUEST_ROOT QUEST_EVAL_SCRIPTS_DIR QUEST_RUNTIME_IMAGE
export SERPER_KEY_ID="${SERPER_KEY_ID:-}"
export JINA_API_KEYS="${JINA_API_KEYS:-${JINA_API_KEY:-}}"
export EVAL_LLM_MODEL_NAME="${EVAL_LLM_MODEL_NAME:-default}"

command -v envsubst >/dev/null || { echo "ERROR: envsubst not found (install gettext-base)" >&2; exit 1; }
TOPOLOGY_PATH="${TOPOLOGY_PATH:-${RUN_DIR}/topology.yaml}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${RUN_DIR}/polar_config.yaml}"
TEMPLATE_VARS='${SGLANG_ROUTER_BASE_URL} ${QUEST_ROOT} ${QUEST_EVAL_SCRIPTS_DIR} ${QUEST_RUNTIME_IMAGE} ${SERPER_KEY_ID} ${JINA_API_KEYS} ${EVAL_LLM_MODEL_NAME}'
envsubst "$TEMPLATE_VARS" < "${SCRIPT_DIR}/topology.yaml" > "${TOPOLOGY_PATH}"
envsubst "$TEMPLATE_VARS" < "${SCRIPT_DIR}/polar_config.yaml" > "${CUSTOM_CONFIG_PATH}"

echo "Using topology: ${TOPOLOGY_PATH}"
echo "Using Polar config: ${CUSTOM_CONFIG_PATH}"
echo "Using QUEST root: ${QUEST_ROOT}"
echo "Using prompt data: ${PROMPT_DATA}"
echo "Using save dir: ${SAVE_DIR}"

PIDS=()
cleanup() {
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    ray stop --force 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

polar serve_rollout -c "${TOPOLOGY_PATH}" &
PIDS+=($!)
sleep 2
polar serve_gateway -c "${TOPOLOGY_PATH}" --node-id localhost-node-01 &
PIDS+=($!)
sleep 2
curl -sf http://127.0.0.1:8080/health >/dev/null || { echo "Polar rollout server not healthy" >&2; exit 1; }

ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-6}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-2}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-60000}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-50000}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-$((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))}"
RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"

ray stop --force 2>/dev/null || true
sleep 1
ray start --head --node-ip-address "${RAY_HEAD_IP}" --num-gpus "${RAY_NUM_GPUS}" --disable-usage-stats

CUDNN_LIB="${CUDNN_LIB:-$("${PYTHON_BIN}" -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)}"
RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ -n "${CUDNN_LIB}" ] && [ -d "${CUDNN_LIB}" ]; then
    RUNTIME_LD_LIBRARY_PATH="${CUDNN_LIB}:${RUNTIME_LD_LIBRARY_PATH}"
fi
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src:${QUEST_ROOT}/training_scripts/rl:${QUEST_ROOT}/training_scripts/rl/recipe/deepresearch\",
    \"PATH\": \"${PYTHON_BIN_DIR}:${PATH}\",
    \"VIRTUAL_ENV\": \"${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WANDB_DIR\": \"${PROJECT_ROOT}/logs\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\"
  }
}"

ray job submit --address="http://${RAY_HEAD_IP}:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${PYTHON_BIN}" "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${HF_CHECKPOINT}" \
    --ref-load "${REF_LOAD}" \
    --load "${LOAD_DIR}" \
    --save "${SAVE_DIR}" \
    --save-interval "${SAVE_INTERVAL:-10}" \
    --update-weights-interval 1 \
    --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async \
    --custom-rm-path slime_bridge.reward.reward_func \
    --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards \
    --custom-config-path "${CUSTOM_CONFIG_PATH}" \
    --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer \
    --prompt-data "${PROMPT_DATA}" \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --rollout-shuffle \
    --reward-key score \
    --num-epoch "${NUM_EPOCH:-1}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}" \
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-16000}" \
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN:-32000}" \
    --dynamic-history \
    --num-steps-per-rollout 1 \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}" \
    --log-probs-chunk-size 256 \
    --advantage-estimator grpo \
    --normalize-advantages \
    --use-tis \
    --use-kl-loss \
    --kl-loss-coef "${KL_LOSS_COEF:-0.001}" \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --optimizer adam \
    --lr "${LR:-1e-6}" \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend auto \
    --no-gradient-accumulation-fusion \
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.8}" \
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}" \
    --sglang-tool-call-parser qwen3_coder \
    --router-policy "${SGLANG_ROUTER_POLICY:-round_robin}" \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT:-polar-quest-rl}" \
    --wandb-group "${WANDB_GROUP:-quest-rl-qwen35-4b-async-grpo}" \
    --sglang-router-port "${SGLANG_ROUTER_PORT}"
