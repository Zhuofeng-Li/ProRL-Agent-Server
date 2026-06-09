# Polar Agent Setup: Best Practices (H200 / Amazon Linux 2)

This document captures the environment-specific setup knowledge learned while
getting the `swegym_slime_grpo` example running on an H200 cluster with
Amazon Linux 2 (glibc 2.26, CUDA driver 550).

---

## Environment Constraints

| Constraint | Value | Impact |
|---|---|---|
| OS | Amazon Linux 2 | glibc 2.26 |
| CUDA driver | 550.144.03 | Max native CUDA 12.4 |
| GPU | 8× H200 SXM (SM90) | Not SM100/B200 |
| System GCC | 7.3.1 | Too old for numpy ≥ 2.4 source builds |

### glibc 2.26 Package Compatibility

Many modern ML packages ship `manylinux_2_28` wheels that require glibc ≥ 2.28.
**Do not try to install these natively**; use Docker.

| Package | Required | Solution |
|---|---|---|
| `torch 2.9.1+cu128` | glibc 2.28 | Run in Docker |
| `numpy ≥ 2.4` | glibc 2.27 | Pin `numpy==2.2.6` on host |
| `pyarrow ≥ 24` | glibc 2.27 | Pin `pyarrow<20` on host |
| `cbor2 ≥ 6` | Rust compiler | Pin `cbor2<6` on host |

### CUDA Forward Compatibility in Docker

Driver 550 supports CUDA 12.4 **natively**, but Docker containers from
`nvcr.io/nvidia/cuda:12.8.1-*` work via NVIDIA forward compatibility.
Even CUDA 13.0 (`lmsysorg/sglang:latest`) functions correctly on driver 550.

```
docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
# Shows: CUDA Version: 12.8  ✓
```

---

## Architecture: Split Host + Docker

Run Polar services on the host (pure Python, no CUDA needed). Run the
training stack (torch, SGLang, Slime, Megatron) inside Docker.

```
Host:
  polar serve_rollout  :8080   ← task coordination
  polar serve_gateway  :8100   ← agent dispatch

Docker (--network host, --gpus all):
  ray start --head
  SGLang inference engines (managed by Slime)
  Megatron GRPO training
```

### Why `--network host`

The training container reaches Polar on the host via 127.0.0.1:8080/8100.
`--network host` is the simplest way; for multi-node setups use the host IP.

---

## Host Python Environment

Use Python 3.12 from conda (`/opt/conda/bin/python3`) for the host venv —
it has a lower glibc ABI baseline than the uv-managed Python 3.13.

```bash
uv venv --python /opt/conda/bin/python3
uv pip install -e ".[swebench]" datasets<5 pyarrow<20 numpy==2.2.6 cbor2<6
```

Install sglang + patch on the host venv too (Polar gateway may reference it):

```bash
uv pip install --prerelease=allow --no-deps sglang==0.5.10
uv pip install pybase64 IPython   # sglang import-time deps
bash scripts/patch/patch_sglang.sh
```

---

## Training Docker Image

Use `nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` as the base (Ubuntu
22.04, glibc 2.35). The `devel` variant includes nvcc and cuDNN headers needed
to build Transformer Engine from source.

```dockerfile
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
# Install python3.12, uv, then:
# 1. torch 2.9.1+cu128 from download.pytorch.org/whl/cu128
# 2. sglang 0.5.10 (--no-deps, torch pre-installed)
# 3. flash-linear-attention 0.5.0
# 4. transformer-engine[pytorch]==2.5.0 (built from source inside container)
```

Build once, reuse for all runs:
```bash
docker build -f examples/swegym_slime_grpo/Dockerfile.training -t polar-training .
```

---

## Apptainer → Docker for SWE-Gym Tasks

Amazon Linux 2 has no `fuse3` package, so apptainer RPM cannot be installed.
**Use Docker instead**: Polar natively supports a `docker` runtime backend.

Changes needed:
1. `polar_config_docker.yaml` — set `runtime.backend: "docker"`, use
   `image: "{sample.metadata.docker_image}"`.
2. `prepare_data.py` — add `docker_image` to each row's metadata:
   ```python
   "docker_image": registry_image_for_instance_id(instance_id)
   # e.g. xingyaoww/sweb.eval.x86_64.astropy_s_astropy-12907:latest
   ```
3. `launch_e2e_docker.sh` / `run_docker.sh` — set
   `POLAR_CONFIG_TEMPLATE=…/polar_config_docker.yaml` and run training via
   `docker run`.

No SIF files needed; Docker images are pulled on demand.

---

## HuggingFace Cache

`/home/zhuofeng/.cache/huggingface/hub` is owned by `root`. Always redirect:

```bash
export HF_HOME=/path/to/project/tmp/hf_cache
```

Or pass `-v "${HF_HOME}:/root/.cache/huggingface"` to the Docker container.

---

## Qwen3.5-4B Model Download

```bash
HF_HOME=/home/zhuofeng/ProRL-Agent-Server/tmp/hf_cache \
  python3 -c "
import os; os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3.5-4B')
"
```

The model is ~8 GB. The Qwen3.5-4B is a **VLM checkpoint**
(`Qwen3_5ForConditionalGeneration`) — weight conversion uses
`slime_plugins.mbridge.qwen3_5` (text_config-aware) via `convert_weights.sh`.

---

## Megatron-LM Version and Patches

Slime v0.2.4 requires a **specific Megatron-LM commit** and two patches:

```bash
cd Megatron-LM
git checkout 3714d81d418c9f1bca4594fc35f9e8289f652862
git apply slime/docker/patch/v0.5.9/megatron.patch
```

The current `main` branch of Megatron-LM removed `megatron.training.tokenizer` — cloning
`main` will fail with `ModuleNotFoundError: No module named 'megatron.training.tokenizer'`.

Also patch out the overly-conservative numpy assertion in Slime:
```python
# slime/slime/backends/megatron_utils/initialize.py
# Remove or comment out:
assert np.__version__.startswith("1."), "Megatron does not support numpy 2.x"
```
sglang 0.5.10's full dep tree requires numpy 2.x; the assertion is incorrect.

---

## Weight Conversion (inside Docker)

Weight conversion requires `torchrun` + Megatron + Slime. Run it inside the
training container so torch 2.9.1 and the correct CUDA are available:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -v $HF_HOME:/root/.cache/huggingface \
  -w /workspace \
  -e HF_CHECKPOINT=Qwen/Qwen3.5-4B \
  -e TORCH_DIST_DIR=/workspace/tmp/checkpoints/Qwen3.5-4B_torch_dist \
  -e SLIME_DIR=/workspace/slime \
  -e MEGATRON_DIR=/workspace/Megatron-LM \
  -e PYTHON_BIN=/opt/training-venv/bin/python3 \
  polar-training \
  bash /workspace/examples/swegym_slime_grpo/convert_weights.sh
```

---

## Key Files Added for This Setup

| File | Purpose |
|---|---|
| `examples/swegym_slime_grpo/Dockerfile.training` | Training container |
| `examples/swegym_slime_grpo/launch_e2e_docker.sh` | Full e2e launcher (H200 variant) |
| `examples/swegym_slime_grpo/run_docker.sh` | Polar + Docker training runner |
| `examples/swegym_slime_grpo/polar_config_docker.yaml` | Polar config with Docker runtime |

---

## Quick Start (after environment is set up)

```bash
# 1. Build training image (once)
docker build -f examples/swegym_slime_grpo/Dockerfile.training -t polar-training .

# 2. Launch
export WANDB_API_KEY=<your-key>
export HF_HOME=$(pwd)/tmp/hf_cache
bash examples/swegym_slime_grpo/launch_e2e_docker.sh
```
