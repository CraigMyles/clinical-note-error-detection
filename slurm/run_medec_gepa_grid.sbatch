#!/usr/bin/env bash
#SBATCH --job-name=medec-gepa-grid
#SBATCH --partition=gpu.A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-27
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# ============================================
# User knobs (override via: sbatch --export=ALL,VAR=VAL,…)
# ============================================

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-dspy}"
PYTHON_BIN="${PYTHON_BIN:-python}"            # used to launch SGLang
RUNNER="${RUNNER:-uv run python}"             # used to run GEPA script
SCRIPT_PATH="${SCRIPT_PATH:-$REPO_ROOT/src/detect_gepa.py}"

TRAIN_CSV="${TRAIN_CSV:-data/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv}"
VAL_CSV="${VAL_CSV:-data/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv}"
OUTDIR="${OUTDIR:-results/gepa_grid}"

RUNS="${RUNS:-1}"                             # evaluation repeats per GEPA programme
SEED="${SEED:-42}"
GEPA_AUTO="${GEPA_AUTO:-heavy}"               # heavy for all experiments

BASE_PORT="${BASE_PORT:-7501}"                # base; per-job offset is added
WAIT_MINUTES="${WAIT_MINUTES:-120}"           # allow long model load
POLL_SECONDS="${POLL_SECONDS:-15}"

# WANDB / OpenAI – we require keys to be set in the environment
: "${OPENAI_API_KEY:?Must set OPENAI_API_KEY in your environment before sbatch}"
: "${WANDB_API_KEY:?Must set WANDB_API_KEY in your environment before sbatch}"
WANDB_PROJECT="${WANDB_PROJECT:-medec-detect-gepa}"

# Caches
#export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
#export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$HOME/.cache/sglang}"
#mkdir -p "$HF_HOME" "$SGLANG_CACHE_DIR" logs

# ============================================
# Experiment grid (reflector | inference)
# ============================================

# 28 combinations, in the order you specified
REFLECTORS=(
  "gpt-5"        "gpt-5"        "gpt-5"        "gpt-5"        "gpt-5"        "gpt-5"        "gpt-5"
  "qwen3-32b"    "qwen3-32b"    "qwen3-32b"    "qwen3-32b"    "qwen3-32b"    "qwen3-32b"
  "qwen3-14b"    "qwen3-14b"    "qwen3-14b"    "qwen3-14b"    "qwen3-14b"
  "qwen3-8b"     "qwen3-8b"     "qwen3-8b"     "qwen3-8b"
  "qwen3-4b"     "qwen3-4b"     "qwen3-4b"
  "qwen3-1.7b"   "qwen3-1.7b"
  "qwen3-0.6b"
)

INFERENCES=(
  "gpt-5"        "qwen3-32b"    "qwen3-14b"    "qwen3-8b"     "qwen3-4b"     "qwen3-1.7b"   "qwen3-0.6b"
  "qwen3-32b"    "qwen3-14b"    "qwen3-8b"     "qwen3-4b"     "qwen3-1.7b"   "qwen3-0.6b"
  "qwen3-14b"    "qwen3-8b"     "qwen3-4b"     "qwen3-1.7b"   "qwen3-0.6b"
  "qwen3-8b"     "qwen3-4b"     "qwen3-1.7b"   "qwen3-0.6b"
  "qwen3-4b"     "qwen3-1.7b"   "qwen3-0.6b"
  "qwen3-1.7b"   "qwen3-0.6b"
  "qwen3-0.6b"
)

# Map Qwen preset → HF repo
qwen_repo_for() {
  case "$1" in
    qwen3-32b)  echo "Qwen/Qwen3-32B" ;;
    qwen3-14b)  echo "Qwen/Qwen3-14B" ;;
    qwen3-8b)   echo "Qwen/Qwen3-8B"  ;;
    qwen3-4b)   echo "Qwen/Qwen3-4B"  ;;
    qwen3-1.7b) echo "Qwen/Qwen3-1.7B" ;;
    qwen3-0.6b) echo "Qwen/Qwen3-0.6B" ;;
    *)          echo "" ;;
  esac
}

is_qwen() {
  case "$1" in
    qwen3-32b|qwen3-14b|qwen3-8b|qwen3-4b|qwen3-1.7b|qwen3-0.6b) return 0 ;;
    *) return 1 ;;
  esac
}

# ============================================
# Environment initialisation
# ============================================

IDX="${SLURM_ARRAY_TASK_ID}"
REF_PRESET="${REFLECTORS[$IDX]}"
INF_PRESET="${INFERENCES[$IDX]}"

echo "Host: $(hostname)"
echo "Array index: $IDX"
echo "Reflector preset: $REF_PRESET"
echo "Inference preset: $INF_PRESET"

# Module system and CUDA
if ! type module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh || true
fi
module load cuda || true

# Conda
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then
  set +u
  . "$CONDA_SH"
  set -u
else
  echo "Warning: conda.sh not found at $CONDA_SH" >&2
fi
conda activate "$CONDA_ENV"

# ============================================
# SGLang orchestration
# ============================================

REF_IS_LOCAL=0
INF_IS_LOCAL=0
is_qwen "$REF_PRESET" && REF_IS_LOCAL=1 || true
is_qwen "$INF_PRESET" && INF_IS_LOCAL=1 || true

LOCAL1_NAME=""
LOCAL2_NAME=""

if [ "$REF_IS_LOCAL" -eq 1 ] && [ "$INF_IS_LOCAL" -eq 1 ]; then
  if [ "$REF_PRESET" = "$INF_PRESET" ]; then
    # same Qwen model used for both roles
    LOCAL1_NAME="$REF_PRESET"
  else
    # two different Qwen models
    LOCAL1_NAME="$INF_PRESET"
    LOCAL2_NAME="$REF_PRESET"
  fi
elif [ "$INF_IS_LOCAL" -eq 1 ]; then
  LOCAL1_NAME="$INF_PRESET"
elif [ "$REF_IS_LOCAL" -eq 1 ]; then
  LOCAL1_NAME="$REF_PRESET"
fi

SGLANG_PIDS=()
PORT_INF=""
PORT_REFLECTOR=""
ORIG_CUDA="${CUDA_VISIBLE_DEVICES:-}"

# launch_sglang MODEL_NAME PORT CUDA_DEVICES TP
launch_sglang() {
  local model_name="$1"
  local port="$2"
  local cuda_devices="$3"
  local tp="$4"

  local hf_model
  hf_model="$(qwen_repo_for "$model_name")"
  if [ -z "$hf_model" ]; then
    echo "Internal error: no HF repo for $model_name" >&2
    exit 2
  fi

  echo "Launching SGLang for $model_name on port $port with CUDA_VISIBLE_DEVICES=$cuda_devices TP=$tp"

  export CUDA_VISIBLE_DEVICES="$cuda_devices"
  "$PYTHON_BIN" -m sglang.launch_server \
    --port "$port" \
    --model-path "$hf_model" \
    --tp "$tp" \
    > "logs/sglang_${SLURM_JOB_ID}_${IDX}_${model_name}.log" 2>&1 &
  local pid=$!
  SGLANG_PIDS+=("$pid")
}

wait_for_ready() {
  local port="$1"
  local label="$2"

  echo "Waiting up to ${WAIT_MINUTES} minutes for $label on port $port"
  local deadline=$(( "$(date +%s)" + WAIT_MINUTES*60 ))
  while true; do
    if curl -sf "http://localhost:${port}/v1/models" | grep -q '"data"'; then
      echo "$label is ready on port $port"
      break
    fi
    if [ "$(date +%s)" -gt "$deadline" ]; then
      echo "Timed out waiting for $label on port $port"
      exit 124
    fi
    local any_alive=0
    for pid in "${SGLANG_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        any_alive=1
      fi
    done
    if [ "$any_alive" -eq 0 ]; then
      echo "All SGLang processes exited unexpectedly while loading"
      exit 125
    fi
    sleep "$POLL_SECONDS"
  done
}

cleanup() {
  echo "Cleaning up SGLang servers"
  for pid in "${SGLANG_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping PID $pid"
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 2
  for pid in "${SGLANG_PIDS[@]}"; do
    pkill -P "$pid" 2>/dev/null || true
  done
  # restore CUDA_VISIBLE_DEVICES
  if [ -n "$ORIG_CUDA" ]; then
    export CUDA_VISIBLE_DEVICES="$ORIG_CUDA"
  else
    unset CUDA_VISIBLE_DEVICES || true
  fi
}
trap cleanup EXIT

# Per-job port block
PORT_BASE=$(( BASE_PORT + 4 * IDX ))
PORT1=$(( PORT_BASE + 0 ))
PORT2=$(( PORT_BASE + 1 ))

# GPU split policy:
# - single local model (LOCAL1_NAME set, LOCAL2_NAME empty): use all 4 GPUs, TP=4
# - two local models (LOCAL1_NAME + LOCAL2_NAME): each gets 2 GPUs, TP=2

if [ -z "$LOCAL1_NAME" ] && [ -z "$LOCAL2_NAME" ]; then
  echo "No local Qwen models needed for this job (GPT-5 only)."

  # No SGLang; ports are irrelevant for OpenAI, but we still pass something.
  PORT_INF="$PORT1"
  PORT_REFLECTOR="$PORT1"

elif [ -n "$LOCAL1_NAME" ] && [ -z "$LOCAL2_NAME" ]; then
  echo "Single local Qwen model in this job: $LOCAL1_NAME"
  launch_sglang "$LOCAL1_NAME" "$PORT1" "0,1,2,3" "4"
  wait_for_ready "$PORT1" "$LOCAL1_NAME"

  # This one server can serve both roles
  PORT_INF="$PORT1"
  PORT_REFLECTOR="$PORT1"

elif [ -n "$LOCAL1_NAME" ] && [ -n "$LOCAL2_NAME" ]; then
  echo "Two different local Qwen models in this job: $LOCAL1_NAME and $LOCAL2_NAME"
  # Inference model on GPUs 0,1; reflector on 2,3 (both TP=2)
  launch_sglang "$LOCAL1_NAME" "$PORT1" "0,1" "2"
  wait_for_ready "$PORT1" "$LOCAL1_NAME"

  launch_sglang "$LOCAL2_NAME" "$PORT2" "2,3" "2"
  wait_for_ready "$PORT2" "$LOCAL2_NAME"

  # Map ports to roles
  if [ "$INF_IS_LOCAL" -eq 1 ]; then
    if [ "$INF_PRESET" = "$LOCAL1_NAME" ]; then
      PORT_INF="$PORT1"
    elif [ "$INF_PRESET" = "$LOCAL2_NAME" ]; then
      PORT_INF="$PORT2"
    fi
  fi

  if [ "$REF_IS_LOCAL" -eq 1 ]; then
    if [ "$REF_PRESET" = "$LOCAL1_NAME" ]; then
      PORT_REFLECTOR="$PORT1"
    elif [ "$REF_PRESET" = "$LOCAL2_NAME" ]; then
      PORT_REFLECTOR="$PORT2"
    fi
  fi
fi

# Fallbacks if one side is GPT-5 (ports unused by OpenAI)
[ -z "${PORT_INF}" ] && PORT_INF="$PORT1"
[ -z "${PORT_REFLECTOR}" ] && PORT_REFLECTOR="$PORT_INF"

echo "Port mapping:"
echo "  Inference preset:  $INF_PRESET  → PORT_INF=$PORT_INF"
echo "  Reflector preset:  $REF_PRESET  → PORT_REFLECTOR=$PORT_REFLECTOR"

# Restore CUDA visibility for the driver process
if [ -n "$ORIG_CUDA" ]; then
  export CUDA_VISIBLE_DEVICES="$ORIG_CUDA"
else
  unset CUDA_VISIBLE_DEVICES || true
fi

# ============================================
# Run GEPA script
# ============================================

mkdir -p "$OUTDIR"

$RUNNER "$SCRIPT_PATH" \
  --preset "$INF_PRESET" \
  --reflector-preset "$REF_PRESET" \
  --port "$PORT_INF" \
  --reflector-port "$PORT_REFLECTOR" \
  --auto "$GEPA_AUTO" \
  --runs "$RUNS" \
  --seed "$SEED" \
  --train-csv "$TRAIN_CSV" \
  --val-csv "$VAL_CSV" \
  --output-dir "$OUTDIR" \
  --wandb

echo "Done: reflector=$REF_PRESET inference=$INF_PRESET on $(hostname)"
