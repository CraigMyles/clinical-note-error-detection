#!/usr/bin/env bash
#SBATCH --job-name=medec-gepa-test
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

# -------------------------------------------------------------------------
# Repo and required environment
# -------------------------------------------------------------------------

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

: "${VAL_CSV:?Must set VAL_CSV to the test CSV (MS or UW) before submitting}"
: "${OUTDIR:?Must set OUTDIR to the output directory before submitting}"
: "${OPENAI_API_KEY:?Must set OPENAI_API_KEY in your environment before sbatch}"
: "${WANDB_API_KEY:?Must set WANDB_API_KEY in your environment before sbatch}"

CONDA_ENV="${CONDA_ENV:-dspy}"
PYTHON_BIN="${PYTHON_BIN:-python}"          # SGLang and helpers
RUNNER="${RUNNER:-uv run python}"           # detect_gepa.py
SCRIPT_PATH="${SCRIPT_PATH:-$REPO_ROOT/src/detect_gepa.py}"

TRAIN_CSV="${TRAIN_CSV:-data/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv}"
RUNS="${RUNS:-3}"
SEED="${SEED:-42}"

BASE_PORT="${BASE_PORT:-7501}"
WAIT_MINUTES="${WAIT_MINUTES:-120}"
POLL_SECONDS="${POLL_SECONDS:-15}"

WANDB_PROJECT="${WANDB_PROJECT:-medec-detect-gepa-tests}"

# Caches
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$HOME/.cache/sglang}"
mkdir -p "$HF_HOME" "$SGLANG_CACHE_DIR" logs "$OUTDIR"

# -------------------------------------------------------------------------
# GEPA run directories (28 programmes)
# Paths are relative to REPO_ROOT.
# Replace these with your own programme directories from Phase 1.
# -------------------------------------------------------------------------

PROGRAM_DIRS=(
  # Update these paths to point to your GEPA compilation output directories.
  # Each directory must contain program.json and summary.json.
  # Example:
  #   "results/gepa_grid/gepa_detect_openai-gpt-5_ref-openai-gpt-5_seed42_..."
  #   "results/gepa_grid/gepa_detect_local-qwen3-32b_ref-openai-gpt-5_seed42_..."
  #   ...
  "PLACEHOLDER_DIR_0"
  "PLACEHOLDER_DIR_1"
  "PLACEHOLDER_DIR_2"
  "PLACEHOLDER_DIR_3"
  "PLACEHOLDER_DIR_4"
  "PLACEHOLDER_DIR_5"
  "PLACEHOLDER_DIR_6"
  "PLACEHOLDER_DIR_7"
  "PLACEHOLDER_DIR_8"
  "PLACEHOLDER_DIR_9"
  "PLACEHOLDER_DIR_10"
  "PLACEHOLDER_DIR_11"
  "PLACEHOLDER_DIR_12"
  "PLACEHOLDER_DIR_13"
  "PLACEHOLDER_DIR_14"
  "PLACEHOLDER_DIR_15"
  "PLACEHOLDER_DIR_16"
  "PLACEHOLDER_DIR_17"
  "PLACEHOLDER_DIR_18"
  "PLACEHOLDER_DIR_19"
  "PLACEHOLDER_DIR_20"
  "PLACEHOLDER_DIR_21"
  "PLACEHOLDER_DIR_22"
  "PLACEHOLDER_DIR_23"
  "PLACEHOLDER_DIR_24"
  "PLACEHOLDER_DIR_25"
  "PLACEHOLDER_DIR_26"
  "PLACEHOLDER_DIR_27"
)

# -------------------------------------------------------------------------
# Preset helpers
# -------------------------------------------------------------------------

qwen_repo_for() {
  case "$1" in
    qwen3-32b)  echo "Qwen/Qwen3-32B" ;;
    qwen3-14b)  echo "Qwen/Qwen3-14B" ;;
    qwen3-8b)   echo "Qwen/Qwen3-8B" ;;
    qwen3-4b)   echo "Qwen/Qwen3-4B" ;;
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

IDX="${SLURM_ARRAY_TASK_ID}"
PROGRAM_DIR="${PROGRAM_DIRS[$IDX]}"
PROGRAM_PATH="$PROGRAM_DIR/program.json"
SUMMARY_PATH="$PROGRAM_DIR/summary.json"

if [ ! -d "$PROGRAM_DIR" ]; then
  echo "Programme directory not found: $PROGRAM_DIR" >&2
  exit 1
fi
if [ ! -f "$PROGRAM_PATH" ]; then
  echo "program.json not found in $PROGRAM_DIR" >&2
  exit 1
fi
if [ ! -f "$SUMMARY_PATH" ]; then
  echo "summary.json not found in $PROGRAM_DIR" >&2
  exit 1
fi

# -------------------------------------------------------------------------
# Environment initialisation
# -------------------------------------------------------------------------

if ! type module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh || true
fi
module load cuda || true

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then
  set +u
  . "$CONDA_SH"
  set -u
else
  echo "Warning: conda.sh not found at $CONDA_SH" >&2
fi
conda activate "$CONDA_ENV"

# Extract inference and reflector presets from summary.json
read INF_PRESET REF_PRESET <<< "$("$PYTHON_BIN" - "$SUMMARY_PATH" << 'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
data = json.loads(p.read_text())
print(data.get("preset", ""), data.get("reflector_preset", ""))
PY
)"

if [ -z "$INF_PRESET" ] || [ -z "$REF_PRESET" ]; then
  echo "Failed to extract presets from $SUMMARY_PATH" >&2
  echo "INF_PRESET='$INF_PRESET' REF_PRESET='$REF_PRESET'" >&2
  exit 1
fi

echo "Host: $(hostname)"
echo "Array index: $IDX"
echo "PROGRAM_DIR: $PROGRAM_DIR"
echo "PROGRAM_PATH: $PROGRAM_PATH"
echo "Inference preset: $INF_PRESET"
echo "Reflector preset: $REF_PRESET"
echo "VAL_CSV: $VAL_CSV"
echo "OUTDIR: $OUTDIR"

# -------------------------------------------------------------------------
# SGLang orchestration (mirrors GEPA grid script)
# -------------------------------------------------------------------------

REF_IS_LOCAL=0
INF_IS_LOCAL=0
is_qwen "$REF_PRESET" && REF_IS_LOCAL=1 || true
is_qwen "$INF_PRESET" && INF_IS_LOCAL=1 || true

LOCAL1_NAME=""
LOCAL2_NAME=""

if [ "$REF_IS_LOCAL" -eq 1 ] && [ "$INF_IS_LOCAL" -eq 1 ]; then
  if [ "$REF_PRESET" = "$INF_PRESET" ]; then
    LOCAL1_NAME="$REF_PRESET"
  else
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

  if [ -n "$ORIG_CUDA" ]; then
    export CUDA_VISIBLE_DEVICES="$ORIG_CUDA"
  else
    unset CUDA_VISIBLE_DEVICES || true
  fi
}
trap cleanup EXIT

PORT_BASE=$(( BASE_PORT + 4 * IDX ))
PORT1=$(( PORT_BASE + 0 ))
PORT2=$(( PORT_BASE + 1 ))

if [ -z "$LOCAL1_NAME" ] && [ -z "$LOCAL2_NAME" ]; then
  echo "No local Qwen models required for this job (GPT-5 only)."
  PORT_INF="$PORT1"
  PORT_REFLECTOR="$PORT1"

elif [ -n "$LOCAL1_NAME" ] && [ -z "$LOCAL2_NAME" ]; then
  echo "Single local Qwen model in this job: $LOCAL1_NAME"
  launch_sglang "$LOCAL1_NAME" "$PORT1" "0,1,2,3" "4"
  wait_for_ready "$PORT1" "$LOCAL1_NAME"

  PORT_INF="$PORT1"
  PORT_REFLECTOR="$PORT1"

elif [ -n "$LOCAL1_NAME" ] && [ -n "$LOCAL2_NAME" ]; then
  echo "Two local Qwen models in this job: $LOCAL1_NAME and $LOCAL2_NAME"

  launch_sglang "$LOCAL1_NAME" "$PORT1" "0,1" "2"
  wait_for_ready "$PORT1" "$LOCAL1_NAME"

  launch_sglang "$LOCAL2_NAME" "$PORT2" "2,3" "2"
  wait_for_ready "$PORT2" "$LOCAL2_NAME"

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

[ -z "$PORT_INF" ] && PORT_INF="$PORT1"
[ -z "$PORT_REFLECTOR" ] && PORT_REFLECTOR="$PORT_INF"

echo "Port mapping:"
echo "  Inference preset:  $INF_PRESET  -> PORT_INF=$PORT_INF"
echo "  Reflector preset:  $REF_PRESET  -> PORT_REFLECTOR=$PORT_REFLECTOR"

if [ -n "$ORIG_CUDA" ]; then
  export CUDA_VISIBLE_DEVICES="$ORIG_CUDA"
else
  unset CUDA_VISIBLE_DEVICES || true
fi

# -------------------------------------------------------------------------
# Run GEPA programme in inference mode (no optimisation)
# -------------------------------------------------------------------------

mkdir -p "$OUTDIR"

$RUNNER "$SCRIPT_PATH" \
  --preset "$INF_PRESET" \
  --reflector-preset "$REF_PRESET" \
  --port "$PORT_INF" \
  --reflector-port "$PORT_REFLECTOR" \
  --runs "$RUNS" \
  --seed "$SEED" \
  --train-csv "$TRAIN_CSV" \
  --val-csv "$VAL_CSV" \
  --output-dir "$OUTDIR" \
  --load-program "$PROGRAM_PATH" \
  --wandb \
  --wandb-project "$WANDB_PROJECT"

echo "Done: programme=$PROGRAM_DIR val_csv=$VAL_CSV on $(hostname)"
