#!/usr/bin/env bash
#SBATCH --job-name=medec-qwen3
#SBATCH --partition=gpu.A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# -------- user knobs (override via: sbatch --export=ALL,VAR=VAL …) --------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-dspy}"
PYTHON_BIN="${PYTHON_BIN:-python}"                 # used to launch SGLang
RUNNER="${RUNNER:-uv run python}"                  # used to run your evaluator
SCRIPT_PATH="${SCRIPT_PATH:-$REPO_ROOT/src/detect_eval.py}"
TP="${TP:-4}"                                      # tensor-parallel shards
BASE_PORT="${BASE_PORT:-7501}"                     # base; array index is added
WAIT_MINUTES="${WAIT_MINUTES:-120}"                # allow long model load
POLL_SECONDS="${POLL_SECONDS:-15}"

TRAIN_CSV="${TRAIN_CSV:-data/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv}"
VAL_CSV="${VAL_CSV:-data/MEDEC-UW/MEDEC-UW-ValidationSet-with-GroundTruth-and-ErrorType.csv}"
OUTDIR="${OUTDIR:-results/baseline}"
PROMPT="${PROMPT:-paper}"
RUNS="${RUNS:-3}"
SEED="${SEED:-42}"

# Optional caches (consider fast scratch if available)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$HOME/.cache/sglang}"
mkdir -p "$HF_HOME" "$SGLANG_CACHE_DIR" logs

# Models and their HF repo names (array index selects)
MODELS=(qwen3-32b qwen3-14b qwen3-8b qwen3-4b qwen3-1.7b qwen3-0.6b)
HF_REPOS=(Qwen/Qwen3-32B Qwen/Qwen3-14B Qwen/Qwen3-8B Qwen/Qwen3-4B Qwen/Qwen3-1.7B Qwen/Qwen3-0.6B)

IDX="${SLURM_ARRAY_TASK_ID}"
PRESET="${MODELS[$IDX]}"
HF_MODEL="${HF_REPOS[$IDX]}"
PORT=$(( BASE_PORT + IDX ))

echo "Host: $(hostname)"
echo "Preset: $PRESET"
echo "HF model: $HF_MODEL"
echo "TP: $TP  Port: $PORT"
echo "Runs: $RUNS  Prompt: $PROMPT"
echo "Output dir: $OUTDIR"
echo "Script: $SCRIPT_PATH"

# ---------- environment ----------
# Load 'module' if available, then CUDA (non-fatal if absent)
if ! type module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh || true
fi
module load cuda || true

# Clean Conda init (avoid ~/.bashrc and unbound vars)
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ -f "$CONDA_SH" ]; then
  set +u
  . "$CONDA_SH"
  set -u
else
  echo "Warning: conda.sh not found at $CONDA_SH" >&2
fi
conda activate "$CONDA_ENV"

# ---------- launch SGLang in background ----------
set +e
"$PYTHON_BIN" -m sglang.launch_server \
  --port "$PORT" \
  --model-path "$HF_MODEL" \
  --tp "$TP" \
  > "logs/sglang_${SLURM_JOB_ID}_${IDX}.log" 2>&1 &
SGLANG_PID=$!
set -e

cleanup() {
  echo "Stopping SGLang PID $SGLANG_PID"
  kill "$SGLANG_PID" 2>/dev/null || true
  sleep 2
  pkill -P "$SGLANG_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ---------- wait for readiness ----------
echo "Waiting up to ${WAIT_MINUTES} minutes for SGLang on port $PORT"
deadline=$(( $(date +%s) + WAIT_MINUTES*60 ))
until curl -sf "http://localhost:${PORT}/v1/models" | grep -q '"data"'; do
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "Timed out waiting for SGLang to load on port $PORT"
    exit 124
  fi
  if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
    echo "SGLang process exited while loading"
    exit 125
  fi
  sleep "$POLL_SECONDS"
done
echo "SGLang is ready on port $PORT"

# ---------- run evaluation ----------
$RUNNER "$SCRIPT_PATH" \
  --preset "$PRESET" \
  --port "$PORT" \
  --prompt "$PROMPT" \
  --runs "$RUNS" \
  --seed "$SEED" \
  --train-csv "$TRAIN_CSV" \
  --val-csv "$VAL_CSV" \
  --output-dir "$OUTDIR" \
  --wandb

echo "Done: $PRESET on $(hostname)"
