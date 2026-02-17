<h1 align="center">
  Importance of Prompt Optimisation for Error Detection in Medical Notes Using Language Models
</h1>

[Publication (coming soon)](#citation) | [MEDEC Dataset](https://github.com/abachaa/MEDEC) | [Citation](#citation)

## Summary

This repository provides reproducibility code for the paper *"Importance of Prompt Optimisation for Error Detection in Medical Notes Using Language Models"*.

Errors in medical text can cause delays or even result in incorrect treatment for patients. We explore the importance of prompt optimisation for small and large language models applied to the task of error detection in clinical notes, performing rigorous experiments across frontier models (GPT-5, Claude Sonnet 4.5, Gemini 2.5 Pro, Grok 4) and open-source models (Qwen3 0.6B-32B). We show that automatic prompt optimisation with Genetic-Pareto (GEPA) improves error detection accuracy from 0.669 to 0.785 with GPT-5 and from 0.578 to 0.690 with Qwen3-32B, approaching the performance of medical doctors and achieving state-of-the-art on the MEDEC benchmark.

---

<img src="https://github.com/user-attachments/assets/8ab36271-222c-4704-a70e-7832ada731f7" width="100%" align="center" />

---

## Overview

The pipeline consists of three phases:

| Phase | Script | Description |
|-------|--------|-------------|
| 1. Baseline | `src/detect_eval.py` | Single-pass inference with the paper prompt |
| 2. GEPA Compilation | `src/detect_gepa.py --auto heavy` | Compile optimised prompts on the validation set (produces `program.json`) |
| 3. GEPA Evaluation | `src/detect_gepa.py --load-program` | Evaluate compiled programmes on held-out test sets |

---

## Setup

```bash
uv sync   # or: pip install .
```

Set whichever API keys you need:

```bash
export OPENAI_API_KEY="..."        # GPT-5
export OPENROUTER_API_KEY="..."    # Claude, Gemini, Grok, DeepSeek via OpenRouter
export WANDB_API_KEY="..."         # Experiment tracking (optional)
```

Local models are served via [SGLang](https://github.com/sgl-project/sglang):

```bash
python -m sglang.launch_server --port 7501 --model-path Qwen/Qwen3-8B --tp 4
```

## Dataset

The [MEDEC dataset](https://github.com/abachaa/MEDEC) is from the MEDIQA-CORR 2024 shared task. We use the **original** dataset (not the [corrected version](https://github.com/abachaa/MEDEC/blame/70268d24e3ce0cd6d0e099ff7bfd4966f2bbcc28/README.md#L49)) to ensure direct comparability with previous benchmarks.

- **MEDEC-MS** — publicly available
- **MEDEC-UW** — requires a Data Use Agreement (see [dataset repo](https://github.com/abachaa/MEDEC) for details)

## Usage

**Baseline inference:**

```bash
# Local model
python src/detect_eval.py \
  --preset qwen3-8b \              # model preset (see Supported Models)
  --port 7501 \                    # SGLang server port
  --prompt paper \                 # prompt style matching the paper
  --runs 3 \                       # independent seeded repeats
  --seed 42 \                      # base random seed
  --val-csv data/MEDEC-MS-ValidationSet.csv \
  --output-dir results/baseline \
  --wandb                          # enable W&B logging (optional)

# API model
python src/detect_eval.py \
  --preset gpt-5 \                 # uses OpenAI API directly
  --prompt paper \
  --runs 3 \
  --seed 42 \
  --val-csv data/MEDEC-MS-ValidationSet.csv \
  --output-dir results/baseline \
  --wandb
```

**GEPA compilation** (Phase 2):

```bash
python src/detect_gepa.py \
  --preset qwen3-8b \              # inference model
  --reflector-preset qwen3-32b \   # reflector model used by GEPA optimiser
  --port 7501 \                    # SGLang port for inference model
  --reflector-port 7502 \          # SGLang port for reflector model
  --auto heavy \                   # GEPA budget (light/medium/heavy)
  --runs 1 \
  --seed 42 \
  --val-csv data/MEDEC-MS-ValidationSet.csv \
  --output-dir results/gepa_grid \
  --wandb
```

**GEPA test evaluation** (Phase 3):

```bash
python src/detect_gepa.py \
  --preset qwen3-8b \
  --reflector-preset qwen3-32b \
  --port 7501 \
  --reflector-port 7502 \
  --seed 42 \
  --val-csv data/MEDEC-MS-TestSet.csv \
  --output-dir results/gepa_test \
  --load-program results/gepa_grid/.../program.json \  # compiled programme from Phase 2
  --wandb
```

## Supported Models

| Preset | Provider | Preset | Provider |
|--------|----------|--------|----------|
| `qwen3-0.6b` | Local (SGLang) | `gpt-5` | OpenAI |
| `qwen3-1.7b` | Local (SGLang) | `claude-sonnet-4.5` | OpenRouter |
| `qwen3-4b` | Local (SGLang) | `gemini-2.5-pro` | OpenRouter |
| `qwen3-8b` | Local (SGLang) | `grok-4` | OpenRouter |
| `qwen3-14b` | Local (SGLang) | `deepseek-r1` | OpenRouter |
| `qwen3-32b` | Local (SGLang) | | |

## SLURM Scripts

We have additionally provided the SLURM scripts which may be of interest to those trying to reproduce this on HPC environments. The `slurm/` directory contains:

- **`run_medec_qwen3_array.sbatch`** -- Baseline across all 6 Qwen3 sizes (array 0--5)
- **`run_medec_gepa_grid.sbatch`** -- Full 28-job reflector x inference grid with intelligent GPU splitting (1 model: TP=4 on 4 GPUs; 2 models: TP=2 each on 2+2 GPUs)
- **`run_medec_gepa_test_inference.sbatch`** -- Load compiled programmes and evaluate on test sets

All scripts auto-launch SGLang, poll for readiness, and clean up on exit. Override defaults via environment variables:

```bash
sbatch --export=ALL,REPO_ROOT=/path/to/repo slurm/run_medec_gepa_grid.sbatch
```

## Citation

Accepted at [HeaLing](https://healing-workshop.github.io/) @ [EACL 2026](https://2026.eacl.org/).

```bibtex
Coming soon.
```
