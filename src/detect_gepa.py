"""GEPA optimisation of the MEDIQA-CORR detection prompt (paper baseline)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import random
import statistics as stats
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Shared helpers


def split_sentences_blob(blob: str) -> List[str]:
    if not isinstance(blob, str):
        return []
    return [ln.strip("\r") for ln in blob.split("\n") if ln.strip() != ""]


def normalize_error_type(raw: Any) -> str:
    if raw is None:
        return "none"
    text = str(raw).strip().lower()
    if text == "":
        return "none"
    mapping = {
        "none": "none",
        "no error": "none",
        "diagnosis": "diagnosis",
        "management": "management",
        "treatment": "treatment",
        "pharmacotherapy": "pharmacotherapy",
        "pharmacology": "pharmacotherapy",
        "causal organism": "causalorganism",
        "causalorganism": "causalorganism",
        "causal-organism": "causalorganism",
    }
    return mapping.get(text, text)


def humanize_error_type(code: str) -> str:
    mapping = {
        "none": "none",
        "diagnosis": "diagnosis",
        "management": "management",
        "treatment": "treatment",
        "pharmacotherapy": "pharmacotherapy",
        "causalorganism": "causal organism",
    }
    return mapping.get(code, code)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass


def short_id(n: int = 5) -> str:
    import secrets
    import string

    alphabet = string.ascii_lowercase + string.digits
    length = max(int(n), 1)
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _lower(x: Any) -> str:
    return (str(x) if x is not None else "").strip().lower()


def _split_sentence_idx_and_text(raw: str, fallback_idx: int) -> Tuple[str, str]:
    stripped = (raw or "").strip()
    if stripped == "":
        return str(fallback_idx), ""

    if "|" in stripped:
        left, right = stripped.split("|", 1)
        idx = left.strip() or str(fallback_idx)
        text = right.strip(" :-")
        return idx, text.strip()

    parts = stripped.split(" ", 1)
    if parts and parts[0].isdigit():
        idx = parts[0]
        text = parts[1].strip() if len(parts) > 1 else ""
        return idx, text

    return str(fallback_idx), stripped


def format_sentences_for_paper_prompt(sentences: List[str]) -> str:
    formatted: List[str] = []
    for idx, raw in enumerate(sentences):
        sent_idx, sent_text = _split_sentence_idx_and_text(raw, idx)
        formatted.append(f"{sent_idx}| {sent_text}")
    return "\n".join(formatted)


def parse_paper_prompt_output(raw: str) -> Tuple[str, Optional[int], str]:
    text = (raw or "").strip()
    if text == "":
        return "correct", None, ""

    upper = text.upper()
    if upper.startswith("CORRECT"):
        return "correct", None, text

    first_token, remainder = text, ""
    if " " in text:
        first_token, remainder = text.split(" ", 1)
    token_clean = first_token.rstrip(":").rstrip("-")

    if token_clean.isdigit():
        try:
            sent_id = int(token_clean)
        except ValueError:
            sent_id = None
        return "error", sent_id, remainder.strip()

    return "error", None, text


# ---------------------------------------------------------------------------
# DSPy programme (paper baseline)

PAPER_PROMPT_TEXT = (
    "The following is a medical narrative about a patient. You are a skilled medical "
    "doctor reviewing the clinical text. The text is either correct or contains one "
    "error. The text has one sentence per line. Each line starts with the sentence ID, "
    "followed by a pipe character then the sentence to check. Check every sentence of "
    "the text. If the text is correct return the following output: CORRECT. If the text "
    "has a medical error related to treatment, management, cause, or diagnosis, return "
    "the sentence id of the sentence containing the error, followed by a space, and "
    "then a corrected version of the sentence. Finding and correcting the error "
    "requires medical knowledge and reasoning."
)

try:
    import dspy  # type: ignore
    from typing import Literal  # noqa: WPS300 (re-export for typing)

    class PaperPromptSignature(dspy.Signature):
        __doc__ = PAPER_PROMPT_TEXT

        input: str = dspy.InputField()
        output: str = dspy.OutputField()

    try:
        PaperPromptSignature.model_rebuild()
    except Exception:
        pass

    class PaperPromptProgramme(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.detect = dspy.Predict(PaperPromptSignature)

        def forward(self, sentences: List[str]) -> dspy.Prediction:
            formatted = format_sentences_for_paper_prompt(sentences)
            completion = self.detect(input=formatted)
            raw_response = str(getattr(completion, "output", "")).strip()

            verdict, parsed_sentence_id, parsed_correction = parse_paper_prompt_output(
                raw_response,
            )
            normalized_verdict = "error" if verdict.lower() == "error" else "correct"

            sentence_id_str: Optional[str] = None
            if parsed_sentence_id is not None:
                sentence_id_str = str(parsed_sentence_id)

            return dspy.Prediction(
                verdict=normalized_verdict,
                raw_response=raw_response,
                predicted_sentence_id=sentence_id_str,
                corrected_sentence=parsed_correction,
            )

except ModuleNotFoundError as exc:  # pragma: no cover - validated at runtime
    dspy = None  # type: ignore
    Literal = None  # type: ignore  # noqa: N816
    DSPY_IMPORT_ERROR = exc
else:
    DSPY_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Model presets

PRESET_MAP: Dict[str, Tuple[str, str]] = {
    "qwen3-32b": ("local", "qwen3-32b"),
    "qwen3-14b": ("local", "qwen3-14b"),
    "qwen3-8b": ("local", "qwen3-8b"),
    "qwen3-4b": ("local", "qwen3-4b"),
    "qwen3-1.7b": ("local", "qwen3-1.7b"),
    "qwen3-0.6b": ("local", "qwen3-0.6b"),
    "gpt-5": ("openai", "gpt-5"),
    "gpt5": ("openai", "gpt-5"),
    "claude-sonnet-4.5": ("openrouter", "anthropic/claude-3.5-sonnet"),
    "gemini-2.5-pro": ("openrouter", "google/gemini-2.5-pro-exp"),
    "grok-4": ("openrouter", "x-ai/grok-2-1212"),
    "deepseek-r1": ("openrouter", "deepseek/deepseek-r1"),
}

PRESET_CHOICES: List[str] = sorted(PRESET_MAP.keys())

def _model_default_decoding(provider: str, model: str) -> Dict[str, Optional[float]]:
    """Return model/provider default decoding settings."""

    lowered = model.lower()
    if provider == "local" and lowered.startswith("qwen3"):
        return {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": 32768,
        }
    if provider == "openai" and lowered.startswith("gpt-5"):
        return {
            "temperature": 1.0, #Important: The following parameters are not supported when using GPT-5 models
            "top_p": None, #Important: The following parameters are not supported when using GPT-5 models
            "max_tokens": 32768,
        }
    if provider == "openrouter" and "claude" in lowered:
        return {
            "temperature": 1.0, # Hard to find a recommended setting, using 1.0 as it was used for their AIME benchmark.
            "max_tokens": 32768,
        }
    if provider == "openrouter" and "gemini" in lowered:
        return {
            "temperature": 1.0, # default based on https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro
            "top_p": 0.95, # default based on https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-pro althrough they refer to it as topP
            "max_tokens": 32768,
        }
    if provider == "openrouter" and "grok" in lowered:
        return {
            "temperature": 1.0, # default based on https://docs.x.ai/docs/api-reference#chat-completions
            "top_p": 1.0,
            "max_tokens": 32768,
        }
    if provider == "openrouter" and "deepseek" in lowered:
        return {
            "temperature": 1.0, # based on https://api-docs.deepseek.com/quick_start/parameter_settings. It suggests various optoins but 1.0 for data analysis, 0.0 for coding/math.
            "top_p": 1.0, # based on https://api-docs.deepseek.com/api/create-chat-completion
            "max_tokens": 32768,
        }
    return {}


def build_lm_from_preset(
    preset: str,
    *,
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
    top_k: Optional[int],
    min_p: Optional[float],
    seed: Optional[int],
    port: int,
    configure: bool,
) -> Tuple[Any, str, Dict[str, Any]]:  # type: ignore[valid-type]
    if dspy is None:
        raise SystemExit("DSPy is required for this script. Install via `pip install dspy`.")
    if preset not in PRESET_MAP:
        raise SystemExit(f"Unknown preset: {preset}")

    provider, model = PRESET_MAP[preset]

    defaults = _model_default_decoding(provider, model)
    lm_kwargs: Dict[str, Any] = {k: v for k, v in defaults.items() if v is not None}
    if temperature is not None:
        lm_kwargs["temperature"] = temperature
    if top_p is not None:
        lm_kwargs["top_p"] = top_p
    if top_k is not None:
        lm_kwargs["top_k"] = top_k
    if min_p is not None:
        lm_kwargs["min_p"] = min_p
    if max_tokens is not None:
        lm_kwargs["max_tokens"] = max_tokens
    if seed is not None:
        lm_kwargs["seed"] = int(seed)

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set (env/.env)")
        for unsupported in ("top_k", "min_p"):
            lm_kwargs.pop(unsupported, None)
        lm = dspy.LM(f"openai/{model}", **lm_kwargs)
        run_name = f"openai-{model}"
    elif provider == "local":
        api_base = f"http://localhost:{int(port)}/v1"
        lm = dspy.LM(
            model=f"openai/{model}",
            api_base=api_base,
            api_key="local",
            model_type="chat",
            **lm_kwargs,
        )
        run_name = f"local-{model.replace('/', '-')}"
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY is not set (env/.env)")
        lm = dspy.LM(
            model=f"openrouter/{model}",
            api_base="https://openrouter.ai/api/v1",
            api_key=api_key,
            **lm_kwargs,
        )
        run_name = f"openrouter-{model.replace('/', '-')}"

    if configure:
        dspy.configure(lm=lm, cache=False)

    lm_info = {
        "provider": provider,
        "model": model,
        "temperature": lm_kwargs.get("temperature"),
        "top_p": lm_kwargs.get("top_p"),
        "top_k": lm_kwargs.get("top_k"),
        "min_p": lm_kwargs.get("min_p"),
        "max_tokens": lm_kwargs.get("max_tokens"),
        "seed": lm_kwargs.get("seed"),
        "port": int(port) if provider == "local" else None,
    }
    return lm, run_name, lm_info


# ---------------------------------------------------------------------------
# Plot helpers


def _plot_binary_confusion(
    y_true: List[int],
    y_pred: List[int],
    labels: List[int],
    title: str,
    path: Path,
    tick_text: Optional[List[str]] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except Exception as exc:  # pragma: no cover - optional dependency guard
        print(f"Unable to plot confusion matrix (missing dependency): {exc}")
        return

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = cm.sum(axis=1, keepdims=True).clip(min=1)
    pct = cm / row_totals
    annot = [
        [f"{count}\n{frac:.1%}" for count, frac in zip(row_counts, row_pct)]
        for row_counts, row_pct in zip(cm, pct)
    ]

    fig, ax = plt.subplots(figsize=(4, 4))
    display_labels = tick_text or [str(lab) for lab in labels]
    sns.heatmap(
        pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        cbar=True,
        linewidths=0.5,
        square=True,
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    try:
        fig.savefig(path, dpi=200)
        print(f"Saved confusion matrix → {path}")
    except Exception as exc:  # pragma: no cover - filesystem errors
        print(f"Failed to save confusion matrix ({exc})")
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset + evaluation


def to_examples(df: pd.DataFrame) -> List["dspy.Example"]:
    if dspy is None:
        raise SystemExit("DSPy is required for this script. Install via `pip install dspy`.")

    examples: List[dspy.Example] = []
    for _, row in df.iterrows():
        sentences = split_sentences_blob(row.get("Sentences", ""))
        flag = int(row.get("Error Flag", 0) or 0)
        verdict = "error" if flag == 1 else "correct"
        err_type = normalize_error_type(row.get("Error Type", "none"))
        if verdict == "correct":
            err_type = "none"

        raw_sentence_id = row.get("Error Sentence ID", -1)
        try:
            err_sentence_id = int(raw_sentence_id)
        except (TypeError, ValueError):
            err_sentence_id = -1
        if verdict == "correct":
            err_sentence_id = -1

        reference_correction = str(row.get("Corrected Sentence", "NA"))
        reference_corrected_text = str(row.get("Corrected Text", ""))

        example = dspy.Example(
            text_id=str(row.get("Text ID", "")),
            sentences=sentences,
            verdict=verdict,
            error_type=err_type,
            error_sentence=str(row.get("Error Sentence", "NA")),
            error_sentence_id=err_sentence_id,
            reference_corrected_sentence=reference_correction,
            reference_corrected_text=reference_corrected_text,
        ).with_inputs("sentences")
        examples.append(example)
    return examples


def evaluate_programme(
    programme: Any,  # type: ignore[valid-type]
    examples: List["dspy.Example"],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    y_true_flags: List[int] = []
    y_pred_flags: List[int] = []

    for ex in examples:
        prediction = programme(sentences=ex.sentences)
        verdict = _lower(getattr(prediction, "verdict", ""))
        if verdict not in {"error", "correct"}:
            verdict = "correct"
        pred_flag = 1 if verdict == "error" else 0

        gt_flag = 1 if _lower(getattr(ex, "verdict", "correct")) == "error" else 0
        error_type = normalize_error_type(getattr(ex, "error_type", "none"))

        gt_sentence_id = int(getattr(ex, "error_sentence_id", -1))
        if gt_flag == 0:
            gt_sentence_id = -1

        raw_pred_sentence = getattr(prediction, "predicted_sentence_id", None)
        pred_sentence_id = -1
        if pred_flag == 1:
            try:
                pred_sentence_id = int(str(raw_pred_sentence).strip())
            except (TypeError, ValueError):
                pred_sentence_id = -1
        else:
            pred_sentence_id = -1

        sentence_detect_correct = 1.0 if pred_sentence_id == gt_sentence_id else 0.0
        sentence_detect_correct_strict = float(
            (gt_flag == 0 and pred_flag == 0 and pred_sentence_id == -1)
            or (gt_flag == 1 and pred_flag == 1 and pred_sentence_id == gt_sentence_id)
        )

        reference_corrected_sentence = getattr(ex, "reference_corrected_sentence", "NA")
        reference_corrected_text = getattr(ex, "reference_corrected_text", "")
        sentences_raw = "\n".join(ex.sentences)
        sentences_formatted = format_sentences_for_paper_prompt(ex.sentences)

        rows.append(
            {
                "text_id": getattr(ex, "text_id", ""),
                "pred_verdict": verdict,
                "pred_error_flag": pred_flag,
                "gt_error_flag": gt_flag,
                "error_type": error_type,
                "detect_correct": 1.0 if pred_flag == gt_flag else 0.0,
                "gt_sentence_id": gt_sentence_id,
                "pred_sentence_id": pred_sentence_id,
                "sentence_detect_correct": sentence_detect_correct,
                "sentence_detect_correct_strict": sentence_detect_correct_strict,
                "raw_response": getattr(prediction, "raw_response", None),
                "predicted_corrected_sentence": getattr(prediction, "corrected_sentence", None),
                "reference_corrected_sentence": reference_corrected_sentence,
                "reference_corrected_text": reference_corrected_text,
                "sentences_raw": sentences_raw,
                "sentences_formatted": sentences_formatted,
            }
        )

        y_true_flags.append(gt_flag)
        y_pred_flags.append(pred_flag)

    summary_df = pd.DataFrame(rows)

    error_flag_accuracy = float("nan")
    error_sentence_accuracy = float("nan")
    error_sentence_accuracy_strict = float("nan")
    detection_by_type_df = pd.DataFrame()
    detection_by_type_json: Dict[str, Dict[str, Any]] = {}

    if not summary_df.empty:
        error_flag_accuracy = float(summary_df["detect_correct"].mean())
        error_sentence_accuracy = float(summary_df["sentence_detect_correct"].mean())
        error_sentence_accuracy_strict = float(summary_df["sentence_detect_correct_strict"].mean())

        err_only = summary_df[(summary_df["gt_error_flag"] == 1) & (summary_df["error_type"] != "none")]
        if not err_only.empty:
            flag_recall_series = err_only.groupby("error_type")["pred_error_flag"].mean().rename("flag_recall")
            sentence_recall_series = (
                err_only.groupby("error_type")["sentence_detect_correct"].mean().rename("sentence_recall")
            )
            count_series = err_only.groupby("error_type")[["sentence_detect_correct"]].count().rename(
                columns={"sentence_detect_correct": "count"}
            )
            detection_by_type_df = pd.concat(
                [flag_recall_series, sentence_recall_series, count_series], axis=1
            ).sort_index()
            detection_by_type_df["flag_recall"] = detection_by_type_df["flag_recall"].astype(float)
            detection_by_type_df["sentence_recall"] = detection_by_type_df["sentence_recall"].astype(float)
            detection_by_type_df["count"] = detection_by_type_df["count"].astype(int)
            detection_by_type_df.index.name = "error_type"
            detection_by_type_json = {
                err: {
                    "label": humanize_error_type(err),
                    "flag_recall": float(row["flag_recall"]),
                    "sentence_recall": float(row["sentence_recall"]),
                    "count": int(row["count"]),
                }
                for err, row in detection_by_type_df.iterrows()
            }

    tp = sum(1 for yt, yp in zip(y_true_flags, y_pred_flags) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true_flags, y_pred_flags) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true_flags, y_pred_flags) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true_flags, y_pred_flags) if yt == 1 and yp == 0)

    def _safe_div(num: float, den: float) -> float:
        return num / den if den else float("nan")

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)

    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)

    if math.isnan(recall) or math.isnan(specificity):
        balanced_accuracy = float("nan")
    else:
        balanced_accuracy = (recall + specificity) / 2

    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if mcc_den:
        mcc = ((tp * tn) - (fp * fn)) / mcc_den
    else:
        mcc = float("nan")

    binary_metrics = {
        "accuracy": error_flag_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "fpr": fpr,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
    }

    confusion_counts = {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

    return {
        "rows": rows,
        "summary_df": summary_df,
        "error_flag_accuracy": error_flag_accuracy,
        "error_sentence_accuracy": error_sentence_accuracy,
        "error_sentence_accuracy_strict": error_sentence_accuracy_strict,
        "binary_metrics": binary_metrics,
        "confusion_counts": confusion_counts,
        "detection_by_type_df": detection_by_type_df,
        "detection_by_type": detection_by_type_json,
        "y_true_flags": y_true_flags,
        "y_pred_flags": y_pred_flags,
    }


def _clean_value(val: Any) -> Any:
    if isinstance(val, float) and math.isnan(val):
        return None
    return val


def save_result_outputs(
    run_dir: Path,
    split_name: str,
    variant: str,
    result: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    prefix = f"{variant}_{split_name}"
    predictions_path = run_dir / f"{prefix}_predictions.csv"
    pd.DataFrame(result["rows"]).to_csv(predictions_path, index=False)

    metrics_path = run_dir / f"{prefix}_metrics_per_example.csv"
    result["summary_df"].to_csv(metrics_path, index=False)

    confusion_png_path = run_dir / f"{prefix}_confusion.png"
    confusion_pdf_path = run_dir / f"{prefix}_confusion.pdf"
    y_true = result.get("y_true_flags", [])
    y_pred = result.get("y_pred_flags", [])
    if y_true and y_pred:
        title = f"{variant} {split_name}"
        _plot_binary_confusion(
            y_true,
            y_pred,
            [0, 1],
            title,
            confusion_png_path,
            tick_text=["0 = correct", "1 = error"],
        )
        _plot_binary_confusion(
            y_true,
            y_pred,
            [0, 1],
            title,
            confusion_pdf_path,
            tick_text=["0 = correct", "1 = error"],
        )
        confusion_png_str = str(confusion_png_path)
        confusion_pdf_str = str(confusion_pdf_path)
    else:
        confusion_png_str = None
        confusion_pdf_str = None

    overall_metrics_payload = {
        "error_flag_accuracy": _clean_value(result.get("error_flag_accuracy")),
        "error_sentence_accuracy": _clean_value(result.get("error_sentence_accuracy")),
        "error_sentence_accuracy_strict": _clean_value(result.get("error_sentence_accuracy_strict")),
        "binary_metrics": {
            key: _clean_value(val)
            for key, val in (result.get("binary_metrics") or {}).items()
        },
        "confusion_counts": result.get("confusion_counts"),
        "totals": {
            "n": int(len(result.get("y_true_flags", []))),
            "n_errors": int(sum(result.get("y_true_flags", []))),
            "n_correct": int(len(result.get("y_true_flags", [])) - sum(result.get("y_true_flags", []))),
        },
    }
    metrics_json_path = run_dir / f"{prefix}_metrics_overall.json"
    metrics_json_path.write_text(json.dumps(overall_metrics_payload, indent=2) + "\n", encoding="utf-8")

    by_type_df = result.get("detection_by_type_df")
    by_type_csv_path: Optional[Path] = None
    by_type_json_path: Optional[Path] = None
    if isinstance(by_type_df, pd.DataFrame) and not by_type_df.empty:
        by_type_csv_path = run_dir / f"{prefix}_metrics_by_error_type.csv"
        by_type_df.to_csv(by_type_csv_path, index=True)
        by_type_json_path = run_dir / f"{prefix}_metrics_by_error_type.json"
        by_type_json_path.write_text(
            json.dumps(result.get("detection_by_type", {}), indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "predictions_csv": str(predictions_path),
        "metrics_csv": str(metrics_path),
        "metrics_json": str(metrics_json_path),
        "metrics_by_type_csv": str(by_type_csv_path) if by_type_csv_path else None,
        "metrics_by_type_json": str(by_type_json_path) if by_type_json_path else None,
        "confusion_png": confusion_png_str,
        "confusion_pdf": confusion_pdf_str,
    }


# ---------------------------------------------------------------------------
# GEPA metric (detection only)


def gepa_detection_metric(gt, pred, trace=None, pred_name=None, pred_trace=None):  # type: ignore[no-untyped-def]
    if dspy is None:
        raise SystemExit("DSPy is required for this script. Install via `pip install dspy`.")

    gt_verdict = _lower(getattr(gt, "verdict", ""))
    pred_verdict = _lower(getattr(pred, "verdict", ""))
    err_type_code = normalize_error_type(getattr(gt, "error_type", "none"))
    err_label = humanize_error_type(err_type_code)
    err_sentence = (getattr(gt, "error_sentence", "") or "").strip()
    ref_sentence = (getattr(gt, "reference_corrected_sentence", "") or "").strip()
    ref_text = (getattr(gt, "reference_corrected_text", "") or "").strip()

    if pred_verdict not in {"error", "correct"}:
        fb = "Format error: respond with 'CORRECT' or '<sent_id> <correction>'."
        return dspy.Prediction(score=0.0, feedback=fb)

    detail_bits: List[str] = []
    if err_type_code != "none":
        detail_bits.append(f"Error type: {err_label}.")
    else:
        detail_bits.append("No medical error in this case.")
    if err_sentence and err_sentence.upper() != "NA":
        detail_bits.append(f"Erroneous sentence: \"{err_sentence}\".")
    if ref_sentence and ref_sentence.upper() != "NA":
        detail_bits.append(f"Reference correction: \"{ref_sentence}\".")
    if ref_text and ref_text.upper() != "NA":
        detail_bits.append(f"Corrected text: \"{ref_text}\".")
    detail_str = " ".join(detail_bits).strip()

    if pred_verdict == gt_verdict:
        prefix = "True positive" if gt_verdict == "error" else "True negative"
        fb = f"{prefix}: correct prediction. {detail_str}".strip()
        return dspy.Prediction(score=1.0, feedback=fb)

    prefix = "False negative" if gt_verdict == "error" else "False positive"
    fb = (
        f"{prefix}: predicted {pred_verdict.upper()} while true label is {gt_verdict.upper()}. "
        f"{detail_str}"
    ).strip()
    return dspy.Prediction(score=0.0, feedback=fb)


# ---------------------------------------------------------------------------
# GEPA driver


def stratified_first_k(examples: List["dspy.Example"], k: int) -> List["dspy.Example"]:
    if not k or k <= 0 or k >= len(examples):
        return examples
    pos = [e for e in examples if _lower(getattr(e, "verdict", "correct")) == "error"]
    neg = [e for e in examples if _lower(getattr(e, "verdict", "correct")) != "error"]
    total = max(len(examples), 1)
    k_pos = int(round(k * (len(pos) / total)))
    k_pos = min(k_pos, len(pos))
    k_neg = min(k - k_pos, len(neg))
    selected = pos[:k_pos] + neg[:k_neg]
    if len(selected) < k:
        remainder = pos[k_pos:] + neg[k_neg:]
        selected += remainder[: (k - len(selected))]
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default="data/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv")
    parser.add_argument(
        "--val-csv",
        default="data/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv",
    )
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--gepa-train-k", type=int, default=0, help="Optional stratified subset for GEPA train")
    parser.add_argument(
        "--gepa-pareto-k",
        type=int,
        default=0,
        help="Optional stratified subset for GEPA Pareto tracking (0 = full val set).",
    )
    parser.add_argument("--preset", choices=PRESET_CHOICES, default="qwen3-8b")
    parser.add_argument(
        "--reflector-preset",
        choices=PRESET_CHOICES,
        default=None,
        help="Reflection LM preset (defaults to --preset).",
    )
    parser.add_argument(
        "--reflector-port",
        type=int,
        default=None,
        help="Optional port for the reflector LM when using local models (defaults to --port).",
    )
    parser.add_argument("--runs", type=int, default=1, help="Evaluation repeats after GEPA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, dest="top_p", default=None)
    parser.add_argument("--top-k", type=int, dest="top_k", default=None)
    parser.add_argument("--min-p", type=float, dest="min_p", default=None)
    parser.add_argument("--max-tokens", type=int, dest="max_tokens", default=None)
    parser.add_argument("--port", type=int, default=7501)
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="medium",
        help="GEPA auto budget preset (ignored if --max-metric-calls or --max-full-evals set)",
    )
    parser.add_argument("--max-metric-calls", type=int, default=None)
    parser.add_argument("--max-full-evals", type=float, default=None)
    parser.add_argument("--output-dir", default="results/gepa")
    parser.add_argument("--save-program", default=None)
    parser.add_argument("--load-program", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="medec-detect-gepa-p1")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.reflector_port is None:
        args.reflector_port = args.port

    if DSPY_IMPORT_ERROR is not None:
        raise SystemExit(
            "DSPy is required for this script but could not be imported: "
            f"{DSPY_IMPORT_ERROR}",
        )

    set_seed(args.seed)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df_train = (
        pd.read_csv(args.train_csv)
        .dropna(how="all")
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )
    df_val = (
        pd.read_csv(args.val_csv)
        .dropna(how="all")
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )

    if args.limit_train and args.limit_train > 0:
        df_train = df_train.head(args.limit_train)
    if args.limit_val and args.limit_val > 0:
        df_val = df_val.head(args.limit_val)

    trainset_full = to_examples(df_train)
    valset_full = to_examples(df_val)

    trainset = stratified_first_k(trainset_full, int(args.gepa_train_k)) if args.gepa_train_k else trainset_full
    pareto_valset = stratified_first_k(valset_full, int(args.gepa_pareto_k)) if args.gepa_pareto_k else valset_full

    if not trainset:
        raise SystemExit("GEPA train set is empty; adjust --limit-train or --gepa-train-k.")
    if not pareto_valset:
        raise SystemExit("GEPA pareto set is empty; adjust --limit-val or --gepa-pareto-k.")

    lm, run_name, lm_info = build_lm_from_preset(
        args.preset,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.seed,
        port=args.port,
        configure=True,
    )

    reflector_preset = args.reflector_preset or args.preset
    reflector_lm, reflector_run_name, reflector_info = build_lm_from_preset(
        reflector_preset,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.seed,
        port=args.reflector_port,
        configure=False,
    )

    ts = dt.datetime.now().strftime("%Y_%m_%d-%H%M")
    suffix = short_id()
    run_dir = out_root / (
        f"gepa_detect_{run_name}_ref-{reflector_run_name}_seed{args.seed}_{ts}_{suffix}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_module = None
    wandb_run = None
    if args.wandb:
        try:
            import wandb  # type: ignore

            wandb_module = wandb
            wandb_run_name = args.wandb_run_name or f"{run_name}-gepa-{ts}-{suffix}"
            wandb_config = {
                "preset": args.preset,
                "reflector": reflector_preset,
                "auto": args.auto,
                "seed": int(args.seed),
                "gepa_train": len(trainset),
                "gepa_pareto": len(pareto_valset),
                "lm": lm_info,
                "reflector_lm": reflector_info,
            }
            wandb_run = wandb_module.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=wandb_run_name,
                config=wandb_config,
            )
        except Exception as exc:
            print(f"W&B disabled: {exc}")
            wandb_module = None
            wandb_run = None

    print("\nRun configuration")
    print(f"  Inference preset: {args.preset}")
    print(f"  Reflector preset: {reflector_preset}")
    print(f"  Train examples (full/GEPA): {len(trainset_full)}/{len(trainset)}")
    print(f"  Val examples (full/Pareto): {len(valset_full)}/{len(pareto_valset)}")
    print(
        "  GEPA budgets → auto={auto} max_metric_calls={mmc} max_full_evals={mfe}".format(
            auto=args.auto if args.max_metric_calls is None and args.max_full_evals is None else None,
            mmc=args.max_metric_calls,
            mfe=args.max_full_evals,
        )
    )
    print(f"  Output directory: {run_dir}\n")

    programme = PaperPromptProgramme()

    gepa_budget_metric_calls: Optional[int] = None
    gepa_budget_full_evals: Optional[float] = None
    gepa_actual_metric_calls: Optional[int] = None
    gepa_actual_full_evals: Optional[float] = None
    gepa_actual_full_val_evals: Optional[int] = None
    gepa_log_dir: Optional[str] = None

    baseline_result: Dict[str, Any] = {}
    baseline_paths: Dict[str, Optional[str]] = {}

    def run_baseline_evaluation() -> None:
        nonlocal baseline_result, baseline_paths
        baseline_result = evaluate_programme(programme, valset_full)
        baseline_paths = save_result_outputs(run_dir, "val", "baseline", baseline_result)
        if wandb_run is not None and wandb_module is not None:
            ef_acc = baseline_result.get("error_flag_accuracy")
            if ef_acc is not None and not math.isnan(ef_acc):
                wandb_module.log({"val/baseline_error_flag_accuracy": ef_acc})

    optimised_program = None

    if args.load_program:
        print(f"Loading compiled programme state from {args.load_program}")
        optimised_program = PaperPromptProgramme()
        try:
            optimised_program.load(args.load_program)
            print(f"Loaded compiled programme state ← {args.load_program}")
        except Exception as exc:
            raise SystemExit(f"Failed to load programme from {args.load_program}: {exc}")
    else:
        run_baseline_evaluation()

        try:
            from dspy import GEPA  # type: ignore

            if args.auto and (args.max_metric_calls is not None or args.max_full_evals is not None):
                raise SystemExit("Specify only one of --auto, --max-metric-calls, or --max-full-evals.")

            gepa_kwargs: Dict[str, Any] = {
                "metric": gepa_detection_metric,
                "track_stats": True,
                "track_best_outputs": True,
                "add_format_failure_as_feedback": True,
                "reflection_lm": reflector_lm,
                "use_wandb": args.wandb,
                "seed": args.seed,
            }
            if args.max_metric_calls is not None:
                gepa_kwargs["max_metric_calls"] = int(args.max_metric_calls)
            elif args.max_full_evals is not None:
                gepa_kwargs["max_full_evals"] = float(args.max_full_evals)
            else:
                gepa_kwargs["auto"] = args.auto

            gepa = GEPA(**gepa_kwargs)
            optimised_program = gepa.compile(programme, trainset=trainset, valset=pareto_valset)

            try:
                budget_val = getattr(gepa, "max_metric_calls", None)
                if isinstance(budget_val, (int, float)):
                    gepa_budget_metric_calls = int(budget_val)
                    denom = len(trainset) + len(pareto_valset)
                    if denom > 0:
                        gepa_budget_full_evals = float(gepa_budget_metric_calls / denom)
            except Exception:
                pass

            try:
                detailed = getattr(optimised_program, "detailed_results", None)
                if detailed is not None:
                    tm_calls = getattr(detailed, "total_metric_calls", None)
                    if isinstance(tm_calls, (int, float)):
                        gepa_actual_metric_calls = int(tm_calls)
                        denom = len(trainset) + len(pareto_valset)
                        if denom > 0:
                            gepa_actual_full_evals = float(gepa_actual_metric_calls / denom)
                    num_full_val = getattr(detailed, "num_full_val_evals", None)
                    if isinstance(num_full_val, (int, float)):
                        gepa_actual_full_val_evals = int(num_full_val)
                    log_dir = getattr(detailed, "log_dir", None)
                    if isinstance(log_dir, str):
                        gepa_log_dir = log_dir
            except Exception:
                pass

            save_path = Path(args.save_program) if args.save_program else (run_dir / "program.json")
            try:
                if hasattr(optimised_program, "save"):
                    optimised_program.save(str(save_path))
                    print(f"Saved compiled programme → {save_path}")
            except Exception as exc:
                print(f"Warning: failed to save programme ({exc})")

        except Exception as exc:
            raise SystemExit(f"GEPA optimisation failed: {exc}") from exc

    if optimised_program is None:
        raise SystemExit("No optimised programme is available for evaluation.")

    seeds = [args.seed + i for i in range(args.runs)]
    per_run_results: List[Dict[str, Any]] = []
    debug_rows: List[Dict[str, Any]] = []
    optimised_paths: Dict[str, Optional[str]] = {}

    for idx, run_seed in enumerate(seeds, 1):
        print(f"=== Evaluation repeat {idx}/{args.runs} seed={run_seed} ===")
        set_seed(run_seed)
        build_lm_from_preset(
            args.preset,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            seed=run_seed,
            port=args.port,
            configure=True,
        )
        result = evaluate_programme(optimised_program, valset_full)
        ef_acc = float(result.get("error_flag_accuracy", float("nan")))
        if math.isnan(ef_acc):
            print("  Error flag accuracy: nan")
        else:
            print(f"  Error flag accuracy: {ef_acc:.3f}")

        run_dir_i = run_dir / f"rep{idx}"
        run_dir_i.mkdir(parents=True, exist_ok=True)
        optimised_paths = save_result_outputs(run_dir_i, "val", f"optimised_rep{idx}", result)

        per_run_results.append(
            {
                "run": idx,
                "seed": run_seed,
                "error_flag_accuracy": _clean_value(result.get("error_flag_accuracy")),
                "precision": _clean_value(result["binary_metrics"].get("precision")),
                "recall": _clean_value(result["binary_metrics"].get("recall")),
                "f1": _clean_value(result["binary_metrics"].get("f1")),
                "balanced_accuracy": _clean_value(result["binary_metrics"].get("balanced_accuracy")),
                "mcc": _clean_value(result["binary_metrics"].get("mcc")),
                "confusion_counts": result.get("confusion_counts"),
            }
        )

        if args.debug:
            debug_rows.append(
                {
                    "run": idx,
                    "seed": run_seed,
                    "rows": result.get("rows", [])[:5],
                }
            )

        if wandb_run is not None and wandb_module is not None:
            payload = {
                "run/index": idx,
                "run/seed": run_seed,
            }
            if not math.isnan(ef_acc):
                payload["val/error_flag_accuracy"] = ef_acc
            wandb_module.log(payload)

    metric_values = [r["error_flag_accuracy"] for r in per_run_results if r.get("error_flag_accuracy") is not None]
    if metric_values:
        mean_val = float(stats.mean(metric_values))
        std_val = float(stats.pstdev(metric_values)) if len(metric_values) > 1 else 0.0
    else:
        mean_val = float("nan")
        std_val = float("nan")

    summary_payload: Dict[str, Any] = {
        "timestamp": ts,
        "run_dir": str(run_dir),
        "preset": args.preset,
        "reflector_preset": reflector_preset,
        "seed": int(args.seed),
        "runs": int(args.runs),
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "lm_info": lm_info,
        "reflector_info": reflector_info,
        "gepa_train_size": len(trainset),
        "gepa_pareto_size": len(pareto_valset),
        "baseline": {
            "error_flag_accuracy": _clean_value(baseline_result.get("error_flag_accuracy")),
            "binary_metrics": {
                k: _clean_value(v) for k, v in baseline_result.get("binary_metrics", {}).items()
            },
            "outputs": baseline_paths,
        },
        "optimised": {
            "per_run": per_run_results,
            "mean_error_flag_accuracy": _clean_value(mean_val),
            "std_error_flag_accuracy": _clean_value(std_val),
            "outputs_last_run": optimised_paths,
        },
        "gepa_budget": {
            "auto": args.auto,
            "max_metric_calls": _clean_value(gepa_budget_metric_calls),
            "max_full_evals": _clean_value(gepa_budget_full_evals),
            "actual_metric_calls": _clean_value(gepa_actual_metric_calls),
            "actual_full_evals": _clean_value(gepa_actual_full_evals),
            "actual_full_val_evals": _clean_value(gepa_actual_full_val_evals),
            "log_dir": gepa_log_dir,
        },
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved summary → {summary_path}")

    if args.debug:
        debug_path = run_dir / "debug_preview.json"
        debug_path.write_text(json.dumps(debug_rows, indent=2) + "\n", encoding="utf-8")
        print(f"Saved debug preview → {debug_path}")

    if wandb_run is not None and wandb_module is not None:
        try:
            wandb_run.summary["baseline_error_flag_accuracy"] = baseline_result.get("error_flag_accuracy")
            wandb_run.summary["optimised_error_flag_accuracy_mean"] = _clean_value(mean_val)
            wandb_run.summary["optimised_error_flag_accuracy_std"] = _clean_value(std_val)
            if gepa_actual_metric_calls is not None:
                wandb_run.summary["gepa_metric_calls"] = gepa_actual_metric_calls
            if gepa_actual_full_evals is not None:
                wandb_run.summary["gepa_full_evals"] = gepa_actual_full_evals
            if gepa_actual_full_val_evals is not None:
                wandb_run.summary["gepa_full_val_evals"] = gepa_actual_full_val_evals
        except Exception:
            pass
        finally:
            try:
                wandb_module.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()
