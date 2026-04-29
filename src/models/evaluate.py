from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.bio import DEFAULT_BIO_OUTPUT_DIR, ENTITY_LABEL, TOKEN_LABELS, BioDataset
from src.data.io import write_json
from src.models import AUTO_HEAD_MODE, load_token_classifier_checkpoint
from src.models.baseline import (
    EncodedSampleMetadata,
    PreparedTokenClassificationDataset,
    TokenClassificationCollator,
)
from src.models.metrics import compute_sequence_labeling_metrics


DEFAULT_MODEL_DIR = "outputs/models/rumodernbert_base_04_28_15_57"
DEFAULT_DEVICE = "cuda:1"
DEFAULT_MAX_LENGTH = 256
DEFAULT_OVERFLOW_HANDLING = "drop"
DEFAULT_EVAL_BATCH_SIZE = 16
TEST_TUNING_WARNING = (
    "Threshold was selected on the test split. Treat the selected metrics as a "
    "tuned-on-test operating point, not as an unbiased test estimate."
)


@dataclass(frozen=True)
class SplitScores:
    metadata: list[EncodedSampleMetadata]
    positive_probabilities: list[list[float]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a saved token-classification checkpoint across probability "
            "thresholds without retraining."
        )
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=(
            "Path to a training run directory or directly to best_model/. "
            "Defaults to the current best RuModernBERT run."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_BIO_OUTPUT_DIR,
        help="Directory with train.jsonl/val.jsonl/test.jsonl splits.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="CUDA device for evaluation. CPU fallback is intentionally disabled.",
    )
    parser.add_argument(
        "--head-mode",
        choices=("auto", "baseline", "neighbor", "combined"),
        default=AUTO_HEAD_MODE,
        help=(
            "Checkpoint head mode. Use auto to read it from config.json; "
            "legacy checkpoints resolve to baseline."
        ),
    )
    parser.add_argument(
        "--select-threshold-on",
        choices=("val", "test"),
        default="test",
        help="Split used both for sweep metrics and threshold selection.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for threshold sweep JSON files. Defaults to "
            "<run_dir>/threshold_sweep when a run directory is available."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Maximum transformer sequence length. If omitted, read from "
            "run_config.json and fall back to 256."
        ),
    )
    parser.add_argument(
        "--overflow-handling",
        choices=("drop", "truncate"),
        default=None,
        help=(
            "Long-sample handling. If omitted, read from run_config.json and "
            "fall back to drop."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help=(
            "Evaluation batch size. If omitted, read from run_config.json and "
            "fall back to 16."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--threshold-min",
        default="0.01",
        help="Minimum threshold in the sweep grid.",
    )
    parser.add_argument(
        "--threshold-max",
        default="0.99",
        help="Maximum threshold in the sweep grid.",
    )
    parser.add_argument(
        "--threshold-step",
        default="0.01",
        help="Threshold grid step.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional first-N sample limit for quick smoke tests.",
    )
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_checkpoint_dir(path: str | Path) -> tuple[Path, Path | None]:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Model path does not exist: {candidate}")

    best_model_dir = candidate / "best_model"
    if best_model_dir.is_dir() and (best_model_dir / "config.json").exists():
        return best_model_dir, candidate

    if (candidate / "config.json").exists():
        run_dir = candidate.parent if candidate.name == "best_model" else None
        return candidate, run_dir

    raise FileNotFoundError(
        "Could not find checkpoint config.json. Pass either a run directory "
        "containing best_model/ or a direct best_model path."
    )


def read_run_config(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        return {}
    payload = read_json(run_config_path)
    config = payload.get("config")
    return config if isinstance(config, dict) else {}


def resolve_config_int(
    *,
    cli_value: int | None,
    run_config: dict[str, Any],
    key: str,
    default: int,
) -> int:
    if cli_value is not None:
        value = cli_value
    else:
        value = run_config.get(key, default)
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer.") from exc
    if resolved <= 0:
        raise ValueError(f"{key} must be positive.")
    return resolved


def resolve_overflow_handling(
    *,
    cli_value: str | None,
    run_config: dict[str, Any],
) -> str:
    value = cli_value or str(run_config.get("overflow_handling", DEFAULT_OVERFLOW_HANDLING))
    if value not in {"drop", "truncate"}:
        raise ValueError("overflow_handling must be either 'drop' or 'truncate'.")
    return value


def resolve_required_cuda_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for threshold evaluation, but it is unavailable.")
        device = torch.device("cuda")
    else:
        device = torch.device(requested_device)

    if device.type != "cuda":
        raise RuntimeError(
            f"Threshold evaluation is GPU-only for this workflow; got device {device}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    device_count = torch.cuda.device_count()
    if device.index is not None and device.index >= device_count:
        raise RuntimeError(
            f"Requested {device}, but only {device_count} CUDA device(s) are visible."
        )
    if device.index is not None:
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")
    return device


def parse_decimal(value: str, *, argument_name: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{argument_name} must be a decimal number.") from exc
    if not parsed.is_finite():
        raise ValueError(f"{argument_name} must be finite.")
    return parsed


def build_threshold_grid(
    *,
    min_value: str,
    max_value: str,
    step: str,
) -> list[Decimal]:
    threshold_min = parse_decimal(min_value, argument_name="--threshold-min")
    threshold_max = parse_decimal(max_value, argument_name="--threshold-max")
    threshold_step = parse_decimal(step, argument_name="--threshold-step")

    if threshold_step <= 0:
        raise ValueError("--threshold-step must be positive.")
    if threshold_min < 0 or threshold_max > 1:
        raise ValueError("Threshold values must be between 0 and 1.")
    if threshold_min > threshold_max:
        raise ValueError("--threshold-min must be <= --threshold-max.")

    thresholds: list[Decimal] = []
    value = threshold_min
    while value <= threshold_max:
        thresholds.append(value)
        value += threshold_step
    if not thresholds:
        raise ValueError("Threshold grid is empty.")
    return thresholds


def format_threshold(threshold: Decimal) -> str:
    return format(threshold.normalize(), "f")


def compute_fbeta(precision: float, recall: float, *, beta: float) -> float:
    beta_squared = beta * beta
    denominator = (beta_squared * precision) + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_squared) * precision * recall / denominator


def add_f2_metrics(metrics: dict[str, float | int]) -> dict[str, float | int]:
    enriched = dict(metrics)
    enriched["token_f2"] = compute_fbeta(
        float(metrics["token_precision"]),
        float(metrics["token_recall"]),
        beta=2.0,
    )
    enriched["span_f2"] = compute_fbeta(
        float(metrics["span_precision"]),
        float(metrics["span_recall"]),
        beta=2.0,
    )
    return enriched


def tags_from_probabilities(
    probabilities: Sequence[float],
    *,
    threshold: Decimal,
) -> list[str]:
    threshold_value = float(threshold)
    return [
        ENTITY_LABEL if probability > threshold_value else "O"
        for probability in probabilities
    ]


def build_predicted_sequences(
    split_scores: SplitScores,
    *,
    threshold: Decimal,
) -> list[list[str]]:
    return [
        tags_from_probabilities(probabilities, threshold=threshold)
        for probabilities in split_scores.positive_probabilities
    ]


def evaluate_threshold(
    split_scores: SplitScores,
    *,
    threshold: Decimal,
) -> dict[str, float | int]:
    gold_sequences = [metadata.gold_tags for metadata in split_scores.metadata]
    predicted_sequences = build_predicted_sequences(split_scores, threshold=threshold)
    return add_f2_metrics(
        compute_sequence_labeling_metrics(gold_sequences, predicted_sequences)
    )


def build_dataloader(
    *,
    dataset_dir: str | Path,
    split_name: str,
    tokenizer,
    max_length: int,
    overflow_handling: str,
    eval_batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> tuple[DataLoader, dict[str, Any]]:
    label_to_id = {label: index for index, label in enumerate(TOKEN_LABELS)}
    samples = list(BioDataset.from_directory(dataset_dir, split=split_name))
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("--max-samples must be positive when provided.")
        samples = samples[:max_samples]

    dataset = PreparedTokenClassificationDataset(
        samples,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=max_length,
        overflow_handling=overflow_handling,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"Split '{split_name}' became empty after tokenization. "
            "Increase --max-length or switch --overflow-handling to truncate."
        )

    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=TokenClassificationCollator(tokenizer),
        num_workers=num_workers,
    )
    stats = {
        "available_samples": len(samples),
        "used_samples": len(dataset),
        "tokenization": dataset.stats.to_dict(),
    }
    return dataloader, stats


def collect_split_scores(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    positive_label_id: int,
    use_word_start_mask: bool,
) -> SplitScores:
    metadata_rows: list[EncodedSampleMetadata] = []
    probability_rows: list[list[float]] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            metadata: list[EncodedSampleMetadata] = batch.pop("metadata")
            batch.pop("labels", None)
            if not use_word_start_mask:
                batch.pop("word_start_mask", None)
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            outputs = model(**batch)
            probabilities = (
                outputs.logits.detach().cpu().softmax(dim=-1)[..., positive_label_id]
            )

            for row_index, sample_metadata in enumerate(metadata):
                row_probabilities = [
                    float(probabilities[row_index, position].item())
                    for position in sample_metadata.first_token_positions
                ]
                if len(row_probabilities) != len(sample_metadata.gold_tags):
                    raise RuntimeError(
                        "Predicted word-level probabilities are not aligned with gold tags."
                    )
                metadata_rows.append(sample_metadata)
                probability_rows.append(row_probabilities)

    return SplitScores(
        metadata=metadata_rows,
        positive_probabilities=probability_rows,
    )


def select_best_threshold(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot select threshold from an empty sweep.")
    return max(
        rows,
        key=lambda row: (
            float(row["metrics"]["span_f2"]),
            float(row["metrics"]["span_recall"]),
            float(row["metrics"]["span_f1"]),
            float(row["metrics"]["span_precision"]),
            float(row["threshold"]),
        ),
    )


def find_threshold_row(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: Decimal,
) -> dict[str, Any] | None:
    threshold_label = format_threshold(threshold)
    for row in rows:
        if row["threshold_label"] == threshold_label:
            return row
    return None


def resolve_output_dir(
    *,
    cli_value: str | None,
    run_dir: Path | None,
    checkpoint_dir: Path,
) -> Path:
    if cli_value is not None:
        return Path(cli_value)
    if run_dir is not None:
        return run_dir / "threshold_sweep"
    return checkpoint_dir / "threshold_sweep"


def main() -> int:
    args = build_parser().parse_args()

    try:
        checkpoint_dir, run_dir = resolve_checkpoint_dir(args.model_dir)
        run_config = read_run_config(run_dir)
        max_length = resolve_config_int(
            cli_value=args.max_length,
            run_config=run_config,
            key="max_length",
            default=DEFAULT_MAX_LENGTH,
        )
        eval_batch_size = resolve_config_int(
            cli_value=args.eval_batch_size,
            run_config=run_config,
            key="eval_batch_size",
            default=DEFAULT_EVAL_BATCH_SIZE,
        )
        overflow_handling = resolve_overflow_handling(
            cli_value=args.overflow_handling,
            run_config=run_config,
        )
        thresholds = build_threshold_grid(
            min_value=args.threshold_min,
            max_value=args.threshold_max,
            step=args.threshold_step,
        )
        device = resolve_required_cuda_device(args.device)

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        if not tokenizer.is_fast:
            raise RuntimeError(
                "A fast tokenizer is required for token classification evaluation."
            )
        model, checkpoint_metadata = load_token_classifier_checkpoint(
            checkpoint_dir,
            requested_head_mode=args.head_mode,
        )
        model.to(device)

        dataloader, dataset_stats = build_dataloader(
            dataset_dir=args.dataset_dir,
            split_name=args.select_threshold_on,
            tokenizer=tokenizer,
            max_length=max_length,
            overflow_handling=overflow_handling,
            eval_batch_size=eval_batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
        )
        split_scores = collect_split_scores(
            model=model,
            dataloader=dataloader,
            device=device,
            positive_label_id=checkpoint_metadata.positive_label_id,
            use_word_start_mask=not checkpoint_metadata.is_legacy,
        )

        sweep_rows: list[dict[str, Any]] = []
        for threshold in thresholds:
            metrics = evaluate_threshold(split_scores, threshold=threshold)
            sweep_rows.append(
                {
                    "threshold": float(threshold),
                    "threshold_label": format_threshold(threshold),
                    "metrics": metrics,
                }
            )

        selected_row = select_best_threshold(sweep_rows)
        baseline_row = find_threshold_row(sweep_rows, threshold=Decimal("0.5"))
        output_dir = resolve_output_dir(
            cli_value=args.output_dir,
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        selection_policy = {
            "primary_metric": "span_f2",
            "tie_breakers": [
                "span_recall",
                "span_f1",
                "span_precision",
                "threshold",
            ],
            "prediction_rule": "Predict EUPHEMISM when positive_probability > threshold.",
        }
        common_metadata = {
            "warning": (
                TEST_TUNING_WARNING
                if args.select_threshold_on == "test"
                else None
            ),
            "model_dir": str(checkpoint_dir),
            "run_dir": str(run_dir) if run_dir is not None else None,
            "dataset_dir": str(Path(args.dataset_dir)),
            "split": args.select_threshold_on,
            "device": str(device),
            "checkpoint": checkpoint_metadata.to_dict(),
            "max_length": max_length,
            "overflow_handling": overflow_handling,
            "eval_batch_size": eval_batch_size,
            "dataset": dataset_stats,
            "selection_policy": selection_policy,
        }
        sweep_payload = {
            **common_metadata,
            "thresholds": sweep_rows,
        }
        selected_payload = {
            **common_metadata,
            "selected_threshold": selected_row["threshold"],
            "selected_threshold_label": selected_row["threshold_label"],
            "selected_metrics": selected_row["metrics"],
            "baseline_threshold_0_5": baseline_row,
        }

        sweep_path = output_dir / f"{args.select_threshold_on}_threshold_sweep.json"
        selected_path = output_dir / "selected_threshold_metrics.json"
        write_json(sweep_path, sweep_payload)
        write_json(selected_path, selected_payload)

        selected_metrics = selected_row["metrics"]
        print(f"Checkpoint: {checkpoint_dir}")
        print(f"Split: {args.select_threshold_on}")
        print(f"Device: {device}")
        print(
            "Selected threshold: "
            f"{selected_row['threshold_label']} "
            f"(span_f2={selected_metrics['span_f2']:.4f}, "
            f"span_recall={selected_metrics['span_recall']:.4f}, "
            f"span_precision={selected_metrics['span_precision']:.4f}, "
            f"span_f1={selected_metrics['span_f1']:.4f})"
        )
        if baseline_row is not None:
            baseline_metrics = baseline_row["metrics"]
            print(
                "Threshold 0.5: "
                f"span_f2={baseline_metrics['span_f2']:.4f}, "
                f"span_recall={baseline_metrics['span_recall']:.4f}, "
                f"span_precision={baseline_metrics['span_precision']:.4f}, "
                f"span_f1={baseline_metrics['span_f1']:.4f}"
            )
        print(f"Sweep JSON: {sweep_path}")
        print(f"Selected JSON: {selected_path}")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
