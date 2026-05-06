#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.bio import DEFAULT_BIO_OUTPUT_DIR, ENTITY_LABEL, TOKEN_LABELS
from src.data.io import write_json
from src.models.metrics import compute_sequence_labeling_metrics, token_labels_to_spans

DEFAULT_THRESHOLD = 0.5
OPTIMIZATION_METRIC = "span_f1"
TIE_BREAKER_METRIC = "token_f1"


@dataclass(frozen=True)
class ScoredSample:
    sample_id: str
    source: str
    source_index: int
    negative_group: str | None
    text: str
    tokens: list[str]
    gold_tags: list[str]
    positive_scores: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _ScoredTokenRef:
    score: float
    sequence_index: int
    token_index: int
    is_gold_positive: bool


@dataclass
class _IncrementalSweepState:
    positive_label: str
    positive_span_label: str
    total_tokens: int
    gold_entity_tokens: int
    gold_spans: int
    token_tp: int
    token_fp: int
    token_fn: int
    token_correct: int
    span_tp: int
    predicted_spans: int
    active_masks: list[bytearray]
    gold_span_sets: list[set[tuple[str, int, int]]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune a positive-class probability threshold for a trained "
            "binary token-classification checkpoint."
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run threshold-sweep smoke checks and exit.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Path to a training run directory or directly to best_model/. "
            "Examples: outputs/models/<run_name> or outputs/models/<run_name>/best_model"
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Directory with train.jsonl/val.jsonl/test.jsonl. "
            "If omitted, the script reads config.dataset_dir from run_config.json "
            f"and falls back to {DEFAULT_BIO_OUTPUT_DIR}."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Dataset split used for threshold tuning.",
    )
    parser.add_argument(
        "--head-mode",
        choices=("auto", "baseline", "neighbor", "combined"),
        default="auto",
        help=(
            "Checkpoint head mode. Use auto to read it from config.json; "
            "legacy checkpoints resolve to baseline."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device string: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Per-device batch size for the single model pass.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Max transformer sequence length. If omitted, the script tries to read "
            "it from run_config.json and falls back to 256."
        ),
    )
    parser.add_argument(
        "--overflow-handling",
        choices=("drop", "truncate"),
        default=None,
        help=(
            "What to do with samples longer than max length. If omitted, the script "
            "reads config.overflow_handling from run_config.json and falls back to drop."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional first-N sample limit for smoke/dry runs.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print progress every N evaluation batches. Use 0 to disable batch logs.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Optional JSON output path. If omitted, writes to "
            "<run_dir>/threshold_tuning/<split>_span_f1.json."
        ),
    )
    return parser


def load_run_config(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}

    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        return {}

    return json.loads(run_config_path.read_text(encoding="utf-8"))


def resolve_dataset_dir(
    *,
    cli_value: str | None,
    run_config: dict[str, Any],
) -> Path:
    if cli_value is not None:
        return Path(cli_value)

    config_value = run_config.get("config", {}).get("dataset_dir")
    if isinstance(config_value, str) and config_value:
        return Path(config_value)

    return Path(DEFAULT_BIO_OUTPUT_DIR)


def resolve_overflow_handling(
    *,
    cli_value: str | None,
    run_config: dict[str, Any],
) -> str:
    if cli_value is not None:
        return cli_value

    config_value = run_config.get("config", {}).get("overflow_handling")
    if config_value in {"drop", "truncate"}:
        return str(config_value)

    return "drop"


def resolve_output_path(
    *,
    cli_value: str | None,
    run_dir: Path | None,
    checkpoint_dir: Path,
    split: str,
) -> Path:
    if cli_value is not None:
        return Path(cli_value)

    output_root = run_dir if run_dir is not None else checkpoint_dir
    return output_root / "threshold_tuning" / f"{split}_span_f1.json"


def limit_samples(samples: Sequence[Any], *, max_samples: int | None) -> list[Any]:
    sample_list = list(samples)
    if max_samples is None:
        return sample_list
    if max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided.")
    return sample_list[:max_samples]


def flatten_scores(score_sequences: Sequence[Sequence[float]]) -> list[float]:
    scores: list[float] = []
    for sequence in score_sequences:
        for score in sequence:
            value = float(score)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite score found during threshold tuning: {score}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "Positive-class scores must be probabilities between 0 and 1. "
                    f"Found: {score}"
                )
            scores.append(value)
    return scores


def build_threshold_candidates(
    score_sequences: Sequence[Sequence[float]],
) -> list[float]:
    unique_scores = sorted(set(flatten_scores(score_sequences)))
    candidates = {0.0, DEFAULT_THRESHOLD, 1.0}

    for left, right in zip(unique_scores, unique_scores[1:]):
        if left != right:
            candidates.add((left + right) * 0.5)

    return sorted(candidates)


def predict_tags_from_scores(
    score_sequences: Sequence[Sequence[float]],
    *,
    threshold: float,
    positive_label: str = ENTITY_LABEL,
) -> list[list[str]]:
    return [
        [
            positive_label if float(score) > threshold else "O"
            for score in score_sequence
        ]
        for score_sequence in score_sequences
    ]


def metrics_for_threshold(
    *,
    score_sequences: Sequence[Sequence[float]],
    gold_sequences: Sequence[Sequence[str]],
    threshold: float,
    positive_label: str = ENTITY_LABEL,
) -> dict[str, float | int]:
    predicted_sequences = predict_tags_from_scores(
        score_sequences,
        threshold=threshold,
        positive_label=positive_label,
    )
    return compute_sequence_labeling_metrics(gold_sequences, predicted_sequences)


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _flatten_scored_token_refs(
    *,
    score_sequences: Sequence[Sequence[float]],
    gold_sequences: Sequence[Sequence[str]],
) -> list[_ScoredTokenRef]:
    token_refs: list[_ScoredTokenRef] = []
    for sequence_index, (scores, gold_tags) in enumerate(
        zip(score_sequences, gold_sequences)
    ):
        for token_index, (score, gold_tag) in enumerate(zip(scores, gold_tags)):
            token_refs.append(
                _ScoredTokenRef(
                    score=float(score),
                    sequence_index=sequence_index,
                    token_index=token_index,
                    is_gold_positive=gold_tag != "O",
                )
            )
    token_refs.sort(key=lambda item: item.score, reverse=True)
    return token_refs


def _span_label_from_tag(tag: str) -> str:
    _, separator, entity_label = tag.partition("-")
    if separator and entity_label:
        return entity_label
    return tag


def _build_incremental_sweep_state(
    *,
    gold_sequences: Sequence[Sequence[str]],
    positive_label: str,
) -> _IncrementalSweepState:
    active_masks = [bytearray(len(gold_tags)) for gold_tags in gold_sequences]
    gold_span_sets = [
        set(token_labels_to_spans(gold_tags))
        for gold_tags in gold_sequences
    ]
    total_tokens = sum(len(gold_tags) for gold_tags in gold_sequences)
    gold_entity_tokens = sum(
        1
        for gold_tags in gold_sequences
        for gold_tag in gold_tags
        if gold_tag != "O"
    )

    return _IncrementalSweepState(
        positive_label=positive_label,
        positive_span_label=_span_label_from_tag(positive_label),
        total_tokens=total_tokens,
        gold_entity_tokens=gold_entity_tokens,
        gold_spans=sum(len(spans) for spans in gold_span_sets),
        token_tp=0,
        token_fp=0,
        token_fn=gold_entity_tokens,
        token_correct=total_tokens - gold_entity_tokens,
        span_tp=0,
        predicted_spans=0,
        active_masks=active_masks,
        gold_span_sets=gold_span_sets,
    )


def _activate_token_for_threshold(
    *,
    state: _IncrementalSweepState,
    token_ref: _ScoredTokenRef,
) -> None:
    sequence_index = token_ref.sequence_index
    token_index = token_ref.token_index
    active = state.active_masks[sequence_index]
    if active[token_index]:
        return

    if token_ref.is_gold_positive:
        state.token_tp += 1
        state.token_fn -= 1
        state.token_correct += 1
    else:
        state.token_fp += 1
        state.token_correct -= 1

    new_span = (
        state.positive_span_label,
        token_index,
        token_index + 1,
    )
    gold_spans = state.gold_span_sets[sequence_index]
    if new_span in gold_spans:
        state.span_tp += 1

    state.predicted_spans += 1
    active[token_index] = 1


def _metrics_from_incremental_state(
    state: _IncrementalSweepState,
) -> dict[str, float | int]:
    token_metrics = _precision_recall_f1(
        state.token_tp,
        state.token_fp,
        state.token_fn,
    )
    span_fp = state.predicted_spans - state.span_tp
    span_fn = state.gold_spans - state.span_tp
    span_metrics = _precision_recall_f1(state.span_tp, span_fp, span_fn)

    return {
        "token_precision": token_metrics["precision"],
        "token_recall": token_metrics["recall"],
        "token_f1": token_metrics["f1"],
        "token_accuracy": _safe_divide(state.token_correct, state.total_tokens),
        "token_tp": token_metrics["tp"],
        "token_fp": token_metrics["fp"],
        "token_fn": token_metrics["fn"],
        "total_tokens": state.total_tokens,
        "gold_entity_tokens": state.gold_entity_tokens,
        "span_precision": span_metrics["precision"],
        "span_recall": span_metrics["recall"],
        "span_f1": span_metrics["f1"],
        "span_tp": span_metrics["tp"],
        "span_fp": span_metrics["fp"],
        "span_fn": span_metrics["fn"],
        "gold_spans": state.gold_spans,
        "predicted_spans": state.predicted_spans,
    }


def build_curve_entry(
    *,
    threshold: float,
    metrics: dict[str, float | int],
) -> dict[str, float | int]:
    return {
        "threshold": threshold,
        "span_precision": metrics["span_precision"],
        "span_recall": metrics["span_recall"],
        "span_f1": metrics["span_f1"],
        "token_precision": metrics["token_precision"],
        "token_recall": metrics["token_recall"],
        "token_f1": metrics["token_f1"],
        "predicted_spans": metrics["predicted_spans"],
    }


def sweep_thresholds(
    *,
    score_sequences: Sequence[Sequence[float]],
    gold_sequences: Sequence[Sequence[str]],
    positive_label: str = ENTITY_LABEL,
    show_progress: bool = False,
) -> dict[str, Any]:
    if len(score_sequences) != len(gold_sequences):
        raise ValueError("Score and gold sequence lists must have the same size.")
    for scores, gold_tags in zip(score_sequences, gold_sequences):
        if len(scores) != len(gold_tags):
            raise ValueError("Score and gold token sequences must be aligned.")

    candidates = build_threshold_candidates(score_sequences)
    token_refs = _flatten_scored_token_refs(
        score_sequences=score_sequences,
        gold_sequences=gold_sequences,
    )
    state = _build_incremental_sweep_state(
        gold_sequences=gold_sequences,
        positive_label=positive_label,
    )
    if show_progress:
        print(
            f"Sweeping {len(candidates)} threshold candidates "
            f"over {len(token_refs)} word-level scores.",
            flush=True,
        )

    scored_thresholds: list[dict[str, Any]] = []
    token_ref_index = 0
    for threshold in sorted(candidates, reverse=True):
        while (
            token_ref_index < len(token_refs)
            and token_refs[token_ref_index].score > threshold
        ):
            _activate_token_for_threshold(
                state=state,
                token_ref=token_refs[token_ref_index],
            )
            token_ref_index += 1

        metrics = _metrics_from_incremental_state(state)
        scored_thresholds.append(
            {
                "threshold": threshold,
                "metrics": metrics,
                "curve_entry": build_curve_entry(
                    threshold=threshold,
                    metrics=metrics,
                ),
            }
        )

    best = max(
        scored_thresholds,
        key=lambda row: (
            float(row["metrics"][OPTIMIZATION_METRIC]),
            float(row["metrics"][TIE_BREAKER_METRIC]),
            -abs(float(row["threshold"]) - DEFAULT_THRESHOLD),
            -float(row["threshold"]),
        ),
    )
    default_metrics = next(
        row["metrics"]
        for row in scored_thresholds
        if float(row["threshold"]) == DEFAULT_THRESHOLD
    )
    scored_thresholds.sort(key=lambda row: float(row["threshold"]))

    return {
        "best_threshold": best["threshold"],
        "best_metrics": best["metrics"],
        "default_threshold": DEFAULT_THRESHOLD,
        "default_threshold_metrics": default_metrics,
        "threshold_curve": [row["curve_entry"] for row in scored_thresholds],
        "candidate_count": len(candidates),
    }


def threshold_tags_match_argmax_for_binary_logits() -> bool:
    logits = [-2.0, 0.0, 2.0]
    scores = [[1.0 / (1.0 + math.exp(-logit)) for logit in logits]]
    threshold_tags = predict_tags_from_scores(
        scores,
        threshold=DEFAULT_THRESHOLD,
        positive_label=ENTITY_LABEL,
    )[0]
    argmax_tags = [
        ENTITY_LABEL if logit > 0.0 else "O"
        for logit in logits
    ]
    return threshold_tags == argmax_tags


def run_self_test() -> None:
    score_sequences = [[0.1, 0.4, 0.6, 0.9]]
    gold_sequences = [["O", "O", ENTITY_LABEL, ENTITY_LABEL]]
    result = sweep_thresholds(
        score_sequences=score_sequences,
        gold_sequences=gold_sequences,
    )
    if result["best_threshold"] != DEFAULT_THRESHOLD:
        raise AssertionError(
            "Expected threshold 0.5 to be selected on the synthetic smoke case. "
            f"Got {result['best_threshold']!r}."
        )
    if float(result["best_metrics"]["span_f1"]) != 1.0:
        raise AssertionError("Synthetic threshold smoke case should reach span_f1=1.0.")
    if not threshold_tags_match_argmax_for_binary_logits():
        raise AssertionError(
            "Threshold 0.5 must match argmax for binary logits shaped as [0, z]."
        )

    exact_score_sequences = [
        [0.0, 0.2, 0.5, 0.7, 1.0],
        [0.51, 0.49, 0.9, 0.1],
    ]
    exact_gold_sequences = [
        ["O", ENTITY_LABEL, ENTITY_LABEL, "O", ENTITY_LABEL],
        [ENTITY_LABEL, "O", ENTITY_LABEL, "O"],
    ]
    exact_result = sweep_thresholds(
        score_sequences=exact_score_sequences,
        gold_sequences=exact_gold_sequences,
    )
    for curve_entry in exact_result["threshold_curve"]:
        brute_metrics = metrics_for_threshold(
            score_sequences=exact_score_sequences,
            gold_sequences=exact_gold_sequences,
            threshold=float(curve_entry["threshold"]),
        )
        for metric_name in (
            "span_precision",
            "span_recall",
            "span_f1",
            "token_precision",
            "token_recall",
            "token_f1",
            "predicted_spans",
        ):
            metric_delta = abs(
                float(curve_entry[metric_name])
                - float(brute_metrics[metric_name])
            )
            if metric_delta > 1e-12:
                raise AssertionError(
                    "Incremental sweep diverged from brute force metrics for "
                    f"{metric_name} at threshold {curve_entry['threshold']}."
                )
    print("Threshold tuning self-test passed.")


def collect_scored_samples(
    *,
    dataset_dir: Path,
    split: str,
    tokenizer,
    model,
    checkpoint_metadata,
    torch_module,
    dataloader_cls,
    dataset_cls,
    collator_cls,
    load_split_samples_fn,
    device,
    eval_batch_size: int,
    max_length: int,
    overflow_handling: str,
    max_samples: int | None,
    log_every: int,
) -> tuple[list[ScoredSample], dict[str, Any]]:
    if eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive.")
    if log_every < 0:
        raise ValueError("--log-every must be non-negative.")

    label_to_id = {label: index for index, label in enumerate(TOKEN_LABELS)}
    samples = limit_samples(
        load_split_samples_fn(dataset_dir, split=split),
        max_samples=max_samples,
    )
    dataset = dataset_cls(
        samples,
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_length=max_length,
        overflow_handling=overflow_handling,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"Split {split!r} became empty after tokenization. "
            "Increase --max-length or switch --overflow-handling to truncate."
        )
    print(
        f"Prepared {len(dataset)} tokenized samples "
        f"from {len(samples)} loaded {split!r} samples.",
        flush=True,
    )

    dataloader = dataloader_cls(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator_cls(tokenizer),
        num_workers=0,
    )

    scored_samples: list[ScoredSample] = []
    positive_label_id = int(checkpoint_metadata.positive_label_id)
    model.eval()

    with torch_module.no_grad():
        total_batches = len(dataloader)
        for batch_index, batch in enumerate(dataloader, start=1):
            metadata = batch.pop("metadata")
            if checkpoint_metadata.is_legacy:
                batch.pop("word_start_mask", None)
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }
            outputs = model(**batch)
            if positive_label_id >= int(outputs.logits.shape[-1]):
                raise RuntimeError(
                    f"Positive label id {positive_label_id} is outside logits shape "
                    f"{tuple(outputs.logits.shape)}."
                )
            positive_scores = torch_module.sigmoid(
                outputs.logits[..., positive_label_id]
            ).detach().cpu()

            for row_index, sample_metadata in enumerate(metadata):
                scores = [
                    float(positive_scores[row_index, position].item())
                    for position in sample_metadata.first_token_positions
                ]
                gold_tags = list(sample_metadata.gold_tags)
                if len(scores) != len(gold_tags):
                    raise RuntimeError(
                        "Scored word-level tags are not aligned with gold tags."
                    )
                scored_samples.append(
                    ScoredSample(
                        sample_id=sample_metadata.sample_id,
                        source=sample_metadata.source,
                        source_index=sample_metadata.source_index,
                        negative_group=sample_metadata.negative_group,
                        text=sample_metadata.text,
                        tokens=list(sample_metadata.tokens),
                        gold_tags=gold_tags,
                        positive_scores=scores,
                    )
                )
            if log_every > 0 and (
                batch_index % log_every == 0 or batch_index == total_batches
            ):
                print(
                    f"Scored batch {batch_index}/{total_batches} "
                    f"({len(scored_samples)} samples).",
                    flush=True,
                )

    return scored_samples, dataset.stats.to_dict()


def build_payload(
    *,
    args,
    model_dir: str,
    checkpoint_dir: Path,
    run_dir: Path | None,
    dataset_dir: Path,
    output_path: Path,
    checkpoint_metadata,
    max_length: int,
    overflow_handling: str,
    scored_samples: list[ScoredSample],
    tokenization_stats: dict[str, Any],
    sweep_result: dict[str, Any],
) -> dict[str, Any]:
    gold_sequences = [sample.gold_tags for sample in scored_samples]
    token_count = sum(len(sequence) for sequence in gold_sequences)
    score_count = sum(len(sample.positive_scores) for sample in scored_samples)

    return {
        "warning": (
            "Threshold was tuned on the selected split. If split='test', these are "
            "tuned-on-test metrics, not an independent final evaluation."
        ),
        "model_dir": model_dir,
        "checkpoint_dir": str(checkpoint_dir),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "output_path": str(output_path),
        "head_mode": checkpoint_metadata.head_mode,
        "checkpoint_architecture": checkpoint_metadata.checkpoint_architecture,
        "checkpoint_metadata": checkpoint_metadata.to_dict(),
        "optimization_metric": OPTIMIZATION_METRIC,
        "tie_breaker_metric": TIE_BREAKER_METRIC,
        "positive_label": ENTITY_LABEL,
        "positive_label_id": checkpoint_metadata.positive_label_id,
        "max_length": max_length,
        "overflow_handling": overflow_handling,
        "eval_batch_size": args.eval_batch_size,
        "max_samples": args.max_samples,
        "counts": {
            "samples": len(scored_samples),
            "tokens": token_count,
            "scores": score_count,
            "gold_spans": sweep_result["best_metrics"]["gold_spans"],
            "best_predicted_spans": sweep_result["best_metrics"]["predicted_spans"],
            "default_predicted_spans": sweep_result["default_threshold_metrics"][
                "predicted_spans"
            ],
            "threshold_candidates": sweep_result["candidate_count"],
        },
        "tokenization": tokenization_stats,
        "best_threshold": sweep_result["best_threshold"],
        "best_metrics": sweep_result["best_metrics"],
        "default_threshold": sweep_result["default_threshold"],
        "default_threshold_metrics": sweep_result["default_threshold_metrics"],
        "threshold_curve": sweep_result["threshold_curve"],
        "scored_samples": [sample.to_dict() for sample in scored_samples],
    }


def print_summary(payload: dict[str, Any]) -> None:
    best = payload["best_metrics"]
    default = payload["default_threshold_metrics"]
    counts = payload["counts"]

    print("Threshold tuning complete.")
    print(
        "Warning: threshold was tuned on this split; "
        "test results are tuned-on-test metrics."
    )
    print(f"Model: {payload['checkpoint_dir']}")
    print(f"Dataset: {payload['dataset_dir']} | split={payload['split']}")
    print(f"Head mode: {payload['head_mode']}")
    print(f"Optimization metric: {payload['optimization_metric']}")
    print(f"Best threshold: {float(payload['best_threshold']):.8f}")
    print(
        "Best span metrics: "
        f"P={float(best['span_precision']):.4f} "
        f"R={float(best['span_recall']):.4f} "
        f"F1={float(best['span_f1']):.4f}"
    )
    print(
        "Best token metrics: "
        f"P={float(best['token_precision']):.4f} "
        f"R={float(best['token_recall']):.4f} "
        f"F1={float(best['token_f1']):.4f}"
    )
    print(
        "Default threshold 0.5 span metrics: "
        f"P={float(default['span_precision']):.4f} "
        f"R={float(default['span_recall']):.4f} "
        f"F1={float(default['span_f1']):.4f}"
    )
    print(
        "Default threshold 0.5 token metrics: "
        f"P={float(default['token_precision']):.4f} "
        f"R={float(default['token_recall']):.4f} "
        f"F1={float(default['token_f1']):.4f}"
    )
    print(
        "Counts: "
        f"samples={counts['samples']} "
        f"tokens={counts['tokens']} "
        f"gold_spans={counts['gold_spans']} "
        f"best_predicted_spans={counts['best_predicted_spans']} "
        f"default_predicted_spans={counts['default_predicted_spans']}"
    )
    print(f"Saved JSON: {payload['output_path']}")


def main() -> int:
    args = build_parser().parse_args()

    if args.self_test:
        run_self_test()
        return 0

    if args.model_dir is None:
        print("--model-dir is required unless --self-test is used.", file=sys.stderr)
        return 2

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer

        from scripts.infer_one_text import (
            resolve_checkpoint_dir,
            resolve_device,
            resolve_max_length,
        )
        from src.models import load_token_classifier_checkpoint
        from src.models.baseline import (
            PreparedTokenClassificationDataset,
            TokenClassificationCollator,
            load_split_samples,
        )
    except ModuleNotFoundError as exc:
        print(
            "Missing threshold tuning dependencies. Install PyTorch and transformers "
            "first, for example: venv/bin/pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Original import error: {exc}", file=sys.stderr)
        return 1

    try:
        checkpoint_dir, run_dir = resolve_checkpoint_dir(args.model_dir)
        run_config = load_run_config(run_dir)
        dataset_dir = resolve_dataset_dir(
            cli_value=args.dataset_dir,
            run_config=run_config,
        )
        max_length = resolve_max_length(
            cli_value=args.max_length,
            run_dir=run_dir,
        )
        overflow_handling = resolve_overflow_handling(
            cli_value=args.overflow_handling,
            run_config=run_config,
        )
        output_path = resolve_output_path(
            cli_value=args.output_path,
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            split=args.split,
        )

        print(f"Checkpoint: {checkpoint_dir}", flush=True)
        print(f"Dataset dir: {dataset_dir} | split={args.split}", flush=True)
        print(
            f"max_length={max_length} | overflow_handling={overflow_handling} | "
            f"eval_batch_size={args.eval_batch_size}",
            flush=True,
        )
        print("Loading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        if not tokenizer.is_fast:
            raise RuntimeError(
                "A fast tokenizer is required for token classification threshold tuning."
            )
        print("Loading checkpoint...", flush=True)
        model, checkpoint_metadata = load_token_classifier_checkpoint(
            checkpoint_dir,
            requested_head_mode=args.head_mode,
        )
        device = resolve_device(args.device, torch)
        print(f"Moving model to device: {device}", flush=True)
        model.to(device)
        print("Collecting word-level positive scores...", flush=True)

        scored_samples, tokenization_stats = collect_scored_samples(
            dataset_dir=dataset_dir,
            split=args.split,
            tokenizer=tokenizer,
            model=model,
            checkpoint_metadata=checkpoint_metadata,
            torch_module=torch,
            dataloader_cls=DataLoader,
            dataset_cls=PreparedTokenClassificationDataset,
            collator_cls=TokenClassificationCollator,
            load_split_samples_fn=load_split_samples,
            device=device,
            eval_batch_size=args.eval_batch_size,
            max_length=max_length,
            overflow_handling=overflow_handling,
            max_samples=args.max_samples,
            log_every=args.log_every,
        )
        score_sequences = [sample.positive_scores for sample in scored_samples]
        gold_sequences = [sample.gold_tags for sample in scored_samples]
        print("Sweeping thresholds...", flush=True)
        sweep_result = sweep_thresholds(
            score_sequences=score_sequences,
            gold_sequences=gold_sequences,
            positive_label=ENTITY_LABEL,
            show_progress=True,
        )
        payload = build_payload(
            args=args,
            model_dir=args.model_dir,
            checkpoint_dir=checkpoint_dir,
            run_dir=run_dir,
            dataset_dir=dataset_dir,
            output_path=output_path,
            checkpoint_metadata=checkpoint_metadata,
            max_length=max_length,
            overflow_handling=overflow_handling,
            scored_samples=scored_samples,
            tokenization_stats=tokenization_stats,
            sweep_result=sweep_result,
        )
        write_json(output_path, payload)
        print_summary(payload)
    except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
