from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scripts.infer_one_text import (
    resolve_checkpoint_dir,
    resolve_device,
    resolve_max_length,
)
from src.evaluation.real_euph import (
    DEFAULT_INPUT_JSON,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TARGET_KEYWORDS_PATH,
    evaluate_real_euphemisms,
    write_evaluation_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a token-classification checkpoint on verified real "
            "euphemism annotations from JSON."
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help=(
            "Path to a training run directory or directly to best_model/. "
            "Examples: outputs/models/<run_name> or outputs/models/<run_name>/best_model"
        ),
    )
    parser.add_argument(
        "--input-json",
        default=DEFAULT_INPUT_JSON,
        help="Path to real euphemism JSON. Only verified=true records are used.",
    )
    parser.add_argument(
        "--target-keywords-path",
        default=DEFAULT_TARGET_KEYWORDS_PATH,
        help="Path to target keyword forms to ignore in gold and predictions.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where metrics, predictions, and analysis files are saved.",
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
        "--max-length",
        type=int,
        default=None,
        help=(
            "Max transformer sequence length. If omitted, the CLI tries to read "
            "it from run_config.json and falls back to 256."
        ),
    )
    parser.add_argument(
        "--window-overlap-words",
        type=int,
        default=32,
        help="How many token units to overlap between inference windows for long texts.",
    )
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.5,
        help=(
            "Positive-class probability threshold. A token is predicted as "
            "EUPHEMISM only when P(EUPHEMISM) is strictly greater than this value."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        import torch
        from transformers import AutoTokenizer
        from src.models import load_token_classifier_checkpoint
    except ModuleNotFoundError as exc:
        print(
            "Missing evaluation dependencies. Install PyTorch and transformers first, "
            "for example: venv/bin/pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Original import error: {exc}", file=sys.stderr)
        return 1

    try:
        checkpoint_dir, run_dir = resolve_checkpoint_dir(args.model_dir)
        max_length = resolve_max_length(
            cli_value=args.max_length,
            run_dir=run_dir,
        )
        if not 0.0 <= args.prediction_threshold <= 1.0:
            raise ValueError("--prediction-threshold must be between 0 and 1.")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        if not tokenizer.is_fast:
            raise RuntimeError(
                "A fast tokenizer is required for token classification inference."
            )

        model, checkpoint_metadata = load_token_classifier_checkpoint(
            checkpoint_dir,
            requested_head_mode=args.head_mode,
        )
        device = resolve_device(args.device, torch)
        model.to(device)

        result = evaluate_real_euphemisms(
            input_json=args.input_json,
            target_keywords_path=args.target_keywords_path,
            model=model,
            tokenizer=tokenizer,
            torch_module=torch,
            device=device,
            checkpoint_dir=checkpoint_dir,
            run_dir=run_dir,
            checkpoint_metadata=checkpoint_metadata,
            max_length=max_length,
            window_overlap_words=args.window_overlap_words,
            prediction_threshold=args.prediction_threshold,
        )
        write_evaluation_outputs(args.output_dir, result)

        metrics = result["metrics"]
        counts = result["counts"]
        print(f"Saved real euphemism evaluation to: {Path(args.output_dir)}")
        print(f"Model: {checkpoint_dir}")
        if run_dir is not None:
            print(f"Run directory: {run_dir}")
        print(f"Head mode: {result['head_mode']}")
        print(f"Checkpoint architecture: {result['checkpoint_architecture']}")
        print(f"Input JSON: {Path(args.input_json)}")
        print(f"Verified samples: {counts['verified_records']}")
        print(f"Gold entities: {counts['gold_entities']}")
        print(f"Ignored target tokens: {counts['ignored_target_tokens']}")
        print(
            "Token metrics: "
            f"P={metrics['token_precision']:.4f} "
            f"R={metrics['token_recall']:.4f} "
            f"F1={metrics['token_f1']:.4f}"
        )
        print(
            "Span metrics: "
            f"P={metrics['span_precision']:.4f} "
            f"R={metrics['span_recall']:.4f} "
            f"F1={metrics['span_f1']:.4f}"
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0
