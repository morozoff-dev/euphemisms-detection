from __future__ import annotations

import argparse
import sys

from src.bio import DEFAULT_BIO_OUTPUT_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a ModernBERT/RuModernBERT token classification baseline."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_BIO_OUTPUT_DIR,
        help="Directory with train.jsonl/val.jsonl/test.jsonl splits.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/models/rumodernbert_baseline",
        help="Directory where checkpoints, metrics, and predictions will be stored.",
    )
    parser.add_argument(
        "--model-name",
        default="deepvk/RuModernBERT-base",
        help="Hugging Face model id or local path for the encoder checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=None,
        help="Optional Hugging Face tokenizer id or local path. "
        "If omitted, the model name/path is used.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional model revision on Hugging Face Hub.",
    )
    parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help="Optional tokenizer revision on Hugging Face Hub. "
        "For deepvk/RuModernBERT-* the CLI uses patched-tokenizer by default.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional local directory for downloaded model files.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--overflow-handling",
        choices=("drop", "truncate"),
        default="drop",
        help="What to do with samples longer than max length.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=8,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=16,
        help="Per-device evaluation batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio over total training steps.",
    )
    parser.add_argument(
        "--grad-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        default="cuda:1",
        help="Device string for PyTorch, for example auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="no",
        help="Optional mixed precision mode for CUDA training.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print training logs every N batches.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional limit for a reproducible train subset.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional limit for a reproducible validation subset.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional limit for a reproducible test subset.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        from src.models.baseline import BaselineTrainingConfig, train_baseline_model
    except ModuleNotFoundError as exc:
        print(
            "Missing training dependencies. Install PyTorch and transformers first, "
            "for example: venv/bin/pip install torch 'transformers>=4.48.0'",
            file=sys.stderr,
        )
        print(f"Original import error: {exc}", file=sys.stderr)
        return 1

    config = BaselineTrainingConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        cache_dir=args.cache_dir,
        max_length=args.max_length,
        overflow_handling=args.overflow_handling,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        device=args.device,
        mixed_precision=args.mixed_precision,
        num_workers=args.num_workers,
        log_every=args.log_every,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )

    try:
        train_baseline_model(config)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
