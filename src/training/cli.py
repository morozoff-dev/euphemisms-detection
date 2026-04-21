from __future__ import annotations

import argparse

from src.training.dataset import build_training_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare BIO train/val/test splits for euphemism sequence labeling."
    )
    parser.add_argument(
        "--positives-path",
        default="outputs/synthetic/data.json",
        help="Path to synthetic positive JSON dataset.",
    )
    parser.add_argument(
        "--negatives-path",
        default="data/negatives.txt",
        help="Path to negative texts, one text per line.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/training/bio_dataset",
        help="Directory where prepared split files will be stored.",
    )
    parser.add_argument(
        "--positive-limit",
        type=int,
        default=None,
        help="Maximum number of positive samples to keep after sampling.",
    )
    parser.add_argument(
        "--negative-limit",
        type=int,
        default=None,
        help="Maximum number of negative samples to keep after sampling.",
    )
    parser.add_argument(
        "--positive-fraction",
        type=float,
        default=None,
        help="Fraction of positive samples to keep, in (0, 1].",
    )
    parser.add_argument(
        "--negative-fraction",
        type=float,
        default=None,
        help="Fraction of negative samples to keep, in (0, 1].",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling and split generation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_training_dataset(
        positives_path=args.positives_path,
        negatives_path=args.negatives_path,
        output_dir=args.output_dir,
        positive_limit=args.positive_limit,
        negative_limit=args.negative_limit,
        positive_fraction=args.positive_fraction,
        negative_fraction=args.negative_fraction,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    counts = manifest["counts"]
    print(f"Prepared BIO dataset in {args.output_dir}")
    print(
        "After sampling: "
        f"{counts['after_sampling']['positive']} positive, "
        f"{counts['after_sampling']['negative']} negative"
    )
    print(
        "Dropped empty-token samples: "
        f"{counts['dropped_empty_token_samples']['positive']} positive, "
        f"{counts['dropped_empty_token_samples']['negative']} negative"
    )
    print(
        "Alignment warnings: "
        f"{counts['alignment_warning_count']}"
    )
    for split_name in ("train", "val", "test"):
        split_counts = counts["splits"][split_name]
        print(
            f"{split_name}: total={split_counts['total']}, "
            f"positive={split_counts['positive']}, "
            f"negative={split_counts['negative']}"
        )
    return 0
