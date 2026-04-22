from __future__ import annotations

import argparse

from src.bio.converter import (
    DEFAULT_BIO_OUTPUT_DIR,
    DEFAULT_DATA_PREP_SPLITS_DIR,
    build_bio_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert pre-split data preparation JSON files into BIO train/val/test files."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_DATA_PREP_SPLITS_DIR,
        help="Directory with train.json/val.json/test.json from src.data_prep.",
    )
    parser.add_argument(
        "--train-path",
        default=None,
        help="Explicit train split JSON path when input_dir is not used.",
    )
    parser.add_argument(
        "--val-path",
        default=None,
        help="Explicit validation split JSON path when input_dir is not used.",
    )
    parser.add_argument(
        "--test-path",
        default=None,
        help="Explicit test split JSON path when input_dir is not used.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_BIO_OUTPUT_DIR,
        help="Directory where BIO jsonl files will be stored.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    use_explicit_paths = any(
        path is not None for path in (args.train_path, args.val_path, args.test_path)
    )
    manifest = build_bio_dataset(
        input_dir=None if use_explicit_paths else args.input_dir,
        train_path=args.train_path,
        val_path=args.val_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
    )
    print(f"Prepared BIO dataset in {args.output_dir}")
    counts = manifest["counts"]["output_splits"]
    print(
        "Dropped empty-token samples: "
        f"{manifest['counts']['dropped_empty_token_samples']['positive']} positive, "
        f"{manifest['counts']['dropped_empty_token_samples']['negative']} negative"
    )
    print(
        "Alignment warnings: "
        f"{manifest['counts']['alignment_warning_count']}"
    )
    for split_name in ("train", "val", "test"):
        split_counts = counts[split_name]
        print(
            f"{split_name}: total={split_counts['total']}, "
            f"positive={split_counts['positive']}, "
            f"negative={split_counts['negative']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
