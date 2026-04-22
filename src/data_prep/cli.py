from __future__ import annotations

import argparse

from src.data_prep.builder import (
    DEFAULT_NEGATIVE_LIMIT,
    DEFAULT_NEGATIVES_PATH,
    DEFAULT_POSITIVE_LIMIT,
    DEFAULT_DATA_PREP_OUTPUT_DIR,
    DEFAULT_OBSERVED_EUPHEMISMS_PATHS,
    DEFAULT_TARGET_REPLACEMENT_FRACTION,
    DEFAULT_TEST_EUPHEMISMS_PATHS,
    DEFAULT_TRAIN_EUPHEMISMS_PATHS,
    build_dataset_splits,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test dataset splits with positives and negatives."
    )
    parser.add_argument(
        "--texts-path",
        default="data/drug_texts_small.txt",
        help="Path to positive corpus texts, one text per line.",
    )
    parser.add_argument(
        "--negatives-path",
        default=DEFAULT_NEGATIVES_PATH,
        help="Path to negative texts, one text per line.",
    )
    parser.add_argument(
        "--target-words-path",
        default="data/target_keywords_forms_drug.txt",
        help="Path to all target drug forms, one item per line.",
    )
    parser.add_argument(
        "--observed-euphemisms-paths",
        nargs="+",
        default=DEFAULT_OBSERVED_EUPHEMISMS_PATHS,
        help="One or more vocab files whose forms will be searched in positive source "
        "texts and annotated without replacement.",
    )
    parser.add_argument(
        "--train-euphemisms-paths",
        nargs="+",
        default=DEFAULT_TRAIN_EUPHEMISMS_PATHS,
        help="One or more euphemism vocabulary files for train positives.",
    )
    parser.add_argument(
        "--val-euphemisms-paths",
        nargs="+",
        default=None,
        help="One or more euphemism vocabulary files for validation positives. "
        "Defaults to the train list.",
    )
    parser.add_argument(
        "--test-euphemisms-paths",
        nargs="+",
        default=DEFAULT_TEST_EUPHEMISMS_PATHS,
        help="One or more euphemism vocabulary files for test positives.",
    )
    parser.add_argument(
        "--positive-limit",
        type=int,
        default=DEFAULT_POSITIVE_LIMIT,
        help="Maximum number of positive source texts to sample before splitting.",
    )
    parser.add_argument(
        "--negative-limit",
        type=int,
        default=DEFAULT_NEGATIVE_LIMIT,
        help="Maximum number of negative texts to sample before splitting.",
    )
    parser.add_argument(
        "--target-replacement-fraction",
        type=float,
        default=DEFAULT_TARGET_REPLACEMENT_FRACTION,
        help="Fraction of target keyword mentions to replace inside each positive text.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_DATA_PREP_OUTPUT_DIR,
        help="Directory where train.json/val.json/test.json will be stored.",
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
        help="Random seed used for sampling, splitting, and euphemism selection.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_dataset_splits(
        texts_path=args.texts_path,
        negatives_path=args.negatives_path,
        observed_euphemisms_paths=args.observed_euphemisms_paths,
        train_euphemisms_paths=args.train_euphemisms_paths,
        val_euphemisms_paths=args.val_euphemisms_paths,
        test_euphemisms_paths=args.test_euphemisms_paths,
        target_words_path=args.target_words_path,
        output_dir=args.output_dir,
        target_replacement_fraction=args.target_replacement_fraction,
        positive_limit=args.positive_limit,
        negative_limit=args.negative_limit,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Prepared dataset splits in {args.output_dir}")
    counts = manifest["counts"]
    print(
        "After sampling: "
        f"{counts['after_sampling']['positive']} positive, "
        f"{counts['after_sampling']['negative']} negative"
    )
    for split_name in ("train", "val", "test"):
        split_info = manifest["splits"][split_name]
        print(
            f"{split_name}: total={split_info['total']}, "
            f"positive={split_info['positive_samples']}, "
            f"negative={split_info['negative_samples']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
