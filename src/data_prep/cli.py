from __future__ import annotations

import argparse

from src.data_prep.builder import (
    DEFAULT_EXTRA_NEGATIVE_GROUP_NAME,
    DEFAULT_EXTRA_NEGATIVE_TEST_PATH,
    DEFAULT_EXTRA_NEGATIVE_TRAIN_VAL_PATH,
    DEFAULT_NEGATIVE_LIMIT,
    DEFAULT_NEGATIVES_PATH,
    DEFAULT_POSITIVE_LIMIT,
    DEFAULT_DATA_PREP_OUTPUT_DIR,
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
        "--extra-negative-train-val-path",
        default=DEFAULT_EXTRA_NEGATIVE_TRAIN_VAL_PATH,
        help=(
            "Path to extra negative texts that are split only between train and val. "
            "Each line is treated as a gold-negative sample."
        ),
    )
    parser.add_argument(
        "--extra-negative-test-path",
        default=DEFAULT_EXTRA_NEGATIVE_TEST_PATH,
        help=(
            "Path to extra negative texts that are added only to test. "
            "Each line is treated as a gold-negative sample."
        ),
    )
    parser.add_argument(
        "--extra-negative-group-name",
        default=DEFAULT_EXTRA_NEGATIVE_GROUP_NAME,
        help="Group name stored in negative_group for extra negative samples.",
    )
    parser.add_argument(
        "--disable-extra-negative-group",
        action="store_true",
        help="Disable default extra negative group injection and use only --negatives-path.",
    )
    parser.add_argument(
        "--target-words-path",
        default="data/target_keywords_forms_drug.txt",
        help="Path to all target drug forms, one item per line.",
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
        "--one-target-per-text",
        type=float,
        default=None,
        help=(
            "Percentage of positive texts with exactly one target keyword. "
            "If any target-count percentage is set, omitted percentages default to 0."
        ),
    )
    parser.add_argument(
        "--two-targets-per-text",
        type=float,
        default=None,
        help=(
            "Percentage of positive texts with exactly two target keywords. "
            "The 4+ percentage is computed as 100 minus the 1/2/3 percentages."
        ),
    )
    parser.add_argument(
        "--three-targets-per-text",
        type=float,
        default=None,
        help=(
            "Percentage of positive texts with exactly three target keywords. "
            "Texts without target keywords are not included in these percentages."
        ),
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
        extra_negative_train_val_path=args.extra_negative_train_val_path,
        extra_negative_test_path=args.extra_negative_test_path,
        extra_negative_group_name=args.extra_negative_group_name,
        enable_extra_negative_group=not args.disable_extra_negative_group,
        train_euphemisms_paths=args.train_euphemisms_paths,
        val_euphemisms_paths=args.val_euphemisms_paths,
        test_euphemisms_paths=args.test_euphemisms_paths,
        target_words_path=args.target_words_path,
        output_dir=args.output_dir,
        target_replacement_fraction=args.target_replacement_fraction,
        positive_limit=args.positive_limit,
        negative_limit=args.negative_limit,
        one_target_per_text=args.one_target_per_text,
        two_targets_per_text=args.two_targets_per_text,
        three_targets_per_text=args.three_targets_per_text,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Prepared dataset splits in {args.output_dir}")
    preprocessing = manifest["preprocessing"]
    print(
        "After preprocessing: "
        f"{preprocessing['positive_texts']['kept_russian']} positive kept "
        f"(lowercased={preprocessing['positive_texts']['lowercased_mostly_uppercase']}, "
        f"empty={preprocessing['positive_texts']['dropped_empty_after_normalization']}), "
        f"{preprocessing['negative_texts']['kept_russian']} negative kept "
        f"(lowercased={preprocessing['negative_texts']['lowercased_mostly_uppercase']}, "
        f"empty={preprocessing['negative_texts']['dropped_empty_after_normalization']})"
    )
    extra_group = manifest["extra_negative_group"]
    if extra_group["enabled"]:
        print(
            "Extra negative group: "
            f"{extra_group['group_name']} | "
            f"train_val_kept={preprocessing['extra_negative_train_val_texts']['kept_russian']}, "
            f"test_kept={preprocessing['extra_negative_test_texts']['kept_russian']}"
        )
    counts = manifest["counts"]
    print(
        "After sampling: "
        f"{counts['after_sampling']['positive']} positive, "
        f"{counts['after_sampling']['negative']} negative, "
        f"{counts['after_sampling']['extra_negative_train_val']} extra train_val negative, "
        f"{counts['after_sampling']['extra_negative_test']} extra test negative"
    )
    target_count_distribution = manifest["sampling"][
        "positive_target_count_distribution"
    ]
    if target_count_distribution["enabled"]:
        print(
            "Positive target-count sampling: "
            f"requested_percentages={target_count_distribution['requested_percentages']}, "
            f"sampled_counts={target_count_distribution['sampled_counts']}"
        )
    for split_name in ("train", "val", "test"):
        split_info = manifest["splits"][split_name]
        print(
            f"{split_name}: total={split_info['total']}, "
            f"positive={split_info['positive_samples']}, "
            f"negative={split_info['negative_samples']}, "
            f"base_negative={split_info['base_negative_samples']}, "
            f"extra_negative={split_info['extra_negative_group_samples']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
