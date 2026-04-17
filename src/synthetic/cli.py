from __future__ import annotations

import argparse

from src.synthetic.generator import build_data_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a synthetic euphemism dataset from raw text files."
    )
    parser.add_argument(
        "--texts-path",
        default="data/drug_texts_small.txt",
        help="Path to corpus texts, one text per line.",
    )
    parser.add_argument(
        "--euphemisms-path",
        default="data/real_euphemisms.txt",
        help="Path to euphemism vocabulary, one item per line.",
    )
    parser.add_argument(
        "--target-words-path",
        default="data/target_keywords_forms_drug.txt",
        help="Path to all target drug forms, one item per line.",
    )
    parser.add_argument(
        "--output-path",
        default="outputs/synthetic/data.json",
        help="Where to store the generated JSON payload.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for euphemism sampling.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_data_json(
        texts_path=args.texts_path,
        euphemisms_path=args.euphemisms_path,
        target_words_path=args.target_words_path,
        output_path=args.output_path,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
