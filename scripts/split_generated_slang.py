#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split generated_slang_euphemisms.txt into a random sample and the remainder."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/generated_euphemisms.txt"),
        help="Path to the source slang list.",
    )
    parser.add_argument(
        "--output-sample",
        type=Path,
        default=Path("data/generated_euphemisms2.txt"),
        help="Path to the sampled output file.",
    )
    parser.add_argument(
        "--output-rest",
        type=Path,
        default=Path("data/generated_euphemisms1.txt"),
        help="Path to the remainder output file.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=66,
        help="Number of lines to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    lines = read_lines(args.input)

    if args.sample_size > len(lines):
        raise ValueError(
            f"sample size {args.sample_size} is larger than input size {len(lines)}"
        )

    rng = random.Random(args.seed)
    selected_indices = set(rng.sample(range(len(lines)), args.sample_size))

    # Keep the original file order while splitting by randomly chosen line indices.
    sample = [line for idx, line in enumerate(lines) if idx in selected_indices]
    rest = [line for idx, line in enumerate(lines) if idx not in selected_indices]

    write_lines(args.output_sample, sample)
    write_lines(args.output_rest, rest)

    print(f"input: {args.input} ({len(lines)} lines)")
    print(f"sample: {args.output_sample} ({len(sample)} lines)")
    print(f"rest: {args.output_rest} ({len(rest)} lines)")
    print(f"seed: {args.seed}")


if __name__ == "__main__":
    main()
