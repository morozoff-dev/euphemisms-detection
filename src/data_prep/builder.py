from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.data.io import load_lines, write_json
from src.data.text import WORD_RE, lookup_norm, nfc
from src.data_prep.morphology import (
    can_inflect_to_plural,
    get_word_number,
    inflect_like,
)
from src.data_prep.targets import build_target_forms


@dataclass(frozen=True)
class EuphemismAnnotation:
    target_word: str
    target_lemma: str
    base_euphemism: str
    euphemism: str
    source_start: int
    source_end: int
    start: int
    end: int


@dataclass(frozen=True)
class ReplacedTextSample:
    text: str
    replaced_text: str
    euphemisms: list[EuphemismAnnotation]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["euphemisms"] = [asdict(item) for item in self.euphemisms]
        return payload


@dataclass(frozen=True)
class EuphemismEntry:
    text: str
    number: str | None
    can_be_pluralized: bool


@dataclass(frozen=True)
class PreparedEntitySpan:
    start: int
    end: int
    text: str
    label: str = "EUPHEMISM"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DataSplitRecord:
    sample_id: str
    source: str
    source_index: int
    text: str
    entities: list[PreparedEntitySpan]
    original_text: str | None = None
    euphemisms: list[EuphemismAnnotation] | None = None

    def to_dict(self) -> dict:
        payload = {
            "sample_id": self.sample_id,
            "source": self.source,
            "source_index": self.source_index,
            "text": self.text,
            "entities": [entity.to_dict() for entity in self.entities],
        }
        if self.original_text is not None:
            payload["original_text"] = self.original_text
        if self.euphemisms is not None:
            payload["euphemisms"] = [asdict(item) for item in self.euphemisms]
        return payload


DEFAULT_TEXTS_PATH = "data/drug_texts_small.txt"
DEFAULT_TARGET_WORDS_PATH = "data/target_keywords_forms_drug.txt"
DEFAULT_NEGATIVES_PATH = "data/negatives.txt"
DEFAULT_DATA_PREP_OUTPUT_DIR = "outputs/data_prep/splits"
DEFAULT_TRAIN_EUPHEMISMS_PATHS = (
    "data/generated_euphemisms.txt",
    "data/generated_slang_euphemisms.txt",
)
DEFAULT_TEST_EUPHEMISMS_PATHS = ("data/real_euphemisms.txt",)
DEFAULT_POSITIVE_LIMIT = 10000
DEFAULT_NEGATIVE_LIMIT = 2000


def is_number_compatible(
    euphemism: EuphemismEntry,
    *,
    target_number: str | None,
) -> bool:
    if target_number == "sing":
        return euphemism.number != "plur"
    if target_number == "plur":
        return euphemism.number == "plur" or euphemism.can_be_pluralized
    return True


def is_replacement_number_compatible(
    replacement: str,
    *,
    target_number: str | None,
) -> bool:
    replacement_number = get_word_number(replacement)
    if target_number == "sing":
        return replacement_number != "plur"
    if target_number == "plur":
        return replacement_number == "plur"
    return True


def choose_euphemism(
    euphemisms: list[EuphemismEntry],
    *,
    target_word: str,
    target_lemma: str,
    target_number: str | None,
    used_in_text: set[str],
    rng: random.Random,
) -> tuple[EuphemismEntry, str]:
    compatible = [
        euphemism
        for euphemism in euphemisms
        if is_number_compatible(euphemism, target_number=target_number)
    ]
    if not compatible:
        raise ValueError(
            f"No number-compatible euphemism found for target number {target_number!r}."
        )

    available = [
        euphemism for euphemism in compatible if euphemism.text not in used_in_text
    ]

    for pool in (available, compatible):
        if not pool:
            continue

        shuffled = pool[:]
        rng.shuffle(shuffled)
        for euphemism in shuffled:
            replacement = inflect_like(euphemism.text, target_word, target_lemma)
            if is_replacement_number_compatible(
                replacement,
                target_number=target_number,
            ):
                return euphemism, replacement

    raise ValueError(
        f"No euphemism produced a number-compatible replacement for {target_word!r}."
    )


def replace_in_text(
    text: str,
    *,
    form_to_lemma: dict[str, str],
    euphemisms: list[EuphemismEntry],
    rng: random.Random,
) -> ReplacedTextSample:
    normalized_text = nfc(text)
    lemma_to_euphemism: dict[str, EuphemismEntry] = {}
    used_euphemisms_in_text: set[str] = set()

    replaced_parts: list[str] = []
    annotations: list[EuphemismAnnotation] = []

    last_pos = 0
    replaced_cursor = 0

    for match in WORD_RE.finditer(normalized_text):
        start, end = match.span()
        token = match.group(0)

        prefix = normalized_text[last_pos:start]
        replaced_parts.append(prefix)
        replaced_cursor += len(prefix)

        token_norm = lookup_norm(token)
        target_lemma = form_to_lemma.get(token_norm)

        if target_lemma is None:
            replaced_parts.append(token)
            replaced_cursor += len(token)
            last_pos = end
            continue

        target_number = get_word_number(token, expected_lemma=target_lemma)
        selected_euphemism = lemma_to_euphemism.get(target_lemma)
        replacement = None

        if selected_euphemism is not None and is_number_compatible(
            selected_euphemism,
            target_number=target_number,
        ):
            candidate_replacement = inflect_like(
                selected_euphemism.text,
                token,
                target_lemma,
            )
            if is_replacement_number_compatible(
                candidate_replacement,
                target_number=target_number,
            ):
                replacement = candidate_replacement

        if replacement is None:
            selected_euphemism, replacement = choose_euphemism(
                euphemisms,
                target_word=token,
                target_lemma=target_lemma,
                target_number=target_number,
                used_in_text=used_euphemisms_in_text,
                rng=rng,
            )
            lemma_to_euphemism[target_lemma] = selected_euphemism
            used_euphemisms_in_text.add(selected_euphemism.text)
        replacement_start = replaced_cursor
        replacement_end = replacement_start + len(replacement)

        replaced_parts.append(replacement)
        replaced_cursor = replacement_end

        annotations.append(
            EuphemismAnnotation(
                target_word=token,
                target_lemma=target_lemma,
                base_euphemism=selected_euphemism.text,
                euphemism=replacement,
                source_start=start,
                source_end=end,
                start=replacement_start,
                end=replacement_end,
            )
        )
        last_pos = end

    tail = normalized_text[last_pos:]
    replaced_parts.append(tail)

    return ReplacedTextSample(
        text=normalized_text,
        replaced_text="".join(replaced_parts),
        euphemisms=annotations,
    )


def coerce_path_list(
    path_or_paths: str | Path | Sequence[str | Path],
) -> list[str]:
    if isinstance(path_or_paths, (str, Path)):
        paths = [path_or_paths]
    else:
        paths = list(path_or_paths)
    if not paths:
        raise ValueError("At least one euphemism path must be provided.")
    return [str(Path(path)) for path in paths]


def load_euphemisms(
    path_or_paths: str | Path | Sequence[str | Path],
) -> list[EuphemismEntry]:
    deduplicated = list(
        dict.fromkeys(
            euphemism
            for path in coerce_path_list(path_or_paths)
            for euphemism in load_lines(path)
        )
    )
    if not deduplicated:
        raise ValueError("Euphemism vocabulary is empty.")
    return [
        EuphemismEntry(
            text=euphemism,
            number=get_word_number(euphemism),
            can_be_pluralized=can_inflect_to_plural(euphemism),
        )
        for euphemism in deduplicated
    ]


def build_replaced_samples(
    texts: Iterable[str],
    *,
    form_to_lemma: dict[str, str],
    euphemisms: list[EuphemismEntry],
    seed: int | None = 42,
) -> list[ReplacedTextSample]:
    rng = random.Random(seed)
    return [
        replace_in_text(
            text,
            form_to_lemma=form_to_lemma,
            euphemisms=euphemisms,
            rng=rng,
        )
        for text in texts
    ]


def build_split_record_from_positive_sample(
    sample: ReplacedTextSample,
    *,
    source_index: int,
) -> DataSplitRecord:
    return DataSplitRecord(
        sample_id=f"positive-{source_index:06d}",
        source="positive",
        source_index=source_index,
        text=sample.replaced_text,
        entities=[
            PreparedEntitySpan(
                start=annotation.start,
                end=annotation.end,
                text=sample.replaced_text[annotation.start : annotation.end],
            )
            for annotation in sample.euphemisms
        ],
        original_text=sample.text,
        euphemisms=sample.euphemisms,
    )


def build_split_record_from_negative_text(
    text: str,
    *,
    source_index: int,
) -> DataSplitRecord:
    return DataSplitRecord(
        sample_id=f"negative-{source_index:06d}",
        source="negative",
        source_index=source_index,
        text=text,
        entities=[],
    )


def contains_target_form(
    text: str,
    *,
    form_to_lemma: dict[str, str],
) -> bool:
    normalized_text = nfc(text)
    return any(
        lookup_norm(match.group(0)) in form_to_lemma
        for match in WORD_RE.finditer(normalized_text)
    )


def choose_sample_size(
    total_size: int,
    *,
    limit: int | None,
) -> int:
    if limit is None:
        return total_size
    if limit < 0:
        raise ValueError("Sample limit must be non-negative.")
    return min(total_size, limit)


def sample_records(
    records: list[str],
    *,
    limit: int | None,
    rng: random.Random,
) -> list[str]:
    sample_size = choose_sample_size(len(records), limit=limit)
    if sample_size == len(records):
        return list(records)
    return rng.sample(records, sample_size)


def validate_split_ratios(
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError("Train/val/test ratios must sum to 1.0.")


def compute_split_counts(total_size: int, ratios: list[float]) -> list[int]:
    raw_counts = [total_size * ratio for ratio in ratios]
    counts = [int(value) for value in raw_counts]
    remaining = total_size - sum(counts)

    remainders = sorted(
        (
            (raw_counts[index] - counts[index], index)
            for index in range(len(ratios))
        ),
        reverse=True,
    )

    for _, index in remainders[:remaining]:
        counts[index] += 1
    return counts


def split_texts(
    texts: list[str],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[str]]:
    shuffled = list(texts)
    rng.shuffle(shuffled)

    train_count, val_count, test_count = compute_split_counts(
        len(shuffled),
        [train_ratio, val_ratio, test_ratio],
    )
    train_end = train_count
    val_end = train_end + val_count

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end : val_end + test_count],
    }


def build_dataset_splits(
    *,
    texts_path: str = DEFAULT_TEXTS_PATH,
    negatives_path: str = DEFAULT_NEGATIVES_PATH,
    train_euphemisms_paths: Sequence[str] = DEFAULT_TRAIN_EUPHEMISMS_PATHS,
    test_euphemisms_paths: Sequence[str] = DEFAULT_TEST_EUPHEMISMS_PATHS,
    val_euphemisms_paths: Sequence[str] | None = None,
    target_words_path: str = DEFAULT_TARGET_WORDS_PATH,
    output_dir: str = DEFAULT_DATA_PREP_OUTPUT_DIR,
    positive_limit: int | None = DEFAULT_POSITIVE_LIMIT,
    negative_limit: int | None = DEFAULT_NEGATIVE_LIMIT,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int | None = 42,
) -> dict:
    validate_split_ratios(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    texts = load_lines(texts_path)
    negatives = load_lines(negatives_path)
    target_words = load_lines(target_words_path)
    form_to_lemma = build_target_forms(target_words)
    rng = random.Random(seed)
    candidate_positive_texts = [
        text
        for text in texts
        if contains_target_form(text, form_to_lemma=form_to_lemma)
    ]
    sampled_positive_texts = sample_records(
        candidate_positive_texts,
        limit=positive_limit,
        rng=rng,
    )
    sampled_negative_texts = sample_records(
        negatives,
        limit=negative_limit,
        rng=rng,
    )

    positive_text_splits = split_texts(
        sampled_positive_texts,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        rng=rng,
    )
    negative_text_splits = split_texts(
        sampled_negative_texts,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        rng=rng,
    )

    split_euphemism_paths = {
        "train": coerce_path_list(train_euphemisms_paths),
        "val": coerce_path_list(
            val_euphemisms_paths
            if val_euphemisms_paths is not None
            else train_euphemisms_paths
        ),
        "test": coerce_path_list(test_euphemisms_paths),
    }

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    split_counts: dict[str, dict[str, int | str | list[str]]] = {}
    for offset, split_name in enumerate(("train", "val", "test")):
        euphemisms = load_euphemisms(split_euphemism_paths[split_name])
        positive_samples = build_replaced_samples(
            positive_text_splits[split_name],
            form_to_lemma=form_to_lemma,
            euphemisms=euphemisms,
            seed=None if seed is None else seed + offset,
        )
        positive_records = [
            build_split_record_from_positive_sample(
                sample,
                source_index=index,
            )
            for index, sample in enumerate(positive_samples)
            if sample.euphemisms
        ]
        negative_records = [
            build_split_record_from_negative_text(
                text,
                source_index=index,
            )
            for index, text in enumerate(negative_text_splits[split_name])
        ]
        split_rows = positive_records + negative_records
        rng.shuffle(split_rows)

        output_path = destination / f"{split_name}.json"
        write_json(output_path, [record.to_dict() for record in split_rows])
        split_counts[split_name] = {
            "positive_input_texts": len(positive_text_splits[split_name]),
            "positive_samples": len(positive_records),
            "negative_samples": len(negative_records),
            "total": len(split_rows),
            "output_path": str(output_path),
            "euphemism_paths": split_euphemism_paths[split_name],
        }

    manifest = {
        "seed": seed,
        "input_paths": {
            "positive_texts": texts_path,
            "negative_texts": negatives_path,
            "target_words": target_words_path,
        },
        "sampling": {
            "positive_limit": positive_limit,
            "negative_limit": negative_limit,
        },
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "counts": {
            "before_sampling": {
                "positive_candidates": len(candidate_positive_texts),
                "negative": len(negatives),
                "total": len(candidate_positive_texts) + len(negatives),
            },
            "after_sampling": {
                "positive": len(sampled_positive_texts),
                "negative": len(sampled_negative_texts),
                "total": len(sampled_positive_texts) + len(sampled_negative_texts),
            },
        },
        "splits": split_counts,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest
