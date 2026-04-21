from __future__ import annotations

import json
import random
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.data.io import load_lines, resolve_input_path, write_json
from src.data.text import WORD_RE

ENTITY_LABEL = "EUPHEMISM"
BIO_LABELS = ["O", f"B-{ENTITY_LABEL}", f"I-{ENTITY_LABEL}"]
MAX_ALIGNMENT_WARNING_EXAMPLES = 20


@dataclass(frozen=True)
class EntitySpan:
    start: int
    end: int
    text: str
    label: str = ENTITY_LABEL

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TokenSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class SourceSample:
    sample_id: str
    source: str
    source_index: int
    text: str
    entities: list[EntitySpan]


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    source: str
    source_index: int
    text: str
    tokens: list[str]
    bio_tags: list[str]
    entities: list[EntitySpan]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["entities"] = [entity.to_dict() for entity in self.entities]
        return payload


@dataclass
class AlignmentWarningTracker:
    total_count: int = 0
    emitted_examples: int = 0
    max_examples: int = MAX_ALIGNMENT_WARNING_EXAMPLES

    def warn(
        self,
        *,
        sample_id: str,
        entity: EntitySpan,
        covered_tokens: list[TokenSpan],
        text: str,
    ) -> None:
        self.total_count += 1
        if self.emitted_examples >= self.max_examples:
            return

        context_start = max(0, entity.start - 40)
        context_end = min(len(text), entity.end + 40)
        context = text[context_start:context_end]
        token_texts = [token.text for token in covered_tokens]
        warnings.warn(
            "Осторожнее: char-level span нечётко совпадает с границами токенов. "
            "BIO-разметка будет построена по overlap-логике. "
            f"sample_id={sample_id}, span=[{entity.start}, {entity.end}), "
            f"entity_text={entity.text!r}, covered_tokens={token_texts!r}, "
            f"context={context!r}",
            stacklevel=2,
        )
        self.emitted_examples += 1


class PreparedBioDataset(Sequence[PreparedSample]):
    def __init__(self, samples: list[PreparedSample]) -> None:
        self._samples = samples

    def __getitem__(self, index: int) -> PreparedSample:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PreparedBioDataset":
        dataset_path = resolve_input_path(path)
        samples: list[PreparedSample] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                samples.append(
                    PreparedSample(
                        sample_id=payload["sample_id"],
                        source=payload["source"],
                        source_index=payload["source_index"],
                        text=payload["text"],
                        tokens=payload["tokens"],
                        bio_tags=payload["bio_tags"],
                        entities=[
                            EntitySpan(**entity) for entity in payload["entities"]
                        ],
                    )
                )
        return cls(samples)

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        *,
        split: str,
    ) -> "PreparedBioDataset":
        return cls.from_jsonl(Path(directory) / f"{split}.jsonl")


def load_positive_samples(path: str | Path) -> list[SourceSample]:
    resolved = resolve_input_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))

    samples: list[SourceSample] = []
    for index, item in enumerate(payload):
        text = item["replaced_text"]
        entities = [
            EntitySpan(
                start=annotation["start"],
                end=annotation["end"],
                text=text[annotation["start"] : annotation["end"]],
            )
            for annotation in item.get("euphemisms", [])
        ]
        if not entities:
            continue
        samples.append(
            SourceSample(
                sample_id=f"positive-{index:06d}",
                source="positive",
                source_index=index,
                text=text,
                entities=sorted(entities, key=lambda entity: (entity.start, entity.end)),
            )
        )
    return samples


def load_negative_samples(path: str | Path) -> list[SourceSample]:
    return [
        SourceSample(
            sample_id=f"negative-{index:06d}",
            source="negative",
            source_index=index,
            text=text,
            entities=[],
        )
        for index, text in enumerate(load_lines(path))
    ]


def tokenize_words(text: str) -> list[TokenSpan]:
    return [
        TokenSpan(text=match.group(0), start=match.start(), end=match.end())
        for match in WORD_RE.finditer(text)
    ]


def is_exact_token_alignment(
    covered_tokens: list[TokenSpan],
    *,
    entity: EntitySpan,
) -> bool:
    if not covered_tokens:
        return False
    if covered_tokens[0].start != entity.start:
        return False
    if covered_tokens[-1].end != entity.end:
        return False
    return all(
        token.start >= entity.start and token.end <= entity.end
        for token in covered_tokens
    )


def build_bio_tags(
    tokens: list[TokenSpan],
    entities: list[EntitySpan],
    *,
    sample_id: str,
    text: str,
    warning_tracker: AlignmentWarningTracker | None = None,
) -> list[str]:
    labels = ["O"] * len(tokens)

    for entity in sorted(entities, key=lambda item: (item.start, item.end)):
        covered_indices: list[int] = []
        covered_tokens: list[TokenSpan] = []
        for index, token in enumerate(tokens):
            if token.start < entity.end and token.end > entity.start:
                covered_indices.append(index)
                covered_tokens.append(token)
        if not covered_indices:
            raise ValueError(
                f"Entity span [{entity.start}, {entity.end}) is not aligned to any token."
            )
        if not is_exact_token_alignment(covered_tokens, entity=entity):
            if warning_tracker is not None:
                warning_tracker.warn(
                    sample_id=sample_id,
                    entity=entity,
                    covered_tokens=covered_tokens,
                    text=text,
                )
        for offset, token_index in enumerate(covered_indices):
            if labels[token_index] != "O":
                raise ValueError(
                    f"Overlapping entities detected around token index {token_index}."
                )
            labels[token_index] = (
                f"B-{ENTITY_LABEL}" if offset == 0 else f"I-{ENTITY_LABEL}"
            )

    return labels


def prepare_sample(
    sample: SourceSample,
    *,
    warning_tracker: AlignmentWarningTracker | None = None,
) -> PreparedSample:
    tokens = tokenize_words(sample.text)
    if not tokens:
        raise ValueError(f"Sample {sample.sample_id} does not contain any word tokens.")
    bio_tags = build_bio_tags(
        tokens,
        sample.entities,
        sample_id=sample.sample_id,
        text=sample.text,
        warning_tracker=warning_tracker,
    )
    return PreparedSample(
        sample_id=sample.sample_id,
        source=sample.source,
        source_index=sample.source_index,
        text=sample.text,
        tokens=[token.text for token in tokens],
        bio_tags=bio_tags,
        entities=sample.entities,
    )


def choose_sample_size(
    total_size: int,
    *,
    limit: int | None,
    fraction: float | None,
) -> int:
    if limit is not None and fraction is not None:
        raise ValueError("Use either a limit or a fraction for sampling, not both.")
    if limit is not None:
        if limit < 0:
            raise ValueError("Sample limit must be non-negative.")
        return min(total_size, limit)
    if fraction is None:
        return total_size
    if not 0 < fraction <= 1:
        raise ValueError("Sample fraction must be in the (0, 1] interval.")
    sample_size = int(total_size * fraction)
    if total_size > 0 and sample_size == 0:
        return 1
    return sample_size


def sample_records(
    records: list[SourceSample],
    *,
    limit: int | None,
    fraction: float | None,
    rng: random.Random,
) -> list[SourceSample]:
    sample_size = choose_sample_size(len(records), limit=limit, fraction=fraction)
    if sample_size == len(records):
        return list(records)
    sampled = rng.sample(records, sample_size)
    sampled.sort(key=lambda item: item.source_index)
    return sampled


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


def split_records(
    records: list[SourceSample],
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[SourceSample]]:
    shuffled = list(records)
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


def merge_split_buckets(
    positive_splits: dict[str, list[SourceSample]],
    negative_splits: dict[str, list[SourceSample]],
    *,
    rng: random.Random,
) -> dict[str, list[SourceSample]]:
    merged: dict[str, list[SourceSample]] = {}
    for split_name in ("train", "val", "test"):
        combined = positive_splits[split_name] + negative_splits[split_name]
        rng.shuffle(combined)
        merged[split_name] = combined
    return merged


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def build_manifest(
    *,
    positives_total: int,
    negatives_total: int,
    positives_sampled: int,
    negatives_sampled: int,
    split_rows: dict[str, list[PreparedSample]],
    dropped_empty_token_samples: dict[str, int],
    alignment_warning_count: int,
    positives_path: str | Path,
    negatives_path: str | Path,
    positive_limit: int | None,
    negative_limit: int | None,
    positive_fraction: float | None,
    negative_fraction: float | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    split_counts = {
        split_name: {
            "total": len(rows),
            "positive": sum(row.source == "positive" for row in rows),
            "negative": sum(row.source == "negative" for row in rows),
            "entity_tokens": sum(
                tag != "O" for row in rows for tag in row.bio_tags
            ),
        }
        for split_name, rows in split_rows.items()
    }

    return {
        "seed": seed,
        "entity_label": ENTITY_LABEL,
        "bio_labels": BIO_LABELS,
        "input_paths": {
            "positives": str(positives_path),
            "negatives": str(negatives_path),
        },
        "sampling": {
            "positive_limit": positive_limit,
            "negative_limit": negative_limit,
            "positive_fraction": positive_fraction,
            "negative_fraction": negative_fraction,
        },
        "split_ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "counts": {
            "before_sampling": {
                "positive": positives_total,
                "negative": negatives_total,
                "total": positives_total + negatives_total,
            },
            "after_sampling": {
                "positive": positives_sampled,
                "negative": negatives_sampled,
                "total": positives_sampled + negatives_sampled,
            },
            "dropped_empty_token_samples": dropped_empty_token_samples,
            "alignment_warning_count": alignment_warning_count,
            "splits": split_counts,
        },
    }


def build_training_dataset(
    *,
    positives_path: str = "outputs/synthetic/data.json",
    negatives_path: str = "data/negatives.txt",
    output_dir: str = "outputs/training/bio_dataset",
    positive_limit: int | None = None,
    negative_limit: int | None = None,
    positive_fraction: float | None = None,
    negative_fraction: float | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    validate_split_ratios(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    rng = random.Random(seed)

    positives = load_positive_samples(positives_path)
    negatives = load_negative_samples(negatives_path)

    sampled_positives = sample_records(
        positives,
        limit=positive_limit,
        fraction=positive_fraction,
        rng=rng,
    )
    sampled_negatives = sample_records(
        negatives,
        limit=negative_limit,
        fraction=negative_fraction,
        rng=rng,
    )

    positive_splits = split_records(
        sampled_positives,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        rng=rng,
    )
    negative_splits = split_records(
        sampled_negatives,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        rng=rng,
    )
    merged_splits = merge_split_buckets(positive_splits, negative_splits, rng=rng)

    prepared_splits: dict[str, list[PreparedSample]] = {}
    dropped_empty_token_samples = {"positive": 0, "negative": 0}
    warning_tracker = AlignmentWarningTracker()
    for split_name, samples in merged_splits.items():
        prepared_rows: list[PreparedSample] = []
        for sample in samples:
            try:
                prepared_rows.append(
                    prepare_sample(sample, warning_tracker=warning_tracker)
                )
            except ValueError as exc:
                if "does not contain any word tokens" not in str(exc):
                    raise
                dropped_empty_token_samples[sample.source] += 1
        prepared_splits[split_name] = prepared_rows

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for split_name, rows in prepared_splits.items():
        write_jsonl(
            destination / f"{split_name}.jsonl",
            [row.to_dict() for row in rows],
        )

    manifest = build_manifest(
        positives_total=len(positives),
        negatives_total=len(negatives),
        positives_sampled=len(sampled_positives),
        negatives_sampled=len(sampled_negatives),
        split_rows=prepared_splits,
        dropped_empty_token_samples=dropped_empty_token_samples,
        alignment_warning_count=warning_tracker.total_count,
        positives_path=positives_path,
        negatives_path=negatives_path,
        positive_limit=positive_limit,
        negative_limit=negative_limit,
        positive_fraction=positive_fraction,
        negative_fraction=negative_fraction,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    write_json(destination / "manifest.json", manifest)
    return manifest
