from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.data.io import resolve_input_path, write_json
from src.data.text import WORD_RE

ENTITY_LABEL = "EUPHEMISM"
BIO_LABELS = ["O", f"B-{ENTITY_LABEL}", f"I-{ENTITY_LABEL}"]
MAX_ALIGNMENT_WARNING_EXAMPLES = 20
DEFAULT_DATA_PREP_SPLITS_DIR = "outputs/data_prep/splits"
DEFAULT_BIO_OUTPUT_DIR = "outputs/bio"


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
class BioSample:
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


class BioDataset(Sequence[BioSample]):
    def __init__(self, samples: list[BioSample]) -> None:
        self._samples = samples

    def __getitem__(self, index: int) -> BioSample:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BioDataset":
        dataset_path = resolve_input_path(path)
        samples: list[BioSample] = []
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                samples.append(
                    BioSample(
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
    ) -> "BioDataset":
        return cls.from_jsonl(Path(directory) / f"{split}.jsonl")


def parse_entities(payload: dict, *, text: str) -> list[EntitySpan]:
    if "entities" in payload:
        return [
            EntitySpan(
                start=entity["start"],
                end=entity["end"],
                text=entity.get("text", text[entity["start"] : entity["end"]]),
                label=entity.get("label", ENTITY_LABEL),
            )
            for entity in payload["entities"]
        ]

    if "euphemisms" in payload:
        return [
            EntitySpan(
                start=annotation["start"],
                end=annotation["end"],
                text=text[annotation["start"] : annotation["end"]],
            )
            for annotation in payload["euphemisms"]
        ]

    return []


def load_split_samples(
    path: str | Path,
    *,
    split_name: str,
) -> list[SourceSample]:
    resolved = resolve_input_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))

    samples: list[SourceSample] = []
    for index, item in enumerate(payload):
        text = item.get("text") or item.get("replaced_text")
        if text is None:
            raise ValueError(f"Missing text field in split sample {index} from {resolved}.")
        entities = parse_entities(item, text=text)
        source = item.get("source")
        if source is None:
            source = "positive" if entities else "negative"
        samples.append(
            SourceSample(
                sample_id=item.get(
                    "sample_id",
                    f"{split_name}-{source}-{index:06d}",
                ),
                source=source,
                source_index=item.get("source_index", index),
                text=text,
                entities=sorted(entities, key=lambda entity: (entity.start, entity.end)),
            )
        )
    return samples


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
) -> BioSample:
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
    return BioSample(
        sample_id=sample.sample_id,
        source=sample.source,
        source_index=sample.source_index,
        text=sample.text,
        tokens=[token.text for token in tokens],
        bio_tags=bio_tags,
        entities=sample.entities,
    )


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def build_manifest(
    *,
    input_paths: dict[str, str | Path],
    input_rows: dict[str, list[SourceSample]],
    output_rows: dict[str, list[BioSample]],
    dropped_empty_token_samples: dict[str, int],
    alignment_warning_count: int,
) -> dict:
    return {
        "entity_label": ENTITY_LABEL,
        "bio_labels": BIO_LABELS,
        "input_paths": {
            split_name: str(path)
            for split_name, path in input_paths.items()
        },
        "counts": {
            "input_splits": {
                split_name: {
                    "total": len(rows),
                    "positive": sum(row.source == "positive" for row in rows),
                    "negative": sum(row.source == "negative" for row in rows),
                }
                for split_name, rows in input_rows.items()
            },
            "output_splits": {
                split_name: {
                    "total": len(rows),
                    "positive": sum(row.source == "positive" for row in rows),
                    "negative": sum(row.source == "negative" for row in rows),
                    "entity_tokens": sum(
                        tag != "O" for row in rows for tag in row.bio_tags
                    ),
                }
                for split_name, rows in output_rows.items()
            },
            "dropped_empty_token_samples": dropped_empty_token_samples,
            "alignment_warning_count": alignment_warning_count,
        },
    }


def build_bio_dataset(
    *,
    input_dir: str | None = DEFAULT_DATA_PREP_SPLITS_DIR,
    train_path: str | None = None,
    val_path: str | None = None,
    test_path: str | None = None,
    output_dir: str = DEFAULT_BIO_OUTPUT_DIR,
) -> dict:
    if input_dir is not None and any(
        path is not None for path in (train_path, val_path, test_path)
    ):
        raise ValueError("Use either input_dir or explicit train/val/test paths, not both.")

    if input_dir is not None:
        input_paths = {
            split_name: Path(input_dir) / f"{split_name}.json"
            for split_name in ("train", "val", "test")
        }
    else:
        if not all(path is not None for path in (train_path, val_path, test_path)):
            raise ValueError(
                "Provide train_path, val_path, and test_path together when input_dir is not used."
            )
        input_paths = {
            "train": train_path,
            "val": val_path,
            "test": test_path,
        }

    input_rows = {
        split_name: load_split_samples(path, split_name=split_name)
        for split_name, path in input_paths.items()
    }

    output_rows: dict[str, list[BioSample]] = {}
    dropped_empty_token_samples = {"positive": 0, "negative": 0}
    warning_tracker = AlignmentWarningTracker()
    for split_name, samples in input_rows.items():
        prepared_rows: list[BioSample] = []
        for sample in samples:
            try:
                prepared_rows.append(
                    prepare_sample(sample, warning_tracker=warning_tracker)
                )
            except ValueError as exc:
                if "does not contain any word tokens" not in str(exc):
                    raise
                dropped_empty_token_samples[sample.source] += 1
        output_rows[split_name] = prepared_rows

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for split_name, rows in output_rows.items():
        write_jsonl(
            destination / f"{split_name}.jsonl",
            [row.to_dict() for row in rows],
        )

    manifest = build_manifest(
        input_paths=input_paths,
        input_rows=input_rows,
        output_rows=output_rows,
        dropped_empty_token_samples=dropped_empty_token_samples,
        alignment_warning_count=warning_tracker.total_count,
    )
    write_json(destination / "manifest.json", manifest)
    return manifest
