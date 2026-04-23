from __future__ import annotations

import json
import math
import random
import re
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    get_linear_schedule_with_warmup,
)

from src.data.io import write_json
from src.models import (
    build_download_help_message,
    build_token_classifier,
    build_tokenizer,
)
from src.bio.converter import (
    BIO_LABELS,
    DEFAULT_BIO_OUTPUT_DIR,
    BioDataset,
    BioSample,
)
from src.models.metrics import (
    build_fp_fn_markdown_report,
    compute_subset_sequence_labeling_metrics,
    compute_sequence_labeling_metrics,
    has_annotation_kind_metadata,
)


@dataclass(frozen=True)
class BaselineTrainingConfig:
    dataset_dir: str = DEFAULT_BIO_OUTPUT_DIR
    output_dir: str = "outputs/models"
    model_name: str = "deepvk/RuModernBERT-base"
    tokenizer_name: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    cache_dir: str | None = None
    max_length: int = 256
    overflow_handling: str = "drop"
    epochs: int = 3
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    mixed_precision: str = "no"
    num_workers: int = 0
    log_every: int = 100
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_test_samples: int | None = None


@dataclass(frozen=True)
class EncodedSampleMetadata:
    sample_id: str
    source: str
    source_index: int
    text: str
    tokens: list[str]
    gold_tags: list[str]
    token_annotation_kinds: list[str | None]
    was_truncated: bool
    original_token_count: int
    kept_token_count: int
    first_token_positions: list[int]


@dataclass(frozen=True)
class TokenizedSplitStats:
    loaded_samples: int
    used_samples: int
    dropped_overflow_samples: int
    truncated_samples: int
    dropped_empty_after_tokenization: int
    max_word_tokens: int
    max_subword_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    metrics: dict[str, float | int]
    subset_metrics: dict[str, dict[str, float | int]]
    predictions: list[dict]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return device


def resolve_mixed_precision(
    requested_mode: str,
    device: torch.device,
) -> tuple[str, torch.dtype | None, torch.amp.GradScaler | None]:
    if requested_mode == "no":
        return requested_mode, None, None

    if device.type != "cuda":
        print(
            f"Mixed precision '{requested_mode}' was requested on device '{device}'. "
            "Falling back to full precision."
        )
        return "no", None, None

    if requested_mode == "bf16":
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if callable(bf16_supported) and not bf16_supported():
            print("CUDA bf16 is not supported on this device. Falling back to fp32.")
            return "no", None, None
        return requested_mode, torch.bfloat16, None

    return requested_mode, torch.float16, torch.amp.GradScaler("cuda")


def maybe_limit_samples(
    samples: Sequence[BioSample],
    *,
    max_samples: int | None,
    seed: int,
) -> list[BioSample]:
    sample_list = list(samples)
    if max_samples is None or max_samples >= len(sample_list):
        return sample_list
    if max_samples <= 0:
        return []

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(sample_list)), max_samples))
    return [sample_list[index] for index in indices]


class PreparedTokenClassificationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[BioSample],
        *,
        tokenizer,
        label_to_id: dict[str, int],
        max_length: int,
        overflow_handling: str,
    ) -> None:
        self._features: list[dict] = []

        dropped_overflow_samples = 0
        truncated_samples = 0
        dropped_empty_after_tokenization = 0
        max_word_tokens = 0
        max_subword_tokens = 0

        for sample in samples:
            max_word_tokens = max(max_word_tokens, len(sample.tokens))
            if len(sample.token_annotation_kinds) != len(sample.tokens):
                raise ValueError(
                    "Each BIO sample must provide token_annotation_kinds aligned with tokens."
                )
            encoding = tokenizer(
                sample.tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
            )
            word_ids = encoding.word_ids()
            if word_ids is None:
                raise RuntimeError(
                    "The tokenizer did not return word ids. "
                    "Use a fast tokenizer for token classification."
                )

            labels: list[int] = []
            first_token_positions: list[int] = []
            kept_word_indices: list[int] = []
            previous_word_id: int | None = None
            for position, word_id in enumerate(word_ids):
                if word_id is None:
                    labels.append(-100)
                    continue
                if word_id != previous_word_id:
                    first_token_positions.append(position)
                    kept_word_indices.append(word_id)
                    labels.append(label_to_id[sample.bio_tags[word_id]])
                    previous_word_id = word_id
                else:
                    labels.append(-100)

            if not kept_word_indices:
                dropped_empty_after_tokenization += 1
                continue

            was_truncated = kept_word_indices[-1] < len(sample.tokens) - 1
            if was_truncated and overflow_handling == "drop":
                dropped_overflow_samples += 1
                continue
            if was_truncated:
                truncated_samples += 1

            max_subword_tokens = max(max_subword_tokens, len(encoding["input_ids"]))
            kept_tokens = [sample.tokens[index] for index in kept_word_indices]
            kept_gold_tags = [sample.bio_tags[index] for index in kept_word_indices]
            kept_token_annotation_kinds = [
                sample.token_annotation_kinds[index] for index in kept_word_indices
            ]

            feature = {key: value for key, value in encoding.items()}
            feature["labels"] = labels
            feature["_metadata"] = EncodedSampleMetadata(
                sample_id=sample.sample_id,
                source=sample.source,
                source_index=sample.source_index,
                text=sample.text,
                tokens=kept_tokens,
                gold_tags=kept_gold_tags,
                token_annotation_kinds=kept_token_annotation_kinds,
                was_truncated=was_truncated,
                original_token_count=len(sample.tokens),
                kept_token_count=len(kept_tokens),
                first_token_positions=first_token_positions,
            )
            self._features.append(feature)

        self.stats = TokenizedSplitStats(
            loaded_samples=len(samples),
            used_samples=len(self._features),
            dropped_overflow_samples=dropped_overflow_samples,
            truncated_samples=truncated_samples,
            dropped_empty_after_tokenization=dropped_empty_after_tokenization,
            max_word_tokens=max_word_tokens,
            max_subword_tokens=max_subword_tokens,
        )

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, index: int) -> dict:
        return dict(self._features[index])


class TokenClassificationCollator:
    def __init__(self, tokenizer) -> None:
        self._collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def __call__(self, features: list[dict]) -> dict:
        metadata = [feature.pop("_metadata") for feature in features]
        batch = self._collator(features)
        batch["metadata"] = metadata
        return batch


def count_model_parameters(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }


def write_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def create_tensorboard_writer(log_dir: str | Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "TensorBoard logging requires the `tensorboard` package. "
            "Install dependencies again, for example: "
            "`venv/bin/pip install -r requirements.txt`."
        ) from exc

    output_path = Path(log_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(output_path))


def log_scalar_tree(
    writer,
    *,
    prefix: str,
    values: dict[str, Any],
    step: int,
) -> None:
    for metric_name, metric_value in values.items():
        tag = f"{prefix}/{metric_name}"
        if isinstance(metric_value, dict):
            log_scalar_tree(
                writer,
                prefix=tag,
                values=metric_value,
                step=step,
            )
            continue
        if isinstance(metric_value, bool):
            continue
        if isinstance(metric_value, (int, float)):
            writer.add_scalar(tag, float(metric_value), step)


def build_tensorboard_custom_scalar_layout() -> dict[str, dict[str, list[Any]]]:
    return {
        "Loss": {
            "Train vs Val vs Test": [
                "Multiline",
                [
                    "epoch/train/loss",
                    "epoch/val/loss",
                    "epoch/test/loss",
                ],
            ],
        },
        "F1": {
            "Val vs Test Token F1": [
                "Multiline",
                [
                    "epoch/val/token_f1",
                    "epoch/test/token_f1",
                ],
            ],
            "Val vs Test Span F1": [
                "Multiline",
                [
                    "epoch/val/span_f1",
                    "epoch/test/span_f1",
                ],
            ],
        },
        "Subset Span F1": {
            "Test Subsets": [
                "Multiline",
                [
                    "epoch/test/subsets/replacement_pool_only/span_f1",
                    "epoch/test/subsets/other_gold_entities_only/span_f1",
                ],
            ],
        },
    }


def prefix_metrics(
    metrics: dict[str, float | int],
    *,
    prefix: str,
) -> dict[str, float | int]:
    return {
        f"{prefix}{metric_name}": metric_value
        for metric_name, metric_value in metrics.items()
    }


def build_entity_subset_definitions(
    token_annotation_kind_sequences: Sequence[Sequence[str | None]],
) -> dict[str, set[str]]:
    observed_kinds = {
        annotation_kind
        for sequence in token_annotation_kind_sequences
        for annotation_kind in sequence
        if annotation_kind is not None
    }
    if not observed_kinds:
        return {}

    subset_definitions: dict[str, set[str]] = {}
    replacement_pool_kinds = {"synthetic_replacement"} & observed_kinds
    if replacement_pool_kinds:
        subset_definitions["replacement_pool_only"] = replacement_pool_kinds

    other_gold_entity_kinds = observed_kinds - {"synthetic_replacement"}
    if other_gold_entity_kinds:
        subset_definitions["other_gold_entities_only"] = other_gold_entity_kinds

    return subset_definitions


def format_subset_metric_summary(
    subset_metrics: dict[str, dict[str, float | int]],
) -> str:
    ordered_names = (
        "replacement_pool_only",
        "other_gold_entities_only",
    )
    parts = [
        (
            f"{subset_name}_span_f1="
            f"{float(subset_metrics[subset_name]['span_f1']):.4f}"
        )
        for subset_name in ordered_names
        if subset_name in subset_metrics
    ]
    return " | ".join(parts)


def build_short_model_name(model_name_or_path: str) -> str:
    normalized_name = model_name_or_path.rstrip("/\\")
    if not normalized_name:
        return "model"

    base_name = normalized_name.split("/")[-1].split("\\")[-1]
    sanitized_name = re.sub(r"[^0-9A-Za-z]+", "_", base_name).strip("_").lower()
    return sanitized_name or "model"


def build_run_output_dir(
    base_output_dir: str | Path,
    *,
    model_name_or_path: str,
    started_at: datetime,
) -> Path:
    timestamp = started_at.strftime("%m_%d_%H_%M")
    run_name = f"{build_short_model_name(model_name_or_path)}_{timestamp}"
    base_path = Path(base_output_dir)
    candidate = base_path / run_name
    collision_index = 2

    while candidate.exists():
        candidate = base_path / f"{run_name}_{collision_index:02d}"
        collision_index += 1

    return candidate


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    id_to_label: dict[int, str],
    device: torch.device,
    amp_dtype: torch.dtype | None,
    collect_predictions: bool,
    compute_entity_subset_metrics: bool = False,
) -> EvaluationResult:
    model.eval()

    predictions: list[dict] = []
    gold_sequences: list[list[str]] = []
    predicted_sequences: list[list[str]] = []
    token_annotation_kind_sequences: list[list[str | None]] = []
    loss_sum = 0.0
    active_label_count = 0

    with torch.no_grad():
        for batch in dataloader:
            metadata: list[EncodedSampleMetadata] = batch.pop("metadata")
            labels = batch["labels"]
            active_labels = int((labels != -100).sum().item())
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else nullcontext()
            )
            with autocast_context:
                outputs = model(**batch)

            loss_sum += outputs.loss.item() * active_labels
            active_label_count += active_labels
            predicted_label_ids = outputs.logits.detach().cpu().argmax(dim=-1)

            for row_index, sample_metadata in enumerate(metadata):
                predicted_tags = [
                    id_to_label[int(predicted_label_ids[row_index, position].item())]
                    for position in sample_metadata.first_token_positions
                ]
                gold_tags = list(sample_metadata.gold_tags)
                if len(predicted_tags) != len(gold_tags):
                    raise RuntimeError(
                        "Predicted word-level tags are not aligned with gold tags."
                    )

                gold_sequences.append(gold_tags)
                predicted_sequences.append(predicted_tags)
                token_annotation_kind_sequences.append(
                    list(sample_metadata.token_annotation_kinds)
                )

                if collect_predictions:
                    predictions.append(
                        {
                            "sample_id": sample_metadata.sample_id,
                            "source": sample_metadata.source,
                            "source_index": sample_metadata.source_index,
                            "text": sample_metadata.text,
                            "tokens": sample_metadata.tokens,
                            "gold_tags": gold_tags,
                            "predicted_tags": predicted_tags,
                            "token_annotation_kinds": (
                                sample_metadata.token_annotation_kinds
                            ),
                            "was_truncated": sample_metadata.was_truncated,
                            "original_token_count": sample_metadata.original_token_count,
                            "kept_token_count": sample_metadata.kept_token_count,
                        }
                    )

    metrics = compute_sequence_labeling_metrics(gold_sequences, predicted_sequences)
    subset_metrics: dict[str, dict[str, float | int]] = {}
    if compute_entity_subset_metrics:
        if not has_annotation_kind_metadata(
            gold_sequences,
            token_annotation_kind_sequences,
        ):
            raise RuntimeError(
                "Test subset metrics require BIO samples with token_annotation_kinds. "
                "Rebuild the BIO dataset with `venv/bin/python -m src.bio` "
                "from split JSON files that still contain `euphemisms` metadata."
            )

        for subset_name, allowed_annotation_kinds in build_entity_subset_definitions(
            token_annotation_kind_sequences
        ).items():
            subset_metrics[subset_name] = compute_subset_sequence_labeling_metrics(
                gold_sequences,
                predicted_sequences,
                token_annotation_kind_sequences,
                allowed_annotation_kinds=allowed_annotation_kinds,
            )

    average_loss = loss_sum / max(1, active_label_count)
    return EvaluationResult(
        loss=average_loss,
        metrics=metrics,
        subset_metrics=subset_metrics,
        predictions=predictions,
    )


def run_training_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    scaler: torch.amp.GradScaler | None,
    grad_accumulation_steps: int,
    max_grad_norm: float,
    epoch_index: int,
    total_epochs: int,
    log_every: int,
) -> dict[str, float | int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss_sum = 0.0
    active_label_count = 0
    optimizer_steps = 0
    start_time = time.time()

    for step_index, batch in enumerate(dataloader, start=1):
        batch.pop("metadata", None)
        labels = batch["labels"]
        active_labels = int((labels != -100).sum().item())
        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        autocast_context = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            outputs = model(**batch)
            loss = outputs.loss

        loss_sum += loss.item() * active_labels
        active_label_count += active_labels

        scaled_loss = loss / grad_accumulation_steps
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = (
            step_index % grad_accumulation_steps == 0
            or step_index == len(dataloader)
        )
        if should_step:
            optimizer_was_run = True
            if max_grad_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            if scaler is not None:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_was_run = scaler.get_scale() >= scale_before
            else:
                optimizer.step()

            if optimizer_was_run:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if log_every > 0 and (
            step_index % log_every == 0 or step_index == len(dataloader)
        ):
            elapsed_seconds = time.time() - start_time
            print(
                f"Epoch {epoch_index}/{total_epochs} | "
                f"step {step_index}/{len(dataloader)} | "
                f"train_loss={loss_sum / max(1, active_label_count):.6f} | "
                f"lr={scheduler.get_last_lr()[0]:.8f} | "
                f"elapsed={elapsed_seconds:.1f}s"
            )

    return {
        "loss": loss_sum / max(1, active_label_count),
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": time.time() - start_time,
    }


def load_split_samples(dataset_dir: str | Path, *, split: str) -> list[BioSample]:
    dataset = BioDataset.from_directory(dataset_dir, split=split)
    return list(dataset)


def train_baseline_model(config: BaselineTrainingConfig) -> dict:
    if config.epochs <= 0:
        raise ValueError("Number of epochs must be positive.")
    if config.train_batch_size <= 0 or config.eval_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if config.grad_accumulation_steps <= 0:
        raise ValueError("Gradient accumulation steps must be positive.")
    if config.max_length <= 0:
        raise ValueError("Max sequence length must be positive.")
    if config.overflow_handling not in {"drop", "truncate"}:
        raise ValueError("Overflow handling must be either 'drop' or 'truncate'.")

    set_seed(config.seed)
    device = resolve_device(config.device)
    resolved_mixed_precision, amp_dtype, scaler = resolve_mixed_precision(
        config.mixed_precision,
        device,
    )

    run_started_at = datetime.now().astimezone()
    output_dir = build_run_output_dir(
        config.output_dir,
        model_name_or_path=config.model_name,
        started_at=run_started_at,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    tensorboard_dir = output_dir / "tensorboard"

    label_to_id = {label: index for index, label in enumerate(BIO_LABELS)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    try:
        tokenizer, resolved_tokenizer_revision = build_tokenizer(
            model_name_or_path=config.model_name,
            tokenizer_name_or_path=config.tokenizer_name,
            tokenizer_revision=config.tokenizer_revision,
            cache_dir=config.cache_dir,
        )
        model = build_token_classifier(
            model_name_or_path=config.model_name,
            num_labels=len(BIO_LABELS),
            id2label=id_to_label,
            label2id=label_to_id,
            model_revision=config.model_revision,
            cache_dir=config.cache_dir,
        )
    except OSError as exc:
        raise RuntimeError(
            build_download_help_message(
                model_name_or_path=config.model_name,
                tokenizer_name_or_path=config.tokenizer_name,
                model_revision=config.model_revision,
                tokenizer_revision=config.tokenizer_revision,
                cache_dir=config.cache_dir,
            )
        ) from exc

    model.to(device)
    parameter_counts = count_model_parameters(model)

    available_samples = {
        split_name: load_split_samples(config.dataset_dir, split=split_name)
        for split_name in ("train", "val", "test")
    }
    selected_samples = {
        "train": maybe_limit_samples(
            available_samples["train"],
            max_samples=config.max_train_samples,
            seed=config.seed + 11,
        ),
        "val": maybe_limit_samples(
            available_samples["val"],
            max_samples=config.max_val_samples,
            seed=config.seed + 17,
        ),
        "test": maybe_limit_samples(
            available_samples["test"],
            max_samples=config.max_test_samples,
            seed=config.seed + 23,
        ),
    }

    datasets = {
        split_name: PreparedTokenClassificationDataset(
            selected_samples[split_name],
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            max_length=config.max_length,
            overflow_handling=config.overflow_handling,
        )
        for split_name in ("train", "val", "test")
    }

    for split_name, dataset in datasets.items():
        if len(dataset) == 0:
            raise RuntimeError(
                f"Split '{split_name}' became empty after tokenization. "
                "Increase --max-length or switch --overflow-handling to truncate."
            )

    collator = TokenClassificationCollator(tokenizer)
    train_generator = torch.Generator()
    train_generator.manual_seed(config.seed)

    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=config.num_workers,
            generator=train_generator,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.num_workers,
        ),
    }

    steps_per_epoch = math.ceil(
        len(dataloaders["train"]) / config.grad_accumulation_steps
    )
    total_training_steps = steps_per_epoch * config.epochs
    warmup_steps = int(total_training_steps * config.warmup_ratio)

    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    summary_stub = {
        "config": asdict(config),
        "resolved_runtime": {
            "device": str(device),
            "mixed_precision": resolved_mixed_precision,
            "run_started_at": run_started_at.isoformat(timespec="seconds"),
            "output_dir": str(output_dir),
            "run_name": output_dir.name,
            "tensorboard_dir": str(tensorboard_dir),
            "tokenizer_revision": resolved_tokenizer_revision,
            "parameter_counts": parameter_counts,
            "total_training_steps": total_training_steps,
            "warmup_steps": warmup_steps,
        },
        "dataset": {
            split_name: {
                "available_samples": len(available_samples[split_name]),
                "selected_samples": len(selected_samples[split_name]),
                "tokenization": datasets[split_name].stats.to_dict(),
            }
            for split_name in ("train", "val", "test")
        },
    }
    write_json(output_dir / "run_config.json", summary_stub)
    writer = create_tensorboard_writer(tensorboard_dir)
    writer.add_custom_scalars(build_tensorboard_custom_scalar_layout())

    print(f"Run directory: {output_dir}")
    print(f"Training device: {device}")
    print(
        "Model parameters: "
        f"total={parameter_counts['total']}, trainable={parameter_counts['trainable']}"
    )
    for split_name in ("train", "val", "test"):
        stats = datasets[split_name].stats
        print(
            f"{split_name}: selected={len(selected_samples[split_name])}, "
            f"used={stats.used_samples}, "
            f"dropped_overflow={stats.dropped_overflow_samples}, "
            f"truncated={stats.truncated_samples}"
        )

    try:
        log_scalar_tree(
            writer,
            prefix="run/config",
            values={
                "epochs": config.epochs,
                "train_batch_size": config.train_batch_size,
                "eval_batch_size": config.eval_batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "warmup_ratio": config.warmup_ratio,
                "grad_accumulation_steps": config.grad_accumulation_steps,
                "max_grad_norm": config.max_grad_norm,
                "max_length": config.max_length,
                "seed": config.seed,
                "total_training_steps": total_training_steps,
                "warmup_steps": warmup_steps,
            },
            step=0,
        )
        for split_name in ("train", "val", "test"):
            log_scalar_tree(
                writer,
                prefix=f"run/dataset/{split_name}",
                values={
                    "available_samples": len(available_samples[split_name]),
                    "selected_samples": len(selected_samples[split_name]),
                    **datasets[split_name].stats.to_dict(),
                },
                step=0,
            )

        history: list[dict] = []
        best_epoch: int | None = None
        best_score: tuple[float, float] | None = None
        best_model_dir = output_dir / "best_model"

        for epoch_index in range(1, config.epochs + 1):
            print(f"Starting epoch {epoch_index}/{config.epochs}")
            train_metrics = run_training_epoch(
                model,
                dataloaders["train"],
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                amp_dtype=amp_dtype,
                scaler=scaler,
                grad_accumulation_steps=config.grad_accumulation_steps,
                max_grad_norm=config.max_grad_norm,
                epoch_index=epoch_index,
                total_epochs=config.epochs,
                log_every=config.log_every,
            )

            val_result = evaluate_model(
                model,
                dataloaders["val"],
                id_to_label=id_to_label,
                device=device,
                amp_dtype=amp_dtype,
                collect_predictions=False,
            )
            test_result = evaluate_model(
                model,
                dataloaders["test"],
                id_to_label=id_to_label,
                device=device,
                amp_dtype=amp_dtype,
                collect_predictions=False,
                compute_entity_subset_metrics=True,
            )

            epoch_summary = {
                "epoch": epoch_index,
                "train_loss": train_metrics["loss"],
                "train_optimizer_steps": train_metrics["optimizer_steps"],
                "train_elapsed_seconds": train_metrics["elapsed_seconds"],
                "val_loss": val_result.loss,
                **val_result.metrics,
                "test_loss": test_result.loss,
                **prefix_metrics(test_result.metrics, prefix="test_"),
            }
            if test_result.subset_metrics:
                epoch_summary["test_subsets"] = test_result.subset_metrics
            history.append(epoch_summary)

            log_scalar_tree(
                writer,
                prefix="epoch/train",
                values={
                    "loss": train_metrics["loss"],
                    "optimizer_steps": train_metrics["optimizer_steps"],
                    "elapsed_seconds": train_metrics["elapsed_seconds"],
                },
                step=epoch_index,
            )
            log_scalar_tree(
                writer,
                prefix="epoch/val",
                values={
                    "loss": val_result.loss,
                    **val_result.metrics,
                },
                step=epoch_index,
            )
            log_scalar_tree(
                writer,
                prefix="epoch/test",
                values={
                    "loss": test_result.loss,
                    **test_result.metrics,
                    "subsets": test_result.subset_metrics,
                },
                step=epoch_index,
            )
            writer.flush()

            print(
                f"Epoch {epoch_index} finished | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_result.loss:.6f} | "
                f"val_token_f1={val_result.metrics['token_f1']:.4f} | "
                f"val_span_f1={val_result.metrics['span_f1']:.4f} | "
                f"test_loss={test_result.loss:.6f} | "
                f"test_token_f1={test_result.metrics['token_f1']:.4f} | "
                f"test_span_f1={test_result.metrics['span_f1']:.4f}"
            )
            if test_result.subset_metrics:
                print(
                    "Test subsets | "
                    f"{format_subset_metric_summary(test_result.subset_metrics)}"
                )

            score = (
                float(val_result.metrics["span_f1"]),
                float(val_result.metrics["token_f1"]),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_epoch = epoch_index
                model.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                write_json(
                    output_dir / "best_model_metrics.json",
                    {
                        "epoch": epoch_index,
                        "val_loss": val_result.loss,
                        **val_result.metrics,
                        "test_loss": test_result.loss,
                        **prefix_metrics(test_result.metrics, prefix="test_"),
                        "test_subsets": test_result.subset_metrics,
                    },
                )
                print(f"Saved new best checkpoint to {best_model_dir}")

        if best_epoch is None:
            raise RuntimeError("Training finished without producing a best checkpoint.")

        best_model = AutoModelForTokenClassification.from_pretrained(best_model_dir)
        best_model.to(device)

        final_val_result = evaluate_model(
            best_model,
            dataloaders["val"],
            id_to_label=id_to_label,
            device=device,
            amp_dtype=amp_dtype,
            collect_predictions=True,
        )
        final_test_result = evaluate_model(
            best_model,
            dataloaders["test"],
            id_to_label=id_to_label,
            device=device,
            amp_dtype=amp_dtype,
            collect_predictions=True,
            compute_entity_subset_metrics=True,
        )

        predictions_dir = output_dir / "predictions"
        write_jsonl(predictions_dir / "val.jsonl", final_val_result.predictions)
        write_jsonl(predictions_dir / "test.jsonl", final_test_result.predictions)

        analysis_dir = output_dir / "analysis"
        write_text(
            analysis_dir / "test_fp_fn.md",
            build_fp_fn_markdown_report(
                final_test_result.predictions,
                split_name="test",
            ),
        )

        metrics_payload = {
            "best_epoch": best_epoch,
            "val": {
                "loss": final_val_result.loss,
                **final_val_result.metrics,
            },
            "test": {
                "loss": final_test_result.loss,
                **final_test_result.metrics,
                "subsets": final_test_result.subset_metrics,
            },
        }
        write_json(output_dir / "metrics.json", metrics_payload)

        log_scalar_tree(
            writer,
            prefix="final/val",
            values={
                "loss": final_val_result.loss,
                **final_val_result.metrics,
            },
            step=best_epoch,
        )
        log_scalar_tree(
            writer,
            prefix="final/test",
            values={
                "loss": final_test_result.loss,
                **final_test_result.metrics,
                "subsets": final_test_result.subset_metrics,
            },
            step=best_epoch,
        )
        writer.add_scalar("final/best_epoch", float(best_epoch), best_epoch)
        writer.flush()

        training_summary = {
            **summary_stub,
            "best_epoch": best_epoch,
            "history": history,
            "final_metrics": metrics_payload,
        }
        write_json(output_dir / "training_summary.json", training_summary)

        print(
            f"Best epoch: {best_epoch} | "
            f"val_span_f1={final_val_result.metrics['span_f1']:.4f} | "
            f"test_span_f1={final_test_result.metrics['span_f1']:.4f}"
        )
        if final_test_result.subset_metrics:
            print(
                "Final test subsets | "
                f"{format_subset_metric_summary(final_test_result.subset_metrics)}"
            )
        print(f"Artifacts saved to {output_dir}")
        return training_summary
    finally:
        writer.close()
