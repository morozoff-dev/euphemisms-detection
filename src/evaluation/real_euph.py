from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from scripts.infer_one_text import predict_tags_for_tokens
from src.bio.converter import (
    ENTITY_LABEL,
    AlignmentWarningTracker,
    EntitySpan,
    assign_entities_to_tokens,
    find_covered_tokens,
    tokenize_words,
    write_jsonl,
)
from src.data.io import load_lines, resolve_input_path, write_json
from src.data.text import lookup_norm
from src.models.metrics import (
    build_fp_fn_markdown_report,
    compute_span_metrics,
    compute_token_metrics,
    token_labels_to_spans,
)

DEFAULT_INPUT_JSON = "data/real_test_euph.json"
DEFAULT_TARGET_KEYWORDS_PATH = "data/target_keywords_forms_drug.txt"
DEFAULT_OUTPUT_DIR = "outputs/evaluation/real_test_euph"


@dataclass(frozen=True)
class RealEvaluationSample:
    sample_id: str
    source_index: int
    text: str
    entities: list[EntitySpan]


def load_target_keyword_lookup(path: str | Path) -> set[str]:
    return {lookup_norm(line) for line in load_lines(path)}


def _require_bool_true(value: Any) -> bool:
    return value is True


def _parse_entity(
    entity_payload: dict,
    *,
    sample_index: int,
    entity_index: int,
    text: str,
) -> EntitySpan:
    try:
        entity_text = entity_payload["text"]
        start = entity_payload["start"]
        end = entity_payload["end"]
    except KeyError as exc:
        raise ValueError(
            "Missing required entity field "
            f"{exc.args[0]!r} in sample index {sample_index}, entity {entity_index}."
        ) from exc

    if not isinstance(entity_text, str):
        raise ValueError(
            f"Entity text must be a string in sample index {sample_index}, "
            f"entity {entity_index}."
        )
    if not isinstance(start, int) or not isinstance(end, int):
        raise ValueError(
            f"Entity start/end must be integers in sample index {sample_index}, "
            f"entity {entity_index}."
        )
    if start < 0 or end < start or end > len(text):
        raise ValueError(
            f"Entity span [{start}, {end}) is outside text bounds in "
            f"sample index {sample_index}, entity {entity_index}."
        )

    actual_text = text[start:end]
    if actual_text != entity_text:
        raise ValueError(
            "Entity offset mismatch in "
            f"sample index {sample_index}, entity {entity_index}: "
            f"expected entity text {entity_text!r}, but text_body slice is "
            f"{actual_text!r}."
        )

    return EntitySpan(start=start, end=end, text=entity_text, label=ENTITY_LABEL)


def load_real_evaluation_samples(path: str | Path) -> tuple[list[RealEvaluationSample], int]:
    resolved = resolve_input_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level JSON list in {resolved}.")

    samples: list[RealEvaluationSample] = []
    for sample_index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object at sample index {sample_index}.")
        if not _require_bool_true(item.get("verified")):
            continue

        text = item.get("text_body")
        if not isinstance(text, str):
            raise ValueError(
                f"Verified sample index {sample_index} must contain string text_body."
            )

        entities_payload = item.get("entities") or []
        if not isinstance(entities_payload, list):
            raise ValueError(
                f"Verified sample index {sample_index} must contain list entities."
            )

        entities = [
            _parse_entity(
                entity_payload,
                sample_index=sample_index,
                entity_index=entity_index,
                text=text,
            )
            for entity_index, entity_payload in enumerate(entities_payload)
        ]
        samples.append(
            RealEvaluationSample(
                sample_id=f"real-test-euph-{sample_index:06d}",
                source_index=sample_index,
                text=text,
                entities=sorted(entities, key=lambda entity: (entity.start, entity.end)),
            )
        )

    return samples, len(payload)


def build_ignored_token_positions(
    tokens: Sequence,
    target_keyword_lookup: set[str],
) -> list[int]:
    return [
        index
        for index, token in enumerate(tokens)
        if lookup_norm(token.text) in target_keyword_lookup
    ]


def mask_ignored_tags(tags: Sequence[str], ignored_positions: Sequence[int]) -> list[str]:
    masked = list(tags)
    for token_index in ignored_positions:
        masked[token_index] = "O"
    return masked


def spans_to_dicts(
    tags: Sequence[str],
    *,
    text: str,
    token_spans: Sequence,
) -> list[dict]:
    spans = sorted(token_labels_to_spans(tags), key=lambda item: (item[1], item[2], item[0]))
    result: list[dict] = []
    for label, start_token, end_token in spans:
        start_char = token_spans[start_token].start
        end_char = token_spans[end_token - 1].end
        result.append(
            {
                "label": label,
                "start_token": start_token,
                "end_token": end_token,
                "start_char": start_char,
                "end_char": end_char,
                "text": text[start_char:end_char],
                "tokens": [
                    token.text for token in token_spans[start_token:end_token]
                ],
            }
        )
    return result


def build_entity_ignore_counts(
    *,
    entities: Sequence[EntitySpan],
    token_spans: Sequence,
    ignored_positions: set[int],
) -> dict[str, int]:
    overlapping_count = 0
    fully_ignored_count = 0

    for entity in entities:
        covered_indices, _ = find_covered_tokens(token_spans, entity=entity)
        if not covered_indices:
            continue
        if any(index in ignored_positions for index in covered_indices):
            overlapping_count += 1
        if all(index in ignored_positions for index in covered_indices):
            fully_ignored_count += 1

    return {
        "gold_entities_overlapping_ignored_target_tokens": overlapping_count,
        "gold_entities_fully_ignored_as_target_keywords": fully_ignored_count,
    }


def build_metrics_prediction_row(
    *,
    sample: RealEvaluationSample,
    tokens: Sequence[str],
    masked_gold_tags: Sequence[str],
    masked_predicted_tags: Sequence[str],
) -> dict:
    return {
        "sample_id": sample.sample_id,
        "source": "real_verified",
        "source_index": sample.source_index,
        "negative_group": None,
        "text": sample.text,
        "tokens": list(tokens),
        "gold_tags": list(masked_gold_tags),
        "predicted_tags": list(masked_predicted_tags),
        "was_truncated": False,
        "original_token_count": len(tokens),
        "kept_token_count": len(tokens),
    }


def evaluate_real_euphemisms(
    *,
    input_json: str | Path,
    target_keywords_path: str | Path,
    model,
    tokenizer,
    torch_module,
    device,
    checkpoint_dir: str | Path,
    run_dir: str | Path | None,
    checkpoint_metadata,
    max_length: int,
    window_overlap_words: int,
    prediction_threshold: float,
) -> dict:
    samples, input_record_count = load_real_evaluation_samples(input_json)
    target_keyword_lookup = load_target_keyword_lookup(target_keywords_path)
    warning_tracker = AlignmentWarningTracker(max_examples=0)

    gold_sequences: list[list[str]] = []
    predicted_sequences: list[list[str]] = []
    token_metric_gold_sequences: list[list[str]] = []
    token_metric_predicted_sequences: list[list[str]] = []
    output_predictions: list[dict] = []
    metrics_prediction_rows: list[dict] = []

    total_tokens = 0
    ignored_target_tokens = 0
    samples_with_ignored_target_tokens = 0
    ignored_gold_entity_tokens = 0
    ignored_predicted_entity_tokens = 0
    gold_entities_overlapping_ignored = 0
    gold_entities_fully_ignored = 0
    chunk_count_total = 0

    for sample in samples:
        token_spans = tokenize_words(sample.text)
        if not token_spans:
            raise ValueError(f"Verified sample {sample.sample_id} does not contain tokens.")

        raw_gold_tags, _ = assign_entities_to_tokens(
            token_spans,
            sample.entities,
            sample_id=sample.sample_id,
            text=sample.text,
            warning_tracker=warning_tracker,
        )
        raw_predicted_tags, chunk_count = predict_tags_for_tokens(
            tokens=token_spans,
            model=model,
            tokenizer=tokenizer,
            torch_module=torch_module,
            device=device,
            max_length=max_length,
            window_overlap_words=window_overlap_words,
            use_word_start_mask=not checkpoint_metadata.is_legacy,
            positive_label_id=checkpoint_metadata.positive_label_id,
            prediction_threshold=prediction_threshold,
        )

        ignored_positions = build_ignored_token_positions(
            token_spans,
            target_keyword_lookup,
        )
        ignored_position_set = set(ignored_positions)
        masked_gold_tags = mask_ignored_tags(raw_gold_tags, ignored_positions)
        masked_predicted_tags = mask_ignored_tags(raw_predicted_tags, ignored_positions)
        token_texts = [token.text for token in token_spans]

        entity_ignore_counts = build_entity_ignore_counts(
            entities=sample.entities,
            token_spans=token_spans,
            ignored_positions=ignored_position_set,
        )

        gold_sequences.append(masked_gold_tags)
        predicted_sequences.append(masked_predicted_tags)
        token_metric_gold_sequences.append(
            [
                tag
                for index, tag in enumerate(masked_gold_tags)
                if index not in ignored_position_set
            ]
        )
        token_metric_predicted_sequences.append(
            [
                tag
                for index, tag in enumerate(masked_predicted_tags)
                if index not in ignored_position_set
            ]
        )
        metrics_prediction_rows.append(
            build_metrics_prediction_row(
                sample=sample,
                tokens=token_texts,
                masked_gold_tags=masked_gold_tags,
                masked_predicted_tags=masked_predicted_tags,
            )
        )

        total_tokens += len(token_spans)
        ignored_target_tokens += len(ignored_positions)
        if ignored_positions:
            samples_with_ignored_target_tokens += 1
        ignored_gold_entity_tokens += sum(
            1
            for position in ignored_positions
            if raw_gold_tags[position] != "O"
        )
        ignored_predicted_entity_tokens += sum(
            1
            for position in ignored_positions
            if raw_predicted_tags[position] != "O"
        )
        gold_entities_overlapping_ignored += entity_ignore_counts[
            "gold_entities_overlapping_ignored_target_tokens"
        ]
        gold_entities_fully_ignored += entity_ignore_counts[
            "gold_entities_fully_ignored_as_target_keywords"
        ]
        chunk_count_total += chunk_count

        output_predictions.append(
            {
                "sample_id": sample.sample_id,
                "source": "real_verified",
                "source_index": sample.source_index,
                "text": sample.text,
                "tokens": token_texts,
                "raw_gold_tags": raw_gold_tags,
                "raw_predicted_tags": raw_predicted_tags,
                "masked_gold_tags": masked_gold_tags,
                "masked_predicted_tags": masked_predicted_tags,
                "ignored_token_positions": ignored_positions,
                "ignored_tokens": [
                    {
                        "token_index": position,
                        "text": token_spans[position].text,
                        "start_char": token_spans[position].start,
                        "end_char": token_spans[position].end,
                    }
                    for position in ignored_positions
                ],
                "gold_spans": spans_to_dicts(
                    raw_gold_tags,
                    text=sample.text,
                    token_spans=token_spans,
                ),
                "predicted_spans": spans_to_dicts(
                    raw_predicted_tags,
                    text=sample.text,
                    token_spans=token_spans,
                ),
                "masked_gold_spans": spans_to_dicts(
                    masked_gold_tags,
                    text=sample.text,
                    token_spans=token_spans,
                ),
                "masked_predicted_spans": spans_to_dicts(
                    masked_predicted_tags,
                    text=sample.text,
                    token_spans=token_spans,
                ),
                "chunk_count": chunk_count,
            }
        )

    counts = {
        "input_records": input_record_count,
        "verified_records": len(samples),
        "gold_entities": sum(len(sample.entities) for sample in samples),
        "total_tokens": total_tokens,
        "evaluated_tokens": total_tokens - ignored_target_tokens,
        "ignored_target_tokens": ignored_target_tokens,
        "samples_with_ignored_target_tokens": samples_with_ignored_target_tokens,
        "ignored_gold_entity_tokens": ignored_gold_entity_tokens,
        "ignored_predicted_entity_tokens": ignored_predicted_entity_tokens,
        "gold_entities_overlapping_ignored_target_tokens": (
            gold_entities_overlapping_ignored
        ),
        "gold_entities_fully_ignored_as_target_keywords": gold_entities_fully_ignored,
        "alignment_warning_count": warning_tracker.total_count,
        "inference_windows": chunk_count_total,
    }
    token_metrics = compute_token_metrics(
        token_metric_gold_sequences,
        token_metric_predicted_sequences,
    )
    span_metrics = compute_span_metrics(gold_sequences, predicted_sequences)
    metrics = {
        "token_precision": token_metrics["precision"],
        "token_recall": token_metrics["recall"],
        "token_f1": token_metrics["f1"],
        "token_accuracy": token_metrics["accuracy"],
        "token_tp": token_metrics["tp"],
        "token_fp": token_metrics["fp"],
        "token_fn": token_metrics["fn"],
        "total_tokens": token_metrics["total_tokens"],
        "gold_entity_tokens": token_metrics["gold_entity_tokens"],
        "span_precision": span_metrics["precision"],
        "span_recall": span_metrics["recall"],
        "span_f1": span_metrics["f1"],
        "span_tp": span_metrics["tp"],
        "span_fp": span_metrics["fp"],
        "span_fn": span_metrics["fn"],
        "gold_spans": span_metrics["gold_spans"],
        "predicted_spans": span_metrics["predicted_spans"],
    }

    return {
        "model_dir": str(checkpoint_dir),
        "run_dir": str(run_dir) if run_dir is not None else None,
        "input_json": str(input_json),
        "target_keywords_path": str(target_keywords_path),
        "device": str(device),
        "head_mode": checkpoint_metadata.head_mode,
        "checkpoint_architecture": checkpoint_metadata.checkpoint_architecture,
        "positive_label_id": checkpoint_metadata.positive_label_id,
        "max_length": max_length,
        "window_overlap_words": window_overlap_words,
        "prediction_threshold": prediction_threshold,
        "counts": counts,
        "metrics": metrics,
        "predictions": output_predictions,
        "analysis_predictions": metrics_prediction_rows,
    }


def write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def build_metrics_payload(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key not in {"predictions", "analysis_predictions"}
    }


def write_evaluation_outputs(output_dir: str | Path, result: dict) -> None:
    output_path = Path(output_dir)
    write_json(output_path / "metrics.json", build_metrics_payload(result))
    write_jsonl(output_path / "predictions.jsonl", result["predictions"])
    write_text(
        output_path / "analysis" / "fp_fn.md",
        build_fp_fn_markdown_report(
            result["analysis_predictions"],
            split_name="real_test_euph",
        ),
    )
