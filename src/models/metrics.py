from __future__ import annotations

from typing import Sequence


def _sorted_spans(
    spans: set[tuple[str, int, int]],
) -> list[tuple[str, int, int]]:
    return sorted(spans, key=lambda item: (item[1], item[2], item[0]))


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def bio_tags_to_spans(tags: Sequence[str]) -> set[tuple[str, int, int]]:
    spans: set[tuple[str, int, int]] = set()
    active_label: str | None = None
    active_start: int | None = None

    for index, tag in enumerate(tags):
        if tag == "O":
            if active_label is not None and active_start is not None:
                spans.add((active_label, active_start, index))
            active_label = None
            active_start = None
            continue

        prefix, _, entity_label = tag.partition("-")
        if not entity_label:
            prefix = "B"
            entity_label = tag

        starts_new_span = (
            prefix == "B"
            or active_label is None
            or active_label != entity_label
        )
        if starts_new_span:
            if active_label is not None and active_start is not None:
                spans.add((active_label, active_start, index))
            active_label = entity_label
            active_start = index

    if active_label is not None and active_start is not None:
        spans.add((active_label, active_start, len(tags)))

    return spans


def span_to_dict(
    span: tuple[str, int, int],
    *,
    tokens: Sequence[str],
) -> dict:
    label, start_token, end_token = span
    span_tokens = list(tokens[start_token:end_token])
    return {
        "label": label,
        "start_token": start_token,
        "end_token": end_token,
        "text": " ".join(span_tokens),
        "tokens": span_tokens,
    }


def render_tokens_with_highlights(
    tokens: Sequence[str],
    tags: Sequence[str],
) -> str:
    if len(tokens) != len(tags):
        raise ValueError("Tokens and tags must have the same length.")

    spans = _sorted_spans(bio_tags_to_spans(tags))
    span_starts = {start: span for span in spans for start in [span[1]]}
    span_ends = {end: span for span in spans for end in [span[2]]}

    rendered: list[str] = []
    for index, token in enumerate(tokens):
        if index in span_starts:
            rendered.append("[[")
        rendered.append(token)
        if index + 1 in span_ends:
            rendered.append("]]")
        if index != len(tokens) - 1:
            rendered.append(" ")
    return "".join(rendered)


def format_span_for_humans(
    span: dict,
) -> str:
    return (
        f"- {span['text']} "
        f"(label={span['label']}, tokens={span['start_token']}:{span['end_token']})"
    )


def build_fp_fn_record(
    *,
    sample_id: str,
    source: str,
    source_index: int,
    text: str,
    tokens: Sequence[str],
    gold_tags: Sequence[str],
    predicted_tags: Sequence[str],
    was_truncated: bool,
    original_token_count: int,
    kept_token_count: int,
) -> dict:
    if len(tokens) != len(gold_tags) or len(tokens) != len(predicted_tags):
        raise ValueError("Tokens, gold tags, and predicted tags must be aligned.")

    gold_spans = bio_tags_to_spans(gold_tags)
    predicted_spans = bio_tags_to_spans(predicted_tags)
    fp_spans = _sorted_spans(predicted_spans - gold_spans)
    fn_spans = _sorted_spans(gold_spans - predicted_spans)

    return {
        "sample_id": sample_id,
        "source": source,
        "source_index": source_index,
        "text": text,
        "tokens": list(tokens),
        "gold_tags": list(gold_tags),
        "predicted_tags": list(predicted_tags),
        "was_truncated": was_truncated,
        "original_token_count": original_token_count,
        "kept_token_count": kept_token_count,
        "counts": {
            "fp": len(fp_spans),
            "fn": len(fn_spans),
        },
        "fp": [span_to_dict(span, tokens=tokens) for span in fp_spans],
        "fn": [span_to_dict(span, tokens=tokens) for span in fn_spans],
    }


def build_fp_fn_rows(
    predictions: Sequence[dict],
    *,
    include_empty: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for prediction in predictions:
        row = build_fp_fn_record(
            sample_id=prediction["sample_id"],
            source=prediction["source"],
            source_index=prediction["source_index"],
            text=prediction["text"],
            tokens=prediction["tokens"],
            gold_tags=prediction["gold_tags"],
            predicted_tags=prediction["predicted_tags"],
            was_truncated=prediction["was_truncated"],
            original_token_count=prediction["original_token_count"],
            kept_token_count=prediction["kept_token_count"],
        )
        has_error = row["counts"]["fp"] > 0 or row["counts"]["fn"] > 0
        if not include_empty and not has_error:
            continue
        rows.append(row)
    return rows


def build_fp_fn_markdown_report(
    predictions: Sequence[dict],
    *,
    split_name: str = "val",
) -> str:
    rows = build_fp_fn_rows(
        predictions,
        include_empty=False,
    )
    rows.sort(
        key=lambda row: (
            row["counts"]["fp"] + row["counts"]["fn"],
            row["counts"]["fn"],
            row["counts"]["fp"],
            row["sample_id"],
        ),
        reverse=True,
    )

    total_fp = sum(row["counts"]["fp"] for row in rows)
    total_fn = sum(row["counts"]["fn"] for row in rows)

    lines = [
        f"# {split_name.upper()} FP / FN Report",
        "",
        "Человеко-читаемый отчёт по ошибкам модели на уровне span-ов.",
        "В отчёт включены только sample-ы, где есть хотя бы один FP или FN.",
        "",
        f"- samples_with_errors: {len(rows)}",
        f"- total_fp: {total_fp}",
        f"- total_fn: {total_fn}",
        "",
        "Обозначения:",
        "- `[[...]]` в строке `Gold` показывает gold-спаны.",
        "- `[[...]]` в строке `Pred` показывает предсказанные моделью спаны.",
        "",
    ]

    if not rows:
        lines.append("Ошибок на этом split не найдено.")
        lines.append("")
        return "\n".join(lines)

    for index, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {index}. {row['sample_id']}",
                "",
                f"- source: {row['source']}",
                f"- source_index: {row['source_index']}",
                f"- counts: FP={row['counts']['fp']} FN={row['counts']['fn']}",
                (
                    f"- truncated: yes "
                    f"(kept {row['kept_token_count']} of {row['original_token_count']} tokens)"
                    if row["was_truncated"]
                    else f"- truncated: no ({row['kept_token_count']} tokens)"
                ),
                "",
                "Text:",
                row["text"],
                "",
                "Gold:",
                render_tokens_with_highlights(row["tokens"], row["gold_tags"]),
                "",
                "Pred:",
                render_tokens_with_highlights(row["tokens"], row["predicted_tags"]),
            ]
        )

        lines.append("")
        lines.append("FP:")
        if row["fp"]:
            lines.extend(format_span_for_humans(span) for span in row["fp"])
        else:
            lines.append("- none")

        lines.append("")
        lines.append("FN:")
        if row["fn"]:
            lines.extend(format_span_for_humans(span) for span in row["fn"])
        else:
            lines.append("- none")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def compute_token_metrics(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
) -> dict[str, float | int]:
    if len(gold_sequences) != len(predicted_sequences):
        raise ValueError("Gold and predicted sequence lists must have the same size.")

    true_positive = 0
    false_positive = 0
    false_negative = 0
    correct = 0
    total = 0
    gold_entity_tokens = 0

    for gold_tags, predicted_tags in zip(gold_sequences, predicted_sequences):
        if len(gold_tags) != len(predicted_tags):
            raise ValueError("Gold and predicted token sequences must be aligned.")

        for gold_tag, predicted_tag in zip(gold_tags, predicted_tags):
            total += 1
            if gold_tag == predicted_tag:
                correct += 1

            if gold_tag != "O":
                gold_entity_tokens += 1

            if predicted_tag != "O":
                if predicted_tag == gold_tag:
                    true_positive += 1
                else:
                    false_positive += 1

            if gold_tag != "O" and predicted_tag != gold_tag:
                false_negative += 1

    metrics = _f1_from_counts(true_positive, false_positive, false_negative)
    metrics["accuracy"] = _safe_divide(correct, total)
    metrics["total_tokens"] = total
    metrics["gold_entity_tokens"] = gold_entity_tokens
    return metrics


def compute_span_metrics(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
) -> dict[str, float | int]:
    if len(gold_sequences) != len(predicted_sequences):
        raise ValueError("Gold and predicted sequence lists must have the same size.")

    true_positive = 0
    false_positive = 0
    false_negative = 0
    gold_span_count = 0
    predicted_span_count = 0

    for gold_tags, predicted_tags in zip(gold_sequences, predicted_sequences):
        gold_spans = bio_tags_to_spans(gold_tags)
        predicted_spans = bio_tags_to_spans(predicted_tags)
        true_positive += len(gold_spans & predicted_spans)
        false_positive += len(predicted_spans - gold_spans)
        false_negative += len(gold_spans - predicted_spans)
        gold_span_count += len(gold_spans)
        predicted_span_count += len(predicted_spans)

    metrics = _f1_from_counts(true_positive, false_positive, false_negative)
    metrics["gold_spans"] = gold_span_count
    metrics["predicted_spans"] = predicted_span_count
    return metrics


def compute_sequence_labeling_metrics(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
) -> dict[str, float | int]:
    token_metrics = compute_token_metrics(gold_sequences, predicted_sequences)
    span_metrics = compute_span_metrics(gold_sequences, predicted_sequences)

    return {
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


def has_annotation_kind_metadata(
    gold_sequences: Sequence[Sequence[str]],
    token_annotation_kind_sequences: Sequence[Sequence[str | None]],
) -> bool:
    if len(gold_sequences) != len(token_annotation_kind_sequences):
        raise ValueError(
            "Gold sequences and annotation-kind sequences must have the same size."
        )

    for gold_tags, token_annotation_kinds in zip(
        gold_sequences,
        token_annotation_kind_sequences,
    ):
        if len(gold_tags) != len(token_annotation_kinds):
            raise ValueError(
                "Gold tag sequences and annotation-kind sequences must be aligned."
            )
        for gold_tag, annotation_kind in zip(gold_tags, token_annotation_kinds):
            if gold_tag != "O" and annotation_kind is None:
                return False
    return True


def compute_subset_sequence_labeling_metrics(
    gold_sequences: Sequence[Sequence[str]],
    predicted_sequences: Sequence[Sequence[str]],
    token_annotation_kind_sequences: Sequence[Sequence[str | None]],
    *,
    allowed_annotation_kinds: set[str],
) -> dict[str, float | int]:
    if len(gold_sequences) != len(predicted_sequences):
        raise ValueError("Gold and predicted sequence lists must have the same size.")
    if len(gold_sequences) != len(token_annotation_kind_sequences):
        raise ValueError(
            "Gold sequences and annotation-kind sequences must have the same size."
        )

    filtered_gold_sequences: list[list[str]] = []
    filtered_predicted_sequences: list[list[str]] = []

    for gold_tags, predicted_tags, token_annotation_kinds in zip(
        gold_sequences,
        predicted_sequences,
        token_annotation_kind_sequences,
    ):
        if len(gold_tags) != len(predicted_tags):
            raise ValueError("Gold and predicted token sequences must be aligned.")
        if len(gold_tags) != len(token_annotation_kinds):
            raise ValueError(
                "Gold tag sequences and annotation-kind sequences must be aligned."
            )

        filtered_gold: list[str] = []
        filtered_predicted: list[str] = []
        for gold_tag, predicted_tag, annotation_kind in zip(
            gold_tags,
            predicted_tags,
            token_annotation_kinds,
        ):
            if (
                annotation_kind is not None
                and annotation_kind not in allowed_annotation_kinds
            ):
                continue
            filtered_gold.append(gold_tag)
            filtered_predicted.append(predicted_tag)

        filtered_gold_sequences.append(filtered_gold)
        filtered_predicted_sequences.append(filtered_predicted)

    metrics = compute_sequence_labeling_metrics(
        filtered_gold_sequences,
        filtered_predicted_sequences,
    )
    metrics["evaluated_sequences"] = len(filtered_gold_sequences)
    return metrics
