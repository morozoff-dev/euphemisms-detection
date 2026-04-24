#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import regex as re_module
except ImportError:  # pragma: no cover - depends on local environment
    import re as re_module


try:
    WORD_RE = re_module.compile(r"\p{L}+(?:-\p{L}+)*", flags=re_module.UNICODE)
except Exception:  # pragma: no cover - stdlib re does not support \p{L}
    WORD_RE = re_module.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*", flags=re_module.UNICODE)


@dataclass(frozen=True)
class TokenSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class PredictedSpan:
    label: str
    start_token: int
    end_token: int
    start_char: int
    end_char: int
    text: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference for one text from a .txt file using a locally trained "
            "token-classification checkpoint."
        )
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help=(
            "Path to a training run directory or directly to best_model/. "
            "Examples: outputs/models/<run_name> or outputs/models/<run_name>/best_model"
        ),
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Path to a .txt file containing one text. Line breaks are preserved.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the full prediction payload as JSON.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device string: auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=(
            "Max transformer sequence length. "
            "If omitted, the script tries to read it from run_config.json and falls back to 256."
        ),
    )
    parser.add_argument(
        "--window-overlap-words",
        type=int,
        default=32,
        help="How many words to overlap between inference windows for long texts.",
    )
    parser.add_argument(
        "--print-tags",
        action="store_true",
        help="Print the predicted BIO tag for every word token.",
    )
    parser.add_argument(
        "--hide-highlight",
        action="store_true",
        help="Do not print the input text with [[...]] span highlights.",
    )
    return parser


def clean_input_text(text: str) -> str:
    return unicodedata.normalize(
        "NFC",
        text.replace("\ufeff", "").replace("\u00A0", " "),
    )


def load_text(path: str | Path) -> str:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input text file does not exist: {input_path}")
    return clean_input_text(input_path.read_text(encoding="utf-8"))


def tokenize_words(text: str) -> list[TokenSpan]:
    return [
        TokenSpan(text=match.group(0), start=match.start(), end=match.end())
        for match in WORD_RE.finditer(text)
    ]


def resolve_checkpoint_dir(path: str | Path) -> tuple[Path, Path | None]:
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"Model path does not exist: {candidate}")

    best_model_dir = candidate / "best_model"
    if best_model_dir.is_dir() and (best_model_dir / "config.json").exists():
        return best_model_dir, candidate

    if (candidate / "config.json").exists():
        run_dir = candidate.parent if candidate.name == "best_model" else None
        return candidate, run_dir

    raise FileNotFoundError(
        "Could not find a checkpoint. Pass either a run directory containing "
        f"'best_model/' or a checkpoint directory with config.json: {candidate}"
    )


def resolve_max_length(
    *,
    cli_value: int | None,
    run_dir: Path | None,
) -> int:
    if cli_value is not None:
        if cli_value <= 0:
            raise ValueError("--max-length must be positive.")
        return cli_value

    if run_dir is not None:
        run_config_path = run_dir / "run_config.json"
        if run_config_path.exists():
            payload = json.loads(run_config_path.read_text(encoding="utf-8"))
            config_value = payload.get("config", {}).get("max_length")
            if isinstance(config_value, int) and config_value > 0:
                return config_value

    return 256


def resolve_device(requested_device: str, torch_module):
    if requested_device == "auto":
        return torch_module.device(
            "cuda" if torch_module.cuda.is_available() else "cpu"
        )

    device = torch_module.device(requested_device)
    if device.type == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return device


def normalize_bio_tags(tags: list[str]) -> list[str]:
    normalized: list[str] = []
    active_label: str | None = None

    for tag in tags:
        if tag == "O":
            normalized.append("O")
            active_label = None
            continue

        prefix, _, entity_label = tag.partition("-")
        if not entity_label:
            prefix = "B"
            entity_label = tag

        if prefix not in {"B", "I"}:
            prefix = "B"

        if prefix == "I" and active_label != entity_label:
            prefix = "B"

        normalized_tag = f"{prefix}-{entity_label}"
        normalized.append(normalized_tag)
        active_label = entity_label

    return normalized


def bio_tags_to_token_spans(tags: list[str]) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    active_label: str | None = None
    active_start: int | None = None

    for index, tag in enumerate(tags):
        if tag == "O":
            if active_label is not None and active_start is not None:
                spans.append((active_label, active_start, index))
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
                spans.append((active_label, active_start, index))
            active_label = entity_label
            active_start = index

    if active_label is not None and active_start is not None:
        spans.append((active_label, active_start, len(tags)))

    return spans


def build_predicted_spans(
    *,
    text: str,
    tokens: list[TokenSpan],
    tags: list[str],
) -> list[PredictedSpan]:
    spans: list[PredictedSpan] = []
    for label, start_token, end_token in bio_tags_to_token_spans(tags):
        start_char = tokens[start_token].start
        end_char = tokens[end_token - 1].end
        spans.append(
            PredictedSpan(
                label=label,
                start_token=start_token,
                end_token=end_token,
                start_char=start_char,
                end_char=end_char,
                text=text[start_char:end_char],
            )
        )
    return spans


def render_highlighted_text(text: str, spans: list[PredictedSpan]) -> str:
    highlighted = text
    for span in sorted(spans, key=lambda item: (item.start_char, item.end_char), reverse=True):
        highlighted = (
            highlighted[: span.end_char]
            + "]]"
            + highlighted[span.end_char :]
        )
        highlighted = (
            highlighted[: span.start_char]
            + "[["
            + highlighted[span.start_char :]
        )
    return highlighted


def predict_tags_for_tokens(
    *,
    tokens: list[TokenSpan],
    model,
    tokenizer,
    torch_module,
    device,
    max_length: int,
    window_overlap_words: int,
) -> tuple[list[str], int]:
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if window_overlap_words < 0:
        raise ValueError("window_overlap_words must be non-negative.")

    words = [token.text for token in tokens]
    score_sums = torch_module.zeros((len(words), model.config.num_labels))
    vote_counts = torch_module.zeros(len(words), dtype=torch_module.int64)
    chunk_count = 0
    start = 0

    model.eval()
    id2label = {
        int(label_id): label
        for label_id, label in model.config.id2label.items()
    }

    with torch_module.no_grad():
        while start < len(words):
            encoding = tokenizer(
                words[start:],
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
                return_tensors="pt",
            )
            word_ids = encoding.word_ids(batch_index=0)
            if word_ids is None:
                raise RuntimeError(
                    "The tokenizer did not return word ids. "
                    "Use a fast tokenizer for token classification."
                )

            first_token_positions: list[tuple[int, int]] = []
            previous_word_id: int | None = None
            for position, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id != previous_word_id:
                    first_token_positions.append((word_id, position))
                    previous_word_id = word_id

            if not first_token_positions:
                raise RuntimeError(
                    "The tokenizer produced no usable word-level positions for inference."
                )

            batch = {
                key: value.to(device)
                for key, value in encoding.items()
            }
            probabilities = model(**batch).logits[0].detach().cpu().softmax(dim=-1)

            for relative_word_index, position in first_token_positions:
                global_word_index = start + relative_word_index
                score_sums[global_word_index] += probabilities[position]
                vote_counts[global_word_index] += 1

            chunk_count += 1
            last_relative_word_index = first_token_positions[-1][0]
            end = start + last_relative_word_index + 1
            if end >= len(words):
                break

            if window_overlap_words == 0:
                start = end
            else:
                start = max(start + 1, end - window_overlap_words)

    if int((vote_counts == 0).sum().item()) > 0:
        raise RuntimeError("Some words were not covered by inference windows.")

    average_scores = score_sums / vote_counts.unsqueeze(-1)
    raw_tags = [
        id2label[int(label_id)]
        for label_id in average_scores.argmax(dim=-1).tolist()
    ]
    return normalize_bio_tags(raw_tags), chunk_count


def write_json(path: str | Path, payload: dict) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()

    try:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ModuleNotFoundError as exc:
        print(
            "Missing inference dependencies. Install PyTorch and transformers first, "
            "for example: venv/bin/pip install -r requirements.txt",
            file=sys.stderr,
        )
        print(f"Original import error: {exc}", file=sys.stderr)
        return 1

    try:
        checkpoint_dir, run_dir = resolve_checkpoint_dir(args.model_dir)
        max_length = resolve_max_length(
            cli_value=args.max_length,
            run_dir=run_dir,
        )
        input_text = load_text(args.input_path)
        tokens = tokenize_words(input_text)
        if not tokens:
            raise ValueError("The input text does not contain any word tokens.")

        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
        if not tokenizer.is_fast:
            raise RuntimeError(
                "A fast tokenizer is required for token classification inference."
            )
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
        device = resolve_device(args.device, torch)
        model.to(device)

        predicted_tags, chunk_count = predict_tags_for_tokens(
            tokens=tokens,
            model=model,
            tokenizer=tokenizer,
            torch_module=torch,
            device=device,
            max_length=max_length,
            window_overlap_words=args.window_overlap_words,
        )
        predicted_spans = build_predicted_spans(
            text=input_text,
            tokens=tokens,
            tags=predicted_tags,
        )

        result = {
            "model_dir": str(checkpoint_dir),
            "run_dir": str(run_dir) if run_dir is not None else None,
            "input_path": str(Path(args.input_path)),
            "device": str(device),
            "max_length": max_length,
            "window_overlap_words": args.window_overlap_words,
            "chunk_count": chunk_count,
            "text": input_text,
            "tokens": [token.text for token in tokens],
            "predicted_tags": predicted_tags,
            "predicted_entities": [asdict(span) for span in predicted_spans],
        }
        if args.output_json is not None:
            write_json(args.output_json, result)

        print(f"Модель: {checkpoint_dir}")
        if run_dir is not None:
            print(f"Run directory: {run_dir}")
        print(f"Входной файл: {Path(args.input_path)}")
        print(f"Устройство: {device}")
        print(f"max_length: {max_length}")
        print(f"window_overlap_words: {args.window_overlap_words}")
        print(f"Словарных токенов: {len(tokens)}")
        print(f"Окон инференса: {chunk_count}")
        print(f"Найдено сущностей: {len(predicted_spans)}")

        if predicted_spans:
            print()
            print("Предсказанные сущности:")
            for index, span in enumerate(predicted_spans, start=1):
                print(
                    f"{index}. {span.text!r} | label={span.label} | "
                    f"chars={span.start_char}:{span.end_char} | "
                    f"tokens={span.start_token}:{span.end_token}"
                )
        else:
            print()
            print("Предсказанные сущности не найдены.")

        if args.print_tags:
            print()
            print("Токены и BIO-теги:")
            for index, (token, tag) in enumerate(
                zip((token.text for token in tokens), predicted_tags),
                start=0,
            ):
                print(f"{index}\t{token}\t{tag}")

        if not args.hide_highlight:
            print()
            print("Текст с подсветкой:")
            print(render_highlighted_text(input_text, predicted_spans))

        if args.output_json is not None:
            print()
            print(f"JSON сохранён в: {Path(args.output_json)}")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
