from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForTokenClassification, AutoTokenizer

from src.models.heads import (
    AUTO_HEAD_MODE,
    EuphemismTokenClassifier,
    resolve_checkpoint_head_info,
)


def is_local_reference(reference: str) -> bool:
    return Path(reference).exists()


def resolve_default_tokenizer_revision(
    model_name_or_path: str,
    tokenizer_revision: str | None,
) -> str | None:
    if tokenizer_revision is not None:
        return tokenizer_revision
    if model_name_or_path.startswith("deepvk/RuModernBERT-"):
        return "patched-tokenizer"
    return None


def _build_pretrained_kwargs(
    reference: str,
    *,
    revision: str | None,
    cache_dir: str | None,
) -> dict:
    kwargs: dict[str, str] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if revision is not None and not is_local_reference(reference):
        kwargs["revision"] = revision
    return kwargs


def build_tokenizer(
    *,
    model_name_or_path: str,
    tokenizer_name_or_path: str | None = None,
    tokenizer_revision: str | None = None,
    cache_dir: str | None = None,
):
    tokenizer_reference = tokenizer_name_or_path or model_name_or_path
    resolved_revision = resolve_default_tokenizer_revision(
        tokenizer_reference,
        tokenizer_revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_reference,
        use_fast=True,
        **_build_pretrained_kwargs(
            tokenizer_reference,
            revision=resolved_revision,
            cache_dir=cache_dir,
        ),
    )
    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast tokenizer is required for word-to-subword alignment."
        )
    return tokenizer, resolved_revision


def build_token_classifier(
    *,
    model_name_or_path: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    head_mode: str = "baseline",
    positive_label_id: int = 1,
    model_revision: str | None = None,
    cache_dir: str | None = None,
):
    if num_labels != 2:
        raise ValueError(
            "The euphemism token classifier is binary and expects exactly 2 labels."
        )
    return EuphemismTokenClassifier.from_encoder_pretrained(
        model_name_or_path,
        id2label=id2label,
        label2id=label2id,
        head_mode=head_mode,
        positive_label_id=positive_label_id,
        model_revision=model_revision,
        cache_dir=cache_dir,
        pretrained_kwargs=_build_pretrained_kwargs(
            model_name_or_path,
            revision=model_revision,
            cache_dir=cache_dir,
        ),
    )


def build_legacy_token_classifier(
    *,
    model_name_or_path: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
    model_revision: str | None = None,
    cache_dir: str | None = None,
):
    return AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        **_build_pretrained_kwargs(
            model_name_or_path,
            revision=model_revision,
            cache_dir=cache_dir,
        ),
    )


def checkpoint_uses_word_start_mask(
    checkpoint_dir: str | Path,
    *,
    requested_head_mode: str = AUTO_HEAD_MODE,
) -> bool:
    metadata = resolve_checkpoint_head_info(
        checkpoint_dir,
        requested_head_mode=requested_head_mode,
    )
    return not metadata.is_legacy


def build_download_help_message(
    *,
    model_name_or_path: str,
    tokenizer_name_or_path: str | None,
    model_revision: str | None,
    tokenizer_revision: str | None,
    cache_dir: str | None,
) -> str:
    tokenizer_reference = tokenizer_name_or_path or model_name_or_path
    resolved_tokenizer_revision = resolve_default_tokenizer_revision(
        tokenizer_reference,
        tokenizer_revision,
    )

    lines = [
        "Could not load the pretrained model/tokenizer automatically.",
        f"Model reference: {model_name_or_path}",
        f"Model revision: {model_revision or 'main'}",
        f"Tokenizer reference: {tokenizer_reference}",
        f"Tokenizer revision: {resolved_tokenizer_revision or 'main'}",
        "",
        "If network access is unavailable, download these Hugging Face snapshots manually",
        "and then rerun the CLI with local paths via --model-name and --tokenizer-name.",
    ]
    if cache_dir is not None:
        lines.append(f"Requested cache dir: {cache_dir}")
    return "\n".join(lines)
