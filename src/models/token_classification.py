from __future__ import annotations

from pathlib import Path

from transformers import AutoModelForTokenClassification, AutoTokenizer


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
