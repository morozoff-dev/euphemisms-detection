from __future__ import annotations

__all__ = [
    "AUTO_HEAD_MODE",
    "HEAD_MODES",
    "LEGACY_MODEL_ARCHITECTURE",
    "MODEL_ARCHITECTURE",
    "EuphemismTokenClassifier",
    "EuphemismTokenClassifierConfig",
    "build_download_help_message",
    "build_model_metadata",
    "build_token_classifier",
    "build_tokenizer",
    "detect_checkpoint_head_info",
    "load_token_classifier_checkpoint",
    "resolve_checkpoint_head_info",
    "resolve_default_tokenizer_revision",
    "validate_head_mode",
]

_HEAD_EXPORTS = {
    "AUTO_HEAD_MODE",
    "HEAD_MODES",
    "LEGACY_MODEL_ARCHITECTURE",
    "MODEL_ARCHITECTURE",
    "EuphemismTokenClassifier",
    "EuphemismTokenClassifierConfig",
    "build_model_metadata",
    "detect_checkpoint_head_info",
    "load_token_classifier_checkpoint",
    "resolve_checkpoint_head_info",
    "validate_head_mode",
}

_TOKEN_CLASSIFICATION_EXPORTS = {
    "build_download_help_message",
    "build_token_classifier",
    "build_tokenizer",
    "resolve_default_tokenizer_revision",
}


def __getattr__(name: str):
    if name in _HEAD_EXPORTS:
        from src.models import heads

        value = getattr(heads, name)
        globals()[name] = value
        return value

    if name in _TOKEN_CLASSIFICATION_EXPORTS:
        from src.models import token_classification

        value = getattr(token_classification, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module 'src.models' has no attribute {name!r}")
