#!/usr/bin/env python3
from __future__ import annotations

import math
import tempfile
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from transformers import AutoModelForTokenClassification, BertConfig

from src.models.heads import (
    EuphemismTokenClassifier,
    EuphemismTokenClassifierConfig,
    load_token_classifier_checkpoint,
)


ID_TO_LABEL = {0: "O", 1: "EUPHEMISM"}
LABEL_TO_ID = {"O": 0, "EUPHEMISM": 1}


def build_encoder_config() -> BertConfig:
    return BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        pad_token_id=0,
    )


def build_model(head_mode: str) -> EuphemismTokenClassifier:
    config = EuphemismTokenClassifierConfig.from_encoder_config(
        build_encoder_config(),
        head_mode=head_mode,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        positive_label_id=LABEL_TO_ID["EUPHEMISM"],
    )
    return EuphemismTokenClassifier(config)


def assert_close(left: torch.Tensor, right: torch.Tensor) -> None:
    if not torch.allclose(left, right, atol=1e-6):
        raise AssertionError(f"Expected {left.item()} to match {right.item()}.")


def assert_raises(expected_message: str, fn) -> None:
    try:
        fn()
    except ValueError as exc:
        if expected_message not in str(exc):
            raise AssertionError(f"Unexpected error message: {exc}") from exc
        return
    raise AssertionError("Expected ValueError was not raised.")


def check_forward_modes() -> None:
    input_ids = torch.tensor(
        [
            [2, 5, 6, 7, 3],
            [2, 8, 3, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    word_start_mask = torch.tensor(
        [
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [
            [-100, 0, -100, 1, -100],
            [-100, 1, -100, -100, -100],
        ],
        dtype=torch.long,
    )

    for head_mode in ("baseline", "neighbor", "combined"):
        model = build_model(head_mode)
        model.eval()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            word_start_mask=word_start_mask,
            labels=labels,
        )
        if tuple(outputs.logits.shape) != (2, 5, 2):
            raise AssertionError(f"{head_mode}: bad logits shape {outputs.logits.shape}.")
        if outputs.loss is None or not math.isfinite(float(outputs.loss.item())):
            raise AssertionError(f"{head_mode}: loss is missing or non-finite.")

        active_mask = labels != -100
        targets = (labels == LABEL_TO_ID["EUPHEMISM"]).to(
            dtype=outputs.logits.dtype
        )
        manual_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs.logits[..., 1][active_mask],
            targets[active_mask],
        )
        assert_close(outputs.loss, manual_loss)


def check_neighbor_fallbacks() -> None:
    hidden_states = torch.arange(1 * 6 * 3, dtype=torch.float32).reshape(1, 6, 3)
    word_start_mask = torch.tensor([[0, 1, 0, 1, 1, 0]], dtype=torch.long)
    neighbors = EuphemismTokenClassifier.build_neighbor_representations(
        hidden_states,
        word_start_mask,
    )
    if not torch.equal(neighbors[0, 1], hidden_states[0, 3]):
        raise AssertionError("First word should use the next word-start state.")
    expected_middle = (hidden_states[0, 1] + hidden_states[0, 4]) * 0.5
    if not torch.equal(neighbors[0, 3], expected_middle):
        raise AssertionError("Middle word should average previous and next states.")
    if not torch.equal(neighbors[0, 4], hidden_states[0, 3]):
        raise AssertionError("Last word should use the previous word-start state.")

    single_word_mask = torch.tensor([[0, 0, 1, 0, 0, 0]], dtype=torch.long)
    single_neighbors = EuphemismTokenClassifier.build_neighbor_representations(
        hidden_states,
        single_word_mask,
    )
    if not torch.equal(single_neighbors[0, 2], torch.zeros(3)):
        raise AssertionError("Single-word input should use a zero neighbor vector.")


def check_checkpoint_compatibility() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)

        custom_dir = root / "custom_combined"
        build_model("combined").save_pretrained(custom_dir)
        _, metadata = load_token_classifier_checkpoint(
            custom_dir,
            requested_head_mode="combined",
        )
        if metadata.is_legacy or metadata.head_mode != "combined":
            raise AssertionError("Custom combined checkpoint did not load as combined.")
        assert_raises(
            "Checkpoint head-mode mismatch",
            lambda: load_token_classifier_checkpoint(
                custom_dir,
                requested_head_mode="baseline",
            ),
        )

        legacy_config = build_encoder_config()
        legacy_config.num_labels = 2
        legacy_config.id2label = ID_TO_LABEL
        legacy_config.label2id = LABEL_TO_ID
        legacy_dir = root / "legacy_baseline"
        AutoModelForTokenClassification.from_config(legacy_config).save_pretrained(
            legacy_dir
        )
        _, legacy_metadata = load_token_classifier_checkpoint(
            legacy_dir,
            requested_head_mode="baseline",
        )
        if not legacy_metadata.is_legacy or legacy_metadata.head_mode != "baseline":
            raise AssertionError("Legacy checkpoint did not resolve to baseline.")
        assert_raises(
            "legacy Hugging Face token-classification",
            lambda: load_token_classifier_checkpoint(
                legacy_dir,
                requested_head_mode="neighbor",
            ),
        )


def main() -> int:
    torch.manual_seed(7)
    check_forward_modes()
    check_neighbor_fallbacks()
    check_checkpoint_compatibility()
    print("custom head smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
