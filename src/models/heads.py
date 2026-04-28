from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.configuration_utils import PretrainedConfig


MODEL_ARCHITECTURE = "euphemism_token_classifier"
LEGACY_MODEL_ARCHITECTURE = "legacy_auto_token_classification"
HEAD_MODES = ("baseline", "neighbor", "combined")
AUTO_HEAD_MODE = "auto"
POSITIVE_LABEL = "EUPHEMISM"


@dataclass(frozen=True)
class CheckpointHeadInfo:
    checkpoint_architecture: str
    head_mode: str
    positive_label_id: int
    alpha: float | None
    is_legacy: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_architecture": self.checkpoint_architecture,
            "head_mode": self.head_mode,
            "positive_label_id": self.positive_label_id,
            "alpha": self.alpha,
            "is_legacy": self.is_legacy,
        }


def validate_head_mode(head_mode: str) -> str:
    if head_mode not in HEAD_MODES:
        raise ValueError(
            f"Unknown head mode: {head_mode!r}. "
            f"Expected one of: {', '.join(HEAD_MODES)}."
        )
    return head_mode


def validate_requested_head_mode(head_mode: str) -> str:
    if head_mode == AUTO_HEAD_MODE:
        return head_mode
    return validate_head_mode(head_mode)


def _coerce_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _positive_label_id_from_mapping(label2id: Any) -> int:
    if not isinstance(label2id, dict):
        return 1
    return _coerce_int(label2id.get(POSITIVE_LABEL), default=1)


def _dropout_from_encoder_config(encoder_config: PretrainedConfig) -> float:
    classifier_dropout = getattr(encoder_config, "classifier_dropout", None)
    if classifier_dropout is not None:
        return float(classifier_dropout)

    hidden_dropout_prob = getattr(encoder_config, "hidden_dropout_prob", None)
    if hidden_dropout_prob is not None:
        return float(hidden_dropout_prob)

    dropout = getattr(encoder_config, "dropout", None)
    if dropout is not None:
        return float(dropout)

    return 0.1


def _encoder_config_from_dict(payload: dict[str, Any]) -> PretrainedConfig:
    model_type = payload.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("Custom checkpoint config is missing encoder_config.model_type.")
    try:
        config_class = CONFIG_MAPPING[model_type]
    except KeyError as exc:
        raise ValueError(
            f"Transformers does not know encoder model_type={model_type!r}. "
            "Upgrade transformers or use the environment that created the checkpoint."
        ) from exc
    return config_class.from_dict(payload)


class EuphemismTokenClassifierConfig(PretrainedConfig):
    model_type = MODEL_ARCHITECTURE

    def __init__(
        self,
        *,
        encoder_config: dict[str, Any] | None = None,
        head_mode: str = "baseline",
        model_architecture: str = MODEL_ARCHITECTURE,
        positive_label_id: int = 1,
        classifier_dropout: float | None = None,
        raw_alpha_init: float = 0.0,
        alpha: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder_config = encoder_config
        self.head_mode = validate_head_mode(head_mode)
        self.model_architecture = model_architecture
        self.positive_label_id = int(positive_label_id)
        self.classifier_dropout = classifier_dropout
        self.raw_alpha_init = float(raw_alpha_init)
        self.alpha = alpha
        self.architectures = ["EuphemismTokenClassifier"]

    @classmethod
    def from_encoder_config(
        cls,
        encoder_config: PretrainedConfig,
        *,
        head_mode: str,
        id2label: dict[int, str],
        label2id: dict[str, int],
        positive_label_id: int,
        raw_alpha_init: float = 0.0,
    ) -> "EuphemismTokenClassifierConfig":
        return cls(
            encoder_config=encoder_config.to_dict(),
            head_mode=head_mode,
            model_architecture=MODEL_ARCHITECTURE,
            positive_label_id=positive_label_id,
            classifier_dropout=_dropout_from_encoder_config(encoder_config),
            raw_alpha_init=raw_alpha_init,
            alpha=(
                1.0 / (1.0 + math.exp(-float(raw_alpha_init)))
                if head_mode == "combined"
                else None
            ),
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
        )

    def build_encoder_config(self) -> PretrainedConfig:
        if self.encoder_config is None:
            raise ValueError("Custom model config is missing encoder_config.")
        return _encoder_config_from_dict(self.encoder_config)


class EuphemismTokenClassifier(PreTrainedModel):
    config_class = EuphemismTokenClassifierConfig
    base_model_prefix = "encoder"
    supports_gradient_checkpointing = True

    def __init__(self, config: EuphemismTokenClassifierConfig) -> None:
        super().__init__(config)
        self.head_mode = validate_head_mode(config.head_mode)
        encoder_config = config.build_encoder_config()
        self.encoder = AutoModel.from_config(encoder_config)

        hidden_size = getattr(encoder_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Encoder config must expose hidden_size.")
        dropout_prob = (
            float(config.classifier_dropout)
            if config.classifier_dropout is not None
            else _dropout_from_encoder_config(encoder_config)
        )

        if self.head_mode in {"baseline", "combined"}:
            self.baseline_dropout = nn.Dropout(dropout_prob)
            self.baseline_classifier = nn.Linear(hidden_size, 1)
        else:
            self.baseline_dropout = None
            self.baseline_classifier = None

        if self.head_mode in {"neighbor", "combined"}:
            self.neighbor_classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.Tanh(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, 1),
            )
        else:
            self.neighbor_classifier = None

        if self.head_mode == "combined":
            self.raw_alpha = nn.Parameter(torch.tensor(float(config.raw_alpha_init)))
        else:
            self.register_parameter("raw_alpha", None)

        self.post_init()

    @classmethod
    def from_encoder_pretrained(
        cls,
        model_name_or_path: str,
        *,
        head_mode: str,
        id2label: dict[int, str],
        label2id: dict[str, int],
        positive_label_id: int,
        raw_alpha_init: float = 0.0,
        model_revision: str | None = None,
        cache_dir: str | None = None,
        pretrained_kwargs: dict[str, Any] | None = None,
    ) -> "EuphemismTokenClassifier":
        validate_head_mode(head_mode)
        kwargs = dict(pretrained_kwargs or {})
        encoder_config = AutoConfig.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
        model_config = EuphemismTokenClassifierConfig.from_encoder_config(
            encoder_config,
            head_mode=head_mode,
            id2label=id2label,
            label2id=label2id,
            positive_label_id=positive_label_id,
            raw_alpha_init=raw_alpha_init,
        )
        model = cls(model_config)
        encoder_kwargs = dict(kwargs)
        if (
            model_revision is not None
            and "revision" not in encoder_kwargs
            and not Path(model_name_or_path).exists()
        ):
            encoder_kwargs["revision"] = model_revision
        if cache_dir is not None and "cache_dir" not in encoder_kwargs:
            encoder_kwargs["cache_dir"] = cache_dir
        model.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            config=encoder_config,
            ignore_mismatched_sizes=True,
            **encoder_kwargs,
        )
        return model

    def get_alpha(self) -> float | None:
        if self.raw_alpha is None:
            return None
        return float(torch.sigmoid(self.raw_alpha.detach()).cpu().item())

    def _sync_config_runtime_metadata(self) -> None:
        self.config.model_architecture = MODEL_ARCHITECTURE
        self.config.head_mode = self.head_mode
        self.config.positive_label_id = int(self.config.positive_label_id)
        self.config.alpha = self.get_alpha() if self.head_mode == "combined" else None

    def save_pretrained(self, *args: Any, **kwargs: Any) -> None:
        self._sync_config_runtime_metadata()
        super().save_pretrained(*args, **kwargs)

    @staticmethod
    def build_neighbor_representations(
        hidden_states: torch.Tensor,
        word_start_mask: torch.Tensor,
    ) -> torch.Tensor:
        if word_start_mask.shape != hidden_states.shape[:2]:
            raise ValueError(
                "word_start_mask must have shape [batch, seq] aligned with hidden states."
            )

        neighbor_states = hidden_states.new_zeros(hidden_states.shape)
        for batch_index in range(hidden_states.shape[0]):
            positions = torch.nonzero(
                word_start_mask[batch_index].bool(),
                as_tuple=False,
            ).flatten()
            if positions.numel() == 0:
                continue

            word_states = hidden_states[batch_index, positions]
            replacements = torch.zeros_like(word_states)
            if positions.numel() == 2:
                replacements[0] = word_states[1]
                replacements[1] = word_states[0]
            elif positions.numel() > 2:
                replacements[0] = word_states[1]
                replacements[-1] = word_states[-2]
                replacements[1:-1] = (word_states[:-2] + word_states[2:]) * 0.5

            neighbor_states[batch_index, positions] = replacements

        return neighbor_states

    def _compute_positive_logits(
        self,
        hidden_states: torch.Tensor,
        word_start_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.head_mode == "baseline":
            if self.baseline_dropout is None or self.baseline_classifier is None:
                raise RuntimeError("baseline head is not initialized.")
            baseline_logits = self.baseline_classifier(
                self.baseline_dropout(hidden_states)
            ).squeeze(-1)
            return baseline_logits

        if word_start_mask is None:
            raise ValueError(
                "word_start_mask is required for neighbor and combined head modes."
            )
        neighbor_states = self.build_neighbor_representations(
            hidden_states,
            word_start_mask,
        )
        if self.neighbor_classifier is None:
            raise RuntimeError("neighbor head is not initialized.")
        neighbor_logits = self.neighbor_classifier(neighbor_states).squeeze(-1)
        if self.head_mode == "neighbor":
            return neighbor_logits

        if self.baseline_dropout is None or self.baseline_classifier is None:
            raise RuntimeError("baseline head is not initialized.")
        baseline_logits = self.baseline_classifier(
            self.baseline_dropout(hidden_states)
        ).squeeze(-1)
        if self.raw_alpha is None:
            raise RuntimeError("combined head is missing raw_alpha parameter.")
        alpha = torch.sigmoid(self.raw_alpha)
        return alpha * baseline_logits + (1.0 - alpha) * neighbor_logits

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        word_start_mask: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> TokenClassifierOutput:
        encoder_inputs: dict[str, Any] = {}
        if input_ids is not None:
            encoder_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            encoder_inputs["attention_mask"] = attention_mask
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        if position_ids is not None:
            encoder_inputs["position_ids"] = position_ids
        if inputs_embeds is not None:
            encoder_inputs["inputs_embeds"] = inputs_embeds
        if output_attentions is not None:
            encoder_inputs["output_attentions"] = output_attentions
        if output_hidden_states is not None:
            encoder_inputs["output_hidden_states"] = output_hidden_states
        encoder_inputs.update(kwargs)
        encoder_inputs["return_dict"] = True

        encoder_outputs = self.encoder(**encoder_inputs)
        hidden_states = encoder_outputs.last_hidden_state
        positive_logits = self._compute_positive_logits(hidden_states, word_start_mask)
        logits = positive_logits.new_zeros(
            (*positive_logits.shape, int(self.config.num_labels))
        )
        logits[..., int(self.config.positive_label_id)] = positive_logits

        loss = None
        if labels is not None:
            active_mask = labels.ne(-100)
            if int(active_mask.sum().item()) == 0:
                loss = positive_logits.sum() * 0.0
            else:
                targets = labels.eq(int(self.config.positive_label_id)).to(
                    dtype=positive_logits.dtype
                )
                loss = nn.functional.binary_cross_entropy_with_logits(
                    positive_logits[active_mask],
                    targets[active_mask],
                )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def build_model_metadata(model: torch.nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    head_mode = getattr(config, "head_mode", "baseline")
    architecture = getattr(config, "model_architecture", MODEL_ARCHITECTURE)
    positive_label_id = getattr(config, "positive_label_id", None)
    if positive_label_id is None:
        positive_label_id = _positive_label_id_from_mapping(
            getattr(config, "label2id", None)
        )

    alpha = None
    if head_mode == "combined":
        get_alpha = getattr(model, "get_alpha", None)
        if callable(get_alpha):
            alpha = get_alpha()
        else:
            alpha = getattr(config, "alpha", None)

    return {
        "model_architecture": architecture,
        "head_mode": head_mode,
        "positive_label_id": int(positive_label_id),
        "alpha": alpha,
    }


def _read_checkpoint_config(checkpoint_dir: str | Path) -> dict[str, Any]:
    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint config.json does not exist: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def detect_checkpoint_head_info(checkpoint_dir: str | Path) -> CheckpointHeadInfo:
    payload = _read_checkpoint_config(checkpoint_dir)
    is_custom = (
        payload.get("model_type") == MODEL_ARCHITECTURE
        or payload.get("model_architecture") == MODEL_ARCHITECTURE
    )
    positive_label_id = _coerce_int(
        payload.get("positive_label_id"),
        default=_positive_label_id_from_mapping(payload.get("label2id")),
    )

    if not is_custom:
        return CheckpointHeadInfo(
            checkpoint_architecture=LEGACY_MODEL_ARCHITECTURE,
            head_mode="baseline",
            positive_label_id=positive_label_id,
            alpha=None,
            is_legacy=True,
        )

    head_mode = validate_head_mode(str(payload.get("head_mode", "baseline")))
    alpha = payload.get("alpha") if head_mode == "combined" else None
    if alpha is not None:
        alpha = float(alpha)
    return CheckpointHeadInfo(
        checkpoint_architecture=str(
            payload.get("model_architecture", MODEL_ARCHITECTURE)
        ),
        head_mode=head_mode,
        positive_label_id=positive_label_id,
        alpha=alpha,
        is_legacy=False,
    )


def resolve_checkpoint_head_info(
    checkpoint_dir: str | Path,
    *,
    requested_head_mode: str = AUTO_HEAD_MODE,
) -> CheckpointHeadInfo:
    requested_head_mode = validate_requested_head_mode(requested_head_mode)
    info = detect_checkpoint_head_info(checkpoint_dir)
    if requested_head_mode == AUTO_HEAD_MODE:
        return info

    if info.is_legacy:
        if requested_head_mode != "baseline":
            raise ValueError(
                "Checkpoint is a legacy Hugging Face token-classification model and "
                "can only be used with --head-mode baseline or --head-mode auto. "
                f"Requested --head-mode {requested_head_mode!r}."
            )
        return info

    if info.head_mode != requested_head_mode:
        raise ValueError(
            "Checkpoint head-mode mismatch: "
            f"checkpoint has {info.head_mode!r}, requested {requested_head_mode!r}."
        )
    return info


def load_token_classifier_checkpoint(
    checkpoint_dir: str | Path,
    *,
    requested_head_mode: str = AUTO_HEAD_MODE,
) -> tuple[torch.nn.Module, CheckpointHeadInfo]:
    info = resolve_checkpoint_head_info(
        checkpoint_dir,
        requested_head_mode=requested_head_mode,
    )
    if info.is_legacy:
        model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)
        return model, info

    config = EuphemismTokenClassifierConfig.from_pretrained(checkpoint_dir)
    model = EuphemismTokenClassifier.from_pretrained(checkpoint_dir, config=config)
    metadata = build_model_metadata(model)
    return model, CheckpointHeadInfo(
        checkpoint_architecture=info.checkpoint_architecture,
        head_mode=metadata["head_mode"],
        positive_label_id=metadata["positive_label_id"],
        alpha=metadata["alpha"],
        is_legacy=False,
    )
