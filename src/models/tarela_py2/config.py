from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class PY2Config:
    # Neural architecture
    lstm_units: int = 50
    encoder_hidden: int = 8
    topic_dim: int = 5
    topic_activation: str = "sparsemax"

    # Objective: selected full TARELA-LSTM C1 reference
    lambda_entropy: float = 1e-3
    entropy_target: float = 0.35
    lambda_dir: float = 1e-3
    lambda_vol: float = 1e-3
    lambda_amp: float = 1e-3
    use_weighted_aux_loss: bool = True
    max_class_weight: float = 5.0

    # Training governance
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 1.0
    gradient_clip_mode: str = "Adam optimizer clipnorm"
    random_seed: int = 42
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-6
    shuffle: bool = False
    deterministic_operations_requested: bool = True

    # Data/model restoration
    revin_eps: float = 1e-5
    feature_scale_eps: float = 1e-8
    scale_doc: bool = True
    use_clipping: bool = True
    expected_validation_targets: int = 28
    expected_test_targets: int = 28

    # Controlled dimensions inherited from PY-1
    lookback: int = 7
    n_y_features: int = 1
    n_num_features: int = 3
    n_doc_features: int = 17

    reference_name: str = "TARELA-LSTM-v1.0-C1"

    def validate(self) -> None:
        if self.lookback != 7:
            raise ValueError("Reference lookback must be 7.")
        if (self.n_y_features, self.n_num_features, self.n_doc_features) != (1, 3, 17):
            raise ValueError("Reference input dimensions must be (1,3,17).")
        if self.topic_dim != 5 or self.encoder_hidden != 8 or self.lstm_units != 50:
            raise ValueError("Reference architecture must be K=5, encoder_hidden=8, LSTM=50.")
        if self.topic_activation not in {"sparsemax", "softmax"}:
            raise ValueError("topic_activation must be sparsemax or softmax.")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")
        if self.gradient_clip_mode != "Adam optimizer clipnorm":
            raise ValueError("Reference clipping semantics are Keras Adam optimizer clipnorm.")
        if self.shuffle:
            raise ValueError("Reference training must not shuffle sequences.")
        if self.patience != 10 or self.min_delta != 1e-6:
            raise ValueError("Reference early-stopping governance changed.")
        if self.revin_eps != 1e-5:
            raise ValueError("Reference RevIN epsilon must be 1e-5.")
        if self.max_class_weight != 5.0:
            raise ValueError("Reference maximum class weight must be 5.0.")

    @classmethod
    def mse_only(cls) -> "PY2Config":
        return replace(
            cls(),
            lambda_entropy=0.0,
            lambda_dir=0.0,
            lambda_vol=0.0,
            lambda_amp=0.0,
            reference_name="TARELA-LSTM-v1.0-MSE-only",
        )

    def objective_variant(self) -> str:
        lambdas = [self.lambda_entropy, self.lambda_dir, self.lambda_vol, self.lambda_amp]
        return "MSE-only" if all(v == 0.0 for v in lambdas) else "full transition-aware objective"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["objective_variant"] = self.objective_variant()
        return d

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
