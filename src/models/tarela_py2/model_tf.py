from __future__ import annotations

import os
import sys
from typing import Any

from .config import PY2Config


def configure_deterministic_environment(seed: int) -> None:
    """Request legacy deterministic controls before TensorFlow import."""
    if "tensorflow" in sys.modules:
        raise RuntimeError(
            "TensorFlow was imported before deterministic environment setup. "
            "Run reference training in a fresh Python process."
        )
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def require_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required for PY-2 neural execution. Install a compatible "
            "TensorFlow/Keras runtime, then run the supplied PY-2 execution script."
        ) from exc
    return tf, layers


def configure_tensorflow_runtime(cfg: PY2Config):
    configure_deterministic_environment(cfg.random_seed)
    tf, _ = require_tensorflow()
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(cfg.random_seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
    return tf


def sparsemax_tf(logits: Any, tf: Any):
    # Exact computational semantics of the latest KNIME Python Script node.
    z = logits - tf.reduce_mean(logits, axis=-1, keepdims=True)
    z_sorted = tf.sort(z, direction="DESCENDING", axis=-1)
    z_cumsum = tf.cumsum(z_sorted, axis=-1)
    k = tf.cast(tf.range(1, tf.shape(z)[-1] + 1), logits.dtype)
    support = 1 + k * z_sorted > z_cumsum
    support_float = tf.cast(support, logits.dtype)
    k_z = tf.reduce_sum(support_float, axis=-1, keepdims=True)
    k_z = tf.maximum(k_z, 1.0)
    z_support_sum = tf.reduce_sum(
        tf.where(support, z_sorted, tf.zeros_like(z_sorted)), axis=-1, keepdims=True
    )
    tau = (z_support_sum - 1.0) / k_z
    return tf.maximum(z - tau, 0.0)


def build_tarela_model(cfg: PY2Config):
    cfg.validate()
    tf, layers = require_tensorflow()

    class SparseLatentRegimeEncoder(tf.keras.Model):
        def __init__(self, hidden_dim: int, topic_dim: int):
            super().__init__()
            self.dense1 = layers.Dense(hidden_dim, activation="relu")
            self.dense2 = layers.Dense(topic_dim)

        def call(self, x):
            h = self.dense1(x)
            logits = self.dense2(h)
            if cfg.topic_activation == "softmax":
                return tf.nn.softmax(logits, axis=-1)
            return sparsemax_tf(logits, tf)

    class TARELALSTM(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.regime_encoder = SparseLatentRegimeEncoder(cfg.encoder_hidden, cfg.topic_dim)
            self.lstm = layers.LSTM(cfg.lstm_units)
            self.output_layer = layers.Dense(1)
            self.dir_head = layers.Dense(3)
            self.vol_head = layers.Dense(3)
            self.amp_head = layers.Dense(1)

        def call(self, inputs, training=False):
            x_y_revin, x_num, x_doc = inputs
            theta_seq = self.regime_encoder(x_doc)
            x = tf.concat([x_y_revin, x_num, theta_seq], axis=-1)
            h = self.lstm(x, training=training)
            y_pred_revin = self.output_layer(h)
            theta_last = theta_seq[:, -1, :]
            dir_logits = self.dir_head(theta_last)
            vol_logits = self.vol_head(theta_last)
            amp_logit = self.amp_head(theta_last)
            return y_pred_revin, theta_seq, dir_logits, vol_logits, amp_logit

    return TARELALSTM()


def build_optimizer(cfg: PY2Config, tf: Any):
    # R1-resolved controlled semantics: Keras Adam optimizer-level clipnorm=1.0.
    return tf.keras.optimizers.Adam(
        learning_rate=cfg.learning_rate,
        clipnorm=cfg.gradient_clip_norm,
    )
