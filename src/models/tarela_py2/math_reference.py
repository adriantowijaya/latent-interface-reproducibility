from __future__ import annotations

import numpy as np


def sparsemax_numpy(logits: np.ndarray) -> np.ndarray:
    """NumPy diagnostic mirror of the legacy TensorFlow sparsemax."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.mean(z, axis=-1, keepdims=True)
    z_sorted = np.sort(z, axis=-1)[..., ::-1]
    z_cumsum = np.cumsum(z_sorted, axis=-1)
    k = np.arange(1, z.shape[-1] + 1, dtype=np.float64)
    support = 1.0 + k * z_sorted > z_cumsum
    k_z = np.maximum(np.sum(support, axis=-1, keepdims=True), 1.0)
    z_support_sum = np.sum(np.where(support, z_sorted, 0.0), axis=-1, keepdims=True)
    tau = (z_support_sum - 1.0) / k_z
    return np.maximum(z - tau, 0.0)


def theta_entropy_norm_numpy(theta_seq: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    theta = np.asarray(theta_seq, dtype=np.float64)
    k = theta.shape[-1]
    theta_clip = np.clip(theta, eps, 1.0)
    entropy = -np.sum(theta_clip * np.log(theta_clip), axis=-1)
    return entropy / np.log(k)


def expected_trainable_parameter_count(n_doc_features: int = 17, encoder_hidden: int = 8,
                                       topic_dim: int = 5, n_y_features: int = 1,
                                       n_num_features: int = 3, lstm_units: int = 50) -> int:
    encoder1 = n_doc_features * encoder_hidden + encoder_hidden
    encoder2 = encoder_hidden * topic_dim + topic_dim
    lstm_input = n_y_features + n_num_features + topic_dim
    lstm = 4 * (lstm_units * (lstm_input + lstm_units) + lstm_units)
    output = lstm_units * 1 + 1
    dir_head = topic_dim * 3 + 3
    vol_head = topic_dim * 3 + 3
    amp_head = topic_dim * 1 + 1
    return int(encoder1 + encoder2 + lstm + output + dir_head + vol_head + amp_head)
