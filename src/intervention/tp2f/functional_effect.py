from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FunctionalEffect:
    y_R_native: Any
    y_R_neutral: Any
    y_S_to_R: Any
    e_R: Any
    e_S_to_R: Any
    epsilon_y: Any
    weights: Any


def receiver_forward_with_theta(tf: Any, receiver_model: Any, x_y_revin, x_num_scaled, theta_seq, *, training=False):
    fused = tf.concat(
        [
            tf.convert_to_tensor(x_y_revin, dtype=tf.float32),
            tf.convert_to_tensor(x_num_scaled, dtype=tf.float32),
            tf.convert_to_tensor(theta_seq, dtype=tf.float32),
        ],
        axis=-1,
    )
    hidden = receiver_model.lstm(fused, training=training)
    return receiver_model.output_layer(hidden), hidden


def neutral_theta_sequence(tf: Any, theta_train, target_shape):
    theta = tf.convert_to_tensor(theta_train, dtype=tf.float32)
    mean = tf.reduce_mean(theta, axis=[0, 1])
    mean = mean / tf.maximum(tf.reduce_sum(mean), tf.constant(1e-12, dtype=tf.float32))
    return tf.broadcast_to(tf.reshape(mean, [1, 1, -1]), target_shape)


def tensor_median(tf: Any, x):
    flat = tf.reshape(tf.cast(x, tf.float32), [-1])
    flat = tf.sort(flat)
    n = tf.shape(flat)[0]
    mid = n // 2
    odd = tf.equal(n % 2, 1)
    return tf.cond(odd, lambda: flat[mid], lambda: 0.5 * (flat[mid - 1] + flat[mid]))


def functional_effect_targets(
    tf: Any,
    receiver_model: Any,
    x_y_revin,
    x_num_scaled,
    x_doc_scaled,
    sender_theta_aligned,
    receiver_train_theta,
):
    receiver_pred, receiver_theta, *_ = receiver_model(
        (
            tf.convert_to_tensor(x_y_revin, dtype=tf.float32),
            tf.convert_to_tensor(x_num_scaled, dtype=tf.float32),
            tf.convert_to_tensor(x_doc_scaled, dtype=tf.float32),
        ),
        training=False,
    )
    receiver_pred = tf.stop_gradient(receiver_pred)
    receiver_theta = tf.stop_gradient(receiver_theta)
    neutral = neutral_theta_sequence(tf, receiver_train_theta, tf.shape(receiver_theta))
    y_neutral, _ = receiver_forward_with_theta(tf, receiver_model, x_y_revin, x_num_scaled, neutral, training=False)
    y_sender, _ = receiver_forward_with_theta(
        tf, receiver_model, x_y_revin, x_num_scaled, sender_theta_aligned, training=False
    )
    y_neutral = tf.stop_gradient(y_neutral)
    e_r = tf.stop_gradient(receiver_pred - y_neutral)
    e_s = y_sender - y_neutral
    eps_y = tf.maximum(
        tf.constant(1e-8, dtype=tf.float32),
        tf.constant(1e-6, dtype=tf.float32) * tf.stop_gradient(tensor_median(tf, tf.abs(receiver_pred))),
    )
    weights = tf.stop_gradient(tf.abs(e_r) / (tf.abs(e_r) + eps_y))
    return FunctionalEffect(
        y_R_native=receiver_pred,
        y_R_neutral=y_neutral,
        y_S_to_R=y_sender,
        e_R=e_r,
        e_S_to_R=e_s,
        epsilon_y=eps_y,
        weights=weights,
    )


def effect_identity_max_error(tf: Any, effect: FunctionalEffect):
    lhs = effect.y_S_to_R - effect.y_R_native
    rhs = effect.e_S_to_R - effect.e_R
    return tf.reduce_max(tf.abs(lhs - rhs))
