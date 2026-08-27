from __future__ import annotations

from typing import Any


LAMBDA_FUNCTIONAL = 0.001
EPSILON_W = 1e-8


def functional_loss(tf: Any, e_sender_to_receiver, e_receiver, epsilon_y, weights, epsilon_w: float = EPSILON_W):
    e_r = tf.stop_gradient(tf.convert_to_tensor(e_receiver, dtype=tf.float32))
    e_s = tf.convert_to_tensor(e_sender_to_receiver, dtype=tf.float32)
    eps_y = tf.stop_gradient(tf.cast(epsilon_y, tf.float32))
    a_i = tf.stop_gradient(tf.convert_to_tensor(weights, dtype=tf.float32))
    term = tf.square((e_s - e_r) / (tf.abs(e_r) + eps_y))
    numerator = tf.reduce_sum(a_i * term)
    denominator = tf.maximum(tf.reduce_sum(a_i), tf.constant(epsilon_w, dtype=tf.float32))
    return numerator / denominator


def scaled_functional_loss(tf: Any, *args, lambda_functional: float = LAMBDA_FUNCTIONAL, **kwargs):
    return tf.constant(lambda_functional, dtype=tf.float32) * functional_loss(tf, *args, **kwargs)

