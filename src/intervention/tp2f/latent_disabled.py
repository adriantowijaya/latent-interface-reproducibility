from __future__ import annotations

from typing import Any

from .functional_effect import neutral_theta_sequence


def latent_disabled_forecaster_theta(tf: Any, encoder_theta_train, target_shape):
    return neutral_theta_sequence(tf, encoder_theta_train, target_shape)


def latent_disabled_forward(tf: Any, model: Any, x_y_revin, x_num_scaled, x_doc_scaled, encoder_theta_train):
    _, theta_seq, dir_logits, vol_logits, amp_logit = model(
        (
            tf.convert_to_tensor(x_y_revin, dtype=tf.float32),
            tf.convert_to_tensor(x_num_scaled, dtype=tf.float32),
            tf.convert_to_tensor(x_doc_scaled, dtype=tf.float32),
        ),
        training=False,
    )
    theta_forecaster = latent_disabled_forecaster_theta(tf, encoder_theta_train, tf.shape(theta_seq))
    fused = tf.concat(
        [
            tf.convert_to_tensor(x_y_revin, dtype=tf.float32),
            tf.convert_to_tensor(x_num_scaled, dtype=tf.float32),
            theta_forecaster,
        ],
        axis=-1,
    )
    hidden = model.lstm(fused, training=False)
    y_pred = model.output_layer(hidden)
    return {
        "prediction": y_pred,
        "encoder_theta": theta_seq,
        "forecaster_theta": theta_forecaster,
        "dir_logits": dir_logits,
        "vol_logits": vol_logits,
        "amp_logit": amp_logit,
    }

