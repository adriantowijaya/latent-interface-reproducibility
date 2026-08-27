from __future__ import annotations

from .config import PY2Config


def theta_entropy_norm(theta_seq, tf):
    eps = 1e-8
    k = tf.cast(tf.shape(theta_seq)[-1], theta_seq.dtype)
    theta_clip = tf.clip_by_value(theta_seq, eps, 1.0)
    entropy = -tf.reduce_sum(theta_clip * tf.math.log(theta_clip), axis=-1)
    return entropy / tf.math.log(k)


def weighted_sparse_ce(labels, logits, class_weights, use_weighted_aux_loss: bool, tf):
    labels = tf.cast(labels, tf.int32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    if use_weighted_aux_loss:
        weights = tf.gather(tf.constant(class_weights, dtype=tf.float32), labels)
        ce = ce * weights
    return tf.reduce_mean(ce)


def weighted_bce_with_logits(labels, logits, pos_weight: float, use_weighted_aux_loss: bool, tf):
    labels = tf.cast(labels, tf.float32)
    if use_weighted_aux_loss:
        loss = tf.nn.weighted_cross_entropy_with_logits(
            labels=labels, logits=logits, pos_weight=tf.cast(pos_weight, tf.float32)
        )
    else:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss)


def joint_loss_transition_aware(y_true_revin, y_pred_revin, theta_seq,
                                dir_logits, vol_logits, amp_logit,
                                dir_label, vol_label, amp_label,
                                dir_class_weights, vol_class_weights, amp_pos_weight,
                                cfg: PY2Config, tf):
    mse = tf.reduce_mean(tf.square(y_true_revin - y_pred_revin))
    h_norm = theta_entropy_norm(theta_seq, tf)
    entropy_target_loss = tf.reduce_mean(tf.square(h_norm - cfg.entropy_target))
    dir_loss = weighted_sparse_ce(
        dir_label, dir_logits, dir_class_weights, cfg.use_weighted_aux_loss, tf
    )
    vol_loss = weighted_sparse_ce(
        vol_label, vol_logits, vol_class_weights, cfg.use_weighted_aux_loss, tf
    )
    amp_loss = weighted_bce_with_logits(
        amp_label, amp_logit, amp_pos_weight, cfg.use_weighted_aux_loss, tf
    )
    total = (
        mse
        + cfg.lambda_entropy * entropy_target_loss
        + cfg.lambda_dir * dir_loss
        + cfg.lambda_vol * vol_loss
        + cfg.lambda_amp * amp_loss
    )
    return total, mse, entropy_target_loss, dir_loss, vol_loss, amp_loss
