from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd

from .config import PY2Config
from .losses_tf import joint_loss_transition_aware
from .math_reference import expected_trainable_parameter_count
from .model_tf import build_optimizer, build_tarela_model, configure_tensorflow_runtime
from .prepare import PreparedPartition, PreparedWindow, revin_denormalize_y


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _check_numeric(tf, name, value):
    tf.debugging.check_numerics(value, f"Non-finite {name}")


def _calc_theta_diagnostics(theta_last: np.ndarray):
    eps = 1e-8
    k = theta_last.shape[1]
    theta_clip = np.clip(theta_last, eps, 1.0)
    entropy = -np.sum(theta_clip * np.log(theta_clip), axis=1)
    entropy_norm = entropy / np.log(k)
    l1 = np.sum(np.abs(theta_last), axis=1)
    l2 = np.sqrt(np.sum(theta_last ** 2, axis=1)) + eps
    hoyer = (np.sqrt(k) - (l1 / l2)) / (np.sqrt(k) - 1.0)
    hoyer = np.clip(hoyer, 0.0, 1.0)
    active_topics = np.sum(theta_last > 1e-4, axis=1)
    return entropy_norm, hoyer, active_topics


def _balanced_accuracy_multiclass(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    recalls = []
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append(float(np.mean(y_pred[mask] == c)))
    return np.nan if not recalls else float(np.mean(recalls))


def _binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    y_pred = (y_prob >= threshold).astype(int)
    if len(y_true) == 0:
        return {"acc": np.nan, "balanced_acc": np.nan, "pos_rate": np.nan, "pred_pos_rate": np.nan}
    acc = float(np.mean(y_true == y_pred))
    pos_mask, neg_mask = y_true == 1, y_true == 0
    tpr = float(np.mean(y_pred[pos_mask] == 1)) if pos_mask.sum() else np.nan
    tnr = float(np.mean(y_pred[neg_mask] == 0)) if neg_mask.sum() else np.nan
    bacc = 0.5 * (tpr + tnr) if np.isfinite(tpr) and np.isfinite(tnr) else np.nan
    return {"acc": acc, "balanced_acc": bacc, "pos_rate": float(np.mean(y_true)), "pred_pos_rate": float(np.mean(y_pred))}


def _smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def _metric_set(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    smape_value = _smape(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-8 else np.nan
    return mae, mse, rmse, smape_value, r2


def train_reference(prepared: PreparedWindow, output_dir: Union[str, Path]) -> Dict[str, object]:
    """Execute one controlled neural run using latest KNIME-script semantics."""
    cfg = prepared.config
    tf = configure_tensorflow_runtime(cfg)
    model = build_tarela_model(cfg)
    optimizer = build_optimizer(cfg, tf)

    # Build variables before parameter-count assertion.
    _ = model(
        (
            tf.convert_to_tensor(prepared.train.X_y_revin[:1]),
            tf.convert_to_tensor(prepared.train.X_num_scaled[:1]),
            tf.convert_to_tensor(prepared.train.X_doc_scaled[:1]),
        ),
        training=False,
    )
    trainable_parameter_count = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    expected = expected_trainable_parameter_count()
    if trainable_parameter_count != expected:
        raise AssertionError(f"Trainable parameter parity failed: {trainable_parameter_count} != {expected}")

    tr = prepared.train
    train_ds = tf.data.Dataset.from_tensor_slices(
        (
            (tr.X_y_revin, tr.X_num_scaled, tr.X_doc_scaled),
            tr.y_revin,
            tr.dir_label.astype(np.int32),
            tr.vol_label.astype(np.int32),
            tr.amp_label.astype(np.float32),
        )
    )
    options = tf.data.Options()
    options.experimental_deterministic = True
    train_ds = train_ds.with_options(options).batch(cfg.batch_size, drop_remainder=False)

    logs = []
    best_val_mse = np.inf
    best_weights = None
    best_epoch = None
    wait = 0
    stopped_epoch = None
    early_stopped = False

    va = prepared.validation
    for epoch in range(cfg.epochs):
        acc = dict(loss=0.0, mse=0.0, entropy=0.0, dir=0.0, vol=0.0, amp=0.0)
        n_batches = 0
        for (xb_y, xb_num, xb_doc), yb, db, vb, ab in train_ds:
            with tf.GradientTape() as tape:
                pred, theta, dlog, vlog, alog = model((xb_y, xb_num, xb_doc), training=True)
                vals = joint_loss_transition_aware(
                    yb, pred, theta, dlog, vlog, alog, db, vb, ab,
                    prepared.dir_class_weights, prepared.vol_class_weights, prepared.amp_pos_weight,
                    cfg, tf,
                )
                for name, value in zip(
                    ["total training loss", "forecasting MSE", "entropy-target loss", "direction loss", "volatility loss", "amplitude loss"],
                    vals,
                ):
                    _check_numeric(tf, name, value)
            grads = tape.gradient(vals[0], model.trainable_variables)
            pairs = []
            for gradient, variable in zip(grads, model.trainable_variables):
                # Auxiliary-head gradients are intentionally None in MSE-only.
                if gradient is None:
                    continue
                checked = tf.debugging.check_numerics(gradient, f"Non-finite gradient in {variable.name}")
                pairs.append((checked, variable))
            if not pairs:
                raise RuntimeError("No valid gradients were produced during training.")
            optimizer.apply_gradients(pairs)
            for key, value in zip(acc.keys(), vals):
                acc[key] += float(value.numpy())
            n_batches += 1

        pred_v, theta_v, dlog_v, vlog_v, alog_v = model(
            (va.X_y_revin, va.X_num_scaled, va.X_doc_scaled), training=False
        )
        vvals = joint_loss_transition_aware(
            va.y_revin, pred_v, theta_v, dlog_v, vlog_v, alog_v,
            va.dir_label, va.vol_label, va.amp_label,
            prepared.dir_class_weights, prepared.vol_class_weights, prepared.amp_pos_weight,
            cfg, tf,
        )
        for name, value in zip(
            ["validation total loss", "validation forecasting MSE", "validation entropy-target loss", "validation direction loss", "validation volatility loss", "validation amplitude loss"],
            vvals,
        ):
            _check_numeric(tf, name, value)
        current_val_mse = float(vvals[1].numpy())
        improved = current_val_mse < (best_val_mse - cfg.min_delta)
        if improved:
            best_val_mse = current_val_mse
            best_weights = model.get_weights()
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
        logs.append({
            "epoch": epoch + 1,
            "train_loss": acc["loss"] / max(n_batches, 1),
            "train_mse": acc["mse"] / max(n_batches, 1),
            "train_entropy_target_loss": acc["entropy"] / max(n_batches, 1),
            "train_dir_loss": acc["dir"] / max(n_batches, 1),
            "train_vol_loss": acc["vol"] / max(n_batches, 1),
            "train_amp_loss": acc["amp"] / max(n_batches, 1),
            "val_loss": float(vvals[0].numpy()),
            "val_mse": current_val_mse,
            "val_entropy_target_loss": float(vvals[2].numpy()),
            "val_dir_loss": float(vvals[3].numpy()),
            "val_vol_loss": float(vvals[4].numpy()),
            "val_amp_loss": float(vvals[5].numpy()),
            "best_val_mse_so_far": best_val_mse,
            "early_stop_wait": wait,
        })
        if cfg.early_stopping and wait >= cfg.patience:
            stopped_epoch = epoch + 1
            early_stopped = True
            break

    epochs_run = len(logs)
    if stopped_epoch is None:
        stopped_epoch = epochs_run
    if best_weights is None or best_epoch is None:
        raise RuntimeError("No valid validation checkpoint was stored.")
    model.set_weights(best_weights)

    def predict_partition(p: PreparedPartition, name: str) -> pd.DataFrame:
        pred_r, theta_seq, dir_logits, vol_logits, amp_logit = model(
            (p.X_y_revin, p.X_num_scaled, p.X_doc_scaled), training=False
        )
        pred_r_np = pred_r.numpy().astype(np.float32)
        pred_raw = revin_denormalize_y(pred_r_np, p.mu, p.sigma)
        pred_clip = np.maximum(pred_raw, 0.0) if cfg.use_clipping else pred_raw.copy()
        theta_np = theta_seq.numpy().astype(np.float32)
        theta_last = theta_np[:, -1, :]
        entropy_norm, hoyer, active_topics = _calc_theta_diagnostics(theta_last)
        dir_prob = tf.nn.softmax(dir_logits, axis=-1).numpy()
        vol_prob = tf.nn.softmax(vol_logits, axis=-1).numpy()
        amp_prob = 1.0 / (1.0 + np.exp(-amp_logit.numpy().reshape(-1)))
        dir_pred = np.argmax(dir_prob, axis=1)
        vol_pred = np.argmax(vol_prob, axis=1)
        amp_pred = (amp_prob >= 0.5).astype(int)
        raw = pred_raw.reshape(-1)
        clip = pred_clip.reshape(-1)
        out = pd.DataFrame({
            "partition": name,
            "target_date": pd.to_datetime(p.dates),
            "y_model_target": p.y_input.reshape(-1),
            "prediction_after_inverse_revin": raw,
            "actual": p.y_actual.reshape(-1),
            "prediction_raw": raw,
            "prediction": clip,
            "was_clipped": (raw < 0).astype(int),
            "clipping_amount": np.maximum(-raw, 0.0),
            "revin_mu": p.mu[:, 0, 0],
            "revin_sigma": p.sigma[:, 0, 0],
            "prediction_revin_scale": pred_r_np.reshape(-1),
            "future_dir_label": p.dir_label.reshape(-1),
            "future_dir_pred": dir_pred,
            "future_vol_label": p.vol_label.reshape(-1),
            "future_vol_pred": vol_pred,
            "future_amp_label": p.amp_label.reshape(-1),
            "future_amp_prob": amp_prob,
            "future_amp_pred": amp_pred,
            "theta_entropy_norm": entropy_norm,
            "theta_hoyer_sparsity": hoyer,
            "theta_active_topics": active_topics,
            "selected_epoch": int(best_epoch),
            "stopped_epoch": int(stopped_epoch),
            "epochs_run": int(epochs_run),
            "early_stopped": bool(early_stopped),
            "restore_best_weights": True,
            "random_seed": int(cfg.random_seed),
        })
        for k in range(dir_prob.shape[1]): out[f"future_dir_prob_{k}"] = dir_prob[:, k]
        for k in range(vol_prob.shape[1]): out[f"future_vol_prob_{k}"] = vol_prob[:, k]
        for k in range(theta_last.shape[1]): out[f"theta_{k}"] = theta_last[:, k]
        return out

    pred_df = pd.concat([
        predict_partition(prepared.train, "train_inner"),
        predict_partition(prepared.validation, "in_sample"),
        predict_partition(prepared.test, "out_sample"),
    ], ignore_index=True)

    metrics_rows = []
    for name in ["train_inner", "in_sample", "out_sample"]:
        d = pred_df[pred_df["partition"] == name]
        raw_metrics = _metric_set(d["actual"].values, d["prediction_raw"].values)
        clip_metrics = _metric_set(d["actual"].values, d["prediction"].values)
        amp_m = _binary_metrics(d["future_amp_label"].values, d["future_amp_prob"].values)
        metrics_rows.append({
            "partition": name, "n": len(d),
            "MAE": clip_metrics[0], "MSE": clip_metrics[1], "RMSE": clip_metrics[2], "SMAPE": clip_metrics[3], "R2": clip_metrics[4],
            "MAE_raw": raw_metrics[0], "MSE_raw": raw_metrics[1], "RMSE_raw": raw_metrics[2], "SMAPE_raw": raw_metrics[3], "R2_raw": raw_metrics[4],
            "n_clipped": int(d["was_clipped"].sum()), "clip_rate": float(d["was_clipped"].mean()),
            "theta_entropy_norm_mean": float(d["theta_entropy_norm"].mean()),
            "theta_hoyer_sparsity_mean": float(d["theta_hoyer_sparsity"].mean()),
            "theta_active_topics_mean": float(d["theta_active_topics"].mean()),
            "dir_accuracy": float(np.mean(d["future_dir_label"] == d["future_dir_pred"])),
            "dir_balanced_accuracy": _balanced_accuracy_multiclass(d["future_dir_label"], d["future_dir_pred"], 3),
            "vol_accuracy": float(np.mean(d["future_vol_label"] == d["future_vol_pred"])),
            "vol_balanced_accuracy": _balanced_accuracy_multiclass(d["future_vol_label"], d["future_vol_pred"], 3),
            "amp_label_positive_rate": amp_m["pos_rate"], "amp_pred_positive_rate": amp_m["pred_pos_rate"],
            "amp_accuracy": amp_m["acc"], "amp_balanced_accuracy": amp_m["balanced_acc"],
        })
    metrics_df = pd.DataFrame(metrics_rows)
    log_df = pd.DataFrame(logs)
    log_df["selected_epoch"] = int(best_epoch)
    log_df["is_selected_epoch"] = log_df["epoch"].astype(int) == int(best_epoch)
    log_df["stopped_epoch"] = int(stopped_epoch)
    log_df["epochs_run"] = int(epochs_run)
    log_df["early_stopped"] = bool(early_stopped)
    log_df["restore_best_weights"] = True

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    log_df.to_csv(out_dir / "training_history.csv", index=False)
    model.save_weights(out_dir / "checkpoint.weights.h5")

    cfg_out = cfg.to_dict()
    cfg_out.update({
        "trainable_parameter_count": trainable_parameter_count,
        "selected_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "epochs_run": int(epochs_run),
        "best_validation_forecasting_mse": float(best_val_mse),
        "dir_class_weights": prepared.dir_class_weights.astype(float).tolist(),
        "vol_class_weights": prepared.vol_class_weights.astype(float).tolist(),
        "amp_pos_weight": float(prepared.amp_pos_weight),
        "x_mean": prepared.x_mean.astype(float).tolist(),
        "x_std": prepared.x_std.astype(float).tolist(),
        "doc_mean": prepared.doc_mean.astype(float).tolist(),
        "doc_std": prepared.doc_std.astype(float).tolist(),
        "tensorflow_version": tf.__version__,
    })
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg_out, f, indent=2)
    manifest = {
        "status": "PASS",
        "objective_variant": cfg.objective_variant(),
        "trainable_parameter_count": trainable_parameter_count,
        "expected_parameter_count": expected,
        "selected_epoch": int(best_epoch),
        "checkpoint_sha256": _sha256(out_dir / "checkpoint.weights.h5"),
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
