from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import PY2Config


@dataclass
class PreparedPartition:
    X_y: np.ndarray
    X_num: np.ndarray
    X_doc: np.ndarray
    y_input: np.ndarray
    y_actual: np.ndarray
    dir_label: np.ndarray
    vol_label: np.ndarray
    amp_label: np.ndarray
    dates: pd.Series

    mu: np.ndarray
    sigma: np.ndarray
    X_y_revin: np.ndarray
    y_revin: np.ndarray
    X_num_scaled: np.ndarray
    X_doc_scaled: np.ndarray


@dataclass
class PreparedWindow:
    train: PreparedPartition
    validation: PreparedPartition
    test: PreparedPartition
    x_mean: np.ndarray
    x_std: np.ndarray
    doc_mean: np.ndarray
    doc_std: np.ndarray
    dir_class_weights: np.ndarray
    vol_class_weights: np.ndarray
    amp_pos_weight: float
    config: PY2Config


def _parse_json_tensor(series: pd.Series) -> np.ndarray:
    return np.asarray(series.apply(json.loads).tolist(), dtype=np.float32)


def parse_sequence_frame(df: pd.DataFrame, partition_name: str, cfg: PY2Config) -> Dict[str, object]:
    required = [
        "X_y_json", "X_num_json", "X_doc_json", "y", "target_date",
        "future_dir_label", "future_vol_label", "future_amp_label",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{partition_name}: missing columns {missing}")

    X_y = _parse_json_tensor(df["X_y_json"])
    X_num = _parse_json_tensor(df["X_num_json"])
    X_doc = _parse_json_tensor(df["X_doc_json"])
    y_input = pd.to_numeric(df["y"], errors="coerce").fillna(0).to_numpy(np.float32).reshape(-1, 1)
    if "New_cases_actual" in df.columns:
        y_actual = pd.to_numeric(df["New_cases_actual"], errors="coerce").fillna(0).to_numpy(np.float32).reshape(-1, 1)
    else:
        y_actual = y_input.copy()
    dir_label = pd.to_numeric(df["future_dir_label"], errors="coerce").fillna(1).to_numpy(np.int32)
    vol_label = pd.to_numeric(df["future_vol_label"], errors="coerce").fillna(1).to_numpy(np.int32)
    amp_label = pd.to_numeric(df["future_amp_label"], errors="coerce").fillna(0).to_numpy(np.float32).reshape(-1, 1)
    dates = pd.to_datetime(df["target_date"])

    if X_y.ndim != 3 or X_num.ndim != 3 or X_doc.ndim != 3:
        raise ValueError(f"{partition_name}: sequence tensors must be rank-3.")
    if X_y.shape[1:] != (cfg.lookback, cfg.n_y_features):
        raise ValueError(f"{partition_name}: X_y shape mismatch {X_y.shape}.")
    if X_num.shape[1:] != (cfg.lookback, cfg.n_num_features):
        raise ValueError(f"{partition_name}: X_num shape mismatch {X_num.shape}.")
    if X_doc.shape[1:] != (cfg.lookback, cfg.n_doc_features):
        raise ValueError(f"{partition_name}: X_doc shape mismatch {X_doc.shape}.")
    for name, arr in [("X_y", X_y), ("X_num", X_num), ("X_doc", X_doc), ("y", y_input)]:
        if not np.isfinite(arr).all():
            raise ValueError(f"{partition_name}: {name} contains non-finite values.")

    return {
        "X_y": X_y,
        "X_num": X_num,
        "X_doc": X_doc,
        "y_input": y_input,
        "y_actual": y_actual,
        "dir_label": dir_label,
        "vol_label": vol_label,
        "amp_label": amp_label,
        "dates": dates,
    }


def compute_revin_stats(X_y: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    # Exact legacy semantics: population std over each 7-point target-history sequence.
    mu = X_y.mean(axis=1, keepdims=True)
    sigma = X_y.std(axis=1, keepdims=True)
    sigma = np.where(sigma < eps, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def revin_normalize_Xy(X_y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((X_y - mu) / sigma).astype(np.float32)


def revin_normalize_y(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((y - mu[:, 0, :]) / sigma[:, 0, :]).astype(np.float32)


def revin_denormalize_y(y_norm: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (y_norm * sigma[:, 0, :] + mu[:, 0, :]).astype(np.float32)


def fit_feature_scaler(X: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    # Exact legacy semantics: flatten overlapping train-inner sequences across instances/timesteps.
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def compute_class_weights(labels: np.ndarray, n_classes: int, max_class_weight: float) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1).astype(int)
    total = len(labels)
    weights = []
    for c in range(n_classes):
        count = np.sum(labels == c)
        w = 1.0 if count < 1 else total / (n_classes * count)
        if not np.isfinite(w):
            w = 1.0
        weights.append(float(np.clip(w, 1.0 / max_class_weight, max_class_weight)))
    return np.asarray(weights, dtype=np.float32)


def calc_pos_weight_binary(labels: np.ndarray, max_class_weight: float) -> float:
    y = np.asarray(labels).reshape(-1).astype(float)
    pos = np.sum(y > 0.5)
    neg = np.sum(y <= 0.5)
    if pos < 1:
        return 1.0
    w = neg / max(pos, 1.0)
    if not np.isfinite(w):
        return 1.0
    return float(np.clip(w, 1.0, max_class_weight))


def _prepare_partition(parsed: Dict[str, object], x_mean: np.ndarray, x_std: np.ndarray,
                       doc_mean: np.ndarray, doc_std: np.ndarray, cfg: PY2Config) -> PreparedPartition:
    X_y = parsed["X_y"]
    X_num = parsed["X_num"]
    X_doc = parsed["X_doc"]
    mu, sigma = compute_revin_stats(X_y, cfg.revin_eps)
    X_y_revin = revin_normalize_Xy(X_y, mu, sigma)
    y_revin = revin_normalize_y(parsed["y_input"], mu, sigma)
    X_num_scaled = apply_feature_scaler(X_num, x_mean, x_std)
    X_doc_scaled = apply_feature_scaler(X_doc, doc_mean, doc_std) if cfg.scale_doc else X_doc.astype(np.float32)
    return PreparedPartition(
        X_y=X_y, X_num=X_num, X_doc=X_doc,
        y_input=parsed["y_input"], y_actual=parsed["y_actual"],
        dir_label=parsed["dir_label"], vol_label=parsed["vol_label"], amp_label=parsed["amp_label"],
        dates=parsed["dates"], mu=mu, sigma=sigma,
        X_y_revin=X_y_revin, y_revin=y_revin,
        X_num_scaled=X_num_scaled, X_doc_scaled=X_doc_scaled,
    )


def prepare_window(train_df: pd.DataFrame, validation_df: pd.DataFrame, test_df: pd.DataFrame,
                   cfg: Optional[PY2Config] = None) -> PreparedWindow:
    cfg = cfg or PY2Config()
    cfg.validate()
    tr = parse_sequence_frame(train_df, "train_inner", cfg)
    va = parse_sequence_frame(validation_df, "in_sample", cfg)
    te = parse_sequence_frame(test_df, "out_sample", cfg)
    if len(va["X_y"]) != cfg.expected_validation_targets:
        raise ValueError("Validation target count mismatch.")
    if len(te["X_y"]) != cfg.expected_test_targets:
        raise ValueError("Test target count mismatch.")

    x_mean, x_std = fit_feature_scaler(tr["X_num"], cfg.feature_scale_eps)
    if cfg.scale_doc:
        doc_mean, doc_std = fit_feature_scaler(tr["X_doc"], cfg.feature_scale_eps)
    else:
        doc_mean = np.zeros(cfg.n_doc_features, dtype=np.float32)
        doc_std = np.ones(cfg.n_doc_features, dtype=np.float32)

    dir_w = compute_class_weights(tr["dir_label"], 3, cfg.max_class_weight)
    vol_w = compute_class_weights(tr["vol_label"], 3, cfg.max_class_weight)
    amp_w = calc_pos_weight_binary(tr["amp_label"], cfg.max_class_weight)

    return PreparedWindow(
        train=_prepare_partition(tr, x_mean, x_std, doc_mean, doc_std, cfg),
        validation=_prepare_partition(va, x_mean, x_std, doc_mean, doc_std, cfg),
        test=_prepare_partition(te, x_mean, x_std, doc_mean, doc_std, cfg),
        x_mean=x_mean, x_std=x_std, doc_mean=doc_mean, doc_std=doc_std,
        dir_class_weights=dir_w, vol_class_weights=vol_w, amp_pos_weight=amp_w, config=cfg,
    )


def prepared_window_audit(p: PreparedWindow) -> dict:
    return {
        "train_n": int(len(p.train.y_input)),
        "validation_n": int(len(p.validation.y_input)),
        "test_n": int(len(p.test.y_input)),
        "X_y_shape_train": list(p.train.X_y_revin.shape),
        "X_num_shape_train": list(p.train.X_num_scaled.shape),
        "X_doc_shape_train": list(p.train.X_doc_scaled.shape),
        "revin_train_mu_mean": float(p.train.mu.mean()),
        "revin_train_sigma_mean": float(p.train.sigma.mean()),
        "x_mean": p.x_mean.astype(float).tolist(),
        "x_std": p.x_std.astype(float).tolist(),
        "doc_mean": p.doc_mean.astype(float).tolist(),
        "doc_std": p.doc_std.astype(float).tolist(),
        "dir_class_weights": p.dir_class_weights.astype(float).tolist(),
        "vol_class_weights": p.vol_class_weights.astype(float).tolist(),
        "amp_pos_weight": float(p.amp_pos_weight),
        "all_finite": bool(all(np.isfinite(a).all() for a in [
            p.train.X_y_revin, p.train.y_revin, p.train.X_num_scaled, p.train.X_doc_scaled,
            p.validation.X_y_revin, p.validation.y_revin, p.validation.X_num_scaled, p.validation.X_doc_scaled,
            p.test.X_y_revin, p.test.y_revin, p.test.X_num_scaled, p.test.X_doc_scaled,
        ])),
    }
