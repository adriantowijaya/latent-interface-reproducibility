from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "reference_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from tarela_py2.config import PY2Config
from tarela_py2.model_tf import build_tarela_model


COUNTRY_DIRS = {
    "India": "india",
    "Indonesia": "indonesia",
    "Italy": "italy",
    "Japan": "japan",
    "South Korea": "south_korea",
    "USA": "united_states",
}


@dataclass(frozen=True)
class FrozenReceiver:
    model: Any
    config: PY2Config
    checkpoint_path: Path
    checkpoint_sha256_before: str
    country: str
    country_dir: str
    window: int
    seed: int
    trainable_variables: int


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def weights_sha256(model: Any) -> str:
    h = hashlib.sha256()
    for arr in model.get_weights():
        a = np.asarray(arr)
        h.update(str(a.shape).encode("ascii"))
        h.update(str(a.dtype).encode("ascii"))
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def tp1d_checkpoint_path(tp1d_root: str | Path, country: str, window: int, seed: int) -> Path:
    cdir = COUNTRY_DIRS[country]
    return (
        Path(tp1d_root)
        / "outputs"
        / "tp1d_training"
        / f"seed_{int(seed)}"
        / cdir
        / f"w{int(window):02d}"
        / "checkpoint.weights.h5"
    )


def tp2e_checkpoint_path(training_root: str | Path, variant: str, country: str, seed: int) -> Path:
    cdir = COUNTRY_DIRS[country]
    return (
        Path(training_root)
        / variant
        / f"seed_{int(seed)}"
        / cdir
        / "w05"
        / "checkpoint.weights.h5"
    )


def _config_from_run_config(run_config_path: Path, seed: int, reference_name: str) -> PY2Config:
    if not run_config_path.exists():
        return replace(PY2Config(), random_seed=int(seed), reference_name=reference_name)
    data = json.loads(run_config_path.read_text())
    allowed = {f.name for f in PY2Config.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in data.items() if k in allowed}
    kwargs["random_seed"] = int(seed)
    kwargs["reference_name"] = reference_name
    return PY2Config(**kwargs)


def load_weights_model(tf: Any, checkpoint_path: str | Path, cfg: PY2Config):
    model = build_tarela_model(cfg)
    _ = model(
        (
            tf.zeros([1, cfg.lookback, cfg.n_y_features], dtype=tf.float32),
            tf.zeros([1, cfg.lookback, cfg.n_num_features], dtype=tf.float32),
            tf.zeros([1, cfg.lookback, cfg.n_doc_features], dtype=tf.float32),
        ),
        training=False,
    )
    model.load_weights(str(checkpoint_path))
    return model


def load_frozen_tp1d_receiver(tf: Any, tp1d_root: str | Path, country: str, window: int, seed: int) -> FrozenReceiver:
    checkpoint = tp1d_checkpoint_path(tp1d_root, country, window, seed)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    before = sha256_file(checkpoint)
    run_config = checkpoint.parent / "run_config.json"
    cfg = _config_from_run_config(run_config, seed, "TARELA-LSTM-v1.0-C1-TP2F-frozen-receiver")
    model = load_weights_model(tf, checkpoint, cfg)
    model.trainable = False
    for var in model.variables:
        var._trainable = False
    after = sha256_file(checkpoint)
    if before != after:
        raise RuntimeError("Receiver checkpoint changed during load")
    return FrozenReceiver(
        model=model,
        config=cfg,
        checkpoint_path=checkpoint,
        checkpoint_sha256_before=before,
        country=country,
        country_dir=COUNTRY_DIRS[country],
        window=int(window),
        seed=int(seed),
        trainable_variables=len(model.trainable_variables),
    )

