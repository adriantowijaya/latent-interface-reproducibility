from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from models.tarela_py2.config import PY2Config
from models.tarela_py2.model_tf import build_tarela_model, require_tensorflow
from models.controlled_receivers import build_gru_receiver_substitution, build_transformer_receiver_substitution

EXPECTED = {"Reference TARELA-LSTM": 12282, "GRU receiver substitution": 9432, "Transformer receiver substitution": 11272}
L, H, K, FUSED_WIDTH = 7, 1, 5, 9

def count(model):
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))

def main():
    tf, layers = require_tensorflow()
    tf.keras.backend.clear_session()
    cfg = PY2Config()
    if (cfg.lookback, cfg.topic_dim) != (L, K):
        raise AssertionError("Protocol geometry mismatch")
    x_y = tf.zeros((2, L, 1), dtype=tf.float32)
    x_num = tf.zeros((2, L, 3), dtype=tf.float32)
    x_doc = tf.zeros((2, L, cfg.n_doc_features), dtype=tf.float32)
    if x_y.shape[-1] + x_num.shape[-1] + K != FUSED_WIDTH:
        raise AssertionError("Fused input width mismatch")
    models = {
        "Reference TARELA-LSTM": build_tarela_model(cfg),
        "GRU receiver substitution": build_gru_receiver_substitution(cfg, tf, layers),
        "Transformer receiver substitution": build_transformer_receiver_substitution(cfg, tf, layers),
    }
    status = "PASS"
    for name, model in models.items():
        _ = model((x_y, x_num, x_doc), training=False)
        got = count(model)
        ok = got == EXPECTED[name]
        print(f"{name}: parameters={got} expected={EXPECTED[name]} {'PASS' if ok else 'FAIL'}")
        if not ok:
            status = "FAIL"
    print(f"fused_input_width={FUSED_WIDTH} K={K} L={L} H={H}")
    print(f"STATUS={status}")
    raise SystemExit(0 if status == "PASS" else 1)

if __name__ == "__main__":
    main()
