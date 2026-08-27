from __future__ import annotations
import hashlib, json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "configs/multiarchitecture/TP2M5A_MULTIARCH_PROTOCOL_FREEZE.json"
digest = hashlib.sha256(path.read_bytes()).hexdigest()
expected = "02f237708fc67b29f6f4e2ba98f2ef5af248d48b979e000a2091a742ef3fa0df"
freeze = json.loads(path.read_text(encoding="utf-8"))
checks = [
    ("sha256", digest == expected),
    ("L", freeze.get("lookback_L") == 7),
    ("H", freeze.get("horizon_H") == 1),
    ("K", freeze.get("Transformer_architecture", {}).get("d_in") == 9 and "LSTM_REFERENCE" in freeze.get("architecture_ids", [])),
    ("lstm_parameters", freeze.get("LSTM_authority_reference", {}).get("total_trainable_model_parameters") == 12282),
    ("gru_parameters", freeze.get("GRU_architecture", {}).get("total_trainable_model_parameters") == 9432),
    ("transformer_parameters", freeze.get("Transformer_architecture", {}).get("total_trainable_model_parameters") == 11272),
]
status = "PASS"
for name, ok in checks:
    print(f"{name}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        status = "FAIL"
print(f"STATUS={status}")
raise SystemExit(0 if status == "PASS" else 1)
