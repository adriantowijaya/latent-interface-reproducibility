# Environment Provenance

Phase: TP-2M.5E.2-R1

SCIENTIFIC_TREE_COMMIT = 11eedf8960b841889805615313fa34a866d47de7

TP2M5A_PROTOCOL_SHA256 = 02f237708fc67b29f6f4e2ba98f2ef5af248d48b979e000a2091a742ef3fa0df

## CANONICAL_EXPERIMENT_SPECIFICATION

The repository canonical reproduction specification is:

- Python 3.9.16
- TensorFlow 2.13.0
- NumPy 1.24.3
- Pandas 2.0.3
- h5py 3.9.0

This specification is represented by `requirements.txt`, `scripts/verify_environment.py`, and `environment-canonical.yml`. A pre-existing local conda environment named `knimeenv_python_39_20260428` was inspected during TP-2M.5E.2-R1 and matched these versions:

`C:\Users\adria\anaconda3\envs\knimeenv_python_39_20260428\python.exe`

No new conda environment was created during TP-2M.5E.2-R1.

## ACTUAL_TRAINING_ENVIRONMENT_EVIDENCE

Persisted GRU and Transformer evidence includes checkpoint manifests, receiver configuration hashes, dataset hashes, TP2M5A protocol hashes, and frozen-weight status:

- `manifests/checkpoints/TP2M5B_GRU_RECEIVER_BANK_MANIFEST.csv`
- `manifests/checkpoints/TP2M5C_TRANSFORMER_RECEIVER_BANK_MANIFEST.csv`
- `manifests/checkpoints/TP2M5B_GRU_RECEIVER_BANK_HASHES.txt`
- `manifests/checkpoints/TP2M5C_TRANSFORMER_RECEIVER_BANK_HASHES.txt`

For GRU training, the persisted manifests evidence the TP2M5A protocol SHA, GRU configuration hashes, checkpoint hashes, dataset hashes, and `weights_frozen=TRUE`. They do not persist the Python executable path or exact Python/TensorFlow/NumPy/Pandas/h5py package versions used at training time.

For Transformer training, the persisted manifests evidence the TP2M5A protocol SHA, Transformer configuration hashes, checkpoint hashes, dataset hashes, and `weights_frozen=TRUE`. They do not persist the Python executable path or exact Python/TensorFlow/NumPy/Pandas/h5py package versions used at training time.

EXACT_TRAINING_PACKAGE_LOCK = NOT_FULLY_RECOVERABLE_FROM_PERSISTED_METADATA

The canonical experiment specification should therefore be used as the recommended scientific reproduction environment, but it is not relabeled here as a proven exact historical training package lock.

## QUICK_REPRODUCTION_VERIFICATION_ENVIRONMENT

The TP2M5E2 quick reproduction environment is distinct from the canonical specification:

- Python 3.9.16
- TensorFlow 2.10.0
- NumPy 1.25.0
- Pandas 2.0.3
- h5py 3.9.0

During TP-2M.5E.2-R1, this environment was inspected as:

`C:\Users\adria\anaconda3\envs\knimeenv_python_39\python.exe`

This environment is recorded in `environment-lock.txt` and `environment-verification.txt`. It is suitable for the zero-training quick reproduction checks performed here, but it is not the declared training environment.

## UNRESOLVED_VERSION_IDENTITY

The repository preserves a clean boundary:

- canonical specification: exact recommended scientific reproduction pins;
- actual training evidence: protocol/config/checkpoint/dataset hashes, without a full package lock;
- quick reproduction: exact local verification package versions.

The TensorFlow and NumPy mismatch observed in quick reproduction is classified as `VERIFICATION_ENVIRONMENT_DIFFERENCE`, not as scientific protocol drift.
