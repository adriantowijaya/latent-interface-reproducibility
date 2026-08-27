# Environment Authority Audit

Phase: TP-2M.5E.2-R1

SCIENTIFIC_TREE_COMMIT = 11eedf8960b841889805615313fa34a866d47de7

TP2M5A_PROTOCOL_SHA256 = 02f237708fc67b29f6f4e2ba98f2ef5af248d48b979e000a2091a742ef3fa0df

## Authority Classes

CANONICAL_EXPERIMENT_SPECIFICATION:
Python 3.9.16; TensorFlow 2.13.0; NumPy 1.24.3; Pandas 2.0.3; h5py 3.9.0.

ACTUAL_GRU_TRAINING_ENVIRONMENT_EVIDENCE:
The TP2M5B/TP2M5B_R1 checkpoint manifests preserve protocol SHA, GRU configuration hashes, checkpoint hashes, dataset hashes, and frozen-weight status. They do not preserve an exact Python executable path or full package-version lock.

ACTUAL_TRANSFORMER_TRAINING_ENVIRONMENT_EVIDENCE:
The TP2M5C checkpoint manifests preserve protocol SHA, Transformer configuration hashes, checkpoint hashes, dataset hashes, and frozen-weight status. They do not preserve an exact Python executable path or full package-version lock.

QUICK_REPRODUCTION_VERIFICATION_ENVIRONMENT:
Python 3.9.16; TensorFlow 2.10.0; NumPy 1.25.0; Pandas 2.0.3; h5py 3.9.0.

## Local Conda Inspection

Pre-existing environment `knimeenv_python_39_20260428` matched the canonical specification.

Pre-existing environment `knimeenv_python_39` matched the TP2M5E2 verification environment.

No new environment was created.

## Synthetic Cross-Check

The canonical environment `knimeenv_python_39_20260428` was used for synthetic-only model construction and parameter-count validation. No real data, `model.fit()`, optimizer step, checkpoint inference, or weight modification was performed.

Canonical-environment synthetic smoke status: PASS.

Expected counts were recovered: Reference TARELA-LSTM = 12282; GRU receiver substitution = 9432; Transformer receiver substitution = 11272. Geometry check recovered K=5, L=7, H=1, fused input width=9.

## Classification

EXACT_TRAINING_PACKAGE_LOCK = NOT_FULLY_RECOVERABLE_FROM_PERSISTED_METADATA

ENVIRONMENT_DISCREPANCY_CLASSIFICATION = VERIFICATION_ENVIRONMENT_DIFFERENCE

The quick reproduction environment is not labeled as the training environment.
