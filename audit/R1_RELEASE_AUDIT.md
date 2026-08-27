# R1 Release Audit

Phase: TP-2M.5E.2-R1

TRAINING = 0

OPTIMIZER_STEPS = 0

WEIGHTS_MODIFIED = NO

SCIENTIFIC_TREE_COMMIT = 11eedf8960b841889805615313fa34a866d47de7

TP2M5A_PROTOCOL_SHA256 = 02f237708fc67b29f6f4e2ba98f2ef5af248d48b979e000a2091a742ef3fa0df

## Environment

Canonical specification: Python 3.9.16; TensorFlow 2.13.0; NumPy 1.24.3; Pandas 2.0.3; h5py 3.9.0.

Actual GRU training environment evidence: protocol/config/checkpoint/dataset manifests only; exact package lock not persisted.

Actual Transformer training environment evidence: protocol/config/checkpoint/dataset manifests only; exact package lock not persisted.

Verification environment: Python 3.9.16; TensorFlow 2.10.0; NumPy 1.25.0; Pandas 2.0.3; h5py 3.9.0.

Environment discrepancy classification: VERIFICATION_ENVIRONMENT_DIFFERENCE.

EXACT_TRAINING_PACKAGE_LOCK = NOT_FULLY_RECOVERABLE_FROM_PERSISTED_METADATA

Canonical-environment synthetic model cross-check: PASS in `knimeenv_python_39_20260428` with exact parameter counts 12282, 9432, and 11272. No real-data inference or optimizer step was run.

## License

WHO: CC BY 4.0 with WHO terms, attribution required, no WHO endorsement or affiliation implication.

Electricity: UCI ElectricityLoadDiagrams20112014, DOI 10.24432/C58C86, CC BY 4.0 attribution.

Dengue: Taiwan CDC Open Government Data License v1.0; processed county-level daily aggregate only; raw case-level Dengue absent.

Third-party code: NO_UNRESOLVED_VENDORED_THIRD_PARTY_CODE.

## Release Status

License status: PASS.

Security status: PASS.

Quick reproduction status: PASS_WITH_VERIFICATION_ENVIRONMENT_DIFFERENCE.

GitHub remote configured: false.

GitHub pushed: false.

TP-2M.5E.2-R1 = PASS_RELEASE_PROVENANCE_RECONCILED_WITH_ENVIRONMENT_BOUNDARY
