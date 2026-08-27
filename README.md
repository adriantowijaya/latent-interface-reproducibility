# Neurocomputing Reproducibility Package

This local repository stages the reproducibility materials for "Latent Representation Instability Across Recurrent and Attention-Based Receivers: Functional Equivalence and Receiver Transfer in Neural Time-Series Forecasting".

## Scientific purpose

The package supports zero-training regeneration of manuscript tables and figures from frozen machine-readable results, plus a documented path for full computational reproduction.

## What this repository reproduces

It covers the reference TARELA-LSTM configuration, GRU controlled receiver substitution, Transformer controlled receiver substitution, WHO, Electricity-37 and Dengue-7 structural replications, TP2B latent analysis, architecture-common functional diagnostics, receiver swaps, TP2D LSTM-specific diagnostics, intervention gates, and manuscript numerical summaries.

## What this repository does not claim

It does not publish a GitHub release, include checkpoint weights, train models during quick reproduction, or claim architecture-universal results beyond the staged receiver substitutions.

## Architecture configurations

The alternative GRU and Transformer configurations are controlled receiver substitutions within the same sparse latent-interface system; they are not standalone state-of-the-art forecasting baselines.

## Data strata

Staged data are processed analytical files for WHO COVID-19, Electricity-37, and Dengue-7. Raw dengue case-level records, raw global WHO archives, and the full raw Electricity archive are excluded.

## Quick reproduction

Run `python scripts/verify_environment.py`, `python scripts/verify_data_hashes.py`, `python scripts/verify_protocol.py`, `python scripts/smoke_test_models.py`, `python scripts/reproduce_tables.py`, and `python scripts/reproduce_figures.py`.

## Full reproduction

See `full_reproduction/README.md`. This route retrains receiver banks and reruns structural and functional audits; it is intentionally not executed in this freeze.

## Repository layout

`src/` contains portable code, `configs/` contains frozen protocol contracts, `data/` contains processed analytical data, `results/` contains machine-readable authority results, `manifests/` contains cohort and checkpoint registers, and `audit/` contains release evidence.

## Environment

The canonical target is Python 3.9.16 with TensorFlow 2.13.0, NumPy 1.24.3, Pandas 2.0.3, and h5py 3.9.0. The local validation environment is recorded in `environment-lock.txt`.

## Data licensing

The MIT License applies to original source code and repository documentation authored for this project. It does not relicense third-party datasets contained under `data/`.

## Checkpoint availability

Checkpoints are not staged in Git. Hash registers and receiver-bank manifests are staged; full checkpoint archives belong in external archival storage such as Zenodo.

## Citation

Use `CITATION.cff`.

## Contact

Adrianto Mahendra Wijaya.

## Scientific claim boundary

No architecture-independent, universal, all-neural-architecture, or all-modern-forecasting-model claim is authorised by this freeze.
