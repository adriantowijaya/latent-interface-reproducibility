# Neurocomputing Reproducibility Package

This repository provides the reproducibility materials for "Latent Representation Instability Across Recurrent and Attention-Based Receivers: Functional Equivalence and Receiver Transfer in Neural Time-Series Forecasting".

## Scientific purpose

The repository supports zero-training regeneration of manuscript tables and figures from frozen machine-readable results, plus a documented path for full computational reproduction.

## What this repository reproduces

It covers the reference TARELA-LSTM configuration, GRU controlled receiver substitution, Transformer controlled receiver substitution, WHO, Electricity-37 and Dengue-7 structural replications, TP2B latent analysis, architecture-common functional diagnostics, receiver swaps, TP2D LSTM-specific diagnostics, intervention gates, and manuscript numerical summaries.

## What this repository does not claim

This repository does not include the frozen checkpoint banks in Git history and does not claim architecture-independent or universal latent-instability behaviour beyond the controlled receiver substitutions evaluated in the study.

## Architecture configurations

The alternative GRU and Transformer configurations are controlled receiver substitutions within the same sparse latent-interface system; they are not standalone state-of-the-art forecasting baselines.

The reference TARELA-LSTM implementation is included to make the study self-contained. Its inclusion in this repository does not position TARELA-LSTM itself as the methodological novelty of the accompanying manuscript.

## Data strata

Staged data are processed analytical files for WHO COVID-19, Electricity-37, and Dengue-7. Raw Taiwan Dengue case-level records, raw full WHO archives, and the full raw Electricity archive are absent from Git history. See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) and [DATA_LICENSES.md](DATA_LICENSES.md).

The Dengue data retain a column named `Country` for frozen analytical compatibility with the shared series-processing interface. In this Dengue subset, `Country` identifies county/city series labels and does not mean sovereign country.

## Reviewer quick path

1. Read [MODEL_ARCHITECTURES.md](MODEL_ARCHITECTURES.md) to identify the reference LSTM and controlled GRU/Transformer receiver substitutions.
2. Read [ENVIRONMENT_PROVENANCE.md](ENVIRONMENT_PROVENANCE.md) to distinguish the canonical environment, quick verification environment, and historical training-evidence boundary.
3. Run the zero-training quick reproduction commands below.
4. Inspect regenerated manuscript outputs under [results/manuscript/](results/manuscript/).
5. Use [audit/](audit/) only for deeper provenance, licensing, environment, and release-closure evidence.

## Quick reproduction

Run:

```bash
python scripts/verify_environment.py
python scripts/verify_data_hashes.py
python scripts/verify_protocol.py
python scripts/smoke_test_models.py
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
```

This path performs zero training, zero optimizer steps, and no checkpoint inference. Expected outputs are regenerated from frozen machine-readable results. The smoke-test identities are LSTM 12282 parameters, GRU 9432 parameters, Transformer 11272 parameters, K=5, L=7, H=1, and fusion width=9.

## Full reproduction

See [full_reproduction/README.md](full_reproduction/README.md). This route documents retraining and complete mechanism reproduction, and is separate from the zero-training reviewer path. Fixed seeds and deterministic TensorFlow controls are used, but bitwise-identical checkpoints across platforms are not guaranteed.

## Known executability boundaries

Palau WHO structural alignment is not executable because the train-inner transition-intensity quantiles collapse. T118 Electricity structural alignment is not executable because the canonical middle context has fewer than 32 observations. The Vanuatu LSTM THETA-RKA comparator reaches a deterministic non-finite-gradient boundary in intervention. These cases were retained in applicable forecast denominators and were not replaced.

## Repository layout

`src/` contains portable code, `configs/` contains frozen protocol contracts, `data/` contains processed analytical data, `results/` contains machine-readable authority results, `manifests/` contains cohort and checkpoint registers, and `audit/` contains release evidence.

## Environment

The canonical scientific reproduction specification is [environment-canonical.yml](environment-canonical.yml): Python 3.9.16 with TensorFlow 2.13.0, NumPy 1.24.3, Pandas 2.0.3, and h5py 3.9.0. The zero-training validation environment is recorded in [environment-verification.txt](environment-verification.txt) and [environment-lock.txt](environment-lock.txt). The exact historical training package lock is not fully recoverable from persisted metadata; the observed TensorFlow and NumPy difference is documented as a provenance boundary, not scientific protocol drift.

## Data licensing

The [MIT License](LICENSE) applies to original source code and repository documentation authored for this project. It does not relicense third-party datasets contained under [data/](data/). Dataset-specific sources and terms are documented in [DATA_LICENSES.md](DATA_LICENSES.md).

## Checkpoint availability

Frozen checkpoint banks are not stored in Git history. Checkpoint hash registers and receiver-bank manifests are provided for provenance. A separately versioned archival package is associated with the repository for public archival deposition.

## Archival record

A versioned archival record of the reproducibility materials, including the frozen checkpoint banks, is associated with Zenodo DOI 10.5281/zenodo.22140513.

DOI: https://doi.org/10.5281/zenodo.22140513

## Citation

Use [CITATION.cff](CITATION.cff).

## Contact

Adrianto Mahendra Wijaya.

## Scientific claim boundary

No architecture-independent, universal, all-neural-architecture, or all-modern-forecasting-model claim is authorised by this freeze.
