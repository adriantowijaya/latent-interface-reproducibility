# Latent-Interface Reliability Reproducibility Package

This repository provides the versioned reproducibility materials for "Evaluating Neural Time-Series Latent-Interface Reliability: Structural Reproducibility, Receiver Compatibility, and Receiver Generalization". Version 1.1.0 preserves the original v1.0.0 experimental foundation and adds the post hoc, zero-training KBS robustness and receiver-generalization analyses.

## Scientific purpose

The repository supports zero-training regeneration of manuscript evidence from frozen machine-readable results, plus a documented path for full computational reproduction.

Version 1.1.0 is an **additive reproducibility extension**. The original v1.0.0 data, configurations, code, results, environment records, and checkpoint provenance remain authoritative for the experiments that preceded the KBS retarget. The new `posthoc_kbs/` directory adds NC-PR4 alignment-panel sensitivity, cross-architecture receiver evidence, and NC-PR7R estimator-harmonized Level-3 evidence. These secondary files do not replace or overwrite the original confirmatory reference-LSTM gate.

## What this repository reproduces

The v1.0.0 foundation covers the reference TARELA-LSTM configuration, GRU controlled receiver substitution, Transformer controlled receiver substitution, WHO, Electricity-37 and Dengue-7 structural replications, TP2B latent analysis, architecture-common functional diagnostics, receiver swaps, TP2D LSTM-specific diagnostics, intervention gates, and original manuscript numerical summaries.

The v1.1.0 KBS extension additionally covers:
- deterministic 96/72/48 alignment-panel sensitivity;
- zero-training cross-architecture receiver evaluation through frozen GRU and Transformer receiver banks;
- paired FUNCTIONAL-minus-THETA-RKA country-level receiver evidence;
- estimator-harmonized Level-3 summaries for reference LSTM, GRU, and Transformer receiving contexts;
- source-hash, integrity, freeze, and execution evidence for the post hoc analyses.

See [`KBS_REPRODUCIBILITY_MAP.md`](KBS_REPRODUCIBILITY_MAP.md) for the manuscript-to-artifact map.

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
3. Run the original zero-training quick reproduction commands below.
4. Run `python scripts/verify_kbs_extension.py` to verify the v1.1.0 post hoc evidence bundle.
5. Read [KBS_REPRODUCIBILITY_MAP.md](KBS_REPRODUCIBILITY_MAP.md) to locate the evidence underlying each KBS table/figure.
6. Use [audit/](audit/) and `posthoc_kbs/` for deeper provenance, integrity, and freeze evidence.

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

`src/` contains portable code, `configs/` contains frozen protocol contracts, `data/` contains processed analytical data, `results/` contains the original machine-readable authority results, `manifests/` contains cohort and checkpoint registers, `audit/` contains release evidence, and `posthoc_kbs/` contains the additive NC-PR4/NC-PR7R zero-training KBS extension.

## Environment

The canonical scientific reproduction specification is [environment-canonical.yml](environment-canonical.yml): Python 3.9.16 with TensorFlow 2.13.0, NumPy 1.24.3, Pandas 2.0.3, and h5py 3.9.0. The zero-training validation environment is recorded in [environment-verification.txt](environment-verification.txt) and [environment-lock.txt](environment-lock.txt). The exact historical training package lock is not fully recoverable from persisted metadata; the observed TensorFlow and NumPy difference is documented as a provenance boundary, not scientific protocol drift.

## Data licensing

The [MIT License](LICENSE) applies to original source code and repository documentation authored for this project. It does not relicense third-party datasets contained under [data/](data/). Dataset-specific sources and terms are documented in [DATA_LICENSES.md](DATA_LICENSES.md).

## Checkpoint availability

Frozen checkpoint banks are not stored in Git history. Checkpoint hash registers and receiver-bank manifests are provided for provenance. A separately versioned archival package is associated with the repository for public archival deposition.

## Archival record

The original v1.0.0 archive is publicly archived at Zenodo DOI `10.5281/zenodo.22140513`. The KBS v1.1.0 extension is intended to be deposited as a **new Zenodo version**, retaining the original foundation files and adding the post hoc KBS extension. The manuscript should cite the **specific v1.1.0 version DOI** once Zenodo assigns and publishes it.

- v1.0.0 foundation DOI: `10.5281/zenodo.22140513`
- v1.1.0 archival DOI: to be recorded after publication of the Zenodo new version.

## Citation

Use [CITATION.cff](CITATION.cff).

## Contact

Adrianto Mahendra Wijaya.

## Scientific claim boundary

No architecture-independent, universal, all-neural-architecture, or all-modern-forecasting-model claim is authorised by this freeze.

## Version lineage

- **v1.0.0** — original reproducibility freeze for the submitted Neurocomputing manuscript; original data/code/configuration/result/checkpoint authority.
- **v1.1.0** — additive KBS reproducibility extension; no new training; adds NC-PR4 and NC-PR7R secondary evidence and KBS manuscript mapping.

The v1.1.0 release does not rewrite the original confirmatory experiment and does not authorize a universal receiver-generalization or intervention-superiority claim.
