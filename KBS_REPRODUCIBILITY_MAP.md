# KBS Reproducibility Map

This map links the Knowledge-Based Systems manuscript evidence to the versioned repository.

## Scientific authority layers

### Foundation — v1.0.0
The original repository remains authoritative for:
- WHO-31, Electricity-37, and Dengue-7 processed analytical series;
- canonical temporal/evaluation protocol;
- reference LSTM, GRU, and Transformer configurations;
- original structural and functional summaries;
- original reference-LSTM intervention gate hierarchy;
- checkpoint manifests and archival checkpoint banks;
- environment and source provenance.

### KBS extension — v1.1.0
`posthoc_kbs/` adds only secondary zero-training analyses created after the Neurocomputing submission:
- NC-PR4 alignment-panel sensitivity;
- NC-PR4 GRU/Transformer receiving-context analysis;
- NC-PR7R harmonized Level-3 estimator evidence.

These files are additive. They do not replace the v1.0.0 confirmatory experiment.

## Manuscript-to-artifact map

| KBS evidence object | Primary repository authority | KBS v1.1.0 extension |
|---|---|---|
| Table 1 — closest prior-art boundary | manuscript literature synthesis | none |
| Level-1 canonical structural results / Table 2 | `results/structural/TP2M4D_LATENT_SERIES_SUMMARY.csv`; `results/structural/TP2M5B_R1_GRU_VS_LSTM_STRUCTURAL_REFERENCE.csv`; `results/structural/TP2M5C_MULTIARCH_STRUCTURAL_REFERENCE.csv` | none |
| Figure 2A/B — aligned TV and ARI | same Level-1 foundation results | none |
| Figure 2C — 96/72/48 panel sensitivity | canonical 96 authority from v1.0.0 | `posthoc_kbs/nc_pr4/NC_PR4_ALIGNMENT_PANEL_SENSITIVITY_SUMMARY.csv` and pairwise file |
| Table 3 / Figure 3 — receiver-conditioned functional diagnostics | `results/functional/TP2M4D_FUNCTIONAL_PER_MODEL.csv`; `results/functional/TP2M5B_R1_GRU_VS_LSTM_FUNCTIONAL_COMMON_CORE.csv`; `results/functional/TP2M5C_MULTIARCH_FUNCTIONAL_COMMON_CORE.csv` | none |
| Original confirmatory gate table / G5 | `results/intervention/TP2M4B2_R1_GATE_TABLE_STRICT_AND_BOUNDARY_AWARE.csv`; `results/intervention/TP2M4B2_R1_COUNTRY_RESULT_MATRIX.csv` | unchanged |
| Cross-architecture receiving-context rows | frozen v1.0.0 checkpoint banks + original receiver definitions | `posthoc_kbs/nc_pr4/NC_PR4_CROSS_ARCH_RECEIVER_ROWS.csv` |
| Paired GRU/Transformer country evidence | same frozen checkpoint foundation | `posthoc_kbs/nc_pr4/NC_PR4_CROSS_ARCH_PAIRED_DELTA_BY_COUNTRY.csv` |
| Figure 4 / harmonized Level-3 summary | original LSTM intervention authority remains separately confirmatory | `posthoc_kbs/nc_pr7r/NC_PR7R_HARMONIZED_LEVEL3_EVIDENCE.csv` plus NC-PR4 paired outputs |
| Table 5 — claim/evidence synthesis | all above | all above |

## Key numerical invariants

Original confirmatory reference-LSTM G5 remains:
- FUNCTIONAL equal-country median: 1.2752816602389132
- THETA-RKA equal-country median: 1.2552870798610352
- frozen difference: +0.019994580377878046
- status: FAIL

Harmonized primary cross-seed descriptive evidence:
- Reference LSTM: median paired delta-rho -0.016129; IQR [-0.168399, 0.058908]; 17/29 countries negative.
- GRU: -0.024422; IQR [-0.074592, 0.063199]; 19/29 negative.
- Transformer: -0.004762; IQR [-0.059147, 0.030107]; 16/29 negative.

The confirmatory G5 estimator and the harmonized paired estimator are different and must not be merged.

## Reproduction boundary

The KBS extension is explicitly **post hoc and zero-training**:
- no optimizer invocation;
- no gradient application;
- no weight modification;
- no new trained checkpoint;
- no outcome-based series replacement;
- no threshold tuning.

Palau, Electricity T118, and Vanuatu retain their documented executability boundaries.
