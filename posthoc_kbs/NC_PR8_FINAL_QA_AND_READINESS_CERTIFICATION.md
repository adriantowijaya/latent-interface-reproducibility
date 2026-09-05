# NC-PR8 — Final Scientific Source QA, Residual-Claim Audit, and Retargeting Readiness Certification

## Final classification

**NC-PR8 = PASS — CONTROLLED POST-REJECTION MASTER CERTIFIED**

Controlled master: **v3.2**

Retargeting status:
- **READY_FOR_JOURNAL_TARGET_SELECTION = YES**
- **READY_FOR_TARGET_TEMPLATE_MIGRATION = YES**
- **READY_FOR_SUBMISSION_AS-IS = NO**
- **NEW TRAINING REQUIRED = NO**
- **NEW EXPERIMENT REQUIRED = NO**

## 1. Critical source-level QA finding and repair

NC-PR8 found one non-scientific source defect in v3.1: the LaTeX source contained a duplicated `\end{abstract}`, which prevented a clean compile. This was a reconstruction artifact, not a scientific-content defect.

The duplicated terminator was removed in v3.2. **No claim, number, estimator, result, citation argument, or experimental boundary was changed by this hotfix.**

## 2. Scientific claim-evidence certification

### Level 1 — PASS

The structural-reproducibility story remains correctly bounded:
- canonical 96-row train-inner alignment remains authoritative;
- 72/48 panels remain deterministic sensitivity perturbations;
- exact permutation stability is not claimed;
- portfolio-level structural-signal preservation is retained;
- continuous aligned TV and ARI remain primary evidence.

### Level 2 — PASS

Receiver-conditioned functional evidence remains based on architecture-common continuous diagnostics. The LSTM phenotype taxonomy is not used as the cross-architecture headline.

### Level 3 — PASS AFTER NC-PR7R HARMONIZATION

The manuscript now keeps two estimators distinct:

**Original confirmatory reference-LSTM G5**
- FUNCTIONAL equal-country median: 1.2752816602389132
- THETA-RKA equal-country median: 1.2552870798610352
- frozen difference: **+0.0199945803778780**
- status: **FAIL**, unchanged.

**Estimator-harmonized primary cross-seed descriptive comparison**
- Reference LSTM: **-0.016129**, IQR [-0.168399, 0.058908], 17/29 negative.
- GRU: **-0.024422**, IQR [-0.074592, 0.063199], 19/29 negative.
- Transformer: **-0.004762**, IQR [-0.059147, 0.030107], 16/29 negative.

The withdrawn primary sign-reversal interpretation does not survive in positive form. The manuscript instead states that receiver-generalization evidence is heterogeneous and estimator-sensitive.

## 3. Residual-claim audit

Withdrawn or prohibited headline claims were checked.

Result:
- primary cross-seed LSTM-vs-GRU/Transformer sign reversal: **ABSENT**
- `functional equivalence` as headline novelty: **ABSENT**
- `receiver transferability` as headline construct: **ABSENT**
- universal FUNCTIONAL superiority: **NOT CLAIMED**
- architecture-independent latent instability: **NOT CLAIMED**
- exact permutation invariance: **NOT CLAIMED**
- semantic/causal latent-regime identifiability: **NOT CLAIMED**

Occurrences of “sign reversal” that remain are either:
1. explicit statements that the Level-3 primary sign-reversal interpretation is **not supported**, or
2. the legitimate Level-2 Dengue GRU forecast-delta-cosine sign reversal.

## 4. Scientific chronology and provenance audit

The manuscript now distinguishes:
- the original prospectively governed/frozen experiment and reference-LSTM intervention; from
- NC-PR4, which is transparently labelled a **post hoc secondary zero-training analysis**, formulated after the submitted v2.9 outcomes/editorial decision were known, with its own computational protocol frozen before inspection of NC-PR4 outcomes.

This chronology is acceptable for resubmission provided the wording is retained.

## 5. Prior-art boundary QA

The bibliography now includes the closest post-submission/2026 boundary papers used in reconstruction:
- Smith et al. (ICML 2025), functional alignment can mislead;
- Athanasiadis et al. (ICML 2026 regular), invariance-aware model stitching;
- Mai et al. (CVPR 2026), heterogeneous vision foundation-model stitching.

The manuscript therefore does **not** claim that functional alignment, cross-model stitching, or cross-mechanism stitchability is itself new.

The defensible contribution remains:
> **formulating and operationalizing an ordered neural time-series latent-interface protocol that separately adjudicates symmetry-resolved structural reproducibility, receiver-conditioned function, and held-out receiver generalization under a common explicit upstream interface.**

## 6. Citation/source QA

- citation keys used: **29**
- BibTeX entries: **29**
- missing citation keys: **NONE**
- unused bibliography entries: **NONE**
- arXiv URLs in manuscript/BibTeX: **NONE**
- abstract length: **198 words**

Official data-source citations are retained for WHO COVID-19 and Taiwan CDC dengue provenance; peer-reviewed sources are used for scientific methodological/literature claims.

## 7. Numerical-invariant QA

Required values present in the controlled source:

- G2A: -0.0038 — PASS
- G2B: +0.6442 — PASS
- G3: 0.1166 — PASS
- G4: 1.0894 — PASS
- G5_exact: +0.01999458 — PASS
- harm_LSTM: -0.016129 — PASS
- harm_GRU: -0.024422 — PASS
- harm_Transformer: -0.004762 — PASS
- same_seed_LSTM: +0.459987 — PASS
- same_seed_GRU: -0.002801 — PASS
- same_seed_Transformer: +0.035155 — PASS

No numerical change was made in NC-PR8.

## 8. Reader-facing temporal QA

W05 is defined in reader-facing form as the fifth chronological evaluation block. For WHO:
- validation: 16 Nov 2021–13 Dec 2021;
- out-of-sample test targets: 14 Dec 2021–10 Jan 2022.

The manuscript no longer relies on the internal token `W05` alone.

## 9. LaTeX/source integrity QA

- environment balance: **PASS**
- duplicate labels: **NONE**
- missing `\ref` labels: **NONE**
- citation-key completeness: **PASS**
- source-level abstract hotfix: **PASS**

The journal-neutral layout still contains wide tables/figure specification placeholders. These are production issues to solve during journal-template migration, not scientific-source blockers.


### Compilation and BibTeX-database validation

After the abstract hotfix, the v3.2 source completes two `pdflatex` passes without a fatal LaTeX error. Cross-reference structure is valid. The journal-neutral layout still reports overfull/underfull boxes in wide tables; these are explicitly deferred to target-template migration.

The `.bib` database passes Biber tool-mode data-model validation after normalizing the CVPR month field. The runtime does not provide a BibTeX executable, so end-to-end bibliography rendering with `plainnat` was not executed; however, all citation keys used by the manuscript are present in the bibliography and the database syntax/data model is valid.


## 10. Retargeting readiness decision

### Certified now

**YES — the manuscript is scientifically ready for journal retargeting and journal-specific reconstruction.**

This means the scientific master is sufficiently stable to:
1. select a target journal;
2. audit journal fit and prior-art expectations;
3. migrate to the journal template;
4. draw/finalize the new figures;
5. update supplementary/reproducibility materials.

### Not certified yet

**NO — v3.2 is not submission-ready as-is.**

Remaining production blockers:
1. Figures 1–4 are still specifications/placeholders rather than final artwork.
2. The public reproducibility release must be updated to include NC-PR4 and NC-PR7R outputs and hashes.
3. Journal-specific template, reference style, author metadata, CRediT/ORCID, declarations, and file-package requirements must be applied.
4. A final compiled-PDF scientific/editorial QA is required after template migration.

## Final controlled disposition

**SCIENTIFIC MASTER = v3.2 CERTIFIED**

**NEXT ACTION = JOURNAL RETARGETING / FIT SELECTION**

No further training or experimental expansion is authorized by the current evidence.
