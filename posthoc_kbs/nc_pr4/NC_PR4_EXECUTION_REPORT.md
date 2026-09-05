# NC_PR4_EXECUTION_REPORT

## 1. Final classification
CANONICAL_96_REPRODUCED = YES
PANEL_72_PORTFOLIO_MEDIAN_SIGNAL_PRESERVED = YES
PANEL_48_PORTFOLIO_MEDIAN_SIGNAL_PRESERVED = YES
Cross-architecture classification: CROSS_ARCH_DIRECTIONALLY_CONSISTENT_FOR_FUNCTIONAL

## 2. Authority and preflight
Authority files were resolved locally by path and SHA-256. The frozen TP2M4B2-R1 matched cohort has 29 countries; Palau remains the alignment boundary and Vanuatu remains excluded from paired FUNCTIONAL-vs-THETA_RKA comparison.
GRU and Transformer receiver banks were recovered as 225/225 valid and frozen. No training was performed.

## 3. Analysis A — alignment-panel sensitivity
The canonical 96-row panel was recomputed before interpreting 72/48 sensitivity. Pairwise outputs were written for LSTM reference, GRU, and Transformer receiver conditions across WHO, Electricity, and Dengue.
- LSTM_REFERENCE / WHO / panel 72: permutation agreement 0.622, median |delta TV| 0, median |delta ARI| 0.
- LSTM_REFERENCE / WHO / panel 48: permutation agreement 0.467, median |delta TV| 0.00189573, median |delta ARI| 0.
- LSTM_REFERENCE / Electricity / panel 72: permutation agreement 0.620, median |delta TV| 0, median |delta ARI| 0.
- LSTM_REFERENCE / Electricity / panel 48: permutation agreement 0.546, median |delta TV| 0, median |delta ARI| 0.
- LSTM_REFERENCE / Dengue / panel 72: permutation agreement 0.667, median |delta TV| 0, median |delta ARI| 0.
- LSTM_REFERENCE / Dengue / panel 48: permutation agreement 0.571, median |delta TV| 0, median |delta ARI| 0.
- GRU_CONTROLLED_ALTERNATIVE / WHO / panel 72: permutation agreement 0.644, median |delta TV| 0, median |delta ARI| 0.
- GRU_CONTROLLED_ALTERNATIVE / WHO / panel 48: permutation agreement 0.500, median |delta TV| 0, median |delta ARI| 0.
- GRU_CONTROLLED_ALTERNATIVE / Electricity / panel 72: permutation agreement 0.528, median |delta TV| 0, median |delta ARI| 0.
- GRU_CONTROLLED_ALTERNATIVE / Electricity / panel 48: permutation agreement 0.444, median |delta TV| 0.000158058, median |delta ARI| 0.
- GRU_CONTROLLED_ALTERNATIVE / Dengue / panel 72: permutation agreement 0.762, median |delta TV| 0, median |delta ARI| 0.
- GRU_CONTROLLED_ALTERNATIVE / Dengue / panel 48: permutation agreement 0.619, median |delta TV| 0, median |delta ARI| 0.
- TRANSFORMER_CONTROLLED_ALTERNATIVE / WHO / panel 72: permutation agreement 0.556, median |delta TV| 0, median |delta ARI| 0.
- TRANSFORMER_CONTROLLED_ALTERNATIVE / WHO / panel 48: permutation agreement 0.456, median |delta TV| 0.00274828, median |delta ARI| 0.
- TRANSFORMER_CONTROLLED_ALTERNATIVE / Electricity / panel 72: permutation agreement 0.537, median |delta TV| 0, median |delta ARI| 0.
- TRANSFORMER_CONTROLLED_ALTERNATIVE / Electricity / panel 48: permutation agreement 0.519, median |delta TV| 0, median |delta ARI| 0.
- TRANSFORMER_CONTROLLED_ALTERNATIVE / Dengue / panel 72: permutation agreement 0.619, median |delta TV| 0, median |delta ARI| 0.
- TRANSFORMER_CONTROLLED_ALTERNATIVE / Dengue / panel 48: permutation agreement 0.286, median |delta TV| 0.0336767, median |delta ARI| 0.

## 4. Analysis B — cross-architecture receiver generalization
- GRU: PRIMARY equal-country median delta_rho -0.0244219, country IQR [-0.07459229547631982, 0.06319928814732345], countries delta_rho < 0 19/29 (0.655), classification FUNCTIONAL_DIRECTIONALLY_BETTER.
- TRANSFORMER: PRIMARY equal-country median delta_rho -0.00476188, country IQR [-0.05914665383008247, 0.03010683303575351], countries delta_rho < 0 16/29 (0.552), classification FUNCTIONAL_DIRECTIONALLY_BETTER.
Native-equivalence maximum absolute error: 0.

## 5. Boundary cases
Palau was retained as the canonical WHO alignment boundary. Vanuatu was not substituted and remains outside the paired FUNCTIONAL-vs-THETA_RKA cohort because THETA_RKA checkpoints are absent for all three seeds. Electricity T118 was retained as an executability boundary in the structural sensitivity scope.

## 6. Integrity closure
NEW MODEL TRAINING PERFORMED: NO
OPTIMIZER INVOKED: NO
GRADIENTS APPLIED: NO
MODEL WEIGHTS MODIFIED: NO
PRE-EXISTING CHECKPOINTS MODIFIED: NO
NEW TRAINED CHECKPOINTS CREATED: NO
VALIDATION/TEST USED TO FIT ALIGNMENT: NO
FUNCTIONAL OUTCOME USED TO SELECT PERMUTATION: NO
COUNTRY/SERIES REPLACEMENT: NO
THRESHOLD TUNING: NO

## 7. Claim implications
The panel-size audit is a robustness descriptor only; it does not establish universal panel invariance. The cross-architecture audit is directional evidence under frozen held-out GRU and Transformer receiver banks, not statistical proof of universal transfer or non-transfer.

## 8. Next-step recommendation
Stop at NC-PR4 outputs. Do not edit the manuscript or launch additional training from this audit.
