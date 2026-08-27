# Full Computational Reproduction

This workflow is prepared for a separate supervised phase and is not run during TP-2M.5E.2.

1. Verify environment and data hashes.
2. Train reference TARELA-LSTM receiver banks.
3. Train GRU external receiver banks.
4. Train Transformer external receiver banks.
5. Freeze the valid checkpoint bank.
6. Run structural alignment audit.
7. Run architecture-common functional audit.
8. Run LSTM-specific TP2D diagnostics.
9. Run WHO intervention.
10. Regenerate manuscript tables and figures.

Do not expect bitwise-identical checkpoints unless the exact platform and TensorFlow stack are reproduced and explicitly validated.
