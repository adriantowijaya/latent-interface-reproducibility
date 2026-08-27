# Model Architectures

## Reference TARELA-LSTM

Complete sparse latent-interface system with dynamic phase-state construction, numeric context, Sparsemax latent encoder, auxiliary heads, RevIN scaling, and an LSTM-50 temporal receiver. Frozen trainable parameter count: 12282.

## GRU Receiver Substitution

Same upstream sparse latent-interface system and training governance; only the LSTM receiver is replaced by a GRU-50 receiver using tanh activation, sigmoid recurrent activation, bias, and `reset_after=True`. Frozen trainable parameter count: 9432.

## Transformer Receiver Substitution

Same upstream sparse latent-interface system; only the temporal receiver is replaced by a frozen compact Transformer: input width 9, projection to 32, fixed sinusoidal positional encoding, one pre-LN causal self-attention block with 4 heads, feed-forward width 96, final normalization, last-token readout, and scalar forecast head. Frozen trainable parameter count: 11272.

These are controlled receiver substitutions, not plain LSTM, plain GRU, or plain Transformer benchmarks.
