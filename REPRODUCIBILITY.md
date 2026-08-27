# Reproducibility

## LEVEL 1 - QUICK / ZERO TRAINING

Regenerates manuscript tables and figures from frozen machine-readable results. Expected runtime is minutes. This level performs environment, data hash, protocol, and parameter smoke checks without optimizer steps or checkpoint inference.

## LEVEL 2 - FULL COMPUTATIONAL REPRODUCTION

Retrains the receiver banks and reruns structural and functional audits. Expected runtime is substantially longer. Fixed seeds and deterministic TensorFlow controls are used, but bitwise-identical checkpoints are not promised unless the exact platform stack is reproduced and independently demonstrated.
