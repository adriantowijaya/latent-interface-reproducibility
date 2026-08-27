from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


K = 5


@dataclass(frozen=True)
class AlignmentResult:
    permutation: tuple[int, ...]
    cost: float
    cost_matrix: np.ndarray
    n_observations: int
    fit_partition: str = "TRAIN_INNER"
    functional_outcome_used: bool = False


def _as_panel(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 3:
        arr = arr[:, -1, :]
    if arr.ndim != 2 or arr.shape[1] != K:
        raise ValueError(f"Expected [n,{K}] or [n,t,{K}] theta panel, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Theta panel contains non-finite values")
    return arr


def assignment_cost_matrix(theta_sender: np.ndarray, theta_receiver: np.ndarray) -> np.ndarray:
    """Frozen TP2B/TP2E mean-squared soft-activation assignment cost."""
    a = _as_panel(theta_sender)
    b = _as_panel(theta_receiver)
    if a.shape != b.shape:
        raise ValueError(f"Aligned panels must have identical shape, got {a.shape} vs {b.shape}")
    cost = np.empty((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            cost[i, j] = float(np.mean((a[:, i] - b[:, j]) ** 2))
    return cost


def optimal_permutation_k5(cost: np.ndarray) -> tuple[tuple[int, ...], float]:
    """Exhaustive K=5 search; itertools order supplies lexicographic tie break."""
    c = np.asarray(cost, dtype=float)
    if c.shape != (K, K):
        raise ValueError(f"Expected a {K}x{K} cost matrix, got {c.shape}")
    best: tuple[int, ...] | None = None
    best_cost = float("inf")
    for p in itertools.permutations(range(K)):
        value = float(sum(c[i, p[i]] for i in range(K)))
        if value < best_cost:
            best = tuple(int(x) for x in p)
            best_cost = value
    if best is None:
        raise RuntimeError("No permutation found")
    return best, best_cost


def apply_permutation(theta: np.ndarray, perm: Sequence[int]) -> np.ndarray:
    """Apply component mapping to the final axis."""
    p = np.asarray(tuple(int(x) for x in perm), dtype=int)
    if p.shape != (K,) or sorted(p.tolist()) != list(range(K)):
        raise ValueError(f"Invalid K=5 permutation: {perm}")
    return np.asarray(theta)[..., p]


def inverse_permutation(perm: Sequence[int]) -> tuple[int, ...]:
    p = np.asarray(tuple(int(x) for x in perm), dtype=int)
    if p.shape != (K,) or sorted(p.tolist()) != list(range(K)):
        raise ValueError(f"Invalid K=5 permutation: {perm}")
    inv = np.empty_like(p)
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(int(x) for x in inv)


def epoch_p_align(
    theta_sender_train_panel: np.ndarray,
    theta_receiver_train_panel: np.ndarray,
    *,
    panel_indices: Iterable[int] | None = None,
) -> AlignmentResult:
    """Compute one stop-gradient train-inner alignment for an epoch.

    The function accepts theta panels only. No functional effect or forecast
    quantity is an input, by design.
    """
    sender = _as_panel(theta_sender_train_panel)
    receiver = _as_panel(theta_receiver_train_panel)
    if panel_indices is not None:
        idx = np.asarray(list(panel_indices), dtype=int)
        sender = sender[idx]
        receiver = receiver[idx]
    cost = assignment_cost_matrix(sender, receiver)
    perm, value = optimal_permutation_k5(cost)
    return AlignmentResult(
        permutation=perm,
        cost=float(value),
        cost_matrix=cost,
        n_observations=int(len(sender)),
    )

