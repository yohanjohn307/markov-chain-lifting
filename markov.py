import numpy as np


EQUALITY_ATOL = 1e-3


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _check_stochastic(P: np.ndarray) -> None:
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {P.shape}")
    if not np.all(P >= 0):
        raise ValueError("transition matrix must have non-negative entries")
    if not np.allclose(P.sum(axis=1), 1, atol=EQUALITY_ATOL):
        raise ValueError("transition matrix must be row-stochastic (each row must sum to 1)")


def _check_mapping(V: np.ndarray, n: int | None = None) -> None:
    if V.ndim != 2:
        raise ValueError(f"V must be a 2D matrix, got shape {V.shape}")
    if not np.all((V == 0) | (V == 1)):
        raise ValueError("mapping matrix V must have binary (0/1) entries")
    if not np.all(V.sum(axis=1) == 1):
        raise ValueError("each row of V must sum to 1 (each virtual state maps to exactly one physical node)")
    if n is not None and V.shape[0] != n:
        raise ValueError(f"V has {V.shape[0]} rows but P has {n} states")


def _check_ergodic_flow(Q: np.ndarray, m: int | None = None) -> None:
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {Q.shape}")
    if m is not None and Q.shape[0] != m:
        raise ValueError(f"Q_bar must be {m}x{m} to match V, got shape {Q.shape}")
    if not np.all(Q >= 0):
        raise ValueError("ergodic flow matrix must have non-negative entries")
    row_sums = Q.sum(axis=1)
    if not np.all(row_sums > 0):
        raise ValueError("ergodic flow matrix must have positive row sums")
    if not np.allclose(row_sums, Q.sum(axis=0), atol=EQUALITY_ATOL):
        raise ValueError("ergodic flow matrix must satisfy Q 1 = Q^T 1 (equal row and column sums)")


# ---------------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------------

def stationary_distribution(Pbar: np.ndarray) -> np.ndarray:
    """Solve for the unique stationary distribution pi of an irreducible MC."""
    _check_stochastic(Pbar)
    n = Pbar.shape[0]
    A = Pbar.T - np.eye(n) + np.ones((n, n))
    return np.linalg.solve(A, np.ones(n))


def transition_to_ergodic_flow(P: np.ndarray) -> np.ndarray:
    """Compute the ergodic flow matrix Q = diag(pi) @ P for a transition matrix P."""
    _check_stochastic(P)
    pi = stationary_distribution(P)
    return np.diag(pi) @ P


def ergodic_flow_to_transition(Q: np.ndarray) -> np.ndarray:
    """Recover the transition matrix P = diag(pi)^{-1} @ Q from an ergodic flow matrix Q."""
    _check_ergodic_flow(Q)
    pi = Q.sum(axis=1)
    return np.diag(1.0 / pi) @ Q


def collapsing(P: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the lifted MC P to the physical space using V.

    Returns (P_bar, pi_bar): the collapsed transition matrix and its stationary distribution.
    """
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    pi = stationary_distribution(P)
    Pi = np.diag(pi)
    Pbar = np.linalg.solve(V.T @ Pi @ V, V.T @ Pi @ P @ V)
    return Pbar, V.T @ pi


def conductance(Pbar: np.ndarray) -> tuple[float, float]:
    """Compute the conductance and its associated Kemeny constant lower bound.

    Enumerates all 2^n - 2 subsets, so only practical for small state spaces.

    Returns (phi, lower_bound) where lower_bound = 1 / (2 * phi).
    """
    _check_stochastic(Pbar)
    n = Pbar.shape[0]
    pi = stationary_distribution(Pbar)
    Qbar = np.diag(pi) @ Pbar
    phi = np.inf
    for mask in range(1, 2**n - 1):
        A  = [i for i in range(n) if     mask & (1 << i)]
        Ac = [i for i in range(n) if not mask & (1 << i)]
        flow = Qbar[np.ix_(A, Ac)].sum()
        pi_A = pi[A].sum()
        phi  = min(phi, flow / (pi_A * (1.0 - pi_A)))
    return float(phi), float(1.0 / (2.0 * phi))
