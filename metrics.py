import numpy as np

from markov import _check_stochastic, _check_mapping, stationary_distribution


def kemeny(Pbar: np.ndarray, W: np.ndarray | None = None, pi: np.ndarray | None = None) -> float:
    """Compute the Kemeny constant.

    W is the edge weight matrix; if None, all weights default to 1.
    pi is Pbar's stationary distribution; if None, it is solved for via
    stationary_distribution(Pbar). Pass pi explicitly when it is already known from
    an ergodic flow matrix Q (pi = Q.sum(axis=1)) rather than re-derived from Pbar:
    that solve is a fresh linear system and can be ill-conditioned for chains with
    very small transition probabilities, even when Q's row sum is exact by
    construction (e.g. from a projected-gradient-descent optimizer).
    """
    _check_stochastic(Pbar)
    n = Pbar.shape[0]
    if W is None:
        W = np.ones((n, n))
    if pi is None:
        pi = stationary_distribution(Pbar)
    rhs = (Pbar * W) @ np.ones(n)
    M = np.zeros((n, n))
    for j in range(n):
        gamma_j = np.ones(n)
        gamma_j[j] = 0
        M[:, j] = np.linalg.solve(np.eye(n) - Pbar @ np.diag(gamma_j), rhs)
    return float(pi @ M @ pi)


def lifted_kemeny(
    P: np.ndarray, V: np.ndarray, W: np.ndarray | None = None, pi: np.ndarray | None = None
) -> float:
    """Compute the lifted Kemeny constant.

    V is the n x m mapping matrix whose j-th column indicates which virtual states
    belong to physical node j. W is the edge weight matrix; if None, all weights default to 1.
    pi is P's stationary distribution; if None it is solved for (see kemeny's docstring
    for why passing a known pi, e.g. from an ergodic flow's row sum, is preferable).
    """
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    n, m = V.shape
    if W is None:
        W = np.ones((n, n))
    if pi is None:
        pi = stationary_distribution(P)
    rhs = (P * W) @ np.ones(n)
    M_lift = np.zeros((n, m))
    for j in range(m):
        D_j = np.diag(V[:, j])
        M_lift[:, j] = np.linalg.solve(np.eye(n) - P + P @ D_j, rhs)
    return float(pi @ M_lift @ V.T @ pi)


def _first_passage_matrices(P: np.ndarray, V: np.ndarray, W: np.ndarray, K: int) -> dict:
    """Compute the (set) first passage probability matrices F^lift_1, ..., F^lift_K.

    Implements the travel-time recursion of Eq. (38):
        F^lift_k = P^(k) V + sum_{w=1}^{w_max} P^(w) [F^lift_{k-w} - (F^lift_{k-w} o V)],
    where P^(w) = P o 1{W = w}. Passing V = I recovers the (unlifted) matrices F_k.
    """
    W = np.asarray(W, dtype=int)
    w_max = int(W.max())
    P_w = {w: P * (W == w) for w in range(1, w_max + 1)}
    F = {0: np.zeros_like(V, dtype=float)}
    for k in range(1, K + 1):
        Fk = np.zeros_like(V, dtype=float)
        if k in P_w:
            Fk += P_w[k] @ V
        for w in range(1, w_max + 1):
            F_prev = F.get(k - w, F[0])
            Fk += P_w[w] @ (F_prev - F_prev * V)
        F[k] = Fk
    return F


def stackelberg(P: np.ndarray, tau: np.ndarray, W: np.ndarray | None = None) -> float:
    """Compute the Stackelberg game metric.

    tau[j] is the number of time steps the attacker needs to complete an attack at node j.
    W is the edge travel-time matrix (in N); if None, all travel times default to 1.
    """
    _check_stochastic(P)
    n = P.shape[0]
    tau = np.asarray(tau, dtype=int)
    if tau.shape != (n,):
        raise ValueError(f"tau must have length n={n}, got shape {tau.shape}")
    if W is None:
        W = np.ones((n, n), dtype=int)
    tau_max = int(tau.max())
    F = _first_passage_matrices(P, np.eye(n), W, tau_max)
    Psi = np.zeros((n, n))
    for k in range(1, tau_max + 1):
        Psi[:, tau >= k] += F[k][:, tau >= k]
    return float(Psi.min())


def lifted_stackelberg(P: np.ndarray, V: np.ndarray, tau: np.ndarray, W: np.ndarray | None = None) -> float:
    """Compute the lifted Stackelberg game metric.

    tau[j] is the number of time steps the attacker needs to complete an attack at physical node j.
    W is the edge travel-time matrix (in N); if None, all travel times default to 1.
    """
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    n, m = V.shape
    tau = np.asarray(tau, dtype=int)
    if tau.shape != (m,):
        raise ValueError(f"tau must have length m={m}, got shape {tau.shape}")
    if W is None:
        W = np.ones((n, n), dtype=int)
    pi = stationary_distribution(P)
    pi_bar = V.T @ pi
    tau_max = int(tau.max())
    F_lift = _first_passage_matrices(P, V, W, tau_max)
    Psi_lift = np.zeros((n, m))
    for k in range(1, tau_max + 1):
        Psi_lift[:, tau >= k] += F_lift[k][:, tau >= k]
    J_lift = (V * pi[:, np.newaxis]).T @ Psi_lift / pi_bar[:, np.newaxis]
    return float(J_lift.min())


def return_time_entropy(
    P: np.ndarray, eta: float = 0.1, W: np.ndarray | None = None, pi: np.ndarray | None = None
) -> float:
    """Compute the truncated Return-Time Entropy.

    K_eta = ceil(1 / (eta * pi_min)) - 1 controls truncation; eta upper-bounds discarded probability.
    W is the edge travel-time matrix (in N); if None, all travel times default to 1.
    pi is P's stationary distribution; if None it is solved for (see kemeny's docstring
    for why passing a known pi, e.g. from an ergodic flow's row sum, is preferable —
    K_eta is especially sensitive to pi.min(), so an ill-conditioned solve here can
    blow up the truncation length).
    """
    _check_stochastic(P)
    n = P.shape[0]
    if W is None:
        W = np.ones((n, n), dtype=int)
    if pi is None:
        pi = stationary_distribution(P)
    K_eta = int(np.ceil(1.0 / (eta * pi.min()))) - 1
    F = _first_passage_matrices(P, np.eye(n), W, K_eta)
    H = 0.0
    for k in range(1, K_eta + 1):
        d = np.diag(F[k])
        mask = d > 0
        H -= float(np.sum(pi[mask] * d[mask] * np.log(d[mask])))
    return H


def lifted_return_time_entropy(
    P: np.ndarray, V: np.ndarray, eta: float = 0.1, W: np.ndarray | None = None,
    pi: np.ndarray | None = None,
) -> float:
    """Compute the truncated lifted Return-Time Entropy.

    H^lift(P) = -sum_j pi_bar_j * sum_{k=1}^{K_eta} R_k(j) log R_k(j),
    where K_eta = ceil(1 / (eta * pi_bar_min)) - 1.
    W is the edge travel-time matrix (in N); if None, all travel times default to 1.
    pi is P's stationary distribution; if None it is solved for (see kemeny's docstring
    for why passing a known pi, e.g. from an ergodic flow's row sum, is preferable —
    K_eta is especially sensitive to pi_bar.min(), so an ill-conditioned solve here can
    blow up the truncation length).
    """
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    n = P.shape[0]
    if W is None:
        W = np.ones((n, n), dtype=int)
    if pi is None:
        pi = stationary_distribution(P)
    pi_bar = V.T @ pi
    K_eta = int(np.ceil(1.0 / (eta * pi_bar.min()))) - 1
    F_lift = _first_passage_matrices(P, V, W, K_eta)
    H = 0.0
    for k in range(1, K_eta + 1):
        Rk = (V * pi[:, np.newaxis] * F_lift[k]).sum(axis=0) / pi_bar
        mask = Rk > 0
        H -= float(np.sum(pi_bar[mask] * Rk[mask] * np.log(Rk[mask])))
    return H
