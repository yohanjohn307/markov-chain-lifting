import numpy as np

from markov import _check_stochastic, _check_mapping, stationary_distribution


def kemeny(Pbar: np.ndarray, W: np.ndarray | None = None) -> float:
    """Compute the Kemeny constant.

    W is the edge weight matrix; if None, all weights default to 1.
    """
    _check_stochastic(Pbar)
    n = Pbar.shape[0]
    if W is None:
        W = np.ones((n, n))
    pi = stationary_distribution(Pbar)
    rhs = (Pbar * W) @ np.ones(n)
    M = np.zeros((n, n))
    for j in range(n):
        gamma_j = np.ones(n)
        gamma_j[j] = 0
        M[:, j] = np.linalg.solve(np.eye(n) - Pbar @ np.diag(gamma_j), rhs)
    return float(pi @ M @ pi)


def lifted_kemeny(P: np.ndarray, V: np.ndarray, W: np.ndarray | None = None) -> float:
    """Compute the lifted Kemeny constant.

    V is the n x m mapping matrix whose j-th column indicates which virtual states
    belong to physical node j. W is the edge weight matrix; if None, all weights default to 1.
    """
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    n, m = V.shape
    if W is None:
        W = np.ones((n, n))
    pi = stationary_distribution(P)
    rhs = (P * W) @ np.ones(n)
    M_lift = np.zeros((n, m))
    for j in range(m):
        D_j = np.diag(V[:, j])
        M_lift[:, j] = np.linalg.solve(np.eye(n) - P + P @ D_j, rhs)
    return float(pi @ M_lift @ V.T @ pi)


def stackelberg(P: np.ndarray, tau: np.ndarray) -> float:
    """Compute the Stackelberg game metric.

    tau[j] is the number of steps the attacker needs to complete an attack at node j.
    """
    _check_stochastic(P)
    n = P.shape[0]
    tau = np.asarray(tau, dtype=int)
    if tau.shape != (n,):
        raise ValueError(f"tau must have length n={n}, got shape {tau.shape}")
    Psi = np.zeros((n, n))
    Fk = P.copy()
    tau_max = int(tau.max())
    for k in range(1, tau_max + 1):
        Psi[:, tau >= k] += Fk[:, tau >= k]
        if k < tau_max:
            Fk = P @ (Fk - np.diag(np.diag(Fk)))
    return float(Psi.min())


def lifted_stackelberg(P: np.ndarray, V: np.ndarray, tau: np.ndarray) -> float:
    """Compute the lifted Stackelberg game metric.

    tau[j] is the number of steps the attacker needs to complete an attack at physical node j.
    """
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    n, m = V.shape
    tau = np.asarray(tau, dtype=int)
    if tau.shape != (m,):
        raise ValueError(f"tau must have length m={m}, got shape {tau.shape}")
    Psi_lift = np.zeros((n, m))
    Fk_lift = P @ V
    V_comp  = 1 - V
    tau_max = int(tau.max())
    for k in range(1, tau_max + 1):
        Psi_lift[:, tau >= k] += Fk_lift[:, tau >= k]
        if k < tau_max:
            Fk_lift = P @ (Fk_lift * V_comp)
    return float(Psi_lift.min())


def return_time_entropy(P: np.ndarray, eta: float = 0.01) -> float:
    """Compute the truncated Return-Time Entropy.

    K_eta = ceil(1 / (eta * pi_min)) - 1 controls truncation; eta upper-bounds discarded probability.
    """
    _check_stochastic(P)
    pi = stationary_distribution(P)
    K_eta = int(np.ceil(1.0 / (eta * pi.min()))) - 1
    H = 0.0
    Fk = P.copy()
    for k in range(1, K_eta + 1):
        d = np.diag(Fk)
        mask = d > 0
        H -= float(np.sum(pi[mask] * d[mask] * np.log(d[mask])))
        if k < K_eta:
            Fk = P @ (Fk - np.diag(d))
    return H


def lifted_return_time_entropy(P: np.ndarray, V: np.ndarray, eta: float = 0.01) -> float:
    """Compute the truncated lifted Return-Time Entropy.

    H^lift(P) = -sum_j pi_bar_j * sum_{k=1}^{K_eta} R_k(j) log R_k(j),
    where K_eta = ceil(1 / (eta * pi_bar_min)) - 1.
    """
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    pi = stationary_distribution(P)
    pi_bar = V.T @ pi
    K_eta = int(np.ceil(1.0 / (eta * pi_bar.min()))) - 1
    V_comp = 1.0 - V
    H = 0.0
    Fk_lift = P @ V
    for k in range(1, K_eta + 1):
        Rk = (V * pi[:, np.newaxis] * Fk_lift).sum(axis=0) / pi_bar
        mask = Rk > 0
        H -= float(np.sum(pi_bar[mask] * Rk[mask] * np.log(Rk[mask])))
        if k < K_eta:
            Fk_lift = P @ (Fk_lift * V_comp)
    return H
