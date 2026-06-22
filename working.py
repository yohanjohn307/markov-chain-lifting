
import numpy as np

def _check_stochastic(P: np.ndarray) -> None:
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {P.shape}")
    if not np.all(P >= 0):
        raise ValueError("transition matrix must have non-negative entries")
    if not np.allclose(P.sum(axis=1), 1):
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

def stationary_distribution(Pbar: np.ndarray) -> np.ndarray:
    """Solve for the unique stationary distribution pi of an irreducible MC."""
    _check_stochastic(Pbar)
    n = Pbar.shape[0]
    A = Pbar.T - np.eye(n) + np.ones((n, n))
    b = np.ones(n)
    return np.linalg.solve(A, b)

def kemeny(Pbar: np.ndarray) -> float:
    """Compute the Kemeny constant."""
    _check_stochastic(Pbar)
    n = Pbar.shape[0]
    pi = stationary_distribution(Pbar)
    M = np.zeros((n, n))
    for j in range(n):
        gamma_j = np.ones(n)
        gamma_j[j] = 0
        M[:, j] = np.linalg.solve(np.eye(n) - Pbar @ np.diag(gamma_j), np.ones(n))
    return float(pi @ M @ pi)

def lifted_kemeny(P: np.ndarray, V: np.ndarray) -> float:
    """Compute the lifted Kemeny constant for a lifted Markov chain with transition matrix P and mapping matrix V.
    V is the n x m mapping matrix whose j-th column indicates which virtual states belong to physical node j.
    """
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    n, m = V.shape
    pi = stationary_distribution(P)
    M_lift = np.zeros((n, m))
    for j in range(m):
        D_j = np.diag(V[:, j])
        M_lift[:, j] = np.linalg.solve(np.eye(n) - P + P @ D_j, np.ones(n))
    return float( pi @ M_lift @ V.T @ pi )

def conductance(P: np.ndarray) -> float:
    """Compute the conductance.
    Enumerates all 2^n - 2 subsets, so only practical for small state spaces.
    """
    _check_stochastic(P)
    n = P.shape[0]
    pi = stationary_distribution(P)
    Q = np.diag(pi) @ P  # ergodic flow matrix Q = ΠP

    phi = np.inf
    for mask in range(1, 2**n - 1):  # exclude empty set and full set
        A  = [i for i in range(n) if     mask & (1 << i)]
        Ac = [i for i in range(n) if not mask & (1 << i)]
        flow  = Q[np.ix_(A, Ac)].sum()
        pi_A  = pi[A].sum()
        phi   = min(phi, flow / (pi_A * (1.0 - pi_A)))

    return float(phi)

def stackelberg(P: np.ndarray, tau: np.ndarray) -> float:
    """Compute the Stackelberg game metric.

    tau[j] is the number of steps the attacker requires to complete an attack at node j.
    Accumulates the capture probability matrix Psi column-wise using the Fk recursion.
    """
    _check_stochastic(P)
    n = P.shape[0]
    tau = np.asarray(tau, dtype=int)
    if tau.shape != (n,):
        raise ValueError(f"tau must have length n={n}, got shape {tau.shape}")

    Psi = np.zeros((n, n))
    Fk = P.copy()  # F_1 = P
    tau_max = int(tau.max())
    for k in range(1, tau_max + 1):
        Psi[:, tau >= k] += Fk[:, tau >= k]
        if k < tau_max:
            Fk = P @ (Fk - np.diag(np.diag(Fk)))
    return float(Psi.min())

def lifted_stackelberg(P: np.ndarray, V: np.ndarray, tau: np.ndarray) -> float:
    """Compute the lifted Stackelberg game metric.

    tau[j] is the number of steps the attacker requires to complete an attack at physical node j.
    Uses F_1^lift = PV and the first-passage recursion F_{k+1}^lift = P(F_k^lift * (1-V)),
    where (1-V) masks out virtual states already in the target set between steps.
    """
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    n, m = V.shape
    tau = np.asarray(tau, dtype=int)
    if tau.shape != (m,):
        raise ValueError(f"tau must have length m={m}, got shape {tau.shape}")

    Psi_lift = np.zeros((n, m))
    Fk_lift = P @ V          # F_1^lift = PV
    V_comp  = 1 - V          # zero out states already in the target set
    tau_max = int(tau.max())
    for k in range(1, tau_max + 1):
        Psi_lift[:, tau >= k] += Fk_lift[:, tau >= k]
        if k < tau_max:
            Fk_lift = P @ (Fk_lift * V_comp)
    return float(Psi_lift.min())

def return_time_entropy(P: np.ndarray, eta: float = 0.01) -> float:
    """Compute the truncated Return-Time Entropy.

    Only diagonal entries Fk(i,i) contribute — these are the first-return probabilities.
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

def collapsing(P: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the lifted transition matrix P to the physical space using the mapping matrix V.
    Return the collapsed transition matrix Pbar and the stationary distribution pi_bar of the collapsed MC."""
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    pi = stationary_distribution(P)
    Pi = np.diag(pi)
    return np.linalg.solve(V.T @ Pi @ V, V.T @ Pi @ P @ V), V.T @ pi

if __name__ == "__main__":
    Pbar = np.array([[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]])
    print("Stationary distribution:", stationary_distribution(Pbar))
    print("Kemeny's constant:", kemeny(Pbar))

    p = 1.0
    P = np.array([
        [0,   p,   0,   0,   0,   1-p],
        [1-p, 0,   p,   0,   0,   0  ],
        [0,   1-p, 0,   p,   0,   0  ],
        [0,   0,   1-p, 0,   p,   0  ],
        [0,   0,   0,   1-p, 0,   p  ],
        [p,   0,   0,   0,   1-p, 0  ],
    ])
    V = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ])
    print("Stationary distribution:", stationary_distribution(P))
    print("Kemeny's constant:", kemeny(P))
    print("Lifted Kemeny's constant:", lifted_kemeny(P, V))