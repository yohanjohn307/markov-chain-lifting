import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

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
    if not np.allclose(row_sums, Q.sum(axis=0), atol=1e-4):
        raise ValueError("ergodic flow matrix must satisfy Q 1 = Q^T 1 (equal row and column sums)")


# ---------------------------------------------------------------------------
# Markov chain utilities
# ---------------------------------------------------------------------------

def stationary_distribution(Pbar: np.ndarray) -> np.ndarray:
    """Solve for the unique stationary distribution pi of an irreducible MC."""
    _check_stochastic(Pbar)
    n = Pbar.shape[0]
    A = Pbar.T - np.eye(n) + np.ones((n, n))
    return np.linalg.solve(A, np.ones(n))


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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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
    """Compute the lifted Kemeny constant.

    V is the n x m mapping matrix whose j-th column indicates which virtual states
    belong to physical node j.
    """
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    n, m = V.shape
    pi = stationary_distribution(P)
    M_lift = np.zeros((n, m))
    for j in range(m):
        D_j = np.diag(V[:, j])
        M_lift[:, j] = np.linalg.solve(np.eye(n) - P + P @ D_j, np.ones(n))
    return float(pi @ M_lift @ V.T @ pi)


def conductance(P: np.ndarray) -> float:
    """Compute the conductance.

    Enumerates all 2^n - 2 subsets, so only practical for small state spaces.
    """
    _check_stochastic(P)
    n = P.shape[0]
    pi = stationary_distribution(P)
    Q = np.diag(pi) @ P
    phi = np.inf
    for mask in range(1, 2**n - 1):
        A  = [i for i in range(n) if     mask & (1 << i)]
        Ac = [i for i in range(n) if not mask & (1 << i)]
        flow = Q[np.ix_(A, Ac)].sum()
        pi_A = pi[A].sum()
        phi  = min(phi, flow / (pi_A * (1.0 - pi_A)))
    return float(phi)


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


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def erdos_renyi_graph(m: int, p: float, seed: int | None = None) -> np.ndarray:
    """Sample a connected undirected Erdős-Rényi graph G(m, p).

    Re-samples until connected (required for irreducible Markov chains).
    Returns A: (m, m) symmetric binary adjacency matrix with ones on the diagonal.
    """
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        U = (rng.random((m, m)) < p).astype(float)
        A = np.triu(U, k=1)
        A = A + A.T
        np.fill_diagonal(A, 1)
        visited = {0}
        queue = [0]
        while queue:
            v = queue.pop()
            for u in np.where(A[v] > 0)[0]:
                if u not in visited:
                    visited.add(u)
                    queue.append(u)
        if len(visited) == m:
            return A
    raise RuntimeError(f"Could not generate a connected G({m}, {p}) after 10000 tries; increase p")


def random_chain(A: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Generate a random irreducible Markov chain on a graph.

    Each row is sampled from a Dirichlet(1) distribution over the neighbours in A.
    Returns P: (m, m) row-stochastic matrix with P[i, j] = 0 when A[i, j] = 0.
    """
    rng = np.random.default_rng(seed)
    m = A.shape[0]
    P = np.zeros((m, m))
    for i in range(m):
        nbrs = np.where(A[i] > 0)[0]
        weights = rng.exponential(1.0, size=len(nbrs))
        P[i, nbrs] = weights / weights.sum()
    return P


def degree_lifting(A: np.ndarray) -> np.ndarray:
    """Build the degree-lifting mapping matrix for a graph.

    Node j gets deg(j) virtual states (one per incident edge), giving n = 2|E| total.
    Virtual states are ordered by physical node: the first deg(0) rows map to node 0, etc.
    Returns V: (n, m) binary mapping matrix with exactly one 1 per row.
    """
    deg = (A - np.diag(np.diag(A))).sum(axis=1).astype(int)
    n = int(deg.sum())
    m = A.shape[0]
    V = np.zeros((n, m), dtype=float)
    idx = 0
    for j, d in enumerate(deg):
        V[idx:idx + d, j] = 1.0
        idx += d
    return V


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def project_Q(Q_tilde: np.ndarray, Q_bar: np.ndarray, V: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Project Q_tilde onto the feasible set of liftings of Q_bar (Eq. 33).

    Solves: min ||Q - Q_tilde||_F  subject to  Q 1_n = Q^T 1_n,
    epsilon * V ceil(Q_bar) V^T <= Q <= V ceil(Q_bar) V^T,  V^T Q V = Q_bar.
    """
    _check_ergodic_flow(Q_bar, m=V.shape[1])
    _check_mapping(V)
    n = V.shape[0]
    U = V @ np.ceil(Q_bar) @ V.T
    Q = cp.Variable((n, n))
    constraints = [
        Q @ np.ones(n) == Q.T @ np.ones(n),
        Q >= epsilon * U,
        Q <= U,
        V.T @ Q @ V == Q_bar,
    ]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Q - Q_tilde)), constraints)
    prob.solve()
    if Q.value is None:
        raise RuntimeError("project_Q: QP did not converge; check that Q_bar is a valid ergodic flow for V")
    return np.maximum(Q.value, 0.0)


def project_Q_bar(
    Q_tilde: np.ndarray,
    Q_bar_ref: np.ndarray,
    pi_bar: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Project Q_tilde onto feasible ergodic flows with fixed stationary distribution.

    Solves: min ||Q - Q_tilde||_F  subject to  Q 1_m = pi_bar,  Q^T 1_m = pi_bar,
    epsilon * A <= Q <= A,  where A = ceil(Q_bar_ref) is the binary graph adjacency.
    """
    m = Q_tilde.shape[0]
    A = np.ceil(Q_bar_ref)
    Q = cp.Variable((m, m))
    constraints = [
        Q @ np.ones(m) == pi_bar,
        Q.T @ np.ones(m) == pi_bar,
        Q >= epsilon * A,
        Q <= A,
    ]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Q - Q_tilde)), constraints)
    prob.solve()
    if Q.value is None:
        raise RuntimeError("project_Q_bar: QP did not converge; check that pi_bar is consistent with Q_bar_ref")
    return np.maximum(Q.value, 0.0)


def _grad_kemeny(Q_bar: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute the Kemeny constant and its gradient w.r.t. Q_bar via the adjoint method."""
    m = Q_bar.shape[0]
    pi_bar = Q_bar.sum(axis=1)
    Pi_bar = np.diag(pi_bar)
    M = 0.0
    grad = np.zeros((m, m))
    for j in range(m):
        gamma_j = np.ones(m)
        gamma_j[j] = 0
        Gamma_j = np.diag(gamma_j)
        m_j     = np.linalg.solve(Pi_bar - Q_bar @ Gamma_j, pi_bar)
        lam_j   = -pi_bar[j] * np.linalg.solve(Pi_bar - Gamma_j @ Q_bar.T, pi_bar)
        M    += pi_bar[j] * (pi_bar @ m_j)
        grad -= np.outer(lam_j, m_j) @ Gamma_j
    return M, grad


def _grad_lifted_kemeny(Q: np.ndarray, V: np.ndarray, pi_bar: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute the lifted Kemeny constant and its gradient w.r.t. Q via the adjoint method."""
    n, m = V.shape
    pi   = Q.sum(axis=1)
    Pi   = np.diag(pi)
    In   = np.eye(n)
    M_val = 0.0
    grad  = np.zeros((n, n))
    for j in range(m):
        D_j   = np.diag(V[:, j])
        I_Dj  = In - D_j
        pib_j = float(pi_bar[j])
        m_Ej  = np.linalg.solve(Pi - Q @ I_Dj, pi)              # forward solve, Eq. 21
        lam_j = np.linalg.solve(Pi - I_Dj @ Q.T, -pib_j * pi)  # adjoint solve, Eq. 35
        u_j   = (pib_j * In + np.diag(lam_j)) @ m_Ej - lam_j
        M_val += pib_j * float(pi @ m_Ej)
        grad  += np.outer(u_j, np.ones(n)) - np.outer(lam_j, m_Ej) @ I_Dj  # Eq. 34
    return M_val, grad


def _lifted_stackelberg_Q(Q, V, tau):
    """JAX-traced lifted Stackelberg metric as a function of ergodic flow Q."""
    pi = Q.sum(axis=1)
    P = Q / pi[:, None]
    n, m = V.shape
    tau_max = int(tau.max())
    Psi_lift = jnp.zeros((n, m))
    Fk_lift = P @ V
    V_comp = 1.0 - V
    for k in range(1, tau_max + 1):
        col_mask = jnp.array(tau >= k)[None, :].astype(Q.dtype)
        Psi_lift = Psi_lift + Fk_lift * col_mask
        if k < tau_max:
            Fk_lift = P @ (Fk_lift * V_comp)
    return jnp.min(Psi_lift)


def _grad_lifted_stackelberg(Q: np.ndarray, V: np.ndarray, tau: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute lifted Stackelberg value and gradient w.r.t. Q via JAX autodiff.

    To maximize J^lift, negate: lambda Q: tuple(-x for x in _grad_lifted_stackelberg(Q, V, tau))
    """
    if jax is None:
        raise ImportError("JAX is required for this gradient; install with: pip install jax")
    tau = np.asarray(tau, dtype=int)
    val, grad = jax.value_and_grad(lambda Q: _lifted_stackelberg_Q(Q, V, tau))(jnp.array(Q))
    return float(val), np.array(grad)


def _rte_Q_bar(Q_bar, pi_bar, K_eta: int):
    """JAX-traced truncated RTE as a function of ergodic flow Q_bar."""
    P_bar = Q_bar / pi_bar[:, None]
    H = jnp.zeros(())
    Fk = P_bar
    for k in range(1, K_eta + 1):
        d = jnp.diag(Fk)
        safe_d = jnp.where(d > 0, d, 1.0)  # avoid log(0) in forward and backward passes
        H = H - jnp.sum(jnp.where(d > 0, pi_bar * d * jnp.log(safe_d), 0.0))
        if k < K_eta:
            Fk = P_bar @ (Fk - jnp.diag(d))
    return H


def _grad_rte(Q_bar: np.ndarray, pi_bar: np.ndarray, eta: float) -> tuple[float, np.ndarray]:
    """Compute truncated RTE value and gradient w.r.t. Q_bar via JAX autodiff.

    pi_bar is fixed by the optimization constraint.
    To maximize H, negate: lambda Q: tuple(-x for x in _grad_rte(Q, pi_bar, eta))
    """
    if jax is None:
        raise ImportError("JAX is required for this gradient; install with: pip install jax")
    K_eta = int(np.ceil(1.0 / (eta * float(pi_bar.min())))) - 1
    val, grad = jax.value_and_grad(lambda Q: _rte_Q_bar(Q, jnp.array(pi_bar), K_eta))(jnp.array(Q_bar))
    return float(val), np.array(grad)


def _lifted_rte_Q(Q, V, pi_bar, K_eta: int):
    """JAX-traced truncated lifted RTE as a function of ergodic flow Q."""
    pi = Q.sum(axis=1)
    P = Q / pi[:, None]
    V_comp = 1.0 - V
    H = jnp.zeros(())
    Fk_lift = P @ V
    for k in range(1, K_eta + 1):
        Rk = (V * pi[:, None] * Fk_lift).sum(axis=0) / pi_bar
        safe_Rk = jnp.where(Rk > 0, Rk, 1.0)  # avoid log(0) in forward and backward passes
        H = H - jnp.sum(jnp.where(Rk > 0, pi_bar * Rk * jnp.log(safe_Rk), 0.0))
        if k < K_eta:
            Fk_lift = P @ (Fk_lift * V_comp)
    return H


def _grad_lifted_rte(Q: np.ndarray, V: np.ndarray, pi_bar: np.ndarray, eta: float) -> tuple[float, np.ndarray]:
    """Compute truncated lifted RTE value and gradient w.r.t. Q via JAX autodiff.

    pi_bar is fixed by the optimization constraint.
    To maximize H^lift, negate: lambda Q: tuple(-x for x in _grad_lifted_rte(Q, V, pi_bar, eta))
    """
    if jax is None:
        raise ImportError("JAX is required for this gradient; install with: pip install jax")
    K_eta = int(np.ceil(1.0 / (eta * float(pi_bar.min())))) - 1
    val, grad = jax.value_and_grad(lambda Q: _lifted_rte_Q(Q, V, pi_bar, K_eta))(jnp.array(Q))
    return float(val), np.array(grad)


def projected_gradient_descent(
    Q0: np.ndarray,
    grad_fn,
    project_fn,
    alpha: float = 1e-2,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[np.ndarray, list[float]]:
    """Minimize an objective via projected gradient descent.

    Each iteration:
      Q_tilde  = Q_k - alpha * grad_fn(Q_k)[1]   (gradient step)
      Q_{k+1} = project_fn(Q_tilde)               (projection step)

    grad_fn and project_fn are closures capturing all fixed parameters. Examples:

      Lifted Kemeny (minimize):
        pi_bar     = Q_bar.sum(axis=1)
        grad_fn    = lambda Q: _grad_lifted_kemeny(Q, V, pi_bar)
        project_fn = lambda Q: project_Q(Q, Q_bar, V)

      Physical Kemeny (minimize):
        pi_bar     = Q0_bar.sum(axis=1)
        grad_fn    = lambda Q: _grad_kemeny(Q)
        project_fn = lambda Q: project_Q_bar(Q, Q0_bar, pi_bar)

      Lifted RTE (maximize — negate grad_fn):
        pi_bar     = Q_bar.sum(axis=1)
        grad_fn    = lambda Q: tuple(-x for x in _grad_lifted_rte(Q, V, pi_bar, eta))
        project_fn = lambda Q: project_Q(Q, Q_bar, V)

    Returns (final Q, history of objective values).
    """
    Q = Q0.copy()
    history: list[float] = []
    for _ in range(n_iter):
        val, grad = grad_fn(Q)
        history.append(val)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break
        Q = project_fn(Q - alpha * grad)
    return Q, history


if __name__ == "__main__":
    Pbar = np.array([[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]])
    print("Stationary distribution:", stationary_distribution(Pbar))
    print("Kemeny's constant:", kemeny(Pbar))

    p = 0.9
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

    P_collapsed, pi_collapsed = collapsing(P, V)
    print("P collapsed:", P_collapsed)
    print("pi collapsed:", pi_collapsed)

    tau = np.array([3, 3, 3, 3])
    print("Stackelberg metric:", stackelberg(Pbar, tau))
    print("Lifted Stackelberg metric:", lifted_stackelberg(P, V, tau))

    print("Return-time entropy:", return_time_entropy(Pbar, eta=0.01))
    print("Lifted return-time entropy:", lifted_return_time_entropy(P, V, eta=0.01))

    A = erdos_renyi_graph(5, 0.1, seed=0)
    print("Erdős-Rényi graph adjacency matrix:\n", A)
    P_random = random_chain(A, seed=0)
    _check_stochastic(P_random)
    print("Random irreducible Markov chain on the graph:\n", P_random)
    V_degree = degree_lifting(A)
    print("Degree-lifting mapping matrix:\n", V_degree)
