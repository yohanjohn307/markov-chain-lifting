
import numpy as np
import cvxpy as cp

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None   # type: ignore[assignment]
    jnp = None   # type: ignore[assignment]

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

def lifted_return_time_entropy(P: np.ndarray, V: np.ndarray, eta: float = 0.01) -> float:
    """Compute the truncated lifted Return-Time Entropy:
        H^lift(P) = -sum_j pi_bar_j * sum_{k=1}^{K_eta} R_k(j) log R_k(j)
    where K_eta = ceil(1 / (eta * pi_bar_min)) - 1.
    """
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    pi = stationary_distribution(P)
    pi_bar = V.T @ pi
    K_eta = int(np.ceil(1.0 / (eta * pi_bar.min()))) - 1
    V_comp = 1.0 - V  # (1_n 1_m^T - V): zeros out target-set entries between steps

    H = 0.0
    Fk_lift = P @ V  # F_1^lift = PV
    for k in range(1, K_eta + 1):
        # R_k(j) = (1/pi_bar_j) * sum_{i in E_j} pi_i * F^lift_k(i, j)
        Rk = (V * pi[:, np.newaxis] * Fk_lift).sum(axis=0) / pi_bar
        mask = Rk > 0
        H -= float(np.sum(pi_bar[mask] * Rk[mask] * np.log(Rk[mask])))
        if k < K_eta:
            Fk_lift = P @ (Fk_lift * V_comp)
    return H

def erdos_renyi_graph(m: int, p: float, seed: int | None = None) -> np.ndarray:
    """Sample a connected undirected Erdős-Rényi graph G(m, p).

    Re-samples until the graph is connected (required for irreducible Markov chains).

    Args:
        m:    number of nodes
        p:    edge probability
        seed: random seed

    Returns:
        A: (m, m) symmetric binary adjacency matrix with ones on the diagonal (self-loops)
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

    Each row of P is sampled from a symmetric Dirichlet(1) distribution over the
    neighbours indicated by A, so every compatible transition matrix is reachable.

    Args:
        A:    (m, m) symmetric binary adjacency matrix (from erdos_renyi_graph)
        seed: random seed

    Returns:
        P: (m, m) row-stochastic transition matrix with P[i, j] = 0 when A[i, j] = 0
    """
    rng = np.random.default_rng(seed)
    m = A.shape[0]
    P = np.zeros((m, m))
    for i in range(m):
        nbrs = np.where(A[i] > 0)[0]
        weights = rng.exponential(1.0, size=len(nbrs))   # Dirichlet(1,...,1) via normalised Exponentials
        P[i, nbrs] = weights / weights.sum()
    return P


def degree_lifting(A: np.ndarray) -> np.ndarray:
    """Build the degree-lifting mapping matrix for a graph.

    Node j gets exactly deg(j) virtual states — one per incident edge — so the
    total number of virtual states equals the number of edge-endpoints, i.e.,
    n = sum_j deg(j) = 2 |E|.  Virtual states are ordered by physical node:
    the first deg(0) rows of V correspond to node 0, the next deg(1) to node 1, etc.

    Args:
        A: (m, m) symmetric binary adjacency matrix

    Returns:
        V: (n, m) binary mapping matrix with exactly one 1 per row
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

def project_Q(Q_tilde: np.ndarray, Q_bar: np.ndarray, V: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Project Q_tilde onto the feasible set of liftings of Q_bar (Eq. 33).

    Solves: min ||Q - Q_tilde||_F  subject to  Q 1_n = Q^T 1_n (balanced flow),
    epsilon * V ceil(Q_bar) V^T <= Q <= V ceil(Q_bar) V^T (graph-compatible bounds),
    and V^T Q V = Q_bar (valid lifting of the physical ergodic flow).

    Q_tilde is the n x n gradient-descent iterate; Q_bar is the m x m physical ergodic
    flow matrix (Q_bar = diag(pi_bar) P_bar); V is the n x m binary mapping matrix.
    """
    _check_ergodic_flow(Q_bar, m=V.shape[1])
    _check_mapping(V)
    n = V.shape[0]
    U = V @ np.ceil(Q_bar) @ V.T  # binary upper-bound matrix (graph structure)

    Q = cp.Variable((n, n))
    constraints = [
        Q @ np.ones(n) == Q.T @ np.ones(n),   # row sums == column sums
        Q >= epsilon * U,                        # reducibility: all edges stay positive
        Q <= U,                                  # graph-compatibility: no new edges
        V.T @ Q @ V == Q_bar,                   # valid lifting (Eq. 13)
    ]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Q - Q_tilde)), constraints)
    prob.solve()
    if Q.value is None:
        raise RuntimeError("project_Q: QP did not converge; check that Q_bar is a valid ergodic flow for V")
    return np.maximum(Q.value, 0.0)  # clip sub-zero noise at non-edge entries

def _grad_kemeny(Qbar: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute M and its gradient w.r.t. Qbar via the adjoint method
    """
    m = Qbar.shape[0]
    pi_bar = Qbar.sum(axis=1)
    Pi_bar = np.diag(pi_bar)
    grad = np.zeros((m, m))
    M = 0
    for j in range(m):
        gamma_j = np.ones(m)
        gamma_j[j] = 0
        Gamma_j = np.diag(gamma_j)
        m_j = np.linalg.solve(Pi_bar - Qbar @ Gamma_j, pi_bar)
        lambda_j = -pi_bar[j] * np.linalg.solve(Pi_bar - Gamma_j @ Qbar.T, pi_bar)
        M += pi_bar[j] * (pi_bar @ m_j)
        grad -= np.outer(lambda_j, m_j) @ Gamma_j
    return M, grad

def _grad_lifted_kemeny(Q: np.ndarray, V: np.ndarray, pi_bar: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute M^lift and its gradient w.r.t. Q via the adjoint method
    """
    n, m = V.shape
    pi     = Q.sum(axis=1)       # virtual stationary distribution pi = Q 1_n
    Pi     = np.diag(pi)
    In     = np.eye(n)

    M_val = 0.0
    grad  = np.zeros((n, n))
    for j in range(m):
        D_j   = np.diag(V[:, j])
        I_Dj  = In - D_j
        pib_j = float(pi_bar[j])

        # Forward: [Pi - Q(I - D_j)] m_Ej = pi
        m_Ej  = np.linalg.solve(Pi - Q @ I_Dj, pi)
        M_val += pib_j * float(pi @ m_Ej)

        # Adjoint: [Pi - (I - D_j) Q^T] lambda_j = -pi_bar_j * pi  (Eq. 35)
        lam_j = np.linalg.solve(Pi - I_Dj @ Q.T, -pib_j * pi)

        # Gradient contribution (Eq. 34)
        u_j   = (pib_j * In + np.diag(lam_j)) @ m_Ej - lam_j
        grad += np.outer(u_j, np.ones(n)) - np.outer(lam_j, m_Ej) @ I_Dj

    return M_val, grad

def _lifted_stackelberg_Q(Q: np.ndarray, V: np.ndarray, tau: np.ndarray):
    """JAX-compatible lifted Stackelberg metric as a function of ergodic flow Q.

    V and tau are treated as static constants; only Q is traced by jax.grad.
    """
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

    Returns (value, gradient).  To maximize J^lift, minimize the negation:
        grad_fn = lambda Q, V: tuple(-x for x in _grad_lifted_stackelberg(Q, V, tau))
    """
    if jax is None:
        raise ImportError("JAX is required for this gradient; install with: pip install jax")
    tau = np.asarray(tau, dtype=int)
    fn = lambda Q: _lifted_stackelberg_Q(Q, V, tau)
    val, grad = jax.value_and_grad(fn)(jnp.array(Q))
    return float(val), np.array(grad)


def _lifted_rte_Q(Q: np.ndarray, V: np.ndarray, pi_bar: np.ndarray, K_eta: int):
    """JAX-compatible truncated lifted RTE as a function of ergodic flow Q.

    pi_bar and K_eta are treated as static constants (fixed by the optimization constraint).
    Uses the jnp.where trick to avoid log(0) in the gradient.
    """
    pi = Q.sum(axis=1)
    P = Q / pi[:, None]
    V_comp = 1.0 - V
    H = jnp.zeros(())
    Fk_lift = P @ V
    for k in range(1, K_eta + 1):
        Rk = (V * pi[:, None] * Fk_lift).sum(axis=0) / pi_bar
        safe_Rk = jnp.where(Rk > 0, Rk, 1.0)          # avoid log(0) in both forward and backward pass
        H = H - jnp.sum(jnp.where(Rk > 0, pi_bar * Rk * jnp.log(safe_Rk), 0.0))
        if k < K_eta:
            Fk_lift = P @ (Fk_lift * V_comp)
    return H


def _grad_lifted_rte(Q: np.ndarray, V: np.ndarray, pi_bar: np.ndarray, eta: float) -> tuple[float, np.ndarray]:
    """Compute truncated lifted RTE value and gradient w.r.t. Q via JAX autodiff.

    pi_bar = Q_bar.sum(axis=1) is fixed by the optimization constraint; pass it in so
    K_eta is determined once from pi_bar rather than re-derived from Q on every call.

    Returns (value, gradient).  To maximize H^lift, minimize the negation:
        grad_fn = lambda Q, V: tuple(-x for x in _grad_lifted_rte(Q, V, pi_bar, eta))
    """
    if jax is None:
        raise ImportError("JAX is required for this gradient; install with: pip install jax")
    K_eta = int(np.ceil(1.0 / (eta * float(pi_bar.min())))) - 1
    fn = lambda Q: _lifted_rte_Q(Q, V, pi_bar, K_eta)
    val, grad = jax.value_and_grad(fn)(jnp.array(Q))
    return float(val), np.array(grad)


def projected_gradient_descent(
    Q0: np.ndarray,
    Q_bar: np.ndarray,
    V: np.ndarray,
    grad_fn,
    alpha: float = 1e-2,
    n_iter: int = 100,
    tol: float = 1e-6,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, list[float]]:
    """Minimize an objective over liftings of Q_bar via projected gradient descent (Eq. 32-33).

    Each iteration:
      Q_tilde  = Q_k - alpha * grad_fn(Q_k, V)[1]    (gradient step)
      Q_{k+1} = project_Q(Q_tilde, Q_bar, V, epsilon) (QP projection, Eq. 33)

    grad_fn(Q, V) -> (value, gradient) computes the objective and its gradient w.r.t. Q.
    pi_bar = Q_bar.sum(axis=1) is fixed by the constraint and should be captured in grad_fn:
        pi_bar = Q_bar.sum(axis=1)
        grad_fn = lambda Q, V: _grad_lifted_kemeny(Q, V, pi_bar)
        grad_fn = lambda Q, V: tuple(-x for x in _grad_lifted_stackelberg(Q, V, tau))
        grad_fn = lambda Q, V: tuple(-x for x in _grad_lifted_rte(Q, V, pi_bar, eta))
    Negate the output for metrics to be maximized (Stackelberg, RTE).

    Q0 must be a valid lifting of Q_bar.  Returns (final Q, history of objective values).
    """
    _check_ergodic_flow(Q_bar, m=V.shape[1])
    _check_mapping(V)
    Q = Q0.copy()

    history: list[float] = []
    for _ in range(n_iter):
        val, grad = grad_fn(Q, V)
        history.append(val)

        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break

        Q = project_Q(Q - alpha * grad, Q_bar, V, epsilon)

    return Q, history

def collapsing(P: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse the lifted transition matrix P to the physical space using the mapping matrix V.
    Return the collapsed transition matrix Pbar and the stationary distribution pi_bar of the collapsed MC."""
    _check_stochastic(P)
    _check_mapping(V, n=P.shape[0])
    pi = stationary_distribution(P)
    Pi = np.diag(pi)
    Pbar = np.linalg.solve(V.T @ Pi @ V, V.T @ Pi @ P @ V)
    pi_bar = V.T @ pi
    return Pbar, pi_bar

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