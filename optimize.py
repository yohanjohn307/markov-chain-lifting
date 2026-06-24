import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp

from markov import _check_mapping, _check_ergodic_flow


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
