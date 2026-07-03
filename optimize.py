import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from markov import _check_mapping, _check_ergodic_flow


def project_Q(Q_tilde: np.ndarray, Q_bar: np.ndarray, V: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Project Q_tilde onto the feasible set of liftings of Q_bar (Eq. 35).

    Solves: min ||Q - Q_tilde||_F  subject to  Q 1_n = Q^T 1_n,
    epsilon * V ceil(Q_bar) V^T <= Q <= V ceil(Q_bar) V^T,  V^T Q V = Q_bar.
    """
    _check_ergodic_flow(Q_bar, m=V.shape[1])
    _check_mapping(V)
    n = V.shape[0]
    Q_bar_ceil = (Q_bar > epsilon).astype(int)
    U = V @ Q_bar_ceil @ V.T
    Q = cp.Variable((n, n))
    constraints = [
        Q @ np.ones(n) == Q.T @ np.ones(n),
        Q >= epsilon * U,
        Q <= U,
        V.T @ Q @ V == Q_bar,
    ]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Q - Q_tilde)), constraints)
    prob.solve(solver=cp.OSQP)
    if Q.value is None:
        raise RuntimeError("project_Q: QP did not converge; check that Q_bar is a valid ergodic flow for V")
    return np.maximum(Q.value, 0.0)


def make_project_Q(
    Q_bar: np.ndarray,
    V: np.ndarray,
    epsilon: float = 1e-6,
):
    """Return a cached projection function for the lifted feasible set.

    Builds the CVXPY problem once with cp.Parameter; each call updates the
    parameter and warm-starts the solver, eliminating per-call canonicalization.
    """
    _check_ergodic_flow(Q_bar, m=V.shape[1])
    _check_mapping(V)
    n = V.shape[0]
    Q_bar_ceil = (Q_bar > 1e-6).astype(int)
    U = V @ Q_bar_ceil @ V.T
    Q_tilde_p = cp.Parameter((n, n))
    Q = cp.Variable((n, n))
    constraints = [
        Q @ np.ones(n) == Q.T @ np.ones(n),
        Q >= epsilon * U,
        Q <= U,
        V.T @ Q @ V == Q_bar,
    ]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Q - Q_tilde_p)), constraints)

    def project(Q_tilde: np.ndarray) -> np.ndarray:
        Q_tilde_p.value = Q_tilde
        prob.solve(solver=cp.OSQP, warm_start=True)
        if Q.value is None:
            raise RuntimeError("project_Q: QP did not converge; check that Q_bar is a valid ergodic flow for V")
        return np.maximum(Q.value, 0.0)

    return project


def project_Q_bar(
    Q_tilde: np.ndarray,
    A: np.ndarray,
    pi_bar: np.ndarray | None = None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Project Q_tilde onto feasible ergodic flows for the physical space.

    Solves: min ||Q - Q_tilde||_F  subject to  graph-compatibility and either:
      - pi_bar provided: Q 1_m = pi_bar, Q^T 1_m = pi_bar  (fixed stationary distribution,
        used for Kemeny and RTE)
      - pi_bar is None:  Q 1_m = Q^T 1_m  (free stationary distribution,
        used for Stackelberg where irreducibility is enforced by the metric)
    In both cases: epsilon * A <= Q <= A, where A is the graph adjacency matrix.
    """
    m = Q_tilde.shape[0]
    Q = cp.Variable((m, m))
    ones = np.ones(m)
    if pi_bar is not None:
        stationarity = [Q @ ones == pi_bar, Q.T @ ones == pi_bar]
    else:
        stationarity = [Q @ ones == Q.T @ ones]
    constraints = stationarity + [Q >= epsilon * A, Q <= A]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Q - Q_tilde)), constraints)
    prob.solve(solver=cp.OSQP)
    if Q.value is None:
        raise RuntimeError("project_Q_bar: QP did not converge; check inputs are consistent")
    return np.maximum(Q.value, 0.0)


def make_project_Q_bar(
    A: np.ndarray,
    pi_bar: np.ndarray | None = None,
    epsilon: float = 1e-6,
):
    """Return a cached projection function for the physical feasible set.

    Builds the CVXPY problem once with cp.Parameter; each call updates the
    parameter and warm-starts the solver, eliminating per-call canonicalization.
    adj is the graph adjacency matrix.
    """
    m = A.shape[0]
    Q_tilde_p = cp.Parameter((m, m))
    Q = cp.Variable((m, m))
    ones = np.ones(m)
    if pi_bar is not None:
        stationarity = [Q @ ones == pi_bar, Q.T @ ones == pi_bar]
    else:
        stationarity = [Q @ ones == Q.T @ ones]
    constraints = stationarity + [Q >= epsilon * A, Q <= A]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Q - Q_tilde_p)), constraints)

    def project(Q_tilde: np.ndarray) -> np.ndarray:
        Q_tilde_p.value = Q_tilde
        prob.solve(solver=cp.OSQP, warm_start=True)
        if Q.value is None:
            raise RuntimeError("project_Q_bar: QP did not converge; check inputs are consistent")
        return np.maximum(Q.value, 0.0)

    return project


def _grad_kemeny(Q_bar: np.ndarray, W: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Compute the Kemeny constant and its gradient w.r.t. Q_bar via the adjoint method.

    W is the edge weight matrix (Sec. VII); if None, all weights default to 1.
    """
    m = Q_bar.shape[0]
    if W is None:
        W = np.ones((m, m))
    pi_bar = Q_bar.sum(axis=1)
    Pi_bar = np.diag(pi_bar)
    rhs = (Q_bar * W).sum(axis=1)
    K_val = 0.0
    grad = np.zeros((m, m))
    for j in range(m):
        gamma_j = np.ones(m)
        gamma_j[j] = 0
        Gamma_j = np.diag(gamma_j)
        m_j     = np.linalg.solve(Pi_bar - Q_bar @ Gamma_j, rhs)
        lam_j   = -pi_bar[j] * np.linalg.solve(Pi_bar - Gamma_j @ Q_bar.T, pi_bar)
        K_val += pi_bar[j] * (pi_bar @ m_j)
        grad -= np.outer(lam_j, m_j) @ Gamma_j + np.outer(lam_j, np.ones(m)) * W
    return K_val, grad


def _grad_lifted_kemeny(Q: np.ndarray, V: np.ndarray, pi_bar: np.ndarray, W: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Compute the lifted Kemeny constant and its gradient w.r.t. Q via the adjoint method.

    W is the edge weight matrix (Sec. VII); if None, all weights default to 1.
    """
    n, m = V.shape
    if W is None:
        W = np.ones((n, n))
    pi   = Q.sum(axis=1)
    Pi   = np.diag(pi)
    In   = np.eye(n)
    rhs  = (Q * W).sum(axis=1)
    K_lift_val = 0.0
    grad  = np.zeros((n, n))
    for j in range(m):
        D_j   = np.diag(V[:, j])
        I_Dj  = In - D_j
        pib_j = float(pi_bar[j])
        m_Sj  = np.linalg.solve(Pi - Q @ I_Dj, rhs)             # forward solve, Eq. (21)
        lam_j = np.linalg.solve(Pi - I_Dj @ Q.T, -pib_j * pi)  # adjoint solve, Eq. (37)
        u_j   = (pib_j * In + np.diag(lam_j)) @ m_Sj
        K_lift_val += pib_j * float(pi @ m_Sj)
        grad  += np.outer(u_j, np.ones(n)) - np.outer(lam_j, np.ones(n)) * W - np.outer(lam_j, m_Sj) @ I_Dj  # Eq. (36)
    return K_lift_val, grad


def _lifted_stackelberg_Q(Q, V, tau, temp):
    """JAX-traced lifted Stackelberg metric as a function of ergodic flow Q.

    The true metric is min(Psi_lift), but jnp.min is flat (zero gradient) almost
    everywhere and only subgradient-informative at the (possibly non-unique) argmin.
    We instead use the softmin -temp * logsumexp(-Psi_lift / temp), which is smooth
    and gives every entry a gradient weighted by its closeness to the minimum.
    As temp -> 0, softmin(Psi_lift) -> min(Psi_lift) (it is always a lower bound,
    with gap <= temp * log(n * m)).
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
    return -temp * logsumexp(-Psi_lift.reshape(-1) / temp)


def make_grad_lifted_stackelberg(V: np.ndarray, tau: np.ndarray):
    """Return a JIT-compiled gradient function for the (softmin) lifted Stackelberg metric w.r.t. Q.

    The returned grad_fn(Q, temp) takes temp as a traced argument rather than a
    Python constant baked in at trace time, so it can be annealed across PGD
    iterations (e.g. via projected_gradient_descent's temp_schedule) without
    triggering recompilation: smaller temp tracks the true min(Psi_lift) more
    closely but concentrates gradients on the near-minimal entries; larger temp
    gives smoother, more informative gradients early in optimization at the cost
    of a looser approximation to the true metric.
    Call once before the optimization loop; the returned callable reuses the compiled
    kernel on every iteration without retracing.
    To maximize J^lift, negate the returned function's outputs.
    """
    tau = np.asarray(tau, dtype=int)
    V_j = jnp.array(V)
    _compiled = jax.jit(
        jax.value_and_grad(lambda Q, temp: _lifted_stackelberg_Q(Q, V_j, tau, temp), argnums=0)
    )

    def grad_fn(Q: np.ndarray, temp: float) -> tuple[float, np.ndarray]:
        val, grad = _compiled(jnp.array(Q), jnp.asarray(temp, dtype=jnp.array(Q).dtype))
        return float(val), np.array(grad)

    return grad_fn


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


def make_grad_lifted_rte(V: np.ndarray, pi_bar: np.ndarray, eta: float):
    """Return a JIT-compiled gradient function for the truncated lifted RTE w.r.t. Q.

    Call once before the optimization loop; the returned callable reuses the compiled
    kernel on every iteration without retracing.
    V and pi_bar are held fixed (baked into the compiled kernel).
    To maximize H^lift, negate the returned function's outputs.
    """
    K_eta = int(np.ceil(1.0 / (eta * float(pi_bar.min())))) - 1
    V_j = jnp.array(V)
    pi_bar_j = jnp.array(pi_bar)
    _compiled = jax.jit(jax.value_and_grad(lambda Q: _lifted_rte_Q(Q, V_j, pi_bar_j, K_eta)))

    def grad_fn(Q: np.ndarray) -> tuple[float, np.ndarray]:
        val, grad = _compiled(jnp.array(Q))
        return float(val), np.array(grad)

    return grad_fn


def projected_gradient_descent(
    Q0: np.ndarray,
    grad_fn,
    project_fn,
    alpha: float = 1e-2,
    n_iter: int = 100,
    tol: float = 1e-6,
    temp_schedule=None,
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
        project_fn = lambda Q: project_Q_bar(Q, adj, pi_bar)

      Lifted RTE (maximize — negate grad_fn):
        pi_bar     = Q_bar.sum(axis=1)
        grad_fn    = lambda Q: tuple(-x for x in _grad_lifted_rte(Q, V, pi_bar, eta))
        project_fn = lambda Q: project_Q(Q, Q_bar, V)

    temp_schedule is optional and only applies to grad_fn's built from
    make_grad_lifted_stackelberg, whose grad_fn(Q, temp) takes the softmin
    temperature as a second, traced (non-retracing) argument. When temp_schedule
    is given, grad_fn is called as grad_fn(Q, temp_schedule(k)) at iteration k
    instead of grad_fn(Q), letting temp anneal (e.g. exponential decay) across
    the PGD loop while reusing the single compiled JIT kernel throughout. Example
    (maximize J^lift, so negate; temp still passed positionally through the
    lambda):

      grad_stackelberg = make_grad_lifted_stackelberg(V, tau)
      grad_fn = lambda Q, temp: tuple(-x for x in grad_stackelberg(Q, temp))
      Q_opt, hist = projected_gradient_descent(
          Q0, grad_fn, project_fn, alpha, n_iter, tol,
          temp_schedule=lambda k: temp0 * decay ** k,
      )

    Returns (final Q, history of objective values).
    """
    Q = Q0.copy()
    history: list[float] = []
    for k in range(n_iter):
        val, grad = grad_fn(Q, temp_schedule(k)) if temp_schedule is not None else grad_fn(Q)
        history.append(val)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break
        Q_tilde = Q - alpha * grad
        Q = project_fn(Q_tilde)

    return Q, history
