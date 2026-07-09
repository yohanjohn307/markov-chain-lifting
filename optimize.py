import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from markov import _check_mapping, _check_ergodic_flow, EQUALITY_ATOL


_SOLVER_TOL = EQUALITY_ATOL * 1e-2
_OK_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
_DEFAULT_MAX_ITER = 10_000


def _solve_projection(prob: cp.Problem, warm_start: bool, max_iter: int = _DEFAULT_MAX_ITER) -> None:
    """Solve a projection QP, falling back to a cold start if warm-started OSQP stalls."""
    prob.solve(solver=cp.OSQP, warm_start=warm_start, eps_abs=_SOLVER_TOL, eps_rel=_SOLVER_TOL, max_iter=max_iter)
    if warm_start and prob.status not in _OK_STATUSES:
        prob.solve(solver=cp.OSQP, warm_start=False, eps_abs=_SOLVER_TOL, eps_rel=_SOLVER_TOL, max_iter=max_iter)


def _clip_to_support(Q_value: np.ndarray, support: np.ndarray, epsilon: float) -> np.ndarray:
    """Clip a solved projection to its own epsilon*support <= Q <= support constraint."""
    return np.where(support > 0, np.maximum(Q_value, epsilon), 0.0)


def make_project_Q(
    Q_bar: np.ndarray,
    V: np.ndarray,
    epsilon: float = 1e-6,
    max_iter: int = _DEFAULT_MAX_ITER,
):
    """Return a cached projection function for the lifted feasible set.

    Builds the CVXPY problem once with cp.Parameter; each call updates the
    parameter and warm-starts the solver, eliminating per-call canonicalization.
    """
    _check_ergodic_flow(Q_bar, m=V.shape[1])
    _check_mapping(V)
    n = V.shape[0]
    Q_bar_ceil = (Q_bar > 1e-6).astype(int) # hardcoded to catch Stackelberg case where epsilon = 0
    U = V @ Q_bar_ceil @ V.T
    Q_tilde_p = cp.Parameter((n, n))
    Q = cp.Variable((n, n))
    constraints = [
        Q @ np.ones(n) == Q.T @ np.ones(n),
        Q >= epsilon * U,
        Q <= U,
        # V.T @ Q @ V == Q_bar,
        # Exact equality is infeasible whenever Q_bar's own row/col sums differ by
        # more than machine precision (e.g. Q_bar came from another PGD projection,
        # which only converges to _SOLVER_TOL): combined with Q @ 1 == Q^T @ 1 above,
        # exact equality here would force Q_bar to have exactly equal row/col sums,
        # which _check_ergodic_flow only guarantees to EQUALITY_ATOL. Relaxing to a
        # band of that width keeps the constraint matrix full rank so OSQP converges.
        cp.abs(V.T @ Q @ V - Q_bar) <= EQUALITY_ATOL,
    ]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Q - Q_tilde_p)), constraints)

    def project(Q_tilde: np.ndarray) -> np.ndarray:
        Q_tilde_p.value = Q_tilde
        _solve_projection(prob, warm_start=True, max_iter=max_iter)
        if Q.value is None or prob.status not in _OK_STATUSES:
            raise RuntimeError(f"project_Q: QP did not converge (status={prob.status}); check that Q_bar is a valid ergodic flow for V")
        return _clip_to_support(Q.value, U, epsilon)

    return project


def make_project_Q_bar(
    A: np.ndarray,
    pi_bar: np.ndarray | None = None,
    epsilon: float = 1e-6,
    max_iter: int = _DEFAULT_MAX_ITER,
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
        _solve_projection(prob, warm_start=True, max_iter=max_iter)
        if Q.value is None or prob.status not in _OK_STATUSES:
            raise RuntimeError(f"project_Q_bar: QP did not converge (status={prob.status}); check inputs are consistent")
        return _clip_to_support(Q.value, A, epsilon)

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
        m_Sj  = np.linalg.solve(Pi - Q @ I_Dj, rhs)            # forward solve, Eq. (21)
        lam_j = np.linalg.solve(Pi - I_Dj @ Q.T, -pib_j * pi)  # adjoint solve, Eq. (37)
        u_j   = (pib_j * In + np.diag(lam_j)) @ m_Sj
        K_lift_val += pib_j * float(pi @ m_Sj)
        grad  += np.outer(u_j, np.ones(n)) - np.outer(lam_j, np.ones(n)) * W - np.outer(lam_j, m_Sj) @ I_Dj  # Eq. (36)
    return K_lift_val, grad


def _lifted_first_passage_history(P: np.ndarray, V: np.ndarray, W: np.ndarray, K: int):
    """Build F^lift_1, ..., F^lift_K via the weighted recursion (Eq. 38).

    W[i, j] is the number of time steps to traverse edge i -> j (entries where P
    is zero are irrelevant). P^(w) = P o 1[W = w] is P restricted to edges of
    travel time w, for w = 1, ..., w_max = max(W).
    F^lift_k = P^(k) V + sum_{w=1}^{w_max} P^(w) [F^lift_{k-w} - (F^lift_{k-w} o V)],
    with F^lift_j := 0 for j <= 0 and P^(k) := 0 for k > w_max. W = 1 (all unit
    weights) recovers the unweighted recursion of Eq. (43).
    Returns the list [F^lift_0 (= 0), F^lift_1, ..., F^lift_K].
    """
    w_max = int(W.max())
    P_w = [P * jnp.asarray(W == w, dtype=P.dtype) for w in range(1, w_max + 1)]
    n, m = V.shape
    F_hist = [jnp.zeros((n, m))]
    for k in range(1, K + 1):
        Fk = P_w[k - 1] @ V if k <= w_max else jnp.zeros((n, m))
        for w in range(1, min(w_max, k) + 1):
            F_prev = F_hist[k - w]
            Fk = Fk + P_w[w - 1] @ (F_prev - F_prev * V)
        F_hist.append(Fk)
    return F_hist


# def _stackelberg_Q(Q: np.ndarray, V: np.ndarray, tau: np.ndarray, W: np.ndarray, lse_temp: float):
#     """JAX-traced lifted Stackelberg metric as a function of ergodic flow Q.

#     The true metric is min(Psi_lift), but jnp.min is flat (zero gradient) almost
#     everywhere and only subgradient-informative at the (possibly non-unique) argmin.
#     We instead use the softmin -lse_temp * logsumexp(-Psi_lift / lse_temp), which is smooth
#     and gives every entry a gradient weighted by its closeness to the minimum.
#     As lse_temp -> 0, softmin(Psi_lift) -> min(Psi_lift) (it is always a lower bound,
#     with gap <= lse_temp * log(n * m)).

#     W is the edge travel-time matrix (Eq. 38); W = 1 recovers the unweighted
#     recursion (Eq. 43).
#     """
#     pi = Q.sum(axis=1)
#     P = Q / pi[:, None]
#     n, m = V.shape
#     tau_max = int(tau.max())
#     F_hist = _lifted_first_passage_history(P, V, W, tau_max)
#     Psi_lift = jnp.zeros((n, m))
#     for k in range(1, tau_max + 1):
#         col_mask = jnp.array(tau >= k)[None, :].astype(Q.dtype)
#         Psi_lift = Psi_lift + F_hist[k] * col_mask
#     return -lse_temp * logsumexp(-Psi_lift.reshape(-1) / lse_temp)



# def make_grad_stackelberg(tau: np.ndarray, V: np.ndarray | None = None, W: np.ndarray | None = None):
#     """Return a JIT-compiled gradient function for the (softmin) lifted Stackelberg metric w.r.t. Q.

#     The returned grad_fn(Q, lse_temp) takes lse_temp as a traced argument rather than a
#     Python constant baked in at trace time, so it can be annealed across PGD
#     iterations (e.g. via projected_gradient_descent's temp_schedule) without
#     triggering recompilation: smaller lse_temp tracks the true min(Psi_lift) more
#     closely but concentrates gradients on the near-minimal entries; larger lse_temp
#     gives smoother, more informative gradients early in optimization at the cost
#     of a looser approximation to the true metric.
#     V is the n x m mapping matrix (Sec. II-C); if None, defaults to the identity
#     (no lifting), recovering the gradient of the standard (non-lifted) Stackelberg
#     metric J (Eq. 39).
#     W is the lifted edge travel-time matrix (Sec. VII, Eq. 38); entries are positive
#     integers giving the number of time steps to traverse each edge, and only entries
#     where the corresponding edge exists are used. If None, all edges default to unit
#     travel time, recovering the unweighted recursion (Eq. 43).
#     Call once before the optimization loop; the returned callable reuses the compiled
#     kernel on every iteration without retracing.
#     To maximize J^lift, negate the returned function's outputs.
#     """
#     tau = np.asarray(tau, dtype=int)
#     V = np.eye(len(tau)) if V is None else np.asarray(V)
#     V_j = jnp.array(V)
#     n = V.shape[0]
#     W = np.ones((n, n)) if W is None else np.asarray(W)
#     _compiled = jax.jit(
#         jax.value_and_grad(lambda Q, lse_temp: _stackelberg_Q(Q, V_j, tau, W, lse_temp), argnums=0)
#     )

#     def grad_fn(Q: np.ndarray, lse_temp: float) -> tuple[float, np.ndarray]:
#         val, grad = _compiled(jnp.array(Q), jnp.asarray(lse_temp, dtype=jnp.array(Q).dtype))
#         return float(val), np.array(grad)

#     return grad_fn


def _stackelberg_Q(Q: np.ndarray, V: np.ndarray, tau: np.ndarray, W: np.ndarray, lse_temp: float):
    """JAX-traced lifted Stackelberg metric as a function of ergodic flow Q (Eq. 41-42).

    Unlike the earlier (commented-out) version, this aggregates the raw set
    capture probabilities over each starting group S_i, weighted by pi and
    normalized by pi_bar_i = sum_{l in S_i} pi_l, i.e. it applies
    U = diag(pi_bar)^{-1} V^T diag(pi) to the n x m matrix of set capture
    probabilities to obtain the m x m matrix
    Psi_lift(i, j) = (1 / pi_bar_i) * sum_{l in S_i} pi_l * P[T_Sj_l <= tau_j]
    (Eq. 41), matching Psi^lift = U * sum_j sum_{k=1}^{tau_j} F_k^lift e_j e_j^T
    (Eq. 42). pi_bar is derived from Q and V (pi_bar = V^T pi) rather than
    passed in, since it is fixed by the lifting constraint V^T Q V = Q_bar.

    The true metric is min(Psi_lift), but jnp.min is flat (zero gradient) almost
    everywhere and only subgradient-informative at the (possibly non-unique) argmin.
    We instead use the softmin -lse_temp * logsumexp(-Psi_lift / lse_temp), which is smooth
    and gives every entry a gradient weighted by its closeness to the minimum.
    As lse_temp -> 0, softmin(Psi_lift) -> min(Psi_lift) (it is always a lower bound,
    with gap <= lse_temp * log(m * m)).

    W is the edge travel-time matrix (Eq. 38); W = 1 recovers the unweighted
    recursion (Eq. 43).
    """
    pi = Q.sum(axis=1)
    # epsilon=0 callers (e.g. the Stackelberg PGD runs) permit intermediate
    # iterates where a virtual state's total flow is exactly zero as long as a
    # sibling state in the same group covers the rest of V^T Q V = Q_bar.
    # Guard both divisions below so that a zero-flow (virtual or group) state
    # yields a zero row/entry instead of 0/0 = NaN, which would otherwise
    # poison the whole gradient.
    P = Q / jnp.where(pi > 0, pi, 1.0)[:, None]
    n, m = V.shape
    pi_bar = V.T @ pi
    tau_max = int(tau.max())
    F_hist = _lifted_first_passage_history(P, V, W, tau_max)
    Psi_tilde = jnp.zeros((n, m))
    for k in range(1, tau_max + 1):
        col_mask = jnp.array(tau >= k)[None, :].astype(Q.dtype)
        Psi_tilde = Psi_tilde + F_hist[k] * col_mask
    Psi_lift = (V.T @ (pi[:, None] * Psi_tilde)) / jnp.where(pi_bar > 0, pi_bar, 1.0)[:, None]
    return -lse_temp * logsumexp(-Psi_lift.reshape(-1) / lse_temp)


def make_grad_stackelberg(tau: np.ndarray, V: np.ndarray | None = None, W: np.ndarray | None = None):
    """Return a JIT-compiled gradient function for the (softmin) lifted Stackelberg metric w.r.t. Q.

    The returned grad_fn(Q, lse_temp) takes lse_temp as a traced argument rather than a
    Python constant baked in at trace time, so it can be annealed across PGD
    iterations (e.g. via projected_gradient_descent's temp_schedule) without
    triggering recompilation: smaller lse_temp tracks the true min(Psi_lift) more
    closely but concentrates gradients on the near-minimal entries; larger lse_temp
    gives smoother, more informative gradients early in optimization at the cost
    of a looser approximation to the true metric.
    pi_bar is not a parameter here: the normalization U = diag(pi_bar)^{-1} V^T diag(pi)
    in Eq. (41)-(42) is recomputed from Q (and the fixed V) on every call, via
    pi_bar = V^T pi, since it is fixed by the lifting constraint V^T Q V = Q_bar.
    V is the n x m mapping matrix (Sec. II-C); if None, defaults to the identity
    (no lifting), recovering the gradient of the standard (non-lifted) Stackelberg
    metric J (Eq. 39).
    W is the lifted edge travel-time matrix (Sec. VII, Eq. 38); entries are positive
    integers giving the number of time steps to traverse each edge, and only entries
    where the corresponding edge exists are used. If None, all edges default to unit
    travel time, recovering the unweighted recursion (Eq. 43).
    Call once before the optimization loop; the returned callable reuses the compiled
    kernel on every iteration without retracing.
    To maximize J^lift, negate the returned function's outputs.
    """
    tau = np.asarray(tau, dtype=int)
    V = np.eye(len(tau)) if V is None else np.asarray(V)
    V_j = jnp.array(V)
    n = V.shape[0]
    W = np.ones((n, n)) if W is None else np.asarray(W)
    _compiled = jax.jit(
        jax.value_and_grad(lambda Q, lse_temp: _stackelberg_Q(Q, V_j, tau, W, lse_temp), argnums=0)
    )

    def grad_fn(Q: np.ndarray, lse_temp: float) -> tuple[float, np.ndarray]:
        val, grad = _compiled(jnp.array(Q), jnp.asarray(lse_temp, dtype=jnp.array(Q).dtype))
        return float(val), np.array(grad)

    return grad_fn


def _rte_Q(Q: np.ndarray, V: np.ndarray, pi_bar: np.ndarray, K_eta: int, W: np.ndarray):
    """JAX-traced truncated lifted RTE as a function of ergodic flow Q.

    W is the edge travel-time matrix (Eq. 38); W = 1 recovers the unweighted
    recursion (Eq. 43).
    """
    pi = Q.sum(axis=1)
    P = Q / pi[:, None]
    F_hist = _lifted_first_passage_history(P, V, W, K_eta)
    H = jnp.zeros(())
    for k in range(1, K_eta + 1):
        Rk = (V * pi[:, None] * F_hist[k]).sum(axis=0) / pi_bar
        safe_Rk = jnp.where(Rk > 0, Rk, 1.0)  # avoid log(0) in forward and backward passes
        H = H - jnp.sum(jnp.where(Rk > 0, pi_bar * Rk * jnp.log(safe_Rk), 0.0))
    return H


def make_grad_rte(pi_bar: np.ndarray, eta: float, V: np.ndarray | None = None, W: np.ndarray | None = None):
    """Return a JIT-compiled gradient function for the truncated lifted RTE w.r.t. Q.

    V is the n x m mapping matrix (Sec. II-C); if None, defaults to the identity
    (no lifting), recovering the gradient of the standard (non-lifted) RTE metric H
    (Eq. 44). In that case pi_bar must equal the stationary distribution pi itself.
    W is the lifted edge travel-time matrix (Sec. VII, Eq. 38); entries are positive
    integers giving the number of time steps to traverse each edge, and only entries
    where the corresponding edge exists are used. If None, all edges default to unit
    travel time, recovering the unweighted recursion (Eq. 43).
    Call once before the optimization loop; the returned callable reuses the compiled
    kernel on every iteration without retracing.
    V, pi_bar, and W are held fixed (baked into the compiled kernel).
    To maximize H^lift, negate the returned function's outputs.
    """
    K_eta = int(np.ceil(1.0 / (eta * float(pi_bar.min())))) - 1
    V = np.eye(len(pi_bar)) if V is None else np.asarray(V)
    V_j = jnp.array(V)
    pi_bar_j = jnp.array(pi_bar)
    n = V.shape[0]
    W = np.ones((n, n)) if W is None else np.asarray(W)
    _compiled = jax.jit(jax.value_and_grad(lambda Q: _rte_Q(Q, V_j, pi_bar_j, K_eta, W)))

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
) -> tuple[np.ndarray, list[float], int]:
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
        pi_bar        = Q_bar.sum(axis=1)
        grad_lifted_H = make_grad_lifted_rte(pi_bar, eta, V)
        grad_fn       = lambda Q: tuple(-x for x in grad_lifted_H(Q))
        project_fn    = lambda Q: project_Q(Q, Q_bar, V)

    temp_schedule is optional and only applies to grad_fn's built from
    make_grad_lifted_stackelberg, whose grad_fn(Q, lse_temp) takes the softmin
    temperature as a second, traced (non-retracing) argument. When temp_schedule
    is given, grad_fn is called as grad_fn(Q, temp_schedule(k)) at iteration k
    instead of grad_fn(Q), letting lse_temp anneal (e.g. exponential decay) across
    the PGD loop while reusing the single compiled JIT kernel throughout. Example
    (maximize J^lift, so negate; lse_temp still passed positionally through the
    lambda):

      grad_stackelberg = make_grad_lifted_stackelberg(tau, V)
      grad_fn = lambda Q, lse_temp: tuple(-x for x in grad_stackelberg(Q, lse_temp))
      Q_opt, hist = projected_gradient_descent(
          Q0, grad_fn, project_fn, alpha, n_iter, tol,
          temp_schedule=lambda k: temp0 * decay ** k,
      )

    Returns (final Q, history of objective values, number of iterations run).
    history[-1] always equals the objective value at the returned Q. The
    iteration count is the number of gradient+projection steps actually taken:
    fewer than n_iter if the tol early-stop fired, else n_iter.
    """
    Q = Q0.copy()
    history: list[float] = []
    n_iters = n_iter
    for k in range(n_iter):
        val, grad = grad_fn(Q, temp_schedule(k)) if temp_schedule is not None else grad_fn(Q)
        history.append(val)
        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            n_iters = k
            break
        Q_tilde = Q - alpha * grad
        Q = project_fn(Q_tilde)
    else:
        # Loop ran to completion without the tol early-stop firing, so Q now
        # reflects one more gradient+projection step than history[-1]. Re-evaluate
        # at the final Q (and final annealed temperature) so history[-1] matches
        # the returned Q, as it already does in the early-stop case above.
        val, _ = grad_fn(Q, temp_schedule(n_iter)) if temp_schedule is not None else grad_fn(Q)
        history.append(val)

    return Q, history, n_iters
