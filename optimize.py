import numpy as np
import cvxpy as cp
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from markov import _check_mapping, _check_ergodic_flow, EQUALITY_ATOL, SUPPORT_ATOL


_SOLVER_TOL = EQUALITY_ATOL * 1e-3
_DEFAULT_MAX_ITER = 10_000
_PROJECTION_VIOLATION_TOL = 1e-4
_LINSOLVE_REG = 1e-10
# "optimal_inaccurate" just means OSQP missed its own strict convergence
# certificate, not that the point is bad -- _projection_violation is the real
# feasibility gate, so both statuses are accepted and gated on violation <= tol.
_ACCEPTABLE_STATUSES = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


def _clip_to_support(Q_value: np.ndarray, support: np.ndarray, epsilon: float) -> np.ndarray:
    """Clip a solved projection to its own epsilon*support <= Q <= support constraint."""
    return np.where(support > 0, np.maximum(Q_value, epsilon), 0.0)


def _projection_violation(Q_value: np.ndarray | None, lower: np.ndarray, upper: np.ndarray) -> float:
    """Max amount Q_value falls outside [lower, upper], or its row/col sums differ
    (0 if fully feasible). Must be evaluated on the *clipped* candidate, since
    _clip_to_support's flooring/zeroing can itself break Q @ 1 == Q.T @ 1 even when
    the raw solve was on-manifold.
    """
    if Q_value is None:
        return np.inf
    box = max(float(np.max(lower - Q_value)), float(np.max(Q_value - upper)), 0.0)
    row_col_gap = float(np.max(np.abs(Q_value.sum(axis=1) - Q_value.sum(axis=0))))
    return max(box, row_col_gap)


def _violation_tolerance(n: int, m: int) -> float:
    """Size-aware feasibility tolerance for _projection_violation.

    The row/col-sum gap is a sum of ~n independent per-entry clipping corrections,
    so a fixed absolute tolerance under-counts drift at larger lifted state spaces.
    Scales _PROJECTION_VIOLATION_TOL by sqrt(n / m) (1 for the unlifted case n == m).
    """
    return _PROJECTION_VIOLATION_TOL * np.sqrt(n / m)


def _solve_projection(
    prob: cp.Problem,
    warm_start: bool,
    Q: cp.Variable,
    lower: np.ndarray,
    upper: np.ndarray,
    support: np.ndarray,
    epsilon: float,
    tol: float,
    max_iter: int = _DEFAULT_MAX_ITER,
) -> tuple[np.ndarray | None, float]:
    """Solve a projection QP and clip the result, falling back to a cold start if
    warm-started OSQP stalls or the (possibly clipped) result is infeasible beyond
    tol. Returns (clipped_Q, violation) for the accepted attempt.
    """
    def attempt() -> tuple[np.ndarray | None, float]:
        if Q.value is None:
            return None, np.inf
        clipped = _clip_to_support(Q.value, support, epsilon)
        return clipped, _projection_violation(clipped, lower, upper)

    prob.solve(solver=cp.OSQP, warm_start=warm_start, eps_abs=_SOLVER_TOL, eps_rel=_SOLVER_TOL, max_iter=max_iter, polish=True)
    clipped, violation = attempt()
    if warm_start and (prob.status not in _ACCEPTABLE_STATUSES or violation > tol):
        prob.solve(solver=cp.OSQP, warm_start=False, eps_abs=_SOLVER_TOL, eps_rel=_SOLVER_TOL, max_iter=max_iter, polish=True)
        clipped, violation = attempt()
    return clipped, violation


def make_project_Q(
    Q_bar: np.ndarray,
    V: np.ndarray,
    epsilon: float = 1e-6,
    max_iter: int = _DEFAULT_MAX_ITER,
    equality_tol: float | None = None,
):
    """Return a cached projection function for the lifted feasible set.

    Builds the CVXPY problem once with cp.Parameter; each call updates the
    parameter and warm-starts the solver, eliminating per-call canonicalization.

    epsilon must stay small: it floors each virtual sub-edge within a physical
    (i, j) group, but all of them must still sum to Q_bar[i, j] within
    equality_tol, so too large an epsilon makes the box and equality-band
    constraints infeasible at larger budgets (confirmed empirically: 1e-3 made
    every restart at budget=3m/3.5m infeasible). The residual solver-noise
    infeasibility left at the default 1e-6 is absorbed by _violation_tolerance.

    equality_tol sets the width of the V.T @ Q @ V ≈ Q_bar band; defaults to
    EQUALITY_ATOL. Tightening it makes the lifted chain's implied physical-node
    stationary distribution track Q_bar's more closely, at the risk of
    infeasibility if Q_bar's own row/col-sum imbalance approaches it -- see
    san_francisco_kemeny_improvement's lift_equality_tol for a caller that does.
    """
    _check_ergodic_flow(Q_bar, m=V.shape[1])
    _check_mapping(V)
    n = V.shape[0]
    m = V.shape[1]
    if equality_tol is None:
        equality_tol = EQUALITY_ATOL
    Q_bar_ceil = (Q_bar > SUPPORT_ATOL).astype(int) # hardcoded to catch Stackelberg case where epsilon = 0
    U = V @ Q_bar_ceil @ V.T
    Q_tilde_p = cp.Parameter((n, n))
    Q = cp.Variable((n, n))
    constraints = [
        Q @ np.ones(n) == Q.T @ np.ones(n),
        Q >= epsilon * U,
        Q <= U,
        # Relaxed V.T @ Q @ V == Q_bar: exact equality would force Q_bar's row/col
        # sums to match exactly, but they're only guaranteed to EQUALITY_ATOL (e.g.
        # Q_bar came from another PGD projection). A band of that width keeps the
        # constraint matrix full rank so OSQP converges.
        cp.abs(V.T @ Q @ V - Q_bar) <= equality_tol,
    ]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(Q - Q_tilde_p)), constraints)

    lower = epsilon * U
    upper = U
    tol = _violation_tolerance(n, m)

    def project(Q_tilde: np.ndarray) -> np.ndarray:
        Q_tilde_p.value = Q_tilde
        clipped, violation = _solve_projection(prob, warm_start=True, Q=Q, lower=lower, upper=upper, support=U, epsilon=epsilon, tol=tol, max_iter=max_iter)
        if clipped is None or prob.status not in _ACCEPTABLE_STATUSES or violation > tol:
            raise RuntimeError(
                f"project_Q: QP did not converge to a feasible point (status={prob.status}, "
                f"violation={violation:.3e}, tol={tol:.3e}); check that Q_bar is a valid ergodic flow for V"
            )
        return clipped

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

    See make_project_Q's docstring for why epsilon must stay small.
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

    lower = epsilon * A
    upper = A
    tol = _violation_tolerance(m, m)  # unlifted physical case: n == m

    def project(Q_tilde: np.ndarray) -> np.ndarray:
        Q_tilde_p.value = Q_tilde
        clipped, violation = _solve_projection(prob, warm_start=True, Q=Q, lower=lower, upper=upper, support=A, epsilon=epsilon, tol=tol, max_iter=max_iter)
        if clipped is None or prob.status not in _ACCEPTABLE_STATUSES or violation > tol:
            raise RuntimeError(
                f"project_Q_bar: QP did not converge to a feasible point (status={prob.status}, "
                f"violation={violation:.3e}, tol={tol:.3e}); check inputs are consistent"
            )
        return clipped

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
    reg = _LINSOLVE_REG * np.eye(m)
    rhs = (Q_bar * W).sum(axis=1)
    K_val = 0.0
    grad = np.zeros((m, m))
    for j in range(m):
        gamma_j = np.ones(m)
        gamma_j[j] = 0
        Gamma_j = np.diag(gamma_j)
        m_j     = np.linalg.solve(Pi_bar - Q_bar @ Gamma_j + reg, rhs)
        lam_j   = -pi_bar[j] * np.linalg.solve(Pi_bar - Gamma_j @ Q_bar.T + reg, pi_bar)
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
    reg  = _LINSOLVE_REG * In
    rhs  = (Q * W).sum(axis=1)
    K_lift_val = 0.0
    grad  = np.zeros((n, n))
    for j in range(m):
        D_j   = np.diag(V[:, j])
        I_Dj  = In - D_j
        pib_j = float(pi_bar[j])
        m_Sj  = np.linalg.solve(Pi - Q @ I_Dj + reg, rhs)            # forward solve, Eq. (21)
        lam_j = np.linalg.solve(Pi - I_Dj @ Q.T + reg, -pib_j * pi)  # adjoint solve, Eq. (37)
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


def _stackelberg_Q(Q: np.ndarray, V: np.ndarray, tau: np.ndarray, W: np.ndarray, lse_temp: float):
    """JAX-traced lifted Stackelberg metric as a function of ergodic flow Q (Eq. 41-42).

    Aggregates raw set capture probabilities over each starting group S_i, weighted
    by pi and normalized by pi_bar_i = sum_{l in S_i} pi_l, giving
    Psi_lift(i, j) = (1 / pi_bar_i) * sum_{l in S_i} pi_l * P[T_Sj_l <= tau_j] (Eq. 41).
    pi_bar = V^T pi is derived from Q rather than passed in, since it's fixed by the
    lifting constraint V^T Q V = Q_bar.

    The true metric is min(Psi_lift), but jnp.min has zero gradient almost everywhere.
    We use the smooth softmin -lse_temp * logsumexp(-Psi_lift / lse_temp) instead, a
    lower bound that -> min(Psi_lift) as lse_temp -> 0 (gap <= lse_temp * log(m * m)).

    W is the edge travel-time matrix (Eq. 38); W = 1 recovers the unweighted recursion
    (Eq. 43).
    """
    pi = Q.sum(axis=1)
    # epsilon=0 callers permit intermediate iterates with a zero-flow virtual state;
    # guard both divisions below so that yields 0 instead of 0/0 = NaN.
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

    grad_fn(Q, lse_temp) takes lse_temp as a traced argument (not baked in at trace
    time) so it can be annealed across PGD iterations without recompiling: smaller
    lse_temp tracks the true min(Psi_lift) more closely but concentrates gradients
    near the minimum; larger gives smoother early-optimization gradients at the cost
    of a looser approximation. pi_bar = V^T pi is recomputed from Q each call rather
    than passed in, since it's fixed by the lifting constraint V^T Q V = Q_bar.
    V is the n x m mapping matrix; if None, defaults to identity (no lifting),
    recovering the standard (non-lifted) Stackelberg metric J (Eq. 39).
    W is the edge travel-time matrix (Eq. 38); if None, unit travel time (Eq. 43).
    Call once before the optimization loop; reuses the compiled kernel every
    iteration. To maximize J^lift, negate the returned function's outputs.
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

    V is the n x m mapping matrix; if None, defaults to identity (no lifting),
    recovering the standard (non-lifted) RTE metric H (Eq. 44) -- pi_bar must then
    equal the stationary distribution pi itself.
    W is the edge travel-time matrix (Eq. 38); if None, unit travel time (Eq. 43).
    V, pi_bar, and W are baked into the compiled kernel. Call once before the
    optimization loop; reuses the compiled kernel every iteration.
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
    max_grad_norm: float | None = None,
) -> tuple[np.ndarray, list[float], int]:
    """Minimize an objective via projected gradient descent.

    Each iteration: Q_tilde = Q_k - alpha * grad_fn(Q_k)[1], then Q_{k+1} =
    project_fn(Q_tilde). grad_fn and project_fn are closures capturing all fixed
    parameters, e.g. for lifted Kemeny: grad_fn = lambda Q: _grad_lifted_kemeny(Q,
    V, pi_bar), project_fn = lambda Q: project_Q(Q, Q_bar, V). To maximize instead
    of minimize, negate grad_fn's outputs.

    temp_schedule is optional, for grad_fn's built from make_grad_stackelberg, whose
    grad_fn(Q, lse_temp) takes the softmin temperature as a traced second argument.
    When given, grad_fn is called as grad_fn(Q, temp_schedule(k)) each iteration
    instead of grad_fn(Q), annealing lse_temp without retracing the JIT kernel.

    max_grad_norm, if given, clips the gradient's Frobenius norm before the step
    (preserving direction). Near-zero stationary mass on a state can make
    _grad_kemeny/_grad_lifted_kemeny's adjoint solves ill-conditioned, producing
    gradient norms orders of magnitude above the typical ~1e2-1e3 range; an
    unclipped step then sends Q_tilde far outside the feasible region, which is
    what makes project_fn raise. Clipping fixes this at the source.

    Returns (final Q, history of objective values, iterations actually run).
    history[-1] always equals the objective at the returned Q. Iteration count is
    below n_iter only if the tol early-stop fired.
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
        if max_grad_norm is not None:
            grad_norm = np.linalg.norm(grad)
            if grad_norm > max_grad_norm:
                grad = grad * (max_grad_norm / grad_norm)
        Q_tilde = Q - alpha * grad
        Q = project_fn(Q_tilde)
    else:
        # No early stop: Q is one step ahead of history[-1]; re-evaluate so
        # history[-1] matches the returned Q, as in the early-stop case above.
        val, _ = grad_fn(Q, temp_schedule(n_iter)) if temp_schedule is not None else grad_fn(Q)
        history.append(val)

    return Q, history, n_iters
