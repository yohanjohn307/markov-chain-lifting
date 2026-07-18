
import time as time
import numpy as np
import networkx as nx

from markov import ergodic_flow_to_transition, conductance, SUPPORT_ATOL
from metrics import stackelberg, lifted_stackelberg
from graph import (
    erdos_renyi_graph,
    erdos_renyi_digraph,
    random_chain,
    degree_lifting,
    san_francisco_graph,
    prune_long_edges,
    uniform_lifting,
    stationary_lifting,
    betweenness_lifting,
    eigenvector_lifting,
    reversible_flow_lifting,
)
from optimize import (
    _grad_kemeny,
    _grad_lifted_kemeny,
    make_grad_stackelberg,
    make_grad_rte,
    make_project_Q_bar,
    make_project_Q,
    projected_gradient_descent,
)


def erdos_renyi_kemeny_improvement(
    m: int = 6,
    p_values=None,
    n_graphs: int = 20,
    concentration: float = 5.0,
    budget: int | None = None,
    n_init: int = 3,
    n_iter_phys: int = 150,
    alpha_phys: float = 2e-3,
    tol_phys: float = 1e-5,
    max_kemeny_attempts: int = 10,
    n_iter_lift: int = 150,
    alpha_lift: float = 2e-3,
    tol_lift: float = 1e-5,
    max_grad_norm_phys: float | None = 2000.0,
    max_grad_norm_lift: float | None = 200.0,
    seed: int = 42,
    save_path: str = 'erdos_renyi_kemeny_diffs.npy',
) -> None:
    """Ridgeline plot of Kemeny improvement via stationary-distribution lifting vs Erdős-Rényi edge probability p.

    For each p, generates n_graphs random connected G(m, p) graphs, each with a random
    target stationary distribution pi_bar ~ Dirichlet(concentration * 1_m). For every graph:
      1. Optimises the Kemeny constant in the physical space via PGD (n_init successful
         starts, via _pgd_restarts -- a restart whose PGD raises LinAlgError/RuntimeError
         draws a fresh random init and retries rather than costing a restart slot). If the
         best K(P_bar*) found still exceeds 1000 (a heuristic threshold for a
         degenerate/poorly-conditioned (A, pi_bar) pairing), a fresh graph A and stationary
         distribution pi_bar are both resampled together and the restarts are retried, up
         to max_kemeny_attempts times before raising RuntimeError.
      2. Applies stationary-distribution lifting (budget virtual states, apportioned
         proportional to pi_bar; defaults to 2*m) and optimises the lifted Kemeny constant
         via PGD (n_init successful starts, via _pgd_restarts).
      3. Records K(P_bar*) - K^lift(P*), along with a 'status' of 'success' (diff > 0),
         'no_improvement' (PGD converged but found no better optimum than the physical
         baseline), or 'all_failed' (every retried restart raised) -- matching the
         conventions of kemeny_lifting_budget_sweep.
    Results are visualised as a joypy ridgeline plot across trials. Physical- and
    lifted-space PGD use separate hyperparameters (alpha_phys/n_iter_phys/tol_phys vs.
    alpha_lift/n_iter_lift/tol_lift), matching the split used in
    erdos_renyi_stackelberg_improvement, erdos_renyi_rte_improvement, and
    san_francisco_kemeny_improvement.

    max_grad_norm_phys/max_grad_norm_lift cap the gradient norm passed to
    projected_gradient_descent for the physical/lifted PGD restarts respectively; see
    _pgd_restarts and kemeny_lifting_budget_sweep's docstring for the rationale.
    max_grad_norm_lift defaults to a stricter 200 (vs. max_grad_norm_phys's 2000, and
    vs. kemeny_lifting_budget_sweep's 2000 default for both stages) based on an empirical
    hyperparameter sweep at budget=2*m: at 2000, occasional ill-conditioned lifted
    gradients still produced outright divergence (K^lift many times worse than K(P_bar*),
    not just a failure to improve) in ~1/10 graphs regardless of alpha_lift; clipping to
    200 eliminated every such case across p in [0.2, 0.8] and alpha_lift in [1e-3, 2e-3],
    roughly doubling the mean percent decrease, with no runtime penalty.
    """
    if p_values is None:
        p_values = np.linspace(0.3, 0.8, 4)
    lifting_budget = budget if budget is not None else 2 * m

    all_diffs: list[list[float]] = []
    all_graphs: list[list[np.ndarray]] = []
    all_Q_bar: list[list[np.ndarray]] = []
    all_Q_lift: list[list[np.ndarray]] = []
    all_times_phys: list[list[float]] = []
    all_times_lift: list[list[float]] = []
    all_iters_phys: list[list[int]] = []
    all_iters_lift: list[list[int]] = []
    all_V: list[list[np.ndarray]] = []
    all_conductance_lb: list[list[float]] = []
    all_pi_bar: list[list[np.ndarray]] = []
    all_status: list[list[str]] = []

    for p_idx, p in enumerate(p_values):
        rng = np.random.default_rng(seed * 1000 + p_idx)
        diffs_p: list[float] = []
        graphs_p: list[np.ndarray] = []
        Q_bar_p: list[np.ndarray] = []
        Q_lift_p: list[np.ndarray] = []
        times_phys_p: list[float] = []
        times_lift_p: list[float] = []
        iters_phys_p: list[int] = []
        iters_lift_p: list[int] = []
        V_p: list[np.ndarray] = []
        conductance_lb_p: list[float] = []
        pi_bar_p: list[np.ndarray] = []
        status_p: list[str] = []

        for graph_idx in range(n_graphs):
            # ------------------------------------------------------------------
            # 1. Optimise Kemeny constant in physical space
            # ------------------------------------------------------------------
            best_Q_bar: np.ndarray | None = None
            kemeny_phys = np.inf
            _t0_phys = time.perf_counter()
            for attempt in range(max_kemeny_attempts):
                A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))
                pi_bar = rng.dirichlet(concentration * np.ones(m))
                phys_proj = make_project_Q_bar(A, pi_bar)

                kemeny_phys, best_Q_bar, best_n_iters_phys, _, _ = _pgd_restarts(
                    make_Q0=lambda seed, _A=A: random_chain(_A, seed=seed),
                    project_fn=phys_proj,
                    grad_fn=_grad_kemeny,
                    alpha=alpha_phys, n_iter=n_iter_phys, tol=tol_phys,
                    n_init=n_init, rng=rng, label="physical",
                    max_grad_norm=max_grad_norm_phys,
                )

                if kemeny_phys <= 1000:
                    break

                print(
                    f"  p={p:.2f}, graph {graph_idx + 1}/{n_graphs}, attempt "
                    f"{attempt + 1}/{max_kemeny_attempts}: K(P_bar*) = {kemeny_phys:.4f} > 1000; "
                    "resampling graph and pi_bar",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    f"p={p:.2f}, graph {graph_idx + 1}/{n_graphs}: physical Kemeny optimization "
                    f"failed to reach K(P_bar*) <= 1000 after {max_kemeny_attempts} attempts"
                )
            _t1_phys = time.perf_counter()

            # Conductance lower bound on the Kemeny constant of the optimised
            # physical MC (Corollary 9): K(P_bar) >= 1 / (2 * Phi(P_bar)).
            _, conductance_lb = conductance(ergodic_flow_to_transition(best_Q_bar))

            # ------------------------------------------------------------------
            # 2. Build stationary-distribution lifting and optimise lifted Kemeny constant
            # ------------------------------------------------------------------
            V = stationary_lifting(pi_bar, budget=lifting_budget)
            A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T

            _t0_lift = time.perf_counter()
            kemeny_lift, best_Q_lift, best_n_iters_lift, n_success_lift, _ = _pgd_restarts(
                make_Q0=lambda seed, _A=A_lift: random_chain(_A, seed=seed),
                project_fn=make_project_Q(best_Q_bar, V),
                grad_fn=lambda Q, _V=V, _pi=pi_bar: _grad_lifted_kemeny(Q, _V, _pi),
                alpha=alpha_lift, n_iter=n_iter_lift, tol=tol_lift,
                n_init=n_init, rng=rng, label="lifted",
                max_grad_norm=max_grad_norm_lift,
            )
            _t1_lift = time.perf_counter()

            diff = kemeny_phys - kemeny_lift
            print(
                f"p={p:.2f}, graph {graph_idx + 1}/{n_graphs}: K(P_bar*)={kemeny_phys:.3f}, "
                f"K^lift(P*)={kemeny_lift:.3f}, diff={diff:.3f}, "
                f"best-run iters: phys={best_n_iters_phys}, lift={best_n_iters_lift}",
                flush=True,
            )

            graphs_p.append(A)
            Q_bar_p.append(best_Q_bar)
            pi_bar_p.append(pi_bar)
            if diff > 0:
                diffs_p.append(diff)
                Q_lift_p.append(best_Q_lift)
                V_p.append(V)
                status_p.append('success')
            else:
                # Lifting underperformed the physical optimum: fall back to the
                # (trivial) identity lifting of P_bar* rather than deploy a
                # worse-performing lifted MC.
                diffs_p.append(0.0)
                Q_lift_p.append(best_Q_bar)
                V_p.append(np.eye(m))
                status_p.append('no_improvement' if n_success_lift > 0 else 'all_failed')
            times_phys_p.append(_t1_phys - _t0_phys)
            times_lift_p.append(_t1_lift - _t0_lift)
            iters_phys_p.append(best_n_iters_phys)
            iters_lift_p.append(best_n_iters_lift)
            conductance_lb_p.append(conductance_lb)

        print(
            f"p={p:.2f}: {len(diffs_p)}/{n_graphs} graphs, "
            + (f"improvement = {np.mean(diffs_p):.3f} ± {np.std(diffs_p):.3f}, " if diffs_p else "")
            + (f"best-run iters: phys={np.mean(iters_phys_p):.1f}, lift={np.mean(iters_lift_p):.1f}" if diffs_p else "") + "\n",
            flush=True,
        )
        all_diffs.append(diffs_p)
        all_graphs.append(graphs_p)
        all_Q_bar.append(Q_bar_p)
        all_Q_lift.append(Q_lift_p)
        all_times_phys.append(times_phys_p)
        all_times_lift.append(times_lift_p)
        all_iters_phys.append(iters_phys_p)
        all_iters_lift.append(iters_lift_p)
        all_V.append(V_p)
        all_conductance_lb.append(conductance_lb_p)
        all_pi_bar.append(pi_bar_p)
        all_status.append(status_p)

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save(save_path,
            {'p_values': np.array(p_values), 'diffs': all_diffs, 'V': all_V,
             'graphs': all_graphs, 'Q_bar': all_Q_bar, 'Q_lift': all_Q_lift,
             'times_phys': all_times_phys, 'times_lift': all_times_lift,
             'iters_phys': all_iters_phys, 'iters_lift': all_iters_lift,
             'conductance_lb': all_conductance_lb, 'pi_bar': all_pi_bar,
             'status': all_status},
            allow_pickle=True)  # type: ignore[arg-type]


def erdos_renyi_stackelberg_improvement(
    m: int = 6,
    p_values=None,
    n_graphs: int = 20,
    budget: int | None = None,
    n_init: int = 3,
    n_iter_phys: int = 250,
    alpha_phys: float = 1.0,
    tol_phys: float = 1e-6,
    max_capt_attempts: int = 500,
    n_iter_lift: int = 250,
    alpha_lift: float = 1.0,
    tol_lift: float = 1e-6,
    temp0: float = 1e-3,
    temp_min: float = 1e-6,
    max_grad_norm_phys: float | None = None,
    max_grad_norm_lift: float | None = None,
    seed: int = 42,
    save_path: str = 'erdos_renyi_stackelberg_diffs.npy',
) -> None:
    """Ridgeline data for Stackelberg improvement via stationary-distribution lifting vs Erdős-Rényi edge probability p.

    Analogous to erdos_renyi_kemeny_improvement, but for the (unweighted) Stackelberg
    game metric J and lifted Stackelberg metric J^lift (Appendix A) instead of the Kemeny
    constant. Unlike the Kemeny case, no target stationary distribution pi_bar is sampled
    or imposed: neither a stationary-distribution constraint nor an explicit irreducibility
    constraint is needed, since the Stackelberg metric enforces irreducibility on its own (a
    reducible chain has some entry of Psi equal to zero, which can never be the max). For
    each p, generates n_graphs random connected G(m, p) graphs. For every graph:
      1. Sets tau[j] = diam(A) for every physical node j (Eq. 38), i.e. the graph's own
         unweighted diameter, the smallest uniform attack duration guaranteeing nonzero
         capture probability from any node to any other.
      2. Maximises the Stackelberg metric J in the physical space via PGD (n_init
         successful starts, via _pgd_restarts with pi_bar free, epsilon=0; grad_stackelberg
         is negated since J is maximised but _pgd_restarts/projected_gradient_descent
         minimise). If the best J(P_bar*) found is still below 0.01 (a heuristic threshold
         for a degenerate/poorly-conditioned graph draw), a fresh graph A (and its diameter
         tau) is resampled and the restarts are retried, up to max_capt_attempts times
         before raising RuntimeError. max_capt_attempts defaults far higher than
         erdos_renyi_kemeny_improvement's max_kemeny_attempts (500 vs. 10): unlike the
         Kemeny constant, which is nearly always well above its degenerate threshold on the
         first attempt, an empirical check found the physical Stackelberg optimum's success
         probability varies enormously and can be close to 0% for some p (0/25 independent
         draws at p=0.3, m=10, n_init=3, n_iter_phys=150 all converged to a reducible,
         zero-capture-probability chain), while other p draw a nonzero-J optimum readily.
         The pre-refactor code handled this via an *unbounded* retry loop (consistent with
         its "took 10 hrs" runtime comment -- it likely spent a very uneven share of that
         time on unfavorable p), so max_capt_attempts is set generously high here rather
         than matching Kemeny's cap, to keep this RuntimeError a genuine rare-failure signal
         rather than something the low end of p_values triggers routinely.
      3. Applies lifting weighted by the *realized* stationary distribution of the
         optimized physical MC (pi = best_Q_bar.sum(axis=1) -- there is no pre-specified
         target pi_bar to weight by here, unlike the Kemeny/RTE cases) via
         stationary_lifting(pi, budget), and maximises the lifted Stackelberg metric
         J^lift via PGD (n_init successful starts, via _pgd_restarts, epsilon=0).
      4. Records J^lift(P*) - J(P_bar*), along with a 'status' of 'success' (diff > 0),
         'no_improvement' (PGD converged but found no better optimum than the physical
         baseline), or 'all_failed' (every retried restart raised) -- matching the
         conventions of erdos_renyi_kemeny_improvement.
    The softmin temperature is annealed geometrically from temp0 to temp_min over
    n_iter_phys iterations (temp_k = temp0 * temp_decay**k, temp_decay = (temp_min /
    temp0)**(1/n_iter_phys)), reusing the same schedule for the lifted-space PGD, as in
    san_francisco_stackelberg_improvement.
    budget defaults to 2*m, matching erdos_renyi_kemeny_improvement.
    max_grad_norm_phys/max_grad_norm_lift cap the gradient norm passed to
    projected_gradient_descent for the physical/lifted PGD restarts respectively (see
    _pgd_restarts); left at None (no clipping) by default. An empirical sweep over
    max_grad_norm_lift in {200, 500, 2000, None} at favorable p (where the physical
    stage succeeds reliably) found *no* difference in outcome (identical success rate,
    mean improvement, and zero blow-ups at every value): the near-zero-stationary-mass
    ill-conditioning that motivated Kemeny's tuned max_grad_norm_lift=200 arises from
    _grad_kemeny/_grad_lifted_kemeny's adjoint linear solves, which this metric's
    JAX-autodiff softmin gradient doesn't share, so clipping is confirmed unnecessary
    here. (A separate empirical check also ruled out temp0 and alpha_phys as fixes for
    the low-p degenerate-optimum problem described above -- see max_capt_attempts.)
    """
    if p_values is None:
        p_values = np.linspace(0.3, 0.8, 4)
    lifting_budget = budget if budget is not None else 2 * m

    temp_decay = (temp_min / temp0) ** (1 / n_iter_phys)

    all_diffs: list[list[float]] = []
    all_graphs: list[list[np.ndarray]] = []
    all_tau: list[list[np.ndarray]] = []
    all_Q_bar: list[list[np.ndarray]] = []
    all_Q_lift: list[list[np.ndarray]] = []
    all_times_phys: list[list[float]] = []
    all_times_lift: list[list[float]] = []
    all_iters_phys: list[list[int]] = []
    all_iters_lift: list[list[int]] = []
    all_V: list[list[np.ndarray]] = []
    all_status: list[list[str]] = []

    for p_idx, p in enumerate(p_values):
        rng = np.random.default_rng(seed * 1000 + p_idx)
        diffs_p: list[float] = []
        graphs_p: list[np.ndarray] = []
        tau_p: list[np.ndarray] = []
        Q_bar_p: list[np.ndarray] = []
        Q_lift_p: list[np.ndarray] = []
        times_phys_p: list[float] = []
        times_lift_p: list[float] = []
        iters_phys_p: list[int] = []
        iters_lift_p: list[int] = []
        V_p: list[np.ndarray] = []
        status_p: list[str] = []

        for graph_idx in range(n_graphs):
            # ------------------------------------------------------------------
            # 1. Maximise Stackelberg metric J in physical space
            # ------------------------------------------------------------------
            best_Q_bar: np.ndarray | None = None
            capt_prob_phys = 0.0
            _t0_phys = time.perf_counter()
            for attempt in range(max_capt_attempts):
                # A = erdos_renyi_digraph(m, p, seed=int(rng.integers(1 << 31)))
                A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))

                # Uniform tau = graph diameter (self-loops excluded; erdos_renyi_graph
                # guarantees connectedness so the diameter is always finite).
                G = nx.from_numpy_array(A - np.diag(np.diag(A)))
                tau = np.full(m, nx.diameter(G), dtype=int)

                phys_proj = make_project_Q_bar(A, pi_bar=None, epsilon=0.0)
                grad_stackelberg = make_grad_stackelberg(tau, W=None)

                _, best_Q_bar, best_n_iters_phys, _, _ = _pgd_restarts(
                    make_Q0=lambda seed, _A=A: random_chain(_A, seed=seed),
                    project_fn=phys_proj,
                    grad_fn=lambda Q, lse_temp, f=grad_stackelberg: tuple(-x for x in f(Q, lse_temp)),
                    alpha=alpha_phys, n_iter=n_iter_phys, tol=tol_phys,
                    n_init=n_init, rng=rng, label="physical",
                    temp_schedule=lambda k: temp0 * temp_decay ** k,
                    max_grad_norm=max_grad_norm_phys,
                )
                # Re-evaluate the exact (non-smooth) metric on the winning Q rather than
                # trusting the softmin-annealed hist[-1]: softmin(Psi) <= min(Psi) always,
                # so the internal PGD value is only a lower-bound approximation of J,
                # matching how the pre-refactor code always re-evaluated via
                # metrics.stackelberg() instead of using the raw PGD history value.
                if best_Q_bar is not None:
                    capt_prob_phys = stackelberg(ergodic_flow_to_transition(best_Q_bar), tau)
                else:
                    capt_prob_phys = 0.0

                if capt_prob_phys >= 0.01:
                    break

                print(
                    f"  p={p:.2f}, graph {graph_idx + 1}/{n_graphs}, attempt "
                    f"{attempt + 1}/{max_capt_attempts}: J(P_bar*) = {capt_prob_phys:.4f} < 0.01; "
                    "resampling graph",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    f"p={p:.2f}, graph {graph_idx + 1}/{n_graphs}: physical Stackelberg optimization "
                    f"failed to reach J(P_bar*) >= 0.01 after {max_capt_attempts} attempts"
                )
            _t1_phys = time.perf_counter()

            # ------------------------------------------------------------------
            # 2. Build stationary-distribution lifting (weighted by the realized
            #    pi of best_Q_bar) and maximise lifted Stackelberg metric J^lift
            # ------------------------------------------------------------------
            V = stationary_lifting(best_Q_bar.sum(axis=1), budget=lifting_budget)
            A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T

            lift_proj = make_project_Q(best_Q_bar, V, epsilon=0.0)
            grad_lifted_stb = make_grad_stackelberg(tau, V)
            _t0_lift = time.perf_counter()
            _, best_Q_lift, best_n_iters_lift, n_success_lift, _ = _pgd_restarts(
                make_Q0=lambda seed, _A=A_lift: random_chain(_A, seed=seed),
                project_fn=lift_proj,
                grad_fn=lambda Q, lse_temp, f=grad_lifted_stb: tuple(-x for x in f(Q, lse_temp)),
                alpha=alpha_lift, n_iter=n_iter_lift, tol=tol_lift,
                n_init=n_init, rng=rng, label="lifted",
                temp_schedule=lambda k: temp0 * temp_decay ** k,
                max_grad_norm=max_grad_norm_lift,
            )
            # Re-evaluate exactly, for the same reason as the physical stage above.
            if best_Q_lift is not None:
                capt_prob_lift = lifted_stackelberg(ergodic_flow_to_transition(best_Q_lift), V, tau)
            else:
                capt_prob_lift = 0.0
            _t1_lift = time.perf_counter()

            diff = capt_prob_lift - capt_prob_phys
            print(
                f"p={p:.2f}, graph {graph_idx + 1}/{n_graphs}: J(P_bar*)={capt_prob_phys:.3f}, "
                f"J^lift(P*)={capt_prob_lift:.3f}, diff={diff:.3f}, "
                f"best-run iters: phys={best_n_iters_phys}, lift={best_n_iters_lift}",
                flush=True,
            )

            graphs_p.append(A)
            tau_p.append(tau)
            Q_bar_p.append(best_Q_bar)
            if diff > 0:
                diffs_p.append(diff)
                Q_lift_p.append(best_Q_lift)
                V_p.append(V)
                status_p.append('success')
            else:
                # Lifting underperformed the physical optimum: fall back to the
                # (trivial) identity lifting of P_bar* rather than deploy a
                # worse-performing lifted MC.
                diffs_p.append(0.0)
                Q_lift_p.append(best_Q_bar)
                V_p.append(np.eye(m))
                status_p.append('no_improvement' if n_success_lift > 0 else 'all_failed')
            times_phys_p.append(_t1_phys - _t0_phys)
            times_lift_p.append(_t1_lift - _t0_lift)
            iters_phys_p.append(best_n_iters_phys)
            iters_lift_p.append(best_n_iters_lift)

        print(
            f"p={p:.2f}: {len(diffs_p)}/{n_graphs} graphs, "
            + (f"improvement = {np.mean(diffs_p):.3f} ± {np.std(diffs_p):.3f}, " if diffs_p else "")
            + (f"best-run iters: phys={np.mean(iters_phys_p):.1f}, lift={np.mean(iters_lift_p):.1f}" if diffs_p else "") + "\n",
            flush=True,
        )
        all_diffs.append(diffs_p)
        all_graphs.append(graphs_p)
        all_tau.append(tau_p)
        all_Q_bar.append(Q_bar_p)
        all_Q_lift.append(Q_lift_p)
        all_times_phys.append(times_phys_p)
        all_times_lift.append(times_lift_p)
        all_iters_phys.append(iters_phys_p)
        all_iters_lift.append(iters_lift_p)
        all_V.append(V_p)
        all_status.append(status_p)

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save(save_path,
            {'p_values': np.array(p_values), 'tau': all_tau, 'diffs': all_diffs, 'V': all_V,
             'graphs': all_graphs, 'Q_bar': all_Q_bar, 'Q_lift': all_Q_lift,
             'times_phys': all_times_phys, 'times_lift': all_times_lift,
             'iters_phys': all_iters_phys, 'iters_lift': all_iters_lift,
             'status': all_status},
            allow_pickle=True)  # type: ignore[arg-type]


def erdos_renyi_rte_improvement(
    m: int = 6,
    p_values=None,
    n_graphs: int = 20,
    budget: int | None = None,
    n_init: int = 3,
    n_iter_phys: int = 200,
    alpha_phys: float = 1e-2,
    tol_phys: float = 1e-6,
    max_rte_attempts: int = 10,
    n_iter_lift: int = 200,
    alpha_lift: float = 1e-3,
    tol_lift: float = 1e-6,
    eta: float = 0.1,
    max_grad_norm_phys: float | None = None,
    max_grad_norm_lift: float | None = None,
    seed: int = 42,
    save_path: str = 'erdos_renyi_rte_diffs.npy',
) -> None:
    """Ridgeline data for RTE improvement via stationary-distribution lifting vs Erdős-Rényi edge probability p.

    Analogous to erdos_renyi_kemeny_improvement, but for the (unweighted) truncated
    Return-Time Entropy metric H and lifted RTE metric H^lift (Appendix B) instead of the
    Kemeny constant. As in the Kemeny case (and unlike Stackelberg), a target stationary
    distribution pi_bar is sampled and imposed via an equality-constrained projection
    (make_project_Q_bar / make_project_Q with pi_bar fixed): both H and H^lift are only
    well-defined for a fixed pi_bar, since the truncation length K_eta depends on
    pi_bar_min and the lifting constraint V^T Q V = Q_bar requires pi_bar = V^T pi. For
    each p, generates n_graphs random connected G(m, p) graphs. For every graph:
      1. Samples a random target stationary distribution pi_bar ~ Dirichlet(5*1_m) and
         maximises the truncated RTE metric H in the physical space via PGD (n_init
         successful starts, via _pgd_restarts; gradient from make_grad_rte negated to
         turn maximisation into the minimisation form expected by
         projected_gradient_descent). If the best H(P_bar*) found is still below 1.0
         (a heuristic threshold guarding against pi_bar/graph combinations for which every
         reachable chain is nearly deterministic), a fresh graph A and stationary
         distribution pi_bar are both resampled together and the restarts are retried, up
         to max_rte_attempts times before raising RuntimeError -- matching
         erdos_renyi_kemeny_improvement's degenerate-optimum handling.
      2. Applies stationary-distribution lifting (budget virtual states, apportioned
         proportional to pi_bar) and maximises the truncated lifted RTE metric H^lift via
         PGD (n_init successful starts, via _pgd_restarts), reusing the same pi_bar.
      3. Records H^lift(P*) - H(P_bar*), along with a 'status' of 'success' (diff > 0),
         'no_improvement' (PGD converged but found no better optimum than the physical
         baseline), or 'all_failed' (every retried restart raised) -- matching the
         conventions of erdos_renyi_kemeny_improvement.
    eta controls the truncation length K_eta = ceil(1 / (eta * pi_bar_min)) - 1 (Eq. 43
    / 45): smaller eta gives a tighter approximation to the untruncated RTE at the cost
    of a longer unrolled recursion inside the JAX-traced gradient (more expensive
    gradient evaluations per PGD iteration).
    Physical- and lifted-space PGD use separate step sizes (alpha_phys, alpha_lift):
    the lifted objective is sensitive to a larger step, so alpha_lift is an order of
    magnitude smaller than alpha_phys, matching the split used in
    san_francisco_kemeny_improvement.
    budget defaults to 2*m, matching erdos_renyi_kemeny_improvement.
    max_grad_norm_phys/max_grad_norm_lift cap the gradient norm passed to
    projected_gradient_descent for the physical/lifted PGD restarts respectively (see
    _pgd_restarts); left at None (no clipping) by default. An empirical sweep over
    max_grad_norm_lift in {200, 500, 2000, None} found *no* difference in outcome
    (identical success rate, mean improvement to within noise, and zero blow-ups at
    every value): the near-zero-stationary-mass ill-conditioning that motivated
    Kemeny's tuned max_grad_norm_lift=200 arises from _grad_kemeny/_grad_lifted_kemeny's
    adjoint linear solves, which this metric's JAX-autodiff entropy gradient doesn't
    share, so clipping is confirmed unnecessary here.

    A full m=10, n_graphs=20, p in linspace(0.25, 0.75, 6) sweep with degree_lifting
    (whose budget grows with p -- see graph.degree_lifting) took ~5 hrs, dominated by the
    lifted stage: at p=0.8 mean lift time per graph was ~234s vs ~37s at p=0.2, since a
    denser graph pushes the degree-lifting budget (and thus the lifted state space n)
    much higher. Switching to stationary_lifting (fixed budget=2*m regardless of p, as
    used here) removes that growth: a small-scale benchmark (m=10, p in {0.25, 0.75}, 3
    graphs each, n_init=10, n_iter=200) measured ~22-28s/graph physical and ~40-43s/graph
    lifted, flat across p, extrapolating to ~2-2.5 hrs for the full sweep -- with no
    further hyperparameter tuning. A follow-up benchmark that halved n_init (10 -> 5),
    cut n_iter by 25% (200 -> 150), and increased both step sizes found *no* meaningful
    speedup (385s vs 398s wall-clock over the same 6 graphs): per-graph cost here is
    dominated by JAX re-tracing/compiling make_grad_rte's unrolled K_eta-step recursion
    for each graph's fresh pi_bar (paid once per graph regardless of n_init/n_iter), not
    by the PGD loop itself, so trimming restarts/iterations mostly forfeits solution
    quality for negligible time savings and is not recommended.
    """
    if p_values is None:
        p_values = np.linspace(0.3, 0.8, 4)
    lifting_budget = budget if budget is not None else 2 * m

    all_diffs: list[list[float]] = []
    all_graphs: list[list[np.ndarray]] = []
    all_Q_bar: list[list[np.ndarray]] = []
    all_Q_lift: list[list[np.ndarray]] = []
    all_times_phys: list[list[float]] = []
    all_times_lift: list[list[float]] = []
    all_iters_phys: list[list[int]] = []
    all_iters_lift: list[list[int]] = []
    all_V: list[list[np.ndarray]] = []
    all_status: list[list[str]] = []

    for p_idx, p in enumerate(p_values):
        rng = np.random.default_rng(seed * 1000 + p_idx)
        diffs_p: list[float] = []
        graphs_p: list[np.ndarray] = []
        Q_bar_p: list[np.ndarray] = []
        Q_lift_p: list[np.ndarray] = []
        times_phys_p: list[float] = []
        times_lift_p: list[float] = []
        iters_phys_p: list[int] = []
        iters_lift_p: list[int] = []
        V_p: list[np.ndarray] = []
        status_p: list[str] = []

        for graph_idx in range(n_graphs):
            # ------------------------------------------------------------------
            # 1. Maximise truncated RTE metric H in physical space
            # ------------------------------------------------------------------
            best_Q_bar: np.ndarray | None = None
            pi_bar: np.ndarray | None = None
            rte_phys = 0.0
            _t0_phys = time.perf_counter()
            for attempt in range(max_rte_attempts):
                A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))
                # A = erdos_renyi_digraph(m, p, seed=int(rng.integers(1 << 31)))

                # Sample a random stationary distribution.  Concentration parameter 5
                # keeps entries bounded away from zero (min entry ≈ 1/(2m) with high prob),
                # which keeps K_eta (and thus the unrolled RTE recursion) bounded.
                pi_bar = rng.dirichlet(5 * np.ones(m))

                phys_proj = make_project_Q_bar(A, pi_bar)
                grad_rte = make_grad_rte(pi_bar, eta)

                neg_val, best_Q_bar, best_n_iters_phys, _, _ = _pgd_restarts(
                    make_Q0=lambda seed, _A=A: random_chain(_A, seed=seed),
                    project_fn=phys_proj,
                    grad_fn=lambda Q, f=grad_rte: tuple(-x for x in f(Q)),
                    alpha=alpha_phys, n_iter=n_iter_phys, tol=tol_phys,
                    n_init=n_init, rng=rng, label="physical",
                    max_grad_norm=max_grad_norm_phys,
                )
                rte_phys = -neg_val if best_Q_bar is not None else 0.0

                if rte_phys >= 1.0:
                    break

                print(
                    f"  p={p:.2f}, graph {graph_idx + 1}/{n_graphs}, attempt "
                    f"{attempt + 1}/{max_rte_attempts}: H(P_bar*) = {rte_phys:.4f} < 1.0; "
                    "resampling graph and pi_bar",
                    flush=True,
                )
            else:
                raise RuntimeError(
                    f"p={p:.2f}, graph {graph_idx + 1}/{n_graphs}: physical RTE optimization "
                    f"failed to reach H(P_bar*) >= 1.0 after {max_rte_attempts} attempts"
                )
            _t1_phys = time.perf_counter()

            # ------------------------------------------------------------------
            # 2. Build stationary-distribution lifting and maximise lifted RTE metric H^lift
            # ------------------------------------------------------------------
            V = stationary_lifting(pi_bar, budget=lifting_budget)
            A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T

            lift_proj = make_project_Q(best_Q_bar, V)
            grad_lifted_rte = make_grad_rte(pi_bar, eta, V)
            _t0_lift = time.perf_counter()
            neg_val_lift, best_Q_lift, best_n_iters_lift, n_success_lift, _ = _pgd_restarts(
                make_Q0=lambda seed, _A=A_lift: random_chain(_A, seed=seed),
                project_fn=lift_proj,
                grad_fn=lambda Q, f=grad_lifted_rte: tuple(-x for x in f(Q)),
                alpha=alpha_lift, n_iter=n_iter_lift, tol=tol_lift,
                n_init=n_init, rng=rng, label="lifted",
                max_grad_norm=max_grad_norm_lift,
            )
            rte_lift = -neg_val_lift if best_Q_lift is not None else 0.0
            _t1_lift = time.perf_counter()

            diff = rte_lift - rte_phys
            print(
                f"p={p:.2f}, graph {graph_idx + 1}/{n_graphs}: H(P_bar*)={rte_phys:.3f}, "
                f"H^lift(P*)={rte_lift:.3f}, diff={diff:.3f}, "
                f"best-run iters: phys={best_n_iters_phys}, lift={best_n_iters_lift}",
                flush=True,
            )

            graphs_p.append(A)
            Q_bar_p.append(best_Q_bar)
            if diff > 0:
                diffs_p.append(diff)
                Q_lift_p.append(best_Q_lift)
                V_p.append(V)
                status_p.append('success')
            else:
                # Lifting underperformed the physical optimum: fall back to the
                # (trivial) identity lifting of P_bar* rather than deploy a
                # worse-performing lifted MC.
                diffs_p.append(0.0)
                Q_lift_p.append(best_Q_bar)
                V_p.append(np.eye(m))
                status_p.append('no_improvement' if n_success_lift > 0 else 'all_failed')
            times_phys_p.append(_t1_phys - _t0_phys)
            times_lift_p.append(_t1_lift - _t0_lift)
            iters_phys_p.append(best_n_iters_phys)
            iters_lift_p.append(best_n_iters_lift)

        print(
            f"p={p:.2f}: {len(diffs_p)}/{n_graphs} graphs, "
            + (f"improvement = {np.mean(diffs_p):.3f} ± {np.std(diffs_p):.3f}, " if diffs_p else "")
            + (f"best-run iters: phys={np.mean(iters_phys_p):.1f}, lift={np.mean(iters_lift_p):.1f}" if diffs_p else "") + "\n",
            flush=True,
        )
        all_diffs.append(diffs_p)
        all_graphs.append(graphs_p)
        all_Q_bar.append(Q_bar_p)
        all_Q_lift.append(Q_lift_p)
        all_times_phys.append(times_phys_p)
        all_times_lift.append(times_lift_p)
        all_iters_phys.append(iters_phys_p)
        all_iters_lift.append(iters_lift_p)
        all_V.append(V_p)
        all_status.append(status_p)

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save(save_path,
            {'p_values': np.array(p_values), 'diffs': all_diffs, 'V': all_V,
             'graphs': all_graphs, 'Q_bar': all_Q_bar, 'Q_lift': all_Q_lift,
             'times_phys': all_times_phys, 'times_lift': all_times_lift,
             'iters_phys': all_iters_phys, 'iters_lift': all_iters_lift,
             'eta': eta, 'status': all_status},
            allow_pickle=True)  # type: ignore[arg-type]


def san_francisco_kemeny_improvement(
    w_max: int | None = None,
    n_trials: int = 10,
    n_init: int = 5,
    n_iter_phys: int = 150,
    alpha_phys: float = 1e-3,
    tol_phys: float = 1e-6,
    n_iter_lift: int = 50,
    alpha_lift: float = 1e-3,
    tol_lift: float = 1e-6,
    max_grad_norm_phys: float | None = 2000.0,
    max_grad_norm_lift: float | None = 200.0,
    seed: int = 42,
    save_path: str = 'results/data/san_francisco_kemeny_diffs.npy',
) -> None:
    """Kemeny improvement via degree lifting on the San Francisco graph (Sec. VII).

    Unlike erdos_renyi_kemeny_improvement, the graph, travel-time weights W,
    and target stationary distribution pi_bar are all fixed real-world data from
    graph.san_francisco_graph() (12-node police district graph, complete digraph,
    pi_bar proportional to crime rate) rather than randomly generated. If w_max is
    given, edges longer than w_max minutes are pruned (graph.prune_long_edges)
    before optimizing either space; if w_max is None (default), the full complete
    graph is used unpruned. Since the graph is fixed, we instead repeat the PGD
    optimization across n_trials random initializations (via _pgd_restarts) to
    characterize the variability of the achieved (weighted) Kemeny and lifted
    Kemeny constants:
      1. Optimises the weighted Kemeny constant in the physical space via PGD
         (n_init successful starts), for pi_bar fixed.
      2. Applies degree lifting and optimises the weighted lifted Kemeny constant
         via PGD (n_init successful starts).
      3. Records K(P_bar*) - K^lift(P*).
    max_grad_norm_phys/max_grad_norm_lift default to the same values tuned for
    erdos_renyi_kemeny_improvement (2000/200): both sweeps share the same
    _grad_kemeny/_grad_lifted_kemeny adjoint-solve machinery, whose near-zero-
    stationary-mass ill-conditioning these clip against, so the ER-tuned values
    transfer directly.
    Benchmarked on the full (w_max=None) complete graph: the physical stage is
    fast and reliable (a single PGD run of n_iter_phys=150 took ~1.2s, converging
    in ~38 iterations to K(P_bar*) ~= 23, already close to the [31]/[33]
    literature baselines of 24.3/21.2 -- see Table II).
    The lifted stage is a different story: make_project_Q's QP at the full
    degree_lifting budget (132 virtual states for this 12-node complete graph)
    took 40-90s per solve attempt in benchmarking, and across 32 attempts
    (20 random initial chains at the default epsilon=1e-6, plus 6 more at each
    of epsilon=1e-5 and 1e-4) not one reached feasibility -- so this is not a
    PGD-hyperparameter problem (no alpha/tol/n_init/epsilon combination tested
    got PGD started at all), but a scale limitation of this QP formulation at
    n=132 that no setting of the parameters below can route around.
    n_iter_lift is kept modest since a run that does eventually find a feasible
    start is not expected to benefit from a larger budget more than it costs;
    alpha_lift/tol_lift are left at their pre-existing values since no
    convergence data could be collected to retune them. Passing w_max to prune
    the graph shrinks the degree_lifting budget (e.g. w_max=6 gives 113 states
    instead of 132) and is the only lever here that changes the QP's size, but
    note pruning did not fully eliminate infeasibility in testing either (the
    *physical* stage also hit mid-optimization projection failures at w_max=6)
    -- so treat the lifted stage of this function as unreliable at any budget
    tested so far, and expect san_francisco_stackelberg_improvement's lifted
    stage (which uses epsilon=0 and is dramatically cheaper/more reliable at
    the same n=132) to remain the practical one of the three for this graph.
    """
    A, W, pi_bar = san_francisco_graph()
    if w_max is not None:
        A = prune_long_edges(A, W, threshold=w_max)
    A_off = A - np.diag(np.diag(A))
    V = degree_lifting(A, budget=int(np.maximum(A_off.sum(1), A_off.sum(0)).sum()))
    print(f"San Francisco graph: {A.shape[0]} nodes, {A.sum()} edges, {V.shape[0]} lifted states", flush=True)
    W_lift = V @ W @ V.T

    rng = np.random.default_rng(seed)
    phys_proj = make_project_Q_bar(A, pi_bar)
    grad_kemeny = lambda Q, _W=W: _grad_kemeny(Q, _W)

    diffs: list[float] = []
    Q_bar_list: list[np.ndarray] = []
    Q_lift_list: list[np.ndarray] = []
    times_phys: list[float] = []
    times_lift: list[float] = []
    iters_phys_list: list[int] = []
    iters_lift_list: list[int] = []

    for t in range(n_trials):
        _t0_phys = time.perf_counter()
        kemeny_phys, best_Q_bar, best_n_iters_phys, _, _ = _pgd_restarts(
            make_Q0=lambda seed, _A=A: random_chain(_A, seed=seed),
            project_fn=phys_proj,
            grad_fn=grad_kemeny,
            alpha=alpha_phys, n_iter=n_iter_phys, tol=tol_phys,
            n_init=n_init, rng=rng, label="physical",
            max_grad_norm=max_grad_norm_phys,
        )
        _t1_phys = time.perf_counter()

        if best_Q_bar is None:
            raise RuntimeError(f"all physical PGD restarts diverged for trial {t + 1}/{n_trials}")

        A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T
        lift_proj = make_project_Q(best_Q_bar, V)
        grad_lifted_kemeny = lambda Q, _V=V, _pi=pi_bar, _W=W_lift: _grad_lifted_kemeny(Q, _V, _pi, _W)
        _t0_lift = time.perf_counter()
        kemeny_lift, best_Q_lift, best_n_iters_lift, n_success_lift, _ = _pgd_restarts(
            make_Q0=lambda seed, _A=A_lift: random_chain(_A, seed=seed),
            project_fn=lift_proj,
            grad_fn=grad_lifted_kemeny,
            alpha=alpha_lift, n_iter=n_iter_lift, tol=tol_lift,
            n_init=n_init, rng=rng, label="lifted",
            max_attempts=4 * n_init, max_grad_norm=max_grad_norm_lift,
        )
        _t1_lift = time.perf_counter()

        if best_Q_lift is None:
            raise RuntimeError(
                f"trial {t + 1}/{n_trials}: all lifted PGD restarts hit a "
                "singular linear system before any converged"
            )

        diff = kemeny_phys - kemeny_lift
        if kemeny_phys > 0 and kemeny_lift > 0:
            diffs.append(max(diff, 0.0))
            Q_bar_list.append(best_Q_bar)
            Q_lift_list.append(best_Q_lift)
            times_phys.append(_t1_phys - _t0_phys)
            times_lift.append(_t1_lift - _t0_lift)
            iters_phys_list.append(best_n_iters_phys)
            iters_lift_list.append(best_n_iters_lift)

        print(
            f"trial {t+1}/{n_trials}: K(P_bar*)={kemeny_phys:.3f}, "
            f"K^lift(P*)={kemeny_lift:.3f}, diff={diff:.3f}, "
            f"best-run iters: phys={best_n_iters_phys}, lift={best_n_iters_lift}",
            flush=True,
        )

    print(
        f"San Francisco graph: {len(diffs)}/{n_trials} trials, "
        + (f"improvement = {np.mean(diffs):.3f} ± {np.std(diffs):.3f}" if diffs else ""),
        flush=True,
    )

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save(save_path,
            {'diffs': diffs, 'V': V, 'A': A, 'W': W, 'pi_bar': pi_bar,
             'Q_bar': Q_bar_list, 'Q_lift': Q_lift_list,
             'times_phys': times_phys, 'times_lift': times_lift,
             'iters_phys': iters_phys_list, 'iters_lift': iters_lift_list},
            allow_pickle=True)  # type: ignore[arg-type]


def san_francisco_stackelberg_improvement(
    tau: int | np.ndarray,
    w_max: int | None = None,
    n_trials: int = 10,
    n_init: int = 3,
    n_iter_phys: int = 250,
    alpha_phys: float = 1.0,
    tol_phys: float = 1e-6,
    n_iter_lift: int = 250,
    alpha_lift: float = 0.3,
    tol_lift: float = 1e-6,
    temp0: float = 1e-3,
    temp_min: float = 1e-6,
    max_grad_norm_phys: float | None = None,
    max_grad_norm_lift: float | None = None,
    seed: int = 42,
    save_path: str = 'results/data/san_francisco_stackelberg_diffs.npy',
) -> None:
    """Stackelberg improvement via degree lifting on the San Francisco graph (Appendix A).

    As in san_francisco_kemeny_improvement, the graph and travel-time weights W
    are fixed real-world data from graph.san_francisco_graph() (12-node police
    district graph, complete digraph). If w_max is given, edges longer than
    w_max minutes are pruned (graph.prune_long_edges) before optimizing either
    space; if w_max is None (default), the full complete graph is used unpruned.
    Unlike the Kemeny case, the target stationary distribution pi_bar is *not*
    imposed: neither a stationary distribution constraint nor an explicit
    irreducibility constraint is needed, since the Stackelberg metric enforces
    irreducibility on its own (a reducible chain has some entry of Psi equal to
    zero, which can never be the max). We repeat the PGD optimization across
    n_trials random initializations (via _pgd_restarts) to characterize the
    variability of the achieved (weighted) Stackelberg and lifted Stackelberg
    metrics:
      1. Maximises the weighted Stackelberg metric J in the physical space via PGD
         (n_init successful starts), with pi_bar free (project_Q_bar called with
         pi_bar=None, epsilon=0).
      2. Applies degree lifting and maximises the weighted lifted Stackelberg
         metric J^lift via PGD (n_init successful starts, epsilon=0).
      3. Records J^lift(P*) - J(P_bar*).
    tau[j] is the attack duration at physical node j (Eq. 38); pass either a
    scalar (broadcast to every node) or an array of length m = A.shape[0] = 12.
    A natural choice is the weighted diameter of the graph actually being
    optimized (pruned if w_max is given, complete otherwise), i.e. the longest
    shortest travel time between any pair of nodes, which is the smallest attack
    duration guaranteeing nonzero capture probability from any node to any
    other. The softmin temperature is annealed geometrically from temp0 to
    temp_min over n_iter_phys iterations (temp_k = temp0 * temp_decay**k,
    temp_decay = (temp_min / temp0)**(1/n_iter_phys)), tightening the softmin ->
    min approximation as optimization progresses while reusing a single
    compiled JIT kernel throughout (see make_grad_stackelberg). The same
    schedule is reused for the lifted-space PGD.
    Benchmarked on the full (w_max=None) complete graph with tau = graph
    diameter: unlike Kemeny/RTE, the Stackelberg lifted projection uses
    epsilon=0 (see make_project_Q), which is dramatically cheaper at the full
    132-state degree_lifting budget (~1-3s per solve, vs 40-90s+ for Kemeny's
    epsilon=1e-6 projection at the same size) and reliably feasible from a
    random initial chain. alpha_lift=0.3 (tuned here, up from an earlier 0.1)
    reached a better J^lift(P*) (0.0958 vs 0.0935) in fewer iterations (41 vs
    50, via the tol early-stop) in benchmarking; alpha_phys=1.0 and
    n_iter_phys=250 were confirmed to already be well-tuned (alpha_phys=1.0
    outperformed 0.5, reaching J(P_bar*) ~= 0.089, close to the [33] literature
    baseline of 0.0975 -- see Table II).
    """
    A, W, _ = san_francisco_graph()
    if w_max is not None:
        A = prune_long_edges(A, W, threshold=w_max)
    A_off = A - np.diag(np.diag(A))
    V = degree_lifting(A, budget=int(np.maximum(A_off.sum(1), A_off.sum(0)).sum()))
    print(f"San Francisco graph: {A.shape[0]} nodes, {A.sum()} edges, {V.shape[0]} lifted states", flush=True)
    W_lift = V @ W @ V.T

    m = A.shape[0]
    tau = np.full(m, tau, dtype=int) if np.isscalar(tau) else np.asarray(tau, dtype=int)
    if tau.shape != (m,):
        raise ValueError(f"tau must be a scalar or have length m={m}, got shape {tau.shape}")

    temp_decay = (temp_min / temp0) ** (1 / n_iter_phys)

    rng = np.random.default_rng(seed)
    phys_proj = make_project_Q_bar(A, pi_bar=None, epsilon=0.0)
    grad_stackelberg = make_grad_stackelberg(tau, W=W)
    grad_lifted_stb = make_grad_stackelberg(tau, V, W_lift)

    diffs: list[float] = []
    Q_bar_list: list[np.ndarray] = []
    Q_lift_list: list[np.ndarray] = []
    times_phys: list[float] = []
    times_lift: list[float] = []
    iters_phys_list: list[int] = []
    iters_lift_list: list[int] = []

    for t in range(n_trials):
        _t0_phys = time.perf_counter()
        _, best_Q_bar, best_n_iters_phys, _, _ = _pgd_restarts(
            make_Q0=lambda seed, _A=A: random_chain(_A, seed=seed),
            project_fn=phys_proj,
            grad_fn=lambda Q, lse_temp, f=grad_stackelberg: tuple(-x for x in f(Q, lse_temp)),
            alpha=alpha_phys, n_iter=n_iter_phys, tol=tol_phys,
            n_init=n_init, rng=rng, label="physical",
            temp_schedule=lambda k: temp0 * temp_decay ** k,
            max_grad_norm=max_grad_norm_phys,
        )
        # Re-evaluate the exact (non-smooth) metric on the winning Q rather than
        # trusting the softmin-annealed PGD history value: softmin(Psi) <=
        # min(Psi) always, so the internal PGD value is only a lower-bound
        # approximation of J.
        capt_prob_phys = stackelberg(ergodic_flow_to_transition(best_Q_bar), tau, W) if best_Q_bar is not None else 0.0
        _t1_phys = time.perf_counter()

        if best_Q_bar is None:
            raise RuntimeError(f"trial {t + 1}/{n_trials}: all physical PGD restarts failed")

        A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T
        lift_proj = make_project_Q(best_Q_bar, V, epsilon=0.0)
        _t0_lift = time.perf_counter()
        _, best_Q_lift, best_n_iters_lift, n_success_lift, _ = _pgd_restarts(
            make_Q0=lambda seed, _A=A_lift: random_chain(_A, seed=seed),
            project_fn=lift_proj,
            grad_fn=lambda Q, lse_temp, f=grad_lifted_stb: tuple(-x for x in f(Q, lse_temp)),
            alpha=alpha_lift, n_iter=n_iter_lift, tol=tol_lift,
            n_init=n_init, rng=rng, label="lifted",
            temp_schedule=lambda k: temp0 * temp_decay ** k,
            max_grad_norm=max_grad_norm_lift,
        )
        # Re-evaluate exactly, for the same reason as the physical stage above.
        capt_prob_lift = lifted_stackelberg(ergodic_flow_to_transition(best_Q_lift), V, tau, W_lift) if best_Q_lift is not None else 0.0
        _t1_lift = time.perf_counter()

        if best_Q_lift is None:
            raise RuntimeError(f"trial {t + 1}/{n_trials}: all lifted PGD restarts failed")

        diff = capt_prob_lift - capt_prob_phys
        if capt_prob_phys > 0 and capt_prob_lift > 0:
            diffs.append(max(diff, 0.0))
            Q_bar_list.append(best_Q_bar)
            Q_lift_list.append(best_Q_lift)
            times_phys.append(_t1_phys - _t0_phys)
            times_lift.append(_t1_lift - _t0_lift)
            iters_phys_list.append(best_n_iters_phys)
            iters_lift_list.append(best_n_iters_lift)

        print(
            f"trial {t+1}/{n_trials}: J(P_bar*)={capt_prob_phys:.3f}, "
            f"J^lift(P*)={capt_prob_lift:.3f}, diff={diff:.3f}, "
            f"best-run iters: phys={best_n_iters_phys}, lift={best_n_iters_lift}",
            flush=True,
        )

    print(
        f"San Francisco graph: {len(diffs)}/{n_trials} trials, "
        + (f"improvement = {np.mean(diffs):.3f} ± {np.std(diffs):.3f}" if diffs else ""),
        flush=True,
    )

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save(save_path,
            {'diffs': diffs, 'V': V, 'A': A, 'W': W, 'tau': tau,
             'Q_bar': Q_bar_list, 'Q_lift': Q_lift_list,
             'times_phys': times_phys, 'times_lift': times_lift,
             'iters_phys': iters_phys_list, 'iters_lift': iters_lift_list},
            allow_pickle=True)  # type: ignore[arg-type]


def san_francisco_rte_improvement(
    w_max: int | None = None,
    n_trials: int = 10,
    n_init: int = 3,
    n_iter_phys: int = 200,
    alpha_phys: float = 1e-2,
    tol_phys: float = 1e-6,
    n_iter_lift: int = 50,
    alpha_lift: float = 1e-3,
    tol_lift: float = 1e-6,
    eta: float = 0.1,
    max_grad_norm_phys: float | None = None,
    max_grad_norm_lift: float | None = None,
    seed: int = 42,
    save_path: str = 'results/data/san_francisco_rte_diffs.npy',
) -> None:
    """RTE improvement via degree lifting on the San Francisco graph (Appendix B).

    As in san_francisco_kemeny_improvement, the graph, travel-time weights W, and
    target stationary distribution pi_bar are all fixed real-world data from
    graph.san_francisco_graph(). If w_max is given, edges longer than w_max
    minutes are pruned (graph.prune_long_edges) before optimizing either space;
    if w_max is None (default), the full complete graph is used unpruned. We
    repeat the PGD optimization across n_trials random initializations (via
    _pgd_restarts) to characterize the variability of the achieved (weighted)
    truncated Return-Time Entropy H and lifted RTE H^lift:
      1. Maximises the weighted truncated RTE metric H in the physical space via
         PGD (n_init successful starts), for pi_bar fixed.
      2. Applies degree lifting and maximises the weighted lifted RTE metric
         H^lift via PGD (n_init successful starts).
      3. Records H^lift(P*) - H(P_bar*).
    eta controls the truncation length K_eta = ceil(1 / (eta * pi_bar_min)) - 1
    (Eq. 43 / 45); defaults to 0.1, matching erdos_renyi_rte_improvement and
    Appendix B. max_grad_norm_phys/max_grad_norm_lift default to None, matching
    erdos_renyi_rte_improvement's finding that gradient clipping made no
    measurable difference for this JAX-autodiff entropy gradient (unlike
    Kemeny's adjoint-solve gradient, which does need it).
    Benchmarked on the full (w_max=None) complete graph: the physical stage's
    make_grad_rte kernel took ~72s to JIT-compile on its first call (a one-time
    cost for the whole sweep, since grad_rte is built once, not per trial) but
    each subsequent call was ~0.02s, so n_iter_phys=200 and alpha_phys=1e-2 are
    cheap and were already well-tuned (matching erdos_renyi_rte_improvement).
    The lifted stage shares san_francisco_kemeny_improvement's make_project_Q
    call (same epsilon=1e-6, same n=132 budget for this graph), which that
    function's docstring found to be infeasible in 0/32 benchmarked attempts
    across three epsilon values -- expect the same unreliability here, for the
    same reasons; n_iter_lift is kept modest for the same reasoning given there.
    """
    A, W, pi_bar = san_francisco_graph()
    if w_max is not None:
        A = prune_long_edges(A, W, threshold=w_max)
    A_off = A - np.diag(np.diag(A))
    V = degree_lifting(A, budget=int(np.maximum(A_off.sum(1), A_off.sum(0)).sum()))
    print(f"San Francisco graph: {A.shape[0]} nodes, {A.sum()} edges, {V.shape[0]} lifted states", flush=True)
    W_lift = V @ W @ V.T

    rng = np.random.default_rng(seed)
    phys_proj = make_project_Q_bar(A, pi_bar)
    grad_rte = make_grad_rte(pi_bar, eta, W=W)
    grad_lifted_rte = make_grad_rte(pi_bar, eta, V, W_lift)

    diffs: list[float] = []
    Q_bar_list: list[np.ndarray] = []
    Q_lift_list: list[np.ndarray] = []
    times_phys: list[float] = []
    times_lift: list[float] = []
    iters_phys_list: list[int] = []
    iters_lift_list: list[int] = []

    for t in range(n_trials):
        _t0_phys = time.perf_counter()
        neg_val, best_Q_bar, best_n_iters_phys, _, _ = _pgd_restarts(
            make_Q0=lambda seed, _A=A: random_chain(_A, seed=seed),
            project_fn=phys_proj,
            grad_fn=lambda Q, f=grad_rte: tuple(-x for x in f(Q)),
            alpha=alpha_phys, n_iter=n_iter_phys, tol=tol_phys,
            n_init=n_init, rng=rng, label="physical",
            max_grad_norm=max_grad_norm_phys,
        )
        rte_phys = -neg_val if best_Q_bar is not None else 0.0
        _t1_phys = time.perf_counter()

        if best_Q_bar is None:
            raise RuntimeError(f"trial {t + 1}/{n_trials}: all physical PGD restarts failed")

        A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T
        lift_proj = make_project_Q(best_Q_bar, V)
        _t0_lift = time.perf_counter()
        neg_val_lift, best_Q_lift, best_n_iters_lift, n_success_lift, _ = _pgd_restarts(
            make_Q0=lambda seed, _A=A_lift: random_chain(_A, seed=seed),
            project_fn=lift_proj,
            grad_fn=lambda Q, f=grad_lifted_rte: tuple(-x for x in f(Q)),
            alpha=alpha_lift, n_iter=n_iter_lift, tol=tol_lift,
            n_init=n_init, rng=rng, label="lifted",
            max_grad_norm=max_grad_norm_lift,
        )
        rte_lift = -neg_val_lift if best_Q_lift is not None else 0.0
        _t1_lift = time.perf_counter()

        if best_Q_lift is None:
            raise RuntimeError(f"trial {t + 1}/{n_trials}: all lifted PGD restarts failed")

        diff = rte_lift - rte_phys
        if rte_phys > 0 and rte_lift > 0:
            diffs.append(max(diff, 0.0))
            Q_bar_list.append(best_Q_bar)
            Q_lift_list.append(best_Q_lift)
            times_phys.append(_t1_phys - _t0_phys)
            times_lift.append(_t1_lift - _t0_lift)
            iters_phys_list.append(best_n_iters_phys)
            iters_lift_list.append(best_n_iters_lift)

        print(
            f"trial {t+1}/{n_trials}: H(P_bar*)={rte_phys:.3f}, "
            f"H^lift(P*)={rte_lift:.3f}, diff={diff:.3f}, "
            f"best-run iters: phys={best_n_iters_phys}, lift={best_n_iters_lift}",
            flush=True,
        )

    print(
        f"San Francisco graph: {len(diffs)}/{n_trials} trials, "
        + (f"improvement = {np.mean(diffs):.3f} ± {np.std(diffs):.3f}" if diffs else ""),
        flush=True,
    )

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save(save_path,
            {'diffs': diffs, 'V': V, 'A': A, 'W': W, 'pi_bar': pi_bar, 'eta': eta,
             'Q_bar': Q_bar_list, 'Q_lift': Q_lift_list,
             'times_phys': times_phys, 'times_lift': times_lift,
             'iters_phys': iters_phys_list, 'iters_lift': iters_lift_list},
            allow_pickle=True)  # type: ignore[arg-type]


def _pgd_restarts(
    make_Q0,
    project_fn,
    grad_fn,
    alpha: float,
    n_iter: int,
    tol: float,
    n_init: int,
    rng: np.random.Generator,
    label: str = "",
    max_attempts: int | None = None,
    verbose: bool = True,
    max_grad_norm: float | None = None,
    temp_schedule=None,
) -> tuple[float, np.ndarray | None, int, int, int]:
    """Run PGD restarts until n_init succeed or the attempt budget runs out.

    make_Q0(seed) draws a fresh raw (unprojected) initial point. A restart
    that raises LinAlgError/RuntimeError (e.g. a marginally-infeasible
    projection or a near-singular first-passage-time solve) draws a new
    random init and retries rather than permanently costing a restart slot,
    up to max_attempts total attempts (default 2 * n_init).
    max_grad_norm is passed through to projected_gradient_descent -- see its
    docstring; it is the primary defense against near-zero-stationary-mass
    virtual states blowing up the gradient and, in turn, the projection step.
    temp_schedule is also passed through to projected_gradient_descent, for
    grad_fn's built from make_grad_stackelberg whose softmin temperature is
    annealed across iterations (see its docstring); leave None otherwise.
    Restarts are ranked by the minimum hist[-1] found (no positivity floor,
    so this works both for grad_fn's that return an objective directly
    (e.g. _grad_kemeny, always positive) and for grad_fn's that negate a
    maximization objective to fit PGD's minimization form (e.g. Stackelberg/
    RTE via `lambda Q: tuple(-x for x in f(Q))`), whose hist values are <= 0;
    callers of the latter must negate best_val back before interpreting it.
    Returns (best_val, best_Q, best_n_iters, n_success, n_failed): best_val is
    np.inf and best_Q is None if every attempt failed.
    """
    if max_attempts is None:
        max_attempts = 2 * n_init
    best_val = np.inf
    best_Q: np.ndarray | None = None
    best_n_iters = 0
    n_success = 0
    n_failed = 0
    attempts = 0
    while n_success < n_init and attempts < max_attempts:
        attempts += 1
        Q0 = make_Q0(int(rng.integers(1 << 31)))
        try:
            Q0 = project_fn(Q0)
            Q_opt, hist, n_iters = projected_gradient_descent(
                Q0, grad_fn, project_fn, alpha, n_iter, tol,
                temp_schedule=temp_schedule, max_grad_norm=max_grad_norm,
            )
        except (np.linalg.LinAlgError, RuntimeError) as e:
            n_failed += 1
            if verbose:
                print(f"  {label} PGD init failed ({e}); retrying with a fresh init", flush=True)
            continue
        n_success += 1
        if hist and hist[-1] < best_val:
            best_val = hist[-1]
            best_Q = Q_opt
            best_n_iters = n_iters
    return best_val, best_Q, best_n_iters, n_success, n_failed


def kemeny_lifting_budget_sweep(
    m: int,
    n_graphs: int = 20,
    p_range: tuple[float, float] = (0.2, 0.8),
    concentration: float = 5.0,
    budget_values=None,
    n_init: int = 3,
    n_iter_phys: int = 150,
    alpha_phys: float = 2e-3,
    tol_phys: float = 1e-5,
    max_kemeny_attempts: int = 10,
    n_iter_lift: int = 150,
    alpha_lift: float = 2e-3,
    tol_lift: float = 1e-5,
    max_grad_norm_phys: float | None = 2000.0,
    max_grad_norm_lift: float | None = 200.0,
    seed: int = 42,
    save_path: str = 'lifting_budget_sweep.npy',
) -> None:
    """Compare six lifting-budget allocation heuristics across a budget sweep, over many random graphs.

    For each of n_graphs random undirected Erdős-Rényi graphs G(m, p), with p drawn
    uniformly from p_range independently per graph, and a random target stationary
    distribution pi_bar ~ Dirichlet(concentration * 1_m), this:
      1. Optimises the Kemeny constant in the physical space via PGD
         (n_init successful starts, via _pgd_restarts -- a restart whose PGD
         raises LinAlgError/RuntimeError draws a fresh random init and retries
         rather than costing a restart slot), once per graph, giving a shared
         baseline K(P_bar*) and best_Q_bar reused across every budget/method
         combination for that graph. If the best K(P_bar*) found still exceeds
         1000 (a heuristic threshold for a degenerate/poorly-conditioned
         (A, pi_bar) pairing), a fresh graph A and stationary distribution
         pi_bar are both resampled together and the restarts are retried, up
         to max_kemeny_attempts times before raising RuntimeError.
      2. For each budget in budget_values and each of six lifting heuristics
         (uniform, stationary, degree, betweenness, eigenvector, reversible_flow —
         see graph.proportional_lifting and its callers), builds the corresponding
         V of shape (budget, m) and optimises the lifted Kemeny constant
         via PGD (n_init successful starts, via _pgd_restarts).
      3. Records K(P_bar*) - K^lift(P*) for every (budget, method) pair, along
         with a 'status' of 'success' (diff > 0), 'no_improvement' (PGD
         converged but found no better optimum than the physical baseline), or
         'all_failed' (every retried restart raised) -- the latter two both
         fall back to the identity lifting of P_bar* in 'Q_lift'/'kemeny_lift',
         but 'status' keeps them distinguishable.
    The reversible_flow method requires best_Q_bar (weights x_i = sum_j
    min(q_bar_ij, q_bar_ji)) and so is built freshly per graph using that graph's
    physical optimum, unlike the other four which only depend on A / pi_bar.
    budget_values defaults to multiples of m from m (the minimum valid budget,
    one virtual state per node, i.e. no lifting) to 4m. Per-graph results (the
    sampled p, the graph A, pi_bar, kemeny_phys, best_Q_bar, and the raw V,
    Q_lift, kemeny_lift, diff, and status for every budget/method combination)
    are collected across all n_graphs and saved to save_path so the distribution of
    improvement for each lifting procedure at each budget can be visualised
    without re-running optimization.

    max_grad_norm_phys/max_grad_norm_lift cap the gradient norm passed to
    projected_gradient_descent (see its docstring) for the physical/lifted PGD
    restarts respectively. Near-zero stationary mass on a (virtual) state --
    more likely at larger lifting budgets or skewed lifting allocations like
    reversible_flow_lifting -- can make _grad_kemeny/_grad_lifted_kemeny's
    linear solves severely ill-conditioned, producing gradient norms many
    orders of magnitude above the typical ~1e2-1e3 range for these problems;
    an unclipped step then sends the projection's input far outside the
    feasible region, which is what actually drives most all-restarts-failed
    outcomes at higher budgets. Set to None to disable clipping.
    max_grad_norm_lift defaults to a stricter 200 (vs. max_grad_norm_phys's 2000)
    based on an empirical hyperparameter sweep across all six lifting methods at
    budget in {2m, 3m}: at 2000, PGD outright diverged (K^lift several times worse
    than K(P_bar*), not merely a failure to improve) on 2/60 (budget, method)
    combinations and landed on 'no_improvement' on a further 10/60; even 500 still
    left 2 divergences and 4 no_improvements. 200 eliminated every divergence and
    no_improvement across all six methods and both budgets, and roughly doubled the
    mean percent decrease versus the 2000 default -- matching the same tuning result
    found for erdos_renyi_kemeny_improvement's stationary-lifting-only, budget=2m case.
    """
    if budget_values is None:
        budget_values = np.arange(m, 4 * m + 1, m)

    rng = np.random.default_rng(seed)

    lifting_method_names = [
        'uniform', 'stationary', 'degree', 'betweenness', 'eigenvector', 'reversible_flow',
    ]

    p_values: list[float] = []
    graphs: list[np.ndarray] = []
    pi_bars: list[np.ndarray] = []
    kemeny_phys_all: list[float] = []
    Q_bar_all: list[np.ndarray] = []
    results_all: list[dict] = []

    for graph_idx in range(n_graphs):
        p = float(rng.uniform(*p_range))

        # ------------------------------------------------------------------
        # 1. Optimise the physical Kemeny constant once; shared across every
        #    budget and lifting method compared below, for this graph.
        # ------------------------------------------------------------------
        best_Q_bar: np.ndarray | None = None
        kemeny_phys = np.inf
        for attempt in range(max_kemeny_attempts):
            A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))
            print(f"=== Graph {graph_idx + 1}/{n_graphs}: p={p:.4f} ===", flush=True)
            pi_bar = rng.dirichlet(concentration * np.ones(m))
            phys_proj = make_project_Q_bar(A, pi_bar)

            kemeny_phys, best_Q_bar, _, _, _ = _pgd_restarts(
                make_Q0=lambda seed, _A=A: random_chain(_A, seed=seed),
                project_fn=phys_proj,
                grad_fn=_grad_kemeny,
                alpha=alpha_phys, n_iter=n_iter_phys, tol=tol_phys,
                n_init=n_init, rng=rng, label="physical",
                max_grad_norm=max_grad_norm_phys,
            )

            if kemeny_phys <= 1000:
                break

            # Every restart for this (A, pi_bar) pair either failed, was
            # rejected as numerically degenerate, or converged to a poorly
            # conditioned optimum; resample the graph and pi_bar together.
            print(
                f"  attempt {attempt + 1}/{max_kemeny_attempts}: K(P_bar*) = {kemeny_phys:.4f} > 1000; "
                "resampling graph and pi_bar",
                flush=True,
            )
        else:
            raise RuntimeError(
                f"graph {graph_idx + 1}/{n_graphs}: physical Kemeny optimization failed to reach "
                f"K(P_bar*) <= 1000 after {max_kemeny_attempts} attempts"
            )
        print(f"  Physical Kemeny optimum: K(P_bar*) = {kemeny_phys:.4f}", flush=True)

        # sparsity pattern of the optimized physical MC, used to build each lifted graph
        support = (best_Q_bar > SUPPORT_ATOL).astype(int)

        lifting_methods = {
            'uniform': lambda budget: uniform_lifting(A, budget),
            'stationary': lambda budget: stationary_lifting(pi_bar, budget),
            'degree': lambda budget: degree_lifting(A, budget),
            'betweenness': lambda budget: betweenness_lifting(A, budget),
            'eigenvector': lambda budget: eigenvector_lifting(A, budget),
            'reversible_flow': lambda budget: reversible_flow_lifting(best_Q_bar, budget),
        }

        results = {
            name: {'budgets': [], 'diffs': [], 'kemeny_lift': [], 'V': [], 'Q_lift': [], 'iters_lift': [], 'status': []}
            for name in lifting_method_names
        }

        for budget in budget_values:
            budget = int(budget)
            for name, make_V in lifting_methods.items():
                V = make_V(budget)
                A_lift = V @ support @ V.T
                lift_proj = make_project_Q(best_Q_bar, V)

                kemeny_lift, best_Q_lift, best_n_iters_lift, n_success_lift, _ = _pgd_restarts(
                    make_Q0=lambda seed, _A=A_lift: random_chain(_A, seed=seed),
                    project_fn=lift_proj,
                    grad_fn=lambda Q, _V=V, _pi=pi_bar: _grad_lifted_kemeny(Q, _V, _pi),
                    alpha=alpha_lift, n_iter=n_iter_lift, tol=tol_lift,
                    n_init=n_init, rng=rng, label=f"[{name}]", verbose=False,
                    max_grad_norm=max_grad_norm_lift,
                )

                diff = kemeny_phys - kemeny_lift
                print(
                    f"  budget={budget}, method={name}: "
                    f"K^lift(P*)={kemeny_lift if diff > 0 else kemeny_phys:.4f}, "
                    f"diff={diff if diff > 0 else 0.0:.4f}",
                    flush=True,
                )
                results[name]['budgets'].append(budget)
                if diff > 0:
                    results[name]['diffs'].append(diff)
                    results[name]['kemeny_lift'].append(kemeny_lift)
                    results[name]['V'].append(V)
                    results[name]['Q_lift'].append(best_Q_lift)
                    results[name]['status'].append('success')
                else:
                    # Lifting underperformed the physical optimum: fall back to the
                    # (trivial) identity lifting of P_bar* rather than deploy a
                    # worse-performing lifted MC, matching the fallback used in
                    # erdos_renyi_kemeny_improvement.
                    results[name]['diffs'].append(0.0)
                    results[name]['kemeny_lift'].append(kemeny_phys)
                    results[name]['V'].append(np.eye(m))
                    results[name]['Q_lift'].append(best_Q_bar)
                    # 'all_failed' means every one of the (retried) PGD restarts
                    # raised, so this fallback reflects a numerical failure, not a
                    # genuine finding that no lifting helps -- distinct from
                    # 'no_improvement', where PGD converged but simply found a
                    # worse optimum than the physical baseline.
                    results[name]['status'].append('no_improvement' if n_success_lift > 0 else 'all_failed')
                results[name]['iters_lift'].append(best_n_iters_lift)

            best_name = max(
                (name for name in lifting_method_names if results[name]['budgets'] and results[name]['budgets'][-1] == budget),
                key=lambda name: results[name]['diffs'][-1],
                default=None,
            )
            if best_name is not None:
                print(f"  budget={budget}: best method = {best_name} (diff={results[best_name]['diffs'][-1]:.4f})\n", flush=True)

        p_values.append(p)
        graphs.append(A)
        pi_bars.append(pi_bar)
        kemeny_phys_all.append(kemeny_phys)
        Q_bar_all.append(best_Q_bar)
        results_all.append(results)

    # Save raw data so the comparison plot can be re-rendered without re-running optimization
    np.save(save_path,
            {'m': m, 'budget_values': np.array(budget_values),
             'p_values': np.array(p_values), 'graphs': graphs, 'pi_bar': pi_bars,
             'kemeny_phys': kemeny_phys_all, 'Q_bar': Q_bar_all, 'results': results_all},
            allow_pickle=True)  # type: ignore[arg-type]


if __name__ == "__main__":
    # # This took 40 mins
    # _t0 = time.time()
    # erdos_renyi_kemeny_improvement(
    #     m=10,
    #     p_values=np.linspace(0.25, 0.75, 6),
    #     n_graphs=20,
    #     n_init=10,
    #     budget=20,
    #     n_iter_phys=150,
    #     alpha_phys=2e-3,
    #     tol_phys=1e-5,
    #     n_iter_lift=150,
    #     alpha_lift=2e-3,
    #     tol_lift=1e-5,
    #     seed=42,
    # )
    # print(f"E-R Kemeny elapsed: {time.time() - _t0:.1f}s")

    # # This took 100 mins
    # _t0 = time.time()
    # erdos_renyi_stackelberg_improvement(
    #     m=10,
    #     p_values=np.linspace(0.25, 0.75, 6),
    #     n_graphs=20,
    #     n_init=10,
    #     budget=20,
    #     n_iter_phys=250,
    #     alpha_phys=1.0,
    #     tol_phys=1e-5,
    #     n_iter_lift=250,
    #     alpha_lift=1.0,
    #     tol_lift=1e-6,
    #     seed=42,
    # )
    # print(f"E-R Stackelberg elapsed: {time.time() - _t0:.1f}s")

    # This took 2.5 hrs.
    _t0 = time.time()
    erdos_renyi_rte_improvement(
        m=10,
        p_values=np.linspace(0.25, 0.75, 6),
        n_graphs=20,
        budget=20,
        n_init=10,
        n_iter_phys=200,
        alpha_phys=1e-2,
        tol_phys=1e-6,
        n_iter_lift=200,
        alpha_lift=1e-3,
        tol_lift=1e-6,
        eta=0.1,
        seed=42,
        save_path='results/data/erdos_renyi_rte_diffs.npy',
    )
    print(f"E-R RTE elapsed: {time.time() - _t0:.1f}s")

    # Tuning notes (benchmarked on the full w_max=None complete graph, see each
    # function's docstring for details): the physical stage of all three
    # metrics is fast and reliable (~1-4s for a single restart, plus RTE's
    # one-time ~72s JIT compile), and its defaults below are already tuned.
    # The lifted stage is only reliable for Stackelberg (epsilon=0 projection,
    # ~1-3s/solve at n=132); Kemeny's and RTE's lifted stage share a
    # make_project_Q call (epsilon=1e-6) that failed to reach feasibility in
    # 0/32 benchmarked attempts at n=132 -- this is a solver-scale limitation,
    # not something any PGD hyperparameter combination here can fix. Expect
    # san_francisco_kemeny_improvement/san_francisco_rte_improvement's lifted
    # stage to frequently raise RuntimeError at the default w_max=None; pass a
    # small w_max to shrink the degree_lifting budget if you need it to
    # actually complete (pruning did not fully eliminate the issue in testing
    # either, so treat this as exploratory rather than a guaranteed fix).

    # This took ~1-2 mins for the physical stage per trial; the lifted stage
    # is expected to frequently fail (see the tuning notes above and this
    # function's docstring) -- n_iter_lift is kept small (50) accordingly.
    # _t0 = time.time()
    # san_francisco_kemeny_improvement(
    #     n_trials=20,
    #     n_init=5,
    #     n_iter_phys=150,
    #     alpha_phys=1e-3,
    #     tol_phys=1e-6,
    #     n_iter_lift=50,
    #     alpha_lift=1e-3,
    #     tol_lift=1e-6,
    #     max_grad_norm_phys=2000.0,
    #     max_grad_norm_lift=200.0,
    #     seed=43,
    #     save_path='results/data/san_francisco_kemeny_diffs.npy',
    # )
    # print(f"SF Kemeny elapsed: {time.time() - _t0:.1f}s")

    # This is the practical one of the three at the default w_max=None: both
    # stages use epsilon=0 (or no projection tolerance issue), and a full
    # n_trials=10, n_init=3 sweep at these tuned values (alpha_lift=0.3,
    # confirmed better and faster-converging than an earlier alpha_lift=0.1 in
    # benchmarking) is estimated at roughly 30-60 mins based on a ~95s/restart
    # lifted-stage benchmark.
    # _t0 = time.time()
    # A_sf, W_sf, _ = san_francisco_graph()
    # G_sf = nx.from_numpy_array(A_sf * W_sf, create_using=nx.DiGraph)
    # G_sf.remove_edges_from(nx.selfloop_edges(G_sf))
    # tau_sf = nx.diameter(G_sf, weight='weight')
    # san_francisco_stackelberg_improvement(
    #     tau=tau_sf,
    #     n_trials=10,
    #     n_init=3,
    #     n_iter_phys=250,
    #     alpha_phys=1.0,
    #     tol_phys=1e-6,
    #     n_iter_lift=250,
    #     alpha_lift=0.3,
    #     tol_lift=1e-6,
    #     seed=42,
    #     save_path='results/data/san_francisco_stackelberg_diffs.npy',
    # )
    # print(f"SF Stackelberg elapsed: {time.time() - _t0:.1f}s")

    # The physical stage is fast (~72s one-time JIT compile, then ~0.02s/iter);
    # the lifted stage shares Kemeny's unreliable make_project_Q call (see the
    # tuning notes above), so n_iter_lift is kept small (50) and RuntimeError
    # is expected frequently at the default w_max=None.
    # _t0 = time.time()
    # san_francisco_rte_improvement(
    #     n_trials=10,
    #     n_init=3,
    #     n_iter_phys=200,
    #     alpha_phys=1e-2,
    #     tol_phys=1e-6,
    #     n_iter_lift=50,
    #     alpha_lift=1e-3,
    #     tol_lift=1e-6,
    #     eta=0.1,
    #     seed=42,
    #     save_path='results/data/san_francisco_rte_diffs.npy',
    # )
    # print(f"SF RTE elapsed: {time.time() - _t0:.1f}s")

    # # This took 260 mins
    # _t0 = time.time()
    # m = 10
    # kemeny_lifting_budget_sweep(
    #     m=m,
    #     n_graphs=20,
    #     p_range=(0.2, 0.8),
    #     budget_values = [round(1.25*m), round(1.5*m), round(1.75*m), 2*m, round(2.25*m), round(2.5*m), round(2.75*m), 3*m],
    #     n_init=10,
    #     n_iter_phys=150,
    #     alpha_phys=2e-3,
    #     tol_phys=1e-5,
    #     n_iter_lift=150,
    #     alpha_lift=2e-3,
    #     tol_lift=1e-5,
    #     seed=42,
    # )
    # print(f"Lifting budget sweep elapsed: {time.time() - _t0:.1f}s")
