
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
    n_init: int = 3,
    n_iter_phys: int = 150,
    alpha_phys: float = 2e-3,
    tol_phys: float = 1e-5,
    n_iter_lift: int = 150,
    alpha_lift: float = 2e-3,
    tol_lift: float = 1e-5,
    seed: int = 42,
) -> None:
    """Ridgeline plot of Kemeny improvement via degree lifting vs Erdős-Rényi edge probability p.

    For each p, generates n_graphs random connected G(m, p) graphs, each with a random
    target stationary distribution.  For every graph:
      1. Optimises the Kemeny constant in the physical space via PGD (n_init starts).
      2. Applies degree lifting and optimises the lifted Kemeny constant via PGD (n_init starts).
      3. Records M(P_bar*) - M^lift(P*).
    Results are visualised as a joypy ridgeline plot across trials. Physical- and
    lifted-space PGD use separate hyperparameters (alpha_phys/n_iter_phys/tol_phys vs.
    alpha_lift/n_iter_lift/tol_lift), matching the split used in
    erdos_renyi_stackelberg_improvement, erdos_renyi_rte_improvement, and
    san_francisco_kemeny_improvement.
    """
    if p_values is None:
        p_values = np.linspace(0.3, 0.8, 4)

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
        while len(diffs_p) < n_graphs:
            A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))
            # A = erdos_renyi_digraph(m, p, seed=int(rng.integers(1 << 31)))

            # ensure reasonable compatibility between stationary distribution and graph
            best_Q_bar: np.ndarray | None = None
            kemeny_phys = np.inf
            while kemeny_phys > 100:

                # Sample a random stationary distribution.  Concentration parameter 5
                # keeps entries bounded away from zero (min entry ≈ 1/(2m) with high prob),
                # which prevents near-singular gradient systems on sparse graphs.
                pi_bar = rng.dirichlet(5 * np.ones(m))

                # ------------------------------------------------------------------
                # 1. Optimise Kemeny constant in physical space
                # ------------------------------------------------------------------
                phys_proj = make_project_Q_bar(A, pi_bar)

                best_n_iters_phys = 0
                _t0_phys = time.perf_counter()
                for _ in range(n_init):
                    Q0 = random_chain(A, seed=int(rng.integers(1 << 31)))
                    Q0 = phys_proj(Q0)

                    try:
                        Q_opt, hist, n_iters_phys = projected_gradient_descent(
                            Q0,
                            _grad_kemeny,
                            phys_proj,
                            alpha_phys, n_iter_phys, tol_phys,
                        )
                    except (np.linalg.LinAlgError, RuntimeError) as e:
                        print(f"  physical PGD init failed ({e}); skipping", flush=True)
                        continue

                    if hist and 0 < hist[-1] < kemeny_phys:
                        kemeny_phys = hist[-1]
                        best_Q_bar = Q_opt
                        best_n_iters_phys = n_iters_phys
                _t1_phys = time.perf_counter()

                if kemeny_phys == np.inf:
                    # Every restart for this (A, pi_bar) pair failed or was
                    # rejected as numerically degenerate; resampling pi_bar
                    # alone is unlikely to help, so draw a fresh graph too.
                    print("  physical PGD failed for every init; resampling graph and pi_bar", flush=True)
                    A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))
                    # A = erdos_renyi_digraph(m, p, seed=int(rng.integers(1 << 31)))

            # Conductance lower bound on the Kemeny constant of the optimised
            # physical MC (Corollary 9): K(P_bar) >= 1 / (2 * Phi(P_bar)).
            _, conductance_lb = conductance(ergodic_flow_to_transition(best_Q_bar))

            # ------------------------------------------------------------------
            # 2. Build degree lifting and optimise lifted Kemeny constant
            # ------------------------------------------------------------------
            A_off = A - np.diag(np.diag(A))
            V = degree_lifting(A, budget=int(np.maximum(A_off.sum(1), A_off.sum(0)).sum()))
            A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T

            best_Q_lift: np.ndarray | None = None
            kemeny_lift = np.inf
            best_n_iters_lift = 0
            lift_proj = make_project_Q(best_Q_bar, V)
            _t0_lift = time.perf_counter()
            for _ in range(n_init):
                Q0_lift = random_chain(A_lift, seed=int(rng.integers(1 << 31)))
                Q0_lift = lift_proj(Q0_lift)

                try:
                    Q_lift_opt, hist_lift, n_iters_lift = projected_gradient_descent(
                        Q0_lift,
                        lambda Q, _V=V, _pi=pi_bar: _grad_lifted_kemeny(Q, _V, _pi),
                        lift_proj,
                        alpha_lift, n_iter_lift, tol_lift,
                    )
                except (np.linalg.LinAlgError, RuntimeError) as e:
                    print(f"  lifted PGD init failed ({e}); skipping", flush=True)
                    continue
                if hist_lift and 0 < hist_lift[-1] < kemeny_lift:
                    kemeny_lift = hist_lift[-1]
                    best_Q_lift = Q_lift_opt
                    best_n_iters_lift = n_iters_lift
            _t1_lift = time.perf_counter()

            diff = kemeny_phys - kemeny_lift
            print(
                f"p={p:.2f}, graph {len(diffs_p) + 1}/{n_graphs}: K(P_bar*)={kemeny_phys:.3f}, "
                f"K^lift(P*)={kemeny_lift:.3f}, diff={diff:.3f}, "
                f"best-run iters: phys={best_n_iters_phys}, lift={best_n_iters_lift}",
                flush=True,
            )

            graphs_p.append(A)
            Q_bar_p.append(best_Q_bar)
            pi_bar_p.append(pi_bar)
            if diff < 0:
                # Lifting underperformed the physical optimum: fall back to the
                # (trivial) identity lifting of P_bar* rather than deploy a
                # worse-performing lifted MC.
                diffs_p.append(0.0)
                Q_lift_p.append(best_Q_bar)
                V_p.append(np.eye(m))
            else:
                diffs_p.append(diff)
                Q_lift_p.append(best_Q_lift)
                V_p.append(V)
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

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save('erdos_renyi_kemeny_diffs.npy',
            {'p_values': np.array(p_values), 'diffs': all_diffs, 'V': all_V,
             'graphs': all_graphs, 'Q_bar': all_Q_bar, 'Q_lift': all_Q_lift,
             'times_phys': all_times_phys, 'times_lift': all_times_lift,
             'iters_phys': all_iters_phys, 'iters_lift': all_iters_lift,
             'conductance_lb': all_conductance_lb, 'pi_bar': all_pi_bar},
            allow_pickle=True)  # type: ignore[arg-type]


def erdos_renyi_stackelberg_improvement(
    m: int = 6,
    p_values=None,
    n_graphs: int = 20,
    n_init: int = 3,
    n_iter_phys: int = 250,
    alpha_phys: float = 1.0,
    tol_phys: float = 1e-6,
    n_iter_lift: int = 250,
    alpha_lift: float = 1.0,
    tol_lift: float = 1e-6,
    temp0: float = 1e-3,
    temp_min: float = 1e-6,
    seed: int = 42,
) -> None:
    """Ridgeline data for Stackelberg improvement via degree lifting vs Erdős-Rényi edge probability p.

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
      2. Maximises the Stackelberg metric J in the physical space via PGD (n_init starts,
         pi_bar free, epsilon=0).
      3. Applies degree lifting and maximises the lifted Stackelberg metric J^lift via PGD
         (n_init starts, epsilon=0).
      4. Records J^lift(P*) - J(P_bar*).
    The softmin temperature is annealed geometrically from temp0 to temp_min over
    n_iter_phys iterations (temp_k = temp0 * temp_decay**k, temp_decay = (temp_min /
    temp0)**(1/n_iter_phys)), reusing the same schedule for the lifted-space PGD, as in
    san_francisco_stackelberg_improvement.
    """
    if p_values is None:
        p_values = np.linspace(0.3, 0.8, 4)

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
        while len(diffs_p) < n_graphs:
            A = erdos_renyi_digraph(m, p, seed=int(rng.integers(1 << 31)))
            # A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))

            # Uniform tau = graph diameter (self-loops excluded; erdos_renyi_graph
            # guarantees connectedness so the diameter is always finite).
            G = nx.from_numpy_array(A - np.diag(np.diag(A)))
            tau = np.full(m, nx.diameter(G), dtype=int)

            # ------------------------------------------------------------------
            # 1. Maximise Stackelberg metric J in physical space
            # ------------------------------------------------------------------
            phys_proj = make_project_Q_bar(A, pi_bar=None, epsilon=0.0)
            grad_stackelberg = make_grad_stackelberg(tau, W=None)

            best_Q_bar: np.ndarray | None = None
            capt_prob_phys = 0.0
            best_n_iters_phys = 0

            _t0_phys = time.perf_counter()
            for _ in range(n_init):
                Q0 = random_chain(A, seed=int(rng.integers(1 << 31)))
                Q0 = phys_proj(Q0)

                Q_opt, hist, n_iters_phys = projected_gradient_descent(
                    Q0,
                    lambda Q, lse_temp, f=grad_stackelberg: tuple(-x for x in f(Q, lse_temp)),
                    phys_proj,
                    alpha_phys, n_iter_phys, tol_phys,
                    temp_schedule=lambda k: temp0 * temp_decay ** k,
                )
                if hist:
                    P_opt = ergodic_flow_to_transition(Q_opt)
                    capt_prob_tmp = stackelberg(P_opt, tau)
                    if capt_prob_tmp > capt_prob_phys:
                        capt_prob_phys = capt_prob_tmp
                        best_Q_bar = Q_opt
                        best_n_iters_phys = n_iters_phys
            _t1_phys = time.perf_counter()

            if capt_prob_phys < 0.01:
                continue

            # ------------------------------------------------------------------
            # 2. Build degree lifting and maximise lifted Stackelberg metric J^lift
            # ------------------------------------------------------------------
            A_off = A - np.diag(np.diag(A))
            V = degree_lifting(A, budget=int(np.maximum(A_off.sum(1), A_off.sum(0)).sum()))
            A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T

            best_Q_lift: np.ndarray | None = None
            capt_prob_lift = 0.0
            best_n_iters_lift = 0
            lift_proj = make_project_Q(best_Q_bar, V, epsilon=0.0)
            grad_lifted_stb = make_grad_stackelberg(tau, V)
            _t0_lift = time.perf_counter()
            for _ in range(n_init):
                Q0_lift = random_chain(A_lift, seed=int(rng.integers(1 << 31)))
                Q0_lift = lift_proj(Q0_lift)

                Q_lift_opt, hist_lift, n_iters_lift = projected_gradient_descent(
                    Q0_lift,
                    lambda Q, lse_temp, f=grad_lifted_stb: tuple(-x for x in f(Q, lse_temp)),
                    lift_proj,
                    alpha_lift, n_iter_lift, tol_lift,
                    temp_schedule=lambda k: temp0 * temp_decay ** k,
                )
                if hist_lift:
                    P_lift_opt = ergodic_flow_to_transition(Q_lift_opt)
                    capt_prob_tmp = lifted_stackelberg(P_lift_opt, V, tau)
                    if capt_prob_tmp > capt_prob_lift:
                        capt_prob_lift = capt_prob_tmp
                        best_Q_lift = Q_lift_opt
                        best_n_iters_lift = n_iters_lift
            _t1_lift = time.perf_counter()

            diff = capt_prob_lift - capt_prob_phys
            print(
                f"p={p:.2f}, graph {len(diffs_p) + 1}/{n_graphs}: J(P_bar*)={capt_prob_phys:.3f}, "
                f"J^lift(P*)={capt_prob_lift:.3f}, diff={diff:.3f}, "
                f"best-run iters: phys={best_n_iters_phys}, lift={best_n_iters_lift}",
                flush=True,
            )
            if capt_prob_phys > 0 and capt_prob_lift > 0:
                diffs_p.append(max(diff, 0.0))
                graphs_p.append(A)
                tau_p.append(tau)
                Q_bar_p.append(best_Q_bar)
                if diff < 0:
                    # Lifting underperformed the physical optimum: fall back to the
                    # (trivial) identity lifting of P_bar* rather than deploy a
                    # worse-performing lifted MC.
                    Q_lift_p.append(best_Q_bar)
                    V_p.append(np.eye(m))
                else:
                    Q_lift_p.append(best_Q_lift)
                    V_p.append(V)
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

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save('erdos_renyi_stackelberg_diffs.npy',
            {'p_values': np.array(p_values), 'tau': all_tau, 'diffs': all_diffs, 'V': all_V,
             'graphs': all_graphs, 'Q_bar': all_Q_bar, 'Q_lift': all_Q_lift,
             'times_phys': all_times_phys, 'times_lift': all_times_lift,
             'iters_phys': all_iters_phys, 'iters_lift': all_iters_lift},
            allow_pickle=True)  # type: ignore[arg-type]


def erdos_renyi_rte_improvement(
    m: int = 6,
    p_values=None,
    n_graphs: int = 20,
    n_init: int = 3,
    n_iter_phys: int = 200,
    alpha_phys: float = 1e-2,
    tol_phys: float = 1e-6,
    n_iter_lift: int = 200,
    alpha_lift: float = 1e-3,
    tol_lift: float = 1e-6,
    eta: float = 0.1,
    seed: int = 42,
) -> None:
    """Ridgeline data for RTE improvement via degree lifting vs Erdős-Rényi edge probability p.

    Analogous to erdos_renyi_kemeny_improvement, but for the (unweighted) truncated
    Return-Time Entropy metric H and lifted RTE metric H^lift (Appendix B) instead of the
    Kemeny constant. As in the Kemeny case (and unlike Stackelberg), a target stationary
    distribution pi_bar is sampled and imposed via an equality-constrained projection
    (make_project_Q_bar / make_project_Q with pi_bar fixed): both H and H^lift are only
    well-defined for a fixed pi_bar, since the truncation length K_eta depends on
    pi_bar_min and the lifting constraint V^T Q V = Q_bar requires pi_bar = V^T pi. For
    each p, generates n_graphs random connected G(m, p) graphs. For every graph:
      1. Samples a random target stationary distribution pi_bar ~ Dirichlet(5*1_m),
         resampling until the physical optimum carries a non-negligible entropy signal
         (H(P_bar*) >= 0.01), which guards against pi_bar/graph combinations for which
         every reachable chain is nearly deterministic.
      2. Maximises the truncated RTE metric H in the physical space via PGD (n_init
         starts, gradient from make_grad_rte negated to turn maximisation into the
         minimisation form expected by projected_gradient_descent).
      3. Applies degree lifting and maximises the truncated lifted RTE metric H^lift
         via PGD (n_init starts), reusing the same pi_bar.
      4. Records H^lift(P*) - H(P_bar*).
    eta controls the truncation length K_eta = ceil(1 / (eta * pi_bar_min)) - 1 (Eq. 43
    / 45): smaller eta gives a tighter approximation to the untruncated RTE at the cost
    of a longer unrolled recursion inside the JAX-traced gradient (more expensive
    gradient evaluations per PGD iteration).
    Physical- and lifted-space PGD use separate step sizes (alpha_phys, alpha_lift):
    the lifted objective is sensitive to a larger step, so alpha_lift is an order of
    magnitude smaller than alpha_phys, matching the split used in
    san_francisco_kemeny_improvement.
    """
    if p_values is None:
        p_values = np.linspace(0.3, 0.8, 4)

    all_diffs: list[list[float]] = []
    all_graphs: list[list[np.ndarray]] = []
    all_Q_bar: list[list[np.ndarray]] = []
    all_Q_lift: list[list[np.ndarray]] = []
    all_times_phys: list[list[float]] = []
    all_times_lift: list[list[float]] = []
    all_iters_phys: list[list[int]] = []
    all_iters_lift: list[list[int]] = []
    all_V: list[list[np.ndarray]] = []

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
        while len(diffs_p) < n_graphs:
            # A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))
            A = erdos_renyi_digraph(m, p, seed=int(rng.integers(1 << 31)))

            # ensure the physical optimum carries a meaningful (non-negligible) entropy signal
            best_Q_bar: np.ndarray | None = None
            pi_bar: np.ndarray | None = None
            rte_phys = 0.0
            while rte_phys < 1.0:

                # Sample a random stationary distribution.  Concentration parameter 5
                # keeps entries bounded away from zero (min entry ≈ 1/(2m) with high prob),
                # which keeps K_eta (and thus the unrolled RTE recursion) bounded.
                pi_bar = rng.dirichlet(5 * np.ones(m))

                # ------------------------------------------------------------------
                # 1. Maximise truncated RTE metric H in physical space
                # ------------------------------------------------------------------
                phys_proj = make_project_Q_bar(A, pi_bar)
                grad_rte = make_grad_rte(pi_bar, eta)

                best_n_iters_phys = 0
                _t0_phys = time.perf_counter()
                for _ in range(n_init):
                    Q0 = random_chain(A, seed=int(rng.integers(1 << 31)))
                    Q0 = phys_proj(Q0)

                    Q_opt, hist, n_iters_phys = projected_gradient_descent(
                        Q0,
                        lambda Q, f=grad_rte: tuple(-x for x in f(Q)),
                        phys_proj,
                        alpha_phys, n_iter_phys, tol_phys,
                    )
                    if hist and -hist[-1] > rte_phys:
                        rte_phys = -hist[-1]
                        best_Q_bar = Q_opt
                        best_n_iters_phys = n_iters_phys
                _t1_phys = time.perf_counter()

            # ------------------------------------------------------------------
            # 2. Build degree lifting and maximise lifted RTE metric H^lift
            # ------------------------------------------------------------------
            A_off = A - np.diag(np.diag(A))
            V = degree_lifting(A, budget=int(np.maximum(A_off.sum(1), A_off.sum(0)).sum()))
            A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T

            best_Q_lift: np.ndarray | None = None
            rte_lift = 0.0
            best_n_iters_lift = 0
            lift_proj = make_project_Q(best_Q_bar, V)
            grad_lifted_rte = make_grad_rte(pi_bar, eta, V)
            _t0_lift = time.perf_counter()
            for _ in range(n_init):
                Q0_lift = random_chain(A_lift, seed=int(rng.integers(1 << 31)))
                Q0_lift = lift_proj(Q0_lift)

                Q_lift_opt, hist_lift, n_iters_lift = projected_gradient_descent(
                    Q0_lift,
                    lambda Q, f=grad_lifted_rte: tuple(-x for x in f(Q)),
                    lift_proj,
                    alpha_lift, n_iter_lift, tol_lift,
                )
                if hist_lift and -hist_lift[-1] > rte_lift:
                    rte_lift = -hist_lift[-1]
                    best_Q_lift = Q_lift_opt
                    best_n_iters_lift = n_iters_lift
            _t1_lift = time.perf_counter()

            diff = rte_lift - rte_phys
            print(
                f"p={p:.2f}, graph {len(diffs_p) + 1}/{n_graphs}: H(P_bar*)={rte_phys:.3f}, "
                f"H^lift(P*)={rte_lift:.3f}, diff={diff:.3f}, "
                f"best-run iters: phys={best_n_iters_phys}, lift={best_n_iters_lift}",
                flush=True,
            )
            if rte_phys > 0 and rte_lift > 0:
                diffs_p.append(max(diff, 0.0))
                graphs_p.append(A)
                Q_bar_p.append(best_Q_bar)
                if diff < 0:
                    # Lifting underperformed the physical optimum: fall back to the
                    # (trivial) identity lifting of P_bar* rather than deploy a
                    # worse-performing lifted MC.
                    Q_lift_p.append(best_Q_bar)
                    V_p.append(np.eye(m))
                else:
                    Q_lift_p.append(best_Q_lift)
                    V_p.append(V)
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

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save('erdos_renyi_rte_diffs.npy',
            {'p_values': np.array(p_values), 'diffs': all_diffs, 'V': all_V,
             'graphs': all_graphs, 'Q_bar': all_Q_bar, 'Q_lift': all_Q_lift,
             'times_phys': all_times_phys, 'times_lift': all_times_lift,
             'iters_phys': all_iters_phys, 'iters_lift': all_iters_lift,
             'eta': eta},
            allow_pickle=True)  # type: ignore[arg-type]


def san_francisco_kemeny_improvement(
    w_max: int = 6,
    n_trials: int = 10,
    n_init: int = 3,
    n_iter_phys: int = 150,
    alpha_phys: float = 1e-3,
    tol_phys: float = 1e-6,
    n_iter_lift: int = 150,
    alpha_lift: float = 1e-3,
    tol_lift: float = 1e-6,
    seed: int = 42,
) -> None:
    """Kemeny improvement via degree lifting on the San Francisco graph (Sec. VII).

    Unlike erdos_renyi_kemeny_improvement, the graph, travel-time weights W,
    and target stationary distribution pi_bar are all fixed real-world data from
    graph.san_francisco_graph() (12-node police district graph, complete digraph,
    pi_bar proportional to crime rate) rather than randomly generated. Since the
    graph is fixed, we instead repeat the PGD optimization across n_trials random
    initializations to characterize the variability of the achieved (weighted)
    Kemeny and lifted Kemeny constants:
      1. Optimises the weighted Kemeny constant in the physical space via PGD
         (n_init starts), for pi_bar fixed.
      2. Applies degree lifting and optimises the weighted lifted Kemeny constant
         via PGD (n_init starts).
      3. Records K(P_bar*) - K^lift(P*).
    """
    A, W, pi_bar = san_francisco_graph()
    A = prune_long_edges(A, W, threshold=w_max)
    A_off = A - np.diag(np.diag(A))
    V = degree_lifting(A, budget=int(np.maximum(A_off.sum(1), A_off.sum(0)).sum()))
    print(f"San Francisco graph: {A.shape[0]} nodes, {A.sum()} edges, {V.shape[0]} lifted states", flush=True)
    W_lift = V @ W @ V.T

    rng = np.random.default_rng(seed)
    phys_proj = make_project_Q_bar(A, pi_bar)

    diffs: list[float] = []
    Q_bar_list: list[np.ndarray] = []
    Q_lift_list: list[np.ndarray] = []
    iters_phys_list: list[int] = []
    iters_lift_list: list[int] = []

    for t in range(n_trials):
        best_Q_bar: np.ndarray | None = None
        kemeny_phys = np.inf
        best_n_iters_phys = 0
        for _ in range(n_init):
            Q0 = random_chain(A, seed=int(rng.integers(1 << 31)))
            Q0 = phys_proj(Q0)

            try:
                Q_opt, hist, n_iters_phys = projected_gradient_descent(
                    Q0,
                    lambda Q, _W=W: _grad_kemeny(Q, _W),
                    phys_proj,
                    alpha_phys, n_iter_phys, tol_phys,
                )
            except (np.linalg.LinAlgError, RuntimeError) as e:
                print(f"  physical PGD init failed ({e}); skipping", flush=True)
                continue
            # A diverged PGD restart (near-singular adjoint solve in _grad_kemeny,
            # or an OSQP projection that failed to converge) can report a spurious
            # negative "Kemeny constant" for an infeasible Q_opt; the true constant
            # is always positive, so reject non-positive values rather than letting
            # a corrupted Q_opt become best_Q_bar.
            if hist and 0 < hist[-1] < kemeny_phys:
                kemeny_phys = hist[-1]
                best_Q_bar = Q_opt
                best_n_iters_phys = n_iters_phys

        if best_Q_bar is None:
            raise RuntimeError(f"all {n_init} physical PGD restarts diverged for trial {t + 1}")

        A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T
        best_Q_lift: np.ndarray | None = None
        kemeny_lift = np.inf
        best_n_iters_lift = 0
        lift_proj = make_project_Q(best_Q_bar, V)
        n_succeeded = 0
        n_attempts = 0
        max_attempts = 4 * n_init
        while n_succeeded < n_init and n_attempts < max_attempts:
            n_attempts += 1
            Q0_lift = random_chain(A_lift, seed=int(rng.integers(1 << 31)))
            Q0_lift = lift_proj(Q0_lift)

            try:
                Q_lift_opt, hist_lift, n_iters_lift = projected_gradient_descent(
                    Q0_lift,
                    lambda Q, _V=V, _pi=pi_bar, _W=W_lift: _grad_lifted_kemeny(Q, _V, _pi, _W),
                    lift_proj,
                    alpha_lift, n_iter_lift, tol_lift,
                )
            except np.linalg.LinAlgError:
                # PGD can occasionally drive a degree_lifting virtual state's
                # stationary probability so close to zero that the linear solves
                # inside _grad_lifted_kemeny go numerically singular. Discard
                # this restart and draw a fresh random init rather than biasing
                # the optimizer away from that (potentially near-optimal) region.
                continue
            n_succeeded += 1
            if hist_lift and hist_lift[-1] < kemeny_lift:
                kemeny_lift = hist_lift[-1]
                best_Q_lift = Q_lift_opt
                best_n_iters_lift = n_iters_lift

        if best_Q_lift is None:
            raise RuntimeError(
                f"trial {t+1}/{n_trials}: all {n_attempts} lifted PGD restarts hit a "
                "singular linear system before any converged"
            )

        diff = kemeny_phys - kemeny_lift
        if kemeny_phys > 0 and kemeny_lift > 0:
            diffs.append(max(diff, 0.0))
            Q_bar_list.append(best_Q_bar)
            Q_lift_list.append(best_Q_lift)
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
    np.save('san_francisco_kemeny_diffs.npy',
            {'diffs': diffs, 'V': V, 'A': A, 'W': W, 'pi_bar': pi_bar,
             'Q_bar': Q_bar_list, 'Q_lift': Q_lift_list,
             'iters_phys': iters_phys_list, 'iters_lift': iters_lift_list},
            allow_pickle=True)  # type: ignore[arg-type]


def san_francisco_stackelberg_improvement(
    tau: int | np.ndarray,
    w_max: int = 6,
    n_trials: int = 10,
    n_init: int = 3,
    n_iter_phys: int = 250,
    alpha_phys: float = 1.0,
    tol_phys: float = 1e-6,
    n_iter_lift: int = 250,
    alpha_lift: float = 0.1,
    tol_lift: float = 1e-6,
    temp0: float = 1e-3,
    temp_min: float = 1e-6,
    seed: int = 42,
) -> None:
    """Stackelberg improvement via degree lifting on the San Francisco graph (Appendix A).

    As in san_francisco_kemeny_improvement, the graph and travel-time weights W
    are fixed real-world data from graph.san_francisco_graph() (12-node police
    district graph, complete digraph). Unlike the Kemeny case, the target
    stationary distribution pi_bar is *not* imposed: neither a stationary
    distribution constraint nor an explicit irreducibility constraint is needed,
    since the Stackelberg metric enforces irreducibility on its own (a reducible
    chain has some entry of Psi equal to zero, which can never be the max). We
    repeat the PGD optimization across n_trials random initializations to
    characterize the variability of the achieved (weighted) Stackelberg and
    lifted Stackelberg metrics:
      1. Maximises the weighted Stackelberg metric J in the physical space via PGD
         (n_init starts), with pi_bar free (project_Q_bar called with pi_bar=None,
         epsilon=0).
      2. Applies degree lifting and maximises the weighted lifted Stackelberg
         metric J^lift via PGD (n_init starts, epsilon=0).
      3. Records J^lift(P*) - J(P_bar*).
    tau[j] is the attack duration at physical node j (Eq. 38); pass either a
    scalar (broadcast to every node) or an array of length m = A.shape[0] = 12.
    A natural choice is the weighted diameter of the (pruned) graph, i.e. the
    longest shortest travel time between any pair of nodes, which is the
    smallest attack duration guaranteeing nonzero capture probability from any
    node to any other. The softmin temperature is annealed geometrically from
    temp0 to temp_min over n_iter_phys iterations (temp_k = temp0 *
    temp_decay**k, temp_decay = (temp_min / temp0)**(1/n_iter_phys)),
    tightening the softmin -> min approximation as optimization progresses
    while reusing a single compiled JIT kernel throughout (see
    make_grad_stackelberg). The same schedule is reused for the lifted-space
    PGD.
    """
    A, W, _ = san_francisco_graph()
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
    iters_phys_list: list[int] = []
    iters_lift_list: list[int] = []

    for t in range(n_trials):
        best_Q_bar: np.ndarray | None = None
        capt_prob_phys = np.inf  # tracks negated J; lower = better (higher J)
        best_n_iters_phys = 0
        for _ in range(n_init):
            Q0 = random_chain(A, seed=int(rng.integers(1 << 31)))
            Q0 = phys_proj(Q0)

            Q_opt, hist, n_iters_phys = projected_gradient_descent(
                Q0,
                lambda Q, lse_temp, f=grad_stackelberg: tuple(-x for x in f(Q, lse_temp)),
                phys_proj,
                alpha_phys, n_iter_phys, tol_phys,
                temp_schedule=lambda k: temp0 * temp_decay ** k,
            )
            if hist and hist[-1] < capt_prob_phys:
                P_opt = ergodic_flow_to_transition(Q_opt)
                capt_prob_phys = stackelberg(P_opt, tau, W)
                best_Q_bar = Q_opt
                best_n_iters_phys = n_iters_phys

        A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T
        best_Q_lift: np.ndarray | None = None
        capt_prob_lift = np.inf  # tracks negated J^lift
        best_n_iters_lift = 0
        lift_proj = make_project_Q(best_Q_bar, V, epsilon=0.0)
        for _ in range(n_init):
            Q0_lift = random_chain(A_lift, seed=int(rng.integers(1 << 31)))
            Q0_lift = lift_proj(Q0_lift)

            Q_lift_opt, hist_lift, n_iters_lift = projected_gradient_descent(
                Q0_lift,
                lambda Q, lse_temp, f=grad_lifted_stb: tuple(-x for x in f(Q, lse_temp)),
                lift_proj,
                alpha_lift, n_iter_lift, tol_lift,
                temp_schedule=lambda k: temp0 * temp_decay ** k,
            )
            if hist_lift and hist_lift[-1] < capt_prob_lift:
                P_lift_opt = ergodic_flow_to_transition(Q_lift_opt)
                capt_prob_lift = lifted_stackelberg(P_lift_opt, V, tau, W_lift)
                best_Q_lift = Q_lift_opt
                best_n_iters_lift = n_iters_lift

        diff = capt_prob_lift - capt_prob_phys
        if capt_prob_phys > 0 and capt_prob_lift > 0:
            diffs.append(max(diff, 0.0))
            Q_bar_list.append(best_Q_bar)
            Q_lift_list.append(best_Q_lift)
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
    np.save('san_francisco_stackelberg_diffs.npy',
            {'diffs': diffs, 'V': V, 'A': A, 'W': W, 'tau': tau,
             'Q_bar': Q_bar_list, 'Q_lift': Q_lift_list,
             'iters_phys': iters_phys_list, 'iters_lift': iters_lift_list},
            allow_pickle=True)  # type: ignore[arg-type]


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
    n_iter_lift: int = 150,
    alpha_lift: float = 2e-3,
    tol_lift: float = 1e-5,
    seed: int = 42,
    save_path: str = 'lifting_budget_sweep.npy',
) -> None:
    """Compare six lifting-budget allocation heuristics across a budget sweep, over many random graphs.

    For each of n_graphs random undirected Erdős-Rényi graphs G(m, p), with p drawn
    uniformly from p_range independently per graph, and a random target stationary
    distribution pi_bar ~ Dirichlet(concentration * 1_m), this:
      1. Optimises the Kemeny constant in the physical space via PGD
         (n_init starts), once per graph, giving a shared baseline K(P_bar*) and
         best_Q_bar reused across every budget/method combination for that graph.
      2. For each budget in budget_values and each of six lifting heuristics
         (uniform, stationary, degree, betweenness, eigenvector, reversible_flow —
         see graph.proportional_lifting and its callers), builds the corresponding
         V of shape (budget, m) and optimises the lifted Kemeny constant
         via PGD (n_init starts).
      3. Records K(P_bar*) - K^lift(P*) for every (budget, method) pair.
    The reversible_flow method requires best_Q_bar (weights x_i = sum_j
    min(q_bar_ij, q_bar_ji)) and so is built freshly per graph using that graph's
    physical optimum, unlike the other four which only depend on A / pi_bar.
    budget_values defaults to multiples of m from m (the minimum valid budget,
    one virtual state per node, i.e. no lifting) to 4m. Per-graph results (the
    sampled p, the graph A, pi_bar, kemeny_phys, best_Q_bar, and the raw V,
    Q_lift, kemeny_lift, and diff for every budget/method combination) are
    collected across all n_graphs and saved to save_path so the distribution of
    improvement for each lifting procedure at each budget can be visualised
    without re-running optimization.
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
        A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))

        print(f"=== Graph {graph_idx + 1}/{n_graphs}: p={p:.4f} ===", flush=True)

        # ------------------------------------------------------------------
        # 1. Optimise the physical Kemeny constant once; shared across every
        #    budget and lifting method compared below, for this graph.
        # ------------------------------------------------------------------
        best_Q_bar: np.ndarray | None = None
        kemeny_phys = np.inf
        while kemeny_phys > 100:
            # ensure reasonable compatibility between stationary distribution and graph
            pi_bar = rng.dirichlet(concentration * np.ones(m))
            phys_proj = make_project_Q_bar(A, pi_bar)

            for _ in range(n_init):
                Q0 = random_chain(A, seed=int(rng.integers(1 << 31)))
                Q0 = phys_proj(Q0)
                try:
                    Q_opt, hist, _ = projected_gradient_descent(
                        Q0,
                        _grad_kemeny,
                        phys_proj,
                        alpha_phys, n_iter_phys, tol_phys,
                    )
                except (np.linalg.LinAlgError, RuntimeError) as e:
                    print(f"  physical PGD init failed ({e}); skipping", flush=True)
                    continue

                if hist and 0 < hist[-1] < kemeny_phys:
                    kemeny_phys = hist[-1]
                    best_Q_bar = Q_opt

            if kemeny_phys == np.inf:
                # Every restart for this (A, pi_bar) pair failed or was
                # rejected as numerically degenerate; resampling pi_bar alone
                # is unlikely to help, so draw a fresh graph too.
                print("  physical PGD failed for every init; resampling graph and pi_bar", flush=True)
                A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))
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
            name: {'budgets': [], 'diffs': [], 'kemeny_lift': [], 'V': [], 'Q_lift': [], 'iters_lift': []}
            for name in lifting_method_names
        }

        for budget in budget_values:
            budget = int(budget)
            for name, make_V in lifting_methods.items():
                V = make_V(budget)
                A_lift = V @ support @ V.T
                lift_proj = make_project_Q(best_Q_bar, V)

                best_Q_lift: np.ndarray | None = None
                kemeny_lift = np.inf
                best_n_iters_lift = 0
                for _ in range(n_init):
                    Q0_lift = random_chain(A_lift, seed=int(rng.integers(1 << 31)))
                    Q0_lift = lift_proj(Q0_lift)
                    try:
                        Q_lift_opt, hist_lift, n_iters_lift = projected_gradient_descent(
                            Q0_lift,
                            lambda Q, _V=V, _pi=pi_bar: _grad_lifted_kemeny(Q, _V, _pi),
                            lift_proj,
                            alpha_lift, n_iter_lift, tol_lift,
                        )
                    except (np.linalg.LinAlgError, RuntimeError) as e:
                        print(f"    [{name}] lifted PGD init failed ({e}); skipping", flush=True)
                        continue
                    if hist_lift and 0 < hist_lift[-1] < kemeny_lift:
                        kemeny_lift = hist_lift[-1]
                        best_Q_lift = Q_lift_opt
                        best_n_iters_lift = n_iters_lift

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
                else:
                    # Lifting underperformed the physical optimum: fall back to the
                    # (trivial) identity lifting of P_bar* rather than deploy a
                    # worse-performing lifted MC, matching the fallback used in
                    # erdos_renyi_kemeny_improvement.
                    results[name]['diffs'].append(0.0)
                    results[name]['kemeny_lift'].append(kemeny_phys)
                    results[name]['V'].append(np.eye(m))
                    results[name]['Q_lift'].append(best_Q_bar)
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
    # # This took 1 hr
    # _t0 = time.time()
    # erdos_renyi_kemeny_improvement(
    #     m=10,
    #     p_values=np.linspace(0.2, 0.8, 7),
    #     n_graphs=20,
    #     n_init=5,
    #     n_iter_phys=100,
    #     alpha_phys=1e-5,
    #     tol_phys=1e-2,
    #     n_iter_lift=100,
    #     alpha_lift=1e-5,
    #     tol_lift=1e-2,
    #     seed=42,
    # )
    # print(f"E-R Kemeny elapsed: {time.time() - _t0:.1f}s")

    # # This took 10 hrs
    # _t0 = time.time()
    # erdos_renyi_stackelberg_improvement(
    #     m=10,
    #     p_values=np.linspace(0.2, 0.8, 7),
    #     n_graphs=20,
    #     n_init=5,
    #     n_iter_phys=250,
    #     alpha_phys=1.0,
    #     tol_phys=1e-5,
    #     n_iter_lift=250,
    #     alpha_lift=0.1,
    #     tol_lift=1e-6,
    #     seed=42,
    # )
    # print(f"E-R Stackelberg elapsed: {time.time() - _t0:.1f}s")

    # # This took 5 hrs
    # _t0 = time.time()
    # erdos_renyi_rte_improvement(
    #     m=10,
    #     p_values=np.linspace(0.2, 0.8, 7),
    #     n_graphs=20,
    #     n_init=5,
    #     n_iter_phys=200,
    #     alpha_phys=1e-2,
    #     tol_phys=1e-6,
    #     n_iter_lift=200,
    #     alpha_lift=1e-3,
    #     tol_lift=1e-6,
    #     eta=0.1,
    #     seed=42,
    # )
    # print(f"E-R RTE elapsed: {time.time() - _t0:.1f}s")

    # # This took ? mins
    # _t0 = time.time()
    # san_francisco_kemeny_improvement(
    #     w_max=6,
    #     n_trials=20,
    #     n_init=5,
    #     n_iter_phys=100,
    #     alpha_phys=1e-3,
    #     tol_phys=1e-6,
    #     n_iter_lift=100,
    #     alpha_lift=1e-3,
    #     tol_lift=1e-6,
    #     seed=43,
    # )
    # print(f"SF Kemeny elapsed: {time.time() - _t0:.1f}s")

    # # This took 30 mins
    # _t0 = time.time()
    # san_francisco_stackelberg_improvement(
    #     tau=13,
    #     w_max=4,
    #     n_trials=5,
    #     n_init=1,
    #     n_iter_phys=250,
    #     alpha_phys=1.0,
    #     tol_phys=1e-6,
    #     n_iter_lift=250,
    #     alpha_lift=0.1,
    #     tol_lift=1e-6,
    #     seed=42,
    # )
    # print(f"SF Stackelberg elapsed: {time.time() - _t0:.1f}s")

    # This took ? mins
    _t0 = time.time()
    m = 10
    kemeny_lifting_budget_sweep(
        m=m,
        n_graphs=20,
        p_range=(0.2, 0.8),
        budget_values = [round(1.5*m), 2*m, round(2.5*m), 3*m, round(3.5*m)],
        n_init=5,
        n_iter_phys=150,
        alpha_phys=2e-3,
        tol_phys=1e-5,
        n_iter_lift=150,
        alpha_lift=2e-3,
        tol_lift=1e-5,
        seed=42,
    )
    print(f"Lifting budget sweep elapsed: {time.time() - _t0:.1f}s")
