
import os
import time as time
import numpy as np
import networkx as nx

from markov import ergodic_flow_to_transition, conductance, SUPPORT_ATOL
from metrics import stackelberg, lifted_stackelberg
from graph import (
    erdos_renyi_graph,
    random_chain,
    degree_lifting,
    san_francisco_graph,
    ctcv_graph,
    prune_long_edges,
    is_strongly_connected,
    graph_diameter,
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
    """Ridgeline data for Kemeny improvement via stationary-distribution lifting vs.
    Erdős-Rényi edge probability p. For each p, generates n_graphs random connected
    G(m, p) graphs with pi_bar ~ Dirichlet(concentration * 1_m); optimizes K(P_bar*)
    via PGD (resampling on degenerate draws), lifts (budget virtual states, default
    2*m), and optimizes K^lift(P*) via PGD, recording status 'success'/
    'no_improvement'/'all_failed'. See NOTES.md for max_grad_norm_lift tuning.
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
            # 1. Optimise Kemeny constant in physical space
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

            # conductance lower bound on K(P_bar) (Corollary 9): K(P_bar) >= 1/(2*Phi(P_bar))
            _, conductance_lb = conductance(ergodic_flow_to_transition(best_Q_bar))

            # 2. Build stationary-distribution lifting and optimise lifted Kemeny constant
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
                # lifting underperformed: fall back to the identity lifting of P_bar*
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
    """Ridgeline data for Stackelberg improvement via stationary-distribution lifting
    vs. Erdős-Rényi edge probability p. Analogous to erdos_renyi_kemeny_improvement,
    for the (unweighted) Stackelberg metric J / lifted J^lift (Appendix A); pi_bar is
    not imposed (Stackelberg enforces irreducibility on its own). For each graph, sets
    tau[j] = diam(A) at every node (Eq. 38), maximizes J via PGD (resampling the graph
    if J(P_bar*) stays below 0.01), lifts weighted by the realized pi, and maximizes
    J^lift via PGD. The softmin temperature anneals geometrically from temp0 to
    temp_min. See NOTES.md for max_capt_attempts/max_grad_norm tuning rationale.
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
            # 1. Maximise Stackelberg metric J in physical space
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
                # Re-evaluate the exact (non-smooth) metric: softmin(Psi) <= min(Psi)
                # always, so the PGD history value is only a lower bound on J.
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

            # 2. Lift (weighted by best_Q_bar's realized pi) and maximise J^lift
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
                # lifting underperformed: fall back to the identity lifting of P_bar*
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
    """Ridgeline data for RTE improvement via stationary-distribution lifting vs.
    Erdős-Rényi edge probability p. Analogous to erdos_renyi_kemeny_improvement, for
    the (unweighted) truncated Return-Time Entropy H / lifted H^lift (Appendix B);
    unlike Stackelberg, pi_bar is sampled and imposed (H/H^lift are only well-defined
    for a fixed pi_bar). For each graph, samples pi_bar ~ Dirichlet(5*1_m), maximizes
    H via PGD (resampling if H(P_bar*) stays below 1.0), lifts, and maximizes H^lift
    via PGD reusing pi_bar. eta controls truncation length K_eta =
    ceil(1/(eta*pi_bar_min)) - 1 (Eq. 43/45). See NOTES.md for lifting-heuristic and
    gradient-clipping tuning rationale.
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
            # 1. Maximise truncated RTE metric H in physical space
            best_Q_bar: np.ndarray | None = None
            pi_bar: np.ndarray | None = None
            rte_phys = 0.0
            _t0_phys = time.perf_counter()
            for attempt in range(max_rte_attempts):
                A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))
                # A = erdos_renyi_digraph(m, p, seed=int(rng.integers(1 << 31)))

                # concentration=5 keeps entries away from zero (min ≈ 1/(2m) w.h.p.),
                # bounding K_eta and thus the unrolled RTE recursion
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

            # 2. Build stationary-distribution lifting and maximise lifted RTE metric H^lift
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
                # lifting underperformed: fall back to the identity lifting of P_bar*
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
    lifting_budget: int | None = None,
    n_trials: int = 10,
    n_init: int = 3,
    n_iter_phys: int = 150,
    alpha_phys: float = 4e-3,
    tol_phys: float = 1e-5,
    n_iter_lift: int = 150,
    alpha_lift: float = 2e-2,
    tol_lift: float = 1e-5,
    max_grad_norm_phys: float | None = 2000.0,
    max_grad_norm_lift: float | None = 500.0,
    lift_equality_tol: float = 1e-4,
    seed: int = 42,
    save_path: str = 'results/data/san_francisco_kemeny_diffs.npy',
) -> None:
    """Kemeny improvement via stationary-distribution lifting on the San Francisco
    graph (Sec. VII). The graph, travel-time weights W, and pi_bar are fixed
    real-world data from graph.san_francisco_graph(); w_max, if given, prunes edges
    longer than w_max minutes (graph.prune_long_edges). Since the graph is fixed, PGD
    is repeated across n_trials random initializations: (1) optimizes the weighted
    Kemeny constant in the physical space via PGD; (2) applies stationary-distribution
    lifting (budget=lifting_budget, default 3*m) and optimizes the weighted lifted
    Kemeny constant via PGD; (3) records K(P_bar*) - K^lift(P*) with status
    'success'/'no_improvement'/'all_failed' -- unlike the ER sweeps, a non-improving
    trial keeps its actual Q_lift/V rather than an identity fallback. See NOTES.md for
    hyperparameter tuning and lift_equality_tol rationale.
    """
    A, W, pi_bar = san_francisco_graph()
    if w_max is not None:
        A = prune_long_edges(A, W, threshold=w_max)
    m = A.shape[0]
    budget = 3 * m if lifting_budget is None else lifting_budget
    V = stationary_lifting(pi_bar, budget=budget)
    print(f"San Francisco graph: {A.shape[0]} nodes, {A.sum()} edges, {V.shape[0]} lifted states", flush=True)
    W_lift = V @ W @ V.T

    rng = np.random.default_rng(seed)
    phys_proj = make_project_Q_bar(A, pi_bar)
    grad_kemeny = lambda Q, _W=W: _grad_kemeny(Q, _W)

    diffs: list[float] = []
    kemeny_phys_list: list[float] = []
    kemeny_lift_list: list[float] = []
    Q_bar_list: list[np.ndarray] = []
    Q_lift_list: list[np.ndarray] = []
    times_phys: list[float] = []
    times_lift: list[float] = []
    iters_phys_list: list[int] = []
    iters_lift_list: list[int] = []
    status_list: list[str] = []

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
        lift_proj = make_project_Q(best_Q_bar, V, equality_tol=lift_equality_tol)
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
        diffs.append(max(diff, 0.0))
        kemeny_phys_list.append(kemeny_phys)
        kemeny_lift_list.append(kemeny_lift)
        Q_bar_list.append(best_Q_bar)
        Q_lift_list.append(best_Q_lift)
        times_phys.append(_t1_phys - _t0_phys)
        times_lift.append(_t1_lift - _t0_lift)
        iters_phys_list.append(best_n_iters_phys)
        iters_lift_list.append(best_n_iters_lift)
        status_list.append('success' if diff > 0 else ('no_improvement' if n_success_lift > 0 else 'all_failed'))

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

    if kemeny_lift_list:
        best_idx = int(np.argmin(kemeny_lift_list))
        print(
            f"Best lifted MC: K^lift(P*)={kemeny_lift_list[best_idx]:.3f} "
            f"(trial {best_idx + 1}/{n_trials}), corresponding physical MC: "
            f"K(P_bar*)={kemeny_phys_list[best_idx]:.3f}",
            flush=True,
        )

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save(save_path,
            {'diffs': diffs, 'V': V, 'A': A, 'W': W, 'pi_bar': pi_bar,
             'lifting_budget': budget,
             'Q_bar': Q_bar_list, 'Q_lift': Q_lift_list,
             'times_phys': times_phys, 'times_lift': times_lift,
             'iters_phys': iters_phys_list, 'iters_lift': iters_lift_list,
             'status': status_list},
            allow_pickle=True)  # type: ignore[arg-type]


def san_francisco_stackelberg_improvement(
    heterogeneous_tau: bool = False,
    tau: np.ndarray | int | None = None,
    w_max: int | None = None,
    lifting_budget: int | None = None,
    n_trials: int = 10,
    n_init: int = 3,
    n_iter_phys: int = 250,
    alpha_phys: float = 1.0,
    tol_phys: float = 1e-6,
    n_iter_lift: int = 250,
    alpha_lift: float = 2.0,
    tol_lift: float = 1e-6,
    temp0: float = 1e-3,
    temp_min: float = 1e-6,
    max_grad_norm_phys: float | None = None,
    max_grad_norm_lift: float | None = None,
    lift_equality_tol: float = 1e-4,
    seed: int = 42,
    save_path: str = 'results/data/san_francisco_stackelberg_diffs.npy',
) -> None:
    """Stackelberg improvement via stationary-distribution lifting on the San
    Francisco graph (Appendix A). As in san_francisco_kemeny_improvement, the graph
    and W are fixed real-world data; w_max prunes edges longer than w_max minutes.
    pi_bar is *not* imposed (Stackelberg enforces irreducibility on its own). PGD is
    repeated across n_trials random initializations: (1) maximizes J in the physical
    space via PGD, pi_bar free; (2) applies stationary-distribution lifting weighted
    by the *realized* pi (so V/W_lift/grad_lifted_stb are rebuilt every trial) and
    maximizes J^lift via PGD; (3) records diff = J^lift(P*) - J(P_bar*), every trial
    recorded regardless of status.

    tau[j] is the attack duration at node j (Eq. 38); defaults to one of two RoSSO
    benchmark scenarios on this graph, selected via heterogeneous_tau (see NOTES.md).
    Passing tau explicitly overrides heterogeneous_tau, for callers needing a fixed
    attack duration across several calls; sweeps.san_francisco_wmax_sweep does this
    automatically when sweeping w_max. The softmin temperature anneals geometrically
    from temp0 to temp_min. See NOTES.md for lifting_budget/alpha_lift/
    lift_equality_tol tuning rationale and benchmark results.
    """
    A, W, _ = san_francisco_graph()
    if w_max is not None:
        A = prune_long_edges(A, W, threshold=w_max)
    m = A.shape[0]
    budget = 2 * m if lifting_budget is None else lifting_budget
    print(f"San Francisco graph: {A.shape[0]} nodes, {A.sum()} edges, lifting budget={budget} virtual states", flush=True)

    if tau is not None:
        tau = np.full(m, tau, dtype=int) if np.isscalar(tau) else np.asarray(tau, dtype=int)
    else:
        tau = (
            np.array([8, 6, 11, 10, 6, 10, 9, 10, 11, 9, 10, 8], dtype=int)
            if heterogeneous_tau
            else np.full(m, 9, dtype=int)
        )

    temp_decay = (temp_min / temp0) ** (1 / n_iter_phys)

    rng = np.random.default_rng(seed)
    phys_proj = make_project_Q_bar(A, pi_bar=None, epsilon=0.0)
    grad_stackelberg = make_grad_stackelberg(tau, W=W)

    diffs: list[float] = []
    stb_phys_list: list[float] = []
    stb_lift_list: list[float] = []
    Q_bar_list: list[np.ndarray] = []
    Q_lift_list: list[np.ndarray] = []
    V_list: list[np.ndarray] = []
    times_phys: list[float] = []
    times_lift: list[float] = []
    iters_phys_list: list[int] = []
    iters_lift_list: list[int] = []
    status_list: list[str] = []

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
        # Re-evaluate the exact (non-smooth) metric: softmin(Psi) <= min(Psi)
        # always, so the PGD history value is only a lower bound on J.
        capt_prob_phys = stackelberg(ergodic_flow_to_transition(best_Q_bar), tau, W) if best_Q_bar is not None else 0.0
        _t1_phys = time.perf_counter()

        if best_Q_bar is None:
            raise RuntimeError(f"trial {t + 1}/{n_trials}: all physical PGD restarts failed")

        # Lifting weighted by the *realized* pi of the optimized physical MC (see docstring).
        V = stationary_lifting(best_Q_bar.sum(axis=1), budget=budget)
        W_lift = V @ W @ V.T
        A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T
        lift_proj = make_project_Q(best_Q_bar, V, epsilon=0.0, equality_tol=lift_equality_tol)
        grad_lifted_stb = make_grad_stackelberg(tau, V, W_lift)
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
        diffs.append(max(diff, 0.0))
        stb_phys_list.append(capt_prob_phys)
        stb_lift_list.append(capt_prob_lift)
        Q_bar_list.append(best_Q_bar)
        Q_lift_list.append(best_Q_lift)
        V_list.append(V)
        times_phys.append(_t1_phys - _t0_phys)
        times_lift.append(_t1_lift - _t0_lift)
        iters_phys_list.append(best_n_iters_phys)
        iters_lift_list.append(best_n_iters_lift)
        status_list.append('success' if diff > 0 else ('no_improvement' if n_success_lift > 0 else 'all_failed'))

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

    if stb_lift_list:
        best_idx = int(np.argmax(stb_lift_list))
        print(
            f"Best lifted MC: J^lift(P*)={stb_lift_list[best_idx]:.3f} "
            f"(trial {best_idx + 1}/{n_trials}), corresponding physical MC: "
            f"J(P_bar*)={stb_phys_list[best_idx]:.3f}",
            flush=True,
        )

    # V is a per-trial list: each trial's lifting is re-derived from its own realized
    # physical stationary distribution, matching erdos_renyi_stackelberg_improvement's 'V'.
    np.save(save_path,
            {'diffs': diffs, 'V': V_list, 'A': A, 'W': W, 'tau': tau,
             'lifting_budget': budget,
             'Q_bar': Q_bar_list, 'Q_lift': Q_lift_list,
             'times_phys': times_phys, 'times_lift': times_lift,
             'iters_phys': iters_phys_list, 'iters_lift': iters_lift_list,
             'status': status_list},
            allow_pickle=True)  # type: ignore[arg-type]


def san_francisco_rte_improvement(
    w_max: int | None = None,
    lifting_budget: int | None = None,
    n_trials: int = 10,
    n_init: int = 3,
    n_iter_phys: int = 200,
    alpha_phys: float = 1e-2,
    tol_phys: float = 1e-6,
    n_iter_lift: int = 300,
    alpha_lift: float = 1e-3,
    tol_lift: float = 1e-6,
    eta: float = 0.1,
    max_grad_norm_phys: float | None = None,
    max_grad_norm_lift: float | None = 50.0,
    lift_equality_tol: float = 1e-4,
    seed: int = 42,
    save_path: str = 'results/data/san_francisco_rte_diffs.npy',
) -> None:
    """RTE improvement via stationary-distribution lifting on the San Francisco graph
    (Appendix B). As in san_francisco_kemeny_improvement, the graph, W, and pi_bar are
    fixed real-world data; w_max prunes edges longer than w_max minutes. PGD is
    repeated across n_trials random initializations: (1) maximizes the weighted
    truncated RTE metric H in the physical space via PGD, pi_bar fixed; (2) applies
    stationary-distribution lifting (budget=lifting_budget, default 2*m) and
    maximizes the weighted lifted RTE metric H^lift via PGD; (3) records
    H^lift(P*) - H(P_bar*) with the usual status, every trial recorded regardless.

    eta controls the truncation length K_eta = ceil(1/(eta * pi_bar_min)) - 1
    (Eq. 43/45); defaults to 0.1, matching erdos_renyi_rte_improvement. See NOTES.md
    for the max_grad_norm_lift/n_iter_lift/lifting_budget tuning history (this
    function was previously built around unreliable degree lifting) and benchmark
    results, and for lift_equality_tol's rationale.
    """
    A, W, pi_bar = san_francisco_graph()
    if w_max is not None:
        A = prune_long_edges(A, W, threshold=w_max)
    m = A.shape[0]
    budget = 2 * m if lifting_budget is None else lifting_budget
    V = stationary_lifting(pi_bar, budget=budget)
    print(f"San Francisco graph: {A.shape[0]} nodes, {A.sum()} edges, {V.shape[0]} lifted states", flush=True)
    W_lift = V @ W @ V.T

    rng = np.random.default_rng(seed)
    phys_proj = make_project_Q_bar(A, pi_bar)
    grad_rte = make_grad_rte(pi_bar, eta, W=W)
    grad_lifted_rte = make_grad_rte(pi_bar, eta, V, W_lift)

    diffs: list[float] = []
    rte_phys_list: list[float] = []
    rte_lift_list: list[float] = []
    Q_bar_list: list[np.ndarray] = []
    Q_lift_list: list[np.ndarray] = []
    times_phys: list[float] = []
    times_lift: list[float] = []
    iters_phys_list: list[int] = []
    iters_lift_list: list[int] = []
    status_list: list[str] = []

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
        lift_proj = make_project_Q(best_Q_bar, V, equality_tol=lift_equality_tol)
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
        diffs.append(max(diff, 0.0))
        rte_phys_list.append(rte_phys)
        rte_lift_list.append(rte_lift)
        Q_bar_list.append(best_Q_bar)
        Q_lift_list.append(best_Q_lift)
        times_phys.append(_t1_phys - _t0_phys)
        times_lift.append(_t1_lift - _t0_lift)
        iters_phys_list.append(best_n_iters_phys)
        iters_lift_list.append(best_n_iters_lift)
        status_list.append('success' if diff > 0 else ('no_improvement' if n_success_lift > 0 else 'all_failed'))

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

    if rte_lift_list:
        best_idx = int(np.argmax(rte_lift_list))
        print(
            f"Best lifted MC: H^lift(P*)={rte_lift_list[best_idx]:.3f} "
            f"(trial {best_idx + 1}/{n_trials}), corresponding physical MC: "
            f"H(P_bar*)={rte_phys_list[best_idx]:.3f}",
            flush=True,
        )

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save(save_path,
            {'diffs': diffs, 'V': V, 'A': A, 'W': W, 'pi_bar': pi_bar, 'eta': eta,
             'lifting_budget': budget,
             'Q_bar': Q_bar_list, 'Q_lift': Q_lift_list,
             'times_phys': times_phys, 'times_lift': times_lift,
             'iters_phys': iters_phys_list, 'iters_lift': iters_lift_list,
             'status': status_list},
            allow_pickle=True)  # type: ignore[arg-type]


_SF_METRIC_FUNCS = {
    'kemeny': san_francisco_kemeny_improvement,
    'stackelberg': san_francisco_stackelberg_improvement,
    'rte': san_francisco_rte_improvement,
}


def _san_francisco_wmax_sweep_one(
    metric: str,
    w_max_values: list[int],
    save_dir: str,
    **kwargs,
) -> dict:
    """Prune the San Francisco graph at each w_max and call the corresponding
    san_francisco_<metric>_improvement for every pruned graph that stays strongly
    connected, saving its raw results to a distinct per-w_max save_path under save_dir.
    A disconnected w_max is skipped ('infeasible_graph' status) before any PGD is
    attempted; a connected w_max where every PGD restart raises RuntimeError is caught
    as 'optimization_failed' rather than aborting the rest of the sweep. Returns (and
    saves) a manifest dict {'metric', 'w_max_values', 'save_paths', 'status',
    'n_edges', 'manifest_path'}.
    """
    improvement_fn = _SF_METRIC_FUNCS[metric]
    A, W, _ = san_francisco_graph()
    os.makedirs(save_dir, exist_ok=True)

    save_paths: dict[int, str] = {}
    status: dict[int, str] = {}
    n_edges: dict[int, int] = {}

    for w_max in w_max_values:
        A_pruned = prune_long_edges(A, W, threshold=w_max)
        n_edges[w_max] = int(A_pruned.sum())
        if not is_strongly_connected(A_pruned):
            print(
                f"[{metric}] w_max={w_max}: pruned graph not strongly connected "
                f"({n_edges[w_max]} edges), skipping",
                flush=True,
            )
            status[w_max] = 'infeasible_graph'
            continue

        save_path = f'{save_dir}/san_francisco_{metric}_wmax{w_max}.npy'
        print(f"=== [{metric}] w_max={w_max}: {n_edges[w_max]} edges ===", flush=True)
        try:
            improvement_fn(w_max=w_max, save_path=save_path, **kwargs)
            save_paths[w_max] = save_path
            status[w_max] = 'success'
        except RuntimeError as exc:
            print(f"[{metric}] w_max={w_max}: optimization failed ({exc}); skipping", flush=True)
            status[w_max] = 'optimization_failed'

    manifest = {
        'metric': metric, 'w_max_values': np.array(w_max_values),
        'save_paths': save_paths, 'status': status, 'n_edges': n_edges,
    }
    manifest_path = f'{save_dir}/san_francisco_{metric}_wmax_manifest.npy'
    np.save(manifest_path, manifest, allow_pickle=True)  # type: ignore[arg-type]
    return {**manifest, 'manifest_path': manifest_path}


def san_francisco_wmax_sweep(
    w_max_values: list[int] | None = None,
    metrics: tuple[str, ...] = ('kemeny', 'stackelberg', 'rte'),
    save_dir: str = 'results/data/san_francisco_wmax_sweep',
    metric_kwargs: dict[str, dict] | None = None,
    lifting_budget: int | None = None,
) -> dict[str, dict]:
    """Sweep the San Francisco graph's edge-pruning threshold w_max (Sec. VII/IX-C)
    and, for each pruned graph, optimize a physical and lifted MC via the requested
    san_francisco_<metric>_improvement function(s), skipping any disconnected w_max
    and saving a distinct save_path per (metric, w_max) under save_dir. w_max_values
    defaults to [4, 5, 6, 7, 8, 9] (see NOTES.md for why).

    metric_kwargs lets callers pass per-metric keyword overrides (each improvement
    function's tuned defaults assume the *full* unpruned graph); an explicit
    metric_kwargs[metric]['lifting_budget'] or
    metric_kwargs['stackelberg']['tau']/['heterogeneous_tau'] always takes precedence
    over the sweep-level defaults below. lifting_budget, if given, is injected as a
    shared default across every metric in metrics, since each
    san_francisco_<metric>_improvement otherwise has its own tuned default (3*m for
    Kemeny, 2*m for Stackelberg/RTE), making cross-metric comparison unfair. For
    'stackelberg', since its own tau default is w_max-independent (fixed at the
    unpruned graph's diameter), this function computes a single fixed tau from the
    sparsest strongly-connected pruning's diameter (unless the caller already passed
    'tau'/'heterogeneous_tau': True) so every w_max in the sweep stays valid.

    Returns {metric: manifest}, suitable for figures.print_san_francisco_wmax_table.
    """
    if w_max_values is None:
        w_max_values = [4, 5, 6, 7, 8, 9]
    metric_kwargs = {k: dict(v) for k, v in (metric_kwargs or {}).items()}

    if lifting_budget is not None:
        for metric in metrics:
            metric_kwargs.setdefault(metric, {}).setdefault('lifting_budget', lifting_budget)

    if 'stackelberg' in metrics:
        sk_kwargs = metric_kwargs.setdefault('stackelberg', {})
        if 'tau' not in sk_kwargs and not sk_kwargs.get('heterogeneous_tau', False):
            A, W, _ = san_francisco_graph()
            for w in sorted(w_max_values):
                A_pruned = prune_long_edges(A, W, threshold=w)
                if is_strongly_connected(A_pruned):
                    sk_kwargs['tau'] = graph_diameter(A_pruned, W)
                    break

    return {
        metric: _san_francisco_wmax_sweep_one(
            metric, w_max_values, save_dir, **metric_kwargs.get(metric, {})
        )
        for metric in metrics
    }


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
    """Run PGD restarts until n_init succeed or the attempt budget (max_attempts,
    default 2*n_init) runs out; see NOTES.md. make_Q0(seed) draws a fresh raw
    (unprojected) initial point. A restart that raises LinAlgError/RuntimeError draws
    a new random init and retries rather than costing a restart slot. Restarts are
    ranked by the minimum hist[-1] found; callers whose grad_fn negates a maximization
    objective must negate best_val back themselves.
    Returns (best_val, best_Q, best_n_iters, n_success, n_failed): best_val is np.inf
    and best_Q is None if every attempt failed.
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
    """Compare six lifting-budget allocation heuristics (uniform, stationary, degree,
    betweenness, eigenvector, reversible_flow -- see graph.proportional_lifting and
    its callers) across a budget sweep, over many random graphs. For each of n_graphs
    random undirected G(m, p) (p ~ Uniform(p_range), pi_bar ~ Dirichlet(concentration
    * 1_m)): (1) optimizes K(P_bar*) via PGD once per graph as a shared baseline,
    resampling on degenerate draws; (2) for each budget in budget_values and each
    heuristic, builds V and optimizes K^lift(P*) via PGD; (3) records
    K(P_bar*) - K^lift(P*) per (budget, method) with status
    'success'/'no_improvement'/'all_failed'. reversible_flow is built fresh per graph
    from that graph's physical optimum; the other five depend only on A/pi_bar.
    budget_values defaults to multiples of m from m (no lifting) to 4m. See NOTES.md
    for max_grad_norm_lift tuning rationale.
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

        # 1. Optimise the physical Kemeny constant once; shared across every
        #    budget and lifting method compared below, for this graph.
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

            # Poorly-conditioned (A, pi_bar) pairing; resample both together.
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
                    # lifting underperformed: fall back to the identity lifting of P_bar*
                    results[name]['diffs'].append(0.0)
                    results[name]['kemeny_lift'].append(kemeny_phys)
                    results[name]['V'].append(np.eye(m))
                    results[name]['Q_lift'].append(best_Q_bar)
                    # 'all_failed': every retried restart raised (numerical failure).
                    # 'no_improvement': PGD converged but found a worse optimum.
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


def ctcv_kemeny_improvement(
    lifting_budget: int | None = None,
    n_trials: int = 10,
    n_init: int = 3,
    n_iter_phys: int = 150,
    alpha_phys: float = 4e-3,
    tol_phys: float = 1e-5,
    n_iter_lift: int = 150,
    alpha_lift: float = 1e-2,
    tol_lift: float = 1e-5,
    max_grad_norm_phys: float | None = 1000.0,
    max_grad_norm_lift: float | None = 20.0,
    lift_equality_tol: float = 1e-4,
    seed: int = 42,
    save_path: str = 'results/data/ctcv_kemeny_diffs.npy',
) -> None:
    """Kemeny improvement via stationary-distribution lifting on the 18-node CTCV
    campus graph (graph.ctcv_graph()), analogous to san_francisco_kemeny_improvement:
    the graph, travel-distance weights W, and pi_bar (uniform, unlike SF's crime-rate
    weighting) are fixed real-world data; w_max, if given, prunes edges longer than
    w_max meters. PGD is repeated across n_trials random initializations: (1)
    optimizes the weighted Kemeny constant in the physical space via PGD; (2) applies
    stationary-distribution lifting (budget=lifting_budget, default 3*m) and optimizes
    the weighted lifted Kemeny constant via PGD; (3) records K(P_bar*) - K^lift(P*)
    with the usual status, a non-improving trial keeping its actual Q_lift/V.

    CTCV's sparser, leafier topology (vs. SF's complete graph) makes both PGD stages
    far more prone to ill-conditioned-adjoint-solve blowups (see NOTES.md on
    max_grad_norm), so alpha_phys/max_grad_norm_phys/alpha_lift/max_grad_norm_lift
    here were independently tuned via grid sweeps rather than inherited from
    san_francisco_kemeny_improvement's -- see NOTES.md for the tuning history and
    benchmark results, and for lift_equality_tol's rationale.
    """
    A, W, pi_bar = ctcv_graph()
    m = A.shape[0]
    budget = 3 * m if lifting_budget is None else lifting_budget
    V = stationary_lifting(pi_bar, budget=budget)
    print(f"CTCV graph: {A.shape[0]} nodes, {A.sum()} edges, {V.shape[0]} lifted states", flush=True)
    W_lift = V @ W @ V.T

    rng = np.random.default_rng(seed)
    phys_proj = make_project_Q_bar(A, pi_bar)
    grad_kemeny = lambda Q, _W=W: _grad_kemeny(Q, _W)

    diffs: list[float] = []
    kemeny_phys_list: list[float] = []
    kemeny_lift_list: list[float] = []
    Q_bar_list: list[np.ndarray] = []
    Q_lift_list: list[np.ndarray] = []
    times_phys: list[float] = []
    times_lift: list[float] = []
    iters_phys_list: list[int] = []
    iters_lift_list: list[int] = []
    status_list: list[str] = []

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
        lift_proj = make_project_Q(best_Q_bar, V, equality_tol=lift_equality_tol)
        grad_lifted_kemeny = lambda Q, _V=V, _pi=pi_bar, _W=W_lift: _grad_lifted_kemeny(Q, _V, _pi, _W)
        _t0_lift = time.perf_counter()
        kemeny_lift, best_Q_lift, best_n_iters_lift, n_success_lift, _ = _pgd_restarts(
            make_Q0=lambda seed, _A=A_lift: random_chain(_A, seed=seed),
            project_fn=lift_proj,
            grad_fn=grad_lifted_kemeny,
            alpha=alpha_lift, n_iter=n_iter_lift, tol=tol_lift,
            n_init=n_init, rng=rng, label="lifted",
            max_grad_norm=max_grad_norm_lift,
        )
        _t1_lift = time.perf_counter()

        if best_Q_lift is None:
            raise RuntimeError(
                f"trial {t + 1}/{n_trials}: all lifted PGD restarts hit a "
                "singular linear system before any converged"
            )

        diff = kemeny_phys - kemeny_lift
        diffs.append(max(diff, 0.0))
        kemeny_phys_list.append(kemeny_phys)
        kemeny_lift_list.append(kemeny_lift)
        Q_bar_list.append(best_Q_bar)
        Q_lift_list.append(best_Q_lift)
        times_phys.append(_t1_phys - _t0_phys)
        times_lift.append(_t1_lift - _t0_lift)
        iters_phys_list.append(best_n_iters_phys)
        iters_lift_list.append(best_n_iters_lift)
        status_list.append('success' if diff > 0 else ('no_improvement' if n_success_lift > 0 else 'all_failed'))

        print(
            f"trial {t+1}/{n_trials}: K(P_bar*)={kemeny_phys:.3f}, "
            f"K^lift(P*)={kemeny_lift:.3f}, diff={diff:.3f}, "
            f"best-run iters: phys={best_n_iters_phys}, lift={best_n_iters_lift}",
            flush=True,
        )

    print(
        f"CTCV graph: {len(diffs)}/{n_trials} trials, "
        + (f"improvement = {np.mean(diffs):.3f} ± {np.std(diffs):.3f}" if diffs else ""),
        flush=True,
    )

    if kemeny_lift_list:
        best_idx = int(np.argmin(kemeny_lift_list))
        print(
            f"Best lifted MC: K^lift(P*)={kemeny_lift_list[best_idx]:.3f} "
            f"(trial {best_idx + 1}/{n_trials}), corresponding physical MC: "
            f"K(P_bar*)={kemeny_phys_list[best_idx]:.3f}",
            flush=True,
        )

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save(save_path,
            {'diffs': diffs, 'V': V, 'A': A, 'W': W, 'pi_bar': pi_bar,
             'lifting_budget': budget,
             'Q_bar': Q_bar_list, 'Q_lift': Q_lift_list,
             'times_phys': times_phys, 'times_lift': times_lift,
             'iters_phys': iters_phys_list, 'iters_lift': iters_lift_list,
             'status': status_list},
            allow_pickle=True)  # type: ignore[arg-type]


def ctcv_stackelberg_improvement(
    tau: np.ndarray | int | None = None,
    lifting_budget: int | None = None,
    n_trials: int = 10,
    n_init: int = 3,
    n_iter_phys: int = 250,
    alpha_phys: float = 1.0,
    tol_phys: float = 1e-6,
    n_iter_lift: int = 250,
    alpha_lift: float = 2.0,
    tol_lift: float = 1e-6,
    temp0: float = 1e-3,
    temp_min: float = 1e-6,
    max_grad_norm_phys: float | None = None,
    max_grad_norm_lift: float | None = None,
    lift_equality_tol: float = 1e-4,
    seed: int = 42,
    save_path: str = 'results/data/ctcv_stackelberg_diffs.npy',
) -> None:
    """Stackelberg improvement via stationary-distribution lifting on the 18-node
    CTCV campus graph (graph.ctcv_graph()), analogous to
    san_francisco_stackelberg_improvement: the graph and travel-distance weights W are
    fixed real-world data; w_max, if given, prunes edges longer than w_max meters.
    pi_bar is *not* imposed. PGD is repeated across n_trials random initializations:
    (1) maximizes J in the physical space via PGD, pi_bar free; (2) applies
    stationary-distribution lifting weighted by the *realized* pi (V/W_lift/
    grad_lifted_stb rebuilt every trial) and maximizes J^lift via PGD; (3) records
    diff = J^lift(P*) - J(P_bar*), every trial recorded regardless of status.

    tau[j] is the attack duration at node j (Eq. 38). CTCV has no published
    Stackelberg benchmark, so tau defaults to 3 * graph_diameter(A, W) applied
    uniformly (computed after any w_max pruning) -- a grid scan found 1x too tight
    (physical PGD stalls at J(P_bar*) ~= 0) and larger multiples make the lifted
    projection QP fragile; see NOTES.md. Passing tau explicitly overrides this
    default. The softmin temperature anneals geometrically from temp0 to temp_min.

    Hyperparameters (lifting_budget=2*m, alpha_phys=1.0, alpha_lift=2.0,
    max_grad_norm_phys=None, max_grad_norm_lift=None, lift_equality_tol=1e-4) are
    inherited unchanged from san_francisco_stackelberg_improvement -- unlike
    ctcv_kemeny_improvement, an independent grid scan found these already near-optimal
    (see NOTES.md). If a run shows persistent 'all_failed'/'no_improvement', try a
    smaller tau first; tuning alpha/clip further is unlikely to help. The lifted-stage
    _pgd_restarts call raises max_attempts to 8*n_init since individual restarts can
    hit transient QP infeasibility.
    """
    A, W, _ = ctcv_graph()
    m = A.shape[0]
    budget = 2 * m if lifting_budget is None else lifting_budget
    print(f"CTCV graph: {A.shape[0]} nodes, {A.sum()} edges, lifting budget={budget} virtual states", flush=True)

    if tau is not None:
        tau = np.full(m, tau, dtype=int) if np.isscalar(tau) else np.asarray(tau, dtype=int)
    else:
        tau = np.full(m, 3 * graph_diameter(A, W), dtype=int)

    temp_decay = (temp_min / temp0) ** (1 / n_iter_phys)

    rng = np.random.default_rng(seed)
    phys_proj = make_project_Q_bar(A, pi_bar=None, epsilon=0.0)
    grad_stackelberg = make_grad_stackelberg(tau, W=W)

    diffs: list[float] = []
    stb_phys_list: list[float] = []
    stb_lift_list: list[float] = []
    Q_bar_list: list[np.ndarray] = []
    Q_lift_list: list[np.ndarray] = []
    V_list: list[np.ndarray] = []
    times_phys: list[float] = []
    times_lift: list[float] = []
    iters_phys_list: list[int] = []
    iters_lift_list: list[int] = []
    status_list: list[str] = []

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
        # Re-evaluate the exact (non-smooth) metric: softmin(Psi) <= min(Psi)
        # always, so the PGD history value is only a lower bound on J.
        capt_prob_phys = stackelberg(ergodic_flow_to_transition(best_Q_bar), tau, W) if best_Q_bar is not None else 0.0
        _t1_phys = time.perf_counter()

        if best_Q_bar is None:
            raise RuntimeError(f"trial {t + 1}/{n_trials}: all physical PGD restarts failed")

        # Lifting weighted by the *realized* pi of the optimized physical MC (see docstring).
        V = stationary_lifting(best_Q_bar.sum(axis=1), budget=budget)
        W_lift = V @ W @ V.T
        A_lift = V @ (best_Q_bar > SUPPORT_ATOL).astype(int) @ V.T
        lift_proj = make_project_Q(best_Q_bar, V, epsilon=0.0, equality_tol=lift_equality_tol)
        grad_lifted_stb = make_grad_stackelberg(tau, V, W_lift)
        _t0_lift = time.perf_counter()
        _, best_Q_lift, best_n_iters_lift, n_success_lift, _ = _pgd_restarts(
            make_Q0=lambda seed, _A=A_lift: random_chain(_A, seed=seed),
            project_fn=lift_proj,
            grad_fn=lambda Q, lse_temp, f=grad_lifted_stb: tuple(-x for x in f(Q, lse_temp)),
            alpha=alpha_lift, n_iter=n_iter_lift, tol=tol_lift,
            n_init=n_init, rng=rng, label="lifted",
            temp_schedule=lambda k: temp0 * temp_decay ** k,
            max_grad_norm=max_grad_norm_lift, max_attempts=8 * n_init,
        )
        # Re-evaluate exactly, for the same reason as the physical stage above.
        capt_prob_lift = lifted_stackelberg(ergodic_flow_to_transition(best_Q_lift), V, tau, W_lift) if best_Q_lift is not None else 0.0
        _t1_lift = time.perf_counter()

        if best_Q_lift is None:
            raise RuntimeError(f"trial {t + 1}/{n_trials}: all lifted PGD restarts failed")

        diff = capt_prob_lift - capt_prob_phys
        diffs.append(max(diff, 0.0))
        stb_phys_list.append(capt_prob_phys)
        stb_lift_list.append(capt_prob_lift)
        Q_bar_list.append(best_Q_bar)
        Q_lift_list.append(best_Q_lift)
        V_list.append(V)
        times_phys.append(_t1_phys - _t0_phys)
        times_lift.append(_t1_lift - _t0_lift)
        iters_phys_list.append(best_n_iters_phys)
        iters_lift_list.append(best_n_iters_lift)
        status_list.append('success' if diff > 0 else ('no_improvement' if n_success_lift > 0 else 'all_failed'))

        print(
            f"trial {t+1}/{n_trials}: J(P_bar*)={capt_prob_phys:.3f}, "
            f"J^lift(P*)={capt_prob_lift:.3f}, diff={diff:.3f}, "
            f"best-run iters: phys={best_n_iters_phys}, lift={best_n_iters_lift}",
            flush=True,
        )

    print(
        f"CTCV graph: {len(diffs)}/{n_trials} trials, "
        + (f"improvement = {np.mean(diffs):.3f} ± {np.std(diffs):.3f}" if diffs else ""),
        flush=True,
    )

    if stb_lift_list:
        best_idx = int(np.argmax(stb_lift_list))
        print(
            f"Best lifted MC: J^lift(P*)={stb_lift_list[best_idx]:.3f} "
            f"(trial {best_idx + 1}/{n_trials}), corresponding physical MC: "
            f"J(P_bar*)={stb_phys_list[best_idx]:.3f}",
            flush=True,
        )

    # V is a per-trial list: each trial's lifting is re-derived from its own realized
    # physical stationary distribution, matching san_francisco_stackelberg_improvement's 'V'.
    np.save(save_path,
            {'diffs': diffs, 'V': V_list, 'A': A, 'W': W, 'tau': tau,
             'lifting_budget': budget,
             'Q_bar': Q_bar_list, 'Q_lift': Q_lift_list,
             'times_phys': times_phys, 'times_lift': times_lift,
             'iters_phys': iters_phys_list, 'iters_lift': iters_lift_list,
             'status': status_list},
            allow_pickle=True)  # type: ignore[arg-type]


def ctcv_rte_improvement(
    lifting_budget: int | None = None,
    n_trials: int = 10,
    n_init: int = 3,
    n_iter_phys: int = 200,
    alpha_phys: float = 1e-2,
    tol_phys: float = 1e-6,
    n_iter_lift: int = 300,
    alpha_lift: float = 1e-3,
    tol_lift: float = 1e-6,
    eta: float = 0.1,
    max_grad_norm_phys: float | None = None,
    max_grad_norm_lift: float | None = 50.0,
    lift_equality_tol: float = 1e-4,
    seed: int = 42,
    save_path: str = 'results/data/ctcv_rte_diffs.npy',
) -> None:
    """RTE improvement via stationary-distribution lifting on the 18-node CTCV campus
    graph (graph.ctcv_graph()), analogous to san_francisco_rte_improvement: the graph
    and travel-distance weights W are fixed real-world data; w_max, if given, prunes
    edges longer than w_max meters; pi_bar is uniform (no natural target distribution
    exists for this graph). PGD is repeated across n_trials random initializations:
    (1) maximizes the weighted truncated RTE metric H in the physical space via PGD,
    pi_bar fixed; (2) applies stationary-distribution lifting (budget=lifting_budget,
    default 2*m) and maximizes the weighted lifted RTE metric H^lift via PGD; (3)
    records H^lift(P*) - H(P_bar*) with the usual status, every trial recorded
    regardless.

    eta controls the truncation length K_eta = ceil(1/(eta * pi_bar_min)) - 1
    (Eq. 43/45); defaults to 0.1, matching san_francisco_rte_improvement. All
    numerical defaults are inherited unchanged from san_francisco_rte_improvement
    rather than independently retuned -- unlike ctcv_kemeny_improvement, this metric's
    JAX-autodiff entropy gradient was confirmed not to be destabilized by CTCV's
    sparser/leafier topology; see NOTES.md for the validation and budget-sweep
    findings, and benchmark results.
    """
    A, W, pi_bar = ctcv_graph()
    m = A.shape[0]
    budget = 2 * m if lifting_budget is None else lifting_budget
    V = stationary_lifting(pi_bar, budget=budget)
    print(f"CTCV graph: {A.shape[0]} nodes, {A.sum()} edges, {V.shape[0]} lifted states", flush=True)
    W_lift = V @ W @ V.T

    rng = np.random.default_rng(seed)
    phys_proj = make_project_Q_bar(A, pi_bar)
    grad_rte = make_grad_rte(pi_bar, eta, W=W)
    grad_lifted_rte = make_grad_rte(pi_bar, eta, V, W_lift)

    diffs: list[float] = []
    rte_phys_list: list[float] = []
    rte_lift_list: list[float] = []
    Q_bar_list: list[np.ndarray] = []
    Q_lift_list: list[np.ndarray] = []
    times_phys: list[float] = []
    times_lift: list[float] = []
    iters_phys_list: list[int] = []
    iters_lift_list: list[int] = []
    status_list: list[str] = []

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
        lift_proj = make_project_Q(best_Q_bar, V, equality_tol=lift_equality_tol)
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
        diffs.append(max(diff, 0.0))
        rte_phys_list.append(rte_phys)
        rte_lift_list.append(rte_lift)
        Q_bar_list.append(best_Q_bar)
        Q_lift_list.append(best_Q_lift)
        times_phys.append(_t1_phys - _t0_phys)
        times_lift.append(_t1_lift - _t0_lift)
        iters_phys_list.append(best_n_iters_phys)
        iters_lift_list.append(best_n_iters_lift)
        status_list.append('success' if diff > 0 else ('no_improvement' if n_success_lift > 0 else 'all_failed'))

        print(
            f"trial {t+1}/{n_trials}: H(P_bar*)={rte_phys:.3f}, "
            f"H^lift(P*)={rte_lift:.3f}, diff={diff:.3f}, "
            f"best-run iters: phys={best_n_iters_phys}, lift={best_n_iters_lift}",
            flush=True,
        )

    print(
        f"CTCV graph: {len(diffs)}/{n_trials} trials, "
        + (f"improvement = {np.mean(diffs):.3f} ± {np.std(diffs):.3f}" if diffs else ""),
        flush=True,
    )

    if rte_lift_list:
        best_idx = int(np.argmax(rte_lift_list))
        print(
            f"Best lifted MC: H^lift(P*)={rte_lift_list[best_idx]:.3f} "
            f"(trial {best_idx + 1}/{n_trials}), corresponding physical MC: "
            f"H(P_bar*)={rte_phys_list[best_idx]:.3f}",
            flush=True,
        )

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save(save_path,
            {'diffs': diffs, 'V': V, 'A': A, 'W': W, 'pi_bar': pi_bar, 'eta': eta,
             'lifting_budget': budget,
             'Q_bar': Q_bar_list, 'Q_lift': Q_lift_list,
             'times_phys': times_phys, 'times_lift': times_lift,
             'iters_phys': iters_phys_list, 'iters_lift': iters_lift_list,
             'status': status_list},
            allow_pickle=True)  # type: ignore[arg-type]


if __name__ == "__main__":
    # See NOTES.md for the invocations (and approximate wall-clock times) used to
    # generate every saved results file referenced throughout this module.
    _t0 = time.time()
    ctcv_rte_improvement(
        lifting_budget=3 * 18,
        n_trials=10,
        seed=42,
        save_path='results/data/ctcv_rte_diffs.npy',
    )
    print(f"CTCV RTE elapsed: {time.time() - _t0:.1f}s")
