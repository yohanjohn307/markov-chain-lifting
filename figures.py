
from math import tau
import time as time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import joypy

from markov import stationary_distribution, collapsing, ergodic_flow_to_transition, conductance
from metrics import kemeny, lifted_kemeny, stackelberg, lifted_stackelberg, return_time_entropy, lifted_return_time_entropy
from graph import erdos_renyi_graph, random_chain, degree_lifting, san_francisco_graph, prune_long_edges
from optimize import (
    project_Q,
    project_Q_bar,
    _grad_kemeny,
    _grad_lifted_kemeny,
    make_grad_stackelberg,
    make_project_Q_bar,
    make_project_Q,
    projected_gradient_descent,
)


def fig2() -> None:
    """Reproduce Fig. 2: Kemeny constant of the lifted MC vs. transition probability p."""
    pvec = np.linspace(0.5, 1, 100)
    kvec = []
    # compute Kemeny's constant for each value of p
    for p in pvec:
        P = np.array([
            [0,   p,   0,   0,   0,   1-p],
            [1-p, 0,   p,   0,   0,   0  ],
            [0,   1-p, 0,   p,   0,   0  ],
            [0,   0,   1-p, 0,   p,   0  ],
            [0,   0,   0,   1-p, 0,   p  ],
            [p,   0,   0,   0,   1-p, 0  ],
        ])
        kvec.append(kemeny(P))

    fig, ax = plt.subplots()
    ax.plot(pvec, kvec)
    ax.axhline(25/6, linestyle='--', label=r'$K(\bar{P}) = 25/6$')
    ax.set_xlabel(r'$p$')
    ax.set_ylabel(r'$K(P)$')
    ax.legend()
    plt.tight_layout()
    plt.savefig('kemeny_vs_p.pdf')

def fig3() -> None:
    """Reproduce Fig. 3: MLE estimation error vs. trajectory length.

    Proposition 4 guarantees the plug-in estimate of the physical-space MC converges to P_bar,
    so the error should decay to zero as the trajectory length grows.
    """
    p = 0.9
    P = np.array([
        [0,   p,   0,   0,   0,   1-p],
        [1-p, 0,   p,   0,   0,   0  ],
        [0,   1-p, 0,   p,   0,   0  ],
        [0,   0,   1-p, 0,   p,   0  ],
        [0,   0,   0,   1-p, 0,   p  ],
        [p,   0,   0,   0,   1-p, 0  ],
    ])
    # mapping matrix from virtual states to physical states
    V = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ])
    n, m = V.shape
    virtual_to_physical = V.argmax(axis=1).tolist()

    # collapsed transition matrix
    pi = stationary_distribution(P)
    Pbar, _ = collapsing(P, V)

    # simulate the lifted Markov chain
    x0 = np.random.choice(n, p=pi)
    x = [x0]
    for _ in range(10000 - 1):
        x.append(np.random.choice(n, p=P[x[-1]]))
    y = [virtual_to_physical[xi] for xi in x]

    # estimate the underlying Markov chain's transition matrix from the simulated trajectory
    T = np.arange(10, 10000, 10)
    estimation_errors = []
    counts = np.zeros((m, m))
    prev = 0
    for t in T:
        for i in range(prev, t - 1):
            counts[y[i], y[i + 1]] += 1
        prev = t - 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # unvisited rows stay zero after division
        P_est = counts / row_sums
        estimation_errors.append(np.linalg.norm(Pbar - P_est, ord='fro'))

    # plot the estimation errors
    fig, ax = plt.subplots()
    ax.plot(T, estimation_errors)
    ax.set_xlabel(r'Trajectory Length $T$')
    ax.set_ylabel(r'$\|\hat{\bar{P}} - \bar{P}\|_F$')
    plt.tight_layout()
    plt.savefig('estimation_errors.pdf')

def fig4() -> None:
    """Reproduce Fig. 4: empirical mean capture time vs. number of trials for P_bar and lifted P.

    For P_bar the mean converges to M(P_bar); for the lifted P it converges to M^lift(P),
    not M(P), demonstrating that the lifted Kemeny constant is the correct performance metric.
    """
    Pbar = np.array([[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]])
    pi_bar = stationary_distribution(Pbar)
    m = Pbar.shape[0]

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
    n = P.shape[0]
    virtual_to_physical = V.argmax(axis=1)
    pi = stationary_distribution(P)

    N_trials = 10000

    # simulate Pbar: patroller and adversary drawn from pi_bar; count first-passage steps (k >= 1)
    capture_times_bar = []
    for _ in range(N_trials):
        patroller = np.random.choice(m, p=pi_bar)
        adversary = np.random.choice(m, p=pi_bar)
        t = 0
        while True:
            patroller = np.random.choice(m, p=Pbar[patroller])
            t += 1
            if patroller == adversary:
                break
        capture_times_bar.append(t)

    # simulate P: patroller drawn from pi (virtual), adversary drawn from pi_bar (physical)
    capture_times_lift = []
    for _ in range(N_trials):
        patroller = np.random.choice(n, p=pi)
        adversary = np.random.choice(m, p=pi_bar)
        t = 0
        while True:
            patroller = np.random.choice(n, p=P[patroller])
            t += 1
            if virtual_to_physical[patroller] == adversary:
                break
        capture_times_lift.append(t)

    # compute the empirical mean capture times as a function of the number of trials
    trials = np.arange(1, N_trials + 1)
    mean_bar  = np.cumsum(capture_times_bar)  / trials
    mean_lift = np.cumsum(capture_times_lift) / trials

    fig, ax = plt.subplots()
    ax.plot(trials, mean_bar,  label=r'Empirical Mean Capture Time $\bar{P}$')
    ax.plot(trials, mean_lift, label=r'Empirical Mean Capture Time $P$', color='orange')
    ax.axhline(kemeny(Pbar),        linestyle='--', label=r'$K(\bar{P})$')
    ax.axhline(lifted_kemeny(P, V), linestyle='--', label=r'$K^{\mathrm{lift}}(P)$', color='orange')
    ax.axhline(kemeny(P),           linestyle='--', label=r'$K(P)$', color='green')
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Mean Capture Time')
    ax.legend()
    plt.tight_layout()
    plt.savefig('mean_capture_time.pdf')

def fig_random_graph(m: int = 8, p: float = 0.4, seed: int = 0) -> None:
    """Plot a randomly generated Erdős-Rényi graph G(m, p) with self-loops."""
    A = erdos_renyi_graph(m, p, seed=seed)

    G = nx.from_numpy_array(A)
    off_deg = (A - np.diag(np.diag(A))).sum(axis=1).astype(int)
    pos = nx.spring_layout(G, seed=seed)

    fig, ax = plt.subplots(figsize=(6, 6))
    nx.draw_networkx(G, pos=pos, ax=ax,
                     node_color=off_deg, cmap='viridis',
                     node_size=700, font_color='white', font_weight='bold')
    ax.set_title(f'Erdős–Rényi $G({m},\\ {p})$, seed={seed}', pad=12)
    plt.tight_layout()
    plt.savefig('random_graph.pdf')


def fig_erdos_renyi_kemeny_improvement(
    m: int = 6,
    p_values=None,
    n_graphs: int = 20,
    n_init: int = 3,
    n_iter: int = 150,
    alpha: float = 2e-3,
    tol: float = 1e-5,
    eps: float = 1e-6,
    seed: int = 42,
) -> None:
    """Ridgeline plot of Kemeny improvement via degree lifting vs Erdős-Rényi edge probability p.

    For each p, generates n_graphs random connected G(m, p) graphs, each with a random
    target stationary distribution.  For every graph:
      1. Optimises the Kemeny constant in the physical space via PGD (n_init starts).
      2. Applies degree lifting and optimises the lifted Kemeny constant via PGD (n_init starts).
      3. Records M(P_bar*) - M^lift(P*).
    Results are visualised as a joypy ridgeline plot across trials.
    """
    if p_values is None:
        p_values = np.linspace(0.3, 0.8, 4)

    all_diffs: list[list[float]] = []
    all_graphs: list[list[np.ndarray]] = []
    all_Q_bar: list[list[np.ndarray]] = []
    all_Q_lift: list[list[np.ndarray]] = []
    all_times_phys: list[list[float]] = []
    all_times_lift: list[list[float]] = []
    all_V: list[list[np.ndarray]] = []
    all_conductance_lb: list[list[float]] = []

    for p_idx, p in enumerate(p_values):
        rng = np.random.default_rng(seed * 1000 + p_idx)
        diffs_p: list[float] = []
        graphs_p: list[np.ndarray] = []
        Q_bar_p: list[np.ndarray] = []
        Q_lift_p: list[np.ndarray] = []
        times_phys_p: list[float] = []
        times_lift_p: list[float] = []
        V_p: list[np.ndarray] = []
        conductance_lb_p: list[float] = []
        while len(diffs_p) < n_graphs:
            A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))

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

                _t0_phys = time.perf_counter()
                for _ in range(n_init):
                    Q0 = random_chain(A, seed=int(rng.integers(1 << 31)))
                    Q0 = phys_proj(Q0)

                    Q_opt, hist = projected_gradient_descent(
                        Q0,
                        _grad_kemeny,
                        phys_proj,
                        alpha, n_iter, tol,
                    )
                    if hist and hist[-1] < kemeny_phys:
                        kemeny_phys = hist[-1]
                        best_Q_bar = Q_opt
                _t1_phys = time.perf_counter()

            # Conductance lower bound on the Kemeny constant of the optimised
            # physical MC (Corollary 9): K(P_bar) >= 1 / (2 * Phi(P_bar)).
            _, conductance_lb = conductance(ergodic_flow_to_transition(best_Q_bar))

            # ------------------------------------------------------------------
            # 2. Build degree lifting and optimise lifted Kemeny constant
            # ------------------------------------------------------------------
            V = degree_lifting(A)
            A_lift = V @ (best_Q_bar > eps).astype(int) @ V.T

            best_Q_lift: np.ndarray | None = None
            kemeny_lift = np.inf
            lift_proj = make_project_Q(best_Q_bar, V)
            _t0_lift = time.perf_counter()
            for _ in range(n_init):
                Q0_lift = random_chain(A_lift, seed=int(rng.integers(1 << 31)))
                Q0_lift = lift_proj(Q0_lift)

                Q_lift_opt, hist_lift = projected_gradient_descent(
                    Q0_lift,
                    lambda Q, _V=V, _pi=pi_bar: _grad_lifted_kemeny(Q, _V, _pi),
                    lift_proj,
                    alpha, n_iter, tol,
                )
                if hist_lift and hist_lift[-1] < kemeny_lift:
                    kemeny_lift = hist_lift[-1]
                    best_Q_lift = Q_lift_opt
            _t1_lift = time.perf_counter()

            diff = kemeny_phys - kemeny_lift
            if kemeny_phys > 0 and kemeny_lift > 0:
                diffs_p.append(max(diff, 0.0))
                graphs_p.append(A)
                Q_bar_p.append(best_Q_bar)
                Q_lift_p.append(best_Q_lift)
                times_phys_p.append(_t1_phys - _t0_phys)
                times_lift_p.append(_t1_lift - _t0_lift)
                V_p.append(V)
                conductance_lb_p.append(conductance_lb)

        print(
            f"p={p:.2f}: {len(diffs_p)}/{n_graphs} graphs, "
            + (f"improvement = {np.mean(diffs_p):.3f} ± {np.std(diffs_p):.3f}" if diffs_p else ""),
            flush=True,
        )
        all_diffs.append(diffs_p)
        all_graphs.append(graphs_p)
        all_Q_bar.append(Q_bar_p)
        all_Q_lift.append(Q_lift_p)
        all_times_phys.append(times_phys_p)
        all_times_lift.append(times_lift_p)
        all_V.append(V_p)
        all_conductance_lb.append(conductance_lb_p)

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save('erdos_renyi_kemeny_diffs.npy',
            {'p_values': np.array(p_values), 'diffs': all_diffs, 'V': all_V,
             'graphs': all_graphs, 'Q_bar': all_Q_bar, 'Q_lift': all_Q_lift,
             'times_phys': all_times_phys, 'times_lift': all_times_lift,
             'conductance_lb': all_conductance_lb},
            allow_pickle=True)  # type: ignore[arg-type]


def fig_san_francisco_kemeny_improvement(
    w_max: int = 6,
    n_trials: int = 10,
    n_init: int = 3,
    n_iter: int = 150,
    alpha: float = 2e-3,
    tol: float = 1e-5,
    eps: float = 1e-6,
    seed: int = 42,
) -> None:
    """Kemeny improvement via degree lifting on the San Francisco graph (Sec. VII).

    Unlike fig_erdos_renyi_kemeny_improvement, the graph, travel-time weights W,
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
    Note: because the SF graph is complete (11 neighbours per node), degree
    lifting yields a 132-state lifted chain, so this can be considerably slower
    per trial than the sparse Erdős-Rényi case; reduce n_trials/n_init/n_iter for
    a quick look.
    """
    A, W, pi_bar = san_francisco_graph()
    A = prune_long_edges(A, W, threshold=w_max)
    V = degree_lifting(A)
    print(f"San Francisco graph: {A.shape[0]} nodes, {A.sum()} edges, {V.shape[0]} lifted states", flush=True)
    W_lift = V @ W @ V.T

    rng = np.random.default_rng(seed)
    phys_proj = make_project_Q_bar(A, pi_bar)

    diffs: list[float] = []
    Q_bar_list: list[np.ndarray] = []
    Q_lift_list: list[np.ndarray] = []

    for t in range(n_trials):
        best_Q_bar: np.ndarray | None = None
        kemeny_phys = np.inf
        for _ in range(n_init):
            Q0 = random_chain(A, seed=int(rng.integers(1 << 31)))
            Q0 = phys_proj(Q0)

            Q_opt, hist = projected_gradient_descent(
                Q0,
                lambda Q, _W=W: _grad_kemeny(Q, _W),
                phys_proj,
                alpha, n_iter, tol,
            )
            if hist and hist[-1] < kemeny_phys:
                kemeny_phys = hist[-1]
                best_Q_bar = Q_opt

        A_lift = V @ (best_Q_bar > eps).astype(int) @ V.T
        best_Q_lift: np.ndarray | None = None
        kemeny_lift = np.inf
        lift_proj = make_project_Q(best_Q_bar, V)
        for _ in range(n_init):
            Q0_lift = random_chain(A_lift, seed=int(rng.integers(1 << 31)))
            Q0_lift = lift_proj(Q0_lift)

            Q_lift_opt, hist_lift = projected_gradient_descent(
                Q0_lift,
                lambda Q, _V=V, _pi=pi_bar, _W=W_lift: _grad_lifted_kemeny(Q, _V, _pi, _W),
                lift_proj,
                alpha, n_iter, tol,
            )
            if hist_lift and hist_lift[-1] < kemeny_lift:
                kemeny_lift = hist_lift[-1]
                best_Q_lift = Q_lift_opt

        diff = kemeny_phys - kemeny_lift
        if kemeny_phys > 0 and kemeny_lift > 0:
            diffs.append(max(diff, 0.0))
            Q_bar_list.append(best_Q_bar)
            Q_lift_list.append(best_Q_lift)

        print(
            f"trial {t+1}/{n_trials}: K(P_bar*)={kemeny_phys:.3f}, "
            f"K^lift(P*)={kemeny_lift:.3f}, diff={diff:.3f}",
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
             'Q_bar': Q_bar_list, 'Q_lift': Q_lift_list},
            allow_pickle=True)  # type: ignore[arg-type]

    fig, ax = plt.subplots()
    ax.hist(diffs, bins=min(10, max(1, len(diffs))), color='steelblue', edgecolor='white')
    ax.axvline(0, color='gray', lw=0.8, ls=':')
    ax.set_xlabel(r'$K(\bar{P}^*) - K^{\mathrm{lift}}(P^*)$')
    ax.set_ylabel('Count')
    ax.set_title('Kemeny improvement via degree lifting, San Francisco graph')
    plt.tight_layout()
    plt.savefig('san_francisco_kemeny_improvement.pdf')
    plt.savefig('san_francisco_kemeny_improvement.png', dpi=150)
    print("Saved: san_francisco_kemeny_improvement.pdf / .png")


def fig_erdos_renyi_stackelberg_improvement(
    m: int = 6,
    p_values=None,
    n_graphs: int = 20,
    n_init: int = 3,
    n_iter_phys: int = 250,
    alpha_phys: float = 1.0,
    tol_phys: float = 1e-6,
    n_iter_lift: int = 250,
    alpha_lift: float = 0.1,
    tol_lift: float = 1e-6,
    eps: float = 1e-6,
    temp0: float = 1e-3,
    temp_min: float = 1e-6,
    seed: int = 42,
) -> None:
    """Ridgeline plot of Stackelberg game metric improvement via degree lifting vs Erdős-Rényi edge probability p.

    For each p, generates n_graphs random connected G(m, p) graphs.  For every graph:
      1. Maximises the Stackelberg metric in the physical space via PGD (n_init starts,
         n_iter_phys iterations, step size alpha_phys, tolerance tol_phys).
         No stationary distribution constraint; epsilon=0 because irreducibility is
         enforced inherently by the Stackelberg metric.
      2. Applies degree lifting and maximises the lifted Stackelberg metric via PGD
         (n_init starts, n_iter_lift iterations, step size alpha_lift, tolerance tol_lift).
      3. Records J^lift(P*) - J(P_bar*).
    tau is set uniformly to the diameter of each graph (smallest duration guaranteeing
    nonzero capture probability from any node to any other).
    The softmin temperature is annealed geometrically from temp0 to temp_min over the
    physical-space PGD's n_iter_phys iterations (temp_k = temp0 * temp_decay**k,
    temp_decay = (temp_min / temp0)**(1/n_iter_phys)), tightening the softmin -> min
    approximation as optimization progresses while reusing a single compiled JIT kernel
    throughout (see make_grad_stackelberg). The same temp0 -> temp_decay schedule is
    reused as-is for the lifted-space PGD.
    Results are visualised as a joypy ridgeline plot across trials.
    """
    if p_values is None:
        p_values = np.linspace(0.3, 0.8, 4)

    temp_decay = (temp_min / temp0) ** (1 / n_iter_phys)

    all_diffs: list[list[float]] = []
    all_graphs: list[list[np.ndarray]] = []
    all_Q_bar: list[list[np.ndarray]] = []
    all_Q_lift: list[list[np.ndarray]] = []
    all_times_phys: list[list[float]] = []
    all_times_lift: list[list[float]] = []
    all_V: list[list[np.ndarray]] = []

    for p_idx, p in enumerate(p_values):
        rng = np.random.default_rng(seed * 1000 + p_idx)
        diffs_p: list[float] = []
        graphs_p: list[np.ndarray] = []
        Q_bar_p: list[np.ndarray] = []
        Q_lift_p: list[np.ndarray] = []
        times_phys_p: list[float] = []
        times_lift_p: list[float] = []
        V_p: list[np.ndarray] = []
        while len(diffs_p) < n_graphs:
            A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))
            A_no_self = A - np.diag(np.diag(A))
            diameter = nx.diameter(nx.from_numpy_array(A_no_self))
            # set attack duration equal to diameter of graph
            tau = diameter * np.ones(m, dtype=int)

            # ------------------------------------------------------------------
            # 1. Maximise Stackelberg metric in physical space (free stationary
            #    distribution; epsilon=0 since metric enforces irreducibility)
            # ------------------------------------------------------------------
            capt_prob_phys = np.inf  # tracks negated J; lower = better (higher J)
            best_Q_bar: np.ndarray | None = None

            # V defaults to I_m, collapsing _stackelberg_Q to the standard Stackelberg metric
            grad_stackelberg = make_grad_stackelberg(tau)
            phys_proj = make_project_Q_bar(A, pi_bar=None, epsilon=0.0)
            _t0_phys = time.perf_counter()
            for _ in range(n_init):
                Q0 = random_chain(A, seed=int(rng.integers(1 << 31)))
                Q0 = phys_proj(Q0)

                Q_opt, hist = projected_gradient_descent(
                    Q0,
                    lambda Q, lse_temp, f=grad_stackelberg: tuple(-x for x in f(Q, lse_temp)),
                    phys_proj,
                    alpha_phys, n_iter_phys, tol_phys,
                    temp_schedule=lambda k: temp0 * temp_decay ** k,
                )
                if hist and hist[-1] < capt_prob_phys:
                    P_opt = ergodic_flow_to_transition(Q_opt)
                    capt_prob_phys = stackelberg(P_opt, tau)
                    best_Q_bar = Q_opt
            _t1_phys = time.perf_counter()

            # ------------------------------------------------------------------
            # 2. Build degree lifting and maximise lifted Stackelberg metric
            # ------------------------------------------------------------------
            V = degree_lifting(A)
            A_lift = V @ (best_Q_bar > eps).astype(int) @ V.T

            best_Q_lift: np.ndarray | None = None
            capt_prob_lift = np.inf  # tracks negated J^lift
            grad_lifted_stb = make_grad_stackelberg(tau, V)
            lift_proj = make_project_Q(best_Q_bar, V, epsilon=0.0)
            _t0_lift = time.perf_counter()
            for _ in range(n_init):
                Q0_lift = random_chain(A_lift, seed=int(rng.integers(1 << 31)))
                Q0_lift_proj = lift_proj(Q0_lift)

                Q_lift_opt, hist_lift = projected_gradient_descent(
                    Q0_lift_proj,
                    lambda Q, lse_temp, f=grad_lifted_stb: tuple(-x for x in f(Q, lse_temp)),
                    lift_proj,
                    alpha_lift, n_iter_lift, tol_lift,
                    temp_schedule=lambda k: temp0 * temp_decay ** k,
                )
                if hist_lift and hist_lift[-1] < capt_prob_lift:
                    P_lift_opt = ergodic_flow_to_transition(Q_lift_opt)
                    capt_prob_lift = lifted_stackelberg(P_lift_opt, V, tau)
                    best_Q_lift = Q_lift_opt
            _t1_lift = time.perf_counter()

            diff = capt_prob_lift - capt_prob_phys
            # Corollary 7 guarantees diff >= 0; violations indicate non-convergence
            if capt_prob_phys > 0 and capt_prob_lift > 0:
                if diff > 0:
                    diffs_p.append(diff)
                else:
                    diffs_p.append(0.0)
                graphs_p.append(A)
                Q_bar_p.append(best_Q_bar)
                Q_lift_p.append(best_Q_lift)
                times_phys_p.append(_t1_phys - _t0_phys)
                times_lift_p.append(_t1_lift - _t0_lift)
                V_p.append(V)

        print(
            f"p={p:.2f}: {len(diffs_p)}/{n_graphs} graphs, "
            + (f"improvement = {np.mean(diffs_p):.3f} ± {np.std(diffs_p):.3f}" if diffs_p else ""),
            flush=True,
        )
        all_diffs.append(diffs_p)
        all_graphs.append(graphs_p)
        all_Q_bar.append(Q_bar_p)
        all_Q_lift.append(Q_lift_p)
        all_times_phys.append(times_phys_p)
        all_times_lift.append(times_lift_p)
        all_V.append(V_p)

    np.save('erdos_renyi_stackelberg_diffs.npy',
            {'p_values': np.array(p_values), 'diffs': all_diffs, 'V': all_V,
             'times_phys': all_times_phys, 'times_lift': all_times_lift,
             'graphs': all_graphs, 'Q_bar': all_Q_bar, 'Q_lift': all_Q_lift},
            allow_pickle=True)  # type: ignore[arg-type]


if __name__ == "__main__":
    # fig_random_graph()
    # fig2()
    # fig3()
    # fig4()

    # # This took 17 hrs!
    # _t0 = time.time()
    # fig_erdos_renyi_kemeny_improvement(
    #     m=10,
    #     p_values=np.linspace(0.2, 0.8, 7),
    #     n_graphs=20,
    #     n_init=5,
    #     n_iter=100,
    #     alpha=1e-5,
    #     tol=1e-2,
    #     seed=42,
    # )
    # print(f"E-R Kemeny elapsed: {time.time() - _t0:.1f}s")

    _t0 = time.time()
    fig_san_francisco_kemeny_improvement(
        w_max=4,
        n_trials=5,
        n_init=5,
        n_iter=250,
        alpha=1e-5,
        tol=1e-2,
        seed=42,
    )
    print(f"SF Kemeny elapsed: {time.time() - _t0:.1f}s")

    # _t0 = time.time()
    # fig_erdos_renyi_stackelberg_improvement(
    #     m=10,
    #     p_values=np.linspace(0.2, 0.8, 4),
    #     n_graphs=5,
    #     n_init=1,
    #     n_iter_phys=250,
    #     alpha_phys=1.0,
    #     tol_phys=1e-6,
    #     n_iter_lift=250,
    #     alpha_lift=0.1,
    #     tol_lift=1e-6,
    #     seed=42,
    # )
    # print(f"Stackelberg elapsed: {time.time() - _t0:.1f}s")
