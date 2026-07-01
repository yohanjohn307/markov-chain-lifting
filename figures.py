
import time as time
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import joypy

from markov import stationary_distribution, collapsing
from metrics import kemeny, lifted_kemeny, stackelberg, lifted_stackelberg, return_time_entropy, lifted_return_time_entropy
from graph import erdos_renyi_graph, random_chain, degree_lifting
from optimize import (
    project_Q,
    project_Q_bar,
    _grad_kemeny,
    _grad_lifted_kemeny,
    make_grad_lifted_rte,
    make_grad_lifted_stackelberg,
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

    for p_idx, p in enumerate(p_values):
        rng = np.random.default_rng(seed * 1000 + p_idx)
        diffs_p: list[float] = []
        graphs_p: list[np.ndarray] = []
        Q_bar_p: list[np.ndarray] = []
        Q_lift_p: list[np.ndarray] = []
        times_phys_p: list[float] = []
        times_lift_p: list[float] = []
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
                if hist and hist_lift[-1] < kemeny_lift:
                    kemeny_lift = hist_lift[-1]
                    best_Q_lift = Q_lift_opt
            _t1_lift = time.perf_counter()

            diff = kemeny_phys - kemeny_lift
            if kemeny_phys > 0 and kemeny_lift > 0:
                if diff > 0:
                    diffs_p.append(diff)
                else:
                    diffs_p.append(0.0)
                graphs_p.append(A)
                Q_bar_p.append(best_Q_bar)
                Q_lift_p.append(best_Q_lift)
                times_phys_p.append(_t1_phys - _t0_phys)
                times_lift_p.append(_t1_lift - _t0_lift)

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

    # Save raw data so the plot can be re-rendered without re-running optimization
    np.save('erdos_renyi_kemeny_diffs.npy',
            {'p_values': np.array(p_values), 'diffs': all_diffs,
             'graphs': all_graphs, 'Q_bar': all_Q_bar, 'Q_lift': all_Q_lift,
             'times_phys': all_times_phys, 'times_lift': all_times_lift},
            allow_pickle=True)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Ridgeline plot via joypy
    # ------------------------------------------------------------------
    valid_idx = [i for i, d in enumerate(all_diffs) if len(d) >= 2]
    valid_p   = [p_values[i] for i in valid_idx]
    valid_d   = [all_diffs[i] for i in valid_idx]

    # Clip to the 1st–99th percentile of all values to prevent sparse-graph
    # outliers from collapsing the dense-graph distributions to a sliver.
    all_vals = [v for d in valid_d for v in d]
    x_min    = max(0.0, float(np.percentile(all_vals, 1)))
    x_max    = float(np.percentile(all_vals, 99)) * 1.05

    # joypy expects a wide-format DataFrame: one column per ridge, rows are samples.
    # Columns are ordered so the lowest p appears at the top of the figure
    # (joypy draws the first column at the top).
    df = pd.DataFrame(
        {f'$p={p:.2f}$': pd.Series(
            [v for v in d if x_min <= v <= x_max]   # clip outliers per-group
        ) for p, d in zip(valid_p, valid_d)}
    )
    n_ridges = len(valid_p)

    _, axes = joypy.joyplot(
        df,
        figsize=(7, 1.0 + 0.7 * n_ridges),
        colormap=plt.cm.viridis_r,  # type: ignore[attr-defined]
        linecolor='white',
        linewidth=0.8,
        alpha=0.8,
        x_range=[x_min, x_max],
        overlap=0.4,  # type: ignore[arg-type]
    )
    axes[-1].set_xlabel(
        r'$K(\bar{P}^*) - K^{\mathrm{lift}}(P^*)$',
        fontsize=11,
    )
    axes[0].set_title(
        rf'Kemeny constant improvement via degree lifting, $G({m},\,p)$',
        fontsize=11, pad=8,
    )
    # Mark zero on every sub-axis
    for ax in axes:
        ax.axvline(0, color='gray', lw=0.8, ls=':', zorder=0)

    plt.tight_layout()
    plt.savefig('erdos_renyi_kemeny_improvement.pdf', bbox_inches='tight')
    plt.savefig('erdos_renyi_kemeny_improvement.png', dpi=150, bbox_inches='tight')
    print("Saved: erdos_renyi_kemeny_improvement.pdf / .png")


def fig_erdos_renyi_rte_improvement(
    m: int = 6,
    p_values=None,
    n_graphs: int = 20,
    n_init: int = 3,
    n_iter: int = 150,
    alpha: float = 2e-3,
    eta: float = 0.05,
    tol: float = 1e-5,
    seed: int = 42,
) -> None:
    """Ridgeline plot of return-time entropy improvement via degree lifting vs Erdős-Rényi edge probability p.

    For each p, generates n_graphs random connected G(m, p) graphs, each with a random
    target stationary distribution.  For every graph:
      1. Maximises the truncated RTE in the physical space via PGD (n_init starts).
      2. Applies degree lifting and maximises the lifted truncated RTE via PGD (n_init starts).
      3. Records H^lift(P*) - H(P_bar*).
    Results are visualised as a joypy ridgeline plot across trials.
    eta controls the RTE truncation accuracy (Eq. 43, 45); discarded probability is bounded by eta.
    """
    if p_values is None:
        p_values = np.linspace(0.3, 0.8, 6)

    all_diffs: list[list[float]] = []
    all_graphs: list[list[np.ndarray]] = []
    all_Q_bar: list[list[np.ndarray]] = []
    all_Q_lift: list[list[np.ndarray]] = []

    for p_idx, p in enumerate(p_values):
        rng = np.random.default_rng(seed * 1000 + p_idx)
        diffs_p: list[float] = []
        graphs_p: list[np.ndarray] = []
        Q_bar_p: list[np.ndarray] = []
        Q_lift_p: list[np.ndarray] = []
        attempts = 0
        max_attempts = n_graphs * 8
        while len(diffs_p) < n_graphs and attempts < max_attempts:
            attempts += 1
            try:
                A = erdos_renyi_graph(m, p, seed=int(rng.integers(1 << 31)))
                pi_bar = rng.dirichlet(5 * np.ones(m))

                # ------------------------------------------------------------------
                # 1. Maximise RTE in physical space (negate grad_fn to use minimizer)
                # ------------------------------------------------------------------
                best_phys_val = np.inf  # tracks negated RTE; lower = better (higher H)
                best_Q_bar = None

                grad_rte = make_grad_lifted_rte(np.eye(m), pi_bar, eta)
                phys_proj = make_project_Q_bar(A, pi_bar)
                for _ in range(n_init):
                    Q0 = random_chain(A, seed=int(rng.integers(1 << 31)))
                    try:
                        Q0 = phys_proj(Q0)
                    except RuntimeError:
                        continue

                    Q_opt, hist = projected_gradient_descent(
                        Q0,
                        lambda Q, f=grad_rte: tuple(-x for x in f(Q)),
                        phys_proj,
                        alpha, n_iter, tol,
                    )
                    if hist and hist[-1] < best_phys_val:
                        best_phys_val = hist[-1]
                        best_Q_bar = Q_opt

                if best_Q_bar is None:
                    continue

                pi_bar_opt = best_Q_bar.sum(axis=1)
                P_bar_opt = best_Q_bar / pi_bar_opt[:, None]
                rte_phys = return_time_entropy(P_bar_opt, eta)

                # ------------------------------------------------------------------
                # 2. Build degree lifting and maximise lifted RTE
                # ------------------------------------------------------------------
                V = degree_lifting(A)
                A_lift = (V @ np.ceil(np.maximum(best_Q_bar, 0)) @ V.T).astype(int)
                np.fill_diagonal(A_lift, 1)

                best_lift_val = np.inf  # tracks negated lifted RTE
                best_Q_lift = None

                grad_lifted_rte = make_grad_lifted_rte(V, pi_bar_opt, eta)
                lift_proj = make_project_Q(best_Q_bar, V)
                for _ in range(n_init):
                    Q0_lift = random_chain(A_lift, seed=int(rng.integers(1 << 31)))
                    try:
                        Q0_lift_proj = lift_proj(Q0_lift)
                    except RuntimeError:
                        continue

                    Q_lift_opt, hist_lift = projected_gradient_descent(
                        Q0_lift_proj,
                        lambda Q, f=grad_lifted_rte: tuple(-x for x in f(Q)),
                        lift_proj,
                        alpha, n_iter, tol,
                    )
                    if hist_lift and hist_lift[-1] < best_lift_val:
                        best_lift_val = hist_lift[-1]
                        best_Q_lift = Q_lift_opt

                if best_Q_lift is None:
                    continue

                pi_lift_opt = best_Q_lift.sum(axis=1)
                P_lift_opt = best_Q_lift / pi_lift_opt[:, None]
                rte_lift = lifted_return_time_entropy(P_lift_opt, V, eta)

                diff = rte_lift - rte_phys
                # Corollary 7 guarantees diff >= 0; violations indicate non-convergence
                if rte_phys > 0 and rte_lift > 0:
                    if diff > 0:
                        diffs_p.append(diff)
                    else:
                        diffs_p.append(0.0)
                    graphs_p.append(A)
                    Q_bar_p.append(best_Q_bar)
                    Q_lift_p.append(best_Q_lift)

            except Exception as e:
                print(f"  p={p:.2f}: trial error: {e}", flush=True)

        print(
            f"p={p:.2f}: {len(diffs_p)}/{n_graphs} trials, "
            f"{attempts} attempts"
            + (f",  improvement = {np.mean(diffs_p):.3f} ± {np.std(diffs_p):.3f}" if diffs_p else ""),
            flush=True,
        )
        all_diffs.append(diffs_p)
        all_graphs.append(graphs_p)
        all_Q_bar.append(Q_bar_p)
        all_Q_lift.append(Q_lift_p)

    np.save('erdos_renyi_rte_diffs.npy',
            {'p_values': np.array(p_values), 'diffs': all_diffs,
             'graphs': all_graphs, 'Q_bar': all_Q_bar, 'Q_lift': all_Q_lift},
            allow_pickle=True)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Ridgeline plot via joypy
    # ------------------------------------------------------------------
    valid_idx = [i for i, d in enumerate(all_diffs) if len(d) >= 2]
    valid_p   = [p_values[i] for i in valid_idx]
    valid_d   = [all_diffs[i] for i in valid_idx]

    all_vals = [v for d in valid_d for v in d]
    x_min    = max(0.0, float(np.percentile(all_vals, 1)))
    x_max    = float(np.percentile(all_vals, 99)) * 1.05

    df = pd.DataFrame(
        {f'$p={p:.2f}$': pd.Series(
            [v for v in d if x_min <= v <= x_max]
        ) for p, d in zip(valid_p, valid_d)}
    )
    n_ridges = len(valid_p)

    _, axes = joypy.joyplot(
        df,
        figsize=(7, 1.0 + 0.7 * n_ridges),
        colormap=plt.cm.viridis_r,  # type: ignore[attr-defined]
        linecolor='white',
        linewidth=0.8,
        alpha=0.8,
        x_range=[x_min, x_max],
        overlap=0.4,  # type: ignore[arg-type]
    )
    axes[-1].set_xlabel(
        r'$H^{\mathrm{lift}}(P^*) - H(\bar{P}^*)$',
        fontsize=11,
    )
    axes[0].set_title(
        rf'Return-time entropy improvement via degree lifting, $G({m},\,p)$',
        fontsize=11, pad=8,
    )
    for ax in axes:
        ax.axvline(0, color='gray', lw=0.8, ls=':', zorder=0)

    plt.tight_layout()
    plt.savefig('erdos_renyi_rte_improvement.pdf', bbox_inches='tight')
    plt.savefig('erdos_renyi_rte_improvement.png', dpi=150, bbox_inches='tight')
    print("Saved: erdos_renyi_rte_improvement.pdf / .png")


def fig_erdos_renyi_stackelberg_improvement(
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
    """Ridgeline plot of Stackelberg game metric improvement via degree lifting vs Erdős-Rényi edge probability p.

    For each p, generates n_graphs random connected G(m, p) graphs.  For every graph:
      1. Maximises the Stackelberg metric in the physical space via PGD (n_init starts).
         No stationary distribution constraint; epsilon=0 because irreducibility is
         enforced inherently by the Stackelberg metric.
      2. Applies degree lifting and maximises the lifted Stackelberg metric via PGD (n_init starts).
      3. Records J^lift(P*) - J(P_bar*).
    tau is set uniformly to the diameter of each graph (smallest duration guaranteeing
    nonzero capture probability from any node to any other).
    Results are visualised as a joypy ridgeline plot across trials.
    """
    if p_values is None:
        p_values = np.linspace(0.3, 0.8, 4)

    all_diffs: list[list[float]] = []
    all_graphs: list[list[np.ndarray]] = []
    all_Q_bar: list[list[np.ndarray]] = []
    all_Q_lift: list[list[np.ndarray]] = []

    for p_idx, p in enumerate(p_values):
        rng = np.random.default_rng(seed * 1000 + p_idx)
        diffs_p: list[float] = []
        graphs_p: list[np.ndarray] = []
        Q_bar_p: list[np.ndarray] = []
        Q_lift_p: list[np.ndarray] = []
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

            # V=I_m collapses _lifted_stackelberg_Q to the standard Stackelberg metric
            grad_stackelberg = make_grad_lifted_stackelberg(np.eye(m), tau)
            phys_proj = make_project_Q_bar(A, pi_bar=None, epsilon=0.0)
            for _ in range(n_init):
                Q0 = random_chain(A, seed=int(rng.integers(1 << 31)))
                Q0 = phys_proj(Q0)

                Q_opt, hist = projected_gradient_descent(
                    Q0,
                    lambda Q, f=grad_stackelberg: tuple(-x for x in f(Q)),
                    phys_proj,
                    alpha, n_iter, tol,
                )
                if hist and hist[-1] < capt_prob_phys:
                    capt_prob_phys = hist[-1]
                    best_Q_bar = Q_opt

            # ------------------------------------------------------------------
            # 2. Build degree lifting and maximise lifted Stackelberg metric
            # ------------------------------------------------------------------
            V = degree_lifting(A)
            A_lift = V @ (best_Q_bar > eps).astype(int) @ V.T

            best_Q_lift: np.ndarray | None = None
            capt_prob_lift = np.inf  # tracks negated J^lift
            grad_lifted_stb = make_grad_lifted_stackelberg(V, tau)
            lift_proj = make_project_Q(best_Q_bar, V, epsilon=0.0)
            for _ in range(n_init):
                Q0_lift = random_chain(A_lift, seed=int(rng.integers(1 << 31)))
                Q0_lift_proj = lift_proj(Q0_lift)

                Q_lift_opt, hist_lift = projected_gradient_descent(
                    Q0_lift_proj,
                    lambda Q, f=grad_lifted_stb: tuple(-x for x in f(Q)),
                    lift_proj,
                    alpha, n_iter, tol,
                )
                if hist_lift and hist_lift[-1] < capt_prob_lift:
                    capt_prob_lift = hist_lift[-1]
                    best_Q_lift = Q_lift_opt

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

        print(
            f"p={p:.2f}: {len(diffs_p)}/{n_graphs} graphs, "
            + (f"improvement = {np.mean(diffs_p):.3f} ± {np.std(diffs_p):.3f}" if diffs_p else ""),
            flush=True,
        )
        all_diffs.append(diffs_p)
        all_graphs.append(graphs_p)
        all_Q_bar.append(Q_bar_p)
        all_Q_lift.append(Q_lift_p)

    np.save('erdos_renyi_stackelberg_diffs.npy',
            {'p_values': np.array(p_values), 'diffs': all_diffs,
             'graphs': all_graphs, 'Q_bar': all_Q_bar, 'Q_lift': all_Q_lift},
            allow_pickle=True)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Ridgeline plot via joypy
    # ------------------------------------------------------------------
    valid_idx = [i for i, d in enumerate(all_diffs) if len(d) >= 2]
    valid_p   = [p_values[i] for i in valid_idx]
    valid_d   = [all_diffs[i] for i in valid_idx]

    all_vals = [v for d in valid_d for v in d]
    x_min    = max(0.0, float(np.percentile(all_vals, 1)))
    x_max    = float(np.percentile(all_vals, 99)) * 1.05

    df = pd.DataFrame(
        {f'$p={p:.2f}$': pd.Series(
            [v for v in d if x_min <= v <= x_max]
        ) for p, d in zip(valid_p, valid_d)}
    )
    n_ridges = len(valid_p)

    _, axes = joypy.joyplot(
        df,
        figsize=(7, 1.0 + 0.7 * n_ridges),
        colormap=plt.cm.viridis_r,  # type: ignore[attr-defined]
        linecolor='white',
        linewidth=0.8,
        alpha=0.8,
        x_range=[x_min, x_max],
        overlap=0.4,  # type: ignore[arg-type]
    )
    axes[-1].set_xlabel(
        r'$J^{\mathrm{lift}}(P^*) - J(\bar{P}^*)$',
        fontsize=11,
    )
    axes[0].set_title(
        rf'Stackelberg metric improvement via degree lifting, $G({m},\,p)$',
        fontsize=11, pad=8,
    )
    for ax in axes:
        ax.axvline(0, color='gray', lw=0.8, ls=':', zorder=0)

    plt.tight_layout()
    plt.savefig('erdos_renyi_stackelberg_improvement.pdf', bbox_inches='tight')
    plt.savefig('erdos_renyi_stackelberg_improvement.png', dpi=150, bbox_inches='tight')
    print("Saved: erdos_renyi_stackelberg_improvement.pdf / .png")


if __name__ == "__main__":
    # fig_random_graph()
    # fig2()
    # fig3()
    # fig4()

    # This took 17 hrs!
    _t0 = time.time()
    fig_erdos_renyi_kemeny_improvement(
        m=10,
        p_values=np.linspace(0.2, 0.8, 7),
        n_graphs=20,
        n_init=5,
        n_iter=100,
        alpha=1e-5,
        tol=1e-2,
        seed=42,
    )
    print(f"Kemeny elapsed: {time.time() - _t0:.1f}s")

    # _t0 = time.time()
    # fig_erdos_renyi_rte_improvement(
    #     m=6,
    #     p_values=np.linspace(0.3, 0.8, 3),
    #     n_graphs=5,
    #     n_init=1,
    #     n_iter=100,
    #     alpha=1e-3,
    #     eta=0.5,
    #     tol=1e-5,
    #     seed=42,
    # )
    # print(f"RTE elapsed: {time.time() - _t0:.1f}s")

    # _t0 = time.time()
    # fig_erdos_renyi_stackelberg_improvement(
    #     m=6,
    #     p_values=np.linspace(0.3, 0.8, 3),
    #     n_graphs=5,
    #     n_init=10,
    #     n_iter=100,
    #     alpha=1e-3,
    #     tol=1e-5,
    #     seed=42,
    # )
    # print(f"Stackelberg elapsed: {time.time() - _t0:.1f}s")
