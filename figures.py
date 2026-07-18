import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joypy
import networkx as nx

from markov import ergodic_flow_to_transition, stationary_distribution, collapsing
from metrics import (
    kemeny, lifted_kemeny, stackelberg, lifted_stackelberg,
    return_time_entropy, lifted_return_time_entropy,
)
from graph import erdos_renyi_graph, erdos_renyi_digraph

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 15,
    'legend.title_fontsize': 15,
})


def fig_random_graphs(m: int = 10, p_values=None, seed: int = 42) -> None:
    """Plot randomly generated Erdős-Rényi graphs G(m, p) with self-loops for several p."""
    if p_values is None:
        p_values = [0.2, 0.4, 0.6, 0.8]

    fig, axes = plt.subplots(1, len(p_values), figsize=(6 * len(p_values), 6))

    rng = np.random.default_rng(seed)

    graphs = []
    for p in p_values:
        A = erdos_renyi_graph(m, p, seed=seed)
        G = nx.from_numpy_array(A, create_using=nx.Graph)
        pi_bar = rng.dirichlet(5 * np.ones(m))
        pos = nx.spring_layout(G, seed=seed)
        graphs.append((G, pi_bar, pos))

    vmin = min(pi_bar.min() for _, pi_bar, _ in graphs)
    vmax = max(pi_bar.max() for _, pi_bar, _ in graphs)

    for ax, p, (G, pi_bar, pos) in zip(axes, p_values, graphs):
        nx.draw_networkx(G, pos=pos, ax=ax,
                         node_color=pi_bar, cmap='viridis', vmin=vmin, vmax=vmax,
                         node_size=700, font_color='white', font_weight='bold')
        ax.set_title(f'Erdős–Rényi $G({m},\\ {p})$', pad=12)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05,
                 label='Stationary distribution $\\bar\\pi$')

    plt.savefig('random_graphs.pdf', bbox_inches='tight')
    plt.savefig('random_graphs.png', dpi=150, bbox_inches='tight')


def fig_random_digraphs(m: int = 10, p_values=None, seed: int = 42) -> None:
    """Plot randomly generated Erdős-Rényi digraphs D(m, p) with self-loops for several p."""
    if p_values is None:
        p_values = [0.2, 0.4, 0.6, 0.8]

    fig, axes = plt.subplots(1, len(p_values), figsize=(6 * len(p_values), 6))

    rng = np.random.default_rng(seed)

    graphs = []
    for p in p_values:
        A = erdos_renyi_digraph(m, p, seed=seed)
        G = nx.from_numpy_array(A, create_using=nx.DiGraph)
        pi_bar = rng.dirichlet(5 * np.ones(m))
        pos = nx.spring_layout(G, seed=seed)
        graphs.append((G, pi_bar, pos))

    vmin = min(pi_bar.min() for _, pi_bar, _ in graphs)
    vmax = max(pi_bar.max() for _, pi_bar, _ in graphs)

    for ax, p, (G, pi_bar, pos) in zip(axes, p_values, graphs):
        nx.draw_networkx(G, pos=pos, ax=ax,
                         node_color=pi_bar, cmap='viridis', vmin=vmin, vmax=vmax,
                         node_size=700, font_color='white', font_weight='bold',
                         arrows=True, arrowstyle='-|>', arrowsize=15,
                         connectionstyle='arc3,rad=0.1')
        ax.set_title(f'Erdős–Rényi $D({m},\\ {p})$', pad=12)

    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.05, pad=0.05,
                 label='Stationary distribution $\\bar\\pi$')

    plt.savefig('random_digraphs.pdf', bbox_inches='tight')
    plt.savefig('random_digraphs.png', dpi=150, bbox_inches='tight')
    

def fig_kemeny_vs_transition_probability() -> None:
    """Reproduce Fig. 2: Kemeny constant of the lifted MC vs. transition probability p."""
    pvec = np.linspace(0, 1, 100)
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
    plt.savefig('kemeny_vs_p.png', dpi=150, bbox_inches='tight')


def fig_estimation_error_vs_trajectory_length(seed: int = 42) -> None:
    """Reproduce Fig. 3: MLE estimation error vs. trajectory length.

    Proposition 4 guarantees the plug-in estimate of the physical-space MC converges to P_bar,
    so the error should decay to zero as the trajectory length grows.
    """
    rng = np.random.default_rng(seed)
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
    x0 = rng.choice(n, p=pi)
    x = [x0]
    for _ in range(2000 - 1):
        x.append(rng.choice(n, p=P[x[-1]]))
    y = [virtual_to_physical[xi] for xi in x]

    # estimate the underlying Markov chain's transition matrix from the simulated trajectory
    T = np.arange(10, 2000, 10)
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
    plt.savefig('estimation_errors.png', dpi=150, bbox_inches='tight')


def fig_mean_capture_time_convergence(seed: int = 42) -> None:
    """Reproduce Fig. 4: empirical mean capture time vs. number of trials for P_bar and lifted P.

    For P_bar the mean converges to M(P_bar); for the lifted P it converges to M^lift(P),
    not M(P), demonstrating that the lifted Kemeny constant is the correct performance metric.
    """
    rng = np.random.default_rng(seed)
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

    N_trials = 3000

    # simulate Pbar: patroller and adversary drawn from pi_bar; count first-passage steps (k >= 1)
    capture_times_bar = []
    for _ in range(N_trials):
        patroller = rng.choice(m, p=pi_bar)
        adversary = rng.choice(m, p=pi_bar)
        t = 0
        while True:
            patroller = rng.choice(m, p=Pbar[patroller])
            t += 1
            if patroller == adversary:
                break
        capture_times_bar.append(t)

    # simulate P: patroller drawn from pi (virtual), adversary drawn from pi_bar (physical)
    capture_times_lift = []
    for _ in range(N_trials):
        patroller = rng.choice(n, p=pi)
        adversary = rng.choice(m, p=pi_bar)
        t = 0
        while True:
            patroller = rng.choice(n, p=P[patroller])
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
    plt.savefig('mean_capture_time.png', dpi=150, bbox_inches='tight')


def print_computation_time_table(
    metric_name: str,
    p_values,
    mean_times_phys: list[float],
    mean_times_lift: list[float],
) -> None:
    """Print mean computation times per edge probability p, for copy/pasting into
    Table I of the manuscript (columns: Phys. / Lift. for the given metric).
    """
    print(f"\n--- {metric_name} mean computation time (s) [Table I] ---")
    print(f"{'p':>6} | {'Phys.':>8} | {'Lift.':>8}")
    for p, tp, tl in zip(p_values, mean_times_phys, mean_times_lift):
        print(f"{p:>6.2f} | {tp:>8.2f} | {tl:>8.2f}")
    print("LaTeX rows:")
    for p, tp, tl in zip(p_values, mean_times_phys, mean_times_lift):
        print(f"$p = {p:.1f}$ & {tp:.2f} & {tl:.2f} \\\\")
    print("")


def print_san_francisco_table(
    kemeny_phys: float, kemeny_lift: float,
    stackelberg_phys: float, stackelberg_lift: float,
    rte_phys: float, rte_lift: float,
) -> None:
    """Print the San Francisco case study results (Sec. IX-C), for copy/pasting into
    Table II of the manuscript, alongside the [31]/[33] literature baseline values.

    kemeny_phys/kemeny_lift, stackelberg_phys/stackelberg_lift, and rte_phys/rte_lift
    are the "Proposed" row's values -- typically the best (min for Kemeny, max for
    Stackelberg/RTE) physical/lifted values returned by fig_san_francisco_kemeny,
    fig_san_francisco_stackelberg, and fig_san_francisco_rte respectively.
    [31] (Duan/George/Bullo, max-RTE) and [33] (RoSSO) do not report every metric for
    this benchmark; missing entries are left blank. These baseline numbers are
    literature constants (RoSSO itself is not vendored/runnable here), not recomputed.
    """
    rows = [
        ('Proposed', kemeny_phys, kemeny_lift, stackelberg_phys, stackelberg_lift, rte_phys, rte_lift),
        ('[31]', 24.3, None, None, None, 5.01, None),
        ('[33]', 21.2, None, 9.75e-2, None, 5.13, None),
    ]

    def fmt(x):
        return f"{x:.3g}" if x is not None else "-"

    print("\n--- San Francisco case study mean metric values [Table II] ---")
    print(
        f"{'Method':<10} | {'Kemeny Phys.':>12} | {'Kemeny Lift.':>12} | "
        f"{'Stack. Phys.':>12} | {'Stack. Lift.':>12} | {'RTE Phys.':>10} | {'RTE Lift.':>10}"
    )
    for name, kp, kl, sp, sl, rp, rl in rows:
        print(
            f"{name:<10} | {fmt(kp):>12} | {fmt(kl):>12} | {fmt(sp):>12} | "
            f"{fmt(sl):>12} | {fmt(rp):>10} | {fmt(rl):>10}"
        )
    print("LaTeX rows:")
    for name, kp, kl, sp, sl, rp, rl in rows:
        print(f"{name} & {fmt(kp)} & {fmt(kl)} & {fmt(sp)} & {fmt(sl)} & {fmt(rp)} & {fmt(rl)} \\\\")
    print("")


def fig_erdos_renyi_kemeny_percent_decrease(
    data_path: str = 'results/data/erdos_renyi_kemeny_diffs.npy',
) -> None:
    """Ridgeline plot of the percent decrease in Kemeny's constant achieved by
    stationary-distribution lifting, vs. Erdős-Rényi edge probability p.

    Loads the raw optimization results saved by sweeps.erdos_renyi_kemeny_improvement,
    recomputes K(P_bar*) and K^lift(P*) for every graph from the stored ergodic flow
    matrices (reusing the stored V rather than re-deriving the lifting, so this stays
    correct regardless of which lifting method/budget the sweep used), and reports
    100 * (K(P_bar*) - K^lift(P*)) / K(P_bar*).
    """
    data = np.load(data_path, allow_pickle=True).item()
    p_values = data['p_values']
    all_Q_bar = data['Q_bar']
    all_Q_lift = data['Q_lift']
    all_V = data['V']
    all_times_phys = data['times_phys']
    all_times_lift = data['times_lift']
    all_conductance_lb = data['conductance_lb']

    all_pct: list[list[float]] = []
    mean_times_phys: list[float] = []
    mean_times_lift: list[float] = []
    for p_idx, p in enumerate(p_values):
        pct_p: list[float] = []
        conductance_pct_p: list[float] = []
        for Q_bar, Q_lift, V, conductance_lb in zip(
            all_Q_bar[p_idx], all_Q_lift[p_idx], all_V[p_idx], all_conductance_lb[p_idx]
        ):
            if Q_bar is None or Q_lift is None:
                continue
            P_bar = ergodic_flow_to_transition(Q_bar)
            P_lift = ergodic_flow_to_transition(Q_lift)
            k_phys = kemeny(P_bar)
            k_lift = lifted_kemeny(P_lift, V)
            if k_phys > 0:
                pct_p.append(100.0 * (k_phys - k_lift) / k_phys)
                conductance_pct_p.append(100.0 * (k_phys - conductance_lb) / k_phys)
        times_phys_p = all_times_phys[p_idx]
        times_lift_p = all_times_lift[p_idx]
        mean_times_phys.append(float(np.mean(times_phys_p)) if len(times_phys_p) else float('nan'))
        mean_times_lift.append(float(np.mean(times_lift_p)) if len(times_lift_p) else float('nan'))
        print(
            f"p={p:.2f}: {len(pct_p)} graphs, "
            + (f"percent decrease = {np.mean(pct_p):.2f}% ± {np.std(pct_p):.2f}%, " if pct_p else "")
            + (f"conductance lower bound percent decrease = {np.mean(conductance_pct_p):.2f}% ± "
               f"{np.std(conductance_pct_p):.2f}%, " if conductance_pct_p else "")
            + (f"mean compute time: phys={np.mean(times_phys_p):.2f}s, lift={np.mean(times_lift_p):.2f}s"
               if len(times_phys_p) else ""),
            flush=True,
        )
        all_pct.append(pct_p)

    print_computation_time_table('Kemeny', p_values, mean_times_phys, mean_times_lift)

    # ------------------------------------------------------------------
    # Ridgeline plot via joypy
    # ------------------------------------------------------------------
    valid_idx = [i for i, d in enumerate(all_pct) if len(d) >= 2]
    valid_p = [p_values[i] for i in valid_idx]
    valid_pct = [all_pct[i] for i in valid_idx]

    all_vals = [v for d in valid_pct for v in d]
    x_min = float(np.percentile(all_vals, 1))
    x_max = float(np.percentile(all_vals, 99)) * 1.05

    df = pd.DataFrame(
        {f'$p={p:.2f}$': pd.Series(
            [v for v in d if x_min <= v <= x_max]
        ) for p, d in zip(valid_p, valid_pct)}
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
        r'Decrease in Kemeny Constant [%]',
        fontsize=18,
    )
    for ax in axes:
        ax.axvline(0, color='gray', lw=0.8, ls=':', zorder=0)

    plt.tight_layout()
    plt.savefig('results/erdos_renyi_kemeny_percent_decrease.pdf', bbox_inches='tight')
    plt.savefig('results/erdos_renyi_kemeny_percent_decrease.png', dpi=150, bbox_inches='tight')
    print("Saved: results/erdos_renyi_kemeny_percent_decrease.pdf / .png")


def fig_erdos_renyi_stackelberg_percent_increase(
    data_path: str = 'results/data/erdos_renyi_stackelberg_diffs.npy',
) -> None:
    """Ridgeline plot of the percent increase in the Stackelberg metric achieved by degree
    lifting, vs. Erdős-Rényi edge probability p.

    Loads the raw optimization results saved by figures.fig_erdos_renyi_stackelberg_improvement,
    recomputes J(P_bar*) and J^lift(P*) for every graph from the stored ergodic flow
    matrices, and reports 100 * (J^lift(P*) - J(P_bar*)) / J(P_bar*).
    """
    data = np.load(data_path, allow_pickle=True).item()
    p_values = data['p_values']
    all_tau = data['tau']
    all_Q_bar = data['Q_bar']
    all_Q_lift = data['Q_lift']
    all_V = data['V']
    all_times_phys = data['times_phys']
    all_times_lift = data['times_lift']

    all_pct: list[list[float]] = []
    mean_times_phys: list[float] = []
    mean_times_lift: list[float] = []
    for p_idx, p in enumerate(p_values):
        pct_p: list[float] = []
        for tau, Q_bar, Q_lift, V in zip(
            all_tau[p_idx], all_Q_bar[p_idx], all_Q_lift[p_idx], all_V[p_idx]
        ):
            if Q_bar is None or Q_lift is None:
                continue
            # V is the actual mapping used to produce Q_lift: the realized-stationary-
            # distribution lifting if it outperformed the physical optimum, or the
            # identity fallback (see erdos_renyi_stackelberg_improvement) otherwise.
            P_bar = ergodic_flow_to_transition(Q_bar)
            P_lift = ergodic_flow_to_transition(Q_lift)
            j_phys = stackelberg(P_bar, tau)
            j_lift = lifted_stackelberg(P_lift, V, tau)
            if j_phys > 0:
                pct_p.append(100.0 * (j_lift - j_phys) / j_phys)
        times_phys_p = all_times_phys[p_idx]
        times_lift_p = all_times_lift[p_idx]
        mean_times_phys.append(float(np.mean(times_phys_p)) if len(times_phys_p) else float('nan'))
        mean_times_lift.append(float(np.mean(times_lift_p)) if len(times_lift_p) else float('nan'))
        print(
            f"p={p:.2f}: {len(pct_p)} graphs, "
            + (f"percent increase = {np.mean(pct_p):.2f}% ± {np.std(pct_p):.2f}%, " if pct_p else "")
            + (f"mean compute time: phys={np.mean(times_phys_p):.2f}s, lift={np.mean(times_lift_p):.2f}s"
               if len(times_phys_p) else ""),
            flush=True,
        )
        all_pct.append(pct_p)

    print_computation_time_table('Stackelberg', p_values, mean_times_phys, mean_times_lift)

    # ------------------------------------------------------------------
    # Ridgeline plot via joypy
    # ------------------------------------------------------------------
    valid_idx = [i for i, d in enumerate(all_pct) if len(d) >= 2]
    valid_p = [p_values[i] for i in valid_idx]
    valid_pct = [all_pct[i] for i in valid_idx]

    all_vals = [v for d in valid_pct for v in d]
    x_min = float(np.percentile(all_vals, 1))
    x_max = float(np.percentile(all_vals, 90)) * 1.05

    df = pd.DataFrame(
        {f'$p={p:.2f}$': pd.Series(
            [v for v in d if x_min <= v <= x_max]
        ) for p, d in zip(valid_p, valid_pct)}
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
        r'Increase in Stackelberg Metric [%]',
        fontsize=18,
    )
    for ax in axes:
        ax.axvline(0, color='gray', lw=0.8, ls=':', zorder=0)

    plt.tight_layout()
    plt.savefig('results/erdos_renyi_stackelberg_percent_increase.pdf', bbox_inches='tight')
    plt.savefig('results/erdos_renyi_stackelberg_percent_increase.png', dpi=150, bbox_inches='tight')
    print("Saved: results/erdos_renyi_stackelberg_percent_increase.pdf / .png")


def fig_erdos_renyi_rte_percent_increase(
    data_path: str = 'results/data/erdos_renyi_rte_diffs.npy',
) -> None:
    """Ridgeline plot of the percent increase in the Return-Time Entropy metric achieved by
    stationary-distribution lifting, vs. Erdős-Rényi edge probability p.

    Loads the raw optimization results saved by sweeps.erdos_renyi_rte_improvement,
    recomputes H(P_bar*) and H^lift(P*) for every graph from the stored ergodic flow
    matrices, and reports 100 * (H^lift(P*) - H(P_bar*)) / H(P_bar*).
    """
    data = np.load(data_path, allow_pickle=True).item()
    p_values = data['p_values']
    all_Q_bar = data['Q_bar']
    all_Q_lift = data['Q_lift']
    all_V = data['V']
    all_times_phys = data['times_phys']
    all_times_lift = data['times_lift']
    eta = data['eta']

    all_pct: list[list[float]] = []
    mean_times_phys: list[float] = []
    mean_times_lift: list[float] = []
    for p_idx, p in enumerate(p_values):
        pct_p: list[float] = []
        for Q_bar, Q_lift, V in zip(
            all_Q_bar[p_idx], all_Q_lift[p_idx], all_V[p_idx]
        ):
            if Q_bar is None or Q_lift is None:
                continue
            # V is the actual mapping used to produce Q_lift: the stationary-
            # distribution lifting if it outperformed the physical optimum, or the
            # identity fallback (see erdos_renyi_rte_improvement) otherwise.
            P_bar = ergodic_flow_to_transition(Q_bar)
            P_lift = ergodic_flow_to_transition(Q_lift)
            # Pass pi = Q.sum(axis=1) directly rather than let return_time_entropy
            # re-derive it from P via a fresh linear solve: that solve is exact in
            # principle but can be catastrophically ill-conditioned for chains with
            # very small transition probabilities, even though Q's row sum is exact
            # by construction (Q came from a projected-gradient-descent optimizer
            # whose projection step pins Q.sum(axis=1) to the target pi).
            h_phys = return_time_entropy(P_bar, eta, pi=Q_bar.sum(axis=1))
            h_lift = lifted_return_time_entropy(P_lift, V, eta, pi=Q_lift.sum(axis=1))
            if h_phys > 0:
                pct_p.append(100.0 * (h_lift - h_phys) / h_phys)
        times_phys_p = all_times_phys[p_idx]
        times_lift_p = all_times_lift[p_idx]
        mean_times_phys.append(float(np.mean(times_phys_p)) if len(times_phys_p) else float('nan'))
        mean_times_lift.append(float(np.mean(times_lift_p)) if len(times_lift_p) else float('nan'))
        print(
            f"p={p:.2f}: {len(pct_p)} graphs, "
            + (f"percent increase = {np.mean(pct_p):.2f}% ± {np.std(pct_p):.2f}%, " if pct_p else "")
            + (f"mean compute time: phys={np.mean(times_phys_p):.2f}s, lift={np.mean(times_lift_p):.2f}s"
               if len(times_phys_p) else ""),
            flush=True,
        )
        all_pct.append(pct_p)

    print_computation_time_table('RTE', p_values, mean_times_phys, mean_times_lift)

    # ------------------------------------------------------------------
    # Ridgeline plot via joypy
    # ------------------------------------------------------------------
    valid_idx = [i for i, d in enumerate(all_pct) if len(d) >= 2]
    valid_p = [p_values[i] for i in valid_idx]
    valid_pct = [all_pct[i] for i in valid_idx]

    all_vals = [v for d in valid_pct for v in d]
    x_min = float(np.percentile(all_vals, 1))
    x_max = float(np.percentile(all_vals, 99)) * 1.05

    df = pd.DataFrame(
        {f'$p={p:.2f}$': pd.Series(
            [v for v in d if x_min <= v <= x_max]
        ) for p, d in zip(valid_p, valid_pct)}
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
        r'Increase in Return-Time Entropy [%]',
        fontsize=18,
    )
    for ax in axes:
        ax.axvline(0, color='gray', lw=0.8, ls=':', zorder=0)

    plt.tight_layout()
    plt.savefig('results/erdos_renyi_rte_percent_increase.pdf', bbox_inches='tight')
    plt.savefig('results/erdos_renyi_rte_percent_increase.png', dpi=150, bbox_inches='tight')
    print("Saved: results/erdos_renyi_rte_percent_increase.pdf / .png")


def fig_san_francisco_kemeny(
    data_path: str = 'results/data/san_francisco_kemeny_diffs.npy',
) -> tuple[float, float]:
    """Report the best physical and lifted (weighted) Kemeny constants achieved on the
    San Francisco graph (Sec. IX-C / Table II).

    Loads the raw optimization results saved by sweeps.san_francisco_kemeny_improvement,
    recomputes K(P_bar*) and K^lift(P*) for every trial from the stored ergodic flow
    matrices, and returns (best K(P_bar*), best K^lift(P*)) -- the minimum over all
    n_trials restarts, i.e. the values that belong in Table II's "Proposed" row.
    """
    data = np.load(data_path, allow_pickle=True).item()
    W = data['W']
    V = data['V']
    all_Q_bar = data['Q_bar']
    all_Q_lift = data['Q_lift']

    kemeny_phys_vals: list[float] = []
    kemeny_lift_vals: list[float] = []
    for Q_bar, Q_lift in zip(all_Q_bar, all_Q_lift):
        P_bar = ergodic_flow_to_transition(Q_bar)
        P_lift = ergodic_flow_to_transition(Q_lift)
        kemeny_phys_vals.append(kemeny(P_bar, W, pi=Q_bar.sum(axis=1)))
        kemeny_lift_vals.append(lifted_kemeny(P_lift, V, W, pi=Q_lift.sum(axis=1)))

    best_phys = min(kemeny_phys_vals)
    best_lift = min(kemeny_lift_vals)
    print(
        f"San Francisco Kemeny: {len(kemeny_phys_vals)} trials, "
        f"K(P_bar*) best={best_phys:.3f} mean={np.mean(kemeny_phys_vals):.3f} ± "
        f"{np.std(kemeny_phys_vals):.3f}, K^lift(P*) best={best_lift:.3f} "
        f"mean={np.mean(kemeny_lift_vals):.3f} ± {np.std(kemeny_lift_vals):.3f}",
        flush=True,
    )
    return best_phys, best_lift


def fig_san_francisco_stackelberg(
    data_path: str = 'results/data/san_francisco_stackelberg_diffs.npy',
) -> tuple[float, float]:
    """Report the best physical and lifted (weighted) Stackelberg metrics achieved on
    the San Francisco graph (Appendix A / Table II).

    Loads the raw optimization results saved by sweeps.san_francisco_stackelberg_improvement,
    recomputes J(P_bar*) and J^lift(P*) for every trial from the stored ergodic flow
    matrices, and returns (best J(P_bar*), best J^lift(P*)) -- the maximum over all
    n_trials restarts, i.e. the values that belong in Table II's "Proposed" row.
    """
    data = np.load(data_path, allow_pickle=True).item()
    W = data['W']
    V = data['V']
    tau = data['tau']
    all_Q_bar = data['Q_bar']
    all_Q_lift = data['Q_lift']
    W_lift = V @ W @ V.T

    stb_phys_vals: list[float] = []
    stb_lift_vals: list[float] = []
    for Q_bar, Q_lift in zip(all_Q_bar, all_Q_lift):
        P_bar = ergodic_flow_to_transition(Q_bar)
        P_lift = ergodic_flow_to_transition(Q_lift)
        stb_phys_vals.append(stackelberg(P_bar, tau, W))
        stb_lift_vals.append(lifted_stackelberg(P_lift, V, tau, W_lift))

    best_phys = max(stb_phys_vals)
    best_lift = max(stb_lift_vals)
    print(
        f"San Francisco Stackelberg: {len(stb_phys_vals)} trials, "
        f"J(P_bar*) best={best_phys:.3f} mean={np.mean(stb_phys_vals):.3f} ± "
        f"{np.std(stb_phys_vals):.3f}, J^lift(P*) best={best_lift:.3f} "
        f"mean={np.mean(stb_lift_vals):.3f} ± {np.std(stb_lift_vals):.3f}",
        flush=True,
    )
    return best_phys, best_lift


def fig_san_francisco_rte(
    data_path: str = 'results/data/san_francisco_rte_diffs.npy',
) -> tuple[float, float]:
    """Report the best physical and lifted (weighted) truncated RTE achieved on the
    San Francisco graph (Appendix B / Table II).

    Loads the raw optimization results saved by sweeps.san_francisco_rte_improvement,
    recomputes H(P_bar*) and H^lift(P*) for every trial from the stored ergodic flow
    matrices, and returns (best H(P_bar*), best H^lift(P*)) -- the maximum over all
    n_trials restarts, i.e. the values that belong in Table II's "Proposed" row.
    """
    data = np.load(data_path, allow_pickle=True).item()
    W = data['W']
    V = data['V']
    eta = data['eta']
    all_Q_bar = data['Q_bar']
    all_Q_lift = data['Q_lift']
    W_lift = V @ W @ V.T

    rte_phys_vals: list[float] = []
    rte_lift_vals: list[float] = []
    for Q_bar, Q_lift in zip(all_Q_bar, all_Q_lift):
        P_bar = ergodic_flow_to_transition(Q_bar)
        P_lift = ergodic_flow_to_transition(Q_lift)
        rte_phys_vals.append(return_time_entropy(P_bar, eta, W, pi=Q_bar.sum(axis=1)))
        rte_lift_vals.append(lifted_return_time_entropy(P_lift, V, eta, W_lift, pi=Q_lift.sum(axis=1)))

    best_phys = max(rte_phys_vals)
    best_lift = max(rte_lift_vals)
    print(
        f"San Francisco RTE: {len(rte_phys_vals)} trials, "
        f"H(P_bar*) best={best_phys:.3f} mean={np.mean(rte_phys_vals):.3f} ± "
        f"{np.std(rte_phys_vals):.3f}, H^lift(P*) best={best_lift:.3f} "
        f"mean={np.mean(rte_lift_vals):.3f} ± {np.std(rte_lift_vals):.3f}",
        flush=True,
    )
    return best_phys, best_lift


def fig_lifting_budget_sweep_boxplot(
    data_path: str = 'results/data/lifting_budget_sweep.npy',
) -> None:
    """Grouped boxplot of percent decrease in Kemeny's constant vs. lifting budget,
    comparing the six lifting-budget allocation heuristics across many random graphs.

    Loads the raw optimization results saved by figures.fig_kemeny_lifting_budget_sweep
    (one entry per graph in 'graphs' / 'pi_bar' / 'kemeny_phys' / 'Q_bar' / 'results'),
    and for every (graph, budget, method) triple reports the already-computed,
    guaranteed-nonnegative 100 * diff / K(P_bar*) = 100 * (K(P_bar*) - K^lift(P*)) / K(P_bar*)
    stored by kemeny_lifting_budget_sweep. This intentionally reuses the stored 'diffs'
    rather than recomputing K^lift(P*) from the saved ergodic flow matrices: a
    from-scratch recompute needs the lifted MC's stationary distribution, and
    re-deriving it via a fresh linear solve (rather than reusing Q_lift.sum(axis=1),
    which is what the optimizer itself used) can be inaccurate enough at near-zero
    stationary mass (see kemeny_lifting_budget_sweep's docstring on max_grad_norm_lift)
    to spuriously flip a positive diff negative. For each budget, one box per method
    summarises the distribution of that percentage across all graphs. Entries where
    every lifted PGD restart failed (Q_lift is None) are skipped.

    Also prints a per-method/per-budget count of 'success' / 'no_improvement' /
    'all_failed' runs (see kemeny_lifting_budget_sweep's docstring), if the
    loaded data has the 'status' field (older saves predating that field are
    reported as 'unknown' instead) -- this is what makes the crash-rate
    reduction from _pgd_restarts and the optimize.py numerical fixes directly
    verifiable, rather than collapsing into indistinguishable 0% points.
    """
    data = np.load(data_path, allow_pickle=True).item()
    m = data['m']
    budget_values_all = np.asarray(data['budget_values'])
    keep_mask = ~np.isclose(budget_values_all / m, 3)
    budget_idx_map = np.where(keep_mask, np.cumsum(keep_mask) - 1, -1)
    budget_values = budget_values_all[keep_mask]
    kemeny_phys_all = data['kemeny_phys']
    results_all = data['results']
    method_names = list(results_all[0].keys())
    n_methods = len(method_names)
    n_budgets = len(budget_values)

    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))  # type: ignore[attr-defined]

    # pct[method_idx][budget_idx] = list of percent decreases across graphs
    pct: list[list[list[float]]] = [[[] for _ in range(n_budgets)] for _ in range(n_methods)]
    # status_counts[method_idx][budget_idx][status] = count across graphs
    status_counts: list[list[dict]] = [[{} for _ in range(n_budgets)] for _ in range(n_methods)]
    for k_phys, results in zip(kemeny_phys_all, results_all):
        if k_phys <= 0:
            continue
        for method_idx, name in enumerate(method_names):
            res = results[name]
            statuses = res.get('status', ['unknown'] * len(res['budgets']))
            for budget, Q_lift, diff, status in zip(res['budgets'], res['Q_lift'], res['diffs'], statuses):
                orig_idx = int(np.searchsorted(budget_values_all, budget))
                budget_idx = int(budget_idx_map[orig_idx])
                if budget_idx < 0:
                    continue
                status_counts[method_idx][budget_idx][status] = (
                    status_counts[method_idx][budget_idx].get(status, 0) + 1
                )
                if Q_lift is None:
                    continue
                pct[method_idx][budget_idx].append(100.0 * diff / k_phys)

    print(f"{'method':<16}{'budget/m':>10}  status counts")
    for method_idx, name in enumerate(method_names):
        for budget_idx, budget in enumerate(budget_values):
            counts = status_counts[method_idx][budget_idx]
            if counts:
                counts_str = ', '.join(f"{k}={v}" for k, v in sorted(counts.items()))
                print(f"{name:<16}{budget / m:>10.2g}  {counts_str}")

    fig, ax = plt.subplots(figsize=(9, 5))
    group_width = 0.8
    box_width = group_width / n_methods

    for method_idx, (name, color) in enumerate(zip(method_names, colors)):
        offset = (method_idx - (n_methods - 1) / 2) * box_width
        positions = np.arange(n_budgets) + offset
        bp = ax.boxplot(
            pct[method_idx],
            positions=positions,
            widths=box_width * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=1.2),
        )
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(color)
        for element in ('whiskers', 'caps'):
            for line in bp[element]:
                line.set_color(color)
        ax.plot([], [], color=color, linewidth=6, alpha=0.7, label=name)

    ax.axhline(0, color='gray', lw=0.8, ls=':', zorder=0)
    # ax.set_ylim(top=30)
    ax.set_xticks(np.arange(n_budgets))
    ax.set_xticklabels([f'{b / m:.2g}' for b in budget_values])
    ax.set_xlabel('Lifting Budget / $m$')
    ax.set_ylabel('Decrease in Kemeny Constant [%]')
    ax.legend(title='Lifting Method')
    plt.tight_layout()
    plt.savefig('results/lifting_budget_sweep_boxplot.pdf')
    plt.savefig('results/lifting_budget_sweep_boxplot.png', dpi=150)
    print("Saved: results/lifting_budget_sweep_boxplot.pdf / .png")


if __name__ == "__main__":
    # fig_random_graphs()
    # fig_random_digraphs()
    # fig_kemeny_vs_transition_probability()
    # fig_estimation_error_vs_trajectory_length()
    # fig_mean_capture_time_convergence()
    # fig_erdos_renyi_kemeny_percent_decrease()
    # fig_erdos_renyi_stackelberg_percent_increase()
    fig_erdos_renyi_rte_percent_increase()
    # fig_lifting_budget_sweep_boxplot()

    # San Francisco case study (Sec. IX-C / Table II) -- requires the three
    # sweeps.san_francisco_*_improvement functions to have been run first, saving
    # their .npy files under results/data/.
    # kemeny_phys, kemeny_lift = fig_san_francisco_kemeny()
    # stackelberg_phys, stackelberg_lift = fig_san_francisco_stackelberg()
    # rte_phys, rte_lift = fig_san_francisco_rte()
    # print_san_francisco_table(
    #     kemeny_phys, kemeny_lift, stackelberg_phys, stackelberg_lift, rte_phys, rte_lift,
    # )
