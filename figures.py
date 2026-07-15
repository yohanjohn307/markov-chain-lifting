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
from graph import degree_lifting, erdos_renyi_graph, erdos_renyi_digraph

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'legend.title_fontsize': 13,
})


def fig_erdos_renyi_kemeny_percent_decrease(
    data_path: str = 'results/erdos_renyi_kemeny_diffs.npy',
) -> None:
    """Ridgeline plot of the percent decrease in Kemeny's constant achieved by degree
    lifting, vs. Erdős-Rényi edge probability p.

    Loads the raw optimization results saved by figures.fig_erdos_renyi_kemeny_improvement,
    recomputes K(P_bar*) and K^lift(P*) for every graph from the stored ergodic flow
    matrices, and reports 100 * (K(P_bar*) - K^lift(P*)) / K(P_bar*).
    """
    data = np.load(data_path, allow_pickle=True).item()
    p_values = data['p_values']
    all_graphs = data['graphs']
    all_Q_bar = data['Q_bar']
    all_Q_lift = data['Q_lift']
    all_times_phys = data['times_phys']
    all_times_lift = data['times_lift']
    all_conductance_lb = data['conductance_lb']

    all_pct: list[list[float]] = []
    for p_idx, p in enumerate(p_values):
        pct_p: list[float] = []
        conductance_pct_p: list[float] = []
        for A, Q_bar, Q_lift, conductance_lb in zip(
            all_graphs[p_idx], all_Q_bar[p_idx], all_Q_lift[p_idx], all_conductance_lb[p_idx]
        ):
            if Q_bar is None or Q_lift is None:
                continue
            V = degree_lifting(A, budget=Q_lift.shape[0])
            P_bar = ergodic_flow_to_transition(Q_bar)
            P_lift = ergodic_flow_to_transition(Q_lift)
            k_phys = kemeny(P_bar)
            k_lift = lifted_kemeny(P_lift, V)
            if k_phys > 0:
                pct_p.append(100.0 * (k_phys - k_lift) / k_phys)
                conductance_pct_p.append(100.0 * (k_phys - conductance_lb) / k_phys)
        times_phys_p = all_times_phys[p_idx]
        times_lift_p = all_times_lift[p_idx]
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
        fontsize=16,
    )
    for ax in axes:
        ax.axvline(0, color='gray', lw=0.8, ls=':', zorder=0)

    plt.tight_layout()
    plt.savefig('results/erdos_renyi_kemeny_percent_decrease.pdf', bbox_inches='tight')
    plt.savefig('results/erdos_renyi_kemeny_percent_decrease.png', dpi=150, bbox_inches='tight')
    print("Saved: results/erdos_renyi_kemeny_percent_decrease.pdf / .png")


def fig_erdos_renyi_stackelberg_percent_increase(
    data_path: str = 'results/erdos_renyi_stackelberg_diffs.npy',
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
    for p_idx, p in enumerate(p_values):
        pct_p: list[float] = []
        for tau, Q_bar, Q_lift, V in zip(
            all_tau[p_idx], all_Q_bar[p_idx], all_Q_lift[p_idx], all_V[p_idx]
        ):
            if Q_bar is None or Q_lift is None:
                continue
            # V is the actual mapping used to produce Q_lift: the degree lifting
            # if it outperformed the physical optimum, or the identity fallback
            # (see fig_erdos_renyi_stackelberg_improvement) otherwise.
            P_bar = ergodic_flow_to_transition(Q_bar)
            P_lift = ergodic_flow_to_transition(Q_lift)
            j_phys = stackelberg(P_bar, tau)
            j_lift = lifted_stackelberg(P_lift, V, tau)
            if j_phys > 0:
                pct_p.append(100.0 * (j_lift - j_phys) / j_phys)
        times_phys_p = all_times_phys[p_idx]
        times_lift_p = all_times_lift[p_idx]
        print(
            f"p={p:.2f}: {len(pct_p)} graphs, "
            + (f"percent increase = {np.mean(pct_p):.2f}% ± {np.std(pct_p):.2f}%, " if pct_p else "")
            + (f"mean compute time: phys={np.mean(times_phys_p):.2f}s, lift={np.mean(times_lift_p):.2f}s"
               if len(times_phys_p) else ""),
            flush=True,
        )
        all_pct.append(pct_p)

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
        r'Increase in Stackelberg Metric [%]',
        fontsize=16,
    )
    for ax in axes:
        ax.axvline(0, color='gray', lw=0.8, ls=':', zorder=0)

    plt.tight_layout()
    plt.savefig('results/erdos_renyi_stackelberg_percent_increase.pdf', bbox_inches='tight')
    plt.savefig('results/erdos_renyi_stackelberg_percent_increase.png', dpi=150, bbox_inches='tight')
    print("Saved: results/erdos_renyi_stackelberg_percent_increase.pdf / .png")


def fig_erdos_renyi_rte_percent_increase(
    data_path: str = 'results/erdos_renyi_rte_diffs.npy',
) -> None:
    """Ridgeline plot of the percent increase in the Return-Time Entropy metric achieved by
    degree lifting, vs. Erdős-Rényi edge probability p.

    Loads the raw optimization results saved by figures.fig_erdos_renyi_rte_improvement,
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
    for p_idx, p in enumerate(p_values):
        pct_p: list[float] = []
        for Q_bar, Q_lift, V in zip(
            all_Q_bar[p_idx], all_Q_lift[p_idx], all_V[p_idx]
        ):
            if Q_bar is None or Q_lift is None:
                continue
            # V is the actual mapping used to produce Q_lift: the degree lifting
            # if it outperformed the physical optimum, or the identity fallback
            # (see fig_erdos_renyi_rte_improvement) otherwise.
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
        print(
            f"p={p:.2f}: {len(pct_p)} graphs, "
            + (f"percent increase = {np.mean(pct_p):.2f}% ± {np.std(pct_p):.2f}%, " if pct_p else "")
            + (f"mean compute time: phys={np.mean(times_phys_p):.2f}s, lift={np.mean(times_lift_p):.2f}s"
               if len(times_phys_p) else ""),
            flush=True,
        )
        all_pct.append(pct_p)

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
        fontsize=16,
    )
    for ax in axes:
        ax.axvline(0, color='gray', lw=0.8, ls=':', zorder=0)

    plt.tight_layout()
    plt.savefig('results/erdos_renyi_rte_percent_increase.pdf', bbox_inches='tight')
    plt.savefig('results/erdos_renyi_rte_percent_increase.png', dpi=150, bbox_inches='tight')
    print("Saved: results/erdos_renyi_rte_percent_increase.pdf / .png")


def fig_lifting_budget_sweep_boxplot(
    data_path: str = 'results/lifting_budget_sweep.npy',
) -> None:
    """Grouped boxplot of percent decrease in Kemeny's constant vs. lifting budget,
    comparing the six lifting-budget allocation heuristics across many random graphs.

    Loads the raw optimization results saved by figures.fig_kemeny_lifting_budget_sweep
    (one entry per graph in 'graphs' / 'pi_bar' / 'kemeny_phys' / 'Q_bar' / 'results'),
    recomputes K^lift(P*) for every (graph, budget, method) triple from the stored
    ergodic flow matrices, and reports 100 * (K(P_bar*) - K^lift(P*)) / K(P_bar*).
    For each budget, one box per method summarises the distribution of that
    percentage across all graphs. Entries where every lifted PGD restart failed
    (Q_lift is None) are skipped.
    """
    data = np.load(data_path, allow_pickle=True).item()
    m = data['m']
    budget_values = np.asarray(data['budget_values'])
    kemeny_phys_all = data['kemeny_phys']
    results_all = data['results']
    method_names = list(results_all[0].keys())
    n_methods = len(method_names)
    n_budgets = len(budget_values)

    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))  # type: ignore[attr-defined]

    # pct[method_idx][budget_idx] = list of percent decreases across graphs
    pct: list[list[list[float]]] = [[[] for _ in range(n_budgets)] for _ in range(n_methods)]
    for k_phys, results in zip(kemeny_phys_all, results_all):
        if k_phys <= 0:
            continue
        for method_idx, name in enumerate(method_names):
            res = results[name]
            for budget, Q_lift, V in zip(res['budgets'], res['Q_lift'], res['V']):
                if Q_lift is None:
                    continue
                budget_idx = int(np.searchsorted(budget_values, budget))
                P_lift = ergodic_flow_to_transition(Q_lift)
                k_lift = lifted_kemeny(P_lift, V)
                pct[method_idx][budget_idx].append(100.0 * (k_phys - k_lift) / k_phys)

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
            showfliers=True,
            flierprops=dict(marker='o', markersize=3, markerfacecolor=color, markeredgecolor='none', alpha=0.5),
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
    ax.set_xticks(np.arange(n_budgets))
    ax.set_xticklabels([f'{b / m:.2g}' for b in budget_values])
    ax.set_xlabel('Lifting Budget / $m$')
    ax.set_ylabel('Decrease in Kemeny Constant [%]')
    ax.legend(title='Lifting Method')
    plt.tight_layout()
    plt.savefig('results/lifting_budget_sweep_boxplot.pdf', bbox_inches='tight')
    plt.savefig('results/lifting_budget_sweep_boxplot.png', dpi=150, bbox_inches='tight')
    print("Saved: results/lifting_budget_sweep_boxplot.pdf / .png")


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
    for _ in range(10000 - 1):
        x.append(rng.choice(n, p=P[x[-1]]))
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

    N_trials = 10000

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


if __name__ == "__main__":
    # fig_random_graphs()
    # fig_random_digraphs()
    # fig_kemeny_vs_transition_probability()
    # fig_estimation_error_vs_trajectory_length()
    # fig_mean_capture_time_convergence()
    # fig_erdos_renyi_kemeny_percent_decrease()
    # fig_erdos_renyi_stackelberg_percent_increase()
    # fig_erdos_renyi_rte_percent_increase()
    fig_lifting_budget_sweep_boxplot()
