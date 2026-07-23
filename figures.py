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
    """Reproduce Fig. 3: MLE estimation error vs. trajectory length. Proposition 4
    guarantees this converges to P_bar, so the error should decay to zero.
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

    pi = stationary_distribution(P)
    Pbar, _ = collapsing(P, V)

    x0 = rng.choice(n, p=pi)
    x = [x0]
    for _ in range(2000 - 1):
        x.append(rng.choice(n, p=P[x[-1]]))
    y = [virtual_to_physical[xi] for xi in x]

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

    fig, ax = plt.subplots()
    ax.plot(T, estimation_errors)
    ax.set_xlabel(r'Trajectory Length $T$')
    ax.set_ylabel(r'$\|\hat{\bar{P}} - \bar{P}\|_F$')
    plt.tight_layout()
    plt.savefig('estimation_errors.pdf')
    plt.savefig('estimation_errors.png', dpi=150, bbox_inches='tight')


def fig_mean_capture_time_convergence(seed: int = 42) -> None:
    """Reproduce Fig. 4: empirical mean capture time vs. number of trials for P_bar and
    lifted P. For P_bar the mean converges to K(P_bar); for lifted P it converges to
    K^lift(P), not K(P), demonstrating that the lifted Kemeny constant is the correct
    performance metric.
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

    # patroller and adversary drawn from pi_bar; count first-passage steps
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

    # patroller drawn from pi (virtual), adversary drawn from pi_bar (physical)
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


def _fmt_sig3(x: float | None, sign: bool = False) -> str:
    """Format x to exactly 3 significant figures (e.g. 21 -> '21.0', 0.1 -> '0.100'),
    without the trailing bare '.' that Python's alternate-form 'g' type leaves on
    whole numbers (e.g. 825 -> '825.'). Returns '-' for None.
    """
    if x is None:
        return "-"
    s = f"{x:+#.3g}" if sign else f"{x:#.3g}"
    return s[:-1] if s.endswith('.') else s


def _fmt_pct_value(x: float | None) -> str:
    """Format a probability x in [0, 1] (e.g. a Stackelberg capture probability) as a
    percentage string with 3 significant figures. Returns '-' for None.
    """
    return f"{_fmt_sig3(100 * x)}%" if x is not None else "-"


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
        print(f"{p:>6.2f} | {_fmt_sig3(tp):>8} | {_fmt_sig3(tl):>8}")
    print("")


def print_san_francisco_wmax_computation_time_table(
    kemeny_manifest: dict | None = None,
    stackelberg_manifest: dict | None = None,
    rte_manifest: dict | None = None,
) -> None:
    """Print mean±std PGD computation time (s) vs. pruning threshold w_max for the San
    Francisco case study, for Table I's San Francisco rows. Each manifest is a dict as
    returned by sweeps.san_francisco_wmax_sweep()[metric] (or reloaded from disk),
    mapping each swept w_max to a 'status' and, for successes, a 'save_paths' entry;
    w_max values without a successful save_path print as '-'.
    """
    manifests = {
        'kemeny': kemeny_manifest,
        'stackelberg': stackelberg_manifest,
        'rte': rte_manifest,
    }
    if not any(manifests.values()):
        raise ValueError("at least one of kemeny_manifest/stackelberg_manifest/rte_manifest must be given")

    all_w_max = sorted(set().union(*(
        {int(w) for w in manifest['w_max_values']} for manifest in manifests.values() if manifest is not None
    )))

    times: dict[str, dict[int, tuple[float, float, float, float]]] = {}
    for name, manifest in manifests.items():
        times[name] = {}
        if manifest is None:
            continue
        for w_max, path in manifest['save_paths'].items():
            data = np.load(path, allow_pickle=True).item()
            times[name][int(w_max)] = (
                float(np.mean(data['times_phys'])), float(np.std(data['times_phys'])),
                float(np.mean(data['times_lift'])), float(np.std(data['times_lift'])),
            )

    def fmt(mean_std):
        return f"{_fmt_sig3(mean_std[0])}±{_fmt_sig3(mean_std[1])}" if mean_std is not None else "-"

    print("\n--- San Francisco pruning (w_max) sweep: PGD computation time (s) [Table I] ---")
    print(
        f"{'w_max':>6} | {'Kem. Phys.':>12} | {'Kem. Lift.':>12} | "
        f"{'Stk. Phys.':>12} | {'Stk. Lift.':>12} | {'RTE Phys.':>12} | {'RTE Lift.':>12}"
    )
    for w in all_w_max:
        k = times['kemeny'].get(w)
        s = times['stackelberg'].get(w)
        r = times['rte'].get(w)
        print(
            f"{w:>6} | "
            f"{fmt(k[0:2] if k else None):>12} | {fmt(k[2:4] if k else None):>12} | "
            f"{fmt(s[0:2] if s else None):>12} | {fmt(s[2:4] if s else None):>12} | "
            f"{fmt(r[0:2] if r else None):>12} | {fmt(r[2:4] if r else None):>12}"
        )
    print("")


def print_san_francisco_table(
    kemeny_phys: float, kemeny_lift: float,
    stackelberg_phys: float, stackelberg_lift: float,
    rte_phys: float, rte_lift: float,
) -> None:
    """Print the San Francisco case study results for Table II, alongside the [31]/[33]
    literature baselines (literature constants, not recomputed here). The kemeny/
    stackelberg/rte phys/lift arguments are the "Proposed" row's values -- the
    physical/lifted pair from the single largest-improvement trial, as returned by
    fig_san_francisco_kemeny/_stackelberg/_rte respectively.
    """
    rows = [
        ('Proposed', kemeny_phys, kemeny_lift, stackelberg_phys, stackelberg_lift, rte_phys, rte_lift),
        ('[31]', 24.3, None, None, None, 5.01, None),
        ('[33]', 21.2, None, 9.75e-2, None, 5.13, None),
    ]

    def fmt(x):
        return _fmt_sig3(x)

    print("\n--- San Francisco case study mean metric values [Table II] ---")
    print(
        f"{'Method':<10} | {'Kemeny Phys.':>12} | {'Kemeny Lift.':>12} | "
        f"{'Stack. Phys.':>12} | {'Stack. Lift.':>12} | {'RTE Phys.':>10} | {'RTE Lift.':>10}"
    )
    for name, kp, kl, sp, sl, rp, rl in rows:
        print(
            f"{name:<10} | {fmt(kp):>12} | {fmt(kl):>12} | {_fmt_pct_value(sp):>12} | "
            f"{_fmt_pct_value(sl):>12} | {fmt(rp):>10} | {fmt(rl):>10}"
        )
    print("")


def print_san_francisco_wmax_table(
    kemeny_manifest: dict | None = None,
    stackelberg_manifest: dict | None = None,
    rte_manifest: dict | None = None,
) -> None:
    """Print physical/lifted Kemeny/Stackelberg/RTE metrics vs. pruning threshold w_max
    for the San Francisco case study. Each manifest is as in
    print_san_francisco_wmax_computation_time_table. Recomputes each successful
    w_max's physical/lifted values via fig_san_francisco_kemeny/stackelberg/rte, so
    the printed pair is jointly achievable from one trial rather than a mix-and-match
    of separate optimizations.
    """
    fig_fns = {
        'kemeny': fig_san_francisco_kemeny,
        'stackelberg': fig_san_francisco_stackelberg,
        'rte': fig_san_francisco_rte,
    }
    manifests = {
        'kemeny': kemeny_manifest,
        'stackelberg': stackelberg_manifest,
        'rte': rte_manifest,
    }
    if not any(manifests.values()):
        raise ValueError("at least one of kemeny_manifest/stackelberg_manifest/rte_manifest must be given")

    all_w_max = sorted(set().union(*(
        {int(w) for w in manifest['w_max_values']} for manifest in manifests.values() if manifest is not None
    )))

    values: dict[str, dict[int, tuple[float, float]]] = {}
    n_edges_by_w: dict[int, int] = {}
    for name, manifest in manifests.items():
        values[name] = {}
        if manifest is None:
            continue
        for w_max, path in manifest['save_paths'].items():
            values[name][int(w_max)] = fig_fns[name](path)
        for w_max, ne in manifest['n_edges'].items():
            n_edges_by_w.setdefault(int(w_max), ne)

    def fmt(x):
        return _fmt_sig3(x)

    def pct(pair, higher_is_better):
        if pair is None or pair[0] == 0:
            return None
        phys, lift = pair
        return 100 * (lift - phys if higher_is_better else phys - lift) / phys

    def fmt_pct(x):
        return f"{_fmt_sig3(x, sign=True)}%" if x is not None else "-"

    print("\n--- San Francisco pruning (w_max) sweep: physical/lifted metrics ---")
    print(
        f"{'w_max':>6} | {'Edges':>6} | {'Kem. Phys.':>10} | {'Kem. Lift.':>10} | {'Kem. %Δ':>8} | "
        f"{'Stk. Phys.':>10} | {'Stk. Lift.':>10} | {'Stk. %Δ':>8} | "
        f"{'RTE Phys.':>9} | {'RTE Lift.':>9} | {'RTE %Δ':>8}"
    )
    for w in all_w_max:
        k = values['kemeny'].get(w)
        s = values['stackelberg'].get(w)
        r = values['rte'].get(w)
        print(
            f"{w:>6} | {n_edges_by_w.get(w, '-'):>6} | "
            f"{fmt(k[0] if k else None):>10} | {fmt(k[1] if k else None):>10} | {fmt_pct(pct(k, higher_is_better=False)):>8} | "
            f"{_fmt_pct_value(s[0] if s else None):>10} | {_fmt_pct_value(s[1] if s else None):>10} | {fmt_pct(pct(s, higher_is_better=True)):>8} | "
            f"{fmt(r[0] if r else None):>9} | {fmt(r[1] if r else None):>9} | {fmt_pct(pct(r, higher_is_better=True)):>8}"
        )
    print("")


def fig_erdos_renyi_kemeny_percent_decrease(
    data_path: str = 'results/data/erdos_renyi_kemeny_diffs.npy',
) -> None:
    """Ridgeline plot of the percent decrease in Kemeny's constant achieved by
    stationary-distribution lifting, vs. Erdős-Rényi edge probability p. Loads the raw
    results saved by sweeps.erdos_renyi_kemeny_improvement, recomputes K(P_bar*) and
    K^lift(P*) from the stored ergodic flow matrices and V, and reports
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
    """Ridgeline plot of the percent increase in the Stackelberg metric achieved by
    lifting, vs. Erdős-Rényi edge probability p. Loads the raw results saved by
    sweeps.erdos_renyi_stackelberg_improvement, recomputes J(P_bar*) and J^lift(P*)
    from the stored ergodic flow matrices, and reports
    100 * (J^lift(P*) - J(P_bar*)) / J(P_bar*).
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
    """Ridgeline plot of the percent increase in the Return-Time Entropy metric
    achieved by lifting, vs. Erdős-Rényi edge probability p. Loads the raw results
    saved by sweeps.erdos_renyi_rte_improvement, recomputes H(P_bar*) and H^lift(P*)
    from the stored ergodic flow matrices, and reports
    100 * (H^lift(P*) - H(P_bar*)) / H(P_bar*).
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
            P_bar = ergodic_flow_to_transition(Q_bar)
            P_lift = ergodic_flow_to_transition(Q_lift)
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


def fig_erdos_renyi_combined_ridgelines(
    kemeny_data_path: str = 'results/data/erdos_renyi_kemeny_diffs.npy',
    stackelberg_data_path: str = 'results/data/erdos_renyi_stackelberg_diffs.npy',
    rte_data_path: str = 'results/data/erdos_renyi_rte_diffs.npy',
    dpi: int = 300,
) -> None:
    """Combine Figs. 6-8 (Kemeny/Stackelberg/RTE percent-change ridgeline plots) into
    one figure, stacked vertically, by rasterizing each standalone
    fig_erdos_renyi_*_percent_* panel and compositing the images (joypy's own figures
    can't be composed directly via gridspec/subfigures; see NOTES.md).
    """
    import io
    import matplotlib.image as mpimg

    panel_fns = [
        (fig_erdos_renyi_kemeny_percent_decrease, kemeny_data_path),
        (fig_erdos_renyi_stackelberg_percent_increase, stackelberg_data_path),
        (fig_erdos_renyi_rte_percent_increase, rte_data_path),
    ]

    panels = []
    for fn, path in panel_fns:
        fn(path)
        panel_fig = plt.gcf()
        buf = io.BytesIO()
        panel_fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        plt.close(panel_fig)
        buf.seek(0)
        panels.append(mpimg.imread(buf))

    heights = [p.shape[0] for p in panels]
    total_height_in = 7 * sum(heights) / panels[0].shape[1]
    fig, axes = plt.subplots(
        3, 1, figsize=(7, total_height_in),
        gridspec_kw={'height_ratios': heights},
    )
    for ax, panel in zip(axes, panels):
        ax.imshow(panel, aspect='auto')
        ax.axis('off')

    plt.subplots_adjust(hspace=0.02)
    plt.savefig('results/erdos_renyi_combined_ridgelines.pdf', bbox_inches='tight')
    plt.savefig('results/erdos_renyi_combined_ridgelines.png', dpi=150, bbox_inches='tight')
    print("Saved: results/erdos_renyi_combined_ridgelines.pdf / .png")


def _select_phys_lift(
    phys_vals: list[float],
    lift_vals: list[float],
    higher_is_better: bool,
) -> tuple[float, float]:
    """Return the (phys, lift) pair from whichever trial maximizes the improvement from
    physical to lifted (phys - lift if lower is better, e.g. Kemeny; lift - phys if
    higher is better, e.g. Stackelberg/RTE), so the reported pair is jointly
    achievable from one trial rather than mixing independently-best values.
    """
    sign = 1 if higher_is_better else -1
    idx = int(np.argmax([sign * (l - p) for p, l in zip(phys_vals, lift_vals)]))
    return phys_vals[idx], lift_vals[idx]


def fig_san_francisco_kemeny(
    data_path: str = 'results/data/san_francisco_kemeny_diffs.npy',
) -> tuple[float, float]:
    """Report the physical and lifted (weighted) Kemeny constants achieved on the
    San Francisco graph (Sec. IX-C / Table II), from the trial with the largest
    improvement (see _select_phys_lift). Loads the raw results saved by
    sweeps.san_francisco_kemeny_improvement and recomputes K(P_bar*)/K^lift(P*) for
    every trial from the stored ergodic flow matrices.
    """
    data = np.load(data_path, allow_pickle=True).item()
    W = data['W']
    V = data['V']
    all_Q_bar = data['Q_bar']
    all_Q_lift = data['Q_lift']
    W_lift = V @ W @ V.T

    kemeny_phys_vals: list[float] = []
    kemeny_lift_vals: list[float] = []
    for Q_bar, Q_lift in zip(all_Q_bar, all_Q_lift):
        P_bar = ergodic_flow_to_transition(Q_bar)
        P_lift = ergodic_flow_to_transition(Q_lift)
        kemeny_phys_vals.append(kemeny(P_bar, W, pi=Q_bar.sum(axis=1)))
        kemeny_lift_vals.append(lifted_kemeny(P_lift, V, W_lift, pi=Q_lift.sum(axis=1)))

    best_phys, best_lift = _select_phys_lift(
        kemeny_phys_vals, kemeny_lift_vals, higher_is_better=False,
    )
    print(
        f"San Francisco Kemeny: {len(kemeny_phys_vals)} trials, "
        f"K(P_bar*)={best_phys:.3f} K^lift(P*)={best_lift:.3f} (largest-improvement trial); "
        f"K(P_bar*) mean={np.mean(kemeny_phys_vals):.3f} ± {np.std(kemeny_phys_vals):.3f}, "
        f"K^lift(P*) mean={np.mean(kemeny_lift_vals):.3f} ± {np.std(kemeny_lift_vals):.3f}",
        flush=True,
    )
    return best_phys, best_lift


def fig_san_francisco_stackelberg(
    data_path: str = 'results/data/san_francisco_stackelberg_diffs.npy',
) -> tuple[float, float]:
    """Report the physical and lifted (weighted) Stackelberg metrics achieved on
    the San Francisco graph (Appendix A / Table II), from the trial with the largest
    improvement (see _select_phys_lift). Loads the raw results saved by
    sweeps.san_francisco_stackelberg_improvement and recomputes J(P_bar*)/J^lift(P*)
    for every trial; V is a per-trial list since each trial's lifting is re-derived
    from that trial's own realized physical stationary distribution, so W_lift is
    recomputed per trial from its own V.
    """
    data = np.load(data_path, allow_pickle=True).item()
    W = data['W']
    all_V = data['V']
    tau = data['tau']
    all_Q_bar = data['Q_bar']
    all_Q_lift = data['Q_lift']

    stb_phys_vals: list[float] = []
    stb_lift_vals: list[float] = []
    for Q_bar, Q_lift, V in zip(all_Q_bar, all_Q_lift, all_V):
        P_bar = ergodic_flow_to_transition(Q_bar)
        P_lift = ergodic_flow_to_transition(Q_lift)
        W_lift = V @ W @ V.T
        stb_phys_vals.append(stackelberg(P_bar, tau, W))
        stb_lift_vals.append(lifted_stackelberg(P_lift, V, tau, W_lift))

    best_phys, best_lift = _select_phys_lift(
        stb_phys_vals, stb_lift_vals, higher_is_better=True,
    )
    print(
        f"San Francisco Stackelberg: {len(stb_phys_vals)} trials, "
        f"J(P_bar*)={best_phys:.3f} J^lift(P*)={best_lift:.3f} (largest-improvement trial); "
        f"J(P_bar*) mean={np.mean(stb_phys_vals):.3f} ± {np.std(stb_phys_vals):.3f}, "
        f"J^lift(P*) mean={np.mean(stb_lift_vals):.3f} ± {np.std(stb_lift_vals):.3f}",
        flush=True,
    )
    return best_phys, best_lift


def fig_san_francisco_rte(
    data_path: str = 'results/data/san_francisco_rte_diffs.npy',
) -> tuple[float, float]:
    """Report the physical and lifted (weighted) truncated RTE achieved on the
    San Francisco graph (Appendix B / Table II), from the trial with the largest
    improvement (see _select_phys_lift). Loads the raw results saved by
    sweeps.san_francisco_rte_improvement and recomputes H(P_bar*)/H^lift(P*) for
    every trial from the stored ergodic flow matrices.
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

    best_phys, best_lift = _select_phys_lift(
        rte_phys_vals, rte_lift_vals, higher_is_better=True,
    )
    print(
        f"San Francisco RTE: {len(rte_phys_vals)} trials, "
        f"H(P_bar*)={best_phys:.3f} H^lift(P*)={best_lift:.3f} (largest-improvement trial); "
        f"H(P_bar*) mean={np.mean(rte_phys_vals):.3f} ± {np.std(rte_phys_vals):.3f}, "
        f"H^lift(P*) mean={np.mean(rte_lift_vals):.3f} ± {np.std(rte_lift_vals):.3f}",
        flush=True,
    )
    return best_phys, best_lift


def fig_san_francisco_mean_capture_time_convergence(
    data_path: str = 'results/data/san_francisco_kemeny_diffs.npy',
    seed: int = 42,
    N_trials: int = 5_000,
) -> None:
    """Reproduce Fig. 4's mean-capture-time convergence plot for the San Francisco case
    study, demonstrating the same phenomenon on real, weighted data. Loads the raw
    results saved by sweeps.san_francisco_kemeny_improvement, picks the trial
    minimizing K^lift(P), and simulates a patroller (drawn from pi, lifted) chasing an
    attacker (drawn from pi_bar, physical), accumulating W-weighted travel time per
    hop (Eq. 30). The running mean converges to K^lift(P), not K(P_bar) -- what a
    naive adversary observing only the physical trajectory would expect.
    """
    rng = np.random.default_rng(seed)
    data = np.load(data_path, allow_pickle=True).item()
    V = data['V']
    W = data['W']
    pi_bar = data['pi_bar']
    all_Q_bar = data['Q_bar']
    all_Q_lift = data['Q_lift']
    W_lift = V @ W @ V.T

    kemeny_phys_vals: list[float] = []
    kemeny_lift_vals: list[float] = []
    for Q_bar, Q_lift in zip(all_Q_bar, all_Q_lift):
        P_bar = ergodic_flow_to_transition(Q_bar)
        P_lift = ergodic_flow_to_transition(Q_lift)
        kemeny_phys_vals.append(kemeny(P_bar, W, pi=Q_bar.sum(axis=1)))
        kemeny_lift_vals.append(lifted_kemeny(P_lift, V, W_lift, pi=Q_lift.sum(axis=1)))

    best_idx = int(np.argmin(kemeny_lift_vals))
    k_phys = kemeny_phys_vals[best_idx]
    k_lift = kemeny_lift_vals[best_idx]
    Q_lift = all_Q_lift[best_idx]

    P_lift = ergodic_flow_to_transition(Q_lift)
    P_lift = P_lift / P_lift.sum(axis=1, keepdims=True)  # exact row-stochasticity for rng.choice
    pi = Q_lift.sum(axis=1)
    pi = pi / pi.sum()
    pi_bar = pi_bar / pi_bar.sum()
    virtual_to_physical = V.argmax(axis=1)
    m = W.shape[0]
    n = P_lift.shape[0]

    capture_times = []
    for _ in range(N_trials):
        patroller = rng.choice(n, p=pi)
        attacker = rng.choice(m, p=pi_bar)
        t = 0.0
        while True:
            next_state = rng.choice(n, p=P_lift[patroller])
            t += W_lift[patroller, next_state]
            patroller = next_state
            if virtual_to_physical[patroller] == attacker:
                break
        capture_times.append(t)

    trials = np.arange(1, N_trials + 1)
    mean_capture_time = np.cumsum(capture_times) / trials

    fig, ax = plt.subplots()
    ax.plot(trials, mean_capture_time, label=r'Empirical Mean Capture Time', color='orange')
    ax.axhline(k_phys, linestyle='--', label=r'$K(\bar{Q})$')
    ax.axhline(k_lift, linestyle='--', label=r'$K^{\mathrm{lift}}(Q)$', color='orange')
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Mean Capture Time')
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/san_francisco_mean_capture_time.pdf', bbox_inches='tight')
    plt.savefig('results/san_francisco_mean_capture_time.png', dpi=150, bbox_inches='tight')
    print("Saved: results/san_francisco_mean_capture_time.pdf / .png")


def fig_lifting_budget_sweep_boxplot(
    data_path: str = 'results/data/lifting_budget_sweep.npy',
) -> None:
    """Grouped boxplot of percent decrease in Kemeny's constant vs. lifting budget,
    comparing the six lifting-budget allocation heuristics across many random graphs.
    Loads the raw results saved by sweeps.kemeny_lifting_budget_sweep and, for every
    (graph, budget, method) triple, reports the stored 100 * diff / K(P_bar*) (reusing
    the optimizer's own 'diffs' rather than recomputing K^lift(P*) from a fresh,
    possibly ill-conditioned linear solve). Also prints a per-method/per-budget count
    of 'success'/'no_improvement'/'all_failed' runs.
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
    # fig_erdos_renyi_rte_percent_increase()
    # fig_lifting_budget_sweep_boxplot()

    # fig_san_francisco_mean_capture_time_convergence(seed=0, N_trials=3_000)
    # print_san_francisco_table(
    #     kemeny_phys, kemeny_lift, stackelberg_phys, stackelberg_lift, rte_phys, rte_lift,
    # )

    _sf_wmax_dir = 'results/data/san_francisco_wmax_sweep'
    _sf_wmax_kemeny_manifest = np.load(
        f'{_sf_wmax_dir}/san_francisco_kemeny_wmax_manifest.npy', allow_pickle=True).item()
    _sf_wmax_stackelberg_manifest = np.load(
        f'{_sf_wmax_dir}/san_francisco_stackelberg_wmax_manifest.npy', allow_pickle=True).item()
    _sf_wmax_rte_manifest = np.load(
        f'{_sf_wmax_dir}/san_francisco_rte_wmax_manifest.npy', allow_pickle=True).item()

    print_san_francisco_wmax_table(
        _sf_wmax_kemeny_manifest, _sf_wmax_stackelberg_manifest, _sf_wmax_rte_manifest,
    )
    print_san_francisco_wmax_computation_time_table(
        _sf_wmax_kemeny_manifest, _sf_wmax_stackelberg_manifest, _sf_wmax_rte_manifest,
    )
