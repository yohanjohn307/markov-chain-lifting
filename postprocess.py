import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joypy

from markov import ergodic_flow_to_transition, stationary_distribution
from metrics import (
    kemeny, lifted_kemeny, stackelberg, lifted_stackelberg,
    return_time_entropy, lifted_return_time_entropy,
)
from graph import degree_lifting


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
        fontsize=11,
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
        fontsize=11,
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
        fontsize=11,
    )
    for ax in axes:
        ax.axvline(0, color='gray', lw=0.8, ls=':', zorder=0)

    plt.tight_layout()
    plt.savefig('results/erdos_renyi_rte_percent_increase.pdf', bbox_inches='tight')
    plt.savefig('results/erdos_renyi_rte_percent_increase.png', dpi=150, bbox_inches='tight')
    print("Saved: results/erdos_renyi_rte_percent_increase.pdf / .png")


def fig_lifting_budget_sweep_scatter(
    data_path: str = 'results/lifting_budget_sweep.npy',
) -> None:
    """Scatter plot of percent decrease in Kemeny's constant vs. lifting budget,
    comparing the five lifting-budget allocation heuristics.

    Loads the raw optimization results saved by figures.fig_lifting_budget_sweep,
    recomputes K(P_bar*) and K^lift(P*) for every (budget, method) pair from the
    stored ergodic flow matrices, and reports 100 * (K(P_bar*) - K^lift(P*)) / K(P_bar*).
    """
    data = np.load(data_path, allow_pickle=True).item()
    Q_bar = data['Q_bar']
    results = data['results']

    k_phys = kemeny(ergodic_flow_to_transition(Q_bar))

    fig, ax = plt.subplots(figsize=(7, 5))
    markers = ['o', 's', '^', 'D', 'v']
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))  # type: ignore[attr-defined]

    for (name, res), marker, color in zip(results.items(), markers, colors):
        pct: list[float] = []
        valid_budgets: list[int] = []
        for budget, Q_lift, V in zip(res['budgets'], res['Q_lift'], res['V']):
            P_lift = ergodic_flow_to_transition(Q_lift)
            k_lift = lifted_kemeny(P_lift, V)
            if k_phys > 0:
                pct.append(100.0 * (k_phys - k_lift) / k_phys)
                valid_budgets.append(budget)
        ax.scatter(valid_budgets, pct, label=name, marker=marker, color=color, s=60, alpha=0.8,
                   edgecolors='white', linewidths=0.5)

    ax.axhline(0, color='gray', lw=0.8, ls=':', zorder=0)
    ax.set_xlabel('Lifting Budget')
    ax.set_ylabel('Decrease in Kemeny Constant [%]')
    ax.legend(title='Lifting Method')
    plt.tight_layout()
    plt.savefig('results/lifting_budget_sweep_scatter.pdf', bbox_inches='tight')
    plt.savefig('results/lifting_budget_sweep_scatter.png', dpi=150, bbox_inches='tight')
    print("Saved: results/lifting_budget_sweep_scatter.pdf / .png")


if __name__ == "__main__":
    # fig_erdos_renyi_kemeny_percent_decrease()
    # fig_erdos_renyi_stackelberg_percent_increase()
    # fig_erdos_renyi_rte_percent_increase()
    fig_lifting_budget_sweep_scatter()
