import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joypy

from markov import ergodic_flow_to_transition
from metrics import kemeny, lifted_kemeny, stackelberg, lifted_stackelberg
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

    all_pct: list[list[float]] = []
    for p_idx, p in enumerate(p_values):
        pct_p: list[float] = []
        for A, Q_bar, Q_lift in zip(all_graphs[p_idx], all_Q_bar[p_idx], all_Q_lift[p_idx]):
            if Q_bar is None or Q_lift is None:
                continue
            V = degree_lifting(A)
            P_bar = ergodic_flow_to_transition(Q_bar)
            P_lift = ergodic_flow_to_transition(Q_lift)
            k_phys = kemeny(P_bar)
            k_lift = lifted_kemeny(P_lift, V)
            if k_phys > 0:
                pct_p.append(100.0 * (k_phys - k_lift) / k_phys)
        print(
            f"p={p:.2f}: {len(pct_p)} graphs, "
            + (f"percent decrease = {np.mean(pct_p):.2f}% ± {np.std(pct_p):.2f}%" if pct_p else ""),
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


def fig_erdos_renyi_stackelberg_percent_decrease(
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
        print(
            f"p={p:.2f}: {len(pct_p)} graphs, "
            + (f"percent increase = {np.mean(pct_p):.2f}% ± {np.std(pct_p):.2f}%" if pct_p else ""),
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
        fontsize=11,
    )
    for ax in axes:
        ax.axvline(0, color='gray', lw=0.8, ls=':', zorder=0)

    plt.tight_layout()
    plt.savefig('results/erdos_renyi_stackelberg_percent_decrease.pdf', bbox_inches='tight')
    plt.savefig('results/erdos_renyi_stackelberg_percent_decrease.png', dpi=150, bbox_inches='tight')
    print("Saved: results/erdos_renyi_stackelberg_percent_decrease.pdf / .png")


if __name__ == "__main__":
    fig_erdos_renyi_kemeny_percent_decrease()
    fig_erdos_renyi_stackelberg_percent_decrease()
