
import numpy as np
import matplotlib.pyplot as plt

from markov import stationary_distribution, collapsing
from metrics import kemeny, lifted_kemeny
from graph import erdos_renyi_graph


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
    ax.axhline(25/6, linestyle='--', label=r'$\mathcal{M}(\bar{P}) = 25/6$')
    ax.set_xlabel(r'$p$')
    ax.set_ylabel(r'$\mathcal{M}(P)$')
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
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\|P_{\text{bar}} - P_{\text{est}}\|_F$')
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
    ax.axhline(kemeny(Pbar),        linestyle='--', label=r'$\mathcal{M}(\bar{P})$')
    ax.axhline(lifted_kemeny(P, V), linestyle='--', label=r'$\mathcal{M}^{\mathrm{lift}}(P)$', color='orange')
    ax.axhline(kemeny(P),           linestyle='--', label=r'$\mathcal{M}(P)$', color='green')
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Mean Capture Time')
    ax.legend()
    plt.tight_layout()
    plt.savefig('mean_capture_time.pdf')

def fig_random_graph(m: int = 8, p: float = 0.4, seed: int = 0) -> None:
    """Plot a randomly generated Erdős-Rényi graph G(m, p) with self-loops."""
    import networkx as nx
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


if __name__ == "__main__":
    # fig2()
    # fig3()
    # fig4()
    fig_random_graph()
