# Markov Chain Lifting

Code accompanying the paper **"Emergent Deception in Robotic Surveillance via
Markov Chain Lifting"** by Yohan John\*, Gilberto Díaz-García\*, Jason R.
Marden, and Francesco Bullo (\*equal contribution), submitted to *IEEE
Transactions on Robotics*.

The paper proposes **Markov chain lifting** as a technique for designing
stochastic robotic surveillance strategies. Lifting augments the state space
of a surveillance Markov chain (MC) with virtual copies of each physical
node, enabling non-reversible behavior that a physical-space MC cannot
represent. The resulting lifted strategies achieve better performance than
physical-space baselines under several patrolling metrics — the Kemeny
constant, a Stackelberg-game capture probability, and return-time entropy —
and provably cannot do worse, per bounds derived from MC conductance. A
further consequence is **emergent deception**: an adversary who observes
only the physical trajectory and estimates a physical-space MC from it will
have its estimate converge to the *collapsed* chain, so it systematically
underestimates the patroller's true performance.

## Repository layout

| File | Contents |
| --- | --- |
| [`markov.py`](markov.py) | Core MC utilities: stationary distribution, transition matrix ↔ ergodic flow conversions, lifting/collapsing (`collapsing`), and conductance. |
| [`metrics.py`](metrics.py) | Surveillance performance metrics: Kemeny constant, Stackelberg capture probability, and return-time entropy, plus their lifted counterparts (`lifted_kemeny`, `lifted_stackelberg`, `lifted_return_time_entropy`). |
| [`graph.py`](graph.py) | Graph generation (Erdős–Rényi graphs/digraphs), lifting-budget assignment strategies (uniform, stationary-distribution, degree, betweenness, eigenvector centrality, reversible flow), and the real-world case-study graphs (San Francisco police patrol, CTCV campus, airport hub). |
| [`optimize.py`](optimize.py) | Projected gradient descent (PGD) for optimizing a physical or lifted MC's ergodic flow against a given metric, with `cvxpy`/OSQP projection steps and closed-form/JAX-autodiff gradients. |
| [`sweeps.py`](sweeps.py) | Experiment drivers that sweep over graphs, lifting budgets, and lifting-assignment methods, calling into `optimize.py` and saving results to `results/data/`. |
| [`figures.py`](figures.py) | Plotting code that consumes `results/data/` and produces the paper's figures/tables in `results/figures/`. |
| [`NOTES.md`](NOTES.md) | Implementation notes on numerical conventions, hyperparameter choices, and the exact invocations used to regenerate each saved result. |
| [`results/`](results/) | Saved sweep outputs (`data/`, as `.npy`) and rendered figures (`figures/`). |

## Installation

Requires Python 3.10+. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

There is no CLI entry point; `sweeps.py` and `figures.py` are run as scripts
with the desired calls uncommented in their `if __name__ == "__main__":`
blocks (see the tail of each file for examples, and [`NOTES.md`](NOTES.md)
for the exact invocations and approximate wall-clock times used to produce
each result in the paper). The general workflow is:

1. **Run a sweep** in `sweeps.py` (e.g. `erdos_renyi_kemeny_improvement`,
   `san_francisco_stackelberg_improvement`, `kemeny_lifting_budget_sweep`)
   to optimize physical- and lifted-space MCs over a batch of graphs and
   save the results to `results/data/*.npy`.
2. **Render a figure** in `figures.py` (e.g.
   `fig_erdos_renyi_kemeny_percent_decrease`, `fig_san_francisco_kemeny`) to
   load the corresponding `.npy` file and produce the plot/table.

Alternatively, `markov.py`, `metrics.py`, `graph.py`, and `optimize.py` can
be used directly as a library, e.g.:

```python
import numpy as np
from graph import erdos_renyi_graph, stationary_lifting
from markov import stationary_distribution
from metrics import kemeny, lifted_kemeny

A = erdos_renyi_graph(m=10, p=0.4, seed=0)
# ... build a physical-space transition matrix Pbar on A ...
# ... build a lifting V (e.g. via stationary_lifting) and lifted P ...
# K = kemeny(Pbar); K_lift = lifted_kemeny(P, V)
```

## License

MIT — see [LICENSE](LICENSE).
