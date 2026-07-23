# Implementation notes

Numerical caveats and hyperparameter-tuning rationale that don't belong in
docstrings. Function names below match their definitions in `markov.py`,
`metrics.py`, `optimize.py`, and `sweeps.py`.

## Numerical conventions

**Passing `pi`/`Q` explicitly instead of re-deriving it.** `kemeny`,
`lifted_kemeny`, `return_time_entropy`, and `lifted_return_time_entropy` all
accept an optional `pi`. Left as `None`, they solve for it fresh via
`stationary_distribution(Pbar)`. That solve is exact in principle but can be
ill-conditioned for chains with very small transition probabilities, even
when a `Q` matrix's row sum is already exact by construction (e.g. `Q` came
out of a PGD optimizer's projection step, which pins `Q.sum(axis=1)` to the
target `pi`). Prefer passing `pi=Q.sum(axis=1)` whenever such a `Q` is
available. This matters most for `return_time_entropy`/
`lifted_return_time_entropy`, since the truncation length `K_eta` is
sensitive to `pi.min()`, and an ill-conditioned solve can blow it up.

**`make_project_Q`'s `epsilon` must stay small.** `epsilon` floors each
virtual sub-edge within a physical `(i, j)` group, but all of them must still
sum to `Q_bar[i, j]` within `equality_tol`, so too large an `epsilon` makes
the box and equality-band constraints infeasible at larger lifting budgets
(confirmed empirically: `epsilon=1e-3` made every restart at `budget in
{3m, 3.5m}` infeasible). The default `1e-6` leaves residual solver noise that
`_violation_tolerance` absorbs.

**`equality_tol` / `lift_equality_tol` band width.** This controls how
tightly a lifted ergodic flow's `V.T @ Q @ V` must match `Q_bar`. Exact
equality would force `Q_bar`'s row/col sums to match exactly, but they're
only guaranteed to `EQUALITY_ATOL` (e.g. `Q_bar` came from another PGD
projection), so a band of that width keeps the constraint matrix full rank
for OSQP. The San Francisco and CTCV case-study sweeps tighten this to
`1e-4` (vs. the shared default `EQUALITY_ATOL=1e-3`) to keep the lifted
chain's implied physical-node stationary distribution tracking `pi_bar` more
closely — the looser `1e-3` default was found to leave it off by up to
~0.003/node, enough to bias capture-time simulations. If a sweep shows more
`all_failed`/`no_improvement` outcomes than expected, loosen this first.

**OSQP `optimal_inaccurate` status.** `_ACCEPTABLE_STATUSES` in `optimize.py`
accepts both `cp.OPTIMAL` and `cp.OPTIMAL_INACCURATE`, since the latter just
means OSQP missed its own strict convergence certificate, not that the point
is bad — `_projection_violation` is the real feasibility gate.

**`_violation_tolerance` scaling.** The row/col-sum gap `_projection_violation`
checks is a sum of ~n independent per-entry clipping corrections, so a fixed
absolute tolerance under-counts drift at larger lifted state spaces. It
scales `_PROJECTION_VIOLATION_TOL` by `sqrt(n / m)` (1 in the unlifted case
`n == m`).

**`max_grad_norm` in `projected_gradient_descent`.** Near-zero stationary
mass on a state can make `_grad_kemeny`/`_grad_lifted_kemeny`'s adjoint
solves severely ill-conditioned, producing gradient norms orders of
magnitude above the typical ~1e2-1e3 range; an unclipped step then sends the
pre-projection point far outside the feasible region, which is what makes
`project_fn` raise. Clipping the gradient's Frobenius norm before the step
fixes this at the source. By contrast, the JAX-autodiff gradients used for
the Stackelberg and RTE metrics are not destabilized this way (confirmed
empirically across every sweep below), so those generally leave
`max_grad_norm` unset.

## `sweeps.py`: shared helpers

**`_pgd_restarts`.** Runs PGD restarts until `n_init` succeed or the attempt
budget (`max_attempts`, default `2 * n_init`) runs out. A restart that raises
`LinAlgError`/`RuntimeError` draws a new random init and retries rather than
permanently costing a restart slot. Restarts are ranked by their minimum
`hist[-1]`; callers whose `grad_fn` negates a maximization objective (so
`hist` values are `<= 0`) must negate `best_val` back themselves.

## `sweeps.py`: Erdős–Rényi sweeps

All three functions share a pattern: sample `n_graphs` random `G(m, p)`
graphs per edge probability `p`, optimize the physical-space metric via PGD
(resampling on degenerate draws), build a lifting, and optimize the lifted
metric via PGD.

**`erdos_renyi_kemeny_improvement`.** Resamples `(A, pi_bar)` up to
`max_kemeny_attempts=10` times if `K(P_bar*)` stays above 1000
(degenerate/poorly-conditioned pairing). `max_grad_norm_lift` defaults to a
stricter 200 than `max_grad_norm_phys`'s 2000: an empirical sweep found 2000
let ill-conditioned lifted gradients diverge outright in ~1/10 graphs
regardless of `alpha_lift`, while 200 eliminated this and roughly doubled the
mean percent decrease, with no runtime penalty.

**`erdos_renyi_stackelberg_improvement`.** No `pi_bar` is imposed —
Stackelberg enforces irreducibility on its own (a reducible chain has a zero
entry of `Psi`, which can never be the max). `max_capt_attempts` defaults far
higher than Kemeny's retry budget (500 vs. 10): the physical optimum's
success probability varies enormously by `p` (an empirical check found 0/25
draws converged to a nonzero-capture chain at `p=0.3`), so a large retry
budget keeps `RuntimeError` a genuine rare-failure signal rather than
something low `p` triggers routinely. `max_grad_norm_phys`/
`max_grad_norm_lift` default to `None`: unlike Kemeny's adjoint-solve
gradient, this metric's JAX-autodiff softmin gradient isn't destabilized by
near-zero stationary mass, so an empirical sweep found clipping made no
difference.

**`erdos_renyi_rte_improvement`.** `eta` controls truncation length
`K_eta = ceil(1/(eta * pi_bar_min)) - 1`: smaller `eta` gives a tighter RTE
approximation at the cost of a longer unrolled (and more expensive)
gradient. `alpha_lift` is an order of magnitude smaller than `alpha_phys`
since the lifted objective is more step-size sensitive.
`max_grad_norm_phys`/`max_grad_norm_lift` default to `None` for the same
reason as the Stackelberg sweep above. Switching the lifting heuristic from
`degree_lifting` to `stationary_lifting` (fixed `budget=2*m` regardless of
`p`) removed a large `p`-dependent cost blowup: `degree_lifting`'s budget
grows with graph density, and a full sweep with it took ~5 hrs (lifted stage
alone ~234s/graph at `p=0.8` vs. ~37s at `p=0.2`); with `stationary_lifting`,
per-graph cost is flat across `p` (~22-28s physical, ~40-43s lifted),
extrapolating to ~2-2.5 hrs. Cost is dominated by JAX retracing
`make_grad_rte`'s unrolled recursion per graph's fresh `pi_bar`, not the PGD
loop, so trimming `n_init`/`n_iter` buys negligible speedup at the cost of
quality.

## `sweeps.py`: San Francisco case study

All three functions optimize on the fixed 12-node `graph.san_francisco_graph()`
graph (real travel-time weights `W`, crime-rate-derived `pi_bar`), repeating
PGD across `n_trials` random initializations rather than across random
graphs. `w_max` prunes edges longer than `w_max` minutes before optimizing
either space.

**`san_francisco_kemeny_improvement`.** `lifting_budget` defaults to `3*m`: a
budget sweep over `{12, ..., 48}` found 36 best (`K^lift(P*) ~= 21.0-21.2`)
but noisier than `budget=24` (std ~0.16 vs. ~0.02), and `budget=48` failed
outright, so larger budgets aren't guaranteed to work with these
hyperparameters. `alpha_phys=4e-3`, `alpha_lift=2e-2`,
`max_grad_norm_lift=500` were tuned specifically for this graph via grid
sweeps (different from the ER sweeps' own tuning). The lifted stage's
per-attempt failure rate here is non-trivial (QP infeasibility mid-PGD), so
its `_pgd_restarts` call raises `max_attempts` to `4*n_init`. At `budget=24`,
`K(P_bar*) ~= 22.3-22.7` (vs. literature baselines 24.3/21.2) and
`K^lift(P*) ~= 21.5-21.6`; at the current default `budget=36`,
`K^lift(P*) ~= 21.0` across a small scan (not yet benchmarked at
`n_trials=10`).

**`san_francisco_stackelberg_improvement`.** `pi_bar` is *not* imposed (see
the ER Stackelberg rationale above). Lifting is weighted by the *realized*
`pi = best_Q_bar.sum(axis=1)` rather than a target `pi_bar`, so `V`/`W_lift`/
the lifted gradient (and its JIT kernel) are rebuilt every trial. `tau`
defaults to one of two fixed scenarios from RoSSO's own benchmarks on this
graph, selected via `heterogeneous_tau`: `False` (default) is `tau=9`
everywhere (RoSSO's `J_SG=9.75e-2` baseline); `True` is the node-indexed
vector `[8,6,11,10,6,10,9,10,11,9,10,8]` from RoSSO's greedy defense
placement (`J_SG=10.2e-2`). `lifting_budget=2*m`, `alpha_lift=2.0`, and
`max_grad_norm_lift=None` were tuned via grid scans: quality plateaus at
`budget >= 24` and `alpha_lift >= 2.0`, and clipping made no measurable
difference. At these settings, `J^lift(P*)` best=0.1057, mean=0.1018 ± 0.0027
over an 8-trial scan — above the literature baseline of 0.0975.

**`san_francisco_rte_improvement`.** Was previously built around degree
lifting at the full budget (132 virtual states), which made the lifted-stage
projection QP unreliable, same as the Kemeny/Stackelberg functions'
pre-migration history. Switching to stationary lifting at `budget=2*m=24`
and retuning resolved this:
- `max_grad_norm_lift=50` (up from unset): a sweep found the *unclipped*
  lifted gradient at `alpha_lift=1e-3` makes every PGD restart's projection
  step fail outright (0/3 successes across two budgets); clipping to 50
  fixed this completely, while 20 or 100-200 either hurt quality or didn't
  restore reliability. Shrinking `alpha_lift` instead (to `1e-4`) is also
  reliable but converges to a visibly lower `H^lift` for the same iteration
  budget, so clipping at `alpha_lift=1e-3` was preferred.
- `n_iter_lift=300` (up from 50): PGD was still improving at
  `n_iter_lift=150` (`H^lift ~3.66`) and continuing to improve, with
  diminishing returns, out to 500 (`~3.70`); 300 (`~3.69`) is a reasonable
  plateau point.
- `lifting_budget=2*m=24` was kept rather than `3*m`, since a `3*m` check
  showed no meaningful quality gain, only slower per-attempt solves.

Benchmarked at these settings (`n_trials=10`): 10/10 trials succeeded,
improvement = 0.046 ± 0.008, best `H^lift(P*)=3.984` vs. `H(P_bar*)=3.932`,
total wall time ~476s. `make_grad_rte`'s kernel takes ~72s to JIT-compile
once on the full graph (built once per call, not per trial).

**`san_francisco_wmax_sweep` / `_san_francisco_wmax_sweep_one`.** Prunes the
graph at each `w_max` and calls the corresponding `san_francisco_<metric>_
improvement` for every pruning that stays strongly connected. Default
`w_max_values=[4, 5, 6, 7, 8, 9]`: `w_max <= 3` disconnects the pruned SF
digraph (verified numerically), `{4, ..., 9}` keeps it strongly connected,
and 9 is the unpruned graph. For `stackelberg`, since its own `tau` default
is `w_max`-independent (fixed at the unpruned graph's diameter of 9), using
it as-is across a sweep would silently zero out the metric for any `w_max`
whose pruned graph has a larger true diameter. Unless the caller already
passed an explicit `tau`/`heterogeneous_tau=True`, the sweep instead computes
a fixed `tau` from the (weighted) diameter of the sparsest strongly-connected
pruning in `w_max_values`, which upper-bounds every other pruning's own
diameter, and injects it so every `w_max` uses the same valid `tau`.

## `sweeps.py`: CTCV campus graph

`ctcv_graph()` is an 18-node, much sparser graph than San Francisco (~52
nonzero adjacency entries including the diagonal, several degree-1 leaf
nodes) with no natural target stationary distribution (`pi_bar` uniform).
This sparsity/leaf structure makes PGD far more prone to the
ill-conditioned-adjoint-solve blowups described above than SF ever sees.

**`ctcv_kemeny_improvement`.** Hyperparameters were independently tuned
(inheriting SF's Kemeny defaults reliably converged to a *worse* lifted
chain, `K^lift(P*) ~= 600-700`, so callers should not use them here):
- `max_grad_norm_phys=1000` (down from SF's 2000): at 2000-4000, 1-2 of 5
  random restarts diverged to `K(P_bar*)` in the 1e5-1e6 range instead of the
  true optimum ~315.5; 1000 gave 5/5 stable convergence.
- `alpha_lift=1e-2`, `max_grad_norm_lift=20` (down from SF's `2e-2`/500): at
  `max_grad_norm_lift in {100, 200, 500}`, 2-3 of 5 lifted restarts diverged
  to `K^lift` in the 1e3-1e4 range; a sweep over `{8, ..., 50}` at
  `alpha_lift=1e-2` found the whole range gives 0/8 divergences, minimized
  around 20-30 (`K^lift(P*) ~= 72.4` mean, ~71.1-71.8 best); `alpha_lift`
  itself mattered little (0.002-0.04 all converge) once the clip was tight
  enough. At these settings `K^lift(P*) ~= 71-74` vs. `K(P_bar*) ~= 315.5-316`
  — a ~77% reduction, substantially larger than SF's own improvement,
  consistent with the running example's prediction that lifting helps more
  on sparser graphs.
- `lifting_budget=3*m=54` (unchanged from SF's convention): a sweep found
  `K^lift(P*)` improves rapidly from `budget=27` (~229) to `budget=54`
  (~72), then plateaus (`budget=63` ~72.0, `budget=72` ~72.1) while wall
  time keeps growing (~50s at 54 vs. ~480s at 72 for a 5-seed check).

No QP-infeasibility failures were observed during tuning, unlike SF's
non-trivial lifted-stage failure rate, so `max_attempts` here stays at the
`2*n_init` default.

**`ctcv_stackelberg_improvement`.** CTCV has no published Stackelberg
benchmark to stay comparable to, so `tau` defaults to a multiple of
`graph_diameter(A, W)` applied uniformly, in the spirit of the ER sweep's
no-baseline convention — but the raw diameter (53 on the unpruned graph) was
found too tight: physical-space PGD stalls at `J(P_bar*) ~= 0`, since
realized first-passage times under any actual patrol chain run far longer
than a single shortest path on this sparse, near-path-shaped graph. A grid
scan over multiples of the diameter (at `n_iter_phys=n_iter_lift=250`) found
`3x` (159) converges cleanly every time (3/3 trials, mean diff = 0.230 ±
0.121), while larger multiples (5x, 8x, 12x) push the physical-stage `Q_bar`
toward the edge of its feasible region, making the lifted projection QP
intermittently or persistently infeasible. `tau` therefore defaults to
`3 * graph_diameter(A, W)`, computed after any `w_max` pruning.

Hyperparameters are otherwise inherited from
`san_francisco_stackelberg_improvement`; an independent grid scan at
`tau=3*diameter` found them already near-optimal:
- `lift_equality_tol=3e-4` (looser than 1e-4): no quality gain (diff=0.236
  vs. 0.234) and *more* lifted-stage QP retries (9 vs. 0) — kept at 1e-4.
- `max_grad_norm in {20, 50}` (vs. unclipped): looked better in short
  (120-iteration) checks (3-4x faster, zero QP retries, same quality), but a
  full-length (250-iteration) validation told the opposite story: mean diff
  dropped to 0.148 ± 0.133 (vs. unclipped's 0.230 ± 0.121) and one trial
  needed 22 lifted-stage retries anyway. Clipping speeds up escaping bad
  early trajectories but truncates the larger steps this metric's softmin
  surface needs late in a full-length run — kept unclipped (`None`).
- `alpha_lift in {1.0, 4.0}` both underperformed the default 2.0 slightly
  (diff=0.226, 0.222 vs. 0.234).

If a run shows persistent `all_failed`/`no_improvement`, try a smaller `tau`
first — the QP gets more fragile, not less, as `tau` grows past the point
capture becomes easy; tuning alpha/clip further is unlikely to help. The
lifted-stage `_pgd_restarts` call raises `max_attempts` to `8*n_init`: one
full-length validation trial needed 22 of the 24 available attempts before
succeeding. A 3-trial, `n_init=3`, full-length validation run took ~36
minutes total (~12 min/trial, mean diff = 0.230 ± 0.121, 3/3 succeeded) —
substantially slower than SF's own ~5 min for 10 trials at `n_init=10`,
since CTCV's sparser graph needs more restart attempts to clear the
lifted-stage QP.

**`ctcv_rte_improvement`.** All numerical defaults are inherited unchanged
from `san_francisco_rte_improvement` rather than independently retuned: an
empirical check found 0 divergent restarts across repeated `n_trials=5` runs
at `lifting_budget in {2*m, 3*m, 4*m} = {36, 54, 72}` and across two random
seeds at `budget=54` — CTCV's sparser/leafier topology does not destabilize
this metric's JAX-autodiff entropy gradient the way it does Kemeny's
adjoint-solve gradient (the same graph-independence the Stackelberg
functions document for the softmin gradient).

`lifting_budget=2*m=36` (matching SF's RTE convention, unlike the `3*m` used
by the Kemeny/Stackelberg CTCV functions) was chosen because a budget sweep
found essentially no quality gain from a larger budget: mean `H^lift(P*)`
was 2.065 at `budget=36`, 2.096 at 54, 2.099 at 72 (best-of-5: 2.263 / 2.300
/ 2.302), while wall time scaled sharply (~248s / ~643s / ~1275s for the
same 5 trials) — the opposite conclusion from the Kemeny sweep.

Benchmarked at these settings (`n_trials=10`, seed=42): 10/10 trials
succeeded, improvement = 0.056 ± 0.031, best `H^lift(P*)=2.288` vs.
`H(P_bar*)=2.172`, total wall time ~402s (this graph's smaller size and
faster JIT-compile make it substantially cheaper than SF's own ~476s).

## `sweeps.py`: lifting-budget sweep

**`kemeny_lifting_budget_sweep`.** Compares six lifting-budget allocation
heuristics (uniform, stationary, degree, betweenness, eigenvector,
reversible_flow) across a budget sweep over many random graphs.
`max_grad_norm_lift` defaults to a stricter 200 than `max_grad_norm_phys`'s
2000, for the same reason as `erdos_renyi_kemeny_improvement`: near-zero
stationary mass on a virtual state (more likely at larger budgets or skewed
allocations like `reversible_flow_lifting`) can make the adjoint solves
severely ill-conditioned. An empirical sweep across all six methods at
`budget in {2m, 3m}` found 2000 let PGD diverge outright on 2/60
(budget, method) combinations and land on `no_improvement` on 10/60 more,
while 200 eliminated every such case and roughly doubled the mean percent
decrease.

## `sweeps.py`: reproducing published results

The commented-out calls at the bottom of `sweeps.py`'s `__main__` block are
the invocations used to generate the manuscript's saved data files, with
their approximate wall-clock time on the development machine:

| Call | Wall time |
|---|---|
| `erdos_renyi_kemeny_improvement(m=10, p_values=linspace(0.25,0.75,6), n_graphs=20, n_init=10, budget=20, ...)` | ~40 min |
| `erdos_renyi_stackelberg_improvement(m=10, p_values=linspace(0.25,0.75,6), n_graphs=20, n_init=10, budget=20, ...)` | ~100 min |
| `erdos_renyi_rte_improvement(m=10, p_values=linspace(0.25,0.75,6), n_graphs=20, budget=20, n_init=10, ...)` | ~2.5 hr |
| `san_francisco_kemeny_improvement(n_trials=20, n_init=10, n_iter_phys=250, n_iter_lift=250, seed=0)` | ~30 min |
| `san_francisco_stackelberg_improvement(heterogeneous_tau=True, n_trials=10, n_init=10, n_iter_phys=250, n_iter_lift=250)` | ~5 min |
| `san_francisco_rte_improvement(n_trials=10, n_init=10, n_iter_phys=200, n_iter_lift=300)` | ~22 min |
| `ctcv_kemeny_improvement(lifting_budget=3*18, n_trials=10, n_init=3)` | ~12 min |
| `ctcv_stackelberg_improvement(lifting_budget=3*18, n_trials=10, n_init=3, n_iter_phys=250, n_iter_lift=250)` | long |
| `ctcv_rte_improvement(lifting_budget=3*18, n_trials=10)` | ~13 min |
| `san_francisco_wmax_sweep(lifting_budget=36, metric_kwargs={...n_trials=10, n_init=10...})` | ~12 hr |
| `kemeny_lifting_budget_sweep(m=10, n_graphs=20, p_range=(0.2,0.8), budget_values=[1.25,...,3]*m, n_init=10, ...)` | ~260 min |

Restore the exact keyword arguments from git history (or this table's
referenced functions' current defaults, which mostly match) if re-running
one of these.

## `figures.py`

**`fig_erdos_renyi_combined_ridgelines`.** joypy's `joyplot()` always claims
the whole enclosing `Figure` for its shared-x-axis trick (it internally calls
`fig.add_subplot(1, 1, 1)` spanning the entire figure, then
`fig.tight_layout()`), so three joyplots cannot be composed as regions of one
Figure via gridspec/subfigures without each call's full-figure axis and
`tight_layout` clobbering the others' layout (confirmed experimentally;
subfigures additionally raise `AttributeError` since `SubFigure` has no
`tight_layout`). Instead, this function calls each standalone
`fig_erdos_renyi_*_percent_*` function unmodified, rasterizes its output, and
stacks the three images as panels.
