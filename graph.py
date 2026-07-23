import networkx as nx
import numpy as np


def _reachable(A: np.ndarray) -> bool:
    """Check whether every node is reachable from node 0 by following edges of A."""
    m = A.shape[0]
    visited = {0}
    queue = [0]
    while queue:
        v = queue.pop()
        for u in np.where(A[v] > 0)[0]:
            if u not in visited:
                visited.add(u)
                queue.append(u)
    return len(visited) == m


def is_strongly_connected(A: np.ndarray) -> bool:
    """Check whether every node can reach, and be reached from, every other node."""
    return _reachable(A) and _reachable(A.T)


def erdos_renyi_graph(m: int, p: float, seed: int | None = None, max_iter: int = 1000) -> np.ndarray:
    """Sample a connected undirected Erdős-Rényi graph G(m, p), re-sampling until
    connected (required for irreducible Markov chains). Returns a symmetric (m, m)
    binary adjacency matrix with ones on the diagonal.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_iter):
        U = (rng.random((m, m)) < p).astype(float)
        A = np.triu(U, k=1)
        A = A + A.T
        np.fill_diagonal(A, 1)
        if _reachable(A):
            return A
    raise RuntimeError(f"Could not generate a connected G({m}, {p}) after {max_iter} tries; increase p")


def erdos_renyi_digraph(m: int, p: float, seed: int | None = None, max_iter: int = 1000) -> np.ndarray:
    """Sample a strongly connected directed Erdős-Rényi digraph D(m, p), with edge
    directions sampled independently (A[i, j] and A[j, i] need not agree). Re-samples
    until strongly connected. Returns a binary (m, m) adjacency matrix with ones on
    the diagonal, where A[i, j] = 1 denotes a directed edge i -> j.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_iter):
        A = (rng.random((m, m)) < p).astype(float)
        np.fill_diagonal(A, 1)
        if _reachable(A) and _reachable(A.T):
            return A
    raise RuntimeError(f"Could not generate a strongly connected D({m}, {p}) after {max_iter} tries; increase p")


def random_chain(A: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Generate a random reversible ergodic flow matrix on a graph, by sampling i.i.d.
    Uniform(0, 1) weights on the upper-triangle edges of A and symmetrising by
    reflection. Returns a non-negative symmetric (m, m) matrix Q satisfying Q 1 = Q^T 1,
    with Q[i, j] = 0 where A[i, j] = 0.
    """
    rng = np.random.default_rng(seed)
    U = np.triu(rng.uniform(0.0, 1.0, size=A.shape) * A)
    return U + U.T


def _lifting(deg: np.ndarray) -> np.ndarray:
    n = int(deg.sum())
    m = deg.shape[0]
    V = np.zeros((n, m), dtype=float)
    idx = 0
    for j, d in enumerate(deg):
        V[idx:idx + d, j] = 1.0
        idx += d
    return V


def proportional_lifting(weights: np.ndarray, budget: int) -> np.ndarray:
    """Build a (budget, m) lifting mapping matrix with virtual-state counts
    proportional to weights. Each physical node gets at least one virtual state;
    the remaining budget - m are apportioned by the largest-remainder (Hamilton)
    method so counts sum exactly to budget.
    """
    weights = np.asarray(weights, dtype=float)
    m = weights.shape[0]
    if budget < m:
        raise ValueError(f"budget ({budget}) must be at least the number of nodes ({m})")
    remaining = budget - m
    if weights.sum() > 0:
        shares = remaining * weights / weights.sum()
    else:
        shares = np.full(m, remaining / m)
    extra = np.floor(shares).astype(int)
    leftover = remaining - int(extra.sum())
    frac = shares - extra
    order = np.argsort(-frac)
    extra[order[:leftover]] += 1
    counts = np.ones(m, dtype=int) + extra
    return _lifting(counts)


def uniform_lifting(A: np.ndarray, budget: int) -> np.ndarray:
    """proportional_lifting with uniform weights: virtual states split evenly across nodes."""
    m = A.shape[0]
    return proportional_lifting(np.ones(m), budget)


def stationary_lifting(pi_bar: np.ndarray, budget: int) -> np.ndarray:
    """proportional_lifting weighted by pi_bar: higher-stationary-probability nodes
    (visited more often) get more virtual states."""
    return proportional_lifting(pi_bar, budget)


def degree_lifting(A: np.ndarray, budget: int) -> np.ndarray:
    """proportional_lifting weighted by max(indegree, outdegree): nodes with more
    incident edges get more virtual states."""
    A_off = A - np.diag(np.diag(A))
    outdeg = A_off.sum(axis=1)
    indeg = A_off.sum(axis=0)
    weights = np.maximum(indeg, outdeg)
    return proportional_lifting(weights, budget)


def betweenness_lifting(A: np.ndarray, budget: int) -> np.ndarray:
    """proportional_lifting weighted by (directed) betweenness centrality: nodes on
    more shortest paths get more virtual states."""
    m = A.shape[0]
    G = nx.from_numpy_array(A - np.diag(np.diag(A)), create_using=nx.DiGraph)
    bc = nx.betweenness_centrality(G)
    weights = np.array([bc[i] for i in range(m)])
    return proportional_lifting(weights, budget)


def eigenvector_lifting(A: np.ndarray, budget: int) -> np.ndarray:
    """proportional_lifting weighted by (directed) eigenvector centrality: nodes
    connected to other well-connected nodes get more virtual states."""
    m = A.shape[0]
    G = nx.from_numpy_array(A - np.diag(np.diag(A)), create_using=nx.DiGraph)
    ec = nx.eigenvector_centrality(G, max_iter=1000)
    weights = np.array([ec[i] for i in range(m)])
    return proportional_lifting(weights, budget)


def reversible_flow_lifting(Q_bar: np.ndarray, budget: int) -> np.ndarray:
    """proportional_lifting weighted by x_i = sum_j min(q_bar_ij, q_bar_ji), the
    portion of node i's ergodic flow balanced (back-and-forth) with its neighbors."""
    weights = np.minimum(Q_bar, Q_bar.T).sum(axis=1)
    return proportional_lifting(weights, budget)


def prune_long_edges(A: np.ndarray, W: np.ndarray, threshold: float) -> np.ndarray:
    """Remove edges of A whose travel time in W exceeds threshold."""
    return A * (W <= threshold)


def graph_diameter(A: np.ndarray, W: np.ndarray | None = None) -> int:
    """(Weighted) diameter of a strongly connected digraph A: the largest shortest
    directed path length between any ordered pair of distinct nodes, excluding
    self-loops. Edge lengths come from W if given, else unit (hop count). Raises via
    networkx if A is not strongly connected (see is_strongly_connected).
    """
    weights = A if W is None else W * A
    weights = weights * (1 - np.eye(A.shape[0]))
    G = nx.from_numpy_array(weights, create_using=nx.DiGraph)
    return int(round(nx.diameter(G, weight='weight')))


def san_francisco_graph() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 12-node SF police district graph from the RoSSO paper (John et al.,
    ICRA 2024): a complete graph over downtown SF intersections, with W(i, j) the
    (asymmetric) driving time in minutes (Alamdari et al. 2014) and pi_bar proportional
    to monthly crime rate per intersection. Returns (A, W, pi_bar).
    """
    Wbar = np.array([
        [1, 3, 3, 5, 4, 6, 3, 5, 7, 4, 6, 6],
        [3, 1, 5, 4, 2, 4, 4, 5, 5, 3, 5, 5],
        [3, 5, 1, 7, 6, 8, 3, 4, 9, 4, 8, 7],
        [6, 4, 7, 1, 5, 6, 4, 7, 5, 6, 6, 7],
        [4, 3, 6, 5, 1, 3, 5, 5, 6, 3, 4, 4],
        [6, 4, 8, 5, 3, 1, 6, 7, 3, 6, 2, 3],
        [2, 5, 3, 5, 6, 7, 1, 5, 7, 5, 7, 8],
        [3, 5, 2, 7, 6, 7, 3, 1, 9, 3, 7, 5],
        [8, 6, 9, 4, 6, 4, 6, 9, 1, 8, 5, 7],
        [4, 3, 4, 6, 3, 5, 5, 3, 7, 1, 5, 3],
        [6, 4, 8, 6, 4, 2, 6, 6, 4, 5, 1, 3],
        [6, 4, 6, 6, 3, 3, 6, 4, 5, 3, 2, 1],
    ], dtype=float)
    A = np.ones_like(Wbar)
    pi_bar = np.array([133, 90, 89, 87, 83, 83, 74, 64, 48, 43, 38, 34], dtype=float) / 866
    return A, Wbar, pi_bar


def ctcv_graph() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 18-node CTCV campus graph from the patrolling_sim benchmark (David
    Portugal, github.com/davidbsp/patrolling_sim, maps/ctcv/ctcv.graph): a sparse
    undirected graph of waypoints at University of Coimbra, with W(i, j) the edge
    distance in meters (converted from the file's pixel distances at 0.05 m/pixel).
    No natural stationary distribution exists for this graph, so pi_bar is uniform.
    Returns (A, W, pi_bar).
    """
    Wbar = np.array([
        [1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [7, 1, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3, 1, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 9, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 1, 7, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 1, 6, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 1, 3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1, 4],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1],
    ], dtype=float)
    A = (Wbar > 0).astype(float)
    pi_bar = np.full(18, 1.0 / 18)
    return A, Wbar, pi_bar


def airport_graph() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 5-node airport hub graph from Patel, Agharkar, and Bullo (IEEE TAC,
    2015), Fig. 3: a complete graph over SEA, JFK, LAX, ANC, and ORL, with W(i, j) the
    travel time between hubs plus service time at hub j (self-loops carry only service
    time). pi_bar is uniform. Returns (A, W, pi_bar).
    """
    Wbar = np.array([
        [1, 6, 6, 2, 3],
        [6, 1, 3, 6, 12],
        [6, 3, 1, 5, 15],
        [2, 6, 5, 1, 5],
        [3, 12, 15, 5, 1]
    ], dtype=float)
    A = np.ones_like(Wbar)
    pi_bar = np.full(5, 1.0 / 5)
    return A, Wbar, pi_bar