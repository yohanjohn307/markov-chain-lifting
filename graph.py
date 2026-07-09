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


def erdos_renyi_graph(m: int, p: float, seed: int | None = None, max_iter: int = 1000) -> np.ndarray:
    """Sample a connected undirected Erdős-Rényi graph G(m, p).

    Re-samples until connected (required for irreducible Markov chains).
    Returns A: (m, m) symmetric binary adjacency matrix with ones on the diagonal.
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
    """Sample a strongly connected directed Erdős-Rényi digraph D(m, p).

    Edge directions are sampled independently, i.e. A[i, j] and A[j, i] need not
    agree. Re-samples until strongly connected (required for irreducible Markov
    chains). Returns A: (m, m) binary adjacency matrix with ones on the diagonal,
    where A[i, j] = 1 denotes a directed edge i -> j.
    """
    rng = np.random.default_rng(seed)
    for _ in range(max_iter):
        A = (rng.random((m, m)) < p).astype(float)
        np.fill_diagonal(A, 1)
        if _reachable(A) and _reachable(A.T):
            return A
    raise RuntimeError(f"Could not generate a strongly connected D({m}, {p}) after {max_iter} tries; increase p")


def random_chain(A: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Generate a random ergodic flow matrix on a graph.

    Samples i.i.d. Uniform(0, 1) weights on the upper-triangle edges of A and
    symmetrises by reflection, giving a reversible (detailed-balance) ergodic flow
    matrix Q with Q[i, j] = 0 when A[i, j] = 0.
    Returns Q: (m, m) non-negative symmetric matrix satisfying Q 1 = Q^T 1.
    """
    rng = np.random.default_rng(seed)
    U = np.triu(rng.uniform(0.0, 1.0, size=A.shape) * A)
    return U + U.T


def degree_lifting(A: np.ndarray) -> np.ndarray:
    """Build the degree-lifting mapping matrix for a graph.

    Node j gets deg(j) virtual states (one per incident edge), giving n = 2|E| total.
    Virtual states are ordered by physical node: the first deg(0) rows map to node 0, etc.
    Returns V: (n, m) binary mapping matrix with exactly one 1 per row.
    """
    deg = (A - np.diag(np.diag(A))).sum(axis=1).astype(int)
    n = int(deg.sum())
    m = A.shape[0]
    V = np.zeros((n, m), dtype=float)
    idx = 0
    for j, d in enumerate(deg):
        V[idx:idx + d, j] = 1.0
        idx += d
    return V


def prune_long_edges(A: np.ndarray, W: np.ndarray, threshold: float) -> np.ndarray:
    """Remove edges whose travel time exceeds a threshold.

    Returns A_pruned: (m, m) binary adjacency matrix equal to A, but with
    A_pruned[i, j] = 0 wherever W[i, j] > threshold.
    """
    return A * (W <= threshold)


def san_francisco_graph() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the 12-node SF police district graph from the RoSSO paper (John et al., ICRA 2024).

    The 12 nodes are important intersections in a downtown San Francisco police
    district, connected as a complete graph. W(i, j) is the driving time in
    minutes from intersection i to j (asymmetric due to one-way streets etc.),
    originally from Alamdari et al. 2014. The desired stationary distribution pi
    is chosen proportional to the monthly crime rate at each intersection.
    Returns A: (12, 12) binary adjacency matrix (all ones, including diagonal).
    Returns W: (12, 12) travel-time weight matrix.
    Returns pi: (12,) desired stationary distribution, summing to 1.
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

