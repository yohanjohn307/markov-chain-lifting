import numpy as np


def erdos_renyi_graph(m: int, p: float, seed: int | None = None) -> np.ndarray:
    """Sample a connected undirected Erdős-Rényi graph G(m, p).

    Re-samples until connected (required for irreducible Markov chains).
    Returns A: (m, m) symmetric binary adjacency matrix with ones on the diagonal.
    """
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        U = (rng.random((m, m)) < p).astype(float)
        A = np.triu(U, k=1)
        A = A + A.T
        np.fill_diagonal(A, 1)
        visited = {0}
        queue = [0]
        while queue:
            v = queue.pop()
            for u in np.where(A[v] > 0)[0]:
                if u not in visited:
                    visited.add(u)
                    queue.append(u)
        if len(visited) == m:
            return A
    raise RuntimeError(f"Could not generate a connected G({m}, {p}) after 10000 tries; increase p")


def random_chain(A: np.ndarray, seed: int | None = None) -> np.ndarray:
    """Generate a random irreducible Markov chain on a graph.

    Each row is sampled from a Dirichlet(1) distribution over the neighbours in A.
    Returns P: (m, m) row-stochastic matrix with P[i, j] = 0 when A[i, j] = 0.
    """
    rng = np.random.default_rng(seed)
    m = A.shape[0]
    P = np.zeros((m, m))
    for i in range(m):
        nbrs = np.where(A[i] > 0)[0]
        weights = rng.exponential(1.0, size=len(nbrs))
        P[i, nbrs] = weights / weights.sum()
    return P


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
