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
