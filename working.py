
import numpy as np

def stationary_distribution(Pbar: np.ndarray) -> np.ndarray:
    n = Pbar.shape[0]
    A = Pbar.T - np.eye(n) + np.ones((n, n))
    b = np.ones(n)
    return np.linalg.solve(A, b)

def kemeny(Pbar: np.ndarray) -> float:
    n = Pbar.shape[0]
    pi = stationary_distribution(Pbar)
    M = np.zeros((n, n))
    for j in range(n):
        gamma_j = np.ones(n)
        gamma_j[j] = 0
        M[:, j] = np.linalg.solve(np.eye(n) - Pbar @ np.diag(gamma_j), np.ones(n))
    return float(pi @ M @ pi)

def lifted_kemeny(P: np.ndarray, V: np.ndarray) -> float:
    n, m = V.shape
    pi = stationary_distribution(P)
    M_lift = np.zeros((n, m))
    for j in range(m):
        D_j = np.diag(V[:, j])
        M_lift[:, j] = np.linalg.solve(np.eye(n) - P + P @ D_j, np.ones(n))
    return float( pi @ M_lift @ V.T @ pi )

if __name__ == "__main__":
    Pbar = np.array([[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0, 0, 1, 0]])
    print("Stationary distribution:", stationary_distribution(Pbar))
    print("Kemeny's constant:", kemeny(Pbar))

    p = 1.0
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
    print("Stationary distribution:", stationary_distribution(P))
    print("Kemeny's constant:", kemeny(P))
    print("Lifted Kemeny's constant:", lifted_kemeny(P, V))