import numpy as np


def sort_dist_to_v(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    f = np.vectorize(lambda u: np.linalg.norm(u - v))
    idxs = np.argsort(f(x), axis=0)[:, 0]
    return np.take(x, idxs, axis=0)
