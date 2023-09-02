from __future__ import annotations
import numpy as np
from multiprocessing import Pool
from annoy import AnnoyIndex
import time


class ParallelIndex(object):

    def __init__(self, x: np.ndarray, k: int, threads: int = 1):
        subarrays = np.array_split(x, threads)
        self._indices = [AnnoyIndex(s, k) for s in subarrays]
        self._threads = threads
        self._query = np.zeros(shape=(1, np.shape(x)[1]))

    def query(self, q: np.ndarray):
        self._query = q
        with Pool(self._threads) as p:
            res_pool = p.map(self._query_single_index, self._indices)

            best_match = res_pool[0]
            for v in res_pool:
                if np.linalg.norm(q - v) < np.linalg.norm(q - best_match):
                    best_match = v
            return best_match

    def _query_single_index(self, index: AnnoyIndex) -> np.ndarray:
        return index.query(self._query)


if __name__ == "__main__":
    n = 50000
    m = 1000
    k = 10
    max_threads = 8

    x = np.random.rand(n, m)
    print("test matrix shape: {}, k = {}".format(x.shape, k))

    for i in range(1, max_threads + 1):
        t_0 = time.time()
        idx = ParallelIndex(x, k, i)
        t_1 = time.time()
        idx.query(np.random.rand(1, m))
        t_2 = time.time()
        build_time = t_1 - t_0
        query_time = t_2 - t_1
        print("{} threads: {} build, {} query".format(i, build_time, query_time))
