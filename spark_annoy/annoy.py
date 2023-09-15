from __future__ import annotations
import numpy as np
import multiprocessing as mp
from spark_annoy.annoy_internal import _AnnoyTree, sort_dist_to_v


class AnnoyIndex(object):
    def __init__(self):
        self._size = 0
        self._k = 0
        self._n = 0
        self._q = np.zeros((0, 0))
        self._trees = []

    @property
    def size(self) -> int:
        return self._size

    def build(
        self,
        x: np.ndarray,
        n_trees: int,
        k: int,
        parallelize: bool = True,
        shuffle: bool = False,
    ) -> None:
        self._k = k

        if shuffle:
            x = np.copy(x)
            np.random.shuffle(x)
        x_splits = np.array_split(x, n_trees)
        if parallelize:
            with mp.Pool(n_trees) as p:
                self._trees = p.map(self._build_tree, x_splits)
        else:
            self._trees = list(map(self._build_tree, x_splits))
        self._size = np.shape(x)[0]

    def query(self, q: np.ndarray, n: int, parallelize: bool = False) -> np.ndarray:
        self._q = q
        self._n = n

        res_pool = []
        if parallelize:
            with mp.Pool(len(self._trees)) as p:
                res_pool = p.map(self._query_tree, self._trees)
        else:
            res_pool = list(map(self._query_tree, self._trees))
        res = sort_dist_to_v(q, np.vstack(res_pool))
        return res[: min(np.shape(res)[0], n)]

    def _build_tree(self, x: np.ndarray) -> _AnnoyTree:
        return _AnnoyTree(x, self._k)

    def _query_tree(self, t: _AnnoyTree) -> np.ndarray:
        return t.query(self._q, self._n)
