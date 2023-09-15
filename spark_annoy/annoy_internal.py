
from __future__ import annotations
import numpy as np
import random


def sort_dist_to_v(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    f = np.vectorize(lambda u: np.linalg.norm(u - v))
    idxs = np.argsort(f(x), axis=0)[:, 0]
    return np.take(x, idxs, axis=0)


class _AnnoyNode(object):
    def __init__(
        self,
        vects: np.ndarray,
        left: _AnnoyNode = None,
        right: _AnnoyNode = None,
        is_leaf: bool = False,
    ):
        self._vects = vects
        self._left = left
        self._right = right
        self._is_leaf = is_leaf

    @property
    def vects(self) -> np.ndarray:
        return self._vects

    @vects.setter
    def vects(self, vects: np.ndarray) -> None:
        self._vects = vects

    @property
    def left(self) -> _AnnoyNode:
        return self._left

    @left.setter
    def left(self, left: _AnnoyNode) -> None:
        self._left = left

    @property
    def right(self) -> _AnnoyNode:
        return self._right

    @right.setter
    def right(self, right: _AnnoyNode) -> None:
        self._right = right

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    @is_leaf.setter
    def is_leaf(self, is_leaf: bool) -> None:
        self._is_leaf = is_leaf


class _AnnoyTree(object):
    def __init__(self, x: np.ndarray, k: int):
        self._size = np.shape(x)[0]
        self._root = self._build_tree(x, k)

    @property
    def size(self, size: int) -> int:
        return self._size

    def _build_tree(self, x: np.ndarray, k: int) -> _AnnoyNode:
        n = np.shape(x)[0]
        if n <= k:
            return _AnnoyNode(x, is_leaf=True)

        i, j = random.sample(range(0, n), 2)
        node = _AnnoyNode(np.vstack([x[i], x[j]]))

        d_i = np.linalg.norm(x - x[i], axis=1)
        d_j = np.linalg.norm(x - x[j], axis=1)
        left = x[np.where(d_i < d_j)]
        right = x[np.where(d_i >= d_j)]
        node.left = self._build_tree(left, k)
        node._right = self._build_tree(right, k)

        return node

    def query(self, q: np.ndarray, n: int):
        res = self._query_rec(self._root, q, n)
        r = sort_dist_to_v(q, res)
        return r

    def _query_rec(self, node: _AnnoyNode, q: np.ndarray, n: int) -> np.ndarray:
        if node.is_leaf:
            return node.vects[: min(np.shape(node.vects)[0], n)]

        d_left = np.linalg.norm(q - node.vects[0])
        d_right = np.linalg.norm(q - node.vects[1])
        left = d_left < d_right
        res = self._query_rec(node.left if left else node.right, q, n)
        size = np.shape(res)[0]
        if size < n:
            res = np.vstack(
                (res, self._query_rec(node.right if left else node.left, q, n))
            )
        return res[: min(np.shape(res)[0], n)]
