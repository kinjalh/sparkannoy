from __future__ import annotations
import numpy as np
import random


class AnnoyIndex(object):

    def __init__(self, x: np.ndarray, k: int):
        self._root = self._build_tree(x, k)

    def _build_tree(self, x: np.ndarray, k: int) -> _AnnoyNode:
        n = np.shape(x)[0]
        if n <= k:
            return _AnnoyNode(x, is_leaf=True)

        i, j = random.sample(range(0, n), 2)
        node = _AnnoyNode(np.stack([x[i], x[j]]))

        d_i = np.linalg.norm(x - x[i], axis=1)
        d_j = np.linalg.norm(x - x[j], axis=1)
        left = x[np.where(d_i <= d_j)]
        right = x[np.where(d_i > d_j)]
        node.left = self._build_tree(left, k)
        node._right = self._build_tree(right, k)

        return node

    def query(self, q: np.ndarray):
        node = self._root
        while not node.is_leaf:
            d_left = np.linalg.norm(q - node.vects[0])
            d_right = np.linalg.norm(q - node.vects[1])
            node = node.left if d_left < d_right else node._right

        best_match = node.vects[0]
        for v in node.vects:
            if np.linalg.norm(q - v) < np.linalg.norm(q - best_match):
                best_match = v
        return best_match


class _AnnoyNode(object):
    def __init__(
        self,
        vects: np.ndarray,
        left: _AnnoyNode = None,
        right: _AnnoyNode = None,
        is_leaf: bool = False
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
    def left(self) -> _AnnoyNode:
        return self._left

    @left.setter
    def left(self, left: _AnnoyNode) -> None:
        self._left = left

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    @is_leaf.setter
    def is_leaf(self, is_leaf: bool) -> None:
        self._is_leaf = is_leaf


if __name__ == "__main__":
    x = np.ndarray(shape=(1000, 3))
    for i in range(0, np.shape(x)[0]):
        x[i] = np.array([i, i, i])
    index = AnnoyIndex(x, 5)
    print(index.query(np.array([11, 12, 10.5])))
