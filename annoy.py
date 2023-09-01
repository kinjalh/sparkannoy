from __future__ import annotations
import numpy as np
import random


class AnnoyIndex(object):

    def __init__(self):
        self._root = None

    def build(self, x: np.ndarray, k: int):
        self._root = _build_tree(x, k)


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


def _build_tree(x: np.ndarray, k: int) -> _AnnoyNode:
    n = np.shape(x)[0]
    if n <= k:
        return _AnnoyNode(x, is_leaf=True)

    i, j = random.sample(range(0, n), 2)
    print("x[{}] = {}".format(i, x[i]))
    print("x[{}] = {}".format(j, x[j]))
    node = _AnnoyNode(np.stack([x[i], x[j]]))

    d_i = np.linalg.norm(x - x[i], axis=1)
    d_j = np.linalg.norm(x - x[j], axis=1)
    left = x[np.where(d_i <= d_j)]
    right = x[np.where(d_i > d_j)]
    node.left = _build_tree(left, k)
    node._right = _build_tree(right, k)

    return node


if __name__ == "__main__":
    x = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    index = AnnoyIndex()
    index.build(x, 2)
