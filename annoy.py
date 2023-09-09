from __future__ import annotations
import time
import numpy as np
import random
import multiprocessing as mp
from pyspark.sql import SparkSession


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

    def query(self, q: np.ndarray, n: int) -> np.ndarray:
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


class SparkAnnoy(object):
    def __init__(self, name: str):
        self._size = 0
        self._trees = None
        self._name = name
        self._spark = (
            SparkSession.builder.appName(self._name)
            .master("local[{}]".format(mp.cpu_count()))
            .getOrCreate()
        )

    @property
    def size(self) -> int:
        return self._size

    def build(self, x: np.ndarray, k: int, shuffle: bool = False):
        if shuffle:
            x = np.copy(x)
            np.random.shuffle(x)
        n_trees = mp.cpu_count()
        x_split = np.array_split(x, n_trees)
        rdd = self._spark.sparkContext.parallelize(x_split)
        self._trees = rdd.mapPartitions(
            lambda x_i: np.array(
                [_AnnoyTree(np.squeeze(np.array(list(x_i)), axis=0), k)]
            )
        )

    def query(self, q: np.ndarray, nn: int):
        res_pool = self._trees.map(lambda x: x.query(q, nn))
        return res_pool.reduce(
            lambda x, y: sort_dist_to_v(q, np.vstack([x, y]))[
                : min(nn, np.shape(x)[0] + np.shape(y)[0])
            ]
        )


if __name__ == "__main__":
    n = 2**20
    m = 2**4
    k = 2**5
    nn = 2**3

    x = np.ndarray(shape=(n, m))
    print("created array: shape = {}".format(np.shape(x)))
    for i in range(0, np.shape(x)[0]):
        x[i] = np.repeat(i, m)
    print("populated array with values")
    np.random.shuffle(x)
    print("shuffled array")

    index_spark = SparkAnnoy("myIndex")
    t_b_0 = time.perf_counter()
    index_spark.build(x, k)
    t_b_1 = time.perf_counter()
    print("took {} s to build spark index".format(t_b_1 - t_b_0))

    index_annoy = AnnoyIndex()
    t_b_0 = time.perf_counter()
    index_annoy.build(x=x, n_trees=mp.cpu_count(), k=k, parallelize=True, shuffle=False)
    t_b_1 = time.perf_counter()
    print("took {} s to build non-spark index".format(t_b_1 - t_b_0))

    while True:
        q = np.repeat(random.randint(0, n), m)
        print(
            "=================================================================================="
        )
        print("query vector:\n{}".format(q))
        t_q_0 = time.perf_counter()
        index_spark.query(q, nn)
        t_q_1 = time.perf_counter()
        print("spark index query time: {} s".format(t_q_1 - t_q_0))

        t_q_0 = time.perf_counter()
        res = index_annoy.query(q, nn)
        t_q_1 = time.perf_counter()
        print("non-spark index query time: {} s".format(t_q_1 - t_q_0))

        time.sleep(5)
