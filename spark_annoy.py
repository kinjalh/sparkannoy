import numpy as np
import multiprocessing as mp
from pyspark.sql import SparkSession
from annoy import _AnnoyTree, sort_dist_to_v


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
