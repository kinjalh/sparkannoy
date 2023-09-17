from enum import Enum
import numpy as np
import multiprocessing as mp
from pyspark.sql import SparkSession
from spark_annoy.annoy_internal import _AnnoyTree, sort_dist_to_v


class SparkMode(Enum):
    LOCAL = 1
    STANDALONE = 2


class SparkAnnoyIndex(object):
    def __init__(self, name: str, mode: SparkMode):
        self._size = 0
        self._trees = None
        self._name = name
        if mode == mode.LOCAL:
            self._spark = (
                SparkSession.builder.appName(self._name)
                .master("local[{}]".format(mp.cpu_count()))
                .getOrCreate()
            )
        elif mode == mode.STANDALONE:
            self._spark = (
                SparkSession.builder.appName(self._name)
                .getOrCreate()
            )

    @property
    def size(self) -> int:
        return self._size

    def build(self, x: np.ndarray, k: int, shuffle: bool = False):
        self._size = np.shape(x)[0]
        if shuffle:
            x = np.copy(x)
            np.random.shuffle(x)
        n_trees = min(mp.cpu_count(), np.shape(x)[0])
        x_split = np.array_split(x, n_trees)
        rdd = self._spark.sparkContext.parallelize(x_split, n_trees)
        self._trees = rdd.mapPartitions(
            lambda x_i: np.array(
                [_AnnoyTree(np.squeeze(np.array(list(x_i)), axis=0), k)]
            )
        )

    def build_from_csv(self, filepath: str, k: int):
        n_trees = mp.cpu_count()
        rdd = self._spark.read.csv(filepath).repartition(n_trees).rdd
        self._size = (rdd.count())
        self._trees = rdd.mapPartitions(
            lambda x: np.array([_AnnoyTree(np.array(list(x)), k)])
        )

    def query(self, q: np.ndarray, nn: int) -> np.ndarray:
        res_pool = self._trees.map(lambda x: x.query(q, nn))
        return res_pool.reduce(
            lambda x, y: sort_dist_to_v(q, np.vstack([x, y]))[
                : min(nn, np.shape(x)[0] + np.shape(y)[0])
            ]
        )
