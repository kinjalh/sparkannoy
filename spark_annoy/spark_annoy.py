import numpy as np
import csv
import time
from tqdm import tqdm
import multiprocessing as mp
from pyspark.sql import SparkSession
from spark_annoy.annoy_internal import _AnnoyTree, sort_dist_to_v
from spark_annoy.annoy import AnnoyIndex


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


if __name__ == "__main__":
    filename = "test.csv"
    n = 2**22
    m = 64
    k = 2**4

    # with open(filename, "w") as f:
    #     writer = csv.writer(f)
    #     for i in tqdm(range(0, n)):
    #         v = np.random.randint(low=0, high=n, size=m)
    #         writer.writerow(v)

    index = SparkAnnoy("test_csv")
    t_start = time.perf_counter()
    index.build_from_csv(filename, k)
    t_end = time.perf_counter()
    print('time to build spark index: {}'.format(t_end - t_start))

    x = np.random.randint(low=0, high=n, size=(n, m))
    index = AnnoyIndex()
    t_start = time.perf_counter()
    index.build(x=x, n_trees=mp.cpu_count(), k=k, parallelize=True, shuffle=False)
    t_end = time.perf_counter()
    print('time to build standard index: {}'.format(t_end - t_start))
