import multiprocessing as mp
import numpy as np
import time
import random
from spark_annoy.annoy import AnnoyIndex
from spark_annoy.spark_annoy import SparkAnnoyIndex


if __name__ == "__main__":
    n = 2**10
    m = 2**11
    k = 2**5
    nn = 2**3

    x = np.ndarray(shape=(n, m))
    print("created array: shape = {}".format(np.shape(x)))
    for i in range(0, np.shape(x)[0]):
        x[i] = np.repeat(i, m)
    print("populated array with values")
    np.random.shuffle(x)
    print("shuffled array")

    index_spark = SparkAnnoyIndex("myIndex")
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

        t_q_0 = time.perf_counter()
        index_spark.query(q, nn)
        t_q_1 = time.perf_counter()
        print("spark index query time: {} s".format(t_q_1 - t_q_0))

        t_q_0 = time.perf_counter()
        res = index_annoy.query(q, nn)
        t_q_1 = time.perf_counter()
        print("non-spark index query time: {} s".format(t_q_1 - t_q_0))

        time.sleep(5)
