import os
import time
import numpy as np
from spark_annoy.spark_annoy import SparkAnnoyIndex, SparkMode
from test import test_utils

N = 2**20
M = 2**6
K = 2**5
NN = 2**4
ITERS = 100
CSV_FILE_NAME = "test_data/test_index_{}x{}".format(N, M)

if __name__ == "__main__":
    if not os.path.isfile(CSV_FILE_NAME):
        test_utils.create_random_vector_csv(CSV_FILE_NAME, N, M)

    index = SparkAnnoyIndex("csv_input_test", SparkMode.LOCAL)
    t_0 = time.perf_counter()
    index.build_from_csv(CSV_FILE_NAME, K)
    t_1 = time.perf_counter()
    print("build time for index size {} x {}: {}".format(N, M, t_1 - t_0))

    t_avg = 0
    for i in range(0, ITERS):
        q = np.random.randint(-1 * N, N, M)
        t_0 = time.perf_counter()
        index.query(q, NN)
        t_1 = time.perf_counter()
        t_avg = t_1 - t_0
    t_avg /= ITERS
    print("average query time over {} iters: {}".format(ITERS, t_avg))

