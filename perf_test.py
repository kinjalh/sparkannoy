import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import time
import annoy
import utils
import matplotlib.pyplot as plt


def linear_search(x: np.ndarray, q: np.ndarray, n: int):
    return utils.sort_dist_to_v(q, x)[:n]


if __name__ == "__main__":
    iters = 100  # num iters to average over
    m = 16  # dim of each vector
    nn = 20  # number of nearest neighbors
    n_cores = mp.cpu_count()  # number of CPU cores (max num. trees)

    n_min = 100
    n_max = 1000
    n_step = 100

    perf_exhaustive = []
    perf_build_st = []
    perf_build_parallel = []
    perf_query_st = []
    perf_query_parallel = []
    for i in range(1, mp.cpu_count() + 1):
        perf_build_st.append([])
        perf_build_parallel.append([])
        perf_query_st.append([])
        perf_query_parallel.append([])

    for n in range(n_min, n_max + 1, n_step):
        print("iters = {}, size = {} x {}, nn = {}\n".format(iters, n, m, nn))

        x = np.random.rand(n, m)

        t_exh = 0
        for i in tqdm(range(iters)):
            q = np.random.rand(m)
            t_0 = time.perf_counter()
            linear_search(x, q, nn)
            t_1 = time.perf_counter()
            t_exh += t_1 - t_0
        t_exh /= iters
        perf_exhaustive.append(t_exh)
        print("exhaustive search: {}".format(t_exh))

        k = 5

        for trees in range(1, mp.cpu_count() + 1):
            t_build_single_thread = 0
            t_build_parallel = 0
            t_query_single_thread = 0
            t_query_parallel = 0
            for i in tqdm(range(iters)):
                q = np.random.rand(m)
                index = annoy.AnnoyIndex()
                t_0 = time.perf_counter()
                index.build(x=x, n_trees=trees, k=k, parallelize=False, shuffle=False)
                t_1 = time.perf_counter()
                index.build(x=x, n_trees=trees, k=k, parallelize=True, shuffle=False)
                t_2 = time.perf_counter()
                index.query(q=q, n=nn, parallelize=False)
                t_3 = time.perf_counter()
                index.query(q=q, n=nn, parallelize=True)
                t_4 = time.perf_counter()

                t_build_single_thread += t_1 - t_0
                t_build_parallel += t_2 - t_1
                t_query_single_thread += t_3 - t_2
                t_query_parallel += t_4 - t_3

            t_build_single_thread /= iters
            t_build_parallel /= iters
            t_query_single_thread /= iters
            t_query_parallel /= iters

            perf_build_st[trees - 1].append(t_build_single_thread)
            perf_build_parallel[trees - 1].append(t_build_parallel)
            perf_query_st[trees - 1].append(t_query_single_thread)
            perf_query_parallel[trees - 1].append(t_query_parallel)

            print(
                (
                    "MP: trees: {}, build(s.t.): {}, build(parallel): {}, query (s.t.): {}, "
                    + "query(parallel): {}"
                ).format(
                    trees,
                    t_build_single_thread,
                    t_build_parallel,
                    t_query_single_thread,
                    t_query_parallel,
                )
            )

    x_plt = np.arange(start=n_min, stop=n_max + 1, step=n_step)
    plt.plot(x_plt, perf_exhaustive, label="exhaustive search")
    for i in range(1, mp.cpu_count() + 1):
        plt.plot(
            x_plt, perf_build_st[i - 1], label="build-single-thread-{}-trees".format(i)
        )
        plt.plot(
            x_plt, perf_build_parallel[i - 1], label="build-parallel-{}-trees".format(i)
        )
        plt.plot(
            x_plt, perf_query_st[i - 1], label="query-single-thread-{}-trees".format(i)
        )
        plt.plot(
            x_plt, perf_query_parallel[i - 1], label="query-parallel-{}-trees".format(i)
        )
    plt.legend()
    plt.show()
