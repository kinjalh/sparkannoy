import pytest
import numpy as np
from spark_annoy.spark_annoy import SparkAnnoyIndex, SparkMode


@pytest.mark.xfail(raises=ValueError)
def test_empty_input():
    x = np.zeros(shape=(0, 8))
    idx = SparkAnnoyIndex("empty_input_test", SparkMode.LOCAL)
    idx.build(x, 1)


def test_single_vector_index():
    x = np.zeros(shape=(1, 3))
    idx = SparkAnnoyIndex("single_vector_input", SparkMode.LOCAL)
    idx.build(x, 5)
    assert np.array_equal(idx.query(np.array([1, 3, 2]), 12), np.zeros(shape=(1, 3)))


def test_accuracy_small_index():
    n = 1000
    m = 4
    x = np.zeros(shape=(n, m))
    for i in range(0, n):
        x[i, :] = np.repeat(i, m)
    index = SparkAnnoyIndex("accuracy_test_small", SparkMode.LOCAL)
    index.build(x, 1, True)
    res = np.sort(index.query(np.array([28, 28, 28, 28]), 5))
    assert np.array_equal(
        np.sort(res, axis=0),
        np.array(
            [
                [26, 26, 26, 26],
                [27, 27, 27, 27],
                [28, 28, 28, 28],
                [29, 29, 29, 29],
                [30, 30, 30, 30],
            ]
        ),
    )
