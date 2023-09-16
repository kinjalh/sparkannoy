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
