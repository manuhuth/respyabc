"""Tests for distance functions."""
import numpy as np

from respyabc.distances import compute_mean_squared_distance


def test_mean_squared_distance_2_times_2():
    x = {"data": np.array([[1, 2], [3, 4]])}
    y = {"data": np.array([[5, 6], [7, 8]])}

    distance = compute_mean_squared_distance(x, y)

    np.testing.assert_almost_equal(distance, 16)


def test_mean_squared_distance_2_times_5():
    x = {"data": np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])}
    y = {"data": np.array([[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]])}

    distance = compute_mean_squared_distance(x, y)

    np.testing.assert_almost_equal(distance, 100)
