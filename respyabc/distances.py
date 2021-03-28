"""This module hosts all distance functions that are supported
to emasure the distance between the summary statistics
of two populations."""

import numpy as np


def compute_mean_squared_distance(x, y):
    """Mean squared distance between model output matrices.

    Parameters
    ----------
    x : dict
        A dictionary containing ``"data"`` as key with
        corresponding matrix as value.

    y : dict
        A dictionary containing ``"data"`` as key with
        corresponding matrix as value.

    Returns
    -------
    Average squared componentwise differnces between x and y.
    """

    return np.mean((x["data"] - y["data"]) ** 2)
