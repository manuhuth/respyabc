import numpy as np


def distance_mean_squared(x, y):
    """Distance between squared frequency matrices.

    Parameters
    ----------
    x : dictionary
        A dictionary created by model_delta.

    y : dictionary
        A dictionary created by model_delta.

    Returns
    -------
    Average squared componentwise differnces between x and y.
    """

    return np.mean((x["data"] - y["data"]) ** 2)
