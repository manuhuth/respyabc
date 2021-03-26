import numpy as np


def compute_mean_squared_distance(x, y):
    """Mean squared distance between model output matrices.

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
