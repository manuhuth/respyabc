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


def compute_projection_distance(X):
    """Compute the projection matrix of a given basis of a vector space.

    Parameters
    ----------
    x : np.array
        A matrix with basis vectors as columns.

    Returns
    -------
    projection_out : np.array
        The projection matrix.
    """

    projection_out = X @ np.linalg.inv(X.transpose() @ X) @ X.transpose()

    return projection_out
