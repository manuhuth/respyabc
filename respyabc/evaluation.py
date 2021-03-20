import pyabc
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import norm  #


def point_estimate(history, run=None):
    """Returns point estimates for the pyabc run.

    Parameters
    ----------
    history : pyabc.history
        History of the pyabc run.

    run : int
        Positive integer determining which pyabc run should be used. If `None`
        last run is used.

    Returns
    -------
    df_stacked_moments : data.frame
        Data frame including the point estimate and its varianc for all parameters.

    """
    if run is None:
        run = history.max_t

    magnitudes, probabilities = history.get_distribution(m=0, t=run)
    mean = np.array(magnitudes).T @ np.array(probabilities)
    var = np.array(magnitudes.var()) * np.sum(np.array(probabilities) ** 2)
    stacked_moments = np.vstack((mean, var))

    df_stacked_moments = pd.DataFrame(
        stacked_moments, columns=magnitudes.columns, index=["estimate", "est_variance"]
    )

    return df_stacked_moments


def central_credible_interval(history, parameter, alpha):
    """Returns credible intervals for the all pyabc runs.

    Parameters
    ----------
    history : pyabc.history
        History of the pyabc run.

    parameter : str
        Parameter for which the credible interval should be computed.

    alpha : float
        Level of credability. Must be between zero and one.

    Returns
    -------
    df_ccf : data.frame
        Data frame containing the credability intervals for all runs.

    """
    ccf = np.full([history.max_t + 1, 3], np.nan)
    for t in range(history.max_t + 1):
        df_point_estimate = point_estimate(history, run=t)
        estimate, variance = df_point_estimate[parameter]
        upper = norm.ppf(1 - alpha / 2, loc=estimate, scale=variance)
        lower = norm.ppf(alpha, loc=estimate, scale=variance)
        ccf[t, :] = np.array([lower, estimate, upper])

    df_ccf = pd.DataFrame(ccf, columns=["lower", "estimate", "upper"])

    return df_ccf


def plot_kernel_density_posterior(history, parameter, xmin, xmax):
    """Plot the Kernel densities of the posterior distribution of an pyABC run.

    Parameters
    ----------
    history : object
        An object created by abc.run().
    parameter : str
        String including the name of the parameter for which
        the posterior should be plotted.
    xmin : float
        Minimum value for the x-axis' range.
    xmax : float
        Maximum value for the x-axis' range.

    Returns
    -------
    Plot with posterior distribution of parameter.
    """

    fig, ax = plt.subplots()
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df, w, xmin=xmin, xmax=xmax, x=parameter, ax=ax, label="PDF t={}".format(t)
        )
    ax.legend()
