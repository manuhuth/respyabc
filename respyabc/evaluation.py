"""This module contains all plots and point estimates that can
be used to conduct model evaluation."""

import pyabc
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import norm
from pyabc.visualization import plot_sample_numbers
from pyabc.visualization import plot_epsilons
from pyabc.visualization import plot_acceptance_rates_trajectory


def compute_point_estimate(history, run=None):
    """Returns point estimates for the pyabc run.

    Parameters
    ----------
    history : pyabc.smc
        An object created by :func:`pyabc.abc.run()` or
        :func:`respyabc.respyabc()`.

    run : int, optional
        Positive integer determining which pyabc run should be used. If `None`
        last run is used.

    Returns
    -------
    df_stacked_moments : pandas.DataFrame
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


def compute_distribution_bounds(history, parameter, alpha, run):
    """Returns distribution bounds from pyabc posterior distribution.

    Parameters
    ----------
    history : pyabc.smc
        An object created by :func:`pyabc.abc.run()` or
        :func:`respyabc.respyabc()`.

    parameter : str
        Parameter for which the credible interval should be computed.

    alpha : float
        Level of credibility. Must be between zero and one.

    run : int
        Positive integer determining which pyabc run should be used. If `None`
        last run is used.

    Returns
    -------
    lower: float
        Lower bound of the interval.

    upper: float
        Upper bound of the interval.

    """
    if run is None:
        run = history.max_t

    magnitudes, probabilities = history.get_distribution(m=0, t=run)
    magnitudes["probabilities"] = probabilities
    magnitudes_sorted = magnitudes.sort_values(by=parameter)
    magnitudes_sorted["cum_probabilities"] = magnitudes_sorted["probabilities"].cumsum()
    cut_magnitudes = magnitudes_sorted[
        (magnitudes_sorted["cum_probabilities"] >= alpha / 2)
        & (magnitudes_sorted["cum_probabilities"] <= 1 - alpha / 2)
    ]
    cut_indexed = cut_magnitudes.reset_index(drop=True)
    cut_magnitudes = cut_indexed[parameter]
    lower = cut_magnitudes[0]
    upper = cut_magnitudes[len(cut_magnitudes) - 1]

    return lower, upper


def compute_central_credible_interval(
    history, parameter, interval_type="simulated", alpha=0.05
):
    """Returns credible intervals for the all pyabc runs.

    Parameters
    ----------
    history : pyabc.smc
        An object created by :func:`pyabc.abc.run()` or
        :func:`respyabc.respyabc()`.

    parameter : str
        Parameter for which the credible interval should be computed.

    interval_type : {"simulated", "mean"}, optional
        Method that is used to compute the interval ranges.
        The default is ``"simulated"``.

    alpha : float, optional
        Level of credibility. Must be between zero and one.

    Returns
    -------
    df_ccf : pandas.DataFrame
        Data frame containing the credibility intervals for all runs.

    """
    ccf = np.full([history.max_t + 1, 3], np.nan)
    for t in range(history.max_t + 1):
        df_point_estimate = compute_point_estimate(history, run=t)
        estimate, variance = df_point_estimate[parameter]
        if interval_type == "simulated":
            lower, upper = compute_distribution_bounds(
                history=history, parameter=parameter, alpha=alpha, run=t
            )
        elif interval_type == "mean":
            upper = norm.ppf(1 - alpha / 2, loc=estimate, scale=variance)
            lower = norm.ppf(alpha, loc=estimate, scale=variance)

        ccf[t, :] = np.array([lower, estimate, upper])

    df_ccf = pd.DataFrame(ccf, columns=["lower", "estimate", "upper"])

    return df_ccf


def plot_kernel_density_posterior(history, parameter, xmin, xmax):
    """Plot the Kernel densities of the posterior distribution of an pyABC run.

    Parameters
    ----------
    history : pyabc.smc
        An object created by :func:`pyabc.abc.run()` or
        :func:`respyabc.respyabc()`.

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


def plot_credible_intervals(
    history,
    parameter,
    interval_type="simulated",
    alpha=0.05,
    main_title="Central Credible Intervals",
    y_label=None,
):
    """Plot the credible intervals of the posterior distribution of an pyABC run.

    Parameters
    ----------
    history : pyabc.smc
        An object created by :func:`pyabc.abc.run()` or
        :func:`respyabc.respyabc()`.

    parameter : str
        String including the name of the parameter for which
        the posterior should be plotted.

    interval_type : {"simulated", "mean"}, optional
        Method that is used to compute the interval ranges.
        The default is ``"simulated"``.

    alpha : float, optional
        Level of credibility. Must be between zero and one.

    main_title : str, optional
        Main title of the plot.

    y_label : str or None, default None
        Label of y axis. If `None`, name of parameter is used.

    Returns
    -------
    Plot with the central credible intervals of the parameter.
    """
    if y_label is None:
        y_label = parameter
    df = compute_central_credible_interval(
        history=history, parameter=parameter, interval_type=interval_type, alpha=alpha
    )
    fig, ax = plt.subplots()
    ax.errorbar(
        range(history.max_t + 1),
        df["estimate"],
        yerr=(df["upper"] - df["estimate"]),
        fmt="o",
    )
    ax.set_xticks(np.arange(history.max_t + 1))
    ax.set_ylabel(ylabel=y_label)
    ax.set_xlabel(xlabel="run")
    fig.suptitle(main_title)


def plot_history_summary(
    history,
    parameter_name,
    parameter_value,
    confidence_levels=[0.95, 0.9, 0.5],
    size=(12, 8),
):
    """Wrapper to plot the credible intervals of the posterior distribution,
    the sample numbers, the epsilons and the acceptance rates of an pyABC run.

    Parameters
    ----------
    history : pyabc.smc
        An object created by :func:`pyabc.abc.run()` or
        :func:`respyabc.respyabc()`.

    parameter_name : str
        String including the name of the parameter for which
        the posterior should be plotted.

    parameter_value : float
        Magnitude of true parameter.

    confidence_levels : list, optional
        A list of floats indicating the levels for which the credible
        intervals are computed.

    size : tuple, optional
        Tuple of floats that is passed to :func:`plt.gcf().set_size_inches()`.

    Returns
    -------
    Credible intervals of the posterior distribution,
    the sample numbers, the epsilons and the acceptance rates
    """
    fig, ax = plt.subplots(2, 2)

    pyabc.visualization.plot_credible_intervals(
        history,
        levels=confidence_levels,
        ts=range(history.max_t + 1),
        show_mean=True,
        show_kde_max_1d=True,
        refval={parameter_name: parameter_value},
        arr_ax=ax[0][0],
    )
    plot_sample_numbers(history, ax=ax[1][0])
    plot_epsilons(history, ax=ax[0][1])
    plot_acceptance_rates_trajectory(history, ax=ax[1][1])

    plt.gcf().set_size_inches(size)
    plt.gcf().tight_layout()
