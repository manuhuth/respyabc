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
from pyabc.visualization import plot_kde_2d
from pyabc.visualization import plot_effective_sample_sizes
from pyabc.transition import MultivariateNormalTransition
from pyabc.visualization.credible import compute_kde_max
from pyabc.visualization.credible import compute_credible_interval
from pyabc.visualization.credible import compute_quantile


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


def plot_history_summary_no_kde(
    history,
    size=(12, 8),
):
    """Wrapper to plot the credible intervals of the posterior distribution,
    the sample numbers, the epsilons and the acceptance rates of an pyABC run.

    Parameters
    ----------
    history : pyabc.smc
        An object created by :func:`pyabc.abc.run()` or
        :func:`respyabc.respyabc()`.

    size : tuple, optional
        Tuple of floats that is passed to :func:`plt.gcf().set_size_inches()`.

    Returns
    -------
    Credible intervals of the posterior distribution,
    the sample numbers, the epsilons and the acceptance rates
    """
    fig, ax = plt.subplots(2, 2)

    plot_effective_sample_sizes(history, ax=ax[0][0])
    plot_sample_numbers(history, ax=ax[1][0])
    plot_epsilons(history, ax=ax[0][1])
    plot_acceptance_rates_trajectory(history, ax=ax[1][1])

    plt.gcf().set_size_inches(size)
    plt.gcf().tight_layout()


def plot_multiple_credible_intervals(
    history,
    parameter_names,
    number_rows,
    number_columns,
    confidence_levels=[0.95, 0.9, 0.5],
    size=(12, 8),
    legend_location="lower right",
    delete_axes=None,
):
    """Wrapper to plot the credible intervals for multiple parameters.

    Parameters
    ----------
    history : pyabc.smc
        An object created by :func:`pyabc.abc.run()` or
        :func:`respyabc.respyabc()`.

    parameter_names : list of str
        Strings including the name of the parameter for which
        the posterior should be plotted.

    number_rows : int
        Positive integer indicating the number of rows of plots.

    number_columns : int
        Positive integer indicating the number of plots per column.

    confidence_levels : list, optional
        A list of floats indicating the levels for which the credible
        intervals are computed.

    size : tuple, optional
        Tuple of floats that is passed to :func:`plt.gcf().set_size_inches()`.

    legend_location : str, optional
        Location of legend in plot. Default is "lower right"

    delete_axes : list of integers or None
        If list of integers, list specifies position of plot that should
        be deleted.

    Returns
    -------
    Credible intervals of the posterior distributions.
    """

    fig, ax = plt.subplots(number_rows, number_columns)

    column_index = 0
    row_index = 0
    for index in range(len(parameter_names)):
        if number_rows == 1:
            plot_credible_intervals_pyabc(
                history,
                levels=confidence_levels,
                par_names=[parameter_names[index]],
                ts=range(history.max_t + 1),
                show_mean=True,
                show_kde_max_1d=True,
                arr_ax=ax[column_index],
            )
        else:
            plot_credible_intervals_pyabc(
                history,
                levels=confidence_levels,
                par_names=[parameter_names[index]],
                ts=range(history.max_t + 1),
                show_mean=True,
                show_kde_max_1d=True,
                arr_ax=ax[row_index][column_index],
            )

        column_index += 1
        if (column_index) == number_columns:
            row_index += 1
            column_index = 0
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = lines_labels[0]

    if delete_axes is not None:
        fig.delaxes(ax[delete_axes[0]][delete_axes[1]])
    plt.gcf().set_size_inches(size)
    plt.gcf().tight_layout()


def plot_2d_histogram(
    history,
    parameter_names,
    parameter_true,
    xmin,
    xmax,
    ymin,
    ymax,
    numx=200,
    numy=200,
    label="true theta",
    figsize=(10, 8),
):
    """Wrapper to plot 2 dimensional kernel density estimates.

    Parameters
    ----------
    history : pyabc.smc
        An object created by :func:`pyabc.abc.run()` or
        :func:`respyabc.respyabc()`.

    parameter_names : list of str
        Strings including the name of the parameter for which
        the posterior should be plotted.

    xmin: float
        Minimum value for axes of first parameter.

    xmax: float
        Maximum value for axes of first parameter.

    ymin: float
        Minimum value for axes of second parameter.

    ymax: float
        Maximum value for axes of second parameter.

    label: str, optional
        Label for the true value.

    figsize : tuple, optional
        Tuple of floats that is passed to figsize.

    Returns
    -------
    Plots for two dimensional kernel density estimates over all populations.
    """

    fig = plt.figure(figsize=figsize)
    for t in range(history.max_t + 1):
        ncol = np.ceil(history.max_t / 3)
        if ncol == 0:
            ncol = 1
        ax = fig.add_subplot(3, ncol, t + 1)

        ax = plot_kde_2d(
            *history.get_distribution(m=0, t=t),
            parameter_names[0],
            parameter_names[1],
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            numx=numx,
            numy=numy,
            ax=ax,
        )
        ax.scatter([parameter_true[0]], [parameter_true[1]], color="C1", label=label)
        ax.set_title("Posterior t={}".format(t))

        ax.legend()
    fig.tight_layout()


def plot_credible_intervals_pyabc(
    history,
    m=0,
    ts=None,
    plot_legend=False,
    par_names=None,
    levels=None,
    show_mean=False,
    show_kde_max=False,
    show_kde_max_1d=False,
    size=None,
    refval=None,
    refval_color="C1",
    kde=None,
    kde_1d=None,
    arr_ax=None,
):
    """Taken from pyABC to adjust legend
    settings. Plot credible intervals over time.

    Parameters
    ----------
    history: History
        The history to extract data from.

    m: int, optional (default = 0)
        The id of the model to plot for.

    ts: Union[List[int], int], optional (default = all)
        The time points to plot for.

    par_names: List[str], optional
        The parameter to plot for. If None, then all parameters are used.

    levels: List[float], optional (default = [0.95])
        Confidence intervals to compute.

    show_mean: bool, optional (default = False)
        Whether to show the mean apart from the median as well.

    show_kde_max: bool, optional (default = False)
        Whether to show the one of the sampled points that gives the highest
        KDE value for the specified KDE.
        Note: It is not attemtped to find the overall hightest KDE value, but
        rather the sampled point with the highest value is taken as an
        approximation (of the MAP-value).

    show_kde_max_1d: bool, optional (default = False)
        Same as `show_kde_max`, but here the KDE is applied componentwise.

    size: tuple of float
        Size of the plot.

    refval: dict, optional (default = None)
        A dictionary of reference parameter values to plot for each of
        `par_names`.

    refval_color: str, optional
        Color to use for the reference value.

    kde: Transition, optional (default = MultivariateNormalTransition)
        The KDE to use for `show_kde_max`.

    kde_1d: Transition, optional (default = MultivariateNormalTransition)
        The KDE to use for `show_kde_max_1d`.

    arr_ax: List, optional
        Array of axes to use. Assumed to be a 1-dimensional list.

    Returns
    -------
    arr_ax: Array of generated axes.
    """
    if levels is None:
        levels = [0.95]
    levels = sorted(levels)
    if par_names is None:
        # extract all parameter names
        df, _ = history.get_distribution(m=m)
        par_names = list(df.columns.values)
    # dimensions
    n_par = len(par_names)
    n_confidence = len(levels)
    if ts is None:
        ts = list(range(0, history.max_t + 1))
    n_pop = len(ts)

    # prepare axes
    if arr_ax is None:
        _, arr_ax = plt.subplots(
            nrows=n_par, ncols=1, sharex=False, sharey=False, figsize=size
        )
    if not isinstance(arr_ax, (list, np.ndarray)):
        arr_ax = [arr_ax]
    fig = arr_ax[0].get_figure()

    # prepare matrices
    cis = np.empty((n_par, n_pop, 2 * n_confidence))
    median = np.empty((n_par, n_pop))
    if show_mean:
        mean = np.empty((n_par, n_pop))
    if show_kde_max:
        kde_max = np.empty((n_par, n_pop))
    if show_kde_max_1d:
        kde_max_1d = np.empty((n_par, n_pop))
    if kde is None and show_kde_max:
        kde = MultivariateNormalTransition()
    if kde_1d is None and show_kde_max_1d:
        kde_1d = MultivariateNormalTransition()

    # fill matrices
    # iterate over populations
    for i_t, t in enumerate(ts):
        df, w = history.get_distribution(m=m, t=t)
        # normalize weights to be sure
        w /= w.sum()
        # fit kde
        if show_kde_max:
            _kde_max_pnt = compute_kde_max(kde, df, w)
        # iterate over parameters
        for i_par, par in enumerate(par_names):
            # as numpy array
            vals = np.array(df[par])
            # median
            median[i_par, i_t] = compute_quantile(vals, w, 0.5)
            # mean
            if show_mean:
                mean[i_par, i_t] = np.sum(w * vals)
            # kde max
            if show_kde_max:
                kde_max[i_par, i_t] = _kde_max_pnt[par]
            if show_kde_max_1d:
                _kde_max_1d_pnt = compute_kde_max(kde_1d, df[[par]], w)
                kde_max_1d[i_par, i_t] = _kde_max_1d_pnt[par]
            # levels
            for i_c, confidence in enumerate(levels):
                lb, ub = compute_credible_interval(vals, w, confidence)
                cis[i_par, i_t, i_c] = lb
                cis[i_par, i_t, -1 - i_c] = ub

    # plot
    for i_par, (par, ax) in enumerate(zip(par_names, arr_ax)):
        for i_c, confidence in reversed(list(enumerate(levels))):
            ax.errorbar(
                x=range(n_pop),
                y=median[i_par].flatten(),
                yerr=[
                    median[i_par] - cis[i_par, :, i_c],
                    cis[i_par, :, -1 - i_c] - median[i_par],
                ],
                capsize=(5.0 / n_confidence) * (i_c + 1),
                label="{:.2f}".format(confidence),
            )
        ax.set_title(f"Parameter {par}")
        # mean
        if show_mean:
            ax.plot(range(n_pop), mean[i_par], "x-", label="Mean")
        # kde max
        if show_kde_max:
            ax.plot(range(n_pop), kde_max[i_par], "x-", label="Max KDE")
        if show_kde_max_1d:
            ax.plot(range(n_pop), kde_max_1d[i_par], "x-", label="Max KDE 1d")
        # reference value
        if refval is not None:
            ax.hlines(
                refval[par],
                xmin=0,
                xmax=n_pop - 1,
                color=refval_color,
                label="Reference value",
            )
        ax.set_xticks(range(n_pop))
        ax.set_xticklabels(ts)
        ax.set_ylabel(par)
        if plot_legend is True:
            ax.legend()

    # format
    arr_ax[-1].set_xlabel("Population t")
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return arr_ax
