"""This module hosts useful tools that can be integrated in one's
workflow or are used to write more concise tests."""

import respy as rp
import pyabc
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

from respyabc.models import compute_model
from respyabc.distances import compute_mean_squared_distance
from respyabc.respyabc import respyabc


def convert_time(seconds):
    """Takes seconds as input and turns it into minutes or hours, if necessary.

    Parameters
    ----------
    seconds : float
        Time in seconds.

    Returns
    -------
    time : float
        Magnitude of time.

    unit : str
        Time unit. Either be seconds, minutes or hours.
    """
    unit = "seconds"
    time = seconds
    if time >= 60:
        time = time / 60
        unit = "minutes"
        if time >= 60:
            time = time / 60
            unit = "hours"

    return time, unit


def prepare_test_respyabc(parameter_true, prior_low, prior_size, descriptives):
    """Wrapes all steps to run respyabc for one parameter.

    Parameters
    ----------
    parameter_true : dict
        A dictionary containing the true parameter

    prior_low : float
        A float with the lower bound for the uniform prior.

    prior_size : float
        A float containing the length of the uniform prior.

    descriptives : {"choice_frequencies", "wage_moments"}
        Determines how the descriptives with which the distance is computed
        are computed.

    Returns
    -------
    Runs respyabc for the specified parameter.
    """

    params, options, data_stored = rp.get_example_model("kw_94_one")
    options["n_periods"] = 40
    options["simulation_agents"] = 1000
    model_to_simulate = rp.get_simulate_func(params, options)

    data = compute_model(
        parameter_true,
        model_to_simulate=model_to_simulate,
        parameter_for_simulation=params,
        options_for_simulation=options,
        descriptives=descriptives,
    )

    key = list(parameter_true.keys())
    parameters_prior = {key[0]: [[prior_low, prior_size], "uniform"]}

    history = respyabc(
        model=compute_model,
        parameters_prior=parameters_prior,
        data=data,
        distance_abc=compute_mean_squared_distance,
        descriptives=descriptives,
        sampler=pyabc.sampler.MulticoreEvalParallelSampler(),
        population_size_abc=2,
        max_nr_populations_abc=1,
        minimum_epsilon_abc=0.2,
        database_path_abc=None,
    )

    return history


def prepare_test_respyabc_model_selection(
    parameter_true, prior_low, prior_size, descriptives
):
    """Wrapes all steps to run respyabc for one parameter.

    Parameters
    ----------
    parameter_true : dict
        A dictionary containing the true parameter

    prior_low : float
        A float with the lower bound for the uniform prior.

    prior_size : float
        A float containing the length of the uniform prior.

    descriptives : {"choice_frequencies", "wage_moments"}
        Determines how the descriptives with which the distance is computed
        are computed. The default is ``"choice_frequencies"``.

    Returns
    -------
    Runs respyabc for the specified parameter.
    """

    params, options, data_stored = rp.get_example_model("kw_94_one")
    options["n_periods"] = 40
    options["simulation_agents"] = 1000
    model_to_simulate = rp.get_simulate_func(params, options)

    data = compute_model(
        parameter_true,
        model_to_simulate=model_to_simulate,
        parameter_for_simulation=params,
        options_for_simulation=options,
        descriptives=descriptives,
    )

    key = list(parameter_true.keys())
    parameters_prior = [
        {key[0]: [[prior_low, prior_size], "uniform"]},
        {key[0]: [[prior_low, prior_size], "uniform"]},
    ]

    history = respyabc(
        model=[compute_model, compute_model],
        parameters_prior=parameters_prior,
        data=data,
        distance_abc=compute_mean_squared_distance,
        descriptives=descriptives,
        sampler=pyabc.sampler.MulticoreEvalParallelSampler(),
        population_size_abc=2,
        max_nr_populations_abc=1,
        minimum_epsilon_abc=0.2,
        database_path_abc=None,
        model_selection=True,
    )

    return history


def prepare_test_respyabc_two_params(
    parameter_true, prior_low, prior_size, descriptives
):
    """Wrapes all steps to run respyabc for two parameter.

    Parameters
    ----------
    parameter_true : dict
        A dictionary containing the true parameter

    prior_low : float
        A float with the lower bound for the uniform prior.

    prior_size : float
        A float containing the length of the uniform prior.

    descriptives : {"choice_frequencies", "wage_moments"}
        Determines how the descriptives with which the distance is computed
        are computed.

    Returns
    -------
    Runs respyabc for the specified parameter.
    """

    params, options, data_stored = rp.get_example_model("kw_94_one")
    options["n_periods"] = 40
    options["simulation_agents"] = 1000
    model_to_simulate = rp.get_simulate_func(params, options)

    data = compute_model(
        parameter_true,
        model_to_simulate=model_to_simulate,
        parameter_for_simulation=params,
        options_for_simulation=options,
        descriptives=descriptives,
    )

    key = list(parameter_true.keys())
    parameters_prior = {
        key[0]: [[prior_low[0], prior_size[0]], "uniform"],
        key[1]: [[prior_low[1], prior_size[1]], "uniform"],
    }

    history = respyabc(
        model=compute_model,
        parameters_prior=parameters_prior,
        data=data,
        distance_abc=compute_mean_squared_distance,
        descriptives=descriptives,
        sampler=pyabc.sampler.MulticoreEvalParallelSampler(),
        population_size_abc=2,
        max_nr_populations_abc=1,
        minimum_epsilon_abc=0.2,
        database_path_abc=None,
    )

    return history


def plot_normal_densities(
    mu1, var1, mu2, var2, vertical_marker, title="Normal prior densities"
):
    """plot two normal densities

    Parameters
    ----------
    mu1 : float
        Mean of first normal random variable.

    var1 : float
        Variance of first normal random variable.

    mu2 : float
        Mean of second normal random variable.

    var2 : float
        Variance of second normal random variable.

    verticel_marker: float
        True parameter value..

    title: str
        Title of the plot.

    Returns
    -------
    Plot of normal densities.
    """

    sigma1 = math.sqrt(var1)
    sigma2 = math.sqrt(var2)
    x = np.linspace(mu1 - 3 * sigma1, mu1 + 3 * sigma1, 100)
    y1 = stats.norm.pdf(x, mu1, sigma1)
    y2 = stats.norm.pdf(x, mu2, sigma2)
    plt.plot(x, y1, label="density 1")
    plt.plot(x, y2, label="density 2")
    plt.vlines(
        vertical_marker,
        colors="seagreen",
        ymin=0,
        ymax=np.max(np.concatenate((y1, y2))) * 1.1,
        linestyles="dashed",
        label="true mean",
    )
    plt.title(title)
    plt.legend()
    plt.show()
