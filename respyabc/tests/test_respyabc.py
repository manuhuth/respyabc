"""Tests for respyabc function."""
import respy as rp
import pyabc

from respyabc.models import model
from respyabc.distances import distance_mean_squared
from respyabc.respyabc import respyabc


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

    Returns
    -------
    Runs respyabc for the specified parameter.
    """

    params, options, data_stored = rp.get_example_model("kw_94_one")
    options["n_periods"] = 40
    options["simulation_agents"] = 1000
    model_to_simulate = rp.get_simulate_func(params, options)

    data = model(
        parameter_true,
        model_to_simulate=model_to_simulate,
        parameter_for_simulation=params,
        options_for_simulation=options,
        descriptives=descriptives,
    )

    key = list(parameter_true.keys())
    parameters_prior = {key[0]: [prior_low, prior_size]}

    respyabc(
        model=model,
        parameters_prior=parameters_prior,
        data=data,
        distance_abc=distance_mean_squared,
        descriptives=descriptives,
        sampler=pyabc.sampler.MulticoreEvalParallelSampler(),
        population_size_abc=3,
        max_nr_populations_abc=1,
        minimum_epsilon_abc=0.2,
        database_path_abc=None,
    )


def test_delta_choice_frequencies():
    parameter_true = {"delta_delta": 0.95}
    prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.9,
        descriptives="choice_frequencies",
    )


def test_wage_a_constant_choice_frequencies():
    parameter_true = {"wage_a_constant": 9.21}
    prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=9,
        prior_size=0.9,
        descriptives="choice_frequencies",
    )


def test_delta_wage_moments():
    parameter_true = {"delta_delta": 0.95}
    prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.9,
        descriptives="wage_moments",
    )


def test_wage_a_constant_wage_moments():
    parameter_true = {"wage_a_constant": 9.21}
    prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=9,
        prior_size=0.9,
        descriptives="wage_moments",
    )
