"""Tests for models function."""
import respy as rp

from respyabc.models import compute_model


def prepare_test_model(parameter_true):
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

    compute_model(
        parameter_true,
        model_to_simulate=model_to_simulate,
        parameter_for_simulation=params,
        options_for_simulation=options,
    )


def test_delta():
    parameter_true = {"delta_delta": 0.95}
    prepare_test_model(parameter_true=parameter_true)


def test_wage_a_constant():
    parameter_true = {"wage_a_constant": 9.21}
    prepare_test_model(parameter_true=parameter_true)
