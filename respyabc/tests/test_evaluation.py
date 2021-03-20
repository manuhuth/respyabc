"""Tests to check if evaluation routines run."""
from respyabc.tools import prepare_test_respyabc
from respyabc.evaluation import point_estimate
from respyabc.evaluation import central_credible_interval
from respyabc.evaluation import plot_kernel_density_posterior


def test_point_estimate():
    parameter_true = {"delta_delta": 0.95}
    history = prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.9,
        descriptives="choice_frequencies",
    )

    point_estimate(history=history)


def test_central_credible_interval():
    parameter_true = {"delta_delta": 0.95}
    history = prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.09,
        descriptives="choice_frequencies",
    )

    central_credible_interval(history=history, parameter="delta_delta", alpha=0.05)


def test_plot_kernel_density_posterior():
    parameter_true = {"delta_delta": 0.95}
    history = prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.09,
        descriptives="choice_frequencies",
    )
    plot_kernel_density_posterior(
        history=history, parameter="delta_delta", xmin=0.9, xmax=0.99
    )
