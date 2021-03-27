"""Tests to check if evaluation routines run."""
from respyabc.tools import prepare_test_respyabc
from respyabc.evaluation import compute_point_estimate
from respyabc.evaluation import compute_central_credible_interval
from respyabc.evaluation import plot_kernel_density_posterior
from respyabc.evaluation import plot_credible_intervals
from respyabc.evaluation import plot_history_summary


def test_point_estimate():
    parameter_true = {"delta_delta": 0.95}
    history = prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.9,
        descriptives="choice_frequencies",
    )

    compute_point_estimate(history=history)


def test_central_credible_interval():
    parameter_true = {"delta_delta": 0.95}
    history = prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.09,
        descriptives="choice_frequencies",
    )

    compute_central_credible_interval(
        history=history, parameter="delta_delta", alpha=0.05
    )


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


def test_plot_credible_intervals_mean():
    parameter_true = {"delta_delta": 0.95}
    history = prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.09,
        descriptives="choice_frequencies",
    )
    plot_credible_intervals(
        history=history,
        parameter="delta_delta",
        interval_type="mean",
    )


def test_plot_credible_intervals_simulated():
    parameter_true = {"delta_delta": 0.95}
    history = prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.09,
        descriptives="choice_frequencies",
    )
    plot_credible_intervals(
        history=history,
        parameter="delta_delta",
        interval_type="simulated",
    )


def test_plot_history_summary():
    parameter_true = {"delta_delta": 0.95}
    history = prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.09,
        descriptives="choice_frequencies",
    )
    plot_history_summary(history, parameter_name="delta_delta", parameter_value=0.95)
