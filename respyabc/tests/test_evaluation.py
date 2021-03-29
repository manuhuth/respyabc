"""Tests to check if evaluation routines run."""

from respyabc.tools import prepare_test_respyabc
from respyabc.tools import prepare_test_respyabc_two_params
from respyabc.evaluation import compute_point_estimate
from respyabc.evaluation import compute_central_credible_interval
from respyabc.evaluation import plot_kernel_density_posterior
from respyabc.evaluation import plot_credible_intervals
from respyabc.evaluation import plot_multiple_credible_intervals
from respyabc.evaluation import plot_history_summary
from respyabc.evaluation import plot_2d_histogram


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


def test_plot_multiple_credible_intervals():
    parameter_true = {"wage_a_constant": 9.21, "wage_b_constant": 8.48}
    history = prepare_test_respyabc_two_params(
        parameter_true=parameter_true,
        prior_low=[9.15, 8.4],
        prior_size=[0.1, 0.1],
        descriptives="choice_frequencies",
    )
    plot_multiple_credible_intervals(
        history=history,
        parameter_names=["wage_a_constant", "wage_b_constant"],
        number_rows=1,
        number_columns=2,
        confidence_levels=[0.95, 0.9, 0.5],
        size=(12, 8),
    )


def test_plot_2d_histogram():
    parameter_true = {"wage_a_constant": 9.21, "wage_b_constant": 8.48}
    history = prepare_test_respyabc_two_params(
        parameter_true=parameter_true,
        prior_low=[9.1, 8.2],
        prior_size=[0.3, 0.3],
        descriptives="choice_frequencies",
    )
    plot_2d_histogram(
        history=history,
        parameter_names=["wage_a_constant", "wage_b_constant"],
        parameter_true=[9.21, 8.48],
        xmin=9.1,
        xmax=9.4,
        ymin=8.2,
        ymax=8.5,
    )
