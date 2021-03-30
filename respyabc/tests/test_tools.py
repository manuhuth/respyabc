"""Tests to check if tools run."""

from respyabc.tools import plot_normal_densities
from respyabc.tools import convert_time
from respyabc.tools import prepare_test_respyabc
from respyabc.tools import prepare_test_respyabc_model_selection


def test_prepare_test_respyabc_choice_frequencies():
    parameter_true = {"delta_delta": 0.95}
    prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.09,
        descriptives="choice_frequencies",
    )


def test_prepare_test_respyabc_wage_moments():
    parameter_true = {"delta_delta": 0.95}
    prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.09,
        descriptives="wage_moments",
    )


def test_prepare_test_respyabc_wage_moments_model_selection():
    parameter_true = {"delta_delta": 0.95}
    prepare_test_respyabc_model_selection(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.09,
        descriptives="wage_moments",
    )


def test_time_convertion1():
    assert convert_time(60)[0] == 1
    assert convert_time(60)[1] == "minutes"


def test_time_convertion2():
    assert convert_time(50)[0] == 50
    assert convert_time(50)[1] == "seconds"


def test_time_convertion3():
    assert convert_time(3600)[0] == 1
    assert convert_time(3600)[1] == "hours"


def plot_normal_densitites():
    plot_normal_densities(mu1=0, var1=1, mu2=1, var2=1, vertical_marker=0.5)
