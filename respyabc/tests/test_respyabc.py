"""Tests to check if respyabc function runs."""
from respyabc.tools import prepare_test_respyabc


def test_delta_choice_frequencies():
    parameter_true = {"delta_delta": 0.95}
    prepare_test_respyabc(
        parameter_true=parameter_true,
        prior_low=0.9,
        prior_size=0.09,
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
        prior_size=0.09,
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
