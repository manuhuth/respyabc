import numpy as np


def model(parameter, model_to_simulate, parameter_for_simulation):
    """Compute K&W 1994 model. Function is a wrapper around
       the respy.get_simulate_func() function to compute the model using the
       parameters from Kean & Wolpin 1994 but being able to vary over thr
       parameters.

    Parameters
    ----------
    parameter : dictionary
        A dictionary contaning the variables as key and the corresponding
        magnitude as value respectively.

    Returns
    -------
    output_frequencies : dictionary
        A dictionary containing the relative frequencies of each choice in
        each period.
    """
    keys = list(parameter.keys())

    for index in keys:
        parameter_for_simulation.loc[index, ("value")] = parameter[index]

    df_simulated_model = model_to_simulate(parameter_for_simulation)
    df_frequencies = choice_frequencies(df_simulated_model)

    for index in ["a", "b", "edu", "home"]:
        if index not in df_frequencies.columns:
            df_frequencies[index] = 0

    df_frequencies.sort_index(axis=1, inplace=True)

    output_frequencies = {"data": np.array(fill_nan(df_frequencies))}

    return output_frequencies


def choice_frequencies(df):
    """Calculate choice frequencies per Period in the discrete choice model.

    Parameters
    ----------
    df : pd.dataframe
        A pandas data frame containing the output of the discrete choice model.

    Returns
    -------
    A pandas data frame containing the relative choice frequencies
    of each period.
    """

    return df.groupby("Period").Choice.value_counts(normalize=True).unstack()


def fill_nan(df):
    """Fill missing values in data frame with zeros.

    Parameters
    ----------
    df : pd.dataframe
        A pandas data frame containing missing values.

    Returns
    -------
    A pandas data frame containing zeros instead of the missing values.
    """

    return df.fillna(0)


def wage_moments(df):
    """Calculate first and second wage moment  in the discrete choice model.

    Parameters
    ----------
    df : pd.dataframe
        A pandas data frame containing the output of the discrete choice model.

    Returns
    -------
    A pandas data frame containing the first and second wage moments
    of each period."""
    return df.groupby(["Period"])["Wage"].describe()[["mean", "std"]]
