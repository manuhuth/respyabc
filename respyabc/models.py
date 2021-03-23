import numpy as np


def compute_model(
    parameter,
    model_to_simulate,
    parameter_for_simulation,
    options_for_simulation,
    descriptives="choice_frequencies",
):
    """Compute K&W 1994 model. Function is a wrapper around
       the respy.get_simulate_func() function to compute the model using the
       parameters from Kean & Wolpin 1994 but being able to vary over thr
       parameters.

    Parameters
    ----------
    parameter : dictionary
        A dictionary contaning the variables as key and the corresponding
        magnitude as value respectively.

    model_to_simulate : object produced by respyabc.get_simulate_func_options
        Model that specififes the respy set-up.

    parameter_for_simulation : data.frame
        Parameter that specify the respy model.

    options_for_simulation : data.frame
        Options that specify the respy model.

    descriptives : str
        Either be `choice_frequencies`` or ``wage_moments``. Determines how the
        descriptives with which the distance is computed are computed. The
        default is ``choice_frequencies``.

    Returns
    -------
    output_frequencies : dictionary
        A dictionary containing the relative frequencies of each choice in
        each period.
    """
    keys = list(parameter.keys())
    params_single_index = transform_multiindex_to_single_index(
        df=parameter_for_simulation, column1="category", column2="name", link="_"
    )

    for index in keys:
        params_single_index.loc[index, ("value")] = parameter[index]

    parameter_for_simulation["value"] = np.array(params_single_index["value"])

    options_for_simulation["simulation_seed"] = np.random.randint(0, 1000)
    df_simulated_model = model_to_simulate(
        parameter_for_simulation, options=options_for_simulation
    )

    if descriptives == "choice_frequencies":
        output = compute_choice_frequencies_to_model_output_frequencies(
            df=df_simulated_model
        )
    elif descriptives == "wage_moments":
        output = {"data": np.array(fill_nan(compute_wage_moments(df_simulated_model)))}

    return output


def compute_choice_frequencies_to_model_output_frequencies(df):
    """Processes the choice frequencies to the output frequencies.

    Parameters
    ----------
    df : data.frame
        Data frame for which the choice frequencies should be created.

    Returns
    -------
    output_frequencies : dictionary
        A dictionary containing the relative frequencies of each choice in
        each period.
    """

    df_frequencies = compute_choice_frequencies(df)

    for index in ["a", "b", "edu", "home"]:
        if index not in df_frequencies.columns:
            df_frequencies[index] = 0

    df_frequencies.sort_index(axis=1, inplace=True)

    output_frequencies = {"data": np.array(fill_nan(df_frequencies))}

    return output_frequencies


def compute_choice_frequencies(df):
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


def compute_wage_moments(df):
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


def transform_multiindex_to_single_index(df, column1, column2, link="_"):
    """Replaces a multiindex with a concatenated single index version
    of the multiindex.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas data frame with a multiindex.

    column1 : str
        Name of first multiindex column.

    column2 : str
        Name of second multiindex column.

    link : str
        String that is used to seperate the two multiindex columns within the
        new string.

    Returns
    -------
    Single index data frame.
    """
    index1 = df.index.get_level_values(column1)
    index2 = df.index.get_level_values(column2)
    single_index = index1 + link + index2

    df2 = df.reset_index(drop=True)
    df2.set_index(single_index, inplace=True)

    return df2
