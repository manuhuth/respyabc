import respy as rp
import pyabc

from functools import partial
from functools import update_wrapper
import tempfile
import os
import matplotlib.pyplot as plt


def respyabc(
    model,
    parameters_prior,
    data,
    distance_abc,
    sampler=pyabc.sampler.MulticoreEvalParallelSampler(),
    population_size_abc=1000,
    max_nr_populations_abc=10,
    minimum_epsilon_abc=0.1,
    database_path_abc=None,
    numb_individuals_respy=1000,
    numb_periods_respy=40,
    simulation_seed_respy=132,
):
    """Compute K&W 1994 model using pyabc. Workhorse of this project. Most
    other functions are used to support this function.

    Parameters
    ----------
    model : function
        A model as defined by ``model``.

    parameters_prior : dictionary
        A dictionary contaning the parameters as keys and the corresponding
        distribution parameters in a tuple as values.

    data : np.array
        Numeric array of shape (``numb_individuals_respy``, ``number of choices``).

    distance_abc : function
        A function that takes two model specifications as inputs and computes
        the difference between the summary statistics of the two model outcomes.

    sampler : pyabc.sampler.function
        A function from the pyabc.sampler class.

    population_size_abc : int
        Positive integer determining the number of particles to be drawn per
        population during the abc algorithm.

    max_nr_populations_abc : int
        Positive integer determining the number of populations that are
        drawn for the abc algorithm.

    minimum_epsilon_abc : float
        Positive float determining the epsilon for the last population run
        of the abc algorithm.

    database_path_abc : str
        Path where the abc runs are stored. If ``None``is supplied, runs
        are saved in the local temp folder.

    numb_individuals_respy : int
        Number of simulated independent individuals in the discrete
        choice model.

    numb_periods_respy : int
        Length of decision horizon in the discrete choice model.

    simulation_seed_respy : int
        Simulation seed for the discrete choice model.

    Returns
    -------
    history : pyabc.object
        An object containing the history of the abc-run.
    """

    params, options, data_stored = rp.get_example_model("kw_94_one")

    options["n_periods"] = numb_periods_respy
    options["simulation_agents"] = numb_individuals_respy
    model_to_simulate = rp.get_simulate_func(params, options)

    uniform = "uniform"
    prior_abc = eval(
        dict_to_pyabc_distribution(
            parameters=parameters_prior, prior_distribution=uniform
        )
    )

    model_abc = wrapped_partial(
        model, model_to_simulate=model_to_simulate, parameter_for_simulation=params
    )

    abc = pyabc.ABCSMC(
        model_abc,
        prior_abc,
        distance_abc,
        population_size=population_size_abc,
        sampler=sampler,
    )

    if database_path_abc is None:
        db_path_abc = "sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db")

    abc.new(db_path_abc, data)
    history = abc.run(
        minimum_epsilon=minimum_epsilon_abc, max_nr_populations=max_nr_populations_abc
    )

    return history


def wrapped_partial(func, *args, **kwargs):
    """Wrapper function to give partial functions the __name__ argument.

    Parameters
    ----------
    func : function
        Function to which partial should be applied.

    Returns
    -------
    partial_func : functools.partial
        Partial function with __name__ argument.

    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)

    return partial_func


def dict_to_pyabc_distribution(parameters, prior_distribution="uniform"):
    """Turn a dictionary including the prior distributions to a string with the
    code of the pyABCdistribution.

    Parameters
    ----------
    parameters : dictionary
        A dictionary contaning the parameters as keys and the corresponding
        distribution parameters in a tuple as values.

    prior_distribution : str
        Type of prior distribution used.

    Returns
    -------
    output_string: str
        string with the code of the pyABCdistribution.
    """
    keys = list(parameters.keys())

    output_string = "pyabc.Distribution("
    for index in keys:
        if index != keys[0]:
            output_string = output_string + (
                f""", {index} = pyabc.RV({prior_distribution },
                {parameters[index][0]}, {parameters[index][1]})"""
            )
        else:
            output_string = (
                output_string
                + f"""{index} = pyabc.RV({prior_distribution },
                {parameters[index][0]}, {parameters[index][1]})"""
            )

    output_string = output_string + ")"

    return output_string


def plot_kernel_density_posterior(history, parameter, xmin, xmax):
    """Plot the Kernel densities of the posterior distribution of an pyABC run.

    Parameters
    ----------
    history : object
        An object created by abc.run().
    parameter : str
        String including the name of the parameter for which
        the posterior should be plotted.
    xmin : float
        Minimum value for x-axis' range.
    xmax : float
        Maximum value for x-axis' range.

    Returns
    -------
    Plot with posterior distribution of parameter.
    """

    fig, ax = plt.subplots()
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(
            df, w, xmin=xmin, xmax=xmax, x=parameter, ax=ax, label="PDF t={}".format(t)
        )
    # ax.axvline(observation, color="k", linestyle="dashed");
    ax.legend()
