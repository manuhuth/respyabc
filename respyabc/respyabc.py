import respy as rp
import pyabc

import tempfile
import os


from respy.pre_processing.model_processing import process_params_and_options
from respy.simulate import simulate
from respy.simulate import _harmonize_simulation_arguments
from respy.simulate import _process_input_df_for_simulation
from respy.solve import get_solve_func
from respy.shared import create_base_draws
from functools import partial
from functools import update_wrapper


def respyabc(
    model,
    parameters_prior,
    data,
    distance_abc,
    descriptives="choice_frequencies",
    sampler=pyabc.sampler.MulticoreEvalParallelSampler(),
    population_size_abc=1000,
    max_nr_populations_abc=10,
    minimum_epsilon_abc=0.1,
    database_path_abc=None,
    numb_individuals_respy=1000,
    numb_periods_respy=40,
    model_selection=False,
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

    data : numpy.array
        Numeric array of shape (``numb_individuals_respy``, ``number of choices``).

    distance_abc : function
        A function that takes two model specifications as inputs and computes
        the difference between the summary statistics of the two model outcomes.

    descriptives : {"choice_frequencies", "wage_moments"}
        Determines how the descriptives with which the distance is computed
        are computed. The default is ``choice_frequencies``.

    sampler : pyabc.sampler.function, optional
        A function from the pyabc.sampler class.

    population_size_abc : int, optional
        Positive integer determining the number of particles to be drawn per
        population during the abc algorithm.

    max_nr_populations_abc : int, optional
        Positive integer determining the number of populations that are
        drawn for the abc algorithm.

    minimum_epsilon_abc : float, optional
        Positive float determining the epsilon for the last population run
        of the abc algorithm.

    database_path_abc : str, optional
        Path where the abc runs are stored. If ``None``is supplied, runs
        are saved in the local temp folder.

    numb_individuals_respy : int, optional
        Number of simulated independent individuals in the discrete
        choice model.

    numb_periods_respy : int, optional
        Length of the decision horizon in the discrete choice model.

    model_selection : bool, optional
        If `True`, the function expects a model selection procedure, if `False`
        single inference is conducted.

    Returns
    -------
    history : pyabc.object
        An object containing the history of the abc-run.
    """

    params, options, data_stored = rp.get_example_model("kw_94_one")

    options["n_periods"] = numb_periods_respy
    options["simulation_agents"] = numb_individuals_respy

    model_to_simulate = get_simulate_func_options(params, options)

    if model_selection is False:
        abc = get_abc_object_inference(
            distance_abc=distance_abc,
            population_size_abc=population_size_abc,
            sampler=sampler,
            parameters_prior=parameters_prior,
            model=model,
            model_to_simulate=model_to_simulate,
            params=params,
            options=options,
            descriptives=descriptives,
        )

    elif model_selection is True:
        abc = get_abc_object_model_selection(
            distance_abc=distance_abc,
            population_size_abc=population_size_abc,
            sampler=sampler,
            parameters_prior=parameters_prior,
            model=model,
            model_to_simulate=model_to_simulate,
            params=params,
            options=options,
            descriptives=descriptives,
        )

    if database_path_abc is None:
        db_path_abc = "sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db")

    abc.new(db_path_abc, data)
    history = abc.run(
        minimum_epsilon=minimum_epsilon_abc, max_nr_populations=max_nr_populations_abc
    )

    return history


def get_abc_object_inference(
    distance_abc,
    population_size_abc,
    sampler,
    parameters_prior,
    model,
    model_to_simulate,
    params,
    options,
    descriptives,
    norm="norm",
    uniform="uniform",
):
    """Returns abc object for inference.

    Parameters
    ----------
    distance_abc : function
        A function that takes two model specifications as inputs and computes
        the difference between the summary statistics of the two model outcomes.

    population_size_abc : int
        Positive integer determining the number of particles to be drawn per
        population during the abc algorithm.

    sampler : pyabc.sampler.function
        A function from the pyabc.sampler class.

    parameters_prior : dictionary
        A dictionary contaning the parameters as keys and the corresponding
        distribution parameters in a tuple as values.

    model : function
        A model as defined by ``model``.

    model_to_simulate : object produced by respyabc.get_simulate_func_options
        Model that specififes the respy set-up.

    params : pandas.DataFrame
        Parameter that specify the respy model.

    options : pandas.DataFrame
        Options that specify the respy model.

    descriptives : {"choice_frequencies", "wage_moments"}
        Determines how the descriptives with which the distance is computed
        are computed.

    norm : str, optional
        Name of the normal distribution. Currently must be set
        for the eval comment.

    uniform : str,
        Name of the uniform distribution. Currently must be set
        for the eval comment.


    Returns
    -------
    history : pyabc.object
        An object containing the history of the abc-run.
    """

    prior_abc = eval(convert_dict_to_pyabc_distribution(parameters=parameters_prior))

    model_abc = wrap_partial(
        model,
        model_to_simulate=model_to_simulate,
        parameter_for_simulation=params,
        options_for_simulation=options,
        descriptives=descriptives,
    )

    abc = pyabc.ABCSMC(
        model_abc,
        prior_abc,
        distance_abc,
        population_size=population_size_abc,
        sampler=sampler,
    )

    return abc


def get_abc_object_model_selection(
    distance_abc,
    population_size_abc,
    sampler,
    parameters_prior,
    model,
    model_to_simulate,
    params,
    options,
    descriptives,
    norm="norm",
    uniform="uniform",
):
    """Returns abc object for model selection.

    Parameters
    ----------
    distance_abc : function
        A function that takes two model specifications as inputs and computes
        the difference between the summary statistics of the two model outcomes.

    population_size_abc : int
        Positive integer determining the number of particles to be drawn per
        population during the abc algorithm.

    sampler : pyabc.sampler.function
        A function from the pyabc.sampler class.

    parameters_prior : dictionary
        A dictionary contaning the parameters as keys and the corresponding
        distribution parameters in a tuple as values.

    model : function
        A model as defined by ``model``.

    model_to_simulate : object produced by respyabc.get_simulate_func_options
        Model that specififes the respy set-up.

    params : pandas.DataFrame
        Parameter that specify the respy model.

    options : pandas.DataFrame
        Options that specify the respy model.

    descriptives : {"choice_frequencies", "wage_moments"}
        Determines how the descriptives with which the distance is computed
        are computed.

    norm : str, optional
        Name of the normal distribution. Currently must be set
        for the eval comment.

    uniform : str, optional
        Name of the uniform distribution. Currently must be set
        for the eval comment.


    Returns
    -------
    history : pyabc.object
        An object containing the history of the abc-run.
    """

    model_abc = []
    prior_abc = []
    for index in range(len(model)):
        prior_abc.append(
            eval(convert_dict_to_pyabc_distribution(parameters=parameters_prior[index]))
        )

        model_abc.append(
            wrap_partial(
                model[index],
                model_to_simulate=model_to_simulate,
                parameter_for_simulation=params,
                options_for_simulation=options,
                descriptives=descriptives,
            )
        )

    abc = pyabc.ABCSMC(
        model_abc,
        prior_abc,
        distance_abc,
        population_size=population_size_abc,
        sampler=sampler,
    )

    return abc


def get_simulate_func_options(
    params,
    options,
    method="n_step_ahead_with_sampling",
    df=None,
    n_simulation_periods=None,
):
    """Rewrite respy's get_simulation_function such that options can be passed
    and therefore the seed be changed before any run. Documentation is adapted
    from `respy.simulate.get_simulate_func`

    Parameters
    ----------
    params : pandas.DataFrame
        DataFrame containing the model parameters.

    options : dict
        Dictionary containing the model options.

    method : {"n_step_ahead_with_sampling", "n_step_ahead_with_data", "one_step_ahead"}
        The simulation method which can be one of three and is explained in more detail
        in :func:`respy.simulate.simulate()`.

    df : pandas.DataFrame or None, default None
        DataFrame containing one or multiple observations per individual.

    n_simulation_periods : int or None, default None
        Simulate data for a number of periods. This options does not affect
        ``options["n_periods"]`` which controls the number of periods for which decision
        rules are computed.

    Returns
    -------
    simulate_function : :func:`simulate`
        Simulation function where all arguments except the parameter vector
        and the options are set.
    """
    optim_paras, options = process_params_and_options(params, options)

    n_simulation_periods, options = _harmonize_simulation_arguments(
        method, df, n_simulation_periods, options
    )

    df = _process_input_df_for_simulation(df, method, options, optim_paras)

    solve = get_solve_func(params, options)

    n_observations = (
        df.shape[0]
        if method == "one_step_ahead"
        else df.shape[0] * n_simulation_periods
    )
    shape = (n_observations, len(optim_paras["choices"]))

    base_draws_sim = create_base_draws(
        shape, next(options["simulation_seed_startup"]), "random"
    )
    base_draws_wage = create_base_draws(
        shape, next(options["simulation_seed_startup"]), "random"
    )

    simulate_function = partial(
        simulate,
        base_draws_sim=base_draws_sim,
        base_draws_wage=base_draws_wage,
        df=df,
        method=method,
        n_simulation_periods=n_simulation_periods,
        solve=solve,
    )

    return simulate_function


def wrap_partial(func, *args, **kwargs):
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


def convert_dict_to_pyabc_distribution(parameters):
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
        prior_distribution = parameters[index][1]
        if index != keys[0]:
            output_string = output_string + (
                f""", {index} = pyabc.RV({prior_distribution},
                {parameters[index][0][0]}, {parameters[index][0][1]})"""
            )
        else:
            output_string = (
                output_string
                + f"""{index} = pyabc.RV({prior_distribution},
                {parameters[index][0][0]}, {parameters[index][0][1]})"""
            )

    output_string = output_string + ")"

    return output_string
