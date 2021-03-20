def time_convertion(seconds):
    """Takes seconds as input and turns it into minutes or hours, if necessary.

    Parameters
    ----------
    seconds : float, time-difference
        Time in seconds

    Returns
    -------
    time : float
        Magnitude of time.

    unit : str
        Time unit. Either be seconds, minutes or hours.
    """
    unit = "seconds"
    time = seconds
    if time >= 60:
        time = time / 60
        unit = "minutes"
    elif time >= 60:
        time = time / 60
        unit = "hours"

    return time, unit
