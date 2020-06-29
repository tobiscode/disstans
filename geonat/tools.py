"""
This module contains helper functions that are not dependent on any of
GeoNAT's classes.

For more specialized processing functions, see :mod:`~geonat.processing`.
"""

import os
import pandas as pd
from multiprocessing import Pool

# set default number of threads to use
from . import defaults
defaults["general"]["num_threads"] = int(len(os.sched_getaffinity(0)) // 2)


def tvec_to_numpycol(timevector, t_reference=None, time_unit='D'):
    """
    Converts a Pandas timestamp series into a NumPy array of relative
    time to a reference time in the given time unit.

    Parameters
    ----------
    timevector : pandas.Series
        :class:`~pandas.Series` of :class:`~pandas.Timestamp`.
    t_reference : str or pandas.Timestamp, optional
        Reference :class:`~pandas.Timestamp` or datetime-like string that can be converted to one.
        Defaults to the first element of ``timevector``.
    time_unit : str, optional
        Time unit for parameters. Possible values are:

        ``W``, ``D``, ``days``, ``day``, ``hours``, ``hour``, ``hr``, ``h``,
        ``m``, ``minute``, ``min``, ``minutes``, ``T``, ``S``, ``seconds``, ``sec``, ``second``,
        ``ms``, ``milliseconds``, ``millisecond``, ``milli``, ``millis``, ``L``,
        ``us``, ``microseconds``, ``microsecond``, ``micro``, ``micros``, ``U``,
        ``ns``, ``nanoseconds``, ``nano``, ``nanos``, ``nanosecond``, ``N``

        Refer to :func:`~pandas.to_timedelta` for more details.
        Defaults to ``D``.

    Returns
    -------
    numpy.ndarray
        Array of time differences.
    """
    # get reference time
    if t_reference is None:
        t_reference = timevector[0]
    else:
        t_reference = pd.Timestamp(t_reference)
    assert isinstance(t_reference, pd.Timestamp), f"'t_reference' must be a pandas.Timestamp object, got {type(t_reference)}."
    # return Numpy array
    return ((timevector - t_reference) / pd.to_timedelta(1, time_unit)).values


def parallelize(func, iterable, num_threads=None, chunksize=1):
    """
    Convenience wrapper that given a function, an iterable set of inputs
    and parallelization settings automatically either runs the function
    in serial or parallel.

    Parameters
    ----------
    func : function
        Function to wrap, can only have a single input argument.
    iterable : iterable
        Iterable object (list, generator expression, etc.) that contains all the
        arguments that ``func`` should be called with.
    num_threads : int, optional
        Number of threads to use. Set to ``0`` if no parallelization is desired.
        Defaults to the value in :attr:`~geonat.defaults`.
    chunksize : int, optional
        Chunk size used in the parallelization pool, see
        :meth:`~python.multiprocessing.Pool.imap`.

    Yields
    ------
    result
        Whenever a result is calculated, return it.

    Example
    -------
    Consider a simple loop to multiply two numbers:

    >>> from numpy import sum
    >>> iterable = [(1, 2), (2, 3)]
    >>> print([sum(i) for i in iterable])
    [3, 5]

    In parallel with 2 threads, this could look like this:

    >>> from multiprocessing import Pool
    >>> with Pool(2) as p:
    ...     print([result for result in p.imap(sum, iterable)])
    ...
    [3, 5]

    Using :func:`~parallelize`, both cases simplify to:

    >>> from geonat.tools import parallelize
    >>> print([result for result in parallelize(sum, iterable, num_threads=0)])
    [3, 5]
    >>> print([result for result in parallelize(sum, iterable, num_threads=2)])
    [3, 5]
    """
    if num_threads is None:
        num_threads = defaults["general"]["num_threads"]
    if num_threads > 0:
        with Pool(num_threads) as p:
            for result in p.imap(func, iterable, chunksize):
                yield result
    else:
        for parameter in iterable:
            yield func(parameter)
