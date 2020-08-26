"""
This module contains helper functions that are not dependent on any of
GeoNAT's classes.

For more specialized processing functions, see :mod:`~geonat.processing`.
"""

import os
import numpy as np
import pandas as pd
from multiprocessing import Pool

# set default number of threads to use
from . import defaults
defaults["general"]["num_threads"] = int(len(os.sched_getaffinity(0)) // 2)


class Timedelta(pd.Timedelta):
    def __new__(cls, *args, **kwargs):
        """
        GeoNAT Timedelta subclassed from :class:`~pandas.Timedelta` but with support
        for the ``'Y'`` year time unit, defined as always exactly 365.25 days.
        Other possible values are:

        ``W``, ``D``, ``days``, ``day``, ``hours``, ``hour``, ``hr``, ``h``,
        ``m``, ``minute``, ``min``, ``minutes``, ``T``,
        ``S``, ``seconds``, ``sec``, ``second``,
        ``ms``, ``milliseconds``, ``millisecond``, ``milli``, ``millis``, ``L``,
        ``us``, ``microseconds``, ``microsecond``, ``micro``, ``micros``, ``U``,
        ``ns``, ``nanoseconds``, ``nano``, ``nanos``, ``nanosecond``, ``N``
        """
        if (len(args) == 2) and (args[1].upper() == "Y"):
            args = (args[0] * 365.25, "D")
        return super().__new__(cls, *args, **kwargs)


def tvec_to_numpycol(timevector, t_reference=None, time_unit='D'):
    """
    Converts a Pandas timestamp series into a NumPy array of relative
    time to a reference time in the given time unit.

    Parameters
    ----------
    timevector : pandas.Series, pandas.DatetimeIndex
        :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
        :class:`~pandas.DatetimeIndex` of when to evaluate the model.
    t_reference : str or pandas.Timestamp, optional
        Reference :class:`~pandas.Timestamp` or datetime-like string
        that can be converted to one.
        Defaults to the first element of ``timevector``.
    time_unit : str, optional
        Time unit for parameters.
        Refer to :class:`~geonat.tools.Timedelta` for more details.
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
    assert isinstance(t_reference, pd.Timestamp), \
        f"'t_reference' must be a pandas.Timestamp object, got {type(t_reference)}."
    # return Numpy array
    return ((timevector - t_reference) / Timedelta(1, time_unit)).values


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
        Defaults to the value in :attr:`~geonat.config.defaults`.
    chunksize : int, optional
        Chunk size used in the parallelization pool, see
        :meth:`~multiprocessing.pool.Pool.imap`.

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


def create_powerlaw_noise(size, exponent, seed=None):
    """
    Creates synthetic noise according to a Power Law model [langbein04]_.

    Parameters
    ----------
    size : int
        Number of (equally-spaced) noise samples of the output noise array.
    exponent : int
        Exponent of the power law noise model.
    seed : int, optional
        Pass an initial seed to the random number generator.

    Returns
    -------
    numpy.ndarray
        Noise output array.

    Notes
    -----
    This function uses Timmer and König's [timmerkoenig95]_ approach to
    generate the noise, and Felix Patzelt's `colorednoise`_ code to calculate
    the theoretical standard deviation.

    References
    ----------

    .. [langbein04] Langbein, J. (2004),
       *Noise in two‐color electronic distance meter measurements revisited*,
       J. Geophys. Res., 109, B04406,
       doi:`10.1029/2003JB002819 <https://doi.org/10.1029/2003JB002819>`_.
    .. [timmerkoenig95] Timmer, J.; König, M. (1995),
       *On generating power law noise*,
       Astronomy and Astrophysics, v.300, p.707.
    .. _`colorednoise`: https://github.com/felixpatzelt/colorednoise
    """
    halfsize = int(size // 2 + 1)
    # step 1-2
    # get Fourier frequencies
    freqs = np.fft.rfftfreq(size)
    # the scaling later can't handle zero frequency, so we need to set it
    # to the minimum frequency possible
    freqs[0] = 1/size
    # scale the frequencies
    freqs_scaled = freqs**(-exponent/2)
    # draw two sets of Gaussian distributed random numbers
    rng = np.random.default_rng(seed)
    real_part = rng.standard_normal(halfsize) * freqs_scaled
    imag_part = rng.standard_normal(halfsize) * freqs_scaled
    # for real signals, there is no imaginary component at the zero frequency
    imag_part[0] = 0
    # for even length signals, the last component (Nyquist frequency)
    # also has to be real because of symmetry properties
    if (size % 2) == 0:
        imag_part[-1] = 0
    # combine the two parts
    fourier_noise = real_part + imag_part * 1j
    # step 3
    # transform from frequency to time domain
    noise = np.fft.irfft(fourier_noise, n=size)
    # additional step: normalize to unit standard deviation
    # estimate the standard deviation
    freqs_sigma_est = freqs_scaled[1:].copy()
    freqs_sigma_est[-1] *= (1 + (size % 2)) / 2
    sigma = 2 * np.sqrt(np.sum(freqs_sigma_est**2)) / size
    # normalize
    noise /= sigma
    return noise


def parse_maintenance_table(csvpath, sitecol, datecols, siteformatter=None, delimiter=',',
                            codecol=None, exclude_codes=None):
    """
    Function that loads a maintenance table from a .csv file (or similar) and returns
    a list of step times for each station. It also provides an interface to ignore
    certain maintenance codes (if present), and modify the site names when loading.

    Parameters
    ----------
    csvpath : str
        Path of the file to load.
    sitecol : int
        Column index of the station names.
    datecols : list
        List of indices that contain the ingredients to convert the input to a valid
        :class:`~pandas.Timestamp`. It should fail gracefully, i.e. return a string
        if Pandas cannot interpret the column(s) appropriately.
    siteformatter : function, optional
        Function that will be called element-wise on the loaded station names to
        produce the output station names.
    delimiter : str, optional
        Delimiter character for the input file.
    codecol : int, optional
        Column index of the maintenance code.
    exclude_codes : list, optional
        List of strings that will lead to a maintenance record being ignored if there
        is an exact match.
        ``codecol`` and ``exclude_cols`` always have to be set both to work.

    Returns
    -------
    dict
        Dictionary of that maps the station names to a list of steptimes.

    Notes
    -----
    If running into problems, also consult the Pandas :func:`~pandas.read_csv`
    function (used to load the ``csvpath`` file) and :class:`~pandas.DataFrame`
    (object on which the filtering happens).
    """
    # load codes and tables
    if (codecol is not None) and (exclude_codes is not None):
        assert isinstance(codecol, int), \
            f"'codecol' needs to be an integer, got {codecol}."
        assert (isinstance(exclude_codes, list) and
                all([isinstance(ecode, str) for ecode in exclude_codes])), \
            f"'exclude_codes' needs to be a list of strings, got {exclude_codes}."
        table = pd.read_csv(csvpath, delimiter=delimiter, usecols=[sitecol, codecol])
        # because we don't know the column names, we need to make sure that the site will
        # always be in the first column for later
        if codecol < sitecol:
            table = table.iloc[:, [1, 0]]
        # save code column name for later
        codecolname = table.columns[1]
    else:
        table = pd.read_csv(csvpath, delimiter=delimiter, usecols=[sitecol])
    # get site column name
    sitecolname = table.columns[0]
    # load and parse time
    time = pd.read_csv(csvpath, delimiter=delimiter, usecols=datecols, squeeze=True,
                       parse_dates=[list(range(len(datecols)))] if len(datecols) > 1 else True)
    timecolname = time.name
    # connect time and data
    table = table.join(time)
    # process site name column with siteformatter and make sure we're not combining stations
    if siteformatter is not None:
        assert callable(siteformatter), \
            f"'siteformatter' needs to be a callable, got {siteformatter}."
        unique_pre = len(table[sitecolname].unique())
        table[sitecolname] = table[sitecolname].apply(siteformatter)
        unique_post = len(table[sitecolname].unique())
        assert unique_pre == unique_post, "While applying siteformatter, stations were merged."
    # now drop all columns where code is exactly one of the elements in exclude_codes
    if (codecol is not None) and (exclude_codes is not None):
        droprows = table[codecolname].isin(exclude_codes)
        print(f"Dropping {droprows.sum()} rows because of exclude_codes={exclude_codes}.")
        table = table[~droprows]
    # now produce a dictionary that maps sites to a list of step dates: {site: [steptimes]}
    return dict(table.groupby(sitecolname)[timecolname].apply(list))
