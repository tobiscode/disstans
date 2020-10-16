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
from .config import defaults
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


class Click():
    """
    Class that enables a GUI to distinguish between clicks (mouse press and release)
    and dragging event (mouse press, move, then release).

    Parameters
    ----------
    ax : matplotlib.axis.Axis
        Axis on which to look for clicks.
    func : function
        Function to call, with the Matplotlib clicking :class:`~matplotlib.backend_bases.Event`
        as its first argument.
    button : int, optional
        Which mouse button to operate on. Defaults to ``1`` (left).
    """
    def __init__(self, ax, func, button=1):
        self._ax = ax
        self._func = func
        self._button = button
        self._press = False
        self._move = False
        self._c1 = self._ax.figure.canvas.mpl_connect('button_press_event', self._onpress)
        self._c2 = self._ax.figure.canvas.mpl_connect('button_release_event', self._onrelease)
        self._c3 = self._ax.figure.canvas.mpl_connect('motion_notify_event', self._onmove)

    def __del__(self):
        for cid in [self._c1, self._c2, self._c3]:
            self._ax.figure.canvas.mpl_disconnect(cid)

    def _onclick(self, event):
        if event.inaxes == self._ax:
            if event.button == self._button:
                self._func(event)

    def _onpress(self, event):
        self._press = True

    def _onmove(self, event):
        if self._press:
            self._move = True

    def _onrelease(self, event):
        if self._press and not self._move:
            self._onclick(event)
        self._press = False
        self._move = False


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
    size : int, list, tuple
        Number of (equally-spaced) noise samples of the output noise array or
        a shape where the first entry defines the number of noise samples for
        the remaining dimensions.
    exponent : int
        Exponent of the power law noise model.
    seed : int, numpy.random.Generator, optional
        Pass an initial seed to the random number generator, or pass
        a :class:`~numpy.random.Generator` instance.

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
    # parse desired output shape as list
    if isinstance(size, tuple) or isinstance(size, list):
        assert all([isinstance(dim, int) for dim in size]), \
            "If passing a non-integer shape, 'size' must be a list or tuple " + \
            f"of integers, got {size}."
        shape = [*size]
        if len(shape) == 1:
            shape.append(1)
    elif isinstance(size, int):
        shape = [size, 1]
    else:
        raise ValueError(f"{size} is not a valid size or shape.")
    # parse starting seed (if seed is already a generator, will be returned unaltered)
    rng = np.random.default_rng(seed)
    # get the number of noise samples and the remaining dimensions
    size = shape[0]
    halfsize = int(size // 2 + 1)
    ndims = np.prod(shape[1:])
    # step 1-2
    # get Fourier frequencies
    freqs = np.fft.rfftfreq(size)
    # the scaling later can't handle zero frequency, so we need to set it
    # to the minimum frequency possible
    freqs[0] = 1/size
    # scale the frequencies
    freqs_scaled = freqs**(-exponent/2)
    # create an empty array and loop over the dimensions
    out = np.empty([size, ndims])
    for idim in range(ndims):
        # draw two sets of Gaussian distributed random numbers
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
        # put into array
        out[:, idim] = noise
    # reshape output and return
    out = out.reshape(shape)
    return out


def parse_maintenance_table(csvpath, sitecol, datecols, siteformatter=None, delimiter=',',
                            codecol=None, exclude=None, include=None, verbose=False):
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
    exclude : list, optional
        Maintenance records that exactly match an element in ``exclude`` will be ignored.
        ``codecol`` has to be set.
    include : list, optional
        Only maintenance records that include an element of ``include`` will be used.
        No exact match is required.
        ``codecol`` has to be set.
    verbose : bool, optional
        If ``True``, print loading information.

    Returns
    -------
    maint_table : pandas.DataFrame
        Parsed maintenance table.
    maint_dict : dict
        Dictionary of that maps the station names to a list of steptimes.

    Notes
    -----
    If running into problems, also consult the Pandas :func:`~pandas.read_csv`
    function (used to load the ``csvpath`` file) and :class:`~pandas.DataFrame`
    (object on which the filtering happens).
    """
    # load codes and tables
    if codecol is not None:
        assert isinstance(codecol, int), \
            f"'codecol' needs to be an integer, got {codecol}."
        if exclude is not None:
            assert (isinstance(exclude, list) and
                    all([isinstance(ecode, str) for ecode in exclude])), \
                f"'exclude' needs to be a list of strings, got {exclude}."
        if include is not None:
            assert (isinstance(include, list) and
                    all([isinstance(icode, str) for icode in include])), \
                f"'include' needs to be a list of strings, got {include}."
        maint_table = pd.read_csv(csvpath, delimiter=delimiter, usecols=[sitecol, codecol])
        # because we don't know the column names, we need to make sure that the site will
        # always be in the first column for later
        if codecol < sitecol:
            maint_table = maint_table.iloc[:, [1, 0]]
        # save code column name for later
        codecolname = maint_table.columns[1]
    else:
        maint_table = pd.read_csv(csvpath, delimiter=delimiter, usecols=[sitecol])
    # get site column name
    sitecolname = maint_table.columns[0]
    # load and parse time
    time = pd.read_csv(csvpath, delimiter=delimiter, usecols=datecols, squeeze=True,
                       parse_dates=[list(range(len(datecols)))] if len(datecols) > 1 else True)
    timecolname = time.name
    # connect time and data
    maint_table = maint_table.join(time)
    if verbose:
        print(f"Loaded {maint_table.shape[0]} maintenance entries.")
    # process site name column with siteformatter and make sure we're not combining stations
    if siteformatter is not None:
        assert callable(siteformatter), \
            f"'siteformatter' needs to be a callable, got {siteformatter}."
        unique_pre = len(maint_table[sitecolname].unique())
        maint_table[sitecolname] = maint_table[sitecolname].apply(siteformatter)
        unique_post = len(maint_table[sitecolname].unique())
        assert unique_pre == unique_post, "While applying siteformatter, stations were merged."
    # now drop all columns where code is exactly one of the elements in exclude
    if (codecol is not None) and (exclude is not None):
        droprows = maint_table[codecolname].isin(exclude)
        if verbose:
            print(f"Dropping {droprows.sum()} rows because of exclude={exclude}.")
        maint_table = maint_table[~droprows]
    # now drop all columns where the code does not contain an element of 'include'
    if (codecol is not None) and (include is not None):
        keeprows = np.any([maint_table[codecolname].str.contains(pat).values
                           for pat in include], axis=0)
        if verbose:
            print(f"Dropping {maint_table.shape[0] - keeprows.sum()} rows "
                  f"because of include={include}.")
        maint_table = maint_table.iloc[keeprows, :]
    # now produce a dictionary that maps sites to a list of step dates: {site: [steptimes]}
    maint_dict = dict(maint_table.groupby(sitecolname)[timecolname].apply(list))
    # rename columns
    maint_table.rename(columns={sitecolname: "station", codecolname: "code",
                                timecolname: "time"}, inplace=True)
    return maint_table, maint_dict


def weighted_median(values, weights, axis=0, percentile=0.5, keepdims=False):
    """
    Calculates the weighted median along a given axis.

    Parameters
    ----------
    values : numpy.ndarray
        Values to calculate the medians for.
    weights : numpy.ndarray
        Weights of each value along the given ``axis``.
    axis : int, optional
        Axis along which to calculate the median. Defaults to the first one (``0``).
    percentile : float, optional
        Changes the percentile (between 0 and 1) of which median to calculate.
        Defaults to ``0.5``.
    keepdims : bool, optional
        If ``True``, squeezes out the axis along which the median was calculated.
        Defaults to ``False``.
    """
    # some checks
    assert isinstance(values, np.ndarray) and isinstance(weights, np.ndarray), \
        "'values' and 'weights' must be NumPy arrays, got " + \
        f"{type(values)} and {type(weights)}."
    assert isinstance(axis, int) and (axis < len(values.shape)), \
        f"Axis {axis} is not a valid index for the shape of 'values' ({values.shape})."
    assert (percentile >= 0) and (percentile <= 1), \
        f"'percentile' must be between 0 and 1, got {percentile}."
    # broadcast the weights
    weights = np.expand_dims(weights,
                             [i for i in range(len(values.shape))
                              if i != axis])
    # sort the values and weights
    sort_indices = np.argsort(values, axis=axis)
    sort_values = np.take_along_axis(values, sort_indices, axis=axis)
    sort_weights = np.take_along_axis(weights, sort_indices, axis=axis)
    # calculate the median cutoff
    cumsum = np.cumsum(sort_weights, axis=axis)
    cutoff = np.sum(sort_weights, axis=axis, keepdims=True) * percentile
    # find the values at that cutoff
    index_array = np.argmax(cumsum >= cutoff, axis=axis)
    medians = np.take_along_axis(sort_values,
                                 np.expand_dims(index_array, axis=axis), axis=axis)
    # squeeze (if desired) and return
    if not keepdims:
        medians = medians.squeeze(axis=axis)
    return medians
