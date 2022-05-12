"""
This module contains helper functions and classes that are not dependent on
any of DISSTANS's classes.

For more specialized processing functions, see :mod:`~disstans.processing`.
"""

import os
import re
import subprocess
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.geodesic as cgeod
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from multiprocessing import Pool
from tqdm import tqdm
from urllib import request, error
from pathlib import Path
from datetime import datetime, timezone
from warnings import warn
from matplotlib.ticker import FuncFormatter
from cmcrameri import cm as scm

from .config import defaults


class Timedelta(pd.Timedelta):
    def __new__(cls, *args, **kwargs):
        """
        DISSTANS Timedelta subclassed from :class:`~pandas.Timedelta` but with support
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
        Refer to :class:`~disstans.tools.Timedelta` for more details.
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


def date2decyear(dates):
    """
    Convert dates (just year, month, day, each day assumed to be centered at noon)
    to decimal years, assuming all years have 365.25 years (JPL convention for
    GIPSY timeseries, also used by UNR NGL).

    Parameters
    ----------
    dates : pandas.Series, pandas.DatetimeIndex, pandas.Timestamp, datetime.datetime
        Input date(s). If a Series, needs to be a series of Timestamp-convertible
        data types.

    Returns
    -------
    numpy.ndarray
        Date(s) as sorted decimal year(s).
    """
    if isinstance(dates, pd.Series):
        tdelta = dates.dt.normalize() - pd.Timestamp(2000, 1, 1)
    elif isinstance(dates, pd.DatetimeIndex) or isinstance(dates, pd.Timestamp):
        tdelta = dates.normalize() - pd.Timestamp(2000, 1, 1)
    elif isinstance(dates, datetime):
        tdelta = datetime(dates.year, dates.month, dates.day) - datetime(2000, 1, 1)
    return np.sort(np.array(2000 + tdelta.total_seconds() / 86400 / 365.25))


def get_cov_dims(num_components):
    r"""
    Given a number of components, return the number of covariances that
    exist between the components.

    Parameters
    ----------
    num_components : int
        Number of components of timeseries or model.

    Returns
    -------
    int
        Number of covariances, calculated as
        :math:`\text{num_components}*(\text{num_components}-1))/2`.

    See Also
    --------
    make_cov_index_map : For an example.
    """
    num_components = int(num_components)
    return int((num_components * (num_components - 1)) / 2)


def make_cov_index_map(num_components):
    r"""
    Given a number of components, create a matrix that shows the indexing
    of where covariance columns present in a timeseries' or model's 2D dataframe
    will show in the covariance matrix of a single observation or parameter.
    Also provides the ordering in a 1D array which can be used together with
    :func:`~numpy.reshape` to create the varaiance-covariance matrix from the columns.

    Parameters
    ----------
    num_components : int
        Number of components of timeseries or model.

    Returns
    -------
    index_map : numpy.ndarray
        Array of shape :math:`(\text{num_components}, \text{num_components})`
        that is NaN everywhere except in the upper triangle, where integer numbers
        denote where the column of a timseries' or model's 2D dataframe belong.
    var_cov_map : numpy.ndarray
        Array of shape :math:`(\text{num_components}^2, )` that can be used to
        assemble the variance-covariance matrix from the columns given a particular
        timestep or parameter.

    Example
    -------

    >>> import numpy as np
    >>> from disstans.tools import get_cov_dims, make_cov_index_map
    >>> num_observations, num_components = 5, 2
    >>> print(f"For {num_components} components, there should be:\n"
    ...       f"- {num_components} data columns,\n"
    ...       f"- {num_components} variance columns,\n"
    ...       f"- and {get_cov_dims(num_components)} covariance columns.")
    For 2 components, there should be:
    - 2 data columns,
    - 2 variance columns,
    - and 1 covariance columns.
    >>> index_map, var_cov_map = make_cov_index_map(num_components)
    >>> test_varcov = np.stack([np.ones(5), np.arange(5)*2, np.ones(5)*0.5], axis=1)
    >>> test_varcov
    array([[1. , 0. , 0.5],
           [1. , 2. , 0.5],
           [1. , 4. , 0.5],
           [1. , 6. , 0.5],
           [1. , 8. , 0.5]])

    The first two columns are the variances, and the third column is the covariance
    column (since there is only one possible covariance).
    ``index_map`` will show where the covariance columns fit into, indexed from ``0``
    to ``get_cov_dims(num_components) - 1``. Since there is only one, the column
    index ``0`` will feature in the upper right corner:

    >>> index_map
    array([[nan,  0.],
           [nan, nan]])

    If we want the full, symmetric variance-covariance matrix for the third
    observation, we use ``var_cov_map``:

    >>> var_cov_map
    array([0, 2, 2, 1])
    >>> test_varcov[2, var_cov_map].reshape(num_components, num_components)
    array([[1. , 0.5],
           [0.5, 4. ]])

    """
    index_map = np.empty((num_components, num_components))
    index_map[:] = np.NaN
    seq_ix = 0
    for irow in range(num_components):
        for icol in range(irow + 1, num_components):
            index_map[irow, icol] = seq_ix
            seq_ix += 1
    var_cov_map = (np.triu(index_map + num_components, 1) +
                   np.triu(index_map + num_components, 1).T)
    var_cov_map = (var_cov_map + np.diag(np.arange(num_components))).astype(int).ravel()
    return index_map, var_cov_map


def get_cov_indices(icomp, index_map=None, num_components=None):
    """
    Given a data or variance component index, retrieve the indices in the covariance columns
    of a timeseries or model that are associated with that component.
    Exactly one of ``index_map`` or ``num_components`` must be provided as input.

    Parameters
    ----------
    icomp : int
        Index of the component.
    index_map : numpy.ndarray
        Output of :func:`~make_cov_index_map`.
    num_components : int
        Number of components of timeseries or model. (Function will call
        :func:`~make_cov_index_map` to get ``index_map``.)

    Returns
    -------
    list
        List of integer covariance column indices associated with ``icomp``.

    Example
    -------
    In a 3D dataset, the second component is associated with two covariances - between
    the first and the second, and the second and the third. In a timeseries or model
    covariance dataframe, this corresponds to the following columns:

    >>> from disstans.tools import get_cov_indices
    >>> get_cov_indices(1, num_components=3)
    [0, 2]
    """
    if (index_map is None) and (num_components is None):
        raise ValueError("Need to specify either 'index_map' or 'num_components'.")
    if index_map is None:
        index_map = make_cov_index_map(num_components)[0]
    assert icomp < int(np.unique(index_map.shape)), "Invalid 'index_map' shape " \
        f"{index_map.shape} for the index {icomp}."
    from_row = index_map[icomp, :]
    from_col = index_map[:, icomp]
    indices = [int(i) for i in from_row if np.isfinite(i)] + \
              [int(i) for i in from_col if np.isfinite(i)]
    return sorted(indices)


def full_cov_mat_to_columns(cov_mat, num_components, include_covariance=False,
                            return_single=False):
    r"""
    Converts a full variance(-covariance) matrix with multiple components into a
    column-based representation like the one used by :class:`~disstans.models.Model` or
    :class:`~disstans.timeseries.Timeseries`. The extraction done basically implies
    the assumption that the cross-parameter/cross-observation covariance is negligible.

    It is assumed the the individual elements
    are ordered such that all components of one parameter or observation are in
    neighboring rows/columns (i.e. the first parameter or observation occupies the
    first ``num_components`` rows/columns, the second one the second ``num_components``
    rows/columns, etc.).

    Parameters
    ----------
    cov_mat : numpy.ndarray
        Square array with dimensions :math:`\text{num_elements} * \text{num_components}`
        where :math:`\text{num_elements}` is the number of elements (e.g. observations
        or parameters) in each of the :math:`\text{num_components}` dimensions.
    num_components : int
        Number of components `cov_mat` contains.
    include_covariance : bool, optional
        If ``True``, also extract the off-diagonal covariances of each element between
        its components. Defaults to ``False``, i.e. only the diagonal covariances.
    return_single : bool, optional
        If ``False`` (default), return two arrays; if ``True``, concatenate the two.

    Returns
    -------
    variance : numpy.ndarray
        Array of shape :math:`(\text{num_elements}, \text{num_components})`.
        If ``include_covariance=True`` and ``return_single=True``, this array
        is concatenated horizontally with ``covariance``, leading to
        :math:`(\text{num_elements}, (\text{num_components}*(\text{num_components}-1))/2)`
        columns instead.
    covariance : numpy.ndarray
        If ``include_covariance=True`` and ``return_single=False``, array of shape
        :math:`\text{num_components}) + (\text{num_components}*(\text{num_components}-1))/2`.
    """
    assert (cov_mat.ndim == 2) and (cov_mat.shape[0] == cov_mat.shape[1]), \
        f"'cov_mat' must be a 2D square matrix, got array of shape {cov_mat.shape}."
    assert (cov_mat.shape[0] % num_components) == 0, f"'cov_mat' can not be divided " \
        f"into {num_components} components because of incompatible dimensions."
    num_elements = int(cov_mat.shape[0] / num_components)
    variance = np.diag(cov_mat) \
        .reshape(num_elements, num_components).astype(dtype=np.double, casting="safe")
    if include_covariance:
        cov_dims = get_cov_dims(num_components)
        index_map = make_cov_index_map(num_components)[0]
        raveled_indices = np.nonzero(np.isfinite(index_map).ravel())[0]
        assert raveled_indices.size == cov_dims
        covariance = np.empty((num_elements, cov_dims))
        for iobs, iblock in zip(range(num_elements),
                                range(0, num_components*num_elements, num_components)):
            sub_mat = cov_mat[iblock:iblock + num_components,
                              iblock:iblock + num_components]
            covariance[iobs, :] = sub_mat.ravel()[raveled_indices]
    else:
        covariance = None
    if include_covariance and return_single:
        return np.concatenate((variance, covariance), axis=1)
    else:
        return variance, covariance


def block_permutation(n_outer, n_inner):
    r"""
    Convenience function to calculate a permutation matrix used to rearrange (permute)
    blockwise-ordered submatrices in a big matrix.  ``n_outer`` outside blocks of
    individual ``n_inner``-sized blocks will become ``n_inner`` outside blocks of
    individual ``n_outer``-sized blocks.

    Transposing the result is equivalent to calling this function with swapped arguments.

    Parameters
    ----------
    n_outer : int
        Number of sub-matrices.
    n_inner : int
        Size of the individual sub-matrices.

    Returns
    -------
    P : numpy.ndarray
        Square permutation matrix with dimensions
        :math:`n = \text{n_outer} * \text{n_inner}`.
        To permute a matrix :math:`A`, calculate :math:`~P A P^T`.

    Example
    -------

    >>> import numpy as np
    >>> from disstans.tools import block_permutation
    >>> n_outer, n_inner = 2, 2
    >>> A = np.block([[np.arange(n_inner**2).reshape(n_inner, n_inner),
    ...                np.zeros((n_inner, n_inner))], [np.zeros((n_inner, n_inner)),
    ...                np.ones((n_inner, n_inner))]])
    >>> A
    array([[0., 1., 0., 0.],
           [2., 3., 0., 0.],
           [0., 0., 1., 1.],
           [0., 0., 1., 1.]])
    >>> P = block_permutation(n_outer, n_inner)
    >>> P @ A @ P.T
    array([[0., 0., 1., 0.],
           [0., 1., 0., 1.],
           [2., 0., 3., 0.],
           [0., 1., 0., 1.]])
    """
    n = n_outer * n_inner
    Pvals = np.ones(n, dtype=int)
    Prowind = np.arange(n, dtype=int)
    Pcolind = np.arange(n, dtype=int).reshape(n_outer, n_inner).T.ravel()
    P = sparse.coo_matrix((Pvals, (Prowind, Pcolind))).tocsr()
    return P


def cov2corr(cov):
    """
    Function that converts a covariance matrix into a (Pearson) correlation
    matrix, taking into account zero-valued variances and setting the
    respective correlation entries to NaN.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix.

    Returns
    -------
    corr : numpy.ndarray
        Correlation matrix.
    """
    var = np.diag(cov)
    var_nonzero = np.flatnonzero(var)
    cov_nonzero = np.ix_(var_nonzero, var_nonzero)
    corr = np.empty_like(cov)
    corr[:] = np.NaN
    Dinv = np.diag(1/np.sqrt(var[var_nonzero]))
    corr[cov_nonzero] = Dinv @ cov[cov_nonzero] @ Dinv
    return corr


def parallelize(func, iterable, num_threads=None, chunksize=1):
    """
    Convenience wrapper that given a function, an iterable set of inputs
    and parallelization settings automatically either runs the function
    in serial or parallel.

    Warning
    -------
    By default on most systems, NumPy will already use multiple cores and threads
    in its routines (you can check this by running some very large and time-consuming
    math, and monitoring the usage of your processors). Just using multiple Python
    threads will give the default number of threads to all new Python threads,
    completely overloading the system since it's now out of processors, slowing
    down the computations by a lot. The Python :mod:`~multiprocessing`
    module does not change these settings, since it is apparently hard to guess which backend
    NumPy uses, see `this thread on GitHub <https://github.com/numpy/numpy/issues/11826>`_.
    So, it is sadly currently up to the user to disable this behavior when using
    multiple Python threads as achieved with this function. For example,
    this snipped might be enough to put at the beginning of a script:
    ``import os; os.environ['OMP_NUM_THREADS'] = '1'``. Then, the number of DISSTANS cores
    can be set by e.g. ``import disstans; disstans.defaults["general"]["num_threads"] = 10``.
    Another important note is that if you're experiencing problems when running a script,
    make sure the settings and the rest of the script are encapsulated in the standard
    ``if __name__ == "__main__": ...`` clause.

    Parameters
    ----------
    func : function
        Function to wrap, can only have a single input argument.
    iterable : iterable
        Iterable object (list, generator expression, etc.) that contains all the
        arguments that ``func`` should be called with.
    num_threads : int, optional
        Number of threads to use. Set to ``0`` if no parallelization is desired.
        Defaults to the value in :attr:`~disstans.config.defaults`.
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

    >>> from disstans.tools import parallelize
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
        E.g. ``0`` corresponds to white (Gaussian) noise, ``1`` to flicker (pink)
        noise, and ``2`` to random walk (red, Brownian) noise.
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
    # now produce a dictionary that maps sites to a list of step dates: {station: [steptimes]}
    maint_dict = dict(maint_table.groupby(sitecolname)[timecolname].apply(list))
    # rename columns
    maint_table.rename(columns={sitecolname: "station", codecolname: "code",
                                timecolname: "time"}, inplace=True)
    return maint_table, maint_dict


def weighted_median(values, weights, axis=0, percentile=0.5,
                    keepdims=False, visualize=False):
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
    visualize : bool, optional
        If ``True``, show a plot of the weighted median calculation.
        Defaults to ``False``.

    Returns
    -------
    numpy.ndarray
        Weighted median of input.
    """
    # some checks
    assert isinstance(values, np.ndarray) and isinstance(weights, np.ndarray), \
        "'values' and 'weights' must be NumPy arrays, got " + \
        f"{type(values)} and {type(weights)}."
    assert isinstance(axis, int) and (axis < len(values.shape)), \
        f"Axis {axis} is not a valid index for the shape of 'values' ({values.shape})."
    assert (percentile >= 0) and (percentile <= 1), \
        f"'percentile' must be between 0 and 1, got {percentile}."
    assert np.all(np.any(~np.isnan(values), axis=axis)), "'values' must at least " + \
        "contain a single non-NaN element along 'axis' for every other dimension."
    # broadcast the weights
    other_axes = [i for i in range(len(values.shape)) if i != axis]
    weights = np.expand_dims(weights, other_axes)
    # sort the values and weights
    sort_indices = np.argsort(values, axis=axis)
    sort_values = np.take_along_axis(values, sort_indices, axis=axis)
    sort_weights = np.take_along_axis(weights, sort_indices, axis=axis)
    sort_weights *= ~np.isnan(sort_values)  # if there are NaNs, zero out their weights
    # calculate the median cutoff
    cumsum = np.cumsum(sort_weights, axis=axis)
    cutoff = np.sum(sort_weights, axis=axis, keepdims=True) * percentile
    # find the values at that cutoff
    index_array = np.expand_dims(np.argmax(cumsum >= cutoff, axis=axis), axis=axis)
    medians = np.take_along_axis(sort_values, index_array, axis=axis)
    # visualize
    if visualize:
        n_vals = values.shape[axis]
        temp_values = sort_values.transpose(axis, *other_axes).reshape(n_vals, -1)
        temp_weights = sort_weights.transpose(axis, *other_axes).reshape(n_vals, -1)
        temp_cumsum = cumsum.transpose(axis, *other_axes).reshape(n_vals, -1)
        temp_cutoff = cutoff.transpose(axis, *other_axes).reshape(1, -1)
        temp_index = index_array.transpose(axis, *other_axes).reshape(1, -1)
        temp_medians = medians.transpose(axis, *other_axes).reshape(1, -1)
        n_other_axes = temp_values.shape[1]
        for i in range(n_other_axes):
            i_values = temp_values[:, i]
            i_weights = temp_weights[:, i]
            i_cumsum = temp_cumsum[:, i]
            i_cutoff = temp_cutoff[0, i]
            i_index = temp_index[0, i]
            i_medians = temp_medians[0, i]
            plt.figure()
            plt.bar(i_cumsum, i_values, -i_weights, align="edge", color="C0")
            plt.bar(i_cumsum, i_cumsum, -i_weights, align="edge",
                    edgecolor="C1", facecolor=None)
            plt.axhline(i_cutoff, color="k", linestyle="--")
            plt.axvline(i_cumsum[i_index], color="r", linestyle="--")
            plt.title(f"axis {i}: {i_medians}")
            plt.show()
    # squeeze (if desired) and return
    if not keepdims:
        medians = medians.squeeze(axis=axis)
    return medians


def download_unr_data(station_list_or_bbox, data_dir, solution="final",
                      rate="24h", reference=None, min_solutions=100,
                      t_min=None, t_max=None, verbose=False, no_pbar=False):
    """
    Downloads GNSS timeseries data from the University of Nevada at Reno's
    `Nevada Geodetic Laboratory`_. When using this data, please cite [blewitt18]_,
    as well as all the original data providers (the relevant info will be
    downloaded as well).

    Files will only be downloaded if there is no matching file already present,
    or the remote file is newer than the local one.

    Parameters
    ----------
    station_list_or_bbox : list
        Defines which stations to look for data and download.
        It can be either a list of station names (list of strings), a list of bounding
        box coordinates (the four floats ``[lon_min, lon_max, lat_min, lat_max]``
        in degrees), or a three-element list defining a circle (location in degrees
        and radius in kilometers ``[center_lon, center_lat, radius]``).
    data_dir : str
        Folder for data.
    solution : str, optional
        Which timeseries solution to download. Possible values are ``'final'``,
        ``'rapid'`` and ``'ultra'``. See the Notes for approximate latency times.
        Defaults to ``'final'``.
    rate : str, optional
        Which sample rate to download. Possible values are ``'24h'`` and
        ``'5min'``. See the Notes for a table of which rates are available for each
        solution. Defaults to ``'24h'``.
    reference : str, optional
        The UNR abbreviation for the reference frame in which to download the data.
        Applies only for daily sample rates and final or rapid orbit solutions.
        Defaults to ``'IGS14'``.
    min_solutions : int, optional
        Only consider stations with at least a certain number of all-time solutions
        according to the station list file.
        Defaults to ``100``.
    t_min : str or pandas.Timestamp, optional
        Only consider stations that have data on or after ``t_min``.
    t_max : str or pandas.Timestamp, optional
        Only consider stations that have data on or before ``t_max``.
    verbose : bool, optional
        If ``True`` (dault: ``False``), individual actions are printed.
    no_pbar : bool, optional
        Suppress the progress bar with ``True`` (default: ``False``).

    Returns
    -------
    stations : pandas.DataFrame
        A DataFrame, built from UNR's data holding list, subset to the stations
        actually selected for download.

    Notes
    -----

    The following combinations of solution and sample rates are available.
    Note that not all stations are equipped to provide all data types.
    Furthermore, only the daily files will be available in a plate
    reference frame.

    +-----------------+----------+-----------+------------------+
    | orbit solutions | 24 hours | 5 minutes | latency          |
    +=================+==========+===========+==================+
    | final           | yes      | yes       | approx. 2 weeks  |
    +-----------------+----------+-----------+------------------+
    | rapid           | yes      | yes       | approx. 24 hours |
    +-----------------+----------+-----------+------------------+
    | ultra           | no       | yes       | approx. 2 hours  |
    +-----------------+----------+-----------+------------------+

    Warning
    -------

    It is your responsibility that different reference frames or solution types are
    not downloaded into the same folders, because this could lead to the overwriting
    of data or ambiguities as to which files represent which solutions. This is because
    this script does not rename files or change the folder structure that it finds
    on UNR's servers.

    References
    ----------

    .. _`Nevada Geodetic Laboratory`: http://geodesy.unr.edu/
    .. [blewitt18] Blewitt, G., Hammond, W., & Kreemer, C. (2018).
       *Harnessing the GPS Data Explosion for Interdisciplinary Science*. Eos, 99.
       doi:`10.1029/2018EO104623 <https://doi.org/10.1029/2018EO104623>`_

    See Also
    --------
    parse_unr_steps : Function to download and parse UNR's main step file.
    """
    # do some checks
    assert solution in ["final", "rapid", "ultra"], \
        f"Please choose a valid orbit solution (got {solution})."
    assert rate in ["24h", "5min"], \
        f"Please choose a valid sample rate (got {rate})."
    if (solution == "ultra") and (rate == "24h"):
        raise ValueError("There are no ultra-rapid daily solutions available.")
    assert isinstance(station_list_or_bbox, list), \
        f"'station_list_or_bbox' needs to be a list, got {type(station_list_or_bbox)}."
    # make the necessary folders
    atr_dir = os.path.join(data_dir, "attributions")
    if verbose:
        print(f"Making sure '{data_dir}' and '{atr_dir}' exist.")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(atr_dir, exist_ok=True)
    # set master station list URL and define the function that returns the URL
    # (since we don't know which stations and times to actually download)
    base_url = "http://geodesy.unr.edu/gps_timeseries/"
    if solution == "final":
        if rate == "24h":
            if (reference is None) or (reference == "IGS14"):
                def get_sta_url(sta):
                    return base_url + f"tenv3/IGS14/{sta}.tenv3"
            else:
                def get_sta_url(sta):
                    return base_url + f"tenv3/plates/{reference}/{sta}.{reference}.tenv3"
        elif rate == "5min":
            def get_sta_url(sta, year):
                return base_url + f"kenv/{sta}/{sta}.{year}.kenv.zip"
        station_list_url = "http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
    elif solution == "rapid":
        if rate == "24h":
            if (reference is None) or (reference == "IGS14"):
                def get_sta_url(sta):
                    return base_url + f"rapids/tenv3/{sta}.tenv3"
            else:
                def get_sta_url(sta):
                    return base_url + f"rapids/plates/tenv3/{reference}/{sta}.{reference}.tenv3"
            station_list_url = "http://geodesy.unr.edu/NGLStationPages/DataHoldingsRapid24hr.txt"
        elif rate == "5min":
            def get_sta_url(sta, year):
                return base_url + f"rapids_5min/kenv/{sta}/{sta}.{year}.kenv.zip"
            station_list_url = "http://geodesy.unr.edu/NGLStationPages/DataHoldingsRapid5min.txt"
    elif solution == "ultra":
        def get_sta_url(sta, year, doy, date):
            return base_url + f"ultracombo/kenv/{year}/{doy}/{date}{sta}_fix.kenv"
        station_list_url = "http://geodesy.unr.edu/NGLStationPages/DataHoldingsUltra5min.txt"
    station_list_path = os.path.join(data_dir, station_list_url.split("/")[-1])
    # download the station list and parse to a DataFrame
    if verbose:
        print(f"Downloading station list from {station_list_url} to {station_list_path}.")
    try:
        request.urlretrieve(station_list_url, station_list_path)
    except error.HTTPError as e:
        raise RuntimeError("Failed to download the station list from "
                           f"{station_list_url}.").with_traceback(e.__traceback__) from e
    stations = pd.read_csv(station_list_path, delim_whitespace=True, usecols=list(range(11)),
                           parse_dates=[7, 8, 9], infer_datetime_format=True)
    # subset according to station_list_or_bbox
    if all([isinstance(site, str) for site in station_list_or_bbox]):
        # list contains stations names
        stations = stations[stations["Sta"].isin(station_list_or_bbox)]
    elif all([isinstance(value, float) or isinstance(value, int)
              for value in station_list_or_bbox]):
        if len(station_list_or_bbox) == 4:
            # list is a bounding box
            [lon_min, lon_max, lat_min, lat_max] = station_list_or_bbox
            assert lat_min < lat_max, "'lat_min' needs to be smaller than 'lat_max'."
            lon_min = (lon_min + 360) % 360
            lon_max = (lon_max + 360) % 360
            if lon_min < lon_max:
                lon_subset = (stations["Long(deg)"] <= lon_max) & \
                             (stations["Long(deg)"] >= lon_min)
            elif lon_min > lon_max:
                lon_subset = (stations["Long(deg)"] <= lon_max) | \
                             (stations["Long(deg)"] >= lon_min)
            else:
                lon_subset = True
            lat_subset = (stations["Lat(deg)"] <= lat_max) & \
                         (stations["Lat(deg)"] >= lat_min)
            stations = stations[lat_subset & lon_subset]
        elif len(station_list_or_bbox) == 3:
            # list is a location and radius
            [center_lon, center_lat, radius] = station_list_or_bbox
            center_lonlat = np.array([[center_lon, center_lat]])
            geoid = cgeod.Geodesic()
            station_lonlat = stations[["Long(deg)", "Lat(deg)"]].values
            distances = geoid.inverse(center_lonlat, station_lonlat)
            distances = np.array(distances)[:, 0] / 1e3
            stations = stations[distances <= radius]
        else:
            raise ValueError("Could not parse 'station_list_or_bbox' " +
                             str(station_list_or_bbox))
    else:
        raise ValueError("Could not parse 'station_list_or_bbox' " +
                         str(station_list_or_bbox))
    # subset according to data availability
    stations = stations[stations["NumSol"] >= min_solutions]
    if t_min is not None:
        stations = stations[stations["Dtend"] >= pd.Timestamp(t_min)]
    if t_max is not None:
        stations = stations[stations["Dtbeg"] <= pd.Timestamp(t_max)]
    # this is now the final list of stations we're trying to download
    stations_list = stations["Sta"].to_list()
    if verbose:
        print(f"List of stations to download: {stations_list}.")
    if len(stations_list) == 0:
        raise RuntimeError("No stations to download after applying all filters.")
    # prepare list of URLs to download
    if rate == "24h":
        list_urls = [get_sta_url(sta) for sta in stations_list]
        if verbose:
            print("No parsing of index pages necessary.")
    else:
        # if it's a 5min sampling rate, there's multiple files per station,
        # and we need to parse the index webpage for all possible links
        list_urls = []
        if (solution == "final") or (solution == "rapid"):
            if solution == "final":
                index_url = base_url + "kenv/"
            else:
                index_url = base_url + "rapids_5min/kenv/"
            iter_stations_list = tqdm(stations_list, desc="Parsing index pages",
                                      ascii=True, unit="station", disable=no_pbar)
            for sta in iter_stations_list:
                pattern = r'(?<=<a href=")' + str(sta) + \
                          r'\.(\d{4})\.kenv\.zip(?=">)'
                extractor = re.compile(pattern, re.IGNORECASE)
                with request.urlopen(index_url + f"{sta}/") as f:
                    index_page = f.read().decode("windows-1252")
                    avail_years = extractor.findall(index_page)
                list_urls.extend([get_sta_url(sta, year) for year in avail_years])
        elif solution == "ultra":
            index_url = base_url + "ultracombo/kenv/"
            pattern_y = r'(?<=<a href=")(\d{4})/(?=">)'
            pattern_d = r'(?<=<a href=")(\d{3})/(?=">)'
            pattern_f = r'(?<=<a href=")(\d{2}\w{3}\d{2})(\w{4})_fix\.kenv(?=">)'
            extractor_y = re.compile(pattern_y, re.IGNORECASE)
            extractor_d = re.compile(pattern_d, re.IGNORECASE)
            extractor_f = re.compile(pattern_f, re.IGNORECASE)
            if verbose:
                print("Parsing main index page... ", end="", flush=True)
            with request.urlopen(index_url) as f:
                index_page = f.read().decode("windows-1252")
                avail_years = extractor_y.findall(index_page)
            if verbose:
                print("Done")
            iter_avail_years = tqdm(avail_years, desc="Parsing daily index pages",
                                    ascii=True, unit="year", disable=no_pbar)
            for year in iter_avail_years:
                with request.urlopen(index_url + f"{year}/") as f:
                    index_page = f.read().decode("windows-1252")
                    avail_doys = extractor_d.findall(index_page)
                for doy in avail_doys:
                    with request.urlopen(index_url + f"{year}/{doy}/") as f:
                        index_page = f.read().decode("windows-1252")
                        avail_files = extractor_f.findall(index_page)
                    for staurl in avail_files:
                        sta, date = staurl
                        if sta in stations_list:
                            list_urls.append(get_sta_url(sta, year, doy, date))
    if len(list_urls) == 0:
        raise RuntimeError("No files to download after looking on the server.")
    elif (len(list_urls) > 10000) and (not no_pbar):
        answer = input("WARNING: The current selection criteria would lead to "
                       f"downloading {len(list_urls)} files (from "
                       f"{len(stations_list)} stations).\nPress ENTER to continue, "
                       "or anything else to abort.")
        if answer != "":
            exit()
    # prepare list of attribution files
    atr_base_url = "http://geodesy.unr.edu/NGLStationPages/attributions/"
    pattern_atr = r'(?<=<a href=")(\w{4}\.atr\d?)(?=">)'
    extractor_atr = re.compile(pattern_atr, re.IGNORECASE)
    if verbose:
        print("Parsing attributions index page... ", end="", flush=True)
    with request.urlopen(atr_base_url) as f:
        index_page = f.read().decode("windows-1252")
        avail_atr_files = extractor_atr.findall(index_page)
    if verbose:
        print("Done")
    list_atr_urls = []
    for sta in stations_list:
        list_atr_urls.append([atr_base_url + atr_file for atr_file in avail_atr_files
                              if sta in atr_file])
    # loop over list of URLs
    iter_list_urls = tqdm(list_urls, desc="Downloading station timeseries",
                          ascii=True, unit="station", disable=no_pbar)
    for staurl, atrurls in zip(iter_list_urls, list_atr_urls):
        # local target path
        local_path = os.path.join(data_dir, staurl.split("/")[-1])
        # check if local file exists and if so, get its last-modified time
        if os.path.isfile(local_path):
            local_time = \
                pd.Timestamp(datetime.fromtimestamp(os.path.getmtime(local_path),
                                                    tz=timezone.utc))
            local_time_str = local_time.isoformat()
        else:
            local_time, local_time_str = None, "N/A"
        # open the remote connection
        try:
            with request.urlopen(staurl) as remote:
                # get the remote last-modified time
                remote_time = pd.Timestamp(remote.headers["Last-Modified"])
                if (local_time is None) or (remote_time > local_time):
                    # need to download the file, since we either don't have a local copy
                    # or the remote one is newer than the one we have
                    status = "NEW FILE" if local_time is None else "UPDATE"
                    with open(local_path, mode="wb") as local:
                        local.write(remote.read())
                else:
                    status = "SKIPPED"
        except error.HTTPError as e:
            warn(f"Failed to download the remote file from {staurl}.\n"
                 f"HTTP Error {e.code}: {e.reason}", category=RuntimeWarning, stacklevel=2)
        else:
            if verbose:
                tqdm.write(f"[{status}] '{staurl}' ({remote_time.isoformat()})"
                           f" -> '{local_path}' ({local_time_str})")
        # repeat the same procedure for the attribution files
        for atrurl in atrurls:
            local_atr_path = os.path.join(atr_dir, atrurl.split("/")[-1])
            if os.path.isfile(local_atr_path):
                local_atr_time = \
                    pd.Timestamp(datetime.fromtimestamp(os.path.getmtime(local_atr_path),
                                                        tz=timezone.utc))
                local_atr_time_str = local_atr_time.isoformat()
            else:
                local_atr_time, local_atr_time_str = None, "N/A"
            try:
                with request.urlopen(atrurl) as remote_atr:
                    remote_atr_time = pd.Timestamp(remote_atr.headers["Last-Modified"])
                    if (local_atr_time is None) or (remote_atr_time > local_atr_time):
                        status = "NEW FILE" if local_atr_time is None else "UPDATE"
                        with open(local_atr_path, mode="wb") as local_atr:
                            local_atr.write(remote_atr.read())
                    else:
                        status = "SKIPPED"
            except error.HTTPError as e:
                warn(f"Failed to download the remote attribution file from {atrurl}.\n"
                     f"HTTP Error {e.code}: {e.reason}", category=RuntimeWarning, stacklevel=2)
            else:
                if verbose:
                    tqdm.write(f"[{status}] '{atrurl}' ({remote_atr_time.isoformat()})"
                               f" -> '{local_atr_path}' ({local_atr_time_str})")
    # return the DataFrame with the downloaded stations
    return stations


def parse_unr_steps(filepath, check_update=True, only_stations=None, verbose=False):
    """
    This functions parses the main step file from UNR and produces two step databases,
    one for maintenance and one for earthquake-related events.
    If a newer step file is found online, the local copy is updated.

    See :func:`~download_unr_data` for more information about UNR's dataset, as well as
    how to access and cite it.

    Parameters
    ----------
    filepath : str
        Path to the step file.
    check_update : bool, optional
        If ``True`` (default), check UNR's server for an updated step file.
    only_stations : list, optional
        If specified, a list of station IDs. Other stations are not included in the output.
    verbose : bool, optional
        If ``True`` (defaults to ``False``) print actions.

    Returns
    -------
    maint_table : pandas.DataFrame
        Parsed maintenance table.
    maint_dict : dict
        Dictionary of that maps the station names to a list of maintenance steptimes.
    eq_table : pandas.DataFrame
        Parsed earthquake table.
    eq_dict : dict
        Dictionary of that maps the station names to a list of earthquake-related steptimes.
    """
    # check if local file exists
    local_exists = os.path.isfile(filepath)
    # abort if no local file exists but also no update should be performed
    assert local_exists or check_update, \
        "The local file does not exist and no update was requested. No parsing possible."
    # see if we need to download a newer step file
    if check_update:
        # check local last-modified time
        if local_exists:
            local_time = pd.Timestamp(datetime.fromtimestamp(os.path.getmtime(filepath),
                                                             tz=timezone.utc))
            local_time_str = local_time.isoformat()
        else:
            local_time, local_time_str = None, "N/A"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        remote_url = "http://geodesy.unr.edu/NGLStationPages/steps.txt"
        # open the remote connection
        try:
            with request.urlopen(remote_url) as remote:
                # get the remote last-modified time
                remote_time = pd.Timestamp(remote.headers["Last-Modified"])
                if (local_time is None) or (remote_time > local_time):
                    # need to download the file, since we either don't have a local copy
                    # or the remote one is newer than the one we have
                    status = "NEW FILE" if local_time is None else "UPDATE"
                    with open(filepath, mode="wb") as local:
                        local.write(remote.read())
                else:
                    status = "SKIPPED"
        except error.HTTPError as e:
            raise RuntimeError(f"Failed to download the remote file from {remote_url}.\n"
                               f"HTTP Error {e.code}: {e.reason}"
                               ).with_traceback(e.__traceback__) from e
        else:
            if verbose:
                tqdm.write(f"[{status}] '{remote_url}' ({remote_time.isoformat()})"
                           f" -> '{filepath}' ({local_time_str})")
    # load the file
    col_names = ["station", "time", "code", "type", "distance", "magnitude", "usgsid"]
    # (for earthquake events, the "type" column is actually the "threshold" column)
    raw = pd.read_csv(filepath, names=col_names, delim_whitespace=True)
    # we now have a locale-dependent time column in the non-standard format yymmmdd
    # (%y%b%d in strptime language) which we need to convert in a hard-coded way, because we
    # shouldn't change the locale temporarily as it affects the entire system
    unrmonthmap = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
                   "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
                   "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"}
    raw["time"] = pd.to_datetime(
        raw["time"].apply(lambda yymmmdd:
                          yymmmdd[:2] + unrmonthmap[yymmmdd[2:5]] + yymmmdd[-2:]),
        format=r"%y%m%d")
    # subset to specified stations
    if only_stations:
        raw = raw[raw["station"].isin(only_stations)]
    # split the DataFrame into two
    maint_table = raw[raw["code"] == 1].iloc[:, [0, 1, 3]]
    eq_table = raw[raw["code"] == 2].iloc[:, [0, 1, 3, 4, 5, 6]]
    eq_table.rename(columns={"type": "threshold"}, inplace=True)
    del raw  # raw also contains a lot of NaNs because of extra columns we don't need to keep
    # print the different maintenance codes and sizes
    if verbose:
        unique_descs = maint_table["type"].unique().tolist()
        print("Maintenance descriptions:\n", unique_descs)
        print("Number of Maintenance Events:", maint_table.shape[0])
        print("Number of Earthquake-related Events:", eq_table.shape[0])
    # make the dictionaries in form {station: [steptimes]}
    maint_dict = dict(maint_table.groupby("station")["time"].apply(list))
    eq_dict = dict(eq_table.groupby("station")["time"].apply(list))
    # return everything
    return maint_table, maint_dict, eq_table, eq_dict


def strain_rotation_invariants(epsilon=None, omega=None):
    r"""
    Given a strain (rate) and/or rotation (rate) tensor, calculate scalar
    invariant quantities of interest. See [tape09]_ for an introduction.

    Parameters
    ----------
    epsilon : numpy.ndarray
        Strain (rate) tensor :math:`\mathbf{\varepsilon}`.
    omega : numpy.ndarray
        Rotation (rate) tensor :math:`\mathbf{\omega}`.

    Returns
    -------
    dilatation : float
        Only if ``epsilon`` is provided. Scalar dilatation (rate) as defined
        by the first invariant of the strain (rate) tensor
        :math:`\Theta = \text{Tr} \left( \mathbf{\varepsilon} \right)`.
    strain : float
        Only if ``epsilon`` is provided. Scalar strain (rate) as defined
        by the Frobenius norm of the strain (rate) tensor
        :math:`\Sigma = \lVert \mathbf{\varepsilon} \rVert_F`
    shear : float
        Only if ``epsilon`` is provided. Scalar shearing (rate) as defined
        by the square root of the second invariant of the deviatoric strain (rate) tensor
        :math:`\text{T} = \sqrt{\frac{1}{2} \text{Tr}(\mathbf{\varepsilon}^2)
        - \frac{1}{6} \text{Tr}(\mathbf{\varepsilon})^2}`.
    rotation : float
        Only if ``omega`` is provided. Scalar rotation (rate) as defined
        by :math:`\Omega = \frac{1}{\sqrt{2}} \lVert \mathbf{\omega} \rVert_F`.

    References
    ----------

    .. [tape09] Tape, C., Musé, P., Simons, M., Dong, D., & Webb, F. (2009),
       *Multiscale estimation of GPS velocity fields*,
       Geophysical Journal International, 179(2), 945–971,
       doi:`10.1111/j.1365-246X.2009.04337.x <https://doi.org/10.1111/j.1365-246X.2009.04337.x>`_.

    """
    if epsilon is not None:
        assert isinstance(epsilon, np.ndarray) and (epsilon.ndim == 2), \
            f"'epsilon' needs to be a 2D NumPy array, got {epsilon}."
        # scalar dilatation (rate)
        dilatation = np.trace(epsilon)
        # scalar strain (rate)
        strain = np.linalg.norm(epsilon, ord="fro")
        # scalar shearing (rate)
        shear = np.sqrt(np.trace(epsilon @ epsilon) / 2 - np.trace(epsilon)**2 / 6)
    if omega is not None:
        assert isinstance(omega, np.ndarray) and (omega.ndim == 2), \
            f"'omega' needs to be a 2D NumPy array, got {omega}."
        # scalar rotation (rate)
        rotation = np.linalg.norm(omega, ord="fro") / np.sqrt(2)
    # return quantities
    if epsilon is not None:
        if omega is not None:
            return dilatation, strain, shear, rotation
        else:
            return dilatation, strain, shear
    elif omega is not None:
        return rotation


class RINEXDataHolding():
    """
    Container class for a database of RINEX files.

    A new object can be created by one of the two classmethods:

    * From one or multiple folder(s) using :meth:`~from_folders`
    * From a previously-saved file using :meth:`~from_file`

    An object can be saved by using Pandas' :meth:`~pandas.DataFrame.to_pickle`
    on the instance's :attr:`~df` attribute (it is recommended to add the ``.gz``
    extension to enable compression).

    The location information and availability metrics can be saved in the
    same way. To load a previously-saved file, you can use the convenience functions
    :meth:`~load_locations_from_file` and :meth:`~load_metrics_from_file`, specify
    the respective paths in the call to :meth:`~from_file`, or alternatively, load
    the data directly with Pandas and assign it to the respective instance attributes.
    """

    GLOBPATTERN = "[0-9][0-9][0-9][0-9]/[0-9][0-9][0-9]/*"
    """ The ``YYYY/DDD`` folder pattern in a glob-readable format. """

    RINEXPATTERN = r"(?P<site>\w{4})(?P<day>\d{3})(?P<sequence>\w{1})\." + \
                   r"(?P<yy>\d{2})(?P<type>\w{1})\.(?P<compression>\w+)"
    """ The regex-style filename pattern for RINEX files. """

    COMPRFILEEXTS = (".Z", ".gz")
    """ The valid (compressed) RINEX file extensions to search for. """

    COLUMNS = ("station", "station_raw", "year", "day", "date", "sequence", "type",
               "compression", "filesize", "filetimeutc", "network", "basefolder")
    """ The necessary information about each RINEX file. """

    METRICCOLS = ("number", "age", "recency", "length", "reliability")
    """ The metrics that can be calculated. """

    def __init__(self, df=None):
        self._df = None
        self._locations_xyz = None
        self._locations_lla = None
        self._metrics = None
        if df is not None:
            self.df = df

    @property
    def num_files(self):
        """ Number of files in the database. """
        return self.df.shape[0]

    @property
    def list_stations(self):
        """ List of stations in the database. """
        return self.df["station"].unique().tolist()

    @property
    def num_stations(self):
        """ Number of stations in the database. """
        return len(self.list_stations)

    @property
    def df(self):
        """ Pandas DataFrame object containing the RINEX files database. """
        if self._df is None:
            raise RuntimeError("RINEX files database has not been loaded yet.")
        return self._df

    @df.setter
    def df(self, new_df):
        try:
            assert all([col in new_df.columns for col in self.COLUMNS]), \
                "Input DataFrame does not contain the necessary columns, try using the " + \
                "class constructor methods."
        except AttributeError as e:
            raise TypeError("Cannot interpret input as a Pandas DataFrame for the RINEX "
                            "file database.").with_traceback(e.__traceback__) from e
        self._df = new_df

    @property
    def locations_xyz(self):
        """
        Dataframe of approximate positions of stations in WGS-84 (x, y, z) [m] coordinates.
        """
        if self._locations_xyz is None:
            raise RuntimeError("Locations have not been loaded yet.")
        if not all([station in self._locations_xyz["station"].values
                    for station in self.list_stations]):
            warn("Locations have likely not been updated since the database changed, "
                 "there are stations missing.", category=RuntimeWarning, stacklevel=2)
        return self._locations_xyz

    @locations_xyz.setter
    def locations_xyz(self, new_xyz):
        if not (isinstance(new_xyz, pd.DataFrame) and
                all([col in new_xyz.columns for col in ["station", "x", "y", "z"]])):
            raise ValueError("Unrecognized input format. 'new_xyz' needs to be a "
                             "Pandas DataFrame with the columns ['station', 'x', 'y', 'z'].")
        if not all([station in new_xyz["station"].values for station in self.list_stations]):
            warn("The new location DataFrame does not contain all stations "
                 "that are currently in the database.", category=RuntimeWarning, stacklevel=2)
        all_xyz = new_xyz[["x", "y", "z"]].values
        all_lla = ccrs.Geodetic().transform_points(ccrs.Geocentric(), all_xyz[:, 0],
                                                   all_xyz[:, 1], all_xyz[:, 2])
        new_lla = pd.DataFrame(new_xyz["station"]
                               ).join(pd.DataFrame(all_lla, columns=["lon", "lat", "alt"]))
        self._locations_xyz = new_xyz
        self._locations_lla = new_lla

    @property
    def locations_lla(self):
        """
        Approximate positions of stations in WGS-84 (longitude [°], latitude [°],
        altitude [m]) coordinates.
        """
        if self._locations_lla is None:
            raise RuntimeError("Locations have not been loaded yet.")
        if not all([station in self._locations_lla["station"].values
                    for station in self.list_stations]):
            warn("Locations have likely not been updated since the database changed, "
                 "there are stations missing.", category=RuntimeWarning, stacklevel=2)
        return self._locations_lla

    @locations_lla.setter
    def locations_lla(self, new_lla):
        if not (isinstance(new_lla, pd.DataFrame) and
                all([col in new_lla.columns for col in ["station", "lon", "lat", "alt"]])):
            raise ValueError("Unrecognized input format. 'new_lla' needs to be a "
                             "Pandas DataFrame with the columns "
                             "['station', 'lon', 'lat', 'alt'].")
        if not all([station in new_lla["station"].values for station in self.list_stations]):
            warn("The new location DataFrame does not contain all stations "
                 "that are currently in the database.", category=RuntimeWarning, stacklevel=2)
        all_lla = new_lla[["lon", "lat", "alt"]].values
        all_xyz = ccrs.Geocentric().transform_points(ccrs.Geodetic(), all_lla[:, 0],
                                                     all_lla[:, 1], all_lla[:, 2])
        new_xyz = pd.DataFrame(new_lla["station"]
                               ).join(pd.DataFrame(all_xyz, columns=["x", "y", "z"]))
        self._locations_xyz = new_xyz
        self._locations_lla = new_lla

    @property
    def metrics(self):
        """
        Contains the station metric calculated by :meth:`calculate_availability_metrics`.
        """
        if self._metrics is None:
            raise RuntimeError("Metrics have not been calculated yet.")
        if not all([station in self._metrics["station"].values
                    for station in self.list_stations]):
            warn("Metrics have likely not been updated since the database changed, "
                 "there are stations missing.", category=RuntimeWarning, stacklevel=2)
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        assert all([col in metrics.columns for col in self.METRICCOLS]), \
            "Not all metrics are in the new DataFrame."
        self._metrics = metrics

    @classmethod
    def from_folders(cls, folders, verbose=False, no_pbar=False):
        """
        Convenience class method that creates a new RINEXDataHolding object and directly
        calls :meth:`~load_db_from_folders`.

        For parameter explanations, see the above method documention.
        """
        instance = cls()
        instance.load_db_from_folders(folders, verbose=verbose, no_pbar=no_pbar)
        return instance

    def load_db_from_folders(self, folders, verbose=False, no_pbar=False):
        """
        Loads a RINEX database from folders in the file system.
        The data should be located in one or multiple folder structure(s) organized by
        ``YYYY/DDD``, where ``YYYY`` is a four-digit year and ``DDD`` is the three-digit
        day of the year.

        Parameters
        ----------
        folders : list, tuple
            Folder(s) in which the different year-folders are found, formatted as a
            single tuple or a list of tuples with name and respective folder
            (``[('network1', '/folder/one/'), ...]``).
        verbose : bool, optional
            If ``True``, print final database size and a sample entry.
            Defaults to ``False``.
        no_pbar : bool, optional
            Suppress the progress bar with ``True`` (default: ``False``).
        """
        # input checks
        if isinstance(folders, tuple):
            folders = [folders]
        if not (isinstance(folders, list) and
                all([isinstance(tup, tuple) and len(tup) == 2 for tup in folders]) and
                all([all([isinstance(elem, str) for elem in tup]) for tup in folders])):
            raise ValueError("Invalid 'folder' argument, pass a list of tuples "
                             "composed of a name and folder string each, "
                             f"got {folders}.")
        # empty starting values
        dfdict = {col: [] for col in self.COLUMNS}
        # determine iterator based on verbosity
        iterfolders = tqdm(folders, desc="Loading folders", ascii=True,
                           unit="folder", disable=no_pbar)
        # loop over folder(s)
        for network, folder in iterfolders:
            # initialize pattern extraction
            rinex_pattern = re.compile(self.RINEXPATTERN, re.IGNORECASE)
            cur_year = None
            # loop over files
            if verbose:
                tqdm.write(f"Loading from folder {folder} the year(s):")
            for pathobj in Path(folder).rglob(self.GLOBPATTERN):
                year, day, filename = pathobj.parts[-3:]
                # get clean year and day in case the globpattern changed
                if self.GLOBPATTERN != RINEXDataHolding.GLOBPATTERN:
                    year = re.findall(r"\d+", year)[0]
                    if len(year) != 4:
                        if len(year) == 2:
                            year = "20" + year
                            warn(f"Parsing {str(pathobj)} assumes a two-digit year is "
                                 "post-2000.", category=RuntimeWarning, stacklevel=2)
                        else:
                            warn(f"File {str(pathobj)} doesn't have a recognizable year, "
                                 "skipping file.", category=RuntimeWarning, stacklevel=2)
                            continue
                    day = re.findall(r"\d+", day)[0]
                    day = f"{int(day):03d}"
                    if not (0 < int(day) < 367):
                        warn(f"File {str(pathobj)} doesn't have a valid day, "
                             "skipping file.", category=RuntimeWarning, stacklevel=2)
                        continue
                # only continue if this is a valid file extension
                if filename.endswith(self.COMPRFILEEXTS):
                    info = rinex_pattern.match(filename)
                    if info is None:
                        warn(f"File {str(pathobj)} can't match RINEX filename pattern, "
                             "skipping file.", category=RuntimeWarning, stacklevel=2)
                        continue
                    info = info.groupdict()
                    if (info["yy"] != year[-2:] or info["day"] != day):
                        skipmsg = f"File '{str(pathobj)} has conflicting year/day information, " \
                                  "skipping file "
                        trystem = f"{info['site']}{day}{info['sequence']}." \
                                  f"{year[-2:]}{info['type']}"
                        tryfile = Path(pathobj.parents[0], trystem + pathobj.suffix)
                        if tryfile.is_file():
                            skipmsg += f"(but {str(tryfile)} exists)."
                        else:
                            skipmsg += f"(and {str(tryfile)} also doesn't exist)."
                        warn(skipmsg, category=RuntimeWarning, stacklevel=2)
                        continue
                    if verbose and (cur_year != year):
                        tqdm.write(f" {year}")
                        cur_year = year
                    date = datetime.strptime(f"{year} {day}", "%Y %j").date()
                    filestat = os.stat(pathobj)
                    filesize = filestat.st_size
                    filetime = datetime.fromtimestamp(filestat.st_mtime, tz=timezone.utc)
                    dfdict["station"].append(info["site"].upper())
                    dfdict["station_raw"].append(info["site"])
                    dfdict["year"].append(year)
                    dfdict["day"].append(day)
                    dfdict["date"].append(date)
                    dfdict["sequence"].append(info["sequence"])
                    dfdict["type"].append(info["type"])
                    dfdict["compression"].append(info["compression"])
                    dfdict["filesize"].append(filesize)
                    dfdict["filetimeutc"].append(filetime)
                    dfdict["network"].append(network)
                    dfdict["basefolder"].append(folder)
        # build DataFrame and save some space
        df = pd.DataFrame(dfdict)
        df = df.astype({"network": pd.CategoricalDtype(), "basefolder": pd.CategoricalDtype()})
        if verbose:
            print(f"\nFound {df.shape[0]} files.\nSample:\n")
            print(df.iloc[0, :])
        # save to attribute
        self.df = df

    @classmethod
    def from_file(cls, db_file, locations_file=None, metrics_file=None, verbose=False):
        """
        Convenience class method that creates a new RINEXDataHolding object from a file
        using :meth:`~load_db_from_file` and then optionally loads the locations and metrics
        from their respective files.

        Parameters
        ----------
        db_file : str
            Path of the main file.
        locations_file : str, optional
            Path of the locations file.
        metrics_file : str, optional
            Path of the metrics file.
        verbose : bool, optional
            If ``True``, print database size and a sample entry.
            Defaults to ``False``.
        """
        # load instance from file
        instance = cls()
        instance.load_db_from_file(db_file, verbose=verbose)
        # optionally load locations and metrics
        if locations_file:
            instance.load_locations_from_file(locations_file)
        if metrics_file:
            instance.load_metrics_from_file(metrics_file)
        return instance

    def load_db_from_file(self, db_file, verbose=False):
        """
        Loads a RINEXDataHolding object from a pickled Pandas DataFrame file.

        Parameters
        ----------
        db_file : str
            Path of the main file.
        verbose : bool, optional
            If ``True``, print database size and a sample entry.
            Defaults to ``False``.
        """
        # load main database
        df = pd.read_pickle(db_file)
        if verbose:
            print(f"\nFound {df.shape[0]} files.\nSample:\n")
            print(df.iloc[0, :])
        # save to attribute
        self.df = df

    def load_locations_from_rinex(self, keep='last', replace_not_found=False, no_pbar=True):
        """
        Scan the RINEX files' headers for approximate locations for
        plotting purposes.

        Parameters
        ----------
        keep : str, optional
            Determine which location to use. Possible values are ``'last'``
            (only scan the most recent file), ``'first'`` (only scan the oldest
            file) or ``'mean'`` (load all files and calculate average).
            Note that ``'mean'`` could take a substantial amount of time, since
            all files have to opened, decompressed and searched.
        replace_not_found : bool, optional
            If a location is not found and ``replace_not_found=True``,
            the location of Null Island (0° Longitude, 0° Latitude) is used
            and a warning is issued.
            If ``False`` (default), an error is raised instead.
        no_pbar : bool, optional
            Suppress the progress bar with ``True`` (default).
        """
        # prepare
        assert keep in ["first", "last", "mean"], \
            f"Unrecognized 'keep' option {keep}."
        XYZ = ["x", "y", "z"]
        df = pd.DataFrame({"station": self.list_stations})
        df = df.join(pd.DataFrame(np.zeros((self.num_stations, 3)), columns=XYZ))
        # get xyz for each station
        iterfiles = tqdm(df.itertuples(), total=df.shape[0], ascii=True, unit="file",
                         desc="Reading RINEX headers", disable=no_pbar)
        for row in iterfiles:
            subset = self.get_files_by(station=row.station).sort_values(by=["date"])
            filenames = self.make_filenames(subset)
            if keep == "last":
                filenames = reversed(filenames)
            approx_xyz = []
            for f in filenames:
                try:
                    found_pos = self.get_rinex_header(f)["APPROX POSITION XYZ"]
                except KeyError:
                    continue
                approx_xyz.append(np.array(found_pos.split(), dtype=float))
                if keep != "mean":
                    break
            if len(approx_xyz) == 0:
                if replace_not_found:
                    warn("Couldn't find an approximate location in the RINEX "
                         f"headers for {row.station}. Using Null Island.", stacklevel=2)
                    approx_xyz.append([0, 6378137, 0])
                else:
                    raise RuntimeError("Couldn't find an approximate location in "
                                       f"the RINEX headers for {row.station}.")
            approx_xyz = np.array(approx_xyz)
            if keep == "mean":
                approx_xyz = np.mean(approx_xyz, axis=0)
            df.loc[row.Index, XYZ] = approx_xyz.squeeze()
        # set the instance properties
        # the xyz-to-lla conversion is done there
        self.locations_xyz = df

    def load_locations_from_file(self, filepath):
        """
        Load a previously-saved DataFrame containing the locations of each station.

        Parameters
        ----------
        filepath : str
            Path to the pickled DataFrame.
        """
        df = pd.read_pickle(filepath)
        assert "station" in df.columns, f"No 'station' column in {filepath}."
        if all(col in df.columns for col in ["x", "y", "z"]):
            self.locations_xyz = df
        elif all(col in df.columns for col in ["lon", "lat", "alt"]):
            self.locations_lla = df
        else:
            raise ValueError(f"Unrecognized DataFrame columns in {filepath}: " +
                             str(df.columns.tolist()))

    def get_files_by(self, station=None, network=None, year=None, between=None,
                     verbose=False):
        """
        Return a subset of the database by criteria.

        Parameters
        ----------
        station : str, list, optional
            Return only files of this/these station(s).
        network : str, list, optional
            Return only files of this/these network(s).
        year : int, list, optional
            Return only files of this/these year(s).
        between : tuple, optional
            Return only files between the start and end date (inclusive)
            given by the length-two tuple.
        verbose : bool, optional
            If ``True``, print the number of selected entries.
            Defaults to ``False``.
        """
        subset = pd.Series(True, index=range(self.num_files))
        # subset by station
        if station is not None:
            if isinstance(station, str):
                station = [station]
            elif isinstance(station, list):
                assert(all([isinstance(s, str) for s in station])), \
                    "Found non-string station entries in 'station'."
            else:
                raise TypeError("Invalid input form for 'station', must be a string or "
                                f"list of strings, got {station}.")
            subset &= self.df["station"].isin(station)
        # subset by network
        if network is not None:
            if isinstance(network, str):
                network = [network]
            elif isinstance(network, list):
                assert(all([isinstance(s, str) for s in network])), \
                    "Found non-string network entries in 'network'."
            else:
                raise TypeError("Invalid input form for 'network', must be a string or "
                                f"list of strings, got {network}.")
            subset &= self.df["network"].isin(network)
        # subset by year
        if year is not None:
            if isinstance(year, int) or isinstance(year, str):
                year = [year]
            elif isinstance(year, list):
                assert((all([isinstance(y, int) for y in year]) or
                        all([isinstance(y, str) for y in year]))), \
                    "Found non-string/integer year entries in 'year'."
            else:
                raise TypeError("Invalid input form for 'year', must be an integer, string or "
                                f"list of integers or strings, got {year}.")
            try:
                year = [str(int(y)) for y in year]
            except TypeError as e:
                raise TypeError("Invalid input form for 'year', must be an integer, string or "
                                f"list of integers or strings, got {year}."
                                ).with_traceback(e.__traceback__) from e
            subset &= self.df["year"].isin(year)
        # subset by time span
        if between is not None:
            try:
                between = (pd.Timestamp(between[0]), pd.Timestamp(between[1]))
            except (ValueError, TypeError, IndexError) as e:
                raise TypeError("Invalid input form for 'between', needs to be a length-two "
                                "tuple of entries that can be converted to Pandas Timestamps, "
                                f"got {between}.").with_traceback(e.__traceback__) from e
            subset &= (self.df["date"] >= between[0]) & (self.df["date"] <= between[1])
        # return
        if verbose:
            print(f"Selected {subset.sum()} files.")
        return self.df[subset]

    def get_location(self, station, lla=True):
        """
        Returns the approximate location of a station.

        Parameters
        ----------
        station : str
            Name of the station
        lla : bool, optional
            If ``True``, returns the coordinates in Longitude [°], Latitude [°] &
            Altitude [m], otherwise in XYZ [m] coordinates.

        Returns
        -------
        pandas.Series
            The location of the station in the specified coordinate system.
        """
        loc_df = self.locations_lla if lla else self.locations_xyz
        try:
            return loc_df[loc_df["station"] == station].iloc[0] \
                   .drop("station").rename(station)
        except IndexError:
            raise KeyError(f"Station {station} not present.")

    def load_metrics_from_file(self, filepath):
        """
        Load a previously-saved DataFrame containing the calculated availability metrics.

        Parameters
        ----------
        filepath : str
            Path to the pickled DataFrame.
        """
        self.metrics = pd.read_pickle(filepath)

    def make_filenames(self, db):
        """
        Recreate the full paths to the individual rinex files from the database or
        a subset thereof.

        Parameters
        ----------
        db : pandas.DataFrame
            :attr:`~df` or a subset thereof.

        Returns
        -------
        filenames : list
            List of paths.

        Raises
        ------
        NotImplementedError
            If :attr:`~GLOBPATTERN` or :attr:`~RINEXPATTERN` for this instance are not the
            same as the default values. In this case, redefine this function with the
            appropriate folder and file patterns.
        """
        if (self.GLOBPATTERN != RINEXDataHolding.GLOBPATTERN) or \
           (self.RINEXPATTERN != RINEXDataHolding.RINEXPATTERN):
            raise NotImplementedError("GLOBPATTERN or RINEXPATTERN of this instance have been "
                                      "modified, therefore the default way of creating full "
                                      "filenames is no longer valid.")
        return [os.path.join(row.basefolder, row.year, row.day,
                             row.station_raw + row.day + row.sequence + "." +
                             row.year[-2:] + row.type + "." + row.compression)
                for row in db.itertuples()]

    def get_rinex_header(self, filepath):
        """
        Open a RINEX file, read the header, and format it as a dictionary.
        No data type conversion or stripping of whitespaces is performed.

        Parameters
        ----------
        filepath : str
            Path to RINEX file.

        Returns
        -------
        headers : dict
            Dictionary of header lines.
        """
        # if it's a compressed file, hope that gzip is installed and we can use
        # it to decompress on-the-fly
        try:
            if filepath.endswith(self.COMPRFILEEXTS):
                rinexfile = subprocess.check_output(["gzip", "-dc", filepath],
                                                    text=True, errors="replace")
            else:
                with open(filepath, mode="rt", errors="replace") as f:
                    rinexfile = f.read()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Couldn't read compressed RINEX file {filepath}. "
                               "Check if 'gzip' is available on your machine, or "
                               "decompress the RINEX file before using this "
                               "function.").with_traceback(e.__traceback__) from e
        except UnicodeError as e:
            raise RuntimeError(f"Can't read file {filepath}."
                               ).with_traceback(e.__traceback__) from e
        # extract headers
        headers = {}
        for line in rinexfile.splitlines():
            content, descriptor = line[:60], line[60:].strip()
            if descriptor == "END OF HEADER":
                break
            if descriptor in headers:
                headers[descriptor] += f"\n{content}"
            else:
                headers[descriptor] = content
        return headers

    def calculate_availability_metrics(self, sampling=Timedelta(1, "D")):
        """
        Calculates the following metrics and stores them in the :attr:`~metrics`
        DataFrame:

        * ``'number'``: Number of available observations.
        * ``'age'``: Time of first observation.
        * ``'recency'``: Time of last observation.
        * ``'length'``: Time between first and last observation.
        * ``'reliability'``: Reliability defined as number of observations divided
          by the maximum amount of possible observations between the first and last
          acquisition given the assumed sampling interval of the data.

        Parameters
        ----------
        sampling : disstans.tools.Timedelta
            Assumed sampling frequency of the data files.
            Defaults to daily.
        """
        # initialize empty DataFrame
        metrics = pd.DataFrame(self.list_stations, columns=["station"])
        metrics = metrics.join(pd.DataFrame(np.zeros((self.num_stations, 5)),
                                            columns=self.METRICCOLS))
        # calculate metrics station by station
        for row in metrics.itertuples():
            subset = self.get_files_by(station=row.station)
            metrics.loc[row.Index, ["number", "age", "recency"]] = \
                [subset.shape[0], subset["date"].min(), subset["date"].max()]
            metrics.loc[row.Index, "length"] = \
                metrics.loc[row.Index, "recency"] - metrics.loc[row.Index, "age"] + sampling
            metrics.loc[row.Index, "reliability"] = (subset.shape[0] * sampling) / \
                metrics.loc[row.Index, "length"]
        # set attribute
        metrics = metrics.astype({"number": int})
        self.metrics = metrics

    def _create_map_figure(self, gui_settings, annotate_stations, figsize):
        """
        Create a basemap of all stations.
        """
        # get location data and projections
        stat_lats = self.locations_lla["lat"].values
        stat_lons = self.locations_lla["lon"].values
        stat_names = self.locations_lla["station"].values
        proj_gui = getattr(ccrs, gui_settings["projection"])()
        proj_lla = ccrs.PlateCarree()
        # create figure and plot stations
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection=proj_gui)
        stat_points = ax.scatter(stat_lons, stat_lats, s=100, facecolor='C0',
                                 linestyle='None', marker='.', transform=proj_lla,
                                 edgecolor='None', zorder=1000)
        if annotate_stations:
            for sname, slon, slat in zip(stat_names, stat_lons, stat_lats):
                ax.annotate(sname, (slon, slat),
                            xycoords=proj_lla._as_mpl_transform(ax),
                            annotation_clip=True, textcoords="offset pixels",
                            xytext=(0, 5), ha="center")
        # create underlay
        map_underlay = False
        if gui_settings["wmts_show"]:
            try:
                ax.add_wmts(gui_settings["wmts_server"],
                            layer_name=gui_settings["wmts_layer"],
                            alpha=gui_settings["wmts_alpha"])
                map_underlay = True
            except Exception as exc:
                print(exc)
        if gui_settings["coastlines_show"]:
            ax.add_feature(cfeature.BORDERS.with_scale(gui_settings["coastlines_res"]),
                           edgecolor="white" if map_underlay else "black")
            ax.add_feature(cfeature.COASTLINE.with_scale(gui_settings["coastlines_res"]),
                           edgecolor="white" if map_underlay else "black")
        return fig, ax, proj_gui, proj_lla, stat_points, stat_names

    def plot_map(self, metric=None, orientation="horizontal", annotate_stations=True,
                 figsize=None, saveas=None, dpi=None, gui_kw_args={}):
        """
        Plot a map of all the stations present in the RINEX database.
        The markers can be colored by the different availability metrics calculated
        by :meth:`~calculate_availability_metrics`.

        Parameters
        ----------
        metric : str, optional
            Calculate the marker color (and respective colormap) given a certain
            metric. If ``None`` (default), no color is applied.
        orientation : str, optional
            Colorbar orientation, see :func:`~matplotlib.pyplot.colorbar`.
            Defaults to ``'horizontal'``.
        annotate_stations : bool, optional
            If ``True`` (default), add the station names to the map.
        figsize : tuple, optional
            Set the figure size (width, height) in inches.
        saveas : str, optional
            If provided, the figure will be saved at this location.
        dpi : float, optional
            Use this DPI for saved figures.
        gui_kw_args : dict, optional
            Override default GUI settings of :attr:`~disstans.config.defaults`.
        """
        # prepare
        gui_settings = defaults["gui"].copy()
        gui_settings.update(gui_kw_args)
        # get basemap
        fig, ax, proj_gui, proj_lla, stat_points, stat_names = \
            self._create_map_figure(gui_settings, annotate_stations, figsize)
        # add colors
        if metric in self.METRICCOLS:
            # get metric in the same order as the stations in the figure
            met = self.metrics[["station", metric]]
            met_fmt = [met[met["station"] == station][metric].values[0]
                       for station in stat_names]
            # metric is a normal numeric value
            if metric in ["number", "reliability"]:
                met_raw = np.array(met_fmt)
                tickformat = None
            # metric is a timestamp, need to convert
            elif metric in ["age", "recency"]:
                met_ref = min(met_fmt)
                met_raw = np.array([(m - met_ref).total_seconds() for m in met_fmt])
                # make a helper function for the tick formatting
                @FuncFormatter  # noqa: E306
                def tickformat(x, pos):
                    return (met_ref + pd.Timedelta(x, "s")).strftime(r"%Y-%m-%d")
            # metric is a timedelta, need to convert
            elif metric == "length":
                met_raw = np.array([m.value for m in met_fmt])
                # make a helper function for the tick formatting
                @FuncFormatter  # noqa: E306
                def tickformat(x, pos):
                    return str(pd.Timedelta(x, "ns").days)
            # get data range and make colormap
            cmin, cmax = met_raw.min(), met_raw.max()
            cmap = mpl.cm.ScalarMappable(cmap=scm.batlow,
                                         norm=mpl.colors.Normalize(vmin=cmin, vmax=cmax))
            # set marker facecolors
            stat_points.set_facecolor(cmap.to_rgba(met_raw))
            fig.canvas.draw_idle()
            # add the colorbar
            cbar = fig.colorbar(cmap, ax=ax, orientation=orientation,
                                fraction=0.05 if orientation == "horizontal" else 0.2,
                                pad=0.03, aspect=10, format=tickformat)
            cticks = cbar.get_ticks()
            if cticks[0] != cmin:
                cticks = [cmin, *cticks]
            if cticks[-1] != cmax:
                cticks = [*cticks, cmax]
            cbar.set_ticks(cticks)
            cbar.set_label(metric)
        elif metric is not None:
            warn(f"Could not interpret '{metric}' as a metric to use for plotting.",
                 stacklevel=2)
        # save
        if saveas is not None:
            fig.savefig(saveas)
        # show
        plt.show()

    def plot_availability(self, sampling=Timedelta(1, "D"), sort_by_latitude=True,
                          saveas=None):
        """
        Create an availability figure for the dataset.

        Parameters
        ----------
        sampling : disstans.tools.Timedelta, optional
            Assume that breaks strictly larger than ``sampling`` constitute a data gap.
            Defaults to daily.
        sort_by_latitude : bool, optional
            If ``True`` (default), sort the stations by latitude, else alphabetical.
            (Always falls back to alphabetical if location information is missing.)
        saveas : str, optional
            If provided, the figure will be saved at this location.
        """
        # find a sorting by latitude to match a map view,
        # otherwise go by alphabet
        sort_stations = None
        if sort_by_latitude:
            try:
                sort_indices = np.argsort(self.locations_lla["lat"].values)
                sort_stations = self.locations_lla["station"].iloc[sort_indices].tolist()
            except RuntimeError:
                pass
        if sort_stations is None:
            sort_stations = list(reversed(sorted([s.lower() for s in self.list_stations])))
        n_stations = len(sort_stations)
        # make an empty figure and start a color loop
        fig, ax = plt.subplots(figsize=(6, 0.25*n_stations))
        colors = [plt.cm.tab10(i) for i in range(10)]
        icolor = 0
        n_files = []
        # loop over stations in the sorted order
        for offset, station in enumerate(sort_stations):
            # get the station in question
            subset = self.get_files_by(station=station)
            # get the file dates and split them by contiguous chunks
            all_dates = subset["date"].sort_values().values
            n_files.append(all_dates.size)
            if all_dates.size > 1:
                split_at = np.nonzero(np.diff(all_dates) > sampling)[0]
                if split_at.size > 0:
                    intervals = np.split(all_dates, split_at + 1)
                else:
                    intervals = [all_dates]
            else:
                intervals = [all_dates]
            # plot a line for each chunk
            for chunk in intervals:
                ax.fill_between([chunk[0], chunk[-1]],
                                [offset + 0.7, offset + 0.7], [offset + 1.3, offset + 1.3],
                                fc=colors[icolor])
            icolor = (icolor + 1) % 10
        # add station labels
        ax.set_yticks(np.arange(n_stations) + 1)
        ax.set_yticklabels(sort_stations)
        ax.tick_params(which="major", axis="y", left=False)
        # do some pretty formatting
        ax.set_title(f"Network(s): {', '.join(self.df['network'].unique().tolist())}\n"
                     f"Files: {sum(n_files)}")
        ax.grid(which="major", axis="x")
        ax.xaxis.set_tick_params(labeltop='on')
        ax.set_axisbelow(True)
        ax.set_ylim(0.5, n_stations + 0.5)
        # add number of files per station
        ax_right = ax.twinx()
        ax_right.set_ylim(0.5, n_stations + 0.5)
        ax_right.set_yticks(np.arange(n_stations) + 1)
        ax_right.set_yticklabels([f"({n})" for n in n_files], fontsize="x-small")
        ax_right.tick_params(which="major", axis="y", right=False)
        # save
        if saveas is not None:
            fig.savefig(saveas)
        # show
        plt.show()
