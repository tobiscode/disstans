"""
This module contains some processing functions that while not belonging
to a specific class, require them to already be loaded and GeoNAT to
be initialized.

For general helper functions, see :mod:`~geonat.tools`.
"""

import numpy as np
import pandas as pd
import warnings
from functools import wraps
from sklearn.decomposition import PCA, FastICA

from . import defaults, Timeseries
from .compiled import maskedmedfilt2d


def unwrap_dict_and_ts(func):
    """
    A wrapper decorator that aims at simplifying the coding of processing functions.
    Ideally, a new function that doesn't need to know if its input is a
    :class:`~geonat.timeseries.Timeseries`, :class:`~pandas.DataFrame`,
    :class:`~numpy.ndarray` or a dictionary containing them, should not need to reimplement
    a check and conversion for all of these because they just represent a data
    array of some form. So, by providing this function decorator, a wrapped function
    only needs to be able to work for a single array (plus some optional keyword arguments).
    The wrapping will extract the data of the input types and convert the returned
    array from ``func`` into the original format.

    Example
    -------
    A basic function of the form::

        def func(in_array, **kw_args):
            # do some things
            return out_array

    that takes and returns a NumPy array can be wrapped as follows
    to be able to also take and return all the other data forms::

        @unwrap_dict_and_ts
        def func(in_array, **kw_args):
            # do some things
            return out_array
    """
    @wraps(func)
    def wrapper(data, *args, **kw_args):
        if not isinstance(data, dict):
            data = {'ts': data}
            was_dict = False
        else:
            was_dict = True
        out = {}
        additional_output = {}
        # loop over components
        for comp, ts in data.items():
            if isinstance(ts, Timeseries):
                array = ts.data.values
            elif isinstance(ts, pd.DataFrame):
                array = ts.values
            elif isinstance(ts, np.ndarray):
                array = ts
            else:
                raise TypeError(f"Cannot unwrap object of type {type(ts)}.")
            func_output = func(array, *args, **kw_args)
            if isinstance(func_output, tuple):
                result = func_output[0]
                try:
                    additional_output[comp] = func_output[1:]
                except IndexError:
                    additional_output[comp] = None
            else:
                result = func_output
                additional_output[comp] = None
            # save results
            if isinstance(ts, Timeseries):
                out[comp] = ts.copy(only_data=True, src=func.__name__)
                out[comp].data = result
            elif isinstance(ts, pd.DataFrame):
                out[comp] = ts.copy()
                out[comp].values = result
            else:
                out[comp] = result
        if all([elem is None for elem in additional_output.values()]):
            has_additional_output = False
        else:
            has_additional_output = True
        if not was_dict:
            out = out['ts']
            additional_output = additional_output['ts']
        if has_additional_output:
            return out, additional_output
        else:
            return out
    return wrapper


@unwrap_dict_and_ts
def median(array, kernel_size):
    """
    Computes the median filter (ignoring NaNs) column-wise, either by calling
    :func:`~numpy.nanmedian` iteratively or by using the precompiled Fortran
    function :func:`~geonat.compiled.maskedmedfilt2d`.

    Parameters
    ----------
    array : numpy.ndarray
        2D input array (can contain NaNs).
        Wrapped by :func:`~geonat.processing.unwrap_dict_and_ts` to also accept
        :class:`~geonat.timeseries.Timeseries`, :class:`~pandas.DataFrame` and
        dictionaries of them as input.
    kernel_size : int
        Kernel size (length of moving window to compute the median over).
        Has to be an odd number.

    Returns
    -------
    filtered : numpy.ndarray
        2D filtered array (may still contain NaNs).
    """
    try:
        filtered = maskedmedfilt2d(array, ~np.isnan(array), kernel_size)
        filtered[np.isnan(array)] = np.NaN
    except BaseException:
        assert (kernel_size % 2) == 1, f"'kernel_size' has to be odd, got {kernel_size}."
        num_obs = array.shape[0]
        array = array.reshape(num_obs, 1 if array.ndim == 1 else -1)
        filtered = np.NaN * np.empty(array.shape)
        # Run filtering while suppressing warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered')
            # Beginning region
            halfwindow = 0
            for i in range(kernel_size // 2):
                filtered[i, :] = np.nanmedian(array[i-halfwindow:i+halfwindow+1, :], axis=0)
                halfwindow += 1
            # Middle region
            halfwindow = kernel_size // 2
            for i in range(halfwindow, num_obs - halfwindow):
                filtered[i, :] = np.nanmedian(array[i-halfwindow:i+halfwindow+1, :], axis=0)
            # Ending region
            halfwindow -= 1
            for i in range(num_obs - halfwindow, num_obs):
                filtered[i, :] = np.nanmedian(array[i-halfwindow:i+halfwindow+1, :], axis=0)
                halfwindow -= 1
    return filtered


@unwrap_dict_and_ts
def common_mode(array, method, num_components=1, plot=False):
    r"""
    Computes the Common Mode Error (CME) with the given method.
    The input array should already be a residual.

    Parameters
    ----------
    array : numpy.ndarray
        Input array of shape :math:`(\text{num_observations},\text{n_stations})`
        (can contain NaNs).
        Wrapped by :func:`~geonat.processing.unwrap_dict_and_ts` to also accept
        :class:`~geonat.timeseries.Timeseries`, :class:`~pandas.DataFrame` and
        dictionaries of them as input.
    method : {'pca', 'ica'}
        Method to use to decompose the array.
        ``'pca'`` uses `Principal Component Analysis`_ (motivated by [dong06]_), whereas
        ``'ica'`` uses `Independent Component Analysis`_ (motivated by [huang12]_).
    num_components : int, optional
        Number of CME bases to estimate. Defaults to ``1``.
    plot : bool, optional
        If ``True``, include not only the modeled CME, but also the components
        themselves in space and time. Defaults to ``False``.

    Returns
    -------
    model : numpy.ndarray
        Modeled CME of shape :math:`(\text{num_observations},\text{n_stations})`.
    temporal : numpy.ndarray
        (Only if ``plot=True``.) CME in time of shape
        :math:`(\text{num_observations},\text{num_components})`.
    spatial : numpy.ndarray
        (Only if ``plot=True``.) CME in space of shape
        :math:`(\text{num_components},\text{n_stations})`.


    References
    ----------

    .. [dong06] Dong, D., Fang, P., Bock, Y., Webb, F., Prawirodirdjo, L., Kedar,
       S., and Jamason, P. (2006), *Spatiotemporal filtering using principal component
       analysis and Karhunen‐Loeve expansion approaches for regional GPS network analysis*,
       J. Geophys. Res., 111, B03405, doi:`10.1029/2005JB003806
       <https://doi.org/10.1029/2005JB003806>`_.
    .. [huang12] Huang, D. W., Dai, W. J., & Luo, F. X. (2012),
       *ICA spatiotemporal filtering method and its application in GPS deformation monitoring*,
       Applied Mechanics and Materials, 204-208, 2806,
       doi:`10.4028/www.scientific.net/AMM.204-208.2806
       <http://dx.doi.org/10.4028/www.scientific.net/AMM.204-208.2806>`_.

    .. _Principal Component Analysis:
       https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    .. _Independent Component Analysis:
       https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html
    """
    # ignore all only-NaN columns
    array_nanind = np.isnan(array)
    finite_cols = np.nonzero(~array_nanind.all(axis=0))[0]
    nan_cols = np.nonzero(array_nanind.all(axis=0))[0]
    array = array[:, finite_cols]
    # fill NaNs with white Gaussian noise
    array_nanmean = np.nanmean(array, axis=0)
    array_nansd = np.nanstd(array, axis=0)
    array_nanind = np.isnan(array)
    for icol in range(array.shape[1]):
        array[array_nanind[:, icol], icol] = array_nansd[icol] \
                                             * np.random.randn(array_nanind[:, icol].sum()) \
                                             + array_nanmean[icol]
    # decompose using the specified solver
    if method == 'pca':
        decomposer = PCA(n_components=num_components, whiten=True)
    elif method == 'ica':
        decomposer = FastICA(n_components=num_components, whiten=True)
    else:
        raise NotImplementedError("Cannot estimate the common mode error "
                                  f"using the '{method}' method.")
    # extract temporal component and build model
    temporal = decomposer.fit_transform(array)
    model = decomposer.inverse_transform(temporal)
    # reduce to where original timeseries were not NaNs and return
    model[array_nanind] = np.NaN
    if nan_cols != []:
        newmod = np.empty((temporal.shape[0], len(finite_cols) + len(nan_cols)))
        newmod[:, finite_cols] = model
        newmod[:, nan_cols] = np.NaN
        model = newmod
    if plot:
        spatial = decomposer.components_
        if nan_cols != []:
            newspat = np.empty((spatial.shape[0], len(finite_cols) + len(nan_cols)))
            newspat[:, finite_cols] = spatial
            newspat[:, nan_cols] = np.NaN
            spatial = newspat
        return model, temporal, spatial
    else:
        return model


def clean(station, ts_in, reference, ts_out=None,
          clean_kw_args={}, reference_callable_args={}):
    """
    Function operating on a single station's timeseries to clean it from outliers,
    and mask it out if the data is not good enough. The criteria are set by
    :attr:`~geonat.config.defaults` but can be overriden by providing ``clean_kw_args``.

    Parameters
    ----------
    station : geonat.station.Station
        Station to operate on.
    ts_in : str
        Description of the timeseries to clean.
    reference : str, geonat.timeseries.Timeseries, function
        Reference timeseries.
        If string, checks for a timeseries with that description in the ``station``.
        If a :class:`~geonat.timeseries.Timeseries` instance, use it directly.
        If a function, the reference timeseries will be calculated as
        ``t_ref = reference(ts_in, **reference_callable_args)``.
    ts_out : str, optional
        If provided, duplicate ``ts_in`` to a new timeseries ``ts_out``
        and clean the copy (to preserve the raw timeseries).
    clean_kw_args : dict
        Override the default cleaning criteria in :attr:`~geonat.config.defaults`.
    reference_callable_args : dict
        If ``reference`` is a function, ``reference_callable_args`` can be used
        to pass additional keyword arguments to the former when calculating
        the reference timeseries.

    Warning
    -------
    By default, this function operates *in-place*. If you don't wish to overwrite
    the raw input timeseries, specify ``ts_out``.
    """
    clean_settings = defaults["clean"].copy()
    clean_settings.update(clean_kw_args)
    # check if we're modifying in-place or copying
    if ts_out is None:
        ts = station[ts_in]
    else:
        ts = station[ts_in].copy(only_data=True, src='clean')
    # check if we have a reference time series or need to calculate one
    # in the latter case, the input is name of function to call
    if not (isinstance(reference, Timeseries)
            or isinstance(reference, str)
            or callable(reference)):
        raise TypeError("'reference' has to either be a Timeseries, the name of one, "
                        f"or a function, got {type(reference)}.")
    if isinstance(reference, Timeseries):
        ts_ref = reference
    elif isinstance(reference, str):
        # get reference time series
        ts_ref = station[reference]
    elif callable(reference):
        ts_ref = reference(ts, **reference_callable_args)
    # check that both timeseries have the same data columns
    if not ts_ref.data_cols == ts.data_cols:
        raise ValueError("Reference time series has to have the same data columns as "
                         f"input time series, but got {ts_ref.data_cols} and {ts.data_cols}.")
    for dcol in ts.data_cols:
        # check for minimum number of observations
        if ts[dcol].count() < clean_settings["min_obs"]:
            ts.mask_out(dcol)
            continue
        # compute residuals
        if (clean_settings["std_outlier"] is not None) \
           or (clean_settings["std_thresh"] is not None):
            residual = ts[dcol].values - ts_ref[dcol].values
            sd = np.nanstd(residual)
        # check for and remove outliers
        if clean_settings["std_outlier"] is not None:
            mask = ~np.isnan(residual)
            mask[mask] &= np.abs(residual[mask]) > clean_settings["std_outlier"] * sd
            ts[dcol][mask] = np.NaN
            residual = ts[dcol].values - ts_ref[dcol].values
            sd = np.nanstd(residual)
        # check for minimum number of clean observations
        if ts[dcol].count() < clean_settings["min_clean_obs"]:
            ts.mask_out(dcol)
            continue
        # check if total standard deviation is still too large
        if (clean_settings["std_thresh"] is not None) and (sd > clean_settings["std_thresh"]):
            ts.mask_out(dcol)
    # if we made a copy, add it to the station, otherwise we're already done
    if ts_out is not None:
        station.add_timeseries(ts_out, ts)
