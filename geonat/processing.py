import numpy as np
import pandas as pd
import warnings
from functools import wraps
from sklearn.decomposition import PCA, FastICA

from . import defaults
from .compiled import maskedmedfilt2d
from .timeseries import Timeseries


def unwrap_dict_and_ts(func):
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
        has_additional_output = False if all([elem is None for elem in additional_output.values()]) else True
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
    Computes the median filter (ignoring NaNs) column-wise,
    either by calling a Numpy function iteratively or by using
    the precompiled Fortran code.
    """
    try:
        filtered = maskedmedfilt2d(array, ~np.isnan(array), kernel_size)
        filtered[np.isnan(array)] = np.NaN
    except BaseException:
        num_obs = array.shape[0]
        array = array.reshape(num_obs, 1 if array.ndim == 1 else -1)
        filtered = np.NaN * np.empty(array.shape)
        # Run filtering while suppressing warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered')
            # Beginning region
            halfWindow = 0
            for i in range(kernel_size // 2):
                filtered[i, :] = np.nanmedian(array[i-halfWindow:i+halfWindow+1, :], axis=0)
                halfWindow += 1
            # Middle region
            halfWindow = kernel_size // 2
            for i in range(halfWindow, num_obs - halfWindow):
                filtered[i, :] = np.nanmedian(array[i-halfWindow:i+halfWindow+1, :], axis=0)
            # Ending region
            halfWindow -= 1
            for i in range(num_obs - halfWindow, num_obs):
                filtered[i, :] = np.nanmedian(array[i-halfWindow:i+halfWindow+1, :], axis=0)
                halfWindow -= 1
    return filtered


@unwrap_dict_and_ts
def common_mode(array, method, n_components=1, plot=False):
    """
    Computes the common mode noise with the given method. Input should already be a residual.
    """
    # fill NaNs with white Gaussian noise
    array_nanmean = np.nanmean(array, axis=0)
    array_nansd = np.nanstd(array, axis=0)
    array_nanind = np.isnan(array)
    for icol in range(array.shape[1]):
        array[array_nanind[:, icol], icol] = array_nansd[icol] * np.random.randn(array_nanind[:, icol].sum()) + array_nanmean[icol]
    # decompose using the specified solver
    if method == 'pca':
        decomposer = PCA(n_components=n_components, whiten=True)
    elif method == 'ica':
        decomposer = FastICA(n_components=n_components, whiten=True)
    else:
        raise NotImplementedError(f"Cannot estimate the common mode error using the '{method}' method.")
    # extract temporal component and build model
    temporal = decomposer.fit_transform(array)
    model = decomposer.inverse_transform(temporal)
    # reduce to where original timeseries were not NaNs and return
    model[array_nanind] = np.NaN
    if plot:
        spatial = decomposer.components_
        return model, temporal, spatial
    else:
        return model


def clean(station, ts_in, reference, ts_out=None, clean_kw_args={}, reference_callable_args={}):
    clean_settings = defaults["clean"].copy()
    clean_settings.update(clean_kw_args)
    # check if we're modifying in-place or copying
    if ts_out is None:
        ts = station[ts_in]
    else:
        ts = station[ts_in].copy(only_data=True, src='clean')
    # check if we have a reference time series or need to calculate one
    # in the latter case, the input is name of function to call
    if not (isinstance(reference, Timeseries) or isinstance(reference, str) or callable(reference)):
        raise TypeError(f"'reference' has to either be a Timeseries, the name of one, or a function, got {type(reference)}.")
    if isinstance(reference, Timeseries):
        ts_ref = reference
    elif isinstance(reference, str):
        # get reference time series
        ts_ref = station[reference]
    elif callable(reference):
        ts_ref = reference(ts, **reference_callable_args)
    # check that both timeseries have the same data columns
    if not ts_ref.data_cols == ts.data_cols:
        raise ValueError(f"Reference time series has to have the same data columns as input time series, but got {ts_ref.data_cols} and {ts.data_cols}.")
    for dcol in ts.data_cols:
        # check for minimum number of observations
        if ts[dcol].count() < clean_settings["min_obs"]:
            ts.mask_out(dcol)
            continue
        # compute residuals
        if (clean_settings["std_outlier"] is not None) or (clean_settings["std_thresh"] is not None):
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
