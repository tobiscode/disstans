"""
This module contains some processing functions that while not belonging
to a specific class, require them to already be loaded and DISSTANS to
be initialized.

For general helper functions, see :mod:`~disstans.tools`.
"""

import numpy as np
import pandas as pd
import warnings
from functools import wraps
from sklearn.decomposition import PCA, FastICA
from scipy.signal import find_peaks
from tqdm import tqdm
from warnings import warn

from .config import defaults
from .timeseries import Timeseries
from .compiled import maskedmedfilt2d
from .tools import Timedelta, parallelize, tvec_to_numpycol


def unwrap_dict_and_ts(func):
    """
    A wrapper decorator that aims at simplifying the coding of processing functions.
    Ideally, a new function that doesn't need to know if its input is a
    :class:`~disstans.timeseries.Timeseries`, :class:`~pandas.DataFrame`,
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
    function :func:`~disstans.compiled.maskedmedfilt2d`.

    Parameters
    ----------
    array : numpy.ndarray
        2D input array (can contain NaNs).
        Wrapped by :func:`~disstans.processing.unwrap_dict_and_ts` to also accept
        :class:`~disstans.timeseries.Timeseries`, :class:`~pandas.DataFrame` and
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
            assert halfwindow == kernel_size // 2
            for i in range(halfwindow, num_obs - halfwindow):
                filtered[i, :] = np.nanmedian(array[i-halfwindow:i+halfwindow+1, :], axis=0)
            # Ending region
            for i in range(num_obs - halfwindow, num_obs):
                halfwindow -= 1
                filtered[i, :] = np.nanmedian(array[i-halfwindow:i+halfwindow+1, :], axis=0)
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
        Wrapped by :func:`~disstans.processing.unwrap_dict_and_ts` to also accept
        :class:`~disstans.timeseries.Timeseries`, :class:`~pandas.DataFrame` and
        dictionaries of them as input.
    method : str
        Method to use to decompose the array. Possible values are ``'pca'`` and ``'ica'}.
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

    .. [dong06] Dong, D., Fang, P., Bock, Y., Webb, F., Prawirodirdjo, L.,
       Kedar, S., & Jamason, P. (2006). *Spatiotemporal filtering using principal component
       analysis and Karhunen-Loeve expansion approaches for regional GPS network analysis*.
       Journal of Geophysical Research: Solid Earth, 111(B3).
       doi:`10.1029/2005JB003806 <https://doi.org/10.1029/2005JB003806>`_.
    .. [huang12] Huang, D. W., Dai, W. J., & Luo, F. X. (2012).
       *ICA Spatiotemporal Filtering Method and Its Application in GPS Deformation Monitoring*.
       Applied Mechanics and Materials, 204–208, 2806–2812.
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
    :attr:`~disstans.config.defaults` but can be overriden by providing ``clean_kw_args``.
    The criteria are:

    - ``'min_obs'``: Minimum number of observations the timeseries has to contain.
    - ``'std_outlier'``: Classify as an outlier any observation that is this many
      standard deviations away from the reference.
    - ``'std_thresh'``: After the removal of outliers, the maximum standard deviation
      that the residual between reference and input timeseries is allowed to have.
    - ``'min_clean_obs'``: After the removal of outliers, the minimum number of
      observations the timeseries has to contain.

    Parameters
    ----------
    station : disstans.station.Station
        Station to operate on.
    ts_in : str
        Description of the timeseries to clean.
    reference : str, disstans.timeseries.Timeseries, function
        Reference timeseries.
        If string, checks for a timeseries with that description in the ``station``.
        If a :class:`~disstans.timeseries.Timeseries` instance, use it directly.
        If a function, the reference timeseries will be calculated as
        ``t_ref = reference(ts_in, **reference_callable_args)``.
    ts_out : str, optional
        If provided, duplicate ``ts_in`` to a new timeseries ``ts_out``
        and clean the copy (to preserve the raw timeseries).
    clean_kw_args : dict, optional
        Override the default cleaning criteria in :attr:`~disstans.config.defaults`,
        see the explanations above.
    reference_callable_args : dict, optional
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
    # check if timeseries is present
    if ts_in not in station.timeseries:
        warn(f"Could not find timeseries '{ts_in}' in station {station.name}.",
             category=RuntimeWarning)
        return
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


class StepDetector():
    r"""
    This class implements a step detector based on the Akaike Information Criterion (AIC).

    A window is moved over the input data, and two linear models are fit in the
    method :meth:`~search`: one containing only a linear polynomial, and one containing an
    additional step in the middle of the window. Then, using the AIC, the relative
    probabilities are calculated, and saved for each timestep.

    In the final step, one can threshold these relative probabilities with the method
    :meth:`~steps`, and look for local maxima, which will correspond to probable steps.

    If the class is constructed with ``kernel_size``, ``x`` and ``y`` passed, it automatically
    calls its method :meth:`~search`, otherwise, :meth:`~search` needs to be called manually.
    Running the method again with a different ``kernel_size``, ``x`` or ``y`` will overwrite
    previous results.

    Parameters
    ----------
    kernel_size : int, optional
        Window size of the detector. Must be odd.
    kernel_size_min : int, optional
        Minimum window size of the detector (for edges). Must be smaller than or equal
        to ``kernel_size``. Defaults to ``0``.
    x : numpy.ndarray, optional
        Input array of shape :math:`(\text{num_observations},)`.
        Should not contain NaNs.
    y : numpy.ndarray, optional
        Input array of shape :math:`(\text{num_observations}, \text{num_components})`.
        Can contain NaNs.
    """
    def __init__(self, kernel_size=None, kernel_size_min=0, x=None, y=None):
        self.kernel_size = kernel_size
        self.kernel_size_min = kernel_size_min
        if (x is not None) and (y is not None) and (kernel_size is not None):
            self.search(x, y, kernel_size)

    @property
    def kernel_size(self):
        """ Kernel (window) size of the detector. """
        if self._kernel_size is None:
            raise ValueError("'kernel_size' has not yet been set.")
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_size):
        if kernel_size is not None:
            assert isinstance(kernel_size, int) and (kernel_size % 2 == 1), \
                f"'kernel_size' must be an odd integer or None, got {kernel_size}."
        self._kernel_size = kernel_size

    @property
    def kernel_size_min(self):
        """ Minimum kernel (window) size of the detector. """
        if self._kernel_size_min is None:
            raise ValueError("'kernel_size_min' has not yet been set.")
        return self._kernel_size_min

    @kernel_size_min.setter
    def kernel_size_min(self, kernel_size_min):
        if kernel_size_min is not None:
            assert kernel_size_min <= self.kernel_size, "'kernel_size_min' must be smaller " + \
                f"or equal to 'kernel_size', but {kernel_size_min} > {self.kernel_size}."
        self._kernel_size_min = kernel_size_min

    @staticmethod
    def AIC_c(rss, n, K):
        r"""
        Calculates the Akaike Information Criterion for small samples for Least Squares
        regression results. Implementation is based on [burnhamanderson02]_ (ch. 2).

        Parameters
        ----------
        rss : float
            Residual sum of squares.
        n : int
            Number of samples.
        K : int
            Degrees of freedom. If the Least Squares model has :math:`\text{num_parameters}`
            parameters (including the mean), then the degrees of freedom are
            :math:`K = \text{num_parameters} + 1`

        Returns
        -------
        float
            The Small Sample AIC for Least Squares :math:`\text{AIC}_c`.

        References
        ----------

        .. [burnhamanderson02] (2002) *Information and Likelihood Theory:
           A Basis for Model Selection and Inference.* In: Burnham K.P., Anderson D.R. (eds)
           Model Selection and Multimodel Inference. Springer, New York, NY.
           doi:`10.1007/978-0-387-22456-5_2 <https://doi.org/10.1007/978-0-387-22456-5_2>`_.
        """
        # input check
        if n - K - 1 <= 0:
            # can't return meaningful statistic, hypothesis unlikely
            return np.NaN
        # calculate AIC for LS
        AIC = n * np.log(rss / n) + 2*K
        # apply small sample size correction
        correction = 2 * K * (K + 1) / (n - K - 1)
        return AIC + correction

    @staticmethod
    def test_single(xwindow, ywindow, valid=None, maxdel=10):
        r"""
        For a single window (of arbitrary, but odd length), perform the AIC hypothesis test
        whether a step is likely present (H1) or not (H0) in the ``y`` data given
        ``x`` coordinates.

        Parameters
        ----------
        xwindow : numpy.ndarray
            Time array of shape :math:`(\text{num_window},)`.
            Should not contain NaNs.
        ywindow : numpy.ndarray
            Data array of shape :math:`(\text{num_window},)`.
            Should not contain NaNs.
        valid : numpy.ndarray, optional
            Mask array of the data of shape :math:`(\text{num_window},)`,
            with ``1`` where the ``ywindow`` is finite (not NaN or infinity).
            If not passed to the function, it is calculated internally, which will slow
            down the computation.
        maxdel : float, optional
            Difference in AIC that should be considered not significantly better.
            (Refers to :math:`\Delta_i = \text{AIC}_{c,i} - \text{AIC}_{c,\text{min}}`.)

        Returns
        -------
        int
            Best hypothesis (``0`` for no step, ``1`` for step).
        float
            If H1 is the best hypothesis (and suffices ``maxdel``), its relative probability,
            otherwise the relative probability of H0 (which therefore can be ``0`` if H0 is also
            the best hypothesis in general).
        tuple
            A 2-tuple of the two mean-squared residuals of the H0 and H1 hypotheses,
            respectively. Assuming the test is unbiased, this is the residual's variance.
            Is ``NaN`` in an element if the least-squares model did not converge.

        See Also
        --------
        AIC_c : For more information about the AIC hypothesis test.
        """
        # do some checks
        assert xwindow.shape[0] == ywindow.shape[0], \
            "'xwindow' and 'ywindow' have to have the same length in the first dimensions, " \
            f"got {xwindow.shape} and {ywindow.shape}."
        assert (xwindow.shape[0] % 2 == 1), \
            "'xwindow' and 'ywindow' must have an odd number of entries, " \
            f"got {xwindow.shape[0]}."
        if valid is None:
            valid = np.isfinite(ywindow)
        else:
            assert ywindow.shape == valid.shape, \
                "'ywindow' and 'valid' have to have the same shape, " \
                f"got {ywindow.shape} and {valid.shape}."
        # get number of valid observations
        i_mid = int(xwindow.shape[0] // 2)
        n_pre = valid[:i_mid].sum()
        n_post = valid[i_mid:].sum()
        n_total = n_pre + n_post
        # return with 0, 0 if we will not be able to get an estimate because of not enough data
        if (n_pre < 2) or (n_post < 2):
            return 0, 0, (np.NaN, np.NaN)
        xfinite = xwindow[valid]
        yfinite = ywindow[valid]
        # build mapping matrix for model H1
        G1 = np.zeros((n_total, 3))
        # first column is mean
        G1[:, 0] = 1
        # second column is slope
        G1[:, 1] = xfinite - xfinite[0]
        # third column is the additional step
        G1[n_pre:, 2] = 1
        # without the step it's just the first two columns
        G0 = G1[:, :2]
        # fit for H1 first
        # (since if that one doesn't converge, we have to go with H0 anyway)
        try:
            rss1 = float(np.linalg.lstsq(G1, yfinite, rcond=None)[1])
        except np.linalg.LinAlgError:
            return 0, 0, (np.NaN, np.NaN)
        # now we can fit for H0, and again just go with that if there is no solution
        try:
            rss0 = float(np.linalg.lstsq(G0, yfinite, rcond=None)[1])
        except np.linalg.LinAlgError:
            return 0, 0, (np.NaN, rss1/n_total)
        # now that both models produce results, let's get the AIC_c values
        # we'll again return the H0 if not both models have a valid AIC_c value
        aic = [StepDetector.AIC_c(rss, n_total, dof) for (rss, dof)
               in zip([rss0, rss1], [3, 4])]
        if np.isnan(aic).sum() > 0:
            return 0, 0, (rss0/n_total, rss1/n_total)
        # let's check the difference between the two as a measure of evidence
        best_hyp = aic.index(min(aic))
        Delta_best = [a - aic[best_hyp] for a in aic]
        # we will only recommend H1 if it has the both the minimum AIC_c, and
        # the difference to H0 is larger than maxdel
        if (best_hyp == 1) and (Delta_best[0] > maxdel):
            return 1, Delta_best[0], (rss0/n_total, rss1/n_total)
        else:
            return 0, Delta_best[best_hyp], (rss0/n_total, rss1/n_total)

    @staticmethod
    def _search(data_and_params):
        """
        Parallelizable part of the search, search_network and search_catalog methods.
        """
        if len(data_and_params) == 5:
            x, y, kernel_size, kernel_size_min, maxdel = data_and_params
            check_only = None
        elif len(data_and_params) == 6:
            # added the optional fifth parameter of which indices of x to check
            x, y, kernel_size, kernel_size_min, maxdel, check_only = data_and_params
        else:
            raise RuntimeError("Passed invalid 'data_and_params' argument: "
                               f"{data_and_params}")
        # some checks
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), \
            f"'x' and 'y' need to be NumPy arrays, got {type(x)} and {type(y)}."
        # get sizes
        num_observations = x.shape[0]
        y = y.reshape(num_observations, 1 if y.ndim == 1 else -1)
        num_components = y.shape[1]
        # get valid array
        valid = np.isfinite(y)
        # make output arrays
        probs = np.empty((num_observations, num_components))
        probs[:] = np.NaN
        var0, var1 = probs.copy(), probs.copy()
        # loop over all columns
        for icomp in range(num_components):
            # loop through all rows, starting with a shrunken kernel at the edges
            # Beginning region
            halfwindow = 0
            for i in range(kernel_size // 2):
                if (check_only and (i not in check_only)) or \
                   (halfwindow*2 + 1 < kernel_size_min):
                    halfwindow += 1
                    continue
                hyp, Del, (rss0, rss1) = \
                    StepDetector.test_single(x[i-halfwindow:i+halfwindow+1],
                                             y[i-halfwindow:i+halfwindow+1, icomp],
                                             valid[i-halfwindow:i+halfwindow+1, icomp],
                                             maxdel=maxdel)
                if hyp == 1:
                    probs[i, icomp] = Del
                var0[i, icomp], var1[i, icomp] = rss0, rss1
                halfwindow += 1
            # Middle region
            assert halfwindow == kernel_size // 2
            range_main = range(halfwindow, num_observations - halfwindow)
            if check_only:
                range_main = [i for i in range_main if i in check_only]
            for i in range_main:
                hyp, Del, (rss0, rss1) = \
                    StepDetector.test_single(x[i-halfwindow:i+halfwindow+1],
                                             y[i-halfwindow:i+halfwindow+1, icomp],
                                             valid[i-halfwindow:i+halfwindow+1, icomp],
                                             maxdel=maxdel)
                if hyp == 1:
                    probs[i, icomp] = Del
                var0[i, icomp], var1[i, icomp] = rss0, rss1
            # Ending region
            for i in range(num_observations - halfwindow, num_observations):
                halfwindow -= 1
                if (check_only and (i not in check_only)) or \
                   (halfwindow*2 + 1 < kernel_size_min):
                    continue
                hyp, Del, (rss0, rss1) = \
                    StepDetector.test_single(x[i-halfwindow:i+halfwindow+1],
                                             y[i-halfwindow:i+halfwindow+1, icomp],
                                             valid[i-halfwindow:i+halfwindow+1, icomp],
                                             maxdel=maxdel)
                if hyp == 1:
                    probs[i, icomp] = Del
                var0[i, icomp], var1[i, icomp] = rss0, rss1
        # return
        if check_only:
            probs = probs[check_only, :].reshape(-1, num_components)
            var0 = var0[check_only, :].reshape(-1, num_components)
            var1 = var1[check_only, :].reshape(-1, num_components)
        return probs, var0, var1

    def search(self, x, y, maxdel=10):
        r"""
        Function that will search for steps in the data.
        Upon successful completion, it will return the relative step probabilities
        as well as the residuals variances of the two hypotheses tested
        (as reported by :meth:`~test_single`).

        Parameters
        ----------
        x : numpy.ndarray
            Input array of shape :math:`(\text{num_observations},)`.
            Should not contain NaNs.
        y : numpy.ndarray
            Input array of shape :math:`(\text{num_observations}, \text{num_components})`.
            Can contain NaNs.
        maxdel : float, optional
            Difference in AIC that should be considered not significantly better.
            (Refers to :math:`\Delta_i = \text{AIC}_{c,i} - \text{AIC}_{c,\text{min}}`.)

        Returns
        -------
        probabilities : numpy.ndarray
            Contains the relative probabilities array.
            Has shape :math:`(\text{num_observations}, \text{num_components})`.
        var0 : numpy.ndarray
            Contains the array of the residuals variance of the hypothesis
            that no step is present.
            Has shape :math:`(\text{num_observations}, \text{num_components})`.
        var1 : numpy.ndarray
            Contains the array of the residuals variance of the hypothesis
            that a step is present.
            Has shape :math:`(\text{num_observations}, \text{num_components})`.

        See Also
        --------
        :meth:`~test_single` : For more explanations about the return values.
        """
        # call individual search function and store result
        return StepDetector._search((x, y, self.kernel_size,
                                     self.kernel_size_min, maxdel))

    def search_network(self, net, ts_description, maxdel=10, threshold=20,
                       gap=2, gap_unit="D"):
        r"""
        Function that searches for steps in an entire network (possibly in parallel),
        thresholds those probabilities, and identifies all the consecutive ranges in which
        steps happen over the network.

        Parameters
        ----------
        net : disstans.network.Network
            Network instance to operate on.
        ts_description : str
            :class:`~disstans.timeseries.Timeseries` description that will be analyzed.
        maxdel : float, optional
            Difference in AIC that should be considered not significantly better.
            (Refers to :math:`\Delta_i = \text{AIC}_{c,i} - \text{AIC}_{c,\text{min}}`.)
        threshold : float, optional
            Minimum :math:`\Delta_i \geq 0` that needs to be satisfied in order to be a step.
        gap : float, optional
            Maximum gap between identified steps to count as a continuous period
            of possible steps.
        gap_unit : str, optional
            Time unit of ``gap``.

        Returns
        -------
        step_table : pandas.DataFrame
            A DataFrame containing the columns ``'station'`` (its name), ``'time'``
            (a timestamp of the station) and ``'probability'`` (maximum :math:`\Delta_i`
            over all components for this timestamp), as well as ``var0``, ``var1``
            (the two hypotheses' residuals variances for the component of
            maximum step probability) and ``varred`` (the variance reduction in percent,
            ``(var0 - var1) / var0``).
        step_ranges : list
            A list of lists containing continuous periods over all stations of the potential
            steps as determined by ``gap`` and ``gap_unit``.
        """
        # initialize DataFrame
        step_table = pd.DataFrame(columns=["station", "time", "probability"])
        # get the stations who have this timeseries
        valid_stations = {name: station for name, station in net.stations.items()
                          if ts_description in station.timeseries}
        # run parallelized StepDetector._search
        iterable_input = ((tvec_to_numpycol(station[ts_description].time),
                           station[ts_description].data.values,
                           self.kernel_size, self.kernel_size_min, maxdel)
                          for station in valid_stations.values())
        for name, station, (probs, var0, var1) in \
            zip(valid_stations.keys(), valid_stations.values(),
                tqdm(parallelize(StepDetector._search, iterable_input), ascii=True,
                     total=len(valid_stations), unit="station", desc="Searching for steps")):
            # find steps given the just calculated probabilities
            # setting the maximum number of steps to infinite to not miss anything
            steps = StepDetector.steps(probs, threshold, np.inf, False)
            # combine all data components and keep largest probability if the step
            # is present in multiple components
            unique_steps = np.sort(np.unique(np.concatenate(steps)))
            stepprobs = probs[unique_steps, :]
            stepsvar0, stepsvar1 = var0[unique_steps, :], var1[unique_steps, :]
            maxprobindices = np.expand_dims(np.argmax(stepprobs, axis=1), axis=1)
            maxstepprobs = np.take_along_axis(stepprobs, maxprobindices, axis=1).squeeze()
            maxstepvar0 = np.take_along_axis(stepsvar0, maxprobindices, axis=1).squeeze()
            maxstepvar1 = np.take_along_axis(stepsvar1, maxprobindices, axis=1).squeeze()
            # isolate the actual timestamps and add to the DataFrame
            steptimes = station[ts_description].time[unique_steps]
            step_table = step_table.append(pd.DataFrame({"station": [name]*len(steptimes),
                                                         "time": steptimes,
                                                         "probability": maxstepprobs,
                                                         "var0": maxstepvar0,
                                                         "var1": maxstepvar1}),
                                           ignore_index=True)
            # this code could be used to create a model object and assign it to the station
            # mdl = disstans.models.Step(steptimes)
            # station.add_local_model(ts_description, "Detections", mdl)
        # sort dataframe by probability
        step_table.sort_values(by="probability", ascending=False, inplace=True)
        # get coefficient of partial determination, i.e. how much the variance is reduced
        # (in percent) by including a step
        step_table["varred"] = (step_table["var0"] - step_table["var1"]) / step_table["var0"]
        # get the consecutive steptime ranges
        if not step_table.empty:
            unique_steps = np.sort(step_table["time"].unique())
            split = np.nonzero((np.diff(unique_steps) / Timedelta(1, gap_unit)) > gap)[0]
            split = np.concatenate([0, split + 1], axis=None)
            step_ranges = [unique_steps[split[i]:split[i + 1]] for i in range(split.size - 1)]
        else:
            step_ranges = []
        return step_table, step_ranges

    def search_catalog(self, net, ts_description, catalog, threshold=None,
                       gap=2, gap_unit="D", keep_nan_probs=True):
        r"""
        Search a dictionary of potential step times for each station in the dictionary
        and assess the probability for each one.

        Parameters
        ----------
        net : disstans.network.Network
            Network instance to operate on.
        ts_description : str
            :class:`~disstans.timeseries.Timeseries` description that will be analyzed.
        catalog : dict, pandas.DataFrame
            Dictionary where each key is a station name and its value is a list of
            :class:`~pandas.Timestamp` compatible potential times/dates.
            Alternatively, a DataFrame with at least the columns ``'station'`` and ``'time'``.
        threshold : float, optional
            Minimum :math:`\Delta_i \geq 0` that needs to be satisfied in order to be a step.
        gap : float, optional
            Maximum gap between identified steps to count as a continuous period
            of possible steps.
        gap_unit : str, optional
            Time unit of ``gap``.
        keep_nan_probs : bool, optional
            (Only applies to a DataFrame-type ``catalog`` input.)
            If a catalogued station is not in the network, or if a catalogued timestamp
            is after the available timeseries, no step probability can be calculated
            and the results will contain NaNs.
            If ``True`` (default), those entries will be kept in the output, and if
            ``False``, they will be dropped.

        Returns
        -------
        step_table : pandas.DataFrame
            A DataFrame containing the columns ``'station'`` (its name), ``'time'``
            (a timestamp of the station) and ``'probability'`` (maximum :math:`\Delta_i`
            over all components for this timestamp) for each potential step in ``catalog``,
            as well as ``var0`` and ``var1`` (the two hypotheses' residuals variances
            for the component of maximum step probability).
            If a DataFrame was passed as ``catalog``, a copy of that will be returned, with
            the added columns specified above.
        step_ranges : list
            A list of lists containing continuous periods over all stations of the potential
            steps as determined by ``gap`` and ``gap_unit``.
        """
        # hard-code maxdel to not filter out any item since we are asking about specific times
        maxdel = 0
        # get a simple dictionary representation if catalog was passed as dictionary
        # and set keep_nan_probs if not specified
        if isinstance(catalog, pd.DataFrame):
            assert all([col in catalog.columns for col in ["station", "time"]]), \
                "Invalid input 'catalog' DataFrame columns."
            catalog_df = catalog
            catalog = dict(catalog_df.groupby("station")["time"].apply(list))
            augment_df = True
        else:
            assert isinstance(catalog, dict), \
                "'catalog' must be either a dictionary or DataFrame."
            augment_df = False
        # initialize DataFrame
        step_table = pd.DataFrame(columns=["station", "time", "probability"])
        # for each station, find the first time index after a catalogued event
        # (alternatively, we could "add" a timestamp without an observation if
        # there isn't a timestamp already present - probably better, but harder)
        check_indices = {}
        # we also need to keep track of the originally requested time (for the output)
        catalog_timeexists = {sta_name: [False] * len(steptimes)
                              for sta_name, steptimes in catalog.items()}
        for sta_name, steptimes in catalog.items():
            # skip if station or timeseries not present
            if (sta_name not in net.stations.keys()) or \
               (ts_description not in net[sta_name].timeseries):
                continue
            check_indices[sta_name] = []
            for ist, st in enumerate(steptimes):
                index_after = (net[sta_name][ts_description].time >= st).tolist()
                try:
                    check_indices[sta_name].append(index_after.index(True))
                except ValueError:  # catalogued time is after the last timestamp
                    continue
                else:
                    catalog_timeexists[sta_name][ist] = True
        # run parallelized StepDetector._search
        stations_overlap = list(check_indices.keys())
        iterable_input = ((tvec_to_numpycol(net[sta_name][ts_description].time),
                           net[sta_name][ts_description].data.values,
                           self.kernel_size, self.kernel_size_min,
                           maxdel, check_indices[sta_name])
                          for sta_name in stations_overlap)
        results_iterator = tqdm(parallelize(StepDetector._search, iterable_input),
                                ascii=True, total=len(stations_overlap), unit="station",
                                desc="Searching for steps")
        for name, (probs, var0, var1) in zip(stations_overlap, results_iterator):
            # probs now contains a row for each catalog item
            # if the probability is NaN, AIC does not see evidence for a step,
            # if it is a float, then that's the likelihood of a step (always positive)
            # var0 and var1 can contain values regardless of the entry in probs
            has_steps = np.any(~np.isnan(probs), axis=1)
            maxstepprobs = probs[:, 0]
            maxstepvar0, maxstepvar1 = var0[:, 0], var1[:, 0]
            # maxstepprobs[has_steps] = np.nanmax(probs[has_steps, :], axis=1)
            maxprobindices = np.expand_dims(np.nanargmax(probs[has_steps, :], axis=1), axis=1)
            maxstepprobs[has_steps], maxstepvar0[has_steps], maxstepvar1[has_steps] = \
                np.take_along_axis(probs[has_steps, :], maxprobindices, axis=1).squeeze(), \
                np.take_along_axis(var0[has_steps, :], maxprobindices, axis=1).squeeze(), \
                np.take_along_axis(var1[has_steps, :], maxprobindices, axis=1).squeeze()
            # isolate the original timestamps and add to the DataFrame
            steptimes = [origtime for i, origtime in enumerate(catalog[name])
                         if catalog_timeexists[name][i]]
            step_table = step_table.append(pd.DataFrame({"station": [name]*len(steptimes),
                                                         "time": steptimes,
                                                         "probability": maxstepprobs,
                                                         "var0": maxstepvar0,
                                                         "var1": maxstepvar1}),
                                           ignore_index=True)
        # merge it with the input dataframe, if provided
        if augment_df:
            catalog_df["probability"] = np.NaN
            catalog_df["var0"] = np.NaN
            catalog_df["var1"] = np.NaN
            for _, row in step_table.iterrows():
                row_location = (catalog_df["station"] == row["station"]) & \
                               (catalog_df["time"] == row["time"])
                catalog_df.loc[row_location, ["probability", "var0", "var1"]] = \
                    row[["probability", "var0", "var1"]].values
            if not keep_nan_probs:
                catalog_df = catalog_df.dropna(how="all", subset=["probability", "var0", "var1"])
            step_table = catalog_df
        # sort
        step_table.sort_values(by="probability", ascending=False, inplace=True)
        # get coefficient of partial determination, i.e. how much the variance is reduced
        # (in percent) by including a step
        step_table["varred"] = (step_table["var0"] - step_table["var1"]) / step_table["var0"]
        # get the consecutive steptime ranges
        unique_steps = np.sort(step_table["time"].unique())
        split = np.nonzero((np.diff(unique_steps) / Timedelta(1, gap_unit)) > gap)[0]
        split = np.concatenate([0, split + 1], axis=None)
        step_ranges = [unique_steps[split[i]:split[i + 1]] for i in range(split.size - 1)]
        return step_table, step_ranges

    @staticmethod
    def steps(probabilities, threshold=2, maxsteps=np.inf, verbose=True):
        r"""
        Threshold the probabilities to return a list of steps.

        Parameters
        ----------
        threshold : float, optional
            Minimum :math:`\Delta_i \geq 0` that needs to be satisfied in order to be a step.
        maxsteps : int, optional
            Return at most ``maxsteps`` number of steps. Can be useful if a good value for
            ``threshold`` has not been found yet.
        verbose : bool, optional
            If ``True`` (default), print warnings when there will be a large number of
            steps identified given the ``threshold``.

        Returns
        -------
        steps : list
            Returns a list of :class:`~numpy.ndarray` arrays that contain the indices of steps
            for each component.
        """
        # initialize
        probabilities[np.isnan(probabilities)] = -1
        steps = []
        for icomp in range(probabilities.shape[1]):
            # find peaks above the threshold
            peaks, properties = find_peaks(probabilities[:, icomp], height=threshold)
            # if maxsteps is set, reduce steps to that number
            if peaks.size > maxsteps:
                largest_ix = np.argpartition(properties["peak_heights"], -maxsteps)[-maxsteps:]
                peaks = peaks[largest_ix]
                # warn about the new effective threshold
                if verbose:
                    print(f"In order to return at most {maxsteps} steps, the threshold has "
                          f"been increased to {properties['peak_heights'][largest_ix].min()}.")
            # warn if a large number of steps have been detected
            if verbose and (peaks.size / probabilities.shape[0] > 0.1):
                warnings.warn(f"In component {icomp}, using threshold={threshold} leads to "
                              f"{peaks.size / probabilities.shape[0]:.2%} of timestamps "
                              "being steps. Consider setting a higher threshold.")
            steps.append(peaks)
        return steps
