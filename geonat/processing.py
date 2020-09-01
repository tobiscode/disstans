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
from scipy.signal import find_peaks

from .config import defaults
from .timeseries import Timeseries
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
        Wrapped by :func:`~geonat.processing.unwrap_dict_and_ts` to also accept
        :class:`~geonat.timeseries.Timeseries`, :class:`~pandas.DataFrame` and
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
    x : numpy.ndarray, optional
        Input array of shape :math:`(\text{num_observations},)`.
        Should not contain NaNs.
    y : numpy.ndarray, optional
        Input array of shape :math:`(\text{num_observations}, \text{num_components})`.
        Can contain NaNs.
    """
    def __init__(self, kernel_size=None, x=None, y=None):
        self.probabilities = None
        r"""
        Contains the probability array from the last :meth:`~search` function call.
        Has shape :math:`(\text{num_observations}, \text{num_components})`.
        """
        self.kernel_size = kernel_size
        if (x is not None) and (y is not None) and (kernel_size is not None):
            self.search(x, y, kernel_size)

    # @property
    # def x(self):
    #     r""" Hash of the ``x`` array (to check whether it changed). """
    #     if self._x is None:
    #         raise ValueError(f"'x' has not yet been set.")
    #     return self._x

    # @x.setter
    # def x(self, x):
    #     if x is not None:
    #         assert isinstance(x, np.ndarray) and (x.ndim == 1), \
    #             f"'x' must be a one-dimensional NumPy array."
    #         self._x = hash(x.tostring())
    #     else:
    #         self._x = None

    # @property
    # def y(self):
    #     r""" Hash of the ``y`` array (to check whether it changed). """
    #     if self._y is None:
    #         raise ValueError(f"'y' has not yet been set.")
    #     return self._x

    # @y.setter
    # def y(self, y):
    #     if y is not None:
    #         assert isinstance(y, np.ndarray) and (y.ndim == 2), \
    #             f"'y' must be a two-dimensional NumPy array."
    #         self._y = hash(y.tostring())
    #     else:
    #         self._y = None

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

        .. [burnhamanderson02] (2002) *Information and Likelihood Theory:*
           *A Basis for Model Selection and Inference.* In: Burnham K.P., Anderson D.R. (eds)
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
            otherwise the relative probability of H0 (which therefore can be 0 if H0 is also
            the best hypothesis in general).
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
            return 0, 0
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
            return 0, 0
        # now we can fit for H0, and again just go with that if there is no solution
        try:
            rss0 = float(np.linalg.lstsq(G0, yfinite, rcond=None)[1])
        except np.linalg.LinAlgError:
            return 0, 0
        # now that both models produce results, let's get the AIC_c values
        # we'll again return the H0 if not both models have a valid AIC_c value
        aic = [StepDetector.AIC_c(rss, n_total, dof) for (rss, dof)
               in zip([rss0, rss1], [3, 4])]
        if np.isnan(aic).sum() > 0:
            return 0, 0
        # let's check the difference between the two as a measure of evidence
        best_hyp = aic.index(min(aic))
        Delta_best = [a - aic[best_hyp] for a in aic]
        # we will only recommend H1 if it has the both the minimum AIC_c, and
        # the difference to H0 is larger than maxdel
        if (best_hyp == 1) and (Delta_best[0] > maxdel):
            return 1, Delta_best[0]
        else:
            return 0, Delta_best[best_hyp]

    def search(self, x, y, kernel_size=None, maxdel=10):
        r"""
        Function that will search for steps in the data.
        Upon successful completion, it will save the step probabilities in
        :attr:`~probabilities`.

        Parameters
        ----------
        x : numpy.ndarray
            Input array of shape :math:`(\text{num_observations},)`.
            Should not contain NaNs.
        y : numpy.ndarray
            Input array of shape :math:`(\text{num_observations}, \text{num_components})`.
            Can contain NaNs.
        kernel_size : int, optional
            Window size of the detector. Must be odd.
            If ``None``, use the previously set :attr:`~kernel_size`
        maxdel : float, optional
            Difference in AIC that should be considered not significantly better.
            (Refers to :math:`\Delta_i = \text{AIC}_{c,i} - \text{AIC}_{c,\text{min}}`.)
        """
        # some checks
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), \
            f"'x' and 'y' need to be NumPy arrays, got {type(x)} and {type(y)}."
        # check whether to update kernel_size
        if kernel_size is not None:
            self.kernel_size = kernel_size
        # get sizes
        num_observations = x.shape[0]
        y = y.reshape(num_observations, 1 if y.ndim == 1 else -1)
        num_components = y.shape[1]
        # get valid array
        valid = np.isfinite(y)
        # make output array
        probs = np.empty((num_observations, num_components))
        probs[:] = np.NaN
        # loop over all columns
        for icomp in range(num_components):
            # loop through all rows, starting with a shrunken kernel at the edges
            # Beginning region
            halfwindow = 0
            for i in range(self.kernel_size // 2):
                hyp, Del = StepDetector.test_single(x[i-halfwindow:i+halfwindow+1],
                                                    y[i-halfwindow:i+halfwindow+1, icomp],
                                                    valid[i-halfwindow:i+halfwindow+1, icomp],
                                                    maxdel=maxdel)
                if hyp == 1:
                    probs[i, icomp] = Del
                halfwindow += 1
            # Middle region
            assert halfwindow == self.kernel_size // 2
            for i in range(halfwindow, num_observations - halfwindow):
                hyp, Del = StepDetector.test_single(x[i-halfwindow:i+halfwindow+1],
                                                    y[i-halfwindow:i+halfwindow+1, icomp],
                                                    valid[i-halfwindow:i+halfwindow+1, icomp],
                                                    maxdel=maxdel)
                if hyp == 1:
                    probs[i, icomp] = Del
            # Ending region
            for i in range(num_observations - halfwindow, num_observations):
                halfwindow -= 1
                hyp, Del = StepDetector.test_single(x[i-halfwindow:i+halfwindow+1],
                                                    y[i-halfwindow:i+halfwindow+1, icomp],
                                                    valid[i-halfwindow:i+halfwindow+1, icomp],
                                                    maxdel=maxdel)
                if hyp == 1:
                    probs[i, icomp] = Del
        self.probabilities = probs

    def steps(self, threshold=2, maxsteps=np.inf, verbose=True):
        r"""
        Threshold the saved probabilities to return a list of steps.

        Parameters
        ----------
        threshold : float
            Minimum :math:`\Delta_i \geq 0` that needs to be satisfied in order to be a step.
        maxsteps : int, optional
            Return at most ``maxsteps`` number of steps. Can be useful if a good value for
            ``threshold`` has not been found yet.

        Returns
        -------
        list
            Returns a list of :class:`~numpy.ndarray` arrays that contain the indices of steps
            for each component.
        """
        assert self.probabilities is not None, \
            "'probabilities' has not been set yet, run StepDetector.search() first."
        probs = self.probabilities
        probs[np.isnan(probs)] = -1
        steps = []
        for icomp in range(probs.shape[1]):
            peaks, properties = find_peaks(probs[:, icomp], height=threshold)
            if peaks.size > maxsteps:
                largest_ix = np.argpartition(properties["peak_heights"], -maxsteps)[-maxsteps:]
                peaks = peaks[largest_ix]
                if verbose:
                    print(f"In order to return at most {maxsteps} steps, the threshold has "
                          f"been increased to {properties['peak_heights'][largest_ix].min()}.")
            if verbose and (peaks.size / probs.shape[0] > 0.1):
                warnings.warn(f"In component {icomp}, using threshold={threshold} leads to "
                              f"{peaks.size / probs.shape[0]:.2%} of timestamps being steps. "
                              "Consider setting a higher threshold.")
            steps.append(peaks)
        return steps
