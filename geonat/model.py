import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from warnings import warn
from scipy.special import comb, factorial
from itertools import product

from . import scm
from .tools import tvec_to_numpycol


class Model():
    """
    General class that defines what a model can have as an input and output.
    Defaults to a linear model.
    """
    def __init__(self, num_parameters, regularize=False, time_unit=None, t_start=None, t_end=None, t_reference=None, zero_before=True, zero_after=True):
        self.num_parameters = int(num_parameters)
        assert self.num_parameters > 0, f"'num_parameters' must be an integer greater or equal to one, got {self.num_parameters}."
        self.is_fitted = False
        self.parameters = None
        self.cov = None
        self.regularize = bool(regularize)
        self.time_unit = str(time_unit)
        self.t_start_str = None if t_start is None else str(t_start)
        self.t_end_str = None if t_start is None else str(t_end)
        self.t_reference_str = None if t_start is None else str(t_reference)
        self.t_start = None if t_start is None else pd.Timestamp(t_start)
        self.t_end = None if t_end is None else pd.Timestamp(t_end)
        self.t_reference = None if t_reference is None else pd.Timestamp(t_reference)
        self.zero_before = bool(zero_before)
        self.zero_after = bool(zero_after)

    def get_arch(self):
        # make base architecture
        arch = {"type": "Model",
                "num_parameters": self.num_parameters,
                "kw_args": {"regularize": self.regularize,
                            "time_unit": self.time_unit,
                            "t_start": self.t_start_str,
                            "t_end": self.t_end_str,
                            "t_reference": self.t_reference_str,
                            "zero_before": self.zero_before,
                            "zero_after": self.zero_after}}
        # get subclass-specific architecture
        instance_arch = self._get_arch()
        # update non-dictionary values
        arch.update({arg: value for arg, value in instance_arch.items() if arg != "kw_args"})
        # update keyword dictionary
        arch["kw_args"].update(instance_arch["kw_args"])
        return arch

    def _get_arch(self):
        raise NotImplementedError("Instantiated model was not subclassed or it does not overwrite the '_get_arch' method.")

    def get_mapping(self, timevector):
        # get active period and initialize coefficient matrix
        active, first, last = self._get_active_period(timevector)
        # if there isn't any active period, return csr-sparse matrix
        if (first is None) and (last is None):  # this is equivalent to not active.any()
            mapping = sparse.bsr_matrix((timevector.size, self.num_parameters))
        # otherwise, build coefficient matrix
        else:
            # build dense sub-matrix
            coefs = self._get_mapping(timevector[active])
            assert coefs.shape[1] == self.num_parameters, \
                f"The child function '_get_mapping' of model {type(self).__name__} returned an invalid shape. " \
                f"Expected was ({last-first+1}, {self.num_parameters}), got {coefs.shape}."
            # build before- and after-matrices
            # either use zeros or the values at the active boundaries for padding
            if self.zero_before:
                before = sparse.csr_matrix((first, self.num_parameters))
            else:
                before = sparse.csr_matrix(np.ones((first, self.num_parameters)) * coefs[0, :].reshape(1, -1))
            if self.zero_after:
                after = sparse.csr_matrix((timevector.size - last - 1, self.num_parameters))
            else:
                after = sparse.csr_matrix(np.ones((timevector.size - last - 1, self.num_parameters)) * coefs[-1, :].reshape(1, -1))
            # stack them (they can have 0 in the first dimension, no problem for sparse.vstack)
            # I think it's faster if to stack them if they're all already csr format
            mapping = sparse.vstack((before, sparse.csr_matrix(coefs), after), format='bsr')
        return mapping

    def _get_mapping(self, timevector):
        raise NotImplementedError("'Model' needs to be subclassed and its child needs to implement a '_get_mapping' function for the active period.")

    def _get_active_period(self, timevector):
        if (self.t_start is None) and (self.t_end is None):
            active = np.ones_like(timevector, dtype=bool)
        elif self.t_start is None:
            active = timevector <= self.t_end
        elif self.t_end is None:
            active = timevector >= self.t_start
        else:
            active = np.all((timevector >= self.t_start, timevector <= self.t_end), axis=0)
        if active.any():
            first, last = int(np.argwhere(active)[0]), int(np.argwhere(active)[-1])
        else:
            first, last = None, None
        return active, first, last

    def tvec_to_numpycol(self, timevector):
        """ Convenience wrapper for tvec_to_numpycol for Model objects that have self.time_unit and self.t_reference attributes. """
        if self.t_reference is None:
            raise ValueError("Can't call 'tvec_to_numpycol' because no reference time was specified in the model.")
        if self.time_unit is None:
            raise ValueError("Can't call 'tvec_to_numpycol' because no time unit was specified in the model.")
        return tvec_to_numpycol(timevector, self.t_reference, self.time_unit)

    def read_parameters(self, parameters, cov):
        assert self.num_parameters == parameters.shape[0], "Read-in parameters have different size than the instantiated model."
        self.parameters = parameters
        if cov is not None:
            assert self.num_parameters == cov.shape[0] == cov.shape[1], "Covariance matrix must have same number of entries than parameters."
            self.cov = cov
        self.is_fitted = True

    def evaluate(self, timevector):
        if not self.is_fitted:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        mapping_matrix = self.get_mapping(timevector=timevector)
        fit = mapping_matrix @ self.parameters
        fit_sigma = mapping_matrix @ np.sqrt(self.cov.diagonal(offset=0, axis1=0, axis2=1).T) if self.cov is not None else None
        return {"time": timevector, "fit": fit, "sigma": fit_sigma}


class Step(Model):
    """
    Step functions at given times.
    """
    def __init__(self, steptimes, zero_after=False, **model_kw_args):
        super().__init__(num_parameters=len(steptimes), zero_after=zero_after, **model_kw_args)
        self.steptimes_str = steptimes
        self.timestamps = [pd.Timestamp(step) for step in self.steptimes_str]
        self.timestamps.sort()

    def _get_arch(self):
        arch = {"type": "Step",
                "kw_args": {"steptimes": self.steptimes_str}}
        return arch

    def _update_from_steptimes(self):
        self.timestamps = [pd.Timestamp(step) for step in self.steptimes_str]
        self.timestamps.sort()
        self.num_parameters = len(self.timestamps)
        self.is_fitted = False
        self.parameters = None
        self.cov = None

    def add_step(self, step):
        if step in self.steptimes_str:
            warn(f"Step '{step}' already present.", category=RuntimeWarning)
        else:
            self.steptimes_str.append(step)
            self._update_from_steptimes()

    def remove_step(self, step):
        try:
            self.steptimes_str.remove(step)
            self._update_from_steptimes()
        except ValueError:
            warn(f"Step '{step}' not present.", category=RuntimeWarning)

    def _get_mapping(self, timevector):
        coefs = np.array(timevector.values.reshape(-1, 1) >= pd.DataFrame(data=self.timestamps, columns=["steptime"]).values.reshape(1, -1), dtype=float)
        return coefs


class Polynomial(Model):
    """
    Polynomial of given order.

    `time_unit` can be the following (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html):
        `W`, `D`, `days`, `day`, `hours`, `hour`, `hr`, `h`, `m`, `minute`, `min`, `minutes`, `T`,
        `S`, `seconds`, `sec`, `second`, `ms`, `milliseconds`, `millisecond`, `milli`, `millis`, `L`,
        `us`, `microseconds`, `microsecond`, `micro`, `micros`, `U`, `ns`, `nanoseconds`, `nano`, `nanos`, `nanosecond`, `N`
    """
    def __init__(self, order, zero_before=False, zero_after=False, **model_kw_args):
        super().__init__(num_parameters=order + 1, zero_before=zero_before, zero_after=zero_after, **model_kw_args)
        self.order = int(order)

    def _get_arch(self):
        arch = {"type": "Polynomial",
                "kw_args": {"order": self.order}}
        return arch

    def _get_mapping(self, timevector):
        coefs = np.ones((timevector.size, self.num_parameters))
        if self.order >= 1:
            # now we actually need the time
            dt = self.tvec_to_numpycol(timevector)
            # the exponents increase by column
            exponents = np.arange(1, self.order + 1)
            # broadcast to all coefficients
            coefs[:, 1:] = dt.reshape(-1, 1) ** exponents.reshape(1, -1)
        return coefs


class BSpline(Model):
    r"""
    Cardinal, centralized B-Splines of certain order/degree and time scale.
    Used for transient temporary signals that return to zero after a given time span.

    Compare the analytic representation of the B-Splines:
    Butzer, P., Schmidt, M., & Stark, E. (1988). Observations on the History of Central B-Splines.
    Archive for History of Exact Sciences, 39(2), 137-156. Retrieved May 14, 2020, from https://www.jstor.org/stable/41133848
    or
    Schoenberg, I. J. (1973). Cardinal Spline Interpolation.
    Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9781611970555
    and some examples on https://bsplines.org/flavors-and-types-of-b-splines/.

    It is important to note that the function will be non-zero on the interval
    \( -(p+1)/2 < x < (p+1)/2 \)
    where p is the degree of the cardinal B-spline (and the degree of the resulting polynomial).
    The order n is related to the degree by the relation n = p + 1.
    The scale determines the width of the spline in the time domain, and corresponds to the interval [0, 1] of the B-Spline.
    The full non-zero time span of the spline is therefore scale * (p+1) = scale * n.

    num_splines will increase the number of splines by shifting the reference point (num_splines - 1)
    times by the spacing (which must be given in the same units as the scale).

    If no spacing is given but multiple splines are requested, the scale will be used as the spacing.

    `time_unit` can be the following (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html):
        `W`, `D`, `days`, `day`, `hours`, `hour`, `hr`, `h`, `m`, `minute`, `min`, `minutes`, `T`,
        `S`, `seconds`, `sec`, `second`, `ms`, `milliseconds`, `millisecond`, `milli`, `millis`, `L`,
        `us`, `microseconds`, `microsecond`, `micro`, `micros`, `U`, `ns`, `nanoseconds`, `nano`, `nanos`, `nanosecond`, `N`
    """
    def __init__(self, degree, scale, t_reference, time_unit, num_splines=1, spacing=None, **model_kw_args):
        self.degree = int(degree)
        self.order = self.degree + 1
        self.scale = float(scale)
        if spacing is not None:
            self.spacing = float(spacing)
            assert abs(self.spacing) > 0, f"'spacing' must be non-zero to avoid singularities, got {self.spacing}."
            if num_splines == 1:
                warn(f"'spacing' ({self.spacing} {time_unit}) is given, but 'num_splines' = 1 splines are requested.")
        elif num_splines > 1:
            self.spacing = self.scale
        else:
            self.spacing = None
        if "t_start" not in model_kw_args or model_kw_args["t_start"] is None:
            model_kw_args["t_start"] = (pd.Timestamp(t_reference) - pd.Timedelta(self.scale, time_unit) * (self.degree + 1)/2).isoformat()
        if "t_end" not in model_kw_args or model_kw_args["t_end"] is None:
            model_kw_args["t_end"] = (pd.Timestamp(t_reference)
                                      + pd.Timedelta(self.spacing, time_unit) * num_splines
                                      + pd.Timedelta(self.scale, time_unit) * (self.degree + 1)/2).isoformat()
        super().__init__(num_parameters=num_splines, t_reference=t_reference, time_unit=time_unit, **model_kw_args)

    def _get_arch(self):
        arch = {"type": "BSpline",
                "kw_args": {"degree": self.degree,
                            "scale": self.scale,
                            "num_splines": self.num_parameters,
                            "spacing": self.spacing}}
        return arch

    def _get_mapping(self, timevector):
        trel = self.tvec_to_numpycol(timevector).reshape(-1, 1, 1) - self.scale * np.arange(self.num_parameters).reshape(1, -1, 1)
        tnorm = trel / self.scale
        krange = np.arange(self.order + 1).reshape(1, 1, -1)
        in_power = tnorm + self.order/2 - krange
        in_sum = (-1)**krange * comb(self.order, krange) * (in_power)**(self.degree) * (in_power >= 0)
        coefs = np.sum(in_sum, axis=2) / factorial(self.degree)
        return coefs

    def get_transient_period(self, timevector):
        trel = self.tvec_to_numpycol(timevector).reshape(-1, 1) - self.scale * np.arange(self.num_parameters).reshape(1, -1)
        transient = np.abs(trel) <= self.scale * self.order
        return transient


class ISpline(Model):
    """
    Integral of cardinal, centralized B-Splines of certain order/degree and time scale.
    The degree p given in the initialization is the degree of the spline *before* the integration, i.e.
    the resulting IBSpline is a piecewise polynomial of degree p + 1.
    Used for transient permanent signals that stay at their maximum value after a given time span.

    See the full documentation in the BSpline class.
    """
    def __init__(self, degree, scale, t_reference, time_unit, num_splines=1, spacing=None, zero_after=False, **model_kw_args):
        self.degree = int(degree)
        self.order = self.degree + 1
        self.scale = float(scale)
        if spacing is not None:
            self.spacing = float(spacing)
            assert abs(self.spacing) > 0, f"'spacing' must be non-zero to avoid singularities, got {self.spacing}."
            if num_splines == 1:
                warn(f"'spacing' ({self.spacing} {time_unit}) is given, but 'num_splines' = 1 splines are requested.")
        elif num_splines > 1:
            self.spacing = self.scale
        else:
            self.spacing = None
        if "t_start" not in model_kw_args or model_kw_args["t_start"] is None:
            model_kw_args["t_start"] = (pd.Timestamp(t_reference) - pd.Timedelta(self.scale, time_unit) * (self.degree + 1)/2).isoformat()
        if "t_end" not in model_kw_args or model_kw_args["t_end"] is None:
            model_kw_args["t_end"] = (pd.Timestamp(t_reference)
                                      + pd.Timedelta(self.spacing, time_unit) * num_splines
                                      + pd.Timedelta(self.scale, time_unit) * (self.degree + 1)/2).isoformat()
        super().__init__(num_parameters=num_splines, t_reference=t_reference, time_unit=time_unit, zero_after=zero_after, **model_kw_args)

    def _get_arch(self):
        arch = {"type": "ISpline",
                "kw_args": {"degree": self.degree,
                            "scale": self.scale,
                            "num_splines": self.num_parameters,
                            "spacing": self.spacing}}
        return arch

    def _get_mapping(self, timevector):
        trel = self.tvec_to_numpycol(timevector).reshape(-1, 1, 1) - self.scale * np.arange(self.num_parameters).reshape(1, -1, 1)
        tnorm = trel / self.scale
        krange = np.arange(self.order + 1).reshape(1, 1, -1)
        in_power = tnorm + self.order/2 - krange
        in_sum = (-1)**krange * comb(self.order, krange) * (in_power)**(self.degree + 1) * (in_power >= 0)
        coefs = np.sum(in_sum, axis=2) / factorial(self.degree + 1)
        return coefs

    def get_transient_period(self, timevector):
        trel = self.tvec_to_numpycol(timevector).reshape(-1, 1) - self.scale * np.arange(self.num_parameters).reshape(1, -1)
        transient = np.abs(trel) <= self.scale * self.order
        return transient


class SplineSet(Model):
    """
    Contains a list of splines that share a common degree, but different center times and scales.

    The set is constructed from a time span (t_center_start and t_center_end) and numbers of centerpoints or length scales.
    The number of splines for each scale will then be chosen such that the resulting set of splines will be complete.
    This means it will contain all splines that are non-zero at least somewhere in the time span.

    This class also sets the spacing equal to the scale.
    """
    def __init__(self, degree, t_center_start, t_center_end, time_unit, splineclass=ISpline,
                 list_scales=None, list_num_knots=None, complete=True, **model_kw_args):
        assert np.logical_xor(list_scales is None, list_num_knots is None), \
            f"To construct a set of Splines, only pass one of 'list_scales' and 'list_num_knots' " \
            f"(got {list_scales} and {list_num_knots})."
        relevant_list = list_scales if list_num_knots is None else list_num_knots
        try:
            if isinstance(splineclass, str):
                splineclass = globals()[splineclass]
            assert issubclass(splineclass, Model)
        except BaseException as e:
            raise LookupError(f"When trying to create the SplineSet, couldn't find the model '{splineclass}' "
                              "(expected Model type argument or string representation of a loaded Model).").with_traceback(e.__traceback__) from e
        # get time range
        t_center_start_tstamp, t_center_end_tstamp = pd.Timestamp(t_center_start), pd.Timestamp(t_center_end)
        t_range_tdelta = t_center_end_tstamp - t_center_start_tstamp
        # if a complete set is requested, we need to find the number of overlaps given the degree on a single side
        num_overlaps = degree if complete else 0
        # for each scale, make a BSplines object
        splset = []
        num_parameters = 0
        for elem in relevant_list:
            # Calculate the scale as float and Timedelta depending on the function call
            if list_scales is not None:
                scale_float = elem
                scale_tdelta = pd.Timedelta(scale_float, time_unit)
            else:
                scale_tdelta = t_range_tdelta / elem
                scale_float = scale_tdelta / pd.to_timedelta(1, time_unit)
            # find the number of center points between t_center_start and t_center_end, plus the overlapping ones
            num_centerpoints = int(t_range_tdelta / scale_tdelta) + 1 + 2*num_overlaps
            num_parameters += num_centerpoints
            # shift the reference to be the first spline
            t_ref = t_center_start_tstamp - num_overlaps*scale_tdelta
            # create model and append
            splset.append(splineclass(degree, scale_float, num_splines=num_centerpoints, t_reference=t_ref, time_unit=time_unit))
        # create the actual Model object
        super().__init__(num_parameters=num_parameters, time_unit=time_unit,
                         zero_after=False if splineclass == ISpline else True, **model_kw_args)
        self.degree = degree
        self.t_center_start = t_center_start
        self.t_center_end = t_center_end
        self.splineclass = splineclass
        self.list_scales = list_scales
        self.list_num_knots = list_num_knots
        self.complete = complete
        self.splines = splset

    def _get_arch(self):
        arch = {"type": "SplineSet",
                "kw_args": {"degree": self.degree,
                            "t_center_start": self.t_center_start,
                            "t_center_end": self.t_center_end,
                            "splineclass": self.splineclass.__name__,
                            "list_scales": self.list_scales,
                            "list_num_knots": self.list_num_knots,
                            "complete": self.complete}}
        return arch

    def _get_mapping(self, timevector):
        coefs = np.empty((timevector.size, self.num_parameters))
        ix_coefs = 0
        for i, model in enumerate(self.splines):
            temp = model.get_mapping(timevector).toarray().squeeze()
            coefs[:, ix_coefs:ix_coefs + model.num_parameters] = temp
            ix_coefs += model.num_parameters
        return coefs

    def read_parameters(self, parameters, cov):
        super().read_parameters(parameters, cov)
        ix_params = 0
        for i, model in enumerate(self.splines):
            cov_model = None if cov is None else cov[ix_params:ix_params + model.num_parameters, ix_params:ix_params + model.num_parameters]
            model.read_parameters(parameters[ix_params:ix_params + model.num_parameters], cov_model)
            ix_params += model.num_parameters

    def make_scalogram(self, t_left, t_right, cmaprange=None, resolution=1000):
        # check input
        assert self.is_fitted, f"SplineSet model needs to have already been fitted."
        # determine dimensions
        num_components = self.parameters.shape[1]
        num_scales = len(self.splines)
        dy_scale = 1/num_scales
        t_plot = pd.Series([tstamp for tstamp in pd.date_range(start=t_left, end=t_right, periods=resolution)])
        # get range of values (if not provided)
        if cmaprange is not None:
            assert isinstance(cmaprange, int) or isinstance(cmaprange, float), \
                f"'cmaprange' must be None or a single float or integer of the one-sided color range of the scalogram, got {cmaprange}."
        else:
            cmaprange = np.percentile(np.concatenate([np.abs(model.parameters) for model in self.splines], axis=0).ravel(), 95)
        cmap = mpl.cm.ScalarMappable(cmap=scm.seismic, norm=mpl.colors.Normalize(vmin=-cmaprange, vmax=cmaprange))
        # start plotting
        fig, ax = plt.subplots(nrows=3, sharex=True)
        for i, model in enumerate(self.splines):
            # where to put this scale
            y_off = 1 - (i + 1)*dy_scale
            # get normalized values
            if self.splineclass == BSpline:
                mdl_mapping = model.get_mapping(t_plot).toarray()
            elif self.splineclass == ISpline:
                mdl_mapping = np.gradient(model.get_mapping(t_plot).toarray(), axis=0)
            else:
                raise NotImplementedError(f"Scalogram undefined for a SplineSet of class {self.splineclass.__name__}.")
            mdl_sum = np.sum(mdl_mapping, axis=1, keepdims=True)
            mdl_sum[mdl_sum == 0] = 1
            y_norm = np.hstack([np.zeros((t_plot.size, 1)), np.cumsum(mdl_mapping / mdl_sum, axis=1)])
            # plot cell
            for j, k in product(range(model.num_parameters), range(num_components)):
                ax[k].fill_between(t_plot, y_off + y_norm[:, j]*dy_scale, y_off + y_norm[:, j+1]*dy_scale, facecolor=cmap.to_rgba(model.parameters[j, k]))
            # plot vertical lines at centerpoints
            for j, k in product(range(model.num_parameters), range(num_components)):
                ax[k].axvline(model.t_reference + pd.Timedelta(j*model.spacing, model.time_unit), y_off, y_off + dy_scale, c='0.5', lw=0.5)
        # finish plot by adding relevant gridlines and labels
        for k in range(num_components):
            for i in range(1, num_scales):
                ax[k].axhline(i*dy_scale, c='0.5', lw=0.5)
            ax[k].set_xlim(t_left, t_right)
            ax[k].set_ylim(0, 1)
            ax[k].set_yticks([i*dy_scale for i in range(num_scales + 1)])
            ax[k].set_yticks([(i + 0.5)*dy_scale for i in range(num_scales)], minor=True)
            ax[k].set_yticklabels(reversed([f"{model.scale:.4g} {model.time_unit}" for model in self.splines]), minor=True)
            ax[k].tick_params(axis='both', labelleft=False, direction='out')
            ax[k].tick_params(axis='y', left=False, which='minor')
        fig.colorbar(cmap, orientation='horizontal', fraction=0.1, pad=0.1, label='Coefficient Value')
        return fig, ax


class Sinusoidal(Model):
    """
    Sinusoidal of given frequency. Estimates amplitude and phase.

    `time_unit` can be the following (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html):
        `W`, `D`, `days`, `day`, `hours`, `hour`, `hr`, `h`, `m`, `minute`, `min`, `minutes`, `T`,
        `S`, `seconds`, `sec`, `second`, `ms`, `milliseconds`, `millisecond`, `milli`, `millis`, `L`,
        `us`, `microseconds`, `microsecond`, `micro`, `micros`, `U`, `ns`, `nanoseconds`, `nano`, `nanos`, `nanosecond`, `N`
    """
    def __init__(self, period, **model_kw_args):
        super().__init__(num_parameters=2, **model_kw_args)
        self.period = float(period)

    def _get_arch(self):
        arch = {"type": "Sinusoidal",
                "kw_args": {"period": self.period}}
        return arch

    def _get_mapping(self, timevector):
        dt = self.tvec_to_numpycol(timevector)
        phase = 2*np.pi * dt / self.period
        coefs = np.stack([np.cos(phase), np.sin(phase)], axis=1)
        return coefs

    @property
    def amplitude(self):
        if not self.is_fitted:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        return np.sqrt(np.sum(self.parameters ** 2))

    @property
    def phase(self):
        if not self.is_fitted:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        return np.arctan2(self.parameters[1], self.parameters[0])


class Logarithmic(Model):
    """
    Geophysical logarithmic `ln(1 + dt/tau)` with a given time constant and time window.
    """
    def __init__(self, tau, zero_after=False, **model_kw_args):
        super().__init__(num_parameters=1, zero_after=zero_after, **model_kw_args)
        if self.t_reference is None:
            warn("No 't_reference' set for Logarithmic model, using 't_start' for it.")
            self.t_reference_str = self.t_start_str
            self.t_reference = self.t_start
        elif self.t_start is None:
            warn("No 't_start' set for Logarithmic model, using 't_reference' for it.")
            self.t_start_str = self.t_reference_str
            self.t_start = self.t_reference
        else:
            assert self.t_reference <= self.t_start, \
                f"Logarithmic model has to have valid bounds, but the reference time {self.t_reference_str} is after the start time {self.t_start_str}."
        self.tau = float(tau)

    def _get_arch(self):
        arch = {"type": "Logarithmic",
                "kw_args": {"tau": self.tau}}
        return arch

    def _get_mapping(self, timevector):
        dt = self.tvec_to_numpycol(timevector)
        coefs = np.log1p(dt / self.tau).reshape(-1, 1)
        return coefs
