"""
This module contains all models that can be used to fit the data
or generate synthetic timeseries.
"""

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
    r"""
    Base class for models.

    The class implements some common methods as to how one interacts with models, with
    the goal that subclasses of it can focus on as few details as possible.

    A model, generally speaking, is defined by parameters (and optionally their
    covariance), and can be evaluated given a series of timestamps. Most models will
    make use of the :attr:`~time_unit` and :attr:`~t_reference` attributes to relate
    time series into data space. Solvers can check the :attr:`~regularize` attribute
    to regularize the model during fitting.

    Models are usually linear (but can be overriden in subclasses), and adhere to the
    nomenclature

    .. math:: \mathbf{G}(\mathbf{t}) \cdot \mathbf{m} = \mathbf{d}

    where :math:`\mathbf{G}` is the mapping matrix, :math:`\mathbf{t}` is the time
    vector, :math:`\mathbf{m}` are the model parameters, and :math:`\mathbf{d}` are
    the observations.

    Models are always active by default, i.e. they implement a certain functional form
    that can be evaluated at any time. By setting the :attr:`~t_start` and :attr:`~t_end`
    attributes, this behavior can be changed, so that it is only defined on that interval,
    and is zero or continuous outside of that interval (see the attributes
    :attr:`~zero_before` and :attr:`~zero_after`).

    The usual workflow is to instantiate a model, then fit it to a timeseries, saving
    the parameters, and then evaluate it later. For synthetic timeseries, it is
    instantiated and the parameters are set manually.

    Parameters
    ----------
    num_parameters : int
        Number of model parameters.
    regularize : bool, optional
        If ``True``, regularization-capable solvers will regularize the parameters of this model.
    time_unit : str, optional
        Time unit for parameters. Possible values are:

        ``W``, ``D``, ``days``, ``day``, ``hours``, ``hour``, ``hr``, ``h``,
        ``m``, ``minute``, ``min``, ``minutes``, ``T``, ``S``, ``seconds``, ``sec``, ``second``,
        ``ms``, ``milliseconds``, ``millisecond``, ``milli``, ``millis``, ``L``,
        ``us``, ``microseconds``, ``microsecond``, ``micro``, ``micros``, ``U``,
        ``ns``, ``nanoseconds``, ``nano``, ``nanos``, ``nanosecond``, ``N``

        Refer to :func:`~pandas.to_timedelta` for more details.
    t_start : str, pandas.Timestamp or None, optional
        Sets the model start time (attributes :attr:`~t_start` and :attr:`t_start_str`).
    t_end : str, pandas.Timestamp or None, optional
        Sets the model end time (attributes :attr:`~t_end` and :attr:`t_end_str`).
    t_reference : str, pandas.Timestamp or None, optional
        Sets the model reference time (attributes :attr:`~t_reference` and :attr:`t_reference_str`).
    zero_before : bool, optional
        Defines whether the model is zero before ``t_start``, or
        if the boundary value should be used (attribute :attr:`~zero_before`).
    zero_after : bool, optional
        Defines whether the model is zero after ``t_end``, or
        if the boundary value should be used (attribute :attr:`~zero_after`).
    parameters : numpy.ndarray, optional
        If provided, already save the model parameters.
    cov : numpy.ndarray, optional
        If provided (and ``parameters`` is provided as well), already save the
        model parameter covariance.

    See Also
    --------
    read_parameters : Function used to read in the model parameters.
    """
    def __init__(self, num_parameters, regularize=False, time_unit=None, t_start=None, t_end=None, t_reference=None,
                 zero_before=True, zero_after=True, parameters=None, cov=None):
        self.num_parameters = int(num_parameters)
        """ Number of parameters that define the model and can be solved for. """
        assert self.num_parameters > 0, f"'num_parameters' must be an integer greater or equal to one, got {self.num_parameters}."
        self.is_fitted = False
        """
        Tracks whether the model has its parameters set (either by fitting or direct) and can therefore
        be evaluated. :meth:`~read_parameters` sets it to ``True``, and changing the model (for example
        by calling :meth:`~Step.add_step`) resets it to ``False``. Prevents model to be evaluated
        if ``False``.
        """
        self.parameters = None
        """ Attribute that contains the parameters as a NumPy array. """
        self.cov = None
        """ Attribute that contains the parameter's covariance as a NumPy array. """
        self.regularize = bool(regularize)
        """ Solvers will check this to see whether the model should be regularized (``True``) or not. """
        self.time_unit = str(time_unit)
        """ Stores the time unit of the parameters as a string. """
        self.t_start_str = None if t_start is None else str(t_start)
        """ String representation of the start time (or ``None``). """
        self.t_end_str = None if t_start is None else str(t_end)
        """ String representation of the end time (or ``None``). """
        self.t_reference_str = None if t_start is None else str(t_reference)
        """ String representation of the reference time (or ``None``). """
        self.t_start = None if t_start is None else pd.Timestamp(t_start)
        """ :class:`~pandas.Timestamp` representation of the start time (or ``None``). """
        self.t_end = None if t_end is None else pd.Timestamp(t_end)
        """ :class:`~pandas.Timestamp` representation of the end time (or ``None``). """
        self.t_reference = None if t_reference is None else pd.Timestamp(t_reference)
        """ :class:`~pandas.Timestamp` representation of the reference time (or ``None``). """
        self.zero_before = bool(zero_before)
        """
        If ``True``, model will evaluate to zero before the start time, otherwise the
        model value at the start time will be used for all times before that.
        """
        self.zero_after = bool(zero_after)
        """
        If ``True``, model will evaluate to zero after the end time, otherwise the
        model value at the end time will be used for all times after that.
        """
        # read in parameters if they were passed to constructor
        if parameters is not None:
            self.read_parameters(parameters, cov)

    def get_arch(self):
        """
        Get a dictionary that describes the model fully and allows it to be recreated.
        Requires the model to be subclassed and implement a :meth:`~_get_arch` method
        that expands the base model keywords to the subclassed model details.

        Returns
        -------
        arch : dict
            Model keyword dictionary.

        Raises
        ------
        NotImplementedError
            If the model has not been subclassed and :meth:`~_get_arch` has not been added.
        """
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
        r"""
        Builds the mapping matrix :math:`\mathbf{G}` given a time vector :math:`\mathbf{t}`.
        Requires the model to be subclassed and implement a :meth:`~_get_mapping` method.

        This method has multiple steps: it first checks the active period of the
        model using :meth:`~get_active_period`. If ``timevector`` is outside the active period,
        it skips the actual calculation and returns an empty sparse matrix. If there is at least
        one timestamp where the model is active, it calls the actual :meth:`~_get_mapping`
        mapping matrix calculation method only for the timestamps where the model is active in
        order to reduce the computational load. Lastly, the dense, evaluated mapping matrix
        gets padded before and after with empty sparse matrices (if the model is zero outside
        its boundaries) or the values at the boundaries themselves.

        Parameters
        ----------
        timevector : pandas.Series, pandas.DatetimeIndex
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
        mapping : scipy.sparse.bsr_matrix
            Sparse mapping matrix.

        Raises
        ------
        NotImplementedError
            If the model has not been subclassed and :meth:`~_get_mapping` has not been added.
        """
        # get active period and initialize coefficient matrix
        active, first, last = self.get_active_period(timevector)
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

    def get_active_period(self, timevector):
        """
        Given a time vector, return at each point whether the model is active.

        Parameters
        ----------
        timevector : pandas.Series, pandas.DatetimeIndex
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        ----------
        active : numpy.ndarray
            Array of same length as ``timevector``, with ``True`` where active.
        first : int
            Index of the first active timestamp.
        last : int
            Index of the last active Timestamp.
        """
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
        """
        Convenience wrapper for :func:`~tools.tvec_to_numpycol` for Model objects that have the
        :attr:`~time_unit` and :attr:`~t_reference` attributes set.

        See Also
        --------
        :func:`~geonat.tools.tvec_to_numpycol` : Convert a Timestamp vector into a NumPy array.
        """
        if self.t_reference is None:
            raise ValueError("Can't call 'tvec_to_numpycol' because no reference time was specified in the model.")
        if self.time_unit is None:
            raise ValueError("Can't call 'tvec_to_numpycol' because no time unit was specified in the model.")
        return tvec_to_numpycol(timevector, self.t_reference, self.time_unit)

    def read_parameters(self, parameters, cov=None):
        r"""
        Reads in the parameters :math:`\mathbf{m}` (optionally also their covariance)
        and stores them in the instance attributes.

        Parameters
        ----------
        parameters : numpy.ndarray
            Model parameters of shape :math:`(\text{num_parameters},)`.
        cov : numpy.ndarray, optional
            Model parameter covariance of shape :math:`(\text{num_parameters}, \text{num_parameters})`.
        """
        assert self.num_parameters == parameters.shape[0], "Read-in parameters have different size than the instantiated model."
        self.parameters = parameters
        if cov is not None:
            assert self.num_parameters == cov.shape[0] == cov.shape[1], "Covariance matrix must have same number of entries than parameters."
            self.cov = cov
        self.is_fitted = True

    def evaluate(self, timevector):
        r"""
        Evaluate the model given a time vector (calculates :math:`\mathbf{d}`
        and its uncertainty, if applicable).

        Parameters
        ----------
        timevector : pandas.Series, pandas.DatetimeIndex
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
        dict
            Dictionary with the keys ``time`` containing the input time vector,
            ``fit`` containing :math:`\mathbf{d}`, and ``sigma`` containing
            the formal uncertainty (square root of the covariance diagonals).

        Raises
        ------
        RuntimeError
            If the model parameters have not yet been set with :meth:`~read_parameters`.
        """
        if not self.is_fitted:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        mapping_matrix = self.get_mapping(timevector=timevector)
        fit = mapping_matrix @ self.parameters
        fit_sigma = mapping_matrix @ np.sqrt(self.cov.diagonal(offset=0, axis1=0, axis2=1).T) if self.cov is not None else None
        if fit.ndim == 1:
            fit = fit.reshape(-1, 1)
            if fit_sigma is not None:
                fit_sigma = fit_sigma.reshape(-1, 1)
        return {"time": timevector, "fit": fit, "sigma": fit_sigma}


class Step(Model):
    """
    Subclasses :class:`~geonat.model.Model`.

    Model that introduces steps at discrete times.

    Parameters
    ----------
    steptimes : list
        List of datetime-like strings that can be converted into :class:`~pandas.Timestamp`.
        Length of it equals the number of model parameters.


    See :class:`~geonat.model.Model` for attribute descriptions and more keyword arguments.
    """
    def __init__(self, steptimes, zero_after=False, **model_kw_args):
        super().__init__(num_parameters=len(steptimes), zero_after=zero_after, **model_kw_args)
        self.timestamps = [pd.Timestamp(step) for step in steptimes]
        """ List of step times as :class:`~pandas.Timestamp`. """
        self.timestamps.sort()
        self.steptimes = [step.isoformat() for step in self.timestamps]
        """ List of step times as datetime-like strings. """

    def _get_arch(self):
        arch = {"type": "Step",
                "kw_args": {"steptimes": self.steptimes}}
        return arch

    def _update_from_steptimes(self):
        self.timestamps = [pd.Timestamp(step) for step in self.steptimes]
        self.timestamps.sort()
        self.steptimes = [step.isoformat() for step in self.timestamps]
        self.num_parameters = len(self.timestamps)
        self.is_fitted = False
        self.parameters = None
        self.cov = None

    def add_step(self, step):
        """
        Add a step to the model.

        Parameters
        ----------
        step : str
            Datetime-like string of the step time to add
        """
        if step in self.steptimes:
            warn(f"Step '{step}' already present.", category=RuntimeWarning)
        else:
            self.steptimes.append(step)
            self._update_from_steptimes()

    def remove_step(self, step):
        """
        Remove a step to the model.

        Parameters
        ----------
        step : str
            Datetime-like string of the step time to remove
        """
        try:
            self.steptimes.remove(step)
            self._update_from_steptimes()
        except ValueError:
            warn(f"Step '{step}' not present.", category=RuntimeWarning)

    def _get_mapping(self, timevector):
        coefs = np.array(timevector.values.reshape(-1, 1) >= pd.DataFrame(data=self.timestamps, columns=["steptime"]).values.reshape(1, -1), dtype=float)
        return coefs


class Polynomial(Model):
    """
    Subclasses :class:`~geonat.model.Model`.

    Polynomial model of given order.

    Parameters
    ----------
    order : int
        Order (highest exponent) of the polynomial. The number of model parameters
        equals ``order + 1``.


    See :class:`~geonat.model.Model` for attribute descriptions and more keyword arguments.
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
    Subclasses :class:`~geonat.model.Model`.

    Model defined by cardinal, centralized B-Splines of certain order/degree and time scale.
    Used for transient temporary signals that return to zero after a given time span.

    Parameters
    ----------
    degree : int
        Degree of the B-Splines.
    scale : float
        Scale of the B-Splines, see Notes.
    t_reference : str or pandas.Timestamp
        Reference (center) time for (first) spline.
    time_unit : str
        Time unit of scale, spacing and model parameters.
    num_splines : int, optional
        Number of splines, separated by ``spacing``. Defaults to ``1``.
    spacing : float, optional
        Spacing between the center times when multiple splines are created.
        Defaults to ``scale``.


    See :class:`~geonat.model.Model` for attribute descriptions and more keyword arguments.

    Notes
    -----

    For an analytic representation of the B-Splines, see [butzer88]_ or [schoenberg73]_.
    Further examples can be found at `<https://bsplines.org/flavors-and-types-of-b-splines/>`_.

    It is important to note that the function will be non-zero on the interval

    .. math:: -(p+1)/2 < x < (p+1)/2

    where :math:`p` is the degree of the cardinal B-spline (and the degree of the resulting polynomial).
    The order :math:`n` is related to the degree by the relation :math:`n = p + 1`.
    The scale determines the width of the spline in the time domain, and corresponds to the interval [0, 1] of the B-Spline.
    The full non-zero time span of the spline is therefore :math:`\text{scale} * (p+1) = \text{scale} * n`.

    :attr:`~num_splines` will increase the number of splines by shifting the reference point
    :math:`(\text{num_splines} - 1)` times by the spacing (which must be given in the same units as the scale).

    If no spacing is given but multiple splines are requested, the scale will be used as the spacing.

    References
    ----------

    .. [butzer88] Butzer, P., Schmidt, M., & Stark, E. (1988).
       *Observations on the History of Central B-Splines.*
       Archive for History of Exact Sciences, 39(2), 137-156. Retrieved May 14, 2020, from `<https://www.jstor.org/stable/41133848>`_
    .. [schoenberg73] Schoenberg, I. J. (1973). *Cardinal Spline Interpolation.*
       Society for Industrial and Applied Mathematics. `<https://doi.org/10.1137/1.9781611970555>`_
    """
    def __init__(self, degree, scale, t_reference, time_unit, num_splines=1, spacing=None, **model_kw_args):
        self.degree = int(degree)
        """ Degree :math:`p` of the B-Splines. """
        self.order = self.degree + 1
        """ Order :math:`n=p+1` of the B-Splines. """
        self.scale = float(scale)
        """ Scale of the splines. """
        if spacing is not None:
            self.spacing = float(spacing)
            """ Spacing between the center times of the splines. """
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
        """
        Returns a mask-like array of where each spline is currently transient (not staying constant).

        Parameters
        ----------
        timevector : pandas.Series, pandas.DatetimeIndex
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
        transient : numpy.ndarray
            NumPy array with ``True`` when a spline is currently transient, ``False`` otherwise.
        """
        trel = self.tvec_to_numpycol(timevector).reshape(-1, 1) - self.scale * np.arange(self.num_parameters).reshape(1, -1)
        transient = np.abs(trel) <= self.scale * self.order
        return transient


class ISpline(Model):
    """
    Subclasses :class:`~geonat.model.Model`.

    Integral of cardinal, centralized B-Splines of certain order/degree and time scale.
    The degree :math:`p` given in the initialization is the degree of the spline *before* the integration, i.e.
    the resulting ISpline is a piecewise polynomial of degree :math:`p + 1`.
    Used for transient permanent signals that stay at their maximum value after a given time span.

    See Also
    --------
    geonat.model.Bspline : More details about B-Splines.
    """
    def __init__(self, degree, scale, t_reference, time_unit, num_splines=1, spacing=None, zero_after=False, **model_kw_args):
        self.degree = int(degree)
        """ Degree :math:`p` of the B-Splines. """
        self.order = self.degree + 1
        """ Order :math:`n=p+1` of the B-Splines. """
        self.scale = float(scale)
        """ Scale of the splines. """
        if spacing is not None:
            self.spacing = float(spacing)
            """ Spacing between the center times of the splines. """
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
        """
        Returns a mask-like array of where each spline is currently transient (not staying constant).

        Parameters
        ----------
        timevector : pandas.Series, pandas.DatetimeIndex
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
        transient : numpy.ndarray
            NumPy array with ``True`` when a spline is currently transient, ``False`` otherwise.
        """
        trel = self.tvec_to_numpycol(timevector).reshape(-1, 1) - self.scale * np.arange(self.num_parameters).reshape(1, -1)
        transient = np.abs(trel) <= self.scale * self.order
        return transient


class SplineSet(Model):
    """
    Subclasses :class:`~geonat.model.Model`.

    Contains a list of splines that share a common degree, but different center times and scales.

    The set is constructed from a time span (``t_center_start`` and ``t_center_end``) and numbers of centerpoints
    or length scales. By default (``complete=True``), the number of splines and center points for each scale will then be
    chosen such that the resulting set of splines will be complete over the input time scale.
    This means it will contain all splines that are non-zero at least somewhere in the time span.
    Otherwise, the spline set will only have center times at or between ``t_center_start`` and ``t_center_end``.

    This class also sets the spacing equal to the scale.

    Parameters
    ----------
    degree : int
        Degree of the splines to be created.
    t_center_start : str or pandas.Timestamp
        Time span start of the spline set.
    t_center_end : str or pandas.Timestamp
        Time span end of the spline set.
    time_unit : str
        Time unit of scale, spacing and model parameters.
    list_scales : list
        List of scales to use for each of the sub-splines.
        Mutually exclusive to setting ``list_num_knots``.
    list_num_knots : list
        List of number of knots to divide the time span into for each of the sub-splines.
        Mutually exclusive to setting ``list_scales``.
    splineclass : Model, optional
        Model class to use for the splines. Defaults to :class:`~geonat.model.ISpline`.
    complete : bool, optional
        See usage description above. Defaults to ``True``.


    See :class:`~geonat.model.Model` for attribute descriptions and more keyword arguments.
    """
    def __init__(self, degree, t_center_start, t_center_end, time_unit, list_scales=None, list_num_knots=None,
                 splineclass=ISpline, complete=True, **model_kw_args):
        assert np.logical_xor(list_scales is None, list_num_knots is None), \
            f"To construct a set of Splines, pass exactly one of 'list_scales' and 'list_num_knots' " \
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
        t_center_start, t_center_end = t_center_start_tstamp.isoformat(), t_center_end_tstamp.isoformat()
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
        """ Degree of the splines. """
        self.t_center_start = t_center_start
        """ Relevant time span start. """
        self.t_center_end = t_center_end
        """ Relevant time span end. """
        self.splineclass = splineclass
        """ Class of the splines contained. """
        self.list_scales = list_scales
        """ List of scales of each of the sub-splines. """
        self.list_num_knots = list_num_knots
        """ List of number of knots the time span is divided into for each of the sub-splines. """
        self.complete = complete
        """ Sets whether the spline coverage of the time span is considered to be complete or not. """
        self.splines = splset
        """ List of spline object contained within the SplineSet. """

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

    def read_parameters(self, parameters, cov=None):
        r"""
        Reads in the parameters :math:`\mathbf{m}` (optionally also their covariance)
        of all the sub-splines and stores them in the respective attributes.

        Parameters
        ----------
        parameters : numpy.ndarray
            Model parameters of shape :math:`(\text{num_parameters},)`.
        cov : numpy.ndarray, optional
            Model parameter covariance of shape :math:`(\text{num_parameters}, \text{num_parameters})`.
        """
        super().read_parameters(parameters, cov)
        ix_params = 0
        for i, model in enumerate(self.splines):
            cov_model = None if cov is None else cov[ix_params:ix_params + model.num_parameters, ix_params:ix_params + model.num_parameters]
            model.read_parameters(parameters[ix_params:ix_params + model.num_parameters], cov_model)
            ix_params += model.num_parameters

    def make_scalogram(self, t_left, t_right, cmaprange=None, resolution=1000):
        """
        Create a scalogram figure of the model parameters.

        A scalogram shows the amplitude of each model parameter plotted over time and
        for all the different scales contained. Model parameters that have overlapping
        influence are also shown as overlapping. The height of each parameter's patch
        is defined by the weight of that parameter relative to the other parameters
        (excluding splines that are not transient at that time).

        Parameters
        ----------
        t_left : str
            Left boundary of the time axis.
        t_right : str
            Right boundary of the time axis.
        cmaprange : float or int, optional
            Maximum absolute amplitude of the color scale to use.
            Defaults to the 95th percentile of the absolute amplitudes of all parameters.
        resolution : int, optional
            Number of points inside the time span to evaluate the scalogram at.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object of the scalogram.
        ax : matplotlib.axes.Axes
            Axes object of the scalogram.

        Raises
        ------
        NotImplementedError
            If the generation method for the scalogram given the SplineSet's
            spline class is not defined in this method yet.
        """
        # check input
        assert self.is_fitted, f"SplineSet model needs to have already been fitted."
        # determine dimensions
        num_components = self.parameters.shape[1]
        num_scales = len(self.splines)
        dy_scale = 1/num_scales
        t_plot = pd.Series(pd.date_range(start=t_left, end=t_right, periods=resolution))
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
    r"""
    Subclasses :class:`~geonat.model.Model`.

    This model provides a sinusoidal of a fixed period, with amplitude and phase
    to be fitted.

    Parameters
    ----------
    period : float
        Period length in :attr:`~geonat.model.Model.time_unit` units.


    See :class:`~geonat.model.Model` for attribute descriptions and more keyword arguments.

    Notes
    -----

    Implements the relationship

    .. math::
        \mathbf{g}(\mathbf{t}) =
        a \cos ( 2 \pi \mathbf{t} / T ) + b \sin ( 2 \pi \mathbf{t} / T )

    with :attr:`~period` :math:`T`, :attr:`~phase` :math:`\phi=\text{atan2}(b,a)`
    and :attr:`~amplitude` :math:`A=\sqrt{a^2 + b^2}`.
    """
    def __init__(self, period, **model_kw_args):
        super().__init__(num_parameters=2, **model_kw_args)
        self.period = float(period)
        """ Period of the sinusoid. """

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
        """ Amplitude of the sinusoid. """
        if not self.is_fitted:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        return np.sqrt(np.sum(self.parameters ** 2))

    @property
    def phase(self):
        """ Phase of the sinusoid. """
        if not self.is_fitted:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        return np.arctan2(self.parameters[1], self.parameters[0])


class Logarithmic(Model):
    r"""
    Subclasses :class:`~geonat.model.Model`.

    This model provides the 'geophysical' logarithmic :math:`\ln(1 + \mathbf{t}/\tau)`
    with a given time constant and zero for :math:`\mathbf{t} < 0`.

    Parameters
    ----------
    tau : float
        Logarithmic time constant :math:`\tau`.


    See :class:`~geonat.model.Model` for attribute descriptions and more keyword arguments.
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
        """ Logarithmic time constant. """

    def _get_arch(self):
        arch = {"type": "Logarithmic",
                "kw_args": {"tau": self.tau}}
        return arch

    def _get_mapping(self, timevector):
        dt = self.tvec_to_numpycol(timevector)
        coefs = np.log1p(dt / self.tau).reshape(-1, 1)
        return coefs