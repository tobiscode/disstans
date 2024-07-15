"""
This module contains all models that can be used to fit the data
or generate synthetic timeseries.
"""

from __future__ import annotations
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from warnings import warn
from scipy.special import comb, factorial
from itertools import product
from cmcrameri import cm as scm
from collections import UserDict
from scipy.interpolate import BSpline as sp_bspl
from typing import Any
from collections.abc import Iterator, ItemsView

from .tools import tvec_to_numpycol, Timedelta, full_cov_mat_to_columns, cov2corr
from .timeseries import Timeseries


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

    A minimal user-defined subclass should look similar to :class:`~disstans.models.Polynomial`
    or :class:`~disstans.models.Exponential`. Three methods need to be provided: an
    ``__init__()`` function that takes in any model-specific parameters and passes all
    other parameters into the parent class through ``super().__init__()``, as well as
    both :meth:`~get_mapping_single` and :meth:`~_get_arch` (see the base class' documentation
    for expected in- and output).

    Appendix A.2 of [koehne23]_ describes in detail the approach to models in DISSTANS.

    Parameters
    ----------
    num_parameters
        Number of model parameters.
    regularize
        If ``True``, regularization-capable solvers will regularize the
        parameters of this model.
    time_unit
        Time unit for parameters.
        Refer to :class:`~disstans.tools.Timedelta` for more details.
    t_start
        Sets the model start time (attributes :attr:`~t_start` and :attr:`t_start_str`).
    t_end
        Sets the model end time (attributes :attr:`~t_end` and :attr:`t_end_str`).
    t_reference
        Sets the model reference time (attributes :attr:`~t_reference`
        and :attr:`t_reference_str`).
    zero_before
        Defines whether the model is zero before ``t_start``, or
        if the boundary value should be used (attribute :attr:`~zero_before`).
    zero_after
        Defines whether the model is zero after ``t_end``, or
        if the boundary value should be used (attribute :attr:`~zero_after`).

    References
    ----------

    .. [koehne23] Köhne, T., Riel, B., & Simons, M. (2023).
       *Decomposition and Inference of Sources through Spatiotemporal Analysis of
       Network Signals: The DISSTANS Python package.*
       Computers & Geosciences, 170, 105247.
       doi:`10.1016/j.cageo.2022.10524 <https://doi.org/10.1016/j.cageo.2022.105247>`_
    """

    EVAL_PREDVAR_PRECISION = np.dtype(np.single)
    """
    To reduce memory impact when estimating the full covariance of the predicted
    timeseries when calling :meth:`~evaluate`, this attribute is by default set to
    single precision, but can be changed to double precision if desired.
    """

    def __init__(self,
                 num_parameters: int,
                 regularize: bool = False,
                 time_unit: str = None,
                 t_start: str | pd.Timestamp | None = None,
                 t_end: str | pd.Timestamp | None = None,
                 t_reference: str | pd.Timestamp | None = None,
                 zero_before: bool = True,
                 zero_after: bool = True
                 ) -> None:
        # define model settings
        self.num_parameters = int(num_parameters)
        """ Number of parameters that define the model and can be solved for. """
        assert self.num_parameters > 0, \
            "'num_parameters' must be an integer greater or equal to one, " \
            f"got {self.num_parameters}."
        self.regularize = bool(regularize)
        """ Indicate to solvers to regularize this model (``True``) or not. """
        self.time_unit = None if time_unit is None else str(time_unit)
        """ Stores the time unit of the parameters as a string. """
        self.t_start_str = None if t_start is None else str(t_start)
        """ String representation of the start time (or ``None``). """
        self.t_end_str = None if t_end is None else str(t_end)
        """ String representation of the end time (or ``None``). """
        self.t_reference_str = None if t_reference is None else str(t_reference)
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
        self.active_parameters = None
        r"""
        By default, all parameters in the model are considered active, and this attribute is
        set to ``None``. Otherwise, this attribute contains an array of shape
        :math:`(\text{num_parameters}, )` with ``True`` where parameters are active, and
        ``False`` otherwise.
        """
        # initialize data variables
        self._par = None
        self._cov = None

    @property
    def par(self) -> np.ndarray:
        r"""
        Array property of shape :math:`(\text{num_parameters}, \text{num_components})`
        that contains the parameters as a NumPy array.
        """
        return self._par

    @property
    def parameters(self) -> np.ndarray:
        """ Alias for :attr:`~par`. """
        return self.par

    @property
    def var(self) -> np.ndarray | None:
        r"""
        Array property of shape :math:`(\text{num_parameters}, \text{num_components})`
        that returns the parameter's individual variances as a NumPy array.
        """
        if self._cov is not None:
            var = np.diag(self._cov).reshape(self.num_parameters, -1)
        else:
            var = None
        return var

    @property
    def cov(self) -> np.ndarray | None:
        r"""
        Square array property with dimensions
        :math:`\text{num_parameters} * \text{num_components}` that contains the parameter's
        full covariance matrix as a NumPy array. The rows (and columns) are ordered such
        that they first correspond to the covariances between all components for the first
        parameter, then the covariance between all components for the second parameter,
        and so forth.
        """
        return self._cov

    def get_cov_by_index(self, index: int) -> np.ndarray | None:
        """
        Return the covariance matrix for a given parameter.

        Parameters
        ----------
        index
            Parameter index.

        Returns
        -------
            Covariance matrix for the selected parameter.
        """
        assert isinstance(index, int) and (0 <= index < self.num_parameters), \
            f"'index' needs to be an integer less than the number of parameters, got {index}."
        if self._cov is not None:
            num_comps = self._par.shape[1]
            return self._cov[index * num_comps:(index + 1) * num_comps,
                             index * num_comps:(index + 1) * num_comps]
        else:
            return None

    def __str__(self) -> str:
        """
        Special function that returns a readable summary of the Model.
        Accessed, for example, by Python's ``print()`` built-in function.

        Returns
        -------
            Model summary.
        """
        arch = self.get_arch()
        info = f"{arch['type']} model ({self.num_parameters} parameters)"
        for k, v in arch["kw_args"].items():
            info += f"\n  {k + ':':<15}{v}"
        return info

    def __eq__(self, other: Model) -> bool:
        """
        Special function that allows for the comparison of models based on their
        type and architecture, regardless of model parameters.

        Parameters
        ----------
        other
            Model to compare to.

        Example
        -------

        >>> from disstans.models import Step, Sinusoid
        >>> step1, step2 = Step(["2020-01-01"]), Step(["2020-01-02"])
        >>> sin1, sin2 = Sinusoid(1, "2020-01-01"), Sinusoid(1, "2020-01-01")
        >>> step1 == step2
        False
        >>> sin1 == sin2
        True

        Note that obviously, the objects themselves are still different:

        >>> step1 is step1
        True
        >>> step1 is step2
        False

        See Also
        --------
        get_arch : Function used to determine the equality.
        """
        return self.get_arch() == other.get_arch()

    def get_arch(self) -> dict[str, Any]:
        """
        Get a dictionary that describes the model fully and allows it to be recreated.
        Requires the model to be subclassed and implement a :meth:`_get_arch` method
        that expands the base model keywords to the subclassed model details.

        Returns
        -------
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

    def _get_arch(self) -> dict[str, Any]:
        """
        Subclass-specific model keyword dictionary.

        Returns
        -------
            Model keyword dictionary. Must have keys ``'type'`` and ``'kw_args'``,
            with a string and a dictionary as values, respectively.
        """
        raise NotImplementedError("Instantiated model was not subclassed or "
                                  "it does not overwrite the '_get_arch' method.")

    def copy(self,
             parameters: bool = True,
             covariances: bool = True,
             active_parameters: bool = True
             ) -> Model:
        """
        Copy the model object.

        Parameters
        ----------
        parameters
            If ``True``, include the read-in parameters in the copy
            (:attr:`~par`), otherwise leave empty.
        covariances
            If ``True``, include the read-in (co)variances in the copy
            (:attr:`~cov`), otherwise leave empty.
        active_parameters
            If ``True``, include the active parameter setting in the copy
            (:attr:`~active_parameters`), otherwise leave empty.

        Returns
        -------
            A copy of the model, based on :meth:`~get_arch`.
        """
        # instantiate
        arch = self.get_arch()
        mdl = globals()[arch["type"]](**arch["kw_args"])
        # update read-in parameters and settings
        if parameters and (self._par is not None):
            mdl._par = self._par.copy()
        if covariances and (self._cov is not None):
            mdl._cov = self._cov.copy()
        if active_parameters and (self.active_parameters is not None):
            mdl.active_parameters = self.active_parameters.copy()
        return mdl

    def convert_units(self, factor: float) -> None:
        """
        Convert the parameter and covariances to a new unit by providing a
        conversion factor.

        Parameters
        ----------
        factor
            Factor to multiply the parameters by to obtain the parameters in the new units.
        """
        # input checks
        try:
            factor = float(factor)
        except TypeError as e:
            raise TypeError(f"'factor' and has to be a scalar, got {type(factor)}."
                            ).with_traceback(e.__traceback__) from e
        # convert parameters
        if self._par is not None:
            self._par *= factor
        # convert covariances
        if self._cov is not None:
            self._cov *= factor**2

    def freeze(self, zero_threshold: float = 1e-10) -> None:
        """
        In case some parameters are estimated to be close to zero and should not
        be considered in future fits and evaluations, this function "freezes"
        the model by setting parameters below the threshold ``zero_threshold``
        to be invalid. The mask will be kept in :attr:`~active_parameters`.

        Only valid parameters will be used by :meth:`~get_mapping` and
        :meth:`~evaluate`.

        Parameters
        ----------
        zero_threshold
            Model parameters with absolute values below ``zero_threshold`` will be
            set to zero and set inactive.

        See Also
        --------
        unfreeze : The reverse method.
        """
        assert float(zero_threshold) > 0, \
            f"'zero_threshold needs to be non-negative, got {zero_threshold}."
        if self.par is None:
            raise RuntimeError("Cannot freeze a model without set parameters.")
        self.active_parameters = np.any(np.abs(self.par) > zero_threshold, axis=1)
        self._par[~self.active_parameters, :] = 0
        if self._cov is not None:
            inactive_ix = np.repeat(~self.active_parameters, self.par.shape[1])
            self._cov[np.ix_(inactive_ix, inactive_ix)] = 0

    def unfreeze(self) -> None:
        """
        Resets previous model freezing done by :meth:`~freeze` such that all parameters
        are active again.

        See Also
        --------
        freeze : The reverse method.
        """
        self.active_parameters = None

    def get_mapping(self,
                    timevector: pd.Series | pd.DatetimeIndex,
                    return_observability: bool = False,
                    ignore_active_parameters: bool = False
                    ) -> sparse.csc_matrix | tuple[sparse.csc_matrix, np.ndarray]:
        r"""
        Builds the mapping matrix :math:`\mathbf{G}` given a time vector :math:`\mathbf{t}`.
        Requires the model to be subclassed and implement a :meth:`~get_mapping_single` method.

        This method has multiple steps: it first checks the active period of the
        model using :meth:`~get_active_period`. If ``timevector`` is outside the active period,
        it skips the actual calculation and returns an empty sparse matrix. If there is at least
        one timestamp where the model is active, it calls the actual :meth:`~get_mapping_single`
        mapping matrix calculation method only for the timestamps where the model is active in
        order to reduce the computational load. Lastly, the dense, evaluated mapping matrix
        gets padded before and after with empty sparse matrices (if the model is zero outside
        its boundaries) or the values at the boundaries themselves.

        This method respects the parameters being set invalid by :meth:`~freeze`, and will
        interpret those parameters to be unobservable.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.
        return_observability
            If true, the function will check if there are any all-zero columns, which
            would point to unobservable parameters, and return a boolean mask with the
            valid indices.
        ignore_active_parameters
            If ``True``, do not set inactive parameters to zero to avoid estimation.

        Returns
        -------
        mapping
            Sparse mapping matrix.
        observable
            Returned if ``return_observability=True``.
            A boolean NumPy array of the same length as ``mapping`` has columns.
            ``False`` indicates (close to) all-zero columns (unobservable parameters).

        Raises
        ------
        NotImplementedError
            If the model has not been subclassed and :meth:`~get_mapping_single`
            has not been added.
        """
        # get active period and initialize coefficient matrix
        active, first, last = self.get_active_period(timevector)
        # if there isn't any active period, return csc-sparse matrix
        if (first is None) and (last is None):  # this is equivalent to not active.any()
            mapping = sparse.csc_matrix((timevector.size, self.num_parameters))
            if return_observability:
                observable = np.zeros(self.num_parameters, dtype=bool)
        # otherwise, build coefficient matrix
        else:
            # build dense sub-matrix
            coefs = self.get_mapping_single(timevector[active])
            assert coefs.shape[1] == self.num_parameters, \
                f"The child function 'get_mapping_single' of model {type(self).__name__} " \
                f"returned an invalid shape. " \
                f"Expected was ({last - first + 1}, {self.num_parameters}), got {coefs.shape}."
            # if model is frozen, zero out inactive parameters
            if (self.active_parameters is not None) and (not ignore_active_parameters):
                coefs[:, ~self.active_parameters] = 0
            if return_observability:
                # check for the number effective non-zero coefficients
                # technically observable where we have at least one such value
                # for regularized models, also skip all columns with just a single value,
                # as this would just map into another constant offset, which should
                # be taken care of by a non-regularized polynomial
                maxamps = np.max(np.abs(coefs), axis=0, keepdims=True)
                maxamps[maxamps == 0] = 1
                numnotzero = np.sum(~np.isclose(coefs / maxamps, 0), axis=0)
                obsnonzero = numnotzero > 1 if self.regularize else numnotzero > 0
                numunique = np.array([np.unique(coefs[:, i]).size
                                      for i in range(self.num_parameters)])
                obsunique = numunique > 1 if self.regularize else numunique > 0
                observable = np.logical_and(obsnonzero, obsunique)
            # build before- and after-matrices
            # either use zeros or the values at the active boundaries for padding
            if self.zero_before:
                before = sparse.csc_matrix((first, self.num_parameters))
            else:
                before = sparse.csc_matrix(np.ones((first, self.num_parameters))
                                           * coefs[0, :].reshape(1, -1))
            if self.zero_after:
                after = sparse.csc_matrix((timevector.size - last - 1, self.num_parameters))
            else:
                after = sparse.csc_matrix(np.ones((timevector.size - last - 1,
                                                   self.num_parameters))
                                          * coefs[-1, :].reshape(1, -1))
            # stack them (they can have 0 in the first dimension, no problem for sparse.vstack)
            # I think it's faster if to stack them if they're all already csc format
            mapping = sparse.vstack((before, sparse.csc_matrix(coefs), after), format='csc')
        # return
        if return_observability:
            return mapping, observable
        else:
            return mapping

    def get_mapping_single(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        r"""
        Build the mapping matrix :math:`\mathbf{G}` given a time vector :math:`\mathbf{t}`
        for the active period. Called inside :meth:`~get_mapping`.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.
            It can and should be assumed that all included timestamps are valid
            (i.e., defined by the model's :attr:`~zero_before` and :attr:`~zero_after`).

        Returns
        -------
            Mapping matrix with the same number of rows as ``timevector`` and
            :attr:`~num_parameters` columns.
        """
        raise NotImplementedError("'Model' needs to be subclassed and its child needs to "
                                  "implement a 'get_mapping_single' method for the active "
                                  "period.")

    def get_active_period(self,
                          timevector: pd.Series | pd.DatetimeIndex
                          ) -> tuple[np.ndarray, int, int]:
        """
        Given a time vector, return at each point whether the model is active.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        ----------
        active
            Array of same length as ``timevector``, with ``True`` where active.
        first
            Index of the first active timestamp.
        last
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
            first, last = np.flatnonzero(active)[[0, -1]].tolist()
        else:
            first, last = None, None
        return active, first, last

    def tvec_to_numpycol(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        """
        Convenience wrapper for :func:`~disstans.tools.tvec_to_numpycol` for Model objects that
        have the :attr:`~time_unit` and :attr:`~t_reference` attributes set.

        See Also
        --------
        :func:`~disstans.tools.tvec_to_numpycol` : Convert a Timestamp vector into a NumPy array.
        """
        if self.t_reference is None:
            raise ValueError("Can't call 'tvec_to_numpycol' because no reference time "
                             "was specified in the model.")
        if self.time_unit is None:
            raise ValueError("Can't call 'tvec_to_numpycol' because no time unit "
                             "was specified in the model.")
        return tvec_to_numpycol(timevector, self.t_reference, self.time_unit)

    def read_parameters(self,
                        parameters: np.ndarray,
                        covariances: np.ndarray | None = None
                        ) -> None:
        r"""
        Reads in the parameters :math:`\mathbf{m}` (optionally also their
        covariance) and stores them in the instance attributes.

        Parameters
        ----------
        parameters
            Model parameters of shape
            :math:`(\text{num_parameters}, \text{num_components})`.
        covariances
            Model component (co-)variances that can either have the same shape as
            ``parameters``, in which case every parameter and component only has a
            variance, or it is square with dimensions
            :math:`\text{num_parameters} * \text{num_components}`, in which case it
            represents a full variance-covariance matrix.
        """
        # quick check if this is just a reset
        if parameters is None:
            self._par = None
            self._cov = None
            return
        # check and set parameters
        assert self.num_parameters == parameters.shape[0], \
            "Read-in parameters have different size than the instantiated model. " + \
            f"Expected {self.num_parameters}, got {parameters.shape[0]}. The input " + \
            f"shape was {parameters.shape[0]}, is there a dimension missing?"
        par = parameters.reshape((self.num_parameters, -1))
        act_params = self.active_parameters
        if act_params is not None:
            assert np.all(par[~act_params, :] == 0), \
                "Something went wrong: inactive parameters should be estimated as 0."
        self._par = par
        # check and set covariances
        if covariances is None:
            self._cov = None
        else:
            if covariances.shape == self._par.shape:
                covariances = np.diag(covariances.ravel())
            elif not covariances.shape == (parameters.size, parameters.size):
                raise ValueError("Covariance matrix must either have shape "
                                 f"{(parameters.size, parameters.size)} or "
                                 f"{self._par.shape}, got {covariances.shape}.")
            if act_params is not None:
                active_ix = np.repeat(act_params, self._par.shape[1])
                assert np.all(covariances[np.ix_(~active_ix, ~active_ix)] == 0), \
                    "Something went wrong: covariance for inactive parameters should be 0."
            self._cov = covariances

    def evaluate(self,
                 timevector: pd.Series | pd.DatetimeIndex,
                 return_full_covariance: bool = False
                 ) -> dict[str, Any]:
        r"""
        Evaluate the model given a time vector, calculating the predicted timeseries
        :math:`\mathbf{d} = \mathbf{Gm}` and (if applicable) its formal covariance matrix
        :math:`\mathbf{C}_d^{\text{pred}} = \mathbf{G} \mathbf{C}_m \mathbf{G}^T`.

        This method ignores the parameters being set invalid by :meth:`~freeze`.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.
        return_full_covariance
            By default (``False``) the covariances between timesteps are ignored,
            and the returned dictionary will only include the component variances and
            covariances for each timestep. If ``True``, the full covariance matrix
            (between components and timesteps) will be returned instead.

        Returns
        -------
            Dictionary with the keys ``time`` containing the input time vector,
            ``fit`` containing :math:`\mathbf{d}`, ``var`` containing the formal
            variance (or ``None``, if not present), and ``cov`` containing the formal
            covariance (or ``None``, if not present). ``fit``, ``var`` and ``cov``
            (if not ``None``) are :class:`~numpy.ndarray` objects.
            If ``return_full_covariance=True``, ``var`` will be omitted and the full
            covariance matrix will be returned in ``cov``.

        Raises
        ------
        RuntimeError
            If the model parameters have not yet been set with :meth:`~read_parameters`.
        """
        if self.par is None:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        mapping_matrix = self.get_mapping(timevector=timevector)
        fit = mapping_matrix @ self.par
        if self.cov is None:
            fit_var = None
            fit_cov = None
        else:
            # repeat the mapping matrix for all components,
            # same order as full_cov_mat_to_columns needs
            num_components = self.par.shape[1]
            map_mat = sparse.kron(mapping_matrix, np.eye(num_components), format="csc")
            # reduce the size of matrix calculation by removing all-zero rows and columns
            var_full = self.cov
            rowcolnonzero = ~np.all(var_full == 0, axis=0)
            assert np.all(rowcolnonzero == ~np.all(var_full == 0, axis=1))
            var_full = var_full[np.ix_(rowcolnonzero, rowcolnonzero)]
            map_mat = map_mat[:, rowcolnonzero].toarray()
            # calculate the predicted variance
            pred_var = np.matmul(map_mat @ var_full, map_mat.T,
                                 dtype=self.EVAL_PREDVAR_PRECISION, casting="same_kind")
            # extract the (block-)diagonal components and reshape
            fit_var, fit_cov = full_cov_mat_to_columns(pred_var, num_components,
                                                       include_covariance=True)
        if fit.ndim == 1:
            fit = fit.reshape(-1, 1)
        if return_full_covariance:
            return {"time": timevector, "fit": fit, "cov": pred_var}
        else:
            return {"time": timevector, "fit": fit, "var": fit_var, "cov": fit_cov}


class Step(Model):
    """
    Subclasses :class:`~disstans.models.Model`.

    Model that introduces steps at discrete times.

    Parameters
    ----------
    steptimes
        List of datetime-like strings that can be converted into :class:`~pandas.Timestamp`.
        Length of it equals the number of model parameters.


    See :class:`~disstans.models.Model` for attribute descriptions and more keyword arguments.
    """
    def __init__(self,
                 steptimes: list[str],
                 zero_after: bool = False,
                 **model_kw_args
                 ) -> None:
        super().__init__(num_parameters=len(steptimes), zero_after=zero_after, **model_kw_args)
        self.timestamps = [pd.Timestamp(step) for step in steptimes]
        """ List of step times as :class:`~pandas.Timestamp`. """
        self.timestamps.sort()
        self.steptimes = [step.isoformat() for step in self.timestamps]
        """ List of step times as datetime-like strings. """

    def _get_arch(self) -> dict[str, Any]:
        arch = {"type": "Step",
                "kw_args": {"steptimes": self.steptimes}}
        return arch

    def _update_from_steptimes(self) -> None:
        self.timestamps = [pd.Timestamp(step) for step in self.steptimes]
        self.timestamps.sort()
        self.steptimes = [step.isoformat() for step in self.timestamps]
        self.num_parameters = len(self.timestamps)
        self._par = None
        self._cov = None

    def add_step(self, step: str) -> None:
        """
        Add a step to the model.

        Parameters
        ----------
        step
            Datetime-like string of the step time to add
        """
        if step in self.steptimes:
            warn(f"Step '{step}' already present.", category=RuntimeWarning, stacklevel=2)
        else:
            self.steptimes.append(step)
            self._update_from_steptimes()

    def remove_step(self, step: str) -> None:
        """
        Remove a step from the model.

        Parameters
        ----------
        step
            Datetime-like string of the step time to remove
        """
        try:
            self.steptimes.remove(step)
            self._update_from_steptimes()
        except ValueError:
            warn(f"Step '{step}' not present.", category=RuntimeWarning, stacklevel=2)

    def get_mapping_single(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        r"""
        Calculate the mapping factors at times :math:`t` as

        .. math:: H \left( t - t_l^{\text{step}} \right)

        where :math:`H \left( t \right)` is the Heaviside step function and
        :math:`t_l^{\text{step}}` are the step times.

        See Appendix A.2.2 in [koehne23]_ for more details.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            Coefficients of the mapping matrix.
        """
        coefs = np.array(timevector.values.reshape(-1, 1) >=
                         pd.DataFrame(data=self.timestamps,
                                      columns=["steptime"]).values.reshape(1, -1),
                         dtype=float)
        return coefs


class Polynomial(Model):
    """
    Subclasses :class:`~disstans.models.Model`.

    Polynomial model of given order.

    Parameters
    ----------
    order
        Order (highest exponent) of the polynomial. The number of model parameters
        equals ``order + 1 - min_exponent``.
    t_reference
        Sets the model reference time.
    min_exponent
        Lowest exponent of the polynomial. Defaults to ``0``, i.e. the constant offset.


    See :class:`~disstans.models.Model` for attribute descriptions and more keyword arguments.
    """
    def __init__(self,
                 order: int,
                 t_reference: str | pd.Timestamp,
                 min_exponent: int = 0,
                 time_unit: str = "D",
                 zero_before: bool = False,
                 zero_after: bool = False,
                 **model_kw_args) -> None:
        super().__init__(num_parameters=order + 1 - min_exponent,
                         t_reference=t_reference, time_unit=time_unit,
                         zero_before=zero_before, zero_after=zero_after, **model_kw_args)
        self.order = int(order)
        self.min_exponent = int(min_exponent)

    def _get_arch(self) -> dict[str, Any]:
        arch = {"type": "Polynomial",
                "kw_args": {"order": self.order, "min_exponent": self.min_exponent}}
        return arch

    def get_mapping_single(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        r"""
        Calculate the mapping factors at times :math:`t` as

        .. math:: t^l

        where :math:`l` are the integer exponents of the model.

        See Appendix A.2.2 in [koehne23]_ for more details.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            Coefficients of the mapping matrix.
        """
        dt = self.tvec_to_numpycol(timevector)
        # the exponents increase by column
        exponents = np.arange(self.min_exponent, self.order + 1)
        # broadcast to all coefficients
        coefs = dt.reshape(-1, 1) ** exponents.reshape(1, -1)
        return coefs

    def get_exp_index(self, exponent: int) -> int:
        """
        Return the row index for :attr:`~Model.par` and :attr:`~Model.var`
        of a given exponent.

        Parameters
        ----------
        exponent
            Exponent for which to return the index. E.g., ``1`` would correspond
            to the index of the linear term.

        Returns
        -------
            Index of the exponent's term.

        Raises
        ------
        ValueError
            Raised if the desired exponent is not present in the model..
        """
        assert isinstance(exponent, int), \
            f"'exponent' needs to be an integer', got {type(exponent)}."
        if self.min_exponent <= exponent <= self.order:
            return exponent - self.min_exponent
        else:
            raise ValueError(f"Exponent {exponent} is not included in the model.")


class BSpline(Model):
    r"""
    Subclasses :class:`~disstans.models.Model`.

    Model defined by cardinal, centralized B-Splines of certain order/degree and time scale.
    Used for transient temporary signals that return to zero after a given time span.

    Parameters
    ----------
    degree
        Degree of the B-Splines.
    scale
        Scale of the B-Splines, see Notes.
    t_reference
        Reference (center) time for (first) spline.
    time_unit
        Time unit of scale, spacing and model parameters.
    num_splines
        Number of splines, separated by ``spacing``.
    spacing
        Spacing between the center times when multiple splines are created.
        ``None`` defaults to ``scale``.
    obs_scale
        Determines how many factors of ``scale`` should be sampled by the ``timevector``
        input to :meth:`~Model.get_mapping` to accept an individual spline as observable.


    See :class:`~disstans.models.Model` for attribute descriptions and more keyword arguments.

    Notes
    -----

    For an analytic representation of the B-Splines, see [butzer88]_ or [schoenberg73]_.
    Further examples can be found at `<https://bsplines.org/flavors-and-types-of-b-splines/>`_.

    It is important to note that the function will be non-zero on the interval

    .. math:: -(p+1)/2 < x < (p+1)/2

    where :math:`p` is the degree of the cardinal B-spline (and the degree of the
    resulting polynomial). The order :math:`n` is related to the degree by the relation
    :math:`n = p + 1`. The scale determines the width of the spline in the time domain,
    and corresponds to the interval [0, 1] of the B-Spline. The full non-zero time span
    of the spline is therefore :math:`\text{scale} \cdot (p+1) = \text{scale} \cdot n`.

    ``num_splines`` will increase the number of splines by shifting the reference
    point :math:`(\text{num_splines} - 1)` times by the spacing (which must be given
    in the same units as the scale).

    If no spacing is given but multiple splines are requested, the scale will be used
    as the spacing.

    References
    ----------

    .. [butzer88] Butzer, P., Schmidt, M., & Stark, E. (1988).
       *Observations on the History of Central B-Splines.*
       Archive for History of Exact Sciences, 39(2), 137-156. Retrieved May 14, 2020,
       from `<https://www.jstor.org/stable/41133848>`_
    .. [schoenberg73] Schoenberg, I. J. (1973). *Cardinal Spline Interpolation.*
       Society for Industrial and Applied Mathematics.
       doi:`10.1137/1.9781611970555 <https://doi.org/10.1137/1.9781611970555>`_
    """
    def __init__(self,
                 degree: int,
                 scale: float,
                 t_reference: str | pd.Timestamp,
                 regularize: bool = True,
                 time_unit: str = "D",
                 num_splines: int = 1,
                 spacing: float | None = None,
                 obs_scale: float = 1.0,
                 **model_kw_args) -> None:
        self.degree = int(degree)
        """ Degree :math:`p` of the B-Splines. """
        assert self.degree >= 0, "'degree' needs to be greater or equal to 0."
        self.order = self.degree + 1
        """ Order :math:`n=p+1` of the B-Splines. """
        self.scale = float(scale)
        """ Scale of the splines. """
        if spacing is not None:
            self.spacing = float(spacing)
            """ Spacing between the center times of the splines. """
            assert abs(self.spacing) > 0, \
                f"'spacing' must be non-zero to avoid singularities, got {self.spacing}."
        elif num_splines > 1:
            self.spacing = self.scale
        else:
            self.spacing = 0.0
        self.observability_scale = float(obs_scale)
        """ Observability scale factor. """
        if "t_start" not in model_kw_args or model_kw_args["t_start"] is None:
            model_kw_args["t_start"] = (pd.Timestamp(t_reference)
                                        - Timedelta(self.scale, time_unit)
                                        * (self.degree + 1) / 2).isoformat()
        if "t_end" not in model_kw_args or model_kw_args["t_end"] is None:
            model_kw_args["t_end"] = (pd.Timestamp(t_reference)
                                      + Timedelta(self.spacing, time_unit)
                                      * num_splines
                                      + Timedelta(self.scale, time_unit)
                                      * (self.degree + 1) / 2).isoformat()
        super().__init__(num_parameters=num_splines, t_reference=t_reference,
                         time_unit=time_unit, regularize=regularize, **model_kw_args)

    @property
    def centertimes(self) -> pd.Series:
        """ Returns a :class:`~pandas.Series` with all center times. """
        return pd.Series([self.t_reference + Timedelta(self.spacing, self.time_unit) * spl
                          for spl in range(self.num_parameters)])

    def _get_arch(self) -> dict[str, Any]:
        arch = {"type": "BSpline",
                "kw_args": {"degree": self.degree,
                            "scale": self.scale,
                            "num_splines": self.num_parameters,
                            "spacing": self.spacing,
                            "obs_scale": self.observability_scale}}
        return arch

    def get_mapping_single(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        r"""
        Calculate the mapping factors at times :math:`t` as

        .. math:: \sum_{k=0}^{n} \frac{{\left( -1 \right)}^{k}}{p!} \cdot \binom{n}{k}
                  \cdot {\left( t_j^\prime + \frac{n}{2} - k \right)}^p

        where :math:`p` is the degree and :math:`n=p+1` is the order (see [butzer88]_
        and [schoenberg73]_). :math:`t^\prime` is the normalized time:

        .. math:: t_j^\prime = \frac{ \left( t - t_{\text{ref}} \right) - j \cdot \rho}{\rho}

        where :math:`t_{\text{ref}}` and :math:`\rho` are the model's reference time and
        timescale, respectively.

        See Appendix A.2.3 in [koehne23]_ for more details.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            Coefficients of the mapping matrix.
        """
        # get relative and normalized time
        trel = (self.tvec_to_numpycol(timevector).reshape(-1, 1, 1)
                - self.scale * np.arange(self.num_parameters).reshape(1, -1, 1))
        tnorm = trel / self.scale
        # calculate coefficients efficiently all at once
        krange = np.arange(self.order + 1).reshape(1, 1, -1)
        in_power = tnorm + self.order / 2 - krange
        in_sum = ((-1)**krange * comb(self.order, krange)
                  * (in_power * (in_power >= 0))**(self.degree))
        coefs = np.sum(in_sum, axis=2) / factorial(self.degree)
        # to avoid numerical issues, set to zero manually outside of valid domains
        coefs[np.abs(tnorm.squeeze(axis=2)) > self.order / 2] = 0
        # to avoid even more numerical issues, set a basis function to zero if we only
        # observe (somewhat arbitrarily) some fraction of the spline's valid domain
        # (setting an entire column to zero will make the calling get_mapping() method
        # flag this parameter as unobservable)
        del_t = np.max(trel.squeeze(axis=2), axis=0) - np.min(trel.squeeze(axis=2), axis=0)
        set_unobservable = del_t < self.scale * self.observability_scale
        coefs[:, set_unobservable] = 0
        return coefs

    def get_transient_period(self,
                             timevector: pd.Series | pd.DatetimeIndex
                             ) -> np.ndarray:
        """
        Returns a mask-like array of where each spline is currently transient
        (not staying constant).

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            NumPy array with ``True`` when a spline is currently transient,
            ``False`` otherwise.
        """
        trel = (self.tvec_to_numpycol(timevector).reshape(-1, 1)
                - self.spacing * np.arange(self.num_parameters).reshape(1, -1))
        transient = np.abs(trel) <= self.scale * self.order
        return transient


class ISpline(Model):
    """
    Subclasses :class:`~disstans.models.Model`.

    Integral of cardinal, centralized B-Splines of certain order/degree and time scale,
    with an amplitude of 1.
    The degree :math:`p` given in the initialization is the degree of the spline
    *before* the integration, i.e. the resulting ISpline is a piecewise polynomial
    of degree :math:`p + 1`. Used for transient permanent signals that stay at their
    maximum value after a given time span.

    See Also
    --------
    disstans.models.BSpline : More details about B-Splines and the available keyword arguments.
    """
    def __init__(self,
                 degree: int,
                 scale: float,
                 t_reference: str | pd.Timestamp,
                 regularize: bool = True,
                 time_unit: str = "D",
                 num_splines: int = 1,
                 spacing: float | None = None,
                 zero_after: bool = False,
                 obs_scale: float = 1.0,
                 **model_kw_args) -> None:
        self.degree = int(degree)
        """ Degree :math:`p` of the B-Splines. """
        assert self.degree >= 0, "'degree' needs to be greater or equal to 0."
        self.order = self.degree + 1
        """ Order :math:`n=p+1` of the B-Splines. """
        self.scale = float(scale)
        """ Scale of the splines. """
        if spacing is not None:
            self.spacing = float(spacing)
            """ Spacing between the center times of the splines. """
            assert abs(self.spacing) > 0, \
                f"'spacing' must be non-zero to avoid singularities, got {self.spacing}."
        elif num_splines > 1:
            self.spacing = self.scale
        else:
            self.spacing = 0.0
        self.observability_scale = float(obs_scale)
        """ Observability scale factor. """
        if "t_start" not in model_kw_args or model_kw_args["t_start"] is None:
            model_kw_args["t_start"] = (pd.Timestamp(t_reference)
                                        - Timedelta(self.scale, time_unit)
                                        * (self.degree + 1) / 2).isoformat()
        if "t_end" not in model_kw_args or model_kw_args["t_end"] is None:
            model_kw_args["t_end"] = (pd.Timestamp(t_reference)
                                      + Timedelta(self.spacing, time_unit)
                                      * num_splines
                                      + Timedelta(self.scale, time_unit)
                                      * (self.degree + 1) / 2).isoformat()
        super().__init__(num_parameters=num_splines, t_reference=t_reference,
                         time_unit=time_unit, zero_after=zero_after,
                         regularize=regularize, **model_kw_args)

    @property
    def centertimes(self) -> pd.Series:
        """ Returns a :class:`~pandas.Series` with all center times. """
        return pd.Series([self.t_reference + Timedelta(self.spacing, self.time_unit) * spl
                          for spl in range(self.num_parameters)])

    def _get_arch(self) -> dict[str, Any]:
        arch = {"type": "ISpline",
                "kw_args": {"degree": self.degree,
                            "scale": self.scale,
                            "num_splines": self.num_parameters,
                            "spacing": self.spacing,
                            "obs_scale": self.observability_scale}}
        return arch

    def get_mapping_single(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        r"""
        Calculate the mapping factors at times :math:`t` as

        .. math:: \sum_{k=0}^{n} \frac{{\left( -1 \right)}^{k}}{\left( p+1 \right) !} \cdot
                  \binom{n}{k} \cdot {\left( t_j^\prime + \frac{n}{2} - k \right)}^{p+1}

        which is the integral over time of :meth:`~BSpline.get_mapping_single`.

        See Appendix A.2.3 in [koehne23]_ for more details.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            Coefficients of the mapping matrix.
        """
        # get relative and normalized time
        trel = (self.tvec_to_numpycol(timevector).reshape(-1, 1, 1)
                - self.scale * np.arange(self.num_parameters).reshape(1, -1, 1))
        tnorm = trel / self.scale
        # calculate coefficients efficiently all at once
        krange = np.arange(self.order + 1).reshape(1, 1, -1)
        in_power = tnorm + self.order / 2 - krange
        in_sum = ((-1)**krange * comb(self.order, krange)
                  * (in_power * (in_power >= 0))**(self.degree + 1))
        coefs = np.sum(in_sum, axis=2) / factorial(self.degree + 1)
        # to avoid numerical issues, set to zero or one manually outside of valid domains
        coefs[tnorm.squeeze(axis=2) < - self.order / 2] = 0
        coefs[tnorm.squeeze(axis=2) > self.order / 2] = 1
        # to avoid even more numerical issues, set a basis function to zero if we only
        # observe (somewhat arbitrarily) < a scale length
        # (setting an entire column to zero will make the calling get_mapping() method
        # flag this parameter as unobservable)
        del_t = np.max(trel.squeeze(axis=2), axis=0) - np.min(trel.squeeze(axis=2), axis=0)
        set_unobservable = del_t < self.scale * self.observability_scale
        coefs[:, set_unobservable] = 0
        return coefs

    def get_transient_period(self,
                             timevector: pd.Series | pd.DatetimeIndex
                             ) -> np.ndarray:
        """
        Returns a mask-like array of where each spline is currently transient
        (not staying constant).

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            NumPy array with ``True`` when a spline is currently transient, ``False`` otherwise.
        """
        trel = (self.tvec_to_numpycol(timevector).reshape(-1, 1)
                - self.spacing * np.arange(self.num_parameters).reshape(1, -1))
        transient = np.abs(trel) <= self.scale * self.order
        return transient


class BaseSplineSet(Model):
    """
    Subclasses :class:`~disstans.models.Model`.

    Raw container class for lists of splines, which are usually created with subclasses
    like :class:`~disstans.models.SplineSet`. The common functionalities are implemented
    here.

    Parameters
    ----------
    splines
        List of spline model objects.
    internal_scaling
        By default, in order to influence the tradeoff between
        splines of different timescales, the mapping matrix of each spline is scaled by its
        own time scale to promote using fewer components. Without this, there would be an
        ambiguity for the solver as to whether fit the signal using many smaller scales or
        with one large scale, as the fit would be almost identical. This behavior can be
        disabled by setting ``internal_scaling=False``.


    See :class:`~disstans.models.Model` for attribute descriptions and more keyword arguments.
    """

    def __init__(self,
                 splines: list[BSpline | ISpline],
                 internal_scaling: bool = True,
                 **model_kw_args) -> None:
        # create attributes specific to spline sets
        assert (isinstance(splines, list) and
                all([isinstance(s, BSpline) or isinstance(s, ISpline) for s in splines])), \
            f"'splines' needs to be a list of spline models, got {splines}."
        self.splines = splines
        """ List of spline object contained within the SplineSet. """
        self.internal_scaling = bool(internal_scaling)
        """ Trackes whether to scale the sub-splines relative to their lengths. """
        self.min_scale = min([m.scale for m in self.splines])
        """ Minimum scale of the sub-splines. """
        self.internal_scales = ((np.concatenate([np.array([m.scale] * m.num_parameters)
                                                for m in self.splines]) /
                                 self.min_scale)**(0.5)
                                if self.internal_scaling else None)
        """
        If :attr:`~internal_scaling` is ``True``, this NumPy array holds the relative
        scaling factors of all parameters over all the sub-splines.
        """
        # create Model object
        num_parameters = sum([s.num_parameters for s in self.splines])
        super().__init__(num_parameters=num_parameters, **model_kw_args)

    def _get_arch(self) -> dict[str, Any]:
        raise NotImplementedError("BaseSplineSet is not designed to be exported and "
                                  "created directly, use a subclass.")

    def get_mapping_single(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        r"""
        Calculate the mapping factors at times by accumulating the mapping factors
        of the different scales.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            Coefficients of the mapping matrix.
        """
        coefs = np.empty((timevector.size, self.num_parameters))
        ix_coefs = 0
        for model in self.splines:
            coefs[:, ix_coefs:ix_coefs + model.num_parameters] = \
                model.get_mapping(timevector).toarray()
            ix_coefs += model.num_parameters
        if self.internal_scaling:
            coefs *= self.internal_scales.reshape(1, self.num_parameters)
        return coefs

    def freeze(self, zero_threshold: float = 1e-10) -> None:
        """
        In case some parameters are estimated to be close to zero and should not
        be considered in future fits and evaluations, this function "freezes"
        the model by setting parameters below the threshold ``zero_threshold``
        to be invalid. The mask will be kept in
        :attr:`~disstans.models.Model.active_parameters`.

        Only valid parameters will be used by :meth:`~disstans.models.Model.get_mapping` and
        :meth:`~disstans.models.Model.evaluate`.

        Parameters
        ----------
        zero_threshold
            Model parameters with absolute values below ``zero_threshold`` will be
            set inactive.

        See Also
        --------
        unfreeze : The reverse method.
        """
        assert float(zero_threshold) > 0, \
            f"'zero_threshold needs to be non-negative, got {zero_threshold}."
        if self.par is None:
            raise RuntimeError("Cannot freeze a model without set parameters.")
        temp_par = (self.par * self.internal_scales.reshape(-1, 1) if self.internal_scaling
                    else self.par)
        self.active_parameters = np.any(np.abs(temp_par) > zero_threshold, axis=1)
        ix_params = 0
        for model in self.splines:
            model.active_parameters = \
                self.active_parameters[ix_params:ix_params + model.num_parameters]
            ix_params += model.num_parameters

    def unfreeze(self) -> None:
        """
        Resets previous model freezing done by :meth:`~freeze` such that all parameters
        are active again.
        """
        self.active_parameters = None
        for model in self.splines:
            model.active_parameters = None

    def read_parameters(self,
                        parameters: np.ndarray,
                        covariances: np.ndarray | None = None
                        ) -> None:
        r"""
        Reads in the parameters :math:`\mathbf{m}` (optionally also their variances)
        of all the sub-splines and stores them in the respective attributes.

        Note that the main ``SplineSet`` object will still contain all the cross-spline
        covariances (if contained in ``covariances``), but the sub-splines cannot.

        Parameters
        ----------
        parameters
            Model parameters of shape
            :math:`(\text{num_parameters}, \text{num_components})`.
        covariances
            Model component (co-)variances that can either have the same shape as
            ``parameters``, in which case every parameter and component only has a
            variance, or it is square with dimensions
            :math:`\text{num_parameters} * \text{num_components}`, in which case it
            represents a full variance-covariance matrix.
        """
        super().read_parameters(parameters, covariances)
        num_components = parameters.shape[1]
        if self.internal_scaling:
            parameters = parameters * self.internal_scales.reshape(-1, 1)
            if covariances is not None:
                if parameters.shape == covariances.shape:
                    covariances = covariances * self.internal_scales.reshape(-1, 1) ** 2
                else:
                    repeat_int_scales = np.repeat(self.internal_scales, num_components)
                    covariances = ((covariances * repeat_int_scales.reshape(-1, 1))
                                   * repeat_int_scales.reshape(1, -1))
        ix_params = 0
        for model in self.splines:
            param_model = parameters[ix_params:ix_params + model.num_parameters, :]
            if covariances is None:
                cov_model = None
            elif parameters.shape == covariances.shape:
                cov_model = covariances[ix_params:ix_params + model.num_parameters, :]
            else:
                ix_start = ix_params * num_components
                ix_end = ix_start + model.num_parameters * num_components
                cov_model = covariances[ix_start:ix_end, ix_start:ix_end]
            model.read_parameters(param_model, cov_model)
            ix_params += model.num_parameters

    def make_scalogram(self,
                       t_left: str | pd.Timestamp,
                       t_right: str | pd.Timestamp,
                       cmaprange: float | None = None,
                       resolution: int = 1000,
                       min_param_mag: float = 0.0
                       ) -> tuple[mpl.Figure, mpl.Axis]:
        """
        Create a scalogram figure of the model parameters.

        A scalogram shows the amplitude of each model parameter plotted over time and
        for all the different scales contained. Model parameters that have overlapping
        influence are also shown as overlapping. The height of each parameter's patch
        is defined by the weight of that parameter relative to the other parameters
        (excluding splines that are not transient at that time).

        Parameters
        ----------
        t_left
            Left boundary of the time axis.
        t_right
            Right boundary of the time axis.
        cmaprange
            Maximum absolute amplitude of the color scale to use. ``None`` defaults to
            the 95th percentile of the absolute amplitudes of all parameters.
        resolution
            Number of points inside the time span to evaluate the scalogram at.
        min_param_mag
            The absolute value under which any value is plotted as zero.

        Returns
        -------
        fig
            Figure object of the scalogram.
        ax
            Axes object of the scalogram.

        Raises
        ------
        NotImplementedError
            If the generation method for the scalogram given the SplineSet's
            spline class is not defined in this method yet.
        """
        # check input
        if self.par is None:
            RuntimeError("SplineSet model needs to have already been fitted.")
        # determine dimensions
        num_components = self.par.shape[1]
        num_scales = len(self.splines)
        dy_scale = 1 / num_scales
        t_plot = pd.Series(pd.date_range(start=t_left, end=t_right, periods=resolution))
        # get range of values (if not provided)
        if cmaprange is not None:
            assert isinstance(cmaprange, int) or isinstance(cmaprange, float), \
                "'cmaprange' must be None or a single float or integer of the " \
                f"one-sided color range of the scalogram, got {cmaprange}."
        else:
            cmaprange = np.max(np.concatenate([np.abs(model.par)
                                               for model in self.splines],
                                              axis=0).ravel())
        cmap = mpl.cm.ScalarMappable(cmap=scm.roma_r,
                                     norm=mpl.colors.Normalize(vmin=-cmaprange,
                                                               vmax=cmaprange))
        # get heights of component axes that leaves room for colorbar
        row_hr = 0.95 / num_components
        height_ratios = [row_hr] * num_components + [0.05]
        # start plotting
        fig, ax = plt.subplots(nrows=num_components + 1, constrained_layout=True,
                               gridspec_kw={"height_ratios": height_ratios})
        for i, model in enumerate(self.splines):
            # where to put this scale
            y_off = 1 - (i + 1) * dy_scale
            # get normalized values
            if isinstance(model, BSpline):
                mdl_mapping = model.get_mapping(t_plot, ignore_active_parameters=True).toarray()
            elif isinstance(model, ISpline):
                mdl_mapping = np.gradient(
                    model.get_mapping(t_plot, ignore_active_parameters=True).toarray(), axis=0)
            else:
                raise NotImplementedError("Scalogram undefined for a SplineSet of class "
                                          f"{type(model)}.")
            mdl_sum = np.sum(mdl_mapping, axis=1, keepdims=True)
            mdl_sum[mdl_sum == 0] = 1
            y_norm = np.hstack([np.zeros((t_plot.size, 1)),
                                np.cumsum(mdl_mapping / mdl_sum, axis=1)])
            # plot cell
            for j, k in product(range(model.num_parameters), range(num_components)):
                if ~np.isnan(model.par[j, k]):
                    facecol = cmap.to_rgba(model.par[j, k]
                                           if np.abs(model.par[j, k]) >= min_param_mag else 0)
                    ax[k].fill_between(t_plot,
                                       y_off + y_norm[:, j] * dy_scale,
                                       y_off + y_norm[:, j + 1] * dy_scale,
                                       facecolor=facecol, zorder=-2)
            # plot vertical lines at centerpoints
            for j, k in product(range(model.num_parameters), range(num_components)):
                ax[k].axvline(model.t_reference
                              + Timedelta(j * model.spacing, model.time_unit),
                              y_off, y_off + dy_scale, c='0.5', lw=0.5, zorder=-1)
        # finish plot by adding relevant gridlines and labels
        for k in range(num_components):
            for i in range(1, num_scales):
                ax[k].axhline(i * dy_scale, c='0.5', lw=0.5, zorder=-1)
            ax[k].set_xlim(t_left, t_right)
            ax[k].set_ylim(0, 1)
            ax[k].set_yticks([i * dy_scale for i in range(num_scales + 1)])
            ax[k].set_yticks([(i + 0.5) * dy_scale for i in range(num_scales)], minor=True)
            ax[k].set_yticklabels(reversed([f"{model.scale:.4g} {model.time_unit}"
                                           for model in self.splines]), minor=True)
            ax[k].tick_params(axis='both', labelleft=False, direction='out')
            ax[k].tick_params(axis='y', left=False, which='minor')
            ax[k].set_rasterization_zorder(0)
        fig.colorbar(cmap, cax=ax[-1], orientation='horizontal',
                     label='Coefficient Value')
        return fig, ax


class SplineSet(BaseSplineSet):
    """
    Subclasses :class:`~disstans.models.BaseSplineSet`.

    Contains a list of splines that cover an entire timespan, and that share a
    common degree, but different center times and scales.

    The set is constructed from a time span (``t_center_start`` and ``t_center_end``)
    and numbers of centerpoints or length scales. By default (``complete=True``),
    the number of splines and center points for each scale will then be chosen such
    that the resulting set of splines will be complete over the input time scale.
    This means it will contain all splines that are non-zero at least somewhere in
    the time span. Otherwise, the spline set will only have center times at or
    between ``t_center_start`` and ``t_center_end``.

    This class also sets the spacing equal to the scale.

    Parameters
    ----------
    degree
        Degree of the splines to be created.
    t_center_start
        Time span start of the spline set.
    t_center_end
        Time span end of the spline set.
    time_unit
        Time unit of scale, spacing and model parameters.
    list_scales
        List of scales to use for each of the sub-splines.
        Mutually exclusive to setting ``list_num_knots``.
    list_num_knots
        List of number of knots to divide the time span into for each of the sub-splines.
        Mutually exclusive to setting ``list_scales``.
    splineclass
        Model class to use for the splines.
    complete
        See usage description.


    See :class:`~disstans.models.BaseSplineSet` and :class:`~disstans.models.Model` for
    attribute descriptions and more keyword arguments.
    """
    def __init__(self,
                 degree: int,
                 t_center_start: str | pd.Timestamp,
                 t_center_end: str | pd.Timestamp,
                 time_unit: str = "D",
                 list_scales: list[float] | None = None,
                 list_num_knots: list[int] | None = None,
                 splineclass: BSpline | ISpline = ISpline,
                 complete: bool = True,
                 regularize: bool = True,
                 **model_kw_args
                 ) -> None:
        assert np.logical_xor(list_scales is None, list_num_knots is None), \
            "To construct a set of Splines, pass exactly one of " \
            "'list_scales' and 'list_num_knots' " \
            f"(got {list_scales} and {list_num_knots})."
        relevant_list = list_scales if list_num_knots is None else list_num_knots
        try:
            if isinstance(splineclass, str):
                splineclass = globals()[splineclass]
            assert issubclass(splineclass, Model)
        except BaseException as e:
            raise LookupError("When trying to create the SplineSet, couldn't find the model "
                              f"'{splineclass}' (expected Model type argument or string "
                              "representation of a loaded Model)."
                              ).with_traceback(e.__traceback__) from e
        # get time range
        t_center_start_tstamp = pd.Timestamp(t_center_start)
        t_center_end_tstamp = pd.Timestamp(t_center_end)
        t_center_start = t_center_start_tstamp.isoformat()
        t_center_end = t_center_end_tstamp.isoformat()
        t_range_tdelta = t_center_end_tstamp - t_center_start_tstamp
        # if a complete set is requested, we need to find the number of overlaps
        # given the degree on a single side
        num_overlaps = int(np.floor(degree / 2)) if complete else 0
        # for each scale, make a spline object
        splines = []
        obs_scale = model_kw_args.pop("obs_scale", 1)
        for elem in relevant_list:
            # Calculate the scale as float and Timedelta depending on the function call
            if list_scales is not None:
                scale_float = elem
                scale_tdelta = Timedelta(scale_float, time_unit)
            else:
                scale_tdelta = t_range_tdelta / (elem - 1)
                scale_float = scale_tdelta / Timedelta(1, time_unit)
            # find the number of center points between t_center_start and t_center_end,
            # plus the overlapping ones
            num_centerpoints = int(t_range_tdelta / scale_tdelta) + 1 + 2 * num_overlaps
            # shift the reference to be the first spline
            t_ref = t_center_start_tstamp - num_overlaps * scale_tdelta
            # create model and append
            splines.append(splineclass(degree, scale_float,
                                       num_splines=num_centerpoints,
                                       t_reference=t_ref,
                                       time_unit=time_unit,
                                       regularize=regularize,
                                       obs_scale=obs_scale))
        # set attributes
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
        """
        List of number of knots the time span is divided into for each of the sub-splines.
        """
        self.complete = complete
        """
        Sets whether the spline coverage of the time span is considered to be complete or not
        (see class documentation).
        """
        # create the BaseSplineSet and therefore Model object
        if "zero_after" not in model_kw_args:
            if splineclass == BSpline:
                model_kw_args["zero_after"] = True
            elif splineclass == ISpline:
                model_kw_args["zero_after"] = False
        super().__init__(splines=splines, time_unit=time_unit, regularize=regularize,
                         **model_kw_args)

    def _get_arch(self) -> dict[str, Any]:
        arch = {"type": "SplineSet",
                "kw_args": {"degree": self.degree,
                            "t_center_start": self.t_center_start,
                            "t_center_end": self.t_center_end,
                            "splineclass": self.splineclass.__name__,
                            "list_scales": self.list_scales,
                            "list_num_knots": self.list_num_knots,
                            "complete": self.complete,
                            "internal_scaling": self.internal_scaling}}
        return arch


class DecayingSplineSet(BaseSplineSet):
    """
    Subclasses :class:`~disstans.models.BaseSplineSet`.

    Contains a list of splines that cover a one-sided timespan, sharing a common
    degree, but with different center times and scales.

    The set is constructed from a starting center time ``t_center_start`` and
    then a list of spline scales with associated number of splines. The splines
    are repeated by the spacing in positive time. Note that if you want to
    make the spline set start at the center time, you need to specify ``t_start``
    manually.

    This class defaults to setting the spacing equal to the scale.

    Parameters
    ----------
    degree
        Degree of the splines to be created.
    t_center_start
        First center time of the spline set.
    list_scales
        List of scales to use for each of the sub-splines.
    list_num_splines
        Number of splines to create for each scale.
    time_unit
        Time unit of scale, spacing and model parameters.
    splineclass
        Model class to use for the splines.


    See :class:`~disstans.models.Model` for attribute descriptions and more keyword arguments.
    """
    def __init__(self,
                 degree: int,
                 t_center_start: str | pd.Timestamp,
                 list_scales: list[float],
                 list_num_splines: int | list[int],
                 time_unit: str = "D",
                 splineclass: BSpline | ISpline = ISpline,
                 regularize: bool = True,
                 **model_kw_args
                 ) -> None:
        # initial checks
        if isinstance(list_num_splines, int):
            list_num_splines = [list_num_splines] * len(list_scales)
        else:
            assert (isinstance(list_num_splines, list) and
                    all([isinstance(n, int) for n in list_num_splines])), \
                f"'list_num_splines' needs to be a (list of) integers, got {list_num_splines}."
        assert len(list_scales) == len(list_num_splines), \
            "'list_scales' and 'list_num_splines' need to have the same lengths, got " \
            f"{len(list_scales)} and {len(list_num_splines)} elements."
        try:
            if isinstance(splineclass, str):
                splineclass = globals()[splineclass]
            assert issubclass(splineclass, Model)
        except BaseException as e:
            raise LookupError("When trying to create the SplineSet, couldn't find the model "
                              f"'{splineclass}' (expected Model type argument or string "
                              "representation of a loaded Model)."
                              ).with_traceback(e.__traceback__) from e
        # create spline set
        t_center_start_tstamp = pd.Timestamp(t_center_start)
        splines = []
        obs_scale = model_kw_args.pop("obs_scale", 1)
        for scale_float, num_centerpoints in zip(list_scales, list_num_splines):
            splines.append(splineclass(degree, scale_float,
                                       num_splines=num_centerpoints,
                                       t_reference=t_center_start_tstamp,
                                       time_unit=time_unit,
                                       regularize=regularize,
                                       obs_scale=obs_scale))
        # save attributes
        self.degree = degree
        """ Degree of the splines. """
        self.t_center_start = t_center_start_tstamp.isoformat()
        """ Relevant time span start. """
        self.splineclass = splineclass
        """ Class of the splines contained. """
        self.list_scales = list_scales
        """ List of scales of each of the sub-splines. """
        self.list_num_splines = list_num_splines
        """ List of number of splines for each scale. """
        # create the BaseSplineSet and therefore Model object
        if "zero_after" not in model_kw_args:
            if splineclass == BSpline:
                model_kw_args["zero_after"] = True
            elif splineclass == ISpline:
                model_kw_args["zero_after"] = False
        super().__init__(splines=splines, time_unit=time_unit, regularize=regularize,
                         **model_kw_args)

    def _get_arch(self) -> dict[str, Any]:
        arch = {"type": "DecayingSplineSet",
                "kw_args": {"degree": self.degree,
                            "t_center_start": self.t_center_start,
                            "splineclass": self.splineclass.__name__,
                            "list_scales": self.list_scales,
                            "list_num_splines": self.list_num_splines,
                            "internal_scaling": self.internal_scaling}}
        return arch


class Sinusoid(Model):
    r"""
    Subclasses :class:`~disstans.models.Model`.

    This model defines a fixed-frequency periodic sinusoid signal with
    constant amplitude and phase to be estimated.

    Parameters
    ----------
    period
        Period length :math:`T` in :attr:`~disstans.models.Model.time_unit` units.


    See :class:`~disstans.models.Model` for attribute descriptions and more keyword arguments.

    Notes
    -----

    Implements the relationship

    .. math::
        \mathbf{g}(\mathbf{t}) = A \cos ( 2 \pi \mathbf{t} / T - \phi ) =
        a \cos ( 2 \pi \mathbf{t} / T ) + b \sin ( 2 \pi \mathbf{t} / T )

    with :attr:`~period` :math:`T`, :attr:`~phase` :math:`\phi=\text{atan2}(b,a)`
    and :attr:`~amplitude` :math:`A=\sqrt{a^2 + b^2}`.
    """
    def __init__(self,
                 period: float,
                 t_reference: str | pd.Timestamp,
                 time_unit: str = "D",
                 **model_kw_args
                 ) -> None:
        super().__init__(num_parameters=2, t_reference=t_reference,
                         time_unit=time_unit, **model_kw_args)
        self.period = float(period)
        """ Period of the sinusoid. """

    def _get_arch(self) -> dict[str, Any]:
        arch = {"type": "Sinusoid",
                "kw_args": {"period": self.period}}
        return arch

    def get_mapping_single(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        r"""
        Calculate the mapping factors at times :math:`t` as

        .. math:: \left( \cos \left( \omega t \right),  \sin \left( \omega t \right) \right)

        where :math:`\omega` is the period of the sinusoid.

        See Appendix A.2.2 in [koehne23]_ for more details.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            Coefficients of the mapping matrix.
        """
        dt = self.tvec_to_numpycol(timevector)
        phase = 2 * np.pi * dt / self.period
        coefs = np.stack([np.cos(phase), np.sin(phase)], axis=1)
        return coefs

    @property
    def amplitude(self) -> np.ndarray:
        """ Amplitude of the sinusoid. """
        if self.par is None:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        return np.sqrt(np.sum(self.par ** 2, axis=0))

    @property
    def phase(self) -> np.ndarray:
        """ Phase of the sinusoid. """
        if self.par is None:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        return np.arctan2(self.par[1, :], self.par[0, :])


class AmpPhModulatedSinusoid(Model):
    r"""
    Subclasses :class:`~disstans.models.Model`.

    This model defines a periodic sinusoid signal with a nominal frequency that allows
    the amplitude and phase (and therefore instantaneous frequency) to vary. This is
    accomplished by enabling the :math:`a` and :math:`b` parameters of the
    :class:`~Sinusoid` model to be time-varying.
    The functional form of these parameters is a full B-Spline basis set, defined by
    :meth:`~scipy.interpolate.BSpline.basis_element` (not a :class:`~SplineSet` of
    purely cardinal splines).

    Parameters
    ----------
    period
        Nominal period length :math:`T` in :attr:`~disstans.models.Model.time_unit` units.
    degree
        Degree :math:`p` of the B-spline to be used.
    num_bases
        Number of basis functions in the B-Spline.
        Needs to be at least ``2``.
    obs_scale
        Determines how many factors of the average scale should be sampled by the
        ``timevector`` input to :meth:`~Model.get_mapping` to accept an individual B-spline
        as observable.


    See Also
    --------
    Sinusoid : For the definition of the functional form of the sinusoid.
    """
    def __init__(self,
                 period: float,
                 degree: int,
                 num_bases: int,
                 t_start: str | pd.Timestamp,
                 t_end: str | pd.Timestamp,
                 t_reference: str | pd.Timestamp | None = None,
                 time_unit: str = "D",
                 obs_scale: float = 2.0,
                 regularize: bool = True,
                 **model_kw_args
                 ) -> None:
        # input tests
        assert num_bases > 1, "'num_bases' needs to be at least 2."
        num_parameters = 2 * num_bases
        if t_reference is None:
            t_reference = t_start
        # initialize Model
        super().__init__(num_parameters=num_parameters, t_start=t_start, t_end=t_end,
                         t_reference=t_reference, time_unit=time_unit, regularize=regularize,
                         **model_kw_args)
        # save some important parameters
        self.period = float(period)
        """ Nominal period of the sinusoid. """
        self.degree = int(degree)
        """ Degree :math:`p` of the B-Spline. """
        self.order = self.degree + 1
        """ Order :math:`n=p+1` of the B-Splines. """
        self.num_bases = int(num_bases)
        """ Number of basis functons in the B-Spline """
        self.observability_scale = float(obs_scale)
        """ Observability scale factor. """
        # define basis functions
        num_knots = int(num_bases) - int(degree) + 1
        inner_knots = np.linspace(0, 1, num=num_knots)
        full_knots = np.concatenate([[0] * self.degree, inner_knots, [1] * self.degree])
        self._bases = [sp_bspl.basis_element(full_knots[i:i + self.degree + 2],
                                             extrapolate=False)
                       for i in range(self.num_bases)]

    def _get_arch(self) -> dict[str, Any]:
        arch = {"type": "AmpPhModulatedSinusoid",
                "kw_args": {"period": self.period,
                            "degree": self.degree,
                            "num_bases": self.num_bases,
                            "obs_scale": self.observability_scale}}
        return arch

    def get_mapping_single(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        r"""
        Calculate the mapping factors at times :math:`t` as

        .. math:: \left( h_j (t) \cos \left( \omega t \right),
                  h_j (t) \sin \left( \omega t \right) \right)

        where :math:`h_j` are envelopes based on B-Splines calculated by
        :class:`~scipy.interpolate.BSpline`, and :math:`\omega` is the period.

        See Appendix A.2.4 in [koehne23]_ for more details.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            Coefficients of the mapping matrix.
        """
        # get phase and normalized [0, 1) phase
        dt = self.tvec_to_numpycol(timevector)
        phase = 2 * np.pi * dt.reshape(-1, 1) / self.period
        t_span = self.t_end - self.t_start
        phase_norm = (timevector - self.t_start) / t_span
        # get the mapping matrices of the sinusoid
        coef_cosine = np.cos(phase)
        coef_sine = np.sin(phase)
        # get the mapping matrix defined by the splines
        coef_bspl = np.stack([base_fn(phase_norm) for base_fn in self._bases], axis=1)
        coef_bspl[np.isnan(coef_bspl)] = 0
        # problem: if we're only observing a small fraction of a basis function, we don't
        # want to try to estimate it, since it's only going to make our solving process less
        # stable. so: if we only observe < obs_scale * average scale length, we ignore it.
        num_bases = len(self._bases)
        avg_scale = t_span / (num_bases - 1)
        is_nonzero = np.abs(coef_bspl) > 0
        t_min_max = np.empty((num_bases, 2))
        t_min_max[:] = np.nan
        for i in range(num_bases):
            if np.any(is_nonzero[:, i]):
                t_min_max[i, :] = [dt[is_nonzero[:, i]].min(), dt[is_nonzero[:, i]].max()]
        del_t = t_min_max[:, 1] - t_min_max[:, 0]
        set_unobservable = del_t < (self.observability_scale *
                                    avg_scale / Timedelta(1, self.time_unit))
        coef_bspl[:, set_unobservable] = 0
        # modulate the sine and cosine mapping matrix with the basis functions
        coefs = np.concatenate([coef_bspl * coef_cosine, coef_bspl * coef_sine], axis=1)
        return coefs

    def get_inst_amplitude_phase(self,
                                 num_points: int = 1000
                                 ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Calculate the instantaenous (time-varying) amplitude and phase of the sinusoid
        over its entire fitted domain.

        Parameters
        ----------
        num_points
            Number of points to use in the discretized, normalized phase vector
            used in the evaluation of the basis functions. For plotting purposes,
            this value should be the length of the timeseries to which this
            model was fitted.

        Returns
        -------
        amplitude
            Amplitude timeseries for each point (rows) and component (columns).
        phase
            Phase timeseries in radians (with the same shape as ``amplitude``).
        """
        if self.par is None:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        # get normalized phase
        phase_norm = np.linspace(0, 1 - 1e-16, num=num_points)
        # get the mapping matrix defined by the splines
        coef_bspl = np.stack([base_fn(phase_norm) for base_fn in self._bases], axis=1)
        coef_bspl[np.isnan(coef_bspl)] = 0
        # calculate the timeseries of a and b
        a_mat = (coef_bspl @ self.par[:coef_bspl.shape[1], :]).reshape(num_points, -1)
        b_mat = (coef_bspl @ self.par[coef_bspl.shape[1]:, :]).reshape(num_points, -1)
        # calculate amplitude and phase
        amplitude = np.sqrt(a_mat**2 + b_mat**2)
        phase = np.arctan2(b_mat, a_mat)
        return amplitude, phase

    @property
    def amplitude(self) -> np.ndarray:
        """ Average amplitude of the sinusoid. """
        if self.par is None:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        return np.mean(self.get_inst_amplitude_phase()[0], axis=0)

    @property
    def phase(self) -> np.ndarray:
        """ Average phase of the sinusoid. """
        if self.par is None:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        return np.mean(self.get_inst_amplitude_phase()[1], axis=0)


class Logarithmic(Model):
    r"""
    Subclasses :class:`~disstans.models.Model`.

    This model provides the "geophysical" logarithmic :math:`\ln(1 + \mathbf{t}/\tau)`
    with a given time constant and zero for :math:`\mathbf{t} < 0`.

    Parameters
    ----------
    tau
        Logarithmic time constant(s) :math:`\tau`.
        It represents the time at which, after zero-crossing at the reference
        time, the logarithm reaches the value 1 (before model scaling).
    sign_constraint
        Can be ``+1`` or ``-1``, and tells the solver to constrain fitted parameters to this
        sign, avoiding sign flips between individual logarithms. This is useful if
        the resulting curve should be monotonous. It can also be a list, where
        each entry applies to one data component (needs to be known at initialization).
        If ``None``, no constraint is enforced.


    See :class:`~disstans.models.Model` for attribute descriptions and more keyword arguments.
    """
    def __init__(self,
                 tau: float | list[float] | np.ndarray,
                 t_reference: str | pd.Timestamp,
                 sign_constraint: int | list[int] | None = None,
                 time_unit: str = "D", t_start=None,
                 zero_after: bool = False,
                 **model_kw_args
                 ) -> None:
        if t_start is None:
            t_start = t_reference
        tau = np.atleast_1d(tau)
        assert tau.ndim <= 1, "'tau' can either be a scalar or one-dimensional vector, got " \
                              f"array of shape {tau.shape}."
        self.tau = tau
        """ Logarithmic time constant(s). """
        assert ((sign_constraint in [1, -1, None]) or
                (isinstance(sign_constraint, list) and
                 all([s in [1, -1, None] for s in sign_constraint]))), "'sign_constraint' " \
            f"must be None, -1, 1, or a list of None, -1, and 1, got {sign_constraint}."
        self.sign_constraint = sign_constraint
        """
        Flag whether the sign of the fitted parameters should be constrained.
        """
        # initialize Model object
        super().__init__(num_parameters=tau.size, t_reference=t_reference, t_start=t_start,
                         time_unit=time_unit, zero_after=zero_after, **model_kw_args)
        assert self.t_reference <= self.t_start, \
            "Logarithmic model has to have valid bounds, but the reference time " + \
            f"{self.t_reference_str} is after the start time {self.t_start_str}."

    def _get_arch(self) -> dict[str, Any]:
        tau = self.tau.tolist() if self.tau.size > 1 else self.tau[0]
        arch = {"type": "Logarithmic",
                "kw_args": {"tau": tau,
                            "sign_constraint": self.sign_constraint}}
        return arch

    def get_mapping_single(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        r"""
        Calculate the mapping factors at times :math:`t` as

        .. math:: \log \left( 1 + \frac{t}{\tau} \right)

        where :math:`\tau` is the logairthmic time constant.

        See Appendix A.2.2 in [koehne23]_ for more details.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            Coefficients of the mapping matrix.
        """
        dt = self.tvec_to_numpycol(timevector)
        coefs = np.log1p(dt.reshape(-1, 1) / self.tau.reshape(1, -1))
        return coefs


class Exponential(Model):
    r"""
    Subclasses :class:`~disstans.models.Model`.

    This model provides the "geophysical" exponential :math:`1-\exp(-\mathbf{t}/\tau)`
    with a given time constant, zero for :math:`\mathbf{t} < 0`, and approaching
    one asymptotically.

    Parameters
    ----------
    tau
        Exponential time constant(s) :math:`\tau`.
        It represents the amount of time that it takes for the (general) exponential
        function's value to be multiplied by :math:`e`.
        Applied to this model, for a given relative amplitude :math:`a` (so :math:`0 < a < 1`,
        before model scaling) to be reached at given :math:`\Delta t` past ``t_start``,
        :math:`\tau = - \frac{\Delta t}{\ln(1 - a)}`
    sign_constraint
        Can be ``+1`` or ``-1``, and tells the solver to constrain fitted parameters to this
        sign, avoiding sign flips between individual exponentials. This is useful if
        the resulting curve should be monotonous. It can also be a list, where
        each entry applies to one data component (needs to be known at initialization).
        If ``None``, no constraint is enforced.


    See :class:`~disstans.models.Model` for attribute descriptions and more keyword arguments.
    """
    def __init__(self,
                 tau: float | list[float] | np.ndarray,
                 t_reference: str | pd.Timestamp,
                 sign_constraint: int | list[int] | None = None,
                 time_unit: str = "D",
                 t_start: str | pd.Timestamp | None = None,
                 zero_after: bool = False,
                 **model_kw_args
                 ) -> None:
        if t_start is None:
            t_start = t_reference
        tau = np.atleast_1d(tau)
        assert tau.ndim <= 1, "'tau' can either be a scalar or one-dimensional vector, got " \
                              f"array of shape {tau.shape}."
        self.tau = tau
        """ Exponential time constant(s). """
        assert ((sign_constraint in [1, -1, None]) or
                (isinstance(sign_constraint, list) and
                 all([s in [1, -1, None] for s in sign_constraint]))), "'sign_constraint' " \
            f"must be None, -1, 1, or a list of None, -1, and 1, got {sign_constraint}."
        self.sign_constraint = sign_constraint
        """
        Flag whether the sign of the fitted parameters should be constrained.
        """
        # initialize Model object
        super().__init__(num_parameters=tau.size, t_reference=t_reference, t_start=t_start,
                         time_unit=time_unit, zero_after=zero_after, **model_kw_args)
        assert self.t_reference <= self.t_start, \
            "Exponential model has to have valid bounds, but the reference time " + \
            f"{self.t_reference_str} is after the start time {self.t_start_str}."

    def _get_arch(self) -> dict[str, Any]:
        tau = self.tau.tolist() if self.tau.size > 1 else self.tau[0]
        arch = {"type": "Exponential",
                "kw_args": {"tau": tau,
                            "sign_constraint": self.sign_constraint}}
        return arch

    def get_mapping_single(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        r"""
        Calculate the mapping factors at times :math:`t` as

        .. math:: \left( 1 - \exp \left( -\frac{t}{\tau} \right) \right)

        where :math:`\tau` is the exponential time constant.

        See Appendix A.2.2 in [koehne23]_ for more details.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            Coefficients of the mapping matrix.
        """
        dt = self.tvec_to_numpycol(timevector)
        coefs = 1 - np.exp(-dt.reshape(-1, 1) / self.tau.reshape(1, -1))
        return coefs


class Arctangent(Model):
    r"""
    Subclasses :class:`~disstans.models.Model`.

    This model provides the arctangent :math:`\arctan(\mathbf{t}/\tau)`,
    stretched with a given time constant and normalized to be between
    :math:`(0, 1)` at the limits.

    Because this model is usually transient, it is recommended not to
    use it in the estimation of parameters, even when using ``t_start``
    and ``t_end`` to make the tails constant (since that introduces high-frequency
    artifacts). (Using ``t_start`` and/or ``t_end`` might be desirable for creating
    syntehtic data, however.)

    Parameters
    ----------
    tau
        Arctangent time constant :math:`\tau`.
        It represents the time at which, after zero-crossing at the reference
        time, the arctangent reaches the value :math:`\pi/4` (before model scaling),
        i.e. half of the one-sided amplitude.


    See :class:`~disstans.models.Model` for attribute descriptions and more keyword arguments.
    """
    def __init__(self,
                 tau: float,
                 t_reference: str | pd.Timestamp,
                 time_unit: str = "D",
                 zero_before: bool = False,
                 zero_after: bool = False,
                 **model_kw_args
                 ) -> None:
        super().__init__(num_parameters=1, t_reference=t_reference, time_unit=time_unit,
                         zero_before=zero_before, zero_after=zero_after, **model_kw_args)
        self.tau = float(tau)
        """ Arctangent time constant. """

    def _get_arch(self) -> dict[str, Any]:
        arch = {"type": "Arctangent",
                "kw_args": {"tau": self.tau}}
        return arch

    def get_mapping_single(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        r"""
        Calculate the mapping factors at times :math:`t` as

        .. math:: \left( \frac{1}{\pi} \arctan \left( \frac{t}{\tau} \right) + 0.5 \right)

        where :math:`\tau` is the arctangent time constant.

        See Appendix A.2.2 in [koehne23]_ for more details.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            Coefficients of the mapping matrix.
        """
        dt = self.tvec_to_numpycol(timevector)
        coefs = np.arctan(dt / self.tau).reshape(-1, 1) / np.pi + 0.5
        return coefs


class HyperbolicTangent(Model):
    r"""
    Subclasses :class:`~disstans.models.Model`.

    This model provides the hyprbolic tangent :math:`\arctan(\mathbf{t}/\tau)`,
    stretched with a given time constant and normalized to be between
    :math:`(0, 1)` at the limits.

    Parameters
    ----------
    tau
        Time constant :math:`\tau`.
        To determine the constant from a characteristic time scale :math:`T` and a
        percentage :math:`0<q<1` of the fraction of magnitude change to have happened
        in that time scale (as counted in both directions from the reference time,
        and given the specified time unit), use the following formula:
        :math:`\tau = T / \left(2 \tanh^{-1} q \right)`.


    See :class:`~disstans.models.Model` for attribute descriptions and more keyword arguments.
    """
    def __init__(self,
                 tau: float,
                 t_reference: str | pd.Timestamp,
                 time_unit: str = "D",
                 zero_before: bool = False,
                 zero_after: bool = False,
                 **model_kw_args
                 ) -> None:
        super().__init__(num_parameters=1, t_reference=t_reference, time_unit=time_unit,
                         zero_before=zero_before, zero_after=zero_after, **model_kw_args)
        self.tau = float(tau)
        """ Time constant. """

    def _get_arch(self) -> dict[str, Any]:
        arch = {"type": "HyperbolicTangent",
                "kw_args": {"tau": self.tau}}
        return arch

    def get_mapping_single(self, timevector: pd.Series | pd.DatetimeIndex) -> np.ndarray:
        r"""
        Calculate the mapping factors at times :math:`t` as

        .. math:: \left( \frac{1}{2} \tanh \left( \frac{t}{\tau} \right) + 0.5 \right)

        where :math:`\tau` is the hyperbolic tangent time constant.

        See Appendix A.2.2 in [koehne23]_ for more details.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.

        Returns
        -------
            Coefficients of the mapping matrix.
        """
        dt = self.tvec_to_numpycol(timevector)
        coefs = np.tanh(dt / self.tau).reshape(-1, 1) / 2 + 0.5
        return coefs


def check_model_dict(models: dict[str, dict]) -> None:
    """
    Checks whether a dictionary has the appropriate structure to be used to
    create :class:`~Model` objects.

    Parameters
    ----------
    models
        Dictionary of structure ``{model_name: {"type": modelclass, "kw_args":
        {**kw_args}}}`` that contains the names, types and necessary keyword arguments
        to create each model object.

    Raises
    ------
    AssertionError
        If the dictionary structure is invalid.
    """
    assert isinstance(models, dict), \
        f"'models' input needs to be a dictionary, got {type(models)}."
    assert all([isinstance(mdl_name, str) for mdl_name in models.keys()]), \
        f"Model names need to be strings, got {models.keys()}."
    assert all([isinstance(mdl_config, dict) for mdl_config in models.values()]), \
        f"Model configurations need to be dictionaries, got {models.keys()}."
    for mdl_name, mdl_config in models.items():
        assert all([key in mdl_config.keys() for key in ["type", "kw_args"]]), \
            f"The configuration dictionary for '{mdl_name}' needs to contain " \
            f"the keys 'type' and 'kw_args', got {mdl_config.keys()}."
        assert isinstance(mdl_config["type"], str), \
            f"'type' in configuration dictionary for '{mdl_name}' needs to be " \
            f"a string, got {mdl_config['type']}."
        assert isinstance(mdl_config["kw_args"], dict), \
            f"'kw_args' in configuration dictionary for '{mdl_name}' needs to be " \
            f"a dictionary, got {mdl_config['kw_args']}."


# make a custom object that serves as the "all models" fit key
class FitCollection(UserDict):
    """
    Class that contains :class:`~disstans.timeseries.Timeseries` model fits,
    and can be used just like a :class:`~dict`.
    Has an additional :attr:`~allfits` attribute that stores the sum of all
    fits.
    """

    def __init__(self, *args, **kw_args) -> None:
        super().__init__(*args, **kw_args)
        self.allfits = None
        """
        Attribute that contains the sum of all fits in :class:`~FitCollection`.
        """


class ModelCollection():
    """
    Class that contains :class:`~Model` objects and is mainly used to keep track
    of across-model variables and relations such as the cross-model covariances.
    It also contains convenience functions that wrap individual models' functions like
    :meth:`~disstans.models.Model.evaluate`, :meth:`~disstans.models.Model.get_mapping`
    or :meth:`~disstans.models.Model.read_parameters` or attributes like
    :attr:`~disstans.models.Model.par`.
    """

    EVAL_PREDVAR_PRECISION = np.dtype(np.single)
    """
    To reduce memory impact when estimating the full covariance of the predicted
    timeseries when calling :meth:`~evaluate`, this attribute is by default set to
    single precision, but can be changed to double precision if desired.
    """

    def __init__(self) -> None:
        self.collection = {}
        """
        Dictionary of :class:`~Model` objects contained in this collection.
        """
        self._par = None
        self._cov = None

    @classmethod
    def from_model_dict(cls, model_dict: dict[str, Model]) -> ModelCollection:
        """
        Creates an empty :class:`~ModelCollection` object, and adds a dictionary of
        model objects to it.

        Parameters
        ----------
        model_dict
            Dictionary with model names as keys, and :class:`~Model` object instances
            as values.

        Return
        ------
            The new :class:`~ModelCollection` object.
        """
        coll = cls()
        for model_description, model in model_dict.items():
            coll[model_description] = model
        return coll

    def __getitem__(self, model_description: str) -> Model:
        """
        Convenience special function to the models contained in :attr:`~collection`.
        """
        if model_description not in self.collection:
            raise KeyError(f"No model '{model_description}' present in collection.")
        return self.collection[model_description]

    def __setitem__(self, model_description: str, model: Model) -> None:
        """
        Convenience special function to add or update a model contained in
        :attr:`~collection`. Setting or updating a model forces the collection's
        parameter and covariance matrices to be reset to ``None``.
        """
        assert isinstance(model_description, str) and isinstance(model, Model), \
            "'model_description' needs to be a string and 'model' needs to be a Model, " \
            f"got {(type(model_description), type(model))}."
        if model_description in self.collection:
            warn(f"ModelCollection: Overwriting model '{model_description}'.",
                 category=RuntimeWarning, stacklevel=2)
        self._par = None
        self._cov = None
        self.collection[model_description] = model

    def __delitem__(self, model_description: str) -> None:
        """
        Convenience special function to delete a model contained in
        :attr:`~collection`. Deleting a model forces the collection's parameter and
        covariance matrices to be reset to ``None``.
        """
        self._par = None
        self._cov = None
        del self.collection[model_description]

    def __iter__(self) -> Iterator[Model]:
        """
        Convenience special function that allows for a shorthand notation to quickly
        iterate over all models in :attr:`~collection`.

        Example
        -------
        If ``mc`` is a :class:`~ModelCollection` instance, then the following
        two loops are equivalent::

            # long version
            for model in mc.collection.values():
                pass
            # shorthand
            for model in mc:
                pass
        """
        for model in self.collection.values():
            yield model

    def __len__(self) -> int:
        """
        Special function that gives quick access to the number of models
        in the collection using Python's built-in ``len()`` function
        to make interactions with iterators easier.
        """
        return len(self.collection)

    def __contains__(self, model_description: str) -> bool:
        """
        Special function that allows to check whether a certain model description
        is in the collection.

        Example
        -------
        If ``mc`` is a :class:`~ModelCollection` instance, and we want to check whether
        ``'mymodel'`` is a model in the collection, the following two are equivalent::

            # long version
            'mymodel' in mc.collection
            # short version
            'mymodel' in mc
        """
        return model_description in self.collection

    def __str__(self) -> str:
        """
        Special function that returns a readable summary of the model collection.
        Accessed, for example, by Python's ``print()`` built-in function.

        Returns
        -------
            Model collection summary.
        """
        info = f"ModelCollection ({self.num_parameters} parameters)"
        for k, v in self.collection.items():
            info += f"\n  {k + ':':<15}{v.get_arch()['type']}"
        return info

    def __eq__(self, other: Model) -> bool:
        """
        Special function that allows for the comparison of model collection based on
        their contents, regardless of model parameters.

        Parameters
        ----------
        other
            Model collection to compare to.

        See Also
        --------
        disstans.models.Model.__eq__ : For more details.
        """
        return self.get_arch() == other.get_arch()

    def get_arch(self) -> dict:
        """
        Get a dictionary that describes the model collection fully and allows it to
        be recreated.

        Returns
        -------
            Model keyword dictionary.
        """
        arch = {"type": "ModelCollection",
                "collection": {k: v.get_arch() for k, v in self.collection.items()}}
        return arch

    def items(self) -> ItemsView:
        """
        Convenience function that returns a key-value-iterator from :attr:`~collection`.
        """
        return self.collection.items()

    def copy(self,
             parameters: bool = True,
             covariances: bool = True,
             active_parameters: bool = True
             ) -> ModelCollection:
        """
        Copy the model collection object.

        Parameters
        ----------
        parameters
            If ``True``, include the read-in parameters in the copy
            (:attr:`~par`), otherwise leave empty.
        covariances
            If ``True``, include the read-in (co)variances in the copy
            (:attr:`~cov`), otherwise leave empty.
        active_parameters
            If ``True``, include the active parameter setting in the copy
            (:attr:`~active_parameters`), otherwise leave empty.

        Returns
        -------
            A copy of the model collection, based on the individual models'
            :meth:`~disstans.models.Model.copy` method.
        """
        return ModelCollection.from_model_dict(
            {mdl_desc: mdl.copy(parameters=parameters, covariances=covariances,
                                active_parameters=active_parameters)
             for mdl_desc, mdl in self.collection.items()})

    def convert_units(self, factor: float) -> None:
        """
        Convert the parameter and covariances to a new unit by providing a
        conversion factor.

        Parameters
        ----------
        factor
            Factor to multiply the parameters by to obtain the parameters in the new units.
        """
        # input checks
        try:
            factor = float(factor)
        except TypeError as e:
            raise TypeError(f"'factor' and has to be a scalar, got {type(factor)}."
                            ).with_traceback(e.__traceback__) from e
        # convert parameters
        if self._par is not None:
            self._par *= factor
        # convert covariances
        if self._cov is not None:
            self._cov *= factor**2
        # update individual models
        for model in self.collection.values():
            model.convert_units(factor)

    @property
    def num_parameters(self) -> int:
        """
        Number of parameters in the model collection, calculated as the sum of all
        the parameters in the contained models.
        """
        return sum([m.num_parameters for m in self])

    @property
    def model_names(self) -> list[str]:
        """
        List of all model names.
        """
        return list(self.collection.keys())

    @property
    def num_regularized(self) -> int:
        """
        Number of all regularized parameters.
        """
        return sum(self.regularized_mask)

    @property
    def regularized_mask(self) -> np.ndarray:
        r"""
        A boolean array mask of shape :math:`(\text{num_parameters}, )`
        where ``True`` denotes a regularized parameter (``False`` otherwise``).
        """
        regularized_mask = []
        for m in self:
            regularized_mask.extend([m.regularize] * m.num_parameters)
        return regularized_mask

    @property
    def internal_scales(self) -> np.ndarray:
        r"""
        Array of shape :math:`(\text{num_parameters}, )` that collects all the
        models' internal scales.
        """
        if len(self) > 0:
            internal_scales = []
            for m in self:
                try:
                    if m.internal_scales is None:
                        internal_scales.append(np.ones(m.num_parameters))
                    else:
                        internal_scales.append(m.internal_scales)
                except AttributeError:
                    internal_scales.append(np.ones(m.num_parameters))
            return np.concatenate(internal_scales)

    @property
    def active_parameters(self) -> np.ndarray:
        r"""
        Either ``None``, if all parameters are active, or an array of shape
        :math:`(\text{num_parameters}, )` that contains ``True`` for all active
        parameters, and ``False`` otherwise.
        """
        if len(self) > 0:
            active_params = [m.active_parameters if m.active_parameters is not None
                             else np.ones(m.num_parameters, dtype=bool) for m in self]
            active_params = np.concatenate(active_params)
            if np.all(active_params):
                return None
            else:
                return active_params

    @property
    def par(self) -> np.ndarray:
        r"""
        Array property of shape :math:`(\text{num_parameters}, \text{num_components})`
        that contains the parameters as a NumPy array.
        """
        test_par = [m.par for m in self]
        test_par_anynone = any([p is None for p in test_par])
        if (self._par is None) and not test_par_anynone:
            warn("Discrepancy between ModelCollection parameters and individual "
                 "parameters: collection is not fitted.", stacklevel=2)
        elif self._par is not None:
            assert self._par.shape[0] == self.num_parameters, \
                "Saved parameter matrix does not match the model list in the collection."
            if test_par_anynone:
                warn("Discrepancy between ModelCollection parameters and individual "
                     "parameters: individual models are not fitted.", stacklevel=2)
            else:
                test_par = np.concatenate(test_par, axis=0)
                if not np.allclose(self._par, test_par):
                    warn("Discrepancy between ModelCollection parameters and individual "
                         "parameters: not matching (returning collection values).",
                         stacklevel=2)
        return self._par

    @property
    def parameters(self) -> np.ndarray:
        """ Alias for :attr:`~par`. """
        return self.par

    @property
    def var(self) -> np.ndarray:
        r"""
        Array property of shape :math:`(\text{num_parameters}, \text{num_components})`
        that returns the parameter's individual variances as a NumPy array.
        """
        if self.cov is None:
            return None
        else:
            return np.diag(self.cov).reshape(self.num_parameters, -1)

    @property
    def cov(self) -> np.ndarray:
        r"""
        Square array property with dimensions
        :math:`\text{num_elements} * \text{num_components}` that contains the parameter's
        full covariance matrix as a NumPy array. The rows (and columns) are ordered such
        that they first correspond to the covariances between all components for the first
        parameter, then the covariance between all components for the second parameter,
        and so forth.
        """
        test_cov = [m.cov for m in self]
        test_cov_anynone = any([p is None for p in test_cov])
        if (self._cov is None) and not test_cov_anynone:
            warn("Discrepancy between ModelCollection covariance and individual "
                 "covariances: collection is not fitted.", stacklevel=2)
        elif self._cov is not None:
            par_size = self.par.shape[0] * self.par.shape[1]
            assert self._cov.shape == (par_size, par_size), \
                "Saved covariance matrix does not match the model list in the collection."
            if test_cov_anynone:
                warn("Discrepancy between ModelCollection covariance and individual "
                     "covariances: individual models are not fitted.", stacklevel=2)
            else:
                test_cov = sp.linalg.block_diag(*test_cov).ravel()
                test_cov_nonzero = np.nonzero(test_cov)
                test_cov = test_cov[test_cov_nonzero]
                cov_nooffdiag = self._cov.ravel()[test_cov_nonzero]
                if not np.allclose(cov_nooffdiag, test_cov, equal_nan=True):
                    warn("Discrepancy between ModelCollection covariance and individual "
                         "covariance: not matching (returning collection values).",
                         stacklevel=2)
        return self._cov

    @property
    def covariances(self) -> np.ndarray:
        """ Alias for :attr:`~cov`. """
        return self.cov

    def freeze(self,
               model_list: list[str] | None = None,
               zero_threshold: float = 1e-10
               ) -> None:
        """
        Convenience function that calls :meth:`~disstans.models.Model.freeze` for all
        models (or a subset thereof) contained in the collection.

        Parameters
        ----------
        model_list
            If ``None``, freeze all models. If a list of strings, only
            freeze the corresponding models in the collection.
        zero_threshold
            Model parameters with absolute values below ``zero_threshold`` will be
            set to zero and set inactive.
        """
        if model_list is not None:
            assert (isinstance(model_list, list)
                    and all([isinstance(mdl, str) for mdl in model_list])), \
                f"'model_list' needs to be a list of strings, got {model_list}."
        for model in [mdl for mdl_description, mdl in self.items()
                      if (model_list is None) or (mdl_description in model_list)]:
            model.freeze(zero_threshold)

    def unfreeze(self, model_list: list[str] | None = None) -> None:
        """
        Convenience function that calls :meth:`~disstans.models.Model.unfreeze` for all
        models (or a subset thereof) contained in the collection.

        Parameters
        ----------
        model_list
            If ``None``, unfreeze all models. If a list of strings, only
            unfreeze the corresponding models in the collection.
        """
        if model_list is not None:
            assert (isinstance(model_list, list)
                    and all([isinstance(mdl, str) for mdl in model_list])), \
                f"'model_list' needs to be a list of strings, got {model_list}."
        for model in [mdl for mdl_description, mdl in self.items()
                      if (model_list is None) or (mdl_description in model_list)]:
            model.unfreeze()

    # "inherit" the read_parameter function from the Model class
    _read_parameters = Model.read_parameters

    def read_parameters(self,
                        parameters: np.ndarray,
                        covariances: np.ndarray | None = None
                        ) -> None:
        r"""
        Reads in the entire collection's parameters :math:`\mathbf{m}` (optionally also
        their covariance) and stores them in the instance attributes.

        Parameters
        ----------
        parameters
            Model collection parameters of shape
            :math:`(\text{num_parameters}, \text{num_components})`.
        covariances
            Model collection component (co-)variances that can either have the same shape
            as ``parameters``, in which case every parameter and component only has a
            variance, or it is square with dimensions
            :math:`\text{num_parameters} * \text{num_components}`, in which case it
            represents a full variance-covariance matrix.
        """
        # read as if the collection was a model for itself
        self._read_parameters(parameters, covariances)
        # distribute to models (they should get just sliced views of the full data)
        ix_params = 0
        for model in self:
            # get parameters slice
            if self._par is None:
                param_model = None
            else:
                param_model = self._par[ix_params:ix_params + model.num_parameters, :]
            # get covariance slice
            if (self._par is None) or (self._cov is None):
                cov_model = None
            else:
                ix_start = ix_params * self._par.shape[1]
                ix_end = ix_start + model.num_parameters * self._par.shape[1]
                cov_model = self._cov[ix_start:ix_end, ix_start:ix_end]
            # pass to model and advance
            model.read_parameters(param_model, cov_model)
            ix_params += model.num_parameters

    # "inherit" the evaluate function from the Model class
    evaluate = Model.evaluate

    def get_mapping(self,
                    timevector: pd.Series | pd.DatetimeIndex,
                    return_observability: bool = False,
                    ignore_active_parameters: bool = False
                    ) -> sparse.csc_matrix | tuple[sparse.csc_matrix, np.ndarray]:
        r"""
        Builds the mapping matrix :math:`\mathbf{G}` given a time vector :math:`\mathbf{t}`
        by concatenating the individual mapping matrices from each contained model using
        their method :meth:`~disstans.models.Model.get_mapping` (see for more details).

        This method respects the parameters being set invalid by :meth:`~freeze`, and will
        interpret those parameters to be unobservable.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.
        return_observability
            If true, the function will check if there are any all-zero columns, which
            would point to unobservable parameters, and return a boolean mask with the
            valid indices.
        ignore_active_parameters
            If ``True``, do not set inactive parameters to zero to avoid estimation.

        Returns
        -------
        mapping
            Sparse mapping matrix.
        observable
            Returned if ``return_observability=True``.
            A boolean NumPy array of the same length as ``mapping`` has columns.
            ``False`` indicates (close to) all-zero columns (unobservable parameters).
        """
        mappings = [m.get_mapping(timevector,
                                  return_observability=return_observability,
                                  ignore_active_parameters=ignore_active_parameters)
                    for m in self]
        if return_observability:
            mapping = sparse.hstack([m[0] for m in mappings], format='csc')
            observable = np.concatenate([m[1] for m in mappings], axis=0)
            return mapping, observable
        else:
            mapping = sparse.hstack(mappings, format='csc')
            return mapping

    def prepare_LS(self,
                   ts: Timeseries,
                   include_regularization: bool = True,
                   reweight_init:
                       np.ndarray | list[np.ndarray] | dict[str, np.ndarray] | None = None,
                   use_internal_scales: bool = False,
                   check_constraints: bool = False
                   ) -> tuple[sparse.spmatrix, np.ndarray, int, int, int, int,
                              int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Helper function that concatenates the mapping matrices of the collection
        models given the timevector in in the input timeseries, and returns some
        relevant sizes.

        It can also combine the regularization masks and reshape the input weights
        into the format used by the solvers, optionally taking into account
        the internal scales.

        Parameters
        ----------
        ts
            The timeseries whose time indices are used to calculate the mapping matrix.
        include_regularization
            If ``True``, expands the returned variables (see below) and computes
            the regularization mask for the observable parameters only.
        reweight_init
            Contains the initial weights for the current iteration of the least squares
            problem. It can be a Numpy array or a list of Numpy arrays, in which case it
            (or the array created by concatenating the list) need to already have the right
            output shape (no check is performed). If it is a dictionary, the keys need to be
            model names, and the values are then the Numpy arrays which will be arranged
            properly to match the mapping matrix.
        use_internal_scales
            If ``True``, also return the internal model scales,
            subset to the observable and regularized parameters.
        check_constraints
            If ``True``, also return an array that contains the signs
            that should be enforced for the parameters.

        Returns
        -------
        G
            Mapping matrix computed by :meth:`~get_mapping`.
        obs_mask
            Observability mask computed by :meth:`~get_mapping`.
        num_time
            Length of the timeseries.
        num_params
            Number of total parameters present in the model collection.
        num_comps
            Number of components in the timeseries.
        num_obs
            Number of observable parameters.
        num_reg
            (Only if ``include_regularization=True``.)
            Number of observable and regularized parameters.
        reg_mask
            (Only if ``include_regularization=True``.)
            Numpy array of shape :math:`(\text{num_obs}, )` that for each observable
            parameter denotes whether that parameter is regularized (``True``) or not.
        init_weights
            (Only if ``include_regularization=True``.)
            Numpy array of shape :math:`(\text{num_reg}, )` that for each observable
            and regularized parameter contains the initial weights.
            ``None`` if ``reweight_init=None``.
        weights_scaling
            (Only if ``include_regularization=True``.)
            Numpy array of shape :math:`(\text{num_reg}, )` that for each observable
            and regularized parameter contains the internal model scale.
            ``None`` if ``use_internal_scales=False``.
        sign_constraints
            (Only if ``check_constraints=True``.)
            Numpy array of shape :math:`(\text{num_obs}, \text{num_comps})` that for
            each observable parameter denotes whether it should be positive (``+1``),
            negative (``-1``), or unconstrained (``NaN``).

        See Also
        --------
        build_LS : Function that follows this one in the regular solving process.
        """
        # G has shape (ts.time.size, self.num_parameters)
        # obs_mask has shape (self.num_parameters, ) and is True if a parameter
        # is observable
        G, obs_mask = self.get_mapping(ts.time, return_observability=True)
        num_time, num_params = G.shape
        assert num_params > 0, f"Mapping matrix is empty, has shape {G.shape}."
        assert num_params == obs_mask.size, "Parameter and observability mismatch."
        num_obs = obs_mask.sum()
        assert num_obs > 0, "Mapping matrix has no observable parameters."
        num_comps = ts.num_components
        return_vals = [G, obs_mask, num_time, num_params, num_comps, num_obs]
        if include_regularization:
            # reg_mask_full has shape (self.num_parameters, ) and is True if
            # a parameter should be regularized
            reg_mask_full = self.regularized_mask
            # now, we need the regularization mask reg_mask for the reduced G matrix, which
            # will only contain observable columns, has therefore shape (sum(obs_mask), )
            reg_mask = np.array(reg_mask_full)[obs_mask]
            # num_reg is the number of parameters that are both observable and regularized
            num_reg = sum(reg_mask)
            if num_reg == 0:
                warn("Regularized solver got no models to regularize.", stacklevel=2)
            # if use_internal_scales, we still need the scales for all the observable and
            # regularized parameters
            # self.internal_scales has shape (self.num_parameters, ) so we can use the
            # combination of obs_mask and reg_mask_full to find the relevant subset
            if use_internal_scales:
                weights_scaling = self.internal_scales[np.logical_and(reg_mask_full, obs_mask)]
            else:
                weights_scaling = None
            # if there are initial weights, distribute those
            if reweight_init is None:
                init_weights = None
            elif isinstance(reweight_init, np.ndarray):
                init_weights = reweight_init
            elif isinstance(reweight_init, list):
                init_weights = np.concatenate(init_weights)
            elif isinstance(reweight_init, dict):
                # concatenace the sub-arrays in the right order, and fill with empty
                # weights (temporarily) in between, so that we can use the global obs_mask
                init_weights = []
                for mdl_description, model in self.items():
                    if model.regularize and mdl_description in reweight_init:
                        init_weights.append(reweight_init[mdl_description])
                    else:
                        temp_weights = np.empty((model.num_parameters, num_comps))
                        temp_weights[:] = np.nan
                        init_weights.append(temp_weights)
                init_weights = \
                    np.concatenate(init_weights)[np.logical_and(reg_mask_full, obs_mask)]
                # this should not contain any NaNs anymore
                assert np.isnan(init_weights).ravel().sum() == 0
            else:
                raise ValueError("Unrecognized input for 'reweight_init'.")
            if init_weights is not None:
                assert init_weights.shape == (num_reg, num_comps), \
                    "The combined 'reweight_init' must have the shape " + \
                    f"{(num_reg, num_comps)}, got {reweight_init.shape}."
            return_vals.extend([num_reg, reg_mask, init_weights, weights_scaling])
        if check_constraints:
            # check for constraints
            sign_constraints = np.empty((num_params, num_comps))
            sign_constraints[:] = np.nan
            i = 0
            for mdl_name, model in self.items():
                try:
                    # one sign is broadcasted, or the list is inserted into the array
                    sign_constraints[i:i + model.num_parameters, :] = model.sign_constraint
                except AttributeError:  # model hasn't implemented constraints, just skip
                    pass
                except ValueError as e:  # list doesn't match number of components
                    raise ValueError(f"The sign constraint in the model '{mdl_name}' is "
                                     "either not a scalar, or the list length does not "
                                     f"match the number of data components ({num_comps})."
                                     ).with_traceback(e.__traceback__) from e
                i += model.num_parameters
            # subset to observables
            sign_constraints = sign_constraints[obs_mask, :]
            return_vals.append(sign_constraints)
        # return all collected outputs
        return tuple(return_vals)

    @staticmethod
    def build_LS(ts: Timeseries,
                 G: sparse.spmatrix,
                 obs_mask: np.ndarray,
                 icomp: int | None = None,
                 return_W_G: bool = False,
                 use_data_var: bool = True,
                 use_data_cov: bool = True
                 ) -> tuple[sparse.spmatrix, sparse.spmatrix, np.ndarray, np.ndarray]:
        r"""
        Helper function that builds the necessary matrices to solve the
        least-squares problem for the observable parameters given observations.

        If the problem only attempts to solve a single data component (by specifying
        its index in ``icomp``), it simply takes the input mapping matrix :math:`\mathbf{G}`,
        creates the weight matrix :math:`\mathbf{W}`, and computes
        :math:`\mathbf{G}^T \mathbf{W} \mathbf{G}` as well as
        :math:`\mathbf{G}^T \mathbf{W} \mathbf{d}`.

        If the problem is joint, i.e. there are multiple data components with covariance
        between them, this function brodcasts the mapping matrix to the components,
        creates the multi-component weight matrix, and then computes those same
        :math:`\mathbf{G}^T \mathbf{W} \mathbf{G}` and
        :math:`\mathbf{G}^T \mathbf{W} \mathbf{d}` matrices.

        Parameters
        ----------
        ts
            The timeseries whose time indices are used to calculate the mapping matrix.
        G
            Single-component mapping matrix.
        obs_mask
            Observability mask.
        icomp
            If provided, the integer index of the component of the data to be fitted.
        return_W_G
            If ``True``, also return the :math:`\mathbf{G}` and
            :math:`\mathbf{W}` matrices, reduced to the observable parameters and
            given observations.
        use_data_var
            If ``True``, use the data variance if present. If ``False``,
            ignore it even if it is present.
        use_data_cov
            If ``True``, use the data covariance if present. If ``False``,
            ignore it even if it is present.

        Returns
        -------
        G
            (If ``return_W_G=True``.) Reduced :math:`\mathbf{G}` matrix.
        W
            (If ``return_W_G=True``.) Reduced :math:`\mathbf{W}` matrix.
        GtWG
            Reduced :math:`\mathbf{G}^T \mathbf{W} \mathbf{G}` matrix.
        GtWd
            Reduced :math:`\mathbf{G}^T \mathbf{W} \mathbf{d}` matrix.

        See Also
        --------
        prepare_LS : Function that precedes this one in the regular solving process.
        """
        num_comps = ts.num_components
        if icomp is not None:
            assert isinstance(icomp, int) and icomp in list(range(num_comps)), \
                "'icomp' must be a valid integer component index (between 0 and " \
                f"{num_comps - 1}), got {icomp}."
            # d and G are dense
            d = ts.df[ts.data_cols[icomp]].values.reshape(-1, 1)
            dnotnan = ~np.isnan(d).squeeze()
            Gout = G.toarray()[np.ix_(dnotnan, obs_mask)]
            # W is sparse
            if (ts.var_cols is not None) and use_data_var:
                W = sparse.diags(1 / ts.df[ts.var_cols[icomp]].values[dnotnan])
            else:
                W = sparse.eye(dnotnan.sum())
        else:
            # d is dense, G and W are sparse
            d = ts.data.values.reshape(-1, 1)
            dnotnan = ~np.isnan(d).squeeze()
            Gout = sparse.kron(G[:, obs_mask], sparse.eye(num_comps), format='csr')
            if dnotnan.sum() < dnotnan.size:
                Gout = Gout[dnotnan, :]
            if (ts.cov_cols is not None) and use_data_var and use_data_cov:
                var_cov_matrix = ts.var_cov.values
                Wblocks = [sp.linalg.pinvh(np.reshape(var_cov_matrix[iobs, ts.var_cov_map],
                                                      (num_comps, num_comps)))
                           for iobs in range(ts.num_observations)]
                W = sparse.block_diag(Wblocks, format='csr')
                W.eliminate_zeros()
                if dnotnan.sum() < dnotnan.size:
                    W = W[dnotnan, :].tocsc()[:, dnotnan]
            elif (ts.var_cols is not None) and use_data_var:
                W = sparse.diags(1 / ts.vars.values.ravel()[dnotnan])
            else:
                W = sparse.eye(dnotnan.sum())
        if dnotnan.sum() < dnotnan.size:
            # double-check
            if np.any(np.isnan(Gout.data)) or np.any(np.isnan(W.data)):
                raise ValueError("Still NaNs in G or W, unexpected error!")
        # everything here will be dense, except GtW when using data covariance
        d = d[dnotnan]
        GtW = Gout.T @ W
        GtWG = GtW @ Gout
        GtWd = (GtW @ d).squeeze()
        if isinstance(GtWG, sparse.spmatrix):
            GtWG = GtWG.toarray()
        if return_W_G:
            return Gout, W, GtWG, GtWd
        else:
            return GtWG, GtWd

    def plot_covariance(self,
                        title: str | None = None,
                        fname: str | None = None,
                        use_corr_coef: bool = False,
                        plot_empty: bool = True,
                        save_kw_args: dict = {"format": "png"}
                        ) -> None:
        """
        Plotting method that displays the covariance (or correlation coefficient) matrix.
        The axes are labeled by model names for easier interpretation.

        Parameters
        ----------
        title
            If provided, the title that is added to the figure.
        fname
            By default (``None``), the figure is shown interarctively to enable zooming in
            etc. If an ``fname`` is provided, the figure is instead directly saved to the
            provided filename.
        use_corr_coef
            By default (``False``), the method plots the covariance matrix.
            If ``True``, the correlation coefficient matrix is plotted instead.
        plot_empty
            By default (``True``), the full matrix is plotted. If it is sparse, it will be
            hard to identify the interplay between the different parameters. Therefore,
            setting ``plot_empty=False`` will only plot the rows and columns corresponding
            to nonzero parameters.
        save_kw_args
            Additional keyword arguments passed to :meth:`~matplotlib.figure.Figure.savefig`,
            used when ``fname`` is provided.
        """
        # check if a covariance has been computed
        if self.cov is None:
            raise RuntimeError("No covariance found to plot.")
        # make a list of model names as well as the indices of the parameter boundaries
        # and the location of where to put the label
        # if a model is a spline collection, make the corresponding list entries another list
        model_labels = []
        boundary_indices = [0]
        label_centerpoints = []
        start_index = 0
        for i, (model_description, model) in enumerate(self.collection.items()):
            num_comps = model.par.shape[1]
            if isinstance(model, SplineSet):
                if plot_empty:
                    model_labels.extend([f"{model_description}\n"
                                         f"{m.scale:.4g} {m.time_unit}"
                                         for m in model.splines])
                    bndrs = np.concatenate([np.array([start_index]),
                                            start_index +
                                            np.cumsum([m.num_parameters * num_comps
                                                       for m in model.splines])])
                    boundary_indices.extend(bndrs[1:-1].tolist())
                    boundary_indices.append(bndrs[-1])
                    cntrpts = (bndrs[1:] + bndrs[:-1]) / 2
                    label_centerpoints.extend(cntrpts.tolist())
                    start_index += model.num_parameters * num_comps
                else:
                    for m in model.splines:
                        mod_nonzero = ~np.all(m.cov == 0, axis=0)
                        num_nonzero = mod_nonzero.sum()
                        if num_nonzero > 0:
                            model_labels.append(f"{model_description}\n"
                                                f"{m.scale:.4g} {m.time_unit}")
                            boundary_indices.append(start_index + num_nonzero)
                            label_centerpoints.append(start_index + num_nonzero / 2)
                            start_index += num_nonzero
            else:
                if plot_empty:
                    model_labels.append(model_description)
                    boundary_indices.append(start_index + model.num_parameters * num_comps)
                    label_centerpoints.append((boundary_indices[-1] + boundary_indices[-2]) / 2)
                    start_index += model.num_parameters * num_comps
                else:
                    mod_nonzero = ~np.all(model.cov == 0, axis=0)
                    num_nonzero = mod_nonzero.sum()
                    if num_nonzero > 0:
                        model_labels.append(model_description)
                        boundary_indices.append(start_index + num_nonzero)
                        label_centerpoints.append(start_index + num_nonzero / 2)
                        start_index += num_nonzero
        # get whatever needs to be plotted
        if not use_corr_coef:  # stick with covariance
            cov_mat = self.cov
            if not plot_empty:
                cov_mat_nonzero = ~np.all(cov_mat == 0, axis=0)
                assert cov_mat_nonzero.sum() == start_index
                cov_mat = cov_mat[np.ix_(cov_mat_nonzero, cov_mat_nonzero)]
            vmax = np.nanpercentile(np.abs(cov_mat.ravel()), 95)
            vmin = -vmax
            clabel = "Covariance"
        else:
            cov_mat = cov2corr(self.cov)
            if not plot_empty:
                cov_mat_nonzero = ~np.all(np.isnan(cov_mat), axis=0)
                assert cov_mat_nonzero.sum() == start_index
                cov_mat = cov_mat[np.ix_(cov_mat_nonzero, cov_mat_nonzero)]
            vmin, vmax = -1, 1
            clabel = "Correlation Coefficient"
        # start the figure
        fig, ax = plt.subplots()
        cov_img = ax.imshow(cov_mat, cmap=scm.roma, vmin=vmin, vmax=vmax, interpolation="none",
                            extent=(0, cov_mat.shape[1], cov_mat.shape[0], 0))
        fig.colorbar(cov_img, ax=ax, label=clabel, fraction=0.1)
        if title is not None:
            ax.set_title(title)
        # give the major axis the calculated boundaries
        ax.set_yticks(boundary_indices)
        ax.set_yticklabels([])
        # give the minor axis the labels
        ax.set_yticks(label_centerpoints, minor=True)
        ax.set_yticklabels(model_labels, minor=True)
        # give the y axis also the model name labels
        ax.tick_params(axis="y", which="major", direction="out", right=True, length=10)
        ax.tick_params(axis="y", which="minor", left=False, right=False)
        # do the same with the x axis, but with rotated labels
        ax.set_xticks(boundary_indices)
        ax.set_xticklabels([])
        ax.set_xticks(label_centerpoints, minor=True)
        ax.set_xticklabels(model_labels, minor=True, rotation="vertical")
        ax.tick_params(axis="x", which="major", direction="out", top=True, length=10)
        ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False,
                       top=False, labeltop=True)
        # plot or save
        if fname is None:
            plt.show()
        else:
            fig.savefig(f"{fname}.{save_kw_args['format']}", **save_kw_args)
            plt.close(fig)
