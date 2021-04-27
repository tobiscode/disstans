"""
This module contains solver routines for fitting models to the timeseries
of stations.
"""

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import cvxpy as cp
import cartopy.geodesic as cgeod
from warnings import warn
from tqdm import tqdm
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Mapping

from .tools import weighted_median, get_cov_dims, full_cov_mat_to_columns
from .models import Model


def _combine_mappings(ts, models, regularize=False, cached_mapping=None,
                      reweight_init=None, use_internal_scales=False):
    """
    Quick helper function that concatenates the mapping matrices of the
    models given the timevector in ts, and returns the relevant sizes.
    If regularize = True, also return an array indicating which model
    is set to be regularized.
    It also makes sure that G only contains columns that contain at least
    one non-zero element, and correspond to parameters that are therefore
    observable.
    Can also match model-specific reweight_init to the respective array.
    use_internal_scales sets whether to check the models for internal
    scaling parameters.
    """
    mapping_matrices = []
    obs_indices = []
    if regularize:
        reg_indices = []
        if use_internal_scales:
            weights_scaling = []
        if reweight_init is None:
            init_weights = None
        elif isinstance(reweight_init, np.ndarray):
            init_weights = reweight_init
        elif isinstance(reweight_init, dict):
            init_weights = []
    for (mdl_description, model) in models.items():
        if cached_mapping and mdl_description in cached_mapping:
            mapping = cached_mapping[mdl_description].loc[ts.time].values
            observable = np.any(mapping != 0, axis=0)
            if regularize and model.regularize:
                nunique = (~np.isclose(np.diff(mapping, axis=0), 0)).sum(axis=0) + 1
                observable = np.logical_and(observable, nunique > 1)
                if isinstance(reweight_init, dict) and (mdl_description in reweight_init):
                    init_weights.append(reweight_init[mdl_description][observable, :])
            mapping = sparse.csc_matrix(mapping[:, observable])
        else:
            mapping, observable = model.get_mapping(ts.time, return_observability=True)
            if int(observable.sum()) == 0:
                warn(f"For model '{mdl_description}' (regularized: {model.regularized}) "
                     f"and a timespan ranging from {ts.time.min()} to {ts.time.max} "
                     f"with {ts.num_observations} observations, no parameter is "
                     "observable. GeoNAT has not been tested for this case.")
            mapping = mapping[:, observable]
            if regularize and model.regularize and \
               isinstance(reweight_init, dict) and (mdl_description in reweight_init):
                init_weights.append(reweight_init[mdl_description][observable, :])
        mapping_matrices.append(mapping)
        obs_indices.append(observable)
        if regularize:
            reg_indices.extend([model.regularize] * int(observable.sum()))
            if use_internal_scales and model.regularize:
                weights_scaling.append(getattr(model, "internal_scales",
                                               np.ones(model.num_parameters))[observable])
    G = sparse.hstack(mapping_matrices, format='csc')
    obs_indices = np.concatenate(obs_indices)
    num_time, num_params = G.shape
    assert num_params > 0, f"Mapping matrix is empty, has shape {G.shape}."
    num_comps = ts.num_components
    if regularize:
        reg_indices = np.array(reg_indices)
        num_reg = reg_indices.sum()
        if use_internal_scales:
            weights_scaling = np.concatenate(weights_scaling)
        else:
            weights_scaling = None
        if num_reg == 0:
            warn("Regularized solver got no models to regularize.")
        if init_weights is not None:
            if isinstance(init_weights, list):
                init_weights = np.concatenate(init_weights)
            assert init_weights.shape == (num_reg, num_comps), \
                "The combined 'reweight_init' must have the shape " + \
                f"{(num_reg, num_comps)}, got {init_weights.shape}."
        return G, obs_indices, num_time, num_params, num_comps, num_reg, \
            reg_indices, init_weights, weights_scaling
    else:
        return G, obs_indices, num_time, num_params, num_comps


def _build_LS(ts, G, icomp=None, return_W_G=False, use_data_var=True, use_data_cov=True):
    """
    Quick helper function that given a multi-component data vector in ts,
    broadcasts the per-component matrices G and W to joint G and W matrices,
    and then computes GtWG and GtWd, the matrices necessary for least squares.
    If icomp is the index of a single component, only build the GtWG and GtWd
    matrices for that component (ignoring covariances).
    """
    num_comps = ts.num_components
    if icomp is not None:
        assert isinstance(icomp, int) and icomp in list(range(num_comps)), \
            "'icomp' must be a valid integer component index (between 0 and " \
            f"{num_comps-1}), got {icomp}."
        # d and G are dense
        d = ts.df[ts.data_cols[icomp]].values.reshape(-1, 1)
        dnotnan = ~np.isnan(d).squeeze()
        Gout = G.A[dnotnan, :]
        # W is sparse
        if (ts.var_cols is not None) and use_data_var:
            W = sparse.diags(1/ts.df[ts.var_cols[icomp]].values[dnotnan])
        else:
            W = sparse.eye(dnotnan.sum())
    else:
        # d is dense, G and W are sparse
        d = ts.data.values.reshape(-1, 1)
        dnotnan = ~np.isnan(d).squeeze()
        Gout = sparse.kron(G, sparse.eye(num_comps), format='csr')
        if dnotnan.sum() < dnotnan.size:
            Gout = Gout[dnotnan, :]
        if (ts.cov_cols is not None) and use_data_var and use_data_cov:
            var_cov_matrix = ts.var_cov.values
            Wblocks = [np.linalg.inv(np.reshape(var_cov_matrix[iobs, ts.var_cov_map],
                                                (num_comps, num_comps)))
                       for iobs in range(ts.num_observations)]
            offsets = list(range(-num_comps, num_comps + 1))
            diags = [np.concatenate([np.concatenate([np.diag(Welem, k), np.zeros(np.abs(k))])
                                     for Welem in Wblocks]) for k in offsets]
            Wn = len(Wblocks) * num_comps
            W = sparse.diags(diags, offsets, shape=(Wn, Wn), format='csr')
            W.eliminate_zeros()
            if dnotnan.sum() < dnotnan.size:
                W = W[dnotnan, :].tocsc()[:, dnotnan]
        elif (ts.var_cols is not None) and use_data_var:
            W = sparse.diags(1/ts.vars.values.reshape(-1, 1))
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
        GtWG = GtWG.A
    if return_W_G:
        return Gout, W, GtWG, GtWd
    else:
        return GtWG, GtWd


# simple helper class used within the Solution class just below
# needs to be picklable, therefore defined here outside
ModelSolution = namedtuple("ModelSolution",
                           field_names=["parameters", "variances",
                                        "covariances", "weights"])
"""
Helper class that contains the solution of an individual model for a single
station. Used within :class:`~Solution`.
"""
ModelSolution.parameters.__doc__ = "Estimated model parameters."
ModelSolution.variances.__doc__ = "Estimated model variances."
ModelSolution.covariances.__doc__ = "Estimated model component covariances."
ModelSolution.weights.__doc__ = "Model parameter weights used by the solver."


class Solution(Mapping):
    r"""
    Class that contains the solution to the problems of the solver functions
    in this module, and distributes the parameters, (co-)variances, and weights (where
    present) into the respective models. Behaves like a (read-only) Python dictionary,
    with the added attributes :attr:`~num_parameters` and :attr:`~num_components`.
    The solution for each model is stored as a :class:`~ModelSolution` object.

    Parameters
    ----------
    models : dict
        Dictionary of :class:`~geonat.models.Model` instances in the format of
        :attr:`~geonat.station.Station.models` (or a subset thereof) that were
        used by the solver.
    parameters : numpy.ndarray, int
        Model parameters of shape
        :math:`(\text{num_solved}, \text{num_components})`.
        If no parameters were estimated, pass the integer
        :math:`\text{num_components}`, which will create a memory-efficient
        array of ``Nan`` of the required dimensions.
    variances : numpy.ndarray, optional
        Model parameter variances of shape
        :math:`(\text{num_solved}, \text{num_components})`.
        If no variances (or parameters) are passed, this is set to ``None``.
    covariances : numpy.ndarray, optional
        Model parameter component covariances of shape
        :math:`(\text{num_solved}, (\text{num_components}*(\text{num_components}-1))/2)`.
        If no covariances (or parameters) are passed, this is set to ``None``.
    weights : numpy.ndarray, optional
        Model parameter regularization weights of shape
        :math:`(\text{num_solved}, \text{num_components})`.
        If no weights (or parameters) are passed, this is set to ``None``.
    obs_indices : numpy.ndarray, optional
        Observation mask of shape
        :math:`(\text{num_parameters}, \text{num_components})`
        with ``True`` at indices where the parameter was actually estimated,
        and ``False`` where the estimation was skipped du to observability or
        other reasons.
        Defaults to all ``True``, which implies that
        :math:`\text{num_parameters} = \text{num_solved}`.
    reg_indices : numpy.ndarray, optional
        Regularization mask of shape :math:`(\text{num_solved}, )`
        with ``True`` where a parameter was subject to regularization,
        and ``False`` otherwise. Defaults to all ``False``.

    Example
    -------
    Access the variances of the model ``'mymodel'`` in the solution object
    ``mysol`` as follows::

        >>> variances = mysol['mymodel'].variances
    """
    def __init__(self, models, parameters, variances=None, covariances=None,
                 weights=None, obs_indices=None, reg_indices=None):
        # input checks
        assert (isinstance(models, dict) and
                all([isinstance(mdl_desc, str) and isinstance(mdl, Model)
                     for (mdl_desc, mdl) in models.items()])), \
            f"'models' is not a valid dictionary of model names and objects, got {models}."
        input_types = [None if indat is None else type(indat) for indat
                       in [parameters, variances, covariances,
                           weights, obs_indices, reg_indices]]
        assert all([(intype is None) or (intype == np.ndarray) for intype in input_types]), \
            f"Unsupported input data types where not None: {input_types}."
        # check if a solution was indeed passed
        if isinstance(parameters, int):
            solution_found = False
            num_components = parameters
            parameters = None
            variances = None
            covariances = None
            weights = None
        else:
            solution_found = True
            num_components = parameters.shape[1]
            cov_dims = get_cov_dims(num_components)
            if reg_indices is not None:
                assert reg_indices.size == parameters.shape[0], \
                    "Unexpected parameter size mismatch: " \
                    f"{reg_indices.size} != {parameters.shape}[0]"
        # get the number of total parameters and match them with obs_indices, if present
        num_parameters = sum([m.num_parameters for m in models.values()])
        if (obs_indices is None) and solution_found:
            assert parameters.shape[0] == num_parameters, \
                "No 'obs_indices' was passed, but the shape of 'parameters' " \
                f"{parameters.shape} does not match the total number of parameters " \
                f"{num_parameters} in its first dimension."
        # skip the packing of weights if there isn't any regularization or weights
        if np.any(weights) and np.any(reg_indices):
            pack_weights = True
            ix_reg = 0
        else:
            pack_weights = False
        # start the iteration over the models
        solutions = {}
        ix_model = 0
        ix_sol = 0
        for (mdl_description, model) in models.items():
            # check which parameters of all possible ones were estimated
            if obs_indices is not None:
                mask = obs_indices[ix_model:ix_model+model.num_parameters]
            else:
                mask = np.ones(model.num_parameters, dtype=bool)
            num_solved = mask.sum()
            # pack the parameters, if present
            if parameters is not None:
                p = np.zeros((model.num_parameters, num_components))
                p[mask, :] = parameters[ix_sol:ix_sol+num_solved, :]
            else:
                p = np.broadcast_to(np.array([[np.NaN] * num_components]),
                                    (model.num_parameters, num_components))
            # initialize optional solution variables
            v, c, w = None, None, None
            # pack the variances, if present
            if variances is not None:
                v = np.empty((model.num_parameters, num_components))
                v[:] = np.NaN
                v[mask, :] = variances[ix_sol:ix_sol+num_solved, :]
                if covariances is not None:
                    c = np.empty((model.num_parameters, cov_dims))
                    c[:] = np.NaN
                    c[mask, :] = covariances[ix_sol:ix_sol+num_solved, :]
            # pack the weights, if present
            if pack_weights:
                mask_reg = reg_indices[ix_sol:ix_sol+num_solved]
                num_solved_reg = mask_reg.sum()
                if num_solved_reg > 0:
                    w = np.empty((model.num_parameters, num_components))
                    w[:] = np.NaN
                    w[np.flatnonzero(mask)[mask_reg], :] = \
                        weights[ix_reg:ix_reg+num_solved_reg, :]
                ix_reg += num_solved_reg
            ix_model += model.num_parameters
            ix_sol += num_solved
            # create ModelSolution object
            solutions[mdl_description] = ModelSolution(p, v, c, w)
        # cleanup checks to see if all the iterations worked as planned
        assert (obs_indices is None) or (ix_model == obs_indices.shape[0]), \
            f"Unexpected model size mismatch: {ix_model} != {obs_indices.shape}[0]"
        assert (parameters is None) or (ix_sol == parameters.shape[0]), \
            f"Unexpected solution size mismatch: {ix_sol} != {parameters.shape}[0]"
        if pack_weights:
            assert ix_reg == weights.shape[0], \
                f"Unexpected regularization size mismatch: {ix_reg} != {weights.shape}[0]"
        # save results to class instance
        self._solutions = solutions
        self.num_parameters = num_parameters
        """ Total number of parameters (solved or unsolved) in solution. """
        self.num_components = num_components
        """ Number of components the solution was computed for. """

    def __getitem__(self, model):
        return self._solutions[model]

    def __iter__(self):
        return iter(self._solutions)

    def __len__(self):
        return len(self._solutions)

    @property
    def model_list(self):
        """ List of models present in the solution. """
        return list(self._solutions.keys())

    @staticmethod
    def aggregate_models(results_dict, mdl_description, key_list=None,
                         stack_parameters=False, stack_variances=False,
                         stack_covariances=False, stack_weights=False):
        """
        For a dictionary of Solution objects (e.g. one per station) and a given
        model description, aggregate the model parameters, variances and parameter
        regularization weights (where present) into combined NumPy arrays.

        Parameters
        ----------
        results_dict : dict
            Dictionary of Solution objects.
        mdl_description : str
            Name of the model to aggregate the parameters, variances and weights for.
        key_list : list, optional
            If provided, aggregate only the selected keys in the dictionary.
            Defaults to all keys.
        stack_parameters : bool, optional
            If ``True``, stack the parameters, otherwise just return ``None``.
            Defaults to ``False``.
        stack_variances : bool, optional
            If ``True``, stack the variances, otherwise just return ``None``.
            Defaults to ``False``.
        stack_covariances : bool, optional
            If ``True``, stack the covariances, otherwise just return ``None``.
            Defaults to ``False``.
        stack_weights : bool, optional
            If ``True``, stack the weights, otherwise just return ``None``.
            Defaults to ``False``.

        Returns
        -------
        numpy.ndarray
            If ``stack_parameters=True``, the stacked model parameters.
        numpy.ndarray
            If ``stack_variances=True`` and variances are present in the models,
            the stacked variances.
        numpy.ndarray
            If ``stack_covariances=True`` and covariances are present in the models,
            the stacked component covariances.
        numpy.ndarray
            If ``stack_weights=True`` and regularization weights are present in the models,
            the stacked weights, ``None`` otherwise.
        """
        # input checks
        stack_mask = [stack_parameters, stack_variances, stack_covariances, stack_weights]
        assert any(stack_mask), "Called 'aggregate_models' without anything to aggregate."
        assert (isinstance(results_dict, dict) and
                all([isinstance(mdl_sol, Solution) for mdl_sol in results_dict.values()])), \
            f"'results_dict' needs to be a dictionary of Solution objects, got {results_dict}."
        if key_list is None:
            key_list = list(results_dict.keys())
        # loop over weights, variances, and weights
        out = []
        for var, dostack in zip(["parameters", "variances", "covariances", "weights"], stack_mask):
            # skip if flagged
            if not dostack:
                continue
            # stack arrays ignoring Nones
            stack = [getattr(results_dict[key][mdl_description], var)
                     for key in key_list if (mdl_description in results_dict[key])
                     and (getattr(results_dict[key][mdl_description], var) is not None)]
            stack_shapes = [mdl.shape for mdl in stack]
            # only stack if all the models have actually the same dimensions
            if (len(stack) > 0) and (stack_shapes.count(stack_shapes[0]) == len(stack)):
                out.append(np.stack(stack))
        # return each item individually
        return (*out, )


class ReweightingFunction(ABC):
    """
    Base class for reweighting functions. At initialization, the :attr:`~eps`
    parameter is set. Inheriting child classes need to define the
    :meth:`~__call__` method that determines the actual reweighting mechanism.
    Instantiated reweighting functions objects can be used as functions but
    still provide access to the :attr:`~eps` parameter.

    Parameters
    ----------
    eps : int, float
        Stability parameter to use for the reweighting function.
    scale : float, optional
        Scale parameter applied to reweighting result.
    """
    def __init__(self, eps, scale=1):
        self.eps = float(eps)
        """ Stability parameter to use for the reweighting function. """
        self.scale = float(scale)
        """ Scaling factor applied to reweighting result. """

    @abstractmethod
    def __call__(self, value):
        pass

    @staticmethod
    def from_name(name, *args, **kw_args):
        """
        Search the local namespace for a reweighting function of that name
        and return an initialized instance of it.

        Parameters
        ----------
        name : str
            Name (or abbreviation) of the reweighting function.
        *args : list
            Argument list passed on to function initialization.
        **kw_args : dict
            Keyword arguments passed on to function initialization.

        Returns
        -------
        ReweightingFunction
            An instantiated reweighting function object.
        """
        if name in ["inv", "InverseReweighting"]:
            return InverseReweighting(*args, **kw_args)
        elif name in ["inv_sq", "InverseSquaredReweighting"]:
            return InverseSquaredReweighting(*args, **kw_args)
        elif name in ["log", "LogarithmicReweighting"]:
            return LogarithmicReweighting(*args, **kw_args)
        else:
            raise NotImplementedError(f"The reweighting function {name} "
                                      "could not be found.")


class InverseReweighting(ReweightingFunction):
    def __call__(self, m):
        r"""
        Reweighting function based on the inverse of the input based on [candes08]_:

        .. math::
            w(m_j) = \frac{1}{|m_j| + \text{eps}}

        Parameters
        ----------
        m : numpy.ndarray
            :math:`\mathbf{m}`
        """
        return self.scale / (np.abs(m) + self.eps)


class InverseSquaredReweighting(ReweightingFunction):
    def __call__(self, m):
        r"""
        Reweighting function based on the inverse squared of the input based on [candes08]_:

        .. math::
            w(m_j) = \frac{1}{m_j^2 + \text{eps}^2}

        Parameters
        ----------
        m : numpy.ndarray
            :math:`\mathbf{m}`
        """
        return self.scale / (m**2 + self.eps**2)


class LogarithmicReweighting(ReweightingFunction):
    def __call__(self, m):
        r"""
        Reweighting function based on the logarithm of the input based on [andrecut11]_:

        .. math::
            w(m_j) = \log_\text{num_reg} \frac{ \| \mathbf{m} \|_1 +
            \text{num_reg} \cdot \text{eps}}{|m_j| + \text{eps}}

        (where :math:`0 < \text{eps} \ll \frac{1}{\text{num_reg}}`).

        Parameters
        ----------
        m : numpy.ndarray
            :math:`\mathbf{m}`

        References
        ----------
        .. [andrecut11] Andrecut, M. (2011).
           *Stochastic Recovery Of Sparse Signals From Random Measurements.*
           Engineering Letters, 19(1), 1-6.
        """
        mags = np.abs(m)
        size = mags.size
        weight = np.log((mags.sum() + size*self.eps) / (mags + self.eps)) / np.log(size)
        return self.scale * weight


def linear_regression(ts, models, formal_variance=False, formal_covariance=False,
                      cached_mapping=None, use_data_variance=True, use_data_covariance=True):
    r"""
    Performs linear, unregularized least squares using :mod:`~scipy.sparse.linalg`.

    The timeseries are the observations :math:`\mathbf{d}`, and the models' mapping
    matrices are stacked together to form a single, sparse mapping matrix
    :math:`\mathbf{G}`. The solver then computes the model parameters
    :math:`\mathbf{m}` that minimize the cost function

    .. math:: f(\mathbf{m}) = \left\| \mathbf{Gm} - \mathbf{d} \right\|_2^2

    where :math:`\mathbf{\epsilon} = \mathbf{Gm} - \mathbf{d}` is the residual.

    If the observations :math:`\mathbf{d}` include a covariance matrix
    :math:`\mathbf{C}_d` (incorporating `var_cols` and possibly also `cov_cols`),
    this data will be used. In this case, :math:`\mathbf{G}` and :math:`\mathbf{d}`
    are replaced by their weighted versions

    .. math:: \mathbf{G} \rightarrow \mathbf{G}^T \mathbf{C}_d^{-1} \mathbf{G}

    and

    .. math:: \mathbf{d} \rightarrow \mathbf{G}^T \mathbf{C}_d^{-1} \mathbf{d}

    The formal model covariance is defined as the pseudo-inverse

    .. math:: \mathbf{C}_m = \left( \mathbf{G}^T \mathbf{C}_d \mathbf{G} \right)^g

    Parameters
    ----------
    ts : geonat.timeseries.Timeseries
        Timeseries to fit.
    models : dict
        Dictionary of :class:`~geonat.models.Model` instances used for fitting.
    formal_variance : bool, optional
        If ``True``, also calculate the formal variance (diagonals of the covariance
        matrix). Defaults to ``False``.
    formal_covariance : bool, optional
        If ``True``, also calculate the formal covariance between components
        (found in the off-diagonals of the covariance matrix). Defaults to ``False``.
    cached_mapping : dict, optional
        If passed, a dictionary containing the mapping matrices as Pandas DataFrames
        for a subset of models and for all timestamps present in ``ts``.
        Mapping matrices not in ``cached_mapping`` will have to be recalculated.
    use_data_variance : bool, optional
        If ``True`` (default) and ``ts`` contains variance information, this
        uncertainty information will be used.
    use_data_covariance : bool, optional
        If ``True`` (default), ``ts`` contains variance and covariance information, and
        ``use_data_variance`` is also ``True``, this uncertainty information will be used.

    Returns
    -------
    model_params_var : dict
        Dictionary of form ``{"model_description": (parameters, variance), ...}``
        which for every model that was fitted, contains a tuple of the best-fit
        parameters and the formal variance (or ``None``, if not calculated).
    """

    # get mapping matrix and sizes
    G, obs_indices, num_time, num_params, num_comps = \
        _combine_mappings(ts, models, cached_mapping=cached_mapping)

    # perform fit and estimate formal covariance (uncertainty) of parameters
    # if there is no covariance, it's num_comps independent problems
    if not formal_variance:
        var = None
    if not formal_covariance:
        cov = None
    if (ts.cov_cols is None) or (not use_data_covariance):
        params = np.zeros((num_params, num_comps))
        if formal_variance:
            var = np.zeros((num_params, num_comps))
        for i in range(num_comps):
            GtWG, GtWd = _build_LS(ts, G, icomp=i, use_data_var=use_data_variance)
            params[:, i] = sp.linalg.lstsq(GtWG, GtWd)[0].squeeze()
            if formal_variance:
                var[:, i] = np.diag(np.linalg.pinv(GtWG))
    else:
        GtWG, GtWd = _build_LS(ts, G, use_data_var=use_data_variance,
                               use_data_cov=use_data_covariance)
        params = sp.linalg.lstsq(GtWG, GtWd)[0].reshape(num_params, num_comps)
        if formal_variance:
            cov_mat = np.linalg.pinv(GtWG)
            var, cov = full_cov_mat_to_columns(cov_mat, num_params, num_comps,
                                               include_covariance=formal_covariance)

    # create solution object and return
    return Solution(models=models, parameters=params, variances=var, covariances=cov,
                    obs_indices=obs_indices)


def ridge_regression(ts, models, penalty, formal_variance=False, formal_covariance=False,
                     cached_mapping=None, use_data_variance=True, use_data_covariance=True):
    r"""
    Performs linear, L2-regularized least squares using :mod:`~scipy.sparse.linalg`.

    The timeseries are the observations :math:`\mathbf{d}`, and the models' mapping
    matrices are stacked together to form a single, sparse mapping matrix
    :math:`\mathbf{G}`. Given the penalty hyperparameter :math:`\lambda`, the solver then
    computes the model parameters :math:`\mathbf{m}` that minimize the cost function

    .. math:: f(\mathbf{m}) = \left\| \mathbf{Gm} - \mathbf{d} \right\|_2^2
              + \lambda \left\| \mathbf{m}_\text{reg} \right\|_2^2

    where :math:`\mathbf{\epsilon} = \mathbf{Gm} - \mathbf{d}` is the residual
    and the subscript :math:`_\text{reg}` masks to zero the model parameters
    not designated to be regularized (see :attr:`~geonat.models.Model.regularize`).

    If the observations :math:`\mathbf{d}` include a covariance matrix
    :math:`\mathbf{C}_d` (incorporating `var_cols` and possibly also `cov_cols`),
    this data will be used. In this case, :math:`\mathbf{G}` and :math:`\mathbf{d}`
    are replaced by their weighted versions

    .. math:: \mathbf{G} \rightarrow \mathbf{G}^T \mathbf{C}_d^{-1} \mathbf{G}

    and

    .. math:: \mathbf{d} \rightarrow \mathbf{G}^T \mathbf{C}_d^{-1} \mathbf{d}

    The formal model covariance is defined as the pseudo-inverse

    .. math:: \mathbf{C}_m = \left( \mathbf{G}^T \mathbf{C}_d \mathbf{G}
                                    + \lambda \mathbf{I}_\text{reg} \right)^g

    where the subscript :math:`_\text{reg}` masks to zero the entries corresponding
    to non-regularized model parameters.

    Parameters
    ----------
    ts : geonat.timeseries.Timeseries
        Timeseries to fit.
    models : dict
        Dictionary of :class:`~geonat.models.Model` instances used for fitting.
    penalty : float
        Penalty hyperparameter :math:`\lambda`.
    formal_variance : bool, optional
        If ``True``, also calculate the formal variance (diagonals of the covariance
        matrix). Defaults to ``False``.
    formal_covariance : bool, optional
        If ``True``, also calculate the formal covariance between components
        (found in the off-diagonals of the covariance matrix). Defaults to ``False``.
    cached_mapping : dict, optional
        If passed, a dictionary containing the mapping matrices as Pandas DataFrames
        for a subset of models and for all timestamps present in ``ts``.
        Mapping matrices not in ``cached_mapping`` will have to be recalculated.
    use_data_variance : bool, optional
        If ``True`` (default) and ``ts`` contains variance information, this
        uncertainty information will be used.
    use_data_covariance : bool, optional
        If ``True`` (default), ``ts`` contains variance and covariance information, and
        ``use_data_variance`` is also ``True``, this uncertainty information will be used.

    Returns
    -------
    model_params_var : dict
        Dictionary of form ``{"model_description": (parameters, variance), ...}``
        which for every model that was fitted, contains a tuple of the best-fit
        parameters and the formal variance (or ``None``, if not calculated).
    """
    if penalty == 0.0:
        warn(f"Ridge Regression (L2-regularized) solver got a penalty of {penalty}, "
             "which effectively removes the regularization.")

    # get mapping and regularization matrix and sizes
    G, obs_indices, num_time, num_params, num_comps, num_reg, reg_indices, _, _ = \
        _combine_mappings(ts, models, regularize=True, cached_mapping=cached_mapping)

    # perform fit and estimate formal covariance (uncertainty) of parameters
    # if there is no covariance, it's num_comps independent problems
    if not formal_variance:
        var = None
    if not formal_covariance:
        cov = None
    if (ts.cov_cols is None) or (not use_data_covariance):
        reg = np.diag(reg_indices) * penalty
        params = np.zeros((num_params, num_comps))
        if formal_variance:
            var = np.zeros((num_params, num_comps))
        for i in range(num_comps):
            GtWG, GtWd = _build_LS(ts, G, icomp=i, use_data_var=use_data_variance)
            GtWGreg = GtWG + reg
            params[:, i] = sp.linalg.lstsq(GtWGreg, GtWd)[0].squeeze()
            if formal_variance:
                var[:, i] = np.diag(np.linalg.pinv(GtWGreg))
    else:
        GtWG, GtWd = _build_LS(ts, G, use_data_var=use_data_variance,
                               use_data_cov=use_data_covariance)
        reg = np.diag(np.repeat(reg_indices, num_comps)) * penalty
        GtWGreg = GtWG + reg
        params = sp.linalg.lstsq(GtWGreg, GtWd)[0].reshape(num_params, num_comps)
        if formal_variance:
            cov_mat = np.linalg.pinv(GtWGreg)
            var, cov = full_cov_mat_to_columns(cov_mat, num_params, num_comps,
                                               include_covariance=formal_covariance)

    # create solution object and return
    return Solution(models=models, parameters=params, variances=var, covariances=cov,
                    obs_indices=obs_indices)


def lasso_regression(ts, models, penalty, reweight_max_iters=None, reweight_func=None,
                     reweight_max_rss=1e-10, reweight_init=None, reweight_coupled=True,
                     formal_variance=False, formal_covariance=False, cached_mapping=None,
                     use_data_variance=True, use_data_covariance=True,
                     use_internal_scales=False, return_weights=False,
                     cvxpy_kw_args={"solver": "CVXOPT", "kktsolver": "robust"}):
    r"""
    Performs linear, L1-regularized least squares using
    `CVXPY <https://www.cvxpy.org/index.html>`_.

    The timeseries are the observations :math:`\mathbf{d}`, and the models' mapping
    matrices are stacked together to form a single, sparse mapping matrix
    :math:`\mathbf{G}`. Given the penalty hyperparameter :math:`\lambda`, the solver then
    computes the model parameters :math:`\mathbf{m}` that minimize the cost function

    .. math:: f(\mathbf{m}) = \left\| \mathbf{Gm} - \mathbf{d} \right\|_2^2
              + \lambda \left\| \mathbf{m}_\text{reg} \right\|_1

    where :math:`\mathbf{\epsilon} = \mathbf{Gm} - \mathbf{d}` is the residual
    and the subscript :math:`_\text{reg}` masks to zero the model parameters
    not designated to be regularized (see :attr:`~geonat.models.Model.regularize`).

    If the observations :math:`\mathbf{d}` include a covariance matrix
    :math:`\mathbf{C}_d` (incorporating `var_cols` and possibly also `cov_cols`),
    this data will be used. In this case, :math:`\mathbf{G}` and :math:`\mathbf{d}`
    are replaced by their weighted versions

    .. math:: \mathbf{G} \rightarrow \mathbf{G}^T \mathbf{C}_d^{-1} \mathbf{G}

    and

    .. math:: \mathbf{d} \rightarrow \mathbf{G}^T \mathbf{C}_d^{-1} \mathbf{d}

    If ``reweight_max_iters`` is specified, sparsity of the solution parameters is promoted
    by iteratively reweighting the penalty parameter for each regularized parameter based
    on its current value, approximating the L0 norm rather than the L1 norm (see Notes).

    The formal model covariance :math:`\mathbf{C}_m` is defined as being zero except in
    the rows and columns corresponding to non-zero parameters, where it is defined
    exactly as the unregularized version (see :func:`~geonat.solvers.linear_regression`),
    restricted to those same rows and columns.

    Parameters
    ----------
    ts : geonat.timeseries.Timeseries
        Timeseries to fit.
    models : dict
        Dictionary of :class:`~geonat.models.Model` instances used for fitting.
    penalty : float
        Penalty hyperparameter :math:`\lambda`.
    reweight_max_iters : int, optional
        If an integer, number of solver iterations (see Notes), resulting in reweighting.
        Defaults to no reweighting (``None``).
    reweight_func : ReweightingFunction, optional
        If reweighting is active, the reweighting function instance to be used.
        Defaults to an inverse reweighting with stability parameter ``eps=1e-4``.
    reweight_max_rss : float, optional
        When reweighting is active and the maximum number of iterations has not yet
        been reached, let the iteration stop early if the solutions do not change much
        anymore (see Notes).
        Defaults to ``1e-10``. Set to ``0`` to deactivate early stopping.
    reweight_init : numpy.ndarray, optional
        When reweighting is active, use this array to initialize the weights.
        It has to have size :math:`\text{num_components} \cdot \text{num_reg}`, where
        :math:`\text{num_components}=1` if covariances are not used (and the actual
        number of timeseries components otherwise) and :math:`\text{num_reg}` is the
        number of regularized model parameters.
    reweight_coupled : bool, optional
        If ``True`` (default) and reweighting is active, the L1 penalty hyperparameter
        is coupled with the reweighting weights (see Notes).
    formal_variance : bool, optional
        If ``True``, also calculate the formal variance (diagonals of the covariance
        matrix). Defaults to ``False``.
    formal_covariance : bool, optional
        If ``True``, also calculate the formal covariance between components
        (found in the off-diagonals of the covariance matrix). Defaults to ``False``.
    cached_mapping : dict, optional
        If passed, a dictionary containing the mapping matrices as Pandas DataFrames
        for a subset of models and for all timestamps present in ``ts``.
        Mapping matrices not in ``cached_mapping`` will have to be recalculated.
    use_data_variance : bool, optional
        If ``True`` (default) and ``ts`` contains variance information, this
        uncertainty information will be used.
    use_data_covariance : bool, optional
        If ``True`` (default), ``ts`` contains variance and covariance information, and
        ``use_data_variance`` is also ``True``, this uncertainty information will be used.
    use_internal_scales : bool, optional
        If ``False`` (default), the reweighting does not look for or take into account
        model-specific internal scaling parameters.
    return_weights : bool, optional
        When reweighting is active, set to ``True`` to return the weights after the last
        update.
        Defaults to ``False``.
    cvxpy_kw_args : dict
        Additional keyword arguments passed on to CVXPY's ``solve()`` function.
        By default, the CVXPY solver options are set to use CVXOPT as the solver
        library, together with CVXPY's *robust* ``kktsolver`` option. This slows down
        the solution significantly, but in general will converge more reliably
        when using L1 regularization. If you want to switch to default CVXPY settings,
        pass an empty dictionary.

    Returns
    -------
    model_params_var : dict
        Dictionary of form ``{"model_description": (parameters, variance), ...}``
        which for every model that was fitted, contains a tuple of the best-fit
        parameters and the formal variance (or ``None``, if not calculated).

    Notes
    -----

    The L0-regularization approximation used by setting ``reweight_max_iters >= 0`` is based
    on [candes08]_. The idea here is to iteratively reduce the cost (before multiplication
    with :math:`\lambda`) of regularized, but significant parameters to 1, and iteratively
    increasing the cost of a regularized, but small parameter to a much larger value.

    This is achieved by introducing an additional parameter vector :math:`\mathbf{w}`
    of the same shape as the regularized parameters, inserting it into the L1 cost,
    and iterating between solving the L1-regularized problem, and using a reweighting
    function on those weights:

    1.  Initialize :math:`\mathbf{w}^{(0)} = \mathbf{1}`
        (or use the array from ``reweight_init``).
    2.  Solve the modified weighted L1-regularized problem minimizing
        :math:`f(\mathbf{m}^{(i)}) = \left\| \mathbf{Gm}^{(i)} -
        \mathbf{d} \right\|_2^2 + \lambda \left\| \mathbf{w}^{(i)} \circ
        \mathbf{m}^{(i)}_\text{reg} \right\|_1`
        where :math:`\circ` is the element-wise multiplication and :math:`i` is
        the iteration step.
    3.  Update the weights element-wise using a predefined reweighting function
        :math:`\mathbf{w}^{(i+1)} = w(\mathbf{m}^{(i)}_\text{reg})`.
    4.  Repeat from step 2 until ``reweight_max_iters`` iterations are reached
        or the root sum of squares of the difference between the last and current
        solution is less than ``reweight_max_rss``.

    The reweighting function is set through the argument ``reweight_func``, see
    :class:`~ReweightingFunction` and its derived classes.

    If reweighting is active and ``reweight_coupled=True``, :math:`\lambda`
    is moved into the norm and combined with :math:`\mathbf{w}`, such that
    the reweighting applies to the product of both.
    Furthermore, if ``reweight_init`` is also not ``None``, then the ``penalty`` is ignored
    since it should already be contained in the passed weights array.
    (If ``reweight_coupled=False``, :math:`\lambda` is always applied separately, regardless
    of whether initial weights are passed or not.)

    Note that the orders of magnitude between the penalties computed by the different
    reweighting functions for the same input parameters can differ significantly, even
    with the same ``penalty``.

    References
    ----------
    .. [candes08] Candès, E. J., Wakin, M. B., & Boyd, S. P. (2008).
       *Enhancing Sparsity by Reweighted* :math:`\ell_1` *Minimization.*
       Journal of Fourier Analysis and Applications, 14(5), 877–905.
       doi:`10.1007/s00041-008-9045-x <https://doi.org/10.1007/s00041-008-9045-x>`_.
    """
    if penalty == 0:
        warn(f"Lasso Regression (L1-regularized) solver got a penalty of {penalty}, "
             "which removes the regularization.")

    # get mapping and regularization matrix
    G, obs_indices, num_time, num_params, num_comps, num_reg, \
        reg_indices, init_weights, weights_scaling = \
        _combine_mappings(ts, models, regularize=True, cached_mapping=cached_mapping,
                          reweight_init=reweight_init, use_internal_scales=use_internal_scales)
    regularize = (num_reg > 0) and (penalty > 0)
    if (not regularize) or (reweight_max_iters is None):
        return_weights = False
    if reweight_max_iters is None:
        n_iters = 1
    else:
        assert isinstance(reweight_max_iters, int) and reweight_max_iters > 0
        n_iters = int(reweight_max_iters)
        if reweight_func is None:
            rw_func = ReweightingFunction.from_name("inv", 1e-4)
        else:
            assert isinstance(reweight_func, ReweightingFunction), "'reweight_func' " \
                f"needs to be None or a ReweightingFunction, got {type(reweight_func)}."
            rw_func = reweight_func

    # solve CVXPY problem while checking for convergence
    def solve_problem(GtWG, GtWd, reg_indices, num_comps, init_weights):
        # build objective function
        m = cp.Variable(GtWd.size)
        objective = cp.norm2(GtWG @ m - GtWd)
        constraints = None
        if regularize:
            if reweight_max_iters is not None:
                reweight_size = num_reg*num_comps
                if init_weights is None:
                    init_weights = np.ones(reweight_size)
                    if reweight_coupled:
                        init_weights *= penalty
                else:
                    assert init_weights.size == reweight_size, \
                        f"'init_weights' must have a size of {reweight_size}, " + \
                        f"got {reweight_init.size}."
                weights = cp.Parameter(shape=reweight_size,
                                       value=init_weights, pos=True)
                z = cp.Variable(shape=reweight_size)
                if reweight_coupled:
                    objective = objective + cp.norm1(z)
                else:
                    lambd = cp.Parameter(value=penalty, pos=True)
                    objective = objective + lambd * cp.norm1(z)
                constraints = [z == cp.multiply(weights, m[reg_indices])]
                old_m = np.zeros(m.shape)
            else:
                lambd = cp.Parameter(value=penalty, pos=True)
                objective = objective + lambd * cp.norm1(m[reg_indices])
        # define problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        # solve
        for i in range(n_iters):  # always solve at least once
            try:
                problem.solve(enforce_dpp=True, **cvxpy_kw_args)
            except cp.error.SolverError as e:
                # no solution found, but actually a more serious problem
                warn(str(e))
                converged = False
                break
            else:
                if m.value is None:  # no solution found
                    converged = False
                    break
                # solved
                converged = True
                # if iterating, extra tasks
                if regularize and reweight_max_iters is not None:
                    # update weights
                    if weights_scaling is not None:
                        weights.value = rw_func(m.value[reg_indices]*weights_scaling)
                    else:
                        weights.value = rw_func(m.value[reg_indices])
                    # check if the solution changed to previous iteration
                    if (i > 0) and (np.sqrt(np.sum((old_m - m.value)**2)) < reweight_max_rss):
                        break
                    # remember previous solution
                    old_m[:] = m.value[:]
        # return
        if converged and (regularize and reweight_max_iters is not None):
            result = (m.value, weights.value)
        elif converged:
            result = (m.value, None)
        else:
            result = (None, None)
        return result

    # perform fit and estimate formal covariance (uncertainty) of parameters
    # if there is no covariance, it's num_comps independent problems
    if not formal_variance:
        var = None
        cov = None
    if not formal_covariance:
        cov = None
    else:
        cov_dims = get_cov_dims(num_comps)
    if not return_weights:
        weights = None
    if (ts.cov_cols is None) or (not use_data_covariance):
        # initialize output
        params = np.zeros((num_params, num_comps))
        if formal_variance:
            var = np.zeros((num_params, num_comps))
        if regularize and return_weights:
            weights = np.zeros((num_reg, num_comps))
        # loop over components
        for i in range(num_comps):
            # build and solve problem
            Gnonan, Wnonan, GtWG, GtWd = _build_LS(ts, G, icomp=i, return_W_G=True,
                                                   use_data_var=use_data_variance)
            solution, wts = solve_problem(GtWG, GtWd, reg_indices, num_comps=1,
                                          init_weights=init_weights[:, i]
                                          if init_weights is not None else None)
            # store results
            if solution is None:
                params[:, i] = np.NaN
                if formal_variance:
                    var[:, i] = np.NaN
                if regularize and return_weights:
                    weights[:, i] = np.NaN
            else:
                params[:, i] = solution
                # if desired, estimate formal variance here
                if formal_variance:
                    best_ind = np.nonzero(solution)[0]
                    Gsub = Gnonan[:, best_ind]
                    GtWG = Gsub.T @ Wnonan @ Gsub
                    if isinstance(GtWG, sparse.spmatrix):
                        GtWG = GtWG.A
                    var[best_ind, i] = np.diag(np.linalg.pinv(GtWG))
                if regularize and return_weights:
                    weights[:, i] = wts
    else:
        # build stacked problem and solve
        Gnonan, Wnonan, GtWG, GtWd = _build_LS(ts, G, return_W_G=True,
                                               use_data_var=use_data_variance,
                                               use_data_cov=use_data_covariance)
        reg_indices = np.repeat(reg_indices, num_comps)
        solution, wts = solve_problem(GtWG, GtWd, reg_indices, num_comps=num_comps,
                                      init_weights=init_weights.ravel()
                                      if init_weights is not None else None)
        # store results
        if solution is None:
            params = np.empty((num_params, num_comps))
            params[:] = np.NaN
            if formal_variance:
                var = np.empty((num_params, num_comps))
                var[:] = np.NaN
                if formal_covariance:
                    cov = np.empty((num_params, cov_dims))
                    cov[:] = np.NaN
            if regularize and return_weights:
                weights = np.empty((num_reg, num_comps))
                weights[:] = np.NaN
        else:
            params = solution.reshape(num_params, num_comps)
            # if desired, estimate formal variance here
            if formal_variance:
                best_ind = np.nonzero(solution)[0]
                Gsub = Gnonan.tocsc()[:, best_ind]
                GtWG = Gsub.T @ Wnonan @ Gsub
                if isinstance(GtWG, sparse.spmatrix):
                    GtWG = GtWG.A
                cov_mat = np.zeros((num_params * num_comps, num_params * num_comps))
                cov_mat[np.ix_(best_ind, best_ind)] = np.linalg.pinv(GtWG)
                var, cov = full_cov_mat_to_columns(cov_mat, num_params, num_comps,
                                                   include_covariance=formal_covariance)
            if regularize and return_weights:
                weights = wts.reshape(num_reg, num_comps)
        # restore reg_indices' original shape
        reg_indices = reg_indices[::num_comps]

    # create solution object and return
    return Solution(models=models, parameters=params, variances=var, covariances=cov,
                    weights=weights, obs_indices=obs_indices, reg_indices=reg_indices)


class SpatialSolver():
    r"""
    Solver class that in combination with :func:`~lasso_regression` solves the
    spatiotemporal, L0-reweighted least squares fitting problem given the models and
    timeseries found in a target :class:`~geonat.network.Network` object.
    This is achieved by following the alternating computation scheme as described
    in :meth:`~solve`.

    Parameters
    ----------
    net : geonat.network.Network
        Network to fit.
    ts_description : str
        Description of the timeseries to fit.
    model_list : list, optional
        List of strings containing the model names of the subset of the models
        to fit. Defaults to all models.
    """

    ROLLMEANKERNEL = 30
    """
    Only used in :meth:`~solve` if ``verbose=2``. This is the kernel size that
    gets used in the analysis of the residuals between each fitting step.
    """
    ZERO = 1e-4
    """
    Absolute values below this threshold will be considered to be almost zero
    when calculating diagnostic statistics.
    """

    def __init__(self, net, ts_description, model_list=None):
        self.net = net
        """ Network object to fit. """
        self.ts_description = ts_description
        """ Name of timeseries to fit. """
        self.model_list = model_list
        """ Names of the models to fit (``None`` for all). """
        self.last_statistics = None
        r"""
        :class:`~typing.NamedTuple` of the statistics of the last call to
        :meth:`~solve` (or ``None``).

        Attributes
        ----------
        num_total : int
            Total number of parameters that were reweighted.
        arr_uniques : numpy.ndarray
            Array of shape :math:`(\text{spatial_reweight_iters}+1, \text{num_components})`
            of the number of unique (i.e., over all stations) parameters that are non-zero
            for each iteration.
        list_nonzeros : list
            List of the total number of non-zero parameters for each iteration.
        dict_rms_diff : dict
            Dictionary that for each reweighted model and contains a list (of length
            ``spatial_reweight_iters``) of the RMS differences of the reweighted parameter
            values between spatial iterations.
        dict_num_changed : dict
            Dictionary that for each reweighted model and contains a list (of length
            ``spatial_reweight_iters``) of the number of reweighted parameters that changed
            from zero to non-zero or vice-versa.
        list_res_stats : list
            (Only present if ``verbose=2`` in :meth:`~solve`.)
            List of the results dataframe returned by
            :meth:`~geonat.network.Network.analyze_residuals` for each iteration.
        dict_cors : dict
            (Only present if ``verbose=2`` in :meth:`~solve`.)
            For each of the reweighting models, contains a list of spatial correlation
            matrices for each iteration and component. E.g., the correlation matrix
            for model ``'my_model'`` after ``5`` reweighting iterations (i.e. the sixth
            solution, taking into account the initial unweighted solution) for the first
            component can be found in ``last_statistics.dict_cors['my_model'][5][0]``
            and has a shape of :math:`(\text{num_stations}, \text{num_stations})`.
        dict_cors_means : dict
            (Only present if ``verbose=2`` in :meth:`~solve`.)
            Same shape as ``dict_cors``, but containing the average of the
            upper triagonal parts of the spatial correlation matrices (i.e. for each
            model, iteration, and component).
        """

    def solve(self, penalty, spatial_reweight_models, spatial_reweight_iters=5,
              spatial_reweight_percentile=0.5, spatial_reweight_max_rms=1e-10,
              spatial_reweight_max_changed=0, continuous_reweight_models=[],
              local_reweight_iters=1, local_reweight_func=None, local_reweight_coupled=True,
              formal_variance=False, formal_covariance=False, use_data_variance=True,
              use_data_covariance=True, use_internal_scales=False,
              verbose=False, extended_stats=False,
              cvxpy_kw_args={"solver": "CVXOPT", "kktsolver": "robust"}):
        r"""
        Solve the network-wide fitting problem as follows:

            1.  Fit the models individually using a single iteration step from
                :func:`~lasso_regression`.
            2.  Collect the L0 weights :math:`\mathbf{w}^{(i)}` from each station.
            3.  Spatially combine (e.g. take the median of) the weights,
                and redistribute them to the stations for the next iteration.
            4.  Repeat from 1.

        The iteration can stop early if either the conditions set by
        ``spatial_reweight_max_rms`` *or* ``spatial_reweight_max_changed`` are satisfied,
        for all models in ``spatial_reweight_models``.

        When done, it will save some key statistics to :attr:`~last_statistics`.

        Parameters
        ----------
        penalty : float
            Penalty hyperparameter :math:`\lambda`. If ``local_reweight_coupled=True``
            (default), this is just the penalty at the first iteration. After that, the
            penalties are largely controlled by ``local_reweight_func``.
        spatial_reweight_models : list
            Names of models to use in the spatial reweighting.
        spatial_reweight_iters : int, optional
            Number of spatial reweighting iterations.
        spatial_reweight_percentile : float, optional
            Percentile used in the spatial reweighting.
            Defaults to ``0.5``.
        spatial_reweight_max_rms : float, optional
            Stop the spatial iterations early if the difference in the RMS (Root Mean Square)
            of the change of the parameters between reweighting iterations is less than
            ``spatial_reweight_max_rms``. Defaults to no early stopping.
        spatial_reweight_max_changed : float, optional
            Stop the spatial iterations early if the number of changed parameters (i.e.,
            flipped between zero and non-zero) falls below a threshold. The threshold
            ``spatial_reweight_max_changed`` is given as the percentage of changed over total
            parameters (including all models and components). Defaults to no early stopping.
        continuous_reweight_models : list
            Names of models that should carry over their weights from one solver iteration
            to the next, but should not be reweighted.
        local_reweight_iters : int, optional
            Number of local reweighting iterations, see ``reweight_max_iters`` in
            :func:`~lasso_regression`.
        local_reweight_func : ReweightingFunction, optional
            An instance of a reweighting function that will be used by :func:`~lasso_regression`.
            Defaults to an inverse reweighting with stability parameter ``eps=1e-4``.
        local_reweight_coupled : bool, optional
            If ``True`` (default) and reweighting is active, the L1 penalty hyperparameter
            is coupled with the reweighting weights (see Notes in :func:`~lasso_regression`).
        formal_variance : bool, optional
            If ``True``, also calculate the formal variance (diagonals of the covariance
            matrix). Defaults to ``False``.
        formal_covariance : bool, optional
            If ``True``, also calculate the formal covariance between components
            (found in the off-diagonals of the covariance matrix). Defaults to ``False``.
        use_data_variance : bool, optional
            If ``True`` (default) and ``ts_description`` contains variance information, this
            uncertainty information will be used.
        use_data_covariance : bool, optional
            If ``True`` (default), ``ts_description`` contains variance and covariance
            information, and ``use_data_variance`` is also ``True``, this uncertainty
            information will be used.
        use_internal_scales : bool, optional
            Sets whether internal scaling should be used when reweighting, see
            ``use_internal_scales`` in :func:`~lasso_regression`.
        verbose : bool, optional
            If ``True`` (default: ``False``), print progress and statistics along the way.
        extended_stats : bool, optional
            If ``True`` (default: ``False``), the fitted models are evaluated at each iteration
            to calculate residual and fit statistics. These extended statistics are added to
            :attr:`~last_statistics` (see there for more details).
        cvxpy_kw_args : dict
            Additional keyword arguments passed on to CVXPY's ``solve()`` function,
            see ``cvxpy_kw_args`` in :func:`~lasso_regression`.
        """
        # input tests
        assert isinstance(spatial_reweight_models, list) and \
            all([isinstance(mdl, str) for mdl in spatial_reweight_models]), \
            "'spatial_reweight_models' must be a list of model name strings, got " + \
            f"{spatial_reweight_models}."
        assert isinstance(spatial_reweight_iters, int) and (spatial_reweight_iters > 0), \
            "'spatial_reweight_iters' must be an integer greater than 0, got " + \
            f"{spatial_reweight_iters}."
        assert float(spatial_reweight_max_rms) >= 0, "'spatial_reweight_max_rms' needs " \
            f"to be greater or equal to 0, got {spatial_reweight_max_rms}."
        assert 0 <= float(spatial_reweight_max_changed) <= 1, "'spatial_reweight_max_changed' " \
            f"needs to be between 0 and 1, got {spatial_reweight_max_changed}."
        if continuous_reweight_models != []:
            assert isinstance(continuous_reweight_models, list) and \
                all([isinstance(mdl, str) for mdl in continuous_reweight_models]), \
                "'continuous_reweight_models' must be a list of model name strings, got " + \
                f"{continuous_reweight_models}."
        all_reweight_models = set(spatial_reweight_models + continuous_reweight_models)
        assert len(all_reweight_models) == len(spatial_reweight_models) + \
            len(continuous_reweight_models), "'spatial_reweight_models' " + \
            "and 'continuous_reweight_models' can not have shared elements"

        # set up reweighting function
        if local_reweight_func is None:
            rw_func = ReweightingFunction.from_name("inv", 1e-4)
        else:
            assert isinstance(local_reweight_func, ReweightingFunction), "'local_reweight_func' " \
                f"needs to be None or a ReweightingFunction, got {type(local_reweight_func)}."
            rw_func = local_reweight_func

        # get scale lengths (correlation lengths)
        # using the average distance to the closest 4 stations
        if verbose:
            tqdm.write("Calculating scale lengths")
        geoid = cgeod.Geodesic()
        station_names = self.net.station_names
        station_lonlat = np.stack([np.array(self.net[name].location)[[1, 0]]
                                   for name in station_names])
        all_distances = np.empty((self.net.num_stations, self.net.num_stations))
        net_avg_closests = []
        for i, name in enumerate(station_names):
            all_distances[i, :] = np.array(geoid.inverse(station_lonlat[i, :].reshape(1, 2),
                                                         station_lonlat))[:, 0]
            net_avg_closests.append(np.sort(all_distances[i, :])[1:1+4].mean())
        distance_weights = np.exp(-all_distances / np.array(net_avg_closests).reshape(1, -1))
        # distance_weights is ignoring whether (1) a station actually has data, and
        # (2) if the spatial extent of the signal we're trying to estimate is correlated
        # to the station geometry

        # first solve, default initial weights
        results = self.net.fit(self.ts_description, model_list=self.model_list,
                               solver="lasso_regression", cached_mapping=True,
                               return_solutions=True,
                               progress_desc="Initial fit",
                               penalty=penalty,
                               reweight_max_iters=local_reweight_iters,
                               reweight_func=rw_func,
                               reweight_coupled=local_reweight_coupled,
                               return_weights=True,
                               formal_variance=formal_variance,
                               formal_covariance=formal_covariance,
                               use_data_variance=use_data_variance,
                               use_data_covariance=use_data_covariance,
                               use_internal_scales=use_internal_scales,
                               cvxpy_kw_args=cvxpy_kw_args)
        num_total = sum([s.models[self.ts_description][m].parameters.size
                         for s in self.net for m in all_reweight_models])
        num_uniques = np.sum(np.stack(
            [np.sum(np.any(np.stack([np.abs(s.models[self.ts_description][m].parameters)
                                     > self.ZERO for s in self.net]), axis=0), axis=0)
             for m in all_reweight_models]), axis=0)
        num_nonzero = sum([(s.models[self.ts_description][m].parameters.ravel()
                            > self.ZERO).sum()
                           for s in self.net for m in all_reweight_models])
        if verbose:
            tqdm.write(f"Number of reweighted non-zero parameters: {num_nonzero}/{num_total}")
            tqdm.write("Number of unique reweighted non-zero parameters per component: "
                       + str(num_uniques.tolist()))

        # initialize the other statistics objects
        num_components = num_uniques.size
        arr_uniques = np.empty((spatial_reweight_iters + 1, num_components))
        arr_uniques[:] = np.NaN
        arr_uniques[0, :] = num_uniques
        list_nonzeros = [np.NaN for _ in range(spatial_reweight_iters + 1)]
        list_nonzeros[0] = num_nonzero
        dict_rms_diff = {m: [np.NaN for _ in range(spatial_reweight_iters)]
                         for m in all_reweight_models}
        dict_num_changed = {m: [np.NaN for _ in range(spatial_reweight_iters)]
                            for m in all_reweight_models}

        # track parameters of weights of the reweighted models for early stopping
        old_params = {mdl_description:
                      Solution.aggregate_models(results_dict=results,
                                                mdl_description=mdl_description,
                                                key_list=station_names,
                                                stack_parameters=True)[0]
                      for mdl_description in all_reweight_models}

        if extended_stats:
            # initialize extra statistics variables
            list_res_stats = []
            dict_cors = {mdl_description: [] for mdl_description in all_reweight_models}
            dict_cors_means = {mdl_description: [] for mdl_description in all_reweight_models}
            dict_cors_det = {mdl_description: [] for mdl_description in all_reweight_models}
            dict_cors_det_means = {mdl_description: [] for mdl_description in all_reweight_models}

            # define a function to save space
            def save_extended_stats():
                iter_name_fit = self.ts_description + "_extendedstats_fit"
                iter_name_res = self.ts_description + "_extendedstats_res"
                # evaluate model fit to timeseries
                self.net.evaluate(self.ts_description, output_description=iter_name_fit)
                # calculate residuals
                self.net.math(iter_name_res, self.ts_description, "-", iter_name_fit)
                # analyze the residuals
                list_res_stats.append(
                    self.net.analyze_residuals(iter_name_res, mean=True, std=True,
                                               max_rolling_dev=self.ROLLMEANKERNEL))
                # for each reweighted model fit, for each component,
                # get its spatial correlation matrix and average value
                for mdl_description in all_reweight_models:
                    net_mdl_df = list(self.net.export_network_ts((self.ts_description,
                                                                  mdl_description)).values())
                    cormats = [mdl_df.df.corr().abs().values for mdl_df in net_mdl_df]
                    cormats_means = [np.nanmean(np.ma.masked_equal(np.triu(cormat, 1), 0))
                                     for cormat in cormats]
                    dict_cors[mdl_description].append(cormats)
                    dict_cors_means[mdl_description].append(cormats_means)
                    for i in range(len(net_mdl_df)):
                        raw_values = net_mdl_df[i].data.values
                        index_valid = np.isfinite(raw_values)
                        for j in range(raw_values.shape[1]):
                            raw_values[index_valid[:, j], j] = \
                                sp.signal.detrend(raw_values[index_valid[:, j], j])
                        net_mdl_df[i].data = raw_values
                    cormats = [mdl_df.df.corr().abs().values for mdl_df in net_mdl_df]
                    cormats_means = [np.nanmean(np.ma.masked_equal(np.triu(cormat, 1), 0))
                                     for cormat in cormats]
                    dict_cors_det[mdl_description].append(cormats)
                    dict_cors_det_means[mdl_description].append(cormats_means)
                # delete temporary timeseries
                self.net.remove_timeseries(iter_name_fit, iter_name_res)

            # run the function for the first time to capture the initial fit
            save_extended_stats()

        # iterate
        for i in range(spatial_reweight_iters):
            if verbose:
                tqdm.write("Updating weights")
            new_net_weights = {statname: {"reweight_init": {}} for statname in station_names}
            # reweighting spatial models
            for mdl_description in spatial_reweight_models:
                stacked_weights, = \
                    Solution.aggregate_models(results_dict=results,
                                              mdl_description=mdl_description,
                                              key_list=station_names,
                                              stack_weights=True)
                # if not None, stacking succeeded
                if np.any(stacked_weights):
                    if verbose:
                        tqdm.write(f"Stacking model {mdl_description}")
                        # print percentiles
                        percs = [np.nanpercentile(stacked_weights, q) for q in [5, 50, 95]]
                        tqdm.write(f"Weight percentiles (5-50-95): {percs}")
                    # now apply the spatial median to parameter weights
                    for station_index, name in enumerate(station_names):
                        new_net_weights[name]["reweight_init"][mdl_description] = \
                            weighted_median(stacked_weights,
                                            distance_weights[station_index, :],
                                            percentile=spatial_reweight_percentile)
                else:  # stacking failed, keep old weights
                    warn(f"{mdl_description} cannot be stacked, reusing old weights.")
                    for name in station_names:
                        if mdl_description in results[name]:
                            new_net_weights[name]["reweight_init"][mdl_description] = \
                                results[name][mdl_description].weights
            # copying over the old weights for the continuous models
            for mdl_description in continuous_reweight_models:
                for name in station_names:
                    if mdl_description in results[name]:
                        new_net_weights[name]["reweight_init"][mdl_description] = \
                            results[name][mdl_description].weights
            # next solver step
            results = self.net.fit(self.ts_description, model_list=self.model_list,
                                   solver="lasso_regression", cached_mapping=True,
                                   return_solutions=True,
                                   local_input=new_net_weights,
                                   progress_desc=f"Fit after {i+1} reweightings",
                                   penalty=penalty,
                                   reweight_max_iters=local_reweight_iters,
                                   reweight_func=rw_func,
                                   reweight_coupled=local_reweight_coupled,
                                   return_weights=True,
                                   formal_variance=formal_variance,
                                   formal_covariance=formal_covariance,
                                   use_data_variance=use_data_variance,
                                   use_data_covariance=use_data_covariance,
                                   cvxpy_kw_args=cvxpy_kw_args)
            # get statistics
            num_nonzero = sum([(s.models[self.ts_description][m].parameters.ravel()
                                > self.ZERO).sum()
                               for s in self.net for m in all_reweight_models])
            num_uniques = np.sum(np.stack(
                [np.sum(np.any(np.stack([np.abs(s.models[self.ts_description][m].parameters)
                                        > self.ZERO for s in self.net]), axis=0), axis=0)
                 for m in all_reweight_models]), axis=0)
            # save statistics
            arr_uniques[i+1, :] = num_uniques
            list_nonzeros[i+1] = num_nonzero
            # print
            if verbose:
                tqdm.write("Number of reweighted non-zero parameters: "
                           f"{num_nonzero}/{num_total}")
                tqdm.write("Number of unique reweighted non-zero parameters per component: "
                           + str(num_uniques.tolist()))
            # save extended statistics
            if extended_stats:
                save_extended_stats()
            # check for early stopping by comparing parameters that were reweighted
            early_stop = True
            for mdl_description in all_reweight_models:
                stacked_params, = \
                    Solution.aggregate_models(results_dict=results,
                                              mdl_description=mdl_description,
                                              key_list=station_names,
                                              stack_parameters=True)
                # check for early stopping criterion and save current parameters
                rms_diff = np.linalg.norm(old_params[mdl_description] - stacked_params)
                num_changed = np.logical_xor(np.abs(old_params[mdl_description]) < self.ZERO,
                                             np.abs(stacked_params) < self.ZERO).sum()
                early_stop &= (rms_diff < spatial_reweight_max_rms) or \
                              (num_changed/num_total < spatial_reweight_max_changed)
                old_params[mdl_description] = stacked_params
                # save statistics
                dict_rms_diff[mdl_description][i] = rms_diff
                dict_num_changed[mdl_description][i] = num_changed
                # print
                if verbose:
                    tqdm.write(f"RMS difference of '{mdl_description}' parameters = "
                               f"{rms_diff} ({num_changed} changed)")
            # check if early stopping was triggered
            if early_stop:
                if verbose:
                    tqdm.write("Stopping iteration early.")
                break

        if verbose:
            tqdm.write("Done")

        # save statistics to attribute as namedtuple
        last_stats_names = ["num_total", "arr_uniques", "list_nonzeros",
                            "dict_rms_diff", "dict_num_changed"]
        last_stats_values = [num_total, arr_uniques, list_nonzeros,
                             dict_rms_diff, dict_num_changed]
        if extended_stats:
            last_stats_names.extend(["list_res_stats", "dict_cors", "dict_cors_means",
                                     "dict_cors_det", "dict_cors_det_means"])
            last_stats_values.extend([list_res_stats, dict_cors, dict_cors_means,
                                      dict_cors_det, dict_cors_det_means])
        self.last_statistics = namedtuple("SpatialSolverStatistics",
                                          last_stats_names)(*last_stats_values)
