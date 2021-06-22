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

from .tools import weighted_median, block_permutation
from .models import ModelCollection


class Solution():
    r"""
    Class that contains the solution output by the solver functions
    in this module, and distributes the parameters, covariances, and weights (where
    present) into the respective models.

    Parameters
    ----------
    geonat.models.ModelCollection
        Model collection object that describes which models were used by the solver.
    parameters : numpy.ndarray
        Model collection parameters of shape
        :math:`(\text{num_solved}, \text{num_components})`.
    covariances : numpy.ndarray, optional
        Full model variance-covariance matrix that has a square shape with dimensions
        :math:`\text{num_solved} * \text{num_components}`.
    weights : numpy.ndarray, optional
        Model parameter regularization weights of shape
        :math:`(\text{num_solved}, \text{num_components})`.
    obs_indices : numpy.ndarray, optional
        Observation mask of shape :math:`(\text{num_parameters}, )`
        with ``True`` at indices where the parameter was actually estimated,
        and ``False`` where the estimation was skipped du to observability or
        other reasons.
        Defaults to all ``True``, which implies that
        :math:`\text{num_parameters} = \text{num_solved}`.
    reg_indices : numpy.ndarray, optional
        Regularization mask of shape :math:`(\text{num_solved}, )`
        with ``True`` where a parameter was subject to regularization,
        and ``False`` otherwise. Defaults to all ``False``.
    """
    def __init__(self, models, parameters, covariances=None, weights=None,
                 obs_indices=None, reg_indices=None):
        # input checks
        assert isinstance(models, ModelCollection), \
            f"'models' is not a valid ModelCollection object, got {type(models)}."
        input_types = [None if indat is None else type(indat) for indat
                       in [parameters, covariances, weights, obs_indices, reg_indices]]
        assert all([(intype is None) or (intype == np.ndarray) for intype in input_types]), \
            f"Unsupported input data types where not None: {input_types}."
        if reg_indices is not None:
            assert reg_indices.size == parameters.shape[0], \
                "Unexpected parameter size mismatch: " \
                f"{reg_indices.size} != {parameters.shape}[0]"
        # get the number of parameters and components, and match them with obs_indices
        num_components = parameters.shape[1]
        num_parameters = models.num_parameters
        if obs_indices is None:
            assert parameters.shape[0] == num_parameters, \
                "No 'obs_indices' was passed, but the shape of 'parameters' " \
                f"{parameters.shape} does not match the total number of parameters " \
                f"{num_parameters} in its first dimension."
            obs_indices = np.s_[:]
        # skip the packing of weights if there isn't any regularization or weights
        if np.any(weights) and np.any(reg_indices):
            pack_weights = True
            # ix_reg = 0
        else:
            pack_weights = False
        # make full arrays to fill with values
        par_full = np.empty((num_parameters, num_components))
        par_full[:] = np.NaN
        par_full[obs_indices, :] = parameters
        if covariances is not None:
            params_times_comps = num_parameters * num_components
            cov_full = np.empty((params_times_comps, params_times_comps))
            cov_full[:] = np.NaN
            mask_cov = np.repeat(obs_indices, num_components)
            cov_full[np.ix_(mask_cov, mask_cov)] = covariances
        if pack_weights:
            weights_full = np.empty((num_parameters, num_components))
            weights_full[:] = np.NaN
            weights_full[np.flatnonzero(obs_indices)[reg_indices], :] = weights
        # start the iteration over the models
        model_ix_start_len = {}
        ix_model = 0
        # ix_sol = 0
        for (mdl_description, model) in models.collection.items():
            # save index range
            model_ix_start_len[mdl_description] = (ix_model, model.num_parameters)
            ix_model += model.num_parameters
        # save results to class instance
        self._model_slice_ranges = model_ix_start_len
        self.parameters = par_full
        """
        Full parameter matrix, where unsolved and unobservable parameters are
        set to ``NaN``.
        """
        self.covariances = cov_full if covariances is not None else None
        """
        Full covariance matrix, where unsolved and unobservable covariances are
        set to ``NaN``.
        """
        self.weights = weights_full if pack_weights else None
        """ Full weights matrix, where unsolved weights are set to ``NaN``. """
        self.obs_mask = obs_indices
        """ Observability mask where ``True`` indicates observable parameters. """
        self.num_parameters = num_parameters
        """ Total number of parameters (solved or unsolved) in solution. """
        self.num_components = num_components
        """ Number of components the solution was computed for. """
        self.converged = np.all(np.isfinite(parameters))
        """
        ``True``, if all values in the estimated parameters were finite (i.e., the solution
        converged), ``False`` otherwise.
        """

    def get_model_indices(self, models, for_cov=False):
        """
        Given a model name or multiple names, returns an array of integer indices that
        can be used to extract the relevant entries from :attr:`~parameters` and
        :attr:`~covariances`.

        Parameters
        ----------
        models : str, list
            Name(s) of model(s).
        for_cov : bool, optional
            If ``False`` (default), return the indices for :attr:`~parameters`, otherwise
            for :attr:`~covariances`

        Returns
        -------
        numpy.ndarray
            Integer index array for the models.
        """
        # (returns empty index array if models not found)
        # check input
        if isinstance(models, str):
            model_list = [models]
        elif isinstance(models, list):
            model_list = models
        else:
            raise ValueError("Model key(s) need to be a string or list of strings, "
                             f"got {models}.")
        # build combined slice
        nc = self.num_components if for_cov else 1
        combined_ranges = np.concatenate([np.arange(ix*nc, (ix+num)*nc) for m, (ix, num)
                                          in self._model_slice_ranges.items()
                                          if m in model_list]).astype(int)
        return np.sort(combined_ranges)

    def parameters_by_model(self, models, zeroed=False):
        """
        Helper function that uses :meth:`~get_model_indices` to quickly
        return the parameters for (a) specific model(s).

        Parameters
        ----------
        models : str, list
            Name(s) of model(s).
        zeroed : bool, optional
            If ``False`` (default), use :attr:`~parameters`, else
            :attr:`~parameters_zeroed`.

        Returns
        -------
        numpy.ndarray
            Parameters of the model subset.
        """
        indices = self.get_model_indices(models)
        if zeroed:
            return self.parameters_zeroed[indices, :]
        else:
            return self.parameters[indices, :]

    def covariances_by_model(self, models, zeroed=False):
        """
        Helper function that uses :meth:`~get_model_indices` to quickly
        return the covariances for (a) specific model(s).

        Parameters
        ----------
        models : str, list
            Name(s) of model(s).
        zeroed : bool, optional
            If ``False`` (default), use :attr:`~covariances`, else
            :attr:`~covariances_zeroed`.

        Returns
        -------
        numpy.ndarray
            Covariances of the model subset.
        """
        if self.covariances is not None:
            indices = self.get_model_indices(models, for_cov=True)
            if zeroed:
                return self.covariances_zeroed[np.ix_(indices, indices)]
            else:
                return self.covariances[np.ix_(indices, indices)]

    def weights_by_model(self, models):
        """
        Helper function that uses :meth:`~get_model_indices` to quickly
        return the weights for specific model parameters.

        Parameters
        ----------
        models : str, list
            Name(s) of model(s).

        Returns
        -------
        numpy.ndarray
            Weights of the model parameter subset.
        """
        if self.weights is not None:
            indices = self.get_model_indices(models)
            return self.weights[indices, :]

    @property
    def parameters_zeroed(self):
        """
        Returns the model parameters but sets the unobservable ones to zero
        to distinguish between unobservable ones and observable ones that
        could not be estimated because of a solver failure (and which are NaNs).
        """
        par = self.parameters.copy()
        par[~self.obs_mask, :] = 0
        return par

    @property
    def covariances_zeroed(self):
        """
        Returns the model covariances but sets the unobservable ones to zero
        to distinguish between unobservable ones and observable ones that
        could not be estimated because of a solver failure (and which are NaNs).
        """
        if self.covariances is not None:
            cov = self.covariances.copy()
            unobs_indices = np.repeat(~self.obs_mask, self.num_components)
            cov[unobs_indices, :] = 0
            cov[:, unobs_indices] = 0
            return cov

    @property
    def model_list(self):
        """ List of models present in the solution. """
        return list(self._model_slice_ranges.keys())

    @staticmethod
    def aggregate_models(results_dict, mdl_description, key_list=None,
                         stack_parameters=False, stack_covariances=False,
                         stack_weights=False, zeroed=False):
        """
        For a dictionary of Solution objects (e.g. one per station) and a given
        model description, aggregate the model parameters, variances and parameter
        regularization weights (where present) into combined NumPy arrays.

        If the shape of the individual parameters etc. do not match between objects
        in the dictionary for the same model, ``None`` is returned instead, without
        raising an error (e.g. if the same model is site-specific and has different
        numbers of parameters).

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
        stack_covariances : bool, optional
            If ``True``, stack the covariances, otherwise just return ``None``.
            Defaults to ``False``.
        stack_weights : bool, optional
            If ``True``, stack the weights, otherwise just return ``None``.
            Defaults to ``False``.
        zeroed : bool, optional
            If ``False`` (default), use :attr:`~parameters` and :attr:`~covariances`,
            else :attr:`~parameters_zeroed` and :attr:`~covariances_zeroed`.

        Returns
        -------
        numpy.ndarray
            If ``stack_parameters=True``, the stacked model parameters.
        numpy.ndarray
            If ``stack_covariances=True`` and covariances are present in the models,
            the stacked component covariances, ``None`` if not present everywhere.
        numpy.ndarray
            If ``stack_weights=True`` and regularization weights are present in the models,
            the stacked weights, ``None`` if not present everywhere.
        """
        # input checks
        assert isinstance(mdl_description, str), \
            f"'mdl_description' needs to be a single string, got {mdl_description}."
        assert any([stack_parameters, stack_covariances, stack_weights]), \
            "Called 'aggregate_models' without anything to aggregate."
        assert (isinstance(results_dict, dict) and
                all([isinstance(mdl_sol, Solution) for mdl_sol in results_dict.values()])), \
            f"'results_dict' needs to be a dictionary of Solution objects, got {results_dict}."
        if key_list is None:
            key_list = list(results_dict.keys())
        # collect output
        out = []
        # parameters
        if stack_parameters:
            stack = [results_dict[key].parameters_by_model(mdl_description, zeroed=zeroed)
                     for key in key_list]
            stack_shapes = [mdl.shape for mdl in stack]
            if (len(stack) > 0) and (stack_shapes.count(stack_shapes[0]) == len(stack)):
                out.append(np.stack(stack))
            else:
                out.append(None)
        # covariances
        if stack_covariances:
            stack = [results_dict[key].covariances_by_model(mdl_description, zeroed=zeroed)
                     for key in key_list if results_dict[key].covariances is not None]
            stack_shapes = [mdl.shape for mdl in stack]
            if (len(stack) > 0) and (stack_shapes.count(stack_shapes[0]) == len(stack)):
                out.append(np.stack(stack))
            else:
                out.append(None)
        # weights
        if stack_weights:
            stack = [results_dict[key].weights_by_model(mdl_description)
                     for key in key_list if results_dict[key].weights is not None]
            stack_shapes = [mdl.shape for mdl in stack]
            if (len(stack) > 0) and (stack_shapes.count(stack_shapes[0]) == len(stack)):
                out.append(np.stack(stack))
            else:
                out.append(None)
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


def linear_regression(ts, models, formal_covariance=False,
                      use_data_variance=True, use_data_covariance=True):
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
    models : geonat.models.ModelCollection
        Model collection used for fitting.
    formal_covariance : bool, optional
        If ``True``, calculate the formal model covariance. Defaults to ``False``.
    use_data_variance : bool, optional
        If ``True`` (default) and ``ts`` contains variance information, this
        uncertainty information will be used.
    use_data_covariance : bool, optional
        If ``True`` (default), ``ts`` contains variance and covariance information, and
        ``use_data_variance`` is also ``True``, this uncertainty information will be used.

    Returns
    -------
    Solution
        Result of the regression.
    """

    # get mapping matrix and sizes
    G, obs_indices, num_time, num_params, num_comps, num_obs = \
        models.prepare_LS(ts, include_regularization=False)

    # perform fit and estimate formal covariance (uncertainty) of parameters
    # if there is no covariance, it's num_comps independent problems
    if not formal_covariance:
        cov = None
    if (ts.cov_cols is None) or (not use_data_covariance):
        params = np.zeros((num_obs, num_comps))
        if formal_covariance:
            cov = []
        for i in range(num_comps):
            GtWG, GtWd = models.build_LS(ts, G, obs_indices, icomp=i,
                                         use_data_var=use_data_variance)
            params[:, i] = sp.linalg.lstsq(GtWG, GtWd)[0].squeeze()
            if formal_covariance:
                cov.append(np.linalg.inv(GtWG))
        if formal_covariance:
            cov = sp.linalg.block_diag(*cov)
            # permute to match ordering
            P = block_permutation(num_comps, num_params)
            cov = P @ cov @ P.T
    else:
        GtWG, GtWd = models.build_LS(ts, G, obs_indices, use_data_var=use_data_variance,
                                     use_data_cov=use_data_covariance)
        params = sp.linalg.lstsq(GtWG, GtWd)[0].reshape(num_obs, num_comps)
        if formal_covariance:
            cov = np.linalg.inv(GtWG)

    # create solution object and return
    return Solution(models=models, parameters=params, covariances=cov,
                    obs_indices=obs_indices)


def ridge_regression(ts, models, penalty, formal_covariance=False,
                     use_data_variance=True, use_data_covariance=True):
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
    models : geonat.models.ModelCollection
        Model collection used for fitting.
    penalty : float
        Penalty hyperparameter :math:`\lambda`.
    formal_covariance : bool, optional
        If ``True``, calculate the formal model covariance. Defaults to ``False``.
    use_data_variance : bool, optional
        If ``True`` (default) and ``ts`` contains variance information, this
        uncertainty information will be used.
    use_data_covariance : bool, optional
        If ``True`` (default), ``ts`` contains variance and covariance information, and
        ``use_data_variance`` is also ``True``, this uncertainty information will be used.

    Returns
    -------
    Solution
        Result of the regression.
    """
    if penalty == 0.0:
        warn(f"Ridge Regression (L2-regularized) solver got a penalty of {penalty}, "
             "which effectively removes the regularization.")

    # get mapping and regularization matrix and sizes
    G, obs_indices, num_time, num_params, num_comps, num_obs, num_reg, reg_indices, _, _ = \
        models.prepare_LS(ts)

    # perform fit and estimate formal covariance (uncertainty) of parameters
    # if there is no covariance, it's num_comps independent problems
    if not formal_covariance:
        cov = None
    if (ts.cov_cols is None) or (not use_data_covariance):
        reg = np.diag(reg_indices) * penalty
        params = np.zeros((num_obs, num_comps))
        if formal_covariance:
            cov = []
        for i in range(num_comps):
            GtWG, GtWd = models.build_LS(ts, G, obs_indices, icomp=i,
                                         use_data_var=use_data_variance)
            GtWGreg = GtWG + reg
            params[:, i] = sp.linalg.lstsq(GtWGreg, GtWd)[0].squeeze()
            if formal_covariance:
                cov.append(np.linalg.inv(GtWGreg))
        if formal_covariance:
            cov = sp.linalg.block_diag(*cov)
            # permute to match ordering
            P = block_permutation(num_comps, num_params)
            cov = P @ cov @ P.T
    else:
        GtWG, GtWd = models.build_LS(ts, G, obs_indices, use_data_var=use_data_variance,
                                     use_data_cov=use_data_covariance)
        reg = np.diag(np.repeat(reg_indices, num_comps)) * penalty
        GtWGreg = GtWG + reg
        params = sp.linalg.lstsq(GtWGreg, GtWd)[0].reshape(num_obs, num_comps)
        if formal_covariance:
            cov = np.linalg.inv(GtWGreg)

    # create solution object and return
    return Solution(models=models, parameters=params, covariances=cov,
                    obs_indices=obs_indices)


def lasso_regression(ts, models, penalty, reweight_max_iters=None, reweight_func=None,
                     reweight_max_rss=1e-9, reweight_init=None, reweight_coupled=True,
                     formal_covariance=False, use_data_variance=True, use_data_covariance=True,
                     use_internal_scales=True, cov_zero_threshold=1e-6, return_weights=False,
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
    the rows and columns corresponding to non-zero parameters (as defined my the absolute
    magnitude set by ``cov_zero_threshold``), where it is defined exactly as the
    unregularized version (see :func:`~geonat.solvers.linear_regression`), restricted to
    those same rows and columns. (This definition might very well be mathematically
    or algorithmically wrong - there probably needs to be some dependence on the
    reweighting function.)

    Parameters
    ----------
    ts : geonat.timeseries.Timeseries
        Timeseries to fit.
    models : geonat.models.ModelCollection
        Model collection used for fitting.
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
        Set to ``0`` to deactivate early stopping.
    reweight_init : numpy.ndarray, optional
        When reweighting is active, use this array to initialize the weights.
        It has to have size :math:`\text{num_components} * \text{num_reg}`, where
        :math:`\text{num_components}=1` if covariances are not used (and the actual
        number of timeseries components otherwise) and :math:`\text{num_reg}` is the
        number of regularized model parameters.
    reweight_coupled : bool, optional
        If ``True`` (default) and reweighting is active, the L1 penalty hyperparameter
        is coupled with the reweighting weights (see Notes).
    formal_covariance : bool, optional
        If ``True``, calculate the formal model covariance. Defaults to ``False``.
    use_data_variance : bool, optional
        If ``True`` (default) and ``ts`` contains variance information, this
        uncertainty information will be used.
    use_data_covariance : bool, optional
        If ``True`` (default), ``ts`` contains variance and covariance information, and
        ``use_data_variance`` is also ``True``, this uncertainty information will be used.
    use_internal_scales : bool, optional
        If ``True`` (default), the reweighting takes into account potential
        model-specific internal scaling parameters, otherwise ignores them.
    cov_zero_threshold : float, optional
        When extracting the formal covariance matrix, assume parameters with absolute
        values smaller than ``cov_zero_threshold`` are effectively zero.
        Internal scales are always respected.
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
    Solution
        Result of the regression.

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
    assert float(cov_zero_threshold) > 0, \
        f"'cov_zero_threshold needs to be non-negative, got {cov_zero_threshold}."

    # get mapping and regularization matrix
    G, obs_indices, num_time, num_params, num_comps, num_obs, num_reg, reg_indices, \
        reweight_init, weights_scaling = models.prepare_LS(
            ts, reweight_init=reweight_init, use_internal_scales=True)
    # determine if a shortcut is possible
    regularize = (num_reg > 0) and (penalty > 0)
    if (not regularize) or (reweight_max_iters is None):
        return_weights = False
    # determine number of maximum iterations, and check reweighting function
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
    # determine if reg_indices and weights_scaling need a reshape
    if (ts.cov_cols is None) or (not use_data_covariance):
        pass
    else:
        reg_indices = np.repeat(reg_indices, num_comps)
        weights_scaling = np.repeat(weights_scaling, num_comps)

    # solve CVXPY problem while checking for convergence
    def solve_problem(GtWG, GtWd, num_comps, init_weights):
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
                        f"got {init_weights.size}."
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
                    if use_internal_scales and (weights_scaling is not None):
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
    if not formal_covariance:
        cov = None
    if not return_weights:
        weights = None
    if (ts.cov_cols is None) or (not use_data_covariance):
        # initialize output
        params = np.zeros((num_obs, num_comps))
        if formal_covariance:
            cov = []
        if regularize and return_weights:
            weights = np.zeros((num_reg, num_comps))
        # loop over components
        for i in range(num_comps):
            # build and solve problem
            Gnonan, Wnonan, GtWG, GtWd = models.build_LS(
                ts, G, obs_indices, icomp=i, return_W_G=True, use_data_var=use_data_variance)
            solution, wts = solve_problem(GtWG, GtWd, num_comps=1,
                                          init_weights=reweight_init[:, i]
                                          if reweight_init is not None else None)
            # store results
            if solution is None:
                params[:, i] = np.NaN
                if formal_covariance:
                    temp_cov = np.empty_like(GtWG)
                    temp_cov[:] = np.NaN
                    cov.append(temp_cov)
                if regularize and return_weights:
                    weights[:, i] = np.NaN
            else:
                params[:, i] = solution
                # if desired, estimate formal variance here
                if formal_covariance:
                    temp_cov = np.empty_like(GtWG)
                    temp_cov[:] = np.NaN
                    scaled_solution = np.abs(solution)
                    if weights_scaling is not None:
                        scaled_solution[reg_indices] *= weights_scaling
                    best_ind = np.nonzero(scaled_solution > cov_zero_threshold)[0]
                    Gsub = Gnonan[:, best_ind]
                    GtWG = Gsub.T @ Wnonan @ Gsub
                    if isinstance(GtWG, sparse.spmatrix):
                        GtWG = GtWG.A
                    temp_cov[np.ix_(best_ind, best_ind)] = np.linalg.inv(GtWG)
                    cov.append(temp_cov)
                if regularize and return_weights:
                    weights[:, i] = wts
        if formal_covariance:
            cov = sp.linalg.block_diag(*cov)
            # permute to match ordering
            P = block_permutation(num_comps, num_params)
            cov = P @ cov @ P.T
    else:
        # build stacked problem and solve
        Gnonan, Wnonan, GtWG, GtWd = models.build_LS(ts, G, obs_indices, return_W_G=True,
                                                     use_data_var=use_data_variance,
                                                     use_data_cov=use_data_covariance)
        solution, wts = solve_problem(GtWG, GtWd, num_comps=num_comps,
                                      init_weights=reweight_init.ravel()
                                      if reweight_init is not None else None)
        # store results
        if solution is None:
            params = np.empty((num_obs, num_comps))
            params[:] = np.NaN
            if formal_covariance:
                cov = np.empty((num_obs * num_comps, num_obs * num_comps))
                cov[:] = np.NaN
            if regularize and return_weights:
                weights = np.empty((num_reg, num_comps))
                weights[:] = np.NaN
        else:
            params = solution.reshape(num_obs, num_comps)
            # if desired, estimate formal variance here
            if formal_covariance:
                scaled_solution = np.abs(solution)
                if weights_scaling is not None:
                    scaled_solution[reg_indices] *= np.repeat(weights_scaling, num_comps)
                best_ind = np.nonzero(scaled_solution > cov_zero_threshold)[0]
                Gsub = Gnonan.tocsc()[:, best_ind]
                GtWG = Gsub.T @ Wnonan @ Gsub
                if isinstance(GtWG, sparse.spmatrix):
                    GtWG = GtWG.A
                cov = np.zeros((num_obs * num_comps, num_obs * num_comps))
                cov[np.ix_(best_ind, best_ind)] = np.linalg.inv(GtWG)
            if regularize and return_weights:
                weights = wts.reshape(num_reg, num_comps)
        # restore reg_indices' original shape
        reg_indices = reg_indices[::num_comps]

    # create solution object and return
    return Solution(models=models, parameters=params, covariances=cov, weights=weights,
                    obs_indices=obs_indices, reg_indices=reg_indices)


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

    def __init__(self, net, ts_description):
        self.net = net
        """ Network object to fit. """
        self.ts_description = ts_description
        """ Name of timeseries to fit. """
        self.valid_stations = {name: station for name, station in net.stations.items()
                               if ts_description in station.timeseries}
        """ Dictionary of all stations that contain the timeseries. """
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

    def solve(self, penalty, spatial_reweight_models, spatial_reweight_iters,
              spatial_reweight_percentile=0.5, spatial_reweight_max_rms=1e-9,
              spatial_reweight_max_changed=0, continuous_reweight_models=[],
              local_reweight_iters=1, local_reweight_func=None, local_reweight_coupled=True,
              formal_covariance=False, use_data_variance=True, use_data_covariance=True,
              use_internal_scales=True, cov_zero_threshold=1e-6, verbose=False,
              extended_stats=False, cvxpy_kw_args={"solver": "CVXOPT", "kktsolver": "robust"}):
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
        spatial_reweight_iters : int
            Number of spatial reweighting iterations.
        spatial_reweight_percentile : float, optional
            Percentile used in the spatial reweighting.
            Defaults to ``0.5``.
        spatial_reweight_max_rms : float, optional
            Stop the spatial iterations early if the difference in the RMS (Root Mean Square)
            of the change of the parameters between reweighting iterations is less than
            ``spatial_reweight_max_rms``.
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
        formal_covariance : bool, optional
            If ``True``, calculate the formal model covariance. Defaults to ``False``.
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
        cov_zero_threshold : float, optional
            When extracting the formal covariance matrix, assume parameters with absolute
            values smaller than ``cov_zero_threshold`` are effectively zero.
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
        station_names = list(self.valid_stations.keys())
        num_stations = len(station_names)
        station_lonlat = np.stack([np.array(self.net[name].location)[[1, 0]]
                                   for name in station_names])
        all_distances = np.empty((num_stations, num_stations))
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
        if verbose:
            tqdm.write("Initial fit")
        results = self.net.fit(self.ts_description,
                               solver="lasso_regression",
                               return_solutions=True,
                               progress_desc=None if verbose else "Initial fit",
                               penalty=penalty,
                               reweight_max_iters=local_reweight_iters,
                               reweight_func=rw_func,
                               reweight_coupled=local_reweight_coupled,
                               return_weights=True,
                               formal_covariance=formal_covariance,
                               use_data_variance=use_data_variance,
                               use_data_covariance=use_data_covariance,
                               use_internal_scales=use_internal_scales,
                               cov_zero_threshold=cov_zero_threshold,
                               cvxpy_kw_args=cvxpy_kw_args)
        num_total = sum([s.models[self.ts_description][m].parameters.size
                         for s in self.valid_stations.values() for m in all_reweight_models])
        num_uniques = np.sum(np.stack(
            [np.sum(np.any(np.stack([np.abs(s.models[self.ts_description][m].parameters)
                                     > self.ZERO for s in self.valid_stations.values()]),
                           axis=0), axis=0) for m in all_reweight_models]), axis=0)
        num_nonzero = sum([(s.models[self.ts_description][m].parameters.ravel()
                            > self.ZERO).sum()
                           for s in self.valid_stations.values() for m in all_reweight_models])
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
                                                stack_parameters=True,
                                                zeroed=True)[0]
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
                        tqdm.write("Weight percentiles (5-50-95): "
                                   f"[{percs[0]:.11g}, {percs[1]:.11g}, {percs[2]:.11g}]")
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
            if verbose:
                tqdm.write(f"Fit after {i+1} reweightings")
            results = self.net.fit(self.ts_description,
                                   solver="lasso_regression",
                                   return_solutions=True,
                                   local_input=new_net_weights,
                                   progress_desc=None if verbose
                                   else f"Fit after {i+1} reweightings",
                                   penalty=penalty,
                                   reweight_max_iters=local_reweight_iters,
                                   reweight_func=rw_func,
                                   reweight_coupled=local_reweight_coupled,
                                   return_weights=True,
                                   formal_covariance=formal_covariance,
                                   use_data_variance=use_data_variance,
                                   use_data_covariance=use_data_covariance,
                                   cov_zero_threshold=cov_zero_threshold,
                                   cvxpy_kw_args=cvxpy_kw_args)
            # get statistics
            num_nonzero = sum([(s.models[self.ts_description][m].parameters.ravel()
                                > self.ZERO).sum()
                               for s in self.valid_stations.values() for m in all_reweight_models])
            num_uniques = np.sum(np.stack(
                [np.sum(np.any(np.stack([np.abs(s.models[self.ts_description][m].parameters)
                                        > self.ZERO for s in self.valid_stations.values()]),
                               axis=0), axis=0) for m in all_reweight_models]), axis=0)
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
                                              stack_parameters=True,
                                              zeroed=True)
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
                               f"{rms_diff:.11g} ({num_changed} changed)")
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
