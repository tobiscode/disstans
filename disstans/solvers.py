"""
This module contains solver routines for fitting models to the timeseries
of stations.
"""

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import cvxpy as cp
from warnings import warn
from abc import ABC, abstractmethod

from .tools import block_permutation
from .models import ModelCollection


class Solution():
    r"""
    Class that contains the solution output by the solver functions
    in this module, and distributes the parameters, covariances, and weights (where
    present) into the respective models.

    Parameters
    ----------
    disstans.models.ModelCollection
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
        and ``False`` where the estimation was skipped due to observability or
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

        The maximum penalty (:math:`y`-intercept) can be approximated as
        :math:`\frac{1}{\text{eps}}`, and the minimum penalty approaches zero
        asymptotically.

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

        The maximum penalty (:math:`y`-intercept) can be approximated as
        :math:`\frac{1}{\text{eps}^2}`, and the minimum penalty approaches zero
        asymptotically.

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
        This reweighting function's calculated penalties for individual elements
        depends on the overall size and 1-norm of the input weight vector.

        The maximum penalty (:math:`y`-intercept) can be approximated as
        :math:`\log_\text{num_reg} \frac{\| \mathbf{m} \|_1}{\text{eps}}`.
        If there is only a single nonzero value, its penalty will be zero (at
        :math:`|m_j|=\| \mathbf{m} \|_1`).
        In the intermediate cases where multiple values are nonzero, their penalties will
        be distributed on a logarithmic slope between the :math:`y`-intercept and zero.

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
    Performs linear, unregularized least squares using :mod:`~scipy.linalg`.

    The timeseries are the observations :math:`\mathbf{d}`, and the models' mapping
    matrices are stacked together to form a single mapping matrix
    :math:`\mathbf{G}`. The solver then computes the model parameters
    :math:`\mathbf{m}` that minimize the cost function

    .. math:: f(\mathbf{m}) = \left\| \mathbf{Gm} - \mathbf{d} \right\|_2^2

    where :math:`\mathbf{\epsilon} = \mathbf{Gm} - \mathbf{d}` is the residual.

    If the observations :math:`\mathbf{d}` include a covariance matrix
    :math:`\mathbf{C}_d`, this information can be used. In this case,
    :math:`\mathbf{G}` and :math:`\mathbf{d}` are replaced by their weighted versions

    .. math:: \mathbf{G} \rightarrow \mathbf{G}^T \mathbf{C}_d^{-1} \mathbf{G}

    and

    .. math:: \mathbf{d} \rightarrow \mathbf{G}^T \mathbf{C}_d^{-1} \mathbf{d}

    The formal model covariance is defined as the pseudo-inverse

    .. math:: \mathbf{C}_m = \left( \mathbf{G}^T \mathbf{C}_d \mathbf{G} \right)^g

    Parameters
    ----------
    ts : disstans.timeseries.Timeseries
        Timeseries to fit.
    models : disstans.models.ModelCollection
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
                cov.append(sp.linalg.pinvh(GtWG))
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
            cov = sp.linalg.pinvh(GtWG)

    # create solution object and return
    return Solution(models=models, parameters=params, covariances=cov,
                    obs_indices=obs_indices)


def ridge_regression(ts, models, penalty, formal_covariance=False,
                     use_data_variance=True, use_data_covariance=True):
    r"""
    Performs linear, L2-regularized least squares using :mod:`~scipy.linalg`.

    The timeseries are the observations :math:`\mathbf{d}`, and the models' mapping
    matrices are stacked together to form a single mapping matrix
    :math:`\mathbf{G}`. Given the penalty hyperparameter :math:`\lambda`, the solver then
    computes the model parameters :math:`\mathbf{m}` that minimize the cost function

    .. math:: f(\mathbf{m}) = \left\| \mathbf{Gm} - \mathbf{d} \right\|_2^2
              + \lambda \left\| \mathbf{m}_\text{reg} \right\|_2^2

    where :math:`\mathbf{\epsilon} = \mathbf{Gm} - \mathbf{d}` is the residual
    and the subscript :math:`_\text{reg}` masks to zero the model parameters
    not designated to be regularized (see :attr:`~disstans.models.Model.regularize`).

    If the observations :math:`\mathbf{d}` include a covariance matrix
    :math:`\mathbf{C}_d`, this information can be used. In this case,
    :math:`\mathbf{G}` and :math:`\mathbf{d}` are replaced by their weighted versions

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
    ts : disstans.timeseries.Timeseries
        Timeseries to fit.
    models : disstans.models.ModelCollection
        Model collection used for fitting.
    penalty : float, list, numpy.ndarray
        Penalty hyperparameter :math:`\lambda`.
        It can either be a single value used for all components, or a list or NumPy array
        specifying a penalty for each component in the data.
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
    # check input penalty shape and value
    penalty = np.array(penalty).ravel()
    if penalty.size == 1:
        penalty = np.repeat(penalty, ts.num_components)
    elif penalty.size != ts.num_components:
        raise ValueError(f"'penalty' has a size of {penalty.size}, but needs to either "
                         "be a single value, or one value per timeseries component "
                         f"({ts.num_components}).")
    if np.any(penalty < 0) or np.all(penalty == 0):
        warn("Ridge Regression (L2-regularized) solver got an invalid penalty of "
             f"{penalty}; penalties should be non-negative, and at least contain "
             "one value larger than 0.", stacklevel=2)

    # get mapping and regularization matrix and sizes
    G, obs_indices, num_time, num_params, num_comps, num_obs, num_reg, reg_indices, _, _ = \
        models.prepare_LS(ts)

    # perform fit and estimate formal covariance (uncertainty) of parameters
    # if there is no covariance, it's num_comps independent problems
    if not formal_covariance:
        cov = None
    if (ts.cov_cols is None) or (not use_data_covariance):
        reg = np.diag(reg_indices)
        params = np.zeros((num_obs, num_comps))
        if formal_covariance:
            cov = []
        for i in range(num_comps):
            GtWG, GtWd = models.build_LS(ts, G, obs_indices, icomp=i,
                                         use_data_var=use_data_variance)
            GtWGreg = GtWG + reg * penalty[i]
            params[:, i] = sp.linalg.lstsq(GtWGreg, GtWd)[0].squeeze()
            if formal_covariance:
                cov.append(sp.linalg.pinvh(GtWGreg))
        if formal_covariance:
            cov = sp.linalg.block_diag(*cov)
            # permute to match ordering
            P = block_permutation(num_comps, num_params)
            cov = P @ cov @ P.T
    else:
        GtWG, GtWd = models.build_LS(ts, G, obs_indices, use_data_var=use_data_variance,
                                     use_data_cov=use_data_covariance)
        reg = np.diag((reg_indices.reshape(-1, 1) * penalty.reshape(1, -1)).ravel())
        GtWGreg = GtWG + reg
        params = sp.linalg.lstsq(GtWGreg, GtWd)[0].reshape(num_obs, num_comps)
        if formal_covariance:
            cov = sp.linalg.pinvh(GtWGreg)

    # create solution object and return
    return Solution(models=models, parameters=params, covariances=cov,
                    obs_indices=obs_indices)


def lasso_regression(ts, models, penalty, reweight_max_iters=None, reweight_func=None,
                     reweight_max_rss=1e-9, reweight_init=None, reweight_coupled=True,
                     formal_covariance=False, use_data_variance=True, use_data_covariance=True,
                     use_internal_scales=True, cov_zero_threshold=1e-6, return_weights=False,
                     check_constraints=True,
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
    not designated to be regularized (see :attr:`~disstans.models.Model.regularize`).

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
    unregularized version (see :func:`~disstans.solvers.linear_regression`), restricted to
    those same rows and columns. (This definition might very well be mathematically
    or algorithmically wrong - there probably needs to be some dependence on the
    reweighting function.)

    Parameters
    ----------
    ts : disstans.timeseries.Timeseries
        Timeseries to fit.
    models : disstans.models.ModelCollection
        Model collection used for fitting.
    penalty : float, list, numpy.ndarray
        Penalty hyperparameter :math:`\lambda`.
        It can either be a single value used for all components, or a list or NumPy array
        specifying a penalty for each component in the data.
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
    reweight_init : numpy.ndarray, dict, list, optional
        When reweighting is active, use this array to initialize the weights.
        It has to have size :math:`\text{num_components} * \text{num_reg}`, where
        :math:`\text{num_components}=1` if covariances are not used (and the actual
        number of timeseries components otherwise) and :math:`\text{num_reg}` is the
        number of regularized model parameters.
        It can be a single NumPy array or a list of NumPy arrays, in which case it
        (or the array created by concatenating the list) need to already have the right
        output shape (no check is performed). If it is a dictionary, the keys need to be
        model names, and the values are then the NumPy arrays which will be arranged
        properly to match the mapping matrix.
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
    check_constraints : bool, optional
        If ``True`` (default), check whether models have sign constraints that should
        be enforced.
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
    # check input penalty shape and value
    penalty = np.array(penalty).ravel()
    if penalty.size == 1:
        penalty = np.repeat(penalty, ts.num_components)
    elif penalty.size != ts.num_components:
        raise ValueError(f"'penalty' has a size of {penalty.size}, but needs to either "
                         "be a single value, or one value per timeseries component "
                         f"({ts.num_components}).")
    if np.any(penalty < 0) or np.all(penalty == 0):
        warn("Lasso Regression (L1-regularized) solver got an invalid penalty of "
             f"{penalty}; penalties should be non-negative, and at least contain "
             "one value larger than 0.", stacklevel=2)
    assert float(cov_zero_threshold) > 0, \
        f"'cov_zero_threshold needs to be non-negative, got {cov_zero_threshold}."

    # get mapping and regularization matrix
    G, obs_indices, num_time, num_params, num_comps, num_obs, num_reg, reg_indices, \
        reweight_init, weights_scaling, sign_constraints = models.prepare_LS(
            ts, reweight_init=reweight_init, use_internal_scales=True,
            check_constraints=check_constraints)
    # determine if a shortcut is possible
    regularize = (num_reg > 0) and (np.any(penalty > 0))
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
    if (ts.cov_cols is not None) and use_data_covariance:
        reg_indices = np.repeat(reg_indices, num_comps)
        weights_scaling = np.repeat(weights_scaling, num_comps)
    # create constraint indices
    if check_constraints and np.isfinite(sign_constraints).sum() > 0:
        nonneg_indices = sign_constraints > 0
        nonpos_indices = sign_constraints < 0
    else:
        check_constraints = False

    # solve CVXPY problem while checking for convergence
    def solve_problem(GtWG, GtWd, pen, num_comps, init_weights, nonneg, nonpos):
        # build objective function
        m = cp.Variable(GtWd.size)
        objective = cp.norm2(GtWG @ m - GtWd)
        constraints = []
        if check_constraints:
            if nonneg is not None:
                constraints.append(m[nonneg] >= 0)
            if nonpos is not None:
                constraints.append(m[nonpos] <= 0)
        if regularize:
            if reweight_max_iters is not None:
                reweight_size = num_reg * num_comps
                if init_weights is None:
                    init_weights = np.ones(reweight_size)
                    if reweight_coupled:
                        init_weights *= pen
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
                    lambd = cp.Parameter(shape=() if num_comps == 1 else pen.size,
                                         value=pen, pos=True)
                    objective = objective + cp.norm1(cp.multiply(lambd, z))
                constraints.append(z == cp.multiply(weights, m[reg_indices]))
                old_m = np.zeros(m.shape)
            else:
                lambd = cp.Parameter(shape=() if num_comps == 1 else pen.size,
                                     value=pen, pos=True)
                objective = objective + cp.norm1(cp.multiply(lambd, m[reg_indices]))
        # define problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        # solve
        for i in range(n_iters):  # always solve at least once
            try:
                problem.solve(enforce_dpp=True, **cvxpy_kw_args)
            except cp.error.SolverError as e:
                # no solution found, but actually a more serious problem
                warn(str(e), stacklevel=2)
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
            solution, wts = solve_problem(GtWG, GtWd, pen=penalty[i], num_comps=1,
                                          init_weights=reweight_init[:, i]
                                          if reweight_init is not None else None,
                                          nonneg=nonneg_indices[:, i]
                                          if check_constraints
                                          and nonneg_indices[:, i].sum() > 0
                                          else None,
                                          nonpos=nonpos_indices[:, i]
                                          if check_constraints
                                          and nonpos_indices[:, i].sum() > 0
                                          else None)
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
                    temp_cov = np.zeros_like(GtWG)
                    scaled_solution = np.abs(solution)
                    if weights_scaling is not None:
                        scaled_solution[reg_indices] *= weights_scaling
                    best_ind = np.nonzero(scaled_solution > cov_zero_threshold)[0]
                    Gsub = Gnonan[:, best_ind]
                    GtWG = Gsub.T @ Wnonan @ Gsub
                    if isinstance(GtWG, sparse.spmatrix):
                        GtWG = GtWG.A
                    temp_cov[np.ix_(best_ind, best_ind)] = sp.linalg.pinvh(GtWG)
                    cov.append(temp_cov)
                if regularize and return_weights:
                    weights[:, i] = wts
        if formal_covariance:
            cov = sp.linalg.block_diag(*cov)
            # permute to match ordering
            P = block_permutation(num_comps, num_obs)
            cov = P @ cov @ P.T
    else:
        # build stacked problem and solve
        Gnonan, Wnonan, GtWG, GtWd = models.build_LS(ts, G, obs_indices, return_W_G=True,
                                                     use_data_var=use_data_variance,
                                                     use_data_cov=use_data_covariance)
        solution, wts = solve_problem(GtWG, GtWd, pen=np.tile(penalty, num_reg),
                                      num_comps=num_comps,
                                      init_weights=reweight_init.ravel()
                                      if reweight_init is not None else None,
                                      nonneg=nonneg_indices.ravel()
                                      if check_constraints and nonneg_indices.sum() > 0
                                      else None,
                                      nonpos=nonpos_indices.ravel()
                                      if check_constraints and nonpos_indices.sum() > 0
                                      else None)
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
                    scaled_solution[reg_indices] *= weights_scaling
                best_ind = np.nonzero(scaled_solution > cov_zero_threshold)[0]
                Gsub = Gnonan.tocsc()[:, best_ind]
                GtWG = Gsub.T @ Wnonan @ Gsub
                if isinstance(GtWG, sparse.spmatrix):
                    GtWG = GtWG.A
                cov = np.zeros((num_obs * num_comps, num_obs * num_comps))
                cov[np.ix_(best_ind, best_ind)] = sp.linalg.pinvh(GtWG)
            if regularize and return_weights:
                weights = wts.reshape(num_reg, num_comps)
        # restore reg_indices' original shape
        reg_indices = reg_indices[::num_comps]

    # create solution object and return
    return Solution(models=models, parameters=params, covariances=cov, weights=weights,
                    obs_indices=obs_indices, reg_indices=reg_indices)
