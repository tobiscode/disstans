"""
This module contains solver routines for fitting models to the timeseries
of stations.
"""

import numpy as np
import scipy.sparse as sparse
import cvxpy as cp
from warnings import warn


def _combine_mappings(ts, models, reg_indices=False):
    """
    Quick helper function that concatenates the mapping matrices of the
    models given the timevector in ts, and returns the relevant sizes.
    If reg_indices = True, also return an array indicating which model
    is set to be regularized.
    """
    mapping_matrices = []
    if reg_indices:
        reg_diag = []
    for (mdl_description, model) in models.items():
        mapping_matrices.append(model.get_mapping(ts.time))
        if reg_indices:
            reg_diag.extend([model.regularize for _ in range(model.num_parameters)])
    G = sparse.hstack(mapping_matrices, format='csr')
    num_time, num_params = G.shape
    num_comps = ts.num_components
    if reg_indices:
        reg_diag = np.array(reg_diag)
        if reg_diag.sum() == 0:
            warn(f"Regularized solver got no models to regularize.")
        return G, reg_diag, num_time, num_params, num_comps
    else:
        return G, num_time, num_params, num_comps


def _build_LS(ts, G, icomp=None, return_W_G=False):
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
        d = sparse.csc_matrix(ts.df[ts.data_cols[icomp]].values.reshape(-1, 1))
        Gout = G
        if ts.var_cols is not None:
            W = sparse.diags(1/ts.df[ts.var_cols[icomp]].values)
        else:
            W = sparse.eye(d.size)
    else:
        d = sparse.csc_matrix(ts.data.values.reshape(-1, 1))
        Gout = sparse.kron(G, sparse.eye(num_comps), format='csr')
        if ts.cov_cols is not None:
            Wblocks = [1/ts.var_cov.values[iobs, ts.var_cov_map].reshape(num_comps, num_comps)
                       for iobs in range(ts.num_observations)]
            W = sparse.block_diag(Wblocks, format='dia')
        elif ts.var_cols is not None:
            W = sparse.diags(1/ts.vars.values.reshape(-1, 1))
        else:
            W = sparse.eye(d.size)
    dnan = np.isnan(d.A)
    if dnan.sum() > 0:
        dnotnan = ~dnan.squeeze()
        d = d[dnotnan]
        # csr-sparse matrices can be easily slices by rows
        Gout = Gout[dnotnan, :]
        # the same is not (yet?) possible for dia-matrices,
        # so in order not to have to create a dense matrix, this is the simplest way:
        W = W.tocsr()[dnotnan, :].tocsc()[:, dnotnan].todia()
        # double-check
        if np.any(np.isnan(Gout.data)) or np.any(np.isnan(W.data)):
            raise ValueError("Still NaNs in G or W, unexpected error!")
    GtW = Gout.T @ W
    GtWG = GtW @ Gout
    GtWd = (GtW @ d).toarray().squeeze()
    if return_W_G:
        return Gout, W, GtWG, GtWd
    else:
        return GtWG, GtWd


def _pack_params_var(models, params, var):
    """
    Quick helper function that distributes the parameters (and variances)
    of the input matrices into the respective models.
    """
    i = 0
    model_params_var = {}
    for (mdl_description, model) in models.items():
        model_params_var[mdl_description] = (params[i:i+model.num_parameters, :],
                                             None if var is None
                                             else var[i:i+model.num_parameters, :])
        i += model.num_parameters
    return model_params_var


def linear_regression(ts, models, formal_variance=False, lsmr_kw_args={}):
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
        Dictionary of :class:`~geonat.model.Model` instances used for fitting.
    formal_variance : bool, optional
        If ``True``, also calculate the formal variance (diagonals of the covariance
        matrix).
    lsmr_kw_args : dict
        Additional keyword arguments passed on to SciPys's
        :func:`~scipy.sparse.linalg.lsmr` function.

    Returns
    -------
    model_params_var : dict
        Dictionary of form ``{"model_description": (parameters, variance), ...}``
        which for every model that was fitted, contains a tuple of the best-fit
        parameters and the formal variance (or ``None``, if not calculated).
    """
    # get mapping matrix and sizes
    G, num_time, num_params, num_comps = _combine_mappings(ts, models)

    # perform fit and estimate formal covariance (uncertainty) of parameters
    # if there is no covariance, it's num_comps independent problems
    if ts.cov_cols is None:
        params = np.zeros((num_params, num_comps))
        if formal_variance:
            var = np.zeros((num_params, num_comps))
        for i in range(num_comps):
            GtWG, GtWd = _build_LS(ts, G, icomp=i)
            params[:, i] = sparse.linalg.lsmr(GtWG, GtWd, **lsmr_kw_args)[0].squeeze()
            if formal_variance:
                var[:, i] = np.diag(np.linalg.pinv(GtWG.toarray()))
    else:
        GtWG, GtWd = _build_LS(ts, G)
        params = sparse.linalg.lsmr(GtWG, GtWd, **lsmr_kw_args)[0].reshape(num_params, num_comps)
        if formal_variance:
            var = np.diag(np.linalg.pinv(GtWG.toarray())).reshape(num_params, num_comps)

    # separate parameters back to models
    model_params_var = _pack_params_var(models, params, var if formal_variance else None)
    return model_params_var


def ridge_regression(ts, models, penalty, formal_variance=False, lsmr_kw_args={}):
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
        Dictionary of :class:`~geonat.model.Model` instances used for fitting.
    penalty : float
        Penalty hyperparameter :math:`\lambda`.
    formal_variance : bool, optional
        If ``True``, also calculate the formal variance (diagonals of the covariance
        matrix).
    lsmr_kw_args : dict
        Additional keyword arguments passed on to SciPys's
        :func:`~scipy.sparse.linalg.lsmr` function.

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
    G, reg_diag, num_time, num_params, num_comps = _combine_mappings(ts, models,
                                                                     reg_indices=True)

    # perform fit and estimate formal covariance (uncertainty) of parameters
    # if there is no covariance, it's num_comps independent problems
    if ts.cov_cols is None:
        reg = sparse.diags(reg_diag, dtype=float) * penalty
        params = np.zeros((num_params, num_comps))
        if formal_variance:
            var = np.zeros((num_params, num_comps))
        for i in range(num_comps):
            GtWG, GtWd = _build_LS(ts, G, icomp=i)
            GtWGreg = GtWG + reg
            params[:, i] = sparse.linalg.lsmr(GtWGreg, GtWd, **lsmr_kw_args)[0].squeeze()
            if formal_variance:
                var[:, i] = np.diag(np.linalg.pinv(GtWGreg.toarray()))
    else:
        GtWG, GtWd = _build_LS(ts, G)
        reg = sparse.diags(np.repeat(reg_diag, num_comps), dtype=float) * penalty
        GtWGreg = GtWG + reg
        params = sparse.linalg.lsmr(GtWGreg, GtWd, **lsmr_kw_args)[0].reshape(num_params, num_comps)
        if formal_variance:
            var = np.diag(np.linalg.pinv(GtWGreg.toarray())).reshape(num_params, num_comps)

    # separate parameters back to models
    model_params_var = _pack_params_var(models, params, var if formal_variance else None)
    return model_params_var


def lasso_regression(ts, models, penalty, formal_variance=False, cvxpy_kw_args={}):
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
        Dictionary of :class:`~geonat.model.Model` instances used for fitting.
    penalty : float
        Penalty hyperparameter :math:`\lambda`.
    formal_variance : bool, optional
        If ``True``, also calculate the formal variance (diagonals of the covariance
        matrix).
    cvxpy_kw_args : dict
        Additional keyword arguments passed on to CVXPY's ``solve()`` function.

    Returns
    -------
    model_params_var : dict
        Dictionary of form ``{"model_description": (parameters, variance), ...}``
        which for every model that was fitted, contains a tuple of the best-fit
        parameters and the formal variance (or ``None``, if not calculated).
    """
    if penalty == 0.0:
        warn(f"Lasso Regression (L1-regularized) solver got a penalty of {penalty}, "
             "which effectively removes the regularization.")

    # get mapping and regularization matrix
    G, reg_diag, num_time, num_params, num_comps = _combine_mappings(ts, models,
                                                                     reg_indices=True)

    def objective_fn(X, Y, beta, reg_diag):
        return cp.norm2(X @ beta - Y) + penalty * cp.norm1(beta[reg_diag])

    # perform fit and estimate formal covariance (uncertainty) of parameters
    # if there is no covariance, it's num_comps independent problems
    if ts.cov_cols is None:
        beta = cp.Variable(num_params)
        params = np.zeros((num_params, num_comps))
        if formal_variance:
            var = np.zeros((num_params, num_comps))
        for i in range(num_comps):
            Gnonan, Wnonan, GtWG, GtWd = _build_LS(ts, G, icomp=i, return_W_G=True)

            # solve cvxpy problem
            problem = cp.Problem(cp.Minimize(objective_fn(GtWG, GtWd, beta, reg_diag)))
            try:
                problem.solve(**cvxpy_kw_args)
            except cp.error.SolverError as e:
                warn(f"CVXPY SolverError encountered: {str(e)}")
                converged = False
            else:
                converged = True

            # check for convergence
            if (not converged) or (beta.value is None):
                params[:, i] = np.NaN
                if formal_variance:
                    var[:, i] = np.NaN
            else:
                params[:, i] = beta.value
                if formal_variance:
                    best_ind = np.nonzero(beta.value)
                    Gsub = Gnonan[:, best_ind]
                    Wsub = Wnonan[:, best_ind]
                    GtWG = (Gsub.T @ Wsub @ Gsub).toarray()
                    var[best_ind, i] = np.diag(np.linalg.pinv(GtWG))
    else:
        Gnonan, Wnonan, GtWG, GtWd = _build_LS(ts, G, return_W_G=True)
        reg_diag = np.repeat(reg_diag, num_comps)
        beta = cp.Variable(num_params*num_comps)
        problem = cp.Problem(cp.Minimize(objective_fn(GtWG, GtWd, beta, reg_diag)))
        try:
            problem.solve(**cvxpy_kw_args)
        except cp.error.SolverError as e:
            warn(f"CVXPY SolverError encountered: {str(e)}")
            converged = False
        else:
            converged = True
        if (not converged) or (beta.value is None):  # couldn't converge
            params = np.empty((num_params, num_comps))
            params[:] = np.NaN
            if formal_variance:
                var = np.empty((num_params, num_comps))
                var[:] = np.NaN
        else:
            params = beta.value.reshape(num_params, num_comps)
            if formal_variance:
                var = np.zeros(num_params * num_comps)
                best_ind = np.nonzero(beta.value)
                Gsub = Gnonan[:, best_ind]
                Wsub = Wnonan[:, best_ind]
                GtWG = (Gsub.T @ Wsub @ Gsub).toarray()
                var[best_ind, :] = np.diag(np.linalg.pinv(GtWG))
                var = var.reshape(num_params, num_comps)

    # separate parameters back to models
    model_params_var = _pack_params_var(models, params, var if formal_variance else None)
    return model_params_var
