"""
This module contains solver routines for fitting models to the timeseries
of stations.
"""

import numpy as np
import scipy.sparse as sparse
import cvxpy as cp
from warnings import warn


def linear_regression(ts, models, formal_covariance=False):
    r"""
    Performs linear, unregularized least squares using :mod:`~scipy.sparse.linalg`.

    The timeseries are the observations :math:`\mathbf{d}`, and the models' mapping
    matrices are stacked together to form a single, sparse mapping matrix
    :math:`\mathbf{G}`. The solver then computes the model parameters
    :math:`\mathbf{m}` that minimize the cost function

    .. math:: f(\mathbf{m}) = \left\| \mathbf{Gm} - \mathbf{d} \right\|_2^2

    where :math:`\mathbf{\epsilon} = \mathbf{Gm} - \mathbf{d}` is the residual.

    The formal model covariance is defined as the pseudo-inverse

    .. math:: \mathbf{C}_m = \left( \mathbf{G}^T \mathbf{C}_d \mathbf{G} \right)^g

    where :math:`\mathbf{C}_d` is the timeseries' data covariance.

    Parameters
    ----------
    ts : geonat.timeseries.Timeseries
        Timeseries to fit.
    models : dict
        Dictionary of :class:`~geonat.model.Model` instances used for fitting.
    formal_covariance : bool, optional
        If ``True``, also calculate the formal covariance.
    """
    mapping_matrices = []
    # get mapping matrices
    for (mdl_description, model) in models.items():
        mapping_matrices.append(model.get_mapping(ts.time))
    G = sparse.hstack(mapping_matrices)
    num_time, num_params = G.shape
    num_components = len(ts.data_cols)
    # perform fit and estimate formal covariance (uncertainty) of parameters
    params = np.zeros((num_params, num_components))
    if formal_covariance:
        cov = np.zeros((num_params, num_params, num_components))
    for i in range(num_components):
        d = sparse.csc_matrix(ts.df[ts.data_cols[i]].values.reshape(-1, 1))
        if ts.sigma_cols[i] is None:
            GtWG = G.T @ G
            GtWd = G.T @ d
        else:
            GtW = G.T @ sparse.diags(1/ts.df[ts.sigma_cols[i]].values**2)
            GtWG = GtW @ G
            GtWd = GtW @ d
        params[:, i] = sparse.linalg.lsqr(GtWG, GtWd.toarray().squeeze())[0].squeeze()
        if formal_covariance:
            cov[:, :, i] = np.linalg.pinv(GtWG.toarray())
    # separate parameters back to models
    i = 0
    fitted_params = {}
    for (mdl_description, model) in models.items():
        fitted_params[mdl_description] = (params[i:i+model.num_parameters, :], cov[i:i+model.num_parameters, i:i+model.num_parameters, :] if formal_covariance else None)
        i += model.num_parameters
    return fitted_params


def ridge_regression(ts, models, penalty, formal_covariance=False):
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
    not designated to be regularized (see :attr:`~geonat.model.Model.regularize`).

    The formal model covariance is defined as the pseudo-inverse

    .. math:: \mathbf{C}_m = \left( \mathbf{G}^T \mathbf{C}_d \mathbf{G}
                                    + \lambda \mathbf{I}_\text{reg} \right)^g

    where :math:`\mathbf{C}_d` is the timeseries' data covariance and the subscript
    :math:`_\text{reg}` masks to zero the entries corresponding to non-regularized
    model parameters.

    Parameters
    ----------
    ts : geonat.timeseries.Timeseries
        Timeseries to fit.
    models : dict
        Dictionary of :class:`~geonat.model.Model` instances used for fitting.
    penalty : float
        Penalty hyperparameter :math:`\lambda`.
    formal_covariance : bool, optional
        If ``True``, also calculate the formal covariance.
    """
    if penalty == 0.0:
        warn(f"Ridge Regression (L2-regularized) solver got a penalty of {penalty}, which effectively removes the regularization.")
    mapping_matrices = []
    reg_diag = []
    # get mapping and regularization matrices
    for (mdl_description, model) in models.items():
        mapping_matrices.append(model.get_mapping(ts.time))
        reg_diag.extend([model.regularize for _ in range(model.num_parameters)])
    G = sparse.hstack(mapping_matrices, format='bsr')
    reg = sparse.diags(reg_diag, dtype=float) * penalty
    num_time, num_params = G.shape
    num_components = len(ts.data_cols)
    if np.sum(reg_diag) == 0:
        warn(f"Ridge Regression (L2-regularized) solver got no models to regularize.")
    # perform fit and estimate formal covariance (uncertainty) of parameters
    params = np.zeros((num_params, num_components))
    if formal_covariance:
        cov = np.zeros((num_params, num_params, num_components))
    for i in range(num_components):
        d = sparse.csc_matrix(ts.df[ts.data_cols[i]].values.reshape(-1, 1))
        if ts.sigma_cols[i] is None:
            GtWG = G.T @ G
            GtWd = G.T @ d
        else:
            GtW = G.T @ sparse.diags(1/ts.df[ts.sigma_cols[i]].values**2)
            GtWG = GtW @ G
            GtWd = GtW @ d
        GtWGreg = GtWG + reg
        params[:, i] = sparse.linalg.lsqr(GtWGreg, GtWd.toarray().squeeze())[0].squeeze()
        if formal_covariance:
            cov[:, :, i] = np.linalg.pinv(GtWGreg.toarray())
    # separate parameters back to models
    i = 0
    fitted_params = {}
    for (mdl_description, model) in models.items():
        fitted_params[mdl_description] = (params[i:i+model.num_parameters, :], cov[i:i+model.num_parameters, i:i+model.num_parameters, :] if formal_covariance else None)
        i += model.num_parameters
    return fitted_params


def lasso_regression(ts, models, penalty, formal_covariance=False):
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
    not designated to be regularized (see :attr:`~geonat.model.Model.regularize`).

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
    formal_covariance : bool, optional
        If ``True``, also calculate the formal covariance.
    """
    if penalty == 0.0:
        warn(f"Lasso Regression (L1-regularized) solver got a penalty of {penalty}, which effectively removes the regularization.")
    mapping_matrices = []
    reg_diag = []

    # get mapping and regularization matrices
    for (mdl_description, model) in models.items():
        mapping_matrices.append(model.get_mapping(ts.time))
        reg_diag.extend([model.regularize for _ in range(model.num_parameters)])
    G = sparse.hstack(mapping_matrices, format='bsr')
    reg_diag = np.array(reg_diag)
    num_time, num_params = G.shape
    num_components = len(ts.data_cols)
    if np.sum(reg_diag) == 0:
        warn(f"Lasso Regression (L1-regularized) solver got no models to regularize.")

    # define cvxpy helper expressions and functions
    beta = cp.Variable(num_params)
    lambd = cp.Parameter(nonneg=True)
    lambd.value = penalty

    def objective_fn(X, Y, beta, lambd, reg):
        return cp.norm2(X @ beta - Y)**2 + lambd * cp.norm1(beta * reg)

    # perform fit and estimate formal covariance (uncertainty) of parameters
    params = np.zeros((num_params, num_components))
    if formal_covariance:
        cov = np.zeros((num_params, num_params, num_components))
    for i in range(num_components):
        d = sparse.csc_matrix(ts.df[ts.data_cols[i]].values.reshape(-1, 1))
        if ts.sigma_cols[i] is None:
            GtWG = G.T @ G
            GtWd = G.T @ d
        else:
            GtW = G.T @ sparse.diags(1/ts.df[ts.sigma_cols[i]].values**2)
            GtWG = GtW @ G
            GtWd = GtW @ d

        # solve cvxpy problem
        problem = cp.Problem(cp.Minimize(objective_fn(GtWG, GtWd.toarray().squeeze(), beta, lambd, reg_diag)))
        problem.solve()
        params[:, i] = beta.value

        # estimate formal covariance
        if formal_covariance:
            best_ind = np.nonzero(beta.value)
            Gsub = G[:, best_ind]
            if ts.sigma_cols[i] is None:
                GtWG = Gsub.T @ Gsub
            else:
                GtW = Gsub.T @ sparse.diags(1/ts.df[ts.sigma_cols[i]].values**2)
                GtWG = GtW @ Gsub
            cov[best_ind, best_ind, i] = np.linalg.pinv(GtWG.toarray())

    # separate parameters back to models
    i = 0
    fitted_params = {}
    for (mdl_description, model) in models.items():
        fitted_params[mdl_description] = (params[i:i+model.num_parameters, :], cov[i:i+model.num_parameters, i:i+model.num_parameters, :] if formal_covariance else None)
        i += model.num_parameters
    return fitted_params
