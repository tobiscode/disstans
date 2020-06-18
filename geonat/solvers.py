import numpy as np
import scipy.sparse as sparse
import cvxpy as cp
from warnings import warn


def linear_regression(ts, models, formal_covariance=False):
    """
    scipy.sparse.linalg wrapper for a linear least squares solver
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
    """
    scipy.sparse.linalg wrapper for a linear L2-regularized least squares solver
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
    """
    cvxpy wrapper for a linear L1-regularized least squares solver
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
    # reg = sparse.diags(reg_diag, dtype=float) * penalty
    reg_diag = np.array(reg_diag)
    num_time, num_params = G.shape
    num_components = len(ts.data_cols)

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
