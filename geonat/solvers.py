"""
This module contains solver routines for fitting models to the timeseries
of stations.
"""

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import cvxpy as cp
from warnings import warn

from . import defaults


def _combine_mappings(ts, models, reg_indices=False, cached_mapping=None):
    """
    Quick helper function that concatenates the mapping matrices of the
    models given the timevector in ts, and returns the relevant sizes.
    If reg_indices = True, also return an array indicating which model
    is set to be regularized.
    It also makes sure that G only contains columns that contain at least
    one non-zero element, and correspond to parameters that are therefore
    observable.
    """
    mapping_matrices = []
    obs_indices = []
    if reg_indices:
        reg_diag = []
    for (mdl_description, model) in models.items():
        if cached_mapping and mdl_description in cached_mapping:
            mapping = cached_mapping[mdl_description].loc[ts.time].values
            observable = np.any(mapping != 0, axis=0)
            if reg_indices and model.regularize:
                nunique = (~np.isclose(np.diff(mapping, axis=0), 0)).sum(axis=0) + 1
                observable = np.logical_and(observable, nunique > 1)
            mapping = sparse.csc_matrix(mapping[:, observable])
        else:
            mapping, observable = model.get_mapping(ts.time, return_observability=True)
            mapping = mapping[:, observable]
        mapping_matrices.append(mapping)
        obs_indices.append(observable)
        if reg_indices:
            reg_diag.extend([model.regularize for _ in range(observable.sum())])
    G = sparse.hstack(mapping_matrices, format='csc')
    obs_indices = np.concatenate(obs_indices)
    num_time, num_params = G.shape
    assert num_params > 0, f"Mapping matrix is empty, has shape {G.shape}."
    num_comps = ts.num_components
    if reg_indices:
        reg_diag = np.array(reg_diag)
        num_reg = reg_diag.sum()
        if num_reg == 0:
            warn(f"Regularized solver got no models to regularize.")
        return G, obs_indices, num_time, num_params, num_comps, num_reg, reg_diag
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
            Wblocks = [np.linalg.inv(np.reshape(ts.var_cov.values[iobs, ts.var_cov_map],
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


def _pack_params_var(models, params, var, obs_indices):
    """
    Quick helper function that distributes the parameters (and variances)
    of the input matrices into the respective models.
    obs_indices indicates whether only certain columns (parameters) were estimated.
    """
    ix_model = 0
    ix_sol = 0
    num_components = params.shape[1]
    model_params_var = {}
    for (mdl_description, model) in models.items():
        mask = obs_indices[ix_model:ix_model+model.num_parameters]
        num_solved = mask.sum()
        p = np.zeros((model.num_parameters, num_components))
        p[mask, :] = params[ix_sol:ix_sol+num_solved, :]
        if var is None:
            v = None
        else:
            v = np.zeros((model.num_parameters, num_components))
            v[mask, :] = var[ix_sol:ix_sol+num_solved, :]
        model_params_var[mdl_description] = (p, v)
        ix_model += model.num_parameters
        ix_sol += num_solved
    return model_params_var


def _get_reweighting_function():
    """
    Collection of reweighting functions that can be used by lasso_regression.
    """
    name = defaults["solvers"]["reweight_func"]
    eps = defaults["solvers"]["reweight_eps"]
    if name == 'inv':
        def rw_func(x):
            return 1/(np.abs(x) + eps)
    elif name == 'invsq':
        def rw_func(x):
            return 1/(x**2 + eps**2)
    else:
        raise NotImplementedError(f"'{name}' is an unrecognized reweighting function.")
    return rw_func


def linear_regression(ts, models, formal_variance=False, cached_mapping=None,
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
    models : dict
        Dictionary of :class:`~geonat.models.Model` instances used for fitting.
    formal_variance : bool, optional
        If ``True``, also calculate the formal variance (diagonals of the covariance
        matrix).
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
            var = np.diag(np.linalg.pinv(GtWG)).reshape(num_params, num_comps)

    # separate parameters back to models
    model_params_var = _pack_params_var(models, params, var if formal_variance else None,
                                        obs_indices)
    return model_params_var


def ridge_regression(ts, models, penalty, formal_variance=False, cached_mapping=None,
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
    models : dict
        Dictionary of :class:`~geonat.models.Model` instances used for fitting.
    penalty : float
        Penalty hyperparameter :math:`\lambda`.
    formal_variance : bool, optional
        If ``True``, also calculate the formal variance (diagonals of the covariance
        matrix).
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
    G, obs_indices, num_time, num_params, num_comps, num_reg, reg_diag = \
        _combine_mappings(ts, models, reg_indices=True, cached_mapping=cached_mapping)

    # perform fit and estimate formal covariance (uncertainty) of parameters
    # if there is no covariance, it's num_comps independent problems
    if (ts.cov_cols is None) or (not use_data_covariance):
        reg = np.diag(reg_diag) * penalty
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
        reg = np.diag(np.repeat(reg_diag, num_comps)) * penalty
        GtWGreg = GtWG + reg
        params = sp.linalg.lstsq(GtWGreg, GtWd)[0].reshape(num_params, num_comps)
        if formal_variance:
            var = np.diag(np.linalg.pinv(GtWGreg)).reshape(num_params, num_comps)

    # separate parameters back to models
    model_params_var = _pack_params_var(models, params, var if formal_variance else None,
                                        obs_indices)
    return model_params_var


def lasso_regression(ts, models, penalty, reweight_max_iters=0, reweight_max_rss=1e-10,
                     formal_variance=False, cached_mapping=None, use_data_variance=True,
                     use_data_covariance=True, cvxpy_kw_args={}):
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
        If greater than zero, include additional solver calls after reweighting
        the regularization parameters (see Notes).
        Defaults to no reweighting (``0``).
    reweight_max_rss : float, optional
        When reweighting is active and the maximum number of iterations has not yet
        been reached, let the iteration stop early if the solutions do not change much
        anymore (see Notes).
        Defaults to ``1e-10``. Set to ``0`` to deactivate eraly stopping.
    formal_variance : bool, optional
        If ``True``, also calculate the formal variance (diagonals of the covariance
        matrix).
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
    cvxpy_kw_args : dict
        Additional keyword arguments passed on to CVXPY's ``solve()`` function.

    Returns
    -------
    model_params_var : dict
        Dictionary of form ``{"model_description": (parameters, variance), ...}``
        which for every model that was fitted, contains a tuple of the best-fit
        parameters and the formal variance (or ``None``, if not calculated).

    Notes
    -----

    The L0-regularization approximation used by setting ``reweight_max_iters > 0`` is based
    on [candes08]_. The idea here is to iteratively reduce the cost (before multiplication
    with :math:`\lambda`) of regularized, but significant parameters to 1, and iteratively
    increasing the cost of a regularized, but small parameter to a much larger value.

    This is achieved by introducing an additional parameter vector :math:`\mathbf{w}`
    of the same shape as the regularized parameters, inserting it into the L1 cost,
    and iterating between solving the L1-regularized problem, and using a reweighting
    function on those weights:

    1.  Initialize :math:`\mathbf{w}^{(0)} = \mathbf{1}`
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

    The reweighting function is set in the :attr:`~geonat.config.defaults` dictionary
    using the ``reweight_func`` key (along with a stabilizing parameter
    ``reweight_eps`` that should not need tuning). Possible values are:

    +--------------+-------------------------------------------------------+
    | ``'inv'``    | :math:`w(m) = \frac{1}{\|m\| + \text{eps}}` (default) |
    +--------------+-------------------------------------------------------+
    | ``'inv_sq'`` | :math:`w(m) = \frac{1}{m^2 + \text{eps}^2}`           |
    +--------------+-------------------------------------------------------+

    References
    ----------
    .. [candes08] Candès, E. J., Wakin, M. B., & Boyd, S. P. (2008).
       *Enhancing Sparsity by Reweighted ℓ1 Minimization.*
       Journal of Fourier Analysis and Applications, 14(5), 877–905.
       doi:`10.1007/s00041-008-9045-x <https://doi.org/10.1007/s00041-008-9045-x>`_.
    """
    if penalty == 0:
        warn(f"Lasso Regression (L1-regularized) solver got a penalty of {penalty}, "
             "which removes the regularization.")

    # get mapping and regularization matrix
    G, obs_indices, num_time, num_params, num_comps, num_reg, reg_diag = \
        _combine_mappings(ts, models, reg_indices=True, cached_mapping=cached_mapping)
    regularize = (num_reg > 0) and (penalty > 0)

    # solve CVXPY problem while checking for convergence
    def solve_problem(GtWG, GtWd, reg_diag):
        # build objective function
        m = cp.Variable(GtWd.size)
        objective = cp.norm2(GtWG @ m - GtWd)
        constraints = None
        if regularize:
            lambd = cp.Parameter(value=penalty, pos=True)
            if reweight_max_iters > 0:
                rw_func = _get_reweighting_function()
                weights = cp.Parameter(shape=num_reg*num_comps,
                                       value=np.ones(num_reg*num_comps), pos=True)
                z = cp.Variable(shape=num_reg*num_comps)
                objective = objective + lambd * cp.norm1(z)
                constraints = [z == cp.multiply(weights, m[reg_diag])]
                old_m = np.zeros(m.shape)
            else:
                objective = objective + lambd * cp.norm1(m[reg_diag])
        # define problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        # solve
        for i in range(reweight_max_iters + 1):  # always solve at least once
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
                if regularize and (i < reweight_max_iters):
                    # check if the solution changed to previous iteration
                    rss = np.sqrt(np.sum((old_m - m.value)**2))  # root sum of squares
                    if (i > 0) and (rss < reweight_max_rss):
                        break
                    # update weights
                    weights.value = rw_func(m.value[reg_diag])
                    old_m[:] = m.value[:]
        # return
        return m.value if converged else None

    # perform fit and estimate formal covariance (uncertainty) of parameters
    # if there is no covariance, it's num_comps independent problems
    if (ts.cov_cols is None) or (not use_data_covariance):
        # initialize output
        params = np.zeros((num_params, num_comps))
        if formal_variance:
            var = np.zeros((num_params, num_comps))
        # loop over components
        for i in range(num_comps):
            # build and solve problem
            Gnonan, Wnonan, GtWG, GtWd = _build_LS(ts, G, icomp=i, return_W_G=True,
                                                   use_data_var=use_data_variance)
            solution = solve_problem(GtWG, GtWd, reg_diag)
            # store results
            if solution is None:
                params[:, i] = np.NaN
                if formal_variance:
                    var[:, i] = np.NaN
            else:
                params[:, i] = solution
                # if desired, estimate formal variance here
                if formal_variance:
                    best_ind = np.nonzero(solution)
                    Gsub = Gnonan[:, best_ind]
                    GtWG = Gsub.T @ Wnonan @ Gsub
                    var[best_ind, i] = np.diag(np.linalg.pinv(GtWG))
    else:
        # build stacked problem and solve
        Gnonan, Wnonan, GtWG, GtWd = _build_LS(ts, G, return_W_G=True,
                                               use_data_var=use_data_variance,
                                               use_data_cov=use_data_covariance)
        reg_diag = np.repeat(reg_diag, num_comps)
        solution = solve_problem(GtWG, GtWd, reg_diag)
        # store results
        if solution is None:
            params = np.empty((num_params, num_comps))
            params[:] = np.NaN
            if formal_variance:
                var = np.empty((num_params, num_comps))
                var[:] = np.NaN
        else:
            params = solution.reshape(num_params, num_comps)
            # if desired, estimate formal variance here
            if formal_variance:
                var = np.zeros(num_params * num_comps)
                best_ind = np.nonzero(solution)
                Gsub = Gnonan.tocsc()[:, best_ind]
                GtWG = Gsub.T @ Wnonan @ Gsub
                var[best_ind, :] = np.diag(np.linalg.pinv(GtWG))
                var = var.reshape(num_params, num_comps)

    # separate parameters back to models
    model_params_var = _pack_params_var(models, params, var if formal_variance else None,
                                        obs_indices)
    return model_params_var
