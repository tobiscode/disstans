Frequently Asked Questions
==========================

1. How do I know which penalties to use in |subst|?
---------------------------------------------------

.. |subst| replace:: :func:`~disstans.solvers.lasso_regression` and
                     :class:`~disstans.solvers.ReweightingFunction`

This is problems-specific because it heavily depends on the noise in the data, as well as
what kinds of transient behaviors are expected. You can use the values from the
:doc:`Examples <examples>` to get started, but after that, you're going to have to
try out systematically different combinations of ``penalty``, ``eps``, ``scale``, and
maybe even the type of :class:`~disstans.solvers.ReweightingFunction`.
Things to look out for is a steadily decreasing count of unique, nonzero regularized
parameters while the root-mean-square misfit to the data does not increase significantly
compared to a L2- or L1-regularized solution.
If the misfit is too large, decrease the penalties.
If the number of nonzero regularized parameters is too large, increase the penalties.

Make sure that you don't use ``eps`` and ``scale`` combinations for the reweighting functions
that would not end up in a significant penalty applied to the parameter, e.g., by using an
:class:`~disstans.solvers.InverseReweighting` function with ``eps=1e-4`` and ``scale=1e-4``.

While finding the best combination of parameters when using the spatial L0 regularization,
it can help to reduce the number of maximum iterations (1 or 2 can already give a good idea
of which parameters will end up getting used), disable the use of data covariance (keep
the data variance, it can actually speed up the iterations), and disable the estimation of
the formal covariance matrix.

:doc:`Tutorial 5 <tutorials/tutorial_5>` shows an example of how to systematically look for
the best reweighting penalties given different noise levels in the data.

2. How can I save my :class:`~disstans.network.Network` object to save time?
----------------------------------------------------------------------------

Please refer to :ref:`Example 1 <examples/example_1:Model parameter estimation>` for an example
of how to save a Network object.

.. warning::

   While loading a Network object from a file givec access to all the timeseries, fits,
   model parameters, etc. from the original one, there can be problems when calculations are 
   continued on the Network object (e.g., by rerunning :meth:`~disstans.network.Network.fit`).
   One way to decrease the chances of that happening is to import all modules that were used
   in the code that saved the Network object *before* loading the file.
   If problems appear (e.g., unexpected fitting resutls), recreate the Network object from
   scratch, and don't load from a file.
