Solvers
=======

.. automodule:: disstans.solvers

Solution Object
---------------

.. autoclass:: disstans.solvers.Solution
   :members:

Solver Functions
----------------

Either used directly, or through :meth:`~disstans.network.Network.fit`.

lasso_regression
................

.. autofunction:: disstans.solvers.lasso_regression

linear_regression
..................

.. autofunction:: disstans.solvers.linear_regression

ridge_regression
................

.. autofunction:: disstans.solvers.ridge_regression

Reweighting Functions
---------------------

For use with :func:`~lasso_regression` and :meth:`~disstans.network.Network.spatialfit`.

ReweightingFunction
...................

.. autoclass:: disstans.solvers.ReweightingFunction
   :members:
   :special-members: __call__

InverseReweighting
..................

.. autoclass:: disstans.solvers.InverseReweighting
   :members:
   :special-members: __call__

InverseSquaredReweighting
.........................

.. autoclass:: disstans.solvers.InverseSquaredReweighting
   :members:
   :special-members: __call__

LogarithmicReweighting
......................

.. autoclass:: disstans.solvers.LogarithmicReweighting
   :members:
   :special-members: __call__
