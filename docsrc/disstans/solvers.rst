Solvers
=======

.. automodule:: disstans.solvers

Local Solver Functions
----------------------

lasso_regression
................

.. autofunction:: disstans.solvers.lasso_regression

linear_regression
..................

.. autofunction:: disstans.solvers.linear_regression

ridge_regression
................

.. autofunction:: disstans.solvers.ridge_regression

Global Solver Classes
---------------------

SpatialSolver
.............

.. autoclass:: disstans.solvers.SpatialSolver
   :members:
   :special-members:

Solution Object
---------------

.. autoclass:: disstans.solvers.Solution
   :members:

Reweighting Functions
---------------------

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
