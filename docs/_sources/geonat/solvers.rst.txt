Solvers
=======

.. automodule:: geonat.solvers

Local Solver Functions
----------------------

lasso_regression
................

.. autofunction:: geonat.solvers.lasso_regression

linear_regression
..................

.. autofunction:: geonat.solvers.linear_regression

ridge_regression
................

.. autofunction:: geonat.solvers.ridge_regression

Global Solver Classes
---------------------

SpatialSolver
.............

.. autoclass:: geonat.solvers.SpatialSolver
   :members:
   :special-members:

Solution Object
---------------

.. autoclass:: geonat.solvers.Solution
   :members:

Reweighting Functions
---------------------

ReweightingFunction
...................

.. autoclass:: geonat.solvers.ReweightingFunction
   :members:
   :special-members: __call__

InverseReweighting
..................

.. autoclass:: geonat.solvers.InverseReweighting
   :members:
   :special-members: __call__

InverseSquaredReweighting
.........................

.. autoclass:: geonat.solvers.InverseSquaredReweighting
   :members:
   :special-members: __call__

LogarithmicReweighting
......................

.. autoclass:: geonat.solvers.LogarithmicReweighting
   :members:
   :special-members: __call__
