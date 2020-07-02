Compiled
========

This module contains subroutines that are written in Fortran90 and compiled
during the installation of GeoNAT. They provide a significant speed boost
compared to pure-Python implementations.

maskedmedfilt2d
---------------

``maskedmedfilt2d`` is a masked running-median filter for 2D arrays.
The masking is a way to include NaNs in the array, and the running median
is computed column-wise with a specified kernel size.

* *array* is the 2D NumPy array
* *mask_in* is a NumPy array of the same size, which is 1 (or True)
  everywhere except where there are NaNs in the original array
  (where it should be 0 or False)
* *kernel* is the integer kernel window size

.. autofunction:: geonat.compiled.maskedmedfilt2d

Source Code
-----------

.. literalinclude:: ../../geonat/compiled.f90
   :language: Fortran
