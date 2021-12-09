Compiled
========

This module contains subroutines that are written in Fortran90 and compiled
during the installation of DISSTANS. They provide a significant speed boost
compared to pure-Python implementations.

maskedmedfilt2d
---------------

``maskedmedfilt2d`` is a masked running-median filter for 2D arrays.
The masking is a way to include NaNs in the array, and the running median
is computed column-wise with a specified kernel size.

* *array* is the 2D NumPy array,
* *mask_in* is a NumPy array of the same size, which is 1 (or True)
  everywhere except where there are NaNs in the original array
  (where it should be 0 or False),
* *kernel* is the (odd) integer kernel window size, and
* *medians* is the output running median, also a 2d NumPy array.

.. autofunction:: disstans.compiled.maskedmedfilt2d

selectpair
----------

``selectpair`` is the key function to calculate the MIDAS velocity estimates,
described in detail in [blewitt16]_. It selects pairs of timestamps for a
one-year period (within a specified tolerance, and not crossing the specified
step times), but relaxes that assumption if no match can be found. Since there
is a lot of iteration, the (slightly modifiedfor NumPy) Fortran code provided
by the author is included here for speed.

* *t* is the array of timestamps in decimal years,
* *tstep* is an array of step epochs in decimal years
  (with an additional, unused element necessarily added),
* *tolerance* specifies how exactly the one-year period should be matched
  when searching for pairs,
* *n* is the number of pairs found, and
* *ip* is two-row NumPy array, where the first *n* columns contain the
  indices of the pairs to use in the MIDAS calculation.

.. autofunction:: disstans.compiled.selectpair

Source Code
-----------

.. literalinclude:: ../../disstans/compiled.f90
   :language: Fortran
