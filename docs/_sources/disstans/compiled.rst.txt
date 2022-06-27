Compiled
========

This module contains subroutines that are written in Fortran90 and compiled
during the installation of DISSTANS. They provide a significant speed boost
compared to pure-Python implementations.

selectpair
----------

``selectpair`` is the key function to calculate the MIDAS velocity estimates,
described in detail in [blewitt16]_. It selects pairs of timestamps for a
one-year period (within a specified tolerance, and not crossing the specified
step times), but relaxes that assumption if no match can be found. Since there
is a lot of iteration, the (slightly modified for NumPy) Fortran code provided
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
