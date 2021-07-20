from numpy.distutils.core import setup, Extension

# specify that the Fortran code should be included and compiled
compiled_utils = Extension('disstans.compiled', sources=['disstans/compiled.f90'])

# run setup
setup(ext_modules=[compiled_utils])
