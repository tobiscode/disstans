from numpy.distutils.core import setup, Extension

# specify that the Fortran code should be included and compiled
compiled_utils = Extension('geonat.compiled', sources=['geonat/compiled.f90'])

# the module needs the colormap files
# these are from the Scientific Colour Maps by Fabio Crameri, see license at geonat/scm/LICENSE
pkg_data = {'geonat': ['scm/*.txt', 'scm/LICENSE']}

# run setup
setup(ext_modules=[compiled_utils],
      package_data=pkg_data)
