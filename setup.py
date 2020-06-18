from numpy.distutils.core import setup, Extension
from geonat import __version__ as geonat_version

# import README
with open('README.md') as f:
    readme = f.read()

# specify that the Fortran code should be included and compiled
compiled_utils = Extension('geonat.compiled', sources=['geonat/compiled.f90'])

# the module needs the colormap files
# these are from the Scientific Colour Maps by Fabio Crameri, see license at geonat/scm/LICENSE
pkg_data = {'geonat': ['scm/*.txt', 'scm/LICENSE']}

# run setup
setup(name='geonat',
      version=geonat_version,
      description='Geodetic Network Analysis Tools',
      long_description=readme,
      author='Tobias Köhne',
      author_email='47008700+tobiscode@users.noreply.github.com',
      url='https://github.com/tobiscode/geonat',
      python_requires='>=3',
      license='GNU General Public License v3 (GPLv3)',
      packages=['geonat'],
      ext_modules=[compiled_utils],
      package_data=pkg_data,
      classifiers=['Intended Audience :: Science/Research',
                   'Topic :: Scientific/Engineering',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                   'Programming Language :: Python :: 3'])
