"""
This module contains the Scientific Colourmaps (v6) by Fabio Crameri
and provides them via ``geonat.scm.<name>``.
Their reversed colormaps can be accessed by ``geonat.scm.<name>_r``.

SCM6 is distributed under an MIT license which can be found in
the source code folder under ``geonat/smc/LICENSE``.
"""

from numpy import loadtxt
from pkg_resources import resource_filename
from matplotlib.colors import LinearSegmentedColormap

scm_names = ['acton', 'bamako', 'batlow', 'berlin', 'bilbao', 'broc', 'buda',
             'cork', 'davos', 'devon', 'grayC', 'hawaii', 'imola', 'lajolla',
             'lapaz', 'lisbon', 'nuuk', 'oleron', 'oslo', 'roma', 'tofino',
             'tokyo', 'turku', 'vik']

for scm_name in scm_names:
    cm_data = loadtxt(resource_filename('geonat', f'scm/{scm_name}.txt'))
    cmap = LinearSegmentedColormap.from_list(scm_name, cm_data)
    vars()[scm_name] = cmap
    vars()[scm_name + "_r"] = cmap.reversed()
