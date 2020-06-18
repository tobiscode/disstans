# load Scientific Colourmaps by Fabio Crameri and provide them via geonat.scm.<name>

from numpy import loadtxt
from pkg_resources import resource_filename
from matplotlib.colors import LinearSegmentedColormap

__all__ = ["rainbow", "seismic", "topography"]

for cm_type, scm_name in zip(__all__, ["batlow", "roma", "oleron"]):
    cm_data = loadtxt(resource_filename('geonat', f'scm/{scm_name}.txt'))
    vars()[cm_type] = LinearSegmentedColormap.from_list(scm_name, cm_data)
