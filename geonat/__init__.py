import multiprocessing
from pandas.plotting import register_matplotlib_converters

# package version
__version__ = '0.1.1'

# preparational steps
multiprocessing.set_start_method('spawn', True)
register_matplotlib_converters()

# set global defaults that can be overriden by the user
defaults = {}
# general
defaults["general"] = {"num_threads": None}
# GUI
defaults["gui"] = {"projection": "Mercator",
                   "coastlines_show": True,
                   "coastlines_res": "50m",
                   "wmts_show": False,
                   "wmts_server": "https://server.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml",
                   "wmts_layer": "World_Imagery",
                   "wmts_alpha": 0.2,
                   "plot_sigmas": 3,
                   "plot_sigmas_alpha": 0.5}
# cleaning timeseries
defaults["clean"] = {"std_thresh": 100,
                     "std_outlier": 5,
                     "min_obs": 100,
                     "min_clean_obs": 100}
# priors from earthquake catalogs
defaults["prior"] = {"alpha": 10,
                     "mu": 30,
                     "threshold": 3}
