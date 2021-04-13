"""
This is the initialization script.

It also provides shorthand notations for the three most common
classes of GeoNAT, namely

- ``geonat.defaults`` for ``geonat.config.defaults``,
- ``geonat.Timeseries`` for ``geonat.timeseries.Timeseries``,
- ``geonat.Station`` for ``geonat.station.Station``, and
- ``geonat.Network`` for ``geonat.network.Network``.
"""
# flake8: noqa

import multiprocessing
from pandas.plotting import register_matplotlib_converters

# import submodules
from . import compiled
from . import config
from . import earthquakes
from . import models
from . import network
from . import processing
from . import solvers
from . import station
from . import timeseries
from . import tools

# import Scientific Colourmaps
from . import scm

# provide shortcuts for commonly used classes by importing them here
from .config import defaults
from .timeseries import Timeseries
from .station import Station
from .network import Network

# package version
__version__ = '0.5.1'

# preparational steps
multiprocessing.set_start_method('spawn', True)
register_matplotlib_converters()
