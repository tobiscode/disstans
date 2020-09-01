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

# provide shortcuts for commonly used classes by importing them here
from .config import defaults
from .timeseries import Timeseries
from .station import Station
from .network import Network

# package version
__version__ = '0.4.1'

# preparational steps
multiprocessing.set_start_method('spawn', True)
register_matplotlib_converters()
