"""
This is the initialization script.

It also provides shorthand notations for the three most common
classes of GeoNAT, namely

- ``geonat.defaults`` for ``geonat.config.defaults``,
- ``geonat.Timeseries`` for ``geonat.timeseries.Timeseries``,
- ``geonat.Station`` for ``geonat.station.Station``, and
- ``geonat.Network`` for ``geonat.network.Network``.
"""

import multiprocessing
from pandas.plotting import register_matplotlib_converters

# provide shortcuts for commonly used classes by importing them here
from .config import defaults  # noqa: W0611
from .timeseries import Timeseries  # noqa: W0611
from .station import Station  # noqa: W0611
from .network import Network  # noqa: W0611

# package version
__version__ = '0.2'

# preparational steps
multiprocessing.set_start_method('spawn', True)
register_matplotlib_converters()
