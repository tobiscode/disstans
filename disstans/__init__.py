"""
This is the initialization script.

It also provides shorthand notations for the three most common
classes of DISSTANS, namely

- ``disstans.defaults`` for ``disstans.config.defaults``,
- ``disstans.Timeseries`` for ``disstans.timeseries.Timeseries``,
- ``disstans.Station`` for ``disstans.station.Station``, and
- ``disstans.Network`` for ``disstans.network.Network``.
"""
# flake8: noqa

# imports for preparations later
import multiprocessing
import importlib.metadata
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams

# import submodules
from . import config
from . import earthquakes
from . import models
from . import network
from . import processing
from . import solvers
from . import station
from . import timeseries
from . import tools

# provide shortcuts for commonly used classes by importing them here
from .config import defaults
from .timeseries import Timeseries
from .station import Station
from .network import Network

# package version
__version__ = importlib.metadata.version("disstans")

# preparational steps
multiprocessing.set_start_method('spawn', True)
register_matplotlib_converters()
rcParams['figure.constrained_layout.use'] = "True"
