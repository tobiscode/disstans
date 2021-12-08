"""
This module contains the :class:`~disstans.network.Network` class, which is the
highest-level container object in DISSTANS.
"""

import numpy as np
import scipy as sp
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.geodesic as cgeod
from copy import deepcopy
from tqdm import tqdm
from warnings import warn
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from cartopy.io.ogc_clients import WMTSRasterSource
from cmcrameri import cm as scm

from . import timeseries as disstans_ts
from . import models as disstans_models
from . import solvers as disstans_solvers
from . import processing as disstans_processing
from .config import defaults
from .timeseries import Timeseries
from .station import Station
from .tools import parallelize, Timedelta, Click, weighted_median
from .earthquakes import okada_displacement
from .solvers import Solution, ReweightingFunction


class Network():
    r"""
    Main class of DISSTANS. Contains information about the network, defines defaults,
    and most importantly, contains a dictionary of all stations in the network.

    Parameters
    ----------
    name : str
        Name of the network.
    default_location_path : str, optional
        If station locations aren't given directly, check for a file with this path for the
        station's location.
        It needs to be an at-least four-column, space-separated text file with the entries
        ``name latitude[°] longitude[°] altitude[m]``. Existing headers have to be able to be
        identified as not being floats in the coordinate columns.
    auto_add : bool, optional
        If true, instatiation will automatically add all stations found in
        ``default_location_path``.
        If ``default_location_path`` is not set, this option is ignored.
    auto_add_filter : numpy.ndarray, optional
        If passed alongside ``default_location_path`` and ``auto_add``, network instantiation
        will only add stations within the latitude, longitude polygon defined by
        ``auto_add_filter`` (shape :math:`(\text{num_points}, \text{2})`).
    default_local_models : dict, optional
        Add a default selection of models for the stations.
    """
    def __init__(self, name, default_location_path=None, auto_add=False, auto_add_filter=None,
                 default_local_models={}):
        self.name = str(name)
        """ Network name. """
        self.default_location_path = str(default_location_path) \
            if default_location_path is not None else None
        """ Fallback path for station locations. """
        self.default_local_models = {}
        """
        Dictionary of default station timeseries models of structure
        ``{model_name: {"type": modelclass, "kw_args": {**kw_args}}}`` that contains
        the names, types and necessary keyword arguments to create each model object.

        These models can be added easily to all stations with :meth:`~add_default_local_models`,
        and will make the JSON export file cleaner.

        Using :meth:`~update_default_local_models` to update the dictionary will perform
        input checks.

        Example
        -------

        If ``net`` is a Network instance, the following adds an annual
        :class:`~disstans.models.Sinusoid` model::

            models = {"Annual": {"type": "Sinusoid",
                                 "kw_args": {"period": 365.25,
                                             "t_reference": "2000-01-01"}}}
            net.update_default_local_models(models)
        """
        self.update_default_local_models(default_local_models)
        self.stations = {}
        """
        Dictionary of network stations, where the keys are their string names
        and the values are their :class:`~disstans.station.Station` objects.
        """
        # try to preload the location data
        # it's a private attribute because there's no guarantee this will be kept
        # up to date over the lifetime of the Network instance (yet)
        if self.default_location_path is None:
            self._network_locations = []
        else:
            with open(self.default_location_path, mode='r') as locfile:
                loclines = [line.strip() for line in locfile.readlines()]
            # check for headers
            for i in range(len(loclines)):
                try:  # test by converting to lla
                    _ = [float(lla) for lla in loclines[i].split()[1:4]]
                except ValueError:
                    continue
                try:  # cut the header off
                    loclines = loclines[i:]
                    break
                except IndexError as e:  # didn't find any data
                    raise RuntimeError("Invalid text format for 'default_location_path'."
                                       ).with_traceback(e.__traceback__) from e
            # make {station: lla} dictionary
            self._network_locations = {line.split()[0]:
                                       [float(lla) for lla in line.split()[1:4]]
                                       for line in loclines}
            # check if the stations should also be added
            if auto_add:
                # check if there is a filter
                if auto_add_filter is not None:
                    assert (isinstance(auto_add_filter, np.ndarray) and
                            auto_add_filter.shape[1] == 2), \
                        "'auto_add_filter' needs to be a (num_points, 2)-shaped " \
                        f"NumPy array, got {auto_add_filter}."
                    net_poly = mpl.path.Path(auto_add_filter)
                    stat_dict = {stat: loc for stat, loc in self._network_locations.items()
                                 if net_poly.contains_point(loc[:2])}
                else:
                    stat_dict = self._network_locations
                # now add stations
                for stat in stat_dict:
                    # location will automatically come from self._network_locations
                    self.create_station(stat)

    @property
    def num_stations(self):
        """ Number of stations present in the network. """
        return len(self.stations)

    @property
    def station_names(self):
        """ Names of stations present in the network. """
        return list(self.stations.keys())

    def __str__(self):
        """
        Special function that returns a readable summary of the network.
        Accessed, for example, by Python's ``print()`` built-in function.

        Returns
        -------
        info : str
            Network summary.
        """
        info = f"Network {self.name}\n" + \
               f"Stations:\n{[key for key in self.stations]}\n"
        return info

    def __getitem__(self, name):
        """
        Convenience special function that provides a shorthand notation
        to access the network's stations.

        Parameters
        ----------
        name : str
            Name of the station.

        Returns
        -------
        disstans.station.Station
            Station in network.

        Example
        -------
        If ``net`` is a :class:`~Network` instance and ``name`` the name of a station,
        the following two are equivalent::

            net.stations[name]
            net[name]
        """
        if name not in self.stations:
            raise KeyError(f"No station '{name}' present.")
        return self.stations[name]

    def __setitem__(self, name, station):
        """
        Convenience special function that allows a dictionary-like adding of stations to
        the network by wrapping :meth:`~add_station`.

        See Also
        --------
        add_station : Add a station to the network instance.
        """
        self.add_station(name, station)

    def __delitem__(self, name):
        """
        Convenience special function that allows a dictionary-like removing of stations
        from the network by wrapping :meth:`~remove_station`.

        See Also
        --------
        remove_station : Remove a station from the network instance.
        """
        self.remove_station(name)

    def __contains__(self, name):
        """
        Special function that allows to check whether a certain station name
        is in the network.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, and we want to see whether
        ``'mystat`` is in the network, the following two are equivalent::

            # long version
            'mystat` in net.stations
            # short version
            'mystat` in net
        """
        return name in self.stations

    def __iter__(self):
        """
        Convenience special function that allows for a shorthand notation to quickly
        iterate over all stations.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, then the following
        two loops are equivalent::

            # long version
            for station in net.stations.values():
                pass
            # shorthand
            for station in net:
                pass
        """
        for station in self.stations.values():
            yield station

    def __len__(self):
        """
        Special function that gives quick access to the number of stations in the
        network (:attr:`~num_stations`) using Python's built-in ``len()`` function
        to make interactions with iterators easier.
        """
        return self.num_stations

    def export_network_ts(self, ts_description, subset_stations=None):
        """
        Collects a specific timeseries from all stations and returns them in a dictionary
        of network-wide :class:`~disstans.timeseries.Timeseries` objects.

        Parameters
        ----------
        ts_description : str, tuple
            :class:`~disstans.timeseries.Timeseries` description that will be collected
            from all stations in the network. If a tuple, specifies the timeseries
            and name of a fitted model for the timeseries at each station.
        subset_stations : list, optional
            If set, this is a list of station names to include in the output, all
            other stations will be ignored.

        Returns
        -------
        network_df : dict
            Dictionary with the data components of the ``ts_description`` timeseries
            as keys and :class:`~disstans.timeseries.Timeseries` objects as values
            (which will have in turn the station names as column names).

        See Also
        --------
        import_network_ts : Inverse function.
        """
        df_dict = {}
        if isinstance(ts_description, str):
            ts_is_timeseries = True
        elif isinstance(ts_description, tuple):
            assert all([isinstance(t, str) for t in ts_description]), \
                "When specifying a timeseries and model name tuple, each component " \
                f"must be a string (got {ts_description})."
            ts_model_ts, ts_model_name = ts_description
            ts_is_timeseries = False
        else:
            raise ValueError("Unrecognized input type for 'ts_description': "
                             f"{type(ts_description)}.")
        if subset_stations:
            stat_names = subset_stations
        else:
            stat_names = self.station_names
        for name in stat_names:
            station = self[name]
            if ts_is_timeseries and (ts_description in station.timeseries):
                ts = station[ts_description]
            elif ((not ts_is_timeseries) and (ts_model_ts in station.fits)
                  and (ts_model_name in station.fits[ts_model_ts])):
                ts = station.fits[ts_model_ts][ts_model_name]
            else:
                continue
            if df_dict == {}:
                network_data_cols = ts.data_cols
                network_src = ts.src
                network_unit = ts.data_unit
            df_dict.update({name: ts.data.astype(pd.SparseDtype())})
        if df_dict == {}:
            raise ValueError(f"No data found in network '{self.name}' "
                             f"for timeseries '{ts_description}'.")
        network_df = pd.concat(df_dict, axis=1)
        network_df.columns = network_df.columns.swaplevel(0, 1)
        network_df.sort_index(level=0, axis=1, inplace=True)
        network_df = {dcol: Timeseries(network_df[dcol], network_src, network_unit,
                                       [col for col in network_df[dcol].columns])
                      for dcol in network_data_cols}
        return network_df

    def import_network_ts(self, ts_description, dict_of_timeseries):
        """
        Distributes a dictionary of network-wide :class:`~disstans.timeseries.Timeseries`
        objects onto the network stations.

        Parameters
        ----------
        ts_description : str
            :class:`~disstans.timeseries.Timeseries` description where the data will be placed.
        dict_of_timeseries : dict
            Dictionary with the data components of the ``ts_description`` timeseries
            as keys and :class:`~disstans.timeseries.Timeseries` objects as values
            (which will have in turn the station names as column names).

        See Also
        --------
        export_network_ts : Inverse function.
        """
        network_df = pd.concat({name: ts.data for name, ts in dict_of_timeseries.items()},
                               axis=1)
        network_df.columns = network_df.columns.swaplevel(0, 1)
        network_df.sort_index(level=0, axis=1, inplace=True)
        data_cols = [dcol for dcol in dict_of_timeseries]
        src = dict_of_timeseries[data_cols[0]].src
        data_unit = dict_of_timeseries[data_cols[0]].data_unit
        for name in network_df.columns.levels[0]:
            self.stations[name].add_timeseries(ts_description,
                                               Timeseries(network_df[name].dropna(),
                                                          src, data_unit, data_cols))

    def add_station(self, name, station):
        """
        Add a station to the network.

        Parameters
        ----------
        name : str
            Name of the station.
        station : disstans.station.Station
            Station object to add.

        See Also
        --------
        __setitem__ : Shorthand notation wrapper.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, ``name`` the name of a new station,
        and ``station`` a :class:`~disstans.station.Station` instance, then the following
        two are equivalent::

            net.add_station(name, station)
            net[name] = station
        """
        if not isinstance(name, str):
            raise TypeError("Cannot add new station: 'name' is not a string.")
        if not isinstance(station, Station):
            raise TypeError("Cannot add new station: 'station' is not a Station object.")
        if name in self.stations:
            warn(f"Overwriting station '{name}'.", category=RuntimeWarning, stacklevel=2)
        self.stations[name] = station

    def create_station(self, name, location=None):
        """
        Create a station and add it to the network.

        Parameters
        ----------
        name : str
            Name of the station.
        location : tuple, list, numpy.ndarray, optional
            Location (Latitude [°], Longitude [°], Altitude [m]) of the station.
            If ``None``, the location information needs to be provided in
            :attr:`~default_location_path`.

        Raises
        ------
        ValueError
            If no location data is passed or found in :attr:`~default_location_path`.
        """
        if location is not None:
            assert (isinstance(location, list) and
                    all([isinstance(loc, float) for loc in location])), \
                "'location' needs to be a list of latitude, longitude and altitude floats, " \
                f"go {location}."
        elif name in self._network_locations:
            location = self._network_locations[name]
        else:
            raise ValueError(f"'location' was not passed to create_station, and {name} could "
                             "not be found in the default network locations, either.")
        self.stations[name] = Station(name, location)

    def remove_station(self, name):
        """
        Remove a station from the network.

        Parameters
        ----------
        name : str
            Name of the station.

        See Also
        --------
        __delitem__ : Shorthand notation wrapper.

        Example
        -------
        If ``net`` is a :class:`~Network` instance and ``name`` is the name of an existing
        station, then the following two are equivalent::

            net.remove_station(name)
            del net[name]
        """
        if name not in self.stations:
            warn(f"Cannot find station '{name}', couldn't delete.",
                 category=RuntimeWarning, stacklevel=2)
        else:
            del self.stations[name]

    @classmethod
    def from_json(cls, path, add_default_local_models=True, no_pbar=False,
                  station_kw_args={}, timeseries_kw_args={}):
        """
        Create a :class:`~disstans.network.Network` instance from a JSON configuration file.

        Parameters
        ----------
        path : str
            Path of input JSON file.
        add_default_local_models : bool, optional
            If false, skip the adding of any default local model found in a station.
        no_pbar : bool, optional
            Suppress the progress bar with ``True`` (default: ``False``).
        station_kw_args : dict, optional
            Additional keyword arguments passed on to the
            :class:`~disstans.station.Station` constructor.
        timeseries_kw_args : dict, optional
            Additional keyword arguments passed on to the
            :class:`~disstans.timeseries.Timeseries` constructor.

        Returns
        -------
        net : disstans.network.Network
            Network instance.

        See Also
        --------
        to_json : Export a network configuration to a JSON file.
        """
        # load configuration
        net_arch = json.load(open(path, mode='r'))
        network_name = net_arch.get("name", "network_from_json")
        network_locations_path = net_arch.get("locations")
        # create Network instance
        net = cls(name=network_name,
                  default_location_path=network_locations_path,
                  default_local_models=net_arch["default_local_models"])
        # create stations
        for station_name, station_cfg in tqdm(net_arch["stations"].items(), ascii=True,
                                              desc="Building Network", unit="station",
                                              disable=no_pbar):
            if "location" in station_cfg:
                station_loc = station_cfg["location"]
            elif station_name in net._network_locations:
                station_loc = net._network_locations[station_name]
            else:
                warn(f"Skipped station '{station_name}'  because location information "
                     "is missing.", stacklevel=2)
                continue
            station = Station(name=station_name, location=station_loc, **station_kw_args)
            # add timeseries to station
            for ts_description, ts_cfg in station_cfg["timeseries"].items():
                ts_class = getattr(disstans_ts, ts_cfg["type"])
                ts = ts_class(**ts_cfg["kw_args"], **timeseries_kw_args)
                station.add_timeseries(ts_description=ts_description, timeseries=ts)
                # add default local models to station
                if add_default_local_models:
                    station.add_local_model_kwargs(ts_description=ts_description,
                                                   model_kw_args=net.default_local_models)
            # add specific local models to station and timeseries
            for ts_description, ts_model_dict in station_cfg["models"].items():
                if ts_description in station.timeseries:
                    station.add_local_model_kwargs(ts_description=ts_description,
                                                   model_kw_args=ts_model_dict)
                else:
                    station.unused_models.update({ts_description: ts_model_dict})
            # add to network
            net.add_station(name=station_name, station=station)
        return net

    def to_json(self, path):
        """
        Export network configuration to a JSON file.

        Parameters
        ----------
        path : str
            Path of output JSON file.

        See Also
        --------
        from_json : Create a :class:`~disstans.network.Network` instance
            from a JSON configuration file.
        """
        # create new dictionary
        net_arch = {"name": self.name,
                    "locations": self.default_location_path,
                    "stations": {},
                    "default_local_models": self.default_local_models}
        # add station representations
        for stat_name, station in self.stations.items():
            stat_arch = station.get_arch()
            # need to remove all models that are actually default models
            for model_description, mdl in self.default_local_models.items():
                for ts_description in stat_arch["models"]:
                    if (model_description in stat_arch["models"][ts_description].keys()) \
                       and (mdl == stat_arch["models"][ts_description][model_description]):
                        del stat_arch["models"][ts_description][model_description]
            # now we can append it to the main json
            net_arch["stations"].update({stat_name: stat_arch})
        # write file
        json.dump(net_arch, open(path, mode='w'), indent=2, sort_keys=True)

    def update_default_local_models(self, models):
        """
        Perform input checks for the structure of ``models`` and if successful,
        update :attr:`~default_local_models`.

        Parameters
        ----------
        models : dict
            Dictionary of structure ``{model_name: {"type": modelclass, "kw_args":
            {**kw_args}}}`` that contains the names, types and necessary keyword arguments
            to create each model object (see :func:`~disstans.models.check_model_dict`).
        """
        disstans_models.check_model_dict(models)
        self.default_local_models.update(models)

    def add_default_local_models(self, ts_description, models=None):
        """
        Add the network's default local models (or a subset thereof) to all stations.

        For example, this method can be used when instantiating a new network object
        from a JSON file using :meth:`~from_json` but skipping the adding of local
        models at that stage.

        Parameters
        ----------
        ts_description : str
            Timeseries description to add the models to.
        models : list, optional
            List of strings containing the model names of the subset of the default
            local models to add. Defaults to all local models.

        See Also
        --------
        :attr:`~default_local_models` : The network's default list of models.
        """
        assert isinstance(ts_description, str), \
            f"'ts_description' must be a string, got {type(ts_description)}."
        if models is None:
            local_models_subset = self.default_local_models
        else:
            if not isinstance(models, str) and not \
               (isinstance(models, list) and all([isinstance(m, str) for m in models])):
                raise ValueError("'models' must be None, a string or a list of strings, "
                                 f"got {models}.")
            if isinstance(models, str):
                models = [models]
            local_models_subset = {name: model for name, model
                                   in self.default_local_models.items() if name in models}
        for station in self:
            for model_description, model_cfg in local_models_subset.items():
                local_copy = deepcopy(model_cfg)
                mdl = getattr(disstans_models, local_copy["type"])(**local_copy["kw_args"])
                station.add_local_model(ts_description=ts_description,
                                        model_description=model_description,
                                        model=mdl)

    def add_unused_local_models(self, target_ts, hidden_ts=None, models=None):
        """
        Add a station's unused models (or subset thereof) to a target timeseries.

        Unused models are models that have been defined for a specific timeseries at the
        import of a JSON file, but there is no associated information for that timeseries
        itself. Usually, this happens when after the network is loaded, the target
        timeseries still needs to be added.

        Parameters
        ----------
        target_ts : str
            Timeseries description to add the models to.
        hidden_ts : str, optional
            Description of the timeseries that contains the unused model.
            Defaults to ``target_ts``.
        models : list, optional
            List of strings containing the model names of the subset of the default
            local models to add. Defaults to all hidden models.
        """
        assert isinstance(target_ts, str), \
            f"'target_ts' must be string, got {type(target_ts)}."
        if hidden_ts is None:
            hidden_ts = target_ts
        else:
            assert isinstance(hidden_ts, str), \
                f"'hidden_ts' must be None or a string, got {type(hidden_ts)}."
        assert ((models is None) or isinstance(models, str) or
                (isinstance(models, list) and all([isinstance(m, str) for m in models]))), \
            f"'models' must be None, a string or a list of strings, got {models}."
        if isinstance(models, str):
            models = [models]
        for station in self:
            if hidden_ts in station.unused_models:
                if models is None:
                    local_models_subset = station.unused_models[hidden_ts]
                else:
                    local_models_subset = {name: model for name, model
                                           in station.unused_models[hidden_ts].items()
                                           if name in models}
                station.add_local_model_kwargs(ts_description=target_ts,
                                               model_kw_args=local_models_subset)

    def add_local_models(self, models, ts_description, station_subset=None):
        """
        For each station in the network (or a subset thereof), add models
        to a timeseries using dictionary keywords.

        Parameters
        ----------
        models : dict
            Dictionary of structure ``{model_name: {"type": modelclass, "kw_args":
            {**kw_args}}}`` that contains the names, types and necessary keyword arguments
            to create each model object (see :func:`~disstans.models.check_model_dict`).
        ts_description : str
            Timeseries to add the models to.
        station_subset : list, optional
            If provided, only add the models to the stations in this list.
        """
        # check input
        assert isinstance(ts_description, str), \
            f"'ts_description' must be a string, got {type(ts_description)}."
        if isinstance(station_subset, list):
            station_list = [self[stat_name] for stat_name in station_subset]
            assert len(station_list) > 0, "No valid stations in 'station_subset'."
        else:
            station_list = self.stations.values()
        disstans_models.check_model_dict(models)
        # loop over stations
        for station in station_list:
            if ts_description not in station.timeseries:
                continue
            for model_description, model_cfg in models.items():
                local_copy = deepcopy(model_cfg)
                mdl = getattr(disstans_models, local_copy["type"])(**local_copy["kw_args"])
                station.add_local_model(ts_description=ts_description,
                                        model_description=model_description,
                                        model=mdl)

    def remove_timeseries(self, *ts_to_remove):
        """
        Convenience function that scans all stations for timeseries of given names
        and removes them (together with models and fits).

        Parameters
        ----------
        *ts_to_remove : list
            Pass all timeseries to remove as function arguments.

        See Also
        --------
        disstans.station.Station.remove_timeseries : Station-specific equivalent
        """
        for station in self:
            for ts_description in ts_to_remove:
                if ts_description in station.timeseries:
                    station.remove_timeseries(ts_description)

    def copy_uncertainties(self, origin_ts, target_ts):
        """
        Convenience function that copies the uncertainties of one timeseries
        to another for all stations.

        Parameters
        ----------
        origin_ts : str
            Name of the timeseries that contains the uncertainty data.
        target_ts : str
            Name of the timeseries that should receive the uncertainty data.
        """
        for station in self:
            if (origin_ts in station.timeseries.keys()) and \
               (target_ts in station.timeseries.keys()):
                station[target_ts].add_uncertainties(timeseries=station[origin_ts])

    def load_maintenance_dict(self, maint_dict, ts_description, model_description,
                              only_active=True):
        """
        Convenience wrapper to add :class:`~disstans.models.Step` models to the stations
        in the network where they experienced maitenance and therefore likely a jump
        in station coordinates.

        Parameters
        ----------
        maint_dict : dict
            Dictionary of structure ``{station_name: [steptimes]}`` where ``station_name``
            is the station name as present in the Network object, and ``steptimes`` is a list
            of either datetime-like strings or :class:`~pandas.Timestamp`.
        ts_description : str
            Timeseries to add the steps to.
        model_description : str
            Name for the step models.
        only_active : bool, optional
            If ``True`` (default), will check for the active time period of the timeseries,
            and only add steps that fall inside it.

        See Also
        --------
        :class:`~disstans.models.Step` : Model used to add steps.
        """
        assert all([isinstance(station, str) for station in maint_dict.keys()]), \
            f"Keys in 'maint_dict' must be strings, got {maint_dict.keys()}."
        assert all([isinstance(steptimes, list) for steptimes in maint_dict.values()]), \
            f"Values in 'maint_dict' must be lists, got {maint_dict.values()}."
        for station, steptimes in maint_dict.items():
            if station in self.stations:
                if only_active:
                    tmin = self[station][ts_description].time.min()
                    tmax = self[station][ts_description].time.max()
                    localsteps = [st for st in steptimes if
                                  ((pd.Timestamp(st) >= tmin) and (pd.Timestamp(st) <= tmax))]
                else:
                    localsteps = steptimes
                self[station].add_local_model(ts_description, model_description,
                                              disstans_models.Step(localsteps))

    def freeze(self, ts_description, model_list=None, zero_threshold=1e-10):
        """
        Convenience method that calls :meth:`~disstans.models.Model.freeze`
        for the :class:`~disstans.models.ModelCollection` for a certain
        timeseries at every station.

        Parameters
        ----------
        ts_description : str
            Description of the timeseries to to freeze the models for.
        model_list : list, optional
            If ``None`` (default), freeze all models. If a list of strings, only
            freeze the corresponding models in the collection.
        zero_threshold : float, optional
            Model parameters with absolute values below ``zero_threshold`` will be
            set to zero and set inactive. Defaults to ``1e-10``.

        See Also
        --------
        unfreeze : The reverse network method.
        """
        for station in self:
            if ts_description in station:
                station.models[ts_description].freeze(model_list=model_list,
                                                      zero_threshold=zero_threshold)

    def unfreeze(self, ts_description, model_list=None):
        """
        Convenience method that resets any frozen parameters for a certain timeseries
        at every station.

        Parameters
        ----------
        ts_description : str
            Description of the timeseries to to unfreeze the models for.
        model_list : list, optional
            If ``None`` (default), freeze all models. If a list of strings, only
            freeze the corresponding models in the collection.

        See Also
        --------
        freeze : The reverse network method.
        """
        for station in self:
            if ts_description in station:
                station.models[ts_description].unfreeze(model_list=model_list)

    def fit(self, ts_description, solver='linear_regression', local_input={},
            return_solutions=False, progress_desc=None, no_pbar=False, **kw_args):
        """
        Fit the models for a specific timeseries at all stations,
        and read the fitted parameters into the station's model collection.
        Also provides a progress bar.
        Will automatically use multiprocessing if parallelization has been enabled in
        the configuration (defaults to parallelization if possible).

        Parameters
        ----------
        ts_description : str
            Description of the timeseries to fit.
        solver : str, function, optional
            Solver function to use. If given a string, will look for a solver with that
            name in :mod:`~disstans.solvers`, otherwise will use the passed function as a
            solver (which needs to adhere to the same input/output structure as the
            included solver functions). Defaults to standard linear least squares.
        local_input : dict, optional
            Provides the ability to pass individual keyword arguments to the solver,
            potentially overriding the (global) keywords in ``kw_args``.
        return_solutions : bool, optional
            If ``True`` (default: ``False``), return a dictionary of all solutions produced
            by the calls to the solver function.
        progress_desc : str, optional
            If provided, override the description of the progress bar.
        no_pbar : bool, optional
            Suppress the progress bar with ``True`` (default: ``False``).
        **kw_args : dict
            Additional keyword arguments that are passed on to the solver function.

        Returns
        -------
        solutions : dict, optional
            If ``return_solutions=True``, a dictionary that contains the
            :class:`~disstans.solvers.Solution` objects for each station.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, ``'mydata'`` is the timeseries to fit,
        and ``mysolver`` is the solver to use, then the following two are equivalent::

            # long version, not parallelized
            for station in net:
                station_ts = station.timeseries['mydata']
                station_models = station.models['mydata']
                sol = mysolver(station_ts, station_models, **kw_args)
                station_models.read_parameters(sol.parameters_zeroed, sol.covariances_zeroed)
            # short version, automatically parallelizes according to disstans.defaults,
            # also allows skipping the import of disstans.solvers
            net.fit('mydata', solver='lasso_regression', **kw_args)

        See Also
        --------
        evaluate : Evaluate the fitted models at all stations.
        :attr:`~disstans.config.defaults` : Dictionary of settings, including parallelization.
        disstans.tools.parallelize : Automatically execute a function in parallel or serial.
        """
        assert isinstance(ts_description, str), \
            f"'ts_description' must be string, got {type(ts_description)}."
        assert any([ts_description in station.timeseries.keys() for station in self]), \
            f"Could not find any station with a timeseries called '{ts_description}'."
        if isinstance(solver, str):
            solver = getattr(disstans_solvers, solver)
        assert callable(solver), f"'solver' must be a callable function, got {type(solver)}."
        progress_desc = str(progress_desc) if progress_desc else "Fitting station models"
        station_names = [name for name, station in self.stations.items()
                         if ts_description in station.timeseries]
        iterable_inputs = ((solver,
                            self[stat_name].timeseries[ts_description],
                            self[stat_name].models[ts_description],
                            kw_args if local_input == {}
                            else {**kw_args, **local_input[stat_name]})
                           for stat_name in station_names)
        if return_solutions:
            solutions = {}
        for i, sol in enumerate(tqdm(parallelize(self._fit_single_station, iterable_inputs),
                                     desc=progress_desc, total=len(station_names),
                                     ascii=True, unit="station", disable=no_pbar)):
            # print warning if the solver didn't converge
            if not sol.converged:
                warn(f"Fitting did not converge for timeseries {ts_description} "
                     f"at {station_names[i]}", category=RuntimeWarning, stacklevel=2)
            self[station_names[i]].models[ts_description] \
                .read_parameters(sol.parameters_zeroed, sol.covariances_zeroed)
            # if raw output is requested, save it
            if return_solutions:
                solutions[station_names[i]] = sol
        if return_solutions:
            return solutions

    @staticmethod
    def _fit_single_station(parameter_tuple):
        solver, station_time, station_models, kw_args = parameter_tuple
        return solver(station_time, station_models, **kw_args)

    def spatialfit(self, ts_description, penalty, spatial_reweight_models,
                   spatial_reweight_iters, spatial_reweight_percentile=0.5,
                   spatial_reweight_max_rms=1e-9, spatial_reweight_max_changed=0,
                   dist_num_avg=4, dist_weight_min=None, dist_weight_max=None,
                   continuous_reweight_models=[], local_reweight_iters=1,
                   local_reweight_func=None, local_reweight_coupled=True,
                   formal_covariance=False, use_data_variance=True,
                   use_data_covariance=True, use_internal_scales=True,
                   cov_zero_threshold=1e-6, verbose=False, no_pbar=False,
                   return_stats=True, extended_stats=False,
                   keep_mdl_res_as=None, return_solutions=False,
                   zero_threshold=1e-4, num_threads_evaluate=None, roll_mean_kernel=30,
                   cvxpy_kw_args={"solver": "CVXOPT", "kktsolver": "robust"}):
        r"""
        Fit the models for a specific timeseries at all stations using the
        spatiotemporal capabilities of :func:`~disstans.solvers.lasso_regression`,
        and read the fitted parameters into the station's model collection.
        Also provides a progress bar.
        Will automatically use multiprocessing if parallelization has been enabled in
        the configuration (defaults to parallelization if possible).

        The function solve the network-wide fitting problem as proposed by [riel14]_:

            1.  Fit the models individually using a single iteration step from
                :func:`~disstans.solvers.lasso_regression`.
            2.  Collect the L0 weights :math:`\mathbf{w}^{(i)}_j` after iteration
                :math:`i` from each station :math:`j`.
            3.  Spatially combine the weights, and redistribute them to the stations
                for the next iteration.
            4.  Repeat from 1.

        The iteration can stop early if either the conditions set by
        ``spatial_reweight_max_rms`` *or* ``spatial_reweight_max_changed`` are satisfied,
        for all models in ``spatial_reweight_models``.

        In the third step, a distance-weighted median is used at each station to combine
        the L0 regularization weights :math:`\mathbf{w}^{(i)}_j`. The distance weights
        :math:`v_{j,k}` from station :math:`j` to station :math:`k` are based on the
        average distance :math:`D_j` from the station in question to the
        ``dist_num_avg`` clostest stations, following an exponential curve:
        :math:`v_{j,k}=\exp \left( - r_{j,k} / D_j \right)` where
        :math:`r_{j,k}` is the distance between the stations :math:`j` and :math:`k`.
        ``dist_weight_min`` and ``dist_weight_max`` allow to set boundaries for
        :math:`D_j`.

        Parameters
        ----------
        ts_description : str
            Description of the timeseries to fit.
        penalty : float, list, numpy.ndarray
            Penalty hyperparameter :math:`\lambda`. For non-reweighted models (i.e.,
            regularized models that are neither in ``spatial_reweight_models`` nor in
            ``continuous_reweight_models``), this is the constant penalty applied for
            every iteration. For the reweighted models, and if ``local_reweight_coupled=True``
            (default), this is just the penalty at the first iteration. After that,
            the penalties are largely controlled by ``local_reweight_func``. If
            ``local_reweight_coupled=False``, the penalty is applied on top of the updated
            weights at each iteration.
            ``penalty`` can either be a single value used for all components, or a list
            or NumPy array specifying a penalty for each component in the data.
        spatial_reweight_models : list
            Names of models to use in the spatial reweighting.
        spatial_reweight_iters : int
            Number of spatial reweighting iterations.
        spatial_reweight_percentile : float, optional
            Percentile used in the spatial reweighting.
            Defaults to ``0.5``.
        spatial_reweight_max_rms : float, optional
            Stop the spatial iterations early if the difference in the RMS (Root Mean Square)
            of the change of the parameters between reweighting iterations is less than
            ``spatial_reweight_max_rms``.
        spatial_reweight_max_changed : float, optional
            Stop the spatial iterations early if the number of changed parameters (i.e.,
            flipped between zero and non-zero) falls below a threshold. The threshold
            ``spatial_reweight_max_changed`` is given as the percentage of changed over total
            parameters (including all models and components). Defaults to no early stopping.
        dist_num_avg : int, optional
            Calculate the characteristic distance for the drop-off of station weights
            as the average of the ``dist_num_avg`` closest stations (default: ``4``).
        dist_weight_min : float, optional
            Enforce a minimum value of :math:`D_j` (in kilometers).
        dist_weight_max : float, optional
            Enforce a maximum value of :math:`D_j` (in kilometers).
        continuous_reweight_models : list
            Names of models that should carry over their weights from one solver iteration
            to the next, but should not be reweighted.
        local_reweight_iters : int, optional
            Number of local reweighting iterations, see ``reweight_max_iters`` in
            :func:`~disstans.solvers.lasso_regression`.
        local_reweight_func : ReweightingFunction, optional
            An instance of a reweighting function that will be used by
            :func:`~disstans.solvers.lasso_regression`.
            Defaults to an inverse reweighting with stability parameter ``eps=1e-4``.
        local_reweight_coupled : bool, optional
            If ``True`` (default) and reweighting is active, the L1 penalty hyperparameter
            is coupled with the reweighting weights (see Notes in
            :func:`~disstans.solvers.lasso_regression`).
        formal_covariance : bool, optional
            If ``True``, calculate the formal model covariance. Defaults to ``False``.
        use_data_variance : bool, optional
            If ``True`` (default) and ``ts_description`` contains variance information, this
            uncertainty information will be used.
        use_data_covariance : bool, optional
            If ``True`` (default), ``ts_description`` contains variance and covariance
            information, and ``use_data_variance`` is also ``True``, this uncertainty
            information will be used.
        use_internal_scales : bool, optional
            Sets whether internal scaling should be used when reweighting, see
            ``use_internal_scales`` in :func:`~disstans.solvers.lasso_regression`.
        verbose : bool, optional
            If ``True`` (default: ``False``), print statistics along the way.
        no_pbar : bool, optional
            Suppress the progress bars with ``True`` (default: ``False``).
        extended_stats : bool, optional
            If ``True`` (default: ``False``), the fitted models are evaluated at each iteration
            to calculate residual and fit statistics. These extended statistics are added to
            ``statistics`` (see Returns below).
        keep_mdl_res_as : tuple, optional
            If ``extended_stats=True``, the network's models are evaluated, and a model fit and
            residuals timeseries are created. Between iterations, they are removed by default,
            but passing this parameter a 2-element tuple of strings keeps the evaluated model
            and residual timeseries after the last iteration. This effectively allows the user
            to skip :meth:`~disstans.network.Network.evaluate` (for ``output_description``)
            and the calculation of the residual as done with ``residual_description`` by
            :meth:`~disstans.network.Network.fitevalres` after finishing the spatial fitting.
        zero_threshold : float, optional
            When extracting the formal covariance matrix or calculating statistics, assume
            parameters with absolute values smaller than ``zero_threshold`` are effectively zero.
        num_threads_evaluate : int, optional
            If ``extended_stats=True`` and ``formal_covariance=True``, there will be calls to
            :meth:`~evaluate` that will estimate the predicted variance,
            which will be memory-intensive if the timeseries are long. Using the same number
            of threads as defined in :attr:`~disstans.config.defaults` might therefore exceed
            the available memory on the system. This option allows to set a different number
            of threads or disable parallelized processing entirely for those calls.
            Defaults to ``None``, which uses the same setting as in the defaults.
        roll_mean_kernel : int, optional
            Only used if ``extended_stats=True``. This is the kernel size that gets used in the
            analysis of the residuals between each fitting step.
        cvxpy_kw_args : dict
            Additional keyword arguments passed on to CVXPY's ``solve()`` function,
            see ``cvxpy_kw_args`` in :func:`~disstans.solvers.lasso_regression`.

        Returns
        -------

        statistics : dict
            See the Notes for an explanation of the solution and convergence statistics
            that are returned.
        solutions : dict, optional
            If ``return_solutions=True``, a dictionary that contains the
            :class:`~disstans.solvers.Solution` objects for each station.

        Notes
        -----

        The ``statistics`` dictionary contains the following entries:

        - ``'num_total'`` (:class:`~int`):
          Total number of parameters that were reweighted.
        - ``'arr_uniques'`` (:class:`~numpy.ndarray`):
          Array of shape :math:`(\text{spatial_reweight_iters}+1, \text{num_components})`
          of the number of unique (i.e., over all stations) parameters that are non-zero
          for each iteration.
        - ``'list_nonzeros'`` (:class:`~list`):
          List of the total number of non-zero parameters for each iteration.
        - ``'dict_rms_diff'`` (:class:`~dict`):
          Dictionary that for each reweighted model and contains a list (of length
          ``spatial_reweight_iters``) of the RMS differences of the reweighted parameter
          values between spatial iterations.
        - ``'dict_num_changed'`` (:class:`~dict`):
          Dictionary that for each reweighted model and contains a list (of length
          ``spatial_reweight_iters``) of the number of reweighted parameters that changed
          from zero to non-zero or vice-versa.
        - ``'list_res_stats'`` (:class:`~list`):
          (Only present if ``extended_stats=True``.) List of the results dataframe returned
          by :meth:`~analyze_residuals` for each iteration.
        - ``'dict_cors'`` (:class:`~dict`):
          (Only present if ``extended_stats=True``.)
          For each of the reweighting models, contains a list of spatial correlation
          matrices for each iteration and component. E.g., the correlation matrix
          for model ``'my_model'`` after ``5`` reweighting iterations (i.e. the sixth
          solution, taking into account the initial unweighted solution) for the first
          component can be found in ``statistics['dict_cors']['my_model'][5][0]``
          and has a shape of :math:`(\text{num_stations}, \text{num_stations})`.
        - ``'dict_cors_means'`` (:class:`~dict`):
          (Only present if ``extended_stats=True``.) Same shape as ``'dict_cors'``,
          but containing the average of the upper triagonal parts of the spatial
          correlation matrices (i.e. for each model, iteration, and component).

        References
        ----------

        .. [riel14] Riel, B., Simons, M., Agram, P., & Zhan, Z. (2014). *Detecting transient
           signals in geodetic time series using sparse estimation techniques*.
           Journal of Geophysical Research: Solid Earth, 119(6), 5140–5160.
           doi:`10.1002/2014JB011077 <https://doi.org/10.1002/2014JB011077>`_
        """

        # input tests
        valid_stations = {name: station for name, station in self.stations.items()
                          if ts_description in station.timeseries}
        station_names = list(valid_stations.keys())
        num_stations = len(station_names)
        assert num_stations > 1, "The number of stations in the network that " \
            "contain the timeseries must be more than one."
        assert isinstance(spatial_reweight_models, list) and \
            all([isinstance(mdl, str) for mdl in spatial_reweight_models]), \
            "'spatial_reweight_models' must be a list of model name strings, got " + \
            f"{spatial_reweight_models}."
        assert isinstance(spatial_reweight_iters, int) and (spatial_reweight_iters > 0), \
            "'spatial_reweight_iters' must be an integer greater than 0, got " + \
            f"{spatial_reweight_iters}."
        assert float(spatial_reweight_max_rms) >= 0, "'spatial_reweight_max_rms' needs " \
            f"to be greater or equal to 0, got {spatial_reweight_max_rms}."
        assert 0 <= float(spatial_reweight_max_changed) <= 1, "'spatial_reweight_max_changed' " \
            f"needs to be between 0 and 1, got {spatial_reweight_max_changed}."
        if continuous_reweight_models != []:
            assert isinstance(continuous_reweight_models, list) and \
                all([isinstance(mdl, str) for mdl in continuous_reweight_models]), \
                "'continuous_reweight_models' must be a list of model name strings, got " + \
                f"{continuous_reweight_models}."
        all_reweight_models = set(spatial_reweight_models + continuous_reweight_models)
        assert len(all_reweight_models) == len(spatial_reweight_models) + \
            len(continuous_reweight_models), "'spatial_reweight_models' " + \
            "and 'continuous_reweight_models' can not have shared elements"

        # set up reweighting function
        if local_reweight_func is None:
            rw_func = ReweightingFunction.from_name("inv", 1e-4)
        else:
            assert isinstance(local_reweight_func, ReweightingFunction), "'local_reweight_func' " \
                f"needs to be None or a ReweightingFunction, got {type(local_reweight_func)}."
            rw_func = local_reweight_func

        # get scale lengths (correlation lengths) using the average distance to the
        # closest dist_num_avg stations and optional boundaries
        if verbose:
            tqdm.write("Calculating scale lengths")
        geoid = cgeod.Geodesic()
        dist_weight_min = 0 if dist_weight_min is None else dist_weight_min * 1e3
        dist_weight_max = None if dist_weight_max is None else dist_weight_max * 1e3
        station_lonlat = np.stack([np.array(self[name].location)[[1, 0]]
                                   for name in station_names])
        all_distances = np.empty((num_stations, num_stations))
        net_avg_closests = []
        for i, name in enumerate(station_names):
            all_distances[i, :] = np.array(geoid.inverse(station_lonlat[i, :].reshape(1, 2),
                                                         station_lonlat))[:, 0]
            net_avg_closests.append(np.sort(all_distances[i, :])
                                    [1:min(num_stations, 1 + dist_num_avg)].mean())
        distance_weights = np.exp(-all_distances /
                                  np.clip(np.array(net_avg_closests), a_min=dist_weight_min,
                                          a_max=dist_weight_max).reshape(1, -1))
        # distance_weights is ignoring whether (1) a station actually has data, and
        # (2) if the spatial extent of the signal we're trying to estimate is correlated
        # to the station geometry
        if verbose:
            nonzero_distances = np.sort(np.triu(all_distances, k=1).ravel()
                                        )[int(num_stations * (num_stations + 1) / 2):]
            percs = [np.percentile(nonzero_distances, q) / 1e3 for q in [5, 50, 95]]
            tqdm.write("Distance percentiles in km (5-50-95): "
                       f"[{percs[0]:.1f}, {percs[1]:.1f}, {percs[2]:.1f}]")

        # first solve, default initial weights
        if verbose:
            tqdm.write("Initial fit")
        solutions = self.fit(ts_description,
                             solver="lasso_regression",
                             return_solutions=True,
                             progress_desc=None if verbose else "Initial fit",
                             no_pbar=no_pbar,
                             penalty=penalty,
                             reweight_max_iters=local_reweight_iters,
                             reweight_func=rw_func,
                             reweight_coupled=local_reweight_coupled,
                             return_weights=True,
                             formal_covariance=formal_covariance,
                             use_data_variance=use_data_variance,
                             use_data_covariance=use_data_covariance,
                             use_internal_scales=use_internal_scales,
                             cov_zero_threshold=cov_zero_threshold,
                             cvxpy_kw_args=cvxpy_kw_args)
        num_total = sum([s.models[ts_description][m].parameters.size
                         for s in valid_stations.values() for m in all_reweight_models])
        num_uniques = np.sum(np.stack(
            [np.sum(np.any(np.stack([np.abs(s.models[ts_description][m].parameters)
                                     > zero_threshold for s in valid_stations.values()]),
                           axis=0), axis=0) for m in all_reweight_models]), axis=0)
        num_nonzero = sum([(s.models[ts_description][m].parameters.ravel()
                            > zero_threshold).sum()
                           for s in valid_stations.values() for m in all_reweight_models])
        if verbose:
            tqdm.write(f"Number of reweighted non-zero parameters: {num_nonzero}/{num_total}")
            tqdm.write("Number of unique reweighted non-zero parameters per component: "
                       + str(num_uniques.tolist()))

        # initialize the other statistics objects
        num_components = num_uniques.size
        arr_uniques = np.empty((spatial_reweight_iters + 1, num_components))
        arr_uniques[:] = np.NaN
        arr_uniques[0, :] = num_uniques
        list_nonzeros = [np.NaN for _ in range(spatial_reweight_iters + 1)]
        list_nonzeros[0] = num_nonzero
        dict_rms_diff = {m: [np.NaN for _ in range(spatial_reweight_iters)]
                         for m in all_reweight_models}
        dict_num_changed = {m: [np.NaN for _ in range(spatial_reweight_iters)]
                            for m in all_reweight_models}

        # track parameters of weights of the reweighted models for early stopping
        old_params = {mdl_description:
                      Solution.aggregate_models(results_dict=solutions,
                                                mdl_description=mdl_description,
                                                key_list=station_names,
                                                stack_parameters=True,
                                                zeroed=True)[0]
                      for mdl_description in all_reweight_models}

        if extended_stats:
            # initialize extra statistics variables
            list_res_stats = []
            dict_cors = {mdl_description: [] for mdl_description in all_reweight_models}
            dict_cors_means = {mdl_description: [] for mdl_description in all_reweight_models}
            dict_cors_det = {mdl_description: [] for mdl_description in all_reweight_models}
            dict_cors_det_means = {mdl_description: [] for mdl_description in all_reweight_models}

            # define a function to save space
            def save_extended_stats(is_last=False):
                if is_last and isinstance(keep_mdl_res_as, tuple) and \
                   (len(keep_mdl_res_as) == 2):
                    keep_last = True
                    iter_name_fit = str(keep_mdl_res_as[0])
                    iter_name_res = str(keep_mdl_res_as[1])
                else:
                    keep_last = False
                    iter_name_fit = ts_description + "_extendedstats_fit"
                    iter_name_res = ts_description + "_extendedstats_res"
                # evaluate model fit to timeseries
                if num_threads_evaluate is not None:
                    curr_num_threads = defaults["general"]["num_threads"]
                    defaults["general"]["num_threads"] = int(num_threads_evaluate)
                self.evaluate(ts_description, output_description=iter_name_fit, no_pbar=no_pbar)
                if num_threads_evaluate is not None:
                    defaults["general"]["num_threads"] = curr_num_threads
                # calculate residuals
                self.math(iter_name_res, ts_description, "-", iter_name_fit)
                # analyze the residuals
                list_res_stats.append(
                    self.analyze_residuals(iter_name_res, mean=True, std=True,
                                           max_rolling_dev=roll_mean_kernel))
                # for each reweighted model fit, for each component,
                # get its spatial correlation matrix and average value
                for mdl_description in all_reweight_models:
                    net_mdl_df = list(self.export_network_ts((ts_description,
                                                              mdl_description)).values())
                    cormats = [mdl_df.df.corr().abs().values for mdl_df in net_mdl_df]
                    cormats_means = [np.nanmean(np.ma.masked_equal(np.triu(cormat, 1), 0))
                                     for cormat in cormats]
                    dict_cors[mdl_description].append(cormats)
                    dict_cors_means[mdl_description].append(cormats_means)
                    for i in range(len(net_mdl_df)):
                        raw_values = net_mdl_df[i].data.values
                        index_valid = np.isfinite(raw_values)
                        for j in range(raw_values.shape[1]):
                            raw_values[index_valid[:, j], j] = \
                                sp.signal.detrend(raw_values[index_valid[:, j], j])
                        net_mdl_df[i].data = raw_values
                    cormats = [mdl_df.df.corr().abs().values for mdl_df in net_mdl_df]
                    cormats_means = [np.nanmean(np.ma.masked_equal(np.triu(cormat, 1), 0))
                                     for cormat in cormats]
                    dict_cors_det[mdl_description].append(cormats)
                    dict_cors_det_means[mdl_description].append(cormats_means)
                # delete temporary timeseries if in between iterations
                if not keep_last:
                    self.remove_timeseries(iter_name_fit, iter_name_res)

            # run the function for the first time to capture the initial fit
            save_extended_stats()

        # iterate
        for i in range(spatial_reweight_iters):
            if verbose:
                tqdm.write("Updating weights")
            new_net_weights = {statname: {"reweight_init": {}} for statname in station_names}
            # reweighting spatial models
            for mdl_description in spatial_reweight_models:
                stacked_weights, = \
                    Solution.aggregate_models(results_dict=solutions,
                                              mdl_description=mdl_description,
                                              key_list=station_names,
                                              stack_weights=True)
                # if not None, stacking succeeded
                if np.any(stacked_weights):
                    if verbose:
                        tqdm.write(f"Stacking model {mdl_description}")
                        # print percentiles
                        percs = [np.nanpercentile(stacked_weights, q) for q in [5, 50, 95]]
                        tqdm.write("Weight percentiles (5-50-95): "
                                   f"[{percs[0]:.11g}, {percs[1]:.11g}, {percs[2]:.11g}]")
                    # now apply the spatial median to parameter weights
                    for station_index, name in enumerate(station_names):
                        new_net_weights[name]["reweight_init"][mdl_description] = \
                            weighted_median(stacked_weights,
                                            distance_weights[station_index, :],
                                            percentile=spatial_reweight_percentile)
                else:  # stacking failed, keep old weights
                    warn(f"{mdl_description} cannot be stacked, reusing old weights.",
                         stacklevel=2)
                    for name in station_names:
                        if mdl_description in solutions[name]:
                            new_net_weights[name]["reweight_init"][mdl_description] = \
                                solutions[name].weights_by_models(mdl_description)
            # copying over the old weights for the continuous models
            for mdl_description in continuous_reweight_models:
                for name in station_names:
                    if mdl_description in solutions[name]:
                        new_net_weights[name]["reweight_init"][mdl_description] = \
                            solutions[name].weights_by_models(mdl_description)
            # for models that were regularized but not spatially or continuously reweighted,
            # set the initial weights to penalty (such that it keeps a constant penalty)
            for name in station_names:
                for mdl_description, mdl in self[name].models[ts_description].collection.items():
                    if mdl.regularize and mdl_description not in all_reweight_models:
                        new_net_weights[name]["reweight_init"][mdl_description] = \
                            penalty * np.ones((mdl.num_parameters, num_components))
            # next solver step
            if verbose:
                tqdm.write(f"Fit after {i+1} reweightings")
            solutions = self.fit(ts_description,
                                 solver="lasso_regression",
                                 return_solutions=True,
                                 local_input=new_net_weights,
                                 progress_desc=None if verbose
                                 else f"Fit after {i+1} reweightings",
                                 no_pbar=no_pbar,
                                 penalty=penalty,
                                 reweight_max_iters=local_reweight_iters,
                                 reweight_func=rw_func,
                                 reweight_coupled=local_reweight_coupled,
                                 return_weights=True,
                                 formal_covariance=formal_covariance,
                                 use_data_variance=use_data_variance,
                                 use_data_covariance=use_data_covariance,
                                 cov_zero_threshold=cov_zero_threshold,
                                 cvxpy_kw_args=cvxpy_kw_args)
            # get statistics
            num_nonzero = sum([(s.models[ts_description][m].parameters.ravel()
                                > zero_threshold).sum()
                               for s in valid_stations.values() for m in all_reweight_models])
            num_uniques = np.sum(np.stack(
                [np.sum(np.any(np.stack([np.abs(s.models[ts_description][m].parameters)
                                        > zero_threshold for s in valid_stations.values()]),
                               axis=0), axis=0) for m in all_reweight_models]), axis=0)
            # save statistics
            arr_uniques[i+1, :] = num_uniques
            list_nonzeros[i+1] = num_nonzero
            # print
            if verbose:
                tqdm.write("Number of reweighted non-zero parameters: "
                           f"{num_nonzero}/{num_total}")
                tqdm.write("Number of unique reweighted non-zero parameters per component: "
                           + str(num_uniques.tolist()))
            # check for early stopping by comparing parameters that were reweighted
            early_stop = True
            for mdl_description in all_reweight_models:
                stacked_params, = \
                    Solution.aggregate_models(results_dict=solutions,
                                              mdl_description=mdl_description,
                                              key_list=station_names,
                                              stack_parameters=True,
                                              zeroed=True)
                # check for early stopping criterion and save current parameters
                rms_diff = np.linalg.norm(old_params[mdl_description] - stacked_params)
                num_changed = np.logical_xor(np.abs(old_params[mdl_description]) < zero_threshold,
                                             np.abs(stacked_params) < zero_threshold).sum()
                early_stop &= (rms_diff < spatial_reweight_max_rms) or \
                              (num_changed/num_total < spatial_reweight_max_changed)
                old_params[mdl_description] = stacked_params
                # save statistics
                dict_rms_diff[mdl_description][i] = rms_diff
                dict_num_changed[mdl_description][i] = num_changed
                # print
                if verbose:
                    tqdm.write(f"RMS difference of '{mdl_description}' parameters = "
                               f"{rms_diff:.11g} ({num_changed} changed)")
            # save extended statistics
            if extended_stats:
                save_extended_stats(is_last=early_stop or (i == spatial_reweight_iters - 1))
            # check if early stopping was triggered
            if early_stop:
                if verbose:
                    tqdm.write("Stopping iteration early.")
                break

        if verbose:
            tqdm.write("Done")

        # save statistics to dictionary
        stats_names = ["num_total", "arr_uniques", "list_nonzeros",
                       "dict_rms_diff", "dict_num_changed"]
        stats_values = [num_total, arr_uniques, list_nonzeros,
                        dict_rms_diff, dict_num_changed]
        if extended_stats:
            stats_names.extend(["list_res_stats", "dict_cors", "dict_cors_means",
                                "dict_cors_det", "dict_cors_det_means"])
            stats_values.extend([list_res_stats, dict_cors, dict_cors_means,
                                 dict_cors_det, dict_cors_det_means])
        statistics = {n: v for n, v in zip(stats_names, stats_values)}

        # return
        if return_solutions:
            return statistics, solutions
        else:
            return statistics

    def evaluate(self, ts_description, timevector=None, output_description=None,
                 progress_desc=None, no_pbar=False):
        """
        Evaluate a timeseries' models at all stations and adds them as a fit to
        the timeseries. Can optionally add the aggregate model as an independent
        timeseries to the station as well.
        Also provides a progress bar.
        Will automatically use multiprocessing if parallelization has been enabled in
        the configuration (defaults to parallelization if possible).

        Parameters
        ----------
        ts_description : str
            Description of the timeseries to evaluate.
        timevector : pandas.Series, pandas.DatetimeIndex, optional
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` of when to evaluate the model.
            Defaults to the timestamps of the timeseries itself.
        output_description : str, optional
            If provided, add the sum of the evaluated models as a new timeseries
            to each station with the provided description (instead of only adding
            the model as a fit *to* the timeseries).
        progress_desc : str, optional
            If provided, override the description of the progress bar.
        no_pbar : bool, optional
            Suppress the progress bar with ``True`` (default: ``False``).

        Example
        -------
        If ``net`` is a :class:`~Network` instance, ``'mydata'`` is the timeseries
        to evaluate the models for, then the following two are equivalent::

            # long version, not parallelized, and not creating a new independent timeseries
            for station in net:
                station_ts = station.timeseries['mydata']
                fit_all = station_models.evaluate(station_ts)
                station.add_fit('mydata', fit_all)
                for (model_description, model) in station_models.items():
                    fit = model.evaluate(station_ts)
                    station.add_fit('mydata', fit, model_description)
            # short version, automatically parallelizes according to disstans.defaults,
            # and creating a new timeseries
            net.evaluate('mydata', output_description='evaluated')

        See Also
        --------
        fit : Fit models at all stations.
        :attr:`~disstans.config.defaults` : Dictionary of settings, including parallelization.
        disstans.tools.parallelize : Automatically execute a function in parallel or serial.
        """
        assert isinstance(ts_description, str), \
            f"'ts_description' must be string, got {type(ts_description)}."
        if output_description is not None:
            assert isinstance(output_description, str), \
                f"'output_description' must be a string, got {type(output_description)}."
        progress_desc = str(progress_desc) if progress_desc else "Evaluating station models"
        station_names = [name for name, station in self.stations.items()
                         if ts_description in station.timeseries]
        iterable_inputs = ((self[stat_name].timeseries[ts_description].time
                            if timevector is None else timevector,
                            self[stat_name].models[ts_description])
                           for stat_name in station_names)
        for i, (sumfit, mdlfit) in enumerate(tqdm(parallelize(self._evaluate_single_station,
                                                              iterable_inputs),
                                                  desc=progress_desc, total=len(station_names),
                                                  ascii=True, unit="station", disable=no_pbar)):
            stat = self[station_names[i]]
            # add overall model
            ts = stat.add_fit(ts_description, sumfit, return_ts=(output_description is not None))
            # if desired, add as timeseries
            if output_description is not None:
                stat.add_timeseries(output_description, ts)
            # add individual fits
            for model_description, fit in mdlfit.items():
                stat.add_fit(ts_description, fit, model_description)

    @staticmethod
    def _evaluate_single_station(parameter_tuple):
        station_time, station_models = parameter_tuple
        sumfit = station_models.evaluate(station_time)
        mdlfit = {}
        for model_description, model in station_models.items():
            mdlfit[model_description] = model.evaluate(station_time)
        return sumfit, mdlfit

    def fitevalres(self, ts_description, solver='linear_regression', local_input={},
                   return_solutions=False, timevector=None, output_description=None,
                   residual_description=None, progress_desc=None, no_pbar=False, **kw_args):
        """
        Convenience method that combines the calls for :meth:`~fit`, :meth:`~evaluate`
        and :meth:`~math` (to compute the fit residual) into one method call.

        Parameters
        ----------
        ts_description : str
            See :meth:`~fit` and :meth:`~evaluate`.
        solver : str, function, optional
            See :meth:`~fit`.
        local_input : dict, optional
            See :meth:`~fit`.
        return_solutions : bool, optional
            See :meth:`~fit`.
        timevector : pandas.Series, pandas.DatetimeIndex, optional
            See :meth:`~evaluate`.
        output_description : str, optional
            See :meth:`~evaluate`.
        residual_description : str, optional
            If provided, calculate the residual as the difference between the data and
            the model fit, and store it as the timeseries ``residual_description``.
        progress_desc : list, tuple, optional
            If provided, set ``progress_desc`` of :meth:`~fit` and :meth:`~evaluate`
            using the first or second element of a tuple or list, respectively.
            Leave ``None`` if only overriding one of them.
        no_pbar : bool, optional
            Suppress the progress bars with ``True`` (default: ``False``).
        **kw_args : dict
            Additional keyword arguments that are passed on to the solver function,
            see :meth:`~fit`.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, ``'mydata'`` is the timeseries to fit
        and evaluate the models for, ``mysolver`` is the solver to use, ``'myfit'`` should
        contain the model fit, and the residuals should be put into the timeseries
        ``'myres'``, then the following two are equivalent::

            # three-line version
            net.fit('mydata', solver=mysolver, **kw_args)
            net.evaluate('mydata', output_description='myfit')
            net.math('myres', 'mydata', '-', 'myfit')
            # one-liner
            net.fitevalres('mydata', solver=mysolver, output_description='myfit',
                           residual_description='myres')

        See Also
        --------
        evaluate : Evaluate the fitted models at all stations.
        fit : Fit models at all stations.
        :attr:`~disstans.config.defaults` : Dictionary of settings, including parallelization.
        disstans.tools.parallelize : Automatically execute a function in parallel or serial.
        """
        if progress_desc is not None:
            assert (isinstance(progress_desc, tuple) or isinstance(progress_desc, list)) \
                   and (len(progress_desc) == 2), "If 'progress_desc' is not None, it needs " \
                   f"to be a 2-element tuple or list, got {progress_desc}."
        else:
            progress_desc = [None, None]
        # fit
        possible_output = self.fit(ts_description=ts_description,
                                   solver=solver, local_input=local_input,
                                   return_solutions=return_solutions,
                                   progress_desc=progress_desc[0],
                                   no_pbar=no_pbar, **kw_args)
        # evaluate
        self.evaluate(ts_description=ts_description, timevector=timevector, no_pbar=no_pbar,
                      output_description=output_description, progress_desc=progress_desc[0])
        # residual
        if residual_description:
            self.math(residual_description, ts_description, "-", output_description)
        # return either None or the solutions dictionary of self.fit():
        return possible_output

    def call_func_ts_return(self, func, ts_in, ts_out=None, no_pbar=False, **kw_args):
        """
        A convenience wrapper that for each station in the network, calls a given
        function which takes timeseries as input and returns a timeseries (which
        is then added to the station). Also provides a progress bar.
        Will automatically use multiprocessing if parallelization has been enabled in
        the configuration (defaults to parallelization if possible).

        Parameters
        ----------
        func : str, function
            Function to use. If provided with a string, will check for that function in
            :mod:`~disstans.processing`, otherwise the function will be assumed to adhere
            to the same input and output format than the included ones.
        ts_in : str
            Name of the timeseries to use as input to ``func``.
        ts_out : str, optional
            Name of the timeseries that the output of ``func`` should be assigned to.
            Defaults to overwriting ``ts_in``.
        no_pbar : bool, optional
            Suppress the progress bar with ``True`` (default: ``False``).
        **kw_args : dict
            Additional keyword arguments to be passed onto ``func``.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, ``'input'`` and ``'output'``
        are the names of the input and output timeseries, respectively, and ``func``
        is the function to use, then the following two are equivalent::

            # long version, not parallelized
            for station in net:
                ts_in = station.timeseries['input']
                ts_out = func(ts_in, **kw_args)
                station.add_timeseries('output', ts_out)
            # short version, automatically parallelizes according to disstans.defaults
            net.call_func_ts_return(func, 'input', ts_out='output', **kw_args)
            # if using a disstans.processing function, no need to import that function
            net.call_func_ts_return('clean', 'input', ts_out='output', **kw_args)

        See Also
        --------
        :attr:`~disstans.config.defaults` : Dictionary of settings, including parallelization.
        disstans.tools.parallelize : Automatically execute a function in parallel or serial.
        """
        if not callable(func):
            if isinstance(func, str):
                try:
                    func = getattr(disstans_processing, func)
                except AttributeError as e:
                    raise AttributeError(
                        f"'{func}' can not be found as a function in disstans.processing."
                        ).with_traceback(e.__traceback__) from e
            else:
                raise RuntimeError(f"'{func}' needs to be a function or a string "
                                   "representation thereof "
                                   "(if loaded from disstans.processing).")
        assert isinstance(ts_in, str), f"'ts_in' must be string, got {type(ts_in)}."
        if ts_out is None:
            ts_out = ts_in
        else:
            assert isinstance(ts_out, str), \
                f"'ts_out' must be None or a string, got {type(ts_out)}."
        station_names = [stat_name for stat_name, stat in self.stations.items()
                         if ts_in in stat.timeseries]
        iterable_inputs = ((func, self[name], ts_in, kw_args) for name in station_names)
        for i, result in enumerate(tqdm(parallelize(self._single_call_func_ts_return,
                                                    iterable_inputs),
                                        desc="Processing station timeseries with "
                                        f"'{func.__name__}'", total=len(station_names),
                                        ascii=True, unit="station", disable=no_pbar)):
            self[station_names[i]].add_timeseries(ts_out, result)

    @staticmethod
    def _single_call_func_ts_return(parameter_tuple):
        func, station, ts_description, kw_args = parameter_tuple
        ts_return = func(station.timeseries[ts_description], **kw_args)
        return ts_return

    def call_netwide_func(self, func, ts_in, ts_out=None, **kw_args):
        """
        A convenience wrapper for functions that operate on an entire network's timeseries
        at once.

        Parameters
        ----------
        func : str, function
            Function to use. If provided with a string, will check for that function in
            :mod:`~disstans.processing`, otherwise the function will be assumed to adhere
            to the same input and output format than the included ones.
        ts_in : str
            Name of the timeseries to use as input to ``func``.
        ts_out : str, optional
            Name of the timeseries that the output of ``func`` should be assigned to.
            Defaults to overwriting ``ts_in``.
        **kw_args : dict
            Additional keyword arguments to be passed onto ``func``.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, ``'input'`` and ``'output'`` are the names
        of the input and output timeseries, respectively, and ``func`` is the function to use,
        then the following two are equivalent::

            # long version
            net_in = net.export_network_ts('input')
            net_out = func(net_in, **kw_args)
            net.import_network_ts('output', net_out)
            # short version
            net.call_netwide_func(func, ts_in='input, ts_out='output', **kw_args)
            # the short version also knows about the functions in disstans.processing
            net.call_netwide_func('decompose', ts_in='input, ts_out='output', **kw_args)

        See Also
        --------
        export_network_ts : Export the timeseries of all stations in the network.
        import_network_ts : Import an entire netwrok's timeseries and distribute to stations.
        """
        if not callable(func):
            if isinstance(func, str):
                try:
                    func = getattr(disstans_processing, func)
                except AttributeError as e:
                    raise AttributeError(
                        f"'{func}' can not be found as a function in disstans.processing."
                        ).with_traceback(e.__traceback__) from e
            else:
                raise RuntimeError(f"'{func}' needs to be a function or a string "
                                   "representation thereof "
                                   "(if loaded from disstans.processing).")
        assert isinstance(ts_in, str), f"'ts_in' must be string, got {type(ts_in)}."
        if ts_out is None:
            ts_out = ts_in
        else:
            assert isinstance(ts_out, str), \
                f"'ts_out' must be None or a string, got {type(ts_out)}."
        net_in = self.export_network_ts(ts_in)
        net_out = func(net_in, **kw_args)
        self.import_network_ts(ts_in if ts_out is None else ts_out, net_out)

    def call_func_no_return(self, func, no_pbar=False, **kw_args):
        """
        A convenience wrapper that for each station in the network, calls a given function on
        each station. Also provides a progress bar.

        Parameters
        ----------
        func : str, function
            Function to use. If provided with a string, will check for that function in
            :mod:`~disstans.processing`, otherwise the function will be assumed to adhere to the
            same input format than the included ones.
        no_pbar : bool, optional
            Suppress the progress bar with ``True`` (default: ``False``).
        **kw_args : dict
            Additional keyword arguments to be passed onto ``func``.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, and ``func`` is the function to use,
        then the following two are equivalent::

            # long version
            for station in net:
                func(station, **kw_args)
            # shorter version
            net.call_func_no_return(func, **kw_args)
            # shorter version, no need to import a disstans.processing function first
            net.call_func_no_return('clean', **kw_args)
        """
        if not callable(func):
            if isinstance(func, str):
                try:
                    func = getattr(disstans_processing, func)
                except AttributeError as e:
                    raise AttributeError(
                        f"'{func}' can not be found as a function in disstans.processing."
                        ).with_traceback(e.__traceback__) from e
            else:
                raise RuntimeError(f"'{func}' needs to be a function or a string "
                                   "representation thereof "
                                   "(if loaded from disstans.processing).")
        for name, station in tqdm(self.stations.items(),
                                  desc=f"Calling function '{func.__name__}' on stations",
                                  ascii=True, unit="station", disable=no_pbar):
            func(station, **kw_args)

    def math(self, result, left, operator, right):
        """
        Convenience method that performs simple math for all stations.
        Syntax is ``result = left operator right``.

        Parameters
        ----------
        result : str
            Name of the result timeseries.
        left : str
            Name of the left timeseries.
        operator : str
            Operator symbol (see :meth:`~disstans.timeseries.Timeseries.prepare_math` for
            all supported operations).
        right : str
            Name of the right timeseries.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, and we want to calculate the residual
        ``'res'`` between the data in timeseries ``'data'`` and the model in the timeseries
        ``'model'``, then the following two are equivalent::

            # long version
            for station in net:
                station['res'] = station['data'] - station['model']

            # short version
            net.math('res', 'data', '-', 'model')
        """
        # check that inputs are valid
        assert all([isinstance(p, str) for p in [result, left, operator, right]]), \
            "All inputs to Network.math() need to be strings, got " \
            f"{[result, left, operator, right]}."
        for station in self:
            if (left in station.timeseries) and (right in station.timeseries):
                station[result] = eval(f"station['{left}'] {operator} station['{right}']")

    def analyze_residuals(self, ts_description, **kw_args):
        """
        Analyze residual timeseries for all stations and components, calling
        :meth:`~disstans.station.Station.analyze_residuals`.

        Parameters
        ----------
        ts_description : str
            Timeseries to analyze. Method assumes it already is a residual.
        **kw_args : dict
            Additional keyword arguments are directly passed on to
            :meth:`~disstans.station.Station.analyze_residuals`.

        Returns
        -------
        pandas.DataFrame
            DataFrame with the station names as index, the different metrics
            as top-level columns, and the different components as lower-level
            columns. Averages of each metric can easily be calculated with the
            Pandas command :meth:`~pandas.DataFrame.mean`.
        """
        # get the stations who have this timeseries
        valid_stations = {name: station for name, station in self.stations.items()
                          if ts_description in station.timeseries}
        # calculate the desired metrics
        metrics_lists = [station.analyze_residuals(ts_description, **kw_args)
                         for station in valid_stations.values()]
        # stack the data over the different metrics and components into a NumPy array
        metrics_arr = np.array([[component for metric in sta_metrics.values()
                                 for component in metric]
                                for sta_metrics in metrics_lists])
        # what are the metrics and components in the dictionary?
        # (need to preserve order, hence not using set())
        metrics = list(dict.fromkeys([metric for sta_metrics in metrics_lists
                                      for metric in sta_metrics]))
        components = list(dict.fromkeys([dcol for station in valid_stations.values()
                                         for dcol in station[ts_description].data_cols]))
        # create a Pandas MultiIndex
        metrics_components = pd.MultiIndex.from_product([metrics, components],
                                                        names=["Metrics", "Components"])
        # create and return a multi-level DataFrame
        return pd.DataFrame(metrics_arr,
                            index=list(valid_stations.keys()),
                            columns=metrics_components).rename_axis("Station")

    def decompose(self, ts_in, ts_out=None, **decompose_kw_args):
        r"""
        Decomposes a timeseries in the network and returns its best-fit models
        and spatiotemporal sources.

        Parameters
        ----------
        ts_in : str
            Name of the timeseries to analyze.
        ts_out : str, optional
            If provided, save the model as a timeseries called ``ts_out`` to
            the stations in the network.
        **decompose_kw_args : dict
            Additional keyword arguments passed to :func:`~disstans.processing.decompose`.

        Returns
        -------
        model : dict
            Dictionary of best-fit models at each station and for each data component.
            The keys are the data components of the ``ts_in`` timeseries, and
            the values are :class:`~disstans.timeseries.Timeseries` objects.
            The Timeseries objects have the station names as columns, and their observation
            record length is the joint length of all stations in the network,
            :math:`\text{num_observations}`. If a station is missing a certain timestamp,
            that value is ``NaN``.
        spatial : numpy.ndarray
            Dictionary of the spatial sources. Each key is one of the data components,
            and the values are :class:`~numpy.ndarray` objects of shape
            :math:`(\text{num_components},\text{n_stations})`.
        temporal : dict
            Dictionary of the temporal sources. Each key is one of the data components,
            and the values are :class:`~numpy.ndarray` objects of shape
            :math:`(\text{num_observations},\text{num_components})`.

        See Also
        --------
        disstans.processing.decompose : Function to get the individual components.
        """
        # prepare
        net_in = self.export_network_ts(ts_in)
        comps = list(net_in.keys())
        ndim = len(comps)
        assert ndim > 0, f"No components found in '{ts_in}'."
        decompose_kw_args.update({'return_sources': True})
        # run decomposer
        model, temp_spat = disstans_processing.decompose(net_in, **decompose_kw_args)
        spatial, temporal = {}, {}
        for comp in model:
            spatial[comp] = temp_spat[comp][1]
            temporal[comp] = temp_spat[comp][0]
        # assign model to network if desired
        if ts_out is not None:
            self.import_network_ts(ts_out, model)
        return model, spatial, temporal

    def _create_map_figure(self, gui_settings, annotate_stations, subset_stations=None):
        # get location data and projections
        if subset_stations:
            stat_names = subset_stations
            stat_list = [self[staname] for staname in stat_names]
        else:
            stat_names = self.station_names
            stat_list = self.stations.values()
        stat_lats = [station.location[0] for station in stat_list]
        stat_lons = [station.location[1] for station in stat_list]
        proj_gui = getattr(ccrs, gui_settings["projection"])()
        proj_lla = ccrs.PlateCarree()
        # create figure and plot stations
        fig_map = plt.figure()
        ax_map = fig_map.add_subplot(projection=proj_gui)
        default_station_edges = ['none'] * len(stat_names)
        stat_points = ax_map.scatter(stat_lons, stat_lats, s=20, facecolor='C0',
                                     edgecolor=default_station_edges, marker='o',
                                     transform=proj_lla, zorder=1000)
        # add labels
        if annotate_stations:
            anno_kw_args = {"xycoords": proj_lla._as_mpl_transform(ax_map),
                            "annotation_clip": True, "textcoords": "offset pixels",
                            "xytext": (0, 5), "ha": "center"}
            if isinstance(annotate_stations, float) or isinstance(annotate_stations, str):
                anno_kw_args["fontsize"] = annotate_stations
            for sname, slon, slat in zip(stat_names, stat_lons, stat_lats):
                ax_map.annotate(sname, (slon, slat), **anno_kw_args)
        # create underlay
        map_underlay = False
        if gui_settings["wmts_show"]:
            try:
                ax_map.add_wmts(gui_settings["wmts_server"],
                                layer_name=gui_settings["wmts_layer"],
                                alpha=gui_settings["wmts_alpha"])
                map_underlay = True
            except Exception as exc:
                print(exc)
        if gui_settings["coastlines_show"]:
            ax_map.coastlines(color="white" if map_underlay else "black",
                              resolution=gui_settings["coastlines_res"])
        return fig_map, ax_map, proj_gui, proj_lla, default_station_edges, \
            stat_points, stat_lats, stat_lons

    def graphical_cme(self, ts_in, ts_out=None, annotate_stations=True, save=False,
                      save_kw_args={"format": "png"}, gui_kw_args={}, **decompose_kw_args):
        """
        Calculates the Common Mode Error (CME) of the network and shows its spatial
        and temporal pattern. Optionally saves the model for each station.

        Parameters
        ----------
        ts_in : str
            Name of the timeseries to analyze.
        ts_out : str, optional
            If provided, save the model as a timeseries called ``ts_out`` to
            the stations in the network.
        annotate_stations : bool, float, str, optional
            If ``True`` (default), add the station names to the map.
            If a float or a string, add the station names to the map with the font size set
            as required by :class:`~matplotlib.test.Text`.
        save : bool, optional
            If ``True``, save the map and timeseries plots to the current folder.
            Defaults to ``False``.
        save_kw_args : dict, optional
            Additional keyword arguments passed to :meth:`~matplotlib.figure.Figure.savefig`,
            used when ``save=True``.
        gui_kw_args : dict, optional
            Override default GUI settings of :attr:`~disstans.config.defaults`.
        **decompose_kw_args : dict
            Additional keyword arguments passed to :func:`~decompose`.

        See Also
        --------
        decompose : Decomposer method to calculate the CME.
        """
        gui_settings = defaults["gui"].copy()
        gui_settings.update(gui_kw_args)
        # get common mode and make sure to return spatial and temporal models
        model, spatial, temporal = self.decompose(ts_in, ts_out=ts_out, **decompose_kw_args)
        comps = list(model.keys())
        ndim = len(comps)
        # extract spatial components
        ndim_max2 = min(ndim, 2)
        fitted_stations = model[comps[0]].data_cols
        latlonenu = np.zeros((len(fitted_stations), 2 + ndim_max2))
        for i, station_name in enumerate(fitted_stations):
            latlonenu[i, :2] = self[station_name].location[:2]
            for j in range(ndim_max2):
                latlonenu[i, 2 + j] = spatial[comps[j]][0, i]
        # make map for spatial component
        fig_map, ax_map, proj_gui, proj_lla, default_station_edges, \
            stat_points, stat_lats, stat_lons = self._create_map_figure(gui_settings,
                                                                        annotate_stations)
        if ndim == 1:
            quiv = ax_map.quiver(latlonenu[:, 1], latlonenu[:, 0],
                                 np.zeros_like(latlonenu[:, 2]), latlonenu[:, 2],
                                 units='xy', transform=proj_lla, clip_on=False)
        else:
            quiv = ax_map.quiver(latlonenu[:, 1], latlonenu[:, 0],
                                 latlonenu[:, 2], latlonenu[:, 3],
                                 units='xy', transform=proj_lla, clip_on=False)
        key_length = np.median(np.sqrt(np.sum(np.atleast_2d(latlonenu[:, 2:2 + ndim_max2]**2),
                                              axis=1)))
        ax_map.quiverkey(quiv, 0.9, 0.9, key_length,
                         f"{key_length:.2e} {model[comps[0]].data_unit:s}",
                         coordinates="figure")
        # make timeseries figure
        fig_ts, ax_ts = plt.subplots(nrows=len(temporal), sharex=True)
        for icomp, (comp, ts) in enumerate(temporal.items()):
            ax_ts[icomp].plot(model[comp].time, ts)
            ax_ts[icomp].set_ylabel(f"{comp:s} [{model[comp].data_unit:s}]")
            ax_ts[icomp].grid()
        # show plots or save
        if save:
            fig_map.savefig(f"cme_spatial.{save_kw_args['format']}", **save_kw_args)
            fig_ts.savefig(f"cme_temporal.{save_kw_args['format']}", **save_kw_args)
            plt.close(fig_map)
            plt.close(fig_ts)
        else:
            plt.show()

    def gui(self, station=None, timeseries=None, fit_list=None, sum_models=True,
            verbose=False, annotate_stations=True, save=False, save_map=False,
            save_kw_args={"format": "png"}, scalogram_kw_args=None, mark_events=None,
            stepdetector={}, trend_kw_args={}, analyze_kw_args={}, rms_on_map={},
            gui_kw_args={}):
        """
        Provides a Graphical User Interface (GUI) to visualize the network and all
        of its different stations, timeseries, and models.

        In its base form, this function will show

        - a window with a map of all stations, underlain optionally with coastlines
          and/or imagery, and
        - another window which will show a station's timeseries including all fitted models.

        Stations are selected by clicking on their markers on the map.
        Optionally, this function can

        - start with a station pre-selected,
        - show only a subset of fitted models,
        - sum the models to an aggregate one,
        - mark events associated with a station's timeseries,
        - restrict the output to a timewindow showing potential steps from multiple sources,
        - color the station markers by RMS and include a colormap,
        - show a scalogram (Model class permitting),
        - save the output timeseries (and scalogram) as an image,
        - print statistics of residuals, and
        - print station information.

        Parameters
        ----------
        station : str, optional
            Pre-select a station.
        timeseries : list, optional
            List of strings with the descriptions of the timeseries to plot.
            Defaults to all timeseries.
        fit_list : list, optional
            List of strings containing the model names of the subset of the models
            to be plotted. Defaults to all fitted models.
        sum_models : bool, optional
            If ``True``, plot the sum of all selected models instead of every
            model individually. Defaults to ``True``.
        verbose : bool, optional
            If ``True``, when clicking on a station, print its details (see
            :meth:`~disstans.station.Station.__str__`). Defaults to ``False``.
        annotate_stations : bool, float, str, optional
            If ``True`` (default), add the station names to the map.
            If a float or a string, add the station names to the map with the font size set
            as required by :class:`~matplotlib.test.Text`.
        save : bool, str, optional
            If ``True``, save the figure of the selected timeseries. If a scalogram
            is also created, save this as well. The output directory is the current folder.
            Ignored if ``stepdetector`` is set. Suppresses all interactive figures.
            If ``save`` is a string, it will be included in the output file name
            for easier referencing.
            Defaults to ``False``.
        save_map : bool, optional
            If ``True``, also save a map if ``save=True``. Defaults to ``False``.
        save_kw_args : dict, optional
            Additional keyword arguments passed to :meth:`~matplotlib.figure.Figure.savefig`,
            used when ``save=True``.
        scalogram_kw_args : dict, optional
            If passed, also plot a scalogram. Defaults to no scalogram shown.
            The dictionary has to contain ``'ts'`` and ``'model'`` keys. The string values
            are the names of the timeseries and associated model that are of the
            :class:`~disstans.models.SplineSet` class, and therefore have a
            :meth:`~disstans.models.SplineSet.make_scalogram` method.
        mark_events : pandas.DataFrame, list, optional
            If passed, a DataFrame or list of DataFrames that contain the columns
            ``'station'`` and ``'time'``. For each timestamp, a vertical line is plotted
            onto the station's timeseries and the relevant entries are printed out.
        stepdetector : dict, optional
            Passing this dictionary will enable the plotting of events related to possible
            steps, both on the map (in case of an earthquake catalog) and in the timeseries
            (in case of detected steps by :class:`~disstans.processing.StepDetector` or a
            maintenance catalog). To reduce cluttering, it will also only show a subset of
            each timeseries, centered around consecutive probable steps. Using the terminal,
            one can cycle through each period.
            The ``stepdetector`` dictionary must contain the keys ``'plot_padding'`` (which
            determines how many days before and after to include in the plotting),
            ``'step_table'`` and ``'step_ranges'`` (both returned by
            :meth:`~disstans.processing.StepDetector.search_network`),
            ``'step_padding'`` (which determines how many days before and after the
            ``'step_ranges'`` should be scanned for possible steps), optionally ``'catalog'``
            and ``'eqcircle'`` (an earthquake catalog of same style as used in
            :mod:`~disstans.earthquakes` and a maximum distance of stations to earthquakes of
            magnitude less than 7.5), and optionally, ``'maint_table'`` (a maintenance table
            as parsed by :func:`~disstans.tools.parse_maintenance_table`).
        trend_kw_args : dict, optional
            If passed, also plot velocity trends on the station map.
            Defaults to no velocity arrows shown.
            The dictionary can contain all the keywords that are passed to
            :meth:`~disstans.station.Station.get_trend`, but has at least has to contain
            the ``ts_description``. If no ``fit_list`` is included, the ``fit_list``
            passed to ``gui()`` will be used instead. If the number of components available
            is 3 or more, only the first two will be used. If two components are plotted,
            they correspond to the East and North components. If only one component is plotted
            (for example for vertical motion), it will be plotted as the North component.
        analyze_kw_args : dict, optional
            If provided and non-empty, call :meth:`~disstans.station.Station.analyze_residuals`
            and pass the dictionary on as keyword arguments (overriding ``'verbose'`` to
            ``True`` to force an output). Defaults to no residual analysis.
        rms_on_map : dict, optional
            If provided and non-empty, this option will call
            :meth:`~disstans.station.Station.analyze_residuals` to calculate
            a residual timeseries' root-mean-squares to color the station markers on the map.
            The dictionary must include the key ``'ts'`` (the residual timeseries' name), and
            can optionally include the keys ``'comps'`` (a list of components to combine for
            the RMS, defaults to all components), ``'c_max'`` (maximum colormap range,
            defaults to maximum of the RMS), ``'t_start', 't_end'`` (to restrict the time
            window, defaults to ``analyze_kw_args`` if given, otherwise the entire timeseries),
            and ``'orientation'`` (for the colorbar orientation).
        gui_kw_args : dict, optional
            Override default GUI settings of :attr:`~disstans.config.defaults`.
        """
        # create map and timeseries figures
        gui_settings = defaults["gui"].copy()
        gui_settings.update(gui_kw_args)
        fig_map, ax_map, proj_gui, proj_lla, default_station_edges, \
            stat_points, stat_lats, stat_lons = self._create_map_figure(gui_settings,
                                                                        annotate_stations)
        ax_map_xmin, ax_map_xmax, ax_map_ymin, ax_map_ymax = ax_map.get_extent()
        fig_ts = plt.figure()
        if scalogram_kw_args is not None:
            assert isinstance(scalogram_kw_args, dict) \
                and all([key in scalogram_kw_args for key in ['ts', 'model']]) \
                and all([isinstance(scalogram_kw_args[key], str)
                         for key in ['ts', 'model']]), \
                "If a scalogram plot is requested, 'scalogram' must be a dictionary " \
                "with 'ts' and 'model' keys with string values of where to find the " \
                f"SplineSet model, got {scalogram_kw_args}."
            scalo_ts = scalogram_kw_args.pop('ts')
            scalo_model = scalogram_kw_args.pop('model')
            fig_scalo = plt.figure()
        station_name, station_index = None, None
        stat_lonlats = np.stack((stat_lons, stat_lats), axis=1)
        geoid = cgeod.Geodesic()

        # make sure that if analyze_kw_args is used, 'verbose' is set and True
        if analyze_kw_args:
            analyze_kw_args["verbose"] = True

        # mark_events is mutually exclusive to stepdetector
        assert not ((mark_events is not None) and stepdetector), \
            "Functions 'mark_events' and 'stepdetector' are mutually exclusive."

        # check the save settings
        if isinstance(save, str):
            fname_add = f"_{save}"
            save = True
        elif save:
            fname_add = ""

        # color the station markers by RMS
        if rms_on_map:
            # check input
            assert isinstance(rms_on_map, dict) and ("ts" in rms_on_map) and \
                   isinstance(rms_on_map["ts"], str), \
                   "'rms_on_map' needs to be a dictionary including the key 'ts' with a " \
                   f"string value, got {rms_on_map}."
            # collect RMS
            rms_comps = rms_on_map.get("comps", True)
            rms_kw_args = {"rms": rms_comps}
            for k in ["t_start", "t_end"]:
                if k in rms_on_map:
                    rms_kw_args[k] = rms_on_map[k]
                elif k in analyze_kw_args:
                    rms_kw_args[k] = analyze_kw_args[k]
            rms = np.zeros(self.num_stations)
            for i, stat in enumerate(self):
                rms[i] = (np.linalg.norm(stat.analyze_residuals(rms_on_map["ts"],
                                                                **rms_kw_args)["RMS"])
                          if rms_on_map["ts"] in stat.timeseries else 0)
            # make colormap
            cmax = rms_on_map.get("c_max", np.max(rms))
            rms_cmap = mpl.cm.ScalarMappable(cmap=scm.lajolla_r,
                                             norm=mpl.colors.Normalize(vmin=0, vmax=cmax))
            # update marker facecolors
            stat_points.set_facecolor(rms_cmap.to_rgba(rms))
            fig_map.canvas.draw_idle()
            # add colorbar
            orientation = rms_on_map.get("orientation", "horizontal")
            cbar = fig_map.colorbar(rms_cmap, ax=ax_map, orientation=orientation,
                                    fraction=0.05, pad=0.03, aspect=10,
                                    extend="max" if np.max(rms) > cmax else "neither")
            cbar.set_label(f"RMS [{stat[rms_on_map['ts']].data_unit}]")

        # add velocity map
        if trend_kw_args:
            assert "ts_description" in trend_kw_args, \
                "'trend_kw_args' dictionary has to include a 'ts_description' keyword."
            if "fit_list" not in trend_kw_args:
                trend_kw_args["fit_list"] = fit_list
            # loop over stations
            trend = np.zeros((self.num_stations, 2))
            trend_sigma = np.zeros_like(trend)
            for i, stat in enumerate(self):
                # need to check the data and time units once
                if i == 0:
                    if ("total" not in trend_kw_args) or (not trend_kw_args["total"]):
                        if "time_unit" in trend_kw_args:
                            tunit = trend_kw_args["time_unit"]
                        else:
                            tunit = "D"
                    else:
                        tunit = None
                    trend_unit = stat[trend_kw_args["ts_description"]].data_unit
                    if tunit is not None:
                        trend_unit += f"/{tunit}"
                # calculate trend
                stat_trend, stat_trend_sigma = stat.get_trend(**trend_kw_args)
                # save trend
                if stat_trend is not None:
                    if stat_trend.size == 1:
                        trend[i, 1] = stat_trend
                    else:
                        trend[i, :] = stat_trend[:2]
                else:
                    trend[i, :] = np.NaN
                # save uncertainty
                if stat_trend_sigma is not None:
                    if stat_trend_sigma.size == 1:
                        trend_sigma[i, 1] = stat_trend_sigma
                    else:
                        trend_sigma[i, :] = stat_trend_sigma[:2]
                else:
                    trend_sigma[i, :] = np.NaN
            # plot arrows
            quiv = ax_map.quiver(np.array(stat_lons), np.array(stat_lats),
                                 trend[:, 0], trend[:, 1],
                                 units='xy', transform=proj_lla)
            key_length = np.nanpercentile(np.sqrt(np.nansum(trend**2, axis=1)), 90)
            ax_map.quiverkey(quiv, 0.9, 0.9, key_length,
                             f"{key_length:.2g} {trend_unit:s}",
                             coordinates="figure")

        # check if mark_events is either a DataFrame or list of DataFrames,
        # containing the necessary columns
        if mark_events is not None:
            if isinstance(mark_events, pd.DataFrame):
                assert all([col in mark_events.columns for col in ["station", "time"]]), \
                    "'mark_events' needs to contain the columns 'station' and 'time'."
            elif (isinstance(mark_events, list) and
                  all([isinstance(elem, pd.DataFrame) for elem in mark_events])):
                assert all([all([col in elem.columns for col in ["station", "time"]])
                            for elem in mark_events]), "All DataFrames in " + \
                    "'mark_events' need to contain the columns 'station' and 'time'."
                mark_events = pd.concat(mark_events, ignore_index=True)
            else:
                raise ValueError("Invalid input format for 'mark_events'.")

        # prepare stuff for the interactive step plotting
        if stepdetector:
            # get quick access
            step_table = stepdetector["step_table"]
            step_ranges = stepdetector["step_ranges"]
            maint_table = stepdetector.get("maint_table", None)
            if "catalog" in stepdetector:
                catalog = stepdetector["catalog"]
                eqcircle = stepdetector["eqcircle"]
            else:
                catalog = None
            step_padding = stepdetector["step_padding"]
            plot_padding = stepdetector["plot_padding"]
            n_ranges = len(step_ranges)
            i_range = 0
            last_eqs = None
            last_eqs_lbl = []

        # define clicking function
        def update_timeseries(event, select_station=None):
            nonlocal analyze_kw_args, gui_settings, station_name, station_index
            # select station
            if select_station is None:
                if event is not None:
                    if (event.xdata is None) \
                       or (event.ydata is None) \
                       or (event.inaxes is not ax_map):
                        return
                    click_lon, click_lat = proj_lla.transform_point(event.xdata, event.ydata,
                                                                    src_crs=proj_gui)
                    distances = geoid.inverse(np.array([[click_lon, click_lat]]), stat_lonlats)
                    station_index = np.argmin(np.array(distances)[:, 0])
                    station_name = self.station_names[station_index]
                elif station_name is None:
                    return
            else:
                station_name = select_station
                station_index = self.station_names.index(station_name)
            if verbose:
                print(self[station_name])
            # change marker edges
            highlight_station_edges = default_station_edges.copy()
            highlight_station_edges[station_index] = 'k'
            stat_points.set_edgecolor(highlight_station_edges)
            fig_map.canvas.draw_idle()
            # get components
            ts_to_plot = {ts_description: ts for ts_description, ts
                          in self[station_name].timeseries.items()
                          if (timeseries is None) or (ts_description in timeseries)}
            n_components = 0
            for ts_description, ts in ts_to_plot.items():
                n_components += ts.num_components
                if analyze_kw_args:
                    self[station_name].analyze_residuals(ts_description, **analyze_kw_args)
            # clear figure and add data
            fig_ts.clear()
            icomp = 0
            ax_ts = []
            if scalogram_kw_args is not None:
                nonlocal fig_scalo
                t_left, t_right = None, None
            for its, (ts_description, ts) in enumerate(ts_to_plot.items()):
                for icol, (data_col, var_col) in enumerate(zip(ts.data_cols,
                                                               [None] * len(ts.data_cols)
                                                               if ts.var_cols is None
                                                               else ts.var_cols)):
                    # add axis
                    ax = fig_ts.add_subplot(n_components, 1, icomp + 1,
                                            sharex=None if icomp == 0 else ax_ts[0])
                    # plot uncertainty
                    if (var_col is not None) and (gui_settings["plot_sigmas"] > 0):
                        fill_upper = ts.df[data_col] \
                            + gui_settings["plot_sigmas"] * (ts.df[var_col] ** 0.5)
                        fill_lower = ts.df[data_col] \
                            - gui_settings["plot_sigmas"] * (ts.df[var_col] ** 0.5)
                        ax.fill_between(ts.time, fill_upper, fill_lower, facecolor='gray',
                                        alpha=gui_settings["plot_sigmas_alpha"], linewidth=0)
                    # plot data
                    ax.plot(ts.time, ts.df[data_col], marker='.', color='k', label="Data"
                            if len(self[station_name].fits[ts_description]) > 0 else None)
                    # overlay models
                    if sum_models and (fit_list is None) and \
                       self[station_name].fits[ts_description].allfits:
                        # get joint model
                        fit = self[station_name].fits[ts_description].allfits
                        # plot all at once
                        if (fit.var_cols is not None) and (gui_settings["plot_sigmas"] > 0):
                            fill_upper = fit.df[fit.data_cols[icol]] \
                                + gui_settings["plot_sigmas"] \
                                * (fit.df[fit.var_cols[icol]] ** 0.5)
                            fill_lower = fit.df[fit.data_cols[icol]] \
                                - gui_settings["plot_sigmas"] \
                                * (fit.df[fit.var_cols[icol]] ** 0.5)
                            ax.fill_between(fit.time, fill_upper, fill_lower,
                                            alpha=gui_settings["plot_sigmas_alpha"],
                                            linewidth=0)
                        ax.plot(fit.time, fit.df[fit.data_cols[icol]], label="Model")
                    elif sum_models:
                        # get model subset
                        fits_to_plot = {model_description: fit for model_description, fit
                                        in self[station_name].fits[ts_description].items()
                                        if ((fit_list is None) or
                                            (model_description in fit_list))}
                        # sum fit
                        sum_fit, sum_var = None, None
                        for model_description, fit in fits_to_plot.items():
                            # initialize
                            if sum_fit is None:
                                sum_fit = fit.df[fit.data_cols[icol]]
                                if (fit.var_cols is not None) \
                                   and (gui_settings["plot_sigmas"] > 0):
                                    sum_var = fit.df[fit.var_cols[icol]]
                            # do all other models
                            else:
                                sum_fit = sum_fit + fit.df[fit.data_cols[icol]]
                                if (fit.var_cols is not None) \
                                   and (gui_settings["plot_sigmas"] > 0):
                                    sum_var = sum_var + fit.df[fit.var_cols[icol]]
                        # plot everything
                        if sum_var is not None:
                            fill_upper = sum_fit + (sum_var ** 0.5) * gui_settings["plot_sigmas"]
                            fill_lower = sum_fit - (sum_var ** 0.5) * gui_settings["plot_sigmas"]
                            ax.fill_between(fit.time, fill_upper, fill_lower,
                                            alpha=gui_settings["plot_sigmas_alpha"],
                                            linewidth=0)
                        if sum_fit is not None:
                            ax.plot(fit.time, sum_fit, label="Model")
                    else:
                        # get model subset
                        fits_to_plot = {model_description: fit for model_description, fit
                                        in self[station_name].fits[ts_description].items()
                                        if ((fit_list is None) or
                                            (model_description in fit_list))}
                        # plot individually
                        for model_description, fit in fits_to_plot.items():
                            if (fit.var_cols is not None) \
                               and (gui_settings["plot_sigmas"] > 0):
                                fill_upper = fit.df[fit.data_cols[icol]] \
                                    + gui_settings["plot_sigmas"] \
                                    * (fit.df[fit.var_cols[icol]] ** 0.5)
                                fill_lower = fit.df[fit.data_cols[icol]] \
                                    - gui_settings["plot_sigmas"] \
                                    * (fit.df[fit.var_cols[icol]] ** 0.5)
                                ax.fill_between(fit.time, fill_upper, fill_lower,
                                                alpha=gui_settings["plot_sigmas_alpha"],
                                                linewidth=0)
                            ax.plot(fit.time, fit.df[fit.data_cols[icol]],
                                    label=model_description)
                    ax.set_ylabel(f"{ts_description}\n{data_col} [{ts.data_unit}]")
                    ax.grid()
                    if len(self[station_name].fits[ts_description]) > 0:
                        ax.legend()
                    ax_ts.append(ax)
                    icomp += 1
                    if scalogram_kw_args is not None:
                        t_left = ts.time[0] if t_left is None else min(ts.time[0], t_left)
                        t_right = ts.time[-1] if t_right is None else max(ts.time[-1], t_right)

            # add vertical lines for events
            if mark_events is not None:
                # subset to the station and sort by time
                mark_subset = mark_events[mark_events["station"] == station_name]
                mark_subset = mark_subset.sort_values(by="time")
                # print all events to be plotted
                print(mark_subset.to_string(), end="\n\n")
                # loop over axes
                ax_t_min, ax_t_max = (mpl.dates.num2date(ax_time).replace(tzinfo=None)
                                      for ax_time in ax_ts[0].get_xlim())
                mark_subset = mark_subset[(mark_subset["time"] >= ax_t_min) &
                                          (mark_subset["time"] <= ax_t_max)]
                if not mark_subset.empty:
                    for ax in ax_ts:
                        for _, row in mark_subset.iterrows():
                            ax.axvline(row["time"], c="0.5")

            # plot possible steps
            if stepdetector:
                nonlocal i_range, last_eqs, last_eqs_lbl
                sub_tmin = step_ranges[i_range][0] - Timedelta(step_padding, "D")
                sub_tmax = step_ranges[i_range][-1] + Timedelta(step_padding, "D")
                station_steps = step_table[(step_table["time"] >= sub_tmin) &
                                           (step_table["time"] <= sub_tmax)]["station"].tolist()
                print(f"\nPeriod {i_range}/{n_ranges}, "
                      f"stations potentially seeing steps are: {station_steps}")
                # get data for this station and time range
                sub_step = step_table[(step_table["station"] == station_name) &
                                      (step_table["time"] >= sub_tmin) &
                                      (step_table["time"] <= sub_tmax)]
                print(f"Station {station_name}: "
                      f"{sub_step.shape[0]} potential steps detected")
                if not sub_step.empty:
                    print(sub_step[["time", "probability"]])
                if maint_table is None:
                    sub_maint = None
                else:
                    sub_maint = maint_table[(maint_table["station"] == station_name) &
                                            (maint_table["time"] >= sub_tmin) &
                                            (maint_table["time"] <= sub_tmax)]
                    if not sub_maint.empty:
                        maintcols = ["time", "code"] if "code" in sub_maint.columns else ["time"]
                        print("Related maintenance:")
                        print(sub_maint[maintcols])
                if catalog is None:
                    sub_cat = None
                else:
                    sub_cat = catalog[(catalog["Date_Origin_Time(JST)"] >= sub_tmin) &
                                      (catalog["Date_Origin_Time(JST)"] <= sub_tmax)]
                    if not sub_cat.empty:
                        sub_lonlats = np.hstack((sub_cat["Longitude(°)"].values.reshape(-1, 1),
                                                sub_cat["Latitude(°)"].values.reshape(-1, 1)))
                        station_lonlat = np.array(self[station_name].location)[[1, 0]]
                        # get distances in km from cartopy
                        distances = geoid.inverse(station_lonlat.reshape(1, 2), sub_lonlats)
                        distances = np.array(distances)[:, 0] / 1e3
                        # get very large eartquakes
                        large_ones = sub_cat["MT_Magnitude(Mw)"].values.squeeze() > 7.5
                        # subset the catalog again
                        sub_cat = sub_cat.iloc[np.any(np.stack([distances <= eqcircle,
                                                                large_ones]), axis=0), :].copy()
                        # print
                        if not sub_cat.empty:
                            # calculate probable earthquake offsets
                            station_disp = np.zeros((sub_cat.shape[0], 3))
                            for i in range(sub_cat.shape[0]):
                                station_disp[i, :] = \
                                    okada_displacement(self[station_name].location,
                                                       sub_cat.iloc[i, :])
                            sub_cat["Okada_Estim_Mag(mm)"] = \
                                np.sqrt(np.sum(station_disp**2, axis=1))
                            print("Related earthquakes:")
                            print(sub_cat[["Date_Origin_Time(JST)", "Latitude(°)",
                                           "Longitude(°)", "MT_Depth(km)",
                                           "MT_Magnitude(Mw)", "Okada_Estim_Mag(mm)"]])

                # plot on all timeseries axes
                xlow = step_ranges[i_range][0] - Timedelta(plot_padding, "D")
                xhigh = step_ranges[i_range][0] + Timedelta(plot_padding, "D")
                if len(ax_ts) > 0:
                    ax_ts[0].set_xlim(xlow, xhigh)
                for ax in ax_ts:
                    ylow, yhigh = np.inf, -np.inf
                    for line in ax.get_lines():
                        xdata = pd.Series(line.get_xdata())
                        tsubset = (pd.Series(xdata) >= xlow) & (pd.Series(xdata) <= xhigh)
                        ydisplayed = line.get_ydata()[tsubset.values]
                        if ydisplayed.size > 0:
                            ylow = min(ylow, np.min(ydisplayed))
                            yhigh = max(yhigh, np.max(ydisplayed))
                    if np.isfinite(ylow) and np.isfinite(yhigh):
                        ymean = (ylow + yhigh) / 2
                        ylow = ymean - (ymean - ylow) * 1.2
                        yhigh = ymean + (yhigh - ymean) * 1.2
                        ax.set_ylim(ylow, yhigh)
                    ax.axvspan(sub_tmin, sub_tmax, ec=None, fc="C6", alpha=0.1, zorder=-100)
                    if catalog is not None:
                        for irow, row in sub_cat.iterrows():
                            ax.axvline(row["Date_Origin_Time(JST)"], c="C3")
                    for irow, row in sub_step.iterrows():
                        ax.axvline(row["time"], ls="--", c="C1")
                    if maint_table is not None:
                        for irow, row in sub_maint.iterrows():
                            ax.axvline(row["time"], ls=":", c="C2")

                # plot earthquakes on map
                if last_eqs is not None:
                    last_eqs.remove()
                    last_eqs = None
                    for lbl in last_eqs_lbl:
                        lbl.remove()
                    last_eqs_lbl = []
                    ax_map.set_extent([ax_map_xmin, ax_map_xmax, ax_map_ymin, ax_map_ymax],
                                      crs=proj_gui)
                if (catalog is not None) and (not sub_cat.empty):
                    ax_map.set_autoscale_on(True)
                    last_eqs = ax_map.scatter(sub_cat["Longitude(°)"], sub_cat["Latitude(°)"],
                                              linestyle='None', marker='*', transform=proj_lla,
                                              facecolor="C3", zorder=100)
                    for irow, row in sub_cat.iterrows():
                        lbl = ax_map.annotate(f"Mw {row['MT_Magnitude(Mw)']}",
                                              (row["Longitude(°)"], row["Latitude(°)"]),
                                              xycoords=proj_lla._as_mpl_transform(ax_map),
                                              textcoords="offset pixels",
                                              xytext=(0, 5), ha="center")
                        last_eqs_lbl.append(lbl)
                fig_map.canvas.draw_idle()

            # finish up
            if len(ax_ts) > 0:  # only call this if there was a timeseries to plot
                ax_ts[0].set_title(station_name)
            fig_ts.canvas.draw_idle()

            # save figure (and map)
            if save and not stepdetector:
                nowtime = pd.Timestamp.now().isoformat()[:19].replace(":", "")
                plotfname = f"{station_name}{fname_add}_{nowtime}.{save_kw_args['format']}"
                fig_ts.savefig(f"ts_{plotfname}", **save_kw_args)
                plt.close(fig_ts)
                if save_map:
                    fig_map.savefig(f"map_{plotfname}", **save_kw_args)
                    plt.close(fig_map)

            # get scalogram
            if scalogram_kw_args is not None:
                try:
                    splset = self[station_name].models[scalo_ts][scalo_model]
                    if isinstance(fig_scalo, plt.Figure):
                        plt.close(fig_scalo)
                    fig_scalo, ax_scalo = splset.make_scalogram(t_left, t_right,
                                                                **scalogram_kw_args)
                    # save figure or show
                    if save and not stepdetector:
                        fig_scalo.savefig(f"scalo_{plotfname}", **save_kw_args)
                        plt.close(fig_scalo)
                    else:
                        fig_scalo.show()
                except KeyError:
                    warn(f"Could not find scalogram model {scalo_model} "
                         f"in timeseries {scalo_ts} for station {station_name}.",
                         category=RuntimeWarning, stacklevel=2)

        click = Click(ax_map, update_timeseries)
        if station is not None:
            update_timeseries(None, station)
        if stepdetector:
            plt.show(block=False)
            while True:
                choice = input("> ")
                if choice == "q":
                    break
                else:
                    new_station = None
                    if choice == "+":
                        i_range = (i_range + 1) % n_ranges
                    elif choice == "-":
                        i_range = (i_range - 1) % n_ranges
                    elif (len(choice) > 1) and (choice[0] == "i"):
                        try:
                            i_range = int(choice[1:]) % n_ranges
                        except ValueError:
                            continue
                    elif (len(choice) > 1) and (choice[0] == "s"):
                        try:
                            new_station = str(choice[1:])
                        except ValueError:
                            continue
                    else:
                        continue
                    update_timeseries(None, new_station)
        elif not save:
            plt.show()
        del click
        plt.close("all")

    def wormplot(self, ts_description, fname=None, fname_animation=None, subset_stations=None,
                 t_min=None, t_max=None, lon_min=None, lon_max=None, lat_min=None, lat_max=None,
                 en_col_names=[0, 1], scale=1e2, interval=10, annotate_stations=True,
                 no_pbar=False, return_figure=False, save_kw_args={"format": "png"},
                 colorbar_kw_args=None, legend_ref_dict=None, gui_kw_args={}):
        """
        Creates an animated worm plot given the data in a timeseries.

        Parameters
        ----------
        ts_description : str, tuple
            Specifies the timeseries to plot. If a string, the name of a timeseries
            directly associated with a station, and if a tuple, specifies the timeseries
            and name of a fitted model for the timeseries.
        fname : str, optional
            If set, save the map to this filename, if not (default), show the map interactively
            (unless ``return_figure=True``).
        fname_animation : str, optional
            If specified, make an animation and save the video to this filename.
        subset_stations : list, optional
            If set, a list of strings that contains the names of stations to be shown.
        t_min : str, pandas.Timestamp, optional
            Start the plot at this time. Defaults to first observation.
        t_max : str, pandas.Timestamp, optional
            End the plot at this time. Defaults to last observation.
        lon_min : float, optional
            Specify the map's minimum longitude (in degrees).
        lon_max : float, optional
            Specify the map's maximum longitude (in degrees).
        lat_min : float, optional
            Specify the map's minimum latitude (in degrees).
        lat_max : float, optional
            Specify the map's maximum latitude (in degrees).
        en_col_names : list, optional
            By default, the first two components of the timeseries will be assumed to be the
            East and North components, respectively, by having the default value ``[0, 1]``
            indicating the desired components as integer indices. Alternatively, this can be
            a list of strings with the two component's column names.
        scale : float, optional
            Specify the conversion scale factor for the displacement timeseries to be
            visible on a map. The final distance for a unit displacement on the map
            will be (timeseries assumed as meters) times (scale). E.g., for a timeseries
            in millimeters and a scale of ``1e2`` (default), one millimeter displacement
            will result in a mapped displacement of 100 meters.
        interval : int, optional
            The number of milliseconds each frame is shown (default: ``10``).
        annotate_stations : bool, float, str, optional
            If ``True`` (default), add the station names to the map.
            If a float or a string, add the station names to the map with the font size set
            as required by :class:`~matplotlib.test.Text`.
        no_pbar : bool, optional
            Suppress the progress bar when creating the animation with ``True``
            (default: ``False``).
        return_figure : bool, optional
            If ``True`` (default: ``False``), return the figure and axis objects instead of
            showing the plot interactively. Only used if ``fname`` is not set.
        save_kw_args : dict, optional
            Additional keyword arguments passed to :meth:`~matplotlib.figure.Figure.savefig`,
            used when ``fname`` is specified.
        colorbar_kw_args : dict, optional
            If ``None`` (default), no colorbar is added to the plot. If a dictionary is passed,
            a colorbar is added, with the dictionary containing additional keyword arguments
            to the :meth:`~matplotlib.figure.Figure.colorbar` method.
        legend_ref_dict : dict, optional
            If ``legend_ref_dict`` is provided and contains all of the following information,
            a reference line will be plotted at the specified location to act as a legend.
            Necessary keys: ``location`` (a tuple containing the longitude and latitude of the
            reference line), ``length`` (the length of the line in meters) and ``label``
            (a label placed below the line).
            The dictionary can optionally include ``rect_args`` and ``rect_kw_args`` entries
            for arguments and parameters that will be passed onto the creation of the
            background :class:`~matplotlib.patches.Rectangle`.
        gui_kw_args : dict, optional
            Override default GUI settings of :attr:`~disstans.config.defaults`.
        """
        # preparations
        gui_settings = defaults["gui"].copy()
        gui_settings.update(gui_kw_args)
        # even if we should show a wmts, we need it to be non-interactive,
        # and therefore is handled later
        prev_wmts_show = gui_settings["wmts_show"]
        gui_settings.update({"wmts_show": False})
        fig_map, ax_map, proj_gui, proj_lla, default_station_edges, \
            stat_points, stat_lats, stat_lons = \
            self._create_map_figure(gui_settings, annotate_stations, subset_stations)
        gui_settings.update({"wmts_show": prev_wmts_show})

        # set map extent
        ax_map_xmin, ax_map_xmax, ax_map_ymin, ax_map_ymax = ax_map.get_extent()
        cur_lon_min, cur_lat_min = \
            proj_lla.transform_point(ax_map_xmin, ax_map_ymin, proj_gui)
        cur_lon_max, cur_lat_max = \
            proj_lla.transform_point(ax_map_xmax, ax_map_ymax, proj_gui)
        map_extent_lonlat = [cur_lon_min, cur_lon_max, cur_lat_min, cur_lat_max]
        if any([lon_min, lon_max, lat_min, lat_max]):
            map_extent_lonlat = [new_val if new_val else cur_val for cur_val, new_val in
                                 zip(map_extent_lonlat, [lon_min, lon_max, lat_min, lat_max])]
            ax_map.set_extent(map_extent_lonlat, crs=proj_lla)

        # collect all the relevant data and subset to time
        network_df = self.export_network_ts(ts_description, subset_stations)
        if isinstance(en_col_names, list) and (len(en_col_names) == 2) and \
           all([isinstance(comp, int) for comp in en_col_names]):
            col_east, col_north = np.array(list(network_df.keys()))[en_col_names].tolist()
        elif (isinstance(en_col_names, list) and (len(en_col_names) == 2) and
              all([isinstance(comp, str) for comp in en_col_names])):
            col_east, col_north = en_col_names
        else:
            raise ValueError("'en_col_names' needs to be a two-element list of integers or "
                             f"strings, got {en_col_names}.")

        # for each station, get displacement timeseries for the east and north components
        # these two dataframes will have a column for each station
        disp_x, disp_y = network_df[col_east], network_df[col_north]
        t_min = t_min if t_min else disp_x.time.min()
        t_max = t_max if t_max else disp_x.time.max()
        disp_x.cut(t_min=t_min, t_max=t_max)
        disp_y.cut(t_min=t_min, t_max=t_max)

        # remove all-nan stations and reference to first observation
        nan_x = disp_x.data.columns[disp_x.data.isna().all()].tolist()
        nan_y = disp_y.data.columns[disp_y.data.isna().all()].tolist()
        nan_stations = set(nan_x + nan_y)
        rel_disp_x = {}
        rel_disp_y = {}
        for name in disp_x.data_cols:
            if name in nan_stations:
                continue
            xy_notnan = np.logical_and(~np.isnan(disp_x[name]), ~np.isnan(disp_y[name]))
            rel_disp_x[name] = disp_x[name] - disp_x[name][xy_notnan][0]
            rel_disp_y[name] = disp_y[name] - disp_y[name][xy_notnan][0]

        # get times and respective colors
        reltimes = ((disp_x.time - disp_x.time[0]) / pd.Timedelta(1, "D")).values
        reltimes = (reltimes - reltimes[0])/(reltimes[-1] - reltimes[0])
        relcolors = scm.batlow(reltimes)
        relcolors[:, 3] = 0
        num_timesteps = relcolors.shape[0]

        # add time span as title
        ax_map.set_title(f"{disp_x.time[0].date()} to {disp_x.time[-1].date()}")

        # get displacement lines
        geoid = cgeod.Geodesic()
        stat_lonlats = {name: np.array(self[name].location)[[1, 0]].reshape(1, 2)
                        for name in rel_disp_x.keys()}
        azi = {name: 90 - np.arctan2(rel_disp_y[name], rel_disp_x[name])*180/np.pi
               for name in rel_disp_x.keys()}
        dist = {name: np.sqrt(rel_disp_x[name]**2 + rel_disp_y[name]**2)
                for name in rel_disp_x.keys()}
        disp_latlon = \
            {name: np.array(geoid.direct(stat_lonlats[name], azi[name],
                                         dist[name]*scale)[:, :2])
             for name in rel_disp_x.keys()}
        lines = \
            {name: ax_map.scatter(disp_latlon[name][:, 0], disp_latlon[name][:, 1],
                                  facecolor=relcolors, edgecolor="none", zorder=10,
                                  transform=proj_lla, rasterized=True)
             for name in rel_disp_x.keys()}

        # add static background image and gridlines
        if gui_settings["wmts_show"]:
            try:
                src = WMTSRasterSource(wmts=gui_settings["wmts_server"],
                                       layer_name=gui_settings["wmts_layer"])
                bbox = ax_map.get_window_extent().transformed(fig_map.dpi_scale_trans.inverted())
                width_height_px = round(bbox.width*fig_map.dpi), round(bbox.height*fig_map.dpi)
                imgs = src.fetch_raster(proj_gui, ax_map.get_extent(), width_height_px)
                for img in imgs:
                    ax_map.imshow(img.image, origin="upper", zorder=-1,
                                  extent=img.extent, alpha=gui_settings["wmts_alpha"])
            except Exception as exc:
                print(exc)
        ax_map.gridlines(draw_labels=True)

        # add a colorbar
        if isinstance(colorbar_kw_args, dict):
            colorbar_kw_args.update({"orientation": "horizontal", "ticks": [0, 1]})
            cbar = fig_map.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=1),
                                                   cmap=scm.batlow),
                                    ax=ax_map, **colorbar_kw_args)
            cbar.ax.set_xticklabels([str(disp_x.time[0].date()), str(disp_x.time[-1].date())])

        # add the legend
        if isinstance(legend_ref_dict, dict):
            if not all([k in legend_ref_dict for k in ["location", "length", "label"]]):
                print("'legend_ref_dict' does not contain all valid keys to create a legend.")
            else:
                ref_disp = np.linspace(0, legend_ref_dict["length"], num=num_timesteps)
                ref_lonlat = np.array(legend_ref_dict["location"]).reshape(1, 2)
                ref_loc = np.array(geoid.direct(ref_lonlat, 90, ref_disp * scale)[:, :2])
                ax_map.scatter(ref_loc[:, 0], ref_loc[:, 1],
                               facecolor=relcolors[:, :3], edgecolor="none",
                               zorder=10, transform=proj_lla, rasterized=True)
                ax_map.annotate(legend_ref_dict["label"], ref_loc[num_timesteps//2, :],
                                xytext=(0, -6), textcoords="offset points",
                                ha="center", va="top", zorder=10, transform=proj_lla,
                                xycoords=proj_lla._as_mpl_transform(ax_map))
                if ("rect_args" in legend_ref_dict) and ("rect_kw_args" in legend_ref_dict):
                    ax_map.add_patch(Rectangle(*legend_ref_dict["rect_args"],
                                               **legend_ref_dict["rect_kw_args"],
                                               transform=proj_lla, zorder=9))

        # only animate if respective output filename is set
        if fname_animation:

            # define animation functions
            def init():
                relcolors[:, 3] = 0
                ax_map.set_title(f"{disp_x.time[0].date()} to {disp_x.time[0].date()}")

            def update(i):
                relcolors[:i+1, 3] = 1
                for name in rel_disp_x.keys():
                    lines[name].set_facecolors(relcolors)
                ax_map.set_title(f"{disp_x.time[0].date()} to {disp_x.time[i].date()}")

            # make actual animation
            try:
                pbar = tqdm(desc="Rendering animation", unit="frame",
                            total=num_timesteps, ascii=True, disable=no_pbar)
                ani = FuncAnimation(fig_map, update, frames=num_timesteps,
                                    init_func=init, interval=interval, blit=False)
                ani.save(fname_animation, progress_callback=lambda i, n: pbar.update())
            finally:
                pbar.close()

        # if we didn't animate, we now need to make the entire worms visible
        relcolors[:, 3] = 1
        for name in rel_disp_x.keys():
            lines[name].set_facecolors(relcolors)

        # finish up
        if fname:
            fig_map.savefig(f"{fname}.{save_kw_args['format']}", **save_kw_args)
            if return_figure:
                warn("'fname' was specified, so 'return_figure=True' is ignored.")
        else:
            if return_figure:
                return fig_map, ax_map
            else:
                plt.show()

    def ampphaseplot(self, ts_description, mdl_description, components=None, phase=True,
                     fname=None, subset_stations=None, lon_min=None, lon_max=None,
                     lat_min=None, lat_max=None, scale=1, annotate_stations=True,
                     legend_refs=None, legend_labels=None, return_figure=False,
                     save_kw_args={"format": "png"}, colorbar_kw_args=None, gui_kw_args={}):
        """
        Plot the amplitude and phase of a :class:`~disstans.models.Sinusoid`
        model on a network map.

        Parameters
        ----------
        ts_description : str
            The name of the timeseries to be used.
        mdl_description : str
            The name of the model to be plotted.
        components : int, list, optional
            Specify the components to be used.
            By default (``None``), all available components are combined.
            If an integer, only a single component is used (in which case the phase
            can be used to color the marker).
            If a list, the indices of the list specify the components to be
            combined.
        phase : bool, optional
            (Only used if a single component is selected.)
            If ``True`` (default), use the phase of the sinusoid to color the
            station markers.
        fname : str, optional
            If set, save the map to this filename, if not (default), show the map interactively
            (unless ``return_figure=True``).
        subset_stations : list, optional
            If set, a list of strings that contains the names of stations to be shown.
        lon_min : float, optional
            Specify the map's minimum longitude (in degrees).
        lon_max : float, optional
            Specify the map's maximum longitude (in degrees).
        lat_min : float, optional
            Specify the map's minimum latitude (in degrees).
        lat_max : float, optional
            Specify the map's maximum latitude (in degrees).
        scale : float, optional
            Scale factor for the markers. Defaults to ``1``.
        annotate_stations : bool, float, str, optional
            If ``True`` (default), add the station names to the map.
            If a float or a string, add the station names to the map with the font size set
            as required by :class:`~matplotlib.test.Text`.
        legend_refs : list, optional
            If set, a list of amplitudes that will be used to generate legend entries.
        legend_labels : list, optional
            If set, a list of labels that will be used for ``legend_refs``.
        return_figure : bool, optional
            If ``True`` (default: ``False``), return the figure and axis objects instead of
            showing the plot interactively. Only used if ``fname`` is not set.
        save_kw_args : dict, optional
            Additional keyword arguments passed to :meth:`~matplotlib.figure.Figure.savefig`,
            used when ``fname`` is specified.
        colorbar_kw_args : dict, optional
            (Only used if a phases are plotted.)
            If ``None`` (default), no colorbar is added to the plot. If a dictionary is passed,
            a colorbar is added, with the dictionary containing additional keyword arguments
            to the :meth:`~matplotlib.figure.Figure.colorbar` method.
        gui_kw_args : dict, optional
            Override default GUI settings of :attr:`~disstans.config.defaults`.
        """
        # preparations
        gui_settings = defaults["gui"].copy()
        gui_settings.update(gui_kw_args)
        fig_map, ax_map, proj_gui, proj_lla, default_station_edges, \
            stat_points, stat_lats, stat_lons = \
            self._create_map_figure(gui_settings, annotate_stations, subset_stations)

        # set map extent
        ax_map_xmin, ax_map_xmax, ax_map_ymin, ax_map_ymax = ax_map.get_extent()
        cur_lon_min, cur_lat_min = \
            proj_lla.transform_point(ax_map_xmin, ax_map_ymin, proj_gui)
        cur_lon_max, cur_lat_max = \
            proj_lla.transform_point(ax_map_xmax, ax_map_ymax, proj_gui)
        map_extent_lonlat = [cur_lon_min, cur_lon_max, cur_lat_min, cur_lat_max]
        if any([lon_min, lon_max, lat_min, lat_max]):
            map_extent_lonlat = [new_val if new_val else cur_val for cur_val, new_val in
                                 zip(map_extent_lonlat, [lon_min, lon_max, lat_min, lat_max])]
            ax_map.set_extent(map_extent_lonlat, crs=proj_lla)

        # collect all the relevant data
        amps = []
        if phase:
            phases = []
        if subset_stations is None:
            subset_stations = self.station_names
        for stat_name in subset_stations:
            mdl = self[stat_name].models[ts_description][mdl_description]
            if components is None:
                components = list(range(mdl.par.shape[1]))
            elif isinstance(components, int):
                components = [components]
            try:
                amps.append(mdl.amplitude[components])
            except AttributeError as e:
                if not isinstance(mdl, disstans_models.Sinusoid):
                    raise RuntimeError("'mdl_description' refers to a model object that "
                                       "does not have an 'amplitude' attribute."
                                       ).with_traceback(e.__traceback__) from e
                else:
                    raise e
            if len(components) > 1:
                amps.append(np.sqrt(np.sum(amps[stat_name]**2)))
            elif phase:
                try:
                    phases.append(mdl.phase[components[0]])
                except AttributeError as e:
                    if not isinstance(mdl, disstans_models.Sinusoid):
                        raise RuntimeError("'mdl_description' refers to a model object that "
                                           "does not have a 'phase' attribute."
                                           ).with_traceback(e.__traceback__) from e
                    else:
                        raise e

        # update the station marker sizes
        stat_points.set_zorder(1)
        stat_points.set_sizes(20*scale*np.array(amps).ravel()**2)

        # update the marker colors
        if phase and (len(phases) > 0):
            stat_points.set_facecolor(scm.romaO((np.array(phases) % (2*np.pi)) / (2*np.pi)))

        # add gridlines
        ax_map.gridlines(draw_labels=True)

        # add a legend
        if isinstance(legend_refs, list):
            if legend_labels is None:
                legend_labels = legend_refs
            num_legends = len(legend_refs)
            ref_lons = [(map_extent_lonlat[0] + map_extent_lonlat[1]) / 2] * num_legends
            ref_lats = [(map_extent_lonlat[2] + map_extent_lonlat[3]) / 2] * num_legends
            ref_scatter = ax_map.scatter(ref_lons, ref_lats,
                                         s=20*scale*np.array(legend_refs)**2,
                                         visible=False, marker='o', transform=proj_lla)
            ref_handles = ref_scatter.legend_elements(
                prop="sizes", num=None,
                markerfacecolor="none", markeredgecolor="k", visible=True,
                func=lambda s: np.sqrt(s / (20*scale)))[0]
            ax_map.legend(ref_handles, legend_labels)

        # add a colorbar
        if isinstance(colorbar_kw_args, dict):
            fig_map.colorbar(ScalarMappable(norm=Normalize(vmin=1, vmax=13-1e-10),
                                            cmap=scm.romaO),
                             ax=ax_map, **colorbar_kw_args)

        # finish up
        if fname:
            fig_map.savefig(f"{fname}.{save_kw_args['format']}", **save_kw_args)
            if return_figure:
                warn("'fname' was specified, so 'return_figure=True' is ignored.")
        else:
            if return_figure:
                return fig_map, ax_map
            else:
                plt.show()
