"""
This module contains the :class:`~disstans.network.Network` class, which is the
highest-level container object in DISSTANS.
"""

import numpy as np
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
from cartopy.io.ogc_clients import WMTSRasterSource
from cmcrameri import cm as scm

from . import timeseries as disstans_ts
from . import models as disstans_models
from . import solvers as disstans_solvers
from . import processing as disstans_processing
from .models import ALLFITS
from .config import defaults
from .timeseries import Timeseries
from .station import Station
from .processing import common_mode
from .tools import parallelize, Timedelta, Click
from .earthquakes import okada_displacement


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
        :class:`~disstans.models.Sinusoidal` model::

            models = {"Annual": {"type": "Sinusoidal",
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
            warn(f"Overwriting station '{name}'.", category=RuntimeWarning)
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
            warn(f"Cannot find station '{name}', couldn't delete.", category=RuntimeWarning)
        else:
            del self.stations[name]

    @classmethod
    def from_json(cls, path, add_default_local_models=True,
                  station_kw_args={}, timeseries_kw_args={}):
        """
        Create a :class:`~disstans.network.Network` instance from a JSON configuration file.

        Parameters
        ----------
        path : str
            Path of input JSON file.
        add_default_local_models : bool, optional
            If false, skip the adding of any default local model found in a station.
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
                                              desc="Building Network", unit="station"):
            if "location" in station_cfg:
                station_loc = station_cfg["location"]
            elif station_name in net._network_locations:
                station_loc = net._network_locations[station_name]
            else:
                warn(f"Skipped station '{station_name}' "
                     "because location information is missing.")
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
            return_solutions=False, progress_desc=None, **kw_args):
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
                                     ascii=True, unit="station")):
            # print warning if the solver didn't converge
            if not sol.converged:
                warn(f"Fitting did not converge for timeseries {ts_description} "
                     f"at {station_names[i]}", category=RuntimeWarning)
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

    def evaluate(self, ts_description, timevector=None, output_description=None,
                 progress_desc=None):
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

        Example
        -------
        If ``net`` is a :class:`~Network` instance, ``'mydata'`` is the timeseries
        to evaluate the models for, then the following two are equivalent::

            # long version, not parallelized, and not creating a new independent timeseries
            for station in net:
                station_ts = station.timeseries['mydata']
                fit_all = station_models.evaluate(station_ts)
                station.add_fit('mydata', ALLFITS, fit_all)
                for (model_description, model) in station_models.items():
                    fit = model.evaluate(station_ts)
                    station.add_fit('mydata', model_description, fit)
            # short version, automatically parallelizes according to disstans.defaults,
            # and creating a new timeseries
            net.evaluate('mydata', output_description='evaluated')

        See Also
        --------
        fit : Fit models at all stations.
        :attr:`~disstans.config.defaults` : Dictionary of settings, including parallelization.
        :attr:`~disstans.models.ALLFITS` : Key used to denote the joint fit using all models.
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
                                                  ascii=True, unit="station")):
            stat = self[station_names[i]]
            # add overall model
            ts = stat.add_fit(ts_description, ALLFITS, sumfit,
                              return_ts=(output_description is not None))
            # if desired, add as timeseries
            if output_description is not None:
                stat.add_timeseries(output_description, ts)
            # add individual fits
            for model_description, fit in mdlfit.items():
                stat.add_fit(ts_description, model_description, fit)

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
                   residual_description=None, progress_desc=None, **kw_args):
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
        **kw_args : dict
            Additional keyword arguments that are passed on to the solver function,
            see :meth:`~fit`.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, ``'mydata'`` is the timeseries to fit
        and evaluate the models for, ``mysolver`` is the solver to use, ``'myfit'`` should
        contain the model fit, and the residuals should be put into the timeseries
        ``'myres'``, then the following two are equivalent::

            # long version, not parallelized, defaulting to all models
            for station in net:
                station_ts = station.timeseries['mydata']
                station_models = station.models['mydata']
                model_params_var = mysolver(station_ts, station_models, **kw_args)
                for model_description, (params, covs) in model_params_var.items():
                    station_models[model_description].read_parameters(params, covs)
                    fit = station_models[model_description].evaluate(station_ts)
                    station.add_fit('mydata', model_description, fit)
            net.fit('mydata', solver=mysolver, **kw_args)
            net.evaluate('mydata', output_description='myfit')
            net.math('myres', 'mydata', '-', 'myfit')
            # shot version, combining everything into one call, using parallelization,
            # offering easy access to subsetting models, reusing previous fits, etc.
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
                                   progress_desc=progress_desc[0], **kw_args)
        # evaluate
        self.evaluate(ts_description=ts_description, timevector=timevector,
                      output_description=output_description, progress_desc=progress_desc[0])
        # residual
        if residual_description:
            self.math(residual_description, ts_description, "-", output_description)
        # return either None or the results dictionary of self.fit():
        return possible_output

    def call_func_ts_return(self, func, ts_in, ts_out=None, **kw_args):
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
                                        ascii=True, unit="station")):
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
            net.call_netwide_func('common_mode', ts_in='input, ts_out='output', **kw_args)

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

    def call_func_no_return(self, func, **kw_args):
        """
        A convenience wrapper that for each station in the network, calls a given function on
        each station. Also provides a progress bar.

        Parameters
        ----------
        func : str, function
            Function to use. If provided with a string, will check for that function in
            :mod:`~disstans.processing`, otherwise the function will be assumed to adhere to the
            same input format than the included ones.
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
                                  ascii=True, unit="station"):
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
        stat_points = ax_map.scatter(stat_lons, stat_lats, s=100, facecolor='C0',
                                     linestyle='None', marker='.', transform=proj_lla,
                                     edgecolor=default_station_edges, zorder=1000)
        # add labels
        if annotate_stations:
            for sname, slon, slat in zip(stat_names, stat_lons, stat_lats):
                ax_map.annotate(sname, (slon, slat),
                                xycoords=proj_lla._as_mpl_transform(ax_map),
                                annotation_clip=True, textcoords="offset pixels",
                                xytext=(0, 5), ha="center")
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
                      save_kw_args={"format": "png"}, gui_kw_args={}, **cme_kw_args):
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
        annotate_stations : bool, optional
            If ``True`` (default), add the station names to the map.
        save : bool, optional
            If ``True``, save the map and timeseries plots to the current folder.
            Defaults to ``False``.
        save_kw_args : dict, optional
            Additional keyword arguments passed to :meth:`~matplotlib.figure.Figure.savefig`,
            used when ``save=True``.
        gui_kw_args : dict, optional
            Override default GUI settings of :attr:`~disstans.config.defaults`.
        **cme_kw_args : dict
            Additional keyword arguments passed to :func:`~disstans.processing.common_mode`.

        See Also
        --------
        disstans.processing.common_mode : CME calculation function.
        """
        # get common mode and make sure to return spatial and temporal models
        gui_settings = defaults["gui"].copy()
        gui_settings.update(gui_kw_args)
        net_in = self.export_network_ts(ts_in)
        comps = list(net_in.keys())
        ndim = len(comps)
        ndim_max2 = min(ndim, 2)
        assert ndim > 0, f"No components found in '{ts_in}'."
        cme_kw_args.update({'plot': True})
        model, temp_spat = common_mode(net_in, **cme_kw_args)
        temporal, spatial = {}, {}
        for comp in model:
            temporal[comp] = temp_spat[comp][0]
            spatial[comp] = temp_spat[comp][1]
        # assign model to network
        if ts_out is not None:
            self.import_network_ts(ts_out, model)
        # extract spatial components
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
        annotate_stations : bool, optional
            If ``True`` (default), add the station names to the map.
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
                    if sum_models and (ALLFITS in self[station_name].fits[ts_description]):
                        fit = self[station_name].fits[ts_description][ALLFITS]
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
                    else:
                        # get model subset
                        fits_to_plot = {model_description: fit for model_description, fit
                                        in self[station_name].fits[ts_description].items()
                                        if (((fit_list is None)
                                             or (model_description in fit_list))
                                            and model_description != ALLFITS)}
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
                         category=RuntimeWarning)

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
                 save_kw_args={"format": "png"}, colorbar_kw_args=None, gui_kw_args={}):
        """
        Creates an animated worm plot given the data in a timeseries.

        Parameters
        ----------
        ts_description : str, tuple
            Specifies the timeseries to plot. If a string, the name of a timeseries
            directly associated with a station, and if a tuple, specifies the timeseries
            and name of a fitted model for the timeseries.
        fname : str, optional
            If set, save the map to this filename, if not (default), show the map interactively.
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
        annotate_stations : bool, optional
            If ``True`` (default), add the station names to the map.
        save_kw_args : dict, optional
            Additional keyword arguments passed to :meth:`~matplotlib.figure.Figure.savefig`,
            used when ``fname`` is specified.
        colorbar_kw_args : dict, optional
            If ``None`` (default), no colorbar is added to the plot. If a dictionary is passed,
            a colorbar is added, with the dictionary containing additional keyword arguments
            to the :meth:`~matplotlib.figure.Figure.colorbar` method.
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
            rel_disp_x[name] = disp_x[name] - disp_x[name][~np.isnan(disp_x[name])][0]
            rel_disp_y[name] = disp_y[name] - disp_y[name][~np.isnan(disp_y[name])][0]

        # get times and respective colors
        reltimes = ((disp_x.time - disp_x.time[0]) / pd.Timedelta(1, "D")).values
        reltimes = (reltimes - reltimes[0])/(reltimes[-1] - reltimes[0])
        relcolors = scm.batlow(reltimes)
        relcolors[:, 3] = 0

        # add time span as title
        ax_map.set_title(f"{disp_x.time[0].date()} to {disp_x.time[-1].date()}")

        # get direction of displacement
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
                            total=relcolors.shape[0], ascii=True)
                ani = FuncAnimation(fig_map, update, frames=relcolors.shape[0],
                                    init_func=init, interval=interval, blit=False)
                ani.save(fname_animation, progress_callback=lambda i, n: pbar.update())
            finally:
                pbar.close()

        # save figure at the last stage if output filename is set, otherwise show
        relcolors[:, 3] = 1
        for name in rel_disp_x.keys():
            lines[name].set_facecolors(relcolors)
        if fname:
            fig_map.savefig(f"{fname}.{save_kw_args['format']}", **save_kw_args)
        else:
            plt.show()

        # close figure
        plt.close(fig_map)
