"""
This module contains the :class:`~geonat.network.Network` class, which is the
highest-level container object in GeoNAT.
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

from . import timeseries as geonat_ts
from . import models as geonat_models
from . import solvers as geonat_solvers
from . import processing as geonat_processing
from . import scm
from .config import defaults
from .timeseries import Timeseries
from .station import Station
from .processing import common_mode
from .tools import parallelize, Timedelta, Click
from .earthquakes import okada_displacement


class Network():
    r"""
    Main class of GeoNAT. Contains information about the network, defines defaults,
    contains global models, and most improtantly, contains a dictionary of all
    stations in the network.

    Parameters
    ----------
    name : str
        Name of the network.
    default_location_path : str, optional
        If station locations aren't given directly, check for a file with this path for the
        station's location.
        It needs to be a four-column, space-separated text file with the entries
        ``name latitude[°] longitude[°] altitude[m]`` (without headers).
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
        :class:`~geonat.models.Sinusoidal` model::

            models = {"Annual": {"type": "Sinusoidal",
                                 "kw_args": {"period": 365.25,
                                             "t_reference": "2000-01-01"}}}
            net.update_default_local_models(models)
        """
        self.update_default_local_models(default_local_models)
        self.stations = {}
        """
        Dictionary of network stations, where the keys are their string names
        and the values are their :class:`~geonat.station.Station` objects. """
        self.global_models = {}
        """
        Dictionary of network-wide models, where the keys are string descriptions
        and the values are their :class:`~geonat.models.Model` objects.
        """
        # try to preload the location data
        # it's a private attribute because there's no guarantee this will be kept
        # up to date over the lifetime of the Network instance (yet)
        if self.default_location_path is None:
            self._network_locations = []
        else:
            with open(self.default_location_path, mode='r') as locfile:
                loclines = [line.strip() for line in locfile.readlines()]
            self._network_locations = {line.split()[0]:
                                       [float(lla) for lla in line.split()[1:]]
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

    def __repr__(self):
        """
        Special function that returns a readable summary of the network.
        Accessed, for example, by Python's ``print()`` built-in function.

        Returns
        -------
        info : str
            Network summary.
        """
        info = f"Network {self.name}\n" + \
               f"Stations:\n{[key for key in self.stations]}\n" + \
               f"Global Models:\n{[key for key in self.global_models]}"
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
        geonat.station.Station
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

    def export_network_ts(self, ts_description):
        """
        Collects a specific timeseries from all stations and returns them in a dictionary
        of network-wide :class:`~geonat.timeseries.Timeseries` objects.

        Parameters
        ----------
        ts_description : str
            :class:`~geonat.timeseries.Timeseries` description that will be collected
            from all stations in the network.

        Returns
        -------
        network_df : dict
            Dictionary with the data components of the ``ts_description`` timeseries
            as keys and :class:`~geonat.timeseries.Timeseries` objects as values
            (which will have in turn the station names as column names).

        See Also
        --------
        import_network_ts : Inverse function.
        """
        df_dict = {}
        for name, station in self.stations.items():
            if ts_description in station.timeseries:
                if df_dict == {}:
                    network_data_cols = station[ts_description].data_cols
                    network_src = station[ts_description].src
                    network_unit = station[ts_description].data_unit
                df_dict.update({name: station[ts_description].data.astype(pd.SparseDtype())})
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
        Distributes a dictionary of network-wide :class:`~geonat.timeseries.Timeseries`
        objects onto the network stations.

        Parameters
        ----------
        ts_description : str
            :class:`~geonat.timeseries.Timeseries` description where the data will be placed.
        dict_of_timeseries : dict
            Dictionary with the data components of the ``ts_description`` timeseries
            as keys and :class:`~geonat.timeseries.Timeseries` objects as values
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
        station : geonat.station.Station
            Station object to add.

        See Also
        --------
        __setitem__ : Shorthand notation wrapper.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, ``name`` the name of a new station,
        and ``station`` a :class:`~geonat.station.Station` instance, then the following
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

    def add_global_model(self, mdl_description, model):
        """
        Add a global model to the network.

        Parameters
        ----------
        mdl_description : str
            Description of the model.
        model : geonat.models.Model
            Model object to add.

        Warning
        -------
        Global models have not been implemented yet.
        """
        if not isinstance(mdl_description, str):
            raise TypeError("Cannot add new global model: 'mdl_description' is not a string.")
        if mdl_description in self.global_models:
            warn(f"Overwriting global model '{mdl_description}'.", category=RuntimeWarning)
        self.global_models[mdl_description] = model

    def remove_global_model(self, mdl_description):
        """
        Remove a global model from the network.

        Parameters
        ----------
        mdl_description : str
            Description of the model.

        Warning
        -------
        Global models have not been implemented yet.
        """
        if mdl_description not in self.global_models:
            warn(f"Cannot find global model '{mdl_description}', couldn't delete.",
                 category=RuntimeWarning)
        else:
            del self.global_models[mdl_description]

    @classmethod
    def from_json(cls, path, add_default_local_models=True,
                  station_kw_args={}, timeseries_kw_args={}):
        """
        Create a :class:`~geonat.network.Network` instance from a JSON configuration file.

        Parameters
        ----------
        path : str
            Path of input JSON file.
        add_default_local_models : bool, optional
            If false, skip the adding of any default local model found in a station.
        station_kw_args : dict, optional
            Additional keyword arguments passed on to the
            :class:`~geonat.station.Station` constructor.
        timeseries_kw_args : dict, optional
            Additional keyword arguments passed on to the
            :class:`~geonat.timeseries.Timeseries` constructor.

        Returns
        -------
        net : geonat.network.Network
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
                ts_class = getattr(geonat_ts, ts_cfg["type"])
                ts = ts_class(**ts_cfg["kw_args"], **timeseries_kw_args)
                station.add_timeseries(ts_description=ts_description, timeseries=ts)
                # add default local models to station
                if add_default_local_models:
                    for model_description, model_cfg in net.default_local_models.items():
                        local_copy = deepcopy(model_cfg)
                        mdl_class = getattr(geonat_models, local_copy["type"])
                        mdl = mdl_class(**local_copy["kw_args"])
                        station.add_local_model(ts_description=ts_description,
                                                model_description=model_description,
                                                model=mdl)
            # add specific local models to station and timeseries
            for ts_description, ts_model_dict in station_cfg["models"].items():
                if ts_description in station.timeseries:
                    for model_description, model_cfg in ts_model_dict.items():
                        mdl_class = getattr(geonat_models, model_cfg["type"])
                        mdl = mdl_class(**model_cfg["kw_args"])
                        station.add_local_model(ts_description=ts_description,
                                                model_description=model_description,
                                                model=mdl)
                else:
                    station.unused_models.update({ts_description: ts_model_dict})
            # add to network
            net.add_station(name=station_name, station=station)
        # add global models
        for model_description, model_cfg in net_arch["global_models"].items():
            mdl_class = getattr(geonat_models, model_cfg["type"])
            mdl = mdl_class(**model_cfg["kw_args"])
            net.add_global_model(mdl_description=model_description, model=mdl)
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
        from_json : Create a :class:`~geonat.network.Network` instance
            from a JSON configuration file.
        """
        # create new dictionary
        net_arch = {"name": self.name,
                    "locations": self.default_location_path,
                    "stations": {},
                    "default_local_models": self.default_local_models,
                    "global_models": {}}
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
        # add global model representations
        for model_description, mdl in self.global_models.items():
            net_arch["global_models"].update({model_description: mdl.get_arch()})
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
            to create each model object.
        """
        assert all([isinstance(mdl_name, str) for mdl_name in models.keys()]), \
            f"Model names need to be strings, got {models.keys()}."
        assert all([isinstance(mdl_cfg, dict) for mdl_cfg in models.values()]), \
            f"Model configurations need to be dictionaries, got {models.keys()}."
        for mdl_name, mdl_config in models.items():
            assert all([key in mdl_config.keys() for key in ["type", "kw_args"]]), \
                f"The configuration dictionary for '{mdl_name}' needs to contain " \
                f"the keys 'type' and 'kw_args', got {mdl_config.keys()}."
            assert isinstance(mdl_config["type"], str), \
                f"'type' in configuration dictionary for '{mdl_name}' needs to be " \
                f"a string, got {mdl_config['type']}."
            assert isinstance(mdl_config["kw_args"], dict), \
                f"'kw_args' in configuration dictionary for '{mdl_name}' needs to be " \
                f"a dictionary, got {mdl_config['kw_args']}."
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
                mdl = getattr(geonat_models, local_copy["type"])(**local_copy["kw_args"])
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
                for model_description, model_cfg in local_models_subset.items():
                    local_copy = deepcopy(model_cfg)
                    mdl_class = getattr(geonat_models, local_copy["type"])
                    mdl = mdl_class(**local_copy["kw_args"])
                    station.add_local_model(ts_description=target_ts,
                                            model_description=model_description,
                                            model=mdl)

    def load_maintenance_dict(self, maint_dict, ts_description, model_description,
                              only_active=True):
        """
        Convenience wrapper to add :class:`~geonat.models.Step` models to the stations
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
        :class:`~geonat.models.Step` : Model used to add steps.
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
                                              geonat_models.Step(localsteps))

    def common_mapping(self, ts_description):
        """
        Return the mapping matrix for all models that are common to a given timeseries
        over all stations, for all the timestamps present in the timeseries.

        Parameters
        ----------
        ts_description : str
            Timeseries to check models for.

        Returns
        -------
        common_mapping : dict
            A dictionary that for every model description contains a
            :class:`~pandas.DataFrame` with a timestamp index and columns for
            each model parameter.
        """
        times = None
        models = None
        for station in self:
            if ts_description in station.timeseries:
                if times is None:
                    times = station[ts_description].time
                else:
                    times = times.union(station[ts_description].time)
                if models is None:
                    models = station.models[ts_description]
                else:
                    models = {mdl_desc: mdl for mdl_desc, mdl in models.items()
                              if (mdl_desc in station.models[ts_description]
                                  and mdl == station.models[ts_description][mdl_desc])}
        common_mapping = {mdl_desc: pd.DataFrame(data=mdl.get_mapping(times).A, index=times)
                          for mdl_desc, mdl in models.items()}
        return common_mapping

    def fit(self, ts_description, model_list=None, solver='linear_regression',
            cached_mapping=True, **kw_args):
        """
        Fit the models (or a subset thereof) for a specific timeseries at all stations.
        Also provides a progress bar.
        Will automatically use multiprocessing if parallelization has been enabled in
        the configuration (defaults to parallelization if possible).

        Parameters
        ----------
        ts_description : str
            Description of the timeseries to fit.
        model_list : list, optional
            List of strings containing the model names of the subset of the models
            to fit. Defaults to all models.
        solver : str, function, optional
            Solver function to use. If given a string, will look for a solver with that
            name in :mod:`~geonat.solvers`, otherwise will use the passed function as a
            solver (which needs to adhere to the same input/output structure as the
            included solver functions). Defaults to standard linear least squares.
        cache_mapping : bool, optional
            If ``True``, the mapping matrices for all models that are common to all stations
            is calculated and inserted into ``kw_args`` as ``'cached_mapping'``, such that
            redundant work is reduced.
            This is mostly impactful for serial processing, but does not have a noticeable
            drawback for parallel processing, which is why it defaults to ``True``.
        **kw_args : dict
            Additional keyword arguments that are passed onto the solver function.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, ``'mydata'`` is the timeseries to fit,
        and ``mysolver`` is the solver to use, then the following two are equivalent::

            # long version, not parallelized, defaulting to all models
            for station in net:
                station_ts = station.timeseries['mydata']
                station_models = station.models['mydata']
                # a subset list of models would need to be collected here
                model_params_var = mysolver(station_ts, station_models, **kw_args)
                for model_description, (params, covs) in model_params_var.items():
                    station.models['mydata'][model_description].read_parameters(params, covs)
            # short version, automatically parallelizes according to geonat.defaults
            net.fit('mydata', solver=mysolver, **kw_args)
            # also allows for subsetting models and skipping the import of geonat.solvers
            net.fit('mydata', model_list=['onlythisone'],
                    solver='lasso_regression', **kw_args)

        See Also
        --------
        evaluate : Evaluate the fitted models at all stations.
        :attr:`~geonat.config.defaults` : Dictionary of settings, including parallelization.
        geonat.tools.parallelize : Automatically execute a function in parallel or serial.
        common_mapping : Used when ``cache_mapping`` is ``True``.
        """
        assert isinstance(ts_description, str), \
            f"'ts_description' must be string, got {type(ts_description)}."
        if isinstance(solver, str):
            solver = getattr(geonat_solvers, solver)
        assert callable(solver), f"'solver' must be a callable function, got {type(solver)}."
        # get common mapping matrices
        if cached_mapping:
            kw_args["cached_mapping"] = self.common_mapping(ts_description)
        iterable_inputs = ((station.timeseries[ts_description],
                            station.models[ts_description] if model_list is None
                            else {m: station.models[ts_description][m] for m in model_list
                                  if m in station.models[ts_description]},
                            solver, kw_args) for station in self)
        station_names = list(self.stations.keys())
        for i, result in enumerate(tqdm(parallelize(self._fit_single_station, iterable_inputs),
                                        desc="Fitting station models",
                                        total=len(self.stations),
                                        ascii=True, unit="station")):
            stat = self[station_names[i]]
            # print warning if the solver didn't converge
            if any([np.isnan(params_var[0]).sum() > 0 for params_var in result.values()]):
                warn(f"Fitting did not converge for timeseries {ts_description} "
                     f"at {station_names[i]}", category=RuntimeWarning)
            for model_description, (params, var) in result.items():
                stat.models[ts_description][model_description].read_parameters(params, var)

    @staticmethod
    def _fit_single_station(parameter_tuple):
        station_time, station_models, solver, kw_args = parameter_tuple
        model_params_var = solver(station_time, station_models, **kw_args)
        return model_params_var

    def evaluate(self, ts_description, model_list=None, timevector=None,
                 output_description=None, reuse=False):
        """
        Evaluate a timeseries' models (or a subset thereof) at all stations and add them
        as a fit to the timeseries. Can optionally add the aggregate model as an independent
        timeseries to the station as well.
        Also provides a progress bar.
        Will automatically use multiprocessing if parallelization has been enabled in
        the configuration (defaults to parallelization if possible).

        Parameters
        ----------
        ts_description : str
            Description of the timeseries to evaluate.
        model_list : list, optional
            List of strings containing the model names of the subset of the models
            to be evaluated. Defaults to all models.
        timevector : pandas.Series, pandas.DatetimeIndex, optional
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` of when to evaluate the model.
            Defaults to the timestamps of the timeseries itself.
        output_description : str, optional
            If provided, add the sum of the evaluated models as a new timeseries
            to each station with the provided description (instead of only adding
            each individual model as a fit *to* the timeseries).
        reuse : bool, optional
            If ``timevector`` is ``None`` and ``output_description`` is set, this flag can
            be used to skip the actual evaluation of the models if they have already been
            added as fits, and instead use those fitted timeseries instead.

        Example
        -------
        If ``net`` is a :class:`~Network` instance, ``'mydata'`` is the timeseries
        to evaluate the models for, and ``mysolver`` is the solver to use, then the
        following two are equivalent::

            # long version, not parallelized, defaulting to all models,
            # not reusing fits, and not creating a new independent timeseries
            for station in net:
                station_ts = station.timeseries['mydata']
                station_models = station.models['mydata']
                # a subset list of models would need to be collected here
                for imodel, (model_description, model) in enumerate(station_models.items()):
                    fit = model.evaluate(station_ts)
                    station.add_fit('mydata', model_description, fit)
            # short version, automatically parallelizes according to geonat.defaults
            net.evaluate('mydata')
            # the short version also allows for easy subsetting models,
            # creating a new timeseries and saving time by reusing the previous fit
            net.evaluate('mydata', model_list=['onlythisone'],
                         output_description='evaluated', reuse=True)

        See Also
        --------
        fit : Fit models at all stations.
        :attr:`~geonat.config.defaults` : Dictionary of settings, including parallelization.
        geonat.tools.parallelize : Automatically execute a function in parallel or serial.
        """
        assert isinstance(ts_description, str), \
            f"'ts_description' must be string, got {type(ts_description)}."
        if output_description is not None:
            assert isinstance(output_description, str), \
                f"'output_description' must be a string, got {type(output_description)}."
        if reuse:
            assert output_description is not None, \
                "When reusing a previous model evaluation, 'output_description' " \
                "must be set (otherwise nothing would happen)."
        # should directly use fit from station timeseries, skip evaluation.
        # can only be used if the timeseries' timevector is used
        # (and wouldn't do anything if output_description is None)
        if reuse:
            assert timevector is None, \
                "When reusing a previous model evaluation, 'timevector' has to be None"
            for station in self:
                stat_subset_models = station.models[ts_description] if model_list is None \
                                     else {m: station.models[ts_description][m]
                                           for m in model_list}
                for imodel, model_description in enumerate(stat_subset_models):
                    try:
                        ts = station.fits[ts_description][model_description]
                    except KeyError as e:
                        raise KeyError(f"Station {station}, Timeseries '{ts_description}' or "
                                       f"Model '{model_description}': Fit not found, cannot "
                                       "evaluate.").with_traceback(e.__traceback__) from e
                    if imodel == 0:
                        model_aggregate = ts
                    else:
                        model_aggregate += ts
                station.add_timeseries(output_description, model_aggregate,
                                       override_src='model',
                                       override_data_cols=station[ts_description].data_cols)
        # if not reusing, have to evaluate the models and add to timeseries' fit
        # and optionally add the aggregate timeseries to the station
        else:
            iterable_inputs = ((station.timeseries[ts_description].time
                                if timevector is None else timevector,
                                station.models[ts_description] if model_list is None
                                else {m: station.models[ts_description][m] for m in model_list
                                      if m in station.models[ts_description]})
                               for station in self)
            station_names = list(self.stations.keys())
            for i, result in enumerate(tqdm(parallelize(self._evaluate_single_station,
                                                        iterable_inputs),
                                            desc="Evaluating station models",
                                            total=len(self.stations), ascii=True,
                                            unit="station")):
                stat_name = station_names[i]
                for imodel, (model_description, fit) in enumerate(result.items()):
                    ts = self[stat_name].add_fit(ts_description, model_description, fit)
                    if output_description is not None:
                        if imodel == 0:
                            model_aggregate = ts
                        else:
                            model_aggregate += ts
                if output_description is not None:
                    stat_ts = self[stat_name][ts_description]
                    self[stat_name].add_timeseries(output_description, model_aggregate,
                                                   override_src='model',
                                                   override_data_cols=stat_ts.data_cols)

    @staticmethod
    def _evaluate_single_station(parameter_tuple):
        station_time, station_models = parameter_tuple
        fit = {}
        for model_description, model in station_models.items():
            fit[model_description] = model.evaluate(station_time)
        return fit

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
            :mod:`~geonat.processing`, otherwise the function will be assumed to adhere
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
            # short version, automatically parallelizes according to geonat.defaults
            net.call_func_ts_return(func, 'input', ts_out='output', **kw_args)
            # if using a geonat.processing function, no need to import that function
            net.call_func_ts_return('clean', 'input', ts_out='output', **kw_args)

        See Also
        --------
        :attr:`~geonat.config.defaults` : Dictionary of settings, including parallelization.
        geonat.tools.parallelize : Automatically execute a function in parallel or serial.
        """
        if not callable(func):
            if isinstance(func, str):
                try:
                    func = getattr(geonat_processing, func)
                except AttributeError as e:
                    raise AttributeError(
                        f"'{func}' can not be found as a function in geonat.processing."
                        ).with_traceback(e.__traceback__) from e
            else:
                raise RuntimeError(f"'{func}' needs to be a function or a string "
                                   "representation thereof "
                                   "(if loaded from geonat.processing).")
        assert isinstance(ts_in, str), f"'ts_in' must be string, got {type(ts_in)}."
        if ts_out is None:
            ts_out = ts_in
        else:
            assert isinstance(ts_out, str), \
                f"'ts_out' must be None or a string, got {type(ts_out)}."
        iterable_inputs = ((func, station, ts_in, kw_args) for station in self)
        station_names = list(self.stations.keys())
        for i, result in enumerate(tqdm(parallelize(self._single_call_func_ts_return,
                                                    iterable_inputs),
                                        desc="Processing station timeseries with "
                                        f"'{func.__name__}'", total=len(self.stations),
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
            :mod:`~geonat.processing`, otherwise the function will be assumed to adhere
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
            # the short version also knows about the functions in geonat.processing
            net.call_netwide_func('common_mode', ts_in='input, ts_out='output', **kw_args)

        See Also
        --------
        export_network_ts : Export the timeseries of all stations in the network.
        import_network_ts : Import an entire netwrok's timeseries and distribute to stations.
        """
        if not callable(func):
            if isinstance(func, str):
                try:
                    func = getattr(geonat_processing, func)
                except AttributeError as e:
                    raise AttributeError(
                        f"'{func}' can not be found as a function in geonat.processing."
                        ).with_traceback(e.__traceback__) from e
            else:
                raise RuntimeError(f"'{func}' needs to be a function or a string "
                                   "representation thereof "
                                   "(if loaded from geonat.processing).")
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
            :mod:`~geonat.processing`, otherwise the function will be assumed to adhere to the
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
            # shorter version, no need to import a geonat.processing function first
            net.call_func_no_return('clean', **kw_args)
        """
        if not callable(func):
            if isinstance(func, str):
                try:
                    func = getattr(geonat_processing, func)
                except AttributeError as e:
                    raise AttributeError(
                        f"'{func}' can not be found as a function in geonat.processing."
                        ).with_traceback(e.__traceback__) from e
            else:
                raise RuntimeError(f"'{func}' needs to be a function or a string "
                                   "representation thereof "
                                   "(if loaded from geonat.processing).")
        for name, station in tqdm(self.stations.items(),
                                  desc=f"Calling function '{func.__name__}' on stations",
                                  ascii=True, unit="station"):
            func(station, **kw_args)

    def _create_map_figure(self, gui_settings):
        # get location data and projections
        stat_lats = [station.location[0] for station in self]
        stat_lons = [station.location[1] for station in self]
        stat_names = list(self.stations.keys())
        proj_gui = getattr(ccrs, gui_settings["projection"])()
        proj_lla = ccrs.PlateCarree()
        # create figure and plot stations
        fig_map = plt.figure()
        ax_map = fig_map.add_subplot(projection=proj_gui)
        default_station_edges = ['none'] * len(stat_lats)
        stat_points = ax_map.scatter(stat_lons, stat_lats, s=100, facecolor='C0',
                                     linestyle='None', marker='.', transform=proj_lla,
                                     edgecolor=default_station_edges, zorder=1000)
        for sname, slon, slat in zip(stat_names, stat_lons, stat_lats):
            ax_map.annotate(sname, (slon, slat),
                            xycoords=proj_lla._as_mpl_transform(ax_map), annotation_clip=True,
                            textcoords="offset pixels", xytext=(0, 5), ha="center")
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

    def graphical_cme(self, ts_in, ts_out=None, gui_kw_args={}, **cme_kw_args):
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
        gui_kw_args : dict
            Override default GUI settings of :attr:`~geonat.config.defaults`.
        **cme_kw_args : dict
            Additional keyword arguments passed to :func:`~geonat.processing.common_mode`.

        See Also
        --------
        geonat.processing.common_mode : CME calculation function.
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
            stat_points, stat_lats, stat_lons = self._create_map_figure(gui_settings)
        if ndim == 1:
            quiv = ax_map.quiver(latlonenu[:, 1], latlonenu[:, 0],
                                 np.zeros_like(latlonenu[:, 2]), latlonenu[:, 2],
                                 units='xy', transform=proj_lla)
        else:
            quiv = ax_map.quiver(latlonenu[:, 1], latlonenu[:, 0],
                                 latlonenu[:, 2], latlonenu[:, 3],
                                 units='xy', transform=proj_lla)
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
        # show plot
        plt.show()

    def gui(self, station=None, timeseries=None, model_list=None, sum_models=True, verbose=False,
            scalogram_kw_args=None, stepdetector={}, trend_kw_args={},
            analyze_kw_args={}, rms_on_map={}, gui_kw_args={}):
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
        - restrict the output to a timewindow showing potential steps from multiple sources,
        - color the station markers by RMS and include a colormap,
        - show a scalogram (Model class permitting),
        - print statistics of residuals, and
        - print station information.

        Parameters
        ----------
        station : str, optional
            Pre-select a station.
        timeseries : list, optional
            List of strings with the descriptions of the timeseries to plot.
            Defaults to all timeseries.
        model_list : list, optional
            List of strings containing the model names of the subset of the models
            to be evaluated. Defaults to all fitted models.
        sum_models : bool, optional
            If ``True``, plot the sum of all selected models instead of every
            model individually. Defaults to ``False``.
        verbose : bool, optional
            If ``True``, when clicking on a station, print its details (see
            :meth:`~geonat.station.Station.__repr__`). Defaults to ``False``.
        scalogram_kw_args : dict, optional
            If passed, also plot a scalogram. Defaults to no scalogram shown.
            The dictionary has to contain ``'ts'`` and ``'model'`` keys. The string values
            are the names of the timeseries and associated model that are of the
            :class:`~geonat.models.SplineSet` class, and therefore have a
            :meth:`~geonat.models.SplineSet.make_scalogram` method.
        stepdetector : dict, optional
            Passing this dictionary will enable the plotting of events related to possible
            steps, both on the map (in case of an earthquake catalog) and in the timeseries
            (in case of detected steps by :class:`~geonat.processing.StepDetector` or a
            maintenance catalog). To reduce cluttering, it will also only show a subset of
            each timeseries, centered around consecutive probable steps. Using the terminal,
            one can cycle through each period.
            The ``stepdetector`` dictionary must contain the keys ``'plot_padding'`` (which
            determines how many days before and after to include in the plotting),
            ``'step_table'`` and ``'step_ranges'`` (both returned by
            :meth:`~geonat.processing.StepDetector.search_network`),
            ``'step_padding'`` (which determines how many days before and after the
            ``'step_ranges'`` should be scanned for possible steps), optionally ``'catalog'``
            and ``'eqcircle'`` (an earthquake catalog of same style as used in
            :mod:`~geonat.earthquakes` and a maximum distance of stations to earthquakes of
            magnitude less than 7.5), and optionally, ``'maint_table'`` (a maintenance table
            as parsed by :func:`~geonat.tools.parse_maintenance_table`).
        trend_kw_args : dict, optional
            If passed, also plot velocity trends on the station map.
            Defaults to no velocity arrows shown.
            The dictionary can contain all the keywords that are passed to
            :meth:`~geonat.station.Station.get_trend`, but has at least has to contain
            the ``ts_description``. If no ``model_list`` is included, the ``model_list``
            passed to ``gui()`` will be used instead. If the number of components available
            is 3 or more, only the first two will be used. If two components are plotted,
            they correspond to the East and North components. If only one component is plotted
            (for example for vertical motion), it will be plotted as the North component.
        analyze_kw_args : dict, optional
            If provided and non-empty, call :meth:`~geonat.station.Station.analyze_residuals`
            and pass the dictionary on as keyword arguments (overriding ``'verbose'`` to
            ``True`` to force an output). Defaults to no residual analysis.
        rms_on_map : dict, optional
            If provided and non-empty, this option will call
            :meth:`~geonat.station.Station.analyze_residuals` to calculate
            a residual timeseries' root-mean-squares to color the station markers on the map.
            The dictionary must include the key ``'ts'`` (the residual timeseries' name), and
            can optionally include the keys ``'comps'`` (a list of components to combine for
            the RMS, defaults to all components), ``'c_max'`` (maximum colormap range,
            defaults to maximum of the RMS), and ``'t_start', 't_end'`` (to restrict the time
            window, defaults to ``analyze_kw_args`` if given, otherwise the entire timeseries).
        gui_kw_args : dict
            Override default GUI settings of :attr:`~geonat.config.defaults`.
        """
        # create map and timeseries figures
        gui_settings = defaults["gui"].copy()
        gui_settings.update(gui_kw_args)
        fig_map, ax_map, proj_gui, proj_lla, default_station_edges, \
            stat_points, stat_lats, stat_lons = self._create_map_figure(gui_settings)
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

        # make sure that if analyze_kw_args is used, 'verbose' is set and True
        if analyze_kw_args:
            analyze_kw_args["verbose"] = True

        # color the station markers by RMS
        if rms_on_map:
            # check input
            assert isinstance(rms_on_map, dict) and ("ts" in rms_on_map) and \
                   isinstance(rms_on_map["ts"], str), \
                   "'rms_on_map' needs to be a dictionary including the key 'ts' with a " \
                   f"string value, got {rms_on_map}."
            # collect RMS
            rms_comps = True if "comps" not in rms_on_map else rms_on_map["comps"]
            rms_kw_args = {"rms": rms_comps}
            for k in ["t_start", "t_end"]:
                if k in rms_on_map:
                    rms_kw_args[k] = rms_on_map[k]
                elif k in analyze_kw_args:
                    rms_kw_args[k] = analyze_kw_args[k]
            rms = np.zeros(self.num_stations)
            for i, stat in enumerate(self):
                rms[i] = stat.analyze_residuals(rms_on_map["ts"], **rms_kw_args)["RMS"]
            # make colormap
            cmax = np.max(rms) if "c_max" not in rms_on_map else rms_on_map["c_max"]
            rms_cmap = mpl.cm.ScalarMappable(cmap=scm.lajolla_r,
                                             norm=mpl.colors.Normalize(vmin=0, vmax=cmax))
            # update marker facecolors
            stat_points.set_facecolor(rms_cmap.to_rgba(rms))
            fig_map.canvas.draw_idle()
            # add colorbar
            cbar = fig_map.colorbar(rms_cmap, ax=ax_map, orientation="horizontal",
                                    fraction=0.05, pad=0.03, aspect=10,
                                    extend="max" if np.max(rms) > cmax else "neither")
            cbar.set_label(f"RMS [{stat[rms_on_map['ts']].data_unit}]")

        # add velocity map
        if trend_kw_args:
            assert "ts_description" in trend_kw_args, \
                "'trend_kw_args' dictionary has to include a 'ts_description' keyword."
            if "model_list" not in trend_kw_args:
                trend_kw_args["model_list"] = model_list
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
            # TODO: plot uncertainty ellipses

        # prepare stuff for the interactive step plotting
        if stepdetector:
            # get quick access
            step_table = stepdetector["step_table"]
            step_ranges = stepdetector["step_ranges"]
            maint_table = stepdetector["maint_table"] if "maint_table" in stepdetector else None
            if "catalog" in stepdetector:
                catalog = stepdetector["catalog"]
                eqcircle = stepdetector["eqcircle"]
            else:
                catalog = None
            step_padding = stepdetector["step_padding"]
            plot_padding = stepdetector["plot_padding"]
            n_ranges = len(step_ranges)
            i_range = 0
            geoid = cgeod.Geodesic()
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
                    station_index = np.argmin(np.sqrt((np.array(stat_lats) - click_lat)**2
                                                      + (np.array(stat_lons) - click_lon)**2))
                    station_name = list(self.stations.keys())[station_index]
                elif station_name is None:
                    return
            else:
                station_name = select_station
                station_index = list(self.stations.keys()).index(station_name)
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
                # get model subset
                fits_to_plot = {model_description: fit for model_description, fit
                                in self[station_name].fits[ts_description].items()
                                if (model_list is None) or (model_description in model_list)}
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
                    if sum_models:
                        fit_sum = np.zeros(ts.time.size)
                        fit_sum_sigma = np.zeros(ts.time.size)
                        for model_description, fit in fits_to_plot.items():
                            fit_sum += fit.df[fit.data_cols[icol]].values
                            if (fit.var_cols is not None) \
                               and (gui_settings["plot_sigmas"] > 0):
                                fit_sum_sigma += fit.df[fit.var_cols[icol]].values
                        if fit_sum_sigma.sum() > 0:
                            fit_sum_sigma = np.sqrt(fit_sum_sigma)
                            fill_upper = fit_sum + gui_settings["plot_sigmas"] * fit_sum_sigma
                            fill_lower = fit_sum - gui_settings["plot_sigmas"] * fit_sum_sigma
                            ax.fill_between(fit.time, fill_upper, fill_lower,
                                            alpha=gui_settings["plot_sigmas_alpha"],
                                            linewidth=0)
                        if np.abs(fit_sum).sum() > 0:
                            ax.plot(fit.time, fit_sum, label="Model")
                    else:
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

            # get scalogram
            if scalogram_kw_args is not None:
                try:
                    splset = self[station_name].models[scalo_ts][scalo_model]
                    if isinstance(fig_scalo, plt.Figure):
                        plt.close(fig_scalo)
                    fig_scalo, ax_scalo = splset.make_scalogram(t_left, t_right,
                                                                **scalogram_kw_args)
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
        else:
            plt.show()
        del click
