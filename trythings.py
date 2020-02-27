import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
import multiprocessing
from copy import deepcopy
from configparser import ConfigParser
from tqdm import tqdm
from multiprocessing import Pool
from functools import wraps
from okada_wrapper import dc3d0wrapper as dc3d0
from pandas.plotting import register_matplotlib_converters
multiprocessing.set_start_method('spawn', True)
register_matplotlib_converters()

# import defaults
config = ConfigParser()
config.read("config.ini")


class Network():
    """
    A container object for multiple stations.
    """
    def __init__(self, name, default_location_path=None, default_local_models={}):
        self.name = name
        self.default_location_path = default_location_path
        self.default_local_models = default_local_models
        self.stations = {}
        self.global_models = {}
        # self.global_priors = {}

    def __repr__(self):
        info = f"Network {self.name}\n" + \
               f"Stations:\n{[key for key in self.stations]}\n" + \
               f"Global Models:\n{[key for key in self.global_models]}\n" + \
               f"Global Priors:\n{[key for key in self.global_priors]}"
        return info

    def __getitem__(self, name):
        if name not in self.stations:
            raise KeyError("No station '{}' present.".format(name))
        return self.stations[name]

    def __setitem__(self, name, station):
        self.add_station(name, station)

    def __delitem__(self, name):
        self.remove_station(name)

    def __iter__(self):
        for station in self.stations.values():
            yield station

    def export_network_ts(self, ts_description):
        """
        Return the timeseries of all stations as a dictionary of Timeseries.
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
            raise ValueError("No data found in network {:s} for timeseries {:s}".format(self.name, ts_description))
        network_df = pd.concat(df_dict, axis=1)
        network_df.columns = network_df.columns.swaplevel(0, 1)
        network_df.sort_index(level=0, axis=1, inplace=True)
        network_df = {dcol: Timeseries(network_df[dcol], network_src, network_unit, [col for col in network_df[dcol].columns]) for dcol in network_data_cols}
        return network_df

    def import_network_ts(self, ts_description, dict_of_timeseries):
        """
        Set the timeseries of all stations as a dictionary of Timeseries.
        """
        network_df = pd.concat({name: ts.data for name, ts in dict_of_timeseries.items()}, axis=1)
        network_df.columns = network_df.columns.swaplevel(0, 1)
        network_df.sort_index(level=0, axis=1, inplace=True)
        data_cols = [dcol for dcol in dict_of_timeseries]
        src = dict_of_timeseries[data_cols[0]].src
        data_unit = dict_of_timeseries[data_cols[0]].data_unit
        for name in network_df.columns.levels[0]:
            self.stations[name].add_timeseries(ts_description, Timeseries(network_df[name].dropna(), src, data_unit, data_cols))

    def add_station(self, name, station):
        if not isinstance(name, str):
            raise TypeError("Cannot add new station: 'name' is not a string.")
        if not isinstance(station, Station):
            raise TypeError("Cannot add new station: 'station' is not a Station object.")
        if name in self.stations:
            Warning("Overwriting station {:s}".format(name))
        self.stations[name] = station

    def remove_station(self, name):
        if name not in self.stations:
            Warning("Cannot find station {}, couldn't delete".format(name))
        else:
            del self.stations[name]

    def add_global_model(self, description, model):
        if not isinstance(description, str):
            raise TypeError("Cannot add new global model: 'description' is not a string.")
        if description in self.global_models:
            Warning("Overwriting global model {:s}".format(description))
        self.global_models[description] = model

    def remove_global_model(self, description):
        if description not in self.global_models:
            Warning("Cannot find global model {}, couldn't delete".format(description))
        else:
            del self.global_models[description]

    # def add_global_prior(self, description, prior):
    #     if description in self.global_priors:
    #         Warning("Overwriting global prior {:s}".format(description))
    #     self.global_priors[description] = prior

    # def remove_global_prior(self, description):
    #     if description not in self.global_priors:
    #         Warning("Cannot find global prior {:s}, couldn't delete".format(description))
    #     else:
    #         del self.global_priors[description]

    @classmethod
    def from_json(cls, path, add_default_local_models=True):
        # load configuration
        net_arch = json.load(open(path, mode='r'))
        network_name = net_arch.get("name", "network_from_json")
        network_locations_path = net_arch.get("locations")
        # create Network instance
        net = cls(name=network_name, default_location_path=network_locations_path, default_local_models=net_arch["default_local_models"])
        # load location information, if present
        if network_locations_path is not None:
            with open(network_locations_path, mode='r') as locfile:
                loclines = [line.strip() for line in locfile.readlines()]
            network_locations = {line.split()[0]: [float(lla) for lla in line.split()[1:]] for line in loclines}
        # create stations
        for station_name, station_cfg in tqdm(net_arch["stations"].items(), ascii=True, desc="Building Network", unit="station"):
            if "location" in station_cfg:
                station_loc = station_cfg["location"]
            elif station_name in network_locations:
                station_loc = network_locations[station_name]
            else:
                Warning("Skipped station {:s} because location information is missing.".format(station_name))
                continue
            station = Station(name=station_name, location=station_loc)
            # add timeseries to station
            for ts_description, ts_cfg in station_cfg["timeseries"].items():
                ts = globals()[ts_cfg["type"]](**ts_cfg["kw_args"])
                station.add_timeseries(description=ts_description, timeseries=ts)
                # add default local models to station
                if add_default_local_models:
                    for model_description, model_cfg in net.default_local_models.items():
                        local_copy = deepcopy(model_cfg)
                        mdl = globals()[local_copy["type"]](**local_copy["kw_args"])
                        station.add_local_model(ts_description=ts_description, model_description=model_description, model=mdl)
            # add specific local models to station and timeseries
            for ts_description, ts_model_dict in station_cfg["models"].items():
                if ts_description in station.timeseries:
                    for model_description, model_cfg in ts_model_dict.items():
                        mdl = globals()[model_cfg["type"]](**model_cfg["kw_args"])
                        station.add_local_model(ts_description=ts_description, model_description=model_description, model=mdl)
                else:
                    station._unused_models.update({ts_description: ts_model_dict})
            # add to network
            net.add_station(name=station_name, station=station)
        # add global models
        for model_description, model_cfg in net_arch["global_models"].items():
            mdl = globals()[model_cfg["type"]](**model_cfg["kw_args"])
            net.add_global_model(description=model_description, model=mdl)
        return net

    def to_json(self, path):
        # create new dictionary
        net_arch = {"name": self.name,
                    "locations": self.default_location_path,
                    "stations": {},
                    "default_local_models": self.default_local_models,
                    "global_models": {}}
        # add station representations
        for stat_name, station in self.stations.items():
            stat_arch = station.get_arch()
            if stat_arch == {}:
                continue
            # need to remove all models that are actually default models
            for mdl_description, mdl in self.default_local_models.items():
                for ts_description in stat_arch["models"]:
                    if (mdl_description in stat_arch["models"][ts_description].keys()) and (mdl == stat_arch["models"][ts_description][mdl_description]):
                        del stat_arch["models"][ts_description][mdl_description]
            # now we can append it to the main json
            net_arch["stations"].update({stat_name: stat_arch})
        # add global model representations
        for mdl_description, mdl in self.global_models.items():
            net_arch["global_models"].update({mdl_description: mdl.get_arch()})
        # write file
        json.dump(net_arch, open(path, mode='w'), indent=2, sort_keys=True)

    def add_default_local_models(self, ts_description, models=None):
        assert isinstance(ts_description, str), f"'ts_description' must be string, got {type(ts_description)}."
        if models is None:
            local_models_subset = net.default_local_models
        else:
            if not isinstance(models, str) or not (isinstance(models, list) and all([isinstance(m, str) for m in models])):
                raise ValueError(f"'models' must be None, a string or a list of strings, got {models}.")
            if isinstance(models, str):
                models = [models]
            local_models_subset = {name: model for name, model in net.default_local_models.items() if name in models}
        for station in self:
            for model_description, model_cfg in local_models_subset.items():
                local_copy = deepcopy(model_cfg)
                mdl = globals()[local_copy["type"]](**local_copy["kw_args"])
                station.add_local_model(ts_description=ts_description, model_description=model_description, model=mdl)

    def add_unused_local_models(self, target_ts, hidden_ts=None, models=None):
        assert isinstance(target_ts, str), f"'target_ts' must be string, got {type(target_ts)}."
        if hidden_ts is None:
            hidden_ts = target_ts
        else:
            assert isinstance(hidden_ts, str), f"'hidden_ts' must be None or a string, got {type(hidden_ts)}."
        assert models is None or isinstance(models, str) or (isinstance(models, list) and all([isinstance(m, str) for m in models])), \
            f"'models' must be None, a string or a list of strings, got {models}."
        if isinstance(models, str):
            models = [models]
        for station in self:
            if hidden_ts in station._unused_models:
                if models is None:
                    local_models_subset = station._unused_models[hidden_ts]
                else:
                    local_models_subset = {name: model for name, model in net.default_local_models.items() if name in models}
                for model_description, model_cfg in local_models_subset.items():
                    local_copy = deepcopy(model_cfg)
                    mdl = globals()[local_copy["type"]](**local_copy["kw_args"])
                    station.add_local_model(ts_description=target_ts, model_description=model_description, model=mdl)

    def fit(self, ts_description, model_list=None, solver=config.get('fit', 'solver'), **kw_args):
        assert isinstance(ts_description, str), f"'ts_description' must be string, got {type(ts_description)}."
        if isinstance(solver, str):
            solver = globals()[solver]
        assert callable(solver), f"'solver' must be a callable function, got {type(solver)}."
        iterable_inputs = ((station.timeseries[ts_description],
                            station.models[ts_description] if model_list is None else {m: station.models[ts_description][m] for m in model_list},
                            solver, kw_args) for station in self)
        station_names = list(self.stations.keys())
        for i, result in enumerate(tqdm(parallelize(self._fit_single_station, iterable_inputs),
                                        desc="Fitting station models", total=len(self.stations), ascii=True, unit="station")):
            for model_description, (params, covs) in result.items():
                self[station_names[i]].models[ts_description][model_description].read_parameters(params, covs)

    @staticmethod
    def _fit_single_station(parameter_tuple):
        station_time, station_models, solver, kw_args = parameter_tuple
        fitted_params = solver(station_time, station_models, **kw_args)
        return fitted_params

    def evaluate(self, ts_description, model_list=None, timevector=None, output_description=None):
        assert isinstance(ts_description, str), f"'ts_description' must be string, got {type(ts_description)}."
        if output_description is not None:
            assert isinstance(output_description, str), f"'output_description' must be a string, got {type(output_description)}."
        iterable_inputs = ((station.timeseries[ts_description].time if timevector is None else timevector,
                            station.models[ts_description] if model_list is None else {m: station.models[ts_description][m] for m in model_list})
                           for station in self)
        station_names = list(self.stations.keys())
        for i, result in enumerate(tqdm(parallelize(self._evaluate_single_station, iterable_inputs),
                                        desc="Evaluating station models", total=len(self.stations), ascii=True, unit="station")):
            station = station_names[i]
            if output_description is None:
                for model_description, fit in result.items():
                    self[station].add_fit(ts_description, model_description, fit)
            else:
                ts_list = []
                for model_description, fit in result.items():
                    ts = self[station].add_fit(ts_description, model_description, fit)
                    ts_list.append(ts)
                output_ts = ts_list[0]
                for its in range(1, len(ts_list)):
                    output_ts = output_ts + ts_list[its]
                self[station].add_timeseries(output_description, output_ts, override_src='model', override_data_cols=self[station][ts_description].data_cols)

    @staticmethod
    def _evaluate_single_station(parameter_tuple):
        station_time, station_models = parameter_tuple
        fit = {}
        for model_description, model in station_models.items():
            fit[model_description] = model.evaluate(station_time)
        return fit

    def call_func_ts_return(self, func, ts_in, ts_out=None, **kw_args):
        assert callable(func), f"'func' must be a callable function, got {type(func)}."
        assert isinstance(ts_in, str), f"'ts_in' must be string, got {type(func)}."
        if ts_out is None:
            ts_out = ts_in
        else:
            assert isinstance(ts_out, str), f"'ts_out' must be None or a string, got {type(func)}."
        iterable_inputs = ((func, station, ts_in, kw_args) for station in self)
        station_names = list(self.stations.keys())
        for i, result in enumerate(tqdm(parallelize(self._single_call_func_ts_return, iterable_inputs),
                                        desc="Processing station timeseries with {:s}".format(func.__name__), total=len(self.stations), ascii=True, unit="station")):
            self[station_names[i]].add_timeseries(ts_out, result)

    @staticmethod
    def _single_call_func_ts_return(parameter_tuple):
        func, station, ts_description, kw_args = parameter_tuple
        ts_return = func(station.timeseries[ts_description], **kw_args)
        return ts_return

    def call_netwide_func(self, func, ts_in, ts_out=None, **kw_args):
        assert callable(func), f"'func' must be a callable function, got {type(func)}."
        assert isinstance(ts_in, str), f"'ts_in' must be string, got {type(func)}."
        if ts_out is None:
            ts_out = ts_in
        else:
            assert isinstance(ts_out, str), f"'ts_out' must be None or a string, got {type(func)}."
        net_in = self.export_network_ts(ts_in)
        net_out = func(net_in, **kw_args)
        self.import_network_ts(ts_in if ts_out is None else ts_out, net_out)

    def call_func_no_return(self, func, **kw_args):
        assert callable(func), f"'func' must be a callable function, got {type(func)}."
        for name, station in tqdm(self.stations.items(), desc="Calling function {:s} on stations".format(func.__name__), ascii=True, unit="station"):
            func(station, **kw_args)

    def _create_map_figure(self):
        # get location data and projections
        stat_lats = [station.location[0] for station in self]
        stat_lons = [station.location[1] for station in self]
        proj_gui = getattr(ccrs, config.get("gui", "projection", fallback="Mercator"))()
        proj_lla = ccrs.PlateCarree()
        # create figure and plot stations
        fig_map = plt.figure()
        ax_map = fig_map.add_subplot(projection=proj_gui)
        default_station_colors = ['b'] * len(stat_lats)
        stat_points = ax_map.scatter(stat_lons, stat_lats, linestyle='None', marker='.', transform=proj_lla, facecolor=default_station_colors, zorder=1000)
        # create underlay
        map_underlay = False
        if "wmts_server" in config.options("gui"):
            try:
                ax_map.add_wmts(config.get("gui", "wmts_server"), layer_name=config.get("gui", "wmts_layer"), alpha=config.getfloat("gui", "wmts_alpha"))
                map_underlay = True
            except Exception as exc:
                print(exc)
        if "coastlines_resolution" in config.options("gui"):
            ax_map.coastlines(color="white" if map_underlay else "black", resolution=config.get("gui", "coastlines_resolution"))
        return fig_map, ax_map, proj_gui, proj_lla, default_station_colors, stat_points, stat_lats, stat_lons

    def graphical_cme(self, ts_in, ts_out=None, **kw_args):
        # get common mode and make sure to return spatial and temporal models
        net_in = self.export_network_ts(ts_in)
        comps = list(net_in.keys())
        ndim = len(comps)
        ndim_max2 = min(ndim, 2)
        assert ndim > 0, f"No components found in {ts_in:s}"
        kw_args.update({'plot': True})
        model, temp_spat = common_mode(net_in, **kw_args)
        temporal, spatial = {}, {}
        for comp in model:
            temporal[comp] = temp_spat[comp][0]
            spatial[comp] = temp_spat[comp][1]
        # assign model to network
        ts_out = ts_in if ts_out is None else ts_out
        self.import_network_ts(ts_out, model)
        # extract spatial components
        fitted_stations = model[comps[0]].data_cols
        latlonenu = np.zeros((len(fitted_stations), 2 + ndim_max2))
        for i, station_name in enumerate(fitted_stations):
            latlonenu[i, :2] = self[station_name].location[:2]
            for j in range(ndim_max2):
                latlonenu[i, 2 + j] = spatial[comps[j]][0, i]
        # make map for spatial component
        fig_map, ax_map, proj_gui, proj_lla, default_station_colors, stat_points, stat_lats, stat_lons = self._create_map_figure()
        if ndim == 1:
            quiv = ax_map.quiver(latlonenu[:, 1], latlonenu[:, 0], latlonenu[:, 2], np.zeros_like(latlonenu[:, 2]), units='xy', transform=proj_lla)
        else:
            quiv = ax_map.quiver(latlonenu[:, 1], latlonenu[:, 0], latlonenu[:, 2], latlonenu[:, 3], units='xy', transform=proj_lla)
        key_length = np.median(np.sqrt(np.sum(np.atleast_2d(latlonenu[:, 2:2 + ndim_max2]**2), axis=1)))
        ax_map.quiverkey(quiv, 0.9, 0.9, key_length, f"{key_length:.2e} {model[comps[0]].data_unit:s}", coordinates="figure")
        # make timeseries figure
        fig_ts, ax_ts = plt.subplots(nrows=len(temporal), sharex=True)
        for icomp, (comp, ts) in enumerate(temporal.items()):
            ax_ts[icomp].plot(model[comp].time, ts)
            ax_ts[icomp].set_ylabel(f"{comp:s} [{model[comp].data_unit:s}]")
            ax_ts[icomp].grid()
        # show plot
        plt.show()

    def gui(self, timeseries=None, verbose=False, **analyze_kw_args):
        # create map and timeseries figures
        fig_map, ax_map, proj_gui, proj_lla, default_station_colors, stat_points, stat_lats, stat_lons = self._create_map_figure()
        fig_ts = plt.figure()

        # define clicking function
        def update_timeseries(event):
            if (event.xdata is None) or (event.ydata is None) or (event.inaxes is not ax_map): return
            click_lon, click_lat = proj_lla.transform_point(event.xdata, event.ydata, src_crs=proj_gui)
            station_index = np.argmin(np.sqrt((np.array(stat_lats) - click_lat)**2 + (np.array(stat_lons) - click_lon)**2))
            station_name = list(self.stations.keys())[station_index]
            if verbose:
                print(self.stations[station_name])
            highlight_station_colors = default_station_colors.copy()
            highlight_station_colors[station_index] = 'r'
            stat_points.set_facecolor(highlight_station_colors)
            fig_map.canvas.draw_idle()
            # get components
            ts_to_plot = {ts_description: ts for ts_description, ts in self[station_name].timeseries.items()
                          if (timeseries is None) or (ts_description in timeseries)}
            n_components = 0
            for ts_desc, ts in ts_to_plot.items():
                n_components += ts.num_components
                if len(analyze_kw_args) > 0:
                    self.stations[station_name].analyze_residuals(ts_desc, **analyze_kw_args)
            # clear figure and add data
            fig_ts.clear()
            icomp = 0
            ax_ts = []
            for its, (ts_description, ts) in enumerate(ts_to_plot.items()):
                for icol, (data_col, sigma_col) in enumerate(zip(ts.data_cols, ts.sigma_cols)):
                    # add axis
                    ax = fig_ts.add_subplot(n_components, 1, icomp + 1, sharex=None if icomp == 0 else ax_ts[0])
                    # plot uncertainty
                    if sigma_col is not None and "plot_sigmas" in config.options("gui"):
                        ax.fill_between(ts.time, ts.df[data_col] + config.getfloat("gui", "plot_sigmas") * ts.df[sigma_col],
                                        ts.df[data_col] - config.getfloat("gui", "plot_sigmas") * ts.df[sigma_col], facecolor='gray',
                                        alpha=config.getfloat("gui", "plot_sigmas_alpha"), linewidth=0)
                    # plot data
                    ax.plot(ts.time, ts.df[data_col], marker='.', color='k', label="Data" if len(self[station_name].fits[ts_description]) > 0 else None)
                    # overlay models
                    for (mdl_description, fit) in self[station_name].fits[ts_description].items():
                        if fit.sigma_cols[icol] is not None and "plot_sigmas" in config.options("gui"):
                            ax.fill_between(fit.time, fit.df[fit.data_cols[icol]] + config.getfloat("gui", "plot_sigmas") * fit.df[fit.sigma_cols[icol]],
                                            fit.df[fit.data_cols[icol]] - config.getfloat("gui", "plot_sigmas") * fit.df[fit.sigma_cols[icol]],
                                            alpha=config.getfloat("gui", "plot_sigmas_alpha"), linewidth=0)
                        ax.plot(fit.time, fit.df[fit.data_cols[icol]], label=mdl_description)
                    ax.set_ylabel("{:s}\n{:s} [{:s}]".format(ts_description, data_col, ts.data_unit))
                    ax.grid()
                    if len(self[station_name].fits[ts_description]) > 0:
                        ax.legend()
                    ax_ts.append(ax)
                    icomp += 1
            ax_ts[0].set_title(station_name)
            fig_ts.canvas.draw_idle()

        cid = fig_map.canvas.mpl_connect("button_press_event", update_timeseries)
        plt.show()
        fig_map.canvas.mpl_disconnect(cid)


class Station():
    """
    Representation of a station, which contains both metadata
    such as location as well as timeseries and associated models
    """
    def __init__(self, name, location):
        self.name = name
        self.location = location
        self.timeseries = {}
        self.models = {}
        self._unused_models = {}
        self.fits = {}

    def __repr__(self):
        info = f"Station {self.name} at {self.location} with timeseries"
        for ts_description, ts in self.timeseries.items():
            info += f"\n[{ts_description}]\n - Source: {ts.src}\n - Units: {ts.data_unit}\n" + \
                    f" - Data: {[key for key in ts.data_cols]}\n" + \
                    f" - Uncertainties: {[key for key in ts.sigma_cols]}"
            if len(self.models[ts_description]) > 0:
                info += f"\n - Models: {[key for key in self.models[ts_description]]}"
            if len(self.fits[ts_description]) > 0:
                info += f"\n - Fits: {[key for key in self.fits[ts_description]]}"
        return info

    def __getitem__(self, description):
        if description not in self.timeseries:
            raise KeyError("Station {:s}: No timeseries '{}' present.".format(self.name, description))
        return self.timeseries[description]

    def __setitem__(self, description, timeseries):
        self.add_timeseries(description, timeseries)

    def __delitem__(self, description):
        self.remove_timeseries(description)

    def __iter__(self):
        for ts in self.timeseries.values():
            yield ts

    @property
    def ts(self):
        return self.timeseries

    def get_arch(self):
        # create empty dictionary
        stat_arch = {"location": self.location,
                     "timeseries": {},
                     "models": {}}
        # add each timeseries and model
        for ts_description, ts in self.timeseries.items():
            ts_arch = ts.get_arch()
            if ts_arch != {}:
                stat_arch["timeseries"].update({ts_description: ts_arch})
            if len(self.models[ts_description]) > 0:
                stat_arch["models"][ts_description] = {}
                for mdl_description, mdl in self.models[ts_description].items():
                    stat_arch["models"][ts_description].update({mdl_description: mdl.get_arch()})
        return stat_arch

    def add_timeseries(self, description, timeseries, override_src=None, override_data_unit=None, override_data_cols=None, override_sigma_cols=None, add_models=None):
        if not isinstance(description, str):
            raise TypeError("Cannot add new timeseries: 'description' is not a string.")
        if not isinstance(timeseries, Timeseries):
            raise TypeError("Cannot add new timeseries: 'timeseries' is not a Timeseries object.")
        if description in self.timeseries:
            Warning("Station {:s}: Overwriting time series {:s}".format(self.name, description))
        if override_src is not None:
            timeseries.src = override_src
        if override_data_unit is not None:
            timeseries.data_unit = override_data_unit
        if override_data_cols is not None:
            timeseries.data_cols = override_data_cols
        if override_sigma_cols is not None:
            timeseries.sigma_cols = override_sigma_cols
        self.timeseries[description] = timeseries
        self.fits[description] = {}
        self.models[description] = {}
        if add_models is not None:
            for model_description, model_cfg in add_models.items():
                local_copy = deepcopy(model_cfg)
                mdl = globals()[local_copy["type"]](**local_copy["kw_args"])
                self.add_local_model(ts_description=description, model_description=model_description, model=mdl)

    def remove_timeseries(self, description):
        if description not in self.timeseries:
            Warning("Station {:s}: Cannot find time series {:s}, couldn't delete".format(self.name, description))
        else:
            del self.timeseries[description]
            del self.fits[description]
            del self.models[description]

    def add_local_model(self, ts_description, model_description, model):
        if not isinstance(ts_description, str):
            raise TypeError("Cannot add new local model: 'ts_description' is not a string.")
        if not isinstance(model_description, str):
            raise TypeError("Cannot add new local model: 'model_description' is not a string.")
        if not isinstance(model, Model):
            raise TypeError("Cannot add new local model: 'model' is not a Model object.")
        assert ts_description in self.timeseries, \
            "Station {:s}: Cannot find timeseries {:s} to add local model {:s}".format(self.name, ts_description, model_description)
        if model_description in self.models[ts_description]:
            Warning("Station {:s}, timeseries {:s}: Overwriting local model {:s}".format(self.name, ts_description, model_description))
        self.models[ts_description].update({model_description: model})

    def remove_local_model(self, ts_description, model_description):
        if ts_description not in self.timeseries:
            Warning("Station {:s}: Cannot find timeseries {:s}, couldn't delete local model {:s}".format(self.name, ts_description, model_description))
        elif model_description not in self.models[ts_description]:
            Warning("Station {:s}, timeseries {:s}: Cannot find local model {:s}, couldn't delete".format(self.name, ts_description, model_description))
        else:
            del self.models[ts_description][model_description]

    def add_fit(self, ts_description, model_description, fit):
        if not isinstance(ts_description, str):
            raise TypeError("Cannot add new fit: 'ts_description' is not a string.")
        if not isinstance(model_description, str):
            raise TypeError("Cannot add new fit: 'model_description' is not a string.")
        assert ts_description in self.timeseries, \
            "Station {:s}: Cannot find timeseries {:s} to add fit for model {:s}".format(self.name, ts_description, model_description)
        assert model_description in self.models[ts_description], \
            "Station {:s}, timeseries {:s}: Cannot find local model {:s}, couldn't add fit".format(self.name, ts_description, model_description)
        if model_description in self.fits[ts_description]:
            Warning("Station {:s}, timeseries {:s}: Overwriting fit of local model {:s}".format(self.name, ts_description, model_description))
        data_cols = [ts_description + "_" + model_description + "_" + dcol for dcol in self.timeseries[ts_description].data_cols]
        fit_ts = Timeseries.from_fit(self.timeseries[ts_description].data_unit, data_cols, fit)
        self.fits[ts_description].update({model_description: fit_ts})
        return fit_ts

    def remove_fit(self, ts_description, model_description, fit):
        if ts_description not in self.timeseries:
            Warning("Station {:s}: Cannot find timeseries {:s}, couldn't delete fit for model {:s}".format(self.name, ts_description, model_description))
        elif model_description not in self.models[ts_description]:
            Warning("Station {:s}, timeseries {:s}: Cannot find local model {:s}, couldn't delete fit".format(self.name, ts_description, model_description))
        elif model_description not in self.fits[ts_description]:
            Warning("Station {:s}, timeseries {:s}: Cannot find fit for local model {:s}, couldn't delete".format(self.name, ts_description, model_description))
        else:
            del self.fits[ts_description][model_description]

    def analyze_residuals(self, ts_description, mean=False, std=False, n_observations=False, std_outlier=0):
        assert isinstance(ts_description, str), f"Station {self.name}: 'ts_description' needs to be a string, got {type(ts_description)}."
        assert ts_description in self.timeseries, f"Station {self.name}: Can't find {ts_description} to analyze."
        print()
        results = {}
        if mean:
            results["Mean"] = self[ts_description].data.mean(axis=0, skipna=True, numeric_only=True).values
        if std:
            results["Standard Deviation"] = self[ts_description].data.std(axis=0, skipna=True, numeric_only=True).values
        if n_observations:
            results["Observations"] = self[ts_description].data.count(axis=0, numeric_only=True).values
            results["Gaps"] = self[ts_description].num_observations - results["Observations"]
        if std_outlier > 0:
            temp = self[ts_description].data.values
            temp[np.isnan(temp)] = 0
            temp -= np.mean(temp, axis=0, keepdims=True)
            temp = temp > np.std(temp, axis=0, keepdims=True) * std_outlier
            results["Outliers"] = np.sum(temp, axis=0, dtype=int)
        print(pd.DataFrame(data=results, index=self[ts_description].data_cols).rename_axis(f"{self.name}: {ts_description}", axis=1))


class Timeseries():
    """
    Container object that for a given time vector contains
    data points.
    """
    def __init__(self, dataframe, src, data_unit, data_cols, sigma_cols=None):
        assert isinstance(dataframe, pd.DataFrame)
        assert isinstance(src, str)
        assert isinstance(data_unit, str)
        assert isinstance(data_cols, list) and all([isinstance(dcol, str) for dcol in data_cols])
        assert all([dcol in dataframe.columns for dcol in data_cols])
        self._df = dataframe
        self._src = src
        self._data_unit = data_unit
        self._data_cols = data_cols
        if sigma_cols is None:
            self._sigma_cols = [None] * len(self._data_cols)
        else:
            assert isinstance(sigma_cols, list) and all([isinstance(scol, str) for scol in sigma_cols])
            assert all([scol in dataframe.columns for scol in sigma_cols if scol is not None])
            assert len(self._data_cols) == len(sigma_cols), \
                "If passing uncertainty columns, the list needs to have the same length as the data columns one. " + \
                "If only certain components have associated uncertainties, leave those list entries as None."
            self._sigma_cols = sigma_cols

    def __repr__(self):
        info = f"Timeseries\n - Source: {self.src}\n - Units: {self.data_unit}\n" + \
               f" - Data: {[key for key in self.data_cols]}\n" + \
               f" - Uncertainties: {[key for key in self.sigma_cols]}"
        return info

    def __getitem__(self, columns):
        if not isinstance(columns, str) and (not isinstance(columns, list) or not all([isinstance(col, str) for col in columns])):
            raise KeyError(f"Error when accessing data in timeseries: 'column' must be a string or list of strings, given was {columns}.")
        return self.df[columns]

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, new_src):
        if not isinstance(new_src, str):
            raise TypeError(f"New 'src' attribute has to be a string, got {type(new_src)}.")
        self._src = new_src

    @property
    def data_unit(self):
        return self._data_unit

    @data_unit.setter
    def data_unit(self, new_data_unit):
        if not isinstance(new_data_unit, str):
            raise TypeError(f"New 'data_unit' attribute has to be a string, got {type(new_data_unit)}.")
        self._data_unit = new_data_unit

    @property
    def data_cols(self):
        return self._data_cols

    @data_cols.setter
    def data_cols(self, new_data_cols):
        assert isinstance(new_data_cols, list) and all([isinstance(dcol, str) for dcol in new_data_cols]), \
            f"New 'data_cols' attribute must be a list of strings of the same length as the current 'data_cols' ({len(self._data_cols)}), got {new_data_cols}."
        self._df.rename(columns={old_col: new_col for old_col, new_col in zip(self._data_cols, new_data_cols)}, errors='raise', inplace=True)
        self._data_cols = new_data_cols

    @property
    def sigma_cols(self):
        return self._sigma_cols

    @sigma_cols.setter
    def sigma_cols(self, new_sigma_cols):
        assert isinstance(new_sigma_cols, list) and all([(scol is None) or isinstance(scol, str) for scol in new_sigma_cols]), \
            f"New 'sigma_cols' attribute must be a list of strings or Nones of the same length as the current 'sigma_cols' ({len(self._sigma_cols)}), got {new_sigma_cols}."
        self._df.rename(columns={old_col: new_col for old_col, new_col in zip(self._sigma_cols, new_sigma_cols) if (old_col is not None) and (new_col is not None)}, errors='raise')
        self._sigma_cols = new_sigma_cols

    @property
    def num_components(self):
        return len(self._data_cols)

    @property
    def df(self):
        return self._df

    @property
    def num_observations(self):
        return self._df.shape[0]

    @property
    def data(self):
        return self._df.loc[:, self._data_cols]

    @data.setter
    def data(self, new_data):
        self._df.loc[:, self._data_cols] = new_data

    @property
    def sigmas(self):
        if not any(self._sigma_cols):
            raise ValueError("No uncertainty columns present to return.")
        return self._df.loc[:, self._sigma_cols]

    @sigmas.setter
    def sigmas(self, new_sigma):
        if not any(self._sigma_cols):
            raise ValueError("No uncertainty columns present to set.")
        self.df.loc[:, self._sigma_cols] = new_sigma

    @property
    def time(self):
        return self._df.index

    def copy(self, only_data=False, src=None):
        new_name = deepcopy(self._src) if src is None else src
        if not only_data:
            return Timeseries(self._df.copy(), new_name, deepcopy(self._data_unit), deepcopy(self._data_cols), deepcopy(self._sigma_cols))
        else:
            return Timeseries(self._df[self._data_cols].copy(), new_name, deepcopy(self._data_unit), deepcopy(self._data_cols), None)

    def mask_out(self, dcol):
        icol = self._data_cols.index(dcol)
        scol = self._sigma_cols[icol]
        self._df[dcol] = np.NaN
        self._df[dcol] = self._df[dcol].astype(pd.SparseDtype(dtype=float))
        if scol is not None:
            self._df[scol] = np.NaN
            self._df[scol] = self._df[scol].astype(pd.SparseDtype(dtype=float))

    def _prepare_math(self, other, operation):
        # check for same type
        if not isinstance(other, Timeseries):
            raise TypeError("Unsupported operand type for {}: {} and {}".format(operation, Timeseries, type(other)))
        # check for same dimensions
        if len(self.data_cols) != len(other.data_cols):
            raise ValueError(("Timeseries math problem: conflicting number of data columns (" +
                              "[ " + "'{}' "*len(self.data_cols) + "] and " +
                              "[ " + "'{}' "*len(other.data_cols) + "])").format(*self.data_cols, *other.data_cols))
        # get intersection of time indices
        out_time = self.time.intersection(other.time, sort=None)
        # check compatible units (+-) or define new one (*/)
        # define new src and data_cols
        if operation in ["+", "-"]:
            if self.data_unit != other.data_unit:
                raise ValueError("Timeseries math problem: conflicting data units '{:s}' and '{:s}'".format(self.data_unit, other.data_unit))
            else:
                out_unit = self.data_unit
                out_src = "{:s}{:s}{:s}".format(self.src, operation, other.src)
                out_data_cols = ["{:s}{:s}{:s}".format(lcol, operation, rcol) for lcol, rcol in zip(self.data_cols, other.data_cols)]
        elif operation in ["*", "/"]:
            out_unit = "({:s}){:s}({:s})".format(self.data_unit, operation, other.data_unit)
            out_src = "({:s}){:s}({:s})".format(self.src, operation, other.src)
            out_data_cols = ["({:s}){:s}({:s})".format(lcol, operation, rcol) for lcol, rcol in zip(self.data_cols, other.data_cols)]
        else:
            raise NotImplementedError("Timeseries math problem: unknown operation {:s}".format(operation))
        # return data unit and column names
        return out_src, out_unit, out_data_cols, out_time

    def __add__(self, other):
        out_src, out_unit, out_data_cols, out_time = self._prepare_math(other, '+')
        out_dict = {newcol: self.df.loc[out_time, lcol].values + other.df.loc[out_time, rcol].values for newcol, lcol, rcol in zip(out_data_cols, self.data_cols, other.data_cols)}
        return Timeseries(pd.DataFrame(data=out_dict, index=out_time), out_src, out_unit, out_data_cols)

    def __sub__(self, other):
        out_src, out_unit, out_data_cols, out_time = self._prepare_math(other, '-')
        out_dict = {newcol: self.df.loc[out_time, lcol].values - other.df.loc[out_time, rcol].values for newcol, lcol, rcol in zip(out_data_cols, self.data_cols, other.data_cols)}
        return Timeseries(pd.DataFrame(data=out_dict, index=out_time), out_src, out_unit, out_data_cols)

    def __mul__(self, other):
        out_src, out_unit, out_data_cols, out_time = self._prepare_math(other, '*')
        out_dict = {newcol: self.df.loc[out_time, lcol].values * other.df.loc[out_time, rcol].values for newcol, lcol, rcol in zip(out_data_cols, self.data_cols, other.data_cols)}
        return Timeseries(pd.DataFrame(data=out_dict, index=out_time), out_src, out_unit, out_data_cols)

    def __truediv__(self, other):
        out_src, out_unit, out_data_cols, out_time = self._prepare_math(other, '/')
        out_dict = {newcol: self.df.loc[out_time, lcol].values / other.df.loc[out_time, rcol].values for newcol, lcol, rcol in zip(out_data_cols, self.data_cols, other.data_cols)}
        return Timeseries(pd.DataFrame(data=out_dict, index=out_time), out_src, out_unit, out_data_cols)

    def get_arch(self):
        return {}

    @classmethod
    def from_fit(cls, data_unit, data_cols, fit):
        df_data = {dcol: fit["fit"][:, icol] for icol, dcol in enumerate(data_cols)}
        sigma_cols = None
        if fit["sigma"] is not None:
            sigma_cols = [dcol + "_sigma" for dcol in data_cols]
            df_data.update({scol: fit["sigma"][:, icol] for icol, scol in enumerate(sigma_cols)})
        df = pd.DataFrame(data=df_data, index=fit["time"])
        return cls(df, "fitted", data_unit, data_cols, sigma_cols)


class GipsyTimeseries(Timeseries):
    """
    Timeseries subclass for GNSS measurements in JPL's Gipsy `.tseries` file format.
    """
    def __init__(self, path):
        self._path = path
        time = pd.to_datetime(pd.read_csv(self._path, delim_whitespace=True, header=None, usecols=[11, 12, 13, 14, 15, 16],
                                          names=['year', 'month', 'day', 'hour', 'minute', 'second'])).to_frame(name='time')
        data = pd.read_csv(self._path, delim_whitespace=True, header=None, usecols=[1, 2, 3, 4, 5, 6], names=['east', 'north', 'up', 'east_sigma', 'north_sigma', 'up_sigma'])
        df = time.join(data).drop_duplicates(subset='time').set_index('time')
        super().__init__(dataframe=df, src='.tseries', data_unit='m',
                         data_cols=['east', 'north', 'up'], sigma_cols=['east_sigma', 'north_sigma', 'up_sigma'])

    def get_arch(self):
        return {"type": "GipsyTimeseries",
                "kw_args": {"path": self._path}}


class Model():
    """
    General class that defines what a model can have as an input and output.
    Defaults to a linear model.
    """
    def __init__(self, num_parameters, regularize=False):
        self.num_parameters = num_parameters
        self.regularize = regularize
        self.is_fitted = False
        self.parameters = None
        self.cov = None

    def get_arch(self):
        raise NotImplementedError()

    def get_mapping(self, timevector):
        raise NotImplementedError()

    def read_parameters(self, parameters, cov):
        assert self.num_parameters == parameters.shape[0], "Read-in parameters have different size than the instantiated model."
        self.parameters = parameters
        if cov is not None:
            assert self.num_parameters == cov.shape[0] == cov.shape[1], "Covariance matrix must have same number of entries than parameters."
            self.cov = cov
        self.is_fitted = True

    def evaluate(self, timevector):
        if not self.is_fitted:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        mapping_matrix = self.get_mapping(timevector=timevector)
        fit = mapping_matrix @ self.parameters
        fit_sigma = mapping_matrix @ np.sqrt(self.cov.diagonal(offset=0, axis1=0, axis2=1).T) if self.cov is not None else None
        return {"time": timevector, "fit": fit, "sigma": fit_sigma}


class Step(Model):
    """
    Step functions at given times.
    """
    def __init__(self, steptimes, regularize=False):
        super().__init__(num_parameters=len(steptimes), regularize=regularize)
        self._steptimes = steptimes
        self.timestamps = [pd.Timestamp(step) for step in self._steptimes]
        self.timestamps.sort()

    def get_arch(self):
        return {"type": "Step",
                "kw_args": {"steptimes": self._steptimes,
                            "regularize": self.regularize}}

    def _update_from_steptimes(self):
        self.timestamps = [pd.Timestamp(step) for step in self._steptimes]
        self.timestamps.sort()
        self.num_parameters = len(self.timestamps)
        self.is_fitted = False
        self.parameters = None
        self.cov = None

    def add_step(self, step):
        if step in self._steptimes:
            Warning("Step {:s} already present.".format(step))
        else:
            self._steptimes.append(step)
            self._update_from_steptimes()

    def remove_step(self, step):
        try:
            self._steptimes.remove(step)
            self._update_from_steptimes()
        except ValueError:
            Warning("Step {:s} not present.".format(step))

    def get_mapping(self, timevector):
        coefs = np.array(timevector.values.reshape(-1, 1) >= pd.DataFrame(data=self.timestamps, columns=["steptime"]).values.reshape(1, -1), dtype=int)
        return coefs


class Polynomial(Model):
    """
    Polynomial of given order.

    `timeunit` can be the following (see https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html#datetime-units):
        `Y`, `M`, `W`, `D`, `h`, `m`, `s`, `ms`, `us`, `ns`, `ps`, `fs`, `as`
    """
    def __init__(self, order, starttime=None, timeunit='Y', regularize=False):
        super().__init__(num_parameters=order + 1, regularize=regularize)
        self.order = order
        self.starttime = starttime
        self.timeunit = timeunit

    def get_arch(self):
        return {"type": "Polynomial",
                "kw_args": {"order": self.order,
                            "starttime": self.starttime,
                            "timeunit": self.timeunit,
                            "regularize": self.regularize}}

    def get_mapping(self, timevector):
        # timevector as a Numpy column vector relative to starttime in the desired unit
        dt = tvec_to_numpycol(timevector, starttime=self.starttime, timeunit=self.timeunit)
        # create Numpy-style 2-D array of coefficients
        coefs = np.ones((timevector.size, self.order + 1))
        if self.order >= 1:
            # the exponents increase by column
            exponents = np.arange(1, self.order + 1).reshape(1, -1)
            # broadcast to all coefficients
            coefs[:, 1:] = dt ** exponents
        return coefs


class Sinusoidal(Model):
    """
    Sinusoidal of given frequency. Estimates amplitude and phase.

    `timeunit` can be the following (see https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html#datetime-units):
        `Y`, `M`, `W`, `D`, `h`, `m`, `s`, `ms`, `us`, `ns`, `ps`, `fs`, `as`
    """
    def __init__(self, period, starttime=None, timeunit='Y', regularize=False):
        super().__init__(num_parameters=2, regularize=regularize)
        self.period = period
        self.starttime = starttime
        self.timeunit = timeunit

    def get_arch(self):
        return {"type": "Sinusoidal",
                "kw_args": {"period": self.period,
                            "starttime": self.starttime,
                            "timeunit": self.timeunit,
                            "regularize": self.regularize}}

    def get_mapping(self, timevector):
        # timevector as a Numpy column vector relative to starttime in the desired unit
        dt = tvec_to_numpycol(timevector, starttime=self.starttime, timeunit=self.timeunit)
        # create Numpy-style 2-D array of coefficients
        phase = 2*np.pi * dt / self.period
        coefs = np.concatenate([np.sin(phase), np.cos(phase)], axis=1)
        return coefs

    @property
    def amplitude(self):
        if not self.is_fitted:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        return np.sqrt(np.sum(self.parameters ** 2))

    @property
    def phase(self):
        if not self.is_fitted:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        return np.arctan2(self.parameters[1], self.parameters[0])


class Logarithmic(Model):
    """
    Geophiscal logarithmic `ln(1 + dt/tau)` with a given time constant and time window.
    """
    def __init__(self, tau, starttime=None, endtime=None, timeunit='Y', regularize=False):
        super().__init__(num_parameters=1, regularize=regularize)
        self.tau = tau
        self.starttime = starttime
        self.endtime = endtime
        self.timeunit = timeunit

    def get_arch(self):
        return {"type": "Logarithmic",
                "kw_args": {"tau": self.tau,
                            "starttime": self.starttime,
                            "endtime": self.endtime,
                            "timeunit": self.timeunit,
                            "regularize": self.regularize}}

    def get_mapping(self, timevector):
        dt = tvec_to_numpycol(timevector, starttime=self.starttime, timeunit=self.timeunit)
        endtime = timevector[-1] if self.endtime is None else pd.Timestamp(self.endtime)
        slice_middle = np.all((dt.squeeze() > 0, timevector < endtime), axis=0)
        slice_end = np.all((dt.squeeze() > 0, timevector >= endtime), axis=0)
        coefs = np.zeros_like(dt)
        if slice_middle.any():
            coefs[slice_middle] = np.log1p(dt[slice_middle] / self.tau)
        if slice_end.any():
            coefs[slice_end] = np.log1p(dt[slice_end][0] / self.tau)
        return coefs


def unwrap_dict_and_ts(func):
    @wraps(func)
    def wrapper(data, *args, **kw_args):
        if not isinstance(data, dict):
            data = {'ts': data}
            was_dict = False
        else:
            was_dict = True
        out = {}
        additional_output = {}
        # loop over components
        for comp, ts in data.items():
            if isinstance(ts, Timeseries):
                array = ts.data.values
            elif isinstance(ts, pd.DataFrame):
                array = ts.values
            elif isinstance(ts, np.ndarray):
                array = ts
            else:
                raise TypeError(f"Cannot unwrap object of type {type(ts)}")
            func_output = func(array, *args, **kw_args)
            if isinstance(func_output, tuple):
                result = func_output[0]
                try:
                    additional_output[comp] = func_output[1:]
                except IndexError:
                    additional_output[comp] = None
            else:
                result = func_output
                additional_output[comp] = None
            # save results
            if isinstance(ts, Timeseries):
                out[comp] = ts.copy(only_data=True, src=func.__name__)
                out[comp].data = result
            elif isinstance(ts, pd.DataFrame):
                out[comp] = ts.copy()
                out[comp].values = result
            else:
                out[comp] = result
        has_additional_output = False if all([elem is None for elem in additional_output.values()]) else True
        if not was_dict:
            out = out['ts']
            additional_output = additional_output['ts']
        if has_additional_output:
            return out, additional_output
        else:
            return out
    return wrapper


@unwrap_dict_and_ts
def median(array, kernel_size):
    """
    Computes the median filter (ignoring NaNs) column-wise,
    either by calling a Numpy function iteratively or by using
    the precompiled Fortran code.
    """
    try:
        from compiled_utils import maskedmedfilt2d
        filtered = maskedmedfilt2d(array, ~np.isnan(array), kernel_size)
        filtered[np.isnan(array)] = np.NaN
    except ImportError:
        num_obs = array.shape[0]
        array = array.reshape(num_obs, 1 if array.ndim == 1 else -1)
        filtered = np.NaN * np.empty(array.shape)
        # Run filtering while suppressing warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice encountered')
            # Beginning region
            halfWindow = 0
            for i in range(kernel_size // 2):
                filtered[i, :] = np.nanmedian(array[i-halfWindow:i+halfWindow+1, :], axis=0)
                halfWindow += 1
            # Middle region
            halfWindow = kernel_size // 2
            for i in range(halfWindow, num_obs - halfWindow):
                filtered[i, :] = np.nanmedian(array[i-halfWindow:i+halfWindow+1, :], axis=0)
            # Ending region
            halfWindow -= 1
            for i in range(num_obs - halfWindow, num_obs):
                filtered[i, :] = np.nanmedian(array[i-halfWindow:i+halfWindow+1, :], axis=0)
                halfWindow -= 1
    return filtered


@unwrap_dict_and_ts
def common_mode(array, method, n_components=1, plot=False):
    """
    Computes the common mode noise with the given method. Input should already be a residual.
    """
    # fill NaNs with white Gaussian noise
    array_nanmean = np.nanmean(array, axis=0)
    array_nansd = np.nanstd(array, axis=0)
    array_nanind = np.isnan(array)
    for icol in range(array.shape[1]):
        array[array_nanind[:, icol], icol] = array_nansd[icol] * np.random.randn(array_nanind[:, icol].sum()) + array_nanmean[icol]
    # decompose using the specified solver
    if method == 'pca':
        from sklearn.decomposition import PCA
        decomposer = PCA(n_components=n_components, whiten=True)
    elif method == 'ica':
        from sklearn.decomposition import FastICA
        decomposer = FastICA(n_components=n_components, whiten=True)
    else:
        raise NotImplementedError(f"Cannot estimate the common mode error using the '{method}' method.")
    # extract temporal component and build model
    temporal = decomposer.fit_transform(array)
    model = decomposer.inverse_transform(temporal)
    # reduce to where original timeseries were not NaNs and return
    model[array_nanind] = np.NaN
    if plot:
        spatial = decomposer.components_
        return model, temporal, spatial
    else:
        return model


def clean(station, ts_in, reference, ts_out=None, residual_out=None,
          std_thresh=config.getfloat('clean', 'std_thresh', fallback=None),
          std_outlier=config.getfloat('clean', 'std_outlier', fallback=None),
          min_obs=config.getint('clean', 'min_obs', fallback=0),
          min_clean_obs=config.getint('clean', 'min_obs', fallback=0), **reference_callable_args):
    # check if we're modifying in-place or copying
    if ts_out is None:
        ts = station[ts_in]
    else:
        ts = station[ts_in].copy(only_data=True, src='clean')
    # check if we have a reference time series or need to calculate one
    # in the latter case, the input is name of function to call
    if not (isinstance(reference, Timeseries) or isinstance(reference, str) or callable(reference)):
        raise TypeError(f"'reference' has to either be a the name of a timeseries in the station, or the name of a function, got {type(reference)}.")
    if isinstance(reference, str):
        # get reference time series
        ts_ref = station[reference]
    elif callable(reference):
        ts_ref = reference(ts, **reference_callable_args)
    # check that both timeseries have the same data columns
    if not ts_ref.data_cols == ts.data_cols:
        raise ValueError(f"Reference time series has to have the same data columns as input time series, but got {ts_ref.data_cols} and {ts.data_cols}.")
    for dcol in ts.data_cols:
        # check for minimum number of observations
        if ts[dcol].count() < min_obs:
            ts.mask_out(dcol)
            continue
        # compute residuals
        if (std_outlier is not None) or (std_thresh is not None):
            residual = ts[dcol].values - ts_ref[dcol].values
            sd = np.nanstd(residual)
        # check for and remove outliers
        if std_outlier is not None:
            mask = ~np.isnan(residual)
            mask[mask] &= np.abs(residual[mask]) > std_outlier * sd
            ts[dcol][mask] = np.NaN
            residual = ts[dcol].values - ts_ref[dcol].values
            sd = np.nanstd(residual)
        # check for minimum number of clean observations
        if ts[dcol].count() < min_clean_obs:
            ts.mask_out(dcol)
            continue
        # check if total standard deviation is still too large
        if (std_thresh is not None) and (sd > std_thresh):
            ts.mask_out(dcol)
    # if we made a copy, add it to the station, otherwise we're already done
    if ts_out is not None:
        station.add_timeseries(ts_out, ts)


def linear_regression(ts, models, formal_covariance=config.getboolean('fit', 'formal_covariance', fallback=False)):
    """
    numpy.linalg wrapper for a linear least squares solver
    """
    mapping_matrices = []
    # get mapping matrices
    for (mdl_description, model) in models.items():
        mapping_matrices.append(model.get_mapping(ts.time))
    mapping_matrices = np.hstack(mapping_matrices)
    num_time, num_params = mapping_matrices.shape
    num_components = len(ts.data_cols)
    # perform fit and estimate formal covariance (uncertainty) of parameters
    params = np.zeros((num_params, num_components))
    if formal_covariance:
        cov = np.zeros((num_params, num_params, num_components))
    for i in range(num_components):
        if ts.sigma_cols[i] is None:
            GtWG = mapping_matrices.T @ mapping_matrices
            GtWd = mapping_matrices.T @ ts.df[ts.data_cols[i]].values.reshape(-1, 1)
        else:
            GtW = dmultr(mapping_matrices.T, 1/ts.df[ts.sigma_cols[i]].values**2)
            GtWG = GtW @ mapping_matrices
            GtWd = GtW @ ts.df[ts.data_cols[i]].values.reshape(-1, 1)
        params[:, i] = np.linalg.lstsq(GtWG, GtWd, rcond=None)[0].squeeze()
        if formal_covariance:
            cov[:, :, i] = np.linalg.pinv(GtWG)
    # separate parameters back to models
    i = 0
    fitted_params = {}
    for (mdl_description, model) in models.items():
        fitted_params[mdl_description] = (params[i:i+model.num_parameters, :], cov[i:i+model.num_parameters, i:i+model.num_parameters, :] if formal_covariance else None)
        i += model.num_parameters
    return fitted_params


def ridge_regression(ts, models, penalty=config.getfloat('fit', 'l2_penalty'), formal_covariance=config.getboolean('fit', 'formal_covariance', fallback=False)):
    """
    numpy.linalg wrapper for a linear L2-regularized least squares solver
    """
    from scipy.sparse import diags
    mapping_matrices = []
    reg_diag = []
    # get mapping and regularization matrices
    for (mdl_description, model) in models.items():
        mapping_matrices.append(model.get_mapping(ts.time))
        reg_diag.extend([model.regularize for _ in range(model.num_parameters)])
    G = np.hstack(mapping_matrices)
    reg = diags(reg_diag, dtype=float) * penalty
    num_time, num_params = G.shape
    num_components = len(ts.data_cols)
    # perform fit and estimate formal covariance (uncertainty) of parameters
    params = np.zeros((num_params, num_components))
    if formal_covariance:
        cov = np.zeros((num_params, num_params, num_components))
    for i in range(num_components):
        if ts.sigma_cols[i] is None:
            GtWG = G.T @ G
            GtWd = G.T @ ts.df[ts.data_cols[i]].values.reshape(-1, 1)
        else:
            GtW = dmultr(G.T, 1/ts.df[ts.sigma_cols[i]].values**2)
            GtWG = GtW @ G
            GtWd = GtW @ ts.df[ts.data_cols[i]].values.reshape(-1, 1)
        GtWGreg = GtWG + reg
        params[:, i] = np.linalg.lstsq(GtWGreg, GtWd, rcond=None)[0].squeeze()
        if formal_covariance:
            cov[:, :, i] = np.linalg.pinv(GtWGreg)
    # separate parameters back to models
    i = 0
    fitted_params = {}
    for (mdl_description, model) in models.items():
        fitted_params[mdl_description] = (params[i:i+model.num_parameters, :], cov[i:i+model.num_parameters, i:i+model.num_parameters, :] if formal_covariance else None)
        i += model.num_parameters
    return fitted_params


def tvec_to_numpycol(timevector, starttime=None, timeunit='D'):
    # get reference time
    if starttime is None:
        starttime = timevector[0]
    else:
        starttime = pd.Timestamp(starttime)
    # return Numpy column vector
    return ((timevector - starttime) / np.timedelta64(1, timeunit)).values.reshape(-1, 1)


def dmultl(dvec, mat):
    '''Left multiply with a diagonal matrix. Faster.

    Args:

        * dvec    -> Diagonal matrix represented as a vector
        * mat     -> Matrix

    Returns:

        * res    -> dot (diag(dvec), mat)'''

    res = (dvec*mat.T).T
    return res


def dmultr(mat, dvec):
    '''Right multiply with a diagonal matrix. Faster.

    Args:

        * dvec    -> Diagonal matrix represented as a vector
        * mat     -> Matrix

    Returns:

        * res     -> dot(mat, diag(dvec))'''

    res = dvec*mat
    return res


def parallelize(func, iterable, num_threads=None, chunksize=1):
    if num_threads is None:
        num_threads = config.getint("general", "num_threads", fallback=0)
    if num_threads > 0:
        with Pool(num_threads) as p:
            for result in p.imap(func, iterable, chunksize):
                yield result
    else:
        for parameter in iterable:
            yield func(parameter)


def _okada_get_displacements(station_and_parameters):
    # unpack inputs
    stations, eq = station_and_parameters
    # rotate from relative lat, lon, alt to xyz
    strike_rad = eq['strike']*np.pi/180
    R = np.array([[np.cos(strike_rad),  np.sin(strike_rad), 0],
                  [np.sin(strike_rad), -np.cos(strike_rad), 0],
                  [0, 0, 1]])
    stations = stations @ R
    # get displacements in xyz-frame
    disp = np.zeros_like(stations)
    for i in range(stations.shape[0]):
        success, u, grad_u = dc3d0(eq['alpha'], stations[i, :], eq['depth'], eq['dip'], eq['potency'])
        if success == 0:
            disp[i, :] = u / 10**12  # output is now in mm
        else:
            Warning("success = {:d} for station {:d}".format(success, i))
    # transform back to lat, lon, alt
    # yes this is the same matrix
    disp = disp @ R
    return disp


def _okada_get_cumdisp(time_and_station):
    eq_times, stat_time, station_disp = time_and_station
    steptimes = []
    for itime in range(len(stat_time) - 1):
        disp = station_disp[(eq_times > stat_time[itime]) & (eq_times <= stat_time[itime + 1]), :]
        cumdisp = np.sum(np.linalg.norm(disp, axis=1), axis=None)
        if cumdisp >= config.getfloat('catalog_prior', 'threshold'):
            steptimes.append(str(stat_time[itime]))
    return steptimes


def okada_prior(network, catalog_path, target_timeseries=None):
    stations_lla = np.array([station.location for station in network])
    # convert height from m to km
    stations_lla[:, 2] /= 1000
    # load earthquake catalog
    catalog = pd.read_csv(catalog_path, header=0, parse_dates=[[0, 1]])
    eq_times = catalog['Date_Origin_Time(JST)']
    eq_lla = catalog[['Latitude(°)', 'Longitude(°)',  'MT_Depth(km)']].values
    eq_lla[:, 2] *= -1
    n_eq = eq_lla.shape[0]
    # relative position in lla
    stations_rel = [np.array(stations_lla - eq_lla[i, :].reshape(1, -1)) for i in range(n_eq)]
    # transform to xyz space, coarse approximation, ignores z-component of stations
    for i in range(n_eq):
        stations_rel[i][:, 0] *= 111.13292 - 0.55982*np.cos(2*eq_lla[i, 0]*np.pi/180)
        stations_rel[i][:, 1] *= 111.41284*np.cos(eq_lla[i, 0]*np.pi/180)
        stations_rel[i][:, 2] = 0

    # compute station displacements
    parameters = ((stations_rel[i], {'alpha': config.getfloat('catalog_prior', 'alpha'), 'lat': eq_lla[i, 0], 'lon': eq_lla[i, 1],
                                     'depth': -eq_lla[i, 2], 'strike': float(catalog['Strike'][i].split(';')[0]),
                                     'dip': float(catalog['Dip'][i].split(';')[0]),
                                     'potency': [catalog['Mo(Nm)'][i]/config.getfloat('catalog_prior', 'mu'), 0, 0, 0]})
                  for i in range(n_eq))
    station_disp = np.zeros((n_eq, stations_lla.shape[0], 3))
    for i, result in enumerate(tqdm(parallelize(_okada_get_displacements, parameters, chunksize=100),
                                    ascii=True, total=n_eq, desc="Simulating Earthquake Displacements", unit="eq")):
        station_disp[i, :, :] = result

    # add steps to station timeseries if they exceed the threshold
    target_timeseries = config.get('catalog_prior', 'timeseries') if target_timeseries is None else target_timeseries
    station_names = list(network.stations.keys())
    cumdisp_parameters = ((eq_times, network.stations[stat_name].timeseries[target_timeseries].time.values, station_disp[:, istat, :])
                          for istat, stat_name in enumerate(station_names))
    for i, result in enumerate(tqdm(parallelize(_okada_get_cumdisp, cumdisp_parameters),
                                    ascii=True, total=len(network.stations), desc="Adding steps where necessary", unit="station")):
        network.stations[station_names[i]].add_local_model(target_timeseries, config.get('catalog_prior', 'model'),
                                                           Step(steptimes=result, regularize=config.getboolean('catalog_prior', 'regularize')))


if __name__ == "__main__":
    # net = Network.from_json(path="net_arch.json", add_default_local_models=False)
    # net = Network.from_json(path="net_arch_catalog1mm.json")
    net = Network.from_json(path="net_arch_boso_okada3mm.json", add_default_local_models=False)
    net.call_func_ts_return(median, ts_in='GNSS', ts_out='filtered', kernel_size=7)
    net.call_func_no_return(clean, ts_in='GNSS', reference='filtered', ts_out='clean')
    for station in net:
        station['residual'] = station['clean'] - station['filtered']
        del station['filtered']
    # net.graphical_cme(ts_in='residual', ts_out='common', method='pca')
    net.call_netwide_func(common_mode, ts_in='residual', ts_out='common', method='pca')
    for station in net:
        del station['residual']
        station.add_timeseries('final', station['clean'] - station['common'],
                               override_data_cols=station['GNSS'].data_cols, add_models=net.default_local_models)
        del station['common']
    net.add_default_local_models('final')
    # okada_prior(net, "data/nied_fnet_catalog.txt", target_timeseries='final')
    net.add_unused_local_models('final')
    net.fit("final", solver="ridge_regression", penalty=10, formal_covariance=False)
    net.evaluate("final", output_description="model")
    for i, station in enumerate(net):
        station.add_timeseries('residual', station['final'] - station['model'],
                               override_data_cols=station['GNSS'].data_cols)
    net.to_json(path="arch_out.json")
    net.gui(timeseries=['final', 'residual'], mean=True, std=True, n_observations=True, std_outlier=5)
