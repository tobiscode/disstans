import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from copy import deepcopy
from tqdm import tqdm
from warnings import warn

from . import defaults
from . import timeseries as geonat_ts
from . import model as geonat_models
from . import solvers as geonat_solvers
from . import processing as geonat_processing
from .processing import common_mode
from .timeseries import Timeseries
from .station import Station
from .tools import parallelize


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

    def __repr__(self):
        info = f"Network {self.name}\n" + \
               f"Stations:\n{[key for key in self.stations]}\n" + \
               f"Global Models:\n{[key for key in self.global_models]}"
        return info

    def __getitem__(self, name):
        if name not in self.stations:
            raise KeyError(f"No station '{name}' present.")
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
            raise ValueError(f"No data found in network '{self.name}' for timeseries '{ts_description}'.")
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
            warn(f"Overwriting station '{name}'.", category=RuntimeWarning)
        self.stations[name] = station

    def remove_station(self, name):
        if name not in self.stations:
            warn(f"Cannot find station '{name}', couldn't delete.", category=RuntimeWarning)
        else:
            del self.stations[name]

    def add_global_model(self, description, model):
        if not isinstance(description, str):
            raise TypeError("Cannot add new global model: 'description' is not a string.")
        if description in self.global_models:
            warn(f"Overwriting global model '{description}'.", category=RuntimeWarning)
        self.global_models[description] = model

    def remove_global_model(self, description):
        if description not in self.global_models:
            warn(f"Cannot find global model '{description}', couldn't delete.", category=RuntimeWarning)
        else:
            del self.global_models[description]

    @classmethod
    def from_json(cls, path, add_default_local_models=True, station_kw_args={}, timeseries_kw_args={}):
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
                warn(f"Skipped station '{station_name}' because location information is missing.")
                continue
            station = Station(name=station_name, location=station_loc, **station_kw_args)
            # add timeseries to station
            for ts_description, ts_cfg in station_cfg["timeseries"].items():
                ts = getattr(geonat_ts, ts_cfg["type"])(**ts_cfg["kw_args"], **timeseries_kw_args)
                station.add_timeseries(description=ts_description, timeseries=ts)
                # add default local models to station
                if add_default_local_models:
                    for model_description, model_cfg in net.default_local_models.items():
                        local_copy = deepcopy(model_cfg)
                        mdl = getattr(geonat_models, local_copy["type"])(**local_copy["kw_args"])
                        station.add_local_model(ts_description=ts_description, model_description=model_description, model=mdl)
            # add specific local models to station and timeseries
            for ts_description, ts_model_dict in station_cfg["models"].items():
                if ts_description in station.timeseries:
                    for model_description, model_cfg in ts_model_dict.items():
                        mdl = getattr(geonat_models, model_cfg["type"])(**model_cfg["kw_args"])
                        station.add_local_model(ts_description=ts_description, model_description=model_description, model=mdl)
                else:
                    station.unused_models.update({ts_description: ts_model_dict})
            # add to network
            net.add_station(name=station_name, station=station)
        # add global models
        for model_description, model_cfg in net_arch["global_models"].items():
            mdl = getattr(geonat_models, model_cfg["type"])(**model_cfg["kw_args"])
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
            # need to remove all models that are actually default models
            for model_description, mdl in self.default_local_models.items():
                for ts_description in stat_arch["models"]:
                    if (model_description in stat_arch["models"][ts_description].keys()) and (mdl == stat_arch["models"][ts_description][model_description]):
                        del stat_arch["models"][ts_description][model_description]
            # now we can append it to the main json
            net_arch["stations"].update({stat_name: stat_arch})
        # add global model representations
        for model_description, mdl in self.global_models.items():
            net_arch["global_models"].update({model_description: mdl.get_arch()})
        # write file
        json.dump(net_arch, open(path, mode='w'), indent=2, sort_keys=True)

    def add_default_local_models(self, ts_description, models=None):
        assert isinstance(ts_description, str), f"'ts_description' must be a string, got {type(ts_description)}."
        if models is None:
            local_models_subset = self.default_local_models
        else:
            if not isinstance(models, str) or not (isinstance(models, list) and all([isinstance(m, str) for m in models])):
                raise ValueError(f"'models' must be None, a string or a list of strings, got {models}.")
            if isinstance(models, str):
                models = [models]
            local_models_subset = {name: model for name, model in self.default_local_models.items() if name in models}
        for station in self:
            for model_description, model_cfg in local_models_subset.items():
                local_copy = deepcopy(model_cfg)
                mdl = getattr(geonat_models, local_copy["type"])(**local_copy["kw_args"])
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
            if hidden_ts in station.unused_models:
                if models is None:
                    local_models_subset = station.unused_models[hidden_ts]
                else:
                    local_models_subset = {name: model for name, model in station.unused_models[hidden_ts].items() if name in models}
                for model_description, model_cfg in local_models_subset.items():
                    local_copy = deepcopy(model_cfg)
                    mdl = getattr(geonat_models, local_copy["type"])(**local_copy["kw_args"])
                    station.add_local_model(ts_description=target_ts, model_description=model_description, model=mdl)

    def fit(self, ts_description, model_list=None, solver='linear_regression', **kw_args):
        assert isinstance(ts_description, str), f"'ts_description' must be string, got {type(ts_description)}."
        if isinstance(solver, str):
            solver = getattr(geonat_solvers, solver)
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

    def evaluate(self, ts_description, model_list=None, timevector=None, output_description=None, reuse=False):
        assert isinstance(ts_description, str), f"'ts_description' must be string, got {type(ts_description)}."
        if output_description is not None:
            assert isinstance(output_description, str), f"'output_description' must be a string, got {type(output_description)}."
        if reuse:
            assert output_description is not None, "When reusing a previous model evaluation, 'output_description' " \
                                                   "must be set (otherwise nothing would happen)."
        # should directly use fit from station timeseries, skip evaluation.
        # can only be used if the timeseries' timevector is used
        # (and wouldn't do anything if output_description is None)
        if reuse:
            assert timevector is None, "When reusing a previous model evaluation, 'timevector' has to be None"
            for station in self:
                stat_subset_models = station.models[ts_description] if model_list is None else {m: station.models[ts_description][m] for m in model_list}
                for imodel, model_description in enumerate(stat_subset_models):
                    try:
                        ts = station.fits[ts_description][model_description]
                    except KeyError as e:
                        raise KeyError(f"Station {station}, Timeseries '{ts_description}' or Model '{model_description}': "
                                       "Fit not found, cannout evaluate.").with_traceback(e.__traceback__) from e
                    if imodel == 0:
                        model_aggregate = ts
                    else:
                        model_aggregate += ts
                station.add_timeseries(output_description, model_aggregate, override_src='model', override_data_cols=station[ts_description].data_cols)
        # if not reusing, have to evaluate the models and add to timeseries' fit
        # and optionally add the aggregate timeseries to the station
        else:
            iterable_inputs = ((station.timeseries[ts_description].time if timevector is None else timevector,
                                station.models[ts_description] if model_list is None else {m: station.models[ts_description][m] for m in model_list})
                               for station in self)
            station_names = list(self.stations.keys())
            for i, result in enumerate(tqdm(parallelize(self._evaluate_single_station, iterable_inputs),
                                            desc="Evaluating station models", total=len(self.stations), ascii=True, unit="station")):
                stat_name = station_names[i]
                for imodel, (model_description, fit) in enumerate(result.items()):
                    ts = self[stat_name].add_fit(ts_description, model_description, fit)
                    if output_description is not None:
                        if imodel == 0:
                            model_aggregate = ts
                        else:
                            model_aggregate += ts
                if output_description is not None:
                    self[stat_name].add_timeseries(output_description, model_aggregate, override_src='model', override_data_cols=self[stat_name][ts_description].data_cols)

    @staticmethod
    def _evaluate_single_station(parameter_tuple):
        station_time, station_models = parameter_tuple
        fit = {}
        for model_description, model in station_models.items():
            fit[model_description] = model.evaluate(station_time)
        return fit

    def call_func_ts_return(self, func, ts_in, ts_out=None, **kw_args):
        if not callable(func):
            if isinstance(func, str):
                try:
                    func = getattr(geonat_processing, func)
                except AttributeError as e:
                    raise AttributeError(f"'{func}' can not be found as a function in geonat.processing.").with_traceback(e.__traceback__) from e
            else:
                raise RuntimeError(f"'{func}' needs to be a function or a string representation thereof (if loaded from geonat.processing).")
        assert isinstance(ts_in, str), f"'ts_in' must be string, got {type(ts_in)}."
        if ts_out is None:
            ts_out = ts_in
        else:
            assert isinstance(ts_out, str), f"'ts_out' must be None or a string, got {type(ts_out)}."
        iterable_inputs = ((func, station, ts_in, kw_args) for station in self)
        station_names = list(self.stations.keys())
        for i, result in enumerate(tqdm(parallelize(self._single_call_func_ts_return, iterable_inputs),
                                        desc=f"Processing station timeseries with '{func.__name__}'", total=len(self.stations), ascii=True, unit="station")):
            self[station_names[i]].add_timeseries(ts_out, result)

    @staticmethod
    def _single_call_func_ts_return(parameter_tuple):
        func, station, ts_description, kw_args = parameter_tuple
        ts_return = func(station.timeseries[ts_description], **kw_args)
        return ts_return

    def call_netwide_func(self, func, ts_in, ts_out=None, **kw_args):
        if not callable(func):
            if isinstance(func, str):
                try:
                    func = getattr(geonat_processing, func)
                except AttributeError as e:
                    raise AttributeError(f"'{func}' can not be found as a function in geonat.processing.").with_traceback(e.__traceback__) from e
            else:
                raise RuntimeError(f"'{func}' needs to be a function or a string representation thereof (if loaded from geonat.processing).")
        assert isinstance(ts_in, str), f"'ts_in' must be string, got {type(ts_in)}."
        if ts_out is None:
            ts_out = ts_in
        else:
            assert isinstance(ts_out, str), f"'ts_out' must be None or a string, got {type(ts_out)}."
        net_in = self.export_network_ts(ts_in)
        net_out = func(net_in, **kw_args)
        self.import_network_ts(ts_in if ts_out is None else ts_out, net_out)

    def call_func_no_return(self, func, **kw_args):
        if not callable(func):
            if isinstance(func, str):
                try:
                    func = getattr(geonat_processing, func)
                except AttributeError as e:
                    raise AttributeError(f"'{func}' can not be found as a function in geonat.processing.").with_traceback(e.__traceback__) from e
            else:
                raise RuntimeError(f"'{func}' needs to be a function or a string representation thereof (if loaded from geonat.processing).")
        for name, station in tqdm(self.stations.items(), desc=f"Calling function '{func.__name__}' on stations", ascii=True, unit="station"):
            func(station, **kw_args)

    def _create_map_figure(self, gui_settings):
        # get location data and projections
        stat_lats = [station.location[0] for station in self]
        stat_lons = [station.location[1] for station in self]
        proj_gui = getattr(ccrs, gui_settings["projection"])()
        proj_lla = ccrs.PlateCarree()
        # create figure and plot stations
        fig_map = plt.figure()
        ax_map = fig_map.add_subplot(projection=proj_gui)
        default_station_colors = ['b'] * len(stat_lats)
        stat_points = ax_map.scatter(stat_lons, stat_lats, linestyle='None', marker='.', transform=proj_lla, facecolor=default_station_colors, zorder=1000)
        # create underlay
        map_underlay = False
        if gui_settings["wmts_show"]:
            try:
                ax_map.add_wmts(gui_settings["wmts_server"], layer_name=gui_settings["wmts_layer"], alpha=gui_settings["wmts_alpha"])
                map_underlay = True
            except Exception as exc:
                print(exc)
        if gui_settings["coastlines_show"]:
            ax_map.coastlines(color="white" if map_underlay else "black", resolution=gui_settings["coastlines_res"])
        return fig_map, ax_map, proj_gui, proj_lla, default_station_colors, stat_points, stat_lats, stat_lons

    def graphical_cme(self, ts_in, ts_out=None, gui_kw_args={}, **cme_kw_args):
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
        fig_map, ax_map, proj_gui, proj_lla, default_station_colors, stat_points, stat_lats, stat_lons = self._create_map_figure(gui_settings)
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

    def gui(self, timeseries=None, model_list=None, sum_models=True, scalogram=None,
            verbose=False, analyze_kw_args={}, gui_kw_args={}):
        # create map and timeseries figures
        gui_settings = defaults["gui"].copy()
        gui_settings.update(gui_kw_args)
        fig_map, ax_map, proj_gui, proj_lla, default_station_colors, stat_points, stat_lats, stat_lons = self._create_map_figure(gui_settings)
        fig_ts = plt.figure()
        if scalogram is not None:
            assert isinstance(scalogram, dict) and all([key in scalogram for key in ['ts', 'model']]) \
                and all([isinstance(scalogram[key], str) for key in ['ts', 'model']]), \
                "If a scalogram plot is requested, 'scalogram' must be a dictionary with 'ts' and 'model' keys " \
                f"with string values of where to find the SplineSet model, got {scalogram}."
            scalo_ts = scalogram.pop('ts')
            scalo_model = scalogram.pop('model')
            fig_scalo = plt.figure()

        # define clicking function
        def update_timeseries(event):
            nonlocal analyze_kw_args, gui_settings
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
            for ts_description, ts in ts_to_plot.items():
                n_components += ts.num_components
                if len(analyze_kw_args) > 0:
                    self.stations[station_name].analyze_residuals(ts_description, **analyze_kw_args)
            # clear figure and add data
            fig_ts.clear()
            icomp = 0
            ax_ts = []
            if scalogram is not None:
                nonlocal fig_scalo
                t_left, t_right = None, None
            for its, (ts_description, ts) in enumerate(ts_to_plot.items()):
                for icol, (data_col, sigma_col) in enumerate(zip(ts.data_cols, ts.sigma_cols)):
                    # add axis
                    ax = fig_ts.add_subplot(n_components, 1, icomp + 1, sharex=None if icomp == 0 else ax_ts[0])
                    # plot uncertainty
                    if (sigma_col is not None) and (gui_settings["plot_sigmas"] > 0):
                        ax.fill_between(ts.time, ts.df[data_col] + gui_settings["plot_sigmas"] * ts.df[sigma_col],
                                        ts.df[data_col] - gui_settings["plot_sigmas"] * ts.df[sigma_col], facecolor='gray',
                                        alpha=gui_settings["plot_sigmas_alpha"], linewidth=0)
                    # plot data
                    ax.plot(ts.time, ts.df[data_col], marker='.', color='k', label="Data" if len(self[station_name].fits[ts_description]) > 0 else None)
                    # overlay models
                    fits_to_plot = {model_description: fit for model_description, fit in self[station_name].fits[ts_description].items()
                                    if (model_list is None) or (model_description in model_list)}
                    if sum_models:
                        fit_sum = np.zeros(ts.time.size)
                        fit_sum_sigma = np.zeros(ts.time.size)
                        for model_description, fit in fits_to_plot.items():
                            fit_sum += fit.df[fit.data_cols[icol]].values
                            if (fit.sigma_cols[icol] is not None) and (gui_settings["plot_sigmas"] > 0):
                                fit_sum_sigma += (fit.df[fit.sigma_cols[icol]].values)**2
                        if fit_sum_sigma.sum() > 0:
                            fit_sum_sigma = np.sqrt(fit_sum_sigma)
                            ax.fill_between(fit.time, fit_sum + gui_settings["plot_sigmas"] * fit_sum_sigma,
                                            fit_sum - gui_settings["plot_sigmas"] * fit_sum_sigma,
                                            alpha=gui_settings["plot_sigmas_alpha"], linewidth=0)
                        if np.abs(fit_sum).sum() > 0:
                            ax.plot(fit.time, fit_sum, label="Model")
                    else:
                        for model_description, fit in fits_to_plot.items():
                            if (fit.sigma_cols[icol] is not None) and (gui_settings["plot_sigmas"] > 0):
                                ax.fill_between(fit.time, fit.df[fit.data_cols[icol]] + gui_settings["plot_sigmas"] * fit.df[fit.sigma_cols[icol]],
                                                fit.df[fit.data_cols[icol]] - gui_settings["plot_sigmas"] * fit.df[fit.sigma_cols[icol]],
                                                alpha=gui_settings["plot_sigmas_alpha"], linewidth=0)
                            ax.plot(fit.time, fit.df[fit.data_cols[icol]], label=model_description)
                    ax.set_ylabel(f"{ts_description}\n{data_col} [{ts.data_unit}]")
                    ax.grid()
                    if len(self[station_name].fits[ts_description]) > 0:
                        ax.legend()
                    ax_ts.append(ax)
                    icomp += 1
                    if scalogram is not None:
                        t_left = ts.time[0] if t_left is None else min(ts.time[0], t_left)
                        t_right = ts.time[-1] if t_right is None else max(ts.time[-1], t_right)
            ax_ts[0].set_title(station_name)
            fig_ts.canvas.draw_idle()
            # get scalogram
            if scalogram is not None:
                try:
                    splset = self[station_name].models[scalo_ts][scalo_model]
                    if isinstance(fig_scalo, plt.Figure):
                        plt.close(fig_scalo)
                    fig_scalo, ax_scalo = splset.make_scalogram(t_left, t_right, **scalogram)
                    fig_scalo.show()
                except KeyError:
                    warn(f"Could not find scalogram model {scalo_model} in timeseries {scalo_ts} "
                         f"for station {station_name}.", category=RuntimeWarning)

        cid = fig_map.canvas.mpl_connect("button_press_event", update_timeseries)
        plt.show()
        fig_map.canvas.mpl_disconnect(cid)
