import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
import multiprocessing
import scipy.sparse as sparse
from copy import deepcopy
from configparser import ConfigParser
from tqdm import tqdm
from multiprocessing import Pool
from functools import wraps
from okada_wrapper import dc3d0wrapper as dc3d0
from pandas.plotting import register_matplotlib_converters
from sklearn.decomposition import PCA, FastICA
from warnings import warn
from scipy.special import comb, factorial

# see if we have compiled_utils
try:
    from compiled_utils import maskedmedfilt2d
except ImportError:
    COMPILED_UTILS = False
else:
    COMPILED_UTILS = True

# load scientific colormaps
CMAPS = {"rainbow": None, "seismic": None, "topography": None}
for cm_type, scm6_name, mpl_name in zip(CMAPS.keys(),
                                        ["batlow", "roma", "oleron"],  # from Scientific Colour Maps
                                        ["rainbow", "seismic", "terrain"]):  # matplotlib defaults
    cmap_path = f"/home/tkoehne/downloads/ScientificColourMaps6/{scm6_name}/{scm6_name}.txt"
    try:
        CMAPS[cm_type] = mpl.colors.LinearSegmentedColormap.from_list(scm6_name, np.loadtxt(cmap_path))
    except FileNotFoundError:
        warn(f"Scientific Colour Maps file {cmap_path} not found, defaulting to {mpl_name}.")
        CMAPS[cm_type] = plt.get_cmap(mpl_name)
del cmap_path

# preparational steps
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
                warn(f"Skipped station '{station_name}' because location information is missing.")
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
        assert isinstance(ts_description, str), f"'ts_description' must be a string, got {type(ts_description)}."
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
                    local_models_subset = {name: model for name, model in station._unused_models[hidden_ts].items() if name in models}
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
                                        desc=f"Processing station timeseries with '{func.__name__}'", total=len(self.stations), ascii=True, unit="station")):
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
        for name, station in tqdm(self.stations.items(), desc=f"Calling function '{func.__name__}' on stations", ascii=True, unit="station"):
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
        assert ndim > 0, f"No components found in '{ts_in}'."
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
            for ts_description, ts in ts_to_plot.items():
                n_components += ts.num_components
                if len(analyze_kw_args) > 0:
                    self.stations[station_name].analyze_residuals(ts_description, **analyze_kw_args)
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
                    ax.set_ylabel(f"{ts_description}\n{data_col} [{ts.data_unit}]")
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
            raise KeyError(f"Station {self.name}: No timeseries '{description}' present.")
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
            warn(f"Station {self.name}: Overwriting time series '{description}'.", category=RuntimeWarning)
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
            warn(f"Station {self.name}: Cannot find time series '{description}', couldn't delete.", category=RuntimeWarning)
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
            f"Station {self.name}: Cannot find timeseries '{ts_description}' to add local model '{model_description}'."
        if model_description in self.models[ts_description]:
            warn(f"Station {self.name}, timeseries {ts_description}: Overwriting local model '{model_description}'.", category=RuntimeWarning)
        self.models[ts_description].update({model_description: model})

    def remove_local_model(self, ts_description, model_description):
        if ts_description not in self.timeseries:
            warn(f"Station {self.name}: Cannot find timeseries '{ts_description}', couldn't delete local model '{model_description}'.", category=RuntimeWarning)
        elif model_description not in self.models[ts_description]:
            warn(f"Station {self.name}, timeseries {ts_description}: Cannot find local model '{model_description}', couldn't delete.", category=RuntimeWarning)
        else:
            del self.models[ts_description][model_description]

    def add_fit(self, ts_description, model_description, fit):
        if not isinstance(ts_description, str):
            raise TypeError("Cannot add new fit: 'ts_description' is not a string.")
        if not isinstance(model_description, str):
            raise TypeError("Cannot add new fit: 'model_description' is not a string.")
        assert ts_description in self.timeseries, \
            f"Station {self.name}: Cannot find timeseries '{ts_description}' to add fit for model '{model_description}'."
        assert model_description in self.models[ts_description], \
            f"Station {self.name}, timeseries {ts_description}: Cannot find local model '{model_description}', couldn't add fit."
        if model_description in self.fits[ts_description]:
            warn(f"Station {self.name}, timeseries {ts_description}: Overwriting fit of local model '{model_description}'.", category=RuntimeWarning)
        data_cols = [ts_description + "_" + model_description + "_" + dcol for dcol in self.timeseries[ts_description].data_cols]
        fit_ts = Timeseries.from_fit(self.timeseries[ts_description].data_unit, data_cols, fit)
        self.fits[ts_description].update({model_description: fit_ts})
        return fit_ts

    def remove_fit(self, ts_description, model_description, fit):
        if ts_description not in self.timeseries:
            warn(f"Station {self.name}: Cannot find timeseries '{ts_description}', couldn't delete fit for model '{model_description}'.", category=RuntimeWarning)
        elif model_description not in self.models[ts_description]:
            warn(f"Station {self.name}, timeseries {ts_description}: Cannot find local model '{model_description}', couldn't delete fit.", category=RuntimeWarning)
        elif model_description not in self.fits[ts_description]:
            warn(f"Station {self.name}, timeseries {ts_description}: Cannot find fit for local model '{model_description}', couldn't delete.", category=RuntimeWarning)
        else:
            del self.fits[ts_description][model_description]

    def analyze_residuals(self, ts_description, mean=False, std=False, n_observations=False, std_outlier=0):
        assert isinstance(ts_description, str), f"Station {self.name}: 'ts_description' needs to be a string, got {type(ts_description)}."
        assert ts_description in self.timeseries, f"Station {self.name}: Can't find '{ts_description}' to analyze."
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
            raise TypeError(f"Unsupported operand type for '{operation}': '{Timeseries}' and '{type(other)}'.")
        # check for same dimensions
        if len(self.data_cols) != len(other.data_cols):
            raise ValueError(f"Timeseries math problem: conflicting number of data columns ({self.data_cols} and {other.data_cols}).")
        # get intersection of time indices
        out_time = self.time.intersection(other.time, sort=None)
        # check compatible units (+-) or define new one (*/)
        # define new src and data_cols
        if operation in ["+", "-"]:
            if self.data_unit != other.data_unit:
                raise ValueError(f"Timeseries math problem: conflicting data units '{self.data_unit}' and '{other.data_unit}'.")
            else:
                out_unit = self.data_unit
                out_src = f"{self.src}{operation}{other.src}"
                out_data_cols = [f"{lcol}{operation}{rcol}" for lcol, rcol in zip(self.data_cols, other.data_cols)]
        elif operation in ["*", "/"]:
            out_unit = f"({self.data_unit}){operation}({other.data_unit})"
            out_src = f"({self.src}){operation}({other.src})"
            out_data_cols = [f"({lcol}){operation}({rcol})" for lcol, rcol in zip(self.data_cols, other.data_cols)]
        else:
            raise NotImplementedError(f"Timeseries math problem: unknown operation '{operation}'.")
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
        df = time.join(data)
        num_duplicates = int(df.duplicated(subset='time').sum())
        if num_duplicates > 0:
            warn(f"Timeseries file {path} contains data for {num_duplicates} duplicate dates. Keeping first occurrences.")
        df = df.drop_duplicates(subset='time').set_index('time')
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
    def __init__(self, num_parameters, regularize=False, time_unit=None, t_start=None, t_end=None, t_reference=None, zero_before=True, zero_after=True):
        self.num_parameters = int(num_parameters)
        assert self.num_parameters > 0, f"'num_parameters' must be an integer greater or equal to one, got {self.num_parameters}."
        self.is_fitted = False
        self.parameters = None
        self.cov = None
        self.regularize = bool(regularize)
        self.time_unit = str(time_unit)
        self._t_start = None if t_start is None else str(t_start)
        self._t_end = None if t_start is None else str(t_end)
        self._t_reference = None if t_start is None else str(t_reference)
        self.t_start = None if t_start is None else pd.Timestamp(t_start)
        self.t_end = None if t_end is None else pd.Timestamp(t_end)
        self.t_reference = None if t_reference is None else pd.Timestamp(t_reference)
        self.zero_before = bool(zero_before)
        self.zero_after = bool(zero_after)

    def get_arch(self):
        # make base architecture
        arch = {"type": "Model",
                "num_parameters": self.num_parameters,
                "kw_args": {"regularize": self.regularize,
                            "time_unit": self.time_unit,
                            "t_start": self._t_start,
                            "t_end": self._t_end,
                            "t_reference": self._t_reference,
                            "zero_before": self.zero_before,
                            "zero_after": self.zero_after}}
        # get subclass-specific architecture
        instance_arch = self._get_arch()
        # update non-dictionary values
        arch.update({arg: value for arg, value in instance_arch.items() if arg != "kw_args"})
        # update keyword dictionary
        arch["kw_args"].update(instance_arch["kw_args"])
        return arch

    def _get_arch(self):
        raise NotImplementedError("Instantiated model was not subclassed or it does not overwrite the '_get_arch' method.")

    def get_mapping(self, timevector):
        # get active period and initialize coefficient matrix
        active, first, last = self._get_active_period(timevector)
        # if there isn't any active period, return csr-sparse matrix
        if (first is None) and (last is None):  # this is equivalent to not active.any()
            mapping = sparse.bsr_matrix((timevector.size, self.num_parameters))
        # otherwise, build coefficient matrix
        else:
            # build dense sub-matrix
            coefs = self._get_mapping(timevector[active])
            assert coefs.shape[1] == self.num_parameters, \
                f"The child function '_get_mapping' of model {type(self).__name__} returned an invalid shape. " \
                f"Expected was ({last-first+1}, {self.num_parameters}), got {coefs.shape}."
            # build before- and after-matrices
            # either use zeros or the values at the active boundaries for padding
            if self.zero_before:
                before = sparse.csr_matrix((first, self.num_parameters))
            else:
                before = sparse.csr_matrix(np.ones((first, self.num_parameters)) * coefs[0, :].reshape(1, -1))
            if self.zero_after:
                after = sparse.csr_matrix((timevector.size - last - 1, self.num_parameters))
            else:
                after = sparse.csr_matrix(np.ones((timevector.size - last - 1, self.num_parameters)) * coefs[-1, :].reshape(1, -1))
            # stack them (they can have 0 in the first dimension, no problem for sparse.vstack)
            # I think it's faster if to stack them if they're all already csr format
            mapping = sparse.vstack((before, sparse.csr_matrix(coefs), after), format='bsr')
        return mapping

    def _get_mapping(self, timevector):
        raise NotImplementedError("'Model' needs to be subclassed and its child needs to implement a '_get_mapping' function for the active period.")

    def _get_active_period(self, timevector):
        if (self.t_start is None) and (self.t_end is None):
            active = np.ones_like(timevector, dtype=bool)
        elif self.t_start is None:
            active = timevector <= self.t_end
        elif self.t_end is None:
            active = timevector >= self.t_start
        else:
            active = np.all((timevector >= self.t_start, timevector <= self.t_end), axis=0)
        if active.any():
            first, last = int(np.argwhere(active)[0]), int(np.argwhere(active)[-1])
        else:
            first, last = None, None
        return active, first, last

    def tvec_to_numpycol(self, timevector):
        """ Convenience wrapper for tvec_to_numpycol for Model objects that have self.time_unit and self.t_reference attributes. """
        if self.t_reference is None:
            raise ValueError("Can't call 'tvec_to_numpycol' because no reference time was specified in the model.")
        if self.time_unit is None:
            raise ValueError("Can't call 'tvec_to_numpycol' because no time unit was specified in the model.")
        return tvec_to_numpycol(timevector, self.t_reference, self.time_unit)

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
    def __init__(self, steptimes, zero_after=False, **model_kw_args):
        super().__init__(num_parameters=len(steptimes), zero_after=zero_after, **model_kw_args)
        self._steptimes = steptimes
        self.timestamps = [pd.Timestamp(step) for step in self._steptimes]
        self.timestamps.sort()

    def _get_arch(self):
        arch = {"type": "Step",
                "kw_args": {"steptimes": self._steptimes}}
        return arch

    def _update_from_steptimes(self):
        self.timestamps = [pd.Timestamp(step) for step in self._steptimes]
        self.timestamps.sort()
        self.num_parameters = len(self.timestamps)
        self.is_fitted = False
        self.parameters = None
        self.cov = None

    def add_step(self, step):
        if step in self._steptimes:
            warn(f"Step '{step}' already present.", category=RuntimeWarning)
        else:
            self._steptimes.append(step)
            self._update_from_steptimes()

    def remove_step(self, step):
        try:
            self._steptimes.remove(step)
            self._update_from_steptimes()
        except ValueError:
            warn(f"Step '{step}' not present.", category=RuntimeWarning)

    def _get_mapping(self, timevector):
        coefs = np.array(timevector.values.reshape(-1, 1) >= pd.DataFrame(data=self.timestamps, columns=["steptime"]).values.reshape(1, -1), dtype=float)
        return coefs


class Polynomial(Model):
    """
    Polynomial of given order.

    `time_unit` can be the following (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html):
        `W`, `D`, `days`, `day`, `hours`, hour`, `hr`, `h`, `m`, `minute`, `min`, `minutes`, `T`,
        `S`, `seconds`, `sec`, `second`, `ms`, `milliseconds`, `millisecond`, `milli`, `millis`, `L`,
        `us`, `microseconds`, `microsecond`, `micro`, `micros`, `U`, `ns`, `nanoseconds`, `nano`, `nanos`, `nanosecond`, `N`
    """
    def __init__(self, order, zero_before=False, zero_after=False, **model_kw_args):
        super().__init__(num_parameters=order + 1, zero_before=zero_before, zero_after=zero_after, **model_kw_args)
        self.order = int(order)

    def _get_arch(self):
        arch = {"type": "Polynomial",
                "kw_args": {"order": self.order}}
        return arch

    def _get_mapping(self, timevector):
        coefs = np.ones((timevector.size, self.num_parameters))
        if self.order >= 1:
            # now we actually need the time
            dt = self.tvec_to_numpycol(timevector)
            # the exponents increase by column
            exponents = np.arange(1, self.order + 1)
            # broadcast to all coefficients
            coefs[:, 1:] = dt.reshape(-1, 1) ** exponents.reshape(1, -1)
        return coefs


class BSpline(Model):
    """
    Cardinal, centralized B-Splines of certain order/degree and time scale.
    Used for transient temporary signals that return to zero after a given time span.

    Compare the analytic representation of the B-Splines:
    Butzer, P., Schmidt, M., & Stark, E. (1988). Observations on the History of Central B-Splines.
    Archive for History of Exact Sciences, 39(2), 137-156. Retrieved May 14, 2020, from https://www.jstor.org/stable/41133848
    or
    Schoenberg, I. J. (1973). Cardinal Spline Interpolation.
    Society for Industrial and Applied Mathematics. https://doi.org/10.1137/1.9781611970555
    and some examples on https://bsplines.org/flavors-and-types-of-b-splines/.

    It is important to note that the function will be non-zero on the interval
    -(p+1)/2 < x < (p+1)/2
    where p is the degree of the cardinal B-spline (and the degree of the resulting polynomial).
    The order n is related to the degree by the relation n = p + 1.
    The scale determines the width of the spline in the time domain, and corresponds to the interval [0, 1] of the B-Spline.
    The full non-zero time span of the spline is therefore scale * (p+1) = scale * n.

    num_splines will increase the number of splines by shifting the reference point (num_splines - 1)
    times by the spacing (which must be given in the same units as the scale).

    If no spacing is given but multiple splines are requested, the scale will be used as the spacing.

    `time_unit` can be the following (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html):
        `W`, `D`, `days`, `day`, `hours`, hour`, `hr`, `h`, `m`, `minute`, `min`, `minutes`, `T`,
        `S`, `seconds`, `sec`, `second`, `ms`, `milliseconds`, `millisecond`, `milli`, `millis`, `L`,
        `us`, `microseconds`, `microsecond`, `micro`, `micros`, `U`, `ns`, `nanoseconds`, `nano`, `nanos`, `nanosecond`, `N`
    """
    def __init__(self, degree, scale, num_splines=1, spacing=None, **model_kw_args):
        super().__init__(num_parameters=num_splines, **model_kw_args)
        self.degree = int(degree)
        self.order = self.degree + 1
        self.scale = float(scale)
        if spacing is not None:
            self.spacing = float(spacing)
            assert abs(self.spacing) > 0, f"'spacing' must be non-zero to avoid singularities, got {self.spacing}."
            if self.num_parameters == 1:
                warn(f"'spacing' ({self.spacing} {self.time_unit}) is given, but 'num_splines' = 1 splines are requested.")
        elif self.num_parameters > 1:
            self.spacing = self.scale
        else:
            self.spacing = None

    def _get_arch(self):
        arch = {"type": "BSpline",
                "kw_args": {"degree": self.degree,
                            "scale": self.scale,
                            "num_splines": self.num_parameters,
                            "spacing": self.spacing}}
        return arch

    def _get_mapping(self, timevector):
        trel = self.tvec_to_numpycol(timevector).reshape(-1, 1, 1) - self.scale * np.arange(self.num_parameters).reshape(1, -1, 1)
        tnorm = trel / self.scale
        krange = np.arange(self.order + 1).reshape(1, 1, -1)
        in_power = tnorm + self.order/2 - krange
        in_sum = (-1)**krange * comb(self.order, krange) * (in_power)**(self.degree) * (in_power >= 0)
        coefs = np.sum(in_sum, axis=2) / factorial(self.degree)
        return coefs


class ISpline(Model):
    """
    Integral of cardinal, centralized B-Splines of certain order/degree and time scale.
    The degree p given in the initialization is the degree of the spline *before* the integration, i.e.
    the resulting IBSpline is a piecewise polynomial of degree p + 1.
    Used for transient permanent signals that stay at their maximum value after a given time span.

    See the full documentation in the BSpline class.
    """
    def __init__(self, degree, scale, num_splines=1, spacing=None, zero_after=False, **model_kw_args):
        super().__init__(num_parameters=num_splines, zero_after=zero_after, **model_kw_args)
        self.degree = int(degree)
        self.order = self.degree + 1
        self.scale = float(scale)
        if spacing is not None:
            self.spacing = float(spacing)
            assert abs(self.spacing) > 0, f"'spacing' must be non-zero to avoid singularities, got {self.spacing}."
            if self.num_parameters == 1:
                warn(f"'spacing' ({self.spacing} {self.time_unit}) is given, but 'num_splines' = 1 splines are requested.")
        elif self.num_parameters > 1:
            self.spacing = self.scale
        else:
            self.spacing = None

    def _get_arch(self):
        arch = {"type": "ISpline",
                "kw_args": {"degree": self.degree,
                            "scale": self.scale,
                            "num_splines": self.num_parameters,
                            "spacing": self.spacing}}
        return arch

    def _get_mapping(self, timevector):
        trel = self.tvec_to_numpycol(timevector).reshape(-1, 1, 1) - self.scale * np.arange(self.num_parameters).reshape(1, -1, 1)
        tnorm = trel / self.scale
        krange = np.arange(self.order + 1).reshape(1, 1, -1)
        in_power = tnorm + self.order/2 - krange
        in_sum = (-1)**krange * comb(self.order, krange) * (in_power)**(self.degree + 1) * (in_power >= 0)
        coefs = np.sum(in_sum, axis=2) / factorial(self.degree + 1)
        return coefs


def build_splineset(degree, t_start, t_end, time_unit, splineclass=ISpline, list_scales=None, list_num_knots=None, complete=True):
    """
    Return a list of splines that share a common degree, but different center times and scales.

    The set is constructed from a time span (t_start and t_end) and numbers of centerpoints or length scales.
    The number of splines for each scale will then be chosen such that the resulting set of splines will be complete.
    This means it will contain all splines that are non-zero at least somewhere in the time span.

    This function also sets the spacing equal to the scale.
    """
    assert np.logical_xor(list_scales is None, list_num_knots is None), \
        f"To construct a set of BSplines, only pass one of 'list_scales' and 'list_num_knots' " \
        f"(got {list_scales} and {list_num_knots})."
    relevant_list = list_scales if list_num_knots is None else list_num_knots
    # get time range
    t_start_tstamp, t_end_tstamp = pd.Timestamp(t_start), pd.Timestamp(t_end)
    t_range_tdelta = t_end_tstamp - t_start_tstamp
    # if a complete set is requested, we need to find the number of overlaps given the degree on a single side
    num_overlaps = degree if complete else 0
    # for each scale, make a BSplines object
    splset = []
    for elem in relevant_list:
        # Calculate the scale as float and Timedelta depending on the function call
        if list_scales is not None:
            scale_float = elem
            scale_tdelta = pd.Timedelta(scale_float, time_unit)
        else:
            scale_tdelta = t_range_tdelta / elem
            scale_float = scale_tdelta / np.timedelta64(1, time_unit)
        # find the number of center points between t_start and t_end, plus the overlapping ones
        num_centerpoints = int(t_range_tdelta / scale_tdelta) + 1 + 2*num_overlaps
        # shift the reference to be the first spline
        t_ref = t_start_tstamp - num_overlaps*scale_tdelta
        # create model and append
        splset.append(splineclass(degree, scale_float, num_splines=num_centerpoints, t_reference=t_ref, time_unit=time_unit))
    return splset


def make_scalogram(list_of_splines, t_left, t_right, cmaprange=None, resolution=1000):
    # check input
    assert isinstance(list_of_splines, list) and all([isinstance(model, Model) for model in list_of_splines]), \
        f"'list_of_splines' needs to be a list of spline models, got {list_of_splines}."
    assert all([model.is_fitted for model in list_of_splines]), \
        f"All models in 'list_of_splines' need to have already been fitted."
    # determine dimensions
    num_scales = len(list_of_splines)
    dy_scale = 1/num_scales
    t_plot = np.array([tstamp for tstamp in pd.date_range(start=t_left, end=t_right, periods=resolution)])
    # get range of values (if not provided)
    if cmaprange is not None:
        assert isinstance(cmaprange, int) or isinstance(cmaprange, float), \
            f"'cmaprange' must be a single float or integer of the one-sided color range of the scalogram, got {cmaprange}."
    else:
        cmaprange = np.ceil(np.max(np.abs(np.array([model.parameters for model in list_of_splines]).ravel())))
    cmap = mpl.cm.ScalarMappable(cmap=CMAPS["seismic"], norm=mpl.colors.Normalize(vmin=-cmaprange, vmax=cmaprange))
    # start plotting
    fig, ax = plt.subplots()
    for i, model in enumerate(list_of_splines):
        # where to put this scale
        y_off = 1 - (i + 1)*dy_scale
        # get normalized values
        mdl_mapping = model.get_mapping(t_plot)
        mdl_sum = np.sum(mdl_mapping, axis=1, keepdims=True)
        mdl_sum[mdl_sum == 0] = 1
        y_norm = np.hstack([np.zeros((t_plot.size, 1)), np.cumsum(mdl_mapping / mdl_sum, axis=1)])
        # plot cell
        for j in range(model.num_parameters):
            ax.fill_between(t_plot, y_off + y_norm[:, j]*dy_scale, y_off + y_norm[:, j+1]*dy_scale, facecolor=cmap.to_rgba(model.parameters[j]))
        # plot vertical lines at centerpoints
        for j in range(model.num_parameters):
            ax.axvline(model.t_reference + pd.Timedelta(j*model.spacing, model.time_unit), y_off, y_off + dy_scale, c='0.7')
    # finish plot by adding relevant gridlines and labels
    for i in range(1, num_scales):
        ax.axhline(i*dy_scale, c='0.7')
    ax.set_xlim(t_left, t_right)
    ax.set_ylim(0, 1)
    ax.set_yticks([i*dy_scale for i in range(num_scales + 1)])
    ax.set_yticks([(i + 0.5)*dy_scale for i in range(num_scales)], minor=True)
    ax.set_yticklabels([f"{model.scale:.3g} {model.time_unit}" for model in list_of_splines], minor=True)
    ax.tick_params(axis='both', labelleft=False, direction='out')
    ax.tick_params(axis='y', left=False, which='minor')
    fig.colorbar(cmap, orientation='horizontal', fraction=0.1, pad=0.1, label='Coefficient Value')
    return fig, ax


class Sinusoidal(Model):
    """
    Sinusoidal of given frequency. Estimates amplitude and phase.

    `time_unit` can be the following (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_timedelta.html):
        `W`, `D`, `days`, `day`, `hours`, hour`, `hr`, `h`, `m`, `minute`, `min`, `minutes`, `T`,
        `S`, `seconds`, `sec`, `second`, `ms`, `milliseconds`, `millisecond`, `milli`, `millis`, `L`,
        `us`, `microseconds`, `microsecond`, `micro`, `micros`, `U`, `ns`, `nanoseconds`, `nano`, `nanos`, `nanosecond`, `N`
    """
    def __init__(self, period, **model_kw_args):
        super().__init__(num_parameters=2, **model_kw_args)
        self.period = float(period)

    def _get_arch(self):
        arch = {"type": "Sinusoidal",
                "kw_args": {"period": self.period}}
        return arch

    def _get_mapping(self, timevector):
        dt = self.tvec_to_numpycol(timevector)
        phase = 2*np.pi * dt / self.period
        coefs = np.stack([np.cos(phase), np.sin(phase)], axis=1)
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
    Geophysical logarithmic `ln(1 + dt/tau)` with a given time constant and time window.
    """
    def __init__(self, tau, **model_kw_args):
        super().__init__(num_parameters=1, **model_kw_args)
        if self.t_reference is None:
            warn("No 't_reference' set for Logarithmic model, using 't_start' for it.")
            self._t_reference = self._t_start
            self.t_reference = self.t_start
        elif self.t_start is None:
            warn("No 't_start' set for Logarithmic model, using 't_reference' for it.")
            self._t_start = self._t_reference
            self.t_start = self.t_reference
        else:
            assert self.t_reference <= self.t_start, \
                f"Logarithmic model has to have valid bounds, but the reference time {self._t_reference} is after the start time {self._t_start}."
        self.tau = float(tau)

    def _get_arch(self):
        arch = {"type": "Logarithmic",
                "kw_args": {"tau": self.tau}}
        return arch

    def _get_mapping(self, timevector):
        dt = self.tvec_to_numpycol(timevector)
        coefs = np.log1p(dt / self.tau).reshape(-1, 1)
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
                raise TypeError(f"Cannot unwrap object of type {type(ts)}.")
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
    if COMPILED_UTILS:
        filtered = maskedmedfilt2d(array, ~np.isnan(array), kernel_size)
        filtered[np.isnan(array)] = np.NaN
    else:
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
        decomposer = PCA(n_components=n_components, whiten=True)
    elif method == 'ica':
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
    G = sparse.hstack(mapping_matrices)
    num_time, num_params = G.shape
    num_components = len(ts.data_cols)
    # perform fit and estimate formal covariance (uncertainty) of parameters
    params = np.zeros((num_params, num_components))
    if formal_covariance:
        cov = np.zeros((num_params, num_params, num_components))
    for i in range(num_components):
        d = sparse.csc_matrix(ts.df[ts.data_cols[i]].values.reshape(-1, 1))
        if ts.sigma_cols[i] is None:
            GtWG = G.T @ G
            GtWd = G.T @ d
        else:
            GtW = G.T @ sparse.diags(1/ts.df[ts.sigma_cols[i]].values**2)
            GtWG = GtW @ G
            GtWd = GtW @ d
        params[:, i] = sparse.linalg.lsqr(GtWG, GtWd.toarray().squeeze())[0].squeeze()
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
    mapping_matrices = []
    reg_diag = []
    # get mapping and regularization matrices
    for (mdl_description, model) in models.items():
        mapping_matrices.append(model.get_mapping(ts.time))
        reg_diag.extend([model.regularize for _ in range(model.num_parameters)])
    G = sparse.hstack(mapping_matrices, format='bsr')
    reg = sparse.diags(reg_diag, dtype=float) * penalty
    num_time, num_params = G.shape
    num_components = len(ts.data_cols)
    # perform fit and estimate formal covariance (uncertainty) of parameters
    params = np.zeros((num_params, num_components))
    if formal_covariance:
        cov = np.zeros((num_params, num_params, num_components))
    for i in range(num_components):
        d = sparse.csc_matrix(ts.df[ts.data_cols[i]].values.reshape(-1, 1))
        if ts.sigma_cols[i] is None:
            GtWG = G.T @ G
            GtWd = G.T @ d
        else:
            GtW = G.T @ sparse.diags(1/ts.df[ts.sigma_cols[i]].values**2)
            GtWG = GtW @ G
            GtWd = GtW @ d
        GtWGreg = GtWG + reg
        params[:, i] = sparse.linalg.lsqr(GtWGreg, GtWd.toarray().squeeze())[0].squeeze()
        if formal_covariance:
            cov[:, :, i] = np.linalg.pinv(GtWGreg.toarray())
    # separate parameters back to models
    i = 0
    fitted_params = {}
    for (mdl_description, model) in models.items():
        fitted_params[mdl_description] = (params[i:i+model.num_parameters, :], cov[i:i+model.num_parameters, i:i+model.num_parameters, :] if formal_covariance else None)
        i += model.num_parameters
    return fitted_params


def tvec_to_numpycol(timevector, t_reference=None, time_unit='D'):
    # get reference time
    if t_reference is None:
        t_reference = timevector[0]
    else:
        t_reference = pd.Timestamp(t_reference)
    assert isinstance(t_reference, pd.Timestamp), f"'t_reference' must be a pandas.Timestamp object, got {type(t_reference)}."
    # return Numpy array
    return ((timevector - t_reference) / np.timedelta64(1, time_unit)).values


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
            warn(f"Success = {success} for station {i}!", category=RuntimeWarning)
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
    # initialize network from an architecture .json file
    # net = Network.from_json(path="net_arch_boso.json", add_default_local_models=False)
    net = Network.from_json(path="net_arch_boso_okada3mm.json", add_default_local_models=False)
    # low-pass filter using a median function, then clean the timeseries (using the settings in the config file)
    net.call_func_ts_return(median, ts_in='GNSS', ts_out='filtered', kernel_size=7)
    net.call_func_no_return(clean, ts_in='GNSS', reference='filtered', ts_out='clean')
    # get the residual for each station, delete the intermediate product
    for station in net:
        station['residual'] = station['clean'] - station['filtered']
        del station['filtered']
    # estimate the common mode, either with a visualization of the result or not (same underlying function)
    # net.graphical_cme(ts_in='residual', ts_out='common', method='pca')
    net.call_netwide_func(common_mode, ts_in='residual', ts_out='common', method='pca')
    # now remove the common mode, call it the 'final' timeseries, and delete intermediate products
    for station in net:
        del station['residual']
        station.add_timeseries('final', station['clean'] - station['common'],
                               override_data_cols=station['GNSS'].data_cols)
        del station['common']
    # add the default models from the architecture .json file that weren't needed before
    net.add_default_local_models('final')
    # either use okada_prior to get time for steps at stations
    # okada_prior(net, "data/nied_fnet_catalog.txt", target_timeseries='final')
    # or after that, load them into the cleaned timeseries
    net.add_unused_local_models('final', models="Catalog")
    # fit the models and evaluate
    net.fit("final", solver="ridge_regression", penalty=1e3, formal_covariance=False)
    net.evaluate("final", output_description="model")
    # calculate the residual between the best-fit model and the clean ('final') timeseries
    for station in net:
        station.add_timeseries('residual', station['final'] - station['model'],
                               override_data_cols=station['GNSS'].data_cols)
    # save the network architecture as a .json file and show the results in a GUI
    net.to_json(path="arch_out.json")
    net.gui(timeseries=['final', 'residual'], mean=True, std=True, n_observations=True, std_outlier=5)
