import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings
from copy import deepcopy
from configparser import ConfigParser
from tqdm import tqdm
from multiprocessing import Pool
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# import defaults
config = ConfigParser()
config.read("config.ini")


class Network():
    """
    A container object for multiple stations.
    """
    def __init__(self, name, default_location_path=None):
        self.name = name
        self._default_location_path = default_location_path
        self.stations = {}
        self.network_locations = {}
        self._default_local_models = {}
        self.global_models = {}
        self.global_priors = {}

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
        self.network_locations[name] = station.location

    def remove_station(self, name):
        if name not in self.stations:
            Warning("Cannot find station {}, couldn't delete".format(name))
        else:
            del self.stations[name]
            del self.network_locations[name]

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
    def from_json(cls, path):
        # load configuration
        net_arch = json.load(open(path, mode='r'))
        network_name = net_arch.get("name", "network_from_json")
        network_locations_path = net_arch.get("locations")
        # create Network instance
        net = cls(name=network_name, default_location_path=network_locations_path)
        # load location information, if present
        if network_locations_path is not None:
            with open(network_locations_path, mode='r') as locfile:
                loclines = [line.strip() for line in locfile.readlines()]
            network_locations = {line.split()[0]: [float(lla) for lla in line.split()[1:]] for line in loclines}
        # load default local models
        net._default_local_models = net_arch["default_local_models"]
        # create stations
        for station_name, station_cfg in tqdm(net_arch["stations"].items(), ascii=True, desc="Building Network", unit="station"):
            if "location" in station_cfg:
                station_loc = station_cfg["location"]
            elif station_name in network_locations:
                station_loc = network_locations[station_name]
            else:
                Warning("Skipped station {:s} because location information is missing.".format(station_name))
                continue
            stat = Station(name=station_name, location=station_loc)
            # add timeseries to station
            for ts_description, ts_cfg in station_cfg["timeseries"].items():
                ts = globals()[ts_cfg["type"]](**ts_cfg["kw_args"])
                stat.add_timeseries(description=ts_description, timeseries=ts)
                # add default local models to station
                for model_description, model_cfg in net._default_local_models.items():
                    local_copy = deepcopy(model_cfg)
                    mdl = globals()[local_copy["type"]](**local_copy["kw_args"])
                    stat.add_local_model(ts_description=ts_description, model_description=model_description, model=mdl)
                # add specific local models to station
                for model_description, model_cfg in station_cfg["models"].items():
                    mdl = globals()[model_cfg["type"]](**model_cfg["kw_args"])
                    stat.add_local_model(ts_description=ts_description, model_description=model_description, model=mdl)
            # add to network
            net.add_station(name=station_name, station=stat)
        # add global models
        for model_description, model_cfg in net_arch["global_models"].items():
            mdl = globals()[model_cfg["type"]](**model_cfg["kw_args"])
            net.add_global_model(description=model_description, model=mdl)
        return net

    def to_json(self, path):
        # create new dictionary
        net_arch = {"name": self.name,
                    "locations": self._default_location_path,
                    "stations": {},
                    "default_local_models": self._default_local_models,
                    "global_models": {}}
        # add station representations
        for stat_name, stat in self.stations.items():
            stat_arch = stat.get_arch()
            if stat_arch == {}:
                continue
            # need to remove all models that are actually default models
            for mdl_description, mdl in self._default_local_models.items():
                if (mdl_description in stat_arch["models"].keys()) and (mdl == stat_arch["models"][mdl_description]):
                    del stat_arch["models"][mdl_description]
            # now we can append it to the main json
            net_arch["stations"].update({stat_name: stat_arch})
        # add global model representations
        for mdl_description, mdl in self.global_models.items():
            net_arch["global_models"].update({mdl_description: mdl.get_arch()})
        # write file
        json.dump(net_arch, open(path, mode='w'), indent=2)

    def fit(self, ts_description, model_list=None, solver=config.get('fit', 'solver'), **kw_args):
        if "num_threads" in config.options("general"):
            # collect station calls
            iterable_inputs = [(stat, ts_description, model_list, solver, kw_args) for stat in self.stations.values()]
            # start multiprocessing pool
            with Pool(config.getint("general", "num_threads")) as p:
                fit_output = list(tqdm(p.imap(self._fit_single_station, iterable_inputs), ascii=True, desc="Fitting station models", total=len(iterable_inputs), unit="station"))
            # redistribute results
            for i, stat in enumerate(self.stations.values()):
                for model_description, (params, covs) in fit_output[i].items():
                    stat.models[ts_description][model_description].read_parameters(params, covs)
        else:
            # run in serial
            for name, stat in tqdm(self.stations.items(), desc="Fitting station models", ascii=True, unit="station"):
                fit_output = self._fit_single_station((stat, ts_description, model_list, solver, kw_args))
                for model_description, (params, covs) in fit_output.items():
                    stat.models[ts_description][model_description].read_parameters(params, covs)

    @staticmethod
    def _fit_single_station(parameter_tuple):
        station, ts_description, model_list, solver, kw_args = parameter_tuple
        fitted_params = globals()[solver](station.timeseries[ts_description], station.models[ts_description] if model_list is None else {m: station.models[ts_description][m] for m in model_list}, **kw_args)
        return fitted_params

    def evaluate(self, ts_description, model_list=None, timevector=None):
        if "num_threads" in config.options("general"):
            # collect station calls
            iterable_inputs = [(stat, ts_description, model_list, timevector) for stat in self.stations.values()]
            # start multiprocessing pool
            with Pool(config.getint("general", "num_threads")) as p:
                eval_output = list(tqdm(p.imap(self._evaluate_single_station, iterable_inputs), ascii=True, desc="Evaluating station models", total=len(iterable_inputs), unit="station"))
            # redistribute results
            for i, stat in enumerate(self.stations.values()):
                stat_fits = eval_output[i]
                for model_description, fit in stat_fits.items():
                    stat.add_fit(ts_description, model_description, fit)
        else:
            # run in serial
            for name, stat in tqdm(self.stations.items(), desc="Evaluating station models", ascii=True, unit="station"):
                stat_fits = self._evaluate_single_station((stat, ts_description, model_list, timevector))
                for model_description, fit in stat_fits.items():
                    stat.add_fit(ts_description, model_description, fit)

    @staticmethod
    def _evaluate_single_station(parameter_tuple):
        station, ts_description, model_list, timevector = parameter_tuple
        fit = {}
        for model_description, model in station.models[ts_description].items() if model_list is None else {m: station.models[ts_description][m] for m in model_list}.items():
            fit[model_description] = model.evaluate(station.timeseries[ts_description].time) if timevector is None else model.evaluate(timevector)
        return fit

    def call_func_ts_return(self, ts_in, func, ts_out=None, **kw_args):
        if "num_threads" in config.options("general"):
            # collect station calls
            iterable_inputs = [(stat, ts_in, func, kw_args) for stat in self.stations.values()]
            # start multiprocessing pool
            with Pool(config.getint("general", "num_threads")) as p:
                ts_return = list(tqdm(p.imap(self._single_call_func_ts_return, iterable_inputs), ascii=True, desc="Processing station timeseries with {:s}".format(func), total=len(iterable_inputs), unit="station"))
            # redistribute results
            for i, stat in enumerate(self.stations.values()):
                stat.add_timeseries(ts_in if ts_out is None else ts_out, ts_return[i])
        else:
            # run in serial
            for name, stat in tqdm(self.stations.items(), desc="Processing station timeseries with {:s}".format(func), ascii=True, unit="station"):
                ts_return = self._single_call_func_ts_return((stat, ts_in, func, kw_args))
                stat.add_timeseries(ts_in if ts_out is None else ts_out, ts_return)

    @staticmethod
    def _single_call_func_ts_return(parameter_tuple):
        station, ts_description, func, kw_args = parameter_tuple
        ts_return = globals()[func](station.timeseries[ts_description], **kw_args)
        return ts_return

    def call_netwide_func(self, ts_in, func, ts_out=None, **kw_args):
        net_in = self.export_network_ts(ts_in)
        net_out = globals()[func](net_in, **kw_args)
        self.import_network_ts(ts_in if ts_out is None else ts_out, net_out)

    def call_func_no_return(self, func, **kw_args):
        for name, stat in tqdm(self.stations.items(), desc="Calling function {:s} on stations".format(func), ascii=True, unit="station"):
            globals()[func](stat, **kw_args)

    def gui(self):
        # get location data and projections
        stat_lats = [lla[0] for lla in self.network_locations.values()]
        stat_lons = [lla[1] for lla in self.network_locations.values()]
        proj_gui = getattr(ccrs, config.get("gui", "projection", fallback="Mercator"))()
        proj_lla = ccrs.Geodetic()

        # create map figure
        fig_map = plt.figure()
        ax_map = fig_map.add_subplot(projection=proj_gui)
        default_station_colors = ['b'] * len(self.network_locations.values())
        stat_points = ax_map.scatter(stat_lons, stat_lats, linestyle='None', marker='.', transform=proj_lla, facecolor=default_station_colors)
        map_underlay = False
        if "wmts_server" in config.options("gui"):
            try:
                ax_map.add_wmts(config.get("gui", "wmts_server"), layer_name=config.get("gui", "wmts_layer"), alpha=config.getfloat("gui", "wmts_alpha"))
                map_underlay = True
            except Exception as exc:
                print(exc)
        if "coastlines_resolution" in config.options("gui"):
            ax_map.coastlines(color="white" if map_underlay else "black", resolution=config.get("gui", "coastlines_resolution"))

        # create empty timeseries figure
        fig_ts = plt.figure()

        # define clicking function
        def update_timeseries(event):
            if (event.xdata is None) or (event.ydata is None) or (event.inaxes is not ax_map): return
            click_lon, click_lat = proj_lla.transform_point(event.xdata, event.ydata, src_crs=proj_gui)
            station_index = np.argmin(np.sqrt((np.array(stat_lats) - click_lat)**2 + (np.array(stat_lons) - click_lon)**2))
            station_name = list(self.network_locations.keys())[station_index]
            highlight_station_colors = default_station_colors.copy()
            highlight_station_colors[station_index] = 'r'
            stat_points.set_facecolor(highlight_station_colors)
            fig_map.canvas.draw_idle()
            # get components and check if they have uncertainties
            n_components = 0
            for ts in self.stations[station_name].timeseries.values():
                n_components += len(ts.data_cols)
            # clear figure and add data
            fig_ts.clear()
            icomp = 0
            ax_ts = []
            for its, (ts_description, ts) in enumerate(self.stations[station_name].timeseries.items()):
                for icol, (data_col, sigma_col) in enumerate(zip(ts.data_cols, ts.sigma_cols)):
                    # add axis
                    ax = fig_ts.add_subplot(n_components, 1, icomp + 1, sharex=None if icomp == 0 else ax_ts[0])
                    # plot uncertainty
                    if sigma_col is not None and "plot_sigmas" in config.options("gui"):
                        ax.fill_between(ts.time, ts.df[data_col] + config.getfloat("gui", "plot_sigmas") * ts.df[sigma_col],
                                        ts.df[data_col] - config.getfloat("gui", "plot_sigmas") * ts.df[sigma_col], facecolor='gray',
                                        alpha=config.getfloat("gui", "plot_sigmas_alpha"), linewidth=0)
                    # plot data
                    ax.plot(ts.time, ts.df[data_col], marker='.', color='k', label="Data")
                    # overlay models
                    for (mdl_description, fit) in self.stations[station_name].fits[ts_description].items():
                        if fit.sigma_cols[icol] is not None and "plot_sigmas" in config.options("gui"):
                            ax.fill_between(fit.time, fit.df[fit.data_cols[icol]] + config.getfloat("gui", "plot_sigmas") * fit.df[fit.sigma_cols[icol]],
                                            fit.df[fit.data_cols[icol]] - config.getfloat("gui", "plot_sigmas") * fit.df[fit.sigma_cols[icol]],
                                            alpha=config.getfloat("gui", "plot_sigmas_alpha"), linewidth=0)
                        ax.plot(fit.time, fit.df[fit.data_cols[icol]], label=mdl_description)
                    ax.set_ylabel("{:s}\n{:s} [{:s}]".format(ts_description, data_col, ts.data_unit))
                    ax.grid()
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
        self.fits = {}

    def __repr__(self):
        info = f"Station {self.name} at {self.location} with timeseries"
        for ts_description, ts in self.timeseries.items():
            info += f"\n[{ts_description}]\n - Source: {ts.src}\n - Units: {ts.data_unit}\n" + \
                    f" - Data: {[key for key in ts.data_cols]}\n" + \
                    f" - Uncertainties: {[key for key in ts.sigma_cols]}\n" + \
                    f" - Models: {[key for key in self.models[ts_description]]}\n" + \
                    f" - Fits: {[key for key in self.fits[ts_description]]}"
        return info

    def __getitem__(self, description):
        if description not in self.timeseries:
            raise KeyError("Station {:s}: No timeseries '{}' present.".format(self.name, description))
        return self.timeseries[description]

    def __setitem__(self, description, timeseries):
        self.add_timeseries(description, timeseries)

    def __delitem__(self, description):
        self.remove_timeseries(description)

    @property
    def ts(self):
        return self.timeseries

    def get_arch(self):
        # create empty dictionary
        stat_arch = {"timeseries": {},
                     "models": {}}
        # add each timeseries and model
        for ts_description, ts in self.timeseries.items():
            stat_arch["timeseries"].update({ts_description: ts.get_arch()})
        # currently, all models are applied to all timeseries, therefore we can just export the last one
        for mdl_description, mdl in self.models[ts_description].items():
            stat_arch["models"].update({mdl_description: mdl.get_arch()})
        return stat_arch

    def add_timeseries(self, description, timeseries):
        if not isinstance(description, str):
            raise TypeError("Cannot add new timeseries: 'description' is not a string.")
        if not isinstance(timeseries, Timeseries):
            raise TypeError("Cannot add new timeseries: 'timeseries' is not a Timeseries object.")
        if description in self.timeseries:
            Warning("Station {:s}: Overwriting time series {:s}".format(self.name, description))
        self.timeseries[description] = timeseries
        self.fits[description] = {}
        self.models[description] = {}

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

    def remove_fit(self, ts_description, model_description, fit):
        if ts_description not in self.timeseries:
            Warning("Station {:s}: Cannot find timeseries {:s}, couldn't delete fit for model {:s}".format(self.name, ts_description, model_description))
        elif model_description not in self.models[ts_description]:
            Warning("Station {:s}, timeseries {:s}: Cannot find local model {:s}, couldn't delete fit".format(self.name, ts_description, model_description))
        elif model_description not in self.fits[ts_description]:
            Warning("Station {:s}, timeseries {:s}: Cannot find fit for local model {:s}, couldn't delete".format(self.name, ts_description, model_description))
        else:
            del self.fits[ts_description][model_description]


class Timeseries():
    """
    Container object that for a given time vector contains
    data points.
    """
    def __init__(self, dataframe, source, data_unit, data_cols, sigma_cols=None):
        self.df = dataframe
        self.src = source
        self.data_unit = data_unit
        self.data_cols = data_cols
        if sigma_cols is None:
            self.sigma_cols = [None] * len(self.data_cols)
        else:
            assert len(self.data_cols) == len(sigma_cols), \
                "If passing uncertainty columns, the list needs to have the same length as the data columns one. " + \
                "If only certain components have associated uncertainties, leave those list entries as None."
            self.sigma_cols = sigma_cols

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
    def num_components(self):
        return len(self.data_cols)

    @property
    def num_observations(self):
        return self.df.shape[0]

    @property
    def data(self):
        return self.df.loc[:, self.data_cols]

    @data.setter
    def data(self, newdata):
        self.df.loc[:, self.data_cols] = newdata

    @property
    def sigmas(self):
        if not any(self.sigma_cols):
            raise ValueError("No uncertainty columns present to return.")
        return self.df.loc[:, self.sigma_cols]

    @sigmas.setter
    def sigmas(self, newsigmas):
        if not any(self.sigma_cols):
            raise ValueError("No uncertainty columns present to set.")
        self.df.loc[:, self.sigma_cols] = newsigmas

    @property
    def time(self):
        return self.df.index

    def copy(self, only_data=False):
        if not only_data:
            return Timeseries(self.df.copy(), deepcopy(self.src), deepcopy(self.data_unit), deepcopy(self.data_cols), deepcopy(self.sigma_cols))
        else:
            return Timeseries(self.df[self.data_cols].copy(), deepcopy(self.src), deepcopy(self.data_unit), deepcopy(self.data_cols), None)

    def mask_out(self, dcol):
        icol = self.data_cols.index(dcol)
        scol = self.sigma_cols[icol]
        self.df[dcol] = np.NaN
        self.df[dcol] = self.df[dcol].astype(pd.SparseDtype(dtype=float))
        if scol is not None:
            self.df[scol] = np.NaN
            self.df[scol] = self.df[scol].astype(pd.SparseDtype(dtype=float))

    def _prepare_math(self, other, operation):
        # check for same type
        if not isinstance(other, Timeseries):
            raise TypeError("Unsupported operand type for {}: {} and {}".format(operation, Timeseries, type(other)))
        # check for same dimensions
        if len(self.data_cols) != len(other.data_cols):
            raise ValueError(("Timeseries math problem: conflicting number of data columns (" +
                              "[ " + "'{}' "*len(self.data_cols) + "] and " +
                              "[ " + "'{}' "*len(other.data_cols) + "])").format(*self.data_cols, *other.data_cols))
        # check for same timestamps
        if not (self.time == other.time).all():
            raise ValueError("Timeseries math problem: different time columns (sizes {:d} and {:d})".format(self.time.size, other.time.size))
        # check compatible units (+-) or define new one (*/)
        # define names of new src and data_cols
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
        return out_src, out_unit, out_data_cols

    def __add__(self, other):
        out_src, out_unit, out_data_cols = self._prepare_math(other, '+')
        out_dict = {newcol: self.df[lcol].values + other.df[rcol].values for newcol, lcol, rcol in zip(out_data_cols, self.data_cols, other.data_cols)}
        return Timeseries(pd.DataFrame(data=out_dict, index=self.time), out_src, out_unit, out_data_cols)

    def __sub__(self, other):
        out_src, out_unit, out_data_cols = self._prepare_math(other, '-')
        out_dict = {newcol: self.df[lcol].values - other.df[rcol].values for newcol, lcol, rcol in zip(out_data_cols, self.data_cols, other.data_cols)}
        return Timeseries(pd.DataFrame(data=out_dict, index=self.time), out_src, out_unit, out_data_cols)

    def __mul__(self, other):
        out_src, out_unit, out_data_cols = self._prepare_math(other, '*')
        out_dict = {newcol: self.df[lcol].values * other.df[rcol].values for newcol, lcol, rcol in zip(out_data_cols, self.data_cols, other.data_cols)}
        return Timeseries(pd.DataFrame(data=out_dict, index=self.time), out_src, out_unit, out_data_cols)

    def __truediv__(self, other):
        out_src, out_unit, out_data_cols = self._prepare_math(other, '/')
        out_dict = {newcol: self.df[lcol].values / other.df[rcol].values for newcol, lcol, rcol in zip(out_data_cols, self.data_cols, other.data_cols)}
        return Timeseries(pd.DataFrame(data=out_dict, index=self.time), out_src, out_unit, out_data_cols)

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
        super().__init__(dataframe=df, source='tseries', data_unit='m',
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
    def __init__(self, order=1, starttime=None, timeunit='Y', regularize=False):
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
    def __init__(self, period=1, starttime=None, timeunit='Y', regularize=False):
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


def nanmedfilt2d(array, kernel_size):
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


def median(data, kernel_size):
    """
    Perform a median filter with a sliding window. For edges, we shrink window.
    Can operate on single timeseries as well as a network-wide dataframe.
    """
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    if not isinstance(data, dict):
        data = {'ts': data}
        was_dict = False
    else:
        was_dict = True
    filt = {}
    # loop over components
    for comp, ts in data.items():
        if isinstance(ts, Timeseries):
            array = ts.data.values
        else:
            array = ts.values
        filtered = nanmedfilt2d(array, kernel_size)
        # save results
        if isinstance(ts, Timeseries):
            filt[comp] = ts.copy(only_data=True)
            filt[comp].data = filtered
        else:
            filt[comp] = ts.copy()
            filt[comp] = filtered
    if not was_dict:
        filt = filt['ts']
    return filt


def clean(station, ts_in, reference, ts_out=None,
          std_thresh=config.getfloat('clean', 'std_thresh', fallback=None),
          std_outlier=config.getfloat('clean', 'std_outlier', fallback=None),
          min_obs=config.getint('clean', 'min_obs', fallback=0),
          min_clean_obs=config.getint('clean', 'min_obs', fallback=0), **kw_args):
    # check if we're modifying in-place or copying
    if ts_out is None:
        ts = station[ts_in]
    else:
        ts = station[ts_in].copy(only_data=True)
    # check if we have a reference time series or need to calculate one
    # in the latter case, the input is name of function to call
    if not isinstance(reference, Timeseries):
        if not isinstance(reference, str):
            raise TypeError("'reference' has to either be a Timeseries object or the name of a function.")
        # get reference time series
        ts_ref = globals()[reference](ts, **kw_args)
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
            ts[dcol][~np.isnan(residual)][np.abs(residual[~np.isnan(residual)]) > std_outlier*sd] = np.NaN
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


def okada_prior(net, catalog_path):
    # import okada
    from okada_wrapper import dc3d0wrapper as dc3d0
    # network locations are in net.station_locations
    stations_lla = np.array([loc for loc in net.network_locations.values()])
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
    # define inputs for the different earthquakes
    eqs = [{'alpha': config.getfloat('catalog_prior', 'alpha'), 'lat': eq_lla[i, 0], 'lon': eq_lla[i, 1],
            'depth': -eq_lla[i, 2], 'strike': float(catalog['Strike'][i].split(';')[0]), 'dip': float(catalog['Dip'][i].split(';')[0]), 'potency': [catalog['Mo(Nm)'][i]/config.getfloat('catalog_prior', 'mu'), 0, 0, 0]} for i in range(n_eq)]
    parameters = [(stations_rel[i], eqs[i]) for i in range(n_eq)]

    # define function that calculates displacement for all stations
    def get_displacements(parameters):
        # unpack inputs
        stations, eq = parameters
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

    # compute
    station_disp = np.zeros((n_eq, stations_lla.shape[0], 3))
    if "num_threads" in config.options("general"):
        with Pool(config.getint("general", "num_threads")) as p:
            results = list(tqdm(p.imap(get_displacements, parameters), ascii=True, total=n_eq, desc="Simulating Earthquake Displacements", unit="eq"))
        for i, r in enumerate(results):
            station_disp[i, :, :] = r
    else:
        for i, param in tqdm(enumerate(parameters), ascii=True, total=n_eq, desc="Simulating Earthquake Displacements", unit="eq"):
            station_disp[i, :, :] = get_displacements(param)

    # add steps to station timeseries if they exceed the threshold
    for istat, stat_name in enumerate(net.network_locations.keys()):
        stat_time = net.stations[stat_name].timeseries[config.get('catalog_prior', 'timeseries')].time.values
        steptimes = []
        for itime in range(len(stat_time) - 1):
            disp = station_disp[(eq_times > stat_time[itime]) & (eq_times <= stat_time[itime + 1]), istat, :]
            cumdisp = np.sum(np.linalg.norm(disp, axis=1), axis=None)
            if cumdisp >= config.getfloat('catalog_prior', 'threshold'):
                steptimes.append(str(stat_time[itime + 1]))
        net.stations[stat_name].add_local_model(config.get('catalog_prior', 'timeseries'), config.get('catalog_prior', 'model'), Step(steptimes=steptimes, regularize=config.getboolean('catalog_prior', 'regularize')))


if __name__ == "__main__":
    # net = Network.from_json(path="net_arch.json")
    # net = Network.from_json(path="net_arch_catalog1mm.json")
    net = Network.from_json(path="net_arch_lite.json")
    # okada_prior(net, "data/nied_fnet_catalog.txt")
    # net.fit()
    # net.fit("GNSS", solver="ridge_regression", penalty=1e10)
    # net.evaluate("GNSS")
    # net.to_json(path="arch_out.json")
    net.call_netwide_func('GNSS', 'median', ts_out='netwide', kernel_size=7)
    net.gui()
