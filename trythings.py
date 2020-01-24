import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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

    def add_station(self, name, station):
        if name in self.stations:
            Warning("Overwriting station {:s}".format(name))
        self.stations[name] = station
        self.network_locations[name] = station.location

    def add_global_model(self, description, model):
        if description in self.global_models:
            Warning("Overwriting global model {:s}".format(description))
        self.global_models[description] = model

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
        for station_name, station_cfg in net_arch["stations"].items():
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

    def fit(self, solver):
        if "num_threads" in config.options("general"):
            # collect station calls
            iterable_inputs = [(stat, solver) for stat in self.stations.values()]
            # start multiprocessing pool
            with Pool(config.getint("general", "num_threads")) as p:
                fit_output = list(tqdm(p.imap(Station.fit_models, iterable_inputs), ascii=True, desc="Fitting station models", total=len(iterable_inputs), unit="station"))
            # redistribute results
            for i, stat in enumerate(self.stations.values()):
                fitted_params = fit_output[i]
                for ts_description, fitted_params in fit_output[i].items():
                    for model_description, (params, covs) in fitted_params.items():
                        stat.models[ts_description][model_description].read_parameters(params, covs)
        else:
            # run in serial
            for name, stat in tqdm(self.stations.items(), desc="Fitting station models", ascii=True, unit="station"):
                fit_output = Station.fit_models((stat, solver))
                for ts_description, fitted_params in fit_output.items():
                    for model_description, (params, covs) in fitted_params.items():
                        stat.models[ts_description][model_description].read_parameters(params, covs)

    def evaluate(self, timeseries=None):
        if "num_threads" in config.options("general"):
            # collect station calls
            iterable_inputs = [(stat, timeseries) for stat in self.stations.values()]
            # start multiprocessing pool
            with Pool(config.getint("general", "num_threads")) as p:
                eval_output = list(tqdm(p.imap(Station.evaluate_models, iterable_inputs), ascii=True, desc="Evaluating station models", total=len(iterable_inputs), unit="station"))
            # redistribute results
            for i, stat in enumerate(self.stations.values()):
                stat_fits = eval_output[i]
                for ts_description in stat_fits.keys():
                    for model_description, fit in stat_fits[ts_description].items():
                        stat.add_fit(ts_description, model_description, fit)
        else:
            # run in serial
            for name, stat in tqdm(self.stations.items(), desc="Evaluating station models", ascii=True, unit="station"):
                stat_fits = Station.evaluate_models((stat, timeseries))
                for ts_description in stat_fits.keys():
                    for model_description, fit in stat_fits[ts_description].items():
                        stat.add_fit(ts_description, model_description, fit)

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
                        ax.fill_between(ts.df['time'], ts.df[data_col] + config.getfloat("gui", "plot_sigmas") * ts.df[sigma_col],
                                        ts.df[data_col] - config.getfloat("gui", "plot_sigmas") * ts.df[sigma_col], facecolor='gray',
                                        alpha=config.getfloat("gui", "plot_sigmas_alpha"), linewidth=0)
                    # plot data
                    ax.plot(ts.df['time'], ts.df[data_col], marker='.', color='k', label="Data")
                    # overlay models
                    for (mdl_description, fit) in self.stations[station_name].fits[ts_description].items():
                        if fit['sigma'] is not None and "plot_sigmas" in config.options("gui"):
                            ax.fill_between(fit['time'], fit['fit'][:, icol] + config.getfloat("gui", "plot_sigmas") * fit['sigma'][:, icol],
                                            fit['fit'][:, icol] - config.getfloat("gui", "plot_sigmas") * fit['sigma'][:, icol],
                                            alpha=config.getfloat("gui", "plot_sigmas_alpha"), linewidth=0)
                        ax.plot(fit['time'], fit['fit'][:, icol], label=mdl_description)
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

    # def aggregate_mappings(self):
    #     data = pd.DataFrame()
    #     uncertainty = pd.DataFrame()
    #     mapping = pd.DataFrame()
    #     for name, station in self.stations.items():
    #         for description, timeseries in station.timeseries.items():
    #             # collect data
    #             temp = timeseries.df[['time'].extend(timeseries.data_cols)]
    #             new_cols = ["{:s}_{:s}_{:s}".format(name, description, old_col) for old_col in timeseries.data_cols]
    #             temp.rename(columns=dict(zip(timeseries.data_cols, new_cols)))
    #             data.append(temp)
    #             # if present, collect uncertainties
    #             if timeseries.sigma_cols is not None:
    #                 temp = timeseries.df[['time'].extend(timeseries.sigma_cols)]
    #                 # match the name of the uncertainty column with the data column, now that they are in different dataframes
    #                 new_cols = ["{:s}_{:s}_{:s}".format(name, description, old_col) for old_col in timeseries.data_cols]
    #                 temp.rename(columns=dict(zip(timeseries.sigma_cols, new_cols)))
    #                 uncertainty.append(temp)
    #             # else, fill with default uncertainty or NaN
    #             else:
    #                 temp[new_cols] = np.NaN
    #                 uncertainty.append(temp)
    #             # generate mapping matrix for timeseries
    #             for (model_description, model), component in product(station.models.items(), timeseries.data_cols):
    #                 coefs = model.get_mapping(timevector=timeseries.df['time'])
    #                 coef_names = ["{:s}_{:s}_{:s}_{:s}_{:d}".format(name, description, component, model_description, parameter) for parameter in model.num_parameters]
    #                 temp = timeseries.df['time'].join(pd.DataFrame(data=coefs, columns=coef_names))
    #                 mapping.append(temp)


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
        assert ts_description in self.timeseries, \
            "Station {:s}: Cannot find timeseries {:s} to add fit for model {:s}".format(self.name, ts_description, model_description)
        assert model_description in self.models[ts_description], \
            "Station {:s}, timeseries {:s}: Cannot find local model {:s}, couldn't add fit".format(self.name, ts_description, model_description)
        if model_description in self.fits[ts_description]:
            Warning("Station {:s}, timeseries {:s}: Overwriting fit of local model {:s}".format(self.name, ts_description, model_description))
        self.fits[ts_description].update({model_description: fit})

    def remove_fit(self, ts_description, model_description, fit):
        if ts_description not in self.timeseries:
            Warning("Station {:s}: Cannot find timeseries {:s}, couldn't delete fit for model {:s}".format(self.name, ts_description, model_description))
        elif model_description not in self.models[ts_description]:
            Warning("Station {:s}, timeseries {:s}: Cannot find local model {:s}, couldn't delete fit".format(self.name, ts_description, model_description))
        elif model_description not in self.fits[ts_description]:
            Warning("Station {:s}, timeseries {:s}: Cannot find fit for local model {:s}, couldn't delete".format(self.name, ts_description, model_description))
        else:
            del self.fits[ts_description][model_description]

    @staticmethod
    def fit_models(station_and_solver):
        station, solver = station_and_solver
        fitted_params = {}
        for (ts_description, timeseries) in station.timeseries.items():
            fitted_params[ts_description] = globals()[solver](timeseries, station.models[ts_description])
        return fitted_params

    @staticmethod
    def evaluate_models(station_and_timeseries):
        if len(station_and_timeseries) == 1:
            station = station_and_timeseries
            timeseries = None
        else:
            station, timeseries = station_and_timeseries
        fit = {}
        for ts_description, ts in station.timeseries.items():
            fit[ts_description] = {}
            for model_description, model in station.models[ts_description].items():
                if model.is_fitted:
                    fit[ts_description][model_description] = model.evaluate(ts.df['time']) if timeseries is None else model.evaluate(timeseries)
        return fit


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

    def get_arch(self):
        raise NotImplementedError()


class GipsyTimeseries(Timeseries):
    """
    Timeseries subclass for GNSS measurements in JPL's Gipsy `.tseries` file format.
    """
    def __init__(self, path):
        self._path = path
        time = pd.to_datetime(pd.read_csv(self._path, delim_whitespace=True, header=0, usecols=[11, 12, 13, 14, 15, 16],
                                          names=['year', 'month', 'day', 'hour', 'minute', 'second'])).to_frame(name='time')
        data = pd.read_csv(self._path, delim_whitespace=True, header=0, usecols=[1, 2, 3, 4, 5, 6], names=['east', 'north', 'up', 'east_sigma', 'north_sigma', 'up_sigma'])
        super().__init__(dataframe=time.join(data), source='tseries', data_unit='m',
                         data_cols=['east', 'north', 'up'], sigma_cols=['east_sigma', 'north_sigma', 'up_sigma'])

    def get_arch(self):
        return {"type": "GipsyTimeseries",
                "kw_args": {"path": self._path}}


class Model():
    """
    General class that defines what a model can have as an input and output.
    Defaults to a linear model.
    """
    def __init__(self, num_parameters):
        self.num_parameters = num_parameters
        self.is_fitted = False
        self.parameters = None
        self.sigmas = None

    def get_arch(self):
        raise NotImplementedError()

    def get_mapping(self, timevector):
        raise NotImplementedError()

    def read_parameters(self, parameters, sigmas):
        assert parameters.shape[0] == self.num_parameters, "Read-in parameters have different size than the instantiated model."
        self.parameters = parameters
        if sigmas is not None:
            assert self.parameters.size == sigmas.size, "Uncertainty sigmas must have same number of entries than parameters."
            self.sigmas = sigmas
        self.is_fitted = True

    def evaluate(self, timevector):
        if not self.is_fitted:
            RuntimeError("Cannot evaluate the model before reading in parameters.")
        mapping_matrix = self.get_mapping(timevector=timevector)
        fit = mapping_matrix @ self.parameters
        fit_sigma = mapping_matrix @ self.sigmas
        return {"time": timevector, "fit": fit, "sigma": fit_sigma}


class Step(Model):
    """
    Step functions at given times.
    """
    def __init__(self, steptimes):
        self._steptimes = steptimes
        self.timestamps = [pd.Timestamp(step) for step in self._steptimes]
        self.timestamps.sort()
        super().__init__(num_parameters=len(self.timestamps))

    def get_arch(self):
        return {"type": "Step",
                "kw_args": {"steptimes": self._steptimes}}

    def _update_from_steptimes(self):
        self.timestamps = [pd.Timestamp(step) for step in self._steptimes]
        self.timestamps.sort()
        self.num_parameters = len(self.timestamps)
        self.is_fitted = False
        self.parameters = None
        self.sigmas = None

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
    def __init__(self, order=1, starttime=None, timeunit='Y'):
        self.order = order
        self.starttime = starttime
        self.timeunit = timeunit
        super().__init__(num_parameters=self.order + 1)

    def get_arch(self):
        return {"type": "Polynomial",
                "kw_args": {"order": self.order,
                            "starttime": self.starttime,
                            "timeunit": self.timeunit}}

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
    def __init__(self, period=1, starttime=None, timeunit='Y'):
        self.period = period
        self.starttime = starttime
        self.timeunit = timeunit
        super().__init__(num_parameters=2)

    def get_arch(self):
        return {"type": "Sinusoidal",
                "kw_args": {"period": self.period,
                            "starttime": self.starttime,
                            "timeunit": self.timeunit}}

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


def linear_regression(ts, models):
    """
    numpy.linalg wrapper for a linear least squares solver
    """
    mapping_matrices = []
    # get mapping matrices
    for (mdl_description, model) in models.items():
        mapping_matrices.append(model.get_mapping(ts.df['time']))
    mapping_matrices = np.hstack(mapping_matrices)
    num_time, num_params = mapping_matrices.shape
    num_components = len(ts.data_cols)
    # perform fit and estimate formal covariance (uncertainty) of parameters
    params = np.zeros((num_params, num_components))
    sigmas = np.zeros((num_params, num_components))
    for i in range(num_components):
        if ts.sigma_cols[i] is None:
            GtWG = mapping_matrices.T @ mapping_matrices
            GtWd = mapping_matrices.T @ ts.df[ts.data_cols[i]].values.reshape(-1, 1)
        else:
            GtW = dmultr(mapping_matrices.T, 1/ts.df[ts.sigma_cols[i]].values**2)
            GtWG = GtW @ mapping_matrices
            GtWd = GtW @ ts.df[ts.data_cols[i]].values.reshape(-1, 1)
        params[:, i] = np.linalg.lstsq(GtWG, GtWd, rcond=None)[0].squeeze()
        sigmas[:, i] = np.sqrt(np.diag(np.linalg.pinv(GtWG)))
    # separate parameters back to models
    i = 0
    fitted_params = {}
    for (mdl_description, model) in models.items():
        fitted_params[mdl_description] = (params[i:i+model.num_parameters, :], sigmas[i:i+model.num_parameters, :])
        i += model.num_parameters
    return fitted_params


def ridge_regression(ts, models, penalty, reg_indices=None):
    """
    numpy.linalg wrapper for a linear L2-regularized least squares solver
    """
    mapping_matrices = []
    # get mapping matrices
    for (mdl_description, model) in models.items():
        mapping_matrices.append(model.get_mapping(ts.df['time']))
    mapping_matrices = np.hstack(mapping_matrices)
    num_time, num_params = mapping_matrices.shape
    num_components = len(ts.data_cols)
    # build regularization matrix
    if reg_indices is None:
        reg_mat = np.diag(np.ones(num_params) * penalty)
    else:
        reg_mat = np.zeros((num_params, num_params))
        reg_mat[reg_indices, reg_indices] = penalty
    # perform fit and estimate formal covariance (uncertainty) of parameters
    params = np.zeros((num_params, num_components))
    sigmas = np.zeros((num_params, num_components))
    for i in range(num_components):
        if ts.sigma_cols[i] is None:
            GtWG = mapping_matrices.T @ mapping_matrices
            GtWd = mapping_matrices.T @ ts.df[ts.data_cols[i]].values.reshape(-1, 1)
        else:
            GtW = dmultr(mapping_matrices.T, 1/ts.df[ts.sigma_cols[i]].values**2)
            GtWG = GtW @ mapping_matrices
            GtWd = GtW @ ts.df[ts.data_cols[i]].values.reshape(-1, 1)
        GtWGreg = GtWG + reg_mat
        params[:, i] = np.linalg.lstsq(GtWGreg, GtWd, rcond=None)[0].squeeze()
        sigmas[:, i] = np.sqrt(np.diag(np.linalg.pinv(GtWGreg)))
    # separate parameters back to models
    i = 0
    fitted_params = {}
    for (mdl_description, model) in models.items():
        fitted_params[mdl_description] = (params[i:i+model.num_parameters, :], sigmas[i:i+model.num_parameters, :])
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


if __name__ == "__main__":
    net = Network.from_json(path="net_arch.json")
    net.stations["0001"].models["GNSS"]["Maintenance"].add_step("2001-12-05")
    net.stations["0002"].models["GNSS"]["Maintenance"].remove_step("2001-12-05")
    net.fit("linear_regression")
    net.evaluate()
    net.to_json(path="arch_out.json")
    net.gui()
