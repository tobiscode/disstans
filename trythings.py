import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
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
    def __init__(self, name):
        self.name = name
        self.stations = {}
        self.network_locations = {}
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
        # create Network instance
        net = cls(network_name)
        # load location information, if present
        network_locations_path = net_arch.get("locations")
        if network_locations_path is not None:
            with open(network_locations_path, mode='r') as locfile:
                loclines = [line.strip() for line in locfile.readlines()]
            network_locations = {line.split()[0]: [float(lla) for lla in line.split()[1:]] for line in loclines}
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
            for ts_cfg in station_cfg["timeseries"]:
                ts = globals()[ts_cfg["type"]](**ts_cfg["kw_args"])
                stat.add_timeseries(description=ts_cfg["description"], timeseries=ts)
                # add default local models to station
                for model_cfg in net_arch["default_local_models"]:
                    mdl = globals()[model_cfg["type"]](**model_cfg["kw_args"])
                    stat.add_local_model(ts_description=ts_cfg["description"], model_description=model_cfg["description"], model=mdl)
                # add specific local models to station
                for model_cfg in station_cfg["models"]:
                    mdl = globals()[model_cfg["type"]](**model_cfg["kw_args"])
                    stat.add_local_model(ts_description=ts_cfg["description"], model_description=model_cfg["description"], model=mdl)
            # add to network
            net.add_station(name=station_name, station=stat)
        # add global models
        for model_cfg in net_arch["global_models"]:
            mdl = globals()[model_cfg["type"]](**model_cfg["kw_args"])
            net.add_global_model(description=model_cfg["description"], model=mdl)
        return net

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
        ax_map.plot(stat_lons, stat_lats, linestyle='None', marker='.', transform=proj_lla)
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

    def add_timeseries(self, description, timeseries):
        if description in self.timeseries:
            Warning("Overwriting time series {:s} at station {:s}".format(description, self.name))
        self.timeseries[description] = timeseries
        self.fits[description] = {}
        self.models[description] = {}

    def add_local_model(self, ts_description, model_description, model):
        assert ts_description in self.timeseries, \
            "Cannot find timeseries {:s} at station {:s}".format(ts_description, self.name)
        if model_description in self.models[ts_description]:
            Warning("Overwriting local model {:s} for timeseries {:s} at station {:s}".format(model_description, ts_description, self.name))
        self.models[ts_description].update({model_description: model})

    def add_fit(self, ts_description, model_description, fit):
        assert ts_description in self.timeseries, \
            "Fitted model must match a previously added local station time series before being assigned as a fit."
        assert model_description in self.models[ts_description], \
            "Fitted model must match a previously added local station model before being assigned as a fit."
        if model_description in self.fits[ts_description]:
            Warning("Overwriting fit of {:s} model to {:s} timeseries at station {:s}".format(model_description, ts_description, self.name))
        self.fits[ts_description].update({model_description: fit})

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


class GipsyTimeseries(Timeseries):
    """
    Timeseries subclass for GNSS measurements in JPL's Gipsy `.tseries` file format.
    """
    def __init__(self, path):
        time = pd.to_datetime(pd.read_csv(path, delim_whitespace=True, header=0, usecols=[11, 12, 13, 14, 15, 16],
                                          names=['year', 'month', 'day', 'hour', 'minute', 'second'])).to_frame(name='time')
        data = pd.read_csv(path, delim_whitespace=True, header=0, usecols=[1, 2, 3, 4, 5, 6], names=['east', 'north', 'up', 'east_sigma', 'north_sigma', 'up_sigma'])
        super().__init__(dataframe=time.join(data), source='tseries', data_unit='m',
                         data_cols=['east', 'north', 'up'], sigma_cols=['east_sigma', 'north_sigma', 'up_sigma'])


class Model():
    """
    General class that defines what a model can have as an input and output.
    Defaults to a linear model.
    """
    def __init__(self, num_parameters, starttime):
        self.num_parameters = num_parameters
        self.starttime = starttime
        self.is_fitted = False
        self.parameters = None
        self.sigmas = None

    def get_mapping(self, timevector):
        raise NotImplementedError()

    def read_parameters(self, parameters, sigmas):
        assert parameters.shape[0] == self.num_parameters, "Cannot change number of parameters after model initialization."
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
    Step function at a given time.
    """
    def __init__(self, starttime):
        super().__init__(num_parameters=1, starttime=starttime)

    def get_mapping(self, timevector):
        coefs = np.zeros((timevector.size, 1))
        coefs[pd.Timestamp(self.starttime) >= timevector, :] = 1
        return coefs


class Polynomial(Model):
    """
    Polynomial of given order.

    `timeunit` can be the following (see https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html#datetime-units):
        `Y`, `M`, `W`, `D`, `h`, `m`, `s`, `ms`, `us`, `ns`, `ps`, `fs`, `as`
    """
    def __init__(self, order=1, starttime=None, timeunit='Y'):
        self.order = order
        self.timeunit = timeunit
        super().__init__(num_parameters=self.order + 1, starttime=starttime)

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
        self.timeunit = timeunit
        super().__init__(num_parameters=2, starttime=starttime)

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


def linear_least_squares(ts, models):
    """
    cvxpy wrapper for a linear least squares solver
    """
    mapping_matrices = []
    # get mapping matrices
    for (model_desc, model) in models.items():
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
    for (model_desc, model) in models.items():
        fitted_params[model_desc] = (params[i:i+model.num_parameters, :], sigmas[i:i+model.num_parameters, :])
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
    net.fit("linear_least_squares")
    net.evaluate()
    net.gui()
