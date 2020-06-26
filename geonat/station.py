import numpy as np
import pandas as pd
from copy import deepcopy
from warnings import warn

from . import model as geonat_models
from .model import Model
from .timeseries import Timeseries


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
        self.unused_models = {}
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
                mdl = getattr(geonat_models, local_copy["type"])(**local_copy["kw_args"])
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
        if results:  # only print if any statistic was recorded
            print()
            print(pd.DataFrame(data=results, index=self[ts_description].data_cols).rename_axis(f"{self.name}: {ts_description}", axis=1))
