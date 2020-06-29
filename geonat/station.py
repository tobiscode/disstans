"""
This module contains the :class:`~Station` class, which is the main
component of a :class:`~geonat.network.Network` and contains
:class:`~geonat.timeseries.Timeseries`, :class:`~geonat.model.Model` and
associated fits.
"""

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
    such as location as well as timeseries, models and fits.

    Timeseries are saved in the :attr:`~timeseries` dictionary, and their associated
    models and fits are saved in the :attr:`~models~ and :attr:~~fits`
    dictionaries, respectively.

    Parameters
    ----------
    name : str
        Name of the station.
    location : tuple, list, numpy.ndarray
        Location (Latitude [°], Longitude [°], Altitude [m]) of the station.
    """
    def __init__(self, name, location):
        self.name = name
        """ Name of the station. """
        self.location = location
        """ Location of the station (Latitude [°], Longitude [°], Altitude [m]). """
        self.timeseries = {}
        """
        Dictionary of timeseries, where the keys are their string descriptions
        and the values are :class:`~geonat.timeseries.Timeseries` instances.
        """
        self.models = {}
        """
        Dictionary of dictionaries of :class:`~geonat.model.Model` associated
        to a timeseries saved in :attr:`~location`.

        Example
        -------
        If ``'myts'`` is a timeseries saved in the station ``stat``,
        and ``'mymodel'`` is the name of a model that is associated with it,
        then the model is saved as::

            stat.models['myts']['mymodel']
        """
        self.unused_models = {}
        """
        Same as :attr:`~models` but for models whose timeseries is not present
        in :attr:`~timeseries`.
        """
        self.fits = {}
        """
        Dictionary of dictionaries of fitted :class:`~geonat.timeseries.Timeseries`
        associated to a timeseries saved in :attr:`~location`.

        Example
        -------
        If ``'myts'`` is a timeseries saved in the station ``stat``,
        and ``'mymodel'`` is the name of a model that is associated with it,
        then the fitted timeseries is saved as::

            stat.fits['myts']['mymodel']
        """

    def __repr__(self):
        """
        Special function that returns a readable summary of the station.
        Accessed, for example, by Python's ``print()`` built-in function.

        Returns
        -------
        info : str
            Station summary.
        """
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

    def __getitem__(self, ts_description):
        """
        Convenience special function that provides a shorthand notation
        to access the station's timeseries.

        Parameters
        ----------
        ts_description : str
            Description of the timeseries.

        Returns
        -------
        geonat.timeseries.Timeseries
            Timeseries in station.

        Example
        -------
        If ``stat`` is a :class:`~Station` instance and ``ts_description`` the
        ts_description of a station, the following two are equivalent::

            stat.timeseries[ts_description]
            stat[ts_description]
        """
        if ts_description not in self.timeseries:
            raise KeyError(f"Station {self.name}: No timeseries '{ts_description}' present.")
        return self.timeseries[ts_description]

    def __setitem__(self, ts_description, timeseries):
        """
        Convenience special function that allows a dictionary-like adding of timeseries to
        the station by wrapping :meth:`~add_timeseries`.

        See Also
        --------
        add_timeseries : Add a timeseries to the station instance.
        """
        self.add_timeseries(ts_description, timeseries)

    def __delitem__(self, ts_description):
        """
        Convenience special function that allows a dictionary-like removing of timeseries from
        the station by wrapping :meth:`~remove_timeseries`.

        See Also
        --------
        remove_timeseries : Remove a timeseries from the station instance.
        """
        self.remove_timeseries(ts_description)

    def __iter__(self):
        """
        Convenience special function that allows for a shorthand notation to quickly
        iterate over all timeseries.

        Example
        -------
        If ``stat`` is a :class:`~Station` instance, then the following two loops are equivalent::

            # long version
            for ts in stat.timeseries.values():
                pass
            # shorthand
            for ts in stat:
                pass
        """
        for ts in self.timeseries.values():
            yield ts

    @property
    def ts(self):
        """
        Abbreviation for :attr:`~timeseries`.
        """
        return self.timeseries

    def get_arch(self):
        """
        Build a dictionary describing the architecture of this station,
        to be used when creating a network JSON configuration file.

        See Also
        --------
        geonat.network.Network.to_json : Export the Network configuration as a JSON file.
        geonat.timeseries.Timeseries.get_arch
            Get the architecture dictionary of a :class:`~geonat.timeseries.Timeseries` instance.
        geonat.model.Model.get_arch
            Get the architecture dictionary of a :class:`~geonat.model.Model` instance.
        """
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

    def add_timeseries(self, ts_description, timeseries, override_src=None, override_data_unit=None,
                       override_data_cols=None, override_sigma_cols=None, add_models=None):
        """
        Add a timeseries to the station.

        Optionally, can override some Timeseries attributes when adding the object,
        and add models.

        Parameters
        ----------
        ts_description : str
            Description of the timeseries.
        timeseries : geonat.timeseries.Timeseries
            Timeseries object to add.
        override_src : str, optional
            Override the :attr:`~geonat.timeseries.Timeseries.src` attribute of ``timeseries``.
        override_data_unit : str, optional
            Override the :attr:`~geonat.timeseries.Timeseries.data_unit` attribute of ``timeseries``.
        override_data_cols : str, optional
            Override the :attr:`~geonat.timeseries.Timeseries.data_cols` attribute of ``timeseries``.
        override_sigma_cols : str, optional
            Override the :attr:`~geonat.timeseries.Timeseries.sigma_cols` attribute of ``timeseries``.
        add_models : dict, optional
            Dictionary of models to add to the timeseries, where the keys are the model description
            and the values are :class:`~geonat.model.Model` objects.

        See Also
        --------
        __setitem__ : Shorthand notation wrapper when no optional arguments are necessary.

        Example
        -------
        If ``stat`` is a :class:`~Station` instance, ``ts_description`` the ts_description of a new timeseries,
        and ``ts`` a :class:`~geonat.timeseries.Timeseries` instance, then the following two are equivalent::

            stat.add_timeseries(ts_description, timeseries)
            stat[ts_description] = timeseries
        """
        if not isinstance(ts_description, str):
            raise TypeError("Cannot add new timeseries: 'ts_description' is not a string.")
        if not isinstance(timeseries, Timeseries):
            raise TypeError("Cannot add new timeseries: 'timeseries' is not a Timeseries object.")
        if ts_description in self.timeseries:
            warn(f"Station {self.name}: Overwriting time series '{ts_description}'.",
                 category=RuntimeWarning)
        if override_src is not None:
            timeseries.src = override_src
        if override_data_unit is not None:
            timeseries.data_unit = override_data_unit
        if override_data_cols is not None:
            timeseries.data_cols = override_data_cols
        if override_sigma_cols is not None:
            timeseries.sigma_cols = override_sigma_cols
        self.timeseries[ts_description] = timeseries
        self.fits[ts_description] = {}
        self.models[ts_description] = {}
        if add_models is not None:
            for model_description, model_cfg in add_models.items():
                local_copy = deepcopy(model_cfg)
                mdl = getattr(geonat_models, local_copy["type"])(**local_copy["kw_args"])
                self.add_local_model(ts_description=ts_description, model_description=model_description, model=mdl)

    def remove_timeseries(self, ts_description):
        """
        Remove a timeseries (including associated fits and models) from the station.

        Parameters
        ----------
        ts_description : str
            Description of the timeseries.

        See Also
        --------
        __setitem__ : Shorthand notation wrapper.

        Example
        -------
        If ``stat`` is a :class:`~Station` instance, ``ts_description`` the ts_description of the timeseries
        to remove, then the following two are equivalent::

            stat.remove_timeseries(ts_description)
            del stat[ts_description]
        """
        if ts_description not in self.timeseries:
            warn(f"Station {self.name}: Cannot find time series '{ts_description}', couldn't delete.",
                 category=RuntimeWarning)
        else:
            del self.timeseries[ts_description]
            del self.fits[ts_description]
            del self.models[ts_description]

    def add_local_model(self, ts_description, model_description, model):
        """
        Add a model to a timeseries (overwrites the model if it has already
        been added with the same description).

        Parameters
        ----------
        ts_description : str
            Timeseries to add the model to.
        model_description : str
            Model description.
        model : geonat.model.Model
            Model object.
        """
        if not isinstance(ts_description, str):
            raise TypeError("Cannot add new local model: 'ts_description' is not a string.")
        if not isinstance(model_description, str):
            raise TypeError("Cannot add new local model: 'model_description' is not a string.")
        if not isinstance(model, Model):
            raise TypeError("Cannot add new local model: 'model' is not a Model object.")
        assert ts_description in self.timeseries, \
            f"Station {self.name}: Cannot find timeseries '{ts_description}' to add local model '{model_description}'."
        if model_description in self.models[ts_description]:
            warn(f"Station {self.name}, timeseries {ts_description}: Overwriting local model '{model_description}'.",
                 category=RuntimeWarning)
        self.models[ts_description].update({model_description: model})

    def remove_local_model(self, ts_description, model_description):
        """
        Remove a model from a timeseries.

        Parameters
        ----------
        ts_description : str
            Timeseries to remove the model from.
        model_description : str
            Model description.
        """
        if ts_description not in self.timeseries:
            warn(f"Station {self.name}: Cannot find timeseries '{ts_description}', "
                 f"couldn't delete local model '{model_description}'.", category=RuntimeWarning)
        elif model_description not in self.models[ts_description]:
            warn(f"Station {self.name}, timeseries {ts_description}: "
                 f"Cannot find local model '{model_description}', couldn't delete.", category=RuntimeWarning)
        else:
            del self.models[ts_description][model_description]

    def add_fit(self, ts_description, model_description, fit):
        """
        Add a fit dictionary to a timeseries' model (overwrites the fit if it has
        already been added for the model).

        Parameters
        ----------
        ts_description : str
            Timeseries to add the fit to.
        model_description : str
            Model description the fit applies to.
        fit : dict
            Dictionary with the keys ``'time'``, ``'fit'`` and ``'sigma'``.

        Returns
        -------
        fit_ts : geonat.timeseries.Timeseries
            The fit as a Timeseries object.

        See Also
        --------
        geonat.timeseries.Timeseries.from_fit
            Convert a fit dictionary to a :class:`~geonat.timeseries.Timeseries` object.
        """
        if not isinstance(ts_description, str):
            raise TypeError("Cannot add new fit: 'ts_description' is not a string.")
        if not isinstance(model_description, str):
            raise TypeError("Cannot add new fit: 'model_description' is not a string.")
        assert ts_description in self.timeseries, \
            f"Station {self.name}: Cannot find timeseries '{ts_description}' to add fit for model '{model_description}'."
        assert model_description in self.models[ts_description], \
            f"Station {self.name}, timeseries {ts_description}: Cannot find local model '{model_description}', couldn't add fit."
        if model_description in self.fits[ts_description]:
            warn(f"Station {self.name}, timeseries {ts_description}: Overwriting fit of local model '{model_description}'.",
                 category=RuntimeWarning)
        data_cols = [ts_description + "_" + model_description + "_" + dcol for dcol in self.timeseries[ts_description].data_cols]
        fit_ts = Timeseries.from_fit(self.timeseries[ts_description].data_unit, data_cols, fit)
        self.fits[ts_description].update({model_description: fit_ts})
        return fit_ts

    def remove_fit(self, ts_description, model_description):
        """
        Remove a fit from a timeseries' model.

        Parameters
        ----------
        ts_description : str
            Timeseries to remove the fit from.
        model_description : str
            Model description to remove the fit from.
        """
        if ts_description not in self.timeseries:
            warn(f"Station {self.name}: Cannot find timeseries '{ts_description}', "
                 f"couldn't delete fit for model '{model_description}'.", category=RuntimeWarning)
        elif model_description not in self.models[ts_description]:
            warn(f"Station {self.name}, timeseries {ts_description}: "
                 f"Cannot find local model '{model_description}', couldn't delete fit.", category=RuntimeWarning)
        elif model_description not in self.fits[ts_description]:
            warn(f"Station {self.name}, timeseries {ts_description}: "
                 f"Cannot find fit for local model '{model_description}', couldn't delete.", category=RuntimeWarning)
        else:
            del self.fits[ts_description][model_description]

    def analyze_residuals(self, ts_description, mean=False, std=False, n_observations=False, std_outlier=0):
        """
        Analyze, print and return the residuals of a station's timeseries according
        to certain metrics defined in the arguments.

        Parameters
        ----------
        ts_description : str
            Timeseries to analyze. Method assumes it already is a residual.
        mean : bool, optional
            If ``True``, calculate the mean of the timeseries.
            Adds the key ``'Mean'`` to the output dictionary.
            Defaults to ``False``.
        std : bool, optional
            If ``True``, calculate the standard deviation of the timeseries.
            Adds the key ``'Standard Deviation'`` to the output dictionary.
            Defaults to ``False``.
        n_observations : bool, optional
            If ``True``, count the number of observations (excluding NaNs) and NaNs.
            Adds the keys ``'Observations'`` and ``'Gaps'`` to the output dictionary.
            Defaults to ``False``.
        std_outlier : int, float
            If ``std_outlier > 0``, count the number of non-NaN outliers, defined
            by the number of standard deviations they are away from the mean.
            Adds the key ``'Outliers'`` to the output dictionary.
            Defaults to ``0``.

        Returns
        -------
        results : dict
            Dictionary that includes the results as defined by the arguments.
            Empty by default.
        """
        assert isinstance(ts_description, str), \
            f"Station {self.name}: 'ts_description' needs to be a string, got {type(ts_description)}."
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
        return results