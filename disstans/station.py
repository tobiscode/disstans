"""
This module contains the :class:`~Station` class, which is the main
component of a :class:`~disstans.network.Network` and contains
:class:`~disstans.timeseries.Timeseries`, :class:`~disstans.models.Model` and
associated fits.
"""

import numpy as np
import pandas as pd
import scipy as sp
import scipy.sparse as sparse
from copy import deepcopy
from warnings import warn
from collections.abc import Iterator
from typing import Any

from . import models as disstans_models
from .timeseries import Timeseries
from .models import Model, ModelCollection, FitCollection
from .tools import tvec_to_numpycol


class Station():
    """
    Representation of a station, which contains both metadata
    such as location as well as timeseries, models and fits.

    Timeseries are saved in the :attr:`~timeseries` dictionary, and their associated
    models and fits are saved in the :attr:`~models` and :attr:`~fits`
    dictionaries, respectively.

    Parameters
    ----------
    name
        Name of the station.
    location
        Location (Latitude [°], Longitude [°], Altitude [m]) of the station.
    """
    def __init__(self,
                 name: str,
                 location: list[float] | tuple[float, float, float] | np.ndarray
                 ) -> np.ndarray:
        self.name = str(name)
        """ Name of the station. """
        if isinstance(location, np.ndarray):
            location = location.squeeze().tolist()
        assert all([isinstance(v, float) for v in location]) and len(location) == 3
        self.location = location
        """ Location of the station (Latitude [°], Longitude [°], Altitude [m]). """
        self.timeseries = {}
        """
        Dictionary of timeseries, where the keys are their string descriptions
        and the values are :class:`~disstans.timeseries.Timeseries` instances.
        """
        self.models = {}
        """
        Dictionary of :class:`~disstans.models.ModelCollection` objects associated
        with a timeseries saved in :attr:`~timeseries`.

        Example
        -------
        If ``'myts'`` is a timeseries saved in the station ``stat``,
        then the model collection object for that timeseries is stored in::

            stat.models['myts']
        """
        self.unused_models = {}
        """
        Same as :attr:`~models` but for models whose timeseries is not present
        in :attr:`~timeseries`.
        """
        self.fits = {}
        """
        Dictionary of :class:`~disstans.models.FitCollection` objects
        associated with a timeseries saved in :attr:`~timeseries`.

        Example
        -------
        If ``'myts'`` is a timeseries saved in the station ``stat``,
        and ``'mymodel'`` is the name of a model that is associated with it,
        then the fitted timeseries is saved as::

            stat.fits['myts']['mymodel']
        """

    def __str__(self) -> str:
        """
        Special function that returns a readable summary of the station.
        Accessed, for example, by Python's ``print()`` built-in function.

        Returns
        -------
            Station summary.
        """
        info = f"Station {self.name} at {self.location} with timeseries"
        for ts_description, ts in self.timeseries.items():
            info += str(ts).replace("Timeseries", f"\n{ts_description}")
            if len(self.models[ts_description]) > 0:
                info += f"\n - Models: {self.models[ts_description].model_names}"
            if len(self.fits[ts_description]) > 0:
                info += f"\n - Fits: {[str(key) for key in self.fits[ts_description]]}"
        return info

    def __getitem__(self, ts_description: str) -> Timeseries:
        """
        Convenience special function that provides a shorthand notation
        to access the station's timeseries.

        Parameters
        ----------
        ts_description
            Description of the timeseries.

        Returns
        -------
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

    def __setitem__(self, ts_description: str, timeseries: Timeseries) -> None:
        """
        Convenience special function that allows a dictionary-like adding of timeseries to
        the station by wrapping :meth:`~add_timeseries`.

        See Also
        --------
        add_timeseries : Add a timeseries to the station instance.
        """
        self.add_timeseries(ts_description, timeseries, warn_existing=False)

    def __delitem__(self, ts_description: str) -> None:
        """
        Convenience special function that allows a dictionary-like removing of timeseries from
        the station by wrapping :meth:`~remove_timeseries`.

        See Also
        --------
        remove_timeseries : Remove a timeseries from the station instance.
        """
        self.remove_timeseries(ts_description)

    def __contains__(self, ts_description: str) -> bool:
        """
        Special function that allows to check whether a certain timeseries
        is in the station.

        Example
        -------
        If ``mystat`` is a :class:`~Station` instance, and we want to see whether
        ``'myts'`` is contained by the station, the following two are equivalent::

            # long version
            'myts' in station.timeseries
            # short version
            'myts' in station
        """
        return ts_description in self.timeseries

    def __iter__(self) -> Iterator[Timeseries]:
        """
        Convenience special function that allows for a shorthand notation to quickly
        iterate over all timeseries.

        Example
        -------
        If ``stat`` is a :class:`~Station` instance,then the following two loops
        are equivalent::

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
    def ts(self) -> dict[str, Timeseries]:
        """
        Abbreviation for :attr:`~timeseries`.
        """
        return self.timeseries

    def get_arch(self) -> dict[str, Any]:
        """
        Build a dictionary describing the architecture of this station,
        to be used when creating a network JSON configuration file.

        Returns
        -------
            Ditionary of attributes.

        See Also
        --------
        disstans.network.Network.to_json
            Export the Network configuration as a JSON file.
        disstans.timeseries.Timeseries.get_arch
            Get the architecture dictionary of a
            :class:`~disstans.timeseries.Timeseries` instance.
        disstans.models.Model.get_arch
            Get the architecture dictionary of a
            :class:`~disstans.models.Model` instance.
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
                stat_arch["models"][ts_description] = self.models[ts_description].get_arch()
        return stat_arch

    def add_timeseries(self,
                       ts_description: str,
                       timeseries: Timeseries,
                       uncertainties_from: Timeseries | None = None,
                       override_src: str | None = None,
                       override_data_unit: str | None = None,
                       override_data_cols: str | None = None,
                       override_var_cols: str | None = None,
                       override_cov_cols: str | None = None,
                       add_models: dict[str, Model] | None = None,
                       warn_existing: bool = True
                       ) -> None:
        """
        Add a timeseries to the station.

        Optionally, can override some Timeseries attributes when adding the object,
        and add models.

        Parameters
        ----------
        ts_description
            Description of the timeseries.
        timeseries
            Timeseries object to add.
        uncertainties_from
            If the variance and covariance data should come from another timeseries,
            specify the source timeseries here.
        override_src
            Override the :attr:`~disstans.timeseries.Timeseries.src`
            attribute of ``timeseries``.
        override_data_unit
            Override the :attr:`~disstans.timeseries.Timeseries.data_unit`
            attribute of ``timeseries``.
        override_data_cols
            Override the :attr:`~disstans.timeseries.Timeseries.data_cols`
            attribute of ``timeseries``.
        override_var_cols
            Override the :attr:`~disstans.timeseries.Timeseries.var_cols`
            attribute of ``timeseries``.
        override_cov_cols
            Override the :attr:`~disstans.timeseries.Timeseries.cov_cols`
            attribute of ``timeseries``.
        add_models
            Dictionary of models to add to the timeseries, where the keys are the
            model description and the values are :class:`~disstans.models.Model` objects.
        warn_existing
            Issue a warning if a station's timeseries is being overwritten.

        See Also
        --------
        __setitem__ : Shorthand notation wrapper when no optional arguments are necessary.

        Example
        -------
        If ``stat`` is a :class:`~Station` instance, ``ts_description`` the ts_description
        of a new timeseries, and ``ts`` a :class:`~disstans.timeseries.Timeseries` instance,
        then the following two are equivalent::

            stat.add_timeseries(ts_description, timeseries)
            stat[ts_description] = timeseries
        """
        if not isinstance(ts_description, str):
            raise TypeError("Cannot add new timeseries: "
                            "'ts_description' is not a string.")
        if not isinstance(timeseries, Timeseries):
            raise TypeError("Cannot add new timeseries: "
                            "'timeseries' is not a Timeseries object.")
        if warn_existing and (ts_description in self.timeseries):
            warn(f"Station {self.name}: Overwriting time series '{ts_description}'.",
                 category=RuntimeWarning, stacklevel=2)
        if uncertainties_from is not None:
            timeseries.add_uncertainties(timeseries=uncertainties_from)
        if override_src is not None:
            timeseries.src = override_src
        if override_data_unit is not None:
            timeseries.data_unit = override_data_unit
        if override_data_cols is not None:
            timeseries.data_cols = override_data_cols
        if override_var_cols is not None:
            timeseries.var_cols = override_var_cols
        if override_cov_cols is not None:
            timeseries.var_cols = override_cov_cols
        self.timeseries[ts_description] = timeseries
        self.fits[ts_description] = FitCollection()
        self.models[ts_description] = ModelCollection()
        if add_models is not None:
            self.add_local_model_dict(ts_description=ts_description, model_kw_args=add_models)

    def remove_timeseries(self, ts_description: str) -> None:
        """
        Remove a timeseries (including associated fits and models) from the station.

        Parameters
        ----------
        ts_description
            Description of the timeseries.

        See Also
        --------
        __delitem__ : Shorthand notation wrapper.

        Example
        -------
        If ``stat`` is a :class:`~Station` instance, ``ts_description`` the ts_description
        of the timeseries to remove, then the following two are equivalent::

            stat.remove_timeseries(ts_description)
            del stat[ts_description]
        """
        if ts_description not in self.timeseries:
            warn(f"Station {self.name}: "
                 f"Cannot find time series '{ts_description}', couldn't delete.",
                 category=RuntimeWarning, stacklevel=2)
        else:
            del self.timeseries[ts_description]
            del self.fits[ts_description]
            del self.models[ts_description]

    def add_local_model(self,
                        ts_description: str,
                        model_description: str,
                        model: Model
                        ) -> None:
        """
        Add a model to a timeseries (overwrites the model if it has already
        been added with the same description).

        Parameters
        ----------
        ts_description
            Timeseries to add the model to.
        model_description
            Model description.
        model
            Model object.
        """
        if not isinstance(ts_description, str):
            raise TypeError("Cannot add new local model: 'ts_description' is not a string.")
        if not isinstance(model_description, str):
            raise TypeError("Cannot add new local model: 'model_description' is not a string.")
        if not isinstance(model, Model):
            raise TypeError("Cannot add new local model: 'model' is not a Model object.")
        assert ts_description in self.timeseries, \
            f"Station {self.name}: " \
            f"Cannot find timeseries '{ts_description}' to add local model '{model_description}'."
        if model_description in self.models[ts_description].model_names:
            warn(f"Station {self.name}, timeseries {ts_description}: "
                 f"Overwriting local model '{model_description}'.",
                 category=RuntimeWarning, stacklevel=2)
        self.models[ts_description][model_description] = model

    def add_local_model_dict(self, ts_description: str, model_dict: dict[str, Model]) -> None:
        """
        Add a dictionary of models to a timeseries (overwrites the models if they have already
        been added with the same description).

        Wraps :meth:`~add_local_model`.

        Parameters
        ----------
        ts_description
            Timeseries to add the model to.
        model_dict
            Dictionary of ``{model_description: model}`` key-value pairs to add.
        """
        assert isinstance(model_dict, dict), \
            f"'model_dict' needs to be a dictionary, got {type(model_dict)}."
        for mdl_desc, mdl in model_dict.items():
            self.add_local_model(ts_description, mdl_desc, mdl)

    def add_local_model_kwargs(self,
                               ts_description: str,
                               model_kw_args: dict[str, dict[str, Any]]
                               ) -> None:
        """
        Add models to a timeseries from a dictionary of keyword arguments needed to create
        them (overwrites the models if they have already been added with the same description).

        Wraps :meth:`~add_local_model`.

        Parameters
        ----------
        ts_description
            Timeseries to add the model to.
        model_kw_args
            Dictionary of structure ``{model_name: {"type": modelclass, "kw_args":
            {**kw_args}}}`` that contains the names, types and necessary keyword arguments
            to create each model object.
        """
        disstans_models.check_model_dict(model_kw_args)
        for mdl_desc, mdl_cfg in model_kw_args.items():
            local_copy = deepcopy(mdl_cfg)
            mdl = getattr(disstans_models, local_copy["type"])(**local_copy["kw_args"])
            self.add_local_model(ts_description=ts_description,
                                 model_description=mdl_desc, model=mdl)

    def remove_local_models(self,
                            ts_description: str,
                            model_descriptions: str | list[str],
                            verbose: bool = False
                            ) -> None:
        """
        Remove models from a timeseries.

        Parameters
        ----------
        ts_description
            Timeseries to remove the model from.
        model_descriptions
            Model description(s).
        verbose
            If ``True``, raise a warning if a timeseries or model could not be found
            in the network.
        """
        # unpack list
        if isinstance(model_descriptions, str):
            model_list = [model_descriptions]
        else:
            model_list = model_descriptions
        # iterate removal
        for mdl_desc in model_list:
            if ts_description in self.timeseries:
                if mdl_desc in self.models[ts_description].model_names:
                    del self.models[ts_description][mdl_desc]
                    if mdl_desc in self.fits[ts_description]:
                        del self.fits[ts_description][mdl_desc]
                elif verbose:
                    warn(f"Station {self.name}, timeseries {ts_description}: "
                         f"Cannot find local model '{mdl_desc}', couldn't delete.",
                         category=RuntimeWarning, stacklevel=2)
            elif verbose:
                warn(f"Station {self.name}: Cannot find timeseries '{ts_description}', "
                     f"couldn't delete local model '{mdl_desc}'.",
                     category=RuntimeWarning, stacklevel=2)

    def add_fit(self,
                ts_description: str,
                fit: dict[str, np.ndarray | None],
                model_description: str | None = None,
                return_ts: bool = False
                ) -> Timeseries | None:
        """
        Add a fit dictionary to a timeseries' model (overwrites the fit if it has
        already been added for the model).

        Parameters
        ----------
        ts_description
            Timeseries to add the fit to.
        fit
            Dictionary with the keys ``'time'``, ``'fit'``, ``'var'`` and ``'cov'``
            (the latter two can be set to ``None``).
        model_description
            Model description the fit applies to.
            If ``None``, the fit is the sum of all individual model fits.
        return_ts
            If ``True``, return the created timeseries.

        Returns
        -------
            (If ``return_ts=True``.) The fit as a Timeseries object.

        See Also
        --------
        disstans.timeseries.Timeseries.from_fit
            Convert a fit dictionary to a :class:`~disstans.timeseries.Timeseries` object.
        """
        if not isinstance(ts_description, str):
            raise TypeError("Cannot add new fit: 'ts_description' is not a string.")
        if not (isinstance(model_description, str) or (model_description is None)):
            raise TypeError("Cannot add new fit: 'model_description' is not a string.")
        assert ts_description in self.timeseries, \
            f"Station {self.name}: Cannot find timeseries '{ts_description}' " \
            f"to add fit for model '{model_description}'."
        if model_description is None:
            data_cols = [ts_description + "_Model_" + dcol
                         for dcol in self.timeseries[ts_description].data_cols]
        else:
            data_cols = [ts_description + "_" + str(model_description) + "_" + dcol
                         for dcol in self.timeseries[ts_description].data_cols]
            assert model_description in self.models[ts_description], \
                f"Station {self.name}, timeseries {ts_description}: " \
                f"Cannot find local model '{model_description}', couldn't add fit."
        fit_ts = Timeseries.from_fit(self.timeseries[ts_description].data_unit, data_cols, fit)
        if model_description is None:
            self.fits[ts_description].allfits = fit_ts
        else:
            self.fits[ts_description][model_description] = fit_ts
        if return_ts:
            return fit_ts

    def remove_fit(self, ts_description: str, model_description: str) -> None:
        """
        Remove a fit from a timeseries' model.

        Parameters
        ----------
        ts_description
            Timeseries to remove the fit from.
        model_description
            Model description to remove the fit from.
        """
        if ts_description not in self.timeseries:
            warn(f"Station {self.name}: Cannot find timeseries '{ts_description}', "
                 f"couldn't delete fit for model '{model_description}'.",
                 category=RuntimeWarning, stacklevel=2)
        elif model_description not in self.models[ts_description]:
            warn(f"Station {self.name}, timeseries {ts_description}: "
                 f"Cannot find local model '{model_description}', couldn't delete fit.",
                 category=RuntimeWarning, stacklevel=2)
        elif model_description not in self.fits[ts_description]:
            warn(f"Station {self.name}, timeseries {ts_description}: "
                 f"Cannot find fit for local model '{model_description}', couldn't delete.",
                 category=RuntimeWarning, stacklevel=2)
        else:
            del self.fits[ts_description][model_description]

    def sum_fits(self,
                 ts_description: str,
                 fit_list: list[str] | None = None
                 ) -> tuple[np.ndarray, np.ndarray | None]:
        r"""
        Method to quickly sum fits of a timeseries.

        Parameters
        ----------
        ts_description
            Timeseries whose fits to sum.
        fit_list
            List of strings containing the model names of the subset of the fitted models
            to be summed. ``None`` defaults to all fitted models.

        Returns
        -------
        fit_sum
            2D array of shape :math:`(\text{n_observations},\text{n_components})`
        fit_sum_var
            2D array of shape :math:`(\text{n_observations},\text{n_components})`.
            Returns ``None`` if no standard deviations are available.
        """
        # shorthand for timeseries
        ts = self[ts_description]
        assert fit_list is None or isinstance(fit_list, list), \
            f"'fit_list' needs to be None or a list, got {type(fit_list)}."
        # get model subset
        fits_to_sum = {model_description: fit
                       for model_description, fit in self.fits[ts_description].items()
                       if ((fit_list is None) or (model_description in fit_list))}
        assert fits_to_sum, \
            f"Station {self.name}, timeseries {ts_description}: Can't find fits for models."
        # sum models and uncertainties
        fit_sum = np.zeros((ts.num_observations, ts.num_components))
        if (ts.var_cols is not None) and all([fit.var_cols is not None
                                              for fit in fits_to_sum.values()]):
            fit_sum_var = np.zeros_like(fit_sum)
        else:
            fit_sum_var = None
        for model_description, fit in fits_to_sum.items():
            fit_sum += fit.data.values
            if fit_sum_var is not None:
                fit_sum_var += fit.vars.values
        return fit_sum, fit_sum_var

    def analyze_residuals(self,
                          ts_description: str,
                          verbose: bool = False,
                          t_start: str | pd.Timestamp | None = None,
                          t_end: str | pd.Timestamp | None = None,
                          mean: bool = False,
                          std: bool = False,
                          rms: bool = False,
                          n_observations: bool = False,
                          std_outlier: int | float = 0,
                          max_rolling_dev: int = 0
                          ) -> dict[str, np.ndarray]:
        """
        Analyze, print and return the residuals of a station's timeseries according
        to certain metrics defined in the arguments.

        Parameters
        ----------
        ts_description
            Timeseries to analyze. Method assumes it already is a residual.
        verbose
            If True, additionally print the results.
        t_start
            If set, specify a lower bound for the analysis. ``t_start`` should be a
            Timestamp-convertible string of the start time, or a datetime-like object.
            ``None`` defaults to the first timestamp present in the timeseries.
        t_end
            If set, specify an upper bound for the analysis. ``t_end`` should be a
            Timestamp-convertible string of the end time, or a datetime-like object.
            ``None`` defaults to the last timestamp present in the timeseries.
        mean
            If ``True``, calculate the mean of the timeseries.
            Adds the key ``'Mean'`` to the output dictionary.
        std
            If ``True``, calculate the standard deviation of the timeseries.
            Adds the key ``'Standard Deviation'`` to the output dictionary.
        rms
            If ``True``, calculate the root-mean-squared over all components combined,
            or if ``rms`` is a list, only using those components.
            Adds the key ``'RMS'`` to the output dictionary.
        n_observations
            If ``True``, count the number of observations (excluding NaNs) and NaNs.
            Adds the keys ``'Observations'`` and ``'Gaps'`` to the output dictionary.
        std_outlier
            If ``std_outlier > 0``, count the number of non-NaN outliers, defined
            by the number of standard deviations they are away from the mean.
            Adds the key ``'Outliers'`` to the output dictionary.
        max_rolling_dev
            If ``max_rolling_dev > 0``, calculate the (sign-aware) maximum of the
            rolling deviation from the mean with window size ``max_rolling_dev`` for
            each component. If this is significantly larger than the standard deviation,
            this is an indicator of model inadequacies, as not all the data trends can
            be captured.

        Returns
        -------
            Dictionary that includes the results as defined by the arguments.
            Empty by default.
        """
        # do checks
        assert isinstance(ts_description, str), \
            f"Station {self.name}: " \
            f"'ts_description' needs to be a string, got {type(ts_description)}."
        assert ts_description in self.timeseries, \
            f"Station {self.name}: Can't find '{ts_description}' to analyze."
        # restrict to time range
        ts = self[ts_description]
        t_start = pd.Timestamp(t_start) if t_start is not None else ts.time[0]
        t_end = pd.Timestamp(t_end) if t_end is not None else ts.time[-1]
        inside = np.all((ts.time >= t_start, ts.time <= t_end), axis=0)
        ts = self[ts_description].data.iloc[inside, :]
        # calculate the different metrics
        results = {}
        if mean:
            mean = ts.mean(axis=0, skipna=True, numeric_only=True).values
            results["Mean"] = mean
        if std:
            std = ts.std(axis=0, skipna=True, numeric_only=True).values
            results["Standard Deviation"] = std
        if rms:
            if not isinstance(rms, list):
                rms = list(range(self[ts_description].num_components))
            rms_result = ((ts.iloc[:, rms] ** 2)
                          .mean(axis=0, skipna=True, numeric_only=True) ** 0.5)
        if n_observations:
            n_obs = ts.count(axis=0, numeric_only=True).values
            results["Observations"] = n_obs
            results["Gaps"] = ts.shape[0] - n_obs
        if std_outlier > 0:
            temp = ts.values.copy()
            temp[np.isnan(temp)] = 0
            temp -= np.mean(temp, axis=0, keepdims=True)
            temp = temp > np.std(temp, axis=0, keepdims=True) * std_outlier
            results["Outliers"] = np.sum(temp, axis=0, dtype=int)
        if max_rolling_dev > 0:
            roll_mean = (ts - ts.mean()).rolling(max_rolling_dev,
                                                 min_periods=max_rolling_dev // 2
                                                 ).mean().values
            if np.isfinite(roll_mean.ravel()).sum() == 0:
                results["Maximum Rolling Deviation"] = np.nan
            else:
                roll_mean_max_ix = np.nanargmax(np.abs(roll_mean), axis=0)
                max_rolling_dev = np.take_along_axis(roll_mean,
                                                     np.expand_dims(roll_mean_max_ix, axis=0),
                                                     axis=0).squeeze(axis=0)
                results["Maximum Rolling Deviation"] = max_rolling_dev
        # print if any statistic was recorded
        if verbose and results:
            print_df = pd.DataFrame(data=results, index=self[ts_description].data_cols)
            print(print_df.rename_axis(f"{self.name}: {ts_description}", axis=1))
            if rms:
                print(f"RMS of components {rms}: "
                      f"{rms_result} {self[ts_description].data_unit}")
        # attach RMS here since it couldn't be added to the DataFrame before
        if rms:
            results["RMS"] = rms_result
        return results

    def get_trend(self,
                  ts_description: str,
                  fit_list: list[str] | None = None,
                  components: list[int] | None = None,
                  total: bool = False,
                  t_start: str | pd.Timestamp | None = None,
                  t_end: str | pd.Timestamp | None = None,
                  use_formal_variance: bool | None = None,
                  include_sigma: bool = False,
                  time_unit: str = "D",
                  ignore_missing: bool = False
                  ) -> tuple[np.ndarray | None, np.ndarray | None]:
        r"""
        Calculates a linear trend through the desired model fits (or timeseries data)
        and over some time span.

        Parameters
        ----------
        ts_description
            Timeseries whose fits to use.
        fit_list
            List of strings containing the model names of the subset of the fitted models
            to be used. ``None`` defaults to all fitted models.
            If ``ts_description`` the trend of the timeseries itself is to be calculated,
            pass an empty list (``[]``).
        components
            List of the numerical indices of which components of the timeseries to use.
            ``None`` defaults to all components.
        total
            By default, the function will return the trend per ``time_unit``.
            If ``True``, the function will instead give the total difference over the
            entire timespan.
        t_start
            Timestamp-convertible string of the start time.
            ``None`` defaults to the first timestamp present in the timeseries.
        t_end
            Timestamp-convertible string of the end time.
            ``None`` defaults to the last timestamp present in the timeseries.
        use_formal_variance
            By default, the trend fitting will use variance information if present. This usually
            makes sense for actual timeseries, but not for fitted models where the formal variance
            can be very unphysical. Setting this parameter overrides the default behavior.
        include_sigma
            If ``True``, also calculate the formal standard deviation on the trend estimate.
        time_unit
            Time unit for output (only required if ``total=False``).
        ignore_missing
            By default, this method will throw an error if the timeseries or
            the models are not present in  the station. Set to ``True`` if the method
            should return ``(None, None)`` instead (as done when the requested timespan
            is outside the observed one).

        Returns
        -------
        trend
            1D array of size ``len(components)`` containing the trends.
        trend_sigma
            1D array of size ``len(components)`` containing the standard deviation of
            the trend estimate. Returns ``None`` if no standard deviations are available.
        """
        assert isinstance(ts_description, str), \
            f"Station {self.name}: " \
            f"'ts_description' needs to be a string, got {type(ts_description)}."
        try:
            assert ts_description in self.timeseries, \
                f"Station {self.name}: Can't find '{ts_description}' to calculate a trend for."
        except AssertionError:
            if ignore_missing:  # fail softly
                return None, None
            else:
                raise
        if fit_list is not None:
            if isinstance(fit_list, str):
                fit_list = [fit_list]
            else:
                assert isinstance(fit_list, list), "'fit_list' needs to be None, a string, " \
                    f"or a list, got {type(fit_list)}."
            if len(fit_list) == 0:
                # use data instead of fits
                use_fits = False
            else:
                # use a model subset list
                try:
                    assert (self.models[ts_description] and
                            any([f in self.models[ts_description] for f in fit_list])), \
                        f"Station {self.name}, timeseries {ts_description}: " \
                        "Can't find any models."
                except AssertionError:
                    if ignore_missing:  # fail softly
                        return None, None
                    else:
                        raise
                else:
                    use_fits = True
        else:
            # default to all fitted models
            use_fits = True
        # check components
        ts = self[ts_description]
        if components is None:
            components = list(range(ts.num_components))
        else:
            assert max(components) < ts.num_components, \
                f"Station {self.name}, timeseries {ts_description}: " \
                "Requesting non-existent components."
        n_comps = len(components)
        # get time span
        t_start = pd.Timestamp(t_start) if t_start is not None else ts.time[0]
        t_end = pd.Timestamp(t_end) if t_end is not None else ts.time[-1]
        inside = np.all((ts.time >= t_start, ts.time <= t_end), axis=0)
        if inside.any():
            t_span = ts.time[inside]
        else:
            return None, None
        # get relevant timeseries
        if use_fits:
            fit_sum, fit_sum_var = self.sum_fits(ts_description, fit_list)
            if (use_formal_variance is not None) and (not use_formal_variance):
                fit_sum_var = None
        else:
            fit_sum = ts.data.values
            if use_formal_variance is None or use_formal_variance:
                fit_sum_var = None if ts.var_cols is None else ts.vars.values
            else:
                fit_sum_var = None
        # initialize fitting
        G = np.ones((t_span.size, 2))
        G[:, 1] = tvec_to_numpycol(t_span, time_unit=time_unit)
        if total:
            G[:, 1] /= G[-1, 1]
        trend = np.zeros(n_comps)
        if include_sigma:
            trend_sigma = np.zeros(n_comps)
        # fit components
        for icomp in components:
            if use_fits:
                validsub = np.ones(inside.sum(), dtype=bool)
                Gsub = G
            else:
                if fit_sum_var is not None:
                    validsub = np.all((np.isfinite(fit_sum[inside, icomp]),
                                       np.isfinite(fit_sum_var[inside, icomp])), axis=0)
                else:
                    validsub = np.isfinite(fit_sum[inside, icomp])
                Gsub = G[validsub, :]
                if Gsub.shape[0] < 2:
                    return None, None
            if fit_sum_var is not None:
                GtW = Gsub.T @ sparse.diags(1 / fit_sum_var[inside, icomp][validsub])
                GtWG = GtW @ Gsub
                GtWd = GtW @ fit_sum[inside, icomp][validsub]
            else:
                GtWG = Gsub.T @ Gsub
                GtWd = Gsub.T @ fit_sum[inside, icomp][validsub]
            trend[icomp] = np.linalg.lstsq(GtWG, GtWd.squeeze(), rcond=None)[0][1]
            if include_sigma:
                if Gsub.shape[0] == 2:
                    trend_sigma[icomp] = 0
                else:
                    trend_sigma[icomp] = np.sqrt(sp.linalg.pinvh(GtWG)[1, 1])
        return (trend, trend_sigma) if include_sigma else (trend, None)

    def get_trend_change(self,
                         ts_description: str,
                         check_times: pd.Series | pd.DatetimeIndex,
                         window_size: float,
                         window_unit: str = "D",
                         min_change: float = 0.0,
                         return_signs: bool = True
                         ) -> list[list[int | None]]:
        """
        Calculates the (linear) trends of a timeseries before and after given timestamps,
        and return for each one whether the trend increased, decreased, or stayed the same.

        Parameters
        ----------
        ts
            Timeseries to analyze.
        check_times
            Timestamps to check the trend before and after.
        window_size
            Width of the trend calculation window.
        window_unit
            Time unit of ``window_size``.
        min_change
            The minimum magnitude of trend change to be considered correct.
            If greater than ``0``, the returned trend change elements can include
            ``None`` if the observed trend change magnitude is smaller than
            ``min_change``.
        return_signs
            If ``True``, only the sign of the change is returned, if ``False``,
            the numerical value of the trend difference is returned.

        Returns
        -------
            A nested array containing for each checked timestamp (first level) and each
            Timeseries data component (second level) whether the trend change is positive
            (``+1``), negative (``-1``), or undetermined (``None``, either due to missing
            data or a ``min_change`` larger than ``0``).
        """
        # initial check
        assert min_change >= 0, f"'min_change' needs to be non-negative, got {min_change}."
        assert ts_description in self, \
            f"Timeseries '{ts_description}' not present in Station {self.name}."
        # initialize output
        ts = self[ts_description]
        trend_change = []
        window = pd.Timedelta(window_size, unit=window_unit)
        # loop over timestamps
        for t in check_times:
            # make sure there is data available
            if ((ts.time <= t).sum() == 0) or ((ts.time >= t).sum() == 0):
                trend_change.append([None] * ts.num_components)
                continue
            # get the available timestamps right before and after the tested times
            t_rounded = pd.Timestamp(t.to_pydatetime().date())
            t_plus = ts.time[ts.time >= t_rounded][0]
            t_minus = ts.time[ts.time < t_rounded][-1]
            # calculate the two trends
            trend_before = self.get_trend(ts_description=ts_description,
                                          fit_list=[],
                                          components=None,
                                          total=False,
                                          t_start=t - window,
                                          t_end=t_minus,
                                          use_formal_variance=True,
                                          include_sigma=False,
                                          time_unit=window_unit,
                                          ignore_missing=False)[0]
            trend_after = self.get_trend(ts_description=ts_description,
                                         fit_list=[],
                                         components=None,
                                         total=False,
                                         t_start=t_plus,
                                         t_end=t + window,
                                         use_formal_variance=True,
                                         include_sigma=False,
                                         time_unit=window_unit,
                                         ignore_missing=False)[0]
            # if at least one trend couldn't be calculated, continue early
            if (trend_before is None) or (trend_after is None):
                trend_change.append([None] * ts.num_components)
                continue
            # save trend change
            trend_diff = trend_after - trend_before
            if return_signs:
                trend_change.append([np.sign(td).astype(int) if np.abs(td) > min_change
                                     else None for td in trend_diff])
            else:
                trend_change.append([td if np.abs(td) > min_change
                                     else None for td in trend_diff])
        # done
        return trend_change
