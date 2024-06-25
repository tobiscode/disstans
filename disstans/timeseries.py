"""
This module contains the :class:`~Timeseries` base class and other
formats included by default in DISSTANS.
"""

from __future__ import annotations
import zipfile
import gzip
import numpy as np
import pandas as pd
from io import BytesIO
from warnings import warn
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Literal

from .tools import get_cov_dims, make_cov_index_map, get_cov_indices, R_ecef2enu


class Timeseries():
    """
    Object that expands the functionality of a :class:`~pandas.DataFrame` object
    for better integration into DISSTANS. Apart from the data itself, it contains
    information about the source and units of the data. It also performs input
    checks and uses property setters/getters to ensure consistency.

    Also enables the ability to perform math on timeseries directly.

    Parameters
    ----------
    dataframe
        The timeseries' data as a DataFrame. The index should be time, whereas
        data columns can be both data and their uncertainties.
    src
        Source description.
    data_unit
        Data unit.
    data_cols
        List of strings with the names of the columns of ``dataframe`` that
        contain the data.
        The length corresponds to the number of components :attr:`~num_components`.
    var_cols
        List of strings with the names of the columns of ``dataframe`` that contain the
        data's variance.
        Must have the same length as ``data_cols``.
        ``None`` defaults to no data variance columns.
    cov_cols
        List of strings with the names of the columns of ``dataframe`` that contain
        the data's covariance.
        Must have length ``(num_components * (num_components - 1)) / 2``, where
        the order of the elements is determined by their row-by-row, sequential
        position in the covariance matrix (see Notes).
        ``None`` defaults to no covariance columns.
    remove_initial_offset
        If ``True``, the data timeseries will be shifted such that it starts at zero.
        The offset will be recorded in :class:`~Timeseries.offset` if it needs to be
        recovered.

    Notes
    -----

    In terms of mapping the covariance matrix of observations into the format
    for the :class:`~Timeseries` class, consider this example for observations with
    three components:

    +-------------------+-------------------+-------------------+
    | ``var_cols[0]``   | ``cov_cols[0]``   | ``cov_cols[1]``   |
    +-------------------+-------------------+-------------------+
    | (symmetric)       | ``var_cols[1]``   | ``cov_cols[2]``   |
    +-------------------+-------------------+-------------------+
    | (symmetric)       | (symmetric)       | ``var_cols[2]``   |
    +-------------------+-------------------+-------------------+
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 src: str,
                 data_unit: str,
                 data_cols: list[str],
                 var_cols: list[str] | None = None,
                 cov_cols: list[str] | None = None,
                 remove_initial_offset: bool = False
                 ) -> None:
        # input type checks
        assert isinstance(dataframe, pd.DataFrame)
        assert dataframe.index.inferred_type == "datetime64"
        assert all([pd.api.types.is_numeric_dtype(dt) for dt in dataframe.dtypes])
        assert isinstance(src, str)
        assert isinstance(data_unit, str)
        assert isinstance(data_cols, list) and all([isinstance(dcol, str) for dcol in data_cols])
        assert all([dcol in dataframe.columns for dcol in data_cols])
        # duplicates check
        index_duplicated = dataframe.index.duplicated()
        if index_duplicated.sum() > 0:
            raise RuntimeError("Input DataFrame cannot have duplicate indices, but found the "
                               "following duplicate timestamps:\n" +
                               str(dataframe.index[index_duplicated].to_list()))
        # save to private instance variables
        self._df = dataframe
        self._src = src
        self._data_unit = data_unit
        self._data_cols = data_cols
        # additional checks for var_cols
        if var_cols is not None:
            assert isinstance(var_cols, list) and \
                all([isinstance(vcol, str) for vcol in var_cols]), \
                f"If not None, 'var_cols' must be a list of strings, got {var_cols}."
            assert all([vcol in dataframe.columns for vcol in var_cols]), \
                "All entries in 'var_cols' must be present in the dataframe."
            assert len(self._data_cols) == len(var_cols), \
                "If passing variance columns, " \
                "the list needs to have the same length as the data columns one."
        self._var_cols = var_cols
        # additional checks for cov_cols
        if cov_cols is not None:
            assert self.num_components > 1, \
                "In order to set covariances, the timeseries needs at least 2 components."
            assert cov_cols is not None, \
                "If setting covariances, the standard deviations must also be set."
            assert isinstance(cov_cols, list) and \
                all([isinstance(ccol, str) for ccol in cov_cols]), \
                f"If not None, 'cov_cols' must be a list of strings, got {cov_cols}."
            assert all([ccol in dataframe.columns for ccol in cov_cols]), \
                "All entries in 'cov_cols' must be present in the dataframe."
            cov_dims = get_cov_dims(self.num_components)
            assert len(cov_cols) == cov_dims, \
                "If passing covariance columns, the list needs to have the appropriate " \
                "length given the data columns length. " \
                f"Expected {cov_dims}, got {len(cov_cols)}."
            # everything should be alright, make index maps
            index_map, var_cov_map = make_cov_index_map(self.num_components)
            self.index_map = index_map
            """
            Matrix that contains the rolling indices of each matrix element used by
            :func:`~disstans.tools.get_cov_indices`.
            """
            self.var_cov_map = var_cov_map
            """
            Contains the column indices needed to create the full variance-covariance
            matrix for a single time.
            """
        self._cov_cols = cov_cols
        # shift data to zero if desired
        if remove_initial_offset:
            offset = self._df.loc[self._df.index[0], self.data_cols]
            self._df.loc[:, self.data_cols] -= offset
        else:
            offset = None
        self.offset = offset
        """ Offset applied to the timeseries data such that it starts at zero. """

    def __str__(self) -> str:
        """
        Special function that returns a readable summary of the timeseries.
        Accessed, for example, by Python's ``print()`` built-in function.

        Returns
        -------
            Timeseries summary.
        """
        info = f"Timeseries\n - Source: {self.src}\n - Units: {self.data_unit}" \
               f"\n - Shape: {self.shape}\n - Offset Removed: {self.offset is not None}" \
               f"\n - Data: {[key for key in self.data_cols]}"
        if self.var_cols is not None:
            info += f"\n - Variances: {[key for key in self.var_cols]}"
        if self.cov_cols is not None:
            info += f"\n - Covariances: {[key for key in self.cov_cols]}"
        return info

    def __getitem__(self, columns: str | list[str]) -> pd.Series | pd.DataFrame:
        """
        Convenience special function that provides a shorthand notation
        to access the timeseries' columns.

        Parameters
        ----------
        columns
            String or list of strings of the columns to return.

        Returns
        -------
            Returns the requested data as a Series (if a single column) or
            DataFrame (if multiple columns).

        Example
        -------
        If ``ts`` is a :class:`~Timeseries` instance and ``columns`` a list
        of column names, the following two are equivalent::

            ts.df[columns]
            ts[ts_description]
        """
        if not isinstance(columns, str) and \
           (not isinstance(columns, list) or
           not all([isinstance(col, str) for col in columns])):
            raise KeyError("Error when accessing data in timeseries: 'column' must be a "
                           f"string or list of strings, given was {columns}.")
        return self.df[columns]

    @property
    def src(self) -> str:
        """ Source information. """
        return self._src

    @src.setter
    def src(self, new_src: str) -> None:
        if not isinstance(new_src, str):
            raise TypeError(f"New 'src' attribute has to be a string, got {type(new_src)}.")
        self._src = new_src

    @property
    def data_unit(self) -> str:
        """ Data unit. """
        return self._data_unit

    @data_unit.setter
    def data_unit(self, new_data_unit: str) -> None:
        if not isinstance(new_data_unit, str):
            raise TypeError("New 'data_unit' attribute has to be a string, "
                            f"got {type(new_data_unit)}.")
        self._data_unit = new_data_unit

    @property
    def data_cols(self) -> list[str]:
        """ List of the column names in :attr:`~df` that contain data. """
        return self._data_cols

    @data_cols.setter
    def data_cols(self, new_data_cols: list[str]) -> None:
        assert isinstance(new_data_cols, list) and \
            all([isinstance(dcol, str) for dcol in new_data_cols]), \
            "New 'data_cols' attribute must be a list of strings of the same length as the " \
            f"current 'data_cols' ({len(self._data_cols)}), got {new_data_cols}."
        self._df.rename(columns={old_col: new_col for old_col, new_col
                                 in zip(self._data_cols, new_data_cols)},
                        errors='raise', inplace=True)
        self._data_cols = new_data_cols

    @property
    def var_cols(self) -> list[str] | None:
        """ List of the column names in :attr:`~df` that contain data variance. """
        return self._var_cols

    @var_cols.setter
    def var_cols(self, new_var_cols: list[str | None]) -> None:
        assert self._var_cols is not None, \
            "No variance columns found that can be renamed."
        assert isinstance(new_var_cols, list) and \
            all([(vcol is None) or isinstance(vcol, str) for vcol in new_var_cols]), \
            "New 'var_cols' attribute must be a list of strings or Nones of the same length " \
            f"as the current 'var_cols' ({len(self._var_cols)}), got {new_var_cols}."
        for ivcol, vcol in enumerate(new_var_cols):
            if vcol is None:
                new_var_cols[ivcol] = self._var_cols[ivcol]
        self._df.rename(columns={old_col: new_col for old_col, new_col
                                 in zip(self._var_cols, new_var_cols)},
                        errors='raise')
        self._var_cols = new_var_cols

    @property
    def cov_cols(self) -> list[str] | None:
        """ List of the column names in :attr:`~df` that contain data covariances. """
        return self._cov_cols

    @cov_cols.setter
    def cov_cols(self, new_cov_cols: list[str | None]) -> None:
        assert self._cov_cols is not None, \
            "No covariance columns found that can be renamed."
        assert isinstance(new_cov_cols, list) and \
            all([(ccol is None) or isinstance(ccol, str) for ccol in new_cov_cols]), \
            "New 'cov_cols' attribute must be a list of strings or Nones of the same length " \
            f"as the current 'cov_cols' ({len(self._cov_cols)}), got {new_cov_cols}."
        for iccol, ccol in enumerate(new_cov_cols):
            if ccol is None:
                new_cov_cols[iccol] = self._cov_cols[iccol]
        self._df.rename(columns={old_col: new_col for old_col, new_col
                                 in zip(self._cov_cols, new_cov_cols)},
                        errors='raise')
        self._cov_cols = new_cov_cols

    @property
    def num_components(self) -> int:
        """ Number of data columns. """
        return len(self._data_cols)

    @property
    def df(self) -> pd.DataFrame:
        """ The entire timeseries' DataFrame. """
        return self._df

    @property
    def num_observations(self) -> int:
        """ Number of observations (rows in :attr:`~df`). """
        return self._df.shape[0]

    @property
    def data(self) -> pd.DataFrame:
        """ View of only the data columns in :attr:`~df`. """
        return self._df.loc[:, self._data_cols]

    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._df[self._data_cols] = new_data

    @property
    def sigmas(self) -> pd.DataFrame:
        """ View of only the data standard deviation columns in :attr:`~df`. """
        return self.vars ** (1 / 2)

    @sigmas.setter
    def sigmas(self, new_sigma: np.ndarray) -> None:
        self.vars = new_sigma ** 2

    @property
    def vars(self) -> pd.DataFrame:
        """ Returns the variances from :attr:`~df`. """
        if self._var_cols is None:
            raise ValueError("No variance columns present to return.")
        return self._df.loc[:, self._var_cols]

    @vars.setter
    def vars(self, new_var: np.ndarray) -> None:
        if self._var_cols is None:
            raise ValueError("No variance columns present to set.")
        self._df[self._var_cols] = new_var

    @property
    def covs(self) -> pd.DataFrame:
        """ Returns the covariances from :attr:`~df`. """
        if self._cov_cols is None:
            raise ValueError("No covariance columns present to return.")
        return self._df.loc[:, self._cov_cols]

    @covs.setter
    def covs(self, new_cov: np.ndarray) -> None:
        if self._cov_cols is None:
            raise ValueError("No covariance columns present to set.")
        self._df[self._cov_cols] = new_cov

    @property
    def var_cov(self) -> pd.DataFrame:
        """
        Returns the variance as well as covariance columns from :attr:`~df`, to be indexed
        by :attr:`~var_cov_map` to yield the full variance-covariance matrix.
        """
        if self._var_cols is None:
            raise ValueError("No variance columns present to return.")
        if self._cov_cols is None:
            raise ValueError("No covariance columns present to return.")
        return self._df.loc[:, self._var_cols + self._cov_cols]

    @var_cov.setter
    def var_cov(self, new_var_cov: np.ndarray) -> None:
        cov_dims = get_cov_dims(self.num_components)
        assert new_var_cov.shape[1] == self.num_components + cov_dims, \
            "Setting 'var_cov' requires a column for each variance (first half of array, " \
            f"{self.num_components}) and covariance (second half, {cov_dims})."
        self.vars = new_var_cov[:, :self.num_components]
        self.covs = new_var_cov[:, self.num_components:]

    @property
    def time(self) -> pd.Index:
        """ Timestamps of the timeseries (index of :attr:`~df`). """
        return self._df.index

    @property
    def shape(self) -> tuple[int, int]:
        r"""
        Returns the shape tuple (similar to NumPy) of the timeseries, which is of shape
        :math:`(\text{n_observations},\text{n_components})`.
        """
        return (self.num_observations, self.num_components)

    @property
    def length(self) -> np.timedelta64:
        """
        Returns the length of the timeseries.
        """
        return self.time.max() - self.time.min()

    @property
    def reliability(self) -> float:
        """
        Returns the reliability (between 0 and 1) defined as the number of available
        observations divided by the the number of expected observations. The expected
        observations are calculated by taking the median timespan between observations,
        and then dividing the total time span by that timespan.

        (Essentially, this assumes that there are not any "close-by" observation, e.g.
        two observation for the same day but a different hour in a dataset of otherwise
        daily observations.)
        """
        # get median time difference
        med_tdiff = np.median(np.diff(self.time.values))
        # expected observations
        exp_obs = int(np.round(self.length / med_tdiff + 1))
        # return ratio
        return self.num_observations / exp_obs

    def cov_at(self, t: pd.Timestamp | str | int) -> np.ndarray:
        """
        Returns the covariance matrix of the timeseries at a given time or index.

        Parameters
        ----------
        t
            A timestamp or timestamp-convertable string to return the covariance
            matrix for. Alternatively, an integer index.

        Returns
        -------
            The full covariance matrix at time ``t``.
        """
        # input checks
        assert self._var_cols is not None, \
            "No variances available."
        if isinstance(t, str):
            t = pd.Timestamp(t)
        elif isinstance(t, int):
            t = self.time[t]
        assert self.time.min() <= t <= self.time.max(), \
            f"Time {t} not inside available timespan."
        assert np.any(np.isfinite(self.vars[t].values)), \
            f"No variances set at time {t}."
        # only variance is set
        if self._cov_cols is None:
            cov_mat = np.diag(self.vars[t].values)
        # full covariance matrix is available
        else:
            cov_mat = np.reshape(self.var_cov[t].values[0, self.var_cov_map],
                                 (self.num_components, self.num_components))
        return cov_mat

    def check_covariances(self) -> np.ndarray:
        r"""
        Check the covariance for each observation and determine whether it's
        positive-definite or not.

        Returns
        -------
            An boolean array of size :math:`(\text{n_observations}, )` containing
            the result of the positive-definite check (``True`` if passed).
        """
        # quick check
        if self._cov_cols is None:
            raise ValueError("No covariances present to check.")
        # prepare input and output
        varcov = self.var_cov.values
        is_posdef = np.full(self.num_observations, False, dtype=bool)
        # loop over timestamps
        for i in range(self.num_observations):
            # test using Cholesky decomposition
            try:
                # skip all the checks that Timeseries.cov_at does
                _ = np.linalg.cholesky(np.reshape(varcov[i, self.var_cov_map],
                                                  (self.num_components, self.num_components)))
                is_posdef[i] = True
            # catch if not positive-definite
            except np.linalg.LinAlgError as e:
                if "Matrix is not positive definite" not in str(e):
                    raise e
        # done
        return is_posdef

    def cut(self,
            t_min: pd.Timestamp | str | None = None,
            t_max: pd.Timestamp | str | None = None,
            i_min: int | None = None,
            i_max: int | None = None,
            keep_inside: bool = True
            ) -> None:
        """
        Cut the timeseries to contain only data between certain times or indices.
        If both a minimum (maximum) timestamp or index is provided, the later (earlier,
        respectively) one is used (i.e., the more restrictive one).
        Also provides the reverse operation, i.e. only removing data between dates.

        This operation changes the timeseries in-place; if it should be done on a
        new timeseries, use :meth:`~copy` first.

        Parameters
        ----------
        t_min
            A timestamp or timestamp-convertable string of the earliest observation
            to keep.
        t_max
            A timestamp or timestamp-convertable string of the latest observation
            to keep.
        i_min
            The index of the earliest observation to keep.
        i_max
            The index of the latest observation to keep.
        keep_inside
            If ``True``, keeps data inside of the specified date range.
            If ``False``, keeps only data outside the specified date range.
        """
        assert any([t_min, t_max, i_min, i_max]), "No cutting operation defined."
        # convert to timestamps (if not already)
        if i_min:
            t_from_i_min = self.time[i_min]
        if i_max:
            t_from_i_max = self.time[i_max]
        if t_min:
            t_min = pd.Timestamp(t_min)
        if t_max:
            t_max = pd.Timestamp(t_max)
        # get more restrictive cutoffs
        if i_min and t_min:
            cut_min = max([t_from_i_min, t_min])
        elif i_min or t_min:
            cut_min = t_min if t_min else t_from_i_min
        else:
            cut_min = self.time[0]
        if i_max and t_max:
            cut_max = min([t_from_i_max, t_max])
        elif i_max or t_max:
            cut_max = t_max if t_max else t_from_i_max
        else:
            cut_max = self.time[-1]
        # cut the dataframe directly
        if keep_inside:
            self._df = self._df[(self.time >= cut_min) & (self.time <= cut_max)]
        else:
            self._df = self._df[(self.time < cut_min) | (self.time > cut_max)]

    def add_uncertainties(self,
                          timeseries: Timeseries | None = None,
                          var_data: np.ndarray | None = None,
                          var_cols: list[str] | None = None,
                          cov_data: np.ndarray | None = None,
                          cov_cols: list[str] | None = None
                          ) -> None:
        """
        Add variance and covariance data and column names to the timeseries.

        Parameters
        ----------
        timeseries
            Another timeseries object that contains uncertainty information.
            If set, the function will ignore the rest of the arguments.
        var_data
            New data variance.
        var_cols
            List of variance column names.
        cov_data
            New data covariance.
            Setting this but not ``var_data`` requires there to already be data variance.
        cov_cols
            List of covariance column names.

        Notes
        -----
        If ``ts`` is a :class:`~disstans.timeseries.Timeseries` instance, just using::

            ts.vars = new_variance
            ts.covs = new_covariance

        will only work when the respective columns already exist in the dataframe.
        (This is the same behavior for renaming variance columns that do not exist.)
        If they do not exist, the calls will results in an error because no column names exist,
        in an effort to make the inner workings more transparent and rigorous.

        This function allows to override the default behavior, and can also generate column
        names by itself if none are specified.
        """
        # check if input is a Timeseries, override rest of input arguments
        if timeseries is not None:
            if not isinstance(timeseries, Timeseries):
                raise TypeError("Cannot use uncertainty data: "
                                "'timeseries' is not a Timeseries object.")
            # get variance data and description
            if timeseries.var_cols is not None:
                var_data = timeseries.vars[timeseries.time.isin(self.time)].values
                var_cols = timeseries.var_cols
            else:
                var_data = None
                var_cols = None
            # get covariance data and description
            if timeseries.cov_cols is not None:
                cov_data = timeseries.covs[timeseries.time.isin(self.time)].values
                cov_cols = timeseries.cov_cols
            else:
                cov_data = None
                cov_cols = None
        # add variance
        if var_data is not None:
            assert isinstance(var_data, np.ndarray) and var_data.shape == self.shape, \
                   "'var_data' must be a NumPy array of the same shape as the data " \
                   f"{self.shape}, got " \
                   f"{var_data.shape if isinstance(var_data, np.ndarray) else var_data}."
            if var_cols is None:
                var_cols = [dcol + "_sigma" for dcol in self.data_cols]
            else:
                assert (isinstance(var_cols, list) and len(var_cols) == self.num_components and
                        all([isinstance(vcol, str) for vcol in var_cols])), \
                       "New 'var_cols' attribute must be a list of strings of the same length " \
                       f"as the current 'data_cols' ({len(self._data_cols)}), got {var_cols}."
            for ivcol, vcol in enumerate(var_cols):
                self._df[vcol] = var_data[:, ivcol]
            self._var_cols = var_cols
        # add covariance, check for variance
        if cov_data is not None:
            assert self._var_cols is not None, \
                "Cannot set covariance data without first adding variance data."
            cov_dims = get_cov_dims(self.num_components)
            assert (isinstance(cov_data, np.ndarray)
                    and cov_data.shape[0] == self.num_observations
                    and cov_data.shape[1] == cov_dims), \
                "'cov_data' must be a NumPy array of the same length as the data " \
                "and width corresponding to all possible covariances. Expected shape " \
                f"({self.num_observations}, {cov_dims}), got " \
                f"{cov_data.shape if isinstance(cov_data, np.ndarray) else cov_data}."
            if cov_cols is None:
                cov_cols = []
                for i1 in range(self.num_components):
                    for i2 in range(i1 + 1, self.num_components):
                        cov_cols.append(f"{self.data_cols[i1]}_{self.data_cols[i2]}_cov")
            else:
                assert (isinstance(cov_cols, list) and len(cov_cols) == cov_dims and
                        all([isinstance(vcol, str) for vcol in cov_cols])), \
                    "New 'cov_cols' attribute must be a list of strings of length " \
                    "corresponding to all possible covariances " \
                    f"( expected {cov_dims}, got {len(cov_cols)}."
            for iccol, ccol in enumerate(cov_cols):
                self._df[ccol] = cov_data[:, iccol]
            self.index_map, self.var_cov_map = make_cov_index_map(self.num_components)
            self._cov_cols = cov_cols

    def copy(self, only_data: bool = False, src: str | None = None) -> Timeseries:
        """
        Return a deep copy of the timeseries instance.

        Parameters
        ----------
        only_data
            If ``True``, only copy the data columns and ignore any uncertainty information.
        src
            Set a new source information attribute for the copy.
            Uses the current one if ``None``.

        Returns
        -------
        Timeseries
            The copy of the timeseries instance.
        """
        new_name = deepcopy(self._src) if src is None else src
        if not only_data:
            return Timeseries(self._df.copy(), new_name, deepcopy(self._data_unit),
                              deepcopy(self._data_cols), deepcopy(self._var_cols),
                              deepcopy(self._cov_cols))
        else:
            return Timeseries(self._df[self._data_cols].copy(), new_name,
                              deepcopy(self._data_unit), deepcopy(self._data_cols), None, None)

    def convert_units(self, factor: float, new_data_unit: str) -> None:
        """
        Convert the data and covariances to a new data unit by providing a
        conversion factor.

        Parameters
        ----------
        factor
            Factor to multiply the data by to obtain the data in the new units.
        new_data_unit
            New data unit to be saved in the :attr:`~data_cols` attribute.
        """
        # input checks
        try:
            factor = float(factor)
        except TypeError as e:
            raise TypeError(f"'factor' and has to be a scalar, got {type(factor)}."
                            ).with_traceback(e.__traceback__) from e
        # update data unit (contains input check)
        self.data_unit = new_data_unit
        # convert data
        self._df.loc[:, self._data_cols] *= factor
        # convert variances
        if self._var_cols is not None:
            self._df.loc[:, self._var_cols] *= factor**2
        # convert covariances
        if self._cov_cols is not None:
            self._df.loc[:, self._cov_cols] *= factor**2

    def mask_out(self, dcol: str) -> None:
        """
        Mask out an entire data column (and if present, its uncertainty column) by setting
        the entire column to ``NaN``. Converts it to a sparse representation to save memory.

        Parameters
        ----------
        dcol
            Name of the data column to mask out.
        """
        icomp = self._data_cols.index(dcol)
        self._df[dcol] = np.nan
        self._df[dcol] = self._df[dcol].astype(pd.SparseDtype(dtype=float))
        if self._var_cols is not None:
            vcol = self._var_cols[icomp]
            self._df[vcol] = np.nan
            self._df[vcol] = self._df[vcol].astype(pd.SparseDtype(dtype=float))
            if self._cov_cols is not None:
                for iccol in get_cov_indices(icomp, index_map=self.index_map):
                    ccol = self._cov_cols[iccol]
                    self._df[ccol] = np.nan
                    self._df[ccol] = self._df[ccol].astype(pd.SparseDtype(dtype=float))

    @staticmethod
    def prepare_math(left: Timeseries | np.ndarray,
                     right: Timeseries | np.ndarray,
                     operation: Literal["+", "-", "*", "/"]
                     ) -> tuple[np.ndarray, np.ndarray, str, str, list[str], pd.Index]:
        r"""
        Tests two timeseries' ability to be cast together in a mathematical operation,
        and returns output characteristics.
        Currently, only addition, subtraction, multiplication, and division are supported.

        All uncertainty information is lost during mathematical operations.

        One of the objects can be a NumPy array. In this case, the array has to have
        the exact same shape as the data in the Timeseries instance.
        Furthermore, the resulting Timeseries object will have the same :attr:`~src`,
        :attr:`~data_unit` and :attr:`~data_cols` attributes (instead of a combination
        of both).

        Parameters
        ----------
        left
            Left term of the operation.
        right
            Right term of the operation.
        operation
            Operation to perform.

        Returns
        -------
        left_data
            View of the 2D left data array of the operation
            with shape (``len(out_time)``, :attr:`~num_components`).
        right_data
            View of the 2D right data array of the operation
            with shape (``len(out_time)``, :attr:`~num_components`).
        out_src
            Combines the sources of each object to a new string.
        out_data_unit
            Combines the data units of each object into a new unit.
        out_data_cols
            List of strings containing the new data column names.
        out_time
            Index object containing the indices of all timestamps common to both.

        Raises
        ------
        TypeError
            If one of the operands is not a :class:`~Timeseries` or :class:`~numpy.ndarray`,
            or if both are :class:`~numpy.ndarray` (since then this function would never be
            called anyway).
        ValueError
            If the number of data columns is not equal between the two operands,
            or if the data units are not the same adding or subtracting.
        AssertionError
            If one of the operands is a NumPy array but does not have the same number of
            rows as the other operand.

        Warning
        -------
        This method is called under-the-hood whenever a mathematical operation is
        performed, and should not need to be used by normal users.

        See Also
        --------
        __add__ : Addition for two Timeseries or a Timeseries and a NumPy array
        __radd__ : Addition for a NumPy array and a Timeseries.
        __sub__ : Subtraction for two Timeseries or a Timeseries and a NumPy array
        __rsub__ : Subtraction for a NumPy array and a Timeseries.
        __mul__ : Multiplication for two Timeseries or a Timeseries and a NumPy array
        __rmul__ : Multiplication for a NumPy array and a Timeseries.
        __truediv__ : Division for two Timeseries or a Timeseries and a NumPy array
        __rtruediv__ : Division for a NumPy array and a Timeseries.
        """
        # check for valid operator
        if operation not in ["+", "-", "*", "/"]:
            raise NotImplementedError("Timeseries math problem: "
                                      f"unknown operation '{operation}'.")
        # check for compatible types
        for operand, position in zip([left, right], ["Left", "Right"]):
            if (not isinstance(operand, Timeseries)) and (not isinstance(operand, np.ndarray)):
                raise TypeError(f"{position} operand has to be either a '{Timeseries}' or "
                                f"'{np.ndarray}' object, got {type(operand)}")
        if all([isinstance(operand, np.ndarray) for operand in [left, right]]):
            raise TypeError(f"At least one of the operands has to be a '{Timeseries}' object, "
                            f"got two '{np.ndarray}' objects.")
        # check for same dimensions
        if isinstance(left, Timeseries):
            len_left_data_cols = left.num_components
        else:
            len_left_data_cols = left.shape[1]
        if isinstance(right, Timeseries):
            len_right_data_cols = right.num_components
        else:
            len_right_data_cols = right.shape[1]
        if len_left_data_cols != len_right_data_cols:
            raise ValueError("Timeseries math problem: conflicting number of data columns "
                             f"({len_left_data_cols} and {len_right_data_cols}).")
        # get output attributes
        if isinstance(left, np.ndarray):
            assert left.shape[0] == right.num_observations, \
                "Timeseries math problem: conflicting number of observation " + \
                f"({left.shape[0]} and {right.num_observations})."
            out_time = right.time
            out_src = right.src
            out_data_unit = right.data_unit
            out_data_cols = right.data_cols
            left_data = left
            right_data = right.data.loc[out_time, :].values
        elif isinstance(right, np.ndarray):
            assert left.num_observations == right.shape[0], \
                "Timeseries math problem: conflicting number of observation " + \
                f"({left.num_observations} and {right.shape[0]})."
            out_time = left.time
            out_src = left.src
            out_data_unit = left.data_unit
            out_data_cols = left.data_cols
            left_data = left.data.loc[out_time, :].values
            right_data = right
        else:
            # time index is the intersection
            out_time = left.time.intersection(right.time, sort=None)
            # check compatible units (+-) or define new one (*/)
            # define new src and data_cols
            if operation in ["+", "-"]:
                if left.data_unit != right.data_unit:
                    raise ValueError("Timeseries math problem: conflicting data units "
                                     f"'{left.data_unit}' and '{right.data_unit}'.")
                else:
                    out_data_unit = left.data_unit
                    out_src = f"{left.src}{operation}{right.src}"
                    out_data_cols = [f"{lcol}{operation}{rcol}" for lcol, rcol
                                     in zip(left.data_cols, right.data_cols)]
            elif operation in ["*", "/"]:
                out_data_unit = f"({left.data_unit}){operation}({right.data_unit})"
                out_src = f"({left.src}){operation}({right.src})"
                out_data_cols = [f"({lcol}){operation}({rcol})" for lcol, rcol
                                 in zip(left.data_cols, right.data_cols)]
            left_data = left.data.loc[out_time, :].values
            right_data = right.data.loc[out_time, :].values
        # return data unit and column names
        return left_data, right_data, out_src, out_data_unit, out_data_cols, out_time

    def __add__(self, other: Timeseries) -> Timeseries:
        """
        Special function that allows two timeseries instances (or a timeseries and
        an equivalently shaped NumPy array) to be added together element-wise.

        Parameters
        ----------
        other
            Timeseries to add to instance.

        Returns
        -------
            New timeseries object containing the sum of the two timeseries.

        See Also
        --------
        prepare_math
            Prepares the two instances for the mathematical operation. Refer to it
            for more details about how the two objects are cast together.

        Example
        -------
        Add two :class:`~Timeseries` ``ts1`` and ``ts2`` and save the result as ``ts3``::

            ts3 = ts1 + ts2
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(self, other, '+')
        return Timeseries(pd.DataFrame(left_data + right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __radd__(self, other: Timeseries) -> Timeseries:
        """
        Reflected operation of :meth:`~__add__` (necessary if first operand is a NumPy array).
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(other, self, '+')
        return Timeseries(pd.DataFrame(left_data + right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __sub__(self, other: Timeseries) -> Timeseries:
        """
        Special function that allows a timeseries instance (or a timeseries and
        an equivalently shaped NumPy array) to be subtracted from another element-wise.

        Parameters
        ----------
        other
            Timeseries to subtract from instance.

        Returns
        -------
            New timeseries object containing the difference of the two timeseries.

        See Also
        --------
        prepare_math
            Prepares the two instances for the mathematical operation. Refer to it
            for more details about how the two objects are cast together.

        Example
        -------
        Subtract two :class:`~Timeseries` ``ts1`` and ``ts2`` and save the result as ``ts3``::

            ts3 = ts1 - ts2
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(self, other, '-')
        return Timeseries(pd.DataFrame(left_data - right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __rsub__(self, other: Timeseries) -> Timeseries:
        """
        Reflected operation of :meth:`~__sub__` (necessary if first operand is a NumPy array).
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(other, self, '-')
        return Timeseries(pd.DataFrame(left_data - right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __mul__(self, other: Timeseries) -> Timeseries:
        """
        Special function that allows two timeseries instances (or a timeseries and
        an equivalently shaped NumPy array) to be multiplied together element-wise.

        Parameters
        ----------
        other
            Timeseries to multiply to instance.

        Returns
        -------
            New timeseries object containing the product of the two timeseries.

        See Also
        --------
        prepare_math
            Prepares the two instances for the mathematical operation. Refer to it
            for more details about how the two objects are cast together.

        Example
        -------
        Multiply two :class:`~Timeseries` ``ts1`` and ``ts2`` and save the result as ``ts3``::

            ts3 = ts1 * ts2
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(self, other, '*')
        return Timeseries(pd.DataFrame(left_data * right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __rmul__(self, other: Timeseries) -> Timeseries:
        """
        Reflected operation of :meth:`~__mul__` (necessary if first operand is a NumPy array).
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(other, self, '*')
        return Timeseries(pd.DataFrame(left_data * right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __truediv__(self, other: Timeseries) -> Timeseries:
        """
        Special function that allows a timeseries instances (or a timeseries and
        an equivalently shaped NumPy array) to be divided by another element-wise.

        Parameters
        ----------
        other
            Timeseries to divide instance by.

        Returns
        -------
            New timeseries object containing the quotient of the two timeseries.

        See Also
        --------
        prepare_math
            Prepares the two instances for the mathematical operation. Refer to it
            for more details about how the two objects are cast together.

        Example
        -------
        Divide two :class:`~Timeseries` ``ts1`` and ``ts2`` and save the result as ``ts3``::

            ts3 = ts1 / ts2
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(self, other, '/')
        return Timeseries(pd.DataFrame(left_data / right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __rtruediv__(self, other: Timeseries) -> Timeseries:
        """
        Reflected operation of :meth:`~__truediv__` (necessary if first operand is a NumPy array).
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = \
            Timeseries.prepare_math(other, self, '/')
        return Timeseries(pd.DataFrame(left_data / right_data,
                                       index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def get_arch(self) -> dict:
        """
        Build a dictionary describing the architecture of this timeseries,
        to be used when creating a network JSON configuration file.

        Without subclassing :class:`~Timeseries`, this function will return an
        empty dictionary by default, since it is unknown how to recreate
        a general Timeseries object from just a JSON-compatible dictionary.

        See Also
        --------
        disstans.network.Network.to_json : Export the Network configuration as a JSON file.
        disstans.timeseries.Timeseries.get_arch
            Get the architecture dictionary of a :class:`~disstans.timeseries.Timeseries` instance.
        disstans.models.Model.get_arch
            Get the architecture dictionary of a :class:`~disstans.models.Model` instance.
        """
        return {}

    @classmethod
    def from_fit(cls,
                 data_unit: str,
                 data_cols: list[str],
                 fit: dict[str, np.ndarray | None]
                 ) -> Timeseries:
        """
        Import a fit dictionary and create a Timeseries instance.

        Parameters
        ----------
        data_unit
            Data unit.
        data_cols
            List of strings containing the data column names.
            Uncertainty column names are generated by adding a '_var'.
        fit
            Dictionary with the keys ``'time'``, ``'fit'``, ``'var'`` and ``'cov'``
            (the latter two can be set to ``None``).

        Returns
        -------
            Timeseries instance created from ``fit``.

        See Also
        --------
        disstans.models.Model.evaluate
            Evaluating a model produces the fit dictionary.
        """
        df_data = {dcol: fit["fit"][:, icol] for icol, dcol in enumerate(data_cols)}
        var_cols = None
        cov_cols = None
        if fit["var"] is not None:
            var_cols = [dcol + "_var" for dcol in data_cols]
            df_data.update({vcol: fit["var"][:, icol] for icol, vcol in enumerate(var_cols)})
            if fit["cov"] is not None:
                cov_cols = []
                num_components = len(data_cols)
                for i1 in range(num_components):
                    for i2 in range(i1 + 1, num_components):
                        cov_cols.append(f"{data_cols[i1]}_{data_cols[i2]}_cov")
                df_data.update({ccol: fit["cov"][:, icol] for icol, ccol in enumerate(cov_cols)})
        df = pd.DataFrame(data=df_data, index=fit["time"])
        return cls(df, "fitted", data_unit, data_cols, var_cols, cov_cols)

    @classmethod
    def from_array(cls,
                   timevector: pd.Series | pd.DatetimeIndex,
                   data: np.ndarray,
                   src: str,
                   data_unit: str,
                   data_cols: str,
                   var: np.ndarray | None = None,
                   var_cols: list[str] | None = None,
                   cov: np.ndarray | None = None,
                   cov_cols: list[str] | None = None
                   ) -> Timeseries:
        r"""
        Constructor method to create a :class:`~Timeseries` instance from a NumPy
        :class:`~numpy.ndarray`.

        Parameters
        ----------
        timevector
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.
        data
            2D NumPy array of shape :math:`(\text{n_observations},\text{n_components})`
            containing the data.
        src
            Source description.
        data_unit
            Data unit.
        data_cols
            List of strings with the names of the columns of ``data``.
        var
            2D NumPy array of shape :math:`(\text{n_observations},\text{n_components})`
            containing the data variances.
            ``None`` defaults to no data uncertainty.
        var_cols
            List of strings with the names of the columns of ``data`` that contain the
            data's variance.
            Must have the same length as ``data_cols``.
            If ``var`` is given but ``var_cols`` is not, it defaults to appending
            ``'_var'`` to ``data_cols``.
        cov
            2D NumPy array of shape :math:`(\text{n_observations},\text{n_components})`
            containing the data covariances (as defined in :class:`~Timeseries`).
            ``None`` defaults to no data uncertainty.
        cov_cols
            List of strings with the names of the columns of ``data`` that contain the
            data's covariance.
            Must have the same length as ``data_cols``.
            If ``cov`` is given but ``cov_cols`` is not, it defaults to appending
            ``'_cov'`` to the two respective entries of ``data_cols``.

        Returns
        -------
            The generated Timeseries object.

        See Also
        --------
        :func:`~pandas.date_range` : Quick function to generate a timevector.
        """
        assert len(timevector) == data.shape[0], \
            "Length of 'timevector' has to match the number of rows in 'data', " \
            f"got {len(timevector)} and {data.shape}."
        assert isinstance(data_cols, list) and \
            all([isinstance(dcol, str) for dcol in data_cols]), \
            f"'data_cols' has to be a list of strings, got {data_cols}."
        # create dataframe with data
        df_data = {dcol: data[:, icol] for icol, dcol in enumerate(data_cols)}
        # add variances
        if var is None:
            var_cols = None
        else:
            assert data.shape == var.shape, \
                "'data' and 'var' need to have the same shape, got " \
                f"{data.shape} and {var.shape}."
            if var_cols is None:
                var_cols = [dcol + "_var" for dcol in data_cols]
            else:
                assert len(var_cols) == var.shape[1]
            df_data.update({vcol: var[:, icol] for icol, vcol in enumerate(var_cols)})
            # if variances are given, can also check for covariances
            if cov is None:
                cov_cols = None
            else:
                num_components = data.shape[1]
                cov_dims = get_cov_dims(num_components)
                if cov.ndim == 1:
                    cov = cov.reshape(-1, 1)
                assert cov.shape == (data.shape[0], cov_dims), \
                    "'data' and 'cov' need to have compatible shapes, got " \
                    f"{data.shape} and {cov.shape} (need {cov_dims} columns)."
                if cov_cols is None:
                    cov_cols = []
                    for i1 in range(num_components):
                        for i2 in range(i1 + 1, num_components):
                            cov_cols.append(f"{data_cols[i1]}_{data_cols[i2]}_cov")
                else:
                    assert len(cov_cols) == cov.shape[1]
                df_data.update({ccol: cov[:, icol] for icol, ccol in enumerate(cov_cols)})
        df = pd.DataFrame(data=df_data, index=timevector)
        return cls(df, src, data_unit, data_cols, var_cols, cov_cols)


class GipsyTimeseries(Timeseries):
    """
    Subclasses :class:`~Timeseries`.

    Timeseries subclass for GNSS measurements in JPL's Gipsy(X) ``.tseries`` file format.

    Parameters
    ----------
    path
        Path to the timeseries file.
    show_warnings
        If ``True``, warn if there are data inconsistencies encountered while loading.
    data_unit
        Can be ``'mm'`` or ``'m'``.
    keep_duplicates
        Argument passed on to :meth:`~pandas.DataFrame.drop_duplicates` as the ``keep``
        parameter when finding duplicate times, but with ``False`` as the default
        (since duplicate timestamps usually imply bad data - legacy behavior would be
        to pass ``'first'``).


    Additional keyword arguments will be passed onto :class:`~Timeseries`.

    Notes
    -----

    The column format is described on `JPL's website`_:

    +---------------+-------------------------------------------------+
    | Columns       | Description                                     |
    +===============+=================================================+
    | Column 1      | Decimal year computed with 365.25 days/yr       |
    +---------------+-------------------------------------------------+
    | Columns 2-4   | East, North and Vertical [m]                    |
    +---------------+-------------------------------------------------+
    | Columns 5-7   | East, North and Vertical standard deviation [m] |
    +---------------+-------------------------------------------------+
    | Columns 8-10  | East, North and Vertical correlation [-]        |
    +---------------+-------------------------------------------------+
    | Column 11     | Time in Seconds past J2000                      |
    +---------------+-------------------------------------------------+
    | Columns 12-17 | Time in YEAR MM DD HR MN SS                     |
    +---------------+-------------------------------------------------+

    Time is GPS time, and the time series are relative to each station's first epoch.

    .. _JPL's website: https://sideshow.jpl.nasa.gov/pub/JPL_GPS_Timeseries/\
repro2018a/raw/position/envseries/0000_README.format

    """
    def __init__(self,
                 path: str,
                 show_warnings: bool = True,
                 data_unit: Literal["mm", "m"] = "mm",
                 keep_duplicates: Literal["first", "last", False] = False,
                 **kw_args
                 ) -> None:
        self._path = str(path)
        self._keep_duplicates = keep_duplicates
        if data_unit == "m":
            factor = 1
        elif data_unit == "mm":
            factor = 1000
        else:
            raise ValueError(f"'data_unit' needs to be 'mm' or 'm', got {data_unit}.")
        # load data
        data_cols = ["east", "north", "up"]
        var_cols = ["east_var", "north_var", "up_var"]
        cov_cols = ["east_north_cov", "east_up_cov", "north_up_cov"]
        all_cols = data_cols + var_cols + cov_cols
        time = pd.read_csv(self._path, sep=r"\s+", header=None,
                           usecols=[11, 12, 13, 14, 15, 16],
                           names=["year", "month", "day", "hour", "minute", "second"])
        time = pd.to_datetime(time).to_frame(name="time")
        data = pd.read_csv(self._path, sep=r"\s+", header=None,
                           usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                           names=all_cols)
        # compute covariance from correlation, still in meters
        data.loc[:, "east_north_cov"] *= data.loc[:, "east_var"] * data.loc[:, "north_var"]
        data.loc[:, "east_up_cov"] *= data.loc[:, "east_var"] * data.loc[:, "up_var"]
        data.loc[:, "north_up_cov"] *= data.loc[:, "north_var"] * data.loc[:, "up_var"]
        # convert standard deviation into variance, still in meters
        data.loc[:, var_cols] *= data.loc[:, var_cols]
        # now convert everything to millimeters
        data.loc[:, data_cols] *= factor
        data.loc[:, var_cols + cov_cols] *= factor**2
        # combine dataframe with time
        df = time.join(data)
        # check for duplicate timestamps
        num_duplicates = int(df.duplicated(subset="time").sum())
        if num_duplicates > 0:
            if show_warnings:
                warn(f"Timeseries file {self._path} contains data for {num_duplicates} "
                     f"duplicate times, dealing with them using 'keep={self._keep_duplicates}'.",
                     stacklevel=2)
            df.drop_duplicates(subset="time", keep=self._keep_duplicates, inplace=True)
        # create time index
        df.set_index("time", inplace=True)
        # check for monotonic time index
        if not df.index.is_monotonic_increasing:
            warn(f"Timeseries file {self._path} is not ordered in time, sorting it now.",
                 stacklevel=2)
            df.sort_index(inplace=True)
        # construct Timeseries object
        super().__init__(dataframe=df, src="Gipsy", data_unit=data_unit,
                         data_cols=data_cols, var_cols=var_cols, cov_cols=cov_cols,
                         **kw_args)

    def get_arch(self) -> dict:
        """
        Returns a JSON-compatible dictionary with all the information necessary to recreate
        the Timeseries instance (provided the data file is available).

        Returns
        -------
        dict
            JSON-compatible dictionary sufficient to recreate the GipsyTimeseries instance.

        See Also
        --------
        Timeseries.get_arch : For further information.
        """
        return {"type": "GipsyTimeseries",
                "kw_args": {"path": self._path,
                            "keep_duplicates": self._keep_duplicates}}


GipsyXTimeseries = GipsyTimeseries
""" Alias for :class:`~GipsyTimeseries`. """


class UNRTimeseries(Timeseries):
    """
    Subclasses :class:`~Timeseries`.

    Timeseries subclass for GNSS measurements in UNR's ``.tenv3`` file format.

    Parameters
    ----------
    path
        Path to the timeseries file.
    show_warnings
        If ``True``, warn if there are data inconsistencies encountered while loading.
    data_unit
        Can be ``'mm'`` or ``'m'``.


    Additional keyword arguments will be passed onto :class:`~Timeseries`.

    Notes
    -----

    The column format is described on `UNR's website`_:

    +---------------+---------------------------------------------------+
    | Columns       | Description                                       |
    +===============+===================================================+
    | Column 1      | Station name                                      |
    +---------------+---------------------------------------------------+
    | Column 2      | Date                                              |
    +---------------+---------------------------------------------------+
    | Column 3      | Decimal year                                      |
    +---------------+---------------------------------------------------+
    | Column 4      | Modified Julian day                               |
    +---------------+---------------------------------------------------+
    | Columns 5-6   | GPS week and day                                  |
    +---------------+---------------------------------------------------+
    | Column 7      | Longitude [°] of reference meridian               |
    +---------------+---------------------------------------------------+
    | Columns 8-9   | Easting [m] from ref. mer., integer and fraction  |
    +---------------+---------------------------------------------------+
    | Columns 10-11 | Northing [m] from equator, integer and fraction   |
    +---------------+---------------------------------------------------+
    | Columns 12-13 | Vertical [m], integer and fraction                |
    +---------------+---------------------------------------------------+
    | Column 14     | Antenna height [m]                                |
    +---------------+---------------------------------------------------+
    | Column 15-17  | East, North, Vertical standard deviation [m]      |
    +---------------+---------------------------------------------------+
    | Column 18     | East-North correlation coefficient [-]            |
    +---------------+---------------------------------------------------+
    | Column 19     | East-Vertical correlation coefficient [-]         |
    +---------------+---------------------------------------------------+
    | Column 20     | North-Vertical correlation coefficient [-]        |
    +---------------+---------------------------------------------------+

    Newer files also contain the following three columns:

    +---------------+---------------------------------------------------+
    | Column 21     | Latitude [°]                                      |
    +---------------+---------------------------------------------------+
    | Column 22     | Longitude [°]                                     |
    +---------------+---------------------------------------------------+
    | Column 23     | Altitude [m]                                      |
    +---------------+---------------------------------------------------+

    The time series are relative to each station's first integer epoch.

    .. _UNR's website: http://geodesy.unr.edu/gps_timeseries/README_tenv3.txt

    """
    def __init__(self,
                 path: str,
                 show_warnings: bool = True,
                 data_unit: Literal["mm", "m"] = "mm",
                 **kw_args
                 ) -> None:
        self._path = str(path)
        if data_unit == "m":
            factor = 1
        elif data_unit == "mm":
            factor = 1000
        else:
            raise ValueError(f"'data_unit' needs to be 'mm' or 'm', got {data_unit}.")
        # load data and check for some warnings
        df = pd.read_csv(self._path, sep=r"\s+",
                         usecols=[0, 3] + list(range(6, 13)) + list(range(14, 20)))
        if show_warnings and len(df['site'].unique()) > 1:
            warn(f"Timeseries file {self._path} contains multiple site codes: "
                 f"{df['site'].unique()}", stacklevel=2)
        if len(df['reflon'].unique()) > 1:
            calculate_stepdates = True
            if show_warnings:
                warn(f"Timeseries file {self._path} contains multiple reference longitudes: "
                     f"{df['reflon'].unique()}. Step dates will be saved to "
                     "'self._reflonsteptimes' and should be added as free model parameters, "
                     "or the affected observations should be removed.", stacklevel=2)
        else:
            calculate_stepdates = False
        if len(df['_e0(m)'].unique()) > 1:
            if show_warnings:
                warn(f"Timeseries file {self._path} contains multiple integer "
                     f"Eastings: {df['_e0(m)'].unique()}", stacklevel=2)
            offsets_east = df["_e0(m)"].values - df["_e0(m)"].values[0]
        else:
            offsets_east = 0
        if len(df['____n0(m)'].unique()) > 1:
            if show_warnings:
                warn(f"Timeseries file {self._path} contains multiple integer "
                     f"Northings: {df['____n0(m)'].unique()}", stacklevel=2)
            offsets_north = df["____n0(m)"].values - df["____n0(m)"].values[0]
        else:
            offsets_north = 0
        if len(df['u0(m)'].unique()) > 1:
            if show_warnings:
                warn(f"Timeseries file {self._path} contains multiple integer "
                     f"Verticals: {df['u0(m)'].unique()}", stacklevel=2)
            offsets_up = df["u0(m)"].values - df["u0(m)"].values[0]
        else:
            offsets_up = 0
        # make the data
        df["east"] = (df["__east(m)"] + offsets_east) * factor
        df["north"] = (df["_north(m)"] + offsets_north) * factor
        df["up"] = (df["____up(m)"] + offsets_up) * factor
        df.drop(columns=["_e0(m)", "__east(m)", "____n0(m)", "_north(m)",
                         "u0(m)", "____up(m)"], inplace=True)
        # make the covariance
        df["__corr_en"] *= df["sig_e(m)"] * df["sig_n(m)"] * factor**2
        df["__corr_eu"] *= df["sig_e(m)"] * df["sig_u(m)"] * factor**2
        df["__corr_nu"] *= df["sig_n(m)"] * df["sig_u(m)"] * factor**2
        df.rename(columns={"__corr_en": "east_north_cov", "__corr_eu": "east_up_cov",
                           "__corr_nu": "north_up_cov"}, inplace=True)
        # make the variance
        old_sig_cols = ["sig_e(m)", "sig_n(m)", "sig_u(m)"]
        df.loc[:, old_sig_cols] = (df.loc[:, old_sig_cols] * factor)**2
        df.rename(columns={"sig_e(m)": "east_var", "sig_n(m)": "north_var",
                           "sig_u(m)": "up_var"}, inplace=True)
        # check for duplicate timestamps and create time index
        num_duplicates = int(df.duplicated(subset="__MJD").sum())
        if show_warnings and (num_duplicates > 0):
            warn(f"Timeseries file {self._path} contains data for {num_duplicates} "
                 "duplicate dates. Keeping first occurrences.", stacklevel=2)
        df.drop_duplicates(subset="__MJD", inplace=True)
        df["__MJD"] = pd.to_datetime(df["__MJD"] + 2400000.5, unit="D", origin='julian')
        df.rename(columns={"__MJD": "time"}, inplace=True)
        df.set_index("time", inplace=True)
        # check for monotonic time index
        if not df.index.is_monotonic_increasing:
            warn(f"Timeseries file {self._path} is not ordered in time, sorting it now.",
                 stacklevel=2)
            df.sort_index(inplace=True)
        # calculate step dates for multiple reference meridians
        if calculate_stepdates:
            self._reflonsteptimes = df.index[df["reflon"].diff().abs() > 0]
        # remove columns that are no longer needed
        df.drop(columns=["site", "reflon"], inplace=True)
        # construct Timeseries object
        super().__init__(dataframe=df, src="UNR", data_unit=data_unit,
                         data_cols=["east", "north", "up"],
                         var_cols=["east_var", "north_var", "up_var"],
                         cov_cols=["east_north_cov", "east_up_cov", "north_up_cov"],
                         **kw_args)

    def get_arch(self) -> dict:
        """
        Returns a JSON-compatible dictionary with all the information necessary to recreate
        the Timeseries instance (provided the data file is available).

        Returns
        -------
        dict
            JSON-compatible dictionary sufficient to recreate the UNRTimeseries instance.

        See Also
        --------
        Timeseries.get_arch : For further information.
        """
        return {"type": "UNRTimeseries",
                "kw_args": {"path": self._path}}


class UNRHighRateTimeseries(Timeseries):
    """
    Subclasses :class:`~Timeseries`.

    Timeseries subclass for GNSS measurements in UNR's ``.kenv`` file format.

    Parameters
    ----------
    path
        Path to the timeseries file.
    show_warnings
        If ``True``, warn if there are data inconsistencies encountered while loading.
    data_unit
        Can be ``'mm'`` or ``'m'``.


    Additional keyword arguments will be passed onto :class:`~Timeseries`.

    Notes
    -----

    The column format is described on `UNR's website`_:

    +---------------+---------------------------------------------------+
    | Columns       | Description                                       |
    +===============+===================================================+
    | Column 1      | Station name                                      |
    +---------------+---------------------------------------------------+
    | Column 2      | GPS seconds since J2000                           |
    +---------------+---------------------------------------------------+
    | Column 3      | Modified Julian Day for GPS day                   |
    +---------------+---------------------------------------------------+
    | Columns 4-7   | Year, month, day of month, day of year            |
    +---------------+---------------------------------------------------+
    | Column 8      | Seconds of the GPS day                            |
    +---------------+---------------------------------------------------+
    | Columns 9-11  | East, North, Up from reference location           |
    +---------------+---------------------------------------------------+
    | Columns 12-14 | East, North, Up from daily mean position          |
    +---------------+---------------------------------------------------+
    | Columns 15-17 | East, North, Up standard deviations               |
    +---------------+---------------------------------------------------+

    Note that only the location timeseries relative to the reference locations
    is loaded.
    The timestamp associated with each observation is in Terrestrial Time.

    """
    def __init__(self,
                 path: str,
                 show_warnings: bool = True,
                 data_unit: Literal["mm", "m"] = "mm",
                 **kw_args
                 ) -> None:
        self._path = str(path)
        if data_unit == "m":
            factor = 1
        elif data_unit == "mm":
            factor = 1000
        else:
            raise ValueError(f"'data_unit' needs to be 'mm' or 'm', got {data_unit}.")
        # load data
        pathobj = Path(self._path)
        # if the path is a folder or a .kenv.zip file, we need to concatenate files
        if pathobj.is_dir() or pathobj.match("*.kenv.zip"):
            # collect list of files if path is a folder
            if pathobj.is_dir():
                paths_years = sorted(pathobj.glob("*.kenv.zip"))
            else:
                paths_years = [pathobj]
            # initialize memory buffer for pandas later
            f = BytesIO()
            # loop over years
            for path_y in paths_years:
                # unzip year
                try:
                    with zipfile.ZipFile(path_y) as zip_y:
                        # get list of daily compressed files
                        paths_days = sorted(zip_y.namelist())
                        # load every day
                        for path_d in paths_days:
                            with zip_y.open(path_d) as zip_d:
                                try:
                                    with gzip.open(zip_d) as gz_d:
                                        # skip header line
                                        gz_d.readline()
                                        # write to buffer
                                        f.write(gz_d.read())
                                except gzip.BadGzipFile as e:
                                    raise RuntimeError(f"Error extracting the file {path_y}."
                                                       ).with_traceback(e.__traceback__) from e
                except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
                    raise RuntimeError(f"Error extracting the file {path_y}."
                                       ).with_traceback(e.__traceback__) from e
            # load data into pandas
            f.seek(0)
            df = pd.read_csv(f, sep=r"\s+",
                             names=["site", "sec-J2000", "___e-ref(m)", "___n-ref(m)",
                                    "___v-ref(m)", "sig_e(m)", "sig_n(m)", "sig_v(m)"],
                             usecols=[0, 1] + list(range(8, 11)) + list(range(14, 17)))
        # if the path is a .kenv.gz file, we only need to extract the single file
        elif pathobj.match("*.kenv.gz"):
            with gzip.open(self._path, mode="r") as f:
                df = pd.read_csv(f, sep=r"\s+",
                                 usecols=[0, 1] + list(range(8, 11)) + list(range(14, 17)))
        # in all other cases, try loading directly
        else:
            df = pd.read_csv(self._path, sep=r"\s+",
                             usecols=[0, 1] + list(range(8, 11)) + list(range(14, 17)))
        # check for duplicate sites
        if show_warnings and len(df['site'].unique()) > 1:
            warn(f"Timeseries file/folder {self._path} contains multiple site codes: "
                 f"{df['site'].unique()}", stacklevel=2)
        # remove site column
        df.drop(columns=["site"], inplace=True)
        # convert data units
        df["___e-ref(m)"] *= factor
        df["___n-ref(m)"] *= factor
        df["___v-ref(m)"] *= factor
        # convert to variance and change units
        df["sig_e(m)"] *= df["sig_e(m)"] * factor**2
        df["sig_n(m)"] *= df["sig_n(m)"] * factor**2
        df["sig_v(m)"] *= df["sig_v(m)"] * factor**2
        # check for duplicate timestamps
        num_duplicates = int(df.duplicated(subset="sec-J2000").sum())
        if show_warnings and (num_duplicates > 0):
            warn(f"Timeseries file/folder {self._path} contains data for {num_duplicates} "
                 "duplicate timestamps. Keeping first occurrences.", stacklevel=2)
        df.drop_duplicates(subset="sec-J2000", inplace=True)
        # create time index and rename columns
        df["sec-J2000"] = (pd.Timestamp("2000-01-01 12:00:00") +
                           pd.to_timedelta(df["sec-J2000"], unit="s"))
        df.rename(columns={"___e-ref(m)": "east", "sig_e(m)": "east_var",
                           "___n-ref(m)": "north", "sig_n(m)": "north_var",
                           "___v-ref(m)": "up", "sig_v(m)": "up_var",
                           "sec-J2000": "time"}, inplace=True)
        df.set_index("time", inplace=True)
        # check for monotonic time index
        if not df.index.is_monotonic_increasing:
            warn(f"Timeseries file/folder {self._path} is not ordered in time, sorting it now.",
                 stacklevel=2)
            df.sort_index(inplace=True)
        # construct Timeseries object
        super().__init__(dataframe=df, src="UNRHighRate", data_unit=data_unit,
                         data_cols=["east", "north", "up"],
                         var_cols=["east_var", "north_var", "up_var"],
                         **kw_args)

    def get_arch(self) -> dict:
        """
        Returns a JSON-compatible dictionary with all the information necessary to recreate
        the Timeseries instance (provided the data file is available).

        Returns
        -------
        dict
            JSON-compatible dictionary sufficient to recreate the UNRTimeseries instance.

        See Also
        --------
        Timeseries.get_arch : For further information.
        """
        return {"type": "UNRHighRateTimeseries",
                "kw_args": {"path": self._path}}


class F5File():
    """
    Reader class that parses the contents of a single F5 file.

    Parameters
    ----------
    path
        Path to the F5 ``.pos`` file.
    """
    F5COLUMNS = ["yyyy", "mm", "dd", "HH:MM:SS",
                 "X (m)", "Y (m)", "Z (m)", "Lat. (deg.)", "Lon. (deg.)", "Height (m)"]
    """ Expected columns in the F5 file """
    F5DTYPES = defaultdict(lambda: "float", {c: str for c in F5COLUMNS[:4]})
    """ Data types in the F5 file """

    def __init__(self,
                 path: str,
                 ) -> None:
        # open file
        with open(path, mode="rt") as f:
            # continuously check whether the format is as expected
            assert f.readline() == "+SITE/INF\n"
            # read site info
            self.site = {elem[0]: elem[1]
                         for elem in [f.readline().split() for _ in range(4)]}
            """ Site information """
            assert f.readline() == "-SITE/INF\n"
            assert f.readline() == "\n"
            assert f.readline() == "+SOLVER/INF\n"
            # read solver info
            self.solver = {elem[0]: "  ".join(elem[1:])
                           for elem in [f.readline().split() for _ in range(7)]}
            """ Solver information """
            assert f.readline() == "-SOLVER/INF\n"
            assert f.readline() == "\n"
            assert f.readline() == "+DATA\n"
            # read data (omitting final "-DATA" row)
            data = pd.read_csv(f, sep=r"\s+", comment="*", engine="c",
                               names=self.F5COLUMNS, dtype=self.F5DTYPES).iloc[:-1, :]
            # parse date columns as datetime
            data["yyyy"] = data[["yyyy", "mm", "dd", "HH:MM:SS"]].agg(" ".join, axis=1)
            data.drop(self.F5COLUMNS[1:4], axis="columns", inplace=True)
            data.rename(columns={"yyyy": "time"}, inplace=True)
            data["time"] = pd.to_datetime(data["time"], format=r"%Y %m %d %H:%M:%S")
            data.set_index("time", inplace=True)
            # done
            self.data = data
            """ Data block """


class F5Timeseries(Timeseries):
    """Subclasses :class:`~Timeseries`.

    Timeseries subclass for GNSS measurements in GEONET's ``.pos`` F5 file format.
    There is no (co)variance information, and the absolute coordinates are converted
    into differences in ENU coordinates from the first data point.

    Parameters
    ----------
    paths
        List of filenames for the same station that are combined.
    data_unit
        Can be ``'mm'`` or ``'m'``.


    Additional keyword arguments will be passed onto :class:`~Timeseries`.

    Notes
    -----

    The column format is described in :attr:`~F5File.F5COLUMNS`
    The timezone assumed with each observation is UTC.

    """
    def __init__(self,
                 paths: list[str],
                 data_unit: Literal["mm", "m"] = "mm",
                 **kw_args
                 ) -> None:
        # initialize
        assert (isinstance(paths, list) and all([isinstance(p, str) for p in paths])
                and (len(paths) > 0)), \
            f"'paths' needs to be a list of string paths, got {paths}."
        self._paths = paths
        if data_unit == "m":
            factor = 1
        elif data_unit == "mm":
            factor = 1000
        else:
            raise ValueError(f"'data_unit' needs to be 'mm' or 'm', got {data_unit}.")
        # load individual files
        f5files = [F5File(p) for p in self._paths]
        # check that they're all for the same station
        longname = f5files[0].site["ID"]
        assert all([longname == f.site["ID"] for f in f5files]), \
            f"Found files for different stations: {set([f.site['ID'] for f in f5files])}"
        shortname = f5files[0].site["RINEX"]
        assert all([shortname == f.site["RINEX"] for f in f5files]), \
            f"Found files for different stations: {set([f.site['RINEX'] for f in f5files])}"
        # concatenate the data into a joint DataFrame
        f5data = pd.concat([f.data for f in f5files])
        f5data.sort_index(inplace=True)
        # get first lat/lon/alt coordinates
        ref_latlonalt = f5data[["Lat. (deg.)", "Lon. (deg.)", "Height (m)"]
                               ].iloc[0, :].values
        # get XYZ differences from first observations
        rel_xyz = (f5data[["X (m)", "Y (m)", "Z (m)"]].values
                   - f5data[["X (m)", "Y (m)", "Z (m)"]].iloc[0, :].values)
        # calculate relative displacement in ENU
        rot_mat = R_ecef2enu(ref_latlonalt[1], ref_latlonalt[0])
        rel_enu = rel_xyz @ rot_mat.T
        # convert away from m if desired
        if data_unit != "m":
            rel_enu *= factor
        # build timeseries dataframe
        data_cols = ["east", "north", "up"]
        df = pd.DataFrame(index=f5data.index, data=rel_enu,
                          columns=data_cols)
        # attach station ID and location
        self._id = longname
        self._rinex = shortname
        self._location = ref_latlonalt.tolist()
        # construct Timeseries object
        super().__init__(dataframe=df, src="F5", data_unit=data_unit,
                         data_cols=data_cols, **kw_args)

    @staticmethod
    def from_folder(folder: str,
                    show_warnings: bool = False,
                    **kw_args
                    ) -> tuple[list[str], list[tuple[float, float, float]], list[F5Timeseries]]:
        """
        Given the base folder of an F5 repository, find all F5 files and extract station names,
        locations, and timeseries.

        Parameters
        ----------
        folder
            Base folder that contains the "YYYY" year folders with annual ``.pos`` F5 files
            inside of them.
        show_warnings
            If ``True``, warn if there are data inconsistencies encountered while loading.
        **kw_args
            Extra arguments that are passed on to the :class:`~F5Timeseries` constructor.

        Returns
        -------
        names
            Station names.
        locations
            Station locations.
        timeseries
            Station :class:`~F5Timeseries` objects.
        """
        # define filename patterns
        globfile5 = "[12][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-9].[0-9][0-9].pos"
        globfile6 = "[12][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-9][0-9].[0-9][0-9].pos"
        # find all files matching the 5- and 6-character length formats
        p = Path(folder)
        names = sorted(list(set([f.parts[-1].split(".")[0] for f in p.glob(globfile5)] +
                                [f.parts[-1].split(".")[0] for f in p.glob(globfile6)])))
        locations = []
        timeseries = []
        # loop over all stations to load their files
        for n in tqdm(names, desc="Loading station files", unit="Station"):
            globstation = f"[0-9][0-9][0-9][0-9]/{n}.[0-9][0-9].pos"
            stationfiles = [str(sf) for sf in p.glob(globstation)]
            ts = F5Timeseries(stationfiles, **kw_args)
            locations.append(ts._location)
            timeseries.append(ts)
            if show_warnings and (ts._id != n):
                warn(f"Files loaded for station ID '{n}' were named '{ts._id}'.")
        # done
        return names, locations, timeseries
