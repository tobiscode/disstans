import numpy as np
import pandas as pd
from warnings import warn
from copy import deepcopy


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
    def __init__(self, path, show_warnings=True):
        self._path = path
        time = pd.to_datetime(pd.read_csv(self._path, delim_whitespace=True, header=None, usecols=[11, 12, 13, 14, 15, 16],
                                          names=['year', 'month', 'day', 'hour', 'minute', 'second'])).to_frame(name='time')
        data = pd.read_csv(self._path, delim_whitespace=True, header=None, usecols=[1, 2, 3, 4, 5, 6], names=['east', 'north', 'up', 'east_sigma', 'north_sigma', 'up_sigma'])
        df = time.join(data)
        num_duplicates = int(df.duplicated(subset='time').sum())
        if (num_duplicates > 0) and show_warnings:
            warn(f"Timeseries file {path} contains data for {num_duplicates} duplicate dates. Keeping first occurrences.")
        df = df.drop_duplicates(subset='time').set_index('time')
        super().__init__(dataframe=df, src='.tseries', data_unit='m',
                         data_cols=['east', 'north', 'up'], sigma_cols=['east_sigma', 'north_sigma', 'up_sigma'])

    def get_arch(self):
        return {"type": "GipsyTimeseries",
                "kw_args": {"path": self._path}}
