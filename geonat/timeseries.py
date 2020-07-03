"""
This module contains the :class:`~Timeseries` base class and other
formats included by default in GeoNAT.
"""

import numpy as np
import pandas as pd
from warnings import warn
from copy import deepcopy


class Timeseries():
    """
    Object that expands the functionality of a :class:`~pandas.DataFrame` object
    for better integration into GeoNAT. Apart from the data itself, it contains
    information about the source and units of the data. It also performs input
    checks and uses property setters/getters to ensure consistency.

    Also enables the ability to perform math on timeseries directly.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The timeseries' data as a DataFrame. The index should be time, whereas
        data columns can be both data and their uncertainties.
    src : str
        Source description.
    data_unit : str
        Data unit.
    data_cols : list
        List of strings with the names of the columns of ``dataframe`` that
        contain the data.
    sigma_cols : list, optional
        List of strings with the names of the columns of ``dataframe`` that contain the
        data's uncertainty (as standard deviations). Must have the same length as ``data_cols``.
        If only certain data columns have uncertainties, set the respective entry
        to ``None``.
        Defaults to no data uncertainty columns.
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
        """
        Special function that returns a readable summary of the timeseries.
        Accessed, for example, by Python's ``print()`` built-in function.

        Returns
        -------
        info : str
            Timeseries summary.
        """
        info = f"Timeseries\n - Source: {self.src}\n - Units: {self.data_unit}\n" + \
               f" - Data: {[key for key in self.data_cols]}\n" + \
               f" - Uncertainties: {[key for key in self.sigma_cols]}"
        return info

    def __getitem__(self, columns):
        """
        Convenience special function that provides a shorthand notation
        to access the timeseries' columns.

        Parameters
        ----------
        columns : str, list
            String or list of strings of the columns to return.

        Returns
        -------
        pandas.Series, pandas.DataFrame
            Returns the requested data as a Series (if a single column) or
            DataFrame (if multiple columns).

        Example
        -------
        If ``ts`` is a :class:`~Timeseries` instance and ``columns`` a list
        of column names, the following two are equivalent::

            ts.df[columns]
            ts[ts_description]
        """
        if not isinstance(columns, str) and (not isinstance(columns, list) or not all([isinstance(col, str) for col in columns])):
            raise KeyError(f"Error when accessing data in timeseries: 'column' must be a string or list of strings, given was {columns}.")
        return self.df[columns]

    @property
    def src(self):
        """ Source information. """
        return self._src

    @src.setter
    def src(self, new_src):
        if not isinstance(new_src, str):
            raise TypeError(f"New 'src' attribute has to be a string, got {type(new_src)}.")
        self._src = new_src

    @property
    def data_unit(self):
        """ Data unit. """
        return self._data_unit

    @data_unit.setter
    def data_unit(self, new_data_unit):
        if not isinstance(new_data_unit, str):
            raise TypeError(f"New 'data_unit' attribute has to be a string, got {type(new_data_unit)}.")
        self._data_unit = new_data_unit

    @property
    def data_cols(self):
        """ List of the column names in :attr:`~df` that contain data. """
        return self._data_cols

    @data_cols.setter
    def data_cols(self, new_data_cols):
        assert isinstance(new_data_cols, list) and all([isinstance(dcol, str) for dcol in new_data_cols]), \
            f"New 'data_cols' attribute must be a list of strings of the same length as the current 'data_cols' ({len(self._data_cols)}), got {new_data_cols}."
        self._df.rename(columns={old_col: new_col for old_col, new_col in zip(self._data_cols, new_data_cols)}, errors='raise', inplace=True)
        self._data_cols = new_data_cols

    @property
    def sigma_cols(self):
        """ List of the column names in :attr:`~df` that contain data uncertainties. """
        return self._sigma_cols

    @sigma_cols.setter
    def sigma_cols(self, new_sigma_cols):
        assert isinstance(new_sigma_cols, list) and all([(scol is None) or isinstance(scol, str) for scol in new_sigma_cols]), \
            f"New 'sigma_cols' attribute must be a list of strings or Nones of the same length as the current 'sigma_cols' ({len(self._sigma_cols)}), got {new_sigma_cols}."
        self._df.rename(columns={old_col: new_col for old_col, new_col in zip(self._sigma_cols, new_sigma_cols) if (old_col is not None) and (new_col is not None)}, errors='raise')
        self._sigma_cols = new_sigma_cols

    @property
    def num_components(self):
        """ Number of data columns. """
        return len(self._data_cols)

    @property
    def df(self):
        """ The entire timeseries' DataFrame. """
        return self._df

    @property
    def num_observations(self):
        """ Number of observations (rows in :attr:`~df`). """
        return self._df.shape[0]

    @property
    def data(self):
        """ View of only the data columns in :attr:`~df`. """
        return self._df.loc[:, self._data_cols]

    @data.setter
    def data(self, new_data):
        self._df.loc[:, self._data_cols] = new_data

    @property
    def sigmas(self):
        """ View of only the data uncertainty columns in :attr:`~df`. """
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
        """ Timestamps of the timeseries (index of :attr:`~df`). """
        return self._df.index

    def copy(self, only_data=False, src=None):
        """
        Return a deep copy of the timeseries instance.

        Parameters
        ----------
        only_data : bool, optional
            If ``True``, only copy the data columns and ignore any uncertainty information.
            Defaults to ``False``.
        src : str, optional
            Set a new source information attribute for the copy.
            Uses the current one by default.

        Returns
        -------
        Timeseries
            The copy of the timeseries instance.
        """
        new_name = deepcopy(self._src) if src is None else src
        if not only_data:
            return Timeseries(self._df.copy(), new_name, deepcopy(self._data_unit), deepcopy(self._data_cols), deepcopy(self._sigma_cols))
        else:
            return Timeseries(self._df[self._data_cols].copy(), new_name, deepcopy(self._data_unit), deepcopy(self._data_cols), None)

    def mask_out(self, dcol):
        """
        Mask out an entire data column (and if present, its uncertainty column) by setting
        the entire column to ``NaN``. Converts it to a sparse representation to save memory.

        Parameters
        ----------
        dcol : str
            Name of the data column to mask out.
        """
        icol = self._data_cols.index(dcol)
        scol = self._sigma_cols[icol]
        self._df[dcol] = np.NaN
        self._df[dcol] = self._df[dcol].astype(pd.SparseDtype(dtype=float))
        if scol is not None:
            self._df[scol] = np.NaN
            self._df[scol] = self._df[scol].astype(pd.SparseDtype(dtype=float))

    @staticmethod
    def prepare_math(left, right, operation):
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

        Warning
        -------
        This method is called under-the-hood whenever a mathematical operation is
        performed, and should not need to be used by normal users.

        Parameters
        ----------
        left : Timeseries or numpy.ndarray
            Left term of the operation.
        right : Timeseries or numpy.ndarray
            Right term of the operation.
        operation : {'+', '-', '*', '/'}
            Operation to perform.

        Returns
        -------
        left_data : numpy.ndarray
            View of the 2D left data array of the operation
            with shape (``len(out_time)``, :attr:`~num_components`).
        right_data : numpy.ndarray
            View of the 2D right data array of the operation
            with shape (``len(out_time)``, :attr:`~num_components`).
        out_src : str
            Combines the sources of each object to a new string.
        out_data_unit : str
            Combines the data units of each object into a new unit.
        out_data_cols : list
            List of strings containing the new data column names.
        out_time : pandas.Index
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
            raise NotImplementedError(f"Timeseries math problem: unknown operation '{operation}'.")
        # check for compatible types
        for operand, position in zip([left, right], ["Left", "Right"]):
            if (not isinstance(operand, Timeseries)) and (not isinstance(operand, np.ndarray)):
                raise TypeError(f"{position} operand has to be either a '{Timeseries}' or '{np.ndarray}' "
                                f"object, got {type(operand)}")
        if all([isinstance(operand, np.ndarray) for operand in [left, right]]):
            raise TypeError(f"At least one of the operands has to be a '{Timeseries}' object, "
                            f"got two '{np.ndarray}' objects.")
        # check for same dimensions
        len_left_data_cols = left.num_components if isinstance(left, Timeseries) else left.shape[1]
        len_right_data_cols = right.num_components if isinstance(right, Timeseries) else right.shape[1]
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
                    raise ValueError(f"Timeseries math problem: conflicting data units '{left.data_unit}' and '{right.data_unit}'.")
                else:
                    out_data_unit = left.data_unit
                    out_src = f"{left.src}{operation}{right.src}"
                    out_data_cols = [f"{lcol}{operation}{rcol}" for lcol, rcol in zip(left.data_cols, right.data_cols)]
            elif operation in ["*", "/"]:
                out_data_unit = f"({left.data_unit}){operation}({right.data_unit})"
                out_src = f"({left.src}){operation}({right.src})"
                out_data_cols = [f"({lcol}){operation}({rcol})" for lcol, rcol in zip(left.data_cols, right.data_cols)]
            left_data = left.data.loc[out_time, :].values
            right_data = right.data.loc[out_time, :].values
        # return data unit and column names
        return left_data, right_data, out_src, out_data_unit, out_data_cols, out_time

    def __add__(self, other):
        """
        Special function that allows two timeseries instances (or a timeseries and an equivalently
        shaped NumPy array) to be added together element-wise.

        Parameters
        ----------
        other : Timeseries
            Timeseries to add to instance.

        Returns
        -------
        Timeseries
            New timeseries object containing the sum of the two timeseries.

        See Also
        --------
        prepare_math
            Prepares the two instances for the mathematical operation. Refer to it for more details
            about how the two objects are cast together.

        Example
        -------
        Add two :class:`~Timeseries` ``ts1`` and ``ts2`` and save the result as ``ts3``::

            ts3 = ts1 + ts2
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = Timeseries.prepare_math(self, other, '+')
        return Timeseries(pd.DataFrame(left_data + right_data, index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __radd__(self, other):
        """
        Reflected operation of :meth:`~__add__` (necessary if first operand is a NumPy array).
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = Timeseries.prepare_math(other, self, '+')
        return Timeseries(pd.DataFrame(left_data + right_data, index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __sub__(self, other):
        """
        Special function that allows a timeseries instance (or a timeseries and an equivalently
        shaped NumPy array) to be subtracted from another element-wise.

        Parameters
        ----------
        other : Timeseries
            Timeseries to subtract from instance.

        Returns
        -------
        Timeseries
            New timeseries object containing the difference of the two timeseries.

        See Also
        --------
        prepare_math
            Prepares the two instances for the mathematical operation. Refer to it for more details
            about how the two objects are cast together.

        Example
        -------
        Subtract two :class:`~Timeseries` ``ts1`` and ``ts2`` and save the result as ``ts3``::

            ts3 = ts1 - ts2
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = Timeseries.prepare_math(self, other, '-')
        return Timeseries(pd.DataFrame(left_data - right_data, index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __rsub__(self, other):
        """
        Reflected operation of :meth:`~__sub__` (necessary if first operand is a NumPy array).
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = Timeseries.prepare_math(other, self, '-')
        return Timeseries(pd.DataFrame(left_data - right_data, index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __mul__(self, other):
        """
        Special function that allows two timeseries instances (or a timeseries and an equivalently
        shaped NumPy array) to be multiplied together element-wise.

        Parameters
        ----------
        other : Timeseries
            Timeseries to multiply to instance.

        Returns
        -------
        Timeseries
            New timeseries object containing the product of the two timeseries.

        See Also
        --------
        prepare_math
            Prepares the two instances for the mathematical operation. Refer to it for more details
            about how the two objects are cast together.

        Example
        -------
        Multiply two :class:`~Timeseries` ``ts1`` and ``ts2`` and save the result as ``ts3``::

            ts3 = ts1 * ts2
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = Timeseries.prepare_math(self, other, '*')
        return Timeseries(pd.DataFrame(left_data * right_data, index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __rmul__(self, other):
        """
        Reflected operation of :meth:`~__mul__` (necessary if first operand is a NumPy array).
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = Timeseries.prepare_math(other, self, '*')
        return Timeseries(pd.DataFrame(left_data * right_data, index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __truediv__(self, other):
        """
        Special function that allows a timeseries instances (or a timeseries and an equivalently
        shaped NumPy array) to be divided by another element-wise.

        Parameters
        ----------
        other : Timeseries
            Timeseries to divide instance by.

        Returns
        -------
        Timeseries
            New timeseries object containing the quotient of the two timeseries.

        See Also
        --------
        prepare_math
            Prepares the two instances for the mathematical operation. Refer to it for more details
            about how the two objects are cast together.

        Example
        -------
        Divide two :class:`~Timeseries` ``ts1`` and ``ts2`` and save the result as ``ts3``::

            ts3 = ts1 / ts2
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = Timeseries.prepare_math(self, other, '/')
        return Timeseries(pd.DataFrame(left_data / right_data, index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def __rtruediv__(self, other):
        """
        Reflected operation of :meth:`~__truediv__` (necessary if first operand is a NumPy array).
        """
        left_data, right_data, out_src, out_unit, out_data_cols, out_time = Timeseries.prepare_math(other, self, '/')
        return Timeseries(pd.DataFrame(left_data / right_data, index=out_time, columns=out_data_cols),
                          out_src, out_unit, out_data_cols)

    def get_arch(self):
        """
        Build a dictionary describing the architecture of this timeseries,
        to be used when creating a network JSON configuration file.

        Without subclassing :class:`~Timeseries`, this function will return an
        empty dictionary by default, since it is unknown how to recreate
        a general Timeseries object from just a JSON-compatible dictionary.

        See Also
        --------
        geonat.network.Network.to_json : Export the Network configuration as a JSON file.
        geonat.timeseries.Timeseries.get_arch
            Get the architecture dictionary of a :class:`~geonat.timeseries.Timeseries` instance.
        geonat.model.Model.get_arch
            Get the architecture dictionary of a :class:`~geonat.model.Model` instance.
        """
        return {}

    @classmethod
    def from_fit(cls, data_unit, data_cols, fit):
        """
        Import a fit dictionary and create a Timeseries instance.

        Parameters
        ----------
        data_unit : str
            Data unit.
        data_cols : list
            List of strings containing the data column names.
            Uncertainty column names are generated by adding a '_sigma'.
        fit : dict
            Dictionary with the keys ``'time'``, ``'fit'`` and ``'sigma'``.

        Returns
        -------
        Timeseries
            Timeseries instance created from ``fit``.

        See Also
        --------
        geonat.model.Model.evaluate
            Evaluating a model produces the fit dictionary.
        """
        df_data = {dcol: fit["fit"][:, icol] for icol, dcol in enumerate(data_cols)}
        sigma_cols = None
        if fit["sigma"] is not None:
            sigma_cols = [dcol + "_sigma" for dcol in data_cols]
            df_data.update({scol: fit["sigma"][:, icol] for icol, scol in enumerate(sigma_cols)})
        df = pd.DataFrame(data=df_data, index=fit["time"])
        return cls(df, "fitted", data_unit, data_cols, sigma_cols)

    @classmethod
    def from_array(cls, timevector, data, src, data_unit, data_cols, sigma=None, sigma_cols=None):
        r"""
        Constructor method to create a :class:`~Timeseries` instance from a NumPy
        :class:`~numpy.ndarray`.

        Parameters
        ----------
        timevector : pandas.Series, pandas.DatetimeIndex
            :class:`~pandas.Series` of :class:`~pandas.Timestamp` or alternatively a
            :class:`~pandas.DatetimeIndex` containing the timestamps of each observation.
        data : numpy.ndarray
            2D NumPy array of shape :math:`(\text{n_observations},\text{n_components})`
            containing the data.
        src : str
            Source description.
        data_unit : str
            Data unit.
        data_cols : list
            List of strings with the names of the columns of ``data``.
        sigma : numpy.ndarray, optional
            2D NumPy array of shape :math:`(\text{n_observations},\text{n_components})`
            containing the data uncertainty (standard deviations).
            Defaults to no data uncertainty.
        sigma_cols : list, optional
            List of strings with the names of the columns of ``data`` that contain the
            data's uncertainty (as standard deviations). Must have the same length as ``data_cols``.
            If only certain data columns have uncertainties, set the respective entry
            to ``None``.
            If ``sigma`` is given but ``sigma_cols`` is not, it defaults to appending
            ``'_sigma'`` to ``data_cols``.

        See Also
        --------
        pandas.date_range : Quick function to generate a timevector.
        """
        assert len(timevector) == data.shape[0], \
            f"length of 'timevector' has to match the number of rows in 'data', got {len(timevector)} and {data.shape}."
        df_data = {dcol: data[:, icol] for icol, dcol in enumerate(data_cols)}
        if sigma is None:
            sigma_cols = None
        else:
            assert data.shape == sigma.shape, \
                f"'data' and 'sigma' need to have the same shape, got {data.shape} and {sigma.shape}."
            if sigma_cols is None:
                sigma_cols = [dcol + "_sigma" for dcol in data_cols]
            else:
                assert len(sigma_cols) == sigma.shape[1]
            df_data.update({scol: sigma[:, icol] for icol, scol in enumerate(sigma_cols) if scol is not None})
        df = pd.DataFrame(data=df_data, index=timevector)
        return cls(df, src, data_unit, data_cols, sigma_cols)


class GipsyTimeseries(Timeseries):
    """
    Subclasses :class:`~Timeseries`.

    Timeseries subclass for GNSS measurements in JPL's Gipsy ``.tseries`` file format.

    Parameters
    ----------
    path : str
        Path to the timeseries file.
    show_warnings : bool, optional
        If ``True``, warn if there are data inconsistencies encountered while loading.
        Defaults to ``True``.
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
                "kw_args": {"path": self._path}}
